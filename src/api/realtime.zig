//! The voice endpoints (docs/dev/VOICE.md): `GET /v1/realtime`, upgraded to a WebSocket that speaks the OpenAI Realtime
//! API's events (`voice/session.zig`), and `GET /voice`, the built-in browser client. Both answer loopback peers
//! only unless `config.voiceRemote` (`voice.remote`, by default on when the daemon listens beyond loopback with
//! pairing required); with `pairing.required` the socket also needs a paired device's token with the
//! inference scope (or the legacy voice: `pair.zig`), and the session's tools need that device's scopes (`voice_local.zig`).

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const routes = @import("routes.zig");
const config = @import("../config.zig");
const ws = @import("../voice/ws.zig");
const Session = @import("../voice/session.zig").Session;
const respond = @import("../voice/respond.zig");
const Local = @import("voice_local.zig").Local;
const pair = @import("pair.zig");

const App = routes.App;
const page_html = @embedFile("../voice/voice.html");

/// The open sessions' sockets, so shutdown can end their reads, and the sessions themselves, so they hear when a voice
/// model is unloaded to make room (`localrouter.voice.paused`, then `localrouter.voice.resumed`).
pub const Registry = struct {
    mutex: Io.Mutex = .init,
    list: std.ArrayList(std.posix.fd_t) = .empty,
    sessions: std.ArrayList(*Session) = .empty,
    closed: bool = false,
    io: ?Io = null, // set by `notifier`
    gpa: ?std.mem.Allocator = null,
    remote_calls: std.atomic.Value(u32) = .init(0), // voice calls open on the front of a multi-node topology (`api/node.zig`)
    remote_until_ns: std.atomic.Value(i64) = .init(0), // until then: the front tells it again with each health check

    fn bind(r: *Registry, io: Io, gpa: std.mem.Allocator, s: *Session) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        r.sessions.append(gpa, s) catch {};
    }

    fn unbind(r: *Registry, io: Io, s: *Session) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        for (r.sessions.items, 0..) |x, i| if (x == s) {
            _ = r.sessions.swapRemove(i);
            return;
        };
    }

    /// The scheduler's hook: every open session hears it before a voice model goes and when they are all back.
    pub fn notifier(r: *Registry, io: Io, gpa: std.mem.Allocator) @import("../sched/scheduler.zig").Notify {
        r.io = io;
        r.gpa = gpa;
        return .{ .ctx = r, .paused = paused, .resumed = resumed, .sessions = openSessions };
    }

    /// The scheduler's count of open calls: this daemon's sessions, and those the front last said it has open (a node
    /// of a multi-node topology guards the voice loop's models for the front's calls as for its own).
    fn openSessions(ctx: *anyopaque) usize {
        const r: *Registry = @ptrCast(@alignCast(ctx));
        const io = r.io orelse return 0;
        const now: i64 = @intCast(Io.Clock.awake.now(io).toNanoseconds());
        const remote: usize = if (now < r.remote_until_ns.load(.acquire)) r.remote_calls.load(.acquire) else 0;
        return r.count(io) + remote;
    }

    /// The front has `n` voice calls open; it says so again within `valid_ms`, or they are taken as ended.
    pub fn remoteCalls(r: *Registry, io: Io, n: u32, valid_ms: i64) void {
        const now: i64 = @intCast(Io.Clock.awake.now(io).toNanoseconds());
        r.remote_calls.store(n, .release);
        r.remote_until_ns.store(now + valid_ms * std.time.ns_per_ms, .release);
    }

    fn paused(ctx: *anyopaque, model: []const u8, for_model: []const u8) void {
        const r: *Registry = @ptrCast(@alignCast(ctx));
        r.broadcast("localrouter.voice.paused", .{ .model = model, .for_model = for_model, .message = "voice is paused while a model makes room; it comes back by itself when the job is done" });
    }

    /// Each session then checks its voice again (the speech model is a new instance) and says when it is ready.
    fn resumed(ctx: *anyopaque) void {
        const r: *Registry = @ptrCast(@alignCast(ctx));
        r.broadcast("localrouter.voice.resumed", .{ .message = "voice is back" });
        const io = r.io orelse return;
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        for (r.sessions.items) |s| {
            s.mutex.lockUncancelable(io);
            defer s.mutex.unlock(io);
            respond.getReady(s);
        }
    }

    fn broadcast(r: *Registry, comptime typ: []const u8, fields: anytype) void {
        const io = r.io orelse return;
        var arena: std.heap.ArenaAllocator = .init(r.gpa orelse return);
        defer arena.deinit();
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        for (r.sessions.items) |s| {
            s.mutex.lockUncancelable(io);
            defer s.mutex.unlock(io);
            if (!s.closing) s.emit.send(arena.allocator(), typ, fields);
        }
    }

    fn add(r: *Registry, io: Io, gpa: std.mem.Allocator, fd: std.posix.fd_t) bool {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        if (r.closed) return false;
        r.list.append(gpa, fd) catch return false;
        return true;
    }

    fn remove(r: *Registry, io: Io, fd: std.posix.fd_t) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        for (r.list.items, 0..) |x, i| if (x == fd) {
            _ = r.list.swapRemove(i);
            return;
        };
    }

    /// The speech model and voice each open session speaks in now (copies in `a`), oldest session first.
    pub fn speechWants(r: *Registry, io: Io, a: std.mem.Allocator) []const @import("../voice/backend.zig").Say {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        var out: std.ArrayList(@import("../voice/backend.zig").Say) = .empty;
        for (r.sessions.items) |s| {
            s.mutex.lockUncancelable(io);
            defer s.mutex.unlock(io);
            if (s.closing) continue;
            out.append(a, .{ .model = a.dupe(u8, s.settings.speech_model) catch continue, .voice = a.dupe(u8, s.settings.voice) catch continue }) catch break;
        }
        return out.items;
    }

    /// How many sessions are open (an operator restarts only when none is).
    pub fn count(r: *Registry, io: Io) usize {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        return r.list.items.len;
    }

    /// Shuts every session's socket down (they then close themselves); new sessions are refused.
    pub fn closeAll(r: *Registry, io: Io) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        r.closed = true;
        for (r.list.items) |fd| _ = std.posix.system.shutdown(fd, std.posix.SHUT.RDWR);
    }

    pub fn deinit(r: *Registry, gpa: std.mem.Allocator) void {
        r.list.deinit(gpa);
        r.sessions.deinit(gpa);
    }
};

/// True for 127.0.0.0/8, ::1 and IPv4-mapped loopback.
pub fn isLoopback(addr: Io.net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |x| x.bytes[0] == 127,
        .ip6 => |x| (std.mem.allEqual(u8, x.bytes[0..15], 0) and x.bytes[15] == 1) or
            (std.mem.allEqual(u8, x.bytes[0..10], 0) and x.bytes[10] == 0xff and x.bytes[11] == 0xff and x.bytes[12] == 127),
    };
}

fn allowed(app: *App, conn: *const h.Conn) bool {
    return config.voiceRemote(app.cfg.*) or isLoopback(conn.stream.socket.address);
}

const refused = "the voice endpoints answer this machine only; the operator can open them with voice.remote in the config (on by default when the daemon listens beyond loopback with pairing required)";

/// GET /voice: the page.
pub fn page(app: *App, conn: *const h.Conn, req: *h.Request, a: std.mem.Allocator) !void {
    if (!allowed(app, conn)) return h.fail(req, a, 403, "invalid_request_error", refused);
    try req.respond(page_html, .{ .extra_headers = &.{
        .{ .name = "content-type", .value = "text/html; charset=utf-8" },
        .{ .name = "cache-control", .value = "no-store" },
        .{ .name = "content-security-policy", .value = "default-src 'self'; script-src 'self' 'unsafe-inline' blob:; style-src 'unsafe-inline'; worker-src blob:; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self' ws: wss:" },
    } });
}

/// The value of `name` in the request's query string, or null.
pub fn query(target: []const u8, name: []const u8) ?[]const u8 {
    const q = std.mem.indexOfScalar(u8, target, '?') orelse return null;
    var it = std.mem.splitScalar(u8, target[q + 1 ..], '&');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse kv.len;
        if (std.mem.eql(u8, kv[0..eq], name)) return if (eq < kv.len) kv[eq + 1 ..] else "";
    }
    return null;
}

/// True when the comma-separated `offered` list names `want`.
pub fn offers(offered: []const u8, want: []const u8) bool {
    var it = std.mem.splitScalar(u8, offered, ',');
    while (it.next()) |p| if (std.mem.eql(u8, std.mem.trim(u8, p, " \t"), want)) return true;
    return false;
}

/// GET /v1/realtime: the WebSocket upgrade, then one session until either side closes.
pub fn serve(app: *App, conn: *const h.Conn, req: *h.Request, a: std.mem.Allocator, caller: pair.Caller) !void {
    if (!allowed(app, conn)) return h.fail(req, a, 403, "invalid_request_error", refused);
    const key = switch (req.upgradeRequested()) {
        .websocket => |k| k orelse return h.fail(req, a, 400, "invalid_request_error", "the WebSocket upgrade has no Sec-WebSocket-Key"),
        else => return h.fail(req, a, 426, "invalid_request_error", "this is a WebSocket endpoint (the OpenAI Realtime API): connect with ws://"),
    };
    // Everything from the head is copied: the head's buffer is reused once the socket is read.
    const model = try a.dupe(u8, query(req.head.target, "model") orelse "");
    const host = try a.dupe(u8, h.host(req));
    const scheme = h.scheme(req); // static
    const token = try a.dupe(u8, @import("../computer/gate.zig").token(req)); // a paired device's, for the computer tools
    var protocol: []const u8 = "";
    var hit = req.iterateHeaders();
    while (hit.next()) |hd| if (std.ascii.eqlIgnoreCase(hd.name, "sec-websocket-protocol")) {
        protocol = try a.dupe(u8, hd.value);
    };
    const sub: []const std.http.Header = if (offers(protocol, "realtime")) &.{.{ .name = "sec-websocket-protocol", .value = "realtime" }} else &.{};
    var up = req.respondWebSocket(.{ .key = try a.dupe(u8, key), .extra_headers = sub }) catch return;
    up.flush() catch return;
    req.server.reader.state = .closing; // the connection ends with the session
    var c: ws.Conn = .{ .io = app.io, .gpa = app.gpa, .in = up.input, .out = up.output };
    defer c.deinit();
    const fd = conn.stream.socket.handle;
    if (!app.voice.add(app.io, app.gpa, fd)) return c.close(1001, "the server is shutting down");
    defer app.scheduler.callClosed(); // after the remove below: a `call` tool's idle TTL counts once no call is open
    defer app.voice.remove(app.io, fd);
    app.scheduler.callOpened(); // the `call` tools load now; the session says so (`localrouter.session.loading`)
    var url_buf: [96]u8 = undefined;
    var local = Local.init(app, host, &url_buf);
    local.as(token, caller.owner()); // `routes.handle` admitted it (pairing.required: a device with inference or voice)
    local.scheme = scheme;
    var sink: Sink = .{ .c = &c };
    const s = Session.create(app.gpa, app.io, local.asBackend(), .{ .ctx = &sink, .send = Sink.send }, app.cfg.voice, if (model.len > 0) model else Local.defaultChatModel(app)) catch
        return c.close(1011, "could not start the session");
    defer s.destroy();
    app.voice.bind(app.io, app.gpa, s);
    defer app.voice.unbind(app.io, s);
    if (app.scheduler.voicePaused()) { // it opened while a voice model is away
        var tmp: std.heap.ArenaAllocator = .init(app.gpa);
        defer tmp.deinit();
        s.mutex.lockUncancelable(app.io);
        s.emit.send(tmp.allocator(), "localrouter.voice.paused", .{ .model = "", .for_model = "", .message = "voice is paused while a model makes room; it comes back by itself when the job is done" });
        s.mutex.unlock(app.io);
    }
    while (true) {
        const m = c.read() catch |err| {
            if (err == error.TooLarge) std.log.warn("voice session: a client message over 4 MiB, closed", .{});
            break;
        };
        if (m.opcode == .text) s.handle(m.data) else s.refuseBinary();
    }
    c.close(1000, "");
}

/// Server events to the socket; a client that is gone just stops receiving.
const Sink = struct {
    c: *ws.Conn,

    fn send(ctx: *anyopaque, text: []const u8) void {
        const k: *Sink = @ptrCast(@alignCast(ctx));
        k.c.sendText(text) catch {};
    }
};

test "loopback peers, query values and offered subprotocols" {
    try std.testing.expect(isLoopback(.{ .ip4 = .loopback(80) }));
    try std.testing.expect(isLoopback(try Io.net.IpAddress.parse("127.8.0.1", 1)));
    try std.testing.expect(isLoopback(try Io.net.IpAddress.parse("::1", 1)));
    try std.testing.expect(isLoopback(try Io.net.IpAddress.parse("::ffff:127.0.0.1", 1)));
    try std.testing.expect(!isLoopback(try Io.net.IpAddress.parse("100.64.1.2", 1)));
    try std.testing.expect(!isLoopback(try Io.net.IpAddress.parse("192.168.1.116", 1)));
    try std.testing.expectEqualStrings("qwen", query("/v1/realtime?model=qwen&x=1", "model").?);
    try std.testing.expect(query("/v1/realtime", "model") == null);
    try std.testing.expect(offers("realtime, openai-insecure-api-key.x, openai-beta.realtime-v1", "realtime"));
    try std.testing.expect(!offers("openai-beta.realtime-v1", "realtime"));
}
