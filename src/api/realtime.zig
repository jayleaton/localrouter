//! The voice endpoints (docs/dev/VOICE.md): `GET /v1/realtime`, upgraded to a WebSocket that speaks the OpenAI Realtime
//! API's events (`voice/session.zig`), and `GET /voice`, the built-in browser client. Both answer loopback peers
//! only unless `voice.remote` is set.

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const routes = @import("routes.zig");
const ws = @import("../voice/ws.zig");
const Session = @import("../voice/session.zig").Session;
const Local = @import("voice_local.zig").Local;

const App = routes.App;
const page_html = @embedFile("../voice/voice.html");

/// The open sessions' sockets, so shutdown can end their reads.
pub const Registry = struct {
    mutex: Io.Mutex = .init,
    list: std.ArrayList(std.posix.fd_t) = .empty,
    closed: bool = false,

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

    /// Shuts every session's socket down (they then close themselves); new sessions are refused.
    pub fn closeAll(r: *Registry, io: Io) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        r.closed = true;
        for (r.list.items) |fd| _ = std.posix.system.shutdown(fd, std.posix.SHUT.RDWR);
    }

    pub fn deinit(r: *Registry, gpa: std.mem.Allocator) void {
        r.list.deinit(gpa);
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
    return app.cfg.voice.remote or isLoopback(conn.stream.socket.address);
}

const refused = "the voice endpoints answer this machine only; the operator can open them with voice.remote in the config";

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
pub fn serve(app: *App, conn: *const h.Conn, req: *h.Request, a: std.mem.Allocator) !void {
    if (!allowed(app, conn)) return h.fail(req, a, 403, "invalid_request_error", refused);
    const key = switch (req.upgradeRequested()) {
        .websocket => |k| k orelse return h.fail(req, a, 400, "invalid_request_error", "the WebSocket upgrade has no Sec-WebSocket-Key"),
        else => return h.fail(req, a, 426, "invalid_request_error", "this is a WebSocket endpoint (the OpenAI Realtime API): connect with ws://"),
    };
    // Everything from the head is copied: the head's buffer is reused once the socket is read.
    const model = try a.dupe(u8, query(req.head.target, "model") orelse "");
    const host = try a.dupe(u8, h.host(req));
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
    defer app.voice.remove(app.io, fd);
    var url_buf: [96]u8 = undefined;
    var local = Local.init(app, host, &url_buf);
    var sink: Sink = .{ .c = &c };
    const s = Session.create(app.gpa, app.io, local.asBackend(), .{ .ctx = &sink, .send = Sink.send }, app.cfg.voice, model) catch
        return c.close(1011, "could not start the session");
    defer s.destroy();
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
