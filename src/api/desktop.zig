//! Computer control over HTTP (docs/dev/COMPUTER.md): `GET /v1/desktop`, the WebSocket a paired desktop helper dials
//! out to, and `/v1/computer/...` for the web UI: status, approvals (list, approve, deny), stop and the audit log. Every
//! route needs a paired device's token with the route's scope (`computer/gate.zig`); the decisions come from the chat
//! model through `decide.Qwen` until the open Jev service replaces it.

const std = @import("std");
const Io = std.Io;
const Value = std.json.Value;
const h = @import("http.zig");
const routes = @import("routes.zig");
const chat = @import("chat.zig");
const ws = @import("../voice/ws.zig");
const hub_mod = @import("../computer/hub.zig");
const audit = @import("../computer/audit.zig");
const decide = @import("../computer/decide.zig");
const gate = @import("../computer/gate.zig");
const pair = @import("pair.zig");

const App = routes.App;
const Hub = hub_mod.Hub;

/// The daemon's computer-control state; lives as long as the server.
pub const Service = struct {
    log: audit.Log,
    hub: Hub,
    qwen: decide.Qwen,
    gate: gate.Gate = .{},
    app: *App,
    mutex: Io.Mutex = .init,
    fds: std.ArrayList(std.posix.fd_t) = .empty, // helper sockets, shut at stop
    closed: bool = false,

    /// Sets up in place (the hub points at the log, the provider at the service).
    pub fn init(s: *Service, app: *App, data_dir: []const u8, a: std.mem.Allocator) !void {
        const cc = app.cfg.computer;
        s.* = .{ .log = .{ .io = app.io, .path = try std.fs.path.join(a, &.{ data_dir, "computer", "audit.jsonl" }) }, .hub = undefined, .qwen = .{ .ctx = undefined, .complete = complete, .extra = app.cfg.voice.chat_extra }, .app = app };
        s.hub = Hub.init(app.gpa, app.io, &s.log);
        s.hub.limits = .{ .max_steps = @min(cc.max_steps, 15), .timeout_ms = @as(i64, @min(cc.timeout_s, 90)) * 1000, .approval_timeout_ms = @as(i64, cc.approval_timeout_s) * 1000 };
        s.hub.approvals.ttl_ms = s.hub.limits.approval_timeout_ms;
        s.qwen.ctx = s;
        s.hub.provider = s.qwen.provider();
        s.gate = gate.Paired(App, pairedDevice).gate(app); // always: computer control never runs without a paired device
    }

    pub fn deinit(s: *Service) void {
        s.hub.deinit();
        s.fds.deinit(s.app.gpa);
    }

    /// Shuts every helper socket (their loops then end); new ones are refused.
    pub fn closeAll(s: *Service) void {
        s.mutex.lockUncancelable(s.app.io);
        defer s.mutex.unlock(s.app.io);
        s.closed = true;
        for (s.fds.items) |fd| _ = std.posix.system.shutdown(fd, std.posix.SHUT.RDWR);
    }

    fn addFd(s: *Service, fd: std.posix.fd_t) bool {
        s.mutex.lockUncancelable(s.app.io);
        defer s.mutex.unlock(s.app.io);
        if (s.closed) return false;
        s.fds.append(s.app.gpa, fd) catch return false;
        return true;
    }

    fn removeFd(s: *Service, fd: std.posix.fd_t) void {
        s.mutex.lockUncancelable(s.app.io);
        defer s.mutex.unlock(s.app.io);
        for (s.fds.items, 0..) |x, i| if (x == fd) {
            _ = s.fds.swapRemove(i);
            return;
        };
    }
};

/// The paired device behind `tok` (`api/pair.zig`), its id and name copied into `buf`.
fn pairedDevice(app: *App, tok: []const u8, buf: []u8) ?gate.Found {
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const d = pair.deviceOf(app, arena.allocator(), tok) orelse return null;
    const id_len = @min(d.id.len, 32);
    const name_len = @min(d.name.len, buf.len - id_len);
    @memcpy(buf[0..id_len], d.id[0..id_len]);
    @memcpy(buf[id_len..][0..name_len], d.name[0..name_len]);
    return .{ .device = .{ .id = buf[0..id_len], .name = buf[id_len..][0..name_len] }, .desktop_helper = d.allows("desktop_helper"), .computer = d.allows("computer"), .approve = d.allows("approve") };
}

/// The Qwen provider's chat call: one completion on the configured decision model (or the chat default).
fn complete(ctx: *anyopaque, a: std.mem.Allocator, body: Value, why: *decide.Why) ?[]const u8 {
    const s: *Service = @ptrCast(@alignCast(ctx));
    const app = s.app;
    var b = body;
    if (app.cfg.computer.decision_model.len > 0 and b == .object) b.object.put(a, "model", .{ .string = app.cfg.computer.decision_model }) catch {};
    var refusal: routes.Refusal = .{};
    const j = chat.startChat(app, a, b, "", &refusal) orelse { // the daemon's own job: no device sees it
        why.set("the decision model refused: {s}", .{refusal.message});
        return null;
    };
    j.done.wait(app.io) catch return null;
    if (j.status != .completed) {
        why.set("the decision model failed: {s}", .{j.error_message});
        return null;
    }
    const text = chat.readCompletion(app, a, j, &refusal) orelse return null;
    const C = struct { choices: []const struct { message: struct { content: ?[]const u8 = null } } = &.{} };
    const c = std.json.parseFromSliceLeaky(C, a, text, .{ .ignore_unknown_fields = true }) catch {
        why.set("the decision model's reply is not a chat completion", .{});
        return null;
    };
    if (c.choices.len == 0) return null;
    return c.choices[0].message.content;
}

fn service(app: *App) ?*Service {
    return app.computer;
}

/// The device behind the request's token if it has `scope`, else the refusal already sent (null).
fn admit(app: *App, req: *h.Request, a: std.mem.Allocator, scope: gate.Scope) !?gate.Device {
    const s = service(app) orelse {
        try h.fail(req, a, 404, "invalid_request_error", "computer control is switched off in this daemon's config (computer.enabled)");
        return null;
    };
    const tok = try a.dupe(u8, gate.token(req));
    switch (s.gate.check(tok, scope)) {
        .ok => |d| return .{ .id = try a.dupe(u8, d.id), .name = try a.dupe(u8, d.name) },
        .off => |m| try h.fail(req, a, 403, "invalid_request_error", m),
        .denied => |m| try h.fail(req, a, 401, "invalid_request_error", m),
    }
    return null;
}

// ---------------------------------------------------------------- the helper's socket

/// GET /v1/desktop: a paired helper's socket, until either side closes.
pub fn serve(app: *App, conn: *const h.Conn, req: *h.Request, a: std.mem.Allocator) !void {
    const s = service(app) orelse return h.fail(req, a, 404, "invalid_request_error", "computer control is switched off in this daemon's config (computer.enabled)");
    const device = (try admit(app, req, a, .desktop_helper)) orelse return;
    const tok = try a.dupe(u8, gate.token(req));
    const key = switch (req.upgradeRequested()) {
        .websocket => |k| k orelse return h.fail(req, a, 400, "invalid_request_error", "the WebSocket upgrade has no Sec-WebSocket-Key"),
        else => return h.fail(req, a, 426, "invalid_request_error", "this is the desktop helper's WebSocket endpoint (docs/dev/COMPUTER.md)"),
    };
    var up = req.respondWebSocket(.{ .key = try a.dupe(u8, key) }) catch return;
    up.flush() catch return;
    req.server.reader.state = .closing;
    var c: ws.Conn = .{ .io = app.io, .gpa = app.gpa, .in = up.input, .out = up.output, .max_message = 4 << 20 };
    defer c.deinit();
    const fd = conn.stream.socket.handle;
    if (!s.addFd(fd)) return c.close(1001, "the server is shutting down");
    defer s.removeFd(fd);
    var sock: Sock = .{ .c = &c, .fd = fd };
    const first = c.read() catch return;
    var why: decide.Why = .{};
    const reg = s.hub.register(a, first.data, device.name, .{ .ctx = &sock, .send = Sock.send, .close = Sock.close }, &why) orelse {
        const err = std.json.Stringify.valueAlloc(a, .{ .type = "error", .@"error" = why.message }, .{}) catch "";
        c.sendText(err) catch {};
        return c.close(1008, "hello refused");
    };
    defer s.hub.unregister(reg.helper);
    c.sendText(reg.welcome) catch return;
    var last_check = Io.Clock.awake.now(app.io);
    while (true) {
        const m = c.read() catch break;
        // A revoked token ends the helper's socket within one ping (20 s).
        const now = Io.Clock.awake.now(app.io);
        if (last_check.durationTo(now).toMilliseconds() > 15_000) {
            last_check = now;
            if (s.gate.check(tok, .desktop_helper) != .ok) {
                c.sendText("{\"type\":\"error\",\"error\":\"this device's token was revoked\"}") catch {};
                break;
            }
        }
        if (m.opcode == .text) s.hub.onFrame(reg.helper, m.data);
    }
    c.close(1000, "");
}

const Sock = struct {
    c: *ws.Conn,
    fd: std.posix.fd_t,

    fn send(ctx: *anyopaque, text: []const u8) bool {
        const k: *Sock = @ptrCast(@alignCast(ctx));
        k.c.sendText(text) catch return false;
        return true;
    }

    fn close(ctx: *anyopaque) void {
        const k: *Sock = @ptrCast(@alignCast(ctx));
        _ = std.posix.system.shutdown(k.fd, std.posix.SHUT.RDWR);
    }
};

// ---------------------------------------------------------------- /v1/computer

/// /v1/computer/status (GET), /approvals (GET), /approvals/{id} (POST {"decision":"approve"|"deny"}), /stop (POST
/// {"task_id"?}), /audit (GET ?limit=).
pub fn route(app: *App, req: *h.Request, a: std.mem.Allocator, rest: []const u8) !void {
    const s = service(app) orelse return h.fail(req, a, 404, "invalid_request_error", "computer control is switched off in this daemon's config (computer.enabled)");
    const m = req.head.method;
    const limit_q = @import("realtime.zig").query(req.head.target, "limit");
    const limit: usize = if (limit_q) |l| std.fmt.parseInt(usize, l, 10) catch 100 else 100;
    const device = (try admit(app, req, a, .approve)) orelse return;
    if (m == .GET and eql(rest, "status")) {
        const st = try s.hub.status(a);
        return h.jsonValue(req, a, .ok, .{ .computers = st.computers, .tasks = st.tasks, .pending_approvals = try pendingCount(s, a) });
    }
    if (m == .GET and eql(rest, "approvals")) {
        const views = try s.hub.approvals.list(app.io, a, s.hub.now());
        const Card = struct { id: []const u8, state: []const u8, expires_in_s: i64, computer: []const u8, task_id: []const u8, step: u32, app: []const u8, title: []const u8, url: []const u8, action: []const u8, kind: []const u8, text: []const u8, why: []const u8, goal: []const u8, requester: []const u8, decided_by: []const u8 };
        const cards = try a.alloc(Card, views.len);
        for (cards, views) |*c, v| c.* = .{ .id = v.id, .state = @tagName(v.state), .expires_in_s = @max(0, @divTrunc(v.expires_ms - s.hub.now(), 1000)), .computer = v.binding.helper_id, .task_id = v.binding.task_id, .step = v.binding.step, .app = v.binding.app, .title = v.binding.title, .url = v.binding.url, .action = try hub_mod.describeAction(a, v.binding), .kind = v.binding.action.kind, .text = v.binding.action.text, .why = v.binding.why, .goal = v.binding.goal, .requester = v.binding.requester, .decided_by = v.decided_by };
        return h.jsonValue(req, a, .ok, .{ .object = "list", .data = cards });
    }
    if (m == .POST and std.mem.startsWith(u8, rest, "approvals/")) {
        const id = rest["approvals/".len..];
        const body = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
        const B = struct { decision: []const u8 };
        const b = std.json.parseFromSliceLeaky(B, a, body, .{ .ignore_unknown_fields = true }) catch return h.fail(req, a, 400, "invalid_request_error", "the body is {\"decision\": \"approve\"} or {\"decision\": \"deny\"}");
        const approve = if (eql(b.decision, "approve")) true else if (eql(b.decision, "deny")) false else return h.fail(req, a, 400, "invalid_request_error", "decision is approve or deny");
        const by = try std.fmt.allocPrint(a, "web: {s}", .{device.name});
        const r = s.hub.approvals.decide(app.io, id, approve, by, s.hub.now());
        s.log.record(app.gpa, "approval.answer", .{ .approval_id = id, .decision = b.decision, .by = by, .result = @tagName(r) });
        return switch (r) {
            .ok => h.jsonValue(req, a, .ok, .{ .id = id, .state = if (approve) "approved" else "denied" }),
            .unknown => h.fail(req, a, 404, "invalid_request_error", "no such approval"),
            .expired => h.fail(req, a, 409, "invalid_request_error", "the approval expired; nothing was done"),
            .not_pending => h.fail(req, a, 409, "invalid_request_error", "the approval was already answered or cancelled"),
        };
    }
    if (m == .POST and eql(rest, "stop")) {
        const body = h.body(req, a) catch "";
        const B = struct { task_id: []const u8 = "" };
        const b = std.json.parseFromSliceLeaky(B, a, if (body.len > 0) body else "{}", .{ .ignore_unknown_fields = true }) catch B{};
        const n = s.hub.stop(a, b.task_id, try std.fmt.allocPrint(a, "web: {s}", .{device.name}));
        return h.jsonValue(req, a, .ok, .{ .stopped = n });
    }
    if (m == .GET and eql(rest, "audit")) {
        const lines = try s.log.tail(a, @min(limit, 1000));
        const parsed = try a.alloc(Value, lines.len);
        for (parsed, lines) |*p, l| p.* = std.json.parseFromSliceLeaky(Value, a, l, .{}) catch .null;
        return h.jsonValue(req, a, .ok, .{ .object = "list", .data = parsed });
    }
    return h.fail(req, a, 404, "invalid_request_error", "no such route");
}

fn pendingCount(s: *Service, a: std.mem.Allocator) !usize {
    var n: usize = 0;
    for (try s.hub.approvals.list(s.app.io, a, s.hub.now())) |v| if (v.state == .pending) {
        n += 1;
    };
    return n;
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}
