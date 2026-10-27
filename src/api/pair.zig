//! Device pairing over HTTP (the devices themselves: `src/pair.zig`). `POST /v1/pair` trades a one-time code for a
//! token and `GET /v1/pair` says whether the token presented is paired; both are served whether or not pairing is
//! required, so a client can pair ahead of it. With `pairing.required`, `admit` gates every other route on the scope
//! it needs (`routes.need`), MCP tools and a voice session's tools check theirs per call, and `Caller.sees` keeps each
//! device to its own jobs and files. A token comes as `Authorization: Bearer <token>`, as a `Sec-WebSocket-Protocol`
//! entry `openai-insecure-api-key.<token>` (the OpenAI browser SDK's), or as `?api_key=<token>` (the built-in page's).

const std = @import("std");
const h = @import("http.zig");
const routes = @import("routes.zig");
const store = @import("../pair.zig");
const realtime = @import("realtime.zig");
const jobs = @import("../sched/jobs.zig");

const App = routes.App;
const key_protocol = "openai-insecure-api-key.";

/// The token in one request header, or null: a Bearer authorization or a subprotocol list's key entry.
pub fn fromHeader(name: []const u8, value: []const u8) ?[]const u8 {
    if (std.ascii.eqlIgnoreCase(name, "authorization")) {
        const v = std.mem.trim(u8, value, " \t");
        if (v.len > 7 and std.ascii.eqlIgnoreCase(v[0..7], "bearer ")) {
            const t = std.mem.trim(u8, v[7..], " \t");
            return if (t.len > 0) t else null;
        }
        return null;
    }
    if (!std.ascii.eqlIgnoreCase(name, "sec-websocket-protocol")) return null;
    var it = std.mem.splitScalar(u8, value, ',');
    while (it.next()) |p| {
        const e = std.mem.trim(u8, p, " \t");
        if (e.len > key_protocol.len and std.mem.startsWith(u8, e, key_protocol)) return e[key_protocol.len..];
    }
    return null;
}

/// The token a request presents, or null. Points into the head: use it before the body is read.
pub fn credential(req: *const h.Request) ?[]const u8 {
    var it = req.iterateHeaders();
    while (it.next()) |hd| if (fromHeader(hd.name, hd.value)) |t| return t;
    const q = realtime.query(req.head.target, "api_key") orelse return null;
    return if (q.len > 0) q else null;
}

fn devices(app: *const App) store.Store {
    return .{ .dir = app.cfg.data_dir };
}

/// The paired device whose token the request presents, or null.
fn device(app: *App, req: *const h.Request, a: std.mem.Allocator) ?store.Device {
    const t = credential(req) orelse return null;
    return deviceOf(app, a, t);
}

/// The paired device whose token `t` is, or null (re-read from the devices file: a revoke counts at once).
pub fn deviceOf(app: *App, a: std.mem.Allocator, t: []const u8) ?store.Device {
    if (t.len == 0) return null;
    return devices(app).verify(app.io, a, t) catch |err| {
        std.log.warn("pairing: cannot read {s}/devices.json: {s}", .{ app.cfg.data_dir, @errorName(err) });
        return null;
    };
}

/// Who is asking, as far as the scopes go: with pairing off, anyone (every scope, every job); with it required, the
/// paired device behind the request's token.
pub const Caller = struct {
    required: bool = false,
    device: ?store.Device = null, // set only when pairing is required

    /// The owner its jobs get: the device's id, or "" with pairing off.
    pub fn owner(c: Caller) []const u8 {
        return if (c.device) |d| d.id else "";
    }

    pub fn may(c: Caller, n: store.Need) bool {
        if (!c.required) return true;
        const d = c.device orelse return false;
        return d.grants(n);
    }

    /// Whether it may see job `j` (its state, outputs and files) or cancel it: with pairing required, only the device
    /// that asked for it, with `read` or the job's own scope. A job of the daemon's own (no owner) is nobody's.
    pub fn sees(c: Caller, j: *const jobs.Job) bool {
        if (!c.required) return true;
        const d = c.device orelse return false;
        return j.owner.len > 0 and std.mem.eql(u8, j.owner, d.id) and (d.grants(.read) or d.grants(store.jobNeed(j.kind())));
    }
};

/// The caller when the request may go on: pairing is off, or it presents a paired device's token whose scopes satisfy
/// `need`. Otherwise null, with the 401 (no or an unknown token) or 403 (a scope missing) already sent. Call it before
/// reading the body or upgrading.
pub fn admit(app: *App, req: *h.Request, a: std.mem.Allocator, need: store.Need) !?Caller {
    if (!app.cfg.pairing.required) return .{};
    if (device(app, req, a)) |d| {
        if (d.grants(need)) return .{ .required = true, .device = d };
        try h.fail(req, a, 403, "invalid_request_error", try std.fmt.allocPrint(a, "this device's token does not have the {s} scope: on the box, `localrouter pair scopes NAME ...` adds it", .{need.scopes()}));
        return null;
    }
    try h.failWith(req, a, 401, "invalid_request_error", "invalid_api_key", "this device is not paired with this LocalRouter: on the box run `localrouter pair add NAME` and enter the code in the app (or `localrouter pair token NAME --scopes ...` for a harness or client that takes a token), then send the token as Authorization: Bearer", &.{.{ .name = "www-authenticate", .value = "Bearer" }});
    return null;
}

/// The caller behind token `t` now, for a long-lived connection (a voice session) checking each action: a revoke or a
/// scope change counts at once. Null when pairing is required and `t` is no longer a paired device's.
pub fn callerOf(app: *App, a: std.mem.Allocator, t: []const u8) ?Caller {
    if (!app.cfg.pairing.required) return .{};
    return .{ .required = true, .device = deviceOf(app, a, t) orelse return null };
}

/// GET /v1/pair: {"required", "paired", "device": {"id", "name", "scopes"} or null}, for whatever token is presented
/// (`scopes` as the device holds them, `all` spelled out as `all` and every scope it holds today, so a client that checks
/// for a scope by name finds it; a device paired before scopes holds `voice` and `mcp`).
pub fn status(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const Dev = struct { id: []const u8, name: []const u8, scopes: []const []const u8 };
    const d = device(app, req, a);
    const text = try std.json.Stringify.valueAlloc(a, .{
        .required = app.cfg.pairing.required,
        .paired = d != null,
        .device = if (d) |x| @as(?Dev, .{ .id = x.id, .name = x.name, .scopes = listed(x) }) else null,
    }, .{});
    try h.json(req, .ok, text);
}

const legacy_listed = [_][]const u8{ "voice", "mcp" };
const all_listed = [_][]const u8{store.all} ++ store.all_scopes;

/// A device's scopes as `GET /v1/pair` lists them.
pub fn listed(d: store.Device) []const []const u8 {
    if (d.scopes.len == 0) return &legacy_listed;
    for (d.scopes) |s| if (std.mem.eql(u8, s, store.all)) return &all_listed;
    return d.scopes;
}

/// POST /v1/pair {"code": "XXXX-XXXX", "name": "optional"}: a new device and its token. A wrong code counts toward
/// the guard; the fifth within 10 minutes discards every pending code.
pub fn redeem(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const B = struct { code: []const u8 = "", name: []const u8 = "" };
    const b = std.json.parseFromSliceLeaky(B, a, text, .{ .ignore_unknown_fields = true }) catch
        return h.fail(req, a, 400, "invalid_request_error", "the body must be {\"code\": \"XXXX-XXXX\", \"name\": \"this device's name\"} (name optional)");
    if (b.code.len == 0) return h.fail(req, a, 400, "invalid_request_error", "code is required: the XXXX-XXXX code `localrouter pair add NAME` printed on the box");
    const name = std.mem.trim(u8, b.name, " \t\r\n");
    if (name.len > store.max_name_len) return h.fail(req, a, 400, "invalid_request_error", "name is too long (at most 64 bytes)");
    const s = devices(app);
    const at = store.now(app.io);
    const m = s.redeem(app.io, a, b.code, name, at) catch |err| {
        std.log.warn("pairing: cannot update {s}/devices.json: {s}", .{ app.cfg.data_dir, @errorName(err) });
        return h.fail(req, a, 500, "server_error", "the server could not record the device");
    };
    const got = m orelse {
        if (app.pair_guard.wrong(app.io, at)) {
            std.log.warn("pairing: {d} wrong codes within 10 minutes; every pending code is discarded (make a new one with localrouter pair add NAME)", .{store.Guard.limit});
            s.discardPending(app.io, a, at) catch |err| std.log.warn("pairing: cannot discard the pending codes: {s}", .{@errorName(err)});
        }
        return h.failWith(req, a, 401, "invalid_request_error", "invalid_pairing_code", "that pairing code is wrong, expired (codes last 10 minutes) or already used: run `localrouter pair add NAME` on the box for a new one", &.{});
    };
    std.log.info("pairing: paired device {s} ({s})", .{ @as([]const u8, &got.id), got.name });
    try h.jsonValue(req, a, .ok, .{ .token = @as([]const u8, &got.token), .device = .{ .id = @as([]const u8, &got.id), .name = got.name }, .required = app.cfg.pairing.required });
}

test "a token from a Bearer header, a subprotocol entry or the query; never the realtime entry" {
    const t = std.testing;
    try t.expectEqualStrings("abc", fromHeader("Authorization", "Bearer abc").?);
    try t.expectEqualStrings("abc", fromHeader("authorization", "  bearer   abc ").?);
    try t.expect(fromHeader("Authorization", "Basic YWJj") == null);
    try t.expect(fromHeader("Authorization", "Bearer ") == null);
    try t.expect(fromHeader("Host", "Bearer abc") == null);
    try t.expectEqualStrings("tok_-9", fromHeader("Sec-WebSocket-Protocol", "realtime, openai-insecure-api-key.tok_-9, openai-beta.realtime-v1").?);
    try t.expect(fromHeader("sec-websocket-protocol", "realtime, openai-beta.realtime-v1") == null);
    try t.expect(fromHeader("sec-websocket-protocol", "openai-insecure-api-key.") == null);

    var in: std.Io.Reader = .fixed("GET /v1/realtime?model=m&api_key=q1 HTTP/1.1\r\nHost: a\r\n\r\n");
    var out_buf: [16]u8 = undefined;
    var out: std.Io.Writer = .fixed(&out_buf);
    var server = std.http.Server.init(&in, &out);
    const q = try server.receiveHead();
    try t.expectEqualStrings("q1", credential(&q).?);
    var in2: std.Io.Reader = .fixed("POST /mcp?api_key=q HTTP/1.1\r\nHost: a\r\nAuthorization: Bearer hdr\r\n\r\n");
    var server2 = std.http.Server.init(&in2, &out);
    const r = try server2.receiveHead();
    try t.expectEqualStrings("hdr", credential(&r).?);
    var in3: std.Io.Reader = .fixed("GET /v1/pair?api_key= HTTP/1.1\r\nHost: a\r\n\r\n");
    var server3 = std.http.Server.init(&in3, &out);
    const n = try server3.receiveHead();
    try t.expect(credential(&n) == null);
}
