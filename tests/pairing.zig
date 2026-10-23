//! Integration tests of device pairing on a real daemon: with `pairing.required`, /mcp and /v1/realtime refuse a client
//! without a paired device's token (401 before the body or the upgrade) and take one as a Bearer header, a WebSocket
//! subprotocol entry or `?api_key=`; `localrouter pair` makes codes and tokens beside the running daemon; POST /v1/pair
//! trades a code once; a revoke takes effect on the next request; five wrong codes discard the pending ones. Without
//! `required`, nothing needs a token.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");
const voice = @import("voice.zig");

const Daemon = harness.Daemon;
const Value = std.json.Value;

const list_tools = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}";
const quiet_voice = "\"voice\": {\"prime_chat\": false, \"warm_speech\": false}";

/// `Daemon.call` with `Authorization: Bearer token`.
fn callBearer(d: *Daemon, a: std.mem.Allocator, method: std.http.Method, path: []const u8, body: ?[]const u8, token: []const u8) !Daemon.Resp {
    var out: Io.Writer.Allocating = .init(a);
    const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}{s}", .{ d.port, path });
    const auth = try std.fmt.allocPrint(a, "Bearer {s}", .{token});
    const res = try d.client.fetch(.{ .location = .{ .url = url }, .method = method, .payload = body, .response_writer = &out.writer, .keep_alive = false, .headers = .{ .authorization = .{ .override = auth } } });
    return .{ .status = res.status, .body = out.written() };
}

/// `localrouter pair ARGS --data <the daemon's data>`: its stdout; it must exit 0.
fn pairCli(d: *Daemon, a: std.mem.Allocator, args: []const []const u8) ![]const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(a, &.{ opts.localrouter_exe, "pair" });
    try argv.appendSlice(a, args);
    try argv.appendSlice(a, &.{ "--data", try std.fmt.allocPrint(a, "{s}/data", .{d.dir}) });
    const r = try std.process.run(a, d.io, .{ .argv = argv.items });
    if (r.term != .exited or r.term.exited != 0) std.debug.print("pair {s}: {s}\n", .{ args[0], r.stderr });
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, r.term);
    return r.stdout;
}

/// The code `pair add` printed (its `code     XXXX-XXXX` line).
fn codeOf(out: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, out, "code     ").? + "code     ".len;
    return out[at .. at + 9];
}

fn has(hay: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, hay, needle) != null) return true;
    std.debug.print("missing '{s}' in '{s}'\n", .{ needle, hay });
    return false;
}

fn redeem(d: *Daemon, a: std.mem.Allocator, code: []const u8, name: []const u8) !Daemon.Resp {
    return d.call(a, .POST, "/v1/pair", try std.fmt.allocPrint(a, "{{\"code\":\"{s}\",\"name\":\"{s}\"}}", .{ code, name }));
}

/// Opens /v1/realtime (`target`) and expects 101, `realtime` selected, never a key echoed, and `session.created`.
fn expectSession(d: *Daemon, a: std.mem.Allocator, target: []const u8, protocol: []const u8) !void {
    var w: voice.Ws = undefined;
    try w.openAt(d.io, a, d.port, target, protocol);
    defer w.close();
    try testing.expect(has(w.head, "101 Switching Protocols"));
    try testing.expect(std.ascii.findIgnoreCase(w.head, "sec-websocket-protocol: realtime\r\n") != null);
    try testing.expect(std.ascii.findIgnoreCase(w.head, "openai-insecure-api-key") == null);
    var seen: std.ArrayList(Value) = .empty;
    _ = try w.until(a, "session.created", &seen);
}

test "pairing required: /mcp and /v1/realtime need a paired token; codes trade once; revoke and the guard act at once" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "pairing", "{" ++ harness.reserve ++ ", \"pairing\": {\"required\": true}, " ++ quiet_voice ++ "}");
    defer d.stop();

    // Unpaired: 401 with OpenAI's envelope and how to pair; the other routes are as before.
    const bare = try d.call(a, .POST, "/mcp", list_tools);
    try testing.expectEqual(std.http.Status.unauthorized, bare.status);
    try testing.expect(has(bare.body, "\"code\":\"invalid_api_key\""));
    try testing.expect(has(bare.body, "\"type\":\"invalid_request_error\""));
    try testing.expect(has(bare.body, "localrouter pair add NAME"));
    try testing.expectEqual(std.http.Status.unauthorized, (try d.call(a, .GET, "/mcp", null)).status);
    try testing.expectEqual(std.http.Status.unauthorized, (try callBearer(&d, a, .POST, "/mcp", list_tools, "wrong")).status);
    try testing.expectEqual(std.http.Status.ok, (try d.call(a, .GET, "/v1/models", null)).status);
    try testing.expect(has((try d.call(a, .GET, "/v1/pair", null)).body, "{\"required\":true,\"paired\":false,\"device\":null}"));
    {
        var w: voice.Ws = undefined;
        try w.open(io, a, d.port, "realtime");
        defer w.close();
        try testing.expect(has(w.head, "HTTP/1.1 401"));
        try testing.expect(std.ascii.findIgnoreCase(w.head, "www-authenticate: Bearer\r\n") != null);
        try testing.expect(std.mem.indexOf(u8, w.head, "101") == null);
    }

    // A token minted by the CLI beside the running daemon: Bearer on /mcp, a subprotocol entry or ?api_key= on the socket.
    const token = std.mem.trim(u8, try pairCli(&d, a, &.{ "token", "claude-code" }), " \r\n");
    try testing.expectEqual(@as(usize, 43), token.len);
    const listed = try callBearer(&d, a, .POST, "/mcp", list_tools, token);
    try testing.expectEqual(std.http.Status.ok, listed.status);
    try testing.expect((try std.json.parseFromSliceLeaky(Value, a, listed.body, .{})).object.get("result").?.object.get("tools").?.array.items.len > 0);
    try expectSession(&d, a, "/v1/realtime", try std.fmt.allocPrint(a, "realtime, openai-insecure-api-key.{s}", .{token}));
    try expectSession(&d, a, try std.fmt.allocPrint(a, "/v1/realtime?api_key={s}", .{token}), "realtime");

    // A code from `pair add`, typed in lower case without its dash; the device is named by the request.
    const added = try pairCli(&d, a, &.{ "add", "JJ's phone", "--url", "https://box.example.ts.net:10443" });
    try testing.expect(has(added, "link     localrouter://pair?url=https%3A%2F%2Fbox.example.ts.net%3A10443&code="));
    const code = codeOf(added);
    var typed: [8]u8 = undefined;
    for (typed[0..4], code[0..4]) |*t, c| t.* = std.ascii.toLower(c);
    for (typed[4..8], code[5..9]) |*t, c| t.* = std.ascii.toLower(c);
    const paired = try redeem(&d, a, &typed, "kitchen tablet");
    try testing.expectEqual(std.http.Status.ok, paired.status);
    const P = struct { token: []const u8, device: struct { id: []const u8, name: []const u8 }, required: bool };
    const p = try std.json.parseFromSliceLeaky(P, a, paired.body, .{});
    try testing.expectEqualStrings("kitchen tablet", p.device.name);
    try testing.expect(p.required);
    try testing.expectEqual(std.http.Status.ok, (try callBearer(&d, a, .POST, "/mcp", list_tools, p.token)).status);
    const me = try callBearer(&d, a, .GET, "/v1/pair", null, p.token);
    try testing.expect(has(me.body, try std.fmt.allocPrint(a, "\"paired\":true,\"device\":{{\"id\":\"{s}\",\"name\":\"kitchen tablet\"}}", .{p.device.id})));
    const again = try redeem(&d, a, code, "");
    try testing.expectEqual(std.http.Status.unauthorized, again.status);
    try testing.expect(has(again.body, "wrong, expired"));
    const devices = try Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/data/devices.json", .{d.dir}), a, .limited(1 << 20));
    try testing.expect(std.mem.indexOf(u8, devices, p.token) == null and std.mem.indexOf(u8, devices, token) == null);

    // list never shows a token; revoke refuses the next request, no restart.
    const table = try pairCli(&d, a, &.{"list"});
    try testing.expect(has(table, "kitchen tablet") and has(table, "claude-code"));
    try testing.expect(std.mem.indexOf(u8, table, token) == null);
    try testing.expect(has(try pairCli(&d, a, &.{ "revoke", "kitchen tablet" }), p.device.id));
    try testing.expectEqual(std.http.Status.unauthorized, (try callBearer(&d, a, .POST, "/mcp", list_tools, p.token)).status);
    try testing.expectEqual(std.http.Status.ok, (try callBearer(&d, a, .POST, "/mcp", list_tools, token)).status);

    // The reused code above was one wrong code; four more within 10 minutes discard the pending code.
    const pending = codeOf(try pairCli(&d, a, &.{ "add", "spare" }));
    for (0..4) |_| try testing.expectEqual(std.http.Status.unauthorized, (try redeem(&d, a, "0000-0000", "")).status);
    try testing.expectEqual(std.http.Status.unauthorized, (try redeem(&d, a, pending, "")).status);
    const fresh = codeOf(try pairCli(&d, a, &.{ "add", "spare" }));
    try testing.expectEqual(std.http.Status.ok, (try redeem(&d, a, fresh, "")).status);
    try testing.expectEqual(std.http.Status.bad_request, (try d.call(a, .POST, "/v1/pair", "{}")).status);
}

test "pairing off (the default): /mcp and /v1/realtime need no token; /v1/pair still pairs" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "pairing-off", "{" ++ harness.reserve ++ ", " ++ quiet_voice ++ "}");
    defer d.stop();

    try testing.expectEqual(std.http.Status.ok, (try d.call(a, .POST, "/mcp", list_tools)).status);
    try testing.expectEqual(std.http.Status.ok, (try callBearer(&d, a, .POST, "/mcp", list_tools, "anything")).status);
    try expectSession(&d, a, "/v1/realtime", "realtime, openai-beta.realtime-v1");
    try testing.expect(has((try d.call(a, .GET, "/v1/pair", null)).body, "{\"required\":false,\"paired\":false,\"device\":null}"));
    const code = codeOf(try pairCli(&d, a, &.{ "add", "phone" }));
    const paired = try redeem(&d, a, code, "");
    try testing.expectEqual(std.http.Status.ok, paired.status);
    try testing.expect(has(paired.body, "\"name\":\"phone\"},\"required\":false"));
}
