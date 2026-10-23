//! Integration tests of device pairing on a real daemon: with `pairing.required`, /mcp and /v1/realtime refuse a client
//! without a paired device's token (401 before the body or the upgrade) and take one as a Bearer header, a WebSocket
//! subprotocol entry or `?api_key=`; `localrouter pair` makes codes and tokens beside the running daemon; POST /v1/pair
//! trades a code once; a revoke takes effect on the next request; five wrong codes discard the pending ones. Without
//! `required`, nothing needs a token. Service scopes (inference, media, decisions, read, manage) gate every route
//! class, the legacy profile (voice, mcp, or none) keeps everything, and a job is its device's own.

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

    // Unpaired: 401 with OpenAI's envelope and how to pair, on /mcp, the socket and the other service routes alike.
    const bare = try d.call(a, .POST, "/mcp", list_tools);
    try testing.expectEqual(std.http.Status.unauthorized, bare.status);
    try testing.expect(has(bare.body, "\"code\":\"invalid_api_key\""));
    try testing.expect(has(bare.body, "\"type\":\"invalid_request_error\""));
    try testing.expect(has(bare.body, "localrouter pair add NAME"));
    try testing.expectEqual(std.http.Status.unauthorized, (try d.call(a, .GET, "/mcp", null)).status);
    try testing.expectEqual(std.http.Status.unauthorized, (try callBearer(&d, a, .POST, "/mcp", list_tools, "wrong")).status);
    try testing.expectEqual(std.http.Status.unauthorized, (try d.call(a, .GET, "/v1/models", null)).status);
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
    try testing.expect(has(me.body, try std.fmt.allocPrint(a, "\"paired\":true,\"device\":{{\"id\":\"{s}\",\"name\":\"kitchen tablet\",\"scopes\":[\"voice\",\"mcp\"]}}", .{p.device.id})));
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

const McpOut = struct { status: std.http.Status = .ok, text: []const u8 = "", is_error: bool = true };

/// One MCP tools/call with a device token; the result's first text and isError.
fn mcpAs(d: *Daemon, a: std.mem.Allocator, token: []const u8, tool: []const u8, args: []const u8) !McpOut {
    const body = try std.fmt.allocPrint(a, "{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{{\"name\":\"{s}\",\"arguments\":{s}}}}}", .{ tool, args });
    const r = try callBearer(d, a, .POST, "/mcp", body, token);
    if (r.status != .ok) return .{ .status = r.status, .text = r.body };
    const v = try std.json.parseFromSliceLeaky(Value, a, r.body, .{});
    const res = v.object.get("result").?.object;
    return .{ .text = res.get("content").?.array.items[0].object.get("text").?.string, .is_error = res.get("isError").?.bool };
}

const McpCall = struct {
    d: *Daemon,
    a: std.mem.Allocator,
    token: []const u8,
    tool: []const u8,
    args: []const u8,
    out: McpOut = .{},
    fn run(c: *McpCall) void {
        c.out = mcpAs(c.d, c.a, c.token, c.tool, c.args) catch .{ .text = "the call failed" };
    }
};

test "computer control needs paired devices even with pairing off: scopes gate every route; a paired helper runs a control; a lock waits for the tap" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "computer", "{" ++ harness.reserve ++ ", " ++ quiet_voice ++ "}");
    defer d.stop();
    const helper = std.mem.trim(u8, try pairCli(&d, a, &.{ "token", "mac", "--scopes", "desktop_helper" }), "\n");
    const phone = std.mem.trim(u8, try pairCli(&d, a, &.{ "token", "phone", "--scopes", "voice,mcp,computer,approve" }), "\n");
    const agent = std.mem.trim(u8, try pairCli(&d, a, &.{ "token", "agent" }), "\n"); // voice and mcp only
    const listed = try pairCli(&d, a, &.{"list"});
    try testing.expect(has(listed, "desktop_helper") and has(listed, "voice,mcp,computer,approve"));

    // Global pairing is off, so /mcp answers anyone; the computer tools, the socket and approvals still need paired devices.
    const anon = try mcpAs(&d, a, "", "computer_control", "{\"control\":{\"kind\":\"lock_screen\"}}");
    try testing.expect(anon.is_error and has(anon.text, "needs device pairing"));
    const refused = try mcpAs(&d, a, agent, "computer_control", "{\"control\":{\"kind\":\"open_app\",\"app\":\"Safari\"}}");
    try testing.expect(refused.is_error and has(refused.text, "computer scope"));
    try testing.expectEqual(std.http.Status.unauthorized, (try callBearer(&d, a, .GET, "/v1/computer/status", null, agent)).status);
    try testing.expectEqual(std.http.Status.unauthorized, (try callBearer(&d, a, .GET, "/v1/computer/status", null, helper)).status);
    // No helper yet: the tool says so.
    const none = try mcpAs(&d, a, phone, "computer_control", "{\"control\":{\"kind\":\"open_app\",\"app\":\"Safari\"}}");
    try testing.expect(none.is_error and has(none.text, "no computer is connected"));

    // The phone's token cannot open the helper's socket; the helper's can.
    var bad: voice.Ws = undefined;
    try bad.openAt(io, a, d.port, "/v1/desktop", try std.fmt.allocPrint(a, "openai-insecure-api-key.{s}", .{phone}));
    try testing.expect(has(bad.head, " 401 "));
    bad.close();
    var w: voice.Ws = undefined;
    try w.openAt(io, a, d.port, "/v1/desktop", try std.fmt.allocPrint(a, "openai-insecure-api-key.{s}", .{helper}));
    defer w.close();
    try testing.expect(has(w.head, " 101 "));
    try w.send("{\"type\":\"hello\",\"protocol\":1,\"helper\":{\"name\":\"Test Mac\",\"platform\":\"darwin\",\"version\":\"0.1.0\"},\"capabilities\":{\"controls\":[\"open_app\",\"lock_screen\"],\"computer_use\":true},\"access\":{\"accessibility\":true}}");
    try testing.expect(has(try w.next(a), "\"type\":\"welcome\""));

    // A simple control runs at once: the box sends it to the helper and returns the helper's sentence.
    var open: McpCall = .{ .d = &d, .a = a, .token = phone, .tool = "computer_control", .args = "{\"control\":{\"kind\":\"open_app\",\"app\":\"Safari\"}}" };
    const t1 = try std.Thread.spawn(.{}, McpCall.run, .{&open});
    const frame = try std.json.parseFromSliceLeaky(Value, a, try w.next(a), .{});
    try testing.expectEqualStrings("control", frame.object.get("type").?.string);
    try testing.expectEqualStrings("Safari", frame.object.get("control").?.object.get("app").?.string);
    try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"result\",\"id\":\"{s}\",\"ok\":true,\"done\":\"Opened Safari\"}}", .{frame.object.get("id").?.string}));
    t1.join();
    try testing.expect(!open.out.is_error and has(open.out.text, "Opened Safari"));

    const st = try callBearer(&d, a, .GET, "/v1/computer/status", null, phone);
    try testing.expectEqual(std.http.Status.ok, st.status);
    try testing.expect(has(st.body, "\"name\":\"Test Mac\""));

    // Locking the screen waits for a tap: the helper hears nothing until the phone denies it, and then nothing at all.
    var lock: McpCall = .{ .d = &d, .a = a, .token = phone, .tool = "computer_control", .args = "{\"control\":{\"kind\":\"lock_screen\"}}" };
    const t2 = try std.Thread.spawn(.{}, McpCall.run, .{&lock});
    var id: []const u8 = "";
    var tries: u32 = 0;
    while (id.len == 0 and tries < 200) : (tries += 1) {
        const cards = try std.json.parseFromSliceLeaky(struct { data: []const struct { id: []const u8, state: []const u8, action: []const u8 } }, a, (try callBearer(&d, a, .GET, "/v1/computer/approvals", null, phone)).body, .{ .ignore_unknown_fields = true });
        for (cards.data) |c| if (std.mem.eql(u8, c.state, "pending")) {
            try testing.expect(has(c.action, "lock the screen"));
            id = c.id;
        };
        if (id.len == 0) harness.sleepMs(io, 50);
    }
    try testing.expect(id.len > 0);
    // An agent cannot answer it; the phone can.
    try testing.expectEqual(std.http.Status.unauthorized, (try callBearer(&d, a, .POST, try std.fmt.allocPrint(a, "/v1/computer/approvals/{s}", .{id}), "{\"decision\":\"approve\"}", agent)).status);
    try testing.expectEqual(std.http.Status.ok, (try callBearer(&d, a, .POST, try std.fmt.allocPrint(a, "/v1/computer/approvals/{s}", .{id}), "{\"decision\":\"deny\"}", phone)).status);
    t2.join();
    try testing.expect(lock.out.is_error and has(lock.out.text, "said no"));

    const log = try callBearer(&d, a, .GET, "/v1/computer/audit?limit=50", null, phone);
    try testing.expect(has(log.body, "\"event\":\"helper.connected\"") and has(log.body, "\"event\":\"approval.answer\"") and has(log.body, "\"status\":\"blocked\""));
}

test "pairing required and scopes: a desktop helper's token reaches neither /mcp nor /v1/realtime; an unscoped device keeps both" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "scopes", "{" ++ harness.reserve ++ ", \"pairing\": {\"required\": true}, " ++ quiet_voice ++ "}");
    defer d.stop();
    const helper = std.mem.trim(u8, try pairCli(&d, a, &.{ "token", "mac", "--scopes", "desktop_helper" }), "\n");
    const old = std.mem.trim(u8, try pairCli(&d, a, &.{ "token", "phone" }), "\n");
    const no_mcp = try callBearer(&d, a, .POST, "/mcp", list_tools, helper);
    try testing.expectEqual(std.http.Status.forbidden, no_mcp.status);
    try testing.expect(has(no_mcp.body, "does not have the inference, media, decisions, read or manage scope"));
    var w: voice.Ws = undefined;
    try w.openAt(io, a, d.port, "/v1/realtime", try std.fmt.allocPrint(a, "realtime, openai-insecure-api-key.{s}", .{helper}));
    try testing.expect(has(w.head, " 403 "));
    w.close();
    try testing.expectEqual(std.http.Status.ok, (try callBearer(&d, a, .POST, "/mcp", list_tools, old)).status);
    _ = try pairCli(&d, a, &.{ "scopes", "phone", "voice" }); // now without mcp: refused on the next request
    try testing.expectEqual(std.http.Status.forbidden, (try callBearer(&d, a, .POST, "/mcp", list_tools, old)).status);
}

/// The status of `method path` with `token` (none: no Authorization header).
fn statusAs(d: *Daemon, a: std.mem.Allocator, method: std.http.Method, path: []const u8, body: ?[]const u8, token: ?[]const u8) !std.http.Status {
    return (if (token) |t| try callBearer(d, a, method, path, body, t) else try d.call(a, method, path, body)).status;
}

/// tools/list's names for `token`.
fn toolNames(d: *Daemon, a: std.mem.Allocator, token: []const u8) ![]const u8 {
    const r = try callBearer(d, a, .POST, "/mcp", list_tools, token);
    try testing.expectEqual(std.http.Status.ok, r.status);
    var w: Io.Writer.Allocating = .init(a);
    for ((try std.json.parseFromSliceLeaky(Value, a, r.body, .{})).object.get("result").?.object.get("tools").?.array.items) |t| try w.writer.print("{s},", .{t.object.get("name").?.string});
    return w.written();
}

test "service scopes: a wrong or missing scope is refused on every route class; the legacy profile keeps everything; manage changes models; a job is its device's own" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "service-scopes", "{" ++ harness.reserve ++ ", \"pairing\": {\"required\": true}, " ++ quiet_voice ++
        ", \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 1}}]}");
    defer d.stop();
    const tok = struct {
        fn of(dd: *Daemon, aa: std.mem.Allocator, name: []const u8, scopes: ?[]const u8) ![]const u8 {
            const out = if (scopes) |s| try pairCli(dd, aa, &.{ "token", name, "--scopes", s }) else try pairCli(dd, aa, &.{ "token", name });
            return std.mem.trim(u8, out, " \r\n");
        }
    }.of;
    const legacy = try tok(&d, a, "phone", null); // paired before scopes: voice and mcp
    const coder = try tok(&d, a, "opencode", "inference,media,read");
    const other = try tok(&d, a, "deepseek", "media");
    const reader = try tok(&d, a, "dashboard", "read");
    const judge = try tok(&d, a, "router", "decisions");
    const helper = try tok(&d, a, "mac", "desktop_helper");
    const operator = try tok(&d, a, "ops", "read,manage");
    try testing.expect(has(try pairCli(&d, a, &.{"list"}), "voice,mcp (legacy)"));

    // Every route class: no token is 401, a token without the scope 403, the scope (or the legacy profile) gets through.
    const image = "{\"model\":\"tp\",\"prompt\":\"bars\",\"size\":\"256x256\",\"response_format\":\"url\"}";
    const Case = struct { method: std.http.Method, path: []const u8, body: ?[]const u8, wrong: []const u8, right: []const u8 };
    const cases = [_]Case{
        .{ .method = .GET, .path = "/v1/models", .body = null, .wrong = judge, .right = reader },
        .{ .method = .GET, .path = "/v1/tools", .body = null, .wrong = other, .right = reader },
        .{ .method = .GET, .path = "/v1/skills/model-manager", .body = null, .wrong = helper, .right = reader },
        .{ .method = .POST, .path = "/v1/images/generations", .body = image, .wrong = reader, .right = other },
        .{ .method = .POST, .path = "/v1/audio/speech", .body = "{\"input\":\"hi\"}", .wrong = judge, .right = other },
        .{ .method = .POST, .path = "/v1/videos", .body = "{\"prompt\":\"x\"}", .wrong = judge, .right = other },
        .{ .method = .POST, .path = "/v1/chat/completions", .body = "{\"messages\":[]}", .wrong = other, .right = coder },
        .{ .method = .POST, .path = "/v1/systemone", .body = "{}", .wrong = coder, .right = judge },
        .{ .method = .POST, .path = "/v1/tools/release", .body = "{}", .wrong = coder, .right = operator },
        .{ .method = .POST, .path = "/mcp", .body = list_tools, .wrong = helper, .right = judge },
    };
    for (cases) |c| {
        try testing.expectEqual(std.http.Status.unauthorized, try statusAs(&d, a, c.method, c.path, c.body, null));
        try testing.expectEqual(std.http.Status.unauthorized, try statusAs(&d, a, c.method, c.path, c.body, "not-a-token"));
        const refused = try callBearer(&d, a, c.method, c.path, c.body, c.wrong);
        if (refused.status != .forbidden) std.debug.print("{s} {s}: {d} {s}\n", .{ @tagName(c.method), c.path, @backingInt(refused.status), refused.body });
        try testing.expectEqual(std.http.Status.forbidden, refused.status);
        try testing.expect(has(refused.body, "scope"));
        for ([_][]const u8{ c.right, legacy }) |t| {
            const s = try statusAs(&d, a, c.method, c.path, c.body, t);
            if (s == .unauthorized or s == .forbidden) std.debug.print("{s} {s}: refused {d}\n", .{ @tagName(c.method), c.path, @backingInt(s) });
            try testing.expect(s != .unauthorized and s != .forbidden);
        }
    }
    // /v1/realtime: inference (or the legacy voice) opens it; media alone does not.
    {
        var w: voice.Ws = undefined;
        try w.openAt(io, a, d.port, "/v1/realtime", try std.fmt.allocPrint(a, "realtime, openai-insecure-api-key.{s}", .{other}));
        try testing.expect(has(w.head, " 403 "));
        w.close();
    }
    try expectSession(&d, a, "/v1/realtime", try std.fmt.allocPrint(a, "realtime, openai-insecure-api-key.{s}", .{coder}));
    try expectSession(&d, a, try std.fmt.allocPrint(a, "/v1/realtime?api_key={s}", .{legacy}), "realtime");
    // The open routes stay open.
    try testing.expectEqual(std.http.Status.ok, try statusAs(&d, a, .GET, "/health", null, null));
    try testing.expectEqual(std.http.Status.ok, try statusAs(&d, a, .GET, "/v1/pair", null, null));
    try testing.expect(has((try callBearer(&d, a, .GET, "/v1/pair", null, coder)).body, "\"scopes\":[\"inference\",\"media\",\"read\"]"));
    try testing.expect(has((try callBearer(&d, a, .GET, "/v1/pair", null, legacy)).body, "\"scopes\":[\"voice\",\"mcp\"]"));

    // MCP: tools/list shows what the token may call, and each tool checks its own scope.
    const judged = try toolNames(&d, a, judge);
    try testing.expect(has(judged, "decide,") and std.mem.indexOf(u8, judged, "generate_image,") == null and std.mem.indexOf(u8, judged, "release_models,") == null);
    const everything = try toolNames(&d, a, legacy);
    for ([_][]const u8{ "list_models,", "generate_image,", "chat,", "decide,", "release_models,", "load_model,", "get_job," }) |n| try testing.expect(has(everything, n));
    const no_media = try mcpAs(&d, a, judge, "generate_image", "{\"prompt\":\"x\",\"model\":\"tp\",\"size\":\"256x256\"}");
    try testing.expect(no_media.is_error and has(no_media.text, "does not have the media scope"));
    // Changing models needs manage, over MCP and HTTP alike; read lists them.
    for ([_][]const u8{ "load_model", "unload_model", "reserve_for", "release_models" }) |n| {
        const r = try mcpAs(&d, a, coder, n, "{\"model\":\"tp\"}");
        try testing.expect(r.is_error and has(r.text, "does not have the manage scope"));
    }
    try testing.expectEqual(std.http.Status.forbidden, try statusAs(&d, a, .POST, "/v1/tools/tp/load", "{}", coder));
    try testing.expectEqual(std.http.Status.forbidden, try statusAs(&d, a, .POST, "/v1/tools/tp/unload", "{}", reader));
    try testing.expectEqual(std.http.Status.ok, try statusAs(&d, a, .POST, "/v1/tools/tp/load", "{}", operator));
    try testing.expect(!(try mcpAs(&d, a, operator, "release_models", "{}")).is_error);
    try testing.expect(!(try mcpAs(&d, a, legacy, "release_models", "{}")).is_error);
    try testing.expect(!(try mcpAs(&d, a, reader, "list_models", "{}")).is_error);

    // Ownership: a job, its state and its files are its device's alone (another device: as if absent).
    const made = try callBearer(&d, a, .POST, "/v1/images/generations", image, coder);
    try testing.expectEqual(std.http.Status.ok, made.status);
    const url = (try std.json.parseFromSliceLeaky(struct { data: []const struct { url: []const u8 } }, a, made.body, .{ .ignore_unknown_fields = true })).data[0].url;
    const path = url[std.mem.indexOf(u8, url, "/v1/files/").?..];
    const id = path["/v1/files/".len..][0..std.mem.indexOfScalar(u8, path["/v1/files/".len..], '/').?];
    const job = try std.fmt.allocPrint(a, "/v1/jobs/{s}", .{id});
    try testing.expectEqual(std.http.Status.ok, try statusAs(&d, a, .GET, path, null, coder));
    try testing.expectEqual(std.http.Status.ok, try statusAs(&d, a, .GET, job, null, coder));
    for ([_][]const u8{ other, legacy, operator }) |t| {
        try testing.expectEqual(std.http.Status.not_found, try statusAs(&d, a, .GET, path, null, t));
        try testing.expectEqual(std.http.Status.not_found, try statusAs(&d, a, .GET, job, null, t));
    }
    try testing.expectEqual(std.http.Status.unauthorized, try statusAs(&d, a, .GET, path, null, null));
    const theirs = try mcpAs(&d, a, other, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\"}}", .{id}));
    try testing.expect(theirs.is_error and has(theirs.text, "no such job"));
    try testing.expect((try mcpAs(&d, a, other, "cancel_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\"}}", .{id}))).is_error);
    try testing.expect(!(try mcpAs(&d, a, coder, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\"}}", .{id}))).is_error);
    // A revoke counts at once, for its jobs too.
    _ = try pairCli(&d, a, &.{ "revoke", "opencode" });
    try testing.expectEqual(std.http.Status.unauthorized, try statusAs(&d, a, .GET, job, null, coder));
}
