//! The harness contract (docs/HARNESS.md) on a real daemon: `GET /v1/capabilities` agrees with the routes it
//! describes, and what our own clients actually sent to the Jarvis box (tests/fixtures/harness/*.jsonl, recorded by
//! tools/harness/record_proxy.py) is still accepted. Replaying a fixture checks shapes, not answers: every HTTP request
//! lands where it did on the box (same status class), and no event or frame is refused as unknown or malformed. Model
//! and voice names are dropped first, since this daemon's models are fakes with other names.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");
const Ws = @import("voice.zig").Ws;

const Daemon = harness.Daemon;
const Value = std.json.Value;

const contract = "2026-10-23";

/// A chat model (the fake TensorFold server), speech and transcription (the fake workers), an image model with a licence.
fn config(a: std.mem.Allocator) ![]const u8 {
    return std.fmt.allocPrint(a, "{{" ++ harness.reserve ++ ", \"voice\": {{\"prime_chat\": false, \"warm_speech\": false}}, \"tools\": [" ++
        "{{\"id\": \"chat\", \"kind\": \"chat\", \"engine\": \"openai_server\", \"resident_bytes\": 41943040, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"]}}}}," ++
        "{{\"id\": \"mouth\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true, \"options\": {{\"voices\": [\"plain\"]}}}}," ++
        "{{\"id\": \"ears\", \"kind\": \"transcription\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true}}," ++
        "{{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {{\"step_ms\": 5}}, " ++
        "\"license\": {{\"name\": \"Test Licence\", \"url\": \"https://example.org/licence\", \"commercial\": false, \"excluded_regions\": [\"Nowhere\"]}}}}]}}", .{ opts.chat_server, opts.speech_worker, opts.speech_worker });
}

fn get(v: Value, path: []const []const u8) Value {
    var x = v;
    for (path) |k| x = if (x == .object) x.object.get(k) orelse .null else .null;
    return x;
}

fn strings(a: std.mem.Allocator, v: Value) ![]const []const u8 {
    const out = try a.alloc([]const u8, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = x.string;
    return out;
}

fn has(list: []const []const u8, s: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

test "capabilities: the contract version, and every surface it lists answers as it says" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, testing.io, "contract-caps", try config(a));
    defer d.stop();

    const r = try d.call(a, .GET, "/v1/capabilities", null);
    try testing.expectEqual(std.http.Status.ok, r.status);
    const caps = try std.json.parseFromSliceLeaky(Value, a, r.body, .{});
    try testing.expectEqualStrings(contract, get(caps, &.{"contract"}).string);
    try testing.expect(get(caps, &.{ "chat", "available" }).bool);
    try testing.expect(!get(caps, &.{ "auth", "pairing_required" }).bool);
    try testing.expectEqualStrings("openai-realtime-beta", get(caps, &.{ "realtime", "profile" }).string);
    const kinds = try strings(a, get(caps, &.{ "models", "kinds" }));
    for ([_][]const u8{ "chat", "speech", "transcription", "image" }) |k| try testing.expect(has(kinds, k));
    try testing.expect(!get(caps, &.{ "decisions", "available" }).bool); // no decision model here

    // Each model's licence, as /v1/models lists it: the configured one, and none (omitted) where nothing states one.
    const models = get(try std.json.parseFromSliceLeaky(Value, a, (try d.call(a, .GET, "/v1/models", null)).body, .{}), &.{"data"}).array.items;
    const licenses = get(caps, &.{ "models", "licenses" }).array.items;
    try testing.expectEqual(models.len, licenses.len);
    for (models, licenses) |m, l| {
        const id = m.object.get("id").?.string;
        try testing.expectEqualStrings(id, l.object.get("id").?.string);
        try testing.expectEqual(std.mem.eql(u8, id, "tp"), m.object.get("license") != null);
        try testing.expectEqual(m.object.get("license") != null, l.object.get("license") != null);
    }
    for (models) |m| {
        const l = m.object.get("license") orelse continue;
        try testing.expectEqualStrings("Test Licence", l.object.get("name").?.string);
        try testing.expectEqualStrings("https://example.org/licence", l.object.get("url").?.string);
        try testing.expect(!l.object.get("commercial").?.bool);
        try testing.expectEqualStrings("Nowhere", l.object.get("excluded_regions").?.array.items[0].string);
    }
    for (licenses) |l| if (l.object.get("license")) |x| try testing.expectEqualStrings("Test Licence", x.object.get("name").?.string);

    // MCP: the same tool names in the same order, and the version initialize negotiates is listed, as is the server's.
    const listed = try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}");
    const tools = get(try std.json.parseFromSliceLeaky(Value, a, listed.body, .{}), &.{ "result", "tools" }).array.items;
    const names = try strings(a, get(caps, &.{ "mcp", "tools" }));
    try testing.expectEqual(tools.len, names.len);
    for (tools, names) |t, n| try testing.expectEqualStrings(t.object.get("name").?.string, n);
    const init = try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}");
    const result = get(try std.json.parseFromSliceLeaky(Value, a, init.body, .{}), &.{"result"});
    try testing.expect(has(try strings(a, get(caps, &.{ "mcp", "protocol_versions" })), get(result, &.{"protocolVersion"}).string));
    try testing.expectEqualStrings(get(caps, &.{ "server", "version" }).string, get(result, &.{ "serverInfo", "version" }).string);

    // Every listed path that takes a GET answers; what the box says it does not serve, it does not.
    for ([_][]const u8{ "/v1/models", "/v1/tools", "/v1/pair" }) |p| try testing.expectEqual(std.http.Status.ok, (try d.call(a, .GET, p, null)).status);
    for (get(caps, &.{"unsupported"}).array.items) |g| {
        const status = g.object.get("status").?.string;
        try testing.expect(std.mem.eql(u8, status, "coming") or std.mem.eql(u8, status, "unsupported"));
    }
    for ([_][]const u8{ "/v1/embeddings", "/v1/completions", "/v1/audio/translations" }) |p|
        try testing.expectEqual(std.http.Status.not_found, (try d.call(a, .POST, p, "{}")).status);

    // Responses is served, stateless: a request without input is the route's own 400, not a 404.
    try testing.expect(get(caps, &.{ "responses", "available" }).bool);
    try testing.expect(!get(caps, &.{ "responses", "store" }).bool);
    try testing.expectEqualStrings("/v1/responses", get(caps, &.{ "responses", "path" }).string);
    const no_input = try d.call(a, .POST, "/v1/responses", "{}");
    try testing.expectEqual(std.http.Status.bad_request, no_input.status);
    try testing.expectEqualStrings("input", get(try std.json.parseFromSliceLeaky(Value, a, no_input.body, .{}), &.{ "error", "param" }).string);
}

test "capabilities with pairing required: readable without a token, and each guarded route refuses one without it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, testing.io, "contract-paired", "{" ++ harness.reserve ++ ", \"pairing\": {\"required\": true}, \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 5}}]}");
    defer d.stop();
    const r = try d.call(a, .GET, "/v1/capabilities", null);
    try testing.expectEqual(std.http.Status.ok, r.status);
    const caps = try std.json.parseFromSliceLeaky(Value, a, r.body, .{});
    try testing.expect(get(caps, &.{ "auth", "pairing_required" }).bool);
    var scope_of_chat: []const u8 = "";
    var scope_of_responses: []const u8 = "";
    for (get(caps, &.{ "auth", "guarded" }).array.items) |g| {
        if (std.mem.eql(u8, g.object.get("route").?.string, "/v1/chat/completions")) scope_of_chat = g.object.get("scope").?.string;
        if (std.mem.eql(u8, g.object.get("route").?.string, "/v1/responses")) scope_of_responses = g.object.get("scope").?.string;
    }
    try testing.expectEqualStrings("inference", scope_of_chat);
    try testing.expectEqualStrings("inference", scope_of_responses);
    for ([_]struct { std.http.Method, []const u8 }{ .{ .POST, "/v1/chat/completions" }, .{ .POST, "/v1/responses" }, .{ .POST, "/v1/images/generations" }, .{ .POST, "/v1/systemone" }, .{ .GET, "/v1/models" }, .{ .POST, "/v1/tools/release" }, .{ .POST, "/mcp" } }) |c|
        try testing.expectEqual(std.http.Status.unauthorized, (try d.call(a, c[0], c[1], if (c[0] == .POST) "{}" else null)).status);
    for (get(caps, &.{ "auth", "open" }).array.items) |p| if (!std.mem.eql(u8, p.string, "/v1/pair"))
        try testing.expect((try d.call(a, .GET, p.string, null)).status != .unauthorized);
}

/// POST `body` to `path` as JSON with `x-forwarded-proto: proto` (none when empty), as a TLS proxy in front would.
fn postProxied(d: *Daemon, a: std.mem.Allocator, path: []const u8, body: []const u8, proto: []const u8) ![]u8 {
    var out: Io.Writer.Allocating = .init(a);
    const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}{s}", .{ d.port, path });
    const extra: []const std.http.Header = if (proto.len > 0) &.{.{ .name = "x-forwarded-proto", .value = proto }} else &.{};
    const res = try d.client.fetch(.{ .location = .{ .url = url }, .method = .POST, .payload = body, .response_writer = &out.writer, .keep_alive = false, .headers = .{ .content_type = .{ .override = "application/json" } }, .extra_headers = extra });
    try testing.expectEqual(std.http.Status.ok, res.status);
    return out.written();
}

test "output URLs keep the scheme the client used: https behind a TLS proxy (X-Forwarded-Proto), else http" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, testing.io, "contract-scheme", "{" ++ harness.reserve ++ ", \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 5}}]}");
    defer d.stop();
    const host = try std.fmt.allocPrint(a, "127.0.0.1:{d}/v1/files/", .{d.port});
    const image = "{\"prompt\":\"a fox\",\"size\":\"256x256\",\"response_format\":\"url\"}";
    const mcp = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"generate_image\",\"arguments\":{\"prompt\":\"a fox\",\"size\":\"256x256\",\"inline_images\":false}}}";
    for ([_][]const u8{ "https", "", "http" }) |proto| {
        const want = try std.fmt.allocPrint(a, "{s}://{s}", .{ if (std.mem.eql(u8, proto, "https")) "https" else "http", host });
        const http = try std.json.parseFromSliceLeaky(Value, a, try postProxied(&d, a, "/v1/images/generations", image, proto), .{});
        try testing.expect(std.mem.startsWith(u8, get(http, &.{"data"}).array.items[0].object.get("url").?.string, want));
        const tool = try postProxied(&d, a, "/mcp", mcp, proto);
        if (std.mem.indexOf(u8, tool, want) == null) {
            std.debug.print("{s}: no {s} in {s}\n", .{ proto, want, tool });
            return error.TestUnexpectedResult;
        }
    }
}

/// Error codes that mean a recorded event's shape is no longer accepted (not merely an id from the recorded session).
const shape_errors = [_][]const u8{ "unknown_event_type", "unknown_event", "invalid_event", "invalid_json", "invalid_session", "missing_session", "invalid_item", "missing_item", "invalid_audio", "internal_error" };

/// 40 ms of silence at 24 kHz (1920 zero bytes), base64: what a recorded `input_audio_buffer.append` replays as.
const silence: [2560]u8 = @splat('A');

/// The value with every `model`, `speech_model` and `voice` member dropped (recursively): this daemon has fakes.
fn dropNames(a: std.mem.Allocator, v: Value) !Value {
    switch (v) {
        .object => |o| {
            var out: std.json.ObjectMap = .empty;
            var it = o.iterator();
            while (it.next()) |e| {
                const k = e.key_ptr.*;
                if (std.mem.eql(u8, k, "model") or std.mem.eql(u8, k, "speech_model") or std.mem.eql(u8, k, "voice")) continue;
                try out.put(a, k, try dropNames(a, e.value_ptr.*));
            }
            return .{ .object = out };
        },
        .array => |l| {
            var out = std.json.Array.init(a);
            for (l.items) |x| try out.append(try dropNames(a, x));
            return .{ .array = out };
        },
        else => return v,
    }
}

const Line = struct { conn: []const u8, dir: []const u8, method: ?[]const u8 = null, path: ?[]const u8 = null, status: ?u16 = null, body: ?Value = null, ws: ?Value = null, repeat: u32 = 1 };

fn load(a: std.mem.Allocator, path: []const u8) ![]Line {
    const text = try Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(4 << 20));
    var out: std.ArrayList(Line) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| if (l.len > 0) try out.append(a, try std.json.parseFromSliceLeaky(Line, a, l, .{ .ignore_unknown_fields = true }));
    return out.items;
}

/// Paths whose recorded answer depended on that session's jobs and files.
fn sessionBound(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "/v1/jobs/") or std.mem.startsWith(u8, path, "/v1/files/") or std.mem.startsWith(u8, path, "/v1/videos/");
}

/// Replays one fixture's HTTP requests: each must land in the status class it had on the box.
fn replayHttp(d: *Daemon, a: std.mem.Allocator, lines: []const Line, token: []const u8) !usize {
    var n: usize = 0;
    for (lines, 0..) |l, i| {
        const method_name = l.method orelse continue;
        if (!std.mem.eql(u8, l.dir, "c2s")) continue;
        const path = l.path.?;
        if (std.mem.eql(u8, path, "/v1/realtime") or std.mem.eql(u8, path, "/v1/desktop") or sessionBound(path)) continue;
        const recorded = for (lines[i + 1 ..]) |x| {
            if (x.status != null and std.mem.eql(u8, x.conn, l.conn)) break x.status.?;
        } else continue;
        const method = std.meta.stringToEnum(std.http.Method, method_name).?;
        const body: ?[]const u8 = if (l.body) |b| try std.json.Stringify.valueAlloc(a, try dropNames(a, b), .{}) else if (method == .POST) "" else null;
        const target = if (std.mem.eql(u8, path, "/mcp") and token.len > 0) try std.fmt.allocPrint(a, "/mcp?api_key={s}", .{token}) else path;
        const r = try d.call(a, method, target, body);
        const got = @backingInt(r.status);
        if (got / 100 != recorded / 100) {
            std.debug.print("{s} {s}: the box answered {d}, this daemon {d}: {s}\n", .{ method_name, path, recorded, got, r.body[0..@min(r.body.len, 300)] });
            return error.TestUnexpectedResult;
        }
        n += 1;
    }
    return n;
}

/// Replays each recorded WebSocket connection to `route` on a fresh socket and fails on any event refused for its
/// shape (on /v1/desktop, any error frame); `first` is a server event that must arrive. The number of lines sent.
fn replayWs(d: *Daemon, a: std.mem.Allocator, lines: []const Line, route: []const u8, token: []const u8, first: []const u8) !usize {
    var n: usize = 0;
    var conns: std.ArrayList([]const u8) = .empty;
    for (lines) |l| if (l.method != null and std.mem.eql(u8, l.path.?, route) and !has(conns.items, l.conn)) try conns.append(a, l.conn);
    for (conns.items) |conn| {
        var w: Ws = undefined;
        const target = if (token.len > 0) try std.fmt.allocPrint(a, "{s}?api_key={s}", .{ route, token }) else route;
        try w.openAt(d.io, a, d.port, target, if (std.mem.eql(u8, route, "/v1/realtime")) "realtime" else "");
        defer w.close();
        try testing.expect(std.mem.indexOf(u8, w.head, " 101 ") != null);
        var sent: usize = 0;
        for (lines) |l| {
            if (!std.mem.eql(u8, l.conn, conn) or !std.mem.eql(u8, l.dir, "c2s")) continue;
            var ev = l.ws orelse continue;
            ev = try dropNames(a, ev);
            if (std.mem.eql(u8, ev.object.get("type").?.string, "input_audio_buffer.append")) try ev.object.put(a, "audio", .{ .string = &silence });
            const text = try std.json.Stringify.valueAlloc(a, ev, .{});
            for (0..l.repeat) |_| try w.send(text);
            sent += 1;
        }
        try testing.expect(sent > 0);
        // A marker the server answers with an error of its own: every refusal of a replayed event comes before it.
        const realtime = std.mem.eql(u8, route, "/v1/realtime");
        try w.send(if (realtime) "{\"type\":\"conversation.item.delete\",\"item_id\":\"contract_end\",\"event_id\":\"contract_end\"}" else "{\"type\":\"ping\"}");
        var saw_first = first.len == 0;
        while (true) {
            const v = try std.json.parseFromSliceLeaky(Value, a, try w.next(a), .{});
            const typ = v.object.get("type").?.string;
            if (std.mem.eql(u8, typ, first)) saw_first = true;
            if (!realtime and std.mem.eql(u8, typ, "pong")) break;
            if (!std.mem.eql(u8, typ, "error")) continue;
            const eid = get(v, &.{ "error", "event_id" }); // the client event it answers
            if (realtime and eid == .string and std.mem.eql(u8, eid.string, "contract_end")) break;
            const code = get(v, &.{ "error", "code" });
            if (!realtime or (code == .string and has(&shape_errors, code.string))) {
                std.debug.print("{s} connection {s}: refused: {s}\n", .{ route, conn, try std.json.Stringify.valueAlloc(a, v, .{}) });
                return error.TestUnexpectedResult;
            }
        }
        try testing.expect(saw_first);
        n += sent;
    }
    return n;
}

/// `localrouter pair token NAME --scopes SCOPES` on the daemon's data: the token.
fn mint(d: *Daemon, a: std.mem.Allocator, name: []const u8, scopes: []const u8) ![]const u8 {
    const cfg = try std.fmt.allocPrint(a, "{s}/config.json", .{d.dir});
    const data = try std.fmt.allocPrint(a, "{s}/data", .{d.dir});
    const r = try std.process.run(a, d.io, .{ .argv = &.{ opts.localrouter_exe, "pair", "token", name, "--scopes", scopes, "--config", cfg, "--data", data } });
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, r.term);
    return std.mem.trim(u8, r.stdout, " \r\n");
}

test "fixtures: what the web page, the mobile app, the desktop helper and Codex sent the box is still accepted" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, testing.io, "contract-replay", try config(a));
    defer d.stop();
    for ([_][]const u8{ "chat", "mouth", "ears" }) |id| for (0..240) |_| {
        if (std.mem.eql(u8, try d.toolState(a, id), "ready")) break;
        harness.sleepMs(d.io, 250);
    };

    const web = try load(a, opts.harness_web);
    const mobile = try load(a, opts.harness_mobile);
    const helper = try load(a, opts.harness_helper);
    const codex = try load(a, opts.harness_codex);
    try testing.expect(try replayHttp(&d, a, web, "") >= 5);
    try testing.expect(try replayWs(&d, a, web, "/v1/realtime", "", "") >= 5);
    try testing.expect(try replayHttp(&d, a, mobile, "") >= 2);
    // Codex CLI 0.164.0 (docs/harnesses/CODEX.md): two turns of streamed /v1/responses, each replaying the whole
    // conversation (reasoning, function calls and their outputs), its MCP tools in a namespace, and its MCP handshake.
    try testing.expect(try replayHttp(&d, a, codex, "") >= 8);
    var streams: usize = 0;
    for (codex) |l| if (l.body != null and l.path != null and std.mem.eql(u8, l.path.?, "/v1/responses")) {
        const r = try d.call(a, .POST, "/v1/responses", try std.json.Stringify.valueAlloc(a, try dropNames(a, l.body.?), .{}));
        if (std.mem.indexOf(u8, r.body, "event: response.completed\n") == null) {
            std.debug.print("a Codex request did not complete: {s}\n", .{r.body[@max(r.body.len, 400) - 400 ..]});
            return error.TestUnexpectedResult;
        }
        streams += 1;
    };
    try testing.expectEqual(@as(usize, 5), streams);
    try testing.expect(try replayWs(&d, a, mobile, "/v1/realtime", "", "") >= 3);

    // What the clients send on barge-in and push to talk, which the recorded runs did not do: the documented shapes,
    // each answered by its own refusal (nothing to truncate, delete, cancel or commit), never as an unknown or bad event.
    {
        var w: Ws = undefined;
        try w.openAt(d.io, a, d.port, "/v1/realtime", "realtime");
        defer w.close();
        const cases = [_]struct { event: []const u8, code: []const u8 }{
            .{ .event = "{\"type\":\"conversation.item.truncate\",\"item_id\":\"item_gone\",\"content_index\":0,\"audio_end_ms\":1200}", .code = "item_truncate_invalid_item_id" },
            .{ .event = "{\"type\":\"conversation.item.delete\",\"item_id\":\"item_gone\"}", .code = "item_delete_invalid_item_id" },
            .{ .event = "{\"type\":\"response.cancel\"}", .code = "response_cancel_not_active" },
            .{ .event = "{\"type\":\"input_audio_buffer.commit\"}", .code = "input_audio_buffer_commit_empty" },
        };
        // Every tool execution mode discovery lists is accepted and echoed (the last one, client, stays set).
        const caps = try std.json.parseFromSliceLeaky(Value, a, (try d.call(a, .GET, "/v1/capabilities", null)).body, .{});
        const modes = try strings(a, get(caps, &.{ "realtime", "tool_execution" }));
        try testing.expect(has(modes, "server") and has(modes, "client"));
        for ([_][]const u8{ "server", "client" }) |mode| {
            try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"session.update\",\"session\":{{\"localrouter\":{{\"tool_execution\":\"{s}\"}}}}}}", .{mode}));
            while (true) {
                const v = try std.json.parseFromSliceLeaky(Value, a, try w.next(a), .{});
                const typ = v.object.get("type").?.string;
                try testing.expect(!std.mem.eql(u8, typ, "error"));
                if (!std.mem.eql(u8, typ, "session.updated")) continue;
                try testing.expectEqualStrings(mode, get(v, &.{ "session", "localrouter", "tool_execution" }).string);
                break;
            }
        }
        try w.send("{\"type\":\"session.update\",\"session\":{\"turn_detection\":null}}");
        for (cases) |c| {
            try w.send(c.event);
            while (true) {
                const v = try std.json.parseFromSliceLeaky(Value, a, try w.next(a), .{});
                if (!std.mem.eql(u8, v.object.get("type").?.string, "error")) continue;
                try testing.expectEqualStrings(c.code, get(v, &.{ "error", "code" }).string);
                break;
            }
        }
    }

    // The helper connects with a desktop_helper token; the agent that drove it had a computer one.
    const helper_token = try mint(&d, a, "helper", "desktop_helper");
    const agent_token = try mint(&d, a, "agent", "computer");
    try testing.expect(try replayHttp(&d, a, helper, agent_token) >= 1);
    try testing.expect(try replayWs(&d, a, helper, "/v1/desktop", helper_token, "welcome") >= 3);
}
