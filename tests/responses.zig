//! Integration tests of /v1/responses (the Responses API over the chat path) on a real daemon whose chat model is the
//! fake TensorFold server (tests/fixtures/chat_server.py). The fixtures in tests/fixtures/responses/ are whole
//! conversations replayed as a stateless client (Codex) replays them: every turn sends the conversation so far, the
//! previous turn's output items exactly as they came back, and the new items; `$call_N` in an input stands for the
//! N-th call id the previous turn returned. Each is played whole and streamed; a stream is checked as a client reads
//! it (`streamed`): the event order, sequence numbers, deltas that join to each item's final text and arguments, and
//! the `.done` items equal to the final response's output. Then the refusals (400 with `param` and `code`, before the model is asked),
//! the model's own errors, and the routes that would need stored responses.

const std = @import("std");
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;

fn start(a: std.mem.Allocator, name: []const u8) !Daemon {
    const cfg = try std.fmt.allocPrint(a, "{{" ++ harness.reserve ++ ", \"tools\": [{{\"id\": \"chat\", \"kind\": \"chat\", \"engine\": \"openai_server\", \"resident_bytes\": 41943040, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"]}}}}]}}", .{opts.chat_server});
    return Daemon.start(testing.allocator, testing.io, name, cfg);
}

fn get(v: Value, path: []const []const u8) Value {
    var x = v;
    for (path) |k| x = if (x == .object) x.object.get(k) orelse .null else .null;
    return x;
}

fn str(v: Value) []const u8 {
    return if (v == .string) v.string else "";
}

fn post(d: *Daemon, a: std.mem.Allocator, body: []const u8) !Daemon.Resp {
    return d.call(a, .POST, "/v1/responses", body);
}

/// A 200 response object, checked for the fields every response has.
fn response(a: std.mem.Allocator, r: Daemon.Resp) !Value {
    if (r.status != .ok) std.debug.print("expected 200, got {d}: {s}\n", .{ @backingInt(r.status), r.body });
    try testing.expectEqual(std.http.Status.ok, r.status);
    return checked(try std.json.parseFromSliceLeaky(Value, a, r.body, .{}));
}

fn checked(v: Value) !Value {
    try testing.expectEqualStrings("response", str(get(v, &.{"object"})));
    try testing.expect(std.mem.startsWith(u8, str(get(v, &.{"id"})), "resp_"));
    try testing.expectEqualStrings("chat", str(get(v, &.{"model"})));
    try testing.expectEqual(false, get(v, &.{"store"}).bool);
    try testing.expect(get(v, &.{"previous_response_id"}) == .null);
    return v;
}

/// The events of a 200 SSE body: `event: T` then `data: {...}` with the same `type`, `sequence_number` 0, 1, 2...
fn events(a: std.mem.Allocator, r: Daemon.Resp) ![]const Value {
    if (r.status != .ok) std.debug.print("expected 200, got {d}: {s}\n", .{ @backingInt(r.status), r.body });
    try testing.expectEqual(std.http.Status.ok, r.status);
    var out: std.ArrayList(Value) = .empty;
    var blocks = std.mem.splitSequence(u8, r.body, "\n\n");
    while (blocks.next()) |b| {
        if (b.len == 0) continue;
        if (!std.mem.startsWith(u8, b, "event: ")) {
            std.debug.print("not an event: '{s}'\n", .{b});
            return error.TestExpectedEqual;
        }
        const nl = std.mem.indexOfScalar(u8, b, '\n') orelse return error.TestExpectedEqual;
        try testing.expect(std.mem.startsWith(u8, b[nl + 1 ..], "data: "));
        const v = try std.json.parseFromSliceLeaky(Value, a, b[nl + 1 + "data: ".len ..], .{});
        try testing.expectEqualStrings(b["event: ".len..nl], str(get(v, &.{"type"})));
        try testing.expectEqual(@as(i64, @intCast(out.items.len)), get(v, &.{"sequence_number"}).integer);
        try out.append(a, v);
    }
    return out.items;
}

fn isType(e: Value, t: []const u8) bool {
    return std.mem.eql(u8, str(get(e, &.{"type"})), t);
}

/// A streamed response read as a client reads it, checked, and its final response object (`response.completed`,
/// `.incomplete` or `.failed`).
fn streamed(a: std.mem.Allocator, r: Daemon.Resp) !Value {
    const evs = try events(a, r);
    try testing.expect(evs.len >= 3);
    try testing.expect(isType(evs[0], "response.created") and isType(evs[1], "response.in_progress"));
    try testing.expectEqualStrings("in_progress", str(get(evs[0], &.{ "response", "status" })));
    const last = evs[evs.len - 1];
    try testing.expect(isType(last, "response.completed") or isType(last, "response.incomplete") or isType(last, "response.failed"));
    const final = get(last, &.{"response"});
    try testing.expectEqualStrings(str(get(evs[0], &.{ "response", "id" })), str(get(final, &.{"id"})));
    const output = get(final, &.{"output"}).array.items;
    // Per output_index: added, then its deltas (joined), then its done events; done items are the final output.
    var done_items: usize = 0;
    var text: std.ArrayList(u8) = .empty;
    var open: ?usize = null;
    for (evs[2 .. evs.len - 1]) |e| {
        const index: usize = @intCast(get(e, &.{"output_index"}).integer);
        if (isType(e, "response.output_item.added")) {
            try testing.expect(open == null); // one item at a time, in order
            try testing.expectEqual(done_items, index);
            open = index;
            text.clearRetainingCapacity();
            continue;
        }
        try testing.expectEqual(open, index);
        if (isType(e, "response.output_text.delta") or isType(e, "response.reasoning_text.delta") or isType(e, "response.function_call_arguments.delta")) {
            try text.appendSlice(a, str(get(e, &.{"delta"})));
        } else if (isType(e, "response.output_text.done") or isType(e, "response.reasoning_text.done")) {
            try testing.expectEqualStrings(text.items, str(get(e, &.{"text"})));
        } else if (isType(e, "response.function_call_arguments.done")) {
            try testing.expectEqualStrings(text.items, str(get(e, &.{"arguments"})));
        } else if (isType(e, "response.output_item.done")) {
            const want = try std.json.Stringify.valueAlloc(a, output[index], .{});
            try testing.expectEqualStrings(want, try std.json.Stringify.valueAlloc(a, get(e, &.{"item"}), .{}));
            done_items += 1;
            open = null;
        } else try testing.expect(isType(e, "response.content_part.added") or isType(e, "response.content_part.done"));
    }
    try testing.expectEqual(output.len, done_items);
    if (isType(last, "response.failed")) return final;
    return checked(final);
}

/// A refusal: `status`, OpenAI's envelope with `code` and `param`, and `needle` in its message.
fn refused(r: Daemon.Resp, status: std.http.Status, code: ?[]const u8, param: ?[]const u8, needle: []const u8) !void {
    if (r.status != status or std.mem.indexOf(u8, r.body, needle) == null) std.debug.print("expected {d} with '{s}', got {d}: {s}\n", .{ @backingInt(status), needle, @backingInt(r.status), r.body });
    try testing.expectEqual(status, r.status);
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(Value, arena.allocator(), r.body, .{});
    if (code) |c| try testing.expectEqualStrings(c, str(get(v, &.{ "error", "code" })));
    if (param) |p| try testing.expectEqualStrings(p, str(get(v, &.{ "error", "param" })));
    try testing.expect(std.mem.indexOf(u8, str(get(v, &.{ "error", "message" })), needle) != null);
}

/// `v` with every string "$call_N" replaced by the N-th call id in `calls`.
fn bind(a: std.mem.Allocator, v: Value, calls: []const []const u8) !Value {
    switch (v) {
        .string => |s| if (std.mem.startsWith(u8, s, "$call_")) {
            const n = try std.fmt.parseInt(usize, s["$call_".len..], 10);
            return .{ .string = calls[n - 1] };
        },
        .array => |arr| {
            var out = std.json.Array.init(a);
            for (arr.items) |x| try out.append(try bind(a, x, calls));
            return .{ .array = out };
        },
        .object => |o| {
            var out: ObjectMap = .empty;
            var it = o.iterator();
            while (it.next()) |e| try out.put(a, e.key_ptr.*, try bind(a, e.value_ptr.*, calls));
            return .{ .object = out };
        },
        else => {},
    }
    return v;
}

/// Plays a fixture's turns against the daemon and checks each response against the turn's `expect`.
fn replay(d: *Daemon, a: std.mem.Allocator, fixture: []const u8, stream: bool) !void {
    const f = try std.json.parseFromSliceLeaky(Value, a, fixture, .{});
    var history = std.json.Array.init(a);
    var calls: std.ArrayList([]const u8) = .empty;
    var item_ids: std.ArrayList([]const u8) = .empty;
    for (get(f, &.{"turns"}).array.items, 0..) |turn, t| {
        for (get(turn, &.{"input"}).array.items) |item| try history.append(try bind(a, item, calls.items));
        var body: ObjectMap = .empty;
        var base = get(f, &.{"request"}).object.iterator();
        while (base.next()) |e| try body.put(a, e.key_ptr.*, e.value_ptr.*);
        if (get(turn, &.{"request"}) == .object) {
            var more = get(turn, &.{"request"}).object.iterator();
            while (more.next()) |e| try body.put(a, e.key_ptr.*, e.value_ptr.*);
        }
        try body.put(a, "input", .{ .array = history });
        if (stream) try body.put(a, "stream", .{ .bool = true });
        const text = try std.json.Stringify.valueAlloc(a, Value{ .object = body }, .{});
        const r = if (stream) try streamed(a, try post(d, a, text)) else try response(a, try post(d, a, text));
        const want = get(turn, &.{"expect"});
        errdefer std.debug.print("turn {d}: {s}\n", .{ t, std.json.Stringify.valueAlloc(a, r, .{}) catch "" });

        try testing.expectEqualStrings(str(get(want, &.{"status"})), str(get(r, &.{"status"})));
        if (get(want, &.{"incomplete"}) == .string) try testing.expectEqualStrings(str(get(want, &.{"incomplete"})), str(get(r, &.{ "incomplete_details", "reason" })));
        try testing.expectEqual(get(want, &.{"input_tokens"}).integer, get(r, &.{ "usage", "input_tokens" }).integer);
        const output = get(r, &.{"output"}).array.items;
        const types = get(want, &.{"types"}).array.items;
        try testing.expectEqual(types.len, output.len);
        calls.clearRetainingCapacity();
        var call_n: usize = 0;
        for (output, types) |item, typ| {
            try testing.expectEqualStrings(typ.string, str(get(item, &.{"type"})));
            const id = str(get(item, &.{"id"}));
            for (item_ids.items) |seen| try testing.expect(!std.mem.eql(u8, seen, id)); // item ids never repeat
            try item_ids.append(a, id);
            if (std.mem.eql(u8, typ.string, "reasoning")) {
                try testing.expectEqualStrings("reasoning_text", str(get(item.object.get("content").?.array.items[0], &.{"type"})));
                if (get(want, &.{"reasoning"}) == .string) try testing.expectEqualStrings(str(get(want, &.{"reasoning"})), str(get(item.object.get("content").?.array.items[0], &.{"text"})));
            } else if (std.mem.eql(u8, typ.string, "message")) {
                try testing.expectEqualStrings("assistant", str(get(item, &.{"role"})));
                const part = item.object.get("content").?.array.items[0];
                try testing.expectEqualStrings("output_text", str(get(part, &.{"type"})));
                try testing.expectEqualStrings(str(get(want, &.{"text"})), str(get(part, &.{"text"})));
            } else {
                const c = get(want, &.{"calls"}).array.items[call_n];
                call_n += 1;
                inline for (.{ "call_id", "name", "arguments" }) |k| try testing.expectEqualStrings(str(get(c, &.{k})), str(get(item, &.{k})));
                try testing.expect(std.mem.startsWith(u8, id, "fc_"));
                try calls.append(a, str(get(item, &.{"call_id"})));
            }
            try history.append(item); // the next turn replays it as it came back
        }
    }
}

test "responses replay: a whole text conversation and a whole function-call conversation, stateless, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "responses-replay");
    defer d.stop();
    for ([_]bool{ false, true }) |stream| {
        try replay(&d, a, @embedFile("fixtures/responses/text.json"), stream);
        try replay(&d, a, @embedFile("fixtures/responses/function_call.json"), stream);
    }
}

test "responses streaming: refused before it began is plain JSON; broken mid-stream is response.failed; a hang-up cancels" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "responses-stream");
    defer d.stop();

    // The model refused before streaming anything: the ordinary JSON error, not a stream.
    const busy = try post(&d, a, "{\"input\":\"overload\",\"stream\":true}");
    try refused(busy, .service_unavailable, null, null, "busy");
    try testing.expect(busy.body[0] == '{');

    // The server dies after two words: what came is closed off, and the response failed with the reason.
    const broke = try streamed(a, try post(&d, a, "{\"input\":\"midfail\",\"stream\":true,\"reasoning\":{\"effort\":\"none\"}}"));
    try testing.expectEqualStrings("failed", str(get(broke, &.{"status"})));
    try testing.expect(str(get(broke, &.{ "error", "message" })).len > 0);
    try testing.expectEqualStrings("w0 w1", str(get(get(broke, &.{"output"}).array.items[0].object.get("content").?.array.items[0], &.{"text"})));

    // A client that hangs up mid-stream cancels the request; the model stays loaded with the same worker.
    for (0..80) |_| {
        if (std.mem.eql(u8, try d.toolState(a, "chat"), "ready")) break;
        harness.sleepMs(d.io, 250);
    }
    const pids = try d.workers(a);
    {
        // "slow" streams 40 words 0.25 s apart (about 11 s); read two text deltas, then close the connection.
        const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/responses", .{d.port});
        var req = try d.client.request(.POST, try std.Uri.parse(url), .{ .keep_alive = false, .headers = .{ .content_type = .{ .override = "application/json" } } });
        defer req.deinit();
        const body = "{\"input\":\"slow\",\"stream\":true,\"reasoning\":{\"effort\":\"none\"}}";
        req.transfer_encoding = .{ .content_length = body.len };
        try req.sendBodyComplete(@constCast(body));
        var redirect: [0]u8 = undefined;
        var res = try req.receiveHead(&redirect);
        try testing.expectEqual(std.http.Status.ok, res.head.status);
        var buf: [4096]u8 = undefined;
        const r = res.reader(&buf);
        var seen: usize = 0;
        while (seen < 2) {
            const line = try r.takeDelimiterInclusive('\n');
            if (std.mem.startsWith(u8, line, "event: response.output_text.delta")) seen += 1;
        }
    }
    var ready = false;
    for (0..20) |_| { // back to ready well before the stream would have ended
        if (std.mem.eql(u8, try d.toolState(a, "chat"), "ready")) {
            ready = true;
            break;
        }
        harness.sleepMs(d.io, 250);
    }
    try testing.expect(ready);
    const after = try d.workers(a);
    try testing.expectEqual(pids.len, after.len);
    try testing.expectEqual(pids[0], after[0]);
    _ = try response(a, try post(&d, a, "{\"input\":\"hi\"}"));
}

test "responses refusals: stored state, hosted tools, other items and parameters are 400s naming them; model errors map" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "responses-refusals");
    defer d.stop();

    try refused(try post(&d, a, "{\"input\":\"hi\",\"previous_response_id\":\"resp_1\"}"), .bad_request, "unsupported_parameter", "previous_response_id", "not stored");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"store\":true}"), .bad_request, "unsupported_value", "store", "store false");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"tools\":[{\"type\":\"web_search\",\"external_web_access\":false}]}"), .bad_request, "unsupported_tool", "tools", "hosted tools");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"tools\":[{\"type\":\"custom\",\"name\":\"apply_patch\"}]}"), .bad_request, "unsupported_tool", "tools", "custom");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"file_search\"}]}]}"), .bad_request, "unsupported_tool", "tools", "function tools only");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"background\":true}"), .bad_request, "unsupported_value", "background", "background");
    try refused(try post(&d, a, "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"https://x/y.png\"}]}]}"), .bad_request, "unsupported_value", "input", "text only");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"logit_bias\":{}}"), .bad_request, "unknown_parameter", "logit_bias", "logit_bias");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"reasoning\":{\"summary\":\"concise\"}}"), .bad_request, "unsupported_value", "reasoning.summary", "summaries");
    try refused(try post(&d, a, "[1]"), .bad_request, "invalid_request", null, "Responses request");
    try refused(try post(&d, a, "{\"input\":\"hi\",\"model\":\"gpt-5\"}"), .not_found, null, null, "chat");

    // The model's server judges what is left: its refusal is a 400 with its message, its failures keep their status.
    try refused(try post(&d, a, "{\"input\":\"overload\"}"), .service_unavailable, null, null, "busy");
    try refused(try post(&d, a, "{\"input\":\"explode\"}"), .internal_server_error, null, null, "generation failed");

    // Nothing is stored, so there is nothing to fetch, delete or cancel.
    try refused(try d.call(a, .GET, "/v1/responses/resp_1", null), .not_found, "not_found", "response_id", "not stored");
    try refused(try d.call(a, .POST, "/v1/responses/resp_1/cancel", ""), .not_found, "not_found", "response_id", "not stored");
}

test "responses: string input, instructions, json format, tool_choice none, and the echoed request fields" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "responses-fields");
    defer d.stop();

    const plain = try response(a, try post(&d, a, "{\"input\":\"hi\",\"instructions\":\"be kind\",\"temperature\":0,\"max_output_tokens\":50,\"metadata\":{\"k\":\"v\"},\"reasoning\":{\"effort\":\"none\"}}"));
    try testing.expectEqualStrings("completed", str(get(plain, &.{"status"})));
    try testing.expectEqualStrings("be kind", str(get(plain, &.{"instructions"})));
    try testing.expectEqualStrings("none", str(get(plain, &.{ "reasoning", "effort" })));
    try testing.expectEqualStrings("v", str(get(plain, &.{ "metadata", "k" })));
    try testing.expectEqual(@as(i64, 50), get(plain, &.{"max_output_tokens"}).integer);
    try testing.expectEqualStrings("text", str(get(plain, &.{ "text", "format", "type" })));
    const out = get(plain, &.{"output"}).array.items;
    try testing.expectEqual(@as(usize, 1), out.len); // reasoning off: no reasoning item
    try testing.expectEqualStrings("echo: hi", str(get(out[0].object.get("content").?.array.items[0], &.{"text"})));
    try testing.expectEqual(@as(i64, 2), get(plain, &.{ "usage", "input_tokens" }).integer); // instructions + input

    const js = try response(a, try post(&d, a, "{\"input\":\"hi\",\"text\":{\"format\":{\"type\":\"json_schema\",\"name\":\"r\",\"schema\":{\"type\":\"object\"}}}}"));
    const msg = get(js, &.{"output"}).array.items[1];
    try testing.expectEqualStrings("{\"echo\": \"hi\"}", str(get(msg.object.get("content").?.array.items[0], &.{"text"})));
    try testing.expectEqualStrings("json_schema", str(get(js, &.{ "text", "format", "type" })));

    const tools = "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]";
    const none = try response(a, try post(&d, a, "{\"input\":\"weather\"," ++ tools ++ ",\"tool_choice\":\"none\"}"));
    try testing.expectEqualStrings("message", str(get(none.object.get("output").?.array.items[1], &.{"type"})));
    try testing.expectEqualStrings("none", str(get(none, &.{"tool_choice"})));
    try testing.expectEqualStrings("get_weather", str(get(none.object.get("tools").?.array.items[0], &.{"name"})));
}
