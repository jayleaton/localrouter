//! Integration tests of /v1/responses (the Responses API over the chat path) on a real daemon whose chat model is the
//! fake TensorFold server (tests/fixtures/chat_server.py). The fixtures in tests/fixtures/responses/ are whole
//! conversations replayed as a stateless client (Codex) replays them: every turn sends the conversation so far, the
//! previous turn's output items exactly as they came back, and the new items; `$call_N` in an input stands for the
//! N-th call id the previous turn returned. Then the refusals (400 with `param` and `code`, before the model is asked),
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
    const v = try std.json.parseFromSliceLeaky(Value, a, r.body, .{});
    try testing.expectEqualStrings("response", str(get(v, &.{"object"})));
    try testing.expect(std.mem.startsWith(u8, str(get(v, &.{"id"})), "resp_"));
    try testing.expectEqualStrings("chat", str(get(v, &.{"model"})));
    try testing.expectEqual(false, get(v, &.{"store"}).bool);
    try testing.expect(get(v, &.{"previous_response_id"}) == .null);
    return v;
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
fn replay(d: *Daemon, a: std.mem.Allocator, fixture: []const u8) !void {
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
        const text = try std.json.Stringify.valueAlloc(a, Value{ .object = body }, .{});
        const r = try response(a, try post(d, a, text));
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

test "responses replay: a whole text conversation and a whole function-call conversation, stateless" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "responses-replay");
    defer d.stop();
    try replay(&d, a, @embedFile("fixtures/responses/text.json"));
    try replay(&d, a, @embedFile("fixtures/responses/function_call.json"));
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
    try refused(try post(&d, a, "{\"input\":\"hi\",\"tools\":[{\"type\":\"namespace\",\"name\":\"multi_agent_v1\",\"tools\":[]}]}"), .bad_request, "unsupported_tool", "tools", "namespace");
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
