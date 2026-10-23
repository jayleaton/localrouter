//! Integration tests of the decision kind on real daemons and workers: the compiled-in `openai_server` engine starts a
//! fake System One server (tests/fixtures/decide_server.py) and the daemon proxies /v1/systemone to it. Questions are
//! checked before anything loads, answers before anything is returned; System One model names reach the default
//! decision model; the decide lane; the MCP `decide` tool.

const std = @import("std");
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const reserve = harness.reserve;
const mcpCall = harness.mcpCall;
const Value = std.json.Value;

/// A decision tool (32 MiB) on the fixture server.
fn decideTool(a: std.mem.Allocator, id: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"id\": \"{s}\", \"kind\": \"decision\", \"engine\": \"openai_server\", \"resident_bytes\": 33554432, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"]}}}}", .{ id, opts.decide_server });
}

fn start(a: std.mem.Allocator, name: []const u8, tools: []const u8) !Daemon {
    const cfg = try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", \"tools\": [{s}]}}", .{tools});
    return Daemon.start(testing.allocator, testing.io, name, cfg);
}

fn expectStatus(r: Daemon.Resp, status: std.http.Status, needles: []const []const u8) !void {
    if (r.status != status) std.debug.print("expected {d}, got {d}: {s}\n", .{ @backingInt(status), @backingInt(r.status), r.body });
    try testing.expectEqual(status, r.status);
    for (needles) |n| if (std.mem.indexOf(u8, r.body, n) == null) {
        std.debug.print("missing '{s}' in {s}\n", .{ n, r.body });
        return error.TestExpectedEqual;
    };
}

/// A System One request: `model`, the state's text and the route question (image, video, reply), plus `more` questions.
fn route(a: std.mem.Allocator, model: []const u8, state: []const u8, more: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"state\":{{\"user_said\":\"{s}\"}},\"questions\":{{\"route\":{{\"type\":\"choice\",\"instructions\":\"Which action?\",\"criteria\":{{\"image\":\"make a picture\",\"video\":\"make a clip\",\"reply\":null}}}}{s}}}}}", .{ model, state, more });
}

fn toolState(d: *Daemon, a: std.mem.Allocator, id: []const u8) ![]const u8 {
    const T = struct { data: []const struct { id: []const u8, state: []const u8, lane: []const u8 } };
    const listed = try std.json.parseFromSliceLeaky(T, a, (try d.call(a, .GET, "/v1/tools", null)).body, .{ .ignore_unknown_fields = true });
    for (listed.data) |t| if (std.mem.eql(u8, t.id, id)) return t.state;
    return error.NoSuchTool;
}

test "decisions: checked questions, checked answers, System One model names, the decide lane" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "decide", try decideTool(a, "jev-local"));
    defer d.stop();

    // Refused before anything loads: a one-option choice, a box question, no state, a request over 16 KB.
    try expectStatus(try d.call(a, .POST, "/v1/systemone", "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{\"only\":null}}}}"), .bad_request, &.{"a choice needs criteria"});
    try expectStatus(try d.call(a, .POST, "/v1/systemone", "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"box\"}}}"), .bad_request, &.{"choice, noul or score"});
    try expectStatus(try d.call(a, .POST, "/v1/systemone", "{\"questions\":{\"q\":{\"type\":\"noul\"}}}"), .bad_request, &.{"state is required"});
    const long = try a.alloc(u8, 17 * 1024);
    @memset(long, 'x');
    try expectStatus(try d.call(a, .POST, "/v1/systemone", try route(a, "jev-local", long, "")), .bad_request, &.{"larger than 16 KB"});
    try testing.expectEqualStrings("unloaded", try toolState(&d, a, "jev-local"));

    // A choice, a noul and a score; the model name a System One client sends (jev-latest) reaches the default model.
    const more = ",\"two\":{\"type\":\"noul\",\"instructions\":\"Two requests?\"},\"urgency\":{\"type\":\"score\",\"criteria\":[\"low\",\"mid\",\"high\"]}";
    const ok = try d.call(a, .POST, "/v1/systemone", try route(a, "jev-latest", "draw an image of a fox, yes", more));
    try expectStatus(ok, .ok, &.{});
    const v = (try std.json.parseFromSliceLeaky(Value, a, ok.body, .{})).object;
    try testing.expectEqualStrings("jev-local", v.get("model").?.string);
    const ans = v.get("answers").?.object;
    try testing.expectEqualStrings("image", ans.get("route").?.object.get("choice").?.string);
    try testing.expectApproxEqAbs(@as(f64, 0.8), ans.get("two").?.object.get("noul").?.float, 1e-9);
    try testing.expectEqualStrings("score", ans.get("urgency").?.object.get("type").?.string);
    try testing.expectEqualStrings("ready", try toolState(&d, a, "jev-local"));

    // Answers that do not fit the questions are the model's error (500), never a decision.
    try expectStatus(try d.call(a, .POST, "/v1/systemone", try route(a, "", "offbook", "")), .internal_server_error, &.{"not one of the offered options"});
    try expectStatus(try d.call(a, .POST, "/v1/systemone", try route(a, "", "badsum", "")), .internal_server_error, &.{"do not sum to 1"});
    try expectStatus(try d.call(a, .POST, "/v1/systemone", try route(a, "", "skip", "")), .internal_server_error, &.{"did not answer 'route'"});
    try expectStatus(try d.call(a, .POST, "/v1/systemone", try route(a, "", "overload", "")), .service_unavailable, &.{"busy"});
    try testing.expectEqualStrings("ready", try toolState(&d, a, "jev-local")); // still warm after the refusals

    // The MCP tool: one line a question, then the reply.
    const r = try mcpCall(&d, a, "decide", "{\"state\":\"make a video of waves\",\"questions\":{\"route\":{\"type\":\"choice\",\"criteria\":{\"image\":\"a picture\",\"video\":\"a clip\"}}}}");
    try testing.expect(!r.object.get("isError").?.bool);
    const line = r.object.get("content").?.array.items[0].object.get("text").?.string;
    try testing.expect(std.mem.startsWith(u8, line, "route: video (0.900)"));
    const bad = try mcpCall(&d, a, "decide", "{\"state\":\"s\",\"questions\":{}}");
    try testing.expect(bad.object.get("isError").?.bool);
}

test "decisions without a decision model: 404 naming none" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "nodecide", "{\"id\": \"img\", \"kind\": \"image\", \"engine\": \"testpattern\"}");
    defer d.stop();
    try expectStatus(try d.call(a, .POST, "/v1/systemone", try route(a, "jev-latest", "hi", "")), .not_found, &.{"no decision model is configured"});
}
