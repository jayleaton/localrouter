//! A multi-node topology on two real daemons (docs/dev/MULTI-SPARK.md): node `s1` runs the chat model (the fake
//! TensorFold server, tests/fixtures/chat_server.py) and the testpattern image model; the front `s2` runs the
//! testpattern speech model, serves every model through one API, and falls back to running the chat model itself
//! while `s1` is down.

const std = @import("std");
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const reserve = harness.reserve;
const sleepMs = harness.sleepMs;

fn tools(a: std.mem.Allocator) ![]const u8 {
    return std.fmt.allocPrint(a,
        \\"tools": [{{"id": "chat", "kind": "chat", "engine": "openai_server", "resident_bytes": 41943040, "keep_loaded": true,
        \\            "options": {{"server": ["python3", "{s}", "--port", "{{port}}", "--name", "{{id}}"]}}}},
        \\          {{"id": "img", "kind": "image", "engine": "testpattern"}},
        \\          {{"id": "tts", "kind": "speech", "engine": "testpattern", "idle_ttl_s": 0}}]
    , .{opts.chat_server});
}

fn topo(a: std.mem.Allocator, self: []const u8, node_port: u16) ![]const u8 {
    return std.fmt.allocPrint(a,
        \\"topology": {{"mode": "multi", "self": "{s}", "health_interval_ms": 150, "down_after": 2, "nodes": [
        \\  {{"id": "s1", "addresses": ["127.0.0.1"], "port": {d}, "models": ["chat", "img"]}},
        \\  {{"id": "s2", "addresses": ["127.0.0.1"], "port": 9, "role": "front", "models": ["tts"], "fallback": ["chat"]}}]}}
    , .{ self, node_port });
}

fn ask(d: *Daemon, a: std.mem.Allocator, prompt: []const u8, stream: bool) !Daemon.Resp {
    return d.call(a, .POST, "/v1/chat/completions", try std.fmt.allocPrint(a, "{{\"model\":\"chat\",\"stream\":{},\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]}}", .{ stream, prompt }));
}

const Listed = struct { data: []const struct { id: []const u8, node: ?[]const u8 = null, local: bool = true, state: []const u8 = "" } };

fn entry(l: Listed, id: []const u8) ?@TypeOf(l.data[0]) {
    for (l.data) |e| if (std.mem.eql(u8, e.id, id)) return e;
    return null;
}

fn listed(d: *Daemon, a: std.mem.Allocator, path: []const u8) !Listed {
    const r = try d.call(a, .GET, path, null);
    try testing.expectEqual(std.http.Status.ok, r.status);
    return std.json.parseFromSliceLeaky(Listed, a, r.body, .{ .ignore_unknown_fields = true });
}

test "multi-node: one front serves every node's models; a node admits its own; chat falls back to the front while its node is down" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try tools(a);

    // a node: its own tools only, and the front's proof that it is up
    var n1 = try Daemon.start(testing.allocator, testing.io, "node-s1", try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", {s}, {s}}}", .{ t, try topo(a, "s1", 9) }));
    var n1_alive = true;
    defer if (n1_alive) n1.stop();
    const own = try listed(&n1, a, "/v1/models");
    try testing.expectEqual(@as(usize, 2), own.data.len); // chat and img: not the front's tts
    try testing.expectEqualStrings("s1", entry(own, "chat").?.node.?);
    const status = try n1.call(a, .GET, "/v1/node/status", null);
    try testing.expectEqual(std.http.Status.ok, status.status);

    var front = try Daemon.start(testing.allocator, testing.io, "node-s2", try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", {s}, {s}}}", .{ t, try topo(a, "s2", n1.port) }));
    defer front.stop();

    // one surface: every model listed by the front, each with the node that runs it
    const all = try listed(&front, a, "/v1/models");
    try testing.expectEqual(@as(usize, 3), all.data.len);
    try testing.expectEqualStrings("s1", entry(all, "chat").?.node.?);
    try testing.expectEqualStrings("s1", entry(all, "img").?.node.?);
    try testing.expectEqualStrings("s2", entry(all, "tts").?.node.?);

    // chat goes to s1, whole and streamed; the front runs no chat server of its own
    const whole = try ask(&front, a, "hello", false);
    try testing.expectEqual(std.http.Status.ok, whole.status);
    try testing.expect(std.mem.indexOf(u8, whole.body, "echo: hello") != null);
    const streamed = try ask(&front, a, "again", true);
    try testing.expectEqual(std.http.Status.ok, streamed.status);
    try testing.expect(std.mem.indexOf(u8, streamed.body, "data: [DONE]") != null);
    try testing.expectEqual(@as(usize, 0), (try front.workers(a)).len);
    const remote_chat = entry(try listed(&front, a, "/v1/tools"), "chat").?;
    try testing.expect(!remote_chat.local);
    try testing.expectEqualStrings("ready", try n1.toolState(a, "chat")); // s1 loaded it, against its own memory

    // an image job runs on s1 and its file comes back through the front, labelled as machine-generated
    const img = try harness.image(&front, a, "img", ",\"seed\":7");
    try testing.expectEqual(std.http.Status.ok, img.status);
    const png = try harness.firstPng(a, img.body);
    try testing.expectEqualSlices(u8, "\x89PNG\r\n\x1a\n", png[0..8]);
    try testing.expectEqualSlices(u8, png, try harness.firstPng(a, (try harness.image(&n1, a, "img", ",\"seed\":7")).body)); // the same bytes as asked of s1 itself
    try testing.expectEqual(std.http.Status.bad_request, (try front.call(a, .POST, "/v1/images/generations", "{\"model\":\"img\",\"prompt\":\"x\",\"size\":\"1000x1000\"}")).status); // s1's refusal, kept

    // the front's own speech model runs on the front
    const sp = try front.call(a, .POST, "/v1/audio/speech", "{\"model\":\"tts\",\"input\":\"hi\",\"response_format\":\"wav\"}");
    try testing.expectEqual(std.http.Status.ok, sp.status);
    try testing.expectEqualSlices(u8, "RIFF", sp.body[0..4]);

    // s1 goes down: chat comes back on the front (fallback), image is refused naming the node
    n1.stop();
    n1_alive = false;
    var answered = false;
    for (0..80) |_| {
        const r = try ask(&front, a, "still there", false);
        if (r.status == .ok) {
            try testing.expect(std.mem.indexOf(u8, r.body, "echo: still there") != null);
            answered = true;
            break;
        }
        try testing.expectEqual(std.http.Status.service_unavailable, r.status);
        sleepMs(testing.io, 100);
    }
    try testing.expect(answered);
    try testing.expect(entry(try listed(&front, a, "/v1/tools"), "chat").?.local);
    try testing.expectEqual(@as(usize, 1), (try front.workers(a)).len); // the chat server, here now
    const down = try harness.image(&front, a, "img", "");
    try testing.expectEqual(std.http.Status.service_unavailable, down.status);
    try testing.expect(std.mem.indexOf(u8, down.body, "s1") != null);
}

test "multi-node: an invalid topology stops the daemon at start with the reason" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", {s}, {s}}}", .{ try tools(a), try topo(a, "s9", 9) });
    const dir = "/tmp/localrouter-it-badtopo";
    try std.Io.Dir.cwd().createDirPath(testing.io, dir);
    defer std.Io.Dir.cwd().deleteTree(testing.io, dir) catch {};
    try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = dir ++ "/config.json", .data = cfg });
    const r = try std.process.run(a, testing.io, .{ .argv = &.{ opts.localrouter_exe, "serve", "--config", dir ++ "/config.json", "--port", "0", "--data", dir ++ "/data" } });
    try testing.expectEqual(@as(u8, 2), r.term.exited);
    try testing.expect(std.mem.indexOf(u8, r.stderr, "node 's9', which is not one of its nodes (s1, s2)") != null);
}
