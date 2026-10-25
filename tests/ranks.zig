//! Tensor-parallel groups on real daemons (docs/dev/MULTI-SPARK.md#tensor-parallel): the fake TensorFold server
//! (tests/fixtures/chat_server.py) keeps the ranks' contract (the other ranks join rank 0 before it serves, and a rank
//! that loses its peer stops), told its rank by the tool's `options.rank_env` as a real engine is. Three nodes: the front
//! s1, and s2 and s3. `big` is split over s2 and s3 (TP=2; the front sends it to rank 0 on s2); `wide` over all three
//! (TP=3, the front its rank 0).

const std = @import("std");
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const reserve = harness.reserve;
const sleepMs = harness.sleepMs;

fn tools(a: std.mem.Allocator, big_port: u16, wide_port: u16) ![]const u8 {
    const one =
        \\{{"id": "{s}", "kind": "chat", "engine": "openai_server", "resident_bytes": 41943040,
        \\  "options": {{"server": ["python3", "{s}", "--port", "{{port}}", "--name", "{{id}}"], "rank_start_ms": 300,
        \\              "rank_env": {{"FAKE_RANK": "{{rank}}", "FAKE_WORLD": "{{world}}", "FAKE_MASTER": "{{master}}:{d}"}}}}}}
    ;
    return std.fmt.allocPrint(a, "\"tools\": [" ++ one ++ ", " ++ one ++ "]", .{ "big", opts.chat_server, big_port, "wide", opts.chat_server, wide_port });
}

fn topo(a: std.mem.Allocator, self: []const u8, ports: [3]u16) ![]const u8 {
    return std.fmt.allocPrint(a,
        \\"topology": {{"mode": "multi", "self": "{s}", "health_interval_ms": 200, "down_after": 2, "nodes": [
        \\  {{"id": "s1", "addresses": ["127.0.0.1"], "port": {d}, "role": "front", "models": ["wide"]}},
        \\  {{"id": "s2", "addresses": ["127.0.0.1"], "port": {d}, "models": ["big", "wide"]}},
        \\  {{"id": "s3", "addresses": ["127.0.0.1"], "port": {d}, "models": ["big", "wide"]}}]}}
    , .{ self, ports[0], ports[1], ports[2] });
}

fn ask(d: *Daemon, a: std.mem.Allocator, model: []const u8) !Daemon.Resp {
    return d.call(a, .POST, "/v1/chat/completions", try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}", .{model}));
}

fn start(a: std.mem.Allocator, name: []const u8, t: []const u8, self: []const u8, ports: [3]u16) !Daemon {
    return Daemon.start(testing.allocator, testing.io, name, try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", {s}, {s}}}", .{ t, try topo(a, self, ports) }));
}

test "tensor parallel: TP=2 and TP=3 groups start their ranks on their nodes, serve from rank 0, and stop together" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rnd: [2]u8 = undefined;
    testing.io.random(&rnd);
    const base: u16 = 30000 + @as(u16, std.mem.readInt(u16, &rnd, .little) % 20000);
    const t = try tools(a, base, base + 1);

    // the nodes first (a rank's node must answer when rank 0 starts the group), each knowing the ports started before it
    var s3 = try start(a, "ranks-s3", t, "s3", .{ 9, 9, 9 });
    var s3_alive = true;
    defer if (s3_alive) s3.stop();
    var s2 = try start(a, "ranks-s2", t, "s2", .{ 9, 9, s3.port });
    defer s2.stop();
    var s1 = try start(a, "ranks-s1", t, "s1", .{ 9, s2.port, s3.port });
    defer s1.stop();

    // rank 0 of each group answers, and says how many ranks it had
    var big = try ask(&s1, a, "big");
    for (0..100) |_| { // s2 loads big's group at its start: wait until it is up
        if (big.status == .ok) break;
        sleepMs(testing.io, 100);
        big = try ask(&s1, a, "big");
    }
    try testing.expectEqual(std.http.Status.ok, big.status);
    try testing.expect(std.mem.indexOf(u8, big.body, "(rank 0 of 2)") != null);
    const wide = try ask(&s1, a, "wide");
    try testing.expectEqual(std.http.Status.ok, wide.status);
    try testing.expect(std.mem.indexOf(u8, wide.body, "(rank 0 of 3)") != null);

    // every node admitted its own rank: s3 holds a rank of each, loaded, and serves neither
    try testing.expectEqualStrings("ready", try s3.toolState(a, "big"));
    try testing.expectEqualStrings("ready", try s3.toolState(a, "wide"));
    try testing.expectEqual(@as(usize, 2), (try s3.workers(a)).len);
    const refused = try ask(&s3, a, "big");
    try testing.expectEqual(std.http.Status.bad_request, refused.status);
    try testing.expect(std.mem.indexOf(u8, refused.body, "rank 1 of 'big'") != null);
    try testing.expect(std.mem.indexOf(u8, refused.body, "s2") != null);
    const models = try s1.call(a, .GET, "/v1/models", null);
    try testing.expect(std.mem.indexOf(u8, models.body, "\"id\":\"big\"") != null);

    // a rank's node goes: its groups stop, and the next requests fail with a reason naming it until it is back
    s3.stop();
    s3_alive = false;
    var named = false;
    for (0..100) |_| {
        const r = try ask(&s1, a, "wide");
        try testing.expect(r.status != .ok or std.mem.indexOf(u8, r.body, "rank 0 of 3") != null);
        if (r.status == .service_unavailable and std.mem.indexOf(u8, r.body, "s3") != null) {
            named = true;
            break;
        }
        sleepMs(testing.io, 100);
    }
    try testing.expect(named);
}

test "tensor parallel: a group's tool must say how its server learns its rank" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try std.mem.replaceOwned(u8, a, try tools(a, 1, 2), "\"rank_env\"", "\"no_rank_env\"");
    const cfg = try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", {s}, {s}}}", .{ t, try topo(a, "s1", .{ 9, 9, 9 }) });
    const dir = "/tmp/localrouter-it-ranks-bad";
    try std.Io.Dir.cwd().createDirPath(testing.io, dir);
    defer std.Io.Dir.cwd().deleteTree(testing.io, dir) catch {};
    try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = dir ++ "/config.json", .data = cfg });
    const r = try std.process.run(a, testing.io, .{ .argv = &.{ opts.localrouter_exe, "serve", "--config", dir ++ "/config.json", "--port", "0", "--data", dir ++ "/data" } });
    try testing.expectEqual(@as(u8, 2), r.term.exited);
    try testing.expect(std.mem.indexOf(u8, r.stderr, "no options.rank_env") != null);
}
