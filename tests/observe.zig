//! GET /v1/status and GET /metrics against a real daemon: the snapshot, Prometheus text, and /health left as it was.

const std = @import("std");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;

test "status and metrics describe tools and memory, and /health stays {\"ok\":true}" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "status", "{" ++ harness.reserve ++
        ", \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 0, \"options\": {\"step_ms\": 5}}]}");
    defer d.stop();

    const health = try d.call(a, .GET, "/health", null);
    try testing.expectEqual(std.http.Status.ok, health.status);
    try testing.expectEqualStrings("{\"ok\":true}", health.body);

    const before = try d.call(a, .GET, "/v1/status", null);
    try testing.expectEqual(std.http.Status.ok, before.status);
    try testing.expect(std.mem.indexOf(u8, before.body, d.dir) == null);
    try testing.expect(std.mem.indexOf(u8, before.body, "/tmp/") == null);
    const snap = try std.json.parseFromSliceLeaky(std.json.Value, a, before.body, .{});
    try testing.expectEqualStrings("localrouter.status", snap.object.get("object").?.string);
    try testing.expectEqual(@as(i64, 0), snap.object.get("evictions").?.integer);
    const mem = snap.object.get("memory").?.object;
    try testing.expect(mem.get("ceiling_bytes").?.integer >= mem.get("used_bytes").?.integer);
    try testing.expectEqual(mem.get("ceiling_bytes").?.integer - mem.get("used_bytes").?.integer, mem.get("free_bytes").?.integer);
    const tool0 = snap.object.get("tools").?.array.items[0].object;
    try testing.expectEqualStrings("tp", tool0.get("id").?.string);
    try testing.expectEqualStrings("unloaded", tool0.get("state").?.string);
    try testing.expectEqual(@as(i64, 0), tool0.get("resident_bytes").?.integer);
    try testing.expectEqual(@as(i64, 0), tool0.get("queue_depth").?.integer);
    try testing.expect(tool0.get("last_load_s") == null);
    try testing.expect(tool0.get("last_error") == null);

    const metrics = try d.call(a, .GET, "/metrics", null);
    try testing.expectEqual(std.http.Status.ok, metrics.status);
    try testing.expect(std.mem.indexOf(u8, metrics.body, "localrouter_evictions_total 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, metrics.body, "localrouter_tool_state{tool=\"tp\",state=\"unloaded\"} 1\n") != null);
    try testing.expect(std.mem.indexOf(u8, metrics.body, "localrouter_memory_bytes{kind=\"free\"}") != null);
    try testing.expect(std.mem.indexOf(u8, metrics.body, d.dir) == null);
    try testing.expect(std.mem.indexOf(u8, metrics.body, "/tmp/") == null);

    // A job loads the tool. idle_ttl 0 unloads it when the job ends; the last load time stays.
    try testing.expectEqual(std.http.Status.ok, (try harness.image(&d, a, "tp", "")).status);
    var saw_load = false;
    for (0..50) |_| {
        const body = (try d.call(a, .GET, "/v1/status", null)).body;
        const again = try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{});
        const tp = again.object.get("tools").?.array.items[0].object;
        if (tp.get("last_load_s") != null and std.mem.eql(u8, tp.get("state").?.string, "unloaded")) {
            try testing.expect(tp.get("last_load_s").?.integer > 0);
            try testing.expectEqual(@as(i64, 0), tp.get("resident_bytes").?.integer);
            try testing.expect(std.mem.indexOf(u8, body, d.dir) == null);
            saw_load = true;
            break;
        }
        harness.sleepMs(io, 20);
    }
    try testing.expect(saw_load);
}
