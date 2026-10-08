//! End to end: the real `localrouter` daemon and real worker processes, driven over HTTP. No part is mocked; the
//! testpattern engine stands in for a GPU engine (same protocol, memory accounting and output paths).
//! `harness.zig` starts the daemon; `router.zig` has the tests of priorities, image inputs and the stdio bridge.

const std = @import("std");
const Io = std.Io;
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const mib = harness.mib;
const reserve = harness.reserve;
const readProc = harness.readProc;
const image = harness.image;
const firstPng = harness.firstPng;
const sleepMs = harness.sleepMs;
const mcpCall = harness.mcpCall;
const mcpText = harness.mcpText;

test {
    _ = @import("router.zig");
}
test "images: deterministic, validated, and concurrent requests queue" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "images", "{" ++ reserve ++ ", \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 20}}]}");
    defer d.stop();

    const r1 = try image(&d, a, "tp", ",\"seed\":7");
    try testing.expectEqual(std.http.Status.ok, r1.status);
    const png1 = try firstPng(a, r1.body);
    try testing.expectEqualSlices(u8, "\x89PNG\r\n\x1a\n", png1[0..8]);
    const png2 = try firstPng(a, (try image(&d, a, "tp", ",\"seed\":7")).body);
    try testing.expectEqualSlices(u8, png1, png2); // same seed, same bytes
    const png3 = try firstPng(a, (try image(&d, a, "tp", ",\"seed\":8")).body);
    try testing.expect(!std.mem.eql(u8, png1, png3));

    try testing.expectEqual(std.http.Status.bad_request, (try d.call(a, .POST, "/v1/images/generations", "{\"model\":\"tp\",\"prompt\":\"x\",\"size\":\"1000x1000\"}")).status);
    try testing.expectEqual(std.http.Status.not_found, (try image(&d, a, "nope", "")).status);
    try testing.expectEqual(std.http.Status.not_found, (try d.call(a, .GET, "/v1/nothing", null)).status);

    // Four clients at once: the scheduler runs them one by one and all succeed.
    const Worker = struct {
        fn run(port: u16, ok: *std.atomic.Value(u32)) void {
            var c: std.http.Client = .{ .allocator = std.heap.page_allocator, .io = testing.io };
            defer c.deinit();
            var buf: [64]u8 = undefined;
            const url = std.fmt.bufPrint(&buf, "http://127.0.0.1:{d}/v1/images/generations", .{port}) catch return;
            var out: Io.Writer.Allocating = .init(std.heap.page_allocator);
            defer out.deinit();
            const res = c.fetch(.{ .location = .{ .url = url }, .method = .POST, .payload = "{\"model\":\"tp\",\"prompt\":\"q\",\"size\":\"256x256\"}", .response_writer = &out.writer, .keep_alive = false }) catch return;
            if (res.status == .ok) _ = ok.fetchAdd(1, .monotonic);
        }
    };
    var ok: std.atomic.Value(u32) = .init(0);
    var threads: [4]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Worker.run, .{ d.port, &ok });
    for (threads) |t| t.join();
    try testing.expectEqual(@as(u32, 4), ok.load(.monotonic));
}

test "mcp: initialize, tools, an image inline and a video by URL, a job polled to completion" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "mcp", "{" ++ reserve ++ ", \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\"}, {\"id\": \"tpv\", \"kind\": \"video\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 200}}]}");
    defer d.stop();

    const init = try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}}");
    const iv = try std.json.parseFromSliceLeaky(std.json.Value, a, init.body, .{});
    try testing.expectEqualStrings("2025-03-26", iv.object.get("result").?.object.get("protocolVersion").?.string);
    try testing.expectEqual(std.http.Status.accepted, (try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}")).status);
    try testing.expectEqual(std.http.Status.method_not_allowed, (try d.call(a, .GET, "/mcp", null)).status);
    const list = try std.json.parseFromSliceLeaky(std.json.Value, a, (try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}")).body, .{});
    try testing.expectEqual(@as(usize, 6), list.object.get("result").?.object.get("tools").?.array.items.len);

    // an image: the PNG inline, and its URL serves the same bytes
    const img = try mcpCall(&d, a, "generate_image", "{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7}");
    try testing.expect(!img.object.get("isError").?.bool);
    const first = img.object.get("content").?.array.items[0].object;
    try testing.expectEqualStrings("image", first.get("type").?.string);
    const dec = std.base64.standard.Decoder;
    const b64 = first.get("data").?.string;
    const png = try a.alloc(u8, try dec.calcSizeForSlice(b64));
    try dec.decode(png, b64);
    const info = try mcpText(a, img);
    try testing.expectEqual(@as(i64, 7), info.object.get("seed").?.integer);
    const url = info.object.get("files").?.array.items[0].string;
    const served = try d.call(a, .GET, url[std.mem.indexOf(u8, url, "/v1/").?..], null);
    try testing.expectEqualSlices(u8, png, served.body);

    // a video not waited for: queued, then get_job waits it out and links the MP4
    const v = try mcpText(a, try mcpCall(&d, a, "generate_video", "{\"prompt\":\"waves\",\"size\":\"320x256\",\"seconds\":1,\"wait_s\":0}"));
    try testing.expect(!std.mem.eql(u8, v.object.get("status").?.string, "completed"));
    const id = v.object.get("id").?.string;
    const done = try mcpText(a, try mcpCall(&d, a, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\",\"wait_s\":60}}", .{id})));
    try testing.expectEqualStrings("completed", done.object.get("status").?.string);
    const mp4 = done.object.get("files").?.array.items[0].string;
    try testing.expect(std.mem.endsWith(u8, mp4, ".mp4"));
    try testing.expect((try d.call(a, .GET, mp4[std.mem.indexOf(u8, mp4, "/v1/").?..], null)).body.len > 1000);

    // refusals are tool errors, not protocol errors; an unknown tool is a protocol error
    try testing.expect((try mcpCall(&d, a, "generate_image", "{\"prompt\":\"x\",\"size\":\"1000x1000\"}")).object.get("isError").?.bool);
    try testing.expect((try mcpCall(&d, a, "generate_image", "{\"size\":\"256x256\"}")).object.get("isError").?.bool);
    try testing.expect(!(try mcpCall(&d, a, "release_models", "{}")).object.get("isError").?.bool);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "tp"));
}

test "memory budget: least recently used tool is evicted, oversized requests get 503" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Budget 100 MiB: A and B (40 MiB each) cannot both stay loaded with a request's working bytes on top.
    var d = try Daemon.start(testing.allocator, io, "budget", "{" ++ reserve ++ ", \"budget_bytes\": 104857600, \"tools\": [" ++
        "{\"id\": \"a\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"resident_mb\": 40}}," ++
        "{\"id\": \"b\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"resident_mb\": 40}}," ++
        "{\"id\": \"big\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"resident_mb\": 200}}]}");
    defer d.stop();

    try testing.expectEqual(std.http.Status.ok, (try image(&d, a, "a", "")).status);
    try testing.expectEqualStrings("ready", try d.toolState(a, "a"));
    const rb = try image(&d, a, "b", ",\"size\":\"2048x2048\"");
    if (rb.status != .ok) std.debug.print("b: {s}\n", .{rb.body});
    try testing.expectEqual(std.http.Status.ok, rb.status);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "a")); // evicted to fit b plus its 24 MiB frame
    try testing.expectEqualStrings("ready", try d.toolState(a, "b"));
    const big = try image(&d, a, "big", "");
    try testing.expectEqual(std.http.Status.service_unavailable, big.status);
    try testing.expect(std.mem.indexOf(u8, big.body, "insufficient_memory") != null);

    // Release unloads everything; no worker process is left.
    try testing.expectEqual(std.http.Status.ok, (try d.call(a, .POST, "/v1/tools/release", "")).status);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "b"));
    try testing.expectEqual(@as(usize, 0), (try d.workers(a)).len);
}

test "a killed worker fails one job; the next one reloads; cancel and idle unload" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "faults", "{" ++ reserve ++ ", \"tools\": [" ++
        "{\"id\": \"slow\", \"kind\": \"video\", \"engine\": \"testpattern\", \"idle_ttl_s\": 1, \"options\": {\"step_ms\": 300}}]}");
    defer d.stop();
    const create = "{\"model\":\"slow\",\"prompt\":\"bars\",\"size\":\"256x256\",\"seconds\":\"1\",\"steps\":20,\"audio\":false}";
    const V = struct { id: []const u8, status: []const u8, @"error": ?struct { code: []const u8, message: []const u8 } = null };

    // Crash: kill the worker mid-job.
    const v1 = try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .POST, "/v1/videos", create)).body, .{ .ignore_unknown_fields = true });
    var pids: []i32 = &.{};
    for (0..50) |_| {
        pids = try d.workers(a);
        if (pids.len > 0) break;
        sleepMs(io, 50);
    }
    try testing.expectEqual(@as(usize, 1), pids.len);
    sleepMs(io, 300);
    _ = std.posix.system.kill(pids[0], std.posix.SIG.KILL);
    sleepMs(io, 300);
    const after = try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{v1.id}), null)).body, .{ .ignore_unknown_fields = true });
    try testing.expectEqualStrings("failed", after.status);
    try testing.expect(std.mem.indexOf(u8, after.@"error".?.message, "worker exited") != null);

    // Cancel: a new job reloads the tool; DELETE kills it.
    const v2 = try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .POST, "/v1/videos", create)).body, .{ .ignore_unknown_fields = true });
    sleepMs(io, 500);
    _ = try d.call(a, .DELETE, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{v2.id}), null);
    sleepMs(io, 300);
    const gone = try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{v2.id}), null)).body, .{ .ignore_unknown_fields = true });
    try testing.expectEqualStrings("cancelled", gone.status);
    try testing.expectEqual(@as(usize, 0), (try d.workers(a)).len);

    // A complete job, then the 1 s idle TTL unloads the tool.
    const fast = "{\"model\":\"slow\",\"prompt\":\"bars\",\"size\":\"256x256\",\"seconds\":\"1\",\"steps\":1}";
    const v3 = try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .POST, "/v1/videos", fast)).body, .{ .ignore_unknown_fields = true });
    var status: []const u8 = "";
    for (0..100) |_| {
        status = (try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{v3.id}), null)).body, .{ .ignore_unknown_fields = true })).status;
        if (std.mem.eql(u8, status, "completed") or std.mem.eql(u8, status, "failed")) break;
        sleepMs(io, 100);
    }
    try testing.expectEqualStrings("completed", status);
    const mp4 = try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}/content", .{v3.id}), null);
    try testing.expectEqual(std.http.Status.ok, mp4.status);
    try testing.expectEqualSlices(u8, "ftyp", mp4.body[4..8]);
    try testing.expectEqualStrings("ready", try d.toolState(a, "slow"));
    sleepMs(io, 2500);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "slow"));
    try testing.expectEqual(@as(usize, 0), (try d.workers(a)).len);
}

test "overhead: idle daemon RSS and API latency" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "overhead", "{" ++ reserve ++ "}");
    defer d.stop();
    var times: [200]i64 = undefined;
    for (&times) |*t| {
        const t0 = Io.Clock.awake.now(io);
        const r = try d.call(a, .GET, "/v1/models", null);
        t.* = t0.durationTo(Io.Clock.awake.now(io)).toMicroseconds();
        try testing.expectEqual(std.http.Status.ok, r.status);
    }
    std.mem.sort(i64, &times, {}, std.sort.asc(i64));
    var buf: [4096]u8 = undefined;
    const status = readProc(try std.fmt.allocPrintSentinel(a, "/proc/{d}/status", .{d.child.id.?}, 0), &buf) orelse return error.NoProc;
    const at = std.mem.indexOf(u8, status, "VmRSS:") orelse return error.NoRss;
    const rss_kb = try std.fmt.parseInt(u64, std.mem.trim(u8, status[at + 6 .. std.mem.indexOfScalarPos(u8, status, at, 'k').?], " \t"), 10);
    std.debug.print("\noverhead: /v1/models p50 {d} us, p99 {d} us; idle daemon RSS {d} KiB\n", .{ times[100], times[198], rss_kb });
    try testing.expect(times[100] < 5000);
    try testing.expect(rss_kb * 1024 < 30 * mib);
}
