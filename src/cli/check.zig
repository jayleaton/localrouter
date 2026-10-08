//! `localrouter check gpu`: opens the CUDA driver through TensorFold's runtime and reports devices and memory.
//! `localrouter check health [URL]`: exit 0 when the daemon answers /health (the container's health check).
//! `localrouter check selftest`: starts a private daemon on a free port with the test-pattern tools, makes one image and
//! one video through the HTTP API, checks the files, and stops it. Needs no GPU; ffmpeg must be on PATH.

const std = @import("std");
const Io = std.Io;
const cuda = @import("cuda");
const memory = @import("../sched/memory.zig");

pub fn run(io: Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len == 1 and std.mem.eql(u8, args[0], "gpu")) return gpu();
    if (args.len == 1 and std.mem.eql(u8, args[0], "selftest")) return selftest(io, gpa);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "qwen-bench")) return @import("qwen_bench.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "h3-te")) return @import("h3_te.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "h3-replay")) return @import("h3_replay.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "h3-avae")) return @import("h3_avae.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "h3-vvae")) return @import("h3_vvae.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "h3-e2e")) return @import("h3_e2e.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "qwen-vae")) return @import("qwen_vae.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "qwen-e2e")) return @import("qwen_e2e.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "qwen-te")) return @import("qwen_te.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "qwen-replay")) return @import("qwen_replay.zig").run(io, gpa, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "health")) return health(io, gpa, if (args.len > 1) args[1] else "http://127.0.0.1:8190");
    std.debug.print("usage: localrouter check gpu|selftest|health [URL]\n", .{});
    return 2;
}

fn gpu() !u8 {
    std.debug.print("MemAvailable: {d} MiB\n", .{(memory.available() orelse 0) >> 20});
    var d = cuda.Driver.open() catch |err| {
        std.debug.print("CUDA driver: unavailable ({s})\n", .{@errorName(err)});
        return 1;
    };
    defer d.close();
    const v = try d.version();
    const n = try d.deviceCount();
    std.debug.print("CUDA driver {d}.{d}, {d} device(s)\n", .{ @divTrunc(v, 1000), @divTrunc(@mod(v, 1000), 10), n });
    if (n == 0) return 1;
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    const m = try ctx.memInfo();
    std.debug.print("device 0: {d} MiB free of {d} MiB\n", .{ m.free >> 20, m.total >> 20 });
    return 0;
}

const selftest_config =
    \\{"reserve_bytes": 67108864, "tools": [
    \\ {"id": "testpattern", "kind": "image", "engine": "testpattern", "idle_ttl_s": 0},
    \\ {"id": "testpattern-video", "kind": "video", "engine": "testpattern", "idle_ttl_s": 0}]}
;

fn selftest(io: Io, gpa: std.mem.Allocator) !u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rnd: [4]u8 = undefined;
    io.random(&rnd);
    const dir = try std.fmt.allocPrint(a, "/tmp/localrouter-selftest-{x}", .{std.mem.readInt(u32, &rnd, .little)});
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const cfg = try std.fs.path.join(a, &.{ dir, "config.json" });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = cfg, .data = selftest_config });
    const exe = try std.process.executablePathAlloc(io, a);
    var child = try std.process.spawn(io, .{ .argv = &.{ exe, "serve", "--config", cfg, "--host", "127.0.0.1", "--port", "0", "--data", try std.fs.path.join(a, &.{ dir, "data" }) }, .stdout = .pipe });
    defer {
        _ = std.posix.system.kill(child.id.?, std.posix.SIG.TERM);
        _ = child.wait(io) catch {};
    }
    var buf: [256]u8 = undefined;
    var r = child.stdout.?.reader(io, &buf);
    const line = try r.interface.takeDelimiterExclusive('\n');
    const base = try a.dupe(u8, line["localrouter listening on ".len..]);
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    const t0 = Io.Clock.awake.now(io);
    const img = try fetch(&client, a, .POST, base, "/v1/images/generations", "{\"model\":\"testpattern\",\"prompt\":\"selftest\",\"size\":\"512x512\",\"seed\":1,\"response_format\":\"url\"}");
    if (img.status != .ok) return report("image", img.body);
    const I = struct { data: []const struct { url: []const u8 } };
    const url = (try std.json.parseFromSliceLeaky(I, a, img.body, .{ .ignore_unknown_fields = true })).data[0].url;
    const png = try fetch(&client, a, .GET, "", url, null);
    if (png.status != .ok or !std.mem.startsWith(u8, png.body, "\x89PNG\r\n\x1a\n")) return report("png", "not a PNG");
    const t1 = Io.Clock.awake.now(io);

    const vid = try fetch(&client, a, .POST, base, "/v1/videos", "{\"model\":\"testpattern-video\",\"prompt\":\"selftest\",\"size\":\"512x288\",\"seconds\":\"1\",\"seed\":1}");
    if (vid.status != .ok) return report("video", vid.body);
    const V = struct { id: []const u8, status: []const u8 };
    const id = (try std.json.parseFromSliceLeaky(V, a, vid.body, .{ .ignore_unknown_fields = true })).id;
    const path = try std.fmt.allocPrint(a, "/v1/videos/{s}", .{id});
    var status: []const u8 = "queued";
    for (0..300) |_| {
        const s = try fetch(&client, a, .GET, base, path, null);
        status = (try std.json.parseFromSliceLeaky(V, a, s.body, .{ .ignore_unknown_fields = true })).status;
        if (!std.mem.eql(u8, status, "queued") and !std.mem.eql(u8, status, "in_progress")) break;
        try Io.sleep(io, .fromMilliseconds(100), .awake);
    }
    if (!std.mem.eql(u8, status, "completed")) return report("video", status);
    const mp4 = try fetch(&client, a, .GET, base, try std.fmt.allocPrint(a, "{s}/content", .{path}), null);
    if (mp4.status != .ok or mp4.body.len < 8 or !std.mem.eql(u8, mp4.body[4..8], "ftyp")) return report("mp4", "not an MP4");
    const t2 = Io.Clock.awake.now(io);
    std.debug.print("selftest ok: image {d} ms (cold start included), video {d} ms, {d} B mp4\n", .{ t0.durationTo(t1).toMilliseconds(), t1.durationTo(t2).toMilliseconds(), mp4.body.len });
    return 0;
}

/// The container's health check: GET /health on the local daemon.
fn health(io: Io, gpa: std.mem.Allocator, base: []const u8) !u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();
    const r = fetch(&client, arena.allocator(), .GET, base, "/health", null) catch return 1;
    return if (r.status == .ok) 0 else 1;
}

const Resp = struct { status: std.http.Status, body: []u8 };

fn fetch(c: *std.http.Client, a: std.mem.Allocator, method: std.http.Method, base: []const u8, path: []const u8, body: ?[]const u8) !Resp {
    var out: Io.Writer.Allocating = .init(a);
    const url = try std.fmt.allocPrint(a, "{s}{s}", .{ base, path });
    const res = try c.fetch(.{ .location = .{ .url = url }, .method = method, .payload = body, .response_writer = &out.writer, .keep_alive = false });
    return .{ .status = res.status, .body = out.written() };
}

fn report(what: []const u8, detail: []const u8) u8 {
    std.debug.print("selftest FAILED at {s}: {s}\n", .{ what, detail });
    return 1;
}
