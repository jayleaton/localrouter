//! `localrouter check qwen-bench PACK [WxH]`: the Zig DiT's speed on this GPU: one prefix (64 text rows), then 10 timed steps
//! (GPU events), plus the weight load time. Prints JSON.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 1) {
        std.debug.print("usage: localrouter check qwen-bench PACK [WxH]\n", .{});
        return 2;
    }
    const size = if (args.len > 1) args[1] else "1024x1024";
    const x = std.mem.indexOfScalar(u8, size, 'x') orelse return 2;
    const width = try std.fmt.parseInt(u32, size[0..x], 10);
    const height = try std.fmt.parseInt(u32, size[x + 1 ..], 10);
    const h = height / 16;
    const w = width / 16;
    const n = h * w;
    const p: u32 = 64;
    var pack = try qi.pack.Pack.open(gpa, io, args[0]);
    defer pack.close(io);
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var maj: c_int = 0;
    var min: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&maj, .compute_capability_major, ctx.device), "cc");
    try d.check(d.api.cuDeviceGetAttribute(&min, .compute_capability_minor, ctx.device), "cc");
    var s = try cuda.Stream.init(&d, false);
    defer s.deinit();
    var nv = try qi.nvfp4_exec.Nvfp4.load(gpa, &d, @intCast(maj), @intCast(min));
    defer nv.unload();
    var ops = try qi.ops_launch.Ops.load(&d, qi.kernels.ops, qi.kernels.attention);
    defer ops.unload();
    const tri_cap = if (args.len > 2) args[2] else return error.NeedCaptureForTritonCubins;
    var cap = try qi.capture.Capture.open(gpa, io, tri_cap);
    defer cap.close();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var tri = try @import("qwen_replay.zig").loadTriton(arena.allocator(), io, &d, ctx.device, &cap);
    defer tri.unload();
    const t0 = std.Io.Clock.awake.now(io);
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    var wts = try qi.weights.Weights.load(gpa, io, &d, &pack, &nv, &up);
    try s.synchronize();
    defer wts.deinit();
    try s.synchronize();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    const om = try qi.smath.omegas(qi.kernels.rope_omega);
    var dit = try qi.dit.Dit.init(gpa, &d, .{ .ops = &ops, .nv = &nv, .tri = &tri, .major = @intCast(maj), .minor = @intCast(min) }, &wts, s, om, n, p);
    defer dit.deinit();
    var cb = try cuda.DeviceBuffer.alloc(&d, @as(usize, p) * 4096 * 2);
    defer cb.free();
    var lat = try cuda.DeviceBuffer.alloc(&d, 64 * @as(usize, n) * 2);
    defer lat.free();
    var vel = try cuda.DeviceBuffer.alloc(&d, 64 * @as(usize, n) * 2);
    defer vel.free();
    try cb.fill8(0x3c, null); // a finite bf16 pattern (about 0.0117): the speed does not depend on the values
    try lat.fill8(0x3c, null);
    try dit.buildPrefix(try cb.at(0), p, 0.5);
    var e0 = try cuda.Event.init(&d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(&d, true);
    defer e1.deinit();
    for (0..2) |_| try dit.step(try lat.at(0), 0.5, h, w, try vel.at(0));
    try e0.record(s);
    for (0..10) |_| try dit.step(try lat.at(0), 0.5, h, w, try vel.at(0));
    try e1.record(s);
    try e1.synchronize();
    const ms = try cuda.Event.elapsedMs(e0, e1) / 10.0;
    var buf: [256]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{{\"qwen_bench\": {{\"gpu\": \"sm_{d}{d}\", \"precision\": \"{s}\", \"size\": \"{s}\", \"step_ms\": {d:.2}, \"load_ms\": {d}}}}}\n", .{ maj, min, pack.precision, size, ms, load_ms });
    try std.Io.File.stdout().writeStreamingAll(io, line);
    return 0;
}
