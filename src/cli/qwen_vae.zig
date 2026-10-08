//! `localrouter check qwen-vae VAE_PACK CAPTURE`: the VAE decoder's bit gate on a GPU. From the capture's final latents, every
//! decoder op replayed alone (from the captured inputs) and chained (from Zig's own outputs), the pixels compared.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 2) {
        std.debug.print("usage: localrouter check qwen-vae VAE_PACK CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[1]);
    defer cap.close();
    var pack = try qi.pack.Pack.open(gpa, io, args[0]);
    defer pack.close(io);
    const lat_ref = cap.ops[cap.find("latents", 0) orelse return error.NoLatents].out("x").?;
    const h: u32 = @intCast(lat_ref.shape[2]);
    const w: u32 = @intCast(lat_ref.shape[3]);

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
    var ops = try qi.ops_launch.Ops.load(&d, qi.kernels.ops, qi.kernels.attention);
    defer ops.unload();
    var tk = try qi.ops_launch.TeOps.load(&d, qi.kernels.te, qi.kernels.gemm);
    defer tk.unload();
    var vk = try qi.vae.VaeOps.load(&d, qi.kernels.vae);
    defer vk.unload();
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    var vae = try qi.vae.Vae.init(gpa, io, &d, &vk, &tk, &ops, s, &pack, &up, args[0], h, w);
    try s.synchronize();
    defer vae.deinit();
    var lat = try cuda.DeviceBuffer.alloc(&d, lat_ref.bytes());
    defer lat.free();
    {
        const host = try a.alloc(u8, lat_ref.bytes());
        try cap.blob(io, lat_ref, host);
        try lat.upload(0, host);
    }
    const npx: usize = 256 * @as(usize, h) * w * vae.out_channels;
    var px = try cuda.DeviceBuffer.alloc(&d, npx);
    defer px.free();

    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"qwen_vae\": {{\"gpu\": \"sm_{d}{d}\", \"latents\": [{d}, {d}]", .{ maj, min, h, w });
    var pass = true;
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        vae.probe = r.probe();
        try vae.decode(try lat.at(0), h, w, try px.at(0));
        try s.synchronize();
        pass = pass and r.stats.differ == 0 and r.stats.unmatched == 0 and r.stats.equal > 0;
        try out.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.skipped, r.stats.unmatched });
        for (r.first_diffs.items, 0..) |m, i| try out.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
        try out.writer.writeAll("]}");
    }
    vae.probe = null;
    { // decode time, warm
        try vae.decode(try lat.at(0), h, w, try px.at(0));
        try s.synchronize();
        const t0 = std.Io.Clock.awake.now(io);
        const reps = 3;
        for (0..reps) |_| try vae.decode(try lat.at(0), h, w, try px.at(0));
        try s.synchronize();
        const us = t0.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds();
        try out.writer.print(", \"decode_ms\": {d:.2}", .{@as(f64, @floatFromInt(us)) / 1000.0 / reps});
    }
    try out.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}
