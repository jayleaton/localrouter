//! `localrouter check qwen-replay PACK CAPTURE`: the M3 bit gate on a GPU. Loads the pack, replays the capture's step 0 through
//! the Zig forward alone (each op from the captured inputs) and chained (from its own outputs), prints a JSON summary.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 2) {
        std.debug.print("usage: localrouter check qwen-replay PACK CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[1]);
    defer cap.close();
    const req = try qi.replay.request(&cap, a);
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
    var s = try cuda.Stream.init(&d, false); // blocking: ordered with the synchronous uploads
    defer s.deinit();

    var nv = try qi.nvfp4_exec.Nvfp4.load(gpa, &d, @intCast(maj), @intCast(min));
    defer nv.unload();
    var ops = try qi.ops_launch.Ops.load(&d, qi.kernels.ops, qi.kernels.attention);
    defer ops.unload();
    var tri = try loadTriton(a, io, &d, ctx.device, &cap);
    defer tri.unload();
    const t0 = std.Io.Clock.awake.now(io);
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    var w = try qi.weights.Weights.load(gpa, io, &d, &pack, &nv, &up);
    try s.synchronize();
    defer w.deinit();
    try s.synchronize();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();

    const n: u32 = (req.height / 16) * (req.width / 16);
    const om = try qi.smath.omegas(qi.kernels.rope_omega);
    var dit = try qi.dit.Dit.init(gpa, &d, .{ .ops = &ops, .nv = &nv, .tri = &tri, .major = @intCast(maj), .minor = @intCast(min) }, &w, s, om, n, req.ctx_rows);
    defer dit.deinit();
    var bufs: [4]cuda.DeviceBuffer = undefined;
    for (&bufs, [_]usize{ @as(usize, req.ctx_rows) * 4096 * 2, 64 * @as(usize, n) * 2, 64 * @as(usize, n) * 2, 64 * @as(usize, n) * 2 }) |*b, len| b.* = try cuda.DeviceBuffer.alloc(&d, len);
    defer for (&bufs) |*b| b.free();
    { // the text context the twin's encoder produced
        const ref = cap.ops[cap.find("text_encoder", 0).?].out("context").?;
        const host = try a.alloc(u8, ref.bytes());
        try cap.blob(io, ref, host);
        try bufs[0].upload(0, host);
    }
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"qwen_replay\": {{\"gpu\": \"sm_{d}{d}\", \"precision\": \"{s}\", \"load_ms\": {d}, \"weights_bytes\": {d}", .{ maj, min, pack.precision, load_ms, w.store.bytes });
    var pass = true;
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        try qi.replay.run(&r, &dit, req, try bufs[0].at(0), try bufs[1].at(0), try bufs[2].at(0), try bufs[3].at(0));
        try s.synchronize();
        pass = pass and r.stats.differ == 0 and r.stats.unmatched == 0;
        try out.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.skipped, r.stats.unmatched });
        for (r.first_diffs.items, 0..) |m, i| try out.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
        try out.writer.writeAll("]}");
    }
    try out.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}

/// The three Triton kernels from the capture's own cubins (the first launch of each).
pub fn loadTriton(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: cuda.abi.Device, cap: *const qi.capture.Capture) !qi.triton_k.Triton {
    const specs = try qi.triton_k.specsFromCapture(a, io, cap.dir);
    return qi.triton_k.Triton.load(d, device, specs[0] orelse return error.NoAdalnCubin, specs[1] orelse return error.NoRopeCubin, specs[2]);
}
