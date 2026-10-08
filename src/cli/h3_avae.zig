//! `localrouter check h3-avae CHECKPOINT CAPTURE`: MiniMax H3's audio VAE decoder (the plain fp32 checkpoint
//! minimax_h3_audio_vae_fp32.safetensors) against a twin capture made with `stk_twin.h3.capture_avae` (ops `avae.*`, the
//! `avae_shapes` note): every op alone (from the captured inputs) and chained (from Zig's own outputs), and the decode
//! time. One JSON line.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 2) {
        std.debug.print("usage: localrouter check h3-avae CHECKPOINT CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[1]);
    defer cap.close();
    const note = (cap.note("avae_shapes") orelse return error.NoShapes).object;
    const a_len: u32 = @intCast(note.get("a").?.integer);
    const zi = cap.find("avae.latent_in", 0) orelse return error.NoLatent;
    const zref = cap.ops[zi].in("z") orelse return error.NoLatent;

    var pack = try qi.pack.Pack.openFile(gpa, io, args[0]);
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
    var ops = try h3.vae_audio.Ops.load(&d, h3.kernels.vae_audio);
    defer ops.unload();
    var gemm = try h3.launch.Ops.load(&d, h3.kernels.ops);
    defer gemm.unload();
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    const t0 = std.Io.Clock.awake.now(io);
    var vae = try h3.vae_audio.Decoder.init(gpa, io, &d, &ops, &gemm, s, &pack, &up, a_len);
    defer vae.deinit();
    try s.synchronize();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();

    // the captured latent, on the device (the chained mode starts from it too)
    const host = try a.alloc(u8, zref.bytes());
    try cap.blob(io, zref, host);
    var lat = try cuda.DeviceBuffer.fromHost(&d, host);
    defer lat.free();
    const latent = try lat.at(0);

    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"h3_avae\": {{\"gpu\": \"sm_{d}{d}\", \"a\": {d}, \"samples\": {d}, \"load_ms\": {d}, \"weights_bytes\": {d}", .{ maj, min, a_len, @as(u64, a_len) * h3.vae_audio.hop, load_ms, vae.store.bytes });
    var pass = true;
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        vae.probe = r.probe();
        _ = try vae.decode(latent, a_len);
        try s.synchronize();
        pass = pass and r.stats.differ == 0 and r.stats.unmatched == 0 and r.stats.equal > 0;
        try out.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.skipped, r.stats.unmatched });
        for (r.first_diffs.items, 0..) |m, i| try out.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
        try out.writer.writeAll("]}");
    }
    vae.probe = null;
    {
        const t1 = std.Io.Clock.awake.now(io);
        _ = try vae.decode(latent, a_len);
        try s.synchronize();
        try out.writer.print(", \"decode_ms\": {d}", .{t1.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds()});
    }
    try out.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}
