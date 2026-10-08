//! `localrouter check h3-vvae CHECKPOINT CAPTURE`: MiniMax H3's video VAE decoder (minimax_h3_video_vae_fp16.safetensors) against a
//! twin capture made with `python -m stk_twin.h3.capture_vvae` (ops `vvae.*`, the `vvae_shapes` note): every recorded op
//! alone (from the captured inputs) and chained (from Zig's own outputs), the decode time, and the final uint8 frames
//! against the capture's hash. The whole video is decoded; only the recorded tiles and layers are compared op by op.
//! One JSON line.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");
const vv = h3.vae_video;

fn hex(h: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(h, .lower);
}

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 2) {
        std.debug.print("usage: localrouter check h3-vvae CHECKPOINT CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[1]);
    defer cap.close();
    const note = (cap.note("vvae_shapes") orelse return error.NoShapes).object;

    // the latents: the capture's fp16 input of the denorm (what ComfyUI hands the VAE after `.to(fp16)`)
    const dn = cap.find("vvae.denorm", 0) orelse return error.NoDenorm;
    const zref = cap.ops[dn].in("z") orelse return error.NoDenorm;
    if (zref.shape.len != 4 or zref.shape[0] != 24 or !std.mem.eql(u8, zref.dtype, "float16")) return error.BadLatents;
    const tz: u32 = @intCast(zref.shape[1]);
    const hz: u32 = @intCast(zref.shape[2]);
    const wz: u32 = @intCast(zref.shape[3]);
    const zhost = try a.alloc(u8, zref.bytes());
    try cap.blob(io, zref, zhost);

    // which tiles and blocks the capture recorded
    var select: ?[][3]i32 = null;
    if (note.get("tiles")) |tv| {
        const items = tv.array.items;
        const sel = try a.alloc([3]i32, items.len);
        for (sel, items) |*o, e| for (o, e.array.items) |*x, y| {
            x.* = @intCast(y.integer);
        };
        select = sel;
    }
    var layers: ?[]u32 = null;
    if (note.get("layers")) |lv| {
        const ls = try a.alloc(u32, lv.array.items.len);
        for (ls, lv.array.items) |*o, e| o.* = @intCast(e.integer);
        layers = ls;
    }

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
    var ops = try vv.Ops.load(&d, h3.kernels.gemm_f16, h3.kernels.vae_video);
    defer ops.unload();
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    const t0 = std.Io.Clock.awake.now(io);
    var dec = try vv.Decoder.init(gpa, io, &d, &ops, s, args[0], &up);
    defer dec.deinit();
    try s.synchronize();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    dec.select = select;
    dec.layers = layers;

    var lat = try cuda.DeviceBuffer.fromHost(&d, zhost);
    defer lat.free();
    const frames = vv.outputFrames(tz);
    const out_bytes: usize = @as(usize, frames) * hz * 16 * wz * 16 * 3;
    var out = try cuda.DeviceBuffer.alloc(&d, out_bytes);
    defer out.free();

    var js: std.Io.Writer.Allocating = .init(a);
    try js.writer.print("{{\"h3_vvae\": {{\"gpu\": \"sm_{d}{d}\", \"latent\": [{d}, {d}, {d}], \"frames\": {d}, \"load_ms\": {d}, \"weights_bytes\": {d}", .{ maj, min, tz, hz, wz, frames, load_ms, dec.weightBytes() });
    var pass = true;
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        dec.probe = r.probe();
        try dec.decode(try lat.at(0), tz, hz, wz, try out.at(0));
        try s.synchronize();
        pass = pass and r.stats.differ == 0 and r.stats.unmatched == 0 and r.stats.equal > 0;
        try js.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.skipped, r.stats.unmatched });
        for (r.first_diffs.items, 0..) |m, i| try js.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
        try js.writer.writeAll("]}");
    }
    dec.probe = null;
    {
        const t1 = std.Io.Clock.awake.now(io);
        try dec.decode(try lat.at(0), tz, hz, wz, try out.at(0));
        try s.synchronize();
        try js.writer.print(", \"decode_ms\": {d}", .{t1.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds()});
    }
    // the final frames against the capture's hash (the whole video, whatever tiles were recorded)
    if (note.get("sha256")) |want| {
        const host = try gpa.alloc(u8, out_bytes);
        defer gpa.free(host);
        try out.download(0, host);
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &h, .{});
        const same = std.mem.eql(u8, &hex(h), want.string);
        pass = pass and same;
        try js.writer.print(", \"frames_equal\": {}", .{same});
    }
    try js.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, js.written());
    return if (pass) 0 else 1;
}
