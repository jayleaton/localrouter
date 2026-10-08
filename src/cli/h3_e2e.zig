//! `localrouter check h3-e2e PACK TE_CKPT VAE_AUDIO VAE_VIDEO CAPTURE`: MiniMax H3 text to video + audio end to end in Zig against
//! a twin run (`python -m stk_twin.h3.generate`): the capture's prompt, seed, size, frames and steps; the token ids, the
//! light ops (the text states, the noise, every step's velocities, denoised state and new state, the final latents, the
//! waveform) compared byte for byte chained from Zig's own outputs, the sha256 of the uint8 frames and of the fp32 waveform;
//! then the same request again, warm, for the times. One JSON line.

const std = @import("std");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");
const pl = h3.pipeline;

fn hex(h: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(h, .lower);
}

fn sha(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return hex(h);
}

fn int(v: std.json.Value) !u32 {
    return switch (v) {
        .integer => |i| @intCast(i),
        else => error.BadCapture,
    };
}

fn float(v: ?std.json.Value) f64 {
    return switch (v orelse return 0) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
}

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 5) {
        std.debug.print("usage: localrouter check h3-e2e PACK TE_CKPT VAE_AUDIO VAE_VIDEO CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[4]);
    defer cap.close();
    const req = (cap.note("request") orelse return error.NoRequest).object;
    const prompt = req.get("prompt").?.string;
    const seed: u64 = @intCast(req.get("seed").?.integer);
    const width = try int(req.get("width").?);
    const height = try int(req.get("height").?);
    const frames = try int(req.get("frames").?);
    const steps = try int(req.get("steps").?);
    const want_ids = (cap.note("te32_ids") orelse return error.NoIds).object.get("ids").?.array.items;
    const result = (cap.note("result") orelse return error.NoResult).object;
    const sh = try pl.shapes(width, height, frames);

    const t0 = std.Io.Clock.awake.now(io);
    const p = try pl.Pipeline.create(gpa, io, .{ .pack = args[0], .te = args[1], .vae_audio = args[2], .vae_video = args[3] }, .{
        .width = width,
        .height = height,
        .frames = frames,
        .tokens = @intCast(@max(want_ids.len + 64, 256)),
    }, .{});
    defer p.destroy();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();

    const px = try a.alloc(u8, @intCast(sh.frame_bytes));
    const wav = try a.alloc(f32, @intCast(sh.audioFloats()));

    // the first run, every light op compared chained
    var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &p.d, .s = p.s, .cap = &cap, .mode = .chained };
    defer r.deinit();
    p.probe = r.probe();
    const res = try p.generate(prompt, width, height, frames, steps, seed, px, wav, null);
    p.probe = null;
    const first = p.times;
    const used_mib = blk: {
        const m = try p.ctx.memInfo();
        break :blk (m.total - m.free) >> 20;
    };
    var ids_equal = want_ids.len == p.ids.len;
    if (ids_equal) for (p.ids, want_ids) |g, w| {
        ids_equal = ids_equal and @as(i64, g) == w.integer;
    };
    const frames_sha = sha(px);
    const wav_sha = sha(std.mem.sliceAsBytes(wav));
    const frames_equal = std.mem.eql(u8, &frames_sha, result.get("frames_sha256").?.string);
    const wav_equal = std.mem.eql(u8, &wav_sha, result.get("waveform_sha256").?.string);
    const light_ok = r.stats.differ == 0 and r.stats.unmatched == 0 and r.stats.equal > 0;

    // the same request again: warm times, the same bits
    _ = try p.generate(prompt, width, height, frames, steps, seed, px, wav, null);
    const warm = p.times;
    const frames_equal_warm = std.mem.eql(u8, &sha(px), &frames_sha) and std.mem.eql(u8, &sha(std.mem.sliceAsBytes(wav)), &wav_sha);

    const tw = if (cap.note("twin_ms")) |n| n.object else null;
    const pass = ids_equal and light_ok and frames_equal and wav_equal and frames_equal_warm;
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"h3_e2e\": {{\"gpu\": \"sm_{d}{d}\", \"size\": \"{d}x{d}\", \"frames\": {d}, \"steps\": {d}, \"tokens\": {d}, " ++
        "\"load_ms\": {d}, \"weights_bytes\": {d}, \"gpu_used_mib\": {d}, \"ids_equal\": {}, " ++
        "\"light\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"first_diffs\": [", .{
        p.major,         p.minor,           width,        height,          res.shapes.frames, steps,                  res.tokens,
        load_ms,         p.weights_bytes,   used_mib,     ids_equal,       r.stats.equal,     r.stats.differ,         r.stats.skipped,
        r.stats.unmatched,
    });
    for (r.first_diffs.items, 0..) |m, i| try out.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
    try out.writer.print("]}}, \"frames_equal\": {}, \"waveform_equal\": {}, \"equal_warm\": {}, \"frames_sha256\": \"{s}\", " ++
        "\"first_ms\": {{\"encode\": {d:.1}, \"sample\": {d:.1}, \"audio\": {d:.1}, \"video\": {d:.1}}}, " ++
        "\"warm_ms\": {{\"encode\": {d:.1}, \"sample\": {d:.1}, \"audio\": {d:.1}, \"video\": {d:.1}}}, " ++
        "\"twin_ms\": {{\"encode\": {d:.1}, \"sample\": {d:.1}, \"audio\": {d:.1}, \"video\": {d:.1}}}, \"pass\": {}}}}}\n", .{
        frames_equal,                             wav_equal,                                   frames_equal_warm, &frames_sha,
        first.encode,                             first.sample,                                first.audio,       first.video,
        warm.encode,                              warm.sample,                                 warm.audio,        warm.video,
        float(if (tw) |t| t.get("encode") else null), float(if (tw) |t| t.get("sample") else null), float(if (tw) |t| t.get("audio") else null),
        float(if (tw) |t| t.get("video") else null), pass,
    });
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}
