//! `stk check h3-bench PACK TE_CKPT VAE_AUDIO VAE_VIDEO [--size WxH] [--frames N] [--steps N] [--unload-te] [--runs N] [--out PATH]`:
//! MiniMax H3's speed and memory of the engine alone: no capture, no twin, no comparison. The pipeline is created with Limits
//! sized for the request (so the DiT's scratch and the canvases are those of this clip, not of the tool's maximum), then
//! `--runs` clips (default 2: the first, and a warm one) are generated from one fixed prompt and seed. One JSON line:
//! the shapes (sequence tokens by part), the load time, every run's phase times (first and warm), the weights by component,
//! the engine's own memory (MemAvailable sampled every 50 ms on a thread across the load and the runs, as start / min /
//! peak drop, with the running drop after the load, the first run and the last; cuMemGetInfo before, after the load and
//! after the runs: GB10's reflects the system memory), and the sha256 of every run's frames and waveform (equal runs show
//! as `equal_runs`). With `--out PATH` the last run's clip is written as an MP4 (H.264 + AAC) after the sampler has stopped,
//! so the encoder's memory is not counted. Run it with nothing else on the machine: MemAvailable is the whole system's.

const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("minimax_h3");
const mp4 = @import("../media/mp4.zig");
const memory = @import("../sched/memory.zig");
const pl = h3.pipeline;

const prompt = "A golden retriever runs along a sunny beach at sunset, waves breaking, seagulls calling overhead.";
const seed: u64 = 1234;
const fps = 24;
const max_runs = 8;
const tokens_limit = 512; // the tool's default (Limits.tokens): the text encoder's and the DiT's text scratch

pub const Opts = struct {
    pack: []const u8,
    te: []const u8,
    vae_audio: []const u8,
    vae_video: []const u8,
    width: u32 = 768,
    height: u32 = 448,
    frames: u32 = 56,
    steps: u32 = 8,
    unload_te: bool = false,
    runs: u32 = 2,
    out: ?[]const u8 = null,
};

pub fn parseArgs(args: []const []const u8) !Opts {
    var o: Opts = .{ .pack = "", .te = "", .vae_audio = "", .vae_video = "" };
    var pos: usize = 0;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--unload-te")) {
            o.unload_te = true;
        } else if (std.mem.eql(u8, a, "--size") or std.mem.eql(u8, a, "--frames") or std.mem.eql(u8, a, "--steps") or
            std.mem.eql(u8, a, "--runs") or std.mem.eql(u8, a, "--out"))
        {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            const v = args[i];
            if (std.mem.eql(u8, a, "--size")) {
                const x = std.mem.indexOfScalar(u8, v, 'x') orelse return error.BadSize;
                o.width = try std.fmt.parseInt(u32, v[0..x], 10);
                o.height = try std.fmt.parseInt(u32, v[x + 1 ..], 10);
            } else if (std.mem.eql(u8, a, "--frames")) {
                o.frames = try std.fmt.parseInt(u32, v, 10);
            } else if (std.mem.eql(u8, a, "--steps")) {
                o.steps = try std.fmt.parseInt(u32, v, 10);
            } else if (std.mem.eql(u8, a, "--runs")) {
                o.runs = try std.fmt.parseInt(u32, v, 10);
            } else o.out = v;
        } else if (std.mem.startsWith(u8, a, "--")) {
            return error.UnknownFlag;
        } else {
            switch (pos) {
                0 => o.pack = a,
                1 => o.te = a,
                2 => o.vae_audio = a,
                3 => o.vae_video = a,
                else => return error.TooManyPaths,
            }
            pos += 1;
        }
    }
    if (pos != 4) return error.NeedFourPaths;
    if (o.runs == 0 or o.runs > max_runs or o.steps == 0 or o.frames == 0) return error.BadCount;
    return o;
}

/// MemAvailable sampled every 50 ms on its own thread; `min_kb` only falls. The caller samples at its checkpoints too (the
/// thread may not have run since the last allocation).
const Sampler = struct {
    io: std.Io,
    stop: std.atomic.Value(bool) = .init(false),
    min_kb: std.atomic.Value(u64) = .init(std.math.maxInt(u64)),
    n: std.atomic.Value(u64) = .init(0),
    start_kb: u64 = 0,
    th: ?std.Thread = null,

    fn sample(s: *Sampler) void {
        const b = memory.available() orelse return;
        _ = s.min_kb.fetchMin(b >> 10, .monotonic);
        _ = s.n.fetchAdd(1, .monotonic);
    }

    fn loop(s: *Sampler) void {
        while (!s.stop.load(.acquire)) {
            s.sample();
            std.Io.sleep(s.io, .fromMilliseconds(50), .awake) catch {};
        }
        s.sample();
    }

    fn begin(s: *Sampler) void {
        s.sample();
        s.start_kb = s.min_kb.load(.monotonic);
        s.th = std.Thread.spawn(.{}, loop, .{s}) catch null; // without a thread the checkpoints still sample
    }

    fn end(s: *Sampler) void {
        s.stop.store(true, .release);
        if (s.th) |t| t.join();
        s.th = null;
    }

    /// The running peak drop from the start, in MiB.
    fn dropMib(s: *Sampler) u64 {
        s.sample();
        const m = s.min_kb.load(.monotonic);
        return if (m < s.start_kb) (s.start_kb - m) >> 10 else 0;
    }
};

/// Device memory in use by every process (cuMemGetInfo), through a context of its own; null without a driver.
fn gpuUsedMib() ?u64 {
    var d = cuda.Driver.open() catch return null;
    defer d.close();
    var c = cuda.Context.init(&d, 0) catch return null;
    defer c.deinit();
    const m = c.memInfo() catch return null;
    return (m.total - m.free) >> 20;
}

fn hex(h: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(h, .lower);
}

fn sha(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return hex(h);
}

const Run = struct { t: pl.Times, frames_sha: [64]u8, wav_sha: [64]u8 };

fn writeTimes(w: *std.Io.Writer, t: pl.Times) !void {
    try w.print("{{\"encode\": {d:.1}, \"sample\": {d:.1}, \"audio\": {d:.1}, \"video\": {d:.1}, \"total\": {d:.1}}}", .{ t.encode, t.sample, t.audio, t.video, t.encode + t.sample + t.audio + t.video });
}

/// A JSON string (a path or an error name: no escapes needed beyond quotes and backslashes) or null.
fn writeOptString(w: *std.Io.Writer, s: ?[]const u8) !void {
    const v = s orelse return w.writeAll("null");
    try w.writeByte('"');
    for (v) |c| {
        if (c == '"' or c == '\\') try w.writeByte('\\');
        try w.writeByte(c);
    }
    try w.writeByte('"');
}

/// The last run's clip as an MP4 at `path` (the audio through a WAV beside it, removed after).
fn writeMp4(io: std.Io, gpa: std.mem.Allocator, path: []const u8, o: Opts, sh: pl.Shapes, px: []const u8, wav: []const f32) !void {
    const wav_path = try std.fmt.allocPrint(gpa, "{s}.wav", .{path});
    defer gpa.free(wav_path);
    const inter = try gpa.alloc(f32, wav.len); // [2, n] planar to interleaved
    defer gpa.free(inter);
    const n: usize = @intCast(sh.samples);
    for (0..n) |i| for (0..pl.audio_channels) |c| {
        inter[i * pl.audio_channels + c] = wav[c * n + i];
    };
    try mp4.writeWav(io, wav_path, inter, pl.sample_rate, pl.audio_channels);
    defer std.Io.Dir.cwd().deleteFile(io, wav_path) catch {};
    var w = try mp4.Writer.open(io, gpa, path, .{ .width = o.width, .height = o.height, .fps = fps, .wav = wav_path });
    errdefer w.abort();
    const fb: usize = @as(usize, o.width) * o.height * 3;
    for (0..sh.frames) |f| try w.frame(px[f * fb ..][0..fb]);
    try w.finish();
}

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    const o = parseArgs(args) catch |err| {
        std.debug.print("h3-bench: {s}\nusage: stk check h3-bench PACK TE_CKPT VAE_AUDIO VAE_VIDEO [--size 768x448] [--frames N] [--steps 8] " ++
            "[--unload-te] [--runs 2] [--out PATH.mp4]\n", .{@errorName(err)});
        return 2;
    };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sh = try pl.shapes(o.width, o.height, o.frames);

    const gpu_before = gpuUsedMib();
    var smp: Sampler = .{ .io = io };
    smp.begin();
    defer smp.end();

    const t0 = std.Io.Clock.awake.now(io);
    const p = try pl.Pipeline.create(gpa, io, .{ .pack = o.pack, .te = o.te, .vae_audio = o.vae_audio, .vae_video = o.vae_video }, .{
        .width = o.width,
        .height = o.height,
        .frames = o.frames,
        .tokens = tokens_limit,
    }, .{ .unload_te = o.unload_te });
    defer p.destroy();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    const drop_load = smp.dropMib();
    const gpu_load = blk: {
        const m = p.ctx.memInfo() catch break :blk null;
        break :blk (m.total - m.free) >> 20;
    };
    // weights by component (the encoder's is 0 here when it is unloaded between requests: it is counted while resident only)
    const w_dit = p.w.store.bytes;
    const w_aud = p.aud.store.bytes;
    const w_vid = p.vid.weightBytes();
    const w_te: u64 = if (p.te) |*t| t.store.bytes else 0;

    const px = try a.alloc(u8, @intCast(sh.frame_bytes));
    const wav = try a.alloc(f32, @intCast(sh.audioFloats()));
    var runs: [max_runs]Run = undefined;
    var drop_first: u64 = 0;
    var text_tokens: u32 = 0;
    var gen_ms: [max_runs]i64 = undefined;
    for (0..o.runs) |r| {
        const g0 = std.Io.Clock.awake.now(io);
        const res = try p.generate(prompt, o.width, o.height, o.frames, o.steps, seed, px, wav, null);
        gen_ms[r] = g0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        text_tokens = res.tokens;
        runs[r] = .{ .t = p.times, .frames_sha = sha(px), .wav_sha = sha(std.mem.sliceAsBytes(wav)) };
        if (r == 0) drop_first = smp.dropMib();
    }
    const drop_all = smp.dropMib();
    const gpu_after = blk: {
        const m = p.ctx.memInfo() catch break :blk null;
        break :blk (m.total - m.free) >> 20;
    };
    smp.end(); // before the encoder: the MP4's memory is not the engine's
    var equal_runs = true;
    for (runs[1..o.runs]) |q| equal_runs = equal_runs and std.mem.eql(u8, &q.frames_sha, &runs[0].frames_sha) and std.mem.eql(u8, &q.wav_sha, &runs[0].wav_sha);

    // a failed encoder (no ffmpeg on the PATH) costs the MP4, not the measurements: it is reported in the line
    var mp4_ms: ?i64 = null;
    var mp4_error: ?[]const u8 = null;
    if (o.out) |path| {
        const m0 = std.Io.Clock.awake.now(io);
        if (writeMp4(io, gpa, path, o, sh, px, wav)) |_| {
            mp4_ms = m0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        } else |err| mp4_error = @errorName(err);
    }

    const seq: u64 = @as(u64, text_tokens) + 2 * @as(u64, sh.audio_t) + @as(u64, sh.latent_t) * (sh.lh / 2) * (sh.lw / 2);
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.print("{{\"h3_bench\": {{\"gpu\": \"sm_{d}{d}\", \"size\": \"{d}x{d}\", \"frames\": {d}, \"steps\": {d}, \"unload_te\": {}, \"runs\": {d}, " ++
        "\"shapes\": {{\"latent_t\": {d}, \"latent_grid\": \"{d}x{d}\", \"audio_t\": {d}, \"text_tokens\": {d}, \"video_tokens\": {d}, \"audio_tokens\": {d}, \"seq_tokens\": {d}}}, " ++
        "\"load_ms\": {d}, \"weights_bytes\": {d}, \"weights\": {{\"dit\": {d}, \"te\": {d}, \"vae_audio\": {d}, \"vae_video\": {d}}}, ", .{
        p.major,       p.minor,    o.width, o.height, sh.frames,  o.steps,       o.unload_te, o.runs,
        sh.latent_t,   sh.lh,      sh.lw,   sh.audio_t, text_tokens, @as(u64, sh.latent_t) * (sh.lh / 2) * (sh.lw / 2), 2 * @as(u64, sh.audio_t), seq,
        load_ms,       p.weights_bytes, w_dit, w_te,   w_aud,    w_vid,
    });
    try w.writeAll("\"first_ms\": ");
    try writeTimes(w, runs[0].t);
    try w.writeAll(", \"warm_ms\": ");
    if (o.runs > 1) try writeTimes(w, runs[o.runs - 1].t) else try w.writeAll("null");
    try w.writeAll(", \"run_ms\": [");
    for (0..o.runs) |r| try w.print("{s}{d}", .{ if (r > 0) ", " else "", gen_ms[r] });
    try w.print("], \"mem\": {{\"memavail_start_mib\": {d}, \"memavail_min_mib\": {d}, \"samples\": {d}, \"peak_drop_mib\": {d}, " ++
        "\"drop_after_load_mib\": {d}, \"drop_after_first_mib\": {d}, \"gpu_used_before_mib\": {?d}, \"gpu_used_after_load_mib\": {?d}, \"gpu_used_after_runs_mib\": {?d}}}, ", .{
        smp.start_kb >> 10, smp.min_kb.load(.monotonic) >> 10, smp.n.load(.monotonic), drop_all, drop_load, drop_first, gpu_before, gpu_load, gpu_after,
    });
    try w.print("\"frames_sha256\": \"{s}\", \"waveform_sha256\": \"{s}\", \"equal_runs\": {}, \"mp4_ms\": {?d}, \"mp4\": ", .{
        &runs[o.runs - 1].frames_sha, &runs[o.runs - 1].wav_sha, equal_runs, mp4_ms,
    });
    try writeOptString(w, o.out);
    try w.writeAll(", \"mp4_error\": ");
    try writeOptString(w, mp4_error);
    try w.writeAll("}}\n");
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (equal_runs) 0 else 1;
}

test "arguments: four paths, then flags in any order" {
    const o = try parseArgs(&.{ "p", "t", "va", "vv", "--size", "512x320", "--frames", "124", "--unload-te", "--runs", "3", "--out", "x.mp4" });
    try std.testing.expectEqualStrings("vv", o.vae_video);
    try std.testing.expectEqual(@as(u32, 512), o.width);
    try std.testing.expectEqual(@as(u32, 320), o.height);
    try std.testing.expectEqual(@as(u32, 124), o.frames);
    try std.testing.expect(o.unload_te);
    try std.testing.expectEqual(@as(u32, 3), o.runs);
    try std.testing.expectEqualStrings("x.mp4", o.out.?);
    const d = try parseArgs(&.{ "--steps", "4", "p", "t", "va", "vv" });
    try std.testing.expectEqual(@as(u32, 768), d.width);
    try std.testing.expectEqual(@as(u32, 4), d.steps);
    try std.testing.expectError(error.NeedFourPaths, parseArgs(&.{ "p", "t", "va" }));
    try std.testing.expectError(error.UnknownFlag, parseArgs(&.{ "p", "t", "va", "vv", "--bogus" }));
    try std.testing.expectError(error.MissingValue, parseArgs(&.{ "p", "t", "va", "vv", "--runs" }));
    try std.testing.expectError(error.BadCount, parseArgs(&.{ "p", "t", "va", "vv", "--runs", "0" }));
    _ = &run; // analyzed, so a type error in the bench fails the tests
}

test "the 5 s clip's shapes: 124 frames, 37 latent frames, 12.4 k video tokens" {
    const s = try pl.shapes(768, 448, 120);
    try std.testing.expectEqual(@as(u32, 124), s.frames);
    try std.testing.expectEqual(@as(u32, 37), s.latent_t);
    try std.testing.expectEqual(@as(u32, 207), s.audio_t);
    try std.testing.expectEqual(@as(u64, 37 * 14 * 24), @as(u64, s.latent_t) * (s.lh / 2) * (s.lw / 2));
}
