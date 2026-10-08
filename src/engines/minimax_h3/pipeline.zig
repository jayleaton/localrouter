//! MiniMax H3 text to video + audio end to end in one process, as the twin's `stk_twin/h3/generate.py` runs it (its docstring
//! has the loop): the prompt's token ids, the 32B text encoder, the DiT's refined context, `steps` res_multistep steps from
//! the portable noise on the packed fp32 state [video | audio], the audio VAE and the video VAE. Bit-exact with the twin.
//! Four inputs: the H3 pack (DiT + `tokenizer.json`), the text encoder checkpoint and the two VAE checkpoints as published.
//! Heap-allocated: its parts point at each other.
//!
//! Memory (a 128 GB GB10 holds all of it; weights are uploaded once and never paged): the text encoder's NVFP4 AWQ
//! checkpoint is about 20 GB on the device (int8 embedding, NVFP4 linears, fp32 norms), the DiT's NVFP4 pack about 20 GB,
//! the VAEs about 5 GB (fp16 video 4.85 GB, fp32 audio the rest). Scratch: the DiT's activations for the longest
//! sequence (768x448, 56 frames: 38 k tokens) are about 10 GB, the VAEs' a few GB, the sampler state 7 buffers of
//! [video | audio] (about 14 MB each at that size). So all resident peaks near 60 GB; with `Options.unload_te` the
//! encoder is freed after each encode and loaded again by the next `generate` (about 20 GB less while sampling and
//! decoding, at the cost of re-reading the checkpoint), for a smaller machine or a shared one.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const launch = @import("launch.zig");
const layout = @import("layout.zig");
const dit_mod = @import("dit.zig");
const te32 = @import("te32.zig");
const sampler = @import("sampler.zig");
const vae_audio = @import("vae_audio.zig");
const vae_video = @import("vae_video.zig");
const tok_mod = @import("tokenizer.zig");
const kernels = @import("minimax_kernels");

pub const Probe = dit_mod.Probe;
pub const Io = dit_mod.Io;

pub const Paths = struct { pack: []const u8, te: []const u8, vae_audio: []const u8, vae_video: []const u8 };

/// The largest request this pipeline serves (width and height in pixels, multiples of 32; frames before alignment).
pub const Limits = struct { width: u32 = 768, height: u32 = 448, frames: u32 = 56, tokens: u32 = 512 };

pub const Options = struct {
    /// Free the text encoder after each encode (see the memory note above).
    unload_te: bool = false,
    /// The encoder's RoPE product (the twin's `TextEncoder32.rope_mode`).
    rope_mode: i32 = 0,
};

/// Wall times of the last `generate`, in milliseconds (the stream synchronized at each boundary): the tokens, encoder (and
/// its reload when unloaded), bf16 hand-off and the DiT's text refiner; the steps; the audio VAE; the video VAE.
pub const Times = struct { encode: f64 = 0, sample: f64 = 0, audio: f64 = 0, video: f64 = 0 };

pub const Progress = struct {
    ctx: *anyopaque,
    report: *const fn (ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void,
};

pub const sample_rate = 32000;
pub const audio_channels = 2;
const audio_scale_out: f32 = 0.25; // process_latent_out: the carried audio back to the VAE's scale (1 / 4)

/// What a request turns into (ComfyUI's `temporal_shape` and `_empty_av_latent`).
pub const Shapes = struct {
    frames: u32, // aligned: 17 k + 5, at least 5
    latent_t: u32, // 5 k + 2
    lh: u32, // latent rows (height / 16)
    lw: u32,
    audio_t: u32, // audio latent frames, 40 a second, 800 samples each
    n_video: u64, // 24 * latent_t * lh * lw
    n_audio: u64, // 32 * 2 * audio_t
    frame_bytes: u64, // frames * height * width * 3 (uint8)
    samples: u64, // per channel

    pub fn audioFloats(s: Shapes) u64 {
        return audio_channels * s.samples;
    }
};

/// `align_frame_count(max(5, n))`: up to the next 17 k + 5.
pub fn alignFrames(n: u32) u32 {
    var f = @max(5, n);
    while (f % 17 != 5) f += 1;
    return f;
}

pub fn shapes(width: u32, height: u32, frames_requested: u32) !Shapes {
    if (width == 0 or height == 0 or width % 32 != 0 or height % 32 != 0) return error.SizeNotMultipleOf32;
    const frames = alignFrames(frames_requested);
    const t: u32 = if (frames <= 5) 2 else ((frames - 5) / 17) * 5 + 2;
    const a: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(frames)) / 24.0 * 40.0)); // no .5 ties at 17 k + 5 frames
    const lh = height / 16;
    const lw = width / 16;
    return .{
        .frames = frames,
        .latent_t = t,
        .lh = lh,
        .lw = lw,
        .audio_t = a,
        .n_video = 24 * @as(u64, t) * lh * lw,
        .n_audio = 32 * 2 * @as(u64, a),
        .frame_bytes = @as(u64, frames) * height * width * 3,
        .samples = @as(u64, a) * vae_audio.hop,
    };
}

pub const Result = struct {
    shapes: Shapes,
    width: u32,
    height: u32,
    tokens: u32,
};

pub const Pipeline = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    opts: Options,
    max: Shapes, // of `Limits`
    max_tokens: u32,
    d: cuda.Driver,
    ctx: cuda.Context,
    s: cuda.Stream,
    major: u32,
    minor: u32,
    nv: qi.nvfp4_exec.Nvfp4,
    ops: qi.ops_launch.Ops,
    tk: qi.ops_launch.TeOps,
    hops: launch.Ops,
    kitchen: launch.Kitchen,
    te_ops: te32.Ops,
    a_ops: vae_audio.Ops,
    v_ops: vae_video.Ops,
    pack: qi.pack.Pack,
    te_pack: qi.pack.Pack,
    tok: tok_mod.Tokenizer,
    w: dit_mod.Weights,
    dit: dit_mod.Dit,
    te: ?te32.TextEncoder = null,
    aud: vae_audio.Decoder,
    vid: vae_video.Decoder,
    /// Device buffers: the state x, the two denoised buffers, bf16 model input, bf16 velocities, the bf16 text, the fp16
    /// video latents, the uint8 frames.
    bufs: [8]cuda.DeviceBuffer,
    /// The light ops of a run are reported here when set (the replay's `Probe`): `text_encoder`, `noise`, `step.{i}`,
    /// `latents`, `waveform`, under the twin's names and roles.
    probe: ?Probe = null,
    times: Times = .{},
    /// The last prompt's ids (valid until the next `generate`).
    ids: []u32 = &.{},
    weights_bytes: u64 = 0,

    /// Loads everything for requests up to `limits`.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, paths: Paths, limits: Limits, opts: Options) !*Pipeline {
        const p = try gpa.create(Pipeline);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.io = io;
        p.opts = opts;
        p.probe = null;
        p.times = .{};
        p.ids = &.{};
        p.te = null;
        p.max = try shapes(limits.width, limits.height, limits.frames);
        p.max_tokens = limits.tokens;
        p.d = try cuda.Driver.open();
        errdefer p.d.close();
        p.ctx = try cuda.Context.init(&p.d, 0);
        errdefer p.ctx.deinit();
        var maj: c_int = 0;
        var min: c_int = 0;
        try p.d.check(p.d.api.cuDeviceGetAttribute(&maj, .compute_capability_major, p.ctx.device), "cc");
        try p.d.check(p.d.api.cuDeviceGetAttribute(&min, .compute_capability_minor, p.ctx.device), "cc");
        p.major = @intCast(maj);
        p.minor = @intCast(min);
        p.s = try cuda.Stream.init(&p.d, false); // blocking: ordered with the synchronous uploads
        errdefer p.s.deinit();
        p.nv = try qi.nvfp4_exec.Nvfp4.load(gpa, &p.d, p.major, p.minor);
        errdefer p.nv.unload();
        p.ops = try qi.ops_launch.Ops.load(&p.d, qi.kernels.ops, qi.kernels.attention);
        errdefer p.ops.unload();
        p.tk = try qi.ops_launch.TeOps.load(&p.d, qi.kernels.te, qi.kernels.gemm);
        errdefer p.tk.unload();
        p.hops = try launch.Ops.load(&p.d, kernels.ops);
        errdefer p.hops.unload();
        p.kitchen = try launch.Kitchen.load(&p.d, kernels.kitchen);
        errdefer p.kitchen.unload();
        p.te_ops = try te32.Ops.load(&p.d, kernels.te32);
        errdefer p.te_ops.unload();
        p.a_ops = try vae_audio.Ops.load(&p.d, kernels.vae_audio);
        errdefer p.a_ops.unload();
        p.v_ops = try vae_video.Ops.load(&p.d, kernels.gemm_f16, kernels.vae_video);
        errdefer p.v_ops.unload();

        p.pack = try qi.pack.Pack.open(gpa, io, paths.pack);
        errdefer p.pack.close(io);
        p.te_pack = try qi.pack.Pack.openFile(gpa, io, paths.te);
        errdefer p.te_pack.close(io);
        p.tok = try tok_mod.Tokenizer.load(gpa, io, paths.pack);
        errdefer p.tok.deinit();

        var up = try qi.upload.Uploader.init(&p.d, io, p.s); // one loader on the stream for every component
        defer up.deinit();
        p.w = try dit_mod.Weights.load(gpa, io, &p.d, &p.pack, &p.nv, &up);
        errdefer p.w.deinit();
        const max_s: u32 = limits.tokens + 2 * p.max.audio_t + p.max.latent_t * (p.max.lh / 2) * (p.max.lw / 2);
        const kk: dit_mod.Kernels = .{ .h3 = &p.hops, .kitchen = &p.kitchen, .te = &p.tk, .ops = &p.ops, .nv = &p.nv, .major = p.major, .minor = p.minor };
        p.dit = try dit_mod.Dit.init(gpa, &p.d, kk, &p.w, p.s, max_s, limits.tokens);
        errdefer p.dit.deinit();
        if (!opts.unload_te) try p.loadTeWith(&up);
        errdefer if (p.te) |*t| t.deinit();
        {
            var ap = try qi.pack.Pack.openFile(gpa, io, paths.vae_audio);
            defer ap.close(io);
            p.aud = try vae_audio.Decoder.init(gpa, io, &p.d, &p.a_ops, &p.hops, p.s, &ap, &up, p.max.audio_t);
        }
        errdefer p.aud.deinit();
        p.vid = try vae_video.Decoder.init(gpa, io, &p.d, &p.v_ops, p.s, paths.vae_video, &up);
        errdefer p.vid.deinit();

        const n: u64 = p.max.n_video + p.max.n_audio;
        const sizes = [8]u64{ 4 * n, 4 * n, 4 * n, 2 * n, 2 * n, @as(u64, limits.tokens) * te32.dim * 2, 2 * p.max.n_video, p.max.frame_bytes };
        var made: usize = 0;
        errdefer for (p.bufs[0..made]) |*b| b.free();
        for (&p.bufs, sizes) |*b, len| {
            b.* = try cuda.DeviceBuffer.alloc(&p.d, @intCast(len));
            made += 1;
        }
        try p.s.synchronize();
        p.weights_bytes = p.w.store.bytes + p.aud.store.bytes + p.vid.weightBytes() + (if (p.te) |*t| t.store.bytes else 0);
        return p;
    }

    pub fn destroy(p: *Pipeline) void {
        for (&p.bufs) |*b| b.free();
        p.vid.deinit();
        p.aud.deinit();
        p.unloadTe();
        p.dit.deinit();
        p.w.deinit();
        p.tok.deinit();
        p.te_pack.close(p.io);
        p.pack.close(p.io);
        p.v_ops.unload();
        p.a_ops.unload();
        p.te_ops.unload();
        p.kitchen.unload();
        p.hops.unload();
        p.tk.unload();
        p.ops.unload();
        p.nv.unload();
        p.s.deinit();
        p.ctx.deinit();
        p.d.close();
        if (p.ids.len > 0) p.gpa.free(p.ids);
        const gpa = p.gpa;
        gpa.destroy(p);
    }

    /// The text encoder's load outside `create` (after an unload): its own loader on the stream.
    fn loadTe(p: *Pipeline) !void {
        if (p.te != null) return;
        var up = try qi.upload.Uploader.init(&p.d, p.io, p.s);
        defer up.deinit();
        try p.loadTeWith(&up);
    }

    fn loadTeWith(p: *Pipeline, up: *qi.upload.Uploader) !void {
        if (p.te != null) return;
        p.te = try te32.TextEncoder.init(p.gpa, p.io, &p.d, &p.te_ops, p.s, &p.te_pack, up, p.max_tokens);
    }

    fn unloadTe(p: *Pipeline) void {
        if (p.te) |*t| t.deinit();
        p.te = null;
    }

    /// The sigmas of `steps` steps (`steps + 1` values, the last 0); caller frees.
    pub fn sigmas(a: std.mem.Allocator, steps: u32) ![]f32 {
        const out = try a.alloc(f32, steps + 1);
        sampler.sigmas(layout.shift_v, out);
        return out;
    }

    fn pre(p: *Pipeline, name: []const u8, ins: []const Io) !void {
        if (p.probe) |pr| try pr.before(pr.ctx, name, ins);
    }

    fn post(p: *Pipeline, name: []const u8, outs: []const Io) !void {
        if (p.probe) |pr| try pr.after(pr.ctx, name, outs);
    }

    /// One clip: `frames_out` gets the uint8 frames [F, height, width, 3] and `audio_out` the fp32 waveform [2, A * 800]
    /// at 32 kHz, both sized by `shapes(width, height, frames)`. The frame count snaps up to 17 k + 5.
    pub fn generate(p: *Pipeline, prompt: []const u8, width: u32, height: u32, frames: u32, steps: u32, seed: u64, frames_out: []u8, audio_out: []f32, progress: ?Progress) !Result {
        const sh = try shapes(width, height, frames);
        if (steps == 0) return error.NoSteps;
        if (width > p.max_width() or height > p.max_height() or sh.frames > p.max.frames) return error.RequestTooLarge;
        if (frames_out.len != sh.frame_bytes) return error.BadFrameBuffer;
        if (audio_out.len != sh.audioFloats()) return error.BadAudioBuffer;
        const gpa = p.gpa;
        var clock = Clock{ .io = p.io, .s = p.s };
        const nv: u64 = sh.n_video;
        const n: u64 = nv + sh.n_audio;
        p.dit.probe = null;
        try clock.start();

        // ---- encode: ids, the encoder, the bf16 hand-off, condition_proj and the token refiner
        const ids = try p.tok.ids(gpa, prompt);
        if (p.ids.len > 0) gpa.free(p.ids);
        p.ids = ids;
        if (ids.len > p.max_tokens) return error.PromptTooLong;
        const l: u32 = @intCast(ids.len);
        try p.loadTe();
        try p.pre("text_encoder", &.{});
        const hidden = try p.te.?.forward(ids, p.opts.rope_mode);
        const text = try p.bufs[5].at(0);
        try p.hops.toBf16(p.s, hidden, text, @as(u64, l) * te32.dim);
        try p.dit.prepare(text, l);
        try p.post("text_encoder", &.{
            .{ .role = "hidden", .ptr = hidden, .bytes = @as(u64, l) * te32.dim * 4 },
            .{ .role = "context", .ptr = text, .bytes = @as(u64, l) * te32.dim * 2 },
        });
        if (p.opts.unload_te) {
            try p.s.synchronize();
            p.unloadTe();
        }
        p.times.encode = try clock.lap();

        // ---- the sampler state: portable noise, one stream, video first
        const x = try p.bufs[0].at(0);
        const xa = x + nv * 4; // the audio half
        {
            const z = try gpa.alloc(f32, @intCast(n));
            defer gpa.free(z);
            qi.noise.fill(seed, z);
            try p.dit.upload(x, std.mem.sliceAsBytes(z));
        }
        try p.pre("noise", &.{});
        try p.post("noise", &.{.{ .role = "x", .ptr = x, .bytes = n * 4 }});
        const sig = try sigmas(gpa, steps);
        defer gpa.free(sig);
        const plan = try gpa.alloc(sampler.Step, steps);
        defer gpa.free(plan);
        sampler.plan(sig, plan);

        // ---- the steps
        const xb = try p.bufs[3].at(0); // bf16 model input [video | audio]
        const vel = try p.bufs[4].at(0); // bf16 velocities, packed alike
        const den = [2]u64{ try p.bufs[1].at(0), try p.bufs[2].at(0) };
        var cur: usize = 0; // den[cur] is this step's, den[1 - cur] the previous step's
        var nb: [24]u8 = undefined;
        for (plan, 0..) |st, i| {
            const name = try std.fmt.bufPrint(&nb, "step.{d}", .{i});
            try p.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = n * 4 }});
            try p.hops.toBf16(p.s, x, xb, n);
            try p.dit.step(xb, xb + nv * 2, sig[i], sh.latent_t, sh.lh, sh.lw, sh.audio_t, vel, vel + nv * 2);
            try p.hops.denoise(p.s, x, vel, sig[i], den[cur], n);
            switch (st) {
                .euler => |e| try p.hops.euler32(p.s, x, den[cur], e.sigma, e.dt, n),
                .res2 => |r| try p.hops.res2(p.s, x, den[cur], den[1 - cur], r.e, r.h, r.b1, r.b2, n),
            }
            try p.post(name, &.{
                .{ .role = "vel_v", .ptr = vel, .bytes = nv * 2 },
                .{ .role = "vel_a", .ptr = vel + nv * 2, .bytes = sh.n_audio * 2 },
                .{ .role = "den", .ptr = den[cur], .bytes = n * 4 },
                .{ .role = "x", .ptr = x, .bytes = n * 4 },
            });
            cur = 1 - cur;
            if (progress) |pr| pr.report(pr.ctx, "denoise", @intCast(i + 1), steps);
        }
        try p.hops.scale32(p.s, xa, audio_scale_out, sh.n_audio); // process_latent_out
        try p.pre("latents", &.{});
        try p.post("latents", &.{ .{ .role = "video", .ptr = x, .bytes = nv * 4 }, .{ .role = "audio", .ptr = xa, .bytes = sh.n_audio * 4 } });
        p.times.sample = try clock.lap();

        // ---- the audio VAE: the waveform [2, A * 800] fp32
        const wav = try p.aud.decode(xa, sh.audio_t);
        try p.pre("waveform", &.{});
        try p.post("waveform", &.{.{ .role = "y", .ptr = wav, .bytes = sh.audioFloats() * 4 }});
        try p.s.synchronize();
        try p.d.check(p.d.api.cuMemcpyDtoH_v2(audio_out.ptr, wav, sh.audioFloats() * 4), "cuMemcpyDtoH");
        p.times.audio = try clock.lap();

        // ---- the video VAE: fp16 latents (the VAE's own cast of the fp32 state, round to nearest even), uint8 frames
        {
            const f32s = try gpa.alloc(f32, @intCast(nv));
            defer gpa.free(f32s);
            try p.d.check(p.d.api.cuMemcpyDtoH_v2(f32s.ptr, x, nv * 4), "cuMemcpyDtoH");
            const f16s = try gpa.alloc(f16, @intCast(nv));
            defer gpa.free(f16s);
            for (f16s, f32s) |*o, v| o.* = @floatCast(v);
            const z16 = try p.bufs[6].at(0);
            try p.dit.upload(z16, std.mem.sliceAsBytes(f16s));
            const px = try p.bufs[7].at(0);
            try p.vid.decode(z16, sh.latent_t, sh.lh, sh.lw, px);
            try p.s.synchronize();
            try p.d.check(p.d.api.cuMemcpyDtoH_v2(frames_out.ptr, px, frames_out.len), "cuMemcpyDtoH");
        }
        p.times.video = try clock.lap();
        return .{ .shapes = sh, .width = width, .height = height, .tokens = l };
    }

    fn max_width(p: *const Pipeline) u32 {
        return p.max.lw * 16;
    }

    fn max_height(p: *const Pipeline) u32 {
        return p.max.lh * 16;
    }
};

const Clock = struct {
    io: std.Io,
    s: cuda.Stream,
    t: std.Io.Timestamp = undefined,

    fn start(c: *Clock) !void {
        try c.s.synchronize();
        c.t = std.Io.Clock.awake.now(c.io);
    }
    fn lap(c: *Clock) !f64 {
        try c.s.synchronize();
        const now = std.Io.Clock.awake.now(c.io);
        const us = c.t.durationTo(now).toMicroseconds();
        c.t = now;
        return @as(f64, @floatFromInt(us)) / 1000.0;
    }
};

test "frame counts snap up to 17 k + 5 and the shapes follow" {
    try std.testing.expectEqual(@as(u32, 5), alignFrames(0));
    try std.testing.expectEqual(@as(u32, 5), alignFrames(5));
    try std.testing.expectEqual(@as(u32, 22), alignFrames(6));
    try std.testing.expectEqual(@as(u32, 56), alignFrames(56));
    try std.testing.expectEqual(@as(u32, 73), alignFrames(57));
    const s = try shapes(768, 448, 56);
    try std.testing.expectEqual(@as(u32, 17), s.latent_t);
    try std.testing.expectEqual(@as(u32, 93), s.audio_t); // 56 / 24 * 40
    try std.testing.expectEqual(@as(u64, 24 * 17 * 28 * 48), s.n_video);
    try std.testing.expectEqual(@as(u64, 64 * 93), s.n_audio);
    try std.testing.expectEqual(@as(u64, 56 * 448 * 768 * 3), s.frame_bytes);
    try std.testing.expectEqual(@as(u64, 93 * 800), s.samples);
    try std.testing.expectEqual(@as(u32, 2), (try shapes(64, 64, 1)).latent_t);
    try std.testing.expectError(error.SizeNotMultipleOf32, shapes(770, 448, 56));
}

test "sigmas of eight steps end in zero" {
    const sig = try Pipeline.sigmas(std.testing.allocator, 8);
    defer std.testing.allocator.free(sig);
    try std.testing.expectEqual(@as(usize, 9), sig.len);
    try std.testing.expectEqual(@as(f32, 0), sig[8]);
    try std.testing.expectEqual(@as(f32, 1.0), sig[0]);
}
