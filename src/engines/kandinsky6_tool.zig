//! The `kandinsky6` engine: Kandinsky-6 Pro distilled text to video with its coupled audio on LocalRouter's Zig
//! pipeline (`kandinsky6.pipeline`), one MP4 a request: 832x480 landscape only (the size a request without one gets;
//! any other size or orientation is refused before any GPU work), 121 frames at 24 fps (H.264) and the mono 44.1 kHz
//! track (AAC; the whole 223,232-sample waveform, as upstream's mux writes it). `weights` is the published snapshot
//! directory (HF kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers, revision 1018d582) as downloaded.
//!
//! The model generates audio and video jointly (PiFlow requires both), so `audio: false` is refused rather than silently
//! generated and dropped. It has one native geometry and one schedule (10 PiFlow steps, no guidance): any other size,
//! length, frame rate, step count or guidance is refused before the model loads. Image to video is not offered until its
//! own reference gates pass.
//!
//! Options:
//!   fp32_linear      required: tf32_rna | tf32_truncate | fp32, the arithmetic of the DiT's FP32 linears (upstream runs
//!                    them under torch's `high` matmul policy, whose SM121 algorithm is captured, not assumed). No default.
//!   clip_linear      required: the same choice for the CLIP text encoder's FP32 linears.
//!   qwen_rope_freq   required: the same choice for Qwen2.5-VL's RoPE frequency matmul (`inv_freq @ position_ids`).
//!   qwen_inv_freq    required: a file of 64 little-endian f32, Qwen2.5-VL's `rotary_emb.inv_freq` as captured from
//!                    the native reference on the target (torch computes it at init; it is not re-derived here).
//!   audio            required: the audio decoder's and vocoder's numeric choices (PR #6), every field named:
//!                    {"prepare": {"eps": 1e-4, "reduce": "serial", "den": "fma"},
//!                     "decode": {"acc": "fma", "unscale": "true_div", "silu": "true_div", "residual": "true_div",
//!                                "norm_residual": {"eps": 1e-4, "reduce": {"strided_tree": 256}, "den": "fma"},
//!                                "norm_qkv": {"eps": 1e-4, "reduce": {"serial_fma": {}}, "den": "fma"},
//!                                "attention": "math_candidate"},
//!                     "vocode": {"conv_acc": "fma", "aa_acc": "fma", "avg": "true_div"}}
//!                    (the shape only: the values native uses are what the audio gates determine). `prepare.reduce`
//!                    is serial | lanes8 | lanes8_tail_diagnostic | lanes4_fused; a normalization `reduce` is
//!                    {"serial_fma": {}} | {"serial_mul_add": {}} | {"strided_tree": N} |
//!                    {"torch_reduce": {"sm_count": S, "max_threads_per_sm": T}}, whose limits must equal the
//!                    running device's (refused at load otherwise).
//!   vae_free_bytes   omit it: the video VAE's tile plan is the qualified one (2301403136 bytes, the free memory the
//!                    reference run recorded; upstream derives its tiles from the free memory it sees, and these tiles
//!                    stop 32 columns before the 864-wide generation edge, so the frame is 832x480). Any other figure
//!                    (here or as vae_free_mb) is refused at load: it would write another size, on tiles the decode
//!                    was never proven on.
//!                    The video VAE's arithmetic is not an option either: the decode runs the qualified policies
//!                    (kernels/cuda/kandinsky6/video_vae/QUALIFICATION.md). The trial fields vae_group_norm, vae_conv,
//!                    vae_attn_qkv_bias and vae_attn_sdpa are refused.
//!   resident_mb      the scheduler's estimate of the loaded footprint.
//! The engine holds the GPU only while loaded: give the tool `idle_ttl_s: 0` to release it after every job.

const std = @import("std");
const engine = @import("../engine/engine.zig");
const mp4 = @import("../media/mp4.zig");
const k6 = @import("kandinsky6");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const entry: engine.Entry = .{ .name = "kandinsky6", .capabilities = &.{.text_to_video}, .needs = needs, .create = create, .check = check, .defaults = defaults, .license = license };

/// The weights' licence, unless a tool's config names its own.
pub const license: @import("../config.zig").License = .{
    .name = "MIT License",
    .url = "https://huggingface.co/kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
    .commercial = true,
};

const S = k6.shapes;
const P = k6.pipeline;

/// Loaded weights (74.1 GiB in their compute dtypes) plus the DiT's, text encoders' and VAE's scratch at the native
/// geometry under the default VAE plan; replaced by the measured peak once the GPU gates record it.
const default_resident_mb = 92_000;

fn needs(cfg: *const ToolConfig, req: *const Request) engine.Needs {
    const working: u64 = switch (req.*) {
        .image, .asset3d, .speech, .transcription, .chat, .decision => 0,
        // frames (at most the nominal size) and waveform on the host, the WAV, and ffmpeg's own buffers
        .video => S.sample_frames * S.width * S.height * 3 + S.audio_samples * (4 + 2) + (64 << 20),
    };
    return .{ .resident = engine.optionInt(cfg, "resident_mb", default_resident_mb) << 20, .working = working };
}

/// A video without a size gets the one this model supports, 832x480.
fn defaults(cfg: *const ToolConfig, req: *Request) void {
    _ = cfg;
    switch (req.*) {
        .video => |*v| if (v.width == 0 and v.height == 0) {
            v.width = S.out_width;
            v.height = S.out_height;
        },
        .image, .asset3d, .speech, .transcription, .chat, .decision => {},
    }
}

fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    _ = cfg;
    const v = switch (req.*) {
        .video => |x| x,
        .image, .asset3d, .speech, .transcription, .chat, .decision => {
            why.writeAll("Kandinsky-6 Pro distilled makes videos only") catch {};
            return false;
        },
    };
    if (!S.supported(.{ .width = v.width, .height = v.height, .seconds = v.seconds, .fps = v.fps }, why)) return false;
    if (v.first_frame != null) {
        why.writeAll("Kandinsky-6 Pro distilled image to video is not available yet") catch {};
        return false;
    }
    if (!v.audio) {
        why.writeAll("Kandinsky-6 Pro distilled generates audio jointly with the video; audio cannot be turned off") catch {};
        return false;
    }
    if (v.steps != 0 and v.steps != S.steps) {
        why.print("Kandinsky-6 Pro distilled runs its distilled schedule of {d} steps only; omit steps", .{S.steps}) catch {};
        return false;
    }
    if (v.guidance != 0 and v.guidance != 1) {
        why.writeAll("Kandinsky-6 Pro distilled runs without guidance; omit guidance") catch {};
        return false;
    }
    if (v.negative_prompt.len != 0) {
        why.writeAll("Kandinsky-6 Pro distilled takes no negative prompt") catch {};
        return false;
    }
    return true;
}

/// The options that select arithmetic, parsed strictly (`kandinsky6.pipeline.parseOptions`): a missing or unknown
/// value is a load failure, never a default.
fn options(cfg: *const ToolConfig) !P.Options {
    return P.parseOptions(cfg.options, .production);
}

const Kandinsky6 = struct {
    env: engine.Env,
    cfg: *const ToolConfig,
    p: ?*P.Pipeline = null,

    fn load(ptr: *anyopaque) anyerror!u64 {
        const t: *Kandinsky6 = @ptrCast(@alignCast(ptr));
        const o = options(t.cfg) catch |err| {
            std.log.err("kandinsky6: options.fp32_linear, clip_linear and qwen_rope_freq must name the FP32 arithmetic (tf32_rna, tf32_truncate or fp32), qwen_inv_freq the captured inv_freq file, and audio every audio policy field", .{});
            return err;
        };
        t.p = try P.Pipeline.create(t.env.gpa, t.env.io, t.cfg.weights, o);
        return engine.optionInt(t.cfg, "resident_mb", default_resident_mb) << 20;
    }

    fn generate(ptr: *anyopaque, job: *const engine.Job, sink: engine.Sink, arena: std.mem.Allocator) anyerror!engine.Output {
        const t: *Kandinsky6 = @ptrCast(@alignCast(ptr));
        const v = switch (job.request) {
            .video => |x| x,
            .image, .asset3d, .speech, .transcription, .chat, .decision => return engine.Refused.Refused,
        };
        var why_buf: [256]u8 = undefined;
        var why: std.Io.Writer = .fixed(&why_buf);
        if (!check(t.cfg, &job.request, &why)) return engine.Refused.Refused;
        const p = t.p orelse return error.NotLoaded;
        const start = std.Io.Clock.awake.now(t.env.io);
        const gpa = t.env.gpa;
        const geo = p.videoGeometry();
        const frame_bytes: usize = @intCast(geo.frameBytes());
        const frames = try gpa.alloc(u8, @intCast(geo.rgbBytes()));
        defer gpa.free(frames);
        const wave = try gpa.alloc(f32, S.audio_samples);
        defer gpa.free(wave);
        var bridge: Bridge = .{ .sink = sink };
        // upstream truncates both prompts (max_length, truncation=True), so no prompt is refused for its length
        try p.generate(.{ .prompt = v.prompt, .seed = v.seed }, frames, wave, bridge.progress());

        // upstream's int16 conversion (clip, times 32767, truncate) is writeWav's
        const wav = try std.fs.path.join(arena, &.{ job.dir, "audio.wav" });
        try mp4.writeWav(t.env.io, wav, wave, S.sample_rate, 1);
        const out = try std.fs.path.join(arena, &.{ job.dir, "video.mp4" });
        var w = try mp4.Writer.open(t.env.io, arena, out, .{ .width = geo.width, .height = geo.height, .fps = S.fps, .wav = wav, .shortest = false, .marker = .{ .model = t.cfg.id } });
        errdefer w.abort();
        for (0..geo.frames) |f| {
            try w.frame(frames[f * frame_bytes ..][0..frame_bytes]);
            if (f % S.fps == 0) sink.report(.{ .phase = "encode", .step = @intCast(f), .of = geo.frames });
        }
        try w.finish();
        std.Io.Dir.cwd().deleteFile(t.env.io, wav) catch {};
        const names = try arena.alloc([]const u8, 1);
        names[0] = "video.mp4";
        const ms = start.durationTo(std.Io.Clock.awake.now(t.env.io)).toMilliseconds();
        return .{ .files = names, .seed = v.seed, .ms = @intCast(ms) };
    }

    fn unload(ptr: *anyopaque) void {
        const t: *Kandinsky6 = @ptrCast(@alignCast(ptr));
        if (t.p) |p| p.destroy();
        t.env.gpa.destroy(t);
    }
};

/// The pipeline's phase reports as the tool's progress.
const Bridge = struct {
    sink: engine.Sink,

    fn progress(b: *Bridge) P.Progress {
        return .{ .ctx = b, .report = report };
    }
    fn report(ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void {
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        b.sink.report(.{ .phase = phase, .step = step, .of = of });
    }
};

fn create(env: engine.Env, cfg: *const ToolConfig) anyerror!engine.Engine {
    const t = try env.gpa.create(Kandinsky6);
    t.* = .{ .env = env, .cfg = cfg };
    return .{ .ptr = t, .vtable = &.{ .load = Kandinsky6.load, .generate = Kandinsky6.generate, .unload = Kandinsky6.unload } };
}

test "only the native request passes the check: 832x480, also when the size is omitted" {
    const cfg: ToolConfig = .{ .id = "kandinsky6-pro-distill", .kind = .video, .engine = "kandinsky6", .weights = "/nonexistent" };
    var buf: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    const ok: Request = .{ .video = .{ .prompt = "waves", .width = 832, .height = 480 } };
    try std.testing.expect(check(&cfg, &ok, &why));
    // a request naming no size gets the model's own (engine.fill), which passes
    var dflt: Request = .{ .video = .{ .prompt = "waves" } };
    engine.fill(&cfg, &dflt);
    try std.testing.expectEqual(@as(u32, 832), dflt.video.width);
    try std.testing.expectEqual(@as(u32, 480), dflt.video.height);
    try std.testing.expect(check(&cfg, &dflt, &why));
    // portrait, the generation size, other sizes and the API's generic default are refused before any GPU work
    for ([_][2]u32{ .{ 480, 832 }, .{ 864, 480 }, .{ 640, 480 }, .{ 0, 0 }, .{ 1344, 768 } }) |wh| {
        why = .fixed(&buf);
        const r: Request = .{ .video = .{ .prompt = "waves", .width = wh[0], .height = wh[1] } };
        try std.testing.expect(!check(&cfg, &r, &why));
        try std.testing.expect(std.mem.startsWith(u8, why.buffered(), "Kandinsky-6 Pro distilled supports 832x480 landscape only"));
    }
    why = .fixed(&buf);
    const silent: Request = .{ .video = .{ .prompt = "waves", .width = 832, .height = 480, .audio = false } };
    try std.testing.expect(!check(&cfg, &silent, &why));
    const steps: Request = .{ .video = .{ .prompt = "waves", .width = 832, .height = 480, .steps = 25 } };
    try std.testing.expect(!check(&cfg, &steps, &why));
    const i2v: Request = .{ .video = .{ .prompt = "waves", .width = 832, .height = 480, .first_frame = "0.png" } };
    try std.testing.expect(!check(&cfg, &i2v, &why));
}

test "arithmetic options have no default" {
    const none: ToolConfig = .{ .id = "k", .kind = .video, .engine = "kandinsky6", .weights = "/w" };
    try std.testing.expectError(error.Fp32LinearNotSet, options(&none));
}

test "every arithmetic and audio policy option is parsed whole or refused" {
    const json =
        \\{"fp32_linear": "fp32", "clip_linear": "fp32", "qwen_rope_freq": "fp32", "qwen_inv_freq": "/f",
        \\ "audio": {"prepare": {"eps": 1e-4, "reduce": "lanes4_fused", "den": "fma"},
        \\  "decode": {"acc": "fma", "unscale": "true_div", "silu": "true_div", "residual": "true_div",
        \\    "norm_residual": {"eps": 1e-4, "reduce": {"torch_reduce": {"sm_count": 48, "max_threads_per_sm": 1536}}, "den": "fma"},
        \\    "norm_qkv": {"eps": 1e-4, "reduce": {"serial_fma": {}}, "den": "mul_add"}, "attention": "math_candidate"},
        \\  "vocode": {"conv_acc": "fma", "aa_acc": "mul_add", "avg": "reciprocal"}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    const cfg: ToolConfig = .{ .id = "k", .kind = .video, .engine = "kandinsky6", .weights = "/w", .options = parsed.value };
    const a = (try P.parseOptions(cfg.options, .production)).audio;
    try std.testing.expectEqual(@as(u32, 48), a.decode.norm_residual.reduce.torch_reduce.sm_count);
    try std.testing.expectEqual(.lanes4_fused, a.prepare.reduce);
    try std.testing.expectEqual(.mul_add, a.vocode.aa_acc);
    // torch_reduce's limits are the running device's, never another's
    try a.checkDevice(.{ .sm_count = 48, .max_threads_per_sm = 1536 });
    try std.testing.expectError(error.AudioDeviceLimitsMismatch, a.checkDevice(.{ .sm_count = 84, .max_threads_per_sm = 1536 }));
    // a missing field is refused, not defaulted
    const partial = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"fp32_linear": "fp32", "clip_linear": "fp32", "qwen_rope_freq": "fp32", "qwen_inv_freq": "/f",
        \\ "audio": {"prepare": {"eps": 1e-4, "reduce": "serial", "den": "fma"}, "vocode": {"conv_acc": "fma", "aa_acc": "fma", "avg": "true_div"}}}
    , .{});
    defer partial.deinit();
    const bad: ToolConfig = .{ .id = "k", .kind = .video, .engine = "kandinsky6", .weights = "/w", .options = partial.value };
    try std.testing.expectError(error.AudioPolicyInvalid, P.parseOptions(bad.options, .production));
}
