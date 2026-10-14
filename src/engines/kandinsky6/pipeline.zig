//! Kandinsky-6 Pro distilled, text to video with its coupled audio, on one GPU: the published snapshot loaded into the
//! components' own layouts, then for each request (pipeline/pipeline.py and piflow_sampler.py at upstream eac06e39)
//!   text      Qwen2.5-VL hidden states [1024, 3584] bf16 + key mask, CLIP pooled [768] cast to bf16
//!   noise     torch CUDA randn: video [31, 60, 108, 16] at seed s, audio [218, 40] at seed s + 1 (bf16)
//!   sampler   10 PiFlow steps: visual input, one fused DiT call (both modalities), the DX policy rollout of each
//!   video     the f32 latent through PR #4's decoder (its own loader, `video_vae/weights.zig`) to uint8 RGB at the VAE
//!             plan's emitted size (`videoGeometry`: 121 x 480 x 864, or narrower when the plan's tiles stop short)
//!   audio     PR #6's decoder (weights prepared on the host at load) and vocoder to the 223,232-sample waveform
//! The position tables (RoPE and the time embeddings' frequencies) depend only on the geometry and are built at load.
//!
//! `create` asks `admission` first: production loads only when every contract is qualified, diagnostic (the check
//! gates) whenever none is unresolved. Precision choices native leaves to run-time libraries are required options; none
//! has a default.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const tokenizer = @import("tokenizer");
const kmod = @import("kandinsky6_kernels");
const dit_kmod = @import("kandinsky6_dit_kernels");
const shapes = @import("shapes.zig");
const weights = @import("weights.zig");
const rope = @import("rope.zig");
const noise = @import("noise.zig");
const piflow = @import("piflow.zig");
const central = @import("central.zig");
const vae_gpu = @import("vae_gpu.zig");
const admission = @import("admission.zig");
const prompt = @import("text/prompt.zig");
const qwen25 = @import("text/qwen25.zig");
const clip = @import("text/clip.zig");
const dit = @import("dit/block.zig");
const vae_decode = @import("video_vae/decode.zig");
const vae_tiles = @import("video_vae/tiles.zig");
const vae_weights = @import("video_vae/weights.zig");
const vae_launch = @import("video_vae/launch.zig");
const audio_vae = @import("audio/vae.zig");
const vocoder = @import("audio/vocoder.zig");
const audio_launch = @import("audio/launch.zig");
const rec_mod = @import("rec.zig");
const Rec = rec_mod.Rec;
const RT = rec_mod.Tensor;

const View = dit.layout.View;

/// The arithmetic of an FP32 linear under native's `high` matmul policy (the executed algorithm is captured on the
/// target, never assumed).
pub const Fp32Linear = enum { tf32_rna, tf32_truncate, fp32 };

pub const Options = struct {
    /// What the pipeline is admitted for (the tool: production; the check gates: diagnostic).
    use: admission.Use,
    /// The DiT's FP32 linears (modulation).
    fp32_linear: Fp32Linear,
    /// CLIP's FP32 linears.
    clip_linear: qwen25.Fp32Math,
    /// Qwen's RoPE frequency matmul `inv_freq @ position_ids`.
    qwen_rope_freq: Fp32Linear,
    /// `model.language_model.rotary_emb.inv_freq` as native holds it: a file of 64 little-endian f32, captured on the
    /// target (qwen25.invFreqCandidate is a host-test value, never used here).
    qwen_inv_freq: []const u8,
    /// The free-memory figure the video VAE's tile plan uses, in bytes (configuration, never a probe). It decides the
    /// tiles, and with them the decoded bits and the emitted width and height.
    vae_free_bytes: u64,
    /// The audio decoder's and vocoder's numeric choices (PR #6), each named, none defaulted.
    audio: AudioPolicy,
    /// Which components load (the component gates load only what a stage needs; the tool and e2e load all).
    parts: Parts = .all,
    hooks: Hooks = .{},
};

pub const Parts = struct {
    text: bool,
    dit: bool,
    video: bool,
    audio: bool,

    pub const all: Parts = .{ .text = true, .dit = true, .video = true, .audio = true };
    pub const none: Parts = .{ .text = false, .dit = false, .video = false, .audio = false };
};

/// Options from a JSON object (a tool config's `options`, or a gate's options file), parsed strictly: every arithmetic
/// and policy field is required, an unknown or missing value is an error, never a default. The VAE's free-memory figure
/// is `vae_free_bytes` (exact bytes, as a reference run's `note_free_bytes` records it) or `vae_free_mb` (MiB), not
/// both; absent, the qualified plan's 2301403136 bytes (`video_vae/launch.zig` qualified.free_bytes; configuration, not
/// arithmetic). In production (the tool) only that plan is accepted: it emits the one supported size, 832x480; the
/// diagnostic gates may replay another. The video VAE's arithmetic is not an option: the decode always runs the qualified policies
/// (kernels/cuda/kandinsky6/video_vae/QUALIFICATION.md), and the trial fields that once chose them (`vae_group_norm`,
/// `vae_conv`, `vae_attn_qkv_bias`, `vae_attn_sdpa`) are refused rather than ignored.
pub fn parseOptions(value: ?std.json.Value, use: admission.Use) !Options {
    const obj = if (value) |v| (if (v == .object) v.object else return error.OptionsNotAnObject) else return error.Fp32LinearNotSet;
    const arith = struct {
        fn get(comptime T: type, o: std.json.ObjectMap, key: []const u8, missing: anyerror) anyerror!T {
            const v = o.get(key) orelse return missing;
            if (v != .string) return missing;
            return std.meta.stringToEnum(T, v.string) orelse missing;
        }
    }.get;
    const inv = obj.get("qwen_inv_freq") orelse return error.QwenInvFreqNotSet;
    if (inv != .string or inv.string.len == 0) return error.QwenInvFreqNotSet;
    const vae_free_bytes = try vaeFreeBytes(obj);
    try refuseRemovedVaeFields(obj);
    // production writes the one supported size, 832x480: the qualified plan only (the gates may replay others)
    if (use == .production and vae_free_bytes != vae_launch.qualified.free_bytes) return error.VaePlanNotSupported;
    const audio = obj.get("audio") orelse return error.AudioPolicyNotSet;
    var mem: [1024]u8 = undefined; // the policy holds no slices; parsing needs scratch only
    var fba: std.heap.FixedBufferAllocator = .init(&mem);
    return .{
        .use = use,
        .fp32_linear = try arith(Fp32Linear, obj, "fp32_linear", error.Fp32LinearNotSet),
        .clip_linear = try arith(qwen25.Fp32Math, obj, "clip_linear", error.ClipLinearNotSet),
        .qwen_rope_freq = try arith(Fp32Linear, obj, "qwen_rope_freq", error.QwenRopeFreqNotSet),
        .qwen_inv_freq = inv.string,
        .vae_free_bytes = vae_free_bytes,
        .audio = std.json.parseFromValueLeaky(AudioPolicy, fba.allocator(), audio, .{}) catch return error.AudioPolicyInvalid,
    };
}

/// The trial fields of the video VAE's arithmetic, removed: the decode runs the qualified policies only.
const removed_vae_fields = [_][]const u8{ "vae_group_norm", "vae_conv", "vae_attn_qkv_bias", "vae_attn_sdpa" };

fn refuseRemovedVaeFields(obj: std.json.ObjectMap) !void {
    for (removed_vae_fields) |k| if (obj.get(k) != null) return error.VaeArithmeticNotAnOption;
}

fn vaeFreeBytes(obj: std.json.ObjectMap) !u64 {
    const bytes = obj.get("vae_free_bytes");
    const mb = obj.get("vae_free_mb");
    if (bytes != null and mb != null) return error.VaeFreeBytesAndMb;
    if (bytes) |v| {
        if (v != .integer or v.integer <= 0) return error.VaeFreeBytesInvalid;
        const n: u64 = @intCast(v.integer);
        return n;
    }
    const v = mb orelse return vae_launch.qualified.free_bytes;
    const n: u64 = if (v == .integer and v.integer > 0) @intCast(v.integer) else return error.VaeFreeMbInvalid;
    return std.math.mul(u64, n, 1 << 20) catch return error.VaeFreeMbInvalid;
}

/// PR #6's explicit policies: the decoder weights' host preparation and the two forwards. The values native uses on
/// the target are what the audio gates determine.
pub const AudioPolicy = struct {
    prepare: audio_vae.PrepareParams,
    decode: audio_launch.DecodePolicy,
    vocode: audio_launch.VocodePolicy,

    /// A `torch_reduce` normalization splits by the device limits it names; PR #6 requires them read from the running
    /// device, so a policy naming any other device is refused before the audio weights load.
    pub fn checkDevice(p: AudioPolicy, dev: noise.Device) error{AudioDeviceLimitsMismatch}!void {
        for ([_]audio_launch.DeviceReduce{ p.decode.norm_residual.reduce, p.decode.norm_qkv.reduce }) |r| switch (r) {
            .torch_reduce => |l| if (l.sm_count != dev.sm_count or l.max_threads_per_sm != dev.max_threads_per_sm)
                return error.AudioDeviceLimitsMismatch,
            .serial_fma, .serial_mul_add, .strided_tree => {},
        };
    }
};

/// Per-op observers for the check gates; null in production.
pub const Hooks = struct {
    audio_decode: ?audio_launch.DecodeHook = null,
    /// The vocoder's stage observer, used only by `replayVocoder` (the audio chain never traces the vocoder).
    audio_vocode: ?audio_launch.DecodeHook = null,
    text: ?qwen25.Hook = null,
    dit: ?dit.Hook = null,
    /// The bit gate's capture: the pipeline records its own ops (the oracle's names and roles) into it.
    rec: ?*Rec = null,
    /// Diagnostic, with `rec`: the video decode's QUALIFICATION.md checkpoints as `k6.aux.vae.trace.<name>`: the
    /// decoder's in pass `vae_trace_pass` (0: the first spatial tile of the first temporal tile), then the first blends
    /// and tile assembly.
    /// Recording synchronizes the stream after each; no launch changes.
    vae_trace: bool = false,
    /// With `vae_trace`: the decoder pass whose checkpoints are recorded (0 the first tile; passes count in the
    /// reference's order, spatial tiles row-major within each temporal tile). decode.input and the blends / tile
    /// assembly stay at their first occurrence.
    vae_trace_pass: u32 = 0,
};

pub const Progress = struct {
    ctx: *anyopaque,
    report: *const fn (ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void,

    fn at(p: Progress, phase: []const u8, step: u32, of: u32) void {
        p.report(p.ctx, phase, step, of);
    }
};

pub const Request = struct {
    prompt: []const u8,
    seed: u64,
    /// The prompt's line in a gate's prompt file: the label the oracle records (`prompt` attrs); 0 otherwise.
    prompt_index: u32 = 0,
};

const mel_frames: u32 = 2 * shapes.audio_rows; // the decoder doubles the latent rows
const mel_bytes: u64 = @as(u64, audio_vae.mel_channels) * mel_frames * 4;
const wave_bytes: u64 = shapes.audio_samples * 4;
// the oracle's recorded shapes (tools/twin/stk_twin/kandinsky6/README.md "Ops")
const video_shape = [4]u64{ shapes.latent_frames, shapes.latent_h, shapes.latent_w, shapes.latent_channels };
const audio_shape = [2]u64{ shapes.audio_rows, shapes.audio_channels };
const video_batch_shape = [5]u64{ 1, shapes.latent_frames, shapes.latent_h, shapes.latent_w, shapes.latent_channels };
const audio_batch_shape = [3]u64{ 1, shapes.audio_rows, shapes.audio_channels };
const pred_v_shape = [6]u64{ 1, shapes.n_grid, shapes.latent_frames, shapes.latent_h, shapes.latent_w, shapes.latent_channels };
const pred_a_shape = [4]u64{ 1, shapes.n_grid, shapes.audio_rows, shapes.audio_channels };
const mel_shape = [3]u64{ 1, audio_vae.mel_channels, mel_frames };
const wave_shape = [3]u64{ 1, 1, shapes.audio_samples };
/// Rows of upstream's RoPE `args` buffers (max_pos): RoPE3D 128 per axis, RoPE1D 2048.
const rope_buffer_rows_3d: usize = 128;
const rope_buffer_rows_1d: usize = 2048;
const dit_shape: dit.layout.Shape = .preset(shapes.text_rows);
const rope_cols: u64 = 4 * rope.pairs; // [rows, 64, 4] f32

/// Device buffers that live as long as the pipeline, as offsets into one allocation.
const Layout = struct {
    text: u64, // [1024, 3584] bf16
    mask: u64, // u8 [1024]
    pooled_f32: u64, // [768]
    pooled: u64, // [768] bf16
    noise_v: u64, // bf16 [video_elems]
    noise_a: u64,
    state_v: u64, // f32
    state_a: u64,
    video_in: u64, // [200,880, 33], f32 at most
    pred_v: u64, // bf16 [10][video_elems]
    pred_a: u64,
    times: u64, // f32 [steps]
    freqs_v: u64, // f32 [2048]
    freqs_a: u64, // f32 [1024]
    rope_visual: u64, // f32 [N, 256]
    rope_audio: u64, // [A, 256]
    rope_text: u64, // [L, 256], both text stacks
    rgb: u64, // uint8 [121][H][W][3] at the VAE plan's emitted size
    mel: u64, // f32 [1, 128, 436]
    wave: u64, // f32 [1, 1, 223,232]
    work: u64, // scratch shared in turn by the encoders, the RoPE arguments, the DiT and the VAE
    work_bytes: u64,
    total: u64,

    fn init(work_bytes: u64, rgb_bytes: u64) Layout {
        var at: u64 = 0;
        const put = struct {
            fn f(cur: *u64, n: u64) u64 {
                const o = cur.*;
                cur.* = std.mem.alignForward(u64, cur.* + n, 256);
                return o;
            }
        }.f;
        const N = dit_shape.tokens();
        var l: Layout = undefined;
        l.text = put(&at, @as(u64, shapes.text_rows) * shapes.text_dim * 2);
        l.mask = put(&at, shapes.text_rows);
        l.pooled_f32 = put(&at, shapes.clip_dim * 4);
        l.pooled = put(&at, shapes.clip_dim * 2);
        l.noise_v = put(&at, shapes.video_elems * 2);
        l.noise_a = put(&at, shapes.audio_elems * 2);
        l.state_v = put(&at, shapes.video_elems * 4);
        l.state_a = put(&at, shapes.audio_elems * 4);
        l.video_in = put(&at, central.video_rows * shapes.video_in_channels * 4);
        l.pred_v = put(&at, @as(u64, shapes.n_grid) * shapes.video_elems * 2);
        l.pred_a = put(&at, @as(u64, shapes.n_grid) * shapes.audio_elems * 2);
        l.times = put(&at, shapes.steps * 4);
        l.freqs_v = put(&at, rope.time_freqs_v * 4);
        l.freqs_a = put(&at, rope.time_freqs_a * 4);
        l.rope_visual = put(&at, N * rope_cols * 4);
        l.rope_audio = put(&at, @as(u64, shapes.audio_rows) * rope_cols * 4);
        l.rope_text = put(&at, @as(u64, shapes.text_rows) * rope_cols * 4);
        l.rgb = put(&at, rgb_bytes);
        l.mel = put(&at, mel_bytes);
        l.wave = put(&at, wave_bytes);
        l.work = put(&at, work_bytes);
        l.work_bytes = work_bytes;
        l.total = at;
        return l;
    }
};

fn math(comptime T: type, x: anytype) T {
    return switch (x) {
        inline else => |t| @field(T, @tagName(t)),
    };
}

pub const Pipeline = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    opts: Options,
    d: cuda.Driver,
    ctx: cuda.Context,
    s: cuda.Stream,
    dev: noise.Device,
    ops: central.Ops,
    text_k: qwen25.Kernels,
    dit_k: dit.launch.Kernels,
    vae: vae_gpu.Gpu,
    audio_ops: audio_launch.Ops,
    qwen_tok: ?tokenizer.Tokenizer = null,
    clip_tok: ?prompt.clip_tokenizer.ClipTokenizer = null,
    dit_store: weights.Store,
    qwen_store: weights.Store,
    clip_store: weights.Store,
    audio_store: weights.Store,
    vae_buf: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    dit_w: dit.Weights = undefined,
    text_blocks: [2 * dit.layout.Config.distill.num_text_blocks]dit.TextBlock = undefined,
    blocks: [dit.layout.Config.distill.num_visual_blocks]dit.FusedBlock = undefined,
    qwen_w: qwen25.Weights = undefined,
    clip_w: clip.Weights = undefined,
    vae_dec: vae_decode.Decoder = undefined,
    audio_dec: audio_vae.Decoder,
    audio_w: audio_vae.DecoderWeights = undefined,
    voc: vocoder.Vocoder,
    voc_w: vocoder.VocoderWeights = undefined,
    vae_plan: vae_tiles.DecodePlan,
    dit_plan: dit.Plan,
    inv_freq: [64]f32 = undefined,
    buf: cuda.DeviceBuffer,
    lay: Layout,
    steps: [shapes.steps]piflow.Step,
    /// Contracts not yet qualified when admitted (0 in production; a diagnostic run's label otherwise).
    open_contracts: usize,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, root: []const u8, opts: Options) !*Pipeline {
        const open_contracts = try admission.admit(opts.use);
        const parts = opts.parts;
        const p = try gpa.create(Pipeline);
        errdefer gpa.destroy(p);
        var tmp: std.heap.ArenaAllocator = .init(gpa);
        defer tmp.deinit();
        const a = tmp.allocator();
        p.* = .{
            .gpa = gpa,
            .io = io,
            .opts = opts,
            .open_contracts = open_contracts,
            .d = undefined,
            .ctx = undefined,
            .s = undefined,
            .dev = undefined,
            .ops = undefined,
            .text_k = undefined,
            .dit_k = undefined,
            .vae = undefined,
            .audio_ops = undefined,
            .dit_store = undefined,
            .qwen_store = undefined,
            .clip_store = undefined,
            .audio_store = undefined,
            .audio_dec = try audio_vae.Decoder.init(1, shapes.audio_rows),
            .voc = try vocoder.Vocoder.init(1, mel_frames),
            .vae_plan = try vae_gpu.plan(shapes.latent_frames, shapes.latent_h, shapes.latent_w, opts.vae_free_bytes),
            .dit_plan = try dit.Plan.init(.distill, dit_shape, dit.Plan.default_ffn_rows),
            .buf = undefined,
            .lay = undefined,
            .steps = undefined,
        };
        piflow.schedule(.{}, &p.steps);
        if (parts.text) {
            p.inv_freq = try readInvFreq(io, opts.qwen_inv_freq);
            p.qwen_tok = try tokenizer.loadTokenizer(io, gpa, try std.fs.path.join(a, &.{ root, weights.paths.qwen_tokenizer }));
            p.clip_tok = try .load(gpa, io, try std.fs.path.join(a, &.{ root, weights.paths.clip_tokenizer }));
        }
        errdefer p.closeTokenizers();

        p.d = try cuda.Driver.open();
        errdefer p.d.close();
        p.ctx = try cuda.Context.init(&p.d, 0);
        errdefer p.ctx.deinit();
        p.s = try cuda.Stream.init(&p.d, false); // blocking: ordered with the synchronous uploads and downloads
        errdefer p.s.deinit();
        if (opts.hooks.rec) |r| r.attach(&p.d, p.s);
        p.dev = try central.device(&p.ctx);
        if (parts.audio) try opts.audio.checkDevice(p.dev);
        p.ops = try central.Ops.load(&p.d, kmod.central);
        errdefer p.ops.unload();
        p.text_k = try qwen25.Kernels.load(&p.d, .{ .text = kmod.text, .qwen_te = qi.kernels.te, .qwen_gemm = qi.kernels.gemm, .text_attention = kmod.text_attention, .clip_linear = kmod.clip_linear });
        errdefer p.text_k.unload();
        p.dit_k = try dit.launch.Kernels.load(&p.d, .{ .gemm = dit_kmod.gemm, .attention = dit_kmod.attention, .flash = dit_kmod.flash, .ops = dit_kmod.ops });
        errdefer p.dit_k.unload();
        p.vae = try vae_gpu.Gpu.load(&p.d, kmod.vae_conv, kmod.vae_ops, p.s);
        errdefer p.vae.unload();
        p.audio_ops = try audio_launch.Ops.load(&p.d, kmod.audio, kmod.audio_conv);
        errdefer p.audio_ops.unload();

        var up = try qi.upload.Uploader.init(&p.d, io, p.s);
        defer up.deinit();
        p.dit_store = .init(&p.d, gpa);
        errdefer p.dit_store.deinit();
        p.qwen_store = .init(&p.d, gpa);
        errdefer p.qwen_store.deinit();
        p.clip_store = .init(&p.d, gpa);
        errdefer p.clip_store.deinit();
        p.audio_store = .init(&p.d, gpa);
        errdefer p.audio_store.deinit();
        if (parts.dit) {
            var src = try weights.Source.openFile(gpa, io, root, weights.paths.transformer);
            defer src.close(io);
            var l: Loader = .{ .store = &p.dit_store, .io = io, .up = &up, .src = &src, .gpa = gpa, .cast = .bf16 };
            defer l.deinit();
            try dit.layout.forEachTensor(.distill, &l);
            p.dit_w = try dit.bindWeights(.distill, &p.dit_store, &p.text_blocks, &p.blocks);
        }
        if (parts.text) {
            {
                var src = try weights.Source.openIndex(gpa, io, root, weights.paths.text_encoder_dir, weights.paths.text_encoder_index);
                defer src.close(io);
                var l: Loader = .{ .store = &p.qwen_store, .io = io, .up = &up, .src = &src, .gpa = gpa, .cast = .as_stored };
                defer l.deinit();
                try qwen25.forEachTensor(&l);
                p.qwen_w = try qwen25.bindWeights(&p.qwen_store);
            }
            var src = try weights.Source.openFile(gpa, io, root, weights.paths.clip);
            defer src.close(io);
            var l: Loader = .{ .store = &p.clip_store, .io = io, .up = &up, .src = &src, .gpa = gpa, .cast = .as_stored };
            defer l.deinit();
            try clip.forEachTensor(&l);
            p.clip_w = try clip.bindWeights(&p.clip_store);
        }
        // PR #4's loader: the whole checkpoint in host memory, one device image, the decoder relocated onto it
        if (parts.video) {
            const file = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, weights.paths.vae }), gpa, .limited(2 << 30));
            defer gpa.free(file);
            var h = try vae_weights.parseHeader(gpa, file);
            defer h.deinit();
            var img = try vae_weights.build(gpa, .{ .header = &h, .payload = file[h.data_start..] });
            defer img.deinit(gpa);
            p.vae_buf = try cuda.DeviceBuffer.alloc(&p.d, img.bytes.len);
            errdefer p.vae_buf.free();
            try p.vae_buf.upload(0, img.bytes);
            p.vae_dec = img.dec;
            vae_weights.relocateDecoder(&p.vae_dec, p.vae_buf.ptr);
        }
        errdefer p.vae_buf.free();
        // PR #6's binders read every tensor through `AudioIo` (checked against the graph's names and shapes) and place
        // it on the device; the decoder's weights are prepared on the host first
        if (parts.audio) {
            {
                var src = try weights.Source.openFile(gpa, io, root, weights.paths.audio_vae);
                defer src.close(io);
                var aio: AudioIo = .{ .store = &p.audio_store, .io = io, .up = &up, .src = &src, .gpa = gpa };
                defer aio.deinit();
                p.audio_w = try audio_vae.bindDecoder(gpa, aio.source(), aio.sink(), opts.audio.prepare, audioUse(opts.use));
            }
            var src = try weights.Source.openFile(gpa, io, root, weights.paths.vocoder);
            defer src.close(io);
            var aio: AudioIo = .{ .store = &p.audio_store, .io = io, .up = &up, .src = &src, .gpa = gpa };
            defer aio.deinit();
            p.voc_w = try vocoder.bindVocoder(aio.source(), aio.sink());
        }

        const work = @max(
            @max(qwen25.scratchBytes(), clip.scratchBytes()),
            @max(if (parts.dit) p.dit_plan.bytes() else 0, if (parts.video) vae_decode.scratchBytes(p.vae_plan) else 0),
            @max(p.audio_dec.scratchBytes(), p.voc.scratchBytes()),
            dit_shape.tokens() * rope.pairs * 4, // the RoPE arguments, at load
        );
        p.lay = .init(work, p.vae_plan.emitted().rgbBytes());
        p.buf = try cuda.DeviceBuffer.alloc(&p.d, p.lay.total);
        errdefer p.buf.free();
        try p.tables();
        try p.s.synchronize();
        return p;
    }

    pub fn destroy(p: *Pipeline) void {
        p.buf.free();
        p.vae_buf.free();
        p.audio_store.deinit();
        p.clip_store.deinit();
        p.qwen_store.deinit();
        p.dit_store.deinit();
        p.audio_ops.unload();
        p.vae.unload();
        p.dit_k.unload();
        p.text_k.unload();
        p.ops.unload();
        p.s.deinit();
        p.ctx.deinit();
        p.d.close();
        p.closeTokenizers();
        p.gpa.destroy(p);
    }

    fn closeTokenizers(p: *Pipeline) void {
        if (p.clip_tok) |*t| t.deinit();
        if (p.qwen_tok) |*t| t.deinit();
    }

    fn at(p: *const Pipeline, off: u64) u64 {
        return p.buf.ptr + off;
    }

    fn upload(p: *Pipeline, off: u64, bytes: []const u8) !void {
        try p.buf.upload(@intCast(off), bytes);
    }

    fn onDevice(p: *const Pipeline, role: []const u8, dtype: rec_mod.Dtype, shape: []const u64, off: u64) RT {
        return .{ .role = role, .dtype = dtype, .shape = shape, .src = .{ .device = p.at(off) } };
    }

    /// The geometry's tables: time embedding frequencies, then each RoPE's arguments on the host, cos / sin on the GPU.
    /// Recorded as `k6.freqs.*` (the frequency buffers, as upstream's modules hold them) and `k6.rope.*` (the tables).
    fn tables(p: *Pipeline) !void {
        const gpa = p.gpa;
        const r = p.opts.hooks.rec;
        var fv: [rope.time_freqs_v]f32 = undefined;
        rope.freqs(&fv);
        try p.upload(p.lay.freqs_v, std.mem.sliceAsBytes(&fv));
        var fa: [rope.time_freqs_a]f32 = undefined;
        rope.freqs(&fa);
        try p.upload(p.lay.freqs_a, std.mem.sliceAsBytes(&fa));
        if (r) |x| {
            try x.op("k6.freqs.time_v", "k6", .{}, &.{}, &.{.{ .role = "f", .dtype = .float32, .shape = &.{rope.time_freqs_v}, .src = .{ .host = std.mem.sliceAsBytes(&fv) } }});
            try x.op("k6.freqs.time_a", "k6", .{}, &.{}, &.{.{ .role = "f", .dtype = .float32, .shape = &.{rope.time_freqs_a}, .src = .{ .host = std.mem.sliceAsBytes(&fa) } }});
        }

        var ft: [rope.axes_pairs[0]]f32 = undefined;
        var fh: [rope.axes_pairs[1]]f32 = undefined;
        rope.freqs(&ft);
        rope.freqs(&fh); // h and w share the size, so the values
        var f1: [rope.pairs]f32 = undefined;
        rope.freqs(&f1);
        var fs: [rope.pairs]f32 = undefined;
        rope.audioFreqs(&f1, &fs);
        const N = dit_shape.tokens();
        const args = try gpa.alloc(f32, @max(N, rope_buffer_rows_1d) * rope.pairs);
        defer gpa.free(args);
        if (r) |x| {
            // upstream's `args` buffers: outer(arange(max_pos), freqs), one f32 product an element (linearArgs)
            inline for (.{ .{ "k6.freqs.rope_t", &ft, rope_buffer_rows_3d }, .{ "k6.freqs.rope_h", &fh, rope_buffer_rows_3d }, .{ "k6.freqs.rope_w", &fh, rope_buffer_rows_3d }, .{ "k6.freqs.rope_text", &f1, rope_buffer_rows_1d }, .{ "k6.freqs.rope_audio", &fs, rope_buffer_rows_1d } }) |e| {
                const cols = e[1].len;
                const buf = args[0 .. e[2] * cols];
                for (0..e[2]) |pos| for (e[1], 0..) |fr, j| {
                    buf[pos * cols + j] = @as(f32, @floatFromInt(pos)) * fr;
                };
                try x.op(e[0], "k6", .{}, &.{}, &.{.{ .role = "f", .dtype = .float32, .shape = &.{ e[2], cols }, .src = .{ .host = std.mem.sliceAsBytes(buf) } }});
            }
        }
        rope.visualArgs(shapes.latent_frames, shapes.patches_h, shapes.patches_w, &ft, &fh, &fh, args[0 .. N * rope.pairs]);
        try p.ropeTable(args[0 .. N * rope.pairs], p.lay.rope_visual, N);
        rope.linearArgs(&f1, args[0 .. @as(usize, shapes.text_rows) * rope.pairs]);
        try p.ropeTable(args[0 .. @as(usize, shapes.text_rows) * rope.pairs], p.lay.rope_text, shapes.text_rows);
        rope.linearArgs(&fs, args[0 .. @as(usize, shapes.audio_rows) * rope.pairs]);
        try p.ropeTable(args[0 .. @as(usize, shapes.audio_rows) * rope.pairs], p.lay.rope_audio, shapes.audio_rows);
        if (r) |x| {
            const T: u64 = shapes.text_rows;
            try x.op("k6.rope.visual", "k6", .{}, &.{}, &.{p.onDevice("table", .float32, &.{ N, rope_cols }, p.lay.rope_visual)});
            try x.op("k6.rope.audio", "k6", .{}, &.{}, &.{p.onDevice("table", .float32, &.{ shapes.audio_rows, rope_cols }, p.lay.rope_audio)});
            try x.op("k6.rope.text_v", "k6", .{}, &.{}, &.{p.onDevice("table", .float32, &.{ T, rope_cols }, p.lay.rope_text)});
            try x.op("k6.rope.text_a", "k6", .{}, &.{}, &.{p.onDevice("table", .float32, &.{ T, rope_cols }, p.lay.rope_text)});
        }

        var times: [shapes.steps]f32 = undefined;
        for (p.steps, &times) |st, *t| t.* = st.time;
        try p.upload(p.lay.times, std.mem.sliceAsBytes(&times));
    }

    fn ropeTable(p: *Pipeline, args: []const f32, out: u64, rows: u64) !void {
        try p.s.synchronize(); // the work region may still hold the previous table's arguments
        try p.upload(p.lay.work, std.mem.sliceAsBytes(args));
        try p.ops.ropeCosSin(p.s, p.at(p.lay.work), p.at(out), rows);
    }

    /// The decoded video's geometry: 121 frames at the VAE plan's emitted size (the DiT's latent stays 60 x 108; a plan
    /// whose full tiles stop before the latent's edge decodes, as upstream does, to a narrower or shorter frame).
    pub fn videoGeometry(p: *const Pipeline) vae_tiles.Geometry {
        return p.vae_plan.emitted();
    }

    /// One request: `frames` uint8 RGB [121][H][W][3] (`videoGeometry`), `wave` the mono waveform [223,232].
    pub fn generate(p: *Pipeline, r: Request, frames: []u8, wave: []f32, progress: Progress) !void {
        if (frames.len != p.videoGeometry().rgbBytes() or wave.len != shapes.audio_samples) return error.InvalidBuffer;
        const parts = p.opts.parts;
        if (!(parts.text and parts.dit and parts.video and parts.audio)) return error.PartNotLoaded;
        progress.at("text", 0, 1);
        const valid = try p.text(r.prompt, r.prompt_index);
        try p.drawNoise(r.seed);
        try p.sample(valid, progress);
        progress.at("decode", 0, 1);
        try p.decodeVideo(frames);
        if (p.opts.hooks.rec) |x| {
            const g = p.videoGeometry();
            const planar = try p.gpa.alloc(u8, frames.len);
            defer p.gpa.free(planar);
            planarFrames(frames, planar, g);
            const fshape = framesShape(g);
            try x.op("k6.e2e.frames", "k6", .{ .fps = shapes.fps, .prompt = r.prompt_index }, &.{}, &.{.{ .role = "frames", .dtype = .uint8, .shape = &fshape, .src = .{ .host = planar } }});
        }
        progress.at("audio", 0, 1);
        try p.audio(wave);
    }

    /// The text stage: Qwen hidden states and key mask, CLIP pooled (cast to bf16). Returns the Qwen valid count.
    /// `index` labels the recorded ops (the prompt's line in the gate's prompt file).
    pub fn text(p: *Pipeline, prompt_text: []const u8, index: u32) !u32 {
        if (!p.opts.parts.text) return error.PartNotLoaded;
        const L = p.lay;
        const s = p.s;
        var ids: [prompt.qwen_tokens]u32 = undefined;
        var mask: [prompt.qwen_tokens]u8 = undefined;
        const valid = try prompt.qwenIds(&p.qwen_tok.?, p.gpa, prompt_text, &ids, &mask);
        var cids: [shapes.clip_tokens]u32 = undefined;
        const c = try prompt.clipIds(&p.clip_tok.?, p.gpa, prompt_text, &cids);
        try p.s.synchronize(); // the mask upload below must not race a previous request's DiT
        try p.upload(L.mask, mask[prompt.qwen_crop..]);
        const qe: qwen25.Encoder = .{ .k = &p.text_k, .s = s, .w = &p.qwen_w, .base = p.at(L.work), .inv_freq = p.inv_freq, .rope_freq = math(qwen25.Fp32Math, p.opts.qwen_rope_freq), .hook = p.opts.hooks.text };
        try qe.forward(&ids, valid, p.at(L.text));
        if (p.opts.hooks.rec) |x| {
            var ids64: [prompt.qwen_tokens]i64 = undefined;
            var mask64: [prompt.qwen_tokens]i64 = undefined;
            for (ids, mask, &ids64, &mask64) |id, m, *a, *b| {
                a.* = id;
                b.* = m;
            }
            try x.op("k6.qwen.embeds", "k6", .{ .prompt = index, .crop_start = prompt.qwen_crop }, &.{
                .{ .role = "ids", .dtype = .int64, .shape = &.{ 1, prompt.qwen_tokens }, .src = .{ .host = std.mem.sliceAsBytes(&ids64) } },
                .{ .role = "mask", .dtype = .int64, .shape = &.{ 1, prompt.qwen_tokens }, .src = .{ .host = std.mem.sliceAsBytes(&mask64) } },
            }, &.{p.onDevice("embeds", .bfloat16, &.{ shapes.text_rows, shapes.text_dim }, L.text)});
        }
        const ce: clip.Encoder = .{ .k = &p.text_k, .s = s, .w = &p.clip_w, .base = p.at(L.work), .linear = math(clip.Fp32Math, p.opts.clip_linear), .hook = p.opts.hooks.text };
        try ce.forward(&cids, c.valid, c.eot, p.at(L.pooled_f32));
        if (p.opts.hooks.rec) |x| {
            var c64: [shapes.clip_tokens]i64 = undefined;
            for (cids, &c64) |id, *o| o.* = id;
            try x.op("k6.clip.pooled", "k6", .{ .prompt = index }, &.{.{ .role = "ids", .dtype = .int64, .shape = &.{ 1, shapes.clip_tokens }, .src = .{ .host = std.mem.sliceAsBytes(&c64) } }}, &.{p.onDevice("pooled", .float32, &.{ 1, shapes.clip_dim }, L.pooled_f32)});
        }
        try ce.castPooled(p.at(L.pooled_f32), p.at(L.pooled));
        return valid;
    }

    /// The starting noise: video at `seed`, audio at `seed + 1`.
    pub fn drawNoise(p: *Pipeline, seed: u64) !void {
        const L = p.lay;
        const da = try noise.audioDraw(seed, p.dev);
        try p.ops.randn(p.s, p.at(L.noise_v), noise.videoDraw(seed, p.dev));
        try p.ops.randn(p.s, p.at(L.noise_a), da);
        if (p.opts.hooks.rec) |x| {
            try x.op("k6.noise.video", "k6", .{ .seed = seed }, &.{}, &.{p.onDevice("x", .bfloat16, &video_shape, L.noise_v)});
            try x.op("k6.noise.audio", "k6", .{ .seed = da.seed }, &.{}, &.{p.onDevice("x", .bfloat16, &audio_shape, L.noise_a)});
        }
    }

    /// The 10 PiFlow steps from the noise to the final f32 states.
    fn sample(p: *Pipeline, valid: u32, progress: Progress) !void {
        const L = p.lay;
        const s = p.s;
        const x: dit.Dit = .{ .k = &p.dit_k, .s = s, .w = &p.dit_w, .p = p.dit_plan, .base = p.at(L.work), .fp32_linear = math(dit.launch.Fp32Linear, p.opts.fp32_linear), .hook = p.opts.hooks.dit };
        const N = dit_shape.tokens();
        const A: u64 = shapes.audio_rows;
        const T: u64 = shapes.text_rows;
        for (p.steps, 0..) |st, i| {
            progress.at("denoise", @intCast(i), shapes.steps);
            const sv = p.stateV(i);
            const sa = p.stateA(i);
            const dt: dit.layout.Dtype = if (i == 0) .bf16 else .f32;
            try p.ops.videoInput(s, sv, null, p.at(L.video_in));
            const time = View.of(p.at(L.times) + i * 4, .f32, 1, 1);
            try x.forward(.{
                .x_video = View.of(p.at(L.video_in), dt, central.video_rows, shapes.video_in_channels),
                .x_audio = View.of(sa.ptr, dt, A, shapes.audio_channels),
                .text = View.of(p.at(L.text), .bf16, T, shapes.text_dim),
                .pooled = View.of(p.at(L.pooled), .bf16, 1, shapes.clip_dim),
                .time_v = time,
                .time_a = time,
                .freqs_v = View.of(p.at(L.freqs_v), .f32, 1, rope.time_freqs_v),
                .freqs_a = View.of(p.at(L.freqs_a), .f32, 1, rope.time_freqs_a),
                .visual_rope = View.of(p.at(L.rope_visual), .f32, N, rope_cols),
                .audio_rope = View.of(p.at(L.rope_audio), .f32, A, rope_cols),
                .text_rope_v = View.of(p.at(L.rope_text), .f32, T, rope_cols),
                .text_rope_a = View.of(p.at(L.rope_text), .f32, T, rope_cols),
                .mask = .{ .keys = .{ .ptr = p.at(L.mask), .valid = valid - prompt.qwen_crop } },
            }, .{
                .video = View.of(p.at(L.pred_v), .bf16, shapes.n_grid * central.video_rows, shapes.latent_channels),
                .audio = View.of(p.at(L.pred_a), .bf16, shapes.n_grid * A, shapes.audio_channels),
            });
            if (p.opts.hooks.rec) |r| {
                const sd = stateDtype(i);
                const tt = [2]f32{ st.time, st.time };
                var nb: [32]u8 = undefined;
                try r.op(try std.fmt.bufPrint(&nb, "k6.dit.step{d}", .{i}), "k6", .{ .step = i }, &.{
                    .{ .role = "x_video", .dtype = sd, .shape = &.{ 1, shapes.latent_frames, shapes.latent_h, shapes.latent_w, shapes.video_in_channels }, .src = .{ .device = p.at(L.video_in) } },
                    .{ .role = "x_audio", .dtype = sd, .shape = &audio_batch_shape, .src = .{ .device = sa.ptr } },
                    .{ .role = "time", .dtype = .float32, .shape = &.{2}, .src = .{ .host = std.mem.sliceAsBytes(&tt) } },
                }, &.{ p.onDevice("video", .bfloat16, &pred_v_shape, L.pred_v), p.onDevice("audio", .bfloat16, &pred_a_shape, L.pred_a) });
            }
            try p.rollout(i, st);
        }
    }

    fn stateV(p: *const Pipeline, i: usize) central.Tensor {
        return if (i == 0) .{ .ptr = p.at(p.lay.noise_v), .dtype = .bf16 } else .{ .ptr = p.at(p.lay.state_v), .dtype = .f32 };
    }

    fn stateA(p: *const Pipeline, i: usize) central.Tensor {
        return if (i == 0) .{ .ptr = p.at(p.lay.noise_a), .dtype = .bf16 } else .{ .ptr = p.at(p.lay.state_a), .dtype = .f32 };
    }

    /// Step `i`'s rollout of both modalities from the state and the predictions in place; recorded as
    /// `k6.piflow.step{i}` (the states before, as read, and after).
    fn rollout(p: *Pipeline, i: usize, st: piflow.Step) !void {
        const L = p.lay;
        const sv = p.stateV(i);
        const sa = p.stateA(i);
        var pre_v: []u8 = &.{};
        var pre_a: []u8 = &.{};
        defer p.gpa.free(pre_v);
        defer p.gpa.free(pre_a);
        const sd = stateDtype(i);
        if (p.opts.hooks.rec != null) { // the fp32 states roll out in place: keep the inputs
            try p.s.synchronize();
            pre_v = try p.gpa.alloc(u8, shapes.video_elems * sd.size());
            pre_a = try p.gpa.alloc(u8, shapes.audio_elems * sd.size());
            try p.buf.download(@intCast(sv.ptr - p.buf.ptr), pre_v);
            try p.buf.download(@intCast(sa.ptr - p.buf.ptr), pre_a);
        }
        var subs_buf: [piflow.Table.capacity]piflow.Substep = undefined;
        const subs = piflow.substeps(.{}, st, &subs_buf);
        try p.ops.rollout(p.s, sv, p.at(L.pred_v), p.at(L.state_v), shapes.video_elems, st, subs);
        try p.ops.rollout(p.s, sa, p.at(L.pred_a), p.at(L.state_a), shapes.audio_elems, st, subs);
        if (p.opts.hooks.rec) |r| {
            var nb: [32]u8 = undefined;
            try r.op(try std.fmt.bufPrint(&nb, "k6.piflow.step{d}", .{i}), "k6", .{ .step = i, .substeps = st.substeps, .time = [2]f32{ st.time, st.time } }, &.{
                .{ .role = "state_v", .dtype = sd, .shape = &video_batch_shape, .src = .{ .host = pre_v } },
                .{ .role = "state_a", .dtype = sd, .shape = &audio_batch_shape, .src = .{ .host = pre_a } },
                p.onDevice("pred_v", .bfloat16, &pred_v_shape, L.pred_v),
                p.onDevice("pred_a", .bfloat16, &pred_a_shape, L.pred_a),
            }, &.{ p.onDevice("state_v", .float32, &video_batch_shape, L.state_v), p.onDevice("state_a", .float32, &audio_batch_shape, L.state_a) });
        }
    }

    /// The video decode of the final f32 state into `frames` (uint8 [121][H][W][3], `videoGeometry`). PR #4 reads the
    /// f32 state as it is (upstream divides it by the scaling factor and casts once to fp16).
    fn decodeVideo(p: *Pipeline, frames: []u8) !void {
        if (!p.opts.parts.video) return error.PartNotLoaded;
        const g = p.videoGeometry();
        if (frames.len != g.rgbBytes()) return error.InvalidBuffer;
        const L = p.lay;
        var trace: ?VaeTrace = if (p.opts.hooks.vae_trace) (if (p.opts.hooks.rec) |x| VaeTrace{ .p = p, .rec = x, .pass = p.opts.hooks.vae_trace_pass } else null) else null;
        defer if (trace) |*t| t.deinit();
        var vl = p.vae.launcher();
        vae_launch.qualified.apply(&vl);
        if (trace) |*t| vl.probe = t.probe();
        try vae_decode.decode(vl, &p.vae_dec, p.vae_plan, .{
            .latent = p.at(L.state_v),
            .latent_dtype = .f32,
            .rgb = p.at(L.rgb),
            .scratch = p.at(L.work),
            .scratch_bytes = L.work_bytes,
        });
        try p.s.synchronize();
        try p.buf.download(@intCast(L.rgb), frames);
        if (p.opts.hooks.rec) |x| {
            const planar = try p.gpa.alloc(u8, frames.len);
            defer p.gpa.free(planar);
            planarFrames(frames, planar, g);
            const fshape = framesShape(g);
            var attrs = vaeDecodeAttrs(p.vae_plan, p.opts.vae_free_bytes);
            attrs.native_group_norm = vl.gn;
            attrs.native_conv_rules = vl.conv;
            attrs.native_attn_qkv_bias = if (vl.attn_qkv_bias == .conv) "separate" else "fused";
            attrs.native_attn_sdpa_mem_efficient = if (vl.attn_sdpa) |v| @as(u16, @bitCast(v)) else null;
            try x.op("k6.vae.decode", "k6", attrs, &.{p.onDevice("latent", .float32, &video_batch_shape, L.state_v)}, &.{.{ .role = "frames", .dtype = .uint8, .shape = &fshape, .src = .{ .host = planar } }});
            if (x.wants("k6.aux.vae.decoder_in")) {
                // the fp16 decoder input the decode left in its scratch, [F][H][W][16] -> the oracle's [1, 16, F, H, W]
                const n: usize = @intCast(shapes.video_elems);
                const thwc = try p.gpa.alloc(u16, n);
                defer p.gpa.free(thwc);
                const cthw = try p.gpa.alloc(u16, n);
                defer p.gpa.free(cthw);
                try p.buf.download(@intCast(L.work + vae_decode.decoderInputOffset(p.vae_plan)), std.mem.sliceAsBytes(thwc));
                const ch: usize = shapes.latent_channels;
                const px = n / ch;
                for (0..px) |i| for (0..ch) |c| {
                    cthw[c * px + i] = thwc[i * ch + c];
                };
                const shape = [5]u64{ 1, shapes.latent_channels, shapes.latent_frames, shapes.latent_h, shapes.latent_w };
                try x.op("k6.aux.vae.decoder_in", "k6", .{}, &.{}, &.{.{ .role = "z", .dtype = .float16, .shape = &shape, .src = .{ .host = std.mem.sliceAsBytes(cthw) } }});
            }
        }
    }

    /// The audio chain: the sampler's latent rows [1, 218, 40] (`k6.audio.latent_in`) -> mel [1, 128, 436]
    /// (`k6.audio.mel`) -> wave [1, 1, 223,232] (`k6.audio.wave`) -> int16 (`k6.audio.pcm16`, the tool's WAV
    /// conversion `shapes.pcm16`), each op chained to the one before.
    fn audio(p: *Pipeline, wave: []f32) !void {
        if (!p.opts.parts.audio) return error.PartNotLoaded;
        const L = p.lay;
        const use = audioUse(p.opts.use);
        const rows = audio_vae.View.contiguous("k6.audio.latent_in", .cuda, p.at(L.state_a), shapes.audio_elems * 4, .borrowed, &audio_batch_shape);
        const mel = audio_vae.View.contiguous("k6.audio.mel", .cuda, p.at(L.mel), mel_bytes, .borrowed, &mel_shape);
        const out = audio_vae.View.contiguous("k6.audio.wave", .cuda, p.at(L.wave), wave_bytes, .borrowed, &wave_shape);
        const scratch = audio_vae.View.contiguous("k6.audio.scratch", .cuda, p.at(L.work), L.work_bytes, .borrowed, &.{L.work_bytes / 4});
        const r = p.opts.hooks.rec;
        const latent = p.onDevice("latent", .float32, &audio_batch_shape, L.state_a);
        if (r) |x| try x.op("k6.audio.latent_in", "k6", .{ .chained = true }, &.{}, &.{latent});
        try audio_launch.decodeTraced(&p.audio_ops, p.s, use, p.audio_dec, &p.audio_w, rows, scratch, mel, p.opts.audio.decode, p.opts.hooks.audio_decode);
        const mel_t = p.onDevice("mel", .float32, &mel_shape, L.mel);
        if (r) |x| try x.op("k6.audio.mel", "k6", .{ .chained = true }, &.{latent}, &.{mel_t});
        try audio_launch.vocode(&p.audio_ops, p.s, use, p.voc, &p.voc_w, mel, scratch, out, p.opts.audio.vocode);
        const wave_t = p.onDevice("wave", .float32, &wave_shape, L.wave);
        if (r) |x| try x.op("k6.audio.wave", "k6", .{ .chained = true, .sample_rate = shapes.sample_rate }, &.{mel_t}, &.{wave_t});
        try p.s.synchronize();
        try p.buf.download(@intCast(L.wave), std.mem.sliceAsBytes(wave));
        if (r) |x| {
            const pcm = try p.gpa.alloc(i16, wave.len);
            defer p.gpa.free(pcm);
            for (wave, pcm) |w, *o| o.* = shapes.pcm16(w);
            try x.op("k6.audio.pcm16", "k6", .{ .chained = true, .sample_rate = shapes.sample_rate }, &.{wave_t}, &.{.{ .role = "pcm", .dtype = .int16, .shape = &.{shapes.audio_samples}, .src = .{ .host = std.mem.sliceAsBytes(pcm) } }});
        }
    }

    // ---- component replays (the k6-components gate): one stage from the oracle's own inputs ----

    /// Reads a captured tensor into the pipeline buffer at `off`, checking its dtype and element count.
    fn load(p: *Pipeline, c: *const Capture, ref: CaptureRef, dtype: []const u8, elems: u64, off: u64) !void {
        if (!std.mem.eql(u8, ref.dtype, dtype)) return error.CaptureDtype;
        const bytes = try p.gpa.alloc(u8, ref.bytes());
        defer p.gpa.free(bytes);
        if (bytes.len != elems * captureElem(dtype)) return error.CaptureShape;
        try c.blob(p.io, ref, bytes);
        try p.s.synchronize();
        try p.upload(off, bytes);
    }

    /// `k6.piflow.step0..9` from the oracle's states and (synthetic) predictions, each step alone.
    pub fn replayPiflow(p: *Pipeline, c: *const Capture) !void {
        const L = p.lay;
        for (p.steps, 0..) |st, i| {
            var nb: [32]u8 = undefined;
            const o = &c.ops[c.find(try std.fmt.bufPrint(&nb, "k6.piflow.step{d}", .{i}), 0) orelse return error.CaptureMissingOp];
            const sd = if (i == 0) "bfloat16" else "float32";
            try p.load(c, o.in("state_v") orelse return error.CaptureMissingRole, sd, shapes.video_elems, p.stateV(i).ptr - p.buf.ptr);
            try p.load(c, o.in("state_a") orelse return error.CaptureMissingRole, sd, shapes.audio_elems, p.stateA(i).ptr - p.buf.ptr);
            try p.load(c, o.in("pred_v") orelse return error.CaptureMissingRole, "bfloat16", shapes.n_grid * shapes.video_elems, L.pred_v);
            try p.load(c, o.in("pred_a") orelse return error.CaptureMissingRole, "bfloat16", shapes.n_grid * shapes.audio_elems, L.pred_a);
            try p.rollout(i, st);
        }
    }

    /// `k6.vae.decode` from the oracle's latent, under this pipeline's VAE policy (`vae_free_bytes`: the oracle's
    /// `note_free_bytes` reproduces its plan, and the recorded attrs let the compare check it).
    pub fn replayVideoDecode(p: *Pipeline, c: *const Capture) !void {
        const o = &c.ops[c.find("k6.vae.decode", 0) orelse return error.CaptureMissingOp];
        try p.load(c, o.in("latent") orelse return error.CaptureMissingRole, "float32", shapes.video_elems, p.lay.state_v);
        const frames = try p.gpa.alloc(u8, p.videoGeometry().rgbBytes());
        defer p.gpa.free(frames);
        try p.decodeVideo(frames);
    }

    /// The GroupNorm fixture: the reference isolate's `k6.aux.vae.gn.fixture` (one torch GroupNorm(32, C, eps 1e-6) and
    /// its SiLU on a recorded fp16 input: roles x [P][C] channels last, gamma, beta [C], norm and silu [P][C]). Every
    /// statistics candidate runs on this device through the production apply (`vae_launch.groupNormWith`), its outputs are
    /// compared bit for bit with torch's on the host (note `gn_fixture`: unequal counts, first index, the stats), and the
    /// first exact candidate (else the closest) is recorded under the fixture's own op and roles, so the twin compare
    /// checks it role by role. Loads no weights: the video VAE's kernels only. Every op named `k6.aux.vae.gn.fixture` or
    /// `k6.aux.vae.gn.fixture.<case>` (stk_twin.kandinsky6.vae_gn_probe: torch's group_norm on other data) is replayed,
    /// one note each: a policy is proven only by a candidate exact on all of them (a tie on one input proves nothing
    /// about the others). Fails unless each op has an exact candidate.
    pub fn replayGroupNormFixture(p: *Pipeline, c: *const Capture) !void {
        const x_rec = p.opts.hooks.rec orelse return error.CaptureNotRecording;
        var seen: usize = 0;
        var inexact: usize = 0;
        for (c.ops) |*o| {
            if (!(std.mem.eql(u8, o.name, "k6.aux.vae.gn.fixture") or std.mem.startsWith(u8, o.name, "k6.aux.vae.gn.fixture."))) continue;
            seen += 1;
            if (!try p.gnFixtureOne(x_rec, c, o)) inexact += 1;
        }
        if (seen == 0) return error.CaptureMissingOp;
        if (inexact != 0) return error.GroupNormFixtureNoExactCandidate;
    }

    fn gnFixtureOne(p: *Pipeline, x_rec: *Rec, c: *const Capture, o: *const qi.capture.Op) !bool {
        const gpa = p.gpa;
        // the isolate writes the fixture only when it reproduced the in-situ bits; its module must be a 32-group norm
        const groups_attr = if (o.attrs == .object) o.attrs.object.get("groups") else null;
        if (groups_attr == null or groups_attr.? != .integer or groups_attr.?.integer != vae_launch.gn_groups) return error.CaptureShape;
        const xr = o.out("x") orelse return error.CaptureMissingRole;
        if (xr.shape.len != 2) return error.CaptureShape;
        const positions: u64 = @intCast(xr.shape[0]);
        const ch: u64 = @intCast(xr.shape[1]);
        if (ch % 8 != 0 or ch % vae_launch.gn_groups != 0 or 256 % (ch / 8) != 0 or positions == 0) return error.CaptureShape;
        const n = positions * ch;
        const x = try fixtureHalf(p, c, o, "x", n);
        defer gpa.free(x);
        const gamma = try fixtureHalf(p, c, o, "gamma", ch);
        defer gpa.free(gamma);
        const beta = try fixtureHalf(p, c, o, "beta", ch);
        defer gpa.free(beta);
        const want_norm = try fixtureHalf(p, c, o, "norm", n);
        defer gpa.free(want_norm);
        const want_silu = try fixtureHalf(p, c, o, "silu", n);
        defer gpa.free(want_silu);

        const groups: u64 = vae_launch.gn_groups;
        const chunks = (positions + vae_launch.gn_chunk - 1) / vae_launch.gn_chunk;
        const al = struct {
            fn f(cur: *u64, bytes: u64) u64 {
                const o0 = cur.*;
                cur.* = std.mem.alignForward(u64, cur.* + bytes, 256);
                return o0;
            }
        }.f;
        var top: u64 = 0;
        const off_x = al(&top, n * 2);
        const off_y = al(&top, n * 2);
        const off_g = al(&top, ch * 2);
        const off_b = al(&top, ch * 2);
        const off_s = al(&top, groups * 8);
        const off_p = al(&top, chunks * groups * 4);
        var dbuf = try cuda.DeviceBuffer.alloc(&p.d, @intCast(top));
        defer dbuf.free();
        try dbuf.upload(@intCast(off_x), std.mem.sliceAsBytes(x));
        try dbuf.upload(@intCast(off_g), std.mem.sliceAsBytes(gamma));
        try dbuf.upload(@intCast(off_b), std.mem.sliceAsBytes(beta));
        var work: vae_launch.Work = undefined;
        work.gn_partial = dbuf.ptr + off_p;
        work.gn_stats = dbuf.ptr + off_s;
        const view = vae_launch.View.dense(dbuf.ptr + off_x, @intCast(positions), 1, 1, @intCast(ch));
        const norm: vae_launch.NormW = .{ .gamma = dbuf.ptr + off_g, .beta = dbuf.ptr + off_b, .c = @intCast(ch) };
        const l = p.vae.launcher();

        const Cand = struct { name: []const u8, policy: vae_launch.GnPolicy, eps: f32 };
        const half_eps = vae_launch.gnEpsHalf();
        const v_names = [8][]const u8{ "torch_welford_v0", "torch_welford_v1", "torch_welford_v2", "torch_welford_v3", "torch_welford_v4", "torch_welford_v5", "torch_welford_v6", "torch_welford_v7" };
        var cands: [11]Cand = undefined;
        cands[0] = .{ .name = "two_pass", .policy = .two_pass, .eps = vae_launch.gn_eps };
        cands[1] = .{ .name = "two_pass_eps_half", .policy = .two_pass, .eps = half_eps };
        // V7 (every contraction nvcc may apply, its default) first: among exact ties the earliest is recorded
        for (0..8) |k| cands[2 + k] = .{ .name = v_names[7 - k], .policy = .{ .torch_welford = @intCast(7 - k) }, .eps = half_eps };
        cands[10] = .{ .name = "torch_welford_v7_eps_f32", .policy = .{ .torch_welford = vae_launch.WelfordVariant.all }, .eps = vae_launch.gn_eps };

        const Result = struct { candidate: []const u8, norm_unequal: u64, norm_first: ?u64, silu_unequal: u64, silu_first: ?u64, stats_bits: [64]u32 };
        var results: [11]Result = undefined;
        const got = try gpa.alloc(u16, n);
        defer gpa.free(got);
        var best: usize = 0;
        for (cands, 0..) |cand, i| {
            var r: Result = .{ .candidate = cand.name, .norm_unequal = 0, .norm_first = null, .silu_unequal = 0, .silu_first = null, .stats_bits = undefined };
            for ([_]bool{ false, true }) |silu| {
                try vae_launch.groupNormWith(l, work, view, norm, silu, dbuf.ptr + off_y, cand.policy, cand.eps);
                try p.s.synchronize();
                try dbuf.download(@intCast(off_y), std.mem.sliceAsBytes(got));
                const want = if (silu) want_silu else want_norm;
                var unequal: u64 = 0;
                var first: ?u64 = null;
                for (got, want, 0..) |a, b, k| if (a != b) {
                    unequal += 1;
                    if (first == null) first = k;
                };
                if (silu) {
                    r.silu_unequal = unequal;
                    r.silu_first = first;
                } else {
                    r.norm_unequal = unequal;
                    r.norm_first = first;
                }
            }
            try dbuf.download(@intCast(off_s), std.mem.sliceAsBytes(&r.stats_bits));
            results[i] = r;
            if (r.norm_unequal + r.silu_unequal < results[best].norm_unequal + results[best].silu_unequal) best = i;
        }
        try x_rec.note("gn_fixture", .{ .qualification = "unqualified", .op = o.name, .positions = positions, .channels = ch, .groups = groups, .eps_half_bits = @as(u32, @bitCast(half_eps)), .best = cands[best].name, .results = &results });
        std.log.info("k6: gn fixture best {s}: norm {d} / silu {d} unequal of {d}", .{ cands[best].name, results[best].norm_unequal, results[best].silu_unequal, n });

        const out_norm = try gpa.alloc(u16, n);
        defer gpa.free(out_norm);
        for ([_]bool{ false, true }) |silu| {
            try vae_launch.groupNormWith(l, work, view, norm, silu, dbuf.ptr + off_y, cands[best].policy, cands[best].eps);
            try p.s.synchronize();
            try dbuf.download(@intCast(off_y), std.mem.sliceAsBytes(if (silu) got else out_norm));
        }
        const pc = [2]u64{ positions, ch };
        const cc = [1]u64{ch};
        try x_rec.op(o.name, "k6", .{ .note_candidate = cands[best].name, .note_policy = cands[best].policy }, &.{}, &.{
            .{ .role = "x", .dtype = .float16, .shape = &pc, .src = .{ .host = std.mem.sliceAsBytes(x) } },
            .{ .role = "gamma", .dtype = .float16, .shape = &cc, .src = .{ .host = std.mem.sliceAsBytes(gamma) } },
            .{ .role = "beta", .dtype = .float16, .shape = &cc, .src = .{ .host = std.mem.sliceAsBytes(beta) } },
            .{ .role = "norm", .dtype = .float16, .shape = &pc, .src = .{ .host = std.mem.sliceAsBytes(out_norm) } },
            .{ .role = "silu", .dtype = .float16, .shape = &pc, .src = .{ .host = std.mem.sliceAsBytes(got) } },
        });
        return results[best].norm_unequal + results[best].silu_unequal == 0;
    }

    /// The tile-blend fixture: each `k6.aux.vae.blend.fixture.<case>` of stk_twin.kandinsky6.vae_blend_isolate (the
    /// pinned upstream blend_t / blend_v / blend_h on fp16 tiles): a and b (before) and y (b after), [t*h*w][3]. Runs
    /// the decode's own k6v_blend launch (decode.zig blendLaunch, the extent clamped as the method clamps it) and
    /// compares b with y. One note `blend_fixture` a case; fails unless every case is exact.
    pub fn replayBlendFixture(p: *Pipeline, c: *const Capture) !void {
        const rec = p.opts.hooks.rec orelse return error.CaptureNotRecording;
        const gpa = p.gpa;
        const prefix = "k6.aux.vae.blend.fixture.";
        var seen: usize = 0;
        var inexact: usize = 0;
        for (c.ops) |*o| {
            if (!std.mem.startsWith(u8, o.name, prefix)) continue;
            seen += 1;
            if (o.attrs != .object) return error.CaptureShape;
            const geo = struct {
                fn get(mp: std.json.ObjectMap, key: []const u8) !u32 {
                    const v = mp.get(key) orelse return error.CaptureShape;
                    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(i32)) return error.CaptureShape;
                    return @intCast(v.integer);
                }
            }.get;
            const ga = o.attrs.object;
            const axis = try geo(ga, "axis");
            const d = [6]u32{ try geo(ga, "fa"), try geo(ga, "ha"), try geo(ga, "wa"), try geo(ga, "fb"), try geo(ga, "hb"), try geo(ga, "wb") };
            if (axis > 2) return error.CaptureShape;
            for (0..3) |k| if (k != axis and d[k] != d[3 + k]) return error.CaptureShape; // the method's other axes agree
            const extent: u32 = @min(try geo(ga, "extent"), @min(d[axis], d[3 + axis]));
            const na: u64 = @as(u64, d[0]) * d[1] * d[2] * 3;
            const nb: u64 = @as(u64, d[3]) * d[4] * d[5] * 3;
            const av = try fixtureHalf(p, c, o, "a", na);
            defer gpa.free(av);
            const bv = try fixtureHalf(p, c, o, "b", nb);
            defer gpa.free(bv);
            const want = try fixtureHalf(p, c, o, "y", nb);
            defer gpa.free(want);
            const off_b = std.mem.alignForward(u64, na * 2, 256);
            var dbuf = try cuda.DeviceBuffer.alloc(&p.d, @intCast(off_b + nb * 2));
            defer dbuf.free();
            try dbuf.upload(0, std.mem.sliceAsBytes(av));
            try dbuf.upload(@intCast(off_b), std.mem.sliceAsBytes(bv));
            var unequal: u64 = 0;
            var first: ?u64 = null;
            if (extent > 0) {
                try p.vae.launcher().run(vae_decode.blendLaunch(.{ .a = dbuf.ptr, .b = dbuf.ptr + off_b, .fa = @intCast(d[0]), .ha = @intCast(d[1]), .wa = @intCast(d[2]), .fb = @intCast(d[3]), .hb = @intCast(d[4]), .wb = @intCast(d[5]), .ch = 3, .axis = @intCast(axis), .extent = @intCast(extent) }));
                try p.s.synchronize();
            }
            const got = try gpa.alloc(u16, @intCast(nb));
            defer gpa.free(got);
            try dbuf.download(@intCast(off_b), std.mem.sliceAsBytes(got));
            for (got, want, 0..) |x, y, i| if (x != y) {
                unequal += 1;
                if (first == null) first = i;
            };
            if (unequal != 0) inexact += 1;
            try rec.note("blend_fixture", .{ .qualification = "unqualified", .case = o.name[prefix.len..], .axis = axis, .extent = extent, .elements = nb, .unequal = unequal, .first = first });
            std.log.info("k6: blend fixture {s}: {d} unequal of {d}", .{ o.name[prefix.len..], unequal, nb });
        }
        if (seen == 0) return error.CaptureMissingOp;
        if (inexact != 0) return error.BlendFixtureNotExact;
    }

    /// The attention-core fixture: `k6.aux.vae.attn.sdpa` of the attention isolate (stk_twin.kandinsky6.vae_attn_isolate:
    /// q, k, v, out [n][c] and the mask [n][n] exactly as F.scaled_dot_product_attention received and returned them).
    /// Checks the mask is the frame-causal one the production kernel computes (0 / -inf by frame, `spf` tokens a frame,
    /// read off row 0), then runs k6v_sdpa_memeff under every valid variant over the whole output and compares with
    /// torch's bits. One note `sdpa_fixture` an op.
    /// Every op named `k6.aux.vae.attn.sdpa` or `k6.aux.vae.attn.sdpa.<probe>` (stk_twin.kandinsky6.vae_sdpa_probe:
    /// crafted inputs through the same torch kernel) is replayed; the stage fails unless each has an exact variant.
    pub fn replaySdpaFixture(p: *Pipeline, c: *const Capture) !void {
        const rec = p.opts.hooks.rec orelse return error.CaptureNotRecording;
        var seen: usize = 0;
        var inexact: usize = 0;
        for (c.ops) |*o| {
            if (!(std.mem.eql(u8, o.name, "k6.aux.vae.attn.sdpa") or std.mem.startsWith(u8, o.name, "k6.aux.vae.attn.sdpa."))) continue;
            seen += 1;
            if (!try p.sdpaFixtureOne(rec, c, o)) inexact += 1;
        }
        if (seen == 0) return error.CaptureMissingOp;
        if (inexact != 0) return error.SdpaFixtureNotExact;
    }

    fn sdpaFixtureOne(p: *Pipeline, rec: *Rec, c: *const Capture, o: *const qi.capture.Op) !bool {
        const gpa = p.gpa;
        const qr = o.out("q") orelse return error.CaptureMissingRole;
        if (qr.shape.len != 2 or qr.shape[0] <= 0 or qr.shape[1] <= 0) return error.CaptureShape;
        const n: u64 = @intCast(qr.shape[0]);
        const ch: u64 = @intCast(qr.shape[1]);
        if (ch % 128 != 0 or n > std.math.maxInt(i32)) return error.CaptureShape;
        const q = try fixtureHalf(p, c, o, "q", n * ch);
        defer gpa.free(q);
        const k = try fixtureHalf(p, c, o, "k", n * ch);
        defer gpa.free(k);
        const v = try fixtureHalf(p, c, o, "v", n * ch);
        defer gpa.free(v);
        const want = try fixtureHalf(p, c, o, "out", n * ch);
        defer gpa.free(want);
        const mask = try fixtureHalf(p, c, o, "mask", n * n);
        defer gpa.free(mask);
        // the frame-causal mask: row 0 sees frame 0 (spf keys), and token i sees j iff j / spf <= i / spf
        const neg_inf: u16 = 0xfc00;
        var spf: u64 = 0;
        while (spf < n and mask[spf] == 0) spf += 1;
        var mask_causal = spf > 0;
        if (mask_causal) {
            rows: for (0..n) |i| {
                for (0..n) |j| {
                    const bits: u16 = if (j / spf <= i / spf) 0 else neg_inf;
                    if (mask[i * n + j] != bits) {
                        mask_causal = false;
                        break :rows;
                    }
                }
            }
        }
        const ldn = std.mem.alignForward(u64, n, 8);
        const vt = try gpa.alloc(u16, @intCast(ch * ldn));
        defer gpa.free(vt);
        @memset(vt, 0);
        for (0..n) |j| for (0..ch) |cc| {
            vt[cc * ldn + j] = v[j * ch + cc];
        };
        const al = struct {
            fn f(cur: *u64, bytes: u64) u64 {
                const o0 = cur.*;
                cur.* = std.mem.alignForward(u64, cur.* + bytes, 256);
                return o0;
            }
        }.f;
        var top: u64 = 0;
        const off_q = al(&top, n * ch * 2);
        const off_k = al(&top, n * ch * 2);
        const off_vt = al(&top, ch * ldn * 2);
        const off_m = al(&top, n * n * 2);
        const off_acc = al(&top, n * ch * 4);
        const off_o = al(&top, n * ch * 2);
        var dbuf = try cuda.DeviceBuffer.alloc(&p.d, @intCast(top));
        defer dbuf.free();
        try dbuf.upload(@intCast(off_q), std.mem.sliceAsBytes(q));
        try dbuf.upload(@intCast(off_k), std.mem.sliceAsBytes(k));
        try dbuf.upload(@intCast(off_vt), std.mem.sliceAsBytes(vt));
        try dbuf.upload(@intCast(off_m), std.mem.sliceAsBytes(mask));
        const l = p.vae.launcher();
        const got = try gpa.alloc(u16, @intCast(n * ch));
        defer gpa.free(got);

        const Res = struct { variant: u16, name: []const u8, unequal: u64, first: ?u64, max_ulp: u32 };
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const ar = arena.allocator();
        var results: std.ArrayList(Res) = .empty;
        var exact: std.ArrayList([]const u8) = .empty;
        var best: ?u16 = null;
        // every valid variant: with the frame-causal mask computed in the kernel (as production runs it) when the
        // recorded mask is that one, else with the recorded mask itself (then the result can't become a policy)
        for (0..1024) |vi| {
            const sv: vae_launch.SdpaVariant = @bitCast(@as(u16, @intCast(vi)));
            if (!sv.valid()) continue;
            var la = try vae_launch.sdpaMemEff(dbuf.ptr + off_q, dbuf.ptr + off_k, dbuf.ptr + off_vt, dbuf.ptr + off_acc, dbuf.ptr + off_o, @intCast(n), @intCast(ch), ldn, @intCast(@max(spf, 1)), sv);
            if (!mask_causal) {
                la.call.sdpa_memeff.mask = dbuf.ptr + off_m;
                la.call.sdpa_memeff.ldm = @intCast(n);
            }
            try l.run(la);
            try p.s.synchronize();
            try dbuf.download(@intCast(off_o), std.mem.sliceAsBytes(got));
            var r: Res = .{ .variant = @intCast(vi), .name = try std.fmt.allocPrint(ar, "{s}_fma_x{d}_fma_exp{d}_fma_mid{d}_last{d}_exp{d}_rcp{d}", .{ if (sv.k8) "k8" else "k16", @intFromBool(sv.fma_x), @intFromBool(sv.fma_exp), @intFromBool(sv.fma_mid), sv.last, sv.exp, sv.rcp }), .unequal = 0, .first = null, .max_ulp = 0 };
            for (got, want, 0..) |a, b, i| if (a != b) {
                r.unequal += 1;
                if (r.first == null) r.first = i;
                r.max_ulp = @max(r.max_ulp, @abs(halfOrder(a) - halfOrder(b)));
            };
            if (r.unequal == 0) {
                try exact.append(ar, r.name);
                if (best == null) best = r.variant;
            }
            try results.append(ar, r);
        }
        // the closest variant's differences, localised: per query row frame, row within the 32-query tile, output
        // column block of 128, and the first differences with both values
        var closest: ?Res = null;
        for (results.items) |r| if (closest == null or r.unequal < closest.?.unequal) {
            closest = r;
        };
        const DiffAt = struct { row: u64, col: u64, got: u16, want: u16 };
        var at_list: std.ArrayList(DiffAt) = .empty;
        var by_frame: std.ArrayList(u64) = .empty;
        var by_tile_row: [32]u64 = @splat(0);
        var by_col_block: [16]u64 = @splat(0);
        var rows_with: u64 = 0;
        var want_zero: u64 = 0;
        if (closest != null and closest.?.unequal != 0) {
            const cl = closest.?;
            var la = try vae_launch.sdpaMemEff(dbuf.ptr + off_q, dbuf.ptr + off_k, dbuf.ptr + off_vt, dbuf.ptr + off_acc, dbuf.ptr + off_o, @intCast(n), @intCast(ch), ldn, @intCast(@max(spf, 1)), @bitCast(cl.variant));
            if (!mask_causal) {
                la.call.sdpa_memeff.mask = dbuf.ptr + off_m;
                la.call.sdpa_memeff.ldm = @intCast(n);
            }
            try l.run(la);
            try p.s.synchronize();
            try dbuf.download(@intCast(off_o), std.mem.sliceAsBytes(got));
            const frames = if (spf > 0) (n + spf - 1) / spf else 1;
            try by_frame.appendNTimes(ar, 0, @intCast(frames));
            for (0..n) |row| {
                var any = false;
                for (0..ch) |col| {
                    const i = row * ch + col;
                    if (got[i] == want[i]) continue;
                    any = true;
                    by_frame.items[@intCast(if (spf > 0) row / spf else 0)] += 1;
                    by_tile_row[row % 32] += 1;
                    by_col_block[@min(col / 128, 15)] += 1;
                    if ((want[i] & 0x7fff) < 0x0400) want_zero += 1; // |want| below the smallest normal fp16
                    if (at_list.items.len < 64) try at_list.append(ar, .{ .row = row, .col = col, .got = got[i], .want = want[i] });
                }
                if (any) rows_with += 1;
            }
        }
        try rec.note("sdpa_fixture", .{
            .qualification = "unqualified",
            .op = o.name,
            .tokens = n,
            .channels = ch,
            .spf = spf,
            .mask_frame_causal = mask_causal,
            .kernel = "k6v_sdpa_memeff (torch: fmha_cutlassF_f16_aligned_32x128_gmem_sm80)",
            .exact = exact.items,
            .best_variant = best,
            .candidates = results.items,
            .closest = closest,
            .closest_diff = .{ .rows_with_diff = rows_with, .by_frame = by_frame.items, .by_tile_row = by_tile_row, .by_col_block = by_col_block, .want_subnormal_or_zero = want_zero, .first = at_list.items },
        });
        std.log.info("k6: sdpa fixture {s}: frame-causal mask {}, {d} exact variants, first {?d}, closest {?d} unequal", .{ o.name, mask_causal, exact.items.len, best, if (closest) |cl| cl.unequal else null });
        return best != null;
    }

    /// The convolution fixture: each `k6.aux.vae.conv.fixture.<name>` op of the reference isolate (one decoder
    /// convolution as torch ran it on a recorded input). Roles: x [ti*hi*wi][ci], w [co][ci][kt][kh][kw] (the module's),
    /// acc (cudnn's output before output.add_(bias)) [to*ho*wo][co], all fp16, channels last; attrs the geometry.
    /// Every candidate K schedule (`vae_launch.ConvOrderArgs`) runs on this device over the first and last
    /// `conv_fixture_rows` output rows and is compared with torch's bits on the host. Then the production kernel runs the
    /// whole output, tap-major, and with the first exact candidate it can express (order 0, k16, no split) as a
    /// conv rule. One note `conv_fixture` a convolution; fails when one has no production-expressible exact order.
    /// `measured`: an optional capture of measured orders (stk_twin.kandinsky6.vae_conv_oracle): its
    /// `k6.aux.vae.conv.spec.<target>` adds explicit-table candidates (order 2) for that target.
    pub fn replayConvFixture(p: *Pipeline, c: *const Capture, measured: ?*const Capture) !void {
        const rec = p.opts.hooks.rec orelse return error.CaptureNotRecording;
        const prefix = "k6.aux.vae.conv.fixture.";
        var seen: usize = 0;
        var inexact: usize = 0;
        for (c.ops) |*o| {
            if (!std.mem.startsWith(u8, o.name, prefix)) continue;
            seen += 1;
            if (!try p.convFixtureOne(rec, c, o, o.name[prefix.len..], measured)) inexact += 1;
        }
        if (seen == 0) return error.CaptureMissingOp;
        if (inexact != 0) return error.ConvFixtureNotExact;
    }

    fn convFixtureOne(p: *Pipeline, rec: *Rec, c: *const Capture, o: *const qi.capture.Op, name: []const u8, measured: ?*const Capture) !bool {
        const gpa = p.gpa;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const ar = arena.allocator();
        if (o.attrs != .object) return error.CaptureShape;
        const ga = o.attrs.object;
        const geo = struct {
            fn get(mp: std.json.ObjectMap, key: []const u8) !u32 {
                const v = mp.get(key) orelse return error.CaptureShape;
                if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.CaptureShape;
                return @intCast(v.integer);
            }
        }.get;
        const ti = try geo(ga, "ti");
        const hi = try geo(ga, "hi");
        const wi = try geo(ga, "wi");
        const ci = try geo(ga, "ci");
        const co = try geo(ga, "co");
        const kt = try geo(ga, "kt");
        const kh = try geo(ga, "kh");
        const kw = try geo(ga, "kw");
        const up_t = try geo(ga, "up_t");
        const up_hw = try geo(ga, "up_hw");
        const to = try geo(ga, "to");
        const ho = try geo(ga, "ho");
        const wo = try geo(ga, "wo");
        // HunyuanVideoCausalConv3d's pads (the production gather's); a fixture with others is refused
        if (kt == 0 or kh == 0 or kw == 0 or ti == 0 or hi == 0 or wi == 0 or to == 0 or ho == 0 or wo == 0) return error.CaptureShape;
        if (try geo(ga, "pt") != kt - 1 or try geo(ga, "ph") != kh / 2 or try geo(ga, "pw") != kw / 2) return error.CaptureShape;
        if (ci % 16 != 0 or co == 0 or !(up_t == 1 or up_t == 2) or !(up_hw == 1 or up_hw == 2)) return error.CaptureShape;
        const taps: u64 = @as(u64, kt) * kh * kw;
        const m: u64 = @as(u64, to) * ho * wo;
        const x = try fixtureHalf(p, c, o, "x", @as(u64, ti) * hi * wi * ci);
        defer gpa.free(x);
        const w = try fixtureHalf(p, c, o, "w", @as(u64, co) * ci * taps);
        defer gpa.free(w);
        const want = try fixtureHalf(p, c, o, "acc", m * co);
        defer gpa.free(want);

        const ldw = std.mem.alignForward(u64, taps * ci, 8);
        const wrep = try gpa.alloc(u16, @intCast(co * ldw));
        defer gpa.free(wrep);
        @memset(wrep, 0);
        for (0..co) |n| for (0..ci) |ch| for (0..taps) |tap| {
            wrep[n * ldw + tap * ci + ch] = w[(n * ci + ch) * taps + tap];
        };

        const ranges = fixtureRanges(m, conv_fixture_rows);
        const checked = ranges[0][1] + ranges[1][1];
        const al = struct {
            fn f(cur: *u64, bytes: u64) u64 {
                const o0 = cur.*;
                cur.* = std.mem.alignForward(u64, cur.* + bytes, 256);
                return o0;
            }
        }.f;
        var top: u64 = 0;
        const off_x = al(&top, x.len * 2);
        const off_w = al(&top, w.len * 2);
        const off_wr = al(&top, wrep.len * 2);
        const off_y = al(&top, checked * co * 2);
        const off_yp = al(&top, m * co * 2);
        var dbuf = try cuda.DeviceBuffer.alloc(&p.d, @intCast(top));
        defer dbuf.free();
        try dbuf.upload(@intCast(off_x), std.mem.sliceAsBytes(x));
        try dbuf.upload(@intCast(off_w), std.mem.sliceAsBytes(w));
        try dbuf.upload(@intCast(off_wr), std.mem.sliceAsBytes(wrep));
        const l = p.vae.launcher();

        // the candidate schedules
        const Cand = struct { a: vae_launch.ConvOrderArgs, name: []const u8 };
        var cands: std.ArrayList(Cand) = .empty;
        const base: vae_launch.ConvOrderArgs = .{ .x = dbuf.ptr + off_x, .w = dbuf.ptr + off_w, .y = 0, .m0 = 0, .mcount = 0, .ti = @intCast(ti), .hi = @intCast(hi), .wi = @intCast(wi), .ci = @intCast(ci), .tu = @intCast(if (up_t == 2) 1 + 2 * (ti - 1) else ti), .hu = @intCast(hi * up_hw), .wu = @intCast(wi * up_hw), .up_t = @intCast(up_t), .up_hw = @intCast(up_hw), .kt = @intCast(kt), .kh = @intCast(kh), .kw = @intCast(kw), .pt = @intCast(kt - 1), .ph = @intCast(kh / 2), .pw = @intCast(kw / 2), .to = @intCast(to), .ho = @intCast(ho), .wo = @intCast(wo), .co = @intCast(co), .order = 0, .ctile = @intCast(ci), .mma_k = 16, .split = 1, .split_mode = 0, .split_len = 0, .split_join = 0 };
        for ([_]u32{ 16, 32, 64, 128, 256, 512 }) |t| if (t <= ci and ci % t == 0) for ([_]i32{ 16, 8 }) |mk| {
            var a = base;
            a.ctile = @intCast(t);
            a.mma_k = mk;
            try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "ctile{d}_k{d}", .{ t, mk }) });
        };
        // quad32 (perm 5): the two chunks of a 32-channel block take alternate 4-channel quads (conv_out's measured
        // order; the production kernels' kslot quad32)
        for ([_]u32{ 32, 64, 128, 256, 512 }) |t| if (t <= ci and ci % t == 0) {
            var a = base;
            a.ctile = @intCast(t);
            a.perm = 5;
            try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_quad32", .{t}) });
        };
        {
            var a = base;
            a.order = 1;
            try cands.append(ar, .{ .a = a, .name = "kctrs_k16" });
        }
        // split-K: interleaved channel tiles (CUTLASS conv slices), contiguous ranges (GEMM slices, gemm_k_size rounded to the
        // tile; tap-major as one 32-wide tile), each joined in fp32 from 0, through fp16, or by a pairwise fp32 tree
        const joins = [_][]const u8{ "fp32", "fp16", "tree" };
        var tiles_seen: [3]u32 = .{ 0, 0, 0 };
        for ([_]u32{ 32, 64, ci }, 0..) |t, ti_| {
            if (t > ci or ci % t != 0 or std.mem.indexOfScalar(u32, tiles_seen[0..ti_], t) != null) continue;
            tiles_seen[ti_] = t;
            const nchunks = (ci / t) * taps * (t / 16);
            const align_chunks: u64 = if (t == ci) 2 else t / 16;
            for ([_]u32{ 2, 3, 4, 8, 16 }) |sp| for (0..3) |jn| {
                var a = base;
                a.ctile = @intCast(t);
                a.split = @intCast(sp);
                a.split_join = @intCast(jn);
                if (t != ci and ci / t >= sp) {
                    try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_split{d}_interleaved_{s}", .{ t, sp, joins[jn] }) });
                }
                const len = std.mem.alignForward(u64, (nchunks + sp - 1) / sp, align_chunks);
                if (len * (sp - 1) < nchunks) {
                    a.split_mode = 1;
                    a.split_len = @intCast(len);
                    try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_split{d}_contiguous_{s}", .{ t, sp, joins[jn] }) });
                }
            };
        }
        // chunk-interleaved accumulators (consecutive mma chains alternate) and the register-direct "quad" k-slot layout,
        // over the tile orders (tap-major included)
        var tiles2: [3]u32 = .{ 0, 0, 0 };
        for ([_]u32{ 32, 64, ci }, 0..) |t, ti_| {
            if (t > ci or ci % t != 0 or std.mem.indexOfScalar(u32, tiles2[0..ti_], t) != null) continue;
            tiles2[ti_] = t;
            for ([_]u32{ 2, 4 }) |sp| for ([_]u32{ 0, 2 }) |jn| {
                var a = base;
                a.ctile = @intCast(t);
                a.split = @intCast(sp);
                a.split_mode = 3;
                a.split_join = @intCast(jn);
                try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_chunks{d}_{s}", .{ t, sp, joins[jn] }) });
                a.perm = 1;
                try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_quad_chunks{d}_{s}", .{ t, sp, joins[jn] }) });
            };
            var q = base;
            q.ctile = @intCast(t);
            q.perm = 1;
            try cands.append(ar, .{ .a = q, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_quad", .{t}) });
        }
        // k-tile structure of a register-only kernel: consecutive k-tiles (L chunks) alternating between two accumulators,
        // or each k-tile summed from 0 and added to a running total; and the 64-channel k-slot layouts (perm 2..4) alone
        // or with the chunk / k-tile alternations
        var tiles3: [2]u32 = .{ 0, 0 };
        for ([_]u32{ 64, ci }, 0..) |t, ti_| {
            if (t > ci or ci % t != 0 or t % 64 != 0 or std.mem.indexOfScalar(u32, tiles3[0..ti_], t) != null) continue;
            tiles3[ti_] = t;
            for ([_]u32{ 4, 2 }) |kl| {
                var pp = base;
                pp.ctile = @intCast(t);
                pp.split = 2;
                pp.split_mode = 4;
                pp.split_len = @intCast(kl);
                try cands.append(ar, .{ .a = pp, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_ktile{d}_pingpong2", .{ t, kl }) });
                pp.split = 1;
                pp.split_mode = 5;
                try cands.append(ar, .{ .a = pp, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_ktile{d}_partials", .{ t, kl }) });
            }
            for ([_]i32{ 2, 3, 4 }) |pm| {
                var pa = base;
                pa.ctile = @intCast(t);
                pa.perm = pm;
                try cands.append(ar, .{ .a = pa, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_perm{d}", .{ t, pm }) });
                pa.split = 2;
                pa.split_mode = 3;
                try cands.append(ar, .{ .a = pa, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_perm{d}_chunks2", .{ t, pm }) });
                pa.split_mode = 4;
                pa.split_len = 4;
                try cands.append(ar, .{ .a = pa, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_perm{d}_ktile4_pingpong2", .{ t, pm }) });
            }
        }
        // K partitioned across the warps of a threadblock: each of P warps a contiguous share of every k-iteration (one
        // tap's channel tile), joined in fp32 by the epilogue
        for ([_]u32{ 32, 64, 128 }) |t| if (t <= ci and ci % t == 0) for ([_]u32{ 2, 4 }) |pw_| if ((t / 16) % pw_ == 0) for ([_]u32{ 0, 2 }) |jn| {
            var a = base;
            a.ctile = @intCast(t);
            a.split = @intCast(pw_);
            a.split_mode = 2;
            a.split_len = @intCast(t / 16);
            a.split_join = @intCast(jn);
            try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "ctile{d}_k16_warps{d}_{s}", .{ t, pw_, joins[jn] }) });
        };

        // a measured order for this target: the oracle's chunks (16 slots, element tap * ci + c), its chains and join;
        // run as measured, as one chain, and with the other fp32 join. A malformed spec is reported in the note and
        // skipped (the other candidates and targets still run)
        var spec_buf: ?cuda.DeviceBuffer = null;
        defer if (spec_buf) |*sb| sb.free();
        var spec_rejected: ?[]const u8 = null;
        if (measured) |sc| {
            var nb: [160]u8 = undefined;
            if (sc.find(try std.fmt.bufPrint(&nb, "k6.aux.vae.conv.spec.{s}", .{name}), 0)) |si| table: {
                const so = &sc.ops[si];
                const er = so.out("elements") orelse {
                    spec_rejected = "no elements";
                    break :table;
                };
                if (er.shape.len != 2 or er.shape[1] != 16 or er.shape[0] <= 0) {
                    spec_rejected = "elements not [chunks][16]";
                    break :table;
                }
                const nchunks: u64 = @intCast(er.shape[0]);
                const cr = so.out("chains");
                if (!std.mem.eql(u8, er.dtype, "int32") or er.bytes() != nchunks * 16 * 4) {
                    spec_rejected = "elements not int32 [chunks][16]";
                    break :table;
                }
                if (cr != null and (!std.mem.eql(u8, cr.?.dtype, "int32") or cr.?.bytes() != nchunks * 4)) {
                    spec_rejected = "chains not int32, one a chunk";
                    break :table;
                }
                const elems = try fixtureI32(p, sc, so, "elements", nchunks * 16);
                defer gpa.free(elems);
                const chains = if (cr != null) try fixtureI32(p, sc, so, "chains", nchunks) else try gpa.alloc(i32, 0);
                defer gpa.free(chains);
                var nchains: i32 = 1;
                if (try specProblem(gpa, elems, chains, taps * ci, &nchains)) |why| {
                    spec_rejected = why;
                    std.log.warn("k6: conv fixture {s}: spec skipped: {s}", .{ name, why });
                    break :table;
                }
                const atom: i32 = if (so.attrs == .object and so.attrs.object.get("atom") != null and so.attrs.object.get("atom").? == .integer and so.attrs.object.get("atom").?.integer == 8) 8 else 16;
                spec_buf = try cuda.DeviceBuffer.alloc(&p.d, @intCast(nchunks * 16 * 4 + nchunks * 4 + 256));
                const sbuf = &spec_buf.?;
                const off_c: u64 = std.mem.alignForward(u64, nchunks * 16 * 4, 256);
                try sbuf.upload(0, std.mem.sliceAsBytes(elems));
                if (chains.len != 0) try sbuf.upload(@intCast(off_c), std.mem.sliceAsBytes(chains));
                var a = base;
                a.order = 2;
                a.ctile = 0;
                a.mma_k = atom;
                a.spec = sbuf.ptr;
                a.spec_chunks = @intCast(nchunks);
                try cands.append(ar, .{ .a = a, .name = "spec_one_chain" });
                if (chains.len != 0 and nchains > 1) {
                    a.spec_slice = sbuf.ptr + off_c;
                    a.split = nchains;
                    for ([_]i32{ 0, 2, 1 }) |jn| {
                        a.split_join = jn;
                        try cands.append(ar, .{ .a = a, .name = try std.fmt.allocPrint(ar, "spec_chains{d}_{s}", .{ nchains, joins[@intCast(jn)] }) });
                    }
                }
            }
        }

        const Res = struct { candidate: []const u8, unequal: u64, first: ?u64 };
        var results: std.ArrayList(Res) = .empty;
        var exact: std.ArrayList([]const u8) = .empty;
        var best: ?vae_launch.ConvOrderArgs = null;
        const got = try gpa.alloc(u16, @intCast(checked * co));
        defer gpa.free(got);
        for (cands.items) |cand| {
            var row0: u64 = 0;
            for (ranges) |rg| {
                if (rg[1] == 0) continue;
                var a = cand.a;
                a.m0 = @intCast(rg[0]);
                a.mcount = @intCast(rg[1]);
                a.y = dbuf.ptr + off_y + row0 * co * 2;
                try l.run(vae_launch.convOrder(a));
                row0 += rg[1];
            }
            try p.s.synchronize();
            try dbuf.download(@intCast(off_y), std.mem.sliceAsBytes(got));
            var unequal: u64 = 0;
            var first: ?u64 = null;
            row0 = 0;
            for (ranges) |rg| {
                const n = rg[1] * co;
                for (got[row0 * co ..][0..n], want[rg[0] * co ..][0..n], 0..) |ga_, wa_, k| if (ga_ != wa_) {
                    unequal += 1;
                    if (first == null) first = (rg[0] * co) + k;
                };
                row0 += rg[1];
            }
            try results.append(ar, .{ .candidate = cand.name, .unequal = unequal, .first = first });
            if (unequal == 0) {
                try exact.append(ar, cand.name);
                if (best == null and expressible(cand.a)) best = cand.a;
            }
        }

        // the production kernel over the whole output: tap-major, then the best expressible channel tile as a rule
        const xv = vae_launch.View.dense(dbuf.ptr + off_x, ti, hi, wi, ci);
        const cw: vae_launch.ConvW = .{ .w = dbuf.ptr + off_wr, .b = 0, .co = co, .ci = ci, .kt = kt, .kh = kh, .kw = kw };
        const spec: vae_launch.ConvSpec = .{ .up_t = up_t, .up_hw = up_hw };
        const od = vae_launch.convOut(xv, cw, spec);
        if (od[0] != to or od[1] != ho or od[2] != wo) return error.CaptureShape;
        const full = try gpa.alloc(u16, @intCast(m * co));
        defer gpa.free(full);
        const prod = struct {
            fn run(pp: *Pipeline, lc: vae_launch.Launcher, xv_: vae_launch.View, cw_: vae_launch.ConvW, sp_: vae_launch.ConvSpec, y: u64, buf: *cuda.DeviceBuffer, off: u64, out: []u16, ref: []const u16) !u64 {
                try lc.run(try vae_launch.conv(xv_, cw_, sp_, y));
                try pp.s.synchronize();
                try buf.download(@intCast(off), std.mem.sliceAsBytes(out));
                var n: u64 = 0;
                for (out, ref) |u, v| n += @intFromBool(u != v);
                return n;
            }
        }.run;
        const prod_default = try prod(p, l, xv, cw, spec, dbuf.ptr + off_yp, &dbuf, off_yp, full, want);
        var prod_best: ?u64 = null;
        var rules: [1]vae_launch.ConvRule = undefined;
        if (best) |b| {
            rules[0] = .{ .ci = ci, .co = co, .kt = kt, .kh = kh, .kw = kw, .to = to, .ho = ho, .wo = wo, .up_t = up_t, .up_hw = up_hw, .ctile = @intCast(b.ctile), .split = @intCast(b.split), .split_by = if (b.split == 2 and b.split_mode == 3) .chunk else .tile, .kslot = if (b.perm == 5) .quad32 else .natural };
            if (!rules[0].valid()) return error.ConvFixtureRuleInvalid;
            var lr = l;
            lr.conv = &rules;
            prod_best = try prod(p, lr, xv, cw, spec, dbuf.ptr + off_yp, &dbuf, off_yp, full, want);
        }
        // torch's whole op, when the fixture carries it: the bias after the product both ways, separate (cudnn or
        // matmul, then output.add_(bias)) and fused (the cuBLASLt bias epilogue), under the rule when there is one
        var y_unequal: ?struct { separate: u64, fused: u64 } = null;
        if (o.out("bias") != null and o.out("y") != null) {
            const bias = try fixtureHalf(p, c, o, "bias", co);
            defer gpa.free(bias);
            const want_y = try fixtureHalf(p, c, o, "y", m * co);
            defer gpa.free(want_y);
            var bbuf = try cuda.DeviceBuffer.alloc(&p.d, @as(usize, co) * 2);
            defer bbuf.free();
            try bbuf.upload(0, std.mem.sliceAsBytes(bias));
            // a projection (attr linear) runs in the decode through dense(), which no vae_conv rule matches: as here
            var lb = l;
            if (best != null and !fixtureLinear(o)) lb.conv = &rules;
            var cwb = cw;
            cwb.b = bbuf.ptr;
            var sp_sep = spec;
            sp_sep.bias = .conv;
            var sp_fus = spec;
            sp_fus.bias = .linear;
            y_unequal = .{
                .separate = try prod(p, lb, xv, cwb, sp_sep, dbuf.ptr + off_yp, &dbuf, off_yp, full, want_y),
                .fused = try prod(p, lb, xv, cwb, sp_fus, dbuf.ptr + off_yp, &dbuf, off_yp, full, want_y),
            };
        }
        try rec.note("conv_fixture", .{
            .qualification = "unqualified",
            .name = name,
            .ci = ci,
            .co = co,
            .k = [3]u32{ kt, kh, kw },
            .out = [3]u32{ to, ho, wo },
            .up = [2]u32{ up_t, up_hw },
            .rows = m,
            .rows_checked = checked,
            .exact = exact.items,
            .candidates = results.items,
            .production_tap_major_unequal = prod_default,
            .production_rule = if (best != null) rules[0] else null,
            .production_rule_unequal = prod_best,
            .production_y_unequal = y_unequal,
            .spec_rejected = spec_rejected,
        });
        std.log.info("k6: conv fixture {s}: {d} exact candidates, production rule ctile {?d} split {?d} unequal {?d}", .{ name, exact.items.len, if (best) |b| b.ctile else null, if (best) |b| b.split else null, prod_best });
        return prod_best != null and prod_best.? == 0;
    }

    /// The audio chain from the oracle's `k6.audio.latent_in`, chained natively from there.
    pub fn replayAudio(p: *Pipeline, c: *const Capture) !void {
        const o = &c.ops[c.find("k6.audio.latent_in", 0) orelse return error.CaptureMissingOp];
        try p.load(c, o.out("latent") orelse return error.CaptureMissingRole, "float32", shapes.audio_elems, p.lay.state_a);
        const wave = try p.gpa.alloc(f32, shapes.audio_samples);
        defer p.gpa.free(wave);
        try p.audio(wave);
    }

    /// Diagnostic: the vocoder alone on the oracle's own `k6.audio.mel`, each stage observed through
    /// `hooks.audio_vocode`. It records no `k6.audio.*` op, so it is never part of the audio chain.
    pub fn replayVocoder(p: *Pipeline, c: *const Capture) !void {
        if (!p.opts.parts.audio) return error.PartNotLoaded;
        const L = p.lay;
        const o = &c.ops[c.find("k6.audio.mel", 0) orelse return error.CaptureMissingOp];
        try p.load(c, o.out("mel") orelse return error.CaptureMissingRole, "float32", mel_bytes / 4, L.mel);
        const mel = audio_vae.View.contiguous("k6.audio.mel", .cuda, p.at(L.mel), mel_bytes, .borrowed, &mel_shape);
        const out = audio_vae.View.contiguous("k6.audio.wave", .cuda, p.at(L.wave), wave_bytes, .borrowed, &wave_shape);
        const scratch = audio_vae.View.contiguous("k6.audio.scratch", .cuda, p.at(L.work), L.work_bytes, .borrowed, &.{L.work_bytes / 4});
        try audio_launch.vocodeTraced(&p.audio_ops, p.s, audioUse(p.opts.use), p.voc, &p.voc_w, mel, scratch, out, p.opts.audio.vocode, p.opts.hooks.audio_vocode);
        try p.s.synchronize();
    }
};

const Capture = qi.capture.Capture;
const CaptureRef = qi.capture.Ref;

/// Whether the production kernel computes candidate `a`'s schedule: channel tiles (or tap-major) with one m16n8k16 a
/// chunk in the natural k-slot layout, alone, as two interleaved split-K slices of 32-multiple tiles, or as two
/// chunk-alternating accumulators, the slices joined in fp32 (sequential from 0 or the pairwise sum: for two slices both
/// are acc0 + acc1, the _k2 kernels' join).
fn expressible(a: vae_launch.ConvOrderArgs) bool {
    if (a.order != 0 or a.mma_k != 16 or a.split_mode == 5) return false;
    if (a.perm == 5) return a.split == 1 and @mod(a.ctile, 32) == 0; // kslot quad32
    if (a.perm != 0) return false;
    if (a.split == 1) return true;
    if (a.split != 2 or !(a.split_join == 0 or a.split_join == 2)) return false;
    return switch (a.split_mode) {
        0 => a.ctile > 0 and @mod(a.ctile, 32) == 0 and @divTrunc(a.ci, a.ctile) >= 2,
        3 => true, // chunk-alternating: the _k2 kernels with split_by chunk
        else => false,
    };
}

/// Output rows of each end of a fixture convolution the order candidates compute (the first and the last).
const conv_fixture_rows: u64 = 16384;

/// The fixture's checked rows of an m-row output: [start, count] pairs, the first `rows` and the last `rows`, or the
/// whole output once (second range empty) when they would overlap.
fn fixtureRanges(m: u64, rows: u64) [2][2]u64 {
    // u64 throughout: @min with a comptime-known operand narrows its result (to u15 for 16,384), and 2 * r overflowed
    const r: u64 = @min(m, rows);
    if (m <= 2 * r) return .{ .{ 0, m }, .{ m, 0 } };
    return .{ .{ 0, r }, .{ m - r, r } };
}

/// An int32 role of a fixture op, `elems` long, read to the host.
fn fixtureI32(p: *Pipeline, c: *const Capture, o: *const qi.capture.Op, role: []const u8, elems: u64) ![]i32 {
    const r = o.out(role) orelse return error.CaptureMissingRole;
    if (!std.mem.eql(u8, r.dtype, "int32")) return error.CaptureDtype;
    if (r.bytes() != elems * 4) return error.CaptureShape;
    const out = try p.gpa.alloc(i32, @intCast(elems));
    errdefer p.gpa.free(out);
    try c.blob(p.io, r, std.mem.sliceAsBytes(out));
    return out;
}

/// A float16 role of a fixture op, `elems` long, read to the host.
fn fixtureHalf(p: *Pipeline, c: *const Capture, o: *const qi.capture.Op, role: []const u8, elems: u64) ![]u16 {
    const r = o.out(role) orelse return error.CaptureMissingRole;
    if (!std.mem.eql(u8, r.dtype, "float16")) return error.CaptureDtype;
    if (r.bytes() != elems * 2) return error.CaptureShape;
    const out = try p.gpa.alloc(u16, @intCast(elems));
    errdefer p.gpa.free(out);
    try c.blob(p.io, r, std.mem.sliceAsBytes(out));
    return out;
}

/// An fp16 bit pattern as an ordered integer (ulps from +0; -0 and +0 both 0), for distances across the sign.
fn halfOrder(bits: u16) i32 {
    const mag: i32 = bits & 0x7fff;
    return if (bits & 0x8000 != 0) -mag else mag;
}

/// A fixture op with attr `linear: true` (a projection: the decode runs it through dense(), never under a conv rule).
fn fixtureLinear(o: *const qi.capture.Op) bool {
    if (o.attrs != .object) return false;
    const v = o.attrs.object.get("linear") orelse return false;
    return v == .bool and v.bool;
}

/// Why a measured order (`k6.aux.vae.conv.spec.*`) can't run, or null: every element 0..total-1 exactly once and
/// every other slot -1; chain ids (when given, one a chunk) in 0..15. `nchains`: the chain count.
fn specProblem(gpa: std.mem.Allocator, elems: []const i32, chains: []const i32, total: u64, nchains: *i32) !?[]const u8 {
    if (chains.len != 0 and chains.len * 16 != elems.len) return "chains not one a chunk";
    const seen = try gpa.alloc(bool, @intCast(total));
    defer gpa.free(seen);
    @memset(seen, false);
    var n: u64 = 0;
    for (elems) |e| {
        if (e == -1) continue;
        if (e < 0 or e >= @as(i64, @intCast(total))) return "element out of range";
        const i: usize = @intCast(e);
        if (seen[i]) return "element repeated";
        seen[i] = true;
        n += 1;
    }
    if (n != total) return "element missing";
    nchains.* = 1;
    for (chains) |ch_| {
        if (ch_ < 0 or ch_ >= 16) return "chain id outside 0..15";
        nchains.* = @max(nchains.*, ch_ + 1);
    }
    return null;
}

fn captureElem(dtype: []const u8) u64 {
    return qi.capture.elemSize(dtype);
}

fn stateDtype(i: usize) rec_mod.Dtype {
    return if (i == 0) .bfloat16 else .float32;
}

/// The video decode's probe under `Hooks.vae_trace`: each decoder checkpoint in decoder pass `pass` (a pass begins at
/// its post_quant_conv), and the first occurrence of every other checkpoint name (decode.input, the blends, the tile
/// assembly), each read once the launches before it have run, recorded as `k6.aux.vae.trace.<name>` fp16 [t * h * w][c]
/// (`note_thw`: the positions' shape). Activations and attention tokens are both channels-last, so the oracle's [N][C]
/// and [T][H][W][C] views of a checkpoint are the same bytes in this one layout.
const VaeTrace = struct {
    p: *Pipeline,
    rec: *Rec,
    pass: u32 = 0,
    /// The decoder pass the marks belong to: -1 before the first post_quant_conv.
    cur: i64 = -1,
    seen: std.StringHashMapUnmanaged(void) = .empty,

    fn deinit(t: *VaeTrace) void {
        var it = t.seen.keyIterator();
        while (it.next()) |k| t.p.gpa.free(k.*);
        t.seen.deinit(t.p.gpa);
    }

    fn probe(t: *VaeTrace) vae_launch.Probe {
        return .{ .ctx = t, .func = mark };
    }

    fn mark(ctx: *anyopaque, name: []const u8, v: vae_launch.View, tokens: bool) anyerror!void {
        _ = tokens;
        const t: *VaeTrace = @ptrCast(@alignCast(ctx));
        if (std.mem.eql(u8, name, "post_quant_conv")) t.cur += 1;
        if (decoderCheckpoint(name) and t.cur != t.pass) return;
        if (t.seen.contains(name)) return;
        const gpa = t.p.gpa;
        const key = try gpa.dupe(u8, name);
        errdefer gpa.free(key);
        try t.seen.put(gpa, key, {});
        var nb: [192]u8 = undefined;
        const op = try std.fmt.bufPrint(&nb, "k6.aux.vae.trace.{s}", .{name});
        if (!t.rec.wants(op)) return;
        try t.p.s.synchronize();
        const n = v.positions();
        const host = try gpa.alloc(u8, @intCast(n * v.c * 2));
        defer gpa.free(host);
        try gatherView(t.p.buf, v, host);
        const shape = [2]u64{ n, v.c };
        try t.rec.op(op, "k6", .{ .note_thw = [3]u32{ v.t, v.h, v.w } }, &.{}, &.{.{ .role = "x", .dtype = .float16, .shape = &shape, .src = .{ .host = host } }});
    }
};

/// A checkpoint inside one decoder pass (QUALIFICATION.md's decoder names), as opposed to decode.input and the tile
/// blending / assembly ones.
fn decoderCheckpoint(name: []const u8) bool {
    return std.mem.eql(u8, name, "post_quant_conv") or std.mem.startsWith(u8, name, "decoder.");
}

/// Copies the fp16 view `v` (inside `buf`) into `out` densely ([t][h][w][c]).
fn gatherView(buf: cuda.DeviceBuffer, v: vae_launch.View, out: []u8) !void {
    if (v.ptr < buf.ptr or v.ptr - buf.ptr >= buf.len) return error.TraceOutsideBuffer;
    const off: u64 = v.ptr - buf.ptr;
    if (v.contiguous()) return buf.download(@intCast(off), out);
    var o: usize = 0;
    for (0..v.t) |ti| for (0..v.h) |y| {
        const at: u64 = off + (ti * v.st + y * v.sh) * 2;
        if (v.sw == v.c) {
            const nb: usize = @intCast(@as(u64, v.w) * v.c * 2);
            try buf.download(@intCast(at), out[o..][0..nb]);
            o += nb;
        } else for (0..v.w) |x| {
            const nb: usize = @as(usize, v.c) * 2;
            try buf.download(@intCast(at + x * v.sw * 2), out[o..][0..nb]);
            o += nb;
        }
    };
}

/// uint8 [F][H][W][3] -> [1][3][F][H][W], the oracle's frame layout.
fn planarFrames(hwc: []const u8, out: []u8, g: vae_tiles.Geometry) void {
    const plane: usize = @as(usize, g.frames) * g.height * g.width;
    std.debug.assert(hwc.len == plane * 3 and out.len == plane * 3);
    for (0..plane) |px| for (0..3) |ch| {
        out[ch * plane + px] = hwc[px * 3 + ch];
    };
}

/// The oracle's `frames` shape: [1, 3, F, H, W].
fn framesShape(g: vae_tiles.Geometry) [5]u64 {
    return .{ 1, 3, g.frames, g.height, g.width };
}

/// `k6.vae.decode`'s attrs: the plan under the oracle's attr names (compared where both sides record them; the oracle's
/// tile_sample_min_num_frames is apply_tiling's 17 - 1), the exact free-memory figure (`note_`, as the oracle records
/// it), and native's own record of the grid and the nominal and emitted geometry.
fn vaeDecodeAttrs(plan: vae_tiles.DecodePlan, free_bytes: u64) VaeDecodeAttrs {
    const t = plan.tiling;
    const n = plan.nominal();
    const e = plan.emitted();
    return .{
        .tile_size = .{ 1, t.ft, t.ht, t.wt },
        .tile_sample_min_num_frames = t.ft - 1,
        .tile_sample_stride_num_frames = t.fs,
        .tile_sample_min_height = t.ht,
        .tile_sample_min_width = t.wt,
        .tile_sample_stride_height = t.hs,
        .tile_sample_stride_width = t.ws,
        .spatial_compression_ratio = vae_tiles.spatial_ratio,
        .temporal_compression_ratio = vae_tiles.temporal_ratio,
        .scaling_factor = vae_decode.scaling_factor,
        .note_free_bytes = free_bytes,
        .native_grid = .{ plan.grid.rows, plan.grid.cols },
        .native_temporal_tiles = plan.tileCount(),
        .native_passes = plan.passes(),
        .native_nominal_fhw = .{ n.frames, n.height, n.width },
        .native_emitted_fhw = .{ e.frames, e.height, e.width },
    };
}

const VaeDecodeAttrs = struct {
    bs: u32 = 1,
    tile_size: [4]u32,
    use_tiling: bool = true,
    use_slicing: bool = false,
    use_framewise_decoding: bool = true,
    tile_sample_min_num_frames: u32,
    tile_sample_stride_num_frames: u32,
    tile_sample_min_height: u32,
    tile_sample_min_width: u32,
    tile_sample_stride_height: u32,
    tile_sample_stride_width: u32,
    spatial_compression_ratio: u32,
    temporal_compression_ratio: u32,
    scaling_factor: f64,
    dtype: []const u8 = "float16",
    note_free_bytes: u64,
    native_grid: [2]u32,
    native_temporal_tiles: u32,
    native_passes: u32,
    native_nominal_fhw: [3]u32,
    native_emitted_fhw: [3]u32,
    native_group_norm: vae_launch.GnPolicy = .two_pass,
    native_conv_rules: []const vae_launch.ConvRule = &.{},
    native_attn_qkv_bias: []const u8 = "fused",
    /// null: the math attention core; else the k6v_sdpa_memeff variant
    native_attn_sdpa_mem_efficient: ?u16 = null,
};

fn audioUse(u: admission.Use) audio_vae.Use {
    return switch (u) {
        inline else => |t| @field(audio_vae.Use, @tagName(t)),
    };
}

/// PR #6's checkpoint reader and device placement over one safetensors file: `read` hands out f32 host copies (valid
/// until the next read), `place` uploads into the audio store.
const AudioIo = struct {
    store: *weights.Store,
    io: std.Io,
    up: *qi.upload.Uploader,
    src: *const weights.Source,
    gpa: std.mem.Allocator,
    buf: []f32 = &.{},
    shape: [8]u64 = undefined,

    fn deinit(a: *AudioIo) void {
        a.gpa.free(a.buf);
    }

    fn source(a: *AudioIo) audio_vae.Source {
        return .{ .ctx = a, .readFn = read };
    }

    fn sink(a: *AudioIo) audio_vae.Sink {
        return .{ .ctx = a, .placeFn = place };
    }

    fn read(ctx: *anyopaque, name: []const u8) anyerror!audio_vae.HostTensor {
        const a: *AudioIo = @ptrCast(@alignCast(ctx));
        const f = try a.src.find(name);
        if (f.t.dtype != .f32 or f.t.rank > a.shape.len) return error.CheckpointMismatch;
        const n = f.t.len() / 4;
        if (a.buf.len < n) a.buf = try a.gpa.realloc(a.buf, n);
        try f.pack.read(a.io, f.t, std.mem.sliceAsBytes(a.buf[0..n]));
        for (0..f.t.rank) |i| a.shape[i] = f.t.shape[i];
        return .{ .shape = a.shape[0..f.t.rank], .data = a.buf[0..n] };
    }

    fn place(ctx: *anyopaque, name: []const u8, data: []const f32, shape: []const u64) anyerror!audio_vae.View {
        const a: *AudioIo = @ptrCast(@alignCast(ctx));
        const bytes = std.mem.sliceAsBytes(data);
        const ptr = try a.store.putBytes(a.up, name, bytes);
        return audio_vae.View.contiguous(name, .cuda, ptr, bytes.len, .borrowed, shape);
    }
};

fn readInvFreq(io: std.Io, path: []const u8) ![64]f32 {
    if (path.len == 0) return error.QwenInvFreqNotSet;
    var raw: [64 * 4 + 1]u8 = undefined;
    const got = try std.Io.Dir.cwd().readFile(io, path, &raw);
    if (got.len != 64 * 4) return error.QwenInvFreqSize;
    var out: [64]f32 = undefined;
    for (&out, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
    return out;
}

/// Checks each tensor a component lists against the file (dtype and shape) and uploads it under `cast`.
const Loader = struct {
    store: *weights.Store,
    io: std.Io,
    up: *qi.upload.Uploader,
    src: *const weights.Source,
    gpa: std.mem.Allocator,
    cast: weights.Cast,
    scratch: []u8 = &.{},

    fn deinit(l: *Loader) void {
        l.gpa.free(l.scratch);
    }

    pub fn tensor(l: *Loader, name: []const u8, dt: anytype, shape: []const u64) !void {
        const want = std.meta.stringToEnum(qi.pack.DType, @tagName(dt)) orelse return error.CheckpointMismatch;
        const f = try l.src.find(name);
        if (!weights.matches(f.t, want, shape)) {
            std.log.err("kandinsky6: {s} is not the expected {t} {any}", .{ name, dt, shape });
            return error.CheckpointMismatch;
        }
        if (l.cast != .as_stored and f.t.dtype == .f32 and l.scratch.len < f.t.len())
            l.scratch = try l.gpa.realloc(l.scratch, f.t.len());
        _ = try l.store.put(l.io, l.up, l.src, name, l.cast, l.scratch);
    }
};

test "the persistent layout holds every buffer apart and aligned" {
    const plan = try vae_gpu.plan(shapes.latent_frames, shapes.latent_h, shapes.latent_w, 8192 << 20);
    const rgb_bytes = plan.emitted().rgbBytes();
    try std.testing.expectEqual(@as(u64, shapes.sample_frames) * shapes.height * shapes.width * 3, rgb_bytes);
    const l: Layout = .init(1 << 30, rgb_bytes);
    const offs = [_]u64{ l.text, l.mask, l.pooled_f32, l.pooled, l.noise_v, l.noise_a, l.state_v, l.state_a, l.video_in, l.pred_v, l.pred_a, l.times, l.freqs_v, l.freqs_a, l.rope_visual, l.rope_audio, l.rope_text, l.rgb, l.mel, l.wave, l.work };
    for (offs[1..], offs[0 .. offs.len - 1]) |b, a| try std.testing.expect(b > a);
    for (offs) |o| try std.testing.expectEqual(@as(u64, 0), o % 256);
    try std.testing.expectEqual(@as(u64, rgb_bytes), l.mel - l.rgb);
    try std.testing.expectEqual(shapes.audio_samples, @as(u64, vocoder.hop) * mel_frames); // the vocoder's 512x
    try std.testing.expectEqual(@as(u64, 50220), dit_shape.tokens());
}

test "every arithmetic choice maps onto the components' enums" {
    for (std.enums.values(Fp32Linear)) |f| {
        try std.testing.expectEqualStrings(@tagName(f), @tagName(math(qwen25.Fp32Math, f)));
        try std.testing.expectEqualStrings(@tagName(f), @tagName(math(dit.launch.Fp32Linear, f)));
    }
}

test "the Qwen inverse frequencies are required" {
    try std.testing.expectError(error.QwenInvFreqNotSet, readInvFreq(std.testing.io, ""));
}

test "fixture rows: whole outputs up to twice the window, else both ends (every decoder conv size)" {
    // conv_in / mid (4,640), up0 / up1 (18,560: 2 x 16,384 = 32,768 overflowed a u15), up2 (133,632), up3 (1,009,664)
    try std.testing.expectEqual([2][2]u64{ .{ 0, 4640 }, .{ 4640, 0 } }, fixtureRanges(4640, conv_fixture_rows));
    try std.testing.expectEqual([2][2]u64{ .{ 0, 18560 }, .{ 18560, 0 } }, fixtureRanges(18560, conv_fixture_rows));
    try std.testing.expectEqual([2][2]u64{ .{ 0, 16384 }, .{ 117248, 16384 } }, fixtureRanges(133632, conv_fixture_rows));
    try std.testing.expectEqual([2][2]u64{ .{ 0, 16384 }, .{ 993280, 16384 } }, fixtureRanges(1009664, conv_fixture_rows));
    try std.testing.expectEqual([2][2]u64{ .{ 0, 32768 }, .{ 32768, 0 } }, fixtureRanges(32768, conv_fixture_rows));
    try std.testing.expectEqual([2][2]u64{ .{ 0, 16384 }, .{ 16385, 16384 } }, fixtureRanges(32769, conv_fixture_rows));
}

test "a measured order runs only as a permutation of the reduction" {
    const gpa = std.testing.allocator;
    var n: i32 = 0;
    var e: [32]i32 = @splat(-1);
    for (0..8) |i| e[i] = @intCast(i);
    for (0..8) |i| e[16 + i] = @intCast(8 + i);
    try std.testing.expectEqual(@as(?[]const u8, null), try specProblem(gpa, &e, &.{ 0, 1 }, 16, &n));
    try std.testing.expectEqual(@as(i32, 2), n);
    try std.testing.expectEqual(@as(?[]const u8, null), try specProblem(gpa, &e, &.{}, 16, &n));
    try std.testing.expectEqual(@as(i32, 1), n);
    try std.testing.expectEqualStrings("chain id outside 0..15", (try specProblem(gpa, &e, &.{ 0, 16 }, 16, &n)).?);
    try std.testing.expectEqualStrings("chains not one a chunk", (try specProblem(gpa, &e, &.{0}, 16, &n)).?);
    try std.testing.expectEqualStrings("element missing", (try specProblem(gpa, &e, &.{}, 17, &n)).?);
    e[17] = 0;
    try std.testing.expectEqualStrings("element repeated", (try specProblem(gpa, &e, &.{}, 16, &n)).?);
    e[17] = 16;
    try std.testing.expectEqualStrings("element out of range", (try specProblem(gpa, &e, &.{}, 16, &n)).?);
    e[17] = -2;
    try std.testing.expectEqualStrings("element out of range", (try specProblem(gpa, &e, &.{}, 16, &n)).?);
}

test "production runs the qualified VAE plan only (832x480); the diagnostic gates may replay another" {
    const gpa = std.testing.allocator;
    const base =
        \\{"fp32_linear": "fp32", "clip_linear": "fp32", "qwen_rope_freq": "fp32", "qwen_inv_freq": "x.f32le",
        \\ "audio": {"prepare": {"eps": 0.0001, "reduce": "serial", "den": "fma"},
        \\  "decode": {"acc": "fma", "unscale": "true_div", "silu": "true_div", "residual": "true_div",
        \\   "norm_residual": {"eps": 0.0001, "reduce": {"serial_fma": {}}, "den": "fma"},
        \\   "norm_qkv": {"eps": 0.0001, "reduce": {"serial_fma": {}}, "den": "fma"}, "attention": "math_candidate"},
        \\  "vocode": {"conv_acc": "fma", "aa_acc": "fma", "avg": "true_div"}}
    ;
    for ([_]struct { extra: []const u8, production_ok: bool }{
        .{ .extra = "}", .production_ok = true },
        .{ .extra = ", \"vae_free_bytes\": 2301403136}", .production_ok = true },
        .{ .extra = ", \"vae_free_mb\": 8192}", .production_ok = false },
        .{ .extra = ", \"vae_free_bytes\": 4000000000}", .production_ok = false },
    }) |c| {
        const text = try std.mem.concat(gpa, u8, &.{ base, c.extra });
        defer gpa.free(text);
        const v = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer v.deinit();
        _ = try parseOptions(v.value, .diagnostic);
        if (c.production_ok) {
            _ = try parseOptions(v.value, .production);
        } else {
            try std.testing.expectError(error.VaePlanNotSupported, parseOptions(v.value, .production));
        }
    }
}

test "the VAE's arithmetic is not an option: the trial fields are refused" {
    const gpa = std.testing.allocator;
    for ([_][]const u8{
        "{\"vae_group_norm\": {\"torch_welford\": 7}}",
        "{\"vae_conv\": []}",
        "{\"vae_attn_qkv_bias\": \"separate\"}",
        "{\"vae_attn_sdpa\": {\"mem_efficient\": 524}}",
    }) |text| {
        const v = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer v.deinit();
        try std.testing.expectError(error.VaeArithmeticNotAnOption, refuseRemovedVaeFields(v.value.object));
    }
    const none = try std.json.parseFromSlice(std.json.Value, gpa, "{\"vae_free_bytes\": 2301403136}", .{});
    defer none.deinit();
    try refuseRemovedVaeFields(none.value.object);
}

test "the VAE free-memory figure: exact bytes or MiB, not both; the plan decides the decoded geometry" {
    const gpa = std.testing.allocator;
    const Case = struct { json: []const u8, want: anyerror!u64 };
    const cases = [_]Case{
        .{ .json = "{}", .want = 2_301_403_136 }, // the qualified plan
        .{ .json = "{\"vae_free_mb\": 8192}", .want = 8192 << 20 },
        .{ .json = "{\"vae_free_bytes\": 2301403136}", .want = 2_301_403_136 },
        .{ .json = "{\"vae_free_bytes\": 0}", .want = error.VaeFreeBytesInvalid },
        .{ .json = "{\"vae_free_bytes\": \"2301403136\"}", .want = error.VaeFreeBytesInvalid },
        .{ .json = "{\"vae_free_bytes\": 2301403136, \"vae_free_mb\": 2194}", .want = error.VaeFreeBytesAndMb },
        .{ .json = "{\"vae_free_mb\": 9223372036854775807}", .want = error.VaeFreeMbInvalid },
    };
    for (cases) |c| {
        const v = try std.json.parseFromSlice(std.json.Value, gpa, c.json, .{});
        defer v.deinit();
        if (c.want) |want| {
            try std.testing.expectEqual(want, try vaeFreeBytes(v.value.object));
        } else |err| {
            try std.testing.expectError(err, vaeFreeBytes(v.value.object));
        }
    }
    // the reference run's figure: 832 wide (MiB cannot express it: 2301403136 / 2^20 = 2194.78)
    const ref = try vae_gpu.plan(shapes.latent_frames, shapes.latent_h, shapes.latent_w, 2_301_403_136);
    try std.testing.expectEqual(vae_tiles.Geometry{ .frames = 121, .height = 480, .width = 832 }, ref.emitted());
    try std.testing.expectEqual([5]u64{ 1, 3, 121, 480, 832 }, framesShape(ref.emitted()));
    const a = vaeDecodeAttrs(ref, 2_301_403_136);
    try std.testing.expectEqual([4]u32{ 1, 17, 256, 232 }, a.tile_size);
    try std.testing.expectEqual(@as(u32, 16), a.tile_sample_min_num_frames);
    try std.testing.expectEqual(@as(u32, 200), a.tile_sample_stride_width);
    try std.testing.expectEqual(@as(u32, 112), a.native_passes);
    try std.testing.expectEqual([3]u32{ 121, 480, 864 }, a.native_nominal_fhw);
    try std.testing.expectEqual([3]u32{ 121, 480, 832 }, a.native_emitted_fhw);
}

test "planar frames: [F][H][W][3] to [3][F][H][W] at any geometry" {
    const g: vae_tiles.Geometry = .{ .frames = 2, .height = 1, .width = 3 };
    var hwc: [18]u8 = undefined;
    for (&hwc, 0..) |*b, i| b.* = @intCast(i);
    var out: [18]u8 = undefined;
    planarFrames(&hwc, &out, g);
    try std.testing.expectEqualSlices(u8, &.{ 0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 16, 2, 5, 8, 11, 14, 17 }, &out);
}
