//! Kandinsky-6 Pro distilled, text to video with its coupled audio, on one GPU: the published snapshot loaded into the
//! components' own layouts, then for each request (pipeline/pipeline.py and piflow_sampler.py at upstream eac06e39)
//!   text      Qwen2.5-VL hidden states [1024, 3584] bf16 + key mask, CLIP pooled [768] cast to bf16
//!   noise     torch CUDA randn: video [31, 60, 108, 16] at seed s, audio [218, 40] at seed s + 1 (bf16)
//!   sampler   10 PiFlow steps: visual input, one fused DiT call (both modalities), the DX policy rollout of each
//!   video     the f32 latent through PR #4's decoder (its own loader, `video_vae/weights.zig`) to uint8 RGB
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
    clip_linear: Fp32Linear,
    /// Qwen's RoPE frequency matmul `inv_freq @ position_ids`.
    qwen_rope_freq: Fp32Linear,
    /// `model.language_model.rotary_emb.inv_freq` as native holds it: a file of 64 little-endian f32, captured on the
    /// target (qwen25.invFreqCandidate is a host-test value, never used here).
    qwen_inv_freq: []const u8,
    /// The free-memory figure the video VAE's tile plan uses (configuration, never a probe).
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
/// and policy field is required, an unknown or missing value is an error, never a default. `vae_free_mb` defaults to
/// 8192 (configuration, not arithmetic).
pub fn parseOptions(value: ?std.json.Value, use: admission.Use) !Options {
    const obj = if (value) |v| (if (v == .object) v.object else return error.OptionsNotAnObject) else return error.Fp32LinearNotSet;
    const arith = struct {
        fn get(o: std.json.ObjectMap, key: []const u8, missing: anyerror) anyerror!Fp32Linear {
            const v = o.get(key) orelse return missing;
            if (v != .string) return missing;
            return std.meta.stringToEnum(Fp32Linear, v.string) orelse missing;
        }
    }.get;
    const inv = obj.get("qwen_inv_freq") orelse return error.QwenInvFreqNotSet;
    if (inv != .string or inv.string.len == 0) return error.QwenInvFreqNotSet;
    const free_mb: u64 = if (obj.get("vae_free_mb")) |v| (if (v == .integer and v.integer > 0) @intCast(v.integer) else return error.VaeFreeMbInvalid) else 8192;
    const audio = obj.get("audio") orelse return error.AudioPolicyNotSet;
    var mem: [1024]u8 = undefined; // the policy holds no slices; parsing needs scratch only
    var fba: std.heap.FixedBufferAllocator = .init(&mem);
    return .{
        .use = use,
        .fp32_linear = try arith(obj, "fp32_linear", error.Fp32LinearNotSet),
        .clip_linear = try arith(obj, "clip_linear", error.ClipLinearNotSet),
        .qwen_rope_freq = try arith(obj, "qwen_rope_freq", error.QwenRopeFreqNotSet),
        .qwen_inv_freq = inv.string,
        .vae_free_bytes = free_mb << 20,
        .audio = std.json.parseFromValueLeaky(AudioPolicy, fba.allocator(), audio, .{}) catch return error.AudioPolicyInvalid,
    };
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
    text: ?qwen25.Hook = null,
    dit: ?dit.Hook = null,
    /// The bit gate's capture: the pipeline records its own ops (the oracle's names and roles) into it.
    rec: ?*Rec = null,
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

const rgb_bytes: u64 = @as(u64, shapes.sample_frames) * shapes.height * shapes.width * 3;
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
const frames_shape = [5]u64{ 1, 3, shapes.sample_frames, shapes.height, shapes.width };
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
    rgb: u64, // uint8 [121][480][864][3]
    mel: u64, // f32 [1, 128, 436]
    wave: u64, // f32 [1, 1, 223,232]
    work: u64, // scratch shared in turn by the encoders, the RoPE arguments, the DiT and the VAE
    work_bytes: u64,
    total: u64,

    fn init(work_bytes: u64) Layout {
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

fn math(comptime T: type, x: Fp32Linear) T {
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
        p.text_k = try qwen25.Kernels.load(&p.d, .{ .text = kmod.text, .qwen_te = qi.kernels.te, .qwen_gemm = qi.kernels.gemm, .qwen_attention = qi.kernels.attention });
        errdefer p.text_k.unload();
        p.dit_k = try dit.launch.Kernels.load(&p.d, .{ .gemm = dit_kmod.gemm, .attention = dit_kmod.attention, .ops = dit_kmod.ops });
        errdefer p.dit_k.unload();
        p.vae = try vae_gpu.Gpu.load(&p.d, kmod.vae_conv, kmod.vae_ops, p.s);
        errdefer p.vae.unload();
        p.audio_ops = try audio_launch.Ops.load(&p.d, kmod.audio);
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
        p.lay = .init(work);
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

    /// One request: `frames` uint8 RGB [121][480][864][3], `wave` the mono waveform [223,232].
    pub fn generate(p: *Pipeline, r: Request, frames: []u8, wave: []f32, progress: Progress) !void {
        if (frames.len != rgb_bytes or wave.len != shapes.audio_samples) return error.InvalidBuffer;
        const parts = p.opts.parts;
        if (!(parts.text and parts.dit and parts.video and parts.audio)) return error.PartNotLoaded;
        progress.at("text", 0, 1);
        const valid = try p.text(r.prompt, r.prompt_index);
        try p.drawNoise(r.seed);
        try p.sample(valid, progress);
        progress.at("decode", 0, 1);
        try p.decodeVideo(frames);
        if (p.opts.hooks.rec) |x| {
            const planar = try p.gpa.alloc(u8, rgb_bytes);
            defer p.gpa.free(planar);
            planarFrames(frames, planar);
            try x.op("k6.e2e.frames", "k6", .{ .fps = shapes.fps, .prompt = r.prompt_index }, &.{}, &.{.{ .role = "frames", .dtype = .uint8, .shape = &frames_shape, .src = .{ .host = planar } }});
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
            try r.op(try std.fmt.bufPrint(&nb, "k6.piflow.step{d}", .{i}), "k6", .{ .step = i, .substeps = st.substeps, .time = st.time }, &.{
                .{ .role = "state_v", .dtype = sd, .shape = &video_batch_shape, .src = .{ .host = pre_v } },
                .{ .role = "state_a", .dtype = sd, .shape = &audio_batch_shape, .src = .{ .host = pre_a } },
                p.onDevice("pred_v", .bfloat16, &pred_v_shape, L.pred_v),
                p.onDevice("pred_a", .bfloat16, &pred_a_shape, L.pred_a),
            }, &.{ p.onDevice("state_v", .float32, &video_batch_shape, L.state_v), p.onDevice("state_a", .float32, &audio_batch_shape, L.state_a) });
        }
    }

    /// The video decode of the final f32 state into `frames` (uint8 [121][480][864][3]). PR #4 reads the f32 state as
    /// it is (upstream divides it by the scaling factor and casts once to fp16).
    fn decodeVideo(p: *Pipeline, frames: []u8) !void {
        if (!p.opts.parts.video) return error.PartNotLoaded;
        const L = p.lay;
        try vae_decode.decode(p.vae.launcher(), &p.vae_dec, p.vae_plan, .{
            .latent = p.at(L.state_v),
            .latent_dtype = .f32,
            .rgb = p.at(L.rgb),
            .scratch = p.at(L.work),
            .scratch_bytes = L.work_bytes,
        });
        try p.s.synchronize();
        try p.buf.download(@intCast(L.rgb), frames);
        if (p.opts.hooks.rec) |x| {
            const planar = try p.gpa.alloc(u8, rgb_bytes);
            defer p.gpa.free(planar);
            planarFrames(frames, planar);
            try x.op("k6.vae.decode", "k6", .{}, &.{p.onDevice("latent", .float32, &video_batch_shape, L.state_v)}, &.{.{ .role = "frames", .dtype = .uint8, .shape = &frames_shape, .src = .{ .host = planar } }});
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
        try audio_launch.decode(&p.audio_ops, p.s, use, p.audio_dec, &p.audio_w, rows, scratch, mel, p.opts.audio.decode);
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

    /// `k6.vae.decode` from the oracle's latent.
    pub fn replayVideoDecode(p: *Pipeline, c: *const Capture) !void {
        const o = &c.ops[c.find("k6.vae.decode", 0) orelse return error.CaptureMissingOp];
        try p.load(c, o.in("latent") orelse return error.CaptureMissingRole, "float32", shapes.video_elems, p.lay.state_v);
        const frames = try p.gpa.alloc(u8, rgb_bytes);
        defer p.gpa.free(frames);
        try p.decodeVideo(frames);
    }

    /// The audio chain from the oracle's `k6.audio.latent_in`, chained natively from there.
    pub fn replayAudio(p: *Pipeline, c: *const Capture) !void {
        const o = &c.ops[c.find("k6.audio.latent_in", 0) orelse return error.CaptureMissingOp];
        try p.load(c, o.out("latent") orelse return error.CaptureMissingRole, "float32", shapes.audio_elems, p.lay.state_a);
        const wave = try p.gpa.alloc(f32, shapes.audio_samples);
        defer p.gpa.free(wave);
        try p.audio(wave);
    }
};

const Capture = qi.capture.Capture;
const CaptureRef = qi.capture.Ref;

fn captureElem(dtype: []const u8) u64 {
    return qi.capture.elemSize(dtype);
}

fn stateDtype(i: usize) rec_mod.Dtype {
    return if (i == 0) .bfloat16 else .float32;
}

/// uint8 [F][H][W][3] -> [1][3][F][H][W], the oracle's frame layout.
fn planarFrames(hwc: []const u8, out: []u8) void {
    const plane: usize = @as(usize, shapes.sample_frames) * shapes.height * shapes.width;
    for (0..plane) |px| for (0..3) |ch| {
        out[ch * plane + px] = hwc[px * 3 + ch];
    };
}

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
    const l: Layout = .init(1 << 30);
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
