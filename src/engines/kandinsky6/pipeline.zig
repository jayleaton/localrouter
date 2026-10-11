//! Kandinsky-6 Pro distilled, text to video with its coupled audio, on one GPU: the published snapshot loaded into the
//! components' own layouts, then for each request (pipeline/pipeline.py and piflow_sampler.py at upstream eac06e39)
//!   text      Qwen2.5-VL hidden states [1024, 3584] bf16 + key mask, CLIP pooled [768] cast to bf16
//!   noise     torch CUDA randn: video [31, 60, 108, 16] at seed s, audio [218, 40] at seed s + 1 (bf16)
//!   sampler   10 PiFlow steps: visual input, one fused DiT call (both modalities), the DX policy rollout of each
//!   video     the f32 latent through PR #4's decoder (its own loader, `video_vae/weights.zig`) to uint8 RGB
//!   audio     PR #6's decoder and vocoder to the 223,232-sample waveform
//! The position tables (RoPE and the time embeddings' frequencies) depend only on the geometry and are built at load.
//!
//! `create` asks `admission` first: production loads only when every contract is qualified, diagnostic (the check
//! gates) whenever none is unresolved. Precision choices native leaves to run-time libraries are required options; none
//! has a default. The audio stage waits for PR #6's whole-pass entry points (admission reports them unresolved).

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
    hooks: Hooks = .{},
};

/// Per-op observers for the check gates; null in production.
pub const Hooks = struct {
    text: ?qwen25.Hook = null,
    dit: ?dit.Hook = null,
    /// The pipeline's own stages (noise, each step's states, the decoded outputs), same signature as the text hook.
    central: ?qwen25.Hook = null,
};

pub const Progress = struct {
    ctx: *anyopaque,
    report: *const fn (ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void,

    fn at(p: Progress, phase: []const u8, step: u32, of: u32) void {
        p.report(p.ctx, phase, step, of);
    }
};

pub const Request = struct { prompt: []const u8, seed: u64 };

const rgb_bytes: u64 = @as(u64, shapes.sample_frames) * shapes.height * shapes.width * 3;
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
    qwen_tok: tokenizer.Tokenizer,
    clip_tok: prompt.clip_tokenizer.ClipTokenizer,
    dit_store: weights.Store,
    qwen_store: weights.Store,
    clip_store: weights.Store,
    vae_buf: cuda.DeviceBuffer,
    dit_w: dit.Weights,
    text_blocks: [2 * dit.layout.Config.distill.num_text_blocks]dit.TextBlock,
    blocks: [dit.layout.Config.distill.num_visual_blocks]dit.FusedBlock,
    qwen_w: qwen25.Weights,
    clip_w: clip.Weights,
    vae_dec: vae_decode.Decoder,
    vae_plan: vae_tiles.DecodePlan,
    dit_plan: dit.Plan,
    inv_freq: [64]f32,
    buf: cuda.DeviceBuffer,
    lay: Layout,
    steps: [shapes.steps]piflow.Step,
    /// Contracts not yet qualified when admitted (0 in production; a diagnostic run's label otherwise).
    open_contracts: usize,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, root: []const u8, opts: Options) !*Pipeline {
        const open_contracts = try admission.admit(opts.use);
        const p = try gpa.create(Pipeline);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.io = io;
        p.opts = opts;
        p.open_contracts = open_contracts;
        p.inv_freq = try readInvFreq(io, opts.qwen_inv_freq);
        piflow.schedule(.{}, &p.steps);
        p.vae_plan = try vae_gpu.plan(shapes.latent_frames, shapes.latent_h, shapes.latent_w, opts.vae_free_bytes);
        p.dit_plan = try dit.Plan.init(.distill, dit_shape, dit.Plan.default_ffn_rows);

        var tmp: std.heap.ArenaAllocator = .init(gpa);
        defer tmp.deinit();
        const a = tmp.allocator();
        p.qwen_tok = try tokenizer.loadTokenizer(io, gpa, try std.fs.path.join(a, &.{ root, weights.paths.qwen_tokenizer }));
        errdefer p.qwen_tok.deinit();
        p.clip_tok = try .load(gpa, io, try std.fs.path.join(a, &.{ root, weights.paths.clip_tokenizer }));
        errdefer p.clip_tok.deinit();

        p.d = try cuda.Driver.open();
        errdefer p.d.close();
        p.ctx = try cuda.Context.init(&p.d, 0);
        errdefer p.ctx.deinit();
        p.s = try cuda.Stream.init(&p.d, false); // blocking: ordered with the synchronous uploads and downloads
        errdefer p.s.deinit();
        p.dev = try central.device(&p.ctx);
        p.ops = try central.Ops.load(&p.d, kmod.central);
        errdefer p.ops.unload();
        p.text_k = try qwen25.Kernels.load(&p.d, .{ .text = kmod.text, .qwen_te = qi.kernels.te, .qwen_gemm = qi.kernels.gemm, .qwen_attention = qi.kernels.attention });
        errdefer p.text_k.unload();
        p.dit_k = try dit.launch.Kernels.load(&p.d, .{ .gemm = dit_kmod.gemm, .attention = dit_kmod.attention, .ops = dit_kmod.ops });
        errdefer p.dit_k.unload();
        p.vae = try vae_gpu.Gpu.load(&p.d, kmod.vae_conv, kmod.vae_ops, p.s);
        errdefer p.vae.unload();

        var up = try qi.upload.Uploader.init(&p.d, io, p.s);
        defer up.deinit();
        p.dit_store = .init(&p.d, gpa);
        errdefer p.dit_store.deinit();
        p.qwen_store = .init(&p.d, gpa);
        errdefer p.qwen_store.deinit();
        p.clip_store = .init(&p.d, gpa);
        errdefer p.clip_store.deinit();
        {
            var src = try weights.Source.openFile(gpa, io, root, weights.paths.transformer);
            defer src.close(io);
            var l: Loader = .{ .store = &p.dit_store, .io = io, .up = &up, .src = &src, .gpa = gpa, .cast = .bf16 };
            defer l.deinit();
            try dit.layout.forEachTensor(.distill, &l);
            p.dit_w = try dit.bindWeights(.distill, &p.dit_store, &p.text_blocks, &p.blocks);
        }
        {
            var src = try weights.Source.openIndex(gpa, io, root, weights.paths.text_encoder_dir, weights.paths.text_encoder_index);
            defer src.close(io);
            var l: Loader = .{ .store = &p.qwen_store, .io = io, .up = &up, .src = &src, .gpa = gpa, .cast = .as_stored };
            defer l.deinit();
            try qwen25.forEachTensor(&l);
            p.qwen_w = try qwen25.bindWeights(&p.qwen_store);
        }
        {
            var src = try weights.Source.openFile(gpa, io, root, weights.paths.clip);
            defer src.close(io);
            var l: Loader = .{ .store = &p.clip_store, .io = io, .up = &up, .src = &src, .gpa = gpa, .cast = .as_stored };
            defer l.deinit();
            try clip.forEachTensor(&l);
            p.clip_w = try clip.bindWeights(&p.clip_store);
        }
        // PR #4's loader: the whole checkpoint in host memory, one device image, the decoder relocated onto it
        {
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

        const work = @max(
            @max(qwen25.scratchBytes(), clip.scratchBytes()),
            @max(p.dit_plan.bytes(), vae_decode.scratchBytes(p.vae_plan)),
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
        p.clip_store.deinit();
        p.qwen_store.deinit();
        p.dit_store.deinit();
        p.vae.unload();
        p.dit_k.unload();
        p.text_k.unload();
        p.ops.unload();
        p.s.deinit();
        p.ctx.deinit();
        p.d.close();
        p.clip_tok.deinit();
        p.qwen_tok.deinit();
        p.gpa.destroy(p);
    }

    fn at(p: *const Pipeline, off: u64) u64 {
        return p.buf.ptr + off;
    }

    fn upload(p: *Pipeline, off: u64, bytes: []const u8) !void {
        try p.buf.upload(@intCast(off), bytes);
    }

    fn tap(p: *const Pipeline, name: []const u8, off: u64, bytes: u64) !void {
        if (p.opts.hooks.central) |h| try h.after(h.ctx, name, p.at(off), bytes);
    }

    /// The geometry's tables: time embedding frequencies, then each RoPE's arguments on the host, cos / sin on the GPU.
    fn tables(p: *Pipeline) !void {
        const gpa = p.gpa;
        var fv: [rope.time_freqs_v]f32 = undefined;
        rope.freqs(&fv);
        try p.upload(p.lay.freqs_v, std.mem.sliceAsBytes(&fv));
        var fa: [rope.time_freqs_a]f32 = undefined;
        rope.freqs(&fa);
        try p.upload(p.lay.freqs_a, std.mem.sliceAsBytes(&fa));

        var ft: [rope.axes_pairs[0]]f32 = undefined;
        var fh: [rope.axes_pairs[1]]f32 = undefined;
        rope.freqs(&ft);
        rope.freqs(&fh); // h and w share the size, so the values
        const N = dit_shape.tokens();
        const args = try gpa.alloc(f32, N * rope.pairs);
        defer gpa.free(args);
        rope.visualArgs(shapes.latent_frames, shapes.patches_h, shapes.patches_w, &ft, &fh, &fh, args);
        try p.ropeTable(args, p.lay.rope_visual, N);

        var f1: [rope.pairs]f32 = undefined;
        rope.freqs(&f1);
        rope.linearArgs(&f1, args[0 .. @as(usize, shapes.text_rows) * rope.pairs]);
        try p.ropeTable(args[0 .. @as(usize, shapes.text_rows) * rope.pairs], p.lay.rope_text, shapes.text_rows);
        var fs: [rope.pairs]f32 = undefined;
        rope.audioFreqs(&f1, &fs);
        rope.linearArgs(&fs, args[0 .. @as(usize, shapes.audio_rows) * rope.pairs]);
        try p.ropeTable(args[0 .. @as(usize, shapes.audio_rows) * rope.pairs], p.lay.rope_audio, shapes.audio_rows);

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
        const L = p.lay;
        const s = p.s;

        // text
        progress.at("text", 0, 2);
        var ids: [prompt.qwen_tokens]u32 = undefined;
        var mask: [prompt.qwen_tokens]u8 = undefined;
        const valid = try prompt.qwenIds(&p.qwen_tok, p.gpa, r.prompt, &ids, &mask);
        var cids: [shapes.clip_tokens]u32 = undefined;
        const c = try prompt.clipIds(&p.clip_tok, p.gpa, r.prompt, &cids);
        try p.s.synchronize(); // the mask upload below must not race a previous request's DiT
        try p.upload(L.mask, mask[prompt.qwen_crop..]);
        const qe: qwen25.Encoder = .{ .k = &p.text_k, .s = s, .w = &p.qwen_w, .base = p.at(L.work), .inv_freq = p.inv_freq, .rope_freq = math(qwen25.Fp32Math, p.opts.qwen_rope_freq), .hook = p.opts.hooks.text };
        try qe.forward(&ids, valid, p.at(L.text));
        progress.at("text", 1, 2);
        const ce: clip.Encoder = .{ .k = &p.text_k, .s = s, .w = &p.clip_w, .base = p.at(L.work), .linear = math(clip.Fp32Math, p.opts.clip_linear), .hook = p.opts.hooks.text };
        try ce.forward(&cids, c.valid, c.eot, p.at(L.pooled_f32));
        try ce.castPooled(p.at(L.pooled_f32), p.at(L.pooled));

        // noise
        try p.ops.randn(s, p.at(L.noise_v), noise.videoDraw(r.seed, p.dev));
        try p.ops.randn(s, p.at(L.noise_a), try noise.audioDraw(r.seed, p.dev));
        try p.tap("k6.noise.video", L.noise_v, shapes.video_elems * 2);
        try p.tap("k6.noise.audio", L.noise_a, shapes.audio_elems * 2);

        // sampler
        const x: dit.Dit = .{ .k = &p.dit_k, .s = s, .w = &p.dit_w, .p = p.dit_plan, .base = p.at(L.work), .fp32_linear = math(dit.launch.Fp32Linear, p.opts.fp32_linear), .hook = p.opts.hooks.dit };
        const N = dit_shape.tokens();
        const A: u64 = shapes.audio_rows;
        const T: u64 = shapes.text_rows;
        var subs_buf: [piflow.Table.capacity]piflow.Substep = undefined;
        for (p.steps, 0..) |st, i| {
            progress.at("denoise", @intCast(i), shapes.steps);
            const sv: central.Tensor = if (i == 0) .{ .ptr = p.at(L.noise_v), .dtype = .bf16 } else .{ .ptr = p.at(L.state_v), .dtype = .f32 };
            const sa: central.Tensor = if (i == 0) .{ .ptr = p.at(L.noise_a), .dtype = .bf16 } else .{ .ptr = p.at(L.state_a), .dtype = .f32 };
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
            const subs = piflow.substeps(.{}, st, &subs_buf);
            try p.ops.rollout(s, sv, p.at(L.pred_v), p.at(L.state_v), shapes.video_elems, st, subs);
            try p.ops.rollout(s, sa, p.at(L.pred_a), p.at(L.state_a), shapes.audio_elems, st, subs);
            var nb: [48]u8 = undefined;
            try p.tap(try std.fmt.bufPrint(&nb, "k6.piflow.step{d}.video", .{i}), L.state_v, shapes.video_elems * 4);
            try p.tap(try std.fmt.bufPrint(&nb, "k6.piflow.step{d}.audio", .{i}), L.state_a, shapes.audio_elems * 4);
        }

        // video: PR #4 reads the f32 state as it is (upstream divides it by the scaling factor and casts once to fp16)
        progress.at("decode", 0, 1);
        try vae_decode.decode(p.vae.launcher(), &p.vae_dec, p.vae_plan, .{
            .latent = p.at(L.state_v),
            .latent_dtype = .f32,
            .rgb = p.at(L.rgb),
            .scratch = p.at(L.work),
            .scratch_bytes = L.work_bytes,
        });
        try p.tap("k6.video.rgb", L.rgb, rgb_bytes);
        try s.synchronize();
        try p.buf.download(@intCast(L.rgb), frames);

        progress.at("audio", 0, 1);
        try p.audio(wave);
    }

    /// The audio chain (`k6.audio.latent_in` -> mel -> wave) through PR #6's whole-pass entry points, which are not
    /// written yet: admission refuses every use until they exist, and this stage is written against them then.
    fn audio(p: *Pipeline, wave: []f32) !void {
        _ = wave;
        try p.tap("k6.audio.latent_in", p.lay.state_a, shapes.audio_elems * 4);
        if (comptime @hasDecl(audio_vae, "decode") and @hasDecl(vocoder, "vocode")) {
            @compileError("PR #6 now declares decode / vocode: write the audio stage against them");
        } else return error.AudioEntryPointsMissing;
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
    const offs = [_]u64{ l.text, l.mask, l.pooled_f32, l.pooled, l.noise_v, l.noise_a, l.state_v, l.state_a, l.video_in, l.pred_v, l.pred_a, l.times, l.freqs_v, l.freqs_a, l.rope_visual, l.rope_audio, l.rope_text, l.rgb, l.work };
    for (offs[1..], offs[0 .. offs.len - 1]) |b, a| try std.testing.expect(b > a);
    for (offs) |o| try std.testing.expectEqual(@as(u64, 0), o % 256);
    try std.testing.expectEqual(@as(u64, rgb_bytes), l.work - l.rgb);
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
