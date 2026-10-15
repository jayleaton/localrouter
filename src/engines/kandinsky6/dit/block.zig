//! The Kandinsky-6 29B Pro-distill DiT forward on caller-owned CUDA resources: the text encoder blocks
//! (`TransformerEncoderBlock`), the 60 fused video + audio blocks (`FusedTransformerDecoderBlock` with ca_rope,
//! cross_gates and fix_modulation), and the whole call (`PiFlowDiffusionTransformer3D.forward` on the fused path:
//! text / pooled / time embeddings, patch embedding, blocks, both heads with the grid split), op for op as upstream
//! kandinsky-6 eac06e39 `kandinsky/core/components/{dit,piflow_dit}.py` orders them.
//!
//! Ownership. The caller (the pipeline) owns the driver, context and stream, the weights (immutable BF16 views; the
//! loader casts the file's F32 tensors to BF16 round-to-nearest-even as upstream `runtime/weights.py` does), the
//! inputs, the outputs and one scratch region of `Plan.bytes()` bytes; nothing here allocates, synchronizes, loads or
//! reads files. Batch 1 (distill runs one DiT call a step, no CFG batch).
//!
//! Every named step reports its output view to an optional `Hook` (capture and per-op GPU events for the replay gate)
//! under the upstream module path, e.g. `visual_transformer_blocks.7.video_dec_block.self_attention`.

const std = @import("std");
const cuda = @import("cuda");
pub const layout = @import("layout.zig");
pub const attention = @import("attention.zig");
pub const launch = @import("launch.zig");

const View = layout.View;
const Config = layout.Config;
const Shape = layout.Shape;
const Mod = layout.Mod;
const Kernels = launch.Kernels;
pub const Error = launch.Error || error{ UnsupportedShape, MissingWeight, NoSpaceLeft };

/// Called after each named step with its output (on the stream, not yet complete): capture for the twin comparison,
/// or an event record for per-op timing. Errors abort the forward.
pub const Hook = struct {
    ctx: *anyopaque,
    op: *const fn (ctx: *anyopaque, name: []const u8, out: View) anyerror!void,
};

// ---------------------------------------------------------------------------------------------------------------------
// Weights

pub const Linear = struct {
    w: View, // bf16 [out, in]
    b: ?View, // bf16 [1, out]
    k_first: u32 = 0, // launch.Kernels.gemm's K order: the native kernel's partial K tile, 0 for none
};

pub const Attn = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: View, // bf16 [1, 128]
    kn: View,
    heads: u32,
};

pub const Ffn = struct { up: View, down: View }; // net.0.proj [ff, d], net.2 [d, ff], no biases

pub const TextBlock = struct { mod: Linear, attn: Attn, ffn: Ffn };

pub const DecBlock = struct { mod: Linear, sa: Attn, ca: Attn, ffn: Ffn };

pub const FusedBlock = struct {
    video: DecBlock,
    audio: DecBlock,
    va: Attn, // video queries, audio keys
    av: Attn, // audio queries, video keys
    va_mod: Linear, // [2 D + Da, T]
    av_mod: Linear, // [2 Da + D, T]
};

pub const TextEmbedding = struct { in: Linear, norm_w: View, norm_b: View };

pub const Modality = struct {
    time1: Linear,
    time2: Linear,
    text: TextEmbedding,
    pooled: TextEmbedding,
    blocks: []TextBlock,
};

pub const Weights = struct {
    visual_in: Linear,
    type_table: View, // bf16 [2, D]
    out_mod: Linear,
    out: Linear, // [4 * 160, D]
    audio_in: TextEmbedding,
    audio_out_mod: Linear,
    audio_out: Linear, // [400, Da]
    video: Modality,
    audio: Modality,
    blocks: []FusedBlock,
};

/// Binds every weight view by name from `lookup.get(name) ?u64` (the device address of the BF16 tensor; shapes are the
/// manifest's, which the loader has checked). `text_blocks` holds 2 * num_text_blocks, `blocks` num_visual_blocks.
pub fn bindWeights(c: Config, lookup: anytype, text_blocks: []TextBlock, blocks: []FusedBlock) Error!Weights {
    std.debug.assert(text_blocks.len == 2 * c.num_text_blocks and blocks.len == c.num_visual_blocks);
    const B = struct {
        fn t(l: anytype, comptime fmt: []const u8, args: anytype, rows: u64, cols: u64) Error!View {
            var nb: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, fmt, args);
            const p = l.get(name) orelse return error.MissingWeight;
            return View.of(p, .bf16, rows, cols);
        }
        fn lin(l: anytype, comptime fmt: []const u8, args: anytype, out: u64, in: u64, bias: bool) Error!Linear {
            return .{
                .w = try t(l, fmt ++ ".weight", args, out, in),
                .b = if (bias) try t(l, fmt ++ ".bias", args, 1, out) else null,
            };
        }
        fn attn(l: anytype, comptime fmt: []const u8, args: anytype, q: u64, kv: u64) Error!Attn {
            return .{
                .q = try lin(l, fmt ++ ".to_query", args, q, q, true),
                .k = try lin(l, fmt ++ ".to_key", args, q, kv, true),
                .v = try lin(l, fmt ++ ".to_value", args, q, kv, true),
                .o = try lin(l, fmt ++ ".out_layer", args, q, q, true),
                .qn = try t(l, fmt ++ ".query_norm.weight", args, 1, 128),
                .kn = try t(l, fmt ++ ".key_norm.weight", args, 1, 128),
                .heads = @intCast(q / 128),
            };
        }
        fn ffn(l: anytype, comptime fmt: []const u8, args: anytype, d: u64, ff: u64) Error!Ffn {
            return .{
                .up = try t(l, fmt ++ ".feed_forward.net.0.proj.weight", args, ff, d),
                .down = try t(l, fmt ++ ".feed_forward.net.2.weight", args, d, ff),
            };
        }
        fn emb(l: anytype, comptime p: []const u8, out: u64, in: u64) Error!TextEmbedding {
            return .{
                .in = try lin(l, p ++ ".in_layer", .{}, out, in, true),
                .norm_w = try t(l, p ++ ".norm.weight", .{}, 1, out),
                .norm_b = try t(l, p ++ ".norm.bias", .{}, 1, out),
            };
        }
        fn dec(l: anytype, comptime p: []const u8, i: usize, d: u64, ff: u64, T: u64) Error!DecBlock {
            const f = "visual_transformer_blocks.{d}." ++ p;
            return .{
                .mod = try lin(l, f ++ ".visual_modulation.out_layer", .{i}, 9 * d, T, true),
                .sa = try attn(l, f ++ ".self_attention", .{i}, d, d),
                .ca = try attn(l, f ++ ".cross_attention", .{i}, d, d),
                .ffn = try ffn(l, f, .{i}, d, ff),
            };
        }
    };
    const D: u64 = c.model_dim;
    const Da: u64 = c.model_dim_a;
    const T: u64 = c.time_dim;
    var w: Weights = .{
        .visual_in = blk: {
            var l = try B.lin(lookup, "visual_embeddings.in_layer", .{}, D, c.patchIn(), true);
            l.k_first = c.patchKFirst();
            break :blk l;
        },
        .type_table = try B.t(lookup, "visual_token_type_embeddings.weight", .{}, c.token_types, D),
        .out_mod = try B.lin(lookup, "out_layer.modulation.out_layer", .{}, 2 * D, T, true),
        .out = try B.lin(lookup, "out_layer.out_layer", .{}, 4 * @as(u64, c.out_visual_dim), D, true),
        .audio_in = try B.emb(lookup, "audio_embeddings", Da, c.in_audio_dim),
        .audio_out_mod = try B.lin(lookup, "audio_out_layer.modulation.out_layer", .{}, 2 * Da, T, true),
        .audio_out = try B.lin(lookup, "audio_out_layer.out_layer", .{}, c.out_audio_dim, Da, true),
        .video = undefined,
        .audio = undefined,
        .blocks = blocks,
    };
    const nt = c.num_text_blocks;
    inline for (.{ .{ "video", D, @as(u64, c.ff_dim), 0 }, .{ "audio", Da, @as(u64, c.ff_dim_a), 1 } }) |m| {
        const md: u64 = m[1];
        const tb = text_blocks[m[3] * nt ..][0..nt];
        for (tb, 0..) |*b, i| b.* = .{
            .mod = try B.lin(lookup, m[0] ++ "_text_transformer_blocks.{d}.text_modulation.out_layer", .{i}, 6 * md, T, true),
            .attn = try B.attn(lookup, m[0] ++ "_text_transformer_blocks.{d}.attn", .{i}, md, md),
            .ffn = try B.ffn(lookup, m[0] ++ "_text_transformer_blocks.{d}", .{i}, md, m[2]),
        };
        @field(w, m[0]) = .{
            .time1 = try B.lin(lookup, m[0] ++ "_time_embeddings.timestep_embedder.linear_1", .{}, T, md, true),
            .time2 = try B.lin(lookup, m[0] ++ "_time_embeddings.timestep_embedder.linear_2", .{}, T, T, true),
            .text = try B.emb(lookup, m[0] ++ "_text_embeddings", md, c.in_text_dim),
            .pooled = try B.emb(lookup, m[0] ++ "_pooled_text_embeddings", T, c.in_text_dim2),
            .blocks = tb,
        };
    }
    for (blocks, 0..) |*b, i| b.* = .{
        .video = try B.dec(lookup, "video_dec_block", i, D, c.ff_dim, T),
        .audio = try B.dec(lookup, "audio_dec_block", i, Da, c.ff_dim_a, T),
        .va = try B.attn(lookup, "visual_transformer_blocks.{d}.va_cross_attention", .{i}, D, Da),
        .av = try B.attn(lookup, "visual_transformer_blocks.{d}.av_cross_attention", .{i}, Da, D),
        .va_mod = try B.lin(lookup, "visual_transformer_blocks.{d}.va_modulation.out_layer", .{i}, 2 * D + Da, T, true),
        .av_mod = try B.lin(lookup, "visual_transformer_blocks.{d}.av_modulation.out_layer", .{i}, 2 * Da + D, T, true),
    };
    return w;
}

// ---------------------------------------------------------------------------------------------------------------------
// Scratch

/// The scratch layout of one DiT call: every buffer the forward needs beyond the caller's inputs and outputs, as
/// 256-byte aligned offsets into one region. The visual token count dominates: at 50,220 tokens the eight [N, 4096]
/// bf16 buffers take ~3.07 GiB; the FFN hidden is bounded by `ffn_rows` (rows are independent, so chunking it keeps
/// the bits). No buffer is quadratic in the sequence.
pub const Plan = struct {
    c: Config,
    sh: Shape,
    ffn_rows: u64,
    rows: u64, // max(visual tokens, text tokens): the stream scratch serves the text blocks too
    // stream scratch, [rows, D] each: normalized input, q, k, v, attention output, projection output, pre-CA, text CA
    v: [8]u64,
    a: [7]u64, // audio [max(A, L), Da]: an, aq, ak, av, aa, ao, apre
    hidden: u64, // [ffn_rows, max(ff, ff_a)]
    vis: u64, // residual streams: [N, D], [A, Da]
    aud: u64,
    text_v: u64, // [L, D], [L, Da]: the text blocks' outputs (cross-attention keys)
    text_a: u64,
    rows_in: u64, // [N, 132] patch rows / [N, 640] head rows (bf16)
    audio_in: u64, // [A, 40] bf16
    mod: u64, // one block's modulation rows (bf16): video 9D, audio 9Da, va, av; text 6D
    mod2: u64,
    mod3: u64,
    mod4: u64,
    tm_v: u64, // [1, T] bf16 each
    tm_a: u64,
    t_sin: u64, // [1, D] f32
    t_h: u64, // [1, T] f32
    t_emb: u64, // [1, T] bf16
    pooled: u64, // [1, T] bf16
    head: u64, // [max(N, A), max(640, 400)] bf16
    total: u64,

    pub const default_ffn_rows: u64 = 4096;

    pub fn init(c: Config, sh: Shape, ffn_rows: u64) Error!Plan {
        try sh.validate();
        if (ffn_rows == 0) return error.UnsupportedShape;
        const N = sh.tokens();
        const A: u64 = sh.audio;
        const L: u64 = sh.text;
        const D: u64 = c.model_dim;
        const Da: u64 = c.model_dim_a;
        const T: u64 = c.time_dim;
        var p: Plan = undefined;
        p.c = c;
        p.sh = sh;
        p.ffn_rows = ffn_rows;
        p.rows = @max(N, L, A); // video k / v also hold the V <- A keys [A, D] and the text keys [L, D]
        var at: u64 = 0;
        const put = struct {
            fn f(cur: *u64, size: u64) u64 {
                const o = cur.*;
                cur.* = (cur.* + size + 255) / 256 * 256;
                return o;
            }
        }.f;
        for (&p.v) |*o| o.* = put(&at, p.rows * D * 2);
        const arows = @max(A, L);
        for (&p.a) |*o| o.* = put(&at, arows * Da * 2);
        p.hidden = put(&at, ffn_rows * @max(c.ff_dim, c.ff_dim_a) * 2);
        p.vis = put(&at, N * D * 2);
        p.aud = put(&at, A * Da * 2);
        p.text_v = put(&at, L * D * 2);
        p.text_a = put(&at, L * Da * 2);
        p.rows_in = put(&at, N * c.patchIn() * 2);
        p.audio_in = put(&at, A * c.in_audio_dim * 2);
        p.mod = put(&at, 9 * D * 2);
        p.mod2 = put(&at, 9 * Da * 2);
        p.mod3 = put(&at, (2 * D + Da) * 2);
        p.mod4 = put(&at, (2 * Da + D) * 2);
        p.tm_v = put(&at, T * 2);
        p.tm_a = put(&at, T * 2);
        p.t_sin = put(&at, D * 4);
        p.t_h = put(&at, T * 4);
        p.t_emb = put(&at, T * 2);
        p.pooled = put(&at, T * 2);
        p.head = put(&at, @max(N, A) * @max(4 * c.out_visual_dim, c.out_audio_dim) * 2);
        p.total = at;
        return p;
    }

    pub fn bytes(p: Plan) u64 {
        return p.total;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// The forward

/// One DiT call's inputs (B = 1), as the native pipeline passes them (`piflow_sampler.py`):
pub const Inputs = struct {
    x_video: View, // [T * H * W, 33] fp32 (bf16 at step 0): cat(latent 16, cond 16, mask 1), channels last
    x_audio: View, // [A, 40] fp32 / bf16
    text: View, // [L, 3584] bf16, the cropped Qwen hidden states padded to L
    pooled: View, // [1, 768] bf16 (CLIP pooler output)
    time_v: View, // [1, 1] fp32: sigma * 1000 (video)
    time_a: View, // [1, 1] fp32 (audio; equal to time_v in the distill sampler, still read separately)
    freqs_v: View, // [1, D / 2] fp32: video_time_embeddings.freqs as the module holds it
    freqs_a: View, // [1, Da / 2] fp32
    visual_rope: View, // [N, 256] fp32: RoPE3D's table, flattened (t, h', w')
    audio_rope: View, // [A, 256] fp32: RoPE1D(128, freqs_scaling 0.144) at positions 0 .. A - 1
    text_rope_v: View, // [L, 256] fp32: RoPE1D(128) at 0 .. L - 1 (video text stack)
    text_rope_a: View, // [L, 256] fp32 (audio text stack)
    mask: attention.Mask, // the text key-padding mask (True = keep)
};

/// The PiFlow head outputs, grid-major and contiguous: video [10 * T * H * W, 16] = [10, T, H, W, 16], audio
/// [10 * A, 40] = [10, A, 40], bf16 (`video.view(.., n_grid, 16).movedim(-2, 1)` and the audio alike).
pub const Outputs = struct { video: View, audio: View };

pub const Dit = struct {
    k: *const Kernels,
    s: cuda.Stream,
    w: *const Weights,
    p: Plan,
    base: u64, // the scratch region, `p.bytes()` bytes, 256-byte aligned
    fp32_linear: launch.Fp32Linear, // no default: native's `high` policy permits TF32; executed algorithm uncaptured, unqualified
    hook: ?Hook = null,

    fn buf(x: *const Dit, off: u64, rows: u64, cols: u64, dt: layout.Dtype) View {
        return View.of(x.base + off, dt, rows, cols);
    }

    fn emit(x: *const Dit, comptime fmt: []const u8, args: anytype, out: View) Error!void {
        const h = x.hook orelse return;
        var nb: [192]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, fmt, args);
        h.op(h.ctx, name, out) catch return error.Invalid;
    }

    fn linear(x: *const Dit, l: Linear, in: View, out: View) Error!void {
        try x.k.gemm(x.s, in, l.w, l.b, out, .bias, l.k_first);
    }

    /// Modulation: bf16(silu(float(tm)) . float(W)^T + float(b)) into `out` [1, n].
    fn modulation(x: *const Dit, l: Linear, tm: View, out: View) Error!void {
        try x.k.gemv(x.s, tm, l.w, l.b, out, true, x.fp32_linear);
    }

    /// FeedForward on `in` [R, d] into `out` [R, d] (rows in chunks of ffn_rows through the hidden buffer).
    fn ffn(x: *const Dit, f: Ffn, in: View, out: View) Error!void {
        const ff = f.up.rows;
        var r: u64 = 0;
        while (r < in.rows) : (r += x.p.ffn_rows) {
            const n = @min(x.p.ffn_rows, in.rows - r);
            const h = x.buf(x.p.hidden, n, ff, .bf16);
            try x.k.gemm(x.s, in.slice(r, n), f.up, null, h, .gelu, 0);
            try x.k.gemm(x.s, h, f.down, null, out.slice(r, n), .bias, 0);
        }
    }

    const Bufs = struct { n: View, q: View, k: View, v: View, a: View, o: View };

    /// One attention module: q from `xq`, k / v from `xkv` (the same view for self-attention), per-head RMSNorm, the
    /// optional ropes, streaming attention, out_layer into b.o. b.q / b.a are [Rq, dq]; b.k / b.v at least [Rkv, dq].
    fn attn(x: *const Dit, m: Attn, xq: View, xkv: View, rope_q: ?View, rope_k: ?View, mask: attention.Mask, b: Bufs) Error!void {
        const dq = m.q.w.rows;
        const q = b.q.columns(0, dq).slice(0, xq.rows);
        const k = View.of(b.k.ptr, .bf16, xkv.rows, dq);
        const v = View.of(b.v.ptr, .bf16, xkv.rows, dq);
        const a = b.a.columns(0, dq).slice(0, xq.rows);
        try x.linear(m.q, xq, q);
        try x.linear(m.k, xkv, k);
        try x.linear(m.v, xkv, v);
        try x.k.qkNormRope(x.s, q, m.qn, rope_q, m.heads);
        try x.k.qkNormRope(x.s, k, m.kn, rope_k, m.heads);
        try x.k.attend(x.s, .{ .q = q, .k = k, .v = v, .out = a, .heads = m.heads, .mask = mask });
        try x.linear(m.o, a, b.o.columns(0, dq).slice(0, xq.rows));
    }

    fn videoBufs(x: *const Dit, rows: u64, d: u64) Bufs {
        return .{
            .n = x.buf(x.p.v[0], rows, d, .bf16),
            .q = x.buf(x.p.v[1], rows, d, .bf16),
            .k = x.buf(x.p.v[2], x.p.rows, d, .bf16),
            .v = x.buf(x.p.v[3], x.p.rows, d, .bf16),
            .a = x.buf(x.p.v[4], rows, d, .bf16),
            .o = x.buf(x.p.v[5], rows, d, .bf16),
        };
    }

    fn audioBufs(x: *const Dit, rows: u64, d: u64) Bufs {
        const ar = @max(@as(u64, x.p.sh.audio), x.p.sh.text);
        return .{
            .n = x.buf(x.p.a[0], rows, d, .bf16),
            .q = x.buf(x.p.a[1], rows, d, .bf16),
            .k = x.buf(x.p.a[2], ar, d, .bf16),
            .v = x.buf(x.p.a[3], ar, d, .bf16),
            .a = x.buf(x.p.a[4], rows, d, .bf16),
            .o = x.buf(x.p.a[5], rows, d, .bf16),
        };
    }

    /// TransformerEncoderBlock on te [L, md] in place (`prefix` "video" or "audio").
    pub fn textBlock(x: *const Dit, b: TextBlock, i: usize, comptime prefix: []const u8, te: View, tm: View, rope: View, mask: attention.Mask) Error!void {
        const md = te.cols;
        const L = te.rows;
        const bufs = x.videoBufs(L, md); // the text blocks run before the visual blocks: the stream scratch is free
        const mod = x.buf(x.p.mod, 1, 6 * md, .bf16);
        try x.modulation(b.mod, tm, mod);
        try x.emit(prefix ++ "_text_transformer_blocks.{d}.text_modulation", .{i}, mod);
        try x.k.normMod(x.s, te, mod.columns(Mod.enc(md, 0, 1), md), mod.columns(Mod.enc(md, 0, 0), md), bufs.n);
        try x.attn(b.attn, bufs.n, bufs.n, rope, rope, mask, bufs);
        try x.emit(prefix ++ "_text_transformer_blocks.{d}.attn", .{i}, bufs.o);
        try x.k.gateSum(x.s, te, bufs.o, mod.columns(Mod.enc(md, 0, 2), md));
        try x.k.normMod(x.s, te, mod.columns(Mod.enc(md, 1, 1), md), mod.columns(Mod.enc(md, 1, 0), md), bufs.n);
        try x.ffn(b.ffn, bufs.n, bufs.o);
        try x.emit(prefix ++ "_text_transformer_blocks.{d}.feed_forward", .{i}, bufs.o);
        try x.k.gateSum(x.s, te, bufs.o, mod.columns(Mod.enc(md, 1, 2), md));
        try x.emit(prefix ++ "_text_transformer_blocks.{d}", .{i}, te);
    }

    pub const BlockIo = struct {
        vis: View, // [N, D] residual, in place
        aud: View, // [A, Da] residual, in place
        text_v: View, // [L, D]
        text_a: View, // [L, Da]
        tm_v: View, // [1, T] bf16
        tm_a: View,
        vis_rope: View, // [N, 256] f32
        aud_rope: View, // [A, 256] f32
        mask: attention.Mask,
    };

    /// FusedTransformerDecoderBlock.forward (dit.py 436-517) with ca_rope, cross_gates and fix_modulation, gate scales
    /// 1 and no modality mask (both bitwise no-ops upstream). The order is upstream's: the video text cross-attention
    /// is computed before the audio stream runs but added only after the audio text cross-attention (the deferred
    /// residual); both cross-modal attentions take their keys from the other stream's pre-CA tensor and their queries
    /// from the stream after its text residual; the video residual takes the av gate and the audio residual the va gate.
    pub fn forwardBlock(x: *const Dit, b: FusedBlock, i: usize, io: BlockIo) Error!void {
        const c = x.p.c;
        const D: u64 = c.model_dim;
        const Da: u64 = c.model_dim_a;
        const N = io.vis.rows;
        const A = io.aud.rows;
        const vb = x.videoBufs(N, D);
        const ab = x.audioBufs(A, Da);
        const vpre = x.buf(x.p.v[6], N, D, .bf16);
        const vct = x.buf(x.p.v[7], N, D, .bf16);
        const apre = x.buf(x.p.a[6], A, Da, .bf16);
        const vm = x.buf(x.p.mod, 1, 9 * D, .bf16);
        const am = x.buf(x.p.mod2, 1, 9 * Da, .bf16);
        const vam = x.buf(x.p.mod3, 1, 2 * D + Da, .bf16);
        const avm = x.buf(x.p.mod4, 1, 2 * Da + D, .bf16);
        const p = "visual_transformer_blocks.{d}.";

        // ---- video backbone: self-attention, then the text cross-attention computed but not yet added
        try x.modulation(b.video.mod, io.tm_v, vm);
        try x.emit(p ++ "video_dec_block.visual_modulation", .{i}, vm);
        try x.k.normMod(x.s, io.vis, vm.columns(Mod.dec(D, 0, 1), D), vm.columns(Mod.dec(D, 0, 0), D), vb.n);
        try x.attn(b.video.sa, vb.n, vb.n, io.vis_rope, io.vis_rope, .none, vb);
        try x.emit(p ++ "video_dec_block.self_attention", .{i}, vb.o);
        try x.k.gateSum(x.s, io.vis, vb.o, vm.columns(Mod.dec(D, 0, 2), D));
        try x.emit(p ++ "video_dec_block.self_attention_residual", .{i}, io.vis);
        try x.k.normMod(x.s, io.vis, vm.columns(Mod.dec(D, 1, 1), D), vm.columns(Mod.dec(D, 1, 0), D), vpre);
        try x.emit(p ++ "video_dec_block.pre_ca", .{i}, vpre);
        try x.attn(b.video.ca, vpre, io.text_v, null, null, io.mask, .{ .n = vb.n, .q = vb.q, .k = vb.k, .v = vb.v, .a = vb.a, .o = vct });
        try x.emit(p ++ "video_dec_block.cross_attention", .{i}, vct);

        // ---- audio backbone: self-attention, text cross-attention added at once
        try x.modulation(b.audio.mod, io.tm_a, am);
        try x.emit(p ++ "audio_dec_block.visual_modulation", .{i}, am);
        try x.k.normMod(x.s, io.aud, am.columns(Mod.dec(Da, 0, 1), Da), am.columns(Mod.dec(Da, 0, 0), Da), ab.n);
        try x.attn(b.audio.sa, ab.n, ab.n, io.aud_rope, io.aud_rope, .none, ab);
        try x.emit(p ++ "audio_dec_block.self_attention", .{i}, ab.o);
        try x.k.gateSum(x.s, io.aud, ab.o, am.columns(Mod.dec(Da, 0, 2), Da));
        try x.k.normMod(x.s, io.aud, am.columns(Mod.dec(Da, 1, 1), Da), am.columns(Mod.dec(Da, 1, 0), Da), apre);
        try x.emit(p ++ "audio_dec_block.pre_ca", .{i}, apre);
        try x.attn(b.audio.ca, apre, io.text_a, null, null, io.mask, ab);
        try x.emit(p ++ "audio_dec_block.cross_attention", .{i}, ab.o);
        try x.k.gateSum(x.s, io.aud, ab.o, am.columns(Mod.dec(Da, 1, 2), Da));

        // ---- cross-modal: va_modulation(t_v), av_modulation(t_a) (fix_modulation)
        try x.modulation(b.va_mod, io.tm_v, vam);
        try x.emit(p ++ "va_modulation", .{i}, vam);
        try x.modulation(b.av_mod, io.tm_a, avm);
        try x.emit(p ++ "av_modulation", .{i}, avm);
        const va_shift = Mod.cross(D, Da, 0);
        const va_scale = Mod.cross(D, Da, 1);
        const va_gate = Mod.cross(D, Da, 2); // [Da]: the audio residual's gate (cross_gates)
        const av_shift = Mod.cross(Da, D, 0);
        const av_scale = Mod.cross(Da, D, 1);
        const av_gate = Mod.cross(Da, D, 2); // [D]: the video residual's gate
        try x.k.gateSum(x.s, io.vis, vct, vm.columns(Mod.dec(D, 1, 2), D)); // the deferred video text residual
        try x.emit(p ++ "video_dec_block.cross_attention_residual", .{i}, io.vis);
        try x.k.normMod(x.s, io.vis, vam.columns(va_scale[0], va_scale[1]), vam.columns(va_shift[0], va_shift[1]), vb.n);
        try x.k.normMod(x.s, io.aud, avm.columns(av_scale[0], av_scale[1]), avm.columns(av_shift[0], av_shift[1]), ab.n);
        // V <- A: queries vis_for_va (visual rope), keys / values aud_pre_ca (audio rope); 32 heads of the video width
        try x.attn(b.va, vb.n, apre, io.vis_rope, io.aud_rope, .none, vb);
        try x.emit(p ++ "va_cross_attention", .{i}, vb.o);
        // A <- V: queries aud_for_av (audio rope), keys / values vis_pre_ca (visual rope); 16 heads of the audio width.
        // Its key / value projections [N, Da] use the video stream's k / v buffers.
        try x.attn(b.av, ab.n, vpre, io.aud_rope, io.vis_rope, .none, .{ .n = ab.n, .q = ab.q, .k = vb.k, .v = vb.v, .a = ab.a, .o = ab.o });
        try x.emit(p ++ "av_cross_attention", .{i}, ab.o);
        try x.k.gateSum(x.s, io.vis, vb.o, avm.columns(av_gate[0], av_gate[1]));
        try x.k.gateSum(x.s, io.aud, ab.o, vam.columns(va_gate[0], va_gate[1]));

        // ---- FFN
        try x.k.normMod(x.s, io.vis, vm.columns(Mod.dec(D, 2, 1), D), vm.columns(Mod.dec(D, 2, 0), D), vb.n);
        try x.ffn(b.video.ffn, vb.n, vb.o);
        try x.emit(p ++ "video_dec_block.feed_forward", .{i}, vb.o);
        try x.k.gateSum(x.s, io.vis, vb.o, vm.columns(Mod.dec(D, 2, 2), D));
        try x.k.normMod(x.s, io.aud, am.columns(Mod.dec(Da, 2, 1), Da), am.columns(Mod.dec(Da, 2, 0), Da), ab.n);
        try x.ffn(b.audio.ffn, ab.n, ab.o);
        try x.emit(p ++ "audio_dec_block.feed_forward", .{i}, ab.o);
        try x.k.gateSum(x.s, io.aud, ab.o, am.columns(Mod.dec(Da, 2, 2), Da));
        try x.emit(p ++ "video", .{i}, io.vis);
        try x.emit(p ++ "audio", .{i}, io.aud);
    }

    /// TextEmbeddings: bf16 in_layer (+bias) then the affine LayerNorm, into `out` [R, d] (in may be the bf16 input).
    fn textEmbedding(x: *const Dit, e: TextEmbedding, in: View, out: View) Error!void {
        try x.linear(e.in, in, out);
        try x.k.layerNorm(x.s, out, e.norm_w, e.norm_b, out);
    }

    /// `_encode_text(prefix)`: token and pooled projections, the time embedding + pooled, the four text blocks.
    fn encodeText(x: *const Dit, m: Modality, comptime prefix: []const u8, in: Inputs, time: View, freqs: View, rope: View, te: View, tm: View) Error!void {
        const T: u64 = x.p.c.time_dim;
        try x.textEmbedding(m.text, in.text, te);
        try x.emit(prefix ++ "_text_embeddings", .{}, te);
        const pooled = x.buf(x.p.pooled, 1, T, .bf16);
        try x.textEmbedding(m.pooled, in.pooled, pooled);
        try x.emit(prefix ++ "_pooled_text_embeddings", .{}, pooled);
        // TimeEmbeddings: [cos, sin] of outer(t, freqs) -> fp32 linear_1 -> SiLU -> fp32 linear_2 -> bf16
        const sincos = x.buf(x.p.t_sin, 1, 2 * freqs.cols, .f32);
        const h = x.buf(x.p.t_h, 1, T, .f32);
        const temb = x.buf(x.p.t_emb, 1, T, .bf16);
        try x.k.timeSincos(x.s, time, freqs, sincos);
        try x.k.gemv(x.s, sincos, m.time1.w, m.time1.b, h, false, x.fp32_linear);
        try x.k.gemv(x.s, h, m.time2.w, m.time2.b, temb, true, x.fp32_linear);
        try x.emit(prefix ++ "_time_embeddings", .{}, temb);
        try x.k.addBf16(x.s, temb, pooled, tm);
        try x.emit(prefix ++ "_time_plus_pooled", .{}, tm);
        for (m.blocks, 0..) |b, i| try x.textBlock(b, i, prefix, te, tm, rope, in.mask);
    }

    /// The whole fused forward (`DiffusionTransformer3D.forward` with both modalities, then the PiFlow grid split).
    pub fn forward(x: *const Dit, in: Inputs, out: Outputs) Error!void {
        const c = x.p.c;
        const sh = x.p.sh;
        const D: u64 = c.model_dim;
        const Da: u64 = c.model_dim_a;
        const T: u64 = c.time_dim;
        const N = sh.tokens();
        const A: u64 = sh.audio;
        const L: u64 = sh.text;
        try in.text.expect(.bf16, L, c.in_text_dim);
        try in.pooled.expect(.bf16, 1, c.in_text_dim2);
        try in.visual_rope.expect(.f32, N, 4 * layout.rope_pairs);
        try in.audio_rope.expect(.f32, A, 4 * layout.rope_pairs);
        try in.text_rope_v.expect(.f32, L, 4 * layout.rope_pairs);
        try in.text_rope_a.expect(.f32, L, 4 * layout.rope_pairs);
        try in.freqs_v.expect(.f32, 1, D / 2);
        try in.freqs_a.expect(.f32, 1, Da / 2);
        try out.video.expect(.bf16, @as(u64, c.n_grid) * sh.t * sh.h * sh.w, c.dxVisual());
        try out.audio.expect(.bf16, @as(u64, c.n_grid) * A, c.dxAudio());
        switch (in.mask) {
            .none => {},
            .keys => |m| if (m.valid > L) return error.ViewMismatch,
        }

        const te_v = x.buf(x.p.text_v, L, D, .bf16);
        const te_a = x.buf(x.p.text_a, L, Da, .bf16);
        const tm_v = x.buf(x.p.tm_v, 1, T, .bf16);
        const tm_a = x.buf(x.p.tm_a, 1, T, .bf16);
        try x.encodeText(x.w.video, "video", in, in.time_v, in.freqs_v, in.text_rope_v, te_v, tm_v);
        try x.encodeText(x.w.audio, "audio", in, in.time_a, in.freqs_a, in.text_rope_a, te_a, tm_a);

        // visual and audio embeddings
        const vis = x.buf(x.p.vis, N, D, .bf16);
        const aud = x.buf(x.p.aud, A, Da, .bf16);
        const prow = x.buf(x.p.rows_in, N, c.patchIn(), .bf16);
        try x.k.patchify(x.s, in.x_video, prow, sh, c.visualIn());
        try x.linear(x.w.visual_in, prow, vis);
        try x.emit("visual_embeddings", .{}, vis);
        const ain = x.buf(x.p.audio_in, A, c.in_audio_dim, .bf16);
        const audio_in = switch (in.x_audio.dtype) {
            .f32 => blk: {
                try x.k.castBf16(x.s, in.x_audio, ain);
                break :blk ain;
            },
            else => in.x_audio,
        };
        try x.textEmbedding(x.w.audio_in, audio_in, aud);
        try x.emit("audio_embeddings", .{}, aud);

        const io: BlockIo = .{
            .vis = vis,
            .aud = aud,
            .text_v = te_v,
            .text_a = te_a,
            .tm_v = tm_v,
            .tm_a = tm_a,
            .vis_rope = in.visual_rope,
            .aud_rope = in.audio_rope,
            .mask = in.mask,
        };
        for (x.w.blocks, 0..) |b, i| try x.forwardBlock(b, i, io);

        // OutLayer: (shift, scale) modulation, norm_mod, out_layer, unpatchify + grid split
        const om = x.buf(x.p.mod, 1, 2 * D, .bf16);
        try x.modulation(x.w.out_mod, tm_v, om);
        const vn = x.buf(x.p.v[0], N, D, .bf16);
        try x.k.normMod(x.s, vis, om.columns(Mod.out(D, 1), D), om.columns(Mod.out(D, 0), D), vn);
        const head_v = x.buf(x.p.head, N, 4 * @as(u64, c.out_visual_dim), .bf16);
        try x.linear(x.w.out, vn, head_v);
        try x.emit("out_layer", .{}, head_v);
        try x.k.unpatchify(x.s, head_v, out.video, sh, c.out_visual_dim, c.n_grid);
        // OutLayerAudio: modulation, norm_mod, the second (non-affine) LayerNorm on bf16, out_layer, grid split
        const am = x.buf(x.p.mod2, 1, 2 * Da, .bf16);
        try x.modulation(x.w.audio_out_mod, tm_a, am);
        const an = x.buf(x.p.a[0], A, Da, .bf16);
        try x.k.normMod(x.s, aud, am.columns(Mod.out(Da, 1), Da), am.columns(Mod.out(Da, 0), Da), an);
        try x.k.layerNorm(x.s, an, null, null, an);
        const head_a = x.buf(x.p.head, A, c.out_audio_dim, .bf16);
        try x.linear(x.w.audio_out, an, head_a);
        try x.emit("audio_out_layer", .{}, head_a);
        try x.k.gridMajor(x.s, head_a, out.audio, c.n_grid);
        try x.emit("video_velocity_grid", .{}, out.video);
        try x.emit("audio_velocity_grid", .{}, out.audio);
    }
};

const testing = std.testing;

test {
    _ = layout;
    _ = attention;
    _ = launch;
}

test "scratch plan: linear in tokens, bounded FFN, no quadratic buffer" {
    const c = Config.distill;
    const p = try Plan.init(c, Shape.preset(1024), Plan.default_ffn_rows);
    const N: u64 = 50220;
    // eight [N, 4096] stream buffers + the residual: ~3.4 GiB; the whole plan stays under 4.5 GiB
    try testing.expect(p.bytes() > 9 * N * 4096 * 2);
    try testing.expect(p.bytes() < 4608 << 20);
    // the FFN hidden is ffn_rows x 16384 bf16, not N x 16384
    try testing.expectEqual(@as(u64, 4096 * 16384 * 2), p.vis - p.hidden);
    try testing.expectError(error.UnsupportedShape, Plan.init(c, Shape.preset(2048), 4096));
    // a video shorter than the audio: the shared k / v buffers still fit the V <- A keys
    const small = try Plan.init(c, .{ .t = 1, .h = 8, .w = 8, .audio = 218, .text = 96 }, 64);
    try testing.expectEqual(@as(u64, 218), small.rows);
}

test "weights bind by the manifest's names" {
    const c = Config.distill;
    // A lookup that answers with a distinct address per manifest name and counts the questions.
    const Look = struct {
        names: std.StringHashMapUnmanaged(u64) = .empty,
        gpa: std.mem.Allocator,
        asked: u64 = 0,
        next: u64 = 0x1000,
        pub fn tensor(self: *@This(), name: []const u8, dt: layout.Dtype, shape: []const u64) !void {
            _ = dt;
            _ = shape;
            try self.names.put(self.gpa, try self.gpa.dupe(u8, name), self.next);
            self.next += 0x100;
        }
        pub fn get(self: *@This(), name: []const u8) ?u64 {
            self.asked += 1;
            return self.names.get(name);
        }
    };
    var look: Look = .{ .gpa = testing.allocator };
    defer {
        var it = look.names.keyIterator();
        while (it.next()) |k| testing.allocator.free(k.*);
        look.names.deinit(testing.allocator);
    }
    try layout.forEachTensor(c, &look);
    var tb: [8]TextBlock = undefined;
    var fb: [60]FusedBlock = undefined;
    const w = try bindWeights(c, &look, &tb, &fb);
    try testing.expectEqual(@as(u64, 4471), look.asked); // every tensor bound once, none unknown
    try testing.expectEqual(@as(u32, 32), w.blocks[59].va.heads);
    try testing.expectEqual(@as(u32, 16), w.blocks[0].av.heads);
    try testing.expectEqual(@as(u64, 2048), w.blocks[0].av.k.w.rows);
    try testing.expectEqual(@as(u64, 4096), w.blocks[0].av.k.w.cols);
    try testing.expectEqual(@as(u64, 10240), w.blocks[0].va_mod.w.rows);
    try testing.expectEqual(@as(u64, 640), w.out.w.rows);
    try testing.expectEqual(@as(u64, 7168), w.audio.blocks[3].ffn.up.rows);
}
