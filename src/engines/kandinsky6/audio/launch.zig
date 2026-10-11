//! Launches of the Kandinsky-6 audio kernels (kernels/cuda/kandinsky6/audio/audio.cu) on the pipeline's borrowed
//! stream: every wrapper validates its views (device, shape, contiguity, bounds) and admits its operator contract
//! (vae.zig `admitOp`) before launching, so a wrong layout or an unresolved operator fails before any work is queued.
//! Nothing here creates a context, stream or allocation: the caller owns the Driver / Context / Stream and the memory
//! (docs/dev/TENSORFOLD-RUNTIME.md contract) and passes views into it. The module image is the `audio` fatbin, built
//! with `-O3 --fmad=false` and no fast math (audio.cu header); without nvcc the caller has no image to load.
//!
//! Every kernel is gpu-qualification-pending: `Use.production` is refused until the SM121 operator, chain, mel and PCM
//! gates against the native twin pass with zero unequal values. Activation normalization takes its reduction order and
//! denominator rounding as explicit arguments, which the GPU harness sweeps against the native twin; the middle
//! attention's `k6a_sdpa_math` is a diagnostic comparison candidate only. Decoder weight preparation runs on the host at
//! load (vae.zig `prepare`), and the vocoder weight-norm fold is not on the native path.
//!
//! `decode` and `vocode` run the whole graphs in the native operator order on weights bound by `vae.bindDecoder` /
//! `vocoder.bindVocoder`, carving every working tensor from one caller-owned scratch allocation of `scratchBytes()`.
//! Their numeric choices are explicit policy arguments; they are diagnostic comparison runs until the target gates pass.

const std = @import("std");
const cuda = @import("cuda");
const vae = @import("vae.zig");
const vocoder = @import("vocoder.zig");

const View = vae.View;
const Weight = vae.Weight;
const Use = vae.Use;
const Acc = vae.Acc;
const ScalarDiv = vae.ScalarDiv;
const Div = vae.Div;
const Error = vae.Error;

/// The device sum-of-squares order (audio.cu K6A_RED_*): one serial chain fused or in two roundings, or `strided_tree`
/// lanes (a power of two up to 1024) each summing a strided slice, then a halving tree. The order SM121's native
/// reduction uses is what the GPU harness determines; nothing here picks one.
pub const DeviceReduce = union(enum) { serial_fma, serial_mul_add, strided_tree: u32 };

const block: u32 = 256;
const conv_block: u32 = 128;
const conv_ot: u32 = 4; // audio.cu K6A_CONV_OT

fn blocks(n: u64) u32 {
    return @intCast((n + block - 1) / block);
}

fn i64of(v: anytype) i64 {
    return @intCast(v);
}

fn i32of(v: anytype) i32 {
    return @intCast(v);
}

pub const Ops = struct {
    module: cuda.Module,
    latent_unscale_fn: cuda.Function,
    mel_affine_fn: cuda.Function,
    clamp_fn: cuda.Function,
    upsample_fn: cuda.Function,
    silu_fn: cuda.Function,
    mp_residual_fn: cuda.Function,
    scale_weights_fn: cuda.Function,
    add_fn: cuda.Function,
    avg3_fn: cuda.Function,
    conv1d_fn: cuda.Function,
    conv_t_fn: cuda.Function,
    snake_fn: cuda.Function,
    up2_fn: cuda.Function,
    down2_fn: cuda.Function,
    qkv_split_fn: cuda.Function,
    attn_merge_fn: cuda.Function,
    l2norm_fn: cuda.Function,
    normalize_apply_fn: cuda.Function,
    sdpa_math_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var o: Ops = undefined;
        o.module = m;
        inline for (.{
            .{ "latent_unscale_fn", "k6a_latent_unscale" }, .{ "mel_affine_fn", "k6a_mel_affine" },
            .{ "clamp_fn", "k6a_clamp" },                   .{ "upsample_fn", "k6a_upsample_nearest_exact2" },
            .{ "silu_fn", "k6a_scaled_silu" },              .{ "mp_residual_fn", "k6a_mp_residual" },
            .{ "scale_weights_fn", "k6a_scale_weights" },   .{ "add_fn", "k6a_add" },
            .{ "avg3_fn", "k6a_avg3" },                     .{ "conv1d_fn", "k6a_conv1d" },
            .{ "conv_t_fn", "k6a_conv_transpose1d" },       .{ "snake_fn", "k6a_snake_beta_log" },
            .{ "up2_fn", "k6a_aa_up2" },                    .{ "down2_fn", "k6a_aa_down2" },
            .{ "qkv_split_fn", "k6a_qkv_split" },           .{ "attn_merge_fn", "k6a_attn_merge" },
            .{ "l2norm_fn", "k6a_l2norm" },                 .{ "normalize_apply_fn", "k6a_normalize_apply" },
            .{ "sdpa_math_fn", "k6a_sdpa_math" },
        }) |e| @field(o, e[0]) = try m.function(e[1]);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, blk: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = blk }, .shared = 0 }, s, args);
    }

    fn flat(f: cuda.Function, s: cuda.Stream, n: u64, args: *cuda.Args) !void {
        try go(f, s, .{ .x = blocks(n) }, block, args);
    }

    /// rows [B, T, 40] -> z [B, 40, T] = rows / scaling_factor + mean_value (vae.Decoder.latentRowsShape / inputShape).
    pub fn latentUnscale(o: *const Ops, s: cuda.Stream, use: Use, rows: View, z: View, scale: ScalarDiv, mean: f32) !void {
        try vae.admitOp(.latent_unscale, use);
        const b = rows.shape[0];
        const t = rows.shape[1];
        const c = rows.shape[2];
        try rows.expect(.cuda, &.{ b, t, c });
        try z.expect(.cuda, &.{ b, c, t });
        var a: cuda.Args = .{};
        a.add(rows.addr());
        a.add(z.addr());
        inline for (.{ b, c, t }) |v| a.add(i64of(v));
        a.add(scale.c);
        a.add(@backingInt(scale.div));
        a.add(mean);
        try flat(o.latent_unscale_fn, s, b * c * t, &a);
    }

    /// VAE.unnormalize on [B, 128, T] with the loaded data_std / data_mean [128]; y may alias x.
    pub fn melAffine(o: *const Ops, s: cuda.Stream, use: Use, x: View, stdv: View, mean: View, y: View) !void {
        try vae.admitOp(.mel_affine, use);
        const b = x.shape[0];
        const c = x.shape[1];
        const t = x.shape[2];
        try x.expect(.cuda, &.{ b, c, t });
        try y.expect(.cuda, &.{ b, c, t });
        try stdv.expect(.cuda, &.{c});
        try mean.expect(.cuda, &.{c});
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), stdv.addr(), mean.addr(), y.addr() }) |p| a.add(p);
        inline for (.{ b, c, t }) |v| a.add(i64of(v));
        try flat(o.mel_affine_fn, s, b * c * t, &a);
    }

    pub fn clamp(o: *const Ops, s: cuda.Stream, use: Use, x: View, y: View, lo: f32, hi: f32) !void {
        try vae.admitOp(.clamp, use);
        try x.expect(.cuda, x.shape[0..x.rank]);
        try y.expect(.cuda, x.shape[0..x.rank]);
        var a: cuda.Args = .{};
        a.add(x.addr());
        a.add(y.addr());
        a.add(i64of(x.numel()));
        a.add(lo);
        a.add(hi);
        try flat(o.clamp_fn, s, x.numel(), &a);
    }

    /// [B, C, T] -> [B, C, 2T], nearest-exact x2.
    pub fn upsampleNearestExact2(o: *const Ops, s: cuda.Stream, use: Use, x: View, y: View) !void {
        try vae.admitOp(.upsample_nearest_exact2, use);
        const b = x.shape[0];
        const c = x.shape[1];
        const t = x.shape[2];
        try x.expect(.cuda, &.{ b, c, t });
        try y.expect(.cuda, &.{ b, c, 2 * t });
        var a: cuda.Args = .{};
        a.add(x.addr());
        a.add(y.addr());
        a.add(i64of(b * c));
        a.add(i64of(t));
        try flat(o.upsample_fn, s, b * c * 2 * t, &a);
    }

    /// silu(x) / divisor (vae.silu_divisor) with the named scalar division; y may alias x.
    pub fn scaledSilu(o: *const Ops, s: cuda.Stream, use: Use, x: View, y: View, c: ScalarDiv) !void {
        try vae.admitOp(.scaled_silu, use);
        try x.expect(.cuda, x.shape[0..x.rank]);
        try y.expect(.cuda, x.shape[0..x.rank]);
        var a: cuda.Args = .{};
        a.add(x.addr());
        a.add(y.addr());
        a.add(i64of(x.numel()));
        a.add(c.c);
        a.add(@backingInt(c.div));
        try flat(o.silu_fn, s, x.numel(), &a);
    }

    /// mp_sum(a = shortcut, b = branch, t) with `vae.MpResidual` constants; y may alias a or b.
    pub fn mpResidual(o: *const Ops, s: cuda.Stream, use: Use, x: View, h: View, y: View, m: vae.MpResidual) !void {
        try vae.admitOp(.mp_residual, use);
        try x.expect(.cuda, x.shape[0..x.rank]);
        try h.expect(.cuda, x.shape[0..x.rank]);
        try y.expect(.cuda, x.shape[0..x.rank]);
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), h.addr(), y.addr() }) |p| a.add(p);
        a.add(i64of(x.numel()));
        a.add(m.c0);
        a.add(m.c1);
        a.add(m.den.c);
        a.add(@backingInt(m.den.div));
        try flat(o.mp_residual_fn, s, x.numel(), &a);
    }

    /// Gain before convolution: `out` = prepared weight x gain, into separate immutable storage of the same shape. The
    /// result carries the source weight's manifest and stays prepared (it is a scaled copy, not a second preparation).
    pub fn gainScaledWeight(o: *const Ops, s: cuda.Stream, use: Use, w: Weight, out: View, gain: f32) !Weight {
        try vae.admitOp(.gain_weight_scale, use);
        try w.requirePrepared(use);
        try w.view.expect(.cuda, w.view.shape[0..w.view.rank]);
        try out.expect(.cuda, w.view.shape[0..w.view.rank]);
        var a: cuda.Args = .{};
        a.add(w.view.addr());
        a.add(out.addr());
        a.add(i64of(w.view.numel()));
        a.add(gain);
        try flat(o.scale_weights_fn, s, w.view.numel(), &a);
        var scaled = w;
        scaled.view = out;
        return scaled;
    }

    pub const ConvGeom = struct { stride: u32 = 1, pad: u32, dil: u32 = 1 };

    /// ConvPrepared1D: x [B, Cin, Lin], prepared w [Cout, Cin, K], optional bias [Cout], y [B, Cout, Lout]. The weight
    /// must be prepared exactly once (stored checkpoint tensors are refused; a gain is applied with gainScaledWeight).
    pub fn conv1d(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, bias: ?View, y: View, g: ConvGeom, acc: Acc) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        const b = x.shape[0];
        const cin = x.shape[1];
        const lin = x.shape[2];
        const cout = w.view.shape[0];
        const k = w.view.shape[2];
        try x.expect(.cuda, &.{ b, cin, lin });
        try w.view.expect(.cuda, &.{ cout, cin, k });
        if (bias) |bv| try bv.expect(.cuda, &.{cout});
        if (lin + 2 * g.pad < g.dil * (k - 1) + 1 or g.stride == 0) return error.InvalidArgument;
        const lout = vae.convOutLen(.{ .b = b, .cin = cin, .cout = cout, .lin = lin, .k = k, .stride = g.stride, .pad = g.pad, .dil = g.dil });
        try y.expect(.cuda, &.{ b, cout, lout });
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), w.view.addr(), if (bias) |bv| bv.addr() else 0, y.addr() }) |p| a.add(@as(u64, p));
        inline for (.{ cin, cout, lin, lout }) |v| a.add(i64of(v));
        inline for (.{ k, g.stride, g.pad, g.dil }) |v| a.add(i32of(v));
        a.add(@backingInt(acc));
        try go(o.conv1d_fn, s, .{ .x = @intCast((lout + conv_block - 1) / conv_block), .y = @intCast((cout + conv_ot - 1) / conv_ot), .z = @intCast(b) }, conv_block, &a);
    }

    /// Prepared ConvTranspose1d: x [B, Cin, Lin], w [Cin, Cout, K], optional bias [Cout], y [B, Cout, Lout].
    pub fn convTranspose1d(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, bias: ?View, y: View, stride: u32, pad: u32, acc: Acc) !void {
        try vae.admitOp(.conv_transpose1d_prepared, use);
        try w.requirePrepared(use);
        const b = x.shape[0];
        const cin = x.shape[1];
        const lin = x.shape[2];
        const cout = w.view.shape[1];
        const k = w.view.shape[2];
        try x.expect(.cuda, &.{ b, cin, lin });
        try w.view.expect(.cuda, &.{ cin, cout, k });
        if (bias) |bv| try bv.expect(.cuda, &.{cout});
        if (stride == 0 or (lin - 1) * stride + k <= 2 * pad) return error.InvalidArgument;
        const lout = vae.convTransposeOutLen(lin, k, stride, pad);
        try y.expect(.cuda, &.{ b, cout, lout });
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), w.view.addr(), if (bias) |bv| bv.addr() else 0, y.addr() }) |p| a.add(@as(u64, p));
        inline for (.{ cin, cout, lin, lout }) |v| a.add(i64of(v));
        inline for (.{ k, stride, pad }) |v| a.add(i32of(v));
        a.add(@backingInt(acc));
        try go(o.conv_t_fn, s, .{ .x = @intCast((lout + conv_block - 1) / conv_block), .y = @intCast((cout + conv_ot - 1) / conv_ot), .z = @intCast(b) }, conv_block, &a);
    }

    /// SnakeBeta with log-scale alpha / beta [C] on [B, C, T]; y may alias x.
    pub fn snakeBetaLog(o: *const Ops, s: cuda.Stream, use: Use, x: View, alpha_log: View, beta_log: View, y: View) !void {
        try vae.admitOp(.snake_beta_log, use);
        const b = x.shape[0];
        const c = x.shape[1];
        const t = x.shape[2];
        try x.expect(.cuda, &.{ b, c, t });
        try y.expect(.cuda, &.{ b, c, t });
        try alpha_log.expect(.cuda, &.{c});
        try beta_log.expect(.cuda, &.{c});
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), alpha_log.addr(), beta_log.addr(), y.addr() }) |p| a.add(p);
        inline for (.{ b, c, t }) |v| a.add(i64of(v));
        a.add(vocoder.snake_eps);
        try flat(o.snake_fn, s, b * c * t, &a);
    }

    /// Activation1d up-sampler: [B, C, T] -> [B, C, 2T] with the loaded 12-tap `filter` buffer.
    pub fn aaUp2(o: *const Ops, s: cuda.Stream, use: Use, x: View, filter: View, y: View, acc: Acc) !void {
        try vae.admitOp(.aa_up2, use);
        const b = x.shape[0];
        const c = x.shape[1];
        const t = x.shape[2];
        try x.expect(.cuda, &.{ b, c, t });
        try filter.expect(.cuda, &.{vocoder.filter_taps});
        try y.expect(.cuda, &.{ b, c, 2 * t });
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), filter.addr(), y.addr() }) |p| a.add(p);
        a.add(i64of(b * c));
        a.add(i64of(t));
        a.add(@backingInt(acc));
        try flat(o.up2_fn, s, b * c * 2 * t, &a);
    }

    /// Activation1d down-sampler: [B, C, T2] -> [B, C, (T2 - 1) / 2 + 1] with the loaded 12-tap `filter` buffer.
    pub fn aaDown2(o: *const Ops, s: cuda.Stream, use: Use, x: View, filter: View, y: View, acc: Acc) !void {
        try vae.admitOp(.aa_down2, use);
        const b = x.shape[0];
        const c = x.shape[1];
        const t2 = x.shape[2];
        const to = vocoder.aaDown2Len(t2);
        try x.expect(.cuda, &.{ b, c, t2 });
        try filter.expect(.cuda, &.{vocoder.filter_taps});
        try y.expect(.cuda, &.{ b, c, to });
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), filter.addr(), y.addr() }) |p| a.add(p);
        inline for (.{ b * c, t2, to }) |v| a.add(i64of(v));
        a.add(@backingInt(acc));
        try flat(o.down2_fn, s, b * c * to, &a);
    }

    /// AMPBlock1 `x = xt + x`; y may alias either.
    pub fn add(o: *const Ops, s: cuda.Stream, use: Use, xt: View, x: View, y: View) !void {
        try vae.admitOp(.vocoder_add, use);
        try xt.expect(.cuda, x.shape[0..x.rank]);
        try x.expect(.cuda, x.shape[0..x.rank]);
        try y.expect(.cuda, x.shape[0..x.rank]);
        var a: cuda.Args = .{};
        inline for (.{ xt.addr(), x.addr(), y.addr() }) |p| a.add(p);
        a.add(i64of(x.numel()));
        try flat(o.add_fn, s, x.numel(), &a);
    }

    /// The stage merge ((r0 + r1) + r2) / 3 with the named scalar division.
    pub fn avg3(o: *const Ops, s: cuda.Stream, use: Use, r: [3]View, y: View, c: ScalarDiv) !void {
        try vae.admitOp(.vocoder_avg3, use);
        for (r) |v| try v.expect(.cuda, y.shape[0..y.rank]);
        try y.expect(.cuda, y.shape[0..y.rank]);
        var a: cuda.Args = .{};
        inline for (.{ r[0].addr(), r[1].addr(), r[2].addr(), y.addr() }) |p| a.add(p);
        a.add(i64of(y.numel()));
        a.add(c.c);
        a.add(@backingInt(c.div));
        try flat(o.avg3_fn, s, y.numel(), &a);
    }

    /// The (normalized) packed QKV [B, H, D, 3, T] -> q, k, v [B, H, T, D].
    pub fn qkvSplit(o: *const Ops, s: cuda.Stream, use: Use, qkv: View, q: View, k: View, v: View) !void {
        try vae.admitOp(.qkv_split, use);
        const b = qkv.shape[0];
        const h = qkv.shape[1];
        const d = qkv.shape[2];
        const t = qkv.shape[4];
        try qkv.expect(.cuda, &.{ b, h, d, 3, t });
        for ([_]View{ q, k, v }) |x| try x.expect(.cuda, &.{ b, h, t, d });
        var a: cuda.Args = .{};
        inline for (.{ qkv.addr(), q.addr(), k.addr(), v.addr() }) |p| a.add(p);
        inline for (.{ b, h, d, t }) |x| a.add(i64of(x));
        try flat(o.qkv_split_fn, s, b * h * t * d, &a);
    }

    /// Attention output [B, H, T, D] -> [B, H * D, T].
    pub fn attnMerge(o: *const Ops, s: cuda.Stream, use: Use, h_in: View, y: View) !void {
        try vae.admitOp(.attn_merge, use);
        const b = h_in.shape[0];
        const h = h_in.shape[1];
        const t = h_in.shape[2];
        const d = h_in.shape[3];
        try h_in.expect(.cuda, &.{ b, h, t, d });
        try y.expect(.cuda, &.{ b, h * d, t });
        var a: cuda.Args = .{};
        a.add(h_in.addr());
        a.add(y.addr());
        inline for (.{ b, h, d, t }) |x| a.add(i64of(x));
        try flat(o.attn_merge_fn, s, b * h * d * t, &a);
    }

    // ------------------------------------------------- operators whose target arithmetic is still being compared

    /// Activation normalization (residual [B, C, T] axis 1, packed QKV [B, H, D, 3, T] axis 2), in place or not:
    /// norms [outer, inner] = ||x||_axis under `reduce`, then y = x / (eps + norm * f32(sqrt(1/n))) with the denominator
    /// rounded per `den`. eps, reduce and den have no defaults. Diagnostic only (vae `activation_normalize`).
    pub fn normalizeActivation(o: *const Ops, s: cuda.Stream, use: Use, x: View, norms: View, y: View, axis: u8, eps: f32, reduce: DeviceReduce, den: Acc) !void {
        try vae.admitOp(.activation_normalize, use);
        const g = try vae.normalizeActivation(x, y, axis);
        try x.expect(.cuda, x.shape[0..x.rank]);
        try y.expect(.cuda, x.shape[0..x.rank]);
        try norms.expect(.cuda, &.{ g.outer, g.inner });
        const lanes: u32 = switch (reduce) {
            .serial_fma, .serial_mul_add => 1,
            .strided_tree => |r| blk: {
                if (r == 0 or r > 1024 or !std.math.isPowerOfTwo(r)) return error.InvalidArgument;
                break :blk r;
            },
        };
        var a: cuda.Args = .{};
        a.add(x.addr());
        a.add(norms.addr());
        inline for (.{ g.outer, g.n, g.inner }) |v| a.add(i64of(v));
        a.add(@as(i32, switch (reduce) {
            .serial_fma => 0,
            .serial_mul_add => 1,
            .strided_tree => 2,
        }));
        a.add(i32of(lanes));
        const threads = @max(32, (lanes + 31) / 32 * 32);
        try cuda.launch.launch(o.l2norm_fn, .{ .grid = .{ .x = @intCast(g.outer * g.inner) }, .block = .{ .x = threads }, .shared = lanes * @sizeOf(f32) }, s, &a);
        var b: cuda.Args = .{};
        inline for (.{ x.addr(), norms.addr(), y.addr() }) |p| b.add(p);
        inline for (.{ g.outer, g.n, g.inner }) |v| b.add(i64of(v));
        b.add(g.scale);
        b.add(eps);
        b.add(@as(i32, @intFromBool(den == .fma)));
        try flat(o.normalize_apply_fn, s, x.numel(), &b);
    }

    /// The middle attention, math formulation (audio.cu k6a_sdpa_math), q, k, v, out [BH, T, D] contiguous, softmax
    /// scale 1/sqrt(D) applied as sqrt of it on each operand; `scratch` holds BH * T * T floats. A comparison candidate
    /// for the native SDPA, whose backend choice is not yet recorded: diagnostic only, never selected automatically.
    pub fn midAttentionMath(o: *const Ops, s: cuda.Stream, use: Use, q: View, k: View, v: View, out: View, scratch: View) !void {
        try vae.admitOp(.mid_attention_sdpa, use);
        const bh = q.shape[0];
        const t = q.shape[1];
        const d = q.shape[2];
        for ([_]View{ q, k, v, out }) |x| try x.expect(.cuda, &.{ bh, t, d });
        try scratch.expect(.cuda, &.{ bh, t, t });
        const c: f32 = @floatCast(@sqrt(1.0 / @sqrt(@as(f64, @floatFromInt(d)))));
        var a: cuda.Args = .{};
        inline for (.{ q.addr(), k.addr(), v.addr(), out.addr(), scratch.addr() }) |p| a.add(p);
        a.add(i64of(t));
        a.add(i64of(d));
        a.add(c);
        try go(o.sdpa_math_fn, s, .{ .x = @intCast(t), .y = @intCast(bh) }, block, &a);
    }
};

// ------------------------------------------------------------------------------------------------ orchestration

/// Views into one contiguous `.cuda` scratch allocation, at float offsets from its first element.
const Carver = struct {
    scratch: View,

    fn init(scratch: View, floats: u64) Error!Carver {
        try scratch.validate();
        if (scratch.device != .cuda) return error.DeviceMismatch;
        if (scratch.rank != 1 or !scratch.isContiguous()) return error.LayoutUnsupported;
        if (scratch.shape[0] < floats) return error.OutOfBounds;
        return .{ .scratch = scratch };
    }

    fn at(c: Carver, name: []const u8, offset: u64, shape: []const u64) View {
        var v = View.contiguous(name, .cuda, c.scratch.base, c.scratch.alloc_bytes, .borrowed, shape);
        v.offset = c.scratch.offset + offset;
        return v;
    }
};

/// One activation normalization's arithmetic: eps, the device reduction order and the denominator rounding.
pub const NormPolicy = struct { eps: f32, reduce: DeviceReduce, den: Acc };

/// Every numeric choice of a decoder forward, named; none has a default. The values the native twin uses on the target
/// are what the GPU harness determines; until then a decode is a diagnostic comparison run.
pub const DecodePolicy = struct {
    /// Convolution accumulation (every decoder conv).
    acc: Acc,
    /// latent_unscale's division by scaling_factor.
    unscale: Div,
    /// scaled SiLU's division by 0.596.
    silu: Div,
    /// mp_sum's division by sqrt((1 - t)^2 + t^2).
    residual: Div,
    /// The residual blocks' normalization (axis 1) and the attention's packed-QKV normalization (axis 2).
    norm_residual: NormPolicy,
    norm_qkv: NormPolicy,
    /// The middle attention. Only the `k6a_sdpa_math` comparison candidate exists, and its contract refuses production;
    /// strict FLASH is not accepted for head dimension 2048.
    attention: enum { math_candidate },
};

/// Decoder1D + VAE.unnormalize: `rows` the sampler's latent rows [B, T, 40] -> `mel` [B, 128, 2T], with the weights from
/// `vae.bindDecoder` and `scratch` a rank-1 `.cuda` view of at least `d.scratchBytes()`. Every operator is admitted
/// before anything is queued; production is refused until the target gates pass.
pub fn decode(o: *const Ops, s: cuda.Stream, use: Use, d: vae.Decoder, w: *const vae.DecoderWeights, rows: View, scratch: View, mel: View, p: DecodePolicy) !void {
    try decodeWith(o, s, use, d, w, rows, scratch, mel, p);
}

/// `decode` over any backend with the Ops launch methods (the tests' recorder checks the sequence and the dataflow).
fn decodeWith(o: anytype, s: anytype, use: Use, d: vae.Decoder, w: *const vae.DecoderWeights, rows: View, scratch: View, mel: View, p: DecodePolicy) !void {
    try d.admit(use, .prepared_native_reference);
    const sc = d.scratch();
    const cv = try Carver.init(scratch, sc.floats);
    try rows.expect(.cuda, &d.latentRowsShape());
    try mel.expect(.cuda, &d.melShape());
    const b: u64 = d.batch;
    var t: u64 = d.frames;
    const silu = ScalarDiv.of(vae.silu_divisor, p.silu);
    const mix = vae.MpResidual.of(vae.residual_t, p.residual);
    const z = cv.at("k6a.decoder.z", sc.aux, &.{ b, vae.latent_channels, t });
    try o.latentUnscale(s, use, rows, z, .of(vae.scaling_factor, p.unscale), vae.mean_value);
    var wi: usize = 0;
    for (vae.decoder_graph) |node| switch (node) {
        .conv => |c| {
            const first = wi == 0;
            const x = if (first) z else cv.at("k6a.decoder.act0", sc.act[0], &.{ b, c.cin, t });
            const y = cv.at(if (first) "k6a.decoder.act0" else "k6a.decoder.act1", sc.act[if (first) 0 else 1], &.{ b, c.cout, t });
            try o.conv1d(s, use, x, w.convs[wi], null, y, .{ .pad = c.k / 2 }, p.acc);
            wi += 1;
        },
        .resblock => |r| {
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, r.cin, t });
            const xn = cv.at("k6a.decoder.act1", sc.act[1], &.{ b, r.cin, t });
            try o.normalizeActivation(s, use, x, cv.at("k6a.decoder.norms", sc.aux, &.{ b, t }), xn, 1, p.norm_residual.eps, p.norm_residual.reduce, p.norm_residual.den);
            const h_in = cv.at("k6a.decoder.act2", sc.act[2], &.{ b, r.cin, t });
            try o.scaledSilu(s, use, xn, h_in, silu);
            const h1 = cv.at("k6a.decoder.act3", sc.act[3], &.{ b, r.cout, t });
            try o.conv1d(s, use, h_in, w.convs[wi], null, h1, .{ .pad = 1 }, p.acc);
            try o.scaledSilu(s, use, h1, h1, silu);
            const h2 = cv.at("k6a.decoder.act2", sc.act[2], &.{ b, r.cout, t });
            try o.conv1d(s, use, h1, w.convs[wi + 1], null, h2, .{ .pad = 1 }, p.acc);
            wi += 2;
            var shortcut = xn;
            if (r.cin != r.cout) {
                shortcut = cv.at("k6a.decoder.act4", sc.act[4], &.{ b, r.cout, t });
                try o.conv1d(s, use, xn, w.convs[wi], null, shortcut, .{ .pad = 0 }, p.acc);
                wi += 1;
            }
            try o.mpResidual(s, use, shortcut, h2, cv.at("k6a.decoder.act0", sc.act[0], &.{ b, r.cout, t }), mix);
        },
        .attention => |a| {
            // the scratch regions are sized for the native graph's one attention (C 2048, at the latent length)
            if (a.c != 2048 or t != d.frames or a.c % a.heads != 0) return error.InvalidArgument;
            const hd = a.c / a.heads;
            const bh = b * a.heads;
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, a.c, t });
            try o.conv1d(s, use, x, w.convs[wi], null, cv.at("k6a.decoder.qkv", sc.qkv, &.{ b, 3 * a.c, t }), .{ .pad = 0 }, p.acc);
            const packed_qkv = cv.at("k6a.decoder.qkv", sc.qkv, &.{ b, a.heads, hd, 3, t });
            try o.normalizeActivation(s, use, packed_qkv, cv.at("k6a.decoder.norms", sc.aux, &.{ bh, 3 * t }), packed_qkv, 2, p.norm_qkv.eps, p.norm_qkv.reduce, p.norm_qkv.den);
            const q = cv.at("k6a.decoder.q", sc.q, &.{ b, a.heads, t, hd });
            const k = cv.at("k6a.decoder.k", sc.k, &.{ b, a.heads, t, hd });
            const v = cv.at("k6a.decoder.v", sc.v, &.{ b, a.heads, t, hd });
            try o.qkvSplit(s, use, packed_qkv, q, k, v);
            switch (p.attention) {
                .math_candidate => try o.midAttentionMath(
                    s,
                    use,
                    cv.at("k6a.decoder.q", sc.q, &.{ bh, t, hd }),
                    cv.at("k6a.decoder.k", sc.k, &.{ bh, t, hd }),
                    cv.at("k6a.decoder.v", sc.v, &.{ bh, t, hd }),
                    cv.at("k6a.decoder.attn_out", sc.out, &.{ bh, t, hd }),
                    cv.at("k6a.decoder.scores", sc.scores, &.{ bh, t, t }),
                ),
            }
            const merged = cv.at("k6a.decoder.act1", sc.act[1], &.{ b, a.c, t });
            try o.attnMerge(s, use, cv.at("k6a.decoder.attn_out", sc.out, &.{ b, a.heads, t, hd }), merged);
            const h = cv.at("k6a.decoder.act2", sc.act[2], &.{ b, a.c, t });
            try o.conv1d(s, use, merged, w.convs[wi + 1], null, h, .{ .pad = 0 }, p.acc);
            wi += 2;
            try o.mpResidual(s, use, x, h, x, mix);
        },
        .clamp => {
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, try decoderChannels(wi), t });
            try o.clamp(s, use, x, x, -vae.clip_act, vae.clip_act);
        },
        .upsample => |c| {
            const up = cv.at("k6a.decoder.act1", sc.act[1], &.{ b, c.cin, 2 * t });
            try o.upsampleNearestExact2(s, use, cv.at("k6a.decoder.act0", sc.act[0], &.{ b, c.cin, t }), up);
            t *= 2;
            try o.conv1d(s, use, up, w.convs[wi], null, cv.at("k6a.decoder.act0", sc.act[0], &.{ b, c.cout, t }), .{ .pad = c.k / 2 }, p.acc);
            wi += 1;
        },
        .silu => {
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, try decoderChannels(wi), t });
            try o.scaledSilu(s, use, x, x, silu);
        },
        .unnormalize => try o.melAffine(s, use, cv.at("k6a.decoder.act1", sc.act[1], &.{ b, vae.mel_channels, t }), w.data_std, w.data_mean, mel),
    };
    if (wi != w.convs.len) return error.InvalidArgument;
}

/// The channel count of the decoder activation after the first `wi` convolutions: the last one's Cout.
fn decoderChannels(wi: usize) Error!u64 {
    if (wi == 0) return error.InvalidArgument;
    return vae.decoder_convs[wi - 1].cout;
}

/// The vocoder's numeric choices, named; none has a default (see DecodePolicy).
pub const VocodePolicy = struct {
    /// conv_pre, the upsamplers, the AMP-block convs and conv_post.
    conv_acc: Acc,
    /// Activation1d's anti-aliasing FIRs.
    aa_acc: Acc,
    /// The stage merge's division by num_kernels (3).
    avg: Div,
};

/// BigVGAN v2: `mel` [B, 128, F] -> `wave` [B, 1, 512F] clamped to [-1, 1], with the weights from
/// `vocoder.bindVocoder` and `scratch` a rank-1 `.cuda` view of at least `v.scratchBytes()`.
pub fn vocode(o: *const Ops, s: cuda.Stream, use: Use, v: vocoder.Vocoder, w: *const vocoder.VocoderWeights, mel: View, scratch: View, wave: View, p: VocodePolicy) !void {
    try vocodeWith(o, s, use, v, w, mel, scratch, wave, p);
}

fn vocodeWith(o: anytype, s: anytype, use: Use, v: vocoder.Vocoder, w: *const vocoder.VocoderWeights, mel: View, scratch: View, wave: View, p: VocodePolicy) !void {
    try v.admit(use, .checkpoint_effective);
    const sc = v.scratch();
    const cv = try Carver.init(scratch, sc.floats);
    try mel.expect(.cuda, &v.melShape());
    try wave.expect(.cuda, &v.waveShape());
    const b: u64 = v.batch;
    var t: u64 = v.frames;
    var c: u64 = vocoder.upsample_initial_channel;
    const work = [2][]const u8{ "k6a.vocoder.work0", "k6a.vocoder.work1" };
    try o.conv1d(s, use, mel, w.conv_pre.w, w.conv_pre.bias, cv.at(work[0], sc.work[0], &.{ b, c, t }), .{ .pad = 3 }, p.conv_acc);
    for (vocoder.upsample_rates, vocoder.upsample_kernel_sizes, 0..) |u, uk, i| {
        const ch: u64 = vocoder.stageChannels(i);
        const stage = cv.at("k6a.vocoder.stage", sc.stage, &.{ b, ch, t * u });
        try o.convTranspose1d(s, use, cv.at(work[0], sc.work[0], &.{ b, c, t }), w.ups[i].w, w.ups[i].bias, stage, u, (uk - u) / 2, p.conv_acc);
        t *= u;
        c = ch;
        var amp: [3]View = undefined;
        for (vocoder.resblock_kernel_sizes, 0..) |rk, j| {
            const blk = &w.blocks[3 * i + j];
            amp[j] = cv.at("k6a.vocoder.amp", sc.amp[j], &.{ b, c, t });
            for (vocoder.resblock_dilations, 0..) |dil, n| {
                const x = if (n == 0) stage else amp[j];
                const w0 = cv.at(work[0], sc.work[0], &.{ b, c, t });
                const w1 = cv.at(work[1], sc.work[1], &.{ b, c, t });
                try activation1d(o, s, use, cv, sc, x, blk.acts[2 * n], w0, p.aa_acc);
                try o.conv1d(s, use, w0, blk.convs1[n].w, blk.convs1[n].bias, w1, .{ .pad = vocoder.samePadding(rk, dil), .dil = dil }, p.conv_acc);
                try activation1d(o, s, use, cv, sc, w1, blk.acts[2 * n + 1], w0, p.aa_acc);
                try o.conv1d(s, use, w0, blk.convs2[n].w, blk.convs2[n].bias, w1, .{ .pad = vocoder.samePadding(rk, 1) }, p.conv_acc);
                try o.add(s, use, w1, x, amp[j]);
            }
        }
        try o.avg3(s, use, amp, cv.at(work[0], sc.work[0], &.{ b, c, t }), .of(@floatFromInt(vocoder.resblock_kernel_sizes.len), p.avg));
    }
    const post = cv.at(work[1], sc.work[1], &.{ b, c, t });
    try activation1d(o, s, use, cv, sc, cv.at(work[0], sc.work[0], &.{ b, c, t }), w.post, post, p.aa_acc);
    const y = cv.at(work[0], sc.work[0], &.{ b, 1, t });
    try o.conv1d(s, use, post, w.conv_post.w, null, y, .{ .pad = 3 }, p.conv_acc);
    try o.clamp(s, use, y, wave, -vocoder.final_clamp, vocoder.final_clamp);
}

/// Activation1d: up x2, SnakeBeta (log scale), down x2; `y` the same shape as `x`, in a different buffer.
fn activation1d(o: anytype, s: anytype, use: Use, cv: Carver, sc: vocoder.Vocoder.Scratch, x: View, a: vocoder.ActWeights, y: View, acc: Acc) !void {
    const up = cv.at("k6a.vocoder.up2", sc.up2, &.{ x.shape[0], x.shape[1], 2 * x.shape[2] });
    try o.aaUp2(s, use, x, a.up, up, acc);
    try o.snakeBetaLog(s, use, up, a.alpha, a.beta, up);
    try o.aaDown2(s, use, up, a.down, y, acc);
}

test {
    std.testing.refAllDecls(Ops);
    std.testing.refAllDecls(@This());
}

const testing = std.testing;

/// A backend for decodeWith / vocodeWith that launches nothing: it records the operator sequence and checks, per call,
/// that every view is in bounds, that each scratch input is exactly the tensor most recently written to its region (no
/// later write overlapped it, and nothing reads scratch that was never written), and the convolution shapes.
const Recorder = struct {
    const Write = struct { base: u64, lo: u64, hi: u64 };
    scratch_base: u64,
    ops: [1024]vae.Op = undefined,
    n_ops: usize = 0,
    writes: [2048]Write = undefined,
    n_writes: usize = 0,

    fn op(r: *Recorder, o: vae.Op, use: Use) !void {
        try vae.admitOp(o, use);
        r.ops[r.n_ops] = o;
        r.n_ops += 1;
    }

    fn read(r: *Recorder, v: View) !void {
        try v.validate();
        const lo = v.offset;
        const hi = lo + v.numel();
        var i = r.n_writes;
        while (i > 0) {
            i -= 1;
            const w = r.writes[i];
            if (w.base != v.base or w.hi <= lo or hi <= w.lo) continue;
            if (w.lo != lo or w.hi != hi) return error.ClobberedInput;
            return;
        }
        if (v.base == r.scratch_base) return error.UninitializedRead;
    }

    fn write(r: *Recorder, v: View) !void {
        try v.validate();
        r.writes[r.n_writes] = .{ .base = v.base, .lo = v.offset, .hi = v.offset + v.numel() };
        r.n_writes += 1;
    }

    fn unary(r: *Recorder, o: vae.Op, use: Use, x: View, y: View) !void {
        try r.op(o, use);
        try r.read(x);
        try r.write(y);
    }

    fn latentUnscale(r: *Recorder, _: void, use: Use, rows: View, z: View, _: ScalarDiv, _: f32) !void {
        try r.unary(.latent_unscale, use, rows, z);
    }
    fn melAffine(r: *Recorder, _: void, use: Use, x: View, stdv: View, mean: View, y: View) !void {
        try r.read(stdv);
        try r.read(mean);
        try r.unary(.mel_affine, use, x, y);
    }
    fn clamp(r: *Recorder, _: void, use: Use, x: View, y: View, _: f32, _: f32) !void {
        try r.unary(.clamp, use, x, y);
    }
    fn upsampleNearestExact2(r: *Recorder, _: void, use: Use, x: View, y: View) !void {
        try testing.expectEqual(2 * x.shape[2], y.shape[2]);
        try r.unary(.upsample_nearest_exact2, use, x, y);
    }
    fn scaledSilu(r: *Recorder, _: void, use: Use, x: View, y: View, _: ScalarDiv) !void {
        try r.unary(.scaled_silu, use, x, y);
    }
    fn mpResidual(r: *Recorder, _: void, use: Use, x: View, h: View, y: View, _: vae.MpResidual) !void {
        try r.read(h);
        try testing.expectEqualSlices(u64, x.shape[0..x.rank], h.shape[0..h.rank]);
        try r.unary(.mp_residual, use, x, y);
    }
    fn conv1d(r: *Recorder, _: void, use: Use, x: View, w: Weight, bias: ?View, y: View, g: Ops.ConvGeom, _: Acc) !void {
        try w.requirePrepared(use);
        if (bias) |b| try r.read(b);
        try r.read(w.view);
        try testing.expectEqual(w.view.shape[1], x.shape[1]);
        try testing.expectEqual(w.view.shape[0], y.shape[1]);
        try testing.expectEqual(vae.convOutLen(.{ .b = x.shape[0], .cin = x.shape[1], .cout = y.shape[1], .lin = x.shape[2], .k = w.view.shape[2], .stride = g.stride, .pad = g.pad, .dil = g.dil }), y.shape[2]);
        try r.unary(.conv1d_prepared, use, x, y);
    }
    fn convTranspose1d(r: *Recorder, _: void, use: Use, x: View, w: Weight, bias: ?View, y: View, stride: u32, pad: u32, _: Acc) !void {
        try w.requirePrepared(use);
        if (bias) |b| try r.read(b);
        try testing.expectEqual(w.view.shape[0], x.shape[1]);
        try testing.expectEqual(w.view.shape[1], y.shape[1]);
        try testing.expectEqual(vae.convTransposeOutLen(x.shape[2], w.view.shape[2], stride, pad), y.shape[2]);
        try r.unary(.conv_transpose1d_prepared, use, x, y);
    }
    fn snakeBetaLog(r: *Recorder, _: void, use: Use, x: View, a: View, b: View, y: View) !void {
        try r.read(a);
        try r.read(b);
        try r.unary(.snake_beta_log, use, x, y);
    }
    fn aaUp2(r: *Recorder, _: void, use: Use, x: View, f: View, y: View, _: Acc) !void {
        try r.read(f);
        try testing.expectEqual(2 * x.shape[2], y.shape[2]);
        try r.unary(.aa_up2, use, x, y);
    }
    fn aaDown2(r: *Recorder, _: void, use: Use, x: View, f: View, y: View, _: Acc) !void {
        try r.read(f);
        try testing.expectEqual(vocoder.aaDown2Len(x.shape[2]), y.shape[2]);
        try r.unary(.aa_down2, use, x, y);
    }
    fn add(r: *Recorder, _: void, use: Use, xt: View, x: View, y: View) !void {
        try r.read(xt);
        try r.unary(.vocoder_add, use, x, y);
    }
    fn avg3(r: *Recorder, _: void, use: Use, rs: [3]View, y: View, _: ScalarDiv) !void {
        try r.read(rs[0]);
        try r.read(rs[1]);
        try r.unary(.vocoder_avg3, use, rs[2], y);
    }
    fn normalizeActivation(r: *Recorder, _: void, use: Use, x: View, norms: View, y: View, axis: u8, _: f32, _: DeviceReduce, _: Acc) !void {
        const g = try vae.normalizeActivation(x, y, axis);
        try testing.expectEqualSlices(u64, &.{ g.outer, g.inner }, norms.shape[0..norms.rank]);
        try r.unary(.activation_normalize, use, x, y);
        try r.write(norms);
    }
    fn qkvSplit(r: *Recorder, _: void, use: Use, qkv: View, q: View, k: View, v: View) !void {
        try r.unary(.qkv_split, use, qkv, q);
        try r.write(k);
        try r.write(v);
    }
    fn midAttentionMath(r: *Recorder, _: void, use: Use, q: View, k: View, v: View, out: View, scores: View) !void {
        try r.read(k);
        try r.read(v);
        try r.write(scores);
        try r.unary(.mid_attention_sdpa, use, q, out);
    }
    fn attnMerge(r: *Recorder, _: void, use: Use, h: View, y: View) !void {
        try r.unary(.attn_merge, use, h, y);
    }
};

fn testView(name: []const u8, base: u64, shape: []const u64) View {
    var n: u64 = 1;
    for (shape) |d| n *= d;
    return .contiguous(name, .cuda, base, n * 4, .borrowed, shape);
}

const norm_policy: NormPolicy = .{ .eps = 1e-4, .reduce = .serial_fma, .den = .fma };

test "decode: the native operator order over the scratch, each input the tensor last written to its region" {
    const d = try vae.Decoder.init(2, 3);
    var w: vae.DecoderWeights = undefined;
    for (vae.decoder_convs, &w.convs, 0..) |c, *cw, i| {
        cw.* = .oraclePrepared(testView(c.name, (i + 1) << 40, &.{ c.cout, c.cin, c.k }), c.name, @splat(0));
    }
    w.gain = 1.0;
    w.data_std = testView("std", 100 << 40, &.{vae.mel_channels});
    w.data_mean = testView("mean", 101 << 40, &.{vae.mel_channels});
    var r: Recorder = .{ .scratch_base = 200 << 40 };
    const scratch = testView("scratch", r.scratch_base, &.{d.scratch().floats});
    const policy: DecodePolicy = .{ .acc = .fma, .unscale = .reciprocal, .silu = .true_div, .residual = .true_div, .norm_residual = norm_policy, .norm_qkv = norm_policy, .attention = .math_candidate };
    const mel = testView("mel", 202 << 40, &d.melShape());
    try decodeWith(&r, {}, .diagnostic, d, &w, testView("rows", 201 << 40, &d.latentRowsShape()), scratch, mel, policy);

    var want: [1024]vae.Op = undefined;
    var n: usize = 0;
    want[n] = .latent_unscale;
    n += 1;
    var buf: [8]vae.Op = undefined;
    for (vae.decoder_graph) |node| for (vae.nodeOps(node, &buf)) |o| {
        if (o == .gain_weight_scale) continue; // conv_out's gain is applied to the weight on the host at bind
        want[n] = o;
        n += 1;
    };
    try testing.expectEqualSlices(vae.Op, want[0..n], r.ops[0..r.n_ops]);
    const last = r.writes[r.n_writes - 1];
    try testing.expectEqual(mel.base, last.base);
    try testing.expectEqual(mel.numel(), last.hi - last.lo);
    // production is refused before anything is recorded
    var p: Recorder = .{ .scratch_base = r.scratch_base };
    try testing.expectError(error.GpuQualificationPending, decodeWith(&p, {}, .production, d, &w, testView("rows", 201 << 40, &d.latentRowsShape()), scratch, mel, policy));
    try testing.expectEqual(0, p.n_ops);
    // a scratch view smaller than scratchBytes is refused
    try testing.expectError(error.OutOfBounds, decodeWith(&p, {}, .diagnostic, d, &w, testView("rows", 201 << 40, &d.latentRowsShape()), testView("scratch", r.scratch_base, &.{d.scratch().floats - 1}), mel, policy));
}

test "vocode: BigVGAN's operator counts over the scratch, each input the tensor last written to its region" {
    const v = try vocoder.Vocoder.init(2, 2);
    var w: vocoder.VocoderWeights = undefined;
    var next: u64 = 1;
    const T = struct {
        fn conv(nx: *u64, shape: [3]u64, bias: ?u64) vocoder.ConvWeights {
            nx.* += 2;
            return .{ .w = .checkpointEffective(testView("w", nx.* << 36, &shape), "w", @splat(0)), .bias = if (bias) |b| testView("b", (nx.* + 1) << 36, &.{b}) else null };
        }
        fn act(nx: *u64, c: u64) vocoder.ActWeights {
            nx.* += 4;
            return .{ .alpha = testView("a", nx.* << 36, &.{c}), .beta = testView("b", (nx.* + 1) << 36, &.{c}), .up = testView("u", (nx.* + 2) << 36, &.{vocoder.filter_taps}), .down = testView("d", (nx.* + 3) << 36, &.{vocoder.filter_taps}) };
        }
    };
    w.conv_pre = T.conv(&next, .{ 1536, 128, 7 }, 1536);
    var cin: u64 = 1536;
    for (0..6) |i| {
        const ch: u64 = vocoder.stageChannels(i);
        w.ups[i] = T.conv(&next, .{ cin, ch, vocoder.upsample_kernel_sizes[i] }, ch);
        for (0..3) |j| {
            const b = &w.blocks[3 * i + j];
            for (0..3) |k| {
                b.convs1[k] = T.conv(&next, .{ ch, ch, vocoder.resblock_kernel_sizes[j] }, ch);
                b.convs2[k] = T.conv(&next, .{ ch, ch, vocoder.resblock_kernel_sizes[j] }, ch);
            }
            for (&b.acts) |*a| a.* = T.act(&next, ch);
        }
        cin = ch;
    }
    w.post = T.act(&next, cin);
    w.conv_post = T.conv(&next, .{ 1, cin, 7 }, null);

    var r: Recorder = .{ .scratch_base = 4000 << 36 };
    const wave = testView("wave", 4002 << 36, &v.waveShape());
    const policy: VocodePolicy = .{ .conv_acc = .fma, .aa_acc = .fma, .avg = .true_div };
    try vocodeWith(&r, {}, .diagnostic, v, &w, testView("mel", 4001 << 36, &v.melShape()), testView("scratch", r.scratch_base, &.{v.scratch().floats}), wave, policy);
    var counts = std.EnumArray(vae.Op, usize).initFill(0);
    for (r.ops[0..r.n_ops]) |o| counts.getPtr(o).* += 1;
    // 6 stages x 3 AMP blocks x 3 (a1, c1, a2, c2, add); 2 Activation1d per step plus activation_post
    try testing.expectEqual(1 + 6 * 9 * 2 + 1, counts.get(.conv1d_prepared));
    try testing.expectEqual(6, counts.get(.conv_transpose1d_prepared));
    inline for (.{ .aa_up2, .snake_beta_log, .aa_down2 }) |o| try testing.expectEqual(6 * 9 * 2 + 1, counts.get(o));
    try testing.expectEqual(6 * 9, counts.get(.vocoder_add));
    try testing.expectEqual(6, counts.get(.vocoder_avg3));
    try testing.expectEqual(vae.Op.clamp, r.ops[r.n_ops - 1]);
    const last = r.writes[r.n_writes - 1];
    try testing.expectEqual(wave.base, last.base);
    try testing.expectEqual(wave.numel(), last.hi - last.lo);
}
