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

const std = @import("std");
const cuda = @import("cuda");
const vae = @import("vae.zig");
const vocoder = @import("vocoder.zig");

const View = vae.View;
const Weight = vae.Weight;
const Use = vae.Use;
const Acc = vae.Acc;
const ScalarDiv = vae.ScalarDiv;
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

test {
    std.testing.refAllDecls(Ops);
}
