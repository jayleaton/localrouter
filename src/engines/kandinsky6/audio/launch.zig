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

/// The device sum-of-squares order (audio.cu K6A_RED_*): one serial chain fused or in two roundings, `strided_tree`
/// lanes (a power of two up to 1024) each summing a strided slice, then a halving tree, or `torch_reduce`: PyTorch's
/// own CUDA reduction order for the running device (k6a_l2norm_torch, split per shape by `vae.torchSplit` from the
/// device's queried limits). `torch_reduce` reproduces the D3 window's native normalization shas offline on every real
/// shape; the kernel itself is not yet run on SM121. Nothing here picks one.
pub const DeviceReduce = union(enum) { serial_fma, serial_mul_add, strided_tree: u32, torch_reduce: vae.SmLimits };

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
    conv_module: cuda.Module,
    first_res_conv_fn: cuda.Function,
    up1_first_res_conv_fn: cuda.Function,
    up1_second_res_conv_fn: cuda.Function,
    up1_shortcut_gemm_fn: cuda.Function,
    up1_shortcut_reduce_fn: cuda.Function,
    up0_shortcut_gemm_fn: cuda.Function,
    up0_shortcut_reduce_fn: cuda.Function,
    conv_tf32_16x10_fn: cuda.Function,
    conv_tf32_128x64_fn: cuda.Function,
    conv_tf32_32x5_fn: cuda.Function,
    gg_gather_a_fn: cuda.Function,
    gg_gather_b_fn: cuda.Function,
    gg_gemm_fn: cuda.Function,
    gg_accumulate_fn: cuda.Function,
    gg_scatter_fn: cuda.Function,
    add_bias_nct_fn: cuda.Function,
    attention_qkv_fn: cuda.Function,
    attention_proj_fn: cuda.Function,
    sdpa_prepare_fn: cuda.Function,
    sdpa_logits_fn: cuda.Function,
    sdpa_reduce_fn: cuda.Function,
    sdpa_softmax_fn: cuda.Function,
    sdpa_value_fn: cuda.Function,
    sdpa_output_fn: cuda.Function,

    tf32_rn_copy_fn: cuda.Function,
    nct_to_ntc_fn: cuda.Function,
    kcs_to_ksc_fn: cuda.Function,
    ntc_to_nct_fn: cuda.Function,
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
    l2norm_torch_fn: cuda.Function,
    normalize_apply_fn: cuda.Function,
    sdpa_math_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8, conv_image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var o: Ops = undefined;
        o.module = m;
        o.conv_module = try cuda.Module.load(d, conv_image);
        errdefer o.conv_module.unload();
        o.first_res_conv_fn = try o.conv_module.function("k6a_first_res_conv_tf32");
        try o.first_res_conv_fn.allowDynamicShared(81920);
        o.up1_first_res_conv_fn = try o.conv_module.function("k6a_up1_first_res_conv_tf32");
        try o.up1_first_res_conv_fn.allowDynamicShared(81920);
        o.up1_second_res_conv_fn = try o.conv_module.function("k6a_up1_second_res_conv_tf32");
        try o.up1_second_res_conv_fn.allowDynamicShared(81920);
        o.up1_shortcut_gemm_fn = try o.conv_module.function("k6a_up1_shortcut_gemm");
        try o.up1_shortcut_gemm_fn.allowDynamicShared(98304);
        o.up1_shortcut_reduce_fn = try o.conv_module.function("k6a_up1_shortcut_reduce");
        o.up0_shortcut_gemm_fn = try o.conv_module.function("k6a_up0_shortcut_gemm");
        try o.up0_shortcut_gemm_fn.allowDynamicShared(81920);
        o.up0_shortcut_reduce_fn = try o.conv_module.function("k6a_up0_shortcut_reduce");
        o.gg_gather_a_fn = try o.conv_module.function("k6a_gg_gather_a");
        o.gg_gather_b_fn = try o.conv_module.function("k6a_gg_gather_b");
        o.gg_gemm_fn = try o.conv_module.function("k6a_gg_gemm");
        try o.gg_gemm_fn.allowDynamicShared(gather_gemm_smem);
        o.gg_accumulate_fn = try o.conv_module.function("k6a_gg_accumulate");
        o.gg_scatter_fn = try o.conv_module.function("k6a_gg_scatter");
        o.conv_tf32_16x10_fn = try o.conv_module.function("k6a_conv_tf32_16x10");
        try o.conv_tf32_16x10_fn.allowDynamicShared(81920); // Tf32Tile.f16x10.smem()
        o.conv_tf32_128x64_fn = try o.conv_module.function("k6a_conv_tf32_128x64");
        try o.conv_tf32_128x64_fn.allowDynamicShared(73728); // Tf32Tile.f128x64.smem()
        o.conv_tf32_32x5_fn = try o.conv_module.function("k6a_conv_tf32_32x5");
        try o.conv_tf32_32x5_fn.allowDynamicShared(81920); // Tf32Tile.f32x5.smem()
        o.add_bias_nct_fn = try o.conv_module.function("k6a_add_bias_nct");
        o.attention_qkv_fn = try o.conv_module.function("k6a_attention_qkv_tf32");
        try o.attention_qkv_fn.allowDynamicShared(98304);
        o.attention_proj_fn = try o.conv_module.function("k6a_attention_proj_tf32");
        try o.attention_proj_fn.allowDynamicShared(98304);
        o.tf32_rn_copy_fn = try o.conv_module.function("k6a_tf32_rn_copy");
        o.sdpa_prepare_fn = try o.conv_module.function("k6a_sdpa_prepare");
        o.sdpa_logits_fn = try o.conv_module.function("k6a_sdpa_logits");
        o.sdpa_reduce_fn = try o.conv_module.function("k6a_sdpa_reduce");
        o.sdpa_softmax_fn = try o.conv_module.function("k6a_sdpa_softmax");
        o.sdpa_value_fn = try o.conv_module.function("k6a_sdpa_value");
        o.sdpa_output_fn = try o.conv_module.function("k6a_sdpa_output");
        try o.sdpa_logits_fn.allowDynamicShared(98304);
        try o.sdpa_value_fn.allowDynamicShared(98304);
        o.nct_to_ntc_fn = try o.conv_module.function("k6a_nct_to_ntc");
        o.kcs_to_ksc_fn = try o.conv_module.function("k6a_kcs_to_ksc");
        o.ntc_to_nct_fn = try o.conv_module.function("k6a_ntc_to_nct");
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
            .{ "l2norm_torch_fn", "k6a_l2norm_torch" },     .{ "sdpa_math_fn", "k6a_sdpa_math" },
        }) |e| @field(o, e[0]) = try m.function(e[1]);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.conv_module.unload();
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

    /// mp_sum(a = shortcut, b = branch, t) in the lerp form with `vae.MpResidual` constants; y may alias a or b.
    pub fn mpResidual(o: *const Ops, s: cuda.Stream, use: Use, x: View, h: View, y: View, m: vae.MpResidual) !void {
        try vae.admitOp(.mp_residual, use);
        try x.expect(.cuda, x.shape[0..x.rank]);
        try h.expect(.cuda, x.shape[0..x.rank]);
        try y.expect(.cuda, x.shape[0..x.rank]);
        var a: cuda.Args = .{};
        inline for (.{ x.addr(), h.addr(), y.addr() }) |p| a.add(p);
        a.add(i64of(x.numel()));
        a.add(m.t);
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

    /// Proven 1x2048x218 residual convolution fixtures only; layout buffers use Decoder.scratch on the caller stream.
    pub fn resConvTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        try x.expect(.cuda, &.{ 1, 2048, 218 });
        try w.view.expect(.cuda, &.{ 2048, 2048, 3 });
        try y.expect(.cuda, &.{ 1, 2048, 218 });
        const cv = try Carver.init(scratch, vae.first_res_tf32_floats);
        const count: u64 = 2048 * 218;
        const a = cv.at("k6a.tf32.input", 0, &.{count});
        const b = cv.at("k6a.tf32.weight", count, &.{2048 * 2048 * 3});
        const out = cv.at("k6a.tf32.output", count + 2048 * 2048 * 3, &.{count});
        for ([_]View{ x, w.view, y, a, b, out }) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
        for ([_]View{ a, b, out }) |copy| for ([_]View{ x, w.view, y }) |source| {
            if (copy.addr() < source.addr() + source.numel() * 4 and source.addr() < copy.addr() + copy.numel() * 4) return error.LayoutUnsupported;
        };
        const operands = [_]View{ x, w.view, y };
        for (operands, 0..) |lhs, i| for (operands[i + 1 ..]) |rhs| {
            if (lhs.addr() < rhs.addr() + rhs.numel() * 4 and rhs.addr() < lhs.addr() + lhs.numel() * 4) return error.LayoutUnsupported;
        };
        var args: cuda.Args = .{};
        args.add(x.addr());
        args.add(a.addr());
        args.add(@as(i32, 2048));
        args.add(@as(i32, 218));
        try flat(o.nct_to_ntc_fn, s, count, &args);
        args = .{};
        args.add(w.view.addr());
        args.add(b.addr());
        args.add(@as(i32, 2048));
        args.add(@as(i32, 3));
        args.add(@as(i32, 2048));
        try flat(o.kcs_to_ksc_fn, s, 2048 * 2048 * 3, &args);
        args = .{};
        args.add(a.addr());
        args.add(b.addr());
        args.add(out.addr());
        try cuda.launch.launch(o.first_res_conv_fn, .{ .grid = .{ .x = 16, .y = 8 }, .block = .{ .x = 128 }, .shared = 81920 }, s, &args);
        args = .{};
        args.add(out.addr());
        args.add(y.addr());
        args.add(@as(i32, 2048));
        args.add(@as(i32, 218));
        try flat(o.ntc_to_nct_fn, s, count, &args);
    }

    /// Proven up1 block0 first-convolution fixture (2048 to 1024 channels) only; layout buffers use Decoder.scratch on the caller stream.
    pub fn up1FirstResConvTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        try x.expect(.cuda, &.{ 1, 2048, 218 });
        try w.view.expect(.cuda, &.{ 1024, 2048, 3 });
        try y.expect(.cuda, &.{ 1, 1024, 218 });
        const cv = try Carver.init(scratch, vae.first_res_tf32_floats);
        const count: u64 = 2048 * 218;
        const output_count: u64 = 1024 * 218;
        const a = cv.at("k6a.tf32.input", 0, &.{count});
        const b = cv.at("k6a.tf32.weight", count, &.{1024 * 2048 * 3});
        const out = cv.at("k6a.tf32.output", count + 1024 * 2048 * 3, &.{output_count});
        for ([_]View{ x, w.view, y, a, b, out }) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
        for ([_]View{ a, b, out }) |copy| for ([_]View{ x, w.view, y }) |source| {
            if (copy.addr() < source.addr() + source.numel() * 4 and source.addr() < copy.addr() + copy.numel() * 4) return error.LayoutUnsupported;
        };
        const operands = [_]View{ x, w.view, y };
        for (operands, 0..) |lhs, i| for (operands[i + 1 ..]) |rhs| {
            if (lhs.addr() < rhs.addr() + rhs.numel() * 4 and rhs.addr() < lhs.addr() + lhs.numel() * 4) return error.LayoutUnsupported;
        };
        var args: cuda.Args = .{};
        args.add(x.addr());
        args.add(a.addr());
        args.add(@as(i32, 2048));
        args.add(@as(i32, 218));
        try flat(o.nct_to_ntc_fn, s, count, &args);
        args = .{};
        args.add(w.view.addr());
        args.add(b.addr());
        args.add(@as(i32, 2048));
        args.add(@as(i32, 3));
        args.add(@as(i32, 1024));
        try flat(o.kcs_to_ksc_fn, s, 1024 * 2048 * 3, &args);
        args = .{};
        args.add(a.addr());
        args.add(b.addr());
        args.add(out.addr());
        try cuda.launch.launch(o.up1_first_res_conv_fn, .{ .grid = .{ .x = 16, .y = 4 }, .block = .{ .x = 128 }, .shared = 81920 }, s, &args);
        args = .{};
        args.add(out.addr());
        args.add(y.addr());
        args.add(@as(i32, 1024));
        args.add(@as(i32, 218));
        try flat(o.ntc_to_nct_fn, s, output_count, &args);
    }

    /// Proven 1024 to 1024 K3 T218 convolution contract (up1 block0 conv2, up1 blocks 1-2 conv1 / conv2, each fixture
    /// all bytes exact); layout buffers use Decoder.scratch on the caller stream.
    pub fn up1SecondResConvTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        try x.expect(.cuda, &.{ 1, 1024, 218 });
        try w.view.expect(.cuda, &.{ 1024, 1024, 3 });
        try y.expect(.cuda, &.{ 1, 1024, 218 });
        const cv = try Carver.init(scratch, vae.first_res_tf32_floats);
        const count: u64 = 1024 * 218;
        const output_count: u64 = 1024 * 218;
        const a = cv.at("k6a.tf32.input", 0, &.{count});
        const b = cv.at("k6a.tf32.weight", count, &.{1024 * 1024 * 3});
        const out = cv.at("k6a.tf32.output", count + 1024 * 1024 * 3, &.{output_count});
        for ([_]View{ x, w.view, y, a, b, out }) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
        for ([_]View{ a, b, out }) |copy| for ([_]View{ x, w.view, y }) |source| {
            if (copy.addr() < source.addr() + source.numel() * 4 and source.addr() < copy.addr() + copy.numel() * 4) return error.LayoutUnsupported;
        };
        const operands = [_]View{ x, w.view, y };
        for (operands, 0..) |lhs, i| for (operands[i + 1 ..]) |rhs| {
            if (lhs.addr() < rhs.addr() + rhs.numel() * 4 and rhs.addr() < lhs.addr() + lhs.numel() * 4) return error.LayoutUnsupported;
        };
        var args: cuda.Args = .{};
        args.add(x.addr());
        args.add(a.addr());
        args.add(@as(i32, 1024));
        args.add(@as(i32, 218));
        try flat(o.nct_to_ntc_fn, s, count, &args);
        args = .{};
        args.add(w.view.addr());
        args.add(b.addr());
        args.add(@as(i32, 1024));
        args.add(@as(i32, 3));
        args.add(@as(i32, 1024));
        try flat(o.kcs_to_ksc_fn, s, 1024 * 1024 * 3, &args);
        args = .{};
        args.add(a.addr());
        args.add(b.addr());
        args.add(out.addr());
        try cuda.launch.launch(o.up1_second_res_conv_fn, .{ .grid = .{ .x = 16, .y = 4 }, .block = .{ .x = 128 }, .shared = 81920 }, s, &args);
        args = .{};
        args.add(out.addr());
        args.add(y.addr());
        args.add(@as(i32, 1024));
        args.add(@as(i32, 218));
        try flat(o.ntc_to_nct_fn, s, output_count, &args);
    }

    /// Proven up1 block0 shortcut fixture (1x1, 2048 to 1024 channels) only: RN operand copies, the 704/704/640 split-K
    /// GEMM and its explicit reduction (conv_tf32.cu k6a_up1_shortcut_*), all in Decoder.scratch on the caller stream.
    pub fn up1FirstResShortcutTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        const l = try Up1Shortcut.carve(x, w.view, y, scratch);
        var args: cuda.Args = .{};
        args.add(x.addr());
        args.add(l.a.addr());
        args.add(i64of(Up1Shortcut.input));
        try flat(o.tf32_rn_copy_fn, s, Up1Shortcut.input, &args);
        args = .{};
        args.add(w.view.addr());
        args.add(l.b.addr());
        args.add(i64of(Up1Shortcut.weights));
        try flat(o.tf32_rn_copy_fn, s, Up1Shortcut.weights, &args);
        args = .{};
        inline for (.{ l.b.addr(), l.a.addr(), l.work.addr() }) |p| args.add(p);
        try cuda.launch.launch(o.up1_shortcut_gemm_fn, .{ .grid = .{ .x = 2, .y = 8, .z = 3 }, .block = .{ .x = 128 }, .shared = 98304 }, s, &args);
        args = .{};
        args.add(l.work.addr());
        args.add(y.addr());
        try cuda.launch.launch(o.up1_shortcut_reduce_fn, .{ .grid = .{ .x = 256, .y = 7 }, .block = .{ .x = 32, .y = 4 } }, s, &args);
    }

    /// Proven up0 block0 shortcut fixture (1x1, 1024 to 512 channels, T 436) only: RN operand copies, the 352/352/320
    /// split-K 256x64x16 GEMM and its explicit reduction (conv_tf32.cu k6a_up0_shortcut_*), all in Decoder.scratch on the
    /// caller stream. Grids: GemmSplitKHorizontalThreadblockSwizzle (N tiles 7, M tiles 2, 3 slices) and ReduceSplitK's
    /// (M / 4, N / 32).
    pub fn up0FirstResShortcutTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        const l = try Up0Shortcut.carve(x, w.view, y, scratch);
        var args: cuda.Args = .{};
        args.add(x.addr());
        args.add(l.a.addr());
        args.add(i64of(Up0Shortcut.input));
        try flat(o.tf32_rn_copy_fn, s, Up0Shortcut.input, &args);
        args = .{};
        args.add(w.view.addr());
        args.add(l.b.addr());
        args.add(i64of(Up0Shortcut.weights));
        try flat(o.tf32_rn_copy_fn, s, Up0Shortcut.weights, &args);
        args = .{};
        inline for (.{ l.b.addr(), l.a.addr(), l.work.addr() }) |p| args.add(p);
        try cuda.launch.launch(o.up0_shortcut_gemm_fn, .{ .grid = .{ .x = 7, .y = 2, .z = 3 }, .block = .{ .x = 128 }, .shared = 81920 }, s, &args);
        args = .{};
        args.add(l.work.addr());
        args.add(y.addr());
        try cuda.launch.launch(o.up0_shortcut_reduce_fn, .{ .grid = .{ .x = 128, .y = 14 }, .block = .{ .x = 32, .y = 4 } }, s, &args);
    }

    /// A proven fprop contract (`tf32_contracts`) on the tile its original ran: NCT -> NTC input copy, KCS -> KSC weight
    /// copy, CUTLASS fprop (conv_tf32.cu k6a_conv_tf32_<tile>) on the GemmIdentityThreadblockSwizzle<4> grid, NTC -> NCT,
    /// then the bias as its own fp32 add (the original adds it in a separate kernel). Caller-carved scratch, caller stream.
    pub fn convTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, bias: ?View, y: View, g: ConvGeom, sc: Tf32Scratch) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        const l = try Tf32Layout.plan(x, w.view, bias, y, g, sc);
        var a: cuda.Args = .{};
        a.add(x.addr());
        a.add(sc.input.addr());
        a.add(i32of(l.ci));
        a.add(i32of(l.t));
        try flat(o.nct_to_ntc_fn, s, l.ci * l.t, &a);
        a = .{};
        a.add(w.view.addr());
        a.add(sc.weight.addr());
        a.add(i32of(l.ci));
        a.add(i32of(l.k));
        a.add(i32of(l.co));
        try flat(o.kcs_to_ksc_fn, s, l.co * l.k * l.ci, &a);
        a = .{};
        inline for (.{ sc.input.addr(), sc.weight.addr(), sc.output.addr() }) |ptr| a.add(ptr);
        inline for (.{ l.ci, l.co, l.t, l.k, l.pad, l.dil }) |v| a.add(i32of(v));
        const grid = tf32Grid(l.tile, l.t, l.co);
        switch (l.tile) {
            .f16x10 => try cuda.launch.launch(o.conv_tf32_16x10_fn, .{ .grid = .{ .x = grid.x, .y = grid.y }, .block = .{ .x = 128 }, .shared = 81920 }, s, &a),
            .f128x64 => try cuda.launch.launch(o.conv_tf32_128x64_fn, .{ .grid = .{ .x = grid.x, .y = grid.y }, .block = .{ .x = 128 }, .shared = 73728 }, s, &a),
            .f32x5 => try cuda.launch.launch(o.conv_tf32_32x5_fn, .{ .grid = .{ .x = grid.x, .y = grid.y }, .block = .{ .x = 128 }, .shared = 81920 }, s, &a),
        }
        a = .{};
        a.add(sc.output.addr());
        a.add(y.addr());
        a.add(i32of(l.co));
        a.add(i32of(l.t));
        try flat(o.ntc_to_nct_fn, s, l.co * l.t, &a);
        if (bias) |bv| {
            a = .{};
            a.add(y.addr());
            a.add(bv.addr());
            a.add(i32of(l.co));
            a.add(i32of(l.t));
            try flat(o.add_bias_nct_fn, s, l.co * l.t, &a);
        }
    }

    /// A proven gathered-GEMM contract (`gather_contracts`): per output phase (one for a convolution, the stride for a
    /// transposed convolution) and row chunk, per slice in the contract's reduce order (one slice unless `split`), the B
    /// operand [kk][co] into `sc.weight`, the A operand [rows][kk] and the GEMM in `sc.rows`, the first slice's into the
    /// running sum D and each later one's added to it (k6a_gg_accumulate); D scattered into y; then the bias as its own
    /// fp32 add. Reduction layout: channel chunk `ck` outer, taps inside, the chunk's channels inner, or (`order`
    /// .flat_k16) the flat KRSC index in k16 blocks (conv_tf32.cu k6a_gg_split).
    pub fn convGather(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, bias: ?View, y: View, g: GatherGeom, sc: GatherScratch) !void {
        try vae.admitOp(if (g.kind == .dgrad) .conv_transpose1d_prepared else .conv1d_prepared, use);
        try w.requirePrepared(use);
        const l = try GatherLayout.plan(x, w.view, bias, y, g, sc);
        const dgrad: i32 = @intFromBool(g.kind == .dgrad);
        const desc: i32 = @intFromBool(l.c.tap_desc);
        const flat_k16: i32 = @intFromBool(l.c.order == .flat_k16);
        for (0..l.u) |ph| {
            const p = l.phase(ph);
            var r0: u64 = 0;
            while (r0 < p.rows) : (r0 += l.chunk) {
                const mc = @min(l.chunk, p.rows - r0);
                const dv = l.rows.at("k6a.gather.d", l.chunk * l.kmax, &.{mc * l.c.co});
                const pv = l.rows.at("k6a.gather.partial", l.chunk * (l.kmax + l.c.co), &.{mc * l.c.co});
                for (0..l.c.split) |i| {
                    const sl = l.sliceAt(i);
                    const kk = l.sliceK(p.nt, sl);
                    const b = l.weight.at("k6a.gather.b", 0, &.{kk * l.c.co});
                    const av = l.rows.at("k6a.gather.a", 0, &.{mc * kk});
                    const out = if (i == 0) dv else pv;
                    var a: cuda.Args = .{};
                    // a single slice's B is the same for every row chunk; a split's B is per slice
                    if (l.c.split > 1 or r0 == 0) {
                        a.add(w.view.addr());
                        a.add(b.addr());
                        inline for (.{ l.c.ci, l.c.co, l.c.k, p.nt, p.s0, p.sstep, l.c.ck }) |v| a.add(i32of(v));
                        a.add(dgrad);
                        a.add(i64of(kk));
                        a.add(i32of(l.c.split));
                        a.add(i32of(sl));
                        a.add(desc);
                        a.add(flat_k16);
                        try flat(o.gg_gather_b_fn, s, kk * l.c.co, &a);
                    }
                    a = .{};
                    a.add(x.addr());
                    a.add(av.addr());
                    a.add(i32of(l.c.ci));
                    a.add(i64of(l.c.t));
                    inline for (.{ p.nt, p.s0, p.sstep, l.c.pad, l.c.dil, l.u, ph, l.c.ck }) |v| a.add(i32of(v));
                    a.add(i64of(r0));
                    a.add(i64of(mc));
                    a.add(dgrad);
                    a.add(i64of(kk));
                    a.add(i32of(l.c.split));
                    a.add(i32of(sl));
                    a.add(desc);
                    a.add(flat_k16);
                    try flat(o.gg_gather_a_fn, s, mc * kk, &a);
                    a = .{};
                    inline for (.{ av.addr(), b.addr(), out.addr() }) |ptr| a.add(ptr);
                    inline for (.{ mc, l.c.co, kk }) |v| a.add(i32of(v));
                    const gr = gatherGrid(mc, l.c.co);
                    try cuda.launch.launch(o.gg_gemm_fn, .{ .grid = .{ .x = gr.x, .y = gr.y }, .block = .{ .x = 128 }, .shared = gather_gemm_smem }, s, &a);
                    if (i > 0) {
                        a = .{};
                        a.add(dv.addr());
                        a.add(pv.addr());
                        a.add(i64of(mc * l.c.co));
                        try flat(o.gg_accumulate_fn, s, mc * l.c.co, &a);
                    }
                }
                var a: cuda.Args = .{};
                a.add(dv.addr());
                a.add(y.addr());
                a.add(i32of(l.c.co));
                a.add(i64of(l.tout));
                a.add(i32of(l.u));
                a.add(i32of(ph));
                a.add(i64of(r0));
                a.add(i64of(mc));
                try flat(o.gg_scatter_fn, s, mc * l.c.co, &a);
            }
        }
        if (bias) |bv| {
            var a: cuda.Args = .{};
            a.add(y.addr());
            a.add(bv.addr());
            a.add(i32of(l.c.co));
            a.add(i32of(l.tout));
            try flat(o.add_bias_nct_fn, s, l.c.co * l.tout, &a);
        }
    }

    /// Exact QKV fixture only; rounded operand copies reuse the accounted convolution scratch.
    pub fn attentionQkvTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        try x.expect(.cuda, &.{ 1, 2048, 218 });
        try w.view.expect(.cuda, &.{ 6144, 2048, 1 });
        try y.expect(.cuda, &.{ 1, 6144, 218 });
        const count: u64 = 2048 * 218;
        const weights: u64 = 6144 * 2048;
        const cv = try Carver.init(scratch, count + weights);
        const a = cv.at("k6a.qkv.tf32.input", 0, &.{count});
        const b = cv.at("k6a.qkv.tf32.weight", count, &.{weights});
        for ([_]View{ x, w.view, y, a, b }) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
        for ([_]View{ a, b }) |copy| for ([_]View{ x, w.view, y }) |source| {
            if (copy.addr() < source.addr() + source.numel() * 4 and source.addr() < copy.addr() + copy.numel() * 4) return error.LayoutUnsupported;
        };
        var args: cuda.Args = .{};
        args.add(x.addr());
        args.add(a.addr());
        args.add(i64of(count));
        try flat(o.tf32_rn_copy_fn, s, count, &args);
        args = .{};
        args.add(w.view.addr());
        args.add(b.addr());
        args.add(i64of(weights));
        try flat(o.tf32_rn_copy_fn, s, weights, &args);
        args = .{};
        args.add(a.addr());
        args.add(b.addr());
        args.add(y.addr());
        try cuda.launch.launch(o.attention_qkv_fn, .{ .grid = .{ .x = 48, .y = 2 }, .block = .{ .x = 128 }, .shared = 98304 }, s, &args);
    }

    /// Exact attention output projection fixture only; rounded operand copies reuse the accounted convolution scratch.
    pub fn attentionProjTf32(o: *const Ops, s: cuda.Stream, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try vae.admitOp(.conv1d_prepared, use);
        try w.requirePrepared(use);
        try x.expect(.cuda, &.{ 1, 2048, 218 });
        try w.view.expect(.cuda, &.{ 2048, 2048, 1 });
        try y.expect(.cuda, &.{ 1, 2048, 218 });
        const count: u64 = 2048 * 218;
        const weights: u64 = 2048 * 2048;
        const cv = try Carver.init(scratch, count + weights);
        const a = cv.at("k6a.proj.tf32.input", 0, &.{count});
        const b = cv.at("k6a.proj.tf32.weight", count, &.{weights});
        for ([_]View{ x, w.view, y, a, b }) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
        for ([_]View{ a, b }) |copy| for ([_]View{ x, w.view, y }) |source| {
            if (copy.addr() < source.addr() + source.numel() * 4 and source.addr() < copy.addr() + copy.numel() * 4) return error.LayoutUnsupported;
        };
        const operands = [_]View{ x, w.view, y };
        for (operands, 0..) |lhs, i| for (operands[i + 1 ..]) |rhs| {
            if (lhs.addr() < rhs.addr() + rhs.numel() * 4 and rhs.addr() < lhs.addr() + lhs.numel() * 4) return error.LayoutUnsupported;
        };
        var args: cuda.Args = .{};
        args.add(x.addr());
        args.add(a.addr());
        args.add(i64of(count));
        try flat(o.tf32_rn_copy_fn, s, count, &args);
        args = .{};
        args.add(w.view.addr());
        args.add(b.addr());
        args.add(i64of(weights));
        try flat(o.tf32_rn_copy_fn, s, weights, &args);
        args = .{};
        args.add(a.addr());
        args.add(b.addr());
        args.add(y.addr());
        try cuda.launch.launch(o.attention_proj_fn, .{ .grid = .{ .x = 16, .y = 2 }, .block = .{ .x = 128 }, .shared = 98304 }, s, &args);
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
        switch (reduce) {
            .torch_reduce => |dev| {
                const sp = try vae.torchSplit(g, dev);
                var a: cuda.Args = .{};
                a.add(x.addr());
                a.add(norms.addr());
                inline for (.{ g.outer, g.n, g.inner }) |v| a.add(i64of(v));
                a.add(i32of(sp.y));
                a.add(i32of(sp.ctas));
                try go(o.l2norm_torch_fn, s, .{ .x = @intCast((g.outer * g.inner + 127) / 128) }, 128, &a);
            },
            .serial_fma, .serial_mul_add, .strided_tree => {
                const lanes: u32 = switch (reduce) {
                    .strided_tree => |r| blk: {
                        if (r == 0 or r > 1024 or !std.math.isPowerOfTwo(r)) return error.InvalidArgument;
                        break :blk r;
                    },
                    else => 1,
                };
                var a: cuda.Args = .{};
                a.add(x.addr());
                a.add(norms.addr());
                inline for (.{ g.outer, g.n, g.inner }) |v| a.add(i64of(v));
                a.add(@as(i32, switch (reduce) {
                    .serial_fma => 0,
                    .serial_mul_add => 1,
                    else => 2,
                }));
                a.add(i32of(lanes));
                const threads = @max(32, (lanes + 31) / 32 * 32);
                try cuda.launch.launch(o.l2norm_fn, .{ .grid = .{ .x = @intCast(g.outer * g.inner) }, .block = .{ .x = threads }, .shared = lanes * @sizeOf(f32) }, s, &a);
            },
        }
        var b: cuda.Args = .{};
        inline for (.{ x.addr(), norms.addr(), y.addr() }) |p| b.add(p);
        inline for (.{ g.outer, g.n, g.inner }) |v| b.add(i64of(v));
        b.add(g.scale);
        b.add(eps);
        b.add(@as(i32, @intFromBool(den == .fma)));
        try flat(o.normalize_apply_fn, s, x.numel(), &b);
    }

    /// The qualified T218 middle-attention SDPA (the native decoder's only attention path). All copies and partition
    /// workspace reuse the caller's accounted scratch.
    pub fn midAttentionTf32(o: *const Ops, s: cuda.Stream, use: Use, q: View, k: View, v: View, out: View, scratch: View) !void {
        try vae.admitOp(.mid_attention_sdpa, use);
        for ([_]View{ q, k, v, out }) |x| try x.expect(.cuda, &.{ 1, 218, 2048 });
        const count: u64 = 446464;
        const scores: u64 = 47524;
        const partitions: u64 = 12 * scores;
        const cv = try Carver.init(scratch, 4 * count + 2 * scores + partitions);
        const qa = cv.at("k6a.sdpa.q", 0, &.{count});
        const kb = cv.at("k6a.sdpa.k", count, &.{count});
        const vc = cv.at("k6a.sdpa.v", 2 * count, &.{count});
        const logits = cv.at("k6a.sdpa.logits", 3 * count, &.{scores});
        const probs = cv.at("k6a.sdpa.probs", 3 * count + scores, &.{scores});
        const work = cv.at("k6a.sdpa.partitions", 3 * count + 2 * scores, &.{partitions});
        const temp = cv.at("k6a.sdpa.value", 3 * count + 2 * scores + partitions, &.{count});
        for ([_]View{ q, k, v, out, qa, kb, vc, logits, probs, work, temp }) |x| if (x.addr() % 16 != 0) return error.LayoutUnsupported;
        for ([_]View{ qa, kb, vc, logits, probs, work, temp }) |copy| for ([_]View{ q, k, v, out }) |source| {
            if (copy.addr() < source.addr() + source.numel() * 4 and source.addr() < copy.addr() + copy.numel() * 4) return error.LayoutUnsupported;
        };
        for ([_]View{ q, k, v }) |source| if (out.addr() < source.addr() + source.numel() * 4 and source.addr() < out.addr() + out.numel() * 4) return error.LayoutUnsupported;
        const scale: f32 = @floatCast(@sqrt(1.0 / @sqrt(@as(f64, 2048.0))));
        var args: cuda.Args = .{};
        inline for (.{ q.addr(), k.addr(), v.addr(), qa.addr(), kb.addr(), vc.addr() }) |p| args.add(p);
        args.add(scale);
        try flat(o.sdpa_prepare_fn, s, count, &args);
        args = .{};
        inline for (.{ qa.addr(), kb.addr(), work.addr() }) |p| args.add(p);
        try cuda.launch.launch(o.sdpa_logits_fn, .{ .grid = .{ .x = 2, .y = 2, .z = 12 }, .block = .{ .x = 128 }, .shared = 98304 }, s, &args);
        args = .{};
        args.add(work.addr());
        args.add(logits.addr());
        try cuda.launch.launch(o.sdpa_reduce_fn, .{ .grid = .{ .x = 55, .y = 7 }, .block = .{ .x = 32, .y = 4 } }, s, &args);
        args = .{};
        args.add(probs.addr());
        args.add(logits.addr());
        try cuda.launch.launch(o.sdpa_softmax_fn, .{ .grid = .{ .x = 55 }, .block = .{ .x = 32, .y = 4 } }, s, &args);
        args = .{};
        args.add(probs.addr());
        args.add(probs.addr());
        args.add(i64of(scores));
        try flat(o.tf32_rn_copy_fn, s, scores, &args);
        args = .{};
        inline for (.{ vc.addr(), probs.addr(), temp.addr() }) |p| args.add(p);
        try cuda.launch.launch(o.sdpa_value_fn, .{ .grid = .{ .x = 16, .y = 2 }, .block = .{ .x = 128 }, .shared = 98304 }, s, &args);
        args = .{};
        args.add(temp.addr());
        args.add(out.addr());
        try flat(o.sdpa_output_fn, s, count, &args);
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

/// The up1 block0 shortcut (1x1, 2048 to 1024, T 218) and the up0 block0 shortcut (1024 to 512, T 436).
const Up1Shortcut = SplitKShortcut(2048, 1024, 218);
const Up0Shortcut = SplitKShortcut(1024, 512, 436);

/// A split-K shortcut's buffers inside the accounted convolution scratch region (vae.first_res_tf32_floats, no growth):
/// RN input [ci, t] at 0, RN weight [co, ci] after it, then the three split-K partitions [co, t]. Nothing may alias x,
/// w or y, and x, w, y may not alias each other (the reduction writes y while x is still live).
fn SplitKShortcut(comptime ci: u64, comptime co: u64, comptime t: u64) type {
    return struct {
        const Self = @This();
        const input: u64 = ci * t;
        const weights: u64 = co * ci;
        const outputs: u64 = co * t;
        const slices: u64 = 3;
        const floats: u64 = input + weights + slices * outputs;
        comptime {
            std.debug.assert(floats <= vae.first_res_tf32_floats);
        }

        a: View,
        b: View,
        work: View,

        fn carve(x: View, w: View, y: View, scratch: View) !Self {
            try x.expect(.cuda, &.{ 1, ci, t });
            try w.expect(.cuda, &.{ co, ci, 1 });
            try y.expect(.cuda, &.{ 1, co, t });
            const cv = try Carver.init(scratch, vae.first_res_tf32_floats);
            const l: Self = .{
                .a = cv.at("k6a.tf32.input", 0, &.{input}),
                .b = cv.at("k6a.tf32.weight", input, &.{weights}),
                .work = cv.at("k6a.tf32.partitions", input + weights, &.{slices * outputs}),
            };
            for ([_]View{ x, w, y, l.a, l.b, l.work }) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
            for ([_]View{ l.a, l.b, l.work }) |copy| for ([_]View{ x, w, y }) |source| {
                if (copy.addr() < source.addr() + source.numel() * 4 and source.addr() < copy.addr() + copy.numel() * 4) return error.LayoutUnsupported;
            };
            const operands = [_]View{ x, w, y };
            for (operands, 0..) |lhs, i| for (operands[i + 1 ..]) |rhs| {
                if (lhs.addr() < rhs.addr() + rhs.numel() * 4 and rhs.addr() < lhs.addr() + lhs.numel() * 4) return error.LayoutUnsupported;
            };
            return l;
        }
    };
}

/// The fprop tile families the originals ran (window-2 isolate profiles). The K tile fixes the order in which partial
/// sums are added (the optimized iterator walks the taps inside each channel tile), so it is part of the arithmetic.
pub const Tf32Tile = enum {
    f16x10,
    f128x64,
    /// 64x64x32, 5 stages: the proven T218 kernel type (conv_tf32.cu `K`) with runtime shapes.
    f32x5,

    fn tileM(t: Tf32Tile) u64 {
        return switch (t) {
            .f16x10, .f32x5 => 64,
            .f128x64 => 128,
        };
    }

    fn smem(t: Tf32Tile) u32 {
        return switch (t) {
            .f16x10, .f32x5 => 81920,
            .f128x64 => 73728,
        };
    }
};

/// One proven convolution: batch 1, stride 1, same-length output, all bytes exact on its reference fixture with `tile`.
pub const Tf32Contract = struct { ci: u64, co: u64, t: u64, k: u64, pad: u64, dil: u64, tile: Tf32Tile };

/// Every proven fprop contract (window-3 audio-w3-candidates-v1, each shape exact on its reference fixture; decoder
/// fixtures from reference 12, vocoder fixtures from vocoder reference v1). On the qualified geometry `decoderConv` /
/// `vocoderConv` reach `convTf32` only for a listed shape; nothing is inferred for any other shape.
pub const tf32_contracts = [_]Tf32Contract{
    // decoder: local T436 (latent T218)
    .{ .ci = 1024, .co = 1024, .t = 436, .k = 3, .pad = 1, .dil = 1, .tile = .f16x10 }, // up.1.upsample.conv
    .{ .ci = 1024, .co = 512, .t = 436, .k = 3, .pad = 1, .dil = 1, .tile = .f16x10 }, // up.0.block.0.conv1
    .{ .ci = 512, .co = 512, .t = 436, .k = 3, .pad = 1, .dil = 1, .tile = .f16x10 }, // up.0 conv2s and blocks 1-2
    .{ .ci = 512, .co = 128, .t = 436, .k = 3, .pad = 1, .dil = 1, .tile = .f16x10 }, // conv_out (gain-folded weight)
    // vocoder: 436 mel frames
    .{ .ci = 128, .co = 1536, .t = 436, .k = 7, .pad = 3, .dil = 1, .tile = .f16x10 }, // conv_pre (+ bias add)
    .{ .ci = 768, .co = 768, .t = 3488, .k = 3, .pad = 1, .dil = 1, .tile = .f128x64 }, // stage 0, k3: convs1[0], convs2
    .{ .ci = 768, .co = 768, .t = 3488, .k = 3, .pad = 3, .dil = 3, .tile = .f128x64 }, // stage 0, k3: convs1[1]
    .{ .ci = 768, .co = 768, .t = 3488, .k = 3, .pad = 5, .dil = 5, .tile = .f128x64 }, // stage 0, k3: convs1[2]
    // vocoder AMP convolutions, window-4 sweep (audio-w4-vsweep-v1): every instance of each shape all bytes exact on the
    // tile its own original ran (dilation-1 shapes: convs1[0] and the three convs2 of the block)
    .{ .ci = 768, .co = 768, .t = 3488, .k = 7, .pad = 3, .dil = 1, .tile = .f128x64 }, // stage 0, k7: 4 convs
    .{ .ci = 768, .co = 768, .t = 3488, .k = 7, .pad = 9, .dil = 3, .tile = .f128x64 },
    .{ .ci = 768, .co = 768, .t = 3488, .k = 7, .pad = 15, .dil = 5, .tile = .f128x64 },
    .{ .ci = 96, .co = 96, .t = 55808, .k = 7, .pad = 3, .dil = 1, .tile = .f32x5 }, // stage 3, k7: 4 convs
    .{ .ci = 96, .co = 96, .t = 55808, .k = 7, .pad = 9, .dil = 3, .tile = .f32x5 },
    .{ .ci = 96, .co = 96, .t = 55808, .k = 7, .pad = 15, .dil = 5, .tile = .f32x5 },
    .{ .ci = 96, .co = 96, .t = 55808, .k = 11, .pad = 5, .dil = 1, .tile = .f32x5 }, // stage 3, k11: 4 convs
    .{ .ci = 96, .co = 96, .t = 55808, .k = 11, .pad = 15, .dil = 3, .tile = .f32x5 },
    .{ .ci = 96, .co = 96, .t = 55808, .k = 11, .pad = 25, .dil = 5, .tile = .f32x5 },
    .{ .ci = 48, .co = 48, .t = 111616, .k = 7, .pad = 3, .dil = 1, .tile = .f32x5 }, // stage 4, k7: 4 convs
    .{ .ci = 48, .co = 48, .t = 111616, .k = 7, .pad = 9, .dil = 3, .tile = .f32x5 },
    .{ .ci = 48, .co = 48, .t = 111616, .k = 7, .pad = 15, .dil = 5, .tile = .f32x5 },
    .{ .ci = 48, .co = 48, .t = 111616, .k = 11, .pad = 5, .dil = 1, .tile = .f32x5 }, // stage 4, k11: 4 convs
    .{ .ci = 48, .co = 48, .t = 111616, .k = 11, .pad = 15, .dil = 3, .tile = .f32x5 },
    .{ .ci = 48, .co = 48, .t = 111616, .k = 11, .pad = 25, .dil = 5, .tile = .f32x5 },
};

fn tf32Contract(ci: u64, co: u64, t: u64, k: u64, pad: u64, dil: u64) ?Tf32Tile {
    for (tf32_contracts) |c| {
        if (c.ci == ci and c.co == co and c.t == t and c.k == k and c.pad == pad and c.dil == dil) return c.tile;
    }
    return null;
}

/// GemmIdentityThreadblockSwizzle<4>::get_grid_shape (vendor threadblock_swizzle.h:110-127) over the implicit GEMM
/// (M = T output positions, N = Cout; every tile family is 64 wide in N): the grid the kernel's own Params derive.
pub fn tf32Grid(tile: Tf32Tile, t: u64, co: u64) struct { x: u32, y: u32 } {
    const m = (t + tile.tileM() - 1) / tile.tileM();
    const n = (co + 63) / 64;
    const log: u6 = if (n >= 3) 2 else if (n >= 2) 1 else 0; // N = 4: the log-3 branch needs N >= 8
    return .{ .x = @intCast(m << log), .y = @intCast((n + (@as(u64, 1) << log) - 1) >> log) };
}

/// One `convTf32`'s caller-carved scratch: the NTC input copy, the KSC weight copy and the NTC output.
pub const Tf32Scratch = struct { input: View, weight: View, output: View };

fn overlaps(a: View, b: View) bool {
    return a.addr() < b.addr() + b.numel() * 4 and b.addr() < a.addr() + a.numel() * 4;
}

/// The checks of a `convTf32` before anything is queued: a listed contract, exact scratch sizes, 16-byte aligned
/// CUTLASS operands, and no buffer aliasing another (the copies are written while x, w and bias are still read).
const Tf32Layout = struct {
    ci: u64,
    co: u64,
    t: u64,
    k: u64,
    pad: u64,
    dil: u64,
    tile: Tf32Tile,

    fn plan(x: View, w: View, bias: ?View, y: View, g: Ops.ConvGeom, sc: Tf32Scratch) !Tf32Layout {
        const ci = x.shape[1];
        const t = x.shape[2];
        const co = w.shape[0];
        const k = w.shape[2];
        try x.expect(.cuda, &.{ 1, ci, t });
        try w.expect(.cuda, &.{ co, ci, k });
        try y.expect(.cuda, &.{ 1, co, t });
        if (bias) |bv| try bv.expect(.cuda, &.{co});
        if (g.stride != 1) return error.InvalidArgument;
        const tile = tf32Contract(ci, co, t, k, g.pad, g.dil) orelse return error.InvalidArgument;
        try sc.input.expect(.cuda, &.{ci * t});
        try sc.weight.expect(.cuda, &.{co * k * ci});
        try sc.output.expect(.cuda, &.{co * t});
        const copies = [_]View{ sc.input, sc.weight, sc.output };
        for (copies) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
        for (copies, 0..) |c, i| {
            for (copies[i + 1 ..]) |other| if (overlaps(c, other)) return error.LayoutUnsupported;
            for ([_]View{ x, w, y }) |src| if (overlaps(c, src)) return error.LayoutUnsupported;
            if (bias) |bv| if (overlaps(c, bv)) return error.LayoutUnsupported;
        }
        if (overlaps(x, y) or overlaps(w, y)) return error.LayoutUnsupported;
        if (bias) |bv| if (overlaps(bv, y)) return error.LayoutUnsupported;
        return .{ .ci = ci, .co = co, .t = t, .k = k, .pad = g.pad, .dil = g.dil, .tile = tile };
    }
};

/// The dynamic shared memory of conv_tf32.cu k6a_gg_gemm: 4 stages x (128 x 16 + 16 x 128) floats (static_assert there).
const gather_gemm_smem: u32 = 65536;

pub const GatherKind = enum { fprop, dgrad };

/// The reduction layout: `chunked` = channel chunks of `ck` (round-robin over `split` slices), taps inside; `flat_k16` =
/// the flat KRSC filter index f = tap * ci + c in k16 blocks, each two MMAs (f mod 4 in {0, 1}, then {2, 3}), padded with
/// zero columns to a multiple of 16 (conv_tf32.cu k6a_gg_split); one slice, `ck` unused.
pub const GatherOrder = enum { chunked, flat_k16 };

/// One proven gathered-GEMM convolution, batch 1. fprop: Conv1d, stride 1, same-length output, weight [co][ci][k];
/// dgrad: ConvTranspose1d, dilation 1, output padding 0, weight [ci][co][k]. `t` is the input length, `ck` the reduction
/// layout's channel chunk (conv_tf32.cu k6a_gg_split). `split` > 1: the chunks go round-robin to that many slices, each
/// its own GEMM, `tap_desc` orders a chunk's taps descending, and the slices' partial sums are added in `reduce` order
/// (empty: ascending). `order` selects the layout.
pub const GatherContract = struct { kind: GatherKind, ci: u64, co: u64, t: u64, k: u64, pad: u64, dil: u64 = 1, stride: u64 = 1, ck: u64, split: u64 = 1, tap_desc: bool = false, reduce: []const u8 = &.{}, order: GatherOrder = .chunked };

/// Every proven gathered-GEMM contract (window-5 sweep audio-w5-sweep2-v1, contracts.json: every instance of each shape
/// all bytes exact with this `ck`, taps ascending, no operand pre-rounding, no split K; one variant per shape group;
/// entries with `split` / `tap_desc` / `reduce` follow the window-6 oracle's measured order instead, verify exact; entries
/// with `order = .flat_k16` window 7's oracle2 order, audio-w7-verify17-v1 exact).
/// On the qualified vocoder `vocoderConv` / `vocoderConvTranspose` reach `convGather` only for a listed shape; nothing is
/// inferred for any other shape.
pub const gather_contracts = [_]GatherContract{
    // vocoder AMP convolutions whose originals ran sm80_xmma_fprop_implicit_gemm_..._nhwckrsc_nchw_128x128x16_stage4
    // (dilation-1 shapes: convs1[0] and the block's three convs2)
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 3, .pad = 1, .dil = 1, .ck = 16 }, // stage 1: kernels 3, 7, 11 at dilations 1, 3, 5
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 3, .pad = 3, .dil = 3, .ck = 16 },
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 3, .pad = 5, .dil = 5, .ck = 16 },
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 7, .pad = 3, .dil = 1, .ck = 16 },
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 7, .pad = 9, .dil = 3, .ck = 16 },
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 7, .pad = 15, .dil = 5, .ck = 16 },
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 11, .pad = 5, .dil = 1, .ck = 16 },
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 11, .pad = 15, .dil = 3, .ck = 16 },
    .{ .kind = .fprop, .ci = 384, .co = 384, .t = 13952, .k = 11, .pad = 25, .dil = 5, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 3, .pad = 1, .dil = 1, .ck = 16 }, // stage 2: kernels 3, 7, 11 at dilations 1, 3, 5
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 3, .pad = 3, .dil = 3, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 3, .pad = 5, .dil = 5, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 7, .pad = 3, .dil = 1, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 7, .pad = 9, .dil = 3, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 7, .pad = 15, .dil = 5, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 11, .pad = 5, .dil = 1, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 11, .pad = 15, .dil = 3, .ck = 16 },
    .{ .kind = .fprop, .ci = 192, .co = 192, .t = 27904, .k = 11, .pad = 25, .dil = 5, .ck = 16 },
    .{ .kind = .fprop, .ci = 96, .co = 96, .t = 55808, .k = 3, .pad = 1, .dil = 1, .ck = 16 }, // stage 3: kernel 3
    .{ .kind = .fprop, .ci = 96, .co = 96, .t = 55808, .k = 3, .pad = 3, .dil = 3, .ck = 16 },
    .{ .kind = .fprop, .ci = 96, .co = 96, .t = 55808, .k = 3, .pad = 5, .dil = 5, .ck = 16 },
    // stage 0 kernel 11 (convs1 at dilations 1, 3, 5 and the three convs2), whose original ran
    // sm80_xmma_fprop_implicit_gemm_indexed_..._128x128x16_stage4 on grid z 17: the window-6 oracle's measured order
    // (audio-w6-oracle-v1, model holds) and gather-gemm v2 exact on all six (audio-w6-verify-v1)
    .{ .kind = .fprop, .ci = 768, .co = 768, .t = 3488, .k = 11, .pad = 5, .dil = 1, .ck = 16, .split = 17 },
    .{ .kind = .fprop, .ci = 768, .co = 768, .t = 3488, .k = 11, .pad = 15, .dil = 3, .ck = 16, .split = 17 },
    .{ .kind = .fprop, .ci = 768, .co = 768, .t = 3488, .k = 11, .pad = 25, .dil = 5, .ck = 16, .split = 17 },
    // stage 5 kernel 11 (resblocks.17: convs1 at dilations 1, 3, 5 and the three convs2), whose original ran
    // sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_..._128x32x64_stage1 alignc4: window 7's oracle2 (groups measured,
    // hypothesis H1 predicting every query) and gather-gemm v2 exact on all six (audio-w7-verify17-v1)
    .{ .kind = .fprop, .ci = 24, .co = 24, .t = 223232, .k = 11, .pad = 5, .dil = 1, .ck = 16, .order = .flat_k16 },
    .{ .kind = .fprop, .ci = 24, .co = 24, .t = 223232, .k = 11, .pad = 15, .dil = 3, .ck = 16, .order = .flat_k16 },
    .{ .kind = .fprop, .ci = 24, .co = 24, .t = 223232, .k = 11, .pad = 25, .dil = 5, .ck = 16, .order = .flat_k16 },
    // vocoder upsamplers ups.1-5, whose originals ran CUTLASS strided dgrad (s1688dgrad_optimized_tf32_64x128_16x6 or
    // _64x64_32x5): `ck` is the kernel's K tile
    .{ .kind = .dgrad, .ci = 768, .co = 384, .t = 3488, .k = 8, .pad = 2, .stride = 4, .ck = 16 },
    .{ .kind = .dgrad, .ci = 384, .co = 192, .t = 13952, .k = 4, .pad = 1, .stride = 2, .ck = 16 },
    .{ .kind = .dgrad, .ci = 192, .co = 96, .t = 27904, .k = 4, .pad = 1, .stride = 2, .ck = 16 },
    .{ .kind = .dgrad, .ci = 96, .co = 48, .t = 55808, .k = 4, .pad = 1, .stride = 2, .ck = 32 },
    .{ .kind = .dgrad, .ci = 48, .co = 24, .t = 111616, .k = 4, .pad = 1, .stride = 2, .ck = 32 },
    // ups.0 (vup0), whose original ran sm80_xmma_dgrad_implicit_gemm_indexed_..._strided on grid z 10: the window-6
    // oracle's measured order (taps descending, slice 9 reduced second) and gather-gemm v2 exact (audio-w6-verify-v1)
    .{ .kind = .dgrad, .ci = 1536, .co = 768, .t = 436, .k = 16, .pad = 4, .stride = 8, .ck = 16, .split = 10, .tap_desc = true, .reduce = &.{ 0, 9, 1, 2, 3, 4, 5, 6, 7, 8 } },
};

fn gatherContract(kind: GatherKind, ci: u64, co: u64, t: u64, k: u64, pad: u64, dil: u64, stride: u64) ?GatherContract {
    for (gather_contracts) |c| {
        if (c.kind == kind and c.ci == ci and c.co == co and c.t == t and c.k == k and c.pad == pad and c.dil == dil and c.stride == stride) return c;
    }
    return null;
}

/// GemmIdentityThreadblockSwizzle<1>::get_grid_shape over a 128x128 tile (log tile 0): the grid k6a_gg_gemm's Params derive.
pub fn gatherGrid(m: u64, n: u64) struct { x: u32, y: u32 } {
    return .{ .x = @intCast((m + 127) / 128), .y = @intCast((n + 127) / 128) };
}

pub const GatherGeom = struct { kind: GatherKind, pad: u32, dil: u32 = 1, stride: u32 = 1 };

/// One `convGather`'s caller-carved scratch (rank-1 regions): `weight` holds one slice's B operand, `rows` one row
/// chunk's A operand, the running sum D and (split > 1) a slice's partial.
pub const GatherScratch = struct { weight: View, rows: View };

/// The checks of a `convGather` before anything is queued: a listed contract, scratch regions large enough for the
/// largest slice's B and at least one 128-row chunk, and no buffer aliasing another.
const GatherLayout = struct {
    c: GatherContract,
    /// output phases: 1 for a convolution, the stride for a transposed convolution
    u: u64,
    tout: u64,
    /// rows per GEMM: as many 128-row tiles as `rows` holds for the widest phase, at most a phase's rows
    chunk: u64,
    /// the largest slice's reduction length over the phases: A is [chunk][kmax] at the start of `rows`, then the
    /// running sum D [chunk][co] and (split > 1) a slice's partial [chunk][co]
    kmax: u64,
    weight: Carver,
    rows: Carver,

    const Phase = struct { nt: u64, s0: u64, sstep: u64, rows: u64 };

    /// Slice `sl`'s reduction length in a phase of `nt` taps: its chunks sl, sl + split, ... (all of ci when split is 1).
    fn sliceK(l: GatherLayout, nt: u64, sl: u64) u64 {
        if (l.c.order == .flat_k16) return (l.c.ci * nt + 15) / 16 * 16;
        if (l.c.split == 1) return l.c.ci * nt;
        return (l.c.ci / l.c.ck - sl + l.c.split - 1) / l.c.split * l.c.ck * nt;
    }

    /// The slice whose partial is added `i`-th.
    fn sliceAt(l: GatherLayout, i: u64) u64 {
        return if (l.c.reduce.len == 0) i else l.c.reduce[i];
    }

    /// Output phase `ph`: its taps s0, s0 + sstep, ... (nt of them) and its output rows t = ph, ph + u, ...
    fn phase(l: GatherLayout, ph: u64) Phase {
        if (l.c.kind == .fprop) return .{ .nt = l.c.k, .s0 = 0, .sstep = 1, .rows = l.tout };
        const s0 = (ph + l.c.pad) % l.u;
        return .{ .nt = if (s0 < l.c.k) (l.c.k - s0 + l.u - 1) / l.u else 0, .s0 = s0, .sstep = l.u, .rows = (l.tout - ph + l.u - 1) / l.u };
    }

    fn plan(x: View, w: View, bias: ?View, y: View, g: GatherGeom, sc: GatherScratch) !GatherLayout {
        const dgrad = g.kind == .dgrad;
        const ci = x.shape[1];
        const t = x.shape[2];
        const co = if (dgrad) w.shape[1] else w.shape[0];
        const k = w.shape[2];
        try x.expect(.cuda, &.{ 1, ci, t });
        if (dgrad) try w.expect(.cuda, &.{ ci, co, k }) else try w.expect(.cuda, &.{ co, ci, k });
        if (bias) |bv| try bv.expect(.cuda, &.{co});
        const c = gatherContract(g.kind, ci, co, t, k, g.pad, g.dil, g.stride) orelse return error.InvalidArgument;
        const tout = if (dgrad) vae.convTransposeOutLen(t, k, g.stride, g.pad) else t;
        try y.expect(.cuda, &.{ 1, co, tout });
        if (c.split > 1 and (c.ci % c.ck != 0 or c.split > c.ci / c.ck)) return error.InvalidArgument;
        if (c.order == .flat_k16 and (c.split != 1 or c.tap_desc or c.reduce.len != 0 or dgrad)) return error.InvalidArgument;
        if (c.reduce.len != 0) {
            if (c.reduce.len != c.split) return error.InvalidArgument;
            var seen: [256]bool = @splat(false);
            for (c.reduce) |r| {
                if (r >= c.split or seen[r]) return error.InvalidArgument;
                seen[r] = true;
            }
        }
        var l: GatherLayout = .{ .c = c, .u = if (dgrad) g.stride else 1, .tout = tout, .chunk = 0, .kmax = 0, .weight = undefined, .rows = undefined };
        var rmax: u64 = 0;
        for (0..l.u) |ph| {
            const p = l.phase(ph);
            if (p.nt == 0 or p.rows == 0) return error.InvalidArgument;
            l.kmax = @max(l.kmax, l.sliceK(p.nt, 0));
            rmax = @max(rmax, p.rows);
        }
        const per_row: u64 = l.kmax + co * @as(u64, if (c.split > 1) 2 else 1);
        l.weight = try Carver.init(sc.weight, l.kmax * co);
        try sc.rows.validate();
        if (sc.rows.rank != 1) return error.LayoutUnsupported;
        const chunk: u64 = sc.rows.shape[0] / per_row / 128 * 128;
        l.chunk = @min(rmax, chunk);
        if (l.chunk == 0) return error.OutOfBounds;
        l.rows = try Carver.init(sc.rows, l.chunk * per_row);
        if (overlaps(sc.weight, sc.rows)) return error.LayoutUnsupported;
        for ([_]View{ sc.weight, sc.rows }) |r| {
            for ([_]View{ x, w, y }) |src| if (overlaps(r, src)) return error.LayoutUnsupported;
            if (bias) |bv| if (overlaps(r, bv)) return error.LayoutUnsupported;
        }
        if (overlaps(x, y) or overlaps(w, y)) return error.LayoutUnsupported;
        if (bias) |bv| if (overlaps(bv, y)) return error.LayoutUnsupported;
        return l;
    }
};

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
    /// Accumulation of every decoder convolution that has no qualified path (`decodeWith`: on the native 1 x 218 decoder
    /// conv_in and nothing else; every convolution of any other geometry). Qualified: `.fma`.
    acc: Acc,
    /// latent_unscale's division by scaling_factor (native, D3 window: `.reciprocal`).
    unscale: Div,
    /// scaled SiLU's division by 0.596 (native, D3 probe dumps: `.reciprocal`, f32(1 / 0.596); expf and the build flags
    /// are not a factor, torch's silu equals x / (1 + expf(-x)) bit for bit).
    silu: Div,
    /// mp_sum's division by sqrt((1 - t)^2 + t^2) (native, D3 window: `.reciprocal`).
    residual: Div,
    /// The residual blocks' normalization (axis 1) and the attention's packed-QKV normalization (axis 2).
    norm_residual: NormPolicy,
    norm_qkv: NormPolicy,
    /// The middle attention off the native geometry: the `k6a_sdpa_math` comparison candidate (the native 1 x 218 decoder
    /// always runs the qualified `midAttentionTf32`).
    attention: enum { math_candidate },
};

/// Optional component diagnostic observer; null on ordinary decoder runs.
pub const DecodeHook = struct {
    ctx: *anyopaque,
    op: *const fn (ctx: *anyopaque, name: []const u8, out: View) anyerror!void,
};

pub fn decodeTraced(o: *const Ops, s: cuda.Stream, use: Use, d: vae.Decoder, w: *const vae.DecoderWeights, rows: View, scratch: View, mel: View, p: DecodePolicy, hook: ?DecodeHook) !void {
    try decodeWith(o, s, use, d, w, rows, scratch, mel, p, hook);
}

/// Decoder1D + VAE.unnormalize: `rows` the sampler's latent rows [B, T, 40] -> `mel` [B, 128, 2T], with the weights from
/// `vae.bindDecoder` and `scratch` a rank-1 `.cuda` view of at least `d.scratchBytes()`. Every operator is admitted
/// before anything is queued; production is refused until the target gates pass.
pub fn decode(o: *const Ops, s: cuda.Stream, use: Use, d: vae.Decoder, w: *const vae.DecoderWeights, rows: View, scratch: View, mel: View, p: DecodePolicy) !void {
    try decodeWith(o, s, use, d, w, rows, scratch, mel, p, null);
}

/// `decode` over any backend with the Ops launch methods (the tests' recorder checks the sequence and the dataflow).
fn decodeWith(o: anytype, s: anytype, use: Use, d: vae.Decoder, w: *const vae.DecoderWeights, rows: View, scratch: View, mel: View, p: DecodePolicy, hook: ?DecodeHook) !void {
    try d.admit(use, .prepared_native_reference);
    const sc = d.scratch();
    const cv = try Carver.init(scratch, sc.floats);
    try rows.expect(.cuda, &d.latentRowsShape());
    try mel.expect(.cuda, &d.melShape());
    const b: u64 = d.batch;
    var t: u64 = d.frames;
    // The native decoder (1 x 218 latent frames) runs every qualified path (QUALIFICATION.md): the fixed T218 kernels by
    // graph position here, the T436 ones by `tf32_contracts` (decoderConv). Any other geometry runs the direct ones.
    const native = qualifiedDecoder(d);
    const silu = ScalarDiv.of(vae.silu_divisor, p.silu);
    const mix = try vae.MpResidual.of(vae.residual_t, p.residual);
    const z = cv.at("k6a.decoder.z", sc.aux, &.{ b, vae.latent_channels, t });
    try o.latentUnscale(s, use, rows, z, .of(vae.scaling_factor, p.unscale), vae.mean_value);
    if (hook) |h| try h.op(h.ctx, "latent_unscale", z);
    var wi: usize = 0;
    // the residual block a following clamp belongs to (null after any other node)
    var last_block: ?[]const u8 = null;
    for (vae.decoder_graph) |node| switch (node) {
        .conv => |c| {
            const first = wi == 0;
            const x = if (first) z else cv.at("k6a.decoder.act0", sc.act[0], &.{ b, c.cin, t });
            const y = cv.at(if (first) "k6a.decoder.act0" else "k6a.decoder.act1", sc.act[if (first) 0 else 1], &.{ b, c.cout, t });
            try decoderConv(o, s, use, d, cv, sc, x, w.convs[wi], y, .{ .pad = c.k / 2 }, p.acc);
            if (hook) |h| try h.op(h.ctx, if (first) "first_conv" else "conv_out", y);
            last_block = null;
            wi += 1;
        },
        .resblock => |r| {
            const blk = blockName(wi);
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, r.cin, t });
            const xn = cv.at("k6a.decoder.act1", sc.act[1], &.{ b, r.cin, t });
            try o.normalizeActivation(s, use, x, cv.at("k6a.decoder.norms", sc.aux, &.{ b, t }), xn, 1, p.norm_residual.eps, p.norm_residual.reduce, p.norm_residual.den);
            try emitBlock(hook, blk, "norm", xn);
            const h_in = cv.at("k6a.decoder.act2", sc.act[2], &.{ b, r.cin, t });
            try o.scaledSilu(s, use, xn, h_in, silu);
            try emitBlock(hook, blk, "silu", h_in);
            const h1 = cv.at("k6a.decoder.act3", sc.act[3], &.{ b, r.cout, t });
            const tf32_scratch = cv.at("k6a.tf32.scratch", sc.conv_tf32, &.{vae.first_res_tf32_floats});
            // the T218 residual blocks: first, mid2 and up2's three (2048 ch), up1's three (2048 -> 1024, then 1024)
            const res2048 = native and (wi == 1 or wi == 5 or wi == 7 or wi == 9 or wi == 11);
            const up1_first = native and wi == 13;
            const up1_rest = native and (wi == 16 or wi == 18);
            if (res2048) {
                try o.resConvTf32(s, use, h_in, w.convs[wi], h1, tf32_scratch);
            } else if (up1_first) {
                try o.up1FirstResConvTf32(s, use, h_in, w.convs[wi], h1, tf32_scratch);
            } else if (up1_rest) {
                try o.up1SecondResConvTf32(s, use, h_in, w.convs[wi], h1, tf32_scratch);
            } else try decoderConv(o, s, use, d, cv, sc, h_in, w.convs[wi], h1, .{ .pad = 1 }, p.acc);
            try emitBlock(hook, blk, "res_conv1", h1);
            try o.scaledSilu(s, use, h1, h1, silu);
            try emitBlock(hook, blk, "res_silu", h1);
            const h2 = cv.at("k6a.decoder.act2", sc.act[2], &.{ b, r.cout, t });
            if (res2048) {
                try o.resConvTf32(s, use, h1, w.convs[wi + 1], h2, tf32_scratch);
            } else if (up1_first or up1_rest) {
                try o.up1SecondResConvTf32(s, use, h1, w.convs[wi + 1], h2, tf32_scratch);
            } else try decoderConv(o, s, use, d, cv, sc, h1, w.convs[wi + 1], h2, .{ .pad = 1 }, p.acc);
            try emitBlock(hook, blk, "res_conv2", h2);
            wi += 2;
            var shortcut = xn;
            if (r.cin != r.cout) {
                shortcut = cv.at("k6a.decoder.act4", sc.act[4], &.{ b, r.cout, t });
                if (native and wi == 15) {
                    try o.up1FirstResShortcutTf32(s, use, xn, w.convs[wi], shortcut, tf32_scratch);
                } else if (native and wi == 23) {
                    try o.up0FirstResShortcutTf32(s, use, xn, w.convs[wi], shortcut, tf32_scratch);
                } else try o.conv1d(s, use, xn, w.convs[wi], null, shortcut, .{ .pad = 0 }, p.acc);
                try emitBlock(hook, blk, "shortcut", shortcut);
                wi += 1;
            }
            try o.mpResidual(s, use, shortcut, h2, cv.at("k6a.decoder.act0", sc.act[0], &.{ b, r.cout, t }), mix);
            try emitBlock(hook, blk, "residual", cv.at("k6a.decoder.act0", sc.act[0], &.{ b, r.cout, t }));
            last_block = blk;
        },
        .attention => |a| {
            // the scratch regions are sized for the native graph's one attention (C 2048, at the latent length)
            if (a.c != 2048 or t != d.frames or a.c % a.heads != 0) return error.InvalidArgument;
            last_block = null;
            const hd = a.c / a.heads;
            const bh = b * a.heads;
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, a.c, t });
            if (native) {
                try o.attentionQkvTf32(s, use, x, w.convs[wi], cv.at("k6a.decoder.qkv", sc.qkv, &.{ b, 3 * a.c, t }), cv.at("k6a.qkv.tf32.scratch", sc.conv_tf32, &.{vae.first_res_tf32_floats}));
            } else {
                try o.conv1d(s, use, x, w.convs[wi], null, cv.at("k6a.decoder.qkv", sc.qkv, &.{ b, 3 * a.c, t }), .{ .pad = 0 }, p.acc);
            }
            const packed_qkv = cv.at("k6a.decoder.qkv", sc.qkv, &.{ b, a.heads, hd, 3, t });
            if (hook) |h| try h.op(h.ctx, "attention_qkv", packed_qkv);
            try o.normalizeActivation(s, use, packed_qkv, cv.at("k6a.decoder.norms", sc.aux, &.{ bh, 3 * t }), packed_qkv, 2, p.norm_qkv.eps, p.norm_qkv.reduce, p.norm_qkv.den);
            if (hook) |h| try h.op(h.ctx, "attention_norm", packed_qkv);
            const q = cv.at("k6a.decoder.q", sc.q, &.{ b, a.heads, t, hd });
            const k = cv.at("k6a.decoder.k", sc.k, &.{ b, a.heads, t, hd });
            const v = cv.at("k6a.decoder.v", sc.v, &.{ b, a.heads, t, hd });
            try o.qkvSplit(s, use, packed_qkv, q, k, v);
            if (native) {
                try o.midAttentionTf32(s, use, cv.at("k6a.decoder.q", sc.q, &.{ bh, t, hd }), cv.at("k6a.decoder.k", sc.k, &.{ bh, t, hd }), cv.at("k6a.decoder.v", sc.v, &.{ bh, t, hd }), cv.at("k6a.decoder.attn_out", sc.out, &.{ bh, t, hd }), cv.at("k6a.sdpa.tf32.scratch", sc.conv_tf32, &.{vae.first_res_tf32_floats}));
            } else switch (p.attention) {
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
            if (hook) |h| try h.op(h.ctx, "attention_out", cv.at("k6a.decoder.attn_out", sc.out, &.{ bh, t, hd }));
            const merged = cv.at("k6a.decoder.act1", sc.act[1], &.{ b, a.c, t });
            try o.attnMerge(s, use, cv.at("k6a.decoder.attn_out", sc.out, &.{ b, a.heads, t, hd }), merged);
            if (hook) |h| try h.op(h.ctx, "attention_merged", merged);
            const h = cv.at("k6a.decoder.act2", sc.act[2], &.{ b, a.c, t });
            if (native) {
                try o.attentionProjTf32(s, use, merged, w.convs[wi + 1], h, cv.at("k6a.proj.tf32.scratch", sc.conv_tf32, &.{vae.first_res_tf32_floats}));
            } else {
                try o.conv1d(s, use, merged, w.convs[wi + 1], null, h, .{ .pad = 0 }, p.acc);
            }
            if (hook) |hk| try hk.op(hk.ctx, "attention_proj", h);
            wi += 2;
            try o.mpResidual(s, use, x, h, x, mix);
            if (hook) |hk| try hk.op(hk.ctx, "attention_residual", x);
        },
        .clamp => {
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, try decoderChannels(wi), t });
            try o.clamp(s, use, x, x, -vae.clip_act, vae.clip_act);
            try emitBlock(hook, last_block, "clamp", x);
        },
        .upsample => |c| {
            const level = upsampleName(wi);
            const up = cv.at("k6a.decoder.act1", sc.act[1], &.{ b, c.cin, 2 * t });
            try o.upsampleNearestExact2(s, use, cv.at("k6a.decoder.act0", sc.act[0], &.{ b, c.cin, t }), up);
            try emitBlock(hook, level, "upsample", up);
            t *= 2;
            const y = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, c.cout, t });
            try decoderConv(o, s, use, d, cv, sc, up, w.convs[wi], y, .{ .pad = c.k / 2 }, p.acc);
            try emitBlock(hook, level, "upsample_conv", y);
            last_block = null;
            wi += 1;
        },
        .silu => {
            const x = cv.at("k6a.decoder.act0", sc.act[0], &.{ b, try decoderChannels(wi), t });
            try o.scaledSilu(s, use, x, x, silu);
            if (hook) |h| try h.op(h.ctx, "final_silu", x);
            last_block = null;
        },
        .unnormalize => {
            try o.melAffine(s, use, cv.at("k6a.decoder.act1", sc.act[1], &.{ b, vae.mel_channels, t }), w.data_std, w.data_mean, mel);
            if (hook) |h| try h.op(h.ctx, "mel", mel);
        },
    };
    if (wi != w.convs.len) return error.InvalidArgument;
}

/// Diagnostic hook prefixes, keyed on each node's graph position: a residual block by the conv index `wi` where it
/// starts (its norm, silu, convs, shortcut, residual and following clamp all take the block's name, never the advanced
/// `wi`), the upsample by its conv index. Names follow the upstream Decoder1D modules: mid.block_1 `first`, mid.block_2
/// `mid2`, up[L].block[i] `upL_{first,second,third}`, up[L].upsample `upL`. The twin records the same names
/// (tools/twin/stk_twin/kandinsky6/audio_trace.py); every hook observes a tensor already written, nothing else changes.
pub const decoder_blocks = [_]struct { wi: usize, name: []const u8 }{
    .{ .wi = 1, .name = "first" },
    .{ .wi = 5, .name = "mid2" },
    .{ .wi = 7, .name = "up2_first" },
    .{ .wi = 9, .name = "up2_second" },
    .{ .wi = 11, .name = "up2_third" },
    .{ .wi = 13, .name = "up1_first" },
    .{ .wi = 16, .name = "up1_second" },
    .{ .wi = 18, .name = "up1_third" },
    .{ .wi = 21, .name = "up0_first" },
    .{ .wi = 24, .name = "up0_second" },
    .{ .wi = 26, .name = "up0_third" },
};
pub const decoder_upsamples = [_]struct { wi: usize, name: []const u8 }{.{ .wi = 20, .name = "up1" }};

fn blockName(wi: usize) ?[]const u8 {
    for (decoder_blocks) |e| if (e.wi == wi) return e.name;
    return null;
}

fn upsampleName(wi: usize) ?[]const u8 {
    for (decoder_upsamples) |e| if (e.wi == wi) return e.name;
    return null;
}

/// `<prefix>_<suffix>` to the hook, when there is a hook and the node has a name.
fn emitBlock(hook: ?DecodeHook, prefix: ?[]const u8, suffix: []const u8, v: View) !void {
    const h = hook orelse return;
    const pre = prefix orelse return;
    var nb: [64]u8 = undefined;
    try h.op(h.ctx, try std.fmt.bufPrint(&nb, "{s}_{s}", .{ pre, suffix }), v);
}

/// The geometry the audio decoder's qualified paths are proven at (QUALIFICATION.md): batch 1, the native 218 latent frames.
fn qualifiedDecoder(d: vae.Decoder) bool {
    return d.batch == 1 and d.frames == vae.native_latent_frames;
}

/// The geometry the vocoder's qualified paths are proven at: batch 1, 436 mel frames (the only vocoder with a `conv_w`
/// region).
fn qualifiedVocoder(v: vocoder.Vocoder) bool {
    return v.batch == 1 and v.frames == vocoder.tf32_frames;
}

/// A decoder convolution: on the native decoder, through `convTf32` when its shape is a listed contract (the T436
/// convolutions: up1's upsample conv, up0's six, conv_out), in the decoder's accounted conv_tf32 region (input, weight,
/// output copies from its start; no growth); else the direct conv1d with `acc`.
fn decoderConv(o: anytype, s: anytype, use: Use, d: vae.Decoder, cv: Carver, sc: vae.Decoder.Scratch, x: View, w: Weight, y: View, g: Ops.ConvGeom, acc: Acc) !void {
    const listed = qualifiedDecoder(d) and g.stride == 1 and tf32Contract(x.shape[1], w.view.shape[0], x.shape[2], w.view.shape[2], g.pad, g.dil) != null;
    if (!listed) return o.conv1d(s, use, x, w, null, y, g, acc);
    const nin = x.shape[1] * x.shape[2];
    const nw = w.view.shape[0] * w.view.shape[1] * w.view.shape[2];
    const nout = y.shape[1] * y.shape[2];
    if (nin + nw + nout > vae.first_res_tf32_floats) return error.OutOfBounds;
    try o.convTf32(s, use, x, w, null, y, g, .{
        .input = cv.at("k6a.tf32.input", sc.conv_tf32, &.{nin}),
        .weight = cv.at("k6a.tf32.weight", sc.conv_tf32 + nin, &.{nw}),
        .output = cv.at("k6a.tf32.output", sc.conv_tf32 + nin + nw, &.{nout}),
    });
}

/// The channel count of the decoder activation after the first `wi` convolutions: the last one's Cout.
fn decoderChannels(wi: usize) Error!u64 {
    if (wi == 0) return error.InvalidArgument;
    return vae.decoder_convs[wi - 1].cout;
}

/// The vocoder's numeric choices, named; none has a default (see DecodePolicy).
pub const VocodePolicy = struct {
    /// Accumulation of every vocoder convolution with no qualified path: on the 1 x 436 vocoder the 19 whose originals ran
    /// cuDNN's legacy FP32 implicit_convolve_sgemm (stage 4 kernel 3, stage 5 kernels 3 and 7, conv_post: exact as
    /// `.fma`'s order, channel outer, taps inner, then the bias add); every convolution of any other geometry. The
    /// other 97 always take their contract (`tf32_contracts` / `gather_contracts`). Qualified: `.fma`.
    conv_acc: Acc,
    /// Activation1d's anti-aliasing FIRs (native, D3 window: `.fma`, all stages).
    aa_acc: Acc,
    /// The stage merge's division by num_kernels (3): ATen multiplies by 1 / 3 for a CPU-scalar divisor. Qualified:
    /// `.reciprocal` (audio-w8-semantics-v1, all six stages).
    avg: Div,
};

/// BigVGAN v2: `mel` [B, 128, F] -> `wave` [B, 1, 512F] clamped to [-1, 1], with the weights from
/// `vocoder.bindVocoder` and `scratch` a rank-1 `.cuda` view of at least `v.scratchBytes()`.
pub fn vocode(o: *const Ops, s: cuda.Stream, use: Use, v: vocoder.Vocoder, w: *const vocoder.VocoderWeights, mel: View, scratch: View, wave: View, p: VocodePolicy) !void {
    try vocodeWith(o, s, use, v, w, mel, scratch, wave, p, null);
}

/// `vocode` with a component diagnostic observer: `conv_pre`, `upsample<i>`, `stage<i>_amp<j>`, `stage<i>_avg`,
/// `post_act`, `conv_post`, `wave`, each the tensor just written (the twin's BigVGAN hooks record the same names).
pub fn vocodeTraced(o: *const Ops, s: cuda.Stream, use: Use, v: vocoder.Vocoder, w: *const vocoder.VocoderWeights, mel: View, scratch: View, wave: View, p: VocodePolicy, hook: ?DecodeHook) !void {
    try vocodeWith(o, s, use, v, w, mel, scratch, wave, p, hook);
}

/// One indexed vocoder stage name to the hook.
fn emitStage(hook: ?DecodeHook, comptime fmt: []const u8, args: anytype, v: View) !void {
    const h = hook orelse return;
    var nb: [32]u8 = undefined;
    try h.op(h.ctx, try std.fmt.bufPrint(&nb, fmt, args), v);
}

fn vocodeWith(o: anytype, s: anytype, use: Use, v: vocoder.Vocoder, w: *const vocoder.VocoderWeights, mel: View, scratch: View, wave: View, p: VocodePolicy, hook: ?DecodeHook) !void {
    try v.admit(use, .checkpoint_effective);
    const sc = v.scratch();
    const cv = try Carver.init(scratch, sc.floats);
    try mel.expect(.cuda, &v.melShape());
    try wave.expect(.cuda, &v.waveShape());
    const b: u64 = v.batch;
    var t: u64 = v.frames;
    var c: u64 = vocoder.upsample_initial_channel;
    const work = [2][]const u8{ "k6a.vocoder.work0", "k6a.vocoder.work1" };
    // the 1 x 436 vocoder takes every qualified path (QUALIFICATION.md); any other geometry the direct ones
    const native = qualifiedVocoder(v);
    try vocoderConv(o, s, use, cv, sc, native, mel, w.conv_pre.w, w.conv_pre.bias, cv.at(work[0], sc.work[0], &.{ b, c, t }), .{ .pad = 3 }, p.conv_acc);
    if (hook) |h| try h.op(h.ctx, "conv_pre", cv.at(work[0], sc.work[0], &.{ b, c, t }));
    for (vocoder.upsample_rates, vocoder.upsample_kernel_sizes, 0..) |u, uk, i| {
        const ch: u64 = vocoder.stageChannels(i);
        const stage = cv.at("k6a.vocoder.stage", sc.stage, &.{ b, ch, t * u });
        try vocoderConvTranspose(o, s, use, cv, sc, native, cv.at(work[0], sc.work[0], &.{ b, c, t }), w.ups[i].w, w.ups[i].bias, stage, u, (uk - u) / 2, p.conv_acc);
        try emitStage(hook, "upsample{d}", .{i}, stage);
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
                try vocoderConv(o, s, use, cv, sc, native, w0, blk.convs1[n].w, blk.convs1[n].bias, w1, .{ .pad = vocoder.samePadding(rk, dil), .dil = dil }, p.conv_acc);
                try activation1d(o, s, use, cv, sc, w1, blk.acts[2 * n + 1], w0, p.aa_acc);
                try vocoderConv(o, s, use, cv, sc, native, w0, blk.convs2[n].w, blk.convs2[n].bias, w1, .{ .pad = vocoder.samePadding(rk, 1) }, p.conv_acc);
                try o.add(s, use, w1, x, amp[j]);
            }
            try emitStage(hook, "stage{d}_amp{d}", .{ i, j }, amp[j]);
        }
        try o.avg3(s, use, amp, cv.at(work[0], sc.work[0], &.{ b, c, t }), .of(@floatFromInt(vocoder.resblock_kernel_sizes.len), p.avg));
        try emitStage(hook, "stage{d}_avg", .{i}, cv.at(work[0], sc.work[0], &.{ b, c, t }));
    }
    const post = cv.at(work[1], sc.work[1], &.{ b, c, t });
    try activation1d(o, s, use, cv, sc, cv.at(work[0], sc.work[0], &.{ b, c, t }), w.post, post, p.aa_acc);
    if (hook) |h| try h.op(h.ctx, "post_act", post);
    const y = cv.at(work[0], sc.work[0], &.{ b, 1, t });
    try o.conv1d(s, use, post, w.conv_post.w, null, y, .{ .pad = 3 }, p.conv_acc);
    if (hook) |h| try h.op(h.ctx, "conv_post", y);
    try o.clamp(s, use, y, wave, -vocoder.final_clamp, vocoder.final_clamp);
    if (hook) |h| try h.op(h.ctx, "wave", wave);
}

/// A vocoder convolution: on the qualified vocoder (`native`), through `convGather` when its shape is a listed fprop
/// `gather_contracts` entry, else through `convTf32` when it is a listed `tf32_contracts` entry (the two are disjoint);
/// otherwise, and on any other geometry, the direct conv1d. The NTC input / output copies use the Activation1d
/// intermediate `up2` (dead while a convolution runs: it is rewritten before its next read), the KSC weight copy the
/// vocoder's `conv_w` region.
fn vocoderConv(o: anytype, s: anytype, use: Use, cv: Carver, sc: vocoder.Vocoder.Scratch, native: bool, x: View, w: Weight, bias: ?View, y: View, g: Ops.ConvGeom, acc: Acc) !void {
    if (native and g.stride == 1 and gatherContract(.fprop, x.shape[1], w.view.shape[0], x.shape[2], w.view.shape[2], g.pad, g.dil, 1) != null)
        return vocoderGather(o, s, use, cv, sc, x, w, bias, y, .{ .kind = .fprop, .pad = g.pad, .dil = g.dil });
    const listed = native and g.stride == 1 and tf32Contract(x.shape[1], w.view.shape[0], x.shape[2], w.view.shape[2], g.pad, g.dil) != null;
    if (!listed) return o.conv1d(s, use, x, w, bias, y, g, acc);
    const nin = x.shape[1] * x.shape[2];
    const nw = w.view.shape[0] * w.view.shape[1] * w.view.shape[2];
    const nout = y.shape[1] * y.shape[2];
    if (nin > sc.slot or nout > sc.slot or nw > sc.conv_w_floats) return error.OutOfBounds;
    try o.convTf32(s, use, x, w, bias, y, g, .{
        .input = cv.at("k6a.vocoder.tf32.input", sc.up2, &.{nin}),
        .weight = cv.at("k6a.vocoder.tf32.weight", sc.conv_w, &.{nw}),
        .output = cv.at("k6a.vocoder.tf32.output", sc.up2 + sc.slot, &.{nout}),
    });
}

/// A vocoder upsampler: on the qualified vocoder, through `convGather` when its shape is a listed dgrad
/// `gather_contracts` entry (all six upsamplers), else the direct transposed convolution.
fn vocoderConvTranspose(o: anytype, s: anytype, use: Use, cv: Carver, sc: vocoder.Vocoder.Scratch, native: bool, x: View, w: Weight, bias: ?View, y: View, stride: u32, pad: u32, acc: Acc) !void {
    if (native and gatherContract(.dgrad, x.shape[1], w.view.shape[1], x.shape[2], w.view.shape[2], pad, 1, stride) != null)
        return vocoderGather(o, s, use, cv, sc, x, w, bias, y, .{ .kind = .dgrad, .pad = pad, .stride = stride });
    try o.convTranspose1d(s, use, x, w, bias, y, stride, pad, acc);
}

/// `convGather` with the vocoder's regions: B in `conv_w` (every listed slice's B fits it), the row chunks' A, D and
/// partial in the 2-slot `up2` (dead while a convolution runs, as for `convTf32`'s copies).
fn vocoderGather(o: anytype, s: anytype, use: Use, cv: Carver, sc: vocoder.Vocoder.Scratch, x: View, w: Weight, bias: ?View, y: View, g: GatherGeom) !void {
    if (comptime !@hasDecl(@typeInfo(@TypeOf(o)).pointer.child, "convGather")) @compileError("vocode backend lacks pub fn convGather");
    try o.convGather(s, use, x, w, bias, y, g, .{
        .weight = cv.at("k6a.vocoder.gather.weight", sc.conv_w, &.{sc.conv_w_floats}),
        .rows = cv.at("k6a.vocoder.gather.rows", sc.up2, &.{2 * sc.slot}),
    });
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
    /// convTf32 calls.
    tf32_convs: usize = 0,
    /// up1FirstResShortcutTf32 calls, and the x / y / scratch offsets of the last one.
    shortcut_tf32: usize = 0,
    shortcut_at: [3]u64 = .{ 0, 0, 0 },
    /// up0FirstResShortcutTf32 calls, and the x / y / scratch offsets of the last one.
    shortcut0_tf32: usize = 0,
    shortcut0_at: [3]u64 = .{ 0, 0, 0 },
    /// convGather calls per GatherKind (fprop, dgrad).
    gather_convs: [2]usize = .{ 0, 0 },
    /// resConvTf32 / up1FirstResConvTf32 / up1SecondResConvTf32 calls (the fixed T218 residual convolutions).
    fixed_convs: usize = 0,
    /// attentionQkvTf32 / midAttentionTf32 / attentionProjTf32 calls.
    attention_tf32: usize = 0,

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
    /// The fixed T218 kernels' checks (Ops.resConvTf32 / up1*ResConvTf32 / attentionQkvTf32 / attentionProjTf32): the
    /// shapes they accept, the scratch copies they carve (input, weight and, with `out_copy`, output), 16-byte alignment,
    /// no copy overlapping an operand and (`pairwise`) the operands disjoint; then recorded as the direct convolution
    /// they replace.
    fn fixedConv(r: *Recorder, use: Use, x: View, w: Weight, y: View, scratch: View, ci: u64, co: u64, k: u64, region: u64, out_copy: bool, pairwise: bool) !void {
        try x.expect(.cuda, &.{ 1, ci, 218 });
        try w.view.expect(.cuda, &.{ co, ci, k });
        try y.expect(.cuda, &.{ 1, co, 218 });
        const cv = try Carver.init(scratch, region);
        const nw = co * ci * k;
        const all = [_]View{ cv.at("in", 0, &.{ci * 218}), cv.at("w", ci * 218, &.{nw}), cv.at("out", ci * 218 + nw, &.{co * 218}) };
        const operands = [_]View{ x, w.view, y };
        for (operands) |v| if (v.addr() % 16 != 0) return error.LayoutUnsupported;
        for (all[0..if (out_copy) 3 else 2]) |c| {
            if (c.addr() % 16 != 0) return error.LayoutUnsupported;
            for (operands) |src| if (overlaps(c, src)) return error.LayoutUnsupported;
        }
        if (pairwise and (overlaps(x, w.view) or overlaps(x, y) or overlaps(w.view, y))) return error.LayoutUnsupported;
        try r.conv1d({}, use, x, w, null, y, .{ .pad = @intCast(k / 2) }, .fma);
    }
    fn resConvTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        r.fixed_convs += 1;
        try r.fixedConv(use, x, w, y, scratch, 2048, 2048, 3, vae.first_res_tf32_floats, true, true);
    }
    fn up1FirstResConvTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        r.fixed_convs += 1;
        try r.fixedConv(use, x, w, y, scratch, 2048, 1024, 3, vae.first_res_tf32_floats, true, true);
    }
    fn up1SecondResConvTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        r.fixed_convs += 1;
        try r.fixedConv(use, x, w, y, scratch, 1024, 1024, 3, vae.first_res_tf32_floats, true, true);
    }
    fn attentionQkvTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        r.attention_tf32 += 1;
        try r.fixedConv(use, x, w, y, scratch, 2048, 6144, 1, 2048 * 218 + 6144 * 2048, false, false);
    }
    fn attentionProjTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        r.attention_tf32 += 1;
        try r.fixedConv(use, x, w, y, scratch, 2048, 2048, 1, 2048 * 218 + 2048 * 2048, false, true);
    }
    /// Ops.midAttentionTf32's checks (shapes, its seven scratch copies 16-byte aligned and clear of q, k, v and out, out
    /// clear of q, k and v), recorded as the SDPA it replaces (its logits / probs stay in its own region).
    fn midAttentionTf32(r: *Recorder, _: void, use: Use, q: View, k: View, v: View, out: View, scratch: View) !void {
        for ([_]View{ q, k, v, out }) |x| try x.expect(.cuda, &.{ 1, 218, 2048 });
        const count: u64 = 446464;
        const scores: u64 = 47524;
        const cv = try Carver.init(scratch, 4 * count + 14 * scores);
        const copies = [_]View{
            cv.at("q", 0, &.{count}),
            cv.at("k", count, &.{count}),
            cv.at("v", 2 * count, &.{count}),
            cv.at("logits", 3 * count, &.{scores}),
            cv.at("probs", 3 * count + scores, &.{scores}),
            cv.at("partitions", 3 * count + 2 * scores, &.{12 * scores}),
            cv.at("value", 3 * count + 14 * scores, &.{count}),
        };
        for ([_]View{ q, k, v, out }) |x| if (x.addr() % 16 != 0) return error.LayoutUnsupported;
        for (copies) |c| {
            if (c.addr() % 16 != 0) return error.LayoutUnsupported;
            for ([_]View{ q, k, v, out }) |src| if (overlaps(c, src)) return error.LayoutUnsupported;
        }
        for ([_]View{ q, k, v }) |src| if (overlaps(out, src)) return error.LayoutUnsupported;
        r.attention_tf32 += 1;
        try r.read(k);
        try r.read(v);
        try r.unary(.mid_attention_sdpa, use, q, out);
    }
    /// The same layout checks as Ops.convTf32, then recorded as the direct convolution it replaces (same operator, same
    /// dataflow).
    pub fn convTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, bias: ?View, y: View, g: Ops.ConvGeom, sc: Tf32Scratch) !void {
        _ = try Tf32Layout.plan(x, w.view, bias, y, g, sc);
        r.tf32_convs += 1;
        try r.conv1d({}, use, x, w, bias, y, g, .fma);
    }
    pub fn up1FirstResShortcutTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try w.requirePrepared(use);
        try r.read(w.view);
        _ = try Up1Shortcut.carve(x, w.view, y, scratch);
        r.shortcut_tf32 += 1;
        r.shortcut_at = .{ x.offset, y.offset, scratch.offset };
        try r.unary(.conv1d_prepared, use, x, y);
    }
    pub fn up0FirstResShortcutTf32(r: *Recorder, _: void, use: Use, x: View, w: Weight, y: View, scratch: View) !void {
        try w.requirePrepared(use);
        try r.read(w.view);
        _ = try Up0Shortcut.carve(x, w.view, y, scratch);
        r.shortcut0_tf32 += 1;
        r.shortcut0_at = .{ x.offset, y.offset, scratch.offset };
        try r.unary(.conv1d_prepared, use, x, y);
    }
    /// pub (vocoderGather checks it with @hasDecl). The same layout checks as Ops.convGather, then recorded as the
    /// convolution or transposed convolution it replaces (same operator, same dataflow).
    pub fn convGather(r: *Recorder, _: void, use: Use, x: View, w: Weight, bias: ?View, y: View, g: GatherGeom, sc: GatherScratch) !void {
        _ = try GatherLayout.plan(x, w.view, bias, y, g, sc);
        r.gather_convs[@backingInt(g.kind)] += 1;
        switch (g.kind) {
            .fprop => try r.conv1d({}, use, x, w, bias, y, .{ .pad = g.pad, .dil = g.dil }, .fma),
            .dgrad => try r.convTranspose1d({}, use, x, w, bias, y, g.stride, g.pad, .fma),
        }
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
    try decodeWith(&r, {}, .diagnostic, d, &w, testView("rows", 201 << 40, &d.latentRowsShape()), scratch, mel, policy, null);

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
    try testing.expectError(error.GpuQualificationPending, decodeWith(&p, {}, .production, d, &w, testView("rows", 201 << 40, &d.latentRowsShape()), scratch, mel, policy, null));
    try testing.expectEqual(0, p.n_ops);
    // a scratch view smaller than scratchBytes is refused
    try testing.expectError(error.OutOfBounds, decodeWith(&p, {}, .diagnostic, d, &w, testView("rows", 201 << 40, &d.latentRowsShape()), testView("scratch", r.scratch_base, &.{d.scratch().floats - 1}), mel, policy, null));
}

test "up1 shortcut TF32: buffers fit the accounted convolution region and every guard refuses a bad layout" {
    const region = vae.first_res_tf32_floats;
    const base: u64 = 300 << 40;
    const scratch = testView("scratch", base, &.{region});
    const x = testView("x", 301 << 40, &.{ 1, 2048, 218 });
    const w = testView("w", 302 << 40, &.{ 1024, 2048, 1 });
    const y = testView("y", 303 << 40, &.{ 1, 1024, 218 });
    const l = try Up1Shortcut.carve(x, w, y, scratch);
    try testing.expectEqual(0, l.a.offset);
    try testing.expectEqual(446464, l.b.offset);
    try testing.expectEqual(2543616, l.work.offset);
    // the split-K workspace: 3 x 1024 x 218 floats = 2,678,784 bytes, ending inside the 53,903,360-byte region
    try testing.expectEqual(2678784, l.work.bytes());
    try testing.expectEqual(53903360, region * 4);
    try testing.expect((l.work.offset + l.work.numel()) * 4 <= region * 4);
    // shapes, device and region size
    try testing.expectError(error.ShapeMismatch, Up1Shortcut.carve(x, testView("w", 302 << 40, &.{ 1024, 2048, 3 }), y, scratch));
    try testing.expectError(error.ShapeMismatch, Up1Shortcut.carve(x, w, testView("y", 303 << 40, &.{ 1, 2048, 218 }), scratch));
    try testing.expectError(error.DeviceMismatch, Up1Shortcut.carve(.contiguous("x", .host, 301 << 40, 446464 * 4, .borrowed, &.{ 1, 2048, 218 }), w, y, scratch));
    try testing.expectError(error.OutOfBounds, Up1Shortcut.carve(x, w, y, testView("scratch", base, &.{region - 1})));
    // a scratch view 4 bytes off 16-byte alignment
    var shifted = View.contiguous("scratch", .cuda, base, (region + 1) * 4, .borrowed, &.{region});
    shifted.offset = 1;
    try testing.expectError(error.LayoutUnsupported, Up1Shortcut.carve(x, w, y, shifted));
    // an operand inside the region the copies or the partitions use
    const inside = View.contiguous("x", .cuda, base, region * 4, .borrowed, &.{ 1, 2048, 218 });
    try testing.expectError(error.LayoutUnsupported, Up1Shortcut.carve(inside, w, y, scratch));
    var tail = View.contiguous("y", .cuda, base, region * 4, .borrowed, &.{ 1, 1024, 218 });
    tail.offset = 2543616;
    try testing.expectError(error.LayoutUnsupported, Up1Shortcut.carve(x, w, tail, scratch));
    // the output aliasing the input
    try testing.expectError(error.LayoutUnsupported, Up1Shortcut.carve(x, w, .contiguous("y", .cuda, 301 << 40, 446464 * 4, .borrowed, &.{ 1, 1024, 218 }), scratch));
}

test "decode: the native 1x218 decoder takes every qualified path; any other geometry the direct ones, the same operators in order" {
    const policy: DecodePolicy = .{ .acc = .fma, .unscale = .reciprocal, .silu = .reciprocal, .residual = .reciprocal, .norm_residual = norm_policy, .norm_qkv = norm_policy, .attention = .math_candidate };
    var w: vae.DecoderWeights = undefined;
    for (vae.decoder_convs, &w.convs, 0..) |c, *cw, i| {
        cw.* = .oraclePrepared(testView(c.name, (i + 1) << 40, &.{ c.cout, c.cin, c.k }), c.name, @splat(0));
    }
    w.gain = 1.0;
    w.data_std = testView("std", 100 << 40, &.{vae.mel_channels});
    w.data_mean = testView("mean", 101 << 40, &.{vae.mel_channels});
    const d = try vae.Decoder.init(1, vae.native_latent_frames);
    const sc = d.scratch();
    var r: Recorder = .{ .scratch_base = 200 << 40 };
    try decodeWith(&r, {}, .diagnostic, d, &w, testView("rows", 201 << 40, &d.latentRowsShape()), testView("scratch", r.scratch_base, &.{sc.floats}), testView("mel", 202 << 40, &d.melShape()), policy, null);
    // QUALIFICATION.md's decoder: the fixed T218 residual convolutions (first, mid2, up2's three: 2 each; up1's three:
    // 2 each), the attention's QKV, SDPA and projection, the T436 contracts (up1's upsample conv, up0's six, conv_out)
    // and both nin shortcuts
    try testing.expectEqual(10 + 6, r.fixed_convs);
    try testing.expectEqual(3, r.attention_tf32);
    try testing.expectEqual(8, r.tf32_convs);
    try testing.expectEqual(1, r.shortcut_tf32);
    try testing.expectEqual(1, r.shortcut0_tf32);
    // x = the normalized block input (act1), y = the shortcut slot (act4), scratch = the accounted convolution region
    try testing.expectEqual([3]u64{ sc.act[1], sc.act[4], sc.conv_tf32 }, r.shortcut_at);
    try testing.expectEqual([3]u64{ sc.act[1], sc.act[4], sc.conv_tf32 }, r.shortcut0_at);
    // the region lies past every activation and attention tensor, so no buffer of the wrappers aliases a live tensor
    try testing.expect(sc.act[4] + sc.slot <= sc.aux and sc.scores + 218 * 218 == sc.conv_tf32);
    try testing.expect(sc.conv_tf32 + Up1Shortcut.floats <= sc.floats and sc.conv_tf32 + Up0Shortcut.floats <= sc.floats);
    try testing.expectEqual(vae.first_res_tf32_floats, sc.floats - sc.conv_tf32);
    // any other geometry: none of them, and the same operators in the same order
    const small = try vae.Decoder.init(2, 3);
    var o: Recorder = .{ .scratch_base = 200 << 40 };
    try decodeWith(&o, {}, .diagnostic, small, &w, testView("rows", 201 << 40, &small.latentRowsShape()), testView("scratch", o.scratch_base, &.{small.scratch().floats}), testView("mel", 202 << 40, &small.melShape()), policy, null);
    try testing.expectEqual(0, o.fixed_convs + o.attention_tf32 + o.tf32_convs + o.shortcut_tf32 + o.shortcut0_tf32);
    try testing.expectEqualSlices(vae.Op, o.ops[0..o.n_ops], r.ops[0..r.n_ops]);
}

test "up0 shortcut TF32: its buffers fit the accounted region and every guard refuses a bad layout" {
    // the graph position decodeWith keys on: conv 23 is up.0.block.0's nin shortcut, 1024 to 512, kernel 1
    try testing.expect(std.mem.endsWith(u8, vae.decoder_convs[23].name, ".nin_shortcut.weight"));
    try testing.expectEqual(1024, vae.decoder_convs[23].cin);
    try testing.expectEqual(512, vae.decoder_convs[23].cout);
    try testing.expectEqual(1, vae.decoder_convs[23].k);
    const region = vae.first_res_tf32_floats;
    const scratch = testView("scratch", 300 << 40, &.{region});
    const x = testView("x", 301 << 40, &.{ 1, 1024, 436 });
    const wv = testView("w", 302 << 40, &.{ 512, 1024, 1 });
    const yv = testView("y", 303 << 40, &.{ 1, 512, 436 });
    const l = try Up0Shortcut.carve(x, wv, yv, scratch);
    try testing.expectEqual(0, l.a.offset);
    try testing.expectEqual(446464, l.b.offset);
    try testing.expectEqual(446464 + 524288, l.work.offset);
    // the split-K workspace: 3 x 512 x 436 floats = 2,678,784 bytes (the probe's reported workspace)
    try testing.expectEqual(2678784, l.work.bytes());
    try testing.expect(l.work.offset + l.work.numel() <= region);
    try testing.expectError(error.ShapeMismatch, Up0Shortcut.carve(testView("x", 301 << 40, &.{ 1, 2048, 218 }), wv, yv, scratch));
    try testing.expectError(error.ShapeMismatch, Up0Shortcut.carve(x, testView("w", 302 << 40, &.{ 1024, 2048, 1 }), yv, scratch));
    try testing.expectError(error.LayoutUnsupported, Up0Shortcut.carve(x, wv, .contiguous("y", .cuda, 301 << 40, 446464 * 4, .borrowed, &.{ 1, 512, 436 }), scratch));
    try testing.expectError(error.LayoutUnsupported, Up0Shortcut.carve(.contiguous("x", .cuda, 300 << 40, region * 4, .borrowed, &.{ 1, 1024, 436 }), wv, yv, scratch));
}

/// A DecodeHook that keeps a copy of every name it is given, in order.
const NameLog = struct {
    names: [128][64]u8 = undefined,
    lens: [128]usize = undefined,
    n: usize = 0,

    fn op(ctx: *anyopaque, name: []const u8, out: View) anyerror!void {
        const l: *NameLog = @ptrCast(@alignCast(ctx));
        try out.validate();
        @memcpy(l.names[l.n][0..name.len], name);
        l.lens[l.n] = name.len;
        l.n += 1;
    }

    fn at(l: *const NameLog, i: usize) []const u8 {
        return l.names[i][0..l.lens[i]];
    }

    fn hook(l: *NameLog) DecodeHook {
        return .{ .ctx = l, .op = op };
    }

    fn expectUnique(l: *const NameLog) !void {
        for (0..l.n) |i| for (i + 1..l.n) |j| try testing.expect(!std.mem.eql(u8, l.at(i), l.at(j)));
    }
};

test "decoder hook names: one table entry per residual block and upsample, at its graph position" {
    var wi: usize = 0;
    var k: usize = 0;
    var u: usize = 0;
    for (vae.decoder_graph) |node| switch (node) {
        .conv => wi += 1,
        .resblock => |r| {
            try testing.expectEqual(decoder_blocks[k].wi, wi);
            k += 1;
            wi += if (r.cin != r.cout) 3 else 2;
        },
        .attention => wi += 2,
        .upsample => {
            try testing.expectEqual(decoder_upsamples[u].wi, wi);
            u += 1;
            wi += 1;
        },
        else => {},
    };
    try testing.expectEqual(decoder_blocks.len, k);
    try testing.expectEqual(decoder_upsamples.len, u);
    for (decoder_blocks, 0..) |a, i| for (decoder_blocks[i + 1 ..]) |c| try testing.expect(a.wi != c.wi and !std.mem.eql(u8, a.name, c.name));
}

test "decode hooks: reference 11's 49 names in order, then every remaining op through mel; no other change" {
    const d = try vae.Decoder.init(2, 3);
    var w: vae.DecoderWeights = undefined;
    for (vae.decoder_convs, &w.convs, 0..) |c, *cw, i| {
        cw.* = .oraclePrepared(testView(c.name, (i + 1) << 40, &.{ c.cout, c.cin, c.k }), c.name, @splat(0));
    }
    w.gain = 1.0;
    w.data_std = testView("std", 100 << 40, &.{vae.mel_channels});
    w.data_mean = testView("mean", 101 << 40, &.{vae.mel_channels});
    const policy: DecodePolicy = .{ .acc = .fma, .unscale = .reciprocal, .silu = .true_div, .residual = .true_div, .norm_residual = norm_policy, .norm_qkv = norm_policy, .attention = .math_candidate };
    const rows = testView("rows", 201 << 40, &d.latentRowsShape());
    const mel = testView("mel", 202 << 40, &d.melShape());
    var plain: Recorder = .{ .scratch_base = 200 << 40 };
    try decodeWith(&plain, {}, .diagnostic, d, &w, rows, testView("scratch", plain.scratch_base, &.{d.scratch().floats}), mel, policy, null);
    var log: NameLog = .{};
    var r: Recorder = .{ .scratch_base = 200 << 40 };
    try decodeWith(&r, {}, .diagnostic, d, &w, rows, testView("scratch", r.scratch_base, &.{d.scratch().floats}), mel, policy, log.hook());
    // observing changes no operator, order or buffer
    try testing.expectEqualSlices(vae.Op, plain.ops[0..plain.n_ops], r.ops[0..r.n_ops]);
    try testing.expectEqual(plain.n_writes, r.n_writes);
    const reference_11 = [_][]const u8{
        "latent_unscale",
        "first_conv",
        "first_norm",
        "first_silu",
        "first_res_conv1",
        "first_res_silu",
        "first_res_conv2",
        "first_residual",
        "attention_qkv",
        "attention_norm",
        "attention_out",
        "attention_merged",
        "attention_proj",
        "attention_residual",
        "mid2_norm",
        "mid2_silu",
        "mid2_res_conv1",
        "mid2_res_silu",
        "mid2_res_conv2",
        "mid2_residual",
        "mid2_clamp",
        "up2_first_norm",
        "up2_first_silu",
        "up2_first_res_conv1",
        "up2_first_res_silu",
        "up2_first_res_conv2",
        "up2_first_residual",
        "up2_first_clamp",
        "up2_second_norm",
        "up2_second_silu",
        "up2_second_res_conv1",
        "up2_second_res_silu",
        "up2_second_res_conv2",
        "up2_second_residual",
        "up2_second_clamp",
        "up2_third_norm",
        "up2_third_silu",
        "up2_third_res_conv1",
        "up2_third_res_silu",
        "up2_third_res_conv2",
        "up2_third_residual",
        "up2_third_clamp",
        "up1_first_norm",
        "up1_first_silu",
        "up1_first_res_conv1",
        "up1_first_res_silu",
        "up1_first_res_conv2",
        "up1_first_shortcut",
        "up1_first_residual",
    };
    const remaining = [_][]const u8{
        "up1_first_clamp",
        "up1_second_norm",
        "up1_second_silu",
        "up1_second_res_conv1",
        "up1_second_res_silu",
        "up1_second_res_conv2",
        "up1_second_residual",
        "up1_second_clamp",
        "up1_third_norm",
        "up1_third_silu",
        "up1_third_res_conv1",
        "up1_third_res_silu",
        "up1_third_res_conv2",
        "up1_third_residual",
        "up1_third_clamp",
        "up1_upsample",
        "up1_upsample_conv",
        "up0_first_norm",
        "up0_first_silu",
        "up0_first_res_conv1",
        "up0_first_res_silu",
        "up0_first_res_conv2",
        "up0_first_shortcut",
        "up0_first_residual",
        "up0_first_clamp",
        "up0_second_norm",
        "up0_second_silu",
        "up0_second_res_conv1",
        "up0_second_res_silu",
        "up0_second_res_conv2",
        "up0_second_residual",
        "up0_second_clamp",
        "up0_third_norm",
        "up0_third_silu",
        "up0_third_res_conv1",
        "up0_third_res_silu",
        "up0_third_res_conv2",
        "up0_third_residual",
        "up0_third_clamp",
        "final_silu",
        "conv_out",
        "mel",
    };
    try testing.expectEqual(reference_11.len + remaining.len, log.n);
    for (reference_11, 0..) |name, i| try testing.expectEqualStrings(name, log.at(i));
    for (remaining, reference_11.len..) |name, i| try testing.expectEqualStrings(name, log.at(i));
    try log.expectUnique();
}

test "tf32 grids: GemmIdentityThreadblockSwizzle<4> reproduces every grid the originals' profiles recorded" {
    const G = struct { tile: Tf32Tile, t: u64, co: u64, x: u32, y: u32 };
    // window-2 isolate profiles: up0 block convs [28,2], up1 upsample conv [28,4], conv_out [14,1], conv_pre [28,6],
    // the stage-0 AMP convs [112,3]
    for ([_]G{
        .{ .tile = .f16x10, .t = 436, .co = 512, .x = 28, .y = 2 },
        .{ .tile = .f16x10, .t = 436, .co = 1024, .x = 28, .y = 4 },
        .{ .tile = .f16x10, .t = 436, .co = 128, .x = 14, .y = 1 },
        .{ .tile = .f16x10, .t = 436, .co = 1536, .x = 28, .y = 6 },
        .{ .tile = .f128x64, .t = 3488, .co = 768, .x = 112, .y = 3 },
        // window-4 sweep profiles of the 32x5 originals: 96 ch x 55808 and 48 ch x 111616 both [1744,1]
        .{ .tile = .f32x5, .t = 55808, .co = 96, .x = 1744, .y = 1 },
        .{ .tile = .f32x5, .t = 111616, .co = 48, .x = 1744, .y = 1 },
    }) |c| {
        const g = tf32Grid(c.tile, c.t, c.co);
        try testing.expectEqual(c.x, g.x);
        try testing.expectEqual(c.y, g.y);
    }
    try testing.expectEqual(@as(u32, 81920), Tf32Tile.f16x10.smem());
    try testing.expectEqual(@as(u32, 73728), Tf32Tile.f128x64.smem());
    try testing.expectEqual(@as(u32, 81920), Tf32Tile.f32x5.smem());
    for (tf32_contracts, 0..) |a, i| for (tf32_contracts[i + 1 ..]) |b| {
        try testing.expect(!(a.ci == b.ci and a.co == b.co and a.t == b.t and a.k == b.k and a.pad == b.pad and a.dil == b.dil));
    };
}

test "convTf32 layout: listed contracts only, exact scratch sizes, aligned and disjoint copies" {
    const x = testView("x", 301 << 40, &.{ 1, 512, 436 });
    const w = testView("w", 302 << 40, &.{ 128, 512, 3 });
    const y = testView("y", 303 << 40, &.{ 1, 128, 436 });
    const bias = testView("b", 304 << 40, &.{128});
    const base: u64 = 300 << 40;
    const nin: u64 = 512 * 436;
    const nw: u64 = 128 * 3 * 512;
    const cv = try Carver.init(testView("scratch", base, &.{nin + nw + 128 * 436 + 4}), nin + nw + 128 * 436 + 4);
    const sc: Tf32Scratch = .{ .input = cv.at("in", 0, &.{nin}), .weight = cv.at("w", nin, &.{nw}), .output = cv.at("out", nin + nw, &.{128 * 436}) };
    const l = try Tf32Layout.plan(x, w, bias, y, .{ .pad = 1 }, sc);
    try testing.expectEqual(Tf32Tile.f16x10, l.tile);
    // an unlisted shape (the T218 contract has its own kernels), a stride, a dilation not proven for it
    try testing.expectError(error.InvalidArgument, Tf32Layout.plan(testView("x", 301 << 40, &.{ 1, 1024, 218 }), testView("w", 302 << 40, &.{ 1024, 1024, 3 }), null, testView("y", 303 << 40, &.{ 1, 1024, 218 }), .{ .pad = 1 }, sc));
    try testing.expectError(error.InvalidArgument, Tf32Layout.plan(x, w, bias, y, .{ .pad = 1, .stride = 2 }, sc));
    try testing.expectError(error.InvalidArgument, Tf32Layout.plan(x, w, bias, y, .{ .pad = 3, .dil = 3 }, sc));
    // a scratch view of the wrong size
    var short = sc;
    short.output = cv.at("out", nin + nw, &.{128 * 436 - 4});
    try testing.expectError(error.ShapeMismatch, Tf32Layout.plan(x, w, bias, y, .{ .pad = 1 }, short));
    // a copy 4 bytes off 16-byte alignment
    var odd = sc;
    odd.output = cv.at("out", nin + nw + 1, &.{128 * 436});
    try testing.expectError(error.LayoutUnsupported, Tf32Layout.plan(x, w, bias, y, .{ .pad = 1 }, odd));
    // a copy overlapping the input, another copy, or the bias
    var alias = sc;
    alias.input = View.contiguous("in", .cuda, 301 << 40, nin * 4, .borrowed, &.{nin});
    try testing.expectError(error.LayoutUnsupported, Tf32Layout.plan(x, w, bias, y, .{ .pad = 1 }, alias));
    var twice = sc;
    twice.output = cv.at("out", 0, &.{128 * 436});
    try testing.expectError(error.LayoutUnsupported, Tf32Layout.plan(x, w, bias, y, .{ .pad = 1 }, twice));
    try testing.expectError(error.LayoutUnsupported, Tf32Layout.plan(x, w, View.contiguous("b", .cuda, base, nin * 4, .borrowed, &.{128}), y, .{ .pad = 1 }, sc));
    // the output aliasing the input
    try testing.expectError(error.LayoutUnsupported, Tf32Layout.plan(x, w, bias, View.contiguous("y", .cuda, 301 << 40, nin * 4, .borrowed, &.{ 1, 128, 436 }), .{ .pad = 1 }, sc));
}

test "gather contracts: listed shapes only, disjoint from tf32_contracts, every tap in one phase, and they fit the vocoder's regions" {
    var n: [2]usize = .{ 0, 0 };
    const v = try vocoder.Vocoder.init(1, vocoder.tf32_frames);
    const vs = v.scratch();
    const weight = testView("conv_w", 400 << 40, &.{vs.conv_w_floats});
    const region = testView("up2", 401 << 40, &.{2 * vs.slot});
    for (gather_contracts, 0..) |a, i| {
        n[@backingInt(a.kind)] += 1;
        for (gather_contracts[i + 1 ..]) |b| try testing.expect(!(a.kind == b.kind and a.ci == b.ci and a.co == b.co and a.t == b.t and a.k == b.k and a.pad == b.pad and a.dil == b.dil and a.stride == b.stride));
        const dgrad = a.kind == .dgrad;
        if (!dgrad) {
            try testing.expectEqual(a.dil * (a.k - 1) / 2, a.pad);
            try testing.expectEqual(1, a.stride);
            try testing.expect(tf32Contract(a.ci, a.co, a.t, a.k, a.pad, a.dil) == null);
        } else try testing.expectEqual(1, a.dil);
        const x = testView("x", 402 << 40, &.{ 1, a.ci, a.t });
        const wv = if (dgrad) testView("w", 403 << 40, &.{ a.ci, a.co, a.k }) else testView("w", 403 << 40, &.{ a.co, a.ci, a.k });
        const tout = if (dgrad) vae.convTransposeOutLen(a.t, a.k, a.stride, a.pad) else a.t;
        const y = testView("y", 404 << 40, &.{ 1, a.co, tout });
        const bias = testView("b", 405 << 40, &.{a.co});
        const l = try GatherLayout.plan(x, wv, bias, y, .{ .kind = a.kind, .pad = @intCast(a.pad), .dil = @intCast(a.dil), .stride = @intCast(a.stride) }, .{ .weight = weight, .rows = region });
        // every tap in exactly one output phase, every output row in exactly one phase, every reduction element in exactly
        // one slice; a reduce order is a permutation of the slices
        var taps: [16]u8 = @splat(0);
        var rows: u64 = 0;
        for (0..l.u) |ph| {
            const p = l.phase(ph);
            for (0..p.nt) |j| taps[p.s0 + j * p.sstep] += 1;
            rows += p.rows;
            var kk: u64 = 0;
            for (0..a.split) |sl| kk += l.sliceK(p.nt, sl);
            // a flat layout pads its last k16 block with zero columns
            try testing.expectEqual(if (a.order == .flat_k16) (a.ci * p.nt + 15) / 16 * 16 else a.ci * p.nt, kk);
            try testing.expect(l.chunk * (l.kmax + a.co * @as(u64, if (a.split > 1) 2 else 1)) <= 2 * vs.slot);
        }
        try testing.expect(a.reduce.len == 0 or a.reduce.len == a.split);
        for (taps[0..a.k]) |c| try testing.expectEqual(1, c);
        try testing.expectEqual(tout, rows);
    }
    try testing.expectEqual([2]usize{ 27, 6 }, n);
    const T = struct {
        fn plan(ci: u64, co: u64, t: u64, k: u64, g: GatherGeom, sc: GatherScratch) !GatherLayout {
            const dgrad = g.kind == .dgrad;
            const tout = if (dgrad) vae.convTransposeOutLen(t, k, g.stride, g.pad) else t;
            return GatherLayout.plan(testView("x", 402 << 40, &.{ 1, ci, t }), if (dgrad) testView("w", 403 << 40, &.{ ci, co, k }) else testView("w", 403 << 40, &.{ co, ci, k }), null, testView("y", 404 << 40, &.{ 1, co, tout }), g, sc);
        }
    };
    const sc: GatherScratch = .{ .weight = weight, .rows = region };
    // row chunks: as many 128-row tiles as the 2-slot region holds, at most a phase's rows
    try testing.expectEqual(2304, (try T.plan(384, 384, 13952, 11, .{ .kind = .fprop, .pad = 5 }, sc)).chunk); // 4224 + 384 per row
    try testing.expectEqual(3488, (try T.plan(768, 384, 3488, 8, .{ .kind = .dgrad, .pad = 2, .stride = 4 }, sc)).chunk); // one chunk per phase
    try testing.expectEqual(89216, (try T.plan(48, 24, 111616, 4, .{ .kind = .dgrad, .pad = 1, .stride = 2 }, sc)).chunk); // two per phase
    // split contracts: A [rows][largest slice], D and a partial [rows][co]; one chunk per phase
    const k11 = try T.plan(768, 768, 3488, 11, .{ .kind = .fprop, .pad = 5 }, sc);
    try testing.expectEqual(528, k11.kmax); // 3 of the 48 chunks x 16 channels x 11 taps
    try testing.expectEqual(352, k11.sliceK(11, 16)); // slices 14-16: 2 chunks
    try testing.expectEqual(3488, k11.chunk);
    const up0 = try T.plan(1536, 768, 436, 16, .{ .kind = .dgrad, .pad = 4, .stride = 8 }, sc);
    try testing.expectEqual(320, up0.kmax); // 10 of the 96 chunks x 16 channels x 2 taps
    try testing.expectEqual(288, up0.sliceK(2, 9));
    try testing.expectEqual(436, up0.chunk);
    try testing.expectEqual(9, up0.sliceAt(1));
    // the flat layout: 264 reduction elements padded to 272 columns, 296 floats per row
    const k11s5 = try T.plan(24, 24, 223232, 11, .{ .kind = .fprop, .pad = 5 }, sc);
    try testing.expectEqual(272, k11s5.kmax);
    try testing.expectEqual(36096, k11s5.chunk);
    try testing.expectEqual(@as(u32, 18), gatherGrid(2304, 384).x);
    try testing.expectEqual(@as(u32, 3), gatherGrid(2304, 384).y);
    // an unlisted padding of ups.0, an unlisted dilation, a shape on a TF32 tile instead
    try testing.expectError(error.InvalidArgument, T.plan(1536, 768, 436, 16, .{ .kind = .dgrad, .pad = 3, .stride = 8 }, sc));
    try testing.expectError(error.InvalidArgument, T.plan(384, 384, 13952, 11, .{ .kind = .fprop, .pad = 35, .dil = 7 }, sc));
    try testing.expectError(error.InvalidArgument, T.plan(768, 768, 3488, 3, .{ .kind = .fprop, .pad = 1 }, sc));
    // a weight region one float short of the largest B, a rows region short of one 128-row chunk
    try testing.expectError(error.OutOfBounds, T.plan(384, 384, 13952, 11, .{ .kind = .fprop, .pad = 5 }, .{ .weight = testView("conv_w", 400 << 40, &.{4224 * 384 - 1}), .rows = region }));
    try testing.expectError(error.OutOfBounds, T.plan(384, 384, 13952, 11, .{ .kind = .fprop, .pad = 5 }, .{ .weight = weight, .rows = testView("up2", 401 << 40, &.{127 * 4608}) }));
    // a region overlapping the input, the two regions overlapping, the output aliasing the input
    try testing.expectError(error.LayoutUnsupported, T.plan(384, 384, 13952, 11, .{ .kind = .fprop, .pad = 5 }, .{ .weight = weight, .rows = testView("up2", 402 << 40, &.{2 * vs.slot}) }));
    try testing.expectError(error.LayoutUnsupported, T.plan(384, 384, 13952, 11, .{ .kind = .fprop, .pad = 5 }, .{ .weight = weight, .rows = testView("up2", 400 << 40, &.{2 * vs.slot}) }));
    try testing.expectError(error.LayoutUnsupported, GatherLayout.plan(testView("x", 402 << 40, &.{ 1, 384, 13952 }), testView("w", 403 << 40, &.{ 384, 384, 11 }), null, testView("y", 402 << 40, &.{ 1, 384, 13952 }), .{ .kind = .fprop, .pad = 5 }, sc));
}

test "vocode: the 1x436 vocoder takes every qualified path; any other geometry the direct ones, the same operators in order" {
    const T = struct {
        fn conv(nx: *u64, shape: [3]u64, bias: ?u64) vocoder.ConvWeights {
            nx.* += 2;
            return .{ .w = .checkpointEffective(testView("w", nx.* << 36, &shape), "w", @splat(0)), .bias = if (bias) |b| testView("b", (nx.* + 1) << 36, &.{b}) else null };
        }
        fn act(nx: *u64, c: u64) vocoder.ActWeights {
            nx.* += 4;
            return .{ .alpha = testView("a", nx.* << 36, &.{c}), .beta = testView("b", (nx.* + 1) << 36, &.{c}), .up = testView("u", (nx.* + 2) << 36, &.{vocoder.filter_taps}), .down = testView("d", (nx.* + 3) << 36, &.{vocoder.filter_taps}) };
        }
        fn weights(ws: *vocoder.VocoderWeights) void {
            var next: u64 = 1;
            ws.conv_pre = conv(&next, .{ 1536, 128, 7 }, 1536);
            var cin: u64 = 1536;
            for (0..6) |i| {
                const ch: u64 = vocoder.stageChannels(i);
                ws.ups[i] = conv(&next, .{ cin, ch, vocoder.upsample_kernel_sizes[i] }, ch);
                for (0..3) |j| {
                    const b = &ws.blocks[3 * i + j];
                    for (0..3) |k| {
                        b.convs1[k] = conv(&next, .{ ch, ch, vocoder.resblock_kernel_sizes[j] }, ch);
                        b.convs2[k] = conv(&next, .{ ch, ch, vocoder.resblock_kernel_sizes[j] }, ch);
                    }
                    for (&b.acts) |*a| a.* = act(&next, ch);
                }
                cin = ch;
            }
            ws.post = act(&next, cin);
            ws.conv_post = conv(&next, .{ 1, cin, 7 }, null);
        }
    };
    var w: vocoder.VocoderWeights = undefined;
    T.weights(&w);
    const policy: VocodePolicy = .{ .conv_acc = .fma, .aa_acc = .fma, .avg = .reciprocal };
    const v = try vocoder.Vocoder.init(1, vocoder.tf32_frames);
    var r: Recorder = .{ .scratch_base = 4000 << 36 };
    try vocodeWith(&r, {}, .diagnostic, v, &w, testView("mel", 4001 << 36, &v.melShape()), testView("scratch", r.scratch_base, &.{v.scratch().floats}), testView("wave", 4002 << 36, &v.waveShape()), policy, null);
    // conv_pre and the TF32-tile AMP blocks, 6 convolutions each (convs1 at dilations 1, 3, 5 and three convs2): stage 0
    // kernels 3 and 7, stages 3 and 4 kernels 7 and 11
    try testing.expectEqual(1 + 6 * 6, r.tf32_convs);
    // the gather shapes: stages 1 and 2 (18 convolutions each), stage 3's kernel-3 block (6), stage 0's and stage 5's
    // kernel-11 blocks (6 each); all six upsamplers. The 19 left (stage 4 kernel 3, stage 5 kernels 3 and 7, conv_post)
    // are the direct conv_acc convolutions.
    try testing.expectEqual([2]usize{ 18 + 18 + 6 + 6 + 6, 6 }, r.gather_convs);
    // any other geometry: none of them, and the same operators in the same order
    const small = try vocoder.Vocoder.init(2, 2);
    var o: Recorder = .{ .scratch_base = 4000 << 36 };
    try vocodeWith(&o, {}, .diagnostic, small, &w, testView("mel", 4001 << 36, &small.melShape()), testView("scratch", o.scratch_base, &.{small.scratch().floats}), testView("wave", 4002 << 36, &small.waveShape()), policy, null);
    try testing.expectEqual(0, o.tf32_convs);
    try testing.expectEqual([2]usize{ 0, 0 }, o.gather_convs);
    try testing.expectEqualSlices(vae.Op, o.ops[0..o.n_ops], r.ops[0..r.n_ops]);
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
    try vocodeWith(&r, {}, .diagnostic, v, &w, testView("mel", 4001 << 36, &v.melShape()), testView("scratch", r.scratch_base, &.{v.scratch().floats}), wave, policy, null);
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

    // the stage hooks: every stage once, in order, observing without changing an operator
    var log: NameLog = .{};
    var traced: Recorder = .{ .scratch_base = r.scratch_base };
    try vocodeWith(&traced, {}, .diagnostic, v, &w, testView("mel", 4001 << 36, &v.melShape()), testView("scratch", r.scratch_base, &.{v.scratch().floats}), wave, policy, log.hook());
    try testing.expectEqualSlices(vae.Op, r.ops[0..r.n_ops], traced.ops[0..traced.n_ops]);
    var want: [128][32]u8 = undefined;
    var wl: [128][]const u8 = undefined;
    var n: usize = 0;
    wl[n] = "conv_pre";
    n += 1;
    for (0..vocoder.upsample_rates.len) |i| {
        wl[n] = try std.fmt.bufPrint(&want[n], "upsample{d}", .{i});
        n += 1;
        for (0..vocoder.resblock_kernel_sizes.len) |j| {
            wl[n] = try std.fmt.bufPrint(&want[n], "stage{d}_amp{d}", .{ i, j });
            n += 1;
        }
        wl[n] = try std.fmt.bufPrint(&want[n], "stage{d}_avg", .{i});
        n += 1;
    }
    for ([_][]const u8{ "post_act", "conv_post", "wave" }) |name| {
        wl[n] = name;
        n += 1;
    }
    try testing.expectEqual(1 + 6 * 5 + 3, n);
    try testing.expectEqual(n, log.n);
    for (wl[0..n], 0..) |name, i| try testing.expectEqualStrings(name, log.at(i));
    try log.expectUnique();
}
