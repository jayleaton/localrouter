//! Kandinsky-6 Pro-distill vocoder on the host side: the BigVGAN v2 graph (MMAudio `bigvgan_v2`, AMPBlock1, SnakeBeta
//! with log-scale parameters, alias-free Activation1d) resolved from the selected HF `1018d582` vocoder config, its
//! weights and memory plan, the admission check, and fp32 host references of SnakeBeta and the anti-aliasing FIR
//! resamplers written op for op like kernels/cuda/kandinsky6/audio/audio.cu. Shares the audio ABI of vae.zig.
//!
//! Weights as stored: the selected HF `1018d582` vocoder checkpoint holds 667 fp32 tensors with plain `weight` (no
//! weight-norm g / v), and the native loader (`load_bigvgan_v2`, MIT) removes weight norm before loading them, so they
//! are the effective weights (`vae.Weight.checkpointEffective`). A weight-norm fold is not on the native path and is
//! refused (`vocoder_weight_norm_fold`). The 12-tap FIR filters are the modules' loaded `filter` buffers;
//! `kaiserSincDesign` below is a diagnostic design in f64, not a production source.

const std = @import("std");
const vae = @import("vae.zig");
const Op = vae.Op;
const Use = vae.Use;
const Acc = vae.Acc;
const Error = vae.Error;

pub const num_mels = vae.mel_channels;
pub const upsample_rates = [6]u32{ 8, 4, 2, 2, 2, 2 };
pub const upsample_kernel_sizes = [6]u32{ 16, 8, 4, 4, 4, 4 };
pub const upsample_initial_channel: u32 = 1536;
pub const resblock_kernel_sizes = [3]u32{ 3, 7, 11 };
pub const resblock_dilations = [3]u32{ 1, 3, 5 };
/// Samples per mel frame: the product of the upsample rates.
pub const hop = 512;
pub const filter_taps = 12;
/// SnakeBeta `no_div_by_zero` (0.000000001) as fp32.
pub const snake_eps: f32 = 1e-9;
/// use_tanh_at_final false: the output is clamped to [-1, 1]; use_bias_at_final false: conv_post has no bias.
pub const final_clamp: f32 = 1.0;

comptime {
    var p: u32 = 1;
    for (upsample_rates) |r| p *= r;
    std.debug.assert(p == hop);
    std.debug.assert(hop * 2 == vae.downsample_factor);
}

/// Stage i's output channels: 1536 / 2^(i+1) (768, 384, 192, 96, 48, 24).
pub fn stageChannels(i: usize) u32 {
    return upsample_initial_channel >> @intCast(i + 1);
}

/// `get_padding(k, d)` = (k * d - d) / 2: "same" length for a stride-1 dilated convolution.
pub fn samePadding(k: u32, d: u32) u32 {
    return (k * d - d) / 2;
}

/// The operators one forward runs (per Activation1d: aa_up2, snake_beta_log, aa_down2).
pub const forward_ops = [_]Op{
    .conv1d_prepared, // conv_pre 128 -> 1536, K7, pad 3, bias
    .conv_transpose1d_prepared, // ups[i]: K = 2u, stride u, pad (k - u) / 2, bias
    .aa_up2,
    .snake_beta_log,
    .aa_down2,
    .conv1d_prepared, // AMPBlock1 convs1 (dilation d) and convs2 (dilation 1), bias
    .vocoder_add,
    .vocoder_avg3,
    .clamp,
};

pub const ParamCounts = struct { weights: u64, biases: u64, snake: u64, filters: u64 };

pub fn params() ParamCounts {
    var p: ParamCounts = .{ .weights = 0, .biases = 0, .snake = 0, .filters = 0 };
    p.weights += num_mels * upsample_initial_channel * 7;
    p.biases += upsample_initial_channel;
    var cin: u64 = upsample_initial_channel;
    for (upsample_rates, upsample_kernel_sizes, 0..) |_, k, i| {
        const ch: u64 = stageChannels(i);
        p.weights += cin * ch * k; // ConvTranspose1d [Cin, Cout, K]
        p.biases += ch;
        for (resblock_kernel_sizes) |rk| {
            p.weights += 6 * ch * ch * rk; // 3 convs1 + 3 convs2
            p.biases += 6 * ch;
            p.snake += 6 * 2 * ch; // 6 Activation1d, alpha and beta [C]
            p.filters += 6 * 2 * filter_taps; // each Activation1d: up and down filter
        }
        cin = ch;
    }
    p.snake += 2 * cin; // activation_post
    p.filters += 2 * filter_taps;
    p.weights += cin * 1 * 7; // conv_post, no bias
    return p;
}

pub const Memory = vae.Memory;

pub const Vocoder = struct {
    batch: u32,
    /// Mel frames (the decoder output's T, 2 x the latent frames).
    frames: u32,

    pub fn init(batch: u32, frames: u32) Error!Vocoder {
        if (batch == 0 or frames == 0) return error.InvalidArgument;
        return .{ .batch = batch, .frames = frames };
    }

    pub fn melShape(v: Vocoder) [3]u64 {
        return .{ v.batch, num_mels, v.frames };
    }

    /// [B, 1, frames * 512] fp32 in [-1, 1]; int16 conversion belongs to media integration.
    pub fn waveShape(v: Vocoder) [3]u64 {
        return .{ v.batch, 1, @as(u64, v.frames) * hop };
    }

    /// Resident: prepared fp32 weights, biases, SnakeBeta parameters and filters. Working (per request), at the largest
    /// stage C*T: the stage input, three AMP-block outputs, two work buffers and the Activation1d's 2x intermediate.
    pub fn memory(v: Vocoder) Memory {
        const p = params();
        var max_ct: u64 = @as(u64, upsample_initial_channel) * v.frames;
        var t: u64 = v.frames;
        for (upsample_rates, 0..) |r, i| {
            t *= r;
            max_ct = @max(max_ct, @as(u64, stageChannels(i)) * t);
        }
        return .{
            .resident_bytes = (p.weights + p.biases + p.snake + p.filters) * 4,
            .working_bytes = (6 * max_ct + 2 * max_ct) * @as(u64, v.batch) * 4,
        };
    }

    /// Every operator under `use` with weights in `state`, before any allocation. Checkpoint weights are used as stored;
    /// stored weight-norm parametrizations would need a fold the native path never runs (refused). What remains is
    /// target qualification (refused for production).
    pub fn admit(v: Vocoder, use: Use, state: vae.WeightState) Error!void {
        _ = v;
        if (state == .stored_checkpoint) try vae.admitOp(.vocoder_weight_norm_fold, use);
        if (state == .prepared_native_reference) return error.InvalidArgument; // nothing prepares vocoder weights
        for (forward_ops) |op| try vae.admitOp(op, use);
    }
};

// -------------------------------------------------------------------------------------- host references (fp32)

/// audio.cu k6a_snake_beta_log: y = x + (1 / (beta + eps)) * sin(x * alpha)^2 with alpha, beta = exp(log params),
/// each step rounded. Zig's libm exp / sin are not CUDA's: a structural reference.
pub fn snakeBetaLog(x: []const f32, alpha_log: []const f32, beta_log: []const f32, y: []f32, b: usize, c: usize, t: usize) void {
    for (0..b * c * t) |e| {
        const ci = (e / t) % c;
        const alpha = @exp(alpha_log[ci]);
        const beta = @exp(beta_log[ci]);
        const s = @sin(x[e] * alpha);
        const inv = 1.0 / (beta + snake_eps);
        y[e] = x[e] + inv * (s * s);
    }
}

inline fn accum(acc: f32, w: f32, x: f32, mode: Acc) f32 {
    return if (mode == .fma) @mulAdd(f32, w, x, acc) else acc + w * x;
}

fn clampIdx(i: isize, n: usize) usize {
    return @intCast(std.math.clamp(i, 0, @as(isize, @intCast(n)) - 1));
}

/// audio.cu k6a_aa_up2: [rows, T] -> [rows, 2T] (replicate pad 5, conv_transpose stride 2, x2, crop 15).
pub fn aaUp2(x: []const f32, f: *const [filter_taps]f32, y: []f32, rows: usize, t: usize, mode: Acc) void {
    for (0..rows) |r| for (0..2 * t) |o| {
        const n = o + 15;
        var acc: f32 = 0.0;
        for (0..filter_taps) |k| {
            if (k > n or (n - k) & 1 != 0) continue;
            const i = (n - k) >> 1;
            if (i >= t + 10) continue;
            acc = accum(acc, f[k], x[r * t + clampIdx(@as(isize, @intCast(i)) - 5, t)], mode);
        }
        y[r * 2 * t + o] = 2.0 * acc;
    };
}

pub fn aaDown2Len(t2: usize) usize {
    return (t2 + 11 - filter_taps) / 2 + 1;
}

/// audio.cu k6a_aa_down2: [rows, T2] -> [rows, (T2 - 1) / 2 + 1] (replicate pad 5 left / 6 right, conv stride 2).
pub fn aaDown2(x: []const f32, f: *const [filter_taps]f32, y: []f32, rows: usize, t2: usize, mode: Acc) void {
    const to = aaDown2Len(t2);
    for (0..rows) |r| for (0..to) |m| {
        var acc: f32 = 0.0;
        for (0..filter_taps) |k| acc = accum(acc, f[k], x[r * t2 + clampIdx(@as(isize, @intCast(2 * m + k)) - 5, t2)], mode);
        y[r * to + m] = acc;
    };
}

/// The vocoder stage merge: ((r0 + r1) + r2) / 3 with the explicit scalar division.
pub fn avg3(r0: []const f32, r1: []const f32, r2: []const f32, y: []f32, c: vae.ScalarDiv) void {
    for (r0, r1, r2, y) |a, b, d, *o| o.* = c.apply((a + b) + d);
}

/// Diagnostic only: the alias-free Kaiser-windowed sinc low-pass (cutoff 0.25, half width 0.3, 12 taps, even) designed
/// in f64 from its published definition. Production uses the checkpoint's `filter` buffers; this never substitutes them.
pub fn kaiserSincDesign() [filter_taps]f64 {
    const cutoff = 0.25;
    const half_width = 0.3;
    const half = filter_taps / 2;
    const a = 2.285 * @as(f64, half - 1) * std.math.pi * (4.0 * half_width) + 7.95;
    const beta: f64 = if (a > 50.0) 0.1102 * (a - 8.7) else if (a >= 21.0) 0.5842 * std.math.pow(f64, a - 21.0, 0.4) + 0.07886 * (a - 21.0) else 0.0;
    var f: [filter_taps]f64 = undefined;
    var sum: f64 = 0;
    for (&f, 0..) |*v, n| {
        const ratio = 2.0 * @as(f64, @floatFromInt(n)) / @as(f64, filter_taps - 1) - 1.0;
        const window = besselI0(beta * @sqrt(1.0 - ratio * ratio)) / besselI0(beta);
        const time = @as(f64, @floatFromInt(n)) - @as(f64, half) + 0.5;
        const xarg = 2.0 * cutoff * time;
        const sinc = if (xarg == 0) 1.0 else @sin(std.math.pi * xarg) / (std.math.pi * xarg);
        v.* = 2.0 * cutoff * window * sinc;
        sum += v.*;
    }
    for (&f) |*v| v.* /= sum;
    return f;
}

fn besselI0(x: f64) f64 {
    var sum: f64 = 1;
    var term: f64 = 1;
    var k: f64 = 1;
    while (k < 64) : (k += 1) {
        term *= (x / (2 * k)) * (x / (2 * k));
        sum += term;
    }
    return sum;
}

// ------------------------------------------------------------------------------------------------------ tests

const testing = std.testing;

test "vocoder graph: channels, lengths, parameters, memory" {
    try testing.expectEqualSlices(u32, &.{ 768, 384, 192, 96, 48, 24 }, &.{ stageChannels(0), stageChannels(1), stageChannels(2), stageChannels(3), stageChannels(4), stageChannels(5) });
    // ConvTranspose1d length (L - 1) * u - 2 * (k - u) / 2 + k = u * L for k = 2u
    var t: usize = 436;
    for (upsample_rates, upsample_kernel_sizes) |u, k| {
        const next = vae.convTransposeOutLen(t, k, u, (k - u) / 2);
        try testing.expectEqual(t * u, next);
        t = next;
    }
    try testing.expectEqual(223232, t);
    for (resblock_kernel_sizes) |k| for (resblock_dilations) |d| {
        try testing.expectEqual(200, vae.convOutLen(.{ .b = 1, .cin = 1, .cout = 1, .lin = 200, .k = k, .pad = samePadding(k, d), .dil = d }));
    };
    const v = try Vocoder.init(1, 436);
    try testing.expectEqualSlices(u64, &.{ 1, 1, 223232 }, &v.waveShape());
    const p = params();
    var want_w: u64 = 128 * 1536 * 7 + 24 * 7;
    var cin: u64 = 1536;
    for (0..6) |i| {
        const ch: u64 = stageChannels(i);
        want_w += cin * ch * upsample_kernel_sizes[i] + 6 * ch * ch * (3 + 7 + 11);
        cin = ch;
    }
    try testing.expectEqual(want_w, p.weights);
    try testing.expectEqual(@as(u64, 18 * 6 + 1) * 2 * filter_taps, p.filters);
    // the largest stage C*T: 1536 x 436 = 669696 at conv_pre; every later stage is 768 x 3488 = 2678784 or 5357568
    const m = v.memory();
    try testing.expectEqual(@as(u64, 8 * 5357568 * 4), m.working_bytes);
}

test "vocoder admission: checkpoint weights as stored, the fold refused, production pending" {
    const v = try Vocoder.init(1, 436);
    try testing.expectError(error.NotOnNativePath, v.admit(.diagnostic, .stored_checkpoint));
    try testing.expectError(error.InvalidArgument, v.admit(.diagnostic, .prepared_native_reference));
    try v.admit(.diagnostic, .checkpoint_effective);
    try testing.expectError(error.GpuQualificationPending, v.admit(.production, .checkpoint_effective));
}

test "anti-alias resamplers: impulse responses and replicate edges" {
    var f: [filter_taps]f32 = undefined;
    for (&f, 0..) |*v, i| v.* = @floatFromInt(i + 1); // distinct taps expose the index arithmetic
    // a constant input stays constant up to the filter sum (replicate padding, no zero edges)
    const c = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var u: [16]f32 = undefined;
    aaUp2(&c, &f, &u, 1, 8, .mul_add);
    // each output takes 6 of the 12 taps (one parity): odd taps 2+4+..+12 = 42 or even 1+3+..+11 = 36, times 2
    for (u, 0..) |v, o| try testing.expectEqual(@as(f32, if ((o + 15) & 1 == 0) 72 else 84), v);
    var d: [4]f32 = undefined;
    try testing.expectEqual(4, aaDown2Len(8));
    aaDown2(&c, &f, &d, 1, 8, .fma);
    for (d) |v| try testing.expectEqual(@as(f32, 78), v);
    // an impulse at x[3] lands on down output m where 2m + k - 5 = 3: k = 8 - 2m
    var imp: [8]f32 = @splat(0);
    imp[3] = 1;
    aaDown2(&imp, &f, &d, 1, 8, .fma);
    try testing.expectEqualSlices(f32, &.{ 9, 7, 5, 3 }, &d);
    // the f64 design is symmetric, sums to 1, and is a low-pass (positive centre taps)
    const k = kaiserSincDesign();
    var s: f64 = 0;
    for (k, 0..) |v, i| {
        s += v;
        try testing.expectApproxEqAbs(v, k[filter_taps - 1 - i], 1e-15);
    }
    try testing.expectApproxEqAbs(1.0, s, 1e-12);
    try testing.expect(k[5] > 0.3 and k[6] > 0.3);
}

test "snake and avg3 references" {
    const x = [_]f32{ 0, 1, -2 };
    const zero = [_]f32{0};
    var y: [3]f32 = undefined;
    snakeBetaLog(&x, &zero, &zero, &y, 1, 1, 3); // alpha = beta = 1
    try testing.expectEqual(@as(f32, 0), y[0]);
    try testing.expectEqual(@as(f32, 1) + (1.0 / (1.0 + snake_eps)) * (@sin(@as(f32, 1)) * @sin(@as(f32, 1))), y[1]);
    const r = [_]f32{ 1, 2, 3 };
    var o: [3]f32 = undefined;
    avg3(&r, &r, &r, &o, .of(3, .true_div));
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, &o);
}
