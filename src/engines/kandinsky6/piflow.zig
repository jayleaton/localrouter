//! The multimodal DX / π-Flow sampler of Kandinsky-6 Pro-distill (kandinsky/core/algo/piflow_sampler.py:41-198,
//! piflow_math.py:8-189 at eac06e39) as torch 2.14.1 CUDA eager runs it at batch 1: every per-batch quantity
//! (tau, sigma, segment, interpolation weights) is a [1, 1, ...] fp32 device tensor, so it is computed here on the
//! host, one IEEE f32 rounding a torch op in torch's order, and handed to `k6_piflow_rollout` as a table
//! (kernels/cuda/kandinsky6/central/central.cu). Python float constants enter as f32 (`Scalar.to<float>`); the
//! segment is Python f64 arithmetic rounded once. Divisions between device tensors are IEEE divisions: the
//! reciprocal multiply of aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu:35-52 (v2.14.1) applies only when the
//! divisor is a CPU scalar (`is_cpu_scalar`, TensorIterator.cpp:796-798), never here. Adds and subtracts are
//! `self + alpha * other` with alpha = +-1 (native/ufunc/add.h:15-17): one rounding, contracted or not. Clamps follow
//! cuda/TensorCompare.cu:59-90 (NaN kept, a value equal to the bound kept), `round` is nearbyintf (half to even,
//! cuda/UnaryFractionKernels.cu:110-114).
//!
//! The state enters step 0 as bf16 (the noise) and leaves every rollout as fp32: `torch.where(mask, x_t_minus, x_t)`
//! promotes (piflow_math.py:185), and num_substeps >= 1.

const std = @import("std");
const shapes = @import("shapes.zig");

pub const n_grid: u32 = shapes.n_grid;

/// The sampler's arguments as Python holds them (piflow_denoise_loop defaults, `pro-distill.yaml` values in shapes).
pub const Config = struct {
    steps: u32 = shapes.steps,
    shift: f64 = shapes.shift,
    eps: f64 = 1e-6,
    final_step_size_scale: f64 = shapes.final_step_size_scale,
    policy_substeps: u32 = shapes.policy_substeps,
};

/// The Python floats as the CUDA kernels see them (f64 rounded once to f32).
const Consts = struct {
    shift: f32,
    shift_m1: f32, // `shift - 1` (shift_timesteps)
    one_m_shift: f32, // `1 - self.shift` (_unwarp_t)
    eps: f32,
    substeps: f32,

    fn of(c: Config) Consts {
        return .{
            .shift = @floatCast(c.shift),
            .shift_m1 = @floatCast(c.shift - 1.0),
            .one_m_shift = @floatCast(1.0 - c.shift),
            .eps = @floatCast(c.eps),
            .substeps = @floatFromInt(c.policy_substeps),
        };
    }
};

/// clamp(min=lo) (TensorCompare.cu:69-75).
fn clampMin(v: f32, lo: f32) f32 {
    if (std.math.isNan(v) or v == lo) return v;
    return if (v < lo) lo else v;
}

/// clamp(min=lo, max=hi) (TensorCompare.cu:69-71, :82-84).
fn clamp(v: f32, lo: f32, hi: f32) f32 {
    if (std.math.isNan(v)) return v;
    const r = if (v < lo) lo else v;
    return if (hi < r) hi else r;
}

/// shift_timesteps (piflow_math.py:150-152): (shift * t) / (1 + (shift - 1) * t).
fn shiftTimesteps(k: Consts, t: f32) f32 {
    return (k.shift * t) / ((k.shift_m1 * t) + 1.0);
}

/// DXPolicy._unwarp_t (piflow_math.py:50-51): sigma / (shift + (1 - shift) * sigma).
fn unwarp(k: Consts, sigma: f32) f32 {
    return sigma / ((k.one_m_shift * sigma) + k.shift);
}

/// The DiT time input of an outer step, `sigma_t_src * 1000` (piflow_sampler.py:108), the same for video and audio.
pub fn ditTime(sigma_t_src: f32) f32 {
    return sigma_t_src * 1000.0;
}

/// One outer step's per-batch scalars.
pub const Step = struct {
    index: u32,
    /// f32(segment_size) (Python f64, :84-92).
    segment: f32,
    /// raw_t_src and tau_dst = (tau_src - segment).clamp(min=eps) (:93-94); tau_dst is the next step's tau_src.
    tau_src: f32,
    tau_dst: f32,
    /// shift_timesteps(tau_src) (:95), and the DiT time input sigma * 1000 (:108).
    sigma_src: f32,
    time: f32,
    /// DXPolicy (piflow_math.py:29-35): raw_t_src re-derived from sigma by _unwarp_t (not tau_src), raw_t_dst and
    /// the clamped segment.
    raw_src: f32,
    raw_dst: f32,
    policy_segment: f32,
    /// policy_rollout_fm (:171-173): delta = tau_src - tau_dst, round(delta * substeps) clamped to >= 1, and
    /// delta / num_substeps.
    delta: f32,
    substeps: u32,
    substep_size: f32,
};

/// Outer step `i` from its tau_src (1 for step 0, the previous step's tau_dst after).
pub fn step(c: Config, i: u32, tau_src: f32) Step {
    const k = Consts.of(c);
    const final = @max(c.final_step_size_scale, c.eps);
    const base = 1.0 / (@as(f64, @floatFromInt(c.steps)) - (1.0 - final));
    const segment: f32 = @floatCast(base * (if (i == c.steps - 1) final else 1.0));
    const tau_dst = clampMin(tau_src - segment, k.eps);
    const sigma = shiftTimesteps(k, tau_src);
    const raw_src = unwarp(k, sigma);
    const raw_dst = clampMin(raw_src - segment, 0.0);
    const delta = tau_src - tau_dst;
    const n: u32 = @intFromFloat(@max(roundHalfEven(delta * k.substeps), 1.0));
    return .{
        .index = i,
        .segment = segment,
        .tau_src = tau_src,
        .tau_dst = tau_dst,
        .sigma_src = sigma,
        .time = ditTime(sigma),
        .raw_src = raw_src,
        .raw_dst = raw_dst,
        .policy_segment = clampMin(raw_src - raw_dst, k.eps),
        .delta = delta,
        .substeps = n,
        .substep_size = delta / @as(f32, @floatFromInt(n)),
    };
}

/// nearbyintf under the default rounding mode: half to even.
fn roundHalfEven(x: f32) f32 {
    const r = @round(x); // half away from zero
    if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2.0) != 0) return r - std.math.sign(x);
    return r;
}

/// All outer steps (out.len == c.steps), tau_src chained from 1.
pub fn schedule(c: Config, out: []Step) void {
    std.debug.assert(out.len == c.steps);
    var tau: f32 = 1.0;
    for (out, 0..) |*s, i| {
        s.* = step(c, @intCast(i), tau);
        tau = s.tau_dst;
    }
}

/// One substep's scalars: the interpolation of the x0 grid at the current sigma (DXPolicy.pi / _interpolate,
/// piflow_math.py:39-48, :61-74) and the Euler update (policy_rollout_fm, :179-187).
pub const Substep = struct {
    sigma: f32,
    raw: f32,
    /// t0 = floor(t.clamp(0, 1) * (n - 1)) clamped to [0, n - 2]; wa = float(t0 + 1) - t, wb = t - float(t0).
    t0: u32,
    wa: f32,
    wb: f32,
    /// sigma.clamp(min=eps), the velocity's divisor.
    div: f32,
    /// sigma_{j+1} - sigma_j with sigma_{j+1} = shift_timesteps((raw - substep_size).clamp(min=0)).
    ds: f32,
    sigma_next: f32,
    raw_next: f32,
};

pub const Weights = struct { t0: u32, wa: f32, wb: f32 };

/// _interpolate's index and weights for a normalized time t (piflow_math.py:43-48), n grid points >= 2.
pub fn weights(t: f32, n: u32) Weights {
    const tc = clamp(t, 0.0, 1.0) * @as(f32, @floatFromInt(n - 1));
    const fl = @floor(tc);
    const t0: u32 = @intFromFloat(@min(@max(fl, 0.0), @as(f32, @floatFromInt(n - 2))));
    return .{ .t0 = t0, .wa = @as(f32, @floatFromInt(t0 + 1)) - tc, .wb = tc - @as(f32, @floatFromInt(t0)) };
}

/// The step's substeps (out.len >= s.substeps); returns the filled prefix.
pub fn substeps(c: Config, s: Step, out: []Substep) []Substep {
    std.debug.assert(out.len >= s.substeps);
    const k = Consts.of(c);
    var sigma = s.sigma_src;
    var raw = s.tau_src;
    for (out[0..s.substeps]) |*o| {
        const t = (unwarp(k, sigma) - s.raw_dst) / s.policy_segment;
        const w = weights(t, n_grid);
        const raw_next = clampMin(raw - s.substep_size, 0.0);
        const sigma_next = shiftTimesteps(k, raw_next);
        o.* = .{
            .sigma = sigma,
            .raw = raw,
            .t0 = w.t0,
            .wa = w.wa,
            .wb = w.wb,
            .div = clampMin(sigma, k.eps),
            .ds = sigma_next - sigma,
            .sigma_next = sigma_next,
            .raw_next = raw_next,
        };
        sigma = sigma_next;
        raw = raw_next;
    }
    return out[0..s.substeps];
}

/// `k6_piflow_rollout`'s by-value table (struct K6PiflowTable in central.cu): up to `capacity` substeps a launch.
pub const Table = extern struct {
    pub const capacity = 32;

    sigma: f32,
    count: i32,
    t0: [capacity]i32,
    wa: [capacity]f32,
    wb: [capacity]f32,
    div: [capacity]f32,
    ds: [capacity]f32,

    /// The table of substeps [first, first + capacity) of a step.
    pub fn of(s: Step, subs: []const Substep, first: usize) Table {
        const part = subs[first..@min(subs.len, first + capacity)];
        var t: Table = std.mem.zeroes(Table);
        t.sigma = s.sigma_src;
        t.count = @intCast(part.len);
        for (part, 0..) |p, j| {
            t.t0[j] = @intCast(p.t0);
            t.wa[j] = p.wa;
            t.wb[j] = p.wb;
            t.div[j] = p.div;
            t.ds[j] = p.ds;
        }
        return t;
    }
};

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The host twin of one element's rollout (the kernel's arithmetic, same op order, f32): x_src the element's
/// state at the step's start (bf16 widened or fp32), pred its 10 grid predictions (bf16 widened).
pub fn rolloutElem(x_src: f32, pred: [n_grid]f32, s: Step, subs: []const Substep) f32 {
    var x0: [n_grid]f32 = undefined;
    for (&x0, pred) |*o, p| o.* = x_src - s.sigma_src * p;
    var x = x_src;
    for (subs) |j| {
        const xi = j.wa * x0[j.t0] + j.wb * x0[j.t0 + 1];
        const v = (x - xi) / j.div;
        x = x + v * j.ds;
    }
    return x;
}

/// The state at the step's start: bf16 bits (step 0) or fp32.
pub const State = union(enum) { bf16: []const u16, f32: []const f32 };

/// The host twin of a whole rollout: pred bf16 bits grid-major [n_grid][n], out fp32 [n].
pub fn rollout(src: State, pred: []const u16, out: []f32, s: Step, subs: []const Substep) void {
    const n = out.len;
    std.debug.assert(pred.len == n_grid * n);
    for (out, 0..) |*o, i| {
        const xs = switch (src) {
            .bf16 => |b| bf16ToF32(b[i]),
            .f32 => |f| f[i],
        };
        var p: [n_grid]f32 = undefined;
        for (&p, 0..) |*q, k| q.* = bf16ToF32(pred[k * n + i]);
        o.* = rolloutElem(xs, p, s, subs);
    }
}

fn fb(x: u32) f32 {
    return @bitCast(x);
}

test "the 10-step schedule, against an independent exact-f32 derivation" {
    // each torch op rounded once to f32 via f64 (exact for one + - * / of f32 operands), Python round half-even
    const want = [_]struct { seg: u32, tau: u32, dst: u32, sig: u32, time: u32, rs: u32, rd: u32, sp: u32, n: u32, sub: u32 }{
        .{ .seg = 0x3dd79436, .tau = 0x3f800000, .dst = 0x3f650d79, .sig = 0x3f800000, .time = 0x447a0000, .rs = 0x3f800000, .rd = 0x3f650d79, .sp = 0x3dd79438, .n = 13, .sub = 0x3c04a9fb },
        .{ .seg = 0x3dd79436, .tau = 0x3f650d79, .dst = 0x3f4a1af2, .sig = 0x3f7a1d6e, .time = 0x447440bd, .rs = 0x3f650d7e, .rd = 0x3f4a1af7, .sp = 0x3dd79438, .n = 13, .sub = 0x3c04a9fb },
        .{ .seg = 0x3dd79436, .tau = 0x3f4a1af2, .dst = 0x3f2f286b, .sig = 0x3f7309b8, .time = 0x446d577e, .rs = 0x3f4a1af0, .rd = 0x3f2f2869, .sp = 0x3dd79438, .n = 13, .sub = 0x3c04a9fb },
        .{ .seg = 0x3dd79436, .tau = 0x3f2f286b, .dst = 0x3f1435e4, .sig = 0x3f6a5dbf, .time = 0x4464df8d, .rs = 0x3f2f286c, .rd = 0x3f1435e5, .sp = 0x3dd79438, .n = 13, .sub = 0x3c04a9fb },
        .{ .seg = 0x3dd79436, .tau = 0x3f1435e4, .dst = 0x3ef286ba, .sig = 0x3f5f7df7, .time = 0x445a4103, .rs = 0x3f1435e3, .rd = 0x3ef286b8, .sp = 0x3dd79438, .n = 13, .sub = 0x3c04a9fb },
        .{ .seg = 0x3dd79436, .tau = 0x3ef286ba, .dst = 0x3ebca1ac, .sig = 0x3f51745c, .time = 0x444c8ba2, .rs = 0x3ef286b9, .rd = 0x3ebca1ac, .sp = 0x3dd79434, .n = 13, .sub = 0x3c04a9fb },
        .{ .seg = 0x3dd79436, .tau = 0x3ebca1ac, .dst = 0x3e86bc9e, .sig = 0x3f3ea366, .time = 0x443a2b92, .rs = 0x3ebca1ac, .rd = 0x3e86bc9e, .sp = 0x3dd79438, .n = 13, .sub = 0x3c04a9fb },
        .{ .seg = 0x3dd79436, .tau = 0x3e86bc9e, .dst = 0x3e21af21, .sig = 0x3f241a40, .time = 0x442041a2, .rs = 0x3e86bc9f, .rd = 0x3e21af23, .sp = 0x3dd79436, .n = 13, .sub = 0x3c04a9fa },
        .{ .seg = 0x3dd79436, .tau = 0x3e21af21, .dst = 0x3d579418, .sig = 0x3ef7bde9, .time = 0x43f1ef76, .rs = 0x3e21af21, .rd = 0x3d579418, .sp = 0x3dd79436, .n = 13, .sub = 0x3c04a9fa },
        .{ .seg = 0x3d579436, .tau = 0x3d579418, .dst = 0x358637bd, .sig = 0x3e5e9bba, .time = 0x43596414, .rs = 0x3d579418, .rd = 0x00000000, .sp = 0x3d579418, .n = 7, .sub = 0x3bf65ee9 },
    };
    // sigma at the end of each rollout (not the next step's sigma_src) and the substeps' interpolation indices
    const end = [_]u32{ 0x3f7a1d6d, 0x3f7309b9, 0x3f6a5dbe, 0x3f5f7df7, 0x3f51745c, 0x3f3ea365, 0x3f241a3e, 0x3ef7bde4, 0x3e5e9bb8, 0x36a7f7d3 };
    var steps: [10]Step = undefined;
    schedule(.{}, &steps);
    var total: u32 = 0;
    for (steps, want, end) |s, w, e| {
        try std.testing.expectEqual(fb(w.seg), s.segment);
        try std.testing.expectEqual(fb(w.tau), s.tau_src);
        try std.testing.expectEqual(fb(w.dst), s.tau_dst);
        try std.testing.expectEqual(fb(w.sig), s.sigma_src);
        try std.testing.expectEqual(fb(w.time), s.time);
        try std.testing.expectEqual(fb(w.rs), s.raw_src);
        try std.testing.expectEqual(fb(w.rd), s.raw_dst);
        try std.testing.expectEqual(fb(w.sp), s.policy_segment);
        try std.testing.expectEqual(w.n, s.substeps);
        try std.testing.expectEqual(fb(w.sub), s.substep_size);
        var buf: [32]Substep = undefined;
        const subs = substeps(.{}, s, &buf);
        try std.testing.expectEqual(fb(e), subs[subs.len - 1].sigma_next);
        const t0s: []const u8 = if (s.index == 9) "8765321" else "8876654432210";
        for (subs, t0s) |j, c| try std.testing.expectEqual(@as(u32, c - '0'), j.t0);
        total += s.substeps;
    }
    try std.testing.expectEqual(@as(u32, 124), total);
    // DXPolicy's raw_t_src is unwarp(sigma), not tau_src
    try std.testing.expect(steps[1].raw_src != steps[1].tau_src);
}

test "interpolation edges" {
    try std.testing.expectEqual(Weights{ .t0 = 8, .wa = 0, .wb = 1 }, weights(1.0, 10)); // t = 1: t0 clamped to n - 2
    try std.testing.expectEqual(Weights{ .t0 = 8, .wa = 0, .wb = 1 }, weights(1.5, 10));
    try std.testing.expectEqual(Weights{ .t0 = 0, .wa = 1, .wb = 0 }, weights(0.0, 10));
    try std.testing.expectEqual(Weights{ .t0 = 0, .wa = 1, .wb = 0 }, weights(-0.25, 10));
    try std.testing.expectEqual(Weights{ .t0 = 4, .wa = 0.5, .wb = 0.5 }, weights(0.5, 10));
    const w = weights(0.3, 10); // 0.3f * 9 = 2.7000000476837158
    try std.testing.expectEqual(@as(u32, 2), w.t0);
    try std.testing.expectEqual(@as(f32, 3.0) - @as(f32, 0.3) * 9.0, w.wa);
    try std.testing.expectEqual(@as(f32, 0.3) * 9.0 - 2.0, w.wb);
}

test "substep count per outer step: round half to even, at least one" {
    try std.testing.expectEqual(@as(f32, 2), roundHalfEven(2.5));
    try std.testing.expectEqual(@as(f32, 4), roundHalfEven(3.5));
    try std.testing.expectEqual(@as(f32, -2), roundHalfEven(-2.5));
    try std.testing.expectEqual(@as(f32, 3), roundHalfEven(2.5000002));
    // a single step: delta = 1 - eps, 128 substeps; a tiny final segment still takes one
    const one = step(.{ .steps = 1 }, 0, 1.0);
    try std.testing.expectEqual(@as(u32, 128), one.substeps);
    const tiny = step(.{ .steps = 1, .final_step_size_scale = 1e-3 }, 0, 1e-3);
    try std.testing.expectEqual(@as(u32, 1), tiny.substeps);
}

test "the fused rollout equals torch's op-by-op sequence" {
    // policy_rollout_fm and DXPolicy as whole-tensor ops (each a pass over the elements, the state promoted to f32
    // by the first where), against the per-element fusion the kernel runs
    const n = 61;
    var prng: std.Random.DefaultPrng = .init(0x6b36);
    const r = prng.random();
    var xb: [n]u16 = undefined;
    var pred: [n_grid * n]u16 = undefined;
    for (&xb) |*v| v.* = @truncate(@as(u32, @bitCast(r.floatNorm(f32))) >> 16);
    for (&pred) |*v| v.* = @truncate(@as(u32, @bitCast(r.floatNorm(f32) * 0.7)) >> 16);
    var steps: [10]Step = undefined;
    schedule(.{}, &steps);
    var state: [n]f32 = undefined;
    for (&state, xb) |*o, v| o.* = bf16ToF32(v);
    for (steps) |s| {
        var buf: [32]Substep = undefined;
        const subs = substeps(.{}, s, &buf);
        var fused: [n]f32 = undefined;
        if (s.index == 0) rollout(.{ .bf16 = &xb }, &pred, &fused, s, subs) else rollout(.{ .f32 = &state }, &pred, &fused, s, subs);
        // op by op
        const k = Consts.of(.{});
        var x0: [n_grid][n]f32 = undefined;
        var prod: [n_grid][n]f32 = undefined;
        for (0..n_grid) |g| for (0..n) |i| {
            prod[g][i] = s.sigma_src * bf16ToF32(pred[g * n + i]);
        };
        for (0..n_grid) |g| for (0..n) |i| {
            x0[g][i] = state[i] - prod[g][i];
        };
        var x = state;
        var sigma = s.sigma_src;
        var raw = s.tau_src;
        for (0..s.substeps) |_| {
            const raw_t = sigma / ((k.one_m_shift * sigma) + k.shift);
            const t = (raw_t - s.raw_dst) / s.policy_segment;
            const tc = clamp(t, 0, 1) * 9.0;
            const t0f = @min(@max(@floor(tc), 0.0), 8.0);
            const wa = (t0f + 1.0) - tc;
            const wb = tc - t0f;
            const t0: usize = @intFromFloat(t0f);
            var a: [n]f32 = undefined;
            var c: [n]f32 = undefined;
            for (&a, x0[t0]) |*o, v| o.* = wa * v;
            for (&c, x0[t0 + 1]) |*o, v| o.* = wb * v;
            var vel: [n]f32 = undefined;
            for (&vel, x, a, c) |*o, xv, av, cv| o.* = xv - (av + cv);
            for (&vel) |*o| o.* = o.* / clampMin(sigma, k.eps);
            const raw_m = clampMin(raw - s.substep_size, 0);
            const sigma_m = (k.shift * raw_m) / ((k.shift_m1 * raw_m) + 1.0);
            const ds = sigma_m - sigma;
            for (&x, vel) |*o, v| o.* = o.* + v * ds;
            sigma = sigma_m;
            raw = raw_m;
        }
        for (fused, x) |f, o| try std.testing.expectEqual(@as(u32, @bitCast(o)), @as(u32, @bitCast(f)));
        state = fused;
        // the kernel table carries the same scalars
        const t = Table.of(s, subs, 0);
        try std.testing.expectEqual(@as(i32, @intCast(s.substeps)), t.count);
        try std.testing.expectEqual(subs[1].ds, t.ds[1]);
    }
    for (state) |v| try std.testing.expect(std.math.isFinite(v));
    try std.testing.expectEqual(@as(usize, 4 + 4 + 5 * 4 * Table.capacity), @sizeOf(Table));
}
