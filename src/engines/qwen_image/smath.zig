//! Portable sin / cos for the RoPE tables: the same IEEE double operations, in the same order, as the twin's
//! `stk_twin/smath.py` (Cody-Waite reduction by pi/2, fdlibm's kernel polynomials), so both get the same bits.

const std = @import("std");

const invpio2 = 6.36619772367581382433e-01;
const pio2_1 = 1.57079632673412561417e+00; // first 33 bits of pi/2: n * pio2_1 is exact for |n| < 2^20
const pio2_2 = 6.07710050630396597660e-11;
const pio2_3 = 2.02226624871116645580e-21;
const s1 = -1.66666666666666324348e-01;
const s2 = 8.33333333332248946124e-03;
const s3 = -1.98412698298579493134e-04;
const s4 = 2.75573137070700676789e-06;
const s5 = -2.50507602534068634195e-08;
const s6 = 1.58969099521155010221e-10;
const c1 = 4.16666666666666019037e-02;
const c2 = -1.38888888888741095749e-03;
const c3 = 2.48015872894767294178e-05;
const c4 = -2.75573143513906633035e-07;
const c5 = 2.08757232129817482790e-09;
const c6 = -1.13596475577881948265e-11;

fn ksin(x: f64) f64 {
    const z = x * x;
    const v = z * x;
    const r = s2 + z * (s3 + z * (s4 + z * (s5 + z * s6)));
    return x + v * (s1 + z * r);
}

fn kcos(x: f64) f64 {
    const z = x * x;
    const r = z * (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * c6)))));
    const hz = 0.5 * z;
    const w = 1.0 - hz;
    return w + (((1.0 - w) - hz) + z * r);
}

pub const SinCos = struct { sin: f64, cos: f64 };

pub fn sincos(x: f64) SinCos {
    const fn_ = @floor(x * invpio2 + 0.5);
    const n: i64 = @intFromFloat(fn_);
    const r = ((x - fn_ * pio2_1) - fn_ * pio2_2) - fn_ * pio2_3;
    const s = ksin(r);
    const c = kcos(r);
    return switch (@as(u2, @truncate(@as(u64, @bitCast(n))))) {
        0 => .{ .sin = s, .cos = c },
        1 => .{ .sin = c, .cos = -s },
        2 => .{ .sin = -s, .cos = -c },
        3 => .{ .sin = -c, .cos = s },
    };
}

const ln2_hi = 6.93147180369123816490e-01;
const ln2_lo = 1.90821492927058770002e-10;
const two54 = 1.80143985094819840000e+16;
const lg1 = 6.666666666666735130e-01;
const lg2 = 3.999999999940941908e-01;
const lg3 = 2.857142874366239149e-01;
const lg4 = 2.222219843214978396e-01;
const lg5 = 1.818357216161805012e-01;
const lg6 = 1.531383769920937332e-01;
const lg7 = 1.479819860511658591e-01;

fn high(x: f64) i32 {
    return @bitCast(@as(u32, @truncate(@as(u64, @bitCast(x)) >> 32)));
}

fn withHigh(x: f64, hi: i32) f64 {
    const lo = @as(u64, @bitCast(x)) & 0xFFFFFFFF;
    return @bitCast((@as(u64, @as(u32, @bitCast(hi))) << 32) | lo);
}

/// fdlibm's __ieee754_log for finite x > 0, operation for operation (`smath.py` `log` is the same).
pub fn log(x_in: f64) f64 {
    var x = x_in;
    var hx = high(x);
    var k: i32 = 0;
    if (hx < 0x00100000) { // subnormal
        k -= 54;
        x *= two54;
        hx = high(x);
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000FFFFF;
    var i: i32 = (hx + 0x95F64) & 0x100000;
    x = withHigh(x, hx | (i ^ 0x3FF00000));
    k += i >> 20;
    const f = x - 1.0;
    const dk: f64 = @floatFromInt(k);
    if ((0x000FFFFF & (2 + hx)) < 3) {
        if (f == 0.0) return if (k == 0) 0.0 else dk * ln2_hi + dk * ln2_lo;
        const r = f * f * (0.5 - 0.33333333333333333 * f);
        return if (k == 0) f - r else dk * ln2_hi - ((r - dk * ln2_lo) - f);
    }
    const s = f / (2.0 + f);
    const z = s * s;
    i = hx - 0x6147A;
    const w = z * z;
    const j: i32 = 0x6B851 - hx;
    const t1 = w * (lg2 + w * (lg4 + w * lg6));
    const t2 = z * (lg1 + w * (lg3 + w * (lg5 + w * lg7)));
    i |= j;
    const r = t2 + t1;
    if (i > 0) {
        const hfsq = 0.5 * f * f;
        return if (k == 0) f - (hfsq - s * (hfsq + r)) else dk * ln2_hi - ((hfsq - (s * (hfsq + r) + dk * ln2_lo)) - f);
    }
    return if (k == 0) f - s * (f - r) else dk * ln2_hi - ((s * (f - r) - dk * ln2_lo) - f);
}

/// The 64 RoPE frequencies, frozen as f64 bit patterns in `kernels/qwen_image/rope_omega.json`.
pub fn omegas(json: []const u8) ![64]f64 {
    var buf: [64 * 1024]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&buf);
    const J = struct { omega_f64_be_hex: [][]const u8 };
    const j = try std.json.parseFromSliceLeaky(J, fba.allocator(), json, .{ .ignore_unknown_fields = true });
    if (j.omega_f64_be_hex.len != 64) return error.BadOmegaTable;
    var out: [64]f64 = undefined;
    for (j.omega_f64_be_hex, &out) |h, *o| {
        var b: [8]u8 = undefined;
        _ = try std.fmt.hexToBytes(&b, h);
        o.* = @bitCast(std.mem.readInt(u64, &b, .big));
    }
    return out;
}

test "sincos is accurate" {
    var x: f64 = -3000.0;
    while (x < 3000.0) : (x += 0.7317) {
        const r = sincos(x);
        try std.testing.expectApproxEqAbs(@sin(x), r.sin, 4e-15);
        try std.testing.expectApproxEqAbs(@cos(x), r.cos, 4e-15);
    }
}

// fdlibm e_exp.c / s_expm1.c constants
const half_s = [2]f64{ 0.5, -0.5 };
const ln2hi_s = [2]f64{ 6.93147180369123816490e-01, -6.93147180369123816490e-01 };
const ln2lo_s = [2]f64{ 1.90821492927058770002e-10, -1.90821492927058770002e-10 };
const invln2: f64 = 1.44269504088896338700e+00;
const p1: f64 = 1.66666666666666019037e-01;
const p2: f64 = -2.77777777770155933842e-03;
const p3: f64 = 6.61375632143793436117e-05;
const p4: f64 = -1.65339022054652515390e-06;
const p5: f64 = 4.13813679705723846039e-08;
const q1: f64 = -3.33333333333331316428e-02;
const q2: f64 = 1.58730158725481460165e-03;
const q3: f64 = -7.93650757867487942473e-05;
const q4: f64 = 4.00821782732936239552e-06;
const q5: f64 = -2.01099218183624371326e-07;

fn addExp(y: f64, k: i32) f64 {
    return withHigh(y, high(y) + (k << 20));
}

/// fdlibm's __ieee754_exp for |x| < 700, operation for operation (`smath.py` `exp` is the same).
pub fn exp(x_in: f64) f64 {
    var x = x_in;
    var hx = high(x);
    const xsb: usize = @intCast((hx >> 31) & 1);
    hx &= 0x7FFFFFFF;
    std.debug.assert(hx < 0x4086232B);
    var k: i32 = 0;
    var hi: f64 = 0;
    var lo: f64 = 0;
    if (hx > 0x3FD62E42) {
        if (hx < 0x3FF0A2B2) {
            hi = x - ln2hi_s[xsb];
            lo = ln2lo_s[xsb];
            k = 1 - @as(i32, @intCast(xsb)) - @as(i32, @intCast(xsb));
        } else {
            k = @intFromFloat(invln2 * x + half_s[xsb]);
            const t: f64 = @floatFromInt(k);
            hi = x - t * ln2hi_s[0];
            lo = t * ln2lo_s[0];
        }
        x = hi - lo;
    } else if (hx < 0x3E300000) {
        return 1.0 + x;
    }
    const t = x * x;
    const c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))));
    if (k == 0) return 1.0 - ((x * c) / (c - 2.0) - x);
    const y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    std.debug.assert(k >= -1021);
    return addExp(y, k);
}

/// fdlibm's expm1 for |x| < 56 ln 2, operation for operation (`smath.py` `expm1` is the same).
pub fn expm1(x_in: f64) f64 {
    var x = x_in;
    var hx = high(x);
    const neg = (hx & @as(i32, @bitCast(@as(u32, 0x80000000)))) != 0;
    hx &= 0x7FFFFFFF;
    std.debug.assert(hx < 0x4043687A);
    var k: i32 = 0;
    var c: f64 = 0;
    if (hx > 0x3FD62E42) {
        var hi: f64 = undefined;
        var lo: f64 = undefined;
        if (hx < 0x3FF0A2B2) {
            if (!neg) {
                hi = x - ln2hi_s[0];
                lo = ln2lo_s[0];
                k = 1;
            } else {
                hi = x + ln2hi_s[0];
                lo = -ln2lo_s[0];
                k = -1;
            }
        } else {
            k = @intFromFloat(invln2 * x + (if (!neg) @as(f64, 0.5) else -0.5));
            const t: f64 = @floatFromInt(k);
            hi = x - t * ln2hi_s[0];
            lo = t * ln2lo_s[0];
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if (hx < 0x3C900000) {
        return x;
    }
    const hfx = 0.5 * x;
    const hxs = x * hfx;
    const r1 = 1.0 + hxs * (q1 + hxs * (q2 + hxs * (q3 + hxs * (q4 + hxs * q5))));
    var t = 3.0 - r1 * hfx;
    var e = hxs * ((r1 - t) / (6.0 - x * t));
    if (k == 0) return x - (x * e - hxs);
    e = x * (e - c) - c;
    e -= hxs;
    if (k == -1) return 0.5 * (x - e) - 0.5;
    if (k == 1) return if (x < -0.25) -2.0 * (e - (x + 0.5)) else 1.0 + 2.0 * (x - e);
    if (k <= -2 or k > 56) return addExp(1.0 - (e - x), k) - 1.0;
    if (k < 20) {
        t = withHigh(1.0, 0x3FF00000 - (@as(i32, 0x200000) >> @intCast(k)));
        return addExp(t - (e - x), k);
    }
    t = withHigh(1.0, (0x3FF - k) << 20);
    var y = x - (e + t);
    y += 1.0;
    return addExp(y, k);
}

test "log, exp and expm1 equal the twin's (smath.py) bit for bit" {
    const Row = struct { x: u64, exp: u64, expm1: u64, log: ?u64 = null };
    const rows = try std.json.parseFromSlice([]Row, std.testing.allocator, @embedFile("smath_table.json"), .{});
    defer rows.deinit();
    for (rows.value) |r| {
        const x: f64 = @bitCast(r.x);
        try std.testing.expectEqual(r.exp, @as(u64, @bitCast(exp(x))));
        try std.testing.expectEqual(r.expm1, @as(u64, @bitCast(expm1(x))));
        if (r.log) |l| try std.testing.expectEqual(l, @as(u64, @bitCast(log(x))));
    }
}
