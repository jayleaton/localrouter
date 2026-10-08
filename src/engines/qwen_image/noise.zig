//! The starting noise, as the twin draws it (`qwen_image.noise`): uniform i is splitmix64's output for state
//! seed + (i + 1) * golden, pair j takes uniforms 2j and 2j + 1 mapped to [-1, 1), and pairs with 0 < s < 1
//! (s = a^2 + b^2) give a * f, b * f with f = sqrt(-2 log(s) / s), all in f64 (`smath.log`), rounded to f32.

const std = @import("std");
const smath = @import("smath.zig");

const golden: u64 = 0x9E3779B97F4A7C15;

pub fn uniform(seed: u64, i: u64) f64 {
    var z = seed +% (i + 1) *% golden;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    z ^= z >> 31;
    return @as(f64, @floatFromInt(z >> 11)) * (1.0 / 9007199254740992.0);
}

/// Fills `out` with standard normals, in order.
pub fn fill(seed: u64, out: []f32) void {
    var k: usize = 0;
    var j: u64 = 0;
    while (k < out.len) : (j += 1) {
        const a = 2.0 * uniform(seed, 2 * j) - 1.0;
        const b = 2.0 * uniform(seed, 2 * j + 1) - 1.0;
        const s = a * a + b * b;
        if (!(s > 0 and s < 1)) continue;
        const f = @sqrt(-2.0 * smath.log(s) / s);
        out[k] = @floatCast(a * f);
        k += 1;
        if (k < out.len) {
            out[k] = @floatCast(b * f);
            k += 1;
        }
    }
}

test "noise is standard normal and prefix consistent" {
    var a: [4096]f32 = undefined;
    fill(42, &a);
    var b: [10]f32 = undefined;
    fill(42, &b);
    try std.testing.expectEqualSlices(f32, a[0..10], &b);
    var sum: f64 = 0;
    var sq: f64 = 0;
    for (a) |v| {
        sum += v;
        sq += @as(f64, v) * v;
    }
    try std.testing.expect(@abs(sum / a.len) < 0.05);
    try std.testing.expect(@abs(sq / a.len - 1.0) < 0.06);
}
