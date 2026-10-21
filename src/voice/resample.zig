//! Mono PCM16 helpers for the voice loop: windowed-sinc rate conversion (TTS output and STT input to and from the
//! 24 kHz wire format) and little-endian byte conversion.

const std = @import("std");

const zero_crossings = 16; // of the low-pass sinc, each side of the centre
const cutoff = 0.95; // share of the lower Nyquist kept

/// Resamples mono PCM16 from `from` Hz to `to` Hz: windowed-sinc interpolation low-passed below the lower Nyquist; equal rates copy.
pub fn resample(a: std.mem.Allocator, in: []const i16, from: u32, to: u32) ![]i16 {
    if (from == 0 or to == 0) return error.BadRate;
    if (from == to) return a.dupe(i16, in);
    const out_len: usize = @intCast((@as(u64, in.len) * to + from / 2) / from);
    const out = try a.alloc(i16, out_len);
    if (in.len == 0) return out;
    // The cutoff in cycles per input sample, and the kernel's half width in input samples.
    const fc: f64 = cutoff * @as(f64, @floatFromInt(@min(from, to))) / 2 / @as(f64, @floatFromInt(from));
    const half: f64 = zero_crossings / (2 * fc);
    const step: f64 = @as(f64, @floatFromInt(from)) / @as(f64, @floatFromInt(to));
    const last: i64 = @intCast(in.len - 1);
    for (out, 0..) |*o, i| {
        const t: f64 = @as(f64, @floatFromInt(i)) * step; // the output sample's position in the input
        const lo: i64 = @max(@as(i64, @intFromFloat(@ceil(t - half))), 0);
        const hi: i64 = @min(@as(i64, @intFromFloat(@floor(t + half))), last);
        var acc: f64 = 0;
        var wsum: f64 = 0;
        var k = lo;
        while (k <= hi) : (k += 1) {
            const d: f64 = t - @as(f64, @floatFromInt(k));
            const w = tap(d, fc, half);
            acc += w * @as(f64, @floatFromInt(in[@intCast(k)]));
            wsum += w;
        }
        const y = if (@abs(wsum) > 1e-6) acc / wsum else acc;
        o.* = @intFromFloat(std.math.clamp(@round(y), -32768, 32767));
    }
    return out;
}

/// The low-pass kernel at `d` input samples from the centre: sinc at cutoff `fc` under a Blackman window.
fn tap(d: f64, fc: f64, half: f64) f64 {
    const x = d / half; // -1 to 1 across the window
    if (x <= -1 or x >= 1) return 0;
    const c = @cos(std.math.pi * x);
    const window = 0.42 + 0.5 * c + 0.08 * (2 * c * c - 1); // cos(2 pi x) = 2 cos(pi x)^2 - 1
    const arg = 2 * fc * d;
    const sinc: f64 = if (@abs(arg) < 1e-9) 1 else @sin(std.math.pi * arg) / (std.math.pi * arg);
    return 2 * fc * sinc * window;
}

/// Little-endian 16-bit bytes as samples; error.OddLength for an odd byte count.
pub fn fromLe(a: std.mem.Allocator, bytes: []const u8) ![]i16 {
    if (bytes.len % 2 != 0) return error.OddLength;
    const out = try a.alloc(i16, bytes.len / 2);
    for (out, 0..) |*s, i| s.* = std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little);
    return out;
}

/// Samples as little-endian 16-bit bytes.
pub fn toLe(a: std.mem.Allocator, samples: []const i16) ![]u8 {
    const out = try a.alloc(u8, samples.len * 2);
    for (samples, 0..) |s, i| std.mem.writeInt(i16, out[i * 2 ..][0..2], s, .little);
    return out;
}

/// One second of a `hz` tone at `rate` Hz with amplitude 10000.
fn sine(a: std.mem.Allocator, rate: u32, hz: f64) ![]i16 {
    const out = try a.alloc(i16, rate);
    for (out, 0..) |*s, i| {
        const t: f64 = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(rate));
        s.* = @intFromFloat(@round(10000 * @sin(2 * std.math.pi * hz * t)));
    }
    return out;
}

/// The RMS of `s` without its first and last 5%, away from the edges.
fn rms(s: []const i16) f64 {
    const mid = s[s.len / 20 .. s.len - s.len / 20];
    var acc: f64 = 0;
    for (mid) |x| {
        const f: f64 = @floatFromInt(x);
        acc += f * f;
    }
    return @sqrt(acc / @as(f64, @floatFromInt(mid.len)));
}

fn crossings(s: []const i16) usize {
    var n: usize = 0;
    for (s[1..], s[0 .. s.len - 1]) |x, prev| {
        if ((x >= 0) != (prev >= 0)) n += 1;
    }
    return n;
}

test "lengths round to the nearest sample" {
    const a = std.testing.allocator;
    const zeros: [24_000]i16 = @splat(0);
    const cases = [_]struct { n: usize, from: u32, to: u32, want: usize }{
        .{ .n = 24_000, .from = 24_000, .to = 16_000, .want = 16_000 },
        .{ .n = 22_050, .from = 22_050, .to = 24_000, .want = 24_000 },
        .{ .n = 1001, .from = 22_050, .to = 24_000, .want = 1090 }, // 1089.52
        .{ .n = 1000, .from = 24_000, .to = 22_050, .want = 919 }, // 918.75
        .{ .n = 0, .from = 24_000, .to = 16_000, .want = 0 },
    };
    for (cases) |c| {
        const out = try resample(a, zeros[0..c.n], c.from, c.to);
        defer a.free(out);
        try std.testing.expectEqual(c.want, out.len);
        for (out) |s| try std.testing.expectEqual(@as(i16, 0), s);
    }
    try std.testing.expectError(error.BadRate, resample(a, &zeros, 0, 16_000));
}

test "equal rates are an exact copy" {
    const a = std.testing.allocator;
    const in = [_]i16{ 1, -2, 32767, -32768, 0, 5 };
    const out = try resample(a, &in, 24_000, 24_000);
    defer a.free(out);
    try std.testing.expectEqualSlices(i16, &in, out);
}

test "a 440 Hz tone keeps its frequency and level, down and up" {
    const a = std.testing.allocator;
    const in = try sine(a, 24_000, 440);
    defer a.free(in);
    for ([_]u32{ 16_000, 22_050, 48_000 }) |to| {
        const out = try resample(a, in, 24_000, to);
        defer a.free(out);
        try std.testing.expectEqual(@as(usize, to), out.len);
        const z = crossings(out);
        try std.testing.expect(z >= 878 and z <= 882); // 440 cycles in one second
        const ratio = rms(out) / rms(in);
        try std.testing.expect(ratio > 0.97 and ratio < 1.03);
    }
}

test "a tone above the target Nyquist is filtered out" {
    const a = std.testing.allocator;
    const in = try sine(a, 24_000, 10_000);
    defer a.free(in);
    const out = try resample(a, in, 24_000, 16_000);
    defer a.free(out);
    try std.testing.expect(rms(out) < 0.1 * rms(in));
}

test "little-endian bytes round trip" {
    const a = std.testing.allocator;
    const bytes = [_]u8{ 0x01, 0x00, 0xff, 0x7f, 0x00, 0x80, 0xfe, 0xff };
    const s = try fromLe(a, &bytes);
    defer a.free(s);
    try std.testing.expectEqualSlices(i16, &.{ 1, 32767, -32768, -2 }, s);
    const back = try toLe(a, s);
    defer a.free(back);
    try std.testing.expectEqualSlices(u8, &bytes, back);
    try std.testing.expectError(error.OddLength, fromLe(a, bytes[0..3]));
}
