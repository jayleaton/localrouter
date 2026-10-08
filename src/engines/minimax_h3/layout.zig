//! The host side of an H3 step, as the twin computes it (`stk_twin/h3/dit.py`): the packed layout [text | audio |
//! video] and its fp64 positions, the RoPE rotation table (bf16, the shared portable sincos), every scalar a step
//! derives from the sampler's sigma (ComfyUI's fp32 tensor arithmetic, op by op), the modulation row of every token,
//! and the curve form's time embedding (torch.lerp's two-branch formula, unfused). Zig does not contract floats, so
//! each f32 expression below rounds where the twin's numpy float32 does.

const std = @import("std");
const smath = @import("qwen_image").smath;

pub const shift_v: f32 = 12.0;
pub const shift_a: f32 = 3.0;
const frame_per_token = [5]f64{ 1, 4, 4, 4, 4 };
const frame_rescale: f64 = 5.0 / 3.0;

/// ComfyUI's time_shift_sigma on an fp32 value: base = s / (from + s * (1 - from)), then to * base / (1 + (to - 1) * base).
pub fn timeShiftSigma(s: f32, from: f32, to: f32) f32 {
    const base = s / (from + s * (1.0 - from));
    return (to * base) / (1.0 + (to - 1.0) * base);
}

pub const Scalars = struct { sigma_v: f32, sigma_a: f32, t_v: f32, t_a: f32, carry: f32, c1: f32, c2: f32 };

/// bf16 round to nearest even of an f32, back as an f32 (finite inputs).
pub fn bf16Round(v: f32) f32 {
    const b: u32 = @bitCast(v);
    const r: u32 = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000;
    return @bitCast(r);
}

pub fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @truncate((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

pub fn stepScalars(sigma: f32) Scalars {
    const timestep = sigma * 1000.0;
    const sigma_v = @max(timestep / 1000.0, 1e-6);
    const sigma_a = timeShiftSigma(sigma_v, shift_v, shift_a);
    const scale = shift_v / shift_a;
    return .{
        .sigma_v = sigma_v,
        .sigma_a = sigma_a,
        .t_v = 1.0 - sigma_v,
        .t_a = 1.0 - sigma_a,
        .carry = bf16Round(sigma_a / sigma_v),
        .c1 = 1.0 - scale,
        .c2 = bf16Round(1.0 + (scale - 1.0) * sigma_a),
    };
}

pub const Layout = struct {
    l: u32,
    t: u32,
    h: u32,
    w: u32,
    a: u32,
    s: u32,

    pub fn init(l: u32, t: u32, h: u32, w: u32, a: u32) Layout {
        return .{ .l = l, .t = t, .h = h, .w = w, .a = a, .s = l + 2 * a + t * (h / 2) * (w / 2) };
    }

    pub fn audioRows(x: Layout) [2]u32 {
        return .{ x.l, x.l + 2 * x.a };
    }
    pub fn videoRows(x: Layout) [2]u32 {
        return .{ x.l + 2 * x.a, x.s };
    }

    fn axis(dim: u32, area: f64, i: u32) f64 {
        const ratio = @as(f64, @floatFromInt(dim)) / area;
        const n: f64 = @floatFromInt(dim / 2);
        return (@as(f64, @floatFromInt(i)) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
    }

    /// pos [S, 3] f64 (t, h, w), as ComfyUI's PackedLayout.
    pub fn positions(x: Layout, out: [][3]f64) void {
        std.debug.assert(out.len == x.s);
        const area = @sqrt(@as(f64, @floatFromInt(x.h * x.w)));
        const w_lo = axis(x.w, area, 0);
        const w_hi = axis(x.w, area, x.w / 2 - 1);
        var r: usize = 0;
        for (0..x.l) |i| {
            out[r] = .{ @floatFromInt(i), 0, 0 };
            r += 1;
        }
        const cursor: f64 = @floatFromInt(x.l);
        for (0..2) |ch| for (0..x.a) |i| {
            out[r] = .{ cursor + @as(f64, @floatFromInt(i)), 0, if (ch == 0) w_lo else w_hi };
            r += 1;
        };
        var cum: f64 = 0; // the spans' exclusive cumsum (f64, in order), then + cursor, as numpy does
        for (0..x.t) |k| {
            const tpos = cursor + cum;
            for (0..x.h / 2) |hh| for (0..x.w / 2) |ww| {
                out[r] = .{ tpos, axis(x.h, area, @intCast(hh)), axis(x.w, area, @intCast(ww)) };
                r += 1;
            };
            cum += frame_rescale * frame_per_token[k % 5];
        }
    }

    /// The rotation table [S, 48, 2, 2] bf16 bits: angle = f32(pos) * inv_freq (f32), cos / sin of it by the
    /// portable sincos in f64, rounded to f32, then bf16; [c, -s, s, c].
    pub fn ropeTable(x: Layout, pos: []const [3]f64, inv: *const [16]f32, out: []u16) void {
        std.debug.assert(out.len == @as(usize, x.s) * 48 * 4);
        for (pos, 0..) |p, r| {
            for (0..3) |ax| for (0..16) |j| {
                const ang: f32 = @as(f32, @floatCast(p[ax])) * inv[j];
                const sc = smath.sincos(@as(f64, ang));
                const c: f32 = @floatCast(sc.cos);
                const s: f32 = @floatCast(sc.sin);
                const o = (r * 48 + ax * 16 + j) * 4;
                out[o] = bf16Bits(c);
                out[o + 1] = bf16Bits(-s);
                out[o + 2] = bf16Bits(s);
                out[o + 3] = bf16Bits(c);
            };
        }
    }

    /// The unique timesteps (ascending) and each token's modulation row t_row * 3 + tag (video 0, text 1, audio 2);
    /// returns how many unique values and the final layer's rows for video and audio.
    pub fn modRows(x: Layout, sc: Scalars, unique: *[2]f32, idx: []i32) struct { m: u32, row_v: u32, row_a: u32 } {
        var m: u32 = 1;
        unique[0] = @min(sc.t_v, sc.t_a);
        if (sc.t_v != sc.t_a) {
            unique[1] = @max(sc.t_v, sc.t_a);
            m = 2;
        }
        const row_v: u32 = if (sc.t_v == unique[0]) 0 else 1;
        const row_a: u32 = if (sc.t_a == unique[0]) 0 else 1;
        const a = x.audioRows();
        @memset(idx[0..x.l], @intCast(row_v * 3 + 1));
        @memset(idx[a[0]..a[1]], @intCast(row_a * 3 + 2));
        @memset(idx[a[1]..x.s], @intCast(row_v * 3 + 0));
        return .{ .m = m, .row_v = row_v, .row_a = row_a };
    }
};

/// The curve form's t_emb [m, k] f32: torch.lerp(table[i0], table[i0 + 1], w) with pos = clamp(t, 0, 1) * (grid - 1).
pub fn curveTEmb(table: []const f32, grid: usize, k: usize, t: []const f32, out: []f32) void {
    for (t, 0..) |tv, r| {
        const pos = std.math.clamp(tv, 0.0, 1.0) * @as(f32, @floatFromInt(grid - 1));
        const lo: usize = @min(@as(usize, @intFromFloat(@floor(pos))), grid - 2);
        const w = pos - @as(f32, @floatFromInt(lo));
        for (0..k) |j| {
            const a = table[lo * k + j];
            const b = table[(lo + 1) * k + j];
            out[r * k + j] = if (w < 0.5) a + w * (b - a) else b - (b - a) * (1.0 - w);
        }
    }
}

test "step 0: sigma 1 gives one timestep and a unit carry" {
    const sc = stepScalars(1.0);
    try std.testing.expectEqual(@as(f32, 0.0), sc.t_v);
    try std.testing.expectEqual(@as(f32, 0.0), sc.t_a);
    try std.testing.expectEqual(@as(f32, 1.0), sc.carry);
    try std.testing.expectEqual(@as(f32, -3.0), sc.c1);
    try std.testing.expectEqual(@as(f32, 4.0), sc.c2);
}

test "layout rows: 768x448, 56 frames" {
    const x = Layout.init(69, 17, 28, 48, 93);
    try std.testing.expectEqual(@as(u32, 69 + 186 + 17 * 14 * 24), x.s);
}
