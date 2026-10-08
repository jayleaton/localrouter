//! Qwen-Image 2.1's 3-axis RoPE tables on the host, as the twin builds them (`dit.rope_table`, `dit.image_ids`):
//! ids [N, 3] (frame / position, row, column); angle = id x frozen omega in f64, the shared sin / cos, rounded to f32.

const std = @import("std");
const smath = @import("smath.zig");

pub const axes = [3]usize{ 16, 56, 56 };
pub const half: usize = (axes[0] + axes[1] + axes[2]) / 2; // 64 = head_dim / 2

/// The text prefix's ids: position p on all three axes.
pub fn textIds(out: [][3]f32, first: usize) void {
    for (out, first..) |*r, p| r.* = @splat(@floatFromInt(p));
}

/// A latent image's ids at `pos` on axis 0, centred rows and columns (half-pixel shifted for odd sides).
pub fn imageIds(out: [][3]f32, h: usize, w: usize, pos: usize, target_h: usize, target_w: usize) void {
    const fh: f32 = @floatFromInt(h);
    const fw: f32 = @floatFromInt(w);
    const sh = -(fh - @as(f32, @floatFromInt(h / 2))) + 0.5 * (@as(f32, @floatFromInt(h % 2)) - @as(f32, @floatFromInt(target_h % 2)));
    const sw = -(fw - @as(f32, @floatFromInt(w / 2))) + 0.5 * (@as(f32, @floatFromInt(w % 2)) - @as(f32, @floatFromInt(target_w % 2)));
    var i: usize = 0;
    for (0..h) |y| {
        for (0..w) |x| {
            out[i] = .{ @floatFromInt(pos), @as(f32, @floatFromInt(y)) + sh, @as(f32, @floatFromInt(x)) + sw };
            i += 1;
        }
    }
}

/// cos and sin [N, 64] f32 for `ids` with the frozen frequencies `om` (`smath.omegas`); outputs hold N * 64.
pub fn table(ids: []const [3]f32, om: *const [half]f64, cos_out: []f32, sin_out: []f32) void {
    var c: usize = 0;
    for (ids) |id| {
        var col: usize = 0;
        for (axes, 0..) |d, ax| {
            for (0..d / 2) |_| {
                const r = smath.sincos(@as(f64, id[ax]) * om[col]);
                cos_out[c] = @floatCast(r.cos);
                sin_out[c] = @floatCast(r.sin);
                c += 1;
                col += 1;
            }
        }
    }
}

test "image ids centre a 4x3 grid" {
    var ids: [12][3]f32 = undefined;
    imageIds(&ids, 4, 3, 7, 4, 3);
    try std.testing.expectEqual([3]f32{ 7, -2, -2 }, ids[0]);
    try std.testing.expectEqual([3]f32{ 7, 1, 0 }, ids[11]);
}
