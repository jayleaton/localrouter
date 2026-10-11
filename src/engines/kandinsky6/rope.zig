//! The DiT's position tables, built as upstream builds them (`core/components/rope.py`, `core/tensors.py:get_freqs`,
//! `pipeline/pipeline.py:146-188`, `core/components/dit.py:45`): the frequencies and the argument tables on the host,
//! cos / sin on the GPU (central.cu `rope_cossin`, libdevice like torch's CUDA cos / sin).
//!
//! The one value not derivable from first principles is `get_freqs`: upstream evaluates `torch.exp` on the CPU, which is
//! SLEEF's vectorized expf on the reference machine (aarch64 NEON there), accurate to 1 ulp but not correctly rounded.
//! The frequencies therefore come from a frozen table captured from the native reference on the target
//! (`stk_twin.kandinsky6.capture --stages noise`, `freqs/*`); until that table is committed, `provisional` is true, the
//! values are this file's correctly rounded `exp`, and the engine reports itself unqualified.
//! Every other step is exact f32 arithmetic in upstream's order: `outer(arange(max_pos), freq)` is one product per
//! element (CPU f32), `freq * 0.144` (audio) one product, and the visual `args / scale_factor` divides by a CPU scalar
//! (multiplication by the f32 reciprocal on CUDA, exact for the scale factors 1 and 2).

const std = @import("std");
const shapes = @import("shapes.zig");

/// True while the frequencies are not the frozen native capture (see the file comment).
pub const provisional = true;

pub const max_period: f32 = 10000.0;
pub const head_dim: u32 = 128;
pub const pairs: u32 = head_dim / 2;
/// RoPE3D axes_dims (transformer_pro.json): t 32, h 48, w 48 -> 16 + 24 + 24 pairs.
pub const axes_pairs = [3]u32{ 16, 24, 24 };
/// pro-distill.yaml generation.scale_factor (t, h, w).
pub const scale_factor = [3]f32{ 1.0, 2.0, 2.0 };
pub const audio_freqs_scaling: f32 = 0.144;
/// TimeEmbeddings(model_dim) frequencies: get_freqs(model_dim // 2) for the video stack, model_dim_a // 2 for audio.
pub const time_freqs_v: u32 = 4096 / 2;
pub const time_freqs_a: u32 = 2048 / 2;

/// `get_freqs(dim)`: exp(-ln(max_period) * arange(dim) / dim) in f32 (f32(-ln 10000) times i, divided by dim, exp).
pub fn freqs(out: []f32) void {
    const c: f32 = @floatCast(-@log(@as(f64, max_period)));
    const dim: f32 = @floatFromInt(out.len);
    for (out, 0..) |*o, i| {
        const fi: f32 = @floatFromInt(i);
        o.* = @exp((c * fi) / dim);
    }
}

/// Visual argument rows [T * H * W][64] (t, h, w row-major at patch level): per axis `arange(n)` times that axis'
/// frequencies, divided by the scale factor, concatenated t | h | w.
pub fn visualArgs(t: u32, h: u32, w: u32, ft: []const f32, fh: []const f32, fw: []const f32, out: []f32) void {
    std.debug.assert(ft.len == axes_pairs[0] and fh.len == axes_pairs[1] and fw.len == axes_pairs[2]);
    std.debug.assert(out.len == @as(usize, t) * h * w * pairs);
    const inv = [3]f32{ 1.0 / scale_factor[0], 1.0 / scale_factor[1], 1.0 / scale_factor[2] };
    var r: usize = 0;
    for (0..t) |ti| for (0..h) |hi| for (0..w) |wi| {
        const row = out[r * pairs ..][0..pairs];
        const pos = [3]f32{ @floatFromInt(ti), @floatFromInt(hi), @floatFromInt(wi) };
        var j: usize = 0;
        for ([3][]const f32{ ft, fh, fw }, 0..) |f, a| for (f) |fr| {
            row[j] = (pos[a] * fr) * inv[a];
            j += 1;
        };
        r += 1;
    };
}

/// 1-D argument rows [n][64] for positions 0 .. n - 1 with frequencies `f` (already scaled for audio).
pub fn linearArgs(f: []const f32, out: []f32) void {
    std.debug.assert(f.len == pairs and out.len % pairs == 0);
    for (0..out.len / pairs) |p| {
        const pos: f32 = @floatFromInt(p);
        for (f, 0..) |fr, j| out[p * pairs + j] = pos * fr;
    }
}

/// The audio RoPE's frequencies: `get_freqs(64) * 0.144` (one f32 product each).
pub fn audioFreqs(base: []const f32, out: []f32) void {
    for (base, out) |b, *o| o.* = b * audio_freqs_scaling;
}

test "frequencies start at 1 and decrease" {
    var f: [64]f32 = undefined;
    freqs(&f);
    try std.testing.expectEqual(@as(f32, 1.0), f[0]);
    for (1..64) |i| try std.testing.expect(f[i] < f[i - 1]);
}

test "visual args: t | h | w per row, h and w halved" {
    var ft: [16]f32 = undefined;
    var fh: [24]f32 = undefined;
    freqs(&ft);
    freqs(&fh);
    var out: [2 * 3 * 4 * 64]f32 = undefined;
    visualArgs(2, 3, 4, &ft, &fh, &fh, &out);
    const last = out[(2 * 3 * 4 - 1) * 64 ..][0..64]; // t 1, h 2, w 3
    try std.testing.expectEqual(ft[5], last[5]);
    try std.testing.expectEqual(2.0 * fh[7] * 0.5, last[16 + 7]);
    try std.testing.expectEqual(3.0 * fh[0] * 0.5, last[40]);
    try std.testing.expectEqual(@as(usize, shapes.visual_tokens * 64), @as(usize, shapes.latent_frames) * shapes.patches_h * shapes.patches_w * 64);
}
