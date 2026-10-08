//! Host-side weight layouts TensorFold v0.6.1's prompt GEMMs read (nvfp4/linear.py), from a pack's checkpoint layout:
//!   fp8Order:   e4m3 [N, K] -> qmm_prefill8w's order [npad/64][K/64][8][32][2][8] (`_fragment_order`)
//!   nvfp4Scales: e4m3 block scales [N, K/16] -> [npad/64][K/64][64][4] (`Fp4Linear.from_checkpoint`'s `bs`)
//! (NVFP4 codes go through the `pack4` kernel on the GPU.) Each is one pass over the output.

const std = @import("std");

/// Position in a k32 step of byte `b` of lane `l`, mapped to `quantize_rows`' input order (`fragment_index`'s kin).
fn kinTable() [32][8]u8 {
    var t: [32][8]u8 = undefined;
    for (0..32) |lane| {
        for (0..8) |byte| {
            const p = 16 * (byte / 4) + 4 * (lane % 4) + (byte % 4);
            t[lane][byte] = @intCast(16 * (p / 16) + 2 * ((p % 16) / 4) + (p % 2) + 8 * ((p % 4) / 2));
        }
    }
    return t;
}

/// `out` (npad * k bytes) <- codes [n, k] in fragment order; rows n..npad are zero.
pub fn fp8Order(codes: []const u8, n: usize, k: usize, npad: usize, out: []u8) void {
    std.debug.assert(npad % 64 == 0);
    fp8OrderBlocks(codes, n, k, 0, npad / 64, out);
}

/// 64-row blocks [b0, b1) of `fp8Order`'s output into `out` ((b1 - b0) * 64 * k bytes): blocks are independent, so
/// the loader fills them in parallel, straight into its upload chunks.
pub fn fp8OrderBlocks(codes: []const u8, n: usize, k: usize, b0: usize, b1: usize, out: []u8) void {
    std.debug.assert(codes.len == n * k and out.len == (b1 - b0) * 64 * k and k % 64 == 0);
    const kin = comptime kinTable();
    const kg = k / 64;
    var i: usize = 0;
    for (b0..b1) |nb| for (0..kg) |g| for (0..8) |a| for (0..32) |lane| for (0..2) |h| {
        const row = nb * 64 + a * 8 + lane / 4;
        const col0 = g * 64 + h * 32;
        for (kin[lane]) |kk| {
            out[i] = if (row < n) codes[row * k + col0 + kk] else 0;
            i += 1;
        }
    };
}

/// `out` (npad * k / 16 bytes) <- scales [n, k / 16] as [npad/64][k/64][64][4]; rows n..npad are zero.
pub fn nvfp4Scales(scales: []const u8, n: usize, k: usize, npad: usize, out: []u8) void {
    const ks = k / 16;
    std.debug.assert(scales.len == n * ks and out.len == npad * ks and k % 64 == 0 and npad % 64 == 0);
    var i: usize = 0;
    for (0..npad / 64) |nb| for (0..k / 64) |g| for (0..64) |r| {
        const row = nb * 64 + r;
        for (0..4) |c| {
            out[i] = if (row < n) scales[row * ks + g * 4 + c] else 0;
            i += 1;
        }
    };
}

test "fp8 order: first lane takes k 0, 1, 8, 9, 16, 17, 24, 25 of row 0" {
    const n = 64;
    const k = 64;
    var codes: [n * k]u8 = undefined;
    for (&codes, 0..) |*c, j| c.* = @truncate(j % k);
    var out: [n * k]u8 = undefined;
    fp8Order(&codes, n, k, n, &out);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 8, 9, 16, 17, 24, 25 }, out[0..8]);
}

test "fp8 order by blocks, in any split, is the whole order" {
    const n = 150; // not a multiple of 64: the last block has zero rows
    const k = 128;
    const npad = 192;
    var codes: [n * k]u8 = undefined;
    for (&codes, 0..) |*c, j| c.* = @truncate(j *% 2654435761 >> 7);
    var whole: [npad * k]u8 = undefined;
    fp8Order(&codes, n, k, npad, &whole);
    var parts: [npad * k]u8 = undefined;
    const blk = 64 * k;
    fp8OrderBlocks(&codes, n, k, 0, 1, parts[0..blk]);
    fp8OrderBlocks(&codes, n, k, 1, 3, parts[blk..]);
    try std.testing.expectEqualSlices(u8, &whole, &parts);
}
