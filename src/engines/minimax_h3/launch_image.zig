//! Launches of MiniMax H3-Image's own kernels (kernels/cuda/minimax/h3_image.cu, one image for CUDA and HIP): the DiT's
//! PDD head mix and the image VAE decoder's RoPE table, split-half RMSNorm + RoPE, unpatchify, tile blend and copies.
//! Everything else H3-Image runs is the H3 kernels' (`launch.zig`, `vae_video_ops.zig`) and the bf16 GEMM and attention.

const std = @import("std");
const gpu = @import("gpu");
const pdd = @import("pdd.zig");

const Ptr = u64;
const block: u32 = 256;

fn blocks(n: u64) u32 {
    return @intCast((n + block - 1) / block);
}
fn i32of(v: anytype) i32 {
    return @intCast(v);
}
fn i64of(v: anytype) i64 {
    return @intCast(v);
}

pub const Ops = struct {
    module: gpu.Module,
    head_mix_fn: gpu.Function,
    rope_table_fn: gpu.Function,
    rms_rope_fn: gpu.Function,
    unshuffle_fn: gpu.Function,
    blend_fn: gpu.Function,
    copy_fn: gpu.Function,

    /// `image`: `h3.kernels.h3_image`.
    pub fn load(d: *const gpu.Driver, image: []const u8) !Ops {
        var m = try gpu.Module.load(d, image);
        errdefer m.unload();
        var o: Ops = undefined;
        o.module = m;
        inline for (.{
            .{ "head_mix_fn", "h3i_head_mix" },  .{ "rope_table_fn", "vi_rope_table" }, .{ "rms_rope_fn", "vi_rms_rope" },
            .{ "unshuffle_fn", "vi_unshuffle" }, .{ "blend_fn", "vi_blend" },           .{ "copy_fn", "vi_copy" },
        }) |e| @field(o, e[0]) = try m.function(e[1]);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
    }

    fn go(f: gpu.Function, s: gpu.Stream, grid: gpu.launch.Dim3, blk: u32, args: *gpu.launch.Args) !void {
        _ = @atomicRmw(u64, &@import("launch.zig").count, .Add, 1, .monotonic);
        try gpu.launch.launch(f, .{ .grid = grid, .block = .{ .x = blk } }, s, args);
    }

    /// The step's head out of a PDD bank of `per` fp32 values a head (the weight's 96 x K, or the bias's 96): row block 0
    /// plus `st`'s dt-weighted offsets, in `sampler.mix`'s order (h3i_head_mix).
    pub fn headMix(o: *const Ops, s: gpu.Stream, bank: Ptr, out: Ptr, per: u64, st: pdd.Step) !void {
        var a: gpu.launch.Args = .{};
        a.add(bank);
        a.add(out);
        a.add(i64of(per));
        a.add(@as(i32, st.nheads));
        for (st.heads) |h| a.add(@as(i32, h));
        for (st.w) |w| a.add(w);
        try go(o.head_mix_fn, s, .{ .x = blocks(per) }, block, &a);
    }

    /// The 2-D rotation table [S, 16, 4] for an h x w tile and `nsuf` suffix tokens.
    pub fn ropeTable(o: *const Ops, s: gpu.Stream, inv_freq: Ptr, table: Ptr, h: u32, w: u32, nsuf: u32) !void {
        var a: gpu.launch.Args = .{};
        a.add(inv_freq);
        a.add(table);
        inline for (.{ h, w, nsuf }) |x| a.add(i32of(x));
        try go(o.rope_table_fn, s, .{ .x = blocks((@as(u64, h) * w + nsuf) * 16) }, block, &a);
    }

    /// In place on the qkv rows [S, heads * 192]; grid ceil(S * heads / 4) x 128.
    pub fn rmsRope(o: *const Ops, s: gpu.Stream, qkv: Ptr, table: Ptr, rows: u64, heads: u32, row_stride: u64, eps: f32) !void {
        var a: gpu.launch.Args = .{};
        a.add(qkv);
        a.add(table);
        a.add(i64of(rows));
        a.add(i32of(heads));
        a.add(i64of(row_stride));
        a.add(@as(i32, 192));
        a.add(eps);
        try go(o.rms_rope_fn, s, .{ .x = @intCast((rows * heads + 3) / 4) }, 128, &a);
    }

    /// proj_out rows [>= h * w, 768] -> tile [3, 16 h, 16 w].
    pub fn unshuffle(o: *const Ops, s: gpu.Stream, rows: Ptr, out: Ptr, h: u32, w: u32) !void {
        var a: gpu.launch.Args = .{};
        a.add(rows);
        a.add(out);
        a.add(i32of(h));
        a.add(i32of(w));
        try go(o.unshuffle_fn, s, .{ .x = blocks(3 * 256 * @as(u64, h) * w) }, block, &a);
    }

    /// A [3, ah, aw] fp16 region with its window at (ay, ax).
    pub const Region = struct { ptr: Ptr, h: u32, w: u32, y: u32 = 0, x: u32 = 0 };

    /// tiled_decode's blend of the tile `b` [3, th, tw] in place against `a`'s window, `e` rows (`dim` 0) or columns (1).
    pub fn blend(o: *const Ops, s: gpu.Stream, b: Ptr, th: u32, tw: u32, a_: Region, e: u32, dim: u1) !void {
        var a: gpu.launch.Args = .{};
        a.add(b);
        a.add(i32of(th));
        a.add(i32of(tw));
        a.add(a_.ptr);
        inline for (.{ a_.h, a_.w, a_.y, a_.x, e }) |x| a.add(i32of(x));
        a.add(@as(i32, dim));
        const n: u64 = 3 * @as(u64, if (dim == 0) e else th) * (if (dim == 0) tw else e);
        try go(o.blend_fn, s, .{ .x = blocks(n) }, block, &a);
    }

    /// The h x w window of `src` at its (y, x) to `dst` at its (y, x).
    pub fn copy(o: *const Ops, s: gpu.Stream, src: Region, dst: Region, h: u32, w: u32) !void {
        if (h == 0 or w == 0) return;
        var a: gpu.launch.Args = .{};
        a.add(src.ptr);
        inline for (.{ src.h, src.w, src.y, src.x }) |x| a.add(i32of(x));
        a.add(dst.ptr);
        inline for (.{ dst.h, dst.w, dst.y, dst.x, h, w }) |x| a.add(i32of(x));
        try go(o.copy_fn, s, .{ .x = blocks(3 * @as(u64, h) * w) }, block, &a);
    }
};
