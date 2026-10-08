//! The launches of the video VAE decoder's kernels (kernels/cuda/minimax/gemm_f16.cu and vae_video.cu, two fatbins), one
//! method a kernel, with the grids, block sizes and argument types of the twin's torch bindings in
//! `stk_twin/h3/vae_video.py` (`_LAUNCH`): the same kernels in the same order with the same arguments, so Zig is bit-exact.

const std = @import("std");
const cuda = @import("cuda");

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

/// `stk_gemm_f16`: C[m, n] = A[m, k] . B[n, k]^T (+ bias[n]), then with `res`: half(res + C * rscale[col]). Pointers are
/// bytes (device addresses; 0 = absent), offsets and strides in half elements, like the twin's `k_gemm`.
pub const Gemm = struct {
    a: Ptr,
    b: Ptr,
    c: Ptr,
    m: u64,
    n: u64,
    k: u64,
    lda: u64,
    ldb: u64,
    ldc: u64,
    bias: Ptr = 0,
    res: Ptr = 0,
    rscale: Ptr = 0,
    a_off: u64 = 0,
    b_off: u64 = 0,
    c_off: u64 = 0,
    sa: u64 = 0,
    sb: u64 = 0,
    sc: u64 = 0,
    batch: u32 = 1,
    ldr: u64 = 0,
    sr: u64 = 0,
};

pub const Ops = struct {
    gemm_module: cuda.Module,
    module: cuda.Module,
    gemm_fn: cuda.Function,
    denorm_fn: cuda.Function,
    gather_fn: cuda.Function,
    suffix_fn: cuda.Function,
    rope_table_fn: cuda.Function,
    rms_fn: cuda.Function,
    ln_fn: cuda.Function,
    rope_fn: cuda.Function,
    vt_fn: cuda.Function,
    softmax_fn: cuda.Function,
    nan_fn: cuda.Function,
    swiglu_fn: cuda.Function,
    unshuffle_fn: cuda.Function,
    place_fn: cuda.Function,
    finalize_fn: cuda.Function,

    /// `gemm_image` is `h3.kernels.gemm_f16`, `image` is `h3.kernels.vae_video` (separate fatbins, both -O3).
    pub fn load(d: *const cuda.Driver, gemm_image: []const u8, image: []const u8) !Ops {
        var g = try cuda.Module.load(d, gemm_image);
        errdefer g.unload();
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{
            .gemm_module = g,
            .module = m,
            .gemm_fn = try g.function("stk_gemm_f16"),
            .denorm_fn = try m.function("vv_denorm"),
            .gather_fn = try m.function("vv_gather_rows"),
            .suffix_fn = try m.function("vv_suffix"),
            .rope_table_fn = try m.function("vv_rope_table"),
            .rms_fn = try m.function("vv_rms_norm"),
            .ln_fn = try m.function("vv_layer_norm"),
            .rope_fn = try m.function("vv_rms_rope"), // variant 0: the engine's contraction (VVAE-PORT 4.3)
            .vt_fn = try m.function("vv_vt"),
            .softmax_fn = try m.function("vv_softmax"),
            .nan_fn = try m.function("vv_nan_to_num"),
            .swiglu_fn = try m.function("vv_swiglu"),
            .unshuffle_fn = try m.function("vv_unshuffle"),
            .place_fn = try m.function("vv_place_tile"),
            .finalize_fn = try m.function("vv_finalize"),
        };
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
        o.gemm_module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, blk: u32, args: *cuda.launch.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = blk } }, s, args);
    }

    /// grid (ceil(n / 128), ceil(m / 128), batch) x 256; A and B advance by `a_off` / `b_off` half elements first.
    pub fn gemm(o: *const Ops, s: cuda.Stream, g: Gemm) !void {
        var a: cuda.launch.Args = .{};
        a.add(g.a + 2 * g.a_off);
        a.add(g.b + 2 * g.b_off);
        a.add(g.bias);
        a.add(g.c + 2 * g.c_off);
        inline for (.{ g.m, g.n, g.k }) |x| a.add(i32of(x));
        inline for (.{ g.lda, g.ldb, g.ldc, g.sa, g.sb, g.sc }) |x| a.add(i64of(x));
        a.add(g.res);
        a.add(g.rscale);
        a.add(i64of(g.ldr));
        a.add(i64of(g.sr));
        try go(o.gemm_fn, s, .{ .x = blocks2(g.n, 128), .y = blocks2(g.m, 128), .z = g.batch }, 256, &a);
    }

    fn blocks2(n: u64, by: u64) u32 {
        return @intCast((n + by - 1) / by);
    }

    /// y = half(half(z * std_c) + mean_c) over [c, plane], plane = n / c.
    pub fn denorm(o: *const Ops, s: cuda.Stream, z: Ptr, stdv: Ptr, mean: Ptr, y: Ptr, c: u64, n: u64) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ z, stdv, mean, y }) |p| a.add(p);
        a.add(i64of(n / c));
        a.add(i64of(n));
        try go(o.denorm_fn, s, .{ .x = blocks(n) }, block, &a);
    }

    /// z [c, tz, hz, wz] -> rows [tn * h * w, c] of the crop (t0, y0, x0), frames clamped to tz - 1.
    pub fn gather(o: *const Ops, s: cuda.Stream, z: Ptr, rows: Ptr, c: u32, tz: u32, hz: u32, wz: u32, t0: u32, tn: u32, y0: u32, h: u32, x0: u32, w: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(z);
        a.add(rows);
        inline for (.{ c, tz, hz, wz, t0, tn, y0, h, x0, w }) |x| a.add(i32of(x));
        try go(o.gather_fn, s, .{ .x = blocks(@as(u64, tn) * h * w * c) }, block, &a);
    }

    /// Rows np .. np + 4 of h: the 4 register tokens and a zero token.
    pub fn suffix(o: *const Ops, s: cuda.Stream, h: Ptr, reg: Ptr, np: u64, d: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(h);
        a.add(reg);
        a.add(i64of(np));
        a.add(i64of(d));
        try go(o.suffix_fn, s, .{ .x = blocks(5 * d) }, block, &a);
    }

    /// The rotation table [S, 24, 4] for a tn x h x w tile and `nsuf` suffix tokens.
    pub fn ropeTable(o: *const Ops, s: cuda.Stream, inv_freq: Ptr, table: Ptr, tn: u32, h: u32, w: u32, nsuf: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(inv_freq);
        a.add(table);
        inline for (.{ tn, h, w, nsuf }) |x| a.add(i32of(x));
        const total = @as(u64, tn) * h * w + nsuf;
        try go(o.rope_table_fn, s, .{ .x = blocks(total * 24) }, block, &a);
    }

    pub fn rmsNorm(o: *const Ops, s: cuda.Stream, x: Ptr, w: Ptr, y: Ptr, rows: u64, d: u64, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, y }) |p| a.add(p);
        a.add(i64of(d));
        a.add(eps);
        try go(o.rms_fn, s, .{ .x = @intCast(rows) }, block, &a);
    }

    pub fn layerNorm(o: *const Ops, s: cuda.Stream, x: Ptr, w: Ptr, b: Ptr, y: Ptr, rows: u64, d: u64, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, b, y }) |p| a.add(p);
        a.add(i64of(d));
        a.add(eps);
        try go(o.ln_fn, s, .{ .x = @intCast(rows) }, block, &a);
    }

    /// In place on the qkv rows [S, heads * 192]; grid ceil(S * heads / 4) x 128.
    pub fn rmsRope(o: *const Ops, s: cuda.Stream, qkv: Ptr, table: Ptr, rows: u64, heads: u32, row_stride: u64, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        a.add(qkv);
        a.add(table);
        a.add(i64of(rows));
        a.add(i32of(heads));
        a.add(i64of(row_stride));
        a.add(@as(i32, 192));
        a.add(eps);
        try go(o.rope_fn, s, .{ .x = blocks2(rows * heads, 4) }, 128, &a);
    }

    /// V transposed per head to [heads, 64, sp], zero padded.
    pub fn vt(o: *const Ops, s: cuda.Stream, qkv: Ptr, out: Ptr, rows: u64, sp: u64, heads: u32, row_stride: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(qkv);
        a.add(out);
        a.add(i64of(rows));
        a.add(i64of(sp));
        a.add(i32of(heads));
        a.add(i64of(row_stride));
        a.add(@as(i32, 192));
        try go(o.vt_fn, s, .{ .x = blocks(@as(u64, heads) * 64 * sp) }, block, &a);
    }

    /// Row softmax in place, `nrows` rows of `sp` (the first `rows` columns real).
    pub fn softmax(o: *const Ops, s: cuda.Stream, sc: Ptr, nrows: u64, rows: u64, sp: u64, scale: f32) !void {
        var a: cuda.launch.Args = .{};
        a.add(sc);
        a.add(i64of(rows));
        a.add(i64of(sp));
        a.add(scale);
        try go(o.softmax_fn, s, .{ .x = @intCast(nrows) }, block, &a);
    }

    pub fn nanToNum(o: *const Ops, s: cuda.Stream, x: Ptr, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(i64of(n));
        try go(o.nan_fn, s, .{ .x = blocks(n) }, block, &a);
    }

    /// [m, 2f] = [gate | up] -> [m, f]; grid (ceil(f / 256), m).
    pub fn swiglu(o: *const Ops, s: cuda.Stream, gu: Ptr, out: Ptr, m: u64, f: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(gu);
        a.add(out);
        a.add(i64of(f));
        try go(o.swiglu_fn, s, .{ .x = blocks(f), .y = @intCast(m) }, block, &a);
    }

    /// proj_out rows [>= tn * h * w, 3072] -> tile [3, 4 tn, 16 h, 16 w].
    pub fn unshuffle(o: *const Ops, s: cuda.Stream, rows: Ptr, out: Ptr, tn: u32, h: u32, w: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(rows);
        a.add(out);
        inline for (.{ tn, h, w }) |x| a.add(i32of(x));
        try go(o.unshuffle_fn, s, .{ .x = blocks(3 * 4 * @as(u64, tn) * 16 * h * 16 * w) }, block, &a);
    }

    /// One tile into the canvas (see vv_place_tile). `ytail` / `ltail`: 0 when absent (their sizes then 0).
    pub const Place = struct {
        b: Ptr,
        frames: u32,
        th: u32,
        tw: u32,
        ytail: Ptr = 0,
        tha: u32 = 0,
        twa: u32 = 0,
        ey: u32 = 0,
        ltail: Ptr = 0,
        thl: u32 = 0,
        twl: u32 = 0,
        ex: u32 = 0,
        canvas: Ptr,
        hc: u32,
        wc: u32,
        oy: u32,
        ox: u32,
        oh: u32,
        ow: u32,
    };

    pub fn place(o: *const Ops, s: cuda.Stream, p: Place) !void {
        var a: cuda.launch.Args = .{};
        a.add(p.b);
        inline for (.{ p.frames, p.th, p.tw }) |x| a.add(i32of(x));
        a.add(p.ytail);
        inline for (.{ p.tha, p.twa, p.ey }) |x| a.add(i32of(x));
        a.add(p.ltail);
        inline for (.{ p.thl, p.twl, p.ex }) |x| a.add(i32of(x));
        a.add(p.canvas);
        inline for (.{ p.hc, p.wc, p.oy, p.ox, p.oh, p.ow }) |x| a.add(i32of(x));
        try go(o.place_fn, s, .{ .x = blocks(3 * @as(u64, p.frames) * p.oh * p.ow) }, block, &a);
    }

    /// write_part's temporal blend + finalise + uint8 (see vv_finalize); `sm` = std[3], mean[3] as fp32.
    pub const Finalize = struct {
        a: Ptr = 0,
        fa: u32 = 0,
        a0: u32 = 0,
        b: Ptr,
        fb: u32,
        b0: u32,
        ext: u32,
        copy: u32,
        h: u32,
        w: u32,
        sm: [6]f32,
        out_u8: Ptr,
        pos: u32,
    };

    pub fn finalize(o: *const Ops, s: cuda.Stream, f: Finalize) !void {
        var a: cuda.launch.Args = .{};
        a.add(f.a);
        inline for (.{ f.fa, f.a0 }) |x| a.add(i32of(x));
        a.add(f.b);
        inline for (.{ f.fb, f.b0, f.ext, f.copy, f.h, f.w }) |x| a.add(i32of(x));
        for (f.sm) |v| a.add(v);
        a.add(f.out_u8);
        a.add(@as(u64, 0)); // no fp32 output
        a.add(i32of(f.pos));
        try go(o.finalize_fn, s, .{ .x = blocks(@as(u64, f.copy) * f.h * f.w * 3) }, block, &a);
    }
};
