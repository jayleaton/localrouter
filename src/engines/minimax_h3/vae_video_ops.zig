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
/// bytes (device addresses; 0 = absent), offsets and strides in half elements, like the twin's `k_gemm`. Which of the four
/// bit-equal kernels runs is `Ops.pick`'s call (the twin's `gemm_pick`), the reference when `STK_VVAE_REF` is set.
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

/// The GEMM kernels of gemm_f16.cu, indexed as the twin's `k_gemm_v` variants: the reference (grid (n, m) of 128 x 128, the
/// first version) and the three template tiles (grid (m, n), dynamic shared memory = stages * (BM + BN) * (BK + 8) * 2 bytes).
const GemmKernel = struct { name: [:0]const u8, bm: u32, bn: u32, threads: u32, smem: u32 };
const gemm_kernels = [_]GemmKernel{
    .{ .name = "stk_gemm_f16_ref", .bm = 128, .bn = 128, .threads = 256, .smem = 0 },
    .{ .name = "stk_gemm_f16", .bm = 256, .bn = 128, .threads = 256, .smem = 3 * (256 + 128) * 40 * 2 },
    .{ .name = "stk_gemm_f16_s", .bm = 128, .bn = 128, .threads = 256, .smem = 2 * (128 + 128) * 40 * 2 },
    .{ .name = "stk_gemm_f16_n", .bm = 64, .bn = 64, .threads = 128, .smem = 3 * (64 + 64) * 40 * 2 },
};

/// Environment variable `name` set to something other than "" or "0".
fn envOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    const t = std.mem.span(v);
    return t.len > 0 and !std.mem.eql(u8, t, "0");
}

/// Environment variable `name` as an unsigned number (`default` when unset or not a number).
fn envInt(name: [*:0]const u8, default: u32) u32 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch default;
}

pub const Ops = struct {
    gemm_module: cuda.Module,
    module: cuda.Module,
    gemm_fns: [gemm_kernels.len]cuda.Function,
    /// `STK_VVAE_REF=1`: every GEMM on the reference kernel and the attention in one piece (the old path, for comparisons).
    ref_only: bool,
    /// `STK_VVAE_HEADS=n`: heads an attention group runs (default 2; 0 or >= 32: all at once). See Tile.attention.
    heads: u32,
    /// `STK_VVAE_PROF=1`: CUDA events around every op; the decode's GPU milliseconds by class go to stderr.
    prof: bool,
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
        var fns: [gemm_kernels.len]cuda.Function = undefined;
        for (gemm_kernels, &fns) |k, *f| {
            f.* = try g.function(k.name);
            if (k.smem > 48 * 1024) try f.allowDynamicShared(k.smem); // 92,160 bytes of the wide tile: opt in
        }
        const ref_only = envOn("STK_VVAE_REF");
        const heads = envInt("STK_VVAE_HEADS", 2);
        return .{
            .gemm_module = g,
            .module = m,
            .gemm_fns = fns,
            .ref_only = ref_only,
            .heads = if (ref_only or heads == 0 or heads >= 32) 32 else heads,
            .prof = envOn("STK_VVAE_PROF"),
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

    /// The kernel for an m x n x k GEMM (an index of `gemm_kernels`, the twin's `gemm_pick`): K <= 64 (scores, the embeddings:
    /// the epilogue dominates) the 128 x 128 tile, N <= 64 (P . V) the 64 x 64, the linears the 256 x 128. All bit-equal.
    pub fn pick(m: u64, n: u64, k: u64) usize {
        _ = m;
        return if (k <= 64) 2 else if (n <= 64) 3 else 1;
    }

    /// Reference: grid (ceil(n / 128), ceil(m / 128), batch) x 256. The others: grid (ceil(m / BM), ceil(n / BN), batch), the m
    /// tile fastest (the blocks resident together read one weight tile once). A and B advance by `a_off` / `b_off` half
    /// elements first.
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
        const v: usize = if (o.ref_only) 0 else pick(g.m, g.n, g.k);
        const k = gemm_kernels[v];
        const grid: cuda.launch.Dim3 = if (v == 0)
            .{ .x = blocks2(g.n, k.bn), .y = blocks2(g.m, k.bm), .z = g.batch }
        else
            .{ .x = blocks2(g.m, k.bm), .y = blocks2(g.n, k.bn), .z = g.batch };
        try cuda.launch.launch(o.gemm_fns[v], .{ .grid = grid, .block = .{ .x = k.threads }, .shared = k.smem }, s, &a);
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

test "gemm kernels: the shape pick and the shared memory the tiles need" {
    // the linears (K, N large) the wide tile; scores (K = 64) and the embeddings (K = 24) the 128 x 128; P . V (N = 64) the narrow
    try std.testing.expectEqual(@as(usize, 1), Ops.pick(1797, 6144, 2048));
    try std.testing.expectEqual(@as(usize, 1), Ops.pick(1797, 2048, 8192));
    try std.testing.expectEqual(@as(usize, 2), Ops.pick(1797, 1797, 64));
    try std.testing.expectEqual(@as(usize, 2), Ops.pick(1792, 2048, 24));
    try std.testing.expectEqual(@as(usize, 2), Ops.pick(1792, 24, 24));
    try std.testing.expectEqual(@as(usize, 3), Ops.pick(1797, 64, 1800));
    // stages * (BM + BN) * (BK + 8) * 2 as gemm_f16.cu's Cfg::SMEM
    try std.testing.expectEqual(@as(u32, 92160), gemm_kernels[1].smem);
    try std.testing.expectEqual(@as(u32, 40960), gemm_kernels[2].smem);
    try std.testing.expectEqual(@as(u32, 30720), gemm_kernels[3].smem);
    try std.testing.expect(gemm_kernels[1].smem <= 99 * 1024); // the opt-in limit of an sm_120 / sm_121 block
}
