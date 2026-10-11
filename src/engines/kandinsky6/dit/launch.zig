//! Launches of the Kandinsky-6 DiT kernels (kernels/cuda/kandinsky6/dit/{gemm,attention,ops}.cu) on a caller-owned
//! stream. `Kernels.load` takes the three fatbin images the build embeds and the pipeline's borrowed `cuda.Driver`;
//! it creates no context, stream or memory. Every launcher takes typed `layout.View`s, checks dtype, shape and the
//! kernel's alignment requirements, then launches with the grid documented beside the kernel. A launch never
//! synchronizes, allocates or reads back.

const std = @import("std");
const cuda = @import("cuda");
const layout = @import("layout.zig");
const attention = @import("attention.zig");

const View = layout.View;

pub const Error = cuda.Error || error{ ViewMismatch, InvalidAttention, InvalidGemm };

/// The fatbin images (kernels/cuda/kandinsky6/dit/*.cu), embedded by the build. Empty slices: no kernels (host-only
/// builds); `Kernels.load` refuses them.
pub const Images = struct {
    gemm: []const u8,
    attention: []const u8,
    ops: []const u8,
};

pub const Epilogue = enum(i32) { bias = 0, gelu = 1 };

/// The arithmetic of the fp32 linears (TimeEmbeddings linear_1 / linear_2, every Modulation). Native upstream gives
/// them FP32 operands and outputs (BF16-cast weights widened by `.float()`) under `set_float32_matmul_precision("high")`
/// (kandinsky/pipeline/factory.py:159-162 at eac06e39), which PERMITS a TF32 algorithm but does not fix the one SM121
/// runs. Which of these, if any, reproduces native bits is unverified: none is a default, none is qualified, and the
/// paths stay unqualified until SM121 byte fixtures of the native oracle (at its own settings) pass. `tf32_rna` /
/// `tf32_truncate`: TF32 tensor-core operands rounded that way; `fp32`: strict FP32, a diagnostic only.
pub const Fp32Linear = enum(i32) { tf32_rna = 0, tf32_truncate = 1, fp32 = 2 };

/// Kernel launches made through this file: a counter for benchmarks.
pub var count: u64 = 0;

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, blk: cuda.launch.Dim3, shared: u32, args: *cuda.launch.Args) cuda.Error!void {
    _ = @atomicRmw(u64, &count, .Add, 1, .monotonic);
    try cuda.launch.launch(f, .{ .grid = grid, .block = blk, .shared = shared }, s, args);
}

/// torch's layer-norm block: (warp, num_threads() / warp) = (32, 4).
const norm_block: cuda.launch.Dim3 = .{ .x = 32, .y = 4 };

/// A row count on grid.x (2^31 - 1 blocks; grid.y and .z stop at 65,535).
fn gridRows(rows: u64) error{ViewMismatch}!u32 {
    if (rows == 0 or rows > std.math.maxInt(i32)) return error.ViewMismatch;
    return @intCast(rows);
}

fn blocks(n: u64, per: u64) u32 {
    return @intCast((n + per - 1) / per);
}

fn i64of(v: anytype) i64 {
    return @intCast(v);
}

fn i32of(v: anytype) i32 {
    return @intCast(v);
}

pub const Kernels = struct {
    gemm_mod: cuda.Module,
    attn_mod: cuda.Module,
    ops_mod: cuda.Module,
    gemm_fn: cuda.Function,
    gemv_fn: cuda.Function,
    gemv_tf32_fn: cuda.Function,
    attn_fn: cuda.Function,
    cast_fn: cuda.Function,
    add_fn: cuda.Function,
    ln_fn: cuda.Function,
    norm_mod_fn: cuda.Function,
    gate_fn: cuda.Function,
    qk_fn: cuda.Function,
    sincos_fn: cuda.Function,
    patchify_fn: cuda.Function,
    unpatchify_fn: cuda.Function,
    grid_major_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, images: Images) Error!Kernels {
        if (images.gemm.len == 0 or images.attention.len == 0 or images.ops.len == 0) return error.Invalid;
        var k: Kernels = undefined;
        k.gemm_mod = try cuda.Module.load(d, images.gemm);
        errdefer k.gemm_mod.unload();
        k.attn_mod = try cuda.Module.load(d, images.attention);
        errdefer k.attn_mod.unload();
        k.ops_mod = try cuda.Module.load(d, images.ops);
        errdefer k.ops_mod.unload();
        k.gemm_fn = try k.gemm_mod.function("k6_gemm_bf16");
        k.gemv_fn = try k.gemm_mod.function("k6_gemv_f32");
        k.gemv_tf32_fn = try k.gemm_mod.function("k6_gemv_tf32");
        k.attn_fn = try k.attn_mod.function("k6_attention");
        try k.attn_fn.allowDynamicShared(attention.shared_bytes);
        inline for (.{
            .{ "cast_fn", "k6_cast_bf16" },        .{ "add_fn", "k6_add_bf16" },      .{ "ln_fn", "k6_layer_norm" },
            .{ "norm_mod_fn", "k6_norm_mod" },     .{ "gate_fn", "k6_gate_sum" },     .{ "qk_fn", "k6_qk_norm_rope" },
            .{ "sincos_fn", "k6_time_sincos" },    .{ "patchify_fn", "k6_patchify" }, .{ "unpatchify_fn", "k6_unpatchify" },
            .{ "grid_major_fn", "k6_grid_major" },
        }) |e| @field(k, e[0]) = try k.ops_mod.function(e[1]);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.ops_mod.unload();
        k.attn_mod.unload();
        k.gemm_mod.unload();
    }

    /// out[M, N] = x[M, K] . w[N, K]^T with the epilogue (bias: + bias[N] in fp32, null for none; gelu: GELU of the
    /// bf16-rounded product, no bias). Rows are independent: a row chunk gives the rows' bits of the whole call.
    pub fn gemm(k: *const Kernels, s: cuda.Stream, x: View, w: View, bias: ?View, out: View, epi: Epilogue) Error!void {
        const m = x.rows;
        const n = w.rows;
        const kd = x.cols;
        try w.expect(.bf16, n, kd);
        try x.expect(.bf16, m, kd);
        try out.expect(.bf16, m, n);
        if (bias) |b| try b.expect(.bf16, 1, n);
        if (epi == .gelu and bias != null) return error.InvalidGemm;
        const vec = kd % 8 == 0 and x.ld % 8 == 0 and w.ld % 8 == 0 and x.ptr % 16 == 0 and w.ptr % 16 == 0;
        if (m == 0 or m > std.math.maxInt(i32) or n > std.math.maxInt(i32)) return error.InvalidGemm;
        var a: cuda.launch.Args = .{};
        a.add(x.ptr);
        a.add(w.ptr);
        a.add(if (bias) |b| b.ptr else @as(u64, 0));
        a.add(out.ptr);
        inline for (.{ m, n, kd }) |v| a.add(i32of(v));
        inline for (.{ x.ld, w.ld, out.ld }) |v| a.add(i64of(v));
        a.add(@backingInt(epi));
        a.add(@as(i32, @intFromBool(vec)));
        try go(k.gemm_fn, s, .{ .x = blocks(n, 128), .y = blocks(m, 128) }, .{ .x = 256 }, 0, &a);
    }

    /// fp32 out[M, N] = act(x[M, K]) . float(w[N, K])^T + float(bias[N]): x fp32 or bf16, act SiLU when `silu`, out
    /// fp32 or bf16 (rounded once), in the `math` the caller picks. The timestep MLP and every Modulation.
    pub fn gemv(k: *const Kernels, s: cuda.Stream, x: View, w: View, bias: ?View, out: View, silu: bool, math: Fp32Linear) Error!void {
        const m = x.rows;
        const n = w.rows;
        try w.expect(.bf16, n, x.cols);
        if (w.ld != x.cols) return error.InvalidGemm;
        if (x.dtype != .f32 and x.dtype != .bf16) return error.ViewMismatch;
        try x.expect(x.dtype, m, x.cols);
        if (out.dtype != .f32 and out.dtype != .bf16) return error.ViewMismatch;
        try out.expect(out.dtype, m, n);
        if (bias) |b| try b.expect(.bf16, 1, n);
        var a: cuda.launch.Args = .{};
        a.add(x.ptr);
        a.add(w.ptr);
        a.add(if (bias) |b| b.ptr else @as(u64, 0));
        a.add(out.ptr);
        inline for (.{ m, n, x.cols }) |v| a.add(i32of(v));
        a.add(i64of(x.ld));
        a.add(i64of(out.ld));
        a.add(@as(i32, @intFromBool(x.dtype == .bf16)));
        a.add(@as(i32, @intFromBool(silu)));
        a.add(@as(i32, @intFromBool(out.dtype == .bf16)));
        switch (math) {
            .fp32 => try go(k.gemv_fn, s, .{ .x = blocks(n, 8) }, .{ .x = 256 }, 0, &a),
            .tf32_rna, .tf32_truncate => {
                a.add(@backingInt(math));
                try go(k.gemv_tf32_fn, s, .{ .x = blocks(n, 64), .y = blocks(m, 16) }, .{ .x = 256 }, 0, &a);
            },
        }
    }

    pub fn attend(k: *const Kernels, s: cuda.Stream, c: attention.Call) Error!void {
        try c.validate();
        var a: cuda.launch.Args = .{};
        inline for (.{ c.q.ptr, c.k.ptr, c.v.ptr, c.out.ptr }) |p| a.add(p);
        a.add(switch (c.mask) {
            .none => @as(u64, 0),
            .keys => |m| m.ptr,
        });
        a.add(i32of(c.q.rows));
        a.add(i32of(c.k.rows));
        inline for (.{ c.q.ld, c.k.ld, c.v.ld, c.out.ld }) |v| a.add(i64of(v));
        inline for (0..5) |_| a.add(@as(i64, 0)); // batch strides: one batch a call
        a.add(c.scale);
        const g = c.grid();
        try go(k.attn_fn, s, .{ .x = g[0], .y = g[1], .z = g[2] }, .{ .x = attention.threads }, attention.shared_bytes, &a);
    }

    /// bf16 copy of fp32 values: x and y contiguous [n].
    pub fn castBf16(k: *const Kernels, s: cuda.Stream, x: View, y: View) Error!void {
        try x.expect(.f32, x.rows, x.cols);
        try y.expect(.bf16, x.rows, x.cols);
        if (x.ld != x.cols or y.ld != y.cols) return error.ViewMismatch;
        const n = x.rows * x.cols;
        var a: cuda.launch.Args = .{};
        a.add(x.ptr);
        a.add(y.ptr);
        a.add(i64of(n));
        try go(k.cast_fn, s, .{ .x = blocks(n, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// y = bf16(a + b), contiguous views of one shape (y may alias a or b).
    pub fn addBf16(k: *const Kernels, s: cuda.Stream, x: View, z: View, y: View) Error!void {
        inline for (.{ x, z, y }) |v| {
            try v.expect(.bf16, x.rows, x.cols);
            if (v.ld != v.cols) return error.ViewMismatch;
        }
        const n = x.rows * x.cols;
        var a: cuda.launch.Args = .{};
        inline for (.{ x.ptr, z.ptr, y.ptr }) |p| a.add(p);
        a.add(i64of(n));
        try go(k.add_fn, s, .{ .x = blocks(n, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// LayerNorm (eps 1e-5) of bf16 rows into bf16 rows, affine when w / b are given. cols % 4 == 0 (torch's
    /// vectorized path, whose order the kernel reproduces).
    pub fn layerNorm(k: *const Kernels, s: cuda.Stream, x: View, w: ?View, b: ?View, y: View) Error!void {
        try x.expect(.bf16, x.rows, x.cols);
        if (x.cols % 4 != 0) return error.ViewMismatch;
        try y.expect(.bf16, x.rows, x.cols);
        if (w) |v| try v.expect(.bf16, 1, x.cols);
        if (b) |v| try v.expect(.bf16, 1, x.cols);
        var a: cuda.launch.Args = .{};
        a.add(x.ptr);
        a.add(if (w) |v| v.ptr else @as(u64, 0));
        a.add(if (b) |v| v.ptr else @as(u64, 0));
        a.add(y.ptr);
        a.add(i32of(x.cols));
        a.add(i64of(x.ld));
        a.add(i64of(y.ld));
        a.add(layout.ln_eps);
        try go(k.ln_fn, s, .{ .x = try gridRows(x.rows) }, norm_block, 0, &a);
    }

    /// apply_scale_shift_norm with a non-affine LayerNorm (eps 1e-5): scale / shift one bf16 row each.
    pub fn normMod(k: *const Kernels, s: cuda.Stream, x: View, scale: View, shift: View, y: View) Error!void {
        try x.expect(.bf16, x.rows, x.cols);
        if (x.cols % 4 != 0) return error.ViewMismatch;
        try y.expect(.bf16, x.rows, x.cols);
        try scale.expect(.bf16, 1, x.cols);
        try shift.expect(.bf16, 1, x.cols);
        var a: cuda.launch.Args = .{};
        inline for (.{ x.ptr, scale.ptr, shift.ptr, y.ptr }) |p| a.add(p);
        a.add(i32of(x.cols));
        a.add(i64of(x.ld));
        a.add(i64of(y.ld));
        a.add(layout.ln_eps);
        try go(k.norm_mod_fn, s, .{ .x = try gridRows(x.rows) }, norm_block, 0, &a);
    }

    /// apply_gate_sum in place: x = bf16(x + gate * out), gate one bf16 row.
    pub fn gateSum(k: *const Kernels, s: cuda.Stream, x: View, out: View, gate: View) Error!void {
        try x.expect(.bf16, x.rows, x.cols);
        try out.expect(.bf16, x.rows, x.cols);
        try gate.expect(.bf16, 1, x.cols);
        var a: cuda.launch.Args = .{};
        inline for (.{ x.ptr, out.ptr, gate.ptr }) |p| a.add(p);
        a.add(i32of(x.cols));
        a.add(i64of(x.ld));
        a.add(i64of(out.ld));
        try go(k.gate_fn, s, .{ .x = try gridRows(x.rows), .y = blocks(x.cols, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// In place: per-head RMSNorm (bf16 weight [128], eps FLT_EPSILON) then the module's RoPE table (fp32 [rows, 64,
    /// 2, 2]), or no rotation when `rope` is null.
    pub fn qkNormRope(k: *const Kernels, s: cuda.Stream, x: View, w: View, rope: ?View, heads: u32) Error!void {
        try x.expect(.bf16, x.rows, @as(u64, heads) * attention.head_dim);
        try w.expect(.bf16, 1, attention.head_dim);
        if (rope) |r| try r.expect(.f32, x.rows, layout.rope_pairs * 4);
        var a: cuda.launch.Args = .{};
        a.add(x.ptr);
        a.add(w.ptr);
        a.add(if (rope) |r| r.ptr else @as(u64, 0));
        a.add(i32of(x.rows));
        a.add(i32of(heads));
        a.add(i64of(x.ld));
        a.add(i64of(if (rope) |r| r.ld else 0));
        a.add(rms_eps);
        try go(k.qk_fn, s, .{ .x = blocks(x.rows * heads, 8) }, .{ .x = 256 }, 0, &a);
    }

    /// The timestep sinusoid: time fp32 [M, 1], freqs fp32 [1, half], out fp32 [M, 2 * half].
    pub fn timeSincos(k: *const Kernels, s: cuda.Stream, time: View, freqs: View, out: View) Error!void {
        try time.expect(.f32, time.rows, 1);
        try freqs.expect(.f32, 1, freqs.cols);
        try out.expect(.f32, time.rows, 2 * freqs.cols);
        if (out.ld != out.cols or time.ld != 1) return error.ViewMismatch;
        var a: cuda.launch.Args = .{};
        inline for (.{ time.ptr, freqs.ptr, out.ptr }) |p| a.add(p);
        a.add(i32of(freqs.cols));
        try go(k.sincos_fn, s, .{ .x = try gridRows(time.rows), .y = blocks(freqs.cols, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// Patchify the [T, H, W, C] channels-last latent (fp32 or bf16, contiguous) into bf16 rows [T * H/2 * W/2, 4C].
    pub fn patchify(k: *const Kernels, s: cuda.Stream, x: View, rows: View, sh: layout.Shape, c: u32) Error!void {
        const n_in = @as(u64, sh.t) * sh.h * sh.w;
        if (x.dtype != .f32 and x.dtype != .bf16) return error.ViewMismatch;
        try x.expect(x.dtype, n_in, c);
        try rows.expect(.bf16, sh.tokens(), 4 * @as(u64, c));
        if (x.ld != c or rows.ld != rows.cols) return error.ViewMismatch;
        var a: cuda.launch.Args = .{};
        a.add(x.ptr);
        a.add(rows.ptr);
        inline for (.{ sh.t, sh.h, sh.w, c }) |v| a.add(i32of(v));
        a.add(@as(i32, @intFromBool(x.dtype == .f32)));
        try go(k.patchify_fn, s, .{ .x = blocks(n_in * c, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// The video head's unpatchify into grid-major [G, T, H, W, C / G] bf16 (contiguous, out viewed as [G * T * H * W,
    /// C / G]).
    pub fn unpatchify(k: *const Kernels, s: cuda.Stream, rows: View, out: View, sh: layout.Shape, c: u32, g: u32) Error!void {
        try rows.expect(.bf16, sh.tokens(), 4 * @as(u64, c));
        if (c % g != 0 or rows.ld != rows.cols) return error.ViewMismatch;
        try out.expect(.bf16, @as(u64, g) * sh.t * sh.h * sh.w, c / g);
        if (out.ld != out.cols) return error.ViewMismatch;
        var a: cuda.launch.Args = .{};
        a.add(rows.ptr);
        a.add(out.ptr);
        inline for (.{ sh.t, sh.h / 2, sh.w / 2, c, g }) |v| a.add(i32of(v));
        try go(k.unpatchify_fn, s, .{ .x = blocks(sh.tokens() * 4 * c, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// [rows, G * Cg] -> grid-major [G * rows, Cg] (contiguous).
    pub fn gridMajor(k: *const Kernels, s: cuda.Stream, x: View, out: View, g: u32) Error!void {
        try x.expect(.bf16, x.rows, x.cols);
        if (x.cols % g != 0 or x.ld != x.cols) return error.ViewMismatch;
        try out.expect(.bf16, x.rows * g, x.cols / g);
        if (out.ld != out.cols) return error.ViewMismatch;
        var a: cuda.launch.Args = .{};
        a.add(x.ptr);
        a.add(out.ptr);
        a.add(i32of(x.rows));
        a.add(i32of(g));
        a.add(i32of(x.cols / g));
        try go(k.grid_major_fn, s, .{ .x = blocks(x.rows * x.cols, 256) }, .{ .x = 256 }, 0, &a);
    }
};

/// nn.RMSNorm(eps=None) on bf16 input with a bf16 weight in torch 2.14: the fused kernel's accumulate type (float)
/// epsilon, 2^-23.
pub const rms_eps: f32 = std.math.floatEps(f32);

test "rms epsilon is float32's, not bf16's" {
    try std.testing.expectEqual(@as(f32, 1.1920928955078125e-07), rms_eps);
}
