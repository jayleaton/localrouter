//! Launches of LocalRouter's own kernels (kernels/cuda/qwen_image/ops.cu, attention.cu, gemm.cu, te.cu) with the launch shapes the
//! twin's bindings use (`stk_twin/ops.py`, `stk_twin/attn.py`): same source, same grid, so the same bits.
//! Device pointers are u64; every launch goes on the caller's stream.

const std = @import("std");
const cuda = @import("cuda");

const Ptr = u64;
const block: u32 = 256;

fn blocks(n: u64) u32 {
    return @intCast((n + block - 1) / block);
}

pub const Ops = struct {
    module: cuda.Module,
    attn_module: cuda.Module,
    residual: cuda.Function,
    silu: cuda.Function,
    tanh: cuda.Function,
    gelu: cuda.Function,
    rms: cuda.Function,
    sinusoid: cuda.Function,
    euler_fn: cuda.Function,
    copy: cuda.Function,
    dense_fn: cuda.Function,
    widen: cuda.Function,
    transpose_fn: cuda.Function,
    attn: cuda.Function,

    /// pattn2_kernel<128, 4, 1, 2, 64, true, 3>: pattn_kernel's bits with faster staging (attention.cu)
    pub const attn_symbol = "_ZN13stk_attention13pattn2_kernelILi128ELi4ELi1ELi2ELi64ELb1ELi3EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiiiif";
    pub const attn_smem: u32 = 32768; // two 64-key staging slots of 128 dims
    pub const attn_rows: u32 = 64; // four warps of 16 query rows, one head a block
    pub const attn_threads: u32 = 128;

    /// `ops_image` / `attn_image`: the fatbins of ops.cu and attention.cu.
    pub fn load(d: *const cuda.Driver, ops_image: []const u8, attn_image: []const u8) !Ops {
        var m = try cuda.Module.load(d, ops_image);
        errdefer m.unload();
        var am = try cuda.Module.load(d, attn_image);
        errdefer am.unload();
        const attn = try am.function(attn_symbol);
        try attn.allowDynamicShared(attn_smem);
        return .{
            .module = m,
            .attn_module = am,
            .residual = try m.function("stk_gated_residual"),
            .silu = try m.function("stk_silu"),
            .tanh = try m.function("stk_tanh"),
            .gelu = try m.function("stk_gelu_tanh"),
            .rms = try m.function("stk_rms_norm_f32"),
            .sinusoid = try m.function("stk_time_sinusoid"),
            .euler_fn = try m.function("stk_euler"),
            .copy = try m.function("stk_copy_rows"),
            .dense_fn = try m.function("stk_dense_bf16"),
            .widen = try m.function("stk_bf16_to_f32"),
            .transpose_fn = try m.function("stk_transpose_bf16"),
            .attn = attn,
        };
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
        o.attn_module.unload();
    }

    pub fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, shared: u32, args: *cuda.launch.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = block }, .shared = shared }, s, args);
    }

    /// x[B, N, D] += a[B, N, D] * g[B, 1, D] (bf16, in place).
    pub fn gatedResidual(o: *const Ops, s: cuda.Stream, x: Ptr, a: Ptr, g: Ptr, b: u64, n: u64, dim: u64) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ x, a, g }) |p| args.add(p);
        args.add(@as(i64, @intCast(b * n * dim)));
        args.add(@as(i64, @intCast(n * dim)));
        args.add(@as(i64, @intCast(dim)));
        try go(o.residual, s, .{ .x = blocks(b * n * dim) }, 0, &args);
    }

    pub const Pointwise = enum { silu, tanh, gelu_tanh };

    /// y = fn(x) over n bf16 values (y may alias x).
    pub fn pointwise(o: *const Ops, s: cuda.Stream, f: Pointwise, x: Ptr, y: Ptr, n: u64) !void {
        var args: cuda.launch.Args = .{};
        args.add(x);
        args.add(y);
        args.add(@as(i64, @intCast(n)));
        const func = switch (f) {
            .silu => o.silu,
            .tanh => o.tanh,
            .gelu_tanh => o.gelu,
        };
        try go(func, s, .{ .x = blocks(n) }, 0, &args);
    }

    /// Rows [m, dim] bf16 -> RMSNorm with fp32 weight w[dim] -> bf16.
    pub fn rmsNorm(o: *const Ops, s: cuda.Stream, x: Ptr, w: Ptr, y: Ptr, m: u64, dim: u64, eps: f32) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ x, w, y }) |p| args.add(p);
        args.add(@as(i64, @intCast(dim)));
        args.add(eps);
        try go(o.rms, s, .{ .x = @intCast(m) }, 0, &args);
    }

    /// t[b] fp32 -> emb[b + 1, 256] bf16 (the last row is t = 0).
    pub fn timeSinusoid(o: *const Ops, s: cuda.Stream, t: Ptr, emb: Ptr, b: u64) !void {
        var args: cuda.launch.Args = .{};
        args.add(t);
        args.add(emb);
        args.add(@as(i64, @intCast(b)));
        try go(o.sinusoid, s, .{ .x = @intCast(b + 1) }, 0, &args);
    }

    /// y = bf16(x + dt * v), n values.
    pub fn euler(o: *const Ops, s: cuda.Stream, x: Ptr, v: Ptr, y: Ptr, n: u64, dt: f32) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ x, v, y }) |p| args.add(p);
        args.add(@as(i64, @intCast(n)));
        args.add(dt);
        try go(o.euler_fn, s, .{ .x = blocks(n) }, 0, &args);
    }

    /// `rows` rows of `row_bytes` from src (row stride src_stride) to dst (row stride dst_stride).
    pub fn copyRows(o: *const Ops, s: cuda.Stream, dst: Ptr, src: Ptr, rows: u64, row_bytes: u64, src_stride: u64, dst_stride: u64) !void {
        const aligned = row_bytes % 16 == 0 and src_stride % 16 == 0 and dst_stride % 16 == 0 and src % 16 == 0 and dst % 16 == 0;
        const lanes = if (aligned) row_bytes / 16 else row_bytes;
        var args: cuda.launch.Args = .{};
        args.add(dst);
        args.add(src);
        inline for (.{ row_bytes, src_stride, dst_stride }) |v| args.add(@as(i64, @intCast(v)));
        try go(o.copy, s, .{ .x = blocks(lanes), .y = @intCast(rows) }, 0, &args);
    }

    /// y[m, n] = x[m, k] @ w[n, k]^T (bf16, fp32 accumulation in k order).
    pub fn dense(o: *const Ops, s: cuda.Stream, x: Ptr, w: Ptr, y: Ptr, m: u64, n: u64, k: u64) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ x, w, y }) |p| args.add(p);
        inline for (.{ m, n, k }) |v| args.add(@as(i64, @intCast(v)));
        try go(o.dense_fn, s, .{ .x = @intCast((n + 63) / 64), .y = @intCast((m + 63) / 64) }, 0, &args);
    }

    /// y[i] = f32(x[i]), n values.
    pub fn toF32(o: *const Ops, s: cuda.Stream, x: Ptr, y: Ptr, n: u64) !void {
        var args: cuda.launch.Args = .{};
        args.add(x);
        args.add(y);
        args.add(@as(i64, @intCast(n)));
        try go(o.widen, s, .{ .x = blocks(n) }, 0, &args);
    }

    /// dst[cols, rows] = src[rows, cols]^T (bf16).
    pub fn transpose(o: *const Ops, s: cuda.Stream, src: Ptr, dst: Ptr, rows: u64, cols: u64) !void {
        var args: cuda.launch.Args = .{};
        args.add(src);
        args.add(dst);
        args.add(@as(i64, @intCast(rows)));
        args.add(@as(i64, @intCast(cols)));
        try go(o.transpose_fn, s, .{ .x = @intCast((cols + 31) / 32), .y = @intCast((rows + 31) / 32) }, 0, &args);
    }

    /// q[w, heads, 128], k / v[nkeys, kv_heads, 128] -> out[w, heads, 128] (query head h reads kv head
    /// h / (heads / kv_heads)); causal: query i (position p0 + i) sees keys 0..p0 + i, else every query sees all nkeys
    /// keys. Scale 1 / sqrt(128).
    pub fn attention(o: *const Ops, s: cuda.Stream, q: Ptr, k: Ptr, v: Ptr, out: Ptr, w: u32, heads: u32, kv_heads: u32, nkeys: u32, causal: bool, p0: u32) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ q, k, v, out }) |p| args.add(p);
        inline for (.{ p0, w, heads, kv_heads, heads / kv_heads, nkeys, @intFromBool(causal) }) |v_| args.add(@as(i32, @intCast(v_)));
        args.add(@as(f32, 1.0) / @sqrt(@as(f32, 128.0))); // as the C side: 1.0f / sqrtf(128.0f), in f32
        try cuda.launch.launch(o.attn, .{ .grid = .{ .x = (w + attn_rows - 1) / attn_rows, .y = heads }, .block = .{ .x = attn_threads }, .shared = attn_smem }, s, &args);
    }
};

/// The text encoder's kernels (te.cu) and the bf16 GEMM (gemm.cu), launched as `stk_twin/ops.py` does.
pub const TeOps = struct {
    te_module: cuda.Module,
    gemm_module: cuda.Module,
    rms: cuda.Function,
    rope: cuda.Function,
    silu_mul_fn: cuda.Function,
    add_fn: cuda.Function,
    embed_fn: cuda.Function,
    gemm_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, te_image: []const u8, gemm_image: []const u8) !TeOps {
        var m = try cuda.Module.load(d, te_image);
        errdefer m.unload();
        var g = try cuda.Module.load(d, gemm_image);
        errdefer g.unload();
        return .{
            .te_module = m,
            .gemm_module = g,
            .rms = try m.function("stk_rms_norm_hf"),
            .rope = try m.function("stk_rope_half"),
            .silu_mul_fn = try m.function("stk_silu_mul"),
            .add_fn = try m.function("stk_add"),
            .embed_fn = try m.function("stk_embed"),
            .gemm_fn = try g.function("stk_gemm_bf16"),
        };
    }

    pub fn unload(o: *TeOps) void {
        o.te_module.unload();
        o.gemm_module.unload();
    }

    /// c[m, n] = a[m, k] @ b[n, k]^T (+ bias[n], 0 for none), bf16, contiguous rows; k % 8 == 0.
    pub fn gemm(o: *const TeOps, s: cuda.Stream, a: Ptr, b: Ptr, bias: Ptr, c: Ptr, m: u64, n: u64, k: u64) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ a, b, bias, c }) |p| args.add(p);
        inline for (.{ m, n, k }) |v| args.add(@as(i32, @intCast(v)));
        inline for (.{ k, k, n }) |v| args.add(@as(i64, @intCast(v)));
        try Ops.go(o.gemm_fn, s, .{ .x = @intCast((n + 127) / 128), .y = @intCast((m + 127) / 128) }, 0, &args);
    }

    /// transformers' RMSNorm over rows of `dim` (weight bf16 [dim]): `rows` rows.
    pub fn rmsNorm(o: *const TeOps, s: cuda.Stream, x: Ptr, w: Ptr, y: Ptr, rows: u64, dim: u64, eps: f32) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ x, w, y }) |p| args.add(p);
        args.add(@as(i64, @intCast(dim)));
        args.add(eps);
        try Ops.go(o.rms, s, .{ .x = @intCast(rows) }, 0, &args);
    }

    /// In place on x[rows, heads, 128] with f32 tables cos / sin [rows, 64].
    pub fn ropeHalf(o: *const TeOps, s: cuda.Stream, x: Ptr, cos: Ptr, sin: Ptr, rows: u64, heads: u64) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ x, cos, sin }) |p| args.add(p);
        args.add(@as(i64, @intCast(rows)));
        args.add(@as(i64, @intCast(heads)));
        try Ops.go(o.rope, s, .{ .x = blocks(rows * heads * 64) }, 0, &args);
    }

    pub fn siluMul(o: *const TeOps, s: cuda.Stream, g: Ptr, u: Ptr, y: Ptr, n: u64) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ g, u, y }) |p| args.add(p);
        args.add(@as(i64, @intCast(n)));
        try Ops.go(o.silu_mul_fn, s, .{ .x = blocks(n) }, 0, &args);
    }

    /// y = x + z (y may alias x).
    pub fn add(o: *const TeOps, s: cuda.Stream, x: Ptr, z: Ptr, y: Ptr, n: u64) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ x, z, y }) |p| args.add(p);
        args.add(@as(i64, @intCast(n)));
        try Ops.go(o.add_fn, s, .{ .x = blocks(n) }, 0, &args);
    }

    /// out[r] = table[ids[r]] for `rows` i32 ids, rows of `dim` bf16.
    pub fn embed(o: *const TeOps, s: cuda.Stream, table: Ptr, ids: Ptr, out: Ptr, rows: u64, dim: u64) !void {
        var args: cuda.launch.Args = .{};
        inline for (.{ table, ids, out }) |p| args.add(p);
        args.add(@as(i64, @intCast(dim)));
        try Ops.go(o.embed_fn, s, .{ .x = blocks(dim), .y = @intCast(rows) }, 0, &args);
    }
};
