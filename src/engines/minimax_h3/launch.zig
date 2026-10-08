//! Launches of the H3 kernels: LocalRouter's (kernels/cuda/minimax/ops.cu) with the twin's grids
//! (`stk_twin/h3/ops.py`), and comfy-kitchen's INT8 attention and fused RMSNorm + RoPE with the wheel's launch
//! sequence (kernels/cuda/minimax/kitchen_launch.cu documents it: kernels, template arguments, grids, workspace).

const std = @import("std");
const cuda = @import("cuda");

const Ptr = u64;
const block: u32 = 256;

fn blocks(n: u64) u32 {
    return @intCast((n + block - 1) / block);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, blk: cuda.launch.Dim3, shared: u32, args: *cuda.launch.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = blk, .shared = shared }, s, args);
}

fn i64of(v: anytype) i64 {
    return @intCast(v);
}

pub const Ops = struct {
    module: cuda.Module,
    norm_mod_fn: cuda.Function,
    gate_add_fn: cuda.Function,
    swiglu_fn: cuda.Function,
    silu_mul_fn: cuda.Function,
    rms_fn: cuda.Function,
    final_mod_fn: cuda.Function,
    scale_fn: cuda.Function,
    uncarry_fn: cuda.Function,
    patchify_fn: cuda.Function,
    unpatchify_fn: cuda.Function,
    pack_audio_fn: cuda.Function,
    unpack_audio_fn: cuda.Function,
    add_fn: cuda.Function,
    gemm_fn: cuda.Function,
    to_bf16_fn: cuda.Function,
    heads_fn: cuda.Function,
    denoise_fn: cuda.Function,
    euler_fn: cuda.Function,
    res2_fn: cuda.Function,
    scale32_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var o: Ops = undefined;
        o.module = m;
        inline for (.{
            .{ "norm_mod_fn", "h3_norm_mod" },             .{ "gate_add_fn", "h3_gate_add" },     .{ "swiglu_fn", "h3_swiglu" },
            .{ "silu_mul_fn", "h3_silu_mul_split" },       .{ "rms_fn", "h3_rms_norm" },          .{ "final_mod_fn", "h3_final_mod" },
            .{ "scale_fn", "h3_scale" },                   .{ "uncarry_fn", "h3_uncarry" },       .{ "patchify_fn", "h3_patchify" },
            .{ "unpatchify_fn", "h3_unpatchify_neg" },     .{ "pack_audio_fn", "h3_pack_audio" }, .{ "unpack_audio_fn", "h3_unpack_audio_neg" },
            .{ "add_fn", "h3_add" },                       .{ "gemm_fn", "h3_gemm_f32" },         .{ "to_bf16_fn", "h3_f32_to_bf16" },
            .{ "heads_fn", "h3_heads_to_rows" },          .{ "denoise_fn", "h3_denoise" },       .{ "euler_fn", "h3_euler32" },
            .{ "res2_fn", "h3_res2" },                    .{ "scale32_fn", "h3_scale32" },
        }) |e| @field(o, e[0]) = try m.function(e[1]);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
    }

    /// y[S, D] = tfvideo's norm_mod of x under the rows' modulation (part 0: attention, 1: MLP); y may alias x.
    pub fn normMod(o: *const Ops, s: cuda.Stream, x: Ptr, w: Ptr, mod: Ptr, idx: Ptr, y: Ptr, rows: u64, dim: u64, part: i32, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, mod, idx, y }) |p| a.add(p);
        a.add(i64of(dim));
        a.add(part);
        a.add(eps);
        try go(o.norm_mod_fn, s, .{ .x = @intCast(rows) }, .{ .x = block }, 0, &a);
    }

    /// x += y * gate (part 0: chunk 2, 1: chunk 5), in place.
    pub fn gateAdd(o: *const Ops, s: cuda.Stream, x: Ptr, y: Ptr, mod: Ptr, idx: Ptr, rows: u64, dim: u64, part: i32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, y, mod, idx }) |p| a.add(p);
        a.add(i64of(dim));
        a.add(part);
        try go(o.gate_add_fn, s, .{ .x = blocks(dim), .y = @intCast(rows) }, .{ .x = block }, 0, &a);
    }

    /// [M, 2F] -> [M, F]: tfvideo's swiglu (`silu_split` false) or ComfyUI's (true: SiLU rounded to bf16 first).
    pub fn swiglu(o: *const Ops, s: cuda.Stream, gu: Ptr, out: Ptr, m: u64, f: u64, comfy: bool) !void {
        var a: cuda.launch.Args = .{};
        a.add(gu);
        a.add(out);
        a.add(i64of(f));
        try go(if (comfy) o.silu_mul_fn else o.swiglu_fn, s, .{ .x = blocks(f), .y = @intCast(m) }, .{ .x = block }, 0, &a);
    }

    /// RMSNorm over rows of `dim` with a bf16 weight; y may alias x.
    pub fn rmsNorm(o: *const Ops, s: cuda.Stream, x: Ptr, w: Ptr, y: Ptr, rows: u64, dim: u64, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, y }) |p| a.add(p);
        a.add(i64of(dim));
        a.add(eps);
        try go(o.rms_fn, s, .{ .x = @intCast(rows) }, .{ .x = block }, 0, &a);
    }

    /// The final layer's fp32 modulation of `rows` rows (scale / shift: the segment's fp32 modulation row).
    pub fn finalMod(o: *const Ops, s: cuda.Stream, x: Ptr, w: Ptr, scale_row: Ptr, shift: Ptr, out: Ptr, rows: u64, dim: u64, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, scale_row, shift, out }) |p| a.add(p);
        a.add(i64of(dim));
        a.add(eps);
        try go(o.final_mod_fn, s, .{ .x = @intCast(rows) }, .{ .x = block }, 0, &a);
    }

    pub fn scale(o: *const Ops, s: cuda.Stream, x: Ptr, c: f32, y: Ptr, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(c);
        a.add(y);
        a.add(i64of(n));
        try go(o.scale_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    pub fn uncarry(o: *const Ops, s: cuda.Stream, carried: Ptr, v: Ptr, c1: f32, c2: f32, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(carried);
        a.add(v);
        a.add(c1);
        a.add(c2);
        a.add(i64of(n));
        try go(o.uncarry_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    /// video bf16 [C, T, H, W] -> rows fp32 [T*(H/2)*(W/2), 4C].
    pub fn patchify(o: *const Ops, s: cuda.Stream, x: Ptr, rows: Ptr, c: u32, t: u32, h: u32, w: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(rows);
        inline for (.{ c, t, h, w }) |v| a.add(@as(i32, @intCast(v)));
        try go(o.patchify_fn, s, .{ .x = blocks(@as(u64, t) * (h / 2) * (w / 2) * c * 4) }, .{ .x = block }, 0, &a);
    }

    pub fn unpatchifyNeg(o: *const Ops, s: cuda.Stream, rows: Ptr, x: Ptr, c: u32, t: u32, h: u32, w: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(rows);
        a.add(x);
        inline for (.{ c, t, h, w }) |v| a.add(@as(i32, @intCast(v)));
        try go(o.unpatchify_fn, s, .{ .x = blocks(@as(u64, c) * t * h * w) }, .{ .x = block }, 0, &a);
    }

    pub fn packAudio(o: *const Ops, s: cuda.Stream, x: Ptr, rows: Ptr, c: u32, n: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(rows);
        a.add(@as(i32, @intCast(c)));
        a.add(@as(i32, @intCast(n)));
        try go(o.pack_audio_fn, s, .{ .x = blocks(2 * @as(u64, n) * c) }, .{ .x = block }, 0, &a);
    }

    pub fn unpackAudioNeg(o: *const Ops, s: cuda.Stream, rows: Ptr, x: Ptr, c: u32, n: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(rows);
        a.add(x);
        a.add(@as(i32, @intCast(c)));
        a.add(@as(i32, @intCast(n)));
        try go(o.unpack_audio_fn, s, .{ .x = blocks(2 * @as(u64, n) * c) }, .{ .x = block }, 0, &a);
    }

    pub fn add(o: *const Ops, s: cuda.Stream, x: Ptr, z: Ptr, y: Ptr, n: u64) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, z, y }) |p| a.add(p);
        a.add(i64of(n));
        try go(o.add_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    /// fp32 c[m, n] = a[m, k] . b[n, k]^T (+ bias, 0 for none), k in order.
    pub fn gemmF32(o: *const Ops, s: cuda.Stream, a_: Ptr, b: Ptr, bias: Ptr, c: Ptr, m: u64, n: u64, k: u64) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ a_, b, bias, c }) |p| a.add(p);
        inline for (.{ m, n, k }) |v| a.add(i64of(v));
        try go(o.gemm_fn, s, .{ .x = @intCast((n + 63) / 64), .y = @intCast((m + 63) / 64) }, .{ .x = block }, 0, &a);
    }

    pub fn toBf16(o: *const Ops, s: cuda.Stream, x: Ptr, y: Ptr, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(y);
        a.add(i64of(n));
        try go(o.to_bf16_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    /// den = x - float(out) * sigma over the packed fp32 state (out: the bf16 velocities packed alike).
    pub fn denoise(o: *const Ops, s: cuda.Stream, x: Ptr, out: Ptr, sigma: f32, den: Ptr, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(out);
        a.add(sigma);
        a.add(den);
        a.add(i64of(n));
        try go(o.denoise_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    /// x += ((x - den) / sigma) * dt.
    pub fn euler32(o: *const Ops, s: cuda.Stream, x: Ptr, den: Ptr, sigma: f32, dt: f32, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(den);
        a.add(sigma);
        a.add(dt);
        a.add(i64of(n));
        try go(o.euler_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    /// x = e * x + h * (b1 * den + b2 * old).
    pub fn res2(o: *const Ops, s: cuda.Stream, x: Ptr, den: Ptr, old: Ptr, e: f32, h: f32, b1: f32, b2: f32, n: u64) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, den, old }) |p| a.add(p);
        inline for (.{ e, h, b1, b2 }) |v| a.add(v);
        a.add(i64of(n));
        try go(o.res2_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    /// x *= c (fp32, in place).
    pub fn scale32(o: *const Ops, s: cuda.Stream, x: Ptr, c: f32, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(c);
        a.add(x);
        a.add(i64of(n));
        try go(o.scale32_fn, s, .{ .x = blocks(n) }, .{ .x = block }, 0, &a);
    }

    /// [H, S, D] -> [S, H * D].
    pub fn headsToRows(o: *const Ops, s: cuda.Stream, x: Ptr, y: Ptr, h: u64, rows: u64, dim: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(y);
        inline for (.{ h, rows, dim }) |v| a.add(i64of(v));
        try go(o.heads_fn, s, .{ .x = blocks(h * rows * dim) }, .{ .x = block }, 0, &a);
    }
};

/// comfy-kitchen's INT8 attention (D 128, no mask, H == HK, batch 1) and fused per-head RMSNorm + split-half RoPE.
pub const Kitchen = struct {
    module: cuda.Module,
    anchor: cuda.Function,
    qk_c64_rot4: cuda.Function,
    qk_c64_rot128: cuda.Function,
    qk_c128: cuda.Function,
    v128: cuda.Function,
    v512: cuda.Function,
    attn_c64: cuda.Function,
    attn_c64_fuse: cuda.Function,
    attn_c128_fuse: cuda.Function,
    rope: cuda.Function,

    const sym_anchor = "_ZN16kitchen_quant_qk15detect_k_anchorI13__nv_bfloat16EEvPKT_Piiiilll";
    const sym_qk_c64_rot4 = "_ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi8ELi128ELi32ELi64ELi64ELi1ELi4ELb1EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll";
    const sym_qk_c64_rot128 = "_ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi8ELi128ELi32ELi64ELi64ELi1ELi128ELb1EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll";
    const sym_qk_c128 = "_ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi16ELi128ELi32ELi128ELi128ELi1ELi128ELb1EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll";
    const sym_v128 = "_ZN15kitchen_quant_v19quant_v_int8_kernelI13__nv_bfloat16Li128EEEvPKT_PaPfiiiilll";
    const sym_v512 = "_ZN15kitchen_quant_v19quant_v_int8_kernelI13__nv_bfloat16Li512EEEvPKT_PaPfiiiilll";
    const sym_attn_c64 = "_Z24qk_int_sv_i8_attn_kernelILj128ELj64ELj16ELj64ELj128EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb0EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf";
    const sym_attn_c64_fuse = "_Z24qk_int_sv_i8_attn_kernelILj128ELj64ELj16ELj64ELj128EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb1EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf";
    const sym_attn_c128_fuse = "_Z24qk_int_sv_i8_attn_kernelILj128ELj128ELj16ELj128ELj128EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb1EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf";
    const sym_rope = "_ZN5comfy16kitchen_rms_rope11rope_kernelI13__nv_bfloat16S2_S2_Lb1ELb1ELb1ELb1ELb1EEEvPKT_S5_PKT0_PKT1_SB_PS3_SC_llliilllllllllllllllllllllllllllf";

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kitchen {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var k: Kitchen = undefined;
        k.module = m;
        inline for (.{
            .{ "anchor", sym_anchor },               .{ "qk_c64_rot4", sym_qk_c64_rot4 }, .{ "qk_c64_rot128", sym_qk_c64_rot128 },
            .{ "qk_c128", sym_qk_c128 },             .{ "v128", sym_v128 },               .{ "v512", sym_v512 },
            .{ "attn_c64", sym_attn_c64 },           .{ "attn_c64_fuse", sym_attn_c64_fuse },
            .{ "attn_c128_fuse", sym_attn_c128_fuse }, .{ "rope", sym_rope },
        }) |e| @field(k, e[0]) = try m.function(e[1]);
        try k.attn_c64.allowDynamicShared(32768);
        try k.attn_c64_fuse.allowDynamicShared(32768);
        try k.attn_c128_fuse.allowDynamicShared(49152);
        return k;
    }

    pub fn unload(k: *Kitchen) void {
        k.module.unload();
    }

    /// The INT8 attention's plan for S keys and queries, head dim 128, `heads` heads: cta_k, padded_k and the
    /// workspace parts (each 256-byte aligned, in the wheel's order).
    pub const Plan = struct {
        cta_k: u32,
        padded_k: u32,
        q8: u64,
        k8: u64,
        v8: u64,
        qs: u64,
        ks: u64,
        vs: u64,
        anchor: u64,
        total: u64,

        pub fn of(heads: u32, s: u32) Plan {
            const d: u64 = 128;
            const cta_k: u32 = if (s > 1024) 128 else 64;
            const padded_k = (s + cta_k - 1) / cta_k * cta_k;
            const q_scales: u64 = @as(u64, (s + 127) / 128) * 32;
            const k_scales: u64 = @as(u64, (s + cta_k - 1) / cta_k) * 4;
            var p: Plan = .{ .cta_k = cta_k, .padded_k = padded_k, .q8 = 0, .k8 = 0, .v8 = 0, .qs = 0, .ks = 0, .vs = 0, .anchor = 0, .total = 0 };
            var at: u64 = 0;
            inline for (.{
                .{ "q8", @as(u64, heads) * s * d },     .{ "k8", @as(u64, heads) * s * d }, .{ "v8", @as(u64, heads) * d * padded_k },
                .{ "qs", @as(u64, heads) * q_scales * 4 }, .{ "ks", @as(u64, heads) * k_scales * 4 }, .{ "vs", @as(u64, heads) * d * 4 },
                .{ "anchor", @as(u64, heads) * 4 },
            }) |e| {
                @field(p, e[0]) = at;
                at = (at + e[1] + 255) / 256 * 256;
            }
            p.total = at;
            return p;
        }
    };

    /// softmax(q k^T / sqrt(128)) v over S tokens: q, k, v views into one qkv buffer [S, 3 * heads * 128] (row stride
    /// 3 * heads * 128), out [heads, S, 128] bf16 contiguous; `ws` at least `Plan.of(heads, S).total` bytes.
    pub fn attention(k: *const Kitchen, s: cuda.Stream, qkv: Ptr, out: Ptr, ws: Ptr, heads: u32, seq: u32) !void {
        const d: u32 = 128;
        const p = Plan.of(heads, seq);
        const row: i64 = 3 * @as(i64, heads) * d; // the qkv buffer's row stride (elements)
        const q = qkv;
        const kk = qkv + @as(u64, heads) * d * 2;
        const v = qkv + 2 * @as(u64, heads) * d * 2;
        const sb: i64 = @as(i64, seq) * row; // batch stride (batch 1: unused)
        const sh: i64 = d; // the head stride within a row
        const sn: i64 = row;
        { // 1. the representative key per head
            var a: cuda.launch.Args = .{};
            a.add(kk);
            a.add(ws + p.anchor);
            a.add(@as(i32, @intCast(seq)));
            a.add(@as(i32, @intCast(d)));
            a.add(@as(i32, @intCast(heads)));
            inline for (.{ sb, sh, sn }) |x| a.add(x);
            try go(k.anchor, s, .{ .x = heads, .y = 1 }, .{ .x = 128 }, 0, &a);
        }
        const q_oblk: u32 = (seq + 127) / 128 * 4;
        const k_oblk: u32 = (seq + p.cta_k - 1) / p.cta_k;
        { // 2. Q and K to INT8 with the Hadamard rotation (H4 at S <= 256, else the signed H128)
            var a: cuda.launch.Args = .{};
            inline for (.{ q, ws + p.q8, ws + p.qs, kk, ws + p.k8, ws + p.ks, ws + p.anchor }) |x| a.add(x);
            inline for (.{ seq, seq, d, q_oblk, heads, heads, q_oblk * 8, k_oblk * 4 }) |x| a.add(@as(i32, @intCast(x)));
            inline for (.{ sb, sh, sn, sb, sh, sn }) |x| a.add(x);
            const f = if (p.cta_k == 128) k.qk_c128 else if (seq <= 256) k.qk_c64_rot4 else k.qk_c64_rot128;
            try go(f, s, .{ .x = q_oblk + k_oblk, .y = heads, .z = 1 }, .{ .x = 128 }, 0, &a);
        }
        { // 3. V to INT8, per channel, padded to padded_k
            var a: cuda.launch.Args = .{};
            inline for (.{ v, ws + p.v8, ws + p.vs }) |x| a.add(x);
            inline for (.{ seq, p.padded_k, heads, d }) |x| a.add(@as(i32, @intCast(x)));
            inline for (.{ sb, sh, sn }) |x| a.add(x);
            const big = seq > 256;
            try go(if (big) k.v512 else k.v128, s, .{ .x = heads * (d / 8) }, .{ .x = if (big) 512 else 128 }, 0, &a);
        }
        { // 4. the attention kernel
            var a: cuda.launch.Args = .{};
            inline for (.{ ws + p.q8, ws + p.k8, ws + p.v8, out, @as(u64, 0), ws + p.qs, ws + p.ks, ws + p.vs, @as(u64, 0), @as(u64, 0) }) |x| a.add(x);
            inline for (.{ @as(i64, 0), @as(i64, 0), @as(i64, 0), @as(i64, 0) }) |x| a.add(x);
            a.add(@as(i32, -1));
            const hsd: u32 = heads * seq * d;
            inline for (.{ seq, seq, @as(u32, 1), hsd, d, seq * d, hsd, d, seq * d, heads * d * p.padded_k, d * p.padded_k, p.padded_k, hsd, d, seq * d }) |x| a.add(@as(u32, x));
            a.add(@as(f32, @bitCast(@as(u32, 0x3db504f3)))); // 128 ** -0.5 rounded to f32
            const fuse = !(seq <= 512 and p.cta_k == 64);
            const f = if (p.cta_k == 128) k.attn_c128_fuse else if (fuse) k.attn_c64_fuse else k.attn_c64;
            const smem: u32 = if (p.cta_k == 128) 49152 else 32768;
            try go(f, s, .{ .x = (seq + 127) / 128, .y = heads, .z = 1 }, .{ .x = 32, .y = 8 }, smem, &a);
        }
    }

    /// In place on q, k (views into the qkv buffer [S, 3 * heads * 128]): per-head RMSNorm (bf16 weights) then
    /// split-half RoPE on the first `rot` dims from the bf16 table [S, rot/2, 2, 2].
    pub fn rmsRope(k: *const Kitchen, s: cuda.Stream, qkv: Ptr, freqs: Ptr, qw: Ptr, kw: Ptr, heads: u32, seq: u32, rot: u32, eps: f32) !void {
        const d: i64 = 128;
        const row: i64 = 3 * @as(i64, heads) * d;
        const q = qkv;
        const kk = qkv + @as(u64, heads) * 128 * 2;
        const fs: i64 = @as(i64, rot) * 2; // a row of the table: rot/2 2x2 matrices
        var a: cuda.launch.Args = .{};
        inline for (.{ q, kk, freqs, qw, kw, q, kk }) |x| a.add(x);
        inline for (.{ @as(i64, 1), @as(i64, seq), @as(i64, heads) }) |x| a.add(x);
        a.add(@as(i32, 128));
        a.add(@as(i32, @intCast(rot)));
        inline for (.{ @as(i64, 1), @as(i64, seq), @as(i64, 1) }) |x| a.add(x);
        const sq = .{ @as(i64, seq) * row, row, d, @as(i64, 1) };
        inline for (0..4) |_| inline for (sq) |x| a.add(x); // q, k, q_out, k_out strides (one qkv buffer)
        inline for (.{ @as(i64, seq) * fs, fs, fs, @as(i64, 4), @as(i64, 2), @as(i64, 1), @as(i64, 1), @as(i64, 1) }) |x| a.add(x);
        a.add(eps);
        const n: u64 = @as(u64, seq) * heads;
        try go(k.rope, s, .{ .x = @intCast((n + 3) / 4) }, .{ .x = 128 }, 0, &a);
    }
};
