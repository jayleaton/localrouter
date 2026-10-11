//! Kandinsky-6's dense text conditioning: Qwen2.5-VL-7B's text stack, text-only, op for op as upstream runs it
//! (kandinsky-6 eac06e39 text_embedder.py:155-173 with transformers v5.18.0 models/qwen2_5_vl/modeling_qwen2_5_vl.py),
//! for the padded 1153-token prompt of `prompt.qwenIds`, producing `hidden_states[-1][:, 129:]`: [1024, 3584] bf16.
//!
//! Upstream facts this file depends on (transformers v5.18.0):
//!   - `hidden_states[-1]` is the output of the final RMSNorm: `capture_outputs` replaces the last captured layer
//!     output with `last_hidden_state` (utils/output_capturing.py:292-305, tie_last_hidden_states default True;
//!     Qwen2_5_VLTextModel.forward modeling_qwen2_5_vl.py:877).
//!   - Positions are 0..1152 on all three MRoPE axes, padding included: upstream passes no mm_token_type_ids, so
//!     compute_3d_position_ids returns None (modeling_qwen2_5_vl.py:1155-1189) and the text model uses arange
//!     (:821-824). The [16, 24, 24] recomposition (:565-570) then selects equal values: plain RoPE, rope_type "default"
//!     (configuration_qwen2_5_vl.py:154-155, legacy "mrope" -> "default"), theta 1e6.
//!   - Attention: sdpa with the 4D bool mask causal AND key padding (masking_utils.py:170-180, 502-511) whenever the
//!     prompt is padded; with no padding (1153 real tokens) the mask is skipped and SDPA runs is_causal
//!     (masking_utils.py:237-284, integrations/sdpa_attention.py). Both are "query i sees keys j <= i with j < valid".
//!   - Decoder layer (:718-768): RMSNorm, q/k/v Linear with bias, RoPE (:590-613) in bf16, attention scale 128^-0.5,
//!     o_proj without bias, residual, RMSNorm, SiLU-gated MLP (:573-587), residual. RMSNorm (:71-76) in fp32 then cast.
//! Precision forms that native leaves to run-time library choices are explicit, required parameters (`Fp32Math`);
//! nothing here is claimed bit-equal to native until the SM121 fixtures listed in kernels/cuda/kandinsky6/text/README.md
//! pass.
//!
//! Kernels: kernels/cuda/kandinsky6/text/text.cu (RoPE table, RMSNorm, the CLIP ops) and the image engine's te.cu
//! (stk_embed, stk_rope_half, stk_silu_mul, stk_add), gemm.cu (stk_gemm_bf16) and attention.cu (pattn2), whose
//! semantics match these ops exactly (README.md lists each). The encoder borrows the driver, stream, weights and
//! scratch; it never allocates device memory, synchronizes or loads files.

const std = @import("std");
const cuda = @import("cuda");

pub const layers: u32 = 28;
pub const dim: u32 = 3584;
pub const heads: u32 = 28;
pub const kv_heads: u32 = 4;
pub const head_dim: u32 = 128;
pub const ffn: u32 = 18944;
pub const vocab: u32 = 152064;
pub const eps: f32 = 1e-6;
pub const theta: f64 = 1e6;
const kv: u32 = kv_heads * head_dim;

/// The padded sequence, the crop and the output rows (= prompt.qwen_tokens / qwen_crop / qwen_rows).
pub const tokens: u32 = 1153;
pub const crop: u32 = 129;
pub const rows: u32 = tokens - crop;

/// 128^-0.5 as SDPA receives it (`self.scaling`, a Python float, rounded to fp32): 0x3db504f3.
pub const attn_scale: f32 = @bitCast(@as(u32, 0x3db504f3));

pub const Error = cuda.Error || error{ MissingWeight, MissingKernels, InvalidPrompt };

/// The arithmetic of an fp32 matmul that native runs under `torch.set_float32_matmul_precision("high")`
/// (kandinsky/pipeline/factory.py:162), which permits TF32 without fixing the algorithm SM121 runs. Same values and
/// meaning as the DiT's `launch.Fp32Linear`: TF32 operands rounded to nearest (away) or truncated, or strict FP32.
/// None is a default and none is qualified (README.md).
pub const Fp32Math = enum(i32) { tf32_rna = 0, tf32_truncate = 1, fp32 = 2 };

pub const Dtype = enum { bf16, f32 };

/// Called after each op with its output (device address, bytes), named by the upstream module path below
/// `model.language_model` (CLIP: below the CLIPTextModel). For per-op captures and fixture injection: the hook runs
/// before anything reads the output, so a parity harness may overwrite it in stream order.
pub const Hook = struct {
    ctx: *anyopaque,
    after: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, bytes: u64) anyerror!void,
};

// ---- kernels ------------------------------------------------------------------------------------------------------

/// The fatbin images: this component's text.cu and the image engine's te.cu, gemm.cu, attention.cu
/// (`qwen_image.kernels.{te, gemm, attention}`). Empty slices: no kernels; `Kernels.load` refuses them.
pub const Images = struct {
    text: []const u8,
    qwen_te: []const u8,
    qwen_gemm: []const u8,
    qwen_attention: []const u8,
};

/// pattn2_kernel<128, 4, 1, 2, 64, true, 3> (src/engines/qwen_image/ops_launch.zig Ops.attn_symbol).
const pattn_symbol = "_ZN13stk_attention13pattn2_kernelILi128ELi4ELi1ELi2ELi64ELb1ELi3EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiiiif";
const pattn_smem: u32 = 32768;
const pattn_rows: u32 = 64;
const pattn_threads: u32 = 128;

fn blocks(n: u64, per: u64) u32 {
    return @intCast((n + per - 1) / per);
}

fn i64of(v: anytype) i64 {
    return @intCast(v);
}

fn i32of(v: anytype) i32 {
    return @intCast(v);
}

/// Module load and one launcher a kernel (the grids are the ones documented beside each kernel). Shared with clip.zig.
pub const Kernels = struct {
    d: *const cuda.Driver,
    text_mod: cuda.Module,
    te_mod: cuda.Module,
    gemm_mod: cuda.Module,
    attn_mod: cuda.Module,
    // text.cu
    rope_table_fn: cuda.Function,
    rms_fn: cuda.Function,
    clip_embed_fn: cuda.Function,
    ln_fn: cuda.Function,
    linear_fn: cuda.Function,
    quick_gelu_fn: cuda.Function,
    add_f32_fn: cuda.Function,
    attn_f32_fn: cuda.Function,
    cast_fn: cuda.Function,
    // the image engine's
    embed_fn: cuda.Function,
    rope_half_fn: cuda.Function,
    silu_mul_fn: cuda.Function,
    add_bf16_fn: cuda.Function,
    gemm_fn: cuda.Function,
    pattn_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, images: Images) Error!Kernels {
        if (images.text.len == 0 or images.qwen_te.len == 0 or images.qwen_gemm.len == 0 or images.qwen_attention.len == 0)
            return error.MissingKernels;
        var k: Kernels = undefined;
        k.d = d;
        k.text_mod = try cuda.Module.load(d, images.text);
        errdefer k.text_mod.unload();
        k.te_mod = try cuda.Module.load(d, images.qwen_te);
        errdefer k.te_mod.unload();
        k.gemm_mod = try cuda.Module.load(d, images.qwen_gemm);
        errdefer k.gemm_mod.unload();
        k.attn_mod = try cuda.Module.load(d, images.qwen_attention);
        errdefer k.attn_mod.unload();
        inline for (.{
            .{ "rope_table_fn", "k6t_rope_table" }, .{ "rms_fn", "k6t_rms_norm" },           .{ "clip_embed_fn", "k6t_clip_embed" },
            .{ "ln_fn", "k6t_layer_norm_f32" },     .{ "linear_fn", "k6t_linear_f32" },      .{ "quick_gelu_fn", "k6t_quick_gelu" },
            .{ "add_f32_fn", "k6t_add_f32" },       .{ "attn_f32_fn", "k6t_attention_f32" }, .{ "cast_fn", "k6t_cast_bf16" },
        }) |e| @field(k, e[0]) = try k.text_mod.function(e[1]);
        inline for (.{
            .{ "embed_fn", "stk_embed" }, .{ "rope_half_fn", "stk_rope_half" }, .{ "silu_mul_fn", "stk_silu_mul" }, .{ "add_bf16_fn", "stk_add" },
        }) |e| @field(k, e[0]) = try k.te_mod.function(e[1]);
        k.gemm_fn = try k.gemm_mod.function("stk_gemm_bf16");
        k.pattn_fn = try k.attn_mod.function(pattn_symbol);
        try k.pattn_fn.allowDynamicShared(pattn_smem);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.attn_mod.unload();
        k.gemm_mod.unload();
        k.te_mod.unload();
        k.text_mod.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, blk: cuda.launch.Dim3, shared: u32, args: *cuda.launch.Args) Error!void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = blk, .shared = shared }, s, args);
    }

    /// Host bytes to device, in stream order (pageable source: staged before the call returns).
    pub fn upload(k: *const Kernels, s: cuda.Stream, dst: u64, bytes: []const u8) Error!void {
        try k.d.check(k.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
    }

    /// cos / sin [n, 64] f32 for positions 0..n-1 from inv_freq [64] f32 (k6t_rope_table).
    pub fn ropeTable(k: *const Kernels, s: cuda.Stream, inv: u64, cos: u64, sin: u64, n: u32, math: Fp32Math) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ inv, cos, sin }) |p| a.add(p);
        a.add(i32of(n));
        a.add(@backingInt(math));
        try go(k.rope_table_fn, s, .{ .x = blocks(@as(u64, n) * 64, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// Qwen2_5_VLRMSNorm over `n` bf16 rows of `d` (row strides ldx, ldy elements), weight bf16 [d].
    pub fn rmsNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, y: u64, n: u32, d: u32, ldx: u64, ldy: u64) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, y }) |p| a.add(p);
        a.add(i32of(d));
        a.add(i64of(ldx));
        a.add(i64of(ldy));
        a.add(eps);
        a.add(@as(f32, 1.0) / @as(f32, @floatFromInt(d))); // torch's MeanOps factor, 1/N in fp32
        try go(k.rms_fn, s, .{ .x = n }, .{ .x = 256 }, 0, &a);
    }

    /// out[r] = table[ids[r]], `n` rows of `d` bf16 (te.cu stk_embed).
    pub fn embed(k: *const Kernels, s: cuda.Stream, table: u64, ids: u64, out: u64, n: u32, d: u32) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ table, ids, out }) |p| a.add(p);
        a.add(i64of(d));
        try go(k.embed_fn, s, .{ .x = blocks(d, 256), .y = n }, .{ .x = 256 }, 0, &a);
    }

    /// In place on x[n, h, 128] bf16 with f32 tables cos / sin [n, 64] (te.cu stk_rope_half).
    pub fn ropeHalf(k: *const Kernels, s: cuda.Stream, x: u64, cos: u64, sin: u64, n: u32, h: u32) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, cos, sin }) |p| a.add(p);
        a.add(i64of(n));
        a.add(i64of(h));
        try go(k.rope_half_fn, s, .{ .x = blocks(@as(u64, n) * h * 64, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// y = bf16(bf16(silu(g)) * u), n values (te.cu stk_silu_mul); y may alias g.
    pub fn siluMul(k: *const Kernels, s: cuda.Stream, g: u64, u: u64, y: u64, n: u64) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ g, u, y }) |p| a.add(p);
        a.add(i64of(n));
        try go(k.silu_mul_fn, s, .{ .x = blocks(n, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// y = bf16(x + z), n values (te.cu stk_add); y may alias x.
    pub fn addBf16(k: *const Kernels, s: cuda.Stream, x: u64, z: u64, y: u64, n: u64) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, z, y }) |p| a.add(p);
        a.add(i64of(n));
        try go(k.add_bf16_fn, s, .{ .x = blocks(n, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// c[m, n] = a[m, kd] . b[n, kd]^T (+ bias[n], 0 for none), bf16, contiguous (gemm.cu stk_gemm_bf16).
    pub fn gemm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, bias: u64, c: u64, m: u32, n: u32, kd: u32) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, bias, c }) |p| a.add(p);
        inline for (.{ m, n, kd }) |v| a.add(i32of(v));
        inline for (.{ kd, kd, n }) |v| a.add(i64of(v));
        try go(k.gemm_fn, s, .{ .x = blocks(n, 128), .y = blocks(m, 128) }, .{ .x = 256 }, 0, &a);
    }

    /// q[n, h, 128], k / v[n, hk, 128] -> out[n, h, 128] bf16; query i sees keys j <= i with j < nkeys (attention.cu
    /// pattn2, `vis = c < nkeys && c <= pos`), GQA group h / hk, scale `attn_scale`.
    pub fn attention(k: *const Kernels, s: cuda.Stream, q: u64, kk: u64, v: u64, out: u64, n: u32, h: u32, hk: u32, nkeys: u32) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ q, kk, v, out }) |p| a.add(p);
        inline for (.{ 0, n, h, hk, h / hk, nkeys, 1 }) |x| a.add(i32of(x));
        a.add(attn_scale);
        try go(k.pattn_fn, s, .{ .x = blocks(n, pattn_rows), .y = h }, .{ .x = pattn_threads }, pattn_smem, &a);
    }

    // CLIP (fp32)

    pub fn clipEmbed(k: *const Kernels, s: cuda.Stream, tok: u64, pos: u64, ids: u64, out: u64, n: u32, d: u32) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ tok, pos, ids, out }) |p| a.add(p);
        a.add(i32of(d));
        try go(k.clip_embed_fn, s, .{ .x = blocks(d, 256), .y = n }, .{ .x = 256 }, 0, &a);
    }

    /// nn.LayerNorm on `n` fp32 rows of `d` (d % 4 == 0), affine; y must not alias x.
    pub fn layerNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, n: u32, d: u32, ldx: u64, ldy: u64, ln_eps: f32) Error!void {
        if (d % 4 != 0) return error.Invalid;
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, b, y }) |p| a.add(p);
        a.add(i32of(d));
        a.add(i64of(ldx));
        a.add(i64of(ldy));
        a.add(ln_eps);
        try go(k.ln_fn, s, .{ .x = n }, .{ .x = 32, .y = 4 }, 0, &a);
    }

    /// c[m, n] = x[m, kd] . w[n, kd]^T + bias[n] in fp32, `math` the operand form.
    pub fn linearF32(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, bias: u64, c: u64, m: u32, n: u32, kd: u32, math: Fp32Math) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, bias, c }) |p| a.add(p);
        inline for (.{ m, n, kd }) |v| a.add(i32of(v));
        a.add(@backingInt(math));
        try go(k.linear_fn, s, .{ .x = blocks(n, 16), .y = blocks(m, 16) }, .{ .x = 16, .y = 16 }, 0, &a);
    }

    pub fn quickGelu(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: u64) Error!void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(y);
        a.add(i64of(n));
        try go(k.quick_gelu_fn, s, .{ .x = blocks(n, 256) }, .{ .x = 256 }, 0, &a);
    }

    pub fn addF32(k: *const Kernels, s: cuda.Stream, x: u64, z: u64, y: u64, n: u64) Error!void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, z, y }) |p| a.add(p);
        a.add(i64of(n));
        try go(k.add_f32_fn, s, .{ .x = blocks(n, 256) }, .{ .x = 256 }, 0, &a);
    }

    /// fp32 attention, head dim 64: q, k, v, out [n, h * 64] (row stride ld); query i sees keys j <= i, j < nkeys.
    pub fn attentionF32(k: *const Kernels, s: cuda.Stream, q: u64, kk: u64, v: u64, out: u64, n: u32, h: u32, ld: u32, nkeys: u32, scale: f32) Error!void {
        if (n > 128 or nkeys == 0 or nkeys > n) return error.Invalid;
        var a: cuda.launch.Args = .{};
        inline for (.{ q, kk, v, out }) |p| a.add(p);
        a.add(i32of(ld));
        a.add(i32of(nkeys));
        a.add(scale);
        try go(k.attn_f32_fn, s, .{ .x = n, .y = h }, .{ .x = 64 }, 0, &a);
    }

    /// y = bf16(x), n values.
    pub fn castBf16(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: u64) Error!void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(y);
        a.add(i64of(n));
        try go(k.cast_fn, s, .{ .x = blocks(n, 256) }, .{ .x = 256 }, 0, &a);
    }
};

// ---- weights ------------------------------------------------------------------------------------------------------

/// Tensors of `text_encoder/model.safetensors.index.json` the text stack does not load: the vision tower (390
/// tensors, `visual.*`) and the LM head (`lm_head.weight`, logits are not used). 729 in the index, 338 loaded.
pub const skipped_prefixes = [_][]const u8{ "visual.", "lm_head." };

/// Every tensor the encoder binds, with its stored dtype and shape (all BF16, bound as stored): `visitor.tensor(name,
/// dtype, shape)`.
pub fn forEachTensor(visitor: anytype) !void {
    var nb: [96]u8 = undefined;
    try visitor.tensor("model.embed_tokens.weight", .bf16, &.{ vocab, dim });
    for (0..layers) |i| {
        inline for (layer_tensors) |t| {
            const name = try std.fmt.bufPrint(&nb, "model.layers.{d}." ++ t[0], .{i});
            try visitor.tensor(name, .bf16, t[1]);
        }
    }
    try visitor.tensor("model.norm.weight", .bf16, &.{dim});
}

const layer_tensors = .{
    .{ "input_layernorm.weight", &[_]u64{dim} },
    .{ "self_attn.q_proj.weight", &[_]u64{ dim, dim } },
    .{ "self_attn.q_proj.bias", &[_]u64{dim} },
    .{ "self_attn.k_proj.weight", &[_]u64{ kv, dim } },
    .{ "self_attn.k_proj.bias", &[_]u64{kv} },
    .{ "self_attn.v_proj.weight", &[_]u64{ kv, dim } },
    .{ "self_attn.v_proj.bias", &[_]u64{kv} },
    .{ "self_attn.o_proj.weight", &[_]u64{ dim, dim } },
    .{ "post_attention_layernorm.weight", &[_]u64{dim} },
    .{ "mlp.gate_proj.weight", &[_]u64{ ffn, dim } },
    .{ "mlp.up_proj.weight", &[_]u64{ ffn, dim } },
    .{ "mlp.down_proj.weight", &[_]u64{ dim, ffn } },
};

pub const Layer = struct {
    input_layernorm: u64,
    q_w: u64,
    q_b: u64,
    k_w: u64,
    k_b: u64,
    v_w: u64,
    v_b: u64,
    o_w: u64,
    post_attention_layernorm: u64,
    gate: u64,
    up: u64,
    down: u64,
};

pub const Weights = struct {
    embed: u64,
    layers: [layers]Layer,
    norm: u64,
};

fn need(lookup: anytype, name: []const u8) Error!u64 {
    return lookup.get(name) orelse error.MissingWeight;
}

/// Device addresses of every tensor by checkpoint name: `lookup.get(name) ?u64` (the BF16 tensor as stored, shapes as
/// `forEachTensor` lists them; the loader checks shapes).
pub fn bindWeights(lookup: anytype) Error!Weights {
    var w: Weights = undefined;
    w.embed = try need(lookup, "model.embed_tokens.weight");
    w.norm = try need(lookup, "model.norm.weight");
    var nb: [96]u8 = undefined;
    for (&w.layers, 0..) |*l, i| {
        const fields = [_][]const u8{ "input_layernorm", "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w", "post_attention_layernorm", "gate", "up", "down" };
        inline for (layer_tensors, 0..) |t, j| {
            const name = std.fmt.bufPrint(&nb, "model.layers.{d}." ++ t[0], .{i}) catch unreachable;
            @field(l, fields[j]) = try need(lookup, name);
        }
    }
    return w;
}

/// transformers' default inverse frequencies `1 / theta^(arange(0, 128, 2) / 128)` (modeling_qwen2_5_vl.py:529-545),
/// computed here as fp32(theta^(2j/128)) with a correctly rounded power, then the fp32 reciprocal. NOT a qualified
/// value: native computes it in torch (CPU float pow at init, or on the device if the buffer is re-initialized there),
/// and its bytes (`model.language_model.rotary_emb.inv_freq`) must be captured and passed to `Encoder.inv_freq`; this
/// candidate exists for host tests and is compared against that capture.
pub fn invFreqCandidate() [64]f32 {
    var out: [64]f32 = undefined;
    for (&out, 0..) |*o, j| {
        const e: f32 = @as(f32, @floatFromInt(2 * j)) / 128.0;
        const p: f32 = @floatCast(std.math.pow(f64, theta, @as(f64, e)));
        o.* = 1.0 / p;
    }
    return out;
}

// ---- scratch ------------------------------------------------------------------------------------------------------

/// Byte offsets of the scratch regions (256-aligned) from the caller's base; `bytes` the total.
pub const Plan = struct {
    ids: u64,
    inv: u64,
    cos: u64,
    sin: u64,
    h: u64,
    x: u64, // norm output; also o_proj / down_proj output (x is dead by then)
    q: u64,
    k: u64,
    v: u64,
    at: u64,
    g: u64, // gate, then silu(gate) * up in place
    u: u64,
    bytes: u64,

    pub fn init() Plan {
        var at: u64 = 0;
        const r: u64 = tokens;
        var p: Plan = undefined;
        inline for (.{
            .{ "ids", r * 4 },     .{ "inv", 64 * 4 },     .{ "cos", r * 64 * 4 }, .{ "sin", r * 64 * 4 },
            .{ "h", r * dim * 2 }, .{ "x", r * dim * 2 },  .{ "q", r * dim * 2 },  .{ "k", r * kv * 2 },
            .{ "v", r * kv * 2 },  .{ "at", r * dim * 2 }, .{ "g", r * ffn * 2 },  .{ "u", r * ffn * 2 },
        }) |e| {
            @field(p, e[0]) = at;
            at += std.mem.alignForward(u64, e[1], 256);
        }
        p.bytes = at;
        return p;
    }
};

/// Scratch the caller provides to `Encoder` (device memory, 256-byte aligned base). Pure.
pub fn scratchBytes() u64 {
    return Plan.init().bytes;
}

// ---- encoder ------------------------------------------------------------------------------------------------------

pub const Encoder = struct {
    k: *const Kernels,
    s: cuda.Stream,
    w: *const Weights,
    /// Device scratch of `scratchBytes()` bytes, 256-byte aligned.
    base: u64,
    /// `rotary_emb.inv_freq` as native holds it (see `invFreqCandidate`).
    inv_freq: [64]f32,
    /// The form of the RoPE frequency matmul `inv_freq @ position_ids` (modeling_qwen2_5_vl.py:557), an fp32 matmul
    /// under native's "high" policy. Required, no default.
    rope_freq: Fp32Math,
    hook: ?Hook = null,

    fn after(e: *const Encoder, name: []const u8, ptr: u64, bytes: u64) Error!void {
        if (e.hook) |h| h.after(h.ctx, name, ptr, bytes) catch return error.CudaFailed;
    }

    /// ids / valid from `prompt.qwenIds` -> out [1024, 3584] bf16 contiguous (rows 129.. of the final norm), in
    /// stream order. `valid` > 129 (the template guarantees it).
    pub fn forward(e: *const Encoder, ids: *const [tokens]u32, valid: u32, out: u64) Error!void {
        if (valid <= crop or valid > tokens) return error.InvalidPrompt;
        for (ids) |id| if (id >= vocab) return error.InvalidPrompt;
        const k = e.k;
        const s = e.s;
        const off = Plan.init();
        const b = e.base;
        const P = struct { ids: u64, inv: u64, cos: u64, sin: u64, h: u64, x: u64, q: u64, k: u64, v: u64, at: u64, g: u64, u: u64 };
        var p: P = undefined;
        inline for (comptime std.meta.fieldNames(P)) |f| @field(p, f) = b + @field(off, f);
        const L: u32 = tokens;
        const n: u64 = @as(u64, L) * dim;
        const nkv: u64 = @as(u64, L) * kv;
        const nff: u64 = @as(u64, L) * ffn;

        try k.upload(s, p.ids, std.mem.sliceAsBytes(ids)); // i32 on the device: every id < 2^31
        try k.upload(s, p.inv, std.mem.sliceAsBytes(&e.inv_freq));
        try k.ropeTable(s, p.inv, p.cos, p.sin, L, e.rope_freq);
        try e.after("rotary_emb.cos", p.cos, @as(u64, L) * 64 * 4);
        try e.after("rotary_emb.sin", p.sin, @as(u64, L) * 64 * 4);
        try k.embed(s, e.w.embed, p.ids, p.h, L, dim);
        try e.after("embed_tokens", p.h, n * 2);

        var nb: [96]u8 = undefined;
        for (e.w.layers, 0..) |l, i| {
            const nm = struct {
                fn f(buf: []u8, layer: usize, op: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ layer, op }) catch unreachable;
                }
            }.f;
            try k.rmsNorm(s, p.h, l.input_layernorm, p.x, L, dim, dim, dim);
            try e.after(nm(&nb, i, "input_layernorm"), p.x, n * 2);
            try k.gemm(s, p.x, l.q_w, l.q_b, p.q, L, dim, dim);
            try e.after(nm(&nb, i, "self_attn.q_proj"), p.q, n * 2);
            try k.gemm(s, p.x, l.k_w, l.k_b, p.k, L, kv, dim);
            try e.after(nm(&nb, i, "self_attn.k_proj"), p.k, nkv * 2);
            try k.gemm(s, p.x, l.v_w, l.v_b, p.v, L, kv, dim);
            try e.after(nm(&nb, i, "self_attn.v_proj"), p.v, nkv * 2);
            try k.ropeHalf(s, p.q, p.cos, p.sin, L, heads);
            try e.after(nm(&nb, i, "self_attn.rope_q"), p.q, n * 2);
            try k.ropeHalf(s, p.k, p.cos, p.sin, L, kv_heads);
            try e.after(nm(&nb, i, "self_attn.rope_k"), p.k, nkv * 2);
            try k.attention(s, p.q, p.k, p.v, p.at, L, heads, kv_heads, valid);
            try e.after(nm(&nb, i, "self_attn.sdpa"), p.at, n * 2);
            try k.gemm(s, p.at, l.o_w, 0, p.x, L, dim, dim);
            try e.after(nm(&nb, i, "self_attn.o_proj"), p.x, n * 2);
            try k.addBf16(s, p.h, p.x, p.h, n);
            try e.after(nm(&nb, i, "attn_residual"), p.h, n * 2);
            try k.rmsNorm(s, p.h, l.post_attention_layernorm, p.x, L, dim, dim, dim);
            try e.after(nm(&nb, i, "post_attention_layernorm"), p.x, n * 2);
            try k.gemm(s, p.x, l.gate, 0, p.g, L, ffn, dim);
            try e.after(nm(&nb, i, "mlp.gate_proj"), p.g, nff * 2);
            try k.gemm(s, p.x, l.up, 0, p.u, L, ffn, dim);
            try e.after(nm(&nb, i, "mlp.up_proj"), p.u, nff * 2);
            try k.siluMul(s, p.g, p.u, p.g, nff);
            try e.after(nm(&nb, i, "mlp.act_mul"), p.g, nff * 2);
            try k.gemm(s, p.g, l.down, 0, p.x, L, dim, ffn);
            try e.after(nm(&nb, i, "mlp.down_proj"), p.x, n * 2);
            try k.addBf16(s, p.h, p.x, p.h, n);
            try e.after(nm(&nb, i, "mlp_residual"), p.h, n * 2);
        }
        // The final norm on the kept rows only (rows are independent), written to the caller's buffer.
        try k.rmsNorm(s, p.h + @as(u64, crop) * dim * 2, e.w.norm, out, rows, dim, dim, dim);
        try e.after("norm[129:]", out, @as(u64, rows) * dim * 2);
    }
};

const testing = std.testing;

test "manifest: 338 tensors, 7,070,619,136 elements, all BF16 (the index's language tensors)" {
    const C = struct {
        n: usize = 0,
        elements: u64 = 0,
        fn tensor(c: *@This(), name: []const u8, dt: Dtype, shape: []const u64) !void {
            for (skipped_prefixes) |pre| try testing.expect(!std.mem.startsWith(u8, name, pre));
            try testing.expectEqual(Dtype.bf16, dt);
            var e: u64 = 1;
            for (shape) |x| e *= x;
            c.n += 1;
            c.elements += e;
        }
    };
    var c: C = .{};
    try forEachTensor(&c);
    try testing.expectEqual(@as(usize, 338), c.n);
    try testing.expectEqual(@as(u64, 7_070_619_136), c.elements);
}

test "bindWeights asks for exactly the manifest's names" {
    const Names = struct {
        set: std.StringHashMapUnmanaged(void) = .empty,
        asked: usize = 0,
        a: std.mem.Allocator,
        fn tensor(c: *@This(), name: []const u8, _: Dtype, _: []const u64) !void {
            try c.set.put(c.a, try c.a.dupe(u8, name), {});
        }
        fn get(c: *@This(), name: []const u8) ?u64 {
            if (!c.set.contains(name)) return null;
            c.asked += 1;
            return 0x1000 + c.asked * 256;
        }
    };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var c: Names = .{ .a = arena.allocator() };
    try forEachTensor(&c);
    const w = try bindWeights(&c);
    try testing.expectEqual(@as(usize, 338), c.asked);
    try testing.expect(w.layers[27].down != 0);
    const Empty = struct {
        fn get(_: @This(), _: []const u8) ?u64 {
            return null;
        }
    };
    try testing.expectError(error.MissingWeight, bindWeights(Empty{}));
}

test "scratch plan: 256-aligned regions, 123.4 MB" {
    const p = Plan.init();
    inline for (.{ "ids", "inv", "cos", "sin", "h", "x", "q", "k", "v", "at", "g", "u" }) |f| try testing.expectEqual(@as(u64, 0), @field(p, f) % 256);
    try testing.expectEqual(@as(u64, 123_385_344), scratchBytes());
    try testing.expectEqual(@as(u64, 1024), @as(u64, rows));
}

test "invFreqCandidate: first frequency 1, decreasing, last ~ theta^(-126/128)" {
    const f = invFreqCandidate();
    try testing.expectEqual(@as(f32, 1), f[0]);
    for (1..64) |j| try testing.expect(f[j] < f[j - 1]);
    try testing.expectApproxEqRel(@as(f32, @floatCast(std.math.pow(f64, theta, -126.0 / 128.0))), f[63], 1e-6);
    try testing.expectEqual(@as(u32, 0x3db504f3), @as(u32, @bitCast(attn_scale)));
}

test "every launcher and the forward pass are analyzed" {
    inline for (comptime std.meta.declarations(Kernels)) |d| _ = &@field(Kernels, d);
    _ = &Encoder.forward;
}
