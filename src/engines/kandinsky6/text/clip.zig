//! Kandinsky-6's pooled text conditioning: CLIP ViT-L/14's text tower in fp32 (`text_encoder_2`, CLIPTextModel built
//! from its config and loaded as stored, text_embedder.py:79-103,130), op for op as transformers v5.18.0
//! models/clip/modeling_clip.py runs it for the 77 ids of `prompt.clipIds` with their attention mask
//! (text_embedder.py:205-213): token + position embeddings (:233-256), 12 pre-LN encoder layers (:353-395:
//! LayerNorm, q/k/v Linear, SDPA with the causal AND key-padding bool mask from create_causal_mask (:535-540;
//! is_causal only when no token is padded), out_proj, residual, LayerNorm, fc1, quick_gelu (activations.py:122-123),
//! fc2, residual), final_layer_norm, and the pooled row at argmax(input_ids) because the config's eos_token_id is 2
//! (:564-576). Output: pooler_output [1, 768] fp32; `castPooled` gives the bf16 the DiT takes (pipeline.py:467).
//!
//! The fp32 Linears run under native's `set_float32_matmul_precision("high")` (factory.py:162): their form is the
//! required `linear` parameter (no default). Attention here is strict FP32 (one form only; native's SDPA kernel for
//! fp32 is uncaptured). Nothing is claimed bit-equal to native until the SM121 fixtures in
//! kernels/cuda/kandinsky6/text/README.md pass. The encoder borrows driver, stream, weights and scratch.

const std = @import("std");
const cuda = @import("cuda");
const qwen25 = @import("qwen25.zig");

pub const Kernels = qwen25.Kernels;
pub const Fp32Math = qwen25.Fp32Math;
pub const Hook = qwen25.Hook;
pub const Dtype = qwen25.Dtype;
pub const Error = qwen25.Error;

pub const layers: u32 = 12;
pub const dim: u32 = 768;
pub const heads: u32 = 12;
pub const head_dim: u32 = 64;
pub const ffn: u32 = 3072;
pub const vocab: u32 = 49408;
pub const tokens: u32 = 77;
pub const eps: f32 = 1e-5;
/// 64^-0.5 = 0.125 exactly (CLIPAttention.scale, modeling_clip.py:291).
pub const attn_scale: f32 = 0.125;

/// Every tensor of `text_encoder_2/model.safetensors` (196, all F32, all loaded; the file has no position_ids):
/// `visitor.tensor(name, dtype, shape)`.
pub fn forEachTensor(visitor: anytype) !void {
    var nb: [96]u8 = undefined;
    try visitor.tensor("embeddings.token_embedding.weight", .f32, &.{ vocab, dim });
    try visitor.tensor("embeddings.position_embedding.weight", .f32, &.{ tokens, dim });
    for (0..layers) |i| {
        inline for (layer_tensors) |t| {
            const name = try std.fmt.bufPrint(&nb, "encoder.layers.{d}." ++ t[0], .{i});
            try visitor.tensor(name, .f32, t[1]);
        }
    }
    try visitor.tensor("final_layer_norm.weight", .f32, &.{dim});
    try visitor.tensor("final_layer_norm.bias", .f32, &.{dim});
}

const layer_tensors = .{
    .{ "layer_norm1.weight", &[_]u64{dim} },               .{ "layer_norm1.bias", &[_]u64{dim} },
    .{ "self_attn.q_proj.weight", &[_]u64{ dim, dim } },   .{ "self_attn.q_proj.bias", &[_]u64{dim} },
    .{ "self_attn.k_proj.weight", &[_]u64{ dim, dim } },   .{ "self_attn.k_proj.bias", &[_]u64{dim} },
    .{ "self_attn.v_proj.weight", &[_]u64{ dim, dim } },   .{ "self_attn.v_proj.bias", &[_]u64{dim} },
    .{ "self_attn.out_proj.weight", &[_]u64{ dim, dim } }, .{ "self_attn.out_proj.bias", &[_]u64{dim} },
    .{ "layer_norm2.weight", &[_]u64{dim} },               .{ "layer_norm2.bias", &[_]u64{dim} },
    .{ "mlp.fc1.weight", &[_]u64{ ffn, dim } },            .{ "mlp.fc1.bias", &[_]u64{ffn} },
    .{ "mlp.fc2.weight", &[_]u64{ dim, ffn } },            .{ "mlp.fc2.bias", &[_]u64{dim} },
};

const layer_fields = [_][]const u8{ "ln1_w", "ln1_b", "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b" };

pub const Layer = struct {
    ln1_w: u64,
    ln1_b: u64,
    q_w: u64,
    q_b: u64,
    k_w: u64,
    k_b: u64,
    v_w: u64,
    v_b: u64,
    o_w: u64,
    o_b: u64,
    ln2_w: u64,
    ln2_b: u64,
    fc1_w: u64,
    fc1_b: u64,
    fc2_w: u64,
    fc2_b: u64,
};

pub const Weights = struct {
    tok: u64,
    pos: u64,
    layers: [layers]Layer,
    final_w: u64,
    final_b: u64,
};

fn need(lookup: anytype, name: []const u8) Error!u64 {
    return lookup.get(name) orelse error.MissingWeight;
}

/// Device addresses by checkpoint name: `lookup.get(name) ?u64` (F32 as stored).
pub fn bindWeights(lookup: anytype) Error!Weights {
    var w: Weights = undefined;
    w.tok = try need(lookup, "embeddings.token_embedding.weight");
    w.pos = try need(lookup, "embeddings.position_embedding.weight");
    w.final_w = try need(lookup, "final_layer_norm.weight");
    w.final_b = try need(lookup, "final_layer_norm.bias");
    var nb: [96]u8 = undefined;
    for (&w.layers, 0..) |*l, i| {
        inline for (layer_tensors, 0..) |t, j| {
            const name = std.fmt.bufPrint(&nb, "encoder.layers.{d}." ++ t[0], .{i}) catch unreachable;
            @field(l, layer_fields[j]) = try need(lookup, name);
        }
    }
    return w;
}

/// Scratch regions (byte offsets, 256-aligned) and the total.
pub const Plan = struct {
    ids: u64,
    h: u64,
    x: u64,
    q: u64,
    k: u64,
    v: u64,
    at: u64,
    o: u64,
    f: u64,
    bytes: u64,

    pub fn init() Plan {
        const r: u64 = tokens;
        var at: u64 = 0;
        var p: Plan = undefined;
        inline for (.{
            .{ "ids", r * 4 },     .{ "h", r * dim * 4 },  .{ "x", r * dim * 4 }, .{ "q", r * dim * 4 }, .{ "k", r * dim * 4 },
            .{ "v", r * dim * 4 }, .{ "at", r * dim * 4 }, .{ "o", r * dim * 4 }, .{ "f", r * ffn * 4 },
        }) |e| {
            @field(p, e[0]) = at;
            at += std.mem.alignForward(u64, e[1], 256);
        }
        p.bytes = at;
        return p;
    }
};

/// Scratch the caller provides (device memory, 256-byte aligned base). Pure.
pub fn scratchBytes() u64 {
    return Plan.init().bytes;
}

pub const Encoder = struct {
    k: *const Kernels,
    s: cuda.Stream,
    w: *const Weights,
    /// Device scratch of `scratchBytes()` bytes, 256-byte aligned.
    base: u64,
    /// The form of every fp32 Linear (q/k/v/out_proj, fc1, fc2). Required, no default.
    linear: Fp32Math,
    hook: ?Hook = null,

    fn after(e: *const Encoder, name: []const u8, ptr: u64, bytes: u64) Error!void {
        if (e.hook) |h| h.after(h.ctx, name, ptr, bytes) catch return error.CudaFailed;
    }

    /// ids / valid / eot from `prompt.clipIds` -> pooled [768] fp32 at `out`, in stream order.
    pub fn forward(e: *const Encoder, ids: *const [tokens]u32, valid: u32, eot: u32, out: u64) Error!void {
        if (valid < 2 or valid > tokens or eot >= tokens) return error.InvalidPrompt;
        for (ids) |id| if (id >= vocab) return error.InvalidPrompt;
        const k = e.k;
        const s = e.s;
        const off = Plan.init();
        const P = struct { ids: u64, h: u64, x: u64, q: u64, k: u64, v: u64, at: u64, o: u64, f: u64 };
        var p: P = undefined;
        inline for (comptime std.meta.fieldNames(P)) |f| @field(p, f) = e.base + @field(off, f);
        const L: u32 = tokens;
        const n: u64 = @as(u64, L) * dim;

        try k.upload(s, p.ids, std.mem.sliceAsBytes(ids));
        try k.clipEmbed(s, e.w.tok, e.w.pos, p.ids, p.h, L, dim);
        try e.after("embeddings", p.h, n * 4);
        var nb: [96]u8 = undefined;
        for (e.w.layers, 0..) |l, i| {
            const nm = struct {
                fn f(buf: []u8, layer: usize, op: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "encoder.layers.{d}.{s}", .{ layer, op }) catch unreachable;
                }
            }.f;
            try k.layerNorm(s, p.h, l.ln1_w, l.ln1_b, p.x, L, dim, dim, dim, eps);
            try e.after(nm(&nb, i, "layer_norm1"), p.x, n * 4);
            try k.linearF32(s, p.x, l.q_w, l.q_b, p.q, L, dim, dim, e.linear);
            try e.after(nm(&nb, i, "self_attn.q_proj"), p.q, n * 4);
            try k.linearF32(s, p.x, l.k_w, l.k_b, p.k, L, dim, dim, e.linear);
            try e.after(nm(&nb, i, "self_attn.k_proj"), p.k, n * 4);
            try k.linearF32(s, p.x, l.v_w, l.v_b, p.v, L, dim, dim, e.linear);
            try e.after(nm(&nb, i, "self_attn.v_proj"), p.v, n * 4);
            try k.attentionF32(s, p.q, p.k, p.v, p.at, L, heads, dim, valid, attn_scale);
            try e.after(nm(&nb, i, "self_attn.sdpa"), p.at, n * 4);
            try k.linearF32(s, p.at, l.o_w, l.o_b, p.o, L, dim, dim, e.linear);
            try e.after(nm(&nb, i, "self_attn.out_proj"), p.o, n * 4);
            try k.addF32(s, p.h, p.o, p.h, n);
            try e.after(nm(&nb, i, "attn_residual"), p.h, n * 4);
            try k.layerNorm(s, p.h, l.ln2_w, l.ln2_b, p.x, L, dim, dim, dim, eps);
            try e.after(nm(&nb, i, "layer_norm2"), p.x, n * 4);
            try k.linearF32(s, p.x, l.fc1_w, l.fc1_b, p.f, L, ffn, dim, e.linear);
            try e.after(nm(&nb, i, "mlp.fc1"), p.f, @as(u64, L) * ffn * 4);
            try k.quickGelu(s, p.f, p.f, @as(u64, L) * ffn);
            try e.after(nm(&nb, i, "mlp.activation_fn"), p.f, @as(u64, L) * ffn * 4);
            try k.linearF32(s, p.f, l.fc2_w, l.fc2_b, p.o, L, dim, ffn, e.linear);
            try e.after(nm(&nb, i, "mlp.fc2"), p.o, n * 4);
            try k.addF32(s, p.h, p.o, p.h, n);
            try e.after(nm(&nb, i, "mlp_residual"), p.h, n * 4);
        }
        // final_layer_norm on the pooled row only (LayerNorm rows are independent), straight into `out`.
        try k.layerNorm(s, p.h + @as(u64, eot) * dim * 4, e.w.final_w, e.w.final_b, out, 1, dim, dim, dim, eps);
        try e.after("pooler_output", out, dim * 4);
    }

    /// out_bf16[768] = bf16(pooled f32), pipeline.py:467.
    pub fn castPooled(e: *const Encoder, pooled: u64, out_bf16: u64) Error!void {
        try e.k.castBf16(e.s, pooled, out_bf16, dim);
    }
};

const testing = std.testing;

test "manifest: 196 F32 tensors, 123,060,480 elements (the whole file)" {
    const C = struct {
        n: usize = 0,
        elements: u64 = 0,
        fn tensor(c: *@This(), _: []const u8, dt: Dtype, shape: []const u64) !void {
            try testing.expectEqual(Dtype.f32, dt);
            var x: u64 = 1;
            for (shape) |d| x *= d;
            c.n += 1;
            c.elements += x;
        }
    };
    var c: C = .{};
    try forEachTensor(&c);
    try testing.expectEqual(@as(usize, 196), c.n);
    try testing.expectEqual(@as(u64, 123_060_480), c.elements);
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
    _ = try bindWeights(&c);
    try testing.expectEqual(@as(usize, 196), c.asked);
}

test "scratch plan" {
    const p = Plan.init();
    inline for (.{ "ids", "h", "x", "q", "k", "v", "at", "o", "f" }) |f| try testing.expectEqual(@as(u64, 0), @field(p, f) % 256);
    try testing.expectEqual(@as(u64, 2_602_496), scratchBytes());
}

test {
    _ = qwen25;
}

test "the forward pass is analyzed" {
    _ = &Encoder.forward;
    _ = &Encoder.castPooled;
}
