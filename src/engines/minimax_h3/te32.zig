//! MiniMax H3's text encoder (Qwen3-VL 32B, 50 layers, fp32 activations) op for op as the twin
//! (`stk_twin/h3/te32.py`) runs it, under the twin's names (`te.embed`, `te.{i}.{...}`), on kernels/cuda/minimax/te32.cu:
//! the NVFP4 AWQ checkpoint as published (int8 embedding with per-row scales; NVFP4 linears dequantized on the fly in
//! the fp32 GEMM; bf16 norms and AWQ scales widened to fp32), RoPE tables from the portable fdlibm math.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const Pack = qi.pack.Pack;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;
const smath = qi.smath;
const kernels = @import("minimax_kernels");
const dit = @import("dit.zig");
const Io = dit.Io;
const Probe = dit.Probe;

pub const layers = 50;
pub const dim = 5120;
pub const heads = 64;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const mlp = 25600;
const eps: f32 = 1e-6;
const theta: f64 = 5_000_000.0;
const block: u32 = 256;

fn blocks(n: u64) u32 {
    return @intCast((n + block - 1) / block);
}

pub const Ops = struct {
    module: cuda.Module,
    embed_fn: cuda.Function,
    add_fn: cuda.Function,
    silu_mul_fn: cuda.Function,
    rms_fn: cuda.Function,
    rope_fn: cuda.Function,
    attn_fn: cuda.Function,
    linear_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{
            .module = m,
            .embed_fn = try m.function("te32_embed_i8"),
            .add_fn = try m.function("te32_add"),
            .silu_mul_fn = try m.function("te32_silu_mul"),
            .rms_fn = try m.function("te32_rms_norm"),
            .rope_fn = try m.function("te32_rope_split_half"),
            .attn_fn = try m.function("te32_attention"),
            .linear_fn = try m.function("te32_linear_nvfp4"),
        };
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, blk: u32, shared: u32, args: *cuda.launch.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = blk }, .shared = shared }, s, args);
    }
};

/// One NVFP4 linear as the checkpoint stores it: codes [N, K/2], the swizzled e4m3 block scales, the fp32 tensor
/// scale, and the AWQ pre_quant_scale widened to fp32 (0 when the layer has none).
const Lin = struct { codes: u64, bscale: u64, tscale: f32, pqs: u64, n: u32, k: u32 };

const Layer = struct {
    input_layernorm: u64,
    post_attention_layernorm: u64,
    q_norm: u64,
    k_norm: u64,
    lin: [7]Lin, // q, k, v, o, gate, up, down
};

const lin_names = [7][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };

pub const TextEncoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Ops,
    s: cuda.Stream,
    probe: ?Probe = null,
    store: Store,
    embed: u64 = 0, // int8 [vocab, 5120]
    embed_scale: u64 = 0, // fp32 [vocab]
    w: [layers]Layer = undefined,
    inv_freq: [64]f32 = undefined,
    max_l: u32,
    // scratch
    ids: u64 = 0, h: u64 = 0, x: u64 = 0, q: u64 = 0, kk: u64 = 0, v: u64 = 0, at: u64 = 0, o: u64 = 0,
    g: u64 = 0, u: u64 = 0, gu: u64 = 0, cos: u64 = 0, sin: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const Ops, s: cuda.Stream, p: *const Pack, up: *Uploader, max_l: u32) !TextEncoder {
        var t: TextEncoder = .{ .gpa = gpa, .d = d, .k = k, .s = s, .store = .init(d, gpa), .max_l = max_l };
        errdefer t.store.deinit();
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(gpa);
        const st = &t.store;
        t.embed = try st.tensor(up, p, "model.embed_tokens.weight");
        t.embed_scale = try st.tensor(up, p, "model.embed_tokens.weight_scale");
        var nb: [96]u8 = undefined;
        for (&t.w, 0..) |*l, i| {
            inline for (.{ "input_layernorm", "post_attention_layernorm" }) |nm|
                @field(l, nm) = try t.widen(io, p, up, &host, try std.fmt.bufPrint(&nb, "model.layers.{d}.{s}.weight", .{ i, nm }));
            l.q_norm = try t.widen(io, p, up, &host, try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.q_norm.weight", .{i}));
            l.k_norm = try t.widen(io, p, up, &host, try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.k_norm.weight", .{i}));
            for (lin_names, 0..) |nm, j| l.lin[j] = try t.linear(io, p, up, &host, try std.fmt.bufPrint(&nb, "model.layers.{d}.{s}", .{ i, nm }));
        }
        t.inv_freq = try invFreq(gpa);
        const r: u64 = max_l;
        inline for (.{
            .{ "ids", r * 4 },             .{ "h", r * dim * 4 },        .{ "x", r * dim * 4 },              .{ "q", r * heads * head_dim * 4 },
            .{ "kk", r * kv_heads * head_dim * 4 }, .{ "v", r * kv_heads * head_dim * 4 }, .{ "at", r * heads * head_dim * 4 },
            .{ "o", r * dim * 4 },         .{ "g", r * mlp * 4 },        .{ "u", r * mlp * 4 },              .{ "gu", r * mlp * 4 },
            .{ "cos", r * 64 * 4 },        .{ "sin", r * 64 * 4 },
        }) |e| @field(t, e[0]) = try st.alloc(@max(e[1], 256));
        try st.done(up); // the weights have landed (the scratch above is plain device memory)
        return t;
    }

    pub fn deinit(t: *TextEncoder) void {
        t.store.deinit();
    }

    /// A bf16 tensor widened to fp32 (exact) on the device.
    fn widen(t: *TextEncoder, io: std.Io, p: *const Pack, up: *Uploader, host: *std.ArrayList(u8), name: []const u8) !u64 {
        const w = try p.get(name);
        if (w.dtype != .bf16) return error.BadWeight;
        try host.resize(t.gpa, w.len());
        try p.read(io, w, host.items);
        const n = w.len() / 2;
        const f = try t.gpa.alloc(f32, n);
        defer t.gpa.free(f);
        const src: []align(1) const u16 = std.mem.bytesAsSlice(u16, host.items);
        for (f, src) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
        return t.store.upload(up, std.mem.sliceAsBytes(f));
    }

    fn linear(t: *TextEncoder, io: std.Io, p: *const Pack, up: *Uploader, host: *std.ArrayList(u8), key: []const u8) !Lin {
        var nb: [128]u8 = undefined;
        const codes = try p.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{key}));
        if (codes.dtype != .u8) return error.BadWeight;
        const n: u32 = @intCast(codes.dim(0));
        const k: u32 = @intCast(codes.dim(1) * 2);
        var l: Lin = .{ .codes = try t.store.tensor(up, p, try std.fmt.bufPrint(&nb, "{s}.weight", .{key})), .bscale = 0, .tscale = 0, .pqs = 0, .n = n, .k = k };
        l.bscale = try t.store.tensor(up, p, try std.fmt.bufPrint(&nb, "{s}.weight_scale", .{key}));
        l.tscale = @floatCast(try Store.scalar(io, p, try std.fmt.bufPrint(&nb, "{s}.weight_scale_2", .{key})));
        const pq = try std.fmt.bufPrint(&nb, "{s}.pre_quant_scale", .{key});
        if (p.tensors.contains(pq)) l.pqs = try t.widen(io, p, up, host, pq);
        return l;
    }

    /// The RoPE frequencies as ComfyUI's CUDA run computes them, frozen in kernels/minimax/te32_inv_freq.json (the
    /// twin reads the same file): 3 of the 64 are an ulp off the correctly rounded powf, so they are not recomputed.
    pub fn invFreq(gpa: std.mem.Allocator) ![64]f32 {
        return qi.te.invFreq(gpa, kernels.te32_inv_freq);
    }

    fn pre(t: *TextEncoder, name: []const u8, ins: []const Io) !void {
        if (t.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(t: *TextEncoder, name: []const u8, outs: []const Io) !void {
        if (t.probe) |p| try p.after(p.ctx, name, outs);
    }

    fn lin(t: *TextEncoder, name: []const u8, l: Lin, in: u64, out: u64, m: u64) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = m * l.k * 4 }});
        var a: cuda.launch.Args = .{};
        inline for (.{ in, l.pqs, l.codes, l.bscale }) |x| a.add(x);
        a.add(l.tscale);
        a.add(@as(u64, 0)); // no bias
        a.add(out);
        inline for (.{ m, l.n, l.k }) |x| a.add(@as(i64, @intCast(x)));
        try Ops.go(t.k.linear_fn, t.s, .{ .x = @intCast((m + 127) / 128), .y = @intCast((l.n + 63) / 64) }, block, 0, &a);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * l.n * 4 }});
    }

    fn norm(t: *TextEncoder, name: []const u8, in: u64, w: u64, out: u64, rows: u64, d: u64) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = rows * d * 4 }});
        var a: cuda.launch.Args = .{};
        inline for (.{ in, w, out }) |x| a.add(x);
        a.add(@as(i64, @intCast(d)));
        a.add(eps);
        try Ops.go(t.k.rms_fn, t.s, .{ .x = @intCast(rows) }, block, 0, &a);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = rows * d * 4 }});
    }

    fn add(t: *TextEncoder, name: []const u8, a_: u64, b: u64, out: u64, n: u64) !void {
        try t.pre(name, &.{ .{ .role = "a", .ptr = a_, .bytes = n * 4 }, .{ .role = "b", .ptr = b, .bytes = n * 4 } });
        var a: cuda.launch.Args = .{};
        inline for (.{ a_, b, out }) |x| a.add(x);
        a.add(@as(i64, @intCast(n)));
        try Ops.go(t.k.add_fn, t.s, .{ .x = blocks(n) }, block, 0, &a);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = n * 4 }});
    }

    fn rope(t: *TextEncoder, x: u64, rows: u32, nh: u32, mode: i32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, t.cos, t.sin }) |p| a.add(p);
        a.add(@as(i64, nh));
        a.add(mode);
        try Ops.go(t.k.rope_fn, t.s, .{ .x = rows, .y = nh }, 64, 0, &a);
    }

    /// ids -> the residual stream after layer index 49, [L, 5120] fp32 at the returned pointer. `rope_mode`: which
    /// product of the rotation is fused (the twin's choice, from its pod test against ComfyUI).
    pub fn forward(t: *TextEncoder, ids: []const u32, rope_mode: i32) !u64 {
        const L: u32 = @intCast(ids.len);
        if (L == 0 or L > t.max_l) return error.PromptTooLong;
        const Lu: u64 = L;
        { // the RoPE tables for positions 0..L-1: f32(pos) * inv, sincos in f64, rounded to f32
            const tab = try t.gpa.alloc(f32, 2 * @as(usize, L) * 64);
            defer t.gpa.free(tab);
            for (0..L) |pos| for (t.inv_freq, 0..) |inv, j| {
                const r = smath.sincos(@as(f64, @as(f32, @floatFromInt(pos)) * inv));
                tab[pos * 64 + j] = @floatCast(r.cos);
                tab[@as(usize, L) * 64 + pos * 64 + j] = @floatCast(r.sin);
            };
            try t.s.synchronize();
            try t.d.check(t.d.api.cuMemcpyHtoD_v2(t.cos, tab.ptr, @as(usize, L) * 64 * 4), "cuMemcpyHtoD");
            try t.d.check(t.d.api.cuMemcpyHtoD_v2(t.sin, tab[@as(usize, L) * 64 ..].ptr, @as(usize, L) * 64 * 4), "cuMemcpyHtoD");
            try t.d.check(t.d.api.cuMemcpyHtoD_v2(t.ids, ids.ptr, ids.len * 4), "cuMemcpyHtoD");
        }
        try t.pre("te.embed", &.{.{ .role = "ids", .ptr = t.ids, .bytes = Lu * 4 }});
        {
            var a: cuda.launch.Args = .{};
            inline for (.{ t.embed, t.embed_scale, t.ids, t.h }) |x| a.add(x);
            a.add(@as(i64, dim));
            a.add(@as(i32, 1)); // round through bf16 (ComfyUI's dequantize_embedding to the compute dtype)
            try Ops.go(t.k.embed_fn, t.s, .{ .x = blocks(dim), .y = L }, block, 0, &a);
        }
        try t.post("te.embed", &.{.{ .role = "y", .ptr = t.h, .bytes = Lu * dim * 4 }});
        var nb: [48]u8 = undefined;
        const nm = struct {
            fn f(buf: []u8, j: usize, op: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "te.{d}.{s}", .{ j, op }) catch unreachable;
            }
        }.f;
        const qn = Lu * heads * head_dim;
        const kn = Lu * kv_heads * head_dim;
        for (&t.w, 0..) |*l, i| {
            try t.norm(nm(&nb, i, "input_layernorm"), t.h, l.input_layernorm, t.x, Lu, dim);
            try t.lin(nm(&nb, i, "q_proj"), l.lin[0], t.x, t.q, Lu);
            try t.lin(nm(&nb, i, "k_proj"), l.lin[1], t.x, t.kk, Lu);
            try t.lin(nm(&nb, i, "v_proj"), l.lin[2], t.x, t.v, Lu);
            try t.norm(nm(&nb, i, "q_norm"), t.q, l.q_norm, t.q, Lu * heads, head_dim);
            try t.norm(nm(&nb, i, "k_norm"), t.kk, l.k_norm, t.kk, Lu * kv_heads, head_dim);
            const rn = nm(&nb, i, "rope");
            try t.pre(rn, &.{ .{ .role = "q", .ptr = t.q, .bytes = qn * 4 }, .{ .role = "k", .ptr = t.kk, .bytes = kn * 4 }, .{ .role = "cos", .ptr = t.cos, .bytes = Lu * 64 * 4 }, .{ .role = "sin", .ptr = t.sin, .bytes = Lu * 64 * 4 } });
            try t.rope(t.q, L, heads, rope_mode);
            try t.rope(t.kk, L, kv_heads, rope_mode);
            try t.post(rn, &.{ .{ .role = "q", .ptr = t.q, .bytes = qn * 4 }, .{ .role = "k", .ptr = t.kk, .bytes = kn * 4 } });
            const an = nm(&nb, i, "attention");
            try t.pre(an, &.{ .{ .role = "q", .ptr = t.q, .bytes = qn * 4 }, .{ .role = "k", .ptr = t.kk, .bytes = kn * 4 }, .{ .role = "v", .ptr = t.v, .bytes = kn * 4 } });
            {
                var a: cuda.launch.Args = .{};
                inline for (.{ t.q, t.kk, t.v, t.at }) |x| a.add(x);
                inline for (.{ L, heads, kv_heads }) |x| a.add(@as(i32, @intCast(x)));
                a.add(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, head_dim)))));
                const smem: u32 = (L + 256) * 4;
                if (smem > 48 * 1024) try t.k.attn_fn.allowDynamicShared(smem);
                try Ops.go(t.k.attn_fn, t.s, .{ .x = L, .y = heads }, 128, smem, &a);
            }
            try t.post(an, &.{.{ .role = "y", .ptr = t.at, .bytes = qn * 4 }});
            try t.lin(nm(&nb, i, "o_proj"), l.lin[3], t.at, t.o, Lu);
            try t.add(nm(&nb, i, "attn_residual"), t.h, t.o, t.h, Lu * dim);
            try t.norm(nm(&nb, i, "post_attention_layernorm"), t.h, l.post_attention_layernorm, t.x, Lu, dim);
            try t.lin(nm(&nb, i, "gate_proj"), l.lin[4], t.x, t.g, Lu);
            try t.lin(nm(&nb, i, "up_proj"), l.lin[5], t.x, t.u, Lu);
            const sn = nm(&nb, i, "silu_mul");
            try t.pre(sn, &.{ .{ .role = "g", .ptr = t.g, .bytes = Lu * mlp * 4 }, .{ .role = "u", .ptr = t.u, .bytes = Lu * mlp * 4 } });
            {
                var a: cuda.launch.Args = .{};
                inline for (.{ t.g, t.u, t.gu }) |x| a.add(x);
                a.add(@as(i64, @intCast(Lu * mlp)));
                try Ops.go(t.k.silu_mul_fn, t.s, .{ .x = blocks(Lu * mlp) }, block, 0, &a);
            }
            try t.post(sn, &.{.{ .role = "y", .ptr = t.gu, .bytes = Lu * mlp * 4 }});
            try t.lin(nm(&nb, i, "down_proj"), l.lin[6], t.gu, t.o, Lu);
            try t.add(nm(&nb, i, "mlp_residual"), t.h, t.o, t.h, Lu * dim);
        }
        return t.h;
    }
};

test "inv_freq: the frozen table, first frequency 1, the rest decreasing, within an ulp of the portable powf" {
    const f = try TextEncoder.invFreq(std.testing.allocator);
    try std.testing.expectEqual(@as(f32, 1.0), f[0]);
    for (1..64) |j| try std.testing.expect(f[j] < f[j - 1]);
    const ln = smath.log(theta);
    for (f, 0..) |g, j| { // the frozen values are ComfyUI's GPU powf: at most an ulp from the correctly rounded one
        const p: f32 = @floatCast(smath.exp(@as(f64, @floatFromInt(2 * j)) / 128.0 * ln));
        const d = @as(i64, @as(u32, @bitCast(g))) - @as(i64, @as(u32, @bitCast(1.0 / p)));
        try std.testing.expect(d >= -1 and d <= 1);
    }
}
