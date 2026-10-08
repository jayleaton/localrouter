//! The text encoder (Qwen3-VL 8B's text model, text-only), op for op as the twin (`stk_twin/te.py`) runs it, with the
//! twin's op names: the token ids of the pipeline's template, 36 decoder layers on LocalRouter's kernels (te.cu,
//! gemm.cu, attention.cu), and the last layer's output without the final norm, the template's system rows dropped.
//! Weights come from the text encoder's pack (`pack.py write_te`, all bf16) with its tokenizer.json.

const std = @import("std");
const cuda = @import("cuda");
const tokenizer = @import("tokenizer");
const Pack = @import("pack.zig").Pack;
const upload_mod = @import("upload.zig");
const ops_launch = @import("ops_launch.zig");
const smath = @import("smath.zig");
const dit = @import("dit.zig");
const Io = dit.Io;
const Probe = dit.Probe;

pub const layers = 36;
pub const dim = 4096;
pub const heads = 32;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const mlp = 12288;
pub const eps: f32 = 1e-6;
const kv = kv_heads * head_dim;

/// transformers' default RoPE frequencies for theta 5e6, frozen (kernels/qwen_image/te_inv_freq.json).
pub fn invFreq(gpa: std.mem.Allocator, json: []const u8) ![64]f32 {
    const J = struct { inv_freq: []const f64 };
    const v = try std.json.parseFromSlice(J, gpa, json, .{ .ignore_unknown_fields = true });
    defer v.deinit();
    if (v.value.inv_freq.len != 64) return error.BadTable;
    var out: [64]f32 = undefined;
    for (&out, v.value.inv_freq) |*o, f| o.* = @floatCast(f);
    return out;
}

/// cos / sin [n, 64] f32 for positions 0..n-1 as the twin's `rope_tables`: freq = f32(inv * p), the shared sincos in
/// f64, rounded to f32.
pub fn ropeTables(inv: *const [64]f32, cos_out: []f32, sin_out: []f32) void {
    for (0..cos_out.len / 64) |p| {
        const fp: f32 = @floatFromInt(p);
        for (inv, 0..) |f, j| {
            const r = smath.sincos(@as(f64, f * fp));
            cos_out[p * 64 + j] = @floatCast(r.cos);
            sin_out[p * 64 + j] = @floatCast(r.sin);
        }
    }
}

const Layer = struct {
    input_layernorm: u64,
    q_proj: u64,
    k_proj: u64,
    v_proj: u64,
    q_norm: u64,
    k_norm: u64,
    o_proj: u64,
    post_attention_layernorm: u64,
    gate_proj: u64,
    up_proj: u64,
    down_proj: u64,
};

/// The pack's template and system-row count, and the tokenizer that turns a prompt into ids.
pub const Prompt = struct {
    arena: std.heap.ArenaAllocator,
    tok: tokenizer.Tokenizer,
    template: []const u8, // "...{}..." (one placeholder)
    drop: u32,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Prompt {
        var p: Prompt = .{ .arena = .init(gpa), .tok = undefined, .template = "", .drop = 0 };
        errdefer p.arena.deinit();
        const a = p.arena.allocator();
        const man = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(16 << 20));
        const M = struct { template: []const u8, drop: u32 };
        const m = try std.json.parseFromSliceLeaky(M, a, man, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        if (std.mem.count(u8, m.template, "{}") != 1) return error.BadTemplate;
        p.template = m.template;
        p.drop = m.drop;
        p.tok = try tokenizer.loadTokenizer(io, gpa, try std.fs.path.join(a, &.{ dir, "tokenizer.json" }));
        return p;
    }

    pub fn deinit(p: *Prompt) void {
        p.tok.deinit();
        p.arena.deinit();
    }

    /// The template's ids for `prompt` (an empty prompt reads " ", as the pipeline); caller frees.
    pub fn ids(p: *const Prompt, a: std.mem.Allocator, prompt: []const u8) ![]u32 {
        const at = std.mem.indexOf(u8, p.template, "{}").?;
        const text = try std.mem.concat(a, u8, &.{ p.template[0..at], if (prompt.len == 0) " " else prompt, p.template[at + 2 ..] });
        defer a.free(text);
        return p.tok.encode(a, text);
    }
};

/// The device weights and scratch for prompts of at most `max_rows` tokens.
pub const TextEncoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const ops_launch.TeOps,
    ops: *const ops_launch.Ops,
    s: cuda.Stream,
    probe: ?Probe = null,
    inv: [64]f32,
    mem: upload_mod.Slab,
    max_rows: u32,
    embed_w: u64 = 0,
    w: [layers]Layer = undefined,
    // scratch
    ids: u64 = 0, h: u64 = 0, x: u64 = 0, q: u64 = 0, kk: u64 = 0, v: u64 = 0, at: u64 = 0, o: u64 = 0,
    g: u64 = 0, u: u64 = 0, gu: u64 = 0, cos: u64 = 0, sin: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const ops_launch.TeOps, ops: *const ops_launch.Ops, s: cuda.Stream, p: *const Pack, up: *upload_mod.Uploader, inv: [64]f32, max_rows: u32) !TextEncoder {
        var t: TextEncoder = .{ .gpa = gpa, .d = d, .k = k, .ops = ops, .s = s, .inv = inv, .max_rows = max_rows, .mem = .init(d, gpa) };
        errdefer t.deinit();
        t.embed_w = try t.tensor(up, p, "te.embed", &.{ 0, dim });
        var nb: [64]u8 = undefined;
        for (&t.w, 0..) |*l, i| {
            inline for (.{
                .{ "input_layernorm", dim, 0 },          .{ "q_proj", heads * head_dim, dim }, .{ "k_proj", kv, dim },
                .{ "v_proj", kv, dim },                  .{ "q_norm", head_dim, 0 },           .{ "k_norm", head_dim, 0 },
                .{ "o_proj", dim, heads * head_dim },    .{ "post_attention_layernorm", dim, 0 }, .{ "gate_proj", mlp, dim },
                .{ "up_proj", mlp, dim },                .{ "down_proj", dim, mlp },
            }) |e| {
                const name = try std.fmt.bufPrint(&nb, "te.{d}.{s}", .{ i, e[0] });
                @field(l, e[0]) = try t.tensor(up, p, name, &.{ e[1], e[2] });
            }
        }
        const r: u64 = max_rows;
        t.ids = try t.alloc(r * 4);
        inline for (.{ "h", "x", "q", "at", "o" }) |nm| @field(t, nm) = try t.alloc(r * dim * 2);
        t.kk = try t.alloc(r * kv * 2);
        t.v = try t.alloc(r * kv * 2);
        inline for (.{ "g", "u", "gu" }) |nm| @field(t, nm) = try t.alloc(r * mlp * 2);
        t.cos = try t.alloc(r * 64 * 4);
        t.sin = try t.alloc(r * 64 * 4);
        return t;
    }

    pub fn deinit(t: *TextEncoder) void {
        t.mem.deinit();
    }

    fn alloc(t: *TextEncoder, n: u64) !u64 {
        return t.mem.alloc(@intCast(n));
    }

    fn upload(t: *TextEncoder, ptr: u64, bytes: []const u8) !void {
        try t.d.check(t.d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    /// A bf16 weight of shape `shape` ([n] when shape[1] is 0, else [n, k]; shape[0] 0: any rows) onto the device.
    fn tensor(t: *TextEncoder, up: *upload_mod.Uploader, p: *const Pack, name: []const u8, shape: []const usize) !u64 {
        const w = try p.get(name);
        if (w.dtype != .bf16) return error.BadWeight;
        if (shape[1] == 0) {
            if (w.rank != 1 or w.shape[0] != shape[0]) return error.BadWeight;
        } else if (w.rank != 2 or w.shape[1] != shape[1] or (shape[0] != 0 and w.shape[0] != shape[0])) return error.BadWeight;
        const ptr = try t.alloc(w.len());
        try up.tensor(p, w, ptr);
        return ptr;
    }

    fn pre(t: *TextEncoder, name: []const u8, ins: []const Io) !void {
        if (t.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(t: *TextEncoder, name: []const u8, outs: []const Io) !void {
        if (t.probe) |p| try p.after(p.ctx, name, outs);
    }

    fn gemm(t: *TextEncoder, name: []const u8, in: u64, w: u64, out: u64, m: u64, n: u64, k: u64) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = m * k * 2 }});
        try t.k.gemm(t.s, in, w, 0, out, m, n, k);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * n * 2 }});
    }

    fn norm(t: *TextEncoder, name: []const u8, in: u64, w: u64, out: u64, rows: u64, d: u64) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = rows * d * 2 }});
        try t.k.rmsNorm(t.s, in, w, out, rows, d, eps);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = rows * d * 2 }});
    }

    fn residual(t: *TextEncoder, name: []const u8, z: u64, n: u64) !void {
        try t.pre(name, &.{ .{ .role = "x", .ptr = t.h, .bytes = n * 2 }, .{ .role = "z", .ptr = z, .bytes = n * 2 } });
        try t.k.add(t.s, t.h, z, t.h, n);
        try t.post(name, &.{.{ .role = "y", .ptr = t.h, .bytes = n * 2 }});
    }

    /// ids -> the hidden states [len, 4096] at `h` (the returned pointer; rows `drop..` are the context).
    pub fn forward(t: *TextEncoder, ids: []const u32) !u64 {
        const L: u32 = @intCast(ids.len);
        if (L == 0 or L > t.max_rows) return error.PromptTooLong;
        const n: u64 = @as(u64, L) * dim;
        {
            const tab = try t.gpa.alloc(f32, 2 * @as(usize, L) * 64);
            defer t.gpa.free(tab);
            ropeTables(&t.inv, tab[0 .. L * 64], tab[L * 64 ..]);
            try t.upload(t.cos, std.mem.sliceAsBytes(tab[0 .. L * 64]));
            try t.upload(t.sin, std.mem.sliceAsBytes(tab[L * 64 ..]));
            try t.upload(t.ids, std.mem.sliceAsBytes(ids));
        }
        try t.pre("te.embed", &.{.{ .role = "ids", .ptr = t.ids, .bytes = @as(u64, L) * 4 }});
        try t.k.embed(t.s, t.embed_w, t.ids, t.h, L, dim);
        try t.post("te.embed", &.{.{ .role = "y", .ptr = t.h, .bytes = n * 2 }});
        var nb: [64]u8 = undefined;
        for (&t.w, 0..) |*l, i| {
            const nm = struct {
                fn f(buf: []u8, layer: usize, op: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "te.{d}.{s}", .{ layer, op }) catch unreachable;
                }
            }.f;
            try t.norm(nm(&nb, i, "input_layernorm"), t.h, l.input_layernorm, t.x, L, dim);
            try t.gemm(nm(&nb, i, "q_proj"), t.x, l.q_proj, t.q, L, heads * head_dim, dim);
            try t.gemm(nm(&nb, i, "k_proj"), t.x, l.k_proj, t.kk, L, kv, dim);
            try t.gemm(nm(&nb, i, "v_proj"), t.x, l.v_proj, t.v, L, kv, dim);
            try t.norm(nm(&nb, i, "q_norm"), t.q, l.q_norm, t.q, @as(u64, L) * heads, head_dim);
            try t.norm(nm(&nb, i, "k_norm"), t.kk, l.k_norm, t.kk, @as(u64, L) * kv_heads, head_dim);
            const tb: u64 = @as(u64, L) * 64 * 4;
            const rope_name = nm(&nb, i, "rope");
            try t.pre(rope_name, &.{ .{ .role = "q", .ptr = t.q, .bytes = n * 2 }, .{ .role = "k", .ptr = t.kk, .bytes = @as(u64, L) * kv * 2 }, .{ .role = "cos", .ptr = t.cos, .bytes = tb }, .{ .role = "sin", .ptr = t.sin, .bytes = tb } });
            try t.k.ropeHalf(t.s, t.q, t.cos, t.sin, L, heads);
            try t.k.ropeHalf(t.s, t.kk, t.cos, t.sin, L, kv_heads);
            try t.post(rope_name, &.{ .{ .role = "q", .ptr = t.q, .bytes = n * 2 }, .{ .role = "k", .ptr = t.kk, .bytes = @as(u64, L) * kv * 2 } });
            const attn_name = nm(&nb, i, "attention");
            try t.pre(attn_name, &.{ .{ .role = "q", .ptr = t.q, .bytes = n * 2 }, .{ .role = "k", .ptr = t.kk, .bytes = @as(u64, L) * kv * 2 }, .{ .role = "v", .ptr = t.v, .bytes = @as(u64, L) * kv * 2 } });
            try t.ops.attention(t.s, t.q, t.kk, t.v, t.at, L, heads, kv_heads, L, true, 0);
            try t.post(attn_name, &.{.{ .role = "y", .ptr = t.at, .bytes = n * 2 }});
            try t.gemm(nm(&nb, i, "o_proj"), t.at, l.o_proj, t.o, L, dim, heads * head_dim);
            try t.residual(nm(&nb, i, "attn_residual"), t.o, n);
            try t.norm(nm(&nb, i, "post_attention_layernorm"), t.h, l.post_attention_layernorm, t.x, L, dim);
            try t.gemm(nm(&nb, i, "gate_proj"), t.x, l.gate_proj, t.g, L, mlp, dim);
            try t.gemm(nm(&nb, i, "up_proj"), t.x, l.up_proj, t.u, L, mlp, dim);
            const sm_name = nm(&nb, i, "silu_mul");
            const nm_bytes = @as(u64, L) * mlp * 2;
            try t.pre(sm_name, &.{ .{ .role = "g", .ptr = t.g, .bytes = nm_bytes }, .{ .role = "u", .ptr = t.u, .bytes = nm_bytes } });
            try t.k.siluMul(t.s, t.g, t.u, t.gu, @as(u64, L) * mlp);
            try t.post(sm_name, &.{.{ .role = "y", .ptr = t.gu, .bytes = nm_bytes }});
            try t.gemm(nm(&nb, i, "down_proj"), t.gu, l.down_proj, t.o, L, dim, mlp);
            try t.residual(nm(&nb, i, "mlp_residual"), t.o, n);
        }
        return t.h;
    }
};

test "rope tables: position 0 is cos 1, sin 0; the first frequency is 1" {
    var inv: [64]f32 = @splat(0.5);
    inv[0] = 1;
    var c: [2 * 64]f32 = undefined;
    var s: [2 * 64]f32 = undefined;
    ropeTables(&inv, &c, &s);
    try std.testing.expectEqual(@as(f32, 1), c[0]);
    try std.testing.expectEqual(@as(f32, 0), s[63]);
    try std.testing.expectEqual(@as(f32, @floatCast(@sin(@as(f64, 1)))), s[64]);
}
