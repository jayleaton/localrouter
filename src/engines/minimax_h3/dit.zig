//! MiniMax H3's DiT forward, op for op as the twin (`stk_twin/h3/dit.py`) runs it, under the twin's op names and roles,
//! so `localrouter check h3-replay` matches a capture op by op. `prepare` once a run (condition_proj and the token refiner),
//! `step` once a denoising step (the audio carry, the fp32 patch projections, 50 blocks, the fp32 heads, the un-carry).
//! Batch 1; NVFP4 linears from the pack (static input scales), comfy-kitchen's INT8 attention and fused RMSNorm + RoPE,
//! LocalRouter's fp32 / bf16 GEMMs and elementwise kernels.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const launch = @import("launch.zig");
const lay = @import("layout.zig");

const Pack = qi.pack.Pack;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;
const Linear = qi.weights.Linear;
const plan = qi.nvfp4_plan;
pub const Io = qi.dit.Io;
pub const Probe = qi.dit.Probe;

pub const layers = 50;
pub const dim = 5376;
pub const heads = 56;
pub const head_dim = 128;
pub const inner = heads * head_dim; // 7168
pub const ffn = 14336;
pub const text_dim = 5120;
pub const t_dim = 8; // the curve basis
pub const grid = 1025; // the curve table's rows
pub const ada_n = 3 * 6 * dim; // a block's adaln output (3 modalities x 6 chunks)
const eps: f32 = 1e-5;
const tile: u32 = 12;
const linear_names = [4][]const u8{ "qkv", "out", "fc1", "fc2" };

pub const Kernels = struct {
    h3: *const launch.Ops,
    kitchen: *const launch.Kitchen,
    te: *const qi.ops_launch.TeOps, // the bf16 GEMM (token refiner)
    ops: *const qi.ops_launch.Ops, // bf16 attention and row copies (token refiner)
    nv: *qi.nvfp4_exec.Nvfp4,
    major: u32,
    minor: u32,
};

const Block = struct {
    lin: [4]Linear,
    norm1: u64, // f32 [D]
    norm2: u64,
    q_norm: u64, // bf16 [128]
    k_norm: u64,
    ada_w: u64, // f32 [ada_n, 8]
    ada_b: u64, // f32 [ada_n]
};

const Refiner = struct { norm1: u64, qkv: u64, q_norm: u64, k_norm: u64, out: u64, norm2: u64, fc1: u64, fc2: u64 };

pub const Weights = struct {
    store: Store,
    blocks: [layers]Block = undefined,
    refiner: [2]Refiner = undefined,
    cond_w: u64 = 0,
    cond_b: u64 = 0,
    final_norm: u64 = 0, // the refiner's
    vp_w: u64 = 0,
    vp_b: u64 = 0,
    ap_w: u64 = 0,
    ap_b: u64 = 0,
    fl_ada_w: u64 = 0,
    fl_ada_b: u64 = 0,
    fl_norm: u64 = 0,
    vo_w: u64 = 0,
    vo_b: u64 = 0,
    ao_w: u64 = 0,
    ao_b: u64 = 0,
    table: []f32 = &.{}, // host: the curve [grid, 8]
    inv_freq: [16]f32 = undefined, // host: RoPE frequencies

    pub fn deinit(w: *Weights) void {
        w.store.gpa.free(w.table);
        w.store.deinit();
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, p: *const Pack, nv: *qi.nvfp4_exec.Nvfp4, up: *Uploader) !Weights {
        var w: Weights = .{ .store = .init(d, gpa) };
        errdefer w.deinit();
        const st = &w.store;
        inline for (.{
            .{ "cond_w", "condition_proj.weight" },                 .{ "cond_b", "condition_proj.bias" },
            .{ "final_norm", "token_refiner.final_norm.weight" },   .{ "vp_w", "video_patch_proj.weight" },
            .{ "vp_b", "video_patch_proj.bias" },                   .{ "ap_w", "audio_patch_proj.weight" },
            .{ "ap_b", "audio_patch_proj.bias" },                   .{ "fl_ada_w", "final_layer.adaln_proj.linear.weight" },
            .{ "fl_ada_b", "final_layer.adaln_proj.linear.bias" },  .{ "fl_norm", "final_layer.norm.weight" },
            .{ "vo_w", "final_layer.video_out.weight" },            .{ "vo_b", "final_layer.video_out.bias" },
            .{ "ao_w", "final_layer.audio_out.weight" },            .{ "ao_b", "final_layer.audio_out.bias" },
        }) |e| @field(w, e[0]) = try st.tensor(up, p, e[1]);
        var nb: [96]u8 = undefined;
        for (&w.refiner, 0..) |*r, i| {
            inline for (.{
                .{ "norm1", "norm1.weight" },           .{ "qkv", "attn.qkv_proj.weight" }, .{ "q_norm", "attn.q_norm.weight" },
                .{ "k_norm", "attn.k_norm.weight" },    .{ "out", "attn.out_proj.weight" }, .{ "norm2", "norm2.weight" },
                .{ "fc1", "mlp.fc1.weight" },           .{ "fc2", "mlp.fc2.weight" },
            }) |e| @field(r, e[0]) = try st.tensor(up, p, try std.fmt.bufPrint(&nb, "token_refiner.blocks.{d}.{s}", .{ i, e[1] }));
        }
        for (&w.blocks, 0..) |*b, i| {
            inline for (.{ "norm1", "norm2", "q_norm", "k_norm", "ada_w", "ada_b" }) |nm|
                @field(b, nm) = try st.tensor(up, p, try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ i, nm }));
            for (linear_names, 0..) |nm, j| b.lin[j] = try st.linear(io, p, nv, up, try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ i, nm }));
        }
        { // host copies: the curve table and the RoPE frequencies
            const t = try p.get("adaln_t_table");
            if (t.dim(0) != grid or t.dim(1) != t_dim) return error.BadCurveTable;
            w.table = try gpa.alloc(f32, grid * t_dim);
            try p.read(io, t, std.mem.sliceAsBytes(w.table));
            try p.read(io, try p.get("rope.inv_freq"), std.mem.sliceAsBytes(&w.inv_freq));
        }
        try st.done(up); // every copy and pack4 has landed
        return w;
    }
};

/// Device scratch for sequences up to `max_s` tokens (text + audio + video) and text up to `max_l` tokens.
pub const Dit = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: Kernels,
    w: *const Weights,
    s: cuda.Stream,
    probe: ?Probe = null,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    max_s: u32,
    max_l: u32,
    // the run's refined text [L, D] bf16
    context: u64 = 0,
    ctx_rows: u32 = 0,
    // per step
    h: u64 = 0, hn: u64 = 0, qkv: u64 = 0, heads_out: u64 = 0, att: u64 = 0, o: u64 = 0, gu: u64 = 0, act: u64 = 0,
    q: u64 = 0, kk: u64 = 0, v: u64 = 0, qc: u64 = 0, qs: u64 = 0, ws: u64 = 0,
    mod: u64 = 0, t_emb: u64 = 0, idx: u64 = 0, rope: u64 = 0, a_in: u64 = 0,
    rows: u64 = 0, arows: u64 = 0, emb: u64 = 0, aemb: u64 = 0, fada: u64 = 0, vout: u64 = 0, aout: u64 = 0,
    layout_key: ?[5]u32 = null,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: Kernels, w: *const Weights, s: cuda.Stream, max_s: u32, max_l: u32) !Dit {
        var x: Dit = .{ .gpa = gpa, .d = d, .k = k, .w = w, .s = s, .max_s = max_s, .max_l = max_l };
        errdefer x.deinit();
        const r: u64 = max_s;
        const bf = 2;
        const qr = plan.rows(@intCast(r), ffn, plan.TB); // the widest quantized rows (K up to 14336)
        inline for (.{
            .{ "context", @as(u64, max_l) * dim * bf },       .{ "h", r * dim * bf },              .{ "hn", r * dim * bf },
            .{ "qkv", r * 3 * inner * bf },                   .{ "heads_out", r * inner * bf },    .{ "att", r * inner * bf },
            .{ "o", r * dim * bf },                           .{ "gu", r * 2 * ffn * bf },         .{ "act", r * ffn * bf },
            .{ "q", @as(u64, max_l) * inner * bf },           .{ "kk", @as(u64, max_l) * inner * bf }, .{ "v", @as(u64, max_l) * inner * bf },
            .{ "qc", qr.fp8_bytes },                          .{ "qs", qr.scales_bytes },          .{ "ws", launch.Kitchen.Plan.of(heads, max_s).total },
            .{ "mod", 2 * ada_n * 4 },                        .{ "t_emb", 2 * t_dim * 4 },         .{ "idx", r * 4 },
            .{ "rope", r * 48 * 4 * bf },                     .{ "a_in", r * bf },                 .{ "rows", r * 96 * 4 },
            .{ "arows", r * 32 * 4 },                         .{ "emb", r * dim * 4 },             .{ "aemb", r * dim * 4 },
            .{ "fada", 2 * 2 * dim * 4 },                     .{ "vout", r * 96 * 4 },             .{ "aout", r * 32 * 4 },
        }) |e| @field(x, e[0]) = try x.alloc(e[1]);
        return x;
    }

    pub fn deinit(x: *Dit) void {
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(x.gpa);
    }

    fn alloc(x: *Dit, n: u64) !u64 {
        var b = try cuda.DeviceBuffer.alloc(x.d, @intCast(@max(n, 256)));
        errdefer b.free();
        try x.bufs.append(x.gpa, b);
        return b.at(0);
    }

    pub fn upload(x: *Dit, ptr: u64, bytes: []const u8) !void {
        try x.s.synchronize();
        try x.d.check(x.d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    fn pre(x: *Dit, name: []const u8, ins: []const Io) !void {
        if (x.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(x: *Dit, name: []const u8, outs: []const Io) !void {
        if (x.probe) |p| try p.after(p.ctx, name, outs);
    }

    // ------------------------------------------------------------------ recorded ops
    fn gemmBf16(x: *Dit, name: []const u8, in: u64, wt: u64, bias: u64, out: u64, m: u64, n: u64, kdim: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = m * kdim * 2 }});
        try x.k.te.gemm(x.s, in, wt, bias, out, m, n, kdim);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * n * 2 }});
    }

    fn gemmF32(x: *Dit, name: []const u8, in: u64, wt: u64, bias: u64, out: u64, m: u64, n: u64, kdim: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = m * kdim * 4 }});
        try x.k.h3.gemmF32(x.s, in, wt, bias, out, m, n, kdim);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * n * 4 }});
    }

    fn rms(x: *Dit, name: []const u8, in: u64, w: u64, out: u64, rows: u64, d: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = rows * d * 2 }});
        try x.k.h3.rmsNorm(x.s, in, w, out, rows, d, eps);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = rows * d * 2 }});
    }

    fn add(x: *Dit, name: []const u8, a: u64, b: u64, out: u64, n: u64) !void {
        try x.pre(name, &.{ .{ .role = "x", .ptr = a, .bytes = n * 2 }, .{ .role = "z", .ptr = b, .bytes = n * 2 } });
        try x.k.h3.add(x.s, a, b, out, n);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = n * 2 }});
    }

    fn linear(x: *Dit, name: []const u8, l: Linear, in: u64, out: u64, rows: u32) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = @as(u64, rows) * l.k * 2 }});
        try l.forward(x.k.nv, x.k.major, x.k.minor, tile, in, out, rows, x.qc, x.qs, x.s);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = @as(u64, rows) * l.n * 2 }});
    }

    // ------------------------------------------------------------------ the run's text
    /// Qwen3-VL layer-50 states `text` [L, 5120] bf16 -> the refined context [L, 5376] (condition_proj, 2 refiner blocks,
    /// final norm), kept for every step of the run.
    pub fn prepare(x: *Dit, text: u64, l: u32) !void {
        if (l > x.max_l) return error.PromptTooLong;
        const L: u64 = l;
        const w = x.w;
        try x.gemmBf16("refiner.condition_proj", text, w.cond_w, w.cond_b, x.h, L, dim, text_dim);
        var nb: [48]u8 = undefined;
        for (w.refiner, 0..) |r, i| {
            const nm = struct {
                fn f(buf: []u8, j: usize, op: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "refiner.{d}.{s}", .{ j, op }) catch unreachable;
                }
            }.f;
            try x.rms(nm(&nb, i, "norm1"), x.h, r.norm1, x.hn, L, dim);
            try x.gemmBf16(nm(&nb, i, "qkv"), x.hn, r.qkv, 0, x.qkv, L, 3 * inner, dim);
            inline for (.{ "q", "kk", "v" }, 0..) |dst, part| // q, k, v as contiguous [L, H, 128] (the twin's .contiguous())
                try x.k.ops.copyRows(x.s, @field(x, dst), x.qkv + part * inner * 2, L, inner * 2, 3 * inner * 2, inner * 2);
            try x.rms(nm(&nb, i, "q_norm"), x.q, r.q_norm, x.q, L * heads, head_dim);
            try x.rms(nm(&nb, i, "k_norm"), x.kk, r.k_norm, x.kk, L * heads, head_dim);
            const an = nm(&nb, i, "attention");
            const sz = L * inner * 2;
            try x.pre(an, &.{ .{ .role = "q", .ptr = x.q, .bytes = sz }, .{ .role = "k", .ptr = x.kk, .bytes = sz }, .{ .role = "v", .ptr = x.v, .bytes = sz } });
            try x.k.ops.attention(x.s, x.q, x.kk, x.v, x.att, l, heads, heads, l, false, 0);
            try x.post(an, &.{.{ .role = "y", .ptr = x.att, .bytes = sz }});
            try x.gemmBf16(nm(&nb, i, "out"), x.att, r.out, 0, x.o, L, dim, inner);
            try x.add(nm(&nb, i, "attn_residual"), x.o, x.h, x.h, L * dim);
            try x.rms(nm(&nb, i, "norm2"), x.h, r.norm2, x.hn, L, dim);
            try x.gemmBf16(nm(&nb, i, "fc1"), x.hn, r.fc1, 0, x.gu, L, 2 * ffn, dim);
            const sn = nm(&nb, i, "swiglu");
            try x.pre(sn, &.{.{ .role = "x", .ptr = x.gu, .bytes = L * 2 * ffn * 2 }});
            try x.k.h3.swiglu(x.s, x.gu, x.act, L, ffn, true);
            try x.post(sn, &.{.{ .role = "y", .ptr = x.act, .bytes = L * ffn * 2 }});
            try x.gemmBf16(nm(&nb, i, "fc2"), x.act, r.fc2, 0, x.o, L, dim, ffn);
            try x.add(nm(&nb, i, "mlp_residual"), x.o, x.h, x.h, L * dim);
        }
        try x.rms("refiner.final_norm", x.h, w.final_norm, x.context, L, dim);
        x.ctx_rows = l;
    }

    // ------------------------------------------------------------------ one step
    /// video bf16 [24, T, H, W] and the sampler's carried audio bf16 [32, 2, A] at `sigma` -> the velocities ComfyUI
    /// returns (negated; the audio back on the carry), into `vel_v` / `vel_a`.
    pub fn step(x: *Dit, video: u64, audio: u64, sigma: f32, t: u32, h: u32, w: u32, a: u32, vel_v: u64, vel_a: u64) !void {
        const L = x.ctx_rows;
        const ly = lay.Layout.init(L, t, h, w, a);
        if (ly.s > x.max_s) return error.VideoTooLong;
        const S: u64 = ly.s;
        const key: [5]u32 = .{ L, t, h, w, a };
        if (x.layout_key == null or !std.mem.eql(u32, &x.layout_key.?, &key)) {
            const pos = try x.gpa.alloc([3]f64, ly.s);
            defer x.gpa.free(pos);
            ly.positions(pos);
            const table = try x.gpa.alloc(u16, @as(usize, ly.s) * 48 * 4);
            defer x.gpa.free(table);
            ly.ropeTable(pos, &x.w.inv_freq, table);
            try x.upload(x.rope, std.mem.sliceAsBytes(table));
            x.layout_key = key;
        }
        const sc = lay.stepScalars(sigma);
        var unique: [2]f32 = undefined;
        const idx = try x.gpa.alloc(i32, ly.s);
        defer x.gpa.free(idx);
        const mr = ly.modRows(sc, &unique, idx);
        var temb: [2 * t_dim]f32 = undefined;
        lay.curveTEmb(x.w.table, grid, t_dim, unique[0..mr.m], temb[0 .. mr.m * t_dim]);
        try x.upload(x.idx, std.mem.sliceAsBytes(idx));
        try x.upload(x.t_emb, std.mem.sliceAsBytes(temb[0 .. mr.m * t_dim]));
        const m: u64 = mr.m;
        const na: u64 = 32 * 2 * @as(u64, a);
        const nv: u64 = @as(u64, t) * (h / 2) * (w / 2);
        const nvx: u64 = 24 * @as(u64, t) * h * w;
        const ar = ly.audioRows();
        const vr = ly.videoRows();

        try x.pre("carry", &.{.{ .role = "x", .ptr = audio, .bytes = na * 2 }});
        try x.k.h3.scale(x.s, audio, sc.carry, x.a_in, na);
        try x.post("carry", &.{.{ .role = "y", .ptr = x.a_in, .bytes = na * 2 }});
        try x.pre("video_rows", &.{.{ .role = "x", .ptr = video, .bytes = nvx * 2 }});
        try x.k.h3.patchify(x.s, video, x.rows, 24, t, h, w);
        try x.post("video_rows", &.{.{ .role = "y", .ptr = x.rows, .bytes = nv * 96 * 4 }});
        try x.pre("audio_rows", &.{.{ .role = "x", .ptr = x.a_in, .bytes = na * 2 }});
        try x.k.h3.packAudio(x.s, x.a_in, x.arows, 32, a);
        try x.post("audio_rows", &.{.{ .role = "y", .ptr = x.arows, .bytes = na * 4 }});
        try x.gemmF32("video_patch_proj", x.rows, x.w.vp_w, x.w.vp_b, x.emb, nv, dim, 96);
        try x.gemmF32("audio_patch_proj", x.arows, x.w.ap_w, x.w.ap_b, x.aemb, 2 * @as(u64, a), dim, 32);
        // [text | audio | video] rows: the context, then the embeddings rounded to bf16
        try x.k.ops.copyRows(x.s, x.h, x.context, L, dim * 2, dim * 2, dim * 2);
        try x.k.h3.toBf16(x.s, x.aemb, x.h + @as(u64, ar[0]) * dim * 2, 2 * @as(u64, a) * dim);
        try x.k.h3.toBf16(x.s, x.emb, x.h + @as(u64, vr[0]) * dim * 2, nv * dim);

        for (&x.w.blocks, 0..) |*b, i| try x.block(i, b, S, m);

        // the final layer: fp32 curve modulation of each target segment, fp32 heads, negated, back to the latents
        try x.gemmF32("final.adaln", x.t_emb, x.w.fl_ada_w, x.w.fl_ada_b, x.fada, m, 2 * dim, t_dim);
        inline for (.{ .{ "final.mod_video", vr, mr.row_v, x.emb }, .{ "final.mod_audio", ar, mr.row_a, x.aemb } }) |e| {
            const rows: u64 = e[1][1] - e[1][0];
            const shift = x.fada + @as(u64, e[2]) * 2 * dim * 4;
            try x.pre(e[0], &.{.{ .role = "x", .ptr = x.h + @as(u64, e[1][0]) * dim * 2, .bytes = rows * dim * 2 }});
            try x.k.h3.finalMod(x.s, x.h + @as(u64, e[1][0]) * dim * 2, x.w.fl_norm, shift + dim * 4, shift, e[3], rows, dim, eps);
            try x.post(e[0], &.{.{ .role = "y", .ptr = e[3], .bytes = rows * dim * 4 }});
        }
        try x.gemmF32("final.video_out", x.emb, x.w.vo_w, x.w.vo_b, x.vout, nv, 96, dim);
        try x.gemmF32("final.audio_out", x.aemb, x.w.ao_w, x.w.ao_b, x.aout, 2 * @as(u64, a), 32, dim);
        try x.pre("video_out", &.{.{ .role = "x", .ptr = x.vout, .bytes = nv * 96 * 4 }});
        try x.k.h3.unpatchifyNeg(x.s, x.vout, vel_v, 24, t, h, w);
        try x.post("video_out", &.{.{ .role = "y", .ptr = vel_v, .bytes = nvx * 2 }});
        try x.pre("audio_out", &.{.{ .role = "x", .ptr = x.aout, .bytes = na * 4 }});
        try x.k.h3.unpackAudioNeg(x.s, x.aout, vel_a, 32, a);
        try x.post("audio_out", &.{.{ .role = "y", .ptr = vel_a, .bytes = na * 2 }});
        try x.pre("uncarry", &.{ .{ .role = "a", .ptr = x.a_in, .bytes = na * 2 }, .{ .role = "v", .ptr = vel_a, .bytes = na * 2 } });
        try x.k.h3.uncarry(x.s, x.a_in, vel_a, sc.c1, sc.c2, na);
        try x.post("uncarry", &.{.{ .role = "y", .ptr = vel_a, .bytes = na * 2 }});
    }

    fn block(x: *Dit, i: usize, b: *const Block, S: u64, m: u64) !void {
        var nb: [24]u8 = undefined;
        const nm = struct {
            fn f(buf: []u8, j: usize, op: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "L{d}.{s}", .{ j, op }) catch unreachable;
            }
        }.f;
        const rows: u32 = @intCast(S);
        const xs = S * dim * 2;
        const mods = m * 3 * 6 * dim * 4;
        try x.gemmF32(nm(&nb, i, "adaln"), x.t_emb, b.ada_w, b.ada_b, x.mod, m, ada_n, t_dim);
        inline for (.{ .{ "norm_mod1", 0, b.norm1 }, .{ "norm_mod2", 1, b.norm2 } }, 0..) |e, half| {
            const n = nm(&nb, i, e[0]);
            try x.pre(n, &.{ .{ .role = "x", .ptr = x.h, .bytes = xs }, .{ .role = "mod", .ptr = x.mod, .bytes = mods }, .{ .role = "idx", .ptr = x.idx, .bytes = S * 4 } });
            try x.k.h3.normMod(x.s, x.h, e[2], x.mod, x.idx, x.hn, S, dim, e[1], eps);
            try x.post(n, &.{.{ .role = "y", .ptr = x.hn, .bytes = xs }});
            if (half == 0) { // attention half
                try x.linear(nm(&nb, i, "qkv"), b.lin[0], x.hn, x.qkv, rows);
                const qkv_bytes = S * 3 * inner * 2;
                const rn = nm(&nb, i, "rms_rope");
                try x.pre(rn, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                try x.k.kitchen.rmsRope(x.s, x.qkv, x.rope, b.q_norm, b.k_norm, heads, rows, 96, eps);
                try x.post(rn, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                const an = nm(&nb, i, "attention");
                try x.pre(an, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                try x.k.kitchen.attention(x.s, x.qkv, x.heads_out, x.ws, heads, rows);
                try x.k.h3.headsToRows(x.s, x.heads_out, x.att, heads, S, head_dim);
                try x.post(an, &.{.{ .role = "y", .ptr = x.att, .bytes = S * inner * 2 }});
                try x.linear(nm(&nb, i, "out"), b.lin[1], x.att, x.o, rows);
            } else { // MLP half
                try x.linear(nm(&nb, i, "fc1"), b.lin[2], x.hn, x.gu, rows);
                const sn = nm(&nb, i, "swiglu");
                try x.pre(sn, &.{.{ .role = "x", .ptr = x.gu, .bytes = S * 2 * ffn * 2 }});
                try x.k.h3.swiglu(x.s, x.gu, x.act, S, ffn, false);
                try x.post(sn, &.{.{ .role = "y", .ptr = x.act, .bytes = S * ffn * 2 }});
                try x.linear(nm(&nb, i, "fc2"), b.lin[3], x.act, x.o, rows);
            }
            const gn = nm(&nb, i, if (half == 0) "gate_add1" else "gate_add2");
            try x.pre(gn, &.{ .{ .role = "x", .ptr = x.h, .bytes = xs }, .{ .role = "y", .ptr = x.o, .bytes = xs } });
            try x.k.h3.gateAdd(x.s, x.h, x.o, x.mod, x.idx, S, dim, e[1]);
            try x.post(gn, &.{.{ .role = "x", .ptr = x.h, .bytes = xs }});
        }
    }
};
