//! MiniMax H3's DiT forward, op for op as the twin (`stk_twin/h3/dit.py`) runs it, under the twin's op names and roles,
//! so `localrouter check h3-replay` matches a capture op by op. `prepare` once a run (condition_proj and the token refiner),
//! `step` once a denoising step (the audio carry, the fp32 patch projections, 50 blocks, the fp32 heads, the un-carry).
//! Batch 1; NVFP4 linears from the pack (static input scales), comfy-kitchen's INT8 attention and fused RMSNorm + RoPE,
//! LocalRouter's fp32 / bf16 GEMMs and elementwise kernels.
//!
//! Two schedules, one set of bits. The reference schedule is the twin's op sequence (it runs whenever a probe is set, and
//! with STK_H3_REF=1). The fast one (default) fuses each gate_add with the norm_mod after it into one kernel
//! (`h3_gate_add_norm_mod`: per element the same arithmetic, ops.cu says why), has the INT8 attention store its output as
//! rows (no [H, S, D] -> rows pass), and replays the step as a CUDA graph per shape. Neither changes any rounding; the
//! tests (`test_fuse.py`, `h3-replay`'s fast passes, `h3-step-bench`'s equality) compare the two byte for byte.

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
const profile_mod = @import("profile.zig");
pub const Io = qi.dit.Io;
pub const Probe = qi.dit.Probe;
pub const Profile = profile_mod.Profile;
const Class = profile_mod.Class;

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

/// A step's shape and buffers: what its launches' arguments depend on (the step graph's key, less the fuse flag).
const Body = struct { video: u64, vel_v: u64, vel_a: u64, t: u32, h: u32, w: u32, a: u32, ly: lay.Layout, m: u32, row_v: u32, row_a: u32 };
const StepGraph = struct { key: [12]u64, exec: ?cuda.graph.Exec }; // exec null: seen once (run eagerly), captured on the next use
const max_graphs = 4;

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
    /// The fast schedule (`fuse`: fused gate_add + norm_mod, attention stored as rows; `graphs`: the step's launches replayed
    /// as a CUDA graph per shape and buffer set). Both default on, both off with STK_H3_REF=1 (the reference path), and a
    /// probe or a profile always runs the eager schedule. `graph_error` is why graphs turned themselves off, if they did.
    fuse: bool = true,
    graphs: bool = true,
    graph_error: ?anyerror = null,
    prof: ?*Profile = null,
    step_graphs: std.ArrayList(StepGraph) = .empty,
    mod2: u64 = 0, // the second modulation buffer: block i + 1's, while block i's is still read by the fused kernel
    idx_host: []i32 = &.{}, // the modulation row of every token, kept per layout (uploaded when it changes)
    idx_key: ?[7]u32 = null,
    extra_launches: u64 = 0, // the NVFP4 and row-copy launches (the H3 kernels count in `launch.count`)

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: Kernels, w: *const Weights, s: cuda.Stream, max_s: u32, max_l: u32) !Dit {
        var x: Dit = .{ .gpa = gpa, .d = d, .k = k, .w = w, .s = s, .max_s = max_s, .max_l = max_l };
        errdefer x.deinit();
        if (std.c.getenv("STK_H3_REF")) |v| {
            if (v[0] == '1') {
                x.fuse = false;
                x.graphs = false;
            }
        }
        x.idx_host = try gpa.alloc(i32, max_s);
        const r: u64 = max_s;
        const bf = 2;
        const qr = plan.rows(@intCast(r), ffn, plan.TB); // the widest quantized rows (K up to 14336)
        inline for (.{
            .{ "context", @as(u64, max_l) * dim * bf },       .{ "h", r * dim * bf },              .{ "hn", r * dim * bf },
            .{ "qkv", r * 3 * inner * bf },                   .{ "heads_out", r * inner * bf },    .{ "att", r * inner * bf },
            .{ "o", r * dim * bf },                           .{ "gu", r * 2 * ffn * bf },         .{ "act", r * ffn * bf },
            .{ "q", @as(u64, max_l) * inner * bf },           .{ "kk", @as(u64, max_l) * inner * bf }, .{ "v", @as(u64, max_l) * inner * bf },
            .{ "qc", qr.fp8_bytes },                          .{ "qs", qr.scales_bytes },          .{ "ws", launch.Kitchen.Plan.of(heads, max_s).total },
            .{ "mod", 2 * ada_n * 4 },                        .{ "mod2", 2 * ada_n * 4 },       .{ "t_emb", 2 * t_dim * 4 },         .{ "idx", r * 4 },
            .{ "rope", r * 48 * 4 * bf },                     .{ "a_in", r * bf },                 .{ "rows", r * 96 * 4 },
            .{ "arows", r * 32 * 4 },                         .{ "emb", r * dim * 4 },             .{ "aemb", r * dim * 4 },
            .{ "fada", 2 * 2 * dim * 4 },                     .{ "vout", r * 96 * 4 },             .{ "aout", r * 32 * 4 },
        }) |e| @field(x, e[0]) = try x.alloc(e[1]);
        return x;
    }

    pub fn deinit(x: *Dit) void {
        for (x.step_graphs.items) |*g| if (g.exec) |*e| e.deinit();
        x.step_graphs.deinit(x.gpa);
        x.gpa.free(x.idx_host);
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(x.gpa);
    }

    /// Kernel launches so far (the H3 and comfy-kitchen kernels, the NVFP4 quantizers and GEMMs, the row copies): a
    /// step's difference is the eager schedule's launch count (a graph replay is one launch).
    pub fn launches(x: *const Dit) u64 {
        return @atomicLoad(u64, &launch.count, .monotonic) + x.extra_launches;
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

    /// Host bytes to the device on the stream, no host sync: the driver stages pageable memory before it returns.
    fn uploadAsync(x: *Dit, ptr: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(ptr, bytes.ptr, bytes.len, x.s.handle), "cuMemcpyHtoDAsync");
    }

    fn markBegin(x: *Dit) !void {
        if (x.prof) |p| try p.begin(x.s);
    }
    fn markEnd(x: *Dit, c: Class) !void {
        if (x.prof) |p| try p.end(x.s, c);
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

    fn linear(x: *Dit, name: []const u8, l: Linear, in: u64, out: u64, rows: u32, gc: Class) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = @as(u64, rows) * l.k * 2 }});
        try x.linearRun(l, in, out, rows, gc);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = @as(u64, rows) * l.n * 2 }});
    }

    /// One block linear: the quantizer and the prompt GEMM (`Linear.forward`). Profiling runs forward's two launches itself
    /// (the same plan calls and arguments) so each gets its own span: `quant`, then `gc`.
    fn linearRun(x: *Dit, l: Linear, in: u64, out: u64, rows: u32, gc: Class) !void {
        x.extra_launches += 2;
        if (x.prof == null or l.kind != .nvfp4) return l.forward(x.k.nv, x.k.major, x.k.minor, tile, in, out, rows, x.qc, x.qs, x.s);
        const tbk = plan.tb(tile, x.k.major, x.k.minor);
        const rr = plan.rows(rows, l.k, tbk);
        try x.markBegin();
        const qp = try plan.quant4(rows, l.k, l.k, tbk, l.act);
        try x.k.nv.run(&qp, .{ .x = in, .codes = x.qc, .scales = x.qs }, x.s);
        try x.markEnd(.quant);
        try x.markBegin();
        const gp = try plan.gemmPrompt(.a4, tile, x.k.major, x.k.minor, rows, l.n, l.k, l.npad, rr.mpad, false, l.alpha());
        try x.k.nv.run(&gp, .{ .x = x.qc, .xs = x.qs, .w = l.w, .ws = l.ws, .out = out }, x.s);
        try x.markEnd(gc);
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
    /// returns (negated; the audio back on the carry), into `vel_v` / `vel_a`. Host work a step repeats is cached per
    /// layout (the RoPE table, the token -> modulation-row map); what changes with sigma (the curve embedding) goes up
    /// on the stream, so a step needs no host sync. The carry and un-carry take sigma-derived scalars as kernel arguments
    /// and stay outside the step graph; everything between them is the graph (one per shape and buffer set: its first
    /// use runs eagerly, which loads whatever a kernel loads lazily, the second captures, later ones replay).
    pub fn step(x: *Dit, video: u64, audio: u64, sigma: f32, t: u32, h: u32, w: u32, a: u32, vel_v: u64, vel_a: u64) !void {
        const L = x.ctx_rows;
        const ly = lay.Layout.init(L, t, h, w, a);
        if (ly.s > x.max_s) return error.VideoTooLong;
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
            x.idx_key = null;
        }
        const sc = lay.stepScalars(sigma);
        var unique: [2]f32 = undefined;
        const mr = ly.modRows(sc, &unique, x.idx_host[0..ly.s]);
        var temb: [2 * t_dim]f32 = undefined;
        lay.curveTEmb(x.w.table, grid, t_dim, unique[0..mr.m], temb[0 .. mr.m * t_dim]);
        const ik: [7]u32 = .{ L, t, h, w, a, mr.row_v, mr.row_a };
        if (x.idx_key == null or !std.mem.eql(u32, &x.idx_key.?, &ik)) {
            try x.uploadAsync(x.idx, std.mem.sliceAsBytes(x.idx_host[0..ly.s]));
            x.idx_key = ik;
        }
        try x.uploadAsync(x.t_emb, std.mem.sliceAsBytes(temb[0 .. mr.m * t_dim]));
        const na: u64 = 32 * 2 * @as(u64, a);

        try x.pre("carry", &.{.{ .role = "x", .ptr = audio, .bytes = na * 2 }});
        try x.markBegin();
        try x.k.h3.scale(x.s, audio, sc.carry, x.a_in, na);
        try x.markEnd(.carry);
        try x.post("carry", &.{.{ .role = "y", .ptr = x.a_in, .bytes = na * 2 }});

        const bd: Body = .{ .video = video, .vel_v = vel_v, .vel_a = vel_a, .t = t, .h = h, .w = w, .a = a, .ly = ly, .m = mr.m, .row_v = mr.row_v, .row_a = mr.row_a };
        if (x.probe != null or x.prof != null or !x.graphs) try x.stepBody(bd) else try x.stepGraph(bd);

        try x.pre("uncarry", &.{ .{ .role = "a", .ptr = x.a_in, .bytes = na * 2 }, .{ .role = "v", .ptr = vel_a, .bytes = na * 2 } });
        try x.markBegin();
        try x.k.h3.uncarry(x.s, x.a_in, vel_a, sc.c1, sc.c2, na);
        try x.markEnd(.carry);
        try x.post("uncarry", &.{.{ .role = "y", .ptr = vel_a, .bytes = na * 2 }});
    }

    /// The step graph of this shape and buffer set: replayed if captured; eager the first time a key is seen; captured
    /// (and launched) the second time. A capture that fails turns graphs off (`graph_error` says why) and runs eagerly.
    fn stepGraph(x: *Dit, bd: Body) !void {
        const gk: [12]u64 = .{ bd.ly.l, bd.t, bd.h, bd.w, bd.a, bd.m, bd.row_v, bd.row_a, bd.video, bd.vel_v, bd.vel_a, @intFromBool(x.fuse) };
        var at: ?usize = null;
        for (x.step_graphs.items, 0..) |sg, i| {
            if (std.mem.eql(u64, &sg.key, &gk)) {
                at = i;
                break;
            }
        }
        if (at) |i| if (x.step_graphs.items[i].exec) |e| return e.launchOn(x.s);
        if (at == null) {
            if (x.step_graphs.items.len >= max_graphs) {
                var old = x.step_graphs.orderedRemove(0);
                if (old.exec) |*e| e.deinit();
            }
            try x.step_graphs.append(x.gpa, .{ .key = gk, .exec = null });
            return x.stepBody(bd);
        }
        cuda.graph.beginCapture(x.s, .thread_local) catch |err| return x.graphFallback(err, bd);
        var failed: ?anyerror = null;
        x.stepBody(bd) catch |err| {
            failed = err;
        };
        var g = cuda.graph.endCapture(x.s) catch |err| return x.graphFallback(failed orelse err, bd);
        defer g.deinit();
        if (failed) |err| return x.graphFallback(err, bd);
        const exec = g.instantiate() catch |err| return x.graphFallback(err, bd);
        x.step_graphs.items[at.?].exec = exec;
        try exec.launchOn(x.s);
    }

    fn graphFallback(x: *Dit, err: anyerror, bd: Body) !void {
        x.graphs = false;
        x.graph_error = err;
        return x.stepBody(bd);
    }

    /// The step's launches between the carry and the un-carry: the patch embeddings, the blocks, the final layer. Nothing
    /// in it depends on sigma but through `t_emb`'s contents (uploaded before), so a graph of it replays at any sigma.
    fn stepBody(x: *Dit, bd: Body) !void {
        const S: u64 = bd.ly.s;
        const m: u64 = bd.m;
        const t = bd.t;
        const h = bd.h;
        const w = bd.w;
        const a = bd.a;
        const na: u64 = 32 * 2 * @as(u64, a);
        const nv: u64 = @as(u64, t) * (h / 2) * (w / 2);
        const nvx: u64 = 24 * @as(u64, t) * h * w;
        const ar = bd.ly.audioRows();
        const vr = bd.ly.videoRows();

        try x.markBegin();
        try x.pre("video_rows", &.{.{ .role = "x", .ptr = bd.video, .bytes = nvx * 2 }});
        try x.k.h3.patchify(x.s, bd.video, x.rows, 24, t, h, w);
        try x.post("video_rows", &.{.{ .role = "y", .ptr = x.rows, .bytes = nv * 96 * 4 }});
        try x.pre("audio_rows", &.{.{ .role = "x", .ptr = x.a_in, .bytes = na * 2 }});
        try x.k.h3.packAudio(x.s, x.a_in, x.arows, 32, a);
        try x.post("audio_rows", &.{.{ .role = "y", .ptr = x.arows, .bytes = na * 4 }});
        try x.gemmF32("video_patch_proj", x.rows, x.w.vp_w, x.w.vp_b, x.emb, nv, dim, 96);
        try x.gemmF32("audio_patch_proj", x.arows, x.w.ap_w, x.w.ap_b, x.aemb, 2 * @as(u64, a), dim, 32);
        // [text | audio | video] rows: the context, then the embeddings rounded to bf16
        try x.k.ops.copyRows(x.s, x.h, x.context, x.ctx_rows, dim * 2, dim * 2, dim * 2);
        x.extra_launches += 1;
        try x.k.h3.toBf16(x.s, x.aemb, x.h + @as(u64, ar[0]) * dim * 2, 2 * @as(u64, a) * dim);
        try x.k.h3.toBf16(x.s, x.emb, x.h + @as(u64, vr[0]) * dim * 2, nv * dim);
        try x.markEnd(.embed);

        if (x.fuse and x.probe == null) {
            try x.blocksFused(S, m);
        } else {
            for (&x.w.blocks, 0..) |*b, i| try x.block(i, b, S, m);
        }

        // the final layer: fp32 curve modulation of each target segment, fp32 heads, negated, back to the latents
        try x.markBegin();
        try x.gemmF32("final.adaln", x.t_emb, x.w.fl_ada_w, x.w.fl_ada_b, x.fada, m, 2 * dim, t_dim);
        inline for (.{ .{ "final.mod_video", vr, bd.row_v, x.emb }, .{ "final.mod_audio", ar, bd.row_a, x.aemb } }) |e| {
            const rows: u64 = e[1][1] - e[1][0];
            const shift = x.fada + @as(u64, e[2]) * 2 * dim * 4;
            try x.pre(e[0], &.{.{ .role = "x", .ptr = x.h + @as(u64, e[1][0]) * dim * 2, .bytes = rows * dim * 2 }});
            try x.k.h3.finalMod(x.s, x.h + @as(u64, e[1][0]) * dim * 2, x.w.fl_norm, shift + dim * 4, shift, e[3], rows, dim, eps);
            try x.post(e[0], &.{.{ .role = "y", .ptr = e[3], .bytes = rows * dim * 4 }});
        }
        try x.gemmF32("final.video_out", x.emb, x.w.vo_w, x.w.vo_b, x.vout, nv, 96, dim);
        try x.gemmF32("final.audio_out", x.aemb, x.w.ao_w, x.w.ao_b, x.aout, 2 * @as(u64, a), 32, dim);
        try x.pre("video_out", &.{.{ .role = "x", .ptr = x.vout, .bytes = nv * 96 * 4 }});
        try x.k.h3.unpatchifyNeg(x.s, x.vout, bd.vel_v, 24, t, h, w);
        try x.post("video_out", &.{.{ .role = "y", .ptr = bd.vel_v, .bytes = nvx * 2 }});
        try x.pre("audio_out", &.{.{ .role = "x", .ptr = x.aout, .bytes = na * 4 }});
        try x.k.h3.unpackAudioNeg(x.s, x.aout, bd.vel_a, 32, a);
        try x.post("audio_out", &.{.{ .role = "y", .ptr = bd.vel_a, .bytes = na * 2 }});
        try x.markEnd(.final);
    }

    fn adalnGemm(x: *Dit, i: usize, dst: u64, m: u64) !void {
        const b = &x.w.blocks[i];
        try x.markBegin();
        try x.k.h3.gemmF32(x.s, x.t_emb, b.ada_w, b.ada_b, dst, m, ada_n, t_dim);
        try x.markEnd(.adaln);
    }

    /// The 50 blocks on the fast schedule (no probes): the reference `block`'s launches in the same order, except that (1)
    /// a gate_add and the norm_mod after it are one kernel, the gate_add1 + norm_mod2 of a block and its gate_add2 + the
    /// next block's norm_mod1 (each under its own block's modulation, so two modulation buffers alternate; the next
    /// block's adaln, which reads no activation, runs just before), and (2) the attention kernel stores rows, so the
    /// [H, S, D] -> rows pass is gone. Each kernel's arithmetic is the reference's (see ops.cu and kitchen_launch.cu).
    fn blocksFused(x: *Dit, S: u64, m: u64) !void {
        const rows: u32 = @intCast(S);
        const blks = &x.w.blocks;
        const mods = [2]u64{ x.mod, x.mod2 };
        try x.adalnGemm(0, mods[0], m);
        try x.markBegin();
        try x.k.h3.normMod(x.s, x.h, blks[0].norm1, mods[0], x.idx, x.hn, S, dim, 0, eps);
        try x.markEnd(.norm_mod);
        for (blks, 0..) |*b, i| {
            const cur = mods[i % 2];
            try x.linearRun(b.lin[0], x.hn, x.qkv, rows, .gemm_qkv);
            try x.markBegin();
            try x.k.kitchen.rmsRope(x.s, x.qkv, x.rope, b.q_norm, b.k_norm, heads, rows, 96, eps);
            try x.markEnd(.rms_rope);
            try x.markBegin();
            try x.k.kitchen.attentionPrep(x.s, x.qkv, x.ws, heads, rows);
            try x.markEnd(.attn_prep);
            try x.markBegin();
            try x.k.kitchen.attentionKernel(x.s, x.att, x.ws, heads, rows, true);
            try x.markEnd(.attn_kernel);
            try x.linearRun(b.lin[1], x.att, x.o, rows, .gemm_out);
            try x.markBegin(); // gate_add1 + norm_mod2
            try x.k.h3.gateAddNormMod(x.s, x.h, x.o, cur, cur, x.idx, b.norm2, x.hn, S, dim, 0, 1, eps);
            try x.markEnd(.gate_norm);
            try x.linearRun(b.lin[2], x.hn, x.gu, rows, .gemm_fc1);
            try x.markBegin();
            try x.k.h3.swiglu(x.s, x.gu, x.act, S, ffn, false);
            try x.markEnd(.swiglu);
            try x.linearRun(b.lin[3], x.act, x.o, rows, .gemm_fc2);
            if (i + 1 < layers) {
                const nxt = mods[(i + 1) % 2];
                try x.adalnGemm(i + 1, nxt, m);
                try x.markBegin(); // gate_add2 + the next block's norm_mod1
                try x.k.h3.gateAddNormMod(x.s, x.h, x.o, cur, nxt, x.idx, blks[i + 1].norm1, x.hn, S, dim, 1, 0, eps);
                try x.markEnd(.gate_norm);
            } else {
                try x.markBegin();
                try x.k.h3.gateAdd(x.s, x.h, x.o, cur, x.idx, S, dim, 1);
                try x.markEnd(.gate_add);
            }
        }
    }

    /// The reference block: the twin's op sequence under its names, probed op by op.
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
        try x.markBegin();
        try x.gemmF32(nm(&nb, i, "adaln"), x.t_emb, b.ada_w, b.ada_b, x.mod, m, ada_n, t_dim);
        try x.markEnd(.adaln);
        inline for (.{ .{ "norm_mod1", 0, b.norm1 }, .{ "norm_mod2", 1, b.norm2 } }, 0..) |e, half| {
            const n = nm(&nb, i, e[0]);
            try x.pre(n, &.{ .{ .role = "x", .ptr = x.h, .bytes = xs }, .{ .role = "mod", .ptr = x.mod, .bytes = mods }, .{ .role = "idx", .ptr = x.idx, .bytes = S * 4 } });
            try x.markBegin();
            try x.k.h3.normMod(x.s, x.h, e[2], x.mod, x.idx, x.hn, S, dim, e[1], eps);
            try x.markEnd(.norm_mod);
            try x.post(n, &.{.{ .role = "y", .ptr = x.hn, .bytes = xs }});
            if (half == 0) { // attention half
                try x.linear(nm(&nb, i, "qkv"), b.lin[0], x.hn, x.qkv, rows, .gemm_qkv);
                const qkv_bytes = S * 3 * inner * 2;
                const rn = nm(&nb, i, "rms_rope");
                try x.pre(rn, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                try x.markBegin();
                try x.k.kitchen.rmsRope(x.s, x.qkv, x.rope, b.q_norm, b.k_norm, heads, rows, 96, eps);
                try x.markEnd(.rms_rope);
                try x.post(rn, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                const an = nm(&nb, i, "attention");
                try x.pre(an, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                try x.markBegin();
                try x.k.kitchen.attentionPrep(x.s, x.qkv, x.ws, heads, rows);
                try x.markEnd(.attn_prep);
                try x.markBegin();
                try x.k.kitchen.attentionKernel(x.s, x.heads_out, x.ws, heads, rows, false);
                try x.markEnd(.attn_kernel);
                try x.markBegin();
                try x.k.h3.headsToRows(x.s, x.heads_out, x.att, heads, S, head_dim);
                try x.markEnd(.heads_rows);
                try x.post(an, &.{.{ .role = "y", .ptr = x.att, .bytes = S * inner * 2 }});
                try x.linear(nm(&nb, i, "out"), b.lin[1], x.att, x.o, rows, .gemm_out);
            } else { // MLP half
                try x.linear(nm(&nb, i, "fc1"), b.lin[2], x.hn, x.gu, rows, .gemm_fc1);
                const sn = nm(&nb, i, "swiglu");
                try x.pre(sn, &.{.{ .role = "x", .ptr = x.gu, .bytes = S * 2 * ffn * 2 }});
                try x.markBegin();
                try x.k.h3.swiglu(x.s, x.gu, x.act, S, ffn, false);
                try x.markEnd(.swiglu);
                try x.post(sn, &.{.{ .role = "y", .ptr = x.act, .bytes = S * ffn * 2 }});
                try x.linear(nm(&nb, i, "fc2"), b.lin[3], x.act, x.o, rows, .gemm_fc2);
            }
            const gn = nm(&nb, i, if (half == 0) "gate_add1" else "gate_add2");
            try x.pre(gn, &.{ .{ .role = "x", .ptr = x.h, .bytes = xs }, .{ .role = "y", .ptr = x.o, .bytes = xs } });
            try x.markBegin();
            try x.k.h3.gateAdd(x.s, x.h, x.o, x.mod, x.idx, S, dim, e[1]);
            try x.markEnd(.gate_add);
            try x.post(gn, &.{.{ .role = "x", .ptr = x.h, .bytes = xs }});
        }
    }
};

test "the step's code type-checks (no GPU here: the h3-replay and h3-step-bench gates run it)" {
    _ = &Dit.step;
    _ = &Dit.prepare;
    _ = &Dit.init;
    _ = &Dit.launches;
}
