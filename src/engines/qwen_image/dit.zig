//! The Qwen-Image 2.1 DiT forward, op for op as the twin (`stk_twin/dit.py`) runs it, with the twin's op names. Each op
//! reports its inputs and outputs to an optional `Probe` (the replay's bit gate); without one it is just the engine.
//! Batch 1. Latents are [64, H * W] (CHW); the step runs the target rows over [kept text prefix | target] keys.

const std = @import("std");
const cuda = @import("cuda");
const W = @import("weights.zig");
const plan = @import("nvfp4_plan.zig");
const Nvfp4 = @import("nvfp4_exec.zig").Nvfp4;
const Ops = @import("ops_launch.zig").Ops;
const Triton = @import("triton_k.zig").Triton;
const rope = @import("rope.zig");

const D = W.dim;
const H = 32;
const HD = 128;
const F = W.mlp;
const eps: f32 = 1e-6;
const tile: u32 = 12; // the prompt tile the twin passes (the bulk-copy GEMM)

/// A device tensor an op reads or writes, by the twin's role name. `contiguous` false: a strided view (not reloadable).
pub const Io = struct { role: []const u8, ptr: u64, bytes: u64, contiguous: bool = true };

pub const Probe = struct {
    ctx: *anyopaque,
    before: *const fn (ctx: *anyopaque, name: []const u8, ins: []const Io) anyerror!void,
    after: *const fn (ctx: *anyopaque, name: []const u8, outs: []const Io) anyerror!void,
};

const StepGraph = struct { key: [5]u64, exec: cuda.graph.Exec };
const max_graphs = 8;

pub const Kernels = struct { ops: *const Ops, nv: *Nvfp4, tri: *const Triton, major: u32, minor: u32 };

/// Device scratch for at most `max_rows` rows a pass (target or prefix) and `max_prefix` kept text rows.
pub const Dit = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: Kernels,
    w: *const W.Weights,
    s: cuda.Stream,
    probe: ?Probe = null,
    om: [64]f64,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    max_rows: u32,
    max_prefix: u32,
    // side path
    t: u64 = 0, emb: u64 = 0, h1: u64 = 0, h1s: u64 = 0, temb: u64 = 0, ts: u64 = 0, m: u64 = 0,
    g1c: u64 = 0, g2c: u64 = 0, tg1: u64 = 0, tg2: u64 = 0, s1f: u64 = 0, s2f: u64 = 0, s1z: u64 = 0, s2z: u64 = 0,
    // per pass
    hn: u64 = 0, qkv: u64 = 0, q: u64 = 0, attn: u64 = 0, o: u64 = 0, h2: u64 = 0, y: u64 = 0,
    qc: u64 = 0, qs: u64 = 0, gc: u64 = 0, gs: u64 = 0, gu: u64 = 0, sw: u64 = 0,
    kbuf: u64 = 0, vbuf: u64 = 0, cos: u64 = 0, sin: u64 = 0, scos: u64 = 0, ssin: u64 = 0,
    // the tables the next block pass reads: the prefix's (cos / sin) or the step's (scos / ssin, kept per size)
    rcos: u64 = 0, rsin: u64 = 0,
    step_tables: ?[3]u32 = null, // (h, w, prefix rows) whose image tables scos / ssin hold
    /// Steps replay as CUDA graphs (one per size, prefix length and buffer pair) unless a probe is set.
    graphs: bool = true,
    step_graphs: std.ArrayList(StepGraph) = .empty,
    sigma_host: f32 = 0,
    xp: u64 = 0, tn: u64 = 0, th: u64 = 0, tg: u64 = 0, xin: u64 = 0, h: u64 = 0,
    os: u64 = 0, nob: u64 = 0, nof: u64 = 0, hout: u64 = 0, po: u64 = 0,
    // the kept prefix: per block k and v [P, 4096]
    pk: [W.layers]u64 = undefined,
    pv: [W.layers]u64 = undefined,
    prefix_rows: u32 = 0,
    mod_sigma: ?f32 = null,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: Kernels, w: *const W.Weights, s: cuda.Stream, om: [64]f64, max_rows: u32, max_prefix: u32) !Dit {
        var x: Dit = .{ .gpa = gpa, .d = d, .k = k, .w = w, .s = s, .om = om, .max_rows = max_rows, .max_prefix = max_prefix };
        errdefer x.deinit();
        const r: u64 = @max(max_rows, max_prefix);
        const qr = plan.rows(@intCast(r), F, plan.TB); // the widest quantized rows (K up to 12288)
        const bf = 2;
        inline for (.{
            .{ "t", 4 },                  .{ "emb", 2 * 256 * bf },      .{ "h1", 2 * D * bf },          .{ "h1s", 2 * D * bf },
            .{ "temb", 2 * D * bf },      .{ "ts", 2 * D * bf },         .{ "m", 2 * 4 * D * bf },      .{ "g1c", 2 * D * bf },
            .{ "g2c", 2 * D * bf },       .{ "tg1", 2 * D * bf },        .{ "tg2", 2 * D * bf },         .{ "s1f", D * 4 },
            .{ "s2f", D * 4 },            .{ "s1z", D * 4 },             .{ "s2z", D * 4 },              .{ "os", D * bf },
            .{ "nob", D * bf },           .{ "nof", D * 4 },
        }) |e| @field(x, e[0]) = try x.alloc(e[1]);
        inline for (.{ "hn", "q", "attn", "o", "h2", "y", "h", "hout" }) |nm| @field(x, nm) = try x.alloc(r * D * bf);
        x.qkv = try x.alloc(r * 3 * D * bf);
        x.qc = try x.alloc(qr.fp8_bytes); // NVFP4 codes (half) or FP8 bytes of the widest rows
        x.qs = try x.alloc(qr.scales_bytes);
        x.gc = try x.alloc(r * F / 2);
        x.gs = try x.alloc(plan.rows(@intCast(r), F, 0).scales_bytes);
        x.gu = try x.alloc(r * 2 * F * bf);
        x.sw = try x.alloc(r * F * bf);
        x.kbuf = try x.alloc((max_rows + max_prefix) * D * bf);
        x.vbuf = try x.alloc((max_rows + max_prefix) * D * bf);
        x.cos = try x.alloc(r * 64 * 4);
        x.sin = try x.alloc(r * 64 * 4);
        x.scos = try x.alloc(@as(u64, max_rows) * 64 * 4);
        x.ssin = try x.alloc(@as(u64, max_rows) * 64 * 4);
        inline for (.{ "xp", "tn", "th", "tg" }) |nm| @field(x, nm) = try x.alloc(@as(u64, max_prefix) * D * bf);
        x.xin = try x.alloc(@as(u64, max_rows) * 64 * bf);
        x.po = try x.alloc(@as(u64, max_rows) * 64 * bf);
        for (0..W.layers) |i| {
            x.pk[i] = try x.alloc(@as(u64, max_prefix) * D * bf);
            x.pv[i] = try x.alloc(@as(u64, max_prefix) * D * bf);
        }
        return x;
    }

    pub fn deinit(x: *Dit) void {
        for (x.step_graphs.items) |*g| g.exec.deinit();
        x.step_graphs.deinit(x.gpa);
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(x.gpa);
    }

    fn alloc(x: *Dit, n: u64) !u64 {
        var b = try cuda.DeviceBuffer.alloc(x.d, @intCast(@max(n, 256)));
        errdefer b.free();
        try x.bufs.append(x.gpa, b);
        return b.at(0);
    }

    /// Host bytes to the device. Synchronous on the legacy stream, which the engine's blocking stream orders with.
    pub fn upload(x: *Dit, ptr: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    fn zero(x: *Dit, ptr: u64, n: u64) !void { // on the stream: capturable into a step graph
        try x.d.check(x.d.api.cuMemsetD8Async(ptr, 0, n, x.s.handle), "cuMemsetD8Async");
    }

    fn pre(x: *Dit, name: []const u8, ins: []const Io) !void {
        if (x.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(x: *Dit, name: []const u8, outs: []const Io) !void {
        if (x.probe) |p| try p.after(p.ctx, name, outs);
    }

    // ------------------------------------------------------------------ ops
    fn dense(x: *Dit, name: []const u8, in: u64, wt: u64, out: u64, m: u64, n: u64, kk: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = m * kk * 2 }});
        try x.k.ops.dense(x.s, in, wt, out, m, n, kk);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * n * 2 }});
    }

    fn pointwise(x: *Dit, name: []const u8, f: Ops.Pointwise, in: u64, out: u64, n: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = n * 2 }});
        try x.k.ops.pointwise(x.s, f, in, out, n);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = n * 2 }});
    }

    fn adaln(x: *Dit, name: []const u8, in: u64, sf: u64, out: u64, rows: u32) !void {
        try x.pre(name, &.{ .{ .role = "x", .ptr = in, .bytes = @as(u64, rows) * D * 2 }, .{ .role = "s", .ptr = sf, .bytes = D * 4 } });
        try x.k.tri.adalnRun(x.s, in, sf, out, 1, rows, eps);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = @as(u64, rows) * D * 2 }});
    }

    /// One block linear (the twin's `linear` op): rows quantized under the static input scale, the prompt GEMM.
    fn linear(x: *Dit, name: []const u8, l: W.Linear, in: u64, out: u64, rows: u32) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = @as(u64, rows) * l.k * 2 }});
        try l.forward(x.k.nv, x.k.major, x.k.minor, tile, in, out, rows, x.qc, x.qs, x.s);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = @as(u64, rows) * l.n * 2 }});
    }

    /// The NVFP4 MLP in one pass (`mlp_prompt`): row-major NVFP4 rows, gate|up with the SwiGLU epilogue writing down's
    /// NVFP4 rows, then down on the plain prompt GEMM.
    fn mlpNvfp4(x: *Dit, name: []const u8, b: *const W.Block, in: u64, out: u64, rows: u32) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = @as(u64, rows) * D * 2 }});
        const kk = x.k;
        const t = F / 64;
        const gate = b.gate_up.tiles(0, t);
        const up = b.gate_up.tiles(t, 2 * t);
        const r = plan.rows(rows, D, 0);
        if (r.zero_scales) try x.zero(x.qs, r.scales_bytes);
        const q = try plan.quant4(rows, D, D, 0, gate.act);
        try kk.nv.run(&q, .{ .x = in, .codes = x.qc, .scales = x.qs }, x.s);
        const g = try plan.gemmGu(rows, gate.npad, D, r.mpad, gate.alpha(), up.alpha(), plan.inv(b.down.act), true);
        try kk.nv.run(&g, .{ .x = x.qc, .xs = x.qs, .w = gate.w, .ws = gate.ws, .up_w = up.w, .up_ws = up.ws, .codes = x.gc, .scales = x.gs }, x.s);
        const dn = try plan.gemm(.a4, rows, b.down.n, F, b.down.npad, r.mpad, 0, false, b.down.alpha(), plan.isGb10(kk.major, kk.minor));
        try kk.nv.run(&dn, .{ .x = x.gc, .xs = x.gs, .w = b.down.w, .ws = b.down.ws, .out = out }, x.s);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = @as(u64, rows) * D * 2 }});
    }

    fn residual(x: *Dit, name: []const u8, stream: u64, a: u64, g: u64, rows: u32) !void {
        const n: u64 = @as(u64, rows) * D * 2;
        try x.pre(name, &.{ .{ .role = "x", .ptr = stream, .bytes = n }, .{ .role = "a", .ptr = a, .bytes = n }, .{ .role = "g", .ptr = g, .bytes = D * 2 } });
        try x.k.ops.gatedResidual(x.s, stream, a, g, 1, rows, D);
        try x.post(name, &.{.{ .role = "x", .ptr = stream, .bytes = n }});
    }

    fn ropeOp(x: *Dit, name: []const u8, norm: u64, part: u32, rows: u32, dst: u64, dst_rows: u32) !void {
        const qkv_bytes = @as(u64, rows) * 3 * D * 2;
        try x.pre(name, &.{ .{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }, .{ .role = "w", .ptr = norm, .bytes = HD * 4 }, .{ .role = "cos", .ptr = x.rcos, .bytes = @as(u64, rows) * 64 * 4 }, .{ .role = "sin", .ptr = x.rsin, .bytes = @as(u64, rows) * 64 * 4 } });
        try x.k.tri.ropeRun(x.s, .{ .src = x.qkv, .w = norm, .cos = x.rcos, .sin = x.rsin, .dst = dst, .src_b_stride = rows * 3 * D, .src_row_stride = 3 * D, .src_off = part * D, .dst_b_stride = dst_rows * D, .dst_row_stride = D, .n = rows, .b = 1 }, eps);
        try x.post(name, &.{.{ .role = "y", .ptr = dst, .bytes = @as(u64, rows) * D * 2 }});
    }

    /// One block over `rows` rows of `stream`; prefix: keys/values into the kept prefix, causal; step: behind it.
    fn block(x: *Dit, i: usize, stream: u64, rows: u32, prefix: bool) !void {
        var nb: [24]u8 = undefined;
        const tag: u8 = if (prefix) 'P' else 'L';
        const b = &x.w.blocks[i];
        const P = x.prefix_rows;
        const nm = struct {
            fn f(buf: []u8, t: u8, i_: usize, s: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "{c}{d}.{s}", .{ t, i_, s }) catch unreachable;
            }
        }.f;
        try x.adaln(nm(&nb, tag, i, "adaln1"), stream, if (prefix) x.s1z else x.s1f, x.hn, rows);
        try x.linear(nm(&nb, tag, i, "qkv"), b.qkv, x.hn, x.qkv, rows);
        try x.ropeOp(nm(&nb, tag, i, "rope_q"), b.norm_q, 0, rows, x.q, rows);
        const kdst = if (prefix) x.pk[i] else x.kbuf + @as(u64, P) * D * 2;
        const vdst = if (prefix) x.pv[i] else x.vbuf + @as(u64, P) * D * 2;
        try x.ropeOp(nm(&nb, tag, i, "rope_k"), b.norm_k, 1, rows, kdst, if (prefix) rows else P + rows);
        { // V = qkv[:, :, 2], a strided copy
            const name = nm(&nb, tag, i, "v");
            try x.pre(name, &.{.{ .role = "src", .ptr = x.qkv + 2 * D * 2, .bytes = @as(u64, rows) * D * 2, .contiguous = false }});
            try x.k.ops.copyRows(x.s, vdst, x.qkv + 2 * D * 2, rows, D * 2, 3 * D * 2, D * 2);
            try x.post(name, &.{.{ .role = "y", .ptr = if (prefix) x.pv[i] else x.vbuf, .bytes = @as(u64, if (prefix) rows else P + rows) * D * 2 }});
        }
        { // attention over the kept prefix (+ this pass's rows)
            const name = nm(&nb, tag, i, "attn");
            const keys: u32 = if (prefix) rows else P + rows;
            const kp = if (prefix) x.pk[i] else x.kbuf;
            const vp = if (prefix) x.pv[i] else x.vbuf;
            try x.pre(name, &.{ .{ .role = "q", .ptr = x.q, .bytes = @as(u64, rows) * D * 2 }, .{ .role = "k", .ptr = kp, .bytes = @as(u64, keys) * D * 2 }, .{ .role = "v", .ptr = vp, .bytes = @as(u64, keys) * D * 2 } });
            try x.k.ops.attention(x.s, x.q, kp, vp, x.attn, rows, H, H, keys, prefix, 0);
            try x.post(name, &.{.{ .role = "y", .ptr = x.attn, .bytes = @as(u64, rows) * D * 2 }});
        }
        try x.linear(nm(&nb, tag, i, "out"), b.out, x.attn, x.o, rows);
        try x.residual(nm(&nb, tag, i, "res1"), stream, x.o, if (prefix) x.tg1 + D * 2 else x.tg1, rows);
        try x.adaln(nm(&nb, tag, i, "adaln2"), stream, if (prefix) x.s2z else x.s2f, x.h2, rows);
        if (b.gate_up.kind == .nvfp4) {
            try x.mlpNvfp4(nm(&nb, tag, i, "mlp"), b, x.h2, x.y, rows);
        } else {
            try x.linear(nm(&nb, tag, i, "gate_up"), b.gate_up, x.h2, x.gu, rows);
            const name = nm(&nb, tag, i, "swiglu");
            try x.pre(name, &.{.{ .role = "gu", .ptr = x.gu, .bytes = @as(u64, rows) * 2 * F * 2 }});
            try x.k.tri.swigluRun(x.s, x.gu, x.sw, rows, F);
            try x.post(name, &.{.{ .role = "y", .ptr = x.sw, .bytes = @as(u64, rows) * F * 2 }});
            try x.linear(nm(&nb, tag, i, "down"), b.down, x.sw, x.y, rows);
        }
        try x.residual(nm(&nb, tag, i, "res2"), stream, x.y, if (prefix) x.tg2 + D * 2 else x.tg2, rows);
    }

    /// The time embedding and modulation for sigma; rows 0 (this step) and 1 (t = 0, the prefix).
    fn modulation(x: *Dit, sigma: f32) !void {
        if (x.mod_sigma) |m| if (m == sigma) return; // the twin computes it once a call; the prefix pass shares it
        x.mod_sigma = sigma;
        try x.upload(x.t, std.mem.asBytes(&sigma));
        try x.modulationKernels();
    }

    /// The modulation's launches from the sigma at `t` (a step graph replays these after writing t).
    fn modulationKernels(x: *Dit) !void {
        try x.pre("time.sinusoid", &.{.{ .role = "t", .ptr = x.t, .bytes = 4 }});
        try x.k.ops.timeSinusoid(x.s, x.t, x.emb, 1);
        try x.post("time.sinusoid", &.{.{ .role = "y", .ptr = x.emb, .bytes = 2 * 256 * 2 }});
        try x.dense("time.lin1", x.emb, x.w.t1, x.h1, 2, D, 256);
        try x.pointwise("time.silu", .silu, x.h1, x.h1s, 2 * D);
        try x.dense("time.lin2", x.h1s, x.w.t2, x.temb, 2, D, D);
        try x.pointwise("mod.silu", .silu, x.temb, x.ts, 2 * D);
        try x.dense("mod", x.ts, x.w.mod, x.m, 2, 4 * D, D);
        // chunks s1 | g1 | s2 | g2 of each row; the gates through tanh (contiguous copies first, as the twin)
        try x.k.ops.copyRows(x.s, x.g1c, x.m + 1 * D * 2, 2, D * 2, 4 * D * 2, D * 2);
        try x.k.ops.copyRows(x.s, x.g2c, x.m + 3 * D * 2, 2, D * 2, 4 * D * 2, D * 2);
        try x.pointwise("mod.tanh_g1", .tanh, x.g1c, x.tg1, 2 * D);
        try x.pointwise("mod.tanh_g2", .tanh, x.g2c, x.tg2, 2 * D);
        try x.k.ops.toF32(x.s, x.m, x.s1f, D); // row 0: this step's shifts in f32 for adaln
        try x.k.ops.toF32(x.s, x.m + 2 * D * 2, x.s2f, D);
        try x.k.ops.toF32(x.s, x.m + 4 * D * 2, x.s1z, D); // row 1: the prefix's (t = 0)
        try x.k.ops.toF32(x.s, x.m + 6 * D * 2, x.s2z, D);
    }

    fn uploadRope(x: *Dit, ids: []const [3]f32, cos: u64, sin: u64) !void {
        const n = ids.len * 64;
        const buf = try x.gpa.alloc(f32, 2 * n);
        defer x.gpa.free(buf);
        rope.table(ids, &x.om, buf[0..n], buf[n..]);
        try x.s.synchronize(); // the previous pass has read the tables
        try x.upload(cos, std.mem.sliceAsBytes(buf[0..n]));
        try x.upload(sin, std.mem.sliceAsBytes(buf[n..]));
        x.rcos = cos;
        x.rsin = sin;
    }

    /// The text prefix through every block (t = 0 modulation, causal), keys and values kept. `ctx` [p, 4096] bf16.
    pub fn buildPrefix(x: *Dit, ctx: u64, p: u32, sigma: f32) !void {
        if (p > x.max_prefix) return error.PrefixTooLong;
        try x.modulation(sigma);
        try x.pre("txt.norm", &.{ .{ .role = "x", .ptr = ctx, .bytes = @as(u64, p) * D * 2 }, .{ .role = "w", .ptr = x.w.txt_norm, .bytes = D * 4 } });
        try x.k.ops.rmsNorm(x.s, ctx, x.w.txt_norm, x.tn, p, D, eps);
        try x.post("txt.norm", &.{.{ .role = "y", .ptr = x.tn, .bytes = @as(u64, p) * D * 2 }});
        try x.dense("txt.in1", x.tn, x.w.txt_in1, x.th, p, D, D);
        try x.pointwise("txt.gelu", .gelu_tanh, x.th, x.tg, @as(u64, p) * D);
        try x.dense("txt.in2", x.tg, x.w.txt_in2, x.xp, p, D, D);
        const ids = try x.gpa.alloc([3]f32, p);
        defer x.gpa.free(ids);
        rope.textIds(ids, 0);
        try x.uploadRope(ids, x.cos, x.sin);
        x.prefix_rows = p;
        for (0..W.layers) |i| try x.block(i, x.xp, p, true);
    }

    /// One denoising step: latents `lat` [64, h * w] (CHW, bf16) at `sigma` -> velocity `vel` [64, h * w].
    /// The prefix must be built (`buildPrefix`) for this prompt; the modulation is recomputed for sigma. Without a
    /// probe the step's launches are captured once per (size, prefix length, lat, vel) and replayed as a CUDA graph:
    /// the same kernels with the same arguments, so the same bits, without the host's launch time.
    pub fn step(x: *Dit, lat: u64, sigma: f32, h: u32, w: u32, vel: u64) !void {
        const n = h * w;
        if (n > x.max_rows) return error.ImageTooLarge;
        const key: [3]u32 = .{ h, w, x.prefix_rows };
        if (x.step_tables == null or !std.mem.eql(u32, &x.step_tables.?, &key)) {
            const ids = try x.gpa.alloc([3]f32, n);
            defer x.gpa.free(ids);
            rope.imageIds(ids, h, w, x.prefix_rows, h, w);
            try x.uploadRope(ids, x.scos, x.ssin);
            x.step_tables = key;
        }
        x.rcos = x.scos;
        x.rsin = x.ssin;
        if (x.probe != null or !x.graphs) {
            try x.modulation(sigma); // skipped when the prefix pass already computed it for this sigma, as the twin

            return x.stepKernels(lat, h, w, vel);
        }
        // the sigma, then the step's graph (both on the stream, in order)
        x.mod_sigma = null; // the graph overwrites the modulation buffers
        x.sigma_host = sigma;
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(x.t, &x.sigma_host, 4, x.s.handle), "cuMemcpyHtoDAsync");
        const gk: [5]u64 = .{ h, w, x.prefix_rows, lat, vel };
        for (x.step_graphs.items) |*g| if (std.mem.eql(u64, &g.key, &gk)) return g.exec.launchOn(x.s);
        if (x.step_graphs.items.len >= max_graphs) {
            var old = x.step_graphs.orderedRemove(0);
            old.exec.deinit();
        }
        try cuda.graph.beginCapture(x.s, .thread_local);
        const captured = blk: {
            x.modulationKernels() catch |err| break :blk err;
            x.stepKernels(lat, h, w, vel) catch |err| break :blk err;
            break :blk {};
        };
        var g = try cuda.graph.endCapture(x.s);
        defer g.deinit();
        try captured;
        var exec = try g.instantiate();
        errdefer exec.deinit();
        try x.step_graphs.append(x.gpa, .{ .key = gk, .exec = exec });
        try exec.launchOn(x.s);
    }

    /// The step's launches after the modulation; the image's RoPE tables are in place.
    fn stepKernels(x: *Dit, lat: u64, h: u32, w: u32, vel: u64) !void {
        const n = h * w;
        const P = x.prefix_rows;
        try x.k.ops.transpose(x.s, lat, x.xin, 64, n); // [64, n] -> [n, 64]
        try x.dense("img_in", x.xin, x.w.img_in, x.h, n, D, 64);
        for (0..W.layers) |i| {
            try x.k.ops.copyRows(x.s, x.kbuf, x.pk[i], P, D * 2, D * 2, D * 2);
            try x.k.ops.copyRows(x.s, x.vbuf, x.pv[i], P, D * 2, D * 2, D * 2);
            try x.block(i, x.h, n, false);
        }
        try x.pointwise("out.silu", .silu, x.temb, x.os, D);
        try x.dense("norm_out", x.os, x.w.norm_out, x.nob, 1, D, D);
        try x.k.ops.toF32(x.s, x.nob, x.nof, D);
        try x.adaln("out.adaln", x.h, x.nof, x.hout, n);
        try x.dense("proj_out", x.hout, x.w.proj_out, x.po, n, 64, D);
        try x.k.ops.transpose(x.s, x.po, vel, n, 64); // [n, 64] -> [64, n]
    }

    /// The Euler step as the scheduler takes it: lat = bf16(lat + (sigma_next - sigma) * vel).
    pub fn euler(x: *Dit, name: []const u8, lat: u64, vel: u64, out: u64, n: u64, dt: f32) !void {
        try x.pre(name, &.{ .{ .role = "x", .ptr = lat, .bytes = n * 2 }, .{ .role = "v", .ptr = vel, .bytes = n * 2 } });
        try x.k.ops.euler(x.s, lat, vel, out, n, dt);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = n * 2 }});
    }
};
