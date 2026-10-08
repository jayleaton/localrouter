//! One tile of the video VAE decoder: ViT3D (`decode_tile` of `stk_twin/h3/vae_video.py`) op for op, under the twin's
//! recorder names (`vvae.c{c}.t{r}_{k}.gather` ... `.L{i}.norm1` ... `.unshuf`) and roles, on the kernels of
//! kernels/cuda/minimax/gemm_f16.cu and vae_video.cu. fp16 weights and activations; a tile is [24, 7, 16, 16] latents ->
//! 1797 tokens (1792 patches, 4 register tokens, 1 zero token), 36 blocks of 32 heads x 64, FFN 8192 gated.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const dit = @import("dit.zig");
const vo = @import("vae_video_ops.zig");

pub const Ops = vo.Ops;
pub const Io = dit.Io;
pub const Probe = dit.Probe;
const Pack = qi.pack.Pack;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;
const smath = qi.smath;

pub const layers = 36;
pub const heads = 32;
pub const hd = 64;
pub const dim = 2048;
pub const ffn = 8192;
pub const zc = 24;
pub const nsuf = 5; // 4 register tokens + 1 zero token
pub const max_tn = 7; // latent frames of a clip
pub const max_p = max_tn * 16 * 16; // patches of the largest tile
pub const max_s = max_p + nsuf;
pub const max_sp = (max_s + 7) / 8 * 8; // the score row pitch (multiple of 8)
const eps: f32 = 1e-5;

fn io(role: []const u8, ptr: u64, bytes: u64) Io {
    return .{ .role = role, .ptr = ptr, .bytes = bytes };
}

/// The recorder: probe calls under the twin's names. `on` is the twin's per-tile `_recording(rec(c, r, k))`; `keep`
/// restricts the recorded layers (null: all). The prefix is "vvae", "vvae.c{c}" or "vvae.c{c}.t{r}_{k}"; inside a
/// block `layer` adds ".L{i}".
pub const Rec = struct {
    probe: ?Probe = null,
    on: bool = false,
    keep: ?[]const u32 = null,
    layer: ?u32 = null,
    pfx: [48]u8 = undefined,
    pfx_len: usize = 0,
    buf: [96]u8 = undefined,

    pub fn setPrefix(r: *Rec, comptime fmt: []const u8, args: anytype) void {
        r.pfx_len = (std.fmt.bufPrint(&r.pfx, fmt, args) catch unreachable).len;
        r.layer = null;
    }

    fn name(r: *Rec, op: []const u8) []const u8 {
        const p = r.pfx[0..r.pfx_len];
        return (if (r.layer) |l| std.fmt.bufPrint(&r.buf, "{s}.L{d}.{s}", .{ p, l, op }) else std.fmt.bufPrint(&r.buf, "{s}.{s}", .{ p, op })) catch unreachable;
    }

    fn active(r: *const Rec) bool {
        if (!r.on or r.probe == null) return false;
        if (r.layer) |l| if (r.keep) |ks| {
            for (ks) |k| if (k == l) return true;
            return false;
        };
        return true;
    }

    pub fn pre(r: *Rec, op: []const u8, ins: []const Io) !void {
        if (r.active()) try r.probe.?.before(r.probe.?.ctx, r.name(op), ins);
    }
    pub fn post(r: *Rec, op: []const u8, outs: []const Io) !void {
        if (r.active()) try r.probe.?.after(r.probe.?.ctx, r.name(op), outs);
    }
};

/// The classes of `STK_VVAE_PROF`'s report: the GPU time between the events around an op goes to the class of the op.
pub const Phase = enum(u8) { gemm, attn_gemm, softmax, norm, rope, swiglu, other, place };
const n_phase = 8;
const phase_names = [n_phase][]const u8{ "gemm", "attn_gemm", "softmax", "norm", "rope", "swiglu", "other", "place" };

/// CUDA events recorded after every op of a tile (and its placement into the canvas): `begin` stamps the start, `mark(class)`
/// the end of an op, `end` waits for the last stamp and credits each interval to its class. Only when `STK_VVAE_PROF` is set
/// and no probe is attached (a probe's copies and compares would be billed to the ops). Every call is a no-op when off.
pub const Prof = struct {
    on: bool = false,
    evs: []cuda.Event = &.{},
    cls: []Phase = &.{},
    n: usize = 0,
    ms: [n_phase]f64 = @splat(0),

    const capacity = 8192; // a tile: 36 x (13 + 3 x 16 groups of 2 heads), at one head a group about 4,000

    fn init(p: *Prof, gpa: std.mem.Allocator, d: *const cuda.Driver) !void {
        p.evs = try gpa.alloc(cuda.Event, capacity);
        errdefer gpa.free(p.evs);
        p.cls = try gpa.alloc(Phase, capacity);
        errdefer gpa.free(p.cls);
        var made: usize = 0;
        errdefer for (p.evs[0..made]) |*e| e.deinit();
        while (made < capacity) : (made += 1) p.evs[made] = try cuda.Event.init(d, true);
    }

    fn deinit(p: *Prof, gpa: std.mem.Allocator) void {
        for (p.evs) |*e| e.deinit();
        gpa.free(p.evs);
        gpa.free(p.cls);
        p.* = .{};
    }

    pub fn begin(p: *Prof, s: cuda.Stream) !void {
        if (!p.on) return;
        try p.evs[0].record(s);
        p.n = 1;
    }

    pub fn mark(p: *Prof, s: cuda.Stream, ph: Phase) !void {
        if (!p.on or p.n == 0 or p.n >= p.evs.len) return;
        try p.evs[p.n].record(s);
        p.cls[p.n] = ph;
        p.n += 1;
    }

    pub fn end(p: *Prof) !void {
        if (!p.on or p.n < 2) {
            p.n = 0;
            return;
        }
        try p.evs[p.n - 1].synchronize();
        for (1..p.n) |i| p.ms[@intFromEnum(p.cls[i])] += try cuda.Event.elapsedMs(p.evs[i - 1], p.evs[i]);
        p.n = 0;
    }

    /// One line on stderr: the milliseconds of the decode by class (summed over its tiles and placements) and their sum.
    pub fn report(p: *const Prof) void {
        var sum: f64 = 0;
        for (p.ms) |m| sum += m;
        std.debug.print("vvae_phase_ms", .{});
        for (phase_names, p.ms) |nm, m| std.debug.print(" {s}={d:.1}", .{ nm, m });
        std.debug.print(" sum={d:.1}\n", .{sum});
    }
};

pub const Layer = struct { norm1: u64, qkv_w: u64, qkv_b: u64, out_w: u64, out_b: u64, scale1: u64, norm2: u64, w1_w: u64, w1_b: u64, w2_w: u64, w2_b: u64, scale2: u64 };

pub const Weights = struct {
    pqc_w: u64,
    pqc_b: u64,
    embed_w: u64,
    embed_b: u64,
    reg: u64,
    norm_out_w: u64,
    norm_out_b: u64,
    proj_w: u64,
    proj_b: u64,
    mean: u64, // latents_mean, std: fp16 [24] as the file holds them
    std: u64,
    inv_freq: u64, // fp16 [8]
    l: [layers]Layer,
};

/// RotaryEmbeddingND(48, 100, 3).inv_freq: 1 / 100 ** arange(0, 1, 0.125) in fp32, then the module's .to(fp16). The power
/// is the correctly rounded fp32 one (portable exp / log in f64, as te32.zig does for its frequencies).
pub fn invFreq() [8]f16 {
    var out: [8]f16 = undefined;
    const ln = smath.log(100.0);
    for (&out, 0..) |*o, k| {
        const p: f32 = @floatCast(smath.exp(@as(f64, @floatFromInt(k)) / 8.0 * ln));
        o.* = @floatCast(@as(f32, 1.0) / p);
    }
    return out;
}

/// Reads the checkpoint's fp16 tensors of exactly the expected shape onto the device.
const Loader = struct {
    st: *Store,
    up: *Uploader,
    p: *const Pack,
    nb: [96]u8 = undefined,

    fn tensor(l: *Loader, name: []const u8, shape: []const usize) !u64 {
        const t = try l.p.get(name);
        if (t.dtype != .f16 or t.rank != shape.len) return error.BadWeight;
        for (shape, 0..) |s, i| if (t.dim(i) != s) return error.BadWeight;
        return l.st.tensor(l.up, l.p, name);
    }

    fn block(l: *Loader, i: usize, suffix: []const u8, shape: []const usize) !u64 {
        const name = try std.fmt.bufPrint(&l.nb, "decoder.transformer_blocks.{d}.{s}", .{ i, suffix });
        return l.tensor(name, shape);
    }
};

pub fn loadWeights(st: *Store, up: *Uploader, p: *const Pack) !Weights {
    var l: Loader = .{ .st = st, .up = up, .p = p };
    var w: Weights = undefined;
    w.pqc_w = try l.tensor("post_quant_conv.weight", &.{ zc, zc }); // [24, 24, 1, 1, 1] in the file
    w.pqc_b = try l.tensor("post_quant_conv.bias", &.{zc});
    w.embed_w = try l.tensor("decoder.x_embedder.weight", &.{ dim, zc });
    w.embed_b = try l.tensor("decoder.x_embedder.bias", &.{dim});
    w.reg = try l.tensor("decoder.register_tokens", &.{ 1, 4, dim });
    w.norm_out_w = try l.tensor("decoder.norm_out.weight", &.{dim});
    w.norm_out_b = try l.tensor("decoder.norm_out.bias", &.{dim});
    w.proj_w = try l.tensor("decoder.proj_out.weight", &.{ 3 * 4 * 16 * 16, dim });
    w.proj_b = try l.tensor("decoder.proj_out.bias", &.{3 * 4 * 16 * 16});
    w.mean = try l.tensor("latents_mean", &.{zc});
    w.std = try l.tensor("latents_std", &.{zc});
    const inv = invFreq();
    w.inv_freq = try st.upload(up, std.mem.sliceAsBytes(&inv));
    for (&w.l, 0..) |*b, i| {
        b.norm1 = try l.block(i, "norm1.weight", &.{dim});
        b.qkv_w = try l.block(i, "attn.to_qkv.weight", &.{ 3 * dim, dim });
        b.qkv_b = try l.block(i, "attn.to_qkv.bias", &.{3 * dim});
        b.out_w = try l.block(i, "attn.to_out.weight", &.{ dim, dim });
        b.out_b = try l.block(i, "attn.to_out.bias", &.{dim});
        b.scale1 = try l.block(i, "scale1", &.{dim});
        b.norm2 = try l.block(i, "norm2.weight", &.{dim});
        b.w1_w = try l.block(i, "ff.w1.weight", &.{ 2 * ffn, dim });
        b.w1_b = try l.block(i, "ff.w1.bias", &.{2 * ffn});
        b.w2_w = try l.block(i, "ff.w2.weight", &.{ dim, ffn });
        b.w2_b = try l.block(i, "ff.w2.bias", &.{dim});
        b.scale2 = try l.block(i, "scale2", &.{dim});
    }
    return w;
}

/// The latents the tiles are cropped from: the denormalised [24, tz, hz, wz] fp16 on the device.
pub const Lat = struct { ptr: u64, tz: u32, hz: u32, wz: u32 };

pub const Tile = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Ops,
    s: cuda.Stream,
    rec: Rec = .{},
    prof: Prof = .{},
    store: Store,
    w: Weights = undefined,
    sc_numel: u64 = 0, // the score buffer's last layout (zeroed whenever it changes, as the twin reallocates it)
    // scratch, sized for the largest tile (7 x 16 x 16 latents)
    rows: u64 = 0,
    pq: u64 = 0,
    hs: u64 = 0,
    n: u64 = 0,
    qkv: u64 = 0,
    table: u64 = 0,
    vt: u64 = 0,
    sc: u64 = 0,
    att: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    proj: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const Ops, s: cuda.Stream, p: *const Pack, up: *Uploader) !Tile {
        var t: Tile = .{ .gpa = gpa, .d = d, .k = k, .s = s, .store = .init(d, gpa) };
        errdefer t.store.deinit();
        if (k.prof) try t.prof.init(gpa, d);
        errdefer t.prof.deinit(gpa);
        t.w = try loadWeights(&t.store, up, p);
        try t.store.done(up); // the weights have landed
        const sizes = .{
            .{ "rows", max_p * zc * 2 },           .{ "pq", max_p * zc * 2 },        .{ "hs", max_s * dim * 2 },        .{ "n", max_s * dim * 2 },
            .{ "qkv", max_s * 3 * dim * 2 },       .{ "table", max_s * 24 * 4 * 2 }, .{ "vt", heads * hd * max_sp * 2 }, .{ "sc", heads * max_s * max_sp * 2 },
            .{ "att", max_s * dim * 2 },           .{ "gu", max_s * 2 * ffn * 2 },   .{ "act", max_s * ffn * 2 },       .{ "proj", max_p * 3 * 4 * 16 * 16 * 2 },
        };
        inline for (sizes) |e| @field(t, e[0]) = try t.store.alloc(e[1]);
        return t;
    }

    pub fn deinit(t: *Tile) void {
        t.prof.deinit(t.gpa);
        t.store.deinit();
    }

    fn zero(t: *Tile, ptr: u64, bytes: u64) !void {
        try t.d.check(t.d.api.cuMemsetD8Async(ptr, 0, bytes, t.s.handle), "cuMemsetD8Async");
    }

    /// x [m, k] . wt [n, k]^T + bias into `out` (rows of n), rounded to half; with `res` / `rscale`: the addcmul residual,
    /// out = half(res + out * rscale) (out may be res). The twin's `_lin`.
    fn lin(t: *Tile, op: []const u8, x: u64, wt: u64, bias: u64, out: u64, m: u64, n: u64, kk: u64, res: u64, rscale: u64) !void {
        const rec = &t.rec;
        if (res == 0) {
            try rec.pre(op, &.{io("x", x, m * kk * 2)});
        } else {
            try rec.pre(op, &.{ io("x", x, m * kk * 2), io("res", res, m * n * 2) });
        }
        try t.k.gemm(t.s, .{ .a = x, .b = wt, .bias = bias, .c = out, .m = m, .n = n, .k = kk, .lda = kk, .ldb = kk, .ldc = n, .res = res, .rscale = rscale, .ldr = if (res == 0) 0 else n });
        try rec.post(op, &.{io("y", out, m * n * 2)});
        try t.prof.mark(t.s, .gemm);
    }

    /// softmax(q k^T / 8) v per head from the rotated qkv [S, 6144] -> rows [S, 2048] (t.att): q k^T (fp16 out), fp32 row
    /// softmax, P v; nan_to_num in place. A tile that is not recorded runs the three steps `Ops.heads` heads at a time
    /// (`attentionGrouped`), a recorded one all 32 batched as the capture names them.
    fn attention(t: *Tile, ss: u64, sp: u64) !void {
        const rec = &t.rec;
        const k = t.k;
        if (t.sc_numel != heads * ss * sp) {
            try t.zero(t.sc, heads * ss * sp * 2);
            t.sc_numel = heads * ss * sp;
        }
        const vt_bytes = heads * hd * sp * 2;
        const sc_bytes = heads * ss * sp * 2;
        try rec.pre("vt", &.{io("qkv", t.qkv, ss * 3 * dim * 2)});
        try k.vt(t.s, t.qkv, t.vt, ss, sp, heads, 3 * dim);
        try rec.post("vt", &.{io("y", t.vt, vt_bytes)});
        try t.prof.mark(t.s, .other);
        if (k.heads < heads and !rec.active()) {
            try t.attentionGrouped(ss, sp);
        } else {
            try rec.pre("sc", &.{io("qkv", t.qkv, ss * 3 * dim * 2)});
            try k.gemm(t.s, .{ .a = t.qkv, .b = t.qkv, .c = t.sc, .m = ss, .n = ss, .k = hd, .lda = 3 * dim, .ldb = 3 * dim, .ldc = sp, .b_off = hd, .sa = 3 * hd, .sb = 3 * hd, .sc = ss * sp, .batch = heads });
            try rec.post("sc", &.{io("y", t.sc, sc_bytes)});
            try t.prof.mark(t.s, .attn_gemm);
            try rec.pre("sm", &.{io("s", t.sc, sc_bytes)});
            try k.softmax(t.s, t.sc, heads * ss, ss, sp, 0.125);
            try rec.post("sm", &.{io("y", t.sc, sc_bytes)});
            try t.prof.mark(t.s, .softmax);
            try rec.pre("pv", &.{ io("p", t.sc, sc_bytes), io("vt", t.vt, vt_bytes) });
            try k.gemm(t.s, .{ .a = t.sc, .b = t.vt, .c = t.att, .m = ss, .n = hd, .k = sp, .lda = sp, .ldb = sp, .ldc = dim, .sa = ss * sp, .sb = hd * sp, .sc = hd, .batch = heads });
            try rec.post("pv", &.{io("y", t.att, ss * dim * 2)});
            try t.prof.mark(t.s, .attn_gemm);
        }
        try rec.pre("nan", &.{io("x", t.att, ss * dim * 2)});
        try k.nanToNum(t.s, t.att, ss * dim);
        try rec.post("nan", &.{io("y", t.att, ss * dim * 2)});
        try t.prof.mark(t.s, .other);
    }

    /// q k^T, softmax and P v for `Ops.heads` heads at a time on the first heads of the score buffer. A head's bits depend on
    /// that head's q, k, v alone, so the output equals the all-heads launches' (test_vae_video "attention_groups", the frames'
    /// hash); the scores of a group (6.5 MB a head at 1797 tokens) stay in the L2 instead of four trips of 207 MB through the
    /// LPDDR5X. Head h0 + i: q and k windows into qkv (offsets h0 * 192), v^T by h0 * 64 * sp, the output columns by h0 * 64.
    fn attentionGrouped(t: *Tile, ss: u64, sp: u64) !void {
        const k = t.k;
        var h0: u32 = 0;
        while (h0 < heads) : (h0 += k.heads) {
            const g: u32 = @min(k.heads, heads - h0);
            try k.gemm(t.s, .{ .a = t.qkv, .b = t.qkv, .c = t.sc, .m = ss, .n = ss, .k = hd, .lda = 3 * dim, .ldb = 3 * dim, .ldc = sp, .a_off = h0 * 3 * hd, .b_off = hd + h0 * 3 * hd, .sa = 3 * hd, .sb = 3 * hd, .sc = ss * sp, .batch = g });
            try t.prof.mark(t.s, .attn_gemm);
            try k.softmax(t.s, t.sc, @as(u64, g) * ss, ss, sp, 0.125);
            try t.prof.mark(t.s, .softmax);
            try k.gemm(t.s, .{ .a = t.sc, .b = t.vt, .c = t.att, .m = ss, .n = hd, .k = sp, .lda = sp, .ldb = sp, .ldc = dim, .b_off = @as(u64, h0) * hd * sp, .c_off = h0 * hd, .sa = ss * sp, .sb = hd * sp, .sc = hd, .batch = g });
            try t.prof.mark(t.s, .attn_gemm);
        }
    }

    /// ViT3DDecoder on post_quant_conv of the crop [t0, t0 + tn) x [y0, y0 + h) x [x0, x0 + w) of `z` (latent frames past
    /// the last repeat it) -> `out` [3, 4 tn, 16 h, 16 w] fp16. The recorder's prefix and `on` are set by the caller.
    pub fn decode(t: *Tile, z: Lat, t0: u32, tn: u32, y0: u32, h: u32, w: u32, x0: u32, out: u64) !void {
        const rec = &t.rec;
        const k = t.k;
        const np: u64 = @as(u64, tn) * h * w;
        const ss = np + nsuf;
        const sp = (ss + 7) / 8 * 8;
        rec.layer = null;
        try t.prof.begin(t.s);
        try rec.pre("gather", &.{io("z", z.ptr, @as(u64, zc) * z.tz * z.hz * z.wz * 2)});
        try k.gather(t.s, z.ptr, t.rows, zc, z.tz, z.hz, z.wz, t0, tn, y0, h, x0, w);
        try rec.post("gather", &.{io("y", t.rows, np * zc * 2)});
        try t.prof.mark(t.s, .other);
        try t.lin("pqc", t.rows, t.w.pqc_w, t.w.pqc_b, t.pq, np, zc, zc, 0, 0);
        try t.lin("embed", t.pq, t.w.embed_w, t.w.embed_b, t.hs, np, dim, zc, 0, 0);
        try rec.pre("suffix", &.{ io("h", t.hs, np * dim * 2), io("reg", t.w.reg, 4 * dim * 2) });
        try k.suffix(t.s, t.hs, t.w.reg, np, dim);
        try rec.post("suffix", &.{io("y", t.hs + np * dim * 2, nsuf * dim * 2)});
        try t.prof.mark(t.s, .other);
        try rec.pre("rope", &.{io("inv_freq", t.w.inv_freq, 16)});
        try k.ropeTable(t.s, t.w.inv_freq, t.table, tn, h, w, nsuf);
        try rec.post("rope", &.{io("y", t.table, ss * 24 * 4 * 2)});
        try t.prof.mark(t.s, .other);
        const hb = ss * dim * 2;
        for (&t.w.l, 0..) |*l, i| {
            rec.layer = @as(u32, @intCast(i));
            try rec.pre("norm1", &.{io("x", t.hs, hb)});
            try k.rmsNorm(t.s, t.hs, l.norm1, t.n, ss, dim, eps);
            try rec.post("norm1", &.{io("y", t.n, hb)});
            try t.prof.mark(t.s, .norm);
            try t.lin("qkv", t.n, l.qkv_w, l.qkv_b, t.qkv, ss, 3 * dim, dim, 0, 0);
            try rec.pre("rr", &.{ io("qkv", t.qkv, ss * 3 * dim * 2), io("table", t.table, ss * 24 * 4 * 2) });
            try k.rmsRope(t.s, t.qkv, t.table, ss, heads, 3 * dim, eps);
            try rec.post("rr", &.{io("qkv", t.qkv, ss * 3 * dim * 2)});
            try t.prof.mark(t.s, .rope);
            try t.attention(ss, sp);
            try t.lin("out", t.att, l.out_w, l.out_b, t.hs, ss, dim, dim, t.hs, l.scale1);
            try rec.pre("norm2", &.{io("x", t.hs, hb)});
            try k.rmsNorm(t.s, t.hs, l.norm2, t.n, ss, dim, eps);
            try rec.post("norm2", &.{io("y", t.n, hb)});
            try t.prof.mark(t.s, .norm);
            try t.lin("w1", t.n, l.w1_w, l.w1_b, t.gu, ss, 2 * ffn, dim, 0, 0);
            try rec.pre("swi", &.{io("x", t.gu, ss * 2 * ffn * 2)});
            try k.swiglu(t.s, t.gu, t.act, ss, ffn);
            try rec.post("swi", &.{io("y", t.act, ss * ffn * 2)});
            try t.prof.mark(t.s, .swiglu);
            try t.lin("w2", t.act, l.w2_w, l.w2_b, t.hs, ss, dim, ffn, t.hs, l.scale2);
        }
        rec.layer = null;
        // the head runs on the patch rows only (a row's bits depend on that row alone)
        try rec.pre("norm_out", &.{io("x", t.hs, np * dim * 2)});
        try k.layerNorm(t.s, t.hs, t.w.norm_out_w, t.w.norm_out_b, t.n, np, dim, eps);
        try rec.post("norm_out", &.{io("y", t.n, np * dim * 2)});
        try t.prof.mark(t.s, .norm);
        try t.lin("proj", t.n, t.w.proj_w, t.w.proj_b, t.proj, np, 3 * 4 * 16 * 16, dim, 0, 0);
        try rec.pre("unshuf", &.{io("x", t.proj, np * 3 * 4 * 16 * 16 * 2)});
        try k.unshuffle(t.s, t.proj, out, tn, h, w);
        try rec.post("unshuf", &.{io("y", out, @as(u64, 3) * 4 * tn * 16 * h * 16 * w * 2)});
        try t.prof.mark(t.s, .other);
        try t.prof.end();
    }
};

test "inv_freq: 1 first, then decreasing, 100 ** -0.875 last" {
    const f = invFreq();
    try std.testing.expectEqual(@as(f16, 1.0), f[0]);
    for (1..8) |k| try std.testing.expect(f[k] < f[k - 1]);
    try std.testing.expectApproxEqRel(@as(f32, 0.0177827941), @as(f32, @floatCast(f[7])), 1e-3);
}

test "recorder names follow the twin's scheme" {
    var r: Rec = .{};
    r.setPrefix("vvae.c{d}.t{d}_{d}", .{ 1, 0, 2 });
    try std.testing.expectEqualStrings("vvae.c1.t0_2.gather", r.name("gather"));
    r.layer = 35;
    try std.testing.expectEqualStrings("vvae.c1.t0_2.L35.w2", r.name("w2"));
    r.setPrefix("vvae", .{});
    try std.testing.expectEqualStrings("vvae.denorm", r.name("denorm"));
}
