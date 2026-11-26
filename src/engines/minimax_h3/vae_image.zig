//! MiniMax H3-Image's VAE decoder (ComfyUI e9afbccc `MiniMaxH3ImageVAE.decode`, the published fp16 checkpoint
//! `minimax_h3_image_vae_fp16.safetensors` read directly; docs/dev/H3-IMAGE.md 1.5): z = z * std + mean (fp16, the
//! H3 latent tables: the file has none), then ComfyUI's `tiled_decode` exactly (H3-IMAGE.md, "Decided"): `split_tiles`
//! 256-px tiles with overlaps of at least 64 px, rows top to bottom and tiles left to right, each tile blended in fp16
//! first vertically against the previous row's already blended strip, then horizontally against its left neighbour's
//! already blended tail, the overlaps cropped into the strip and the tail, the rest copied to the canvas; one tile a call
//! (the declared substitution for ComfyUI's memory-dependent batching). A tile is the 2-D ViT: post_quant_conv (1x1),
//! x_embedder 24 -> 1024, 4 register tokens and a zero token, 24 blocks of 16 heads x 64 (RMSNorm, qkv, per-head RMSNorm +
//! split-half RoPE over 32 of the 64 dims on (y, x), fp16 attention, nan_to_num, the addcmul residuals, the gated FFN
//! 8192 -> 4096), norm_out, proj_out 1024 -> 3 x 16 x 16. Then the pixels: clamp(v * std + mean, 0, 1) in fp32 and uint8.
//!
//! Kernels: the video decoder's (gemm_f16.cu, vae_video.cu: denorm, gather, suffix, the norms, V^T, softmax, nan_to_num,
//! SwiGLU, finalize) where they fit, and h3_image.cu's 2-D RoPE table, RMSNorm + RoPE over half the head, 3 x 16 x 16
//! unpatchify, tile blend and copies. Recorder names follow the video decoder's under `ivae` (`ivae.t{r}_{c}.L{i}.qkv`).

const std = @import("std");
const gpu = @import("gpu");
const qi = @import("qwen_image");
const vt = @import("vae_video_tile.zig");
const vv = @import("vae_video.zig");
const li = @import("launch_image.zig");

pub const Ops = vt.Ops;
pub const ImageOps = li.Ops;
pub const Io = vt.Io;
pub const Probe = vt.Probe;
const Pack = qi.pack.Pack;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;

pub const layers = 24;
pub const heads = 16;
pub const hd = 64;
pub const dim = heads * hd; // 1024
pub const ffn = 4096;
pub const zc = 24;
pub const nsuf = 5; // 4 register tokens + 1 zero token
pub const ratio = vv.ratio; // pixels a latent cell
pub const tile_size = vv.tile_size;
pub const tile_overlap_min = vv.tile_overlap_min;
const tile_lat = tile_size / ratio; // 16 latent cells a tile side
pub const max_p = tile_lat * tile_lat; // patches of a tile
pub const max_s = max_p + nsuf;
pub const max_sp = (max_s + 7) / 8 * 8; // the score row pitch
const out_ch = 3 * 16 * 16; // proj_out: (c, py, px)
const eps: f32 = 1e-5;

/// ComfyUI's LATENTS_MEAN / LATENTS_STD (comfy/ldm/minimax/vae.py): the image VAE file has no latent statistics, so
/// `sd.setdefault` gives these (torch.tensor: fp32), and the fp16 VAE holds them as fp16.
pub const latents_mean = [zc]f64{
    0.858090341091156,    -0.9606591463088989, 1.0661640167236328,   -0.5090325474739075,
    -0.2727581858634949,  -1.3675414323806763, -0.2553254961967468,  -0.26907554268836975,
    -0.5376840829849243,  -0.0464097298681736, 0.6657370328903198,   0.19690127670764923,
    -0.5460608005523682,  -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607,   -1.1206848621368408,  0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942,  0.22864092886447906,  0.7056031823158264,
};
pub const latents_std = [zc]f64{
    1.2223774194717407, 1.2767263650894165,  1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665,   0.96531379222869875, 1.05698859691619875,
    0.841948926448822,  0.7729952931404114,  1.8955937623977661,  0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745,  0.69362932443618775,
    2.961095094680786,  2.7694199085235595,  3.0496184825897215,  2.1088054180145265,
    3.276226282119751,  3.1627357006073,     2.28168129920959475, 2.6127843856811525,
};
const imagenet_mean = [3]f32{ 0.485, 0.456, 0.406 };
const imagenet_std = [3]f32{ 0.229, 0.224, 0.225 };

/// A Python float through torch.tensor (fp32) and the fp16 module: the half it holds.
fn half(v: f64) f16 {
    return @floatCast(@as(f32, @floatCast(v)));
}

/// `latents_mean` / `latents_std` as the fp16 VAE holds them (the encoder's posterior normalizes against these too:
/// H3ImageVAE.encode's `.to(latents)` of the fp16 buffers).
pub fn latentStats(mean: *[zc]f16, stdv: *[zc]f16) void {
    for (mean, stdv, latents_mean, latents_std) |*m, *s, mv, sv| {
        m.* = half(mv);
        s.* = half(sv);
    }
}

/// An fp16 buffer's value as the fp32 it is read as (`pixel_std.to(float32)` of the fp16 module's buffer).
fn f16Const(v: f32) f32 {
    return @as(f16, @floatCast(v));
}

// ---------------------------------------------------------------------------------------------------- the plan (host)

/// The largest overlap `vv.splitTiles(len)` hands out (0 for one tile): the surplus goes 16 px at a time round robin.
pub fn maxOverlap(len: u32) u32 {
    if (tile_size >= len) return 0;
    const n = vv.tileCount(len);
    const remaining: u32 = @intCast(@as(i64, tile_size) * n - @as(i64, tile_overlap_min) * (n - 1) - len);
    return tile_overlap_min + ratio * ((remaining / ratio + n - 2) / (n - 1));
}

/// blend()'s value at position p of an extent e, as vi_blend computes it (fp16 a, b; w_b = half(p * (1 / e)) in fp32,
/// w_a = half(1 - w_b), each product rounded to half, the sum rounded to half). The host's copy of the kernel's formula.
pub fn blendValue(a: f16, b: f16, p: u32, e: u32) f16 {
    const inv: f32 = 1.0 / @as(f32, @floatFromInt(e));
    const wb: f32 = @as(f16, @floatCast(@as(f32, @floatFromInt(p)) * inv));
    const wa: f32 = @as(f16, @floatCast(1.0 - wb));
    const pa: f32 = @as(f16, @floatCast(@as(f32, a) * wa));
    const pb: f32 = @as(f16, @floatCast(@as(f32, b) * wb));
    return @floatCast(pa + pb);
}

// ---------------------------------------------------------------------------------------------------- weights

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
    mean: u64, // fp16 [24]
    std: u64,
    inv_freq: u64, // fp16 [8]: RotaryEmbeddingND(32, 100, 2), the video decoder's frequencies
    l: [layers]Layer,
};

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
        return l.tensor(try std.fmt.bufPrint(&l.nb, "decoder.transformer_blocks.{d}.{s}", .{ i, suffix }), shape);
    }
};

fn loadWeights(st: *Store, up: *Uploader, p: *const Pack) !Weights {
    var l: Loader = .{ .st = st, .up = up, .p = p };
    var w: Weights = undefined;
    w.pqc_w = try l.tensor("post_quant_conv.weight", &.{ zc, zc, 1, 1 }); // the 2-D 1x1 conv
    w.pqc_b = try l.tensor("post_quant_conv.bias", &.{zc});
    w.embed_w = try l.tensor("decoder.x_embedder.weight", &.{ dim, zc });
    w.embed_b = try l.tensor("decoder.x_embedder.bias", &.{dim});
    w.reg = try l.tensor("decoder.register_tokens", &.{ 1, 4, dim });
    w.norm_out_w = try l.tensor("decoder.norm_out.weight", &.{dim});
    w.norm_out_b = try l.tensor("decoder.norm_out.bias", &.{dim});
    w.proj_w = try l.tensor("decoder.proj_out.weight", &.{ out_ch, dim });
    w.proj_b = try l.tensor("decoder.proj_out.bias", &.{out_ch});
    var mean: [zc]f16 = undefined;
    var sd: [zc]f16 = undefined;
    for (&mean, &sd, latents_mean, latents_std) |*m, *s, mv, sv| {
        m.* = half(mv);
        s.* = half(sv);
    }
    w.mean = try st.upload(up, std.mem.sliceAsBytes(&mean));
    w.std = try st.upload(up, std.mem.sliceAsBytes(&sd));
    const inv = vt.invFreq();
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
    if (p.get("decoder.transformer_blocks.24.scale1")) |_| return error.NotTheImageDecoder else |_| {}
    return w;
}

// ---------------------------------------------------------------------------------------------------- the decoder

/// The fixed tile scratch `Decoder.init` allocates (bytes each), sized for a 256-px tile.
const TileScratch = struct { rows: u64, pq: u64, hs: u64, n: u64, qkv: u64, table: u64, vt: u64, sc: u64, att: u64, gu: u64, act: u64, proj: u64, out: u64 };

fn tileScratch() TileScratch {
    return .{
        .rows = max_p * zc * 2,
        .pq = max_p * zc * 2,
        .hs = max_s * dim * 2,
        .n = max_s * dim * 2,
        .qkv = max_s * 3 * dim * 2,
        .table = max_s * 16 * 4 * 2,
        .vt = heads * hd * max_sp * 2,
        .sc = heads * max_s * max_sp * 2,
        .att = max_s * dim * 2,
        .gu = max_s * 2 * ffn * 2,
        .act = max_s * ffn * 2,
        .proj = max_p * out_ch * 2,
        .out = 3 * tile_size * tile_size * 2,
    };
}

/// The device bytes `Decoder.init` allocates besides the weights (each buffer as allocated).
pub fn tileBytes() u64 {
    var total: u64 = 0;
    const sc = tileScratch();
    inline for (@typeInfo(TileScratch).@"struct".field_names) |name| total += @field(sc, name);
    return total;
}

/// The device scratch one `decode` of an hz x wz latent takes (freed when it returns), besides `out`: the denormalised
/// latent, the fp16 canvas, two strips of the largest vertical overlap and the left tail of the largest horizontal one.
pub fn decodeBytes(hz: u32, wz: u32) u64 {
    const hh: u64 = @as(u64, hz) * ratio;
    const ww: u64 = @as(u64, wz) * ratio;
    const z: u64 = @as(u64, zc) * hz * wz * 2;
    const canvas = 3 * hh * ww * 2;
    const strips = 2 * 3 * @as(u64, maxOverlap(@intCast(hh))) * ww * 2;
    const tail = 3 * @as(u64, tile_size) * maxOverlap(@intCast(ww)) * 2;
    return z + canvas + @max(strips, 256) + @max(tail, 256);
}

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    k: *const Ops,
    im: *const ImageOps,
    s: gpu.Stream,
    store: Store,
    w: Weights = undefined,
    /// Probe calls under the recorder names (null: none); `rec.on` follows it.
    rec: vt.Rec = .{},
    sc_numel: u64 = 0,
    t: TileScratch = undefined, // device pointers, the same fields

    /// `path`: minimax_h3_image_vae_fp16.safetensors, as published (its encoder is not loaded).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const gpu.Driver, k: *const Ops, im: *const ImageOps, s: gpu.Stream, path: []const u8, up: *Uploader) !Decoder {
        var pack = try vv.openCheckpoint(gpa, io, path);
        defer pack.close(io);
        var v: Decoder = .{ .gpa = gpa, .d = d, .k = k, .im = im, .s = s, .store = .init(d, gpa) };
        errdefer v.store.deinit();
        v.w = try loadWeights(&v.store, up, &pack);
        try v.store.done(up);
        const sizes = tileScratch();
        inline for (@typeInfo(TileScratch).@"struct".field_names) |name| @field(v.t, name) = try v.store.alloc(@field(sizes, name));
        return v;
    }

    pub fn deinit(v: *Decoder) void {
        v.store.deinit();
    }

    /// The weights' bytes (the store less the tile scratch).
    pub fn weightBytes(v: *const Decoder) u64 {
        return v.store.bytes - tileBytes();
    }

    fn rio(role: []const u8, ptr: u64, bytes: u64) Io {
        return .{ .role = role, .ptr = ptr, .bytes = bytes };
    }

    /// x [m, k] . wt [n, k]^T + bias into `out`, rounded to half; with `res` / `rscale` the addcmul residual (the video
    /// decoder's `lin`).
    fn lin(v: *Decoder, op: []const u8, x: u64, wt: u64, bias: u64, out: u64, m: u64, n: u64, kk: u64, res: u64, rscale: u64) !void {
        try v.rec.pre(op, &.{rio("x", x, m * kk * 2)});
        try v.k.gemm(v.s, .{ .a = x, .b = wt, .bias = bias, .c = out, .m = m, .n = n, .k = kk, .lda = kk, .ldb = kk, .ldc = n, .res = res, .rscale = rscale, .ldr = if (res == 0) 0 else n });
        try v.rec.post(op, &.{rio("y", out, m * n * 2)});
    }

    /// softmax(q k^T / 8) v per head from the rotated qkv [S, 3072] -> rows [S, 1024]: q k^T (fp16 out), fp32 row softmax,
    /// P v, `Ops.heads` heads a group (a head's bits depend on that head alone), then nan_to_num.
    fn attention(v: *Decoder, ss: u64, sp: u64) !void {
        const k = v.k;
        const t = &v.t;
        if (v.sc_numel != heads * ss * sp) {
            try v.d.check(v.d.api.cuMemsetD8Async(t.sc, 0, heads * ss * sp * 2, v.s.handle), "cuMemsetD8Async");
            v.sc_numel = heads * ss * sp;
        }
        try v.rec.pre("vt", &.{rio("qkv", t.qkv, ss * 3 * dim * 2)});
        try k.vt(v.s, t.qkv, t.vt, ss, sp, heads, 3 * dim);
        try v.rec.post("vt", &.{rio("y", t.vt, heads * hd * sp * 2)});
        const group: u32 = @min(k.heads, heads);
        try v.rec.pre("attn", &.{rio("qkv", t.qkv, ss * 3 * dim * 2)});
        var h0: u32 = 0;
        while (h0 < heads) : (h0 += group) {
            const g: u32 = @min(group, heads - h0);
            try k.gemm(v.s, .{ .a = t.qkv, .b = t.qkv, .c = t.sc, .m = ss, .n = ss, .k = hd, .lda = 3 * dim, .ldb = 3 * dim, .ldc = sp, .a_off = h0 * 3 * hd, .b_off = hd + h0 * 3 * hd, .sa = 3 * hd, .sb = 3 * hd, .sc = ss * sp, .batch = g });
            try k.softmax(v.s, t.sc, @as(u64, g) * ss, ss, sp, 0.125);
            try k.gemm(v.s, .{ .a = t.sc, .b = t.vt, .c = t.att, .m = ss, .n = hd, .k = sp, .lda = sp, .ldb = sp, .ldc = dim, .b_off = @as(u64, h0) * hd * sp, .c_off = h0 * hd, .sa = ss * sp, .sb = hd * sp, .sc = hd, .batch = g });
        }
        try k.nanToNum(v.s, t.att, ss * dim);
        try v.rec.post("attn", &.{rio("y", t.att, ss * dim * 2)});
    }

    /// The ViT on post_quant_conv of the crop [y0, y0 + h) x [x0, x0 + w) of the denormalised latent `z` [24, 1, hz, wz]
    /// -> `out` [3, 16 h, 16 w] fp16 (the tile scratch's `out`).
    fn decodeTile(v: *Decoder, z: u64, hz: u32, wz: u32, y0: u32, h: u32, x0: u32, w: u32) !void {
        const k = v.k;
        const t = &v.t;
        const rec = &v.rec;
        const np: u64 = @as(u64, h) * w;
        const ss = np + nsuf;
        const sp = (ss + 7) / 8 * 8;
        rec.layer = null;
        try rec.pre("gather", &.{rio("z", z, @as(u64, zc) * hz * wz * 2)});
        try k.gather(v.s, z, t.rows, zc, 1, hz, wz, 0, 1, y0, h, x0, w);
        try rec.post("gather", &.{rio("y", t.rows, np * zc * 2)});
        try v.lin("pqc", t.rows, v.w.pqc_w, v.w.pqc_b, t.pq, np, zc, zc, 0, 0);
        try v.lin("embed", t.pq, v.w.embed_w, v.w.embed_b, t.hs, np, dim, zc, 0, 0);
        try rec.pre("suffix", &.{ rio("h", t.hs, np * dim * 2), rio("reg", v.w.reg, 4 * dim * 2) });
        try k.suffix(v.s, t.hs, v.w.reg, np, dim);
        try rec.post("suffix", &.{rio("y", t.hs + np * dim * 2, nsuf * dim * 2)});
        try rec.pre("rope", &.{rio("inv_freq", v.w.inv_freq, 16)});
        try v.im.ropeTable(v.s, v.w.inv_freq, t.table, h, w, nsuf);
        try rec.post("rope", &.{rio("y", t.table, ss * 16 * 4 * 2)});
        const hb = ss * dim * 2;
        for (&v.w.l, 0..) |*l, i| {
            rec.layer = @intCast(i);
            try rec.pre("norm1", &.{rio("x", t.hs, hb)});
            try k.rmsNorm(v.s, t.hs, l.norm1, t.n, ss, dim, eps);
            try rec.post("norm1", &.{rio("y", t.n, hb)});
            try v.lin("qkv", t.n, l.qkv_w, l.qkv_b, t.qkv, ss, 3 * dim, dim, 0, 0);
            try rec.pre("rr", &.{ rio("qkv", t.qkv, ss * 3 * dim * 2), rio("table", t.table, ss * 16 * 4 * 2) });
            try v.im.rmsRope(v.s, t.qkv, t.table, ss, heads, 3 * dim, eps);
            try rec.post("rr", &.{rio("qkv", t.qkv, ss * 3 * dim * 2)});
            try v.attention(ss, sp);
            try v.lin("out", t.att, l.out_w, l.out_b, t.hs, ss, dim, dim, t.hs, l.scale1);
            try rec.pre("norm2", &.{rio("x", t.hs, hb)});
            try k.rmsNorm(v.s, t.hs, l.norm2, t.n, ss, dim, eps);
            try rec.post("norm2", &.{rio("y", t.n, hb)});
            try v.lin("w1", t.n, l.w1_w, l.w1_b, t.gu, ss, 2 * ffn, dim, 0, 0);
            try rec.pre("swi", &.{rio("x", t.gu, ss * 2 * ffn * 2)});
            try k.swiglu(v.s, t.gu, t.act, ss, ffn);
            try rec.post("swi", &.{rio("y", t.act, ss * ffn * 2)});
            try v.lin("w2", t.act, l.w2_w, l.w2_b, t.hs, ss, dim, ffn, t.hs, l.scale2);
        }
        rec.layer = null;
        // the head on the patch rows only (a row's bits depend on that row alone)
        try rec.pre("norm_out", &.{rio("x", t.hs, np * dim * 2)});
        try k.layerNorm(v.s, t.hs, v.w.norm_out_w, v.w.norm_out_b, t.n, np, dim, eps);
        try rec.post("norm_out", &.{rio("y", t.n, np * dim * 2)});
        try v.lin("proj", t.n, v.w.proj_w, v.w.proj_b, t.proj, np, out_ch, dim, 0, 0);
        try rec.pre("unshuf", &.{rio("x", t.proj, np * out_ch * 2)});
        try v.im.unshuffle(v.s, t.proj, t.out, h, w);
        try rec.post("unshuf", &.{rio("y", t.out, 3 * 256 * np * 2)});
    }

    /// `lat`: the normalized fp16 latent [24, 1, hz, wz] on the device (what ComfyUI hands the VAE after `.to(fp16)`);
    /// `out`: uint8 [16 hz, 16 wz, 3] on the device. Everything is enqueued on the stream; synchronize after.
    pub fn decode(v: *Decoder, lat: u64, hz: u32, wz: u32, out: u64) !void {
        return v.decodeTo(lat, hz, wz, out, 0);
    }

    /// As `decode`, with the fp32 pixels [16 hz, 16 wz, 3] in [0, 1] (VAEDecode's image, before SaveImage's uint8) also
    /// written to `out_f32` (0: not): the masked edit composites them (`image.composite`).
    pub fn decodeTo(v: *Decoder, lat: u64, hz: u32, wz: u32, out: u64, out_f32: u64) !void {
        const gpa = v.gpa;
        const k = v.k;
        const im = v.im;
        const t = &v.t;
        v.rec.on = v.rec.probe != null;
        const hh: u32 = hz * ratio;
        const ww: u32 = wz * ratio;
        var scratch: Store = .init(v.d, gpa);
        defer scratch.deinit();
        const zbytes: u64 = @as(u64, zc) * hz * wz * 2;
        const zd = try scratch.alloc(zbytes);
        v.rec.setPrefix("ivae", .{});
        try v.rec.pre("denorm", &.{ rio("z", lat, zbytes), rio("std", v.w.std, 48), rio("mean", v.w.mean, 48) });
        try k.denorm(v.s, lat, v.w.std, v.w.mean, zd, zc, zbytes / 2);
        try v.rec.post("denorm", &.{rio("y", zd, zbytes)});

        var sy = try vv.splitTiles(gpa, hh);
        defer sy.deinit(gpa);
        var sx = try vv.splitTiles(gpa, ww);
        defer sx.deinit(gpa);
        const oy_max = maxOverlap(hh);
        const ox_max = maxOverlap(ww);
        const canvas = try scratch.alloc(3 * @as(u64, hh) * ww * 2);
        const strips = [2]u64{ try scratch.alloc(@max(3 * @as(u64, oy_max) * ww * 2, 256)), try scratch.alloc(@max(3 * @as(u64, oy_max) * ww * 2, 256)) };
        const tail = try scratch.alloc(@max(3 * @as(u64, tile_size) * ox_max * 2, 256));

        const rows = sy.starts.len;
        const cols = sx.starts.len;
        var out_y: u32 = 0;
        for (0..rows) |i| {
            const th = sy.lens[i];
            const strip = strips[(i + 1) % 2]; // the previous row's
            const new_strip = strips[i % 2];
            var out_x: u32 = 0;
            var tile_h: u32 = th;
            for (0..cols) |j| {
                const tw = sx.lens[j];
                v.rec.setPrefix("ivae.t{d}_{d}", .{ i, j });
                try v.decodeTile(zd, hz, wz, sy.starts[i] / ratio, th / ratio, sx.starts[j] / ratio, tw / ratio);
                v.rec.layer = null;
                const tile: li.Ops.Region = .{ .ptr = t.out, .h = th, .w = tw };
                try v.rec.pre("place", &.{rio("tile", t.out, 3 * @as(u64, th) * tw * 2)});
                if (i > 0) { // blend(strip[..., x : x + tw], tile, y_overlap[i - 1], dim=-2): the strip's last e rows
                    const ey = sy.overlaps[i - 1];
                    const e = @min(ey, th);
                    try im.blend(v.s, t.out, th, tw, .{ .ptr = strip, .h = ey, .w = ww, .y = ey - e, .x = sx.starts[j] }, e, 0);
                }
                if (j > 0) { // blend(left_tail, tile, x_overlap[j - 1], dim=-1): the tail's last e columns
                    const ex = sx.overlaps[j - 1];
                    const e = @min(ex, tw);
                    try im.blend(v.s, t.out, th, tw, .{ .ptr = tail, .h = th, .w = ex, .x = ex - e }, e, 1);
                }
                var cw = tw;
                if (j + 1 < cols) { // left_tail = tile[..., -x_overlap[j]:]; the tile loses it
                    const ox = sx.overlaps[j];
                    cw = tw - ox;
                    try im.copy(v.s, .{ .ptr = t.out, .h = th, .w = tw, .x = cw }, .{ .ptr = tail, .h = th, .w = ox }, th, ox);
                }
                var ch = th;
                if (i + 1 < rows) { // new_strip[..., out_x : out_x + cw] = tile[..., -y_overlap[i]:, :]; the tile loses it
                    const oy = sy.overlaps[i];
                    ch = th - oy;
                    try im.copy(v.s, .{ .ptr = t.out, .h = th, .w = tw, .y = ch }, .{ .ptr = new_strip, .h = oy, .w = ww, .x = out_x }, oy, cw);
                }
                try im.copy(v.s, tile, .{ .ptr = canvas, .h = hh, .w = ww, .y = out_y, .x = out_x }, ch, cw);
                try v.rec.post("place", &.{rio("canvas", canvas, 3 * @as(u64, hh) * ww * 2)});
                tile_h = ch;
                out_x += cw;
            }
            if (out_x != ww) return error.TilePlanMismatch;
            out_y += tile_h;
        }
        if (out_y != hh) return error.TilePlanMismatch;
        v.rec.setPrefix("ivae", .{});
        try v.rec.pre("finalize", &.{rio("canvas", canvas, 3 * @as(u64, hh) * ww * 2)});
        try k.finalize(v.s, .{
            .b = canvas,
            .fb = 1,
            .b0 = 0,
            .ext = 0,
            .copy = 1,
            .h = hh,
            .w = ww,
            .sm = .{ f16Const(imagenet_std[0]), f16Const(imagenet_std[1]), f16Const(imagenet_std[2]), f16Const(imagenet_mean[0]), f16Const(imagenet_mean[1]), f16Const(imagenet_mean[2]) },
            .out_u8 = out,
            .pos = 0,
            .out_f32 = out_f32,
        });
        try v.rec.post("finalize", &.{rio("y", out, 3 * @as(u64, hh) * ww)});
    }
};

test "the tile plan is ComfyUI's split_tiles at every side 16 .. 4096 (vae_image_tiles.json, from vae.py itself)" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("vae_image_tiles.json"), .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqual(@as(i64, tile_size), root.get("tile_size").?.integer);
    try std.testing.expectEqual(@as(i64, tile_overlap_min), root.get("tile_overlap_min").?.integer);
    try std.testing.expectEqual(@as(i64, ratio), root.get("vae_ratio").?.integer);
    var n: usize = 0;
    for (root.get("plans").?.array.items) |pv| {
        const p = pv.object;
        const side: u32 = @intCast(p.get("side").?.integer);
        const sp = try vv.splitTiles(gpa, side);
        defer sp.deinit(gpa);
        inline for (.{ "starts", "lens", "overlaps" }) |f| {
            const want = p.get(f).?.array.items;
            const got = @field(sp, f);
            try std.testing.expectEqual(want.len, got.len);
            for (want, got) |wv, g| try std.testing.expectEqual(wv.integer, g);
        }
        var most: u32 = 0;
        for (sp.overlaps) |o| most = @max(most, o);
        try std.testing.expectEqual(most, maxOverlap(side));
        // the plan covers the side: the tiles less their trailing overlaps
        var covered: u32 = 0;
        for (sp.lens, 0..) |l, i| covered += l - (if (i < sp.overlaps.len) sp.overlaps[i] else 0);
        try std.testing.expectEqual(side, covered);
        n += 1;
    }
    try std.testing.expectEqual(@as(usize, 256), n);
}

test "the blend's weights and values are ComfyUI's blend() on fp16 tensors, for every overlap the plans use" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("vae_image_tiles.json"), .{});
    defer parsed.deinit();
    const blends = parsed.value.object.get("blends").?.array.items;
    try std.testing.expect(blends.len >= 12);
    for (blends) |bv| {
        const b = bv.object;
        const e: u32 = @intCast(b.get("extent").?.integer);
        const wa = b.get("wa").?.array.items;
        const wb = b.get("wb").?.array.items;
        const ra = b.get("a").?.array.items;
        const rb = b.get("b").?.array.items;
        const out = b.get("out").?.array.items;
        try std.testing.expectEqual(@as(usize, e), out.len);
        for (0..e) |p| {
            const pp: u32 = @intCast(p);
            const h = struct {
                fn f(v: std.json.Value) f16 {
                    return @bitCast(@as(u16, @intCast(v.integer)));
                }
            }.f;
            try std.testing.expectEqual(@as(u16, @bitCast(h(wa[p]))), @as(u16, @bitCast(blendValue(1, 0, pp, e))));
            try std.testing.expectEqual(@as(u16, @bitCast(h(wb[p]))), @as(u16, @bitCast(blendValue(0, 1, pp, e))));
            try std.testing.expectEqual(@as(u16, @bitCast(h(out[p]))), @as(u16, @bitCast(blendValue(h(ra[p]), h(rb[p]), pp, e))));
        }
    }
}

test "1024 px: five tiles a side at 0, 192, 384, 576, 768; the scratch a decode takes" {
    const sp = try vv.splitTiles(std.testing.allocator, 1024);
    defer sp.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 0, 192, 384, 576, 768 }, sp.starts);
    try std.testing.expectEqual(@as(u32, 64), maxOverlap(1024));
    try std.testing.expectEqual(@as(u32, 0), maxOverlap(256));
    try std.testing.expectEqual(@as(u32, 240), maxOverlap(272));
    const b = decodeBytes(64, 64);
    try std.testing.expectEqual(@as(u64, 24 * 64 * 64 * 2 + 3 * 1024 * 1024 * 2 + 2 * 3 * 64 * 1024 * 2 + 3 * 256 * 64 * 2), b);
    try std.testing.expect(tileBytes() < 64 << 20);
}

test "the latent statistics as the fp16 module holds them: through fp32, then the nearest half" {
    // 1.68317747116088865 is not an fp32: torch.tensor rounds it first (to 1.6831774711608887), then .to(fp16)
    try std.testing.expectEqual(@as(u16, 0x3EBC), @as(u16, @bitCast(half(latents_std[2]))));
    try std.testing.expectApproxEqAbs(@as(f32, 0.858090341091156), @as(f32, half(latents_mean[0])), 5e-4);
    for (latents_std) |s| try std.testing.expect(half(s) > 0);
}

test "the decoder's code type-checks (no GPU here)" {
    _ = &Decoder.init;
    _ = &Decoder.decode;
}
