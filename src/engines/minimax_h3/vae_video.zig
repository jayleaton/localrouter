//! MiniMax H3's video VAE decoder (ComfyUI's `MiniMaxH3VideoVAE.decode`, the fp16 checkpoint) as the twin
//! (`stk_twin/h3/vae_video.py`) runs it: Zig launches the same kernels (gemm_f16.cu, vae_video.cu) in the same order with
//! the same arguments, so the frames are bit-exact. `decode`: z = z * std + mean (fp16), clips of 7 latent frames every 5
//! (the last pads by repeating the final latent frame), each clip tiled spatially in 256 px tiles (overlap >= 64 px) that
//! the ViT3D (`vae_video_tile.zig`) decodes one at a time, blended in fp16 into a clip canvas; a clip's two halves are
//! blended in time (5 frames), finalised and written as uint8. The recorder names, attrs-free roles and the per-tile
//! recording selection are the twin's (tools/twin/VVAE-PORT.md, "Recorder names"), so a twin capture replays.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const tile_mod = @import("vae_video_tile.zig");
const Pack = qi.pack.Pack;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;

pub const Ops = tile_mod.Ops;
pub const Probe = tile_mod.Probe;
pub const Io = tile_mod.Io;
pub const Tile = tile_mod.Tile;
pub const Lat = tile_mod.Lat;

pub const ratio = 16; // pixels a latent cell
pub const ratio_t = 4; // frames a latent frame
pub const tile_size = 256;
pub const tile_overlap_min = 64;
const clip_length = 17;
const token_drop = 3;
const frame_pre_padding = (ratio_t - clip_length % ratio_t) % ratio_t; // 3
const tokens_chunk = (clip_length + ratio_t - 1) / ratio_t; // 5
const token_overlap = (tokens_chunk - token_drop % tokens_chunk) % tokens_chunk; // 2
const frame_overlap = @max(token_overlap * ratio_t - frame_pre_padding, 0); // 5
const imagenet_mean = [3]f32{ 0.485, 0.456, 0.406 };
const imagenet_std = [3]f32{ 0.229, 0.224, 0.225 };

// ---------------------------------------------------------------------------------------------------- the plan (host)

/// `split_tiles`: starts, lengths and overlaps (pixels) of the tiles along one dimension.
pub const Split = struct {
    starts: []u32,
    lens: []u32,
    overlaps: []u32,

    pub fn deinit(s: Split, gpa: std.mem.Allocator) void {
        gpa.free(s.starts);
        gpa.free(s.lens);
        gpa.free(s.overlaps);
    }
};

pub fn splitTiles(gpa: std.mem.Allocator, len: u32) !Split {
    if (tile_size >= len) {
        const st = try gpa.alloc(u32, 1);
        const ln = try gpa.alloc(u32, 1);
        st[0] = 0;
        ln[0] = len;
        return .{ .starts = st, .lens = ln, .overlaps = try gpa.alloc(u32, 0) };
    }
    var n: u32 = (len + tile_size - 1) / tile_size;
    while (@as(i64, tile_size) * n - @as(i64, tile_overlap_min) * (n - 1) - len < 0) n += 1;
    const remaining: u32 = @intCast(@as(i64, tile_size) * n - @as(i64, tile_overlap_min) * (n - 1) - len);
    const ov = try gpa.alloc(u32, n - 1);
    errdefer gpa.free(ov);
    @memset(ov, tile_overlap_min);
    for (0..remaining / ratio) |i| ov[i % (n - 1)] += ratio;
    const st = try gpa.alloc(u32, n);
    errdefer gpa.free(st);
    const ln = try gpa.alloc(u32, n);
    st[0] = 0;
    for (1..n) |i| st[i] = st[i - 1] + tile_size - ov[i - 1];
    @memset(ln, tile_size);
    return .{ .starts = st, .lens = ln, .overlaps = ov };
}

pub const Chunks = struct { pad: u32, num: u32 };

/// `_decode_temporal_chunks`: the latent frames of padding and the clip count for `z_len` latent frames.
pub fn temporalChunks(z_len: u32) Chunks {
    var pseudo = z_len + token_drop;
    var pad = (tokens_chunk - pseudo % tokens_chunk) % tokens_chunk;
    pseudo += pad;
    var num: u32 = pseudo / tokens_chunk - 1;
    if (num < 1) {
        pad += tokens_chunk;
        num += 1;
    }
    return .{ .pad = pad, .num = num };
}

fn padFrames(z_len: u32, pad: u32) i64 {
    if (pad == 0) return 0;
    const intra_tail = clip_length % ratio_t;
    if (intra_tail == 0) return @as(i64, pad) * ratio_t;
    const before = z_len - pad;
    var total: i64 = 0;
    for (0..pad) |k| total += if ((before + k) % tokens_chunk == 0) intra_tail else ratio_t;
    return total;
}

fn framePlan(z_len: u32, num: u32, pad: u32) i64 {
    const chunk_dec: i64 = tokens_chunk * ratio_t;
    var total: i64 = 0;
    var overlap: i64 = 0;
    for (0..num) |i| {
        const t0: i64 = @as(i64, @intCast(i)) * tokens_chunk;
        const t1 = t0 + tokens_chunk + token_overlap;
        const clip_frames = @max(0, @min(t1, z_len) - @min(t0, z_len)) * ratio_t;
        for (0..2) |j| {
            const f0 = @as(i64, @intCast(j)) * chunk_dec;
            const f1 = @min(f0 + chunk_dec, clip_frames);
            const frames = @max(0, f1 - f0 - frame_pre_padding);
            if (j == 0) total += frames else overlap = frames;
        }
    }
    return total + overlap - padFrames(z_len, pad);
}

/// `decode_output_shape`'s frame count for `z_len` latent frames (17 k + 5 for z_len = 5 k + 2).
pub fn outputFrames(z_len: u32) u32 {
    if (z_len == 1) return 1;
    const c = temporalChunks(z_len);
    return @intCast(framePlan(z_len + c.pad, c.num, c.pad));
}

/// ImageNet constant as the decoder holds it: fp32 tensor -> module.to(fp16) -> fp32.
fn f16Const(v: f32) f32 {
    const h: f16 = @floatCast(v);
    return h;
}

// ---------------------------------------------------------------------------------------------------- the checkpoint

/// A plain safetensors file as `Pack.openFile` reads it, but tolerant of the rank-5 convolution weights
/// (`post_quant_conv.weight` is [24, 24, 1, 1, 1]: trailing unit dims are dropped) and keeping only fp16 tensors.
/// The int8 ConvRot checkpoint (`*.comfy_quant` keys) is refused, as the twin does.
pub fn openCheckpoint(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Pack {
    var p: Pack = .{ .arena = .init(gpa), .file = undefined, .data_start = 0, .data_len = 0 };
    errdefer p.arena.deinit();
    const a = p.arena.allocator();
    p.file = try std.Io.Dir.cwd().openFile(io, path, .{});
    errdefer p.file.close(io);
    var len_buf: [8]u8 = undefined;
    if (try p.file.readPositionalAll(io, &len_buf, 0) != 8) return error.BadPack;
    const hlen = std.mem.readInt(u64, &len_buf, .little);
    if (hlen > 64 << 20) return error.BadPack;
    const header = try a.alloc(u8, hlen);
    if (try p.file.readPositionalAll(io, header, 8) != hlen) return error.BadPack;
    p.data_start = 8 + hlen;
    p.data_len = (try p.file.stat(io)).size - p.data_start;
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, header, .{});
    var it = v.object.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) continue;
        if (std.mem.endsWith(u8, key, ".comfy_quant")) return error.Int8CheckpointUnsupported;
        const o = kv.value_ptr.object;
        if (!std.mem.eql(u8, o.get("dtype").?.string, "F16")) continue;
        const shape = o.get("shape").?.array.items;
        const offs = o.get("data_offsets").?.array.items;
        if (offs.len != 2) return error.BadPack;
        var rank: usize = shape.len;
        if (rank > 4) { // [24, 24, 1, 1, 1] -> [24, 24]
            while (rank > 2 and shape[rank - 1].integer == 1) rank -= 1;
            if (rank > 4) continue; // the encoder's 3-D convolutions [co, ci, 3, 3, 3]: the decoder never reads them
        }
        var t: qi.pack.Tensor = .{ .dtype = .f16, .rank = @intCast(rank), .begin = @intCast(offs[0].integer), .end = @intCast(offs[1].integer) };
        var n: usize = 2;
        for (shape) |s| n *= @intCast(s.integer);
        for (shape[0..rank], 0..) |s, i| t.shape[i] = @intCast(s.integer);
        if (t.end < t.begin or t.end > p.data_len or t.len() != n) return error.BadPack;
        try p.tensors.put(a, key, t);
    }
    return p;
}

// ---------------------------------------------------------------------------------------------------- the decoder

/// Where a carried half-clip lives: a canvas, its frame count, the first frame and the frames carried.
const Carry = struct { canvas: u64, frames: u32, b0: u32, nb: u32 };

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Ops,
    s: cuda.Stream,
    tile: Tile,
    /// Probe calls under the twin's names (null: none).
    probe: ?Probe = null,
    /// The tiles `(chunk, row, col)` whose ops are recorded (the twin's `rec` callback; clip-level ops, row = col = -1,
    /// and the denorm, chunk = -1, are always recorded when a selection is set); null: every tile.
    select: ?[]const [3]i32 = null,
    /// The block indices whose ops are recorded (null: all 36).
    layers: ?[]const u32 = null,

    /// `path`: minimax_h3_video_vae_fp16.safetensors (the int8 file is refused). Loads the decoder, post_quant_conv and the
    /// latent statistics (4.85 GB of fp16).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const Ops, s: cuda.Stream, path: []const u8, up: *Uploader) !Decoder {
        var pack = try openCheckpoint(gpa, io, path);
        defer pack.close(io);
        return .{ .gpa = gpa, .d = d, .k = k, .s = s, .tile = try Tile.init(gpa, d, k, s, &pack, up) };
    }

    pub fn deinit(v: *Decoder) void {
        v.tile.deinit();
    }

    pub fn weightBytes(v: *const Decoder) u64 {
        return v.tile.store.bytes;
    }

    fn isOn(v: *const Decoder, c: i32, r: i32, col: i32) bool {
        const sel = v.select orelse return true;
        if (r < 0 and col < 0) return true;
        for (sel) |e| if (e[0] == c and e[1] == r and e[2] == col) return true;
        return false;
    }

    /// `lat`: fp16 latents [24, tz, hz, wz] on the device (what ComfyUI hands the VAE after `.to(fp16)`); `out`: uint8
    /// [F, 16 hz, 16 wz, 3] on the device, F = `outputFrames(tz)`. Everything is enqueued on the stream; synchronize after.
    pub fn decode(v: *Decoder, lat: u64, tz: u32, hz: u32, wz: u32, out: u64) !void {
        const gpa = v.gpa;
        const t = &v.tile;
        t.rec.probe = v.probe;
        t.rec.keep = v.layers;
        const hh: u32 = hz * ratio;
        const ww: u32 = wz * ratio;
        const frames = outputFrames(tz);
        var scratch: Store = .init(v.d, gpa);
        defer scratch.deinit();
        const zbytes: u64 = @as(u64, tile_mod.zc) * tz * hz * wz * 2;
        const zd = try scratch.alloc(zbytes);
        t.rec.setPrefix("vvae", .{});
        t.rec.on = v.isOn(-1, -1, -1);
        try t.rec.pre("denorm", &.{ .{ .role = "z", .ptr = lat, .bytes = zbytes }, .{ .role = "std", .ptr = t.w.std, .bytes = 48 }, .{ .role = "mean", .ptr = t.w.mean, .bytes = 48 } });
        try v.k.denorm(v.s, lat, t.w.std, t.w.mean, zd, tile_mod.zc, zbytes / 2);
        try t.rec.post("denorm", &.{.{ .role = "y", .ptr = zd, .bytes = zbytes }});
        try v.d.check(v.d.api.cuMemsetD8Async(out, 0, @as(u64, frames) * hh * ww * 3, v.s.handle), "cuMemsetD8Async");

        var sy = try splitTiles(gpa, hh);
        defer sy.deinit(gpa);
        var sx = try splitTiles(gpa, ww);
        defer sx.deinit(gpa);
        const tn_max: u32 = if (tz == 1) 1 else tile_mod.max_tn;
        const cbytes: u64 = 3 * ratio_t * @as(u64, tn_max) * hh * ww * 2;
        var run: Run = .{
            .v = v,
            .lat = .{ .ptr = zd, .tz = tz, .hz = hz, .wz = wz },
            .sy = sy,
            .sx = sx,
            .hh = hh,
            .ww = ww,
            .frames = frames,
            .out = out,
            .cv = .{ try scratch.alloc(cbytes), try scratch.alloc(cbytes) },
            .tbufs = try gpa.alloc(u64, 2 * sx.starts.len),
            .tbytes = 3 * ratio_t * @as(u64, tn_max) * tile_size * tile_size * 2,
        };
        defer gpa.free(run.tbufs);
        for (run.tbufs) |*b| b.* = try scratch.alloc(run.tbytes);
        if (v.probe != null) run.region = try scratch.alloc(run.tbytes);

        if (tz == 1) {
            try run.clip(0, 0, 1, run.cv[0]);
            t.rec.setPrefix("vvae.c0", .{});
            t.rec.on = v.isOn(0, -1, -1);
            try run.writePart("part0", .{ .canvas = run.cv[0], .frames = 4, .b0 = 3, .nb = 1 }, null);
        } else {
            const ch = temporalChunks(tz);
            const tpad = tz + ch.pad;
            const chunk_dec: u32 = tokens_chunk * ratio_t;
            var carry: ?Carry = null;
            for (0..ch.num) |ci_| {
                const ci: u32 = @intCast(ci_);
                const t0 = ci * tokens_chunk;
                const t1 = t0 + tokens_chunk + token_overlap;
                const tn = @min(t1, tpad) - t0;
                const cv = run.cv[ci & 1];
                try run.clip(ci, t0, tn, cv);
                t.rec.setPrefix("vvae.c{d}", .{ci});
                t.rec.on = v.isOn(@intCast(ci), -1, -1);
                for (0..2) |j_| {
                    const j: u32 = @intCast(j_);
                    const f0 = j * chunk_dec;
                    const f1 = @min(f0 + chunk_dec, tn * ratio_t);
                    const b0 = f0 + frame_pre_padding;
                    const nb = if (f1 > b0) f1 - b0 else 0;
                    const cur: Carry = .{ .canvas = cv, .frames = tn * ratio_t, .b0 = b0, .nb = nb };
                    if (j == 0) {
                        try run.writePart("part0", cur, carry);
                        carry = null;
                    } else carry = cur;
                }
                if (ci == ch.num - 1) if (carry) |c| {
                    try run.writePart("part1", c, null);
                    carry = null;
                };
            }
        }
        if (run.pos != frames) return error.FrameCountMismatch;
    }
};

/// The state of one `decode`: the plan, the canvases (two: the previous clip's carried frames stay readable), the raw
/// tiles of the current and previous tile row, and the write position.
const Run = struct {
    v: *Decoder,
    lat: Lat,
    sy: Split,
    sx: Split,
    hh: u32,
    ww: u32,
    frames: u32,
    out: u64,
    cv: [2]u64,
    tbufs: []u64, // [row parity][column]
    tbytes: u64,
    region: u64 = 0, // contiguous copy of a placed region, for the probe
    pos: u32 = 0,

    /// `tiled_decode` of the clip of `tn` latent frames from `t0` into the (zeroed) canvas `cv`: every tile decoded alone,
    /// blended with the raw tiles above and to the left, cropped by its trailing overlaps.
    fn clip(r: *Run, ci: u32, t0: u32, tn: u32, cv: u64) !void {
        const v = r.v;
        const t = &v.tile;
        const nx = r.sx.starts.len;
        const ny = r.sy.starts.len;
        const fr: u32 = tn * ratio_t;
        try v.d.check(v.d.api.cuMemsetD8Async(cv, 0, 3 * @as(u64, fr) * r.hh * r.ww * 2, v.s.handle), "cuMemsetD8Async");
        var out_y: u32 = 0;
        for (0..ny) |i| {
            var out_x: u32 = 0;
            var oh: u32 = 0;
            const il = r.sy.lens[i];
            for (0..nx) |j| {
                const jl = r.sx.lens[j];
                t.rec.setPrefix("vvae.c{d}.t{d}_{d}", .{ ci, i, j });
                t.rec.on = v.isOn(@intCast(ci), @intCast(i), @intCast(j));
                const cur = r.tbufs[(i & 1) * nx + j];
                try t.decode(r.lat, t0, tn, r.sy.starts[i] / ratio, il / ratio, jl / ratio, r.sx.starts[j] / ratio, cur);
                var p: Ops.Place = .{ .b = cur, .frames = fr, .th = il, .tw = jl, .canvas = cv, .hc = r.hh, .wc = r.ww, .oy = out_y, .ox = out_x, .oh = 0, .ow = 0 };
                if (i > 0) {
                    p.ytail = r.tbufs[((i + 1) & 1) * nx + j];
                    p.tha = r.sy.lens[i - 1];
                    p.twa = jl;
                    p.ey = r.sy.overlaps[i - 1];
                }
                if (j > 0) {
                    p.ltail = r.tbufs[(i & 1) * nx + j - 1];
                    p.thl = il;
                    p.twl = r.sx.lens[j - 1];
                    p.ex = r.sx.overlaps[j - 1];
                }
                p.oh = il - (if (i < ny - 1) r.sy.overlaps[i] else 0);
                p.ow = jl - (if (j < nx - 1) r.sx.overlaps[j] else 0);
                const tb = 3 * @as(u64, fr) * il * jl * 2;
                var ins: [3]Io = undefined;
                var n: usize = 0;
                ins[n] = .{ .role = "b", .ptr = cur, .bytes = tb };
                n += 1;
                if (p.ytail != 0) {
                    ins[n] = .{ .role = "ytail", .ptr = p.ytail, .bytes = 3 * @as(u64, fr) * p.tha * p.twa * 2 };
                    n += 1;
                }
                if (p.ltail != 0) {
                    ins[n] = .{ .role = "ltail", .ptr = p.ltail, .bytes = 3 * @as(u64, fr) * p.thl * p.twl * 2 };
                    n += 1;
                }
                try t.rec.pre("place", ins[0..n]);
                try v.k.place(v.s, p);
                if (t.rec.on and v.probe != null) {
                    try r.copyRegion(cv, fr, out_y, out_x, p.oh, p.ow);
                    try t.rec.post("place", &.{.{ .role = "y", .ptr = r.region, .bytes = 3 * @as(u64, fr) * p.oh * p.ow * 2 }});
                }
                out_x += p.ow;
                oh = p.oh;
            }
            out_y += oh;
        }
    }

    /// canvas[:, :, oy : oy + oh, ox : ox + ow] contiguous into `region` (the twin records the strided view contiguously).
    fn copyRegion(r: *Run, cv: u64, fr: u32, oy: u32, ox: u32, oh: u32, ow: u32) !void {
        const v = r.v;
        try v.s.synchronize();
        var dst = r.region;
        for (0..3 * fr) |cf| for (0..oh) |row| {
            const src = cv + ((@as(u64, cf) * r.hh + oy + row) * r.ww + ox) * 2;
            try v.d.check(v.d.api.cuMemcpyDtoD_v2(dst, src, @as(u64, ow) * 2), "cuMemcpyDtoD");
            dst += @as(u64, ow) * 2;
        };
    }

    /// `write_part`: the carried overlap `a` blended into the first frames of the part `b`, finalised, written at `pos`
    /// (truncated to the frames the plan allots). The recorder's prefix is "vvae.c{c}".
    fn writePart(r: *Run, op: []const u8, b: Carry, a: ?Carry) !void {
        if (b.nb == 0) return;
        const v = r.v;
        const t = &v.tile;
        const copy = @min(b.nb, r.frames -| r.pos);
        const ext: u32 = if (a) |c| @min(c.nb, b.nb, frame_overlap) else 0;
        if (copy > 0) {
            const plane: u64 = @as(u64, r.hh) * r.ww * 2;
            var ins: [2]Io = undefined;
            ins[0] = .{ .role = "b", .ptr = b.canvas, .bytes = 3 * b.nb * plane, .contiguous = false };
            var n: usize = 1;
            if (ext > 0) {
                ins[1] = .{ .role = "a", .ptr = a.?.canvas, .bytes = 3 * ext * plane, .contiguous = false };
                n = 2;
            }
            try t.rec.pre(op, ins[0..n]);
            try v.k.finalize(v.s, .{
                .a = if (ext > 0) a.?.canvas else 0,
                .fa = if (ext > 0) a.?.frames else 0,
                .a0 = if (a) |c| c.b0 else 0,
                .b = b.canvas,
                .fb = b.frames,
                .b0 = b.b0,
                .ext = ext,
                .copy = copy,
                .h = r.hh,
                .w = r.ww,
                .sm = .{ f16Const(imagenet_std[0]), f16Const(imagenet_std[1]), f16Const(imagenet_std[2]), f16Const(imagenet_mean[0]), f16Const(imagenet_mean[1]), f16Const(imagenet_mean[2]) },
                .out_u8 = r.out,
                .pos = r.pos,
            });
            const fb: u64 = @as(u64, r.hh) * r.ww * 3;
            try t.rec.post(op, &.{.{ .role = "y", .ptr = r.out + r.pos * fb, .bytes = copy * fb }});
        }
        r.pos += copy;
    }
};

// ---------------------------------------------------------------------------------------------------- tests

test "tile tables of VVAE-PORT.md section 3" {
    const gpa = std.testing.allocator;
    const cases = .{
        .{ 448, &[_]u32{ 0, 192 }, &[_]u32{64} },
        .{ 768, &[_]u32{ 0, 160, 336, 512 }, &[_]u32{ 96, 80, 80 } },
        .{ 1344, &[_]u32{ 0, 176, 352, 528, 704, 896, 1088 }, &[_]u32{ 80, 80, 80, 80, 64, 64 } },
    };
    inline for (cases) |c| {
        const s = try splitTiles(gpa, c[0]);
        defer s.deinit(gpa);
        try std.testing.expectEqualSlices(u32, c[1], s.starts);
        try std.testing.expectEqualSlices(u32, c[2], s.overlaps);
    }
    const one = try splitTiles(gpa, 192);
    defer one.deinit(gpa);
    try std.testing.expectEqualSlices(u32, &.{192}, one.lens);
}

test "clip plan and frame counts" {
    try std.testing.expectEqual(@as(u32, 7), temporalChunks(37).num);
    try std.testing.expectEqual(@as(u32, 0), temporalChunks(37).pad);
    try std.testing.expectEqual(@as(u32, 124), outputFrames(37));
    try std.testing.expectEqual(@as(u32, 22), outputFrames(7));
    try std.testing.expectEqual(@as(u32, 1), outputFrames(1));
    try std.testing.expectEqual(@as(u32, 1), temporalChunks(7).num);
}

test "the decode path type-checks (lazy analysis)" {
    _ = Decoder.init;
    _ = Decoder.decode;
    _ = Decoder.deinit;
    _ = openCheckpoint;
}

test "constants as the decoder holds them: fp16-rounded" {
    try std.testing.expectEqual(@as(f32, @floatCast(@as(f16, 0.229))), f16Const(0.229));
    try std.testing.expect(f16Const(0.229) != 0.229);
}
