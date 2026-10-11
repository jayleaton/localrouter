//! Kandinsky-6 video VAE decode: postprocess_video (pinned upstream kandinsky-6 eac06e39 core/algo/postprocess_video.py)
//! from the DiT's bf16 visual latent to uint8 RGB frames, through AutoencoderKLHunyuanVideo.decode with the tile plan of
//! tiles.zig. No RNG, no muxing.
//!
//!   latent  bf16 [frames][lh][lw][16], the packed THWC layout the DiT produces: all 31 frames for T2V and for I2V under
//!           the default pretrain scheme (and explicit i2v); no frame is appended or dropped
//!   rgb     uint8 [(frames - 1) * 4 + 1][8 lh][8 lw][3], frame-major interleaved RGB (rawvideo rgb24 per frame); the
//!           reference's (bs, 3, T, H, W) uint8 tensor permuted to (T, H, W, 3), same values
//!
//! Per temporal tile (5 latent frames -> 17 frames at the native size): post_quant_conv, decoder (conv_in, mid block,
//! four up blocks, norm/SiLU, conv_out) into an fp16 tile, the reference's in-place blends, then the kept frames are
//! converted to uint8 directly into the caller's output. Tile outputs that later blends read stay in scratch.

const std = @import("std");
const launch = @import("launch.zig");
const tiles = @import("tiles.zig");

const View = launch.View;
const ConvW = launch.ConvW;

/// HunyuanVideoUpBlock3D: three resnets, then (blocks 0..2) the upsampler's causal conv on a nearest-upsampled input.
pub const UpW = struct { resnets: [3]launch.ResnetW, upsampler: ?ConvW };

pub const Decoder = struct {
    post_quant: ConvW,
    conv_in: ConvW,
    mid: launch.MidW,
    up: [4]UpW,
    norm_out: launch.NormW,
    conv_out: ConvW,
};

/// Upsample factors (time, space) of up blocks 0..2: time_compression_ratio 4 upsamples time in blocks 1 and 2 only.
pub const up_factors = [3][2]u32{ .{ 1, 2 }, .{ 2, 2 }, .{ 2, 2 } };
pub const scaling_factor: f64 = 0.476986;

/// float(1.0 / 0.476986): torch 2.14.1's reciprocal of a CPU scalar divisor, taken in double and cast to the opmath
/// type (BinaryDivTrueKernel.cu:46). For this factor it equals 1.0f / float(0.476986) bit for bit (0x40062d04).
pub fn invScale() f32 {
    return @floatCast(1.0 / scaling_factor);
}

/// Largest fp16 activation (elements) of one decoder pass over a latent tile of t x h x w: the up blocks' channel
/// counts at their resolutions, with the upsampled convolution outputs.
pub fn maxActivation(t: u32, h: u32, w: u32) u64 {
    var best: u64 = @as(u64, t) * h * w * 512;
    var tt: u64 = t;
    var hh: u64 = h;
    var ww: u64 = w;
    const ch = [4][2]u64{ .{ 512, 512 }, .{ 512, 512 }, .{ 512, 256 }, .{ 256, 128 } };
    for (ch, 0..) |io, i| {
        best = @max(best, tt * hh * ww * @max(io[0], io[1]));
        if (i < 3) {
            if (up_factors[i][0] == 2) tt = 2 * tt - 1;
            hh *= 2;
            ww *= 2;
            best = @max(best, tt * hh * ww * io[1]);
        }
    }
    return best;
}

/// The decoder's scratch, carved from the caller's buffer in this order. `bytes` sizes it (measure mode).
pub const Layout = struct {
    latent16: u64,
    big: [2]u64,
    work: launch.Work,
    tile_out: u64, // spatial tiles of one temporal tile ([rows * cols] x [F][8 tlh][8 tlw][3]), when tiled
    tile_elems: u64,
    frames_buf: [2]u64, // the current and previous temporal tile, [F][H][W][3]

    pub fn init(arena: *launch.Arena, plan: tiles.DecodePlan) launch.Error!Layout {
        const g = plan.grid;
        const ft = tiles.sampleFrames(plan.tile_lat_frames);
        const act = maxActivation(plan.tile_lat_frames, g.tile_lh, g.tile_lw);
        const sh: u64 = g.sample_h;
        const sw: u64 = g.sample_w;
        const tile_elems: u64 = @as(u64, ft) * g.tile_lh * tiles.spatial_ratio * g.tile_lw * tiles.spatial_ratio * 3;
        var l: Layout = undefined;
        l.latent16 = try arena.take(@as(u64, plan.latent_frames) * plan.lh * plan.lw * tiles.latent_channels * 2);
        l.big = .{ try arena.take(act * 2), try arena.take(act * 2) };
        const positions = @as(u64, ft) * g.tile_lh * g.tile_lw * 64;
        l.work = try launch.Work.init(arena, act, positions, plan.tile_lat_frames * g.tile_lh * g.tile_lw, 512);
        l.tile_elems = tile_elems;
        l.tile_out = if (g.tiled()) try arena.take(tile_elems * g.count() * 2) else 0;
        l.frames_buf = .{ try arena.take(@as(u64, ft) * sh * sw * 3 * 2), try arena.take(@as(u64, ft) * sh * sw * 3 * 2) };
        return l;
    }
};

/// Scratch bytes `decode` needs for `plan`.
pub fn scratchBytes(plan: tiles.DecodePlan) u64 {
    var arena: launch.Arena = .{ .base = 0, .cap = 0, .measure = true };
    _ = Layout.init(&arena, plan) catch unreachable;
    return arena.peak;
}

pub const LatentDtype = launch.LatentDtype;

pub const Io = struct {
    latent: u64,
    latent_dtype: LatentDtype,
    rgb: u64,
    scratch: u64,
    scratch_bytes: u64,
};

/// One decoder pass: z (16 channels, any strides) -> out fp16 dense [4 (t - 1) + 1][8 h][8 w][3].
pub fn decodeTile(l: launch.Launcher, w: *const Decoder, lay: Layout, z: View, out: u64) !void {
    const work = lay.work;
    var cur = lay.big[0];
    var spare = lay.big[1];
    const zq = View.dense(spare, z.t, z.h, z.w, z.c);
    if (zq.elems() > work.act_elems) return error.BufferTooSmall;
    try l.run(try launch.conv(z, w.post_quant, .{ .causal_pad = false }, zq.ptr));
    try l.mark("post_quant_conv", .{}, zq, false);
    try l.run(try launch.conv(zq, w.conv_in, .{}, cur));
    var x = View.dense(cur, z.t, z.h, z.w, w.conv_in.co);
    try l.mark("decoder.conv_in", .{}, x, false);
    x = try launch.midBlock(l, work, x, w.mid, spare, "decoder");
    if (x.ptr == spare) std.mem.swap(u64, &cur, &spare);
    var name: [64]u8 = undefined;
    for (w.up, 0..) |ub, i| {
        for (ub.resnets, 0..) |r, j| {
            x = try launch.resnet(l, work, x, r, spare, try std.fmt.bufPrint(&name, "decoder.up_blocks.{d}.resnets.{d}", .{ i, j }));
            std.mem.swap(u64, &cur, &spare);
        }
        if (ub.upsampler) |u| {
            const f = up_factors[i];
            const o = launch.convOut(x, u, .{ .up_t = f[0], .up_hw = f[1] });
            if (@as(u64, o[0]) * o[1] * o[2] * u.co > work.act_elems) return error.BufferTooSmall;
            try l.run(try launch.conv(x, u, .{ .up_t = f[0], .up_hw = f[1] }, spare));
            x = View.dense(spare, o[0], o[1], o[2], u.co);
            try l.mark("decoder.up_blocks.{d}.upsamplers.0", .{i}, x, false);
            std.mem.swap(u64, &cur, &spare);
        }
    }
    try launch.groupNorm(l, work, x, w.norm_out, true, work.a);
    const act = View.dense(work.a, x.t, x.h, x.w, x.c);
    try l.mark("decoder.conv_act", .{}, act, false);
    try l.run(try launch.conv(act, w.conv_out, .{}, out));
    try l.mark("decoder.conv_out", .{}, View.dense(out, x.t, x.h, x.w, w.conv_out.co), false);
}

/// postprocess_video for one batch item: io.latent (fp32 or bf16, `io.latent_dtype`) -> io.rgb (uint8), under `plan`
/// (tiles.decodePlan of the decoded latent frames, built by the caller from its own free-memory figure: this component
/// never probes memory). Launches only; the caller synchronises its stream.
pub fn decode(l: launch.Launcher, w: *const Decoder, plan: tiles.DecodePlan, io: Io) !void {
    var arena: launch.Arena = .{ .base = io.scratch, .cap = io.scratch_bytes };
    const lay = try Layout.init(&arena, plan);
    const g = plan.grid;
    const lat_n: u64 = @as(u64, plan.latent_frames) * plan.lh * plan.lw * tiles.latent_channels;
    try l.run(.{ .call = .{ .latent_in = .{ .z = io.latent, .y = lay.latent16, .count = @intCast(lat_n), .inv_sf = invScale(), .z_f32 = @intFromBool(io.latent_dtype == .f32) } }, .grid = .{ @intCast((lat_n + 255) / 256), 1, 1 }, .block = .{ 256, 1, 1 } });
    const z_all = View.dense(lay.latent16, plan.latent_frames, plan.lh, plan.lw, tiles.latent_channels);
    try l.mark("decode.input", .{}, z_all, false);
    const sh = g.sample_h;
    const sw = g.sample_w;
    const frame_elems: u64 = @as(u64, sh) * sw * 3;
    var prev: ?struct { ptr: u64, frames: u32 } = null;
    var slot: usize = 0;
    for (0..plan.tileCount()) |k| {
        const tt = plan.temporal(@intCast(k));
        const buf = lay.frames_buf[slot];
        const zt = z_all.box(tt.lat_start, tt.lat_frames, 0, plan.lh, 0, plan.lw);
        if (!g.tiled()) {
            try decodeTile(l, w, lay, zt, buf);
        } else {
            try spatial(l, w, lay, plan, zt, tt.frames, buf);
        }
        const cur_ptr = buf + tt.drop * frame_elems * 2;
        const cur_frames = tt.frames - tt.drop;
        if (tt.blend > 0) {
            const p = prev.?;
            try l.run(blendLaunch(.{ .a = p.ptr, .b = cur_ptr, .fa = @intCast(p.frames), .ha = @intCast(sh), .wa = @intCast(sw), .fb = @intCast(cur_frames), .hb = @intCast(sh), .wb = @intCast(sw), .ch = 3, .axis = 0, .extent = @intCast(tt.blend) }));
        }
        if (tt.keep > 0) {
            const n = @as(u64, tt.keep) * frame_elems;
            try l.run(.{ .call = .{ .rgb_out = .{ .x = cur_ptr, .y = io.rgb + @as(u64, tt.out_frame) * frame_elems, .count = @intCast(n) } }, .grid = .{ @intCast((n + 255) / 256), 1, 1 }, .block = .{ 256, 1, 1 } });
        }
        prev = .{ .ptr = cur_ptr, .frames = cur_frames };
        slot ^= 1;
    }
}

fn blendLaunch(a: launch.BlendArgs) launch.Launch {
    const d0: u64 = @intCast(if (a.axis == 0) a.extent else a.fb);
    const d1: u64 = @intCast(if (a.axis == 1) a.extent else a.hb);
    const d2: u64 = @intCast(if (a.axis == 2) a.extent else a.wb);
    const n = d0 * d1 * d2 * @as(u64, @intCast(a.ch));
    return .{ .call = .{ .blend = a }, .grid = .{ @intCast((n + 255) / 256), 1, 1 }, .block = .{ 256, 1, 1 } };
}

/// tiled_decode of one temporal tile: every spatial tile decoded, then blend_v / blend_h in place in the reference's
/// (row, column) order, then each tile's kept box placed into `out` ([frames][sample_h][sample_w][3]) and cropped.
fn spatial(l: launch.Launcher, w: *const Decoder, lay: Layout, plan: tiles.DecodePlan, zt: View, frames: u32, out: u64) !void {
    const g = plan.grid;
    const th = g.tile_lh * tiles.spatial_ratio;
    const tw = g.tile_lw * tiles.spatial_ratio;
    const tile_bytes = lay.tile_elems * 2;
    for (0..g.rows) |i| for (0..g.cols) |j| {
        const t = g.tile(@intCast(i), @intCast(j));
        try decodeTile(l, w, lay, zt.box(0, zt.t, t.lat_y, g.tile_lh, t.lat_x, g.tile_lw), lay.tile_out + (i * g.cols + j) * tile_bytes);
    };
    const f: i32 = @intCast(frames);
    for (0..g.rows) |i| for (0..g.cols) |j| {
        const b = lay.tile_out + (i * g.cols + j) * tile_bytes;
        if (i > 0) try l.run(blendLaunch(.{ .a = lay.tile_out + ((i - 1) * g.cols + j) * tile_bytes, .b = b, .fa = f, .ha = @intCast(th), .wa = @intCast(tw), .fb = f, .hb = @intCast(th), .wb = @intCast(tw), .ch = 3, .axis = 1, .extent = @intCast(g.blend_h) }));
        if (j > 0) try l.run(blendLaunch(.{ .a = lay.tile_out + (i * g.cols + j - 1) * tile_bytes, .b = b, .fa = f, .ha = @intCast(th), .wa = @intCast(tw), .fb = f, .hb = @intCast(th), .wb = @intCast(tw), .ch = 3, .axis = 2, .extent = @intCast(g.blend_w) }));
    };
    for (0..g.rows) |i| for (0..g.cols) |j| {
        const t = g.tile(@intCast(i), @intCast(j));
        if (t.out_y >= g.sample_h or t.out_x >= g.sample_w) continue;
        const h = @min(t.keep_h, g.sample_h - t.out_y);
        const wd = @min(t.keep_w, g.sample_w - t.out_x);
        const n: u64 = @as(u64, frames) * h * wd * 3;
        try l.run(.{ .call = .{ .copy = .{ .src = lay.tile_out + (i * g.cols + j) * tile_bytes, .dst = out, .sf = f, .sh = @intCast(th), .sw = @intCast(tw), .df = f, .dh = @intCast(g.sample_h), .dw = @intCast(g.sample_w), .f = f, .h = @intCast(h), .w = @intCast(wd), .ch = 3, .sf0 = 0, .sh0 = 0, .sw0 = 0, .df0 = 0, .dh0 = @intCast(t.out_y), .dw0 = @intCast(t.out_x) } }, .grid = .{ @intCast((n + 255) / 256), 1, 1 }, .block = .{ 256, 1, 1 } });
    };
}

// ---- Tests ----------------------------------------------------------------------------------------------------------

const testing = std.testing;

/// Assigns every checkpoint tensor an address in a weights region (sizes of the repacked layouts) and binds a Decoder.
pub fn bindForTest(base: u64) struct { dec: Decoder, bytes: u64 } {
    var top: u64 = base;
    const S = struct {
        fn conv(t: *u64, co: u32, ci: u32, k: u32) ConvW {
            const w = std.mem.alignForward(u64, t.*, 256);
            t.* = w + launch.convWeightBytes(co, ci, k, k, k);
            const b = std.mem.alignForward(u64, t.*, 256);
            t.* = b + co * 2;
            return .{ .w = w, .b = b, .co = co, .ci = ci, .kt = k, .kh = k, .kw = k };
        }
        fn norm(t: *u64, c: u32) launch.NormW {
            const g = std.mem.alignForward(u64, t.*, 256);
            t.* = g + c * 4;
            return .{ .gamma = g, .beta = g + c * 2, .c = c };
        }
        fn res(t: *u64, ci: u32, co: u32) launch.ResnetW {
            return .{ .norm1 = norm(t, ci), .conv1 = conv(t, co, ci, 3), .norm2 = norm(t, co), .conv2 = conv(t, co, co, 3), .shortcut = if (ci != co) conv(t, co, ci, 1) else null };
        }
        fn mid(t: *u64) launch.MidW {
            return .{ .res0 = res(t, 512, 512), .attn = .{ .norm = norm(t, 512), .q = conv(t, 512, 512, 1), .k = conv(t, 512, 512, 1), .v = conv(t, 512, 512, 1), .out = conv(t, 512, 512, 1) }, .res1 = res(t, 512, 512) };
        }
    };
    var d: Decoder = undefined;
    d.post_quant = S.conv(&top, 16, 16, 1);
    d.conv_in = S.conv(&top, 512, 16, 3);
    d.mid = S.mid(&top);
    const ch = [4][2]u32{ .{ 512, 512 }, .{ 512, 512 }, .{ 512, 256 }, .{ 256, 128 } };
    for (ch, 0..) |io, i| {
        for (0..3) |r| d.up[i].resnets[r] = S.res(&top, if (r == 0) io[0] else io[1], io[1]);
        d.up[i].upsampler = if (i < 3) S.conv(&top, io[1], io[1], 3) else null;
    }
    d.norm_out = S.norm(&top, 128);
    d.conv_out = S.conv(&top, 3, 128, 3);
    return .{ .dec = d, .bytes = top - base };
}

test "decoder activation bound: 17 x 480 x 864 x 256 at the native tile, the scratch plan" {
    try testing.expectEqual(@as(u64, 17 * 480 * 864 * 256), maxActivation(5, 60, 108));
    const plan = try tiles.decodePlan(31, 60, 108, .unbounded);
    const bytes = scratchBytes(plan);
    // four activations (2 big + work a, b) of 3.61 GB, latent, attention (32,400 tokens), two 17-frame RGB tiles
    try testing.expect(bytes > 4 * maxActivation(5, 60, 108) * 2);
    try testing.expect(bytes < 15_000_000_000);
}

/// Runs `decode` in the simulator: bounds and race checks on every launch, the data-movement kernels executed on the host.
fn simDecode(plan: tiles.DecodePlan, execute: bool, latent: []const u16, rgb: []u8) !launch.Sim {
    var sim: launch.Sim = .{ .execute = execute };
    const wb = bindForTest(0x1000_0000_0000);
    sim.region(0x1000_0000_0000, wb.bytes, "weights");
    const sbytes = scratchBytes(plan);
    var scratch: []align(4096) u8 = &.{};
    var sbase: u64 = 0x2000_0000_0000;
    if (execute) {
        scratch = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(4096), sbytes); // lazily committed
        sbase = @intFromPtr(scratch.ptr);
    }
    defer if (execute) std.heap.page_allocator.free(scratch);
    sim.region(sbase, sbytes, "scratch");
    const lat_addr: u64 = if (execute) @intFromPtr(latent.ptr) else 0x3000_0000_0000;
    const rgb_addr: u64 = if (execute) @intFromPtr(rgb.ptr) else 0x4000_0000_0000;
    const lat_bytes = @as(u64, plan.latent_frames) * plan.lh * plan.lw * 16 * 2;
    const rgb_bytes = @as(u64, plan.sampleFramesTotal()) * plan.grid.sample_h * plan.grid.sample_w * 3;
    sim.region(lat_addr, lat_bytes, "latent");
    sim.region(rgb_addr, rgb_bytes, "rgb");
    try decode(sim.launcher(), &wb.dec, plan, .{ .latent = lat_addr, .latent_dtype = .bf16, .rgb = rgb_addr, .scratch = sbase, .scratch_bytes = sbytes });
    return sim;
}

test "native 5 s decode: every launch in bounds and race free, 14 decoder passes, 121 frames" {
    const plan = try tiles.decodePlan(31, 60, 108, .unbounded);
    const sim = try simDecode(plan, false, &.{}, &.{});
    try testing.expectEqual(@as(u64, 14), sim.counts[@backingInt(launch.Kernel.transpose)]); // one attention a pass
    try testing.expectEqual(@as(u64, 1), sim.counts[@backingInt(launch.Kernel.latent_in)]);
    try testing.expectEqual(@as(u64, 13), sim.counts[@backingInt(launch.Kernel.blend)]);
    try testing.expectEqual(@as(u64, 14), sim.counts[@backingInt(launch.Kernel.rgb_out)]);
}

/// The reference's tile loops, transcribed (in place, in its order) over recorded decoder outputs, to uint8.
const Reference = struct {
    fn temporal(gpa: std.mem.Allocator, plan: tiles.DecodePlan, outs: []const []f16) ![]u8 {
        const sh = plan.grid.sample_h;
        const sw = plan.grid.sample_w;
        const fe = @as(usize, sh) * sw * 3;
        const total = plan.sampleFramesTotal();
        var result: std.ArrayList(f16) = .empty;
        defer result.deinit(gpa);
        // row: per tile, the (possibly dropped) frames, mutated in place by blend_t
        var rows = try gpa.alloc([]f16, outs.len);
        defer gpa.free(rows);
        for (outs, 0..) |o, i| rows[i] = if (i > 0) o[fe..] else o;
        const fmin = plan.tiling.ft - 1;
        for (rows, 0..) |tile, i| {
            const len = tile.len / fe;
            if (i > 0) {
                const a = rows[i - 1];
                const alen = a.len / fe;
                const e = @min(@min(alen, len), fmin - plan.tiling.fs);
                for (0..e) |x| for (0..fe) |q| {
                    tile[x * fe + q] = launch.Sim.blendPair(a[(alen - e + x) * fe + q], tile[x * fe + q], @intCast(x), @intCast(e));
                };
                const lim: usize = if (i == rows.len - 1) fmin else plan.tiling.fs;
                try result.appendSlice(gpa, tile[0 .. @min(lim, len) * fe]);
            } else {
                try result.appendSlice(gpa, tile[0 .. @min(plan.tiling.fs + 1, len) * fe]);
            }
        }
        const out = try gpa.alloc(u8, total * fe);
        for (out, 0..) |*u, i| u.* = launch.Sim.rgbOut(result.items[i]);
        return out;
    }

    /// tiled_decode over recorded tiles [rows][cols] of [F][th][tw][3]: returns [F][sh][sw][3].
    fn spatial(gpa: std.mem.Allocator, g: tiles.SpatialGrid, frames: usize, tl: []const []f16) ![]f16 {
        const th: usize = g.tile_lh * 8;
        const tw: usize = g.tile_lw * 8;
        for (0..g.rows) |i| for (0..g.cols) |j| {
            const b = tl[i * g.cols + j];
            if (i > 0) {
                const a = tl[(i - 1) * g.cols + j];
                const e = g.blend_h;
                for (0..frames) |f| for (0..e) |y| for (0..tw * 3) |q| {
                    const ib = (f * th + y) * tw * 3 + q;
                    const ia = (f * th + th - e + y) * tw * 3 + q;
                    b[ib] = launch.Sim.blendPair(a[ia], b[ib], @intCast(y), e);
                };
            }
            if (j > 0) {
                const a = tl[i * g.cols + j - 1];
                const e = g.blend_w;
                for (0..frames) |f| for (0..th) |y| for (0..e) |x| for (0..3) |c| {
                    const ib = ((f * th + y) * tw + x) * 3 + c;
                    const ia = ((f * th + y) * tw + tw - e + x) * 3 + c;
                    b[ib] = launch.Sim.blendPair(a[ia], b[ib], @intCast(x), e);
                };
            }
        };
        // concatenate kept boxes, crop to the sample size
        const out = try gpa.alloc(f16, frames * g.sample_h * g.sample_w * 3);
        var oy: usize = 0;
        for (0..g.rows) |i| {
            const hl: usize = if (i == g.rows - 1) th else g.keep_h;
            var ox: usize = 0;
            for (0..g.cols) |j| {
                const wl: usize = if (j == g.cols - 1) tw else g.keep_w;
                const t = tl[i * g.cols + j];
                for (0..frames) |f| for (0..hl) |y| for (0..wl) |x| for (0..3) |c| {
                    if (oy + y >= g.sample_h or ox + x >= g.sample_w) continue;
                    out[((f * g.sample_h + oy + y) * g.sample_w + ox + x) * 3 + c] = t[((f * th + y) * tw + x) * 3 + c];
                };
                ox += wl;
            }
            oy += hl;
        }
        return out;
    }
};

/// Records the decoder's conv_out tiles as the simulator produces them (the same pseudo-random sequence).
fn recordTiles(gpa: std.mem.Allocator, plan: tiles.DecodePlan) ![][]f16 {
    const g = plan.grid;
    var seed: u64 = (launch.Sim{}).seed;
    const n_tiles = plan.tileCount() * g.count();
    const out = try gpa.alloc([]f16, n_tiles);
    for (0..n_tiles) |k| {
        const tt = plan.temporal(@intCast(k / g.count()));
        const m = @as(usize, tt.frames) * g.tile_lh * 8 * g.tile_lw * 8;
        out[k] = try gpa.alloc(f16, m * 3);
        for (out[k]) |*v| {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            const u: f32 = @floatFromInt(seed >> 40);
            v.* = @floatCast(u / 16777216.0 * 2.5 - 1.25);
        }
    }
    return out;
}

fn checkAgainstReference(plan: tiles.DecodePlan) !void {
    const gpa = testing.allocator;
    const latent = try gpa.alloc(u16, @as(usize, plan.latent_frames) * plan.lh * plan.lw * 16);
    defer gpa.free(latent);
    @memset(latent, 0x3f80); // bf16 1.0
    const fe = @as(usize, plan.grid.sample_h) * plan.grid.sample_w * 3;
    const rgb = try gpa.alloc(u8, plan.sampleFramesTotal() * fe);
    defer gpa.free(rgb);
    @memset(rgb, 0xaa);
    const sim = try simDecode(plan, true, latent, rgb);
    try testing.expectEqual(@as(u32, @intCast(plan.tileCount() * plan.grid.count())), sim.conv_out_calls);

    const recorded = try recordTiles(gpa, plan);
    defer {
        for (recorded) |r| gpa.free(r);
        gpa.free(recorded);
    }
    const g = plan.grid;
    var per_t = try gpa.alloc([]f16, plan.tileCount());
    defer {
        if (g.tiled()) for (per_t) |p| gpa.free(p);
        gpa.free(per_t);
    }
    for (0..plan.tileCount()) |k| {
        if (g.tiled()) {
            const tt = plan.temporal(@intCast(k));
            per_t[k] = try Reference.spatial(gpa, g, tt.frames, recorded[k * g.count() ..][0..g.count()]);
        } else per_t[k] = recorded[k];
    }
    const want = try Reference.temporal(gpa, plan, per_t);
    defer gpa.free(want);
    try testing.expectEqualSlices(u8, want, rgb);
}

test "temporal tiling, blending and frame emission equal the reference's loops (31 latent frames)" {
    try checkAgainstReference(try tiles.decodePlan(31, 2, 3, .unbounded));
}

test "uneven temporal tail (30 latent frames, last tile 4 frames) equals the reference" {
    try checkAgainstReference(try tiles.decodePlan(30, 2, 3, .unbounded));
}

test "spatial tiles with overlapping in-place blends (2 x 3 grid, blend 32 > stride 24) equal the reference" {
    const plan = try tiles.decodePlan(7, 10, 13, .{ .free_bytes = 69_632_000 });
    try testing.expectEqual(@as(u32, 2), plan.grid.rows);
    try testing.expectEqual(@as(u32, 3), plan.grid.cols);
    try testing.expectEqual(@as(u32, 2), plan.temporal_tiles);
    try checkAgainstReference(plan);
}

test "short clip decodes directly in one pass" {
    try checkAgainstReference(try tiles.decodePlan(3, 2, 3, .unbounded));
}
