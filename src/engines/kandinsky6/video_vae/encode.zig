//! Kandinsky-6 video VAE encode for I2V: the VAE half of encode_i2va_first_frame (pinned upstream kandinsky-6 eac06e39
//! core/algo/prepare_latents.py). The caller has already decoded, resized and cropped the image (uint8 RGB); this runs
//! tensor / 127.5 - 1.0, AutoencoderKLHunyuanVideo.encode(x, opt_tiling=False) (always one direct encoder pass: no
//! tiles), DiagonalGaussianDistribution, and latent * scaling_factor in the packed THWC layout.
//!
//!   rgb     uint8 [frames][h][w][3] (I2V: one frame)
//!   eps     fp16 [lt][lh][lw][16] standard normal noise for .sample(), drawn by the caller (this component owns no RNG);
//!           0 gives .mode() (the mean) instead. Native draws it from torch's implicit default CUDA generator
//!           (latent_dist.sample() with no generator, prepare_latents.py:236), before and apart from the seeded video
//!           noise, so the video seed alone does not reproduce it
//!   mean, stdv  fp16 [lt][lh][lw][16] distribution statistics (optional, 0 to skip)
//!   latent  fp16 [lt][lh][lw][16]: (mean + std * eps) * 0.476986
//! with lt = ((frames - 1) / 2) / 2 + 1 and lh, lw the height and width after three stride-2 causal convolutions
//! (480 x 864 -> 60 x 108).

const std = @import("std");
const launch = @import("launch.zig");
const tiles = @import("tiles.zig");

const View = launch.View;
const ConvW = launch.ConvW;

/// HunyuanVideoDownBlock3D: two resnets, then (blocks 0..2) the downsampler's strided causal conv.
pub const DownW = struct { resnets: [2]launch.ResnetW, downsampler: ?ConvW };

pub const Encoder = struct {
    conv_in: ConvW,
    down: [4]DownW,
    mid: launch.MidW,
    norm_out: launch.NormW,
    conv_out: ConvW,
    quant: ConvW,
};

/// Strides (time, space) of down blocks 0..2: temporal_compression_ratio 4 downsamples time in blocks 1 and 2 only.
pub const down_strides = [3][2]u32{ .{ 1, 2 }, .{ 2, 2 }, .{ 2, 2 } };

/// floor((n + 2 - 3) / s) + 1: a causal (time) or symmetric (space) replicate-padded kernel-3 convolution of stride s.
fn down(n: u32, s: u32) u32 {
    return (n - 1) / s + 1;
}

pub const Shape = struct { t: u32, h: u32, w: u32 };

pub fn latentShape(frames: u32, h: u32, w: u32) Shape {
    var s: Shape = .{ .t = frames, .h = h, .w = w };
    for (down_strides) |d| s = .{ .t = down(s.t, d[0]), .h = down(s.h, d[1]), .w = down(s.w, d[1]) };
    return s;
}

/// Largest fp16 activation of the encoder over frames x h x w: the down blocks' channels at their resolutions.
pub fn maxActivation(frames: u32, h: u32, w: u32) u64 {
    const ch = [4][2]u64{ .{ 128, 128 }, .{ 128, 256 }, .{ 256, 512 }, .{ 512, 512 } };
    var s: Shape = .{ .t = frames, .h = h, .w = w };
    var best: u64 = 0;
    for (ch, 0..) |io, i| {
        const p: u64 = @as(u64, s.t) * s.h * s.w;
        best = @max(best, p * @max(io[0], io[1]));
        if (i < 3) s = .{ .t = down(s.t, down_strides[i][0]), .h = down(s.h, down_strides[i][1]), .w = down(s.w, down_strides[i][1]) };
    }
    return best;
}

pub const Layout = struct {
    input: u64,
    big: [2]u64,
    work: launch.Work,
    moments: u64,

    pub fn init(arena: *launch.Arena, frames: u32, h: u32, w: u32) launch.Error!Layout {
        const act = maxActivation(frames, h, w);
        const ls = latentShape(frames, h, w);
        const lp: u64 = @as(u64, ls.t) * ls.h * ls.w;
        var l: Layout = undefined;
        l.input = try arena.take(@as(u64, frames) * h * w * 3 * 2);
        l.big = .{ try arena.take(act * 2), try arena.take(act * 2) };
        l.work = try launch.Work.init(arena, act, @as(u64, frames) * h * w, @intCast(lp), 512);
        l.moments = try arena.take(lp * 32 * 2);
        return l;
    }
};

pub fn scratchBytes(frames: u32, h: u32, w: u32) u64 {
    var arena: launch.Arena = .{ .base = 0, .cap = 0, .measure = true };
    _ = Layout.init(&arena, frames, h, w) catch unreachable;
    return arena.peak;
}

pub const Io = struct {
    rgb: u64,
    frames: u32,
    h: u32,
    w: u32,
    eps: u64 = 0,
    mean: u64 = 0,
    stdv: u64 = 0,
    latent: u64,
    scratch: u64,
    scratch_bytes: u64,
};

/// One direct encoder pass and the posterior. Launches only; the caller synchronises its stream.
pub fn encode(l: launch.Launcher, e: *const Encoder, io: Io) !void {
    if (io.frames == 0 or io.h == 0 or io.w == 0) return error.InvalidShape;
    var arena: launch.Arena = .{ .base = io.scratch, .cap = io.scratch_bytes };
    const lay = try Layout.init(&arena, io.frames, io.h, io.w);
    const work = lay.work;
    const n: u64 = @as(u64, io.frames) * io.h * io.w * 3;
    try l.run(.{ .call = .{ .rgb8_in = .{ .x = io.rgb, .y = lay.input, .count = @intCast(n) } }, .grid = .{ @intCast((n + 255) / 256), 1, 1 }, .block = .{ 256, 1, 1 } });
    var cur = lay.big[0];
    var spare = lay.big[1];
    try l.run(try launch.conv(View.dense(lay.input, io.frames, io.h, io.w, 3), e.conv_in, .{}, cur));
    var x = View.dense(cur, io.frames, io.h, io.w, e.conv_in.co);
    for (e.down, 0..) |db, i| {
        for (db.resnets) |r| {
            x = try launch.resnet(l, work, x, r, spare);
            std.mem.swap(u64, &cur, &spare);
        }
        if (db.downsampler) |d| {
            const s = down_strides[i];
            const spec: launch.ConvSpec = .{ .st = s[0], .sh = s[1], .sw = s[1] };
            const o = launch.convOut(x, d, spec);
            try l.run(try launch.conv(x, d, spec, spare));
            x = View.dense(spare, o[0], o[1], o[2], d.co);
            std.mem.swap(u64, &cur, &spare);
        }
    }
    x = try launch.midBlock(l, work, x, e.mid, spare);
    if (x.ptr == spare) std.mem.swap(u64, &cur, &spare);
    try launch.groupNorm(l, work, x, e.norm_out, true, work.a);
    try l.run(try launch.conv(View.dense(work.a, x.t, x.h, x.w, x.c), e.conv_out, .{}, spare));
    try l.run(try launch.conv(View.dense(spare, x.t, x.h, x.w, e.conv_out.co), e.quant, .{ .causal_pad = false }, lay.moments));
    const lp = x.positions();
    try l.run(.{ .call = .{ .posterior = .{ .moments = lay.moments, .eps = io.eps, .mean = io.mean, .stdv = io.stdv, .latent = io.latent, .positions = @intCast(lp), .lc = @intCast(tiles.latent_channels), .sf = @floatCast(@as(f64, 0.476986)) } }, .grid = .{ @intCast((lp * tiles.latent_channels + 255) / 256), 1, 1 }, .block = .{ 256, 1, 1 } });
}

// ---- Tests ----------------------------------------------------------------------------------------------------------

const testing = std.testing;

fn bindForTest(base: u64) struct { enc: Encoder, bytes: u64 } {
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
    };
    var e: Encoder = undefined;
    e.conv_in = S.conv(&top, 128, 3, 3);
    const ch = [4][2]u32{ .{ 128, 128 }, .{ 128, 256 }, .{ 256, 512 }, .{ 512, 512 } };
    for (ch, 0..) |io, i| {
        for (0..2) |r| e.down[i].resnets[r] = S.res(&top, if (r == 0) io[0] else io[1], io[1]);
        e.down[i].downsampler = if (i < 3) S.conv(&top, io[1], io[1], 3) else null;
    }
    e.mid = .{ .res0 = S.res(&top, 512, 512), .attn = .{ .norm = S.norm(&top, 512), .q = S.conv(&top, 512, 512, 1), .k = S.conv(&top, 512, 512, 1), .v = S.conv(&top, 512, 512, 1), .out = S.conv(&top, 512, 512, 1) }, .res1 = S.res(&top, 512, 512) };
    e.norm_out = S.norm(&top, 512);
    e.conv_out = S.conv(&top, 32, 512, 3);
    e.quant = S.conv(&top, 32, 32, 1);
    return .{ .enc = e, .bytes = top - base };
}

test "latent shape: 480 x 864 image -> 1 x 60 x 108; 17 frames -> 5 latent frames" {
    try testing.expectEqual(Shape{ .t = 1, .h = 60, .w = 108 }, latentShape(1, 480, 864));
    try testing.expectEqual(Shape{ .t = 5, .h = 60, .w = 108 }, latentShape(17, 480, 864));
    try testing.expectEqual(Shape{ .t = 31, .h = 60, .w = 108 }, latentShape(121, 480, 864));
    // odd sizes round up as the reference's padded stride-2 convolutions do
    try testing.expectEqual(Shape{ .t = 1, .h = 61, .w = 108 }, latentShape(1, 484, 864));
}

test "I2V encode at 864x480: every launch in bounds and race free, one scalar-gather conv_in, one attention" {
    var sim: launch.Sim = .{};
    const wb = bindForTest(0x1000_0000_0000);
    sim.region(0x1000_0000_0000, wb.bytes, "weights");
    const sbytes = scratchBytes(1, 480, 864);
    sim.region(0x2000_0000_0000, sbytes, "scratch");
    sim.region(0x3000_0000_0000, 480 * 864 * 3, "rgb");
    const lat: u64 = 60 * 108 * 16 * 2;
    sim.region(0x4000_0000_0000, 4 * lat, "outputs");
    try encode(sim.launcher(), &wb.enc, .{ .rgb = 0x3000_0000_0000, .frames = 1, .h = 480, .w = 864, .eps = 0x4000_0000_0000, .mean = 0x4000_0000_0000 + lat, .stdv = 0x4000_0000_0000 + 2 * lat, .latent = 0x4000_0000_0000 + 3 * lat, .scratch = 0x2000_0000_0000, .scratch_bytes = sbytes });
    try testing.expectEqual(@as(u64, 1), sim.counts[@backingInt(launch.Kernel.conv_wide_s)]);
    try testing.expectEqual(@as(u64, 1), sim.counts[@backingInt(launch.Kernel.transpose)]);
    try testing.expectEqual(@as(u64, 1), sim.counts[@backingInt(launch.Kernel.posterior)]);
    // 6,480 tokens: seven row blocks of attention scores
    try testing.expectEqual(@as(u64, 7), sim.counts[@backingInt(launch.Kernel.softmax_rows)]);
    try testing.expect(sbytes < 1_000_000_000);
}

test "encode refuses a scratch buffer smaller than its plan" {
    var sim: launch.Sim = .{};
    const wb = bindForTest(0x1000_0000_0000);
    try testing.expectError(error.BufferTooSmall, encode(sim.launcher(), &wb.enc, .{ .rgb = 0, .frames = 1, .h = 480, .w = 864, .latent = 0, .scratch = 0x2000_0000_0000, .scratch_bytes = scratchBytes(1, 480, 864) - 1 }));
}
