//! Tile plans of the Kandinsky-6 video VAE, transcribed from AutoencoderKLHunyuanVideo (pinned upstream kandinsky-6
//! eac06e39 kandinsky/core/components/hunyuan_vae.py): get_enc_optimal_tiling / get_dec_optimal_tiling, apply_tiling,
//! _decode, _temporal_tiled_decode, tiled_decode and _encode. Pure functions of the shape and a memory policy, no I/O.
//!
//! The reference picks its tiles from `torch.cuda.mem_get_info()` at call time. That makes its output depend on how much
//! memory happened to be free, so the plan here takes the policy explicitly: `.unbounded` is the reference with ample free
//! memory (the native 864x480 clip needs about 14.4 GB free to be decoded without spatial tiles), `.free_bytes` replays a
//! recorded reference run. Native fixtures and this runtime must use the same policy to compare bits.
//!
//! There is no temporal feature cache in this VAE: every tile is a whole, independent decoder pass (GroupNorm statistics
//! over that tile only) and tiles meet through the reference's linear blends. Reusing state across tiles would change bits.
//!
//! Where the reference's loops would leave part of the output uncovered (its ranges stop before the edge for some shapes)
//! or would not run at all (a zero stride), the plan is refused instead of reproducing a short or failed decode.

const std = @import("std");

pub const spatial_ratio: u32 = 8;
pub const temporal_ratio: u32 = 4;
pub const latent_channels: u32 = 16;

/// Decoded frames of `latent_frames` latent frames: (T - 1) * 4 + 1 (31 -> 121, the native five-second clip at 24 fps).
pub fn sampleFrames(latent_frames: u32) u32 {
    std.debug.assert(latent_frames >= 1);
    return (latent_frames - 1) * temporal_ratio + 1;
}

pub const MemPolicy = union(enum) {
    unbounded,
    free_bytes: u64,
};

/// get_enc_optimal_tiling's result in sample pixels: tile (1, ft, ht, wt), stride (fs, hs, ws).
pub const Tiling = struct {
    ft: u32,
    ht: u32,
    wt: u32,
    fs: u32,
    hs: u32,
    ws: u32,
};

/// get_enc_optimal_tiling(shape[..3:] = h, w): Python float arithmetic in f64, the same operations in the same order.
pub fn optimalTiling(h: u32, w: u32, policy: MemPolicy) Tiling {
    const hf: f64 = @floatFromInt(h);
    const wf: f64 = @floatFromInt(w);
    const max_area: f64 = switch (policy) {
        .unbounded => std.math.inf(f64),
        .free_bytes => |b| @as(f64, @floatFromInt(b)) / 256.0 / 17.0 / 8.0,
    };
    const num_vals: u64 = 256 * 17 * (@as(u64, h) + 32) * (@as(u64, w) + 32);
    const area: u64 = @as(u64, h) * w;
    if (@as(f64, @floatFromInt(area)) < max_area and num_vals < (1 << 31)) {
        return .{ .ft = 17, .ht = h, .wt = w, .fs = 8, .hs = h, .ws = w };
    }
    const k = @max(hf / wf, wf / hf);
    const n_area: u64 = @intFromFloat(@ceil(@as(f64, @floatFromInt(area)) / max_area));
    const n_vals: u64 = @intFromFloat(@ceil(@as(f64, @floatFromInt(num_vals)) / @as(f64, 1 << 31)));
    const n: f64 = @floatFromInt(@max(n_area, n_vals));
    const a: u64 = @intFromFloat(@ceil(@sqrt(n / k)));
    const b: u64 = @intFromFloat(@ceil(@sqrt(n * k)));
    const wn, const hn = if (h >= w) .{ a, b } else .{ b, a };
    var t: Tiling = .{ .ft = 17, .ht = h, .wt = w, .fs = 8, .hs = h, .ws = w };
    if (wn > 1) {
        t.wt = @as(u32, @intFromFloat(@ceil(wf / @as(f64, @floatFromInt(wn)) / 8.0))) * 8 + 16;
        t.ws = t.wt -% 32;
    }
    if (hn > 1) {
        t.ht = @as(u32, @intFromFloat(@ceil(hf / @as(f64, @floatFromInt(hn)) / 8.0))) * 8 + 16;
        t.hs = t.ht -% 32;
    }
    return t;
}

pub const Error = error{ InvalidShape, UnsupportedTiling };

/// One spatial tile of a decode: latent box and the sample-space crop the reference keeps.
pub const SpatialTile = struct {
    lat_y: u32,
    lat_x: u32,
    /// Rows / columns of the decoded tile kept (height_lim, width_lim), from its top-left corner.
    keep_h: u32,
    keep_w: u32,
    /// Where the kept box lands in the assembled sample (sum of the previous rows' / columns' keeps).
    out_y: u32,
    out_x: u32,
};

/// The spatial grid of one temporal tile (tiled_decode), or a single direct tile.
pub const SpatialGrid = struct {
    rows: u32,
    cols: u32,
    /// Latent size of every tile (all tiles are full: the reference's ranges never start a short tile).
    tile_lh: u32,
    tile_lw: u32,
    lat_stride_h: u32,
    lat_stride_w: u32,
    /// Sample rows / columns blended between vertical / horizontal neighbours (blend_v / blend_h extent, min-ed).
    blend_h: u32,
    blend_w: u32,
    keep_h: u32, // stride rows kept from a non-last row of tiles
    keep_w: u32,
    sample_h: u32,
    sample_w: u32,

    pub fn count(g: SpatialGrid) u32 {
        return g.rows * g.cols;
    }

    pub fn tiled(g: SpatialGrid) bool {
        return g.count() > 1 or g.tile_lh * spatial_ratio != g.sample_h or g.tile_lw * spatial_ratio != g.sample_w;
    }

    pub fn tile(g: SpatialGrid, i: u32, j: u32) SpatialTile {
        std.debug.assert(i < g.rows and j < g.cols);
        const th = g.tile_lh * spatial_ratio;
        const tw = g.tile_lw * spatial_ratio;
        return .{
            .lat_y = i * g.lat_stride_h,
            .lat_x = j * g.lat_stride_w,
            .keep_h = if (i == g.rows - 1) th else g.keep_h,
            .keep_w = if (j == g.cols - 1) tw else g.keep_w,
            .out_y = i * g.keep_h,
            .out_x = j * g.keep_w,
        };
    }
};

fn rangeCount(stop_exclusive: i64, step: i64) u32 {
    // len(range(0, stop, step)) for step > 0
    if (stop_exclusive <= 0) return 0;
    return @intCast(@divFloor(stop_exclusive - 1, step) + 1);
}

/// tiled_decode's grid for a latent of lh x lw under tiling `t`, or the direct decoder when it would not tile.
pub fn spatialGrid(lh: u32, lw: u32, t: Tiling) Error!SpatialGrid {
    const sh = lh * spatial_ratio;
    const sw = lw * spatial_ratio;
    const tlh = t.ht / spatial_ratio;
    const tlw = t.wt / spatial_ratio;
    const direct: SpatialGrid = .{ .rows = 1, .cols = 1, .tile_lh = lh, .tile_lw = lw, .lat_stride_h = lh, .lat_stride_w = lw, .blend_h = 0, .blend_w = 0, .keep_h = sh, .keep_w = sw, .sample_h = sh, .sample_w = sw };
    if (!(lw > tlw or lh > tlh)) return direct;
    const slh = t.hs / spatial_ratio;
    const slw = t.ws / spatial_ratio;
    if (slh == 0 or slw == 0 or tlh == 0 or tlw == 0) return error.UnsupportedTiling; // range(.., 0) raises
    if (t.hs > t.ht or t.ws > t.wt) return error.UnsupportedTiling;
    const rows = rangeCount(@as(i64, lh) - tlh + 1, slh);
    const cols = rangeCount(@as(i64, lw) - tlw + 1, slw);
    if (rows == 0 or cols == 0) return error.UnsupportedTiling;
    const th = tlh * spatial_ratio;
    const tw = tlw * spatial_ratio;
    const g: SpatialGrid = .{
        .rows = rows,
        .cols = cols,
        .tile_lh = tlh,
        .tile_lw = tlw,
        .lat_stride_h = slh,
        .lat_stride_w = slw,
        .blend_h = @min(th, t.ht - t.hs),
        .blend_w = @min(tw, t.wt - t.ws),
        .keep_h = t.hs,
        .keep_w = t.ws,
        .sample_h = sh,
        .sample_w = sw,
    };
    // The reference concatenates the kept boxes and crops to the sample size; a shortfall would be a smaller video.
    if ((rows - 1) * t.hs + @min(th, t.ht) < sh or (cols - 1) * t.ws + @min(tw, t.wt) < sw) return error.UnsupportedTiling;
    return g;
}

/// One temporal tile of a decode (_temporal_tiled_decode), or the whole latent when the decode is not temporally tiled.
pub const TemporalTile = struct {
    lat_start: u32,
    lat_frames: u32,
    /// Decoded frames of this tile ((lat_frames - 1) * 4 + 1) and the leading frames dropped (1 for every tile but the first).
    frames: u32,
    drop: u32,
    /// Frames blended with the previous tile (blend_t extent), 0 for the first.
    blend: u32,
    /// Frames emitted from the start of the (dropped) tile: t_lim, clipped to the clip length.
    keep: u32,
    /// Index of the first emitted frame in the output clip.
    out_frame: u32,
};

pub const DecodePlan = struct {
    latent_frames: u32,
    lh: u32,
    lw: u32,
    tiling: Tiling,
    temporal_tiles: u32,
    lat_stride_t: u32,
    tile_lat_frames: u32, // tile_latent_min_num_frames + 1 (the frames a temporal tile reads, at most)
    grid: SpatialGrid,

    pub fn sampleFramesTotal(p: DecodePlan) u32 {
        return sampleFrames(p.latent_frames);
    }

    pub fn temporallyTiled(p: DecodePlan) bool {
        return p.temporal_tiles > 0;
    }

    pub fn tileCount(p: DecodePlan) u32 {
        return if (p.temporal_tiles > 0) p.temporal_tiles else 1;
    }

    /// Temporal tile `i` with its emit window. Tiles must be visited in order to place out_frame; this recomputes it.
    pub fn temporal(p: DecodePlan, i: u32) TemporalTile {
        const total = p.sampleFramesTotal();
        if (p.temporal_tiles == 0) return .{ .lat_start = 0, .lat_frames = p.latent_frames, .frames = total, .drop = 0, .blend = 0, .keep = total, .out_frame = 0 };
        std.debug.assert(i < p.temporal_tiles);
        const fs = p.tiling.fs;
        const fmin = p.tiling.ft - 1; // tile_sample_min_num_frames
        var out: u32 = 0;
        var prev_len: u32 = 0;
        var k: u32 = 0;
        while (true) : (k += 1) {
            const start = k * p.lat_stride_t;
            const lat = @min(p.tile_lat_frames, p.latent_frames - start);
            const frames = sampleFrames(lat);
            const drop: u32 = if (k > 0) 1 else 0;
            const len = frames - drop;
            const lim: u32 = if (k == 0) fs + 1 else if (k == p.temporal_tiles - 1) fmin else fs;
            const keep = @min(@min(lim, len), total -| out);
            const blend: u32 = if (k > 0) @min(@min(prev_len, len), fmin - fs) else 0;
            if (k == i) return .{ .lat_start = start, .lat_frames = lat, .frames = frames, .drop = drop, .blend = blend, .keep = keep, .out_frame = out };
            out += keep;
            prev_len = len;
        }
    }
};

/// AutoencoderKLHunyuanVideo.decode(z) for z of `latent_frames` x lh x lw (latent units): which tiles run and how they meet.
pub fn decodePlan(latent_frames: u32, lh: u32, lw: u32, policy: MemPolicy) Error!DecodePlan {
    if (latent_frames == 0 or lh == 0 or lw == 0) return error.InvalidShape;
    // get_dec_optimal_tiling: the encoder-input shape [b, 3, 4 (f - 1) + 1, 8 h, 8 w].
    const t = optimalTiling(lh * spatial_ratio, lw * spatial_ratio, policy);
    const lt_min = (t.ft - 1) / temporal_ratio; // tile_latent_min_num_frames
    var p: DecodePlan = .{
        .latent_frames = latent_frames,
        .lh = lh,
        .lw = lw,
        .tiling = t,
        .temporal_tiles = 0,
        .lat_stride_t = t.fs / temporal_ratio,
        .tile_lat_frames = lt_min + 1,
        .grid = try spatialGrid(lh, lw, t),
    };
    if (latent_frames > lt_min + 1) {
        if (p.lat_stride_t == 0 or t.fs > t.ft - 1) return error.UnsupportedTiling;
        p.temporal_tiles = rangeCount(@as(i64, latent_frames) - lt_min + 1, p.lat_stride_t);
        // Every frame of the clip must be emitted (the reference crops a longer concatenation, never pads a shorter one).
        const last = p.temporal(p.temporal_tiles - 1);
        if (last.out_frame + last.keep != p.sampleFramesTotal()) return error.UnsupportedTiling;
    } else {
        p.tile_lat_frames = latent_frames;
    }
    return p;
}

/// AutoencoderKLHunyuanVideo._encode under encode(x, opt_tiling): true when the reference runs the encoder once on the
/// whole input. I2V (encode_i2va_first_frame) passes opt_tiling=False, which always encodes directly. Tiled encodes are
/// outside this component's contract and are refused by the encoder.
pub fn encodeIsDirect(frames: u32, h: u32, w: u32, opt_tiling: bool, policy: MemPolicy) bool {
    if (!opt_tiling) return true;
    const t = optimalTiling(h, w, policy);
    if (frames > (t.ft - 1) + 1) return false;
    return !(w > t.wt or h > t.ht);
}

test "frame contract: 31 latent frames decode to 121" {
    try std.testing.expectEqual(@as(u32, 121), sampleFrames(31));
    try std.testing.expectEqual(@as(u32, 1), sampleFrames(1));
}

test "native 864x480: no spatial tiles, 14 temporal tiles of 5 latent frames, 121 frames emitted once" {
    const p = try decodePlan(31, 60, 108, .unbounded);
    try std.testing.expectEqual(Tiling{ .ft = 17, .ht = 480, .wt = 864, .fs = 8, .hs = 480, .ws = 864 }, p.tiling);
    try std.testing.expect(!p.grid.tiled());
    try std.testing.expectEqual(@as(u32, 14), p.temporal_tiles);
    var next: u32 = 0;
    for (0..p.temporal_tiles) |i| {
        const t = p.temporal(@intCast(i));
        try std.testing.expectEqual(@as(u32, @intCast(i)) * 2, t.lat_start);
        try std.testing.expectEqual(@as(u32, 5), t.lat_frames);
        try std.testing.expectEqual(@as(u32, 17), t.frames);
        try std.testing.expectEqual(next, t.out_frame);
        try std.testing.expectEqual(@as(u32, if (i == 0) 9 else if (i == 13) 16 else 8), t.keep);
        try std.testing.expectEqual(@as(u32, if (i == 0) 0 else 8), t.blend);
        next += t.keep;
    }
    try std.testing.expectEqual(@as(u32, 121), next);
}

test "memory policy: the native clip tiles spatially when the reference saw little free memory" {
    // 14.4 GB is the threshold: h * w < free / 256 / 17 / 8.
    const ample = optimalTiling(480, 864, .{ .free_bytes = 15_000_000_000 });
    try std.testing.expectEqual(@as(u32, 864), ample.wt);
    const tight = optimalTiling(480, 864, .{ .free_bytes = 8_000_000_000 });
    // max_area = 8e9 / 256 / 17 / 8 = 229779.4: N = ceil(414720 / 229779.4) = 2, k = 1.8, a = b = 2 (values from the
    // reference function run in Python)
    try std.testing.expectEqual(Tiling{ .ft = 17, .ht = 256, .wt = 448, .fs = 8, .hs = 224, .ws = 416 }, tight);
    const g = try spatialGrid(60, 108, tight);
    // rows range(0, 60 - 32 + 1, 28) = 0, 28; cols range(0, 108 - 56 + 1, 52) = 0, 52
    try std.testing.expectEqual(@as(u32, 2), g.rows);
    try std.testing.expectEqual(@as(u32, 2), g.cols);
    try std.testing.expectEqual(@as(u32, 32), g.blend_h);
    try std.testing.expectEqual(@as(u32, 32), g.blend_w);
    const last = g.tile(1, 1);
    try std.testing.expectEqual(@as(u32, 28), last.lat_y);
    try std.testing.expectEqual(@as(u32, 52), last.lat_x);
    try std.testing.expectEqual(@as(u32, 256), last.keep_h);
    try std.testing.expectEqual(@as(u32, 224), last.out_y);
    try std.testing.expectEqual(@as(u32, 416), last.out_x);
    try std.testing.expectEqual(@as(u32, 480), last.out_y + last.keep_h);
    try std.testing.expectEqual(@as(u32, 864), last.out_x + last.keep_w);
}

test "large frames tile spatially by the 2^31 element bound even with unbounded memory" {
    const t = optimalTiling(720, 1280, .unbounded);
    // num_vals = 256 * 17 * 752 * 1312 = 4,293,853,184: N = 2, k = 16/9; a = ceil(sqrt(1.125)) = 2, b = ceil(sqrt(3.56)) = 2
    try std.testing.expectEqual(Tiling{ .ft = 17, .ht = 376, .wt = 656, .fs = 8, .hs = 344, .ws = 624 }, t);
    const p = try decodePlan(31, 90, 160, .unbounded);
    try std.testing.expectEqual(@as(u32, 2), p.grid.rows);
    try std.testing.expectEqual(@as(u32, 2), p.grid.cols);
}

test "short clips decode directly; uncovered and zero-stride plans are refused" {
    const p = try decodePlan(5, 60, 108, .unbounded);
    try std.testing.expectEqual(@as(u32, 0), p.temporal_tiles);
    try std.testing.expectEqual(@as(u32, 17), p.temporal(0).keep);
    // 128 x 192 with a tight budget: columns range(0, 15, 6) stop at latent 22 of 24 (176 of 192 columns): refused.
    const t = optimalTiling(128, 192, .{ .free_bytes = 17_825_792 * 4 });
    try std.testing.expectError(error.UnsupportedTiling, spatialGrid(16, 24, t));
    const z = optimalTiling(32, 48, .{ .free_bytes = 17_825_792 });
    try std.testing.expectError(error.UnsupportedTiling, spatialGrid(4, 6, z));
}

test "every temporal plan from 6 to 64 latent frames emits each frame exactly once or is refused" {
    var lat: u32 = 6;
    while (lat <= 64) : (lat += 1) {
        const p = decodePlan(lat, 60, 108, .unbounded) catch |e| {
            try std.testing.expectEqual(error.UnsupportedTiling, e);
            continue;
        };
        var next: u32 = 0;
        for (0..p.temporal_tiles) |i| {
            const t = p.temporal(@intCast(i));
            try std.testing.expectEqual(next, t.out_frame);
            try std.testing.expect(t.lat_start + t.lat_frames <= lat);
            next += t.keep;
        }
        try std.testing.expectEqual(sampleFrames(lat), next);
    }
}

test "encode: I2V (opt_tiling false) is always direct" {
    try std.testing.expect(encodeIsDirect(1, 480, 864, false, .{ .free_bytes = 1 }));
    try std.testing.expect(encodeIsDirect(1, 480, 864, true, .unbounded));
    try std.testing.expect(!encodeIsDirect(121, 480, 864, true, .unbounded));
}
