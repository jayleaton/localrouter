//! Host-side still-image preparation matching the Qwen edit twin's RGB patches and RGBA VAE input.
const std = @import("std");

pub const Error = error{ InvalidDimensions, EmptyGrid, TooLarge, OutOfMemory };
pub const Prepared = struct {
    width: u32,
    height: u32,
    grid: [3]u32,
    rgba: []u8,
    rgb: []u8,
    patches: []f32,
    pub fn deinit(p: Prepared, a: std.mem.Allocator) void { a.free(p.rgba); a.free(p.rgb); a.free(p.patches); }
};

/// Python dimensions(): round-to-even at each 32-pixel grid conversion.
pub fn dimensions(width: u32, height: u32, area: u64) Error!struct { width: u32, height: u32 } {
    if (width == 0 or height == 0 or area == 0) return error.InvalidDimensions;
    const ratio = @as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height));
    const w0 = roundEven(@sqrt(@as(f64, @floatFromInt(area)) * ratio) / 32.0) * 32.0;
    const h0 = roundEven(w0 / ratio / 32.0) * 32.0;
    if (w0 < 32 or h0 < 32 or w0 > @as(f64, @floatFromInt(std.math.maxInt(u32))) or h0 > @as(f64, @floatFromInt(std.math.maxInt(u32)))) return error.EmptyGrid;
    return .{ .width = @intFromFloat(w0), .height = @intFromFloat(h0) };
}

/// The largest reference `dimensions(_, _, area)` can return with both sides at most `max_side`, in 16x16 latent
/// tokens: what the reference scratch is sized for. A 32-pixel output pair (w, h) is reachable when some ratio r
/// rounds to both, sqrt(area r) within 16 of w and w / r within 16 of h; ties count as reachable (an upper bound).
pub fn maxArea(area: u64, max_side: u32) u32 {
    const a: f64 = @floatFromInt(area);
    var best: u32 = 0;
    var w: u32 = 32;
    while (w <= max_side) : (w += 32) {
        var h: u32 = 32;
        while (h <= max_side) : (h += 32) {
            const wf: f64 = @floatFromInt(w);
            const hf: f64 = @floatFromInt(h);
            const lo = @max((wf - 16) * (wf - 16) / a, wf / (hf + 16));
            const hi = @min((wf + 16) * (wf + 16) / a, wf / (hf - 16));
            if (lo <= hi) best = @max(best, (w / 16) * (h / 16));
        }
    }
    return best;
}

test "maxArea bounds every reference dimensions() returns within the side limit" {
    const cap = maxArea(1 << 20, 1360);
    try std.testing.expect(cap >= 4096);
    var width: u32 = 16;
    while (width <= 4096) : (width += 5) {
        var height: u32 = 16;
        while (height <= 4096) : (height += 7) {
            const d = try dimensions(width, height, 1 << 20);
            if (d.width <= 1360 and d.height <= 1360) try std.testing.expect((d.width / 16) * (d.height / 16) <= cap);
        }
    }
}

fn roundEven(x: f64) f64 {
    const f = @floor(x); const d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return if (@mod(f, 2.0) == 0) f else f + 1;
}

/// Input RGBA8 is resized by Pillow-style separable Lanczos (support 3, 22-bit coefficients), with RGBA premultiplied
/// during filtering. `rgb` matches Pillow's `RGB` paste over white; `patches` are fp32, merge-block order.
pub fn prepare(a: std.mem.Allocator, rgba: []const u8, width: u32, height: u32, area: u64) Error!Prepared {
    if (width == 0 or height == 0 or rgba.len != @as(usize, width) * @as(usize, height) * 4) return error.InvalidDimensions;
    const dim = try dimensions(width, height, area);
    const count = @as(usize, dim.width) * @as(usize, dim.height);
    if (count > (1 << 28)) return error.TooLarge;
    const resized = try resizeRgba(a, rgba, width, height, dim.width, dim.height);
    errdefer a.free(resized);
    const rgb = a.alloc(u8, count * 3) catch return error.OutOfMemory;
    errdefer a.free(rgb);
    for (0..count) |i| {
        const a32: u32 = resized[i * 4 + 3];
        for (0..3) |c| rgb[i * 3 + c] = @intCast((@as(u32, resized[i * 4 + c]) * a32 + 255 * (255 - a32) + 127) / 255);
    }
    const patch_rows = count / 256;
    const patches = a.alloc(f32, patch_rows * 1536) catch return error.OutOfMemory;
    errdefer a.free(patches);
    var row: usize = 0;
    const blocks_y: usize = dim.height / 32; const blocks_x: usize = dim.width / 32;
    for (0..blocks_y) |by| for (0..blocks_x) |bx| for (0..2) |my| for (0..2) |mx| {
        var at = row * 1536;
        for (0..3) |c| for (0..2) |_| for (0..16) |py| for (0..16) |px| {
            const y = by * 32 + my * 16 + py; const x = bx * 32 + mx * 16 + px;
            const v: f32 = @floatFromInt(rgb[(y * dim.width + x) * 3 + c]);
            const scaled = v * (1.0 / 255.0);
            patches[at] = (scaled - 0.5) / 0.5;
            at += 1;
        };
        row += 1;
    };
    return .{ .width = dim.width, .height = dim.height, .grid = .{ 1, dim.height / 16, dim.width / 16 }, .rgba = resized, .rgb = rgb, .patches = patches };
}

/// Pillow's RGBA resize with LANCZOS: premultiplied (RGBA -> RGBa), filtered, then un-premultiplied; a copy when the
/// size is unchanged. Caller owns the result.
pub fn resizeRgba(a: std.mem.Allocator, rgba: []const u8, width: u32, height: u32, dw: u32, dh: u32) Error![]u8 {
    if (dw == width and dh == height) return a.dupe(u8, rgba) catch error.OutOfMemory;
    const premul = a.alloc(u8, @as(usize, width) * @as(usize, height) * 4) catch return error.OutOfMemory;
    defer a.free(premul);
    for (0..@as(usize, width) * @as(usize, height)) |i| {
        const alpha = rgba[i * 4 + 3];
        for (0..3) |c| premul[i * 4 + c] = @intCast((@as(u16, rgba[i * 4 + c]) * alpha + 127) / 255);
        premul[i * 4 + 3] = alpha;
    }
    const out = try lanczos(a, premul, width, height, dw, dh);
    for (0..@as(usize, dw) * dh) |i| {
        const alpha: u32 = out[i * 4 + 3];
        if (alpha > 0 and alpha < 255) for (0..3) |c| {
            out[i * 4 + c] = @intCast(@min(255, (@as(u32, out[i * 4 + c]) * 255) / alpha));
        };
    }
    return out;
}

const precision: i64 = 1 << 22;
const Coeff = struct { first: usize, weights: []i32 };

fn sinc(x: f64) f64 {
    if (x == 0) return 1;
    const p = std.math.pi * x;
    return @sin(p) / p;
}

fn coefficients(a: std.mem.Allocator, source: usize, target: usize) Error![]Coeff {
    const out = a.alloc(Coeff, target) catch return error.OutOfMemory;
    errdefer a.free(out);
    const scale = @as(f64, @floatFromInt(source)) / @as(f64, @floatFromInt(target));
    const filter_scale = @max(1.0, scale); const support = 3.0 * filter_scale;
    for (0..target) |i| {
        const center = (@as(f64, @floatFromInt(i)) + 0.5) * scale;
        const lo = @max(0, @as(i64, @intFromFloat(@floor(center - support + 0.5))));
        const hi = @min(@as(i64, @intCast(source)), @as(i64, @intFromFloat(@floor(center + support + 0.5))));
        const n: usize = @intCast(hi - lo);
        const w = a.alloc(i32, n) catch return error.OutOfMemory;
        errdefer a.free(w);
        var sum: f64 = 0;
        for (0..n) |j| { const x = (@as(f64, @floatFromInt(lo + @as(i64, @intCast(j)))) + 0.5 - center) / filter_scale; const q = if (@abs(x) < 3) sinc(x) * sinc(x / 3) else 0; sum += q; }
        if (sum == 0) return error.InvalidDimensions;
        for (0..n) |j| {
            const x = (@as(f64, @floatFromInt(lo + @as(i64, @intCast(j)))) + 0.5 - center) / filter_scale;
            const q = if (@abs(x) < 3) sinc(x) * sinc(x / 3) / sum else 0;
            w[j] = @intFromFloat(@round(q * @as(f64, @floatFromInt(precision))));
        }
        out[i] = .{ .first = @intCast(lo), .weights = w };
    }
    return out;
}

fn lanczos(a: std.mem.Allocator, src: []const u8, sw: u32, sh: u32, dw: u32, dh: u32) Error![]u8 {
    const cx = try coefficients(a, @intCast(sw), @intCast(dw)); defer { for (cx) |c| a.free(c.weights); a.free(cx); }
    const cy = try coefficients(a, @intCast(sh), @intCast(dh)); defer { for (cy) |c| a.free(c.weights); a.free(cy); }
    const tmp = a.alloc(u8, @as(usize, dw) * @as(usize, sh) * 4) catch return error.OutOfMemory;
    defer a.free(tmp);
    for (0..@as(usize, sh)) |y| for (0..@as(usize, dw)) |x| for (0..4) |c| {
        const cf = cx[x]; var sum: i64 = 0;
        for (cf.weights, 0..) |weight, j| sum += @as(i64, src[(y * @as(usize, sw) + cf.first + j) * 4 + c]) * @as(i64, weight);
        tmp[(y * @as(usize, dw) + x) * 4 + c] = clampFixed(sum);
    };
    const dst = a.alloc(u8, @as(usize, dw) * @as(usize, dh) * 4) catch return error.OutOfMemory;
    errdefer a.free(dst);
    for (0..@as(usize, dh)) |y| for (0..@as(usize, dw)) |x| for (0..4) |c| {
        const cf = cy[y]; var sum: i64 = 0;
        for (cf.weights, 0..) |weight, j| sum += @as(i64, tmp[((cf.first + j) * @as(usize, dw) + x) * 4 + c]) * @as(i64, weight);
        dst[(y * @as(usize, dw) + x) * 4 + c] = clampFixed(sum);
    };
    return dst;
}

fn clampFixed(sum: i64) u8 { return @intCast(@max(0, @min(255, @divFloor(sum + precision / 2, precision)))); }

test "same-size preparation preserves transparent RGBA samples" {
    const a = std.testing.allocator;
    const src = try a.alloc(u8, 32 * 32 * 4);
    defer a.free(src);
    for (0..32 * 32) |i| src[i * 4 ..][0..4].* = .{ 213, 77, 31, if (i == 0) 0 else 128 };
    const p = try prepare(a, src, 32, 32, 1024);
    defer p.deinit(a);
    try std.testing.expectEqualSlices(u8, src, p.rgba);
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255 }, p.rgb[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 234, 166, 143 }, p.rgb[3..6]);
}

test "dimensions rounds half-way values to even grid multiples" {
    const d = try dimensions(1, 1, 6400);
    try std.testing.expectEqual(@as(u32, 64), d.width);
    try std.testing.expectEqual(@as(u32, 64), d.height);
}

test "Lanczos resize preserves a constant opaque image" {
    const a = std.testing.allocator;
    const src = try a.alloc(u8, 32 * 32 * 4);
    defer a.free(src);
    for (0..32 * 32) |i| src[i * 4 ..][0..4].* = .{ 80, 129, 211, 255 };
    const p = try prepare(a, src, 32, 32, 4096);
    defer p.deinit(a);
    try std.testing.expectEqual(@as(u32, 64), p.width);
    try std.testing.expectEqual(@as(u32, 64), p.height);
    const expected = [_]i16{ 80, 129, 211 };
    for (0..64 * 64) |i| {
        for (0..3) |c| try std.testing.expect(@abs(@as(i16, p.rgba[i * 4 + c]) - expected[c]) <= 1);
        try std.testing.expectEqual(@as(u8, 255), p.rgba[i * 4 + 3]);
    }
}
