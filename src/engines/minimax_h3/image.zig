//! The first frame's pixels, as ComfyUI 0.37.0 prepares them (docs/dev/H3-I2V.md): `LoadImage` (uint8 / 255 in float32),
//! the node's `_resize(image, width, height, "disabled")` = `common_upscale(.., "lanczos", crop="disabled")`, which is PIL:
//!   q = trunc(255 * x)  in float32 (`np.clip(255. * image, 0, 255).astype(uint8)`: a truncation, not a rounding),
//!   Image.resize((w, h), LANCZOS) on the uint8 RGB image (a copy when the size is unchanged),
//!   result / 255.0 in float32.
//! The PIL resize is Pillow's `Resample.c` for 8-bit images, operation for operation: separable, horizontal pass first, filter
//! support 3 * max(scale, 1), coefficients in double (`precompute_coeffs`), normalised, quantised to 22 fractional bits
//! (`normalize_coeffs_8bpc`), each output sample `(2^21 + sum(pixel * k)) >> 22` clipped to 0..255. The sinc is the toolkit's
//! portable fdlibm `smath.sincos` (the twin's `image.py` uses the same), where PIL calls libm's sin: the two differ by at most an
//! ulp, which moves a quantised coefficient only on a rounding tie (probability about 1e-9 a coefficient).
//! Decoding is the caller's: PNG through `media/png_decode.zig` (the tool); JPEG and WebP are not decoded.

const std = @import("std");
const smath = @import("qwen_image").smath;

pub const Rgb = struct { width: u32, height: u32, data: []const u8 }; // [height, width, 3] uint8

const precision_bits = 22;
const support = 3.0; // LANCZOS

fn sinc(x: f64) f64 {
    if (x == 0.0) return 1.0;
    const y = x * std.math.pi;
    return smath.sincos(y).sin / y;
}

fn lanczos(x: f64) f64 {
    if (-3.0 <= x and x < 3.0) return sinc(x) * sinc(x / 3.0);
    return 0.0;
}

/// Pillow's `precompute_coeffs` + `normalize_coeffs_8bpc` for the whole axis (in0 = 0, in1 = in_size).
const Coeffs = struct {
    ksize: usize,
    bounds: []i32, // per output: xmin, count
    kk: []i32, // per output: ksize quantised coefficients

    fn init(gpa: std.mem.Allocator, in_size: u32, out_size: u32) !Coeffs {
        const scale: f64 = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
        const fscale = @max(scale, 1.0);
        const sup = support * fscale;
        const ksize: usize = @as(usize, @intFromFloat(@ceil(sup))) * 2 + 1;
        const bounds = try gpa.alloc(i32, 2 * out_size);
        errdefer gpa.free(bounds);
        const kk = try gpa.alloc(i32, ksize * out_size);
        errdefer gpa.free(kk);
        const pre = try gpa.alloc(f64, ksize);
        defer gpa.free(pre);
        for (0..out_size) |xx| {
            const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            const ss = 1.0 / fscale;
            var xmin: i64 = @intFromFloat(center - sup + 0.5); // C's (int): truncation toward zero
            if (xmin < 0) xmin = 0;
            var xmax: i64 = @intFromFloat(center + sup + 0.5);
            if (xmax > in_size) xmax = in_size;
            xmax -= xmin;
            var ww: f64 = 0.0;
            const n: usize = @intCast(xmax);
            for (0..n) |x| {
                const w = lanczos((@as(f64, @floatFromInt(@as(i64, @intCast(x)) + xmin)) - center + 0.5) * ss);
                pre[x] = w;
                ww += w;
            }
            for (0..n) |x| {
                if (ww != 0.0) pre[x] /= ww;
            }
            for (n..ksize) |x| pre[x] = 0.0;
            bounds[xx * 2] = @intCast(xmin);
            bounds[xx * 2 + 1] = @intCast(xmax);
            const q: f64 = @floatFromInt(@as(u32, 1) << precision_bits);
            for (0..ksize) |x| {
                const p = pre[x];
                kk[xx * ksize + x] = if (p < 0) @intFromFloat(-0.5 + p * q) else @intFromFloat(0.5 + p * q);
            }
        }
        return .{ .ksize = ksize, .bounds = bounds, .kk = kk };
    }

    fn deinit(c: Coeffs, gpa: std.mem.Allocator) void {
        gpa.free(c.bounds);
        gpa.free(c.kk);
    }
};

fn clip8(v: i64) u8 {
    return @intCast(std.math.clamp(v >> precision_bits, 0, 255));
}

/// `Image.resize((dw, dh), LANCZOS)` of an RGB uint8 image, as Pillow computes it; caller frees.
pub fn resizeLanczos(gpa: std.mem.Allocator, src: Rgb, dw: u32, dh: u32) ![]u8 {
    if (dw == 0 or dh == 0 or src.width == 0 or src.height == 0 or
        @as(u64, dw) * dh > 1 << 28 or src.data.len != @as(u64, src.width) * src.height * 3) return error.BadImage;
    if (src.width == dw and src.height == dh) return gpa.dupe(u8, src.data); // PIL returns a copy
    const sw: usize = src.width;
    const sh: usize = src.height;
    var cur: []const u8 = src.data;
    var cw = sw;
    var tmp: ?[]u8 = null;
    defer if (tmp) |t| gpa.free(t);
    if (dw != src.width) { // horizontal pass
        const c = try Coeffs.init(gpa, src.width, dw);
        defer c.deinit(gpa);
        const out = try gpa.alloc(u8, sh * dw * 3);
        for (0..sh) |y| for (0..dw) |xx| {
            const xmin: usize = @intCast(c.bounds[xx * 2]);
            const cnt: usize = @intCast(c.bounds[xx * 2 + 1]);
            const k = c.kk[xx * c.ksize ..][0..c.ksize];
            for (0..3) |ch| {
                var s: i64 = 1 << (precision_bits - 1);
                for (0..cnt) |x| s += @as(i64, cur[(y * cw + x + xmin) * 3 + ch]) * k[x];
                out[(y * dw + xx) * 3 + ch] = clip8(s);
            }
        };
        tmp = out;
        cur = out;
        cw = dw;
    }
    var result: []u8 = undefined;
    if (dh != src.height) { // vertical pass
        const c = Coeffs.init(gpa, src.height, dh) catch |e| return e;
        defer c.deinit(gpa);
        result = try gpa.alloc(u8, @as(usize, dh) * cw * 3);
        for (0..dh) |yy| {
            const ymin: usize = @intCast(c.bounds[yy * 2]);
            const cnt: usize = @intCast(c.bounds[yy * 2 + 1]);
            const k = c.kk[yy * c.ksize ..][0..c.ksize];
            for (0..cw) |x| for (0..3) |ch| {
                var s: i64 = 1 << (precision_bits - 1);
                for (0..cnt) |y| s += @as(i64, cur[((y + ymin) * cw + x) * 3 + ch]) * k[y];
                result[(yy * cw + x) * 3 + ch] = clip8(s);
            };
        }
    } else {
        result = try gpa.dupe(u8, cur);
    }
    return result;
}

/// `uint8 -> float32 / 255 -> trunc(255 * x) -> uint8` (the LoadImage / `lanczos()` round trip, in float32).
pub fn viaFloat(k: u8) u8 {
    const f: f32 = @as(f32, @floatFromInt(k)) / 255.0;
    const t: f32 = 255.0 * f;
    return @intFromFloat(std.math.clamp(t, 0.0, 255.0));
}

/// The first frame on the canvas: float32 RGB [h, w, 3] in [0, 1], ComfyUI's `_resize(first_frame, w, h, "disabled")` of the
/// loaded image. `out.len == w * h * 3`.
pub fn canvas(gpa: std.mem.Allocator, src: Rgb, w: u32, h: u32, out: []f32) !void {
    std.debug.assert(out.len == @as(usize, w) * h * 3);
    if (src.data.len != @as(usize, src.width) * src.height * 3 or src.width == 0 or src.height == 0) return error.BadImage;
    const q = try gpa.alloc(u8, src.data.len);
    defer gpa.free(q);
    for (q, src.data) |*o, v| o.* = viaFloat(v);
    const r = try resizeLanczos(gpa, .{ .width = src.width, .height = src.height, .data = q }, w, h);
    defer gpa.free(r);
    for (out, r) |*o, v| o.* = @as(f32, @floatFromInt(v)) / 255.0;
}

test "float round trip: which bytes survive the truncation" {
    var bad: u32 = 0;
    for (0..256) |k| {
        if (viaFloat(@intCast(k)) != k) bad += 1;
    }
    // ComfyUI truncates, so a byte can come back one lower when 255 * (k / 255) rounds below k; whichever bytes those are, they are
    // the same ones in numpy (float32 both ways), and the count is small.
    try std.testing.expect(bad < 128);
    try std.testing.expectEqual(@as(u8, 0), viaFloat(0));
    try std.testing.expectEqual(@as(u8, 255), viaFloat(255));
}

test "a constant image stays constant, up and down" {
    const gpa = std.testing.allocator;
    var px: [20 * 14 * 3]u8 = undefined;
    @memset(&px, 117);
    const src: Rgb = .{ .width = 20, .height = 14, .data = &px };
    for ([_][2]u32{ .{ 7, 5 }, .{ 20, 5 }, .{ 33, 14 }, .{ 64, 96 }, .{ 20, 14 } }) |sz| {
        const r = try resizeLanczos(gpa, src, sz[0], sz[1]);
        defer gpa.free(r);
        try std.testing.expectEqual(@as(usize, sz[0] * sz[1] * 3), r.len);
        for (r) |v| try std.testing.expectEqual(@as(u8, 117), v);
    }
}

test "an exact 2x downscale of a flat-pair image averages through the filter and keeps the order of a ramp" {
    const gpa = std.testing.allocator;
    var px: [16 * 2 * 3]u8 = undefined;
    for (0..2) |y| for (0..16) |x| for (0..3) |c| {
        px[(y * 16 + x) * 3 + c] = @intCast(x * 16);
    };
    const r = try resizeLanczos(gpa, .{ .width = 16, .height = 2, .data = &px }, 8, 2);
    defer gpa.free(r);
    var last: i32 = -1;
    for (0..8) |x| {
        const v: i32 = r[x * 3];
        try std.testing.expect(v >= last);
        last = v;
    }
    try std.testing.expect(r[7 * 3] > r[0]);
}

test "canvas: same size is the byte round trip" {
    const gpa = std.testing.allocator;
    var px: [8 * 8 * 3]u8 = undefined;
    for (&px, 0..) |*p, i| p.* = @truncate(i * 7);
    var out: [8 * 8 * 3]f32 = undefined;
    try canvas(gpa, .{ .width = 8, .height = 8, .data = &px }, 8, 8, &out);
    for (out, px) |o, v| try std.testing.expectEqual(@as(f32, @floatFromInt(viaFloat(v))) / 255.0, o);
}
