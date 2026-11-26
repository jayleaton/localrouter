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
//!
//! H3-Image edits (ComfyUI e9afbccc and #157's workflows; docs/dev/H3-IMAGE.md section 1.6; the twin's `h3i/edit.py`,
//! which `test_edit.py` holds to ComfyUI's own code bit for bit) add, on float32 RGB [h, w, 3] images: `loadRgba`
//! (LoadImage's image and mask), `totalPixelsSize` and `area` (ImageScaleToTotalPixels), `shortEdgeSize` and `resizeTo`
//! (the conditioning node's two copies), `cropCenter` and `scaleCenter` (ImageScale lanczos, crop center), `bilinear`
//! (the composite's mask resize), `composite` and `toU8` (ImageCompositeMasked, SaveImage). The host test holds each to
//! the twin's `edit_image.json` bit for bit.

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

// ---------------------------------------------------------------------------------------------------------- edits

/// LoadImage of an 8-bit image: `img` [h, w, 3] = rgb / 255 in float32 and, for RGBA, `mask` [h, w] = 1 - alpha / 255
/// (1 where transparent). `px` is [h, w, channels], channels 3 or 4; `mask` must be null for RGB.
pub fn loadRgba(px: []const u8, channels: u32, img: []f32, mask: ?[]f32) !void {
    if ((channels != 3 and channels != 4) or px.len % channels != 0 or img.len != px.len / channels * 3) return error.BadImage;
    if ((mask != null) != (channels == 4)) return error.BadImage;
    const n = px.len / channels;
    if (mask) |m| if (m.len != n) return error.BadImage;
    for (0..n) |i| {
        for (0..3) |c| img[i * 3 + c] = @as(f32, @floatFromInt(px[i * channels + c])) / 255.0;
        if (mask) |m| m[i] = 1.0 - @as(f32, @floatFromInt(px[i * channels + 3])) / 255.0;
    }
}

/// Python's `round` of a float: to the nearest integer, ties to even.
pub fn pyRound(x: f64) f64 {
    if (@abs(x - @trunc(x)) == 0.5) return 2.0 * @round(x / 2.0);
    return @round(x);
}

/// ImageScaleToTotalPixels' size (`nodes_post_processing.py:249-256`): round(side * sqrt(MP * 2^20 / area) / steps) * steps.
pub fn totalPixelsSize(w: u32, h: u32, megapixels: f64, steps: u32) [2]u32 {
    const scale = @sqrt(megapixels * 1024.0 * 1024.0 / @as(f64, @floatFromInt(@as(u64, w) * h)));
    const st: f64 = @floatFromInt(steps);
    var out: [2]u32 = undefined;
    for ([2]u32{ w, h }, 0..) |side, i| out[i] = @intFromFloat(pyRound(@as(f64, @floatFromInt(side)) * scale / st) * st);
    return out;
}

/// `_size_short_edge(w, h, low, high)` (nodes_minimax_h3.py:677-680): the short edge into [low, high], rounded to 32.
pub fn shortEdgeSize(w: u32, h: u32, low: u32, high: u32) [2]u32 {
    const mn = @min(w, h);
    const scale = @as(f64, @floatFromInt(@min(@max(mn, low), high))) / @as(f64, @floatFromInt(mn));
    var out: [2]u32 = undefined;
    for ([2]u32{ w, h }, 0..) |side, i| out[i] = @max(32, @as(u32, @intFromFloat(pyRound(@as(f64, @floatFromInt(side)) * scale / 32.0))) * 32);
    return out;
}

/// `F.interpolate(mode="area")` of a float32 [h, w, 3] image to [H, W, 3] (adaptive_avg_pool2d, channels last): each
/// output's window [floor(o in / out), ceil((o + 1) in / out)) summed in fp32 row by row, then / kh / kw.
pub fn area(src: []const f32, w: u32, h: u32, out: []f32, W: u32, H: u32) void {
    std.debug.assert(src.len == @as(usize, w) * h * 3 and out.len == @as(usize, W) * H * 3);
    for (0..H) |oh| {
        const a = oh * h / H;
        const b = ((oh + 1) * h + H - 1) / H;
        for (0..W) |ow| {
            const l = ow * w / W;
            const r = ((ow + 1) * w + W - 1) / W;
            var s: [3]f32 = .{ 0, 0, 0 };
            for (a..b) |ih| for (l..r) |iw| {
                for (0..3) |c| s[c] += src[(ih * w + iw) * 3 + c];
            };
            for (0..3) |c| out[(oh * W + ow) * 3 + c] = s[c] / @as(f32, @floatFromInt(b - a)) / @as(f32, @floatFromInt(r - l));
        }
    }
}

/// `np.clip(255. * x, 0, 255).astype(np.uint8)` of a float32 (the 8-bit round trip of `lanczos`, and SaveImage).
pub fn toU8(x: f32) u8 {
    const t: f32 = 255.0 * x;
    return @intFromFloat(std.math.clamp(t, 0.0, 255.0));
}

/// `comfy.utils.lanczos` of a float32 [h, w, 3] image: to uint8 (`toU8`), Pillow's LANCZOS, / 255 in float32.
pub fn lanczosF(gpa: std.mem.Allocator, src: []const f32, w: u32, h: u32, out: []f32, W: u32, H: u32) !void {
    if (src.len != @as(usize, w) * h * 3 or out.len != @as(usize, W) * H * 3) return error.BadImage;
    const q = try gpa.alloc(u8, src.len);
    defer gpa.free(q);
    for (q, src) |*o, v| o.* = toU8(v);
    const r = try resizeLanczos(gpa, .{ .width = w, .height = h, .data = q }, W, H);
    defer gpa.free(r);
    for (out, r) |*o, v| o.* = @as(f32, @floatFromInt(v)) / 255.0;
}

/// `_resize_to(image, (W, H))`: the image itself at its own size, else `lanczosF`.
pub fn resizeTo(gpa: std.mem.Allocator, src: []const f32, w: u32, h: u32, out: []f32, W: u32, H: u32) !void {
    if (w == W and h == H) {
        if (out.len != src.len) return error.BadImage;
        @memcpy(out, src);
        return;
    }
    return lanczosF(gpa, src, w, h, out, W, H);
}

/// `common_upscale`'s crop="center" offsets (comfy/utils.py:1145-1156) for a w x h image going to W x H: x, y.
pub fn cropCenter(w: u32, h: u32, W: u32, H: u32) [2]u32 {
    const wf: f64 = @floatFromInt(w);
    const hf: f64 = @floatFromInt(h);
    const old = wf / hf;
    const new = @as(f64, @floatFromInt(W)) / @as(f64, @floatFromInt(H));
    if (old > new) return .{ @intFromFloat(pyRound((wf - wf * (new / old)) / 2.0)), 0 };
    if (old < new) return .{ 0, @intFromFloat(pyRound((hf - hf * (old / new)) / 2.0)) };
    return .{ 0, 0 };
}

/// ImageScale(lanczos, crop center) to W x H: the centered crop of the aspect, then `lanczosF`.
pub fn scaleCenter(gpa: std.mem.Allocator, src: []const f32, w: u32, h: u32, out: []f32, W: u32, H: u32) !void {
    if (src.len != @as(usize, w) * h * 3) return error.BadImage;
    const c = cropCenter(w, h, W, H);
    const cw = w - 2 * c[0];
    const ch = h - 2 * c[1];
    const crop = try gpa.alloc(f32, @as(usize, cw) * ch * 3);
    defer gpa.free(crop);
    for (0..ch) |y| @memcpy(crop[y * cw * 3 ..][0 .. cw * 3], src[((y + c[1]) * w + c[0]) * 3 ..][0 .. cw * 3]);
    return lanczosF(gpa, crop, cw, ch, out, W, H);
}

const Lin = struct { j0: u32, j1: u32, l0: f32, l1: f32 };

fn linIndex(n_in: u32, n_out: u32, i: u32) Lin {
    if (n_in == n_out) return .{ .j0 = i, .j1 = i, .l0 = 1, .l1 = 0 };
    const scale: f32 = @as(f32, @floatFromInt(n_in)) / @as(f32, @floatFromInt(n_out));
    var src: f32 = scale * (@as(f32, @floatFromInt(i)) + 0.5) - 0.5; // Zig does not contract: two roundings, as ATen's
    if (src < 0) src = 0;
    const j0: u32 = @min(@as(u32, @intFromFloat(@floor(src))), n_in - 1);
    const l1 = std.math.clamp(src - @as(f32, @floatFromInt(j0)), 0.0, 1.0);
    return .{ .j0 = j0, .j1 = j0 + @intFromBool(j0 < n_in - 1), .l0 = 1.0 - l1, .l1 = l1 };
}

/// The composite's mask resize, `F.interpolate(mode="bilinear", align_corners=False)` of [h, w] to [H, W] as ATen's
/// generic CPU kernel computes it (the kernel for H + W > 128; a smaller output other than the identity is refused):
/// along W, then along H, each lerp fma(t0, l0, t1 * l1).
pub fn bilinear(m: []const f32, w: u32, h: u32, out: []f32, W: u32, H: u32) !void {
    if (m.len != @as(usize, w) * h or out.len != @as(usize, W) * H) return error.BadImage;
    if (w == W and h == H) {
        @memcpy(out, m);
        return;
    }
    if (W + H <= 128) return error.MaskResizeTooSmall;
    for (0..H) |y| {
        const ly = linIndex(h, H, @intCast(y));
        for (0..W) |x| {
            const lx = linIndex(w, W, @intCast(x));
            var r: [2]f32 = undefined;
            for ([2]u32{ ly.j0, ly.j1 }, 0..) |yy, k| {
                const t0 = m[yy * w + lx.j0];
                const t1 = m[yy * w + lx.j1];
                r[k] = @mulAdd(f32, t0, lx.l0, t1 * lx.l1);
            }
            out[y * W + x] = @mulAdd(f32, r[0], ly.l0, r[1] * ly.l1);
        }
    }
}

/// ImageCompositeMasked at (0, 0), the mask already at the size: m * src + (1 - m) * dest, each product and the sum
/// rounded to fp32. `dest`, `src`, `out` [H, W, 3]; `mask` [H, W].
pub fn composite(dest: []const f32, src: []const f32, mask: []const f32, out: []f32) void {
    std.debug.assert(dest.len == src.len and out.len == src.len and mask.len * 3 == src.len);
    for (mask, 0..) |m, p| {
        const inv: f32 = 1.0 - m;
        for (0..3) |c| {
            const i = p * 3 + c;
            out[i] = m * src[i] + inv * dest[i];
        }
    }
}

test "edits: every step the twin's edit.py writes (edit_image.json), bit for bit" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("edit_image.json"), .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    for (root.get("total_pixels").?.array.items) |v| {
        const r = v.array.items;
        const got = totalPixelsSize(int(r[0]), int(r[1]), num(r[2]), 32);
        try std.testing.expectEqual([2]u32{ int(r[3]), int(r[4]) }, got);
    }
    for (root.get("short_edge").?.array.items) |v| {
        const r = v.array.items;
        try std.testing.expectEqual([2]u32{ int(r[4]), int(r[5]) }, shortEdgeSize(int(r[0]), int(r[1]), int(r[2]), int(r[3])));
    }
    for (root.get("crop").?.array.items) |v| {
        const r = v.array.items;
        try std.testing.expectEqual([2]u32{ int(r[4]), int(r[5]) }, cropCenter(int(r[0]), int(r[1]), int(r[2]), int(r[3])));
    }
    const im = root.get("image").?.object;
    const w = int(im.get("w").?);
    const h = int(im.get("h").?);
    const n: usize = @as(usize, w) * h;
    const rgba = try unhex(gpa, im.get("rgba").?.string);
    defer gpa.free(rgba);
    const img = try gpa.alloc(f32, n * 3);
    defer gpa.free(img);
    const mask = try gpa.alloc(f32, n);
    defer gpa.free(mask);
    try loadRgba(rgba, 4, img, mask);
    try expectSha(im.get("rgb").?.string, img);
    try expectSha(im.get("mask").?.string, mask);
    for (im.get("cases").?.array.items) |cv| {
        const c = cv.object;
        const W = int(c.get("w").?);
        const H = int(c.get("h").?);
        const a = try gpa.alloc(f32, @as(usize, W) * H * 3);
        defer gpa.free(a);
        area(img, w, h, a, W, H);
        try expectSha(c.get("area").?.string, a);
        for (c.get("lanczos").?.array.items) |lv| {
            const l = lv.array.items;
            const out = try gpa.alloc(f32, @as(usize, int(l[0])) * int(l[1]) * 3);
            defer gpa.free(out);
            try resizeTo(gpa, a, W, H, out, int(l[0]), int(l[1]));
            try expectSha(l[2].string, out);
        }
        const ce = c.get("center").?.array.items;
        try std.testing.expectEqual([2]u32{ int(ce[2]), int(ce[3]) }, cropCenter(W, H, int(ce[0]), int(ce[1])));
        const out = try gpa.alloc(f32, @as(usize, int(ce[0])) * int(ce[1]) * 3);
        defer gpa.free(out);
        try scaleCenter(gpa, a, W, H, out, int(ce[0]), int(ce[1]));
        try expectSha(ce[4].string, out);
    }
    var m96: []f32 = &.{};
    defer gpa.free(m96);
    for (im.get("bilinear").?.array.items) |bv| {
        const b = bv.array.items;
        const H = int(b[0]);
        const W = int(b[1]);
        const out = try gpa.alloc(f32, @as(usize, W) * H);
        try bilinear(mask, w, h, out, W, H);
        try expectSha(b[2].string, out);
        if (H == 96 and W == 64 and m96.len == 0) m96 = out else gpa.free(out);
    }
    const dec_u8 = try unhex(gpa, im.get("decoded").?.string);
    defer gpa.free(dec_u8);
    const dec = try gpa.alloc(f32, 96 * 64 * 3);
    defer gpa.free(dec);
    try loadRgba(dec_u8, 3, dec, null);
    const dest = try gpa.alloc(f32, 96 * 64 * 3);
    defer gpa.free(dest);
    try scaleCenter(gpa, img, w, h, dest, 64, 96);
    try expectSha(im.get("dest").?.string, dest);
    const comp = try gpa.alloc(f32, 96 * 64 * 3);
    defer gpa.free(comp);
    composite(dest, dec, m96, comp);
    try expectSha(im.get("composite").?.string, comp);
    const u8s = try gpa.alloc(u8, comp.len);
    defer gpa.free(u8s);
    for (u8s, comp) |*o, v| o.* = toU8(v);
    try expectSha(im.get("u8").?.string, u8s);
    try std.testing.expectError(error.MaskResizeTooSmall, bilinear(mask, w, h, dest[0 .. 40 * 30], 40, 30));
}

test "Python's round: ties to even" {
    for ([_][2]f64{ .{ 0.5, 0 }, .{ 1.5, 2 }, .{ 2.5, 2 }, .{ -2.5, -2 }, .{ 3.4999, 3 }, .{ 37.5, 38 }, .{ 38.5, 38 } }) |c|
        try std.testing.expectEqual(c[1], pyRound(c[0]));
}

pub fn unhex(gpa: std.mem.Allocator, s: []const u8) ![]u8 {
    const out = try gpa.alloc(u8, s.len / 2);
    errdefer gpa.free(out);
    _ = try std.fmt.hexToBytes(out, s);
    return out;
}

/// The test fixtures' digests: the SHA-256 of a slice's bytes (little-endian), in hex.
pub fn expectSha(want: []const u8, data: anytype) !void {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(data), &d, .{});
    const got = std.fmt.bytesToHex(d, .lower);
    try std.testing.expectEqualStrings(want, &got);
}

fn int(v: std.json.Value) u32 {
    return @intCast(v.integer);
}

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
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
