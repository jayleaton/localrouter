//! PNG decoder to 8-bit RGB and its alpha: every non-interlaced colour type (grey, RGB, palette, grey + alpha, RGBA) at the bit depths
//! the format allows (1, 2, 4, 8, 16; 16 keeps the high byte for RGB/alpha; grey clamps to 255, matching Pillow), zlib via std.compress.flate.
//! `rgb` drops the alpha (ComfyUI's LoadImage converts to "RGB", which discards it); `alpha` keeps it (255 for a type
//! without one; 16-bit keeps the high byte), for edits, which read RGBA. Chunk CRCs are checked. Not handled:
//! Adam7 interlacing (`error.Unsupported`); ancillary chunks (gamma, ICC, tRNS) are ignored. No colour management.
//! JPEG and WebP inputs reach it as PNG through `image_in.zig`.

const std = @import("std");
const flate = std.compress.flate;

pub const Error = error{ NotPng, Truncated, Unsupported, Corrupt, TooLarge, OutOfMemory };

pub const Decoded = struct {
    width: u32,
    height: u32,
    rgb: []u8, // [height, width, 3], owned by the caller's allocator
    alpha: []u8, // [height, width]

    pub fn deinit(d: Decoded, gpa: std.mem.Allocator) void {
        gpa.free(d.rgb);
        gpa.free(d.alpha);
    }

    /// [height, width, 4], in `a`.
    pub fn rgba(d: Decoded, a: std.mem.Allocator) ![]u8 {
        const out = try a.alloc(u8, d.alpha.len * 4);
        for (d.alpha, 0..) |al, i| {
            @memcpy(out[i * 4 ..][0..3], d.rgb[i * 3 ..][0..3]);
            out[i * 4 + 3] = al;
        }
        return out;
    }
};

const signature = "\x89PNG\r\n\x1a\n";
const max_pixels: u64 = 1 << 28; // 16384 x 16384

fn channels(color: u8) ?u32 {
    return switch (color) {
        0, 3 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        else => null,
    };
}

fn depthOk(color: u8, depth: u8) bool {
    return switch (color) {
        0 => depth == 1 or depth == 2 or depth == 4 or depth == 8 or depth == 16,
        3 => depth == 1 or depth == 2 or depth == 4 or depth == 8,
        else => depth == 8 or depth == 16,
    };
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p: i32 = @as(i32, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/// Undoes the row filters in place: `raw` is `h` rows of (1 filter byte + `stride` bytes), `bpp` the bytes per pixel (at least 1).
fn unfilter(raw: []u8, stride: usize, h: usize, bpp: usize) Error!void {
    var prev: ?[]const u8 = null;
    for (0..h) |y| {
        const row = raw[y * (stride + 1) ..][0 .. stride + 1];
        const ft = row[0];
        const cur = row[1..];
        for (0..stride) |i| {
            const a: u8 = if (i >= bpp) cur[i - bpp] else 0;
            const b: u8 = if (prev) |p| p[i] else 0;
            const c: u8 = if (i >= bpp and prev != null) prev.?[i - bpp] else 0;
            cur[i] +%= switch (ft) {
                0 => 0,
                1 => a,
                2 => b,
                3 => @as(u8, @intCast((@as(u16, a) + b) / 2)),
                4 => paeth(a, b, c),
                else => return error.Corrupt,
            };
        }
        prev = cur;
    }
}

/// One sample of a row at index `i` (a channel of a pixel) as 8 bits.
fn sample(cur: []const u8, i: usize, depth: u8) u8 {
    return switch (depth) {
        8 => cur[i],
        16 => cur[i * 2],
        else => blk: {
            const per: usize = 8 / depth;
            const shift: u3 = @intCast(8 - @as(usize, depth) - (i % per) * depth);
            const mask: u8 = (@as(u8, 1) << @intCast(depth)) - 1;
            break :blk (cur[i / per] >> shift) & mask;
        },
    };
}

pub fn decode(gpa: std.mem.Allocator, bytes: []const u8) Error!Decoded {
    if (bytes.len < 8 or !std.mem.eql(u8, bytes[0..8], signature)) return error.NotPng;
    var pos: usize = 8;
    var w: u32 = 0;
    var h: u32 = 0;
    var depth: u8 = 0;
    var color: u8 = 0;
    var have_hdr = false;
    var pal: [256][3]u8 = @splat(.{ 0, 0, 0 });
    var pal_n: usize = 0;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(gpa);
    var ended = false;
    var have_idat = false;
    var idat_closed = false;
    while (pos + 12 <= bytes.len) {
        const len: usize = std.mem.readInt(u32, bytes[pos..][0..4], .big);
        const tag = bytes[pos + 4 ..][0..4];
        if (pos + 12 + len > bytes.len) return error.Truncated;
        const data = bytes[pos + 8 ..][0..len];
        var crc: std.hash.Crc32 = .init();
        crc.update(tag);
        crc.update(data);
        if (crc.final() != std.mem.readInt(u32, bytes[pos + 8 + len ..][0..4], .big)) return error.Corrupt;
        pos += 12 + len;
        if (!have_hdr and !std.mem.eql(u8, tag, "IHDR")) return error.Corrupt;
        if (have_idat and !std.mem.eql(u8, tag, "IDAT")) idat_closed = true;
        if (std.mem.eql(u8, tag, "IHDR")) {
            if (have_hdr or len != 13) return error.Corrupt;
            w = std.mem.readInt(u32, data[0..4], .big);
            h = std.mem.readInt(u32, data[4..8], .big);
            depth = data[8];
            color = data[9];
            if (data[10] != 0 or data[11] != 0) return error.Unsupported;
            if (data[12] != 0) return error.Unsupported; // Adam7
            if (w == 0 or h == 0) return error.Corrupt;
            if (@as(u64, w) * h > max_pixels) return error.TooLarge;
            if (channels(color) == null or !depthOk(color, depth)) return error.Unsupported;
            have_hdr = true;
        } else if (std.mem.eql(u8, tag, "PLTE")) {
            if (have_idat or pal_n != 0 or len == 0 or len % 3 != 0 or len > 768 or color == 0 or color == 4) return error.Corrupt;
            if (color == 3 and len / 3 > (@as(usize, 1) << @intCast(depth))) return error.Corrupt;
            pal_n = len / 3;
            for (0..pal_n) |i| pal[i] = .{ data[i * 3], data[i * 3 + 1], data[i * 3 + 2] };
        } else if (std.mem.eql(u8, tag, "IDAT")) {
            if (idat_closed or (color == 3 and pal_n == 0)) return error.Corrupt;
            have_idat = true;
            idat.appendSlice(gpa, data) catch return error.OutOfMemory;
        } else if (std.mem.eql(u8, tag, "IEND")) {
            if (len != 0 or !have_idat) return error.Corrupt;
            if (pos != bytes.len) return error.Corrupt;
            ended = true;
            break;
        } else if (tag[0] & 0x20 == 0) {
            return error.Unsupported; // unknown critical chunk
        }
    }
    if (!have_hdr or !ended) return error.Truncated;
    if (color == 3 and pal_n == 0) return error.Corrupt;

    const ch = channels(color).?;
    const bits: usize = @as(usize, ch) * depth;
    const stride: usize = (@as(usize, w) * bits + 7) / 8;
    const raw = gpa.alloc(u8, (stride + 1) * h) catch return error.OutOfMemory;
    defer gpa.free(raw);
    {
        var in = std.Io.Reader.fixed(idat.items);
        var window: [flate.max_window_len]u8 = undefined;
        var d = flate.Decompress.init(&in, .zlib, &window);
        d.reader.readSliceAll(raw) catch return error.Corrupt;
        // readSliceAll may stop before the footer; force completion and verify the checksum (std flate only reads it).
        if (d.reader.takeByte()) |_| return error.Corrupt else |err| {
            if (err != error.EndOfStream or d.err != null) return error.Corrupt;
        }
        if (d.container_metadata.zlib.adler != std.hash.Adler32.hash(raw)) return error.Corrupt;
        if (in.bufferedLen() != 0) return error.Corrupt;
    }
    try unfilter(raw, stride, h, @max(1, bits / 8));

    const rgb = gpa.alloc(u8, @as(usize, w) * h * 3) catch return error.OutOfMemory;
    errdefer gpa.free(rgb);
    const alpha = gpa.alloc(u8, @as(usize, w) * h) catch return error.OutOfMemory;
    errdefer gpa.free(alpha);
    @memset(alpha, 255);
    for (0..h) |y| {
        const cur = raw[y * (stride + 1) + 1 ..][0..stride];
        for (0..w) |x| {
            const o = (y * w + x) * 3;
            if (color == 4) alpha[y * w + x] = sample(cur, x * ch + 1, depth);
            if (color == 6) alpha[y * w + x] = sample(cur, x * ch + 3, depth);
            switch (color) {
                0, 4 => {
                    var g = sample(cur, x * ch, depth);
                    if (color == 0 and depth == 16) g = @intCast(@min(255, std.mem.readInt(u16, cur[x * 2 ..][0..2], .big)));
                    if (color == 0 and depth < 8) g = @intCast(@as(u16, g) * 255 / ((@as(u16, 1) << @intCast(depth)) - 1));
                    rgb[o] = g;
                    rgb[o + 1] = g;
                    rgb[o + 2] = g;
                },
                3 => {
                    const idx = sample(cur, x, depth);
                    if (idx >= pal_n) return error.Corrupt;
                    rgb[o] = pal[idx][0];
                    rgb[o + 1] = pal[idx][1];
                    rgb[o + 2] = pal[idx][2];
                },
                else => {
                    rgb[o] = sample(cur, x * ch, depth);
                    rgb[o + 1] = sample(cur, x * ch + 1, depth);
                    rgb[o + 2] = sample(cur, x * ch + 2, depth);
                },
            }
        }
    }
    return .{ .width = w, .height = h, .rgb = rgb, .alpha = alpha };
}

test "round trip through the encoder: RGB and RGBA" {
    const png = @import("png.zig");
    const gpa = std.testing.allocator;
    const w = 37;
    const h = 11;
    var px: [w * h * 4]u8 = undefined;
    for (&px, 0..) |*p, i| p.* = @truncate(i *% 2654435761 >> 7);
    for ([_]png.Format{ .rgb8, .rgba8 }) |fmt| {
        const n: usize = if (fmt == .rgb8) 3 else 4;
        var out: std.Io.Writer.Allocating = .init(gpa);
        defer out.deinit();
        try png.encode(&out.writer, px[0 .. w * h * n], w, h, fmt, .{});
        const d = try decode(gpa, out.written());
        defer d.deinit(gpa);
        try std.testing.expectEqual(@as(u32, w), d.width);
        try std.testing.expectEqual(@as(u32, h), d.height);
        for (0..w * h) |i| try std.testing.expectEqualSlices(u8, px[i * n ..][0..3], d.rgb[i * 3 ..][0..3]);
        for (0..w * h) |i| try std.testing.expectEqual(if (n == 4) px[i * 4 + 3] else 255, d.alpha[i]);
    }
}

test "rejects garbage and a damaged CRC" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.NotPng, decode(gpa, "not a png at all"));
    const png = @import("png.zig");
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const px = [_]u8{ 1, 2, 3, 4, 5, 6 };
    try png.encode(&out.writer, &px, 2, 1, .rgb8, .{});
    const bad = try gpa.dupe(u8, out.written());
    defer gpa.free(bad);
    bad[20] ^= 0xff; // inside IHDR
    try std.testing.expectError(error.Corrupt, decode(gpa, bad));
}

test "sub-byte grey samples" {
    const row = [_]u8{0b1011_0000};
    try std.testing.expectEqual(@as(u8, 1), sample(&row, 0, 1));
    try std.testing.expectEqual(@as(u8, 0), sample(&row, 1, 1));
    try std.testing.expectEqual(@as(u8, 2), sample(&row, 0, 2));
    try std.testing.expectEqual(@as(u8, 3), sample(&row, 1, 2));
    try std.testing.expectEqual(@as(u8, 0xb), sample(&row, 0, 4));
}

test "palette packed 2-bit" {
    const d = try decode(std.testing.allocator, "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x03\x00\x00\x00\x01\x02\x03\x00\x00\x00\x66\x8e\xfc\x27\x00\x00\x00\x09\x50\x4c\x54\x45\x01\x02\x03\x04\x05\x06\x07\x08\x09\x25\x85\x56\xf0\x00\x00\x00\x0a\x49\x44\x41\x54\x78\x9c\x63\x90\x00\x00\x00\x1a\x00\x19\x2d\x88\xf4\x36\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82");
    defer d.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, d.rgb);
}

test "grey16 saturates" {
    const d = try decode(std.testing.allocator, "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x03\x00\x00\x00\x01\x10\x00\x00\x00\x00\x6e\x1b\x97\x2b\x00\x00\x00\x0f\x49\x44\x41\x54\x78\x9c\x63\x60\xa8\x67\x64\xf8\xff\x1f\x00\x05\x83\x02\x7f\xb0\x77\x81\x51\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82");
    defer d.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u8, &.{ 127, 127, 127, 255, 255, 255, 255, 255, 255 }, d.rgb);
}

test "Paeth RGB scanlines" {
    const d = try decode(std.testing.allocator, "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00\xfd\xd4\x9a\x73\x00\x00\x00\x16\x49\x44\x41\x54\x78\x9c\x63\x60\x64\x62\x66\x61\x65\x63\x61\x63\x63\x63\x66\x66\x06\x00\x01\x61\x00\x35\x06\x21\x4c\xa9\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82");
    defer d.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, d.rgb);
}
