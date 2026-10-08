//! PNG encoder: 8-bit RGB/RGBA, Up-filtered rows, zlib via std.compress.flate, no allocator needed.
const std = @import("std");
const Writer = std.Io.Writer;
const flate = std.compress.flate;

/// Pixel layout of the input buffer (8 bits per channel).
pub const Format = enum { rgb8, rgba8 };
/// Compression effort, mapped onto std flate levels.
pub const Level = enum { fastest, fast, default, best };
/// A tEXt chunk (key must be 1-79 Latin-1 bytes, value is Latin-1).
pub const Text = struct { key: []const u8, value: []const u8 };
/// Encoder options.
pub const Options = struct { level: Level = .fast, text: []const Text = &.{} };

const signature = "\x89PNG\r\n\x1a\n";

fn flateOptions(l: Level) flate.Compress.Options {
    return switch (l) {
        .fastest => .level_1,
        .fast => .level_3,
        .default => .level_6,
        .best => .level_9,
    };
}

/// Chunk under construction: tracks CRC over tag and data.
const Chunk = struct {
    out: *Writer,
    crc: std.hash.Crc32,

    fn begin(out: *Writer, tag: *const [4]u8, len: usize) !Chunk {
        try out.writeInt(u32, @intCast(len), .big);
        try out.writeAll(tag);
        var c: Chunk = .{ .out = out, .crc = .init() };
        c.crc.update(tag);
        return c;
    }
    fn put(c: *Chunk, bytes: []const u8) !void {
        c.crc.update(bytes);
        try c.out.writeAll(bytes);
    }
    fn end(c: *Chunk) !void {
        try c.out.writeInt(u32, c.crc.final(), .big);
    }
};

/// Writer that wraps everything written to it into IDAT chunks (one per drain).
const IdatWriter = struct {
    writer: Writer,
    out: *Writer,

    fn drain(w: *Writer, data: []const []const u8, splat: usize) Writer.Error!usize {
        const self: *IdatWriter = @fieldParentPtr("writer", w);
        const last = data[data.len - 1];
        var total: usize = 0;
        for (data[0 .. data.len - 1]) |d| total += d.len;
        const consumed = total + last.len * splat;
        if (w.end + consumed == 0) return 0;
        var c = try Chunk.begin(self.out, "IDAT", w.end + consumed);
        try c.put(w.buffer[0..w.end]);
        for (data[0 .. data.len - 1]) |d| try c.put(d);
        for (0..splat) |_| try c.put(last);
        try c.end();
        w.end = 0;
        return consumed;
    }
};

/// Encodes tightly packed `pixels` (width*height*channels bytes) as a PNG into `w`; caller flushes `w`.
/// Uses ~100 KiB of stack (flate window + compressor state); every row uses filter 2 (Up), a single
/// linear pass that auto-vectorizes and does well on photographic/noisy content.
pub fn encode(w: *Writer, pixels: []const u8, width: u32, height: u32, format: Format, opts: Options) !void {
    const channels: usize = if (format == .rgb8) 3 else 4;
    const stride = @as(usize, width) * channels;
    if (width == 0 or height == 0 or pixels.len != stride * height) return error.InvalidDimensions;

    try w.writeAll(signature);
    var ihdr = try Chunk.begin(w, "IHDR", 13);
    var hdr: [13]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], width, .big);
    std.mem.writeInt(u32, hdr[4..8], height, .big);
    hdr[8..13].* = .{ 8, if (format == .rgb8) 2 else 6, 0, 0, 0 }; // depth, color type, deflate, filter, no interlace
    try ihdr.put(&hdr);
    try ihdr.end();

    for (opts.text) |t| {
        if (t.key.len == 0 or t.key.len > 79) return error.InvalidTextKey;
        var c = try Chunk.begin(w, "tEXt", t.key.len + 1 + t.value.len);
        try c.put(t.key);
        try c.put(&.{0});
        try c.put(t.value);
        try c.end();
    }

    var idat_buf: [16 * 1024]u8 = undefined;
    var idat: IdatWriter = .{
        .writer = .{ .vtable = &.{ .drain = IdatWriter.drain }, .buffer = &idat_buf },
        .out = w,
    };
    var window: [flate.max_window_len]u8 = undefined;
    var comp = try flate.Compress.init(&idat.writer, &window, .zlib, flateOptions(opts.level));

    var tmp: [4096]u8 = undefined;
    for (0..height) |y| {
        const row = pixels[y * stride ..][0..stride];
        try comp.writer.writeAll(&.{2});
        var x: usize = 0;
        while (x < stride) {
            const n = @min(tmp.len, stride - x);
            if (y == 0) {
                @memcpy(tmp[0..n], row[x..][0..n]); // first row: Up against zeros
            } else {
                const prev = pixels[(y - 1) * stride + x ..][0..n];
                for (tmp[0..n], row[x..][0..n], prev) |*o, cur, p| o.* = cur -% p;
            }
            try comp.writer.writeAll(tmp[0..n]);
            x += n;
        }
    }
    try comp.finish();
    try idat.writer.flush();

    var iend = try Chunk.begin(w, "IEND", 0);
    try iend.end();
}

const testing = std.testing;

fn gradient(a: std.mem.Allocator, w: u32, h: u32, ch: usize) ![]u8 {
    const px = try a.alloc(u8, w * h * ch);
    for (0..h) |y| for (0..w) |x| for (0..ch) |c| {
        px[(y * w + x) * ch + c] = @truncate(x * 7 + y * 3 + c * 50 + (x * y) / 5);
    };
    return px;
}

/// Parses chunks (verifying CRCs), inflates IDAT, un-filters, and compares with `px`.
fn roundTrip(w: u32, h: u32, format: Format, opts: Options) !void {
    const a = testing.allocator;
    const ch: usize = if (format == .rgb8) 3 else 4;
    const px = try gradient(a, w, h, ch);
    defer a.free(px);
    var aw: Writer.Allocating = .init(a);
    defer aw.deinit();
    try encode(&aw.writer, px, w, h, format, opts);
    const png = aw.written();

    try testing.expectEqualStrings(signature, png[0..8]);
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(a);
    var pos: usize = 8;
    var texts: usize = 0;
    var saw_iend = false;
    while (pos < png.len) {
        const len = std.mem.readInt(u32, png[pos..][0..4], .big);
        const tag = png[pos + 4 ..][0..4];
        const body = png[pos + 8 ..][0..len];
        const crc = std.mem.readInt(u32, png[pos + 8 + len ..][0..4], .big);
        try testing.expectEqual(std.hash.Crc32.hash(png[pos + 4 ..][0 .. 4 + len]), crc);
        if (pos == 8) {
            try testing.expectEqualStrings("IHDR", tag);
            try testing.expectEqual(w, std.mem.readInt(u32, body[0..4], .big));
            try testing.expectEqual(h, std.mem.readInt(u32, body[4..8], .big));
            try testing.expectEqual(@as(u8, if (ch == 3) 2 else 6), body[9]);
        }
        if (std.mem.eql(u8, tag, "IDAT")) try idat.appendSlice(a, body);
        if (std.mem.eql(u8, tag, "tEXt")) texts += 1;
        if (std.mem.eql(u8, tag, "IEND")) saw_iend = true;
        pos += 12 + len;
    }
    try testing.expect(saw_iend);
    try testing.expectEqual(opts.text.len, texts);

    var in: std.Io.Reader = .fixed(idat.items);
    var win: [flate.max_window_len]u8 = undefined;
    var dec: flate.Decompress = .init(&in, .zlib, &win);
    const raw = try a.alloc(u8, (w * ch + 1) * h);
    defer a.free(raw);
    try dec.reader.readSliceAll(raw);
    const stride = w * ch;
    for (0..h) |y| {
        const line = raw[y * (stride + 1) ..][0 .. stride + 1];
        try testing.expectEqual(@as(u8, 2), line[0]);
        for (line[1..], 0..) |f, i| {
            const up = if (y == 0) 0 else px[(y - 1) * stride + i];
            try testing.expectEqual(px[y * stride + i], f +% up);
        }
    }
}

test "rgb and rgba round trip" {
    try roundTrip(37, 29, .rgb8, .{});
    try roundTrip(64, 64, .rgba8, .{ .level = .best });
    try roundTrip(1500, 3, .rgb8, .{ .level = .fastest }); // row longer than scratch block
}

test "tEXt chunks and multi-chunk IDAT" {
    const t = [_]Text{ .{ .key = "prompt", .value = "a cat" }, .{ .key = "seed", .value = "42" } };
    try roundTrip(300, 200, .rgba8, .{ .level = .default, .text = &t });
}

test "invalid dimensions" {
    var buf: [64]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try testing.expectError(error.InvalidDimensions, encode(&w, &.{}, 2, 2, .rgb8, .{}));
}
