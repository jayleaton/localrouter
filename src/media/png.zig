//! PNG encoder: 8-bit RGB/RGBA, Up-filtered rows, zlib via std.compress.flate, no allocator needed. Writes the
//! machine-generated label (`marker`) on request, and adds it to a PNG made elsewhere (`mark`).
const std = @import("std");
const Writer = std.Io.Writer;
const flate = std.compress.flate;
const marker = @import("marker.zig");

/// Pixel layout of the input buffer (8 bits per channel).
pub const Format = enum { rgb8, rgba8 };
/// Compression effort, mapped onto std flate levels.
pub const Level = enum { fastest, fast, default, best };
/// A tEXt chunk (key must be 1-79 Latin-1 bytes, value is Latin-1).
pub const Text = struct { key: []const u8, value: []const u8 };
/// Encoder options. `marker`: the machine-generated label, written right after IHDR (as `mark` inserts it).
pub const Options = struct { level: Level = .fast, text: []const Text = &.{}, marker: ?marker.Marker = null };

pub const signature = "\x89PNG\r\n\x1a\n";
/// The XMP packet's iTXt keyword.
const xmp_key = "XML:com.adobe.xmp";

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

    if (opts.marker) |m| try writeMarker(w, m, true);
    for (opts.text) |t| try writeText(w, t);

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

fn writeText(w: *Writer, t: Text) !void {
    if (t.key.len == 0 or t.key.len > 79) return error.InvalidTextKey;
    var c = try Chunk.begin(w, "tEXt", t.key.len + 1 + t.value.len);
    try c.put(t.key);
    try c.put(&.{0});
    try c.put(t.value);
    try c.end();
}

/// The label's chunks: tEXt `Software` and `Comment`, then (with `with_xmp`) the XMP packet as an uncompressed iTXt.
fn writeMarker(w: *Writer, m: marker.Marker, with_xmp: bool) !void {
    var cbuf: [marker.comment_max]u8 = undefined;
    try writeText(w, .{ .key = "Software", .value = marker.software });
    try writeText(w, .{ .key = "Comment", .value = marker.comment(&cbuf, m) });
    if (!with_xmp) return;
    var xbuf: [marker.xmp_max]u8 = undefined;
    const packet = marker.xmp(&xbuf, m);
    // keyword, NUL, compression flag and method (none), empty language tag and translated keyword (two NULs)
    var c = try Chunk.begin(w, "iTXt", xmp_key.len + 5 + packet.len);
    try c.put(xmp_key);
    try c.put(&.{ 0, 0, 0, 0, 0 });
    try c.put(packet);
    try c.end();
}

/// Adds the label to a PNG made elsewhere (a `cmd` tool's): the chunks go right after IHDR, everything else is copied
/// byte for byte. Null when a text chunk already carries a label; the XMP packet is left out when the file has one.
/// Caller frees the result.
pub fn mark(gpa: std.mem.Allocator, data: []const u8, m: marker.Marker) !?[]u8 {
    const ihdr_end = signature.len + 12 + 13;
    if (data.len < ihdr_end or !std.mem.eql(u8, data[0..signature.len], signature) or !std.mem.eql(u8, data[12..16], "IHDR")) return error.NotPng;
    var has_xmp = false;
    var pos: usize = signature.len;
    while (pos + 12 <= data.len) {
        const len = std.mem.readInt(u32, data[pos..][0..4], .big);
        if (len > data.len - pos - 12) return error.NotPng;
        const tag = data[pos + 4 ..][0..4];
        const body = data[pos + 8 ..][0..len];
        if (std.mem.eql(u8, tag, "tEXt") or std.mem.eql(u8, tag, "iTXt")) {
            if (marker.present(body)) return null;
            if (std.mem.startsWith(u8, body, xmp_key ++ "\x00")) has_xmp = true;
        }
        if (std.mem.eql(u8, tag, "IEND")) break;
        pos += 12 + len;
    }
    var out: Writer.Allocating = try .initCapacity(gpa, data.len + marker.comment_max + marker.xmp_max);
    errdefer out.deinit();
    try out.writer.writeAll(data[0..ihdr_end]);
    try writeMarker(&out.writer, m, !has_xmp);
    try out.writer.writeAll(data[ihdr_end..]);
    return try out.toOwnedSlice();
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
    var labels: usize = 0;
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
        if (std.mem.eql(u8, tag, "iTXt") and std.mem.startsWith(u8, body, xmp_key ++ "\x00") and marker.present(body)) labels += 1;
        if (std.mem.eql(u8, tag, "IEND")) saw_iend = true;
        pos += 12 + len;
    }
    try testing.expect(saw_iend);
    try testing.expectEqual(opts.text.len + if (opts.marker != null) @as(usize, 2) else 0, texts); // Software, Comment
    try testing.expectEqual(@as(usize, if (opts.marker != null) 1 else 0), labels);

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

/// The concatenated IDAT payloads of a PNG.
fn idatBytes(a: std.mem.Allocator, data: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = signature.len;
    while (pos < data.len) {
        const len = std.mem.readInt(u32, data[pos..][0..4], .big);
        if (std.mem.eql(u8, data[pos + 4 ..][0..4], "IDAT")) try out.appendSlice(a, data[pos + 8 ..][0..len]);
        pos += 12 + len;
    }
    return out.toOwnedSlice(a);
}

test "machine-generated label: chunks valid, pixel data identical, mark matches the writer and is idempotent" {
    const a = testing.allocator;
    const m: marker.Marker = .{ .model = "qwen-image", .edit = true };
    const t = [_]Text{.{ .key = "seed", .value = "42" }};
    try roundTrip(97, 61, .rgb8, .{ .text = &t, .marker = m }); // CRCs, chunk counts, pixels

    const px = try gradient(a, 97, 61, 3);
    defer a.free(px);
    var plain: Writer.Allocating = .init(a);
    defer plain.deinit();
    try encode(&plain.writer, px, 97, 61, .rgb8, .{ .text = &t });
    var labelled: Writer.Allocating = .init(a);
    defer labelled.deinit();
    try encode(&labelled.writer, px, 97, 61, .rgb8, .{ .text = &t, .marker = m });
    const ia = try idatBytes(a, plain.written());
    defer a.free(ia);
    const ib = try idatBytes(a, labelled.written());
    defer a.free(ib);
    try testing.expectEqualSlices(u8, ia, ib); // the compressed pixels, byte for byte

    const marked = (try mark(a, plain.written(), m)).?;
    defer a.free(marked);
    try testing.expectEqualSlices(u8, labelled.written(), marked);
    try testing.expect((try mark(a, marked, m)) == null);
    var cbuf: [marker.comment_max]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, marked, "Comment\x00" ++ "AI-edited") != null);
    try testing.expect(std.mem.indexOf(u8, marked, marker.comment(&cbuf, m)) != null);
    try testing.expect(std.mem.indexOf(u8, marked, marker.composite) != null);
    try testing.expectError(error.NotPng, mark(a, "not a png at all, just some bytes here", m));
}

test "invalid dimensions" {
    var buf: [64]u8 = undefined;
    var w: Writer = .fixed(&buf);
    try testing.expectError(error.InvalidDimensions, encode(&w, &.{}, 2, 2, .rgb8, .{}));
}
