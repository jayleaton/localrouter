//! PLY headers (the Stanford polygon format a 3D Gaussian splat ships in): read one, and label one as machine-generated
//! with `comment` lines, the format's own metadata. The body (the vertices) is never touched.

const std = @import("std");
const marker = @import("marker.zig");

/// Headers longer than this are refused (a real one is a few hundred bytes).
pub const max_header = 64 << 10;

pub const Format = enum { ascii, binary_little_endian, binary_big_endian };

/// What a header says: its length in bytes (through `end_header\n`), the format, and the first element's count and
/// row size in bytes (0 when a property is a list or of an unknown type, so the size cannot be known).
pub const Header = struct {
    len: usize,
    format: Format,
    element: []const u8, // the first element's name ("vertex" for a splat)
    count: u64,
    stride: u64,
    properties: u32, // of the first element
    elements: u32,

    /// The bytes a binary file with only this element takes; null when it cannot be known.
    pub fn bodyBytes(h: Header) ?u64 {
        if (h.format == .ascii or h.stride == 0 or h.elements != 1) return null;
        return std.math.mul(u64, h.count, h.stride) catch null;
    }
};

/// The header at the start of `bytes` (which must hold all of it); null when it is not a PLY header.
pub fn header(bytes: []const u8) ?Header {
    if (!std.mem.startsWith(u8, bytes, "ply\n")) return null;
    const end_tag = "end_header\n";
    const end = std.mem.indexOf(u8, bytes[0..@min(bytes.len, max_header)], end_tag) orelse return null;
    var h: Header = .{ .len = end + end_tag.len, .format = .ascii, .element = "", .count = 0, .stride = 0, .properties = 0, .elements = 0 };
    var known_format = false;
    var sized = true;
    var lines = std.mem.splitScalar(u8, bytes[4..end], '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        const key = words.next() orelse continue;
        if (eql(key, "format")) {
            h.format = std.meta.stringToEnum(Format, words.next() orelse return null) orelse return null;
            known_format = true;
        } else if (eql(key, "element")) {
            h.elements += 1;
            if (h.elements > 1) continue;
            h.element = words.next() orelse return null;
            h.count = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
        } else if (eql(key, "property")) {
            if (h.elements != 1) continue;
            h.properties += 1;
            const t = words.next() orelse return null;
            const size = typeSize(t);
            if (size == 0) sized = false;
            h.stride += size;
        } else if (!eql(key, "comment") and !eql(key, "obj_info")) return null;
    }
    if (!known_format or h.elements == 0) return null;
    if (!sized) h.stride = 0;
    return h;
}

/// The size of a scalar property type in bytes; 0 for a list or an unknown type.
fn typeSize(t: []const u8) u64 {
    const sizes = [_]struct { []const u8, u64 }{
        .{ "char", 1 },  .{ "uchar", 1 },   .{ "int8", 1 },   .{ "uint8", 1 },
        .{ "short", 2 }, .{ "ushort", 2 },  .{ "int16", 2 },  .{ "uint16", 2 },
        .{ "int", 4 },   .{ "uint", 4 },    .{ "int32", 4 },  .{ "uint32", 4 },
        .{ "float", 4 }, .{ "float32", 4 }, .{ "double", 8 }, .{ "float64", 8 },
    };
    for (sizes) |s| if (eql(s[0], t)) return s[1];
    return 0;
}

/// `data` (a whole PLY file) with the machine-generated label as `comment` lines right after the `format` line, in
/// `gpa`; null when its header already carries one (or it has no header).
pub fn mark(gpa: std.mem.Allocator, data: []const u8, m: marker.Marker) !?[]u8 {
    const h = header(data) orelse return null;
    if (marker.present(data[0..h.len])) return null;
    const fmt_at = std.mem.indexOf(u8, data[0..h.len], "\nformat ") orelse return null;
    const after = (std.mem.indexOfScalarPos(u8, data, fmt_at + 1, '\n') orelse return null) + 1;
    var c: [marker.comment_max]u8 = undefined;
    var label_buf: [marker.comment_max + 2 * 64 + 256]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "comment {s}\ncomment DigitalSourceType {s}\ncomment CreatorTool {s}\n", .{ marker.comment(&c, m), marker.sourceType(m), marker.software });
    return try std.mem.concat(gpa, u8, &.{ data[0..after], label, data[after..] });
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

const testing = std.testing;

/// A binary splat PLY of `n` rows with the 17 float properties a 3DGS file has, all zero, for tests.
pub fn testSplat(a: std.mem.Allocator, n: u32) ![]u8 {
    const props = [_][]const u8{ "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };
    var w: std.Io.Writer.Allocating = .init(a);
    errdefer w.deinit();
    try w.writer.print("ply\nformat binary_little_endian 1.0\nelement vertex {d}\n", .{n});
    for (props) |p| try w.writer.print("property float {s}\n", .{p});
    try w.writer.writeAll("end_header\n");
    try w.writer.splatByteAll(0, @as(usize, n) * props.len * 4);
    return w.toOwnedSlice();
}

test "a splat header: format, count, row size; garbage and lists are not sized" {
    const a = testing.allocator;
    const data = try testSplat(a, 3);
    defer a.free(data);
    const h = header(data).?;
    try testing.expectEqual(Format.binary_little_endian, h.format);
    try testing.expectEqualStrings("vertex", h.element);
    try testing.expectEqual(@as(u64, 3), h.count);
    try testing.expectEqual(@as(u64, 68), h.stride);
    try testing.expectEqual(@as(u32, 17), h.properties);
    try testing.expectEqual(@as(?u64, 204), h.bodyBytes());
    try testing.expectEqual(data.len, h.len + 204);
    try testing.expect(header("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\n") == null); // no end
    try testing.expect(header("plz\nformat ascii 1.0\nelement vertex 1\nend_header\n") == null);
    try testing.expect(header("ply\nformat weird 1.0\nelement vertex 1\nend_header\n") == null);
    const mesh = header("ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n").?;
    try testing.expectEqual(@as(u32, 2), mesh.elements);
    try testing.expect(mesh.bodyBytes() == null);
}

test "mark adds the label after the format line once; the body is unchanged" {
    const a = testing.allocator;
    const data = try testSplat(a, 2);
    defer a.free(data);
    const marked = (try mark(a, data, .{ .model = "idol", .edit = true })).?;
    defer a.free(marked);
    const h = header(marked).?;
    try testing.expect(std.mem.startsWith(u8, marked, "ply\nformat binary_little_endian 1.0\ncomment AI-edited (machine-generated) from an input image by model idol with LocalRouter\ncomment DigitalSourceType " ++ marker.composite ++ "\n"));
    try testing.expectEqual(@as(u64, 2), h.count);
    try testing.expectEqualSlices(u8, data[header(data).?.len..], marked[h.len..]);
    try testing.expect(try mark(a, marked, .{ .model = "idol" }) == null); // already labelled
}
