//! An input image (an edit's reference or mask, a video's first frame) from the job directory as 8-bit RGBA. PNG is
//! decoded here (`png_decode.zig`: every non-interlaced type). JPEG, WebP and interlaced PNG, which the API also takes,
//! are converted to RGBA PNG by ffmpeg first (one frame, no colour management), so every model reads the same pixels
//! from the same file. The converted file is written beside the input and kept with the job.

const std = @import("std");
const png = @import("png_decode.zig");

pub const Image = struct {
    width: u32,
    height: u32,
    rgba: []u8, // [height, width, 4]
};

/// Reads `name` (a file name in `dir`, no path) into `a`.
pub fn load(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) !Image {
    if (name.len == 0 or !std.mem.eql(u8, name, std.fs.path.basename(name))) return error.BadName;
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 << 20));
    const d = png.decode(a, bytes) catch |err| switch (err) {
        error.NotPng, error.Unsupported => try png.decode(a, try convert(io, a, path)),
        else => return err,
    };
    return .{ .width = d.width, .height = d.height, .rgba = try d.rgba(a) };
}

/// `path` as an RGBA PNG through ffmpeg, its bytes.
fn convert(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]u8 {
    const out = try std.fmt.allocPrint(a, "{s}.rgba.png", .{path});
    var child = try std.process.spawn(io, .{
        .argv = &.{ "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-frames:v", "1", "-pix_fmt", "rgba", "-f", "image2", "-c:v", "png", out },
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .inherit,
    });
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.ConvertFailed;
    return std.Io.Dir.cwd().readFileAlloc(io, out, a, .limited(256 << 20));
}

test "a PNG loads as RGBA with its alpha; a name with a path is refused" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rnd: [8]u8 = undefined;
    io.random(&rnd);
    const dir = try std.fmt.allocPrint(a, "/tmp/localrouter-image-in-{x}", .{std.mem.readInt(u64, &rnd, .little)});
    try std.Io.Dir.cwd().createDirPath(io, dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    var out: std.Io.Writer.Allocating = .init(a);
    try @import("png.zig").encode(&out.writer, &.{ 1, 2, 3, 4, 5, 6, 7, 0 }, 2, 1, .rgba8, .{});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "ref_0.png" }), .data = out.written() });
    const img = try load(io, a, dir, "ref_0.png");
    try std.testing.expectEqual(@as(u32, 2), img.width);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 0 }, img.rgba);
    try std.testing.expectError(error.BadName, load(io, a, dir, "../ref_0.png"));
}
