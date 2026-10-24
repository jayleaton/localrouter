//! Decode the capture's archived PNG and verify its source hash before reconstructing the requested canvas.
const std = @import("std");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");
const png = @import("../media/png_decode.zig");

pub fn canvas(gpa: std.mem.Allocator, io: std.Io, cap: *const qi.capture.Capture, width: u32, height: u32) !?[]f32 {
    const req = (cap.note("request") orelse return error.NoRequest).object;
    const image = req.get("image") orelse return null;
    if (image == .null) return null;
    if (image != .string or !std.mem.eql(u8, image.string, "input.png")) return error.BadCaptureImage;
    const path = try std.fs.path.join(gpa, &.{ cap.dir, "input.png" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(32 << 20));
    defer gpa.free(bytes);
    const want = req.get("image_sha256") orelse return error.NoImageHash;
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(sha, .lower), want.string)) return error.ImageHashMismatch;
    const decoded = try png.decode(gpa, bytes);
    defer decoded.deinit(gpa);
    const out = try gpa.alloc(f32, @as(usize, width) * height * 3);
    errdefer gpa.free(out);
    try h3.image.canvas(gpa, .{ .width = decoded.width, .height = decoded.height, .data = decoded.rgb }, width, height, out);
    return out;
}
