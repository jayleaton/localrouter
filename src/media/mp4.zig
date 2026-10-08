//! MP4 output through ffmpeg: rgb24 frames piped to its stdin, audio from a WAV file we write first.
//! ffmpeg ships in the image; H.264 on the CPU (veryfast) costs seconds next to minutes of sampling.

const std = @import("std");
const Io = std.Io;

pub const Options = struct {
    width: u32,
    height: u32,
    fps: u32,
    wav: ?[]const u8 = null, // audio track, muxed as AAC
    crf: u8 = 18,
};

pub const Writer = struct {
    io: Io,
    child: std.process.Child,
    frame_bytes: usize,

    /// Starts ffmpeg writing `path`; send exactly width x height x 3 bytes a frame.
    pub fn open(io: Io, gpa: std.mem.Allocator, path: []const u8, o: Options) !Writer {
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(gpa);
        var size_buf: [32]u8 = undefined;
        var fps_buf: [16]u8 = undefined;
        var crf_buf: [8]u8 = undefined;
        const size = try std.fmt.bufPrint(&size_buf, "{d}x{d}", .{ o.width, o.height });
        const fps = try std.fmt.bufPrint(&fps_buf, "{d}", .{o.fps});
        const crf = try std.fmt.bufPrint(&crf_buf, "{d}", .{o.crf});
        try argv.appendSlice(gpa, &.{ "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", fps, "-i", "pipe:0" });
        if (o.wav) |w| try argv.appendSlice(gpa, &.{ "-i", w });
        try argv.appendSlice(gpa, &.{ "-c:v", "libx264", "-preset", "veryfast", "-crf", crf, "-pix_fmt", "yuv420p", "-movflags", "+faststart" });
        if (o.wav != null) try argv.appendSlice(gpa, &.{ "-c:a", "aac", "-b:a", "192k", "-shortest" });
        try argv.append(gpa, path);
        const child = try std.process.spawn(io, .{ .argv = argv.items, .stdin = .pipe, .stdout = .ignore, .stderr = .inherit });
        return .{ .io = io, .child = child, .frame_bytes = @as(usize, o.width) * o.height * 3 };
    }

    pub fn frame(w: *Writer, rgb: []const u8) !void {
        std.debug.assert(rgb.len == w.frame_bytes);
        w.child.stdin.?.writeStreamingAll(w.io, rgb) catch return error.EncoderFailed;
    }

    /// Closes the input and waits; an ffmpeg failure is `error.EncoderFailed`.
    pub fn finish(w: *Writer) !void {
        w.child.stdin.?.close(w.io);
        w.child.stdin = null;
        const term = try w.child.wait(w.io);
        if (term != .exited or term.exited != 0) return error.EncoderFailed;
    }

    /// Kills ffmpeg after a failure elsewhere.
    pub fn abort(w: *Writer) void {
        w.child.kill(w.io);
    }
};

/// Writes interleaved f32 samples in [-1, 1] as 16-bit PCM WAV.
pub fn writeWav(io: Io, path: []const u8, samples: []const f32, rate: u32, channels: u16) !void {
    var file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;
    const data: u32 = @intCast(samples.len * 2);
    try w.writeAll("RIFF");
    try w.writeInt(u32, 36 + data, .little);
    try w.writeAll("WAVEfmt ");
    try w.writeInt(u32, 16, .little);
    try w.writeInt(u16, 1, .little); // PCM
    try w.writeInt(u16, channels, .little);
    try w.writeInt(u32, rate, .little);
    try w.writeInt(u32, rate * channels * 2, .little);
    try w.writeInt(u16, channels * 2, .little);
    try w.writeInt(u16, 16, .little);
    try w.writeAll("data");
    try w.writeInt(u32, data, .little);
    for (samples) |s| try w.writeInt(i16, @intFromFloat(std.math.clamp(s, -1, 1) * 32767), .little);
    try w.flush();
}
