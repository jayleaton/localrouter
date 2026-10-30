//! MP4 output through ffmpeg: rgb24 frames piped to its stdin, audio from a WAV file we write first.
//! ffmpeg ships in the image; H.264 on the CPU (veryfast) costs seconds next to minutes of sampling.
//! The machine-generated label is two ffmpeg `-metadata` tags (`comment`, `description`: `©cmt` and `desc` in
//! `moov/udta/meta/ilst`); `mark` adds them to an MP4 made elsewhere by a `-c copy` remux.

const std = @import("std");
const Io = std.Io;
const marker = @import("marker.zig");

pub const Options = struct {
    width: u32,
    height: u32,
    fps: u32,
    wav: ?[]const u8 = null, // audio track, muxed as AAC
    crf: u8 = 18,
    /// End the file with the shorter stream (ffmpeg `-shortest`). False keeps every audio sample, as a model whose
    /// native mux writes its whole waveform past the last frame does.
    shortest: bool = true,
    marker: ?marker.Marker = null, // the machine-generated label
};

/// The label as ffmpeg output options: `-metadata comment=<comment> -metadata description=<IPTC source type>`.
const Label = struct {
    comment: [marker.comment_max + 8]u8 = undefined,
    description: [160]u8 = undefined,

    fn args(l: *Label, m: marker.Marker) [4][]const u8 {
        var c: [marker.comment_max]u8 = undefined;
        return .{
            "-metadata",
            std.fmt.bufPrint(&l.comment, "comment={s}", .{marker.comment(&c, m)}) catch unreachable,
            "-metadata",
            std.fmt.bufPrint(&l.description, "description={s}", .{marker.sourceType(m)}) catch unreachable,
        };
    }
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
        if (o.wav != null) {
            try argv.appendSlice(gpa, &.{ "-c:a", "aac", "-b:a", "192k" });
            if (o.shortest) try argv.append(gpa, "-shortest");
        }
        var label: Label = .{};
        if (o.marker) |m| try argv.appendSlice(gpa, &label.args(m));
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

/// Whether the MP4 at `path` already carries a label: its `moov` box names an IPTC trained-algorithmic-media type.
/// Reads the top-level box headers and the `moov` box only.
pub fn hasMarker(io: Io, gpa: std.mem.Allocator, path: []const u8) !bool {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const size = try f.length(io);
    var pos: u64 = 0;
    while (pos + 8 <= size) {
        var hdr: [16]u8 = undefined;
        if (try f.readPositionalAll(io, hdr[0..8], pos) < 8) return error.NotMp4;
        var len: u64 = std.mem.readInt(u32, hdr[0..4], .big);
        var head: u64 = 8;
        if (len == 1) { // a 64-bit size follows the type
            if (try f.readPositionalAll(io, hdr[8..16], pos + 8) < 8) return error.NotMp4;
            len = std.mem.readInt(u64, hdr[8..16], .big);
            head = 16;
        } else if (len == 0) len = size - pos; // runs to the end of the file
        if (len < head or len > size - pos) return error.NotMp4;
        if (std.mem.eql(u8, hdr[4..8], "moov")) {
            if (len > 256 << 20) return error.NotMp4;
            const body = try gpa.alloc(u8, @intCast(len - head));
            defer gpa.free(body);
            if (try f.readPositionalAll(io, body, pos + head) < body.len) return error.NotMp4;
            return marker.present(body);
        }
        pos += len;
    }
    return error.NotMp4; // no moov
}

/// Adds the label to the MP4 at `path` (a `cmd` tool's): ffmpeg copies every stream as is (`-map 0 -c copy`) and the
/// existing metadata, sets the two tags, and the result replaces the file. Sample data is unchanged; only the
/// container's boxes are rewritten. An ffmpeg failure is `error.EncoderFailed` and leaves the file as it was.
pub fn mark(io: Io, gpa: std.mem.Allocator, path: []const u8, m: marker.Marker) !void {
    const tmp = try std.mem.concat(gpa, u8, &.{ path, ".marking.mp4" });
    defer gpa.free(tmp);
    errdefer Io.Dir.cwd().deleteFile(io, tmp) catch {};
    var label: Label = .{};
    const lb = label.args(m);
    const argv = [_][]const u8{ "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-map", "0", "-c", "copy", "-map_metadata", "0" } ++ lb ++ [_][]const u8{ "-movflags", "+faststart", tmp };
    var child = try std.process.spawn(io, .{ .argv = &argv, .stdin = .ignore, .stdout = .ignore, .stderr = .inherit });
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.EncoderFailed;
    try Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, io);
}

const testing = std.testing;

/// The payload of every `mdat` box: the coded samples.
fn mdatBytes(a: std.mem.Allocator, data: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (pos + 8 <= data.len) {
        const len = std.mem.readInt(u32, data[pos..][0..4], .big);
        if (len < 8) return error.NotMp4;
        if (std.mem.eql(u8, data[pos + 4 ..][0..4], "mdat")) try out.appendSlice(a, data[pos + 8 ..][0 .. len - 8]);
        pos += len;
    }
    return out.toOwnedSlice(a);
}

fn encodeTest(io: Io, a: std.mem.Allocator, path: []const u8, m: ?marker.Marker) !void {
    var w = try Writer.open(io, a, path, .{ .width = 32, .height = 32, .fps = 4, .marker = m });
    errdefer w.abort();
    var px: [32 * 32 * 3]u8 = undefined;
    for (0..8) |f| {
        for (&px, 0..) |*p, i| p.* = @truncate(i *% 7 +% f * 31);
        try w.frame(&px);
    }
    try w.finish();
}

test "machine-generated label: tags in moov, samples identical, mark matches and is idempotent (needs ffmpeg)" {
    const io = testing.io;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    defer a.free(dir);
    const plain = try std.fs.path.join(a, &.{ dir, "plain.mp4" });
    defer a.free(plain);
    const labelled = try std.fs.path.join(a, &.{ dir, "labelled.mp4" });
    defer a.free(labelled);
    const m: marker.Marker = .{ .model = "minimax-h3" };
    try encodeTest(io, a, plain, null);
    try encodeTest(io, a, labelled, m);

    try testing.expect(!try hasMarker(io, a, plain));
    try testing.expect(try hasMarker(io, a, labelled));
    const p = try tmp.dir.readFileAlloc(io, "plain.mp4", a, .unlimited);
    defer a.free(p);
    const l = try tmp.dir.readFileAlloc(io, "labelled.mp4", a, .unlimited);
    defer a.free(l);
    var c: [marker.comment_max]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, l, "\xa9cmt") != null);
    try testing.expect(std.mem.indexOf(u8, l, marker.comment(&c, m)) != null);
    const mp = try mdatBytes(a, p);
    defer a.free(mp);
    const ml = try mdatBytes(a, l);
    defer a.free(ml);
    try testing.expectEqualSlices(u8, mp, ml); // the coded video, byte for byte

    try mark(io, a, plain, m); // a cmd tool's MP4, labelled afterwards
    try testing.expect(try hasMarker(io, a, plain));
    const r = try tmp.dir.readFileAlloc(io, "plain.mp4", a, .unlimited);
    defer a.free(r);
    const mr = try mdatBytes(a, r);
    defer a.free(mr);
    try testing.expectEqualSlices(u8, mp, mr);
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "not a container, only text" });
    const notes = try std.fs.path.join(a, &.{ dir, "notes.txt" });
    defer a.free(notes);
    try testing.expectError(error.NotMp4, hasMarker(io, a, notes));
}
