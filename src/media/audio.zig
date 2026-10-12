//! Audio for the speech routes: 16-bit PCM WAV parsing and writing, the input every speech_to_text tool receives
//! (`normalize`: 16 kHz mono 16-bit PCM WAV, whatever was uploaded) and the response encodings of text to speech
//! (`encode`). A WAV already in the input format is copied; anything else goes through ffmpeg, bounded: demuxers from
//! a fixed list (no playlists or concat scripts that could open other files or URLs), local files only, a wall-clock
//! deadline that kills it, and a cap on the decoded length.

const std = @import("std");
const Io = std.Io;
const posix = std.posix;

/// The transcription input: 16 kHz, mono, 16-bit PCM (what Whisper-style models take).
pub const input_rate: u32 = 16_000;

/// The demuxers ffmpeg may pick for an upload: the formats OpenAI's transcription API takes (flac, mp3, mp4, mpeg,
/// mpga, m4a, ogg, wav, webm) and a few plain containers. Matched by ffmpeg against demuxer names.
const demuxers = "wav,w64,mp3,flac,ogg,mov,matroska,aac,caf,aiff";

pub const Invalid = error{Invalid};

fn fail(why: *[]const u8, msg: []const u8) Invalid {
    why.* = msg;
    return error.Invalid;
}

/// A parsed WAV: its format and the bytes of its data chunk.
pub const Wav = struct {
    rate: u32,
    channels: u16,
    bits: u16,
    pcm: bool, // integer PCM (format 1, or extensible with the PCM subformat)
    data: []const u8,

    /// Milliseconds of audio in `data`.
    pub fn ms(w: Wav) u64 {
        const frame = @as(u64, w.channels) * (w.bits / 8);
        if (frame == 0 or w.rate == 0) return 0;
        return @as(u64, w.data.len / frame) * 1000 / w.rate;
    }
};

/// The RIFF/WAVE file in `bytes`, or null when it is not one or is cut short (its data chunk claims more than is there).
pub fn parseWav(bytes: []const u8) ?Wav {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return null;
    var fmt: ?Wav = null;
    var at: usize = 12;
    while (at + 8 <= bytes.len) {
        const id = bytes[at..][0..4];
        const size: usize = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        const body = at + 8;
        if (size > bytes.len - body) return null;
        if (std.mem.eql(u8, id, "fmt ")) {
            if (size < 16) return null;
            const f = bytes[body..][0..16];
            const tag = std.mem.readInt(u16, f[0..2], .little);
            var pcm = tag == 1;
            if (tag == 0xFFFE and size >= 40) pcm = std.mem.readInt(u16, bytes[body + 24 ..][0..2], .little) == 1; // subformat GUID's first field
            fmt = .{ .rate = std.mem.readInt(u32, f[4..8], .little), .channels = std.mem.readInt(u16, f[2..4], .little), .bits = std.mem.readInt(u16, f[14..16], .little), .pcm = pcm, .data = &.{} };
        } else if (std.mem.eql(u8, id, "data")) {
            var w = fmt orelse return null;
            w.data = bytes[body .. body + size];
            return w;
        }
        at = body + size + (size & 1); // chunks are padded to even sizes
    }
    return null;
}

/// A 44-byte header for `data_len` bytes of 16-bit PCM.
pub fn header(rate: u32, channels: u16, data_len: u32) [44]u8 {
    var h: [44]u8 = undefined;
    @memcpy(h[0..4], "RIFF");
    std.mem.writeInt(u32, h[4..8], 36 +| data_len, .little);
    @memcpy(h[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, h[16..20], 16, .little);
    std.mem.writeInt(u16, h[20..22], 1, .little);
    std.mem.writeInt(u16, h[22..24], channels, .little);
    std.mem.writeInt(u32, h[24..28], rate, .little);
    std.mem.writeInt(u32, h[28..32], rate * channels * 2, .little);
    std.mem.writeInt(u16, h[32..34], channels * 2, .little);
    std.mem.writeInt(u16, h[34..36], 16, .little);
    @memcpy(h[36..40], "data");
    std.mem.writeInt(u32, h[40..44], data_len, .little);
    return h;
}

/// `pcm` (16-bit little-endian samples) as a WAV file in `a`.
pub fn wav(a: std.mem.Allocator, rate: u32, channels: u16, pcm: []const u8) ![]u8 {
    const out = try a.alloc(u8, 44 + pcm.len);
    out[0..44].* = header(rate, channels, @intCast(pcm.len));
    @memcpy(out[44..], pcm);
    return out;
}

pub const Limits = struct {
    max_ms: u64 = 30 * 60 * 1000,
    timeout_ms: u32 = 120_000, // ffmpeg's wall clock, decode or encode
};

/// A transcription input: `audio.wav` (16 kHz mono 16-bit PCM) and its length.
pub const Normalized = struct { wav: []const u8, ms: u64 };

/// Turns an upload into the transcription input. `tmp_dir` holds the upload while ffmpeg reads it (it is removed).
/// Refusals (empty, undecodable, too long, no audio) set `why` and return `error.Invalid`.
pub fn normalize(io: Io, gpa: std.mem.Allocator, a: std.mem.Allocator, tmp_dir: []const u8, upload: []const u8, l: Limits, why: *[]const u8) (Invalid || error{ OutOfMemory, DecoderFailed })!Normalized {
    if (upload.len == 0) return fail(why, "the audio file is empty");
    const max_bytes = l.max_ms * input_rate * 2 / 1000;
    if (parseWav(upload)) |w| if (w.pcm and w.rate == input_rate and w.channels == 1 and w.bits == 16) {
        const pcm = w.data[0 .. w.data.len & ~@as(usize, 1)];
        if (pcm.len > max_bytes) return fail(why, tooLong(l));
        if (pcm.len == 0) return fail(why, "the audio has no samples");
        return .{ .wav = try wav(a, input_rate, 1, pcm), .ms = w.ms() };
    };
    var limit_buf: [24]u8 = undefined;
    const limit = std.fmt.bufPrint(&limit_buf, "{d}.{d:0>3}", .{ (l.max_ms + 1000) / 1000, (l.max_ms + 1000) % 1000 }) catch unreachable;
    const pcm = ffmpeg(io, gpa, a, tmp_dir, upload, &.{ "-protocol_whitelist", "file", "-format_whitelist", demuxers }, &.{ "-vn", "-sn", "-dn", "-t", limit, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le" }, max_bytes + input_rate * 2 * 2, l.timeout_ms) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Failed => return fail(why, "the audio could not be decoded (send WAV, MP3, FLAC, OGG/Opus, M4A/MP4, WebM or AAC)"),
        error.TooLarge => return fail(why, tooLong(l)),
        error.Timeout => return fail(why, "decoding the audio took too long"),
        error.Spawn => return error.DecoderFailed,
    };
    if (pcm.len > max_bytes) return fail(why, tooLong(l));
    if (pcm.len < 2) return fail(why, "the audio has no samples");
    const even = pcm[0 .. pcm.len & ~@as(usize, 1)];
    return .{ .wav = try wav(a, input_rate, 1, even), .ms = @as(u64, even.len / 2) * 1000 / input_rate };
}

fn tooLong(l: Limits) []const u8 {
    return if (l.max_ms == 30 * 60 * 1000) "the audio is longer than 30 minutes" else "the audio is longer than this model takes";
}

/// Text to speech response formats. `wav` and `pcm` need no encoder.
pub const Format = enum {
    wav,
    pcm,
    flac,
    mp3,
    opus,
    aac,

    pub fn contentType(f: Format) []const u8 {
        return switch (f) {
            .wav => "audio/wav",
            .pcm => "audio/pcm",
            .flac => "audio/flac",
            .mp3 => "audio/mpeg",
            .opus => "audio/ogg",
            .aac => "audio/aac",
        };
    }
};

/// A tool's `speech.wav` in `format`: `pcm` is its raw samples (16-bit little-endian at the WAV's rate), the
/// compressed formats come from ffmpeg.
pub fn encode(io: Io, gpa: std.mem.Allocator, a: std.mem.Allocator, tmp_dir: []const u8, speech: []const u8, format: Format, l: Limits) error{ OutOfMemory, BadWav, EncoderFailed }![]const u8 {
    const w = parseWav(speech) orelse return error.BadWav;
    if (!w.pcm or w.bits != 16 or w.channels == 0) return error.BadWav;
    const codec: []const []const u8 = switch (format) {
        .wav => return speech,
        .pcm => return w.data,
        .flac => &.{ "-c:a", "flac", "-f", "flac" },
        .mp3 => &.{ "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3" },
        .opus => &.{ "-c:a", "libopus", "-b:a", "64k", "-ar", "48000", "-f", "ogg" },
        .aac => &.{ "-c:a", "aac", "-b:a", "128k", "-f", "adts" },
    };
    return ffmpeg(io, gpa, a, tmp_dir, speech, &.{ "-protocol_whitelist", "file", "-f", "wav" }, codec, speech.len + (1 << 20), l.timeout_ms) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.EncoderFailed,
    };
}

const RunError = error{ OutOfMemory, Spawn, Failed, TooLarge, Timeout };

/// Runs ffmpeg on `input` (written to a file in `tmp_dir` first, so no pipe can deadlock) and returns what it writes
/// to stdout, at most `max_out` bytes, within `timeout_ms`. Its process group is killed on any failure.
fn ffmpeg(io: Io, gpa: std.mem.Allocator, a: std.mem.Allocator, tmp_dir: []const u8, input: []const u8, in_opts: []const []const u8, out_opts: []const []const u8, max_out: u64, timeout_ms: u32) RunError![]u8 {
    var rnd: [8]u8 = undefined;
    io.random(&rnd);
    const path = std.fmt.allocPrint(gpa, "{s}/.audio-{x}", .{ tmp_dir, std.mem.readInt(u64, &rnd, .little) }) catch return error.OutOfMemory;
    defer gpa.free(path);
    Io.Dir.cwd().createDirPath(io, tmp_dir) catch return error.Spawn;
    Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = input }) catch return error.Spawn;
    defer Io.Dir.cwd().deleteFile(io, path) catch {};

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    argv.appendSlice(gpa, &.{ "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "2" }) catch return error.OutOfMemory;
    argv.appendSlice(gpa, in_opts) catch return error.OutOfMemory;
    argv.appendSlice(gpa, &.{ "-i", path }) catch return error.OutOfMemory;
    argv.appendSlice(gpa, out_opts) catch return error.OutOfMemory;
    argv.append(gpa, "pipe:1") catch return error.OutOfMemory;
    var child = std.process.spawn(io, .{ .argv = argv.items, .stdin = .ignore, .stdout = .pipe, .stderr = .ignore, .pgid = 0 }) catch return error.Spawn;
    var done = false;
    defer if (!done) {
        _ = posix.system.kill(-child.id.?, posix.SIG.KILL);
        _ = child.wait(io) catch {};
    };

    var out: std.ArrayList(u8) = .empty;
    const fd = child.stdout.?.handle;
    const start = Io.Clock.awake.now(io);
    var chunk: [64 * 1024]u8 = undefined;
    while (true) {
        const elapsed: u64 = @intCast(@max(0, start.durationTo(Io.Clock.awake.now(io)).toMilliseconds()));
        if (elapsed >= timeout_ms) return error.Timeout;
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, @intCast(timeout_ms - elapsed)) catch return error.Failed;
        if (ready == 0) return error.Timeout;
        const rc = posix.system.read(fd, &chunk, chunk.len);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) break;
                const n: usize = @intCast(rc);
                if (out.items.len + n > max_out) return error.TooLarge;
                out.appendSlice(a, chunk[0..n]) catch return error.OutOfMemory;
            },
            .INTR, .AGAIN => {},
            else => return error.Failed,
        }
    }
    const term = child.wait(io) catch return error.Failed;
    done = true;
    if (term != .exited or term.exited != 0) return error.Failed;
    return out.items;
}

test "WAV round trip, chunk walking and truncation" {
    const a = std.testing.allocator;
    const pcm = [_]u8{ 1, 0, 2, 0, 0xff, 0x7f, 0, 0x80 };
    const w = try wav(a, 16_000, 1, &pcm);
    defer a.free(w);
    const p = parseWav(w).?;
    try std.testing.expectEqual(@as(u32, 16_000), p.rate);
    try std.testing.expect(p.pcm and p.channels == 1 and p.bits == 16);
    try std.testing.expectEqualSlices(u8, &pcm, p.data);
    // A LIST chunk (odd size, padded) before the data chunk is skipped.
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    try list.appendSlice(a, w[0..36]);
    try list.appendSlice(a, "LIST\x03\x00\x00\x00abc\x00");
    try list.appendSlice(a, w[36..]);
    try std.testing.expectEqualSlices(u8, &pcm, parseWav(list.items).?.data);
    try std.testing.expect(parseWav(w[0 .. w.len - 1]) == null); // the data chunk claims more than is there
    try std.testing.expect(parseWav("RIFF\x00\x00\x00\x00WAVE") == null);
    try std.testing.expect(parseWav("\x89PNG\r\n\x1a\n....") == null);
    try std.testing.expectEqual(@as(u64, 0), p.ms());
}

test "normalize: an exact WAV is copied, others decode through ffmpeg, refusals are clear" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const tmp = "/tmp/localrouter-audio-test";
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};
    var why: []const u8 = "";

    // One second of a 16 kHz ramp: copied sample for sample.
    const samples = try a.alloc(u8, 32_000);
    for (0..16_000) |i| std.mem.writeInt(i16, samples[i * 2 ..][0..2], @intCast(@as(i32, @intCast(i % 2000)) - 1000), .little);
    const exact = try wav(a, 16_000, 1, samples);
    const n = try normalize(io, gpa, a, tmp, exact, .{}, &why);
    try std.testing.expectEqual(@as(u64, 1000), n.ms);
    try std.testing.expectEqualSlices(u8, samples, parseWav(n.wav).?.data);

    // 48 kHz stereo: resampled and mixed down by ffmpeg to 16 kHz mono, same length.
    const stereo = try a.alloc(u8, 48_000 * 4);
    @memset(stereo, 0);
    const n2 = try normalize(io, gpa, a, tmp, try wav(a, 48_000, 2, stereo), .{}, &why);
    try std.testing.expectEqual(@as(u64, 1000), n2.ms);
    try std.testing.expectEqual(@as(u32, 16_000), parseWav(n2.wav).?.rate);

    const refusals = [_]struct { data: []const u8, l: Limits = .{}, msg: []const u8 }{
        .{ .data = "", .msg = "the audio file is empty" },
        .{ .data = "\x89PNG\r\n\x1a\nnot audio at all, just bytes", .msg = "the audio could not be decoded (send WAV, MP3, FLAC, OGG/Opus, M4A/MP4, WebM or AAC)" },
        .{ .data = "#EXTM3U\n#EXTINF:10,\nhttp://127.0.0.1:9/x.ts\n#EXT-X-ENDLIST\n", .msg = "the audio could not be decoded (send WAV, MP3, FLAC, OGG/Opus, M4A/MP4, WebM or AAC)" },
        .{ .data = exact, .l = .{ .max_ms = 500 }, .msg = "the audio is longer than this model takes" },
        .{ .data = try wav(a, 48_000, 2, stereo), .l = .{ .max_ms = 500 }, .msg = "the audio is longer than this model takes" },
        .{ .data = try wav(a, 16_000, 1, ""), .msg = "the audio has no samples" },
    };
    for (refusals) |r| {
        try std.testing.expectError(error.Invalid, normalize(io, gpa, a, tmp, r.data, r.l, &why));
        try std.testing.expectEqualStrings(r.msg, why);
    }
}

test "encode: wav as is, pcm raw, flac through ffmpeg" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pcm = try a.alloc(u8, 24_000 * 2);
    for (0..24_000) |i| std.mem.writeInt(i16, pcm[i * 2 ..][0..2], @intFromFloat(8000 * @sin(@as(f32, @floatFromInt(i)) * 0.1)), .little);
    const w = try wav(a, 24_000, 1, pcm);
    try std.testing.expectEqualSlices(u8, w, try encode(io, gpa, a, "/tmp/localrouter-audio-test", w, .wav, .{}));
    try std.testing.expectEqualSlices(u8, pcm, try encode(io, gpa, a, "/tmp/localrouter-audio-test", w, .pcm, .{}));
    const flac = try encode(io, gpa, a, "/tmp/localrouter-audio-test", w, .flac, .{});
    try std.testing.expect(std.mem.startsWith(u8, flac, "fLaC"));
    try std.testing.expectError(error.BadWav, encode(io, gpa, a, "/tmp/localrouter-audio-test", "nope", .mp3, .{}));
    Io.Dir.cwd().deleteTree(io, "/tmp/localrouter-audio-test") catch {};
}
