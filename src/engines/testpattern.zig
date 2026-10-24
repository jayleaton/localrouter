//! The self-test engine: deterministic seeded images and videos (with a tone) made on the CPU, through the same
//! PNG and MP4 paths, progress reports and memory accounting as the GPU engines. `localrouter check selftest` runs it.
//! Input images (an edit's references, a video's first frame) are hashed into the pattern, so a test can see that
//! they reached the engine. It declares every capability. Options: `resident_mb` (held while loaded, really allocated), `load_ms`, `step_ms` (a sleep each step),
//! `max_side` (pixels: larger requests are refused before loading, as a GPU engine's are).
//! Speech is a 24 kHz tone, 60 ms a character divided by `speed`, pitched by the seed and voice: the fixture other
//! text_to_speech adapters are routed and tested against. A transcription reads `audio.wav` and writes a transcript
//! naming its length and a hash of its samples (so a test sees what reached the engine) with one segment a second.

const std = @import("std");
const engine = @import("../engine/engine.zig");
const png = @import("../media/png.zig");
const mp4 = @import("../media/mp4.zig");
const audio = @import("../media/audio.zig");
const marker = @import("../media/marker.zig");
const request = @import("../tool/request.zig");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const entry: engine.Entry = .{ .name = "testpattern", .capabilities = &.{ .text_to_image, .image_edit, .text_to_video, .image_to_video, .text_to_speech, .speech_to_text }, .needs = needs, .create = create, .check = check };

fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    const max = engine.optionInt(cfg, "max_side", 0);
    const side: u64 = switch (req.*) {
        .image => |v| @max(v.width, v.height),
        .video => |v| @max(v.width, v.height),
        .speech, .transcription, .chat, .decision => return true,
    };
    if (max == 0 or side <= max) return true;
    why.print("model '{s}' takes sizes up to {d} pixels a side", .{ cfg.id, max }) catch {};
    return false;
}

const default_steps = 4;

fn residentBytes(cfg: *const ToolConfig) u64 {
    return engine.optionInt(cfg, "resident_mb", 32) << 20;
}

fn needs(cfg: *const ToolConfig, req: *const Request) engine.Needs {
    const working: u64 = switch (req.*) {
        .image => |v| @as(u64, v.width) * v.height * 3 * 2, // one frame + the encoder's copy
        .video => |v| @as(u64, v.width) * v.height * 3 * 2 + (64 << 20), // a frame, plus ffmpeg's own buffers
        .speech => |v| speechSamples(v) * 2 * 2, // the samples, and the file's copy
        .transcription => |v| v.duration_ms * audio.input_rate / 1000 * 2 + (1 << 20), // the samples read back, and the transcript
        .chat, .decision => 0, // never served: chat and decision tools run a model server (`openai_server`)
    };
    return .{ .resident = residentBytes(cfg), .working = working };
}

const TestPattern = struct {
    env: engine.Env,
    cfg: *const ToolConfig,
    held: []u8 = &.{},

    fn load(ptr: *anyopaque) anyerror!u64 {
        const t: *TestPattern = @ptrCast(@alignCast(ptr));
        t.held = try t.env.gpa.alloc(u8, residentBytes(t.cfg));
        @memset(t.held, 0xA5); // touch every page, so the resident bytes are real
        try sleepMs(t.env.io, engine.optionInt(t.cfg, "load_ms", 0));
        return t.held.len;
    }

    fn generate(ptr: *anyopaque, job: *const engine.Job, sink: engine.Sink, arena: std.mem.Allocator) anyerror!engine.Output {
        const t: *TestPattern = @ptrCast(@alignCast(ptr));
        const start = std.Io.Clock.awake.now(t.env.io);
        const files = switch (job.request) {
            .image => |v| try t.image(job.dir, v, try t.inputHash(job.dir, v.references), sink, arena),
            .video => |v| try t.video(job.dir, v, try t.inputHash(job.dir, if (v.first_frame) |f| &.{f} else &.{}), sink, arena),
            .speech => |v| try t.speech(job.dir, v, sink, arena),
            .transcription => |v| try t.transcribe(job.dir, v, sink, arena),
            .chat, .decision => return engine.Refused.Refused,
        };
        const ms = start.durationTo(std.Io.Clock.awake.now(t.env.io)).toMilliseconds();
        return .{ .files = files, .seed = job.request.seed(), .ms = @intCast(ms) };
    }

    /// A hash of the input files' contents (0 for none); a missing file is the caller's fault.
    fn inputHash(t: *TestPattern, dir: []const u8, names: []const []const u8) !u64 {
        var h: std.hash.Wyhash = .init(0);
        for (names) |name| {
            const path = try std.fs.path.join(t.env.gpa, &.{ dir, name });
            defer t.env.gpa.free(path);
            const data = std.Io.Dir.cwd().readFileAlloc(t.env.io, path, t.env.gpa, .limited(64 << 20)) catch return error.Refused;
            defer t.env.gpa.free(data);
            h.update(data);
        }
        return if (names.len == 0) 0 else h.final();
    }

    fn steps(t: *TestPattern, n: u32, sink: engine.Sink, phase: []const u8) !void {
        const of = if (n == 0) default_steps else n;
        for (0..of) |i| {
            try sleepMs(t.env.io, engine.optionInt(t.cfg, "step_ms", 0));
            sink.report(.{ .phase = phase, .step = @intCast(i + 1), .of = of });
        }
    }

    fn image(t: *TestPattern, dir: []const u8, v: anytype, mix: u64, sink: engine.Sink, arena: std.mem.Allocator) ![]const []const u8 {
        try t.steps(v.steps, sink, "denoise");
        const px = try t.env.gpa.alloc(u8, @as(usize, v.width) * v.height * 3);
        defer t.env.gpa.free(px);
        const names = try arena.alloc([]const u8, v.n);
        for (names, 0..) |*name, i| {
            paint(px, v.width, v.height, (v.seed +% i) ^ mix, 0);
            name.* = try std.fmt.allocPrint(arena, "{d}.png", .{i});
            const path = try std.fs.path.join(arena, &.{ dir, name.* });
            var seed_buf: [24]u8 = undefined;
            const seed = try std.fmt.bufPrint(&seed_buf, "{d}", .{v.seed +% i});
            const label: marker.Marker = .{ .model = t.cfg.id, .edit = v.references.len > 0 };
            try writePng(t.env.io, path, px, v.width, v.height, &.{ .{ .key = "prompt", .value = v.prompt }, .{ .key = "seed", .value = seed } }, label);
        }
        return names;
    }

    fn video(t: *TestPattern, dir: []const u8, v: anytype, mix: u64, sink: engine.Sink, arena: std.mem.Allocator) ![]const []const u8 {
        try t.steps(v.steps, sink, "denoise");
        const frames = v.seconds * v.fps;
        const out = try std.fs.path.join(arena, &.{ dir, "video.mp4" });
        var wav: ?[]const u8 = null;
        if (v.audio) {
            const rate = 48_000;
            const samples = try t.env.gpa.alloc(f32, rate * v.seconds);
            defer t.env.gpa.free(samples);
            const hz: f32 = 220 + @as(f32, @floatFromInt(v.seed % 440));
            for (samples, 0..) |*s, i| s.* = 0.2 * @sin(2 * std.math.pi * hz * @as(f32, @floatFromInt(i)) / rate);
            wav = try std.fs.path.join(arena, &.{ dir, "audio.wav" });
            try mp4.writeWav(t.env.io, wav.?, samples, rate, 1);
        }
        const label: marker.Marker = .{ .model = t.cfg.id, .edit = v.first_frame != null };
        var w = try mp4.Writer.open(t.env.io, arena, out, .{ .width = v.width, .height = v.height, .fps = v.fps, .wav = wav, .marker = label });
        errdefer w.abort();
        const px = try t.env.gpa.alloc(u8, @as(usize, v.width) * v.height * 3);
        defer t.env.gpa.free(px);
        for (0..frames) |f| {
            paint(px, v.width, v.height, v.seed ^ mix, @intCast(f));
            try w.frame(px);
            if (f % v.fps == 0) sink.report(.{ .phase = "encode", .step = @intCast(f), .of = frames });
        }
        try w.finish();
        if (wav) |p| std.Io.Dir.cwd().deleteFile(t.env.io, p) catch {};
        const names = try arena.alloc([]const u8, 1);
        names[0] = "video.mp4";
        return names;
    }

    fn speech(t: *TestPattern, dir: []const u8, v: request.Speech, sink: engine.Sink, arena: std.mem.Allocator) ![]const []const u8 {
        try t.steps(0, sink, "synthesize");
        const n = speechSamples(v);
        const pcm = try t.env.gpa.alloc(u8, n * 2);
        defer t.env.gpa.free(pcm);
        const hz: f32 = 180 + @as(f32, @floatFromInt((v.seed ^ std.hash.Wyhash.hash(0, v.voice)) % 300));
        for (0..n) |i| {
            const x = 0.25 * @sin(2 * std.math.pi * hz * @as(f32, @floatFromInt(i)) / speech_rate);
            std.mem.writeInt(i16, pcm[i * 2 ..][0..2], @intFromFloat(x * 32767), .little);
        }
        try t.writeFile(dir, "speech.wav", &.{ &audio.header(speech_rate, 1, @intCast(pcm.len)), pcm });
        const names = try arena.alloc([]const u8, 1);
        names[0] = "speech.wav";
        return names;
    }

    fn transcribe(t: *TestPattern, dir: []const u8, v: request.Transcription, sink: engine.Sink, arena: std.mem.Allocator) ![]const []const u8 {
        const path = try std.fs.path.join(arena, &.{ dir, v.audio });
        const data = std.Io.Dir.cwd().readFileAlloc(t.env.io, path, arena, .limited(256 << 20)) catch return error.Refused;
        const w = audio.parseWav(data) orelse return error.Refused;
        if (!w.pcm or w.rate != audio.input_rate or w.channels != 1 or w.bits != 16) return error.Refused;
        const ms = w.ms();
        const seconds: u32 = @intCast(@max(1, (ms + 999) / 1000));
        try t.steps(@min(seconds, 1000), sink, "transcribe");
        const text = try std.fmt.allocPrint(arena, "testpattern heard {d} ms of audio, hash {x}{s}{s}", .{ ms, std.hash.Wyhash.hash(0, w.data), if (v.language.len > 0) ", language " else "", v.language });
        const segments = try arena.alloc(request.Transcript.Segment, seconds);
        for (segments, 0..) |*seg, i| seg.* = .{ .id = @intCast(i), .start = @floatFromInt(i), .end = @min(@as(f64, @floatFromInt(i + 1)), @as(f64, @floatFromInt(ms)) / 1000), .text = if (i == 0) text else "" };
        var words: ?[]const request.Transcript.Word = null;
        if (v.timestamps == .word) {
            const list = try arena.alloc(request.Transcript.Word, 2);
            list[0] = .{ .word = "testpattern", .start = 0, .end = 0.5 };
            list[1] = .{ .word = "heard", .start = 0.5, .end = 1 };
            words = list;
        }
        const out: request.Transcript = .{ .text = text, .language = if (v.language.len > 0) v.language else "en", .duration = @as(f64, @floatFromInt(ms)) / 1000, .segments = segments, .words = words };
        const json = try std.json.Stringify.valueAlloc(arena, out, .{ .emit_null_optional_fields = false });
        try t.writeFile(dir, "transcript.json", &.{json});
        const names = try arena.alloc([]const u8, 1);
        names[0] = "transcript.json";
        return names;
    }

    fn writeFile(t: *TestPattern, dir: []const u8, name: []const u8, parts: []const []const u8) !void {
        const path = try std.fs.path.join(t.env.gpa, &.{ dir, name });
        defer t.env.gpa.free(path);
        var file = try std.Io.Dir.cwd().createFile(t.env.io, path, .{});
        defer file.close(t.env.io);
        var buf: [64 * 1024]u8 = undefined;
        var fw = file.writer(t.env.io, &buf);
        for (parts) |p| try fw.interface.writeAll(p);
        try fw.interface.flush();
    }

    fn unload(ptr: *anyopaque) void {
        const t: *TestPattern = @ptrCast(@alignCast(ptr));
        t.env.gpa.free(t.held);
        t.env.gpa.destroy(t);
    }
};

const speech_rate: u32 = 24_000;

/// 60 ms of tone a character, divided by the speed.
fn speechSamples(v: request.Speech) u64 {
    const ms = @as(f64, @floatFromInt(@max(1, v.input.len) * 60)) / @max(v.speed, 0.25);
    return @intFromFloat(ms * speech_rate / 1000);
}

fn create(env: engine.Env, cfg: *const ToolConfig) anyerror!engine.Engine {
    const t = try env.gpa.create(TestPattern);
    t.* = .{ .env = env, .cfg = cfg };
    return .{ .ptr = t, .vtable = &.{ .load = TestPattern.load, .generate = TestPattern.generate, .unload = TestPattern.unload } };
}

/// Colour bars shifted by the seed and frame, plus a hashed grain: one pass over the pixels.
fn paint(px: []u8, w: u32, h: u32, seed: u64, frame: u32) void {
    const shift: u32 = @truncate(seed *% 0x9E3779B97F4A7C15 >> 40);
    var i: usize = 0;
    for (0..h) |y| {
        const row_tint: u8 = @truncate(y * 255 / h);
        for (0..w) |x| {
            const bar: u32 = (@as(u32, @intCast(x)) + shift + frame * 8) * 8 / w % 8;
            var g: u32 = @truncate((@as(u64, @intCast(i)) ^ seed) *% 0x2545F4914F6CDD1D >> 56);
            g &= 0x1F;
            px[i] = @truncate(((bar & 1) * 200 + g) ^ row_tint);
            px[i + 1] = @truncate(((bar >> 1 & 1) * 200 + g));
            px[i + 2] = @truncate(((bar >> 2 & 1) * 200 + g) ^ (255 - row_tint));
            i += 3;
        }
    }
}

fn writePng(io: std.Io, path: []const u8, px: []const u8, w: u32, h: u32, text: []const png.Text, label: marker.Marker) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = file.writer(io, &buf);
    try png.encode(&fw.interface, px, w, h, .rgb8, .{ .level = .fastest, .text = text, .marker = label });
    try fw.interface.flush();
}

fn sleepMs(io: std.Io, ms: u64) !void {
    if (ms > 0) try std.Io.sleep(io, .fromMilliseconds(@intCast(ms)), .awake);
}

test "paint is deterministic in the seed" {
    var a: [16 * 16 * 3]u8 = undefined;
    var b: [16 * 16 * 3]u8 = undefined;
    paint(&a, 16, 16, 42, 0);
    paint(&b, 16, 16, 42, 0);
    try std.testing.expectEqualSlices(u8, &a, &b);
    paint(&b, 16, 16, 43, 0);
    try std.testing.expect(!std.mem.eql(u8, &a, &b));
}
