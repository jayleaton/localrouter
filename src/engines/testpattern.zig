//! The self-test engine: deterministic seeded images and videos (with a tone) made on the CPU, through the same
//! PNG and MP4 paths, progress reports and memory accounting as the GPU engines. `localrouter check selftest` runs it.
//! Input images (an edit's references, a video's first frame) are hashed into the pattern, so a test can see that
//! they reached the engine. It declares every capability. Options: `resident_mb` (held while loaded, really allocated), `load_ms`, `step_ms` (a sleep each step),
//! `max_side` (pixels: larger requests are refused before loading, as a GPU engine's are).

const std = @import("std");
const engine = @import("../engine/engine.zig");
const png = @import("../media/png.zig");
const mp4 = @import("../media/mp4.zig");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const entry: engine.Entry = .{ .name = "testpattern", .capabilities = &.{ .text_to_image, .image_edit, .text_to_video, .image_to_video }, .needs = needs, .create = create, .check = check };

fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    const max = engine.optionInt(cfg, "max_side", 0);
    const side: u64 = switch (req.*) {
        inline else => |v| @max(v.width, v.height),
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
            try writePng(t.env.io, path, px, v.width, v.height, &.{ .{ .key = "prompt", .value = v.prompt }, .{ .key = "seed", .value = seed } });
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
        var w = try mp4.Writer.open(t.env.io, arena, out, .{ .width = v.width, .height = v.height, .fps = v.fps, .wav = wav });
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

    fn unload(ptr: *anyopaque) void {
        const t: *TestPattern = @ptrCast(@alignCast(ptr));
        t.env.gpa.free(t.held);
        t.env.gpa.destroy(t);
    }
};

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

fn writePng(io: std.Io, path: []const u8, px: []const u8, w: u32, h: u32, text: []const png.Text) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = file.writer(io, &buf);
    try png.encode(&fw.interface, px, w, h, .rgb8, .{ .level = .fastest, .text = text });
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
