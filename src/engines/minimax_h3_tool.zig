//! The `minimax_h3` engine: MiniMax H3 text to video + audio on LocalRouter's Zig pipeline (`minimax_h3.pipeline`), one
//! MP4 a request (24 fps H.264, the 32 kHz stereo track as AAC unless `audio` is false). `weights` is a directory laid
//! out like ComfyUI's models folder plus the pack: `<pack>/` (`python -m stk_twin.h3.build`: the DiT with the Turbo
//! LoRA merged and `tokenizer.json`), `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`,
//! `vae/minimax_h3_audio_vae_fp32.safetensors` and `vae/minimax_h3_video_vae_fp16.safetensors`, all as published.
//! Options: `pack` (default h3-turbo-nvfp4), `max_width` / `max_height` (pixels, multiples of 32, default 768 x 448),
//! `max_seconds` (default 5), `steps` (default 8, the Turbo LoRA's), `unload_te` (1: free the 32B encoder after each
//! encode), `resident_mb` (the scheduler's estimate).

const std = @import("std");
const engine = @import("../engine/engine.zig");
const mp4 = @import("../media/mp4.zig");
const h3 = @import("minimax_h3");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const entry: engine.Entry = .{ .name = "minimax_h3", .capabilities = &.{.text_to_video}, .needs = needs, .create = create, .check = check, .defaults = defaults }; // image_to_video: the first-frame path is not ported yet

const P = h3.pipeline;
const fps = 24; // the model's frame rate: requests at another rate are refused
const default_steps = 8;
// Covers the measured engine peak (GB10, 2026-10-08): 36.7 GiB at 56 frames, 38.8 GiB for a 5 s clip, text encoder resident.
const default_resident_mb = 40_000;

/// The largest request the pipeline is built for (its buffers are allocated at load for exactly this), from the options.
fn limits(cfg: *const ToolConfig) P.Limits {
    return .{
        .width = @intCast(engine.optionInt(cfg, "max_width", 768) / 32 * 32),
        .height = @intCast(engine.optionInt(cfg, "max_height", 448) / 32 * 32),
        .frames = @intCast(engine.optionInt(cfg, "max_seconds", 5) * fps),
    };
}

/// A video without a size gets the largest this model takes.
fn defaults(cfg: *const ToolConfig, req: *Request) void {
    switch (req.*) {
        .video => |*v| if (v.width == 0 and v.height == 0) {
            const l = limits(cfg);
            v.width = l.width;
            v.height = l.height;
        },
        .image, .speech, .transcription, .chat => {},
    }
}

/// Refuses what `generate` would, before the 40 GB load evicts anything: a size past the pipeline's buffers or off its
/// 32-pixel grid, a clip longer than `max_seconds`, another frame rate, or a first frame.
fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    const v = switch (req.*) {
        .video => |x| x,
        .image, .speech, .transcription, .chat => return true, // the kind check refuses it
    };
    const l = limits(cfg);
    if (v.width > l.width or v.height > l.height or v.width % 32 != 0 or v.height % 32 != 0) {
        why.print("model '{s}' takes sizes up to {d}x{d}, multiples of 32 (omit size for {d}x{d})", .{ cfg.id, l.width, l.height, l.width, l.height }) catch {};
        return false;
    }
    if (v.seconds * fps > l.frames) {
        why.print("model '{s}' takes clips up to {d} seconds", .{ cfg.id, l.frames / fps }) catch {};
        return false;
    }
    if (v.fps != fps) {
        why.print("model '{s}' makes {d} fps video: omit fps or send {d}", .{ cfg.id, fps, fps }) catch {};
        return false;
    }
    if (v.first_frame != null) {
        why.print("model '{s}' does not take a first frame yet", .{cfg.id}) catch {};
        return false;
    }
    return true;
}

fn needs(cfg: *const ToolConfig, req: *const Request) engine.Needs {
    const working: u64 = switch (req.*) {
        .image, .speech, .transcription, .chat => 0,
        // the frames and the waveform on the host, plus ffmpeg's own buffers
        .video => |v| blk: {
            const sh = P.shapes(v.width, v.height, v.seconds * fps) catch break :blk 0;
            break :blk sh.frame_bytes + sh.audioFloats() * 4 * 2 + (64 << 20);
        },
    };
    return .{ .resident = engine.optionInt(cfg, "resident_mb", default_resident_mb) << 20, .working = working };
}

const MiniMaxH3 = struct {
    env: engine.Env,
    cfg: *const ToolConfig,
    p: ?*P.Pipeline = null,
    arena: std.heap.ArenaAllocator,

    fn load(ptr: *anyopaque) anyerror!u64 {
        const t: *MiniMaxH3 = @ptrCast(@alignCast(ptr));
        const a = t.arena.allocator();
        const root = t.cfg.weights;
        const paths: P.Paths = .{
            .pack = try std.fs.path.join(a, &.{ root, engine.optionString(t.cfg, "pack", "h3-turbo-nvfp4") }),
            .te = try std.fs.path.join(a, &.{ root, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" }),
            .vae_audio = try std.fs.path.join(a, &.{ root, "vae", "minimax_h3_audio_vae_fp32.safetensors" }),
            .vae_video = try std.fs.path.join(a, &.{ root, "vae", "minimax_h3_video_vae_fp16.safetensors" }),
        };
        const opts: P.Options = .{ .unload_te = engine.optionInt(t.cfg, "unload_te", 0) != 0 };
        t.p = try P.Pipeline.create(t.env.gpa, t.env.io, paths, limits(t.cfg), opts);
        // where the load went (the worker's log), as Qwen's: context, kernels, each component, read / wait / host time
        std.log.info("load ms: {s}", .{try std.json.Stringify.valueAlloc(a, t.p.?.load, .{})});
        return engine.optionInt(t.cfg, "resident_mb", default_resident_mb) << 20;
    }

    fn generate(ptr: *anyopaque, job: *const engine.Job, sink: engine.Sink, arena: std.mem.Allocator) anyerror!engine.Output {
        const t: *MiniMaxH3 = @ptrCast(@alignCast(ptr));
        const v = switch (job.request) {
            .video => |x| x,
            .image, .speech, .transcription, .chat => return engine.Refused.Refused,
        };
        if (v.first_frame != null or v.fps != fps) return engine.Refused.Refused; // image to video is not ported
        const p = t.p orelse return error.NotLoaded;
        const start = std.Io.Clock.awake.now(t.env.io);
        const steps = if (v.steps == 0) @as(u32, @intCast(engine.optionInt(t.cfg, "steps", default_steps))) else v.steps;
        const sh = P.shapes(v.width, v.height, v.seconds * fps) catch return engine.Refused.Refused;
        const gpa = t.env.gpa;
        const frames = try gpa.alloc(u8, sh.frame_bytes);
        defer gpa.free(frames);
        const audio = try gpa.alloc(f32, sh.audioFloats());
        defer gpa.free(audio);
        var bridge: Bridge = .{ .sink = sink };
        _ = p.generate(v.prompt, v.width, v.height, v.seconds * fps, steps, v.seed, frames, audio, bridge.progress()) catch |err| switch (err) {
            error.SizeNotMultipleOf32, error.RequestTooLarge, error.PromptTooLong => return engine.Refused.Refused,
            else => return err,
        };

        var wav: ?[]const u8 = null;
        if (v.audio) {
            const inter = try gpa.alloc(f32, audio.len); // [2, n] planar to interleaved
            defer gpa.free(inter);
            for (0..sh.samples) |i| for (0..P.audio_channels) |c| {
                inter[i * P.audio_channels + c] = audio[c * sh.samples + i];
            };
            wav = try std.fs.path.join(arena, &.{ job.dir, "audio.wav" });
            try mp4.writeWav(t.env.io, wav.?, inter, P.sample_rate, P.audio_channels);
        }
        const out = try std.fs.path.join(arena, &.{ job.dir, "video.mp4" });
        var w = try mp4.Writer.open(t.env.io, arena, out, .{ .width = v.width, .height = v.height, .fps = fps, .wav = wav });
        errdefer w.abort();
        const fb = @as(usize, v.width) * v.height * 3;
        for (0..sh.frames) |f| {
            try w.frame(frames[f * fb ..][0..fb]);
            if (f % fps == 0) sink.report(.{ .phase = "encode", .step = @intCast(f), .of = sh.frames });
        }
        try w.finish();
        if (wav) |path| std.Io.Dir.cwd().deleteFile(t.env.io, path) catch {};
        const names = try arena.alloc([]const u8, 1);
        names[0] = "video.mp4";
        const ms = start.durationTo(std.Io.Clock.awake.now(t.env.io)).toMilliseconds();
        return .{ .files = names, .seed = v.seed, .ms = @intCast(ms) };
    }

    fn unload(ptr: *anyopaque) void {
        const t: *MiniMaxH3 = @ptrCast(@alignCast(ptr));
        if (t.p) |p| p.destroy();
        t.arena.deinit();
        t.env.gpa.destroy(t);
    }
};

/// The pipeline's phase reports as the tool's progress.
const Bridge = struct {
    sink: engine.Sink,

    fn progress(b: *Bridge) P.Progress {
        return .{ .ctx = b, .report = report };
    }
    fn report(ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void {
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        b.sink.report(.{ .phase = phase, .step = step, .of = of });
    }
};

fn create(env: engine.Env, cfg: *const ToolConfig) anyerror!engine.Engine {
    const t = try env.gpa.create(MiniMaxH3);
    t.* = .{ .env = env, .cfg = cfg, .arena = .init(env.gpa) };
    return .{ .ptr = t, .vtable = &.{ .load = MiniMaxH3.load, .generate = MiniMaxH3.generate, .unload = MiniMaxH3.unload } };
}

test "a video without a size gets the model's own; one it cannot make is refused before loading" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "h3", "kind": "video", "engine": "minimax_h3", "options": {"max_width": 768, "max_height": 448, "max_seconds": 5}}]}
    );
    const cfg = c.tool("h3").?;
    var r: Request = .{ .video = .{ .prompt = "waves", .seconds = 2 } };
    engine.fill(cfg, &r);
    try std.testing.expectEqual(@as(u32, 768), r.video.width);
    try std.testing.expectEqual(@as(u32, 448), r.video.height);
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    try std.testing.expect(check(cfg, &r, &why));
    const big: Request = .{ .video = .{ .prompt = "waves", .width = 1344, .height = 768 } };
    try std.testing.expect(!check(cfg, &big, &why));
    try std.testing.expectEqualStrings("model 'h3' takes sizes up to 768x448, multiples of 32 (omit size for 768x448)", why.buffered());
    why = .fixed(&msg);
    const long: Request = .{ .video = .{ .prompt = "waves", .width = 768, .height = 448, .seconds = 6 } };
    try std.testing.expect(!check(cfg, &long, &why));
    try std.testing.expectEqualStrings("model 'h3' takes clips up to 5 seconds", why.buffered());
}
