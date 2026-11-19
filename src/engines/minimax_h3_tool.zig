//! The `minimax_h3` engine: MiniMax H3 text or image to video + audio on LocalRouter's Zig pipeline (`minimax_h3.pipeline`), one
//! MP4 a request (24 fps H.264, the 32 kHz stereo track as AAC unless `audio` is false). `weights` is a directory laid
//! out like ComfyUI's models folder plus the pack: `<pack>/` (`python -m stk_twin.h3.build`: the DiT with the Turbo
//! LoRA merged and `tokenizer.json`), `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`,
//! `vae/minimax_h3_audio_vae_fp32.safetensors` and `vae/minimax_h3_video_vae_fp16.safetensors`, all as published.
//! Options: `pack` (default h3-turbo-nvfp4), `max_pixels` (a frame's area at `max_seconds`, default 768 x 448: a frame
//! of any shape within it, landscape or portrait; a larger one, up to the canvas's area, for a shorter clip), `max_side` (the long side, default
//! the model's 1344), `max_seconds` (default 5), `default_size` (`WIDTHxHEIGHT`, what a request without a size gets;
//! default the largest landscape), `steps` (default 8, the Turbo LoRA's), `unload_te` (1: free the 32B encoder after
//! each encode), `resident_mb` (a floor for the booking, which is otherwise the weights plus the buffers for the limits;
//! a request adds its own scratch). A config of `max_width` / `max_height` gets their product as
//! `max_pixels` and either of them as `max_side` where above 1344. The pipeline's buffers are sized at load for the
//! largest clip, so the memory a job takes follows `max_pixels` and `max_seconds` (plus the video VAE's per-decode scratch,
//! which follows the frame's area and width), while its time follows the clip asked for. Limits outside the API's, or
//! whose largest clip does not fit the daemon's memory budget, stop the daemon at start (`validate`).

const std = @import("std");
const engine = @import("../engine/engine.zig");
const mp4 = @import("../media/mp4.zig");
const image_in = @import("../media/image_in.zig");
const h3 = @import("minimax_h3");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const entry: engine.Entry = .{ .name = "minimax_h3", .capabilities = &.{ .text_to_video, .image_to_video }, .needs = needs, .create = create, .check = check, .defaults = defaults, .validate = validate, .license = license };

/// The weights' licence, unless a tool's config names its own.
pub const license: @import("../config.zig").License = .{
    .name = "MiniMax H3 Community License Agreement",
    .url = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
    .use = "only outside the excluded territories, under its Acceptable Use Policy; over USD 20 million yearly revenue needs MiniMax's written authorization",
    .commercial = true,
    .excluded_regions = &.{ "United States", "European Union", "United Kingdom", "South Korea" },
    .notice = "content published to the public must be disclosed as machine-generated; a commercial product shows \"MiniMax H3\" in its user interface",
};

const P = h3.pipeline;
const fps = 24; // the model's frame rate: requests at another rate are refused
const default_steps = 8;
// Covers the measured engine peak (GB10, 2026-10-08): 36.7 GiB at 56 frames, 38.8 GiB for a 5 s clip, text encoder resident.
const default_resident_mb = 40_000;

const api: @import("../tool/request.zig").Limits = .{};

/// An integer option as a u32, `ceiling` where it is larger (`validate` refuses that at load).
fn option32(cfg: *const ToolConfig, key: []const u8, default: u32, ceiling: u32) u32 {
    return @intCast(@min(engine.optionInt(cfg, key, default), ceiling));
}

/// A config's `max_width` x `max_height` (each rounded down to 32), 768 x 448 where one is not set.
fn legacySize(cfg: *const ToolConfig) [2]u32 {
    return .{ option32(cfg, "max_width", 768, api.max_side) / 32 * 32, option32(cfg, "max_height", 448, api.max_side) / 32 * 32 };
}

/// The largest request the pipeline is built for (its buffers are allocated at load for exactly this), from the options.
fn limits(cfg: *const ToolConfig) P.Limits {
    const legacy = legacySize(cfg);
    return .{
        .pixels = option32(cfg, "max_pixels", legacy[0] * legacy[1], max_pixels) / 1024 * 1024,
        .frames = option32(cfg, "max_seconds", 5, api.max_seconds) * fps,
    };
}

/// The long side a request may have: the option within the API's.
fn maxSide(cfg: *const ToolConfig) u32 {
    const legacy = legacySize(cfg);
    return option32(cfg, "max_side", @max(P.canvas_side, legacy[0], legacy[1]), api.max_side) / 32 * 32;
}

/// The largest `max_pixels`: a square frame at the API's longest side.
const max_pixels = api.max_side * api.max_side;

/// The API's shortest side, rounded up to the 32-pixel grid.
const min_side = (api.min_side + 31) / 32 * 32;

/// Whether `width` x `height` is on the 32-pixel grid within the API's sides and the long side, and fits the buffers for `frames`.
fn takes(cfg: *const ToolConfig, width: u32, height: u32, frames: u32) bool {
    const side = maxSide(cfg);
    return @min(width, height) >= min_side and @max(width, height) <= side and P.fits(limits(cfg), width, height, frames);
}

/// The largest landscape frame about 16:9 that a full-length clip takes, else the largest with its width at the long side.
fn widest(cfg: *const ToolConfig) [2]u32 {
    const l = limits(cfg);
    const side = maxSide(cfg);
    for ([_]bool{ false, true }) |capped| {
        var h: u32 = side;
        while (h >= min_side) : (h -= 32) {
            const w = if (capped) @min(h * 16 / 9 / 32 * 32, side) else h * 16 / 9 / 32 * 32;
            if (takes(cfg, w, h, l.frames)) return .{ w, h };
        }
    }
    return .{ min_side, min_side };
}

/// The size a video without one gets, the first a full-length clip takes: `default_size`, `max_width` x `max_height`
/// where the config sets either, the widest.
fn defaultSize(cfg: *const ToolConfig) [2]u32 {
    const l = limits(cfg);
    if (@import("../tool/request.zig").parseSize(engine.optionString(cfg, "default_size", ""))) |d| {
        if (takes(cfg, d[0], d[1], l.frames)) return d;
    }
    if (engine.optionInt(cfg, "max_width", 0) > 0 or engine.optionInt(cfg, "max_height", 0) > 0) {
        const d = legacySize(cfg);
        if (takes(cfg, d[0], d[1], l.frames)) return d;
    }
    return widest(cfg);
}

/// A video without a size gets `defaultSize`.
fn defaults(cfg: *const ToolConfig, req: *Request) void {
    switch (req.*) {
        .video => |*v| if (v.width == 0 and v.height == 0) {
            const d = defaultSize(cfg);
            v.width = d[0];
            v.height = d[1];
        },
        .image, .asset3d, .speech, .transcription, .chat, .decision => {},
    }
}

/// Refuses what `generate` would, before the 40 GB load evicts anything: a size off the 32-pixel grid or past the long
/// side, a clip longer than `max_seconds`, a frame and length past the pipeline's buffers, a first frame whose canvas
/// leaves the prompt no text tokens, or another frame rate.
fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    const v = switch (req.*) {
        .video => |x| x,
        .image, .asset3d, .speech, .transcription, .chat, .decision => return true, // the kind check refuses it
    };
    const l = limits(cfg);
    var seconds = l.frames / fps;
    while (seconds > 0 and !takes(cfg, v.width, v.height, seconds * fps)) seconds -= 1;
    if (seconds == 0) {
        const d = defaultSize(cfg);
        const w = widest(cfg);
        why.print("model '{s}' takes sizes up to {d} pixels ({d}x{d} or {d}x{d}), at most {d} a side, multiples of 32 (omit size for {d}x{d})", .{ cfg.id, l.pixels, w[0], w[1], w[1], w[0], maxSide(cfg), d[0], d[1] }) catch {};
        return false;
    }
    if (v.seconds > seconds) {
        if (seconds == l.frames / fps) {
            why.print("model '{s}' takes clips up to {d} seconds", .{ cfg.id, seconds }) catch {};
        } else {
            why.print("model '{s}' takes {d}x{d} for clips up to {d} seconds (up to {d} seconds at {d} pixels or fewer)", .{ cfg.id, v.width, v.height, seconds, l.frames / fps, l.pixels }) catch {};
        }
        return false;
    }
    if (v.first_frame != null and !P.firstFrameFits(l, v.width, v.height)) {
        why.print("model '{s}' takes a first frame of up to {d} 32x32 cells: {d}x{d} is {d}, which leaves the prompt no room in the text encoder; send a smaller size", .{ cfg.id, P.firstFrameCells(l), v.width, v.height, (v.width / 32) * (v.height / 32) }) catch {};
        return false;
    }
    if (v.fps != fps) {
        why.print("model '{s}' makes {d} fps video: omit fps or send {d}", .{ cfg.id, fps, fps }) catch {};
        return false;
    }
    return true;
}

/// The device bytes besides the configured buffers (`P.scratchBytes`): the weights, the CUDA context and kernels. The rest
/// of the measured peak (`default_resident_mb` for the default limits) once their buffers and a full clip's request
/// scratch are taken out.
fn fixedBytes() u64 {
    const l: P.Limits = .{ .pixels = 768 * 448, .frames = 5 * fps };
    const measured = (P.scratchBytes(l) catch unreachable) + (P.requestBytes(768, 448, l.frames, false) catch unreachable);
    return (default_resident_mb << 20) - measured;
}

/// What the tool holds while loaded: the weights and the buffers for its limits, or `resident_mb` where that is more.
fn residentBytes(cfg: *const ToolConfig) u64 {
    const scratch = P.scratchBytes(limits(cfg)) catch std.math.maxInt(u64) / 2;
    return @max(fixedBytes() +| scratch, engine.optionInt(cfg, "resident_mb", 0) <<| 20);
}

/// A request's memory besides `residentBytes`: for the whole job the frames and the waveform (and its interleaved copy)
/// on the host, a first frame's pixels and ffmpeg's buffers; with them the pipeline's largest phase (`P.requestBytes`).
fn workingBytes(v: @import("../tool/request.zig").Video) u64 {
    const sh = P.shapes(v.width, v.height, v.seconds * fps) catch return 0;
    const phase = P.requestBytes(v.width, v.height, v.seconds * fps, v.first_frame != null) catch return 0;
    const pixels: u64 = if (v.first_frame != null) 3 * @as(u64, v.width) * v.height * 4 else 0;
    return sh.frame_bytes + sh.audioFloats() * 4 * 2 + pixels + (64 << 20) + phase;
}

fn needs(cfg: *const ToolConfig, req: *const Request) engine.Needs {
    const working: u64 = switch (req.*) {
        .image, .asset3d, .speech, .transcription, .chat, .decision => 0,
        .video => |v| workingBytes(v),
    };
    return .{ .resident = residentBytes(cfg), .working = working };
}

/// The most memory a request the tool takes can book: its resident bytes and the largest working set among the clips
/// `check` admits (a first frame's included).
fn peakBytes(cfg: *const ToolConfig) u64 {
    const l = limits(cfg);
    const side = maxSide(cfg);
    var most: u64 = 0;
    var w: u32 = min_side;
    while (w <= side) : (w += 32) {
        var h: u32 = min_side;
        while (h <= side) : (h += 32) {
            var seconds = l.frames / fps;
            while (seconds > 0 and !takes(cfg, w, h, seconds * fps)) seconds -= 1;
            if (seconds == 0) continue;
            most = @max(most, workingBytes(.{ .prompt = "", .width = w, .height = h, .seconds = seconds, .first_frame = if (P.firstFrameFits(l, w, h)) "" else null }));
        }
    }
    return residentBytes(cfg) +| most;
}

/// Refuses, when the daemon starts, limits outside what the API and the model take, and limits whose buffers and
/// largest request do not fit `budget` (0: unknown, not checked).
fn validate(cfg: *const ToolConfig, budget: u64, why: *std.Io.Writer) bool {
    const Range = struct { key: []const u8, min: u64, max: u64 };
    for ([_]Range{
        .{ .key = "max_pixels", .min = min_side * min_side, .max = max_pixels },
        .{ .key = "max_width", .min = min_side, .max = api.max_side },
        .{ .key = "max_height", .min = min_side, .max = api.max_side },
        .{ .key = "max_side", .min = min_side, .max = std.math.maxInt(i64) },
        .{ .key = "max_seconds", .min = 1, .max = api.max_seconds },
    }) |r| {
        const v = engine.optionInt(cfg, r.key, r.min);
        if (v < r.min or v > r.max) {
            why.print("tool '{s}': options.{s} is {d}; it takes {d} to {d}", .{ cfg.id, r.key, v, r.min, r.max }) catch {};
            return false;
        }
    }
    if (budget == 0) return true;
    const peak = peakBytes(cfg);
    if (peak <= budget) return true;
    why.print("tool '{s}': its limits need {d} MiB ({d} MiB loaded, the rest for its largest clip), more than the {d} MiB budget: lower max_pixels or max_seconds", .{ cfg.id, peak >> 20, residentBytes(cfg) >> 20, budget >> 20 }) catch {};
    return false;
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
        return residentBytes(t.cfg);
    }

    fn generate(ptr: *anyopaque, job: *const engine.Job, sink: engine.Sink, arena: std.mem.Allocator) anyerror!engine.Output {
        const t: *MiniMaxH3 = @ptrCast(@alignCast(ptr));
        const v = switch (job.request) {
            .video => |x| x,
            .image, .asset3d, .speech, .transcription, .chat, .decision => return engine.Refused.Refused,
        };
        if (v.fps != fps) return refuse("this model makes {d} fps video: omit fps or send {d}", .{ fps, fps });
        const p = t.p orelse return error.NotLoaded;
        const start = std.Io.Clock.awake.now(t.env.io);
        const steps = if (v.steps == 0) @as(u32, @intCast(engine.optionInt(t.cfg, "steps", default_steps))) else v.steps;
        const sh = P.shapes(v.width, v.height, v.seconds * fps) catch return refuse("{d}x{d} for {d} s is not a shape this model takes: omit size and seconds for its defaults", .{ v.width, v.height, v.seconds });
        const gpa = t.env.gpa;
        const frames = try gpa.alloc(u8, sh.frame_bytes);
        defer gpa.free(frames);
        const audio = try gpa.alloc(f32, sh.audioFloats());
        defer gpa.free(audio);
        var bridge: Bridge = .{ .sink = sink };
        var rgb: ?[]f32 = null;
        defer if (rgb) |pixels| gpa.free(pixels);
        if (v.first_frame) |name| {
            // as ComfyUI's LoadImage: RGB, the alpha dropped; cropped to the clip's shape (centred), then resized to it
            const in = image_in.load(t.env.io, arena, job.dir, name) catch |err| return refuse("the first frame could not be read ({s}): send a PNG, JPEG or WebP", .{@errorName(err)});
            const pix = try coverCrop(arena, in, v.width, v.height);
            rgb = try gpa.alloc(f32, @as(usize, v.width) * v.height * 3);
            h3.image.canvas(gpa, pix, v.width, v.height, rgb.?) catch |err| return refuse("the first frame of {d}x{d} cannot be fitted to {d}x{d} ({s})", .{ in.width, in.height, v.width, v.height, @errorName(err) });
        }
        _ = p.generateImage(v.prompt, v.width, v.height, v.seconds * fps, steps, v.seed, rgb, frames, audio, bridge.progress()) catch |err| switch (err) {
            error.SizeNotMultipleOf32 => return refuse("size {d}x{d}: each side must be a multiple of 32", .{ v.width, v.height }),
            error.RequestTooLarge => return refuse("{d}x{d} for {d} s is more than this model takes: lower size or seconds", .{ v.width, v.height, v.seconds }),
            error.PromptTooLong => return refuse("the prompt (with the first frame's tokens) is longer than the text encoder takes: shorten it", .{}),
            error.BadVisionCanvas, error.VisionCanvasTooLarge, error.BadImageCanvas => return refuse("the first frame does not fit a {d}x{d} canvas: send another size", .{ v.width, v.height }),
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
        var w = try mp4.Writer.open(t.env.io, arena, out, .{ .width = v.width, .height = v.height, .fps = fps, .wav = wav, .marker = .{ .model = t.cfg.id } });
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

/// The image's RGB (alpha dropped) cut to the clip's aspect ratio, centred, so the resize to the clip keeps its shape:
/// ComfyUI's H3 node stretches a first frame of another shape (`crop="disabled"`); an API caller rarely means that.
fn coverCrop(a: std.mem.Allocator, in: image_in.Image, w: u32, h: u32) !h3.image.Rgb {
    // the largest window of the clip's aspect: width-limited or height-limited
    var cw: u64 = in.width;
    var ch: u64 = @as(u64, in.width) * h / w;
    if (ch > in.height) {
        ch = in.height;
        cw = @as(u64, in.height) * w / h;
    }
    cw = @max(cw, 1);
    ch = @max(ch, 1);
    const x0 = (in.width - cw) / 2;
    const y0 = (in.height - ch) / 2;
    const out = try a.alloc(u8, cw * ch * 3);
    for (0..ch) |y| for (0..cw) |x| {
        @memcpy(out[(y * cw + x) * 3 ..][0..3], in.rgba[((y0 + y) * in.width + x0 + x) * 4 ..][0..3]);
    };
    return .{ .width = @intCast(cw), .height = @intCast(ch), .data = out };
}

test "a first frame of another shape is cut to the clip's, centred; one of the same shape is kept whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const px = try a.alloc(u8, 8 * 8 * 4);
    for (px, 0..) |*p, i| p.* = @intCast(i / 4 % 256); // pixel index in every channel
    const sq: image_in.Image = .{ .width = 8, .height = 8, .rgba = px };
    const wide = try coverCrop(a, sq, 16, 8); // 2:1 from 8x8: 8x4, rows 2..5
    try std.testing.expectEqual([2]u32{ 8, 4 }, [2]u32{ wide.width, wide.height });
    try std.testing.expectEqual(@as(u8, 16), wide.data[0]);
    const same = try coverCrop(a, sq, 32, 32);
    try std.testing.expectEqual([2]u32{ 8, 8 }, [2]u32{ same.width, same.height });
    try std.testing.expectEqual(@as(u8, 0), same.data[0]);
}

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
    try std.testing.expectEqualStrings("model 'h3' takes sizes up to 344064 pixels (768x448 or 448x768), at most 1344 a side, multiples of 32 (omit size for 768x448)", why.buffered());
    why = .fixed(&msg);
    const long: Request = .{ .video = .{ .prompt = "waves", .width = 768, .height = 448, .seconds = 6 } };
    try std.testing.expect(!check(cfg, &long, &why));
    try std.testing.expectEqualStrings("model 'h3' takes clips up to 5 seconds", why.buffered());
}

test "default_size: a video without a size gets it, within the limits; refusals name the limits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "h3", "kind": "video", "engine": "minimax_h3", "options": {"max_width": 1280, "max_height": 704, "max_seconds": 8, "default_size": "768x448"}},
        \\           {"id": "bad", "kind": "video", "engine": "minimax_h3", "options": {"max_width": 768, "max_height": 448, "default_size": "1280x704"}}]}
    );
    var r: Request = .{ .video = .{ .prompt = "waves", .seconds = 8 } };
    engine.fill(c.tool("h3").?, &r);
    try std.testing.expectEqual([2]u32{ 768, 448 }, [2]u32{ r.video.width, r.video.height });
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    try std.testing.expect(check(c.tool("h3").?, &r, &why));
    const big: Request = .{ .video = .{ .prompt = "waves", .width = 1280, .height = 720, .seconds = 8 } };
    try std.testing.expect(!check(c.tool("h3").?, &big, &why));
    try std.testing.expectEqualStrings("model 'h3' takes sizes up to 901120 pixels (1248x704 or 704x1248), at most 1344 a side, multiples of 32 (omit size for 768x448)", why.buffered());
    var b: Request = .{ .video = .{ .prompt = "waves" } };
    engine.fill(c.tool("bad").?, &b);
    try std.testing.expectEqual([2]u32{ 768, 448 }, [2]u32{ b.video.width, b.video.height }); // past the limits: the largest
}

test "one area budget takes landscape and portrait; a larger frame takes a shorter clip; the long side is the model's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "h3", "kind": "video", "engine": "minimax_h3", "options": {"max_pixels": 901120, "max_seconds": 8}},
        \\           {"id": "old", "kind": "video", "engine": "minimax_h3", "options": {"max_width": 1280, "max_height": 704, "max_seconds": 8}},
        \\           {"id": "canvas", "kind": "video", "engine": "minimax_h3", "options": {"max_pixels": 1032192, "max_seconds": 5}}]}
    );
    const Case = struct { tool: []const u8, w: u32, h: u32, s: u32, ok: bool };
    const cases = [_]Case{
        .{ .tool = "h3", .w = 1280, .h = 704, .s = 8, .ok = true },
        .{ .tool = "h3", .w = 704, .h = 1248, .s = 8, .ok = true },
        .{ .tool = "old", .w = 1280, .h = 704, .s = 8, .ok = true },
        .{ .tool = "old", .w = 704, .h = 1248, .s = 8, .ok = true },
        .{ .tool = "h3", .w = 768, .h = 1344, .s = 6, .ok = true },
        .{ .tool = "h3", .w = 768, .h = 1344, .s = 7, .ok = false },
        .{ .tool = "canvas", .w = 768, .h = 1344, .s = 5, .ok = true },
        .{ .tool = "canvas", .w = 1344, .h = 768, .s = 5, .ok = true },
        .{ .tool = "canvas", .w = 640, .h = 1376, .s = 1, .ok = false }, // past the long side, within the area
        .{ .tool = "h3", .w = 960, .h = 1024, .s = 8, .ok = false },
        .{ .tool = "h3", .w = 960, .h = 1024, .s = 1, .ok = true }, // past the area, for a clip that fits the same buffers
        .{ .tool = "h3", .w = 1024, .h = 1024, .s = 1, .ok = false }, // past the canvas's area
        .{ .tool = "h3", .w = 720, .h = 1280, .s = 1, .ok = false },
        .{ .tool = "h3", .w = 704, .h = 1250, .s = 1, .ok = false },
    };
    var msg: [256]u8 = undefined;
    for (cases) |k| {
        var why: std.Io.Writer = .fixed(&msg);
        const r: Request = .{ .video = .{ .prompt = "waves", .width = k.w, .height = k.h, .seconds = k.s } };
        std.testing.expectEqual(k.ok, check(c.tool(k.tool).?, &r, &why)) catch |err| {
            std.debug.print("{s} {d}x{d} {d} s: {s}\n", .{ k.tool, k.w, k.h, k.s, why.buffered() });
            return err;
        };
    }
    var why: std.Io.Writer = .fixed(&msg);
    const tall: Request = .{ .video = .{ .prompt = "waves", .width = 768, .height = 1344, .seconds = 8 } };
    try std.testing.expect(!check(c.tool("h3").?, &tall, &why));
    try std.testing.expectEqualStrings("model 'h3' takes 768x1344 for clips up to 6 seconds (up to 8 seconds at 901120 pixels or fewer)", why.buffered());
    var d: Request = .{ .video = .{ .prompt = "waves" } };
    engine.fill(c.tool("h3").?, &d);
    try std.testing.expectEqual([2]u32{ 1248, 704 }, [2]u32{ d.video.width, d.video.height });
}

test "a first frame whose canvas fills the text encoder is refused before loading; the canvas sizes are taken" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "h3", "kind": "video", "engine": "minimax_h3", "options": {"max_pixels": 901120, "max_seconds": 8}},
        \\           {"id": "big", "kind": "video", "engine": "minimax_h3", "options": {"max_pixels": 1638400, "max_seconds": 1}}]}
    );
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    const past: Request = .{ .video = .{ .prompt = "waves", .width = 1344, .height = 1344, .seconds = 1, .first_frame = "f.png" } };
    try std.testing.expect(!check(c.tool("h3").?, &past, &why)); // past the canvas's area: a size refusal
    why = .fixed(&msg);
    const tall: Request = .{ .video = .{ .prompt = "waves", .width = 768, .height = 1344, .seconds = 6, .first_frame = "f.png" } };
    try std.testing.expect(check(c.tool("h3").?, &tall, &why));
    const square: Request = .{ .video = .{ .prompt = "waves", .width = 1280, .height = 1280, .seconds = 1, .first_frame = "f.png" } };
    try std.testing.expect(!check(c.tool("big").?, &square, &why));
    try std.testing.expectEqualStrings("model 'big' takes a first frame of up to 1519 32x32 cells: 1280x1280 is 1600, which leaves the prompt no room in the text encoder; send a smaller size", why.buffered());
    var t2v = square;
    t2v.video.first_frame = null;
    why = .fixed(&msg);
    try std.testing.expect(check(c.tool("big").?, &t2v, &why));
}

test "defaults: legacy limits keep their size where it fits; otherwise one the API and the tool take" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "wide", "kind": "video", "engine": "minimax_h3", "options": {"max_width": 1024}},
        \\           {"id": "shrunk", "kind": "video", "engine": "minimax_h3", "options": {"max_width": 1280, "max_height": 704, "max_pixels": 344064}},
        \\           {"id": "narrow", "kind": "video", "engine": "minimax_h3", "options": {"max_side": 384}},
        \\           {"id": "plain", "kind": "video", "engine": "minimax_h3"}]}
    );
    const Case = struct { tool: []const u8, size: [2]u32 };
    for ([_]Case{
        .{ .tool = "wide", .size = .{ 1024, 448 } },
        .{ .tool = "shrunk", .size = .{ 768, 448 } },
        .{ .tool = "narrow", .size = .{ 384, 384 } },
        .{ .tool = "plain", .size = .{ 768, 448 } },
    }) |k| {
        const cfg = c.tool(k.tool).?;
        var r: Request = .{ .video = .{ .prompt = "waves" } };
        engine.fill(cfg, &r);
        try std.testing.expectEqual(k.size, [2]u32{ r.video.width, r.video.height });
        var why_buf: []const u8 = "";
        try @import("../tool/request.zig").validate(r, .{}, &why_buf);
        var msg: [256]u8 = undefined;
        var why: std.Io.Writer = .fixed(&msg);
        try std.testing.expect(check(cfg, &r, &why));
    }
}

/// Refuses the request (the API's 400) with a reason the caller can act on.
fn refuse(comptime fmt: []const u8, args: anytype) engine.Refused {
    engine.explain(fmt, args);
    return engine.Refused.Refused;
}

test "the booking follows the configured buffers; limits past the API or the memory budget are refused at load" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "plain", "kind": "video", "engine": "minimax_h3"},
        \\           {"id": "jarvis", "kind": "video", "engine": "minimax_h3", "options": {"max_pixels": 901120, "max_seconds": 8, "default_size": "768x448", "resident_mb": 53567}},
        \\           {"id": "huge", "kind": "video", "engine": "minimax_h3", "options": {"max_side": 2048, "max_pixels": 4194304, "max_seconds": 20}},
        \\           {"id": "wide", "kind": "video", "engine": "minimax_h3", "options": {"max_pixels": 4294967296}},
        \\           {"id": "long", "kind": "video", "engine": "minimax_h3", "options": {"max_seconds": 9999999999999}}]}
    );
    const budget: u64 = 113 << 30; // a GB10's 121 GiB less the default reserve
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    const plain = c.tool("plain").?;
    try std.testing.expectEqual(@as(u64, default_resident_mb << 20), residentBytes(plain) + try P.requestBytes(768, 448, 5 * fps, false)); // the measured peak
    try std.testing.expect(validate(plain, budget, &why));
    try std.testing.expect(validate(c.tool("jarvis").?, budget, &why));

    const huge = c.tool("huge").?;
    const clip: Request = .{ .video = .{ .prompt = "waves", .width = 2048, .height = 2048, .seconds = 20 } };
    try std.testing.expect(check(huge, &clip, &why));
    const n = needs(huge, &clip);
    try std.testing.expect(n.resident + n.working > 135 << 30); // the DiT's and the pipeline's buffers alone
    try std.testing.expect(!validate(huge, budget, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.buffered(), "more than the 115712 MiB budget") != null);

    for ([_][]const u8{ "wide", "long" }) |id| {
        const cfg = c.tool(id).?;
        var r: Request = .{ .video = .{ .prompt = "waves", .width = 768, .height = 448, .seconds = 2 } };
        engine.fill(cfg, &r);
        why = .fixed(&msg);
        _ = check(cfg, &r, &why);
        _ = needs(cfg, &r);
        why = .fixed(&msg);
        try std.testing.expect(!validate(cfg, 0, &why));
    }
    try std.testing.expectEqualStrings("tool 'long': options.max_seconds is 9999999999999; it takes 1 to 20", why.buffered());
}

test "a request books its largest phase on top of its host buffers: the decode with the latents on the host, or a first frame's encode or condition" {
    const sh = try P.shapes(2048, 2048, 5 * fps);
    const decode = h3.vae_video.decodeBytes(sh.latent_t, sh.lh, sh.lw) + sh.n_video * (4 + 2); // the fp32 and fp16 latents on the host
    const host = sh.frame_bytes + sh.audioFloats() * 4 * 2 + (64 << 20);
    try std.testing.expect(workingBytes(.{ .prompt = "", .width = 2048, .height = 2048, .seconds = 5 }) >= host + decode);
    const s2 = try P.shapes(768, 1344, 6 * fps);
    const pixels: u64 = 3 * 768 * 1344 * 4;
    const latent: u64 = 24 * @as(u64, s2.lh) * s2.lw * 4;
    const host2 = s2.frame_bytes + s2.audioFloats() * 4 * 2 + (64 << 20) + pixels;
    const i2v = workingBytes(.{ .prompt = "", .width = 768, .height = 1344, .seconds = 6, .first_frame = "" });
    try std.testing.expect(i2v >= host2 + h3.vae_video.decodeBytes(s2.latent_t, s2.lh, s2.lw) + s2.n_video * (4 + 2));
    try std.testing.expect(i2v >= host2 + h3.vae_video_enc.Encoder.deviceBytes(1344, 768) + h3.te_vision.VisionEncoder.deviceBytes(s2.lh * s2.lw) + pixels + latent);
    try std.testing.expect(i2v >= host2 + latent + h3.dit.Dit.conditionBytes(s2.lh, s2.lw));
}

test "the default size is within the API's longest side, as checked sizes are" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "h3", "kind": "video", "engine": "minimax_h3", "options": {"max_side": 4096, "max_pixels": 532480, "default_size": "2080x256"}}]}
    );
    const cfg = c.tool("h3").?;
    var r: Request = .{ .video = .{ .prompt = "waves" } };
    engine.fill(cfg, &r);
    var why_buf: []const u8 = "";
    try @import("../tool/request.zig").validate(r, .{}, &why_buf);
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    try std.testing.expect(check(cfg, &r, &why));
    const explicit: Request = .{ .video = .{ .prompt = "waves", .width = 2048, .height = 256 } };
    try std.testing.expect(check(cfg, &explicit, &why));
}
