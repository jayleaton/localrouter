//! The `qwen_image` and `qwen_image_turbo` engines: Qwen-Image 2.1 and Qwen-Image 2.1 Turbo text to image on
//! LocalRouter's Zig pipeline (`qwen_image.pipeline`). Turbo is the same 7B DiT with its own weights and the 8-step
//! schedule its checkpoint stores (sample_sigmas, shift 1, CFG 1); it shares the text encoder and VAE packs.
//! `weights` is a directory of packs: the DiT's (`<precision>/` for Qwen-Image 2.1, `turbo-<precision>/` for Turbo, with
//! tfimage's Triton cubins in `triton/`), `te/` and `vae/` (`python -m stk_twin pack`). Options: `precision` (fp8s |
//! nvfp4, default fp8s), `dit` (the DiT pack's directory, default as above), `max_side` (pixels, the largest side a
//! request may ask for, default 1664), `max_pixels` (width x height at most, default max_side squared: the scratch,
//! about 4.6 KiB a pixel on GB10 FP8, is sized by it, so 1048576 serves 1024x1024, 1360x768 and 768x1360 in 8 GiB
//! less than max_side 1664 does), `steps` (default 25; Turbo 8, the only count its schedule has), `resident_mb`
//! (the scheduler's estimate). Neither runs classifier-free guidance: `guidance` and `negative_prompt` are not used.
//! Both also edit (`image_edit`: a prompt and 1 to `edit_refs` input images, default 3, optionally a mask) with the same
//! checkpoint, once the edit pack is installed (`edit.json` beside the packs, matching the `te/` and `vae/` manifests:
//! the vision tower and the VAE encoder, docs/dev/IMAGE-EDIT.md); without it they list text to image only. An edit
//! without a size gets the last image's shape at about max_pixels. A mask (alpha 0 where the edit goes, as OpenAI's)
//! keeps the first image's pixels wherever it is opaque.

const std = @import("std");
const engine = @import("../engine/engine.zig");
const png = @import("../media/png.zig");
const image_in = @import("../media/image_in.zig");
const qi = @import("qwen_image");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const entry: engine.Entry = .{ .name = "qwen_image", .capabilities = &.{ .text_to_image, .image_edit }, .available = available, .needs = needs, .create = create(.base), .check = check(.base), .license = license };
pub const turbo_entry: engine.Entry = .{ .name = "qwen_image_turbo", .capabilities = &.{ .text_to_image, .image_edit }, .available = available, .needs = needs, .create = create(.turbo), .check = check(.turbo), .license = license };

/// Both checkpoints' weights (the same LICENSE file at each revision), unless a tool's config names its own.
pub const license: @import("../config.zig").License = .{
    .name = "Qwen Research License Agreement",
    .url = "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
    .use = "non-commercial: research or evaluation only; commercial use needs a separate licence from Qwen (model-business@notice.qwencloud.com)",
    .commercial = false,
};

/// Editing needs the edit pack, valid for the current manifests and for `edit_refs` images; text to image always works.
fn available(io: std.Io, cfg: *const ToolConfig, cap: engine.Capability) bool {
    if (cap != .image_edit) return true;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const limit = qi.edit_pack.validate(arena.allocator(), io, cfg.weights) catch return false;
    const refs = editRefs(cfg);
    return refs > 0 and refs <= limit;
}

fn editRefs(cfg: *const ToolConfig) u32 {
    return @intCast(@min(engine.optionInt(cfg, "edit_refs", 3), 5));
}

const default_precision = "fp8s"; // closer to the original model; "nvfp4" is the faster option
const default_max_side = 1664;

/// Which checkpoint a tool serves: its DiT pack's default directory and step count.
const Variant = enum {
    base,
    turbo,

    fn defaultSteps(v: Variant) u64 {
        return switch (v) {
            .base => 25,
            .turbo => 8, // the length of the checkpoint's sample_sigmas; the pack's schedule is checked at load
        };
    }
    /// The steps a request without its own count gets.
    fn steps(v: Variant, cfg: *const ToolConfig) u32 {
        return @intCast(engine.optionInt(cfg, "steps", v.defaultSteps()));
    }
    /// The DiT pack's directory under `weights`.
    fn ditDir(v: Variant, cfg: *const ToolConfig, buf: []u8) []const u8 {
        const precision = engine.optionString(cfg, "precision", default_precision);
        const dflt = switch (v) {
            .base => precision,
            .turbo => std.fmt.bufPrint(buf, "turbo-{s}", .{precision}) catch precision,
        };
        return engine.optionString(cfg, "dit", dflt);
    }
};

fn maxSide(cfg: *const ToolConfig) u32 {
    return @intCast(engine.optionInt(cfg, "max_side", default_max_side) / 16 * 16);
}

/// Width x height at most (multiples of 256): `max_pixels`, else max_side squared.
fn maxPixels(cfg: *const ToolConfig) u64 {
    const side: u64 = maxSide(cfg);
    return @min(engine.optionInt(cfg, "max_pixels", side * side), side * side) / 256 * 256;
}

/// Refuses before loading what the engine would refuse after: a side past `max_side`, more pixels than `max_pixels`,
/// and for a checkpoint that stores its schedule (Turbo) another step count.
fn check(comptime v: Variant) fn (cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    return struct {
        fn f(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
            const img = switch (req.*) {
                .image => |x| x,
                .video, .speech, .transcription, .chat, .decision => return true, // the kind check refuses it
            };
            if (img.references.len > editRefs(cfg)) {
                why.print("model '{s}' edits with up to {d} input image{s}", .{ cfg.id, editRefs(cfg), if (editRefs(cfg) == 1) "" else "s" }) catch {};
                return false;
            }
            if (img.mask != null and img.references.len == 0) {
                why.print("a mask needs an input image to edit", .{}) catch {};
                return false;
            }
            if (img.width == 0 and img.height == 0) return img.references.len > 0 or failSize(cfg, why); // an edit's own size
            if (@max(img.width, img.height) > maxSide(cfg)) {
                why.print("model '{s}' takes sizes up to {d} pixels a side", .{ cfg.id, maxSide(cfg) }) catch {};
                return false;
            }
            if (@as(u64, img.width) * img.height > maxPixels(cfg)) {
                why.print("model '{s}' takes at most {d} pixels (width x height), such as 1024x1024 or 1360x768", .{ cfg.id, maxPixels(cfg) }) catch {};
                return false;
            }
            if (v == .turbo and img.steps != 0 and img.steps != v.steps(cfg)) {
                why.print("model '{s}' runs its checkpoint's {d}-step schedule: omit steps or send {d}", .{ cfg.id, v.steps(cfg), v.steps(cfg) }) catch {};
                return false;
            }
            return true;
        }
    }.f;
}

test "turbo fixes its steps and DiT pack; both refuse oversized requests before loading" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [
        \\ {"id": "base", "kind": "image", "engine": "qwen_image"},
        \\ {"id": "turbo", "kind": "image", "engine": "qwen_image_turbo"},
        \\ {"id": "turbo4", "kind": "image", "engine": "qwen_image_turbo", "options": {"precision": "nvfp4", "max_side": 1024}},
        \\ {"id": "own", "kind": "image", "engine": "qwen_image_turbo", "options": {"dit": "mine"}},
        \\ {"id": "mp", "kind": "image", "engine": "qwen_image_turbo", "options": {"max_side": 1360, "max_pixels": 1048576}}
        \\]}
    );
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("fp8s", Variant.base.ditDir(c.tool("base").?, &buf));
    try std.testing.expectEqualStrings("turbo-fp8s", Variant.turbo.ditDir(c.tool("turbo").?, &buf));
    try std.testing.expectEqualStrings("turbo-nvfp4", Variant.turbo.ditDir(c.tool("turbo4").?, &buf));
    try std.testing.expectEqualStrings("mine", Variant.turbo.ditDir(c.tool("own").?, &buf));
    try std.testing.expectEqual(@as(u32, 25), Variant.base.steps(c.tool("base").?));
    try std.testing.expectEqual(@as(u32, 8), Variant.turbo.steps(c.tool("turbo").?));
    try std.testing.expectEqualStrings("qwen_image_turbo", turbo_entry.name);

    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    const ok: Request = .{ .image = .{ .prompt = "a fox", .width = 1664, .height = 928 } };
    try std.testing.expect(check(.turbo)(c.tool("turbo").?, &ok, &why));
    const eight: Request = .{ .image = .{ .prompt = "a fox", .steps = 8 } };
    try std.testing.expect(check(.turbo)(c.tool("turbo").?, &eight, &why));
    const many: Request = .{ .image = .{ .prompt = "a fox", .steps = 25 } };
    try std.testing.expect(check(.base)(c.tool("base").?, &many, &why));
    try std.testing.expect(!check(.turbo)(c.tool("turbo").?, &many, &why));
    try std.testing.expectEqualStrings("model 'turbo' runs its checkpoint's 8-step schedule: omit steps or send 8", why.buffered());
    why = .fixed(&msg);
    const big: Request = .{ .image = .{ .prompt = "a fox", .width = 1280, .height = 768 } };
    try std.testing.expect(!check(.turbo)(c.tool("turbo4").?, &big, &why));
    try std.testing.expectEqualStrings("model 'turbo4' takes sizes up to 1024 pixels a side", why.buffered());
    // max_pixels: about one megapixel in any shape the side allows, and not more
    try std.testing.expectEqual(@as(u64, 1664 * 1664), maxPixels(c.tool("turbo").?));
    try std.testing.expectEqual(@as(u64, 1048576), maxPixels(c.tool("mp").?));
    for ([_][2]u32{ .{ 1024, 1024 }, .{ 1360, 768 }, .{ 768, 1360 } }) |wh| {
        const r: Request = .{ .image = .{ .prompt = "a fox", .width = wh[0], .height = wh[1] } };
        try std.testing.expect(check(.turbo)(c.tool("mp").?, &r, &why));
    }
    why = .fixed(&msg);
    const wide: Request = .{ .image = .{ .prompt = "a fox", .width = 1360, .height = 1024 } };
    try std.testing.expect(!check(.turbo)(c.tool("mp").?, &wide, &why));
    try std.testing.expectEqualStrings("model 'mp' takes at most 1048576 pixels (width x height), such as 1024x1024 or 1360x768", why.buffered());
}

/// The scheduler's resident estimate in MiB: GB10's measured engine peaks at 1024x1024 (M5, MemAvailable) are 30 GiB
/// NVFP4 and 32 GiB FP8 (the 16.4 GB text encoder, the DiT, the VAE and the scratch sized at max_side), plus margin.
fn failSize(cfg: *const ToolConfig, why: *std.Io.Writer) bool {
    why.print("model '{s}' needs a size for text to image", .{cfg.id}) catch {};
    return false;
}

/// An edit without a size: the last image's shape at about max_pixels (the reference pipeline's calculate_dimensions
/// at one megapixel, here at the model's area), shrunk 32 pixels of area at a time until it fits both limits.
fn editSize(cfg: *const ToolConfig, width: u32, height: u32) ![2]u32 {
    var area: u64 = @min(maxPixels(cfg), qi.pipeline.edit_ref_pixels);
    while (area >= 32 * 32) : (area -= 32 * 32) {
        const d = try qi.edit_prepare.dimensions(width, height, area);
        if (@max(d.width, d.height) <= maxSide(cfg) and @as(u64, d.width) * d.height <= maxPixels(cfg)) return .{ d.width, d.height };
    }
    return error.ImageTooLarge;
}

test "an edit without a size keeps the image's shape within the model's limits" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [{"id": "mp", "kind": "image", "engine": "qwen_image_turbo", "options": {"max_side": 1360, "max_pixels": 1048576}}]}
    );
    const cfg = c.tool("mp").?;
    try std.testing.expectEqual([2]u32{ 1024, 1024 }, try editSize(cfg, 512, 512));
    try std.testing.expectEqual([2]u32{ 1376 - 32, 768 }, try editSize(cfg, 1920, 1080));
    for ([_][2]u32{ .{ 928, 1184 }, .{ 3000, 1000 }, .{ 100, 4000 }, .{ 1, 1 } }) |wh| {
        const s = try editSize(cfg, wh[0], wh[1]);
        try std.testing.expect(@max(s[0], s[1]) <= 1360 and s[0] * s[1] <= 1048576 and s[0] % 32 == 0 and s[1] % 32 == 0);
    }
    var msg: [128]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    const many: Request = .{ .image = .{ .prompt = "night", .width = 0, .height = 0, .references = &.{ "a", "b", "c", "d" } } };
    try std.testing.expect(!check(.turbo)(cfg, &many, &why));
    try std.testing.expectEqualStrings("model 'mp' edits with up to 3 input images", why.buffered());
    const auto: Request = .{ .image = .{ .prompt = "night", .width = 0, .height = 0, .references = &.{"a"} } };
    try std.testing.expect(check(.turbo)(cfg, &auto, &why));
}

fn residentMb(cfg: *const ToolConfig) u64 {
    const default_mb: u64 = if (std.mem.eql(u8, engine.optionString(cfg, "precision", default_precision), "nvfp4")) 32_000 else 36_000;
    return engine.optionInt(cfg, "resident_mb", default_mb);
}

test "precision defaults to fp8s and its resident estimate follows the precision" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try @import("../config.zig").parse(arena.allocator(),
        \\{"tools": [
        \\ {"id": "a", "kind": "image", "engine": "qwen_image"},
        \\ {"id": "b", "kind": "image", "engine": "qwen_image", "options": {"precision": "nvfp4"}},
        \\ {"id": "c", "kind": "image", "engine": "qwen_image", "options": {"precision": "fp8s", "resident_mb": 1234}}
        \\]}
    );
    try std.testing.expectEqualStrings("fp8s", engine.optionString(c.tool("a").?, "precision", default_precision));
    try std.testing.expectEqual(@as(u64, 36_000), residentMb(c.tool("a").?));
    try std.testing.expectEqual(@as(u64, 32_000), residentMb(c.tool("b").?));
    try std.testing.expectEqual(@as(u64, 1234), residentMb(c.tool("c").?));
}

fn needs(cfg: *const ToolConfig, req: *const Request) engine.Needs {
    const working: u64 = switch (req.*) {
        // the pixels on the host and the PNG's copy (an edit's own size: at most max_pixels), plus each input decoded,
        // resized and patched on the host (about 32 bytes a pixel at one megapixel)
        .image => |v| (if (v.width == 0) maxPixels(cfg) else @as(u64, v.width) * v.height) * 4 * 2 + v.references.len * (qi.pipeline.edit_ref_pixels * 32 + (32 << 20)),
        .video, .speech, .transcription, .chat, .decision => 0,
    };
    return .{ .resident = residentMb(cfg) << 20, .working = working };
}

const QwenImage = struct {
    env: engine.Env,
    cfg: *const ToolConfig,
    variant: Variant,
    p: ?*qi.pipeline.Pipeline = null,
    arena: std.heap.ArenaAllocator,

    fn load(ptr: *anyopaque) anyerror!u64 {
        const t: *QwenImage = @ptrCast(@alignCast(ptr));
        const a = t.arena.allocator();
        const root = t.cfg.weights;
        var dir_buf: [64]u8 = undefined;
        const packs: qi.pipeline.Packs = .{
            .dit = try std.fs.path.join(a, &.{ root, t.variant.ditDir(t.cfg, &dir_buf) }),
            .te = try std.fs.path.join(a, &.{ root, "te" }),
            .vae = try std.fs.path.join(a, &.{ root, "vae" }),
        };
        t.p = if (available(t.env.io, t.cfg, .image_edit))
            try qi.pipeline.Pipeline.createEdit(t.env.gpa, t.env.io, packs, .pack, maxSide(t.cfg), maxPixels(t.cfg), editRefs(t.cfg))
        else
            try qi.pipeline.Pipeline.create(t.env.gpa, t.env.io, packs, .pack, maxSide(t.cfg), maxPixels(t.cfg));
        // a schedule stored with the checkpoint (Turbo's sample_sigmas) fixes the step count: the config must agree
        if (qi.sampler.fixedSteps(t.p.?.sched)) |n| if (n != t.variant.steps(t.cfg)) {
            std.log.err("{s}: the DiT pack's schedule has {d} steps, the tool's steps option {d}", .{ packs.dit, n, t.variant.steps(t.cfg) });
            return error.ScheduleStepsMismatch;
        };
        // where the load went (the worker's log): context, kernels, each component, read / wait / host time
        std.log.info("load ms: {s}", .{try std.json.Stringify.valueAlloc(a, t.p.?.load, .{})});
        return residentMb(t.cfg) << 20;
    }

    fn generate(ptr: *anyopaque, job: *const engine.Job, sink: engine.Sink, arena: std.mem.Allocator) anyerror!engine.Output {
        const t: *QwenImage = @ptrCast(@alignCast(ptr));
        const v = switch (job.request) {
            .image => |x| x,
            .video, .speech, .transcription, .chat, .decision => return engine.Refused.Refused,
        };
        const p = t.p orelse return error.NotLoaded;
        if (v.references.len > 0 and p.edit_state == null) return refuse("this model's edit pack is not installed: send no input images", .{});
        const start = std.Io.Clock.awake.now(t.env.io);
        const steps = if (v.steps == 0) t.variant.steps(t.cfg) else v.steps;
        const inputs = try arena.alloc(image_in.Image, v.references.len);
        const refs = try arena.alloc(qi.edit_prepare.Prepared, v.references.len);
        for (v.references, inputs, refs) |name, *in, *ref| {
            in.* = image_in.load(t.env.io, arena, job.dir, name) catch |err| return refuse("an input image could not be read ({s}): send a PNG, JPEG or WebP", .{@errorName(err)});
            ref.* = qi.edit_prepare.prepare(arena, in.rgba, in.width, in.height, qi.pipeline.edit_ref_pixels) catch return refuse("an input image of {d}x{d} cannot be edited: send one with sides of at least 32 pixels", .{ in.width, in.height });
        }
        var width = v.width;
        var height = v.height;
        if (width == 0 and refs.len > 0) {
            const last = inputs[inputs.len - 1];
            const wh = editSize(t.cfg, last.width, last.height) catch return refuse("no size this model takes keeps the input's shape: send size", .{});
            width = wh[0];
            height = wh[1];
        }
        var keep: ?Keep = null;
        if (v.mask) |name| keep = Keep.init(t.env.io, arena, job.dir, name, inputs[0], width, height) catch |err| return refuse("the mask could not be used ({s}): send a PNG of the first image's size, transparent where the edit goes", .{@errorName(err)});
        const pixels = try t.env.gpa.alloc(u8, @as(usize, width) * height * p.vae.out_channels);
        defer t.env.gpa.free(pixels);
        const names = try arena.alloc([]const u8, v.n);
        var bridge: Bridge = .{ .sink = sink };
        for (names, 0..) |*name, i| {
            const seed = v.seed +% i;
            const made = if (refs.len > 0) p.edit(v.prompt, refs, height, width, steps, seed, pixels, bridge.progress(), null) else p.generate(v.prompt, height, width, steps, seed, pixels, bridge.progress());
            _ = made catch |err| switch (err) {
                error.ImageTooLarge => return refuse("{d}x{d} is larger than this model takes: lower size", .{ width, height }),
                error.SizeNotMultipleOf16 => return refuse("size {d}x{d}: each side must be a multiple of 16", .{ width, height }),
                error.PromptTooLong => return refuse("the prompt (with the input images' tokens) is longer than the text encoder takes: shorten it", .{}),
                error.BadReferences => return refuse("this model edits with up to {d} input images", .{editRefs(t.cfg)}),
                else => return err,
            };
            if (keep) |k| k.apply(pixels, p.vae.out_channels);
            name.* = try std.fmt.allocPrint(arena, "{d}.png", .{i});
            var seed_buf: [24]u8 = undefined;
            const seed_s = try std.fmt.bufPrint(&seed_buf, "{d}", .{seed});
            var file = try std.Io.Dir.cwd().createFile(t.env.io, try std.fs.path.join(arena, &.{ job.dir, name.* }), .{});
            defer file.close(t.env.io);
            var buf: [64 * 1024]u8 = undefined;
            var fw = file.writer(t.env.io, &buf);
            const fmt: png.Format = if (p.vae.out_channels == 4) .rgba8 else .rgb8;
            try png.encode(&fw.interface, pixels, width, height, fmt, .{ .level = .fastest, .text = &.{ .{ .key = "prompt", .value = v.prompt }, .{ .key = "seed", .value = seed_s } }, .marker = .{ .model = t.cfg.id } });
            try fw.interface.flush();
        }
        const ms = start.durationTo(std.Io.Clock.awake.now(t.env.io)).toMilliseconds();
        return .{ .files = names, .seed = v.seed, .ms = @intCast(ms) };
    }

    fn unload(ptr: *anyopaque) void {
        const t: *QwenImage = @ptrCast(@alignCast(ptr));
        if (t.p) |p| p.destroy();
        t.arena.deinit();
        t.env.gpa.destroy(t);
    }
};

/// A mask's composite: the first input image at the output's size, kept wherever the mask is opaque (OpenAI's edits:
/// alpha 0 marks where the edit goes). The mask must be the first image's size; both are resized as the references
/// are (premultiplied Lanczos), and the mask's alpha blends the two linearly at the edges.
const Keep = struct {
    original: []const u8, // RGBA8 at the output's size
    alpha: []const u8, // the mask's alpha at the output's size

    fn init(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8, first: image_in.Image, width: u32, height: u32) !Keep {
        const m = try image_in.load(io, a, dir, name);
        if (m.width != first.width or m.height != first.height) return error.MaskSizeMismatch;
        const original = try qi.edit_prepare.resizeRgba(a, first.rgba, first.width, first.height, width, height);
        // the alpha alone, as an opaque grey image, so premultiplication leaves it as it is
        const grey = try a.alloc(u8, m.rgba.len);
        for (0..@as(usize, m.width) * m.height) |i| {
            @memset(grey[i * 4 ..][0..3], m.rgba[i * 4 + 3]);
            grey[i * 4 + 3] = 255;
        }
        const scaled = try qi.edit_prepare.resizeRgba(a, grey, m.width, m.height, width, height);
        const alpha = try a.alloc(u8, @as(usize, width) * height);
        for (alpha, 0..) |*x, i| x.* = scaled[i * 4];
        return .{ .original = original, .alpha = alpha };
    }

    fn apply(k: Keep, pixels: []u8, channels: u32) void {
        for (k.alpha, 0..) |al, i| for (0..channels) |c| {
            const o = k.original[i * 4 + c];
            const e = pixels[i * channels + c];
            pixels[i * channels + c] = @intCast((@as(u32, o) * al + @as(u32, e) * (255 - al) + 127) / 255);
        };
    }
};

test "a mask keeps the original where it is opaque and the edit where it is clear" {
    const k: Keep = .{ .original = &.{ 10, 20, 30, 255, 10, 20, 30, 255 }, .alpha = &.{ 255, 0 } };
    var px = [_]u8{ 200, 200, 200, 255, 200, 200, 200, 255 };
    k.apply(&px, 4);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 255, 200, 200, 200, 255 }, &px);
}

/// The pipeline's step reports as the tool's progress.
const Bridge = struct {
    sink: engine.Sink,

    fn progress(b: *Bridge) qi.pipeline.Progress {
        return .{ .ctx = b, .report = report };
    }
    fn report(ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void {
        const b: *Bridge = @ptrCast(@alignCast(ctx));
        b.sink.report(.{ .phase = phase, .step = step, .of = of });
    }
};

fn create(comptime v: Variant) fn (env: engine.Env, cfg: *const ToolConfig) anyerror!engine.Engine {
    return struct {
        fn f(env: engine.Env, cfg: *const ToolConfig) anyerror!engine.Engine {
            const t = try env.gpa.create(QwenImage);
            t.* = .{ .env = env, .cfg = cfg, .variant = v, .arena = .init(env.gpa) };
            return .{ .ptr = t, .vtable = &.{ .load = QwenImage.load, .generate = QwenImage.generate, .unload = QwenImage.unload } };
        }
    }.f;
}

/// Refuses the request (the API's 400) with a reason the caller can act on.
fn refuse(comptime fmt: []const u8, args: anytype) engine.Refused {
    engine.explain(fmt, args);
    return engine.Refused.Refused;
}
