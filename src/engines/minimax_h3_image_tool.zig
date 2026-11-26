//! The `minimax_h3_image` engine: MiniMax H3-Image text to image on LocalRouter's Zig pipeline
//! (`minimax_h3.pipeline_image`), one PNG an image (docs/dev/H3-IMAGE.md). `weights` is a directory laid out like ComfyUI's
//! models folder plus the pack: `<pack>/` (`python -m stk_twin.h3i.build`: the DiT in bf16 and `tokenizer.json`),
//! `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (the H3 encoder) and
//! `vae/minimax_h3_image_vae_fp16.safetensors`, as published. Edits (`image_edit`, #157's edit workflows): 1 to
//! `max_pictures` input images (default 5, 0: no edits), the first the one edited, each through the vision tower and the
//! VAE encoder as a reference; with a mask (the first image's size, alpha 0 where the edit goes) the decode is composited
//! over the first image, at 1 megapixel and 6 steps by default, without one the whole scene is redrawn at 2 megapixels
//! and 30 steps; an edit without a size takes the first image's shape at that area. One DiT forward a step (cfg 1, as ComfyUI's workflows run it): `guidance` and
//! `negative_prompt` are not used. On HIP and CUDA alike the DiT runs the bf16 attention (ROCM.md section 3).
//! Options: `pack` (default h3i-bf16), `max_pixels` (an image's area, default 1024 x 1024), `max_side` (default the API's
//! 2048), `default_size` (`WIDTHxHEIGHT`, default 1024x1024), `steps` (default 6, the PDD heads' own count),
//! `max_pictures` (default 5), `edit_pixels` (an edit's largest target and VAE copy, default 2176 x 1024), `resident_mb`
//! (a floor for the booking). The resident buffers are sized at load for `max_pixels`; an edit's are its own request's working
//! bytes, scaled by its pictures (ROCM.md section 4), so edits cost text to image nothing resident. Limits outside the API's, or whose largest
//! image does not fit the daemon's memory budget, stop the daemon at start (`validate`). Every byte figure here is an
//! *estimate* (ROCM.md section 4) until a gfx1151 run measures the peak, as `minimax_h3`'s `default_resident_mb` was.

const std = @import("std");
const engine = @import("../engine/engine.zig");
const png = @import("../media/png.zig");
const h3 = @import("minimax_h3");
const request = @import("../tool/request.zig");
const Request = request.Request;
const config = @import("../config.zig");
const ToolConfig = config.ToolConfig;

pub const entry: engine.Entry = .{ .name = "minimax_h3_image", .capabilities = &.{ .text_to_image, .image_edit }, .needs = needs, .create = create, .check = check, .defaults = defaults, .validate = validate, .license = license };

/// The weights' licence (MiniMax-H3-Image Community License Agreement, 2026-11-10), unless a tool's config names its own.
pub const license: config.License = .{
    .name = "MiniMax H3 Image Community License",
    .url = "https://huggingface.co/MiniMaxAI/MiniMax-H3-Image",
    .use = "only outside the excluded territories, under its Acceptable Use Policy; over USD 1 million yearly revenue, or any hosted service to third parties, needs MiniMax's prior written authorization",
    .commercial = true,
    .excluded_regions = &.{ "United States", "European Union", "United Kingdom", "South Korea" },
    .notice = "content published to the public must be disclosed as machine-generated; a commercial product shows \"MiniMax-H3-Image\" in its user interface",
};

const P = h3.pipeline_image;
const E = h3.edit;
const image_in = @import("../media/image_in.zig");
const api: request.Limits = .{};

/// The PDD checkpoint's own step count (its six heads, one a step: H3-IMAGE.md 1.4; #157's text to image).
const default_steps = 6;

/// The default `max_pixels`, 1024 x 1024: #157 measured text to image at 1024² (ROCM.md section 1), the size section 3's
/// attention estimate takes (about 1.3 k DiT rows), and section 4's resident estimate (about 58 GB of weights) is booked
/// beside the 96 GiB box's other models, so the per-image buffers stay small (*estimate*: the DiT's 0.6 GiB, a decode's
/// 10 MB). The model is tuned for 2048² of area; a config may raise it up to the API's largest side squared.
pub const default_max_pixels = 1024 * 1024;

/// The weights' device bytes (*estimates* from the checkpoints' headers): the bf16 pack (40.2 GB of bf16 checkpoint
/// with the block norms and adaLN linears in fp32), the 32B encoder's NVFP4 AWQ checkpoint and the VAE decoder.
pub const dit_bytes: u64 = 40_322_954_080;
pub const te_bytes: u64 = 15_700_000_000;
pub const vae_bytes: u64 = 807_781_040;
/// The runtime's context, the kernels' code objects and the allocator's slack (*estimate*).
const runtime_bytes: u64 = 1 << 30;
/// ffmpeg-free, but the PNG encoder's and the worker's own host buffers (*estimate*).
const host_slack: u64 = 64 << 20;

/// An integer option as a u32, `ceiling` where it is larger (`validate` refuses that at load).
fn option32(cfg: *const ToolConfig, key: []const u8, default: u32, ceiling: u32) u32 {
    return @intCast(@min(engine.optionInt(cfg, key, default), ceiling));
}

/// The API's shortest side on the 32-pixel grid.
const min_side = (api.min_side + 31) / 32 * 32;
/// The largest `max_pixels`: a square at the API's longest side.
const max_max_pixels = api.max_side * api.max_side;

fn maxSide(cfg: *const ToolConfig) u32 {
    return option32(cfg, "max_side", api.max_side, api.max_side) / 32 * 32;
}

fn limits(cfg: *const ToolConfig) P.Limits {
    return .{
        .pixels = option32(cfg, "max_pixels", default_max_pixels, max_max_pixels) / 1024 * 1024,
        .pictures = maxPictures(cfg),
        .edit_pixels = option32(cfg, "edit_pixels", P.default_edit_pixels, max_max_pixels) / 1024 * 1024,
    };
}

/// The input images an edit takes at most (`max_pictures`, at most `h3.edit.max_pictures`; 0: no edits).
fn maxPictures(cfg: *const ToolConfig) u32 {
    return option32(cfg, "max_pictures", E.max_pictures, E.max_pictures);
}

/// Whether an edit may be `width` x `height`: on the 32-pixel grid, within the sides and within the edit target.
fn editTakes(cfg: *const ToolConfig, width: u32, height: u32) bool {
    if (width % 32 != 0 or height % 32 != 0) return false;
    if (@min(width, height) < min_side or @max(width, height) > maxSide(cfg)) return false;
    const l = limits(cfg);
    return @as(u64, width) * height <= @max(l.pixels, l.edit_pixels);
}

fn defaultSize(cfg: *const ToolConfig) [2]u32 {
    if (request.parseSize(engine.optionString(cfg, "default_size", ""))) |d| if (takes(cfg, d[0], d[1])) return d;
    var side: u32 = @min(maxSide(cfg), 1024);
    while (side > min_side and !takes(cfg, side, side)) side -= 32;
    return .{ side, side };
}

/// Whether `width` x `height` is on the 32-pixel grid, within the sides, and within the buffers.
fn takes(cfg: *const ToolConfig, width: u32, height: u32) bool {
    if (width % 32 != 0 or height % 32 != 0) return false;
    if (@min(width, height) < min_side or @max(width, height) > maxSide(cfg)) return false;
    return P.fits(limits(cfg), width, height);
}

/// An image request without a size (0 x 0) gets `default_size`; an edit keeps 0 x 0 (the first image's shape).
fn defaults(cfg: *const ToolConfig, req: *Request) void {
    switch (req.*) {
        .image => |*v| if (v.width == 0 and v.height == 0 and v.references.len == 0) {
            const d = defaultSize(cfg);
            v.width = d[0];
            v.height = d[1];
        },
        .video, .asset3d, .speech, .transcription, .chat, .decision => {},
    }
}

/// Refuses what `generate` would, before the 57 GB load evicts anything: more input images than `max_pictures`, a mask
/// without one, a side off the 32-pixel grid or outside the sides, more pixels than `max_pixels` (an edit's size:
/// `edit_pixels`). The input images' own sizes are checked once they are read.
fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    const v = switch (req.*) {
        .image => |x| x,
        .video, .asset3d, .speech, .transcription, .chat, .decision => return true, // the kind check refuses it
    };
    if (v.references.len > maxPictures(cfg)) {
        if (maxPictures(cfg) == 0) {
            why.print("model '{s}' makes images from text only here (max_pictures is 0): send no input images or mask", .{cfg.id}) catch {};
        } else why.print("model '{s}' edits with 1 to {d} input images: {d} were sent", .{ cfg.id, maxPictures(cfg), v.references.len }) catch {};
        return false;
    }
    if (v.mask != null and v.references.len == 0) {
        why.print("a mask needs an input image to edit", .{}) catch {};
        return false;
    }
    if (v.references.len > 0) return checkEdit(cfg, v, why);
    if (v.width % 32 != 0 or v.height % 32 != 0 or v.width == 0 or v.height == 0) {
        why.print("model '{s}' takes sides that are multiples of 32: {d}x{d} is not (omit size for {d}x{d})", .{ cfg.id, v.width, v.height, defaultSize(cfg)[0], defaultSize(cfg)[1] }) catch {};
        return false;
    }
    if (@min(v.width, v.height) < min_side or @max(v.width, v.height) > maxSide(cfg)) {
        why.print("model '{s}' takes sides of {d} to {d} pixels", .{ cfg.id, min_side, maxSide(cfg) }) catch {};
        return false;
    }
    if (!P.fits(limits(cfg), v.width, v.height)) {
        why.print("model '{s}' takes at most {d} pixels (width x height), such as {d}x{d}", .{ cfg.id, limits(cfg).pixels, defaultSize(cfg)[0], defaultSize(cfg)[1] }) catch {};
        return false;
    }
    return true;
}

fn checkEdit(cfg: *const ToolConfig, v: request.Image, why: *std.Io.Writer) bool {
    if (v.width == 0 and v.height == 0) return true; // the first image's shape
    if (v.width % 32 != 0 or v.height % 32 != 0 or v.width == 0 or v.height == 0) {
        why.print("model '{s}' takes sides that are multiples of 32: {d}x{d} is not (omit size for the first image's shape)", .{ cfg.id, v.width, v.height }) catch {};
        return false;
    }
    if (@min(v.width, v.height) < min_side or @max(v.width, v.height) > maxSide(cfg)) {
        why.print("model '{s}' takes sides of {d} to {d} pixels", .{ cfg.id, min_side, maxSide(cfg) }) catch {};
        return false;
    }
    if (!editTakes(cfg, v.width, v.height)) {
        const l = limits(cfg);
        why.print("model '{s}' edits to at most {d} pixels (width x height)", .{ cfg.id, @max(l.pixels, l.edit_pixels) }) catch {};
        return false;
    }
    return true;
}

/// What the tool holds while loaded (*estimate*): the weights, the runtime and the buffers for its limits, or
/// `resident_mb` where that is more.
fn residentBytes(cfg: *const ToolConfig) u64 {
    const fixed = dit_bytes + te_bytes + vae_bytes + runtime_bytes;
    return @max(fixed +| P.scratchBytes(limits(cfg)), engine.optionInt(cfg, "resident_mb", 0) <<| 20);
}

/// A request's memory besides `residentBytes` (*estimate*): the pixels on the host and the PNG's copy, the decode's
/// scratch with the latent on the host (`P.requestBytes`), and the encoder's and worker's host slack.
fn workingBytes(width: u32, height: u32) u64 {
    const phase = P.requestBytes(width, height) catch return 0;
    return 2 * 3 * @as(u64, width) * height + phase + host_slack;
}

/// The largest edit a request may be before its images are read (*estimate*), scaled by its `n` pictures as
/// `reference_working_bytes` is: the edit's own DiT and text-encoder scratch and sampler buffers for `n` pictures
/// (`P.editScratchBytes`, none of it resident); `n` pictures of 2048 x 1024 (VAE copies at `edit_pixels`' default,
/// 2048 vision rows each) at the request's size or the default target; the device phase at its largest
/// (`P.editRequestBytes`), the host's copies (`P.editHostBytes`), each picture's RGBA at the API's largest side squared
/// (and the mask's), the output and the PNG's copy, the worker's slack.
fn editWorkingBytes(cfg: *const ToolConfig, n: usize, masked: bool, width: u32, height: u32) u64 {
    const k = @min(@max(n, 1), E.max_pictures);
    const sizes: [E.max_pictures]E.Size = @splat(.{ .w = 2048, .h = 1024 });
    const plan = E.Plan.of(sizes[0..k], masked, width, height) catch return 0;
    const device = P.editRequestBytes(&plan) catch return 0;
    const rgba = 4 * @as(u64, api.max_side) * api.max_side;
    return P.editScratchBytes(P.EditDims.most(limits(cfg), k)) + device + P.editHostBytes(&plan) + (k + @intFromBool(masked)) * rgba + 2 * 3 * @as(u64, plan.target.w) * plan.target.h + host_slack;
}

fn needs(cfg: *const ToolConfig, req: *const Request) engine.Needs {
    const working: u64 = switch (req.*) {
        .image => |v| if (v.references.len > 0) editWorkingBytes(cfg, v.references.len, v.mask != null, v.width, v.height) else workingBytes(v.width, v.height),
        .video, .asset3d, .speech, .transcription, .chat, .decision => 0,
    };
    return .{ .resident = residentBytes(cfg), .working = working };
}

/// The most a request the tool takes can book: its resident bytes and the largest working set `check` admits.
fn peakBytes(cfg: *const ToolConfig) u64 {
    const side = maxSide(cfg);
    var most: u64 = 0;
    var w: u32 = min_side;
    while (w <= side) : (w += 32) {
        var h: u32 = min_side;
        while (h <= side) : (h += 32) {
            if (takes(cfg, w, h)) most = @max(most, workingBytes(w, h));
        }
    }
    if (maxPictures(cfg) > 0) for ([_]bool{ false, true }) |masked| {
        most = @max(most, editWorkingBytes(cfg, maxPictures(cfg), masked, 0, 0));
    };
    return residentBytes(cfg) +| most;
}

/// Refuses, when the daemon starts, options outside what the API and the model take, and limits whose buffers and
/// largest image do not fit `budget` (0: unknown, not checked).
fn validate(cfg: *const ToolConfig, budget: u64, why: *std.Io.Writer) bool {
    const Range = struct { key: []const u8, min: u64, max: u64 };
    for ([_]Range{
        .{ .key = "max_pixels", .min = min_side * min_side, .max = max_max_pixels },
        .{ .key = "max_side", .min = min_side, .max = api.max_side },
        .{ .key = "steps", .min = 1, .max = api.max_steps },
        .{ .key = "max_pictures", .min = 0, .max = E.max_pictures },
        .{ .key = "edit_pixels", .min = min_side * min_side, .max = max_max_pixels },
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
    why.print("tool '{s}': its limits need {d} MiB ({d} MiB loaded, the rest for its largest image), more than the {d} MiB budget: lower max_pixels", .{ cfg.id, peak >> 20, residentBytes(cfg) >> 20, budget >> 20 }) catch {};
    return false;
}

/// Refuses the request (the API's 400) with a reason the caller can act on.
fn refuse(comptime fmt: []const u8, args: anytype) engine.Refused {
    engine.explain(fmt, args);
    return engine.Refused.Refused;
}

const MiniMaxH3Image = struct {
    env: engine.Env,
    cfg: *const ToolConfig,
    p: ?*P.Pipeline = null,
    arena: std.heap.ArenaAllocator,

    fn load(ptr: *anyopaque) anyerror!u64 {
        const t: *MiniMaxH3Image = @ptrCast(@alignCast(ptr));
        const a = t.arena.allocator();
        const root = t.cfg.weights;
        const paths: P.Paths = .{
            .pack = try std.fs.path.join(a, &.{ root, engine.optionString(t.cfg, "pack", "h3i-bf16") }),
            .te = try std.fs.path.join(a, &.{ root, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" }),
            .vae = try std.fs.path.join(a, &.{ root, "vae", "minimax_h3_image_vae_fp16.safetensors" }),
        };
        t.p = try P.Pipeline.create(t.env.gpa, t.env.io, paths, limits(t.cfg));
        return residentBytes(t.cfg);
    }

    fn generate(ptr: *anyopaque, job: *const engine.Job, sink: engine.Sink, arena: std.mem.Allocator) anyerror!engine.Output {
        const t: *MiniMaxH3Image = @ptrCast(@alignCast(ptr));
        const v = switch (job.request) {
            .image => |x| x,
            .video, .asset3d, .speech, .transcription, .chat, .decision => return engine.Refused.Refused,
        };
        const p = t.p orelse return error.NotLoaded;
        const start = std.Io.Clock.awake.now(t.env.io);
        if (v.references.len > maxPictures(t.cfg)) return refuse("this model edits with 1 to {d} input images", .{maxPictures(t.cfg)});
        if (v.mask != null and v.references.len == 0) return refuse("a mask needs an input image to edit", .{});
        const pictures = try arena.alloc(P.Picture, v.references.len);
        for (v.references, pictures) |name, *q| {
            const in = image_in.load(t.env.io, arena, job.dir, name) catch |err| return refuse("an input image could not be read ({s}): send a PNG, JPEG or WebP", .{@errorName(err)});
            q.* = .{ .rgba = in.rgba, .width = in.width, .height = in.height };
        }
        var mask: ?[]const u8 = null;
        if (v.mask) |name| {
            const m = image_in.load(t.env.io, arena, job.dir, name) catch |err| return refuse("the mask could not be read ({s}): send a PNG of the first image's size, transparent where the edit goes", .{@errorName(err)});
            if (m.width != pictures[0].width or m.height != pictures[0].height) return refuse("the mask is {d}x{d}, the first image {d}x{d}: send a mask of the first image's size", .{ m.width, m.height, pictures[0].width, pictures[0].height });
            mask = m.rgba;
        }
        const edit = pictures.len > 0;
        var width = v.width;
        var height = v.height;
        if (edit) {
            var loaded: [E.max_pictures]E.Size = undefined;
            for (pictures, loaded[0..pictures.len]) |q, *s| s.* = .{ .w = q.width, .h = q.height };
            const s = P.Pipeline.editSize(loaded[0..pictures.len], mask != null, width, height) catch return refuse("an input image is empty", .{});
            width = s.w;
            height = s.h;
        }
        const default: u32 = if (!edit) @intCast(engine.optionInt(t.cfg, "steps", default_steps)) else if (mask != null) E.masked_steps else E.unmasked_steps;
        const steps = if (v.steps == 0) default else v.steps;
        const pixels = try t.env.gpa.alloc(u8, @as(usize, width) * height * 3);
        defer t.env.gpa.free(pixels);
        const names = try arena.alloc([]const u8, v.n);
        var bridge: Bridge = .{ .sink = sink };
        for (names, 0..) |*name, i| {
            const seed = v.seed +% i;
            const made = if (edit) p.edit(v.prompt, pictures, mask, v.width, v.height, steps, seed, pixels, bridge.progress()) else p.generate(v.prompt, width, height, steps, seed, pixels, bridge.progress());
            _ = made catch |err| switch (err) {
                error.SizeNotMultipleOf32 => return refuse("size {d}x{d}: each side must be a multiple of 32", .{ width, height }),
                error.ImageTooLarge => return refuse("{d}x{d} is larger than this model takes: lower size", .{ width, height }),
                error.PromptTooLong => return refuse("the prompt (with the input images' tokens) is longer than the text encoder takes: shorten it", .{}),
                error.PictureTooLarge => return refuse("an input image is too long and thin to edit here: at most about 2:1 (edit_pixels)", .{}),
                error.PicturesTooLarge => return refuse("the input images together are larger than this model's vision budget: send fewer or squarer images", .{}),
                error.MaskKeepsEverything => return refuse("the mask has no transparent pixel: make it transparent where the edit goes", .{}),
                error.MaskResizeTooSmall => return refuse("{d}x{d} is too small to composite a mask onto: send a larger size", .{ width, height }),
                error.BadPicture, error.BadMask => return refuse("an input image or the mask could not be used: send 8-bit images, the mask of the first image's size", .{}),
                error.TooManyPictures, error.NoEdits => return refuse("this model edits with 1 to {d} input images", .{maxPictures(t.cfg)}),
                else => return err,
            };
            name.* = try std.fmt.allocPrint(arena, "{d}.png", .{i});
            var seed_buf: [24]u8 = undefined;
            const seed_s = try std.fmt.bufPrint(&seed_buf, "{d}", .{seed});
            var file = try std.Io.Dir.cwd().createFile(t.env.io, try std.fs.path.join(arena, &.{ job.dir, name.* }), .{});
            defer file.close(t.env.io);
            var buf: [64 * 1024]u8 = undefined;
            var fw = file.writer(t.env.io, &buf);
            try png.encode(&fw.interface, pixels, width, height, .rgb8, .{ .level = .fastest, .text = &.{ .{ .key = "prompt", .value = v.prompt }, .{ .key = "seed", .value = seed_s } }, .marker = .{ .model = t.cfg.id } });
            try fw.interface.flush();
        }
        const ms = start.durationTo(std.Io.Clock.awake.now(t.env.io)).toMilliseconds();
        return .{ .files = names, .seed = v.seed, .ms = @intCast(ms) };
    }

    fn unload(ptr: *anyopaque) void {
        const t: *MiniMaxH3Image = @ptrCast(@alignCast(ptr));
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
    const t = try env.gpa.create(MiniMaxH3Image);
    t.* = .{ .env = env, .cfg = cfg, .arena = .init(env.gpa) };
    return .{ .ptr = t, .vtable = &.{ .load = MiniMaxH3Image.load, .generate = MiniMaxH3Image.generate, .unload = MiniMaxH3Image.unload } };
}

test "sizes: multiples of 32 within the sides and max_pixels; refusals name the limit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "h3i", "kind": "image", "engine": "minimax_h3_image"},
        \\           {"id": "big", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pixels": 4194304, "default_size": "2048x2048"}},
        \\           {"id": "narrow", "kind": "image", "engine": "minimax_h3_image", "options": {"max_side": 768}}]}
    );
    const cfg = c.tool("h3i").?;
    const Case = struct { w: u32, h: u32, ok: bool };
    for ([_]Case{
        .{ .w = 1024, .h = 1024, .ok = true },
        .{ .w = 1344, .h = 768, .ok = true },
        .{ .w = 768, .h = 1344, .ok = true },
        .{ .w = 2048, .h = 512, .ok = true },
        .{ .w = 256, .h = 256, .ok = true },
        .{ .w = 1056, .h = 1024, .ok = false }, // past max_pixels
        .{ .w = 1000, .h = 1000, .ok = false }, // off the grid
        .{ .w = 4096, .h = 256, .ok = false }, // past the side
        .{ .w = 224, .h = 1024, .ok = false }, // under the API's shortest side
    }) |k| {
        var msg: [256]u8 = undefined;
        var why: std.Io.Writer = .fixed(&msg);
        const r: Request = .{ .image = .{ .prompt = "a cat", .width = k.w, .height = k.h } };
        std.testing.expectEqual(k.ok, check(cfg, &r, &why)) catch |err| {
            std.debug.print("{d}x{d}: {s}\n", .{ k.w, k.h, why.buffered() });
            return err;
        };
    }
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .width = 1056, .height = 1024 } }, &why));
    try std.testing.expectEqualStrings("model 'h3i' takes at most 1048576 pixels (width x height), such as 1024x1024", why.buffered());
    why = .fixed(&msg);
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .width = 1000, .height = 1024 } }, &why));
    try std.testing.expectEqualStrings("model 'h3i' takes sides that are multiples of 32: 1000x1024 is not (omit size for 1024x1024)", why.buffered());
    why = .fixed(&msg);
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .width = 4096, .height = 256 } }, &why));
    try std.testing.expectEqualStrings("model 'h3i' takes sides of 256 to 2048 pixels", why.buffered());
    try std.testing.expect(check(c.tool("big").?, &.{ .image = .{ .prompt = "a", .width = 2048, .height = 2048 } }, &why));

    var r: Request = .{ .image = .{ .prompt = "a", .width = 0, .height = 0 } };
    engine.fill(c.tool("big").?, &r);
    try std.testing.expectEqual([2]u32{ 2048, 2048 }, [2]u32{ r.image.width, r.image.height });
    r = .{ .image = .{ .prompt = "a", .width = 0, .height = 0 } };
    engine.fill(c.tool("narrow").?, &r);
    try std.testing.expectEqual([2]u32{ 768, 768 }, [2]u32{ r.image.width, r.image.height });
    why = .fixed(&msg);
    try std.testing.expect(check(c.tool("narrow").?, &r, &why));
}

test "needs: the weights and the buffers for max_pixels resident, the decode and the pixels working (estimates)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "h3i", "kind": "image", "engine": "minimax_h3_image"},
        \\           {"id": "big", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pixels": 4194304}},
        \\           {"id": "floor", "kind": "image", "engine": "minimax_h3_image", "options": {"resident_mb": 80000}},
        \\           {"id": "wide", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pixels": 8388608}},
        \\           {"id": "off", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pictures": 0}},
        \\           {"id": "steps", "kind": "image", "engine": "minimax_h3_image", "options": {"steps": 0}}]}
    );
    const one: Request = .{ .image = .{ .prompt = "a", .width = 1024, .height = 1024 } };
    const n = needs(c.tool("h3i").?, &one);
    // ROCM.md section 4: about 58 GB resident (weights and buffers) for text to image, a request's decode on top; an
    // edit's buffers are its own request's working bytes, so the resident set is the same with edits on or off
    const text_only = needs(c.tool("off").?, &one).resident;
    try std.testing.expect(text_only > 56_000_000_000 and text_only < 60_000_000_000);
    try std.testing.expectEqual(text_only, n.resident);
    try std.testing.expect(n.working > try P.requestBytes(1024, 1024));
    try std.testing.expect(n.working < 128 << 20);
    try std.testing.expect(needs(c.tool("big").?, &one).resident > n.resident); // the buffers follow max_pixels
    try std.testing.expectEqual(@as(u64, 80000) << 20, needs(c.tool("floor").?, &one).resident);
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    const budget: u64 = 90 << 30; // a 96 GiB GTT box less a reserve
    try std.testing.expect(validate(c.tool("h3i").?, budget, &why));
    try std.testing.expect(validate(c.tool("big").?, budget, &why));
    try std.testing.expect(!validate(c.tool("h3i").?, 40 << 30, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.buffered(), "more than the 40960 MiB budget") != null);
    why = .fixed(&msg);
    try std.testing.expect(!validate(c.tool("wide").?, 0, &why));
    try std.testing.expectEqualStrings("tool 'wide': options.max_pixels is 8388608; it takes 65536 to 4194304", why.buffered());
    why = .fixed(&msg);
    try std.testing.expect(!validate(c.tool("steps").?, 0, &why));
}

test "the licence is H3-Image's own: hosted services to third parties need MiniMax's authorization" {
    try std.testing.expectEqualStrings("MiniMax H3 Image Community License", license.name);
    try std.testing.expect(std.mem.indexOf(u8, license.use, "hosted service to third parties") != null);
    try std.testing.expectEqual(@as(usize, 4), license.excluded_regions.len);
    try std.testing.expectEqualSlices(engine.Capability, &.{ .text_to_image, .image_edit }, entry.capabilities);
}

test "edits: 1 to max_pictures images, a mask needs one, an edit's size is its own or within edit_pixels" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "h3i", "kind": "image", "engine": "minimax_h3_image"},
        \\           {"id": "two", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pictures": 2}},
        \\           {"id": "off", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pictures": 0}}]}
    );
    const cfg = c.tool("h3i").?;
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    const five = [_][]const u8{ "a.png", "b.png", "c.png", "d.png", "e.png" };
    try std.testing.expect(check(cfg, &.{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = &five } }, &why));
    try std.testing.expect(check(cfg, &.{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = five[0..1], .mask = "m.png" } }, &why));
    // a whole-scene edit's 2 megapixels are past max_pixels but within edit_pixels
    try std.testing.expect(check(cfg, &.{ .image = .{ .prompt = "a", .width = 1440, .height = 1440, .references = five[0..1] } }, &why));
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .width = 1440, .height = 1440 } }, &why));
    why = .fixed(&msg);
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .width = 2048, .height = 2048, .references = five[0..1] } }, &why));
    try std.testing.expectEqualStrings("model 'h3i' edits to at most 2228224 pixels (width x height)", why.buffered());
    why = .fixed(&msg);
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .width = 1000, .height = 1024, .references = five[0..1] } }, &why));
    try std.testing.expectEqualStrings("model 'h3i' takes sides that are multiples of 32: 1000x1024 is not (omit size for the first image's shape)", why.buffered());
    why = .fixed(&msg);
    try std.testing.expect(!check(c.tool("two").?, &.{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = five[0..3] } }, &why));
    try std.testing.expectEqualStrings("model 'two' edits with 1 to 2 input images: 3 were sent", why.buffered());
    why = .fixed(&msg);
    try std.testing.expect(!check(c.tool("off").?, &.{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = five[0..1] } }, &why));
    try std.testing.expectEqualStrings("model 'off' makes images from text only here (max_pictures is 0): send no input images or mask", why.buffered());
    why = .fixed(&msg);
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .mask = "m.png" } }, &why));
    try std.testing.expectEqualStrings("a mask needs an input image to edit", why.buffered());
    // an edit without a size keeps 0 x 0 for the first image's shape; text to image gets default_size
    var r: Request = .{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = five[0..1] } };
    engine.fill(cfg, &r);
    try std.testing.expectEqual([2]u32{ 0, 0 }, [2]u32{ r.image.width, r.image.height });
}

test "edit needs: the vision tower, the encoder, the references and the composite on top of a decode (estimates)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "h3i", "kind": "image", "engine": "minimax_h3_image"},
        \\           {"id": "off", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pictures": 0}},
        \\           {"id": "pics", "kind": "image", "engine": "minimax_h3_image", "options": {"max_pictures": 6}}]}
    );
    const cfg = c.tool("h3i").?;
    const t2i = needs(cfg, &.{ .image = .{ .prompt = "a", .width = 1024, .height = 1024 } });
    const one = needs(cfg, &.{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = &.{"a.png"}, .mask = "m.png" } });
    const five = needs(cfg, &.{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = &.{ "a", "b", "c", "d", "e" } } });
    const two = needs(cfg, &.{ .image = .{ .prompt = "a", .width = 0, .height = 0, .references = &.{ "a", "b" } } });
    try std.testing.expect(one.working > t2i.working);
    try std.testing.expect(two.working > one.working and five.working > two.working);
    // ROCM.md section 4: the edit's buffers (its DiT and text-encoder scratch, the references' and vision tokens'
    // rows) are working bytes scaled by the pictures the request sends, as reference_working_bytes; never resident
    const l = limits(cfg);
    try std.testing.expect(one.working > P.editScratchBytes(P.EditDims.most(l, 1)));
    try std.testing.expect(five.working > P.editScratchBytes(P.EditDims.most(l, 5)));
    try std.testing.expect(P.editScratchBytes(P.EditDims.most(l, 5)) > P.editScratchBytes(P.EditDims.most(l, 1)));
    try std.testing.expectEqual(t2i.resident, one.resident);
    try std.testing.expectEqual(t2i.resident, five.resident);
    // the vision tower and the VAE encoder load per request: at least their weights and buffers
    try std.testing.expect(one.working > h3.vae_video_enc.Encoder.deviceBytes(1024, 2048));
    try std.testing.expect(one.working > h3.te_vision.VisionEncoder.deviceBytes(128 * 64));
    try std.testing.expect(five.working < 16 << 30);
    // max_pictures 0 (edits off) loads the same resident set
    try std.testing.expectEqual(t2i.resident, needs(c.tool("off").?, &.{ .image = .{ .prompt = "a", .width = 1024, .height = 1024 } }).resident);
    var msg: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&msg);
    try std.testing.expect(validate(cfg, 90 << 30, &why));
    try std.testing.expect(!validate(c.tool("pics").?, 0, &why));
    try std.testing.expectEqualStrings("tool 'pics': options.max_pictures is 6; it takes 0 to 5", why.buffered());
}
