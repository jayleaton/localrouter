//! The `minimax_h3_image` engine: MiniMax H3-Image text to image on LocalRouter's Zig pipeline
//! (`minimax_h3.pipeline_image`), one PNG an image (docs/dev/H3-IMAGE.md). `weights` is a directory laid out like ComfyUI's
//! models folder plus the pack: `<pack>/` (`python -m stk_twin.h3i.build`: the DiT in bf16 and `tokenizer.json`),
//! `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (the H3 encoder) and
//! `vae/minimax_h3_image_vae_fp16.safetensors`, as published. Edits (`image_edit`) are PR 1.7's: a request with input
//! images or a mask is refused. One DiT forward a step (cfg 1, as ComfyUI's workflows run it): `guidance` and
//! `negative_prompt` are not used. On HIP and CUDA alike the DiT runs the bf16 attention (ROCM.md section 3).
//! Options: `pack` (default h3i-bf16), `max_pixels` (an image's area, default 1024 x 1024), `max_side` (default the API's
//! 2048), `default_size` (`WIDTHxHEIGHT`, default 1024x1024), `steps` (default 6, the PDD heads' own count), `resident_mb`
//! (a floor for the booking). The buffers are sized at load for `max_pixels`; limits outside the API's, or whose largest
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

pub const entry: engine.Entry = .{ .name = "minimax_h3_image", .capabilities = &.{.text_to_image}, .needs = needs, .create = create, .check = check, .defaults = defaults, .validate = validate, .license = license };

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
    return .{ .pixels = option32(cfg, "max_pixels", default_max_pixels, max_max_pixels) / 1024 * 1024 };
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

/// An image request without a size (0 x 0) gets `default_size`.
fn defaults(cfg: *const ToolConfig, req: *Request) void {
    switch (req.*) {
        .image => |*v| if (v.width == 0 and v.height == 0) {
            const d = defaultSize(cfg);
            v.width = d[0];
            v.height = d[1];
        },
        .video, .asset3d, .speech, .transcription, .chat, .decision => {},
    }
}

/// Refuses what `generate` would, before the 57 GB load evicts anything: input images or a mask (edits are not
/// available yet), a side off the 32-pixel grid or outside the sides, more pixels than `max_pixels`.
fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    const v = switch (req.*) {
        .image => |x| x,
        .video, .asset3d, .speech, .transcription, .chat, .decision => return true, // the kind check refuses it
    };
    if (v.references.len > 0 or v.mask != null) {
        why.print("model '{s}' makes images from text only: send no input images or mask", .{cfg.id}) catch {};
        return false;
    }
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

fn needs(cfg: *const ToolConfig, req: *const Request) engine.Needs {
    const working: u64 = switch (req.*) {
        .image => |v| workingBytes(v.width, v.height),
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
        if (v.references.len > 0 or v.mask != null) return refuse("this model makes images from text only: send no input images or mask", .{});
        const p = t.p orelse return error.NotLoaded;
        const start = std.Io.Clock.awake.now(t.env.io);
        const steps = if (v.steps == 0) @as(u32, @intCast(engine.optionInt(t.cfg, "steps", default_steps))) else v.steps;
        const pixels = try t.env.gpa.alloc(u8, @as(usize, v.width) * v.height * 3);
        defer t.env.gpa.free(pixels);
        const names = try arena.alloc([]const u8, v.n);
        var bridge: Bridge = .{ .sink = sink };
        for (names, 0..) |*name, i| {
            const seed = v.seed +% i;
            _ = p.generate(v.prompt, v.width, v.height, steps, seed, pixels, bridge.progress()) catch |err| switch (err) {
                error.SizeNotMultipleOf32 => return refuse("size {d}x{d}: each side must be a multiple of 32", .{ v.width, v.height }),
                error.ImageTooLarge => return refuse("{d}x{d} is larger than this model takes: lower size", .{ v.width, v.height }),
                error.PromptTooLong => return refuse("the prompt is longer than the text encoder takes: shorten it", .{}),
                else => return err,
            };
            name.* = try std.fmt.allocPrint(arena, "{d}.png", .{i});
            var seed_buf: [24]u8 = undefined;
            const seed_s = try std.fmt.bufPrint(&seed_buf, "{d}", .{seed});
            var file = try std.Io.Dir.cwd().createFile(t.env.io, try std.fs.path.join(arena, &.{ job.dir, name.* }), .{});
            defer file.close(t.env.io);
            var buf: [64 * 1024]u8 = undefined;
            var fw = file.writer(t.env.io, &buf);
            try png.encode(&fw.interface, pixels, v.width, v.height, .rgb8, .{ .level = .fastest, .text = &.{ .{ .key = "prompt", .value = v.prompt }, .{ .key = "seed", .value = seed_s } }, .marker = .{ .model = t.cfg.id } });
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

test "sizes: multiples of 32 within the sides and max_pixels; refusals name the limit; edits are refused" {
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
    why = .fixed(&msg);
    try std.testing.expect(!check(cfg, &.{ .image = .{ .prompt = "a", .references = &.{"in.png"} } }, &why));
    try std.testing.expectEqualStrings("model 'h3i' makes images from text only: send no input images or mask", why.buffered());
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
        \\           {"id": "steps", "kind": "image", "engine": "minimax_h3_image", "options": {"steps": 0}}]}
    );
    const one: Request = .{ .image = .{ .prompt = "a", .width = 1024, .height = 1024 } };
    const n = needs(c.tool("h3i").?, &one);
    // ROCM.md section 4: about 58 GB resident (weights and buffers), a request's decode on top
    try std.testing.expect(n.resident > 56_000_000_000 and n.resident < 60_000_000_000);
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
    try std.testing.expectEqualSlices(engine.Capability, &.{.text_to_image}, entry.capabilities);
}
