//! The `qwen_image` and `qwen_image_turbo` engines: Qwen-Image 2.1 and Qwen-Image 2.1 Turbo text to image on
//! LocalRouter's Zig pipeline (`qwen_image.pipeline`). Turbo is the same 7B DiT with its own weights and the 8-step
//! schedule its checkpoint stores (sample_sigmas, shift 1, CFG 1); it shares the text encoder and VAE packs.
//! `weights` is a directory of packs: the DiT's (`<precision>/` for Qwen-Image 2.1, `turbo-<precision>/` for Turbo, with
//! tfimage's Triton cubins in `triton/`), `te/` and `vae/` (`python -m stk_twin pack`). Options: `precision` (fp8s |
//! nvfp4, default fp8s), `dit` (the DiT pack's directory, default as above), `max_side` (pixels, the largest side a
//! request may ask for, default 1664), `steps` (default 25; Turbo 8, the only count its schedule has), `resident_mb`
//! (the scheduler's estimate). Neither runs classifier-free guidance: `guidance` and `negative_prompt` are not used.

const std = @import("std");
const engine = @import("../engine/engine.zig");
const png = @import("../media/png.zig");
const qi = @import("qwen_image");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

// image_edit is not ported yet, so the engines declare text to image only.
pub const entry: engine.Entry = .{ .name = "qwen_image", .capabilities = &.{.text_to_image}, .needs = needs, .create = create(.base), .check = check(.base) };
pub const turbo_entry: engine.Entry = .{ .name = "qwen_image_turbo", .capabilities = &.{.text_to_image}, .needs = needs, .create = create(.turbo), .check = check(.turbo) };

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

/// Refuses before loading what the engine would refuse after: a side past `max_side`, and for a checkpoint that stores
/// its schedule (Turbo) another step count.
fn check(comptime v: Variant) fn (cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    return struct {
        fn f(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
            const img = switch (req.*) {
                .image => |x| x,
                .video => return true, // the kind check refuses it
            };
            if (@max(img.width, img.height) > maxSide(cfg)) {
                why.print("model '{s}' takes sizes up to {d} pixels a side", .{ cfg.id, maxSide(cfg) }) catch {};
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
        \\ {"id": "own", "kind": "image", "engine": "qwen_image_turbo", "options": {"dit": "mine"}}
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
}

/// The scheduler's resident estimate in MiB: GB10's measured engine peaks at 1024x1024 (M5, MemAvailable) are 30 GiB
/// NVFP4 and 32 GiB FP8 (the 16.4 GB text encoder, the DiT, the VAE and the scratch sized at max_side), plus margin.
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
        .image => |v| @as(u64, v.width) * v.height * 4 * 2, // the pixels on the host and the PNG's copy
        .video => 0,
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
        t.p = try qi.pipeline.Pipeline.create(t.env.gpa, t.env.io, packs, .pack, maxSide(t.cfg));
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
            .video => return engine.Refused.Refused,
        };
        if (v.references.len > 0) return engine.Refused.Refused; // edits are M6
        const p = t.p orelse return error.NotLoaded;
        const start = std.Io.Clock.awake.now(t.env.io);
        const steps = if (v.steps == 0) t.variant.steps(t.cfg) else v.steps;
        const pixels = try t.env.gpa.alloc(u8, @as(usize, v.width) * v.height * p.vae.out_channels);
        defer t.env.gpa.free(pixels);
        const names = try arena.alloc([]const u8, v.n);
        var bridge: Bridge = .{ .sink = sink };
        for (names, 0..) |*name, i| {
            const seed = v.seed +% i;
            _ = p.generate(v.prompt, v.height, v.width, steps, seed, pixels, bridge.progress()) catch |err| switch (err) {
                error.ImageTooLarge, error.SizeNotMultipleOf16, error.PromptTooLong => return engine.Refused.Refused,
                else => return err,
            };
            name.* = try std.fmt.allocPrint(arena, "{d}.png", .{i});
            var seed_buf: [24]u8 = undefined;
            const seed_s = try std.fmt.bufPrint(&seed_buf, "{d}", .{seed});
            var file = try std.Io.Dir.cwd().createFile(t.env.io, try std.fs.path.join(arena, &.{ job.dir, name.* }), .{});
            defer file.close(t.env.io);
            var buf: [64 * 1024]u8 = undefined;
            var fw = file.writer(t.env.io, &buf);
            const fmt: png.Format = if (p.vae.out_channels == 4) .rgba8 else .rgb8;
            try png.encode(&fw.interface, pixels, v.width, v.height, fmt, .{ .level = .fastest, .text = &.{ .{ .key = "prompt", .value = v.prompt }, .{ .key = "seed", .value = seed_s } } });
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
