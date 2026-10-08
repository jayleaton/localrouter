//! The `qwen_image` engine: Qwen-Image 2.1 text to image on LocalRouter's Zig pipeline (`qwen_image.pipeline`).
//! `weights` is a directory of packs: `<precision>/` (the DiT's, with tfimage's Triton cubins in `triton/`), `te/` and
//! `vae/` (`python -m stk_twin pack`). Options: `precision` (fp8s | nvfp4, default fp8s), `max_side` (pixels, the
//! largest side a request may ask for, default 1664), `steps` (default 25), `resident_mb` (the scheduler's estimate).

const std = @import("std");
const engine = @import("../engine/engine.zig");
const png = @import("../media/png.zig");
const qi = @import("qwen_image");
const Request = @import("../tool/request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

// image_edit is not ported yet, so the engine declares text to image only.
pub const entry: engine.Entry = .{ .name = "qwen_image", .capabilities = &.{.text_to_image}, .needs = needs, .create = create };

const default_steps = 25;
const default_precision = "fp8s"; // closer to the original model; "nvfp4" is the faster option

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
    p: ?*qi.pipeline.Pipeline = null,
    arena: std.heap.ArenaAllocator,

    fn load(ptr: *anyopaque) anyerror!u64 {
        const t: *QwenImage = @ptrCast(@alignCast(ptr));
        const a = t.arena.allocator();
        const root = t.cfg.weights;
        const precision = engine.optionString(t.cfg, "precision", default_precision);
        const packs: qi.pipeline.Packs = .{
            .dit = try std.fs.path.join(a, &.{ root, precision }),
            .te = try std.fs.path.join(a, &.{ root, "te" }),
            .vae = try std.fs.path.join(a, &.{ root, "vae" }),
        };
        const side: u32 = @intCast(engine.optionInt(t.cfg, "max_side", 1664));
        t.p = try qi.pipeline.Pipeline.create(t.env.gpa, t.env.io, packs, .pack, side / 16 * 16);
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
        const steps = if (v.steps == 0) @as(u32, @intCast(engine.optionInt(t.cfg, "steps", default_steps))) else v.steps;
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

fn create(env: engine.Env, cfg: *const ToolConfig) anyerror!engine.Engine {
    const t = try env.gpa.create(QwenImage);
    t.* = .{ .env = env, .cfg = cfg, .arena = .init(env.gpa) };
    return .{ .ptr = t, .vtable = &.{ .load = QwenImage.load, .generate = QwenImage.generate, .unload = QwenImage.unload } };
}
