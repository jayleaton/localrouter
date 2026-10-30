//! The video VAE decoder on the GPU: the component (`video_vae/`) issues launches through a caller `Launcher` and owns
//! no CUDA state, so this file is its host: the two modules (conv.cu, ops.cu), the function table and the adapter onto
//! the engine's stream. Its weights come from the component's own loader (`video_vae/weights.zig`).

const std = @import("std");
const cuda = @import("cuda");
const launch = @import("video_vae/launch.zig");
const tiles = @import("video_vae/tiles.zig");

pub const Gpu = struct {
    conv: cuda.Module,
    ops: cuda.Module,
    fns: [std.enums.values(launch.Kernel).len]cuda.Function,
    stream: cuda.Stream,

    pub fn load(d: *const cuda.Driver, conv_image: []const u8, ops_image: []const u8, s: cuda.Stream) !Gpu {
        if (conv_image.len == 0 or ops_image.len == 0) return error.KernelsNotBuilt;
        var g: Gpu = .{ .conv = try cuda.Module.load(d, conv_image), .ops = undefined, .fns = undefined, .stream = s };
        errdefer g.conv.unload();
        g.ops = try cuda.Module.load(d, ops_image);
        errdefer g.ops.unload();
        for (std.enums.values(launch.Kernel)) |k| {
            const m = if (launch.image(k) == .conv) g.conv else g.ops;
            g.fns[@backingInt(k)] = try m.function(launch.symbol(k));
        }
        return g;
    }

    pub fn unload(g: *Gpu) void {
        g.ops.unload();
        g.conv.unload();
    }

    pub fn launcher(g: *Gpu) launch.Launcher {
        return .{ .ctx = g, .func = run };
    }

    fn run(ctx: *anyopaque, l: launch.Launch) anyerror!void {
        const g: *Gpu = @ptrCast(@alignCast(ctx));
        var a: cuda.Args = .{};
        switch (l.call) {
            inline else => |p| a.add(p), // one by-value struct parameter
        }
        try cuda.launch.launch(g.fns[@backingInt(l.call)], .{
            .grid = .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] },
            .block = .{ .x = l.block[0], .y = l.block[1], .z = l.block[2] },
        }, g.stream, &a);
    }
};

/// The decode plan at the native geometry under `free_bytes` (the tool's exact `vae_free_bytes`, or `vae_free_mb`). Its
/// `emitted` geometry is the decoded frame size, which can be narrower than the latent's nominal 8x when the plan's tiles
/// stop before the edge.
pub fn plan(latent_frames: u32, lh: u32, lw: u32, free_bytes: u64) !tiles.DecodePlan {
    return tiles.decodePlan(latent_frames, lh, lw, .{ .free_bytes = free_bytes });
}
