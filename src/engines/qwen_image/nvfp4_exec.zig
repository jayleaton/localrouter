//! Runs `nvfp4_plan` launches: the four TensorFold fatbins (act, gemm_ck, gemm_ws, lane4) loaded once, each launch's
//! argument roles bound to device pointers, shared memory opted into once per kernel.

const std = @import("std");
const cuda = @import("cuda");
const plan = @import("nvfp4_plan.zig");
const kernels = @import("nvfp4_kernels");

/// The device pointers a launch's roles name (0 where unused).
pub const Bind = struct {
    x: u64 = 0,
    xs: u64 = 0,
    w: u64 = 0,
    ws: u64 = 0,
    out: u64 = 0,
    codes: u64 = 0,
    scales: u64 = 0,
    src: u64 = 0,
    dst: u64 = 0,
    up_w: u64 = 0, // gemm_gu_ck: the up half's weight and block scales
    up_ws: u64 = 0,
};

const Entry = struct { symbol: []const u8, f: cuda.Function, smem_set: u32 };

pub const Nvfp4 = struct {
    gpa: std.mem.Allocator,
    mods: [4]cuda.Module,
    cache: std.ArrayList(Entry) = .empty,
    major: u32,
    minor: u32,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, major: u32, minor: u32) !Nvfp4 {
        var mods: [4]cuda.Module = undefined;
        const images = [_][]const u8{ kernels.act, kernels.gemm_ck, kernels.gemm_ws, kernels.lane4 };
        var n: usize = 0;
        errdefer for (mods[0..n]) |*m| m.unload();
        for (images, 0..) |img, i| {
            mods[i] = try cuda.Module.load(d, img);
            n += 1;
        }
        return .{ .gpa = gpa, .mods = mods, .major = major, .minor = minor };
    }

    pub fn unload(e: *Nvfp4) void {
        for (&e.mods) |*m| m.unload();
        e.cache.deinit(e.gpa);
    }

    /// A kernel by symbol, its dynamic shared memory opted into up to `smem`; a short list (a dozen kernels).
    fn function(e: *Nvfp4, l: *const plan.Launch) !cuda.Function {
        for (e.cache.items) |*c| if (std.mem.eql(u8, c.symbol, l.symbol)) {
            if (l.needsSmemOptIn() and c.smem_set < l.smem) {
                try c.f.allowDynamicShared(l.smem);
                c.smem_set = l.smem;
            }
            return c.f;
        };
        const f = try e.mods[@intFromEnum(l.module)].function(l.symbol);
        if (l.needsSmemOptIn()) try f.allowDynamicShared(l.smem);
        try e.cache.append(e.gpa, .{ .symbol = l.symbol, .f = f, .smem_set = l.smem });
        return f;
    }

    pub fn run(e: *Nvfp4, l: *const plan.Launch, b: Bind, s: cuda.Stream) !void {
        var args: cuda.launch.Args = .{};
        for (l.slice()) |a| switch (a) {
            .x => args.add(b.x),
            .xs => args.add(b.xs),
            .w => args.add(b.w),
            .ws => args.add(b.ws),
            .out => args.add(b.out),
            .codes => args.add(b.codes),
            .scales => args.add(b.scales),
            .src => args.add(b.src),
            .dst => args.add(b.dst),
            .null_ptr => args.add(@as(u64, 0)),
            .up_zero => args.add(std.mem.zeroes(plan.Up)),
            .up => |u| args.add(plan.Up{ .w = b.up_w, .ws = b.up_ws, .alpha = u.alpha, .codes = b.codes, .scales = b.scales, .qg = u.qg, .kd = u.kd }),
            .i32 => |v| args.add(v),
            .f32 => |v| args.add(v),
        };
        const f = try e.function(l);
        const g = l.grid;
        const bl = l.block;
        try cuda.launch.launch(f, .{ .grid = .{ .x = g.x, .y = g.y, .z = g.z }, .block = .{ .x = bl.x, .y = bl.y, .z = bl.z }, .shared = l.smem }, s, &args);
    }
};
