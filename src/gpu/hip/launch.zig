//! Kernel launches: argument values packed where hipModuleLaunchKernel reads them (the same kernelParams array as
//! cuLaunchKernel, so the engines' argument lists are unchanged).

const std = @import("std");
const abi = @import("abi.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const Error = @import("driver.zig").Error;

pub const Dim3 = abi.Dim3;

/// Argument values in kernel order; each must have exactly the C type of its kernel parameter.
pub const Args = struct {
    pub const max_args = 48;
    pub const max_bytes = 2048;

    storage: [max_bytes]u8 align(16) = undefined,
    offsets: [max_args]u16 = undefined,
    sizes: [max_args]u16 = undefined,
    ptrs: [max_args]?*anyopaque = undefined,
    used: usize = 0,
    count: usize = 0,

    /// Copies `value` (a pointer-sized address, scalar, or extern struct passed by value) at its C alignment.
    pub fn add(self: *Args, value: anytype) void {
        const T = @TypeOf(value);
        comptime std.debug.assert(@sizeOf(T) > 0);
        const at = std.mem.alignForward(usize, self.used, @alignOf(T));
        std.debug.assert(self.count < max_args and at + @sizeOf(T) <= max_bytes);
        @memcpy(self.storage[at..][0..@sizeOf(T)], std.mem.asBytes(&value));
        self.offsets[self.count] = @intCast(at);
        self.sizes[self.count] = @sizeOf(T);
        self.count += 1;
        self.used = at + @sizeOf(T);
    }

    /// The address array, rebuilt here so a moved Args never hands out stale addresses.
    pub fn pointers(self: *Args) ?[*]?*anyopaque {
        if (self.count == 0) return null;
        for (self.offsets[0..self.count], 0..) |off, i| self.ptrs[i] = &self.storage[off];
        return &self.ptrs;
    }
};

/// A plain launch: the engines use no clusters, programmatic dependent launch or cooperative launches.
pub const Config = struct {
    grid: Dim3,
    block: Dim3,
    shared: u32 = 0,

    pub fn validate(self: Config) Error!void {
        const g = self.grid;
        const b = self.block;
        if (g.x == 0 or g.y == 0 or g.z == 0 or b.x == 0 or b.y == 0 or b.z == 0) return error.Invalid;
        if (@as(u64, b.x) * b.y * b.z > 1024) return error.Invalid;
    }
};

pub fn launch(f: Function, cfg: Config, stream: Stream, args: *Args) Error!void {
    try cfg.validate();
    const g = cfg.grid;
    const b = cfg.block;
    const res = f.d.api.cuLaunchKernel(f.handle, g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared, stream.handle, args.pointers(), null);
    if (res == abi.success) return;
    var buf: [160]u8 = undefined;
    std.log.err("hipModuleLaunchKernel: {s} ({d}); {s}", .{ f.d.errorName(res), res, describe(&buf, cfg) });
    return @import("driver.zig").Driver.errorOf(res);
}

/// "grid XxYxZ, block XxYxZ, shared N B".
pub fn describe(buf: []u8, cfg: Config) []const u8 {
    const g = cfg.grid;
    const b = cfg.block;
    var w = std.Io.Writer.fixed(buf);
    w.print("grid {d}x{d}x{d}, block {d}x{d}x{d}, shared {d} B", .{ g.x, g.y, g.z, b.x, b.y, b.z, cfg.shared }) catch {};
    return w.buffered();
}

test "args keep C alignment and sizes; a bad geometry is refused before the runtime" {
    var a: Args = .{};
    a.add(@as(u64, 0x1122334455667788));
    a.add(@as(i32, -3));
    a.add(@as(f32, 2.5));
    a.add(@as(i64, 9));
    try std.testing.expectEqualSlices(u16, &.{ 0, 8, 12, 16 }, a.offsets[0..a.count]);
    const ptrs = a.pointers().?;
    try std.testing.expectEqual(@as(i32, -3), @as(*align(1) const i32, @ptrCast(ptrs[1].?)).*);
    const bad: Config = .{ .grid = .{ .x = 1 }, .block = .{ .x = 32, .y = 64 } };
    try std.testing.expectError(error.Invalid, bad.validate());
}
