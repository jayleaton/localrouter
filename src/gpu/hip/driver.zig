//! The HIP runtime opened at run time: libamdhip64's entry points in one table (abi.Api), and checked calls.

const std = @import("std");
const abi = @import("abi.zig");

pub const Error = error{ OutOfMemory, DriverUnavailable, MissingSymbol, CudaFailed, CudaContextLost, GraphNotPermitted, OutOfDeviceMemory, NotReady, NotFound, Invalid };

/// The runtime library: ROCm 7's soname, else `LOCALROUTER_HIP_LIB` (a path) when set.
pub const default_path = "libamdhip64.so.7";

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Driver {
        const env = std.c.getenv("LOCALROUTER_HIP_LIB");
        return openPath(if (env) |p| std.mem.span(p) else default_path);
    }

    /// Resolves every field of `abi.Api` by its HIP symbol; a missing symbol refuses the whole runtime.
    pub fn openPath(path: []const u8) Error!Driver {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        const d: Driver = .{ .lib = lib, .api = try resolve(&lib, path) };
        try d.check(d.api.cuInit(0), "hipInit");
        return d;
    }

    /// The entry table from an opened library, without initialising the runtime (no GPU needed).
    pub fn resolve(lib: *std.DynLib, path: []const u8) Error!abi.Api {
        var api: abi.Api = undefined;
        const info = @typeInfo(abi.Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, abi.symbol(name)) orelse {
                std.log.err("{s} has no {s}", .{ path, abi.symbol(name) });
                return error.MissingSymbol;
            };
        }
        return api;
    }

    pub fn close(self: *Driver) void {
        self.lib.close();
    }

    pub fn synchronize(self: *const Driver) Error!void {
        try self.check(self.api.cuCtxSynchronize(), "hipDeviceSynchronize");
    }

    /// Logs a failed call with the runtime's name and text for the code; NOT_READY is a state, not a failure.
    pub fn check(self: *const Driver, res: abi.Result, what: []const u8) Error!void {
        if (res == abi.success) return;
        if (res == abi.error_not_ready) return error.NotReady;
        std.log.err("{s}: {s} ({d}) {s}", .{ what, self.errorName(res), res, self.errorText(res) });
        return errorOf(res);
    }

    pub fn result(_: *const Driver, res: abi.Result) Error!void {
        if (res == abi.success) return;
        return errorOf(res);
    }

    pub fn errorOf(res: abi.Result) Error {
        return switch (res) {
            abi.error_out_of_memory => error.OutOfDeviceMemory,
            abi.error_not_found => error.NotFound,
            abi.error_not_ready => error.NotReady,
            700, 719 => error.CudaContextLost, // illegal address, launch failure: the context is unusable
            else => error.CudaFailed,
        };
    }

    pub fn errorName(self: *const Driver, res: abi.Result) []const u8 {
        return if (self.api.cuGetErrorName(res)) |p| std.mem.span(p) else "hipErrorUnknown";
    }

    pub fn errorText(self: *const Driver, res: abi.Result) []const u8 {
        return if (self.api.cuGetErrorString(res)) |p| std.mem.span(p) else "";
    }

    /// HIP's runtime does not name a kernel from its handle.
    pub fn functionName(_: *const Driver, _: abi.Function) []const u8 {
        return "?";
    }

    /// The runtime's version (HIP_VERSION: 10000000 * major + 100000 * minor + patch).
    pub fn version(self: *const Driver) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.cuDriverGetVersion(&v), "hipDriverGetVersion");
        return v;
    }

    pub fn deviceCount(self: *const Driver) Error!c_int {
        var n: c_int = 0;
        try self.check(self.api.cuDeviceGetCount(&n), "hipGetDeviceCount");
        return n;
    }

    /// The gfx target's major and minor as 10 * major + minor (gfx1151: 115), read without making a context.
    pub fn capability(self: *const Driver, ordinal: c_int) Error!u32 {
        var major: c_int = 0;
        var minor: c_int = 0;
        try self.check(self.api.cuDeviceGetAttribute(&major, .compute_capability_major, ordinal), "hipDeviceGetAttribute");
        try self.check(self.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, ordinal), "hipDeviceGetAttribute");
        return @intCast(10 * major + minor);
    }

    pub fn deviceName(self: *const Driver, ordinal: c_int, buf: []u8) Error![]const u8 {
        if (buf.len < 2) return error.Invalid;
        var dev: abi.Device = 0;
        try self.check(self.api.cuDeviceGet(&dev, ordinal), "hipDeviceGet");
        try self.check(self.api.cuDeviceGetName(buf.ptr, @intCast(buf.len), dev), "hipDeviceGetName");
        return std.mem.sliceTo(buf, 0);
    }
};

test "a missing runtime is DriverUnavailable, and codes map as CUDA's do" {
    try std.testing.expectError(error.DriverUnavailable, Driver.openPath("/nonexistent/libamdhip64.so.7"));
    try std.testing.expectEqual(error.OutOfDeviceMemory, Driver.errorOf(2));
    try std.testing.expectEqual(error.NotFound, Driver.errorOf(500));
    try std.testing.expectEqual(error.CudaContextLost, Driver.errorOf(700));
    try std.testing.expectEqual(error.CudaFailed, Driver.errorOf(1));
}
