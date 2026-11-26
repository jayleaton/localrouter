//! Loaded GPU code: code objects (.hsaco) or offload bundles, their kernels by symbol name and their device globals.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

/// The most LDS a workgroup may use on RDNA3 / RDNA3.5 (gfx11): 64 KiB, with no per-kernel opt-in.
pub const max_lds: u32 = 65536;

pub const Module = struct {
    d: *const Driver,
    handle: abi.Module,

    /// A code object for this device's gfx target (8-byte aligned); the runtime copies what it needs.
    pub fn load(d: *const Driver, image: []const u8) Error!Module {
        if (image.len == 0) {
            std.log.err("empty GPU image: this binary was built without kernels (-Dhip)", .{});
            return error.Invalid;
        }
        if (@intFromPtr(image.ptr) % 8 != 0) return error.Invalid;
        var m: abi.Module = null;
        try d.check(d.api.cuModuleLoadData(&m, image.ptr), "hipModuleLoadData");
        return .{ .d = d, .handle = m };
    }

    pub fn unload(self: *Module) void {
        _ = self.d.api.cuModuleUnload(self.handle);
        self.* = undefined;
    }

    pub fn function(self: Module, name: [:0]const u8) Error!Function {
        var f: abi.Function = null;
        self.d.check(self.d.api.cuModuleGetFunction(&f, self.handle, name.ptr), "hipModuleGetFunction") catch |e| {
            std.log.err("kernel symbol not in module: {s}", .{name});
            return e;
        };
        return .{ .d = self.d, .handle = f };
    }

    pub const Global = struct { ptr: abi.DevicePtr, len: usize };

    pub fn global(self: Module, name: [:0]const u8) Error!Global {
        var g: Global = .{ .ptr = 0, .len = 0 };
        try self.d.check(self.d.api.cuModuleGetGlobal_v2(&g.ptr, &g.len, self.handle, name.ptr), "hipModuleGetGlobal");
        return g;
    }
};

pub const Function = struct {
    d: *const Driver,
    handle: abi.Function,

    pub fn attribute(self: Function, a: abi.FunctionAttribute) Error!c_int {
        var v: c_int = 0;
        try self.d.check(self.d.api.cuFuncGetAttribute(&v, a, self.handle), "hipFuncGetAttribute");
        return v;
    }

    /// CUDA needs an opt-in above 48 KiB; gfx11 gives any kernel up to `max_lds`, so this only refuses more.
    pub fn allowDynamicShared(_: Function, bytes: u32) Error!void {
        if (bytes > max_lds) return error.Invalid;
    }
};

test "dynamic LDS: up to 64 KiB needs no opt-in, more is refused" {
    const f: Function = .{ .d = undefined, .handle = null };
    try f.allowDynamicShared(49152);
    try f.allowDynamicShared(max_lds);
    try std.testing.expectError(error.Invalid, f.allowDynamicShared(max_lds + 1));
}
