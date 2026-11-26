//! Device memory (hipMalloc: on an APU, GTT-backed system memory) and page-locked host memory, with checked copies.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const DeviceBuffer = struct {
    d: *const Driver,
    ptr: abi.DevicePtr,
    len: usize,

    /// `len` bytes (hipMalloc aligns to at least 256); zero bytes allocate nothing and hold address 0.
    pub fn alloc(d: *const Driver, len: usize) Error!DeviceBuffer {
        var p: abi.DevicePtr = 0;
        if (len > 0) try d.check(d.api.cuMemAlloc_v2(&p, len), "hipMalloc");
        return .{ .d = d, .ptr = p, .len = len };
    }

    pub fn fromHost(d: *const Driver, bytes: []const u8) Error!DeviceBuffer {
        var b = try alloc(d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        return b;
    }

    pub fn free(self: *DeviceBuffer) void {
        if (self.ptr != 0) _ = self.d.api.cuMemFree_v2(self.ptr);
        self.* = undefined;
    }

    pub fn at(self: DeviceBuffer, offset: usize) Error!abi.DevicePtr {
        if (offset > self.len) return error.Invalid;
        return self.ptr + offset;
    }

    fn span(self: DeviceBuffer, offset: usize, n: usize) Error!abi.DevicePtr {
        if (offset > self.len or n > self.len - offset) return error.Invalid;
        return self.ptr + offset;
    }

    /// Returns once the bytes have landed.
    pub fn upload(self: DeviceBuffer, offset: usize, bytes: []const u8) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "hipMemcpyHtoD");
    }

    pub fn download(self: DeviceBuffer, offset: usize, out: []u8) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyDtoH_v2(out.ptr, src, out.len), "hipMemcpyDtoH");
    }

    /// `bytes` must stay valid and unchanged until the stream reaches the copy (pinned memory for a true async copy).
    pub fn uploadAsync(self: DeviceBuffer, offset: usize, bytes: []const u8, stream: abi.Stream) Error!void {
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, stream), "hipMemcpyHtoDAsync");
    }

    pub fn downloadAsync(self: DeviceBuffer, offset: usize, out: []u8, stream: abi.Stream) Error!void {
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.api.cuMemcpyDtoHAsync_v2(out.ptr, src, out.len, stream), "hipMemcpyDtoHAsync");
    }

    pub fn copyFrom(self: DeviceBuffer, offset: usize, src: abi.DevicePtr, n: usize, stream: ?abi.Stream) Error!void {
        const dst = try self.span(offset, n);
        if (n == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, s), "hipMemcpyDtoDAsync");
        } else {
            try self.d.check(self.d.api.cuMemcpyDtoD_v2(dst, src, n), "hipMemcpyDtoD");
        }
    }

    pub fn fill8(self: DeviceBuffer, value: u8, stream: ?abi.Stream) Error!void {
        if (self.len == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.api.cuMemsetD8Async(self.ptr, value, self.len, s), "hipMemsetD8Async");
        } else {
            try self.d.check(self.d.api.cuMemsetD8_v2(self.ptr, value, self.len), "hipMemsetD8");
        }
    }

    /// Fills whole 32-bit words; the length must be a multiple of four.
    pub fn fill32(self: DeviceBuffer, value: u32, stream: ?abi.Stream) Error!void {
        if (self.len % 4 != 0) return error.Invalid;
        if (self.len == 0) return;
        const v: c_int = @bitCast(value);
        if (stream) |s| {
            try self.d.check(self.d.api.cuMemsetD32Async(self.ptr, v, self.len / 4, s), "hipMemsetD32Async");
        } else {
            try self.d.check(self.d.api.cuMemsetD32_v2(self.ptr, v, self.len / 4), "hipMemsetD32");
        }
    }
};

/// Page-locked host memory: what an asynchronous copy reads or writes without staging.
pub const HostBuffer = struct {
    d: *const Driver,
    bytes: []align(16) u8,

    pub fn alloc(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_alloc_portable);
    }

    /// Pinned memory kernels write directly (mapped into the device's address space).
    pub fn allocMapped(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_alloc_portable | abi.host_alloc_devicemap);
    }

    pub fn device(self: HostBuffer) Error!abi.DevicePtr {
        var p: abi.DevicePtr = 0;
        try self.d.check(self.d.api.cuMemHostGetDevicePointer_v2(&p, self.bytes.ptr, 0), "hipHostGetDevicePointer");
        return p;
    }

    fn allocFlags(d: *const Driver, len: usize, flags: c_uint) Error!HostBuffer {
        if (len == 0) return error.Invalid;
        var p: ?*anyopaque = null;
        try d.check(d.api.cuMemHostAlloc(&p, len, flags), "hipHostMalloc");
        const base: [*]align(16) u8 = @ptrCast(@alignCast(p.?));
        return .{ .d = d, .bytes = base[0..len] };
    }

    pub fn free(self: *HostBuffer) void {
        _ = self.d.api.cuMemFreeHost(self.bytes.ptr);
        self.* = undefined;
    }

    pub fn slice(self: HostBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.bytes[0 .. self.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
};

test "a span past the buffer is refused before any runtime call" {
    const b: DeviceBuffer = .{ .d = undefined, .ptr = 0x1000, .len = 64 };
    try std.testing.expectEqual(@as(abi.DevicePtr, 0x1040), try b.at(64));
    try std.testing.expectError(error.Invalid, b.at(65));
    var out: [8]u8 = undefined;
    try std.testing.expectError(error.Invalid, b.download(60, &out));
    try std.testing.expectError(error.Invalid, b.upload(57, &out));
}
