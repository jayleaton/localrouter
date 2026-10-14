//! Gap G3: the identity of the CUDA driver the daemon already opened. TensorFold's `Driver.open` loads libcuda.so.1 by
//! soname for the whole process; that is not changed and nothing here opens a second libcuda. Instead the driver that
//! is in use is observed and compared with a record:
//! - the NVIDIA kernel module's version, from the kernel's own module metadata (/sys/module/nvidia/version), a separate
//!   record from the CUDA API integer and never taken from a file name;
//! - cuDriverGetVersion's integer, and the raw return codes of cuInit, cuDriverGetVersion, cuDeviceGetCount and
//!   cuDeviceGet;
//! - the host kernel (/proc/sys/kernel/osrelease and /proc/sys/kernel/version);
//! - that every entry point in TensorFold's table, and its dlopen handle, belong to ONE mapped object, and that this
//!   object's mapping (path, device, inode) is a file with an fs-verity digest, size and sha256 equal to the record.
//!
//! What stays unproven until a grant measures it on the target: whether the deployed libcuda can carry fs-verity at all
//! (an overlay or a bind mount may not), and whether /proc/self/maps' device and inode equal the opened file's there.
//! Either missing refuses; so does any unrecorded value. Nothing here is relaxed to make a target pass.

const std = @import("std");
const sys_mod = @import("sys.zig");
const pinned = @import("pinned.zig");
const Sys = sys_mod.Sys;
const log = @import("report.zig"); // .err at run time, .warn in tests

pub const module_version_path = "/sys/module/nvidia/version";

/// The record a target is compared with. Every null is pending and refuses.
pub const Expect = struct {
    kernel_module: ?[]const u8 = null,
    api_version: ?c_int = null,
    host_release: ?[]const u8 = null,
    host_version: ?[]const u8 = null,
    /// The mapped libcuda, by the loader's recorded path and exact identity.
    libcuda: ?pinned.Resident = null,
};

pub const Error = error{
    DriverUnrecorded,
    DriverCallFailed,
    DriverNoDevice,
    DriverMetadataUnreadable,
    DriverKernelMismatch,
    DriverApiMismatch,
    DriverHostMismatch,
    DriverSymbolsSplit,
    DriverObjectMismatch,
    DriverExtraObject,
} || pinned.Error;

/// What the process actually has: a future grant records this, `validate` compares it.
pub const Observation = struct {
    /// The kernel module version, host release and host version, as read (see the accessors).
    text: [3][256]u8,
    lens: [3]usize,
    api_version: c_int,
    devices: c_int,
    /// The one object holding every driver entry point and the handle (its l_addr).
    base: usize,
    path: []const u8,
    dev_major: u32,
    dev_minor: u32,
    ino: u64,
    /// Set only by `validate`, after every comparison passed.
    validated: bool = false,

    pub fn kernelModule(self: *const Observation) []const u8 {
        return self.text[0][0..self.lens[0]];
    }
    pub fn hostRelease(self: *const Observation) []const u8 {
        return self.text[1][0..self.lens[1]];
    }
    pub fn hostVersion(self: *const Observation) []const u8 {
        return self.text[2][0..self.lens[2]];
    }

    pub fn deinit(self: *Observation, gpa: std.mem.Allocator) void {
        gpa.free(self.path);
    }
};

/// `api` is TensorFold's `cuda.abi.Api` table as the daemon's `Driver` resolved it, `handle` that driver's dlopen
/// handle (`driver.lib.inner.handle`). Calls only the four read-only entry points named above.
pub fn observe(gpa: std.mem.Allocator, api: anytype, handle: *anyopaque, sys: Sys) Error!Observation {
    var o: Observation = undefined;
    o.validated = false;
    o.lens[0] = try small(sys, module_version_path, &o.text[0]);
    o.lens[1] = try small(sys, "/proc/sys/kernel/osrelease", &o.text[1]);
    o.lens[2] = try small(sys, "/proc/sys/kernel/version", &o.text[2]);

    try raw("cuInit", api.cuInit(0));
    o.api_version = 0;
    try raw("cuDriverGetVersion", api.cuDriverGetVersion(&o.api_version));
    o.devices = 0;
    try raw("cuDeviceGetCount", api.cuDeviceGetCount(&o.devices));
    if (o.devices < 1) return error.DriverNoDevice;
    var dev: c_int = -1;
    try raw("cuDeviceGet", api.cuDeviceGet(&dev, 0));

    // every entry point, and the handle, in one object
    o.base = sys.handleBase(sys.ctx, handle) orelse return error.DriverSymbolsSplit;
    const info = @typeInfo(@TypeOf(api.*)).@"struct";
    inline for (info.field_names) |name| {
        const at = @intFromPtr(@field(api.*, name));
        if (sys.addrBase(sys.ctx, at) != o.base) {
            log.err("driver: {s} is not in the object the driver handle names", .{name});
            return error.DriverSymbolsSplit;
        }
    }

    const objs = sys.objects(sys.ctx, gpa) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Unavailable => error.DriverObjectMismatch,
    };
    defer sys_mod.freeObjects(gpa, objs);
    var found: ?sys_mod.Object = null;
    for (objs) |obj| if (obj.kind == .library and obj.base == o.base) {
        if (found != null) return error.DriverObjectMismatch;
        found = obj;
    };
    const f = found orelse return error.DriverObjectMismatch;
    if (f.ino == 0) return error.DriverObjectMismatch;
    // a second object from the same file, or another libcuda by name (a detector; the pinned set's closure check is
    // what identifies every object exactly)
    for (objs) |obj| {
        if (obj.kind != .library or obj.base == o.base) continue;
        const same_file = obj.ino == f.ino and obj.dev_major == f.dev_major and obj.dev_minor == f.dev_minor;
        if (same_file or std.mem.startsWith(u8, std.fs.path.basename(obj.path), "libcuda.so")) {
            log.err("driver: another libcuda object is mapped: {s}", .{obj.path});
            return error.DriverExtraObject;
        }
    }
    o.path = try gpa.dupe(u8, f.path);
    o.dev_major = f.dev_major;
    o.dev_minor = f.dev_minor;
    o.ino = f.ino;
    return o;
}

/// Observes, then refuses unless every field equals `expect` and the mapped libcuda is the recorded immutable file.
/// The validated observation is the evidence a grant session's driver step takes; the caller deinits it.
pub fn validate(gpa: std.mem.Allocator, api: anytype, handle: *anyopaque, expect: Expect, sys: Sys) Error!Observation {
    const want_module = expect.kernel_module orelse return unrecorded("kernel module version");
    const want_api = expect.api_version orelse return unrecorded("CUDA API version");
    const want_release = expect.host_release orelse return unrecorded("host kernel release");
    const want_version = expect.host_version orelse return unrecorded("host kernel version");
    const want_lib = expect.libcuda orelse return unrecorded("libcuda identity");

    var o = try observe(gpa, api, handle, sys);
    errdefer o.deinit(gpa);
    if (!std.mem.eql(u8, o.kernelModule(), want_module)) {
        log.err("driver: kernel module {s}, recorded {s}", .{ o.kernelModule(), want_module });
        return error.DriverKernelMismatch;
    }
    if (o.api_version != want_api) {
        log.err("driver: CUDA API {d}, recorded {d}", .{ o.api_version, want_api });
        return error.DriverApiMismatch;
    }
    if (!std.mem.eql(u8, o.hostRelease(), want_release) or !std.mem.eql(u8, o.hostVersion(), want_version)) return error.DriverHostMismatch;
    if (!std.mem.eql(u8, o.path, want_lib.path)) {
        log.err("driver: libcuda mapped from {s}, recorded {s}", .{ o.path, want_lib.path });
        return error.DriverObjectMismatch;
    }
    const buf = try gpa.alloc(u8, pinned.chunk_bytes);
    defer gpa.free(buf);
    const id = try pinned.verifyResident(sys, want_lib, buf);
    if (id.ino != o.ino or id.dev_major != o.dev_major or id.dev_minor != o.dev_minor) {
        log.err("driver: the mapped libcuda ({d}:{d}/{d}) is not the verified file ({d}:{d}/{d})", .{ o.dev_major, o.dev_minor, o.ino, id.dev_major, id.dev_minor, id.ino });
        return error.DriverObjectMismatch;
    }
    o.validated = true;
    return o;
}

fn unrecorded(what: []const u8) Error {
    log.err("driver: {s} is not recorded (G3 pending)", .{what});
    return error.DriverUnrecorded;
}

fn raw(what: []const u8, rc: c_int) Error!void {
    if (rc == 0) return;
    log.err("driver: {s} returned {d}", .{ what, rc });
    return error.DriverCallFailed;
}

/// Reads a small kernel-provided file into `buf`; its length.
fn small(sys: Sys, path: [*:0]const u8, buf: *[256]u8) Error!usize {
    const s = sys.readSmall(sys.ctx, path, buf) orelse {
        log.err("driver: cannot read {s}", .{path});
        return error.DriverMetadataUnreadable;
    };
    if (s.len == 0 or s.len > buf.len) return error.DriverMetadataUnreadable;
    std.mem.copyForwards(u8, buf[0..s.len], s);
    return s.len;
}

// ---- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;
const Fake = @import("fake.zig").Fake;

const TestApi = struct {
    cuInit: *const fn (c_uint) callconv(.c) c_int,
    cuDriverGetVersion: *const fn (*c_int) callconv(.c) c_int,
    cuDeviceGet: *const fn (*c_int, c_int) callconv(.c) c_int,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) c_int,
};

const T = struct {
    var count_rc: c_int = 0;
    fn init(_: c_uint) callconv(.c) c_int {
        return 0;
    }
    fn version(v: *c_int) callconv(.c) c_int {
        v.* = 13030;
        return 0;
    }
    fn get(d: *c_int, _: c_int) callconv(.c) c_int {
        d.* = 0;
        return 0;
    }
    fn count(n: *c_int) callconv(.c) c_int {
        n.* = 1;
        return count_rc;
    }
};

const libcuda_bytes = "the libcuda bytes";
const verity = @as([32]u8, @splat(0xcd));

fn driverFake(fs: []const Fake.File, objs: []const sys_mod.Object) Fake {
    return .{ .files = fs, .preexisting = objs, .fixed_base = 0x7e000000 };
}

const files = [_]Fake.File{
    .{ .path = "/sys/module/nvidia/version", .bytes = "999.1.2", .ino = 1 },
    .{ .path = "/proc/sys/kernel/osrelease", .bytes = "6.0.0-test", .ino = 2 },
    .{ .path = "/proc/sys/kernel/version", .bytes = "#1 SMP test", .ino = 3 },
    .{ .path = "/usr/lib/libcuda.so.1", .bytes = libcuda_bytes, .ino = 40, .verity = verity },
};

fn record() Expect {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(libcuda_bytes, &d, .{});
    return .{
        .kernel_module = "999.1.2",
        .api_version = 13030,
        .host_release = "6.0.0-test",
        .host_version = "#1 SMP test",
        .libcuda = .{ .path = "/usr/lib/libcuda.so.1", .soname = "libcuda.so.1", .size = libcuda_bytes.len, .sha256 = d, .verity = verity },
    };
}

test "the driver in use matches the record, and every mismatch refuses" {
    const api: TestApi = .{ .cuInit = T.init, .cuDriverGetVersion = T.version, .cuDeviceGet = T.get, .cuDeviceGetCount = T.count };
    const handle: *anyopaque = @ptrFromInt(0x7e000000);
    var objs = [_]sys_mod.Object{
        .{ .path = "/usr/lib/libcuda.so.1", .base = 0x7e000000, .dev_major = 8, .dev_minor = 1, .ino = 40, .kind = .library },
    };
    var fake = driverFake(&files, &objs);
    var v = try validate(testing.allocator, &api, handle, record(), fake.sys());
    try testing.expect(v.validated);
    try testing.expectEqual(@as(c_int, 13030), v.api_version);
    v.deinit(testing.allocator);

    try testing.expectError(error.DriverUnrecorded, validate(testing.allocator, &api, handle, .{}, fake.sys()));
    var e = record();
    e.kernel_module = "610.43.02";
    try testing.expectError(error.DriverKernelMismatch, validate(testing.allocator, &api, handle, e, fake.sys()));
    e = record();
    e.api_version = 13000;
    try testing.expectError(error.DriverApiMismatch, validate(testing.allocator, &api, handle, e, fake.sys()));
    e = record();
    e.libcuda.?.verity = null;
    try testing.expectError(error.PinNotImmutable, validate(testing.allocator, &api, handle, e, fake.sys()));

    // the mapping is another inode than the verified file
    objs[0].ino = 41;
    try testing.expectError(error.DriverObjectMismatch, validate(testing.allocator, &api, handle, record(), fake.sys()));
    objs[0].ino = 40;

    // one entry point resolves elsewhere
    fake.foreign_addr = @intFromPtr(api.cuDeviceGet);
    try testing.expectError(error.DriverSymbolsSplit, validate(testing.allocator, &api, handle, record(), fake.sys()));
    fake.foreign_addr = 0;

    T.count_rc = 3;
    defer T.count_rc = 0;
    try testing.expectError(error.DriverCallFailed, validate(testing.allocator, &api, handle, record(), fake.sys()));
}

test "a second libcuda refuses" {
    const api: TestApi = .{ .cuInit = T.init, .cuDriverGetVersion = T.version, .cuDeviceGet = T.get, .cuDeviceGetCount = T.count };
    const objs = [_]sys_mod.Object{
        .{ .path = "/usr/lib/libcuda.so.1", .base = 0x7e000000, .dev_major = 8, .dev_minor = 1, .ino = 40, .kind = .library },
        .{ .path = "/opt/compat/libcuda.so.1", .base = 0x6e000000, .dev_major = 8, .dev_minor = 1, .ino = 50, .kind = .library },
    };
    var fake = driverFake(&files, &objs);
    try testing.expectError(error.DriverExtraObject, validate(testing.allocator, &api, @ptrFromInt(0x7e000000), record(), fake.sys()));
}
