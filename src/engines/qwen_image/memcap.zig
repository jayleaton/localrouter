//! Per-process caps on native GPU memory, enforced in front of the driver. `open` (instead of `cuda.Driver.open`)
//! routes the driver table's allocation entry points through this file: cuMemAlloc_v2 (device / UMA), cuMemHostAlloc
//! and cuMemHostRegister_v2 (pinned host), with their frees. Each allocation checks and reserves against the caps
//! BEFORE the driver is called; over a cap it is refused (CUDA_ERROR_OUT_OF_MEMORY) without calling the driver, and
//! no caller falls back. Device and pinned-host bytes are counted separately, and both inside one total. The table is
//! the only way anything in this binary reaches those entry points (TensorFold's DeviceBuffer, HostBuffer and
//! carveout all call `d.api.*`), so patching it covers every path; the calls, their sizes and their order are
//! unchanged, so kernels and allocation order are too.
//! Caps come from the environment, in bytes (absent: that cap is off, the counters still run):
//!   LOCALROUTER_MEMCAP_DEVICE_BYTES, LOCALROUTER_MEMCAP_HOST_BYTES, LOCALROUTER_MEMCAP_TOTAL_BYTES
//! LOCALROUTER_MEMCAP_RECEIPT names a file that receives the receipt (caps, current, peak, counts) as one JSON line:
//! written when the first driver is opened (the caps as read back), after every refusal, and at exit.
//! Counted bytes are the sizes requested; the driver's rounding and its own memory (context, modules, workspaces)
//! are outside, in the declared non-allocator overhead.

const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const abi = cuda.abi;
const testdir = @import("testdir.zig");

pub const Caps = struct {
    device: u64 = std.math.maxInt(u64),
    host: u64 = std.math.maxInt(u64),
    total: u64 = std.math.maxInt(u64),
};

pub const Counters = struct {
    device: u64 = 0,
    host: u64 = 0,
    peak_device: u64 = 0,
    peak_host: u64 = 0,
    peak_total: u64 = 0,
    allocations: u64 = 0,
    frees: u64 = 0,
    refused: u64 = 0,
    driver_failed: u64 = 0,
    unknown_frees: u64 = 0,
    rollback_failed: u64 = 0,
    drivers: u64 = 0,
    /// Set once memory the driver handed out could neither be recorded nor given back: its bytes stay counted (they
    /// are still resident) and every later allocation is refused.
    failed_closed: bool = false,
};

const Kind = enum(u8) { device, host };
const Entry = struct { len: u64, kind: Kind };
/// Which driver call made an allocation, so a rollback gives it back the matching way.
const Origin = enum { device_alloc, host_alloc, host_register };

/// Tests only: makes recording an allocation fail (compiled out of every non-test build).
var fail_track_for_test = false;

var lock_flag: std.atomic.Value(bool) = .init(false);
var caps: Caps = .{};
var counters: Counters = .{};
var live: std.AutoHashMapUnmanaged(u64, Entry) = .empty; // address -> what it holds
var configured = false;
var receipt_path: ?[*:0]const u8 = null;

// the driver's own entry points, captured from the first table patched (one libcuda per process)
var real_alloc: ?*const fn (*abi.DevicePtr, usize) callconv(.c) abi.Result = null;
var real_free: ?*const fn (abi.DevicePtr) callconv(.c) abi.Result = null;
var real_host_alloc: ?*const fn (*?*anyopaque, usize, c_uint) callconv(.c) abi.Result = null;
var real_free_host: ?*const fn (?*anyopaque) callconv(.c) abi.Result = null;
var real_register: ?*const fn (?*anyopaque, usize, c_uint) callconv(.c) abi.Result = null;
var real_unregister: ?*const fn (?*anyopaque) callconv(.c) abi.Result = null;

const out_of_memory: abi.Result = 2; // CUDA_ERROR_OUT_OF_MEMORY

fn lock() void {
    while (lock_flag.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}
fn unlock() void {
    lock_flag.store(false, .release);
}

/// The driver, with its allocation entry points behind the caps. Use this everywhere a driver is opened.
pub fn open() cuda.Error!cuda.Driver {
    var d = try cuda.Driver.open();
    install(&d);
    return d;
}

/// Routes `d`'s allocation entry points through the caps (idempotent). Reads the caps on first use.
pub fn install(d: *cuda.Driver) void {
    lock();
    if (!configured) configureFromEnv();
    if (real_alloc == null) {
        real_alloc = d.api.cuMemAlloc_v2;
        real_free = d.api.cuMemFree_v2;
        real_host_alloc = d.api.cuMemHostAlloc;
        real_free_host = d.api.cuMemFreeHost;
        real_register = d.api.cuMemHostRegister_v2;
        real_unregister = d.api.cuMemHostUnregister;
    }
    const fresh = d.api.cuMemAlloc_v2 != &capAlloc;
    if (fresh) {
        d.api.cuMemAlloc_v2 = &capAlloc;
        d.api.cuMemFree_v2 = &capFree;
        d.api.cuMemHostAlloc = &capHostAlloc;
        d.api.cuMemFreeHost = &capFreeHost;
        d.api.cuMemHostRegister_v2 = &capRegister;
        d.api.cuMemHostUnregister = &capUnregister;
        counters.drivers += 1;
    }
    const first = fresh and counters.drivers == 1;
    unlock();
    if (first) {
        const c = caps;
        std.log.info("memcap: caps device {d} host {d} total {d} bytes (maxInt: off)", .{ c.device, c.host, c.total });
        writeReceipt();
    }
}

/// Whether `d`'s allocation entry points go through the caps.
pub fn installed(d: *const cuda.Driver) bool {
    return d.api.cuMemAlloc_v2 == &capAlloc and d.api.cuMemHostAlloc == &capHostAlloc and d.api.cuMemHostRegister_v2 == &capRegister;
}

pub fn snapshot() Counters {
    lock();
    defer unlock();
    return counters;
}

pub fn currentCaps() Caps {
    lock();
    defer unlock();
    return caps;
}

extern "c" fn atexit(f: *const fn () callconv(.c) void) c_int;

fn configureFromEnv() void {
    configured = true;
    caps = .{
        .device = envBytes("LOCALROUTER_MEMCAP_DEVICE_BYTES"),
        .host = envBytes("LOCALROUTER_MEMCAP_HOST_BYTES"),
        .total = envBytes("LOCALROUTER_MEMCAP_TOTAL_BYTES"),
    };
    receipt_path = if (std.c.getenv("LOCALROUTER_MEMCAP_RECEIPT")) |p| p else null;
    if (receipt_path != null) _ = atexit(&exitReceipt);
}

/// A cap in bytes from the environment; absent means off. A value that does not parse is the smallest cap (0), so a
/// typo refuses every allocation rather than lifting the cap.
fn envBytes(name: [*:0]const u8) u64 {
    const v = std.c.getenv(name) orelse return std.math.maxInt(u64);
    return std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0;
}

/// Sets the caps directly (tests and embedders); `null` keeps the receipt path as it is.
pub fn configure(c: Caps, receipt: ?[*:0]const u8) void {
    lock();
    configured = true;
    caps = c;
    if (receipt) |r| receipt_path = r;
    unlock();
}

/// Forgets every count and reservation (tests only: the live table must describe real allocations otherwise).
pub fn resetForTest() void {
    lock();
    counters = .{};
    fail_track_for_test = false;
    live.clearAndFree(std.heap.page_allocator);
    real_alloc = null;
    real_free = null;
    real_host_alloc = null;
    real_free_host = null;
    real_register = null;
    real_unregister = null;
    unlock();
}

/// Reserves `n` bytes of `kind` if every cap allows it (lock held).
fn reserve(kind: Kind, n: u64) bool {
    if (counters.failed_closed) return false;
    const dev = counters.device + if (kind == .device) n else 0;
    const host = counters.host + if (kind == .host) n else 0;
    if (dev > caps.device or host > caps.host or dev + host > caps.total or dev < counters.device or host < counters.host) return false;
    counters.device = dev;
    counters.host = host;
    counters.peak_device = @max(counters.peak_device, dev);
    counters.peak_host = @max(counters.peak_host, host);
    counters.peak_total = @max(counters.peak_total, dev + host);
    return true;
}

fn release(kind: Kind, n: u64) void {
    switch (kind) {
        .device => counters.device -= n,
        .host => counters.host -= n,
    }
}

/// Check and reserve `n` bytes of `kind` before the driver is called; false: refused (counted, receipt written).
fn begin(kind: Kind, n: usize) bool {
    lock();
    const ok = reserve(kind, n);
    if (!ok) counters.refused += 1;
    unlock();
    if (!ok) writeReceipt();
    return ok;
}

fn track(addr: u64, e: Entry) !void {
    if (builtin.is_test and fail_track_for_test) return error.OutOfMemory;
    try live.put(std.heap.page_allocator, addr, e);
}

/// After the driver call: record the allocation at `addr`, or roll the reservation back if the driver failed. If it
/// cannot be recorded, it is given back the way it was made (free, free-host or unregister) and refused; the
/// reservation is released only once that give-back succeeded. A give-back that fails leaves the bytes counted (they
/// are still resident, but no longer findable) and fails closed: every later allocation is refused.
fn finish(origin: Origin, n: usize, res: abi.Result, addr: u64) abi.Result {
    const kind: Kind = if (origin == .device_alloc) .device else .host;
    lock();
    defer unlock();
    if (res != abi.success) {
        release(kind, n);
        counters.driver_failed += 1;
        return res;
    }
    track(addr, .{ .len = n, .kind = kind }) catch {
        counters.refused += 1;
        const undo = switch (origin) {
            .device_alloc => real_free.?(addr),
            .host_alloc => real_free_host.?(@ptrFromInt(addr)),
            .host_register => real_unregister.?(@ptrFromInt(addr)),
        };
        if (undo == abi.success) {
            release(kind, n);
        } else {
            counters.rollback_failed += 1;
            counters.failed_closed = true;
        }
        return out_of_memory;
    };
    counters.allocations += 1;
    return res;
}

/// Forgets `addr` (lock held); an address this file did not record counts as an unknown free.
fn forget(addr: u64) void {
    if (live.fetchRemove(addr)) |kv| {
        release(kv.value.kind, kv.value.len);
        counters.frees += 1;
    } else counters.unknown_frees += 1;
}

fn capAlloc(p: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    if (!begin(.device, n)) return out_of_memory;
    const res = real_alloc.?(p, n);
    return finish(.device_alloc, n, res, if (res == abi.success) p.* else 0);
}

fn capHostAlloc(p: *?*anyopaque, n: usize, flags: c_uint) callconv(.c) abi.Result {
    if (!begin(.host, n)) return out_of_memory;
    const res = real_host_alloc.?(p, n, flags);
    return finish(.host_alloc, n, res, if (res == abi.success) @intFromPtr(p.*) else 0);
}

fn capRegister(p: ?*anyopaque, n: usize, flags: c_uint) callconv(.c) abi.Result {
    if (!begin(.host, n)) return out_of_memory;
    const res = real_register.?(p, n, flags);
    return finish(.host_register, n, res, @intFromPtr(p));
}

fn capFree(p: abi.DevicePtr) callconv(.c) abi.Result {
    const res = real_free.?(p);
    if (res == abi.success) {
        lock();
        forget(p);
        unlock();
    }
    return res;
}

fn capFreeHost(p: ?*anyopaque) callconv(.c) abi.Result {
    const res = real_free_host.?(p);
    if (res == abi.success) {
        lock();
        forget(@intFromPtr(p));
        unlock();
    }
    return res;
}

fn capUnregister(p: ?*anyopaque) callconv(.c) abi.Result {
    const res = real_unregister.?(p);
    if (res == abi.success) {
        lock();
        forget(@intFromPtr(p));
        unlock();
    }
    return res;
}

fn exitReceipt() callconv(.c) void {
    writeReceipt();
}

/// One JSON line with the caps, the counters and the peak, replacing the file (no allocation, libc only).
pub fn writeReceipt() void {
    const path = receipt_path orelse return;
    lock();
    const c = caps;
    const n = counters;
    unlock();
    var buf: [1024]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "{{\"schema\": \"localrouter-memcap-v1\", \"caps\": {{\"device\": {d}, \"host\": {d}, \"total\": {d}}}, " ++
        "\"current\": {{\"device\": {d}, \"host\": {d}}}, \"peak\": {{\"device\": {d}, \"host\": {d}, \"total\": {d}}}, " ++
        "\"allocations\": {d}, \"frees\": {d}, \"refused\": {d}, \"driver_failed\": {d}, \"unknown_frees\": {d}, " ++
        "\"rollback_failed\": {d}, \"failed_closed\": {}, \"drivers\": {d}}}\n", .{
        c.device,      c.host,  c.total,   n.device,        n.host,          n.peak_device,     n.peak_host,     n.peak_total,
        n.allocations, n.frees, n.refused, n.driver_failed, n.unknown_frees, n.rollback_failed, n.failed_closed, n.drivers,
    }) catch return;
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    _ = std.c.write(fd, line.ptr, line.len);
}

// ------------------------------------------------------------------------------------------------ CPU-only controls
// A stand-in driver table: the real TensorFold DeviceBuffer / HostBuffer code runs against it, so these tests exercise
// the same call paths the engines use, without a GPU.

const Fake = struct {
    var next: u64 = 0x1000;
    var calls: u32 = 0;
    var fail_next = false;
    var fail_undo = false; // free / free-host / unregister report failure
    var frees: u32 = 0;
    var free_hosts: u32 = 0;
    var unregisters: u32 = 0;
    var host_mem: [4][64]u8 align(16) = undefined;
    var host_i: usize = 0;

    fn alloc(p: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
        calls += 1;
        if (fail_next) {
            fail_next = false;
            return 999;
        }
        p.* = next;
        next += std.mem.alignForward(u64, @max(n, 1), 256);
        return abi.success;
    }
    fn free(_: abi.DevicePtr) callconv(.c) abi.Result {
        frees += 1;
        return if (fail_undo) 999 else abi.success;
    }
    fn hostAlloc(p: *?*anyopaque, _: usize, _: c_uint) callconv(.c) abi.Result {
        calls += 1;
        p.* = @ptrCast(&host_mem[host_i]);
        host_i = (host_i + 1) % host_mem.len;
        return abi.success;
    }
    fn freeHost(_: ?*anyopaque) callconv(.c) abi.Result {
        free_hosts += 1;
        return if (fail_undo) 999 else abi.success;
    }
    fn register(_: ?*anyopaque, _: usize, _: c_uint) callconv(.c) abi.Result {
        calls += 1;
        return abi.success;
    }
    fn unregister(_: ?*anyopaque) callconv(.c) abi.Result {
        unregisters += 1;
        return if (fail_undo) 999 else abi.success;
    }
    fn errName(_: abi.Result, s: *?[*:0]const u8) callconv(.c) abi.Result {
        s.* = "CUDA_ERROR_FAKE";
        return abi.success;
    }
    fn errText(_: abi.Result, s: *?[*:0]const u8) callconv(.c) abi.Result {
        s.* = "fake";
        return abi.success;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = &alloc;
        d.api.cuMemFree_v2 = &free;
        d.api.cuMemHostAlloc = &hostAlloc;
        d.api.cuMemFreeHost = &freeHost;
        d.api.cuMemHostRegister_v2 = &register;
        d.api.cuMemHostUnregister = &unregister;
        d.api.cuGetErrorName = &errName;
        d.api.cuGetErrorString = &errText;
        return d;
    }
};

fn freshDriver(c: Caps) cuda.Driver {
    resetForTest();
    configure(c, null);
    Fake.calls = 0;
    Fake.fail_undo = false;
    Fake.frees = 0;
    Fake.free_hosts = 0;
    Fake.unregisters = 0;
    var d = Fake.driver();
    install(&d);
    return d;
}

test "an allocation over the device cap is refused before the driver is called; under it, it is counted" {
    var d = freshDriver(.{ .device = 1000 });
    try std.testing.expect(installed(&d));
    var a = try cuda.DeviceBuffer.alloc(&d, 600);
    try std.testing.expectEqual(@as(u32, 1), Fake.calls);
    var p: abi.DevicePtr = 0;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemAlloc_v2(&p, 500)); // DeviceBuffer.alloc maps 2 to OutOfDeviceMemory
    try std.testing.expectEqual(@as(u32, 1), Fake.calls); // refused: the driver never saw it
    const s = snapshot();
    try std.testing.expectEqual(@as(u64, 600), s.device);
    try std.testing.expectEqual(@as(u64, 1), s.refused);
    a.free();
    try std.testing.expectEqual(@as(u64, 0), snapshot().device);
}

test "frees give the bytes back; the peak stays; a freed slot can be used again" {
    var d = freshDriver(.{ .device = 1000 });
    var a = try cuda.DeviceBuffer.alloc(&d, 700);
    a.free();
    var b = try cuda.DeviceBuffer.alloc(&d, 900);
    defer b.free();
    const s = snapshot();
    try std.testing.expectEqual(@as(u64, 900), s.device);
    try std.testing.expectEqual(@as(u64, 900), s.peak_device);
    try std.testing.expectEqual(@as(u64, 2), s.allocations);
    try std.testing.expectEqual(@as(u64, 1), s.frees);
}

test "pinned host bytes count on their own and inside the total; register counts as pinned host" {
    var d = freshDriver(.{ .device = 1000, .host = 100, .total = 1050 });
    var a = try cuda.DeviceBuffer.alloc(&d, 1000);
    defer a.free();
    var h = try cuda.HostBuffer.alloc(&d, 40);
    var hp: ?*anyopaque = null;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemHostAlloc(&hp, 20, 0)); // total 1060 > 1050
    var dummy: [8]u8 = undefined;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemHostRegister_v2(@ptrCast(&dummy), 70, 0)); // host 110 > 100
    try std.testing.expectEqual(abi.success, d.api.cuMemHostRegister_v2(@ptrCast(&dummy), 10, 0));
    var s = snapshot();
    try std.testing.expectEqual(@as(u64, 1000), s.device);
    try std.testing.expectEqual(@as(u64, 50), s.host);
    try std.testing.expectEqual(@as(u64, 1050), s.peak_total);
    try std.testing.expectEqual(@as(u64, 2), s.refused);
    try std.testing.expectEqual(abi.success, d.api.cuMemHostUnregister(@ptrCast(&dummy)));
    h.free();
    s = snapshot();
    try std.testing.expectEqual(@as(u64, 0), s.host);
}

test "a driver failure rolls the reservation back; an unknown free is counted, not trusted" {
    var d = freshDriver(.{ .device = 1000 });
    Fake.fail_next = true;
    var p: abi.DevicePtr = 0;
    try std.testing.expectEqual(@as(abi.Result, 999), d.api.cuMemAlloc_v2(&p, 800));
    var s = snapshot();
    try std.testing.expectEqual(@as(u64, 0), s.device);
    try std.testing.expectEqual(@as(u64, 1), s.driver_failed);
    _ = d.api.cuMemFree_v2(0xdead00);
    s = snapshot();
    try std.testing.expectEqual(@as(u64, 1), s.unknown_frees);
    try std.testing.expectEqual(@as(u64, 0), s.device);
}

test "install is idempotent; a second driver shares the same process caps" {
    var d = freshDriver(.{ .device = 1000 });
    install(&d);
    var e = Fake.driver();
    install(&e);
    try std.testing.expect(installed(&e));
    try std.testing.expectEqual(@as(u64, 2), snapshot().drivers);
    var a = try cuda.DeviceBuffer.alloc(&d, 600);
    defer a.free();
    var p: abi.DevicePtr = 0;
    try std.testing.expectEqual(out_of_memory, e.api.cuMemAlloc_v2(&p, 600));
}

test "no caps set: nothing is refused, everything is still counted" {
    var d = freshDriver(.{});
    var a = try cuda.DeviceBuffer.alloc(&d, 1 << 40);
    defer a.free();
    try std.testing.expectEqual(@as(u64, 1 << 40), snapshot().device);
    try std.testing.expectEqual(@as(u64, 0), snapshot().refused);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

test "caps from the environment: absent is off; an unparsable value is 0, so it refuses rather than lifts" {
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), envBytes("LOCALROUTER_MEMCAP_TEST_UNSET_9f3c"));
    _ = setenv("LOCALROUTER_MEMCAP_TEST_9f3c", "12x", 1);
    try std.testing.expectEqual(@as(u64, 0), envBytes("LOCALROUTER_MEMCAP_TEST_9f3c"));
    _ = setenv("LOCALROUTER_MEMCAP_TEST_9f3c", "4096", 1);
    try std.testing.expectEqual(@as(u64, 4096), envBytes("LOCALROUTER_MEMCAP_TEST_9f3c"));
}

test "the receipt: caps as read back, current, peak and counts; rewritten after a refusal" {
    var tmp: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer tmp.deinit();
    const dir = try testdir.make(std.testing.io, tmp.allocator(), "/tmp", "localrouter-memcap-receipt-test");
    const path = try testdir.file(tmp.allocator(), dir, "receipt.json");
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, dir) catch {};
    var d = freshDriver(.{ .device = 1000, .host = 100, .total = 1050 });
    configure(.{ .device = 1000, .host = 100, .total = 1050 }, path.ptr);
    defer configure(.{}, null);
    var a = try cuda.DeviceBuffer.alloc(&d, 600);
    defer a.free();
    var p: abi.DevicePtr = 0;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemAlloc_v2(&p, 500)); // writes the receipt
    const io = std.testing.io;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(text);
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
    const R = struct {
        schema: []const u8,
        caps: struct { device: u64, host: u64, total: u64 },
        current: struct { device: u64, host: u64 },
        peak: struct { device: u64, host: u64, total: u64 },
        allocations: u64,
        refused: u64,
    };
    const r = try std.json.parseFromSlice(R, std.testing.allocator, text, .{ .ignore_unknown_fields = true });
    defer r.deinit();
    try std.testing.expectEqualStrings("localrouter-memcap-v1", r.value.schema);
    try std.testing.expectEqual(@as(u64, 1000), r.value.caps.device);
    try std.testing.expectEqual(@as(u64, 1050), r.value.caps.total);
    try std.testing.expectEqual(@as(u64, 600), r.value.current.device);
    try std.testing.expectEqual(@as(u64, 600), r.value.peak.total);
    try std.testing.expectEqual(@as(u64, 1), r.value.allocations);
    try std.testing.expectEqual(@as(u64, 1), r.value.refused);
}

test "tracking fails, the give-back succeeds: the matching call returns it, the reservation is released, nothing latches" {
    var d = freshDriver(.{ .device = 1000, .host = 100, .total = 1100 });
    fail_track_for_test = true;
    var p: abi.DevicePtr = 0;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemAlloc_v2(&p, 600));
    var hp: ?*anyopaque = null;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemHostAlloc(&hp, 40, 0));
    var buf: [8]u8 = undefined;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemHostRegister_v2(@ptrCast(&buf), 30, 0));
    try std.testing.expectEqual(@as(u32, 1), Fake.frees); // device memory given back with free
    try std.testing.expectEqual(@as(u32, 1), Fake.free_hosts); // cuMemHostAlloc memory with free-host
    try std.testing.expectEqual(@as(u32, 1), Fake.unregisters); // registered memory with unregister, never free-host
    var s = snapshot();
    try std.testing.expectEqual(@as(u64, 0), s.device);
    try std.testing.expectEqual(@as(u64, 0), s.host);
    try std.testing.expectEqual(@as(u64, 3), s.refused);
    try std.testing.expect(!s.failed_closed);
    fail_track_for_test = false;
    var a = try cuda.DeviceBuffer.alloc(&d, 1000); // the cap is whole again
    a.free();
    s = snapshot();
    try std.testing.expectEqual(@as(u64, 0), s.device);
}

test "tracking fails and the give-back fails: the bytes stay counted and no further allocation is permitted" {
    var d = freshDriver(.{ .device = 1000, .host = 100, .total = 1100 });
    fail_track_for_test = true;
    Fake.fail_undo = true;
    var p: abi.DevicePtr = 0;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemAlloc_v2(&p, 600));
    fail_track_for_test = false;
    Fake.fail_undo = false;
    var s = snapshot();
    try std.testing.expectEqual(@as(u64, 600), s.device); // still resident, still counted
    try std.testing.expectEqual(@as(u64, 1), s.rollback_failed);
    try std.testing.expect(s.failed_closed);
    const before = Fake.calls;
    // even far under every cap, nothing more reaches the driver
    try std.testing.expectEqual(out_of_memory, d.api.cuMemAlloc_v2(&p, 1));
    var hp: ?*anyopaque = null;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemHostAlloc(&hp, 1, 0));
    var buf: [8]u8 = undefined;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemHostRegister_v2(@ptrCast(&buf), 1, 0));
    try std.testing.expectEqual(before, Fake.calls);
    s = snapshot();
    try std.testing.expectEqual(@as(u64, 600), s.device);
    try std.testing.expectEqual(@as(u64, 4), s.refused);
}

test "a registered range whose give-back fails latches too, and the receipt says so" {
    var tmp: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer tmp.deinit();
    const dir = try testdir.make(std.testing.io, tmp.allocator(), "/tmp", "localrouter-memcap-rollback-test");
    const path = try testdir.file(tmp.allocator(), dir, "receipt.json");
    defer std.Io.Dir.cwd().deleteTree(std.testing.io, dir) catch {};
    var d = freshDriver(.{ .device = 1000, .host = 100, .total = 1100 });
    configure(.{ .device = 1000, .host = 100, .total = 1100 }, path.ptr);
    defer configure(.{}, null);
    fail_track_for_test = true;
    Fake.fail_undo = true;
    var buf: [8]u8 = undefined;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemHostRegister_v2(@ptrCast(&buf), 30, 0));
    fail_track_for_test = false;
    Fake.fail_undo = false;
    try std.testing.expectEqual(@as(u32, 1), Fake.unregisters);
    try std.testing.expectEqual(@as(u32, 0), Fake.free_hosts);
    var p: abi.DevicePtr = 0;
    try std.testing.expectEqual(out_of_memory, d.api.cuMemAlloc_v2(&p, 1)); // refused, and writes the receipt
    const io = std.testing.io;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(text);
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
    const R = struct { current: struct { device: u64, host: u64 }, rollback_failed: u64, failed_closed: bool, refused: u64 };
    const r = try std.json.parseFromSlice(R, std.testing.allocator, text, .{ .ignore_unknown_fields = true });
    defer r.deinit();
    try std.testing.expectEqual(@as(u64, 30), r.value.current.host);
    try std.testing.expectEqual(@as(u64, 1), r.value.rollback_failed);
    try std.testing.expect(r.value.failed_closed);
    try std.testing.expectEqual(@as(u64, 2), r.value.refused);
}
