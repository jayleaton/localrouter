//! Free memory as the kernel reports it. On GB10 device allocations come out of system memory, so MemAvailable is
//! the budget the GPU tools and any other service on the machine share.
//!
//! Apple Silicon is unified memory too, so it gets one budget, never a VRAM figure plus a RAM figure. `available` there
//! is the smaller of what the VM system can give (total less wired, compressed and non-purgeable anonymous pages: file
//! cache counts as free, as in MemAvailable) and the GPU's headroom (its wired-memory ceiling less what is wired now:
//! Metal buffers in use are wired). Under critical memory pressure it is 0, so nothing new is admitted.

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

const darwin = builtin.os.tag.isDarwin();

/// What the kernel reports on memory pressure (macOS); `unknown` elsewhere.
pub const Pressure = enum { normal, warn, critical, unknown };

/// One reading of the host's memory, in bytes.
pub const Snapshot = struct {
    total: u64,
    available: u64, // before the GPU ceiling and pressure (`available` applies them)
    wired: u64 = 0,
    gpu_ceiling: u64, // the most the GPU may have wired; `total` where it is not separately limited
    pressure: Pressure = .unknown,

    /// The budget: the VM's free memory, within the GPU's headroom, 0 under critical pressure.
    pub fn budget(s: Snapshot) u64 {
        if (s.pressure == .critical) return 0;
        return @min(s.available, s.gpu_ceiling -| s.wired);
    }
};

/// The bytes a new model may take now (MemAvailable on Linux; see the file comment for macOS); null when unreadable.
pub fn available() ?u64 {
    if (darwin) return if (snapshot()) |s| s.budget() else null;
    return meminfo("MemAvailable:");
}

/// The host's memory in bytes; null when it cannot be read.
pub fn total() ?u64 {
    if (darwin) return sysctlInt(u64, "hw.memsize");
    return meminfo("MemTotal:");
}

pub fn snapshot() ?Snapshot {
    if (!darwin) {
        const t = meminfo("MemTotal:") orelse return null;
        return .{ .total = t, .available = meminfo("MemAvailable:") orelse return null, .gpu_ceiling = t };
    }
    const t = sysctlInt(u64, "hw.memsize") orelse return null;
    const page: u64 = sysctlInt(u64, "hw.pagesize") orelse return null;
    var vm: VmStatistics64 = undefined;
    var count: u32 = @sizeOf(VmStatistics64) / @sizeOf(u32);
    if (host_statistics64(std.c.mach_host_self(), host_vm_info64, &vm, &count) != 0) return null;
    const used = (@as(u64, vm.wire_count) + vm.compressor_page_count + (vm.internal_page_count -| vm.purgeable_count)) * page;
    return .{
        .total = t,
        .available = t -| used,
        .wired = @as(u64, vm.wire_count) * page,
        .gpu_ceiling = gpuCeiling(t),
        .pressure = switch (sysctlInt(u32, "kern.memorystatus_vm_pressure_level") orelse 0) {
            1 => .normal,
            2 => .warn,
            4 => .critical,
            else => .unknown,
        },
    };
}

fn meminfo(comptime key: []const u8) ?u64 {
    var buf: [4096]u8 = undefined;
    const fd = posix.openatZ(posix.AT.FDCWD, "/proc/meminfo", .{}, 0) catch return null;
    defer _ = posix.system.close(fd);
    const n = posix.read(fd, &buf) catch return null;
    return parseField(buf[0..n], key);
}

/// Parses the "MemAvailable:  123 kB" line.
pub fn parseAvailable(text: []const u8) ?u64 {
    return parseField(text, "MemAvailable:");
}

fn parseField(text: []const u8, key: []const u8) ?u64 {
    const at = std.mem.indexOf(u8, text, key) orelse return null;
    const rest = std.mem.trimStart(u8, text[at + key.len ..], " \t");
    const end = std.mem.indexOfAny(u8, rest, " \n") orelse rest.len;
    const kb = std.fmt.parseInt(u64, rest[0..end], 10) catch return null;
    return kb * 1024;
}

// ---- macOS ----

fn sysctlInt(comptime T: type, comptime name: [:0]const u8) ?T {
    var v: T = 0;
    var len: usize = @sizeOf(T);
    if (std.c.sysctlbyname(name, &v, &len, null, 0) != 0 or len != @sizeOf(T)) return null;
    return v;
}

/// The GPU's wired-memory ceiling: `iogpu.wired_limit_mb` when an administrator set one, else Metal's
/// `recommendedMaxWorkingSetSize` for the default device, else three quarters of memory. Read once.
fn gpuCeiling(t: u64) u64 {
    var bytes = ceiling.load(.acquire);
    if (bytes == unread) {
        // two first readers both ask Metal: harmless, they get the same figure
        bytes = if (sysctlInt(u32, "iogpu.wired_limit_mb")) |mb| (if (mb > 0) @as(u64, mb) << 20 else metalWorkingSet()) else metalWorkingSet();
        ceiling.store(bytes, .release);
    }
    return if (bytes > 0) @min(bytes, t) else t / 4 * 3;
}

const unread = std.math.maxInt(u64);
var ceiling: std.atomic.Value(u64) = .init(unread);

/// `MTLCreateSystemDefaultDevice().recommendedMaxWorkingSetSize`, 0 when there is no device.
fn metalWorkingSet() u64 {
    const dev = MTLCreateSystemDefaultDevice() orelse return 0;
    const get: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) u64 = @ptrCast(&objc_msgSend);
    const release: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void = @ptrCast(&objc_msgSend);
    const bytes = get(dev, sel_registerName("recommendedMaxWorkingSetSize"));
    release(dev, sel_registerName("release"));
    return bytes;
}

extern "c" fn MTLCreateSystemDefaultDevice() ?*anyopaque;
extern "c" fn sel_registerName(name: [*:0]const u8) ?*anyopaque;
extern "c" fn objc_msgSend() void;
extern "c" fn host_statistics64(host: std.c.mach_port_t, flavor: c_int, info: *VmStatistics64, count: *u32) c_int;
const host_vm_info64 = 4;

/// mach/vm_statistics.h `struct vm_statistics64`, in pages.
const VmStatistics64 = extern struct {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
    swapped_count: u64,
};

test "parse meminfo and read the real one" {
    try std.testing.expectEqual(@as(?u64, 4096 * 1024), parseAvailable("MemTotal: 9 kB\nMemAvailable:     4096 kB\n"));
    try std.testing.expect(parseAvailable("MemTotal: 9 kB\n") == null);
    const s = snapshot().?;
    try std.testing.expect(s.available > 0 and s.available <= s.total);
    try std.testing.expect(s.gpu_ceiling > 0 and s.gpu_ceiling <= s.total);
    try std.testing.expectEqual(total().?, s.total);
    try std.testing.expect(s.budget() <= s.available); // one reading: the machine's memory moves between two
    _ = available().?;
}

test "one budget: the smaller of free memory and the GPU's headroom, none under critical pressure" {
    const s: Snapshot = .{ .total = 96 << 30, .available = 60 << 30, .wired = 10 << 30, .gpu_ceiling = 64 << 30, .pressure = .normal };
    try std.testing.expectEqual(@as(u64, 54 << 30), s.budget());
    try std.testing.expectEqual(@as(u64, 40 << 30), (Snapshot{ .total = 96 << 30, .available = 40 << 30, .wired = 1 << 30, .gpu_ceiling = 64 << 30 }).budget());
    var c = s;
    c.pressure = .critical;
    try std.testing.expectEqual(@as(u64, 0), c.budget());
}
