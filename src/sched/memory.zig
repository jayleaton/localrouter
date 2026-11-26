//! Free memory as the kernel reports it. On GB10 device allocations come out of system memory, so MemAvailable is
//! the budget the GPU tools and any other service on the machine share.
//!
//! Apple Silicon is unified memory too, so it gets one budget, never a VRAM figure plus a RAM figure. `available` there
//! is the smaller of what the VM system can give (total less wired, compressed and non-purgeable anonymous pages: file
//! cache counts as free, as in MemAvailable) and the GPU's headroom (its wired-memory ceiling less what is wired now:
//! Metal buffers in use are wired). Under critical memory pressure it is 0, so nothing new is admitted.
//!
//! An AMD APU (Strix Halo) maps system memory through its GTT pool, which `amdgpu.gttsize` / `ttm.pages_limit` cap below
//! RAM; an allocation past it fails even while MemAvailable has room. There the GTT is the GPU's ceiling and what it
//! holds is wired: the budget is the smaller of MemAvailable and the GTT left (`readGtt`, from sysfs).

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

/// The bytes a new model may take now (MemAvailable on Linux, within an AMD APU's GTT; see the file comment for macOS);
/// null when unreadable.
pub fn available() ?u64 {
    if (darwin) return if (snapshot()) |s| s.budget() else null;
    const free = meminfo("MemAvailable:") orelse return null;
    const g = readGtt("/sys") orelse return free;
    return @min(free, g.total -| g.used);
}

/// The host's memory in bytes; null when it cannot be read.
pub fn total() ?u64 {
    if (darwin) return sysctlInt(u64, "hw.memsize");
    return meminfo("MemTotal:");
}

/// The most the scheduler can ever book: `total`, within an AMD APU's GTT on Linux; null when unreadable.
pub fn capacity() ?u64 {
    const t = total() orelse return null;
    if (darwin) return t;
    return if (readGtt("/sys")) |g| @min(t, g.total) else t;
}

pub fn snapshot() ?Snapshot {
    if (!darwin) {
        const t = meminfo("MemTotal:") orelse return null;
        const free = meminfo("MemAvailable:") orelse return null;
        const g = readGtt("/sys") orelse return .{ .total = t, .available = free, .gpu_ceiling = t };
        return .{ .total = t, .available = free, .wired = g.used, .gpu_ceiling = @min(t, g.total) };
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

// ---- Linux: an AMD APU's GTT ----

/// An AMD GPU's GTT pool (the system memory it may map), in bytes, when it is the GPU's memory: an APU, whose VRAM is a
/// BIOS carve-out smaller than its GTT (tools/install/lib.sh `hw_vram` draws the same line). Null on a discrete card,
/// with no AMD card, and when an NVIDIA GPU is present (the engines run there, as lib.sh's detection prefers it).
pub const Gtt = struct { total: u64, used: u64 };

/// The first AMD (vendor 0x1002) DRM card with a GTT counter under `sysfs` (normally "/sys"), as lib.sh's `amd_card`.
pub fn readGtt(sysfs: []const u8) ?Gtt {
    var path: [std.fs.max_path_bytes]u8 = undefined;
    var amd: ?Gtt = null;
    var card: u32 = 0;
    while (card < 16) : (card += 1) {
        const dev = std.fmt.bufPrint(&path, "{s}/class/drm/card{d}/device/", .{ sysfs, card }) catch return null;
        const vendor = readInt(&path, dev.len, "vendor") orelse continue;
        if (vendor == 0x10de) return null;
        if (vendor != 0x1002 or amd != null) continue;
        const gtt = readInt(&path, dev.len, "mem_info_gtt_total") orelse continue;
        const vram = readInt(&path, dev.len, "mem_info_vram_total") orelse 0;
        if (vram >= gtt) return null; // a discrete card: its VRAM is its memory
        amd = .{ .total = gtt, .used = readInt(&path, dev.len, "mem_info_gtt_used") orelse 0 };
    }
    return amd;
}

/// A sysfs file's one number (decimal, or hex with 0x), at `path[0..dir_len] ++ name`.
fn readInt(path: *[std.fs.max_path_bytes]u8, dir_len: usize, name: []const u8) ?u64 {
    if (dir_len + name.len + 1 > path.len) return null;
    @memcpy(path[dir_len..][0..name.len], name);
    path[dir_len + name.len] = 0;
    const fd = posix.openatZ(posix.AT.FDCWD, path[0 .. dir_len + name.len :0], .{}, 0) catch return null;
    defer _ = posix.system.close(fd);
    var buf: [64]u8 = undefined;
    const n = posix.read(fd, &buf) catch return null;
    const text = std.mem.trim(u8, buf[0..n], " \t\n");
    if (std.mem.startsWith(u8, text, "0x")) return std.fmt.parseInt(u64, text[2..], 16) catch null;
    return std.fmt.parseInt(u64, text, 10) catch null;
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
    if (!darwin) try std.testing.expectEqual(s.gpu_ceiling, capacity().?); // MemTotal, within an APU's GTT
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

test "an AMD APU's GTT, read as lib.sh reads it, from the recorded hosts (tests/fixtures/amd)" {
    const strix = readGtt("tests/fixtures/amd/strix-halo/sys").?; // card1 after DP/eDP connector entries
    try std.testing.expectEqual(@as(u64, 103079215104), strix.total);
    try std.testing.expectEqual(@as(u64, 57671680), strix.used);
    try std.testing.expect(readGtt("tests/fixtures/amd/discrete/sys") == null); // 24 GiB VRAM above its 16 GiB GTT
    try std.testing.expect(readGtt("tests/fixtures/amd/nvidia/sys") == null); // an NVIDIA GPU beside an AMD card
    try std.testing.expect(readGtt("tests/fixtures/amd/none/sys") == null);
    try std.testing.expect(readGtt("/nonexistent") == null);
    const s: Snapshot = .{ .total = 128 << 30, .available = 120 << 30, .wired = strix.used, .gpu_ceiling = strix.total };
    try std.testing.expectEqual(strix.total - strix.used, s.budget()); // GTT, not MemAvailable, is the bound here
}
