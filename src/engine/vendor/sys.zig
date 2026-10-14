//! The OS calls behind the pinned loader and the driver check, in one table so every refusal is unit-tested without a
//! GPU, a vendor library or the real dynamic loader. `real` calls glibc; nothing here decides anything.

const std = @import("std");
const linux = std.os.linux;

pub const Fd = c_int;

pub const Stat = struct {
    dev_major: u32,
    dev_minor: u32,
    ino: u64,
    size: u64,
    mode: u16,
    /// STATX_ATTR_VERITY: the kernel enforces the file's fs-verity Merkle tree on every read, and refuses writes.
    verity: bool,

    pub fn regular(self: Stat) bool {
        return self.mode & linux.S.IFMT == linux.S.IFREG;
    }
};

pub const Kind = enum {
    /// The program itself: its mapping's inode is /proc/self/exe's.
    main,
    /// The kernel's vDSO: its mapping is the one /proc/self/maps labels [vdso].
    vdso,
    library,
};

/// One object the dynamic loader has mapped (dl_iterate_phdr), with the file mapping holding its program headers.
pub const Object = struct {
    /// dlpi_name, as the loader recorded it.
    path: []const u8,
    /// dlpi_addr (the load bias, the link map's l_addr).
    base: usize,
    /// /proc/self/maps' device and inode for the mapping that holds the object's program headers (0 when anonymous).
    dev_major: u32,
    dev_minor: u32,
    ino: u64,
    kind: Kind,
};

/// glibc's dlpi_adds / dlpi_subs: how many objects were ever loaded and unloaded.
pub const Counts = struct { adds: u64, subs: u64 };

pub const ObjectsError = error{ OutOfMemory, Unavailable };

pub const Sys = struct {
    ctx: ?*anyopaque = null,
    envPresent: *const fn (?*anyopaque, [*:0]const u8) bool,
    /// O_RDONLY | O_CLOEXEC, and O_NOFOLLOW unless `follow`.
    open: *const fn (?*anyopaque, [*:0]const u8, follow: bool) ?Fd,
    stat: *const fn (?*anyopaque, Fd) ?Stat,
    /// Fills `buf` from `offset`; the byte count read (short only at the end of the file).
    pread: *const fn (?*anyopaque, Fd, []u8, u64) ?usize,
    /// memfd_create(MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_EXEC).
    memfd: *const fn (?*anyopaque, [*:0]const u8) ?Fd,
    writeAll: *const fn (?*anyopaque, Fd, []const u8) bool,
    addSeals: *const fn (?*anyopaque, Fd, u32) bool,
    getSeals: *const fn (?*anyopaque, Fd) ?u32,
    /// FS_IOC_MEASURE_VERITY with SHA-256: the fs-verity file digest.
    measureVerity: *const fn (?*anyopaque, Fd) ?[32]u8,
    /// A read-only shared mapping of the first `len` bytes.
    mapRead: *const fn (?*anyopaque, Fd, usize) ?[]const u8,
    unmap: *const fn (?*anyopaque, []const u8) void,
    close: *const fn (?*anyopaque, Fd) void,
    pageSize: *const fn (?*anyopaque) usize,
    /// dlopen(path, RTLD_NOW | RTLD_GLOBAL).
    dlopen: *const fn (?*anyopaque, [*:0]const u8) ?*anyopaque,
    dlsym: *const fn (?*anyopaque, *anyopaque, [*:0]const u8) ?*anyopaque,
    /// dlclose's raw result (0 on success).
    dlclose: *const fn (?*anyopaque, *anyopaque) c_int,
    /// The link map's l_addr for a dlopen handle (dlinfo RTLD_DI_LINKMAP).
    handleBase: *const fn (?*anyopaque, *anyopaque) ?usize,
    /// The l_addr of the object defining the code at `addr` (dladdr1 RTLD_DL_LINKMAP).
    addrBase: *const fn (?*anyopaque, usize) ?usize,
    objects: *const fn (?*anyopaque, std.mem.Allocator) ObjectsError![]Object,
    counts: *const fn (?*anyopaque) ?Counts,
    /// Reads a small file (sysfs, procfs) into `buf`; trailing newlines are trimmed.
    readSmall: *const fn (?*anyopaque, [*:0]const u8, []u8) ?[]const u8,
};

pub fn freeObjects(gpa: std.mem.Allocator, objs: []Object) void {
    for (objs) |o| gpa.free(o.path);
    gpa.free(objs);
}

// glibc, not in std.c
const Dl_info = extern struct {
    dli_fname: ?[*:0]const u8,
    dli_fbase: ?*anyopaque,
    dli_sname: ?[*:0]const u8,
    dli_saddr: ?*anyopaque,
};
const LinkMap = extern struct { l_addr: usize, l_name: ?[*:0]const u8, l_ld: ?*anyopaque, l_next: ?*LinkMap, l_prev: ?*LinkMap };
const rtld_di_linkmap = 2;
const rtld_dl_linkmap = 2;
extern "c" fn dladdr1(addr: *const anyopaque, info: *Dl_info, extra: *?*anyopaque, flags: c_int) c_int;
extern "c" fn dlinfo(handle: *anyopaque, request: c_int, arg: *anyopaque) c_int;

/// glibc's dl_phdr_info with the members after dlpi_phnum (present when the callback's size covers them).
const PhdrInfo = extern struct {
    addr: usize,
    name: ?[*:0]const u8,
    phdr: usize,
    phnum: u16,
    adds: c_ulonglong,
    subs: c_ulonglong,
};

/// FS_IOC_MEASURE_VERITY: _IOWR('f', 134, struct fsverity_digest).
const fs_ioc_measure_verity: u32 = 0xC0046686;
const fsverity_hash_alg_sha256 = 1;

pub const real: Sys = .{
    .envPresent = R.envPresent,
    .open = R.open,
    .stat = R.stat,
    .pread = R.pread,
    .memfd = R.memfd,
    .writeAll = R.writeAll,
    .addSeals = R.addSeals,
    .getSeals = R.getSeals,
    .measureVerity = R.measureVerity,
    .mapRead = R.mapRead,
    .unmap = R.unmap,
    .close = R.close,
    .pageSize = R.pageSize,
    .dlopen = R.dlopen,
    .dlsym = R.dlsym,
    .dlclose = R.dlclose,
    .handleBase = R.handleBase,
    .addrBase = R.addrBase,
    .objects = R.objects,
    .counts = R.counts,
    .readSmall = R.readSmall,
};

const R = struct {
    fn envPresent(_: ?*anyopaque, name: [*:0]const u8) bool {
        return std.c.getenv(name) != null;
    }

    fn open(_: ?*anyopaque, path: [*:0]const u8, follow: bool) ?Fd {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = !follow });
        return if (fd < 0) null else fd;
    }

    fn stat(_: ?*anyopaque, fd: Fd) ?Stat {
        var b: linux.Statx = undefined;
        const mask: linux.STATX = .{ .TYPE = true, .MODE = true, .INO = true, .SIZE = true };
        if (std.c.statx(fd, "", linux.AT.EMPTY_PATH, mask, &b) != 0) return null;
        return .{
            .dev_major = b.dev_major,
            .dev_minor = b.dev_minor,
            .ino = b.ino,
            .size = b.size,
            .mode = b.mode,
            .verity = b.attributes_mask.VERITY and b.attributes.VERITY,
        };
    }

    fn pread(_: ?*anyopaque, fd: Fd, buf: []u8, offset: u64) ?usize {
        var got: usize = 0;
        while (got < buf.len) {
            const n = std.c.pread(fd, buf[got..].ptr, buf.len - got, @intCast(offset + got));
            if (n < 0) return null;
            if (n == 0) break;
            got += @intCast(n);
        }
        return got;
    }

    fn memfd(_: ?*anyopaque, name: [*:0]const u8) ?Fd {
        const mfd_exec = 0x0010; // the bytes are loaded as code; refused (not retried) by a kernel without it
        const fd = std.c.memfd_create(name, linux.MFD.CLOEXEC | linux.MFD.ALLOW_SEALING | mfd_exec);
        return if (fd < 0) null else fd;
    }

    fn writeAll(_: ?*anyopaque, fd: Fd, bytes: []const u8) bool {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
            if (n <= 0) return false;
            done += @intCast(n);
        }
        return true;
    }

    fn addSeals(_: ?*anyopaque, fd: Fd, seals: u32) bool {
        return std.c.fcntl(fd, linux.F.ADD_SEALS, @as(c_int, @intCast(seals))) == 0;
    }

    fn getSeals(_: ?*anyopaque, fd: Fd) ?u32 {
        const s = std.c.fcntl(fd, linux.F.GET_SEALS);
        return if (s < 0) null else @intCast(s);
    }

    fn measureVerity(_: ?*anyopaque, fd: Fd) ?[32]u8 {
        // struct fsverity_digest { __u16 digest_algorithm; __u16 digest_size; __u8 digest[]; }
        var d: extern struct { alg: u16, size: u16, digest: [64]u8 } = .{ .alg = 0, .size = 64, .digest = undefined };
        if (std.c.ioctl(fd, @bitCast(fs_ioc_measure_verity), &d) != 0) return null;
        if (d.alg != fsverity_hash_alg_sha256 or d.size != 32) return null;
        return d.digest[0..32].*;
    }

    fn mapRead(_: ?*anyopaque, fd: Fd, len: usize) ?[]const u8 {
        const p = std.c.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
        if (p == std.c.MAP_FAILED) return null;
        return @as([*]const u8, @ptrCast(p))[0..len];
    }

    fn unmap(_: ?*anyopaque, bytes: []const u8) void {
        _ = std.c.munmap(@ptrCast(@alignCast(bytes.ptr)), bytes.len);
    }

    fn close(_: ?*anyopaque, fd: Fd) void {
        _ = std.c.close(fd);
    }

    fn pageSize(_: ?*anyopaque) usize {
        return std.heap.pageSize();
    }

    fn dlopen(_: ?*anyopaque, path: [*:0]const u8) ?*anyopaque {
        return std.c.dlopen(path, .{ .NOW = true, .GLOBAL = true });
    }

    fn dlsym(_: ?*anyopaque, handle: *anyopaque, name: [*:0]const u8) ?*anyopaque {
        return @call(.never_tail, std.c.dlsym, .{ handle, name });
    }

    fn dlclose(_: ?*anyopaque, handle: *anyopaque) c_int {
        return std.c.dlclose(handle);
    }

    fn handleBase(_: ?*anyopaque, handle: *anyopaque) ?usize {
        var lm: ?*LinkMap = null;
        if (dlinfo(handle, rtld_di_linkmap, @ptrCast(&lm)) != 0) return null;
        return if (lm) |m| m.l_addr else null;
    }

    fn addrBase(_: ?*anyopaque, addr: usize) ?usize {
        var info: Dl_info = undefined;
        var extra: ?*anyopaque = null;
        if (dladdr1(@ptrFromInt(addr), &info, &extra, rtld_dl_linkmap) == 0) return null;
        const lm: *const LinkMap = @ptrCast(@alignCast(extra orelse return null));
        return lm.l_addr;
    }

    const Collect = struct {
        gpa: std.mem.Allocator,
        list: std.ArrayList(Raw) = .empty,
        failed: bool = false,
        const Raw = struct { path: []u8, base: usize, phdr: usize };
    };

    fn collect(info: *std.c.dl_phdr_info, _: usize, data: ?*anyopaque) callconv(.c) c_int {
        const c: *Collect = @ptrCast(@alignCast(data.?));
        const name = if (info.name) |n| std.mem.span(n) else "";
        const path = c.gpa.dupe(u8, name) catch {
            c.failed = true;
            return 1;
        };
        c.list.append(c.gpa, .{ .path = path, .base = info.addr, .phdr = @intFromPtr(info.phdr) }) catch {
            c.gpa.free(path);
            c.failed = true;
            return 1;
        };
        return 0;
    }

    fn objects(_: ?*anyopaque, gpa: std.mem.Allocator) ObjectsError![]Object {
        var c: Collect = .{ .gpa = gpa };
        defer c.list.deinit(gpa);
        _ = std.c.dl_iterate_phdr(collect, &c);
        if (c.failed) {
            for (c.list.items) |r| gpa.free(r.path);
            return error.OutOfMemory;
        }
        const maps = readAll(gpa, "/proc/self/maps") orelse {
            for (c.list.items) |r| gpa.free(r.path);
            return error.Unavailable;
        };
        defer gpa.free(maps);
        const exe = exeIdentity();

        const out = gpa.alloc(Object, c.list.items.len) catch {
            for (c.list.items) |r| gpa.free(r.path);
            return error.OutOfMemory;
        };
        for (c.list.items, out) |r, *o| {
            const m = mapping(maps, r.phdr);
            const kind: Kind = if (m.vdso)
                .vdso
            else if (exe != null and m.ino != 0 and m.ino == exe.?.ino and m.dev_major == exe.?.dev_major and m.dev_minor == exe.?.dev_minor)
                .main
            else
                .library;
            o.* = .{ .path = r.path, .base = r.base, .dev_major = m.dev_major, .dev_minor = m.dev_minor, .ino = m.ino, .kind = kind };
        }
        return out;
    }

    fn exeIdentity() ?Stat {
        const fd = open(null, "/proc/self/exe", true) orelse return null;
        defer close(null, fd);
        return stat(null, fd);
    }

    const Mapping = struct { dev_major: u32 = 0, dev_minor: u32 = 0, ino: u64 = 0, vdso: bool = false };

    /// The /proc/self/maps line whose range holds `addr`: "start-end perms offset major:minor inode path".
    fn mapping(maps: []const u8, addr: usize) Mapping {
        var lines = std.mem.tokenizeScalar(u8, maps, '\n');
        while (lines.next()) |line| {
            var f = std.mem.tokenizeScalar(u8, line, ' ');
            const range = f.next() orelse continue;
            const dash = std.mem.indexOfScalar(u8, range, '-') orelse continue;
            const lo = std.fmt.parseInt(usize, range[0..dash], 16) catch continue;
            const hi = std.fmt.parseInt(usize, range[dash + 1 ..], 16) catch continue;
            if (addr < lo or addr >= hi) continue;
            _ = f.next(); // perms
            _ = f.next(); // offset
            const dev = f.next() orelse return .{};
            const ino = f.next() orelse return .{};
            const path = f.rest();
            const colon = std.mem.indexOfScalar(u8, dev, ':') orelse return .{};
            return .{
                .dev_major = std.fmt.parseInt(u32, dev[0..colon], 16) catch 0,
                .dev_minor = std.fmt.parseInt(u32, dev[colon + 1 ..], 16) catch 0,
                .ino = std.fmt.parseInt(u64, ino, 10) catch 0,
                .vdso = std.mem.eql(u8, path, "[vdso]"),
            };
        }
        return .{};
    }

    fn readAll(gpa: std.mem.Allocator, path: [*:0]const u8) ?[]u8 {
        const fd = open(null, path, true) orelse return null;
        defer close(null, fd);
        var list: std.ArrayList(u8) = .empty;
        while (true) {
            list.ensureUnusedCapacity(gpa, 64 * 1024) catch {
                list.deinit(gpa);
                return null;
            };
            const n = std.c.read(fd, list.unusedCapacitySlice().ptr, list.unusedCapacitySlice().len);
            if (n < 0) {
                list.deinit(gpa);
                return null;
            }
            if (n == 0) break;
            list.items.len += @intCast(n);
        }
        return list.toOwnedSlice(gpa) catch {
            list.deinit(gpa);
            return null;
        };
    }

    fn countsCb(info: *std.c.dl_phdr_info, size: usize, data: ?*anyopaque) callconv(.c) c_int {
        const out: *?Counts = @ptrCast(@alignCast(data.?));
        if (size < @offsetOf(PhdrInfo, "subs") + @sizeOf(c_ulonglong)) return 1;
        const full: *const PhdrInfo = @ptrCast(@alignCast(info));
        out.* = .{ .adds = full.adds, .subs = full.subs };
        return 1; // the counters are the same in every entry
    }

    fn counts(_: ?*anyopaque) ?Counts {
        var out: ?Counts = null;
        _ = std.c.dl_iterate_phdr(countsCb, &out);
        return out;
    }

    fn readSmall(_: ?*anyopaque, path: [*:0]const u8, buf: []u8) ?[]const u8 {
        const fd = open(null, path, true) orelse return null;
        defer close(null, fd);
        const n = std.c.read(fd, buf.ptr, buf.len);
        if (n <= 0 or n == buf.len) return null; // empty, failed, or possibly cut short
        return std.mem.trimEnd(u8, buf[0..@intCast(n)], "\n");
    }
};

test "maps line lookup" {
    const maps =
        \\aaaa0000-aaaa1000 r--p 00000000 fd:01 1234 /usr/lib/libc.so.6
        \\ffff0000-ffff2000 r-xp 00000000 00:00 0 [vdso]
        \\bbbb0000-bbbb4000 r-xp 00000000 00:01 99 /memfd:libcudnn.so.9 (deleted)
    ;
    const a = R.mapping(maps, 0xaaaa0040);
    try std.testing.expectEqual(@as(u32, 0xfd), a.dev_major);
    try std.testing.expectEqual(@as(u32, 1), a.dev_minor);
    try std.testing.expectEqual(@as(u64, 1234), a.ino);
    try std.testing.expect(R.mapping(maps, 0xffff0100).vdso);
    try std.testing.expectEqual(@as(u64, 99), R.mapping(maps, 0xbbbb0000).ino);
    try std.testing.expectEqual(@as(u64, 0), R.mapping(maps, 0x1).ino);
}
