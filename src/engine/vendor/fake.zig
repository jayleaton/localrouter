//! A fake OS behind `Sys` for the vendor module's tests: files, memfds with seals, a loader that maps memfds at fixed
//! bases, and the loader's counters, all in memory.

const std = @import("std");
const sys_mod = @import("sys.zig");
const Sys = sys_mod.Sys;
const Fd = sys_mod.Fd;
const Counts = sys_mod.Counts;
const linux = std.os.linux;

pub const Fake = struct {
    files: []const File,
    env: []const []const u8 = &.{},
    preexisting: []const sys_mod.Object = &.{},
    open_fds: [32]?Open = @splat(null),
    mem: [32][4096]u8 = undefined,
    loaded: [16]sys_mod.Object = undefined,
    n_loaded: usize = 0,
    extra: ?sys_mod.Object = null,
    adds: u64 = 10,
    subs: u64 = 0,
    memfds: usize = 0,
    drop_seal: bool = false,
    dlclosed: usize = 0,
    /// dlclose fails (rc 1) for the n-th call when set.
    dlclose_fail_at: ?usize = null,
    /// dlopen fails when this many objects are already loaded.
    dlopen_fail_at: ?usize = null,
    /// When set, every code address belongs to the object at this base, except `foreign_addr`.
    fixed_base: ?usize = null,
    foreign_addr: usize = 0,

    pub const File = struct { path: []const u8, bytes: []const u8, ino: u64, verity: ?[32]u8 = null };
    const Open = struct { file: ?*const File, len: usize = 0, seals: u32 = 0 };

    fn of(ctx: ?*anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx.?));
    }

    pub fn sys(self: *Fake) Sys {
        return .{
            .ctx = self,
            .envPresent = envPresent,
            .open = open,
            .stat = stat,
            .pread = pread,
            .memfd = memfd,
            .writeAll = writeAll,
            .addSeals = addSeals,
            .getSeals = getSeals,
            .measureVerity = measureVerity,
            .mapRead = mapRead,
            .unmap = unmap,
            .close = close,
            .pageSize = pageSize,
            .dlopen = dlopen,
            .dlsym = dlsym,
            .dlclose = dlclose,
            .handleBase = handleBase,
            .addrBase = addrBase,
            .objects = objects,
            .counts = counts,
            .readSmall = readSmall,
        };
    }

    fn slot(self: *Fake, o: Open) ?Fd {
        for (&self.open_fds, 0..) |*s, i| if (s.* == null) {
            s.* = o;
            return @intCast(i + 3);
        };
        return null;
    }
    fn get(self: *Fake, fd: Fd) *Open {
        return &self.open_fds[@intCast(fd - 3)].?;
    }
    fn bytes(self: *Fake, fd: Fd) []const u8 {
        const o = self.get(fd);
        return if (o.file) |f| f.bytes else self.mem[@intCast(fd - 3)][0..o.len];
    }

    fn envPresent(ctx: ?*anyopaque, name: [*:0]const u8) bool {
        for (of(ctx).env) |e| if (std.mem.eql(u8, e, std.mem.span(name))) return true;
        return false;
    }
    fn open(ctx: ?*anyopaque, path: [*:0]const u8, _: bool) ?Fd {
        const self = of(ctx);
        for (self.files) |*f| if (std.mem.eql(u8, f.path, std.mem.span(path))) return self.slot(.{ .file = f });
        return null;
    }
    fn stat(ctx: ?*anyopaque, fd: Fd) ?sys_mod.Stat {
        const self = of(ctx);
        const o = self.get(fd);
        const ino = if (o.file) |f| f.ino else 1000 + @as(u64, @intCast(fd));
        const verity = if (o.file) |f| f.verity != null else false;
        return .{ .dev_major = if (o.file != null) 8 else 0, .dev_minor = 1, .ino = ino, .size = self.bytes(fd).len, .mode = linux.S.IFREG | 0o444, .verity = verity };
    }
    fn pread(ctx: ?*anyopaque, fd: Fd, buf: []u8, off: u64) ?usize {
        const b = of(ctx).bytes(fd);
        if (off >= b.len) return 0;
        const n = @min(buf.len, b.len - off);
        @memcpy(buf[0..n], b[@intCast(off)..][0..n]);
        return n;
    }
    fn memfd(ctx: ?*anyopaque, _: [*:0]const u8) ?Fd {
        const self = of(ctx);
        self.memfds += 1;
        return self.slot(.{ .file = null });
    }
    fn writeAll(ctx: ?*anyopaque, fd: Fd, b: []const u8) bool {
        const self = of(ctx);
        const o = self.get(fd);
        if (o.seals & linux.F.SEAL_WRITE != 0) return false;
        @memcpy(self.mem[@intCast(fd - 3)][o.len..][0..b.len], b);
        o.len += b.len;
        return true;
    }
    fn addSeals(ctx: ?*anyopaque, fd: Fd, s: u32) bool {
        const self = of(ctx);
        self.get(fd).seals |= if (self.drop_seal) s & ~@as(u32, linux.F.SEAL_WRITE) else s;
        return true;
    }
    fn getSeals(ctx: ?*anyopaque, fd: Fd) ?u32 {
        return of(ctx).get(fd).seals;
    }
    fn measureVerity(ctx: ?*anyopaque, fd: Fd) ?[32]u8 {
        const o = of(ctx).get(fd);
        return if (o.file) |f| f.verity else null;
    }
    fn mapRead(ctx: ?*anyopaque, fd: Fd, len: usize) ?[]const u8 {
        const b = of(ctx).bytes(fd);
        return if (len <= b.len) b[0..len] else null;
    }
    fn unmap(_: ?*anyopaque, _: []const u8) void {}
    fn close(ctx: ?*anyopaque, fd: Fd) void {
        of(ctx).open_fds[@intCast(fd - 3)] = null;
    }
    fn pageSize(_: ?*anyopaque) usize {
        return 4096;
    }
    fn dlopen(ctx: ?*anyopaque, path: [*:0]const u8) ?*anyopaque {
        const self = of(ctx);
        const p = std.mem.span(path);
        const fd = std.fmt.parseInt(Fd, p["/proc/self/fd/".len..], 10) catch return null;
        if (self.dlopen_fail_at) |n| if (n == self.n_loaded) return null;
        const st = stat(ctx, fd).?;
        const base = 0x100000 * (self.n_loaded + 1);
        self.loaded[self.n_loaded] = .{ .path = p, .base = base, .dev_major = st.dev_major, .dev_minor = st.dev_minor, .ino = st.ino, .kind = .library };
        self.n_loaded += 1;
        self.adds += 1;
        return @ptrFromInt(base);
    }
    fn dlsym(_: ?*anyopaque, handle: *anyopaque, name: [*:0]const u8) ?*anyopaque {
        const n = std.mem.span(name);
        if (std.mem.eql(u8, n, "missing_fn")) return null;
        if (std.mem.eql(u8, n, "foreign_fn")) return @ptrFromInt(0x7700000 + 16);
        return @ptrFromInt(@intFromPtr(handle) + 16);
    }
    fn dlclose(ctx: ?*anyopaque, _: *anyopaque) c_int {
        const self = of(ctx);
        if (self.dlclose_fail_at) |n| if (n == self.dlclosed) return 1;
        self.dlclosed += 1;
        return 0;
    }
    fn handleBase(ctx: ?*anyopaque, handle: *anyopaque) ?usize {
        return of(ctx).fixed_base orelse @intFromPtr(handle);
    }
    fn addrBase(ctx: ?*anyopaque, addr: usize) ?usize {
        const self = of(ctx);
        if (self.fixed_base) |b| return if (addr == self.foreign_addr) b + 0x100000 else b;
        return addr & ~@as(usize, 0xfffff);
    }
    fn objects(ctx: ?*anyopaque, gpa: std.mem.Allocator) sys_mod.ObjectsError![]sys_mod.Object {
        const self = of(ctx);
        var list: std.ArrayList(sys_mod.Object) = .empty;
        errdefer {
            for (list.items) |o| gpa.free(o.path);
            list.deinit(gpa);
        }
        const extra: []const sys_mod.Object = if (self.extra) |*e| e[0..1] else &.{};
        for ([_][]const sys_mod.Object{ self.preexisting, self.loaded[0..self.n_loaded], extra }) |group| for (group) |o| {
            var c = o;
            c.path = try gpa.dupe(u8, o.path);
            list.append(gpa, c) catch |e| {
                gpa.free(c.path);
                return e;
            };
        };
        return list.toOwnedSlice(gpa);
    }
    fn counts(ctx: ?*anyopaque) ?Counts {
        const self = of(ctx);
        return .{ .adds = self.adds, .subs = self.subs };
    }
    fn readSmall(ctx: ?*anyopaque, path: [*:0]const u8, _: []u8) ?[]const u8 {
        for (of(ctx).files) |f| if (std.mem.eql(u8, f.path, std.mem.span(path))) return f.bytes;
        return null;
    }
};
