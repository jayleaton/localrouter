//! What the tests read about other processes: children, name, command line, resident size. /proc on Linux; libproc
//! and sysctl on macOS, which has no /proc.

const std = @import("std");
const darwin = @import("builtin").os.tag.isDarwin();

/// The pids whose parent is `ppid`.
pub fn children(a: std.mem.Allocator, io: std.Io, ppid: i32) ![]i32 {
    var list: std.ArrayList(i32) = .empty;
    if (darwin) {
        var pids: [512]i32 = undefined;
        const n = proc_listchildpids(ppid, &pids, @sizeOf(@TypeOf(pids)));
        if (n > 0) try list.appendSlice(a, pids[0..@intCast(n)]);
        return list.items;
    }
    var proc = try std.Io.Dir.cwd().openDir(io, "/proc", .{ .iterate = true });
    defer proc.close(io);
    var it = proc.iterate();
    while (try it.next(io)) |e| {
        const pid = std.fmt.parseInt(i32, e.name, 10) catch continue;
        var buf: [4096]u8 = undefined;
        const status = read(try std.fmt.allocPrintSentinel(a, "/proc/{d}/status", .{pid}, 0), &buf) orelse continue;
        if ((field(status, "PPid:") orelse continue) == ppid) try list.append(a, pid);
    }
    return list.items;
}

/// The process's name (its executable's basename, as `Name:` in /proc/<pid>/status), "" when it is gone.
pub fn name(a: std.mem.Allocator, pid: i32, buf: []u8) []const u8 {
    if (darwin) {
        const n = proc_name(pid, buf.ptr, @intCast(buf.len));
        return if (n > 0) buf[0..@intCast(n)] else "";
    }
    const status = read(std.fmt.allocPrintSentinel(a, "/proc/{d}/status", .{pid}, 0) catch return "", buf) orelse return "";
    const at = std.mem.indexOf(u8, status, "Name:\t") orelse return "";
    const end = std.mem.indexOfScalarPos(u8, status, at, '\n') orelse status.len;
    return status[at + 6 .. end];
}

/// The process's arguments, NUL-separated (macOS: preceded by its executable path), "" when it is gone.
pub fn cmdline(a: std.mem.Allocator, pid: i32, buf: []u8) []const u8 {
    if (darwin) {
        var mib = [3]c_int{ 1, 49, pid }; // CTL_KERN, KERN_PROCARGS2
        var len: usize = buf.len;
        if (std.c.sysctl(&mib, mib.len, buf.ptr, &len, null, 0) != 0 or len < 4) return "";
        return buf[4..len]; // after argc
    }
    return read(std.fmt.allocPrintSentinel(a, "/proc/{d}/cmdline", .{pid}, 0) catch return "", buf) orelse "";
}

/// The process's resident bytes, null when it is gone.
pub fn rss(a: std.mem.Allocator, pid: i32) ?u64 {
    if (darwin) {
        var info: [128]u8 align(8) = undefined; // struct proc_taskinfo (96 bytes): pti_resident_size at offset 8
        if (proc_pidinfo(pid, 4, 0, &info, info.len) <= 0) return null; // PROC_PIDTASKINFO
        return std.mem.readInt(u64, info[8..16], .little);
    }
    var buf: [4096]u8 = undefined;
    const status = read(std.fmt.allocPrintSentinel(a, "/proc/{d}/status", .{pid}, 0) catch return null, &buf) orelse return null;
    const kb = field(status, "VmRSS:") orelse return null;
    return @as(u64, @intCast(kb)) * 1024;
}

/// A /proc file (they report size 0, so read until EOF with plain reads).
pub fn read(path: [:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, path, .{}, 0) catch return null;
    defer _ = std.posix.system.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const got = std.posix.read(fd, buf[n..]) catch return null;
        if (got == 0) break;
        n += got;
    }
    return buf[0..n];
}

/// The number after `key` on its line ("PPid:\t123", "VmRSS:   456 kB").
fn field(text: []const u8, key: []const u8) ?i64 {
    const at = std.mem.indexOf(u8, text, key) orelse return null;
    var it = std.mem.tokenizeAny(u8, text[at + key.len ..], " \t\n");
    return std.fmt.parseInt(i64, it.next() orelse return null, 10) catch null;
}

/// Whether an exclusive flock on `fd` can be taken now (it is released again).
pub fn lockFree(fd: c_int) bool {
    if (flock(fd, std.posix.LOCK.EX | std.posix.LOCK.NB) != 0) return false;
    _ = flock(fd, std.posix.LOCK.UN);
    return true;
}

extern "c" fn flock(fd: c_int, op: c_int) c_int;
extern "c" fn proc_listchildpids(ppid: i32, buffer: *anyopaque, size: c_int) c_int;
extern "c" fn proc_name(pid: i32, buffer: [*]u8, size: u32) c_int;
extern "c" fn proc_pidinfo(pid: i32, flavor: c_int, arg: u64, buffer: *anyopaque, size: c_int) c_int;
