//! Free memory as the kernel reports it. On GB10 device allocations come out of system memory, so MemAvailable is
//! the budget the GPU tools and any other service on the machine share.

const std = @import("std");
const posix = std.posix;

/// MemAvailable from /proc/meminfo in bytes; null when it cannot be read.
pub fn available() ?u64 {
    var buf: [4096]u8 = undefined;
    const fd = posix.openatZ(posix.AT.FDCWD, "/proc/meminfo", .{}, 0) catch return null;
    defer _ = posix.system.close(fd);
    const n = posix.read(fd, &buf) catch return null;
    return parseAvailable(buf[0..n]);
}

/// Parses the "MemAvailable:  123 kB" line.
pub fn parseAvailable(text: []const u8) ?u64 {
    const key = "MemAvailable:";
    const at = std.mem.indexOf(u8, text, key) orelse return null;
    const rest = std.mem.trimStart(u8, text[at + key.len ..], " \t");
    const end = std.mem.indexOfAny(u8, rest, " \n") orelse rest.len;
    const kb = std.fmt.parseInt(u64, rest[0..end], 10) catch return null;
    return kb * 1024;
}

test "parse meminfo and read the real one" {
    try std.testing.expectEqual(@as(?u64, 4096 * 1024), parseAvailable("MemTotal: 9 kB\nMemAvailable:     4096 kB\n"));
    try std.testing.expect(parseAvailable("MemTotal: 9 kB\n") == null);
    try std.testing.expect(available().? > 0);
}
