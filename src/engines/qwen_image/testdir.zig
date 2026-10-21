//! A test's own scratch directory, unique per run (a random suffix), so what another user or an earlier run left at a
//! fixed path (a root-owned directory, say) or a parallel run of the same tests can never collide with it.

const std = @import("std");
const Io = std.Io;

/// Creates `<base>/<name>-<16 hex digits>` and returns its path, allocated with `a` (an arena: nothing to free).
pub fn make(io: Io, a: std.mem.Allocator, base: []const u8, name: []const u8) ![:0]u8 {
    var rnd: [8]u8 = undefined;
    io.random(&rnd);
    const dir = try std.fmt.allocPrintSentinel(a, "{s}/{s}-{x:0>16}", .{ base, name, std.mem.readInt(u64, &rnd, .little) }, 0);
    try Io.Dir.cwd().createDirPath(io, dir);
    return dir;
}

/// `dir/name` as a 0-terminated path.
pub fn file(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
}
