//! Size-based rotation for a tool's log (`<data>/logs/<id>.log`). When the live file is already at `max_bytes`, the
//! next append renames it to `<path>.1` (and `.1` to `.2`, up to `keep`) and starts a new file. `max_bytes` 0 does
//! not rotate. A single append may leave the live file one chunk over the limit; the following one rotates, so the
//! newest lines stay in the live file (what a failure's log tail reads).

const std = @import("std");
const posix = std.posix;
const Io = std.Io;

pub const Policy = struct {
    max_bytes: u64 = 8 << 20,
    keep: u32 = 3,
};

pub fn append(io: Io, path: []const u8, policy: Policy, chunk: []const u8) !void {
    if (chunk.len == 0) return;
    if (std.fs.path.dirname(path)) |d| Io.Dir.cwd().createDirPath(io, d) catch {};
    if (policy.max_bytes > 0 and fileLen(io, path) >= policy.max_bytes) try rotate(io, path, policy.keep);
    const file = try Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
    defer file.close(io);
    _ = posix.system.lseek(file.handle, 0, posix.SEEK.END);
    try file.writeStreamingAll(io, chunk);
}

pub fn rotate(io: Io, path: []const u8, keep: u32) !void {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (keep == 0) {
        Io.Dir.cwd().deleteFile(io, path) catch {};
        return;
    }
    const drop = try std.fmt.allocPrint(a, "{s}.{d}", .{ path, keep });
    Io.Dir.cwd().deleteFile(io, drop) catch {};
    var i: u32 = keep;
    while (i > 1) : (i -= 1) {
        const from = try std.fmt.allocPrint(a, "{s}.{d}", .{ path, i - 1 });
        const to = try std.fmt.allocPrint(a, "{s}.{d}", .{ path, i });
        Io.Dir.cwd().rename(from, Io.Dir.cwd(), to, io) catch {};
    }
    const first = try std.fmt.allocPrint(a, "{s}.1", .{path});
    Io.Dir.cwd().rename(path, Io.Dir.cwd(), first, io) catch {};
}

fn fileLen(io: Io, path: []const u8) u64 {
    const file = Io.Dir.cwd().openFile(io, path, .{}) catch return 0;
    defer file.close(io);
    const end = posix.system.lseek(file.handle, 0, posix.SEEK.END);
    if (end <= 0) return 0;
    return @intCast(end);
}

// ---------------------------------------------------------------- tests

const t = std.testing;

test "a log rotates by size and keeps N older files" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const path = try std.fmt.allocPrint(t.allocator, "{s}/tool.log", .{dir});
    defer t.allocator.free(path);
    const policy: Policy = .{ .max_bytes = 100, .keep = 2 };
    var chunk: [150]u8 = undefined;
    @memset(&chunk, 'a');
    try append(io, path, policy, &chunk);
    try t.expectEqual(@as(u64, 150), fileLen(io, path));
    try append(io, path, policy, &chunk); // the first file was over the limit: it becomes .1
    try t.expectEqual(@as(u64, 150), fileLen(io, path));
    const older = try std.fmt.allocPrint(t.allocator, "{s}.1", .{path});
    defer t.allocator.free(older);
    try t.expectEqual(@as(u64, 150), fileLen(io, older));
    try append(io, path, policy, &chunk);
    const oldest = try std.fmt.allocPrint(t.allocator, "{s}.2", .{path});
    defer t.allocator.free(oldest);
    try t.expectEqual(@as(u64, 150), fileLen(io, oldest));
    const gone = try std.fmt.allocPrint(t.allocator, "{s}.3", .{path});
    defer t.allocator.free(gone);
    try t.expectEqual(@as(u64, 0), fileLen(io, gone));
    // 0 keeps nothing: the next over-limit append drops the live file instead of renaming it
    try append(io, path, .{ .max_bytes = 100, .keep = 0 }, "z");
    try t.expectEqual(@as(u64, 1), fileLen(io, path));
}
