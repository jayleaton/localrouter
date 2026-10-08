//! The worker protocol: one JSON object a line each way over the worker's stdin and stdout. Files carry the bytes.
//!   daemon -> worker  {"op":"load"} | {"op":"generate","id","dir","request"} | {"op":"exit"}
//!   worker -> daemon  {"progress":{...}}* then {"ok":true,...} or {"ok":false,"error","type"}

const std = @import("std");
const posix = std.posix;
const Request = @import("request.zig").Request;
const Progress = @import("tool.zig").Progress;

pub const Op = enum { load, generate, exit };

pub const Command = struct {
    op: Op,
    id: []const u8 = "",
    dir: []const u8 = "",
    request: ?Request = null,
};

/// Every worker line: a progress note, or the final reply of the current command.
pub const Reply = struct {
    ok: ?bool = null,
    progress: ?Progress = null,
    resident: u64 = 0,
    files: []const []const u8 = &.{},
    seed: u64 = 0,
    ms: u64 = 0,
    @"error": []const u8 = "",
    type: []const u8 = "",
};

pub const max_line = 1 << 20;

/// Serializes `v` as one line and writes it whole to `fd`.
pub fn send(fd: posix.fd_t, gpa: std.mem.Allocator, v: anytype) !void {
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try std.json.Stringify.value(v, .{ .emit_null_optional_fields = false }, &out.writer);
    try out.writer.writeByte('\n');
    try writeAll(fd, out.written());
}

pub fn writeAll(fd: posix.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const rc = posix.system.write(fd, bytes[done..].ptr, bytes.len - done);
        switch (posix.errno(rc)) {
            .SUCCESS => done += @intCast(rc),
            .INTR, .AGAIN => continue,
            else => return error.BrokenPipe,
        }
    }
}

/// Buffered line reads from a pipe, each with an optional deadline (poll, so a hung peer cannot block us).
pub const LineReader = struct {
    fd: posix.fd_t,
    buf: []u8,
    start: usize = 0,
    end: usize = 0,

    pub const Error = error{ EndOfStream, Timeout, LineTooLong, ReadFailed };

    /// The next line without its newline, valid until the next call. `timeout_ms` < 0 waits forever.
    pub fn next(r: *LineReader, timeout_ms: i32) Error![]const u8 {
        var scanned: usize = r.start;
        while (true) {
            if (std.mem.indexOfScalarPos(u8, r.buf[0..r.end], scanned, '\n')) |nl| {
                const line = r.buf[r.start..nl];
                r.start = nl + 1;
                return line;
            }
            scanned = r.end;
            if (r.start > 0) { // compact so the line can grow
                std.mem.copyForwards(u8, r.buf, r.buf[r.start..r.end]);
                r.end -= r.start;
                scanned -= r.start;
                r.start = 0;
            }
            if (r.end == r.buf.len) return error.LineTooLong;
            var fds = [_]posix.pollfd{.{ .fd = r.fd, .events = posix.POLL.IN, .revents = 0 }};
            const ready = posix.poll(&fds, timeout_ms) catch return error.ReadFailed;
            if (ready == 0) return error.Timeout;
            const rc = posix.system.read(r.fd, r.buf[r.end..].ptr, r.buf.len - r.end);
            switch (posix.errno(rc)) {
                .SUCCESS => {
                    if (rc == 0) return error.EndOfStream;
                    r.end += @intCast(rc);
                },
                .INTR, .AGAIN => {},
                else => return error.ReadFailed,
            }
        }
    }
};

test "lines round-trip through a pipe" {
    const gpa = std.testing.allocator;
    var fds: [2]posix.fd_t = undefined;
    try std.testing.expectEqual(@as(usize, 0), @as(usize, @bitCast(@as(isize, posix.system.pipe(&fds)))));
    defer _ = posix.system.close(fds[0]);
    {
        defer _ = posix.system.close(fds[1]);
        try send(fds[1], gpa, Command{ .op = .generate, .id = "img_1", .dir = "/tmp/j", .request = .{ .image = .{ .prompt = "fox", .seed = 7 } } });
        try send(fds[1], gpa, Reply{ .progress = .{ .phase = "denoise", .step = 3, .of = 25 } });
    }
    var buf: [4096]u8 = undefined;
    var r: LineReader = .{ .fd = fds[0], .buf = &buf };
    const a = try std.json.parseFromSlice(Command, gpa, try r.next(1000), .{});
    defer a.deinit();
    try std.testing.expectEqual(@as(u64, 7), a.value.request.?.image.seed);
    const b = try std.json.parseFromSlice(Reply, gpa, try r.next(1000), .{});
    defer b.deinit();
    try std.testing.expectEqual(@as(u32, 25), b.value.progress.?.of);
    try std.testing.expect(b.value.ok == null);
    try std.testing.expectError(error.EndOfStream, r.next(1000));
}
