//! The computer-control audit log (docs/dev/COMPUTER.md): one JSON object a line in `<data>/computer/audit.jsonl`,
//! appended under a lock, for every task, grant, approval, action, refusal and outcome. The user's screen text stays
//! out of it: typed text is its length and a SHA-256 prefix, labels and titles are clipped to 80 characters. The file
//! is rotated to its newest half past 8 MiB.

const std = @import("std");
const Io = std.Io;

const max_bytes: u64 = 8 << 20;
pub const clip_len = 80;

pub const Log = struct {
    io: Io,
    path: []const u8, // "": nothing is written (tests, or no data directory)
    mutex: Io.Mutex = .init,

    /// Appends `{"ts": <unix ms>, "event": event, ...fields}` as one line; best effort (a full disk loses lines, never
    /// a task).
    pub fn record(l: *Log, gpa: std.mem.Allocator, event: []const u8, fields: anytype) void {
        if (l.path.len == 0) return;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const line = format(arena.allocator(), Io.Clock.real.now(l.io).toMilliseconds(), event, fields) catch return;
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        l.append(line) catch |err| std.log.warn("computer audit: {s}", .{@errorName(err)});
    }

    fn append(l: *Log, line: []const u8) !void {
        const cwd = Io.Dir.cwd();
        if (std.fs.path.dirname(l.path)) |d| try cwd.createDirPath(l.io, d);
        const file = try cwd.createFile(l.io, l.path, .{ .truncate = false });
        defer file.close(l.io);
        const end = std.posix.system.lseek(file.handle, 0, std.posix.SEEK.END);
        if (end > 0 and @as(u64, @intCast(end)) > max_bytes) return l.rotate(line);
        try file.writeStreamingAll(l.io, line);
    }

    /// Past `max_bytes`: the file keeps its newest half (whole lines), then the new line.
    fn rotate(l: *Log, line: []const u8) !void {
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const data = try Io.Dir.cwd().readFileAlloc(l.io, l.path, a, .limited(max_bytes + (4 << 20)));
        var keep = data[data.len / 2 ..];
        if (std.mem.indexOfScalar(u8, keep, '\n')) |i| keep = keep[i + 1 ..];
        try Io.Dir.cwd().writeFile(l.io, .{ .sub_path = l.path, .data = try std.mem.concat(a, u8, &.{ keep, line }) });
    }

    /// The newest `limit` lines, oldest first, as raw JSON texts in `a`.
    pub fn tail(l: *Log, a: std.mem.Allocator, limit: usize) ![]const []const u8 {
        if (l.path.len == 0) return &.{};
        l.mutex.lockUncancelable(l.io);
        defer l.mutex.unlock(l.io);
        const data = Io.Dir.cwd().readFileAlloc(l.io, l.path, a, .limited(max_bytes + (1 << 20))) catch |err| switch (err) {
            error.FileNotFound => return &.{},
            else => return err,
        };
        return lastLines(a, data, limit);
    }
};

/// One audit line (with its newline).
pub fn format(a: std.mem.Allocator, ts_ms: i64, event: []const u8, fields: anytype) ![]const u8 {
    var w: Io.Writer.Allocating = .init(a);
    try w.writer.print("{{\"ts\":{d},\"event\":", .{ts_ms});
    try std.json.Stringify.value(event, .{}, &w.writer);
    const body = try std.json.Stringify.valueAlloc(a, fields, .{ .emit_null_optional_fields = false });
    if (body.len > 2) { // `{...}`: its members after ours
        try w.writer.writeByte(',');
        try w.writer.writeAll(body[1 .. body.len - 1]);
    }
    try w.writer.writeAll("}\n");
    return w.written();
}

fn lastLines(a: std.mem.Allocator, data: []const u8, limit: usize) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var end = data.len;
    while (end > 0 and out.items.len < limit) {
        while (end > 0 and data[end - 1] == '\n') end -= 1;
        if (end == 0) break;
        const start = if (std.mem.lastIndexOfScalar(u8, data[0..end], '\n')) |i| i + 1 else 0;
        try out.append(a, data[start..end]);
        end = start;
    }
    std.mem.reverse([]const u8, out.items);
    return out.items;
}

/// Typed text as the log keeps it: its length and the first 12 hex digits of its SHA-256.
pub const Digest = struct {
    len: usize,
    sha256: [12]u8,

    pub fn jsonStringify(d: Digest, jw: anytype) !void {
        try jw.write(.{ .len = d.len, .sha256 = @as([]const u8, &d.sha256) });
    }
};

pub fn digest(text: []const u8) Digest {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &h, .{});
    const hex = std.fmt.bytesToHex(h, .lower);
    return .{ .len = text.len, .sha256 = hex[0..12].* };
}

/// `s` cut to `clip_len` bytes at a UTF-8 boundary.
pub fn clip(s: []const u8) []const u8 {
    if (s.len <= clip_len) return s;
    var n: usize = clip_len;
    while (n > 0 and (s[n] & 0xC0) == 0x80) n -= 1;
    return s[0..n];
}

// ---------------------------------------------------------------- tests

const t = std.testing;

test "a line is one JSON object: ts, event, then the fields" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const line = try format(a, 42, "grant", .{ .task_id = "cu_1", .step = 3, .text = digest("on my way") });
    try t.expect(line[line.len - 1] == '\n');
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{});
    try t.expectEqual(@as(i64, 42), v.object.get("ts").?.integer);
    try t.expectEqualStrings("grant", v.object.get("event").?.string);
    try t.expectEqualStrings("cu_1", v.object.get("task_id").?.string);
    // The typed text itself is not in the line.
    try t.expect(std.mem.indexOf(u8, line, "on my way") == null);
    try t.expectEqual(@as(i64, 9), v.object.get("text").?.object.get("len").?.integer);
    try t.expectEqualStrings("{\"ts\":1,\"event\":\"x\"}\n", try format(a, 1, "x", .{}));
}

test "clip keeps whole UTF-8 characters" {
    var s: [100]u8 = undefined; // 50 × "é"
    for (0..50) |i| @memcpy(s[2 * i ..][0..2], "é");
    const c = clip(&s);
    try t.expect(c.len <= clip_len);
    try t.expect(std.unicode.utf8ValidateSlice(c));
    try t.expectEqualStrings("short", clip("short"));
}

test "the tail is the newest lines, oldest first" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const lines = try lastLines(arena.allocator(), "a\nb\nc\n", 2);
    try t.expectEqual(@as(usize, 2), lines.len);
    try t.expectEqualStrings("b", lines[0]);
    try t.expectEqualStrings("c", lines[1]);
    try t.expectEqual(@as(usize, 0), (try lastLines(arena.allocator(), "", 5)).len);
}

test "the log appends to its file and reads its tail" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rnd: [8]u8 = undefined;
    t.io.random(&rnd);
    const dir = try std.fmt.allocPrint(a, "/tmp/localrouter-computer-audit-test-{x:0>16}", .{std.mem.readInt(u64, &rnd, .little)});
    try Io.Dir.cwd().createDirPath(t.io, dir);
    defer Io.Dir.cwd().deleteTree(t.io, dir) catch {};
    var l: Log = .{ .io = t.io, .path = try std.fs.path.join(a, &.{ dir, "computer", "audit.jsonl" }) };
    l.record(t.allocator, "task.start", .{ .task_id = "cu_1" });
    l.record(t.allocator, "task.end", .{ .task_id = "cu_1", .status = "done" });
    const lines = try l.tail(a, 10);
    try t.expectEqual(@as(usize, 2), lines.len);
    try t.expect(std.mem.indexOf(u8, lines[1], "task.end") != null);
}
