//! A twin capture (`tools/twin/stk_twin/rec.py`): `ops.jsonl` (one op a line, in run order) and content-addressed
//! `blobs/<sha256>` holding each op's inputs before and outputs after. The replay walks it alongside the Zig forward.

const std = @import("std");
const Io = std.Io;

/// A tensor as captured: its bytes are `blobs/<sha>`.
pub const Ref = struct {
    sha: []const u8,
    dtype: []const u8,
    shape: []const i64,

    pub fn bytes(r: Ref) usize {
        var n: usize = elemSize(r.dtype);
        for (r.shape) |d| n *= @intCast(d);
        return n;
    }
};

pub fn elemSize(dtype: []const u8) usize {
    const two = .{ "bfloat16", "float16", "int16" };
    const four = .{ "float32", "int32" };
    inline for (two) |t| if (std.mem.eql(u8, dtype, t)) return 2;
    inline for (four) |t| if (std.mem.eql(u8, dtype, t)) return 4;
    if (std.mem.eql(u8, dtype, "int64") or std.mem.eql(u8, dtype, "float64")) return 8;
    return 1; // uint8, int8, bool, float8_e4m3fn
}

pub const Op = struct {
    i: u32,
    name: []const u8,
    kind: []const u8,
    attrs: std.json.Value,
    ins: std.json.ArrayHashMap(Ref),
    outs: std.json.ArrayHashMap(Ref),

    pub fn in(o: *const Op, role: []const u8) ?Ref {
        return o.ins.map.get(role);
    }
    pub fn out(o: *const Op, role: []const u8) ?Ref {
        return o.outs.map.get(role);
    }
};

pub const Capture = struct {
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    ops: []Op,
    notes: []std.json.Value,

    /// Parses `ops.jsonl`; notes (sigmas, the request) are kept apart from ops.
    pub fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Capture {
        var c: Capture = .{ .arena = .init(gpa), .dir = "", .ops = &.{}, .notes = &.{} };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        c.dir = try a.dupe(u8, dir);
        const text = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "ops.jsonl" }), a, .limited(256 << 20));
        var ops: std.ArrayList(Op) = .empty;
        var notes: std.ArrayList(std.json.Value) = .empty;
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const v = try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{});
            if (v.object.get("note") != null) {
                try notes.append(a, v);
                continue;
            }
            try ops.append(a, try std.json.parseFromValueLeaky(Op, a, v, .{ .ignore_unknown_fields = true }));
        }
        c.ops = ops.items;
        c.notes = notes.items;
        return c;
    }

    pub fn close(c: *Capture) void {
        c.arena.deinit();
    }

    /// The note named `name` (e.g. "sigmas"), or null.
    pub fn note(c: *const Capture, name: []const u8) ?std.json.Value {
        for (c.notes) |n| if (std.mem.eql(u8, n.object.get("note").?.string, name)) return n;
        return null;
    }

    /// Reads a blob into `out` (exactly its size).
    pub fn blob(c: *const Capture, io: Io, r: Ref, out: []u8) !void {
        var path_buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/blobs/{s}", .{ c.dir, r.sha });
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        if (out.len != r.bytes()) return error.SizeMismatch;
        if (try file.readPositionalAll(io, out, 0) != out.len) return error.ShortBlob;
    }

    /// The first op named `name` at or after index `from`, or null.
    pub fn find(c: *const Capture, name: []const u8, from: usize) ?usize {
        for (c.ops[from..], from..) |o, i| if (std.mem.eql(u8, o.name, name)) return i;
        return null;
    }
};

test "parse a capture: ops, notes, blobs" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "/tmp/stk-capture-test";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir ++ "/blobs");
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/ops.jsonl", .data =
        \\{"i": 0, "note": "sigmas", "values": [1.0, 0.5, 0.0]}
        \\{"i": 1, "name": "L0.adaln1", "kind": "adaln", "attrs": {"eps": 1e-06}, "ins": {"x": {"sha": "aa", "dtype": "bfloat16", "shape": [1, 2]}}, "outs": {"y": {"sha": "bb", "dtype": "bfloat16", "shape": [1, 2]}}, "launches": []}
        \\
    });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/blobs/aa", .data = &.{ 1, 2, 3, 4 } });
    var c = try Capture.open(gpa, io, dir);
    defer c.close();
    try std.testing.expectEqual(@as(usize, 1), c.ops.len);
    try std.testing.expectEqualStrings("adaln", c.ops[0].kind);
    try std.testing.expectEqual(@as(usize, 4), c.ops[0].in("x").?.bytes());
    try std.testing.expectEqual(@as(usize, 3), c.note("sigmas").?.object.get("values").?.array.items.len);
    var buf: [4]u8 = undefined;
    try c.blob(io, c.ops[0].in("x").?, &buf);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &buf);
    try std.testing.expectEqual(@as(?usize, 0), c.find("L0.adaln1", 0));
}
