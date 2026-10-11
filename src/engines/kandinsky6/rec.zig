//! The native side of the Kandinsky-6 bit gate: writes a capture in the twin's format (`tools/twin/stk_twin/rec.py`,
//! read by `stk_twin.kandinsky6.compare`): `ops.jsonl`, one op or note a line in run order, and `blobs/<sha256>`, the
//! raw little-endian bytes of every tensor. Op and role names are the oracle's (tools/twin/stk_twin/kandinsky6/README.md
//! "Ops"). Device tensors are read after the stream synchronizes, so recording serializes the pipeline: a gate run, not
//! a timed one. `ops.jsonl` is rewritten after every record, so a run that fails midway leaves what it reached.

const std = @import("std");
const cuda = @import("cuda");

pub const Dtype = enum {
    bfloat16,
    float32,
    uint8,
    int16,
    int64,

    pub fn size(d: Dtype) u64 {
        return switch (d) {
            .uint8 => 1,
            .bfloat16, .int16 => 2,
            .float32 => 4,
            .int64 => 8,
        };
    }
};

/// One role of an op: where its bytes are, and what they are.
pub const Tensor = struct {
    role: []const u8,
    dtype: Dtype,
    shape: []const u64,
    src: union(enum) { device: u64, host: []const u8 },

    pub fn bytes(t: Tensor) u64 {
        var n = t.dtype.size();
        for (t.shape) |d| n *= d;
        return n;
    }
};

pub const Rec = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    d: ?*const cuda.Driver = null,
    s: ?cuda.Stream = null,
    lines: std.ArrayList(u8) = .empty,
    i: u32 = 0,
    host: []u8 = &.{},
    /// Op name prefixes to record (the requested stages'); empty records every op.
    only: []const []const u8 = &.{},

    /// A new capture at `path`, which must not exist (the gate never mixes runs).
    pub fn create(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Rec {
        const cwd = std.Io.Dir.cwd();
        if (cwd.access(io, path, .{})) |_| return error.OutputExists else |_| {}
        try cwd.createDirPath(io, path);
        var dir = try cwd.openDir(io, path, .{});
        errdefer dir.close(io);
        try dir.createDirPath(io, "blobs");
        return .{ .gpa = gpa, .io = io, .dir = dir };
    }

    pub fn deinit(r: *Rec) void {
        r.lines.deinit(r.gpa);
        r.gpa.free(r.host);
        r.dir.close(r.io);
    }

    /// The device the pipeline runs on: device tensors are read through it after `s` synchronizes.
    pub fn attach(r: *Rec, d: *const cuda.Driver, s: cuda.Stream) void {
        r.d = d;
        r.s = s;
    }

    /// A note: {"i", "note": name, ...the fields of `value`}.
    pub fn note(r: *Rec, name: []const u8, value: anytype) !void {
        const body = try std.json.Stringify.valueAlloc(r.gpa, value, .{});
        defer r.gpa.free(body);
        if (body.len < 2 or body[0] != '{') return error.NoteNotAnObject;
        const w = &r.lines;
        try w.print(r.gpa, "{{\"i\": {d}, \"note\": ", .{r.i});
        try r.str(name);
        if (body.len > 2) try w.print(r.gpa, ", {s}", .{body[1..]}) else try w.appendSlice(r.gpa, "}");
        try w.append(r.gpa, '\n');
        r.i += 1;
        try r.flush();
    }

    /// An op: {"i", "name", "kind", "attrs", "ins": {role: ref}, "outs": {role: ref}, "launches": []}.
    pub fn op(r: *Rec, name: []const u8, kind: []const u8, attrs: anytype, ins: []const Tensor, outs: []const Tensor) !void {
        if (!r.wants(name)) return;
        if (r.s) |s| try s.synchronize();
        const a = try std.json.Stringify.valueAlloc(r.gpa, attrs, .{});
        defer r.gpa.free(a);
        const w = &r.lines;
        try w.print(r.gpa, "{{\"i\": {d}, \"name\": ", .{r.i});
        try r.str(name);
        try w.appendSlice(r.gpa, ", \"kind\": ");
        try r.str(kind);
        try w.print(r.gpa, ", \"attrs\": {s}, \"ins\": ", .{a});
        try r.refs(ins);
        try w.appendSlice(r.gpa, ", \"outs\": ");
        try r.refs(outs);
        try w.appendSlice(r.gpa, ", \"launches\": []}\n");
        r.i += 1;
        try r.flush();
    }

    pub fn wants(r: *const Rec, name: []const u8) bool {
        if (r.only.len == 0) return true;
        for (r.only) |pre| if (std.mem.startsWith(u8, name, pre)) return true;
        return false;
    }

    /// `v` as a JSON string literal.
    fn str(r: *Rec, v: []const u8) !void {
        const q = try std.json.Stringify.valueAlloc(r.gpa, v, .{});
        defer r.gpa.free(q);
        try r.lines.appendSlice(r.gpa, q);
    }

    fn refs(r: *Rec, ts: []const Tensor) !void {
        const w = &r.lines;
        try w.append(r.gpa, '{');
        for (ts, 0..) |t, k| {
            if (k > 0) try w.appendSlice(r.gpa, ", ");
            const sha = try r.blob(t);
            try r.str(t.role);
            try w.print(r.gpa, ": {{\"sha\": \"{s}\", \"dtype\": \"{t}\", \"shape\": [", .{ &sha, t.dtype });
            for (t.shape, 0..) |d, j| try w.print(r.gpa, "{s}{d}", .{ if (j > 0) ", " else "", d });
            try w.appendSlice(r.gpa, "]}");
        }
        try w.append(r.gpa, '}');
    }

    /// Stores `t`'s bytes as blobs/<sha256> (once) and returns the hex digest.
    fn blob(r: *Rec, t: Tensor) ![64]u8 {
        const n: usize = @intCast(t.bytes());
        const data: []const u8 = switch (t.src) {
            .host => |h| blk: {
                if (h.len != n) return error.ShapeMismatch;
                break :blk h;
            },
            .device => |ptr| blk: {
                const d = r.d orelse return error.NoDevice;
                if (r.host.len < n) r.host = try r.gpa.realloc(r.host, n);
                const view: cuda.DeviceBuffer = .{ .d = d, .ptr = ptr, .len = n, .borrowed = true };
                try view.download(0, r.host[0..n]);
                break :blk r.host[0..n];
            },
        };
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(data, &h, .{});
        const hex = std.fmt.bytesToHex(h, .lower);
        var nb: [80]u8 = undefined;
        const path = try std.fmt.bufPrint(&nb, "blobs/{s}", .{&hex});
        if (r.dir.access(r.io, path, .{})) |_| {} else |_| try r.dir.writeFile(r.io, .{ .sub_path = path, .data = data });
        return hex;
    }

    fn flush(r: *Rec) !void {
        try r.dir.writeFile(r.io, .{ .sub_path = "ops.jsonl", .data = r.lines.items });
    }
};

test "a capture the twin's reader parses: ops, notes, content-addressed blobs" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/tmp/k6-rec-test";
    std.Io.Dir.cwd().deleteTree(io, path) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, path) catch {};
    var r = try Rec.create(gpa, io, path);
    defer r.deinit();
    try r.note("stage", .{ .stage = "noise", .ok = true });
    const x = [_]u16{ 0x3f80, 0xbf80 };
    try r.op("k6.audio.latent_in", "k6", .{ .chained = true }, &.{}, &.{.{ .role = "latent", .dtype = .bfloat16, .shape = &.{ 1, 2 }, .src = .{ .host = std.mem.sliceAsBytes(&x) } }});
    try std.testing.expectError(error.OutputExists, Rec.create(gpa, io, path));

    const qc = @import("qwen_image").capture;
    var c = try qc.Capture.open(gpa, io, path);
    defer c.close();
    try std.testing.expectEqual(@as(usize, 1), c.ops.len);
    try std.testing.expect(c.note("stage") != null);
    const ref = c.ops[0].out("latent").?;
    try std.testing.expectEqualStrings("bfloat16", ref.dtype);
    var back: [4]u8 = undefined;
    try c.blob(io, ref, &back);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&x), &back);
    try std.testing.expectEqual(true, c.ops[0].attrs.object.get("chained").?.bool);
}
