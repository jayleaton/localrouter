//! A converted weight pack (`tools/twin/stk_twin/pack.py`): `weights.safetensors` + `manifest.json`. The header is
//! read once. Bulk tensor bytes are read with direct I/O (O_DIRECT: no page cache), so loading never holds two
//! copies of the weights (on GB10 the page cache is GPU memory) and runs at the drive's speed: on GB10 the cached
//! path manages about 1 GB/s, direct reads 12 GB/s. Small reads, and filesystems that refuse O_DIRECT, use pread and
//! drop the page cache as they go.

const std = @import("std");
const Io = std.Io;
const posix = std.posix;
const linux = std.os.linux;

pub const DType = enum {
    bf16,
    f32,
    u8,
    f8_e4m3,
    i32,
    i8,
    f16,

    pub fn size(d: DType) usize {
        return switch (d) {
            .u8, .f8_e4m3, .i8 => 1,
            .f16 => 2,
            .bf16 => 2,
            .f32, .i32 => 4,
        };
    }

    fn parse(s: []const u8) ?DType {
        const names = .{ .{ "BF16", .bf16 }, .{ "F32", .f32 }, .{ "U8", .u8 }, .{ "F8_E4M3", .f8_e4m3 }, .{ "I32", .i32 }, .{ "I8", .i8 }, .{ "F16", .f16 } };
        inline for (names) |n| if (std.mem.eql(u8, s, n[0])) return n[1];
        return null;
    }
};

pub const Tensor = struct {
    dtype: DType,
    shape: [4]usize = .{ 1, 1, 1, 1 },
    rank: u8,
    begin: usize, // within the data region
    end: usize,

    pub fn len(t: Tensor) usize {
        return t.end - t.begin;
    }
    pub fn dim(t: Tensor, i: usize) usize {
        return if (i < t.rank) t.shape[i] else 1;
    }
};

/// One linear's storage kind, from the manifest.
pub const Kind = enum { nvfp4, fp8, bf16 };

/// Direct I/O's alignment of offsets, lengths and buffers (the page size covers every logical block size).
pub const block: usize = 4096;

pub const Pack = struct {
    arena: std.heap.ArenaAllocator,
    tensors: std.StringArrayHashMapUnmanaged(Tensor) = .empty,
    kinds: std.StringArrayHashMapUnmanaged(Kind) = .empty,
    precision: []const u8 = "",
    file: Io.File,
    direct: posix.fd_t = -1, // the same file opened O_DIRECT, or -1 where the filesystem refuses it
    data_start: usize,
    data_len: usize,

    pub fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Pack {
        var p: Pack = .{ .arena = .init(gpa), .file = undefined, .data_start = 0, .data_len = 0 };
        errdefer p.arena.deinit();
        const a = p.arena.allocator();
        const man_text = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(16 << 20));
        try p.parseManifest(a, man_text);
        const path = try std.fs.path.joinZ(a, &.{ dir, "weights.safetensors" });
        p.file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer p.file.close(io);
        p.direct = posix.openatZ(posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .DIRECT = true, .CLOEXEC = true }, 0) catch -1;
        errdefer if (p.direct >= 0) {
            _ = linux.close(p.direct);
        };
        var len_buf: [8]u8 = undefined;
        if (try p.file.readPositionalAll(io, &len_buf, 0) != 8) return error.BadPack;
        const hlen = std.mem.readInt(u64, &len_buf, .little);
        if (hlen > 64 << 20) return error.BadPack;
        const header = try a.alloc(u8, hlen);
        if (try p.file.readPositionalAll(io, header, 8) != hlen) return error.BadPack;
        p.data_start = 8 + hlen;
        const size = (try p.file.stat(io)).size;
        p.data_len = size - p.data_start;
        try p.parseHeader(a, header);
        return p;
    }

    /// A plain safetensors file (a checkpoint as published): its tensors, no manifest (no kinds, no precision).
    pub fn openFile(gpa: std.mem.Allocator, io: Io, path: []const u8) !Pack {
        var p: Pack = .{ .arena = .init(gpa), .file = undefined, .data_start = 0, .data_len = 0 };
        errdefer p.arena.deinit();
        const a = p.arena.allocator();
        p.file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer p.file.close(io);
        var len_buf: [8]u8 = undefined;
        if (try p.file.readPositionalAll(io, &len_buf, 0) != 8) return error.BadPack;
        const hlen = std.mem.readInt(u64, &len_buf, .little);
        if (hlen > 64 << 20) return error.BadPack;
        const header = try a.alloc(u8, hlen);
        if (try p.file.readPositionalAll(io, header, 8) != hlen) return error.BadPack;
        p.data_start = 8 + hlen;
        p.data_len = (try p.file.stat(io)).size - p.data_start;
        try p.parseHeader(a, header);
        return p;
    }

    pub fn close(p: *Pack, io: Io) void {
        if (p.direct >= 0) _ = linux.close(p.direct);
        p.file.close(io);
        p.arena.deinit();
    }

    /// The tensor named `name`; `error.MissingTensor` names a pack that does not match the engine.
    pub fn get(p: *const Pack, name: []const u8) !Tensor {
        return p.tensors.get(name) orelse error.MissingTensor;
    }

    pub fn kind(p: *const Pack, linear: []const u8) !Kind {
        return p.kinds.get(linear) orelse error.MissingTensor;
    }

    /// Reads `t`'s bytes into `out` (exactly `t.len()` bytes) and lets the page cache go.
    pub fn read(p: *const Pack, io: Io, t: Tensor, out: []u8) !void {
        std.debug.assert(out.len == t.len());
        try p.readRange(io, t, 0, out);
    }

    /// Reads `out.len` bytes of `t` from byte `at` of it, and lets the page cache go.
    pub fn readRange(p: *const Pack, io: Io, t: Tensor, at: usize, out: []u8) !void {
        std.debug.assert(at + out.len <= t.len());
        const off = p.data_start + t.begin + at;
        if (try p.file.readPositionalAll(io, out, off) != out.len) return error.BadPack;
        _ = linux.fadvise(p.file.handle, @intCast(off), @intCast(out.len), linux.POSIX_FADV.DONTNEED);
    }

    /// Bytes [at, at + n) of `t` into `buf` (block-aligned, at least `n + 2 * block` bytes) with direct I/O: the
    /// covering block-aligned range is read, and the result is where the bytes start in `buf`. Without a direct
    /// descriptor it reads them at 0 the buffered way.
    pub fn readAligned(p: *const Pack, io: Io, t: Tensor, at: usize, n: usize, buf: []align(block) u8) !usize {
        std.debug.assert(at + n <= t.len() and buf.len >= n + 2 * block);
        if (p.direct < 0) {
            try p.readRange(io, t, at, buf[0..n]);
            return 0;
        }
        const off = p.data_start + t.begin + at;
        const start = off / block * block;
        const lead = off - start;
        const len = std.mem.alignForward(usize, lead + n, block);
        var got: usize = 0;
        while (got < lead + n) { // the last block may run past the end of the file: a short read covers it
            const rc = linux.pread(p.direct, buf.ptr + got, len - got, @intCast(start + got));
            switch (linux.errno(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                else => |e| return posix.unexpectedErrno(e),
            }
            if (rc == 0) return error.BadPack;
            got += rc;
        }
        return lead;
    }

    fn parseManifest(p: *Pack, a: std.mem.Allocator, text: []const u8) !void {
        const M = struct { format: []const u8, precision: []const u8, kinds: std.json.ArrayHashMap([]const u8) };
        const m = try std.json.parseFromSliceLeaky(M, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        if (!std.mem.eql(u8, m.format, "stk-pack/1")) return error.UnknownPackFormat;
        p.precision = m.precision;
        var it = m.kinds.map.iterator();
        while (it.next()) |kv| {
            const k = std.meta.stringToEnum(Kind, kv.value_ptr.*) orelse return error.UnknownKind;
            try p.kinds.put(a, kv.key_ptr.*, k);
        }
    }

    fn parseHeader(p: *Pack, a: std.mem.Allocator, json: []const u8) !void {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
        var it = v.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const o = kv.value_ptr.object;
            const dt = DType.parse(o.get("dtype").?.string) orelse return error.UnknownDType;
            const shape = o.get("shape").?.array.items;
            const offs = o.get("data_offsets").?.array.items;
            if (offs.len != 2) return error.BadPack;
            if (shape.len > 4) continue; // e.g. a vision tower's 3-D patch embedding: no engine here reads one; asking for it is MissingTensor
            var t: Tensor = .{ .dtype = dt, .rank = @intCast(shape.len), .begin = @intCast(offs[0].integer), .end = @intCast(offs[1].integer) };
            var n: usize = dt.size();
            for (shape, 0..) |s, i| {
                t.shape[i] = @intCast(s.integer);
                n *= t.shape[i];
            }
            if (t.end < t.begin or t.end > p.data_len or t.len() != n) return error.BadPack;
            try p.tensors.put(a, kv.key_ptr.*, t);
        }
    }
};

test "open a pack: manifest kinds, tensors, reads" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "/tmp/stk-pack-test";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/manifest.json", .data =
        \\{"format": "stk-pack/1", "model": "qwen-image-2.1", "precision": "nvfp4", "kinds": {"L0.qkv": "nvfp4", "L0.out": "fp8"}, "tensors": {}}
    });
    const header = "{\"L0.qkv.codes\":{\"dtype\":\"U8\",\"shape\":[2,3],\"data_offsets\":[0,6]},\"L0.out.w8\":{\"dtype\":\"F8_E4M3\",\"shape\":[2],\"data_offsets\":[6,8]},\"__metadata__\":{}}";
    var file: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8 .. 8 + header.len], header);
    @memcpy(file[8 + header.len ..], &[_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/weights.safetensors", .data = &file });

    var p = try Pack.open(gpa, io, dir);
    defer p.close(io);
    try std.testing.expectEqualStrings("nvfp4", p.precision);
    try std.testing.expectEqual(Kind.fp8, try p.kind("L0.out"));
    const t = try p.get("L0.out.w8");
    try std.testing.expectEqual(DType.f8_e4m3, t.dtype);
    var buf: [2]u8 = undefined;
    try p.read(io, t, &buf);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8 }, &buf);
    try std.testing.expectError(error.MissingTensor, p.get("nope"));
}

test "aligned reads: odd offsets and lengths come back exact, direct and buffered alike" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "/var/tmp/stk-pack-direct-test"; // a disk filesystem, so O_DIRECT is taken where it exists
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/manifest.json", .data =
        \\{"format": "stk-pack/1", "precision": "nvfp4", "kinds": {}, "tensors": {}}
    });
    const n = 3 * block + 777; // a tensor that starts mid-block (after the odd header) and ends mid-block
    const header = "{\"t\":{\"dtype\":\"U8\",\"shape\":[" ++ std.fmt.comptimePrint("{d}", .{n}) ++ "],\"data_offsets\":[0," ++ std.fmt.comptimePrint("{d}", .{n}) ++ "]}}";
    const file = try gpa.alloc(u8, 8 + header.len + n);
    defer gpa.free(file);
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8 .. 8 + header.len], header);
    const data = file[8 + header.len ..];
    for (data, 0..) |*c, j| c.* = @truncate(j *% 131 +% 7);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/weights.safetensors", .data = file });

    var p = try Pack.open(gpa, io, dir);
    defer p.close(io);
    const t = try p.get("t");
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(block), n + 2 * block);
    defer gpa.free(buf);
    for ([_][2]usize{ .{ 0, n }, .{ 1, 5 }, .{ block - 3, 9 }, .{ 2 * block + 1, n - 2 * block - 1 }, .{ 100, 2 * block } }) |r| {
        const from = try p.readAligned(io, t, r[0], r[1], buf);
        try std.testing.expectEqualSlices(u8, data[r[0]..][0..r[1]], buf[from..][0..r[1]]);
    }
}
