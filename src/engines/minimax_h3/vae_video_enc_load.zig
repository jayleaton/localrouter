//! Encoder weights: validate original rank-five shapes, retain the final causal temporal tap, pad GEMM K to eight.
const std = @import("std");
const qi = @import("qwen_image");
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;
pub const Conv = struct { w: u64, b: u64, ci: u32, co: u32, k: u32 };
pub const Norm = struct { w: u64, b: u64, c: u32 };
pub const Block = struct { n1: Norm, n2: Norm, c1: Conv, c2: Conv, shortcut: ?Conv };
pub const Weights = struct {
    input: Conv,
    blocks: [6][2]Block,
    down: [4]Conv,
    norm_out: Norm,
    output: Conv,
    quant: Conv,
    mean: u64,
    stdv: u64,
};
const Loader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    header: std.json.Value,
    start: u64,
    length: u64,
    store: *Store,
    up: *Uploader,
    fn read(l: *Loader, name: []const u8, shape: []const usize) ![]u8 {
        const entry = l.header.object.get(name) orelse return error.MissingWeight;
        const obj = entry.object;
        const dtype = obj.get("dtype") orelse return error.BadWeight;
        if (!std.mem.eql(u8, dtype.string, "F16")) return error.BadWeight;
        const dims = (obj.get("shape") orelse return error.BadWeight).array.items;
        if (dims.len != shape.len) return error.BadWeight;
        var bytes: usize = 2;
        for (shape, dims) |expected, actual| {
            if (actual.integer < 0 or @as(usize, @intCast(actual.integer)) != expected) return error.BadWeight;
            bytes = try std.math.mul(usize, bytes, expected);
        }
        const offsets = (obj.get("data_offsets") orelse return error.BadWeight).array.items;
        if (offsets.len != 2 or offsets[0].integer < 0 or offsets[1].integer < offsets[0].integer) return error.BadWeight;
        const begin: u64 = @intCast(offsets[0].integer);
        const end: u64 = @intCast(offsets[1].integer);
        if (end > l.length or end - begin != bytes) return error.BadWeight;
        const out = try l.gpa.alloc(u8, bytes);
        errdefer l.gpa.free(out);
        if (try l.file.readPositionalAll(l.io, out, l.start + begin) != bytes) return error.BadWeight;
        return out;
    }
    fn tensor(l: *Loader, name: []const u8, shape: []const usize) !u64 {
        const raw = try l.read(name, shape);
        defer l.gpa.free(raw);
        return l.store.upload(l.up, raw);
    }
    fn suffix(l: *Loader, prefix: []const u8, ending: []const u8, shape: []const usize) !u64 {
        var name: [128]u8 = undefined;
        return l.tensor(try std.fmt.bufPrint(&name, "{s}.{s}", .{ prefix, ending }), shape);
    }
    fn conv(l: *Loader, prefix: []const u8, ci: u32, co: u32, k: u32) !Conv {
        var name: [128]u8 = undefined;
        const raw = try l.read(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}), &.{ co, ci, k, k, k });
        defer l.gpa.free(raw);
        const padded = try lastTap(l.gpa, raw, ci, co, k);
        defer l.gpa.free(padded);
        return .{ .w = try l.store.upload(l.up, padded), .b = try l.suffix(prefix, "bias", &.{co}), .ci = ci, .co = co, .k = k };
    }
    fn norm(l: *Loader, prefix: []const u8, c: u32) !Norm {
        return .{ .w = try l.suffix(prefix, "weight", &.{c}), .b = try l.suffix(prefix, "bias", &.{c}), .c = c };
    }
};
pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8, store: *Store, up: *Uploader) !Weights {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var len_buf: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &len_buf, 0) != 8) return error.BadPack;
    const hlen = std.mem.readInt(u64, &len_buf, .little);
    const size = (try file.stat(io)).size;
    if (hlen > 64 << 20 or size < 8 or hlen > size - 8) return error.BadPack;
    const raw = try gpa.alloc(u8, hlen);
    defer gpa.free(raw);
    if (try file.readPositionalAll(io, raw, 8) != hlen) return error.BadPack;
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer parsed.deinit();
    var it = parsed.value.object.iterator();
    while (it.next()) |kv| if (std.mem.endsWith(u8, kv.key_ptr.*, ".comfy_quant")) return error.Int8CheckpointUnsupported;
    var l: Loader = .{ .gpa = gpa, .io = io, .file = file, .header = parsed.value, .start = 8 + hlen, .length = size - 8 - hlen, .store = store, .up = up };
    var w: Weights = undefined;
    w.input = try l.conv("encoder.conv_in", 3, 128, 3);
    var ci: u32 = 128;
    const mult = [_]u32{ 1, 2, 2, 4, 4, 8 };
    for (&w.blocks, 0..) |*level, lev| {
        const co = 128 * mult[lev];
        for (level, 0..) |*b, i| {
            var prefix: [96]u8 = undefined;
            var child: [128]u8 = undefined;
            const p = try std.fmt.bufPrint(&prefix, "encoder.down.{d}.block.{d}", .{ lev, i });
            b.n1 = try l.norm(try std.fmt.bufPrint(&child, "{s}.norm1", .{p}), ci);
            b.n2 = try l.norm(try std.fmt.bufPrint(&child, "{s}.norm2", .{p}), co);
            b.c1 = try l.conv(try std.fmt.bufPrint(&child, "{s}.conv1", .{p}), ci, co, 3);
            b.c2 = try l.conv(try std.fmt.bufPrint(&child, "{s}.conv2", .{p}), co, co, 3);
            b.shortcut = if (ci != co) try l.conv(try std.fmt.bufPrint(&child, "{s}.nin_shortcut", .{p}), ci, co, 1) else null;
            ci = co;
        }
        if (lev < 4) {
            var name: [96]u8 = undefined;
            w.down[lev] = try l.conv(try std.fmt.bufPrint(&name, "encoder.down.{d}.downsample.conv", .{lev}), co, co, 3);
        }
    }
    w.norm_out = try l.norm("encoder.norm_out", ci);
    w.output = try l.conv("encoder.conv_out", ci, 48, 3);
    w.quant = try l.conv("quant_conv", 48, 48, 1);
    w.mean = try l.tensor("latents_mean", &.{24});
    w.stdv = try l.tensor("latents_std", &.{24});
    try store.done(up);
    return w;
}

fn lastTap(gpa: std.mem.Allocator, raw: []const u8, ci: u32, co: u32, k: u32) ![]u8 {
    const K: usize = (ci * k * k + 7) / 8 * 8;
    const padded = try gpa.alloc(u8, @as(usize, co) * K * 2);
    @memset(padded, 0);
    const spatial: usize = k * k * 2;
    for (0..co) |o| for (0..ci) |c| {
        const from = ((o * ci + c) * k + k - 1) * spatial;
        const to = (o * K + c * k * k) * 2;
        @memcpy(padded[to..][0..spatial], raw[from..][0..spatial]);
    };
    return padded;
}
test "one-frame convolution retains final temporal slice and zero pads K" {
    var raw: [108]u8 = undefined;
    for (0..54) |i| std.mem.writeInt(u16, raw[2 * i ..][0..2], @intCast(i), .little);
    const padded = try lastTap(std.testing.allocator, &raw, 1, 2, 3);
    defer std.testing.allocator.free(padded);
    try std.testing.expectEqual(@as(usize, 64), padded.len);
    for (0..2) |o| {
        for (0..9) |j| try std.testing.expectEqual(@as(u16, @intCast(o * 27 + 18 + j)), std.mem.readInt(u16, padded[(o * 16 + j) * 2 ..][0..2], .little));
        for (9..16) |j| try std.testing.expectEqual(@as(u16, 0), std.mem.readInt(u16, padded[(o * 16 + j) * 2 ..][0..2], .little));
    }
}
