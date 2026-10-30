//! Kandinsky-6 video VAE weights from the checkpoint (`vae/diffusion_pytorch_model.safetensors`, 248 fp32 tensors) into
//! one device image: every convolution and linear repacked to `[co][ldw]` fp16 rows (`launch.repackConv`), biases and
//! GroupNorm parameters fp16, each tensor 256-byte aligned. The image is built on the host with addresses relative to
//! its start; `relocate` adds the device address after the caller uploads it. fp32 -> fp16 rounds to nearest even, as
//! torch's `.to(float16)` when the native loader passes `torch_dtype=float16`.

const std = @import("std");
const launch = @import("launch.zig");
const encode = @import("encode.zig");
const decode = @import("decode.zig");

const ConvW = launch.ConvW;

pub const Error = error{ InvalidCheckpoint, MissingTensor, UnexpectedTensor, ShapeMismatch, DuplicateTensor };

/// One tensor of the safetensors header: fp32 only (the checkpoint has no other dtype).
pub const Entry = struct { name: []const u8, shape: []const u32, begin: u64, end: u64 };

/// The parsed header of a safetensors file: entries sorted by name and where the payload starts.
pub const Header = struct {
    arena: std.heap.ArenaAllocator,
    entries: []Entry,
    data_start: u64,

    pub fn deinit(self: *Header) void {
        self.arena.deinit();
    }

    pub fn find(self: Header, name: []const u8) ?Entry {
        return if (self.index(name)) |i| self.entries[i] else null;
    }

    fn index(self: Header, name: []const u8) ?usize {
        var lo: usize = 0;
        var hi: usize = self.entries.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            switch (std.mem.order(u8, self.entries[mid].name, name)) {
                .lt => lo = mid + 1,
                .gt => hi = mid,
                .eq => return mid,
            }
        }
        return null;
    }
};

/// Parses the 8-byte length and the JSON header at the start of `file` (only the header bytes need be present).
pub fn parseHeader(gpa: std.mem.Allocator, file: []const u8) !Header {
    if (file.len < 8) return error.InvalidCheckpoint;
    const n = std.mem.readInt(u64, file[0..8], .little);
    if (n > file.len - 8 or n > 100 << 20) return error.InvalidCheckpoint;
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, file[8..][0..n], .{});
    if (root != .object) return error.InvalidCheckpoint;
    var list: std.ArrayList(Entry) = .empty;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const o = kv.value_ptr.*;
        if (o != .object) return error.InvalidCheckpoint;
        const dt = o.object.get("dtype") orelse return error.InvalidCheckpoint;
        if (dt != .string or !std.mem.eql(u8, dt.string, "F32")) return error.InvalidCheckpoint;
        const sh = o.object.get("shape") orelse return error.InvalidCheckpoint;
        const off = o.object.get("data_offsets") orelse return error.InvalidCheckpoint;
        if (sh != .array or off != .array or off.array.items.len != 2) return error.InvalidCheckpoint;
        const shape = try a.alloc(u32, sh.array.items.len);
        var elems: u64 = 1;
        for (sh.array.items, shape) |v, *d| {
            if (v != .integer or v.integer < 0) return error.InvalidCheckpoint;
            d.* = @intCast(v.integer);
            elems *= d.*;
        }
        const b = off.array.items[0];
        const e = off.array.items[1];
        if (b != .integer or e != .integer or b.integer < 0 or e.integer < b.integer) return error.InvalidCheckpoint;
        if (@as(u64, @intCast(e.integer - b.integer)) != elems * 4) return error.InvalidCheckpoint;
        try list.append(a, .{ .name = kv.key_ptr.*, .shape = shape, .begin = @intCast(b.integer), .end = @intCast(e.integer) });
    }
    const entries = try list.toOwnedSlice(a);
    std.mem.sort(Entry, entries, {}, struct {
        fn lt(_: void, x: Entry, y: Entry) bool {
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.lt);
    return .{ .arena = arena, .entries = entries, .data_start = 8 + n };
}

/// The device image: `bytes` to upload, and both graphs' weights at addresses relative to the image's start.
pub const Image = struct {
    bytes: []align(256) u8,
    enc: encode.Encoder,
    dec: decode.Decoder,

    pub fn deinit(self: *Image, gpa: std.mem.Allocator) void {
        gpa.free(self.bytes);
    }
};

/// Where tensor data comes from: `payload` is the safetensors data section (everything after the header).
pub const Source = struct {
    header: *const Header,
    payload: []const u8,

    fn floats(self: Source, e: Entry) Error![]align(1) const f32 {
        if (e.end > self.payload.len) return error.InvalidCheckpoint;
        return std.mem.bytesAsSlice(f32, self.payload[e.begin..e.end]);
    }
};

const Builder = struct {
    gpa: std.mem.Allocator,
    src: Source,
    out: std.ArrayList(u8) = .empty,
    used: []bool,
    scratch: std.ArrayList(f32) = .empty,

    fn take(self: *Builder, name: []const u8, shape: []const u32) !Entry {
        const i = self.src.header.index(name) orelse return error.MissingTensor;
        const e = self.src.header.entries[i];
        if (!std.mem.eql(u32, e.shape, shape)) return error.ShapeMismatch;
        if (self.used[i]) return error.DuplicateTensor;
        self.used[i] = true;
        return e;
    }

    fn reserve(self: *Builder, bytes: u64) !u64 {
        const at = std.mem.alignForward(u64, self.out.items.len, 256);
        try self.out.appendNTimes(self.gpa, 0, at + bytes - self.out.items.len);
        return at;
    }

    fn halves(self: *Builder, at: u64, n: usize) []align(1) f16 {
        return std.mem.bytesAsSlice(f16, self.out.items[at..][0 .. n * 2]);
    }

    fn vec(self: *Builder, name: []const u8, c: u32) !u64 {
        const e = try self.take(name, &.{c});
        const at = try self.reserve(@as(u64, c) * 2);
        const src = try self.src.floats(e);
        for (src, self.halves(at, c)) |s, *d| d.* = @floatCast(s);
        return at;
    }

    fn conv(self: *Builder, prefix: []const u8, co: u32, ci: u32, k: u32, linear: bool) !ConvW {
        var nb: [128]u8 = undefined;
        const shape5 = [_]u32{ co, ci, k, k, k };
        const e = try self.take(try std.fmt.bufPrint(&nb, "{s}.weight", .{prefix}), if (linear) shape5[0..2] else &shape5);
        const ldw = std.mem.alignForward(u32, k * k * k * ci, 8);
        const at = try self.reserve(@as(u64, co) * ldw * 2);
        const src = try self.src.floats(e);
        try self.scratch.resize(self.gpa, src.len);
        for (src, self.scratch.items) |s, *d| d.* = s;
        const dst = try self.gpa.alloc(f16, @as(usize, co) * ldw);
        defer self.gpa.free(dst);
        try launch.repackConv(self.scratch.items, dst, co, ci, k, k, k);
        @memcpy(self.out.items[at..][0 .. dst.len * 2], std.mem.sliceAsBytes(dst));
        const b = try self.vec(try std.fmt.bufPrint(&nb, "{s}.bias", .{prefix}), co);
        return .{ .w = at, .b = b, .co = co, .ci = ci, .kt = k, .kh = k, .kw = k };
    }

    fn norm(self: *Builder, prefix: []const u8, c: u32) !launch.NormW {
        var nb: [128]u8 = undefined;
        const g = try self.vec(try std.fmt.bufPrint(&nb, "{s}.weight", .{prefix}), c);
        const b = try self.vec(try std.fmt.bufPrint(&nb, "{s}.bias", .{prefix}), c);
        return .{ .gamma = g, .beta = b, .c = c };
    }

    fn res(self: *Builder, prefix: []const u8, ci: u32, co: u32) !launch.ResnetW {
        var p: [128]u8 = undefined;
        return .{
            .norm1 = try self.norm(try std.fmt.bufPrint(&p, "{s}.norm1", .{prefix}), ci),
            .conv1 = try self.conv(try std.fmt.bufPrint(&p, "{s}.conv1.conv", .{prefix}), co, ci, 3, false),
            .norm2 = try self.norm(try std.fmt.bufPrint(&p, "{s}.norm2", .{prefix}), co),
            .conv2 = try self.conv(try std.fmt.bufPrint(&p, "{s}.conv2.conv", .{prefix}), co, co, 3, false),
            .shortcut = if (ci != co) try self.conv(try std.fmt.bufPrint(&p, "{s}.conv_shortcut.conv", .{prefix}), co, ci, 1, false) else null,
        };
    }

    fn mid(self: *Builder, prefix: []const u8) !launch.MidW {
        var p: [128]u8 = undefined;
        const r0 = try self.res(try std.fmt.bufPrint(&p, "{s}.mid_block.resnets.0", .{prefix}), 512, 512);
        const a = "{s}.mid_block.attentions.0.";
        const attn: launch.AttnW = .{
            .norm = try self.norm(try std.fmt.bufPrint(&p, a ++ "group_norm", .{prefix}), 512),
            .q = try self.conv(try std.fmt.bufPrint(&p, a ++ "to_q", .{prefix}), 512, 512, 1, true),
            .k = try self.conv(try std.fmt.bufPrint(&p, a ++ "to_k", .{prefix}), 512, 512, 1, true),
            .v = try self.conv(try std.fmt.bufPrint(&p, a ++ "to_v", .{prefix}), 512, 512, 1, true),
            .out = try self.conv(try std.fmt.bufPrint(&p, a ++ "to_out.0", .{prefix}), 512, 512, 1, true),
        };
        const r1 = try self.res(try std.fmt.bufPrint(&p, "{s}.mid_block.resnets.1", .{prefix}), 512, 512);
        return .{ .res0 = r0, .attn = attn, .res1 = r1 };
    }
};

/// Builds the device image from every tensor of the checkpoint; refuses missing, unexpected or misshapen tensors.
pub fn build(gpa: std.mem.Allocator, src: Source) !Image {
    const used = try gpa.alloc(bool, src.header.entries.len);
    defer gpa.free(used);
    @memset(used, false);
    var b: Builder = .{ .gpa = gpa, .src = src, .used = used };
    defer b.scratch.deinit(gpa);
    errdefer b.out.deinit(gpa);
    var p: [128]u8 = undefined;

    var e: encode.Encoder = undefined;
    e.conv_in = try b.conv("encoder.conv_in.conv", 128, 3, 3, false);
    const enc_ch = [4][2]u32{ .{ 128, 128 }, .{ 128, 256 }, .{ 256, 512 }, .{ 512, 512 } };
    for (enc_ch, 0..) |io, i| {
        for (0..2) |r| e.down[i].resnets[r] = try b.res(try std.fmt.bufPrint(&p, "encoder.down_blocks.{d}.resnets.{d}", .{ i, r }), if (r == 0) io[0] else io[1], io[1]);
        e.down[i].downsampler = if (i < 3) try b.conv(try std.fmt.bufPrint(&p, "encoder.down_blocks.{d}.downsamplers.0.conv.conv", .{i}), io[1], io[1], 3, false) else null;
    }
    e.mid = try b.mid("encoder");
    e.norm_out = try b.norm("encoder.conv_norm_out", 512);
    e.conv_out = try b.conv("encoder.conv_out.conv", 32, 512, 3, false);
    e.quant = try b.conv("quant_conv", 32, 32, 1, false);

    var d: decode.Decoder = undefined;
    d.post_quant = try b.conv("post_quant_conv", 16, 16, 1, false);
    d.conv_in = try b.conv("decoder.conv_in.conv", 512, 16, 3, false);
    d.mid = try b.mid("decoder");
    const dec_ch = [4][2]u32{ .{ 512, 512 }, .{ 512, 512 }, .{ 512, 256 }, .{ 256, 128 } };
    for (dec_ch, 0..) |io, i| {
        for (0..3) |r| d.up[i].resnets[r] = try b.res(try std.fmt.bufPrint(&p, "decoder.up_blocks.{d}.resnets.{d}", .{ i, r }), if (r == 0) io[0] else io[1], io[1]);
        d.up[i].upsampler = if (i < 3) try b.conv(try std.fmt.bufPrint(&p, "decoder.up_blocks.{d}.upsamplers.0.conv.conv", .{i}), io[1], io[1], 3, false) else null;
    }
    d.norm_out = try b.norm("decoder.conv_norm_out", 128);
    d.conv_out = try b.conv("decoder.conv_out.conv", 3, 128, 3, false);

    for (used) |u| if (!u) return error.UnexpectedTensor;
    const len = std.mem.alignForward(usize, b.out.items.len, 256);
    const bytes = try gpa.alignedAlloc(u8, .fromByteUnits(256), len);
    @memcpy(bytes[0..b.out.items.len], b.out.items);
    @memset(bytes[b.out.items.len..], 0);
    b.out.deinit(gpa);
    return .{ .bytes = bytes, .enc = e, .dec = d };
}

/// Adds `base` (the image's device address) to every weight address of the encoder.
pub fn relocateEncoder(e: *encode.Encoder, base: u64) void {
    convAt(&e.conv_in, base);
    for (&e.down) |*d| {
        for (&d.resnets) |*r| resnetAt(r, base);
        if (d.downsampler) |*c| convAt(c, base);
    }
    midAt(&e.mid, base);
    normAt(&e.norm_out, base);
    convAt(&e.conv_out, base);
    convAt(&e.quant, base);
}

/// Adds `base` (the image's device address) to every weight address of the decoder.
pub fn relocateDecoder(d: *decode.Decoder, base: u64) void {
    convAt(&d.post_quant, base);
    convAt(&d.conv_in, base);
    midAt(&d.mid, base);
    for (&d.up) |*u| {
        for (&u.resnets) |*r| resnetAt(r, base);
        if (u.upsampler) |*c| convAt(c, base);
    }
    normAt(&d.norm_out, base);
    convAt(&d.conv_out, base);
}

fn convAt(c: *ConvW, base: u64) void {
    c.w += base;
    c.b += base;
}

fn normAt(n: *launch.NormW, base: u64) void {
    n.gamma += base;
    n.beta += base;
}

fn resnetAt(r: *launch.ResnetW, base: u64) void {
    normAt(&r.norm1, base);
    convAt(&r.conv1, base);
    normAt(&r.norm2, base);
    convAt(&r.conv2, base);
    if (r.shortcut) |*c| convAt(c, base);
}

fn midAt(m: *launch.MidW, base: u64) void {
    resnetAt(&m.res0, base);
    normAt(&m.attn.norm, base);
    for ([_]*ConvW{ &m.attn.q, &m.attn.k, &m.attn.v, &m.attn.out }) |c| convAt(c, base);
    resnetAt(&m.res1, base);
}

// ---- Tests ----------------------------------------------------------------------------------------------------------

const testing = std.testing;

/// A safetensors file with the pinned checkpoint's names and shapes (testdata/vae_tensors.txt) whose tensors all share
/// one payload (offsets overlap from 0) holding `f(i)` at element i, so the binding runs without the 986 MB file.
fn syntheticCheckpoint(gpa: std.mem.Allocator, value: *const fn (u64) f32) ![]u8 {
    const fixture = @embedFile("testdata/vae_tensors.txt");
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.append(gpa, '{');
    var it = std.mem.splitScalar(u8, fixture[std.mem.indexOfScalar(u8, fixture, '\n').? + 1 ..], '\n');
    var first = true;
    var largest: u64 = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        var f = std.mem.splitScalar(u8, line, ' ');
        const name = f.next().?;
        _ = f.next();
        var elems: u64 = 1;
        var di = std.mem.splitScalar(u8, f.next().?, 'x');
        var shape: std.ArrayList(u8) = .empty;
        defer shape.deinit(gpa);
        while (di.next()) |d| {
            if (shape.items.len > 0) try shape.append(gpa, ',');
            try shape.appendSlice(gpa, d);
            elems *= try std.fmt.parseInt(u64, d, 10);
        }
        largest = @max(largest, elems);
        try json.print(gpa, "{s}\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{s}],\"data_offsets\":[0,{d}]}}", .{ if (first) "" else ",", name, shape.items, elems * 4 });
        first = false;
    }
    try json.appendSlice(gpa, ",\"__metadata__\":{\"format\":\"pt\"}}");
    const out = try gpa.alloc(u8, 8 + json.items.len + largest * 4);
    std.mem.writeInt(u64, out[0..8], json.items.len, .little);
    @memcpy(out[8..][0..json.items.len], json.items);
    const payload = std.mem.bytesAsSlice(f32, out[8 + json.items.len ..]);
    for (payload, 0..) |*v, i| v.* = value(i);
    return out;
}

fn indexValue(i: u64) f32 {
    return @floatFromInt(i % 2048);
}

test "build: every pinned tensor bound once, repacked, fp16, 256-aligned; relocation moves every address" {
    const gpa = testing.allocator;
    const file = try syntheticCheckpoint(gpa, indexValue);
    defer gpa.free(file);
    var h = try parseHeader(gpa, file);
    defer h.deinit();
    try testing.expectEqual(@as(usize, 248), h.entries.len);
    var img = try build(gpa, .{ .header = &h, .payload = file[h.data_start..] });
    defer img.deinit(gpa);

    // weights: 246,478,803 fp16 values plus the k padding of encoder.conv_in (81 -> 88 per row, 128 rows) and alignment
    try testing.expect(img.bytes.len >= 246_478_803 * 2 + 128 * 7 * 2);
    try testing.expect(img.bytes.len < 246_478_803 * 2 + 2_000_000);
    try testing.expectEqual(@as(u64, 0), img.enc.conv_in.w % 256);

    // encoder.conv_in.weight [128][3][3][3][3]: element (o, c, dt, dy, dx) of the file lands at row o, k = tap * 3 + c
    const first: u64 = 0; // every synthetic tensor starts at payload element 0
    const row = std.mem.bytesAsSlice(f16, img.bytes[img.enc.conv_in.w..][0 .. 88 * 2]);
    for (0..3) |c| for (0..27) |tap| {
        try testing.expectEqual(@as(f16, @floatCast(indexValue(first + c * 27 + tap))), row[tap * 3 + c]);
    };
    for (81..88) |k| try testing.expectEqual(@as(f16, 0), row[k]);
    // a linear: decoder to_q [512][512], ldw 512, rows copied in order
    const qrow = std.mem.bytesAsSlice(f16, img.bytes[img.dec.mid.attn.q.w + 512 * 2 ..][0 .. 512 * 2]);
    try testing.expectEqual(@as(f16, @floatCast(indexValue(512 + 7))), qrow[7]);
    // a GroupNorm bias
    const beta = std.mem.bytesAsSlice(f16, img.bytes[img.dec.norm_out.beta..][0..256]);
    try testing.expectEqual(@as(f16, @floatCast(indexValue(100))), beta[100]);

    var dec = img.dec;
    relocateDecoder(&dec, 0x7f00_0000_0000);
    try testing.expectEqual(img.dec.conv_out.w + 0x7f00_0000_0000, dec.conv_out.w);
    try testing.expectEqual(img.dec.up[3].resnets[0].shortcut.?.b + 0x7f00_0000_0000, dec.up[3].resnets[0].shortcut.?.b);
    try testing.expectEqual(img.dec.mid.attn.norm.gamma + 0x7f00_0000_0000, dec.mid.attn.norm.gamma);
    try testing.expectEqual(img.dec.conv_out.co, dec.conv_out.co);
}

test "build refuses a checkpoint with a missing or extra tensor" {
    const gpa = testing.allocator;
    const file = try syntheticCheckpoint(gpa, indexValue);
    defer gpa.free(file);
    var h = try parseHeader(gpa, file);
    defer h.deinit();
    const saved = h.entries;
    h.entries = saved[0 .. saved.len - 1];
    try testing.expectError(error.MissingTensor, build(gpa, .{ .header = &h, .payload = file[h.data_start..] }));
    h.entries = saved;
}
