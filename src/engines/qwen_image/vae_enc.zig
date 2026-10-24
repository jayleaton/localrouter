//! Single-frame reference encoder; decoder arithmetic and scratch are shared, weights use Store/Uploader.
const std = @import("std");
const cuda = @import("cuda");
const vae = @import("vae.zig");
const Pack = @import("pack.zig").Pack;
const Store = @import("weights.zig").Store;
const Uploader = @import("upload.zig").Uploader;
const Shape = vae.Vae.Shape;
const block = 256;

pub const Ops = struct {
    module: cuda.Module,
    avg: cuda.Function,
    normalize: cuda.Function,
    from_u8: cuda.Function,
    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .avg = try m.function("stk_avg_down"), .normalize = try m.function("stk_chan_normalize"), .from_u8 = try m.function("stk_from_u8_rgba") };
    }
    pub fn unload(o: *Ops) void { o.module.unload(); }
};
const Down = struct { cin: u32, cout: u32, factor_t: u32, fs: u32, group_size: u32 };
const Manifest = struct {
    encoder: struct {
        latent_channels: u32, quant_channels: u32,
        convs: std.json.ArrayHashMap(struct { stride: u32, pad: [4]u32 }),
        norms: std.json.ArrayHashMap(f64),
        downs: std.json.ArrayHashMap(Down),
    },
};

pub const Encoder = struct {
    v: *vae.Vae,
    k: *const Ops,
    store: Store,
    downs: std.StringHashMapUnmanaged(Down) = .empty,
    mean: u64 = 0,
    stdv: u64 = 0,
    /// `v` outlives the encoder; its arena owns the metadata and its decoder pool is reused serially.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, v: *vae.Vae, k: *const Ops, p: *const Pack, up: *Uploader, dir: []const u8) !Encoder {
        var e: Encoder = .{ .v = v, .k = k, .store = .init(v.d, gpa) };
        errdefer e.deinit();
        const a = v.arena.allocator();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{dir, "manifest.json"}), a, .limited(64 << 20));
        const m = (try std.json.parseFromSliceLeaky(Manifest, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always })).encoder;
        if (m.latent_channels != 64 or m.quant_channels != 128) return error.UnsupportedEncoder;
        var nb: [192]u8 = undefined;
        var ci = m.convs.map.iterator();
        while (ci.next()) |entry| {
            const name = entry.key_ptr.*;
            const wt = try p.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{name}));
            if (wt.dtype != .bf16 or wt.rank != 4) return error.BadWeight;
            const cout: u32 = @intCast(wt.shape[0]);
            const cin: u32 = @intCast(wt.shape[1]);
            const kh: u32 = @intCast(wt.shape[2]);
            const kw: u32 = @intCast(wt.shape[3]);
            const kk = cin * kh * kw;
            const kp = (kk + 7) / 8 * 8;
            if (cout % 2 != 0 or entry.value_ptr.stride == 0) return error.UnsupportedConv;
            var w: u64 = undefined;
            if (kp == kk) {
                w = try e.store.tensorOf(up, p, wt);
            } else {
                const raw = try e.store.readBulk(io, p, wt);
                const padded = try padConvWeights(gpa, raw, cout, kk);
                defer gpa.free(padded);
                w = try e.store.upload(up, padded);
            }
            const bn = try std.fmt.bufPrint(&nb, "{s}.bias", .{name});
            const b = if (p.tensors.get(bn)) |bt| b: {
                if (bt.dtype != .bf16 or bt.len() != @as(usize, cout) * 2) return error.BadWeight;
                break :b try e.store.tensorOf(up, p, bt);
            } else 0;
            try v.convs.put(a, name, .{ .w = w, .b = b, .cin = cin, .cout = cout, .kh = kh, .kw = kw, .stride = entry.value_ptr.stride, .pad = entry.value_ptr.pad, .kpad = kp });
        }
        var ni = m.norms.map.iterator();
        while (ni.next()) |entry| {
            const t = try p.get(try std.fmt.bufPrint(&nb, "{s}.gamma", .{entry.key_ptr.*}));
            const channels = try normChannels(v, entry.key_ptr.*);
            if (t.dtype != .bf16 or t.rank != 1 or t.shape[0] != channels or t.len() != @as(usize, channels) * 2 or !std.math.isFinite(entry.value_ptr.*)) return error.BadWeight;
            try v.norms.put(a, entry.key_ptr.*, .{ .gamma = try e.store.tensorOf(up, p, t), .c = @intCast(t.shape[0]), .scale = @floatCast(entry.value_ptr.*) });
        }
        var di = m.downs.map.iterator();
        while (di.next()) |entry| {
            const d = entry.value_ptr.*;
            if (d.cout == 0 or d.fs == 0 or d.factor_t == 0 or @as(u64, d.cin) * d.factor_t * d.fs * d.fs != @as(u64, d.group_size) * d.cout) return error.UnsupportedEncoder;
            try e.downs.put(a, entry.key_ptr.*, d);
        }
        const mean = try p.get("vae.enc_latents_mean");
        const stdv = try p.get("vae.enc_latents_std");
        if (mean.dtype != .bf16 or stdv.dtype != .bf16 or mean.rank != 1 or stdv.rank != 1 or mean.shape[0] != 64 or stdv.shape[0] != 64 or mean.len() != 128 or stdv.len() != 128) return error.BadWeight;
        e.mean = try e.store.tensorOf(up, p, mean);
        e.stdv = try e.store.tensorOf(up, p, stdv);
        return e;
    }
    pub fn deinit(e: *Encoder) void { e.store.deinit(); }

    fn avg(e: *Encoder, name: []const u8, x: u64, sh: Shape, y: u64) !Shape {
        const d = e.downs.get(name) orelse return error.MissingDown;
        if (sh.c != d.cin or sh.h % d.fs != 0 or sh.w % d.fs != 0) return error.ShapeMismatch;
        const out: Shape = .{ .c = d.cout, .h = sh.h / d.fs, .w = sh.w / d.fs };
        try e.v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = vae.Vae.bytesOf(sh) }});
        var args: cuda.launch.Args = .{};
        args.add(x); args.add(y);
        inline for (.{ sh.c, sh.h, sh.w, d.cout, d.factor_t, d.fs }) |n| args.add(@as(i32, @intCast(n)));
        try vae.Vae.go(e.k.avg, e.v.s, .{ .x = @intCast((vae.Vae.bytesOf(out) / 2 + block - 1) / block) }, .{ .x = block }, &args);
        try e.v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = vae.Vae.bytesOf(out) }});
        return out;
    }

    // ZeroPad2d is an unrecorded view/data move in the twin; preserve the padded conv input for Probe.before.
    fn padDown(e: *Encoder, x: u64, sh: Shape, y: u64) !Shape {
        const out: Shape = .{ .c = sh.c, .h = sh.h + 1, .w = sh.w + 1 };
        if (vae.Vae.bytesOf(out) > e.v.act_bytes) return error.ImageTooLarge;
        try e.v.s.synchronize();
        try e.v.d.check(e.v.d.api.cuMemsetD8_v2(y, 0, vae.Vae.bytesOf(out)), "cuMemsetD8");
        for (0..sh.c) |c| for (0..sh.h) |r| {
            const src = x + (c * sh.h + r) * sh.w * 2;
            const dst = y + (c * out.h + r) * out.w * 2;
            try e.v.d.check(e.v.d.api.cuMemcpyDtoD_v2(dst, src, @as(u64, sh.w) * 2), "cuMemcpyDtoD");
        };
        return out;
    }

    fn downBlock(e: *Encoder, i: usize, xi: usize, sh: Shape) !struct { usize, Shape } {
        const v = e.v;
        var name_buf: [80]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "vae.encoder.down_blocks.{d}", .{i});
        var nb: [192]u8 = undefined;
        var cur = xi;
        var cs = sh;
        var j: usize = 0;
        while (v.convs.contains(try std.fmt.bufPrint(&nb, "{s}.resnets.{d}.conv1", .{name, j}))) : (j += 1) {
            const r = try v.resBlock(try std.fmt.bufPrint(&nb, "{s}.resnets.{d}", .{name, j}), cur, cs);
            if (cur != xi) v.give(cur);
            cur = r[0]; cs = r[1];
        }
        if (cur == xi) return error.UnsupportedEncoder;
        const dn = try std.fmt.bufPrint(&nb, "{s}.downsampler.resample.1", .{name});
        if (v.convs.contains(dn)) {
            const pi = try v.take();
            const padded = try e.padDown(v.pool[cur], cs, v.pool[pi]);
            cs = try v.conv(dn, v.pool[pi], padded, v.pool[cur]);
            v.give(pi);
        }
        const ai = try v.take();
        const ash = try e.avg(try std.fmt.bufPrint(&nb, "{s}.avg_shortcut", .{name}), v.pool[xi], sh, v.pool[ai]);
        if (ash.c != cs.c or ash.h != cs.h or ash.w != cs.w) return error.ShapeMismatch;
        try v.add(try std.fmt.bufPrint(&nb, "{s}.shortcut_add", .{name}), v.pool[cur], v.pool[ai], v.pool[cur], vae.Vae.bytesOf(cs));
        v.give(ai); v.give(xi);
        return .{cur, cs};
    }

    /// CUDA RGBA u8 HWC -> normalized bf16 CHW [64,H/16,W/16], posterior mean only.
    pub fn encode(e: *Encoder, rgba: u64, h: u32, w: u32, out: u64) !void {
        const v = e.v;
        if (h == 0 or w == 0 or h % 16 != 0 or w % 32 != 0) return error.UnsupportedReferenceSize;
        if (h / 16 > v.max_h or w / 16 > v.max_w) return error.ImageTooLarge;
        @memset(&v.busy, false);
        defer @memset(&v.busy, false);
        const sh: Shape = .{ .c = 4, .h = h, .w = w };
        const ii = try v.take();
        const hw: u64 = @as(u64, h) * w;
        try v.pre("vae.from_u8", &.{.{ .role = "x", .ptr = rgba, .bytes = hw * 4 }});
        var args: cuda.launch.Args = .{};
        args.add(rgba); args.add(v.pool[ii]); args.add(@as(i32, @intCast(h))); args.add(@as(i32, @intCast(w)));
        try vae.Vae.go(e.k.from_u8, v.s, .{ .x = @intCast((hw * 4 + block - 1) / block) }, .{ .x = block }, &args);
        try v.post("vae.from_u8", &.{.{ .role = "y", .ptr = v.pool[ii], .bytes = vae.Vae.bytesOf(sh) }});
        var cur = try v.take();
        var cs = try v.conv("vae.encoder.conv_in", v.pool[ii], sh, v.pool[cur]);
        v.give(ii);
        var i: usize = 0;
        var nb: [128]u8 = undefined;
        while (v.convs.contains(try std.fmt.bufPrint(&nb, "vae.encoder.down_blocks.{d}.resnets.0.conv1", .{i}))) : (i += 1) {
            const r = try e.downBlock(i, cur, cs); cur = r[0]; cs = r[1];
        }
        const r0 = try v.resBlock("vae.encoder.mid_block.resnets.0", cur, cs); v.give(cur);
        const ai = try v.attention("vae.encoder.mid_block.attentions.0", r0[0], r0[1]); v.give(r0[0]);
        const r1 = try v.resBlock("vae.encoder.mid_block.resnets.1", ai, r0[1]); v.give(ai);
        cur = r1[0]; cs = r1[1];
        try v.norm("vae.encoder.norm_out", v.pool[cur], cs, v.pool[cur]);
        try v.silu("vae.encoder.act_out", v.pool[cur], cs);
        const oi = try v.take();
        const os = try v.conv("vae.encoder.conv_out", v.pool[cur], cs, v.pool[oi]); v.give(cur);
        const qi = try v.take();
        const qs = try v.conv("vae.quant_conv", v.pool[oi], os, v.pool[qi]); v.give(oi);
        if (qs.c != 128 or qs.h != h / 16 or qs.w != w / 16) return error.ShapeMismatch;
        const elems: u64 = @as(u64, 64) * qs.h * qs.w;
        try v.pre("vae.normalize", &.{ .{ .role = "x", .ptr = v.pool[qi], .bytes = elems * 2 }, .{ .role = "mean", .ptr = e.mean, .bytes = 128 }, .{ .role = "std", .ptr = e.stdv, .bytes = 128 } });
        args = .{};
        inline for (.{v.pool[qi], e.mean, e.stdv, out}) |ptr| args.add(ptr);
        args.add(@as(i64, @intCast(@as(u64, qs.h) * qs.w))); args.add(@as(i64, @intCast(elems)));
        try vae.Vae.go(e.k.normalize, v.s, .{ .x = @intCast((elems + block - 1) / block) }, .{ .x = block }, &args);
        try v.post("vae.normalize", &.{.{ .role = "y", .ptr = out, .bytes = elems * 2 }});
    }
};

// The twin expands each convolution row independently to ceil(K/8), rather than appending a tensor-wide tail.
fn padConvWeights(gpa: std.mem.Allocator, raw: []const u8, cout: u32, kk: u32) ![]u8 {
    if (cout == 0 or kk == 0 or raw.len != @as(usize, cout) * kk * 2) return error.BadWeight;
    const kp = (kk + 7) / 8 * 8;
    const padded = try gpa.alloc(u8, @as(usize, cout) * kp * 2);
    @memset(padded, 0);
    for (0..cout) |i| @memcpy(padded[i * kp * 2 ..][0 .. kk * 2], raw[i * kk * 2 ..][0 .. kk * 2]);
    return padded;
}

fn normChannels(v: *const vae.Vae, name: []const u8) !u32 {
    var nb: [192]u8 = undefined;
    const pairs = .{ .{ ".norm1", ".conv1" }, .{ ".norm2", ".conv2" }, .{ ".norm_out", ".conv_out" }, .{ ".norm", ".to_qkv" } };
    inline for (pairs) |pair| {
        if (std.mem.endsWith(u8, name, pair[0])) {
            const key = try std.fmt.bufPrint(&nb, "{s}{s}", .{ name[0 .. name.len - pair[0].len], pair[1] });
            return (v.convs.get(key) orelse return error.MissingConv).cin;
        }
    }
    return error.UnsupportedEncoderNorm;
}

test "encoder input convolution pads each 36-element row to 40 without shifting channels" {
    var raw: [2 * 36 * 2]u8 = undefined;
    for (&raw, 0..) |*b, i| b.* = @intCast(i + 1);
    const padded = try padConvWeights(std.testing.allocator, &raw, 2, 36);
    defer std.testing.allocator.free(padded);
    try std.testing.expectEqual(@as(usize, 160), padded.len);
    try std.testing.expectEqualSlices(u8, raw[0..72], padded[0..72]);
    try std.testing.expectEqualSlices(u8, raw[72..144], padded[80..152]);
    try std.testing.expectEqualSlices(u8, &@as([8]u8, @splat(0)), padded[72..80]);
    try std.testing.expectEqualSlices(u8, &@as([8]u8, @splat(0)), padded[152..160]);
    try std.testing.expectError(error.BadWeight, padConvWeights(std.testing.allocator, raw[0..143], 2, 36));
}
