//! One-frame video encoder, op-for-op with the twin. Canvas RGB fp32 -> normalized fp32 [24,1,H/16,W/16].
//! Tiles retain raw moments for both neighbors; decoder GEMM and spatial placement are shared unchanged.
const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const decoder = @import("vae_video.zig");
const tile_mod = @import("vae_video_tile.zig");
const weights = @import("vae_video_enc_load.zig");
pub const Ops = @import("vae_video_enc_ops.zig").Ops;
pub const Probe = decoder.Probe;
pub const Io = decoder.Io;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;
fn tensor(role: []const u8, ptr: u64, bytes: u64) Io {
    return .{ .role = role, .ptr = ptr, .bytes = bytes };
}
pub const Encoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Ops,
    shared: *const decoder.Ops,
    s: cuda.Stream,
    store: Store,
    w: weights.Weights,
    probe: ?Probe = null,
    select: ?[]const [2]i32 = null,
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const Ops, shared: *const decoder.Ops, s: cuda.Stream, path: []const u8, up: *Uploader) !Encoder {
        var store = Store.init(d, gpa);
        errdefer store.deinit();
        const w = try weights.load(gpa, io, path, &store, up);
        return .{ .gpa = gpa, .d = d, .k = k, .shared = shared, .s = s, .store = store, .w = w };
    }
    pub fn deinit(v: *Encoder) void {
        v.store.deinit();
    }
    pub fn weightBytes(v: *const Encoder) u64 {
        return v.store.bytes;
    }
    fn selected(v: *const Encoder, row: i32, col: i32) bool {
        if (row < 0 and col < 0) return true;
        const sel = v.select orelse return true;
        for (sel) |p| if (p[0] == row and p[1] == col) return true;
        return false;
    }
    /// Device addresses: RGB contiguous fp32 [H,W,3] in [0,1], output fp32 [24,1,H/16,W/16].
    /// All work uses s; scratch release synchronizes that stream before returning.
    pub fn encode(v: *Encoder, rgb: u64, H: u32, W: u32, out: u64) !void {
        if (H < 32 or W < 32 or H % 16 != 0 or W % 16 != 0) return error.BadImageCanvas;
        var sy = try decoder.splitTiles(v.gpa, H);
        defer sy.deinit(v.gpa);
        var sx = try decoder.splitTiles(v.gpa, W);
        defer sx.deinit(v.gpa);
        var scratch = Store.init(v.d, v.gpa);
        defer scratch.deinit();
        // Maximum activation is level zero: 256*256*128; im2col has nine taps per channel.
        const act_bytes = 256 * 256 * 128 * 2;
        var run: Run = .{ .v = v, .rec = .{ .probe = v.probe }, .buf = undefined, .col = try scratch.alloc(act_bytes * 9), .rgb = rgb, .H = H, .W = W };
        for (&run.buf) |*p| p.* = try scratch.alloc(act_bytes);
        const canvas = try scratch.alloc(48 * @as(u64, H / 16) * (W / 16) * 2);
        const tiles = try v.gpa.alloc(u64, 2 * sx.starts.len);
        defer v.gpa.free(tiles);
        for (tiles) |*p| p.* = try scratch.alloc(48 * 16 * 16 * 2);
        const region = if (v.probe != null) try scratch.alloc(48 * 16 * 16 * 2) else 0;
        // Even on a probe error, queued kernels finish before scratch is released.
        defer v.s.synchronize() catch {};
        var oy: u32 = 0;
        for (sy.starts, 0..) |y0, row| {
            var ox: u32 = 0;
            var oh: u32 = 0;
            for (sx.starts, 0..) |x0, col| {
                run.rec.setPrefix("venc.t{d}_{d}", .{ row, col });
                run.rec.on = v.selected(@intCast(row), @intCast(col));
                const current = tiles[(row & 1) * sx.starts.len + col];
                try run.tile(y0, x0, sy.lens[row], sx.lens[col], current);
                var place: decoder.Ops.Place = .{ .b = current, .frames = 16, .th = sy.lens[row] / 16, .tw = sx.lens[col] / 16, .canvas = canvas, .hc = H / 16, .wc = W / 16, .oy = oy, .ox = ox, .oh = sy.lens[row] / 16 - (if (row + 1 < sy.starts.len) sy.overlaps[row] / 16 else 0), .ow = sx.lens[col] / 16 - (if (col + 1 < sx.starts.len) sx.overlaps[col] / 16 else 0) };
                if (row > 0) {
                    place.ytail = tiles[((row + 1) & 1) * sx.starts.len + col];
                    place.tha = sy.lens[row - 1] / 16;
                    place.twa = sx.lens[col] / 16;
                    place.ey = sy.overlaps[row - 1] / 16;
                }
                if (col > 0) {
                    place.ltail = tiles[(row & 1) * sx.starts.len + col - 1];
                    place.thl = sy.lens[row] / 16;
                    place.twl = sx.lens[col - 1] / 16;
                    place.ex = sx.overlaps[col - 1] / 16;
                }
                var ins: [3]Io = undefined;
                var n: usize = 1;
                ins[0] = tensor("b", current, 48 * @as(u64, place.th) * place.tw * 2);
                if (place.ytail != 0) {
                    ins[n] = tensor("ytail", place.ytail, 48 * @as(u64, place.tha) * place.twa * 2);
                    n += 1;
                }
                if (place.ltail != 0) {
                    ins[n] = tensor("ltail", place.ltail, 48 * @as(u64, place.thl) * place.twl * 2);
                    n += 1;
                }
                try run.rec.pre("place", ins[0..n]);
                try v.shared.place(v.s, place);
                if (run.rec.on and v.probe != null) {
                    try v.s.synchronize();
                    var dst = region;
                    for (0..48) |c| for (0..place.oh) |y| {
                        const src = canvas + ((@as(u64, c) * (H / 16) + oy + y) * (W / 16) + ox) * 2;
                        try v.d.check(v.d.api.cuMemcpyDtoD_v2(dst, src, @as(u64, place.ow) * 2), "cuMemcpyDtoD");
                        dst += @as(u64, place.ow) * 2;
                    };
                    try run.rec.post("place", &.{tensor("y", region, 48 * @as(u64, place.oh) * place.ow * 2)});
                }
                ox += place.ow;
                oh = place.oh;
            }
            oy += oh;
        }
        run.rec.setPrefix("venc", .{});
        run.rec.on = v.selected(-1, -1);
        const P = (H / 16) * (W / 16);
        try run.rec.pre("latent", &.{ tensor("moments", canvas, 48 * @as(u64, P) * 2), tensor("mean", v.w.mean, 48), tensor("std", v.w.stdv, 48) });
        try v.k.latent(v.s, canvas, v.w.mean, v.w.stdv, out, P);
        try run.rec.post("latent", &.{tensor("y", out, 24 * @as(u64, P) * 4)});
        try v.s.synchronize();
    }
};
const Run = struct {
    v: *Encoder,
    rec: tile_mod.Rec,
    buf: [4]u64,
    col: u64,
    rgb: u64,
    H: u32,
    W: u32,
    fn conv(r: *Run, name: []const u8, w: weights.Conv, x: u64, out: u64, h: u32, width: u32, stride: u32, pad: u32) !void {
        const K = (w.ci * w.k * w.k + 7) / 8 * 8;
        const M = @as(u64, h / stride) * (width / stride);
        var tmp: [96]u8 = undefined;
        const im = try std.fmt.bufPrint(&tmp, "{s}.im2col", .{name});
        try r.rec.pre(im, &.{tensor("x", x, @as(u64, h) * width * w.ci * 2)});
        try r.v.k.im2col(r.v.s, x, r.col, h, width, w.ci, w.k, stride, if (w.k == 1) 0 else pad);
        try r.rec.post(im, &.{tensor("y", r.col, M * K * 2)});
        const gemm = try std.fmt.bufPrint(&tmp, "{s}.gemm", .{name});
        try r.rec.pre(gemm, &.{tensor("x", r.col, M * K * 2)});
        try r.v.shared.gemm(r.v.s, .{ .a = r.col, .b = w.w, .bias = w.b, .c = out, .m = M, .n = w.co, .k = K, .lda = K, .ldb = K, .ldc = w.co });
        try r.rec.post(gemm, &.{tensor("y", out, M * w.co * 2)});
    }
    fn norm(r: *Run, name: []const u8, w: weights.Norm, x: u64, out: u64, P: u32) !void {
        const bytes = @as(u64, P) * w.c * 2;
        var tmp: [96]u8 = undefined;
        const gn = try std.fmt.bufPrint(&tmp, "{s}.gn", .{name});
        try r.rec.pre(gn, &.{tensor("x", x, bytes)});
        try r.v.k.norm(r.v.s, x, w.w, w.b, out, w.c, P);
        try r.rec.post(gn, &.{tensor("y", out, bytes)});
        const silu = try std.fmt.bufPrint(&tmp, "{s}.silu", .{name});
        try r.rec.pre(silu, &.{tensor("x", out, bytes)});
        try r.v.k.silu(r.v.s, out, bytes / 2);
        try r.rec.post(silu, &.{tensor("y", out, bytes)});
    }
    fn advance(r: *Run) void {
        std.mem.swap(u64, &r.buf[0], &r.buf[2]);
    }
    fn tile(r: *Run, y0: u32, x0: u32, hi: u32, wi: u32, out: u64) !void {
        const v = r.v;
        var h = hi;
        var w = wi;
        try r.rec.pre("pixels", &.{tensor("rgb", r.rgb, @as(u64, r.H) * r.W * 3 * 4)});
        try v.k.pixels(v.s, r.rgb, r.buf[0], r.W, y0, x0, h, w);
        try r.rec.post("pixels", &.{tensor("y", r.buf[0], @as(u64, h) * w * 3 * 2)});
        try r.conv("conv_in", v.w.input, r.buf[0], r.buf[2], h, w, 1, 1);
        r.advance();
        for (v.w.blocks, 0..) |level, l| {
            for (level, 0..) |b, i| {
                var name: [64]u8 = undefined;
                const n1 = try std.fmt.bufPrint(&name, "L{d}.B{d}.norm1", .{ l, i });
                try r.norm(n1, b.n1, r.buf[0], r.buf[1], h * w);
                const c1 = try std.fmt.bufPrint(&name, "L{d}.B{d}.conv1", .{ l, i });
                try r.conv(c1, b.c1, r.buf[1], r.buf[2], h, w, 1, 1);
                // Match twin order: shortcut precedes norm2; buf[3] preserves it until the residual addition.
                var residual = r.buf[0];
                if (b.shortcut) |sc| {
                    const sn = try std.fmt.bufPrint(&name, "L{d}.B{d}.shortcut", .{ l, i });
                    try r.conv(sn, sc, r.buf[0], r.buf[3], h, w, 1, 0);
                    residual = r.buf[3];
                }
                const n2 = try std.fmt.bufPrint(&name, "L{d}.B{d}.norm2", .{ l, i });
                try r.norm(n2, b.n2, r.buf[2], r.buf[1], h * w);
                const c2 = try std.fmt.bufPrint(&name, "L{d}.B{d}.conv2", .{ l, i });
                try r.conv(c2, b.c2, r.buf[1], r.buf[2], h, w, 1, 1);
                const bytes = @as(u64, h) * w * b.c2.co * 2;
                const add = try std.fmt.bufPrint(&name, "L{d}.B{d}.add", .{ l, i });
                try r.rec.pre(add, &.{ tensor("x", r.buf[2], bytes), tensor("residual", residual, bytes) });
                try v.k.add(v.s, r.buf[2], residual, bytes / 2);
                try r.rec.post(add, &.{tensor("y", r.buf[2], bytes)});
                r.advance();
            }
            if (l < 4) {
                var name: [32]u8 = undefined;
                try r.conv(try std.fmt.bufPrint(&name, "L{d}.down", .{l}), v.w.down[l], r.buf[0], r.buf[2], h, w, 2, 0);
                h /= 2;
                w /= 2;
                r.advance();
            }
        }
        try r.norm("norm_out", v.w.norm_out, r.buf[0], r.buf[1], h * w);
        try r.conv("conv_out", v.w.output, r.buf[1], r.buf[2], h, w, 1, 1);
        try r.conv("quant", v.w.quant, r.buf[2], r.buf[1], h, w, 1, 0);
        const bytes = 48 * @as(u64, h) * w * 2;
        try r.rec.pre("moments", &.{tensor("x", r.buf[1], bytes)});
        try v.k.moments(v.s, r.buf[1], out, h * w);
        try r.rec.post("moments", &.{tensor("y", out, bytes)});
    }
};
