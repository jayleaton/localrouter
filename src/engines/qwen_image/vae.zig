//! The VAE decoder (diffusers' AutoencoderKLQwenImage21, one frame), op for op as the twin (`stk_twin/vae_ops.py`)
//! runs it, with the twin's op names (tools/twin/VAE-PORT.md): latents [64, h, w] -> the image's uint8 pixels
//! [16h, 16w, C] (HWC). Convolutions are im2col + the bf16 GEMM (gemm.cu) in bands of output rows (a GEMM row's bits do
//! not depend on the band), everything else vae.cu and ops.cu's silu. Weights from the VAE's pack (`pack.py write_vae`).

const std = @import("std");
const cuda = @import("cuda");
const Pack = @import("pack.zig").Pack;
const upload_mod = @import("upload.zig");
const ops_launch = @import("ops_launch.zig");
const dit = @import("dit.zig");
const Io = dit.Io;
const Probe = dit.Probe;

const Ptr = u64;
const block: u32 = 256;
const band_bytes: u64 = 512 << 20; // im2col columns a band, as the twin's default
const z_dim = 64;

fn blocks(n: u64) u32 {
    return @intCast((n + block - 1) / block);
}

pub const VaeOps = struct {
    module: cuda.Module,
    im2col: cuda.Function,
    transpose: cuda.Function,
    rms: cuda.Function,
    add: cuda.Function,
    up2: cuda.Function,
    dup: cuda.Function,
    softmax: cuda.Function,
    affine: cuda.Function,
    to_u8: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !VaeOps {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{
            .module = m,
            .im2col = try m.function("stk_im2col"),
            .transpose = try m.function("stk_transpose_bf16"),
            .rms = try m.function("stk_channel_rms_norm"),
            .add = try m.function("stk_add_bf16"),
            .up2 = try m.function("stk_upsample_nearest2x"),
            .dup = try m.function("stk_dup_up"),
            .softmax = try m.function("stk_softmax_rows"),
            .affine = try m.function("stk_chan_affine"),
            .to_u8 = try m.function("stk_to_u8_hwc"),
        };
    }

    pub fn unload(o: *VaeOps) void {
        o.module.unload();
    }
};

/// One conv: weight [cout, k] (k = cin * kh * kw, a multiple of 8), bias [cout].
const Conv = struct { w: Ptr, b: Ptr, cin: u32, cout: u32, kh: u32, kw: u32, stride: u32, pad: [4]u32 };
const Norm = struct { gamma: Ptr, c: u32, scale: f32 };
const Dup = struct { cin: u32, cout: u32, fs: u32, factor: u32, repeats: u32, fti: u32 };

const Manifest = struct {
    convs: std.json.ArrayHashMap(struct { stride: u32, pad: [4]u32 }),
    norms: std.json.ArrayHashMap(f64),
    dups: std.json.ArrayHashMap(struct { cin: u32, cout: u32, fs: u32, factor: u32, repeats: u32, fti: u32 }),
    out_channels: u32,
};

pub const Vae = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    d: *const cuda.Driver,
    k: *const VaeOps,
    tk: *const ops_launch.TeOps,
    ops: *const ops_launch.Ops,
    s: cuda.Stream,
    probe: ?Probe = null,
    mem: upload_mod.Slab,
    convs: std.StringHashMapUnmanaged(Conv) = .empty,
    norms: std.StringHashMapUnmanaged(Norm) = .empty,
    dups: std.StringHashMapUnmanaged(Dup) = .empty,
    mean: Ptr = 0,
    stdv: Ptr = 0,
    out_channels: u32 = 0,
    // activations: a pool of equal buffers (the largest activation of the largest image), and the im2col band
    pool: [7]Ptr = @splat(0),
    busy: [7]bool = @splat(false),
    act_bytes: u64 = 0,
    cols: Ptr = 0,
    pm: Ptr = 0,
    max_h: u32,
    max_w: u32,

    /// Weights from the pack `p` and scratch for latents up to max_h x max_w (the image is 16x that).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const VaeOps, tk: *const ops_launch.TeOps, ops: *const ops_launch.Ops, s: cuda.Stream, p: *const Pack, up: *upload_mod.Uploader, dir: []const u8, max_h: u32, max_w: u32) !Vae {
        var v: Vae = .{ .gpa = gpa, .arena = .init(gpa), .d = d, .k = k, .tk = tk, .ops = ops, .s = s, .max_h = max_h, .max_w = max_w, .mem = .init(d, gpa) };
        errdefer v.deinit();
        const a = v.arena.allocator();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(64 << 20));
        const m = try std.json.parseFromSliceLeaky(Manifest, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        v.out_channels = m.out_channels;
        var nb: [160]u8 = undefined;
        var it = m.convs.map.iterator();
        while (it.next()) |e| {
            const wt = try p.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{e.key_ptr.*}));
            if (wt.rank != 4 or wt.dtype != .bf16) return error.BadWeight;
            const c: Conv = .{
                .w = try v.tensor(up, p, wt),
                .b = try v.tensor(up, p, try p.get(try std.fmt.bufPrint(&nb, "{s}.bias", .{e.key_ptr.*}))),
                .cout = @intCast(wt.shape[0]),
                .cin = @intCast(wt.shape[1]),
                .kh = @intCast(wt.shape[2]),
                .kw = @intCast(wt.shape[3]),
                .stride = e.value_ptr.stride,
                .pad = e.value_ptr.pad,
            };
            if ((c.cin * c.kh * c.kw) % 8 != 0 or c.cout % 2 != 0) return error.UnsupportedConv; // the twin pads; none here
            try v.convs.put(a, e.key_ptr.*, c);
        }
        var ni = m.norms.map.iterator();
        while (ni.next()) |e| {
            const g = try p.get(try std.fmt.bufPrint(&nb, "{s}.gamma", .{e.key_ptr.*}));
            try v.norms.put(a, e.key_ptr.*, .{ .gamma = try v.tensor(up, p, g), .c = @intCast(g.shape[0]), .scale = @floatCast(e.value_ptr.*) });
        }
        var di = m.dups.map.iterator();
        while (di.next()) |e| {
            const x = e.value_ptr.*;
            try v.dups.put(a, e.key_ptr.*, .{ .cin = x.cin, .cout = x.cout, .fs = x.fs, .factor = x.factor, .repeats = x.repeats, .fti = x.fti });
        }
        v.mean = try v.tensor(up, p, try p.get("vae.latents_mean"));
        v.stdv = try v.tensor(up, p, try p.get("vae.latents_std"));

        // the largest activation: C x (16h x 16w) over the decoder's stages is 288 x 16h x 16w (block 3's upsample
        // and block 4's input); the attention's [L, L] scores at L = h * w; the decoder's widest stage 1152 x h x w x 4
        const hw: u64 = @as(u64, max_h) * max_w;
        v.act_bytes = 2 * @max(@max(288 * 256 * hw, hw * hw + 8 * hw), 1152 * 16 * hw);
        for (&v.pool) |*b| b.* = try v.alloc(v.act_bytes);
        // a band of columns (or one output row of the widest conv), and its GEMM output: cout / k <= 2 (conv_in)
        const row: u64 = 16 * @as(u64, max_w) * 1152 * 9 * 2;
        v.cols = try v.alloc(band_bytes + row);
        v.pm = try v.alloc(2 * (band_bytes + row));
        return v;
    }

    pub fn deinit(v: *Vae) void {
        v.mem.deinit();
        v.arena.deinit();
    }

    fn alloc(v: *Vae, n: u64) !Ptr {
        return v.mem.alloc(@intCast(n));
    }

    fn tensor(v: *Vae, up: *upload_mod.Uploader, p: *const Pack, t: @import("pack.zig").Tensor) !Ptr {
        if (t.dtype != .bf16) return error.BadWeight;
        const ptr = try v.alloc(t.len());
        try up.tensor(p, t, ptr);
        return ptr;
    }

    // ------------------------------------------------------------------ the activation pool
    fn take(v: *Vae) !usize {
        for (&v.busy, 0..) |*b, i| if (!b.*) {
            b.* = true;
            return i;
        };
        return error.VaePoolExhausted;
    }
    fn give(v: *Vae, i: usize) void {
        v.busy[i] = false;
    }

    fn pre(v: *Vae, name: []const u8, ins: []const Io) !void {
        if (v.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(v: *Vae, name: []const u8, outs: []const Io) !void {
        if (v.probe) |p| try p.after(p.ctx, name, outs);
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, blk: cuda.launch.Dim3, args: *cuda.launch.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = blk, .shared = 0 }, s, args);
    }

    /// y[c * ldy + off + r] = x[r * ldx + c] for r < rows, c < cols (stk_transpose_bf16).
    fn transposeRaw(v: *Vae, x: Ptr, rows: u64, cols: u64, ldx: u64, y: Ptr, ldy: u64) !void {
        var args: cuda.launch.Args = .{};
        args.add(x);
        args.add(y);
        inline for (.{ rows, cols, ldx, ldy }) |n| args.add(@as(i64, @intCast(n)));
        try go(v.k.transpose, v.s, .{ .x = @intCast((rows + 31) / 32), .y = @intCast((cols + 31) / 32) }, .{ .x = 32, .y = 8 }, &args);
    }

    // ------------------------------------------------------------------ ops (each one twin op)
    const Shape = struct { c: u32, h: u32, w: u32 };

    fn bytesOf(sh: Shape) u64 {
        return @as(u64, sh.c) * sh.h * sh.w * 2;
    }

    fn conv(v: *Vae, name: []const u8, x: Ptr, sh: Shape, y: Ptr) !Shape {
        const c = v.convs.get(name) orelse return error.MissingConv;
        if (c.cin != sh.c) return error.ShapeMismatch;
        const ho = (sh.h + c.pad[0] + c.pad[1] - c.kh) / c.stride + 1;
        const wo = (sh.w + c.pad[2] + c.pad[3] - c.kw) / c.stride + 1;
        const out: Shape = .{ .c = c.cout, .h = ho, .w = wo };
        try v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = bytesOf(sh) }});
        const kpad: u64 = @as(u64, c.cin) * c.kh * c.kw;
        const rows: u64 = @max(1, band_bytes / (@as(u64, wo) * kpad * 2));
        var y0: u64 = 0;
        while (y0 < ho) : (y0 += rows) {
            const r = @min(rows, ho - y0);
            const np = r * wo;
            var args: cuda.launch.Args = .{};
            args.add(x);
            args.add(v.cols);
            inline for (.{ sh.c, sh.h, sh.w, c.kh, c.kw, c.stride, c.pad[0], c.pad[1], c.pad[2], c.pad[3], 1, kpad }) |n| args.add(@as(i32, @intCast(n)));
            args.add(@as(i64, @intCast(y0 * wo)));
            args.add(@as(i64, @intCast(np)));
            try go(v.k.im2col, v.s, .{ .x = blocks(np * kpad) }, .{ .x = block }, &args);
            try v.tk.gemm(v.s, v.cols, c.w, c.b, v.pm, np, c.cout, kpad);
            try v.transposeRaw(v.pm, np, c.cout, c.cout, y + y0 * wo * 2, @as(u64, ho) * wo);
        }
        try v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = bytesOf(out) }});
        return out;
    }

    /// y may alias x.
    fn norm(v: *Vae, name: []const u8, x: Ptr, sh: Shape, y: Ptr) !void {
        const n = v.norms.get(name) orelse return error.MissingNorm;
        if (n.c != sh.c) return error.ShapeMismatch;
        const hw: u64 = @as(u64, sh.h) * sh.w;
        try v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = bytesOf(sh) }});
        var args: cuda.launch.Args = .{};
        inline for (.{ x, n.gamma, y }) |p| args.add(p);
        args.add(@as(i32, @intCast(sh.c)));
        args.add(@as(i64, @intCast(hw)));
        args.add(n.scale);
        try go(v.k.rms, v.s, .{ .x = blocks(hw) }, .{ .x = block }, &args);
        try v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = bytesOf(sh) }});
    }

    fn silu(v: *Vae, name: []const u8, x: Ptr, sh: Shape) !void {
        try v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = bytesOf(sh) }});
        try v.ops.pointwise(v.s, .silu, x, x, bytesOf(sh) / 2);
        try v.post(name, &.{.{ .role = "y", .ptr = x, .bytes = bytesOf(sh) }});
    }

    /// out = x + z (out may alias either); the twin's roles: inputs x, y; output y.
    fn add(v: *Vae, name: []const u8, x: Ptr, z: Ptr, out: Ptr, bytes: u64) !void {
        try v.pre(name, &.{ .{ .role = "x", .ptr = x, .bytes = bytes }, .{ .role = "y", .ptr = z, .bytes = bytes } });
        var args: cuda.launch.Args = .{};
        inline for (.{ x, z, out }) |p| args.add(p);
        args.add(@as(i64, @intCast(bytes / 2)));
        try go(v.k.add, v.s, .{ .x = blocks(bytes / 2) }, .{ .x = block }, &args);
        try v.post(name, &.{.{ .role = "y", .ptr = out, .bytes = bytes }});
    }

    fn gemmOp(v: *Vae, name: []const u8, a: Ptr, b: Ptr, bias: Ptr, out: Ptr, m: u64, n: u64, kk: u64, b_is_act: bool) !void {
        if (b_is_act) {
            try v.pre(name, &.{ .{ .role = "a", .ptr = a, .bytes = m * kk * 2 }, .{ .role = "b", .ptr = b, .bytes = n * kk * 2 } });
        } else try v.pre(name, &.{.{ .role = "a", .ptr = a, .bytes = m * kk * 2 }});
        try v.tk.gemm(v.s, a, b, bias, out, m, n, kk);
        try v.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * n * 2 }});
    }

    /// x [rows, cols] -> y [cols, ld] (ld >= rows; columns past rows zero).
    fn transposeOp(v: *Vae, name: []const u8, x: Ptr, rows: u64, cols: u64, y: Ptr, ld: u64) !void {
        try v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = rows * cols * 2 }});
        if (ld > rows) try v.d.check(v.d.api.cuMemsetD8_v2(y, 0, cols * ld * 2), "cuMemsetD8");
        try v.transposeRaw(x, rows, cols, cols, y, ld);
        try v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = cols * ld * 2 }});
    }

    // ------------------------------------------------------------------ blocks
    /// QwenImage21ResidualBlock: x (pool slot xi) -> a new slot (returned); xi is left as it was.
    fn resBlock(v: *Vae, name: []const u8, xi: usize, sh: Shape) !struct { usize, Shape } {
        var nb: [160]u8 = undefined;
        const x = v.pool[xi];
        var hi: ?usize = null; // the shortcut's slot, when there is a conv
        defer if (hi) |i| v.give(i);
        var h = x;
        const sc = try std.fmt.bufPrint(&nb, "{s}.conv_shortcut", .{name});
        if (v.convs.contains(sc)) {
            hi = try v.take();
            h = v.pool[hi.?];
            _ = try v.conv(sc, x, sh, h);
        }
        const ai = try v.take();
        const bi = try v.take();
        defer v.give(bi);
        const a = v.pool[ai];
        const b = v.pool[bi];
        try v.norm(try std.fmt.bufPrint(&nb, "{s}.norm1", .{name}), x, sh, a);
        try v.silu(try std.fmt.bufPrint(&nb, "{s}.act1", .{name}), a, sh);
        const sh1 = try v.conv(try std.fmt.bufPrint(&nb, "{s}.conv1", .{name}), a, sh, b);
        try v.norm(try std.fmt.bufPrint(&nb, "{s}.norm2", .{name}), b, sh1, b);
        try v.silu(try std.fmt.bufPrint(&nb, "{s}.act2", .{name}), b, sh1);
        const sh2 = try v.conv(try std.fmt.bufPrint(&nb, "{s}.conv2", .{name}), b, sh1, a);
        try v.add(try std.fmt.bufPrint(&nb, "{s}.add", .{name}), a, h, a, bytesOf(sh2));
        return .{ ai, sh2 };
    }

    /// The mid block's attention: one head over the h * w pixels, channel dim C. x (slot xi) -> a new slot.
    fn attention(v: *Vae, name: []const u8, xi: usize, sh: Shape) !usize {
        var nb: [160]u8 = undefined;
        const C: u64 = sh.c;
        const L: u64 = @as(u64, sh.h) * sh.w;
        const lpad = (L + 7) / 8 * 8;
        const nq = try std.fmt.bufPrint(&nb, "{s}.to_qkv", .{name});
        const qkv = v.convs.get(nq) orelse return error.MissingConv;
        const proj = v.convs.get(try std.fmt.bufPrint(&nb, "{s}.proj", .{name})) orelse return error.MissingConv;
        var slots: [5]usize = undefined;
        for (&slots) |*i| i.* = try v.take();
        defer for (slots[1..]) |i| v.give(i);
        const xn = v.pool[slots[0]]; // reused for the output
        const xt = v.pool[slots[1]];
        const q = v.pool[slots[2]];
        const kk = v.pool[slots[3]];
        const vv = v.pool[slots[4]];
        try v.norm(try std.fmt.bufPrint(&nb, "{s}.norm", .{name}), v.pool[xi], sh, xn);
        try v.transposeOp(try std.fmt.bufPrint(&nb, "{s}.xt", .{name}), xn, C, L, xt, C);
        const wrow = C * C * 2; // one C x C slice of to_qkv's weight
        try v.gemmOp(try std.fmt.bufPrint(&nb, "{s}.q", .{name}), xt, qkv.w, qkv.b, q, L, C, C, false);
        try v.gemmOp(try std.fmt.bufPrint(&nb, "{s}.k", .{name}), xt, qkv.w + wrow, qkv.b + C * 2, kk, L, C, C, false);
        try v.gemmOp(try std.fmt.bufPrint(&nb, "{s}.v", .{name}), xt, qkv.w + 2 * wrow, qkv.b + 2 * C * 2, vv, L, C, C, false);
        // vt [C, Lpad] into xt's slot (xt is done), scores S [L, L] into xn's, P [L, Lpad] into vv's
        try v.transposeOp(try std.fmt.bufPrint(&nb, "{s}.vt", .{name}), vv, L, C, xt, lpad);
        try v.gemmOp(try std.fmt.bufPrint(&nb, "{s}.scores", .{name}), q, kk, 0, xn, L, L, C, true);
        const sm = try std.fmt.bufPrint(&nb, "{s}.softmax", .{name});
        try v.pre(sm, &.{.{ .role = "s", .ptr = xn, .bytes = L * L * 2 }});
        {
            var args: cuda.launch.Args = .{};
            args.add(xn);
            args.add(vv);
            args.add(@as(i64, @intCast(L)));
            args.add(@as(i64, @intCast(lpad)));
            args.add(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(C))))));
            try go(v.k.softmax, v.s, .{ .x = @intCast(L) }, .{ .x = block }, &args);
        }
        try v.post(sm, &.{.{ .role = "y", .ptr = vv, .bytes = L * lpad * 2 }});
        try v.gemmOp(try std.fmt.bufPrint(&nb, "{s}.pv", .{name}), vv, xt, 0, q, L, C, lpad, true);
        try v.gemmOp(try std.fmt.bufPrint(&nb, "{s}.proj", .{name}), q, proj.w, proj.b, kk, L, C, C, false);
        try v.transposeOp(try std.fmt.bufPrint(&nb, "{s}.proj_t", .{name}), kk, L, C, xn, L);
        try v.add(try std.fmt.bufPrint(&nb, "{s}.add", .{name}), xn, v.pool[xi], xn, C * L * 2);
        return slots[0];
    }

    /// QwenImage21ResidualUpBlock i: x (slot xi, released) -> a new slot.
    fn upBlock(v: *Vae, i: usize, xi: usize, sh: Shape) !struct { usize, Shape } {
        var nb: [160]u8 = undefined;
        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "vae.decoder.up_blocks.{d}", .{i});
        var cur = xi;
        var cs = sh;
        for (0..3) |j| {
            const r = try v.resBlock(try std.fmt.bufPrint(&nb, "{s}.resnets.{d}", .{ name, j }), cur, cs);
            if (cur != xi) v.give(cur);
            cur = r[0];
            cs = r[1];
        }
        const up_conv = try std.fmt.bufPrint(&nb, "{s}.upsampler.resample.1", .{name});
        if (!v.convs.contains(up_conv)) {
            v.give(xi);
            return .{ cur, cs };
        }
        // upsample (resample.0), then the conv (resample.1)
        const ui = try v.take();
        const up_name = try std.fmt.bufPrint(&nb, "{s}.upsampler.resample.0", .{name});
        const us: Shape = .{ .c = cs.c, .h = 2 * cs.h, .w = 2 * cs.w };
        try v.pre(up_name, &.{.{ .role = "x", .ptr = v.pool[cur], .bytes = bytesOf(cs) }});
        {
            var args: cuda.launch.Args = .{};
            args.add(v.pool[cur]);
            args.add(v.pool[ui]);
            inline for (.{ cs.c, cs.h, cs.w }) |n| args.add(@as(i32, @intCast(n)));
            try go(v.k.up2, v.s, .{ .x = blocks(bytesOf(us) / 2) }, .{ .x = block }, &args);
        }
        try v.post(up_name, &.{.{ .role = "y", .ptr = v.pool[ui], .bytes = bytesOf(us) }});
        const cs2 = try v.conv(try std.fmt.bufPrint(&nb, "{s}.upsampler.resample.1", .{name}), v.pool[ui], us, v.pool[cur]);
        // the DupUp shortcut of the block's input into ui, then the add
        const dn = try std.fmt.bufPrint(&nb, "{s}.avg_shortcut", .{name});
        const du = v.dups.get(dn) orelse return error.MissingDup;
        const ds: Shape = .{ .c = du.cout, .h = sh.h * du.fs, .w = sh.w * du.fs };
        try v.pre(dn, &.{.{ .role = "x", .ptr = v.pool[xi], .bytes = bytesOf(sh) }});
        {
            var args: cuda.launch.Args = .{};
            args.add(v.pool[xi]);
            args.add(v.pool[ui]);
            inline for (.{ du.cout, sh.h, sh.w, du.fs, du.factor, du.repeats, du.fti }) |n| args.add(@as(i32, @intCast(n)));
            try go(v.k.dup, v.s, .{ .x = blocks(bytesOf(ds) / 2) }, .{ .x = block }, &args);
        }
        try v.post(dn, &.{.{ .role = "y", .ptr = v.pool[ui], .bytes = bytesOf(ds) }});
        if (ds.c != cs2.c or ds.h != cs2.h or ds.w != cs2.w) return error.ShapeMismatch;
        try v.add(try std.fmt.bufPrint(&nb, "{s}.shortcut_add", .{name}), v.pool[cur], v.pool[ui], v.pool[cur], bytesOf(cs2));
        v.give(ui);
        v.give(xi);
        return .{ cur, cs2 };
    }

    /// Normalised latents `lat` [64, h, w] (the sampler's) -> uint8 pixels `out` [16h, 16w, out_channels] (HWC).
    pub fn decode(v: *Vae, lat: Ptr, h: u32, w: u32, out: Ptr) !void {
        if (h > v.max_h or w > v.max_w) return error.ImageTooLarge;
        @memset(&v.busy, false);
        const zs: Shape = .{ .c = z_dim, .h = h, .w = w };
        const zi = try v.take();
        const hw: u64 = @as(u64, h) * w;
        try v.pre("vae.denorm", &.{ .{ .role = "x", .ptr = lat, .bytes = bytesOf(zs) }, .{ .role = "mean", .ptr = v.mean, .bytes = z_dim * 2 }, .{ .role = "std", .ptr = v.stdv, .bytes = z_dim * 2 } });
        {
            var args: cuda.launch.Args = .{};
            inline for (.{ lat, v.mean, v.stdv, v.pool[zi] }) |p| args.add(p);
            args.add(@as(i64, @intCast(hw)));
            args.add(@as(i64, @intCast(z_dim * hw)));
            try go(v.k.affine, v.s, .{ .x = blocks(z_dim * hw) }, .{ .x = block }, &args);
        }
        try v.post("vae.denorm", &.{.{ .role = "y", .ptr = v.pool[zi], .bytes = bytesOf(zs) }});
        const pi = try v.take();
        const ps = try v.conv("vae.post_quant_conv", v.pool[zi], zs, v.pool[pi]);
        v.give(zi);
        const ci = try v.take();
        var cs = try v.conv("vae.decoder.conv_in", v.pool[pi], ps, v.pool[ci]);
        v.give(pi);
        var cur = ci;
        { // the mid block
            const r0 = try v.resBlock("vae.decoder.mid_block.resnets.0", cur, cs);
            v.give(cur);
            const ai = try v.attention("vae.decoder.mid_block.attentions.0", r0[0], r0[1]);
            v.give(r0[0]);
            const r1 = try v.resBlock("vae.decoder.mid_block.resnets.1", ai, r0[1]);
            v.give(ai);
            cur = r1[0];
            cs = r1[1];
        }
        var i: usize = 0;
        var nb: [80]u8 = undefined;
        while (v.convs.contains(try std.fmt.bufPrint(&nb, "vae.decoder.up_blocks.{d}.resnets.0.conv1", .{i}))) : (i += 1) {
            const r = try v.upBlock(i, cur, cs);
            cur = r[0];
            cs = r[1];
        }
        try v.norm("vae.decoder.norm_out", v.pool[cur], cs, v.pool[cur]);
        try v.silu("vae.decoder.act_out", v.pool[cur], cs);
        const oi = try v.take();
        const os = try v.conv("vae.decoder.conv_out", v.pool[cur], cs, v.pool[oi]);
        // torch.clamp(out, -1, 1) is not an op of ours: to_u8's clamp to [0, 1] after x * 0.5 + 0.5 gives the same bytes
        try v.pre("vae.to_u8", &.{.{ .role = "x", .ptr = v.pool[oi], .bytes = bytesOf(os) }});
        {
            var args: cuda.launch.Args = .{};
            args.add(v.pool[oi]);
            args.add(out);
            inline for (.{ os.c, os.h, os.w }) |n| args.add(@as(i32, @intCast(n)));
            try go(v.k.to_u8, v.s, .{ .x = blocks(bytesOf(os) / 2) }, .{ .x = block }, &args);
        }
        try v.post("vae.to_u8", &.{.{ .role = "y", .ptr = out, .bytes = bytesOf(os) / 2 }});
    }
};
