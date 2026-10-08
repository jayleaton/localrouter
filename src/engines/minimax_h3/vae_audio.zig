//! MiniMax H3's audio VAE decoder (a BigVGAN) op for op as the twin (`stk_twin/h3/vae_audio.py`) runs it, under the
//! twin's names (`avae.*`), on kernels/cuda/minimax/vae_audio.cu (+ `h3_gemm_f32` of ops.cu through `launch.Ops`):
//! fp32 channel-last activations [B = 2 stereo items, T, C]; every convolution is im2col + the fp32 GEMM on weights
//! repacked by pure permutations (conv: [Cout, K * Cin] tap-major; conv_transpose: per output phase [Cout, J * Cin]),
//! the three AMP blocks of a stage averaged, clamp, then vae_decode_audio's std normalisation. Plain fp32 checkpoint
//! (weight norm already folded); the encoder tensors are not loaded.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const Pack = qi.pack.Pack;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;
const launch = @import("launch.zig");
const dit = @import("dit.zig");
const Io = dit.Io;
const Probe = dit.Probe;

pub const up_rates = [7]u32{ 5, 5, 2, 2, 2, 2, 2 };
const res_dilations = [3]u32{ 1, 3, 5 };
pub const latent_ch = 32;
pub const stereo = 2;
pub const hop = 800; // samples per latent frame
const avg3_recip: i32 = 1; // twin AVG3_RECIP = True
const block: u32 = 256;

fn blocks(n: u64) u32 {
    return @intCast((n + block - 1) / block);
}

fn i64of(v: anytype) i64 {
    return @intCast(v);
}

pub const Ops = struct {
    module: cuda.Module,
    latent_in_fn: cuda.Function,
    im2col_fn: cuda.Function,
    ct_im2col_fn: cuda.Function,
    ct_store_fn: cuda.Function,
    up2_fn: cuda.Function,
    down2_fn: cuda.Function,
    snake_fn: cuda.Function,
    add_fn: cuda.Function,
    avg3_fn: cuda.Function,
    clamp_fn: cuda.Function,
    std_fn: cuda.Function,
    div_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var o: Ops = undefined;
        o.module = m;
        inline for (.{
            .{ "latent_in_fn", "avae_latent_in" }, .{ "im2col_fn", "avae_im2col" }, .{ "ct_im2col_fn", "avae_ct_im2col" },
            .{ "ct_store_fn", "avae_ct_store" },   .{ "up2_fn", "avae_up2" },       .{ "down2_fn", "avae_down2" },
            .{ "snake_fn", "avae_snake" },         .{ "add_fn", "avae_add" },       .{ "avg3_fn", "avae_avg3" },
            .{ "clamp_fn", "avae_clamp" },         .{ "std_fn", "avae_std_scale" }, .{ "div_fn", "avae_div_scale" },
        }) |e| @field(o, e[0]) = try m.function(e[1]);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: u32, args: *cuda.launch.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = block }, .shared = 0 }, s, args);
    }
};

/// Stride-1 same-padded Conv1d: weight [Cout, K * Cin] (tap-major), bias (0 = none).
const Conv = struct { w: u64, bias: u64, cin: u32, cout: u32, k: u32, dil: u32, pad: u32 };
/// One ConvTranspose1d output phase: taps J, input offset qoff, weight [Cout, J * Cin].
const Phase = struct { j: u32, qoff: u32, w: u64 };
const Up = struct { bias: u64, cin: u32, cout: u32, k: u32, u: u32, pad: u32, ph: [5]Phase };
/// Activation1d: SnakeBeta log-parameters [C] and the two 12-tap filters.
const Act = struct { alpha: u64, beta: u64, up: u64, down: u64 };
const Layer = struct { a1: Act, c1: Conv, a2: Act, c2: Conv };

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Ops,
    g: *const launch.Ops,
    s: cuda.Stream,
    probe: ?Probe = null,
    store: Store,
    max_a: u32,
    mean: u64 = 0,
    std_: u64 = 0,
    dec_in: Conv = undefined,
    conv_pre: Conv = undefined,
    ups: [7]Up = undefined,
    res: [7][3][3]Layer = undefined,
    post_act: Act = undefined,
    conv_post: Conv = undefined,
    // scratch: x stage input, u upsampled, r the three AMP outputs, a / b work, up2 the Activation1d's 2T
    // intermediate, col im2col, ph one transpose phase, sc the std scale, wav the output
    x: u64 = 0, u: u64 = 0, r: [3]u64 = .{ 0, 0, 0 }, a: u64 = 0, b: u64 = 0, up2: u64 = 0, col: u64 = 0, ph: u64 = 0,
    sc: u64 = 0, wav: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const Ops, g: *const launch.Ops, s: cuda.Stream, p: *const Pack, up: *Uploader, max_a: u32) !Decoder {
        var t: Decoder = .{ .gpa = gpa, .d = d, .k = k, .g = g, .s = s, .store = .init(d, gpa), .max_a = max_a };
        errdefer t.store.deinit();
        var host: std.ArrayList(u8) = .empty;
        defer host.deinit(gpa);
        var nb: [96]u8 = undefined;
        t.mean = try t.store.tensor(up, p, "latents_mean");
        t.std_ = try t.store.tensor(up, p, "latents_std");
        t.dec_in = try t.conv(io, p, up, &host, "dec_in_proj", 1, true);
        t.conv_pre = try t.conv(io, p, up, &host, "decoder.conv_pre", 1, true);
        for (&t.ups, up_rates, 0..) |*u, rate, i|
            u.* = try t.convT(io, p, up, &host, try std.fmt.bufPrint(&nb, "decoder.ups.{d}.0", .{i}), rate);
        for (&t.res, 0..) |*stage, i| for (stage, 0..) |*blk, j| for (blk, 0..) |*l, n| {
            const rb = 3 * i + j;
            l.a1 = try t.act(up, p, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.activations.{d}", .{ rb, 2 * n }));
            l.c1 = try t.conv(io, p, up, &host, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.convs1.{d}", .{ rb, n }), res_dilations[n], true);
            l.a2 = try t.act(up, p, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.activations.{d}", .{ rb, 2 * n + 1 }));
            l.c2 = try t.conv(io, p, up, &host, try std.fmt.bufPrint(&nb, "decoder.resblocks.{d}.convs2.{d}", .{ rb, n }), 1, true);
        };
        t.post_act = try t.act(up, p, "decoder.activation_post");
        t.conv_post = try t.conv(io, p, up, &host, "decoder.conv_post", 1, false);

        const A: u64 = max_a;
        const act_n = stereo * hop * A * 8; // the widest activation: [2, 800 A, 8] (every stage from ups.1 on)
        inline for (.{
            .{ "x", act_n * 4 },       .{ "u", act_n * 4 },      .{ "a", act_n * 4 },  .{ "b", act_n * 4 },
            .{ "up2", act_n * 8 },     .{ "col", 140800 * A * 4 }, .{ "ph", act_n * 4 }, .{ "sc", 256 },
            .{ "wav", stereo * hop * A * 4 },
        }) |e| @field(t, e[0]) = try t.store.alloc(e[1]);
        for (&t.r) |*r| r.* = try t.store.alloc(act_n * 4);
        try t.store.done(up); // the weights have landed
        return t;
    }

    pub fn deinit(t: *Decoder) void {
        t.store.deinit();
    }

    // ---------------------------------------------------------------------------------------------- weights

    fn f32s(host: []const u8) []align(1) const f32 {
        return std.mem.bytesAsSlice(f32, host);
    }

    fn read(t: *Decoder, io: std.Io, p: *const Pack, host: *std.ArrayList(u8), name: []const u8) !qi.pack.Tensor {
        const w = try p.get(name);
        if (w.dtype != .f32) return error.BadWeight;
        try host.resize(t.gpa, w.len());
        try p.read(io, w, host.items);
        return w;
    }

    fn vec(t: *Decoder, up: *Uploader, p: *const Pack, key: []const u8, suffix: []const u8) !u64 {
        var nb: [128]u8 = undefined;
        return t.store.tensor(up, p, try std.fmt.bufPrint(&nb, "{s}.{s}", .{ key, suffix }));
    }

    fn conv(t: *Decoder, io: std.Io, p: *const Pack, up: *Uploader, host: *std.ArrayList(u8), key: []const u8, dil: u32, bias: bool) !Conv {
        var nb: [128]u8 = undefined;
        const w = try t.read(io, p, host, try std.fmt.bufPrint(&nb, "{s}.weight", .{key}));
        const co = w.dim(0);
        const ci = w.dim(1);
        const kk = w.dim(2);
        const dst = try t.gpa.alloc(f32, w.len() / 4);
        defer t.gpa.free(dst);
        const src = f32s(host.items);
        for (0..co) |o| for (0..kk) |k| for (0..ci) |c| {
            dst[(o * kk + k) * ci + c] = src[(o * ci + c) * kk + k];
        };
        const wp = try t.store.upload(up, std.mem.sliceAsBytes(dst));
        return .{
            .w = wp,
            .bias = if (bias) try t.vec(up, p, key, "bias") else 0,
            .cin = @intCast(ci),
            .cout = @intCast(co),
            .k = @intCast(kk),
            .dil = dil,
            .pad = @intCast((kk * dil - dil) / 2),
        };
    }

    fn convT(t: *Decoder, io: std.Io, p: *const Pack, up: *Uploader, host: *std.ArrayList(u8), key: []const u8, u: u32) !Up {
        var nb: [128]u8 = undefined;
        const w = try t.read(io, p, host, try std.fmt.bufPrint(&nb, "{s}.weight", .{key}));
        const ci = w.dim(0);
        const co = w.dim(1);
        const kk = w.dim(2);
        const pad: u32 = @intCast((kk - u) / 2);
        if (pad >= u or u > 5) return error.BadWeight; // the phase decomposition needs 0 <= padding < stride
        var us: Up = .{ .bias = try t.vec(up, p, key, "bias"), .cin = @intCast(ci), .cout = @intCast(co), .k = @intCast(kk), .u = u, .pad = pad, .ph = undefined };
        const src = f32s(host.items);
        for (0..u) |r| {
            const J = (kk - r + u - 1) / u;
            const dst = try t.gpa.alloc(f32, co * J * ci);
            defer t.gpa.free(dst);
            for (0..co) |o| for (0..J) |j| for (0..ci) |c| {
                dst[(o * J + j) * ci + c] = src[(c * co + o) * kk + r + j * u];
            };
            us.ph[r] = .{ .j = @intCast(J), .qoff = if (r < pad) 1 else 0, .w = try t.store.upload(up, std.mem.sliceAsBytes(dst)) };
        }
        return us;
    }

    fn act(t: *Decoder, up: *Uploader, p: *const Pack, key: []const u8) !Act {
        return .{
            .alpha = try t.vec(up, p, key, "act.alpha"),
            .beta = try t.vec(up, p, key, "act.beta"),
            .up = try t.vec(up, p, key, "upsample.filter"),
            .down = try t.vec(up, p, key, "downsample.lowpass.filter"),
        };
    }

    // ---------------------------------------------------------------------------------------------- ops

    fn pre(t: *Decoder, name: []const u8, ins: []const Io) !void {
        if (t.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(t: *Decoder, name: []const u8, outs: []const Io) !void {
        if (t.probe) |p| try p.after(p.ctx, name, outs);
    }

    fn conv1d(t: *Decoder, name: []const u8, c: Conv, in: u64, out: u64, B: u64, T: u64) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = B * T * c.cin * 4 }});
        var col = in; // K = 1: the activations are the GEMM's rows already
        if (c.k != 1) {
            var a: cuda.launch.Args = .{};
            a.add(in);
            a.add(t.col);
            inline for (.{ B, T, c.cin }) |v| a.add(i64of(v));
            inline for (.{ c.k, c.dil, c.pad }) |v| a.add(@as(i32, @intCast(v)));
            try Ops.go(t.k.im2col_fn, t.s, blocks(B * T * c.k * c.cin), &a);
            col = t.col;
        }
        try t.g.gemmF32(t.s, col, c.w, c.bias, out, B * T, c.cout, @as(u64, c.k) * c.cin);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = B * T * c.cout * 4 }});
    }

    fn convTranspose(t: *Decoder, name: []const u8, c: *const Up, in: u64, out: u64, B: u64, L: u64) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = B * L * c.cin * 4 }});
        for (c.ph[0..c.u], 0..) |ph, r| {
            var a: cuda.launch.Args = .{};
            a.add(in);
            a.add(t.col);
            inline for (.{ B, L, c.cin }) |v| a.add(i64of(v));
            a.add(@as(i32, @intCast(ph.j)));
            a.add(@as(i32, @intCast(ph.qoff)));
            try Ops.go(t.k.ct_im2col_fn, t.s, blocks(B * L * ph.j * c.cin), &a);
            try t.g.gemmF32(t.s, t.col, ph.w, c.bias, t.ph, B * L, c.cout, @as(u64, ph.j) * c.cin);
            var s: cuda.launch.Args = .{};
            s.add(t.ph);
            s.add(out);
            inline for (.{ B, L, c.cout }) |v| s.add(i64of(v));
            inline for (.{ c.u, c.pad, r, ph.qoff }) |v| s.add(@as(i32, @intCast(v)));
            try Ops.go(t.k.ct_store_fn, t.s, blocks(B * L * c.cout), &s);
        }
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = B * L * c.u * c.cout * 4 }});
    }

    /// Activation1d: up x2, SnakeBeta (in place on the 2T intermediate), down x2. [B, T, C] -> [B, T, C].
    fn activation(t: *Decoder, name: []const u8, c: Act, in: u64, out: u64, B: u64, T: u64, C: u64) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = B * T * C * 4 }});
        const n2 = B * 2 * T * C;
        var u: cuda.launch.Args = .{};
        inline for (.{ in, c.up, t.up2 }) |v| u.add(v);
        inline for (.{ T, C, n2 }) |v| u.add(i64of(v));
        try Ops.go(t.k.up2_fn, t.s, blocks(n2), &u);
        var sn: cuda.launch.Args = .{};
        inline for (.{ t.up2, c.alpha, c.beta, t.up2 }) |v| sn.add(v);
        inline for (.{ C, n2 }) |v| sn.add(i64of(v));
        try Ops.go(t.k.snake_fn, t.s, blocks(n2), &sn);
        var dn: cuda.launch.Args = .{};
        inline for (.{ t.up2, c.down, out }) |v| dn.add(v);
        inline for (.{ 2 * T, C, B * T * C }) |v| dn.add(i64of(v));
        try Ops.go(t.k.down2_fn, t.s, blocks(B * T * C), &dn);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = B * T * C * 4 }});
    }

    /// The latent (device fp32 [1, 32, 2, A], after process_latent_out) -> the waveform fp32 [1, 2, A * 800] at the
    /// returned device pointer (valid until the next decode).
    pub fn decode(t: *Decoder, latent: u64, a_len: u32) !u64 {
        if (a_len == 0 or a_len > t.max_a) return error.LatentTooLong;
        const A: u64 = a_len;
        const B: u64 = stereo;
        var nb: [64]u8 = undefined;
        var pb: [48]u8 = undefined;
        const zn: u64 = latent_ch * stereo * A;
        try t.pre("avae.latent_in", &.{.{ .role = "z", .ptr = latent, .bytes = zn * 4 }});
        {
            var a: cuda.launch.Args = .{};
            inline for (.{ latent, t.mean, t.std_, t.b }) |v| a.add(v);
            inline for (.{ 1, latent_ch, stereo, A }) |v| a.add(i64of(v));
            try Ops.go(t.k.latent_in_fn, t.s, blocks(zn), &a);
        }
        try t.post("avae.latent_in", &.{.{ .role = "y", .ptr = t.b, .bytes = zn * 4 }});
        try t.conv1d("avae.dec_in_proj", t.dec_in, t.b, t.a, B, A);
        try t.conv1d("avae.conv_pre", t.conv_pre, t.a, t.x, B, A);
        var T = A; // frames at the stage input
        for (&t.ups, 0..) |*up, i| {
            try t.convTranspose(try std.fmt.bufPrint(&nb, "avae.ups.{d}", .{i}), up, t.x, t.u, B, T);
            T *= up.u;
            const C: u64 = up.cout;
            const n = B * T * C;
            for (0..3) |j| {
                var cur = t.u;
                for (&t.res[i][j], 0..) |*l, nl| {
                    const pfx = try std.fmt.bufPrint(&pb, "avae.res.{d}.{d}.{d}", .{ i, j, nl });
                    try t.activation(try std.fmt.bufPrint(&nb, "{s}.a1", .{pfx}), l.a1, cur, t.a, B, T, C);
                    try t.conv1d(try std.fmt.bufPrint(&nb, "{s}.c1", .{pfx}), l.c1, t.a, t.b, B, T);
                    try t.activation(try std.fmt.bufPrint(&nb, "{s}.a2", .{pfx}), l.a2, t.b, t.a, B, T, C);
                    try t.conv1d(try std.fmt.bufPrint(&nb, "{s}.c2", .{pfx}), l.c2, t.a, t.b, B, T);
                    try t.add(try std.fmt.bufPrint(&nb, "{s}.add", .{pfx}), t.b, cur, t.r[j], n);
                    cur = t.r[j];
                }
            }
            const an = try std.fmt.bufPrint(&nb, "avae.avg.{d}", .{i});
            try t.pre(an, &.{ .{ .role = "a", .ptr = t.r[0], .bytes = n * 4 }, .{ .role = "b", .ptr = t.r[1], .bytes = n * 4 }, .{ .role = "c", .ptr = t.r[2], .bytes = n * 4 } });
            {
                var a: cuda.launch.Args = .{};
                inline for (.{ t.r[0], t.r[1], t.r[2], t.x }) |v| a.add(v);
                a.add(i64of(n));
                a.add(avg3_recip);
                try Ops.go(t.k.avg3_fn, t.s, blocks(n), &a);
            }
            try t.post(an, &.{.{ .role = "y", .ptr = t.x, .bytes = n * 4 }});
        }
        try t.activation("avae.post", t.post_act, t.x, t.a, B, T, 8);
        try t.conv1d("avae.conv_post", t.conv_post, t.a, t.b, B, T);
        const wn = B * T; // samples over both channels (conv_post has one output channel)
        try t.pre("avae.clamp", &.{.{ .role = "x", .ptr = t.b, .bytes = wn * 4 }});
        {
            var a: cuda.launch.Args = .{};
            inline for (.{ t.b, t.a }) |v| a.add(v);
            a.add(i64of(wn));
            try Ops.go(t.k.clamp_fn, t.s, blocks(wn), &a);
        }
        try t.post("avae.clamp", &.{.{ .role = "y", .ptr = t.a, .bytes = wn * 4 }});
        // vae_decode_audio's normalisation: one std over both channels (Bb = 1 row of 2 * A * 800 samples)
        try t.pre("avae.std_scale", &.{.{ .role = "x", .ptr = t.a, .bytes = wn * 4 }});
        {
            var a: cuda.launch.Args = .{};
            inline for (.{ t.a, t.sc }) |v| a.add(v);
            a.add(i64of(wn));
            try Ops.go(t.k.std_fn, t.s, 1, &a);
        }
        try t.post("avae.std_scale", &.{.{ .role = "sc", .ptr = t.sc, .bytes = 4 }});
        try t.pre("avae.div_scale", &.{ .{ .role = "x", .ptr = t.a, .bytes = wn * 4 }, .{ .role = "sc", .ptr = t.sc, .bytes = 4 } });
        {
            var a: cuda.launch.Args = .{};
            inline for (.{ t.a, t.sc, t.wav }) |v| a.add(v);
            inline for (.{ wn, wn }) |v| a.add(i64of(v));
            try Ops.go(t.k.div_fn, t.s, blocks(wn), &a);
        }
        try t.post("avae.div_scale", &.{.{ .role = "y", .ptr = t.wav, .bytes = wn * 4 }});
        return t.wav;
    }

    fn add(t: *Decoder, name: []const u8, a_: u64, b_: u64, out: u64, n: u64) !void {
        try t.pre(name, &.{ .{ .role = "a", .ptr = a_, .bytes = n * 4 }, .{ .role = "b", .ptr = b_, .bytes = n * 4 } });
        var a: cuda.launch.Args = .{};
        inline for (.{ a_, b_, out }) |v| a.add(v);
        a.add(i64of(n));
        try Ops.go(t.k.add_fn, t.s, blocks(n), &a);
        try t.post(name, &.{.{ .role = "y", .ptr = out, .bytes = n * 4 }});
    }
};

test "up_rates multiply to the 800-sample hop" {
    var h: u32 = 1;
    for (up_rates) |u| h *= u;
    try std.testing.expectEqual(@as(u32, hop), h);
}
