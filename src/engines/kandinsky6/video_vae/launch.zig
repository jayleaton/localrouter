//! Kandinsky-6 video VAE launch ABI: the kernels of kernels/cuda/kandinsky6/video_vae/{conv,ops}.cu, their by-value
//! parameter blocks, the weight layout the kernels read, and the block sequences (GroupNorm, resnet, attention) shared by
//! the encoder and the decoder.
//!
//! Ownership. The caller owns the CUDA context, the stream, the loaded modules and every buffer. Components receive device
//! addresses (`u64`) and issue launches through a `Launcher`, so this code imports no CUDA runtime and never allocates,
//! loads or frees anything. With TensorFold's `cuda` module (docs/dev/TENSORFOLD-RUNTIME.md) an adapter is:
//!
//!     fn run(ctx: *anyopaque, l: Launch) !void {
//!         const self: *Gpu = @ptrCast(@alignCast(ctx));
//!         var a: cuda.launch.Args = .{};
//!         switch (l.call) { inline else => |*p| a.add(p.*) }   // one by-value struct parameter
//!         try cuda.launch.launch(self.fns[@intFromEnum(l.call)], .{ .grid = .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] },
//!             .block = .{ .x = l.block[0], .y = l.block[1], .z = l.block[2] } }, self.stream, &a);
//!     }
//!
//! with `fns[k]` = `module.function(symbol(k))` from the conv or ops image (`image(k)`). All shared memory is static.
//!
//! Precision (all from the pinned reference, AutoencoderKLHunyuanVideo loaded with torch_dtype=float16): parameters are
//! the checkpoint's fp32 tensors rounded to fp16 (round to nearest even), activations fp16, GEMM accumulation fp32, norms
//! and softmax in fp32, each torch op one rounding; the exact expressions are documented beside each kernel. The fp32
//! boundaries are the I2V input normalisation (fp32 on the CPU, then fp16) and the decode input (bf16 DiT latent divided
//! by the scaling factor in bf16, then fp16).

const std = @import("std");

pub const Kernel = enum(u8) {
    conv_wide,
    conv_narrow,
    conv_wide_s,
    conv_narrow_s,
    gn_partial,
    gn_finalize,
    gn_apply,
    softmax_rows,
    transpose,
    latent_in,
    rgb8_in,
    rgb_out,
    posterior,
    blend,
    copy,
};

pub const Image = enum { conv, ops };

pub fn image(k: Kernel) Image {
    return switch (k) {
        .conv_wide, .conv_narrow, .conv_wide_s, .conv_narrow_s => .conv,
        else => .ops,
    };
}

/// The kernel's symbol in its fatbin.
pub fn symbol(k: Kernel) [:0]const u8 {
    return switch (k) {
        inline else => |t| "k6v_" ++ @tagName(t),
    };
}

// ---- Parameter blocks: field for field the structs of conv.cu / ops.cu --------------------------------------------------

pub const ConvArgs = extern struct {
    x: u64,
    w: u64,
    bias: u64,
    res: u64,
    y: u64,
    sxt: i64,
    sxh: i64,
    sxw: i64,
    ldw: i64,
    ldy: i64,
    ti: i32,
    hi: i32,
    wi: i32,
    ci: i32,
    tu: i32,
    hu: i32,
    wu: i32,
    up_t: i32,
    up_hw: i32,
    kt: i32,
    kh: i32,
    kw: i32,
    pt: i32,
    ph: i32,
    pw: i32,
    st: i32,
    sh: i32,
    sw: i32,
    to: i32,
    ho: i32,
    wo: i32,
    co: i32,
    k: i32,
    bias_mode: i32,
    out_f32: i32,
};

pub const GnArgs = extern struct { x: u64, partial: u64, stats: u64, positions: i64, c: i32, groups: i32, pass: i32, eps: f32 };
pub const GnApplyArgs = extern struct { x: u64, y: u64, gamma: u64, beta: u64, stats: u64, positions: i64, c: i32, groups: i32, silu: i32 };
pub const SoftmaxArgs = extern struct { s: u64, p: u64, lds: i64, ldp: i64, row0: i32, spf: i32, cols: i32, scale: f32 };
pub const TransposeArgs = extern struct { x: u64, y: u64, n: i64, ldn: i64, c: i32 };
pub const LatentInArgs = extern struct { z: u64, y: u64, count: i64, inv_sf: f32 };
pub const Rgb8InArgs = extern struct { x: u64, y: u64, count: i64 };
pub const RgbOutArgs = extern struct { x: u64, y: u64, count: i64 };
pub const PosteriorArgs = extern struct { moments: u64, eps: u64, mean: u64, stdv: u64, latent: u64, positions: i64, lc: i32, sf: f32 };
pub const BlendArgs = extern struct { a: u64, b: u64, fa: i32, ha: i32, wa: i32, fb: i32, hb: i32, wb: i32, ch: i32, axis: i32, extent: i32 };
pub const CopyArgs = extern struct {
    src: u64,
    dst: u64,
    sf: i32,
    sh: i32,
    sw: i32,
    df: i32,
    dh: i32,
    dw: i32,
    f: i32,
    h: i32,
    w: i32,
    ch: i32,
    sf0: i32,
    sh0: i32,
    sw0: i32,
    df0: i32,
    dh0: i32,
    dw0: i32,
};

comptime {
    // The C layouts (x86-64 / aarch64 LP64): any drift from conv.cu / ops.cu breaks every launch.
    std.debug.assert(@sizeOf(ConvArgs) == 184 and @offsetOf(ConvArgs, "ti") == 80 and @offsetOf(ConvArgs, "out_f32") == 176);
    std.debug.assert(@sizeOf(GnArgs) == 48 and @sizeOf(GnApplyArgs) == 64 and @sizeOf(SoftmaxArgs) == 48);
    std.debug.assert(@sizeOf(TransposeArgs) == 40 and @sizeOf(LatentInArgs) == 32 and @sizeOf(Rgb8InArgs) == 24);
    std.debug.assert(@sizeOf(PosteriorArgs) == 56 and @sizeOf(BlendArgs) == 56 and @sizeOf(CopyArgs) == 80);
}

pub const Call = union(Kernel) {
    conv_wide: ConvArgs,
    conv_narrow: ConvArgs,
    conv_wide_s: ConvArgs,
    conv_narrow_s: ConvArgs,
    gn_partial: GnArgs,
    gn_finalize: GnArgs,
    gn_apply: GnApplyArgs,
    softmax_rows: SoftmaxArgs,
    transpose: TransposeArgs,
    latent_in: LatentInArgs,
    rgb8_in: Rgb8InArgs,
    rgb_out: RgbOutArgs,
    posterior: PosteriorArgs,
    blend: BlendArgs,
    copy: CopyArgs,
};

pub const Launch = struct {
    call: Call,
    grid: [3]u32,
    block: [3]u32,
};

/// The caller's stream: one launch at a time, in order.
pub const Launcher = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, l: Launch) anyerror!void,

    pub fn run(self: Launcher, l: Launch) !void {
        return self.func(self.ctx, l);
    }
};

pub const Error = error{ InvalidShape, Misaligned, BufferTooSmall };

// ---- Weights --------------------------------------------------------------------------------------------------------

/// A convolution or linear: rows [co][ldw] of fp16, k = kt * kh * kw * ci in tap-major, channel-minor order, zero padded
/// to ldw = k rounded up to 8; bias fp16 [co].
pub const ConvW = struct {
    w: u64,
    b: u64,
    co: u32,
    ci: u32,
    kt: u32 = 3,
    kh: u32 = 3,
    kw: u32 = 3,

    pub fn k(c: ConvW) u32 {
        return c.kt * c.kh * c.kw * c.ci;
    }
    pub fn ldw(c: ConvW) u32 {
        return std.mem.alignForward(u32, c.k(), 8);
    }
};

pub const NormW = struct { gamma: u64, beta: u64, c: u32 };
pub const ResnetW = struct { norm1: NormW, conv1: ConvW, norm2: NormW, conv2: ConvW, shortcut: ?ConvW };
pub const AttnW = struct { norm: NormW, q: ConvW, k: ConvW, v: ConvW, out: ConvW };
pub const MidW = struct { res0: ResnetW, attn: AttnW, res1: ResnetW };

/// Bytes of one repacked conv/linear weight ([co][ldw] fp16).
pub fn convWeightBytes(co: u32, ci: u32, kt: u32, kh: u32, kw: u32) u64 {
    return @as(u64, co) * std.mem.alignForward(u32, kt * kh * kw * ci, 8) * 2;
}

/// The loader's repack of a checkpoint conv weight (fp32 [co][ci][kt][kh][kw], torch order) into [co][ldw] fp16 rows,
/// k = ((dt * kh + dy) * kw + dx) * ci + c, zero padding; fp32 -> fp16 rounds to nearest even as torch's .to(float16).
/// A linear weight [co][ci] is the same call with kt = kh = kw = 1.
pub fn repackConv(src: []const f32, dst: []f16, co: u32, ci: u32, kt: u32, kh: u32, kw: u32) Error!void {
    const taps = kt * kh * kw;
    const ldw = std.mem.alignForward(u32, taps * ci, 8);
    if (src.len != @as(usize, co) * ci * taps or dst.len != @as(usize, co) * ldw) return error.InvalidShape;
    for (0..co) |o| {
        const row = dst[o * ldw ..][0..ldw];
        @memset(row, 0);
        for (0..ci) |c| for (0..taps) |tap| {
            row[tap * ci + c] = @floatCast(src[(o * ci + c) * taps + tap]);
        };
    }
}

/// fp32 -> fp16 (round to nearest even) for biases and norm parameters.
pub fn toHalf(src: []const f32, dst: []f16) Error!void {
    if (src.len != dst.len) return error.InvalidShape;
    for (src, dst) |s, *d| d.* = @floatCast(s);
}

// ---- Views and launch builders ----------------------------------------------------------------------------------------

/// A channels-last activation [t][h][w][c] at `ptr` with element strides (channel stride 1).
pub const View = struct {
    ptr: u64,
    t: u32,
    h: u32,
    w: u32,
    c: u32,
    st: u64,
    sh: u64,
    sw: u64,

    pub fn dense(ptr: u64, t: u32, h: u32, w: u32, c: u32) View {
        return .{ .ptr = ptr, .t = t, .h = h, .w = w, .c = c, .sw = c, .sh = @as(u64, w) * c, .st = @as(u64, h) * w * c };
    }
    pub fn positions(v: View) u64 {
        return @as(u64, v.t) * v.h * v.w;
    }
    pub fn elems(v: View) u64 {
        return v.positions() * v.c;
    }
    pub fn contiguous(v: View) bool {
        return v.sw == v.c and v.sh == @as(u64, v.w) * v.c and v.st == @as(u64, v.h) * v.w * v.c;
    }
    /// A sub-box (frames [t0, t0 + nt), rows [y0, ..), columns [x0, ..)), same strides.
    pub fn box(v: View, t0: u32, nt: u32, y0: u32, nh: u32, x0: u32, nw: u32) View {
        std.debug.assert(t0 + nt <= v.t and y0 + nh <= v.h and x0 + nw <= v.w);
        return .{ .ptr = v.ptr + 2 * (t0 * v.st + y0 * v.sh + x0 * v.sw), .t = nt, .h = nh, .w = nw, .c = v.c, .st = v.st, .sh = v.sh, .sw = v.sw };
    }
};

pub const BiasMode = enum(i32) {
    /// half(acc + bias): F.linear
    linear = 0,
    /// half(half(acc) + bias): mirrors the reference's convolution then output.add_(bias); unqualified until SM121 fixtures
    conv = 1,
};

pub const ConvSpec = struct {
    up_t: u32 = 1,
    up_hw: u32 = 1,
    st: u32 = 1,
    sh: u32 = 1,
    sw: u32 = 1,
    /// HunyuanVideoCausalConv3d pads (k - 1 front in time, k / 2 each side spatially, replicate); nn.Conv3d (the quant
    /// convolutions) pads nothing.
    causal_pad: bool = true,
    bias: BiasMode = .conv,
    res: ?u64 = null,
};

fn i32of(v: anytype) i32 {
    return @intCast(v);
}

fn grid1(n: u64) [3]u32 {
    return .{ @intCast((n + 255) / 256), 1, 1 };
}

/// Output dims of a convolution of `x` under `s` (upsampled grid, replicate padded, strided).
pub fn convOut(x: View, w: ConvW, s: ConvSpec) [3]u32 {
    const tu = if (s.up_t == 2) 1 + 2 * (x.t - 1) else x.t;
    const hu = x.h * s.up_hw;
    const wu = x.w * s.up_hw;
    const pt = if (s.causal_pad) w.kt - 1 else 0;
    const ph = if (s.causal_pad) w.kh / 2 else 0;
    const pw = if (s.causal_pad) w.kw / 2 else 0;
    return .{ (tu + pt - w.kt) / s.st + 1, (hu + 2 * ph - w.kh) / s.sh + 1, (wu + 2 * pw - w.kw) / s.sw + 1 };
}

/// One convolution: x (channels last, any strides) -> y dense [to][ho][wo][co].
pub fn conv(x: View, w: ConvW, s: ConvSpec, y: u64) Error!Launch {
    if (x.c != w.ci) return error.InvalidShape;
    if (!(s.up_t == 1 or s.up_t == 2) or !(s.up_hw == 1 or s.up_hw == 2)) return error.InvalidShape;
    const o = convOut(x, w, s);
    const vec = w.ci % 8 == 0 and x.st % 8 == 0 and x.sh % 8 == 0 and x.sw % 8 == 0 and x.ptr % 16 == 0;
    if (w.w % 16 != 0) return error.Misaligned;
    const narrow = w.co <= 32;
    const a: ConvArgs = .{
        .x = x.ptr,
        .w = w.w,
        .bias = w.b,
        .res = s.res orelse 0,
        .y = y,
        .sxt = @intCast(x.st),
        .sxh = @intCast(x.sh),
        .sxw = @intCast(x.sw),
        .ldw = w.ldw(),
        .ldy = w.co,
        .ti = i32of(x.t),
        .hi = i32of(x.h),
        .wi = i32of(x.w),
        .ci = i32of(x.c),
        .tu = i32of(if (s.up_t == 2) 1 + 2 * (x.t - 1) else x.t),
        .hu = i32of(x.h * s.up_hw),
        .wu = i32of(x.w * s.up_hw),
        .up_t = i32of(s.up_t),
        .up_hw = i32of(s.up_hw),
        .kt = i32of(w.kt),
        .kh = i32of(w.kh),
        .kw = i32of(w.kw),
        .pt = if (s.causal_pad) i32of(w.kt - 1) else 0,
        .ph = if (s.causal_pad) i32of(w.kh / 2) else 0,
        .pw = if (s.causal_pad) i32of(w.kw / 2) else 0,
        .st = i32of(s.st),
        .sh = i32of(s.sh),
        .sw = i32of(s.sw),
        .to = i32of(o[0]),
        .ho = i32of(o[1]),
        .wo = i32of(o[2]),
        .co = i32of(w.co),
        .k = i32of(w.k()),
        .bias_mode = @backingInt(s.bias),
        .out_f32 = 0,
    };
    const m: u64 = @as(u64, o[0]) * o[1] * o[2];
    if ((m + 127) / 128 > std.math.maxInt(i32)) return error.InvalidShape;
    const bn: u32 = if (narrow) 32 else 128;
    const grid: [3]u32 = .{ @intCast((m + 127) / 128), (w.co + bn - 1) / bn, 1 };
    const block: [3]u32 = .{ if (narrow) 128 else 256, 1, 1 };
    const call: Call = if (narrow)
        (if (vec) .{ .conv_narrow = a } else .{ .conv_narrow_s = a })
    else
        (if (vec) .{ .conv_wide = a } else .{ .conv_wide_s = a });
    return .{ .call = call, .grid = grid, .block = block };
}

/// A dense GEMM through the convolution kernel: C[m][n] = A[m][0..k] . B[n][0..k]^T. A has row stride lda and B ldb, both
/// multiples of 8, and both zero past k up to the next multiple of 8. `out_f32` writes raw fp32 sums (no bias / residual).
pub const Dense = struct {
    a: u64,
    lda: u64,
    b: u64,
    ldb: u64,
    bias: u64 = 0,
    res: u64 = 0,
    c: u64,
    ldc: u64,
    m: u32,
    n: u32,
    k: u32,
    out_f32: bool = false,
};

pub fn dense(d: Dense) Error!Launch {
    if (d.lda % 8 != 0 or d.ldb % 8 != 0 or d.a % 16 != 0 or d.b % 16 != 0) return error.Misaligned;
    if (d.lda < std.mem.alignForward(u64, d.k, 8) or d.ldb < std.mem.alignForward(u64, d.k, 8)) return error.InvalidShape;
    const a: ConvArgs = .{
        .x = d.a,
        .w = d.b,
        .bias = d.bias,
        .res = d.res,
        .y = d.c,
        .sxt = 0,
        .sxh = 0,
        .sxw = @intCast(d.lda),
        .ldw = @intCast(d.ldb),
        .ldy = @intCast(d.ldc),
        .ti = 1,
        .hi = 1,
        .wi = i32of(d.m),
        // ci = k rounded to 8: the gather reads whole 16-byte chunks of the zero-padded rows (bits as at ci = k).
        .ci = i32of(std.mem.alignForward(u32, d.k, 8)),
        .tu = 1,
        .hu = 1,
        .wu = i32of(d.m),
        .up_t = 1,
        .up_hw = 1,
        .kt = 1,
        .kh = 1,
        .kw = 1,
        .pt = 0,
        .ph = 0,
        .pw = 0,
        .st = 1,
        .sh = 1,
        .sw = 1,
        .to = 1,
        .ho = 1,
        .wo = i32of(d.m),
        .co = i32of(d.n),
        .k = i32of(d.k),
        .bias_mode = @backingInt(BiasMode.linear),
        .out_f32 = @intFromBool(d.out_f32),
    };
    const narrow = d.n <= 32;
    const bn: u32 = if (narrow) 32 else 128;
    return .{
        .call = if (narrow) .{ .conv_narrow = a } else .{ .conv_wide = a },
        .grid = .{ (d.m + 127) / 128, (d.n + bn - 1) / bn, 1 },
        .block = .{ if (narrow) 128 else 256, 1, 1 },
    };
}

// ---- Scratch arena ----------------------------------------------------------------------------------------------------

/// A bump allocator over the caller's scratch. `measure` runs the same graph to size it: requests only record the peak.
pub const Arena = struct {
    base: u64,
    cap: u64,
    top: u64 = 0,
    peak: u64 = 0,
    measure: bool = false,

    pub fn take(self: *Arena, bytes: u64) Error!u64 {
        const at = std.mem.alignForward(u64, self.top, 256);
        const end = at + bytes;
        if (!self.measure and end > self.cap) return error.BufferTooSmall;
        self.top = end;
        self.peak = @max(self.peak, end);
        return self.base + at;
    }
    pub fn mark(self: *Arena) u64 {
        return self.top;
    }
    pub fn reset(self: *Arena, m: u64) void {
        self.top = m;
    }
};

pub const gn_chunk: u64 = 2048; // K6V_GN_CHUNK
pub const gn_groups: u32 = 32;
pub const gn_eps: f32 = 1e-6;
/// Rows of attention scores per block (S: br x ldn fp32 and P: br x ldn fp16). Fixed: the P . V k-extent of a block
/// depends on it, so it is part of the fixed arithmetic.
pub const attn_rows: u32 = 1024;

/// Working buffers of the blocks: four activations of `act` elements each, GroupNorm and attention scratch.
pub const Work = struct {
    a: u64,
    b: u64,
    gn_partial: u64,
    gn_stats: u64,
    act_elems: u64,
    attn: Attn,

    pub const Attn = struct { q: u64, k: u64, v: u64, vt: u64, o: u64, s: u64, p: u64, tokens: u32 };

    /// Carves the blocks' scratch for activations up to `act_elems` (fp16) and attention over up to `tokens` tokens.
    pub fn init(arena: *Arena, act_elems: u64, max_positions: u64, tokens: u32, c_attn: u32) Error!Work {
        const ldn = std.mem.alignForward(u64, tokens, 8);
        const br = @min(attn_rows, tokens);
        const chunks = (max_positions + gn_chunk - 1) / gn_chunk;
        return .{
            .a = try arena.take(act_elems * 2),
            .b = try arena.take(act_elems * 2),
            .gn_partial = try arena.take(chunks * gn_groups * 4),
            .gn_stats = try arena.take(gn_groups * 2 * 4),
            .act_elems = act_elems,
            .attn = .{
                .q = try arena.take(@as(u64, tokens) * c_attn * 2),
                .k = try arena.take(@as(u64, tokens) * c_attn * 2),
                .v = try arena.take(@as(u64, tokens) * c_attn * 2),
                .vt = try arena.take(@as(u64, c_attn) * ldn * 2),
                .o = try arena.take(@as(u64, tokens) * c_attn * 2),
                .s = try arena.take(@as(u64, br) * ldn * 4),
                .p = try arena.take(@as(u64, br) * ldn * 2),
                .tokens = tokens,
            },
        };
    }
};

fn fits(work: Work, elems: u64) Error!void {
    if (elems > work.act_elems) return error.BufferTooSmall;
}

/// nn.GroupNorm(32, C, eps=1e-6) (+ SiLU) of dense x into dense y (y may be x): five launches.
pub fn groupNorm(l: Launcher, work: Work, x: View, n: NormW, silu: bool, y: u64) !void {
    if (!x.contiguous() or x.c != n.c or x.c % 8 != 0 or 256 % (x.c / 8) != 0 or x.c % gn_groups != 0) return error.InvalidShape;
    if (x.ptr % 16 != 0 or y % 16 != 0) return error.Misaligned;
    const p = x.positions();
    const chunks = (p + gn_chunk - 1) / gn_chunk;
    var a: GnArgs = .{ .x = x.ptr, .partial = work.gn_partial, .stats = work.gn_stats, .positions = @intCast(p), .c = i32of(x.c), .groups = gn_groups, .pass = 0, .eps = gn_eps };
    for (0..2) |pass| {
        a.pass = @intCast(pass);
        try l.run(.{ .call = .{ .gn_partial = a }, .grid = .{ @intCast(chunks), 1, 1 }, .block = .{ 256, 1, 1 } });
        try l.run(.{ .call = .{ .gn_finalize = a }, .grid = .{ gn_groups, 1, 1 }, .block = .{ 256, 1, 1 } });
    }
    try l.run(.{ .call = .{ .gn_apply = .{ .x = x.ptr, .y = y, .gamma = n.gamma, .beta = n.beta, .stats = work.gn_stats, .positions = @intCast(p), .c = i32of(x.c), .groups = gn_groups, .silu = @intFromBool(silu) } }, .grid = grid1(p * x.c / 8), .block = .{ 256, 1, 1 } });
}

/// HunyuanVideoResnetBlockCausal3D: y = conv2(silu(norm2(conv1(silu(norm1(x)))))) + (shortcut(x) or x).
/// x and y dense, distinct; uses work.a and work.b. Returns y's view.
pub fn resnet(l: Launcher, work: Work, x: View, r: ResnetW, y: u64) !View {
    const co = r.conv1.co;
    try fits(work, x.positions() * @max(x.c, co));
    const a = View.dense(work.a, x.t, x.h, x.w, x.c);
    try groupNorm(l, work, x, r.norm1, true, a.ptr);
    try l.run(try conv(a, r.conv1, .{}, work.b));
    const b = View.dense(work.b, x.t, x.h, x.w, co);
    try groupNorm(l, work, b, r.norm2, true, work.a);
    var res = x.ptr;
    if (r.shortcut) |sc| {
        try l.run(try conv(x, sc, .{}, work.b));
        res = work.b;
    } else if (x.c != co) return error.InvalidShape;
    try l.run(try conv(View.dense(work.a, x.t, x.h, x.w, co), r.conv2, .{ .res = res }, y));
    return View.dense(y, x.t, x.h, x.w, co);
}

pub fn attnScale(c: u32) f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c))));
}

/// The mid block's Attention (one head of dim C, group_norm, frame-causal mask, residual): y = to_out(sdpa(q, k, v)) + x.
pub fn attention(l: Launcher, work: Work, x: View, w: AttnW, y: u64) !View {
    const n64 = x.positions();
    const c = x.c;
    if (n64 > work.attn.tokens or c != w.q.ci) return error.BufferTooSmall;
    const n: u32 = @intCast(n64);
    const spf = x.h * x.w;
    const ldn = std.mem.alignForward(u32, n, 8);
    try fits(work, n64 * c);
    try groupNorm(l, work, x, w.norm, false, work.a);
    const at = work.attn;
    for ([_]struct { ConvW, u64 }{ .{ w.q, at.q }, .{ w.k, at.k }, .{ w.v, at.v } }) |pw| {
        try l.run(try dense(.{ .a = work.a, .lda = c, .b = pw[0].w, .ldb = pw[0].ldw(), .bias = pw[0].b, .c = pw[1], .ldc = c, .m = n, .n = c, .k = c }));
    }
    try l.run(.{ .call = .{ .transpose = .{ .x = at.v, .y = at.vt, .n = n, .ldn = ldn, .c = i32of(c) } }, .grid = .{ (ldn + 31) / 32, (c + 31) / 32, 1 }, .block = .{ 32, 8, 1 } });
    var r0: u32 = 0;
    while (r0 < n) : (r0 += attn_rows) {
        const rows = @min(attn_rows, n - r0);
        const cols = ((r0 + rows - 1) / spf + 1) * spf; // keys the block's last row sees
        try l.run(try dense(.{ .a = at.q + @as(u64, r0) * c * 2, .lda = c, .b = at.k, .ldb = c, .c = at.s, .ldc = ldn, .m = rows, .n = cols, .k = c, .out_f32 = true }));
        try l.run(.{ .call = .{ .softmax_rows = .{ .s = at.s, .p = at.p, .lds = ldn, .ldp = ldn, .row0 = i32of(r0), .spf = i32of(spf), .cols = i32of(cols), .scale = attnScale(c) } }, .grid = .{ rows, 1, 1 }, .block = .{ 256, 1, 1 } });
        try l.run(try dense(.{ .a = at.p, .lda = ldn, .b = at.vt, .ldb = ldn, .c = at.o + @as(u64, r0) * c * 2, .ldc = c, .m = rows, .n = c, .k = cols }));
    }
    try l.run(try dense(.{ .a = at.o, .lda = c, .b = w.out.w, .ldb = w.out.ldw(), .bias = w.out.b, .res = x.ptr, .c = y, .ldc = c, .m = n, .n = c, .k = c }));
    return View.dense(y, x.t, x.h, x.w, c);
}

/// HunyuanVideoMidBlock3D: resnet, attention, resnet. x is consumed (its buffer is reused); returns the output view,
/// in `x` or `spare`.
pub fn midBlock(l: Launcher, work: Work, x: View, m: MidW, spare: u64) !View {
    const r0 = try resnet(l, work, x, m.res0, spare);
    const at = try attention(l, work, r0, m.attn, x.ptr);
    return resnet(l, work, at, m.res1, spare);
}

// ---- Weight names (the checkpoint's) ----------------------------------------------------------------------------------

/// Shape of each checkpoint tensor by name, for loaders: `fn(name, shape)` is called once per tensor of the video VAE
/// (248 tensors, all fp32 in the file; this component reads them as fp16 in the layouts above).
pub fn forEachTensor(ctx: anytype, comptime f: fn (@TypeOf(ctx), []const u8, []const u32) anyerror!void) !void {
    var buf: [128]u8 = undefined;
    const S = struct {
        fn convT(c: anytype, b: []u8, comptime g: anytype, prefix: []const u8, co: u32, ci: u32, k: u32) !void {
            try g(c, try std.fmt.bufPrint(b, "{s}.weight", .{prefix}), &.{ co, ci, k, k, k });
            try g(c, try std.fmt.bufPrint(b, "{s}.bias", .{prefix}), &.{co});
        }
        fn normT(c: anytype, b: []u8, comptime g: anytype, prefix: []const u8, ch: u32) !void {
            try g(c, try std.fmt.bufPrint(b, "{s}.weight", .{prefix}), &.{ch});
            try g(c, try std.fmt.bufPrint(b, "{s}.bias", .{prefix}), &.{ch});
        }
        fn resT(c: anytype, b: []u8, comptime g: anytype, prefix: []const u8, ci: u32, co: u32) !void {
            var p: [96]u8 = undefined;
            try normT(c, b, g, try std.fmt.bufPrint(&p, "{s}.norm1", .{prefix}), ci);
            try convT(c, b, g, try std.fmt.bufPrint(&p, "{s}.conv1.conv", .{prefix}), co, ci, 3);
            try normT(c, b, g, try std.fmt.bufPrint(&p, "{s}.norm2", .{prefix}), co);
            try convT(c, b, g, try std.fmt.bufPrint(&p, "{s}.conv2.conv", .{prefix}), co, co, 3);
            if (ci != co) try convT(c, b, g, try std.fmt.bufPrint(&p, "{s}.conv_shortcut.conv", .{prefix}), co, ci, 1);
        }
        fn midT(c: anytype, b: []u8, comptime g: anytype, prefix: []const u8) !void {
            var p: [96]u8 = undefined;
            try resT(c, b, g, try std.fmt.bufPrint(&p, "{s}.mid_block.resnets.0", .{prefix}), 512, 512);
            try resT(c, b, g, try std.fmt.bufPrint(&p, "{s}.mid_block.resnets.1", .{prefix}), 512, 512);
            try normT(c, b, g, try std.fmt.bufPrint(&p, "{s}.mid_block.attentions.0.group_norm", .{prefix}), 512);
            inline for (.{ "to_q", "to_k", "to_v", "to_out.0" }) |lin| {
                try g(c, try std.fmt.bufPrint(b, "{s}.mid_block.attentions.0." ++ lin ++ ".weight", .{prefix}), &.{ 512, 512 });
                try g(c, try std.fmt.bufPrint(b, "{s}.mid_block.attentions.0." ++ lin ++ ".bias", .{prefix}), &.{512});
            }
        }
    };
    var p: [96]u8 = undefined;
    // encoder
    try S.convT(ctx, &buf, f, "encoder.conv_in.conv", 128, 3, 3);
    const enc_ch = [4][2]u32{ .{ 128, 128 }, .{ 128, 256 }, .{ 256, 512 }, .{ 512, 512 } };
    for (enc_ch, 0..) |io, i| {
        for (0..2) |r| try S.resT(ctx, &buf, f, try std.fmt.bufPrint(&p, "encoder.down_blocks.{d}.resnets.{d}", .{ i, r }), if (r == 0) io[0] else io[1], io[1]);
        if (i < 3) try S.convT(ctx, &buf, f, try std.fmt.bufPrint(&p, "encoder.down_blocks.{d}.downsamplers.0.conv.conv", .{i}), io[1], io[1], 3);
    }
    try S.midT(ctx, &buf, f, "encoder");
    try S.normT(ctx, &buf, f, "encoder.conv_norm_out", 512);
    try S.convT(ctx, &buf, f, "encoder.conv_out.conv", 32, 512, 3);
    try S.convT(ctx, &buf, f, "quant_conv", 32, 32, 1);
    // decoder
    try S.convT(ctx, &buf, f, "post_quant_conv", 16, 16, 1);
    try S.convT(ctx, &buf, f, "decoder.conv_in.conv", 512, 16, 3);
    try S.midT(ctx, &buf, f, "decoder");
    const dec_ch = [4][2]u32{ .{ 512, 512 }, .{ 512, 512 }, .{ 512, 256 }, .{ 256, 128 } };
    for (dec_ch, 0..) |io, i| {
        for (0..3) |r| try S.resT(ctx, &buf, f, try std.fmt.bufPrint(&p, "decoder.up_blocks.{d}.resnets.{d}", .{ i, r }), if (r == 0) io[0] else io[1], io[1]);
        if (i < 3) try S.convT(ctx, &buf, f, try std.fmt.bufPrint(&p, "decoder.up_blocks.{d}.upsamplers.0.conv.conv", .{i}), io[1], io[1], 3);
    }
    try S.normT(ctx, &buf, f, "decoder.conv_norm_out", 128);
    try S.convT(ctx, &buf, f, "decoder.conv_out.conv", 3, 128, 3);
}

test "parameter blocks match the CUDA structs, symbols and images" {
    try std.testing.expectEqualStrings("k6v_conv_wide", symbol(.conv_wide));
    try std.testing.expectEqualStrings("k6v_gn_apply", symbol(.gn_apply));
    try std.testing.expectEqual(Image.conv, image(.conv_narrow_s));
    try std.testing.expectEqual(Image.ops, image(.copy));
}

test "checkpoint inventory matches the pinned header: 248 tensors, names, shapes, 246,478,803 parameters" {
    const gpa = std.testing.allocator;
    const Collect = struct {
        lines: std.ArrayList([]u8) = .empty,
        elems: u64 = 0,
        fn add(self: *@This(), name: []const u8, shape: []const u32) !void {
            var line: std.ArrayList(u8) = .empty;
            defer line.deinit(std.testing.allocator);
            try line.print(std.testing.allocator, "{s} F32 ", .{name});
            var e: u64 = 1;
            for (shape, 0..) |d, i| {
                try line.print(std.testing.allocator, "{s}{d}", .{ if (i == 0) "" else "x", d });
                e *= d;
            }
            self.elems += e;
            try self.lines.append(std.testing.allocator, try line.toOwnedSlice(std.testing.allocator));
        }
    };
    var c: Collect = .{};
    defer {
        for (c.lines.items) |x| gpa.free(x);
        c.lines.deinit(gpa);
    }
    try forEachTensor(&c, Collect.add);
    std.mem.sort([]u8, c.lines.items, {}, struct {
        fn lt(_: void, x: []u8, y: []u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    const fixture = @embedFile("testdata/vae_tensors.txt");
    var it = std.mem.splitScalar(u8, fixture[std.mem.indexOfScalar(u8, fixture, '\n').? + 1 ..], '\n');
    var i: usize = 0;
    while (it.next()) |want| {
        if (want.len == 0) continue;
        try std.testing.expectEqualStrings(want, c.lines.items[i]);
        i += 1;
    }
    try std.testing.expectEqual(@as(usize, 248), i);
    try std.testing.expectEqual(i, c.lines.items.len);
    try std.testing.expectEqual(@as(u64, 246_478_803), c.elems);
}

test "repack: tap-major channel-minor rows, zero padded to 8, fp16 round to nearest even" {
    // co 1, ci 3, kernel 1 x 1 x 2 (two taps): src [c][tap]
    const src = [_]f32{ 1, 2, 3, 4, 5, 1.0 + 1.0 / 2048.0 };
    var dst: [8]f16 = undefined;
    try repackConv(&src, &dst, 1, 3, 1, 1, 2);
    // k = tap * 3 + c: tap 0: c0 1, c1 3, c2 5; tap 1: 2, 4, 1+2^-11 (a tie: rounds to even, 1.0)
    try std.testing.expectEqualSlices(f16, &.{ 1, 3, 5, 2, 4, 1, 0, 0 }, &dst);
}

test "conv builder: causal pads, upsampling and strides give the reference's output sizes" {
    const w: ConvW = .{ .w = 0, .b = 0, .co = 512, .ci = 512 };
    const x = View.dense(0, 5, 60, 108, 512);
    try std.testing.expectEqual([3]u32{ 5, 60, 108 }, convOut(x, w, .{}));
    try std.testing.expectEqual([3]u32{ 5, 120, 216 }, convOut(x, w, .{ .up_hw = 2 }));
    try std.testing.expectEqual([3]u32{ 9, 120, 216 }, convOut(x, w, .{ .up_t = 2, .up_hw = 2 }));
    const e = View.dense(0, 1, 480, 864, 128);
    try std.testing.expectEqual([3]u32{ 1, 240, 432 }, convOut(e, .{ .w = 0, .b = 0, .co = 128, .ci = 128 }, .{ .sh = 2, .sw = 2 }));
    try std.testing.expectEqual([3]u32{ 9, 240, 432 }, convOut(View.dense(0, 17, 480, 864, 128), .{ .w = 0, .b = 0, .co = 128, .ci = 128 }, .{ .st = 2, .sh = 2, .sw = 2 }));
    const l = try conv(View.dense(0, 1, 480, 864, 3), .{ .w = 0, .b = 0, .co = 128, .ci = 3 }, .{}, 0);
    try std.testing.expect(l.call == .conv_wide_s);
    try std.testing.expectEqual(@as(i32, 81), l.call.conv_wide_s.k);
    try std.testing.expectEqual(@as(i64, 88), l.call.conv_wide_s.ldw);
    const q = try conv(View.dense(0, 5, 60, 108, 16), .{ .w = 0, .b = 0, .co = 16, .ci = 16, .kt = 1, .kh = 1, .kw = 1 }, .{ .causal_pad = false }, 0);
    try std.testing.expect(q.call == .conv_narrow);
    try std.testing.expectEqual([3]u32{ (5 * 60 * 108 + 127) / 128, 1, 1 }, q.grid);
}

// ---- Host simulation of the launch stream (tests) ----------------------------------------------------------------------

/// A host stand-in for the device, for tests of the graphs: every launch's grid is checked against its parameters, every
/// byte it may read or write must lie in a declared region, and writes may not overlap the call's reads (but for the
/// aliasing the kernels allow). With `execute`, the data-movement kernels (latent_in, blend, copy, rgb_out) run on the
/// host with the kernels' arithmetic, and the decoder's last convolution (co = 3) writes a deterministic pattern, so the
/// tile placement, blending order and frame emission can be compared with the reference's loops. Addresses are host
/// addresses when executing; otherwise they are only checked, never dereferenced.
pub const Sim = struct {
    regions: [8]Region = undefined,
    nregions: usize = 0,
    execute: bool = false,
    launches: u64 = 0,
    counts: [@typeInfo(Kernel).@"enum".field_names.len]u64 = @splat(0),
    /// Called on every co = 3 convolution output when executing (the decoder's conv_out): fills y.
    seed: u64 = 0x9e3779b97f4a7c15,
    conv_out_calls: u32 = 0,

    pub const Region = struct { base: u64, bytes: u64, name: []const u8 };
    const Range = struct { lo: u64, hi: u64 };

    pub fn region(self: *Sim, base: u64, bytes: u64, name: []const u8) void {
        self.regions[self.nregions] = .{ .base = base, .bytes = bytes, .name = name };
        self.nregions += 1;
    }

    pub fn launcher(self: *Sim) Launcher {
        return .{ .ctx = self, .func = run };
    }

    fn inside(self: *Sim, r: Range) !void {
        if (r.hi == r.lo) return;
        for (self.regions[0..self.nregions]) |g| if (r.lo >= g.base and r.hi <= g.base + g.bytes) return;
        std.debug.print("access [{x}, {x}) outside every region\n", .{ r.lo, r.hi });
        return error.OutOfBounds;
    }

    fn rng(p: u64, bytes: u64) Range {
        return .{ .lo = p, .hi = p + bytes };
    }

    fn disjoint(a: Range, b: Range) !void {
        if (a.hi <= b.lo or b.hi <= a.lo or a.hi == a.lo or b.hi == b.lo) return;
        return error.Overlap;
    }

    fn run(ctx: *anyopaque, l: Launch) anyerror!void {
        const self: *Sim = @ptrCast(@alignCast(ctx));
        self.launches += 1;
        self.counts[@backingInt(l.call)] += 1;
        if (l.grid[0] == 0 or l.grid[1] == 0 or l.grid[2] == 0) return error.EmptyGrid;
        var reads: [6]Range = undefined;
        var nr: usize = 0;
        var writes: [3]Range = undefined;
        var nw: usize = 0;
        switch (l.call) {
            .conv_wide, .conv_narrow, .conv_wide_s, .conv_narrow_s => |a| {
                const wide = l.call == .conv_wide or l.call == .conv_wide_s;
                const bm: u64 = 128;
                const bn: u64 = if (wide) 128 else 32;
                const m: u64 = @as(u64, @intCast(a.to)) * @as(u64, @intCast(a.ho)) * @as(u64, @intCast(a.wo));
                if (l.grid[0] != (m + bm - 1) / bm or l.grid[1] != (@as(u64, @intCast(a.co)) + bn - 1) / bn) return error.BadGrid;
                if (l.block[0] != (if (wide) @as(u32, 256) else 128)) return error.BadBlock;
                const vec = l.call == .conv_wide or l.call == .conv_narrow;
                if (vec and (@mod(a.ci, 8) != 0 or a.x % 16 != 0)) return error.Misaligned;
                const kp: u64 = std.mem.alignForward(u64, @intCast(a.k), 8);
                if (a.ldw < kp or @mod(a.ldw, 8) != 0 or a.w % 16 != 0) return error.Misaligned;
                const xs: u64 = @as(u64, @intCast(a.ti - 1)) * @as(u64, @intCast(a.sxt)) + @as(u64, @intCast(a.hi - 1)) * @as(u64, @intCast(a.sxh)) + @as(u64, @intCast(a.wi - 1)) * @as(u64, @intCast(a.sxw)) + @as(u64, @intCast(a.ci));
                reads[nr] = rng(a.x, xs * 2);
                nr += 1;
                reads[nr] = rng(a.w, ((@as(u64, @intCast(a.co)) - 1) * @as(u64, @intCast(a.ldw)) + kp) * 2);
                nr += 1;
                if (a.bias != 0) {
                    reads[nr] = rng(a.bias, @as(u64, @intCast(a.co)) * 2);
                    nr += 1;
                }
                const ysz: u64 = ((m - 1) * @as(u64, @intCast(a.ldy)) + @as(u64, @intCast(a.co))) * @as(u64, if (a.out_f32 != 0) 4 else 2);
                writes[nw] = rng(a.y, ysz);
                nw += 1;
                if (a.res != 0 and a.res != a.y) {
                    reads[nr] = rng(a.res, ysz);
                    nr += 1;
                }
                if (self.execute and a.co == 3 and a.out_f32 == 0) {
                    const y: [*]f16 = @ptrFromInt(a.y);
                    for (0..m * 3) |i| {
                        self.seed = self.seed *% 6364136223846793005 +% 1442695040888963407;
                        const u: f32 = @floatFromInt(self.seed >> 40);
                        y[i] = @floatCast(u / 16777216.0 * 2.5 - 1.25); // [-1.25, 1.25): exercises the clamp
                    }
                    self.conv_out_calls += 1;
                }
            },
            .gn_partial, .gn_finalize => |a| {
                const p: u64 = @intCast(a.positions);
                const chunks = (p + gn_chunk - 1) / gn_chunk;
                const c: u64 = @intCast(a.c);
                if (l.call == .gn_partial) {
                    if (l.grid[0] != chunks or a.x % 16 != 0) return error.BadGrid;
                    reads[nr] = rng(a.x, p * c * 2);
                    nr += 1;
                    if (a.pass == 1) {
                        reads[nr] = rng(a.stats, @as(u64, @intCast(a.groups)) * 8);
                        nr += 1;
                    }
                    writes[nw] = rng(a.partial, chunks * @as(u64, @intCast(a.groups)) * 4);
                    nw += 1;
                } else {
                    if (l.grid[0] != @as(u32, @intCast(a.groups))) return error.BadGrid;
                    reads[nr] = rng(a.partial, chunks * @as(u64, @intCast(a.groups)) * 4);
                    nr += 1;
                    writes[nw] = rng(a.stats, @as(u64, @intCast(a.groups)) * 8);
                    nw += 1;
                }
            },
            .gn_apply => |a| {
                const n: u64 = @as(u64, @intCast(a.positions)) * @as(u64, @intCast(a.c));
                if (@as(u64, l.grid[0]) * 256 < n / 8) return error.BadGrid;
                reads[nr] = rng(a.gamma, @as(u64, @intCast(a.c)) * 2);
                nr += 1;
                reads[nr] = rng(a.beta, @as(u64, @intCast(a.c)) * 2);
                nr += 1;
                reads[nr] = rng(a.stats, @as(u64, @intCast(a.groups)) * 8);
                nr += 1;
                if (a.x != a.y) {
                    reads[nr] = rng(a.x, n * 2);
                    nr += 1;
                }
                writes[nw] = rng(a.y, n * 2);
                nw += 1;
            },
            .softmax_rows => |a| {
                const rows: u64 = l.grid[0];
                const last_valid: u64 = @intCast(@divFloor(a.row0 + @as(i32, @intCast(rows)) - 1, a.spf) * a.spf + a.spf);
                if (last_valid > @as(u64, @intCast(a.cols)) or a.cols > a.lds or a.cols > a.ldp) return error.BadShape;
                reads[nr] = rng(a.s, ((rows - 1) * @as(u64, @intCast(a.lds)) + @as(u64, @intCast(a.cols))) * 4);
                nr += 1;
                writes[nw] = rng(a.p, rows * @as(u64, @intCast(a.ldp)) * 2);
                nw += 1;
            },
            .transpose => |a| {
                reads[nr] = rng(a.x, @as(u64, @intCast(a.n)) * @as(u64, @intCast(a.c)) * 2);
                nr += 1;
                writes[nw] = rng(a.y, @as(u64, @intCast(a.c)) * @as(u64, @intCast(a.ldn)) * 2);
                nw += 1;
            },
            .latent_in => |a| {
                const n: u64 = @intCast(a.count);
                reads[nr] = rng(a.z, n * 2);
                nr += 1;
                writes[nw] = rng(a.y, n * 2);
                nw += 1;
                if (self.execute) {
                    const z: [*]const u16 = @ptrFromInt(a.z);
                    const y: [*]f16 = @ptrFromInt(a.y);
                    for (0..n) |i| {
                        const f: f32 = @bitCast(@as(u32, z[i]) << 16);
                        y[i] = @floatCast(@as(f32, @bitCast(@as(u32, bf16Bits(f * a.inv_sf)) << 16)));
                    }
                }
            },
            .rgb8_in => |a| {
                const n: u64 = @intCast(a.count);
                reads[nr] = rng(a.x, n);
                nr += 1;
                writes[nw] = rng(a.y, n * 2);
                nw += 1;
            },
            .rgb_out => |a| {
                const n: u64 = @intCast(a.count);
                reads[nr] = rng(a.x, n * 2);
                nr += 1;
                writes[nw] = rng(a.y, n);
                nw += 1;
                if (self.execute) {
                    const x: [*]const f16 = @ptrFromInt(a.x);
                    const y: [*]u8 = @ptrFromInt(a.y);
                    for (0..n) |i| y[i] = rgbOut(x[i]);
                }
            },
            .posterior => |a| {
                const n: u64 = @as(u64, @intCast(a.positions)) * @as(u64, @intCast(a.lc));
                reads[nr] = rng(a.moments, n * 4);
                nr += 1;
                if (a.eps != 0) {
                    reads[nr] = rng(a.eps, n * 2);
                    nr += 1;
                }
                writes[nw] = rng(a.latent, n * 2);
                nw += 1;
                if (a.mean != 0) {
                    writes[nw] = rng(a.mean, n * 2);
                    nw += 1;
                }
                if (a.stdv != 0) {
                    writes[nw] = rng(a.stdv, n * 2);
                    nw += 1;
                }
            },
            .blend => |a| {
                const ext: [3]i32 = .{ a.fb, a.hb, a.wb };
                const ea: [3]i32 = .{ a.fa, a.ha, a.wa };
                const ax: usize = @intCast(a.axis);
                if (a.extent > ext[ax] or a.extent > ea[ax]) return error.BadShape;
                for (0..3) |d| if (d != ax and ext[d] != ea[d]) return error.BadShape;
                reads[nr] = rng(a.a, @as(u64, @intCast(a.fa * a.ha * a.wa * a.ch)) * 2);
                nr += 1;
                writes[nw] = rng(a.b, @as(u64, @intCast(a.fb * a.hb * a.wb * a.ch)) * 2);
                nw += 1;
                if (self.execute) blendHost(a);
            },
            .copy => |a| {
                if (a.sf0 + a.f > a.sf or a.sh0 + a.h > a.sh or a.sw0 + a.w > a.sw) return error.BadShape;
                if (a.df0 + a.f > a.df or a.dh0 + a.h > a.dh or a.dw0 + a.w > a.dw) return error.BadShape;
                reads[nr] = rng(a.src, @as(u64, @intCast(a.sf * a.sh * a.sw * a.ch)) * 2);
                nr += 1;
                writes[nw] = rng(a.dst, @as(u64, @intCast(a.df * a.dh * a.dw * a.ch)) * 2);
                nw += 1;
                if (self.execute) copyHost(a);
            },
        }
        for (reads[0..nr]) |r| try self.inside(r);
        for (writes[0..nw]) |w| {
            try self.inside(w);
            for (reads[0..nr]) |r| try disjoint(w, r);
        }
    }

    fn bf16Bits(f: f32) u16 {
        const u: u32 = @bitCast(f);
        if (std.math.isNan(f)) return @truncate((u >> 16) | 0x40);
        const round = 0x7fff + ((u >> 16) & 1);
        return @truncate((u +% round) >> 16);
    }

    pub fn rgbOut(h: f16) u8 {
        const c = @min(@max(@as(f32, h), -1.0), 1.0);
        const u: f32 = @as(f16, @floatCast(c + 1.0));
        const v: f32 = @as(f16, @floatCast(u * 127.5));
        if (std.math.isNan(v)) return 0;
        return @intFromFloat(@trunc(v));
    }

    pub fn blendPair(av: f16, bv: f16, pos: u32, extent: u32) f16 {
        const r = @as(f64, @floatFromInt(pos)) / @as(f64, @floatFromInt(extent));
        const w1: f32 = @floatCast(1.0 - r);
        const w2: f32 = @floatCast(r);
        const pa: f32 = @as(f16, @floatCast(@as(f32, av) * w1));
        const pb: f32 = @as(f16, @floatCast(@as(f32, bv) * w2));
        return @floatCast(pa + pb);
    }

    fn blendHost(a: BlendArgs) void {
        const A: [*]const f16 = @ptrFromInt(a.a);
        const B: [*]f16 = @ptrFromInt(a.b);
        const fb: usize = @intCast(a.fb);
        const hb: usize = @intCast(a.hb);
        const wb: usize = @intCast(a.wb);
        const ch: usize = @intCast(a.ch);
        const e: usize = @intCast(a.extent);
        for (0..fb) |f| for (0..hb) |y| for (0..wb) |x| {
            const pos = switch (a.axis) {
                0 => f,
                1 => y,
                else => x,
            };
            if (pos >= e) continue;
            const fa = if (a.axis == 0) @as(usize, @intCast(a.fa)) - e + f else f;
            const ya = if (a.axis == 1) @as(usize, @intCast(a.ha)) - e + y else y;
            const xa = if (a.axis == 2) @as(usize, @intCast(a.wa)) - e + x else x;
            for (0..ch) |c| {
                const ib = ((f * hb + y) * wb + x) * ch + c;
                const ia = ((fa * @as(usize, @intCast(a.ha)) + ya) * @as(usize, @intCast(a.wa)) + xa) * ch + c;
                B[ib] = blendPair(A[ia], B[ib], @intCast(pos), @intCast(e));
            }
        };
    }

    fn copyHost(a: CopyArgs) void {
        const S: [*]const f16 = @ptrFromInt(a.src);
        const D: [*]f16 = @ptrFromInt(a.dst);
        const ch: usize = @intCast(a.ch);
        for (0..@intCast(a.f)) |f| for (0..@intCast(a.h)) |y| for (0..@intCast(a.w)) |x| for (0..ch) |c| {
            const is = (((@as(usize, @intCast(a.sf0)) + f) * @as(usize, @intCast(a.sh)) + @as(usize, @intCast(a.sh0)) + y) * @as(usize, @intCast(a.sw)) + @as(usize, @intCast(a.sw0)) + x) * ch + c;
            const id = (((@as(usize, @intCast(a.df0)) + f) * @as(usize, @intCast(a.dh)) + @as(usize, @intCast(a.dh0)) + y) * @as(usize, @intCast(a.dw)) + @as(usize, @intCast(a.dw0)) + x) * ch + c;
            D[id] = S[is];
        };
    }
};
