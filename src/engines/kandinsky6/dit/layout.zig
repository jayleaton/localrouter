//! The Kandinsky-6 29B Pro-distill DiT's host-side contract: the transformer config (HF revision
//! 1018d5828716e8bea3de99609f45c924de9c5564, `transformer/config.json`), the typed tensor views the launchers take, the
//! token shapes of a request, the modulation chunk offsets and the checkpoint's tensor manifest (names, file dtypes,
//! shapes), all as upstream kandinsky-6 eac06e39 `kandinsky/core/components/{dit,piflow_dit}.py` define them.
//! Pure and I/O-free: no CUDA, no allocation outside the manifest test.

const std = @import("std");

pub const Dtype = enum(u8) {
    bf16,
    f32,
    u8,
    i32,

    pub fn size(d: Dtype) u64 {
        return switch (d) {
            .bf16 => 2,
            .f32, .i32 => 4,
            .u8 => 1,
        };
    }

    /// The safetensors header's name for it.
    pub fn header(d: Dtype) []const u8 {
        return switch (d) {
            .bf16 => "BF16",
            .f32 => "F32",
            .u8 => "U8",
            .i32 => "I32",
        };
    }
};

/// A row-major 2-D device tensor: `rows` rows of `cols` elements, row r at `ptr + r * ld * dtype.size()`. Every DiT
/// activation is [tokens, features]; a token's heads are contiguous within its row (head h at h * 128). A view never
/// owns its memory: weights are immutable views into the loader's buffers, activations views into the caller's
/// residual buffers or the block scratch.
pub const View = struct {
    ptr: u64,
    dtype: Dtype,
    rows: u64,
    cols: u64,
    ld: u64,

    pub fn of(ptr: u64, dtype: Dtype, rows: u64, cols: u64) View {
        return .{ .ptr = ptr, .dtype = dtype, .rows = rows, .cols = cols, .ld = cols };
    }

    pub fn bytes(v: View) u64 {
        if (v.rows == 0) return 0;
        return ((v.rows - 1) * v.ld + v.cols) * v.dtype.size();
    }

    /// Rows [r0, r0 + n).
    pub fn slice(v: View, r0: u64, n: u64) View {
        std.debug.assert(r0 + n <= v.rows);
        return .{ .ptr = v.ptr + r0 * v.ld * v.dtype.size(), .dtype = v.dtype, .rows = n, .cols = v.cols, .ld = v.ld };
    }

    /// Columns [c0, c0 + n) of every row (a chunk of a modulation row, one head's columns ...).
    pub fn columns(v: View, c0: u64, n: u64) View {
        std.debug.assert(c0 + n <= v.cols);
        return .{ .ptr = v.ptr + c0 * v.dtype.size(), .dtype = v.dtype, .rows = v.rows, .cols = n, .ld = v.ld };
    }

    pub fn expect(v: View, dtype: Dtype, rows: u64, cols: u64) error{ViewMismatch}!void {
        if (v.dtype != dtype or v.rows != rows or v.cols != cols or v.ld < cols or v.ptr == 0) return error.ViewMismatch;
    }
};

/// The transformer config of the selected checkpoint (Pro-distill, `PiFlowDiffusionTransformer3D` with n_grid 10).
/// The K tile of the native (cuBLASLt) kernel of the patch embedding on SM121; see `Config.patchKFirst`.
pub const native_patch_k_tile = 32;

pub const Config = struct {
    model_dim: u32 = 4096,
    model_dim_a: u32 = 2048,
    time_dim: u32 = 1024,
    ff_dim: u32 = 16384,
    ff_dim_a: u32 = 7168,
    head_dim: u32 = 128, // sum(axes_dims) = 32 + 48 + 48, the same for audio (axes_dims_a)
    axes_dims: [3]u32 = .{ 32, 48, 48 },
    in_text_dim: u32 = 3584,
    in_text_dim2: u32 = 768,
    in_visual_dim: u32 = 16,
    in_audio_dim: u32 = 40,
    n_grid: u32 = 10,
    out_visual_dim: u32 = 160, // = dx 16 x n_grid
    out_audio_dim: u32 = 400, // = dx 40 x n_grid
    num_text_blocks: u32 = 4,
    num_visual_blocks: u32 = 60,
    token_types: u32 = 2, // visual_token_type_num_embeddings: generated frame / I2V reference frame
    audio_freqs_scaling: f32 = 0.144,
    scale_factor: [3]f32 = .{ 1.0, 2.0, 2.0 }, // RoPE3D's divisor per axis (the pipeline's visual rope)
    // Flags of the fused block: ca_rope, cross_gates, fix_modulation and text_token_padding all true in this checkpoint;
    // the block code implements exactly that combination.

    pub const distill: Config = .{};

    /// 2 * in_visual_dim + 1 (visual_cond: noisy latent, condition latent, mask) channels.
    pub fn visualIn(c: Config) u32 {
        return 2 * c.in_visual_dim + 1;
    }
    /// The patch embedding's input features: patch (1, 2, 2) of `visualIn` channels.
    pub fn patchIn(c: Config) u32 {
        return 4 * c.visualIn();
    }
    /// The patch embedding GEMM's k_first (launch.Kernels.gemm): K mod the native kernel's K tile. Captured on SM121
    /// (D1 window, T0 1791816926): cuBLASLt algo 23, tile 256x128, stages 32x3 (K tile 32) for [50,220, 132] x
    /// [4096, 132]^T, so 132 mod 32 = 4 elements are reduced first. Every other DiT GEMM has K % 16 == 0 and needs none.
    pub fn patchKFirst(c: Config) u32 {
        return c.patchIn() % native_patch_k_tile;
    }
    pub fn heads(c: Config) u32 {
        return c.model_dim / c.head_dim;
    }
    pub fn headsA(c: Config) u32 {
        return c.model_dim_a / c.head_dim;
    }
    pub fn dxVisual(c: Config) u32 {
        return c.out_visual_dim / c.n_grid;
    }
    pub fn dxAudio(c: Config) u32 {
        return c.out_audio_dim / c.n_grid;
    }
};

/// nn.LayerNorm's default epsilon (every LayerNorm of the DiT).
pub const ln_eps: f32 = 1e-5;
/// F.scaled_dot_product_attention's default scale for head dim 128: 1 / sqrt(128) rounded to f32.
pub const attn_scale: f32 = @bitCast(@as(u32, 0x3db504f3));
/// RoPE pairs a head: head_dim / 2 (16 + 24 + 24 for the visual axes; RoPE1D's 64 for text and audio).
pub const rope_pairs: u32 = 64;

/// One request's token counts. Video latents are [T, H, W, 33] channels-last at VAE stride 8 (spatial) and 4 (time);
/// tokens = T * (H / 2) * (W / 2), row order (t, h', w'). The native default visual_cond_scheme "pretrain"
/// (pipeline/config.py:134; pro-distill.yaml sets no override) serves T2V and I2V alike: I2V puts the reference latent
/// in the separate cond[:, 0] channels with the frame-0 mask and keeps the noisy video state, T = 31 and the token
/// count (denoise_loop.py:30-31,49-50,60,64). The optional "tail_cond_first_frame" branch (T + 1, token-type ids) and
/// the explicit "i2v" overwrite branch are not supported here: neither has a native-byte gate.
pub const Shape = struct {
    t: u32, // latent frames
    h: u32, // latent height (pixels / 8)
    w: u32,
    audio: u32, // audio latent rows
    text: u32, // text tokens (padded length; the mask marks the real ones)

    /// The 5 s preset: 864x480, 121 frames -> 31 latent frames ((31 - 1) * 4 + 1), 218 audio rows, 1024 text tokens
    /// (padded; `text_embedder.max_length`). T2V and the default I2V alike (pro-distill.yaml:12-20).
    pub fn preset(text: u32) Shape {
        return .{ .t = 31, .h = 480 / 8, .w = 864 / 8, .audio = 218, .text = text };
    }

    pub fn tokens(s: Shape) u64 {
        return @as(u64, s.t) * (s.h / 2) * (s.w / 2);
    }
    pub fn perFrame(s: Shape) u64 {
        return @as(u64, s.h / 2) * (s.w / 2);
    }

    pub const max_text: u32 = 1024;

    pub fn validate(s: Shape) error{UnsupportedShape}!void {
        if (s.t == 0 or s.h == 0 or s.w == 0 or s.h % 2 != 0 or s.w % 2 != 0) return error.UnsupportedShape;
        if (s.audio == 0 or s.text == 0 or s.text > max_text) return error.UnsupportedShape;
        // RoPE3D's tables hold 128 positions an axis (max_pos), RoPE1D's 2048.
        if (s.t > 128 or s.h / 2 > 128 or s.w / 2 > 128 or s.audio > 2048) return error.UnsupportedShape;
    }
};

/// Modulation chunk offsets (elements of the modulation row) as `torch.chunk` / `torch.split` cut it.
pub const Mod = struct {
    /// TransformerDecoderBlock.visual_modulation [9 d]: chunk 3 -> (self-attention, cross-attention, FFN), each chunk
    /// 3 -> (shift, scale, gate). part 0..2, which 0 shift / 1 scale / 2 gate.
    pub fn dec(d: u64, part: u64, which: u64) u64 {
        return (part * 3 + which) * d;
    }
    /// TransformerEncoderBlock.text_modulation [6 d]: (self-attention, FFN) x (shift, scale, gate).
    pub fn enc(d: u64, part: u64, which: u64) u64 {
        return (part * 3 + which) * d;
    }
    /// OutLayer / OutLayerAudio modulation [2 d]: chunk 2 -> (shift, scale).
    pub fn out(d: u64, which: u64) u64 {
        return which * d;
    }
    /// The fused block's cross gates (cross_gates): va_modulation [2 D + Da] splits (shift D, scale D, gate Da),
    /// av_modulation [2 Da + D] splits (shift Da, scale Da, gate D). Returns (offset, length).
    pub fn cross(d_self: u64, d_other: u64, which: u64) [2]u64 {
        return switch (which) {
            0 => .{ 0, d_self },
            1 => .{ d_self, d_self },
            else => .{ 2 * d_self, d_other },
        };
    }
};

/// Every tensor of the checkpoint's transformer file: name, file dtype, shape. Upstream loads each as BF16
/// (`runtime/weights.py` casts the F32 ones); the engine's weight views are BF16 throughout.
pub fn forEachTensor(c: Config, visitor: anytype) !void {
    var nb: [160]u8 = undefined;
    const D: u64 = c.model_dim;
    const Da: u64 = c.model_dim_a;
    const T: u64 = c.time_dim;
    const hd: u64 = c.head_dim;
    const V = struct {
        fn one(v: anytype, buf: []u8, comptime fmt: []const u8, args: anytype, dt: Dtype, shape: []const u64) !void {
            try v.tensor(try std.fmt.bufPrint(buf, fmt, args), dt, shape);
        }
        fn linear(v: anytype, buf: []u8, prefix: []const u8, dt: Dtype, out: u64, in: u64, bias: bool) !void {
            try one(v, buf, "{s}.weight", .{prefix}, dt, &.{ out, in });
            if (bias) try one(v, buf, "{s}.bias", .{prefix}, dt, &.{out});
        }
        fn attn(v: anytype, buf: []u8, prefix: []const u8, q: u64, kv: u64, head: u64) !void {
            var pb: [160]u8 = undefined;
            try linear(v, buf, try std.fmt.bufPrint(&pb, "{s}.to_query", .{prefix}), .bf16, q, q, true);
            try linear(v, buf, try std.fmt.bufPrint(&pb, "{s}.to_key", .{prefix}), .bf16, q, kv, true);
            try linear(v, buf, try std.fmt.bufPrint(&pb, "{s}.to_value", .{prefix}), .bf16, q, kv, true);
            try linear(v, buf, try std.fmt.bufPrint(&pb, "{s}.out_layer", .{prefix}), .bf16, q, q, true);
            try one(v, buf, "{s}.query_norm.weight", .{prefix}, .f32, &.{head});
            try one(v, buf, "{s}.key_norm.weight", .{prefix}, .f32, &.{head});
        }
        fn ffn(v: anytype, buf: []u8, prefix: []const u8, d: u64, ff: u64) !void {
            try one(v, buf, "{s}.feed_forward.net.0.proj.weight", .{prefix}, .bf16, &.{ ff, d });
            try one(v, buf, "{s}.feed_forward.net.2.weight", .{prefix}, .bf16, &.{ d, ff });
        }
        fn textEmb(v: anytype, buf: []u8, prefix: []const u8, out: u64, in: u64) !void {
            var pb: [160]u8 = undefined;
            try linear(v, buf, try std.fmt.bufPrint(&pb, "{s}.in_layer", .{prefix}), .f32, out, in, true);
            try one(v, buf, "{s}.norm.weight", .{prefix}, .f32, &.{out});
            try one(v, buf, "{s}.norm.bias", .{prefix}, .f32, &.{out});
        }
    };
    var pb: [160]u8 = undefined;
    try V.linear(visitor, &nb, "visual_embeddings.in_layer", .f32, D, c.patchIn(), true);
    try V.one(visitor, &nb, "visual_token_type_embeddings.weight", .{}, .f32, &.{ c.token_types, D });
    try V.linear(visitor, &nb, "out_layer.modulation.out_layer", .bf16, 2 * D, T, true);
    try V.linear(visitor, &nb, "out_layer.out_layer", .bf16, 4 * @as(u64, c.out_visual_dim), D, true);
    try V.textEmb(visitor, &nb, "audio_embeddings", Da, c.in_audio_dim);
    try V.linear(visitor, &nb, "audio_out_layer.modulation.out_layer", .bf16, 2 * Da, T, true);
    try V.linear(visitor, &nb, "audio_out_layer.out_layer", .bf16, c.out_audio_dim, Da, true);
    inline for (.{ .{ "video", D, @as(u64, c.ff_dim) }, .{ "audio", Da, @as(u64, c.ff_dim_a) } }) |m| {
        const p = m[0];
        const md: u64 = m[1];
        try V.linear(visitor, &nb, p ++ "_time_embeddings.timestep_embedder.linear_1", .f32, T, md, true);
        try V.linear(visitor, &nb, p ++ "_time_embeddings.timestep_embedder.linear_2", .f32, T, T, true);
        try V.textEmb(visitor, &nb, p ++ "_text_embeddings", md, c.in_text_dim);
        try V.textEmb(visitor, &nb, p ++ "_pooled_text_embeddings", T, c.in_text_dim2);
        for (0..c.num_text_blocks) |i| {
            const b = try std.fmt.bufPrint(&pb, p ++ "_text_transformer_blocks.{d}", .{i});
            var ab: [160]u8 = undefined;
            try V.attn(visitor, &nb, try std.fmt.bufPrint(&ab, "{s}.attn", .{b}), md, md, hd);
            try V.ffn(visitor, &nb, b, md, m[2]);
            try V.linear(visitor, &nb, try std.fmt.bufPrint(&ab, "{s}.text_modulation.out_layer", .{b}), .bf16, 6 * md, T, true);
        }
    }
    for (0..c.num_visual_blocks) |i| {
        inline for (.{ .{ "video_dec_block", D, @as(u64, c.ff_dim) }, .{ "audio_dec_block", Da, @as(u64, c.ff_dim_a) } }) |m| {
            const b = try std.fmt.bufPrint(&pb, "visual_transformer_blocks.{d}." ++ m[0], .{i});
            var ab: [160]u8 = undefined;
            try V.attn(visitor, &nb, try std.fmt.bufPrint(&ab, "{s}.self_attention", .{b}), m[1], m[1], hd);
            try V.attn(visitor, &nb, try std.fmt.bufPrint(&ab, "{s}.cross_attention", .{b}), m[1], m[1], hd);
            try V.ffn(visitor, &nb, b, m[1], m[2]);
            try V.linear(visitor, &nb, try std.fmt.bufPrint(&ab, "{s}.visual_modulation.out_layer", .{b}), .bf16, 9 * @as(u64, m[1]), T, true);
        }
        var ab: [160]u8 = undefined;
        try V.attn(visitor, &nb, try std.fmt.bufPrint(&ab, "visual_transformer_blocks.{d}.va_cross_attention", .{i}), D, Da, hd);
        try V.attn(visitor, &nb, try std.fmt.bufPrint(&ab, "visual_transformer_blocks.{d}.av_cross_attention", .{i}), Da, D, hd);
        try V.linear(visitor, &nb, try std.fmt.bufPrint(&ab, "visual_transformer_blocks.{d}.va_modulation.out_layer", .{i}), .bf16, 2 * D + Da, T, true);
        try V.linear(visitor, &nb, try std.fmt.bufPrint(&ab, "visual_transformer_blocks.{d}.av_modulation.out_layer", .{i}), .bf16, 2 * Da + D, T, true);
    }
}

/// The manifest's size: tensors and elements.
pub fn manifestTotals(c: Config) struct { tensors: u64, elements: u64 } {
    const Count = struct {
        tensors: u64 = 0,
        elements: u64 = 0,
        pub fn tensor(self: *@This(), name: []const u8, dt: Dtype, shape: []const u64) !void {
            _ = name;
            _ = dt;
            var n: u64 = 1;
            for (shape) |d| n *= d;
            self.tensors += 1;
            self.elements += n;
        }
    };
    var k: Count = .{};
    forEachTensor(c, &k) catch unreachable;
    return .{ .tensors = k.tensors, .elements = k.elements };
}

const testing = std.testing;

test "preset token counts" {
    const t2v = Shape.preset(1024);
    try testing.expectEqual(@as(u64, 31 * 30 * 54), t2v.tokens());
    try testing.expectEqual(@as(u64, 50220), t2v.tokens());
    try testing.expectEqual(@as(u64, 1620), t2v.perFrame());
    try t2v.validate();
    try testing.expectError(error.UnsupportedShape, (Shape{ .t = 31, .h = 61, .w = 108, .audio = 218, .text = 8 }).validate());
    try testing.expectError(error.UnsupportedShape, Shape.preset(1025).validate());
}

test "config derivations" {
    const c = Config.distill;
    try testing.expectEqual(@as(u32, 132), c.patchIn());
    try testing.expectEqual(@as(u32, 4), c.patchKFirst());
    try testing.expectEqual(@as(u32, 32), c.heads());
    try testing.expectEqual(@as(u32, 16), c.headsA());
    try testing.expectEqual(@as(u32, 16), c.dxVisual());
    try testing.expectEqual(@as(u32, 40), c.dxAudio());
    try testing.expectEqual(c.head_dim, c.axes_dims[0] + c.axes_dims[1] + c.axes_dims[2]);
    try testing.expectEqual(@as(f32, 1.0) / @sqrt(@as(f32, 128.0)), attn_scale);
    // va_modulation [D, D, Da] and av_modulation [Da, Da, D] (dit.py:485-486): the gates' (offset, length)
    try testing.expectEqual([2]u64{ 8192, 2048 }, Mod.cross(4096, 2048, 2));
    try testing.expectEqual([2]u64{ 4096, 4096 }, Mod.cross(2048, 4096, 2));
    try testing.expectEqual(@as(u64, 7 * 4096), Mod.dec(4096, 2, 1));
}

test "manifest equals the archived distill transformer header" {
    // Header: HF kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers @ 1018d582 transformer/diffusion_pytorch_model
    // .safetensors (archived header sha256 63242b7d593e847898ef571b63a7d5dab5fb764213df17dc09877d715411ebc8). Digest:
    // sha256 over the names sorted bytewise, each line "name\tDTYPE\td0xd1...\n" (fixtures/manifest_digest.py).
    const c = Config.distill;
    const gpa = testing.allocator;
    const Entry = struct { line: []u8 };
    const Collect = struct {
        gpa: std.mem.Allocator,
        list: []Entry,
        n: usize = 0,
        pub fn tensor(self: *@This(), name: []const u8, dt: Dtype, shape: []const u64) !void {
            var sb: [64]u8 = undefined;
            var len: usize = 0;
            for (shape, 0..) |d, i| {
                if (i > 0) {
                    sb[len] = 'x';
                    len += 1;
                }
                len += (try std.fmt.bufPrint(sb[len..], "{d}", .{d})).len;
            }
            self.list[self.n] = .{ .line = try std.fmt.allocPrint(self.gpa, "{s}\t{s}\t{s}\n", .{ name, dt.header(), sb[0..len] }) };
            self.n += 1;
        }
    };
    const totals = manifestTotals(c);
    try testing.expectEqual(@as(u64, 4471), totals.tensors);
    try testing.expectEqual(@as(u64, 30_139_423_760), totals.elements);
    const list = try gpa.alloc(Entry, totals.tensors);
    defer gpa.free(list);
    var col: Collect = .{ .gpa = gpa, .list = list };
    defer for (list[0..col.n]) |e| gpa.free(e.line);
    try forEachTensor(c, &col);
    std.mem.sort(Entry, list, {}, struct {
        fn lt(_: void, a: Entry, b: Entry) bool {
            const na = a.line[0..std.mem.indexOfScalar(u8, a.line, '\t').?];
            const nb = b.line[0..std.mem.indexOfScalar(u8, b.line, '\t').?];
            return std.mem.order(u8, na, nb) == .lt;
        }
    }.lt);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (list) |e| h.update(e.line);
    var d: [32]u8 = undefined;
    h.final(&d);
    var hex: [64]u8 = undefined;
    _ = try std.fmt.bufPrint(&hex, "{x}", .{&d});
    try testing.expectEqualStrings("a5339cbebf5506a6e3f4700acaabd16b20bd1186eb6c55cf9da4ce346c3bc9aa", &hex);
}
