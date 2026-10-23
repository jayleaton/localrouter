//! The DiT's attention sites and the streaming attention plan (kernels/cuda/kandinsky6/dit/attention.cu). Every site is
//! softmax(q k^T / sqrt(128)) v, bidirectional, head dim 128, bf16, computed one 64-query x 64-key tile at a time: no
//! score matrix exists (at 50,220 visual tokens the full [32, 50220, 50220] fp32 scores would be ~300 GiB, ~150 GiB in
//! bf16). The only memory is the kernel's 32 KiB of shared staging a block. Upstream (`dit.py`) selects SDPA for the
//! masked sites (text_token_padding) and the configured engine elsewhere; the reference for this engine is SDPA at
//! every site with the default scale. Pure and I/O-free.

const std = @import("std");
const layout = @import("layout.zig");

const View = layout.View;

pub const head_dim: u32 = 128;
/// Queries a block (4 warps of 16) and keys a tile.
pub const block_rows: u32 = 64;
pub const tile_keys: u32 = 64;
/// Dynamic shared memory a block: two 64-key staging slots of 128 bf16 columns.
pub const shared_bytes: u32 = 2 * 64 * 128 * 2;
pub const threads: u32 = 128;

/// Which keys a query sees: all of them, or those whose byte in `kmask` is nonzero (torch's bool key-padding mask
/// [B, 1, 1, K], True = keep; SDPA turns it into an additive 0 / -inf mask). Upstream passes the text mask to the text
/// self-attention and to both text cross-attentions only.
pub const Mask = union(enum) {
    none,
    keys: struct { ptr: u64, valid: u32 }, // device u8 [nkeys]; `valid`: how many are nonzero (host-known, >= 1)
};

/// One attention call: q [W, heads * 128], k / v [nkeys, heads * 128], out [W, heads * 128], all bf16 row views.
pub const Call = struct {
    q: View,
    k: View,
    v: View,
    out: View,
    heads: u32,
    mask: Mask = .none,
    scale: f32 = layout.attn_scale,

    pub fn validate(c: Call) error{ InvalidAttention, ViewMismatch }!void {
        const width = @as(u64, c.heads) * head_dim;
        try c.q.expect(.bf16, c.q.rows, width);
        try c.k.expect(.bf16, c.k.rows, width);
        try c.v.expect(.bf16, c.k.rows, width);
        try c.out.expect(.bf16, c.q.rows, width);
        if (c.heads == 0 or c.q.rows == 0 or c.k.rows == 0) return error.InvalidAttention;
        if (c.q.rows > std.math.maxInt(i32) or c.k.rows > std.math.maxInt(i32)) return error.InvalidAttention;
        // 32-bit query loads and output stores; 16-byte cp.async of keys and values
        if (c.q.ld % 2 != 0 or c.out.ld % 2 != 0 or c.q.ptr % 4 != 0 or c.out.ptr % 4 != 0) return error.InvalidAttention;
        if (c.k.ld % 8 != 0 or c.v.ld % 8 != 0 or c.k.ptr % 16 != 0 or c.v.ptr % 16 != 0) return error.InvalidAttention;
        switch (c.mask) {
            .none => {},
            .keys => |m| if (m.ptr == 0 or m.valid == 0 or m.valid > c.k.rows) return error.InvalidAttention,
        }
    }

    pub fn grid(c: Call) [3]u32 {
        return .{ @intCast((c.q.rows + block_rows - 1) / block_rows), c.heads, 1 };
    }
};

/// The bytes a dense score matrix would take ([heads, W, nkeys] in `elem` bytes): what streaming avoids.
pub fn denseScoreBytes(heads: u64, w: u64, nkeys: u64, elem: u64) u64 {
    return heads * w * nkeys * elem;
}

/// The attention sites of one DiT call (fused video + audio path), for the coverage manifest and the capture names.
pub const Site = enum {
    text_video_self, // 4 encoder blocks, model_dim 4096, 32 heads, text rope, mask
    text_audio_self, // 4 encoder blocks, model_dim_a 2048, 16 heads, text rope, mask
    video_self, // visual tokens, 32 heads, visual rope (RoPE3D)
    audio_self, // audio rows, 16 heads, audio rope (RoPE1D, freqs x 0.144)
    video_text, // q visual pre-CA, k / v video text, no rope, mask
    audio_text, // q audio pre-CA, k / v audio text, no rope, mask
    video_from_audio, // va_cross_attention: q vis_for_va (visual rope), k / v audio pre-CA (audio rope), 32 heads
    audio_from_video, // av_cross_attention: q aud_for_av (audio rope), k / v visual pre-CA (visual rope), 16 heads

    pub fn name(s: Site) []const u8 {
        return @tagName(s);
    }

    pub fn masked(s: Site) bool {
        return switch (s) {
            .text_video_self, .text_audio_self, .video_text, .audio_text => true,
            else => false,
        };
    }

    /// (queries, keys, heads) at a shape.
    pub fn dims(s: Site, c: layout.Config, sh: layout.Shape) [3]u64 {
        const v = sh.tokens();
        return switch (s) {
            .text_video_self => .{ sh.text, sh.text, c.heads() },
            .text_audio_self => .{ sh.text, sh.text, c.headsA() },
            .video_self => .{ v, v, c.heads() },
            .audio_self => .{ sh.audio, sh.audio, c.headsA() },
            .video_text => .{ v, sh.text, c.heads() },
            .audio_text => .{ sh.audio, sh.text, c.headsA() },
            .video_from_audio => .{ v, sh.audio, c.heads() },
            .audio_from_video => .{ sh.audio, v, c.headsA() },
        };
    }
};

const testing = std.testing;

test "no site materializes scores; the dense matrix would not fit" {
    const c = layout.Config.distill;
    const sh = layout.Shape.preset(1024);
    const d = Site.video_self.dims(c, sh);
    try testing.expectEqual([3]u64{ 50220, 50220, 32 }, d);
    // ~150 GiB in bf16 for the visual self-attention alone (the discovery's sizing input), 2x in fp32.
    try testing.expect(denseScoreBytes(d[2], d[0], d[1], 2) > 150 * (1 << 30));
    try testing.expectEqual([3]u64{ 218, 50220, 16 }, Site.audio_from_video.dims(c, sh));
    try testing.expectEqual([3]u64{ 50220, 218, 32 }, Site.video_from_audio.dims(c, sh));
    try testing.expect(Site.video_text.masked() and !Site.video_from_audio.masked());
}

test "call validation: alignment, mask and ragged counts" {
    const w: u64 = 32 * 128;
    var call: Call = .{
        .q = View.of(0x1000, .bf16, 50220, w),
        .k = View.of(0x100000, .bf16, 218, w),
        .v = View.of(0x200000, .bf16, 218, w),
        .out = View.of(0x300000, .bf16, 50220, w),
        .heads = 32,
    };
    try call.validate();
    try testing.expectEqual([3]u32{ 785, 32, 1 }, call.grid()); // 50220 = 784 * 64 + 44: a ragged last block
    call.mask = .{ .keys = .{ .ptr = 0x400000, .valid = 0 } };
    try testing.expectError(error.InvalidAttention, call.validate());
    call.mask = .{ .keys = .{ .ptr = 0x400000, .valid = 218 } };
    try call.validate();
    call.k.ptr += 2; // not 16-byte aligned
    try testing.expectError(error.InvalidAttention, call.validate());
}
