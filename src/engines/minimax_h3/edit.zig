//! MiniMax H3-Image edits on the host (docs/dev/H3-IMAGE.md section 1.6; #157's minimax-h3-image-edit*.json; the twin's
//! `h3i/edit.py`): the sizes the graph derives from the request's pictures (`Plan`), the encoder's posterior sample and
//! normalization (`posterior`), and the cap on pictures. The pixel steps are `image.zig`'s.
//!
//! The graph, per request (pictures in order; picture 1 is the one edited):
//!   picture 1   ImageScaleToTotalPixels(area, MP, 32): MP 1.0 with a mask (the masked workflow), 2.0 without
//!               (whole scene); pictures 2 .. n are used as loaded
//!   each        Qwen's copy at `_size_short_edge(w, h, 1024, 1024)` and the VAE's at `(.., 1024, 2048)`, through
//!               `_resize_to` (the picture itself when the size matches, else the 8-bit lanczos)
//!   target      the request's size, else picture 1's scaled size (GetImageSize of node 11)
//!   masked      after the decode: ImageScale(picture 1 scaled, lanczos, target, crop center) as the background, the
//!               request's mask (1 - alpha / 255) bilinear to the target, ImageCompositeMasked, SaveImage's uint8
//! The posterior: fma(exp(0.5 clamp(logvar, -30, 20)), noise, mean) (CPU addcmul is fused), then (z - mean) / std
//! against the checkpoint's fp16 statistics, in fp32; the noise is the portable generator at seed 42 and exp the
//! portable fdlibm exp rounded to fp32 (declared substitutions for CPU torch.randn and torch.exp; test_edit.py).

const std = @import("std");
const qi = @import("qwen_image");
const image = @import("image.zig");
const smath = qi.smath;

/// The pictures a request may carry: #157's cap (`references`: image_1 .. image_5).
pub const max_pictures = 5;
/// `IMAGE_ENCODE_SEED`: the posterior's noise, whatever the request's seed.
pub const encode_seed: u64 = 42;
/// node 11's megapixels in the masked and the whole-scene workflow.
pub const masked_megapixels = 1.0;
pub const unmasked_megapixels = 2.0;
/// The two copies' short edges (IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE).
pub const vision_short_edge = 1024;
pub const vae_max_short_edge = 2048;
/// Steps a masked and a whole-scene edit default to (#157's workflows).
pub const masked_steps = 6;
pub const unmasked_steps = 30;

pub const Size = struct { w: u32, h: u32 };

/// One picture's sizes: as loaded, as the graph uses it (picture 1 scaled), Qwen's copy, the VAE's copy.
pub const Picture = struct { loaded: Size, source: Size, vision: Size, vae: Size };

pub const Plan = struct {
    pictures: [max_pictures]Picture = undefined,
    n: u32,
    target: Size,
    masked: bool,

    /// The sizes for `loaded` pictures (1 .. max_pictures), a mask or not, and the request's size (0 x 0: none).
    pub fn of(loaded: []const Size, masked: bool, width: u32, height: u32) !Plan {
        if (loaded.len == 0) return error.NoPicture;
        if (loaded.len > max_pictures) return error.TooManyPictures;
        var p: Plan = .{ .n = @intCast(loaded.len), .target = undefined, .masked = masked };
        for (loaded, 0..) |l, i| {
            if (l.w == 0 or l.h == 0) return error.BadPicture;
            var src = l;
            if (i == 0) {
                const s = image.totalPixelsSize(l.w, l.h, if (masked) masked_megapixels else unmasked_megapixels, 32);
                if (s[0] == 0 or s[1] == 0) return error.BadPicture;
                src = .{ .w = s[0], .h = s[1] };
            }
            const v = image.shortEdgeSize(src.w, src.h, vision_short_edge, vision_short_edge);
            const e = image.shortEdgeSize(src.w, src.h, vision_short_edge, vae_max_short_edge);
            p.pictures[i] = .{ .loaded = l, .source = src, .vision = .{ .w = v[0], .h = v[1] }, .vae = .{ .w = e[0], .h = e[1] } };
        }
        p.target = if (width != 0 and height != 0) .{ .w = width, .h = height } else p.pictures[0].source;
        return p;
    }

    pub fn list(p: *const Plan) []const Picture {
        return p.pictures[0..p.n];
    }

    /// The text tokens the pictures take: each `(w/32)(h/32)` vision rows and its two markers (the labels are the
    /// tokenizer's).
    pub fn visionRows(p: *const Plan) u32 {
        var n: u32 = 0;
        for (p.list()) |q| n += (q.vision.w / 32) * (q.vision.h / 32) + 2;
        return n;
    }

    /// The DiT rows the references take: each VAE copy's (w/32)(h/32).
    pub fn refRows(p: *const Plan) u32 {
        var n: u32 = 0;
        for (p.list()) |q| n += (q.vae.w / 32) * (q.vae.h / 32);
        return n;
    }
};

/// `moments` fp16 [48, P] (the encoder's canvas: 24 means then 24 log-variances), the checkpoint's `latents_mean` /
/// `latents_std` fp16 [24], `noise` fp32 [24 * P] -> the normalized latent fp32 [24, P] in `out`.
pub fn posterior(moments: []const f16, mean: *const [24]f16, stdv: *const [24]f16, noise: []const f32, out: []f32) void {
    const P = noise.len / 24;
    std.debug.assert(moments.len == 48 * P and noise.len == 24 * P and out.len == 24 * P);
    for (0..24) |c| {
        const m: f32 = mean[c];
        const s: f32 = stdv[c];
        for (0..P) |p| {
            const mu: f32 = moments[c * P + p];
            const lv: f32 = moments[(24 + c) * P + p];
            const half: f32 = 0.5 * std.math.clamp(lv, -30.0, 20.0);
            const e: f32 = @floatCast(smath.exp(@as(f64, half)));
            const z = @mulAdd(f32, e, noise[c * P + p], mu);
            out[c * P + p] = (z - m) / s;
        }
    }
}

test "posterior: the twin's edit.py (edit_image.json), bit for bit, noise included" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("edit_image.json"), .{});
    defer parsed.deinit();
    const po = parsed.value.object.get("posterior").?.object;
    const P: usize = @intCast(po.get("p").?.integer);
    const mb = try image.unhex(gpa, po.get("moments").?.string);
    defer gpa.free(mb);
    const moments = try gpa.alloc(f16, 48 * P);
    defer gpa.free(moments);
    @memcpy(std.mem.sliceAsBytes(moments), mb);
    var mean: [24]f16 = undefined;
    var stdv: [24]f16 = undefined;
    const meb = try image.unhex(gpa, po.get("mean").?.string);
    defer gpa.free(meb);
    const stb = try image.unhex(gpa, po.get("std").?.string);
    defer gpa.free(stb);
    @memcpy(std.mem.sliceAsBytes(&mean), meb);
    @memcpy(std.mem.sliceAsBytes(&stdv), stb);
    const noise = try gpa.alloc(f32, 24 * P);
    defer gpa.free(noise);
    qi.noise.fill(encode_seed, noise);
    try image.expectSha(po.get("noise").?.string, noise);
    const z = try gpa.alloc(f32, 24 * P);
    defer gpa.free(z);
    posterior(moments, &mean, &stdv, noise, z);
    try image.expectSha(po.get("z").?.string, z);
}

test "the plan: picture 1 at 1 MP (masked) or 2 MP, both copies, the target, the caps" {
    const sq = try Plan.of(&.{.{ .w = 1024, .h = 1024 }}, true, 0, 0);
    try std.testing.expectEqual(Size{ .w = 1024, .h = 1024 }, sq.target);
    try std.testing.expectEqual(Size{ .w = 1024, .h = 1024 }, sq.pictures[0].vae);
    const whole = try Plan.of(&.{.{ .w = 1024, .h = 1024 }}, false, 0, 0);
    try std.testing.expectEqual(Size{ .w = 1440, .h = 1440 }, whole.target); // #157's 2 MP: 1440 x 1440
    try std.testing.expectEqual(Size{ .w = 1440, .h = 1440 }, whole.pictures[0].vae);
    try std.testing.expectEqual(Size{ .w = 1024, .h = 1024 }, whole.pictures[0].vision);
    const wide = try Plan.of(&.{ .{ .w = 4000, .h = 3000 }, .{ .w = 640, .h = 480 } }, true, 768, 512);
    try std.testing.expectEqual(Size{ .w = 768, .h = 512 }, wide.target);
    try std.testing.expectEqual(Size{ .w = 1184, .h = 896 }, wide.pictures[0].source); // round(4000 * s / 32) * 32
    try std.testing.expectEqual(Size{ .w = 640, .h = 480 }, wide.pictures[1].source); // used as loaded
    try std.testing.expectEqual(Size{ .w = 1376, .h = 1024 }, wide.pictures[1].vision);
    try std.testing.expectEqual(@as(u32, (1344 / 32) * (1024 / 32) + (1376 / 32) * (1024 / 32) + 4), wide.visionRows());
    var six: [6]Size = @splat(.{ .w = 64, .h = 64 });
    try std.testing.expectError(error.TooManyPictures, Plan.of(&six, true, 0, 0));
    try std.testing.expectError(error.NoPicture, Plan.of(&.{}, true, 0, 0));
}
