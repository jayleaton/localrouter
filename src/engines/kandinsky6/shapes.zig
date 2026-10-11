//! Kandinsky-6 Pro distilled: the one generation geometry it ships with (kandinskylab/kandinsky-6 eac06e39,
//! `configs/checkpoints/pro-distill.yaml`, HF revision 1018d5828716e8bea3de99609f45c924de9c5564) and every count the
//! pipeline derives from it. Pure: no I/O, no allocation, so the tool's `needs` and `check` can use it.
//!
//! Default T2V and default I2V (`pretrain` conditioning at latent frame 0) both decode 121 frames at 24 fps from 31
//! latent frames ((31 - 1) * 4 + 1). Audio is always generated with the video (PiFlow requires both): 218 latent rows
//! (ceil(121 / 24 * 44100 / 1024)), decoded to mono 44.1 kHz and converted to int16 the way upstream does
//! (`postprocess_audio.py`: clip to [-1, 1], times 32767 in f32, truncated toward zero).

const std = @import("std");

pub const width: u32 = 864;
pub const height: u32 = 480;
pub const fps: u32 = 24;
pub const latent_frames: u32 = 31;
pub const temporal_ratio: u32 = 4;
pub const spatial_ratio: u32 = 8;
pub const sample_frames: u32 = (latent_frames - 1) * temporal_ratio + 1;
pub const latent_h: u32 = height / spatial_ratio;
pub const latent_w: u32 = width / spatial_ratio;
pub const latent_channels: u32 = 16;
/// DiT patch (t, h, w) = (1, 2, 2).
pub const patch_h: u32 = 2;
pub const patch_w: u32 = 2;
pub const patches_h: u32 = latent_h / patch_h;
pub const patches_w: u32 = latent_w / patch_w;
pub const visual_tokens: u64 = @as(u64, latent_frames) * patches_h * patches_w;
/// Latent elements of the video state [T, H, W, C].
pub const video_elems: u64 = @as(u64, latent_frames) * latent_h * latent_w * latent_channels;
/// Visual-conditioning input width of the DiT: latent 16 + cond 16 + mask 1 (`denoise_loop._build_video_input`).
pub const video_in_channels: u32 = 2 * latent_channels + 1;

pub const sample_rate: u32 = 44100;
pub const audio_downsample: u32 = 1024;
pub const audio_channels: u32 = 40;
pub const audio_rows: u32 = audioRows(latent_frames);
pub const audio_elems: u64 = @as(u64, audio_rows) * audio_channels;
/// The vocoder's output length: two mel frames a latent row, 512 samples a mel frame.
pub const audio_samples: u64 = @as(u64, audio_rows) * audio_downsample;

/// Qwen2.5-VL: the template's system rows cropped from the front, then a fixed 1024-row text (text_token_padding).
pub const text_rows: u32 = 1024;
pub const text_crop: u32 = 129;
pub const text_dim: u32 = 3584;
pub const clip_tokens: u32 = 77;
pub const clip_dim: u32 = 768;

/// PiFlow (`piflow_sampler.py`, `pro-distill.yaml` scheduler block).
pub const steps: u32 = 10;
pub const n_grid: u32 = 10;
pub const shift: f32 = 5.0;
pub const eps: f32 = 1e-6;
pub const final_step_size_scale: f32 = 0.5;
pub const policy_substeps: u32 = 128;

/// Upstream `audio_latent_duration`: ceil(sample_frames / fps * audio_fps / downsample) in f64.
pub fn audioRows(frames: u32) u32 {
    const sf: f64 = @floatFromInt((frames - 1) * temporal_ratio + 1);
    return @intFromFloat(@ceil(sf / @as(f64, fps) * @as(f64, sample_rate) / @as(f64, audio_downsample)));
}

/// The request geometry the engine generates. Any other size, length or frame rate is refused before load.
pub const Request = struct { width: u32, height: u32, seconds: u32, fps: u32 };

pub fn supported(r: Request, why: *std.Io.Writer) bool {
    if (r.width != width or r.height != height) {
        why.print("Kandinsky-6 Pro distilled generates {d}x{d} only, not {d}x{d}", .{ width, height, r.width, r.height }) catch {};
        return false;
    }
    if (r.fps != fps) {
        why.print("Kandinsky-6 Pro distilled generates {d} fps only, not {d}", .{ fps, r.fps }) catch {};
        return false;
    }
    if (r.seconds != 5) {
        why.print("Kandinsky-6 Pro distilled generates 5 seconds ({d} frames) only, not {d}", .{ sample_frames, r.seconds }) catch {};
        return false;
    }
    return true;
}

/// upstream int16 conversion of one decoded sample: `np.clip(w, -1, 1) * 32767` in f32, then `astype(int16)`
/// (truncation toward zero). NaN maps as numpy's clip leaves it and the cast is undefined, so it is refused upstream of
/// here by the decoder's finiteness check.
pub fn pcm16(x: f32) i16 {
    const c = @min(@max(x, -1.0), 1.0);
    const y: f32 = c * 32767.0;
    return @intFromFloat(@trunc(y));
}

test "the native geometry" {
    try std.testing.expectEqual(@as(u32, 121), sample_frames);
    try std.testing.expectEqual(@as(u32, 60), latent_h);
    try std.testing.expectEqual(@as(u32, 108), latent_w);
    try std.testing.expectEqual(@as(u64, 50220), visual_tokens);
    try std.testing.expectEqual(@as(u32, 218), audio_rows);
    try std.testing.expectEqual(@as(u64, 223232), audio_samples);
    try std.testing.expectEqual(@as(u32, 33), video_in_channels);
}

test "pcm16 clips and truncates toward zero like numpy" {
    try std.testing.expectEqual(@as(i16, 32767), pcm16(1.5));
    try std.testing.expectEqual(@as(i16, -32767), pcm16(-2.0));
    try std.testing.expectEqual(@as(i16, 0), pcm16(0.00003));
    try std.testing.expectEqual(@as(i16, 16383), pcm16(0.5)); // 16383.5 truncates
    try std.testing.expectEqual(@as(i16, -16383), pcm16(-0.5));
}

test "only the native request is supported" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try std.testing.expect(supported(.{ .width = 864, .height = 480, .seconds = 5, .fps = 24 }, &w));
    try std.testing.expect(!supported(.{ .width = 1344, .height = 768, .seconds = 5, .fps = 24 }, &w));
}
