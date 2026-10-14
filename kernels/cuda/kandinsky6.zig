//! LocalRouter's Kandinsky-6 kernels besides the DiT's (kandinsky6/dit/fatbins.zig) as fatbins: the video VAE's conv.cu
//! and ops.cu, the audio decoder's audio.cu, the pipeline's central.cu, the text encoders' text.cu and the PiFlow
//! sampler's policy.cu (passed to `kandinsky6_sampler`'s `Kernels.load`), built by
//! build/kandinsky6.zig with each component's flags. Without nvcc the images are empty and every loader refuses them.

const options = @import("k6_options");

pub const with_kernels: bool = options.with_kernels;

fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(name).*; // the driver loads images from 8-byte aligned memory
    };
}

pub const vae_conv: []const u8 = if (with_kernels) &Blob("k6_fatbin_vae_conv").bytes else &.{};
pub const vae_ops: []const u8 = if (with_kernels) &Blob("k6_fatbin_vae_ops").bytes else &.{};
pub const audio_conv: []const u8 = if (with_kernels) &Blob("k6_fatbin_audio_conv").bytes else &.{};
pub const audio: []const u8 = if (with_kernels) &Blob("k6_fatbin_audio").bytes else &.{};
pub const central: []const u8 = if (with_kernels) &Blob("k6_fatbin_central").bytes else &.{};
pub const text: []const u8 = if (with_kernels) &Blob("k6_fatbin_text").bytes else &.{};
pub const text_attention: []const u8 = if (with_kernels) &Blob("k6_fatbin_text_attention").bytes else &.{};
pub const clip_linear: []const u8 = if (with_kernels) &Blob("k6_fatbin_clip_linear").bytes else &.{};
pub const sampler_policy: []const u8 = if (with_kernels) &Blob("k6_sampler_fatbin_policy").bytes else &.{};
