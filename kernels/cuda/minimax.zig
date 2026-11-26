//! LocalRouter's MiniMax H3 kernels as fatbins (ops.cu; kitchen_launch.cu with comfy-kitchen's copied device code; te32.cu; the VAEs' vae_audio.cu, gemm_f16.cu and vae_video.cu; H3-Image's h3_image.cu),
//! built by build/minimax.zig with the flags the video twin's extensions use, or as HIP code objects of the same sources
//! (-Dgpu=hip -Dhip, build/hip.zig). Without a compiler, or for a source not built for HIP, an image is empty.

const std = @import("std");
const options = @import("minimax_options");

/// Whether the build embedded any image (a CUDA fatbin each, or a HIP code object each).
pub const with_kernels: bool = options.images.len > 0;

fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(name).*; // the driver loads images from 8-byte aligned memory
    };
}

/// The embedded image of source `name`, empty where the build has none (no compiler, or not ported to this runtime).
fn image(comptime name: []const u8) []const u8 {
    inline for (options.images) |have| {
        if (comptime std.mem.eql(u8, have, name)) return &Blob("minimax_image_" ++ name).bytes;
    }
    return &.{};
}

pub const ops: []const u8 = image("ops");
pub const kitchen: []const u8 = image("kitchen");
pub const te32: []const u8 = image("te32");
pub const vae_audio: []const u8 = image("vae_audio");
pub const gemm_f16: []const u8 = image("gemm_f16");
pub const vae_video: []const u8 = image("vae_video");

/// The 32B text encoder's RoPE frequencies as ComfyUI's CUDA run computes them (kernels/minimax/te32_inv_freq.json).
pub const te32_inv_freq: []const u8 = @embedFile("minimax_te32_inv_freq");

pub const vae_enc: []const u8 = image("vae_enc");
pub const te_vision: []const u8 = image("te_vision");
/// MiniMax H3-Image's own kernels (h3_image.cu): the PDD head mix and the image VAE decoder's RoPE, unpatchify and tile blend.
pub const h3_image: []const u8 = image("h3_image");
