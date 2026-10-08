//! LocalRouter's MiniMax H3 kernels as fatbins (ops.cu; kitchen_launch.cu with comfy-kitchen's copied device code; te32.cu; the VAEs' vae_audio.cu, gemm_f16.cu and vae_video.cu),
//! built by build/minimax.zig with the flags the video twin's extensions use. Without nvcc the images are empty.

const options = @import("minimax_options");

pub const with_kernels: bool = options.with_kernels;

fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(name).*; // the driver loads images from 8-byte aligned memory
    };
}

pub const ops: []const u8 = if (with_kernels) &Blob("minimax_fatbin_ops").bytes else &.{};
pub const kitchen: []const u8 = if (with_kernels) &Blob("minimax_fatbin_kitchen").bytes else &.{};
pub const te32: []const u8 = if (with_kernels) &Blob("minimax_fatbin_te32").bytes else &.{};
pub const vae_audio: []const u8 = if (with_kernels) &Blob("minimax_fatbin_vae_audio").bytes else &.{};
pub const gemm_f16: []const u8 = if (with_kernels) &Blob("minimax_fatbin_gemm_f16").bytes else &.{};
pub const vae_video: []const u8 = if (with_kernels) &Blob("minimax_fatbin_vae_video").bytes else &.{};

/// The 32B text encoder's RoPE frequencies as ComfyUI's CUDA run computes them (kernels/minimax/te32_inv_freq.json).
pub const te32_inv_freq: []const u8 = @embedFile("minimax_te32_inv_freq");
