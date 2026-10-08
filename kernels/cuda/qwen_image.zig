//! LocalRouter's own Qwen-Image kernels as fatbins (ops.cu, attention.cu, gemm.cu, te.cu, vae.cu; built by build/cuda.zig with the flags the
//! twin's torch extensions use) and the frozen RoPE frequencies. Without nvcc the images are empty.

const options = @import("qwen_options");

pub const with_kernels: bool = options.with_kernels;

fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(name).*; // the driver loads images from 8-byte aligned memory
    };
}

pub const ops: []const u8 = if (with_kernels) &Blob("qwen_fatbin_ops").bytes else &.{};
pub const attention: []const u8 = if (with_kernels) &Blob("qwen_fatbin_attention").bytes else &.{};
pub const gemm: []const u8 = if (with_kernels) &Blob("qwen_fatbin_gemm").bytes else &.{};
pub const te: []const u8 = if (with_kernels) &Blob("qwen_fatbin_te").bytes else &.{};
pub const vae: []const u8 = if (with_kernels) &Blob("qwen_fatbin_vae").bytes else &.{};
pub const rope_omega: []const u8 = @embedFile("qwen_rope_omega");
pub const te_inv_freq: []const u8 = @embedFile("qwen_te_inv_freq");
