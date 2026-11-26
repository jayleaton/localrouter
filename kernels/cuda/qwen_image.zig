//! LocalRouter's own Qwen-Image kernels as fatbins (ops.cu, attention.cu, gemm.cu, te.cu, vae.cu; built by build/cuda.zig with the flags the
//! twin's torch extensions use, or as HIP code objects of the same sources: -Dgpu=hip -Dhip, build/hip.zig) and the frozen
//! RoPE frequencies. Without a compiler, or for a source not built for HIP, an image is empty.

const std = @import("std");
const options = @import("qwen_options");

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
        if (comptime std.mem.eql(u8, have, name)) return &Blob("qwen_image_" ++ name).bytes;
    }
    return &.{};
}

pub const ops: []const u8 = image("ops");
pub const attention: []const u8 = image("attention");
pub const gemm: []const u8 = image("gemm");
pub const te: []const u8 = image("te");
pub const vae: []const u8 = image("vae");
pub const rope_omega: []const u8 = @embedFile("qwen_rope_omega");
pub const te_inv_freq: []const u8 = @embedFile("qwen_te_inv_freq");

pub const vision: []const u8 = image("vision");
pub const vae_enc: []const u8 = image("vae_enc");
