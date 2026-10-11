//! The Kandinsky-6 DiT kernels (gemm.cu, attention.cu, ops.cu) as embedded fatbins, for the build's `kandinsky6_dit_kernels`
//! module: options `k6_dit_options.with_kernels` and the anonymous imports `k6_dit_fatbin_{gemm,attention,ops}` (see
//! README.md, integration). Without nvcc the images are empty and `launch.Kernels.load` refuses them.

const options = @import("k6_dit_options");

pub const with_kernels: bool = options.with_kernels;

fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(name).*; // the driver loads images from 8-byte aligned memory
    };
}

pub const gemm: []const u8 = if (with_kernels) &Blob("k6_dit_fatbin_gemm").bytes else &.{};
pub const attention: []const u8 = if (with_kernels) &Blob("k6_dit_fatbin_attention").bytes else &.{};
pub const ops: []const u8 = if (with_kernels) &Blob("k6_dit_fatbin_ops").bytes else &.{};
