//! The HIP half of the root build (`-Dgpu=hip`, docs/dev/ROCM.md): LocalRouter's HIP runtime, the module the engines
//! import as `gpu` in place of TensorFold's CUDA runtime.

const std = @import("std");

/// src/gpu/hip as a module.
pub fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("src/gpu/hip/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
}
