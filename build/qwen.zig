//! The Qwen-Image engine's build: LocalRouter's own kernels (ops.cu, attention.cu) as fatbins with the nvcc flags the
//! twin's torch extensions use (torch's defaults + -O3, arch-specific targets), the `qwen_kernels` module embedding them
//! with the frozen RoPE tables, and the `qwen_image` engine module.

const std = @import("std");

/// torch.utils.cpp_extension's nvcc defaults (torch 2.13), as the twin's load_inline builds compile ops.cu / attention.cu.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
    "-O3",
};

const sources = [_][]const u8{ "ops", "attention", "gemm", "te", "vae" };

/// `nvcc` / `sms` as given to build/cuda.zig (-Dnvcc, -Dsm); `cuda` the runtime, `nvfp4` the TensorFold kernel module.
pub fn engine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, nvcc: ?[]const u8, sms: []const u8, cuda: *std.Build.Module, nvfp4: *std.Build.Module, tokenizer: *std.Build.Module, fatbin_step: *std.Build.Step) *std.Build.Module {
    const opts = b.addOptions();
    opts.addOption(bool, "with_kernels", nvcc != null);
    opts.addOption([]const u8, "set", "qwen_image"); // distinct contents: options files must not be shared by modules
    const kmod = b.createModule(.{ .root_source_file = b.path("kernels/cuda/qwen_image.zig"), .target = target, .optimize = optimize });
    kmod.addOptions("qwen_options", opts);
    kmod.addAnonymousImport("qwen_rope_omega", .{ .root_source_file = b.path("kernels/qwen_image/rope_omega.json") });
    kmod.addAnonymousImport("qwen_te_inv_freq", .{ .root_source_file = b.path("kernels/qwen_image/te_inv_freq.json") });
    if (nvcc) |tool| for (sources) |name| {
        const run = b.addSystemCommand(&.{ tool, "-fatbin" });
        run.addArgs(&torch_flags);
        var it = std.mem.tokenizeScalar(u8, sms, ',');
        while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}a,code=sm_{s}a", .{ sm, sm }));
        run.addArgs(&.{ "-MD", "-MF" });
        _ = run.addDepFileOutputArg2(b.fmt("qwen_{s}.d", .{name}), .{});
        run.addArg("-o");
        const out = run.addOutputFileArg(b.fmt("qwen_{s}.fatbin", .{name}));
        run.addFileArg(b.path(b.fmt("kernels/cuda/qwen_image/{s}.cu", .{name})));
        kmod.addAnonymousImport(b.fmt("qwen_fatbin_{s}", .{name}), .{ .root_source_file = out });
        fatbin_step.dependOn(&b.addInstallFile(out, b.fmt("fatbin/qwen_{s}.fatbin", .{name})).step);
    };
    return b.createModule(.{
        .root_source_file = b.path("src/engines/qwen_image/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "nvfp4_kernels", .module = nvfp4 },
            .{ .name = "qwen_kernels", .module = kmod },
            .{ .name = "tokenizer", .module = tokenizer }, // TensorFold's tokenizer.json implementation
        },
    });
}
