//! The MiniMax H3 engine's build: its own kernels as fatbins with the nvcc flags the video twin's torch extensions use
//! (ops.cu: torch's defaults + -O3, like the image kernels; kitchen_launch.cu: torch's defaults then comfy-kitchen's
//! own flags, as `stk_twin/h3/kitchen.py`), the `minimax_kernels` module embedding them, and the `minimax_h3` module.
//! The engine shares the image engine's pack reader, linears, bf16 GEMM, attention and replay (`qwen_image`).

const std = @import("std");

const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// comfy-kitchen 0.2.35's flags (backends/cuda/CMakeLists.txt), after torch's: --use_fast_math decides the
/// quantizers' divisions, rsqrt and exp.
const kitchen_flags = [_][]const u8{
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
};

const Source = struct { name: []const u8, file: []const u8, extra: []const []const u8 };
const sources = [_]Source{
    .{ .name = "ops", .file = "kernels/cuda/minimax/ops.cu", .extra = &.{"-O3"} },
    .{ .name = "kitchen", .file = "kernels/cuda/minimax/kitchen_launch.cu", .extra = &kitchen_flags },
    .{ .name = "te32", .file = "kernels/cuda/minimax/te32.cu", .extra = &.{"-O3"} }, // no fast math (h3/te32.py)
    .{ .name = "vae_audio", .file = "kernels/cuda/minimax/vae_audio.cu", .extra = &.{ "-O3", "--fmad=false" } }, // h3/vae_audio.py
    .{ .name = "gemm_f16", .file = "kernels/cuda/minimax/gemm_f16.cu", .extra = &.{"-O3"} }, // h3/vae_video.py
    .{ .name = "vae_video", .file = "kernels/cuda/minimax/vae_video.cu", .extra = &.{"-O3"} },
};

pub fn engine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, nvcc: ?[]const u8, sms: []const u8, cuda: *std.Build.Module, qwen: *std.Build.Module, fatbin_step: *std.Build.Step) *std.Build.Module {
    const opts = b.addOptions();
    opts.addOption(bool, "with_kernels", nvcc != null);
    opts.addOption([]const u8, "set", "minimax_h3"); // distinct contents: options files must not be shared by modules
    const kmod = b.createModule(.{ .root_source_file = b.path("kernels/cuda/minimax.zig"), .target = target, .optimize = optimize });
    kmod.addOptions("minimax_options", opts);
    kmod.addAnonymousImport("minimax_te32_inv_freq", .{ .root_source_file = b.path("kernels/minimax/te32_inv_freq.json") });
    if (nvcc) |tool| for (sources) |src| {
        const run = b.addSystemCommand(&.{ tool, "-fatbin" });
        run.addArgs(&torch_flags);
        run.addArgs(src.extra);
        var it = std.mem.tokenizeScalar(u8, sms, ',');
        while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}a,code=sm_{s}a", .{ sm, sm }));
        run.addArgs(&.{ "-MD", "-MF" });
        _ = run.addDepFileOutputArg2(b.fmt("minimax_{s}.d", .{src.name}), .{});
        run.addArg("-o");
        const out = run.addOutputFileArg(b.fmt("minimax_{s}.fatbin", .{src.name}));
        run.addFileArg(b.path(src.file));
        kmod.addAnonymousImport(b.fmt("minimax_fatbin_{s}", .{src.name}), .{ .root_source_file = out });
        fatbin_step.dependOn(&b.addInstallFile(out, b.fmt("fatbin/minimax_{s}.fatbin", .{src.name})).step);
    };
    return b.createModule(.{
        .root_source_file = b.path("src/engines/minimax_h3/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "qwen_image", .module = qwen },
            .{ .name = "minimax_kernels", .module = kmod },
        },
    });
}
