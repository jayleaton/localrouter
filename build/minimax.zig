//! The MiniMax H3 engine's build: its own kernels as fatbins with the nvcc flags the video twin's torch extensions use
//! (ops.cu: torch's defaults + -O3, like the image kernels; kitchen_launch.cu: torch's defaults then comfy-kitchen's
//! own flags, as `stk_twin/h3/kitchen.py`), the `minimax_kernels` module embedding them, and the `minimax_h3` module.
//! The engine shares the image engine's pack reader, linears, bf16 GEMM, attention and replay (`qwen_image`).

const std = @import("std");
const Hip = @import("hip.zig").Hip;

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

/// `extra`: nvcc's flags after torch's; `hip`: clang's for the HIP code object, built from `hip_file` where the CUDA
/// source does not compile for gfx11 (mma.sync, cp.async, ldmatrix) and from `file` otherwise. kitchen on HIP is the
/// rms_rope port alone (UPSTREAM.md 15): H3-Image runs the bf16 attention, not comfy-kitchen's INT8 one (ROCM.md 3).
const Source = struct { name: []const u8, file: []const u8, extra: []const []const u8, hip: []const []const u8, hip_file: ?[]const u8 = null };
const sources = [_]Source{
    .{ .name = "ops", .file = "kernels/cuda/minimax/ops.cu", .extra = &.{"-O3"}, .hip = &.{"-O3"} },
    .{ .name = "kitchen", .file = "kernels/cuda/minimax/kitchen_launch.cu", .extra = &kitchen_flags, .hip = &.{"-O3"}, .hip_file = "kernels/hip/minimax/kitchen/rms_rope.hip" },
    .{ .name = "te32", .file = "kernels/cuda/minimax/te32.cu", .extra = &.{"-O3"}, .hip = &.{"-O3"} }, // no fast math (h3/te32.py)
    .{ .name = "vae_audio", .file = "kernels/cuda/minimax/vae_audio.cu", .extra = &.{ "-O3", "--fmad=false" }, .hip = &.{ "-O3", "-ffp-contract=off" } }, // h3/vae_audio.py
    .{ .name = "gemm_f16", .file = "kernels/cuda/minimax/gemm_f16.cu", .extra = &.{"-O3"}, .hip = &.{"-O3"}, .hip_file = "kernels/hip/minimax/gemm_f16.hip" }, // h3/vae_video.py
    .{ .name = "te_vision", .file = "kernels/cuda/minimax/te_vision.cu", .extra = &.{"-O3"}, .hip = &.{"-O3"} },
    .{ .name = "vae_enc", .file = "kernels/cuda/minimax/vae_enc.cu", .extra = &.{"-O3"}, .hip = &.{"-O3"} },
    .{ .name = "vae_video", .file = "kernels/cuda/minimax/vae_video.cu", .extra = &.{"-O3"}, .hip = &.{"-O3"} },
};

/// `nvcc` set: CUDA fatbins; `hip` with a ROCm directory: HIP code objects (build.zig never sets both).
pub fn engine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, nvcc: ?[]const u8, sms: []const u8, hip: ?Hip, gpu: *std.Build.Module, qwen: *std.Build.Module, fatbin_step: *std.Build.Step) *std.Build.Module {
    var images: std.ArrayList([]const u8) = .empty;
    const opts = b.addOptions();
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
        kmod.addAnonymousImport(b.fmt("minimax_image_{s}", .{src.name}), .{ .root_source_file = out });
        images.append(b.allocator, src.name) catch @panic("OOM");
        fatbin_step.dependOn(&b.addInstallFile(out, b.fmt("fatbin/minimax_{s}.fatbin", .{src.name})).step);
    };
    if (hip) |h| if (h.root != null) for (sources) |src| {
        const out = @import("hip.zig").codeObject(b, h, b.fmt("minimax_{s}", .{src.name}), src.hip_file orelse src.file, src.hip);
        kmod.addAnonymousImport(b.fmt("minimax_image_{s}", .{src.name}), .{ .root_source_file = out });
        images.append(b.allocator, src.name) catch @panic("OOM");
    };
    opts.addOption([]const []const u8, "images", images.items);
    return b.createModule(.{
        .root_source_file = b.path("src/engines/minimax_h3/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "gpu", .module = gpu },
            .{ .name = "qwen_image", .module = qwen },
            .{ .name = "minimax_kernels", .module = kmod },
        },
    });
}
