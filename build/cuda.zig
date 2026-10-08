//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, embedded by a module.
//! `-Dnvcc=<nvcc or kernels/cuda/nvcc-docker.sh>` builds them; without it the module's images are empty slices.

const std = @import("std");

/// One .cu in kernels/cuda/nvfp4 with the flags its extension passes in `extra_cuda_cflags`; `arch_specific`: the
/// `a` targets (sm_121a, sm_120a), which TensorFold's arch_flags picks for NVFP4 on every SM 12.x GPU.
const Kernel = struct { name: []const u8, flags: []const []const u8, arch_specific: bool };

/// tensorfold_nvfp4_ck_v6 (src/tensorfold/cuda/nvfp4/checkpoint.py `_ext`): extra_cuda_cflags=["-O3"].
const nvfp4_kernels = [_]Kernel{
    .{ .name = "act", .flags = &.{"-O3"}, .arch_specific = true },
    .{ .name = "gemm_ck", .flags = &.{"-O3"}, .arch_specific = true },
    .{ .name = "gemm_ws", .flags = &.{"-O3"}, .arch_specific = true },
    .{ .name = "lane4", .flags = &.{"-O3"}, .arch_specific = true },
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

pub const Nvfp4 = struct {
    /// `nvfp4_kernels`: the embedded images and the symbols (kernels/cuda/nvfp4.zig).
    module: *std.Build.Module,
    /// Host tests of that module and of the launch plan; hang them on the unit-test steps.
    tests: *std.Build.Step,
    /// -Dnvcc and -Dsm, and the `fatbins` step, for the other kernel sets (build/qwen.zig).
    nvcc: ?[]const u8,
    sms: []const u8,
    fatbins: *std.Build.Step,
};

/// Adds `-Dnvcc`, `-Dsm` and the `fatbins` step; returns the module `nvfp4_kernels` and its host tests.
pub fn nvfp4(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Nvfp4 {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper such as kernels/cuda/nvcc-docker.sh) that builds the NVFP4 fatbins");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (arch-specific: 121 gives sm_121a)") orelse "121,120";
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (nvcc) |tool| blk: {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;

    const fatbin_step = b.step("fatbins", "Build and install the NVFP4 kernel fatbins alone (needs -Dnvcc)");
    var images: [nvfp4_kernels.len]?std.Build.LazyPath = @splat(null);
    for (nvfp4_kernels, &images) |k, *image| {
        if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
            fatbin_step.dependOn(&b.addInstallFile(image.*.?, b.fmt("fatbin/nvfp4_{s}.fatbin", .{k.name})).step);
        }
    }
    if (nvcc == null) fatbin_step.dependOn(&b.addFail("the fatbins step needs -Dnvcc=<nvcc>").step);

    const opts = b.addOptions();
    opts.addOption(bool, "with_kernels", nvcc != null);
    opts.addOption([]const u8, "set", "nvfp4"); // distinct contents: options files must not be shared by modules
    const m = b.createModule(.{ .root_source_file = b.path("kernels/cuda/nvfp4.zig"), .target = target, .optimize = optimize });
    m.addOptions("nvfp4_options", opts);
    if (nvcc != null) for (nvfp4_kernels, images) |k, image| m.addAnonymousImport(b.fmt("nvfp4_fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });

    // host tests: the symbol tables and the launch plan, always against the host-only module
    const host_opts = b.addOptions();
    host_opts.addOption(bool, "with_kernels", false);
    host_opts.addOption([]const u8, "set", "nvfp4-host");
    const host_m = b.createModule(.{ .root_source_file = b.path("kernels/cuda/nvfp4.zig"), .target = target, .optimize = optimize });
    host_m.addOptions("nvfp4_options", host_opts);
    const plan = b.createModule(.{ .root_source_file = b.path("src/engines/qwen_image/nvfp4_plan.zig"), .target = target, .optimize = optimize });
    plan.addImport("nvfp4_kernels", host_m);
    const tests = b.step("test-nvfp4", "Host tests of the NVFP4 symbols and launch plan");
    tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host_m })).step);
    tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = plan })).step);
    return .{ .module = m, .tests = tests, .nvcc = nvcc, .sms = sms, .fatbins = fatbin_step };
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("nvfp4_{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("kernels/cuda/nvfp4/{s}.cu", .{k.name})));
    return out;
}
