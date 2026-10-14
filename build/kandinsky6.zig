//! The Kandinsky-6 Pro distilled engine's build: its kernels as fatbins (each component's own nvcc flags, as its
//! README / receipt records them), the modules embedding them, and the `kandinsky6` engine module. The DiT kernels keep
//! the module the DiT component defines (`kandinsky6_dit_kernels`, options `k6_dit_options`); the rest are embedded by
//! `kandinsky6_kernels` (kernels/cuda/kandinsky6.zig). The engine shares the image engine's pack reader and uploader.
//! The PiFlow sampler and the DiT checkpoint loader (component M2) are their own modules, `kandinsky6_sampler` and
//! `kandinsky6_dit_weights`, importing only `cuda`; the engine does not import them yet.

const std = @import("std");

const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

const Source = struct { set: enum { dit, k6, sampler }, name: []const u8, file: []const u8, extra: []const []const u8 };
const sources = [_]Source{
    // kernels/cuda/kandinsky6/dit/README.md: torch's defaults plus -O3, default --fmad
    .{ .set = .dit, .name = "gemm", .file = "kernels/cuda/kandinsky6/dit/gemm.cu", .extra = &.{"-O3"} },
    .{ .set = .dit, .name = "attention", .file = "kernels/cuda/kandinsky6/dit/attention.cu", .extra = &.{"-O3"} },
    .{ .set = .dit, .name = "flash", .file = "kernels/cuda/kandinsky6/dit/flash.cu", .extra = &.{ "-O3", "--extended-lambda", "-DUNFUSE_FMA", "-DFLASHATTENTION_DISABLE_ALIBI", "-DFLASHATTENTION_DISABLE_SOFTCAP" } },
    .{ .set = .dit, .name = "ops", .file = "kernels/cuda/kandinsky6/dit/ops.cu", .extra = &.{"-O3"} },
    // kernels/cuda/kandinsky6/video_vae/RECEIPT.md: torch's flags plus -O3
    .{ .set = .k6, .name = "vae_conv", .file = "kernels/cuda/kandinsky6/video_vae/conv.cu", .extra = &.{"-O3"} },
    .{ .set = .k6, .name = "vae_ops", .file = "kernels/cuda/kandinsky6/video_vae/ops.cu", .extra = &.{"-O3"} },
    // kernels/cuda/kandinsky6/audio/NOTICE.md: -O3 --fmad=false
    .{ .set = .k6, .name = "audio", .file = "kernels/cuda/kandinsky6/audio/audio.cu", .extra = &.{ "-O3", "--fmad=false" } },
    .{ .set = .k6, .name = "audio_conv", .file = "kernels/cuda/kandinsky6/audio/conv_tf32.cu", .extra = &.{ "-O3", "--fmad=false" } },
    // the pipeline's own kernels: torch's flags and -O3 with contraction ON, as torch builds curand's Box-Muller (randn
    // only matches with it); the rollout, video-input and rope kernels use explicit _rn ops, their SASS identical either way
    .{ .set = .k6, .name = "central", .file = "kernels/cuda/kandinsky6/central/central.cu", .extra = &.{"-O3"} },
    // Torch's LayerNorm contracts Welford and affine expressions; other text ops use explicit rounding intrinsics.
    .{ .set = .k6, .name = "text", .file = "kernels/cuda/kandinsky6/text/text.cu", .extra = &.{"-O3"} },
    .{ .set = .k6, .name = "text_attention", .file = "kernels/cuda/kandinsky6/text/attention.cu", .extra = &.{"-O3"} },
    .{ .set = .k6, .name = "clip_linear", .file = "kernels/cuda/kandinsky6/text/clip_linear.cu", .extra = &.{"-O3"} },
    // kernels/cuda/kandinsky6/sampler/README.md: every op its own _rn intrinsic, nothing contracted
    .{ .set = .sampler, .name = "policy", .file = "kernels/cuda/kandinsky6/sampler/policy.cu", .extra = &.{ "-O3", "--fmad=false" } },
};

/// Component M2's host sources (kernels/cuda/kandinsky6/sampler/README.md, "Integration").
pub const sampler_source = "src/engines/kandinsky6/sampler.zig";
pub const dit_weights_source = "src/engines/kandinsky6/dit_weights.zig";

pub const Engine = struct {
    module: *std.Build.Module,
    /// The M2 modules, for their own host test steps.
    sampler: *std.Build.Module,
    dit_weights: *std.Build.Module,
};

pub fn engine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, nvcc: ?[]const u8, sms: []const u8, cuda: *std.Build.Module, qwen: *std.Build.Module, tokenizer: *std.Build.Module, vendor: *std.Build.Module, fatbin_step: *std.Build.Step) Engine {
    const dit_opts = b.addOptions();
    dit_opts.addOption(bool, "with_kernels", nvcc != null);
    dit_opts.addOption([]const u8, "set", "kandinsky6_dit"); // distinct contents (see below)
    const dit_k = b.createModule(.{ .root_source_file = b.path("kernels/cuda/kandinsky6/dit/fatbins.zig"), .target = target, .optimize = optimize });
    dit_k.addOptions("k6_dit_options", dit_opts);

    const opts = b.addOptions();
    opts.addOption(bool, "with_kernels", nvcc != null);
    opts.addOption([]const u8, "set", "kandinsky6"); // distinct contents: options files must not be shared by modules
    const kmod = b.createModule(.{ .root_source_file = b.path("kernels/cuda/kandinsky6.zig"), .target = target, .optimize = optimize });
    kmod.addOptions("k6_options", opts);

    if (nvcc) |tool| for (sources) |src| {
        const run = b.addSystemCommand(&.{ tool, "-fatbin" });
        run.addArgs(&torch_flags);
        run.addArgs(src.extra);
        if (std.mem.eql(u8, src.name, "gemm") or std.mem.eql(u8, src.name, "attention") or std.mem.eql(u8, src.name, "text_attention") or std.mem.eql(u8, src.name, "clip_linear")) {
            run.addArg("-I");
            run.addDirectoryArg(b.path("kernels/cuda/kandinsky6/text/vendor/torch/include"));
            run.addArg("-I");
            run.addDirectoryArg(b.path("kernels/cuda/kandinsky6/text/vendor/cutlass/include"));
        }
        if (std.mem.eql(u8, src.name, "flash")) {
            run.addArg("-I");
            run.addDirectoryArg(b.path("kernels/cuda/kandinsky6/dit/vendor/flash"));
            run.addArg("-I");
            run.addDirectoryArg(b.path("kernels/cuda/kandinsky6/dit/vendor/cutlass/include"));
        }
        if (std.mem.eql(u8, src.name, "audio_conv")) {
            run.addArg("-I");
            run.addDirectoryArg(b.path("kernels/cuda/kandinsky6/audio/vendor/cutlass/include"));
            run.addArg("-gencode=arch=compute_80,code=compute_80");
        } else {
            var it = std.mem.tokenizeScalar(u8, sms, ',');
            while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}a,code=sm_{s}a", .{ sm, sm }));
        }
        run.addArgs(&.{ "-MD", "-MF" });
        const prefix = switch (src.set) {
            .dit => "k6_dit",
            .k6 => "k6",
            .sampler => "k6_sampler",
        };
        _ = run.addDepFileOutputArg2(b.fmt("{s}_{s}.d", .{ prefix, src.name }), .{});
        run.addArg("-o");
        const out = run.addOutputFileArg(b.fmt("{s}_{s}.fatbin", .{ prefix, src.name }));
        run.addFileArg(b.path(src.file));
        const m = if (src.set == .dit) dit_k else kmod; // the sampler's image is embedded with the pipeline's kernels
        m.addAnonymousImport(b.fmt("{s}_fatbin_{s}", .{ prefix, src.name }), .{ .root_source_file = out });
        fatbin_step.dependOn(&b.addInstallFile(out, b.fmt("fatbin/{s}_{s}.fatbin", .{ prefix, src.name })).step);
    };
    const sampler = b.createModule(.{ .root_source_file = b.path(sampler_source), .target = target, .optimize = optimize, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const dit_weights = b.createModule(.{ .root_source_file = b.path(dit_weights_source), .target = target, .optimize = optimize, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const module = b.createModule(.{
        .root_source_file = b.path("src/engines/kandinsky6/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "qwen_image", .module = qwen },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "vendor", .module = vendor },
            .{ .name = "kandinsky6_dit_kernels", .module = dit_k },
            .{ .name = "kandinsky6_kernels", .module = kmod },
        },
    });
    return .{ .module = module, .sampler = sampler, .dit_weights = dit_weights };
}
