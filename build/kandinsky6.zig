//! The Kandinsky-6 Pro distilled engine's build: its kernels as fatbins (each component's own nvcc flags, as its
//! README / receipt records them), the modules embedding them, and the `kandinsky6` engine module. The DiT kernels keep
//! the module the DiT component defines (`kandinsky6_dit_kernels`, options `k6_dit_options`); the rest are embedded by
//! `kandinsky6_kernels` (kernels/cuda/kandinsky6.zig). The engine shares the image engine's pack reader and uploader.

const std = @import("std");

const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

const Source = struct { set: enum { dit, k6 }, name: []const u8, file: []const u8, extra: []const []const u8 };
const sources = [_]Source{
    // kernels/cuda/kandinsky6/dit/README.md: torch's defaults plus -O3, default --fmad
    .{ .set = .dit, .name = "gemm", .file = "kernels/cuda/kandinsky6/dit/gemm.cu", .extra = &.{"-O3"} },
    .{ .set = .dit, .name = "attention", .file = "kernels/cuda/kandinsky6/dit/attention.cu", .extra = &.{"-O3"} },
    .{ .set = .dit, .name = "ops", .file = "kernels/cuda/kandinsky6/dit/ops.cu", .extra = &.{"-O3"} },
    // kernels/cuda/kandinsky6/video_vae/RECEIPT.md: torch's flags plus -O3
    .{ .set = .k6, .name = "vae_conv", .file = "kernels/cuda/kandinsky6/video_vae/conv.cu", .extra = &.{"-O3"} },
    .{ .set = .k6, .name = "vae_ops", .file = "kernels/cuda/kandinsky6/video_vae/ops.cu", .extra = &.{"-O3"} },
    // kernels/cuda/kandinsky6/audio/NOTICE.md: -O3 --fmad=false
    .{ .set = .k6, .name = "audio", .file = "kernels/cuda/kandinsky6/audio/audio.cu", .extra = &.{ "-O3", "--fmad=false" } },
    // the pipeline's own kernels: torch's flags and -O3 with contraction ON, as torch builds curand's Box-Muller (randn
    // only matches with it); the rollout, video-input and rope kernels use explicit _rn ops, their SASS identical either way
    .{ .set = .k6, .name = "central", .file = "kernels/cuda/kandinsky6/central/central.cu", .extra = &.{"-O3"} },
    // the text encoders': every op one rounding
    .{ .set = .k6, .name = "text", .file = "kernels/cuda/kandinsky6/text/text.cu", .extra = &.{ "-O3", "--fmad=false" } },
};

pub fn engine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, nvcc: ?[]const u8, sms: []const u8, cuda: *std.Build.Module, qwen: *std.Build.Module, tokenizer: *std.Build.Module, fatbin_step: *std.Build.Step) *std.Build.Module {
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
        var it = std.mem.tokenizeScalar(u8, sms, ',');
        while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}a,code=sm_{s}a", .{ sm, sm }));
        run.addArgs(&.{ "-MD", "-MF" });
        const prefix = if (src.set == .dit) "k6_dit" else "k6";
        _ = run.addDepFileOutputArg2(b.fmt("{s}_{s}.d", .{ prefix, src.name }), .{});
        run.addArg("-o");
        const out = run.addOutputFileArg(b.fmt("{s}_{s}.fatbin", .{ prefix, src.name }));
        run.addFileArg(b.path(src.file));
        const m = if (src.set == .dit) dit_k else kmod;
        m.addAnonymousImport(b.fmt("{s}_fatbin_{s}", .{ prefix, src.name }), .{ .root_source_file = out });
        fatbin_step.dependOn(&b.addInstallFile(out, b.fmt("fatbin/{s}_{s}.fatbin", .{ prefix, src.name })).step);
    };
    return b.createModule(.{
        .root_source_file = b.path("src/engines/kandinsky6/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "qwen_image", .module = qwen },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "kandinsky6_dit_kernels", .module = dit_k },
            .{ .name = "kandinsky6_kernels", .module = kmod },
        },
    });
}
