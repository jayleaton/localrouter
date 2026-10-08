//! LocalRouter: one `localrouter` binary (daemon, worker, client, checks) on TensorFold's pinned CUDA runtime.

const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (!std.mem.eql(u8, builtin.zig_version_string, "0.17.0")) @compileError("LocalRouter requires Zig 0.17.0");
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const cuda = cudaRuntime(b, target, optimize);
    const nvfp4 = @import("build/cuda.zig").nvfp4(b, target, optimize); // -Dnvcc, -Dsm, `fatbins`, module nvfp4_kernels
    const tokenizer = b.createModule(.{ .root_source_file = b.dependency("tensorfold", .{}).path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const qwen = @import("build/qwen.zig").engine(b, target, optimize, nvfp4.nvcc, nvfp4.sms, cuda, nvfp4.module, tokenizer, nvfp4.fatbins);
    const h3 = @import("build/minimax.zig").engine(b, target, optimize, nvfp4.nvcc, nvfp4.sms, cuda, qwen, nvfp4.fatbins);

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "nvfp4_kernels", .module = nvfp4.module }, .{ .name = "qwen_image", .module = qwen }, .{ .name = "minimax_h3", .module = h3 } },
    });
    const exe = b.addExecutable(.{ .name = "localrouter", .root_module = root });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.addPassthruArgs();
    b.step("run", "Run localrouter").dependOn(&run.step);

    // Unit tests: every module reachable from main.zig.
    const unit = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    // Integration tests: the real `localrouter` binary as daemon and workers, driven over HTTP.
    const it_opts = b.addOptions();
    it_opts.addOptionPath("localrouter_exe", exe.getEmittedBin());
    const it_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    it_mod.addOptions("build_options", it_opts);
    const integration = b.addRunArtifact(b.addTest(.{ .root_module = it_mod }));
    integration.has_side_effects = true;

    // Self-test: the binary starts its own daemon and makes an image and a video through the API.
    const selftest = b.addRunArtifact(exe);
    selftest.addArgs(&.{ "check", "selftest" });
    selftest.has_side_effects = true;

    const test_step = b.step("test", "Unit, integration and self tests");
    test_step.dependOn(&unit.step);
    test_step.dependOn(nvfp4.tests);
    test_step.dependOn(&integration.step);
    test_step.dependOn(&selftest.step);
    const qwen_tests = b.addRunArtifact(b.addTest(.{ .root_module = qwen }));
    test_step.dependOn(&qwen_tests.step);
    const h3_tests = b.addRunArtifact(b.addTest(.{ .root_module = h3 }));
    test_step.dependOn(&h3_tests.step);
    const unit_step = b.step("test-unit", "Unit tests only");
    unit_step.dependOn(&qwen_tests.step);
    unit_step.dependOn(&unit.step);
    unit_step.dependOn(nvfp4.tests);
    b.step("test-integration", "Integration tests only").dependOn(&integration.step);
}

/// TensorFold's CUDA runtime (driver via dlopen, memory, streams, launch, graphs, cuBLASLt) from the pinned
/// dependency, host side only: our engines embed their own fatbins, so TensorFold's are left out.
fn cudaRuntime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const tf = b.dependency("tensorfold", .{});
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const stagger = b.createModule(.{ .root_source_file = tf.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
    const cuda = b.createModule(.{ .root_source_file = tf.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger);
    return cuda;
}
