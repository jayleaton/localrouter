//! LocalRouter: one `localrouter` binary (daemon, worker, client, checks). On Linux the GPU engines run on TensorFold's
//! pinned CUDA runtime (`-Dgpu=cuda`, the default: Qwen-Image, MiniMax H3, Kandinsky-6) or on LocalRouter's HIP runtime
//! (`-Dgpu=hip`: MiniMax H3, docs/dev/ROCM.md); the engines import either as `gpu`. On macOS the GPU engines, their checks
//! and the vendor loader are left out: the binary serves out-of-process servers and `cmd` workers there (docs/dev/MAC-PORT.md).

const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (!std.mem.eql(u8, builtin.zig_version_string, "0.17.0")) @compileError("LocalRouter requires Zig 0.17.0");
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const cuda = cudaRuntime(b, target, optimize);
    const choice = b.option(Gpu, "gpu", "The GPU engines' runtime on Linux: cuda (default) or hip (docs/dev/ROCM.md)") orelse .cuda;
    const nvfp4 = @import("build/cuda.zig").nvfp4(b, target, optimize); // -Dnvcc, -Dsm, `fatbins`, module nvfp4_kernels
    const hip = @import("build/hip.zig").options(b); // -Dhip, -Dgfx, `hsaco`
    if (choice == .hip and nvfp4.nvcc != null) {
        b.default_step.dependOn(&b.addFail("-Dnvcc builds CUDA fatbins: a -Dgpu=hip build embeds none").step);
    }
    if (choice == .cuda and hip.root != null) {
        b.default_step.dependOn(&b.addFail("-Dhip builds HIP code objects: add -Dgpu=hip").step);
    }
    const hip_images: ?@import("build/hip.zig").Hip = if (choice == .hip) hip else null;
    // the runtime the engines import as `gpu`, and the CUDA compiler their kernels get (none for a HIP build)
    const gpu = if (choice == .hip) @import("build/hip.zig").runtime(b, target, optimize) else cuda;
    const nvcc = if (choice == .hip) null else nvfp4.nvcc;
    const tokenizer = b.createModule(.{ .root_source_file = b.dependency("tensorfold", .{}).path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const qwen = @import("build/qwen.zig").engine(b, target, optimize, nvcc, nvfp4.sms, hip_images, gpu, nvfp4.module, tokenizer, nvfp4.fatbins);
    const h3 = @import("build/minimax.zig").engine(b, target, optimize, nvcc, nvfp4.sms, hip_images, gpu, qwen, nvfp4.fatbins);
    // pinned vendor libraries (docs/ARCHITECTURE.md; no exception open): loaded at run time, so host-built and tested
    const vendor = b.createModule(.{ .root_source_file = b.path("src/engine/vendor/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const k6_build = @import("build/kandinsky6.zig").engine(b, target, optimize, nvfp4.nvcc, nvfp4.sms, cuda, qwen, tokenizer, vendor, nvfp4.fatbins);
    const k6 = k6_build.module;
    // GPU engines build on Linux only; `build_config.gpu` tells the source which runtime, and so which table, it gets.
    const backend: Backend = if (target.result.os.tag != .linux) .none else if (choice == .hip) .hip else .cuda;
    const build_config = b.addOptions();
    build_config.addOption(Backend, "gpu", backend);

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = switch (backend) {
            .none => &.{},
            .cuda => &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "gpu", .module = gpu }, .{ .name = "nvfp4_kernels", .module = nvfp4.module }, .{ .name = "qwen_image", .module = qwen }, .{ .name = "minimax_h3", .module = h3 }, .{ .name = "kandinsky6", .module = k6 } },
            .hip => &.{ .{ .name = "gpu", .module = gpu }, .{ .name = "qwen_image", .module = qwen }, .{ .name = "minimax_h3", .module = h3 } },
        },
    });
    root.addOptions("build_config", build_config);
    if (target.result.os.tag.isDarwin()) { // sched/memory.zig: the GPU's working-set ceiling
        root.linkFramework("Metal", .{});
        root.linkSystemLibrary("objc", .{});
    }
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
    it_opts.addOptionPath("speech_worker", b.path("tests/fixtures/speech_worker.py"));
    it_opts.addOptionPath("chat_server", b.path("tests/fixtures/chat_server.py"));
    it_opts.addOptionPath("decide_server", b.path("tests/fixtures/decide_server.py"));
    it_opts.addOptionPath("drama3_worker", b.path("tools/speech/drama3/drama3_worker.py"));
    it_opts.addOptionPath("drama3_pins", b.path("tools/speech/drama3/pins.json"));
    it_opts.addOptionPath("drama3_pilot", b.path("tests/fixtures/drama3_pilot.py"));
    it_opts.addOptionPath("voice_chat_server", b.path("tests/fixtures/voice_chat_server.py"));
    for ([_][]const u8{ "web", "mobile", "helper", "codex" }) |client|
        it_opts.addOptionPath(b.fmt("harness_{s}", .{client}), b.path(b.fmt("tests/fixtures/harness/{s}.jsonl", .{client})));
    const it_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    it_mod.addOptions("build_options", it_opts);
    const it_filters = b.option([]const []const u8, "test-filter", "Run only the integration tests whose names contain this (repeatable)") orelse &.{};
    const it_test = b.addTest(.{ .root_module = it_mod, .filters = it_filters });
    const integration = b.addRunArtifact(it_test);
    integration.has_side_effects = true;
    // The same tests as a program, run outside the build runner so each test's name and output appear as it runs.
    b.step("integration-exe", "Install the integration tests as zig-out/bin/integration-test").dependOn(&b.addInstallArtifact(it_test, .{ .dest_sub_path = "integration-test" }).step);

    // Self-test: the binary starts its own daemon and makes an image and a video through the API.
    const selftest = b.addRunArtifact(exe);
    selftest.addArgs(&.{ "check", "selftest" });
    selftest.has_side_effects = true;

    const test_step = b.step("test", "Unit, integration and self tests");
    test_step.dependOn(&unit.step);
    test_step.dependOn(&integration.step);
    test_step.dependOn(&selftest.step);
    const unit_step = b.step("test-unit", "Unit tests only");
    unit_step.dependOn(&unit.step);
    if (backend != .none) {
        test_step.dependOn(nvfp4.tests);
        unit_step.dependOn(nvfp4.tests);
        // the Kandinsky-6 sampler and DiT checkpoint loader (M2): host tests, not yet reachable from main.zig
        const cuda_only = [_]*std.Build.Module{ qwen, k6, vendor, k6_build.sampler, k6_build.dit_weights };
        const modules: []const *std.Build.Module = if (backend == .hip) &.{ h3, gpu } else &(cuda_only ++ [_]*std.Build.Module{h3});
        for (modules) |m| {
            const t = b.addRunArtifact(b.addTest(.{ .root_module = m }));
            test_step.dependOn(&t.step);
            if (m != h3) unit_step.dependOn(&t.step);
        }
    }
    b.step("test-integration", "Integration tests only").dependOn(&integration.step);
}

/// `-Dgpu`: the runtime the Linux GPU engines are built on.
const Gpu = enum { cuda, hip };
/// `build_config.gpu`: `none` where no GPU engine is built (macOS).
const Backend = enum { none, cuda, hip };

/// TensorFold's CUDA runtime (driver via dlopen, memory, streams, launch, graphs, cuBLASLt) from the pinned
/// dependency, host side only: our engines embed their own fatbins, so TensorFold's are left out.
fn cudaRuntime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const tf = b.dependency("tensorfold", .{});
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const stagger = b.createModule(.{ .root_source_file = tf.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
    // `sys` (dlopen, env, clocks) and `gpu_memory` (fit checks) as TensorFold's own zig/build/cuda_runtime.zig wires them.
    const sys = b.createModule(.{ .root_source_file = tf.path("zig/src/platform/sys.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const gpu_memory = b.createModule(.{ .root_source_file = tf.path("zig/src/gpu_memory.zig"), .target = target, .optimize = optimize });
    const cuda = b.createModule(.{ .root_source_file = tf.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger);
    cuda.addImport("sys", sys);
    cuda.addImport("gpu_memory", gpu_memory);
    return cuda;
}
