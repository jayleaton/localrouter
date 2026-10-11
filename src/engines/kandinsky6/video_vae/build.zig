//! The Kandinsky-6 video VAE qualification harness (qual.zig), built on its own so the component can be qualified on
//! the target before it is wired into localrouter. The kernels are not built here: qual.zig loads the conv and ops
//! fatbins that kernels/cuda/nvcc-docker.sh builds, and records their sha256.
//!
//!   zig build -Dtarget=aarch64-linux-gnu          zig-out/bin/k6v-qual for the GB10 node
//!   zig build test                                 this directory's unit tests (also run by the root `zig build test`
//!                                                  only once the component is wired in)

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSafe });
    const tf = b.dependency("tensorfold", .{});
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const stagger = b.createModule(.{ .root_source_file = tf.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
    const cuda = b.createModule(.{ .root_source_file = tf.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger);

    const qual = b.createModule(.{ .root_source_file = b.path("qual.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    b.installArtifact(b.addExecutable(.{ .name = "k6v-qual", .root_module = qual }));

    const test_step = b.step("test", "Video VAE unit tests");
    for ([_][]const u8{ "launch.zig", "tiles.zig", "encode.zig", "decode.zig", "weights.zig", "evidence.zig" }) |f| {
        const m = b.createModule(.{ .root_source_file = b.path(f), .target = b.graph.host, .optimize = .Debug });
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
}
