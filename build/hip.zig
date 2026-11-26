//! The HIP half of the root build (`-Dgpu=hip`, docs/dev/ROCM.md): LocalRouter's HIP runtime, the module the engines
//! import as `gpu` in place of TensorFold's CUDA runtime, and the kernels' code objects. `-Dhip=<ROCm directory>` (its
//! `lib/llvm/bin/clang++`) builds each CUDA source for `-Dgfx` unchanged, through kernels/hip/compat; without it the
//! images are empty, as without `-Dnvcc`.

const std = @import("std");

/// src/gpu/hip as a module.
pub fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("src/gpu/hip/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// torch.utils.cpp_extension's hipcc defaults (COMMON_HIPCC_FLAGS) in place of its nvcc ones, and C++20 as on CUDA.
const torch_flags = [_][]const u8{ "-D__HIP_NO_HALF_OPERATORS__=1", "-D__HIP_NO_HALF_CONVERSIONS__=1", "-std=c++20" };

pub const Hip = struct {
    /// The ROCm directory (-Dhip); null: no code objects.
    root: ?[]const u8,
    /// The one offload target (-Dgfx).
    gfx: []const u8,
    /// The compiler's version text, an input of every code object, so a new compiler rebuilds them all.
    version: ?std.Build.LazyPath,
    /// Installs every code object under zig-out/hsaco (`zig build hsaco`).
    step: *std.Build.Step,
};

/// Adds `-Dhip`, `-Dgfx` and the `hsaco` step.
pub fn options(b: *std.Build) Hip {
    const root = b.option([]const u8, "hip", "ROCm directory (its lib/llvm/bin/clang++ builds the HIP code objects), e.g. /opt/rocm");
    const gfx = b.option([]const u8, "gfx", "HIP offload target, one gfx name (default gfx1151)") orelse "gfx1151";
    const step = b.step("hsaco", "Build and install the HIP code objects alone (needs -Dhip)");
    if (std.mem.indexOfScalar(u8, gfx, ',') != null) step.dependOn(&b.addFail("-Dgfx takes one target: one code object per build").step);
    const version: ?std.Build.LazyPath = if (root) |r| blk: {
        const run = b.addSystemCommand(&.{ clang(b, r), "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    if (root == null) step.dependOn(&b.addFail("the hsaco step needs -Dhip=<ROCm directory>").step);
    return .{ .root = root, .gfx = gfx, .version = version, .step = step };
}

/// kernels/hip/test/wmma_map_test.cpp, run on the host: the WMMA GEMMs' element map, tile order and row independence
/// (no GPU, no ROCm; ReleaseFast whatever -Doptimize, as it emulates every tile).
pub fn mapTest(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step {
    const mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libcpp = true });
    mod.addCSourceFile(.{ .file = b.path("kernels/hip/test/wmma_map_test.cpp"), .flags = &.{ "-std=c++20", "-Wall", "-Wextra", "-Werror" } });
    const run = b.addRunArtifact(b.addExecutable(.{ .name = "wmma_map_test", .root_module = mod }));
    run.expectExitCode(0);
    return &run.step;
}

fn clang(b: *std.Build, root: []const u8) []const u8 {
    return b.pathJoin(&.{ root, "lib", "llvm", "bin", "clang++" });
}

/// `file` (a CUDA source) as a code object for `h.gfx`, with torch's hipcc defaults, the source's own `extra` flags and
/// the compat headers; installed as hsaco/<name>.hsaco.
pub fn codeObject(b: *std.Build, h: Hip, name: []const u8, file: []const u8, extra: []const []const u8) std.Build.LazyPath {
    const root = h.root.?;
    const run = b.addSystemCommand(&.{ clang(b, root), "-x", "hip", b.fmt("--offload-arch={s}", .{h.gfx}), "--offload-device-only", "--no-gpu-bundle-output", b.fmt("--rocm-path={s}", .{root}) });
    // ROCm's pip and tarball layouts keep the device libraries beside clang; /opt/rocm keeps them where --rocm-path looks
    const bitcode = b.pathJoin(&.{ root, "lib", "llvm", "amdgcn", "bitcode" });
    if (std.Io.Dir.cwd().access(b.graph.io, bitcode, .{})) |_| run.addArg(b.fmt("--rocm-device-lib-path={s}", .{bitcode})) else |_| {}
    run.addFileInput(h.version.?);
    run.addArgs(&torch_flags);
    run.addArgs(extra);
    run.addArg("-I");
    run.addDirectoryArg(b.path("kernels/hip/compat"));
    // the driver drops -MD / -MF for a device-only compile, so the dependency file is asked of the device's cc1
    run.addArgs(&.{ "-include", "cuda_compat.h", "-Xclang", "-MT", "-Xclang", "hsaco", "-Xclang", "-dependency-file", "-Xclang" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.hsaco", .{name}));
    run.addArg("-c");
    run.addFileArg(b.path(file));
    h.step.dependOn(&b.addInstallFile(out, b.fmt("hsaco/{s}.hsaco", .{name})).step);
    return out;
}
