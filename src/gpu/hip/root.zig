//! LocalRouter's HIP runtime (`-Dgpu=hip`, docs/dev/ROCM.md): libamdhip64 through dlopen, no ROCm at build time. It
//! declares the subset of TensorFold's CUDA runtime the H3 engines use, under the same names, so the engines import
//! either one as `gpu`. `zig build -Dgpu=hip` compiling the engines is the check that every name they reach exists.

pub const abi = @import("abi.zig");
pub const Driver = @import("driver.zig").Driver;
pub const Error = @import("driver.zig").Error;
pub const Context = @import("context.zig").Context;
pub const DeviceBuffer = @import("memory.zig").DeviceBuffer;
pub const HostBuffer = @import("memory.zig").HostBuffer;
pub const Stream = @import("stream.zig").Stream;
pub const Event = @import("stream.zig").Event;
pub const Module = @import("module.zig").Module;
pub const Function = @import("module.zig").Function;
pub const launch = @import("launch.zig");
pub const Args = launch.Args;
pub const Config = launch.Config;
pub const Dim3 = launch.Dim3;
pub const graph = @import("graph.zig");

/// Marks this runtime as HIP: the engines' per-vendor kernel tables test `@hasDecl(gpu, "hip_runtime")` (TensorFold's
/// CUDA runtime declares no such name).
pub const hip_runtime = true;

/// TensorFold keeps a copy-source scope per capture (pinned staging it owns); HIP's capture has none to close.
pub const copy_source = struct {
    pub fn captureEnd(_: Stream) void {}
};

test {
    _ = abi;
    _ = @import("driver.zig");
    _ = @import("module.zig");
    _ = @import("memory.zig");
    _ = launch;
}
