//! HIP runtime API types and entry points (ROCm 7.2+, hip_runtime_api.h), declared by hand. The entry table's fields
//! keep the CUDA driver names the engines call (`d.api.cuMemcpyHtoD_v2`), so engine code is the same on both runtimes;
//! `symbol` maps each field to its HIP export. Enum values are HIP's own, read from the ROCm 7.13 headers.

pub const Result = c_int;
pub const Device = c_int;
pub const DevicePtr = u64; // hipDeviceptr_t is a void*: the same 64-bit argument

pub const Context = ?*opaque {};
pub const Module = ?*opaque {};
pub const Function = ?*opaque {};
pub const Stream = ?*opaque {};
pub const Event = ?*opaque {};
pub const Graph = ?*opaque {};
pub const GraphNode = ?*opaque {};
pub const GraphExec = ?*opaque {};

pub const success: Result = 0;
pub const error_out_of_memory: Result = 2;
pub const error_not_found: Result = 500;
pub const error_not_ready: Result = 600;

pub const stream_non_blocking: c_uint = 1; // hipStreamNonBlocking
pub const event_disable_timing: c_uint = 2; // hipEventDisableTiming
pub const host_alloc_portable: c_uint = 1; // hipHostMallocPortable
pub const host_alloc_devicemap: c_uint = 2; // hipHostMallocMapped
pub const host_register_devicemap: c_uint = 2; // hipHostRegisterMapped

pub const CaptureMode = enum(c_int) { global = 0, thread_local = 1, relaxed = 2 };
pub const CaptureStatus = enum(c_int) { none = 0, active = 1, invalidated = 2, _ };

/// hipDeviceAttribute_t, under the CUDA names the engines use.
pub const DeviceAttribute = enum(c_int) {
    clock_rate = 5,
    integrated = 16,
    l2_cache_size = 19,
    compute_capability_major = 23, // gfx1151: 11
    max_threads_per_block = 56,
    compute_capability_minor = 61, // gfx1151: 5
    multiprocessor_count = 63,
    max_shared_memory_per_block = 74,
    max_shared_memory_per_block_optin = 75,
    warp_size = 87,
    max_shared_memory_per_multiprocessor = 10002,
};

/// hipFunction_attribute.
pub const FunctionAttribute = enum(c_int) {
    max_threads_per_block = 0,
    shared_size_bytes = 1,
    num_regs = 4,
    max_dynamic_shared_size_bytes = 8,
};

pub const Dim3 = extern struct { x: c_uint = 1, y: c_uint = 1, z: c_uint = 1 };

const R = Result;
const P = DevicePtr;

/// The runtime's entry points the engines reach, by the CUDA driver's names.
pub const Api = struct {
    cuInit: *const fn (c_uint) callconv(.c) R,
    cuDriverGetVersion: *const fn (*c_int) callconv(.c) R,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) R,
    cuDeviceGet: *const fn (*Device, c_int) callconv(.c) R,
    cuDeviceGetName: *const fn ([*]u8, c_int, Device) callconv(.c) R,
    cuDeviceGetAttribute: *const fn (*c_int, DeviceAttribute, Device) callconv(.c) R,
    cuDevicePrimaryCtxRetain: *const fn (*Context, Device) callconv(.c) R,
    cuDevicePrimaryCtxRelease_v2: *const fn (Device) callconv(.c) R,
    cuCtxSetCurrent: *const fn (Context) callconv(.c) R,
    cuCtxSynchronize: *const fn () callconv(.c) R,
    cuGetErrorName: *const fn (R) callconv(.c) ?[*:0]const u8,
    cuGetErrorString: *const fn (R) callconv(.c) ?[*:0]const u8,
    cuMemGetInfo_v2: *const fn (*usize, *usize) callconv(.c) R,
    cuMemAlloc_v2: *const fn (*P, usize) callconv(.c) R,
    cuMemFree_v2: *const fn (P) callconv(.c) R,
    cuMemHostAlloc: *const fn (*?*anyopaque, usize, c_uint) callconv(.c) R,
    cuMemFreeHost: *const fn (?*anyopaque) callconv(.c) R,
    cuMemHostRegister_v2: *const fn (?*anyopaque, usize, c_uint) callconv(.c) R,
    cuMemHostUnregister: *const fn (?*anyopaque) callconv(.c) R,
    cuMemHostGetDevicePointer_v2: *const fn (*P, ?*anyopaque, c_uint) callconv(.c) R,
    cuMemcpyHtoD_v2: *const fn (P, ?*const anyopaque, usize) callconv(.c) R,
    cuMemcpyDtoH_v2: *const fn (?*anyopaque, P, usize) callconv(.c) R,
    cuMemcpyDtoD_v2: *const fn (P, P, usize) callconv(.c) R,
    cuMemcpyHtoDAsync_v2: *const fn (P, ?*const anyopaque, usize, Stream) callconv(.c) R,
    cuMemcpyDtoHAsync_v2: *const fn (?*anyopaque, P, usize, Stream) callconv(.c) R,
    cuMemcpyDtoDAsync_v2: *const fn (P, P, usize, Stream) callconv(.c) R,
    cuMemsetD8_v2: *const fn (P, u8, usize) callconv(.c) R,
    cuMemsetD8Async: *const fn (P, u8, usize, Stream) callconv(.c) R,
    cuMemsetD32_v2: *const fn (P, c_int, usize) callconv(.c) R,
    cuMemsetD32Async: *const fn (P, c_int, usize, Stream) callconv(.c) R,
    cuStreamCreate: *const fn (*Stream, c_uint) callconv(.c) R,
    cuStreamDestroy_v2: *const fn (Stream) callconv(.c) R,
    cuStreamSynchronize: *const fn (Stream) callconv(.c) R,
    cuStreamQuery: *const fn (Stream) callconv(.c) R,
    cuStreamWaitEvent: *const fn (Stream, Event, c_uint) callconv(.c) R,
    cuEventCreate: *const fn (*Event, c_uint) callconv(.c) R,
    cuEventDestroy_v2: *const fn (Event) callconv(.c) R,
    cuEventRecord: *const fn (Event, Stream) callconv(.c) R,
    cuEventSynchronize: *const fn (Event) callconv(.c) R,
    cuEventQuery: *const fn (Event) callconv(.c) R,
    cuEventElapsedTime: *const fn (*f32, Event, Event) callconv(.c) R,
    cuModuleLoadData: *const fn (*Module, ?*const anyopaque) callconv(.c) R,
    cuModuleUnload: *const fn (Module) callconv(.c) R,
    cuModuleGetFunction: *const fn (*Function, Module, [*:0]const u8) callconv(.c) R,
    cuModuleGetGlobal_v2: *const fn (*P, *usize, Module, [*:0]const u8) callconv(.c) R,
    cuFuncGetAttribute: *const fn (*c_int, FunctionAttribute, Function) callconv(.c) R,
    cuLaunchKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, ?[*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) R,
    cuStreamBeginCapture_v2: *const fn (Stream, CaptureMode) callconv(.c) R,
    cuStreamEndCapture: *const fn (Stream, *Graph) callconv(.c) R,
    cuStreamIsCapturing: *const fn (Stream, *CaptureStatus) callconv(.c) R,
    cuGraphInstantiateWithFlags: *const fn (*GraphExec, Graph, c_ulonglong) callconv(.c) R,
    cuGraphDestroy: *const fn (Graph) callconv(.c) R,
    cuGraphExecDestroy: *const fn (GraphExec) callconv(.c) R,
    cuGraphUpload: *const fn (GraphExec, Stream) callconv(.c) R,
    cuGraphLaunch: *const fn (GraphExec, Stream) callconv(.c) R,
};

/// The HIP export behind each `Api` field.
pub fn symbol(comptime field: []const u8) [:0]const u8 {
    const map = .{
        .{ "cuInit", "hipInit" },                                           .{ "cuDriverGetVersion", "hipDriverGetVersion" },
        .{ "cuDeviceGetCount", "hipGetDeviceCount" },                       .{ "cuDeviceGet", "hipDeviceGet" },
        .{ "cuDeviceGetName", "hipDeviceGetName" },                         .{ "cuDeviceGetAttribute", "hipDeviceGetAttribute" },
        .{ "cuDevicePrimaryCtxRetain", "hipDevicePrimaryCtxRetain" },       .{ "cuDevicePrimaryCtxRelease_v2", "hipDevicePrimaryCtxRelease" },
        .{ "cuCtxSetCurrent", "hipCtxSetCurrent" },                         .{ "cuCtxSynchronize", "hipDeviceSynchronize" },
        .{ "cuGetErrorName", "hipGetErrorName" },                           .{ "cuGetErrorString", "hipGetErrorString" },
        .{ "cuMemGetInfo_v2", "hipMemGetInfo" },                            .{ "cuMemAlloc_v2", "hipMalloc" },
        .{ "cuMemFree_v2", "hipFree" },                                     .{ "cuMemHostAlloc", "hipHostMalloc" },
        .{ "cuMemFreeHost", "hipHostFree" },                                .{ "cuMemHostRegister_v2", "hipHostRegister" },
        .{ "cuMemHostUnregister", "hipHostUnregister" },                    .{ "cuMemHostGetDevicePointer_v2", "hipHostGetDevicePointer" },
        .{ "cuMemcpyHtoD_v2", "hipMemcpyHtoD" },                            .{ "cuMemcpyDtoH_v2", "hipMemcpyDtoH" },
        .{ "cuMemcpyDtoD_v2", "hipMemcpyDtoD" },                            .{ "cuMemcpyHtoDAsync_v2", "hipMemcpyHtoDAsync" },
        .{ "cuMemcpyDtoHAsync_v2", "hipMemcpyDtoHAsync" },                  .{ "cuMemcpyDtoDAsync_v2", "hipMemcpyDtoDAsync" },
        .{ "cuMemsetD8_v2", "hipMemsetD8" },                                .{ "cuMemsetD8Async", "hipMemsetD8Async" },
        .{ "cuMemsetD32_v2", "hipMemsetD32" },                              .{ "cuMemsetD32Async", "hipMemsetD32Async" },
        .{ "cuStreamCreate", "hipStreamCreateWithFlags" },                  .{ "cuStreamDestroy_v2", "hipStreamDestroy" },
        .{ "cuStreamSynchronize", "hipStreamSynchronize" },                 .{ "cuStreamQuery", "hipStreamQuery" },
        .{ "cuStreamWaitEvent", "hipStreamWaitEvent" },                     .{ "cuEventCreate", "hipEventCreateWithFlags" },
        .{ "cuEventDestroy_v2", "hipEventDestroy" },                        .{ "cuEventRecord", "hipEventRecord" },
        .{ "cuEventSynchronize", "hipEventSynchronize" },                   .{ "cuEventQuery", "hipEventQuery" },
        .{ "cuEventElapsedTime", "hipEventElapsedTime" },                   .{ "cuModuleLoadData", "hipModuleLoadData" },
        .{ "cuModuleUnload", "hipModuleUnload" },                           .{ "cuModuleGetFunction", "hipModuleGetFunction" },
        .{ "cuModuleGetGlobal_v2", "hipModuleGetGlobal" },                  .{ "cuFuncGetAttribute", "hipFuncGetAttribute" },
        .{ "cuLaunchKernel", "hipModuleLaunchKernel" },                     .{ "cuStreamBeginCapture_v2", "hipStreamBeginCapture" },
        .{ "cuStreamEndCapture", "hipStreamEndCapture" },                   .{ "cuStreamIsCapturing", "hipStreamIsCapturing" },
        .{ "cuGraphInstantiateWithFlags", "hipGraphInstantiateWithFlags" }, .{ "cuGraphDestroy", "hipGraphDestroy" },
        .{ "cuGraphExecDestroy", "hipGraphExecDestroy" },                   .{ "cuGraphUpload", "hipGraphUpload" },
        .{ "cuGraphLaunch", "hipGraphLaunch" },
    };
    inline for (map) |e| if (comptime std.mem.eql(u8, e[0], field)) return e[1];
    @compileError("no HIP symbol for " ++ field);
}

const std = @import("std");

test "every entry point has its HIP symbol, each named once" {
    @setEvalBranchQuota(20000);
    const fields = @typeInfo(Api).@"struct".field_names;
    inline for (fields, 0..) |a, i| inline for (fields[i + 1 ..]) |b| {
        try std.testing.expect(!std.mem.eql(u8, symbol(a), symbol(b)));
    };
}
