//! HIP graphs captured from a stream, instantiated once and replayed.

const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const Stream = @import("stream.zig").Stream;

/// Starts capturing `stream`; every launch on it is recorded until `endCapture`, nothing runs.
pub fn beginCapture(stream: Stream, mode: abi.CaptureMode) Error!void {
    try stream.d.check(stream.d.api.cuStreamBeginCapture_v2(stream.handle, mode), "hipStreamBeginCapture");
}

/// Ends the capture; an invalidated capture returns an error and no graph.
pub fn endCapture(stream: Stream) Error!Graph {
    var g: abi.Graph = null;
    try stream.d.check(stream.d.api.cuStreamEndCapture(stream.handle, &g), "hipStreamEndCapture");
    return .{ .d = stream.d, .handle = g };
}

pub fn captureStatus(stream: Stream) Error!abi.CaptureStatus {
    var s: abi.CaptureStatus = .none;
    try stream.d.check(stream.d.api.cuStreamIsCapturing(stream.handle, &s), "hipStreamIsCapturing");
    return s;
}

pub const Graph = struct {
    d: *const Driver,
    handle: abi.Graph,

    pub fn deinit(self: *Graph) void {
        _ = self.d.api.cuGraphDestroy(self.handle);
        self.* = undefined;
    }

    pub fn instantiate(self: Graph) Error!Exec {
        var e: abi.GraphExec = null;
        try self.d.check(self.d.api.cuGraphInstantiateWithFlags(&e, self.handle, 0), "hipGraphInstantiateWithFlags");
        return .{ .d = self.d, .handle = e };
    }
};

pub const Exec = struct {
    d: *const Driver,
    handle: abi.GraphExec,

    pub fn deinit(self: *Exec) void {
        _ = self.d.api.cuGraphExecDestroy(self.handle);
        self.* = undefined;
    }

    pub fn upload(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.cuGraphUpload(self.handle, stream.handle), "hipGraphUpload");
    }

    pub fn launchOn(self: Exec, stream: Stream) Error!void {
        try self.d.check(self.d.api.cuGraphLaunch(self.handle, stream.handle), "hipGraphLaunch");
    }
};
