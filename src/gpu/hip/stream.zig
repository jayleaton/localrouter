//! Streams and events: ordering, waits and GPU-side timing.

const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Stream = struct {
    d: *const Driver,
    handle: abi.Stream,

    /// A non-blocking stream never waits on the null stream (what every engine stream should be).
    pub fn init(d: *const Driver, non_blocking: bool) Error!Stream {
        var s: abi.Stream = null;
        try d.check(d.api.cuStreamCreate(&s, if (non_blocking) abi.stream_non_blocking else 0), "hipStreamCreateWithFlags");
        return .{ .d = d, .handle = s };
    }

    /// A stream its caller owns (null: the null stream); never deinit it.
    pub fn borrowed(d: *const Driver, handle: abi.Stream) Stream {
        return .{ .d = d, .handle = handle };
    }

    pub fn deinit(self: *Stream) void {
        _ = self.d.api.cuStreamSynchronize(self.handle);
        _ = self.d.api.cuStreamDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn synchronize(self: Stream) Error!void {
        try self.d.check(self.d.api.cuStreamSynchronize(self.handle), "hipStreamSynchronize");
    }

    pub fn done(self: Stream) Error!bool {
        self.d.check(self.d.api.cuStreamQuery(self.handle), "hipStreamQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    pub fn wait(self: Stream, event: Event) Error!void {
        try self.d.check(self.d.api.cuStreamWaitEvent(self.handle, event.handle, 0), "hipStreamWaitEvent");
    }
};

pub const Event = struct {
    d: *const Driver,
    handle: abi.Event,

    pub fn init(d: *const Driver, timing: bool) Error!Event {
        var e: abi.Event = null;
        try d.check(d.api.cuEventCreate(&e, if (timing) 0 else abi.event_disable_timing), "hipEventCreateWithFlags");
        return .{ .d = d, .handle = e };
    }

    pub fn deinit(self: *Event) void {
        _ = self.d.api.cuEventDestroy_v2(self.handle);
        self.* = undefined;
    }

    pub fn record(self: Event, stream: Stream) Error!void {
        try self.d.check(self.d.api.cuEventRecord(self.handle, stream.handle), "hipEventRecord");
    }

    pub fn synchronize(self: Event) Error!void {
        try self.d.check(self.d.api.cuEventSynchronize(self.handle), "hipEventSynchronize");
    }

    pub fn done(self: Event) Error!bool {
        self.d.check(self.d.api.cuEventQuery(self.handle), "hipEventQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    pub fn elapsedMs(start: Event, end: Event) Error!f32 {
        var ms: f32 = 0;
        try start.d.check(start.d.api.cuEventElapsedTime(&ms, start.handle, end.handle), "hipEventElapsedTime");
        return ms;
    }
};
