//! A cancellation token for one response: set once from any thread (barge-in, response.cancel, session close). An
//! armed hook runs at once, which is how a blocking read on the chat socket ends.

const std = @import("std");
const Io = std.Io;

pub const Stop = struct {
    flag: std.atomic.Value(bool) = .init(false),
    mutex: Io.Mutex = .init,
    hook: ?Hook = null,

    pub const Hook = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque) void };

    pub fn cancelled(s: *const Stop) bool {
        return s.flag.load(.acquire);
    }

    /// Sets the flag and runs the armed hook, if any.
    pub fn set(s: *Stop, io: Io) void {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.flag.store(true, .release);
        if (s.hook) |h| h.call(h.ctx);
    }

    /// Arms `h` until `disarm`; false (not armed) when already set.
    pub fn arm(s: *Stop, io: Io, h: Hook) bool {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        if (s.cancelled()) return false;
        s.hook = h;
        return true;
    }

    /// After it returns the hook is not running and will not run.
    pub fn disarm(s: *Stop, io: Io) void {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.hook = null;
    }
};

const Count = struct {
    n: u32 = 0,

    fn call(ctx: *anyopaque) void {
        const c: *Count = @ptrCast(@alignCast(ctx));
        c.n += 1;
    }
};

test "a hook runs when set, never after disarm, and cannot be armed once set" {
    const io = std.testing.io;
    var c: Count = .{};
    var s: Stop = .{};
    try std.testing.expect(s.arm(io, .{ .ctx = &c, .call = Count.call }));
    s.disarm(io);
    s.set(io);
    try std.testing.expectEqual(@as(u32, 0), c.n);
    try std.testing.expect(!s.arm(io, .{ .ctx = &c, .call = Count.call }));
    var t: Stop = .{};
    try std.testing.expect(t.arm(io, .{ .ctx = &c, .call = Count.call }));
    t.set(io);
    try std.testing.expect(t.cancelled());
    try std.testing.expectEqual(@as(u32, 1), c.n);
}
