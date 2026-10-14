//! The vendor module's diagnostics: every refusal and failed release is logged through `err`, at .err in the daemon.
//! Test builds log the same message at .warn, because Zig's test runner counts every .err log as a failure
//! (lib/compiler/test_runner.zig `log`) and the vendor tests drive each refusal and failed release on purpose. Those
//! tests assert the returned error and the resulting state, never the log.

const std = @import("std");
const builtin = @import("builtin");
const scoped = std.log.scoped(.vendor);

pub fn err(comptime format: []const u8, args: anytype) void {
    if (builtin.is_test) scoped.warn(format, args) else scoped.err(format, args);
}
