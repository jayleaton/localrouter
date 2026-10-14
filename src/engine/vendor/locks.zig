//! The vendor library lock format. No library is locked: the one candidate set (Kandinsky-6 audio's route A, cuDNN and
//! cuBLASLt) was withdrawn once the native kernels reproduced every audio operator
//! (kernels/cuda/kandinsky6/audio/QUALIFICATION.md). For an entry, when one is added:
//! - `path` is where packaging installs the file; null until the packaging owner settles it, and a null path refuses
//!   `pins`;
//! - `soname` is the recorded name; the loader checks it against each file's DT_SONAME;
//! - its package version, provenance and redistribution terms must be recorded before production.
//! The CUDA driver is not in this lock: the daemon's existing libcuda is validated separately (driver.zig, G3).

const std = @import("std");
const pinned = @import("pinned.zig");
const driver = @import("driver.zig");

pub const Origin = enum { wheel, system };

pub const Entry = struct {
    soname: []const u8,
    sha256: [32]u8,
    size: u64,
    /// The install path; null until packaging is coordinated.
    path: ?[:0]const u8 = null,
    /// Where the recorded run mapped it: a Python wheel or the system library directory.
    origin: Origin,
};

/// G3: the driver record. D3's run reported CUDA API 13030; it is not entered here until it is recorded against the
/// kernel module version, the host kernel and the mapped libcuda's identity together, by a future grant. Nothing is
/// assumed from a compat file name.
pub const driver_record: driver.Expect = .{};

/// The residents' identities (libc, ld.so, libcuda, ...), each with an fs-verity digest: none recorded yet, so the
/// set refuses (pinned.zig, gap G3 / G4).
pub const residents = [_]pinned.Resident{};

pub const Error = error{PinPathPending};

/// The pins for `pinned.Set.open`; refuses while any install path is pending.
pub fn pins(entries: []const Entry, out: []pinned.Pin) Error![]const pinned.Pin {
    std.debug.assert(out.len >= entries.len);
    for (entries, out[0..entries.len]) |e, *p| {
        p.* = .{ .name = e.soname, .path = e.path orelse return error.PinPathPending, .sha256 = e.sha256, .size = e.size };
    }
    return out[0..entries.len];
}

fn hex(comptime s: []const u8) [32]u8 {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch unreachable;
    return out;
}

test "an entry without an install path refuses, and no driver is recorded" {
    const pending = [_]Entry{.{ .soname = "libexample.so.1", .sha256 = hex("0000000000000000000000000000000000000000000000000000000000000000"), .size = 1, .origin = .system }};
    var buf: [pending.len]pinned.Pin = undefined;
    try std.testing.expectError(error.PinPathPending, pins(&pending, &buf));
    try std.testing.expect(driver_record.kernel_module == null and driver_record.libcuda == null);
}
