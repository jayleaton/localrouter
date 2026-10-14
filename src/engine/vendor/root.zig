//! Pinned vendor libraries, the mechanism a scoped exception in docs/ARCHITECTURE.md would use (none is open: Kandinsky-6
//! audio's route A was withdrawn): a generic sealed-memfd loader with its closure checks, the driver identity check
//! (G3), an explicit-algorithm cuBLASLt adapter, the lock format (no library locked), and the grant-scoped diagnostic
//! contract. Nothing here admits production; an engine's admission contract would.

pub const sys = @import("sys.zig");
pub const elf = @import("elf.zig");
pub const pinned = @import("pinned.zig");
pub const driver = @import("driver.zig");
pub const cublaslt = @import("cublaslt.zig");
pub const locks = @import("locks.zig");
pub const grant = @import("grant.zig");
pub const quiesce = @import("quiesce.zig");

pub const Sys = sys.Sys;
pub const Pin = pinned.Pin;
pub const Resident = pinned.Resident;
pub const Policy = pinned.Policy;
pub const Set = pinned.Set;
pub const Library = pinned.Library;
pub const Quiesced = pinned.Quiesced;

test {
    _ = sys;
    _ = elf;
    _ = pinned;
    _ = driver;
    _ = cublaslt;
    _ = locks;
    _ = grant;
    _ = quiesce;
}
