//! Pinned vendor libraries: each library's bytes are copied from its lock path into a sealed memfd (F_SEAL_WRITE |
//! GROW | SHRINK | SEAL), the sealed bytes hashed against the lock, and the memfd loaded with RTLD_NOW | RTLD_GLOBAL in
//! DT_NEEDED order. The mapped bytes are the hashed bytes; the seals, not memory residency, are what keeps them so.
//!
//! What this does NOT prove, and what callers must not claim:
//! - RTLD_GLOBAL plus each pin's DT_SONAME is the intended way a library's later dlopen(soname) finds the loaded copy;
//!   it does not prevent a disk search (gap G6). `verifyClosure` detects one after the fact, as an extra mapping.
//! - The static DT_NEEDED closure does not cover a later dlopen of anything else; only `verifyClosure` and the per-call
//!   `checkUnchanged` see that, and they detect, they do not prevent: unpinned code may already have run.
//! - Objects mapped before the set opens (libc, ld.so, libcuda, ...) are accepted only by exact identity: the lock's
//!   path, size, sha256 and fs-verity digest, and the mapping's device and inode. A resident without an fs-verity
//!   digest has no immutable provenance and refuses; rehashing a file by name is not proof.
//! - Unloading while GPU work queued through these libraries may still run is never safe: `close` unloads only with a
//!   valid `quiesce.Quiesced` (an actual bounded stream query that synchronized) and no dependent plan or handle held;
//!   a failed unload is sticky, keeps the rest mapped and owned, and is reported with its raw result. Without
//!   quiescence, `abandon` keeps everything mapped and owned for the life of the process.

const std = @import("std");
const elf = @import("elf.zig");
const sys_mod = @import("sys.zig");
const quiesce = @import("quiesce.zig");
const Sys = sys_mod.Sys;
const Fd = sys_mod.Fd;
const Counts = sys_mod.Counts;
const Sha256 = std.crypto.hash.sha2.Sha256;
const log = @import("report.zig"); // .err at run time, .warn in tests

pub const Pin = struct {
    /// The DT_SONAME the library must carry; consumers and other pins' DT_NEEDED name it by this.
    name: []const u8,
    /// Absolute, from the lock; never searched, never taken from config or the environment.
    path: [:0]const u8,
    sha256: [32]u8,
    size: u64,
};

/// An object the process maps before the set opens, accepted by exact identity only.
pub const Resident = struct {
    /// dlpi_name, exactly as the loader recorded it.
    path: [:0]const u8,
    soname: []const u8,
    size: u64,
    sha256: [32]u8,
    /// The fs-verity SHA-256 file digest; null (no immutable provenance) refuses the set.
    verity: ?[32]u8,
};

pub const Policy = struct {
    preexisting: []const Resident,
    /// Present in the environment refuses the open and the closure check.
    deny_env: []const [:0]const u8,
};

/// The default deny list: loader search and preload, and the vendor overrides that change numerics or add logging.
pub const deny_env_default = [_][:0]const u8{
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_LOG_LEVEL",
    "CUBLASLT_LOG_MASK",
    "CUBLASLT_LOG_FILE",
    "CUDNN_LOGINFO_DBG",
    "CUDNN_LOGWARN_DBG",
    "CUDNN_LOGERR_DBG",
    "CUDNN_LOGLEVEL_DBG",
    "CUDNN_LOGDEST_DBG",
    "CUDA_MODULE_LOADING",
};

/// The bound the vendor's own internal host allocations are budgeted at (handle creation, module loading, plan
/// finalize, NVRTC). Unmeasured, the GB10 policy charges 8 GiB, and production stays refused (gap G5).
pub const VendorBound = struct { bytes: u64, measured: bool };
pub const unmeasured_vendor_bound: VendorBound = .{ .bytes = 8 << 30, .measured = false };

/// Host bytes the set will take, each counted once:
/// - `shared`: the memfds' pages; the loader's and the parser's read-only mappings share them;
/// - `private`: the writable segments' copy-on-write pages (data, relro, bss), page-rounded;
/// - `transient`: the one copy / hash buffer;
/// - `vendor_internal`: the vendor's own allocations, as bounded.
pub const Needs = struct {
    shared: u64 = 0,
    private: u64 = 0,
    transient: u64 = 0,
    vendor_internal: u64 = 0,

    pub fn total(self: Needs) u64 {
        return self.shared + self.private + self.transient + self.vendor_internal;
    }
};

pub const chunk_bytes = 1 << 20;
pub const seals_required: u32 = std.os.linux.F.SEAL_WRITE | std.os.linux.F.SEAL_GROW | std.os.linux.F.SEAL_SHRINK | std.os.linux.F.SEAL_SEAL;
const max_phdrs = 64;
pub const max_pins = 16;
pub const max_residents = 64;

pub const Error = error{
    PinPathNotAbsolute,
    PinDuplicate,
    PinTooMany,
    PinOpenFailed,
    PinNotRegular,
    PinSizeMismatch,
    PinHashMismatch,
    PinSealFailed,
    PinNotImmutable,
    PinElfInvalid,
    PinSonameMismatch,
    PinNeededUnpinned,
    PinOrderCycle,
    PinBudgetExceeded,
    PinLoadFailed,
    PinMissingSymbol,
    PinSymbolElsewhere,
    PinEnvRefused,
    PinUnverified,
    PinClosureMismatch,
    PinMappedSetChanged,
    OutOfMemory,
};

/// Proof that nothing queued on the GPU through these libraries can still run: built only from an observed bounded
/// stream query (`quiesce.poll` / `quiesce.proof`), re-validated by every consumer.
pub const Quiesced = quiesce.Quiesced;

pub const CloseError = error{ NotQuiesced, DependentsOutstanding, ReleaseFailed, Retained, AlreadyReleased };

/// What `close` did. A failure leaves the failed library and everything loaded before it mapped, their memfds open and
/// the set's bookkeeping kept: custody stays with the set, and every later `close` refuses.
pub const Release = struct {
    unloaded: usize,
    failed: ?struct { name: []const u8, rc: c_int } = null,

    pub fn ok(self: Release) bool {
        return self.failed == null;
    }
};

pub const Custody = enum { open, released, release_failed, retained };

/// An init that refused and whose own cleanup then failed: the init's error, kept apart from the cleanup's first raw
/// status, and the objects it could not release, owned here. Nothing in `held` is freed, retried or dropped; the
/// caller keeps the record for the life of the process.
pub fn InitRetained(comptime Primary: type, comptime Held: type) type {
    return struct {
        primary: Primary,
        cleanup: struct { what: []const u8, rc: c_int },
        held: Held,
    };
}

/// `Set.open`'s: `held` is the set with custody `retained`, `libs[0..loaded]` still mapped with their memfds open.
pub const OpenRetained = InitRetained(Error, Set);

pub const Library = struct {
    pin: *const Pin,
    sys: Sys,
    fd: Fd,
    handle: ?*anyopaque = null,
    base: usize = 0,
    dev_major: u32,
    dev_minor: u32,
    ino: u64,

    /// Resolves every field of `Api` by its exact name; a missing symbol, or one defined in any object other than this
    /// library (found through its dependencies or the global scope), refuses.
    pub fn resolve(self: *const Library, comptime Api: type) Error!Api {
        const h = self.handle orelse return error.PinUnverified;
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            const p = self.sys.dlsym(self.sys.ctx, h, name) orelse {
                log.err("{s} has no {s}", .{ self.pin.name, name });
                return error.PinMissingSymbol;
            };
            if (self.sys.addrBase(self.sys.ctx, @intFromPtr(p)) != self.base) {
                log.err("{s}: {s} resolves outside the pinned object", .{ self.pin.name, name });
                return error.PinSymbolElsewhere;
            }
            @field(api, name) = @as(T, @ptrCast(@alignCast(p)));
        }
        return api;
    }
};

pub const Identity = struct { dev_major: u32, dev_minor: u32, ino: u64 };

pub const Set = struct {
    gpa: std.mem.Allocator,
    sys: Sys,
    policy: Policy,
    /// The lock's pins, in the lock's order (`lockDigest`).
    pins: []const Pin,
    /// In load order (every library after the pins it needs).
    libs: []Library,
    /// `libs[0..loaded]` are mapped and owned.
    loaded: usize,
    residents: []Identity,
    needs: Needs,
    /// The loader's counters when `verifyClosure` passed; null until then.
    closure: ?Counts = null,
    /// The first change `checkUnchanged` saw; once set, every later check refuses.
    changed: ?Counts = null,
    /// Plans, handles and other objects built on these libraries and not yet destroyed (`hold` / `drop`).
    dependents: u32 = 0,
    custody: Custody = .open,

    /// Checks the environment; reads each pin's program headers and refuses unless the whole `Needs` fits
    /// `host_budget` (before any memfd or buffer exists); verifies each resident's identity; copies each pin into a
    /// sealed memfd, hashing the source and then the sealed bytes; checks DT_SONAME and that every DT_NEEDED is a pin
    /// or a resident; loads in dependency order. Any refusal unloads what it loaded (no GPU work exists yet). If that
    /// unload fails, the refusal still returns its own error, and `retained` receives the set it could not release.
    pub fn open(gpa: std.mem.Allocator, pins: []const Pin, policy: Policy, host_budget: u64, vendor: VendorBound, sys: Sys, retained: *?OpenRetained) Error!Set {
        retained.* = null;
        // set when a failed unload hands the bookkeeping to `retained`; the frees below then do not run
        var kept_by_retained = false;
        try checkEnv(policy, sys);
        for (pins, 0..) |p, i| {
            if (!std.fs.path.isAbsolute(p.path)) return error.PinPathNotAbsolute;
            for (pins[0..i]) |q| if (std.mem.eql(u8, p.name, q.name)) return error.PinDuplicate;
        }

        // 1. sources and the budget, before allocating
        var src_buf: [max_pins]Fd = undefined;
        if (pins.len > max_pins) return error.PinTooMany;
        const srcs = src_buf[0..pins.len];
        var opened: usize = 0;
        defer for (srcs[0..opened]) |fd| sys.close(sys.ctx, fd);
        var needs: Needs = .{ .transient = chunk_bytes, .vendor_internal = vendor.bytes };
        const page = sys.pageSize(sys.ctx);
        for (pins, srcs) |p, *fd| {
            fd.* = sys.open(sys.ctx, p.path, false) orelse return error.PinOpenFailed;
            opened += 1;
            const st = sys.stat(sys.ctx, fd.*) orelse return error.PinOpenFailed;
            if (!st.regular()) return error.PinNotRegular;
            if (st.size != p.size) return error.PinSizeMismatch;
            needs.shared += p.size;
            needs.private += try privateBytes(sys, fd.*, page);
        }
        if (needs.total() > host_budget) {
            log.err("pinned set needs {d} host bytes (shared {d}, private {d}, transient {d}, vendor {d}); budget {d}", .{ needs.total(), needs.shared, needs.private, needs.transient, needs.vendor_internal, host_budget });
            return error.PinBudgetExceeded;
        }

        const buf = try gpa.alloc(u8, chunk_bytes);
        defer gpa.free(buf);

        // 2. residents by exact identity
        if (policy.preexisting.len > max_residents) return error.PinTooMany;
        const residents = try gpa.alloc(Identity, policy.preexisting.len);
        errdefer if (!kept_by_retained) gpa.free(residents);
        for (policy.preexisting, residents) |r, *id| id.* = try verifyResident(sys, r, buf);

        // 3. sealed copies
        const libs = try gpa.alloc(Library, pins.len);
        defer gpa.free(libs);
        var sealed: usize = 0;
        errdefer for (libs[0..sealed]) |l| sys.close(sys.ctx, l.fd);
        const maps = try gpa.alloc([]const u8, pins.len);
        defer gpa.free(maps);
        var mapped: usize = 0;
        defer for (maps[0..mapped]) |m| sys.unmap(sys.ctx, m);
        for (pins, srcs, libs, maps) |*p, src, *l, *m| {
            l.* = try seal(sys, p, src, buf);
            sealed += 1;
            m.* = sys.mapRead(sys.ctx, l.fd, @intCast(p.size)) orelse return error.PinSealFailed;
            mapped += 1;
            var digest: [32]u8 = undefined;
            Sha256.hash(m.*, &digest, .{}); // the sealed bytes the loader will map
            if (!std.mem.eql(u8, &digest, &p.sha256)) return error.PinHashMismatch;
        }

        // 4. DT_SONAME, DT_NEEDED and the load order
        const dyns = try gpa.alloc(elf.Dynamic, pins.len);
        defer gpa.free(dyns);
        for (pins, maps, dyns) |p, m, *d| {
            d.* = elf.dynamic(m) catch return error.PinElfInvalid;
            const so = d.soname orelse return error.PinSonameMismatch;
            if (!std.mem.eql(u8, so, p.name)) {
                log.err("{s}: DT_SONAME is {s}", .{ p.path, so });
                return error.PinSonameMismatch;
            }
        }
        for (pins, dyns) |p, d| {
            for (d.needed()) |n| {
                if (pinIndex(pins, n) == null and residentIndex(policy.preexisting, n) == null) {
                    log.err("{s} needs {s}, neither pinned nor resident", .{ p.name, n });
                    return error.PinNeededUnpinned;
                }
            }
        }
        const order = try gpa.alloc(usize, pins.len);
        defer gpa.free(order);
        try loadOrder(pins, dyns, order);
        const ordered = try gpa.alloc(Library, pins.len);
        errdefer if (!kept_by_retained) gpa.free(ordered);
        for (order, ordered) |i, *o| o.* = libs[i];
        sealed = 0; // `ordered` owns the fds from here

        // 5. load
        var loaded: usize = 0;
        errdefer {
            // no GPU work exists yet; still, a failed unload keeps it and everything before it mapped, open and owned
            var k = loaded;
            const failed: ?c_int = while (k > 0) {
                k -= 1;
                const rc = sys.dlclose(sys.ctx, ordered[k].handle.?);
                if (rc != 0) {
                    log.err("dlclose {s} failed ({d}); it and {d} earlier libraries stay mapped and retained", .{ ordered[k].pin.name, rc, k });
                    break rc;
                }
                ordered[k].handle = null;
            } else null;
            const kept = if (failed != null) k + 1 else 0;
            for (ordered[kept..]) |l| sys.close(sys.ctx, l.fd);
            if (failed) |rc| {
                kept_by_retained = true;
                retained.* = .{
                    .primary = error.PinLoadFailed,
                    .cleanup = .{ .what = ordered[k].pin.name, .rc = rc },
                    .held = .{ .gpa = gpa, .sys = sys, .policy = policy, .pins = pins, .libs = ordered, .loaded = kept, .residents = residents, .needs = needs, .custody = .retained },
                };
            }
        }
        for (ordered) |*l| {
            var path_buf: [32]u8 = undefined;
            const path = std.mem.printSentinel(&path_buf, "/proc/self/fd/{d}", .{l.fd}, 0) catch unreachable;
            l.handle = sys.dlopen(sys.ctx, path) orelse {
                log.err("dlopen {s} ({s}) failed", .{ l.pin.name, path });
                return error.PinLoadFailed;
            };
            loaded += 1;
            l.base = sys.handleBase(sys.ctx, l.handle.?) orelse return error.PinLoadFailed;
        }
        return .{ .gpa = gpa, .sys = sys, .policy = policy, .pins = pins, .libs = ordered, .loaded = ordered.len, .residents = residents, .needs = needs };
    }

    pub fn get(self: *const Set, name: []const u8) ?*const Library {
        for (self.libs) |*l| if (std.mem.eql(u8, l.pin.name, name)) return l;
        return null;
    }

    /// The closure proof, run by the engine at load after its warm-up executes (under a grant) and before admission:
    /// every object the loader maps must be the program, the vDSO, a pin (its handle's base and its own memfd's inode)
    /// or a resident (its exact path and verified inode), each pin exactly once and each resident at most once. Then
    /// the loader's add / remove counters are recorded for `checkUnchanged`.
    pub fn verifyClosure(self: *Set) Error!void {
        try checkEnv(self.policy, self.sys);
        const objs = self.sys.objects(self.sys.ctx, self.gpa) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Unavailable => error.PinClosureMismatch,
        };
        defer sys_mod.freeObjects(self.gpa, objs);
        var seen_lib = @as([max_pins]bool, @splat(false));
        var seen_res = @as([max_residents]bool, @splat(false));
        for (objs) |o| {
            if (o.kind != .library) continue;
            if (self.libIndex(o)) |i| {
                if (seen_lib[i]) return closureFail(o, "loaded twice");
                seen_lib[i] = true;
            } else if (self.residentFor(o)) |i| {
                if (seen_res[i]) return closureFail(o, "mapped twice");
                seen_res[i] = true;
            } else return closureFail(o, "not pinned and not a resident");
        }
        for (self.libs, 0..) |l, i| if (!seen_lib[i]) {
            log.err("closure: {s} is not mapped from its sealed memfd", .{l.pin.name});
            return error.PinClosureMismatch;
        };
        self.closure = self.sys.counts(self.sys.ctx) orelse return error.PinClosureMismatch;
        self.changed = null;
    }

    /// Before AND after every vendor call: the loader's counters must equal those `verifyClosure` recorded. An add, a
    /// remove or a reload refuses and is recorded. Detection only: it does not show that no unpinned code ran between
    /// two checks, and a failed check must tear the set down (after quiescing), never fall back.
    pub fn checkUnchanged(self: *Set) Error!void {
        const want = self.closure orelse return error.PinUnverified;
        if (self.changed != null) return error.PinMappedSetChanged;
        const now = self.sys.counts(self.sys.ctx) orelse return error.PinMappedSetChanged;
        if (now.adds == want.adds and now.subs == want.subs) return;
        self.changed = now;
        log.err("mapped set changed since the closure check: adds {d} -> {d}, removes {d} -> {d}", .{ want.adds, now.adds, want.subs, now.subs });
        return error.PinMappedSetChanged;
    }

    /// A plan, handle or other object built on these libraries: `close` refuses while any is held. Dropped only when
    /// its destruction succeeded.
    pub fn hold(self: *Set) void {
        self.dependents += 1;
    }

    pub fn drop(self: *Set) void {
        std.debug.assert(self.dependents > 0);
        self.dependents -= 1;
    }

    /// Unloads in reverse load order, closing each memfd after its library unloaded, then frees the bookkeeping. Refuses
    /// (unloading nothing) without a valid quiescence proof, while any dependent is held, or after an earlier failed
    /// release or `abandon`. The first failed dlclose stops it: see `Release`.
    pub fn close(self: *Set, q: Quiesced) CloseError!Release {
        switch (self.custody) {
            .open => {},
            .release_failed => return error.ReleaseFailed,
            .retained => return error.Retained,
            .released => return error.AlreadyReleased,
        }
        if (!q.valid()) return error.NotQuiesced;
        if (self.dependents != 0) {
            log.err("pinned set: {d} dependent plans / handles still held; nothing unloaded", .{self.dependents});
            return error.DependentsOutstanding;
        }
        var out: Release = .{ .unloaded = 0 };
        while (self.loaded > 0) {
            const l = &self.libs[self.loaded - 1];
            const rc = self.sys.dlclose(self.sys.ctx, l.handle.?);
            if (rc != 0) {
                log.err("dlclose {s} failed ({d}); it and {d} earlier libraries stay mapped and owned", .{ l.pin.name, rc, self.loaded - 1 });
                self.custody = .release_failed;
                out.failed = .{ .name = l.pin.name, .rc = rc };
                return out;
            }
            self.sys.close(self.sys.ctx, l.fd);
            l.handle = null;
            self.loaded -= 1;
            out.unloaded += 1;
        }
        self.custody = .released;
        self.gpa.free(self.libs);
        self.gpa.free(self.residents);
        return out;
    }

    /// When quiescence could not be established within the owner's bound, or a dependent's destruction failed: keeps
    /// every library mapped, every memfd open and the bookkeeping, for the life of the process. Sticky.
    pub fn abandon(self: *Set) void {
        if (self.custody == .open) self.custody = .retained;
        log.err("pinned set retained without release: {d} libraries stay mapped", .{self.loaded});
    }

    fn libIndex(self: *const Set, o: sys_mod.Object) ?usize {
        for (self.libs, 0..) |l, i| {
            if (o.base == l.base and o.ino == l.ino and o.dev_major == l.dev_major and o.dev_minor == l.dev_minor) return i;
        }
        return null;
    }

    fn residentFor(self: *const Set, o: sys_mod.Object) ?usize {
        for (self.policy.preexisting, self.residents, 0..) |r, id, i| {
            if (std.mem.eql(u8, o.path, r.path) and o.ino == id.ino and o.dev_major == id.dev_major and o.dev_minor == id.dev_minor) return i;
        }
        return null;
    }
};

/// sha256 over the pins in order (each: name, a 0 byte, sha256, size little-endian): what a grant binds.
pub fn lockDigest(pins: []const Pin) [32]u8 {
    var h = Sha256.init(.{});
    for (pins) |p| {
        h.update(p.name);
        h.update(&.{0});
        h.update(&p.sha256);
        var size: [8]u8 = undefined;
        std.mem.writeInt(u64, &size, p.size, .little);
        h.update(&size);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

fn closureFail(o: sys_mod.Object, why: []const u8) Error {
    log.err("closure: {s} (base 0x{x}, inode {d}:{d}/{d}) {s}", .{ o.path, o.base, o.dev_major, o.dev_minor, o.ino, why });
    return error.PinClosureMismatch;
}

fn checkEnv(policy: Policy, sys: Sys) Error!void {
    for (policy.deny_env) |name| if (sys.envPresent(sys.ctx, name)) {
        log.err("{s} is set; the pinned set refuses it", .{name});
        return error.PinEnvRefused;
    };
}

/// The private bytes of a source's writable segments, from its ELF and program headers alone.
fn privateBytes(sys: Sys, fd: Fd, page: usize) Error!u64 {
    var ehdr: [elf.ehdr_size]u8 = undefined;
    if ((sys.pread(sys.ctx, fd, &ehdr, 0) orelse return error.PinOpenFailed) != ehdr.len) return error.PinElfInvalid;
    const h = elf.header(&ehdr) catch return error.PinElfInvalid;
    if (h.phnum > max_phdrs) return error.PinElfInvalid;
    var ph: [max_phdrs * 56]u8 = undefined;
    const n: usize = @intCast(h.phdrBytes());
    if ((sys.pread(sys.ctx, fd, ph[0..n], h.phoff) orelse return error.PinOpenFailed) != n) return error.PinElfInvalid;
    return elf.writableBytes(ph[0..n], h.phnum, page) catch error.PinElfInvalid;
}

/// Copies exactly `pin.size` bytes from `src` into a new memfd, hashing them on the way; the source must end there.
/// Then seals it and reads the seals back.
fn seal(sys: Sys, pin: *const Pin, src: Fd, buf: []u8) Error!Library {
    var name_buf: [96]u8 = undefined;
    const name = std.mem.printSentinel(&name_buf, "vendor:{s}", .{pin.name}, 0) catch return error.PinSealFailed;
    const fd = sys.memfd(sys.ctx, name) orelse return error.PinSealFailed;
    errdefer sys.close(sys.ctx, fd);
    var h = Sha256.init(.{});
    var off: u64 = 0;
    while (off < pin.size) {
        const want: usize = @intCast(@min(buf.len, pin.size - off));
        const n = sys.pread(sys.ctx, src, buf[0..want], off) orelse return error.PinOpenFailed;
        if (n != want) return error.PinSizeMismatch;
        h.update(buf[0..n]);
        if (!sys.writeAll(sys.ctx, fd, buf[0..n])) return error.PinSealFailed;
        off += n;
    }
    var tail: [1]u8 = undefined;
    if ((sys.pread(sys.ctx, src, &tail, off) orelse return error.PinOpenFailed) != 0) return error.PinSizeMismatch;
    var digest: [32]u8 = undefined;
    h.final(&digest);
    if (!std.mem.eql(u8, &digest, &pin.sha256)) {
        log.err("{s}: sha256 differs from the lock", .{pin.path});
        return error.PinHashMismatch;
    }
    if (!sys.addSeals(sys.ctx, fd, seals_required)) return error.PinSealFailed;
    const got = sys.getSeals(sys.ctx, fd) orelse return error.PinSealFailed;
    if (got & seals_required != seals_required) return error.PinSealFailed;
    const st = sys.stat(sys.ctx, fd) orelse return error.PinSealFailed;
    if (st.size != pin.size) return error.PinSizeMismatch;
    return .{ .pin = pin, .sys = sys, .fd = fd, .dev_major = st.dev_major, .dev_minor = st.dev_minor, .ino = st.ino };
}

/// A resident's identity: an fs-verity file whose digest, size and sha256 equal the lock's.
pub fn verifyResident(sys: Sys, r: Resident, buf: []u8) Error!Identity {
    const want_verity = r.verity orelse {
        log.err("{s}: no fs-verity digest in the lock, so no immutable provenance", .{r.path});
        return error.PinNotImmutable;
    };
    const fd = sys.open(sys.ctx, r.path, true) orelse return error.PinOpenFailed;
    defer sys.close(sys.ctx, fd);
    const st = sys.stat(sys.ctx, fd) orelse return error.PinOpenFailed;
    if (!st.regular()) return error.PinNotRegular;
    if (!st.verity) return error.PinNotImmutable;
    const v = sys.measureVerity(sys.ctx, fd) orelse return error.PinNotImmutable;
    if (!std.mem.eql(u8, &v, &want_verity)) return error.PinHashMismatch;
    if (st.size != r.size) return error.PinSizeMismatch;
    var h = Sha256.init(.{});
    var off: u64 = 0;
    while (off < r.size) {
        const want: usize = @intCast(@min(buf.len, r.size - off));
        const n = sys.pread(sys.ctx, fd, buf[0..want], off) orelse return error.PinOpenFailed;
        if (n != want) return error.PinSizeMismatch;
        h.update(buf[0..n]);
        off += n;
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    if (!std.mem.eql(u8, &digest, &r.sha256)) return error.PinHashMismatch;
    return .{ .dev_major = st.dev_major, .dev_minor = st.dev_minor, .ino = st.ino };
}

fn pinIndex(pins: []const Pin, soname: []const u8) ?usize {
    for (pins, 0..) |p, i| if (std.mem.eql(u8, p.name, soname)) return i;
    return null;
}

fn residentIndex(rs: []const Resident, soname: []const u8) ?usize {
    for (rs, 0..) |r, i| if (std.mem.eql(u8, r.soname, soname)) return i;
    return null;
}

/// Every pin after the pins it needs; a cycle refuses.
fn loadOrder(pins: []const Pin, dyns: []const elf.Dynamic, out: []usize) Error!void {
    var placed = @as([max_pins]bool, @splat(false));
    var n: usize = 0;
    while (n < pins.len) {
        const before = n;
        for (pins, 0..) |_, i| {
            if (placed[i]) continue;
            const ready = for (dyns[i].needed()) |need| {
                if (pinIndex(pins, need)) |j| if (!placed[j]) break false;
            } else true;
            if (!ready) continue;
            placed[i] = true;
            out[n] = i;
            n += 1;
        }
        if (n == before) return error.PinOrderCycle;
    }
}

// ---- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;
const Fake = @import("fake.zig").Fake;

const Fixture = struct {
    a_buf: [1024]u8 = undefined,
    b_buf: [1024]u8 = undefined,
    libc_bytes: []const u8 = "the libc bytes",
    files: [3]Fake.File = undefined,
    pins: [2]Pin = undefined,
    residents: [1]Resident = undefined,
    fake: Fake = undefined,
    retained: ?OpenRetained = null,

    /// liba needs libb and libc; libb needs libc; libc is a verified resident.
    fn init(self: *Fixture) void {
        const a = elf.testObject(&self.a_buf, "liba.so.1", &.{ "libb.so.2", "libc.so.6" }, 0x1800);
        const b = elf.testObject(&self.b_buf, "libb.so.2", &.{"libc.so.6"}, 0x800);
        self.files = .{
            .{ .path = "/opt/vendor/liba.so.1", .bytes = a, .ino = 11 },
            .{ .path = "/opt/vendor/libb.so.2", .bytes = b, .ino = 12 },
            .{ .path = "/usr/lib/libc.so.6", .bytes = self.libc_bytes, .ino = 13, .verity = @as([32]u8, @splat(0xab)) },
        };
        self.pins = .{
            .{ .name = "liba.so.1", .path = "/opt/vendor/liba.so.1", .sha256 = sha(a), .size = a.len },
            .{ .name = "libb.so.2", .path = "/opt/vendor/libb.so.2", .sha256 = sha(b), .size = b.len },
        };
        self.residents = .{.{ .path = "/usr/lib/libc.so.6", .soname = "libc.so.6", .size = self.libc_bytes.len, .sha256 = sha(self.libc_bytes), .verity = @as([32]u8, @splat(0xab)) }};
        self.fake = .{ .files = &self.files, .preexisting = &fixture_objects };
    }

    fn policy(self: *const Fixture) Policy {
        return .{ .preexisting = &self.residents, .deny_env = &deny_env_default };
    }

    fn open(self: *Fixture) Error!Set {
        return Set.open(testing.allocator, &self.pins, self.policy(), 64 << 30, unmeasured_vendor_bound, self.fake.sys(), &self.retained);
    }
};

const fixture_objects = [_]sys_mod.Object{
    .{ .path = "", .base = 0x400000, .dev_major = 8, .dev_minor = 1, .ino = 99, .kind = .main },
    .{ .path = "linux-vdso.so.1", .base = 0x7fff0000, .dev_major = 0, .dev_minor = 0, .ino = 0, .kind = .vdso },
    .{ .path = "/usr/lib/libc.so.6", .base = 0x7f000000, .dev_major = 8, .dev_minor = 1, .ino = 13, .kind = .library },
};

fn sha(b: []const u8) [32]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(b, &d, .{});
    return d;
}

test "opens, orders by DT_NEEDED, verifies the closure, and resolves only its own symbols" {
    var f: Fixture = .{};
    f.init();
    var set = try f.open();
    defer closeOk(&set);
    // libb is needed by liba, so it loads first
    try testing.expectEqualStrings("libb.so.2", set.libs[0].pin.name);
    try testing.expectEqualStrings("liba.so.1", set.libs[1].pin.name);
    try testing.expectEqual(@as(u64, f.pins[0].size + f.pins[1].size), set.needs.shared);
    try testing.expectEqual(@as(u64, 0x2000 + 0x1000), set.needs.private);
    try testing.expectError(error.PinUnverified, set.checkUnchanged());
    try set.verifyClosure();
    try set.checkUnchanged();

    const Api = struct { a_fn: *const fn () callconv(.c) c_int };
    _ = try set.get("liba.so.1").?.resolve(Api);
    try testing.expectError(error.PinMissingSymbol, set.get("liba.so.1").?.resolve(struct { missing_fn: *const fn () callconv(.c) void }));
    try testing.expectError(error.PinSymbolElsewhere, set.get("liba.so.1").?.resolve(struct { foreign_fn: *const fn () callconv(.c) void }));
    try testing.expect(set.get("libc.so.6") == null);
}

test "an extra mapping, a reload or an unload after the closure check refuses" {
    var f: Fixture = .{};
    f.init();
    var set = try f.open();
    defer closeOk(&set);
    f.fake.extra = .{ .path = "/usr/lib/libcudnn_graph.so.9", .base = 0x9000000, .dev_major = 8, .dev_minor = 1, .ino = 77, .kind = .library };
    try testing.expectError(error.PinClosureMismatch, set.verifyClosure());
    f.fake.extra = null;
    try set.verifyClosure();
    f.fake.adds += 1;
    f.fake.subs += 1; // one object loaded and one unloaded: the count of mapped objects is unchanged, the counters are not
    try testing.expectError(error.PinMappedSetChanged, set.checkUnchanged());
    f.fake.adds -= 1;
    f.fake.subs -= 1;
    try testing.expectError(error.PinMappedSetChanged, set.checkUnchanged()); // once changed, stays refused
}

test "a resident at an unverified inode, or one without fs-verity, refuses" {
    var f: Fixture = .{};
    f.init();
    {
        var set = try f.open();
        defer closeOk(&set);
        var objs = fixture_objects;
        objs[2].ino = 14; // same path, a different file
        f.fake.preexisting = &objs;
        try testing.expectError(error.PinClosureMismatch, set.verifyClosure());
    }
    f.residents[0].verity = null;
    try testing.expectError(error.PinNotImmutable, f.open());
    f.residents[0].verity = @as([32]u8, @splat(0xab));
    f.files[2].verity = null;
    try testing.expectError(error.PinNotImmutable, f.open());
}

test "lock mismatches refuse before anything loads" {
    var f: Fixture = .{};
    f.init();
    f.pins[0].path = "opt/vendor/liba.so.1";
    try testing.expectError(error.PinPathNotAbsolute, f.open());
    f.init();
    f.pins[0].size += 1;
    try testing.expectError(error.PinSizeMismatch, f.open());
    f.init();
    f.pins[0].sha256[0] ^= 1;
    try testing.expectError(error.PinHashMismatch, f.open());
    f.init();
    f.pins[1].name = "libb.so.3";
    try testing.expectError(error.PinSonameMismatch, f.open());
    f.init();
    f.residents[0].soname = "libc.so.7"; // liba / libb need libc.so.6, now neither pinned nor resident
    try testing.expectError(error.PinNeededUnpinned, f.open());
    f.init();
    f.fake.drop_seal = true;
    try testing.expectError(error.PinSealFailed, f.open());
    f.init();
    f.fake.env = &.{"LD_LIBRARY_PATH"};
    try testing.expectError(error.PinEnvRefused, f.open());
    try testing.expectEqual(@as(usize, 0), f.fake.n_loaded);
}

test "the budget is checked before any memfd exists" {
    var f: Fixture = .{};
    f.init();
    const sizes = f.pins[0].size + f.pins[1].size;
    const exact = sizes + 0x3000 + chunk_bytes + unmeasured_vendor_bound.bytes;
    try testing.expectError(error.PinBudgetExceeded, Set.open(testing.allocator, &f.pins, f.policy(), exact - 1, unmeasured_vendor_bound, f.fake.sys(), &f.retained));
    try testing.expectEqual(@as(usize, 0), f.fake.memfds);
    var set = try Set.open(testing.allocator, &f.pins, f.policy(), exact, unmeasured_vendor_bound, f.fake.sys(), &f.retained);
    closeOk(&set);
}

test "close needs a valid quiescence proof and no held dependents; abandon retains everything" {
    var f: Fixture = .{};
    f.init();
    var set = try f.open();
    var timed_out = quiesce.testProof();
    timed_out.obs.outcome = .timed_out;
    try testing.expectError(error.NotQuiesced, set.close(timed_out));
    set.hold();
    try testing.expectError(error.DependentsOutstanding, set.close(quiesce.testProof()));
    try testing.expectEqual(@as(usize, 0), f.fake.dlclosed);
    // the plan's destruction failed, so the dependent is never dropped: the owner retains the set
    set.abandon();
    try testing.expectError(error.Retained, set.close(quiesce.testProof()));
    try testing.expectEqual(@as(usize, 0), f.fake.dlclosed);
    try testing.expectEqual(@as(usize, 2), set.loaded);
    freeRetained(&set);
}

test "a failed unload is sticky, reported with its raw result, and keeps the rest mapped and owned" {
    var f: Fixture = .{};
    f.init();
    var set = try f.open();
    f.fake.dlclose_fail_at = 0; // the first unload (liba, loaded last) fails
    const r = try set.close(quiesce.testProof());
    try testing.expect(!r.ok());
    try testing.expectEqual(@as(c_int, 1), r.failed.?.rc);
    try testing.expectEqualStrings("liba.so.1", r.failed.?.name);
    try testing.expectEqual(@as(usize, 2), set.loaded);
    try testing.expectEqual(Custody.release_failed, set.custody);
    f.fake.dlclose_fail_at = null;
    try testing.expectError(error.ReleaseFailed, set.close(quiesce.testProof())); // no second attempt unloads
    try testing.expectEqual(@as(usize, 0), f.fake.dlclosed);
    freeRetained(&set);
}

test "a refused load whose unload fails keeps its handles, memfds and bookkeeping, and both errors" {
    var f: Fixture = .{};
    f.init();
    f.fake.dlopen_fail_at = 1; // libb loads, then liba's dlopen fails
    f.fake.dlclose_fail_at = 0; // and unloading libb fails
    try testing.expectError(error.PinLoadFailed, f.open());
    const r = &f.retained.?;
    try testing.expectEqual(Error.PinLoadFailed, r.primary);
    try testing.expectEqualStrings("libb.so.2", r.cleanup.what);
    try testing.expectEqual(@as(c_int, 1), r.cleanup.rc);
    try testing.expectEqual(Custody.retained, r.held.custody);
    try testing.expectEqual(@as(usize, 1), r.held.loaded);
    try testing.expectEqualStrings("libb.so.2", r.held.libs[0].pin.name);
    try testing.expect(r.held.libs[0].handle != null);
    try testing.expect(f.fake.open_fds[@intCast(r.held.libs[0].fd - 3)] != null); // libb's memfd stays open
    try testing.expect(f.fake.open_fds[@intCast(r.held.libs[1].fd - 3)] == null); // liba's, never loaded, is closed
    try testing.expectEqual(@as(usize, 1), r.held.residents.len);
    // the record is not a release path: nothing retries the unload
    f.fake.dlclose_fail_at = null;
    try testing.expectError(error.Retained, r.held.close(quiesce.testProof()));
    try testing.expectEqual(@as(usize, 0), f.fake.dlclosed);
    freeRetained(&r.held);

    // a refused load whose unload succeeds leaves no record
    f.init();
    f.retained = null;
    f.fake.dlopen_fail_at = 1;
    try testing.expectError(error.PinLoadFailed, f.open());
    try testing.expect(f.retained == null);
    try testing.expectEqual(@as(usize, 1), f.fake.dlclosed);
}

fn closeOk(set: *Set) void {
    const r = set.close(quiesce.testProof()) catch unreachable;
    std.debug.assert(r.ok());
}

/// Tests only: frees a retained set's bookkeeping (in a process, it stays for the process's life).
fn freeRetained(set: *Set) void {
    set.gpa.free(set.libs);
    set.gpa.free(set.residents);
}
