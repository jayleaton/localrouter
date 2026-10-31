//! Dying with the parent, so an orphan never holds model memory. Linux has PR_SET_PDEATHSIG; macOS has no such thing,
//! so there a thread waits on the parent with kqueue (EVFILT_PROC, NOTE_EXIT) and sends this process the same signal.
//!
//! A `cmd` worker is someone else's program and cannot arm anything, so on macOS the daemon starts it through
//! `localrouter guard PARENT -- CMD...` (`guard`): a watchdog child is forked off, then CMD is exec'd in place, keeping
//! the pid, pipes and process group the daemon gave it. When the daemon exits the watchdog kills the worker's group
//! (the worker, any server it started, and the watchdog itself); when the worker exits so does the watchdog.

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const linux = std.os.linux;

const darwin = builtin.os.tag.isDarwin();

/// Whether `cmd` workers need `guard` (no parent-death signal on this OS).
pub const needs_guard = darwin;

var armed_sig: std.atomic.Value(u8) = .init(0);
var watching: std.atomic.Value(bool) = .init(false);

/// This process gets `sig` when its parent exits (again with a new signal: the last one wins). False on macOS when the
/// parent is already gone, or cannot be watched.
pub fn arm(sig: posix.SIG) bool {
    if (builtin.os.tag == .linux) {
        _ = linux.prctl(@backingInt(linux.PR.SET_PDEATHSIG), @backingInt(sig), 0, 0, 0);
        return true;
    }
    if (!darwin) return false;
    armed_sig.store(@intCast(@backingInt(sig)), .release);
    if (watching.swap(true, .acq_rel)) return true;
    const kq = watchExit(std.c.getppid()) orelse return false;
    const t = std.Thread.spawn(.{ .stack_size = 64 << 10 }, waitParent, .{kq}) catch return false;
    t.detach();
    return true;
}

fn waitParent(kq: c_int) void {
    _ = waitAny(kq);
    _ = std.c.kill(std.c.getpid(), @fromBackingInt(@intCast(armed_sig.load(.acquire))));
}

/// A kqueue that reports when `pid` exits; null when it already has (ESRCH) or kqueue fails.
fn watchExit(pid: posix.pid_t) ?c_int {
    if (pid <= 1) return null; // reparented to init/launchd: the parent is gone
    const kq = std.c.kqueue();
    if (kq < 0) return null;
    if (!addExit(kq, pid)) {
        _ = std.c.close(kq);
        return null;
    }
    return kq;
}

fn addExit(kq: c_int, pid: posix.pid_t) bool {
    const ch = [1]std.c.Kevent{.{ .ident = @intCast(pid), .filter = std.c.EVFILT.PROC, .flags = std.c.EV.ADD | std.c.EV.ONESHOT, .fflags = std.c.NOTE.EXIT, .data = 0, .udata = @intCast(pid) }};
    var none: [0]std.c.Kevent = .{};
    return std.c.kevent(kq, &ch, 1, &none, 0, null) == 0;
}

/// Blocks until one of the watched processes exits; its pid.
fn waitAny(kq: c_int) posix.pid_t {
    var ev: [1]std.c.Kevent = undefined;
    while (true) {
        const n = std.c.kevent(kq, &[0]std.c.Kevent{}, 0, &ev, 1, null);
        if (n == 1) return @intCast(ev[0].ident);
        if (n < 0 and std.c._errno().* != @backingInt(std.c.E.INTR)) return 0;
    }
}

/// `localrouter guard PARENT -- CMD...` (macOS): starts the watchdog, then execs CMD in this process. Returns only on
/// a usage error, when the daemon is already gone (0), or when CMD cannot be exec'd (127).
pub fn guard(args: []const [:0]const u8) u8 {
    if (!darwin or args.len < 3 or !std.mem.eql(u8, args[1], "--")) {
        std.debug.print("usage: localrouter guard PARENT -- CMD [ARGS...] (macOS)\n", .{});
        return 2;
    }
    const parent = std.fmt.parseInt(posix.pid_t, args[0], 10) catch return 2;
    if (std.c.getppid() != parent) return 0; // the daemon is already gone: start nothing
    var argv: [256:null]?[*:0]const u8 = @splat(null);
    if (args.len - 2 > argv.len) return 2;
    for (args[2..], 0..) |a, i| argv[i] = a.ptr;
    const worker = std.c.getpid();
    // Fork twice so the watchdog is launchd's child, not a zombie of CMD's; it stays in this process group, so the
    // daemon's group kill takes it too. A kqueue is not inherited, so the watchdog makes its own.
    const mid = fork();
    if (mid < 0) return 127;
    if (mid == 0) {
        if (fork() == 0) watchdog(parent, worker);
        std.c._exit(0);
    }
    _ = std.c.waitpid(mid, null, 0);
    _ = execvp(args[2].ptr, &argv);
    std.debug.print("guard: cannot run {s}: errno {d}\n", .{ args[2], std.c._errno().* });
    return 127;
}

/// Holds none of the worker's pipes, so the daemon still sees EOF when the worker dies. Exits with the worker; kills
/// the worker's group when the daemon exits first.
fn watchdog(parent: posix.pid_t, worker: posix.pid_t) noreturn {
    const null_fd = std.c.open("/dev/null", .{ .ACCMODE = .RDWR });
    if (null_fd >= 0) {
        _ = std.c.dup2(null_fd, 0);
        _ = std.c.dup2(null_fd, 1);
    }
    const kq = std.c.kqueue();
    if (kq < 0) std.c._exit(1);
    if (!addExit(kq, worker)) std.c._exit(0); // the worker is gone (exec failed)
    const parent_gone = !addExit(kq, parent) or waitAny(kq) == parent;
    if (parent_gone) {
        // the worker leads its own group (the daemon spawns it so): take the group, else just the worker
        _ = std.c.kill(if (getpgrp() == worker) -worker else worker, .KILL);
    }
    std.c._exit(0);
}

extern "c" fn fork() posix.pid_t;
extern "c" fn execvp(file: [*:0]const u8, argv: [*:null]const ?[*:0]const u8) c_int;
extern "c" fn getpgrp() posix.pid_t;
