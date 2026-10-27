//! Speech first, for the chat model: while a voice reply's speech is behind, the chat model's server is paused (SIGSTOP)
//! so speech has the GPU, and resumed (SIGCONT) once speech is ahead. Two processes on GB10 time-slice the GPU with no
//! priority between them (docs/dev/CHAT.md); beside a decoding chat model Drama 3 comes at 0.55x real time, alone at
//! 1.4x. Pausing is lossless: the server's decode resumes where it stopped, with the same numbers.
//!
//! A holder holds a lease (`hold`, renewed with each piece of speech) and lets go (`release`); the server runs again
//! when no lease is live, so a speaker that stalls or a session that goes away never leaves it stopped for longer than
//! one lease. Only the server is stopped, not the worker that supervises it: a daemon that dies still reaches the
//! worker, which kills the server (SIGKILL ends a stopped process).

const std = @import("std");
const Io = std.Io;
const posix = std.posix;

const max_holders = 8;

pub const Yielder = struct {
    mutex: Io.Mutex = .init,
    cond: Io.Condition = .init,
    pid: posix.pid_t = 0, // the stopped server; 0: none stopped
    holders: [max_holders]Holder = @splat(.{}),
    thread: ?std.Thread = null,
    closing: bool = false,
    stops: u64 = 0, // pauses so far (logged when the daemon stops)

    const Holder = struct { id: usize = 0, until: i96 = 0 };

    /// Resumes the server and ends the watchdog.
    pub fn deinit(y: *Yielder, io: Io) void {
        y.mutex.lockUncancelable(io);
        y.closing = true;
        for (&y.holders) |*h| h.* = .{};
        _ = y.apply(io);
        y.cond.broadcast(io);
        const t = y.thread;
        y.mutex.unlock(io);
        if (t) |th| th.join();
    }

    /// `id` (non-zero) keeps `pid` stopped for `lease_ns` more; false when it cannot (no watchdog, too many holders).
    pub fn hold(y: *Yielder, io: Io, pid: posix.pid_t, id: usize, lease_ns: i96) bool {
        if (pid <= 0 or id == 0) return false;
        y.mutex.lockUncancelable(io);
        defer y.mutex.unlock(io);
        if (y.closing) return false;
        if (y.thread == null) y.thread = std.Thread.spawn(.{}, watch, .{ y, io }) catch return false;
        if (y.pid != 0 and y.pid != pid) y.wake(); // another server (reloaded): only one is ever stopped
        const until = now(io) + lease_ns;
        const slot = for (&y.holders) |*h| {
            if (h.id == id) break h;
        } else for (&y.holders) |*h| {
            if (h.id == 0) break h;
        } else return false;
        slot.* = .{ .id = id, .until = until };
        if (y.pid == 0) {
            if (posix.system.kill(pid, posix.SIG.STOP) != 0) {
                slot.* = .{};
                return false;
            }
            y.pid = pid;
            y.stops += 1;
        }
        y.cond.broadcast(io);
        return true;
    }

    /// `id` lets go; the server runs again once no lease is live.
    pub fn release(y: *Yielder, io: Io, id: usize) void {
        y.mutex.lockUncancelable(io);
        defer y.mutex.unlock(io);
        for (&y.holders) |*h| if (h.id == id) {
            h.* = .{};
        };
        _ = y.apply(io);
    }

    /// Whether a server is stopped now.
    pub fn stopped(y: *Yielder, io: Io) bool {
        y.mutex.lockUncancelable(io);
        defer y.mutex.unlock(io);
        return y.pid != 0;
    }

    /// Under the mutex: expired leases go; with none left the server runs. Returns the next lease's end (0: none).
    fn apply(y: *Yielder, io: Io) i96 {
        const t = now(io);
        var next: i96 = 0;
        for (&y.holders) |*h| {
            if (h.id == 0) continue;
            if (h.until <= t) {
                h.* = .{};
                continue;
            }
            if (next == 0 or h.until < next) next = h.until;
        }
        if (next == 0) y.wake();
        return next;
    }

    fn wake(y: *Yielder) void {
        if (y.pid == 0) return;
        _ = posix.system.kill(y.pid, posix.SIG.CONT);
        y.pid = 0;
    }

    /// The watchdog: ends each lease when it runs out, whatever its holder is doing.
    fn watch(y: *Yielder, io: Io) void {
        y.mutex.lockUncancelable(io);
        defer y.mutex.unlock(io);
        while (!y.closing) {
            const next = y.apply(io);
            const wait_ns: u64 = if (next == 0) std.time.ns_per_s else @intCast(@max(next - now(io), 1));
            y.cond.waitTimeout(io, &y.mutex, .{ .duration = .{ .raw = .fromNanoseconds(wait_ns), .clock = .awake } }) catch {};
        }
    }
};

fn now(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

/// A /proc file's text (its size reads 0, so it is read to its end), at most `buf.len`; "" when it cannot be read.
fn readProc(path: []const u8, buf: []u8) []const u8 {
    const fd = posix.openat(posix.AT.FDCWD, path, .{ .CLOEXEC = true }, 0) catch return "";
    defer _ = std.c.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const got = posix.read(fd, buf[n..]) catch return "";
        if (got == 0) break;
        n += got;
    }
    return buf[0..n];
}

/// The first child of `pid` (`/proc/<pid>/task/<pid>/children`: its main thread's), or 0.
pub fn childOf(pid: posix.pid_t) posix.pid_t {
    if (pid <= 0) return 0;
    var path_buf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/proc/{d}/task/{d}/children", .{ pid, pid }) catch return 0;
    var buf: [256]u8 = undefined;
    const text = readProc(path, &buf);
    var it = std.mem.tokenizeAny(u8, text, " \n");
    const first = it.next() orelse return 0;
    return std.fmt.parseInt(posix.pid_t, first, 10) catch 0;
}

const testing = std.testing;

/// A child that sleeps: something to stop and resume.
fn sleeper() !std.process.Child {
    return std.process.spawn(testing.io, .{ .argv = &.{ "sleep", "30" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
}

/// The process's state letter from /proc (T: stopped).
fn stateOf(pid: posix.pid_t) u8 {
    var path_buf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/proc/{d}/stat", .{pid}) catch return 0;
    var buf: [512]u8 = undefined;
    const text = readProc(path, &buf);
    const close = std.mem.lastIndexOfScalar(u8, text, ')') orelse return 0;
    return if (close + 2 < text.len) text[close + 2] else 0;
}

test "a lease stops the server; it runs again on release, when the last of two lets go, and when a lease runs out" {
    var child = try sleeper();
    defer {
        _ = posix.system.kill(child.id.?, posix.SIG.KILL);
        _ = child.wait(testing.io) catch {};
    }
    const pid = child.id.?;
    var y: Yielder = .{};
    defer y.deinit(testing.io);
    try testing.expect(y.hold(testing.io, pid, 1, std.time.ns_per_s));
    try testing.expectEqual(@as(u8, 'T'), stateOf(pid));
    try testing.expect(y.hold(testing.io, pid, 2, std.time.ns_per_s));
    y.release(testing.io, 1);
    try testing.expectEqual(@as(u8, 'T'), stateOf(pid)); // 2 still holds it
    y.release(testing.io, 2);
    try testing.expect(stateOf(pid) != 'T');
    try testing.expect(y.hold(testing.io, pid, 3, 50 * std.time.ns_per_ms)); // a holder that never lets go
    try testing.expectEqual(@as(u8, 'T'), stateOf(pid));
    for (0..100) |_| {
        if (!y.stopped(testing.io)) break;
        Io.sleep(testing.io, .fromMilliseconds(10), .awake) catch {};
    }
    try testing.expect(!y.stopped(testing.io));
    try testing.expect(stateOf(pid) != 'T');
    try testing.expectEqual(@as(u64, 2), y.stops);
}

test "the daemon going away resumes a stopped server" {
    var child = try sleeper();
    defer {
        _ = posix.system.kill(child.id.?, posix.SIG.KILL);
        _ = child.wait(testing.io) catch {};
    }
    var y: Yielder = .{};
    try testing.expect(y.hold(testing.io, child.id.?, 7, 10 * std.time.ns_per_s));
    try testing.expectEqual(@as(u8, 'T'), stateOf(child.id.?));
    y.deinit(testing.io);
    try testing.expect(stateOf(child.id.?) != 'T');
    try testing.expect(!y.hold(testing.io, child.id.?, 7, std.time.ns_per_s)); // closed
}

test "childOf finds a spawned child" {
    try testing.expectEqual(@as(posix.pid_t, 0), childOf(0));
    var child = try sleeper();
    defer {
        _ = posix.system.kill(child.id.?, posix.SIG.KILL);
        _ = child.wait(testing.io) catch {};
    }
    const me: posix.pid_t = @intCast(std.os.linux.getpid());
    const found = childOf(me);
    try testing.expect(found > 0); // the test runner may have other children; this one is among them on Linux
}
