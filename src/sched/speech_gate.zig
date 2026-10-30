//! Speech first: while the daemon makes speech, image and video workers pause between their steps, so speech has the
//! GPU and plays without gaps. On GB10 a reply spoken while a picture was generating came below real time and
//! stalled for 0.45 to 2.2 s.
//!
//! The daemon holds an exclusive flock on `<data>/locks/speech.gate` while any speech job runs (`Holder`). At each of
//! its progress reports (every diffusion step, and each phase), an image or video worker takes a shared lock, waiting
//! until it can (`Waiter`). A step already running finishes; the next waits. The wait is bounded, so a heavy job is
//! slowed, never stopped: at most `max_wait_ns` at a time and `budget_ns` a job.

const std = @import("std");
const Io = std.Io;
const posix = std.posix;

/// The worker's environment variable naming the gate file.
pub const env_name = "LOCALROUTER_SPEECH_GATE";

const linger_ns: i96 = 250 * std.time.ns_per_ms; // quiet after speech before a step starts: a reply's next sentence
const poll_ns: u64 = 10; // ms between checks while speech runs
const max_wait_ns: i96 = 6 * std.time.ns_per_s; // one wait at most (a long reply: the job goes on, slower)
const budget_ns: i96 = 30 * std.time.ns_per_s; // waiting a job may do in all

fn now(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

/// Opens (creating it) the gate file; -1 when it cannot, and then nothing waits.
fn open(path: []const u8) posix.fd_t {
    if (path.len == 0) return -1;
    return posix.openat(posix.AT.FDCWD, path, .{ .CLOEXEC = true, .CREAT = true }, 0o644) catch -1;
}

/// The daemon's side: speech jobs in progress; the exclusive lock is held while there is at least one.
pub const Holder = struct {
    fd: posix.fd_t = -1,
    mutex: Io.Mutex = .init,
    count: u32 = 0,

    pub fn init(path: []const u8) Holder {
        return .{ .fd = open(path) };
    }

    pub fn deinit(h: *Holder) void {
        if (h.fd >= 0) _ = std.c.close(h.fd);
        h.fd = -1;
    }

    /// A speech job starts.
    pub fn enter(h: *Holder, io: Io) void {
        h.mutex.lockUncancelable(io);
        defer h.mutex.unlock(io);
        if (h.count == 0 and h.fd >= 0) _ = std.c.flock(h.fd, posix.LOCK.EX); // a worker holds its shared lock for an instant
        h.count += 1;
    }

    /// A speech job ended.
    pub fn leave(h: *Holder, io: Io) void {
        h.mutex.lockUncancelable(io);
        defer h.mutex.unlock(io);
        h.count -|= 1;
        if (h.count == 0 and h.fd >= 0) _ = std.c.flock(h.fd, posix.LOCK.UN);
    }

    pub fn busy(h: *Holder, io: Io) bool {
        h.mutex.lockUncancelable(io);
        defer h.mutex.unlock(io);
        return h.count > 0;
    }

    /// Before a heavy model loads: waits (bounded) for speech in progress to finish.
    pub fn waitIdle(h: *Holder, io: Io) void {
        const start = now(io);
        while (h.busy(io) and now(io) - start < max_wait_ns) Io.sleep(io, .fromMilliseconds(poll_ns), .awake) catch return;
    }
};

/// A worker's side: waits at each progress report while the daemon makes speech.
pub const Waiter = struct {
    fd: posix.fd_t = -1,
    waited_ns: i96 = 0, // this job's

    pub fn init(path: []const u8) Waiter {
        return .{ .fd = open(path) };
    }

    /// A new job: its budget starts again.
    pub fn reset(w: *Waiter) void {
        w.waited_ns = 0;
    }

    fn free(w: *const Waiter) bool {
        if (std.c.flock(w.fd, posix.LOCK.SH | posix.LOCK.NB) != 0) return false;
        _ = std.c.flock(w.fd, posix.LOCK.UN);
        return true;
    }

    /// Returns at once while nobody speaks; otherwise once speech has been over for `linger_ns`, or the wait or the
    /// job's budget runs out.
    pub fn wait(w: *Waiter, io: Io) void {
        if (w.fd < 0 or w.waited_ns >= budget_ns or w.free()) return;
        const start = now(io);
        var quiet_since: i96 = 0;
        while (true) {
            Io.sleep(io, .fromMilliseconds(poll_ns), .awake) catch break;
            const t = now(io);
            if (t - start >= max_wait_ns or w.waited_ns + (t - start) >= budget_ns) break;
            if (!w.free()) {
                quiet_since = 0;
            } else if (quiet_since == 0) {
                quiet_since = t;
            } else if (t - quiet_since >= linger_ns) break;
        }
        w.waited_ns += now(io) - start;
    }
};

test "a worker waits while the daemon holds the gate, and not after" {
    const io = std.testing.io;
    var pbuf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "/tmp/localrouter-speech-gate-{d}", .{std.c.getpid()});
    defer Io.Dir.cwd().deleteFile(io, path) catch {};
    var h: Holder = .init(path);
    defer h.deinit();
    var w: Waiter = .init(path); // its own open file description, as a worker process has
    defer _ = std.c.close(w.fd);
    try std.testing.expect(h.fd >= 0 and w.fd >= 0);
    try std.testing.expect(w.free());
    h.enter(io);
    try std.testing.expect(!w.free());
    h.enter(io); // two speech jobs: free only when both end
    h.leave(io);
    try std.testing.expect(!w.free());
    h.leave(io);
    try std.testing.expect(w.free());
    const t0 = now(io);
    w.wait(io); // free: no wait
    try std.testing.expect(now(io) - t0 < 50 * std.time.ns_per_ms);
}
