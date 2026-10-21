//! The scheduler owns the GPU: one executor thread a lane runs one job at a time, loads and unloads tools inside the
//! memory budget, and unloads idle tools after their TTL. Other threads only queue work, cancel, and read state.
//!
//! Lanes: `main` (image, video), `chat` and `realtime` (speech and transcription by default, `config.laneOf`), so a
//! short TTS or STT job never waits behind a chat reply or a video, nor a chat turn behind a video. A tool is loaded and
//! run only by its own lane. Residency changes (admission, eviction, the reservation of a load, the GPU lock) are
//! serialized by `gate`; memory admitted but perhaps not yet allocated (a load in progress, the working bytes of a
//! running job) is held back from MemAvailable.
//!
//! Eviction: when a request needs room, idle tools are unloaded lowest `priority` first, least recently used among
//! equals, until it fits. Priority only orders the victims: a request is never refused while unloading idle tools
//! (whatever their priority) would make room; a loading or busy tool (on any lane) is never a victim. `keep_loaded`
//! tools are loaded at start when they fit, skip the idle TTL, and come back by themselves once their lane has been
//! idle for a moment: a kept tool that does not fit unloads idle tools of lower priority, as a request would (never one
//! of its own priority or above, so two kept tools cannot unload each other), unless it was unloaded through the API.
//!
//! The host's GPU lock (`Config.gpu_lock`) guards the machine against GPU users outside LocalRouter. The scheduler is
//! the one admission authority inside it: it holds the lock once, shared and without waiting, while any tool is loaded,
//! and releases it when none is. Co-residency is decided by the memory budget alone; a tool never takes the lock itself.

const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const tool_mod = @import("../tool/tool.zig");
const Tool = tool_mod.Tool;
const jobs = @import("jobs.zig");
const memory = @import("memory.zig");
const config = @import("../config.zig");
const Config = config.Config;

pub const Lane = config.Lane;

pub const Slot = struct {
    tool: Tool,
    resident: u64 = 0, // measured while loaded (the estimate while it loads)
    last_used_ns: i96 = 0,
    released: bool = false, // unloaded through the API: `keep_loaded` does not bring it back until it is loaded again
    visited: bool = false, // scratch of `loadKept`
    retry_ns: i96 = 0, // a failed `keep_loaded` load is not retried before this time
    in_use: bool = false, // its lane is loading or running it: never a victim (under `gate`)
    pending: u64 = 0, // admitted bytes that MemAvailable may not show yet (under `gate`)
    unload_after: bool = false, // unloaded on request while in use: its lane unloads it when done
};

/// How long a lane must be idle before a `keep_loaded` tool is reloaded, and how often it checks the memory.
const reload_grace_ns: i96 = 2 * std.time.ns_per_s;
const reload_poll_ns: i96 = 2 * std.time.ns_per_s;
const reload_backoff_ns: i96 = 60 * std.time.ns_per_s;

/// Control work from the API (load, unload, release). Loads run on the tool's lane, the rest on the main lane.
pub const Control = struct {
    op: enum { load, unload, release },
    tool: usize = 0,
    done: Io.Event = .unset,
    failed: bool = false,
    message: []const u8 = "",
};

const Item = union(enum) { job: *jobs.Job, control: *Control };

const LaneState = struct {
    queue: std.Deque(Item) = .empty,
    cond: Io.Condition = .init,
    running: ?*jobs.Job = null,
    idle_since_ns: i96 = 0, // when the executor last finished an item
};

const lane_count = 3;

fn index(l: Lane) usize {
    return switch (l) {
        .main => 0,
        .realtime => 1,
        .chat => 2,
    };
}

pub const Scheduler = struct {
    io: Io,
    gpa: std.mem.Allocator,
    cfg: *const Config,
    slots: []Slot,
    table: *jobs.Table,
    mutex: Io.Mutex = .init, // the queues and `running`
    gate: Io.Mutex = .init, // residency: admission, eviction, `in_use`, `pending`, `resident`, the GPU lock
    lanes: [lane_count]LaneState = @splat(.{}),
    stopping: bool = false,
    gpu_lock_fd: posix.fd_t = -1, // the host's GPU lock while any tool is loaded (`holdGpuLock`); under `gate`
    lock_msg: [512]u8 = undefined,
    lock_len: usize = 0,

    pub fn deinit(s: *Scheduler) void {
        s.releaseGpuLock();
        for (&s.lanes) |*l| l.queue.deinit(s.gpa);
    }

    /// Under `gate`: takes the host's GPU lock for LocalRouter (shared, without waiting) if it is configured and not
    /// yet held; false, with `lock_msg` set, when a GPU user outside LocalRouter holds it.
    fn holdGpuLock(s: *Scheduler) bool {
        const path = s.cfg.gpu_lock;
        if (path.len == 0 or s.gpu_lock_fd >= 0) return true;
        const fd = posix.openat(posix.AT.FDCWD, path, .{ .CLOEXEC = true }, 0) catch |err| {
            s.lockRefusal("the GPU lock {s} cannot be opened: {s}", .{ path, @errorName(err) });
            return false;
        };
        if (std.c.flock(fd, posix.LOCK.SH | posix.LOCK.NB) != 0) {
            _ = std.c.close(fd);
            s.lockRefusal("the GPU lock {s} is held by another job outside LocalRouter; nothing loads until it is free", .{path});
            return false;
        }
        s.gpu_lock_fd = fd;
        return true;
    }

    fn lockRefusal(s: *Scheduler, comptime fmt: []const u8, args: anytype) void {
        const msg = std.fmt.bufPrint(&s.lock_msg, fmt, args) catch {
            s.lock_len = 0;
            return;
        };
        s.lock_len = msg.len;
        std.log.warn("{s}", .{msg});
    }

    fn lockMessage(s: *Scheduler) []const u8 {
        return if (s.lock_len > 0) s.lock_msg[0..s.lock_len] else "the host's GPU lock is held by another job outside LocalRouter";
    }

    /// Under `gate`: releases the GPU lock once no tool is loaded, loading or in use, so GPU users outside LocalRouter may run.
    fn releaseIdleGpuLock(s: *Scheduler) void {
        if (s.gpu_lock_fd < 0) return;
        for (s.slots) |slot| if (slot.in_use or slot.tool.state() != .unloaded) return;
        s.releaseGpuLock();
    }

    fn releaseGpuLock(s: *Scheduler) void {
        if (s.gpu_lock_fd < 0) return;
        _ = std.c.close(s.gpu_lock_fd); // closing the descriptor releases the lock
        s.gpu_lock_fd = -1;
    }

    /// `releaseIdleGpuLock` under `gate`.
    fn releaseIdle(s: *Scheduler) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        s.releaseIdleGpuLock();
    }

    pub fn laneOfTool(s: *const Scheduler, i: usize) Lane {
        return config.laneOf(s.slots[i].tool.config());
    }

    /// Queues a job on its tool's lane; `error.QueueFull` past `max_queue` waiting jobs there.
    pub fn submit(s: *Scheduler, j: *jobs.Job) !void {
        try s.push(s.laneOfTool(j.tool), .{ .job = j });
    }

    /// Runs `c` on its lane and waits for it.
    pub fn control(s: *Scheduler, c: *Control) !void {
        try s.push(if (c.op == .load) s.laneOfTool(c.tool) else .main, .{ .control = c });
        try c.done.wait(s.io);
    }

    fn push(s: *Scheduler, lane: Lane, item: Item) !void {
        try s.mutex.lock(s.io);
        defer s.mutex.unlock(s.io);
        const l = &s.lanes[index(lane)];
        if (s.stopping) return error.ShuttingDown;
        if (l.queue.len >= s.cfg.max_queue) return error.QueueFull;
        try l.queue.pushBack(s.gpa, item);
        l.cond.signal(s.io);
    }

    /// Cancels a queued job at once, or aborts the running one (its worker is killed).
    pub fn cancel(s: *Scheduler, j: *jobs.Job) void {
        j.cancel.store(true, .release);
        s.mutex.lockUncancelable(s.io);
        const running = s.lanes[index(s.laneOfTool(j.tool))].running == j;
        s.mutex.unlock(s.io);
        if (running) s.slots[j.tool].tool.abort();
    }

    /// Cancels a queued job; a running one finishes and its result is ignored (the voice loop: no worker reload).
    pub fn drop(_: *Scheduler, j: *jobs.Job) void {
        j.cancel.store(true, .release);
    }

    pub fn stop(s: *Scheduler) void {
        s.mutex.lockUncancelable(s.io);
        s.stopping = true;
        for (&s.lanes) |*l| l.cond.signal(s.io);
        s.mutex.unlock(s.io);
    }

    /// The executor loop of `lane`; returns after `stop`, with the lane's tools unloaded.
    pub fn run(s: *Scheduler, lane: Lane) void {
        defer {
            s.gate.lockUncancelable(s.io);
            for (s.slots, 0..) |*slot, i| if (s.laneOfTool(i) == lane and !slot.in_use) s.unloadSlot(slot);
            s.releaseIdleGpuLock();
            s.gate.unlock(s.io);
        }
        s.loadKept(lane, false);
        const l = &s.lanes[index(lane)];
        while (true) {
            const item = s.next(lane) orelse return;
            switch (item) {
                .job => |j| s.runJob(j),
                .control => |c| s.runControl(c),
            }
            s.mutex.lockUncancelable(s.io);
            l.running = null;
            s.mutex.unlock(s.io);
            l.idle_since_ns = nowNs(s.io);
            s.releaseIdle();
        }
    }

    /// Waits for work; while idle, unloads the lane's tools whose TTL passed and expires old outputs (main lane).
    fn next(s: *Scheduler, lane: Lane) ?Item {
        const l = &s.lanes[index(lane)];
        while (true) {
            s.mutex.lockUncancelable(s.io);
            while (l.queue.popFront()) |item| {
                if (item == .job and item.job.cancel.load(.acquire)) {
                    s.finishCancelled(item.job);
                    continue;
                }
                if (item == .job) l.running = item.job;
                s.mutex.unlock(s.io);
                return item;
            }
            if (s.stopping) {
                s.mutex.unlock(s.io);
                return null;
            }
            const wait_ns = s.nextExpiryNs(lane);
            l.cond.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(wait_ns), .clock = .awake } }) catch {};
            s.mutex.unlock(s.io);
            s.unloadIdle(lane);
            s.loadKept(lane, true);
            s.releaseIdle();
            if (lane == .main) s.table.expire(s.io, nowUnix(s.io), s.cfg.keep_outputs_s);
        }
    }

    /// Nanoseconds until the earliest idle TTL of the lane's tools ends (capped at 60 s, the output expiry tick).
    fn nextExpiryNs(s: *Scheduler, lane: Lane) i96 {
        const now = nowNs(s.io);
        var wait: i96 = 60 * std.time.ns_per_s;
        for (s.slots, 0..) |slot, i| {
            if (s.laneOfTool(i) != lane) continue;
            const c = slot.tool.config();
            if (c.keep_loaded and !slot.released and slot.tool.state() == .unloaded) wait = @min(wait, reload_poll_ns);
            if (c.keep_loaded) continue;
            if (slot.resident == 0 and slot.tool.state() != .ready) continue;
            const end = slot.last_used_ns + @as(i96, c.idle_ttl_s) * std.time.ns_per_s;
            wait = @min(wait, @max(end - now, 1));
        }
        return wait;
    }

    fn unloadIdle(s: *Scheduler, lane: Lane) void {
        const now = nowNs(s.io);
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        for (s.slots, 0..) |*slot, i| {
            if (s.laneOfTool(i) != lane or slot.in_use) continue;
            if (slot.tool.state() != .ready or slot.tool.config().keep_loaded) continue;
            if (now - slot.last_used_ns >= @as(i96, slot.tool.config().idle_ttl_s) * std.time.ns_per_s) s.unloadSlot(slot);
        }
    }

    /// Loads the lane's `keep_loaded` tools that are down, highest priority first, each if it fits once idle tools of
    /// lower priority (on any lane) are unloaded. At start (`wait_idle` false) it logs the outcome; later it waits for
    /// `reload_grace_ns` of the lane's idleness.
    fn loadKept(s: *Scheduler, lane: Lane, wait_idle: bool) void {
        if (wait_idle and nowNs(s.io) - s.lanes[index(lane)].idle_since_ns < reload_grace_ns) return;
        for (s.slots, 0..) |*slot, i| if (s.laneOfTool(i) == lane) {
            slot.visited = false;
        };
        while (true) {
            s.gate.lockUncancelable(s.io);
            var best: ?usize = null;
            for (s.slots, 0..) |*slot, i| {
                const c = slot.tool.config();
                if (s.laneOfTool(i) != lane or !c.keep_loaded or slot.visited or slot.released or slot.tool.state() != .unloaded) continue;
                if (best == null or c.priority > s.slots[best.?].tool.config().priority) best = i;
            }
            const i = best orelse {
                s.gate.unlock(s.io);
                return;
            };
            const slot = &s.slots[i];
            slot.visited = true;
            const c = slot.tool.config();
            const estimate = slot.tool.needs(&placeholder(c.kind)).resident;
            if (nowNs(s.io) < slot.retry_ns or !s.admit(i, estimate, c.priority)) {
                s.gate.unlock(s.io);
                if (!wait_idle and nowNs(s.io) >= slot.retry_ns) std.log.warn("keep_loaded {s}: not loaded at start, it does not fit", .{c.id});
                continue;
            }
            s.reserve(slot, estimate);
            s.gate.unlock(s.io);
            s.loadSlot(slot) catch |err| {
                s.done(slot);
                std.log.warn("keep_loaded {s}: load failed: {s}", .{ c.id, if (err == error.GpuLockHeld) s.lockMessage() else slot.tool.failure().message });
                slot.retry_ns = nowNs(s.io) + reload_backoff_ns;
                continue;
            };
            s.done(slot);
            std.log.info("keep_loaded {s}: loaded ({d} MiB)", .{ c.id, slot.resident >> 20 });
        }
    }

    /// Marks `slot` in use with `bytes` admitted; under `gate`.
    fn reserve(_: *Scheduler, slot: *Slot, bytes: u64) void {
        slot.in_use = true;
        slot.pending = bytes;
    }

    /// The lane is finished with `slot`: it becomes a possible victim again, and an unload asked for meanwhile happens.
    fn done(s: *Scheduler, slot: *Slot) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        slot.in_use = false;
        slot.pending = 0;
        if (slot.resident > 0 or slot.tool.state() == .ready) slot.last_used_ns = nowNs(s.io);
        if (slot.unload_after) {
            slot.unload_after = false;
            s.unloadSlot(slot);
        }
        s.releaseIdleGpuLock(); // a failed load or an unload may leave nothing loaded
    }

    /// Under `gate`, or by the slot's own lane while it holds `in_use`. Every lane is woken: the tool's own lane may be
    /// asleep until a TTL far off and must notice now that a kept tool is down (another lane evicted it).
    fn unloadSlot(s: *Scheduler, slot: *Slot) void {
        if (slot.tool.state() == .unloaded and slot.resident == 0) return;
        slot.tool.unload();
        slot.resident = 0;
        s.mutex.lockUncancelable(s.io); // lock order: gate, then mutex (nothing takes them the other way)
        defer s.mutex.unlock(s.io);
        for (&s.lanes) |*l| l.cond.signal(s.io);
    }

    /// Unloads `slot` now, or when its lane is done with it.
    fn unloadOrDefer(s: *Scheduler, slot: *Slot) void {
        slot.released = true;
        if (slot.in_use) slot.unload_after = true else s.unloadSlot(slot);
    }

    fn runControl(s: *Scheduler, c: *Control) void {
        defer c.done.set(s.io);
        switch (c.op) {
            .unload => {
                s.gate.lockUncancelable(s.io);
                defer s.gate.unlock(s.io);
                s.unloadOrDefer(&s.slots[c.tool]);
                s.releaseIdleGpuLock(); // before `done`: the caller sees the lock free once nothing is loaded
            },
            .release => {
                s.gate.lockUncancelable(s.io);
                defer s.gate.unlock(s.io);
                for (s.slots) |*slot| s.unloadOrDefer(slot);
                s.releaseIdleGpuLock();
            },
            .load => {
                const slot = &s.slots[c.tool];
                if (slot.tool.state() == .ready) return;
                const n = slot.tool.needs(&placeholder(slot.tool.config().kind));
                s.gate.lockUncancelable(s.io);
                if (!s.admit(c.tool, n.resident, null)) {
                    s.gate.unlock(s.io);
                    c.failed = true;
                    c.message = "insufficient memory to load this tool";
                    return;
                }
                s.reserve(slot, n.resident);
                s.gate.unlock(s.io);
                defer s.done(slot);
                s.loadSlot(slot) catch |err| {
                    c.failed = true;
                    c.message = if (err == error.GpuLockHeld) s.lockMessage() else slot.tool.failure().message;
                };
            },
        }
    }

    /// Loads outside `gate` (the reservation holds the memory) while LocalRouter holds the host's GPU lock, then records
    /// the measured bytes; `error.GpuLockHeld` (why: `lockMessage`) when a GPU user outside LocalRouter holds it.
    fn loadSlot(s: *Scheduler, slot: *Slot) (tool_mod.Error || error{GpuLockHeld})!void {
        const estimate = slot.tool.needs(&placeholder(slot.tool.config().kind)).resident;
        s.gate.lockUncancelable(s.io);
        const locked = s.holdGpuLock();
        s.gate.unlock(s.io);
        if (!locked) return error.GpuLockHeld;
        const measured = try slot.tool.load();
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        slot.resident = if (measured > 0) measured else estimate;
        slot.pending -|= estimate; // now visible in MemAvailable
        slot.last_used_ns = nowNs(s.io);
        slot.released = false;
    }

    fn runJob(s: *Scheduler, j: *jobs.Job) void {
        const slot = &s.slots[j.tool];
        s.setStatus(j, .in_progress);
        const n = slot.tool.needs(&j.request);
        s.gate.lockUncancelable(s.io);
        const cold = slot.tool.state() != .ready;
        const extra = n.working + if (cold) n.resident else 0;
        if (!s.admit(j.tool, extra, null)) {
            s.gate.unlock(s.io);
            var buf: [160]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "insufficient memory: this request needs {d} MiB more than is free within the budget", .{extra >> 20}) catch "insufficient memory";
            return s.finish(j, .failed, 503, "insufficient_memory", msg, null);
        }
        s.reserve(slot, extra);
        s.gate.unlock(s.io);
        defer s.done(slot);
        if (cold) {
            s.setPhase(j, "load");
            s.loadSlot(slot) catch |err| {
                if (err == error.GpuLockHeld) return s.finish(j, .failed, 503, "server_error", s.lockMessage(), null);
                return s.finishError(j, slot, @errorCast(err));
            };
        }
        var arena: std.heap.ArenaAllocator = .init(s.gpa);
        defer arena.deinit();
        const job: tool_mod.Job = .{ .id = j.id, .dir = j.dir, .request = j.request };
        var pctx: ProgressCtx = .{ .s = s, .j = j };
        const out = slot.tool.generate(&job, .{ .ctx = &pctx, .progress = ProgressCtx.report }, arena.allocator()) catch |err| {
            if (slot.tool.state() == .unloaded) s.forget(slot);
            return s.finishError(j, slot, err);
        };
        if (slot.tool.config().idle_ttl_s == 0 and !slot.tool.config().keep_loaded) {
            s.gate.lockUncancelable(s.io);
            slot.unload_after = true;
            s.gate.unlock(s.io);
        }
        s.finish(j, .completed, 200, "", "", out);
    }

    /// The worker died or was killed: it holds no memory now.
    fn forget(s: *Scheduler, slot: *Slot) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        slot.resident = 0;
    }

    /// Under `gate`: true when `extra` more bytes fit; unloads idle tools, lowest priority then least recently used
    /// first, until they do. A tool in use by any lane is never a victim. With `below`, only tools of a lower priority
    /// than it are victims (a `keep_loaded` tool coming back), and nothing is unloaded unless unloading all of them
    /// would make room: a reload that cannot happen costs no one their model.
    fn admit(s: *Scheduler, target: usize, extra: u64, below: ?i32) bool {
        if (below) |p| {
            var freeable: u64 = 0;
            for (s.slots, 0..) |slot, i| {
                if (i != target and !slot.in_use and slot.tool.state() == .ready and slot.tool.config().priority < p) freeable += slot.resident;
            }
            if (!s.fits(extra -| freeable)) return false;
        }
        while (true) {
            if (s.fits(extra)) return true;
            var victim: ?*Slot = null;
            for (s.slots, 0..) |*slot, i| {
                if (i == target or slot.in_use or slot.tool.state() != .ready) continue;
                if (below) |p| if (slot.tool.config().priority >= p) continue;
                if (victim == null or evictsBefore(slot, victim.?)) victim = slot;
            }
            s.unloadSlot(victim orelse return false);
        }
    }

    fn evictsBefore(a: *const Slot, b: *const Slot) bool {
        const pa, const pb = .{ a.tool.config().priority, b.tool.config().priority };
        return if (pa != pb) pa < pb else a.last_used_ns < b.last_used_ns;
    }

    /// Under `gate`. Bytes other lanes admitted but may not have allocated yet are held back from MemAvailable.
    fn fits(s: *Scheduler, extra: u64) bool {
        var held: u64 = 0;
        var pending: u64 = 0;
        for (s.slots) |slot| {
            held += slot.resident;
            pending += slot.pending;
        }
        const avail = memory.available() orelse 0;
        if (avail < s.cfg.reserve_bytes or avail - s.cfg.reserve_bytes < extra +| pending) return false;
        if (s.cfg.budget_bytes == 0) return true;
        return held + pending + extra <= s.cfg.budget_bytes;
    }

    fn finishError(s: *Scheduler, j: *jobs.Job, slot: *Slot, err: tool_mod.Error) void {
        const f = slot.tool.failure();
        switch (err) {
            error.Cancelled => s.finish(j, .cancelled, 409, "cancelled", "the job was cancelled", null),
            error.Timeout => s.finish(j, .failed, 504, "timeout", f.message, null),
            error.GenerateFailed => if (std.mem.eql(u8, f.type, "invalid_request_error"))
                s.finish(j, .failed, 400, f.type, f.message, null)
            else if (std.mem.eql(u8, f.type, "service_unavailable")) // the model's own server is busy or not ready: retry later
                s.finish(j, .failed, 503, "server_error", f.message, null)
            else
                s.finish(j, .failed, 500, "server_error", f.message, null),
            error.OutOfMemory => s.finish(j, .failed, 503, "insufficient_memory", "out of memory", null),
            error.LoadFailed, error.WorkerDied => s.finish(j, .failed, 503, "server_error", f.message, null),
        }
    }

    fn finishCancelled(s: *Scheduler, j: *jobs.Job) void {
        s.table.mutex.lockUncancelable(s.io);
        j.status = .cancelled;
        j.http_status = 409;
        j.error_type = "cancelled";
        j.error_message = "the job was cancelled";
        j.completed_at = nowUnix(s.io);
        s.table.mutex.unlock(s.io);
        j.done.set(s.io);
    }

    fn finish(s: *Scheduler, j: *jobs.Job, status: jobs.Status, http: u16, typ: []const u8, msg: []const u8, out: ?tool_mod.Output) void {
        s.table.mutex.lockUncancelable(s.io);
        const a = j.arena.allocator();
        j.status = status;
        j.http_status = http;
        j.error_type = typ;
        j.error_message = a.dupe(u8, msg) catch "";
        j.completed_at = nowUnix(s.io);
        if (out) |o| {
            j.percent = 100;
            j.seed = o.seed;
            j.ms = o.ms;
            j.finish_reason = a.dupe(u8, o.finish_reason) catch "";
            j.segments = a.dupe(u8, o.segments) catch "";
            if (a.alloc([]const u8, o.files.len)) |files| {
                for (files, o.files) |*d, f| d.* = a.dupe(u8, f) catch "";
                j.files = files;
            } else |_| {}
        }
        s.table.mutex.unlock(s.io);
        j.done.set(s.io);
    }

    fn setStatus(s: *Scheduler, j: *jobs.Job, status: jobs.Status) void {
        s.table.mutex.lockUncancelable(s.io);
        j.status = status;
        s.table.mutex.unlock(s.io);
    }

    fn setPhase(s: *Scheduler, j: *jobs.Job, phase: []const u8) void {
        s.table.mutex.lockUncancelable(s.io);
        j.phase = @splat(0);
        @memcpy(j.phase[0..@min(phase.len, j.phase.len - 1)], phase[0..@min(phase.len, j.phase.len - 1)]);
        s.table.mutex.unlock(s.io);
    }
};

/// The Sink of a running job: phase and percent (at most 99 until it completes), under the table's lock.
const ProgressCtx = struct {
    s: *Scheduler,
    j: *jobs.Job,

    fn report(ctx: *anyopaque, p: tool_mod.Progress) void {
        const c: *ProgressCtx = @ptrCast(@alignCast(ctx));
        c.s.table.mutex.lockUncancelable(c.s.io);
        defer c.s.table.mutex.unlock(c.s.io);
        c.j.percent = if (p.of == 0) 0 else @intCast(@min(99, @as(u64, p.step) * 100 / p.of));
        const n = @min(p.phase.len, c.j.phase.len - 1);
        @memcpy(c.j.phase[0..n], p.phase[0..n]);
        c.j.phase[n] = 0;
    }
};

/// A minimal request of `kind`, for pricing a load on its own (`needs().resident` ignores the request's size).
fn placeholder(kind: @import("../tool/request.zig").Kind) @import("../tool/request.zig").Request {
    return switch (kind) {
        .image => .{ .image = .{ .prompt = "" } },
        .video => .{ .video = .{ .prompt = "" } },
        .speech => .{ .speech = .{ .input = "" } },
        .transcription => .{ .transcription = .{ .audio = "" } },
        .chat => .{ .chat = .{ .body = "" } },
    };
}

pub fn nowNs(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

pub fn nowUnix(io: Io) i64 {
    return @intCast(@divTrunc(Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_s));
}
