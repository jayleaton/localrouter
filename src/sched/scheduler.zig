//! The scheduler owns the GPU: one executor thread a lane runs one job at a time, loads and unloads tools inside the
//! memory budget, and unloads idle tools after their TTL. Other threads only queue work, cancel, and read state.
//!
//! Lanes: `main` (image, video), `chat`, `realtime` (speech) and `listen` (transcription; `config.laneOf`), so a
//! short TTS or STT job never waits behind a chat reply or a video, nor a chat turn behind a video. A tool is loaded and
//! run only by its own lane. Residency changes (admission, eviction, the reservation of a load, the GPU lock) are
//! serialized by `gate`; memory admitted but perhaps not yet allocated (a load in progress, the working bytes of a
//! running job) is held back from MemAvailable, less what MemAvailable has dropped since it was admitted (that part has
//! landed and is already counted there).
//!
//! Eviction: when a request needs room, idle tools are unloaded lowest `priority` first, least recently used among
//! equals, until it fits. Priority only orders the victims: a request is never refused while unloading idle tools
//! (whatever their priority) would make room; a loading or busy tool (on any lane) is never a victim. `keep_loaded`
//! tools are loaded at start when they fit, skip the idle TTL, and come back by themselves once their lane has been
//! idle for a moment: a kept tool that does not fit unloads idle tools of lower priority, as a request would (never one
//! of its own priority or above, so two kept tools cannot unload each other), unless it was unloaded through the API.
//! Kept tools come back highest priority first across lanes: one waits while a higher one is due to load, and comes back
//! only with room to spare for the working bytes of the kept tools above it (else their next job evicts it again).
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
const speech_gate = @import("speech_gate.zig");
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
    backoff_ns: i96 = reload_backoff_min_ns, // the next wait after a failed load: doubles to the maximum, reset by a load
    in_use: bool = false, // its lane is loading or running it: never a victim (under `gate`)
    pending: u64 = 0, // admitted bytes that MemAvailable may not show yet (under `gate`)
    pending_ns: i96 = 0, // when `pending` was admitted (under `gate`)
    pending_avail: u64 = 0, // MemAvailable then: what has landed since shows as its drop, not counted twice
    unfit: bool = false, // a `keep_loaded` reload last found no room: lower priorities stop deferring to it
    unload_after: bool = false, // unloaded on request while in use: its lane unloads it when done
    displaced: bool = false, // unloaded to make room (reserve_for, unload_model, or an eviction of a kept tool): it comes
    // back by itself once `hold_until_ns` has passed and it fits again; a load or a release clears it (under `gate`)
    held: bool = false, // the target of a reservation: kept reloads do not evict it until its next job is done
    pausing: bool = false, // a voice model displaced for a model of another kind: voice sessions wait for it (under `gate`)
};

/// Told before a voice model (speech or transcription) is unloaded to make room, and when every one of them is back,
/// so open voice sessions can say so and wait rather than fail. Called under `gate`.
pub const Notify = struct {
    ctx: *anyopaque,
    paused: *const fn (ctx: *anyopaque, model: []const u8, for_model: []const u8) void,
    resumed: *const fn (ctx: *anyopaque) void,
};

/// The most models one reservation unloads (a box has a handful).
pub const max_victims = 16;

/// How long a lane must be idle before a `keep_loaded` tool is reloaded, and how often it checks the memory.
const reload_grace_ns: i96 = 2 * std.time.ns_per_s;
const reload_poll_ns: i96 = 2 * std.time.ns_per_s;
const reload_backoff_min_ns: i96 = 5 * std.time.ns_per_s; // a model refused by its own launcher (memory held elsewhere)
const reload_backoff_max_ns: i96 = 60 * std.time.ns_per_s; // so it comes back within seconds once the memory frees

/// Control work from the API (load, unload, release; displace and reserve for the cooperative model tools). Loads and
/// reservations run on the tool's lane, the rest on the main lane.
pub const Control = struct {
    op: enum { load, unload, release, displace, reserve },
    tool: usize = 0,
    request: ?Request = null, // reserve: the job to make room for (null: the tool's default request)
    hold_s: u32 = 120, // displace, reserve: the unloaded tools come back after this at the latest
    done: Io.Event = .unset,
    failed: bool = false,
    message: []const u8 = "",
    plan: Plan = .{}, // reserve: what was unloaded
};

/// What a reservation unloads (or would): slot indices, the bytes the target needs and what was free.
pub const Plan = struct {
    victims: [max_victims]usize = undefined,
    count: usize = 0,
    needed: u64 = 0, // bytes the target needs beyond what it holds (its load if cold, and a job's working bytes)
    available: u64 = 0, // MemAvailable less the reserve, before anything is unloaded
    fits: bool = false, // false: not even unloading every idle model but the chat models makes room

    pub fn list(p: *const Plan) []const usize {
        return p.victims[0..p.count];
    }
};

const Request = @import("../tool/request.zig").Request;

const Item = union(enum) { job: *jobs.Job, control: *Control };

const LaneState = struct {
    queue: std.Deque(Item) = .empty,
    cond: Io.Condition = .init,
    running: ?*jobs.Job = null,
    idle_since_ns: i96 = 0, // when the executor last finished an item
};

const lane_count = 4;

fn index(l: Lane) usize {
    return switch (l) {
        .main => 0,
        .realtime => 1,
        .chat => 2,
        .listen => 3,
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
    speech: speech_gate.Holder = .{}, // held while a speech job runs: image and video workers pause between steps
    lock_msg: [512]u8 = undefined,
    lock_len: usize = 0,
    hold_until_ns: i96 = 0, // displaced tools wait until this (a reservation's job ends it early); under `gate`
    notify: ?Notify = null, // voice sessions, told of voice models leaving and coming back
    voice_paused: std.atomic.Value(bool) = .init(false), // a voice model is displaced: sessions wait for it

    /// True while a speech or transcription model is unloaded to make room; voice sessions wait instead of failing.
    pub fn voicePaused(s: *const Scheduler) bool {
        return s.voice_paused.load(.acquire);
    }

    /// Under `gate`: `slot` is about to be unloaded to make room for slot `for_tool`. A voice model's sessions are told
    /// first; the slot comes back by itself later.
    fn displace(s: *Scheduler, slot: *Slot, for_tool: ?usize) void {
        slot.displaced = true;
        if (!isVoice(slot)) return;
        if (for_tool) |t| if (isVoice(&s.slots[t])) return; // one voice model making room for another: no pause
        slot.pausing = true;
        s.voice_paused.store(true, .release);
        if (s.notify) |n| n.paused(n.ctx, slot.tool.config().id, if (for_tool) |t| s.slots[t].tool.config().id else "");
    }

    fn isVoice(slot: *const Slot) bool {
        const k = slot.tool.config().kind;
        return k == .speech or k == .transcription;
    }

    /// Under `gate`: once no voice model that paused voice is down, sessions are told voice is back.
    fn checkResumed(s: *Scheduler) void {
        if (!s.voice_paused.load(.acquire)) return;
        for (s.slots) |slot| if (slot.pausing and slot.tool.state() != .ready) return;
        s.voice_paused.store(false, .release);
        std.log.info("voice models are back", .{});
        if (s.notify) |n| n.resumed(n.ctx);
    }

    /// What reserving room for `req` on slot `target` would unload: idle models other than chat models, lowest priority
    /// then least recently used first, until it fits. Nothing is unloaded here. Under `gate`.
    fn planFor(s: *Scheduler, target: usize, req: *const Request) Plan {
        var p: Plan = .{};
        const slot = &s.slots[target];
        const n = slot.tool.needs(req);
        p.needed = n.working + if (slot.tool.state() == .ready) 0 else n.resident;
        p.available = (memory.available() orelse 0) -| s.cfg.reserve_bytes;
        for (s.slots) |*x| x.visited = false;
        var freed: u64 = 0;
        while (!s.fits(p.needed -| freed)) {
            var victim: ?usize = null;
            for (s.slots, 0..) |*x, i| {
                if (i == target or x.in_use or x.visited or x.tool.state() != .ready or x.tool.config().kind == .chat) continue;
                if (victim == null or evictsBefore(x, &s.slots[victim.?])) victim = i;
            }
            const v = victim orelse return p; // fits stays false
            if (p.count == max_victims) return p;
            s.slots[v].visited = true;
            p.victims[p.count] = v;
            p.count += 1;
            freed += s.slots[v].resident;
        }
        p.fits = true;
        return p;
    }

    /// A dry run of a reservation: what it would unload, from any thread.
    pub fn plan(s: *Scheduler, target: usize, req: ?Request) Plan {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const r = req orelse placeholder(s.slots[target].tool.config().kind);
        return s.planFor(target, &r);
    }

    pub fn deinit(s: *Scheduler) void {
        s.releaseGpuLock();
        s.speech.deinit();
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
        try s.push(if (c.op == .load or c.op == .reserve) s.laneOfTool(c.tool) else .main, .{ .control = c });
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

    /// Cancels a queued job at once, or ends the running one: early with the tool kept loaded when it offers that
    /// (`Tool.cancel`), else by aborting it (its worker is killed).
    pub fn cancel(s: *Scheduler, j: *jobs.Job) void {
        j.cancel.store(true, .release);
        if (s.isRunning(j)) s.end(j, true);
    }

    /// Cancels a queued job, and ends a running one early only if its tool can stay loaded; otherwise the running job
    /// finishes and its result is ignored (the voice loop: never a reload).
    pub fn drop(s: *Scheduler, j: *jobs.Job) void {
        j.cancel.store(true, .release);
        if (s.isRunning(j)) s.end(j, false);
    }

    fn isRunning(s: *Scheduler, j: *const jobs.Job) bool {
        return s.runningOn(s.laneOfTool(j.tool), j);
    }

    /// Whether `j` runs on `lane`; compares the pointer only, so it is safe after the job was freed.
    fn runningOn(s: *Scheduler, lane: Lane, j: *const jobs.Job) bool {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        return s.lanes[index(lane)].running == j;
    }

    /// How long a soft cancel may take before a hard one (`cancel`) follows.
    const soft_cancel_ns: i96 = 5 * std.time.ns_per_s;

    fn end(s: *Scheduler, j: *jobs.Job, hard: bool) void {
        const t = s.slots[j.tool].tool;
        if (t.cancel()) {
            if (!hard) return;
            const watchdog = std.Thread.spawn(.{}, abortLate, .{ s, j.tool, @as(*const jobs.Job, j) }) catch return t.abort();
            watchdog.detach();
        } else if (hard) t.abort();
    }

    /// The soft cancel did not end the job in time: abort its tool (the job itself is never read again).
    fn abortLate(s: *Scheduler, tool: usize, j: *const jobs.Job) void {
        const lane = s.laneOfTool(tool);
        const until = nowNs(s.io) + soft_cancel_ns;
        while (nowNs(s.io) < until) {
            if (!s.runningOn(lane, j)) return;
            Io.sleep(s.io, .fromMilliseconds(50), .awake) catch {};
        }
        if (s.runningOn(lane, j)) s.slots[tool].tool.abort();
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
            if (((c.keep_loaded and !slot.released) or slot.displaced) and slot.tool.state() == .unloaded) wait = @min(wait, reload_poll_ns);
            if (c.keep_loaded) continue;
            if (slot.resident == 0 and slot.tool.state() != .ready) continue;
            const ttl_end = slot.last_used_ns + @as(i96, c.idle_ttl_s) * std.time.ns_per_s;
            wait = @min(wait, @max(ttl_end - now, 1));
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
                if (s.laneOfTool(i) != lane or slot.visited or slot.tool.state() != .unloaded) continue;
                if (!((c.keep_loaded and !slot.released) or slot.displaced)) continue;
                if (slot.displaced and nowNs(s.io) < s.hold_until_ns) continue; // the job it made room for has not run yet
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
            if (s.higherKeptDue(i)) { // it takes its room first; this one comes back on a later poll
                s.gate.unlock(s.io);
                continue;
            }
            if (nowNs(s.io) < slot.retry_ns or !s.admit(i, estimate +| s.keptHeadroom(i), c.priority)) {
                if (nowNs(s.io) >= slot.retry_ns) slot.unfit = true;
                s.gate.unlock(s.io);
                if (!wait_idle and nowNs(s.io) >= slot.retry_ns) std.log.warn("keep_loaded {s}: not loaded at start, it does not fit", .{c.id});
                continue;
            }
            s.reserve(slot, estimate);
            s.gate.unlock(s.io);
            s.loadSlot(slot) catch |err| {
                s.done(slot);
                std.log.warn("keep_loaded {s}: load failed: {s}", .{ c.id, if (err == error.GpuLockHeld) s.lockMessage() else slot.tool.failure().message });
                slot.retry_ns = nowNs(s.io) + slot.backoff_ns;
                slot.backoff_ns = @min(slot.backoff_ns * 2, reload_backoff_max_ns);
                continue;
            };
            slot.backoff_ns = reload_backoff_min_ns;
            s.done(slot);
            std.log.info("{s} {s}: loaded ({d} MiB)", .{ if (c.keep_loaded) "keep_loaded" else "restored", c.id, slot.resident >> 20 });
            s.gate.lockUncancelable(s.io);
            s.checkResumed();
            s.gate.unlock(s.io);
        }
    }

    /// Under `gate`: a `keep_loaded` tool of a higher priority than slot `i` is down and due to load (not released, not
    /// backing off, not just found not to fit). Kept tools come back highest priority first across lanes, so a lower
    /// one never takes the room a higher one is about to need (DGX-01: the image model loaded before chat at a restart,
    /// and chat then unloaded Drama 3 and Whisper to fit).
    fn higherKeptDue(s: *Scheduler, i: usize) bool {
        const p = s.slots[i].tool.config().priority;
        const now = nowNs(s.io);
        for (s.slots, 0..) |*slot, j| {
            const c = slot.tool.config();
            if (j == i or !c.keep_loaded or c.priority <= p or slot.released or slot.unfit or now < slot.retry_ns) continue;
            const st = slot.tool.state();
            if (st == .unloaded or st == .loading) return true;
        }
        return false;
    }

    /// The working bytes of the loaded `keep_loaded` tools of a higher priority than slot `i`: a kept tool comes back only
    /// with that much to spare, or their next job unloads it again (DGX-01: every voice turn's transcription evicted the
    /// kept image model, which reloaded as soon as the turn ended).
    fn keptHeadroom(s: *Scheduler, i: usize) u64 {
        const p = s.slots[i].tool.config().priority;
        var bytes: u64 = 0;
        for (s.slots, 0..) |*slot, j| {
            const c = slot.tool.config();
            if (j == i or !c.keep_loaded or c.priority <= p or slot.tool.state() == .unloaded) continue;
            bytes +|= slot.tool.needs(&placeholder(c.kind)).working;
        }
        return bytes;
    }

    /// Marks `slot` in use with `bytes` admitted; under `gate`.
    fn reserve(s: *Scheduler, slot: *Slot, bytes: u64) void {
        slot.in_use = true;
        slot.pending = bytes;
        slot.pending_ns = nowNs(s.io);
        slot.pending_avail = memory.available() orelse 0;
        slot.unfit = false;
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
                for (s.slots) |*slot| {
                    slot.displaced = false;
                    slot.pausing = false;
                    slot.held = false;
                    s.unloadOrDefer(slot);
                }
                s.voice_paused.store(false, .release); // the operator unloaded everything: nothing is coming back
                s.releaseIdleGpuLock();
            },
            .displace => if (s.displaceNow(c.tool, c.hold_s)) |why| {
                c.failed = true;
                c.message = why;
            },
            .reserve => s.reserveRoom(c),
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

    /// unload_model, from any thread: unloads tool `i` now (or when its lane is done with it) to make room, and brings it
    /// back by itself `hold_s` from now; a voice model's sessions are told first. A chat model is refused: it is the
    /// one answering. Null when done, else why not.
    pub fn displaceNow(s: *Scheduler, i: usize, hold_s: u32) ?[]const u8 {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const slot = &s.slots[i];
        const c = slot.tool.config();
        if (c.kind == .chat) return "a chat model is not unloaded through the model tools: it is the one answering";
        std.log.info("unload_model {s}: unloaded on request, back within {d} s", .{ c.id, hold_s });
        s.displace(slot, null);
        s.hold_until_ns = @max(s.hold_until_ns, nowNs(s.io) + @as(i96, hold_s) * std.time.ns_per_s);
        if (slot.in_use) slot.unload_after = true else s.unloadSlot(slot);
        s.releaseIdleGpuLock();
        return null;
    }

    /// What tool `i` needs for its default request: resident bytes, and a job's working bytes.
    pub fn defaultNeeds(s: *Scheduler, i: usize) tool_mod.Needs {
        const r = placeholder(s.slots[i].tool.config().kind);
        return s.slots[i].tool.needs(&r);
    }

    /// From any thread: whether tool `i` is waiting to come back (displaced) or held for a reservation's job.
    pub fn residency(s: *Scheduler, i: usize) struct { displaced: bool, held: bool } {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        return .{ .displaced = s.slots[i].displaced, .held = s.slots[i].held };
    }

    /// On the target's lane: unloads what `planFor` names (voice sessions told first), loads the target and holds it
    /// until its next job is done or `hold_s` passes; then the unloaded models come back by themselves.
    fn reserveRoom(s: *Scheduler, c: *Control) void {
        const slot = &s.slots[c.tool];
        const req = c.request orelse placeholder(slot.tool.config().kind);
        s.gate.lockUncancelable(s.io);
        c.plan = s.planFor(c.tool, &req);
        if (!c.plan.fits) {
            s.gate.unlock(s.io);
            c.failed = true;
            c.message = "not enough memory even after unloading every idle model but the chat models";
            return;
        }
        for (c.plan.list()) |v| {
            const victim = &s.slots[v];
            std.log.info("reserve_for {s}: unloading {s} (priority {d}, {d} MiB); it comes back after the job", .{ slot.tool.config().id, victim.tool.config().id, victim.tool.config().priority, victim.resident >> 20 });
            s.displace(victim, c.tool);
            s.unloadSlot(victim);
        }
        s.hold_until_ns = @max(s.hold_until_ns, nowNs(s.io) + @as(i96, c.hold_s) * std.time.ns_per_s);
        slot.held = true;
        if (slot.tool.state() == .ready) {
            s.gate.unlock(s.io);
            return;
        }
        const n = slot.tool.needs(&req);
        s.reserve(slot, n.resident);
        s.gate.unlock(s.io);
        defer s.done(slot);
        s.loadSlot(slot) catch |err| {
            c.failed = true;
            c.message = if (err == error.GpuLockHeld) s.lockMessage() else slot.tool.failure().message;
            s.endHold(slot); // nothing to wait for: what was unloaded comes back
        };
    }

    /// The reservation of `slot` is over (its job ran, or its load failed): displaced models may come back now.
    fn endHold(s: *Scheduler, slot: *Slot) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        if (!slot.held) return;
        slot.held = false;
        s.hold_until_ns = 0;
        s.mutex.lockUncancelable(s.io); // wake every lane: their displaced tools may load now
        defer s.mutex.unlock(s.io);
        for (&s.lanes) |*l| l.cond.signal(s.io);
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
        slot.pending_ns = nowNs(s.io); // what is left (a job's working bytes) lands from here
        slot.pending_avail = memory.available() orelse 0;
        slot.last_used_ns = nowNs(s.io);
        slot.released = false;
        slot.displaced = false;
        slot.pausing = false;
        s.checkResumed();
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
        defer s.endHold(slot); // after `done`: a reservation ends with its job, and what it unloaded comes back
        defer s.done(slot);
        const heavy = j.request == .image or j.request == .video;
        if (cold and heavy) s.speech.waitIdle(s.io); // a load next to speech starves it too
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
        const speaking = j.request == .speech;
        if (speaking) s.speech.enter(s.io);
        defer if (speaking) s.speech.leave(s.io);
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
                if (i != target and !slot.in_use and !slot.held and slot.tool.state() == .ready and slot.tool.config().priority < p) freeable += slot.resident;
            }
            if (!s.fits(extra -| freeable)) return false;
        }
        while (true) {
            if (s.fits(extra)) return true;
            var victim: ?*Slot = null;
            for (s.slots, 0..) |*slot, i| {
                if (i == target or slot.in_use or slot.tool.state() != .ready) continue;
                if (below) |p| if (slot.tool.config().priority >= p or slot.held) continue; // a reservation's target waits for its job
                if (victim == null or evictsBefore(slot, victim.?)) victim = slot;
            }
            const v = victim orelse return false;
            if (v.tool.config().keep_loaded and !v.released) s.displace(v, target); // it comes back after this job
            // the journal says what evicted what: on a box that keeps models warm an unload is never silent
            std.log.info("evicting {s} (priority {d}) for {s}: it needs {d} MiB more, {d} MiB available of which {d} MiB is the reserve", .{
                v.tool.config().id,
                v.tool.config().priority,
                s.slots[target].tool.config().id,
                extra >> 20,
                (memory.available() orelse 0) >> 20,
                s.cfg.reserve_bytes >> 20,
            });
            s.unloadSlot(v);
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
        var first_ns: ?i96 = null;
        var first_avail: u64 = 0;
        for (s.slots) |slot| {
            held += slot.resident;
            if (slot.pending == 0) continue;
            pending += slot.pending;
            if (first_ns == null or slot.pending_ns < first_ns.?) {
                first_ns = slot.pending_ns;
                first_avail = slot.pending_avail;
            }
        }
        const avail = memory.available() orelse 0;
        // A load in progress already holds part of what it was admitted for, and MemAvailable shows that part: only the
        // rest is still to come. Counting all of it again evicted ready models (DGX-01: a 2 GiB transcription unloaded
        // Drama 3 with 36 GiB free while the image model reloaded).
        const due = pending -| (first_avail -| avail);
        if (avail < s.cfg.reserve_bytes or avail - s.cfg.reserve_bytes < extra +| due) return false;
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
