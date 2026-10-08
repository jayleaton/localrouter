//! The scheduler owns the GPU: one executor thread runs one job at a time, loads and unloads tools inside the
//! memory budget, and unloads idle tools after their TTL. Other threads only queue work, cancel, and read state.
//!
//! Eviction: when a request needs room, idle tools are unloaded lowest `priority` first, least recently used among
//! equals, until it fits. Priority only orders the victims: a request is never refused while unloading idle tools
//! (whatever their priority) would make room; a loading or busy tool is never a victim. `keep_loaded` tools are
//! loaded at start when they fit, skip the idle TTL, and come back by themselves (without evicting anything) once
//! the scheduler has been idle for a moment and they fit again, unless they were unloaded on request through the API.

const std = @import("std");
const Io = std.Io;
const tool_mod = @import("../tool/tool.zig");
const Tool = tool_mod.Tool;
const jobs = @import("jobs.zig");
const memory = @import("memory.zig");
const Config = @import("../config.zig").Config;

pub const Slot = struct {
    tool: Tool,
    resident: u64 = 0, // measured while loaded
    last_used_ns: i96 = 0,
    released: bool = false, // unloaded through the API: `keep_loaded` does not bring it back until it is loaded again
    visited: bool = false, // scratch of `loadKept`
    retry_ns: i96 = 0, // a failed `keep_loaded` load is not retried before this time
};

/// How long the scheduler must be idle before a `keep_loaded` tool is reloaded, and how often it checks the memory.
const reload_grace_ns: i96 = 2 * std.time.ns_per_s;
const reload_poll_ns: i96 = 2 * std.time.ns_per_s;
const reload_backoff_ns: i96 = 60 * std.time.ns_per_s;

/// Control work from the API (load, unload, release) runs on the executor too, so only it touches tools.
pub const Control = struct {
    op: enum { load, unload, release },
    tool: usize = 0,
    done: Io.Event = .unset,
    failed: bool = false,
    message: []const u8 = "",
};

const Item = union(enum) { job: *jobs.Job, control: *Control };

pub const Scheduler = struct {
    io: Io,
    gpa: std.mem.Allocator,
    cfg: *const Config,
    slots: []Slot,
    table: *jobs.Table,
    mutex: Io.Mutex = .init,
    cond: Io.Condition = .init,
    queue: std.Deque(Item) = .empty,
    running: ?*jobs.Job = null,
    stopping: bool = false,
    idle_since_ns: i96 = 0, // when the executor last finished an item

    pub fn deinit(s: *Scheduler) void {
        s.queue.deinit(s.gpa);
    }

    /// Queues a job; `error.QueueFull` past `max_queue` waiting jobs.
    pub fn submit(s: *Scheduler, j: *jobs.Job) !void {
        try s.push(.{ .job = j });
    }

    /// Runs `c` on the executor and waits for it.
    pub fn control(s: *Scheduler, c: *Control) !void {
        try s.push(.{ .control = c });
        try c.done.wait(s.io);
    }

    fn push(s: *Scheduler, item: Item) !void {
        try s.mutex.lock(s.io);
        defer s.mutex.unlock(s.io);
        if (s.stopping) return error.ShuttingDown;
        if (s.queue.len >= s.cfg.max_queue) return error.QueueFull;
        try s.queue.pushBack(s.gpa, item);
        s.cond.signal(s.io);
    }

    /// Cancels a queued job at once, or aborts the running one (its worker is killed).
    pub fn cancel(s: *Scheduler, j: *jobs.Job) void {
        j.cancel.store(true, .release);
        s.mutex.lockUncancelable(s.io);
        const running = s.running == j;
        s.mutex.unlock(s.io);
        if (running) s.slots[j.tool].tool.abort();
    }

    pub fn stop(s: *Scheduler) void {
        s.mutex.lockUncancelable(s.io);
        s.stopping = true;
        s.cond.signal(s.io);
        s.mutex.unlock(s.io);
    }

    /// The executor loop; returns after `stop`, with every tool unloaded.
    pub fn run(s: *Scheduler) void {
        defer for (s.slots) |*slot| s.unloadSlot(slot);
        s.loadKept(false);
        while (true) {
            const item = s.next() orelse return;
            switch (item) {
                .job => |j| s.runJob(j),
                .control => |c| s.runControl(c),
            }
            s.mutex.lockUncancelable(s.io);
            s.running = null;
            s.mutex.unlock(s.io);
            s.idle_since_ns = nowNs(s.io);
        }
    }

    /// Waits for work; while idle, unloads tools whose TTL passed and expires old outputs.
    fn next(s: *Scheduler) ?Item {
        while (true) {
            s.mutex.lockUncancelable(s.io);
            while (s.queue.popFront()) |item| {
                if (item == .job and item.job.cancel.load(.acquire)) {
                    s.finishCancelled(item.job);
                    continue;
                }
                if (item == .job) s.running = item.job;
                s.mutex.unlock(s.io);
                return item;
            }
            if (s.stopping) {
                s.mutex.unlock(s.io);
                return null;
            }
            const wait_ns = s.nextExpiryNs();
            s.cond.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(wait_ns), .clock = .awake } }) catch {};
            s.mutex.unlock(s.io);
            s.unloadIdle();
            s.loadKept(true);
            s.table.expire(s.io, nowUnix(s.io), s.cfg.keep_outputs_s);
        }
    }

    /// Nanoseconds until the earliest idle TTL ends (capped at 60 s, the output expiry tick).
    fn nextExpiryNs(s: *Scheduler) i96 {
        const now = nowNs(s.io);
        var wait: i96 = 60 * std.time.ns_per_s;
        for (s.slots) |slot| {
            const c = slot.tool.config();
            if (c.keep_loaded and !slot.released and slot.tool.state() == .unloaded) wait = @min(wait, reload_poll_ns);
            if (c.keep_loaded) continue;
            if (slot.resident == 0 and slot.tool.state() != .ready) continue;
            const end = slot.last_used_ns + @as(i96, slot.tool.config().idle_ttl_s) * std.time.ns_per_s;
            wait = @min(wait, @max(end - now, 1));
        }
        return wait;
    }

    fn unloadIdle(s: *Scheduler) void {
        const now = nowNs(s.io);
        for (s.slots) |*slot| {
            if (slot.tool.state() != .ready or slot.tool.config().keep_loaded) continue;
            if (now - slot.last_used_ns >= @as(i96, slot.tool.config().idle_ttl_s) * std.time.ns_per_s) s.unloadSlot(slot);
        }
    }

    /// Loads the `keep_loaded` tools that are down, highest priority first, each only if it fits without unloading
    /// anything. At start (`wait_idle` false) it logs the outcome; later it waits for `reload_grace_ns` of idleness.
    fn loadKept(s: *Scheduler, wait_idle: bool) void {
        if (wait_idle and nowNs(s.io) - s.idle_since_ns < reload_grace_ns) return;
        for (s.slots) |*slot| slot.visited = false;
        while (true) {
            var best: ?*Slot = null;
            for (s.slots) |*slot| {
                const c = slot.tool.config();
                if (!c.keep_loaded or slot.visited or slot.released or slot.tool.state() != .unloaded) continue;
                if (best == null or c.priority > best.?.tool.config().priority) best = slot;
            }
            const slot = best orelse return;
            slot.visited = true;
            const c = slot.tool.config();
            if (nowNs(s.io) < slot.retry_ns) continue;
            if (!s.fits(slot.tool.needs(&placeholder(c.kind)).resident)) {
                if (!wait_idle) std.log.warn("keep_loaded {s}: not loaded at start, it does not fit", .{c.id});
                continue;
            }
            s.loadSlot(slot) catch {
                std.log.warn("keep_loaded {s}: load failed: {s}", .{ c.id, slot.tool.failure().message });
                slot.retry_ns = nowNs(s.io) + reload_backoff_ns;
                continue;
            };
            std.log.info("keep_loaded {s}: loaded ({d} MiB)", .{ c.id, slot.resident >> 20 });
        }
    }

    fn unloadSlot(s: *Scheduler, slot: *Slot) void {
        _ = s;
        if (slot.tool.state() == .unloaded and slot.resident == 0) return;
        slot.tool.unload();
        slot.resident = 0;
    }

    fn runControl(s: *Scheduler, c: *Control) void {
        defer c.done.set(s.io);
        switch (c.op) {
            .unload => {
                s.slots[c.tool].released = true;
                s.unloadSlot(&s.slots[c.tool]);
            },
            .release => for (s.slots) |*slot| {
                slot.released = true;
                s.unloadSlot(slot);
            },
            .load => {
                const slot = &s.slots[c.tool];
                if (slot.tool.state() == .ready) return;
                const n = slot.tool.needs(&placeholder(slot.tool.config().kind));
                if (!s.admit(c.tool, n.resident)) {
                    c.failed = true;
                    c.message = "insufficient memory to load this tool";
                    return;
                }
                s.loadSlot(slot) catch {
                    c.failed = true;
                    c.message = slot.tool.failure().message;
                };
            },
        }
    }

    fn loadSlot(s: *Scheduler, slot: *Slot) !void {
        const measured = try slot.tool.load();
        const estimate = slot.tool.needs(&placeholder(slot.tool.config().kind)).resident;
        slot.resident = if (measured > 0) measured else estimate;
        slot.last_used_ns = nowNs(s.io);
        slot.released = false;
    }

    fn runJob(s: *Scheduler, j: *jobs.Job) void {
        const slot = &s.slots[j.tool];
        s.setStatus(j, .in_progress);
        const n = slot.tool.needs(&j.request);
        const extra = n.working + if (slot.tool.state() == .ready) 0 else n.resident;
        if (!s.admit(j.tool, extra)) {
            var buf: [160]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "insufficient memory: this request needs {d} MiB more than is free within the budget", .{extra >> 20}) catch "insufficient memory";
            return s.finish(j, .failed, 503, "insufficient_memory", msg, null);
        }
        if (slot.tool.state() != .ready) {
            s.setPhase(j, "load");
            s.loadSlot(slot) catch |err| return s.finishError(j, slot, err);
        }
        var arena: std.heap.ArenaAllocator = .init(s.gpa);
        defer arena.deinit();
        const job: tool_mod.Job = .{ .id = j.id, .dir = j.dir, .request = j.request };
        var pctx: ProgressCtx = .{ .s = s, .j = j };
        const out = slot.tool.generate(&job, .{ .ctx = &pctx, .progress = ProgressCtx.report }, arena.allocator()) catch |err| {
            slot.last_used_ns = nowNs(s.io);
            if (slot.tool.state() == .unloaded) slot.resident = 0;
            return s.finishError(j, slot, err);
        };
        slot.last_used_ns = nowNs(s.io);
        s.finish(j, .completed, 200, "", "", out);
        if (slot.tool.config().idle_ttl_s == 0 and !slot.tool.config().keep_loaded) s.unloadSlot(slot);
    }

    /// True when `extra` more bytes fit; unloads idle tools, lowest priority then least recently used first, until they do.
    fn admit(s: *Scheduler, target: usize, extra: u64) bool {
        while (true) {
            if (s.fits(extra)) return true;
            var victim: ?*Slot = null;
            for (s.slots, 0..) |*slot, i| {
                if (i == target or slot.tool.state() != .ready) continue;
                if (victim == null or evictsBefore(slot, victim.?)) victim = slot;
            }
            s.unloadSlot(victim orelse return false);
        }
    }

    fn evictsBefore(a: *const Slot, b: *const Slot) bool {
        const pa, const pb = .{ a.tool.config().priority, b.tool.config().priority };
        return if (pa != pb) pa < pb else a.last_used_ns < b.last_used_ns;
    }

    fn fits(s: *Scheduler, extra: u64) bool {
        const avail = memory.available() orelse 0;
        if (avail < s.cfg.reserve_bytes or avail - s.cfg.reserve_bytes < extra) return false;
        if (s.cfg.budget_bytes == 0) return true;
        var held: u64 = 0;
        for (s.slots) |slot| held += slot.resident;
        return held + extra <= s.cfg.budget_bytes;
    }

    fn finishError(s: *Scheduler, j: *jobs.Job, slot: *Slot, err: tool_mod.Error) void {
        const f = slot.tool.failure();
        switch (err) {
            error.Cancelled => s.finish(j, .cancelled, 409, "cancelled", "the job was cancelled", null),
            error.Timeout => s.finish(j, .failed, 504, "timeout", f.message, null),
            error.GenerateFailed => if (std.mem.eql(u8, f.type, "invalid_request_error"))
                s.finish(j, .failed, 400, f.type, f.message, null)
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
    };
}

pub fn nowNs(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

pub fn nowUnix(io: Io) i64 {
    return @intCast(@divTrunc(Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_s));
}
