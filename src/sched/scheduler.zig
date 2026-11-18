//! The scheduler owns the GPU: one executor thread a lane runs one job at a time, loads and unloads tools inside the
//! memory budget, and unloads idle tools after their TTL. Other threads only queue work, cancel, and read state.
//!
//! Lanes: `main` (image, video), `chat`, `realtime` (speech), `listen` (transcription) and `decide` (typed decisions;
//! `config.laneOf`), so a
//! short TTS or STT job never waits behind a chat reply or a video, nor a chat turn behind a video. A tool is loaded and
//! run only by its own lane. Residency changes (admission, eviction, the reservation of a load, the GPU lock) are
//! serialized by `gate`; memory admitted but perhaps not yet allocated (a load in progress, the working bytes of a
//! running job) is held back from MemAvailable, less what MemAvailable has dropped since it was admitted (that part has
//! landed and is already counted there).
//!
//! Residency (`config.Residency`, changed live through `setResidency`): an `always` tool is what `keep_loaded` was, a
//! `call` tool is kept the same way while a realtime voice session is open (`callOpened` asks for it as a call
//! connects) and counts its idle TTL from the end of the last call as well as from its last job, and an `on_demand`
//! tool loads with its first request. Below, "kept" means `always`, or `call` during a call (`keepOf`).
//!
//! Eviction: when a request needs room, idle tools are unloaded lowest `priority` first, least recently used among
//! equals, until it fits. Priority only orders the victims: a request is never refused while unloading idle tools
//! (whatever their priority) would make room. A tool that is loading, committing admitted bytes, or running a job
//! (on any lane) is never a victim. A request that cannot fit while such a job holds its memory is refused at once
//! (`insufficient_memory`, with `retry_after`); one that will fit once a load's temporary overshoot settles waits
//! for that, and does not take the loading tool. Kept
//! tools are loaded at start when they fit, skip the idle TTL, and come back by themselves once their lane has been
//! idle for a moment: a kept tool that does not fit unloads idle tools of lower priority, as a request would (never one
//! of its own priority or above, so two kept tools cannot unload each other), unless it was unloaded through the API.
//! A `spare_room_only` tool (an on-demand model nothing waits on) is the exception to "never refused while unloading
//! would make room": it never unloads a kept tool, and loads only with room left for the kept tools' working bytes.
//! So is a call: while a realtime voice session is open, a request (or an API load) for a tool outside the voice loop
//! never unloads a voice-loop model (chat, speech, transcription) of a higher priority, nor a `call` tool of any; and
//! `call` tools never unload each other. It is refused instead, with a
//! message the chat model can say, rather than cut the call's speech mid-sentence (DGX-01, 2026-10-24: a picture asked
//! for in a call loaded the cold image model and unloaded Whisper, then Drama 3; the call went silent for two minutes).
//! The cooperative `reserve_for`, which tells the session first and brings the voice back after the job, is unchanged.
//! Kept tools come back highest priority first across lanes: one waits while a higher one is due to load, and comes back
//! only with room to spare for the working bytes of the kept tools above it (else their next job evicts it again).
//! A `call` tool loads beside higher ones still due when all of them fit (a call's models load at once, not in turn),
//! and may unload idle tools that are not kept, of any priority, as a request would.
//!
//! The host's GPU lock (`Config.gpu_lock`) guards the machine against GPU users outside LocalRouter. The scheduler is
//! the one admission authority inside it: it holds the lock once, shared and without waiting, while any tool is loaded,
//! and releases it when none is. Co-residency is decided by the memory budget alone; a tool never takes the lock itself.
//!
//! A remote tool (`Tool.local` false: another node of a multi-node topology runs it, `tool/node.zig`) holds no memory
//! here and needs no lock: its node admits it against its own. It is never a victim, and holds nothing back.

const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const tool_mod = @import("../tool/tool.zig");
const Tool = tool_mod.Tool;
const jobs = @import("jobs.zig");
const memory = @import("memory.zig");
const speech_gate = @import("speech_gate.zig");
const yielder = @import("yielder.zig");
const marker = @import("../media/marker.zig");
const asset3d = @import("../media/asset3d.zig");
const config = @import("../config.zig");
const Config = config.Config;

pub const Lane = config.Lane;
pub const Residency = config.Residency;

/// Why a tool stays loaded now: its residency is `always`, or `call` while a call is open; else it is `no`t kept.
pub const Keep = enum { no, always, call };

/// A `call` tool's load for an open call, as voice sessions show it (`backend.CallModel`).
pub const CallLoad = struct {
    tool: usize,
    state: enum { queued, loading, warming, ready, failed, no_room, disabled }, // warming: loaded, its voice's warm-up runs
    elapsed_ms: u64,
};

/// The residency budget (`GET /v1/tools`): the most the scheduler may book, what it has booked now (loaded and
/// admitted), and what the `always` tools take (their resident bytes and one job's working bytes each).
pub const Budget = struct { ceiling_bytes: u64, booked_bytes: u64, always_bytes: u64 };

pub const Slot = struct {
    tool: Tool,
    resident: u64 = 0, // measured while loaded (the estimate while it loads)
    last_used_ns: i96 = 0,
    released: bool = false, // unloaded through the API: not kept (`keepOf`) until it is loaded again
    visited: bool = false, // scratch of `loadKept`
    retry_ns: i96 = 0, // a failed kept load is not retried before this time
    backoff_ns: i96 = reload_backoff_min_ns, // the next wait after a failed load: doubles to the maximum, reset by a load
    in_use: bool = false, // its lane is loading or running it: never a victim (under `gate`)
    /// Where a job's memory is, under `gate`. `loading` from admission until the worker has loaded; `committing`
    /// while admitted bytes are still landing after the worker reports ready; `running` for the generate. `free`
    /// is idle. Anything but `free` is never a victim, even when `in_use` has not been set yet.
    phase: enum { free, loading, committing, running } = .free,
    pending: u64 = 0, // admitted bytes that MemAvailable may not show yet (under `gate`)
    pending_ns: i96 = 0, // when `pending` was admitted (under `gate`)
    pending_avail: u64 = 0, // MemAvailable then: what has landed since shows as its drop, not counted twice
    unfit: bool = false, // a kept reload last found no room: lower priorities stop deferring to it
    unload_after: bool = false, // unloaded on request while in use: its lane unloads it when done
    displaced: bool = false, // unloaded to make room (reserve_for, unload_model, or an eviction of a kept tool): it comes
    // back by itself once `hold_until_ns` has passed and it fits again; a load or a release clears it (under `gate`)
    held: bool = false, // the target of a reservation: kept reloads do not evict it until its next job is done
    pausing: bool = false, // a voice model displaced for a model of another kind: voice sessions wait for it (under `gate`)
    // Set through the API (`setResidency`, `<data_dir>/residency.json`); null: the config's. Under `gate`.
    residency: ?config.Residency = null,
    ttl_s: ?u32 = null,
    armed_ns: i96 = 0, // its residency or TTL last changed: its idle TTL counts from here at the earliest
    call_ns: i96 = 0, // a call last asked for it (a `call` tool connecting while it was down)
    loaded_since_s: i64 = 0, // unix seconds of its current load; 0: not loaded
    last_load_ms: ?u64 = null, // how long its last load took
    last_job_ms: ?u64 = null, // how long its last successful generate took (`retryAfter`'s estimate)
    job_started_ns: i96 = 0, // when the generate now running began; 0: none (under `gate`)
    // A speech tool's voice warm in this load: the voice of the last speech job it completed since it loaded (`said`
    // bytes of `voice`), so a session knows whether its voice's first sentence is paid already (`speechWarm`). One voice:
    // the Mac Drama server keeps one reference prefix and rebuilds it on a switch (4 to 6 s on the M5 Ultra); a server
    // that keeps more only warms once more than it needs. Under `gate`.
    voice: [64]u8 = undefined,
    said: u8 = 0,
    warm: bool = false, // a speech job completed since it loaded
    warm_job: ?*const jobs.Job = null, // the warm-up `Loaded` queued after its load, until it ends (compared, never read)
};

/// Told after every load of a speech tool (at start, a kept tool coming back, an API load, a call's, a request's), outside
/// `gate` on its lane's thread: it may queue a job (the voice's warm-up, returned), but must not wait for one.
pub const Loaded = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, tool: usize) ?*const jobs.Job,
};

/// Told before a voice model (speech or transcription) is unloaded to make room, and when every one of them is back,
/// so open voice sessions can say so and wait rather than fail. Called under `gate`.
pub const Notify = struct {
    ctx: *anyopaque,
    paused: *const fn (ctx: *anyopaque, model: []const u8, for_model: []const u8) void,
    resumed: *const fn (ctx: *anyopaque) void,
    /// How many realtime voice sessions are open: while any is, requests do not unload the voice loop's models.
    sessions: *const fn (ctx: *anyopaque) usize,
};

/// The most models one reservation unloads (a box has a handful).
pub const max_victims = 16;

/// How long a lane must be idle before a kept tool is reloaded, and how often it checks the memory.
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
    woken: bool = false, // `wakeAll` since it last looked: it looks again before it waits (under `mutex`)
};

const lane_count = 5;

fn index(l: Lane) usize {
    return switch (l) {
        .main => 0,
        .realtime => 1,
        .chat => 2,
        .listen => 3,
        .decide => 4,
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
    mem_gen: u64 = 0, // bumped under `mutex` by `poke`: a job waiting out a load's overshoot
    mem_cond: Io.Condition = .init,
    lanes: [lane_count]LaneState = @splat(.{}),
    stopping: bool = false,
    gpu_lock_fd: posix.fd_t = -1, // the host's GPU lock while any tool is loaded (`holdGpuLock`); under `gate`
    speech: speech_gate.Holder = .{}, // held while a speech job runs: image and video workers pause between steps
    chat_yield: yielder.Yielder = .{}, // a voice reply's speech behind: the chat model's server is paused (yielder.zig)
    yield_lane: Lane = .chat, // the lane of the tool `chat_yield` last paused: anything queued there ends the pause; under `mutex`
    lock_msg: [512]u8 = undefined,
    lock_len: usize = 0,
    hold_until_ns: i96 = 0, // displaced tools wait until this (a reservation's job ends it early); under `gate`
    notify: ?Notify = null, // voice sessions, told of voice models leaving and coming back
    loaded: ?Loaded = null, // a kept speech tool loaded: its voice is warmed (`api/voice_local.zig`)
    voice_paused: std.atomic.Value(bool) = .init(false), // a voice model is displaced: sessions wait for it
    settings: Io.Mutex = .init, // a residency change and its write to residency.json, one at a time (`api/routes.zig`)
    call_end_ns: i96 = 0, // the last open call ended: `call` tools count their idle TTL from here too; under `gate`

    /// True while a speech or transcription model is unloaded to make room; voice sessions wait instead of failing.
    pub fn voicePaused(s: *const Scheduler) bool {
        return s.voice_paused.load(.acquire);
    }

    /// Whether speech tool `i` is loaded and its last job since it loaded spoke `voice` (its resolved voice: the tool's
    /// `default_voice` for an empty one), so that voice's next sentence starts at the warm speed.
    pub fn speechWarm(s: *Scheduler, i: usize, voice: []const u8) bool {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const slot = &s.slots[i];
        return slot.tool.state() == .ready and slot.warm and std.mem.eql(u8, slot.voice[0..slot.said], voice);
    }

    /// Under `gate`: a finished speech job of `slot`: a completed sentence makes its voice the warm one; a clone or a
    /// sentence spoken from a request's own references leaves no voice warm (the Mac worker restarts its server to add
    /// a clone, and a request's references are their own prefix).
    fn spoke(slot: *Slot, sp: *const @import("../tool/request.zig").Speech, completed: bool) void {
        if (sp.clone.len > 0 or sp.references.len > 0 or sp.voice.len > slot.voice.len) {
            slot.warm = false;
            return;
        }
        if (!completed) return;
        @memcpy(slot.voice[0..sp.voice.len], sp.voice);
        slot.said = @intCast(sp.voice.len);
        slot.warm = true;
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

    /// Whether a realtime voice session is open (the voice loop's models are then guarded from requests).
    fn inCall(s: *const Scheduler) bool {
        const n = s.notify orelse return false;
        return n.sessions(n.ctx) > 0;
    }

    /// A tool's residency now: set through the API, else its config's.
    pub fn ruleOf(slot: *const Slot) Residency {
        return slot.residency orelse config.residencyOf(slot.tool.config());
    }

    /// A tool's idle TTL now: set through the API, else its config's.
    pub fn ttlOf(slot: *const Slot) u32 {
        return slot.ttl_s orelse slot.tool.config().idle_ttl_s;
    }

    /// Why `slot` stays loaded now (`in_call`: a call is open). One unloaded through the API is not kept until it loads
    /// again (or, a `call` tool, until the next call connects).
    fn keepOf(slot: *const Slot, in_call: bool) Keep {
        if (slot.released) return .no;
        return switch (ruleOf(slot)) {
            .always => .always,
            .call => if (in_call) .call else .no,
            .on_demand, .disabled => .no,
        };
    }

    /// When `slot`'s idle TTL starts counting: its last job, a change of its settings, and for a `call` tool the end of
    /// the last call.
    fn idleFrom(s: *const Scheduler, slot: *const Slot) i96 {
        const t = @max(slot.last_used_ns, slot.armed_ns);
        return if (ruleOf(slot) == .call) @max(t, s.call_end_ns) else t;
    }

    /// A realtime voice session connected (it counts in `Notify.sessions` already): the `call` tools that are down load
    /// now, on their lanes, kept until the last call ends. One unloaded through the API, or that failed or found no room
    /// before, is tried again for this call.
    pub fn callOpened(s: *Scheduler) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const now = nowNs(s.io);
        for (s.slots) |*slot| {
            if (ruleOf(slot) != .call) continue;
            slot.released = false;
            if (slot.tool.state() == .ready or slot.tool.state() == .busy) continue;
            slot.call_ns = now;
            slot.unfit = false;
            slot.retry_ns = 0;
            slot.backoff_ns = reload_backoff_min_ns;
        }
        s.wakeAll();
    }

    /// A realtime voice session ended: once none is open, the `call` tools' idle TTL counts from now.
    pub fn callClosed(s: *Scheduler) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        if (s.inCall()) return;
        s.call_end_ns = nowNs(s.io);
        s.wakeAll(); // their lanes may sleep until a far TTL: they count the new one
    }

    /// The `call` tools and how far their loads for the open calls are, in `out` (at most its length), and every disabled
    /// voice-loop model (chat, speech, transcription) as `disabled`: a session says why one is missing rather than
    /// waiting for it, and a client need not read /v1/tools to explain it.
    pub fn callLoads(s: *Scheduler, out: []CallLoad) []CallLoad {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const now = nowNs(s.io);
        var n: usize = 0;
        for (s.slots, 0..) |*slot, i| {
            const rule = ruleOf(slot);
            if (!(rule == .call or (rule == .disabled and voiceLoop(slot.tool.config().kind))) or n == out.len) continue;
            const st = slot.tool.state();
            if (rule == .disabled) {
                out[n] = .{ .tool = i, .state = .disabled, .elapsed_ms = 0 };
                n += 1;
                continue;
            }
            out[n] = .{ .tool = i, .elapsed_ms = if (slot.call_ns > 0) @intCast(@divTrunc(@max(now - slot.call_ns, 0), std.time.ns_per_ms)) else 0, .state = switch (st) {
                .ready, .busy => if (slot.warm_job != null) .warming else .ready,
                .loading => .loading,
                .unloaded, .failed => if (slot.in_use) .loading else if (slot.unfit) .no_room else if (now < slot.retry_ns or st == .failed) .failed else .queued,
            } };
            n += 1;
        }
        return out[0..n];
    }

    /// Under `gate`: what the scheduler may book at most: `budget_bytes`, else the machine's memory less the reserve.
    fn ceiling(s: *const Scheduler) u64 {
        if (s.cfg.budget_bytes > 0) return s.cfg.budget_bytes;
        return (memory.total() orelse 0) -| s.cfg.reserve_bytes;
    }

    /// Under `gate`: what tool `i` takes kept loaded: its measured bytes (its estimate when not loaded) and a job's
    /// working bytes.
    fn keptBytes(s: *Scheduler, i: usize) u64 {
        const slot = &s.slots[i];
        const n = slot.tool.needs(&placeholder(slot.tool.config().kind));
        return (if (slot.resident > 0) slot.resident else n.resident) +| n.working;
    }

    /// Under `gate`: what the `always` tools take (`keptBytes`), with tool `with` counted as one too.
    fn alwaysBytes(s: *Scheduler, with: ?usize) u64 {
        var bytes: u64 = 0;
        for (s.slots, 0..) |*slot, i| {
            if (!slot.tool.local()) continue; // its node holds it
            if (ruleOf(slot) == .always or i == with) bytes +|= s.keptBytes(i);
        }
        return bytes;
    }

    /// From any thread: what tool `i` books when loaded (`keptBytes`): its measured bytes, else its estimate, and a job's
    /// working bytes; what /v1/tools and list_models show as `booked_bytes`, loaded or not.
    pub fn bookedBytes(s: *Scheduler, i: usize) u64 {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        return s.keptBytes(i);
    }

    pub fn budget(s: *Scheduler) Budget {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        var booked: u64 = 0;
        for (s.slots) |slot| booked +|= slot.resident +| slot.pending;
        return .{ .ceiling_bytes = s.ceiling(), .booked_bytes = booked, .always_bytes = s.alwaysBytes(null) };
    }

    /// Why `setResidency` refused: `call` for a tool outside the voice loop, or an `always` set past the ceiling.
    pub const Refusal = union(enum) {
        not_voice: @import("../tool/request.zig").Kind,
        over_budget: struct { needs: u64, always: u64, ceiling: u64 }, // the tool's bytes, the always set's with it
    };

    /// Sets tool `i`'s residency and idle TTL overrides (null: the config's) and applies them now: a tool that becomes
    /// `always` loads when it fits (its lane's next kept load, at once if the lane is idle), and one that leaves it, or
    /// whose TTL changed, counts its idle TTL from now. Null when done, else why not (nothing changed).
    pub fn setResidency(s: *Scheduler, i: usize, rule: ?Residency, ttl_s: ?u32) ?Refusal {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const slot = &s.slots[i];
        const c = slot.tool.config();
        const after = rule orelse config.residencyOf(c);
        if (after == .call and !config.callAllowed(c.kind)) return .{ .not_voice = c.kind };
        if (after == .always and ruleOf(slot) != .always and slot.tool.local()) {
            const total = s.alwaysBytes(i);
            if (total > s.ceiling()) return .{ .over_budget = .{ .needs = s.keptBytes(i), .always = total, .ceiling = s.ceiling() } };
        }
        const before = ruleOf(slot);
        const before_ttl = ttlOf(slot);
        slot.residency = rule;
        slot.ttl_s = ttl_s;
        if (before != after or before_ttl != ttlOf(slot)) slot.armed_ns = nowNs(s.io);
        if (after == .disabled and before != .disabled) { // gone now, or as soon as its running job ends; never back
            slot.displaced = false;
            slot.held = false;
            if (slot.in_use) slot.unload_after = true else s.unloadSlot(slot);
            if (slot.pausing) { // a voice session waiting for it would wait forever
                slot.pausing = false;
                s.checkResumed();
            }
            s.releaseIdleGpuLock();
        }
        if (after == .always and before != .always) { // a fresh start: not released, no backoff from an earlier failure
            slot.released = false;
            slot.unfit = false;
            slot.retry_ns = 0;
            slot.backoff_ns = reload_backoff_min_ns;
        }
        std.log.info("residency {s}: {s}, idle_ttl_s {d} (was {s}, {d})", .{ c.id, @tagName(after), ttlOf(slot), @tagName(before), before_ttl });
        s.wakeAll();
        return null;
    }

    /// Wakes every lane: each looks at its TTLs and kept tools again. Under `gate` or not; never under `mutex`.
    fn wakeAll(s: *Scheduler) void {
        s.mutex.lockUncancelable(s.io); // lock order: gate, then mutex
        defer s.mutex.unlock(s.io);
        for (&s.lanes) |*l| {
            l.woken = true; // a lane between its look and its wait would miss the signal
            l.cond.signal(s.io);
        }
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
                if (i == target or protected(x) or x.visited or x.tool.state() != .ready or x.tool.config().kind == .chat or !x.tool.local()) continue;
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
        s.chat_yield.deinit(s.io);
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
        for (s.slots) |slot| if (slot.tool.local() and (slot.in_use or slot.tool.state() != .unloaded)) return;
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
        // A job queued for a tool that is already loading or committing claims it before the lane reaches
        // `reserve`, so another lane cannot evict it in that gap. An idle ready model stays evictable.
        s.gate.lockUncancelable(s.io);
        const slot = &s.slots[j.tool];
        const claim = slot.phase == .free and (slot.tool.state() == .loading or slot.pending > 0);
        if (claim) slot.phase = .loading;
        s.gate.unlock(s.io);
        s.push(s.laneOfTool(j.tool), .{ .job = j }) catch |err| {
            if (claim) {
                s.gate.lockUncancelable(s.io);
                if (slot.phase == .loading and !slot.in_use) slot.phase = .free;
                s.gate.unlock(s.io);
            }
            return err;
        };
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
        if (lane == s.yield_lane) s.chat_yield.interrupt(s.io); // a paused server would stall it: it runs now
    }

    /// Speech first: `holder` keeps tool `i`'s server paused for `lease_ns` more, but only while nothing else waits on
    /// its lane. The server is shared, so a pause would stall another session's reply, a computer-use step or a prime;
    /// with any of them queued the pause ends instead, and `push` ends it when one arrives. False: not paused.
    pub fn yieldChat(s: *Scheduler, i: usize, holder: usize, lease_ns: i96) bool {
        const pid = s.slots[i].tool.serverPid();
        if (pid <= 0) return false;
        const lane = s.laneOfTool(i);
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.yield_lane = lane;
        if (s.lanes[index(lane)].queue.len > 0 or s.stopping) {
            s.chat_yield.release(s.io, holder);
            return false;
        }
        return s.chat_yield.hold(s.io, pid, holder, lease_ns);
    }

    /// Cancels a queued job at once, or ends the running one: early with the tool kept loaded when it offers that
    /// (`Tool.cancel`), else by aborting it (its worker is killed).
    pub fn cancel(s: *Scheduler, j: *jobs.Job) void {
        j.cancel.store(true, .seq_cst); // seq_cst: `ProcessTool.generate` reads it after publishing its write

        if (s.isRunning(j)) s.end(j, true);
    }

    /// Cancels a queued job, and ends a running one early only if its tool can stay loaded; otherwise the running job
    /// finishes and its result is ignored (the voice loop: never a reload).
    pub fn drop(s: *Scheduler, j: *jobs.Job) void {
        j.cancel.store(true, .seq_cst);
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
            if (!l.woken) l.cond.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(wait_ns), .clock = .awake } }) catch {};
            l.woken = false;
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
        const in_call = s.inCall();
        var wait: i96 = 60 * std.time.ns_per_s;
        for (s.slots, 0..) |*slot, i| {
            if (s.laneOfTool(i) != lane) continue;
            const keep = keepOf(slot, in_call);
            if ((keep != .no or slot.displaced) and slot.tool.state() == .unloaded) wait = @min(wait, reload_poll_ns);
            if (keep != .no) continue;
            if (slot.resident == 0 and slot.tool.state() != .ready) continue;
            const ttl_end = s.idleFrom(slot) + @as(i96, ttlOf(slot)) * std.time.ns_per_s;
            wait = @min(wait, @max(ttl_end - now, 1));
        }
        return wait;
    }

    fn unloadIdle(s: *Scheduler, lane: Lane) void {
        const now = nowNs(s.io);
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const in_call = s.inCall();
        for (s.slots, 0..) |*slot, i| {
            if (s.laneOfTool(i) != lane or slot.in_use) continue;
            if (slot.tool.state() != .ready or keepOf(slot, in_call) != .no) continue;
            if (now - s.idleFrom(slot) >= @as(i96, ttlOf(slot)) * std.time.ns_per_s) {
                std.log.info("{s}: idle for {d} s, unloaded ({s})", .{ slot.tool.config().id, ttlOf(slot), @tagName(ruleOf(slot)) });
                s.unloadSlot(slot);
            }
        }
    }

    /// Loads the lane's kept tools (`keepOf`) that are down, highest priority first, each if it fits once idle tools of
    /// lower priority (on any lane; for a `call` tool also idle ones not kept) are unloaded. At start (`wait_idle` false)
    /// it logs the outcome; later it waits for `reload_grace_ns` of the lane's idleness.
    fn loadKept(s: *Scheduler, lane: Lane, wait_idle: bool) void {
        if (wait_idle and nowNs(s.io) - s.lanes[index(lane)].idle_since_ns < reload_grace_ns) return;
        for (s.slots, 0..) |*slot, i| if (s.laneOfTool(i) == lane) {
            slot.visited = false;
        };
        while (true) {
            s.gate.lockUncancelable(s.io);
            const in_call = s.inCall();
            var best: ?usize = null;
            for (s.slots, 0..) |*slot, i| {
                const c = slot.tool.config();
                if (s.laneOfTool(i) != lane or slot.visited or slot.tool.state() != .unloaded) continue;
                if ((keepOf(slot, in_call) == .no and !slot.displaced) or ruleOf(slot) == .disabled) continue;
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
            const keep = keepOf(slot, in_call);
            const why = if (keep == .call) "call" else if (keep == .always) "keep_loaded" else "restored";
            const estimate = slot.tool.needs(&placeholder(c.kind)).resident;
            if (s.higherKeptDue(i, in_call, if (keep == .call) estimate +| s.keptHeadroom(i, in_call) else null)) { // it takes its room first; this one comes back on a later poll
                s.gate.unlock(s.io);
                continue;
            }
            if (nowNs(s.io) < slot.retry_ns or !s.admit(i, estimate +| s.keptHeadroom(i, in_call), c.priority)) {
                const now = nowNs(s.io);
                const first = !slot.unfit;
                if (now >= slot.retry_ns) slot.unfit = true;
                s.gate.unlock(s.io);
                if (now >= slot.retry_ns and (!wait_idle or (keep == .call and first))) std.log.warn("{s} {s}: not loaded, it does not fit", .{ why, c.id });
                continue;
            }
            s.reserve(slot, estimate);
            s.gate.unlock(s.io);
            s.loadSlot(slot) catch |err| {
                s.done(slot);
                std.log.warn("{s} {s}: load failed: {s}", .{ why, c.id, if (err == error.GpuLockHeld) s.lockMessage() else slot.tool.failure().message });
                slot.retry_ns = nowNs(s.io) + slot.backoff_ns;
                slot.backoff_ns = @min(slot.backoff_ns * 2, reload_backoff_max_ns);
                continue;
            };
            slot.backoff_ns = reload_backoff_min_ns;
            s.done(slot);
            std.log.info("{s} {s}: loaded ({d} MiB in {d} ms)", .{ why, c.id, slot.resident >> 20, slot.last_load_ms orelse 0 });
            s.gate.lockUncancelable(s.io);
            s.checkResumed();
            s.gate.unlock(s.io);
        }
    }

    /// Under `gate`: a kept tool of a higher priority than slot `i` is down and due to load (not released, not backing
    /// off, not just found not to fit). Kept tools come back highest priority first across lanes, so a lower one never
    /// takes the room a higher one is about to need (DGX-01: the image model loaded before chat at a restart, and chat
    /// then unloaded Drama 3 and Whisper to fit). With `alongside` (a `call` tool's bytes) slot `i` waits only when it
    /// and the higher ones still down do not all fit: a call's models then load together.
    fn higherKeptDue(s: *Scheduler, i: usize, in_call: bool, alongside: ?u64) bool {
        const p = s.slots[i].tool.config().priority;
        const now = nowNs(s.io);
        var due = false;
        var bytes: u64 = alongside orelse 0;
        for (s.slots, 0..) |*slot, j| {
            const c = slot.tool.config();
            if (j == i or keepOf(slot, in_call) == .no or c.priority <= p or slot.unfit or now < slot.retry_ns) continue;
            const st = slot.tool.state();
            if (st == .unloaded) bytes +|= slot.tool.needs(&placeholder(c.kind)).resident; // a loading one is admitted already
            if (st == .unloaded or st == .loading) due = true;
        }
        if (!due) return false;
        return alongside == null or !s.fits(bytes);
    }

    /// The working bytes of the loaded kept tools of a higher priority than slot `i`: a kept tool comes back only
    /// with that much to spare, or their next job unloads it again (DGX-01: every voice turn's transcription evicted the
    /// kept image model, which reloaded as soon as the turn ended).
    /// A `spare_room_only` tool leaves room for every loaded kept tool's working bytes, whatever their priority.
    fn keptHeadroom(s: *Scheduler, i: usize, in_call: bool) u64 {
        const p = s.slots[i].tool.config().priority;
        const every = s.slots[i].tool.config().spare_room_only;
        var bytes: u64 = 0;
        for (s.slots, 0..) |*slot, j| {
            const c = slot.tool.config();
            if (j == i or keepOf(slot, in_call) == .no or (c.priority <= p and !every) or slot.tool.state() == .unloaded) continue;
            bytes +|= slot.tool.needs(&placeholder(c.kind)).working;
        }
        return bytes;
    }

    /// Marks `slot` in use with `bytes` admitted; under `gate`. A tool already ready is `running`; anything else is
    /// `loading` until `loadInstance` records the measured bytes.
    fn reserve(s: *Scheduler, slot: *Slot, bytes: u64) void {
        slot.in_use = true;
        if (slot.phase == .free) slot.phase = if (slot.tool.state() == .ready or slot.tool.state() == .busy) .running else .loading;
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
        slot.phase = .free;
        slot.job_started_ns = 0;
        if (slot.resident > 0 or slot.tool.state() == .ready) slot.last_used_ns = nowNs(s.io);
        if (slot.unload_after) {
            slot.unload_after = false;
            s.unloadSlot(slot);
        }
        s.releaseIdleGpuLock(); // a failed load or an unload may leave nothing loaded
        s.poke();
    }

    /// Under `gate`, or by the slot's own lane while it holds `in_use`. Every lane is woken: the tool's own lane may be
    /// asleep until a TTL far off and must notice now that a kept tool is down (another lane evicted it).
    fn unloadSlot(s: *Scheduler, slot: *Slot) void {
        if (slot.tool.state() == .unloaded and slot.resident == 0) return;
        slot.tool.unload();
        slot.resident = 0;
        slot.loaded_since_s = 0;
        slot.warm_job = null; // a warm-up of the gone instance warms nothing
        s.wakeAll(); // lock order: gate, then mutex (nothing takes them the other way)
        s.poke();
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
                if (ruleOf(slot) == .disabled) {
                    s.gate.unlock(s.io);
                    c.failed = true;
                    c.message = "the model is disabled in settings";
                    return;
                }
                if (!s.admit(c.tool, n.resident, null)) {
                    const call = s.guardsCall(slot.tool.config());
                    s.gate.unlock(s.io);
                    c.failed = true;
                    c.message = if (call) "busy: a voice call is open, and loading this tool would unload the voice models the call is using; load it when the call ends" else "insufficient memory to load this tool";
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

    /// From any thread: where tool `i` runs has changed (its node went down or came back, `tool/node.zig`). It is unloaded
    /// now, or when its lane is done with it, without counting as released: a `keep_loaded` tool comes back by itself
    /// (on its new side, at once: a load that failed while its node was down waits no backoff), and any other loads
    /// with its next request.
    pub fn refresh(s: *Scheduler, i: usize) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const slot = &s.slots[i];
        slot.retry_ns = 0;
        slot.backoff_ns = reload_backoff_min_ns;
        if (slot.in_use) slot.unload_after = true else s.unloadSlot(slot);
        s.releaseIdleGpuLock();
    }

    /// What tool `i` needs for its default request: resident bytes, and a job's working bytes.
    pub fn defaultNeeds(s: *Scheduler, i: usize) tool_mod.Needs {
        const r = placeholder(s.slots[i].tool.config().kind);
        return s.slots[i].tool.needs(&r);
    }

    /// From any thread: tool `i`'s residency and idle TTL now (and what was set through the API, null for the config's),
    /// its load times, and whether it is waiting to come back (displaced) or held for a reservation's job.
    pub fn residency(s: *Scheduler, i: usize) struct { displaced: bool, held: bool, rule: Residency, idle_ttl_s: u32, rule_override: ?Residency, ttl_override: ?u32, last_load_ms: ?u64, loaded_since_s: i64 } {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        const slot = &s.slots[i];
        return .{ .displaced = slot.displaced, .held = slot.held, .rule = ruleOf(slot), .idle_ttl_s = ttlOf(slot), .rule_override = slot.residency, .ttl_override = slot.ttl_s, .last_load_ms = slot.last_load_ms, .loaded_since_s = slot.loaded_since_s };
    }

    /// On the target's lane: unloads what `planFor` names (voice sessions told first), loads the target and holds it
    /// until its next job is done or `hold_s` passes; then the unloaded models come back by themselves.
    fn reserveRoom(s: *Scheduler, c: *Control) void {
        const slot = &s.slots[c.tool];
        const req = c.request orelse placeholder(slot.tool.config().kind);
        s.gate.lockUncancelable(s.io);
        if (ruleOf(slot) == .disabled) {
            s.gate.unlock(s.io);
            c.failed = true;
            c.message = "the model is disabled in settings";
            return;
        }
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

    /// `loadInstance`, then a speech tool's warm-up (`warmAfterLoad`): its first sentence is paid now, not in a call's
    /// first reply (4.4 s for the Mac's Drama 3 to build its voice prefix). A job's own load warms after the job.
    fn loadSlot(s: *Scheduler, slot: *Slot) (tool_mod.Error || error{GpuLockHeld})!void {
        try s.loadInstance(slot);
        s.warmAfterLoad(slot);
    }

    /// After every load of a speech tool (at start, a kept one coming back, an API load, a call's, a request's once its
    /// job is done): its voice's warm-up (`Loaded`), unless the tool unloads right after each job (`idle_ttl_s` 0, not
    /// kept), where the warm-up would only load it again, and again after that.
    fn warmAfterLoad(s: *Scheduler, slot: *Slot) void {
        if (slot.tool.config().kind != .speech) return;
        const l = s.loaded orelse return;
        s.gate.lockUncancelable(s.io);
        const st = slot.tool.state();
        const skip = (ttlOf(slot) == 0 and keepOf(slot, s.inCall()) == .no) or (st != .ready and st != .busy);
        s.gate.unlock(s.io);
        if (skip) return;
        const i = (@intFromPtr(slot) - @intFromPtr(s.slots.ptr)) / @sizeOf(Slot);
        const j = l.f(l.ctx, i); // null when its voice is warm already (the job that loaded it spoke in it)
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        slot.warm_job = j;
    }

    /// Under `gate`: `j` ended; if it was `slot`'s warm-up, the slot is warm (or as warm as it gets).
    fn warmEnded(s: *Scheduler, slot: *Slot, j: *const jobs.Job) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        if (slot.warm_job == j) slot.warm_job = null;
    }

    /// Loads outside `gate` (the reservation holds the memory) while LocalRouter holds the host's GPU lock, then records
    /// the measured bytes; `error.GpuLockHeld` (why: `lockMessage`) when a GPU user outside LocalRouter holds it.
    fn loadInstance(s: *Scheduler, slot: *Slot) (tool_mod.Error || error{GpuLockHeld})!void {
        const estimate = slot.tool.needs(&placeholder(slot.tool.config().kind)).resident;
        if (slot.tool.local()) { // a remote tool's node holds its own lock
            s.gate.lockUncancelable(s.io);
            const locked = s.holdGpuLock();
            s.gate.unlock(s.io);
            if (!locked) return error.GpuLockHeld;
        }
        const began = nowNs(s.io);
        const measured = try slot.tool.load();
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        slot.last_load_ms = @intCast(@divTrunc(@max(nowNs(s.io) - began, 0), std.time.ns_per_ms));
        slot.loaded_since_s = nowUnix(s.io);
        slot.resident = if (measured > 0) measured else estimate;
        slot.pending -|= estimate; // now visible in MemAvailable
        slot.pending_ns = nowNs(s.io); // what is left (a job's working bytes) lands from here
        slot.pending_avail = memory.available() orelse 0;
        // Admitted bytes have landed. A direct `loadSlot` (no `reserve`) leaves `phase` alone.
        if (slot.phase == .loading or slot.phase == .committing) slot.phase = .running;
        slot.last_used_ns = nowNs(s.io);
        slot.released = false;
        slot.displaced = false;
        slot.pausing = false;
        slot.warm = false; // a new instance: no voice is warm in it yet
        s.checkResumed();
        s.poke(); // a request waiting out this load's overshoot can admit now
    }

    fn runJob(s: *Scheduler, j: *jobs.Job) void {
        const slot = &s.slots[j.tool];
        defer s.warmEnded(slot, j);
        s.setStatus(j, .in_progress);
        const n = slot.tool.needs(&j.request);
        s.gate.lockUncancelable(s.io);
        const wait_s = s.settleWait(j.tool);
        s.gate.unlock(s.io);
        const deadline = nowNs(s.io) + @as(i96, wait_s) * std.time.ns_per_s;
        var cold: bool = true;
        var extra: u64 = 0;
        var room: u64 = 0;
        while (true) {
            s.gate.lockUncancelable(s.io);
            if (ruleOf(slot) == .disabled) { // queued before it was disabled
                s.gate.unlock(s.io);
                var buf: [160]u8 = undefined;
                return s.finish(j, .failed, 409, "model_disabled", std.fmt.bufPrint(&buf, "{s} is disabled in settings", .{slot.tool.config().id}) catch "the model is disabled in settings", null);
            }
            if (j.cancel.load(.acquire)) {
                s.gate.unlock(s.io);
                return s.finishCancelled(j);
            }
            cold = slot.tool.state() != .ready;
            extra = n.working + if (cold) n.resident else 0;
            const spare = slot.tool.config().spare_room_only;
            // A spare-room tool's load must also leave the kept tools room for their jobs (else their next job evicts it).
            room = extra +| if (spare and cold) s.keptHeadroom(j.tool, s.inCall()) else 0;
            switch (s.admission(j.tool, room)) {
                .proceed => {
                    if (!s.admit(j.tool, room, null)) {
                        const hint = s.retryAfter(j.tool);
                        const call = s.guardsCall(slot.tool.config());
                        s.gate.unlock(s.io);
                        return s.refuseMemory(j, slot, room, extra, hint, call);
                    }
                    s.reserve(slot, extra);
                    s.gate.unlock(s.io);
                    break;
                },
                .wait => {
                    // Hold `mutex` before releasing `gate`, so a `poke` cannot land in the gap (gate, then mutex).
                    if (nowNs(s.io) >= deadline) {
                        const hint = s.retryAfter(j.tool);
                        const call = s.guardsCall(slot.tool.config());
                        s.gate.unlock(s.io);
                        return s.refuseMemory(j, slot, room, extra, hint, call);
                    }
                    s.mutex.lockUncancelable(s.io);
                    const seen = s.mem_gen;
                    s.gate.unlock(s.io);
                    while (s.mem_gen == seen) {
                        const left = deadline - nowNs(s.io);
                        if (left <= 0) break;
                        s.mem_cond.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(left), .clock = .awake } }) catch {};
                    }
                    s.mutex.unlock(s.io);
                },
                .refuse => {
                    const hint = s.retryAfter(j.tool);
                    const call = s.guardsCall(slot.tool.config());
                    s.gate.unlock(s.io);
                    return s.refuseMemory(j, slot, room, extra, hint, call);
                },
            }
        }
        defer s.endHold(slot); // after `done`: a reservation ends with its job, and what it unloaded comes back
        defer s.done(slot);
        const heavy = j.request == .image or j.request == .video or j.request == .asset3d;
        if (cold and heavy) s.speech.waitIdle(s.io); // a load next to speech starves it too
        if (cold) {
            s.setPhase(j, "load");
            s.loadInstance(slot) catch |err| {
                if (err == error.GpuLockHeld) return s.finish(j, .failed, 503, "server_error", s.lockMessage(), null);
                return s.finishError(j, slot, @errorCast(err));
            };
        }
        defer if (cold) s.warmAfterLoad(slot); // after the job (before `done`), which may have warmed its voice itself
        var arena: std.heap.ArenaAllocator = .init(s.gpa);
        defer arena.deinit();
        const job: tool_mod.Job = .{ .id = j.id, .dir = j.dir, .request = j.request, .cancelled = &j.cancel };
        var pctx: ProgressCtx = .{ .s = s, .j = j };
        const speaking = j.request == .speech;
        if (speaking) s.speech.enter(s.io);
        defer if (speaking) s.speech.leave(s.io);
        s.gate.lockUncancelable(s.io);
        const gen_began = nowNs(s.io);
        slot.job_started_ns = gen_began;
        s.gate.unlock(s.io);
        const out = slot.tool.generate(&job, .{ .ctx = &pctx, .progress = ProgressCtx.report }, arena.allocator()) catch |err| {
            if (slot.tool.state() == .unloaded) s.forget(slot);
            if (speaking) {
                s.gate.lockUncancelable(s.io);
                spoke(slot, &j.request.speech, false);
                s.gate.unlock(s.io);
            }
            return s.finishError(j, slot, err);
        };
        s.gate.lockUncancelable(s.io);
        slot.last_job_ms = @intCast(@divTrunc(@max(nowNs(s.io) - gen_began, 0), std.time.ns_per_ms));
        slot.job_started_ns = 0;
        if (ttlOf(slot) == 0 and keepOf(slot, s.inCall()) == .no) slot.unload_after = true;
        s.gate.unlock(s.io);
        // a 3D asset completes only with a manifest naming whole files (`media/asset3d.zig`)
        if (j.request == .asset3d) {
            var why: []const u8 = "";
            if (asset3d.check(s.io, arena.allocator(), j.dir, out.files, j.request.asset3d, &why) == null) return s.finish(j, .failed, 500, "server_error", why, null);
        }
        // every image, video and 3D asset handed out carries the machine-generated label, whichever tool wrote it
        if (heavy) marker.outputs(s.io, s.gpa, j.dir, out.files, marker.forRequest(slot.tool.config().id, j.request));
        if (speaking) {
            s.gate.lockUncancelable(s.io);
            spoke(slot, &j.request.speech, true);
            s.gate.unlock(s.io);
        }
        s.finish(j, .completed, 200, "", "", out);
    }

    /// The worker died or was killed: it holds no memory now.
    fn forget(s: *Scheduler, slot: *Slot) void {
        s.gate.lockUncancelable(s.io);
        defer s.gate.unlock(s.io);
        slot.resident = 0;
        slot.loaded_since_s = 0;
    }

    /// Under `gate`: true when `extra` more bytes fit; unloads idle tools, lowest priority then least recently used
    /// first, until they do. A tool that is loading, committing, or running a job is never a victim. With `below`,
    /// only tools of a lower priority than it are victims (a kept tool coming back; a `call` one also takes idle
    /// tools that are not kept); a `spare_room_only` target never unloads a kept tool (`mayEvict`). Nothing is
    /// unloaded unless unloading all of those would make room: a load that cannot happen costs no one their model.
    fn admit(s: *Scheduler, target: usize, extra: u64, below: ?i32) bool {
        if (!s.roomIfEvicted(target, extra, below)) return false;
        const in_call = s.inCall();
        while (!s.fits(extra)) {
            var victim: ?*Slot = null;
            for (s.slots, 0..) |*slot, i| {
                if (!s.isVictim(target, slot, i, below)) continue;
                if (victim == null or evictsBefore(slot, victim.?)) victim = slot;
            }
            const v = victim orelse return false;
            if (keepOf(v, in_call) != .no) s.displace(v, target); // it comes back after this job
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
        return true;
    }

    /// What `runJob` does when `extra` does not fit yet. `wait`: a load's admitted bytes overshoot what it will hold,
    /// and the request fits once that settles (the loading tool stays). `refuse`: it cannot fit while a non-evictable
    /// job holds its memory, or it cannot fit at all.
    const Admission = enum { proceed, wait, refuse };

    /// Under `gate`.
    fn admission(s: *Scheduler, target: usize, extra: u64) Admission {
        if (s.roomIfEvicted(target, extra, null)) return .proceed;
        if (s.roomAfterLoadingSettles(target, extra, null)) return .wait;
        return .refuse;
    }

    /// Under `gate`: `extra` fits once every idle victim is unloaded. Unloads nothing. The victims' bytes come off what
    /// is held and back into what the host has available (`fitsFreeing`); taking them off `extra` instead could not
    /// recover when usage is already over the budget (a worker that reports more than its estimate).
    fn roomIfEvicted(s: *Scheduler, target: usize, extra: u64, below: ?i32) bool {
        var freeable: u64 = 0;
        for (s.slots, 0..) |*slot, i| if (s.isVictim(target, slot, i, below)) {
            freeable +|= slot.resident;
        };
        return s.fitsFreeing(extra, freeable);
    }

    /// Under `gate`: true when `extra` does not fit now but does once every loading or committing slot is booked at
    /// the bytes it will hold when the load settles, and idle victims are unloaded. Those slots are not unloaded here.
    fn roomAfterLoadingSettles(s: *Scheduler, target: usize, extra: u64, below: ?i32) bool {
        if (s.slots.len > 64) return false;
        var old_r: [64]u64 = undefined;
        var old_p: [64]u64 = undefined;
        var any = false;
        for (s.slots, 0..) |*slot, i| {
            old_r[i] = slot.resident;
            old_p[i] = slot.pending;
            if (i == target or !loadingPin(slot)) continue;
            any = true;
            slot.resident = steadyResident(slot);
            slot.pending = 0;
        }
        defer for (s.slots, 0..) |*slot, i| {
            slot.resident = old_r[i];
            slot.pending = old_p[i];
        };
        if (!any) return false;
        return s.roomIfEvicted(target, extra, below);
    }

    /// Bytes `slot` holds once a load in progress has settled: the measured resident or the estimate, whichever is
    /// more, plus a job's working bytes while the lane is in the slot.
    fn steadyResident(slot: *const Slot) u64 {
        const n = slot.tool.needs(&placeholder(slot.tool.config().kind));
        const base = @max(slot.resident, n.resident);
        return if (slot.in_use) base +| n.working else base;
    }

    /// Under `gate`. Never unload this slot to make room: its lane is in it, it is loading or committing, or it
    /// still holds admitted bytes MemAvailable may not show yet. A tool whose own state is `loading` or `busy`
    /// counts even when the scheduler has not set `phase` (the worker got there first).
    fn protected(slot: *const Slot) bool {
        if (slot.in_use or slot.phase != .free or slot.pending > 0) return true;
        return switch (slot.tool.state()) {
            .loading, .busy => true,
            else => false,
        };
    }

    /// Under `gate`. A protected slot whose memory may still shrink: not a generate already running.
    fn loadingPin(slot: *const Slot) bool {
        if (!protected(slot) or slot.phase == .running or slot.tool.state() == .busy) return false;
        return slot.phase == .loading or slot.phase == .committing or slot.tool.state() == .loading or slot.pending > 0;
    }

    /// Under `gate`: `slot` may be unloaded to make `extra` fit for `target`.
    fn isVictim(s: *Scheduler, target: usize, slot: *Slot, i: usize, below: ?i32) bool {
        if (i == target or protected(slot) or slot.tool.state() != .ready or !slot.tool.local()) return false;
        if (below != null and slot.held) return false; // a reservation's target waits for its job
        const want = s.slots[target].tool.config();
        const in_call = s.inCall();
        const take: Take = .{ .below = below, .call = !voiceLoop(want.kind) and in_call, .for_call = keepOf(&s.slots[target], in_call) == .call };
        return mayEvict(want, slot.tool.config(), keepOf(slot, in_call), take);
    }

    /// Under `gate`: the `retry_after` hint for a request refused for memory: when the memory it waits on should be free.
    /// For each tool that cannot be evicted now: a running job's tool frees its memory when the job ends, estimated as
    /// that tool's last job less the time this one has run; a loading or committing tool, after the rest of its load
    /// (its last load less the time since admission) and then its job. The longest of those, at least the config's
    /// `retry_after_s` (30 when unset or 0) and at most 600 s.
    fn retryAfter(s: *const Scheduler, target: usize) u32 {
        var hint: u32 = if (s.cfg.retry_after_s == 0) 30 else s.cfg.retry_after_s;
        const now = nowNs(s.io);
        for (s.slots, 0..) |*slot, i| {
            if (i == target or !protected(slot)) continue;
            const ms = remainingMs(slot, now) orelse continue;
            const sec: u32 = @intCast(@min(@max((ms + 999) / 1000, 1), 600));
            if (sec > hint) hint = sec;
        }
        return hint;
    }

    /// Under `gate`: the longest a request waits for a load's overshoot to settle (`admission`'s `.wait`): the config's
    /// `retry_after_s` (30 when 0), or a blocker's last load when longer, capped at 10 minutes. A blocker's job is not
    /// counted: the wait ends when the load settles, not when the job does.
    fn settleWait(s: *const Scheduler, target: usize) u32 {
        var hint: u32 = if (s.cfg.retry_after_s == 0) 30 else s.cfg.retry_after_s;
        for (s.slots, 0..) |*slot, i| {
            if (i == target or !protected(slot)) continue;
            const ms = slot.last_load_ms orelse continue;
            const sec: u32 = @intCast(@min(@max((ms + 999) / 1000, 1), 600));
            if (sec > hint) hint = sec;
        }
        return hint;
    }

    /// Under `gate`: how long until protected `slot` should be free again, from its last load and job; null: no history.
    fn remainingMs(slot: *const Slot, now: i96) ?u64 {
        const since = struct {
            fn ms(t0: i96, t: i96) u64 {
                return if (t0 == 0) 0 else @intCast(@divTrunc(@max(t - t0, 0), std.time.ns_per_ms));
            }
        }.ms;
        if (slot.job_started_ns != 0) { // the generate is running
            const job = slot.last_job_ms orelse return null;
            return job -| since(slot.job_started_ns, now);
        }
        if (slot.phase == .loading or slot.phase == .committing) {
            if (slot.last_load_ms == null and slot.last_job_ms == null) return null;
            const load = (slot.last_load_ms orelse 0) -| since(slot.pending_ns, now);
            return load +| (slot.last_job_ms orelse 0);
        }
        // in use but between phases (the lane holds it): its last job is the best guess
        return slot.last_job_ms orelse slot.last_load_ms;
    }

    /// The request cannot run: `retry_after` is set, the pinned tools stay. `call` was read under `gate`.
    fn refuseMemory(s: *Scheduler, j: *jobs.Job, slot: *Slot, room: u64, extra: u64, hint: u32, call: bool) void {
        j.retry_after = hint;
        const spare = slot.tool.config().spare_room_only;
        var buf: [320]u8 = undefined;
        const msg = (if (call)
            std.fmt.bufPrint(&buf, "busy: a voice call is open, and {s} needs {d} MiB more than is free without unloading the voice models the call is using; it was not started, so the call keeps its voice. Try again when the call ends", .{ slot.tool.config().id, room >> 20 })
        else if (spare)
            std.fmt.bufPrint(&buf, "insufficient memory: this request needs {d} MiB more than is spare beside the models kept loaded, and {s} never unloads them (spare_room_only)", .{ room >> 20, slot.tool.config().id })
        else
            std.fmt.bufPrint(&buf, "insufficient memory: this request needs {d} MiB more than is free within the budget", .{extra >> 20})) catch "insufficient memory";
        s.finish(j, .failed, 503, "insufficient_memory", msg, null);
    }

    /// Wakes jobs waiting in `admission`'s `.wait`. Safe under `gate` (lock order: gate, then mutex).
    fn poke(s: *Scheduler) void {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.mem_gen +%= 1;
        s.mem_cond.broadcast(s.io);
    }

    /// Under `gate`: whether a load or job of `target` must leave the voice loop's models alone, a call being open.
    fn guardsCall(s: *const Scheduler, target: *const config.ToolConfig) bool {
        return !voiceLoop(target.kind) and s.inCall();
    }

    /// The models a voice call runs on: the chat model, speech and transcription.
    fn voiceLoop(kind: @import("../tool/request.zig").Kind) bool {
        return kind == .chat or kind == .speech or kind == .transcription;
    }

    /// What a load may unload: `below`, a kept tool coming back (only lower priorities); `call`, a voice session is open
    /// and the target is outside the voice loop; `for_call`, the target is a `call` tool kept by an open call.
    const Take = struct { below: ?i32 = null, call: bool = false, for_call: bool = false };

    /// Whether an idle tool (`victim`, kept now as `keep` says) may be unloaded to make room for `target`: with `below`
    /// only one of a lower priority (or, for a call's tool, one that is not kept); for a `spare_room_only` target never a
    /// kept one; with `call` never a voice-loop model of a higher priority than `target`, nor a call's; and a call's
    /// tools never one another. Otherwise any: priority only orders the victims.
    fn mayEvict(target: *const config.ToolConfig, victim: *const config.ToolConfig, keep: Keep, t: Take) bool {
        if (t.for_call and keep == .call) return false;
        if (t.below) |p| if (victim.priority >= p and !(t.for_call and keep == .no)) return false;
        if (target.spare_room_only and keep != .no) return false;
        if (t.call and voiceLoop(victim.kind) and (victim.priority > target.priority or keep == .call)) return false;
        return true;
    }

    fn evictsBefore(a: *const Slot, b: *const Slot) bool {
        const pa, const pb = .{ a.tool.config().priority, b.tool.config().priority };
        return if (pa != pb) pa < pb else a.last_used_ns < b.last_used_ns;
    }

    /// Under `gate`. Bytes other lanes admitted but may not have allocated yet are held back from MemAvailable.
    fn fits(s: *Scheduler, extra: u64) bool {
        return s.fitsFreeing(extra, 0);
    }

    /// Under `gate`: `fits`, as it would be once tools holding `freed` resident bytes are unloaded: those bytes leave
    /// what is held and return to the host's MemAvailable.
    fn fitsFreeing(s: *Scheduler, extra: u64, freed: u64) bool {
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
        held -|= freed;
        const avail = (memory.available() orelse 0) +| freed;
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
        .decision => .{ .decision = .{ .body = "" } },
        .asset3d => .{ .asset3d = .{ .image = "" } },
    };
}

pub fn nowNs(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

pub fn nowUnix(io: Io) i64 {
    return @intCast(@divTrunc(Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_s));
}

test "victims: priority orders them, a kept tool coming back takes only lower ones, a spare-room tool never a kept one" {
    const T = config.ToolConfig;
    const kept: T = .{ .id = "image", .kind = .image, .priority = 10, .keep_loaded = true };
    const video: T = .{ .id = "video", .kind = .video, .priority = 0 };
    const idle: T = .{ .id = "old", .kind = .image, .priority = 0 };
    const decision: T = .{ .id = "decide", .kind = .decision, .priority = 5, .spare_room_only = true };
    const t = std.testing;
    const may = Scheduler.mayEvict;
    // A request (a video) may unload anything idle, the kept image model included: priority only orders the victims.
    try t.expect(may(&video, &kept, .always, .{}));
    // A spare-room model never unloads a kept one, of any priority, but may take an idle tool that is not kept...
    try t.expect(!may(&decision, &kept, .always, .{}));
    try t.expect(!may(&decision, &.{ .id = "chat", .kind = .chat, .priority = 100, .keep_loaded = true }, .always, .{}));
    try t.expect(may(&decision, &idle, .no, .{}));
    // A kept tool coming back (below its priority) takes only lower priorities, the spare-room model among them.
    try t.expect(may(&kept, &decision, .no, .{ .below = kept.priority }));
    try t.expect(!may(&kept, &.{ .id = "voice", .kind = .speech, .priority = 95, .keep_loaded = true }, .always, .{ .below = kept.priority }));
}

test "victims during a call: a request outside the voice loop never takes a voice-loop model of a higher priority" {
    const T = config.ToolConfig;
    const image: T = .{ .id = "image", .kind = .image, .priority = 10, .keep_loaded = true };
    const video: T = .{ .id = "video", .kind = .video, .priority = 0 };
    const tts: T = .{ .id = "tts", .kind = .speech, .priority = 95, .keep_loaded = true };
    const stt: T = .{ .id = "stt", .kind = .transcription, .priority = 90, .keep_loaded = true };
    const chat: T = .{ .id = "chat", .kind = .chat, .priority = 100, .keep_loaded = true };
    const t = std.testing;
    const may = Scheduler.mayEvict;
    // DGX-01, 2026-10-24: the image model loading for a picture asked for in a call unloaded Whisper, then Drama 3.
    for ([_]*const T{ &tts, &stt, &chat }) |voice| {
        try t.expect(may(&image, voice, .always, .{})); // no call: priority only orders the victims
        try t.expect(!may(&image, voice, .always, .{ .call = true }));
        try t.expect(!may(&video, voice, .always, .{ .call = true }));
    }
    // What is not the voice loop's still makes room during a call: a video unloads the image model.
    try t.expect(may(&video, &image, .always, .{ .call = true }));
    // A voice-loop model making room for another (speech for transcription) is not guarded against itself.
    try t.expect(may(&stt, &tts, .always, .{}));
}

/// A chat model whose server is `pid`: only what `yieldChat` and `submit` ask of a tool.
const FakeChat = struct {
    cfg: config.ToolConfig = .{ .id = "chat", .kind = .chat },
    pid: posix.pid_t,

    const vtable: Tool.VTable = .{ .config = cfgOf, .needs = needsOf, .load = load, .generate = generate, .unload = none, .abort = none, .cancel = cancel, .state = state, .failure = failure, .server_pid = serverPid };
    fn tool(f: *FakeChat) Tool {
        return .{ .ptr = f, .vtable = &vtable };
    }
    fn me(p: *anyopaque) *FakeChat {
        return @ptrCast(@alignCast(p));
    }
    fn cfgOf(p: *anyopaque) *const config.ToolConfig {
        return &me(p).cfg;
    }
    fn needsOf(_: *anyopaque, _: *const Request) tool_mod.Needs {
        return .{ .resident = 0, .working = 0 };
    }
    fn load(_: *anyopaque) tool_mod.Error!u64 {
        return 0;
    }
    fn generate(_: *anyopaque, _: *const tool_mod.Job, _: tool_mod.Sink, _: std.mem.Allocator) tool_mod.Error!tool_mod.Output {
        return error.GenerateFailed;
    }
    fn none(_: *anyopaque) void {}
    fn cancel(_: *anyopaque) bool {
        return false;
    }
    fn state(_: *anyopaque) tool_mod.State {
        return .ready;
    }
    fn failure(_: *anyopaque) tool_mod.Failure {
        return .{};
    }
    fn serverPid(p: *anyopaque) posix.pid_t {
        return me(p).pid;
    }
};

test "two sessions: one session's speech never pauses the chat model while another session's request waits for it" {
    const t = std.testing;
    var child = try yielder.sleeper();
    defer {
        _ = posix.system.kill(child.id.?, posix.SIG.KILL);
        _ = child.wait(t.io) catch {};
    }
    const pid = child.id.?;
    var fake: FakeChat = .{ .pid = pid };
    var slots = [_]Slot{.{ .tool = fake.tool() }};
    const cfg: Config = .{};
    var table: jobs.Table = .{ .gpa = t.allocator };
    defer table.deinit();
    var s: Scheduler = .{ .io = t.io, .gpa = t.allocator, .cfg = &cfg, .slots = &slots, .table = &table };
    defer s.deinit();
    const a: usize = 1; // session A's reply, speaking
    var job_b: jobs.Job = .{ .arena = .init(t.allocator), .id = "b", .tool = 0, .request = .{ .chat = .{ .body = "{}" } }, .dir = "", .created_at = 0 };
    defer job_b.arena.deinit();

    // A's speech is behind and nothing else waits: the chat model pauses for A.
    try t.expect(s.yieldChat(0, a, 10 * std.time.ns_per_s));
    try t.expect(yielder.becomes(pid, true));
    // Session B (JJ's other device) asks the chat model: it runs again at once, not when A's lease runs out.
    try s.submit(&job_b);
    try t.expect(yielder.becomes(pid, false));
    try t.expect(!s.chat_yield.stopped(t.io));
    // While B's request waits, A's speech behind does not pause it again.
    try t.expect(!s.yieldChat(0, a, 10 * std.time.ns_per_s));
    try t.expect(!s.chat_yield.stopped(t.io));
    // Once nothing waits, A may pause it again.
    _ = s.lanes[index(.chat)].queue.popFront();
    try t.expect(s.yieldChat(0, a, 10 * std.time.ns_per_s));
    try t.expect(yielder.becomes(pid, true));
    s.chat_yield.release(t.io, a);
    try t.expect(yielder.becomes(pid, false));
}

test "a speech instance's warm voice: the last sentence it completed since it loaded; a clone or a request's own references leave none" {
    const t = std.testing;
    var slot: Slot = .{ .tool = undefined };
    try t.expect(!slot.warm);
    Scheduler.spoke(&slot, &.{ .input = "Hello.", .voice = "Mae" }, true);
    try t.expect(slot.warm);
    try t.expectEqualStrings("Mae", slot.voice[0..slot.said]);
    Scheduler.spoke(&slot, &.{ .input = "Hi.", .voice = "Arlo" }, false); // failed: the instance's prefix is unknown, Mae stays
    try t.expectEqualStrings("Mae", slot.voice[0..slot.said]);
    Scheduler.spoke(&slot, &.{ .input = "Hi.", .voice = "Arlo" }, true);
    try t.expectEqualStrings("Arlo", slot.voice[0..slot.said]);
    Scheduler.spoke(&slot, &.{ .input = "", .voice = "", .clone = "clone.json" }, true);
    try t.expect(!slot.warm);
    Scheduler.spoke(&slot, &.{ .input = "Hi.", .voice = "Arlo" }, true);
    Scheduler.spoke(&slot, &.{ .input = "Hi.", .voice = "Arlo", .references = "references.json" }, true);
    try t.expect(!slot.warm);
}

/// Tests (here and `api/routes.zig`): a tool that holds `bytes` once loaded; its load takes `load_ms` (and fails with `fail`): what residency asks of a tool.
pub const FakeTool = struct {
    cfg: config.ToolConfig,
    bytes: u64 = 1 << 20,
    loaded_bytes: u64 = 0, // what `load` reports, when not `bytes` (0): a worker that holds more than its estimate
    load_ms: u32 = 0,
    fail: bool = false,
    st: std.atomic.Value(tool_mod.State) = .init(.unloaded),

    const vtable: Tool.VTable = .{ .config = cfgOf, .needs = needsOf, .load = load, .generate = generate, .unload = unload, .abort = unload, .cancel = cancel, .state = state, .failure = failure };
    fn tool(f: *FakeTool) Tool {
        return .{ .ptr = f, .vtable = &vtable };
    }
    fn me(p: *anyopaque) *FakeTool {
        return @ptrCast(@alignCast(p));
    }
    fn cfgOf(p: *anyopaque) *const config.ToolConfig {
        return &me(p).cfg;
    }
    fn needsOf(p: *anyopaque, _: *const Request) tool_mod.Needs {
        return .{ .resident = me(p).bytes, .working = 0 };
    }
    fn load(p: *anyopaque) tool_mod.Error!u64 {
        const f = me(p);
        f.st.store(.loading, .release);
        if (f.load_ms > 0) Io.sleep(std.testing.io, .fromMilliseconds(f.load_ms), .awake) catch {};
        if (f.fail) {
            f.st.store(.unloaded, .release);
            return error.LoadFailed;
        }
        f.st.store(.ready, .release);
        return if (f.loaded_bytes > 0) f.loaded_bytes else f.bytes;
    }
    fn generate(_: *anyopaque, _: *const tool_mod.Job, _: tool_mod.Sink, _: std.mem.Allocator) tool_mod.Error!tool_mod.Output {
        return error.GenerateFailed;
    }
    fn unload(p: *anyopaque) void {
        me(p).st.store(.unloaded, .release);
    }
    fn cancel(_: *anyopaque) bool {
        return false;
    }
    fn state(p: *anyopaque) tool_mod.State {
        return me(p).st.load(.acquire);
    }
    fn failure(p: *anyopaque) tool_mod.Failure {
        return if (me(p).fail) .{ .message = "the fake failed to load" } else .{};
    }
};

/// Open realtime voice sessions, as the registry would count them.
const FakeCalls = struct {
    n: std.atomic.Value(usize) = .init(0),

    fn notify(c: *FakeCalls) Notify {
        return .{ .ctx = c, .paused = paused, .resumed = resumed, .sessions = sessions };
    }
    fn paused(_: *anyopaque, _: []const u8, _: []const u8) void {}
    fn resumed(_: *anyopaque) void {}
    fn sessions(p: *anyopaque) usize {
        const c: *FakeCalls = @ptrCast(@alignCast(p));
        return c.n.load(.acquire);
    }
};

/// A scheduler over `fakes`, with a budget of `budget_mib` and no reserve (the machine's own memory never decides).
pub const Bench = struct {
    slots: []Slot,
    table: jobs.Table,
    cfg: Config,
    calls: FakeCalls = .{},
    s: Scheduler = undefined,

    pub fn init(b: *Bench, fakes: []FakeTool, budget_mib: u64) !void {
        const t = std.testing;
        b.slots = try t.allocator.alloc(Slot, fakes.len);
        for (b.slots, fakes) |*slot, *f| slot.* = .{ .tool = f.tool() };
        b.table = .{ .gpa = t.allocator };
        b.cfg = .{ .budget_bytes = budget_mib << 20, .reserve_bytes = 0 };
        b.calls = .{};
        b.s = .{ .io = t.io, .gpa = t.allocator, .cfg = &b.cfg, .slots = b.slots, .table = &b.table };
        b.s.notify = b.calls.notify();
    }
    pub fn deinit(b: *Bench) void {
        b.s.deinit();
        b.table.deinit();
        std.testing.allocator.free(b.slots);
    }
    /// Every clock the idle TTLs read moves `s` seconds into the past.
    fn age(b: *Bench, s: i96) void {
        const d = s * std.time.ns_per_s;
        for (b.slots) |*slot| {
            slot.last_used_ns -= d;
            slot.armed_ns -= d;
        }
        b.s.call_end_ns -= d;
    }
    fn state(b: *Bench, i: usize) tool_mod.State {
        return b.slots[i].tool.state();
    }
};

test "residency at start: always loads, call and on_demand wait for a call or a request; keep_loaded is always" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "chat", .kind = .chat, .priority = 100, .keep_loaded = true } },
        .{ .cfg = .{ .id = "tts", .kind = .speech, .priority = 95, .residency = .call } },
        .{ .cfg = .{ .id = "img", .kind = .image } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    for ([_]Lane{ .chat, .realtime, .main }) |l| b.s.loadKept(l, false);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    try t.expectEqual(tool_mod.State.unloaded, b.state(1));
    try t.expectEqual(tool_mod.State.unloaded, b.state(2));
    try t.expect(b.slots[0].loaded_since_s > 0 and b.slots[0].last_load_ms != null);
    try t.expectEqual(Residency.always, Scheduler.ruleOf(&b.slots[0]));
    // An always tool is never unloaded for its idle TTL.
    b.age(100_000);
    b.s.unloadIdle(.chat);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
}

test "a call loads its call tools together, keeps them while it is open, and unloads them idle_ttl_s after the last one" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "tts", .kind = .speech, .priority = 95, .residency = .call, .idle_ttl_s = 60 } },
        .{ .cfg = .{ .id = "stt", .kind = .transcription, .priority = 90, .residency = .call, .idle_ttl_s = 60 } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    b.calls.n.store(1, .release);
    b.s.callOpened();
    var buf: [4]CallLoad = undefined;
    for (b.s.callLoads(&buf)) |l| try t.expectEqual(.queued, l.state);
    // Whisper loads beside Drama 3, higher and still down: both fit, so neither waits for the other.
    b.s.loadKept(.listen, true);
    try t.expectEqual(tool_mod.State.ready, b.state(1));
    b.s.loadKept(.realtime, true);
    for (b.s.callLoads(&buf)) |l| try t.expectEqual(.ready, l.state);
    // Kept while the call is open, however long they are idle.
    b.age(1000);
    b.s.unloadIdle(.realtime);
    b.s.unloadIdle(.listen);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    // The call ends: their TTL counts from its end, not from their last job long before.
    b.calls.n.store(0, .release);
    b.s.callClosed();
    b.s.unloadIdle(.realtime);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    b.age(61);
    b.s.unloadIdle(.realtime);
    b.s.unloadIdle(.listen);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
    try t.expectEqual(tool_mod.State.unloaded, b.state(1));
    try t.expectEqual(@as(i64, 0), b.slots[0].loaded_since_s);
    // Outside a call a call tool is not kept: nothing brings it back until a request or the next call.
    b.s.loadKept(.realtime, true);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
}

test "a call's load that fails or finds no room is reported, and tried again at the next call" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "tts", .kind = .speech, .residency = .call }, .fail = true },
        .{ .cfg = .{ .id = "chat", .kind = .chat, .residency = .call }, .bytes = 200 << 20 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    b.calls.n.store(1, .release);
    b.s.callOpened();
    b.s.loadKept(.realtime, true);
    b.s.loadKept(.chat, true);
    var buf: [4]CallLoad = undefined;
    const loads = b.s.callLoads(&buf);
    try t.expectEqual(.failed, loads[0].state);
    try t.expectEqual(.no_room, loads[1].state);
    fakes[0].fail = false;
    b.calls.n.store(2, .release);
    b.s.callOpened(); // a second call: the failed load's backoff does not hold it back
    try t.expectEqual(.queued, b.s.callLoads(&buf)[0].state);
    b.s.loadKept(.realtime, true);
    try t.expectEqual(.ready, b.s.callLoads(&buf)[0].state);
}

test "on_demand unloads idle_ttl_s after its last job; a TTL change re-arms the timer from now" {
    const t = std.testing;
    var fakes = [_]FakeTool{.{ .cfg = .{ .id = "img", .kind = .image, .idle_ttl_s = 60 } }};
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    try b.s.loadSlot(&b.slots[0]);
    b.age(59);
    b.s.unloadIdle(.main);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    b.age(2);
    try t.expect(b.s.setResidency(0, null, 30) == null); // idle 61 s, TTL now 30: it counts from the change
    b.s.unloadIdle(.main);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    b.age(31);
    b.s.unloadIdle(.main);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
}

test "always to on_demand: no longer kept, it unloads idle_ttl_s after the switch; back to always, it loads" {
    const t = std.testing;
    var fakes = [_]FakeTool{.{ .cfg = .{ .id = "img", .kind = .image, .keep_loaded = true, .idle_ttl_s = 60 } }};
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    b.s.loadKept(.main, false);
    b.age(1000);
    try t.expect(b.s.setResidency(0, .on_demand, null) == null);
    try t.expectEqual(Residency.on_demand, b.s.residency(0).rule);
    b.s.unloadIdle(.main);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    b.age(61);
    b.s.unloadIdle(.main);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
    try t.expect(b.s.setResidency(0, .always, null) == null);
    b.s.loadKept(.main, true);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
}

test "an always set over the budget is refused with its numbers, and nothing changes; call is for the voice loop only" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "chat", .kind = .chat, .keep_loaded = true } },
        .{ .cfg = .{ .id = "img", .kind = .image }, .bytes = 2 << 20 },
        .{ .cfg = .{ .id = "video", .kind = .video } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 3);
    defer b.deinit();
    try t.expectEqual(@as(u64, 1 << 20), b.s.budget().always_bytes);
    try t.expect(b.s.setResidency(1, .always, null) == null); // 1 + 2 MiB: exactly the budget
    const why = b.s.setResidency(2, .always, 10).?;
    try t.expectEqual(@as(u64, 1 << 20), why.over_budget.needs);
    try t.expectEqual(@as(u64, 4 << 20), why.over_budget.always);
    try t.expectEqual(@as(u64, 3 << 20), why.over_budget.ceiling);
    try t.expectEqual(Residency.on_demand, b.s.residency(2).rule);
    try t.expect(b.s.residency(2).ttl_override == null);
    try t.expectEqual(.not_voice, std.meta.activeTag(b.s.setResidency(1, .call, null).?));
    try t.expect(b.s.setResidency(0, .call, null) == null); // chat may
}

test "residency.json: overrides survive a restart; reset drops a tool's entry so its config applies" {
    const t = std.testing;
    const persist = @import("residency.zig");
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "chat", .kind = .chat, .keep_loaded = true, .idle_ttl_s = 600 } },
        .{ .cfg = .{ .id = "tts", .kind = .speech, .keep_loaded = true } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    try t.expect(b.s.setResidency(0, .call, 30) == null);
    try t.expect(b.s.setResidency(1, .disabled, null) == null);
    try persist.save(t.io, t.allocator, dir, &b.s);
    // A restart: fresh slots over the same config read the file.
    var again: Bench = undefined;
    try again.init(&fakes, 100);
    defer again.deinit();
    persist.apply(t.io, t.allocator, dir, again.slots, null);
    try t.expectEqual(Residency.call, again.s.residency(0).rule);
    try t.expectEqual(@as(u32, 30), again.s.residency(0).idle_ttl_s);
    try t.expectEqual(Residency.disabled, again.s.residency(1).rule);
    try t.expectEqual(@as(u32, 120), again.s.residency(1).idle_ttl_s); // not set: the config's
    // Reset the chat model: its entry goes, and the next start uses its config.
    try t.expect(b.s.setResidency(0, null, null) == null);
    try persist.save(t.io, t.allocator, dir, &b.s);
    var third: Bench = undefined;
    try third.init(&fakes, 100);
    defer third.deinit();
    persist.apply(t.io, t.allocator, dir, third.slots, null);
    try t.expectEqual(Residency.always, third.s.residency(0).rule);
    try t.expectEqual(@as(u32, 600), third.s.residency(0).idle_ttl_s);
    try t.expect(third.s.residency(0).rule_override == null);
    try t.expectEqual(Residency.disabled, third.s.residency(1).rule);
    // Reset the disabled one: its config's always again.
    try t.expect(b.s.setResidency(1, null, null) == null);
    try persist.save(t.io, t.allocator, dir, &b.s);
    var fourth: Bench = undefined;
    try fourth.init(&fakes, 100);
    defer fourth.deinit();
    persist.apply(t.io, t.allocator, dir, fourth.slots, null);
    try t.expectEqual(Residency.always, fourth.s.residency(1).rule);
}

test "victims during a call: a call's tools are never taken by a tool outside the voice loop, nor by one another" {
    const T = config.ToolConfig;
    const image: T = .{ .id = "image", .kind = .image, .priority = 50 };
    const stt: T = .{ .id = "stt", .kind = .transcription, .priority = 0, .residency = .call };
    const tts: T = .{ .id = "tts", .kind = .speech, .priority = 95, .residency = .call };
    const kept: T = .{ .id = "kept", .kind = .image, .priority = 99, .keep_loaded = true };
    const t = std.testing;
    const may = Scheduler.mayEvict;
    // Whisper at priority 0 is safe from the image model at 50 while a call keeps it...
    try t.expect(!may(&image, &stt, .call, .{ .call = true }));
    // ...and not once no call is open.
    try t.expect(may(&image, &stt, .no, .{}));
    // A call's tools never unload one another, whatever their priorities.
    try t.expect(!may(&tts, &stt, .call, .{ .below = tts.priority, .for_call = true }));
    // Coming up for a call, Drama 3 may unload an idle on_demand model of any priority, but a kept one only below it.
    try t.expect(may(&tts, &.{ .id = "big", .kind = .image, .priority = 99 }, .no, .{ .below = tts.priority, .for_call = true }));
    try t.expect(!may(&tts, &kept, .always, .{ .below = tts.priority, .for_call = true }));
}

test "lanes load a call's tools as it connects: queued, loading, then ready" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "tts", .kind = .speech, .residency = .call }, .load_ms = 300 },
        .{ .cfg = .{ .id = "stt", .kind = .transcription, .residency = .call }, .load_ms = 300 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var threads: [2]std.Thread = undefined;
    for (&threads, [_]Lane{ .realtime, .listen }) |*th, l| th.* = try std.Thread.spawn(.{}, Scheduler.run, .{ &b.s, l });
    defer for (threads) |th| th.join();
    defer b.s.stop();
    b.calls.n.store(1, .release);
    b.s.callOpened();
    var buf: [4]CallLoad = undefined;
    var saw_loading = false;
    const began = nowNs(t.io);
    while (nowNs(t.io) - began < 5 * std.time.ns_per_s) : (Io.sleep(t.io, .fromMilliseconds(20), .awake) catch {}) {
        var ready: usize = 0;
        for (b.s.callLoads(&buf)) |l| {
            if (l.state == .loading) saw_loading = true;
            if (l.state == .ready) ready += 1;
            try t.expect(l.state != .failed and l.state != .no_room);
        }
        if (ready == 2) break;
    } else return error.TestUnexpectedResult;
    try t.expect(saw_loading);
    // Both loaded at once, not one after the other: well under twice a load.
    try t.expect(nowNs(t.io) - began < 2 * 300 * std.time.ns_per_ms + 500 * std.time.ns_per_ms);
}

/// `Loaded` in a test: counts the warm-ups it was asked for and hands back `job` as the one it queued.
const FakeWarm = struct {
    n: usize = 0,
    job: jobs.Job = .{ .arena = undefined, .id = "warm", .tool = 0, .request = .{ .speech = .{ .input = "Hello." } }, .dir = "", .created_at = 0 },

    fn hook(w: *FakeWarm) Loaded {
        return .{ .ctx = w, .f = f };
    }
    fn f(ctx: *anyopaque, _: usize) ?*const jobs.Job {
        const w: *FakeWarm = @ptrCast(@alignCast(ctx));
        w.n += 1;
        return &w.job;
    }
};

test "every load of a speech tool is warmed, whatever asked for it; a call's tool is ready only once its warm-up ends" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "tts", .kind = .speech, .residency = .call } },
        .{ .cfg = .{ .id = "stt", .kind = .transcription, .residency = .call } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var warm: FakeWarm = .{};
    b.s.loaded = warm.hook();
    // A call's load: the speech tool reports warming until its warm-up job ends; transcription has none.
    b.calls.n.store(1, .release);
    b.s.callOpened();
    b.s.loadKept(.realtime, true);
    b.s.loadKept(.listen, true);
    try t.expectEqual(@as(usize, 1), warm.n);
    var buf: [4]CallLoad = undefined;
    try t.expectEqual(.warming, b.s.callLoads(&buf)[0].state);
    try t.expectEqual(.ready, b.s.callLoads(&buf)[1].state);
    b.s.warmEnded(&b.slots[0], &warm.job);
    try t.expectEqual(.ready, b.s.callLoads(&buf)[0].state);
    // An API load, or a request's, warms it as well; an unload forgets a warm-up still queued.
    b.s.unloadSlot(&b.slots[0]);
    try b.s.loadSlot(&b.slots[0]);
    try t.expectEqual(@as(usize, 2), warm.n);
    b.s.unloadSlot(&b.slots[0]);
    try t.expect(b.slots[0].warm_job == null);
    // One that unloads right after each job (idle_ttl_s 0, not kept) is not warmed: the warm-up would load it again.
    b.calls.n.store(0, .release);
    try t.expect(b.s.setResidency(0, .on_demand, 0) == null);
    try b.s.loadSlot(&b.slots[0]);
    try t.expectEqual(@as(usize, 2), warm.n);
}

test "disabled: never loads (start, call, kept reload), refuses its jobs with 409, and unloads only after its running job" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "img", .kind = .image, .keep_loaded = true } },
        .{ .cfg = .{ .id = "stt", .kind = .transcription, .residency = .disabled } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    b.s.loadKept(.listen, false);
    b.calls.n.store(1, .release);
    b.s.callOpened();
    b.s.loadKept(.listen, true);
    try t.expectEqual(tool_mod.State.unloaded, b.state(1)); // configured disabled: not at start, not for a call
    // The kept image model is running a job when it is disabled: it stays until the job is done, then goes.
    b.s.loadKept(.main, false);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    b.slots[0].in_use = true;
    try t.expect(b.s.setResidency(0, .disabled, null) == null);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    b.s.done(&b.slots[0]);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
    b.s.loadKept(.main, true); // was always: not brought back
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
    // A job queued for it before it was disabled is refused, not loaded.
    var job: jobs.Job = .{ .arena = .init(t.allocator), .id = "j", .tool = 0, .request = .{ .image = .{ .prompt = "x" } }, .dir = "", .created_at = 0 };
    defer job.arena.deinit();
    b.s.runJob(&job);
    try t.expectEqual(@as(u16, 409), job.http_status);
    try t.expectEqualStrings("model_disabled", job.error_type);
    try t.expectEqualStrings("img is disabled in settings", job.error_message);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
    // An API load is refused too.
    var c: Control = .{ .op = .load, .tool = 0 };
    b.s.runControl(&c);
    try t.expect(c.failed);
    // Reset: back to its config (always), and it loads again.
    try t.expect(b.s.setResidency(0, null, null) == null);
    b.s.loadKept(.main, true);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
}

test "disabled during a call: a call's model is unloaded and reported disabled, not waited for" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "tts", .kind = .speech, .residency = .call } },
        .{ .cfg = .{ .id = "stt", .kind = .transcription, .residency = .call } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    b.calls.n.store(1, .release);
    b.s.callOpened();
    b.s.loadKept(.realtime, true);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    try t.expect(b.s.setResidency(0, .disabled, null) == null); // a voice-loop model, mid-call
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
    var buf: [4]CallLoad = undefined;
    const loads = b.s.callLoads(&buf);
    try t.expectEqual(@as(usize, 2), loads.len);
    try t.expectEqual(.disabled, loads[0].state);
    try t.expectEqual(.queued, loads[1].state);
    b.s.loadKept(.realtime, true); // kept by the call no more
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
}

test "a voice model disabled before any call is listed disabled for every call; other kinds are not; booked bytes show unloaded" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "stt", .kind = .transcription, .residency = .disabled }, .bytes = 3 << 20 },
        .{ .cfg = .{ .id = "img", .kind = .image, .residency = .disabled } },
        .{ .cfg = .{ .id = "chat", .kind = .chat, .keep_loaded = true } },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var buf: [4]CallLoad = undefined;
    const loads = b.s.callLoads(&buf);
    try t.expectEqual(@as(usize, 1), loads.len);
    try t.expectEqual(@as(usize, 0), loads[0].tool);
    try t.expectEqual(.disabled, loads[0].state);
    // Not loaded: resident 0, but what it would book is known.
    try t.expectEqual(@as(u64, 0), b.slots[0].resident);
    try t.expectEqual(@as(u64, 3 << 20), b.s.bookedBytes(0));
}

test "a tool running a job is never a victim: a contender that needs its room is refused (503 insufficient_memory), then loads once the job is done" {
    // 2026-11-16: a video model's render and the agent's chat model, each about the whole budget. The journal showed
    // the video model unloaded for the chat model, but only after its worker had died; this pins that a running job's
    // model stays, and the contender is refused at once with a retryable error rather than waiting or evicting it.
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "video", .kind = .video, .priority = 10 }, .bytes = 94 << 20 },
        .{ .cfg = .{ .id = "chat", .kind = .chat, .priority = 100 }, .bytes = 88 << 20 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var load: Control = .{ .op = .load, .tool = 0 };
    b.s.runControl(&load);
    try t.expect(!load.failed);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    // The main lane is running a video job: `runJob` holds the slot in use (`reserve`) until `done`.
    b.s.gate.lockUncancelable(t.io);
    b.s.reserve(&b.slots[0], 0);
    b.s.gate.unlock(t.io);
    var chat: jobs.Job = .{ .arena = .init(t.allocator), .id = "c", .tool = 1, .request = .{ .chat = .{ .body = "{}" } }, .dir = "", .created_at = 0 };
    defer chat.arena.deinit();
    b.s.runJob(&chat);
    try t.expectEqual(jobs.Status.failed, chat.status);
    try t.expectEqual(@as(u16, 503), chat.http_status);
    try t.expectEqualStrings("insufficient_memory", chat.error_type);
    try t.expectEqual(@as(u32, 30), chat.retry_after);
    try t.expectEqual(tool_mod.State.ready, b.state(0)); // the render keeps its model
    try t.expectEqual(tool_mod.State.unloaded, b.state(1));
    // Nor does an API load of the contender take it.
    var load_chat: Control = .{ .op = .load, .tool = 1 };
    b.s.runControl(&load_chat);
    try t.expect(load_chat.failed);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    // The job ends: the idle video model is a victim again, and the retried request loads.
    b.s.done(&b.slots[0]);
    var again: jobs.Job = .{ .arena = .init(t.allocator), .id = "c2", .tool = 1, .request = .{ .chat = .{ .body = "{}" } }, .dir = "", .created_at = 0 };
    defer again.arena.deinit();
    b.s.runJob(&again);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0));
    try t.expectEqual(tool_mod.State.ready, b.state(1));
    try t.expect(!std.mem.eql(u8, again.error_type, "insufficient_memory")); // the fake's generate fails; its load did not
}

fn chatJob(id: []const u8, tool: usize) jobs.Job {
    return .{ .arena = .init(std.testing.allocator), .id = id, .tool = tool, .request = .{ .chat = .{ .body = "{}" } }, .dir = "", .created_at = 0 };
}

test "a tool loading or committing memory for a job is never a victim: a request that cannot fit beside it is refused at once" {
    // 2026-11-17: a chat request evicted minimax-h3 (priority 10) while it was still loading for an image-to-video
    // job. The running-job test only pins `in_use` after the model is ready; this pins the loading and committing
    // phases, and the refusal (not a wait until the client's timeout).
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "minimax-h3", .kind = .video, .priority = 10 }, .bytes = 94 << 20 },
        .{ .cfg = .{ .id = "qwen3.8-flash-next", .kind = .chat, .priority = 100 }, .bytes = 88 << 20 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    // Loading, and not yet `in_use`: the worker has said it is loading, the bytes are admitted.
    fakes[0].st.store(.loading, .release);
    b.slots[0].phase = .loading;
    b.slots[0].pending = 94 << 20;
    var chat = chatJob("c", 1);
    defer chat.arena.deinit();
    const began = nowNs(t.io);
    b.s.runJob(&chat);
    try t.expect(nowNs(t.io) - began < std.time.ns_per_s); // refused, not parked until a timeout
    try t.expectEqual(jobs.Status.failed, chat.status);
    try t.expectEqual(@as(u16, 503), chat.http_status);
    try t.expectEqualStrings("insufficient_memory", chat.error_type);
    try t.expectEqual(@as(u32, 30), chat.retry_after);
    try t.expectEqual(tool_mod.State.loading, b.state(0));
    try t.expectEqual(tool_mod.State.unloaded, b.state(1));
    // Committing: the worker has reported ready, admitted bytes have not settled, `in_use` is still false.
    fakes[0].st.store(.ready, .release);
    b.slots[0].phase = .committing;
    b.slots[0].pending = 94 << 20;
    b.slots[0].in_use = false;
    var chat2 = chatJob("c2", 1);
    defer chat2.arena.deinit();
    const began2 = nowNs(t.io);
    b.s.runJob(&chat2);
    try t.expect(nowNs(t.io) - began2 < std.time.ns_per_s);
    try t.expectEqual(jobs.Status.failed, chat2.status);
    try t.expectEqualStrings("insufficient_memory", chat2.error_type);
    try t.expectEqual(@as(u32, 30), chat2.retry_after);
    try t.expectEqual(tool_mod.State.ready, b.state(0)); // not evicted
    try t.expectEqual(@as(u64, 94 << 20), b.slots[0].pending);
    // A measured load longer than the default raises the hint.
    b.slots[0].last_load_ms = 45_000;
    var chat3 = chatJob("c3", 1);
    defer chat3.arena.deinit();
    b.s.runJob(&chat3);
    try t.expectEqual(@as(u32, 45), chat3.retry_after);
    b.cfg.retry_after_s = 12;
    b.slots[0].last_load_ms = null;
    var chat4 = chatJob("c4", 1);
    defer chat4.arena.deinit();
    b.s.runJob(&chat4);
    try t.expectEqual(@as(u32, 12), chat4.retry_after);
}

test "a request that fits by unloading an idle model still does, and does not unload a tool that is loading" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "video", .kind = .video, .priority = 10 }, .bytes = 20 << 20 },
        .{ .cfg = .{ .id = "image", .kind = .image, .priority = 5 }, .bytes = 70 << 20 },
        .{ .cfg = .{ .id = "chat", .kind = .chat, .priority = 100 }, .bytes = 75 << 20 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var load: Control = .{ .op = .load, .tool = 1 };
    b.s.runControl(&load);
    try t.expectEqual(tool_mod.State.ready, b.state(1));
    fakes[0].st.store(.loading, .release);
    b.slots[0].phase = .loading;
    b.slots[0].pending = 20 << 20;
    var chat = chatJob("c", 2);
    defer chat.arena.deinit();
    b.s.runJob(&chat);
    try t.expectEqual(tool_mod.State.loading, b.state(0)); // the load stays
    try t.expectEqual(@as(u64, 20 << 20), b.slots[0].pending);
    try t.expectEqual(tool_mod.State.unloaded, b.state(1)); // the idle image went
    try t.expectEqual(tool_mod.State.ready, b.state(2)); // the chat loaded
    try t.expect(!std.mem.eql(u8, chat.error_type, "insufficient_memory"));
}

test "usage over the budget recovers: an idle worker that reports more than its estimate is evicted for the next load" {
    // Review of the protect-loading change: the room check took the victims' bytes off the request (`extra -| freeable`),
    // which clamps at 0 and then still asks the over-budget usage to fit. A 60 MiB estimate that loads at 120 MiB on a
    // 100 MiB budget left every later load refused with 503, though the big worker was idle and could go.
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "big", .kind = .image, .priority = 10 }, .bytes = 60 << 20, .loaded_bytes = 120 << 20 },
        .{ .cfg = .{ .id = "small", .kind = .chat, .priority = 100 }, .bytes = 10 << 20 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var load: Control = .{ .op = .load, .tool = 0 };
    b.s.runControl(&load);
    try t.expect(!load.failed);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    try t.expectEqual(@as(u64, 120 << 20), b.slots[0].resident); // over the budget, idle
    b.s.gate.lockUncancelable(t.io);
    try t.expect(b.s.roomIfEvicted(1, 10 << 20, null));
    try t.expectEqual(Scheduler.Admission.proceed, b.s.admission(1, 10 << 20));
    b.s.gate.unlock(t.io);
    var chat = chatJob("c", 1);
    defer chat.arena.deinit();
    b.s.runJob(&chat);
    try t.expect(!std.mem.eql(u8, chat.error_type, "insufficient_memory"));
    try t.expect(chat.http_status != 503);
    try t.expectEqual(tool_mod.State.unloaded, b.state(0)); // the idle over-budget worker went
    try t.expectEqual(tool_mod.State.ready, b.state(1));
}

test "a request that fits once a load's overshoot settles waits, and does not evict the loading tool" {
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "video", .kind = .video, .priority = 10 }, .bytes = 30 << 20 },
        .{ .cfg = .{ .id = "chat", .kind = .chat, .priority = 100 }, .bytes = 40 << 20 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    b.cfg.retry_after_s = 5; // a missed wake fails here, not after the default half minute
    fakes[0].st.store(.loading, .release);
    b.slots[0].phase = .loading;
    b.slots[0].in_use = true;
    b.slots[0].pending = 80 << 20; // admitted high; the estimate (`bytes`) it will hold is 30
    const Ctx = struct {
        s: *Scheduler,
        slot: *Slot,
        fake: *FakeTool,
        fn go(c: *@This()) void {
            Io.sleep(std.testing.io, .fromMilliseconds(80), .awake) catch {};
            c.s.gate.lockUncancelable(std.testing.io);
            // Still loading when we get here: the waiter did not evict it.
            if (c.fake.st.load(.acquire) != .loading) {
                c.s.gate.unlock(std.testing.io);
                return;
            }
            c.fake.st.store(.ready, .release);
            c.slot.resident = c.fake.bytes;
            c.slot.pending = 0;
            c.slot.phase = .running;
            c.s.gate.unlock(std.testing.io);
            c.s.poke();
        }
    };
    var ctx: Ctx = .{ .s = &b.s, .slot = &b.slots[0], .fake = &fakes[0] };
    const th = try std.Thread.spawn(.{}, Ctx.go, .{&ctx});
    defer th.join();
    var chat = chatJob("c", 1);
    defer chat.arena.deinit();
    const began = nowNs(t.io);
    b.s.runJob(&chat);
    try t.expect(nowNs(t.io) - began < 2 * std.time.ns_per_s);
    try t.expectEqual(tool_mod.State.ready, b.state(0));
    try t.expectEqual(@as(u64, 30 << 20), b.slots[0].resident);
    try t.expectEqual(tool_mod.State.ready, b.state(1));
    try t.expect(!std.mem.eql(u8, chat.error_type, "insufficient_memory"));
}

test "retry_after estimates when the blocker frees its memory: a running job from its tool's last job, a load from its last load and job" {
    // 2026-11-18: a 10 s video render held the memory for 16 min while every refused chat request was told to retry
    // in 30 s, so the agent asked about thirty times. The hint now follows the blocker's own history.
    const t = std.testing;
    var fakes = [_]FakeTool{
        .{ .cfg = .{ .id = "video", .kind = .video, .priority = 10 }, .bytes = 94 << 20 },
        .{ .cfg = .{ .id = "chat", .kind = .chat, .priority = 100 }, .bytes = 88 << 20 },
    };
    var b: Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var load: Control = .{ .op = .load, .tool = 0 };
    b.s.runControl(&load);
    b.s.gate.lockUncancelable(t.io);
    b.s.reserve(&b.slots[0], 0);
    // running, no history: the configured floor
    b.slots[0].job_started_ns = nowNs(t.io);
    try t.expectEqual(@as(u32, 30), b.s.retryAfter(1));
    // its last job took 900 s and this one has run 300 s: about 600 s left (the cap)
    b.slots[0].last_job_ms = 900_000;
    b.slots[0].job_started_ns = nowNs(t.io) - 300 * std.time.ns_per_s;
    const left = b.s.retryAfter(1);
    try t.expect(left >= 599 and left <= 600);
    // 240 s of a 300 s job: about 60 s
    b.slots[0].last_job_ms = 300_000;
    b.slots[0].job_started_ns = nowNs(t.io) - 240 * std.time.ns_per_s;
    const near = b.s.retryAfter(1);
    try t.expect(near >= 59 and near <= 61);
    // past its estimate: the floor, never 0
    b.slots[0].job_started_ns = nowNs(t.io) - 400 * std.time.ns_per_s;
    try t.expectEqual(@as(u32, 30), b.s.retryAfter(1));
    // loading: the rest of the load, then the job
    b.slots[0].job_started_ns = 0;
    b.slots[0].phase = .loading;
    b.slots[0].last_load_ms = 60_000;
    b.slots[0].last_job_ms = 120_000;
    b.slots[0].pending_ns = nowNs(t.io) - 20 * std.time.ns_per_s;
    const loading = b.s.retryAfter(1);
    try t.expect(loading >= 159 and loading <= 161);
    // the overshoot wait still counts the load only
    try t.expectEqual(@as(u32, 60), b.s.settleWait(1));
    b.s.gate.unlock(t.io);
    b.s.done(&b.slots[0]);
    try t.expectEqual(@as(i96, 0), b.slots[0].job_started_ns);
}
