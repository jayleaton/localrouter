//! The box side of computer control (docs/dev/COMPUTER.md): the paired desktop helpers, their tasks, and the authority
//! over every step. A helper's socket (`api/desktop.zig`) feeds its frames to `onFrame`; the MCP tools and the voice
//! session start tasks with `control` and `startUse`. Every on-screen step a helper takes is granted here first
//! (`authorize`): the policy decides, the decision provider judges what the lexicon cannot, and consequential steps wait
//! for the user's tap. One task a helper at a time, within the step and working-time budget, both checked here whatever
//! the helper does.

const std = @import("std");
const Io = std.Io;
const Value = std.json.Value;
const policy = @import("policy.zig");
const approvals = @import("approvals.zig");
const audit = @import("audit.zig");
const decide = @import("decide.zig");

pub const protocol_version = 1;

pub const Limits = struct {
    max_steps: u32 = 15,
    timeout_ms: i64 = 90_000, // working time: waiting for a tap does not count
    approval_timeout_ms: i64 = 60_000,
    control_timeout_ms: i64 = 20_000, // a fixed control's reply
};

/// Who asked for a task. `utterance`: the user's own words in the voice turn that asked (empty for MCP).
pub const Requester = struct {
    source: []const u8, // "voice" or "mcp"
    device: []const u8 = "", // the paired device's name
    utterance: []const u8 = "",
};

/// A helper's socket, as the hub sees it. `send` is called without the hub's lock held.
pub const Peer = struct {
    ctx: *anyopaque,
    send: *const fn (ctx: *anyopaque, text: []const u8) bool,
    close: *const fn (ctx: *anyopaque) void,
};

/// Told about a task as it runs (the voice session speaks from it). Called without the hub's lock held.
pub const Listener = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, kind: EventKind, text: []const u8) void,
};

pub const EventKind = enum { progress, approval, finished };

pub const Helper = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    name: []const u8,
    version: []const u8,
    device: []const u8,
    controls: []const []const u8,
    computer_use: bool,
    accessibility: bool,
    screen_recording: bool,
    peer: Peer,
    closed: bool = false,
    refs: u32 = 1, // the socket's own reference; request threads and calls take one each
    rpcs: std.ArrayList(*Rpc) = .empty,
    next_rpc: u64 = 1,
    task: ?*Task = null, // its running task
    connected_ms: i64,
};

const Rpc = struct {
    id: []const u8,
    done: bool = false,
    reply: []const u8 = "", // the whole result frame, owned by the hub's gpa
};

pub const Kind = enum { control, use };

pub const Task = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    kind: Kind,
    helper_id: []const u8,
    helper_name: []const u8,
    goal: []const u8,
    text: ?[]const u8,
    requester: Requester,
    listener: ?Listener = null,
    running: bool = true,
    status: []const u8 = "running", // then done, stuck, blocked, unavailable, stopped, failed
    reason: []const u8 = "",
    surface: []const u8 = "",
    steps: std.ArrayList([]const u8) = .empty,
    started_ms: i64,
    ended_ms: i64 = 0,
    waited_ms: i64 = 0, // time spent waiting for taps
    authorized: u32 = 0,
    decided: u32 = 0,
    stopping: bool = false,
    approval_id: []const u8 = "", // the approval it waits for now
};

const max_kept_tasks = 32;

pub const Hub = struct {
    gpa: std.mem.Allocator,
    io: Io,
    limits: Limits = .{},
    log: *audit.Log,
    provider: ?decide.Provider = null,
    approvals: approvals.Store,
    mutex: Io.Mutex = .init,
    // Held while a task's listener is called and while one is detached, so a listener is never called after `detach`
    // returns. Taken without `mutex` held; a listener may take its own locks, never the hub's.
    listen_mutex: Io.Mutex = .init,
    helpers: std.ArrayList(*Helper) = .empty,
    tasks: std.ArrayList(*Task) = .empty,
    clock_ms: ?*const fn () i64 = null, // tests
    poll_ms: u32 = 50,

    pub fn init(gpa: std.mem.Allocator, io: Io, log: *audit.Log) Hub {
        return .{ .gpa = gpa, .io = io, .log = log, .approvals = .{ .gpa = gpa } };
    }

    pub fn deinit(hub: *Hub) void {
        for (hub.tasks.items) |x| destroyTask(hub.gpa, x);
        hub.tasks.deinit(hub.gpa);
        for (hub.helpers.items) |x| destroyHelper(hub.gpa, x);
        hub.helpers.deinit(hub.gpa);
        hub.approvals.deinit();
    }

    pub fn now(hub: *const Hub) i64 {
        if (hub.clock_ms) |f| return f();
        return @intCast(Io.Clock.real.now(hub.io).toMilliseconds());
    }

    fn lock(hub: *Hub) void {
        hub.mutex.lockUncancelable(hub.io);
    }

    fn unlock(hub: *Hub) void {
        hub.mutex.unlock(hub.io);
    }

    fn sleep(hub: *Hub) void {
        Io.sleep(hub.io, .fromMilliseconds(hub.poll_ms), .awake) catch {};
    }

    fn randomId(hub: *Hub, a: std.mem.Allocator, prefix: []const u8) ![]const u8 {
        var b: [8]u8 = undefined;
        hub.io.random(&b);
        const hex = std.fmt.bytesToHex(b, .lower);
        return std.fmt.allocPrint(a, "{s}_{s}", .{ prefix, &hex });
    }

    // ------------------------------------------------------------ helpers

    pub const Hello = struct {
        type: []const u8,
        protocol: u32,
        helper: struct { name: []const u8 = "Mac", platform: []const u8 = "", version: []const u8 = "" } = .{},
        capabilities: struct { controls: []const []const u8 = &.{}, computer_use: bool = false } = .{},
        access: struct { accessibility: bool = false, screen_recording: bool = false } = .{},
    };

    /// A helper said hello on a paired device's socket: registered, and its `welcome` frame (in `a`) to send back.
    /// Another helper of the same device is replaced (closed). Null with `why` set when the hello is not usable.
    pub fn register(hub: *Hub, a: std.mem.Allocator, text: []const u8, device: []const u8, peer: Peer, why: *decide.Why) ?struct { helper: *Helper, welcome: []const u8 } {
        const hello = std.json.parseFromSliceLeaky(Hello, a, text, .{ .ignore_unknown_fields = true }) catch {
            why.set("the first frame must be hello (docs/dev/COMPUTER.md)", .{});
            return null;
        };
        if (!std.mem.eql(u8, hello.type, "hello")) {
            why.set("the first frame must be hello, not {s}", .{hello.type[0..@min(hello.type.len, 40)]});
            return null;
        }
        if (hello.protocol != protocol_version) {
            why.set("protocol {d} is not offered; this box speaks {d}", .{ hello.protocol, protocol_version });
            return null;
        }
        const x = hub.gpa.create(Helper) catch return null;
        x.* = .{ .arena = .init(hub.gpa), .id = "", .name = "", .version = "", .device = "", .controls = &.{}, .computer_use = hello.capabilities.computer_use, .accessibility = hello.access.accessibility, .screen_recording = hello.access.screen_recording, .peer = peer, .connected_ms = hub.now() };
        const ha = x.arena.allocator();
        const ok = blk: {
            x.id = hub.randomId(ha, "mac") catch break :blk false;
            x.name = ha.dupe(u8, audit.clip(hello.helper.name)) catch break :blk false;
            x.version = ha.dupe(u8, audit.clip(hello.helper.version)) catch break :blk false;
            x.device = ha.dupe(u8, device) catch break :blk false;
            var controls: std.ArrayList([]const u8) = .empty;
            for (hello.capabilities.controls) |c| if (std.meta.stringToEnum(policy.ControlKind, c) != null) controls.append(ha, ha.dupe(u8, c) catch break :blk false) catch break :blk false;
            x.controls = controls.items;
            break :blk true;
        };
        if (!ok) {
            destroyHelper(hub.gpa, x);
            return null;
        }
        var replaced: ?*Helper = null;
        hub.lock();
        for (hub.helpers.items) |old| if (!old.closed and std.mem.eql(u8, old.device, device)) {
            replaced = old;
        };
        hub.helpers.append(hub.gpa, x) catch {
            hub.unlock();
            destroyHelper(hub.gpa, x);
            return null;
        };
        hub.unlock();
        if (replaced) |old| old.peer.close(old.peer.ctx); // its socket's loop then unregisters it
        hub.log.record(hub.gpa, "helper.connected", .{ .helper_id = x.id, .name = x.name, .device = device, .version = x.version, .computer_use = x.computer_use, .accessibility = x.accessibility });
        const welcome = std.json.Stringify.valueAlloc(a, .{ .type = "welcome", .helper_id = x.id, .limits = .{ .max_steps = hub.limits.max_steps, .timeout_ms = hub.limits.timeout_ms, .approval_timeout_ms = hub.limits.approval_timeout_ms } }, .{}) catch return null;
        return .{ .helper = x, .welcome = welcome };
    }

    /// The helper's socket closed: its running task fails, its calls end, its approvals are cancelled.
    pub fn unregister(hub: *Hub, x: *Helper) void {
        hub.lock();
        x.closed = true;
        const task = x.task;
        if (task) |tk| finishLocked(hub, tk, "failed", "The Mac disconnected.", "");
        for (x.rpcs.items) |r| r.done = true;
        hub.unlock();
        if (task) |tk| {
            hub.approvals.cancelTask(hub.io, tk.id);
            hub.notifyFinished(tk);
        }
        hub.log.record(hub.gpa, "helper.disconnected", .{ .helper_id = x.id });
        hub.release(x);
    }

    fn retain(hub: *Hub, x: *Helper) void {
        hub.lock();
        defer hub.unlock();
        x.refs += 1;
    }

    fn release(hub: *Hub, x: *Helper) void {
        hub.lock();
        x.refs -= 1;
        const last = x.refs == 0;
        if (last) for (hub.helpers.items, 0..) |y, i| if (y == x) {
            _ = hub.helpers.orderedRemove(i);
            break;
        };
        hub.unlock();
        if (last) destroyHelper(hub.gpa, x);
    }

    /// The connected helpers' names (for messages and the status view).
    pub fn connected(hub: *Hub) usize {
        hub.lock();
        defer hub.unlock();
        var n: usize = 0;
        for (hub.helpers.items) |x| if (!x.closed) {
            n += 1;
        };
        return n;
    }

    /// Under the lock: the helper `name` names (case-insensitive), or the only one connected when `name` is empty.
    fn pick(hub: *Hub, name: []const u8, why: *decide.Why) ?*Helper {
        var found: ?*Helper = null;
        var n: usize = 0;
        for (hub.helpers.items) |x| {
            if (x.closed) continue;
            n += 1;
            if (name.len == 0 or std.ascii.eqlIgnoreCase(x.name, name)) {
                if (found != null and name.len == 0) {
                    why.set("more than one computer is connected; name one with `computer`", .{});
                    return null;
                }
                found = x;
            }
        }
        if (found == null) {
            if (n == 0) why.set("no computer is connected: start the LocalRouter desktop helper on the Mac (apps/desktop-helper)", .{}) else why.set("no connected computer is named {s}", .{name[0..@min(name.len, 80)]});
        }
        return found;
    }

    // ------------------------------------------------------------ frames from a helper

    /// One frame from helper `x` after its hello. Requests that may wait (decide, authorize) run on their own thread.
    pub fn onFrame(hub: *Hub, x: *Helper, text: []const u8) void {
        var arena: std.heap.ArenaAllocator = .init(hub.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const v = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return hub.sendError(x, "a frame that is not JSON");
        if (v != .object) return hub.sendError(x, "a frame must be a JSON object");
        const typ = str(v.object.get("type")) orelse return hub.sendError(x, "a frame needs a type");
        if (eql(typ, "ping")) {
            _ = x.peer.send(x.peer.ctx, "{\"type\":\"pong\"}");
        } else if (eql(typ, "pong")) {} else if (eql(typ, "result")) {
            hub.resolve(x, v, text);
        } else if (eql(typ, "progress")) {
            hub.progress(x, v);
        } else if (eql(typ, "finished")) {
            hub.finished(x, v);
        } else if (eql(typ, "decide") or eql(typ, "authorize")) {
            const copy = hub.gpa.dupe(u8, text) catch return;
            hub.retain(x);
            const th = std.Thread.spawn(.{}, serveRequest, .{ hub, x, copy }) catch {
                hub.gpa.free(copy);
                hub.release(x);
                return;
            };
            th.detach();
        } else hub.sendError(x, "an unknown frame type");
    }

    fn sendError(hub: *Hub, x: *Helper, msg: []const u8) void {
        var buf: [256]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "{{\"type\":\"error\",\"error\":\"{s}\"}}", .{msg}) catch return;
        _ = hub;
        _ = x.peer.send(x.peer.ctx, text);
    }

    fn serveRequest(hub: *Hub, x: *Helper, text: []u8) void {
        defer hub.gpa.free(text);
        defer hub.release(x);
        var arena: std.heap.ArenaAllocator = .init(hub.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const v = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return;
        const id = v.object.get("id") orelse return hub.sendError(x, "a request needs an id");
        const reply = if (eql(str(v.object.get("type")) orelse "", "decide")) hub.decideFrame(a, x, v) else hub.authorizeFrame(a, x, v);
        var o: std.json.ObjectMap = .empty;
        o.put(a, "type", .{ .string = "result" }) catch return;
        o.put(a, "id", id) catch return;
        var it = reply.object.iterator();
        while (it.next()) |e| o.put(a, e.key_ptr.*, e.value_ptr.*) catch return;
        const out = std.json.Stringify.valueAlloc(a, Value{ .object = o }, .{}) catch return;
        _ = x.peer.send(x.peer.ctx, out);
    }

    fn resolve(hub: *Hub, x: *Helper, v: Value, text: []const u8) void {
        const id = str(v.object.get("id")) orelse return;
        hub.lock();
        defer hub.unlock();
        for (x.rpcs.items) |r| if (!r.done and eql(r.id, id)) {
            r.reply = hub.gpa.dupe(u8, text) catch "";
            r.done = true;
            return;
        };
    }

    fn progress(hub: *Hub, x: *Helper, v: Value) void {
        const step = str(v.object.get("step")) orelse return;
        const tid = str(v.object.get("task_id")) orelse return;
        hub.lock();
        const tk = x.task orelse return hub.unlock();
        if (!eql(tk.id, tid) or !tk.running) return hub.unlock();
        const ta = tk.arena.allocator();
        const s = ta.dupe(u8, audit.clip(step)) catch return hub.unlock();
        tk.steps.append(ta, s) catch {};
        hub.unlock();
        hub.log.record(hub.gpa, "task.progress", .{ .task_id = tid, .step = s });
        hub.emit(tk, .progress, s);
    }

    fn finished(hub: *Hub, x: *Helper, v: Value) void {
        const tid = str(v.object.get("task_id")) orelse return;
        const o = v.object.get("outcome") orelse return;
        if (o != .object) return;
        var outcome = str(o.object.get("status")) orelse "failed";
        const known = [_][]const u8{ "done", "stuck", "blocked", "unavailable", "stopped", "failed" };
        var ok = false;
        for (known) |k| if (eql(k, outcome)) {
            ok = true;
        };
        if (!ok) outcome = "failed";
        hub.lock();
        const tk = x.task orelse return hub.unlock();
        if (!eql(tk.id, tid) or !tk.running) return hub.unlock();
        if (o.object.get("steps")) |steps| if (steps == .array and tk.steps.items.len == 0) for (steps.array.items) |s| if (s == .string) {
            tk.steps.append(tk.arena.allocator(), tk.arena.allocator().dupe(u8, audit.clip(s.string)) catch continue) catch {};
        };
        finishLocked(hub, tk, outcome, str(o.object.get("reason")) orelse "", str(o.object.get("surface")) orelse "");
        hub.unlock();
        hub.approvals.cancelTask(hub.io, tid);
        hub.notifyFinished(tk);
    }

    fn notifyFinished(hub: *Hub, tk: *Task) void {
        hub.log.record(hub.gpa, "task.end", .{ .task_id = tk.id, .status = tk.status, .reason = audit.clip(tk.reason), .steps = tk.steps.items.len, .surface = audit.clip(tk.surface), .ms = hub.now() - tk.started_ms });
        hub.emit(tk, .finished, tk.status);
    }

    // ------------------------------------------------------------ calls to a helper

    /// Sends a request frame (its `id` is added) and waits for the helper's result, at most `timeout_ms`; the whole
    /// result frame in `a`, or null when the helper is gone or did not answer.
    fn call(hub: *Hub, a: std.mem.Allocator, x: *Helper, frame: Value, timeout_ms: i64) ?Value {
        if (frame != .object) return null;
        const r = hub.gpa.create(Rpc) catch return null;
        defer hub.gpa.destroy(r);
        var idbuf: [24]u8 = undefined;
        hub.lock();
        if (x.closed) return unlockNull(hub);
        r.* = .{ .id = std.fmt.bufPrint(&idbuf, "b{d}", .{x.next_rpc}) catch unreachable };
        x.next_rpc += 1;
        x.rpcs.append(x.arena.allocator(), r) catch return unlockNull(hub);
        x.refs += 1;
        hub.unlock();
        defer {
            hub.lock();
            for (x.rpcs.items, 0..) |y, i| if (y == r) {
                _ = x.rpcs.swapRemove(i);
                break;
            };
            hub.unlock();
            if (r.reply.len > 0) hub.gpa.free(r.reply);
            hub.release(x);
        }
        var f = frame.object;
        f.put(a, "id", .{ .string = r.id }) catch return null;
        const text = std.json.Stringify.valueAlloc(a, Value{ .object = f }, .{}) catch return null;
        if (!x.peer.send(x.peer.ctx, text)) return null;
        const deadline = hub.now() + timeout_ms;
        while (true) {
            hub.lock();
            const done = r.done;
            hub.unlock();
            if (done) break;
            if (hub.now() >= deadline) return null;
            hub.sleep();
        }
        if (r.reply.len == 0) return null; // the helper closed
        return std.json.parseFromSliceLeaky(Value, a, r.reply, .{ .allocate = .alloc_always }) catch null;
    }

    fn unlockNull(hub: *Hub) ?Value {
        hub.unlock();
        return null;
    }

    // ------------------------------------------------------------ tasks

    pub const Started = struct { ok: bool, text: []const u8, task_id: []const u8 = "" };

    /// Under the lock: a new task on `x`, in the task list.
    fn newTask(hub: *Hub, kind: Kind, x: *Helper, goal: []const u8, text: ?[]const u8, who: Requester, listener: ?Listener) !*Task {
        const tk = try hub.gpa.create(Task);
        tk.* = .{ .arena = .init(hub.gpa), .id = "", .kind = kind, .helper_id = "", .helper_name = "", .goal = "", .text = null, .requester = .{ .source = "" }, .listener = listener, .started_ms = hub.now() };
        errdefer destroyTask(hub.gpa, tk);
        const ta = tk.arena.allocator();
        tk.id = try hub.randomId(ta, if (kind == .use) "cu" else "cc");
        tk.helper_id = try ta.dupe(u8, x.id);
        tk.helper_name = try ta.dupe(u8, x.name);
        tk.goal = try ta.dupe(u8, goal);
        tk.text = if (text) |s| try ta.dupe(u8, s) else null;
        tk.requester = .{ .source = try ta.dupe(u8, who.source), .device = try ta.dupe(u8, who.device), .utterance = try ta.dupe(u8, who.utterance) };
        hub.pruneTasks();
        try hub.tasks.append(hub.gpa, tk);
        return tk;
    }

    /// Tells the task's listener, if it still has one (never after `detach`).
    fn emit(hub: *Hub, tk: *Task, kind: EventKind, text: []const u8) void {
        hub.listen_mutex.lockUncancelable(hub.io);
        defer hub.listen_mutex.unlock(hub.io);
        if (tk.listener) |l| l.event(l.ctx, kind, text);
    }

    /// The task's listener is going away: it is not called again once this returns.
    pub fn detach(hub: *Hub, task_id: []const u8) void {
        hub.listen_mutex.lockUncancelable(hub.io);
        defer hub.listen_mutex.unlock(hub.io);
        hub.lock();
        defer hub.unlock();
        for (hub.tasks.items) |tk| if (eql(tk.id, task_id)) {
            tk.listener = null;
        };
    }

    /// Finished tasks beyond the newest few, a minute after they ended (a caller may still be reading one).
    fn pruneTasks(hub: *Hub) void {
        var i: usize = 0;
        const now_ms = hub.now();
        while (hub.tasks.items.len >= max_kept_tasks and i < hub.tasks.items.len) {
            const x = hub.tasks.items[i];
            if (!x.running and now_ms - x.ended_ms > 60_000) {
                destroyTask(hub.gpa, hub.tasks.orderedRemove(i));
            } else i += 1;
        }
    }

    /// A requester may start a task: a voice turn must carry the user's own words; the voice request alone never grants
    /// more than one bounded task.
    fn admit(who: Requester, why: *decide.Why) bool {
        if (eql(who.source, "voice") and std.mem.trim(u8, who.utterance, " \t\r\n").len == 0) {
            why.set("no user request in this voice turn", .{});
            return false;
        }
        if (!eql(who.source, "voice") and !eql(who.source, "mcp")) {
            why.set("unknown requester", .{});
            return false;
        }
        return true;
    }

    /// A fixed control (`{"kind": "open_app", "app": "Safari"}`, ...): the policy, a tap when it asks for one, then the
    /// helper runs it. The result's text is the helper's sentence for the user, or why it did not run.
    pub fn control(hub: *Hub, a: std.mem.Allocator, who: Requester, computer: []const u8, ctl: Value) Started {
        var why: decide.Why = .{};
        if (!admit(who, &why)) return .{ .ok = false, .text = a.dupe(u8, why.message) catch "" };
        if (ctl != .object) return .{ .ok = false, .text = "control must be an object with a kind" };
        const kind = str(ctl.object.get("kind")) orelse return .{ .ok = false, .text = "control needs a kind" };
        const d = policy.control(kind);
        if (d.verdict == .refused) return .{ .ok = false, .text = d.why };
        const summary = describeControl(a, ctl);
        hub.lock();
        const x = hub.pick(computer, &why) orelse {
            hub.unlock();
            return .{ .ok = false, .text = a.dupe(u8, why.message) catch "" };
        };
        var offered = false;
        for (x.controls) |c| if (eql(c, kind)) {
            offered = true;
        };
        if (!offered) {
            hub.unlock();
            return .{ .ok = false, .text = std.fmt.allocPrint(a, "{s} does not offer {s}", .{ x.name, kind }) catch "" };
        }
        if (x.task != null) {
            hub.unlock();
            return .{ .ok = false, .text = std.fmt.allocPrint(a, "{s} is busy with another task; stop it first", .{x.name}) catch "" };
        }
        const tk = hub.newTask(.control, x, summary, null, who, null) catch {
            hub.unlock();
            return .{ .ok = false, .text = "out of memory" };
        };
        x.task = tk;
        x.refs += 1;
        hub.unlock();
        defer hub.release(x);
        hub.log.record(hub.gpa, "task.start", .{ .task_id = tk.id, .kind = "control", .helper_id = x.id, .control = kind, .goal = audit.clip(summary), .requester = who.source, .device = who.device });
        if (d.verdict == .tap) {
            const b: approvals.Binding = .{ .helper_id = x.id, .task_id = tk.id, .step = 0, .action = .{ .kind = "control", .name = kind }, .why = d.why, .goal = summary, .requester = who.source };
            const g = hub.awaitTap(tk, b);
            if (!g.ok) return hub.endControl(tk, x, "blocked", g.reason);
        }
        var f: std.json.ObjectMap = .empty;
        f.put(a, "type", .{ .string = "control" }) catch return .{ .ok = false, .text = "out of memory" };
        f.put(a, "task_id", .{ .string = tk.id }) catch return .{ .ok = false, .text = "out of memory" };
        f.put(a, "control", ctl) catch return .{ .ok = false, .text = "out of memory" };
        const reply = hub.call(a, x, .{ .object = f }, hub.limits.control_timeout_ms) orelse return hub.endControl(tk, x, "failed", "The Mac did not answer.");
        if (reply.object.get("ok")) |ok| if (ok == .bool and ok.bool) {
            const done = str(reply.object.get("done")) orelse "Done.";
            const r = hub.endControl(tk, x, "done", done);
            return .{ .ok = true, .text = a.dupe(u8, r.text) catch "", .task_id = r.task_id };
        };
        return hub.endControl(tk, x, "failed", str(reply.object.get("error")) orelse "The Mac could not do it.");
    }

    fn endControl(hub: *Hub, tk: *Task, x: *Helper, outcome: []const u8, reason: []const u8) Started {
        hub.lock();
        finishLocked(hub, tk, outcome, reason, x.name);
        hub.unlock();
        hub.notifyFinished(tk);
        return .{ .ok = eql(outcome, "done"), .text = tk.reason, .task_id = tk.id };
    }

    /// Starts a computer-use task on the named (or only) helper; it runs on the helper and ends with its `finished`.
    pub fn startUse(hub: *Hub, a: std.mem.Allocator, who: Requester, computer: []const u8, goal: []const u8, text: ?[]const u8, listener: ?Listener) Started {
        var why: decide.Why = .{};
        if (!admit(who, &why)) return .{ .ok = false, .text = a.dupe(u8, why.message) catch "" };
        const g = std.mem.trim(u8, goal, " \t\r\n");
        if (g.len == 0 or g.len > 1000) return .{ .ok = false, .text = "goal is required (at most 1000 characters)" };
        if (text) |s| if (s.len == 0 or s.len > policy.max_text) return .{ .ok = false, .text = "text is at most 4000 characters" };
        if (hub.provider == null) return .{ .ok = false, .text = "computer use needs a decision model (the chat model): none is configured" };
        hub.lock();
        const x = hub.pick(computer, &why) orelse {
            hub.unlock();
            return .{ .ok = false, .text = a.dupe(u8, why.message) catch "" };
        };
        if (!x.computer_use) {
            hub.unlock();
            return .{ .ok = false, .text = std.fmt.allocPrint(a, "{s} cannot read its screen: give the helper Accessibility access in System Settings", .{x.name}) catch "" };
        }
        if (x.task != null) {
            hub.unlock();
            return .{ .ok = false, .text = std.fmt.allocPrint(a, "{s} is busy with another task; stop it first", .{x.name}) catch "" };
        }
        const tk = hub.newTask(.use, x, g, text, who, listener) catch {
            hub.unlock();
            return .{ .ok = false, .text = "out of memory" };
        };
        x.task = tk;
        x.refs += 1;
        hub.unlock();
        defer hub.release(x);
        hub.log.record(hub.gpa, "task.start", .{ .task_id = tk.id, .kind = "use", .helper_id = x.id, .goal = audit.clip(g), .text = if (text) |s| audit.digest(s) else null, .requester = who.source, .device = who.device });
        const frame = std.json.parseFromSliceLeaky(Value, a, std.json.Stringify.valueAlloc(a, .{ .type = "use.start", .task_id = tk.id, .goal = g, .text = text, .limits = .{ .max_steps = hub.limits.max_steps, .timeout_ms = hub.limits.timeout_ms, .approval_timeout_ms = hub.limits.approval_timeout_ms } }, .{}) catch return .{ .ok = false, .text = "out of memory" }, .{}) catch return .{ .ok = false, .text = "out of memory" };
        const reply = hub.call(a, x, frame, hub.limits.control_timeout_ms);
        const ok = if (reply) |r| (if (r.object.get("ok")) |o| o == .bool and o.bool else false) else false;
        if (!ok) {
            const reason = if (reply) |r| str(r.object.get("error")) orelse "The Mac refused the task." else "The Mac did not answer.";
            hub.lock();
            if (tk.running) finishLocked(hub, tk, "failed", reason, "");
            hub.unlock();
            hub.notifyFinished(tk);
            return .{ .ok = false, .text = tk.reason, .task_id = tk.id };
        }
        return .{ .ok = true, .text = "started", .task_id = tk.id };
    }

    /// Stops a task (any, when `task_id` is empty: every running one). The helper stops before its next step; a
    /// pending approval for it can no longer be granted.
    pub fn stop(hub: *Hub, a: std.mem.Allocator, task_id: []const u8, by: []const u8) usize {
        var targets: std.ArrayList(struct { x: *Helper, id: []const u8 }) = .empty;
        hub.lock();
        for (hub.helpers.items) |x| if (x.task) |tk| if (tk.running and (task_id.len == 0 or eql(tk.id, task_id))) {
            tk.stopping = true;
            x.refs += 1;
            targets.append(a, .{ .x = x, .id = a.dupe(u8, tk.id) catch "" }) catch {
                x.refs -= 1;
            };
        };
        hub.unlock();
        for (targets.items) |t| {
            hub.approvals.cancelTask(hub.io, t.id);
            hub.log.record(hub.gpa, "task.stop", .{ .task_id = t.id, .by = by });
            const text = std.json.Stringify.valueAlloc(a, .{ .type = "stop", .task_id = t.id, .id = "stop" }, .{}) catch "";
            if (text.len > 0) _ = t.x.peer.send(t.x.peer.ctx, text);
            hub.release(t.x);
        }
        return targets.items.len;
    }

    pub const TaskView = struct { task_id: []const u8, kind: []const u8, computer: []const u8, goal: []const u8, status: []const u8, reason: []const u8, surface: []const u8, steps: []const []const u8, waiting_for_approval: ?[]const u8 = null, ms: i64 };

    /// A task's state, copied into `a`; null when the hub no longer knows it.
    pub fn view(hub: *Hub, a: std.mem.Allocator, task_id: []const u8) ?TaskView {
        hub.lock();
        defer hub.unlock();
        for (hub.tasks.items) |tk| if (eql(tk.id, task_id)) return viewLocked(hub, a, tk) catch null;
        return null;
    }

    fn viewLocked(hub: *Hub, a: std.mem.Allocator, tk: *Task) !TaskView {
        const steps = try a.alloc([]const u8, tk.steps.items.len);
        for (steps, tk.steps.items) |*o, s| o.* = try a.dupe(u8, s);
        return .{ .task_id = try a.dupe(u8, tk.id), .kind = @tagName(tk.kind), .computer = try a.dupe(u8, tk.helper_name), .goal = try a.dupe(u8, tk.goal), .status = try a.dupe(u8, tk.status), .reason = try a.dupe(u8, tk.reason), .surface = try a.dupe(u8, tk.surface), .steps = steps, .waiting_for_approval = if (tk.approval_id.len > 0) try a.dupe(u8, tk.approval_id) else null, .ms = hub.now() - tk.started_ms };
    }

    /// Waits until the task ends or `timeout_ms` passes; its state then.
    pub fn wait(hub: *Hub, a: std.mem.Allocator, task_id: []const u8, timeout_ms: i64) ?TaskView {
        const deadline = hub.now() + timeout_ms;
        while (true) {
            hub.lock();
            var running = false;
            var known = false;
            for (hub.tasks.items) |tk| if (eql(tk.id, task_id)) {
                known = true;
                running = tk.running;
            };
            hub.unlock();
            if (!known) return null;
            if (!running or hub.now() >= deadline) return hub.view(a, task_id);
            hub.sleep();
        }
    }

    pub const HelperView = struct { helper_id: []const u8, name: []const u8, device: []const u8, version: []const u8, controls: []const []const u8, computer_use: bool, accessibility: bool, screen_recording: bool, busy: ?[]const u8, connected_s: i64 };

    /// The connected helpers and the recent tasks, newest first, copied into `a`.
    pub fn status(hub: *Hub, a: std.mem.Allocator) !struct { computers: []HelperView, tasks: []TaskView } {
        hub.lock();
        defer hub.unlock();
        var hs: std.ArrayList(HelperView) = .empty;
        for (hub.helpers.items) |x| if (!x.closed) {
            const controls = try a.alloc([]const u8, x.controls.len);
            for (controls, x.controls) |*o, c| o.* = try a.dupe(u8, c);
            try hs.append(a, .{ .helper_id = try a.dupe(u8, x.id), .name = try a.dupe(u8, x.name), .device = try a.dupe(u8, x.device), .version = try a.dupe(u8, x.version), .controls = controls, .computer_use = x.computer_use, .accessibility = x.accessibility, .screen_recording = x.screen_recording, .busy = if (x.task) |tk| try a.dupe(u8, tk.id) else null, .connected_s = @divTrunc(hub.now() - x.connected_ms, 1000) });
        };
        var ts: std.ArrayList(TaskView) = .empty;
        var i = hub.tasks.items.len;
        while (i > 0) {
            i -= 1;
            try ts.append(a, try viewLocked(hub, a, hub.tasks.items[i]));
        }
        return .{ .computers = hs.items, .tasks = ts.items };
    }

    // ------------------------------------------------------------ decide and authorize

    /// Under the lock: the helper's running use task named `tid`, or null with `why` set.
    fn ownTask(x: *Helper, tid: []const u8, why: *decide.Why) ?*Task {
        const tk = x.task orelse {
            why.set("no task is running on this computer", .{});
            return null;
        };
        if (!eql(tk.id, tid) or tk.kind != .use) {
            why.set("that task is not this computer's running task", .{});
            return null;
        }
        if (!tk.running or tk.stopping) {
            why.set("the task was stopped", .{});
            return null;
        }
        return tk;
    }

    fn workedMs(hub: *const Hub, tk: *const Task) i64 {
        return hub.now() - tk.started_ms - tk.waited_ms;
    }

    fn refusal(a: std.mem.Allocator, field: []const u8, msg: []const u8) Value {
        var o: std.json.ObjectMap = .empty;
        o.put(a, "ok", .{ .bool = false }) catch {};
        o.put(a, field, .{ .string = a.dupe(u8, msg) catch "" }) catch {};
        if (!eql(field, "error")) o.put(a, "error", .{ .string = a.dupe(u8, msg) catch "" }) catch {};
        return .{ .object = o };
    }

    pub fn decideFrame(hub: *Hub, a: std.mem.Allocator, x: *Helper, v: Value) Value {
        var why: decide.Why = .{};
        const tid = str(v.object.get("task_id")) orelse "";
        hub.lock();
        const tk = ownTask(x, tid, &why) orelse {
            hub.unlock();
            return refusal(a, "error", why.message);
        };
        // A step can ask a surface question and a next-step question; anything beyond that is a runaway loop.
        if (tk.decided >= 2 * hub.limits.max_steps + 4) {
            hub.unlock();
            return refusal(a, "error", "the task used up its decisions");
        }
        if (hub.workedMs(tk) > hub.limits.timeout_ms) {
            hub.unlock();
            return refusal(a, "error", "It was taking too long.");
        }
        tk.decided += 1;
        hub.unlock();
        const provider = hub.provider orelse return refusal(a, "error", "no decision model is configured");
        const questions = (decide.parseQuestions(a, v.object.get("questions") orelse .null, &why) catch return refusal(a, "error", "out of memory")) orelse return refusal(a, "error", why.message);
        const state = std.json.Stringify.valueAlloc(a, v.object.get("state") orelse Value{ .object = .empty }, .{}) catch return refusal(a, "error", "out of memory");
        if (state.len > 64 << 10) return refusal(a, "error", "the state is larger than 64 KiB");
        const answers = provider.ask(a, state, questions, &why) catch return refusal(a, "error", if (why.message.len > 0) why.message else "the decision failed");
        var picks: std.ArrayList(struct { id: []const u8, choice: []const u8, p: f64 }) = .empty;
        for (answers) |ans| picks.append(a, .{ .id = ans.id, .choice = ans.choice, .p = ans.probability }) catch {};
        hub.log.record(hub.gpa, "decide", .{ .task_id = tid, .provider = provider.name, .answers = picks.items });
        var o: std.json.ObjectMap = .empty;
        o.put(a, "ok", .{ .bool = true }) catch {};
        o.put(a, "answers", decide.answersValue(a, answers) catch .null) catch {};
        return .{ .object = o };
    }

    const AuthFrame = struct {
        task_id: []const u8,
        step: u32 = 0,
        observation: struct { app: []const u8 = "", title: []const u8 = "", url: []const u8 = "" } = .{},
        action: struct {
            kind: []const u8,
            ref: []const u8 = "",
            role: []const u8 = "",
            label: []const u8 = "",
            text: []const u8 = "",
            key: []const u8 = "",
            field: ?struct { role: []const u8 = "", label: []const u8 = "" } = null,
            direction: []const u8 = "",
            what: []const u8 = "",
            name: []const u8 = "",
        },
    };

    fn authorizeFrame(hub: *Hub, a: std.mem.Allocator, x: *Helper, v: Value) Value {
        const f = std.json.parseFromValueLeaky(AuthFrame, a, v, .{ .ignore_unknown_fields = true }) catch return refusal(a, "denied", "an authorize request needs task_id and an action");
        const act: policy.Action = .{ .kind = f.action.kind, .ref = f.action.ref, .role = f.action.role, .label = f.action.label, .text = f.action.text, .key = f.action.key, .field_role = if (f.action.field) |fl| fl.role else "", .field_label = if (f.action.field) |fl| fl.label else "", .direction = f.action.direction, .what = f.action.what, .name = f.action.name };
        const g = hub.authorize(x, f.task_id, f.step, .{ .app = f.observation.app, .title = f.observation.title, .url = f.observation.url }, act);
        if (!g.ok) return refusal(a, "denied", g.reason);
        var o: std.json.ObjectMap = .empty;
        o.put(a, "ok", .{ .bool = true }) catch {};
        o.put(a, "grant", .{ .string = if (g.approved) "approved" else "allow" }) catch {};
        return .{ .object = o };
    }

    pub const Observed = struct { app: []const u8 = "", title: []const u8 = "", url: []const u8 = "" };
    pub const Grant = struct { ok: bool, approved: bool = false, reason: []const u8 = "" };

    /// One on-screen step: within the task's budget, by the policy, the provider's judgment where it asks for one,
    /// and the user's tap where the step is consequential. `reason` is static or in the task's arena.
    pub fn authorize(hub: *Hub, x: *Helper, task_id: []const u8, step: u32, obs: Observed, act: policy.Action) Grant {
        var why: decide.Why = .{};
        hub.lock();
        const tk = ownTask(x, task_id, &why) orelse {
            hub.unlock();
            return .{ .ok = false, .reason = "the task was stopped" };
        };
        if (tk.authorized >= hub.limits.max_steps + 1) { // the surface, then at most max_steps actions
            hub.unlock();
            return hub.deny(tk, step, act, "It did not finish within its steps.");
        }
        if (hub.workedMs(tk) > hub.limits.timeout_ms) {
            hub.unlock();
            return hub.deny(tk, step, act, "It was taking too long.");
        }
        tk.authorized += 1;
        hub.unlock();
        var d = policy.screen(act);
        var p: ?f64 = null;
        if (d.verdict == .judge) {
            p = hub.judge(tk, obs, act);
            d = policy.afterJudge(p);
        }
        switch (d.verdict) {
            .refused => return hub.deny(tk, step, act, d.why),
            .auto => {
                hub.log.record(hub.gpa, "grant", .{ .task_id = task_id, .step = step, .action = actionLog(act), .app = audit.clip(obs.app), .judged = p });
                return .{ .ok = true };
            },
            .tap, .judge => {
                const b: approvals.Binding = .{ .helper_id = x.id, .task_id = task_id, .step = step, .app = obs.app, .title = obs.title, .url = obs.url, .action = act, .why = d.why, .goal = tk.goal, .requester = tk.requester.source };
                const g = hub.awaitTap(tk, b);
                if (!g.ok) return g;
                hub.log.record(hub.gpa, "grant", .{ .task_id = task_id, .step = step, .action = actionLog(act), .app = audit.clip(obs.app), .approved = true });
                return g;
            },
        }
    }

    fn deny(hub: *Hub, tk: *Task, step: u32, act: policy.Action, reason: []const u8) Grant {
        hub.log.record(hub.gpa, "deny", .{ .task_id = tk.id, .step = step, .action = actionLog(act), .reason = reason });
        return .{ .ok = false, .reason = reason };
    }

    /// The provider's yes/no: could this step send, buy, delete, post or submit something? Null when it cannot say.
    fn judge(hub: *Hub, tk: *Task, obs: Observed, act: policy.Action) ?f64 {
        const provider = hub.provider orelse return null;
        var arena: std.heap.ArenaAllocator = .init(hub.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const state = std.json.Stringify.valueAlloc(a, .{ .goal = tk.goal, .screen = .{ .app = obs.app, .title = obs.title, .url = obs.url }, .action = .{ .kind = act.kind, .role = act.role, .label = act.label } }, .{}) catch return null;
        const q = [_]decide.Question{.{ .id = "consequential", .kind = .noul, .instructions = "If the program does this action on this screen, could it send a message or email, post or publish something, buy or pay for something, delete or remove something, or submit a form? Answer yes if it could, or if you cannot tell." }};
        var why: decide.Why = .{};
        const answers = provider.ask(a, state, &q, &why) catch return null;
        for (answers) |ans| if (eql(ans.id, "consequential")) return ans.probability;
        return null;
    }

    /// Creates the approval, tells the requester, and waits for the tap (not counted as working time). Deny, expiry,
    /// a stop or the helper going away ends the step.
    fn awaitTap(hub: *Hub, tk: *Task, b: approvals.Binding) Grant {
        var idb: [12]u8 = undefined;
        hub.io.random(&idb);
        const start = hub.now();
        const id = hub.approvals.create(hub.io, b, start, idb) catch return .{ .ok = false, .reason = "too many approvals are waiting" };
        const digest = b.digest();
        const ta = tk.arena.allocator();
        const card = describeAction(ta, b) catch "an action";
        hub.lock();
        tk.approval_id = ta.dupe(u8, id) catch "";
        hub.unlock();
        hub.log.record(hub.gpa, "approval.request", .{ .task_id = tk.id, .approval_id = id, .step = b.step, .action = actionLog(b.action), .app = audit.clip(b.app), .why = b.why });
        hub.emit(tk, .approval, card);
        var outcome: approvals.Store.Taken = .pending;
        while (outcome == .pending) {
            outcome = hub.approvals.take(hub.io, id, digest, hub.now());
            if (outcome != .pending) break;
            hub.lock();
            const gone = tk.stopping or !tk.running;
            hub.unlock();
            if (gone) {
                hub.approvals.cancelTask(hub.io, tk.id);
                outcome = .cancelled;
                break;
            }
            hub.sleep();
        }
        hub.lock();
        tk.waited_ms += hub.now() - start;
        tk.approval_id = "";
        hub.unlock();
        hub.log.record(hub.gpa, "approval.end", .{ .task_id = tk.id, .approval_id = id, .outcome = @tagName(outcome) });
        return switch (outcome) {
            .approved => .{ .ok = true, .approved = true },
            .denied => .{ .ok = false, .reason = "You said no on the approval screen." },
            .expired => .{ .ok = false, .reason = "Nobody approved it in time." },
            .cancelled => .{ .ok = false, .reason = "It was stopped." },
            else => .{ .ok = false, .reason = "The approval did not match this step." },
        };
    }
};

/// Under the lock: the task's end. A helper's task slot is freed.
fn finishLocked(hub: *Hub, tk: *Task, status: []const u8, reason: []const u8, surface: []const u8) void {
    if (!tk.running) return;
    const ta = tk.arena.allocator();
    tk.running = false;
    tk.ended_ms = hub.now();
    tk.status = ta.dupe(u8, status) catch "failed";
    tk.reason = ta.dupe(u8, reason[0..@min(reason.len, 600)]) catch "";
    if (surface.len > 0) tk.surface = ta.dupe(u8, audit.clip(surface)) catch "";
    for (hub.helpers.items) |x| if (x.task == tk) {
        x.task = null;
    };
}

fn destroyHelper(gpa: std.mem.Allocator, x: *Helper) void {
    x.arena.deinit();
    gpa.destroy(x);
}

fn destroyTask(gpa: std.mem.Allocator, tk: *Task) void {
    tk.arena.deinit();
    gpa.destroy(tk);
}

/// An action for the audit log: its kind and target, typed text as length and digest only.
fn actionLog(act: policy.Action) struct { kind: []const u8, role: []const u8, label: []const u8, key: []const u8, name: []const u8, text: ?audit.Digest } {
    return .{ .kind = act.kind, .role = act.role, .label = audit.clip(act.label), .key = act.key, .name = audit.clip(act.name), .text = if (act.text.len > 0) audit.digest(act.text) else null };
}

/// The card's sentence for an approval: what exactly would happen, where.
pub fn describeAction(a: std.mem.Allocator, b: approvals.Binding) ![]const u8 {
    const act = b.action;
    const where = if (b.app.len > 0) b.app else "the computer";
    if (eql(act.kind, "type")) return std.fmt.allocPrint(a, "type “{s}” into {s} in {s}", .{ act.text, nonEmpty(act.label, "a text box"), where });
    if (eql(act.kind, "press")) return std.fmt.allocPrint(a, "press {s} in {s} ({s})", .{ act.key, nonEmpty(act.field_label, "the focused field"), where });
    if (eql(act.kind, "click")) return std.fmt.allocPrint(a, "click “{s}” in {s}", .{ nonEmpty(act.label, act.role), where });
    if (eql(act.kind, "control")) return std.fmt.allocPrint(a, "{s} on {s}", .{ b.goal, nonEmpty(b.app, "the Mac") });
    return std.fmt.allocPrint(a, "{s} in {s}", .{ act.kind, where });
}

/// A fixed control in words ("open Safari", "set the volume to 30").
pub fn describeControl(a: std.mem.Allocator, ctl: Value) []const u8 {
    const kind = str(ctl.object.get("kind")) orelse "";
    const app = str(ctl.object.get("app")) orelse "";
    const action = str(ctl.object.get("action")) orelse "";
    if (eql(kind, "open_app")) return std.fmt.allocPrint(a, "open {s}", .{app}) catch kind;
    if (eql(kind, "quit_app")) return std.fmt.allocPrint(a, "quit {s}", .{app}) catch kind;
    if (eql(kind, "volume")) return std.fmt.allocPrint(a, "volume {s}", .{action}) catch kind;
    if (eql(kind, "media")) return std.fmt.allocPrint(a, "{s} on {s}", .{ action, str(ctl.object.get("player")) orelse "the player" }) catch kind;
    if (eql(kind, "dark_mode")) return std.fmt.allocPrint(a, "dark mode {s}", .{action}) catch kind;
    if (eql(kind, "lock_screen")) return "lock the screen";
    return kind;
}

fn nonEmpty(s: []const u8, other: []const u8) []const u8 {
    return if (s.len > 0) s else other;
}

fn str(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

test {
    _ = @import("hub_test.zig");
}
