//! The hub with an in-process fake helper (its frames answered as a helper would), a fake decision provider and a
//! fake clock: budgets, the policy, approvals and the task lifecycle as the box enforces them.

const std = @import("std");
const t = std.testing;
const Value = std.json.Value;
const hub_mod = @import("hub.zig");
const audit = @import("audit.zig");
const decide = @import("decide.zig");
const approvals = @import("approvals.zig");

const Hub = hub_mod.Hub;

var clock: std.atomic.Value(i64) = .init(1_000_000);
fn fakeNow() i64 {
    return clock.load(.acquire);
}

/// A helper that answers `control` and `use.start` at once, and records every frame it was sent.
const FakeHelper = struct {
    hub: *Hub,
    helper: ?*hub_mod.Helper = null,
    frames: std.ArrayList([]u8) = .empty,
    closed: bool = false,
    refuse: bool = false,
    mutex: std.Io.Mutex = .init,

    fn peer(f: *FakeHelper) hub_mod.Peer {
        return .{ .ctx = f, .send = send, .close = close };
    }

    fn send(ctx: *anyopaque, text: []const u8) bool {
        const f: *FakeHelper = @ptrCast(@alignCast(ctx));
        f.mutex.lockUncancelable(t.io);
        f.frames.append(t.allocator, t.allocator.dupe(u8, text) catch return false) catch return false;
        f.mutex.unlock(t.io);
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const v = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return true;
        const typ = v.object.get("type").?.string;
        const x = f.helper orelse return true;
        if (std.mem.eql(u8, typ, "control") or std.mem.eql(u8, typ, "use.start")) {
            const id = v.object.get("id").?.string;
            const reply = if (f.refuse)
                std.fmt.allocPrint(a, "{{\"type\":\"result\",\"id\":\"{s}\",\"ok\":false,\"error\":\"Computer control is switched off on this Mac\"}}", .{id}) catch return false
            else
                std.fmt.allocPrint(a, "{{\"type\":\"result\",\"id\":\"{s}\",\"ok\":true,\"done\":\"Opened Safari\"}}", .{id}) catch return false;
            f.hub.onFrame(x, reply);
        }
        return true;
    }

    fn close(ctx: *anyopaque) void {
        const f: *FakeHelper = @ptrCast(@alignCast(ctx));
        f.closed = true;
    }

    fn deinit(f: *FakeHelper) void {
        for (f.frames.items) |x| t.allocator.free(x);
        f.frames.deinit(t.allocator);
    }

    fn sawType(f: *FakeHelper, typ: []const u8) bool {
        f.mutex.lockUncancelable(t.io);
        defer f.mutex.unlock(t.io);
        var buf: [64]u8 = undefined;
        const needle = std.fmt.bufPrint(&buf, "\"type\":\"{s}\"", .{typ}) catch return false;
        for (f.frames.items) |x| if (std.mem.indexOf(u8, x, needle) != null) return true;
        return false;
    }
};

/// Answers every Choice with its first criterion and every Noul with `noul_p`.
const FakeProvider = struct {
    noul_p: f64 = 0.05,
    calls: u32 = 0,
    choice_override: []const u8 = "",

    fn provider(p: *FakeProvider) decide.Provider {
        return .{ .ptr = p, .name = "fake", .askFn = ask };
    }

    fn ask(ptr: *anyopaque, a: std.mem.Allocator, _: []const u8, questions: []const decide.Question, _: *decide.Why) decide.Error![]decide.Answer {
        const p: *FakeProvider = @ptrCast(@alignCast(ptr));
        p.calls += 1;
        var out: std.ArrayList(decide.Answer) = .empty;
        for (questions) |q| switch (q.kind) {
            .choice => try out.append(a, .{ .id = q.id, .kind = .choice, .choice = if (p.choice_override.len > 0) p.choice_override else q.criteria[0].key, .probability = 0.9 }),
            .noul => try out.append(a, .{ .id = q.id, .kind = .noul, .probability = p.noul_p }),
        };
        return out.items;
    }
};

const Fixture = struct {
    log: audit.Log,
    hub: Hub,
    fake: FakeHelper,
    provider: FakeProvider = .{},
    arena: std.heap.ArenaAllocator,

    fn init(f: *Fixture) !void {
        f.provider = .{}; // `var f: Fixture = undefined` leaves field defaults unset
        f.log = .{ .io = t.io, .path = "" };
        f.hub = Hub.init(t.allocator, t.io, &f.log);
        f.hub.clock_ms = fakeNow;
        f.hub.poll_ms = 5;
        f.hub.provider = f.provider.provider();
        f.fake = .{ .hub = &f.hub };
        f.arena = .init(t.allocator);
        var why: decide.Why = .{};
        const r = f.hub.register(f.arena.allocator(), hello, "JJ's Mac", f.fake.peer(), &why) orelse return error.NotRegistered;
        f.fake.helper = r.helper;
    }

    fn deinit(f: *Fixture) void {
        if (f.fake.helper) |x| f.hub.unregister(x);
        f.hub.deinit();
        f.fake.deinit();
        f.arena.deinit();
    }

    fn a(f: *Fixture) std.mem.Allocator {
        return f.arena.allocator();
    }
};

const hello =
    \\{"type":"hello","protocol":1,"helper":{"name":"MacBook Pro","platform":"darwin","version":"0.1.0"},
    \\ "capabilities":{"controls":["open_app","quit_app","volume","media","dark_mode","lock_screen","rm_rf"],"computer_use":true},
    \\ "access":{"accessibility":true,"screen_recording":false}}
;

const voice: hub_mod.Requester = .{ .source = "voice", .device = "phone", .utterance = "open my notes and search for groceries" };

fn json(a: std.mem.Allocator, text: []const u8) Value {
    return std.json.parseFromSliceLeaky(Value, a, text, .{}) catch unreachable;
}

test "a hello registers the helper with only the controls the box knows; anything else is refused" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const x = f.fake.helper.?;
    try t.expectEqualStrings("MacBook Pro", x.name);
    try t.expectEqual(@as(usize, 6), x.controls.len); // rm_rf dropped
    var why: decide.Why = .{};
    try t.expect(f.hub.register(f.a(), "{\"type\":\"hello\",\"protocol\":2}", "d", f.fake.peer(), &why) == null);
    try t.expect(std.mem.indexOf(u8, why.message, "protocol 2") != null);
    try t.expect(f.hub.register(f.a(), "{\"type\":\"control\",\"protocol\":1}", "d", f.fake.peer(), &why) == null);
}

test "a simple fixed control runs at once and comes back with the helper's sentence" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const r = f.hub.control(f.a(), voice, "", json(f.a(), "{\"kind\":\"open_app\",\"app\":\"Safari\"}"));
    try t.expect(r.ok);
    try t.expectEqualStrings("Opened Safari", r.text);
    try t.expect(f.fake.sawType("control"));
    try t.expect(f.fake.helper.?.task == null); // the helper is free again
}

test "an unknown control is refused before the helper hears of it; a refused one fails with the helper's reason" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const r = f.hub.control(f.a(), voice, "", json(f.a(), "{\"kind\":\"shell\",\"command\":\"ls\"}"));
    try t.expect(!r.ok);
    try t.expect(!f.fake.sawType("control"));
    f.fake.refuse = true;
    const off = f.hub.control(f.a(), voice, "", json(f.a(), "{\"kind\":\"volume\",\"action\":\"up\",\"target\":\"system\"}"));
    try t.expect(!off.ok);
    try t.expect(std.mem.indexOf(u8, off.text, "switched off") != null);
}

test "a voice turn without the user's words cannot start anything" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const silent: hub_mod.Requester = .{ .source = "voice", .utterance = "  " };
    try t.expect(!f.hub.control(f.a(), silent, "", json(f.a(), "{\"kind\":\"open_app\",\"app\":\"Safari\"}")).ok);
    try t.expect(!f.hub.startUse(f.a(), silent, "", "search", null, null).ok);
    try t.expect(!f.hub.control(f.a(), .{ .source = "screen" }, "", json(f.a(), "{\"kind\":\"open_app\",\"app\":\"Safari\"}")).ok);
}

fn approveSoon(hub: *Hub, deny: bool) void {
    // Waits for the approval to appear, then answers it as the web UI would.
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    var tries: u32 = 0;
    while (tries < 2000) : (tries += 1) {
        const views = hub.approvals.list(t.io, arena.allocator(), fakeNow()) catch return;
        for (views) |v| if (v.state == .pending) {
            _ = hub.approvals.decide(t.io, v.id, !deny, "web: test", fakeNow());
            return;
        };
        std.Io.sleep(t.io, .fromMilliseconds(2), .awake) catch {};
    }
}

test "locking the screen waits for a tap: approved it runs, denied it does not" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const th = try std.Thread.spawn(.{}, approveSoon, .{ &f.hub, false });
    const r = f.hub.control(f.a(), voice, "", json(f.a(), "{\"kind\":\"lock_screen\"}"));
    th.join();
    try t.expect(r.ok);
    const th2 = try std.Thread.spawn(.{}, approveSoon, .{ &f.hub, true });
    const denied = f.hub.control(f.a(), voice, "", json(f.a(), "{\"kind\":\"lock_screen\"}"));
    th2.join();
    try t.expect(!denied.ok);
    try t.expectEqualStrings("You said no on the approval screen.", denied.text);
}

fn started(f: *Fixture) ![]const u8 {
    const r = f.hub.startUse(f.a(), voice, "", "open my notes and search for groceries", null, null);
    try t.expect(r.ok);
    try t.expect(f.fake.sawType("use.start"));
    return r.task_id;
}

test "computer use: benign steps are granted, a consequential one waits for the tap and is denied" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const id = try started(&f);
    const x = f.fake.helper.?;
    const obs: Hub.Observed = .{ .app = "Notes", .title = "All notes" };
    try t.expect(f.hub.authorize(x, id, 0, obs, .{ .kind = "surface", .what = "app", .name = "Notes" }).ok);
    try t.expect(f.hub.authorize(x, id, 1, obs, .{ .kind = "type", .ref = "r1", .role = "textbox", .label = "Search", .text = "groceries" }).ok);
    try t.expect(f.hub.authorize(x, id, 2, obs, .{ .kind = "press", .key = "enter", .field_role = "textbox", .field_label = "Search" }).ok);
    // A judged button the provider finds harmless runs; one it finds risky asks.
    try t.expect(f.hub.authorize(x, id, 3, obs, .{ .kind = "click", .ref = "r2", .role = "button", .label = "Show All" }).ok);
    const th = try std.Thread.spawn(.{}, approveSoon, .{ &f.hub, true });
    const g = f.hub.authorize(x, id, 4, obs, .{ .kind = "click", .ref = "r3", .role = "button", .label = "Delete" });
    th.join();
    try t.expect(!g.ok);
    try t.expectEqualStrings("You said no on the approval screen.", g.reason);
}

test "an approved step is granted as approved, and a judgment that says it could send asks" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const id = try started(&f);
    const x = f.fake.helper.?;
    f.provider.noul_p = 0.8;
    const th = try std.Thread.spawn(.{}, approveSoon, .{ &f.hub, false });
    const g = f.hub.authorize(x, id, 1, .{ .app = "Mail" }, .{ .kind = "click", .ref = "r9", .role = "button", .label = "OK" });
    th.join();
    try t.expect(g.ok and g.approved);
}

test "the step budget is the box's: past max_steps every step is refused" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    f.hub.limits.max_steps = 2;
    const id = try started(&f);
    const x = f.fake.helper.?;
    const scroll: @import("policy.zig").Action = .{ .kind = "scroll", .direction = "down" };
    try t.expect(f.hub.authorize(x, id, 0, .{}, scroll).ok);
    try t.expect(f.hub.authorize(x, id, 1, .{}, scroll).ok);
    try t.expect(f.hub.authorize(x, id, 2, .{}, scroll).ok);
    const over = f.hub.authorize(x, id, 3, .{}, scroll);
    try t.expect(!over.ok);
    try t.expectEqualStrings("It did not finish within its steps.", over.reason);
}

test "the working-time budget: past it steps are refused, and time waiting for a tap does not count" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const id = try started(&f);
    const x = f.fake.helper.?;
    const scroll: @import("policy.zig").Action = .{ .kind = "scroll", .direction = "down" };
    clock.store(fakeNow() + 80_000, .release);
    try t.expect(f.hub.authorize(x, id, 0, .{}, scroll).ok);
    // A tap that took 30 s of clock: approved, and not charged to the budget.
    const Advance = struct {
        fn run(h: *Hub) void {
            var arena: std.heap.ArenaAllocator = .init(t.allocator);
            defer arena.deinit();
            var tries: u32 = 0;
            while (tries < 2000) : (tries += 1) {
                const views = h.approvals.list(t.io, arena.allocator(), fakeNow()) catch return;
                for (views) |v| if (v.state == .pending) {
                    clock.store(fakeNow() + 30_000, .release);
                    _ = h.approvals.decide(t.io, v.id, true, "web", fakeNow());
                    return;
                };
                std.Io.sleep(t.io, .fromMilliseconds(2), .awake) catch {};
            }
        }
    };
    const th = try std.Thread.spawn(.{}, Advance.run, .{&f.hub});
    try t.expect(f.hub.authorize(x, id, 1, .{}, .{ .kind = "type", .ref = "r", .role = "textbox", .label = "Message", .text = "hi" }).ok);
    th.join();
    try t.expect(f.hub.authorize(x, id, 2, .{}, scroll).ok); // 80 s worked, 30 s waited
    clock.store(fakeNow() + 11_000, .release);
    const late = f.hub.authorize(x, id, 3, .{}, scroll);
    try t.expect(!late.ok);
    try t.expectEqualStrings("It was taking too long.", late.reason);
}

test "steps for another task, or after a stop, are refused; the stop reaches the helper" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const id = try started(&f);
    const x = f.fake.helper.?;
    const scroll: @import("policy.zig").Action = .{ .kind = "scroll", .direction = "down" };
    try t.expect(!f.hub.authorize(x, "cu_other", 0, .{}, scroll).ok);
    try t.expectEqual(@as(usize, 1), f.hub.stop(f.a(), id, "web"));
    try t.expect(f.fake.sawType("stop"));
    try t.expect(!f.hub.authorize(x, id, 0, .{}, scroll).ok);
}

test "one task at a time: a second start is refused until the first finishes" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const id = try started(&f);
    try t.expect(!f.hub.startUse(f.a(), voice, "", "something else", null, null).ok);
    const x = f.fake.helper.?;
    const fin = try std.fmt.allocPrint(f.a(), "{{\"type\":\"finished\",\"task_id\":\"{s}\",\"outcome\":{{\"status\":\"done\",\"reason\":\"\",\"steps\":[\"typed groceries into Search\"],\"surface\":\"Notes: All notes\"}}}}", .{id});
    f.hub.onFrame(x, fin);
    const v = f.hub.view(f.a(), id).?;
    try t.expectEqualStrings("done", v.status);
    try t.expectEqual(@as(usize, 1), v.steps.len);
    try t.expect(x.task == null);
    try t.expect(f.hub.startUse(f.a(), voice, "", "something else", null, null).ok);
}

test "decisions: questions go to the provider and only offered keys come back" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const id = try started(&f);
    const x = f.fake.helper.?;
    const frame = try std.fmt.allocPrint(f.a(), "{{\"type\":\"decide\",\"id\":\"h1\",\"task_id\":\"{s}\",\"state\":{{\"goal\":\"x\"}},\"questions\":{{\"next\":{{\"type\":\"choice\",\"instructions\":\"What next?\",\"criteria\":{{\"c0\":\"Click Search\",\"stuck\":\"None\"}}}}}}}}", .{id});
    const good = f.hub.decideFrame(f.a(), x, json(f.a(), frame));
    try t.expect(good.object.get("ok").?.bool);
    try t.expectEqualStrings("c0", good.object.get("answers").?.object.get("next").?.object.get("choice").?.string);
    f.provider.choice_override = "c42"; // a made-up control
    const made_up = f.hub.decideFrame(f.a(), x, json(f.a(), frame));
    try t.expectEqual(@as(usize, 0), made_up.object.get("answers").?.object.count());
    const wrong = f.hub.decideFrame(f.a(), x, json(f.a(), "{\"type\":\"decide\",\"id\":\"h2\",\"task_id\":\"cu_nope\",\"questions\":{}}"));
    try t.expect(!wrong.object.get("ok").?.bool);
}

test "a helper that disconnects fails its running task" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const id = try started(&f);
    const x = f.fake.helper.?;
    f.fake.helper = null;
    f.hub.unregister(x);
    const v = f.hub.view(f.a(), id).?;
    try t.expectEqualStrings("failed", v.status);
    try t.expectEqual(@as(usize, 0), f.hub.connected());
    try t.expect(!f.hub.control(f.a(), voice, "", json(f.a(), "{\"kind\":\"open_app\",\"app\":\"Safari\"}")).ok);
}
