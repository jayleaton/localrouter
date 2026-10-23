//! Approvals: the user's tap in the web UI for one consequential computer action (docs/dev/COMPUTER.md). An approval is
//! bound to the helper, the task, the step, what was on screen and the exact action, by a SHA-256 digest of all of it;
//! it expires (60 s by default), is used once, and only the step it was made for can take it. Kept in memory: a restart
//! forgets every pending approval, which then never runs. Time is passed in, so the rules are tested without a clock.

const std = @import("std");
const Io = std.Io;
const policy = @import("policy.zig");

/// What the tap approves: shown on the card in full, and hashed into the approval.
pub const Binding = struct {
    helper_id: []const u8,
    task_id: []const u8,
    step: u32,
    app: []const u8 = "",
    title: []const u8 = "",
    url: []const u8 = "",
    action: policy.Action,
    why: []const u8 = "", // the policy's reason, for the card
    goal: []const u8 = "", // what the user asked for
    requester: []const u8 = "", // who started the task ("voice", "mcp: <device>")

    /// SHA-256 over every field, each length-prefixed, so no two different bindings share a digest by concatenation.
    pub fn digest(b: *const Binding) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        const fields = [_][]const u8{ b.helper_id, b.task_id, b.app, b.title, b.url, b.action.kind, b.action.ref, b.action.role, b.action.label, b.action.text, b.action.key, b.action.field_role, b.action.field_label, b.action.direction, b.action.what, b.action.name };
        var step: [4]u8 = undefined;
        std.mem.writeInt(u32, &step, b.step, .little);
        h.update(&step);
        for (fields) |f| {
            var n: [8]u8 = undefined;
            std.mem.writeInt(u64, &n, f.len, .little);
            h.update(&n);
            h.update(f);
        }
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
};

pub const State = enum { pending, approved, denied, expired, used, cancelled };

pub const Approval = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    binding: Binding, // copied into `arena`
    digest: [32]u8,
    created_ms: i64,
    expires_ms: i64,
    state: State = .pending,
    decided_by: []const u8 = "",
};

pub const default_ttl_ms: i64 = 60_000;
/// Finished approvals are kept this long for the card's last state, then dropped.
const keep_ms: i64 = 10 * 60_000;
const max_items = 256;

pub const Store = struct {
    gpa: std.mem.Allocator,
    mutex: Io.Mutex = .init,
    items: std.ArrayList(*Approval) = .empty,
    ttl_ms: i64 = default_ttl_ms,

    pub fn deinit(s: *Store) void {
        for (s.items.items) |x| destroy(s.gpa, x);
        s.items.deinit(s.gpa);
    }

    /// A pending approval for `b`, its id in the store (valid while the store holds it); `id_bytes` is its random id.
    pub fn create(s: *Store, io: Io, b: Binding, now_ms: i64, id_bytes: [12]u8) ![]const u8 {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.prune(now_ms);
        if (s.items.items.len >= max_items) return error.TooManyApprovals;
        const x = try s.gpa.create(Approval);
        x.* = .{ .arena = .init(s.gpa), .id = "", .binding = undefined, .digest = b.digest(), .created_ms = now_ms, .expires_ms = now_ms + s.ttl_ms };
        errdefer destroy(s.gpa, x);
        const a = x.arena.allocator();
        const hex = std.fmt.bytesToHex(id_bytes, .lower);
        x.id = try std.fmt.allocPrint(a, "appr_{s}", .{&hex});
        x.binding = try copy(a, b);
        try s.items.append(s.gpa, x);
        return x.id;
    }

    pub const Decided = enum { ok, unknown, not_pending, expired };

    /// The user's answer from the web UI. Only a pending, unexpired approval can be answered, once.
    pub fn decide(s: *Store, io: Io, id: []const u8, approve: bool, by: []const u8, now_ms: i64) Decided {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const x = s.find(id) orelse return .unknown;
        expire(x, now_ms);
        if (x.state == .expired) return .expired;
        if (x.state != .pending) return .not_pending;
        x.state = if (approve) .approved else .denied;
        x.decided_by = x.arena.allocator().dupe(u8, by) catch "";
        return .ok;
    }

    pub const Taken = enum { approved, denied, expired, pending, unknown, mismatch, cancelled };

    /// The waiting step's check: `approved` once, for the same binding only (then the approval is used up).
    pub fn take(s: *Store, io: Io, id: []const u8, digest: [32]u8, now_ms: i64) Taken {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const x = s.find(id) orelse return .unknown;
        if (!std.mem.eql(u8, &x.digest, &digest)) return .mismatch;
        expire(x, now_ms);
        return switch (x.state) {
            .pending => .pending,
            .approved => blk: {
                x.state = .used;
                break :blk .approved;
            },
            .denied => .denied,
            .expired => .expired,
            .used => .unknown, // used up: a second take is not a second approval
            .cancelled => .cancelled,
        };
    }

    /// The task ended or was stopped: its pending approvals can no longer be granted.
    pub fn cancelTask(s: *Store, io: Io, task_id: []const u8) void {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        for (s.items.items) |x| if (x.state == .pending and std.mem.eql(u8, x.binding.task_id, task_id)) {
            x.state = .cancelled;
        };
    }

    /// A copy of one approval's card for the UI, in `a`.
    pub const View = struct { id: []const u8, state: State, created_ms: i64, expires_ms: i64, decided_by: []const u8, binding: Binding };

    /// Every approval still pending, then the ones that ended in the last ten minutes, newest first, copied into `a`.
    pub fn list(s: *Store, io: Io, a: std.mem.Allocator, now_ms: i64) ![]View {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.prune(now_ms);
        var out: std.ArrayList(View) = .empty;
        var i = s.items.items.len;
        while (i > 0) {
            i -= 1;
            const x = s.items.items[i];
            expire(x, now_ms);
            try out.append(a, .{ .id = try a.dupe(u8, x.id), .state = x.state, .created_ms = x.created_ms, .expires_ms = x.expires_ms, .decided_by = try a.dupe(u8, x.decided_by), .binding = try copy(a, x.binding) });
        }
        std.mem.sort(View, out.items, {}, pendingFirst);
        return out.items;
    }

    fn pendingFirst(_: void, l: View, r: View) bool {
        return l.state == .pending and r.state != .pending;
    }

    fn find(s: *Store, id: []const u8) ?*Approval {
        for (s.items.items) |x| if (std.mem.eql(u8, x.id, id)) return x;
        return null;
    }

    fn prune(s: *Store, now_ms: i64) void {
        var i: usize = 0;
        while (i < s.items.items.len) {
            const x = s.items.items[i];
            expire(x, now_ms);
            if (x.state != .pending and now_ms - x.created_ms > keep_ms) {
                destroy(s.gpa, x);
                _ = s.items.orderedRemove(i);
            } else i += 1;
        }
    }
};

fn expire(x: *Approval, now_ms: i64) void {
    if ((x.state == .pending or x.state == .approved) and now_ms >= x.expires_ms) x.state = .expired;
}

fn destroy(gpa: std.mem.Allocator, x: *Approval) void {
    x.arena.deinit();
    gpa.destroy(x);
}

fn copy(a: std.mem.Allocator, b: Binding) !Binding {
    var out = b;
    inline for (.{ "helper_id", "task_id", "app", "title", "url", "why", "goal", "requester" }) |f| @field(out, f) = try a.dupe(u8, @field(b, f));
    inline for (@typeInfo(policy.Action).@"struct".field_names) |f| @field(out.action, f) = try a.dupe(u8, @field(b.action, f));
    return out;
}

// ---------------------------------------------------------------- tests

const t = std.testing;

fn binding(text: []const u8) Binding {
    return .{ .helper_id = "mac_1", .task_id = "cu_1", .step = 3, .app = "Messages", .title = "Dana", .action = .{ .kind = "type", .ref = "r7", .role = "textbox", .label = "Message", .text = text } };
}

const id1: [12]u8 = @splat(1);
const id2: [12]u8 = @splat(2);

test "an approval grants its own step once" {
    var s: Store = .{ .gpa = t.allocator };
    defer s.deinit();
    const io = t.io;
    const b = binding("on my way");
    const id = try s.create(io, b, 1000, id1);
    try t.expectEqual(Store.Taken.pending, s.take(io, id, b.digest(), 1100));
    try t.expectEqual(Store.Decided.ok, s.decide(io, id, true, "web: JJ's phone", 1200));
    try t.expectEqual(Store.Taken.approved, s.take(io, id, b.digest(), 1300));
    try t.expectEqual(Store.Taken.unknown, s.take(io, id, b.digest(), 1301)); // used up
    try t.expectEqual(Store.Decided.not_pending, s.decide(io, id, true, "again", 1302));
}

test "an approval binds the exact action: other text, another step or another task cannot take it" {
    var s: Store = .{ .gpa = t.allocator };
    defer s.deinit();
    const io = t.io;
    const b = binding("on my way");
    const id = try s.create(io, b, 0, id1);
    _ = s.decide(io, id, true, "web", 1);
    try t.expectEqual(Store.Taken.mismatch, s.take(io, id, binding("send me your password").digest(), 2));
    var other_step = b;
    other_step.step = 4;
    try t.expectEqual(Store.Taken.mismatch, s.take(io, id, other_step.digest(), 2));
    var other_task = b;
    other_task.task_id = "cu_2";
    try t.expectEqual(Store.Taken.mismatch, s.take(io, id, other_task.digest(), 2));
    var other_window = b;
    other_window.title = "Mum";
    try t.expectEqual(Store.Taken.mismatch, s.take(io, id, other_window.digest(), 2));
    try t.expectEqual(Store.Taken.approved, s.take(io, id, b.digest(), 3)); // the real step still can
}

test "digests do not collide by moving bytes between fields" {
    var x = binding("ab");
    x.action.label = "c";
    var y = binding("a");
    y.action.label = "bc";
    try t.expect(!std.mem.eql(u8, &x.digest(), &y.digest()));
}

test "approvals expire: neither a late tap nor a late take grants anything" {
    var s: Store = .{ .gpa = t.allocator, .ttl_ms = 60_000 };
    defer s.deinit();
    const io = t.io;
    const b = binding("x");
    const id = try s.create(io, b, 0, id1);
    try t.expectEqual(Store.Decided.expired, s.decide(io, id, true, "web", 60_000));
    try t.expectEqual(Store.Taken.expired, s.take(io, id, b.digest(), 60_001));
    // Approved in time but taken too late: still expired.
    const id_b = try s.create(io, b, 100_000, id2);
    _ = s.decide(io, id_b, true, "web", 100_010);
    try t.expectEqual(Store.Taken.expired, s.take(io, id_b, b.digest(), 160_000));
}

test "deny and cancel end an approval" {
    var s: Store = .{ .gpa = t.allocator };
    defer s.deinit();
    const io = t.io;
    const b = binding("x");
    const id = try s.create(io, b, 0, id1);
    try t.expectEqual(Store.Decided.ok, s.decide(io, id, false, "web", 1));
    try t.expectEqual(Store.Taken.denied, s.take(io, id, b.digest(), 2));
    const id_b = try s.create(io, b, 0, id2);
    s.cancelTask(io, "cu_1");
    try t.expectEqual(Store.Taken.cancelled, s.take(io, id_b, b.digest(), 2));
    try t.expectEqual(Store.Decided.not_pending, s.decide(io, id_b, true, "web", 3));
    try t.expectEqual(Store.Decided.unknown, s.decide(io, "appr_nope", true, "web", 3));
}

test "the list shows pending approvals first, with the whole binding" {
    var s: Store = .{ .gpa = t.allocator };
    defer s.deinit();
    const io = t.io;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const done = try s.create(io, binding("first"), 0, id1);
    _ = s.decide(io, done, false, "web", 1);
    _ = try s.create(io, binding("second"), 2, id2);
    const views = try s.list(io, arena.allocator(), 3);
    try t.expectEqual(@as(usize, 2), views.len);
    try t.expectEqual(State.pending, views[0].state);
    try t.expectEqualStrings("second", views[0].binding.action.text);
    try t.expectEqual(State.denied, views[1].state);
}
