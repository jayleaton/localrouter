//! Built-in tool calls of a voice session: the image or video job a call asks for, a waiter thread a call that reports
//! the result as `localrouter.job.done` and asks for a spoken follow-up.

const std = @import("std");
const session = @import("session.zig");
const backend = @import("backend.zig");

const Session = session.Session;
const Value = std.json.Value;

const Call = struct { call_id: []u8, name: []u8, arguments: []u8 };

/// Under the lock: a waiter thread for a built-in tool call.
pub fn start(s: *Session, it: *const session.Item) void {
    const c: Call = .{
        .call_id = s.gpa.dupe(u8, it.call_id) catch return,
        .name = s.gpa.dupe(u8, it.name) catch return,
        .arguments = s.gpa.dupe(u8, it.arguments) catch return,
    };
    const t = std.Thread.spawn(.{}, wait, .{ s, c }) catch return;
    s.waiters.append(s.gpa, t) catch t.detach();
}

fn wait(s: *Session, c: Call) void {
    defer {
        s.gpa.free(c.call_id);
        s.gpa.free(c.name);
        s.gpa.free(c.arguments);
    }
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const a = tmp.allocator();
    var why: backend.Why = .{};
    var job_id: []const u8 = "";
    const kind: backend.JobKind = if (std.mem.eql(u8, c.name, "generate_video")) .video else .image;
    const outcome: backend.Outcome = blk: {
        const spec = jobSpec(a, kind, c.arguments, &why) orelse break :blk .{ .status = "failed", .@"error" = why.message };
        const job = s.backend.startJob(a, spec, &why) catch break :blk .{ .status = "failed", .@"error" = why.message };
        job_id = job.id;
        s.mutex.lockUncancelable(s.io);
        s.emit.send(a, "localrouter.job.created", .{ .call_id = c.call_id, .job_id = job.id, .kind = @tagName(kind) });
        s.mutex.unlock(s.io);
        break :blk s.backend.waitJob(a, job, s.voice.tool_wait_s, &s.close_stop);
    };
    if (s.close_stop.cancelled()) return;
    const output = std.json.Stringify.valueAlloc(a, .{ .status = outcome.status, .job_id = job_id, .urls = outcome.urls, .@"error" = outcome.@"error" }, .{}) catch return;
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    if (s.closing) return;
    const prev = s.lastId();
    const id = s.newId(a, "item") catch return;
    const it = s.addItem(.function_call_output, id) catch return;
    it.call_id = it.arena.allocator().dupe(u8, c.call_id) catch "";
    it.text = it.arena.allocator().dupe(u8, output) catch "";
    s.emit.send(a, "conversation.item.created", .{ .previous_item_id = prev, .item = session.itemValue(a, it, s.settings.audio) catch Value.null });
    s.emit.send(a, "localrouter.job.done", .{ .call_id = c.call_id, .job_id = job_id, .kind = @tagName(kind), .status = outcome.status, .urls = outcome.urls, .@"error" = outcome.@"error" });
    s.fresh = true;
    s.push(.{ .respond = .{ .auto = true } }) catch {};
}

/// The job a tool call's arguments ask for; null with `why` set when they are not usable.
fn jobSpec(a: std.mem.Allocator, kind: backend.JobKind, arguments: []const u8, why: *backend.Why) ?backend.JobSpec {
    const Args = struct { prompt: []const u8 = "", size: []const u8 = "", seconds: u32 = 0 };
    const args = std.json.parseFromSliceLeaky(Args, a, arguments, .{ .ignore_unknown_fields = true }) catch {
        why.set("the tool call's arguments are not valid JSON: {s}", .{arguments[0..@min(arguments.len, 200)]});
        return null;
    };
    if (std.mem.trim(u8, args.prompt, " ").len == 0) {
        why.set("the tool call has no prompt", .{});
        return null;
    }
    return .{ .kind = kind, .prompt = args.prompt, .size = args.size, .seconds = args.seconds };
}

test "a tool call's arguments: a prompt is needed, JSON only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: backend.Why = .{};
    try std.testing.expect(jobSpec(a, .image, "{\"prompt\": \"a fox\", \"size\": \"512x512\"}", &why) != null);
    try std.testing.expect(jobSpec(a, .image, "{\"size\": \"512x512\"}", &why) == null);
    try std.testing.expectEqualStrings("the tool call has no prompt", why.message);
    try std.testing.expect(jobSpec(a, .video, "not json", &why) == null);
}
