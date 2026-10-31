//! Built-in tool calls of a voice session: the image or video job a call asks for (a waiter thread reports the result
//! as `localrouter.job.done` and asks for a spoken follow-up once voice is back), and the model management tools
//! (list_models, reserve_for, unload_model, load_model), whose JSON answers the call. The session keeps the file of
//! its last picture: edit_image edits it unless the call names another, generate_video's from_last_image animates it,
//! and generate_3d makes a 3D asset of it.

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
    if (std.mem.eql(u8, c.name, "computer_control") or std.mem.eql(u8, c.name, "computer_use")) {
        // A computer task (docs/dev/COMPUTER.md): the user's own words in this turn go with it, so the model alone
        // cannot start one; lines worth hearing while it runs are spoken (an approval waiting for a tap).
        var sp: SessionSpeaker = .{ .s = s };
        const m = s.backend.computer(a, c.name, c.arguments, lastUserWords(s, a), .{ .ctx = &sp, .say = SessionSpeaker.say }, &s.close_stop);
        if (s.close_stop.cancelled()) return;
        const text = if (m.ok) m.text else std.fmt.allocPrint(a, "{{\"error\": {s}}}", .{std.json.Stringify.valueAlloc(a, m.text, .{}) catch return}) catch return;
        return answer(s, a, c, text);
    }
    if (!isJob(c.name)) {
        // A model management tool (list_models, reserve_for, unload_model, load_model): its JSON is the output.
        const m = s.backend.manage(a, c.name, c.arguments);
        const text = if (m.ok) m.text else std.fmt.allocPrint(a, "{{\"error\": {s}}}", .{std.json.Stringify.valueAlloc(a, m.text, .{}) catch return}) catch return;
        return answer(s, a, c, text);
    }
    var why: backend.Why = .{};
    var job_id: []const u8 = "";
    const kind = jobKind(c.name);
    const outcome: backend.Outcome = blk: {
        const spec = jobSpec(a, c.name, c.arguments, lastImage(s, a), &why) orelse break :blk .{ .status = "failed", .@"error" = why.message };
        const job = s.backend.startJob(a, spec, &why) catch break :blk .{ .status = "failed", .@"error" = why.message };
        job_id = job.id;
        s.mutex.lockUncancelable(s.io);
        s.emit.send(a, "localrouter.job.created", .{ .call_id = c.call_id, .job_id = job.id, .kind = @tagName(kind) });
        s.mutex.unlock(s.io);
        break :blk s.backend.waitJob(a, job, s.voice.tool_wait_s, &s.close_stop);
    };
    if (s.close_stop.cancelled()) return;
    s.mutex.lockUncancelable(s.io);
    if (kind == .image and std.mem.eql(u8, outcome.status, "completed") and outcome.files.len > 0) {
        if (s.gpa.dupe(u8, outcome.files[0])) |p| {
            if (s.last_image) |old| s.gpa.free(old);
            s.last_image = p;
        } else |_| {}
    }
    if (!s.closing) s.emit.send(a, "localrouter.job.done", .{ .call_id = c.call_id, .job_id = job_id, .kind = @tagName(kind), .status = outcome.status, .urls = outcome.urls, .@"error" = outcome.@"error" });
    s.mutex.unlock(s.io);
    waitForVoice(s); // the models a video unloaded come back right after it: then say it is ready
    const back = comingBack(s, a);
    const output = std.json.Stringify.valueAlloc(a, .{ .status = outcome.status, .job_id = job_id, .urls = outcome.urls, .@"error" = outcome.@"error", .coming_back = back, .note = if (back.len > 0) "these models were unloaded to make room and come back by themselves" else null }, .{ .emit_null_optional_fields = false }) catch return;
    answer(s, a, c, output);
}

/// The built-in tools that run a job (the rest answer at once).
fn isJob(name: []const u8) bool {
    for ([_][]const u8{ "generate_image", "edit_image", "generate_video", "generate_3d" }) |n| if (std.mem.eql(u8, name, n)) return true;
    return false;
}

fn jobKind(name: []const u8) backend.JobKind {
    if (std.mem.eql(u8, name, "generate_video")) return .video;
    if (std.mem.eql(u8, name, "generate_3d")) return .asset3d;
    return .image;
}

/// The session's last picture's file, copied into `a` ("" when it has made none).
fn lastImage(s: *Session, a: std.mem.Allocator) []const u8 {
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    return a.dupe(u8, s.last_image orelse return "") catch "";
}

/// The server's own memory check after a job, so the chat model need not call list_models around a picture or a video:
/// the kept-loaded models not running now (unloaded to make room). The scheduler brings them back by itself.
fn comingBack(s: *Session, a: std.mem.Allocator) []const []const u8 {
    const m = s.backend.manage(a, "list_models", "{}");
    if (!m.ok) return &.{};
    const Model = struct { id: []const u8, keep_loaded: bool = false, running: bool = true };
    const list = std.json.parseFromSliceLeaky(struct { models: []const Model = &.{} }, a, m.text, .{ .ignore_unknown_fields = true }) catch return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    for (list.models) |x| if (x.keep_loaded and !x.running) out.append(a, x.id) catch {};
    return out.items;
}

/// Waits (without the session's lock) while a voice model is away making room, at most `tool_wait_s`, so the reply
/// about a finished job is spoken rather than text only.
fn waitForVoice(s: *Session) void {
    var waited: u64 = 0;
    while (s.backend.paused() and waited < @as(u64, s.voice.tool_wait_s) * 1000) : (waited += 250) {
        if (s.close_stop.cancelled()) return;
        std.Io.sleep(s.io, .fromMilliseconds(250), .awake) catch return;
    }
}

/// The user's words in the latest user turn that has a transcript, copied into `a` ("" when there is none).
fn lastUserWords(s: *Session, a: std.mem.Allocator) []const u8 {
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    var i = s.items.items.len;
    while (i > 0) {
        i -= 1;
        const it = s.items.items[i];
        if (it.kind == .user and !it.pending and it.text.len > 0) return a.dupe(u8, it.text) catch "";
    }
    return "";
}

/// Speaks a fixed line as the assistant (queued after whatever the session is saying).
const SessionSpeaker = struct {
    s: *Session,

    fn say(ctx: *anyopaque, text: []const u8) void {
        const sp: *SessionSpeaker = @ptrCast(@alignCast(ctx));
        const s = sp.s;
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (s.closing) return;
        const line = s.gpa.dupe(u8, text) catch return;
        s.push(.{ .respond = .{ .auto = false, .line = line } }) catch s.gpa.free(line);
    }
};

/// Adds the call's output to the conversation and asks for the reply that reads it.
fn answer(s: *Session, a: std.mem.Allocator, c: Call, output: []const u8) void {
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    if (s.closing) return;
    const prev = s.lastId();
    const id = s.newId(a, "item") catch return;
    const it = s.addItem(.function_call_output, id) catch return;
    it.call_id = it.arena.allocator().dupe(u8, c.call_id) catch "";
    it.text = it.arena.allocator().dupe(u8, output) catch "";
    s.emit.send(a, "conversation.item.created", .{ .previous_item_id = prev, .item = session.itemValue(a, it, s.settings.audio) catch Value.null });
    s.fresh = true;
    s.push(.{ .respond = .{ .auto = true } }) catch {};
}

/// The job tool call `name`'s arguments ask for, with `last` the file of the session's last picture ("": none); null
/// with `why` set when they are not usable. edit_image edits `image` (a URL) or else the last picture; generate_video
/// with from_last_image starts from the last picture.
fn jobSpec(a: std.mem.Allocator, name: []const u8, arguments: []const u8, last: []const u8, why: *backend.Why) ?backend.JobSpec {
    const Args = struct { prompt: []const u8 = "", size: []const u8 = "", seconds: u32 = 0, image: []const u8 = "", from_last_image: bool = false, input_mode: []const u8 = "" };
    const args = std.json.parseFromSliceLeaky(Args, a, arguments, .{ .ignore_unknown_fields = true }) catch {
        why.set("the tool call's arguments are not valid JSON: {s}", .{arguments[0..@min(arguments.len, 200)]});
        return null;
    };
    const kind = jobKind(name);
    if (kind == .asset3d) { // a picture, not a prompt: the one named, else the last
        if (args.image.len == 0 and last.len == 0) {
            why.set("there is no picture in this conversation yet to make a 3D model of: make one with generate_image first, or send its URL as image", .{});
            return null;
        }
        return .{ .kind = .asset3d, .prompt = "", .input_url = args.image, .input_file = if (args.image.len > 0) "" else last, .input_mode = args.input_mode };
    }
    if (std.mem.trim(u8, args.prompt, " ").len == 0) {
        why.set("the tool call has no prompt", .{});
        return null;
    }
    const edit = std.mem.eql(u8, name, "edit_image");
    var spec: backend.JobSpec = .{ .kind = kind, .prompt = args.prompt, .size = args.size, .seconds = args.seconds };
    if (edit and args.image.len > 0) {
        spec.input_url = args.image;
    } else if (edit or (spec.kind == .video and args.from_last_image)) {
        if (last.len == 0) {
            why.set("there is no picture in this conversation yet to {s}: make one with generate_image first", .{if (edit) "edit" else "animate"});
            return null;
        }
        spec.input_file = last;
    }
    return spec;
}

test "a tool call's arguments: a prompt is needed, JSON only; an edit takes the last picture, or says there is none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: backend.Why = .{};
    try std.testing.expect(jobSpec(a, "generate_image", "{\"prompt\": \"a fox\", \"size\": \"512x512\"}", "", &why) != null);
    try std.testing.expect(jobSpec(a, "generate_image", "{\"size\": \"512x512\"}", "", &why) == null);
    try std.testing.expectEqualStrings("the tool call has no prompt", why.message);
    try std.testing.expect(jobSpec(a, "generate_video", "not json", "", &why) == null);
    try std.testing.expect(jobSpec(a, "edit_image", "{\"prompt\": \"make it night\"}", "", &why) == null);
    try std.testing.expectEqualStrings("there is no picture in this conversation yet to edit: make one with generate_image first", why.message);
    const e = jobSpec(a, "edit_image", "{\"prompt\": \"make it night\"}", "/jobs/img_1/0.png", &why).?;
    try std.testing.expectEqual(backend.JobKind.image, e.kind);
    try std.testing.expectEqualStrings("/jobs/img_1/0.png", e.input_file);
    const u = jobSpec(a, "edit_image", "{\"prompt\": \"make it night\", \"image\": \"http://x/v1/files/img_2/0.png\"}", "/jobs/img_1/0.png", &why).?;
    try std.testing.expectEqualStrings("http://x/v1/files/img_2/0.png", u.input_url);
    try std.testing.expectEqualStrings("", u.input_file);
    const v = jobSpec(a, "generate_video", "{\"prompt\": \"waves\", \"from_last_image\": true}", "/jobs/img_1/0.png", &why).?;
    try std.testing.expectEqual(backend.JobKind.video, v.kind);
    try std.testing.expectEqualStrings("/jobs/img_1/0.png", v.input_file);
    try std.testing.expectEqualStrings("", jobSpec(a, "generate_video", "{\"prompt\": \"waves\"}", "/jobs/img_1/0.png", &why).?.input_file);
    // generate_3d: no prompt; the last picture, or the URL named, or a refusal saying there is none
    const m = jobSpec(a, "generate_3d", "{\"input_mode\": \"real\"}", "/jobs/img_1/0.png", &why).?;
    try std.testing.expectEqual(backend.JobKind.asset3d, m.kind);
    try std.testing.expectEqualStrings("/jobs/img_1/0.png", m.input_file);
    try std.testing.expectEqualStrings("real", m.input_mode);
    try std.testing.expectEqualStrings("http://x/v1/files/img_2/0.png", jobSpec(a, "generate_3d", "{\"image\": \"http://x/v1/files/img_2/0.png\"}", "", &why).?.input_url);
    try std.testing.expect(jobSpec(a, "generate_3d", "{}", "", &why) == null);
    try std.testing.expect(std.mem.startsWith(u8, why.message, "there is no picture in this conversation yet to make a 3D model of"));
}
