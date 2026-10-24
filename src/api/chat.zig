//! The chat route, OpenAI's Chat Completions API: POST /v1/chat/completions, whole or streamed (server-sent events).
//! The body goes to the model's own server as sent, but for `model` (set to the tool's id), so messages, tools and
//! tool_choice, sampling, stop, response_format and the reasoning switches pass through, and the model refuses what it
//! does not take. The reply is the server's own bytes: `completion.json`, or `completion.sse` sent on as the scheduler
//! writes it. A client that leaves a stream cancels the request; the model stays loaded. A small limit switches thinking
//! off, and a reply whose reasoning used its whole limit says so (`budget.zig`: `warnings` on the completion, or a last
//! chunk before `[DONE]`). The MCP `chat` tool and the Responses route share `startChat` and `readCompletion`.

const std = @import("std");
const Io = std.Io;
const posix = std.posix;
const h = @import("http.zig");
const routes = @import("routes.zig");
const jobs = @import("../sched/jobs.zig");
const budget = @import("budget.zig");

const App = routes.App;
const Value = std.json.Value;

/// Why a chat job was not started: the HTTP status, OpenAI's error type and the message.
pub const Refusal = routes.Refusal;

pub const not_a_request = "the body is not a valid chat request (a JSON object with messages)";

fn refuse(why: *Refusal, msg: []const u8) ?*jobs.Job {
    why.status = 400;
    why.typ = "invalid_request_error";
    why.message = msg;
    return null;
}

/// Checks a chat request (a parsed JSON object), points it at the model that serves it and queues it for `owner`
/// (`routes.enqueue`); null with `why` set on refusal. Only what LocalRouter needs is checked here; the model's server
/// judges the rest. `plan` gets what the thinking budget did (`budget.apply`), for the reply's warnings.
pub fn startChat(app: *App, a: std.mem.Allocator, body: Value, owner: []const u8, why: *Refusal, plan: *budget.Plan) ?*jobs.Job {
    if (body != .object) return refuse(why, not_a_request);
    var obj = body.object;
    const messages = obj.get("messages") orelse return refuse(why, "messages is required: a non-empty array of messages");
    if (messages != .array or messages.array.items.len == 0) return refuse(why, "messages is required: a non-empty array of messages");
    const stream = if (obj.get("stream")) |s| switch (s) {
        .bool => |b| b,
        .null => false,
        else => return refuse(why, "stream must be true or false"),
    } else false;
    const model: ?[]const u8 = if (obj.get("model")) |m| switch (m) {
        .string => |s| s,
        .null => null,
        else => return refuse(why, "model must be a string"),
    } else null;
    const tool = routes.resolve(app, model, .chat_completion, why) orelse return null;
    const cfg = app.scheduler.slots[tool].tool.config();
    obj.put(a, "model", .{ .string = cfg.id }) catch return refuse(why, "out of memory");
    plan.* = budget.apply(a, &obj, cfg, plan.api) catch return refuse(why, "out of memory");
    const text = std.json.Stringify.valueAlloc(a, Value{ .object = obj }, .{}) catch return refuse(why, "out of memory");
    const seed: u64 = if (obj.get("seed")) |s| (if (s == .integer and s.integer >= 0) @intCast(s.integer) else 0) else 0;
    return routes.enqueue(app, tool, .{ .chat = .{ .body = text, .stream = stream, .seed = seed } }, &.{}, owner, why);
}

/// The finished job's whole reply (`completion.json`), as the model's server sent it.
pub fn readCompletion(app: *App, a: std.mem.Allocator, j: *jobs.Job, why: *Refusal) ?[]const u8 {
    const path = std.fs.path.join(a, &.{ j.dir, "completion.json" }) catch return null;
    return Io.Dir.cwd().readFileAlloc(app.io, path, a, .limited(64 << 20)) catch {
        why.status = 500;
        why.typ = "server_error";
        why.message = "the model's server wrote no reply";
        return null;
    };
}

/// POST /v1/chat/completions.
pub fn completions(app: *App, req: *h.Request, a: std.mem.Allocator, owner: []const u8) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const body = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return h.fail(req, a, 400, "invalid_request_error", not_a_request);
    var why: Refusal = .{};
    var plan: budget.Plan = .{};
    const j = startChat(app, a, body, owner, &why, &plan) orelse return h.fail(req, a, why.status, why.typ, why.message);
    if (j.request.chat.stream) return streamChat(app, req, a, j, plan);
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const out = readCompletion(app, a, j, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    return h.json(req, .ok, try budget.annotate(a, out, plan));
}

const poll_ms = 20;

/// Sends `completion.sse` as the scheduler writes it, line by line. A request refused before the server streamed
/// anything gets the ordinary JSON error; one that breaks off mid-stream ends with an error event and `[DONE]`; one
/// with warnings sends them in a last chunk before `[DONE]`.
fn streamChat(app: *App, req: *h.Request, a: std.mem.Allocator, j: *jobs.Job, plan: budget.Plan) !void {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/completion.sse", .{j.dir}, 0);
    var fd: posix.fd_t = -1;
    defer closeFd(fd);
    while (true) { // the server accepted the request (the file exists), or the job ended first
        const done = j.done.isSet();
        fd = posix.openatZ(posix.AT.FDCWD, path, .{}, 0) catch -1;
        if (fd >= 0 or done) break;
        Io.sleep(app.io, .fromMilliseconds(poll_ms), .awake) catch {};
    }
    if (fd < 0) {
        if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
        return h.fail(req, a, 500, "server_error", "the model's server sent no event stream");
    }
    var send: [16 * 1024]u8 = undefined;
    var body = req.respondStreaming(&send, .{ .respond_options = .{ .extra_headers = &.{
        .{ .name = "content-type", .value = "text/event-stream" },
        .{ .name = "cache-control", .value = "no-cache" },
    } } }) catch return gone(app, j);
    var seen: budget.Seen = .{};
    var ended = false; // the server's `data: [DONE]` came: held back until the warnings are sent
    var pending: std.ArrayList(u8) = .empty; // the stream's last line, until its newline arrives
    var chunk: [16 * 1024]u8 = undefined;
    var at: u64 = 0;
    while (true) {
        const done = j.done.isSet();
        var sent = false;
        while (true) {
            const got = readAt(fd, &chunk, at);
            if (got == 0) break;
            at += got;
            try pending.appendSlice(a, chunk[0..got]);
            var start: usize = 0;
            while (std.mem.indexOfScalarPos(u8, pending.items, start, '\n')) |nl| {
                const l = pending.items[start..nl];
                if (!ended) if (isDone(l)) {
                    ended = true;
                } else {
                    try seen.line(a, l);
                    body.writer.writeAll(pending.items[start .. nl + 1]) catch return gone(app, j);
                };
                start = nl + 1;
            }
            const rest = pending.items.len - start;
            std.mem.copyForwards(u8, pending.items[0..rest], pending.items[start..]);
            pending.shrinkRetainingCapacity(rest);
            sent = true;
        }
        if (sent) {
            body.writer.flush() catch return gone(app, j);
            body.flush() catch return gone(app, j);
        }
        if (done) break;
        Io.sleep(app.io, .fromMilliseconds(poll_ms), .awake) catch {};
    }
    if (!ended and pending.items.len > 0) body.writer.writeAll(pending.items) catch return gone(app, j);
    if (j.status != .completed) {
        const event = try std.json.Stringify.valueAlloc(a, .{ .@"error" = .{ .message = j.error_message, .type = j.error_type } }, .{});
        body.writer.print("data: {s}\n\ndata: [DONE]\n\n", .{event}) catch return gone(app, j);
    } else if (ended) {
        const w = try budget.warnings(a, plan, seen);
        if (w.len > 0) body.writer.writeAll(try budget.chunk(a, seen, w)) catch return gone(app, j);
        body.writer.writeAll("data: [DONE]\n\n") catch return gone(app, j);
    }
    body.end() catch return gone(app, j);
}

/// The server's end-of-stream line.
fn isDone(l: []const u8) bool {
    const t = std.mem.trimEnd(u8, l, "\r");
    if (!std.mem.startsWith(u8, t, "data:")) return false;
    return std.mem.eql(u8, std.mem.trim(u8, t["data:".len..], " "), "[DONE]");
}

fn closeFd(fd: posix.fd_t) void {
    if (fd >= 0) _ = posix.system.close(fd);
}

fn readAt(fd: posix.fd_t, buf: []u8, offset: u64) usize {
    const rc = posix.system.pread(fd, buf.ptr, buf.len, @intCast(offset));
    if (posix.errno(rc) != .SUCCESS) return 0;
    return @intCast(rc);
}

/// The client left: cancel its request (the proxied call ends; the model stays loaded). Not an error of this
/// daemon's, so nothing is returned for the connection loop to log.
fn gone(app: *App, j: *jobs.Job) void {
    if (!j.done.isSet()) app.scheduler.cancel(j);
}
