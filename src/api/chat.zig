//! The chat route, OpenAI's Chat Completions API: POST /v1/chat/completions, whole or streamed (server-sent events).
//! The body goes to the model's own server as sent, but for `model` (set to the tool's id), so messages, tools and
//! tool_choice, sampling, stop, response_format and the reasoning switches pass through, and the model refuses what it
//! does not take. The reply is the server's own bytes: `completion.json`, or `completion.sse` sent on as the scheduler
//! writes it. A client that leaves a stream cancels the request; the model stays loaded. The MCP `chat` tool shares
//! `startChat` and `readCompletion`.

const std = @import("std");
const Io = std.Io;
const posix = std.posix;
const h = @import("http.zig");
const routes = @import("routes.zig");
const jobs = @import("../sched/jobs.zig");

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

/// Checks a chat request (a parsed JSON object), points it at the model that serves it and queues it; null with `why`
/// set on refusal. Only what LocalRouter needs is checked here; the model's server judges the rest.
pub fn startChat(app: *App, a: std.mem.Allocator, body: Value, why: *Refusal) ?*jobs.Job {
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
    obj.put(a, "model", .{ .string = app.scheduler.slots[tool].tool.config().id }) catch return refuse(why, "out of memory");
    const text = std.json.Stringify.valueAlloc(a, Value{ .object = obj }, .{}) catch return refuse(why, "out of memory");
    const seed: u64 = if (obj.get("seed")) |s| (if (s == .integer and s.integer >= 0) @intCast(s.integer) else 0) else 0;
    return routes.enqueue(app, tool, .{ .chat = .{ .body = text, .stream = stream, .seed = seed } }, &.{}, why);
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
pub fn completions(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const body = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return h.fail(req, a, 400, "invalid_request_error", not_a_request);
    var why: Refusal = .{};
    const j = startChat(app, a, body, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    if (j.request.chat.stream) return streamChat(app, req, a, j);
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const out = readCompletion(app, a, j, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    return h.json(req, .ok, out);
}

const poll_ms = 20;

/// Sends `completion.sse` as the scheduler writes it. A request refused before the server streamed anything gets the
/// ordinary JSON error; one that breaks off mid-stream ends with an error event and `[DONE]`.
fn streamChat(app: *App, req: *h.Request, a: std.mem.Allocator, j: *jobs.Job) !void {
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
    var chunk: [16 * 1024]u8 = undefined;
    var at: u64 = 0;
    while (true) {
        const done = j.done.isSet();
        var sent = false;
        while (true) {
            const got = readAt(fd, &chunk, at);
            if (got == 0) break;
            body.writer.writeAll(chunk[0..got]) catch return gone(app, j);
            at += got;
            sent = true;
        }
        if (sent) {
            body.writer.flush() catch return gone(app, j);
            body.flush() catch return gone(app, j);
        }
        if (done) break;
        Io.sleep(app.io, .fromMilliseconds(poll_ms), .awake) catch {};
    }
    if (j.status != .completed) {
        const event = try std.json.Stringify.valueAlloc(a, .{ .@"error" = .{ .message = j.error_message, .type = j.error_type } }, .{});
        body.writer.print("data: {s}\n\ndata: [DONE]\n\n", .{event}) catch return gone(app, j);
    }
    body.end() catch return gone(app, j);
}

fn closeFd(fd: posix.fd_t) void {
    if (fd >= 0) _ = posix.system.close(fd);
}

fn readAt(fd: posix.fd_t, buf: []u8, offset: u64) usize {
    const rc = posix.system.pread(fd, buf.ptr, buf.len, @intCast(offset));
    if (posix.errno(rc) != .SUCCESS) return 0;
    return @intCast(rc);
}

/// The client left: cancel its request (the proxied call ends; the model stays loaded).
fn gone(app: *App, j: *jobs.Job) error{ClientGone} {
    if (!j.done.isSet()) app.scheduler.cancel(j);
    return error.ClientGone;
}
