//! The chat model, as the voice loop sees it: OpenAI's `/v1/chat/completions` with `stream: true` against any base URL
//! (this daemon's own by default). Text deltas reach a sink as they arrive; tool calls are assembled from their
//! fragments. A `Stop` shuts the socket down, which ends the generation on the server (barge-in).

const std = @import("std");
const Io = std.Io;
const Stop = @import("stop.zig").Stop;

pub const Function = struct { name: []const u8, arguments: []const u8 };
pub const ToolCall = struct { id: []const u8, type: []const u8 = "function", function: Function };

/// One chat message; `content` null with `tool_calls` for an assistant's call, `tool_call_id` for a tool's result.
pub const Message = struct {
    role: []const u8,
    content: ?[]const u8 = null,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
};

pub const Tool = struct {
    type: []const u8 = "function",
    function: struct { name: []const u8, description: []const u8 = "", parameters: std.json.Value },
};

pub const Request = struct {
    model: ?[]const u8 = null,
    messages: []const Message,
    tools: []const Tool = &.{},
    tool_choice: ?std.json.Value = null,
    temperature: ?f32 = null,
    max_tokens: ?u32 = null,
};

/// Receives text as the model writes it (reasoning excluded).
pub const Sink = struct {
    ctx: *anyopaque,
    text: *const fn (ctx: *anyopaque, delta: []const u8) void,
};

pub const Result = struct {
    finish_reason: []const u8 = "",
    text: []const u8 = "",
    tool_calls: []const ToolCall = &.{},
};

pub const Error = error{ Cancelled, Failed, OutOfMemory };

/// A failure's message, in `buf` or static.
pub const Why = struct {
    buf: [512]u8 = undefined,
    message: []const u8 = "",

    pub fn set(w: *Why, comptime fmt: []const u8, args: anytype) void {
        w.message = std.fmt.bufPrint(&w.buf, fmt, args) catch fmt;
    }
};

/// The JSON body OpenAI's API takes for `r`.
pub fn body(a: std.mem.Allocator, r: *const Request) ![]u8 {
    const Body = struct {
        model: ?[]const u8,
        messages: []const Message,
        stream: bool = true,
        stream_options: struct { include_usage: bool = true } = .{},
        tools: ?[]const Tool,
        tool_choice: ?std.json.Value,
        temperature: ?f32,
        max_tokens: ?u32,
    };
    const b: Body = .{ .model = r.model, .messages = r.messages, .tools = if (r.tools.len > 0) r.tools else null, .tool_choice = if (r.tools.len > 0) r.tool_choice else null, .temperature = r.temperature, .max_tokens = r.max_tokens };
    return std.json.Stringify.valueAlloc(a, b, .{ .emit_null_optional_fields = false });
}

/// Folds server-sent event lines into text deltas and tool calls.
pub const Parser = struct {
    a: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Partial) = .empty,
    finish_reason: []const u8 = "",
    done: bool = false,

    const Partial = struct { id: std.ArrayList(u8) = .empty, name: std.ArrayList(u8) = .empty, args: std.ArrayList(u8) = .empty };

    const Chunk = struct {
        choices: []const struct {
            delta: struct {
                content: ?[]const u8 = null,
                tool_calls: ?[]const struct {
                    index: usize = 0,
                    id: ?[]const u8 = null,
                    function: ?struct { name: ?[]const u8 = null, arguments: ?[]const u8 = null } = null,
                } = null,
            } = .{},
            finish_reason: ?[]const u8 = null,
        } = &.{},
    };

    /// One line of the stream; a text delta in it goes to `sink`. `error.Failed` for an error event.
    pub fn line(p: *Parser, raw: []const u8, sink: Sink, why: *Why) Error!void {
        const l = std.mem.trimEnd(u8, raw, "\r");
        if (!std.mem.startsWith(u8, l, "data:")) return; // comments, event names, blank separators
        const data = std.mem.trim(u8, l["data:".len..], " ");
        if (std.mem.eql(u8, data, "[DONE]")) {
            p.done = true;
            return;
        }
        const v = std.json.parseFromSliceLeaky(std.json.Value, p.a, data, .{}) catch return;
        if (v == .object) if (v.object.get("error")) |e| {
            const m = if (e == .object) e.object.get("message") else null;
            why.set("the chat model failed: {s}", .{if (m != null and m.? == .string) m.?.string else "an error event"});
            return error.Failed;
        };
        const c = std.json.parseFromValueLeaky(Chunk, p.a, v, .{ .ignore_unknown_fields = true }) catch return;
        if (c.choices.len == 0) return; // the usage chunk
        const ch = c.choices[0];
        if (ch.finish_reason) |f| p.finish_reason = try p.a.dupe(u8, f);
        if (ch.delta.content) |t| if (t.len > 0) {
            try p.text.appendSlice(p.a, t);
            sink.text(sink.ctx, t);
        };
        for (ch.delta.tool_calls orelse &.{}) |tc| {
            if (tc.index > 64) continue;
            while (p.calls.items.len <= tc.index) try p.calls.append(p.a, .{});
            const call = &p.calls.items[tc.index];
            if (tc.id) |id| try call.id.appendSlice(p.a, id);
            if (tc.function) |f| {
                if (f.name) |n| try call.name.appendSlice(p.a, n);
                if (f.arguments) |x| try call.args.appendSlice(p.a, x);
            }
        }
    }

    pub fn result(p: *Parser) !Result {
        const calls = try p.a.alloc(ToolCall, p.calls.items.len);
        for (calls, p.calls.items, 0..) |*d, s, i| {
            const id = if (s.id.items.len > 0) s.id.items else try std.fmt.allocPrint(p.a, "call_{d}", .{i});
            d.* = .{ .id = id, .function = .{ .name = s.name.items, .arguments = if (s.args.items.len > 0) s.args.items else "{}" } };
        }
        return .{ .finish_reason = p.finish_reason, .text = p.text.items, .tool_calls = calls };
    }
};

/// A chat server: `base_url` ends in /v1 (`{base_url}/chat/completions`).
pub const Client = struct {
    io: Io,
    gpa: std.mem.Allocator,
    base_url: []const u8,
    api_key: []const u8 = "",

    /// Streams `r`; text deltas go to `sink`. `error.Cancelled` once `stop` is set, `error.Failed` with `why`.
    pub fn stream(c: *const Client, a: std.mem.Allocator, r: *const Request, sink: Sink, stop: *Stop, why: *Why) Error!Result {
        const payload = try body(a, r);
        const url = try std.fmt.allocPrint(a, "{s}/chat/completions", .{std.mem.trimEnd(u8, c.base_url, "/")});
        const uri = std.Uri.parse(url) catch {
            why.set("the chat URL {s} is not a URL", .{url});
            return error.Failed;
        };
        var client: std.http.Client = .{ .allocator = c.gpa, .io = c.io };
        defer client.deinit();
        const auth: std.http.Client.Request.Headers.Value = if (c.api_key.len > 0) .{ .override = try std.fmt.allocPrint(a, "Bearer {s}", .{c.api_key}) } else .omit;
        var req = client.request(.POST, uri, .{
            .keep_alive = false,
            .redirect_behavior = .unhandled,
            .headers = .{ .content_type = .{ .override = "application/json" }, .accept_encoding = .omit, .authorization = auth },
            .extra_headers = &.{.{ .name = "accept", .value = "text/event-stream" }},
        }) catch |err| {
            why.set("the chat model at {s} is not reachable: {s}", .{ c.base_url, @errorName(err) });
            return error.Failed;
        };
        defer req.deinit();
        var shut: Shut = .{ .handle = req.connection.?.stream_reader.stream.socket.handle };
        if (!stop.arm(c.io, .{ .ctx = &shut, .call = Shut.call })) return error.Cancelled;
        defer stop.disarm(c.io); // before deinit closes the socket
        return c.exchange(a, &req, payload, sink, stop, why) catch |err| switch (err) {
            error.Failed, error.OutOfMemory => |e| e,
            error.Cancelled => error.Cancelled,
            else => {
                if (stop.cancelled()) return error.Cancelled;
                why.set("the chat stream broke: {s}", .{@errorName(err)});
                return error.Failed;
            },
        };
    }

    fn exchange(c: *const Client, a: std.mem.Allocator, req: *std.http.Client.Request, payload: []u8, sink: Sink, stop: *Stop, why: *Why) !Result {
        _ = c;
        try req.sendBodyComplete(payload);
        var redirect: [2048]u8 = undefined;
        var res = try req.receiveHead(&redirect);
        const transfer = try a.alloc(u8, 64 * 1024);
        const rd = res.reader(transfer);
        if (res.head.status != .ok) {
            const status = @backingInt(res.head.status);
            const text = rd.allocRemaining(a, .limited(64 * 1024)) catch "";
            why.set("the chat model answered {d}: {s}", .{ status, errorMessage(a, text) });
            return error.Failed;
        }
        var p: Parser = .{ .a = a };
        while (!p.done) {
            if (stop.cancelled()) return error.Cancelled;
            const l = (try rd.takeDelimiter('\n')) orelse break;
            try p.line(l, sink, why);
        }
        if (stop.cancelled()) return error.Cancelled;
        return p.result();
    }
};

/// OpenAI's `{"error":{"message":…}}`, else the body itself (cut short).
fn errorMessage(a: std.mem.Allocator, text: []const u8) []const u8 {
    const E = struct { @"error": struct { message: []const u8 = "" } };
    if (std.json.parseFromSliceLeaky(E, a, text, .{ .ignore_unknown_fields = true })) |e| {
        if (e.@"error".message.len > 0) return e.@"error".message;
    } else |_| {}
    return text[0..@min(text.len, 200)];
}

/// Ends a blocking read on the chat socket from another thread.
const Shut = struct {
    handle: std.posix.fd_t,

    fn call(ctx: *anyopaque) void {
        const s: *Shut = @ptrCast(@alignCast(ctx));
        _ = std.posix.system.shutdown(s.handle, std.posix.SHUT.RDWR);
    }
};

const Collect = struct {
    out: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator,

    fn text(ctx: *anyopaque, d: []const u8) void {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.out.appendSlice(c.a, d) catch {};
    }
};

test "the SSE stream: text deltas, tool call fragments, usage and [DONE]" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var col: Collect = .{ .a = a };
    const sink: Sink = .{ .ctx = &col, .text = Collect.text };
    var why: Why = .{};
    var p: Parser = .{ .a = a };
    const lines = [_][]const u8{
        ": keep-alive",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\r",
        "",
        "data: {\"choices\":[{\"delta\":{\"content\":\"Sure, \"}}]}",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"generate_image\",\"arguments\":\"\"}}]}}]}",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"prompt\\\": \"}}]}}]}",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a fox\\\"}\"}}]}}]}",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3}}",
        "data: [DONE]",
    };
    for (lines) |l| try p.line(l, sink, &why);
    try std.testing.expect(p.done);
    const r = try p.result();
    try std.testing.expectEqualStrings("Sure, ", col.out.items);
    try std.testing.expectEqualStrings("Sure, ", r.text);
    try std.testing.expectEqualStrings("tool_calls", r.finish_reason);
    try std.testing.expectEqual(@as(usize, 1), r.tool_calls.len);
    try std.testing.expectEqualStrings("call_1", r.tool_calls[0].id);
    try std.testing.expectEqualStrings("generate_image", r.tool_calls[0].function.name);
    try std.testing.expectEqualStrings("{\"prompt\": \"a fox\"}", r.tool_calls[0].function.arguments);
    try std.testing.expectError(error.Failed, p.line("data: {\"error\":{\"message\":\"overloaded\"}}", sink, &why));
    try std.testing.expectEqualStrings("the chat model failed: overloaded", why.message);
}

test "the request body is OpenAI's, with tools only when there are some" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const calls = [_]ToolCall{.{ .id = "c1", .function = .{ .name = "generate_image", .arguments = "{}" } }};
    const msgs = [_]Message{ .{ .role = "system", .content = "be brief" }, .{ .role = "assistant", .tool_calls = &calls }, .{ .role = "tool", .tool_call_id = "c1", .content = "{\"ok\":true}" } };
    const plain = try body(a, &.{ .messages = &msgs });
    try std.testing.expect(std.mem.indexOf(u8, plain, "\"stream\":true,\"stream_options\":{\"include_usage\":true}") != null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "\"tools\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "\"model\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"generate_image\",\"arguments\":\"{}\"}}]}") != null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "{\"role\":\"tool\",\"content\":\"{\\\"ok\\\":true}\",\"tool_call_id\":\"c1\"}") != null);
    const tools = [_]Tool{.{ .function = .{ .name = "generate_image", .parameters = .{ .object = .empty } } }};
    const with = try body(a, &.{ .model = "qwen", .messages = &msgs, .tools = &tools, .temperature = 0.5 });
    try std.testing.expect(std.mem.indexOf(u8, with, "\"model\":\"qwen\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, with, "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"generate_image\",\"description\":\"\",\"parameters\":{}}}]") != null);
    try std.testing.expectEqualStrings("overloaded", errorMessage(a, "{\"error\":{\"message\":\"overloaded\",\"type\":\"server_error\"}}"));
    try std.testing.expectEqualStrings("plain text", errorMessage(a, "plain text"));
}
