//! The Responses route, OpenAI's Responses API: POST /v1/responses, for harnesses that speak nothing else (Codex). It
//! is a protocol adapter over the chat path, not a second model path: a request's items become one Chat Completions
//! body for the chat model's own server (`chat.startChat`), and its completion (`completion.json`) becomes a response
//! object. Streaming is not offered yet (`stream: true` is refused; coming: H6). Nothing is stored: `store` is false,
//! a conversation is replayed whole each turn, and anything that needs stored state (`previous_response_id`,
//! `conversation`, item references, GET by id) or that the box cannot supply (hosted tools, background mode, images
//! and files, encrypted reasoning, summaries) is a 400 naming it, before any model loads. Function calls come back to
//! the caller, which runs them and sends `function_call_output` items. Reasoning arrives as a `reasoning` item's
//! `reasoning_text`, never as output text, and goes back to the model when the item is replayed.

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const routes = @import("routes.zig");
const chat = @import("chat.zig");

const App = routes.App;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Allocator = std.mem.Allocator;

/// Why a request was refused: OpenAI's error envelope with `param` and `code`.
pub const Bad = struct {
    status: u16 = 400,
    typ: []const u8 = "invalid_request_error",
    code: ?[]const u8 = null,
    param: ?[]const u8 = null,
    message: []const u8 = "",
    buf: [512]u8 = undefined,

    fn set(b: *Bad, code: []const u8, param: []const u8, comptime fmt: []const u8, args: anytype) void {
        b.code = code;
        b.param = param;
        b.message = std.fmt.bufPrint(&b.buf, fmt, args) catch fmt;
    }
};

const Refused = error{Refused};
const Error = error{ Refused, OutOfMemory };

/// The top-level fields this route takes. Anything else is `unknown_parameter`, as OpenAI answers it.
const known = [_][]const u8{
    "model",               "input",      "instructions",      "tools",             "tool_choice",
    "parallel_tool_calls", "reasoning",  "text",              "max_output_tokens", "temperature",
    "top_p",               "stream",     "stream_options",    "store",             "include",
    "metadata",            "user",       "safety_identifier", "prompt_cache_key",  "prompt_cache_retention",
    "client_metadata",     "truncation", "service_tier",      "background",        "previous_response_id",
    "conversation",        "prompt",     "top_logprobs",      "max_tool_calls",
};

/// What a request asked for, echoed in its response object as OpenAI does.
pub const Asked = struct {
    model: []const u8 = "",
    instructions: Value = .null,
    tools: Value = .null,
    tool_choice: Value = .null,
    parallel_tool_calls: bool = true,
    temperature: Value = .null,
    top_p: Value = .null,
    max_output_tokens: Value = .null,
    effort: Value = .null,
    summary: Value = .null,
    format: Value = .null,
    metadata: Value = .null,
    stream: bool = false,
};

/// A Responses request turned into a Chat Completions body (`model` as asked; `startChat` points it at the tool).
pub const Translated = struct { chat: Value, asked: Asked };

fn str(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn isNull(v: ?Value) bool {
    return v == null or v.? == .null;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn object(a: Allocator, fields: anytype) Allocator.Error!Value {
    var m: ObjectMap = .empty;
    inline for (@typeInfo(@TypeOf(fields)).@"struct".field_names) |f| try m.put(a, f, @field(fields, f));
    return .{ .object = m };
}

fn string(s: []const u8) Value {
    return .{ .string = s };
}

/// Checks a Responses request and translates it; `error.Refused` with `bad` set when it asks for something this box
/// cannot do.
pub fn translate(a: Allocator, body: Value, bad: *Bad) Error!Translated {
    if (body != .object) {
        bad.set("invalid_request", "", "the body is not a valid Responses request (a JSON object with input)", .{});
        return error.Refused;
    }
    const o = body.object;
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        for (known) |n| {
            if (eql(n, k)) break;
        } else {
            bad.set("unknown_parameter", k, "Unknown parameter: '{s}'.", .{k});
            return error.Refused;
        }
    }
    try unsupported(o, bad);

    var asked: Asked = .{};
    var out: ObjectMap = .empty;
    if (o.get("model")) |m| switch (m) {
        .string => |s| {
            asked.model = s;
            try out.put(a, "model", m);
        },
        .null => {},
        else => return refuse(bad, "invalid_type", "model", "model must be a string", .{}),
    };
    if (o.get("stream")) |s| switch (s) {
        .bool => |b| {
            if (b) return refuse(bad, "unsupported_value", "stream", "streamed responses are not offered yet (coming: H6): send stream false", .{});
            asked.stream = b;
        },
        .null => {},
        else => return refuse(bad, "invalid_type", "stream", "stream must be true or false", .{}),
    };

    var messages = std.json.Array.init(a);
    if (o.get("instructions")) |ins| switch (ins) {
        .string => |s| {
            asked.instructions = ins;
            if (s.len > 0) try messages.append(try object(a, .{ .role = string("system"), .content = ins }));
        },
        .null => {},
        else => return refuse(bad, "invalid_type", "instructions", "instructions must be a string", .{}),
    };
    const input = o.get("input") orelse return refuse(bad, "missing_required_parameter", "input", "input is required: a string or an array of items", .{});
    switch (input) {
        .string => try messages.append(try object(a, .{ .role = string("user"), .content = input })),
        .array => |items| try conversation(a, items.items, &messages, bad),
        else => return refuse(bad, "invalid_type", "input", "input must be a string or an array of items", .{}),
    }
    if (messages.items.len == 0) return refuse(bad, "invalid_value", "input", "input has no messages", .{});
    try out.put(a, "messages", .{ .array = messages });

    if (o.get("tools")) |t| if (t != .null) {
        if (t != .array) return refuse(bad, "invalid_type", "tools", "tools must be an array", .{});
        asked.tools = t;
        if (t.array.items.len > 0) try out.put(a, "tools", try functionTools(a, t.array.items, bad));
    };
    if (o.get("tool_choice")) |c| if (c != .null) {
        asked.tool_choice = c;
        try out.put(a, "tool_choice", try toolChoice(a, c, bad));
    };
    if (o.get("parallel_tool_calls")) |p| switch (p) {
        .bool => |b| {
            asked.parallel_tool_calls = b;
            try out.put(a, "parallel_tool_calls", p);
        },
        .null => {},
        else => return refuse(bad, "invalid_type", "parallel_tool_calls", "parallel_tool_calls must be true or false", .{}),
    };
    inline for (.{ "temperature", "top_p" }) |k| if (o.get(k)) |v| switch (v) {
        .integer, .float => {
            @field(asked, k) = v;
            try out.put(a, k, v);
        },
        .null => {},
        else => return refuse(bad, "invalid_type", k, k ++ " must be a number", .{}),
    };
    if (o.get("max_output_tokens")) |v| switch (v) {
        .integer => |n| {
            if (n < 1) return refuse(bad, "invalid_value", "max_output_tokens", "max_output_tokens must be at least 1", .{});
            asked.max_output_tokens = v;
            try out.put(a, "max_tokens", v); // the reply's whole limit, reasoning included, as in Responses
        },
        .null => {},
        else => return refuse(bad, "invalid_type", "max_output_tokens", "max_output_tokens must be an integer", .{}),
    };
    if (o.get("reasoning")) |r| if (r != .null) try reasoning(a, r, &asked, &out, bad);
    if (o.get("text")) |t| if (t != .null) try textFormat(a, t, &asked, &out, bad);
    if (o.get("metadata")) |m| switch (m) {
        .object, .null => asked.metadata = m,
        else => return refuse(bad, "invalid_type", "metadata", "metadata must be an object of strings", .{}),
    };
    if (asked.stream) {
        try out.put(a, "stream", .{ .bool = true });
        try out.put(a, "stream_options", try object(a, .{ .include_usage = Value{ .bool = true } }));
    }
    return .{ .chat = .{ .object = out }, .asked = asked };
}

fn refuse(bad: *Bad, code: []const u8, param: []const u8, comptime fmt: []const u8, args: anytype) Refused {
    bad.set(code, param, fmt, args);
    return error.Refused;
}

/// The fields that ask for state or services this box does not keep, each refused with what to do instead. The
/// accepted values are the ones that change nothing (`store: false`, `background: false`, `truncation: disabled`...).
fn unsupported(o: ObjectMap, bad: *Bad) Refused!void {
    if (!isNull(o.get("previous_response_id")))
        return refuse(bad, "unsupported_parameter", "previous_response_id", "responses are not stored here: send the whole conversation in input (with store false) instead of previous_response_id", .{});
    if (!isNull(o.get("conversation")))
        return refuse(bad, "unsupported_parameter", "conversation", "conversations are not stored here: send the whole conversation in input", .{});
    if (!isNull(o.get("prompt")))
        return refuse(bad, "unsupported_parameter", "prompt", "stored prompt templates are not offered: send instructions and input", .{});
    if (o.get("store")) |s| if (s == .bool and s.bool)
        return refuse(bad, "unsupported_value", "store", "responses are not stored here: send store false and replay the conversation each turn", .{});
    if (o.get("background")) |b| if (b == .bool and b.bool)
        return refuse(bad, "unsupported_value", "background", "background responses are not offered: wait for the response, and close the connection to cancel it", .{});
    if (str(o.get("truncation"))) |t| if (!eql(t, "disabled"))
        return refuse(bad, "unsupported_value", "truncation", "truncation {s} is not offered: only disabled (an input over the model's context is refused)", .{t});
    if (o.get("top_logprobs")) |t| if (!(t == .null or (t == .integer and t.integer == 0)))
        return refuse(bad, "unsupported_parameter", "top_logprobs", "log probabilities are not offered", .{});
    if (!isNull(o.get("max_tool_calls")))
        return refuse(bad, "unsupported_parameter", "max_tool_calls", "max_tool_calls is not offered: the caller runs the tools and decides when to stop", .{});
    if (str(o.get("service_tier"))) |t| if (!eql(t, "auto") and !eql(t, "default"))
        return refuse(bad, "unsupported_value", "service_tier", "service_tier {s} is not offered: one tier (auto or default)", .{t});
    if (o.get("include")) |inc| if (inc != .null) {
        if (inc != .array) return refuse(bad, "invalid_type", "include", "include must be an array", .{});
        // Reasoning is always returned as the reasoning item's reasoning_text, which replays as it is; nothing is
        // encrypted, so the item carries no encrypted_content.
        for (inc.array.items) |x| if (!(x == .string and eql(x.string, "reasoning.encrypted_content")))
            return refuse(bad, "unsupported_value", "include", "include {s} is not offered: only reasoning.encrypted_content (reasoning comes back as reasoning_text)", .{if (x == .string) x.string else "(not a string)"});
    };
    if (o.get("stream_options")) |so| if (so != .null) {
        if (so != .object) return refuse(bad, "invalid_type", "stream_options", "stream_options must be an object", .{});
        var it = so.object.iterator();
        while (it.next()) |e| if (!eql(e.key_ptr.*, "include_obfuscation"))
            return refuse(bad, "unknown_parameter", "stream_options", "Unknown parameter: 'stream_options.{s}'.", .{e.key_ptr.*});
    };
}

/// The input items as chat messages. An assistant turn (its reasoning, its message and its function calls, in that
/// order) becomes one assistant message with `reasoning_content`, `content` and `tool_calls`; a function's output a
/// `tool` message; `developer` keeps its place as a `system` message.
fn conversation(a: Allocator, items: []const Value, messages: *std.json.Array, bad: *Bad) Error!void {
    var turn: ?ObjectMap = null; // the assistant turn being assembled
    for (items, 0..) |item, i| {
        if (item != .object) return refuse(bad, "invalid_type", "input", "input[{d}] is not an item object", .{i});
        const o = item.object;
        const typ = str(o.get("type")) orelse (if (o.get("role") != null) "message" else return refuse(bad, "missing_required_parameter", "input", "input[{d}] has no type", .{i}));
        if (eql(typ, "message")) {
            const role = str(o.get("role")) orelse return refuse(bad, "missing_required_parameter", "input", "input[{d}] is a message without a role", .{i});
            const content = try messageContent(a, o.get("content"), i, role, bad);
            if (eql(role, "assistant")) {
                if (turn) |t| if (t.get("content") != null or t.get("tool_calls") != null) try flush(a, messages, &turn);
                if (turn == null) turn = .empty;
                try turn.?.put(a, "content", content);
                continue;
            }
            const chat_role = if (eql(role, "user")) "user" else if (eql(role, "system") or eql(role, "developer")) "system" else return refuse(bad, "invalid_value", "input", "input[{d}] has role {s}: use user, assistant, system or developer", .{ i, role });
            try flush(a, messages, &turn);
            try messages.append(try object(a, .{ .role = string(chat_role), .content = content }));
        } else if (eql(typ, "function_call")) {
            const call_id = str(o.get("call_id")) orelse return refuse(bad, "missing_required_parameter", "input", "input[{d}] is a function_call without a call_id", .{i});
            const name = str(o.get("name")) orelse return refuse(bad, "missing_required_parameter", "input", "input[{d}] is a function_call without a name", .{i});
            const args = str(o.get("arguments")) orelse return refuse(bad, "missing_required_parameter", "input", "input[{d}] is a function_call without arguments (a JSON string)", .{i});
            if (turn == null) turn = .empty;
            const calls = turn.?.getPtr("tool_calls") orelse blk: {
                try turn.?.put(a, "tool_calls", .{ .array = std.json.Array.init(a) });
                break :blk turn.?.getPtr("tool_calls").?;
            };
            try calls.array.append(try object(a, .{ .id = string(call_id), .type = string("function"), .function = try object(a, .{ .name = string(name), .arguments = string(args) }) }));
        } else if (eql(typ, "function_call_output")) {
            const call_id = str(o.get("call_id")) orelse return refuse(bad, "missing_required_parameter", "input", "input[{d}] is a function_call_output without a call_id", .{i});
            const output = try callOutput(a, o.get("output"), i, bad);
            try flush(a, messages, &turn);
            try messages.append(try object(a, .{ .role = string("tool"), .tool_call_id = string(call_id), .content = output }));
        } else if (eql(typ, "reasoning")) {
            if (!isNull(o.get("encrypted_content")))
                return refuse(bad, "unsupported_value", "input", "input[{d}] is encrypted reasoning, which this box cannot read (it never makes any): drop it or replay the reasoning item this box returned", .{i});
            const text = try reasoningText(a, o, i, bad);
            if (turn != null) try flush(a, messages, &turn);
            turn = .empty;
            if (text.len > 0) try turn.?.put(a, "reasoning_content", string(text));
        } else if (eql(typ, "item_reference")) {
            return refuse(bad, "unsupported_value", "input", "input[{d}] is an item_reference: items are not stored here, send the item itself", .{i});
        } else {
            return refuse(bad, "unsupported_value", "input", "input[{d}] is a {s} item: only message, function_call, function_call_output and reasoning items are offered", .{ i, typ });
        }
    }
    try flush(a, messages, &turn);
}

fn flush(a: Allocator, messages: *std.json.Array, turn: *?ObjectMap) Allocator.Error!void {
    var t = turn.* orelse return;
    turn.* = null;
    try t.put(a, "role", string("assistant"));
    if (t.get("content") == null) try t.put(a, "content", .null);
    try messages.append(.{ .object = t });
}

/// A message's content: a string as it is, an array of text parts as chat text parts. Images, files and audio are
/// refused: the chat model takes text only.
fn messageContent(a: Allocator, content: ?Value, i: usize, role: []const u8, bad: *Bad) Error!Value {
    const c = content orelse return refuse(bad, "missing_required_parameter", "input", "input[{d}] is a message without content", .{i});
    switch (c) {
        .string => return c,
        .array => |parts| {
            var out = std.json.Array.init(a);
            for (parts.items, 0..) |p, j| {
                const typ = if (p == .object) str(p.object.get("type")) orelse "" else "";
                const text = if (eql(typ, "refusal")) str(p.object.get("refusal")) else if (eql(typ, "input_text") or eql(typ, "output_text") or eql(typ, "text")) str(p.object.get("text")) else null;
                if (text == null) {
                    if (eql(typ, "input_image") or eql(typ, "input_file") or eql(typ, "input_audio"))
                        return refuse(bad, "unsupported_value", "input", "input[{d}].content[{d}] is {s}: the chat model here takes text only", .{ i, j, typ });
                    return refuse(bad, "invalid_value", "input", "input[{d}].content[{d}] is not a text part of a {s} message", .{ i, j, role });
                }
                try out.append(try object(a, .{ .type = string("text"), .text = string(text.?) }));
            }
            return .{ .array = out };
        },
        else => return refuse(bad, "invalid_type", "input", "input[{d}].content must be a string or an array of parts", .{i}),
    }
}

/// A function's output: a string, or an array of text parts joined. Images and files are refused (text only).
fn callOutput(a: Allocator, output: ?Value, i: usize, bad: *Bad) Error!Value {
    const v = output orelse return refuse(bad, "missing_required_parameter", "input", "input[{d}] is a function_call_output without output", .{i});
    switch (v) {
        .string => return v,
        .array => |parts| {
            var text: std.ArrayList(u8) = .empty;
            for (parts.items, 0..) |p, j| {
                const typ = if (p == .object) str(p.object.get("type")) orelse "" else "";
                if (!eql(typ, "input_text") and !eql(typ, "output_text") and !eql(typ, "text"))
                    return refuse(bad, "unsupported_value", "input", "input[{d}].output[{d}] is {s}: function outputs here are text only", .{ i, j, if (typ.len > 0) typ else "not a part" });
                try text.appendSlice(a, str(p.object.get("text")) orelse "");
            }
            return string(text.items);
        },
        else => return refuse(bad, "invalid_type", "input", "input[{d}].output must be a string or an array of text parts", .{i}),
    }
}

/// A replayed reasoning item's text: its `reasoning_text` content, else its summary.
fn reasoningText(a: Allocator, o: ObjectMap, i: usize, bad: *Bad) Error![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    inline for (.{ .{ "content", "reasoning_text" }, .{ "summary", "summary_text" } }) |src| {
        if (o.get(src[0])) |list| if (list == .array) for (list.array.items) |p| {
            const typ = if (p == .object) str(p.object.get("type")) orelse "" else "";
            if (!eql(typ, src[1])) return refuse(bad, "invalid_value", "input", "input[{d}].{s} holds a part that is not {s}", .{ i, src[0], src[1] });
            try text.appendSlice(a, str(p.object.get("text")) orelse "");
        };
        if (text.items.len > 0) return text.items;
    }
    return text.items;
}

/// Function tools as chat tools. Every other tool type is refused: hosted tools need OpenAI's services, and custom
/// (freeform) and namespace tools have no chat equivalent here.
fn functionTools(a: Allocator, tools: []const Value, bad: *Bad) Error!Value {
    var out = std.json.Array.init(a);
    for (tools, 0..) |t, i| {
        const typ = if (t == .object) str(t.object.get("type")) orelse "" else "";
        if (!eql(typ, "function")) {
            if (eql(typ, "custom") or eql(typ, "namespace"))
                return refuse(bad, "unsupported_tool", "tools", "tools[{d}] is a {s} tool: only function tools are offered (in Codex, turn off the feature that adds it)", .{ i, typ });
            if (typ.len == 0) return refuse(bad, "invalid_value", "tools", "tools[{d}] has no type", .{i});
            return refuse(bad, "unsupported_tool", "tools", "tools[{d}] is a {s} tool: hosted tools are not offered here, only function tools the caller runs", .{ i, typ });
        }
        const name = str(t.object.get("name")) orelse return refuse(bad, "missing_required_parameter", "tools", "tools[{d}] is a function without a name", .{i});
        var f: ObjectMap = .empty;
        try f.put(a, "name", string(name));
        if (t.object.get("description")) |d| if (d != .null) try f.put(a, "description", d);
        try f.put(a, "parameters", t.object.get("parameters") orelse try object(a, .{ .type = string("object"), .properties = Value{ .object = .empty } }));
        if (t.object.get("strict")) |s| if (s == .bool) try f.put(a, "strict", s);
        try out.append(try object(a, .{ .type = string("function"), .function = Value{ .object = f } }));
    }
    return .{ .array = out };
}

fn toolChoice(a: Allocator, c: Value, bad: *Bad) Error!Value {
    if (c == .string) {
        for ([_][]const u8{ "none", "auto", "required" }) |ok| if (eql(c.string, ok)) return c;
        return refuse(bad, "invalid_value", "tool_choice", "tool_choice {s}: use none, auto, required or {{type: function, name}}", .{c.string});
    }
    const typ = if (c == .object) str(c.object.get("type")) orelse "" else "";
    if (eql(typ, "function")) {
        const name = str(c.object.get("name")) orelse return refuse(bad, "missing_required_parameter", "tool_choice", "tool_choice names no function", .{});
        return object(a, .{ .type = string("function"), .function = try object(a, .{ .name = string(name) }) });
    }
    return refuse(bad, "unsupported_value", "tool_choice", "tool_choice {s} is not offered: use none, auto, required or {{type: function, name}}", .{if (typ.len > 0) typ else "(no type)"});
}

/// `reasoning.effort` goes to the model's server as `reasoning_effort` (Qwen on TensorFold: none switches thinking off;
/// minimal to max think, and the server refuses other values). There is no summarizer: `summary` auto or none (no
/// summary is made; the reasoning itself is returned as reasoning_text), concise and detailed are refused.
fn reasoning(a: Allocator, r: Value, asked: *Asked, out: *ObjectMap, bad: *Bad) Error!void {
    if (r != .object) return refuse(bad, "invalid_type", "reasoning", "reasoning must be an object", .{});
    var it = r.object.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const v = e.value_ptr.*;
        if (eql(k, "effort")) {
            if (v == .null) continue;
            if (v != .string) return refuse(bad, "invalid_type", "reasoning.effort", "reasoning.effort must be a string", .{});
            asked.effort = v;
            try out.put(a, "reasoning_effort", v);
        } else if (eql(k, "summary") or eql(k, "generate_summary")) {
            if (v == .null) continue;
            if (v != .string) return refuse(bad, "invalid_type", "reasoning.summary", "reasoning.summary must be a string", .{});
            if (!eql(v.string, "auto") and !eql(v.string, "none"))
                return refuse(bad, "unsupported_value", "reasoning.summary", "reasoning summaries are not made here: use summary auto or none (the reasoning itself comes back as reasoning_text)", .{});
            asked.summary = v;
        } else return refuse(bad, "unknown_parameter", "reasoning", "Unknown parameter: 'reasoning.{s}'.", .{k});
    }
}

/// `text.format` as the chat `response_format`: text (the default), json_object, or json_schema with its name,
/// schema, strict and description. `verbosity` has no control here: only medium (the default).
fn textFormat(a: Allocator, t: Value, asked: *Asked, out: *ObjectMap, bad: *Bad) Error!void {
    if (t != .object) return refuse(bad, "invalid_type", "text", "text must be an object", .{});
    var it = t.object.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const v = e.value_ptr.*;
        if (eql(k, "verbosity")) {
            if (str(v)) |s| if (!eql(s, "medium")) return refuse(bad, "unsupported_value", "text.verbosity", "text.verbosity {s} is not offered: only medium", .{s});
        } else if (eql(k, "format")) {
            if (v == .null) continue;
            const typ = if (v == .object) str(v.object.get("type")) orelse "" else "";
            asked.format = v;
            if (eql(typ, "text")) continue;
            if (eql(typ, "json_object")) {
                try out.put(a, "response_format", try object(a, .{ .type = string("json_object") }));
            } else if (eql(typ, "json_schema")) {
                var s: ObjectMap = .empty;
                try s.put(a, "name", string(str(v.object.get("name")) orelse return refuse(bad, "missing_required_parameter", "text.format.name", "a json_schema format needs a name", .{})));
                try s.put(a, "schema", v.object.get("schema") orelse return refuse(bad, "missing_required_parameter", "text.format.schema", "a json_schema format needs a schema", .{}));
                inline for (.{ "strict", "description" }) |f| if (v.object.get(f)) |x| if (x != .null) try s.put(a, f, x);
                try out.put(a, "response_format", try object(a, .{ .type = string("json_schema"), .json_schema = Value{ .object = s } }));
            } else return refuse(bad, "invalid_value", "text.format", "text.format {s}: use text, json_object or json_schema", .{if (typ.len > 0) typ else "(no type)"});
        } else return refuse(bad, "unknown_parameter", "text", "Unknown parameter: 'text.{s}'.", .{k});
    }
}

// ---------------------------------------------------------------- the response object

/// What a finished response says about itself.
pub const Outcome = struct {
    status: []const u8 = "completed", // completed, incomplete or failed
    incomplete: ?[]const u8 = null, // incomplete_details.reason
    failure: ?struct { code: []const u8, message: []const u8 } = null,
    usage: Value = .null,
};

/// The response object, OpenAI's fields in OpenAI's order. `output` and `usage` are as made so far.
pub fn responseObject(a: Allocator, id: []const u8, created_at: i64, asked: *const Asked, output: []const Value, o: Outcome) Allocator.Error!Value {
    var out = std.json.Array.init(a);
    try out.appendSlice(output);
    var m: ObjectMap = .empty;
    try m.put(a, "id", string(id));
    try m.put(a, "object", string("response"));
    try m.put(a, "created_at", .{ .integer = created_at });
    try m.put(a, "status", string(o.status));
    try m.put(a, "background", .{ .bool = false });
    try m.put(a, "error", if (o.failure) |f| try object(a, .{ .code = string(f.code), .message = string(f.message) }) else Value.null);
    try m.put(a, "incomplete_details", if (o.incomplete) |r| try object(a, .{ .reason = string(r) }) else Value.null);
    try m.put(a, "instructions", asked.instructions);
    try m.put(a, "max_output_tokens", asked.max_output_tokens);
    try m.put(a, "max_tool_calls", .null);
    try m.put(a, "model", string(asked.model));
    try m.put(a, "output", .{ .array = out });
    try m.put(a, "parallel_tool_calls", .{ .bool = asked.parallel_tool_calls });
    try m.put(a, "previous_response_id", .null);
    try m.put(a, "reasoning", try object(a, .{ .effort = asked.effort, .summary = asked.summary }));
    try m.put(a, "store", .{ .bool = false });
    try m.put(a, "temperature", asked.temperature);
    try m.put(a, "text", try object(a, .{ .format = if (asked.format == .null) try object(a, .{ .type = string("text") }) else asked.format }));
    try m.put(a, "tool_choice", if (asked.tool_choice == .null) string("auto") else asked.tool_choice);
    try m.put(a, "tools", if (asked.tools == .null) Value{ .array = std.json.Array.init(a) } else asked.tools);
    try m.put(a, "top_logprobs", .{ .integer = 0 });
    try m.put(a, "top_p", asked.top_p);
    try m.put(a, "truncation", string("disabled"));
    try m.put(a, "usage", o.usage);
    try m.put(a, "metadata", if (asked.metadata == .null) Value{ .object = .empty } else asked.metadata);
    return .{ .object = m };
}

/// Responses usage from a chat completion's.
fn usage(a: Allocator, u: ?Value) Allocator.Error!Value {
    const x = u orelse return .null;
    if (x != .object) return .null;
    const int = struct {
        fn f(v: ?Value, path: []const []const u8) i64 {
            var cur = v orelse return 0;
            for (path) |k| cur = if (cur == .object) cur.object.get(k) orelse return 0 else return 0;
            return if (cur == .integer) cur.integer else 0;
        }
    }.f;
    const in = int(x, &.{"prompt_tokens"});
    const out = int(x, &.{"completion_tokens"});
    return object(a, .{
        .input_tokens = Value{ .integer = in },
        .input_tokens_details = try object(a, .{ .cached_tokens = Value{ .integer = int(x, &.{ "prompt_tokens_details", "cached_tokens" }) } }),
        .output_tokens = Value{ .integer = out },
        .output_tokens_details = try object(a, .{ .reasoning_tokens = Value{ .integer = int(x, &.{ "completion_tokens_details", "reasoning_tokens" }) } }),
        .total_tokens = Value{ .integer = int(x, &.{"total_tokens"}) },
    });
}

/// Item ids: a prefix and 24 random hex digits.
pub const Ids = struct {
    io: Io,
    a: Allocator,

    pub fn make(ids: Ids, prefix: []const u8) Allocator.Error![]const u8 {
        var rnd: [12]u8 = undefined;
        ids.io.random(&rnd);
        const hex = std.fmt.bytesToHex(rnd, .lower);
        return std.fmt.allocPrint(ids.a, "{s}_{s}", .{ prefix, &hex });
    }
};

fn incompleteReason(finish: []const u8) ?[]const u8 {
    if (eql(finish, "length")) return "max_output_tokens";
    if (eql(finish, "content_filter")) return "content_filter";
    return null;
}

fn reasoningItem(a: Allocator, id: []const u8, text: []const u8) Allocator.Error!Value {
    var parts = std.json.Array.init(a);
    try parts.append(try object(a, .{ .type = string("reasoning_text"), .text = string(text) }));
    return object(a, .{ .id = string(id), .type = string("reasoning"), .summary = Value{ .array = std.json.Array.init(a) }, .content = Value{ .array = parts } });
}

fn textPart(a: Allocator, text: []const u8) Allocator.Error!Value {
    return object(a, .{ .type = string("output_text"), .text = string(text), .annotations = Value{ .array = std.json.Array.init(a) }, .logprobs = Value{ .array = std.json.Array.init(a) } });
}

fn messageItem(a: Allocator, id: []const u8, status: []const u8, text: ?[]const u8) Allocator.Error!Value {
    var parts = std.json.Array.init(a);
    if (text) |t| try parts.append(try textPart(a, t));
    return object(a, .{ .id = string(id), .type = string("message"), .status = string(status), .role = string("assistant"), .content = Value{ .array = parts } });
}

fn callItem(a: Allocator, id: []const u8, status: []const u8, call_id: []const u8, name: []const u8, arguments: []const u8) Allocator.Error!Value {
    return object(a, .{ .id = string(id), .type = string("function_call"), .status = string(status), .call_id = string(call_id), .name = string(name), .arguments = string(arguments) });
}

/// A whole chat completion as a response object: its reasoning, its text and its tool calls, in that order, each an
/// output item. The model server's call ids are kept (they are what the caller answers).
pub fn fromCompletion(a: Allocator, ids: Ids, id: []const u8, created_at: i64, asked: *const Asked, completion: []const u8) Error!Value {
    const v = std.json.parseFromSliceLeaky(Value, a, completion, .{}) catch return error.Refused;
    if (v != .object) return error.Refused;
    const choices = v.object.get("choices") orelse return error.Refused;
    if (choices != .array or choices.array.items.len == 0 or choices.array.items[0] != .object) return error.Refused;
    const choice = choices.array.items[0].object;
    const msg = (choice.get("message") orelse return error.Refused);
    if (msg != .object) return error.Refused;
    const finish = str(choice.get("finish_reason")) orelse "stop";
    const incomplete = incompleteReason(finish);
    var output: std.ArrayList(Value) = .empty;
    if (str(msg.object.get("reasoning_content"))) |r| if (r.len > 0) try output.append(a, try reasoningItem(a, try ids.make("rs"), r));
    if (str(msg.object.get("content"))) |c| if (c.len > 0) try output.append(a, try messageItem(a, try ids.make("msg"), if (incomplete != null) "incomplete" else "completed", c));
    if (msg.object.get("tool_calls")) |calls| if (calls == .array) for (calls.array.items) |c| {
        if (c != .object) continue;
        const f = c.object.get("function") orelse continue;
        if (f != .object) continue;
        const call_id = str(c.object.get("id")) orelse try ids.make("call");
        try output.append(a, try callItem(a, try ids.make("fc"), "completed", call_id, str(f.object.get("name")) orelse "", str(f.object.get("arguments")) orelse "{}"));
    };
    return responseObject(a, id, created_at, asked, output.items, .{
        .status = if (incomplete != null) "incomplete" else "completed",
        .incomplete = incomplete,
        .usage = try usage(a, v.object.get("usage")),
    });
}

// ---------------------------------------------------------------- the route

fn failBad(req: *h.Request, a: Allocator, bad: *const Bad) !void {
    const Env = struct { @"error": struct { message: []const u8, type: []const u8, param: ?[]const u8, code: ?[]const u8 } };
    const text = try std.json.Stringify.valueAlloc(a, Env{ .@"error" = .{ .message = bad.message, .type = bad.typ, .param = bad.param, .code = bad.code } }, .{});
    try req.respond(text, .{ .status = @fromBackingInt(@intCast(bad.status)), .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
}

/// POST /v1/responses.
pub fn create(app: *App, req: *h.Request, a: Allocator, owner: []const u8) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const body = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return h.fail(req, a, 400, "invalid_request_error", "the body is not a valid Responses request (a JSON object with input)");
    var bad: Bad = .{};
    var t = translate(a, body, &bad) catch |err| switch (err) {
        error.Refused => return failBad(req, a, &bad),
        error.OutOfMemory => return error.OutOfMemory,
    };
    var why: routes.Refusal = .{};
    const j = chat.startChat(app, a, t.chat, owner, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    t.asked.model = app.scheduler.slots[j.tool].tool.config().id;
    const ids: Ids = .{ .io = app.io, .a = a };
    const id = try ids.make("resp");
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const completion = chat.readCompletion(app, a, j, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    const r = fromCompletion(a, ids, id, j.created_at, &t.asked, completion) catch |err| switch (err) {
        error.Refused => return h.fail(req, a, 502, "server_error", "the model's server sent a reply that is not a chat completion"),
        error.OutOfMemory => return error.OutOfMemory,
    };
    try h.jsonValue(req, a, .ok, r);
}

/// The other /v1/responses/... routes: nothing is stored, so there is nothing to get, delete, cancel or list.
pub fn stored(req: *h.Request, a: Allocator) !void {
    var bad: Bad = .{ .status = 404 };
    bad.set("not_found", "response_id", "responses are not stored here (store is always false): replay the conversation in input, and cancel a response by closing its connection", .{});
    return failBad(req, a, &bad);
}

// ---------------------------------------------------------------- tests

const testing = std.testing;

fn parse(a: Allocator, text: []const u8) !Value {
    return std.json.parseFromSliceLeaky(Value, a, text, .{});
}

fn json(a: Allocator, v: Value) ![]const u8 {
    return std.json.Stringify.valueAlloc(a, v, .{});
}

fn refusedWith(a: Allocator, text: []const u8, code: []const u8, needle: []const u8) !void {
    var bad: Bad = .{};
    if (translate(a, try parse(a, text), &bad)) |_| {
        std.debug.print("accepted: {s}\n", .{text});
        return error.TestExpectedError;
    } else |err| try testing.expectEqual(error.Refused, err);
    testing.expectEqualStrings(code, bad.code.?) catch |e| {
        std.debug.print("{s}: {s}\n", .{ text, bad.message });
        return e;
    };
    if (std.mem.indexOf(u8, bad.message, needle) == null) {
        std.debug.print("'{s}' not in '{s}'\n", .{ needle, bad.message });
        return error.TestExpectedEqual;
    }
}

test "a Codex-shaped turn translates to one chat body: developer as system in place, parts kept, calls merged" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bad: Bad = .{};
    const t = try translate(a, try parse(a,
        \\{"model":"m","instructions":"be brief","store":false,"include":["reasoning.encrypted_content"],
        \\ "reasoning":{"effort":"low","summary":"auto"},"prompt_cache_key":"k","client_metadata":{"x":"y"},
        \\ "tool_choice":"auto","parallel_tool_calls":true,"max_output_tokens":64,
        \\ "tools":[{"type":"function","name":"exec_command","description":"run","strict":false,"parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}],
        \\ "input":[
        \\  {"type":"message","id":"msg_1","role":"developer","content":[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]},
        \\  {"type":"message","role":"user","content":[{"type":"input_text","text":"make hi.txt"}]},
        \\  {"type":"reasoning","id":"rs_1","summary":[],"content":[{"type":"reasoning_text","text":"I should write it"}],"encrypted_content":null},
        \\  {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Writing it."}]},
        \\  {"type":"function_call","call_id":"call_a","name":"exec_command","arguments":"{\"cmd\":\"echo hi > hi.txt\"}"},
        \\  {"type":"function_call","call_id":"call_b","name":"exec_command","arguments":"{\"cmd\":\"cat hi.txt\"}"},
        \\  {"type":"function_call_output","call_id":"call_a","output":"ok"},
        \\  {"type":"function_call_output","call_id":"call_b","output":[{"type":"input_text","text":"hi"}]},
        \\  {"role":"user","content":"thanks"}
        \\ ]}
    ), &bad);
    try testing.expectEqualStrings(
        \\{"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"system","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},{"role":"user","content":[{"type":"text","text":"make hi.txt"}]},{"reasoning_content":"I should write it","content":[{"type":"text","text":"Writing it."}],"tool_calls":[{"id":"call_a","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"echo hi > hi.txt\"}"}},{"id":"call_b","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"cat hi.txt\"}"}}],"role":"assistant"},{"role":"tool","tool_call_id":"call_a","content":"ok"},{"role":"tool","tool_call_id":"call_b","content":"hi"},{"role":"user","content":"thanks"}],"tools":[{"type":"function","function":{"name":"exec_command","description":"run","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}},"strict":false}}],"tool_choice":"auto","parallel_tool_calls":true,"max_tokens":64,"reasoning_effort":"low"}
    , try json(a, t.chat));
    try testing.expectEqualStrings("auto", t.asked.summary.string);
}

test "a string input, json_schema text format and a named tool_choice" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bad: Bad = .{};
    const t = try translate(a, try parse(a,
        \\{"input":"hi","text":{"format":{"type":"json_schema","name":"r","schema":{"type":"object"},"strict":true},"verbosity":"medium"},
        \\ "tools":[{"type":"function","name":"f"}],"tool_choice":{"type":"function","name":"f"},"temperature":0}
    ), &bad);
    try testing.expectEqualStrings(
        \\{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}],"tool_choice":{"type":"function","function":{"name":"f"}},"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"r","schema":{"type":"object"},"strict":true}}}
    , try json(a, t.chat));
}

test "what the box cannot do is a 400 naming it, before anything runs" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try refusedWith(a, "{\"input\":\"x\",\"previous_response_id\":\"resp_1\"}", "unsupported_parameter", "not stored");
    try refusedWith(a, "{\"input\":\"x\",\"store\":true}", "unsupported_value", "store false");
    try refusedWith(a, "{\"input\":\"x\",\"background\":true}", "unsupported_value", "background");
    try refusedWith(a, "{\"input\":\"x\",\"stream\":true}", "unsupported_value", "stream false");
    try refusedWith(a, "{\"input\":\"x\",\"conversation\":\"conv_1\"}", "unsupported_parameter", "conversations");
    try refusedWith(a, "{\"input\":\"x\",\"tools\":[{\"type\":\"web_search\"}]}", "unsupported_tool", "hosted tools");
    try refusedWith(a, "{\"input\":\"x\",\"tools\":[{\"type\":\"namespace\",\"name\":\"multi_agent_v1\",\"tools\":[]}]}", "unsupported_tool", "namespace");
    try refusedWith(a, "{\"input\":\"x\",\"tools\":[{\"type\":\"custom\",\"name\":\"apply_patch\"}]}", "unsupported_tool", "custom");
    try refusedWith(a, "{\"input\":\"x\",\"tool_choice\":{\"type\":\"web_search\"}}", "unsupported_value", "web_search");
    try refusedWith(a, "{\"input\":\"x\",\"reasoning\":{\"summary\":\"detailed\"}}", "unsupported_value", "summaries");
    try refusedWith(a, "{\"input\":\"x\",\"include\":[\"message.output_text.logprobs\"]}", "unsupported_value", "logprobs");
    try refusedWith(a, "{\"input\":\"x\",\"text\":{\"verbosity\":\"low\"}}", "unsupported_value", "verbosity");
    try refusedWith(a, "{\"input\":\"x\",\"truncation\":\"auto\"}", "unsupported_value", "truncation");
    try refusedWith(a, "{\"input\":\"x\",\"frequency_penalty\":1}", "unknown_parameter", "frequency_penalty");
    try refusedWith(a, "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"data:\"}]}]}", "unsupported_value", "text only");
    try refusedWith(a, "{\"input\":[{\"type\":\"item_reference\",\"id\":\"msg_1\"}]}", "unsupported_value", "not stored");
    try refusedWith(a, "{\"input\":[{\"type\":\"reasoning\",\"summary\":[],\"encrypted_content\":\"gAAAA\"}]}", "unsupported_value", "encrypted");
    try refusedWith(a, "{\"input\":[{\"type\":\"local_shell_call\",\"call_id\":\"c\"}]}", "unsupported_value", "local_shell_call");
    try refusedWith(a, "{\"input\":[{\"role\":\"critic\",\"content\":\"x\"}]}", "invalid_value", "critic");
    try refusedWith(a, "{\"model\":\"m\"}", "missing_required_parameter", "input");
    // What changes nothing is accepted: the defaults spelled out.
    var bad: Bad = .{};
    _ = try translate(a, try parse(a, "{\"input\":\"x\",\"store\":false,\"background\":false,\"truncation\":\"disabled\",\"service_tier\":\"auto\",\"top_logprobs\":0,\"previous_response_id\":null,\"stream_options\":{\"include_obfuscation\":false},\"user\":\"u\",\"safety_identifier\":\"s\"}"), &bad);
}

test "a whole completion becomes reasoning, message and function_call items with the server's call ids" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const asked: Asked = .{ .model = "qwen" };
    const r = try fromCompletion(a, .{ .io = testing.io, .a = a }, "resp_1", 7, &asked,
        \\{"choices":[{"index":0,"message":{"role":"assistant","content":"Doing it.","reasoning_content":"think","tool_calls":[{"id":"call_x","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},"finish_reason":"tool_calls"}],
        \\ "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":2}}}
    );
    const o = r.object;
    try testing.expectEqualStrings("completed", o.get("status").?.string);
    try testing.expectEqual(false, o.get("store").?.bool);
    const out = o.get("output").?.array.items;
    try testing.expectEqual(@as(usize, 3), out.len);
    try testing.expectEqualStrings("reasoning", out[0].object.get("type").?.string);
    try testing.expectEqualStrings("think", out[0].object.get("content").?.array.items[0].object.get("text").?.string);
    try testing.expectEqualStrings("Doing it.", out[1].object.get("content").?.array.items[0].object.get("text").?.string);
    try testing.expectEqualStrings("call_x", out[2].object.get("call_id").?.string);
    try testing.expect(std.mem.startsWith(u8, out[2].object.get("id").?.string, "fc_"));
    try testing.expectEqual(@as(i64, 2), o.get("usage").?.object.get("output_tokens_details").?.object.get("reasoning_tokens").?.integer);

    const cut = try fromCompletion(a, .{ .io = testing.io, .a = a }, "resp_2", 7, &asked, "{\"choices\":[{\"message\":{\"content\":\"a b\"},\"finish_reason\":\"length\"}]}");
    try testing.expectEqualStrings("incomplete", cut.object.get("status").?.string);
    try testing.expectEqualStrings("max_output_tokens", cut.object.get("incomplete_details").?.object.get("reason").?.string);
}
