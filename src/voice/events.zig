//! The OpenAI Realtime API's session object and server events for the voice loop: settings with the daemon's defaults,
//! `session.update` applied field by field (each refusal names the field), and the emitter that stamps event ids.

const std = @import("std");
const Io = std.Io;
const config = @import("../config.zig");
const vad = @import("vad.zig");
const backend = @import("backend.zig");
const chat = @import("chat.zig");

const Value = std.json.Value;

/// Who runs the model's tool calls (`session.localrouter.tool_execution`). `server`: the built-in image, video, model
/// and computer tools are offered beside the client's and run here. `client`: only the client's tools exist, every call
/// goes to the client and its output must answer that call while it is still waiting (docs/dev/VOICE.md).
pub const ToolExecution = enum { server, client };

/// A session's settings; strings live in the session's settings arena.
pub const Settings = struct {
    tool_execution: ToolExecution = .server,
    audio: bool = true, // modalities include audio (else text only)
    instructions: []const u8 = "",
    voice: []const u8 = "",
    model: []const u8 = "", // the chat model; "": the chat server's default
    speech_model: []const u8 = "",
    transcription_model: []const u8 = "",
    language: []const u8 = "",
    transcription_prompt: []const u8 = "",
    vad: ?vad.Config = .{}, // null: push to talk (turn_detection null)
    create_response: bool = true,
    interrupt_response: bool = true,
    tools_json: []const u8 = "[]", // the client's tools, in the Realtime API's shape
    tool_choice_json: []const u8 = "\"auto\"",
    temperature: ?f32 = null,
    max_tokens: ?u32 = null,
    partial_interval_ms: u32 = 0,

    pub fn defaults(v: config.Voice) Settings {
        return .{ .instructions = v.instructions, .voice = v.voice, .model = v.chat_model, .speech_model = v.speech_model, .transcription_model = v.transcription_model, .partial_interval_ms = v.partial_interval_ms };
    }

    /// A deep copy in `a`.
    pub fn dupe(s: Settings, a: std.mem.Allocator) !Settings {
        var c = s;
        inline for (.{ "instructions", "voice", "model", "speech_model", "transcription_model", "language", "transcription_prompt", "tools_json", "tool_choice_json" }) |f| @field(c, f) = try a.dupe(u8, @field(s, f));
        return c;
    }
};

pub const Invalid = error{ Invalid, OutOfMemory };

/// Applies a `session` object to `s` (whose strings must already live in `a`); on refusal `why` names the field.
pub fn apply(a: std.mem.Allocator, s: *Settings, v: Value, b: backend.Backend, why: *backend.Why) Invalid!void {
    if (v != .object) return bad(why, "session must be an object", .{});
    const o = v.object;
    if (o.get("modalities")) |m| {
        if (m != .array) return bad(why, "session.modalities must be a list such as [\"audio\", \"text\"]", .{});
        var audio = false;
        for (m.array.items) |x| {
            const t = str(x) orelse return bad(why, "session.modalities must be strings", .{});
            if (std.mem.eql(u8, t, "audio")) audio = true else if (!std.mem.eql(u8, t, "text")) return bad(why, "session.modalities takes audio and text, not {s}", .{t});
        }
        s.audio = audio;
    }
    inline for (.{ "input_audio_format", "output_audio_format" }) |f| if (o.get(f)) |x| {
        const t = str(x) orelse return bad(why, "session." ++ f ++ " must be a string", .{});
        if (!std.mem.eql(u8, t, "pcm16")) return bad(why, "session." ++ f ++ " {s} is not supported: use pcm16 (24 kHz mono)", .{t});
    };
    if (o.get("instructions")) |x| s.instructions = try a.dupe(u8, str(x) orelse return bad(why, "session.instructions must be a string", .{}));
    if (o.get("voice")) |x| s.voice = try a.dupe(u8, str(x) orelse return bad(why, "session.voice must be a string", .{}));
    if (o.get("model")) |x| s.model = try model(a, b, .chat, x, "model", why);
    if (o.get("speech_model")) |x| s.speech_model = try model(a, b, .speech, x, "speech_model", why);
    if (o.get("input_audio_transcription")) |x| switch (x) {
        .null => {},
        .object => |t| {
            if (t.get("model")) |m| s.transcription_model = try model(a, b, .transcription, m, "input_audio_transcription.model", why);
            if (t.get("language")) |l| s.language = try a.dupe(u8, str(l) orelse return bad(why, "input_audio_transcription.language must be a string", .{}));
            if (t.get("prompt")) |p| s.transcription_prompt = try a.dupe(u8, str(p) orelse return bad(why, "input_audio_transcription.prompt must be a string", .{}));
        },
        else => return bad(why, "session.input_audio_transcription must be an object or null", .{}),
    };
    if (o.get("turn_detection")) |x| try turnDetection(s, x, why);
    if (o.get("localrouter")) |x| try extension(s, x, why);
    if (o.get("tools")) |x| s.tools_json = try tools(a, x, why);
    if (o.get("tool_choice")) |x| {
        const ok = switch (x) {
            .string => |t| std.mem.eql(u8, t, "auto") or std.mem.eql(u8, t, "none") or std.mem.eql(u8, t, "required"),
            .object => |t| str(t.get("name") orelse .null) != null,
            else => false,
        };
        if (!ok) return bad(why, "session.tool_choice must be auto, none, required or {{\"type\":\"function\",\"name\":…}}", .{});
        s.tool_choice_json = try std.json.Stringify.valueAlloc(a, x, .{});
    }
    if (o.get("localrouter") != null or o.get("tools") != null or o.get("tool_choice") != null) try checkTools(a, s, why);
    if (o.get("temperature")) |x| s.temperature = @as(f32, @floatCast(num(x) orelse return bad(why, "session.temperature must be a number", .{})));
    if (o.get("max_response_output_tokens")) |x| s.max_tokens = switch (x) {
        .integer => |n| if (n >= 1 and n <= 1 << 20) @as(u32, @intCast(n)) else return bad(why, "session.max_response_output_tokens must be 1 to 1048576 or \"inf\"", .{}),
        .string => |t| if (std.mem.eql(u8, t, "inf")) null else return bad(why, "session.max_response_output_tokens must be a number or \"inf\"", .{}),
        else => return bad(why, "session.max_response_output_tokens must be a number or \"inf\"", .{}),
    };
    if (o.get("partial_interval_ms")) |x| s.partial_interval_ms = switch (x) {
        .integer => |n| if (n >= 0 and n <= 60_000) @intCast(n) else return bad(why, "session.partial_interval_ms must be 0 to 60000", .{}),
        else => return bad(why, "session.partial_interval_ms must be a number", .{}),
    };
}

fn turnDetection(s: *Settings, x: Value, why: *backend.Why) Invalid!void {
    if (x == .null) {
        s.vad = null;
        return;
    }
    if (x != .object) return bad(why, "session.turn_detection must be an object or null", .{});
    const t = x.object;
    if (t.get("type")) |ty| {
        const name = str(ty) orelse "";
        if (!std.mem.eql(u8, name, "server_vad")) return bad(why, "turn_detection.type {s} is not supported: use server_vad, or null for push to talk", .{name});
    }
    var c = s.vad orelse vad.Config{};
    if (t.get("threshold")) |v| {
        const f = num(v) orelse return bad(why, "turn_detection.threshold must be a number", .{});
        if (!(f >= 0 and f <= 1)) return bad(why, "turn_detection.threshold must be 0 to 1", .{});
        c.threshold = @floatCast(f);
    }
    inline for (.{ "prefix_padding_ms", "silence_duration_ms" }) |f| if (t.get(f)) |v| {
        const n = if (v == .integer) v.integer else return bad(why, "turn_detection." ++ f ++ " must be a whole number", .{});
        if (n < 0 or n > 10_000) return bad(why, "turn_detection." ++ f ++ " must be 0 to 10000", .{});
        @field(c, f) = @intCast(n);
    };
    if (c.silence_duration_ms < 100) return bad(why, "turn_detection.silence_duration_ms must be at least 100", .{});
    inline for (.{ "create_response", "interrupt_response" }) |f| if (t.get(f)) |v| {
        @field(s, f) = if (v == .bool) v.bool else return bad(why, "turn_detection." ++ f ++ " must be true or false", .{});
    };
    s.vad = c;
}

/// The `localrouter` extension object: `{"tool_execution": "server" | "client"}`. Anything else is refused, so a
/// client asking for a mode this server lacks finds out at once.
fn extension(s: *Settings, x: Value, why: *backend.Why) Invalid!void {
    if (x != .object) return bad(why, "session.localrouter must be an object", .{});
    var it = x.object.iterator();
    while (it.next()) |e| {
        if (!std.mem.eql(u8, e.key_ptr.*, "tool_execution")) return bad(why, "session.localrouter.{s} is not supported: only tool_execution is", .{e.key_ptr.*});
        const t = str(e.value_ptr.*) orelse return bad(why, "session.localrouter.tool_execution must be \"server\" or \"client\"", .{});
        s.tool_execution = std.meta.stringToEnum(ToolExecution, t) orelse return bad(why, "session.localrouter.tool_execution {s} is not supported: use server or client", .{t});
    }
}

/// The Realtime API's flat tools (`{type, name, description, parameters}`), checked, as JSON text.
fn tools(a: std.mem.Allocator, x: Value, why: *backend.Why) Invalid![]const u8 {
    if (x != .array) return bad(why, "session.tools must be a list", .{});
    for (x.array.items) |t| {
        if (t != .object) return bad(why, "each of session.tools must be an object", .{});
        const ty = str(t.object.get("type") orelse .null) orelse "function";
        if (!std.mem.eql(u8, ty, "function")) return bad(why, "session.tools: only function tools are supported, not {s}", .{ty});
        _ = str(t.object.get("name") orelse .null) orelse return bad(why, "each of session.tools needs a name", .{});
    }
    return std.json.Stringify.valueAlloc(a, x, .{});
}

/// The tools against the execution mode, once the whole update is applied. Server: the built-in names are this
/// server's. Client: any name, but each once and in the chat API's form, and a named tool_choice must be one of them.
fn checkTools(a: std.mem.Allocator, s: *const Settings, why: *backend.Why) Invalid!void {
    const v = std.json.parseFromSliceLeaky(Value, a, s.tools_json, .{}) catch return bad(why, "session.tools must be a list", .{});
    const list = if (v == .array) v.array.items else &.{};
    for (list, 0..) |t, i| {
        const name = str(t.object.get("name") orelse .null) orelse "";
        switch (s.tool_execution) {
            .server => if (isBuiltin(name)) return bad(why, "session.tools: {s} is a built-in tool of this server", .{name}),
            .client => {
                if (!validName(name)) return bad(why, "session.tools: the name {s} must be 1 to 64 letters, digits, _ or -", .{name});
                for (list[0..i]) |u| if (std.mem.eql(u8, str(u.object.get("name") orelse .null) orelse "", name))
                    return bad(why, "session.tools: {s} is listed twice", .{name});
            },
        }
    }
    if (s.tool_execution == .server) return;
    const choice = std.json.parseFromSliceLeaky(Value, a, s.tool_choice_json, .{}) catch return;
    if (choice != .object) return;
    const want = str(choice.object.get("name") orelse .null) orelse return;
    for (list) |t| if (std.mem.eql(u8, str(t.object.get("name") orelse .null) orelse "", want)) return;
    return bad(why, "session.tool_choice names {s}, which is not one of session.tools", .{want});
}

fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return false;
    return true;
}

pub fn isBuiltin(name: []const u8) bool {
    for ([_][]const u8{ "generate_image", "generate_video", "list_models", "reserve_for", "unload_model", "load_model", "computer_control", "computer_use" }) |n| if (std.mem.eql(u8, name, n)) return true;
    return false;
}

/// The client's tools in the chat API's nested shape, parsed into `a`.
pub fn chatTools(a: std.mem.Allocator, tools_json: []const u8) ![]chat.Tool {
    const v = std.json.parseFromSliceLeaky(Value, a, tools_json, .{}) catch return &.{};
    if (v != .array) return &.{};
    const out = try a.alloc(chat.Tool, v.array.items.len);
    for (out, v.array.items) |*d, t| d.* = .{ .function = .{
        .name = str(t.object.get("name") orelse .null) orelse "",
        .description = str(t.object.get("description") orelse .null) orelse "",
        .parameters = t.object.get("parameters") orelse .{ .object = .empty },
    } };
    return out;
}

/// The chat API's tool_choice for the Realtime one (`{"type":"function","name":x}` nests the name).
pub fn chatToolChoice(a: std.mem.Allocator, json: []const u8) !?Value {
    const v = std.json.parseFromSliceLeaky(Value, a, json, .{}) catch return null;
    if (v != .object) return v;
    var f: std.json.ObjectMap = .empty;
    try f.put(a, "name", v.object.get("name") orelse .null);
    var o: std.json.ObjectMap = .empty;
    try o.put(a, "type", .{ .string = "function" });
    try o.put(a, "function", .{ .object = f });
    return .{ .object = o };
}

fn model(a: std.mem.Allocator, b: backend.Backend, role: backend.Role, x: Value, comptime field: []const u8, why: *backend.Why) Invalid![]const u8 {
    const name = str(x) orelse return bad(why, "session." ++ field ++ " must be a string", .{});
    if (name.len > 0 and !b.check(role, name, why)) return error.Invalid;
    return a.dupe(u8, name);
}

fn bad(why: *backend.Why, comptime fmt: []const u8, args: anytype) Invalid {
    why.set(fmt, args);
    return error.Invalid;
}

pub fn str(v: Value) ?[]const u8 {
    return if (v == .string) v.string else null;
}

pub fn num(v: Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

/// The session object of session.created and session.updated.
pub fn sessionObject(a: std.mem.Allocator, id: []const u8, s: *const Settings) !Value {
    const Td = struct { type: []const u8 = "server_vad", threshold: f32, prefix_padding_ms: u32, silence_duration_ms: u32, create_response: bool, interrupt_response: bool };
    const td: ?Td = if (s.vad) |c| .{ .threshold = c.threshold, .prefix_padding_ms = c.prefix_padding_ms, .silence_duration_ms = c.silence_duration_ms, .create_response = s.create_response, .interrupt_response = s.interrupt_response } else null;
    const modalities: []const []const u8 = if (s.audio) &.{ "audio", "text" } else &.{"text"};
    const obj = .{
        .id = id,
        .object = "realtime.session",
        .model = s.model,
        .modalities = modalities,
        .instructions = s.instructions,
        .voice = s.voice,
        .speech_model = s.speech_model,
        .input_audio_format = "pcm16",
        .output_audio_format = "pcm16",
        .input_audio_transcription = .{ .model = s.transcription_model, .language = s.language, .prompt = s.transcription_prompt },
        .turn_detection = td,
        .tools = std.json.parseFromSliceLeaky(Value, a, s.tools_json, .{}) catch Value{ .array = .init(a) },
        .tool_choice = std.json.parseFromSliceLeaky(Value, a, s.tool_choice_json, .{}) catch Value{ .string = "auto" },
        .temperature = s.temperature,
        .max_response_output_tokens = if (s.max_tokens) |n| Value{ .integer = n } else Value{ .string = "inf" },
        .partial_interval_ms = s.partial_interval_ms,
        .localrouter = .{ .tool_execution = @tagName(s.tool_execution) },
    };
    const text = try std.json.Stringify.valueAlloc(a, obj, .{});
    return std.json.parseFromSliceLeaky(Value, a, text, .{});
}

/// Stamps `type` and `event_id` on each server event and hands it to the sink.
pub const Emitter = struct {
    out: backend.Out,
    next: std.atomic.Value(u64) = .init(1),

    /// Sends `{"type":typ,"event_id":…, …fields}`; a failure to serialize drops the event.
    pub fn send(e: *Emitter, a: std.mem.Allocator, typ: []const u8, fields: anytype) void {
        const text = e.format(a, typ, fields) catch return;
        e.out.send(e.out.ctx, text);
    }

    pub fn format(e: *Emitter, a: std.mem.Allocator, typ: []const u8, fields: anytype) ![]const u8 {
        const body = try std.json.Stringify.valueAlloc(a, fields, .{});
        const n = e.next.fetchAdd(1, .monotonic);
        const head = try std.json.Stringify.valueAlloc(a, .{ .type = typ, .event_id = try std.fmt.allocPrint(a, "event_{d}", .{n}) }, .{});
        if (body.len <= 2) return head; // {}
        return std.fmt.allocPrint(a, "{s},{s}", .{ head[0 .. head.len - 1], body[1..] });
    }

    /// The `error` event: OpenAI's `{type, code, message, param, event_id}`.
    pub fn fail(e: *Emitter, a: std.mem.Allocator, typ: []const u8, code: []const u8, message: []const u8, event_id: ?[]const u8) void {
        e.send(a, "error", .{ .@"error" = .{ .type = typ, .code = code, .message = message, .param = @as(?[]const u8, null), .event_id = event_id } });
    }
};

const Sent = struct {
    list: std.ArrayList([]const u8) = .empty,
    a: std.mem.Allocator,

    fn send(ctx: *anyopaque, text: []const u8) void {
        const s: *Sent = @ptrCast(@alignCast(ctx));
        s.list.append(s.a, s.a.dupe(u8, text) catch return) catch {};
    }
};

test "events carry type and event_id first; an empty body is just the stamp" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var sent: Sent = .{ .a = a };
    var e: Emitter = .{ .out = .{ .ctx = &sent, .send = Sent.send } };
    e.send(a, "input_audio_buffer.cleared", .{});
    e.send(a, "input_audio_buffer.speech_started", .{ .audio_start_ms = 120, .item_id = "item_1" });
    e.fail(a, "invalid_request_error", "unknown_event", "no such event", null);
    try std.testing.expectEqualStrings("{\"type\":\"input_audio_buffer.cleared\",\"event_id\":\"event_1\"}", sent.list.items[0]);
    try std.testing.expectEqualStrings("{\"type\":\"input_audio_buffer.speech_started\",\"event_id\":\"event_2\",\"audio_start_ms\":120,\"item_id\":\"item_1\"}", sent.list.items[1]);
    try std.testing.expect(std.mem.startsWith(u8, sent.list.items[2], "{\"type\":\"error\",\"event_id\":\"event_3\",\"error\":{\"type\":\"invalid_request_error\",\"code\":\"unknown_event\""));
}

const NoModels = struct {
    fn check(_: *anyopaque, role: backend.Role, name: []const u8, why: *backend.Why) bool {
        if (role == .chat or std.mem.eql(u8, name, "kitten")) return true;
        why.set("unknown model '{s}'", .{name});
        return false;
    }
};

test "session.update: fields apply one by one, refusals name the field" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var vt: backend.Backend.VTable = undefined;
    vt.check = NoModels.check;
    var dummy: u8 = 0;
    const b: backend.Backend = .{ .ptr = &dummy, .vtable = &vt };
    var s: Settings = try Settings.defaults(.{}).dupe(a);
    var why: backend.Why = .{};
    const ok = try std.json.parseFromSliceLeaky(Value, a,
        \\{"modalities": ["text"], "voice": "Rosie", "speech_model": "kitten", "model": "qwen",
        \\ "turn_detection": {"type": "server_vad", "threshold": 0.7, "silence_duration_ms": 350},
        \\ "tools": [{"type": "function", "name": "lookup", "description": "d", "parameters": {"type": "object"}}],
        \\ "tool_choice": {"type": "function", "name": "lookup"}, "max_response_output_tokens": "inf", "temperature": 0.6}
    , .{});
    try apply(a, &s, ok, b, &why);
    try std.testing.expect(!s.audio);
    try std.testing.expectEqualStrings("Rosie", s.voice);
    try std.testing.expectEqual(@as(u32, 350), s.vad.?.silence_duration_ms);
    try std.testing.expectEqual(@as(?u32, null), s.max_tokens);
    const ct = try chatTools(a, s.tools_json);
    try std.testing.expectEqualStrings("lookup", ct[0].function.name);
    const choice = (try chatToolChoice(a, s.tool_choice_json)).?;
    try std.testing.expectEqualStrings("lookup", choice.object.get("function").?.object.get("name").?.string);
    const cases = [_]struct { json: []const u8, msg: []const u8 }{
        .{ .json = "{\"input_audio_format\": \"g711_ulaw\"}", .msg = "session.input_audio_format g711_ulaw is not supported: use pcm16 (24 kHz mono)" },
        .{ .json = "{\"speech_model\": \"nope\"}", .msg = "unknown model 'nope'" },
        .{ .json = "{\"turn_detection\": {\"type\": \"semantic_vad\"}}", .msg = "turn_detection.type semantic_vad is not supported: use server_vad, or null for push to talk" },
        .{ .json = "{\"turn_detection\": {\"threshold\": 2}}", .msg = "turn_detection.threshold must be 0 to 1" },
        .{ .json = "{\"tools\": [{\"type\": \"function\", \"name\": \"generate_image\"}]}", .msg = "session.tools: generate_image is a built-in tool of this server" },
        .{ .json = "{\"modalities\": [\"video\"]}", .msg = "session.modalities takes audio and text, not video" },
    };
    for (cases) |c| {
        try std.testing.expectError(error.Invalid, apply(a, &s, try std.json.parseFromSliceLeaky(Value, a, c.json, .{}), b, &why));
        try std.testing.expectEqualStrings(c.msg, why.message);
    }
    try apply(a, &s, try std.json.parseFromSliceLeaky(Value, a, "{\"turn_detection\": null}", .{}), b, &why);
    try std.testing.expect(s.vad == null);
    const obj = try sessionObject(a, "sess_1", &s);
    try std.testing.expect(obj.object.get("turn_detection").? == .null);
    try std.testing.expectEqualStrings("inf", obj.object.get("max_response_output_tokens").?.string);
    try std.testing.expectEqualStrings("server", obj.object.get("localrouter").?.object.get("tool_execution").?.string);
}

test "client tool execution: built-in names are the client's, each name once, and the mode fails closed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var vt: backend.Backend.VTable = undefined;
    vt.check = NoModels.check;
    var dummy: u8 = 0;
    const b: backend.Backend = .{ .ptr = &dummy, .vtable = &vt };
    var s: Settings = try Settings.defaults(.{}).dupe(a);
    var why: backend.Why = .{};
    const parse = struct {
        fn f(al: std.mem.Allocator, json: []const u8) !Value {
            return std.json.parseFromSliceLeaky(Value, al, json, .{});
        }
    }.f;
    // One update: the mode and a tool named like a built-in, in either order of the fields.
    try apply(a, &s, try parse(a, "{\"tools\": [{\"type\": \"function\", \"name\": \"generate_image\"}, {\"name\": \"get-time_2\"}], \"localrouter\": {\"tool_execution\": \"client\"}}"), b, &why);
    try std.testing.expectEqual(ToolExecution.client, s.tool_execution);
    try std.testing.expectEqualStrings("client", (try sessionObject(a, "sess_1", &s)).object.get("localrouter").?.object.get("tool_execution").?.string);
    const cases = [_]struct { json: []const u8, msg: []const u8 }{
        .{ .json = "{\"tools\": [{\"name\": \"lookup\"}, {\"name\": \"lookup\"}]}", .msg = "session.tools: lookup is listed twice" },
        .{ .json = "{\"tools\": [{\"name\": \"look up\"}]}", .msg = "session.tools: the name look up must be 1 to 64 letters, digits, _ or -" },
        .{ .json = "{\"tool_choice\": {\"type\": \"function\", \"name\": \"lookup\"}}", .msg = "session.tool_choice names lookup, which is not one of session.tools" },
        .{ .json = "{\"localrouter\": {\"tool_execution\": \"server\"}}", .msg = "session.tools: generate_image is a built-in tool of this server" }, // back to server with a built-in name listed
        .{ .json = "{\"localrouter\": {\"tool_execution\": \"both\"}}", .msg = "session.localrouter.tool_execution both is not supported: use server or client" },
        .{ .json = "{\"localrouter\": {\"profile\": \"v2\"}}", .msg = "session.localrouter.profile is not supported: only tool_execution is" },
    };
    for (cases) |c| {
        var t = s;
        try std.testing.expectError(error.Invalid, apply(a, &t, try parse(a, c.json), b, &why));
        try std.testing.expectEqualStrings(c.msg, why.message);
    }
    try apply(a, &s, try parse(a, "{\"tools\": [], \"localrouter\": {\"tool_execution\": \"server\"}}"), b, &why);
    try std.testing.expectEqual(ToolExecution.server, s.tool_execution);
}
