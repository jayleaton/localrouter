//! A chat request's thinking budget. A reasoning model spends its tokens on reasoning first, so a small limit can end
//! the reply before it says anything: every token in `reasoning_content`, `content` empty, `finish_reason: "length"`.
//! For Chat Completions, the Responses API and the MCP `chat` tool alike (each starts in `chat.startChat`):
//! - A request whose limit (`max_completion_tokens`, else `max_tokens`; Responses' `max_output_tokens`) is below the
//!   tool's `min_thinking_tokens`, and that does not choose reasoning itself (`reasoning_effort`, `reasoning`,
//!   `thinking`, `chat_template_kwargs.enable_thinking`), goes to the model with thinking off (the tool's
//!   `thinking_off` members, merged into the request) and says so.
//! - A reply cut at its limit with no text and no tool call says that its reasoning used the whole budget.
//! Both are said in `warnings: [{code, message}]`: on the completion (streamed: one last chunk with `choices: []`
//! before `[DONE]`) and on the response object (Responses). `finish_reason` stays the model's own.

const std = @import("std");
const ToolConfig = @import("../config.zig").ToolConfig;

const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const Allocator = std.mem.Allocator;

/// The members that switch thinking off for a tool without `thinking_off`: the chat template's own switch, which
/// Qwen-family servers take (TensorFold, vLLM, SGLang, llama.cpp).
pub const default_off = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}";

/// The API a request came in on, for the field names its warnings use.
pub const Api = enum { chat, responses };

/// What `apply` found and did.
pub const Plan = struct {
    api: Api = .chat,
    limit: ?i64 = null, // the reply's limit as asked; null: the server's default
    field: []const u8 = "max_tokens", // the chat field the limit came in
    floor: u32 = 0, // the tool's min_thinking_tokens
    off: bool = false, // thinking was switched off here

    fn limitName(p: Plan) []const u8 {
        return if (p.api == .responses) "max_output_tokens" else p.field;
    }
};

/// Whether the request says itself whether to reason.
fn chooses(o: ObjectMap) bool {
    for ([_][]const u8{ "reasoning_effort", "reasoning", "thinking" }) |k| if (o.get(k)) |v| if (v != .null) return true;
    if (o.get("chat_template_kwargs")) |k| if (k == .object) if (k.object.get("enable_thinking")) |v| return v != .null;
    return false;
}

/// Switches thinking off in `body` (a chat request) when its limit is below `cfg.min_thinking_tokens` and it does not
/// choose reasoning itself.
pub fn apply(a: Allocator, body: *ObjectMap, cfg: *const ToolConfig, api: Api) !Plan {
    var p: Plan = .{ .api = api, .floor = cfg.min_thinking_tokens };
    for ([_][]const u8{ "max_completion_tokens", "max_tokens" }) |k| if (body.get(k)) |v| {
        p.limit = switch (v) { // an MCP client's number may come as a float
            .integer => |n| n,
            .float => |f| if (f >= 0 and f < 1e15 and @floor(f) == f) @as(i64, @intFromFloat(f)) else continue,
            else => continue,
        };
        p.field = k;
        break;
    };
    const limit = p.limit orelse return p;
    if (limit >= p.floor or chooses(body.*)) return p;
    const off = cfg.thinking_off orelse try std.json.parseFromSliceLeaky(Value, a, default_off, .{});
    if (off != .object or off.object.count() == 0) return p;
    var it = off.object.iterator();
    while (it.next()) |e| try merge(a, body, e.key_ptr.*, e.value_ptr.*);
    p.off = true;
    return p;
}

/// Puts `v` at `k`; into an object already there it merges, so a request's own `chat_template_kwargs` keep their other
/// members.
fn merge(a: Allocator, into: *ObjectMap, k: []const u8, v: Value) !void {
    if (v == .object) if (into.getPtr(k)) |cur| if (cur.* == .object) {
        var it = v.object.iterator();
        while (it.next()) |e| try merge(a, &cur.object, e.key_ptr.*, e.value_ptr.*);
        return;
    };
    try into.put(a, k, v);
}

fn str(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

/// What a reply said, as far as the warnings need: why it ended, whether it said or called anything, its reasoning
/// tokens; and a stream's id, model and created, for the warnings chunk.
pub const Seen = struct {
    finish: []const u8 = "",
    text: bool = false,
    calls: bool = false,
    reasoning_tokens: ?i64 = null,
    id: []const u8 = "",
    model: []const u8 = "",
    created: i64 = 0,

    /// One parsed completion (`chat.completion`) or stream chunk (`chat.completion.chunk`). Kept strings are copied
    /// into `a`.
    pub fn take(s: *Seen, a: Allocator, v: Value) Allocator.Error!void {
        if (v != .object) return;
        const o = v.object;
        if (s.id.len == 0) if (str(o.get("id"))) |x| {
            s.id = try a.dupe(u8, x);
        };
        if (s.model.len == 0) if (str(o.get("model"))) |x| {
            s.model = try a.dupe(u8, x);
        };
        if (o.get("created")) |c| if (c == .integer) {
            s.created = c.integer;
        };
        if (o.get("usage")) |u| if (u == .object) if (u.object.get("completion_tokens_details")) |d| if (d == .object) if (d.object.get("reasoning_tokens")) |r| if (r == .integer) {
            s.reasoning_tokens = r.integer;
        };
        const choices = o.get("choices") orelse return;
        if (choices != .array or choices.array.items.len == 0 or choices.array.items[0] != .object) return;
        const c = choices.array.items[0].object;
        if (str(c.get("finish_reason"))) |f| s.finish = try a.dupe(u8, f);
        const said = c.get("message") orelse c.get("delta") orelse return;
        if (said != .object) return;
        if (str(said.object.get("content"))) |t| if (t.len > 0) {
            s.text = true;
        };
        if (said.object.get("tool_calls")) |t| if (t == .array and t.array.items.len > 0) {
            s.calls = true;
        };
    }

    /// One line of a streamed completion (`data: {...}`; anything else is skipped).
    pub fn line(s: *Seen, a: Allocator, raw: []const u8) Allocator.Error!void {
        const l = std.mem.trimEnd(u8, raw, "\r");
        if (!std.mem.startsWith(u8, l, "data:")) return;
        const data = std.mem.trim(u8, l["data:".len..], " ");
        if (std.mem.eql(u8, data, "[DONE]")) return;
        const v = std.json.parseFromSliceLeaky(Value, a, data, .{}) catch return;
        return s.take(a, v);
    }
};

pub const Warning = struct { code: []const u8, message: []const u8 };

/// The warnings for a request planned as `p` whose reply was `s`, in the order they happened.
pub fn warnings(a: Allocator, p: Plan, s: Seen) Allocator.Error![]const Warning {
    var list: std.ArrayList(Warning) = .empty;
    const name = p.limitName();
    if (p.off) try list.append(a, .{ .code = "thinking_off_for_small_budget", .message = try std.fmt.allocPrint(a, "{s} {d} is below this model's min_thinking_tokens ({d}), so it answered without thinking; {s} to choose", .{
        name, p.limit orelse 0, p.floor, if (p.api == .responses) "send reasoning.effort" else "send reasoning_effort (or chat_template_kwargs.enable_thinking)",
    }) });
    if (std.mem.eql(u8, s.finish, "length") and !s.text and !s.calls) {
        const limit = if (p.limit) |l| try std.fmt.allocPrint(a, "{s} {d}", .{ name, l }) else "the server's default limit";
        const spent = if (s.reasoning_tokens) |r| try std.fmt.allocPrint(a, " ({d} reasoning tokens)", .{r}) else "";
        try list.append(a, .{ .code = "reasoning_used_budget", .message = try std.fmt.allocPrint(a, "the reply reached {s} while still reasoning{s}, so it has no content: raise {s}, or switch thinking off ({s})", .{
            limit, spent, name, if (p.api == .responses) "reasoning.effort: none" else "reasoning_effort: none",
        }) });
    }
    return list.items;
}

/// `warnings` as a JSON array.
pub fn value(a: Allocator, w: []const Warning) Allocator.Error!Value {
    var arr = std.json.Array.init(a);
    for (w) |x| {
        var m: ObjectMap = .empty;
        try m.put(a, "code", .{ .string = x.code });
        try m.put(a, "message", .{ .string = x.message });
        try arr.append(.{ .object = m });
    }
    return .{ .array = arr };
}

/// The warnings for a whole completion (the model server's JSON); none when it is not JSON.
pub fn ofCompletion(a: Allocator, completion: []const u8, p: Plan) Allocator.Error![]const Warning {
    const v = std.json.parseFromSliceLeaky(Value, a, completion, .{}) catch return &.{};
    var s: Seen = .{};
    try s.take(a, v);
    return warnings(a, p, s);
}

/// A whole completion with its `warnings` added; as it came when there are none, or it is not a JSON object.
pub fn annotate(a: Allocator, completion: []const u8, p: Plan) Allocator.Error![]const u8 {
    var v = std.json.parseFromSliceLeaky(Value, a, completion, .{}) catch return completion;
    if (v != .object) return completion;
    var s: Seen = .{};
    try s.take(a, v);
    const w = try warnings(a, p, s);
    if (w.len == 0) return completion;
    try v.object.put(a, "warnings", try value(a, w));
    return std.json.Stringify.valueAlloc(a, v, .{});
}

/// The last chunk of a stream with warnings (`choices: []`), as an SSE event.
pub fn chunk(a: Allocator, s: Seen, w: []const Warning) Allocator.Error![]const u8 {
    const text = try std.json.Stringify.valueAlloc(a, .{ .id = s.id, .object = "chat.completion.chunk", .created = s.created, .model = s.model, .choices = @as([]const Value, &.{}), .warnings = w }, .{});
    return std.fmt.allocPrint(a, "data: {s}\n\n", .{text});
}

// ---------------------------------------------------------------- tests

const testing = std.testing;

fn plan(a: Allocator, body: []const u8, cfg: ToolConfig) !struct { p: Plan, body: []const u8 } {
    var v = try std.json.parseFromSliceLeaky(Value, a, body, .{});
    const p = try apply(a, &v.object, &cfg, .chat);
    return .{ .p = p, .body = try std.json.Stringify.valueAlloc(a, v, .{}) };
}

test "a small limit switches thinking off unless the request chooses; the tool's floor and members" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: ToolConfig = .{ .id = "chat", .kind = .chat };
    const off = try plan(a, "{\"max_tokens\":16,\"chat_template_kwargs\":{\"x\":1}}", cfg);
    try testing.expect(off.p.off);
    try testing.expectEqualStrings("{\"max_tokens\":16,\"chat_template_kwargs\":{\"x\":1,\"enable_thinking\":false}}", off.body);
    try testing.expect((try plan(a, "{\"max_completion_tokens\":1023,\"max_tokens\":5000}", cfg)).p.off);
    for ([_][]const u8{
        "{\"max_tokens\":1024}",                                                   "{}",
        "{\"max_tokens\":16,\"reasoning_effort\":\"low\"}",                        "{\"max_tokens\":16,\"reasoning\":{\"effort\":\"low\"}}",
        "{\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":true}}", "{\"max_tokens\":16,\"thinking\":{\"type\":\"enabled\"}}",
    }) |b| {
        const r = try plan(a, b, cfg);
        try testing.expect(!r.p.off);
        try testing.expectEqualStrings(b, r.body);
    }
    try testing.expect(!(try plan(a, "{\"max_tokens\":16}", .{ .id = "chat", .kind = .chat, .min_thinking_tokens = 0 })).p.off);
    const own = try std.json.parseFromSliceLeaky(Value, a, "{\"reasoning_effort\":\"none\"}", .{});
    const custom = try plan(a, "{\"max_tokens\":16}", .{ .id = "chat", .kind = .chat, .thinking_off = own });
    try testing.expectEqualStrings("{\"max_tokens\":16,\"reasoning_effort\":\"none\"}", custom.body);
}

test "warnings: thinking switched off, and a reply whose reasoning used the whole limit" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const spent =
        \\{"id":"c1","choices":[{"message":{"content":"","reasoning_content":"let me see"},"finish_reason":"length"}],"usage":{"completion_tokens":16,"completion_tokens_details":{"reasoning_tokens":16}}}
    ;
    const out = try annotate(a, spent, .{ .limit = 16 });
    const v = try std.json.parseFromSliceLeaky(Value, a, out, .{});
    const w = v.object.get("warnings").?.array.items;
    try testing.expectEqual(@as(usize, 1), w.len);
    try testing.expectEqualStrings("reasoning_used_budget", w[0].object.get("code").?.string);
    try testing.expectEqualStrings("the reply reached max_tokens 16 while still reasoning (16 reasoning tokens), so it has no content: raise max_tokens, or switch thinking off (reasoning_effort: none)", w[0].object.get("message").?.string);
    // Text, a tool call, or a reply that stopped by itself: nothing to say, and the bytes are the server's.
    for ([_][]const u8{
        "{\"choices\":[{\"message\":{\"content\":\"17\"},\"finish_reason\":\"length\"}]}",
        "{\"choices\":[{\"message\":{\"content\":null,\"tool_calls\":[{\"id\":\"x\"}]},\"finish_reason\":\"length\"}]}",
        "{\"choices\":[{\"message\":{\"content\":\"\"},\"finish_reason\":\"stop\"}]}",
        "not json",
    }) |c| try testing.expectEqualStrings(c, try annotate(a, c, .{ .limit = 16 }));
    const both = try warnings(a, .{ .api = .responses, .limit = 16, .floor = 1024, .off = true }, .{ .finish = "length" });
    try testing.expectEqualStrings("thinking_off_for_small_budget", both[0].code);
    try testing.expectEqualStrings("max_output_tokens 16 is below this model's min_thinking_tokens (1024), so it answered without thinking; send reasoning.effort to choose", both[0].message);
    try testing.expect(std.mem.indexOf(u8, both[1].message, "reasoning.effort: none") != null);
}

test "a stream's chunks: what it said, and the warnings chunk" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s: Seen = .{};
    for ([_][]const u8{
        "data: {\"id\":\"c9\",\"created\":3,\"model\":\"m\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}",
        ": a comment",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\r",
        "data: [DONE]",
    }) |l| try s.line(a, l);
    try testing.expectEqualStrings("length", s.finish);
    try testing.expect(!s.text and !s.calls);
    const w = try warnings(a, .{ .limit = 4 }, s);
    try testing.expectEqualStrings(
        \\data: {"id":"c9","object":"chat.completion.chunk","created":3,"model":"m","choices":[],"warnings":[{"code":"reasoning_used_budget","message":"the reply reached max_tokens 4 while still reasoning, so it has no content: raise max_tokens, or switch thinking off (reasoning_effort: none)"}]}
        \\
        \\
    , try chunk(a, s, w));
    var t: Seen = .{};
    try t.line(a, "data: {\"choices\":[{\"delta\":{\"content\":\"3\"},\"finish_reason\":\"length\"}]}");
    try testing.expectEqual(@as(usize, 0), (try warnings(a, .{ .limit = 4 }, t)).len);
}
