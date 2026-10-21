//! Integration tests of the chat kind on real daemons and workers: the compiled-in `openai_server` engine starts a
//! fake TensorFold server (tests/fixtures/chat_server.py) and the daemon proxies /v1/chat/completions to it, whole and
//! streamed. Routing and defaults, passthrough, tool calls, error mapping, the qualification gate, eviction priority,
//! cancellation that keeps the model warm, the GPU lock, and the MCP `chat` tool.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const reserve = harness.reserve;
const mcpCall = harness.mcpCall;
const mcpText = harness.mcpText;
const sleepMs = harness.sleepMs;
const Value = std.json.Value;
const Obj = std.json.ObjectMap;

/// A chat tool (40 MiB) on the fixture server: `extra` config keys and `args` server arguments, each led by a comma.
fn chatTool(a: std.mem.Allocator, id: []const u8, extra: []const u8, args: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"id\": \"{s}\", \"kind\": \"chat\", \"engine\": \"openai_server\", \"resident_bytes\": 41943040{s}, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"{s}]}}}}", .{ id, extra, opts.chat_server, args });
}

/// A daemon with `top` config keys (led by a comma) and `tools` (comma-joined tool objects).
fn start(a: std.mem.Allocator, name: []const u8, top: []const u8, tools: []const u8) !Daemon {
    const cfg = try std.fmt.allocPrint(a, "{{" ++ reserve ++ "{s}, \"tools\": [{s}]}}", .{ top, tools });
    return Daemon.start(testing.allocator, testing.io, name, cfg);
}

fn post(d: *Daemon, a: std.mem.Allocator, body: []const u8) !Daemon.Resp {
    return d.call(a, .POST, "/v1/chat/completions", body);
}

/// A chat request of one user message `prompt`, plus `extra` body keys (led by a comma).
fn ask(d: *Daemon, a: std.mem.Allocator, prompt: []const u8, extra: []const u8) !Daemon.Resp {
    return post(d, a, try std.fmt.allocPrint(a, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]{s}}}", .{ prompt, extra }));
}

fn expectStatus(r: Daemon.Resp, status: std.http.Status, needles: []const []const u8) !void {
    if (r.status != status) std.debug.print("expected {d}, got {d}: {s}\n", .{ @backingInt(status), @backingInt(r.status), r.body });
    try testing.expectEqual(status, r.status);
    for (needles) |n| if (std.mem.indexOf(u8, r.body, n) == null) {
        std.debug.print("missing '{s}' in {s}\n", .{ n, r.body });
        return error.TestExpectedEqual;
    };
}

/// A whole reply (200) as a JSON object.
fn whole(a: std.mem.Allocator, r: Daemon.Resp) !Obj {
    try expectStatus(r, .ok, &.{});
    return (try std.json.parseFromSliceLeaky(Value, a, r.body, .{})).object;
}

fn first(o: Obj) Obj {
    return o.get("choices").?.array.items[0].object;
}

fn message(o: Obj) Obj {
    return first(o).get("message").?.object;
}

/// A string member, or null when it is absent or not a string.
fn str(o: Obj, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

const Stream = struct { content: []const u8 = "", reasoning: []const u8 = "", arguments: []const u8 = "", tool: []const u8 = "", finish: []const u8 = "", usage: bool = false };

/// The joined deltas of a 200 SSE body; every line is `data: ...` and the last is `data: [DONE]`.
fn sse(a: std.mem.Allocator, r: Daemon.Resp) !Stream {
    try expectStatus(r, .ok, &.{});
    if (!std.mem.endsWith(u8, r.body, "data: [DONE]\n\n")) {
        std.debug.print("not ended by [DONE]: {s}\n", .{r.body});
        return error.TestExpectedEqual;
    }
    var content: std.ArrayList(u8) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    var args: std.ArrayList(u8) = .empty;
    var out: Stream = .{};
    var lines = std.mem.tokenizeScalar(u8, r.body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) {
            std.debug.print("not an SSE data line: '{s}' in {s}\n", .{ line, r.body });
            return error.TestExpectedEqual;
        }
        if (std.mem.eql(u8, line[6..], "[DONE]")) break;
        const ev = (try std.json.parseFromSliceLeaky(Value, a, line[6..], .{})).object;
        if (ev.get("usage") != null) out.usage = true;
        const choices = (ev.get("choices") orelse continue).array.items;
        if (choices.len == 0) continue;
        if (str(choices[0].object, "finish_reason")) |f| out.finish = f;
        const delta = choices[0].object.get("delta").?.object;
        if (str(delta, "content")) |s| try content.appendSlice(a, s);
        if (str(delta, "reasoning_content")) |s| try reasoning.appendSlice(a, s);
        if (delta.get("tool_calls")) |tc| {
            const f = tc.array.items[0].object.get("function").?.object;
            if (str(f, "name")) |n| out.tool = n;
            if (str(f, "arguments")) |s| try args.appendSlice(a, s);
        }
    }
    out.content = content.items;
    out.reasoning = reasoning.items;
    out.arguments = args.items;
    return out;
}

/// The response headers of a chat POST, lowercased names, one "name: value" a line (the body is not read).
fn headers(d: *Daemon, a: std.mem.Allocator, body: []const u8) ![]const u8 {
    const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/chat/completions", .{d.port});
    var req = try d.client.request(.POST, try std.Uri.parse(url), .{ .keep_alive = false, .headers = .{ .content_type = .{ .override = "application/json" } } });
    defer req.deinit();
    req.transfer_encoding = .{ .content_length = body.len };
    try req.sendBodyComplete(@constCast(body));
    var redirect: [0]u8 = undefined;
    const res = try req.receiveHead(&redirect);
    var out: Io.Writer.Allocating = .init(a);
    var it = res.head.iterateHeaders();
    while (it.next()) |hd| try out.writer.print("{s}: {s}\n", .{ try std.ascii.allocLowerString(a, hd.name), hd.value });
    return out.written();
}

fn waitState(d: *Daemon, a: std.mem.Allocator, id: []const u8, want: []const u8, tries: usize) !void {
    for (0..tries) |_| {
        if (std.mem.eql(u8, try d.toolState(a, id), want)) return;
        sleepMs(d.io, 250);
    }
    std.debug.print("{s}: wanted {s}, is {s}\n", .{ id, want, try d.toolState(a, id) });
    return error.TestExpectedEqual;
}

/// The i-th text of an MCP result.
fn text(result: Value, i: usize) []const u8 {
    return result.object.get("content").?.array.items[i].object.get("text").?.string;
}

const weather_tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]";

test "chat models and routing: kind and capability, priority 100 and keep_loaded by default, refusals" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try std.fmt.allocPrint(a, "{s}, {s}, {{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\"}}", .{ try chatTool(a, "chat", "", ""), try chatTool(a, "chat2", ", \"keep_loaded\": false, \"priority\": 5", "") });
    var d = try start(a, "chat-models", "", tools);
    defer d.stop();

    const M = struct { data: []const struct { id: []const u8, kind: []const u8, capabilities: []const []const u8, default_for: []const []const u8 } };
    const models = try std.json.parseFromSliceLeaky(M, a, (try d.call(a, .GET, "/v1/models", null)).body, .{ .ignore_unknown_fields = true });
    try testing.expectEqualStrings("chat", models.data[0].kind);
    try testing.expectEqual(@as(usize, 1), models.data[0].capabilities.len);
    try testing.expectEqualStrings("chat_completion", models.data[0].capabilities[0]);
    try testing.expectEqual(@as(usize, 1), models.data[0].default_for.len);
    try testing.expectEqualStrings("chat_completion", models.data[0].default_for[0]);
    try testing.expectEqual(@as(usize, 0), models.data[1].default_for.len);

    // The config says neither for `chat`: a chat tool defaults to priority 100 and keep_loaded; explicit values win.
    const T = struct { data: []const struct { id: []const u8, priority: i32, keep_loaded: bool } };
    const listed = try std.json.parseFromSliceLeaky(T, a, (try d.call(a, .GET, "/v1/tools", null)).body, .{ .ignore_unknown_fields = true });
    try testing.expectEqual(@as(i32, 100), listed.data[0].priority);
    try testing.expect(listed.data[0].keep_loaded);
    try testing.expectEqual(@as(i32, 5), listed.data[1].priority);
    try testing.expect(!listed.data[1].keep_loaded);
    try testing.expectEqual(@as(i32, 0), listed.data[2].priority);
    try testing.expect(!listed.data[2].keep_loaded);

    // No model: the default. The fixture refuses any model but its --name, so this also shows the rewrite.
    try testing.expectEqualStrings("chat", str(try whole(a, try ask(&d, a, "hi", "")), "model").?);
    try testing.expectEqualStrings("chat2", str(try whole(a, try ask(&d, a, "hi", ",\"model\":\"chat2\"")), "model").?);
    try expectStatus(try ask(&d, a, "hi", ",\"model\":\"nope\""), .not_found, &.{"unknown model 'nope'; models that can do chat_completion: chat, chat2"});
    try expectStatus(try ask(&d, a, "hi", ",\"model\":\"tp\""), .bad_request, &.{"model 'tp' is an image model, not chat; models that can do chat_completion: chat, chat2"});
    try expectStatus(try ask(&d, a, "hi", ",\"model\":5"), .bad_request, &.{"model must be a string"});
    try expectStatus(try harness.image(&d, a, "chat", ""), .bad_request, &.{ "'chat' is a chat model, not image", "text_to_image: tp" });
}

test "chat whole replies: the body passes through, reasoning on by default and off on request, limits" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "chat-whole", "", try chatTool(a, "chat", "", ""));
    defer d.stop();

    const body = "{\"model\":\"chat\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":0,\"top_p\":0.9,\"seed\":5,\"max_tokens\":50," ++
        "\"stop\":[\"zzz\"],\"top_k\":3,\"x_extra\":\"y\",\"response_format\":{\"type\":\"text\"}}";
    const r = try post(&d, a, body);
    const o = try whole(a, r);
    try testing.expectEqualStrings("chat", str(o, "model").?);
    try testing.expectEqualStrings("echo: hi", str(message(o), "content").?);
    try testing.expectEqualStrings("thinking about hi", str(message(o), "reasoning_content").?);
    try testing.expectEqualStrings("stop", str(first(o), "finish_reason").?);
    // What the engine received: every field, known to OpenAI or not.
    const R = struct { received: struct { model: []const u8, temperature: f64, top_p: f64, seed: u64, max_tokens: u32, stop: []const []const u8, top_k: i64, x_extra: []const u8, response_format: struct { type: []const u8 } } };
    const got = (try std.json.parseFromSliceLeaky(R, a, r.body, .{ .ignore_unknown_fields = true })).received;
    try testing.expectEqualStrings("chat", got.model);
    try testing.expectEqual(@as(f64, 0), got.temperature);
    try testing.expectEqual(@as(f64, 0.9), got.top_p);
    try testing.expectEqual(@as(u64, 5), got.seed);
    try testing.expectEqual(@as(u32, 50), got.max_tokens);
    try testing.expectEqualStrings("zzz", got.stop[0]);
    try testing.expectEqual(@as(i64, 3), got.top_k);
    try testing.expectEqualStrings("y", got.x_extra);
    try testing.expectEqualStrings("text", got.response_format.type);

    // Reasoning off, both ways the engine takes it.
    try testing.expect(message(try whole(a, try ask(&d, a, "hi", ",\"reasoning_effort\":\"none\""))).get("reasoning_content") == null);
    try testing.expect(message(try whole(a, try ask(&d, a, "hi", ",\"chat_template_kwargs\":{\"enable_thinking\":false}"))).get("reasoning_content") == null);

    // Limits and formats: max_tokens cuts with finish_reason length, stop cuts, JSON mode gives JSON.
    const cut = try whole(a, try ask(&d, a, "hi", ",\"max_tokens\":1"));
    try testing.expectEqualStrings("echo:", str(message(cut), "content").?);
    try testing.expectEqualStrings("length", str(first(cut), "finish_reason").?);
    try testing.expectEqualStrings("echo: ", str(message(try whole(a, try ask(&d, a, "hi", ",\"stop\":\"hi\""))), "content").?);
    const js = try whole(a, try ask(&d, a, "hi", ",\"response_format\":{\"type\":\"json_object\"}"));
    const parsed = try std.json.parseFromSliceLeaky(Value, a, str(message(js), "content").?, .{});
    try testing.expectEqualStrings("hi", parsed.object.get("echo").?.string);
}

test "chat streaming: SSE framing, deltas join to the whole reply, the usage chunk, headers" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "chat-stream", "", try chatTool(a, "chat", "", ""));
    defer d.stop();

    const o = try whole(a, try ask(&d, a, "tell me more", ""));
    const s = try sse(a, try ask(&d, a, "tell me more", ",\"stream\":true"));
    try testing.expectEqualStrings(str(message(o), "content").?, s.content);
    try testing.expectEqualStrings(str(message(o), "reasoning_content").?, s.reasoning);
    try testing.expectEqualStrings("stop", s.finish);
    try testing.expect(!s.usage);

    // include_usage: a final chunk with no choices and the usage.
    const ur = try ask(&d, a, "tell me more", ",\"stream\":true,\"stream_options\":{\"include_usage\":true}");
    try expectStatus(ur, .ok, &.{ "\"choices\":[]", "\"completion_tokens\":4" });
    try testing.expect((try sse(a, ur)).usage);
    const short = try sse(a, try ask(&d, a, "tell me more", ",\"stream\":true,\"max_tokens\":2"));
    try testing.expectEqualStrings("echo: tell", short.content);
    try testing.expectEqualStrings("length", short.finish);

    const head = try headers(&d, a, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true}");
    for ([_][]const u8{ "content-type: text/event-stream", "cache-control: no-cache" }) |n| if (std.mem.indexOf(u8, head, n) == null) {
        std.debug.print("missing '{s}' in {s}\n", .{ n, head });
        return error.TestExpectedEqual;
    };
    try testing.expect(std.mem.indexOf(u8, try headers(&d, a, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"), "content-type: application/json") != null);
}

test "chat tool calls: whole and streamed, tool_choice none, and the follow-up with the tool's result" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "chat-tools", "", try chatTool(a, "chat", "", ""));
    defer d.stop();

    const o = try whole(a, try ask(&d, a, "weather in Paris", weather_tools ++ ",\"tool_choice\":\"auto\",\"parallel_tool_calls\":false"));
    try testing.expectEqualStrings("tool_calls", str(first(o), "finish_reason").?);
    try testing.expect(message(o).get("content").? == .null);
    const call = message(o).get("tool_calls").?.array.items[0].object;
    try testing.expectEqualStrings("call_1", str(call, "id").?);
    const f = call.get("function").?.object;
    try testing.expectEqualStrings("get_weather", str(f, "name").?);
    try testing.expectEqualStrings("{\"echo\": \"weather in Paris\"}", str(f, "arguments").?);

    const s = try sse(a, try ask(&d, a, "weather in Paris", weather_tools ++ ",\"stream\":true"));
    try testing.expectEqualStrings("get_weather", s.tool);
    try testing.expectEqualStrings(str(f, "arguments").?, s.arguments);
    try testing.expectEqualStrings("tool_calls", s.finish);
    try testing.expectEqualStrings("", s.content);

    const none = try whole(a, try ask(&d, a, "weather in Paris", weather_tools ++ ",\"tool_choice\":\"none\""));
    try testing.expectEqualStrings("echo: weather in Paris", str(message(none), "content").?);

    // The assistant's call and the tool's answer go back; the engine answers from the tool message.
    const follow =
        \\{"messages":[{"role":"user","content":"weather in Paris"},
        \\{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"echo\": \"weather in Paris\"}"}}]},
        \\{"role":"tool","tool_call_id":"call_1","content":"sunny"}]}
    ;
    const after = try whole(a, try post(&d, a, follow));
    try testing.expectEqualStrings("echo: sunny", str(message(after), "content").?);
    try testing.expectEqual(@as(usize, 3), after.get("received").?.object.get("messages").?.array.items.len);
}

test "chat errors: engine refusals mapped, request validation, a dead server reloads, a failed start" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try std.fmt.allocPrint(a, "{s}, {s}", .{ try chatTool(a, "chat", "", ""), try chatTool(a, "broken", ", \"keep_loaded\": false", ", \"--fail-start\"") });
    var d = try start(a, "chat-errors", "", tools);
    defer d.stop();

    try expectStatus(try ask(&d, a, "hi", ",\"n\":2"), .bad_request, &.{ "n must be 1; multiple choices are not supported", "invalid_request_error" });
    try expectStatus(try ask(&d, a, "hi", ",\"logprobs\":true"), .bad_request, &.{"logprobs are not supported by this model"});
    try expectStatus(try ask(&d, a, "overload", ""), .service_unavailable, &.{ "server busy", "server_error" });
    try expectStatus(try ask(&d, a, "explode", ""), .internal_server_error, &.{ "generation failed", "server_error" });
    // Refused before streaming: the plain JSON error, not SSE.
    const refused = try ask(&d, a, "hi", ",\"n\":2,\"stream\":true");
    try expectStatus(refused, .bad_request, &.{"n must be 1"});
    try testing.expect(refused.body[0] == '{' and std.mem.indexOf(u8, refused.body, "data: ") == null);

    try expectStatus(try post(&d, a, "[1,2]"), .bad_request, &.{"the body is not a valid chat request (a JSON object with messages)"});
    try expectStatus(try post(&d, a, "not json"), .bad_request, &.{"the body is not a valid chat request"});
    try expectStatus(try post(&d, a, "{\"messages\":[]}"), .bad_request, &.{"messages is required: a non-empty array of messages"});
    try expectStatus(try post(&d, a, "{\"model\":\"chat\"}"), .bad_request, &.{"messages is required"});
    try expectStatus(try ask(&d, a, "hi", ",\"stream\":\"yes\""), .bad_request, &.{"stream must be true or false"});

    // The server dies mid-request: 503, then the next request loads it again.
    try expectStatus(try ask(&d, a, "die", ""), .service_unavailable, &.{ "server_error", "the model's server is gone" });
    try expectStatus(try ask(&d, a, "hi", ""), .ok, &.{"echo: hi"});
    // It dies mid-stream: what arrived is kept, then an error event and [DONE]; the next request loads it again.
    const mid = try ask(&d, a, "midfail", ",\"stream\":true");
    const s = try sse(a, mid);
    try testing.expectEqualStrings("w0 w1", s.content);
    try expectStatus(mid, .ok, &.{"data: {\"error\":{"});
    try expectStatus(try ask(&d, a, "hi", ""), .ok, &.{"echo: hi"});

    // A server that exits before it is healthy fails the load, saying so.
    try expectStatus(try ask(&d, a, "hi", ",\"model\":\"broken\""), .service_unavailable, &.{ "server_error", "load failed", "exited before it was ready" });
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "broken"));
}

test "chat qualification gate: unqualified refuses, is never loaded, and serves once the file says qualified" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rnd: [8]u8 = undefined; // unique per run, like the daemon's own directory
    io.random(&rnd);
    const q = try std.fmt.allocPrint(a, "/tmp/localrouter-it-chat-qual-{x:0>16}.json", .{std.mem.readInt(u64, &rnd, .little)});
    defer Io.Dir.cwd().deleteFile(io, q) catch {};
    var d = try start(a, "chat-qual", "", try chatTool(a, "chat", try std.fmt.allocPrint(a, ", \"qualification\": \"{s}\"", .{q}), ""));
    defer d.stop();

    try expectStatus(try d.call(a, .GET, "/v1/models", null), .ok, &.{"\"qualified\":false"});
    try expectStatus(try ask(&d, a, "hi", ""), .service_unavailable, &.{ "server_error", try std.fmt.allocPrint(a, "model 'chat' is not GPU-qualified on this host: {s} does not exist", .{q}) });
    sleepMs(io, 3000);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "chat")); // keep_loaded does not load it
    try expectStatus(try d.call(a, .POST, "/v1/tools/chat/load", ""), .service_unavailable, &.{"not GPU-qualified"});

    try Io.Dir.cwd().writeFile(io, .{ .sub_path = q, .data = "{\"qualified\": false}" });
    try expectStatus(try ask(&d, a, "hi", ""), .service_unavailable, &.{"does not say \\\"qualified\\\": true"});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = q, .data = "{\"qualified\": true}" });
    try expectStatus(try d.call(a, .GET, "/v1/models", null), .ok, &.{"\"qualified\":true"});
    try expectStatus(try ask(&d, a, "hi", ""), .ok, &.{"echo: hi"});
}

test "chat eviction: priority 100 outlasts image tools, still yields when nothing else makes room, comes back" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Budget 100 MiB. `chat` 40 MiB (priority 100, keep_loaded); `img-a`, `img-b` 40 MiB and `huge` 90 MiB (priority 0);
    // a 256x256 image needs 384 KiB of working memory on top.
    const images = "{\"id\": \"img-a\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 600, \"options\": {\"resident_mb\": 40}}," ++
        "{\"id\": \"img-b\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 600, \"options\": {\"resident_mb\": 40}}," ++
        "{\"id\": \"huge\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 1, \"options\": {\"resident_mb\": 90}}";
    var d = try start(a, "chat-evict", ", \"budget_bytes\": 104857600", try std.fmt.allocPrint(a, "{s}, {s}", .{ try chatTool(a, "chat", "", ""), images }));
    defer d.stop();

    try waitState(&d, a, "chat", "ready", 80);
    // chat + img-a fit (80 MiB); img-b does not fit beside both, and img-a, not the more recently used chat, goes.
    try expectStatus(try harness.image(&d, a, "img-a", ""), .ok, &.{});
    try testing.expectEqualStrings("ready", try d.toolState(a, "chat"));
    try expectStatus(try harness.image(&d, a, "img-b", ""), .ok, &.{});
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "img-a"));
    try testing.expectEqualStrings("ready", try d.toolState(a, "img-b"));
    try testing.expectEqualStrings("ready", try d.toolState(a, "chat"));

    // `huge` fits only alone: img-b goes first, then chat.
    try expectStatus(try harness.image(&d, a, "huge", ""), .ok, &.{});
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "img-b"));
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "chat"));

    // `huge` idles out after 1 s; chat reloads by itself once the scheduler is idle and it fits.
    try waitState(&d, a, "huge", "unloaded", 20);
    try waitState(&d, a, "chat", "ready", 60);
    try expectStatus(try ask(&d, a, "hi", ""), .ok, &.{"echo: hi"});
}

test "chat cancel: a streaming client that hangs up cancels the job and the model stays loaded" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "chat-cancel", "", try chatTool(a, "chat", "", ""));
    defer d.stop();

    try waitState(&d, a, "chat", "ready", 80);
    const pids = try d.workers(a);
    try testing.expectEqual(@as(usize, 1), pids.len);
    {
        // "slow" streams 40 words 0.25 s apart (about 11 s); read two content events, then close the connection.
        const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/chat/completions", .{d.port});
        var req = try d.client.request(.POST, try std.Uri.parse(url), .{ .keep_alive = false, .headers = .{ .content_type = .{ .override = "application/json" } } });
        defer req.deinit();
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"slow\"}],\"stream\":true}";
        req.transfer_encoding = .{ .content_length = body.len };
        try req.sendBodyComplete(@constCast(body));
        var redirect: [0]u8 = undefined;
        var res = try req.receiveHead(&redirect);
        try testing.expectEqual(std.http.Status.ok, res.head.status);
        var buf: [4096]u8 = undefined;
        const r = res.reader(&buf);
        var seen: usize = 0;
        while (seen < 2) {
            const line = try r.takeDelimiterInclusive('\n');
            if (std.mem.indexOf(u8, line, "\"content\"") != null) seen += 1;
        }
    }
    // Back to ready well before the stream would have ended, with the same worker.
    try waitState(&d, a, "chat", "ready", 20);
    const after = try d.workers(a);
    try testing.expectEqual(@as(usize, 1), after.len);
    try testing.expectEqual(pids[0], after[0]);
    try expectStatus(try ask(&d, a, "hi", ""), .ok, &.{"echo: hi"});
}

/// Opens `path` and takes an exclusive flock without waiting: the fd that holds it, or null when another holds it.
fn tryLock(path: [:0]const u8) !?std.posix.fd_t {
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, path, .{}, 0);
    const e = std.posix.errno(std.posix.system.flock(fd, std.posix.LOCK.EX | std.posix.LOCK.NB));
    if (e == .SUCCESS) return fd;
    _ = std.posix.system.close(fd);
    if (e == .AGAIN) return null;
    std.debug.print("flock {s}: {s}\n", .{ path, @tagName(e) });
    return error.Unexpected;
}

test "chat GPU lock: held while the server runs, released on unload, a held lock fails the load" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const lock = "/tmp/localrouter-it-chat-gpu.lock"; // outside the daemon's directory: it must exist before the start
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = lock, .data = "" });
    defer Io.Dir.cwd().deleteFile(io, lock) catch {};
    var d = try start(a, "chat-lock", ", \"gpu_lock\": \"" ++ lock ++ "\"", try chatTool(a, "chat", "", ""));
    defer d.stop();

    try waitState(&d, a, "chat", "ready", 80);
    try testing.expect((try tryLock(lock)) == null);
    try expectStatus(try d.call(a, .POST, "/v1/tools/chat/unload", ""), .ok, &.{});
    const held = (try tryLock(lock)) orelse return error.TestExpectedEqual;

    // Held by someone else: the load fails, over the API and for a request.
    try expectStatus(try d.call(a, .POST, "/v1/tools/chat/load", ""), .service_unavailable, &.{"the GPU lock " ++ lock ++ " is held by another job"});
    try expectStatus(try ask(&d, a, "hi", ""), .service_unavailable, &.{ "server_error", "GPU lock" });
    _ = std.posix.system.close(held);
    try expectStatus(try ask(&d, a, "hi", ""), .ok, &.{"echo: hi"});
    try testing.expect((try tryLock(lock)) == null);
}

test "chat over MCP: list_models, prompt and messages, reasoning off, tool calls, tool errors" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try start(a, "chat-mcp", "", try chatTool(a, "chat", "", ""));
    defer d.stop();

    const models = (try mcpText(a, try mcpCall(&d, a, "list_models", "{}"))).object.get("models").?.array.items;
    try testing.expectEqualStrings("chat", str(models[0].object, "kind").?);
    try testing.expectEqualStrings("chat_completion", models[0].object.get("capabilities").?.array.items[0].string);
    try testing.expect(models[0].object.get("keep_loaded").?.bool);
    try testing.expectEqual(@as(i64, 100), models[0].object.get("priority").?.integer);

    // The reply's content first, then the whole completion.
    const hi = try mcpCall(&d, a, "chat", "{\"prompt\":\"hi\"}");
    try testing.expect(!hi.object.get("isError").?.bool);
    try testing.expectEqualStrings("echo: hi", text(hi, 0));
    try testing.expectEqualStrings("thinking about hi", str(message((try std.json.parseFromSliceLeaky(Value, a, text(hi, 1), .{})).object), "reasoning_content").?);
    const plain = (try std.json.parseFromSliceLeaky(Value, a, text(try mcpCall(&d, a, "chat", "{\"prompt\":\"hi\",\"reasoning\":false}"), 1), .{})).object;
    try testing.expect(message(plain).get("reasoning_content") == null);
    const sent = plain.get("received").?.object;
    try testing.expectEqualStrings("none", str(sent, "reasoning_effort").?);
    try testing.expect(!sent.get("chat_template_kwargs").?.object.get("enable_thinking").?.bool);

    // system first, then messages, then prompt.
    const conv = try mcpCall(&d, a, "chat", "{\"prompt\":\"hi\",\"system\":\"be brief\",\"max_tokens\":9,\"messages\":[{\"role\":\"user\",\"content\":\"first\"},{\"role\":\"assistant\",\"content\":\"ok\"}]}");
    const got = (try std.json.parseFromSliceLeaky(Value, a, text(conv, 1), .{})).object.get("received").?.object;
    const msgs = got.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 4), msgs.len);
    for (msgs, [_][]const u8{ "system", "user", "assistant", "user" }, [_][]const u8{ "be brief", "first", "ok", "hi" }) |m, role, content| {
        try testing.expectEqualStrings(role, str(m.object, "role").?);
        try testing.expectEqualStrings(content, str(m.object, "content").?);
    }
    const mt = got.get("max_tokens").?;
    try testing.expect((mt == .integer and mt.integer == 9) or (mt == .float and mt.float == 9));

    // Tool calls: the first text is the calls' JSON.
    const tc = try mcpCall(&d, a, "chat", "{\"prompt\":\"weather in Paris\"" ++ weather_tools ++ "}");
    try testing.expect(!tc.object.get("isError").?.bool);
    const calls = try std.json.parseFromSliceLeaky(Value, a, text(tc, 0), .{});
    try testing.expectEqualStrings("get_weather", str(calls.array.items[0].object.get("function").?.object, "name").?);

    const busy = try mcpCall(&d, a, "chat", "{\"prompt\":\"overload\"}");
    try testing.expect(busy.object.get("isError").?.bool);
    try testing.expect(std.mem.indexOf(u8, text(busy, 0), "server_error") != null);
    const empty = try mcpCall(&d, a, "chat", "{}");
    try testing.expect(empty.object.get("isError").?.bool);
    try testing.expect(std.mem.indexOf(u8, text(empty, 0), "prompt or messages is required") != null);
}
