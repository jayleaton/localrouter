//! The Model Context Protocol on `/mcp` (Streamable HTTP, stateless): any MCP client on the network calls LocalRouter's
//! tools directly. Each POST carries one JSON-RPC message; a request gets its JSON response in the body, a
//! notification 202. No session and no server-sent stream, so GET and DELETE are 405. The tools are the API's jobs:
//! a model loads on its first call and unloads after its idle TTL, as for the HTTP routes.
//!
//!   list_models      the tools, their kind, capabilities, priority and whether they are loaded
//!   generate_image   waits for the image(s); returns them inline (PNG) and as URLs on this server; `images` makes it an edit
//!   generate_video   queues a clip and waits up to `wait_s` (default 600): the MP4's URL when done, else the job id;
//!                    `image` makes it image to video
//!   transcribe_audio waits up to `wait_s` for the transcript of an audio file (base64, data URL or URL): its text and
//!                    JSON (language, segments, words), else the job id; get_job returns the transcript once done
//!   synthesize_speech waits for spoken text; returns the audio inline (WAV) and as a URL
//!   chat             a chat completion from the chat model (whole): the reply's text, then the completion as JSON
//!   get_job          a job's status, progress and output URLs; `wait_s` waits for it to finish
//!   cancel_job       cancels a queued or running job
//!   release_models   unloads every tool now (memory back to the machine before the idle TTL)

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const routes = @import("routes.zig");
const request = @import("../tool/request.zig");
const jobs = @import("../sched/jobs.zig");
const sched = @import("../sched/scheduler.zig");
const inputs = @import("inputs.zig");
const engine = @import("../engine/engine.zig");
const speech = @import("speech.zig");
const chat = @import("chat.zig");
const audio = @import("../media/audio.zig");
const config = @import("../config.zig");
const sched_memory = @import("../sched/memory.zig");

const App = routes.App;
const Value = std.json.Value;

const versions = [_][]const u8{ "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05" };
const max_wait_s = 3000; // a tool call longer than this returns the job id; get_job picks it up

pub fn handle(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8) !void {
    if (req.head.method != .POST) {
        try req.respond("", .{ .status = .method_not_allowed, .extra_headers = &.{.{ .name = "allow", .value = "POST" }} });
        return;
    }
    const text = h.body(req, a) catch return rpcError(req, a, .null, -32600, "body too large");
    const msg = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return rpcError(req, a, .null, -32700, "parse error");
    if (msg != .object) return rpcError(req, a, .null, -32600, "one JSON-RPC message a request (no batches)");
    const id = msg.object.get("id") orelse {
        try req.respond("", .{ .status = .accepted }); // a notification (initialized, cancelled) or a response
        return;
    };
    const method = str(msg.object.get("method")) orelse return rpcError(req, a, id, -32600, "no method");
    const params: Value = msg.object.get("params") orelse .{ .object = .empty };
    if (eql(method, "initialize")) return initialize(req, a, id, params);
    if (eql(method, "ping")) return reply(req, a, id, .{});
    if (eql(method, "tools/list")) return reply(req, a, id, .{ .tools = tool_list });
    if (eql(method, "tools/call")) return call(app, req, a, host, id, params);
    return rpcError(req, a, id, -32601, "method not found");
}

fn initialize(req: *h.Request, a: std.mem.Allocator, id: Value, params: Value) !void {
    var version: []const u8 = versions[1];
    if (params == .object) if (str(params.object.get("protocolVersion"))) |want| {
        for (versions) |v| if (eql(v, want)) {
            version = v;
        };
    };
    try reply(req, a, id, .{
        .protocolVersion = version,
        .capabilities = .{ .tools = .{ .listChanged = false } },
        .serverInfo = .{ .name = "localrouter", .version = "0.1.0" },
        .instructions = instructions,
    });
}

const instructions =
    \\Chat, generative and speech models running locally on one GPU machine. Call list_models first. A model loads on its first use (a few seconds),
    \\then answers in seconds; it unloads by itself when idle. Send one request at a time. Images come back inline and as
    \\URLs; videos as an MP4 URL (generate_video waits for it, or returns a job id for get_job); transcripts as text and
    \\JSON; speech as WAV inline and a URL; chat replies as text (chat keeps its model loaded, so it answers at once).
;

// ---------------------------------------------------------------- tools/call

const Content = struct {
    type: []const u8,
    text: ?[]const u8 = null,
    data: ?[]const u8 = null,
    mimeType: ?[]const u8 = null,
    uri: ?[]const u8 = null,
    name: ?[]const u8 = null,
};

fn call(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, id: Value, params: Value) !void {
    if (params != .object) return rpcError(req, a, id, -32602, "params must be an object");
    const name = str(params.object.get("name")) orelse return rpcError(req, a, id, -32602, "no tool name");
    const args: Value = params.object.get("arguments") orelse .{ .object = .empty };
    var out: std.ArrayList(Content) = .empty;
    calling = name;
    const result: ToolError!bool = if (eql(name, "list_models"))
        listModels(app, a, &out)
    else if (eql(name, "list_voices"))
        listVoices(app, a, &out)
    else if (eql(name, "generate_image"))
        generateImage(app, a, host, args, &out)
    else if (eql(name, "generate_video"))
        generateVideo(app, a, host, args, &out)
    else if (eql(name, "transcribe_audio"))
        transcribeAudio(app, a, host, args, &out)
    else if (eql(name, "synthesize_speech"))
        synthesizeSpeech(app, a, host, args, &out)
    else if (eql(name, "chat"))
        chatTool(app, a, args, &out)
    else if (eql(name, "get_job"))
        getJob(app, a, host, args, &out)
    else if (eql(name, "cancel_job"))
        cancelJob(app, a, host, args, &out)
    else if (eql(name, "release_models"))
        release(app, a, &out)
    else
        return rpcError(req, a, id, -32602, try std.fmt.allocPrint(a, "unknown tool '{s}'; tools/list names them: {s}", .{ name, tool_names }));
    const ok = result catch |err| switch (err) {
        error.ToolFailed, error.BadArguments => false, // the reason is already in `out`
        error.OutOfMemory => return error.OutOfMemory,
    };
    try reply(req, a, id, .{ .content = out.items, .isError = !ok });
}

const ToolError = error{ ToolFailed, BadArguments, OutOfMemory };

fn fail(a: std.mem.Allocator, out: *std.ArrayList(Content), comptime fmt: []const u8, args: anytype) ToolError {
    try out.append(a, .{ .type = "text", .text = try std.fmt.allocPrint(a, fmt, args) });
    return error.ToolFailed;
}

/// The tool being called on this connection's thread, for argument errors that name it.
threadlocal var calling: []const u8 = "";

fn parseArgs(comptime T: type, a: std.mem.Allocator, args: Value, out: *std.ArrayList(Content)) ToolError!T {
    return std.json.parseFromValueLeaky(T, a, args, .{ .ignore_unknown_fields = true }) catch |err| {
        const ename = @errorName(err);
        const why = if (eql(ename, "MissingField")) "a required argument is missing" else if (eql(ename, "OutOfMemory")) return error.OutOfMemory else "an argument has the wrong type or is out of range";
        try out.append(a, .{ .type = "text", .text = try std.fmt.allocPrint(a, "invalid arguments for {s}: {s}; its inputSchema in tools/list names the arguments, their types and which are required", .{ calling, why }) });
        return error.BadArguments;
    };
}

fn listModels(app: *App, a: std.mem.Allocator, out: *std.ArrayList(Content)) ToolError!bool {
    const Entry = struct { id: []const u8, name: []const u8, kind: []const u8, state: []const u8, running: bool, resident_bytes: u64, lane: []const u8, capabilities: []const engine.Capability, default_for: []const engine.Capability, priority: i32, keep_loaded: bool, voices: ?[]const Value = null, default_voice: ?[]const u8 = null, presets: ?[]const Value = null, speed: ?[2]f32 = null, max_input_chars: ?u64 = null, stream: ?bool = null, references: ?bool = null, max_audio_s: ?u64 = null, qualified: ?bool = null };
    const list = try a.alloc(Entry, app.scheduler.slots.len);
    for (list, app.scheduler.slots, 0..) |*e, slot, i| {
        const c = slot.tool.config();
        const st = slot.tool.state();
        e.* = .{ .id = c.id, .name = if (c.name.len > 0) c.name else c.id, .kind = @tagName(c.kind), .state = @tagName(st), .running = st == .ready or st == .busy, .resident_bytes = slot.resident, .lane = @tagName(config.laneOf(c)), .capabilities = try engine.capabilityList(a, c), .default_for = try routes.defaultFor(app, a, i), .priority = c.priority, .keep_loaded = c.keep_loaded };
        const info = engine.audioInfo(c);
        inline for (@typeInfo(engine.AudioInfo).@"struct".field_names) |f| @field(e.*, f) = @field(info, f);
        var buf: [512]u8 = undefined;
        if (c.qualification.len > 0) e.qualified = engine.unqualified(app.io, a, c, &buf) == null;
    }
    const memory = .{ .mem_available_bytes = sched_memory.available() orelse 0, .reserve_bytes = app.cfg.reserve_bytes, .budget_bytes = app.cfg.budget_bytes };
    try out.append(a, .{ .type = "text", .text = try std.json.Stringify.valueAlloc(a, .{ .models = list, .machine = app.cfg.machine, .memory = memory }, .{ .emit_null_optional_fields = false }) });
    return true;
}

/// Every speech model's voices, its default voice, whether it is loaded now and whether it may serve (qualified).
fn listVoices(app: *App, a: std.mem.Allocator, out: *std.ArrayList(Content)) ToolError!bool {
    const Entry = struct { model: []const u8, name: []const u8, voices: []const Value, default_voice: ?[]const u8 = null, state: []const u8, default: bool, qualified: ?bool = null };
    var list: std.ArrayList(Entry) = .empty;
    for (app.scheduler.slots, 0..) |slot, i| {
        const c = slot.tool.config();
        if (c.kind != .speech) continue;
        const info = engine.audioInfo(c);
        var buf: [512]u8 = undefined;
        const defaults = try routes.defaultFor(app, a, i);
        try list.append(a, .{ .model = c.id, .name = if (c.name.len > 0) c.name else c.id, .voices = info.voices orelse &.{}, .default_voice = info.default_voice, .state = @tagName(slot.tool.state()), .default = std.mem.indexOfScalar(engine.Capability, defaults, .text_to_speech) != null, .qualified = if (c.qualification.len > 0) engine.unqualified(app.io, a, c, &buf) == null else null });
    }
    if (list.items.len == 0) return fail(a, out, "no speech model is configured on this machine", .{});
    try out.append(a, .{ .type = "text", .text = try std.json.Stringify.valueAlloc(a, .{ .speech_models = list.items }, .{ .emit_null_optional_fields = false }) });
    return true;
}

fn generateImage(app: *App, a: std.mem.Allocator, host: []const u8, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    const Args = struct { prompt: []const u8, model: ?[]const u8 = null, size: []const u8 = "1024x1024", n: u32 = 1, seed: ?u64 = null, steps: u32 = 0, negative_prompt: []const u8 = "", inline_images: bool = true, images: []const []const u8 = &.{}, wait_s: u32 = 600 };
    const b = try parseArgs(Args, a, args, out);
    const wh = request.parseSize(b.size) orelse return fail(a, out, "size must be WIDTHxHEIGHT", .{});
    const files = try inputFiles(app, a, host, "ref", b.images, out);
    var why: routes.Refusal = .{};
    const tool = routes.resolve(app, b.model, if (files.len > 0) .image_edit else .text_to_image, &why) orelse return fail(a, out, "{s}", .{why.message});
    const refs = try a.alloc([]const u8, files.len);
    for (refs, files) |*d, f| d.* = f.name;
    const r: request.Request = .{ .image = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .n = b.n, .seed = b.seed orelse routes.randomSeed(app.io), .steps = b.steps, .references = refs } };
    const j = try start(app, a, tool, r, files, out);
    try waitFor(app, j, b.wait_s);
    if (!finished(app, j)) return report(app, a, host, j, out); // still loading or drawing: the job id, for get_job
    if (j.status != .completed) return fail(a, out, "{s}: {s}", .{ j.error_type, j.error_message });
    if (b.inline_images) for (j.files) |f| {
        const png = Io.Dir.cwd().readFileAlloc(app.io, try std.fs.path.join(a, &.{ j.dir, f }), a, .limited(h.max_body)) catch
            return fail(a, out, "could not read {s}", .{f});
        const enc = std.base64.standard.Encoder;
        try out.append(a, .{ .type = "image", .data = enc.encode(try a.alloc(u8, enc.calcSize(png.len)), png), .mimeType = "image/png" });
    };
    try summary(app, a, host, j, out);
    return true;
}

fn generateVideo(app: *App, a: std.mem.Allocator, host: []const u8, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    const Args = struct { prompt: []const u8, model: ?[]const u8 = null, size: []const u8 = "", seconds: u32 = 5, fps: u32 = 24, seed: ?u64 = null, steps: u32 = 0, audio: bool = true, negative_prompt: []const u8 = "", wait_s: u32 = 600, image: ?[]const u8 = null };
    const b = try parseArgs(Args, a, args, out);
    const wh: [2]u32 = if (b.size.len == 0) .{ 0, 0 } else request.parseSize(b.size) orelse return fail(a, out, "size must be WIDTHxHEIGHT", .{});
    const files = try inputFiles(app, a, host, "first_frame", if (b.image) |x| &.{x} else &.{}, out);
    var why: routes.Refusal = .{};
    const tool = routes.resolve(app, b.model, if (files.len > 0) .image_to_video else .text_to_video, &why) orelse return fail(a, out, "{s}", .{why.message});
    const r: request.Request = .{ .video = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .seconds = b.seconds, .fps = b.fps, .seed = b.seed orelse routes.randomSeed(app.io), .steps = b.steps, .audio = b.audio, .first_frame = if (files.len > 0) files[0].name else null } };
    const j = try start(app, a, tool, r, files, out);
    try waitFor(app, j, b.wait_s);
    return report(app, a, host, j, out);
}

fn transcribeAudio(app: *App, a: std.mem.Allocator, host: []const u8, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    const Args = struct { audio: ?[]const u8 = null, path: ?[]const u8 = null, model: ?[]const u8 = null, language: []const u8 = "", prompt: []const u8 = "", temperature: f32 = 0, timestamps: request.Timestamps = .none, seed: ?u64 = null, wait_s: u32 = 600 };
    const b = try parseArgs(Args, a, args, out);
    var msg: []const u8 = "";
    const upload = if (b.path) |p| blk: {
        if (b.audio != null) return fail(a, out, "send audio or path, not both", .{});
        break :blk try localFile(app, a, p, out);
    } else inputs.resolve(app.io, app.gpa, a, b.audio orelse return fail(a, out, "audio (base64, a data URL or an http(s) URL) or path (a file on this machine) is required", .{}), app.policy(host), &msg) catch return fail(a, out, "{s}", .{msg});
    if (upload.len > inputs.max_bytes) return fail(a, out, "the audio file is larger than 32 MB", .{});
    var why: routes.Refusal = .{};
    const j = speech.startTranscription(app, a, .{ .model = b.model, .language = b.language, .prompt = b.prompt, .temperature = b.temperature, .timestamps = b.timestamps, .seed = b.seed }, upload, &why) orelse return fail(a, out, "{s}", .{why.message});
    try waitFor(app, j, b.wait_s);
    return report(app, a, host, j, out);
}

/// The bytes of a file on this machine, by absolute path, only inside a directory the operator listed in the config's
/// `local_files` (the API has no authentication, so nothing else on the disk is readable through it).
fn localFile(app: *App, a: std.mem.Allocator, path: []const u8, out: *std.ArrayList(Content)) ToolError![]const u8 {
    if (app.cfg.local_files.len == 0) return fail(a, out, "path is off on this server: the operator lists the directories it may read in the config's local_files; send the audio as base64 or an http(s) URL instead", .{});
    if (!std.fs.path.isAbsolute(path)) return fail(a, out, "path must be absolute (it names a file on the machine that runs LocalRouter)", .{});
    const real = Io.Dir.realPathFileAbsoluteAlloc(app.io, path, a) catch return fail(a, out, "no such file: {s}", .{path});
    for (app.cfg.local_files) |root| {
        const r = Io.Dir.realPathFileAbsoluteAlloc(app.io, root, a) catch continue;
        if (!std.mem.startsWith(u8, real, r) or real.len <= r.len or real[r.len] != '/') continue;
        return Io.Dir.cwd().readFileAlloc(app.io, real, a, .limited(inputs.max_bytes + 1)) catch |err| return switch (err) {
            error.StreamTooLong => fail(a, out, "the audio file is larger than 32 MB", .{}),
            else => fail(a, out, "could not read {s}: {s}", .{ path, @errorName(err) }),
        };
    }
    return fail(a, out, "{s} is outside the directories this server may read (local_files: {s}{s})", .{ path, app.cfg.local_files[0], if (app.cfg.local_files.len > 1) ", ..." else "" });
}

fn synthesizeSpeech(app: *App, a: std.mem.Allocator, host: []const u8, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    const Args = struct { input: []const u8, model: ?[]const u8 = null, voice: []const u8 = "", speed: f32 = 1, preset: []const u8 = "", seed: ?u64 = null, format: []const u8 = "wav", inline_audio: bool = true, references: ?Value = null };
    const b = try parseArgs(Args, a, args, out);
    const format = std.meta.stringToEnum(audio.Format, b.format) orelse return fail(a, out, "format must be wav, pcm, flac, mp3, opus or aac", .{});
    var why: routes.Refusal = .{};
    const j = speech.startSpeech(app, a, .{ .model = b.model, .input = b.input, .voice = b.voice, .speed = b.speed, .preset = b.preset, .seed = b.seed, .references = b.references }, &why) orelse return fail(a, out, "{s}", .{why.message});
    j.done.wait(app.io) catch return fail(a, out, "cancelled", .{});
    if (j.status != .completed) return fail(a, out, "{s}: {s}", .{ j.error_type, j.error_message });
    if (b.inline_audio) {
        const bytes = speech.encodeSpeech(app, a, j, format, &why) orelse return fail(a, out, "{s}", .{why.message});
        const enc = std.base64.standard.Encoder;
        try out.append(a, .{ .type = "audio", .data = enc.encode(try a.alloc(u8, enc.calcSize(bytes.len)), bytes), .mimeType = format.contentType() });
    }
    try summary(app, a, host, j, out);
    return true;
}

/// A whole chat completion. `prompt` becomes the last user message (after `messages`, after a `system` message);
/// `reasoning` false switches the model's thinking off, true on (absent: the model's default). Returns the reply's
/// text (its tool calls as JSON when it made some instead), then the whole completion.
fn chatTool(app: *App, a: std.mem.Allocator, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    if (args != .object) return fail(a, out, "arguments must be an object", .{});
    const given = args.object;
    var body: std.json.ObjectMap = .empty;
    var messages = std.json.Array.init(a);
    if (str(given.get("system"))) |s| try messages.append(try chatMessage(a, "system", s));
    if (given.get("messages")) |m| {
        if (m != .array) return fail(a, out, "messages must be an array of messages", .{});
        try messages.appendSlice(m.array.items);
    }
    if (str(given.get("prompt"))) |s| try messages.append(try chatMessage(a, "user", s));
    if (messages.items.len == 0 or (given.get("prompt") == null and given.get("messages") == null)) return fail(a, out, "prompt or messages is required", .{});
    try body.put(a, "messages", .{ .array = messages });
    for ([_][]const u8{ "model", "max_tokens", "temperature", "top_p", "stop", "seed", "tools", "tool_choice", "response_format" }) |k|
        if (given.get(k)) |v| try body.put(a, k, v);
    if (given.get("reasoning")) |r| if (r == .bool) {
        var kwargs: std.json.ObjectMap = .empty;
        try kwargs.put(a, "enable_thinking", .{ .bool = r.bool });
        try body.put(a, "chat_template_kwargs", .{ .object = kwargs });
        if (!r.bool) try body.put(a, "reasoning_effort", .{ .string = "none" });
    };
    var why: routes.Refusal = .{};
    const j = chat.startChat(app, a, .{ .object = body }, &why) orelse return fail(a, out, "{s}: {s}", .{ why.typ, why.message });
    j.done.wait(app.io) catch return fail(a, out, "cancelled", .{});
    if (j.status != .completed) return fail(a, out, "{s}: {s}", .{ j.error_type, j.error_message });
    const text = chat.readCompletion(app, a, j, &why) orelse return fail(a, out, "{s}: {s}", .{ why.typ, why.message });
    try out.append(a, .{ .type = "text", .text = replyText(a, text) catch "" });
    try out.append(a, .{ .type = "text", .text = text });
    return true;
}

fn chatMessage(a: std.mem.Allocator, role: []const u8, content: []const u8) !Value {
    var m: std.json.ObjectMap = .empty;
    try m.put(a, "role", .{ .string = role });
    try m.put(a, "content", .{ .string = content });
    return .{ .object = m };
}

/// The first choice's text, or its tool calls as JSON when it has no text.
fn replyText(a: std.mem.Allocator, completion: []const u8) ![]const u8 {
    const v = try std.json.parseFromSliceLeaky(Value, a, completion, .{});
    const choices = v.object.get("choices") orelse return "";
    if (choices != .array or choices.array.items.len == 0) return "";
    const msg = choices.array.items[0].object.get("message") orelse return "";
    if (str(msg.object.get("content"))) |s| if (s.len > 0) return s;
    if (msg.object.get("tool_calls")) |calls| return std.json.Stringify.valueAlloc(a, calls, .{});
    return "";
}

fn getJob(app: *App, a: std.mem.Allocator, host: []const u8, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    const b = try parseArgs(struct { id: []const u8, wait_s: u32 = 0 }, a, args, out);
    const j = app.table.get(app.io, b.id) orelse return fail(a, out, "no such job: {s}", .{b.id});
    try waitFor(app, j, b.wait_s);
    return report(app, a, host, j, out);
}

fn cancelJob(app: *App, a: std.mem.Allocator, host: []const u8, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    const b = try parseArgs(struct { id: []const u8 }, a, args, out);
    const j = app.table.get(app.io, b.id) orelse return fail(a, out, "no such job: {s}", .{b.id});
    app.scheduler.cancel(j);
    try waitFor(app, j, 10);
    try summary(app, a, host, j, out);
    return true;
}

fn release(app: *App, a: std.mem.Allocator, out: *std.ArrayList(Content)) ToolError!bool {
    var c: sched.Control = .{ .op = .release };
    app.scheduler.control(&c) catch |err| return fail(a, out, "release failed: {s}", .{@errorName(err)});
    if (c.failed) return fail(a, out, "release failed: {s}", .{c.message});
    try out.append(a, .{ .type = "text", .text = "every model is unloaded" });
    return true;
}

fn start(app: *App, a: std.mem.Allocator, tool: usize, r: request.Request, files: []const inputs.File, out: *std.ArrayList(Content)) ToolError!*jobs.Job {
    var why: routes.Refusal = .{};
    return routes.enqueue(app, tool, r, files, &why) orelse fail(a, out, "{s}", .{why.message});
}

/// The input images named by `specs` (base64, data URLs or http(s) URLs; at most 5), fetched and checked.
fn inputFiles(app: *App, a: std.mem.Allocator, host: []const u8, stem: []const u8, specs: []const []const u8, out: *std.ArrayList(Content)) ToolError![]inputs.File {
    var why: []const u8 = "";
    const raws = try a.alloc([]const u8, specs.len);
    for (raws, specs) |*d, sp| d.* = inputs.resolve(app.io, app.gpa, a, sp, app.policy(host), &why) catch return fail(a, out, "{s}", .{why});
    return inputs.validate(a, stem, raws, (request.Limits{}).max_references, &why) catch fail(a, out, "{s}", .{why});
}

/// Waits until the job finishes or `wait_s` (capped) passes; polls, so a slow client never holds the job.
fn waitFor(app: *App, j: *jobs.Job, wait_s: u32) ToolError!void {
    const deadline = std.Io.Clock.awake.now(app.io).addDuration(.fromSeconds(@min(wait_s, max_wait_s)));
    while (!finished(app, j)) {
        if (std.Io.Clock.awake.now(app.io).nanoseconds >= deadline.nanoseconds) return;
        std.Io.sleep(app.io, .fromMilliseconds(250), .awake) catch return;
    }
}

fn finished(app: *App, j: *jobs.Job) bool {
    app.table.mutex.lockUncancelable(app.io);
    defer app.table.mutex.unlock(app.io);
    return j.finished();
}

/// A finished job's summary (true), a failed one's reason (false), or a running one's progress (true: not an error).
/// A completed transcription leads with its text, then the transcript as JSON.
fn report(app: *App, a: std.mem.Allocator, host: []const u8, j: *jobs.Job, out: *std.ArrayList(Content)) ToolError!bool {
    if (j.kind() == .transcription and finished(app, j) and j.status == .completed) {
        var why: routes.Refusal = .{};
        const t = speech.readTranscript(app, a, j, &why) orelse return fail(a, out, "{s}", .{why.message});
        try out.append(a, .{ .type = "text", .text = t.get("text").?.string });
        try out.append(a, .{ .type = "text", .text = try std.json.Stringify.valueAlloc(a, Value{ .object = t }, .{}) });
    }
    try summary(app, a, host, j, out);
    return j.status != .failed;
}

/// The job as JSON text, plus a link to each output file.
fn summary(app: *App, a: std.mem.Allocator, host: []const u8, j: *jobs.Job, out: *std.ArrayList(Content)) ToolError!void {
    app.table.mutex.lockUncancelable(app.io);
    defer app.table.mutex.unlock(app.io);
    const urls = try a.alloc([]const u8, if (j.status == .completed) j.files.len else 0);
    for (urls, 0..) |*u, i| u.* = try std.fmt.allocPrint(a, "http://{s}/v1/files/{s}/{s}", .{ host, j.id, j.files[i] });
    const obj = .{
        .id = j.id,
        .model = app.scheduler.slots[j.tool].tool.config().id,
        .status = @tagName(j.status),
        .progress = j.percent,
        .phase = try a.dupe(u8, j.phaseName()),
        .seed = j.request.seed(),
        .ms = j.ms,
        .files = urls,
        .finish_reason = if (j.finish_reason.len > 0) @as(?[]const u8, j.finish_reason) else null,
        .segments = if (j.segments.len > 0) @as(?[]const u8, j.segments) else null,
        .@"error" = if (j.status == .failed or j.status == .cancelled) @as(?[]const u8, try a.dupe(u8, j.error_message)) else null,
        .next = if (j.finished()) @as(?[]const u8, null) else "call get_job with this id (wait_s up to 3000) until status is completed",
    };
    try out.append(a, .{ .type = "text", .text = try std.json.Stringify.valueAlloc(a, obj, .{ .emit_null_optional_fields = false }) });
    for (urls, 0..) |u, i| {
        const f = j.files[i];
        try out.append(a, .{ .type = "resource_link", .uri = u, .name = f, .mimeType = routes.contentType(f) });
    }
}

// ---------------------------------------------------------------- JSON-RPC

fn reply(req: *h.Request, a: std.mem.Allocator, id: Value, result: anytype) !void {
    try h.jsonValue(req, a, .ok, .{ .jsonrpc = "2.0", .id = id, .result = result });
}

fn rpcError(req: *h.Request, a: std.mem.Allocator, id: Value, code: i32, message: []const u8) !void {
    try h.jsonValue(req, a, .ok, .{ .jsonrpc = "2.0", .id = id, .@"error" = .{ .code = code, .message = message } });
}

fn str(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

// ---------------------------------------------------------------- tools/list

const Tool = struct { name: []const u8, title: []const u8, description: []const u8, inputSchema: RawJson };

/// A JSON fragment written as is (the input schemas).
const RawJson = struct {
    text: []const u8,
    pub fn jsonStringify(r: RawJson, w: anytype) !void {
        try w.beginWriteRaw();
        try w.writer.writeAll(r.text);
        w.endWriteRaw();
    }
};

const tool_list = [_]Tool{
    .{ .name = "list_models", .title = "List models", .description = "The models on this machine: id, kind (image, video, speech, transcription or chat), capabilities (text_to_image, image_edit, text_to_video, image_to_video, text_to_speech, speech_to_text, chat_completion), voices (speech models), default_for (the capabilities it serves when a request names no model), priority and keep_loaded, whether each is loaded now (state: ready or busy is fast; unloaded loads on first use), its resident_bytes and scheduler lane, qualified (false: refused until this machine has qualified it), and the machine's memory (mem_available_bytes, reserve_bytes). Call it first.", .inputSchema = .{ .text =
    \\{"type":"object","properties":{},"additionalProperties":false}
    } },
    .{ .name = "list_voices", .title = "List voices", .description = "The voices of every speech model: model id, voices, default_voice, state (ready: speaks at once), default (true: synthesize_speech uses it when no model is named) and qualified. Pass one of these voices to synthesize_speech.", .inputSchema = .{ .text =
    \\{"type":"object","properties":{},"additionalProperties":false}
    } },
    .{ .name = "generate_image", .title = "Generate image", .description = "Text to image, or an image edit when `images` is given. Waits up to wait_s (seconds when the model is loaded, plus its load time when not) and returns the PNG(s) inline and as URLs on this server; past wait_s it returns the job id for get_job. Reuse a returned seed to reproduce an image. Edits need a model whose capabilities (list_models) include image_edit.", .inputSchema = .{ .text =
    \\{"type":"object","required":["prompt"],"properties":{
    \\"prompt":{"type":"string","description":"What to draw."},
    \\"model":{"type":"string","description":"An image model id from list_models; default: the model whose default_for lists the capability this request needs (text_to_image, or image_edit with images)."},
    \\"size":{"type":"string","description":"WIDTHxHEIGHT, multiples of 16, 256 to 2048 a side; about 1 megapixel is best (1024x1024, 1360x768, 768x1360).","default":"1024x1024"},
    \\"n":{"type":"integer","minimum":1,"maximum":4,"default":1},
    \\"seed":{"type":"integer","minimum":0,"description":"Omit for a random one."},
    \\"steps":{"type":"integer","minimum":0,"description":"0: the model's default."},
    \\"negative_prompt":{"type":"string"},
    \\"images":{"type":"array","maxItems":5,"items":{"type":"string"},"description":"Input images (PNG, JPEG or WebP, up to 32 MB each) as base64 strings, data URLs or http(s) URLs. Given: the request is an edit of these images, guided by the prompt."},
    \\"inline_images":{"type":"boolean","default":true,"description":"false: URLs only (smaller replies)."},
    \\"wait_s":{"type":"integer","minimum":0,"maximum":3000,"default":600,"description":"How long to wait before returning the job id instead (use a small value if your tool calls time out)."}},
    \\"additionalProperties":false}
    } },
    .{ .name = "generate_video", .title = "Generate video", .description = "Text to video with audio (MP4), or image to video when `image` is given. Waits up to wait_s seconds; returns the MP4's URL when done, otherwise the job id for get_job.", .inputSchema = .{ .text =
    \\{"type":"object","required":["prompt"],"properties":{
    \\"prompt":{"type":"string","description":"The scene, the motion and the sound."},
    \\"model":{"type":"string","description":"A video model id from list_models; default: the model whose default_for lists the capability this request needs (text_to_video, or image_to_video with image)."},
    \\"size":{"type":"string","description":"WIDTHxHEIGHT within the model's limits; omit it for the model's own size."},
    \\"seconds":{"type":"integer","minimum":1,"maximum":20,"default":5},
    \\"fps":{"type":"integer","default":24},
    \\"seed":{"type":"integer","minimum":0},
    \\"steps":{"type":"integer","minimum":0,"description":"0: the model's default."},
    \\"audio":{"type":"boolean","default":true},
    \\"negative_prompt":{"type":"string"},
    \\"image":{"type":"string","description":"First frame (image to video): a base64 string, data URL or http(s) URL of a PNG, JPEG or WebP up to 32 MB. Needs a model whose capabilities (list_models) include image_to_video."},
    \\"wait_s":{"type":"integer","minimum":0,"maximum":3000,"default":600,"description":"How long to wait before returning the job id instead."}},
    \\"additionalProperties":false}
    } },
    .{ .name = "transcribe_audio", .title = "Transcribe audio", .description = "Speech to text in the spoken language (no translation). Waits up to wait_s for the transcript and returns its text, then JSON with language, duration and segments (and words when timestamps is word); a longer job returns its id for get_job. Audio up to 32 MB and 30 minutes: WAV, MP3, FLAC, OGG/Opus, M4A/MP4, WebM or AAC.", .inputSchema = .{ .text =
    \\{"type":"object","properties":{
    \\"audio":{"type":"string","description":"The audio file as a base64 string, a data URL or an http(s) URL (the daemon's own /v1/files URLs too). Send this or path."},
    \\"path":{"type":"string","description":"An absolute path to an audio file on the machine that runs LocalRouter, inside a directory the operator listed in local_files (off by default). Send this or audio."},
    \\"model":{"type":"string","description":"A transcription model id from list_models; default: the one whose default_for lists speech_to_text."},
    \\"language":{"type":"string","description":"ISO-639-1 code of the spoken language (en, de, ja...); omit to detect it."},
    \\"prompt":{"type":"string","description":"Text that guides spelling and style (names, terms, the previous passage)."},
    \\"timestamps":{"type":"string","enum":["none","segment","word"],"default":"none"},
    \\"temperature":{"type":"number","minimum":0,"maximum":1,"default":0},
    \\"wait_s":{"type":"integer","minimum":0,"maximum":3000,"default":600,"description":"How long to wait before returning the job id instead (get_job returns the transcript; cancel_job stops it)."}},
    \\"additionalProperties":false}
    } },
    .{ .name = "synthesize_speech", .title = "Synthesize speech", .description = "Text to speech. Waits for the audio and returns it inline (WAV unless format says otherwise) and as a URL on this server. Voices are listed per model in list_models.", .inputSchema = .{ .text =
    \\{"type":"object","required":["input"],"properties":{
    \\"input":{"type":"string","description":"The text to speak, up to 4096 characters."},
    \\"model":{"type":"string","description":"A speech model id from list_models; default: the one whose default_for lists text_to_speech."},
    \\"voice":{"type":"string","description":"A voice id from the model's voices in list_models; omit for its default."},
    \\"speed":{"type":"number","minimum":0.25,"maximum":4,"default":1,"description":"Within the model's own speed range (list_models speed); a model without a speed control takes only 1."},
    \\"preset":{"type":"string","description":"A model-native style from the model's presets in list_models; omit for none."},
    \\"format":{"type":"string","enum":["wav","pcm","flac","mp3","opus","aac"],"default":"wav"},
    \\"seed":{"type":"integer","minimum":0},
    \\"references":{"type":"array","items":{"type":"object","required":["text","vq_codes"],"properties":{"text":{"type":"string"},"vq_codes":{"type":"array","items":{"type":"array","items":{"type":"integer"}}}}},"description":"Voice references (a clip's transcript and codec codes) for a model whose list_models entry says references: true."},
    \\"inline_audio":{"type":"boolean","default":true,"description":"false: the URL only (smaller replies)."}},
    \\"additionalProperties":false}
    } },
    .{ .name = "chat", .title = "Chat", .description = "A chat completion from the local chat model (OpenAI Chat Completions semantics). Returns the reply's text (or its tool calls as JSON when it called a tool instead), then the whole completion as JSON (message, reasoning_content when the model thought, finish_reason, usage). The chat model stays loaded, so replies start at once.", .inputSchema = .{ .text =
    \\{"type":"object","properties":{
    \\"prompt":{"type":"string","description":"The user's message (appended after messages)."},
    \\"messages":{"type":"array","items":{"type":"object"},"description":"OpenAI chat messages (role system, user, assistant or tool; content; tool_calls; tool_call_id), for a multi-turn conversation."},
    \\"system":{"type":"string","description":"A system message, put first."},
    \\"model":{"type":"string","description":"A chat model id from list_models; default: the one whose default_for lists chat_completion."},
    \\"reasoning":{"type":"boolean","description":"false: answer without thinking (faster); true: think first. Omit for the model's default."},
    \\"max_tokens":{"type":"integer","minimum":1},
    \\"temperature":{"type":"number","minimum":0,"description":"0: greedy and deterministic."},
    \\"top_p":{"type":"number"},
    \\"stop":{"description":"A string or a list of strings that end the reply."},
    \\"seed":{"type":"integer","minimum":0},
    \\"tools":{"type":"array","items":{"type":"object"},"description":"OpenAI function tools the model may call."},
    \\"tool_choice":{"description":"none, auto, required, or {\"type\":\"function\",\"function\":{\"name\":...}}."},
    \\"response_format":{"type":"object","description":"{\"type\":\"json_object\"} or {\"type\":\"json_schema\",\"json_schema\":{...}} for structured output."}},
    \\"additionalProperties":false}
    } },
    .{ .name = "get_job", .title = "Get job", .description = "A job's status (queued, in_progress, completed, failed, cancelled), progress and output URLs. wait_s waits for it to finish.", .inputSchema = .{ .text =
    \\{"type":"object","required":["id"],"properties":{"id":{"type":"string","description":"A job id from generate_video, generate_image, transcribe_audio or a /v1 response."},"wait_s":{"type":"integer","minimum":0,"maximum":3000,"default":0,"description":"How long to wait for it to finish."}},"additionalProperties":false}
    } },
    .{ .name = "cancel_job", .title = "Cancel job", .description = "Cancels a queued or running job (an image, video or transcription by the id those tools or get_job returned); a chat or speech model stays loaded, any other job's worker is stopped.", .inputSchema = .{ .text =
    \\{"type":"object","required":["id"],"properties":{"id":{"type":"string"}},"additionalProperties":false}
    } },
    .{ .name = "release_models", .title = "Release models", .description = "Unloads every model now, returning its memory to the machine (they also unload by themselves when idle).", .inputSchema = .{ .text =
    \\{"type":"object","properties":{},"additionalProperties":false}
    } },
};

/// "list_models, list_voices, ...": the tool names, for an unknown tool's error.
const tool_names = blk: {
    var s: []const u8 = "";
    for (tool_list, 0..) |t, i| s = s ++ (if (i > 0) ", " else "") ++ t.name;
    break :blk s;
};
