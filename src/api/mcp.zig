//! The Model Context Protocol on `/mcp` (Streamable HTTP, stateless): any MCP client on the network calls LocalRouter's
//! tools directly. Each POST carries one JSON-RPC message; a request gets its JSON response in the body, a
//! notification 202. No session and no server-sent stream, so GET and DELETE are 405. The tools are the API's jobs:
//! a model loads on its first call and unloads after its idle TTL, as for the HTTP routes.
//!
//!   list_models      the tools, their kind, capabilities, priority and whether they are loaded
//!   generate_image   waits for the image(s); returns them inline (PNG) and as URLs on this server; `images` makes it an edit
//!   generate_video   queues a clip and waits up to `wait_s` (default 600): the MP4's URL when done, else the job id;
//!                    `image` makes it image to video
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
    \\Generative models running locally on one GPU machine. Call list_models first. A model loads on its first use (a few seconds),
    \\then answers in seconds; it unloads by itself when idle. Send one request at a time. Images come back inline and as
    \\URLs; videos as an MP4 URL (generate_video waits for it, or returns a job id for get_job).
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
    const result: ToolError!bool = if (eql(name, "list_models"))
        listModels(app, a, &out)
    else if (eql(name, "generate_image"))
        generateImage(app, a, host, args, &out)
    else if (eql(name, "generate_video"))
        generateVideo(app, a, host, args, &out)
    else if (eql(name, "get_job"))
        getJob(app, a, host, args, &out)
    else if (eql(name, "cancel_job"))
        cancelJob(app, a, host, args, &out)
    else if (eql(name, "release_models"))
        release(app, a, &out)
    else
        return rpcError(req, a, id, -32602, "unknown tool");
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

fn parseArgs(comptime T: type, a: std.mem.Allocator, args: Value, out: *std.ArrayList(Content)) ToolError!T {
    return std.json.parseFromValueLeaky(T, a, args, .{ .ignore_unknown_fields = true }) catch |err| {
        try out.append(a, .{ .type = "text", .text = try std.fmt.allocPrint(a, "invalid arguments: {s}", .{@errorName(err)}) });
        return error.BadArguments;
    };
}

fn listModels(app: *App, a: std.mem.Allocator, out: *std.ArrayList(Content)) ToolError!bool {
    const Entry = struct { id: []const u8, name: []const u8, kind: []const u8, running: bool, capabilities: []const engine.Capability, priority: i32, keep_loaded: bool };
    const list = try a.alloc(Entry, app.scheduler.slots.len);
    for (list, app.scheduler.slots) |*e, slot| {
        const c = slot.tool.config();
        const st = slot.tool.state();
        e.* = .{ .id = c.id, .name = if (c.name.len > 0) c.name else c.id, .kind = @tagName(c.kind), .running = st == .ready or st == .busy, .capabilities = try engine.capabilityList(a, c), .priority = c.priority, .keep_loaded = c.keep_loaded };
    }
    try out.append(a, .{ .type = "text", .text = try std.json.Stringify.valueAlloc(a, .{ .models = list, .machine = app.cfg.machine }, .{}) });
    return true;
}

fn generateImage(app: *App, a: std.mem.Allocator, host: []const u8, args: Value, out: *std.ArrayList(Content)) ToolError!bool {
    const Args = struct { prompt: []const u8, model: ?[]const u8 = null, size: []const u8 = "1024x1024", n: u32 = 1, seed: ?u64 = null, steps: u32 = 0, negative_prompt: []const u8 = "", inline_images: bool = true, images: []const []const u8 = &.{} };
    const b = try parseArgs(Args, a, args, out);
    const wh = request.parseSize(b.size) orelse return fail(a, out, "size must be WIDTHxHEIGHT", .{});
    const files = try inputFiles(app, a, host, "ref", b.images, out);
    const tool = routes.resolve(app, b.model, if (files.len > 0) .image_edit else .text_to_image) orelse return fail(a, out, "unknown image model (see list_models)", .{});
    const refs = try a.alloc([]const u8, files.len);
    for (refs, files) |*d, f| d.* = f.name;
    const r: request.Request = .{ .image = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .n = b.n, .seed = b.seed orelse routes.randomSeed(app.io), .steps = b.steps, .references = refs } };
    const j = try start(app, a, tool, r, files, out);
    j.done.wait(app.io) catch return fail(a, out, "cancelled", .{});
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
    const Args = struct { prompt: []const u8, model: ?[]const u8 = null, size: []const u8 = "768x448", seconds: u32 = 5, fps: u32 = 24, seed: ?u64 = null, steps: u32 = 0, audio: bool = true, negative_prompt: []const u8 = "", wait_s: u32 = 600, image: ?[]const u8 = null };
    const b = try parseArgs(Args, a, args, out);
    const wh = request.parseSize(b.size) orelse return fail(a, out, "size must be WIDTHxHEIGHT", .{});
    const files = try inputFiles(app, a, host, "first_frame", if (b.image) |x| &.{x} else &.{}, out);
    const tool = routes.resolve(app, b.model, if (files.len > 0) .image_to_video else .text_to_video) orelse return fail(a, out, "unknown video model (see list_models)", .{});
    const r: request.Request = .{ .video = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .seconds = b.seconds, .fps = b.fps, .seed = b.seed orelse routes.randomSeed(app.io), .steps = b.steps, .audio = b.audio, .first_frame = if (files.len > 0) files[0].name else null } };
    const j = try start(app, a, tool, r, files, out);
    try waitFor(app, j, b.wait_s);
    return report(app, a, host, j, out);
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
fn report(app: *App, a: std.mem.Allocator, host: []const u8, j: *jobs.Job, out: *std.ArrayList(Content)) ToolError!bool {
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
        .@"error" = if (j.status == .failed or j.status == .cancelled) @as(?[]const u8, try a.dupe(u8, j.error_message)) else null,
        .next = if (j.finished()) @as(?[]const u8, null) else "call get_job with this id (wait_s up to 3000) until status is completed",
    };
    try out.append(a, .{ .type = "text", .text = try std.json.Stringify.valueAlloc(a, obj, .{ .emit_null_optional_fields = false }) });
    for (urls, 0..) |u, i| {
        const f = j.files[i];
        const mime = if (std.mem.endsWith(u8, f, ".png")) "image/png" else if (std.mem.endsWith(u8, f, ".mp4")) "video/mp4" else "application/octet-stream";
        try out.append(a, .{ .type = "resource_link", .uri = u, .name = f, .mimeType = mime });
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
    .{ .name = "list_models", .title = "List models", .description = "The models on this machine: id, kind (image or video), capabilities (text_to_image, image_edit, text_to_video, image_to_video), priority and keep_loaded, and whether each is loaded now (fast) or will load on first use.", .inputSchema = .{ .text =
    \\{"type":"object","properties":{}}
    } },
    .{ .name = "generate_image", .title = "Generate image", .description = "Text to image, or an image edit when `images` is given. Waits for the result (seconds when the model is loaded, plus its load time when not) and returns the PNG(s) inline and as URLs on this server. Reuse a returned seed to reproduce an image. Edits need a model whose capabilities (list_models) include image_edit.", .inputSchema = .{ .text =
    \\{"type":"object","required":["prompt"],"properties":{
    \\"prompt":{"type":"string","description":"What to draw."},
    \\"model":{"type":"string","description":"An image model id from list_models; default: the first image model."},
    \\"size":{"type":"string","description":"WIDTHxHEIGHT, multiples of 16, 256 to 2048 a side; about 1 megapixel is best (1024x1024, 1360x768, 768x1360).","default":"1024x1024"},
    \\"n":{"type":"integer","minimum":1,"maximum":4,"default":1},
    \\"seed":{"type":"integer","minimum":0,"description":"Omit for a random one."},
    \\"steps":{"type":"integer","minimum":0,"description":"0: the model's default."},
    \\"negative_prompt":{"type":"string"},
    \\"images":{"type":"array","maxItems":5,"items":{"type":"string"},"description":"Input images (PNG, JPEG or WebP, up to 32 MB each) as base64 strings, data URLs or http(s) URLs. Given: the request is an edit of these images, guided by the prompt."},
    \\"inline_images":{"type":"boolean","default":true,"description":"false: URLs only (smaller replies)."}}}
    } },
    .{ .name = "generate_video", .title = "Generate video", .description = "Text to video with audio (MP4), or image to video when `image` is given. Waits up to wait_s seconds; returns the MP4's URL when done, otherwise the job id for get_job.", .inputSchema = .{ .text =
    \\{"type":"object","required":["prompt"],"properties":{
    \\"prompt":{"type":"string","description":"The scene, the motion and the sound."},
    \\"model":{"type":"string","description":"A video model id from list_models; default: the first video model."},
    \\"size":{"type":"string","description":"WIDTHxHEIGHT within the model's limits.","default":"768x448"},
    \\"seconds":{"type":"integer","minimum":1,"maximum":20,"default":5},
    \\"fps":{"type":"integer","default":24},
    \\"seed":{"type":"integer","minimum":0},
    \\"steps":{"type":"integer","minimum":0,"description":"0: the model's default."},
    \\"audio":{"type":"boolean","default":true},
    \\"negative_prompt":{"type":"string"},
    \\"image":{"type":"string","description":"First frame (image to video): a base64 string, data URL or http(s) URL of a PNG, JPEG or WebP up to 32 MB. Needs a model whose capabilities (list_models) include image_to_video."},
    \\"wait_s":{"type":"integer","minimum":0,"maximum":3000,"default":600,"description":"How long to wait before returning the job id instead."}}}
    } },
    .{ .name = "get_job", .title = "Get job", .description = "A job's status (queued, in_progress, completed, failed, cancelled), progress and output URLs. wait_s waits for it to finish.", .inputSchema = .{ .text =
    \\{"type":"object","required":["id"],"properties":{"id":{"type":"string"},"wait_s":{"type":"integer","minimum":0,"maximum":3000,"default":0}}}
    } },
    .{ .name = "cancel_job", .title = "Cancel job", .description = "Cancels a queued or running job.", .inputSchema = .{ .text =
    \\{"type":"object","required":["id"],"properties":{"id":{"type":"string"}}}
    } },
    .{ .name = "release_models", .title = "Release models", .description = "Unloads every model now, returning its memory to the machine (they also unload by themselves when idle).", .inputSchema = .{ .text =
    \\{"type":"object","properties":{}}
    } },
};
