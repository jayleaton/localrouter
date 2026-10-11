//! The routes: OpenAI's /v1/models, /v1/images/generations, /v1/images/edits and /v1/videos (with a model gateway's
//! `kind`, `running` and `machine` extras), output files, /v1/tools for residency, and /mcp (`mcp.zig`, the same
//! jobs as MCP tools). Every request becomes a scheduler job. Edits and image to video take their input images as
//! multipart files or JSON base64 / URLs (`inputs.zig`); they are saved in the job directory and named in the request.

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const request = @import("../tool/request.zig");
const jobs = @import("../sched/jobs.zig");
const sched = @import("../sched/scheduler.zig");
const Config = @import("../config.zig").Config;
const mcp = @import("mcp.zig");
const inputs = @import("inputs.zig");
const engine = @import("../engine/engine.zig");

pub const App = struct {
    gpa: std.mem.Allocator,
    io: Io,
    cfg: *const Config,
    scheduler: *sched.Scheduler,
    table: *jobs.Table,
    jobs_dir: []const u8,
    self_addr: Io.net.IpAddress = .{ .ip4 = .loopback(0) }, // where this daemon listens, for fetching its own output URLs

    /// What URL inputs may reach for a request that arrived with this Host header.
    pub fn policy(app: *const App, host: []const u8) inputs.Policy {
        return .{ .allow_private = app.cfg.allow_private_urls, .host_header = host, .self_addr = app.self_addr };
    }
};

pub fn handle(app: *App, req: *h.Request) !void {
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const p = h.path(req);
    const m = req.head.method;
    const host = try a.dupe(u8, h.host(req)); // headers are gone once the body is read
    const ctype = try a.dupe(u8, req.head.content_type orelse "");
    if (m == .GET and std.mem.eql(u8, p, "/health")) return h.json(req, .ok, "{\"ok\":true}");
    if (eql(p, "/mcp")) return mcp.handle(app, req, a, host);
    if (m == .GET and (eql(p, "/v1/models") or eql(p, "/models"))) return models(app, req, a);
    if (m == .POST and eql(p, "/v1/images/generations")) return images(app, req, a, host, ctype, false);
    if (m == .POST and eql(p, "/v1/images/edits")) return images(app, req, a, host, ctype, true);
    if (m == .POST and eql(p, "/v1/videos")) return videoCreate(app, req, a, host, ctype);
    if (std.mem.startsWith(u8, p, "/v1/videos/")) return videoRoute(app, req, a, p["/v1/videos/".len..]);
    if (m == .GET and std.mem.startsWith(u8, p, "/v1/files/")) return file(app, req, a, p["/v1/files/".len..]);
    if (m == .GET and eql(p, "/v1/tools")) return tools(app, req, a);
    if (m == .POST and std.mem.startsWith(u8, p, "/v1/tools/")) return toolControl(app, req, a, p["/v1/tools/".len..]);
    return h.fail(req, a, 404, "invalid_request_error", "no such route");
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

// ---------------------------------------------------------------- models and tools

fn models(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const Entry = struct { id: []const u8, object: []const u8 = "model", created: i64 = 0, owned_by: []const u8 = "localrouter", name: []const u8, kind: []const u8, machine: []const u8, running: bool, resident_bytes: u64, capabilities: []const engine.Capability, default_for: []const engine.Capability };
    const list = try a.alloc(Entry, app.scheduler.slots.len);
    for (list, app.scheduler.slots, 0..) |*e, slot, i| {
        const c = slot.tool.config();
        const st = slot.tool.state();
        e.* = .{ .id = c.id, .name = if (c.name.len > 0) c.name else c.id, .kind = @tagName(c.kind), .machine = app.cfg.machine, .running = st == .ready or st == .busy, .resident_bytes = slot.resident, .capabilities = try engine.capabilityList(a, c), .default_for = try defaultFor(app, a, i) };
    }
    try h.jsonValue(req, a, .ok, .{ .object = "list", .data = list });
}

fn tools(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const Entry = struct { id: []const u8, kind: []const u8, state: []const u8, resident_bytes: u64, idle_ttl_s: u32, priority: i32, keep_loaded: bool, capabilities: []const engine.Capability, last_error: []const u8 };
    const list = try a.alloc(Entry, app.scheduler.slots.len);
    for (list, app.scheduler.slots) |*e, slot| {
        const c = slot.tool.config();
        e.* = .{ .id = c.id, .kind = @tagName(c.kind), .state = @tagName(slot.tool.state()), .resident_bytes = slot.resident, .idle_ttl_s = c.idle_ttl_s, .priority = c.priority, .keep_loaded = c.keep_loaded, .capabilities = try engine.capabilityList(a, c), .last_error = slot.tool.failure().message };
    }
    const mem = @import("../sched/memory.zig").available() orelse 0;
    try h.jsonValue(req, a, .ok, .{ .object = "list", .data = list, .mem_available_bytes = mem, .reserve_bytes = app.cfg.reserve_bytes, .budget_bytes = app.cfg.budget_bytes });
}

/// POST /v1/tools/{id}/load, /v1/tools/{id}/unload, /v1/tools/release.
fn toolControl(app: *App, req: *h.Request, a: std.mem.Allocator, rest: []const u8) !void {
    var c: sched.Control = .{ .op = .release };
    if (!eql(rest, "release")) {
        const slash = std.mem.lastIndexOfScalar(u8, rest, '/') orelse return h.fail(req, a, 404, "invalid_request_error", "no such route");
        const verb = rest[slash + 1 ..];
        c.op = if (eql(verb, "load")) .load else if (eql(verb, "unload")) .unload else return h.fail(req, a, 404, "invalid_request_error", "no such route");
        c.tool = toolIndex(app, rest[0..slash]) orelse return h.fail(req, a, 404, "invalid_request_error", "unknown model");
    }
    app.scheduler.control(&c) catch |err| return h.fail(req, a, 503, "server_error", @errorName(err));
    if (c.failed) return h.fail(req, a, 503, "server_error", c.message);
    return tools(app, req, a);
}

fn toolIndex(app: *App, id: []const u8) ?usize {
    for (app.scheduler.slots, 0..) |slot, i| if (eql(slot.tool.config().id, id)) return i;
    return null;
}

/// The model for a request that needs `cap`: the one named, else the configured default for `cap` (`Config.defaults`),
/// else the first tool that can do it, else the first of its kind (so `enqueue`'s refusal names a model). Null with
/// `why` set when the named model is unknown (404) or of another kind (400), or no model of the kind exists (404).
pub fn resolve(app: *App, model: ?[]const u8, cap: request.Capability, why: *Refusal) ?usize {
    const kind = @tagName(cap.kind());
    if (model orelse app.cfg.defaults.get(cap)) |m| {
        const i = toolIndex(app, m) orelse {
            why.* = .{ .status = 404 };
            why.message = std.fmt.bufPrint(why.buf[0..384], "unknown model '{s}'; models that can do {s}: {s}", .{ m, @tagName(cap), able(app, cap, &why.buf) }) catch "unknown model";
            return null;
        };
        const c = app.scheduler.slots[i].tool.config();
        if (c.kind != cap.kind()) {
            why.message = std.fmt.bufPrint(why.buf[0..384], "model '{s}' is {s} model, not {s}; models that can do {s}: {s}", .{ m, article(c.kind), kind, @tagName(cap), able(app, cap, &why.buf) }) catch "the model is of another kind";
            return null;
        }
        return i;
    }
    var first: ?usize = null;
    for (app.scheduler.slots, 0..) |slot, i| {
        const c = slot.tool.config();
        if (c.kind != cap.kind()) continue;
        if (engine.supports(c, cap)) return i;
        if (first == null) first = i;
    }
    if (first == null) {
        why.* = .{ .status = 404 };
        why.message = std.fmt.bufPrint(why.buf[0..384], "no {s} model is configured", .{kind}) catch "no model of this kind";
    }
    return first;
}

fn article(k: request.Kind) []const u8 {
    return switch (k) {
        .image => "an image",
        .video => "a video",
    };
}

/// "a, b": the models that can do `cap`, in the tail of `buf` (the message takes the head).
fn able(app: *App, cap: request.Capability, buf: *[512]u8) []const u8 {
    var w: Io.Writer = .fixed(buf[384..]);
    for (app.scheduler.slots) |slot| {
        const c = slot.tool.config();
        if (engine.supports(c, cap)) w.print("{s}{s}", .{ if (w.end > 0) ", " else "", c.id }) catch break;
    }
    return if (w.end > 0) w.buffered() else "none";
}

/// The capabilities for which tool `i` serves requests that name no model.
pub fn defaultFor(app: *App, a: std.mem.Allocator, i: usize) ![]const request.Capability {
    var list: std.ArrayList(request.Capability) = .empty;
    for (std.enums.values(request.Capability)) |cap| {
        var why: Refusal = .{};
        const t = resolve(app, null, cap, &why) orelse continue;
        if (t == i and engine.supports(app.scheduler.slots[i].tool.config(), cap)) try list.append(a, cap);
    }
    return list.items;
}

// ---------------------------------------------------------------- images

const ImageBody = struct {
    model: ?[]const u8 = null,
    prompt: []const u8 = "",
    negative_prompt: []const u8 = "",
    n: u32 = 1,
    size: []const u8 = "1024x1024",
    response_format: []const u8 = "b64_json",
    seed: ?u64 = null,
    steps: u32 = 0,
    guidance: f32 = 0,
};

/// Parses a JSON or multipart body into `T` and collects the input images found under `keys` (multipart parts or JSON
/// members: a base64 string, a data URL, an http(s) URL, or a list of them), as bytes.
pub fn parseBody(comptime T: type, app: *App, a: std.mem.Allocator, host: []const u8, text: []const u8, ctype: []const u8, comptime keys: []const []const u8, raws: *std.ArrayList([]const u8), why: *[]const u8) inputs.Invalid!T {
    var specs: std.ArrayList([]const u8) = .empty;
    var b: T = undefined;
    if (inputs.Form.boundary(ctype)) |bound| {
        const form = inputs.Form.parse(a, text, bound) catch {
            why.* = "the multipart body is malformed";
            return error.Invalid;
        };
        b = try form.fill(T, why);
        for (form.parts) |p| inline for (keys) |k| if (eql(p.name, k)) {
            const list = if (p.filename != null) raws else &specs;
            list.append(a, p.data) catch return error.Invalid;
        };
    } else {
        b = std.json.parseFromSliceLeaky(T, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch {
            why.* = "the body is not a valid request";
            return error.Invalid;
        };
        const v = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.Invalid;
        if (v != .object) return error.Invalid;
        inline for (keys) |k| try inputs.specs(a, v.object.get(k), &specs, why);
    }
    for (specs.items) |sp| raws.append(a, try inputs.resolve(app.io, app.gpa, a, sp, app.policy(host), why)) catch return error.Invalid;
    return b;
}

fn images(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, ctype: []const u8, edit: bool) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    var why: []const u8 = "the body is not a valid image request";
    var raws: std.ArrayList([]const u8) = .empty;
    const b = (if (edit) parseBody(ImageBody, app, a, host, text, ctype, &.{ "image", "image[]", "images" }, &raws, &why) else parseBody(ImageBody, app, a, host, text, ctype, &.{}, &raws, &why)) catch
        return h.fail(req, a, 400, "invalid_request_error", why);
    const files = inputs.validate(a, "ref", raws.items, (request.Limits{}).max_references, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    if (edit and files.len == 0) return h.fail(req, a, 400, "invalid_request_error", "image is required: send image (or image[]) as files, base64 or URLs");
    const wh = request.parseSize(if (eql(b.size, "auto")) "1024x1024" else b.size) orelse return h.fail(req, a, 400, "invalid_request_error", "size must be WIDTHxHEIGHT");
    const refs = try a.alloc([]const u8, files.len);
    for (refs, files) |*d, f| d.* = f.name;
    var why_model: Refusal = .{};
    const tool = resolve(app, b.model, if (edit) .image_edit else .text_to_image, &why_model) orelse return h.fail(req, a, why_model.status, why_model.typ, why_model.message);
    const r: request.Request = .{ .image = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .n = b.n, .seed = b.seed orelse randomSeed(app.io), .steps = b.steps, .guidance = b.guidance, .references = refs } };
    const j = (try submit(app, req, a, tool, r, files)) orelse return;
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const Item = struct { b64_json: ?[]const u8 = null, url: ?[]const u8 = null };
    const items = try a.alloc(Item, j.files.len);
    const as_url = eql(b.response_format, "url");
    for (items, j.files) |*it, f| {
        if (as_url) {
            it.* = .{ .url = try std.fmt.allocPrint(a, "http://{s}/v1/files/{s}/{s}", .{ host, j.id, f }) };
        } else {
            const path = try std.fs.path.join(a, &.{ j.dir, f });
            const png = try Io.Dir.cwd().readFileAlloc(app.io, path, a, .limited(h.max_body));
            const enc = std.base64.standard.Encoder;
            it.* = .{ .b64_json = enc.encode(try a.alloc(u8, enc.calcSize(png.len)), png) };
        }
    }
    try h.jsonValue(req, a, .ok, .{ .created = j.created_at, .data = items, .model = app.scheduler.slots[tool].tool.config().id, .seed = j.seed });
}

/// Why `enqueue` refused: the HTTP status, OpenAI's error type and a message (which may point into `buf`).
pub const Refusal = struct { status: u16 = 400, typ: []const u8 = "invalid_request_error", message: []const u8 = "", buf: [512]u8 = undefined };

/// Validates and queues `r` for `tool` (the HTTP routes and the MCP tools share it), saving `files` in its job
/// directory first; null with `why` set on refusal. A model without the capability the request needs is refused.
pub fn enqueue(app: *App, tool: usize, r: request.Request, files: []const inputs.File, why: *Refusal) ?*jobs.Job {
    const cfg = app.scheduler.slots[tool].tool.config();
    if (!engine.supports(cfg, r.capability())) {
        why.message = std.fmt.bufPrint(why.buf[0..384], "model '{s}' cannot do {s} (its capabilities: {s}); pick a model that lists {s} in list_models", .{ cfg.id, @tagName(r.capability()), capabilityNames(&why.buf, cfg), @tagName(r.capability()) }) catch "the model lacks a capability this request needs";
        return null;
    }
    request.validate(r, .{}, &why.message) catch return null;
    var w: Io.Writer = .fixed(why.buf[0..384]);
    if (!engine.check(cfg, &r, &w)) {
        why.message = if (w.end > 0) w.buffered() else "the model cannot serve this request";
        return null;
    }
    const j = app.table.create(app.io, app.jobs_dir, tool, r, sched.nowUnix(app.io)) catch |err| {
        why.* = .{ .status = 500, .typ = "server_error", .message = @errorName(err) };
        return null;
    };
    if (files.len > 0) {
        var dir = Io.Dir.cwd().openDir(app.io, j.dir, .{}) catch return abandon(app, j, why, 500, "could not open the job directory");
        defer dir.close(app.io);
        for (files) |f| dir.writeFile(app.io, .{ .sub_path = f.name, .data = f.data }) catch return abandon(app, j, why, 500, "could not save an input image");
    }
    app.scheduler.submit(j) catch |err| return abandon(app, j, why, 503, if (err == error.QueueFull) "the queue is full" else @errorName(err));
    return j;
}

/// "text_to_image, image_edit" for the refusal message, written in the tail of `buf` (the message takes the head).
fn capabilityNames(buf: *[512]u8, cfg: *const @import("../config.zig").ToolConfig) []const u8 {
    var w: Io.Writer = .fixed(buf[384..]);
    for (std.enums.values(request.Capability)) |c| if (engine.supports(cfg, c)) w.print("{s}{s}", .{ if (w.end > 0) ", " else "", @tagName(c) }) catch break;
    return w.buffered();
}

/// Marks a job that will never run as failed (so it expires) and fills the refusal.
fn abandon(app: *App, j: *jobs.Job, why: *Refusal, status: u16, msg: []const u8) ?*jobs.Job {
    app.table.mutex.lockUncancelable(app.io);
    j.status = .failed;
    j.completed_at = sched.nowUnix(app.io);
    app.table.mutex.unlock(app.io);
    why.* = .{ .status = status, .typ = "server_error", .message = msg };
    return null;
}

/// `enqueue` for the HTTP routes: on refusal the error response is already sent and the result is null.
fn submit(app: *App, req: *h.Request, a: std.mem.Allocator, tool: usize, r: request.Request, files: []const inputs.File) !?*jobs.Job {
    var why: Refusal = .{};
    const j = enqueue(app, tool, r, files, &why) orelse {
        try h.fail(req, a, why.status, why.typ, why.message);
        return null;
    };
    return j;
}

pub fn randomSeed(io: Io) u64 {
    var b: [8]u8 = undefined;
    io.random(&b);
    return std.mem.readInt(u64, &b, .little) >> 11; // below 2^53: exact in every JSON consumer (JavaScript too)
}

// ---------------------------------------------------------------- videos (OpenAI Videos API)

const VideoBody = struct {
    model: ?[]const u8 = null,
    prompt: []const u8 = "",
    negative_prompt: []const u8 = "",
    size: []const u8 = "1344x768",
    seconds: []const u8 = "5", // OpenAI sends seconds as a string
    fps: u32 = 24,
    seed: ?u64 = null,
    steps: u32 = 0,
    guidance: f32 = 0,
    audio: bool = true,
};

/// `input_reference` (OpenAI's name) is the first frame: a multipart file, or JSON base64, a data URL or a URL.
fn videoCreate(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, ctype: []const u8) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    var why: []const u8 = "the body is not a valid video request";
    var raws: std.ArrayList([]const u8) = .empty;
    const b = parseBody(VideoBody, app, a, host, text, ctype, &.{"input_reference"}, &raws, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    const files = inputs.validate(a, "first_frame", raws.items, 1, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    const wh = request.parseSize(b.size) orelse return h.fail(req, a, 400, "invalid_request_error", "size must be WIDTHxHEIGHT");
    const seconds = std.fmt.parseInt(u32, b.seconds, 10) catch return h.fail(req, a, 400, "invalid_request_error", "seconds must be a whole number");
    var why_model: Refusal = .{};
    const tool = resolve(app, b.model, if (files.len > 0) .image_to_video else .text_to_video, &why_model) orelse return h.fail(req, a, why_model.status, why_model.typ, why_model.message);
    const r: request.Request = .{ .video = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .seconds = seconds, .fps = b.fps, .seed = b.seed orelse randomSeed(app.io), .steps = b.steps, .guidance = b.guidance, .audio = b.audio, .first_frame = if (files.len > 0) files[0].name else null } };
    const j = (try submit(app, req, a, tool, r, files)) orelse return;
    try videoObject(app, req, a, j);
}

/// GET {id}, GET {id}/content, DELETE {id}.
fn videoRoute(app: *App, req: *h.Request, a: std.mem.Allocator, rest: []const u8) !void {
    const content = std.mem.endsWith(u8, rest, "/content");
    const id = if (content) rest[0 .. rest.len - "/content".len] else rest;
    const j = app.table.get(app.io, id) orelse return h.fail(req, a, 404, "invalid_request_error", "no such video");
    if (j.kind() != .video) return h.fail(req, a, 404, "invalid_request_error", "no such video");
    switch (req.head.method) {
        .DELETE => {
            app.scheduler.cancel(j);
            return videoObject(app, req, a, j);
        },
        .GET => {
            if (!content) return videoObject(app, req, a, j);
            if (j.status != .completed or j.files.len == 0) return h.fail(req, a, 409, "invalid_request_error", "the video is not ready");
            const path = try std.fs.path.join(a, &.{ j.dir, j.files[0] });
            const data = try Io.Dir.cwd().readFileAlloc(app.io, path, a, .limited(1 << 31));
            return h.bytes(req, "video/mp4", data);
        },
        else => return h.fail(req, a, 405, "invalid_request_error", "method not allowed"),
    }
}

fn videoObject(app: *App, req: *h.Request, a: std.mem.Allocator, j: *jobs.Job) !void {
    const Err = struct { code: []const u8, message: []const u8 };
    try app.table.mutex.lock(app.io);
    const v = j.request.video;
    const obj = .{
        .id = j.id,
        .object = "video",
        .model = app.scheduler.slots[j.tool].tool.config().id,
        .status = @tagName(j.status),
        .progress = j.percent,
        .phase = try a.dupe(u8, j.phaseName()),
        .created_at = j.created_at,
        .completed_at = if (j.finished()) @as(?i64, j.completed_at) else null,
        .size = try std.fmt.allocPrint(a, "{d}x{d}", .{ v.width, v.height }),
        .seconds = try std.fmt.allocPrint(a, "{d}", .{v.seconds}),
        .seed = v.seed,
        .@"error" = if (j.status == .failed or j.status == .cancelled) @as(?Err, .{ .code = j.error_type, .message = try a.dupe(u8, j.error_message) }) else null,
    };
    app.table.mutex.unlock(app.io);
    try h.jsonValue(req, a, .ok, obj);
}

// ---------------------------------------------------------------- files

/// GET /v1/files/{job}/{name}: an output file (`response_format: url`).
fn file(app: *App, req: *h.Request, a: std.mem.Allocator, rest: []const u8) !void {
    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return h.fail(req, a, 404, "invalid_request_error", "no such file");
    const j = app.table.get(app.io, rest[0..slash]) orelse return h.fail(req, a, 404, "invalid_request_error", "no such file");
    const name = rest[slash + 1 ..];
    for (j.files) |f| {
        if (!eql(f, name)) continue;
        const data = try Io.Dir.cwd().readFileAlloc(app.io, try std.fs.path.join(a, &.{ j.dir, f }), a, .limited(1 << 31));
        const ct = if (std.mem.endsWith(u8, f, ".png")) "image/png" else if (std.mem.endsWith(u8, f, ".mp4")) "video/mp4" else "application/octet-stream";
        return h.bytes(req, ct, data);
    }
    return h.fail(req, a, 404, "invalid_request_error", "no such file");
}
