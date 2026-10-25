//! The routes a front calls on another node of a multi-node topology (`tool/node.zig`); like any route not named in
//! `routes.need`, they need the `manage` scope when pairing is required, so the front's token is a paired device's.
//!   GET  /v1/node/status            the node's tools and memory; `x-localrouter-calls: N` says the front has N voice
//!                                   calls open, which guards the voice loop's models here as a local call would
//!   POST /v1/node/jobs              {"tool", "request", "files": [{"name", "data" (base64)}]}: queues the job, -> {"id"}
//!   GET  /v1/node/jobs/{id}         its state, and once done its files and what the tool said (seed, ms, ...)
//!   POST /v1/node/jobs/{id}/cancel  cancels it
//! Outputs come from the ordinary GET /v1/files/{id}/{name}.

const std = @import("std");
const h = @import("http.zig");
const routes = @import("routes.zig");
const pair = @import("pair.zig");
const request = @import("../tool/request.zig");
const inputs = @import("inputs.zig");
const safeName = @import("../tool/node.zig").safeName;

pub fn route(app: *routes.App, req: *h.Request, a: std.mem.Allocator, rest: []const u8, caller: pair.Caller) !void {
    const m = req.head.method;
    if (m == .GET and eql(rest, "status")) return status(app, req, a);
    if (m == .POST and eql(rest, "jobs")) return create(app, req, a, caller);
    if (std.mem.startsWith(u8, rest, "jobs/")) {
        const tail = rest["jobs/".len..];
        if (m == .POST and std.mem.endsWith(u8, tail, "/cancel")) return cancel(app, req, a, tail[0 .. tail.len - "/cancel".len], caller);
        if (m == .GET) return jobState(app, req, a, tail, caller);
    }
    return h.fail(req, a, 404, "invalid_request_error", "no such route");
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn status(app: *routes.App, req: *h.Request, a: std.mem.Allocator) !void {
    var it = req.iterateHeaders();
    while (it.next()) |hd| if (std.ascii.eqlIgnoreCase(hd.name, "x-localrouter-calls")) {
        const n = std.fmt.parseInt(u32, std.mem.trim(u8, hd.value, " "), 10) catch 0;
        app.voice.remoteCalls(app.io, n, 3 * @as(i64, @max(app.cfg.topology.health_interval_ms, 1000)));
    };
    const Entry = struct { id: []const u8, state: []const u8, resident_bytes: u64, last_error: []const u8 };
    const list = try a.alloc(Entry, app.scheduler.slots.len);
    for (list, app.scheduler.slots) |*e, slot| {
        const c = slot.tool.config();
        e.* = .{ .id = c.id, .state = @tagName(slot.tool.state()), .resident_bytes = slot.resident, .last_error = slot.tool.failure().message };
    }
    const mem = @import("../sched/memory.zig").available() orelse 0;
    try h.jsonValue(req, a, .ok, .{ .node = app.cfg.machine, .tools = list, .mem_available_bytes = mem, .reserve_bytes = app.cfg.reserve_bytes, .voice_sessions = app.voice.count(app.io) });
}

fn create(app: *routes.App, req: *h.Request, a: std.mem.Allocator, caller: pair.Caller) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const Body = struct { tool: []const u8, request: request.Request, files: []const struct { name: []const u8, data: []const u8 } = &.{} };
    const b = std.json.parseFromSliceLeaky(Body, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch
        return h.fail(req, a, 400, "invalid_request_error", "the body is not a node job: {\"tool\", \"request\", \"files\"}");
    const i = routes.toolIndex(app, b.tool) orelse return h.fail(req, a, 404, "invalid_request_error", "this node does not serve that model");
    if (app.scheduler.slots[i].tool.config().kind != b.request.kind()) return h.fail(req, a, 400, "invalid_request_error", "the request is not of the model's kind");
    const files = try a.alloc(inputs.File, b.files.len);
    for (files, b.files) |*f, src| {
        if (!safeName(src.name)) return h.fail(req, a, 400, "invalid_request_error", "an input file name is not a plain file name");
        const dec = std.base64.standard.Decoder;
        const n = dec.calcSizeForSlice(src.data) catch return h.fail(req, a, 400, "invalid_request_error", "an input file is not base64");
        const data = try a.alloc(u8, n);
        dec.decode(data, src.data) catch return h.fail(req, a, 400, "invalid_request_error", "an input file is not base64");
        f.* = .{ .name = src.name, .data = data };
    }
    var why: routes.Refusal = .{};
    const j = routes.enqueue(app, i, b.request, files, caller.owner(), &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    try h.jsonValue(req, a, .ok, .{ .id = j.id });
}

fn jobState(app: *routes.App, req: *h.Request, a: std.mem.Allocator, id: []const u8, caller: pair.Caller) !void {
    const j = app.table.get(app.io, id) orelse return h.fail(req, a, 404, "invalid_request_error", "no such job");
    if (!caller.sees(j)) return h.fail(req, a, 404, "invalid_request_error", "no such job");
    try app.table.mutex.lock(app.io);
    defer app.table.mutex.unlock(app.io);
    const done = j.status == .completed;
    try h.jsonValue(req, a, .ok, .{
        .id = j.id,
        .status = @tagName(j.status),
        .http_status = j.http_status,
        .error_type = try a.dupe(u8, j.error_type),
        .error_message = try a.dupe(u8, j.error_message),
        .percent = j.percent,
        .phase = try a.dupe(u8, j.phaseName()),
        .files = if (done) j.files else &.{},
        .seed = j.seed,
        .ms = j.ms,
        .finish_reason = try a.dupe(u8, j.finish_reason),
        .segments = try a.dupe(u8, j.segments),
    });
}

fn cancel(app: *routes.App, req: *h.Request, a: std.mem.Allocator, id: []const u8, caller: pair.Caller) !void {
    const j = app.table.get(app.io, id) orelse return h.fail(req, a, 404, "invalid_request_error", "no such job");
    if (!caller.sees(j)) return h.fail(req, a, 404, "invalid_request_error", "no such job");
    app.scheduler.cancel(j);
    try h.json(req, .ok, "{\"ok\":true}");
}
