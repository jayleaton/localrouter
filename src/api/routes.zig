//! The routes: OpenAI's /v1/models, /v1/images/generations, /v1/images/edits, /v1/videos, /v1/audio (`speech.zig`) and
//! /v1/chat/completions (`chat.zig`), /v1/responses (`responses.zig`, the Responses API over the chat path), /v1/systemone (`decide.zig`, typed decisions)
//! (with a model gateway's `kind`, `running` and `machine` extras), output files, /v1/jobs/{id} (any job's state and outputs),
//! /v1/tools for residency, /v1/pair (device pairing, `pair.zig`), /v1/capabilities (the harness contract,
//! `capabilities.zig`), and /mcp
//! (`mcp.zig`, the same jobs as MCP tools). Every request becomes a scheduler job. Edits and image to video take their input images as
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
const speech = @import("speech.zig");
const chat = @import("chat.zig");
const responses = @import("responses.zig");
const decide = @import("decide.zig");
const realtime = @import("realtime.zig");
const pair = @import("pair.zig");
const desktop = @import("desktop.zig");
const voices = @import("voices.zig");
const Need = @import("../pair.zig").Need;

pub const App = struct {
    gpa: std.mem.Allocator,
    io: Io,
    cfg: *const Config,
    scheduler: *sched.Scheduler,
    table: *jobs.Table,
    jobs_dir: []const u8,
    self_addr: Io.net.IpAddress = .{ .ip4 = .loopback(0) }, // where this daemon listens, for fetching its own output URLs
    decode_mutex: Io.Mutex = .init, // audio uploads decode one at a time (`speech.startTranscription`)
    voice: realtime.Registry = .{}, // open voice sessions, closed at shutdown
    pair_guard: @import("../pair.zig").Guard = .{}, // wrong pairing codes (`pair.redeem`)
    computer: ?*desktop.Service = null, // computer control (desktop.zig); null when the config switches it off
    cloned: @import("../voices.zig").Store = .{}, // cloned voices (`voices.zig`, `api/voices.zig`)
    assistant: @import("../assistant.zig").Store = .{}, // the assistant's name (`assistant.zig`, `api/assistant.zig`)
    plan: ?*const @import("../topology.zig").Plan = null, // where each tool runs (a multi-node topology); null: here

    /// Why tool `c` is refused on this machine until it is qualified here (`engine.unqualified`), or null; a tool that
    /// runs on another node now is that node's to check.
    pub fn unqualified(app: *App, a: std.mem.Allocator, c: *const @import("../config.zig").ToolConfig, buf: []u8) ?[]const u8 {
        if (toolIndex(app, c.id)) |i| if (!app.scheduler.slots[i].tool.local()) return null;
        return engine.unqualified(app.io, a, c, buf);
    }

    /// The node that runs tool `id` ("" in a single topology): its own node for a tool served here, else its host's.
    pub fn nodeOf(app: *const App, id: []const u8) []const u8 {
        const p = app.plan orelse return "";
        if (!p.multi) return "";
        return if (p.hostOf(id)) |n| n.id else p.nodes[p.self].id;
    }

    /// What URL inputs may reach for a request that arrived with this Host header.
    pub fn policy(app: *const App, host: []const u8) inputs.Policy {
        return .{ .allow_private = app.cfg.allow_private_urls, .host_header = host, .self_addr = app.self_addr };
    }
};

pub fn handle(app: *App, conn: *const h.Conn, req: *h.Request) !void {
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const p = h.path(req);
    const m = req.head.method;
    const host = try a.dupe(u8, h.host(req)); // headers are gone once the body is read
    const ctype = try a.dupe(u8, req.head.content_type orelse "");
    const scheme = h.scheme(req); // a static string: it outlives the headers
    // With pairing required, every route but the open ones needs its scope: refused before the body or the upgrade.
    const caller: pair.Caller = if (need(m, p)) |n| (try pair.admit(app, req, a, n)) orelse return else .{};
    if (m == .GET and std.mem.eql(u8, p, "/health")) return h.json(req, .ok, "{\"ok\":true}");
    if (eql(p, "/mcp")) return mcp.handle(app, req, a, host, caller);
    if (m == .GET and eql(p, "/v1/realtime")) return realtime.serve(app, conn, req, a, caller);
    if (m == .GET and eql(p, "/v1/desktop")) return desktop.serve(app, conn, req, a);
    if (std.mem.startsWith(u8, p, "/v1/computer/")) return desktop.route(app, req, a, p["/v1/computer/".len..]);
    if (m == .GET and (eql(p, "/voice") or eql(p, "/voice/"))) return realtime.page(app, conn, req, a);
    if (m == .POST and eql(p, "/v1/pair")) return pair.redeem(app, req, a);
    if (m == .GET and eql(p, "/v1/pair")) return pair.status(app, req, a);
    if (m == .GET and eql(p, "/")) return req.respond("", .{ .status = .found, .extra_headers = &.{.{ .name = "location", .value = "/voice" }} });
    if (m == .GET and (eql(p, "/v1/models") or eql(p, "/models"))) return models(app, req, a);
    if (m == .GET and eql(p, "/v1/capabilities")) return @import("capabilities.zig").serve(app, req, a);
    if (m == .POST and eql(p, "/v1/images/generations")) return images(app, req, a, host, scheme, ctype, false, caller);
    if (m == .POST and eql(p, "/v1/images/edits")) return images(app, req, a, host, scheme, ctype, true, caller);
    if (m == .POST and eql(p, "/v1/videos")) return videoCreate(app, req, a, host, ctype, caller);
    if (std.mem.startsWith(u8, p, "/v1/videos/")) return videoRoute(app, req, a, p["/v1/videos/".len..], caller);
    if (m == .POST and eql(p, "/v1/audio/speech")) return speech.speech(app, req, a, caller.owner());
    if (m == .POST and eql(p, "/v1/audio/transcriptions")) return speech.transcriptions(app, req, a, host, ctype, caller.owner());
    if (m == .POST and eql(p, "/v1/chat/completions")) return chat.completions(app, req, a, caller.owner());
    if (m == .POST and eql(p, "/v1/responses")) return responses.create(app, req, a, caller.owner());
    if (std.mem.startsWith(u8, p, "/v1/responses/")) return responses.stored(req, a);
    if (m == .POST and eql(p, "/v1/systemone")) return decide.systemOne(app, req, a, caller.owner());
    if (eql(p, "/v1/audio/translations")) return h.fail(req, a, 404, "invalid_request_error", "translation is not offered: transcribe with /v1/audio/transcriptions, which keeps the spoken language");
    if (m == .GET and std.mem.startsWith(u8, p, "/v1/files/")) return file(app, req, a, p["/v1/files/".len..], caller);
    if (m == .GET and std.mem.startsWith(u8, p, "/v1/jobs/")) return jobObject(app, req, a, p["/v1/jobs/".len..], caller);
    if (m == .GET and eql(p, "/v1/skills/model-manager")) return h.jsonValue(req, a, .ok, .{ .enabled = app.cfg.voice.tools and app.cfg.voice.manage_models, .text = try @import("voice_local.zig").modelSkill(app, a) });
    if (eql(p, "/v1/voices")) return voices.route(app, req, a, host, ctype, "", caller);
    if (std.mem.startsWith(u8, p, "/v1/voices/")) return voices.route(app, req, a, host, ctype, p["/v1/voices/".len..], caller);
    if (eql(p, "/v1/assistant")) return @import("assistant.zig").route(app, req, a);
    if (m == .GET and eql(p, "/v1/tools")) return tools(app, req, a);
    if (m == .POST and std.mem.startsWith(u8, p, "/v1/tools/")) return toolControl(app, req, a, p["/v1/tools/".len..]);
    if (std.mem.startsWith(u8, p, "/v1/node/")) return @import("node.zig").route(app, req, a, p["/v1/node/".len..], caller);
    return h.fail(req, a, 404, "invalid_request_error", "no such route");
}

/// What a route needs from the caller's device when pairing is required (`pair.admit`); null only for the routes anyone
/// reaches: /health, pairing itself, the voice page and the redirect to it, and computer control (which checks its own
/// scopes always: computer/gate.zig). The job routes need only a paired device: a job is then its owner's alone
/// (`Caller.sees`). Anything not named here (another method, a new route, no route at all) needs `manage`, the
/// strictest: a route is never open by omission.
pub fn need(m: std.http.Method, p: []const u8) ?Need {
    const starts = std.mem.startsWith;
    if (eql(p, "/health") or eql(p, "/v1/pair") or eql(p, "/voice") or eql(p, "/voice/") or eql(p, "/")) return null;
    if (m == .GET and eql(p, "/v1/capabilities")) return null; // how to get in has to be readable before getting in
    if (eql(p, "/v1/desktop") or starts(u8, p, "/v1/computer/")) return null;
    if (eql(p, "/mcp")) return .mcp;
    if (eql(p, "/v1/realtime")) return .realtime;
    if (m == .GET and (eql(p, "/v1/models") or eql(p, "/models") or eql(p, "/v1/tools") or eql(p, "/v1/skills/model-manager"))) return .read;
    if (m == .GET and (eql(p, "/v1/voices") or starts(u8, p, "/v1/voices/"))) return .read; // creating, renaming and deleting a voice: manage
    if (m == .GET and eql(p, "/v1/assistant")) return .paired; // every client shows the name; renaming it: manage
    if (eql(p, "/v1/chat/completions") or eql(p, "/v1/responses") or starts(u8, p, "/v1/responses/")) return .inference;
    if (eql(p, "/v1/systemone")) return .decisions;
    if (eql(p, "/v1/images/generations") or eql(p, "/v1/images/edits") or eql(p, "/v1/videos") or starts(u8, p, "/v1/audio/")) return .media;
    if (starts(u8, p, "/v1/videos/") or starts(u8, p, "/v1/jobs/") or starts(u8, p, "/v1/files/")) return .owned;
    return .manage; // POST /v1/tools/..., and whatever is not named above
}

test "every route class needs its scope; only the open routes need none; an unnamed route needs manage" {
    const t = std.testing;
    const open = [_][]const u8{ "/health", "/v1/pair", "/voice", "/voice/", "/", "/v1/desktop", "/v1/computer/approvals/x", "/v1/capabilities" };
    for (open) |p| try t.expect(need(.GET, p) == null);
    const Case = struct { std.http.Method, []const u8, Need };
    const cases = [_]Case{
        .{ .POST, "/mcp", .mcp },                       .{ .GET, "/v1/realtime", .realtime },
        .{ .GET, "/v1/models", .read },                 .{ .GET, "/models", .read },
        .{ .GET, "/v1/tools", .read },                  .{ .GET, "/v1/skills/model-manager", .read },
        .{ .POST, "/v1/tools/release", .manage },       .{ .POST, "/v1/tools/x/load", .manage },
        .{ .POST, "/v1/tools/x/unload", .manage },      .{ .POST, "/v1/chat/completions", .inference },
        .{ .POST, "/v1/tools/x/settings", .manage },    .{ .POST, "/v1/responses", .inference },
        .{ .GET, "/v1/responses/resp_1", .inference },  .{ .POST, "/v1/systemone", .decisions },
        .{ .POST, "/v1/images/generations", .media },   .{ .POST, "/v1/images/edits", .media },
        .{ .POST, "/v1/videos", .media },               .{ .POST, "/v1/audio/speech", .media },
        .{ .POST, "/v1/audio/transcriptions", .media }, .{ .GET, "/v1/videos/v_1/content", .owned },
        .{ .DELETE, "/v1/videos/v_1", .owned },         .{ .GET, "/v1/jobs/img_1", .owned },
        .{ .GET, "/v1/files/img_1/0.png", .owned },     .{ .POST, "/v1/models", .manage },
        .{ .GET, "/v1/something-new", .manage },        .{ .GET, "/v1/voices", .read },
        .{ .GET, "/v1/voices/JJ", .read },              .{ .POST, "/v1/voices", .manage },
        .{ .PATCH, "/v1/voices/draft-1", .manage },     .{ .DELETE, "/v1/voices/JJ", .manage },
        .{ .GET, "/v1/assistant", .paired },            .{ .PATCH, "/v1/assistant", .manage },
        .{ .GET, "/v1/node/status", .manage },          .{ .POST, "/v1/node/jobs", .manage },
    };
    for (cases) |c| try t.expectEqual(@as(?Need, c[2]), need(c[0], c[1]));
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

// ---------------------------------------------------------------- models and tools

fn models(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const Entry = struct { id: []const u8, object: []const u8 = "model", created: i64 = 0, owned_by: []const u8 = "localrouter", name: []const u8, kind: []const u8, machine: []const u8, running: bool, resident_bytes: u64, capabilities: []const engine.Capability, default_for: []const engine.Capability, voices: ?[]const std.json.Value = null, default_voice: ?[]const u8 = null, presets: ?[]const std.json.Value = null, speed: ?[2]f32 = null, max_input_chars: ?u64 = null, stream: ?bool = null, references: ?bool = null, instructions: ?bool = null, max_audio_s: ?u64 = null, languages: ?[]const std.json.Value = null, qualified: ?bool = null, license: ?@import("../config.zig").License = null, min_thinking_tokens: ?u32 = null, node: ?[]const u8 = null };
    const list = try a.alloc(Entry, app.scheduler.slots.len);
    for (list, app.scheduler.slots, 0..) |*e, slot, i| {
        const c = slot.tool.config();
        const st = slot.tool.state();
        e.* = .{ .id = c.id, .name = if (c.name.len > 0) c.name else c.id, .kind = @tagName(c.kind), .machine = app.cfg.machine, .running = st == .ready or st == .busy, .resident_bytes = slot.resident, .capabilities = try engine.capabilityList(a, c), .default_for = try defaultFor(app, a, i), .license = engine.license(c) };
        var info = engine.audioInfo(c);
        if (c.kind == .speech) info.voices = try speechVoices(app, a, c);
        inline for (@typeInfo(engine.AudioInfo).@"struct".field_names) |f| @field(e.*, f) = @field(info, f);
        var buf: [512]u8 = undefined;
        if (c.qualification.len > 0 and slot.tool.local()) e.qualified = engine.unqualified(app.io, a, c, &buf) == null; // listed either way; a remote tool's node lists its own
        const n = app.nodeOf(c.id);
        if (n.len > 0) e.node = n;
        if (c.kind == .chat) e.min_thinking_tokens = c.min_thinking_tokens; // a smaller limit answers without thinking
    }
    try h.jsonValue(req, a, .ok, .{ .object = "list", .data = list });
}

/// A speech model's voices as listed: its configured `voices`, then the voices cloned for it (null: it lists none and
/// has none cloned).
pub fn speechVoices(app: *App, a: std.mem.Allocator, c: *const @import("../config.zig").ToolConfig) !?[]const std.json.Value {
    const own = engine.voices(c);
    const cloned = try app.cloned.names(app.io, a, c.id);
    if (cloned.len == 0) return own;
    var out: std.ArrayList(std.json.Value) = .empty;
    try out.appendSlice(a, own orelse &.{});
    for (cloned) |n| try out.append(a, .{ .string = n });
    return out.items;
}

/// Whether speech model `c` has `voice`: configured, or cloned for it (a draft included).
pub fn hasVoice(app: *App, c: *const @import("../config.zig").ToolConfig, voice: []const u8) bool {
    return engine.hasVoice(c, voice) or app.cloned.has(app.io, c.id, voice);
}

/// One tool as `GET /v1/tools` lists it. `residency` and `idle_ttl_s` are what applies now (set through
/// `POST /v1/tools/{id}/settings`, else the config's: `residency_default`, `idle_ttl_s_default`); `keep_loaded` is
/// `residency` "always". `last_load_ms` (how long its last load took) and `loaded_since` (unix seconds) are null until
/// it loads, and `loaded_since` while it is not loaded.
pub const ToolEntry = struct {
    id: []const u8,
    kind: []const u8,
    state: []const u8,
    resident_bytes: u64, // held now: 0 while not loaded
    booked_bytes: u64, // what it books when loaded, loaded or not: measured bytes (else its estimate) and a job's working bytes
    idle_ttl_s: u32,
    idle_ttl_s_default: u32,
    residency: []const u8,
    residency_default: []const u8,
    allowed_residency: []const []const u8,
    last_load_ms: std.json.Value,
    loaded_since: std.json.Value,
    supported: bool, // false: it cannot run on this host (`unsupported_reason` says why); listed for diagnosis, settings refused
    unsupported_reason: std.json.Value, // a string, or null when supported
    priority: i32,
    keep_loaded: bool,
    spare_room_only: bool,
    lane: []const u8,
    capabilities: []const engine.Capability,
    last_error: []const u8,
    license: ?@import("../config.zig").License,
    node: ?[]const u8 = null,
    local: bool = true,
};

const voice_residency = [_][]const u8{ "always", "call", "on_demand", "disabled" };
const plain_residency = [_][]const u8{ "always", "on_demand", "disabled" };

pub fn toolEntry(app: *App, a: std.mem.Allocator, i: usize) !ToolEntry {
    const config = @import("../config.zig");
    const slot = &app.scheduler.slots[i];
    const c = slot.tool.config();
    const n = app.nodeOf(c.id);
    const r = app.scheduler.residency(i);
    const st = slot.tool.state();
    const loaded = st == .ready or st == .busy;
    const why_not = try unsupported(app, a, i);
    return .{
        .supported = why_not == null,
        .unsupported_reason = if (why_not) |w| .{ .string = w } else .null,
        .node = if (n.len > 0) n else null,
        .local = slot.tool.local(),
        .id = c.id,
        .kind = @tagName(c.kind),
        .state = @tagName(st),
        .resident_bytes = slot.resident,
        .booked_bytes = app.scheduler.bookedBytes(i),
        .idle_ttl_s = r.idle_ttl_s,
        .idle_ttl_s_default = c.idle_ttl_s,
        .residency = @tagName(r.rule),
        .residency_default = @tagName(config.residencyOf(c)),
        .allowed_residency = if (config.callAllowed(c.kind)) &voice_residency else &plain_residency,
        .last_load_ms = if (r.last_load_ms) |ms| .{ .integer = @intCast(ms) } else .null,
        .loaded_since = if (loaded and r.loaded_since_s > 0) .{ .integer = r.loaded_since_s } else .null,
        .priority = c.priority,
        .keep_loaded = r.rule == .always,
        .spare_room_only = c.spare_room_only,
        .lane = @tagName(config.laneOf(c)),
        .capabilities = try engine.capabilityList(a, c),
        .last_error = slot.tool.failure().message,
        .license = engine.license(c),
    };
}

fn tools(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const list = try a.alloc(ToolEntry, app.scheduler.slots.len);
    for (list, 0..) |*e, i| e.* = try toolEntry(app, a, i);
    const mem = @import("../sched/memory.zig").available() orelse 0;
    try h.jsonValue(req, a, .ok, .{ .object = "list", .data = list, .mem_available_bytes = mem, .reserve_bytes = app.cfg.reserve_bytes, .budget_bytes = app.cfg.budget_bytes, .budget = app.scheduler.budget(), .voice_sessions = app.voice.count(app.io) });
}

/// POST /v1/tools/{id}/load, /v1/tools/{id}/unload, /v1/tools/{id}/settings, /v1/tools/release.
fn toolControl(app: *App, req: *h.Request, a: std.mem.Allocator, rest: []const u8) !void {
    var c: sched.Control = .{ .op = .release };
    if (!eql(rest, "release")) {
        const p = try toolPath(a, rest) orelse return h.fail(req, a, 404, "invalid_request_error", "no such route");
        const verb = p.verb;
        c.tool = toolIndex(app, p.id) orelse return h.fail(req, a, 404, "invalid_request_error", "unknown model");
        if (eql(verb, "settings")) return toolSettings(app, req, a, c.tool);
        if (eql(verb, "load") and disabled(app, c.tool)) {
            var why: Refusal = .{};
            refuseDisabled(app, c.tool, &why);
            return h.fail(req, a, why.status, why.typ, why.message);
        }
        c.op = if (eql(verb, "load")) .load else if (eql(verb, "unload")) .unload else return h.fail(req, a, 404, "invalid_request_error", "no such route");
    }
    app.scheduler.control(&c) catch |err| return h.fail(req, a, 503, "server_error", @errorName(err));
    if (c.failed) return h.fail(req, a, 503, "server_error", c.message);
    return tools(app, req, a);
}

/// `{id}/{verb}` after /v1/tools/: the id percent-decoded (`fishaudio%2Fdrama-3`, as a client escapes a path segment, is
/// `fishaudio/drama-3`; an id with a raw slash works too, the verb being the last segment), in `a`.
fn toolPath(a: std.mem.Allocator, rest: []const u8) !?struct { id: []const u8, verb: []const u8 } {
    const q = std.mem.indexOfScalar(u8, rest, '?') orelse rest.len;
    const path = rest[0..q];
    const slash = std.mem.lastIndexOfScalar(u8, path, '/') orelse return null;
    if (slash == 0) return null;
    return .{ .id = std.Uri.percentDecodeInPlace(try a.dupe(u8, path[0..slash])), .verb = path[slash + 1 ..] };
}

test "a tool's id in /v1/tools/{id}/... is percent-decoded; a raw slash works too" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "fishaudio%2Fdrama-3/load", "fishaudio%2fdrama-3/load", "fishaudio/drama-3/load" }) |rest| {
        const p = (try toolPath(a, rest)).?;
        try t.expectEqualStrings("fishaudio/drama-3", p.id);
        try t.expectEqualStrings("load", p.verb);
    }
    const s = (try toolPath(a, "qwen3.8-27b/settings")).?;
    try t.expectEqualStrings("qwen3.8-27b", s.id);
    try t.expectEqualStrings("settings", s.verb);
    try t.expectEqualStrings("unload", (try toolPath(a, "whisper%20large/unload")).?.verb);
    try t.expectEqualStrings("whisper large", (try toolPath(a, "whisper%20large/unload")).?.id);
    try t.expect(try toolPath(a, "load") == null);
    try t.expect(try toolPath(a, "/load") == null);
}

/// POST /v1/tools/{id}/settings: `{"residency"?, "idle_ttl_s"?, "reset"?}`; the updated entry.
fn toolSettings(app: *App, req: *h.Request, a: std.mem.Allocator, i: usize) !void {
    const text = try h.body(req, a);
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, if (std.mem.trim(u8, text, " \t\r\n").len > 0) text else "{}", .{}) catch
        return h.fail(req, a, 400, "invalid_request_error", "the body is not JSON");
    switch (try applySettings(app, a, i, v)) {
        .ok => return h.jsonValue(req, a, .ok, try toolEntry(app, a, i)),
        .bad => |m| return h.fail(req, a, 400, "invalid_request_error", m),
        .over_budget => |o| {
            const Env = struct { @"error": struct { message: []const u8, type: []const u8 = "insufficient_memory", code: []const u8 = "over_budget", needs_bytes: u64, always_bytes: u64, ceiling_bytes: u64 } };
            return h.jsonValue(req, a, .conflict, Env{ .@"error" = .{ .message = o.message, .needs_bytes = o.needs, .always_bytes = o.always, .ceiling_bytes = o.ceiling } });
        },
        .not_saved => |m| return h.fail(req, a, 500, "server_error", m),
    }
}

/// What `applySettings` did: applied (and saved), refused (400, 409: nothing changed), or applied but not saved.
pub const Settings = union(enum) {
    ok,
    bad: []const u8,
    over_budget: struct { message: []const u8, needs: u64, always: u64, ceiling: u64 },
    not_saved: []const u8,
};

/// Sets tool `i`'s residency and idle TTL from `v` (`{"residency"?: "always"|"call"|"on_demand", "idle_ttl_s"?: int
/// >= 0, "reset"?: true}`; reset first drops what was set before, so the config's applies) and saves every override to
/// `<data_dir>/residency.json`. Shared by the route and MCP's `set_model_residency` (which also passes `model`).
pub fn applySettings(app: *App, a: std.mem.Allocator, i: usize, v: std.json.Value) !Settings {
    const config = @import("../config.zig");
    if (v != .object) return .{ .bad = "the body must be a JSON object: {\"residency\"?, \"idle_ttl_s\"?, \"reset\"?}" };
    var rule: ?config.Residency = null;
    var ttl: ?u32 = null;
    var reset = false;
    var it = v.object.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const x = e.value_ptr.*;
        if (eql(k, "residency")) {
            if (x != .string) return .{ .bad = "residency must be a string: always, call, on_demand or disabled" };
            rule = std.meta.stringToEnum(config.Residency, x.string) orelse
                return .{ .bad = try std.fmt.allocPrint(a, "unknown residency '{s}': always, call, on_demand or disabled", .{x.string}) };
        } else if (eql(k, "idle_ttl_s")) {
            if (x != .integer or x.integer < 0 or x.integer > std.math.maxInt(u32)) return .{ .bad = "idle_ttl_s must be a whole number of seconds, 0 or more (0: unload right after each job)" };
            ttl = @intCast(x.integer);
        } else if (eql(k, "reset")) {
            if (x != .bool) return .{ .bad = "reset must be true or false" };
            reset = x.bool;
        } else if (!eql(k, "model")) {
            return .{ .bad = try std.fmt.allocPrint(a, "unknown field '{s}': the settings are residency, idle_ttl_s and reset", .{k}) };
        }
    }
    const s = app.scheduler;
    const c = s.slots[i].tool.config();
    if (try unsupported(app, a, i)) |why| return .{ .bad = try std.fmt.allocPrint(a, "{s} cannot run on this host, so it has no settings: {s}", .{ c.id, why }) };
    s.settings.lockUncancelable(app.io);
    defer s.settings.unlock(app.io);
    const now = s.residency(i);
    const next_rule = rule orelse if (reset) null else now.rule_override;
    const next_ttl = ttl orelse if (reset) null else now.ttl_override;
    if (s.setResidency(i, next_rule, next_ttl)) |why| switch (why) {
        .not_voice => |k| return .{ .bad = try std.fmt.allocPrint(a, "residency 'call' is for chat, speech and transcription models; {s} is {s} model (allowed: always, on_demand, disabled)", .{ c.id, article(k) }) },
        .over_budget => |o| return .{ .over_budget = .{
            .message = try std.fmt.allocPrint(a, "{s} cannot be always loaded: it needs {d:.1} GiB, which brings the always-loaded models to {d:.1} GiB, over the {d:.1} GiB budget. Set another model to call or on_demand first.", .{ c.id, gibs(o.needs), gibs(o.always), gibs(o.ceiling) }),
            .needs = o.needs,
            .always = o.always,
            .ceiling = o.ceiling,
        } },
    };
    @import("../sched/residency.zig").save(app.io, app.gpa, app.cfg.data_dir, s) catch |err| {
        std.log.warn("residency: saving {s}/residency.json failed: {s}", .{ app.cfg.data_dir, @errorName(err) });
        return .{ .not_saved = try std.fmt.allocPrint(a, "applied now, but {s}/residency.json could not be written ({s}): the change will not survive a restart", .{ app.cfg.data_dir, @errorName(err) }) };
    };
    return .ok;
}

/// Why tool `i` cannot run on this host, in `a`; null when it can (or it runs on another node, which decides). Its
/// engine not built here (the CUDA engines off Linux), its program or weights not installed, its qualification not
/// passed, or no capability left with its weights (`engine.resolve`).
pub fn unsupported(app: *App, a: std.mem.Allocator, i: usize) !?[]const u8 {
    const builtin = @import("builtin");
    const slot = &app.scheduler.slots[i];
    if (!slot.tool.local()) return null;
    const c = slot.tool.config();
    const host = @tagName(builtin.os.tag) ++ "-" ++ @tagName(builtin.cpu.arch);
    if (c.engine.len > 0 and engine.find(c.engine) == null) return try std.fmt.allocPrint(a, "its engine '{s}' is not built for {s}", .{ c.engine, host });
    if (c.engine.len == 0 and c.cmd.len == 0) return "it names neither an engine nor a cmd";
    const cwd = Io.Dir.cwd();
    if (c.cmd.len > 0 and std.fs.path.isAbsolute(c.cmd[0])) {
        cwd.access(app.io, c.cmd[0], .{}) catch return try std.fmt.allocPrint(a, "its program {s} is not installed", .{c.cmd[0]});
    }
    if (c.weights.len > 0 and std.fs.path.isAbsolute(c.weights)) {
        cwd.access(app.io, c.weights, .{}) catch return try std.fmt.allocPrint(a, "its weights are not installed at {s}", .{c.weights});
    }
    const buf = try a.alloc(u8, 384);
    if (app.unqualified(a, c, buf)) |why| return why;
    if ((try engine.capabilityList(a, c)).len == 0) return "none of its capabilities is available with its weights";
    return null;
}

fn gibs(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

pub fn toolIndex(app: *App, id: []const u8) ?usize {
    for (app.scheduler.slots, 0..) |slot, i| if (eql(slot.tool.config().id, id)) return i;
    return null;
}

/// The model for a request that needs `cap`: the one named, else the configured default for `cap` (`Config.defaults`),
/// else the first tool that can do it, else the first of its kind (so `enqueue`'s refusal names a model); a disabled
/// tool is skipped unless named (a disabled default: the next enabled tool of its kind). Null with `why` set when the
/// named model is unknown (404), of another kind (400) or disabled (409), or no enabled model of the kind exists (404,
/// or 409 when the only ones are disabled).
pub fn resolve(app: *App, model: ?[]const u8, cap: request.Capability, why: *Refusal) ?usize {
    const kind = @tagName(cap.kind());
    if (model orelse app.cfg.defaults.get(cap)) |m| named: {
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
        if (!disabled(app, i)) return i;
        if (model == null) break :named; // the configured default is disabled: the next enabled tool of its kind
        refuseDisabled(app, i, why);
        return null;
    }
    var first: ?usize = null;
    var off: ?usize = null;
    for (app.scheduler.slots, 0..) |slot, i| {
        const c = slot.tool.config();
        if (c.kind != cap.kind()) continue;
        if (disabled(app, i)) {
            if (off == null) off = i;
            continue;
        }
        if (engine.supports(c, cap)) return i;
        if (first == null) first = i;
    }
    if (first == null) if (off) |i| {
        refuseDisabled(app, i, why);
        return null;
    };
    if (first == null) {
        why.* = .{ .status = 404 };
        why.message = std.fmt.bufPrint(why.buf[0..384], "no {s} model is configured", .{kind}) catch "no model of this kind";
    }
    return first;
}

/// Whether tool `i` is disabled in settings (residency `disabled`): it never loads, and its requests are refused.
pub fn disabled(app: *App, i: usize) bool {
    return app.scheduler.residency(i).rule == .disabled;
}

/// 409 `model_disabled`, as every refusal of a disabled tool reads.
fn refuseDisabled(app: *App, i: usize, why: *Refusal) void {
    why.* = .{ .status = 409, .typ = "model_disabled" };
    why.message = std.fmt.bufPrint(why.buf[0..384], "{s} is disabled in settings", .{app.scheduler.slots[i].tool.config().id}) catch "the model is disabled in settings";
}

/// The speech model for a request: the one it names, else the text_to_speech default; but when it names no model and
/// a voice the default lacks, the first speech model that lists that voice (voices are per model, so one picker can
/// offer every model's).
pub fn resolveSpeech(app: *App, model: ?[]const u8, voice: []const u8, why: *Refusal) ?usize {
    const i = resolve(app, model, .text_to_speech, why) orelse return null;
    if (model != null or voice.len == 0 or hasVoice(app, app.scheduler.slots[i].tool.config(), voice)) return i;
    for (app.scheduler.slots, 0..) |slot, k| {
        const c = slot.tool.config();
        if (engine.supports(c, .text_to_speech) and hasVoice(app, c, voice) and !disabled(app, k)) return k;
    }
    return i;
}

fn article(k: request.Kind) []const u8 {
    return switch (k) {
        .image => "an image",
        .video => "a video",
        .speech => "a speech",
        .transcription => "a transcription",
        .chat => "a chat",
        .decision => "a decision",
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
    size: []const u8 = "", // "": 1024x1024, or for an edit the input's shape
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

fn images(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, scheme: []const u8, ctype: []const u8, edit: bool, caller: pair.Caller) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    var why: []const u8 = "the body is not a valid image request";
    var raws: std.ArrayList([]const u8) = .empty;
    const b = (if (edit) parseBody(ImageBody, app, a, host, text, ctype, &.{ "image", "image[]", "images" }, &raws, &why) else parseBody(ImageBody, app, a, host, text, ctype, &.{}, &raws, &why)) catch
        return h.fail(req, a, 400, "invalid_request_error", why);
    var mask_raw: std.ArrayList([]const u8) = .empty;
    if (edit) _ = parseBody(struct {}, app, a, host, text, ctype, &.{"mask"}, &mask_raw, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    if (mask_raw.items.len > 1) return h.fail(req, a, 400, "invalid_request_error", "send one mask");
    const refs_files = inputs.validate(a, "ref", raws.items, (request.Limits{}).max_references, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    if (edit and refs_files.len == 0) return h.fail(req, a, 400, "invalid_request_error", "image is required: send image (or image[]) as files, base64 or URLs");
    const mask_files = inputs.validate(a, "mask", mask_raw.items, 1, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    const files = try std.mem.concat(a, inputs.File, &.{ refs_files, mask_files });
    // an edit without a size (or "auto") keeps the input's shape, at the size the model picks
    const wh = if (edit and (b.size.len == 0 or eql(b.size, "auto"))) [2]u32{ 0, 0 } else request.parseSize(if (b.size.len == 0 or eql(b.size, "auto")) "1024x1024" else b.size) orelse return h.fail(req, a, 400, "invalid_request_error", "size must be WIDTHxHEIGHT");
    const refs = try a.alloc([]const u8, refs_files.len);
    for (refs, refs_files) |*d, f| d.* = f.name;
    var why_model: Refusal = .{};
    const tool = resolve(app, b.model, if (edit) .image_edit else .text_to_image, &why_model) orelse return h.fail(req, a, why_model.status, why_model.typ, why_model.message);
    const r: request.Request = .{ .image = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .n = b.n, .seed = b.seed orelse randomSeed(app.io), .steps = b.steps, .guidance = b.guidance, .references = refs, .mask = if (mask_files.len > 0) mask_files[0].name else null } };
    const j = (try submit(app, req, a, tool, r, files, caller.owner())) orelse return;
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const Item = struct { b64_json: ?[]const u8 = null, url: ?[]const u8 = null };
    const items = try a.alloc(Item, j.files.len);
    const as_url = eql(b.response_format, "url");
    for (items, j.files) |*it, f| {
        if (as_url) {
            it.* = .{ .url = try std.fmt.allocPrint(a, "{s}://{s}/v1/files/{s}/{s}", .{ scheme, host, j.id, f }) };
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

/// Fills in the model's defaults, then validates and queues `asked` for `tool` on behalf of `owner` (a device id, or
/// "": `pair.Caller.owner`; the HTTP routes, the MCP tools and voice sessions share it), saving `files` in its job
/// directory first; null with `why` set on refusal. A model without the capability the request needs is refused.
pub fn enqueue(app: *App, tool: usize, asked: request.Request, files: []const inputs.File, owner: []const u8, why: *Refusal) ?*jobs.Job {
    const cfg = app.scheduler.slots[tool].tool.config();
    if (app.plan) |p| if (p.places[tool] == .follower) {
        why.message = std.fmt.bufPrint(why.buf[0..384], "this node runs rank {d} of '{s}', which serves no requests: send them to rank 0's node, {s} (or the front)", .{ p.places[tool].follower.rank, cfg.id, p.nodes[p.places[tool].follower.nodes[0]].id }) catch "this node runs another rank of the model: send requests to its rank 0";
        return null;
    };
    if (disabled(app, tool)) {
        refuseDisabled(app, tool, why);
        return null;
    }
    var r = asked;
    engine.fill(cfg, &r); // what the request left to the model (a video's size)
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
    // A model gated by a qualification file is refused before anything loads until this machine has qualified it.
    var gate: std.heap.ArenaAllocator = .init(app.gpa);
    defer gate.deinit();
    if (app.unqualified(gate.allocator(), cfg, why.buf[0..384])) |msg| {
        why.status = 503;
        why.typ = "server_error";
        why.message = msg;
        return null;
    }
    const j = app.table.create(app.io, app.jobs_dir, tool, r, owner, sched.nowUnix(app.io)) catch |err| {
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
fn submit(app: *App, req: *h.Request, a: std.mem.Allocator, tool: usize, r: request.Request, files: []const inputs.File, owner: []const u8) !?*jobs.Job {
    var why: Refusal = .{};
    const j = enqueue(app, tool, r, files, owner, &why) orelse {
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
    size: []const u8 = "", // "": the model's own size
    seconds: []const u8 = "5", // OpenAI sends seconds as a string
    fps: u32 = 24,
    seed: ?u64 = null,
    steps: u32 = 0,
    guidance: f32 = 0,
    audio: bool = true,
};

/// `input_reference` (OpenAI's name) is the first frame: a multipart file, or JSON base64, a data URL or a URL.
fn videoCreate(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, ctype: []const u8, caller: pair.Caller) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    var why: []const u8 = "the body is not a valid video request";
    var raws: std.ArrayList([]const u8) = .empty;
    const b = parseBody(VideoBody, app, a, host, text, ctype, &.{"input_reference"}, &raws, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    const files = inputs.validate(a, "first_frame", raws.items, 1, &why) catch return h.fail(req, a, 400, "invalid_request_error", why);
    const wh: [2]u32 = if (b.size.len == 0) .{ 0, 0 } else request.parseSize(b.size) orelse return h.fail(req, a, 400, "invalid_request_error", "size must be WIDTHxHEIGHT");
    const seconds = std.fmt.parseInt(u32, b.seconds, 10) catch return h.fail(req, a, 400, "invalid_request_error", "seconds must be a whole number");
    var why_model: Refusal = .{};
    const tool = resolve(app, b.model, if (files.len > 0) .image_to_video else .text_to_video, &why_model) orelse return h.fail(req, a, why_model.status, why_model.typ, why_model.message);
    const r: request.Request = .{ .video = .{ .prompt = b.prompt, .negative_prompt = b.negative_prompt, .width = wh[0], .height = wh[1], .seconds = seconds, .fps = b.fps, .seed = b.seed orelse randomSeed(app.io), .steps = b.steps, .guidance = b.guidance, .audio = b.audio, .first_frame = if (files.len > 0) files[0].name else null } };
    const j = (try submit(app, req, a, tool, r, files, caller.owner())) orelse return;
    try videoObject(app, req, a, j);
}

/// GET {id}, GET {id}/content, DELETE {id}.
fn videoRoute(app: *App, req: *h.Request, a: std.mem.Allocator, rest: []const u8, caller: pair.Caller) !void {
    const content = std.mem.endsWith(u8, rest, "/content");
    const id = if (content) rest[0 .. rest.len - "/content".len] else rest;
    const j = app.table.get(app.io, id) orelse return h.fail(req, a, 404, "invalid_request_error", "no such video");
    if (j.kind() != .video or !caller.sees(j)) return h.fail(req, a, 404, "invalid_request_error", "no such video");
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

/// GET /v1/jobs/{id}: any job's state and, once done, its output URLs (a client that lost a job's result event, such as a
/// voice session that reconnected while its picture was made, asks here).
fn jobObject(app: *App, req: *h.Request, a: std.mem.Allocator, id: []const u8, caller: pair.Caller) !void {
    const j = app.table.get(app.io, id) orelse return h.fail(req, a, 404, "invalid_request_error", "no such job");
    if (!caller.sees(j)) return h.fail(req, a, 404, "invalid_request_error", "no such job"); // another device's: as if absent
    const Err = struct { code: []const u8, message: []const u8 };
    try app.table.mutex.lock(app.io);
    defer app.table.mutex.unlock(app.io);
    const urls = try a.alloc([]const u8, if (j.status == .completed) j.files.len else 0);
    for (urls, 0..) |*u, i| u.* = try std.fmt.allocPrint(a, "/v1/files/{s}/{s}", .{ j.id, j.files[i] });
    try h.jsonValue(req, a, .ok, .{
        .id = j.id,
        .object = "job",
        .kind = @tagName(j.kind()),
        .status = @tagName(j.status),
        .progress = j.percent,
        .phase = try a.dupe(u8, j.phaseName()),
        .urls = urls,
        .@"error" = if (j.status == .failed or j.status == .cancelled) @as(?Err, .{ .code = j.error_type, .message = try a.dupe(u8, j.error_message) }) else null,
    });
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

/// The content type of an output file, by its extension.
pub fn contentType(name: []const u8) []const u8 {
    const types = [_][2][]const u8{ .{ ".png", "image/png" }, .{ ".mp4", "video/mp4" }, .{ ".wav", "audio/wav" }, .{ ".json", "application/json" }, .{ ".txt", "text/plain; charset=utf-8" } };
    for (types) |t| if (std.mem.endsWith(u8, name, t[0])) return t[1];
    return "application/octet-stream";
}

/// GET /v1/files/{job}/{name}: an output file (`response_format: url`).
fn file(app: *App, req: *h.Request, a: std.mem.Allocator, rest: []const u8, caller: pair.Caller) !void {
    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return h.fail(req, a, 404, "invalid_request_error", "no such file");
    const j = app.table.get(app.io, rest[0..slash]) orelse return h.fail(req, a, 404, "invalid_request_error", "no such file");
    if (!caller.sees(j)) return h.fail(req, a, 404, "invalid_request_error", "no such file");
    const name = rest[slash + 1 ..];
    for (j.files) |f| {
        if (!eql(f, name)) continue;
        const data = try Io.Dir.cwd().readFileAlloc(app.io, try std.fs.path.join(a, &.{ j.dir, f }), a, .limited(1 << 31));
        return h.bytes(req, contentType(f), data);
    }
    return h.fail(req, a, 404, "invalid_request_error", "no such file");
}

test "disabled tools: refused by name with 409 model_disabled, skipped for a default; unsupported tools say why" {
    const t = std.testing;
    var fakes = [_]sched.FakeTool{
        .{ .cfg = .{ .id = "a", .kind = .chat, .cmd = &.{"x"} } },
        .{ .cfg = .{ .id = "b", .kind = .chat, .cmd = &.{"x"} } },
        .{ .cfg = .{ .id = "img", .kind = .image, .engine = "no_such_engine" } },
        .{ .cfg = .{ .id = "img2", .kind = .image, .cmd = &.{"x"}, .weights = "/nonexistent/localrouter-weights" } },
    };
    var b: sched.Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    b.cfg.defaults.chat_completion = "a";
    var app: App = .{ .gpa = t.allocator, .io = t.io, .cfg = &b.cfg, .scheduler = &b.s, .table = &b.table, .jobs_dir = "" };
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: Refusal = .{};
    try t.expectEqual(@as(?usize, 0), resolve(&app, null, .chat_completion, &why));
    try t.expect(b.s.setResidency(0, .disabled, null) == null);
    // The configured default is disabled: a request that names no model goes to the next enabled chat model.
    try t.expectEqual(@as(?usize, 1), resolve(&app, null, .chat_completion, &why));
    // Named, it is refused.
    try t.expect(resolve(&app, "a", .chat_completion, &why) == null);
    try t.expectEqual(@as(u16, 409), why.status);
    try t.expectEqualStrings("model_disabled", why.typ);
    try t.expectEqualStrings("a is disabled in settings", why.message);
    why = .{};
    try t.expect(enqueue(&app, 0, .{ .chat = .{ .body = "{}" } }, &.{}, "", &why) == null);
    try t.expectEqual(@as(u16, 409), why.status);
    // None of its kind enabled: refused as disabled, not as missing.
    try t.expect(b.s.setResidency(1, .disabled, null) == null);
    why = .{};
    try t.expect(resolve(&app, null, .chat_completion, &why) == null);
    try t.expectEqualStrings("model_disabled", why.typ);
    // What can run here, and why the others cannot; their settings are refused.
    try t.expect(try unsupported(&app, a, 0) == null);
    try t.expect(std.mem.indexOf(u8, (try unsupported(&app, a, 2)).?, "engine 'no_such_engine' is not built for") != null);
    try t.expect(std.mem.indexOf(u8, (try unsupported(&app, a, 3)).?, "weights are not installed") != null);
    try t.expect(!(try toolEntry(&app, a, 2)).supported);
    const body = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"residency\": \"disabled\"}", .{});
    try t.expect(try applySettings(&app, a, 2, body) == .bad);
}
