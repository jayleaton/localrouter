//! Typed decisions: POST /v1/systemone (the System One shape, `tool/systemone.zig`), the MCP `decide` tool and `ask`,
//! the call the daemon's own features (the voice loop's shadow decisions) use. Every path checks the questions before
//! a job is queued and the model's answers against them before anyone reads them, so a caller only ever sees a reply in
//! which every question is answered with its own type, a choice among the offered options and probabilities in range.
//!
//! A System One client names its model (`jev-latest`, say); a name that is not a decision tool here goes to the
//! default decision model, so existing clients only change their base URL. A decision changes nothing by itself: what
//! to do with it is the caller's code.

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const routes = @import("routes.zig");
const jobs = @import("../sched/jobs.zig");
const systemone = @import("../tool/systemone.zig");

const App = routes.App;
const Value = std.json.Value;
pub const Refusal = routes.Refusal;

/// A checked decision: the answers (in the questions' order), the model that gave them and how long the job took.
pub const Result = struct {
    model: []const u8,
    answers: []const systemone.Answer,
    ms: u64,
    raw: []const u8, // the model's reply, already checked against the questions
};

fn refuse(why: *Refusal, status: u16, typ: []const u8, msg: []const u8) ?*jobs.Job {
    why.status = status;
    why.typ = typ;
    why.message = msg;
    return null;
}

/// The decision tool for `model`: that tool when it names one of this machine's decision tools, else the default.
fn pick(app: *App, model: ?[]const u8, why: *Refusal) ?usize {
    if (model) |m| for (app.scheduler.slots, 0..) |slot, i| {
        const c = slot.tool.config();
        if (c.kind == .decision and std.mem.eql(u8, c.id, m)) return i;
    };
    return routes.resolve(app, null, .typed_decision, why);
}

/// Queues `body` (a parsed System One request whose `questions` are `qs`) for the decision model it names.
fn start(app: *App, a: std.mem.Allocator, body: Value, why: *Refusal) ?*jobs.Job {
    var obj = body.object;
    const model: ?[]const u8 = if (obj.get("model")) |m| (if (m == .string) m.string else null) else null;
    const tool = pick(app, model, why) orelse return null;
    obj.put(a, "model", .{ .string = app.scheduler.slots[tool].tool.config().id }) catch return refuse(why, 500, "server_error", "out of memory");
    const text = std.json.Stringify.valueAlloc(a, Value{ .object = obj }, .{}) catch return refuse(why, 500, "server_error", "out of memory");
    return routes.enqueue(app, tool, .{ .decision = .{ .body = text } }, &.{}, why);
}

/// Waits for `j` and checks its reply against `qs`; null with `why` set on failure.
fn finish(app: *App, a: std.mem.Allocator, j: *jobs.Job, qs: []const systemone.Question, why: *Refusal) ?Result {
    j.done.wait(app.io) catch return null;
    if (j.status != .completed) {
        why.status = j.http_status;
        why.typ = j.error_type;
        why.message = j.error_message;
        return null;
    }
    const path = std.fs.path.join(a, &.{ j.dir, "decision.json" }) catch return null;
    const raw = Io.Dir.cwd().readFileAlloc(app.io, path, a, .limited(4 << 20)) catch {
        _ = refuse(why, 500, "server_error", "the decision model wrote no reply");
        return null;
    };
    var bad: []const u8 = "";
    const answers = systemone.parseAnswers(a, qs, raw, &bad) catch |err| {
        _ = refuse(why, 500, "server_error", if (err == error.OutOfMemory) "out of memory" else bad);
        return null;
    };
    const model = app.scheduler.slots[j.tool].tool.config().id;
    return .{ .model = model, .answers = answers, .ms = j.ms, .raw = raw };
}

/// Asks the default decision model `questions` about `state_json` (JSON text) and waits for checked answers. For the
/// daemon's own features; null (with `why`) when no decision model is configured, it is not qualified, or it failed.
pub fn ask(app: *App, a: std.mem.Allocator, state_json: []const u8, questions: []const systemone.Question, why: *Refusal) ?Result {
    const text = systemone.encodeRequest(a, "", state_json, questions) catch {
        _ = refuse(why, 500, "server_error", "out of memory");
        return null;
    };
    const body = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch {
        _ = refuse(why, 400, "invalid_request_error", "the decision state is not JSON");
        return null;
    };
    var bad: []const u8 = "";
    _ = systemone.parseQuestions(a, body, &bad) catch {
        _ = refuse(why, 400, "invalid_request_error", bad);
        return null;
    };
    const j = start(app, a, body, why) orelse return null;
    return finish(app, a, j, questions, why);
}

/// Whether the default decision model is loaded now. Shadow decisions ask only then: a decision that had to load its
/// model could unload another one to make room, which shadow work should not cause (a model unloaded between this check
/// and the job is the one narrow exception: the job then loads it as any request would).
pub fn available(app: *App) bool {
    var why: Refusal = .{};
    const i = routes.resolve(app, null, .typed_decision, &why) orelse return false;
    return switch (app.scheduler.slots[i].tool.state()) {
        .ready, .busy => true,
        .unloaded, .loading, .failed => false,
    };
}

/// A parsed request body, checked and run: the checked result, or null with `why` set.
pub fn decide(app: *App, a: std.mem.Allocator, body: Value, why: *Refusal) ?Result {
    var bad: []const u8 = "";
    const qs = systemone.parseQuestions(a, body, &bad) catch |err| {
        _ = refuse(why, if (err == error.OutOfMemory) 500 else 400, "invalid_request_error", if (err == error.OutOfMemory) "out of memory" else bad);
        return null;
    };
    const j = start(app, a, body, why) orelse return null;
    return finish(app, a, j, qs, why);
}

/// POST /v1/systemone.
pub fn systemOne(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const body = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return h.fail(req, a, 400, "invalid_request_error", "the body is not JSON: send {model, state, questions}");
    var why: Refusal = .{};
    const r = decide(app, a, body, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    return h.json(req, .ok, r.raw);
}
