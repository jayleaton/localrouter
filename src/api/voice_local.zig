//! The voice loop's backend in the daemon: speech to text and text to speech are ordinary scheduler jobs (on the
//! realtime lane by default), the chat model is OpenAI's streaming API at `voice.chat_url` (this daemon's own /v1 when
//! unset), and tool calls queue image and video jobs whose files are served from /v1/files. Voice jobs keep their
//! audio only until the next expiry tick.

const std = @import("std");
const Io = std.Io;
const routes = @import("routes.zig");
const inputs = @import("inputs.zig");
const mcp = @import("mcp.zig");
const speech = @import("speech.zig");
const request = @import("../tool/request.zig");
const jobs = @import("../sched/jobs.zig");
const engine = @import("../engine/engine.zig");
const audio = @import("../media/audio.zig");
const backend = @import("../voice/backend.zig");
const skill = @import("../voice/skill.zig");
const memory = @import("../sched/memory.zig");
const chat = @import("../voice/chat.zig");
const resample = @import("../voice/resample.zig");
const Stop = @import("../voice/stop.zig").Stop;
const gate = @import("../computer/gate.zig");
const hub_mod = @import("../computer/hub.zig");
const decide = @import("decide.zig");
const systemone = @import("../tool/systemone.zig");
const pair = @import("pair.zig");
const Need = @import("../pair.zig").Need;

const App = routes.App;
const Why = backend.Why;
const poll_ms = 20; // how often a wait looks at its stop flag

pub const Local = struct {
    app: *App,
    host: []const u8, // the Host the session connected to, for file URLs
    scheme: []const u8 = "http", // and the scheme it used (`h.scheme`: https behind tailscale serve)
    client: chat.Client,
    token: []const u8 = "", // the session's device token (computer/gate.zig): computer tools only for a device allowed them
    owner: []const u8 = "", // the session's device id (`pair.Caller.owner`): its jobs are that device's

    pub fn init(app: *App, host: []const u8, base_url_buf: *[96]u8) Local {
        const v = app.cfg.voice;
        const base = if (v.chat_url.len > 0) v.chat_url else std.fmt.bufPrint(base_url_buf, "http://{f}/v1", .{app.self_addr}) catch "http://127.0.0.1:8190/v1";
        return .{ .app = app, .host = host, .client = .{ .io = app.io, .gpa = app.gpa, .base_url = base, .api_key = v.chat_api_key } };
    }

    /// The session as device `token` (its id `owner`): its jobs are that device's, and when the chat model is this
    /// daemon's own /v1, the session's chat requests carry the token, so they need the device's scopes as any other.
    pub fn as(l: *Local, token: []const u8, owner: []const u8) void {
        l.token = token;
        l.owner = owner;
        if (l.app.cfg.voice.chat_url.len == 0 and token.len > 0) l.client.api_key = token;
    }

    /// Who the session is now (`pair.callerOf`, re-read for every action: a revoke or a scope change counts at once).
    fn caller(l: *const Local, a: std.mem.Allocator) ?pair.Caller {
        return pair.callerOf(l.app, a, l.token);
    }

    /// Whether the session's device may `n` now.
    fn may(l: *const Local, n: Need) bool {
        var arena: std.heap.ArenaAllocator = .init(l.app.gpa);
        defer arena.deinit();
        const c = l.caller(arena.allocator()) orelse return false;
        return c.may(n);
    }

    /// The paired device behind the session's token, when it may use the computer tools.
    fn computerDevice(l: *const Local, why: *[]const u8) ?gate.Device {
        const svc = l.app.computer orelse {
            why.* = "computer control is switched off in this daemon's config (computer.enabled)";
            return null;
        };
        if (!l.app.cfg.computer.voice) {
            why.* = "computer control is not offered to voice sessions here (computer.voice)";
            return null;
        }
        return switch (svc.gate.check(l.token, .computer)) {
            .ok => |d| d,
            .off => |m| blk: {
                why.* = m;
                break :blk null;
            },
            .denied => |m| blk: {
                why.* = m;
                break :blk null;
            },
        };
    }

    /// The chat model a session starts with: `voice.chat_model`, else (chat served here) this daemon's chat default.
    pub fn defaultChatModel(app: *App) []const u8 {
        const v = app.cfg.voice;
        if (v.chat_model.len > 0 or v.chat_url.len > 0) return v.chat_model;
        var ref: routes.Refusal = .{};
        const i = routes.resolve(app, null, .chat_completion, &ref) orelse return "";
        const c = app.scheduler.slots[i].tool.config();
        return if (engine.supports(c, .chat_completion)) c.id else "";
    }

    pub fn asBackend(l: *Local) backend.Backend {
        return .{ .ptr = l, .vtable = &vtable };
    }
};

const vtable: backend.Backend.VTable = .{ .transcribe = transcribe, .speak = speak, .stream = chatStream, .start_job = startJob, .wait_job = waitJob, .offer = offer, .check = check, .paused = paused, .manage = manage, .computer = computer, .shadow = shadow, .client_tools = clientTools, .yield_chat = yieldChat, .assistant_name = assistantName, .model_skill = modelSkillOf, .speech_warm = speechWarm, .call_models = callModels };

/// The `call` tools and their loads for the open calls (`Scheduler.callLoads`); a failed one says why.
fn callModels(ptr: *anyopaque, a: std.mem.Allocator) error{OutOfMemory}![]const backend.CallModel {
    const sched = self(ptr).app.scheduler;
    var buf: [32]@import("../sched/scheduler.zig").CallLoad = undefined;
    const loads = sched.callLoads(&buf);
    const out = try a.alloc(backend.CallModel, loads.len);
    for (out, loads) |*m, l| {
        const t = sched.slots[l.tool].tool;
        m.* = .{ .id = t.config().id, .state = @tagName(l.state), .elapsed_ms = l.elapsed_ms, .@"error" = switch (l.state) {
            .failed => try a.dupe(u8, t.failure().message),
            .no_room => "it does not fit beside the models kept loaded",
            .warming => "",
            .disabled => "disabled in settings",
            else => "",
        } };
    }
    return out;
}

fn assistantName(ptr: *anyopaque, a: std.mem.Allocator) error{OutOfMemory}![]const u8 {
    const app = self(ptr).app;
    return app.assistant.name(app.io, a);
}

fn modelSkillOf(ptr: *anyopaque, a: std.mem.Allocator) error{OutOfMemory}![]const u8 {
    return modelSkill(self(ptr).app, a);
}

/// The model management procedure (`voice/skill.zig`) for this machine: each configured model's role, id and the
/// memory it needs loaded (as list_models prices it), within the budget they share (`budget_bytes`, else the machine's
/// memory less `reserve_bytes`); `you` is the voice's chat model when it runs here.
pub fn modelSkill(app: *App, a: std.mem.Allocator) error{OutOfMemory}![]const u8 {
    var ref: routes.Refusal = .{};
    const you: ?usize = if (app.cfg.voice.chat_url.len > 0) null else routes.resolve(app, named(app.cfg.voice.chat_model, ""), .chat_completion, &ref);
    const members = try a.alloc(skill.Member, app.scheduler.slots.len);
    for (app.scheduler.slots, members, 0..) |*slot, *m, i| {
        const c = slot.tool.config();
        m.* = .{ .kind = c.kind, .id = c.id, .bytes = app.scheduler.defaultNeeds(i).total(), .you = if (you) |y| y == i else false };
    }
    const budget = if (app.cfg.budget_bytes > 0) app.cfg.budget_bytes else (memory.total() orelse 0) -| app.cfg.reserve_bytes;
    return skill.render(a, budget, members);
}

/// Client tool execution makes the session a harness's inference service: its device needs the inference scope.
fn clientTools(ptr: *anyopaque, why: *Why) bool {
    if (self(ptr).may(.inference)) return true;
    why.set("client tool execution needs the inference scope: on the box, `localrouter pair scopes NAME ...` adds it", .{});
    return false;
}

fn self(ptr: *anyopaque) *Local {
    return @ptrCast(@alignCast(ptr));
}

fn failed(why: *Why, msg: []const u8) backend.Error {
    why.set("{s}", .{msg});
    return error.Failed;
}

/// The configured model for a role when the session names none.
fn named(name: []const u8, default: []const u8) ?[]const u8 {
    if (name.len > 0) return name;
    return if (default.len > 0) default else null;
}

/// Waits for `j`; when `stop` is set first, the job is dropped (a running one finishes unseen) and the wait ends.
fn wait(app: *App, j: *jobs.Job, stop: *Stop) backend.Error!void {
    while (!j.done.isSet()) {
        if (stop.cancelled()) {
            app.scheduler.drop(j);
            forget(app, j);
            return error.Cancelled;
        }
        j.done.waitTimeout(app.io, .{ .duration = .{ .raw = .fromMilliseconds(poll_ms), .clock = .awake } }) catch {};
    }
}

/// While a voice model is unloaded to make room (a video, an explicit unload_model or reserve_for), a turn or a reply
/// waits for it to come back instead of failing; the session was told (`localrouter.voice.paused`). At most
/// `voice.tool_wait_s`, and never past `stop` (barge-in, the session closing).
fn waitVoice(app: *App, stop: *Stop) backend.Error!void {
    if (!app.scheduler.voicePaused()) return;
    std.log.info("voice: waiting for the voice models to come back", .{});
    const limit = std.Io.Clock.awake.now(app.io).addDuration(.fromSeconds(app.cfg.voice.tool_wait_s));
    while (app.scheduler.voicePaused()) {
        if (stop.cancelled()) return error.Cancelled;
        if (std.Io.Clock.awake.now(app.io).nanoseconds >= limit.nanoseconds) return;
        std.Io.sleep(app.io, .fromMilliseconds(250), .awake) catch return error.Cancelled;
    }
}

/// How long one hold keeps the chat server paused unless renewed (each piece of speech renews it): a speaker that stalls
/// lets the chat model run again within this.
const yield_lease_ns: i96 = 600 * std.time.ns_per_ms;

/// Speech first: the session's chat model is paused only when it is this daemon's own server (chat_url unset).
fn yieldChat(ptr: *anyopaque, model: []const u8, op: backend.Backend.Yield, holder: usize) bool {
    const app = self(ptr).app;
    const sched = app.scheduler;
    if (op == .release) {
        sched.chat_yield.release(app.io, holder);
        return true;
    }
    if (app.cfg.voice.chat_url.len > 0) return false;
    var ref: routes.Refusal = .{};
    const i = routes.resolve(app, if (model.len > 0) model else null, .chat_completion, &ref) orelse return false;
    if (op == .check) return sched.slots[i].tool.serverPid() > 0;
    return sched.yieldChat(i, holder, yield_lease_ns); // never while other chat work waits (another session's)
}

fn paused(ptr: *anyopaque) bool {
    return self(ptr).app.scheduler.voicePaused();
}

/// A model management tool call: the same tool as MCP's (`mcp.callText`), so its guardrails (never the chat model,
/// voice sessions told before a voice model goes, every unload logged) are the same for every caller.
fn manage(ptr: *anyopaque, a: std.mem.Allocator, name: []const u8, arguments: []const u8) backend.Managed {
    const l = self(ptr);
    const args = std.json.parseFromSliceLeaky(std.json.Value, a, if (arguments.len > 0) arguments else "{}", .{}) catch
        return .{ .ok = false, .text = "the tool call's arguments are not JSON" };
    std.log.info("voice: model tool {s} {s}", .{ name, arguments[0..@min(arguments.len, 200)] }); // the chat model's call, or the session's own check
    const who = l.caller(a) orelse return .{ .ok = false, .text = "this device is no longer paired with this LocalRouter" };
    const r = mcp.callText(l.app, a, l.host, l.scheme, who, name, args) orelse return .{ .ok = false, .text = "no such tool" }; // as the session's device: its scopes apply
    return .{ .ok = r.ok, .text = r.text };
}

/// The job's directory goes at the next expiry tick, once it has finished.
fn forget(app: *App, j: *jobs.Job) void {
    app.table.mutex.lockUncancelable(app.io);
    defer app.table.mutex.unlock(app.io);
    j.keep_s = 0;
}

fn transcribe(ptr: *anyopaque, a: std.mem.Allocator, pcm16k: []const i16, o: backend.Hear, stop: *Stop, why: *Why) backend.Error![]const u8 {
    const l = self(ptr);
    const app = l.app;
    var ref: routes.Refusal = .{};
    try waitVoice(app, stop);
    const tool = routes.resolve(app, named(o.model, app.cfg.voice.transcription_model), .speech_to_text, &ref) orelse return failed(why, ref.message);
    const wav = audio.wav(a, 16_000, 1, try resample.toLe(a, pcm16k)) catch return error.OutOfMemory;
    const lang = try std.ascii.allocLowerString(a, app.cfg.sttLanguage(o.language));
    const r: request.Request = .{ .transcription = .{ .audio = "audio.wav", .duration_ms = pcm16k.len / 16, .language = lang, .prompt = o.prompt, .seed = routes.randomSeed(app.io), .decode = if (o.greedy) .greedy else null } };
    const j = routes.enqueue(app, tool, r, &.{.{ .name = "audio.wav", .data = wav }}, l.owner, &ref) orelse return failed(why, ref.message);
    try wait(app, j, stop);
    defer forget(app, j);
    if (j.status != .completed) return failed(why, j.error_message);
    if (o.worker_ms) |p| p.* = j.ms;
    const t = speech.readTranscript(app, a, j, &ref) orelse return failed(why, ref.message);
    if (o.detected) |p| {
        if (t.get("language")) |found| {
            if (found == .string) p.* = try a.dupe(u8, found.string);
        }
    }
    const text = t.get("text") orelse return "";
    return if (text == .string) try a.dupe(u8, text.string) else "";
}

/// The speech models to try, in order: the one the session or config names, else the configured `speech_models` that
/// exist here and are qualified (those that list `voice` first, so a voice picked from any model's list is spoken in
/// it), else the text_to_speech default (null).
fn speechCandidates(app: *App, a: std.mem.Allocator, named_model: []const u8, voice: []const u8) ![]const ?[]const u8 {
    if (named_model.len > 0) return a.dupe(?[]const u8, &.{named_model});
    var list: std.ArrayList(?[]const u8) = .empty;
    var buf: [512]u8 = undefined;
    for ([_]bool{ true, false }) |with_voice| for (app.cfg.voice.speech_models) |id| {
        const c = app.cfg.tool(id) orelse continue;
        if (!engine.supports(c, .text_to_speech) or app.unqualified(a, c, &buf) != null) continue;
        if ((voice.len > 0 and routes.hasVoice(app, c, voice)) == with_voice) try list.append(a, id);
    };
    if (list.items.len == 0) try list.append(a, null);
    return list.items;
}

/// The voice a call speaks in when it names none (a hint beside the assistant's name, `api/assistant.zig`): the first
/// speech model a call would try and its voice (`voice.voice`, else the model's `default_voice`); null without one.
pub fn defaultVoice(app: *App, a: std.mem.Allocator) !?struct { model: []const u8, voice: []const u8 } {
    const first = (try speechCandidates(app, a, app.cfg.voice.speech_model, app.cfg.voice.voice))[0];
    var ref: routes.Refusal = .{};
    const i = routes.resolveSpeech(app, first, app.cfg.voice.voice, &ref) orelse return null;
    const c = app.scheduler.slots[i].tool.config();
    const wanted = voiceFor(app, c.id, app.cfg.voice.voice);
    return .{ .model = c.id, .voice = if (wanted.len > 0) wanted else engine.optionString(c, "default_voice", "") };
}

/// Counts what reaches the session, so a model that failed after it began speaking is not followed by another.
const Counted = struct {
    sink: backend.PcmSink,
    samples: usize = 0,

    fn write(ctx: *anyopaque, samples: []const i16, rate: u32) void {
        const c: *Counted = @ptrCast(@alignCast(ctx));
        c.samples += samples.len;
        c.sink.write(c.sink.ctx, samples, rate);
    }
};

/// Speaks with the first candidate that serves; one that fails before any of its audio was sent falls through to the
/// next (with its own default voice, since voices are per model). A voice no candidate has (a client's "default",
/// another API's name) is spoken in the first candidate's default voice rather than failing the turn.
fn speak(ptr: *anyopaque, a: std.mem.Allocator, text: []const u8, o: backend.Say, sink: backend.PcmSink, stop: *Stop, why: *Why) backend.Error!void {
    const l = self(ptr);
    const app = l.app;
    const wanted = if (o.voice.len > 0) o.voice else app.cfg.voice.voice;
    // While a voice model is away (a video is being made) a reply is text only: waiting for the voice would hold the
    // reply, and with it the tool call it carries, until the job it waits for is done. The job's follow-up waits instead.
    if (app.scheduler.voicePaused()) {
        std.log.info("voice: voice is paused; this reply is text only", .{});
        return;
    }
    const candidates = try speechCandidates(app, a, if (o.model.len > 0) o.model else app.cfg.voice.speech_model, wanted);
    var counted: Counted = .{ .sink = sink };
    const out: backend.PcmSink = .{ .ctx = &counted, .write = Counted.write };
    for (candidates, 0..) |model, n| {
        const voice = if (n > 0) "" else voiceFor(app, model, wanted);
        speakWith(app, a, l.owner, model, voice, text, out, stop, why) catch |err| {
            if (err != error.Failed or n + 1 == candidates.len or counted.samples > 0) return err;
            std.log.warn("voice: speech model {s} failed ({s}); trying {s}", .{ model orelse "default", why.message, candidates[n + 1] orelse "default" });
            continue;
        };
        return;
    }
    return failed(why, "no speech model");
}

/// Whether the voice a session's sentence would be spoken in (as `speak` picks it from `o`) is the one warm in its speech
/// model's loaded instance (`Scheduler.speechWarm`): its first sentence then starts at the warm speed.
fn speechWarm(ptr: *anyopaque, a: std.mem.Allocator, o: backend.Say) bool {
    const app = self(ptr).app;
    const wanted = if (o.voice.len > 0) o.voice else app.cfg.voice.voice;
    const first = (speechCandidates(app, a, if (o.model.len > 0) o.model else app.cfg.voice.speech_model, wanted) catch return false)[0];
    const voice = voiceFor(app, first, wanted);
    var ref: routes.Refusal = .{};
    const i = routes.resolveSpeech(app, first, voice, &ref) orelse return false;
    const c = app.scheduler.slots[i].tool.config();
    return app.scheduler.speechWarm(i, if (voice.len > 0) voice else engine.optionString(c, "default_voice", ""));
}

/// The scheduler's hook after every load of a speech tool (`sched.Loaded`: at start, a kept tool coming back, an API
/// load, a call's, a request's): one word in the voice a call speaks in on it (`voice.voice` on the voice loop's model,
/// else the tool's default voice), the audio dropped, so the first job after the load is paid before anyone waits on it:
/// about 7 s to first audio with the voice's references on the Mac's Drama 3 (0.5 s warm), 0.84 s on DGX-01 (0.1 s
/// warm). It only queues the job, and returns it (null: none) so a call's readiness waits for it (`callLoads`: warming).
pub fn warmLoaded(ctx: *anyopaque, tool: usize) ?*const jobs.Job {
    const app: *App = @ptrCast(@alignCast(ctx));
    if (!app.cfg.voice.warm_speech) return null;
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const c = app.scheduler.slots[tool].tool.config();
    if (!engine.supports(c, .text_to_speech)) return null;
    var voice = engine.optionString(c, "default_voice", "");
    if (defaultVoice(app, a) catch null) |d| if (std.mem.eql(u8, d.model, c.id)) {
        voice = d.voice;
    };
    // An open call that speaks on it (a call's load: the call is why it loaded): its voice, not the default, or the call
    // warms a second time (Nora was ready 5.5 s after Mae on the M5 Ultra).
    if (callVoice(app, a, tool, app.voice.speechWants(app.io, a))) |v| voice = v;
    if (app.scheduler.speechWarm(tool, voice)) return null;
    var ref: routes.Refusal = .{};
    const j = speech.startSpeech(app, a, .{ .model = c.id, .input = backend.warm_sentence, .voice = voice, .stream = engine.optionBool(c, "stream", false) }, "", &ref) orelse {
        std.log.warn("voice: {s} not warmed after its load: {s}", .{ c.id, ref.message });
        return null;
    };
    forget(app, j);
    std.log.info("voice: warming {s} in voice '{s}' after its load ({s})", .{ c.id, voice, j.id });
    return j;
}

/// The voice the first of `wants` (open sessions' speech model and voice) speaks in on speech tool `tool`, resolved as
/// a reply resolves it (an empty voice: the tool's default voice); null when none of them speaks on it.
fn callVoice(app: *App, a: std.mem.Allocator, tool: usize, wants: []const backend.Say) ?[]const u8 {
    for (wants) |o| {
        const wanted = if (o.voice.len > 0) o.voice else app.cfg.voice.voice;
        const first = (speechCandidates(app, a, if (o.model.len > 0) o.model else app.cfg.voice.speech_model, wanted) catch continue)[0];
        const voice = voiceFor(app, first, wanted);
        var ref: routes.Refusal = .{};
        const i = routes.resolveSpeech(app, first, voice, &ref) orelse continue;
        if (i != tool) continue;
        return if (voice.len > 0) voice else engine.optionString(app.scheduler.slots[i].tool.config(), "default_voice", "");
    }
    return null;
}

/// `wanted` when `model` has that voice (or lists none, or is the text_to_speech default, which picks its own), else ""
/// (the model's default voice).
fn voiceFor(app: *App, model: ?[]const u8, wanted: []const u8) []const u8 {
    const c = app.cfg.tool(model orelse return wanted) orelse return wanted;
    if (wanted.len == 0 or engine.voices(c) == null or routes.hasVoice(app, c, wanted)) return wanted;
    std.log.warn("voice: {s} has no voice '{s}'; speaking in its default voice", .{ c.id, wanted });
    return "";
}

/// The speech job (`owner`'s), streamed when the model streams (its `stream` option): its audio then reaches `sink` as it
/// is made.
fn speakWith(app: *App, a: std.mem.Allocator, owner: []const u8, model: ?[]const u8, voice: []const u8, text: []const u8, sink: backend.PcmSink, stop: *Stop, why: *Why) backend.Error!void {
    var ref: routes.Refusal = .{};
    const streams = if (model) |m| (if (app.cfg.tool(m)) |c| engine.optionBool(c, "stream", false) else false) else false;
    const j = speech.startSpeech(app, a, .{ .model = model, .input = text, .voice = voice, .stream = streams }, owner, &ref) orelse return failed(why, ref.message);
    defer forget(app, j);
    try tail(app, j, sink, stop, why);
}

/// Hands `speech.wav` to `sink` as the job writes it (a streaming model writes it as it speaks; others all at once at
/// the end), until the job is done; when `stop` is set the job is dropped (ended early if its tool can stay loaded).
fn tail(app: *App, j: *jobs.Job, sink: backend.PcmSink, stop: *Stop, why: *Why) backend.Error!void {
    var path_buf: [4096]u8 = undefined;
    const name = std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], "{s}/speech.wav", .{j.dir}) catch return failed(why, "the job directory's path is too long");
    path_buf[name.len] = 0;
    const path: [:0]const u8 = path_buf[0..name.len :0];
    var fd: std.posix.fd_t = -1;
    defer closeFd(fd);
    var head: ?Head = null;
    var at: u64 = 0;
    var buf: [32 * 1024]u8 align(2) = undefined;
    while (true) {
        if (stop.cancelled()) {
            app.scheduler.drop(j);
            return error.Cancelled;
        }
        const done = j.done.isSet(); // read once more after the job ends: everything it wrote is in the file
        if (fd < 0) fd = std.posix.openatZ(std.posix.AT.FDCWD, path, .{}, 0) catch -1;
        if (fd >= 0 and head == null) {
            const n = readAt(fd, &buf, 0);
            head = parseHead(buf[0..n]) catch |err| return failed(why, switch (err) {
                error.NotPcm16Mono => "the speech model's audio is not 16-bit mono PCM",
                error.NotWav => "the speech model wrote audio that is not a WAV file",
            });
            if (head) |h| at = h.data_at;
        }
        if (head) |h| {
            while (true) {
                const n = readAt(fd, &buf, at) & ~@as(usize, 1); // whole samples; an odd byte waits for its pair
                if (n == 0) break;
                sink.write(sink.ctx, std.mem.bytesAsSlice(i16, buf[0..n]), h.rate);
                at += n;
            }
        }
        if (done) break;
        j.done.waitTimeout(app.io, .{ .duration = .{ .raw = .fromMilliseconds(5), .clock = .awake } }) catch {};
    }
    if (j.status == .cancelled) return error.Cancelled;
    if (j.status != .completed) return failed(why, j.error_message);
    if (head == null) return failed(why, "the speech model wrote no audio");
}

const Head = struct { rate: u32, data_at: u64 };

/// The format and the start of the data of a WAV header that may still be growing: null until the data chunk begins.
/// Its sizes are ignored (a streaming writer fills them in last).
fn parseHead(b: []const u8) error{ NotWav, NotPcm16Mono }!?Head {
    if (b.len < 12) return null;
    if (!std.mem.eql(u8, b[0..4], "RIFF") or !std.mem.eql(u8, b[8..12], "WAVE")) return error.NotWav;
    var rate: u32 = 0;
    var at: usize = 12;
    while (at + 8 <= b.len) {
        const id = b[at..][0..4];
        const size = std.mem.readInt(u32, b[at + 4 ..][0..4], .little);
        if (std.mem.eql(u8, id, "data")) return if (rate == 0) error.NotPcm16Mono else .{ .rate = rate, .data_at = at + 8 };
        if (std.mem.eql(u8, id, "fmt ")) {
            if (at + 24 > b.len) return null;
            const format = std.mem.readInt(u16, b[at + 8 ..][0..2], .little);
            const channels = std.mem.readInt(u16, b[at + 10 ..][0..2], .little);
            const bits = std.mem.readInt(u16, b[at + 22 ..][0..2], .little);
            if ((format != 1 and format != 0xFFFE) or channels != 1 or bits != 16) return error.NotPcm16Mono;
            rate = std.mem.readInt(u32, b[at + 12 ..][0..4], .little);
        }
        at += 8 + size + (size & 1);
    }
    return null;
}

fn closeFd(fd: std.posix.fd_t) void {
    if (fd >= 0) _ = std.posix.system.close(fd);
}

fn readAt(fd: std.posix.fd_t, buf: []u8, offset: u64) usize {
    const rc = std.posix.system.pread(fd, buf.ptr, buf.len, @intCast(offset));
    if (std.posix.errno(rc) != .SUCCESS) return 0;
    return @intCast(rc);
}

test "a growing WAV header: nothing until the data chunk starts; then its rate and offset" {
    const full = audio.header(44_100, 1, 0xFFFF_FFFF);
    try std.testing.expect((try parseHead(full[0..20])) == null);
    const h = (try parseHead(&full)).?;
    try std.testing.expectEqual(@as(u32, 44_100), h.rate);
    try std.testing.expectEqual(@as(u64, 44), h.data_at);
    const stereo = audio.header(24_000, 2, 0);
    try std.testing.expectError(error.NotPcm16Mono, parseHead(&stereo));
    try std.testing.expectError(error.NotWav, parseHead("RIFX0000WAVEfmt "));
}

fn chatStream(ptr: *anyopaque, a: std.mem.Allocator, r: *const chat.Request, sink: chat.Sink, stop: *Stop, why: *Why) backend.Error!chat.Result {
    return self(ptr).client.stream(a, r, sink, stop, why);
}

fn startJob(ptr: *anyopaque, a: std.mem.Allocator, spec: backend.JobSpec, why: *Why) backend.Error!backend.Job {
    const l = self(ptr);
    const app = l.app;
    if (!l.may(.media)) return failed(why, "this device's token does not have the media scope: on the box, `localrouter pair scopes NAME ...` adds it");
    var ref: routes.Refusal = .{};
    const seed = routes.randomSeed(app.io);
    // an input image: an edit's reference, a video's first frame
    var files: []const inputs.File = &.{};
    if (spec.input_file.len > 0 or spec.input_url.len > 0) {
        const raw = try inputImage(l, a, spec, why);
        var msg: []const u8 = "";
        files = inputs.validate(a, if (spec.kind == .image) "ref" else "first_frame", &.{raw}, 1, &msg) catch return failed(why, msg);
    }
    var r: request.Request = undefined;
    switch (spec.kind) {
        .image => {
            // an edit without a size keeps the input's shape (the model picks)
            const wh: [2]u32 = if (spec.size.len == 0 and files.len > 0) .{ 0, 0 } else request.parseSize(if (spec.size.len > 0) spec.size else "1024x1024") orelse return failed(why, "size must be WIDTHxHEIGHT, such as 1024x1024");
            const refs = try a.alloc([]const u8, files.len);
            for (refs, files) |*d, f| d.* = f.name;
            r = .{ .image = .{ .prompt = spec.prompt, .width = wh[0], .height = wh[1], .seed = seed, .references = refs } };
        },
        .video => r = .{ .video = .{ .prompt = spec.prompt, .seconds = if (spec.seconds > 0) spec.seconds else 5, .seed = seed, .first_frame = if (files.len > 0) files[0].name else null } },
    }
    const tool = routes.resolve(app, null, r.capability(), &ref) orelse return failed(why, ref.message);
    const j = routes.enqueue(app, tool, r, files, l.owner, &ref) orelse return failed(why, ref.message);
    return .{ .id = try a.dupe(u8, j.id), .kind = spec.kind, .handle = j };
}

/// A voice job's input image: the file an earlier job of the session made, or the URL the model gave (this daemon's own
/// /v1/files URL of one of the device's jobs is read from the job's directory; any other is fetched as the API does).
fn inputImage(l: *Local, a: std.mem.Allocator, spec: backend.JobSpec, why: *Why) backend.Error![]const u8 {
    const app = l.app;
    var path = spec.input_file;
    if (spec.input_url.len > 0) {
        const at = std.mem.indexOf(u8, spec.input_url, "/v1/files/") orelse {
            var msg: []const u8 = "";
            return inputs.resolve(app.io, app.gpa, a, spec.input_url, app.policy(l.host), &msg) catch failed(why, msg);
        };
        const rest = spec.input_url[at + "/v1/files/".len ..];
        const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return failed(why, "image: a /v1/files URL names a job and one of its files");
        const name = rest[slash + 1 ..];
        const j = app.table.get(app.io, rest[0..slash]) orelse return failed(why, "image: no such picture on this machine (outputs expire): make a new one");
        app.table.mutex.lockUncancelable(app.io);
        const listed = for (j.files) |f| {
            if (std.mem.eql(u8, f, name)) break true;
        } else false;
        app.table.mutex.unlock(app.io);
        if (!listed or !std.mem.eql(u8, j.owner, l.owner)) return failed(why, "image: no such picture on this machine (outputs expire): make a new one");
        path = try std.fs.path.join(a, &.{ j.dir, name });
    }
    return Io.Dir.cwd().readFileAlloc(app.io, path, a, .limited(inputs.max_bytes)) catch failed(why, "the picture could not be read (outputs expire): make a new one");
}

fn waitJob(ptr: *anyopaque, a: std.mem.Allocator, job: backend.Job, timeout_s: u32, stop: *Stop) backend.Outcome {
    const l = self(ptr);
    const app = l.app;
    const j: *jobs.Job = @ptrCast(@alignCast(job.handle));
    const start = Io.Clock.awake.now(app.io);
    while (!j.done.isSet()) {
        if (stop.cancelled()) return .{ .status = "cancelled" };
        const waited = Io.Clock.awake.now(app.io).toNanoseconds() - start.toNanoseconds();
        if (waited > @as(i96, timeout_s) * std.time.ns_per_s) return .{ .status = "timeout", .@"error" = "the job is still running; ask for its status later" };
        j.done.waitTimeout(app.io, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
    }
    app.table.mutex.lockUncancelable(app.io);
    defer app.table.mutex.unlock(app.io);
    if (j.status != .completed) return .{ .status = @tagName(j.status), .@"error" = a.dupe(u8, j.error_message) catch "" };
    const urls = a.alloc([]const u8, j.files.len) catch return .{ .status = "completed" };
    for (urls, j.files) |*u, f| u.* = std.fmt.allocPrint(a, "{s}://{s}/v1/files/{s}/{s}", .{ l.scheme, l.host, j.id, f }) catch "";
    const paths = a.alloc([]const u8, j.files.len) catch return .{ .status = "completed", .urls = urls };
    for (paths, j.files) |*p, f| p.* = std.fs.path.join(a, &.{ j.dir, f }) catch "";
    return .{ .status = "completed", .urls = urls, .files = paths };
}

/// One shadow decision at a time: a turn that ends while one is asked is not asked about (shadow work never queues).
var shadow_busy: std.atomic.Value(bool) = .init(false);

/// A finished turn, copied for the shadow thread: the route question's options and the state it is asked about.
const ShadowTurn = struct {
    arena: std.heap.ArenaAllocator,
    session_id: []const u8,
    response_id: []const u8,
    did: []const u8, // what the chat model did: the first tool it called, or "reply"
    called: []const []const u8,
    options: []systemone.Option,
    state: []const u8, // JSON
    words: []const u8,
    turn_ms: u64,
};

/// How the decision model would have routed a finished voice turn, logged beside what the chat model did
/// (`voice shadow decision {json}`). It runs on its own thread and touches only the app, never the session (which may
/// be gone by then); nothing acts on the decision.
fn shadow(ptr: *anyopaque, turn: backend.Shadow) void {
    const app = self(ptr).app;
    if (!decide.available(app)) return;
    if (shadow_busy.swap(true, .acq_rel)) {
        std.log.info("voice {s}: shadow decision skipped for {s}: the previous one is still being asked", .{ turn.session_id, turn.response_id });
        return;
    }
    const t = app.gpa.create(ShadowTurn) catch return shadow_busy.store(false, .release);
    t.* = copyTurn(app.gpa, turn) catch {
        app.gpa.destroy(t);
        return shadow_busy.store(false, .release);
    };
    const th = std.Thread.spawn(.{}, askShadow, .{ app, t }) catch {
        t.arena.deinit();
        app.gpa.destroy(t);
        return shadow_busy.store(false, .release);
    };
    th.detach();
}

const reply_option = "reply";
const max_words = 2000;

fn copyTurn(gpa: std.mem.Allocator, turn: backend.Shadow) !ShadowTurn {
    var t: ShadowTurn = .{ .arena = .init(gpa), .session_id = "", .response_id = "", .did = reply_option, .called = &.{}, .options = &.{}, .state = "", .words = "", .turn_ms = turn.turn_ms };
    errdefer t.arena.deinit();
    const a = t.arena.allocator();
    t.session_id = try a.dupe(u8, turn.session_id);
    t.response_id = try a.dupe(u8, turn.response_id);
    t.words = try a.dupe(u8, clip(turn.words, max_words));
    const called = try a.alloc([]const u8, turn.called.len);
    for (called, turn.called) |*d, n| d.* = try a.dupe(u8, n);
    t.called = called;
    if (called.len > 0) t.did = called[0];
    var opts: std.ArrayList(systemone.Option) = .empty;
    for (turn.tools) |tool| {
        if (opts.items.len + 1 >= systemone.max_options) break;
        if (std.mem.eql(u8, tool.function.name, reply_option)) continue;
        try opts.append(a, .{ .id = try a.dupe(u8, tool.function.name), .description = try a.dupe(u8, clip(tool.function.description, 400)) });
    }
    try opts.append(a, .{ .id = reply_option, .description = "Answer in speech; no tool is needed for this." });
    t.options = opts.items;
    t.state = try std.json.Stringify.valueAlloc(a, .{ .assistant_said_before = clip(turn.previous, 600), .user_said = t.words }, .{});
    return t;
}

/// At most `n` bytes of `s`, cut at a UTF-8 boundary.
fn clip(s: []const u8, n: usize) []const u8 {
    if (s.len <= n) return s;
    var end = n;
    while (end > 0 and (s[end] & 0xC0) == 0x80) end -= 1;
    return s[0..end];
}

fn askShadow(app: *App, t: *ShadowTurn) void {
    defer {
        t.arena.deinit();
        app.gpa.destroy(t);
        shadow_busy.store(false, .release);
    }
    const a = t.arena.allocator();
    const qs = [_]systemone.Question{.{
        .name = "route",
        .instructions = "The user just spoke to their voice assistant (user_said). Which one action should the assistant take for these words: the tool the request needs, or reply when no tool is needed?",
        .ask = .{ .choice = t.options },
    }};
    var why: decide.Refusal = .{};
    const r = decide.ask(app, a, t.state, &qs, &why) orelse {
        std.log.warn("voice {s}: shadow decision for {s} failed: {s}", .{ t.session_id, t.response_id, why.message });
        return;
    };
    const c = r.answers[0].choice;
    const chose = t.options[c.choice].id;
    const Prob = struct { option: []const u8, p: f64 };
    var top: [3]Prob = undefined;
    const n = @min(top.len, t.options.len);
    var used: [3]usize = undefined;
    for (0..n) |k| { // the three most probable options, for reading the log
        var best: usize = 0;
        var best_p: f64 = -1;
        for (c.probabilities, 0..) |p, i| {
            if (std.mem.indexOfScalar(usize, used[0..k], i) != null) continue;
            if (p > best_p) {
                best = i;
                best_p = p;
            }
        }
        used[k] = best;
        top[k] = .{ .option = t.options[best].id, .p = best_p };
    }
    const line = std.json.Stringify.valueAlloc(a, .{
        .session = t.session_id,
        .response = t.response_id,
        .words = t.words,
        .chat_did = t.did,
        .chat_called = t.called,
        .decision = chose,
        .p = c.probabilities[c.choice],
        .agree = std.mem.eql(u8, chose, t.did),
        .top = top[0..n],
        .model = r.model,
        .decide_ms = r.ms,
        .turn_ms = t.turn_ms,
    }, .{}) catch return;
    std.log.info("voice shadow decision {s}", .{line});
}

fn offer(ptr: *anyopaque) backend.Offer {
    const app = self(ptr).app;
    var o: backend.Offer = .{};
    if (self(ptr).may(.media)) { // the picture and video tools only for a device allowed media
        for (app.scheduler.slots) |slot| {
            const c = slot.tool.config();
            if (engine.supports(c, .text_to_image)) o.image = true;
            if (engine.supports(c, .image_edit)) o.image_edit = true;
            if (engine.supports(c, .text_to_video)) o.video = true;
            if (engine.supports(c, .image_to_video)) o.image_to_video = true;
        }
    }
    var why: []const u8 = "";
    if (self(ptr).computerDevice(&why) != null) o.computer = app.computer.?.hub.connected() > 0;
    return o;
}

/// A computer tool call from the voice session's chat model: as the requesting device, with the user's own words for
/// the turn; computer_use waits for the task to end (its working time is bounded; taps are not), and a session that
/// closes stops it.
fn computer(ptr: *anyopaque, a: std.mem.Allocator, name: []const u8, arguments: []const u8, utterance: []const u8, speaker: backend.Speaker, stop: *Stop) backend.Managed {
    const l = self(ptr);
    var why: []const u8 = "";
    const d = l.computerDevice(&why) orelse return .{ .ok = false, .text = why };
    const hub = &l.app.computer.?.hub;
    const who: hub_mod.Requester = .{ .source = "voice", .device = d.name, .utterance = utterance };
    const Args = struct { control: ?std.json.Value = null, goal: []const u8 = "", text: ?[]const u8 = null, computer: []const u8 = "" };
    const args = std.json.parseFromSliceLeaky(Args, a, if (arguments.len > 0) arguments else "{}", .{ .ignore_unknown_fields = true }) catch
        return .{ .ok = false, .text = "the tool call's arguments are not valid JSON" };
    std.log.info("voice: computer tool {s}", .{name});
    if (std.mem.eql(u8, name, "computer_control")) {
        const ctl = args.control orelse return .{ .ok = false, .text = "computer_control needs control" };
        const r = hub.control(a, who, args.computer, ctl);
        return .{ .ok = r.ok, .text = std.json.Stringify.valueAlloc(a, .{ .done = r.ok, .result = r.text }, .{}) catch r.text };
    }
    var listener: VoiceListener = .{ .speaker = speaker };
    const r = hub.startUse(a, who, args.computer, args.goal, args.text, .{ .ctx = &listener, .event = VoiceListener.event });
    if (!r.ok) {
        if (r.task_id.len > 0) hub.detach(r.task_id);
        return .{ .ok = false, .text = r.text };
    }
    defer hub.detach(r.task_id); // the listener lives on this stack
    var stopped = false;
    while (true) {
        var tmp: std.heap.ArenaAllocator = .init(l.app.gpa);
        defer tmp.deinit();
        const v = hub.wait(tmp.allocator(), r.task_id, 500) orelse break;
        if (!std.mem.eql(u8, v.status, "running")) break;
        if (stop.cancelled() and !stopped) {
            stopped = true;
            _ = hub.stop(tmp.allocator(), r.task_id, "the voice session closed");
            break; // the helper ends the task; nobody is left to hear it
        }
    }
    const v = hub.view(a, r.task_id) orelse return .{ .ok = false, .text = "the task is gone" };
    return .{ .ok = std.mem.eql(u8, v.status, "done"), .text = std.json.Stringify.valueAlloc(a, v, .{ .emit_null_optional_fields = false }) catch "{}" };
}

/// Speaks what the user must hear while a computer task runs: an approval waiting for their tap.
const VoiceListener = struct {
    speaker: backend.Speaker,

    fn event(ctx: *anyopaque, kind: hub_mod.EventKind, text: []const u8) void {
        const vl: *VoiceListener = @ptrCast(@alignCast(ctx));
        if (kind != .approval) return;
        var buf: [600]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "I need your OK in the LocalRouter app to {s}.", .{text[0..@min(text.len, 500)]}) catch return;
        vl.speaker.say(vl.speaker.ctx, line);
    }
};

fn check(ptr: *anyopaque, role: backend.Role, name: []const u8, why: *Why) bool {
    const app = self(ptr).app;
    const cap: request.Capability = switch (role) {
        .chat => return true, // the chat server knows its own models
        .speech => .text_to_speech,
        .transcription => .speech_to_text,
    };
    var ref: routes.Refusal = .{};
    const i = routes.resolve(app, name, cap, &ref) orelse {
        why.set("{s}", .{ref.message});
        return false;
    };
    if (!engine.supports(app.scheduler.slots[i].tool.config(), cap)) {
        why.set("model '{s}' cannot do {s}", .{ name, @tagName(cap) });
        return false;
    }
    return true;
}

test "a speech model's post-load warm-up speaks the open call's voice; an empty one is its default voice" {
    const t = std.testing;
    const sched = @import("../sched/scheduler.zig");
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const opts = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"default_voice\": \"Mae\", \"voices\": [\"Mae\", \"Nora\"]}", .{});
    var fakes = [_]sched.FakeTool{
        .{ .cfg = .{ .id = "chat", .kind = .chat, .cmd = &.{"x"} } },
        .{ .cfg = .{ .id = "drama", .kind = .speech, .cmd = &.{"x"}, .options = opts } },
    };
    var b: sched.Bench = undefined;
    try b.init(&fakes, 100);
    defer b.deinit();
    var app: App = .{ .gpa = t.allocator, .io = t.io, .cfg = &b.cfg, .scheduler = &b.s, .table = &b.table, .jobs_dir = "" };
    try t.expectEqualStrings("Nora", callVoice(&app, a, 1, &.{.{ .voice = "Nora" }}).?);
    try t.expectEqualStrings("Mae", callVoice(&app, a, 1, &.{.{}}).?); // a call that named none: the default voice
    try t.expect(callVoice(&app, a, 1, &.{}) == null); // no call: warmLoaded keeps the default
    try t.expect(callVoice(&app, a, 0, &.{.{ .voice = "Nora" }}) == null); // not the tool it speaks on
}
