//! The voice loop's backend in the daemon: speech to text and text to speech are ordinary scheduler jobs (on the
//! realtime lane by default), the chat model is OpenAI's streaming API at `voice.chat_url` (this daemon's own /v1 when
//! unset), and tool calls queue image and video jobs whose files are served from /v1/files. Voice jobs keep their
//! audio only until the next expiry tick.

const std = @import("std");
const Io = std.Io;
const routes = @import("routes.zig");
const speech = @import("speech.zig");
const request = @import("../tool/request.zig");
const jobs = @import("../sched/jobs.zig");
const engine = @import("../engine/engine.zig");
const audio = @import("../media/audio.zig");
const backend = @import("../voice/backend.zig");
const chat = @import("../voice/chat.zig");
const resample = @import("../voice/resample.zig");
const Stop = @import("../voice/stop.zig").Stop;

const App = routes.App;
const Why = backend.Why;
const poll_ms = 20; // how often a wait looks at its stop flag

pub const Local = struct {
    app: *App,
    host: []const u8, // the Host the session connected to, for file URLs
    client: chat.Client,

    pub fn init(app: *App, host: []const u8, base_url_buf: *[96]u8) Local {
        const v = app.cfg.voice;
        const base = if (v.chat_url.len > 0) v.chat_url else std.fmt.bufPrint(base_url_buf, "http://{f}/v1", .{app.self_addr}) catch "http://127.0.0.1:8190/v1";
        return .{ .app = app, .host = host, .client = .{ .io = app.io, .gpa = app.gpa, .base_url = base, .api_key = v.chat_api_key } };
    }

    pub fn asBackend(l: *Local) backend.Backend {
        return .{ .ptr = l, .vtable = &vtable };
    }
};

const vtable: backend.Backend.VTable = .{ .transcribe = transcribe, .speak = speak, .stream = chatStream, .start_job = startJob, .wait_job = waitJob, .offer = offer, .check = check };

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
    const tool = routes.resolve(app, named(o.model, app.cfg.voice.transcription_model), .speech_to_text, &ref) orelse return failed(why, ref.message);
    const wav = audio.wav(a, 16_000, 1, try resample.toLe(a, pcm16k)) catch return error.OutOfMemory;
    const lang = try std.ascii.allocLowerString(a, std.mem.trim(u8, o.language, " "));
    const r: request.Request = .{ .transcription = .{ .audio = "audio.wav", .duration_ms = pcm16k.len / 16, .language = lang, .prompt = o.prompt, .seed = routes.randomSeed(app.io) } };
    const j = routes.enqueue(app, tool, r, &.{.{ .name = "audio.wav", .data = wav }}, &ref) orelse return failed(why, ref.message);
    try wait(app, j, stop);
    defer forget(app, j);
    if (j.status != .completed) return failed(why, j.error_message);
    const t = speech.readTranscript(app, a, j, &ref) orelse return failed(why, ref.message);
    const text = t.get("text") orelse return "";
    return if (text == .string) try a.dupe(u8, text.string) else "";
}

fn speak(ptr: *anyopaque, a: std.mem.Allocator, text: []const u8, o: backend.Say, stop: *Stop, why: *Why) backend.Error!backend.Pcm {
    const l = self(ptr);
    const app = l.app;
    var ref: routes.Refusal = .{};
    const voice = if (o.voice.len > 0) o.voice else app.cfg.voice.voice;
    const j = speech.startSpeech(app, .{ .model = named(o.model, app.cfg.voice.speech_model), .input = text, .voice = voice }, &ref) orelse return failed(why, ref.message);
    try wait(app, j, stop);
    defer forget(app, j);
    if (j.status != .completed) return failed(why, j.error_message);
    if (j.files.len == 0) return failed(why, "the speech model wrote no audio");
    const path = try std.fs.path.join(a, &.{ j.dir, j.files[0] });
    const bytes = Io.Dir.cwd().readFileAlloc(app.io, path, a, .limited(1 << 30)) catch return failed(why, "could not read the speech model's audio");
    const w = audio.parseWav(bytes) orelse return failed(why, "the speech model wrote audio that is not a WAV file");
    if (!w.pcm or w.bits != 16 or w.channels != 1) return failed(why, "the speech model's audio is not 16-bit mono PCM");
    const samples = resample.fromLe(a, w.data[0 .. w.data.len & ~@as(usize, 1)]) catch return error.OutOfMemory;
    return .{ .samples = samples, .rate = w.rate };
}

fn chatStream(ptr: *anyopaque, a: std.mem.Allocator, r: *const chat.Request, sink: chat.Sink, stop: *Stop, why: *Why) backend.Error!chat.Result {
    return self(ptr).client.stream(a, r, sink, stop, why);
}

fn startJob(ptr: *anyopaque, a: std.mem.Allocator, spec: backend.JobSpec, why: *Why) backend.Error!backend.Job {
    const l = self(ptr);
    const app = l.app;
    var ref: routes.Refusal = .{};
    const seed = routes.randomSeed(app.io);
    var r: request.Request = undefined;
    switch (spec.kind) {
        .image => {
            const wh = request.parseSize(if (spec.size.len > 0) spec.size else "1024x1024") orelse return failed(why, "size must be WIDTHxHEIGHT, such as 1024x1024");
            r = .{ .image = .{ .prompt = spec.prompt, .width = wh[0], .height = wh[1], .seed = seed } };
        },
        .video => r = .{ .video = .{ .prompt = spec.prompt, .seconds = if (spec.seconds > 0) spec.seconds else 5, .seed = seed } },
    }
    const tool = routes.resolve(app, null, r.capability(), &ref) orelse return failed(why, ref.message);
    const j = routes.enqueue(app, tool, r, &.{}, &ref) orelse return failed(why, ref.message);
    return .{ .id = try a.dupe(u8, j.id), .kind = spec.kind, .handle = j };
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
    for (urls, j.files) |*u, f| u.* = std.fmt.allocPrint(a, "http://{s}/v1/files/{s}/{s}", .{ l.host, j.id, f }) catch "";
    return .{ .status = "completed", .urls = urls };
}

fn offer(ptr: *anyopaque) backend.Offer {
    const app = self(ptr).app;
    var o: backend.Offer = .{};
    for (app.scheduler.slots) |slot| {
        const c = slot.tool.config();
        if (engine.supports(c, .text_to_image)) o.image = true;
        if (engine.supports(c, .text_to_video)) o.video = true;
    }
    return o;
}

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
