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
    const r: request.Request = .{ .transcription = .{ .audio = "audio.wav", .duration_ms = pcm16k.len / 16, .language = lang, .prompt = o.prompt, .seed = routes.randomSeed(app.io), .decode = if (o.greedy) .greedy else null } };
    const j = routes.enqueue(app, tool, r, &.{.{ .name = "audio.wav", .data = wav }}, &ref) orelse return failed(why, ref.message);
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
        if (!engine.supports(c, .text_to_speech) or engine.unqualified(app.io, a, c, &buf) != null) continue;
        if ((voice.len > 0 and engine.hasVoice(c, voice)) == with_voice) try list.append(a, id);
    };
    if (list.items.len == 0) try list.append(a, null);
    return list.items;
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
    const candidates = try speechCandidates(app, a, if (o.model.len > 0) o.model else app.cfg.voice.speech_model, wanted);
    var counted: Counted = .{ .sink = sink };
    const out: backend.PcmSink = .{ .ctx = &counted, .write = Counted.write };
    for (candidates, 0..) |model, n| {
        const voice = if (n > 0) "" else voiceFor(app, model, wanted);
        speakWith(app, a, model, voice, text, out, stop, why) catch |err| {
            if (err != error.Failed or n + 1 == candidates.len or counted.samples > 0) return err;
            std.log.warn("voice: speech model {s} failed ({s}); trying {s}", .{ model orelse "default", why.message, candidates[n + 1] orelse "default" });
            continue;
        };
        return;
    }
    return failed(why, "no speech model");
}

/// `wanted` when `model` has that voice (or lists none, or is the text_to_speech default, which picks its own), else ""
/// (the model's default voice).
fn voiceFor(app: *App, model: ?[]const u8, wanted: []const u8) []const u8 {
    const c = app.cfg.tool(model orelse return wanted) orelse return wanted;
    if (wanted.len == 0 or engine.voices(c) == null or engine.hasVoice(c, wanted)) return wanted;
    std.log.warn("voice: {s} has no voice '{s}'; speaking in its default voice", .{ c.id, wanted });
    return "";
}

/// The speech job, streamed when the model streams (its `stream` option): its audio then reaches `sink` as it is made.
fn speakWith(app: *App, a: std.mem.Allocator, model: ?[]const u8, voice: []const u8, text: []const u8, sink: backend.PcmSink, stop: *Stop, why: *Why) backend.Error!void {
    var ref: routes.Refusal = .{};
    const streams = if (model) |m| (if (app.cfg.tool(m)) |c| engine.optionBool(c, "stream", false) else false) else false;
    const j = speech.startSpeech(app, a, .{ .model = model, .input = text, .voice = voice, .stream = streams }, &ref) orelse return failed(why, ref.message);
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
