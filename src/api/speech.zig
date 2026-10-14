//! The speech routes, OpenAI's Audio API: POST /v1/audio/speech (text to speech; the tool's WAV, or another format
//! encoded here) and POST /v1/audio/transcriptions (speech to text; json, text, verbose_json, srt or vtt). Both are
//! synchronous jobs, like image generation. Uploads become the 16 kHz mono WAV every speech_to_text tool takes
//! (`media/audio.zig`), one decode at a time, before the job is queued, so its length is known when it is priced.
//! Not offered, and refused with a reason: translation, streamed responses, and speech `instructions`.
//! The MCP tools (`mcp.zig`) share `startSpeech`, `startTranscription`, `readTranscript` and `encodeSpeech`.

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const routes = @import("routes.zig");
const request = @import("../tool/request.zig");
const jobs = @import("../sched/jobs.zig");
const inputs = @import("inputs.zig");
const engine = @import("../engine/engine.zig");
const audio = @import("../media/audio.zig");

const App = routes.App;

/// Why a speech job was not started: the HTTP status, OpenAI's error type and the message.
pub const Refusal = routes.Refusal;

fn refuse(why: *Refusal, status: u16, typ: []const u8, msg: []const u8) void {
    why.status = status;
    why.typ = typ;
    why.message = msg;
}

// ---------------------------------------------------------------- text to speech

pub const SpeechArgs = struct {
    model: ?[]const u8 = null,
    input: []const u8 = "",
    voice: []const u8 = "",
    speed: f32 = 1,
    preset: []const u8 = "",
    seed: ?u64 = null,
    instructions: []const u8 = "",
};

/// Validates and queues a text to speech job; null with `why` set on refusal.
pub fn startSpeech(app: *App, b: SpeechArgs, why: *Refusal) ?*jobs.Job {
    const tool = routes.resolve(app, b.model, .text_to_speech, why) orelse return null;
    if (b.instructions.len > 0) {
        why.message = std.fmt.bufPrint(why.buf[0..384], "model '{s}' does not take instructions; send the text to speak as input", .{app.scheduler.slots[tool].tool.config().id}) catch "instructions are not supported";
        return null;
    }
    const cfg = app.scheduler.slots[tool].tool.config();
    const r: request.Request = .{ .speech = engine.speechDefaults(cfg, .{ .input = b.input, .voice = b.voice, .speed = b.speed, .preset = b.preset, .seed = b.seed orelse routes.randomSeed(app.io) }) };
    return routes.enqueue(app, tool, r, &.{}, why);
}

/// The finished job's `speech.wav` in `format`, in `a`.
pub fn encodeSpeech(app: *App, a: std.mem.Allocator, j: *jobs.Job, format: audio.Format, why: *Refusal) ?[]const u8 {
    if (j.files.len == 0) return failed(why, "the model wrote no audio");
    const wav = Io.Dir.cwd().readFileAlloc(app.io, std.fs.path.join(a, &.{ j.dir, j.files[0] }) catch return failed(why, "out of memory"), a, .limited(1 << 30)) catch
        return failed(why, "could not read the model's audio");
    return audio.encode(app.io, app.gpa, a, j.dir, wav, format, .{}) catch |err| switch (err) {
        error.OutOfMemory => failed(why, "out of memory"),
        error.BadWav => failed(why, "the model wrote audio that is not 16-bit PCM WAV"),
        error.EncoderFailed => failed(why, "encoding the audio failed (is ffmpeg with this codec installed?)"),
    };
}

fn failed(why: *Refusal, msg: []const u8) ?[]const u8 {
    refuse(why, 500, "server_error", msg);
    return null;
}

const SpeechBody = struct {
    model: ?[]const u8 = null,
    input: []const u8 = "",
    voice: []const u8 = "",
    speed: f32 = 1,
    preset: []const u8 = "",
    seed: ?u64 = null,
    instructions: []const u8 = "",
    response_format: []const u8 = "wav",
    stream_format: []const u8 = "audio",
};

/// POST /v1/audio/speech: JSON in, audio bytes out.
pub fn speech(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const b = std.json.parseFromSliceLeaky(SpeechBody, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch
        return h.fail(req, a, 400, "invalid_request_error", "the body is not a valid speech request (JSON with input, voice, model)");
    const format = std.meta.stringToEnum(audio.Format, b.response_format) orelse
        return h.fail(req, a, 400, "invalid_request_error", "response_format must be wav, pcm, flac, mp3, opus or aac");
    if (!std.mem.eql(u8, b.stream_format, "audio")) return h.fail(req, a, 400, "invalid_request_error", "stream_format sse is not supported: the audio comes back whole");
    var why: Refusal = .{};
    const j = startSpeech(app, .{ .model = b.model, .input = b.input, .voice = b.voice, .speed = b.speed, .preset = b.preset, .seed = b.seed, .instructions = b.instructions }, &why) orelse
        return h.fail(req, a, why.status, why.typ, why.message);
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const bytes = encodeSpeech(app, a, j, format, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    if (format != .pcm) return h.bytes(req, format.contentType(), bytes);
    // Raw samples carry no header: say what they are (16-bit signed little-endian; not audio/L16, which is big-endian).
    const wav = Io.Dir.cwd().readFileAlloc(app.io, try std.fs.path.join(a, &.{ j.dir, j.files[0] }), a, .limited(1 << 30)) catch return h.fail(req, a, 500, "server_error", "could not read the model's audio");
    const w = audio.parseWav(wav).?;
    try h.bytesWith(req, format.contentType(), bytes, &.{
        .{ .name = "x-sample-rate", .value = try std.fmt.allocPrint(a, "{d}", .{w.rate}) },
        .{ .name = "x-channels", .value = try std.fmt.allocPrint(a, "{d}", .{w.channels}) },
        .{ .name = "x-sample-format", .value = "s16le" },
    });
}

// ---------------------------------------------------------------- speech to text

pub const TranscriptionArgs = struct {
    model: ?[]const u8 = null,
    language: []const u8 = "",
    prompt: []const u8 = "",
    temperature: f32 = 0,
    timestamps: request.Timestamps = .none,
    seed: ?u64 = null,
};

/// Decodes `upload` and queues a transcription job; null with `why` set on refusal. The model is resolved first, so
/// an unknown one costs no decode; uploads decode one at a time.
pub fn startTranscription(app: *App, a: std.mem.Allocator, b: TranscriptionArgs, upload: []const u8, why: *Refusal) ?*jobs.Job {
    const tool = routes.resolve(app, b.model, .speech_to_text, why) orelse return null;
    if (!engine.supports(app.scheduler.slots[tool].tool.config(), .speech_to_text)) {
        _ = routes.enqueue(app, tool, .{ .transcription = .{ .audio = "audio.wav" } }, &.{}, why); // refuses, naming the capability
        return null;
    }
    const cfg = app.scheduler.slots[tool].tool.config();
    const max_s: u64 = @min(engine.optionInt(cfg, "max_audio_s", 30 * 60), 30 * 60);
    const limits: audio.Limits = .{ .max_ms = max_s * 1000 };
    app.decode_mutex.lock(app.io) catch {
        refuse(why, 503, "server_error", "shutting down");
        return null;
    };
    var msg: []const u8 = "";
    const n = audio.normalize(app.io, app.gpa, a, app.jobs_dir, upload, limits, &msg);
    app.decode_mutex.unlock(app.io);
    const decoded = n catch |err| {
        switch (err) {
            error.Invalid => refuse(why, 400, "invalid_request_error", msg),
            error.OutOfMemory => refuse(why, 503, "insufficient_memory", "out of memory decoding the audio"),
            error.DecoderFailed => refuse(why, 500, "server_error", "could not run ffmpeg to decode the audio"),
        }
        return null;
    };
    const lang = std.ascii.allocLowerString(a, std.mem.trim(u8, b.language, " ")) catch "";
    const r: request.Request = .{ .transcription = .{ .audio = "audio.wav", .duration_ms = decoded.ms, .language = lang, .prompt = b.prompt, .temperature = b.temperature, .timestamps = b.timestamps, .seed = b.seed orelse routes.randomSeed(app.io) } };
    return routes.enqueue(app, tool, r, &.{.{ .name = "audio.wav", .data = decoded.wav }}, why);
}

/// The finished job's transcript (`request.Transcript`, with whatever else the tool wrote kept), as a JSON object.
pub fn readTranscript(app: *App, a: std.mem.Allocator, j: *jobs.Job, why: *Refusal) ?std.json.ObjectMap {
    for (j.files) |f| {
        if (!std.mem.eql(u8, f, "transcript.json")) continue;
        const text = Io.Dir.cwd().readFileAlloc(app.io, std.fs.path.join(a, &.{ j.dir, f }) catch break, a, .limited(64 << 20)) catch break;
        _ = std.json.parseFromSliceLeaky(request.Transcript, a, text, .{ .ignore_unknown_fields = true }) catch break; // the fields we rely on are there
        const v = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch break;
        if (v == .object) return v.object;
    }
    refuse(why, 500, "server_error", "the model wrote no valid transcript.json");
    return null;
}

const TranscriptionBody = struct {
    model: ?[]const u8 = null,
    language: []const u8 = "",
    prompt: []const u8 = "",
    temperature: f32 = 0,
    seed: ?u64 = null,
    response_format: []const u8 = "json",
    stream: bool = false,
};

const ResponseFormat = enum { json, text, verbose_json, srt, vtt };

/// POST /v1/audio/transcriptions: a multipart `file` (or JSON `file` as base64, a data URL or a URL) in, the
/// transcript out.
pub fn transcriptions(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, ctype: []const u8) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    var msg: []const u8 = "the body is not a valid transcription request";
    var raws: std.ArrayList([]const u8) = .empty;
    const b = routes.parseBody(TranscriptionBody, app, a, host, text, ctype, &.{"file"}, &raws, &msg) catch return h.fail(req, a, 400, "invalid_request_error", msg);
    if (b.stream) return h.fail(req, a, 400, "invalid_request_error", "stream is not supported: the transcript comes back whole");
    const format = std.meta.stringToEnum(ResponseFormat, b.response_format) orelse
        return h.fail(req, a, 400, "invalid_request_error", "response_format must be json, text, verbose_json, srt or vtt");
    if (raws.items.len != 1) return h.fail(req, a, 400, "invalid_request_error", "file is required: send one audio file as the multipart field file");
    if (raws.items[0].len > inputs.max_bytes) return h.fail(req, a, 413, "invalid_request_error", "the audio file is larger than 32 MB");
    var granularities = granularitiesOf(a, text, ctype) catch return h.fail(req, a, 400, "invalid_request_error", "timestamp_granularities must be segment and/or word");
    if (granularities != .none and format != .verbose_json) return h.fail(req, a, 400, "invalid_request_error", "timestamp_granularities needs response_format verbose_json");
    if (granularities == .none and format != .json and format != .text) granularities = .segment; // verbose_json, srt and vtt carry segments
    var why: Refusal = .{};
    const j = startTranscription(app, a, .{ .model = b.model, .language = b.language, .prompt = b.prompt, .temperature = b.temperature, .timestamps = granularities, .seed = b.seed }, raws.items[0], &why) orelse
        return h.fail(req, a, why.status, why.typ, why.message);
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const t = readTranscript(app, a, j, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    const transcript = t.get("text").?.string;
    const duration = number(t.get("duration")) orelse @as(f64, @floatFromInt(j.request.transcription.duration_ms)) / 1000;
    switch (format) {
        .json => try h.jsonValue(req, a, .ok, .{ .text = transcript, .usage = .{ .type = "duration", .seconds = @as(u64, @intFromFloat(@ceil(duration))) } }),
        .text => try h.bytes(req, "text/plain; charset=utf-8", try std.fmt.allocPrint(a, "{s}\n", .{transcript})),
        .verbose_json => {
            var obj: std.json.ObjectMap = .empty;
            try obj.put(a, "task", .{ .string = "transcribe" });
            try obj.put(a, "language", t.get("language") orelse .{ .string = "" });
            try obj.put(a, "duration", .{ .float = duration });
            try obj.put(a, "text", .{ .string = transcript });
            try obj.put(a, "segments", t.get("segments") orelse .{ .array = .init(a) });
            if (granularities == .word) try obj.put(a, "words", t.get("words") orelse .{ .array = .init(a) });
            try h.jsonValue(req, a, .ok, std.json.Value{ .object = obj });
        },
        .srt, .vtt => try h.bytes(req, if (format == .srt) "application/x-subrip; charset=utf-8" else "text/vtt; charset=utf-8", try subtitles(a, t, format == .vtt)),
    }
}

/// `timestamp_granularities[]` (multipart, repeated) or `timestamp_granularities` (JSON array): the finest asked for.
fn granularitiesOf(a: std.mem.Allocator, text: []const u8, ctype: []const u8) !request.Timestamps {
    var out: request.Timestamps = .none;
    var names: std.ArrayList([]const u8) = .empty;
    if (inputs.Form.boundary(ctype)) |bound| {
        const form = inputs.Form.parse(a, text, bound) catch return out;
        for (form.parts) |p| if (p.filename == null and (std.mem.eql(u8, p.name, "timestamp_granularities[]") or std.mem.eql(u8, p.name, "timestamp_granularities"))) try names.append(a, std.mem.trim(u8, p.data, " "));
    } else {
        const v = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return out;
        if (v == .object) if (v.object.get("timestamp_granularities")) |g| {
            if (g != .array) return error.Invalid;
            for (g.array.items) |x| try names.append(a, if (x == .string) x.string else return error.Invalid);
        };
    }
    for (names.items) |n| {
        if (std.mem.eql(u8, n, "word")) out = .word else if (std.mem.eql(u8, n, "segment")) {
            if (out == .none) out = .segment;
        } else return error.Invalid;
    }
    return out;
}

fn number(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

/// SubRip or WebVTT cues, one a segment.
fn subtitles(a: std.mem.Allocator, t: std.json.ObjectMap, vtt: bool) ![]const u8 {
    var w: Io.Writer.Allocating = .init(a);
    if (vtt) try w.writer.writeAll("WEBVTT\n\n");
    const segs = t.get("segments") orelse return w.written();
    if (segs != .array) return w.written();
    var n: usize = 0;
    for (segs.array.items) |s| {
        if (s != .object) continue;
        const seg_text = s.object.get("text") orelse continue;
        if (seg_text != .string) continue;
        const body = std.mem.trim(u8, seg_text.string, " ");
        if (body.len == 0) continue;
        n += 1;
        if (!vtt) try w.writer.print("{d}\n", .{n});
        try stamp(&w.writer, number(s.object.get("start")) orelse 0, vtt);
        try w.writer.writeAll(" --> ");
        try stamp(&w.writer, number(s.object.get("end")) orelse 0, vtt);
        try w.writer.print("\n{s}\n\n", .{body});
    }
    return w.written();
}

/// HH:MM:SS,mmm (SubRip) or HH:MM:SS.mmm (WebVTT).
fn stamp(w: *Io.Writer, seconds: f64, vtt: bool) !void {
    const ms: u64 = @intFromFloat(@round(@max(seconds, 0) * 1000));
    try w.print("{d:0>2}:{d:0>2}:{d:0>2}{c}{d:0>3}", .{ ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, @as(u8, if (vtt) '.' else ','), ms % 1000 });
}

test "subtitles from segments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a,
        \\{"text": "Hello there. Bye.", "segments": [{"id": 0, "start": 0, "end": 1.5, "text": " Hello there."},
        \\  {"id": 1, "start": 1.5, "end": 3, "text": " "}, {"id": 2, "start": 3661.25, "end": 3662, "text": " Bye."}]}
    , .{});
    try std.testing.expectEqualStrings("1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n2\n01:01:01,250 --> 01:01:02,000\nBye.\n\n", try subtitles(a, v.object, false));
    try std.testing.expectEqualStrings("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello there.\n\n01:01:01.250 --> 01:01:02.000\nBye.\n\n", try subtitles(a, v.object, true));
}

test "timestamp granularities from a form or JSON" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const form = "--b\r\nContent-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\nsegment\r\n--b\r\nContent-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\nword\r\n--b--\r\n";
    try std.testing.expectEqual(request.Timestamps.word, try granularitiesOf(a, form, "multipart/form-data; boundary=b"));
    try std.testing.expectEqual(request.Timestamps.segment, try granularitiesOf(a, "{\"timestamp_granularities\": [\"segment\"]}", "application/json"));
    try std.testing.expectEqual(request.Timestamps.none, try granularitiesOf(a, "{}", "application/json"));
    try std.testing.expectError(error.Invalid, granularitiesOf(a, "{\"timestamp_granularities\": [\"phoneme\"]}", "application/json"));
}
