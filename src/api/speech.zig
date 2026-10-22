//! The speech routes, OpenAI's Audio API: POST /v1/audio/speech (text to speech; the tool's WAV, or another format
//! encoded here) and POST /v1/audio/transcriptions (speech to text; json, text, verbose_json, srt or vtt). Both are
//! synchronous jobs, like image generation. Uploads become the 16 kHz mono WAV every speech_to_text tool takes
//! (`media/audio.zig`), one decode at a time, before the job is queued, so its length is known when it is priced.
//! A speech tool whose options say `stream` streams: `stream: true` sends raw PCM as HTTP/1.1 chunks as the tool writes
//! it (`streamSpeech`). Not offered, and refused with a reason: translation, SSE streams, and speech `instructions`.
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
const posix = std.posix;

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
    stream: bool = false,
    references: ?std.json.Value = null, // [{text, vq_codes}], for a tool whose options say `references`
};

/// Validates and queues a text to speech job; null with `why` set on refusal. `references` become the job's
/// `references.json`, allocated in `a`.
pub fn startSpeech(app: *App, a: std.mem.Allocator, b: SpeechArgs, why: *Refusal) ?*jobs.Job {
    const tool = routes.resolveSpeech(app, b.model, b.voice, why) orelse return null;
    if (b.instructions.len > 0) {
        why.message = std.fmt.bufPrint(why.buf[0..384], "model '{s}' does not take instructions; send the text to speak as input", .{app.scheduler.slots[tool].tool.config().id}) catch "instructions are not supported";
        return null;
    }
    const cfg = app.scheduler.slots[tool].tool.config();
    var refs: [1]inputs.File = undefined;
    var n: usize = 0;
    if (b.references) |v| if (!(v == .null or (v == .array and v.array.items.len == 0))) {
        refs[0] = .{ .name = "references.json", .data = referencesFile(a, v, why) orelse return null };
        n = 1;
    };
    const r: request.Request = .{ .speech = engine.speechDefaults(cfg, .{ .input = b.input, .voice = b.voice, .speed = b.speed, .preset = b.preset, .seed = b.seed orelse routes.randomSeed(app.io), .stream = b.stream, .references = if (n > 0) "references.json" else "" }) };
    return routes.enqueue(app, tool, r, refs[0..n], why);
}

/// A request's voice references as the file the tool reads: a list of `{text, vq_codes}` (a transcript and its codec
/// codes, one list of integers a codebook). Only the shape is checked here; the model refuses the rest in its own words.
fn referencesFile(a: std.mem.Allocator, v: std.json.Value, why: *Refusal) ?[]const u8 {
    const shape = "references must be a list of objects, each with text (a string) and vq_codes (a list of lists of integers)";
    if (v != .array) return bad(why, shape);
    for (v.array.items) |ref| {
        if (ref != .object) return bad(why, shape);
        const text = ref.object.get("text") orelse return bad(why, shape);
        const codes = ref.object.get("vq_codes") orelse return bad(why, shape);
        if (text != .string or codes != .array) return bad(why, shape);
        for (codes.array.items) |row| {
            if (row != .array) return bad(why, shape);
            for (row.array.items) |c| if (c != .integer) return bad(why, shape);
        }
    }
    return std.json.Stringify.valueAlloc(a, v, .{}) catch return failed(why, "out of memory");
}

fn bad(why: *Refusal, msg: []const u8) ?[]const u8 {
    refuse(why, 400, "invalid_request_error", msg);
    return null;
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
    stream: bool = false,
    references: ?std.json.Value = null,
};

/// POST /v1/audio/speech: JSON in, audio bytes out (whole, or streamed with `stream`). A tool that says why its audio
/// ended sets `x-finish-reason` (`stop`, `length`: clipped at its frame limit, `silence`: ended by its silence guard),
/// and a long-form reply `x-long-form-segments`.
pub fn speech(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
    const b = std.json.parseFromSliceLeaky(SpeechBody, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch
        return h.fail(req, a, 400, "invalid_request_error", "the body is not a valid speech request (JSON with input, voice, model)");
    const format = std.meta.stringToEnum(audio.Format, b.response_format) orelse
        return h.fail(req, a, 400, "invalid_request_error", "response_format must be wav, pcm, flac, mp3, opus or aac");
    if (!std.mem.eql(u8, b.stream_format, "audio")) return h.fail(req, a, 400, "invalid_request_error", "stream_format sse is not supported: send stream true for raw PCM chunks");
    if (b.stream and format != .pcm) return h.fail(req, a, 400, "invalid_request_error", "stream true needs response_format pcm: the other formats are encoded from the whole audio");
    var why: Refusal = .{};
    const j = startSpeech(app, a, .{ .model = b.model, .input = b.input, .voice = b.voice, .speed = b.speed, .preset = b.preset, .seed = b.seed, .instructions = b.instructions, .stream = b.stream, .references = b.references }, &why) orelse
        return h.fail(req, a, why.status, why.typ, why.message);
    if (b.stream) return streamSpeech(app, req, a, j);
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const bytes = encodeSpeech(app, a, j, format, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    var extra: [5]std.http.Header = undefined;
    var n: usize = 0;
    if (j.finish_reason.len > 0) {
        extra[n] = .{ .name = "x-finish-reason", .value = j.finish_reason };
        n += 1;
    }
    if (j.segments.len > 0) {
        extra[n] = .{ .name = "x-long-form-segments", .value = j.segments };
        n += 1;
    }
    if (format != .pcm) return h.bytesWith(req, format.contentType(), bytes, extra[0..n]);
    // Raw samples carry no header: say what they are (16-bit signed little-endian; not audio/L16, which is big-endian).
    const wav = Io.Dir.cwd().readFileAlloc(app.io, try std.fs.path.join(a, &.{ j.dir, j.files[0] }), a, .limited(1 << 30)) catch return h.fail(req, a, 500, "server_error", "could not read the model's audio");
    const w = audio.parseWav(wav).?;
    extra[n] = .{ .name = "x-sample-rate", .value = try std.fmt.allocPrint(a, "{d}", .{w.rate}) };
    extra[n + 1] = .{ .name = "x-channels", .value = try std.fmt.allocPrint(a, "{d}", .{w.channels}) };
    extra[n + 2] = .{ .name = "x-sample-format", .value = "s16le" };
    try h.bytesWith(req, format.contentType(), bytes, extra[0 .. n + 3]);
}

/// `stream: true`: the tool writes `speech.wav` as it goes, a canonical 44-byte header first (its sizes are fixed when
/// the audio is complete), and each new run of samples is sent as an HTTP/1.1 chunk of raw 16-bit PCM, with the headers
/// of `response_format: pcm`. A job that fails before any audio is an ordinary error reply. One that fails after the
/// first chunk ends the connection without the terminating chunk, so a client cannot take the audio for complete; a
/// client that leaves cancels the job. Audio cut short ends with an `x-finish-reason` trailer, a long-form reply with
/// `x-long-form-segments` too; audio that ended by itself ends with the plain last chunk.
fn streamSpeech(app: *App, req: *h.Request, a: std.mem.Allocator, j: *jobs.Job) !void {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/speech.wav", .{j.dir}, 0);
    var fd: posix.fd_t = -1;
    defer closeFd(fd);
    while (true) { // the header and the first samples, or the end of the job
        const done = j.done.isSet();
        if (fd < 0) fd = posix.openatZ(posix.AT.FDCWD, path, .{}, 0) catch -1;
        if ((fd >= 0 and fileSize(fd) > wav_header) or done) break;
        Io.sleep(app.io, .fromMilliseconds(poll_ms), .awake) catch {};
    }
    if (j.done.isSet() and j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    var head: [wav_header]u8 = undefined;
    if (fd < 0 or readAt(fd, &head, 0) != wav_header or streamRate(&head) == null) {
        if (!j.done.isSet()) app.scheduler.cancel(j);
        return h.fail(req, a, 500, "server_error", "the model wrote no streamable audio (speech.wav with a canonical 16-bit mono PCM header)");
    }
    var rate: [16]u8 = undefined;
    var send: [16 * 1024]u8 = undefined;
    var body = req.respondStreaming(&send, .{ .respond_options = .{ .extra_headers = &.{
        .{ .name = "content-type", .value = audio.Format.pcm.contentType() },
        .{ .name = "x-sample-rate", .value = std.fmt.bufPrint(&rate, "{d}", .{streamRate(&head).?}) catch unreachable },
        .{ .name = "x-channels", .value = "1" },
        .{ .name = "x-sample-format", .value = "s16le" },
        .{ .name = "trailer", .value = "x-finish-reason, x-long-form-segments" },
    } } }) catch return gone(app, j);
    var chunk: [64 * 1024]u8 = undefined;
    var at: u64 = wav_header;
    while (true) {
        const done = j.done.isSet();
        var end = fileSize(fd);
        if (done and j.status == .completed and readAt(fd, &head, 0) == wav_header)
            end = @min(end, wav_header + @as(u64, std.mem.readInt(u32, head[40..44], .little))); // the final size
        var sent = false;
        while (end > at and end - at >= 2) {
            const got = readAt(fd, chunk[0..@intCast(@min(chunk.len, end - at))], at) & ~@as(usize, 1); // whole samples
            if (got == 0) break;
            body.writer.writeAll(chunk[0..got]) catch return gone(app, j);
            at += got;
            sent = true;
        }
        if (sent) {
            body.writer.flush() catch return gone(app, j);
            body.flush() catch return gone(app, j);
        }
        if (done) break;
        Io.sleep(app.io, .fromMilliseconds(poll_ms), .awake) catch {};
    }
    if (j.status != .completed) return error.StreamIncomplete; // no terminating chunk: the client sees an incomplete body
    var trailers: [2]std.http.Header = undefined;
    var n: usize = 0;
    const cut = j.finish_reason.len > 0 and !std.mem.eql(u8, j.finish_reason, "stop");
    if (cut or (j.segments.len > 0 and j.finish_reason.len > 0)) {
        trailers[n] = .{ .name = "x-finish-reason", .value = j.finish_reason };
        n += 1;
    }
    if (j.segments.len > 0) {
        trailers[n] = .{ .name = "x-long-form-segments", .value = j.segments };
        n += 1;
    }
    body.endChunked(.{ .trailers = trailers[0..n] }) catch return gone(app, j);
}

const wav_header = 44;
const poll_ms = 20;

/// The sample rate of a canonical header of 16-bit mono PCM (fmt then data, nothing between), else null.
fn streamRate(head: *const [wav_header]u8) ?u32 {
    if (!std.mem.eql(u8, head[0..4], "RIFF") or !std.mem.eql(u8, head[8..16], "WAVEfmt ") or !std.mem.eql(u8, head[36..40], "data")) return null;
    if (std.mem.readInt(u32, head[16..20], .little) != 16 or std.mem.readInt(u16, head[20..22], .little) != 1) return null;
    if (std.mem.readInt(u16, head[22..24], .little) != 1 or std.mem.readInt(u16, head[34..36], .little) != 16) return null;
    const rate = std.mem.readInt(u32, head[24..28], .little);
    return if (rate > 0) rate else null;
}

fn closeFd(fd: posix.fd_t) void {
    if (fd >= 0) _ = posix.system.close(fd);
}

fn fileSize(fd: posix.fd_t) u64 {
    const n = posix.system.lseek(fd, 0, posix.SEEK.END);
    return if (n > 0) @intCast(n) else 0;
}

/// Bytes read into `buf` from `offset`; 0 on error.
fn readAt(fd: posix.fd_t, buf: []u8, offset: u64) usize {
    const rc = posix.system.pread(fd, buf.ptr, buf.len, @intCast(offset));
    if (posix.errno(rc) != .SUCCESS) return 0;
    return @intCast(rc);
}

/// The client left mid-stream: stop the job too.
fn gone(app: *App, j: *jobs.Job) error{ClientGone} {
    app.scheduler.cancel(j);
    return error.ClientGone;
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

test "a streamed reply needs a canonical 16-bit mono PCM header; its sizes may be placeholders" {
    var head: [wav_header]u8 = undefined;
    @memcpy(head[0..4], "RIFF");
    std.mem.writeInt(u32, head[4..8], 0xFFFFFFFF, .little);
    @memcpy(head[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, head[16..20], 16, .little);
    std.mem.writeInt(u16, head[20..22], 1, .little);
    std.mem.writeInt(u16, head[22..24], 1, .little);
    std.mem.writeInt(u32, head[24..28], 44100, .little);
    std.mem.writeInt(u32, head[28..32], 88200, .little);
    std.mem.writeInt(u16, head[32..34], 2, .little);
    std.mem.writeInt(u16, head[34..36], 16, .little);
    @memcpy(head[36..40], "data");
    std.mem.writeInt(u32, head[40..44], 0xFFFFFFDB, .little);
    try std.testing.expectEqual(@as(?u32, 44100), streamRate(&head));
    std.mem.writeInt(u16, head[22..24], 2, .little); // stereo
    try std.testing.expectEqual(@as(?u32, null), streamRate(&head));
    std.mem.writeInt(u16, head[22..24], 1, .little);
    @memcpy(head[36..40], "LIST"); // another chunk before the samples
    try std.testing.expectEqual(@as(?u32, null), streamRate(&head));
}
