//! The voices routes: a voice cloned from someone's own recordings (`voices.zig`), for any speech model with the
//! `voice_clone` capability, then spoken like any named voice (speech, voice sessions, MCP).
//!
//! - POST /v1/voices: clips (multipart `file`, repeated; WAV, MP3, WebM, Ogg, FLAC, M4A), `consent=true`, optional
//!   `transcript` (one per clip, in order; none, or a blank one: the transcription model writes it), `name` (none: a draft, kept an
//!   hour, for a preview), `model`, `language`, `source` (`owner` or `consented`) and `note`. The clips are decoded,
//!   checked and levelled (`media/voiceprep.zig`), transcribed, and encoded by the model (a `Speech.clone` job); the
//!   clips are deleted as soon as the codes are back.
//! - GET /v1/voices[?model=ID]: every voice, the configured ones (`kind: named`) and the cloned (`kind: cloned`).
//! - GET /v1/voices/{id}, PATCH /v1/voices/{id} `{"name"}` (saves a draft, or renames), DELETE /v1/voices/{id}.
//!
//! Creating, renaming and deleting need `manage`; listing needs `read` (`routes.need`); speaking needs `media`.

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const routes = @import("routes.zig");
const speech = @import("speech.zig");
const inputs = @import("inputs.zig");
const pair = @import("pair.zig");
const request = @import("../tool/request.zig");
const engine = @import("../engine/engine.zig");
const audio = @import("../media/audio.zig");
const voiceprep = @import("../media/voiceprep.zig");
const store = @import("../voices.zig");
const jobs = @import("../sched/jobs.zig");
const sched = @import("../sched/scheduler.zig");

const App = routes.App;
const Refusal = routes.Refusal;

pub const max_clips = 8;
const max_clip_ms = 120_000; // one upload, before trimming
const consent_statement = "The person cloning this voice confirmed it is their own voice, or one whose owner consented on record.";

pub fn route(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, ctype: []const u8, rest: []const u8, caller: pair.Caller) !void {
    const m = req.head.method;
    if (rest.len == 0) return switch (m) {
        .GET => list(app, req, a),
        .POST => create(app, req, a, host, ctype, caller),
        else => h.fail(req, a, 405, "invalid_request_error", "method not allowed"),
    };
    const now = sched.nowUnix(app.io);
    switch (m) {
        .GET => {
            const found = try app.cloned.list(app.io, a, null, rest, now);
            if (found.len == 0) return h.fail(req, a, 404, "invalid_request_error", "no such voice");
            return h.jsonValue(req, a, .ok, found[0]);
        },
        .PATCH => {
            const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
            const b = std.json.parseFromSliceLeaky(struct { name: []const u8 = "" }, a, text, .{ .ignore_unknown_fields = true }) catch
                return h.fail(req, a, 400, "invalid_request_error", "send {\"name\": \"...\"}");
            if (try nameProblem(app, a, b.name, rest)) |p| return h.fail(req, a, p.status, "invalid_request_error", p.message);
            const v = app.cloned.rename(app.io, a, rest, b.name, now) catch |err| return storeFailure(req, a, err);
            return h.jsonValue(req, a, .ok, v);
        },
        .DELETE => {
            app.cloned.remove(app.io, rest) catch |err| return storeFailure(req, a, err);
            return h.jsonValue(req, a, .ok, .{ .object = "voice.deleted", .id = rest, .deleted = true });
        },
        else => return h.fail(req, a, 405, "invalid_request_error", "method not allowed"),
    }
}

fn storeFailure(req: *h.Request, a: std.mem.Allocator, err: store.Error) !void {
    return switch (err) {
        error.NotFound => h.fail(req, a, 404, "invalid_request_error", "no such voice"),
        error.Taken => h.fail(req, a, 409, "invalid_request_error", "a voice of that name exists: pick another"),
        error.OutOfMemory => h.fail(req, a, 503, "insufficient_memory", "out of memory"),
        error.WriteFailed => h.fail(req, a, 500, "server_error", "could not save the voice file"),
    };
}

const Problem = struct { status: u16, message: []const u8 };

/// Why `name` cannot name a voice (`except`: the voice being renamed), else null.
fn nameProblem(app: *App, a: std.mem.Allocator, name: []const u8, except: []const u8) !?Problem {
    if (!store.validName(name)) return .{ .status = 400, .message = "name must be 1 to 64 of A-Z a-z 0-9 . _ - starting with a letter or digit, not default, not draft-..." };
    for (app.scheduler.slots) |slot| {
        const c = slot.tool.config();
        if (c.kind != .speech) continue;
        for (engine.voices(c) orelse &.{}) |v| if (v == .string and std.ascii.eqlIgnoreCase(v.string, name))
            return .{ .status = 409, .message = try std.fmt.allocPrint(a, "'{s}' is one of model '{s}''s own voices: pick another name", .{ v.string, c.id }) };
    }
    if (!std.ascii.eqlIgnoreCase(name, except) and app.cloned.taken(app.io, name)) return .{ .status = 409, .message = "a voice of that name exists: pick another" };
    return null;
}

/// GET /v1/voices: the configured voices of every speech model, then the cloned ones; `?model=` keeps one model's.
fn list(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    const want = if (@import("realtime.zig").query(req.head.target, "model")) |q| try decoded(a, q) else null;
    var out: Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.writeAll("{\"object\":\"list\",\"data\":[");
    var first = true;
    const Named = struct { object: []const u8 = "voice", id: []const u8, name: []const u8, model: []const u8, kind: []const u8 = "named", default: bool };
    for (app.scheduler.slots) |slot| {
        const c = slot.tool.config();
        if (c.kind != .speech or (want != null and !std.mem.eql(u8, want.?, c.id))) continue;
        const dv = engine.optionString(c, "default_voice", "");
        for (engine.voices(c) orelse &.{}) |v| if (v == .string) {
            if (!first) try w.writeByte(',');
            first = false;
            try std.json.Stringify.value(Named{ .id = v.string, .name = v.string, .model = c.id, .default = std.mem.eql(u8, v.string, dv) }, .{}, w);
        };
    }
    for (try app.cloned.list(app.io, a, want, null, sched.nowUnix(app.io))) |v| {
        if (!first) try w.writeByte(',');
        first = false;
        try std.json.Stringify.value(v, .{}, w);
    }
    try w.writeAll("]}");
    try h.json(req, .ok, out.written());
}

fn decoded(a: std.mem.Allocator, q: []const u8) ![]const u8 {
    const buf = try a.dupe(u8, q);
    return std.Uri.percentDecodeInPlace(buf);
}

const Form = struct {
    model: ?[]const u8 = null,
    name: []const u8 = "",
    consent: bool = false,
    language: []const u8 = "",
    source: []const u8 = "owner",
    note: []const u8 = "",
};

/// POST /v1/voices.
fn create(app: *App, req: *h.Request, a: std.mem.Allocator, host: []const u8, ctype: []const u8, caller: pair.Caller) !void {
    const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "the clips are larger than 64 MB in all: send shorter clips");
    var msg: []const u8 = "the body is not a valid voice request (multipart: file, consent, name, transcript)";
    var raws: std.ArrayList([]const u8) = .empty;
    const b = routes.parseBody(Form, app, a, host, text, ctype, &.{ "file", "file[]", "files" }, &raws, &msg) catch return h.fail(req, a, 400, "invalid_request_error", msg);
    if (!b.consent) return h.fail(req, a, 400, "invalid_request_error", "consent is required: send consent=true to confirm the voice is your own, or that its owner consented on record (the model's licence forbids cloning anyone else)");
    const kind = if (std.mem.eql(u8, b.source, "owner") or std.mem.eql(u8, b.source, "consented")) b.source else return h.fail(req, a, 400, "invalid_request_error", "source must be owner or consented");
    if (std.mem.eql(u8, kind, "consented") and std.mem.trim(u8, b.note, " \t\r\n").len == 0) return h.fail(req, a, 400, "invalid_request_error", "a consented voice needs a note saying where the owner's consent is recorded");
    if (b.note.len > 1024 or !std.unicode.utf8ValidateSlice(b.note)) return h.fail(req, a, 400, "invalid_request_error", "note must be UTF-8 text of at most 1024 bytes");
    if (raws.items.len == 0) return h.fail(req, a, 400, "invalid_request_error", "file is required: send one or more recordings as the multipart field file (10 to 60 s of speech in all)");
    if (raws.items.len > max_clips) return h.fail(req, a, 400, "invalid_request_error", "at most 8 clips");
    for (raws.items, 1..) |r, n| if (r.len > inputs.max_bytes) return h.fail(req, a, 413, "invalid_request_error", try std.fmt.allocPrint(a, "clip {d} is larger than 32 MB", .{n}));
    const transcripts = transcriptsOf(a, text, ctype) catch return h.fail(req, a, 400, "invalid_request_error", "transcript must be text (one per clip)");
    if (transcripts.len > 0 and transcripts.len != raws.items.len) return h.fail(req, a, 400, "invalid_request_error", try std.fmt.allocPrint(a, "send one transcript per clip ({d} clips, {d} transcripts), or none", .{ raws.items.len, transcripts.len }));
    for (transcripts, 1..) |t, n| if (given(t)) if (textProblem(t)) |p| return h.fail(req, a, 400, "invalid_request_error", try std.fmt.allocPrint(a, "transcript {d} {s}", .{ n, p }));
    if (b.name.len > 0) if (try nameProblem(app, a, b.name, "")) |p| return h.fail(req, a, p.status, "invalid_request_error", p.message);

    var why: Refusal = .{};
    const tool = routes.resolve(app, b.model, .voice_clone, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    const cfg = app.scheduler.slots[tool].tool.config();
    if (!engine.supports(cfg, .voice_clone)) return h.fail(req, a, 400, "invalid_request_error", try std.fmt.allocPrint(a, "model '{s}' cannot clone a voice; models that can list voice_clone in /v1/models", .{cfg.id}));
    var gate_buf: [512]u8 = undefined;
    if (app.unqualified(a, cfg, &gate_buf)) |gate| return h.fail(req, a, 503, "server_error", gate);

    // decode and prepare, one decode at a time
    const clips = try a.alloc(voiceprep.Clip, raws.items.len);
    for (clips, raws.items, 1..) |*c, raw, n| {
        app.decode_mutex.lock(app.io) catch return h.fail(req, a, 503, "server_error", "shutting down");
        var dmsg: []const u8 = "";
        const samples = audio.decodeMono(app.io, app.gpa, a, app.jobs_dir, raw, voiceprep.rate, .{ .max_ms = max_clip_ms }, &dmsg);
        app.decode_mutex.unlock(app.io);
        c.* = .{ .samples = samples catch |err| return switch (err) {
            error.Invalid => h.fail(req, a, 400, "invalid_request_error", try std.fmt.allocPrint(a, "clip {d}: {s}", .{ n, if (std.mem.eql(u8, dmsg, "the audio is longer than this model takes")) "longer than 2 minutes: send shorter clips" else dmsg })),
            error.OutOfMemory => h.fail(req, a, 503, "insufficient_memory", "out of memory decoding the clips"),
            error.DecoderFailed => h.fail(req, a, 500, "server_error", "could not run ffmpeg to decode the clips"),
        }, .transcript = transcripts.len > 0 and given(transcripts[n - 1]) };
    }
    var pmsg: []const u8 = "";
    const prepared = voiceprep.prepare(a, clips, &pmsg) catch |err| return switch (err) {
        error.Invalid => h.fail(req, a, 400, "invalid_request_error", pmsg),
        error.OutOfMemory => h.fail(req, a, 503, "insufficient_memory", "out of memory preparing the clips"),
    };

    // the references' words: as given (a clip with a transcript is one reference), else transcribed
    const owner = caller.owner();
    const refs = prepared.references;
    const texts = try a.alloc([]const u8, refs.len);
    const wavs = try a.alloc([]const u8, refs.len);
    for (refs, wavs) |r, *wv| wv.* = try audio.wav(a, voiceprep.rate, 1, std.mem.sliceAsBytes(r.samples));
    for (refs, texts, wavs, 1..) |r, *t, wv, n| {
        if (clips[r.clip].transcript) {
            t.* = std.mem.trim(u8, transcripts[r.clip], " \t\r\n");
            continue;
        }
        t.* = transcribe(app, a, wv, b.language, owner, &why) orelse return h.fail(req, a, why.status, why.typ, try std.fmt.allocPrint(a, "transcribing reference {d}: {s}", .{ n, why.message }));
        if (std.mem.trim(u8, t.*, " \t\r\n.").len == 0) return h.fail(req, a, 400, "invalid_request_error", try std.fmt.allocPrint(a, "no words were heard in reference {d} (clip {d}): speak clearly, closer to the microphone", .{ n, r.clip + 1 }));
        if (textProblem(t.*)) |p| return h.fail(req, a, 400, "invalid_request_error", try std.fmt.allocPrint(a, "the transcript of reference {d} {s}", .{ n, p }));
    }

    // the encode: a voice_clone job on the model, its clips deleted with its directory once the codes are read
    const Manifest = struct { name: []const u8, references: []const struct { audio: []const u8, text: []const u8 } };
    const entries = try a.alloc(@typeInfo(@FieldType(Manifest, "references")).pointer.child, refs.len);
    const files = try a.alloc(inputs.File, refs.len + 1);
    for (entries, files[0..refs.len], texts, wavs, 0..) |*e, *f, t, wv, i| {
        const file_name = try std.fmt.allocPrint(a, "ref{d}.wav", .{i});
        e.* = .{ .audio = file_name, .text = t };
        f.* = .{ .name = file_name, .data = wv };
    }
    var rnd: [4]u8 = undefined;
    app.io.random(&rnd);
    const id = if (b.name.len > 0) b.name else try std.fmt.allocPrint(a, "{s}{x:0>8}", .{ store.draft_prefix, std.mem.readInt(u32, &rnd, .little) });
    files[refs.len] = .{ .name = "clone.json", .data = try std.json.Stringify.valueAlloc(a, Manifest{ .name = id, .references = entries }, .{}) };
    const j = routes.enqueue(app, tool, .{ .speech = .{ .input = "", .clone = "clone.json" } }, files, owner, &why) orelse return h.fail(req, a, why.status, why.typ, why.message);
    defer scrub(app, j);
    j.done.wait(app.io) catch return;
    if (j.status != .completed) return h.fail(req, a, j.http_status, j.error_type, j.error_message);
    const codes = readCodes(app, a, j, refs.len) orelse return h.fail(req, a, 500, "server_error", "the model wrote no valid voice.json");

    const now = sched.nowUnix(app.io);
    const stored = try a.alloc(store.Reference, refs.len);
    const prep = try a.alloc(store.ReferenceInfo, refs.len);
    for (stored, prep, refs, codes.references, texts) |*s, *p, r, c, t| {
        s.* = .{ .text = t, .seconds = r.seconds, .frames = c.frames, .vq_codes = c.vq_codes };
        p.* = .{ .text = t, .seconds = r.seconds, .frames = c.frames, .clip = r.clip, .level_dbfs = r.level_dbfs, .gain_db = r.gain_db, .snr_db = r.snr_db };
    }
    const clip_info = try a.alloc(store.Clip, prepared.clips.len);
    for (clip_info, prepared.clips) |*o, c| o.* = .{ .seconds_in = c.seconds_in, .seconds_kept = c.seconds_kept, .speech_left_out_s = c.speech_left_out_s, .references = c.references };
    const v: store.Voice = .{
        .id = id,
        .model = cfg.id,
        .created_at = now,
        .source = .{ .kind = kind, .note = b.note },
        .consent = .{ .statement = consent_statement, .by = owner, .at = now },
        .transcribed = transcribed(clips),
        .seconds = prepared.seconds,
        .clips = clip_info,
        .prep = prep,
        .references = stored,
    };
    const saved = app.cloned.add(app.io, a, v, now) catch |err| return storeFailure(req, a, err);
    try h.jsonValue(req, a, .ok, saved);
}

/// Whether a clip's `transcript` field holds its words (blank: the transcription model writes them).
fn given(t: []const u8) bool {
    return std.mem.trim(u8, t, " \t\r\n").len > 0;
}

fn transcribed(clips: []const voiceprep.Clip) bool {
    for (clips) |c| if (!c.transcript) return true;
    return false;
}

/// Why a transcript cannot be a reference's (the model's prompt syntax, blank, too long), else null.
fn textProblem(t: []const u8) ?[]const u8 {
    if (!std.unicode.utf8ValidateSlice(t)) return "is not UTF-8 text";
    if (std.mem.trim(u8, t, " \t\r\n").len == 0) return "is blank";
    if (t.len > 4096) return "is longer than 4096 bytes";
    if (std.mem.indexOf(u8, t, "<|") != null) return "holds '<|', the model's tag syntax";
    return null;
}

/// The plain `transcript` (or `transcript[]`) fields of a multipart body in order, or a JSON body's `transcript`
/// (a string or a list of them).
fn transcriptsOf(a: std.mem.Allocator, text: []const u8, ctype: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    if (inputs.Form.boundary(ctype)) |bound| {
        const form = try inputs.Form.parse(a, text, bound);
        for (form.parts) |p| if (p.filename == null and (std.mem.eql(u8, p.name, "transcript") or std.mem.eql(u8, p.name, "transcript[]"))) try out.append(a, p.data);
        return out.items;
    }
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const t = (if (v == .object) v.object.get("transcript") else null) orelse return out.items;
    switch (t) {
        .string => |s| try out.append(a, s),
        .array => |arr| for (arr.items) |x| {
            if (x != .string) return error.Invalid;
            try out.append(a, x.string);
        },
        .null => {},
        else => return error.Invalid,
    }
    return out.items;
}

/// A reference's words from the transcription model; its job (the clip's audio) is deleted.
fn transcribe(app: *App, a: std.mem.Allocator, wav: []const u8, language: []const u8, owner: []const u8, why: *Refusal) ?[]const u8 {
    const j = speech.startTranscription(app, a, .{ .language = language }, wav, owner, why) orelse return null;
    defer scrub(app, j);
    j.done.wait(app.io) catch return null;
    if (j.status != .completed) {
        why.* = .{ .status = j.http_status, .typ = j.error_type, .message = a.dupe(u8, j.error_message) catch "the transcription failed" };
        return null;
    }
    const t = speech.readTranscript(app, a, j, why) orelse return null;
    const said = t.get("text") orelse return null;
    return if (said == .string) std.mem.trim(u8, said.string, " \t\r\n") else null;
}

/// Removes a finished job's directory now: it holds someone's voice (clips, codes, a transcript), which is kept only
/// in the voice itself.
fn scrub(app: *App, j: *jobs.Job) void {
    Io.Dir.cwd().deleteTree(app.io, j.dir) catch |err| std.log.warn("voices: could not delete {s}: {s}", .{ j.dir, @errorName(err) });
}

/// The encode's `voice.json`, checked: one reference per clip sent, each a 10-row grid of `frames` integers.
fn readCodes(app: *App, a: std.mem.Allocator, j: *jobs.Job, want: usize) ?request.ClonedCodes {
    const path = std.fs.path.join(a, &.{ j.dir, "voice.json" }) catch return null;
    const text = Io.Dir.cwd().readFileAlloc(app.io, path, a, .limited(64 << 20)) catch return null;
    const c = std.json.parseFromSliceLeaky(request.ClonedCodes, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch return null;
    if (c.references.len != want) return null;
    for (c.references) |r| {
        if (r.vq_codes != .array or r.vq_codes.array.items.len != 10 or r.frames == 0) return null;
        for (r.vq_codes.array.items) |row| {
            if (row != .array or row.array.items.len != r.frames) return null;
            for (row.array.items) |x| if (x != .integer) return null;
        }
    }
    return c;
}
