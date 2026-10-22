//! One voice conversation (docs/dev/VOICE.md): the client's Realtime API events, the input buffer and its VAD, the
//! conversation items, and barge-in. Everything that waits on a model runs on the session's worker thread
//! (`respond.zig`); this side only parses, buffers and decides, so an interruption takes effect within the audio chunk
//! that carried it. All state is under `mutex`; server events are sent under it too, so their order is the state's.

const std = @import("std");
const Io = std.Io;
const config = @import("../config.zig");
const backend = @import("backend.zig");
const events = @import("events.zig");
const vad = @import("vad.zig");
const resample = @import("resample.zig");
const respond = @import("respond.zig");
const Stop = @import("stop.zig").Stop;

const Value = std.json.Value;

pub const rate: u32 = 24_000; // pcm16 in and out
const keep_silence_ms: u64 = 500; // audio kept before an onset beyond prefix_padding_ms, while nobody speaks
const max_buffer_ms: u64 = 10 * 60 * 1000; // a manual turn longer than this is refused
const warm_idle_ns: i96 = 20 * std.time.ns_per_s; // STT idle this long: warm it when someone starts speaking

pub const Kind = enum { user, assistant, system, function_call, function_call_output };

/// A conversation item; its strings live in its own arena.
pub const Item = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    kind: Kind,
    text: []const u8 = "", // what was said or typed, what the assistant spoke, a tool's output
    audio: bool = false, // a user turn from the microphone
    pending: bool = false, // its transcript is not in yet
    call_id: []const u8 = "",
    name: []const u8 = "",
    arguments: []const u8 = "",
    response_id: []const u8 = "",
    spoken: std.ArrayList(Spoken) = .empty, // assistant: sentences sent as audio

    pub const Spoken = struct { end_ms: u64, text_end: usize };
};

pub const Task = union(enum) {
    partial: Hearing,
    draft: Hearing,
    final: Hearing,
    respond: Ask,
    warm: void, // a throwaway transcription of silence: speech started after the STT model sat idle

    /// key: the draft's quiet run (session ms where it began); for a final, the draft it may reuse (0: none).
    pub const Hearing = struct { item_id: []u8, pcm: []i16, key: u64 = 0 };
    pub const Ask = struct { auto: bool, instructions: ?[]u8 = null, audio: ?bool = null };

    pub fn free(t: Task, gpa: std.mem.Allocator) void {
        switch (t) {
            .partial, .draft, .final => |h| {
                gpa.free(h.item_id);
                gpa.free(h.pcm);
            },
            .respond => |r| if (r.instructions) |i| gpa.free(i),
            .warm => {},
        }
    }
};

const Turn = struct { item_id: []u8, start_ms: u64, partial_ms: u64 };

/// A transcript of the turn so far, made when the speaker paused: if the pause is the turn's end, the turn's final
/// transcript is this one, already done, and one that ends a sentence ends the turn early.
pub const Draft = struct {
    item_id: []u8,
    key: u64, // session ms where the pause began
    text: ?[]u8 = null, // null while being transcribed, or if that failed
    start_ns: i96 = 0,
    end_ns: i96 = 0,
    worker_ms: u64 = 0,

    fn free(d: Draft, gpa: std.mem.Allocator) void {
        gpa.free(d.item_id);
        if (d.text) |t| gpa.free(t);
    }
};

pub const Session = struct {
    gpa: std.mem.Allocator,
    io: Io,
    backend: backend.Backend,
    emit: events.Emitter,
    voice: config.Voice,
    id: []const u8,
    mutex: Io.Mutex = .init,
    settings_arena: std.heap.ArenaAllocator,
    settings: events.Settings,
    detector: ?vad.Vad = null,
    detector_ms: u64 = 0, // session time the detector started at (its events count from there)
    input: std.ArrayList(i16) = .empty, // since the last commit or clear (only a recent tail while nobody speaks)
    input_start_ms: u64 = 0, // session time of input[0]
    heard: u64 = 0, // samples received this session
    turn: ?Turn = null,
    items: std.ArrayList(*Item) = .empty,
    next_id: u64 = 1,
    tasks: std.ArrayList(Task) = .empty,
    cond: Io.Condition = .init,
    closing: bool = false,
    worker: ?std.Thread = null,
    resp: ?*respond.Response = null,
    partial_busy: bool = false,
    fresh: bool = false, // something to answer arrived since the last response
    waiters: std.ArrayList(std.Thread) = .empty,
    close_stop: Stop = .{}, // set when the session closes: ends transcriptions and tool waits
    // The last spoken turn's marks (awake-clock ns; 0: none), for the timing of the response that answers it.
    stop_ns: i96 = 0, // VAD ended the turn
    stt_start_ns: i96 = 0,
    stt_end_ns: i96 = 0,
    stt_worker_ms: u64 = 0, // the transcription as the worker timed it
    early_end: bool = false, // the turn ended on its draft's sentence end
    stt_draft: bool = false, // its final transcript is the draft made when the speaker paused
    last_heard_ns: i96 = 0, // the last transcription ended (or a warm-up was queued)
    primer: ?std.Thread = null, // the chat prefix prime (respond.prime), joined before the next and at the end
    priming: bool = false,
    draft: ?Draft = null,
    language_buf: [16]u8 = undefined, // the language the last transcript was in: the next one's, if the session sets none
    language_len: usize = 0,

    /// A session; it sends session.created and starts its worker.
    pub fn create(gpa: std.mem.Allocator, io: Io, b: backend.Backend, out: backend.Out, voice: config.Voice, chat_model: []const u8) !*Session {
        const s = try gpa.create(Session);
        errdefer gpa.destroy(s);
        var rnd: [8]u8 = undefined;
        io.random(&rnd);
        s.* = .{ .gpa = gpa, .io = io, .backend = b, .emit = .{ .out = out }, .voice = voice, .id = "", .settings_arena = .init(gpa), .settings = .{} };
        errdefer s.settings_arena.deinit();
        s.id = try std.fmt.allocPrint(gpa, "sess_{x:0>16}", .{std.mem.readInt(u64, &rnd, .little)});
        errdefer gpa.free(s.id);
        const sa = s.settings_arena.allocator();
        s.settings = try events.Settings.defaults(voice).dupe(sa);
        if (chat_model.len > 0) s.settings.model = try sa.dupe(u8, chat_model);
        s.resetDetector();
        var tmp: std.heap.ArenaAllocator = .init(gpa);
        defer tmp.deinit();
        s.emit.send(tmp.allocator(), "session.created", .{ .session = try events.sessionObject(tmp.allocator(), s.id, &s.settings) });
        s.worker = try std.Thread.spawn(.{}, respond.work, .{s});
        s.mutex.lockUncancelable(io);
        respond.prime(s); // the system prompt and tools, before the first turn needs them
        s.mutex.unlock(io);
        return s;
    }

    /// Ends the session: the response is cancelled, the worker and tool waiters are joined, everything is freed.
    pub fn destroy(s: *Session) void {
        s.mutex.lockUncancelable(s.io);
        s.closing = true;
        if (s.resp) |r| r.stop.set(s.io);
        s.cond.broadcast(s.io);
        s.mutex.unlock(s.io);
        s.close_stop.set(s.io);
        if (s.worker) |t| t.join();
        if (s.primer) |t| t.join(); // the worker has finished: no response can start another
        for (s.waiters.items) |t| t.join();
        s.waiters.deinit(s.gpa);
        for (s.tasks.items) |t| t.free(s.gpa);
        s.tasks.deinit(s.gpa);
        for (s.items.items) |it| freeItem(s.gpa, it);
        s.items.deinit(s.gpa);
        s.input.deinit(s.gpa);
        if (s.turn) |t| s.gpa.free(t.item_id);
        s.dropDraft();
        s.settings_arena.deinit();
        s.gpa.free(s.id);
        s.gpa.destroy(s);
    }

    /// One client event (a WebSocket text message).
    pub fn handle(s: *Session, text: []const u8) void {
        var tmp: std.heap.ArenaAllocator = .init(s.gpa);
        defer tmp.deinit();
        const a = tmp.allocator();
        const v = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return s.emit.fail(a, "invalid_request_error", "invalid_json", "the event is not valid JSON", null);
        if (v != .object) return s.emit.fail(a, "invalid_request_error", "invalid_event", "an event must be a JSON object", null);
        const eid = events.str(v.object.get("event_id") orelse .null);
        const typ = events.str(v.object.get("type") orelse .null) orelse return s.emit.fail(a, "invalid_request_error", "invalid_event", "the event has no type", eid);
        s.dispatch(a, typ, v.object, eid) catch |err| s.emit.fail(a, "server_error", "internal_error", @errorName(err), eid);
    }

    /// A binary WebSocket message: the protocol is JSON text, audio travels base64 inside it.
    pub fn refuseBinary(s: *Session) void {
        var tmp: std.heap.ArenaAllocator = .init(s.gpa);
        defer tmp.deinit();
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        s.emit.fail(tmp.allocator(), "invalid_request_error", "invalid_event", "events are JSON text messages; send audio as base64 in input_audio_buffer.append", null);
    }

    fn dispatch(s: *Session, a: std.mem.Allocator, typ: []const u8, o: std.json.ObjectMap, eid: ?[]const u8) !void {
        if (eql(typ, "input_audio_buffer.append")) return s.append(a, o, eid);
        if (eql(typ, "session.update")) return s.update(a, o, eid);
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (eql(typ, "input_audio_buffer.commit")) return s.commitManual(a, eid);
        if (eql(typ, "input_audio_buffer.clear")) {
            s.input.clearRetainingCapacity();
            s.input_start_ms = s.nowMs();
            s.endTurn();
            s.dropDraft();
            if (s.detector) |*d| d.reset();
            return s.emit.send(a, "input_audio_buffer.cleared", .{});
        }
        if (eql(typ, "conversation.item.create")) return s.createItem(a, o, eid);
        if (eql(typ, "conversation.item.truncate")) return s.truncate(a, o, eid);
        if (eql(typ, "conversation.item.delete")) return s.deleteItem(a, o, eid);
        if (eql(typ, "response.create")) return s.responseCreate(a, o, eid);
        if (eql(typ, "response.cancel")) {
            if (s.resp == null) return s.emit.fail(a, "invalid_request_error", "response_cancel_not_active", "there is no response to cancel", eid);
            return s.cancelResponse("client_cancelled");
        }
        s.emit.fail(a, "invalid_request_error", "unknown_event_type", try std.fmt.allocPrint(a, "unknown event type {s}", .{typ}), eid);
    }

    // ------------------------------------------------------------ settings

    fn update(s: *Session, a: std.mem.Allocator, o: std.json.ObjectMap, eid: ?[]const u8) !void {
        const obj = o.get("session") orelse return s.emit.fail(a, "invalid_request_error", "missing_session", "session.update needs a session object", eid);
        var next: std.heap.ArenaAllocator = .init(s.gpa);
        s.mutex.lockUncancelable(s.io);
        var st = s.settings.dupe(next.allocator()) catch {
            s.mutex.unlock(s.io);
            next.deinit();
            return error.OutOfMemory;
        };
        s.mutex.unlock(s.io);
        var why: backend.Why = .{};
        events.apply(next.allocator(), &st, obj, s.backend, &why) catch |err| {
            next.deinit();
            if (err == error.OutOfMemory) return error.OutOfMemory;
            return s.emit.fail(a, "invalid_request_error", "invalid_session", why.message, eid);
        };
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        const vad_changed = !std.meta.eql(st.vad, s.settings.vad);
        var old = s.settings_arena;
        s.settings = st;
        s.settings_arena = next;
        old.deinit();
        if (vad_changed and s.turn == null) s.resetDetector();
        s.emit.send(a, "session.updated", .{ .session = try events.sessionObject(a, s.id, &s.settings) });
    }

    fn resetDetector(s: *Session) void {
        s.detector = if (s.settings.vad) |c| vad.Vad.init(withPause(c, s.voice.draft_after_ms), rate) else null;
        s.detector_ms = s.nowMs();
    }

    // ------------------------------------------------------------ audio in

    fn append(s: *Session, a: std.mem.Allocator, o: std.json.ObjectMap, eid: ?[]const u8) !void {
        const b64 = events.str(o.get("audio") orelse .null) orelse return s.emit.fail(a, "invalid_request_error", "invalid_audio", "input_audio_buffer.append needs audio (base64 pcm16)", eid);
        const dec = std.base64.standard.Decoder;
        const n = dec.calcSizeForSlice(b64) catch return s.emit.fail(a, "invalid_request_error", "invalid_audio", "audio is not valid base64", eid);
        const raw = try a.alloc(u8, n);
        dec.decode(raw, b64) catch return s.emit.fail(a, "invalid_request_error", "invalid_audio", "audio is not valid base64", eid);
        const pcm = resample.fromLe(a, raw) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return s.emit.fail(a, "invalid_request_error", "invalid_audio", "audio must be whole 16-bit samples", eid),
        };
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (s.detector == null and s.input.items.len / (rate / 1000) > max_buffer_ms) return s.emit.fail(a, "invalid_request_error", "input_audio_buffer_full", "the input buffer holds 10 minutes; commit or clear it", eid);
        try s.input.appendSlice(s.gpa, pcm);
        s.heard += pcm.len;
        if (s.detector == null) return;
        var evs: std.ArrayList(vad.Event) = .empty;
        try s.detector.?.feed(a, pcm, &evs);
        for (evs.items) |e| switch (e.kind) {
            .speech_started => try s.speechStarted(a, s.detector_ms + e.ms),
            .speech_paused => try s.speechPaused(s.detector_ms + e.ms),
            .speech_stopped => try s.speechStopped(a, s.detector_ms + e.ms, s.reusable(s.detector_ms + e.ms), false),
        };
        if (s.turn == null and !s.detector.?.speaking()) s.trimSilence();
        try s.maybePartial();
        try s.maybeEndEarly(a);
    }

    /// The awake clock, in ns (stage timings).
    pub fn clock(s: *const Session) i96 {
        return Io.Clock.awake.now(s.io).toNanoseconds();
    }

    /// Session milliseconds of audio received so far.
    pub fn nowMs(s: *const Session) u64 {
        return s.heard * 1000 / rate;
    }

    /// The input samples between two session times.
    fn span(s: *Session, from_ms: u64, to_ms: u64) []const i16 {
        const per_ms = rate / 1000;
        const lo: usize = @intCast(@min((from_ms -| s.input_start_ms) * per_ms, s.input.items.len));
        const hi: usize = @intCast(@min((to_ms -| s.input_start_ms) * per_ms, s.input.items.len));
        return s.input.items[lo..@max(lo, hi)];
    }

    /// Drops input older than `to_ms`.
    fn consume(s: *Session, to_ms: u64) void {
        const per_ms = rate / 1000;
        const cut: usize = @intCast(@min((to_ms -| s.input_start_ms) * per_ms, s.input.items.len));
        const rest = s.input.items.len - cut;
        std.mem.copyForwards(i16, s.input.items[0..rest], s.input.items[cut..]);
        s.input.shrinkRetainingCapacity(rest);
        s.input_start_ms = @max(s.input_start_ms, to_ms);
    }

    fn trimSilence(s: *Session) void {
        const pad: u64 = if (s.settings.vad) |c| c.prefix_padding_ms else 0;
        const now = s.nowMs();
        if (now > pad + keep_silence_ms + 1000 and now - s.input_start_ms > pad + keep_silence_ms + 1000) s.consume(now - pad - keep_silence_ms);
    }

    fn speechStarted(s: *Session, a: std.mem.Allocator, ms: u64) !void {
        s.endTurn();
        const id = try s.newId(s.gpa, "item");
        s.turn = .{ .item_id = id, .start_ms = @max(ms, s.input_start_ms), .partial_ms = ms };
        s.emit.send(a, "input_audio_buffer.speech_started", .{ .audio_start_ms = ms, .item_id = id });
        if (s.last_heard_ns == 0 or s.clock() - s.last_heard_ns > warm_idle_ns) { // on GB10 the first decode after
            // a load or another model's job took 1.1 to 1.7 s, against 0.3 s warm: pay it while the user still talks
            s.last_heard_ns = s.clock();
            try s.push(.warm);
        }
        if (!s.settings.interrupt_response) return;
        if (s.resp != null) s.cancelResponse("turn_detected");
        var i: usize = 0;
        while (i < s.tasks.items.len) {
            if (s.tasks.items[i] == .respond) {
                s.tasks.orderedRemove(i).free(s.gpa);
            } else i += 1;
        }
    }

    /// The turn ends at `ms`; its final transcript is the draft `key` (0: none), `early` on that draft's sentence end.
    fn speechStopped(s: *Session, a: std.mem.Allocator, ms: u64, key: u64, early: bool) !void {
        const t = s.turn orelse return;
        if (key == 0) s.dropDraft();
        s.emit.send(a, "input_audio_buffer.speech_stopped", .{ .audio_end_ms = ms, .item_id = t.item_id });
        s.stop_ns = s.clock();
        s.early_end = early;
        try s.commit(a, t.item_id, t.start_ms, ms, key);
        s.endTurn();
        if (s.settings.create_response) try s.push(.{ .respond = .{ .auto = true } });
    }

    /// The speaker paused: a draft transcript of the turn so far, on the worker.
    fn speechPaused(s: *Session, ms: u64) !void {
        const t = s.turn orelse return;
        s.dropDraft();
        const pcm = try s.gpa.dupe(i16, s.span(t.start_ms, s.nowMs()));
        errdefer s.gpa.free(pcm);
        const id = try s.gpa.dupe(u8, t.item_id);
        errdefer s.gpa.free(id);
        const mine = try s.gpa.dupe(u8, t.item_id);
        errdefer s.gpa.free(mine);
        try s.push(.{ .draft = .{ .item_id = id, .pcm = pcm, .key = ms } });
        s.draft = .{ .item_id = mine, .key = ms };
    }

    /// Whether the draft `d` is of the open turn and nobody spoke since it began.
    fn holds(s: *Session, d: Draft) bool {
        const t = s.turn orelse return false;
        const since = (s.detector orelse return false).quietSince() orelse return false;
        return eql(t.item_id, d.item_id) and s.detector_ms + since == d.key;
    }

    /// The draft the turn ending at `stop_ms` (VAD) can reuse: one whose quiet run is the one that ended it; 0: none.
    fn reusable(s: *Session, stop_ms: u64) u64 {
        const d = s.draft orelse return 0;
        const t = s.turn orelse return 0;
        const silence = (s.detector orelse return 0).silenceMs();
        return if (eql(t.item_id, d.item_id) and d.key + silence == stop_ms) d.key else 0;
    }

    /// Ends the turn now if its draft is in, ends a sentence, and `early_end_ms` of quiet have followed (the worker
    /// calls this too, when a draft comes in after that).
    pub fn maybeEndEarly(s: *Session, a: std.mem.Allocator) !void {
        const d = s.draft orelse return;
        const text = d.text orelse return;
        if (s.voice.early_end_ms == 0 or !endsSentence(text) or !s.holds(d) or s.nowMs() < d.key + s.voice.early_end_ms) return;
        s.detector.?.reset();
        try s.speechStopped(a, s.nowMs(), d.key, true);
    }

    /// Under the lock, for the worker: whether draft task `h` is still worth transcribing.
    pub fn draftWanted(s: *Session, h: Task.Hearing) bool {
        const d = s.draft orelse return false;
        if (d.key != h.key or !eql(d.item_id, h.item_id)) return false; // superseded
        const t = s.turn orelse return true; // the turn ended on this pause: its final reuses the draft
        return !eql(t.item_id, h.item_id) or s.holds(d);
    }

    /// Under the lock, for the worker: draft `h`'s transcript is in.
    pub fn drafted(s: *Session, a: std.mem.Allocator, h: Task.Hearing, text: []const u8, began: i96, worker_ms: u64) void {
        const d = if (s.draft) |*x| x else return;
        if (d.key != h.key or !eql(d.item_id, h.item_id)) return;
        d.text = s.gpa.dupe(u8, std.mem.trim(u8, text, " \t\r\n")) catch return;
        d.start_ns = began;
        d.end_ns = s.clock();
        d.worker_ms = worker_ms;
        // The draft is the turn so far: the client sees it as an interim transcript, at no cost to the final.
        if (s.turn) |t| if (eql(t.item_id, h.item_id) and d.text.?.len > 0)
            s.emit.send(a, "localrouter.transcription.partial", .{ .item_id = h.item_id, .transcript = d.text.? });
        s.maybeEndEarly(a) catch {};
    }

    /// Under the lock, for the worker: final task `h`'s transcript from its draft (in `a`), or null to transcribe it.
    pub fn reuseDraft(s: *Session, a: std.mem.Allocator, h: Task.Hearing) ?[]const u8 {
        const d = s.draft orelse return null;
        if (h.key == 0 or d.key != h.key or !eql(d.item_id, h.item_id)) return null;
        defer s.dropDraft();
        const text = a.dupe(u8, d.text orelse return null) catch return null;
        s.stt_start_ns = d.start_ns;
        s.stt_end_ns = d.end_ns;
        s.stt_worker_ms = d.worker_ms;
        return text;
    }

    fn dropDraft(s: *Session) void {
        if (s.draft) |d| d.free(s.gpa);
        s.draft = null;
    }

    /// The language to transcribe in when the session sets none: the one the last transcript was in.
    pub fn heardLanguage(s: *const Session) []const u8 {
        return s.language_buf[0..s.language_len];
    }

    pub fn noteLanguage(s: *Session, language: []const u8) void {
        if (language.len == 0 or language.len > s.language_buf.len) return;
        @memcpy(s.language_buf[0..language.len], language);
        s.language_len = language.len;
    }

    fn endTurn(s: *Session) void {
        if (s.turn) |t| s.gpa.free(t.item_id);
        s.turn = null;
    }

    fn commitManual(s: *Session, a: std.mem.Allocator, eid: ?[]const u8) !void {
        if (s.input.items.len < rate / 10) return s.emit.fail(a, "invalid_request_error", "input_audio_buffer_commit_empty", "the input buffer holds less than 100 ms of audio", eid);
        const from = if (s.turn) |t| t.start_ms else s.input_start_ms;
        const id = if (s.turn) |t| try s.gpa.dupe(u8, t.item_id) else try s.newId(s.gpa, "item");
        defer s.gpa.free(id);
        s.endTurn();
        if (s.detector) |*d| d.reset();
        s.dropDraft();
        try s.commit(a, id, from, s.nowMs(), 0);
    }

    /// The turn [from_ms, to_ms) becomes a user item, transcribed on the worker.
    fn commit(s: *Session, a: std.mem.Allocator, item_id: []const u8, from_ms: u64, to_ms: u64, key: u64) !void {
        const pcm = try s.gpa.dupe(i16, s.span(from_ms, to_ms));
        errdefer s.gpa.free(pcm);
        s.consume(to_ms);
        const prev = s.lastId();
        const it = try s.addItem(.user, item_id);
        it.audio = true;
        it.pending = true;
        s.emit.send(a, "input_audio_buffer.committed", .{ .previous_item_id = prev, .item_id = item_id });
        s.emit.send(a, "conversation.item.created", .{ .previous_item_id = prev, .item = try itemValue(a, it, s.settings.audio) });
        const owned = try s.gpa.dupe(u8, item_id);
        errdefer s.gpa.free(owned);
        try s.push(.{ .final = .{ .item_id = owned, .pcm = pcm, .key = key } });
    }

    fn maybePartial(s: *Session) !void {
        if (s.turn == null) return;
        const t = &s.turn.?;
        if (s.settings.partial_interval_ms == 0 or s.partial_busy) return;
        const now = s.nowMs();
        if (now - t.partial_ms < s.settings.partial_interval_ms) return;
        t.partial_ms = now;
        const pcm = try s.gpa.dupe(i16, s.span(t.start_ms, now));
        errdefer s.gpa.free(pcm);
        const id = try s.gpa.dupe(u8, t.item_id);
        errdefer s.gpa.free(id);
        try s.push(.{ .partial = .{ .item_id = id, .pcm = pcm } });
        s.partial_busy = true;
    }

    // ------------------------------------------------------------ items

    pub fn newId(s: *Session, a: std.mem.Allocator, prefix: []const u8) ![]u8 {
        defer s.next_id += 1;
        return std.fmt.allocPrint(a, "{s}_{d:0>6}", .{ prefix, s.next_id });
    }

    pub fn lastId(s: *const Session) ?[]const u8 {
        return if (s.items.items.len > 0) s.items.items[s.items.items.len - 1].id else null;
    }

    /// A new item at the end of the conversation (its id copied into its arena).
    pub fn addItem(s: *Session, kind: Kind, id: []const u8) !*Item {
        const it = try s.gpa.create(Item);
        errdefer s.gpa.destroy(it);
        it.* = .{ .arena = .init(s.gpa), .id = "", .kind = kind };
        errdefer it.arena.deinit();
        it.id = try it.arena.allocator().dupe(u8, id);
        try s.items.append(s.gpa, it);
        return it;
    }

    pub fn find(s: *Session, id: []const u8) ?*Item {
        for (s.items.items) |it| if (eql(it.id, id)) return it;
        return null;
    }

    fn createItem(s: *Session, a: std.mem.Allocator, o: std.json.ObjectMap, eid: ?[]const u8) !void {
        const v = o.get("item") orelse return s.emit.fail(a, "invalid_request_error", "missing_item", "conversation.item.create needs an item", eid);
        if (v != .object) return s.emit.fail(a, "invalid_request_error", "invalid_item", "item must be an object", eid);
        const io_ = v.object;
        const typ = events.str(io_.get("type") orelse .null) orelse "message";
        const given = events.str(io_.get("id") orelse .null);
        if (given) |g| if (s.find(g) != null) return s.emit.fail(a, "invalid_request_error", "duplicate_item_id", "an item with this id exists", eid);
        const id = if (given) |g| try a.dupe(u8, g) else try s.newId(a, "item");
        const prev = s.lastId();
        if (eql(typ, "function_call_output")) {
            const call = events.str(io_.get("call_id") orelse .null) orelse return s.emit.fail(a, "invalid_request_error", "invalid_item", "a function_call_output needs call_id", eid);
            const out = events.str(io_.get("output") orelse .null) orelse return s.emit.fail(a, "invalid_request_error", "invalid_item", "a function_call_output needs output (a string)", eid);
            const it = try s.addItem(.function_call_output, id);
            it.call_id = try it.arena.allocator().dupe(u8, call);
            it.text = try it.arena.allocator().dupe(u8, out);
            s.fresh = true;
            return s.emit.send(a, "conversation.item.created", .{ .previous_item_id = prev, .item = try itemValue(a, it, s.settings.audio) });
        }
        if (!eql(typ, "message")) return s.emit.fail(a, "invalid_request_error", "invalid_item", try std.fmt.allocPrint(a, "item type {s} is not supported: send a message or a function_call_output", .{typ}), eid);
        const role = events.str(io_.get("role") orelse .null) orelse "user";
        const kind: Kind = if (eql(role, "user")) .user else if (eql(role, "system")) .system else if (eql(role, "assistant")) .assistant else return s.emit.fail(a, "invalid_request_error", "invalid_item", "role must be user, assistant or system", eid);
        var text: std.ArrayList(u8) = .empty;
        const content = io_.get("content") orelse Value.null;
        if (content == .array) {
            for (content.array.items) |part| {
                if (part != .object) continue;
                const t = events.str(part.object.get("text") orelse .null) orelse events.str(part.object.get("transcript") orelse .null) orelse continue;
                if (text.items.len > 0) try text.append(a, ' ');
                try text.appendSlice(a, t);
            }
        }
        if (text.items.len == 0) return s.emit.fail(a, "invalid_request_error", "invalid_item", "a message needs content with text (input_text)", eid);
        const it = try s.addItem(kind, id);
        it.text = try it.arena.allocator().dupe(u8, text.items);
        if (kind != .assistant) s.fresh = true;
        s.emit.send(a, "conversation.item.created", .{ .previous_item_id = prev, .item = try itemValue(a, it, s.settings.audio) });
    }

    /// Cuts an assistant item to the sentences that had been played by `audio_end_ms`.
    fn truncate(s: *Session, a: std.mem.Allocator, o: std.json.ObjectMap, eid: ?[]const u8) !void {
        const id = events.str(o.get("item_id") orelse .null) orelse return s.emit.fail(a, "invalid_request_error", "invalid_item_id", "conversation.item.truncate needs item_id", eid);
        const end = o.get("audio_end_ms") orelse Value{ .integer = 0 };
        if (end != .integer or end.integer < 0) return s.emit.fail(a, "invalid_request_error", "invalid_audio_end_ms", "audio_end_ms must be a whole number of milliseconds", eid);
        const it = s.find(id) orelse return s.emit.fail(a, "invalid_request_error", "item_truncate_invalid_item_id", "no such item", eid);
        if (it.kind != .assistant) return s.emit.fail(a, "invalid_request_error", "item_truncate_invalid_item_id", "only assistant items can be truncated", eid);
        var keep: usize = 0;
        for (it.spoken.items) |sp| if (sp.end_ms <= @as(u64, @intCast(end.integer))) {
            keep = sp.text_end;
        };
        if (keep < it.text.len) it.text = std.mem.trimEnd(u8, it.text[0..keep], " ");
        s.emit.send(a, "conversation.item.truncated", .{ .item_id = id, .content_index = 0, .audio_end_ms = end.integer });
    }

    fn deleteItem(s: *Session, a: std.mem.Allocator, o: std.json.ObjectMap, eid: ?[]const u8) !void {
        const id = events.str(o.get("item_id") orelse .null) orelse return s.emit.fail(a, "invalid_request_error", "invalid_item_id", "conversation.item.delete needs item_id", eid);
        for (s.items.items, 0..) |it, i| if (eql(it.id, id)) {
            if (s.resp) |r| if (eql(r.item_id, id)) return s.emit.fail(a, "invalid_request_error", "item_delete_invalid_item_id", "the item belongs to the response in progress", eid);
            freeItem(s.gpa, s.items.orderedRemove(i));
            return s.emit.send(a, "conversation.item.deleted", .{ .item_id = id });
        };
        s.emit.fail(a, "invalid_request_error", "item_delete_invalid_item_id", "no such item", eid);
    }

    // ------------------------------------------------------------ responses

    fn responseCreate(s: *Session, a: std.mem.Allocator, o: std.json.ObjectMap, eid: ?[]const u8) !void {
        if (s.resp != null or s.queuedResponse()) return s.emit.fail(a, "invalid_request_error", "conversation_already_has_active_response", "a response is in progress; cancel it first", eid);
        var ask: Task.Ask = .{ .auto = false };
        if (o.get("response")) |r| if (r == .object) {
            if (events.str(r.object.get("instructions") orelse .null)) |i| ask.instructions = try s.gpa.dupe(u8, i);
            if (r.object.get("modalities")) |m| if (m == .array) {
                var audio = false;
                for (m.array.items) |x| if (events.str(x)) |t| {
                    if (eql(t, "audio")) audio = true;
                };
                ask.audio = audio;
            };
        };
        s.push(.{ .respond = ask }) catch |err| {
            if (ask.instructions) |i| s.gpa.free(i);
            return err;
        };
    }

    fn queuedResponse(s: *const Session) bool {
        for (s.tasks.items) |t| if (t == .respond) return true;
        return false;
    }

    /// Under `mutex`: the response stops sending at once; its worker winds down and sends response.done.
    pub fn cancelResponse(s: *Session, reason: []const u8) void {
        const r = s.resp orelse return;
        if (r.stop.cancelled()) return;
        r.reason = reason;
        r.stop.set(s.io);
    }

    /// Under `mutex`: queues work for the worker.
    pub fn push(s: *Session, t: Task) !void {
        try s.tasks.append(s.gpa, t);
        s.cond.signal(s.io);
    }
};

pub fn freeItem(gpa: std.mem.Allocator, it: *Item) void {
    it.spoken.deinit(gpa);
    it.arena.deinit();
    gpa.destroy(it);
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn withPause(c: vad.Config, pause_ms: u32) vad.Config {
    var out = c;
    out.pause_ms = pause_ms;
    return out;
}

/// Whether a transcript ends a sentence: . ! ? (or a CJK mark), then perhaps closing quotes; not a trailing "...".
pub fn endsSentence(text: []const u8) bool {
    const t = std.mem.trimEnd(u8, text, " \t\r\n\"')]");
    for ([_][]const u8{ "。", "！", "？" }) |m| if (std.mem.endsWith(u8, t, m)) return true;
    if (t.len == 0 or std.mem.endsWith(u8, t, "..") or std.mem.endsWith(u8, t, "…")) return false;
    return t[t.len - 1] == '.' or t[t.len - 1] == '!' or t[t.len - 1] == '?';
}

test "a transcript ends a sentence at . ! ? but not at a comma or a trailing ellipsis" {
    try std.testing.expect(endsSentence("What is the capital of France?"));
    try std.testing.expect(endsSentence("He said \"stop.\" "));
    try std.testing.expect(endsSentence("東京はどこですか？"));
    try std.testing.expect(!endsSentence("Tell me a story about"));
    try std.testing.expect(!endsSentence("Well,"));
    try std.testing.expect(!endsSentence("So..."));
    try std.testing.expect(!endsSentence(""));
}

/// The Realtime API's item object.
pub fn itemValue(a: std.mem.Allocator, it: *const Item, audio: bool) !Value {
    var o: std.json.ObjectMap = .empty;
    try o.put(a, "id", .{ .string = it.id });
    try o.put(a, "object", .{ .string = "realtime.item" });
    switch (it.kind) {
        .function_call => {
            try o.put(a, "type", .{ .string = "function_call" });
            try o.put(a, "status", .{ .string = "completed" });
            try o.put(a, "call_id", .{ .string = it.call_id });
            try o.put(a, "name", .{ .string = it.name });
            try o.put(a, "arguments", .{ .string = it.arguments });
        },
        .function_call_output => {
            try o.put(a, "type", .{ .string = "function_call_output" });
            try o.put(a, "call_id", .{ .string = it.call_id });
            try o.put(a, "output", .{ .string = it.text });
        },
        .user, .assistant, .system => {
            try o.put(a, "type", .{ .string = "message" });
            try o.put(a, "status", .{ .string = if (it.pending) "in_progress" else "completed" });
            try o.put(a, "role", .{ .string = @tagName(it.kind) });
            var part: std.json.ObjectMap = .empty;
            if (it.kind == .user and it.audio) {
                try part.put(a, "type", .{ .string = "input_audio" });
                try part.put(a, "transcript", if (it.pending) .null else .{ .string = it.text });
            } else if (it.kind == .assistant and audio) {
                try part.put(a, "type", .{ .string = "audio" });
                try part.put(a, "transcript", .{ .string = it.text });
            } else {
                try part.put(a, "type", .{ .string = if (it.kind == .assistant) "text" else "input_text" });
                try part.put(a, "text", .{ .string = it.text });
            }
            var content: std.json.Array = .init(a);
            try content.append(.{ .object = part });
            try o.put(a, "content", .{ .array = content });
        },
    }
    return .{ .object = o };
}
