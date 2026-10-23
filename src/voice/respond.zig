//! A voice session's worker: transcriptions, and responses. A response streams the chat model's reply, cuts it into
//! sentences, and a speaker thread turns each into speech while the model writes the next. Tool calls to the built-in
//! image and video tools become jobs; a waiter thread reports each result and asks for a spoken follow-up.

const std = @import("std");
const Io = std.Io;
const session = @import("session.zig");
const backend = @import("backend.zig");
const events = @import("events.zig");
const chat = @import("chat.zig");
const chunker = @import("chunker.zig");
const resample = @import("resample.zig");
const Stop = @import("stop.zig").Stop;
const tool_calls = @import("tools.zig");
const timing = @import("timing.zig");

const Session = session.Session;
const Task = session.Task;
const Value = std.json.Value;

/// One response in progress. Its arena belongs to the worker thread; `queue`, `spoken`, `text` and `audio_ms` are
/// read and written under the session's mutex.
pub const Response = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8 = "",
    item_id: []const u8 = "",
    stop: Stop = .{},
    reason: []const u8 = "", // why it was cancelled: turn_detected or client_cancelled
    audio: bool = true,
    client_tools: bool = false, // the session's tool_execution was client when it began: its calls go to the client
    line: []const u8 = "", // a fixed sentence to say instead of a chat reply (Task.Ask.line)
    ack_at: ?usize = null, // where in `text` the session's own "making it" for a tool call begins (Feed.onTool)
    text: std.ArrayList(u8) = .empty,
    queue: std.ArrayList(Chunk) = .empty, // sentences for the speaker
    closed: bool = false, // no more sentences will come
    cond: Io.Condition = .init,
    spoken: std.ArrayList(session.Item.Spoken) = .empty, // gpa: moves to the item
    audio_ms: u64 = 0, // sent so far
    marks: timing.Marks = .{},
    // Speech with the GPU to itself: when the text was complete with no tool arguments to write (0: not yet), and the
    // audio sent by then (for free_speech_rate).
    free_ns: i96 = 0,
    free_audio_at: u64 = 0,

    const Chunk = struct { text: []const u8, text_end: usize };
};

/// How fast speech came the last time it had the GPU to itself (audio ms a wall s, x1000; 0: not measured yet in this
/// process): the audio a reply sent from its text_done to its speech_done, over that time, once it holds free_min_ms
/// (the part of a sentence under way, and each later one's start, count). A client given it with text_done can start
/// a reply already written at that rate, instead of measuring the rate of a stream that so far shared the GPU with the
/// chat model (a fresh session's long reply was held 2.9 to 5.2 s for that on DGX-01).
var free_speech_rate: std.atomic.Value(u32) = .init(0);
const free_min_ms = 500; // shorter spans read low (a sentence's start counts): the safe side

/// free_speech_rate as a rate (null: not measured yet).
fn speechRate() ?f64 {
    const v = free_speech_rate.load(.monotonic);
    return if (v == 0) null else @as(f64, @floatFromInt(v)) / 1000;
}

/// The worker loop: one task at a time until the session closes.
pub fn work(s: *Session) void {
    while (true) {
        s.mutex.lockUncancelable(s.io);
        while (s.tasks.items.len == 0 and !s.closing) s.cond.waitUncancelable(s.io, &s.mutex);
        if (s.closing) {
            s.mutex.unlock(s.io);
            return;
        }
        const t = s.tasks.orderedRemove(0);
        s.mutex.unlock(s.io);
        defer t.free(s.gpa);
        switch (t) {
            .partial => |h| hear(s, h, .partial),
            .early => |h| hear(s, h, .early),
            .final => |h| hear(s, h, .final),
            .respond => |ask| run(s, ask),
            .warm => warm(s),
        }
    }
}

// ---------------------------------------------------------------- transcription

const Pass = enum { partial, early, final };

/// Under the lock: the chat server reads the conversation as the next turn's request will begin it (the same system
/// prompt, tools and history; `max_tokens` 1, the reply discarded), so that request finds it in the server's prefix
/// cache and pays only for the new turn. On DGX-01 the first request after a new assistant message or a barge-in took
/// 260 to 390 ms to its first token, the others 165 to 185. One at a time, off the worker thread.
pub fn prime(s: *Session) void {
    if (!s.voice.prime_chat or s.priming or s.closing) return;
    if (s.primer) |t| t.join(); // finished: `priming` is cleared as its last step
    s.primer = null;
    s.priming = true;
    s.primer = std.Thread.spawn(.{}, primeChat, .{s}) catch {
        s.priming = false;
        return;
    };
}

fn primeChat(s: *Session) void {
    var arena: std.heap.ArenaAllocator = .init(s.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    s.mutex.lockUncancelable(s.io);
    var req = request(s, a, .{ .auto = true }) catch {
        s.priming = false;
        s.mutex.unlock(s.io);
        return;
    };
    s.mutex.unlock(s.io);
    req.max_tokens = 1;
    std.log.info("voice {s}: chat prime (the conversation so far, no new turn; max_tokens 1)", .{s.id});
    var why: backend.Why = .{};
    _ = s.backend.stream(a, &req, .{ .ctx = s, .text = discard }, &s.close_stop, &why) catch {};
    s.mutex.lockUncancelable(s.io);
    s.priming = false;
    s.mutex.unlock(s.io);
}

fn discard(_: *anyopaque, _: []const u8) void {}

/// A second of silence through the session's STT model, the result discarded (see Task.warm).
fn warm(s: *Session) void {
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const a = tmp.allocator();
    s.mutex.lockUncancelable(s.io);
    const st = &s.settings;
    const language = if (st.language.len > 0) st.language else s.heardLanguage();
    const o: backend.Hear = .{ .model = a.dupe(u8, st.transcription_model) catch "", .language = a.dupe(u8, language) catch "", .greedy = s.voice.greedy_transcripts };
    s.mutex.unlock(s.io);
    const silence = a.alloc(i16, 16_000) catch return;
    @memset(silence, 0);
    var why: backend.Why = .{};
    _ = s.backend.transcribe(a, silence, o, &s.close_stop, &why) catch {};
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    s.last_heard_ns = s.clock();
}

var speech_warm: std.atomic.Value(bool) = .init(false); // a sentence has been spoken in this process
var speech_warming: std.atomic.Value(bool) = .init(false); // one session at a time speaks the warm-up
const warm_sentence = "Hello.";

/// Nothing has been spoken in this process yet (voice.warm_speech).
pub fn speechCold() bool {
    return !speech_warm.load(.monotonic);
}

/// On its own thread at connect (Session.warmer): the warm-up sentence through the session's speech model and voice,
/// the audio discarded, so the model's first job is paid before a reply needs it. One short word: the engine speaks one
/// request at a time, and a turn's reply waits behind it (a cold first sentence of 2 s took 4.7 s on DGX-01). Its rate
/// is not kept (a first job's, and a second sentence collided with the first turn there).
pub fn warmSpeech(s: *Session) void {
    if (!speechCold() or speech_warming.swap(true, .acq_rel)) return;
    defer speech_warming.store(false, .release);
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const a = tmp.allocator();
    s.mutex.lockUncancelable(s.io);
    const say: backend.Say = .{ .model = a.dupe(u8, s.settings.speech_model) catch "", .voice = a.dupe(u8, s.settings.voice) catch "" };
    s.mutex.unlock(s.io);
    var why: backend.Why = .{};
    const began = s.clock();
    s.backend.speak(a, warm_sentence, say, .{ .ctx = s, .write = discardPcm }, &s.close_stop, &why) catch return;
    speech_warm.store(true, .monotonic);
    std.log.info("voice {s}: speech warmed in {d} ms", .{ s.id, @divTrunc(s.clock() - began, std.time.ns_per_ms) });
}

fn discardPcm(_: *anyopaque, _: []const i16, _: u32) void {}

fn hear(s: *Session, h: Task.Hearing, pass: Pass) void {
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const a = tmp.allocator();
    var why: backend.Why = .{};
    s.mutex.lockUncancelable(s.io);
    if (pass == .partial and s.find(h.item_id) != null) { // committed meanwhile: the final transcript is coming
        s.partial_busy = false;
        s.mutex.unlock(s.io);
        return;
    }
    if (pass == .early and !s.earlyWanted(h)) return s.mutex.unlock(s.io); // the speaker went on
    if (pass == .final) if (s.reuseEarly(a, h)) |early| { // made in the wait, of the same audio: nothing more to hear
        defer s.mutex.unlock(s.io);
        return completed(s, a, h.item_id, early, &why);
    };
    const st = &s.settings;
    const language = if (st.language.len > 0) st.language else s.heardLanguage(); // no detection pass on each turn
    const o: backend.Hear = .{ .model = a.dupe(u8, st.transcription_model) catch "", .language = a.dupe(u8, language) catch "", .prompt = a.dupe(u8, st.transcription_prompt) catch "", .greedy = s.voice.greedy_transcripts };
    s.mutex.unlock(s.io);
    var worker_ms: u64 = 0;
    var detected: []const u8 = "";
    var heard = o;
    heard.worker_ms = &worker_ms;
    heard.detected = &detected;
    const began = s.clock();
    const text: backend.Error![]const u8 = if (resample.resample(a, h.pcm, session.rate, 16_000)) |pcm16|
        s.backend.transcribe(a, pcm16, heard, &s.close_stop, &why)
    else |_|
        error.OutOfMemory;
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    s.noteLanguage(detected);
    s.last_heard_ns = s.clock();
    switch (pass) {
        .partial => {
            s.partial_busy = false;
            const t = text catch return;
            const turn = s.turn orelse return;
            if (std.mem.eql(u8, turn.item_id, h.item_id)) s.emit.send(a, "localrouter.transcription.partial", .{ .item_id = h.item_id, .transcript = std.mem.trim(u8, t, " \t\r\n") });
        },
        .early => if (text) |t| s.earlyDone(h, std.mem.trim(u8, t, " \t\r\n"), began, worker_ms) else |_| {},
        .final => {
            s.stt_start_ns = began;
            s.stt_end_ns = s.clock();
            s.stt_worker_ms = worker_ms;
            completed(s, a, h.item_id, text, &why);
        },
    }
}

/// Under the lock: the final transcript of user item `item_id` (deleted meanwhile: nothing).
fn completed(s: *Session, a: std.mem.Allocator, item_id: []const u8, text: backend.Error![]const u8, why: *const backend.Why) void {
    const it = s.find(item_id) orelse return;
    it.pending = false;
    if (text) |t| {
        it.text = it.arena.allocator().dupe(u8, std.mem.trim(u8, t, " \t\r\n")) catch "";
        if (it.text.len > 0) s.fresh = true;
        s.emit.send(a, "conversation.item.input_audio_transcription.completed", .{ .item_id = it.id, .content_index = 0, .transcript = it.text });
    } else |err| {
        if (err == error.Cancelled) return;
        s.emit.send(a, "conversation.item.input_audio_transcription.failed", .{ .item_id = it.id, .content_index = 0, .@"error" = .{ .type = "server_error", .code = "transcription_failed", .message = why.message } });
    }
}

// ---------------------------------------------------------------- responses

fn run(s: *Session, ask: Task.Ask) void {
    const r = s.gpa.create(Response) catch return;
    r.* = .{ .arena = .init(s.gpa) };
    defer {
        r.spoken.deinit(s.gpa);
        r.arena.deinit();
        s.gpa.destroy(r);
    }
    var req: chat.Request = undefined;
    var say: backend.Say = undefined;
    if (!begin(s, r, ask, &req, &say)) return;
    if (r.marks.from_turn) std.log.info("voice {s}: chat request for {s}, {d} ms after the turn ended", .{ s.id, r.id, @divTrunc(s.clock() - r.marks.t0, std.time.ns_per_ms) }) else std.log.info("voice {s}: chat request for {s}", .{ s.id, r.id });
    var why: backend.Why = .{};
    const result = stream(s, r, &req, say, &why);
    finish(s, r, result, &why);
}

/// The chat stream into text and sentences, and the speaker turning those into audio, until both are done.
fn stream(s: *Session, r: *Response, req: *chat.Request, say: backend.Say, why: *backend.Why) backend.Error!chat.Result {
    const speaker: ?std.Thread = if (r.audio) std.Thread.spawn(.{}, speak, .{ s, r, say }) catch null else null;
    var feed: Feed = .{ .s = s, .r = r, .chunks = .{ .gpa = s.gpa } };
    defer feed.chunks.deinit();
    defer feed.think.deinit(s.gpa);
    defer feed.lead.deinit(s.gpa);
    r.marks.chat_start = s.clock();
    const result = if (r.line.len > 0) line: {
        Feed.onText(&feed, r.line); // a fixed sentence: spoken as if the model had written it
        break :line @as(backend.Error!chat.Result, .{ .finish_reason = "stop", .text = r.line });
    } else s.backend.stream(r.arena.allocator(), req, .{ .ctx = &feed, .text = Feed.onText, .tool = Feed.onTool }, &r.stop, why);
    feed.end();
    if (speaker) |t| t.join();
    return result;
}

/// Under the lock: copies what the response needs, adds the assistant item and announces the response.
fn begin(s: *Session, r: *Response, ask: Task.Ask, req: *chat.Request, say: *backend.Say) bool {
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const ea = tmp.allocator();
    const a = r.arena.allocator();
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    if (s.closing or (ask.auto and !s.fresh)) return false; // nothing new to answer (an empty transcript)
    s.responded = true;
    r.client_tools = s.settings.tool_execution == .client;
    if (r.client_tools) s.expireCalls(ea, "response_started"); // an output that has not come by now never will count
    r.audio = ask.audio orelse s.settings.audio;
    if (ask.line) |l| r.line = a.dupe(u8, l) catch return false;
    r.marks.from_turn = ask.auto and s.stop_ns > 0; // answering a spoken turn: timed from its end
    r.marks.t0 = if (r.marks.from_turn) s.stop_ns else s.clock();
    if (r.marks.from_turn) {
        r.marks.stt_start = s.stt_start_ns;
        r.marks.stt_end = s.stt_end_ns;
        r.marks.stt_worker_ms = s.stt_worker_ms;
        r.marks.speech_end = s.speech_end_ns;
    }
    s.stop_ns = 0;
    r.id = s.newId(a, "resp") catch return false;
    r.item_id = s.newId(a, "item") catch return false;
    req.* = request(s, a, ask) catch return false;
    say.* = .{ .model = a.dupe(u8, s.settings.speech_model) catch return false, .voice = a.dupe(u8, s.settings.voice) catch return false };
    const prev = s.lastId();
    const it = s.addItem(.assistant, r.item_id) catch return false;
    it.response_id = it.arena.allocator().dupe(u8, r.id) catch "";
    it.pending = true;
    s.fresh = false;
    s.resp = r;
    const item = session.itemValue(ea, it, r.audio) catch return true;
    s.emit.send(ea, "response.created", .{ .response = responseValue(r, "in_progress", null, &.{}) });
    s.emit.send(ea, "response.output_item.added", .{ .response_id = r.id, .output_index = 0, .item = item });
    s.emit.send(ea, "conversation.item.created", .{ .previous_item_id = prev, .item = item });
    s.emit.send(ea, "response.content_part.added", .{ .response_id = r.id, .item_id = r.item_id, .output_index = 0, .content_index = 0, .part = part(ea, r.audio, "") });
    return true;
}

/// A content part: `{type: audio, transcript}` or `{type: text, text}`.
fn part(a: std.mem.Allocator, audio: bool, text: []const u8) Value {
    var o: std.json.ObjectMap = .empty;
    o.put(a, "type", .{ .string = if (audio) "audio" else "text" }) catch return .null;
    o.put(a, if (audio) "transcript" else "text", .{ .string = text }) catch return .null;
    return .{ .object = o };
}

fn responseValue(r: *const Response, status: []const u8, details: ?Value, output: []const Value) struct { id: []const u8, object: []const u8 = "realtime.response", status: []const u8, status_details: ?Value, output: []const Value, usage: ?u8 = null } {
    return .{ .id = r.id, .status = status, .status_details = details, .output = output };
}

/// The chat stream's sink: visible text to the client, sentences to the speaker.
const Feed = struct {
    s: *Session,
    r: *Response,
    chunks: chunker.Chunker,
    think: chunker.ThinkFilter = .{},
    lead: chunker.Lead = .{}, // holds the reply's start until no leaked reasoning can precede it
    search: usize = 0, // where the next sentence is looked for in the text
    ended: bool = false, // the spoken text is complete (endText)

    fn onText(ctx: *anyopaque, delta: []const u8) void {
        const f: *Feed = @ptrCast(@alignCast(ctx));
        if (f.ended) return; // text after a tool call began is not spoken
        if (f.r.marks.first_token == 0) f.r.marks.first_token = f.s.clock();
        const visible = f.think.filter(f.s.gpa, f.r.arena.allocator(), delta) catch return;
        f.add(f.lead.pass(f.s.gpa, f.r.arena.allocator(), visible, false) catch visible, false);
    }

    fn add(f: *Feed, visible: []const u8, last: bool) void {
        const s, const r = .{ f.s, f.r };
        const a = r.arena.allocator();
        var tmp: std.heap.ArenaAllocator = .init(s.gpa);
        defer tmp.deinit();
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (r.stop.cancelled()) return;
        if (visible.len > 0) {
            r.text.appendSlice(a, visible) catch return;
            s.emit.send(tmp.allocator(), if (r.audio) "response.audio_transcript.delta" else "response.text.delta", .{ .response_id = r.id, .item_id = r.item_id, .output_index = 0, .content_index = 0, .delta = visible });
            if (r.audio) f.chunks.push(visible) catch return;
        }
        if (!r.audio) return;
        while (f.chunks.next(a) catch null) |c| f.queue(c);
        if (last) if (f.chunks.finish(a) catch null) |c| f.queue(c);
    }

    /// Under the lock: a sentence for the speaker, with where it ends in the text (for truncation).
    fn queue(f: *Feed, c: []const u8) void {
        if (f.r.marks.first_chunk == 0) f.r.marks.first_chunk = f.s.clock();
        const at = std.mem.indexOfPos(u8, f.r.text.items, f.search, c);
        const upto = if (at) |i| i + c.len else f.r.text.items.len;
        f.search = upto;
        f.r.queue.append(f.r.arena.allocator(), .{ .text = c, .text_end = upto }) catch return;
        f.r.cond.signal(f.s.io);
    }

    /// The stream ended: what the filter held back, the last sentence, and no more.
    fn end(f: *Feed) void {
        f.endText(false);
    }

    /// The server's own memory check before a video, in place of the chat model calling reserve_for to find out: whether
    /// making room for it unloads a voice model (then the user hears that the voice pauses).
    fn pausesVoice(s: *Session, a: std.mem.Allocator) bool {
        const m = s.backend.manage(a, "reserve_for", "{\"model\": \"video\", \"dry_run\": true}");
        if (!m.ok) return false;
        const plan = std.json.parseFromSliceLeaky(struct { pauses_voice: bool = false }, a, m.text, .{ .ignore_unknown_fields = true }) catch return false;
        return plan.pauses_voice;
    }

    /// The model has begun a tool call: the rest of the stream is its arguments, which nobody hears. A picture or video
    /// call with nothing said before it gets the short "making it" its tool description asks for, so the user hears
    /// that the turn was taken while the arguments are written and the job runs (DGX-01: 12 to 15 s of silence before).
    fn onTool(ctx: *anyopaque, name: []const u8) void {
        const f: *Feed = @ptrCast(@alignCast(ctx));
        if (f.ended) return;
        const a = f.r.arena.allocator();
        const held = f.think.flush(f.s.gpa, a) catch "";
        f.add(f.lead.pass(f.s.gpa, a, held, true) catch held, false); // what the filters held back is said first
        if (f.r.client_tools) return f.endText(true); // the client's tool, whatever its name: the client says what it likes
        const said = std.mem.trim(u8, f.r.text.items, " \t\r\n").len > 0;
        f.r.ack_at = f.r.text.items.len; // what is added from here is the session's, about this call
        const video = std.mem.eql(u8, name, "generate_video");
        if (!said and f.r.audio and (video or std.mem.eql(u8, name, "generate_image"))) f.add(if (video) ack_video else ack_image, false);
        if (!said and f.r.audio and std.mem.eql(u8, name, "computer_use")) f.add(ack_computer, false);
        if (video and f.r.audio and pausesVoice(f.s, a)) f.add(ack_pause, false);
        f.endText(true);
    }

    /// The spoken text is complete (the stream ended, or its tool calls began): the last sentence goes to the speaker,
    /// which may finish once it is spoken, and the client hears of it (`tool_calls`: the model still writes arguments).
    fn endText(f: *Feed, args_follow: bool) void {
        if (f.ended) return;
        f.ended = true;
        const rest = f.think.flush(f.s.gpa, f.r.arena.allocator()) catch "";
        f.add(f.lead.pass(f.s.gpa, f.r.arena.allocator(), rest, true) catch rest, true);
        f.s.mutex.lockUncancelable(f.s.io);
        defer f.s.mutex.unlock(f.s.io);
        f.r.closed = true;
        f.r.cond.signal(f.s.io);
        // From the stream's end speech has the GPU to itself (it shares it while chat decodes, and runs slower than
        // playback then), so a client can stop holding the reply's start; while tool arguments are still written it
        // waits for localrouter.response.speech_done instead (docs/dev/VOICE.md).
        if (f.r.stop.cancelled()) return;
        if (!args_follow) {
            f.r.free_ns = f.s.clock();
            f.r.free_audio_at = f.r.audio_ms;
        }
        var tmp: std.heap.ArenaAllocator = .init(f.s.gpa);
        defer tmp.deinit();
        f.s.emit.send(tmp.allocator(), "localrouter.response.text_done", .{ .response_id = f.r.id, .item_id = f.r.item_id, .tool_calls = args_follow, .chars = f.r.text.items.len, .speech_rate = if (args_follow) null else speechRate() });
    }
};

/// The speaker thread: one speech job a sentence, sent as 100 ms audio deltas unless the response was cancelled.
fn speak(s: *Session, r: *Response, say: backend.Say) void {
    var next: usize = 0;
    while (true) {
        s.mutex.lockUncancelable(s.io);
        while (next >= r.queue.items.len and !r.closed and !r.stop.cancelled()) r.cond.waitUncancelable(s.io, &s.mutex);
        if (r.stop.cancelled() or next >= r.queue.items.len) {
            defer s.mutex.unlock(s.io);
            if (r.stop.cancelled()) return;
            // Every sentence of the reply is spoken and sent: a client holding the start for it can begin.
            measureFree(s, r);
            var tmp: std.heap.ArenaAllocator = .init(s.gpa);
            defer tmp.deinit();
            s.emit.send(tmp.allocator(), "localrouter.response.speech_done", .{ .response_id = r.id, .item_id = r.item_id });
            return;
        }
        const c = r.queue.items[next];
        next += 1;
        s.mutex.unlock(s.io);
        if (!sentence(s, r, say, c)) return;
    }
}

/// Speaks one sentence; false when the response is over for the speaker (cancelled or failed). The audio is sent as
/// the speech model makes it: each piece is resampled to the wire rate and sent once 40 ms of it are ready.
fn sentence(s: *Session, r: *Response, say: backend.Say, c: Response.Chunk) bool {
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const a = tmp.allocator();
    const text = chunker.speakable(a, c.text) catch return false;
    if (text.len == 0) return true;
    var out: Voice = .{ .s = s, .r = r, .a = a };
    defer out.rs.deinit(s.gpa);
    var why: backend.Why = .{};
    if (r.marks.tts_start == 0) r.marks.tts_start = s.clock();
    s.backend.speak(a, text, say, .{ .ctx = &out, .write = Voice.write }, &r.stop, &why) catch |err| {
        if (err == error.Cancelled) return false;
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        if (!r.stop.cancelled()) s.emit.fail(a, "server_error", "speech_failed", why.message, null);
        return false;
    };
    if (!out.finish()) return false;
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    r.spoken.append(s.gpa, .{ .end_ms = r.audio_ms, .text_end = c.text_end }) catch {};
    speech_warm.store(true, .monotonic);
    return true;
}

/// Under the lock, once every sentence is spoken: the reply's speech with the GPU to itself, as free_speech_rate.
fn measureFree(s: *Session, r: *const Response) void {
    if (r.free_ns == 0 or r.audio_ms - r.free_audio_at < free_min_ms) return;
    const span_ns = s.clock() - r.free_ns;
    const span_ms: u64 = if (span_ns >= std.time.ns_per_ms) @intCast(@divTrunc(span_ns, std.time.ns_per_ms)) else 1;
    const rate: u64 = (r.audio_ms - r.free_audio_at) * 1000 / span_ms;
    free_speech_rate.store(if (rate > 100_000) 100_000 else @intCast(rate), .monotonic);
}

/// One sentence's audio on its way to the client: resampled as it arrives, sent in pieces of 40 to 100 ms.
const Voice = struct {
    s: *Session,
    r: *Response,
    a: std.mem.Allocator,
    rs: resample.Stream = .{ .from = 0, .to = session.rate },
    ready: std.ArrayList(i16) = .empty, // resampled, not yet sent
    failed: bool = false,

    const min_piece = session.rate / 25; // 40 ms: the first audio leaves as soon as there is this much
    const max_piece = session.rate / 10; // 100 ms

    fn write(ctx: *anyopaque, samples: []const i16, rate: u32) void {
        const v: *Voice = @ptrCast(@alignCast(ctx));
        if (v.r.marks.tts_first_pcm == 0) v.r.marks.tts_first_pcm = v.s.clock();
        if (v.failed or v.r.stop.cancelled()) return;
        if (v.rs.from == 0) v.rs.from = rate;
        if (rate != v.rs.from) { // a model that changes its rate mid-sentence: not ours to guess
            v.failed = true;
            return;
        }
        const got = v.rs.push(v.s.gpa, v.a, samples, false) catch {
            v.failed = true;
            return;
        };
        v.ready.appendSlice(v.a, got) catch {
            v.failed = true;
            return;
        };
        if (!v.send(false)) v.failed = true;
    }

    /// The rest of the sentence; false when it was not all sent.
    fn finish(v: *Voice) bool {
        if (v.failed or v.r.stop.cancelled()) return false;
        if (v.rs.from != 0) {
            const got = v.rs.push(v.s.gpa, v.a, &.{}, true) catch return false;
            v.ready.appendSlice(v.a, got) catch return false;
        }
        return v.send(true);
    }

    /// Sends what is ready in pieces of at most 100 ms (all of it when `all`, else pieces of at least 40 ms).
    fn send(v: *Voice, all: bool) bool {
        const s, const r = .{ v.s, v.r };
        const enc = std.base64.standard.Encoder;
        var at: usize = 0;
        while (v.ready.items.len - at >= (if (all) 1 else min_piece)) {
            const n: usize = @min(v.ready.items.len - at, max_piece);
            const samples = v.ready.items[at..][0..n];
            const bytes = resample.toLe(v.a, samples) catch return false;
            const b64 = enc.encode(v.a.alloc(u8, enc.calcSize(bytes.len)) catch return false, bytes);
            s.mutex.lockUncancelable(s.io);
            defer s.mutex.unlock(s.io);
            if (r.stop.cancelled()) return false; // barge-in: not one more piece after speech_started
            s.emit.send(v.a, "response.audio.delta", .{ .response_id = r.id, .item_id = r.item_id, .output_index = 0, .content_index = 0, .delta = b64 });
            if (r.marks.first_audio == 0) r.marks.first_audio = s.clock();
            if (r.marks.first_voiced == 0) if (timing.firstLoud(samples)) |i| {
                r.marks.first_voiced = s.clock();
                r.marks.voiced_audio_ms = r.audio_ms + i * 1000 / session.rate;
            };
            r.audio_ms += n * 1000 / session.rate;
            at += n;
        }
        const rest = v.ready.items.len - at;
        std.mem.copyForwards(i16, v.ready.items[0..rest], v.ready.items[at..]);
        v.ready.shrinkRetainingCapacity(rest);
        return true;
    }
};

/// Under the lock: the done events, the tool calls, response.done; the session is free for the next response.
fn finish(s: *Session, r: *Response, result: backend.Error!chat.Result, why: *backend.Why) void {
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const a = tmp.allocator();
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    defer s.resp = null;
    var status: []const u8 = "completed";
    var details: ?Value = null;
    var calls: []const chat.ToolCall = &.{};
    if (result) |res| {
        calls = res.tool_calls;
    } else |err| if (err != error.Cancelled) {
        status = "failed";
        s.emit.fail(a, "server_error", "chat_failed", why.message, null);
        details = statusDetails(a, "failed", "", why.message) catch null;
    }
    if (r.stop.cancelled()) {
        status = "cancelled";
        calls = &.{};
        details = statusDetails(a, "cancelled", if (r.reason.len > 0) r.reason else "client_cancelled", "") catch null;
    }
    var kept: []const u8 = r.text.items;
    if (r.stop.cancelled() and r.audio) {
        var end: usize = 0;
        for (r.spoken.items) |sp| end = sp.text_end;
        kept = std.mem.trimEnd(u8, kept[0..@min(end, kept.len)], " \n");
    }
    // A cancelled response makes no calls, so the session's own "Making that now." for one describes nothing that
    // happened: kept, the model read a picture promised without the tool and copied it (1 in 5 spoken picture turns
    // after such a barge-in went without the call on DGX-01).
    if (r.stop.cancelled()) if (r.ack_at) |at| {
        kept = std.mem.trimEnd(u8, kept[0..@min(at, kept.len)], " \n");
    };
    var outputs: std.ArrayList(Value) = .empty;
    if (s.find(r.item_id)) |it| {
        it.text = it.arena.allocator().dupe(u8, kept) catch "";
        it.pending = false;
        it.spoken = r.spoken;
        r.spoken = .empty;
        var item = session.itemValue(a, it, r.audio) catch Value.null;
        if (item == .object and !std.mem.eql(u8, status, "completed")) item.object.put(a, "status", .{ .string = "incomplete" }) catch {};
        if (r.audio) s.emit.send(a, "response.audio.done", .{ .response_id = r.id, .item_id = r.item_id, .output_index = 0, .content_index = 0 });
        if (r.audio)
            s.emit.send(a, "response.audio_transcript.done", .{ .response_id = r.id, .item_id = r.item_id, .output_index = 0, .content_index = 0, .transcript = it.text })
        else
            s.emit.send(a, "response.text.done", .{ .response_id = r.id, .item_id = r.item_id, .output_index = 0, .content_index = 0, .text = it.text });
        s.emit.send(a, "response.content_part.done", .{ .response_id = r.id, .item_id = r.item_id, .output_index = 0, .content_index = 0, .part = part(a, r.audio, it.text) });
        s.emit.send(a, "response.output_item.done", .{ .response_id = r.id, .output_index = 0, .item = item });
        outputs.append(a, item) catch {};
    }
    for (calls, 1..) |call, index| {
        const id = s.newId(a, "item") catch break;
        // A client matches outputs by call_id, so each is unique in the session (a server's "call_0" repeats each reply).
        const call_id = if (r.client_tools and (call.id.len == 0 or s.findCall(call.id) != null)) (s.newId(a, "call") catch break) else call.id;
        const prev = s.lastId();
        const it = s.addItem(.function_call, id) catch break;
        const ia = it.arena.allocator();
        it.call_id = ia.dupe(u8, call_id) catch "";
        if (r.client_tools) it.call = .pending;
        it.name = ia.dupe(u8, call.function.name) catch "";
        it.arguments = ia.dupe(u8, call.function.arguments) catch "";
        it.response_id = ia.dupe(u8, r.id) catch "";
        const item = session.itemValue(a, it, r.audio) catch continue;
        s.emit.send(a, "response.output_item.added", .{ .response_id = r.id, .output_index = index, .item = item });
        s.emit.send(a, "conversation.item.created", .{ .previous_item_id = prev, .item = item });
        s.emit.send(a, "response.function_call_arguments.delta", .{ .response_id = r.id, .item_id = it.id, .output_index = index, .call_id = it.call_id, .delta = it.arguments });
        s.emit.send(a, "response.function_call_arguments.done", .{ .response_id = r.id, .item_id = it.id, .output_index = index, .call_id = it.call_id, .name = it.name, .arguments = it.arguments });
        s.emit.send(a, "response.output_item.done", .{ .response_id = r.id, .output_index = index, .item = item });
        outputs.append(a, item) catch {};
        if (!r.client_tools and events.isBuiltin(it.name) and s.voice.tools) tool_calls.start(s, it);
    }
    s.emit.send(a, "localrouter.response.timing", timing.event(r.id, r.marks));
    s.emit.send(a, "response.done", .{ .response = responseValue(r, status, details, outputs.items) });
    if (std.mem.eql(u8, status, "completed")) shadow(s, a, r, calls);
    prime(s);
}

fn statusDetails(a: std.mem.Allocator, typ: []const u8, reason: []const u8, message: []const u8) !Value {
    var o: std.json.ObjectMap = .empty;
    try o.put(a, "type", .{ .string = typ });
    if (reason.len > 0) try o.put(a, "reason", .{ .string = reason });
    if (message.len > 0) {
        var e: std.json.ObjectMap = .empty;
        try e.put(a, "type", .{ .string = "server_error" });
        try e.put(a, "code", .{ .string = "chat_failed" });
        try e.put(a, "message", .{ .string = message });
        try o.put(a, "error", .{ .object = e });
    }
    return .{ .object = o };
}

// ---------------------------------------------------------------- the chat request

/// Said for a picture or video call the model made without saying anything first (Feed.onTool), as its tool
/// description asks it to.
const ack_image = "Making that now.";
const ack_video = "Making that now. A video takes a few minutes.";
/// Said for a computer_use call the model made without saying anything first.
const ack_computer = "Working on it on your computer.";
/// Said when making room for a video unloads a voice model (Feed.pausesVoice).
const ack_pause = "My voice pauses while it is made, and comes back about half a minute after.";

const builtin_tools =
    \\[{"type": "function", "function": {"name": "generate_image", "description": "Start making a picture on this machine from a detailed description. It takes several seconds and the user sees it when it is done. When you call this, only say briefly that you are making it (for example: Making that now). Never say or imply it is ready, and do not describe it: you get the result later as the tool output, and then you say it is ready, or that it failed.",
    \\  "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "what to draw, in detail"},
    \\  "size": {"type": "string", "description": "WIDTHxHEIGHT in multiples of 16, e.g. 1024x1024 (default) or 1344x768"}}, "required": ["prompt"]}}},
    \\ {"type": "function", "function": {"name": "generate_video", "description": "Start making a short video clip with sound on this machine from a description. It takes minutes. When you call this, only say that you are making it and that it takes a few minutes. Never say it is ready: you get the result later as the tool output, and then you say it is ready, or that it failed.",
    \\  "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "what happens, in detail"},
    \\  "seconds": {"type": "integer", "description": "length, 1 to 20 (default 5)"}}, "required": ["prompt"]}}}]
;

/// The model management procedure (`voice.manage_models`): appended to the instructions, with its tools offered.
pub const model_skill = @import("skill.zig").model_skill;

const manage_tools =
    \\[{"type": "function", "function": {"name": "list_models", "description": "This machine's models: state (ready, unloaded), needs_bytes, displaced (unloaded to make room, coming back), and memory.mem_available_bytes.", "parameters": {"type": "object", "properties": {}}}},
    \\ {"type": "function", "function": {"name": "reserve_for", "description": "Make room for a model before its job: dry_run true says what it would unload; false unloads that (never you), loads the model and holds it for its job. What it unloads comes back by itself after the job.", "parameters": {"type": "object", "properties": {"model": {"type": "string"}, "seconds": {"type": "integer"}, "dry_run": {"type": "boolean"}}, "required": ["model"]}}},
    \\ {"type": "function", "function": {"name": "unload_model", "description": "Unload one model now (never you, the chat model); it comes back by itself after hold_s seconds.", "parameters": {"type": "object", "properties": {"model": {"type": "string"}, "hold_s": {"type": "integer"}}, "required": ["model"]}}},
    \\ {"type": "function", "function": {"name": "load_model", "description": "Load a model now, for example one you unloaded.", "parameters": {"type": "object", "properties": {"model": {"type": "string"}}, "required": ["model"]}}}]
;

/// Computer control on the user's paired computer (docs/dev/COMPUTER.md), offered while a desktop helper is connected
/// and the session's device may use it.
const computer_tools =
    \\[{"type": "function", "function": {"name": "computer_control", "description": "Do a simple thing on the user's computer at once: open or quit an app, change the volume, play, pause or skip music, dark mode on or off, lock the screen. Use it only when the user asks for that in this turn. Say briefly what you did after the result comes.",
    \\  "parameters": {"type": "object", "properties": {"control": {"type": "object", "description": "{\"kind\":\"open_app\",\"app\":\"Safari\"}, {\"kind\":\"quit_app\",\"app\":\"Music\"}, {\"kind\":\"volume\",\"action\":\"up|down|mute|unmute|set\",\"level\":0-100,\"target\":\"system\"}, {\"kind\":\"media\",\"action\":\"play|pause|next|previous\",\"player\":\"Spotify|Music\"}, {\"kind\":\"dark_mode\",\"action\":\"on|off|toggle\"}, {\"kind\":\"lock_screen\"}"}}, "required": ["control"]}}},
    \\ {"type": "function", "function": {"name": "computer_use", "description": "Do a small task on the user's computer screen, in a browser tab or an app, when the user asks for it in this turn: find a tab, click through a page, search, start typing a message. It works step by step for up to a minute and a half. Sending, buying, deleting, posting and typing into a message field each need the user to tap Approve in the LocalRouter app; you will be told, and then tell the user. When you call this, only say briefly that you are on it. You get the outcome later as the tool output: then say what happened, or why it stopped.",
    \\  "parameters": {"type": "object", "properties": {"goal": {"type": "string", "description": "what to do on screen, in plain words, with exact names in quotes"},
    \\  "text": {"type": "string", "description": "the exact text to type, if any (a search, a message the user dictated)"}}, "required": ["goal"]}}}]
;

/// Under the lock: the chat request from the settings and the conversation, copied into `a`.
fn request(s: *Session, a: std.mem.Allocator, ask: Task.Ask) !chat.Request {
    const st = &s.settings;
    var msgs: std.ArrayList(chat.Message) = .empty;
    const base = ask.instructions orelse st.instructions;
    const manage = s.voice.tools and s.voice.manage_models and st.tool_execution == .server;
    const instructions = if (manage) try std.fmt.allocPrint(a, "{s}\n\n{s}", .{ base, model_skill }) else base;
    if (instructions.len > 0) try msgs.append(a, .{ .role = "system", .content = try a.dupe(u8, instructions) });
    try history(s, a, &msgs);
    trim(&msgs, s.voice.max_history_chars, if (instructions.len > 0) 1 else 0);
    const tools = try offered(s, a);
    return .{
        .model = if (st.model.len > 0) try a.dupe(u8, st.model) else null,
        .messages = msgs.items,
        .tools = tools,
        .tool_choice = if (tools.len > 0) try events.chatToolChoice(a, st.tool_choice_json) else null,
        .temperature = st.temperature,
        .max_tokens = st.max_tokens,
        .extra = s.voice.chat_extra,
    };
}

/// Under the lock: the tools a reply is offered: the built-in image and video tools a model can serve, the model
/// management tools when enabled, the computer tools when a helper is paired, then the client's own (only those with
/// client tool execution).
fn offered(s: *Session, a: std.mem.Allocator) ![]const chat.Tool {
    var tools: std.ArrayList(chat.Tool) = .empty;
    const offer = s.backend.offer();
    if (s.voice.tools and s.settings.tool_execution == .server) {
        const all = try std.json.parseFromSliceLeaky([]const chat.Tool, a, builtin_tools, .{});
        if (offer.image) try tools.append(a, all[0]);
        if (offer.video) try tools.append(a, all[1]);
        if (s.voice.manage_models) try tools.appendSlice(a, try std.json.parseFromSliceLeaky([]const chat.Tool, a, manage_tools, .{}));
        if (offer.computer) try tools.appendSlice(a, try std.json.parseFromSliceLeaky([]const chat.Tool, a, computer_tools, .{}));
    }
    try tools.appendSlice(a, try events.chatTools(a, s.settings.tools_json));
    return tools.items;
}

/// Under the lock: a completed reply to a user's turn goes to the decision model in shadow (`voice.shadow_decisions`):
/// the backend logs how it would have routed the turn beside what the chat model did, and nothing acts on it.
fn shadow(s: *Session, a: std.mem.Allocator, r: *const Response, calls: []const chat.ToolCall) void {
    if (!s.voice.shadow_decisions or !r.marks.from_turn) return;
    const items = s.items.items;
    var u = items.len;
    while (u > 0) : (u -= 1) {
        const it = items[u - 1];
        if (it.kind == .user and it.text.len > 0 and !it.pending) break;
    } else return; // no user words to decide about
    var previous: []const u8 = "";
    var p = u - 1;
    while (p > 0) : (p -= 1) {
        if (items[p - 1].kind == .assistant and items[p - 1].text.len > 0) {
            previous = items[p - 1].text;
            break;
        }
    }
    const tools = offered(s, a) catch return;
    const called = a.alloc([]const u8, calls.len) catch return;
    for (called, calls) |*n, c| n.* = c.function.name;
    s.backend.shadow(.{
        .session_id = s.id,
        .response_id = r.id,
        .words = items[u - 1].text,
        .previous = previous,
        .tools = tools,
        .called = called,
        .turn_ms = @intCast(@divTrunc(@max(s.clock() - r.marks.t0, 0), std.time.ns_per_ms)),
    });
}

/// What the model reads for a client call cancelled before its output came (Session.expireCalls).
const call_cancelled = "{\"status\":\"cancelled\",\"note\":\"interrupted before it finished: there is no result\"}";

/// The items as chat messages. A run of tool calls becomes one assistant message followed by a tool message for
/// each call: its output when it came right after, else a placeholder (the chat API needs one); an output that
/// arrives later is a system note where it arrived.
fn history(s: *Session, a: std.mem.Allocator, msgs: *std.ArrayList(chat.Message)) !void {
    const items = s.items.items;
    var i: usize = 0;
    while (i < items.len) {
        const it = items[i];
        switch (it.kind) {
            .user, .system => if (it.text.len > 0 and !it.pending) try msgs.append(a, .{ .role = @tagName(it.kind), .content = try a.dupe(u8, it.text) }),
            .assistant => if (it.text.len > 0) try msgs.append(a, .{ .role = "assistant", .content = try a.dupe(u8, it.text) }),
            .function_call_output => {
                const name = if (callNamed(items[0..i], it.call_id)) |c| c.name else "tool";
                try msgs.append(a, .{ .role = "system", .content = try std.fmt.allocPrint(a, "The {s} call {s} finished: {s}", .{ name, it.call_id, it.text }) });
            },
            .function_call => {
                var j = i;
                var calls: std.ArrayList(chat.ToolCall) = .empty;
                while (j < items.len and items[j].kind == .function_call) : (j += 1)
                    try calls.append(a, .{ .id = try a.dupe(u8, items[j].call_id), .function = .{ .name = try a.dupe(u8, items[j].name), .arguments = try a.dupe(u8, items[j].arguments) } });
                const last = if (msgs.items.len > 0) &msgs.items[msgs.items.len - 1] else null;
                const merge = last != null and std.mem.eql(u8, last.?.role, "assistant") and last.?.tool_calls == null and i > 0 and std.mem.eql(u8, items[i - 1].response_id, it.response_id);
                if (merge) last.?.tool_calls = calls.items else try msgs.append(a, .{ .role = "assistant", .tool_calls = calls.items });
                var k = j;
                var answered: std.ArrayList([]const u8) = .empty;
                while (k < items.len and items[k].kind == .function_call_output and hasCall(calls.items, items[k].call_id)) : (k += 1) {
                    try msgs.append(a, .{ .role = "tool", .tool_call_id = try a.dupe(u8, items[k].call_id), .content = try a.dupe(u8, items[k].text) });
                    try answered.append(a, items[k].call_id);
                }
                for (items[i..j], calls.items) |c, cc| if (!contains(answered.items, cc.id))
                    try msgs.append(a, .{ .role = "tool", .tool_call_id = cc.id, .content = if (c.call == .expired) call_cancelled else "{\"status\":\"in_progress\"}" });
                i = k;
                continue;
            },
        }
        i += 1;
    }
}

fn callNamed(items: []const *session.Item, call_id: []const u8) ?*const session.Item {
    for (items) |it| if (it.kind == .function_call and std.mem.eql(u8, it.call_id, call_id)) return it;
    return null;
}

fn hasCall(calls: []const chat.ToolCall, id: []const u8) bool {
    for (calls) |c| if (std.mem.eql(u8, c.id, id)) return true;
    return false;
}

fn contains(list: []const []const u8, x: []const u8) bool {
    for (list) |y| if (std.mem.eql(u8, x, y)) return true;
    return false;
}

/// Drops the oldest messages after the first `keep` until the text fits `max` characters; a tool message never
/// starts what is left.
fn trim(msgs: *std.ArrayList(chat.Message), max: u32, keep: usize) void {
    var total: usize = 0;
    for (msgs.items) |m| total += size(m);
    while (msgs.items.len > keep + 1 and (total > max or std.mem.eql(u8, msgs.items[keep].role, "tool"))) total -= size(msgs.orderedRemove(keep));
}

fn size(m: chat.Message) usize {
    var n: usize = if (m.content) |c| c.len else 0;
    for (m.tool_calls orelse &.{}) |c| n += c.function.arguments.len + c.function.name.len;
    return n;
}

test "history: tool calls group, outputs follow them or become notes, old turns are trimmed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var msgs: std.ArrayList(chat.Message) = .empty;
    try msgs.append(a, .{ .role = "system", .content = "be brief" });
    try msgs.append(a, .{ .role = "tool", .tool_call_id = "x", .content = "orphan" });
    try msgs.append(a, .{ .role = "user", .content = "hello there" });
    try msgs.append(a, .{ .role = "assistant", .content = "hi" });
    trim(&msgs, 1000, 1);
    try std.testing.expectEqual(@as(usize, 3), msgs.items.len); // the leading tool message went
    try std.testing.expectEqualStrings("user", msgs.items[1].role);
    trim(&msgs, 5, 1);
    try std.testing.expectEqual(@as(usize, 2), msgs.items.len); // the oldest turn went, the last stays
    try std.testing.expectEqualStrings("hi", msgs.items[1].content.?);
}
