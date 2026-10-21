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
const tools = @import("tools.zig");
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
    text: std.ArrayList(u8) = .empty,
    queue: std.ArrayList(Chunk) = .empty, // sentences for the speaker
    closed: bool = false, // no more sentences will come
    cond: Io.Condition = .init,
    spoken: std.ArrayList(session.Item.Spoken) = .empty, // gpa: moves to the item
    audio_ms: u64 = 0, // sent so far
    marks: timing.Marks = .{},

    const Chunk = struct { text: []const u8, text_end: usize };
};

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
            .partial => |h| hear(s, h, false),
            .final => |h| hear(s, h, true),
            .respond => |ask| run(s, ask),
        }
    }
}

// ---------------------------------------------------------------- transcription

fn hear(s: *Session, h: Task.Hearing, final: bool) void {
    var tmp: std.heap.ArenaAllocator = .init(s.gpa);
    defer tmp.deinit();
    const a = tmp.allocator();
    s.mutex.lockUncancelable(s.io);
    if (!final and s.find(h.item_id) != null) { // committed meanwhile: the final transcript is coming
        s.partial_busy = false;
        s.mutex.unlock(s.io);
        return;
    }
    const st = &s.settings;
    const o: backend.Hear = .{ .model = a.dupe(u8, st.transcription_model) catch "", .language = a.dupe(u8, st.language) catch "", .prompt = a.dupe(u8, st.transcription_prompt) catch "" };
    s.mutex.unlock(s.io);
    var why: backend.Why = .{};
    var worker_ms: u64 = 0;
    var heard = o;
    heard.worker_ms = &worker_ms;
    const began = s.clock();
    const text: backend.Error![]const u8 = if (resample.resample(a, h.pcm, session.rate, 16_000)) |pcm16|
        s.backend.transcribe(a, pcm16, heard, &s.close_stop, &why)
    else |_|
        error.OutOfMemory;
    s.mutex.lockUncancelable(s.io);
    defer s.mutex.unlock(s.io);
    if (final) {
        s.stt_start_ns = began;
        s.stt_end_ns = s.clock();
        s.stt_worker_ms = worker_ms;
    }
    if (!final) {
        s.partial_busy = false;
        const t = text catch return;
        const turn = s.turn orelse return;
        if (std.mem.eql(u8, turn.item_id, h.item_id)) s.emit.send(a, "localrouter.transcription.partial", .{ .item_id = h.item_id, .transcript = std.mem.trim(u8, t, " \t\r\n") });
        return;
    }
    const it = s.find(h.item_id) orelse return; // deleted meanwhile
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
    const a = r.arena.allocator();
    var req: chat.Request = undefined;
    var say: backend.Say = undefined;
    if (!begin(s, r, ask, &req, &say)) return;
    const speaker: ?std.Thread = if (r.audio) std.Thread.spawn(.{}, speak, .{ s, r, say }) catch null else null;
    var feed: Feed = .{ .s = s, .r = r, .chunks = .{ .gpa = s.gpa } };
    defer feed.chunks.deinit();
    defer feed.think.deinit(s.gpa);
    defer feed.lead.deinit(s.gpa);
    var why: backend.Why = .{};
    r.marks.chat_start = s.clock();
    const result = s.backend.stream(a, &req, .{ .ctx = &feed, .text = Feed.onText }, &r.stop, &why);
    feed.end();
    if (speaker) |t| t.join();
    finish(s, r, result, &why);
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
    r.audio = ask.audio orelse s.settings.audio;
    r.marks.from_turn = ask.auto and s.stop_ns > 0; // answering a spoken turn: timed from its end
    r.marks.t0 = if (r.marks.from_turn) s.stop_ns else s.clock();
    if (r.marks.from_turn) {
        r.marks.stt_start = s.stt_start_ns;
        r.marks.stt_end = s.stt_end_ns;
        r.marks.stt_worker_ms = s.stt_worker_ms;
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

    fn onText(ctx: *anyopaque, delta: []const u8) void {
        const f: *Feed = @ptrCast(@alignCast(ctx));
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
        const rest = f.think.flush(f.s.gpa, f.r.arena.allocator()) catch "";
        f.add(f.lead.pass(f.s.gpa, f.r.arena.allocator(), rest, true) catch rest, true);
        f.s.mutex.lockUncancelable(f.s.io);
        f.r.closed = true;
        f.r.cond.signal(f.s.io);
        f.s.mutex.unlock(f.s.io);
    }
};

/// The speaker thread: one speech job a sentence, sent as 100 ms audio deltas unless the response was cancelled.
fn speak(s: *Session, r: *Response, say: backend.Say) void {
    var next: usize = 0;
    while (true) {
        s.mutex.lockUncancelable(s.io);
        while (next >= r.queue.items.len and !r.closed and !r.stop.cancelled()) r.cond.waitUncancelable(s.io, &s.mutex);
        if (r.stop.cancelled() or next >= r.queue.items.len) {
            s.mutex.unlock(s.io);
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
    return true;
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
        const prev = s.lastId();
        const it = s.addItem(.function_call, id) catch break;
        const ia = it.arena.allocator();
        it.call_id = ia.dupe(u8, call.id) catch "";
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
        if (events.isBuiltin(it.name) and s.voice.tools) tools.start(s, it);
    }
    s.emit.send(a, "localrouter.response.timing", timing.event(r.id, r.marks));
    s.emit.send(a, "response.done", .{ .response = responseValue(r, status, details, outputs.items) });
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

const builtin_tools =
    \\[{"type": "function", "function": {"name": "generate_image", "description": "Make a picture on this machine from a detailed description. The user gets the file; say briefly that it is being made.",
    \\  "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "what to draw, in detail"},
    \\  "size": {"type": "string", "description": "WIDTHxHEIGHT in multiples of 16, e.g. 1024x1024 (default) or 1344x768"}}, "required": ["prompt"]}}},
    \\ {"type": "function", "function": {"name": "generate_video", "description": "Make a short video clip with sound on this machine from a description. It takes minutes; say so.",
    \\  "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "what happens, in detail"},
    \\  "seconds": {"type": "integer", "description": "length, 1 to 20 (default 5)"}}, "required": ["prompt"]}}}]
;

/// Under the lock: the chat request from the settings and the conversation, copied into `a`.
fn request(s: *Session, a: std.mem.Allocator, ask: Task.Ask) !chat.Request {
    const st = &s.settings;
    var msgs: std.ArrayList(chat.Message) = .empty;
    const instructions = ask.instructions orelse st.instructions;
    if (instructions.len > 0) try msgs.append(a, .{ .role = "system", .content = try a.dupe(u8, instructions) });
    try history(s, a, &msgs);
    trim(&msgs, s.voice.max_history_chars, if (instructions.len > 0) 1 else 0);
    var tools: std.ArrayList(chat.Tool) = .empty;
    const offer = s.backend.offer();
    if (s.voice.tools) {
        const all = try std.json.parseFromSliceLeaky([]const chat.Tool, a, builtin_tools, .{});
        if (offer.image) try tools.append(a, all[0]);
        if (offer.video) try tools.append(a, all[1]);
    }
    try tools.appendSlice(a, try events.chatTools(a, st.tools_json));
    return .{
        .model = if (st.model.len > 0) try a.dupe(u8, st.model) else null,
        .messages = msgs.items,
        .tools = tools.items,
        .tool_choice = if (tools.items.len > 0) try events.chatToolChoice(a, st.tool_choice_json) else null,
        .temperature = st.temperature,
        .max_tokens = st.max_tokens,
    };
}

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
                for (calls.items) |c| if (!contains(answered.items, c.id))
                    try msgs.append(a, .{ .role = "tool", .tool_call_id = c.id, .content = "{\"status\":\"in_progress\"}" });
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
