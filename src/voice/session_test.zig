//! The voice session's state machine against in-process fakes of speech to text, the chat model, text to speech and
//! jobs: spoken turns found by VAD, the history the model sees, sentence-chunked speech, barge-in cancelling both the
//! chat stream and the speech, a tool call that becomes an image job, an edit of the session's last picture, and the
//! error paths.

const std = @import("std");
const Io = std.Io;
const testing = std.testing;
const config = @import("../config.zig");
const backend = @import("backend.zig");
const chat = @import("chat.zig");
const Stop = @import("stop.zig").Stop;
const Session = @import("session.zig").Session;

const Fake = struct {
    io: Io,
    mutex: Io.Mutex = .init,
    arena: std.heap.ArenaAllocator,
    heard: []const u8 = "hello there",
    hears: u32 = 0, // transcriptions made
    delta_ms: u32 = 2,
    speak_ms: u32 = 1,
    requests: std.ArrayList([]const u8) = .empty, // chat request bodies
    spoke: std.ArrayList([]const u8) = .empty,
    specs: std.ArrayList(backend.JobSpec) = .empty,
    chat_cancelled: std.atomic.Value(bool) = .init(false),
    no_inference: bool = false, // the session's device lacks the inference scope
    edit: std.atomic.Value(bool) = .init(false), // a model can edit pictures (image_edit)
    speech_cancelled: std.atomic.Value(bool) = .init(false),
    can_yield: bool = false, // the chat model can be paused for speech (speech first)
    yields: std.atomic.Value(u32) = .init(0), // holds taken or renewed
    held: std.atomic.Value(bool) = .init(false), // a hold is in place now

    fn b(f: *Fake) backend.Backend {
        return .{ .ptr = f, .vtable = &.{ .transcribe = transcribe, .speak = speak, .stream = chatFn, .start_job = startJob, .wait_job = waitJob, .offer = offer, .check = check, .client_tools = clientTools, .yield_chat = yieldChat } };
    }

    fn self(p: *anyopaque) *Fake {
        return @ptrCast(@alignCast(p));
    }

    fn keep(f: *Fake, list: *std.ArrayList([]const u8), text: []const u8) void {
        f.mutex.lockUncancelable(f.io);
        defer f.mutex.unlock(f.io);
        const a = f.arena.allocator();
        list.append(a, a.dupe(u8, text) catch return) catch {};
    }

    /// Sleeps `ms` in 1 ms steps; false once `stop` is set.
    fn nap(f: *Fake, ms: u32, stop: *Stop) bool {
        for (0..ms) |_| {
            if (stop.cancelled()) return false;
            Io.sleep(f.io, .fromMilliseconds(1), .awake) catch {};
        }
        return !stop.cancelled();
    }

    fn transcribe(p: *anyopaque, a: std.mem.Allocator, pcm: []const i16, _: backend.Hear, _: *Stop, _: *backend.Why) backend.Error![]const u8 {
        const f = self(p);
        if (pcm.len == 0) return "";
        f.mutex.lockUncancelable(f.io);
        defer f.mutex.unlock(f.io);
        f.hears += 1;
        return a.dupe(u8, f.heard);
    }

    fn speak(p: *anyopaque, a: std.mem.Allocator, text: []const u8, _: backend.Say, sink: backend.PcmSink, stop: *Stop, why: *backend.Why) backend.Error!void {
        const f = self(p);
        if (std.mem.indexOf(u8, text, "unspeakable") != null) {
            why.set("the fake voice failed", .{});
            return error.Failed;
        }
        if (!f.nap(f.speak_ms, stop)) {
            f.speech_cancelled.store(true, .release);
            return error.Cancelled;
        }
        f.keep(&f.spoke, text);
        const samples = try a.alloc(i16, 2205 * @max(text.len / 10, 1)); // 100 ms at 22.05 kHz per 10 characters
        @memset(samples, 1000);
        const half = samples.len / 2; // streamed, in two pieces
        sink.write(sink.ctx, samples[0..half], 22_050);
        if (!f.nap(f.speak_ms, stop)) return error.Cancelled;
        sink.write(sink.ctx, samples[half..], 22_050);
    }

    fn chatFn(p: *anyopaque, a: std.mem.Allocator, r: *const chat.Request, sink: chat.Sink, stop: *Stop, why: *backend.Why) backend.Error!chat.Result {
        const f = self(p);
        f.keep(&f.requests, chat.body(a, r) catch "");
        const last = r.messages[r.messages.len - 1];
        const said = last.content orelse "";
        if (std.mem.eql(u8, last.role, "tool")) {
            // after the models were listed, the picture; after the picture, the confirmation
            if (std.mem.eql(u8, last.tool_call_id orelse "", "call_lm")) return f.call(a, sink, "call_7", "generate_image", "{\"prompt\": \"a red fox\"}");
            return f.say(a, &.{"Your picture is ready."}, sink, stop);
        }
        // A client's tools: one call (its id repeats each reply, as a chat server's "call_0" does), two in one reply, and
        // a call after a long spoken lead-in (time to cancel it).
        if (std.mem.indexOf(u8, said, "time and weather") != null) return f.calls(a, sink, &.{ .{ .id = "call_a", .function = .{ .name = "get_time", .arguments = "{}" } }, .{ .id = "call_b", .function = .{ .name = "get_weather", .arguments = "{\"city\": \"Oslo\"}" } } });
        if (std.mem.indexOf(u8, said, "slowly check the time") != null) {
            var parts: [40][]const u8 = undefined;
            for (&parts, 0..) |*x, i| x.* = if (i % 2 == 0) "Let me look that up for you" else ". ";
            _ = try f.say(a, &parts, sink, stop);
            return f.call(a, sink, "call_s", "get_time", "{}");
        }
        if (std.mem.indexOf(u8, said, "time") != null) return f.call(a, sink, "call_0", "get_time", "{\"zone\": \"UTC\"}");
        if (std.mem.indexOf(u8, said, "slowly draw") != null) { // a picture call whose arguments take long: barge-in lands in them
            if (sink.tool) |t| t(sink.ctx, "generate_image");
            if (!f.nap(3000, stop)) return error.Cancelled;
            return f.call(a, sink, "call_d", "generate_image", "{\"prompt\": \"a cat\"}");
        }
        if (std.mem.indexOf(u8, said, "check") != null) return f.call(a, sink, "call_lm", "list_models", "{}");
        if (std.mem.indexOf(u8, said, "fail") != null) {
            why.set("the fake model failed", .{});
            return error.Failed;
        }
        if (std.mem.indexOf(u8, said, "edit that") != null) return f.call(a, sink, "call_e", "edit_image", "{\"prompt\": \"make it night\"}");
        if (std.mem.indexOf(u8, said, "picture") != null) return f.call(a, sink, "call_7", "generate_image", "{\"prompt\": \"a red fox\"}");
        if (std.mem.indexOf(u8, said, "slow") != null) {
            var parts: [40][]const u8 = undefined;
            for (&parts, 0..) |*x, i| x.* = try std.fmt.allocPrint(a, "{s}", .{if (i % 2 == 0) "This is one more sentence" else ". "});
            return f.say(a, &parts, sink, stop);
        }
        return f.say(a, &.{ "Hello", " there! How can I", " help you today? I can make", " pictures too." }, sink, stop);
    }

    /// One tool call, reported as a streaming server reports it (its name, before the arguments).
    fn call(f: *Fake, a: std.mem.Allocator, sink: chat.Sink, id: []const u8, name: []const u8, arguments: []const u8) backend.Error!chat.Result {
        return f.calls(a, sink, &.{.{ .id = id, .function = .{ .name = name, .arguments = arguments } }});
    }

    fn calls(_: *Fake, a: std.mem.Allocator, sink: chat.Sink, list: []const chat.ToolCall) backend.Error!chat.Result {
        if (sink.tool) |t| t(sink.ctx, list[0].function.name);
        return .{ .finish_reason = "tool_calls", .tool_calls = try a.dupe(chat.ToolCall, list) };
    }

    fn say(f: *Fake, a: std.mem.Allocator, parts: []const []const u8, sink: chat.Sink, stop: *Stop) backend.Error!chat.Result {
        var text: std.ArrayList(u8) = .empty;
        for (parts) |x| {
            if (!f.nap(f.delta_ms * 15, stop)) {
                f.chat_cancelled.store(true, .release);
                return error.Cancelled;
            }
            sink.text(sink.ctx, x);
            try text.appendSlice(a, x);
        }
        return .{ .finish_reason = "stop", .text = text.items };
    }

    fn startJob(p: *anyopaque, a: std.mem.Allocator, spec: backend.JobSpec, _: *backend.Why) backend.Error!backend.Job {
        const f = self(p);
        f.mutex.lockUncancelable(f.io);
        defer f.mutex.unlock(f.io);
        const fa = f.arena.allocator();
        f.specs.append(fa, .{ .kind = spec.kind, .prompt = try fa.dupe(u8, spec.prompt), .input_file = try fa.dupe(u8, spec.input_file), .input_url = try fa.dupe(u8, spec.input_url) }) catch {};
        return .{ .id = try a.dupe(u8, "img_1"), .kind = spec.kind, .handle = f };
    }

    fn waitJob(_: *anyopaque, a: std.mem.Allocator, _: backend.Job, _: u32, _: *Stop) backend.Outcome {
        const urls = a.alloc([]const u8, 1) catch return .{ .status = "failed" };
        urls[0] = "http://127.0.0.1:8190/v1/files/img_1/0.png";
        return .{ .status = "completed", .urls = urls, .files = &.{"/jobs/img_1/0.png"} };
    }

    fn offer(p: *anyopaque) backend.Offer {
        return .{ .image = true, .image_edit = self(p).edit.load(.acquire) };
    }

    fn yieldChat(p: *anyopaque, _: []const u8, op: backend.Backend.Yield, _: usize) bool {
        const f = self(p);
        switch (op) {
            .check => return f.can_yield,
            .hold => {
                if (!f.can_yield) return false;
                _ = f.yields.fetchAdd(1, .acq_rel);
                f.held.store(true, .release);
            },
            .release => f.held.store(false, .release),
        }
        return true;
    }

    fn clientTools(p: *anyopaque, why: *backend.Why) bool {
        if (!self(p).no_inference) return true;
        why.set("client tool execution needs the inference scope", .{});
        return false;
    }

    fn check(_: *anyopaque, _: backend.Role, name: []const u8, why: *backend.Why) bool {
        if (std.mem.eql(u8, name, "nope")) {
            why.set("unknown model 'nope'", .{});
            return false;
        }
        return true;
    }
};

/// The server events, in order.
const Rec = struct {
    io: Io,
    mutex: Io.Mutex = .init,
    arena: std.heap.ArenaAllocator,
    list: std.ArrayList([]const u8) = .empty,

    fn send(ctx: *anyopaque, text: []const u8) void {
        const r: *Rec = @ptrCast(@alignCast(ctx));
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        const a = r.arena.allocator();
        r.list.append(a, a.dupe(u8, text) catch return) catch {};
    }

    fn typeAt(r: *Rec, i: usize) []const u8 {
        const t = r.list.items[i];
        const at = std.mem.indexOf(u8, t, "\"type\":\"").? + 8;
        return t[at..std.mem.indexOfScalarPos(u8, t, at, '"').?];
    }

    /// The index of the `nth` (0-based) event of `typ`, waiting up to 5 s for it.
    fn wait(r: *Rec, typ: []const u8, nth: usize) !usize {
        for (0..1000) |_| {
            r.mutex.lockUncancelable(r.io);
            var seen: usize = 0;
            for (0..r.list.items.len) |i| if (std.mem.eql(u8, r.typeAt(i), typ)) {
                if (seen == nth) {
                    r.mutex.unlock(r.io);
                    return i;
                }
                seen += 1;
            };
            r.mutex.unlock(r.io);
            Io.sleep(r.io, .fromMilliseconds(5), .awake) catch {};
        }
        std.debug.print("no {s} #{d}; events:\n", .{ typ, nth });
        for (r.list.items) |e| std.debug.print("  {s}\n", .{e[0..@min(e.len, 160)]});
        return error.NoSuchEvent;
    }

    fn count(r: *Rec, typ: []const u8, from: usize) usize {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        var n: usize = 0;
        for (from..r.list.items.len) |i| {
            if (std.mem.eql(u8, r.typeAt(i), typ)) n += 1;
        }
        return n;
    }

    fn get(r: *Rec, a: std.mem.Allocator, i: usize) !std.json.Value {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        return std.json.parseFromSliceLeaky(std.json.Value, a, r.list.items[i], .{});
    }
};

const Rig = struct {
    fake: Fake,
    rec: Rec,
    s: *Session = undefined,

    fn start(rig: *Rig, voice: config.Voice) !void {
        const io = testing.io;
        var v = voice;
        v.warm_speech = false; // the tests count what is spoken
        rig.* = .{ .fake = .{ .io = io, .arena = .init(testing.allocator) }, .rec = .{ .io = io, .arena = .init(testing.allocator) } };
        rig.s = try Session.create(testing.allocator, io, rig.fake.b(), .{ .ctx = &rig.rec, .send = Rec.send }, v, "");
    }

    fn stop(rig: *Rig) void {
        rig.s.destroy();
        rig.fake.arena.deinit();
        rig.rec.arena.deinit();
    }

    /// `ms` of a 440 Hz tone (or silence) as 100 ms input_audio_buffer.append events.
    fn audio(rig: *Rig, ms: u32, tone: bool) !void {
        var buf: [2400]i16 = undefined;
        var bytes: [4800]u8 = undefined;
        var text: [8000]u8 = undefined;
        var t: usize = 0;
        var left = ms;
        while (left > 0) : (left -|= 100) {
            for (&buf) |*x| {
                x.* = if (tone) @intFromFloat(8000 * @sin(@as(f32, @floatFromInt(t)) * 2 * std.math.pi * 440 / 24000)) else 0;
                t += 1;
            }
            for (buf, 0..) |x, i| std.mem.writeInt(i16, bytes[i * 2 ..][0..2], x, .little);
            var b64: [6400]u8 = undefined;
            const enc = std.base64.standard.Encoder.encode(&b64, &bytes);
            rig.s.handle(try std.fmt.bufPrint(&text, "{{\"type\":\"input_audio_buffer.append\",\"audio\":\"{s}\"}}", .{enc}));
        }
    }

    fn heard(rig: *Rig, text: []const u8) void {
        rig.fake.mutex.lockUncancelable(rig.fake.io);
        defer rig.fake.mutex.unlock(rig.fake.io);
        rig.fake.heard = text;
    }
};

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

test "a spoken turn: VAD boundaries, the transcript, a streamed reply in sentences, and the history" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 300 });
    defer rig.stop();
    try rig.audio(300, false);
    try rig.audio(500, true);
    _ = try rig.rec.wait("localrouter.transcription.partial", 0); // while the user still speaks
    try rig.audio(400, true);
    try rig.audio(700, false);
    const done = try rig.rec.wait("response.done", 0);
    const order = [_][]const u8{ "session.created", "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed", "conversation.item.input_audio_transcription.completed", "response.created", "response.audio_transcript.delta", "response.audio.delta", "response.audio.done", "response.done" };
    var last: usize = 0;
    for (order) |typ| {
        const i = try rig.rec.wait(typ, 0);
        try testing.expect(i >= last);
        last = i;
    }
    const started = try rig.rec.get(a, try rig.rec.wait("input_audio_buffer.speech_started", 0));
    try testing.expectEqual(@as(i64, 0), started.object.get("audio_start_ms").?.integer); // onset 300 minus 300 of padding
    const stopped = try rig.rec.get(a, try rig.rec.wait("input_audio_buffer.speech_stopped", 0));
    try testing.expect(@abs(stopped.object.get("audio_end_ms").?.integer - 1700) <= 40); // 300 + 900 + 500
    const d = try rig.rec.get(a, done);
    try testing.expectEqualStrings("completed", d.object.get("response").?.object.get("status").?.string);
    // The first sentence is spoken on its own, as soon as it is complete; the rest follows in larger chunks.
    try testing.expect(rig.fake.spoke.items.len >= 2);
    try testing.expectEqualStrings("Hello there!", rig.fake.spoke.items[0]);

    rig.heard("and once more");
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 1);
    const second = rig.fake.requests.items[1];
    const want = [_][]const u8{ "\"role\":\"system\"", "{\"role\":\"user\",\"content\":\"hello there\"}", "{\"role\":\"assistant\",\"content\":\"Hello there! How can I help you today? I can make pictures too.\"}", "{\"role\":\"user\",\"content\":\"and once more\"}" };
    var at: usize = 0;
    for (want) |w| {
        at = (std.mem.indexOfPos(u8, second, at, w) orelse return error.HistoryOutOfOrder) + w.len;
    }
    try testing.expect(contains(second, "\"name\":\"generate_image\""));
    try testing.expect(!contains(second, "generate_video")); // no video model here
}

test "the chat server is primed with the conversation so far: at the start and after each reply, max_tokens 1" {
    var rig: Rig = undefined;
    try rig.start(.{});
    defer rig.stop();
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 0);
    var primes: usize = 0;
    var after_reply = false;
    for (0..2000) |_| {
        rig.fake.mutex.lockUncancelable(rig.fake.io);
        primes = 0;
        for (rig.fake.requests.items) |body| {
            if (!contains(body, "\"max_tokens\":1")) continue;
            primes += 1;
            if (contains(body, "{\"role\":\"assistant\",\"content\":\"Hello there!")) after_reply = true;
        }
        rig.fake.mutex.unlock(rig.fake.io);
        if (after_reply) break;
        Io.sleep(rig.fake.io, .fromMilliseconds(1), .awake) catch {};
    }
    try testing.expect(after_reply);
    try testing.expect(primes >= 2); // the system prompt alone, then with the turn and its reply
}

test "the wait is the end of the turn: a finished sentence does not end it sooner, and the whole turn is transcribed once" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false });
    defer rig.stop();
    rig.heard("What is two plus two?");
    try rig.audio(300, false);
    try rig.audio(600, true);
    try rig.audio(400, false); // under silence_duration_ms (500): the turn stays open, finished sentence or not
    try testing.expectEqual(@as(usize, 0), rig.rec.count("input_audio_buffer.speech_stopped", 0));
    try rig.audio(300, false);
    _ = try rig.rec.wait("response.done", 0);
    const stopped = try rig.rec.get(a, try rig.rec.wait("input_audio_buffer.speech_stopped", 0));
    try testing.expect(@abs(stopped.object.get("audio_end_ms").?.integer - (900 + 500)) <= 40);
    const heard = try rig.rec.get(a, try rig.rec.wait("conversation.item.input_audio_transcription.completed", 0));
    try testing.expectEqualStrings("What is two plus two?", heard.object.get("transcript").?.string);
    try testing.expectEqual(@as(usize, 0), rig.rec.count("localrouter.transcription.partial", 0)); // nothing before the end
    try testing.expectEqual(@as(u32, 2), rig.fake.hears); // the idle STT's warm-up, then the whole turn once
}

test "pauses under the wait never add up: one turn, nothing transcribed or asked before it ends, then one reply" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false });
    defer rig.stop();
    const s = rig.s;
    s.handle("{\"type\":\"session.update\",\"session\":{\"turn_detection\":{\"type\":\"server_vad\",\"silence_duration_ms\":1000}}}");
    _ = try rig.rec.wait("session.updated", 0);
    rig.heard("I want a cup of tea. With milk.");
    try rig.audio(300, false);
    try rig.audio(600, true);
    for (0..3) |_| { // three 700 ms pauses: 2.1 s of quiet in all, never 1 s in one run
        try rig.audio(700, false);
        Io.sleep(rig.fake.io, .fromMilliseconds(150), .awake) catch {}; // what the worker would do in a pause, it does
        try rig.audio(400, true);
    }
    try testing.expectEqual(@as(usize, 0), rig.rec.count("input_audio_buffer.speech_stopped", 0));
    try testing.expectEqual(@as(usize, 0), rig.rec.count("input_audio_buffer.committed", 0));
    try testing.expectEqual(@as(usize, 0), rig.rec.count("response.created", 0));
    try testing.expectEqual(@as(usize, 0), rig.fake.requests.items.len); // no chat request at all
    try testing.expectEqual(@as(usize, 0), rig.rec.count("conversation.item.input_audio_transcription.completed", 0));
    rig.fake.mutex.lockUncancelable(rig.fake.io);
    const heard_before = rig.fake.hears; // the early transcripts made in the pauses (nothing considered them)
    rig.fake.mutex.unlock(rig.fake.io);
    try rig.audio(1100, false);
    _ = try rig.rec.wait("response.done", 0);
    Io.sleep(rig.fake.io, .fromMilliseconds(100), .awake) catch {};
    try testing.expectEqual(@as(usize, 1), rig.rec.count("input_audio_buffer.speech_stopped", 0));
    try testing.expectEqual(@as(usize, 1), rig.rec.count("response.created", 0));
    try testing.expectEqual(@as(usize, 1), rig.fake.requests.items.len);
    const stopped_at = try rig.rec.wait("input_audio_buffer.speech_stopped", 0);
    const stopped = try rig.rec.get(a, stopped_at);
    try testing.expect(@abs(stopped.object.get("audio_end_ms").?.integer - (300 + 600 + 3 * 1100 + 1000)) <= 40); // 1 s after the last word
    try testing.expect(try rig.rec.wait("response.created", 0) > stopped_at);
    const heard = try rig.rec.get(a, try rig.rec.wait("conversation.item.input_audio_transcription.completed", 0));
    try testing.expectEqualStrings("I want a cup of tea. With milk.", heard.object.get("transcript").?.string);
    try testing.expectEqual(heard_before + 1, rig.fake.hears); // the last pause's early transcript was the turn's: no second decode
}

test "a long reply is spoken once its text is complete (voice.speak_written); sentence by sentence while written without" {
    for ([_]bool{ true, false }) |written| {
        var rig: Rig = undefined;
        try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0, .speak_written = written });
        defer rig.stop();
        rig.heard("tell me something slow"); // 20 sentences, written over about 1.2 s
        try rig.audio(600, true);
        try rig.audio(700, false);
        _ = try rig.rec.wait("response.done", 0);
        const text_done = try rig.rec.wait("localrouter.response.text_done", 0);
        const first_audio = try rig.rec.wait("response.audio.delta", 0);
        // Written first, the chat model has the GPU to itself, and so does speech after it: no audio while it writes.
        try testing.expectEqual(written, first_audio > text_done);
        try testing.expect(rig.fake.spoke.items.len >= 2); // still in sentences, the first on its own
        try testing.expectEqualStrings("This is one more sentence.", rig.fake.spoke.items[0]);
    }
}

test "speech first: the chat model yields while speech is behind; each sentence is spoken as it is written" {
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0, .speak_written = true }); // speech first takes precedence
    defer rig.stop();
    rig.fake.can_yield = true;
    rig.fake.speak_ms = 200; // speech slower than the writing: sentences wait behind the one being spoken
    rig.heard("tell me something slow"); // 20 sentences, written over about 1.2 s
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 0);
    const first = try rig.rec.wait("localrouter.response.speech_first", 0);
    const first_audio = try rig.rec.wait("response.audio.delta", 0);
    try testing.expect(first < first_audio); // told before any audio, so a client can start on it
    try testing.expect(first_audio < try rig.rec.wait("localrouter.response.text_done", 0)); // spoken while written
    try testing.expect(rig.fake.yields.load(.acquire) > 0); // the reply's speech was behind at its start
    try testing.expect(!rig.fake.held.load(.acquire)); // and nothing stays paused after it
}

test "barge-in: speech over the reply cancels the chat stream and the speech, and keeps only what was spoken" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0, .speak_written = false }); // speech while it is written
    defer rig.stop();
    rig.heard("tell me something slow");
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.audio.delta", 0);
    try rig.audio(200, true); // the user talks over the assistant
    const barge = try rig.rec.wait("input_audio_buffer.speech_started", 1);
    const done = try rig.rec.get(a, try rig.rec.wait("response.done", 0));
    const resp = done.object.get("response").?.object;
    try testing.expectEqualStrings("cancelled", resp.get("status").?.string);
    try testing.expectEqualStrings("turn_detected", resp.get("status_details").?.object.get("reason").?.string);
    try testing.expect(rig.fake.chat_cancelled.load(.acquire)); // the model stopped writing
    try testing.expectEqual(@as(usize, 0), rig.rec.count("response.audio.delta", barge)); // not one chunk after speech_started
    const kept = resp.get("output").?.array.items[0].object.get("content").?.array.items[0].object.get("transcript").?.string;
    try testing.expect(kept.len < "This is one more sentence. ".len * 20);
    try testing.expectEqualStrings("incomplete", resp.get("output").?.array.items[0].object.get("status").?.string);
}

test "a tool call becomes an image job; its URL comes back as an event and the assistant confirms it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    rig.heard("make me a picture of a fox");
    try rig.audio(600, true);
    try rig.audio(700, false);
    const args = try rig.rec.get(a, try rig.rec.wait("response.function_call_arguments.done", 0));
    try testing.expectEqualStrings("generate_image", args.object.get("name").?.string);
    try testing.expectEqualStrings("call_7", args.object.get("call_id").?.string);
    const job = try rig.rec.get(a, try rig.rec.wait("localrouter.job.done", 0));
    try testing.expectEqualStrings("completed", job.object.get("status").?.string);
    try testing.expectEqualStrings("http://127.0.0.1:8190/v1/files/img_1/0.png", job.object.get("urls").?.array.items[0].string);
    _ = try rig.rec.wait("response.done", 1);
    try testing.expectEqualStrings("a red fox", rig.fake.specs.items[0].prompt);
    try testing.expectEqual(backend.JobKind.image, rig.fake.specs.items[0].kind);
    const follow = rig.fake.requests.items[1];
    try testing.expect(contains(follow, "\"tool_calls\":[{\"id\":\"call_7\""));
    try testing.expect(contains(follow, "\"role\":\"tool\""));
    try testing.expect(contains(follow, "/v1/files/img_1/0.png"));
    try testing.expect(contains(follow, "\"tool_call_id\":\"call_7\""));
    try testing.expectEqualStrings("Your picture is ready.", rig.fake.spoke.items[rig.fake.spoke.items.len - 1]);
    // Nothing was said before the call: the session says the short "making it" the tool asks for, ends the spoken text
    // there (the arguments are not spoken) and says when that speech is all sent.
    try testing.expectEqualStrings("Making that now.", rig.fake.spoke.items[0]);
    const done = try rig.rec.get(a, try rig.rec.wait("localrouter.response.text_done", 0));
    try testing.expect(done.object.get("tool_calls").?.bool);
    _ = try rig.rec.wait("localrouter.response.speech_done", 0);
    try testing.expect(contains(follow, "\"content\":\"Making that now.\""));
}

test "edit_image is offered only while a model can edit pictures" {
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 0);
    try testing.expect(!contains(rig.fake.requests.items[0], "\"name\":\"edit_image\""));
    rig.fake.edit.store(true, .release);
    rig.heard("and once more");
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 1);
    try testing.expect(contains(rig.fake.requests.items[1], "\"name\":\"edit_image\""));
    try testing.expect(contains(rig.fake.requests.items[1], "\"name\":\"generate_image\""));
}

test "edit_image before any picture: no job, and the model reads that there is no picture yet" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    rig.fake.edit.store(true, .release);
    rig.heard("edit that picture to make it night");
    try rig.audio(600, true);
    try rig.audio(700, false);
    const job = try rig.rec.get(a, try rig.rec.wait("localrouter.job.done", 0));
    try testing.expectEqualStrings("failed", job.object.get("status").?.string);
    _ = try rig.rec.wait("response.done", 1);
    try testing.expectEqual(@as(usize, 0), rig.fake.specs.items.len);
    try testing.expect(contains(rig.fake.requests.items[1], "there is no picture in this conversation yet to edit"));
}

test "the session remembers its last picture once the job completes, and edit_image edits it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    rig.fake.edit.store(true, .release);
    rig.heard("make me a picture of a fox");
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("localrouter.job.done", 0);
    _ = try rig.rec.wait("response.done", 1);
    {
        rig.s.mutex.lockUncancelable(rig.s.io);
        defer rig.s.mutex.unlock(rig.s.io);
        try testing.expectEqualStrings("/jobs/img_1/0.png", rig.s.last_image.?);
    }
    rig.heard("now edit that picture to make it night");
    try rig.audio(600, true);
    try rig.audio(700, false);
    const created = try rig.rec.get(a, try rig.rec.wait("localrouter.job.created", 1));
    try testing.expectEqualStrings("image", created.object.get("kind").?.string);
    const job = try rig.rec.get(a, try rig.rec.wait("localrouter.job.done", 1));
    try testing.expectEqualStrings("completed", job.object.get("status").?.string);
    try testing.expectEqualStrings("call_e", job.object.get("call_id").?.string);
    _ = try rig.rec.wait("response.done", 3);
    try testing.expectEqual(@as(usize, 2), rig.fake.specs.items.len);
    const edit = rig.fake.specs.items[1];
    try testing.expectEqual(backend.JobKind.image, edit.kind);
    try testing.expectEqualStrings("make it night", edit.prompt);
    try testing.expectEqualStrings("/jobs/img_1/0.png", edit.input_file);
}

test "a turn whose answer makes several tool calls still says aloud that the picture is ready" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    rig.heard("check the models and make me a picture of a fox");
    try rig.audio(600, true);
    try rig.audio(700, false);
    // list_models, its output, then generate_image, the job, its output (with the session's own memory check), then the
    // spoken confirmation
    const first = try rig.rec.get(a, try rig.rec.wait("response.function_call_arguments.done", 0));
    try testing.expectEqualStrings("list_models", first.object.get("name").?.string);
    const second = try rig.rec.get(a, try rig.rec.wait("response.function_call_arguments.done", 1));
    try testing.expectEqualStrings("generate_image", second.object.get("name").?.string);
    const job = try rig.rec.wait("localrouter.job.done", 0);
    const confirm = try rig.rec.wait("response.audio_transcript.done", 2);
    try testing.expect(confirm > job);
    try testing.expectEqualStrings("Your picture is ready.", (try rig.rec.get(a, confirm)).object.get("transcript").?.string);
    try testing.expectEqualStrings("Your picture is ready.", rig.fake.spoke.items[rig.fake.spoke.items.len - 1]);
    const output = rig.fake.requests.items[rig.fake.requests.items.len - 1];
    try testing.expect(contains(output, "coming_back")); // the job's output, as the chat model reads it
}

test "text_done carries how fast speech came with the GPU to itself, measured on an earlier reply" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    rig.fake.delta_ms = 0; // the text is complete at once: its later sentences are spoken with the GPU free
    rig.heard("slow please");
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("localrouter.response.speech_done", 0);
    rig.heard("and once more");
    try rig.audio(600, true);
    try rig.audio(700, false);
    const done = try rig.rec.get(a, try rig.rec.wait("localrouter.response.text_done", 1));
    try testing.expect(!done.object.get("tool_calls").?.bool);
    const rate: f64 = switch (done.object.get("speech_rate").?) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => return error.NoSpeechRate,
    };
    try testing.expect(rate > 1); // the fake speaks far faster than real time
}

test "a client seeds its history with a tool call and its output: the chat sees them as a call, nothing runs again" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false });
    defer rig.stop();
    const s = rig.s;
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"a cow please\"}]}}");
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_s1\",\"name\":\"generate_image\",\"arguments\":\"{\\\"prompt\\\":\\\"a cow\\\"}\"}}");
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"function_call_output\",\"call_id\":\"call_s1\",\"output\":\"{\\\"status\\\":\\\"completed\\\",\\\"urls\\\":[\\\"/v1/files/img_9/0.png\\\"]}\"}}");
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Here is your cow.\"}]}}");
    const call = try rig.rec.get(a, try rig.rec.wait("conversation.item.created", 1));
    try testing.expectEqualStrings("function_call", call.object.get("item").?.object.get("type").?.string);
    try testing.expectEqualStrings("call_s1", call.object.get("item").?.object.get("call_id").?.string);
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"function_call\",\"name\":\"generate_image\"}}");
    const e = try rig.rec.get(a, try rig.rec.wait("error", 0));
    try testing.expectEqualStrings("invalid_item", e.object.get("error").?.object.get("code").?.string);
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]}}");
    s.handle("{\"type\":\"response.create\"}");
    _ = try rig.rec.wait("response.done", 0);
    const body = rig.fake.requests.items[rig.fake.requests.items.len - 1];
    try testing.expect(contains(body, "\"tool_calls\":[{\"id\":\"call_s1\""));
    try testing.expect(contains(body, "\"tool_call_id\":\"call_s1\""));
    try testing.expect(contains(body, "/v1/files/img_9/0.png"));
    try testing.expectEqual(@as(usize, 0), rig.fake.specs.items.len); // a seeded call is history: no job
}

test "error paths, push to talk and a text-only response" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false });
    defer rig.stop();
    const s = rig.s;
    const cases = [_]struct { event: []const u8, code: []const u8 }{
        .{ .event = "not json", .code = "invalid_json" },
        .{ .event = "{\"type\":\"session.dance\"}", .code = "unknown_event_type" },
        .{ .event = "{\"type\":\"input_audio_buffer.append\",\"audio\":\"***\"}", .code = "invalid_audio" },
        .{ .event = "{\"type\":\"session.update\",\"session\":{\"input_audio_format\":\"g711_ulaw\"}}", .code = "invalid_session" },
        .{ .event = "{\"type\":\"session.update\",\"session\":{\"speech_model\":\"nope\"}}", .code = "invalid_session" },
        .{ .event = "{\"type\":\"response.cancel\"}", .code = "response_cancel_not_active" },
        .{ .event = "{\"type\":\"conversation.item.delete\",\"item_id\":\"missing\"}", .code = "item_delete_invalid_item_id" },
        .{ .event = "{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\",\"role\":\"user\",\"content\":[]}}", .code = "invalid_item" },
        .{ .event = "{\"type\":\"input_audio_buffer.commit\"}", .code = "input_audio_buffer_commit_empty" },
    };
    for (cases, 0..) |c, n| {
        s.handle(c.event);
        const e = try rig.rec.get(a, try rig.rec.wait("error", n));
        try testing.expectEqualStrings(c.code, e.object.get("error").?.object.get("code").?.string);
    }
    s.refuseBinary();
    _ = try rig.rec.wait("error", cases.len);

    // Push to talk: no VAD; a commit makes the item, and only response.create answers it.
    s.handle("{\"type\":\"session.update\",\"session\":{\"turn_detection\":null}}");
    _ = try rig.rec.wait("session.updated", 0);
    try rig.audio(500, true);
    try testing.expectEqual(@as(usize, 0), rig.rec.count("input_audio_buffer.speech_started", 0));
    s.handle("{\"type\":\"input_audio_buffer.commit\"}");
    _ = try rig.rec.wait("conversation.item.input_audio_transcription.completed", 0);
    Io.sleep(testing.io, .fromMilliseconds(100), .awake) catch {};
    try testing.expectEqual(@as(usize, 0), rig.rec.count("response.created", 0));
    s.handle("{\"type\":\"response.create\"}");
    _ = try rig.rec.wait("response.done", 0);

    // The chat model fails: an error event, and the response is failed.
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"please fail\"}]}}");
    s.handle("{\"type\":\"response.create\"}");
    const failed = try rig.rec.get(a, try rig.rec.wait("response.done", 1));
    try testing.expectEqualStrings("failed", failed.object.get("response").?.object.get("status").?.string);
    _ = try rig.rec.wait("error", cases.len + 1);

    // Text only: text deltas, no audio.
    s.handle("{\"type\":\"session.update\",\"session\":{\"modalities\":[\"text\"]}}");
    _ = try rig.rec.wait("session.updated", 1);
    s.handle("{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]}}");
    const before = rig.rec.count("response.audio.delta", 0);
    s.handle("{\"type\":\"response.create\"}");
    _ = try rig.rec.wait("response.done", 2);
    _ = try rig.rec.wait("response.text.delta", 0);
    try testing.expectEqual(before, rig.rec.count("response.audio.delta", 0));
}

const client_mode =
    \\{"type":"session.update","session":{"localrouter":{"tool_execution":"client"},"tools":[
    \\ {"type":"function","name":"generate_image","description":"the client's own","parameters":{"type":"object"}},
    \\ {"type":"function","name":"get_time","parameters":{"type":"object"}},
    \\ {"type":"function","name":"get_weather","parameters":{"type":"object"}}]}}
;

fn typed(s: *Session, text: []const u8) !void {
    var buf: [512]u8 = undefined;
    s.handle(try std.fmt.bufPrint(&buf, "{{\"type\":\"conversation.item.create\",\"item\":{{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"{s}\"}}]}}}}", .{text}));
}

fn sendOutput(s: *Session, call_id: []const u8, out: []const u8) !void {
    var buf: [512]u8 = undefined;
    s.handle(try std.fmt.bufPrint(&buf, "{{\"type\":\"conversation.item.create\",\"item\":{{\"type\":\"function_call_output\",\"call_id\":\"{s}\",\"output\":\"{s}\"}}}}", .{ call_id, out }));
}

fn errorCode(rig: *Rig, a: std.mem.Allocator, nth: usize) ![]const u8 {
    const e = try rig.rec.get(a, try rig.rec.wait("error", nth));
    return e.object.get("error").?.object.get("code").?.string;
}

test "client tool execution: only the client's tools exist, a call waits for its one output, and the continuation speaks" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .manage_models = true });
    defer rig.stop();
    const s = rig.s;
    s.handle(client_mode);
    const updated = try rig.rec.get(a, try rig.rec.wait("session.updated", 0));
    try testing.expectEqualStrings("client", updated.object.get("session").?.object.get("localrouter").?.object.get("tool_execution").?.string);

    try typed(s, "what time is it");
    s.handle("{\"type\":\"response.create\"}");
    const args = try rig.rec.get(a, try rig.rec.wait("response.function_call_arguments.done", 0));
    try testing.expectEqualStrings("get_time", args.object.get("name").?.string);
    try testing.expectEqualStrings("call_0", args.object.get("call_id").?.string);
    try testing.expectEqualStrings("{\"zone\": \"UTC\"}", args.object.get("arguments").?.string); // complete, once
    _ = try rig.rec.wait("response.done", 0);
    try testing.expectEqual(@as(usize, 1), rig.rec.count("response.function_call_arguments.done", 0));
    // Offered: the client's three tools, none of the server's, and not the model management procedure.
    const first = rig.fake.requests.items[0];
    try testing.expect(contains(first, "\"name\":\"get_time\"") and contains(first, "\"name\":\"generate_image\""));
    try testing.expect(!contains(first, "generate_video") and !contains(first, "list_models") and !contains(first, "reserve_for") and !contains(first, "computer_"));
    try testing.expect(contains(first, "the client's own"));
    // Nothing ran here and nothing was said for it.
    Io.sleep(testing.io, .fromMilliseconds(50), .awake) catch {};
    try testing.expectEqual(@as(usize, 0), rig.fake.specs.items.len);
    try testing.expectEqual(@as(usize, 0), rig.rec.count("localrouter.job.created", 0));
    try testing.expectEqual(@as(usize, 0), rig.fake.spoke.items.len);
    try testing.expectEqual(@as(usize, 1), rig.rec.count("response.created", 0)); // no follow-up of its own

    try sendOutput(s, "call_x", "noon");
    try testing.expectEqualStrings("unknown_call_id", try errorCode(&rig, a, 0));
    try sendOutput(s, "call_0", "noon");
    _ = try rig.rec.wait("conversation.item.created", 3); // user, call, (assistant item), output
    try sendOutput(s, "call_0", "noon again");
    try testing.expectEqualStrings("duplicate_call_output", try errorCode(&rig, a, 1));
    s.handle("{\"type\":\"response.create\"}");
    const cont = try rig.rec.get(a, try rig.rec.wait("response.done", 1));
    try testing.expectEqualStrings("completed", cont.object.get("response").?.object.get("status").?.string);
    try testing.expectEqualStrings("Your picture is ready.", rig.fake.spoke.items[rig.fake.spoke.items.len - 1]); // heard
    const second = rig.fake.requests.items[1];
    try testing.expect(contains(second, "{\"role\":\"tool\",\"content\":\"noon\",\"tool_call_id\":\"call_0\"}") or contains(second, "\"tool_call_id\":\"call_0\",\"content\":\"noon\""));
    try testing.expect(!contains(second, "noon again"));

    // The mode is the session's from its first response on.
    s.handle("{\"type\":\"session.update\",\"session\":{\"localrouter\":{\"tool_execution\":\"server\"}}}");
    try testing.expectEqualStrings("invalid_session", try errorCode(&rig, a, 2));

    // The model's next call reuses its id: the session gives it its own, and a response begun without its output
    // cancels it, so the output that comes after is refused.
    try typed(s, "and the time again");
    s.handle("{\"type\":\"response.create\"}");
    const again = try rig.rec.get(a, try rig.rec.wait("response.function_call_arguments.done", 1));
    const minted = again.object.get("call_id").?.string;
    try testing.expect(!std.mem.eql(u8, minted, "call_0"));
    _ = try rig.rec.wait("response.done", 2);
    s.handle("{\"type\":\"response.create\"}");
    const cancelled = try rig.rec.get(a, try rig.rec.wait("localrouter.function_call.cancelled", 0));
    try testing.expectEqualStrings(minted, cancelled.object.get("call_id").?.string);
    try testing.expectEqualStrings("response_started", cancelled.object.get("reason").?.string);
    _ = try rig.rec.wait("response.done", 3);
    try sendOutput(s, minted, "too late");
    try testing.expectEqualStrings("call_expired", try errorCode(&rig, a, 3));
    try testing.expect(contains(rig.fake.requests.items[3], "\"status\\\":\\\"cancelled\\\""));
}

test "client tool execution: several calls in one reply; barge-in cancels the waiting one, and its late output never answers the new turn" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    const s = rig.s;
    s.handle(client_mode);
    _ = try rig.rec.wait("session.updated", 0);
    rig.heard("what is the time and weather");
    try rig.audio(600, true);
    try rig.audio(700, false);
    const first = try rig.rec.get(a, try rig.rec.wait("response.function_call_arguments.done", 0));
    const second = try rig.rec.get(a, try rig.rec.wait("response.function_call_arguments.done", 1));
    try testing.expectEqualStrings("call_a", first.object.get("call_id").?.string);
    try testing.expectEqualStrings("call_b", second.object.get("call_id").?.string);
    try testing.expectEqual(@as(i64, 2), second.object.get("output_index").?.integer);
    const done = try rig.rec.get(a, try rig.rec.wait("response.done", 0));
    try testing.expectEqual(@as(usize, 3), done.object.get("response").?.object.get("output").?.array.items.len); // the message and both calls
    try sendOutput(s, "call_a", "noon");
    _ = try rig.rec.wait("conversation.item.created", 4);

    rig.heard("never mind, and once more");
    try rig.audio(600, true); // the user talks before the weather came back
    const cancelled = try rig.rec.get(a, try rig.rec.wait("localrouter.function_call.cancelled", 0));
    try testing.expectEqualStrings("call_b", cancelled.object.get("call_id").?.string);
    try testing.expectEqualStrings("turn_detected", cancelled.object.get("reason").?.string);
    try testing.expectEqual(@as(usize, 1), rig.rec.count("localrouter.function_call.cancelled", 0)); // call_a was answered
    try sendOutput(s, "call_b", "sunny"); // late: while the user still speaks
    try testing.expectEqualStrings("call_expired", try errorCode(&rig, a, 0));
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 1);
    try sendOutput(s, "call_b", "sunny"); // late: after the new turn's answer
    try testing.expectEqualStrings("call_expired", try errorCode(&rig, a, 1));
    Io.sleep(testing.io, .fromMilliseconds(100), .awake) catch {};
    try testing.expectEqual(@as(usize, 2), rig.rec.count("response.created", 0)); // the late output started nothing
    const body = rig.fake.requests.items[1];
    try testing.expect(contains(body, "noon") and !contains(body, "sunny"));
    try testing.expect(contains(body, "\"tool_call_id\":\"call_b\""));
    try testing.expect(contains(body, "never mind, and once more"));
}

test "client tool execution: a cancelled response makes no call, and nothing waits for an output" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false });
    defer rig.stop();
    const s = rig.s;
    s.handle(client_mode);
    _ = try rig.rec.wait("session.updated", 0);
    try typed(s, "slowly check the time");
    s.handle("{\"type\":\"response.create\"}");
    _ = try rig.rec.wait("response.audio_transcript.delta", 0);
    s.handle("{\"type\":\"response.cancel\"}");
    const done = try rig.rec.get(a, try rig.rec.wait("response.done", 0));
    try testing.expectEqualStrings("cancelled", done.object.get("response").?.object.get("status").?.string);
    try testing.expectEqual(@as(usize, 0), rig.rec.count("response.function_call_arguments.done", 0));
    try sendOutput(s, "call_s", "noon");
    try testing.expectEqualStrings("unknown_call_id", try errorCode(&rig, a, 0));
}

test "server tool execution stays the default: a client's tool may not take a built-in name" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false });
    defer rig.stop();
    const created = try rig.rec.get(a, try rig.rec.wait("session.created", 0));
    try testing.expectEqualStrings("server", created.object.get("session").?.object.get("localrouter").?.object.get("tool_execution").?.string);
    rig.s.handle("{\"type\":\"session.update\",\"session\":{\"tools\":[{\"type\":\"function\",\"name\":\"generate_image\"}]}}");
    try testing.expectEqualStrings("invalid_session", try errorCode(&rig, a, 0));
}

test "client tool execution needs the inference scope: without it the session stays on server tools" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false });
    defer rig.stop();
    rig.fake.no_inference = true;
    rig.s.handle(client_mode);
    const e = try rig.rec.get(a, try rig.rec.wait("error", 0));
    try testing.expectEqualStrings("invalid_session", e.object.get("error").?.object.get("code").?.string);
    try testing.expectEqualStrings("client tool execution needs the inference scope", e.object.get("error").?.object.get("message").?.string);
    try testing.expectEqual(@as(usize, 0), rig.rec.count("session.updated", 0));
    rig.fake.no_inference = false;
    rig.s.handle(client_mode);
    const updated = try rig.rec.get(a, try rig.rec.wait("session.updated", 0));
    try testing.expectEqualStrings("client", updated.object.get("session").?.object.get("localrouter").?.object.get("tool_execution").?.string);
}

test "a barge-in while a picture call is written: no call, and the session's \"Making that now.\" leaves the history with it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .prime_chat = false, .partial_interval_ms = 0 });
    defer rig.stop();
    rig.heard("slowly draw me a cat");
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("localrouter.response.speech_done", 0); // "Making that now." was said while the arguments are written
    try testing.expectEqualStrings("Making that now.", rig.fake.spoke.items[0]);
    rig.heard("never mind, hello there");
    try rig.audio(600, true);
    const cut = try rig.rec.get(a, try rig.rec.wait("response.done", 0));
    try testing.expectEqualStrings("cancelled", cut.object.get("response").?.object.get("status").?.string);
    try testing.expectEqual(@as(usize, 0), rig.rec.count("response.function_call_arguments.done", 0));
    const said = cut.object.get("response").?.object.get("output").?.array.items[0].object.get("content").?.array.items[0].object.get("transcript").?.string;
    try testing.expectEqualStrings("", said);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 1);
    const next = rig.fake.requests.items[rig.fake.requests.items.len - 1];
    try testing.expect(contains(next, "never mind, hello there"));
    try testing.expect(!contains(next, "Making that now."));
}
