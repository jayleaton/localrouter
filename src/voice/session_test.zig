//! The voice session's state machine against in-process fakes of speech to text, the chat model, text to speech and
//! jobs: spoken turns found by VAD, the history the model sees, sentence-chunked speech, barge-in cancelling both the
//! chat stream and the speech, a tool call that becomes an image job, and the error paths.

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
    speech_cancelled: std.atomic.Value(bool) = .init(false),

    fn b(f: *Fake) backend.Backend {
        return .{ .ptr = f, .vtable = &.{ .transcribe = transcribe, .speak = speak, .stream = chatFn, .start_job = startJob, .wait_job = waitJob, .offer = offer, .check = check } };
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
        if (std.mem.eql(u8, last.role, "tool")) return f.say(a, &.{"Your picture is ready."}, sink, stop);
        if (std.mem.indexOf(u8, said, "fail") != null) {
            why.set("the fake model failed", .{});
            return error.Failed;
        }
        if (std.mem.indexOf(u8, said, "picture") != null) {
            const calls = try a.alloc(chat.ToolCall, 1);
            calls[0] = .{ .id = "call_7", .function = .{ .name = "generate_image", .arguments = "{\"prompt\": \"a red fox\"}" } };
            return .{ .finish_reason = "tool_calls", .tool_calls = calls };
        }
        if (std.mem.indexOf(u8, said, "slow") != null) {
            var parts: [40][]const u8 = undefined;
            for (&parts, 0..) |*x, i| x.* = try std.fmt.allocPrint(a, "{s}", .{if (i % 2 == 0) "This is one more sentence" else ". "});
            return f.say(a, &parts, sink, stop);
        }
        return f.say(a, &.{ "Hello", " there! How can I", " help you today? I can make", " pictures too." }, sink, stop);
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
        f.specs.append(fa, .{ .kind = spec.kind, .prompt = try fa.dupe(u8, spec.prompt) }) catch {};
        return .{ .id = try a.dupe(u8, "img_1"), .kind = spec.kind, .handle = f };
    }

    fn waitJob(_: *anyopaque, a: std.mem.Allocator, _: backend.Job, _: u32, _: *Stop) backend.Outcome {
        const urls = a.alloc([]const u8, 1) catch return .{ .status = "failed" };
        urls[0] = "http://127.0.0.1:8190/v1/files/img_1/0.png";
        return .{ .status = "completed", .urls = urls };
    }

    fn offer(_: *anyopaque) backend.Offer {
        return .{ .image = true };
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
        rig.* = .{ .fake = .{ .io = io, .arena = .init(testing.allocator) }, .rec = .{ .io = io, .arena = .init(testing.allocator) } };
        rig.s = try Session.create(testing.allocator, io, rig.fake.b(), .{ .ctx = &rig.rec, .send = Rec.send }, voice, "");
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
    try rig.start(.{ .partial_interval_ms = 300 });
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

test "a pause starts a draft transcript: one ending a sentence ends the turn early, and the final transcript reuses it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{});
    defer rig.stop();
    rig.heard("What is two plus two?");
    try rig.audio(300, false);
    try rig.audio(600, true);
    try rig.audio(400, false); // under silence_duration_ms (500): only the draft's sentence end can end this turn
    _ = try rig.rec.wait("response.done", 0);
    const stopped = try rig.rec.get(a, try rig.rec.wait("input_audio_buffer.speech_stopped", 0));
    const end = stopped.object.get("audio_end_ms").?.integer;
    try testing.expect(end >= 900 + 300 and end <= 1300); // speech ends at 900; early_end_ms 300
    const heard = try rig.rec.get(a, try rig.rec.wait("conversation.item.input_audio_transcription.completed", 0));
    try testing.expectEqualStrings("What is two plus two?", heard.object.get("transcript").?.string);
    try testing.expectEqual(@as(u32, 2), rig.fake.hears); // the first speech warms the idle STT, then the draft only

    rig.heard("and once more"); // no sentence end: the turn waits for silence_duration_ms, then reuses the draft
    try rig.audio(600, true);
    try rig.audio(700, false);
    _ = try rig.rec.wait("response.done", 1);
    const second = try rig.rec.get(a, try rig.rec.wait("input_audio_buffer.speech_stopped", 1));
    try testing.expect(@abs(second.object.get("audio_end_ms").?.integer - (1300 + 600 + 500)) <= 40);
    try testing.expectEqual(@as(u32, 3), rig.fake.hears); // no warm-up: it transcribed just now
}

test "barge-in: speech over the reply cancels the chat stream and the speech, and keeps only what was spoken" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{ .partial_interval_ms = 0 });
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
    try rig.start(.{ .partial_interval_ms = 0 });
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
}

test "error paths, push to talk and a text-only response" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig: Rig = undefined;
    try rig.start(.{});
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
