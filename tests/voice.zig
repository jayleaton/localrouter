//! Integration tests of the voice loop on a real daemon over a real WebSocket: speech to text and text to speech by
//! external workers (tests/fixtures/speech_worker.py: what was "said" is chosen by the tone's pitch), the chat model
//! by an OpenAI-compatible streaming fake (tests/fixtures/voice_chat_server.py), and an image tool by the testpattern engine.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const sleepMs = harness.sleepMs;

/// The fake chat server, on an ephemeral port; it exits when its stdin closes.
const ChatServer = struct {
    child: std.process.Child,
    port: u16,

    fn start(io: Io) !ChatServer {
        var child = try std.process.spawn(io, .{ .argv = &.{ "python3", opts.voice_chat_server, "--delay-ms", "5" }, .stdin = .pipe, .stdout = .pipe, .stderr = .inherit });
        var buf: [128]u8 = undefined;
        var r = child.stdout.?.reader(io, &buf);
        const line = try r.interface.takeDelimiterExclusive('\n');
        const port = try std.fmt.parseInt(u16, std.mem.trim(u8, line["listening ".len..], " \r"), 10);
        return .{ .child = child, .port = port };
    }

    fn stop(c: *ChatServer, io: Io) void {
        _ = std.posix.system.kill(c.child.id.?, std.posix.SIG.TERM);
        _ = c.child.wait(io) catch {};
    }

    const Record = struct { messages: []const std.json.Value = &.{}, tools: []const []const u8 = &.{}, completed: bool = false, cancelled: bool = false };

    fn stats(c: *ChatServer, d: *Daemon, a: std.mem.Allocator) ![]const Record {
        var out: Io.Writer.Allocating = .init(a);
        const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/stats", .{c.port});
        _ = try d.client.fetch(.{ .location = .{ .url = url }, .response_writer = &out.writer, .keep_alive = false });
        const T = struct { requests: []const Record };
        return (try std.json.parseFromSliceLeaky(T, a, out.written(), .{ .ignore_unknown_fields = true })).requests;
    }
};

/// A minimal WebSocket client: masked text frames out, server frames in. A watchdog shuts the socket down when a
/// read waits longer than 20 s, so a missing event fails the test instead of hanging it.
pub const Ws = struct {
    io: Io,
    stream: Io.net.Stream,
    rbuf: [16 * 1024]u8 = undefined,
    wbuf: [16 * 1024]u8 = undefined,
    reader: Io.net.Stream.Reader = undefined,
    writer: Io.net.Stream.Writer = undefined,
    head: []const u8 = "",
    deadline_ns: std.atomic.Value(i64) = .init(std.math.maxInt(i64)),
    closing: std.atomic.Value(bool) = .init(false),
    dog: ?std.Thread = null,

    pub fn open(w: *Ws, io: Io, a: std.mem.Allocator, port: u16, protocol: []const u8) !void {
        return w.openAt(io, a, port, "/v1/realtime", protocol);
    }

    /// `open` with any request target (a query string such as `?api_key=`); `head` is the response head, whatever it is.
    pub fn openAt(w: *Ws, io: Io, a: std.mem.Allocator, port: u16, target: []const u8, protocol: []const u8) !void {
        const addr = try Io.net.IpAddress.parse("127.0.0.1", port);
        w.* = .{ .io = io, .stream = try addr.connect(io, .{ .mode = .stream }) };
        w.reader = w.stream.reader(io, &w.rbuf);
        w.writer = w.stream.writer(io, &w.wbuf);
        const extra = if (protocol.len > 0) try std.fmt.allocPrint(a, "Sec-WebSocket-Protocol: {s}\r\n", .{protocol}) else "";
        try w.writer.interface.print("GET {s} HTTP/1.1\r\nHost: 127.0.0.1:{d}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n{s}\r\n", .{ target, port, extra });
        try w.writer.interface.flush();
        var head: std.ArrayList(u8) = .empty;
        while (true) {
            const line = try w.reader.interface.takeDelimiterInclusive('\n');
            try head.appendSlice(a, line);
            if (std.mem.eql(u8, line, "\r\n")) break;
        }
        w.head = head.items;
        w.dog = try std.Thread.spawn(.{}, watch, .{w});
    }

    fn watch(w: *Ws) void {
        while (!w.closing.load(.acquire)) {
            if (now(w.io) > w.deadline_ns.load(.acquire)) {
                std.debug.print("voice test: no WebSocket message within 20 s\n", .{});
                _ = std.posix.system.shutdown(w.stream.socket.handle, std.posix.SHUT.RDWR);
                return;
            }
            sleepMs(w.io, 50);
        }
    }

    pub fn close(w: *Ws) void {
        w.closing.store(true, .release);
        if (w.dog) |t| t.join();
        w.stream.close(w.io);
    }

    pub fn send(w: *Ws, text: []const u8) !void {
        const out = &w.writer.interface;
        const mask = [4]u8{ 0x37, 0xfa, 0x21, 0x3d };
        try out.writeByte(0x81);
        if (text.len < 126) try out.writeByte(0x80 | @as(u8, @intCast(text.len))) else if (text.len <= 0xffff) {
            try out.writeByte(0x80 | 126);
            try out.writeInt(u16, @intCast(text.len), .big);
        } else {
            try out.writeByte(0x80 | 127);
            try out.writeInt(u64, text.len, .big);
        }
        try out.writeAll(&mask);
        for (text, 0..) |c, i| try out.writeByte(c ^ mask[i % 4]);
        try out.flush();
    }

    /// The next text message (control frames skipped); error.Closed on a close frame.
    pub fn next(w: *Ws, a: std.mem.Allocator) ![]const u8 {
        w.deadline_ns.store(now(w.io) + 20 * std.time.ns_per_s, .release);
        defer w.deadline_ns.store(std.math.maxInt(i64), .release);
        const in = &w.reader.interface;
        while (true) {
            const h = try in.takeArray(2);
            const op = h[0] & 0x0f;
            const n: usize = switch (h[1] & 0x7f) {
                126 => try in.takeInt(u16, .big),
                127 => @intCast(try in.takeInt(u64, .big)),
                else => |x| x,
            };
            const data = try a.alloc(u8, n);
            try in.readSliceAll(data);
            if (op == 8) return error.Closed;
            if (op == 1) return data;
        }
    }

    /// Reads events until one of type `typ`; every event read is appended to `seen`.
    pub fn until(w: *Ws, a: std.mem.Allocator, typ: []const u8, seen: *std.ArrayList(std.json.Value)) !std.json.Value {
        while (true) {
            const v = try std.json.parseFromSliceLeaky(std.json.Value, a, try w.next(a), .{});
            try seen.append(a, v);
            if (std.mem.eql(u8, v.object.get("type").?.string, typ)) return v;
        }
    }

    /// `ms` of a sine at `hz` (0: silence), 24 kHz pcm16, as 100 ms appends.
    fn audio(w: *Ws, a: std.mem.Allocator, ms: u32, hz: f32) !void {
        var t: usize = 0;
        var left = ms;
        while (left > 0) : (left -|= 100) {
            var bytes: [4800]u8 = undefined;
            for (0..2400) |i| {
                const x: i16 = if (hz == 0) 0 else @intFromFloat(8000 * @sin(@as(f32, @floatFromInt(t)) * 2 * std.math.pi * hz / 24000));
                std.mem.writeInt(i16, bytes[i * 2 ..][0..2], x, .little);
                t += 1;
            }
            var b64: [6400]u8 = undefined;
            try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"input_audio_buffer.append\",\"audio\":\"{s}\"}}", .{std.base64.standard.Encoder.encode(&b64, &bytes)}));
        }
    }
};

pub fn now(io: Io) i64 {
    return @intCast(Io.Clock.awake.now(io).toNanoseconds());
}

fn typeOf(v: std.json.Value) []const u8 {
    return v.object.get("type").?.string;
}

fn countType(seen: []const std.json.Value, typ: []const u8) usize {
    var n: usize = 0;
    for (seen) |v| {
        if (std.mem.eql(u8, typeOf(v), typ)) n += 1;
    }
    return n;
}

fn config(a: std.mem.Allocator, chat_port: u16) ![]const u8 {
    return std.fmt.allocPrint(a, "{{" ++ harness.reserve ++ ", \"voice\": {{\"chat_url\": \"http://127.0.0.1:{d}/v1\", \"partial_interval_ms\": 0, \"prime_chat\": false, \"warm_speech\": false}}, \"tools\": [" ++
        "{{\"id\": \"ears\", \"kind\": \"transcription\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true, \"priority\": 50," ++
        " \"options\": {{\"phrases\": [{{\"hz\": 440, \"text\": \"hello there\"}}, {{\"hz\": 660, \"text\": \"tell me something slow\"}}, {{\"hz\": 880, \"text\": \"make me a picture of a fox\"}}]}}}}," ++
        "{{\"id\": \"mouth\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true, \"priority\": 50, \"options\": {{\"voices\": [\"plain\", \"bright\"], \"default_voice\": \"plain\", \"speak_ms\": 300}}}}," ++
        "{{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {{\"step_ms\": 5}}}}," ++
        "{{\"id\": \"tpv\", \"kind\": \"video\", \"engine\": \"testpattern\", \"options\": {{\"step_ms\": 400}}}}]}}", .{ chat_port, opts.speech_worker, opts.speech_worker });
}

/// Speaks one turn: a pause, `ms` of tone at `hz`, then the silence that ends it.
fn turn(w: *Ws, a: std.mem.Allocator, hz: f32, ms: u32) !void {
    try w.audio(a, 300, 0);
    try w.audio(a, ms, hz);
    try w.audio(a, 800, 0);
}

test "voice: a spoken turn over the WebSocket, barge-in, a tool call that makes an image, the page and the lanes" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var chat = try ChatServer.start(io);
    defer chat.stop(io);
    var d = try Daemon.start(testing.allocator, io, "voice", try config(a, chat.port));
    defer d.stop();

    // The page, a plain GET on the WebSocket route, and the lanes the speech tools run on.
    const page = try d.call(a, .GET, "/voice", null);
    try testing.expectEqual(std.http.Status.ok, page.status);
    try testing.expect(std.mem.indexOf(u8, page.body, "/v1/realtime") != null);
    const plain = try d.call(a, .GET, "/v1/realtime", null);
    try testing.expectEqual(@as(u10, 426), @backingInt(plain.status));
    const tools = try d.call(a, .GET, "/v1/tools", null);
    try testing.expect(std.mem.indexOf(u8, tools.body, "\"id\":\"mouth\",\"kind\":\"speech\"") != null);
    try testing.expect(std.mem.indexOf(u8, tools.body, "\"lane\":\"realtime\"") != null);

    // A speech job is not queued behind a long video job: they run on different lanes.
    const video = try d.call(a, .POST, "/v1/videos", "{\"model\": \"tpv\", \"prompt\": \"waves\", \"size\": \"256x256\", \"seconds\": \"1\"}");
    try testing.expectEqual(std.http.Status.ok, video.status);
    const vid = (try std.json.parseFromSliceLeaky(struct { id: []const u8 }, a, video.body, .{ .ignore_unknown_fields = true })).id;
    sleepMs(io, 300);
    const said = try d.call(a, .POST, "/v1/audio/speech", "{\"model\": \"mouth\", \"input\": \"beside the video\"}");
    try testing.expectEqual(std.http.Status.ok, said.status);
    const vstate = try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{vid}), null);
    try testing.expect(std.mem.indexOf(u8, vstate.body, "\"status\":\"in_progress\"") != null);
    _ = try d.call(a, .DELETE, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{vid}), null);

    var w: Ws = undefined;
    try w.open(io, a, d.port, "realtime, openai-beta.realtime-v1");
    defer w.close();
    try testing.expect(std.mem.indexOf(u8, w.head, "101 Switching Protocols") != null);
    try testing.expect(std.ascii.findIgnoreCase(w.head, "sec-websocket-protocol: realtime") != null);
    var seen: std.ArrayList(std.json.Value) = .empty;
    _ = try w.until(a, "session.created", &seen);

    // One spoken turn: VAD ends it, the fake STT hears "hello there", the reply streams back as text and speech.
    try turn(&w, a, 440, 900);
    const stopped_at = now(io);
    _ = try w.until(a, "input_audio_buffer.speech_stopped", &seen);
    const heard = try w.until(a, "conversation.item.input_audio_transcription.completed", &seen);
    try testing.expectEqualStrings("hello there", heard.object.get("transcript").?.string);
    _ = try w.until(a, "response.audio.delta", &seen);
    std.debug.print("voice: speech_stopped to first audio delta with zero-latency fakes: {d} ms\n", .{@divTrunc(now(io) - stopped_at, std.time.ns_per_ms)});
    const done = try w.until(a, "response.done", &seen);
    try testing.expectEqualStrings("completed", done.object.get("response").?.object.get("status").?.string);
    try testing.expect(countType(seen.items, "response.audio.delta") >= 2);
    const first = (try chat.stats(&d, a))[0];
    try testing.expect(first.completed);
    try testing.expectEqualStrings("hello there", first.messages[first.messages.len - 1].object.get("content").?.string);
    try testing.expectEqualStrings("generate_image", first.tools[0]); // an image model exists; generate_video, and edit_image (it edits) too
    try testing.expectEqual(@as(usize, 3), first.tools.len);

    // Barge-in: talking over a long reply stops the audio at once and cancels the chat stream (still writing: its speech
    // starts once the text is complete, voice.speak_written).
    try turn(&w, a, 660, 900);
    _ = try w.until(a, "response.audio_transcript.delta", &seen);
    try w.audio(a, 400, 440);
    _ = try w.until(a, "input_audio_buffer.speech_started", &seen);
    const mark = seen.items.len;
    const cut = try w.until(a, "response.done", &seen);
    try testing.expectEqualStrings("cancelled", cut.object.get("response").?.object.get("status").?.string);
    try testing.expectEqual(@as(usize, 0), countType(seen.items[mark..], "response.audio.delta"));
    var cancelled = false;
    for (0..60) |_| {
        const st = try chat.stats(&d, a);
        if (st.len >= 2 and st[1].cancelled) cancelled = true;
        if (cancelled) break;
        sleepMs(io, 50);
    }
    try testing.expect(cancelled);
    // The speech in progress was cancelled in the worker, not by killing it: it was loaded once.
    const mouth_log = try Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/data/logs/mouth.log", .{d.dir}), a, .limited(1 << 20));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, mouth_log, "speech worker loaded"));
    try w.audio(a, 800, 0); // that turn ends; its answer comes and is ignored here
    _ = try w.until(a, "response.done", &seen);

    // A tool call: "make me a picture" becomes a testpattern image job; its URL is an event and serves the PNG.
    try turn(&w, a, 880, 900);
    const job = try w.until(a, "localrouter.job.done", &seen);
    try testing.expectEqualStrings("completed", job.object.get("status").?.string);
    const url = job.object.get("urls").?.array.items[0].string;
    const path = url[std.mem.indexOf(u8, url, "/v1/files/").?..];
    const png = try d.call(a, .GET, path, null);
    try testing.expectEqual(std.http.Status.ok, png.status);
    try testing.expectEqualSlices(u8, "\x89PNG", png.body[0..4]);
    // The same job by its id (what a client that lost the event asks): completed, with the same file.
    const looked = try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/jobs/{s}", .{job.object.get("job_id").?.string}), null);
    try testing.expectEqual(std.http.Status.ok, looked.status);
    const jo = try std.json.parseFromSliceLeaky(std.json.Value, a, looked.body, .{});
    try testing.expectEqualStrings("completed", jo.object.get("status").?.string);
    try testing.expectEqualStrings(path, jo.object.get("urls").?.array.items[0].string);
    const missing = try d.call(a, .GET, "/v1/jobs/img_nope", null);
    try testing.expectEqual(std.http.Status.not_found, missing.status);
    _ = try w.until(a, "response.done", &seen);
    const st = try chat.stats(&d, a);
    const follow = st[st.len - 1].messages;
    try testing.expectEqualStrings("tool", follow[follow.len - 1].object.get("role").?.string);
    try testing.expect(std.mem.indexOf(u8, follow[follow.len - 1].object.get("content").?.string, "/v1/files/") != null);

    // A refusal keeps the session open.
    try w.send("{\"type\":\"session.update\",\"session\":{\"speech_model\":\"tp\"}}");
    const refused = try w.until(a, "error", &seen);
    try testing.expect(std.mem.indexOf(u8, refused.object.get("error").?.object.get("message").?.string, "tp") != null);
    try w.send("{\"type\":\"session.update\",\"session\":{\"voice\":\"bright\"}}");
    _ = try w.until(a, "session.updated", &seen);
}

test "voice: client tool execution over the WebSocket: a client's generate_image is called once, runs nowhere here, and its output is answered" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var chat = try ChatServer.start(io);
    defer chat.stop(io);
    var d = try Daemon.start(testing.allocator, io, "voice-client-tools", try config(a, chat.port));
    defer d.stop();
    var w: Ws = undefined;
    try w.open(io, a, d.port, "realtime");
    defer w.close();
    var seen: std.ArrayList(std.json.Value) = .empty;
    _ = try w.until(a, "session.created", &seen);
    try w.send("{\"type\":\"session.update\",\"session\":{\"localrouter\":{\"tool_execution\":\"client\"},\"tools\":[{\"type\":\"function\",\"name\":\"generate_image\",\"description\":\"draws on the device\",\"parameters\":{\"type\":\"object\"}}]}}");
    const updated = try w.until(a, "session.updated", &seen);
    try testing.expectEqualStrings("client", updated.object.get("session").?.object.get("localrouter").?.object.get("tool_execution").?.string);

    // "make me a picture of a fox": the fake model streams a generate_image call in pieces; the client gets it whole.
    try turn(&w, a, 880, 900);
    const call = try w.until(a, "response.function_call_arguments.done", &seen);
    try testing.expectEqualStrings("generate_image", call.object.get("name").?.string);
    try testing.expectEqualStrings("{\"prompt\": \"a red fox\"}", call.object.get("arguments").?.string);
    const call_id = call.object.get("call_id").?.string;
    _ = try w.until(a, "response.done", &seen);
    const first = (try chat.stats(&d, a))[0];
    try testing.expectEqual(@as(usize, 1), first.tools.len); // the client's tool alone: no built-in video or image tool
    sleepMs(io, 300);
    try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"conversation.item.create\",\"item\":{{\"type\":\"function_call_output\",\"call_id\":\"{s}\",\"output\":\"{{\\\"drawn\\\":true}}\"}}}}", .{call_id}));
    _ = try w.until(a, "conversation.item.created", &seen);
    try w.send("{\"type\":\"response.create\"}");
    _ = try w.until(a, "response.audio.delta", &seen);
    const done = try w.until(a, "response.done", &seen);
    try testing.expectEqualStrings("completed", done.object.get("response").?.object.get("status").?.string);
    try testing.expectEqual(@as(usize, 0), countType(seen.items, "localrouter.job.created")); // nothing ran on the box
    try testing.expectEqual(@as(usize, 0), countType(seen.items, "localrouter.job.done"));
    const st = try chat.stats(&d, a);
    const follow = st[st.len - 1].messages;
    try testing.expectEqualStrings("tool", follow[follow.len - 1].object.get("role").?.string);
    try testing.expectEqualStrings("{\"drawn\":true}", follow[follow.len - 1].object.get("content").?.string);
    try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"conversation.item.create\",\"item\":{{\"type\":\"function_call_output\",\"call_id\":\"{s}\",\"output\":\"again\"}}}}", .{call_id}));
    const refused = try w.until(a, "error", &seen);
    try testing.expectEqualStrings("duplicate_call_output", refused.object.get("error").?.object.get("code").?.string);
}
