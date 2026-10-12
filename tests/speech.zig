//! Integration tests of the speech kinds on real daemons and workers: text to speech and speech to text over HTTP and
//! MCP, upload decoding and limits, response formats, refusals, duration-priced memory, cancellation and worker
//! failure, through the compiled-in testpattern engine and an external `cmd` worker (tests/fixtures/speech_worker.py).

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const reserve = harness.reserve;
const mcpCall = harness.mcpCall;
const mcpText = harness.mcpText;
const sleepMs = harness.sleepMs;

const boundary = "speechboundary";
const form_type = "multipart/form-data; boundary=" ++ boundary;
const Field = struct { name: []const u8, value: []const u8 };

/// A multipart body: text `fields`, then the audio as the part `file`.
fn form(a: std.mem.Allocator, fields: []const Field, file: ?[]const u8) ![]u8 {
    var w: Io.Writer.Allocating = .init(a);
    for (fields) |f| try w.writer.print("--{s}\r\nContent-Disposition: form-data; name=\"{s}\"\r\n\r\n{s}\r\n", .{ boundary, f.name, f.value });
    if (file) |data| {
        try w.writer.print("--{s}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n", .{boundary});
        try w.writer.writeAll(data);
        try w.writer.writeAll("\r\n");
    }
    try w.writer.print("--{s}--\r\n", .{boundary});
    return w.written();
}

fn transcribe(d: *Daemon, a: std.mem.Allocator, fields: []const Field, file: ?[]const u8) !Daemon.Resp {
    return d.callAs(a, .POST, "/v1/audio/transcriptions", try form(a, fields, file), form_type);
}

/// `ms` of 16-bit PCM WAV at `rate` with `channels`, a ramp the length of the file.
fn wav(a: std.mem.Allocator, rate: u32, channels: u16, ms: u32) ![]u8 {
    const frames = rate / 1000 * ms;
    const out = try a.alloc(u8, 44 + frames * channels * 2);
    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(out.len - 8), .little);
    @memcpy(out[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    std.mem.writeInt(u16, out[20..22], 1, .little);
    std.mem.writeInt(u16, out[22..24], channels, .little);
    std.mem.writeInt(u32, out[24..28], rate, .little);
    std.mem.writeInt(u32, out[28..32], rate * channels * 2, .little);
    std.mem.writeInt(u16, out[32..34], channels * 2, .little);
    std.mem.writeInt(u16, out[34..36], 16, .little);
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], frames * channels * 2, .little);
    for (0..frames * channels) |i| std.mem.writeInt(i16, out[44 + i * 2 ..][0..2], @intCast(@as(i32, @intCast(i % 4000)) - 2000), .little);
    return out;
}

/// The response headers of a JSON POST, lowercased names, one "name: value" a line.
fn headers(d: *Daemon, a: std.mem.Allocator, path: []const u8, body: []const u8) ![]const u8 {
    const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}{s}", .{ d.port, path });
    var req = try d.client.request(.POST, try std.Uri.parse(url), .{ .keep_alive = false, .headers = .{ .content_type = .{ .override = "application/json" } } });
    defer req.deinit();
    req.transfer_encoding = .{ .content_length = body.len };
    try req.sendBodyComplete(@constCast(body));
    var redirect: [0]u8 = undefined;
    const res = try req.receiveHead(&redirect);
    var out: Io.Writer.Allocating = .init(a);
    var it = res.head.iterateHeaders();
    while (it.next()) |hd| try out.writer.print("{s}: {s}\n", .{ try std.ascii.allocLowerString(a, hd.name), hd.value });
    return out.written();
}

fn b64(a: std.mem.Allocator, data: []const u8) ![]const u8 {
    const enc = std.base64.standard.Encoder;
    return enc.encode(try a.alloc(u8, enc.calcSize(data.len)), data);
}

fn decode(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const dec = std.base64.standard.Decoder;
    const out = try a.alloc(u8, try dec.calcSizeForSlice(text));
    try dec.decode(out, text);
    return out;
}

fn expectStatus(r: Daemon.Resp, status: std.http.Status, needles: []const []const u8) !void {
    if (r.status != status) std.debug.print("expected {d}, got {d}: {s}\n", .{ @backingInt(status), @backingInt(r.status), r.body });
    try testing.expectEqual(status, r.status);
    for (needles) |n| if (std.mem.indexOf(u8, r.body, n) == null) {
        std.debug.print("missing '{s}' in {s}\n", .{ n, r.body });
        return error.TestExpectedEqual;
    };
}

/// Samples in a WAV's data chunk (16-bit mono), and its rate.
fn wavInfo(body: []const u8) !struct { rate: u32, samples: usize } {
    try testing.expectEqualStrings("RIFF", body[0..4]);
    try testing.expectEqualStrings("WAVE", body[8..12]);
    return .{ .rate = std.mem.readInt(u32, body[24..28], .little), .samples = std.mem.readInt(u32, body[40..44], .little) / 2 };
}

const speech_tools =
    \\{"id": "tone", "kind": "speech", "engine": "testpattern", "options": {"voices": ["alpha", "beta"]}},
    \\{"id": "fixed", "kind": "speech", "engine": "testpattern", "options": {"voices": ["Bruno", "Rosie"], "default_voice": "Rosie",
    \\  "min_speed": 1, "max_speed": 1, "presets": ["stable"], "max_input_chars": 20}},
    \\{"id": "ears", "kind": "transcription", "engine": "testpattern"},
    \\{"id": "short-ears", "kind": "transcription", "engine": "testpattern", "options": {"max_audio_s": 1}}
;

test "text to speech: HTTP and MCP, formats, voices, speed and refusals" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "tts", "{" ++ reserve ++ ", \"tools\": [" ++ speech_tools ++ "]}");
    defer d.stop();

    // 5 characters at 60 ms each, 24 kHz: 7200 samples; speed 2 halves it.
    const r = try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"tone\",\"input\":\"hello\",\"voice\":\"alpha\",\"seed\":1}");
    try expectStatus(r, .ok, &.{});
    const info = try wavInfo(r.body);
    try testing.expectEqual(@as(u32, 24_000), info.rate);
    try testing.expectEqual(@as(usize, 7200), info.samples);
    const again = try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hello\",\"voice\":\"alpha\",\"seed\":1}"); // the default speech model
    try testing.expectEqualSlices(u8, r.body, again.body);
    const fast = try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hello\",\"speed\":2}");
    try testing.expectEqual(@as(usize, 3600), (try wavInfo(fast.body)).samples);
    const pcm = try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hello\",\"voice\":\"alpha\",\"seed\":1,\"response_format\":\"pcm\"}");
    try testing.expectEqualSlices(u8, r.body[44..], pcm.body);
    const flac = try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hello\",\"response_format\":\"flac\"}");
    try testing.expectEqualStrings("fLaC", flac.body[0..4]);
    const opus = try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hello\",\"response_format\":\"opus\"}");
    try testing.expectEqualStrings("OggS", opus.body[0..4]);

    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"voice\":\"gamma\"}"), .bad_request, &.{"model 'tone' has no voice 'gamma'; its voices: alpha, beta"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"instructions\":\"whisper it\"}"), .bad_request, &.{"does not take instructions"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"stream_format\":\"sse\"}"), .bad_request, &.{"stream_format sse is not supported"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"  \"}"), .bad_request, &.{"input is required"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"speed\":9}"), .bad_request, &.{"speed must be 0.25 to 4"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"response_format\":\"ogg\"}"), .bad_request, &.{"response_format must be wav, pcm, flac, mp3, opus or aac"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"model\":\"ears\"}"), .bad_request, &.{ "model 'ears' is a transcription model, not speech", "models that can do text_to_speech: tone" });
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"model\":\"nope\"}"), .not_found, &.{"unknown model 'nope'"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "not json"), .bad_request, &.{"not a valid speech request"});

    // A model's own limits refuse before it loads: nothing is resampled or ignored quietly.
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"fixed\",\"input\":\"hi\",\"speed\":1.5}"), .bad_request, &.{"model 'fixed' takes only speed 1"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"fixed\",\"input\":\"hi\",\"preset\":\"expressive\"}"), .bad_request, &.{"model 'fixed' has no preset 'expressive'; its presets: stable"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"fixed\",\"input\":\"twenty-one characters\"}"), .bad_request, &.{"model 'fixed' takes at most 20 characters of input"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"tone\",\"input\":\"hi\",\"preset\":\"stable\"}"), .bad_request, &.{"model 'tone' has no preset 'stable'; its presets: none"});
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "fixed"));
    // An empty voice is the tool's default_voice by the time the engine sees it.
    const default_voice = try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"fixed\",\"input\":\"hi\",\"seed\":4,\"preset\":\"stable\"}");
    try expectStatus(default_voice, .ok, &.{});
    try testing.expectEqualSlices(u8, default_voice.body, (try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"fixed\",\"input\":\"hi\",\"seed\":4,\"voice\":\"Rosie\"}")).body);
    try testing.expect(!std.mem.eql(u8, default_voice.body, (try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"fixed\",\"input\":\"hi\",\"seed\":4,\"voice\":\"Bruno\"}")).body));
    // Raw PCM says what it is in headers: rate, channels, and 16-bit little-endian samples.
    const head = try headers(&d, a, "/v1/audio/speech", "{\"model\":\"tone\",\"input\":\"hi\",\"response_format\":\"pcm\"}");
    for ([_][]const u8{ "content-type: audio/pcm", "x-sample-rate: 24000", "x-channels: 1", "x-sample-format: s16le" }) |n| if (std.mem.indexOf(u8, head, n) == null) {
        std.debug.print("missing '{s}' in {s}\n", .{ n, head });
        return error.TestExpectedEqual;
    };
    try testing.expect(std.mem.indexOf(u8, (try headers(&d, a, "/v1/audio/speech", "{\"model\":\"tone\",\"input\":\"hi\"}")), "content-type: audio/wav") != null);

    // /v1/models lists the voices and what each model serves by default.
    const models = try d.call(a, .GET, "/v1/models", null);
    try expectStatus(models, .ok, &.{ "\"voices\":[\"Bruno\",\"Rosie\"],\"default_voice\":\"Rosie\",\"presets\":[\"stable\"],\"speed\":[1,1],\"max_input_chars\":20", "\"max_audio_s\":1800", "\"voices\":[\"alpha\",\"beta\"]", "\"kind\":\"speech\"", "\"default_for\":[\"text_to_speech\"]", "\"default_for\":[\"speech_to_text\"]" });

    // MCP: the audio inline (WAV) and a URL to the same file.
    const res = try mcpCall(&d, a, "synthesize_speech", "{\"input\":\"hello\",\"voice\":\"alpha\",\"seed\":1}");
    try testing.expect(!res.object.get("isError").?.bool);
    const content = res.object.get("content").?.array.items;
    try testing.expectEqualStrings("audio", content[0].object.get("type").?.string);
    try testing.expectEqualStrings("audio/wav", content[0].object.get("mimeType").?.string);
    try testing.expectEqualSlices(u8, r.body, try decode(a, content[0].object.get("data").?.string));
    const link = content[content.len - 1].object;
    try testing.expectEqualStrings("audio/wav", link.get("mimeType").?.string);
    const url = link.get("uri").?.string;
    const file = try d.call(a, .GET, url[std.mem.indexOf(u8, url, "/v1/files/").?..], null);
    try testing.expectEqualSlices(u8, r.body, file.body);
    const bad = try mcpCall(&d, a, "synthesize_speech", "{\"input\":\"hello\",\"voice\":\"gamma\"}");
    try testing.expect(bad.object.get("isError").?.bool);
}

test "speech to text: uploads decode to 16 kHz mono, response formats, limits and refusals" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "stt", "{" ++ reserve ++ ", \"tools\": [" ++ speech_tools ++ "]}");
    defer d.stop();
    const T = struct { text: []const u8, usage: struct { type: []const u8, seconds: u64 } };

    // An exact 16 kHz mono WAV reaches the engine sample for sample: the hash it reports is of these samples.
    const exact = try wav(a, 16_000, 1, 1500);
    const r = try transcribe(&d, a, &.{.{ .name = "model", .value = "ears" }}, exact);
    try expectStatus(r, .ok, &.{});
    const t = try std.json.parseFromSliceLeaky(T, a, r.body, .{});
    try testing.expectEqualStrings(try std.fmt.allocPrint(a, "testpattern heard 1500 ms of audio, hash {x}", .{std.hash.Wyhash.hash(0, exact[44..])}), t.text);
    try testing.expectEqualStrings("duration", t.usage.type);
    try testing.expectEqual(@as(u64, 2), t.usage.seconds);

    // 48 kHz stereo goes through ffmpeg: 16 kHz mono of the same length.
    try expectStatus(try transcribe(&d, a, &.{}, try wav(a, 48_000, 2, 1500)), .ok, &.{"heard 1500 ms of audio"});
    // JSON with the file as base64 (an extension; OpenAI's API is multipart only).
    const json_body = try std.fmt.allocPrint(a, "{{\"file\":\"{s}\",\"language\":\"de\"}}", .{try b64(a, exact)});
    try expectStatus(try d.call(a, .POST, "/v1/audio/transcriptions", json_body), .ok, &.{", language de"});

    // Formats.
    const text = try transcribe(&d, a, &.{.{ .name = "response_format", .value = "text" }}, exact);
    try testing.expectEqualStrings(try std.fmt.allocPrint(a, "{s}\n", .{t.text}), text.body);
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "response_format", .value = "verbose_json" }}, exact), .ok, &.{ "\"task\":\"transcribe\"", "\"language\":\"en\"", "\"duration\":1.5", "\"segments\":[{\"id\":0,\"start\":0,\"end\":1" });
    const words = try transcribe(&d, a, &.{ .{ .name = "response_format", .value = "verbose_json" }, .{ .name = "timestamp_granularities[]", .value = "word" }, .{ .name = "timestamp_granularities[]", .value = "segment" } }, exact);
    try expectStatus(words, .ok, &.{ "\"words\":[{\"word\":\"testpattern\",\"start\":0,\"end\":0.5}", "\"segments\"" });
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "response_format", .value = "srt" }}, exact), .ok, &.{"1\n00:00:00,000 --> 00:00:01,000\ntestpattern heard"});
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "response_format", .value = "vtt" }}, exact), .ok, &.{"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\ntestpattern heard"});

    // Refusals: what the API does not offer, bad uploads, and model limits.
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "stream", .value = "true" }}, exact), .bad_request, &.{"stream is not supported"});
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "response_format", .value = "diarized_json" }}, exact), .bad_request, &.{"response_format must be json, text, verbose_json, srt or vtt"});
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "timestamp_granularities[]", .value = "word" }}, exact), .bad_request, &.{"timestamp_granularities needs response_format verbose_json"});
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "language", .value = "english" }}, exact), .bad_request, &.{"language must be an ISO-639-1 code"});
    try expectStatus(try transcribe(&d, a, &.{}, null), .bad_request, &.{"file is required"});
    try expectStatus(try transcribe(&d, a, &.{}, ""), .bad_request, &.{"the audio file is empty"});
    try expectStatus(try transcribe(&d, a, &.{}, "\x89PNG\r\n\x1a\nnot audio"), .bad_request, &.{"the audio could not be decoded"});
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "model", .value = "short-ears" }}, exact), .bad_request, &.{"the audio is longer than this model takes"});
    try expectStatus(try transcribe(&d, a, &.{.{ .name = "model", .value = "tone" }}, exact), .bad_request, &.{"model 'tone' is a speech model, not transcription"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/translations", "{}"), .not_found, &.{"translation is not offered"});

    // MCP: the transcript's text first, then its JSON.
    const res = try mcpCall(&d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"audio\":\"data:audio/wav;base64,{s}\",\"timestamps\":\"segment\"}}", .{try b64(a, exact)}));
    try testing.expect(!res.object.get("isError").?.bool);
    const content = res.object.get("content").?.array.items;
    try testing.expectEqualStrings(t.text, content[0].object.get("text").?.string);
    const tj = try std.json.parseFromSliceLeaky(std.json.Value, a, content[1].object.get("text").?.string, .{});
    try testing.expectEqual(@as(usize, 2), tj.object.get("segments").?.array.items.len);
    const refused = try mcpCall(&d, a, "transcribe_audio", "{\"audio\":\"bm90IGF1ZGlv\"}");
    try testing.expect(refused.object.get("isError").?.bool);
}

test "transcription jobs: memory priced by duration, cancel kills the worker, a crashed worker fails one job" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 32 MiB resident; 2 s of audio needs 64 KB + 1 MiB on top, 60 s about 2.9 MiB: only the first fits in 34 MiB.
    var d = try Daemon.start(testing.allocator, io, "stt-jobs", "{" ++ reserve ++ ", \"budget_bytes\": 35651584, \"tools\": [" ++
        "{\"id\": \"slow-ears\", \"kind\": \"transcription\", \"engine\": \"testpattern\", \"idle_ttl_s\": 60, \"options\": {\"step_ms\": 400}}]}");
    defer d.stop();

    try expectStatus(try transcribe(&d, a, &.{}, try wav(a, 16_000, 1, 2000)), .ok, &.{"heard 2000 ms"});
    try expectStatus(try transcribe(&d, a, &.{}, try wav(a, 16_000, 1, 60_000)), .service_unavailable, &.{ "insufficient_memory", "needs 2 MiB more" });

    // Cancel: an MCP transcription returns its job id when it outlasts wait_s; cancel_job kills the worker.
    const audio = try std.fmt.allocPrint(a, "{{\"audio\":\"{s}\",\"wait_s\":0}}", .{try b64(a, try wav(a, 16_000, 1, 10_000))});
    const started = try mcpText(a, try mcpCall(&d, a, "transcribe_audio", audio));
    const id = started.object.get("id").?.string;
    try testing.expect(std.mem.startsWith(u8, id, "transcription_"));
    sleepMs(io, 600);
    try testing.expectEqual(@as(usize, 1), (try d.workers(a)).len);
    const cancelled = try mcpText(a, try mcpCall(&d, a, "cancel_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\"}}", .{id})));
    try testing.expectEqualStrings("cancelled", cancelled.object.get("status").?.string);
    try testing.expectEqual(@as(usize, 0), (try d.workers(a)).len);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "slow-ears"));

    // A worker killed mid-job fails that job; the next one reloads it, and get_job returns the transcript.
    const again = try mcpText(a, try mcpCall(&d, a, "transcribe_audio", audio));
    var pids: []i32 = &.{};
    for (0..50) |_| {
        pids = try d.workers(a);
        if (pids.len > 0) break;
        sleepMs(io, 50);
    }
    try testing.expectEqual(@as(usize, 1), pids.len);
    sleepMs(io, 300);
    _ = std.posix.system.kill(pids[0], std.posix.SIG.KILL);
    const crashed = try mcpCall(&d, a, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\",\"wait_s\":10}}", .{again.object.get("id").?.string}));
    try testing.expect(crashed.object.get("isError").?.bool);
    const why = (try mcpText(a, crashed)).object.get("error").?.string;
    try testing.expect(std.mem.indexOf(u8, why, "worker exited") != null);
    const short = try std.fmt.allocPrint(a, "{{\"audio\":\"{s}\",\"wait_s\":0}}", .{try b64(a, try wav(a, 16_000, 1, 500))});
    const next = try mcpText(a, try mcpCall(&d, a, "transcribe_audio", short));
    const done = try mcpCall(&d, a, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\",\"wait_s\":30}}", .{next.object.get("id").?.string}));
    try testing.expect(!done.object.get("isError").?.bool);
    try testing.expect(std.mem.startsWith(u8, done.object.get("content").?.array.items[0].object.get("text").?.string, "testpattern heard 500 ms"));
}

test "an external speech worker: the cmd protocol for both kinds, its refusals, cancellation and a crash" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", \"tools\": [" ++
        "{{\"id\": \"py-tts\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"options\": {{\"voices\": [\"plain\", \"bright\", \"missing\"]}}}}," ++
        "{{\"id\": \"py-stt\", \"kind\": \"transcription\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608}}]}}", .{ opts.speech_worker, opts.speech_worker });
    var d = try Daemon.start(testing.allocator, io, "stt-cmd", cfg);
    defer d.stop();

    const s = try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"twenty chars of text.\",\"voice\":\"bright\"}");
    try expectStatus(s, .ok, &.{});
    const info = try wavInfo(s.body);
    try testing.expectEqual(@as(u32, 22050), info.rate);
    try testing.expectEqual(@as(usize, 23152), info.samples); // 22050 x 0.05 x 21
    // A voice the config lists but the worker lacks: the worker's own refusal is a 400 too.
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hi\",\"voice\":\"missing\"}"), .bad_request, &.{"no voice 'missing'"});

    try expectStatus(try transcribe(&d, a, &.{ .{ .name = "response_format", .value = "verbose_json" }, .{ .name = "timestamp_granularities[]", .value = "word" } }, try wav(a, 48_000, 2, 1000)), .ok, &.{ "external worker heard 16000 samples", "\"avg_logprob\":-0.1", "\"words\":[{\"word\":\"external\"" });

    // A worker that dies mid-job fails that job with its log tail; the next request starts it again.
    const crashed = try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"crash\"}");
    try expectStatus(crashed, .service_unavailable, &.{"worker exited"});
    try expectStatus(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"fine again\"}"), .ok, &.{});
    try testing.expectEqualStrings("ready", try d.toolState(a, "py-tts"));
}
