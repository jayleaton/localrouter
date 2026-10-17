//! Integration tests of the Drama 3 adapter (tools/speech/drama3/drama3_worker.py) on a real daemon, against a stand-in
//! for drama-tensorfold's launcher and server (tests/fixtures/drama3_pilot.py): the refused-until-qualified gate, routing
//! by model id, request validation, error mapping (400, 503, 500), whole and streamed replies with their finish
//! reasons, voices, references and long form, the shared GPU lock, a crashed server, and unloading on the idle TTL.

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

const model = "fishaudio/drama-3";
const root = "/tmp/localrouter-it-drama3-root";
const native = root ++ "/build/tensorfold/zig-out/native/bin/tensorfold-native";
const marker = root ++ "/qualification.json";
const lock = root ++ "/gpu.lock";

/// A raw HTTP/1.1 reply: its status, its head (lowercased), the body (de-chunked), the trailers (lowercased) and
/// whether a chunked body ended with its terminating chunk.
const Raw = struct { status: u16, head: []const u8, body: []const u8, trailers: []const u8 = "", complete: bool = true };

/// POST /v1/audio/speech over a plain socket, so the chunks, the trailers and an unterminated stream are all visible.
fn speak(d: *Daemon, a: std.mem.Allocator, json: []const u8) !Raw {
    const io = d.io;
    const addr: Io.net.IpAddress = .{ .ip4 = .loopback(d.port) };
    const s = try addr.connect(io, .{ .mode = .stream });
    defer s.close(io);
    var wbuf: [4096]u8 = undefined;
    var w = s.writer(io, &wbuf);
    try w.interface.print("POST /v1/audio/speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ json.len, json });
    try w.interface.flush();
    try s.shutdown(io, .send); // the daemon reads the end of the connection after this request and closes it once it has answered
    var rbuf: [16 * 1024]u8 = undefined;
    var r = s.reader(io, &rbuf);
    return parse(a, try r.interface.allocRemaining(a, .unlimited));
}

fn parse(a: std.mem.Allocator, all: []const u8) !Raw {
    const end = std.mem.indexOf(u8, all, "\r\n\r\n") orelse return error.NoHead;
    const head = try std.ascii.allocLowerString(a, all[0..end]);
    const status = try std.fmt.parseInt(u16, head[9..12], 10);
    var rest = all[end + 4 ..];
    if (std.mem.indexOf(u8, head, "transfer-encoding: chunked") == null) return .{ .status = status, .head = head, .body = rest };
    var body: std.ArrayList(u8) = .empty;
    while (true) {
        const nl = std.mem.indexOf(u8, rest, "\r\n") orelse break;
        const size = try std.fmt.parseInt(usize, rest[0..nl], 16);
        rest = rest[nl + 2 ..];
        if (size == 0) {
            if (std.mem.startsWith(u8, rest, "\r\n")) return .{ .status = status, .head = head, .body = body.items };
            const t = std.mem.indexOf(u8, rest, "\r\n\r\n") orelse break;
            return .{ .status = status, .head = head, .body = body.items, .trailers = try std.ascii.allocLowerString(a, rest[0 .. t + 2]) };
        }
        if (rest.len < size + 2) {
            try body.appendSlice(a, rest);
            break;
        }
        try body.appendSlice(a, rest[0..size]);
        rest = rest[size + 2 ..];
    }
    return .{ .status = status, .head = head, .body = body.items, .complete = false };
}

fn expectHas(haystack: []const u8, needles: []const []const u8) !void {
    for (needles) |n| if (std.mem.indexOf(u8, haystack, n) == null) {
        std.debug.print("missing '{s}' in:\n{s}\n", .{ n, haystack });
        return error.TestExpectedEqual;
    };
}

fn expectStatus(r: Raw, status: u16, needles: []const []const u8) !void {
    if (r.status != status) std.debug.print("expected {d}, got {d}: {s}\n{s}\n", .{ status, r.status, r.head, r.body });
    try testing.expectEqual(status, r.status);
    try expectHas(r.body, needles);
}

/// Every sample of a 16-bit mono PCM body is `level`, and there are `n` of them.
fn expectLevel(pcm: []const u8, n: usize, level: i16) !void {
    try testing.expectEqual(n * 2, pcm.len);
    var i: usize = 0;
    while (i < pcm.len) : (i += 2) try testing.expectEqual(level, std.mem.readInt(i16, pcm[i..][0..2], .little));
}

/// Whether the shared GPU lock is free now: a non-blocking exclusive flock on it succeeds (and is released again).
fn lockFree(io: Io) !bool {
    const f = try Io.Dir.cwd().openFile(io, lock, .{});
    defer f.close(io);
    const rc = std.os.linux.flock(f.handle, std.posix.LOCK.EX | std.posix.LOCK.NB);
    if (std.os.linux.errno(rc) != .SUCCESS) return false;
    _ = std.os.linux.flock(f.handle, std.posix.LOCK.UN);
    return true;
}

fn sha256Hex(io: Io, a: std.mem.Allocator, path: []const u8) ![64]u8 {
    const data = try Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

/// A free loopback port for the stand-in server.
fn freePort(io: Io) !u16 {
    const addr: Io.net.IpAddress = .{ .ip4 = .loopback(0) };
    var s = try addr.listen(io, .{});
    defer s.deinit(io);
    return s.socket.address.getPort();
}

test "Drama 3: refused until qualified, whole and streamed speech, voices, references, long form, errors and unload" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = Io.Dir.cwd();
    cwd.deleteTree(io, root) catch {};
    defer cwd.deleteTree(io, root) catch {};
    try cwd.createDirPath(io, root ++ "/build/tensorfold/zig-out/native/bin");
    try cwd.createDirPath(io, root ++ "/drama-3");
    try cwd.createDirPath(io, root ++ "/voices");
    try cwd.writeFile(io, .{ .sub_path = native, .data = "#!/bin/sh\nexit 1\n", .flags = .{ .permissions = .executable_file } });
    try cwd.writeFile(io, .{ .sub_path = root ++ "/drama-3/codec.pth", .data = "" });
    try cwd.writeFile(io, .{ .sub_path = root ++ "/voices/narrator.json", .data = "{\"format\": \"drama3-voice/1\"}" });
    try cwd.writeFile(io, .{ .sub_path = lock, .data = "" });
    const port = try freePort(io);
    const cfg = try std.fmt.allocPrint(a, "{{" ++ reserve ++ ", \"gpu_lock\": \"{s}\", \"tools\": [" ++
        "{{\"id\": \"fishaudio/drama-3\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"weights\": \"{s}\", \"qualification\": \"{s}\"," ++
        " \"resident_bytes\": 8388608, \"idle_ttl_s\": 3, \"load_timeout_s\": 60, \"generate_timeout_s\": 60, \"options\": {{" ++
        "\"voices\": [\"default\", \"narrator\"], \"default_voice\": \"default\", \"min_speed\": 1, \"max_speed\": 1, \"stream\": true," ++
        " \"references\": true, \"port\": {d}, \"long_form\": true, \"long_form_chars\": 40, \"voices_dir\": \"{s}/voices\", \"verify\": \"none\", \"pilot\": \"{s}\"}}}}," ++
        "{{\"id\": \"tone\", \"kind\": \"speech\", \"engine\": \"testpattern\"}}]}}", .{ lock, opts.drama3_worker, root, marker, port, root, opts.drama3_pilot });
    var d = try Daemon.start(testing.allocator, io, "drama3", cfg);
    defer d.stop();
    const say = "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hello there.\"}"; // 12 characters: 5292 samples

    // Listed, and refused before anything starts until this machine is qualified.
    const models = try d.call(a, .GET, "/v1/models", null);
    try expectHas(models.body, &.{ "\"id\":\"fishaudio/drama-3\"", "\"stream\":true,\"references\":true", "\"qualified\":false" });
    try expectStatus(try speak(&d, a, say), 503, &.{ "model 'fishaudio/drama-3' is not GPU-qualified on this host", marker ++ " does not exist" });
    const load = try d.call(a, .POST, "/v1/tools/fishaudio/drama-3/load", null);
    try testing.expectEqual(std.http.Status.service_unavailable, load.status);
    try expectHas(load.body, &.{"is not GPU-qualified on this host"});
    try testing.expectEqualStrings("unloaded", try d.toolState(a, model));
    try testing.expect(try lockFree(io));
    // A model without a gate is not affected.
    try testing.expectEqual(std.http.Status.ok, (try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"tone\",\"input\":\"hi\"}")).status);

    // A marker for other pins passes the daemon's gate but not the worker's: the load fails with the reason.
    var host_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
    const host = try std.posix.gethostname(&host_buf);
    const bin_sha = try sha256Hex(io, a, native);
    try cwd.writeFile(io, .{ .sub_path = marker, .data = try std.fmt.allocPrint(a, "{{\"qualified\": true, \"host\": \"{s}\", \"pins_sha256\": \"{s}\", \"native_sha256\": \"{s}\"}}", .{ host, "0000000000000000000000000000000000000000000000000000000000000000", &bin_sha }) });
    try expectStatus(try speak(&d, a, say), 503, &.{ "load failed", "qualified other pins" });
    try testing.expect(try lockFree(io));
    const pins_sha = try sha256Hex(io, a, opts.drama3_pins);
    try cwd.writeFile(io, .{ .sub_path = marker, .data = try std.fmt.allocPrint(a, "{{\"qualified\": true, \"host\": \"{s}\", \"pins_sha256\": \"{s}\", \"native_sha256\": \"{s}\"}}", .{ host, &pins_sha, &bin_sha }) });
    try expectHas((try d.call(a, .GET, "/v1/models", null)).body, &.{"\"qualified\":true"});

    // Whole replies: WAV at 44.1 kHz with x-finish-reason, and the same samples as raw PCM.
    const whole = try speak(&d, a, say);
    try expectStatus(whole, 200, &.{"RIFF"});
    try expectHas(whole.head, &.{ "content-type: audio/wav", "x-finish-reason: stop" });
    try testing.expectEqual(@as(u32, 44100), std.mem.readInt(u32, whole.body[24..28], .little));
    try expectLevel(whole.body[44..], 5292, 1000);
    try testing.expect(!try lockFree(io)); // the launcher holds the shared GPU lock while the model is loaded
    const pcm = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hello there.\",\"response_format\":\"pcm\"}");
    try expectHas(pcm.head, &.{ "x-finish-reason: stop", "x-sample-rate: 44100", "x-sample-format: s16le" });
    try testing.expectEqualSlices(u8, whole.body[44..], pcm.body);
    try expectHas((try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Cut short [silence]\"}")).head, &.{"x-finish-reason: silence"});

    // Streamed: chunked raw PCM, the same samples; a clipped stream ends with the trailer, a finished one without.
    const streamed = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hello there.\",\"response_format\":\"pcm\",\"stream\":true}");
    try testing.expectEqual(@as(u16, 200), streamed.status);
    try expectHas(streamed.head, &.{ "transfer-encoding: chunked", "content-type: audio/pcm", "x-sample-rate: 44100", "trailer: x-finish-reason, x-long-form-segments" });
    try testing.expect(streamed.complete);
    try testing.expectEqualStrings("", streamed.trailers);
    try testing.expectEqualSlices(u8, pcm.body, streamed.body);
    const clipped = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Too long [length]\",\"response_format\":\"pcm\",\"stream\":true}");
    try testing.expect(clipped.complete);
    try expectLevel(clipped.body, 441 * 17, 1000);
    try expectHas(clipped.trailers, &.{"x-finish-reason: length"});

    // Long form: the segments in a whole reply's head, and both trailers at the end of a stream.
    const long_text = "One sentence that is long. And another one after it.";
    const long_whole = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"" ++ long_text ++ "\"}");
    try expectHas(long_whole.head, &.{ "x-finish-reason: stop", "x-long-form-segments: 2;11466" });
    const long_stream = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"" ++ long_text ++ "\",\"response_format\":\"pcm\",\"stream\":true}");
    try testing.expect(long_stream.complete);
    try expectHas(long_stream.trailers, &.{ "x-finish-reason: stop", "x-long-form-segments: 2;11466" });

    // Voices and references reach the server; refusals come back as 400s, from the daemon or from the server.
    const voiced = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hi.\",\"voice\":\"narrator\",\"response_format\":\"pcm\"}");
    try expectLevel(voiced.body, 441 * 3, 2000);
    const codes = "[[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]]";
    const referenced = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hi.\",\"response_format\":\"pcm\",\"references\":[{\"text\":\"a clip\",\"vq_codes\":" ++ codes ++ "}]}");
    try expectLevel(referenced.body, 441 * 3, 1300);
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hi.\",\"references\":[{\"text\":\"a clip\",\"vq_codes\":[[1,2],[3]]}]}"), 400, &.{"Drama 3 request: references[0].vq_codes must be 10 rows of equal length"});
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hi.\",\"references\":[{\"text\":\"a clip\",\"vq_codes\":\"AAAA\"}]}"), 400, &.{"references must be a list of objects"});
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hi.\",\"voice\":\"nobody\"}"), 400, &.{"model 'fishaudio/drama-3' has no voice 'nobody'; its voices: default, narrator"});
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hi.\",\"stream\":true}"), 400, &.{"stream true needs response_format pcm"});
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hi.\",\"speed\":1.5}"), 400, &.{"model 'fishaudio/drama-3' takes only speed 1"});
    try expectStatus(try speak(&d, a, "{\"model\":\"tone\",\"input\":\"Hi.\",\"response_format\":\"pcm\",\"stream\":true}"), 400, &.{"model 'tone' does not stream its audio"});
    try expectStatus(try speak(&d, a, "{\"model\":\"tone\",\"input\":\"Hi.\",\"references\":[{\"text\":\"a\",\"vq_codes\":[[1]]}]}"), 400, &.{"model 'tone' takes no voice references"});

    // The server's own errors: 503 while it is busy, 500 for its faults, and a stream that breaks off has no end.
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Wait [busy]\"}"), 503, &.{"the engine is busy"});
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Oops [fault]\"}"), 500, &.{"speech synthesis failed"});
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Gone [break]\"}"), 500, &.{"without complete audio"});
    const broken = try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"A long stream that breaks off [break]\",\"response_format\":\"pcm\",\"stream\":true}");
    try testing.expectEqual(@as(u16, 200), broken.status);
    try testing.expect(!broken.complete);
    try testing.expectEqualStrings("ready", try d.toolState(a, model));

    // MCP: the same model by id, with the finish reason in the job summary.
    const res = try mcpCall(&d, a, "synthesize_speech", "{\"model\":\"fishaudio/drama-3\",\"input\":\"Hello there.\"}");
    try testing.expect(!res.object.get("isError").?.bool);
    try testing.expectEqualStrings("audio/wav", res.object.get("content").?.array.items[0].object.get("mimeType").?.string);
    try testing.expectEqualStrings("stop", (try mcpText(a, res)).object.get("finish_reason").?.string);

    // A server that dies takes the worker with it: that job is a 503, and the next request starts it again.
    try expectStatus(try speak(&d, a, "{\"model\":\"fishaudio/drama-3\",\"input\":\"Down [crash]\"}"), 503, &.{"worker exited"});
    try testing.expectEqualStrings("unloaded", try d.toolState(a, model));
    try expectStatus(try speak(&d, a, say), 200, &.{"RIFF"});

    // Idle past its TTL, the model unloads: the launcher stops its server and releases the GPU lock.
    var state: []const u8 = "";
    for (0..100) |_| {
        state = try d.toolState(a, model);
        if (std.mem.eql(u8, state, "unloaded")) break;
        sleepMs(io, 100);
    }
    try testing.expectEqualStrings("unloaded", state);
    try testing.expect(try lockFree(io));
}
