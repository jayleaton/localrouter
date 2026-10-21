//! Integration tests of one box running chat and voice together: the chat model (the `openai_server` engine on the fake
//! TensorFold server) and an external text to speech and speech to text worker (tests/fixtures/speech_worker.py), all
//! `keep_loaded`, under a host GPU lock. The scheduler admits them side by side by memory alone and holds the lock once
//! for all of them; a job that needs the machine evicts them lowest priority first, and they come back by themselves.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const sleepMs = harness.sleepMs;

/// Budget 100 MiB. chat 40 MiB (priority 100 by default), tts 8 MiB (50), stt 8 MiB (10): 56 MiB warm together.
/// img-a 50 MiB and img-b 56 MiB (priority 0) and a video tool of 30 MiB plus 64.4 MiB for a 256x256 clip fit only
/// after evictions.
fn config(a: std.mem.Allocator, lock: []const u8) ![]const u8 {
    const template = "{{" ++ harness.reserve ++ ", \"budget_bytes\": 104857600, \"gpu_lock\": \"{s}\", \"tools\": [" ++
        "{{\"id\": \"chat\", \"kind\": \"chat\", \"engine\": \"openai_server\", \"resident_bytes\": 41943040, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"]}}}}," ++
        "{{\"id\": \"tts\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true, \"priority\": 50, \"options\": {{\"voices\": [\"plain\", \"bright\"]}}}}," ++
        "{{\"id\": \"stt\", \"kind\": \"transcription\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true, \"priority\": 10}}," ++
        "{{\"id\": \"img-a\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 600, \"options\": {{\"resident_mb\": 50}}}}," ++
        "{{\"id\": \"img-b\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 600, \"options\": {{\"resident_mb\": 56}}}}," ++
        "{{\"id\": \"video\", \"kind\": \"video\", \"engine\": \"testpattern\", \"idle_ttl_s\": 1, \"options\": {{\"resident_mb\": 30}}}}]}}";
    return std.fmt.allocPrint(a, template, .{ lock, opts.chat_server, opts.speech_worker, opts.speech_worker });
}

fn expectState(d: *Daemon, a: std.mem.Allocator, id: []const u8, want: []const u8) !void {
    const got = try d.toolState(a, id);
    if (!std.mem.eql(u8, got, want)) std.debug.print("{s}: wanted {s}, is {s}\n", .{ id, want, got });
    try testing.expectEqualStrings(want, got);
}

fn waitState(d: *Daemon, a: std.mem.Allocator, id: []const u8, want: []const u8, tries: usize) !void {
    for (0..tries) |_| {
        if (std.mem.eql(u8, try d.toolState(a, id), want)) return;
        sleepMs(d.io, 250);
    }
    return expectState(d, a, id, want);
}

fn waitWarm(d: *Daemon, a: std.mem.Allocator) !void {
    for ([_][]const u8{ "chat", "tts", "stt" }) |id| try waitState(d, a, id, "ready", 80);
}

/// True when the test process could take `path` exclusively right now (and lets it go again).
fn lockFree(path: [:0]const u8) !bool {
    const fd = try std.posix.openatZ(std.posix.AT.FDCWD, path, .{}, 0);
    defer _ = std.posix.system.close(fd);
    return std.posix.errno(std.posix.system.flock(fd, std.posix.LOCK.EX | std.posix.LOCK.NB)) == .SUCCESS;
}

/// 200 ms of 16 kHz mono 16-bit PCM WAV, base64.
fn wavBase64(a: std.mem.Allocator) ![]const u8 {
    const frames = 3200;
    const wav = try a.alloc(u8, 44 + frames * 2);
    @memcpy(wav[0..4], "RIFF");
    std.mem.writeInt(u32, wav[4..8], @intCast(wav.len - 8), .little);
    @memcpy(wav[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, wav[16..20], 16, .little);
    std.mem.writeInt(u16, wav[20..22], 1, .little);
    std.mem.writeInt(u16, wav[22..24], 1, .little);
    std.mem.writeInt(u32, wav[24..28], 16000, .little);
    std.mem.writeInt(u32, wav[28..32], 32000, .little);
    std.mem.writeInt(u16, wav[32..34], 2, .little);
    std.mem.writeInt(u16, wav[34..36], 16, .little);
    @memcpy(wav[36..40], "data");
    std.mem.writeInt(u32, wav[40..44], frames * 2, .little);
    for (0..frames) |i| std.mem.writeInt(i16, wav[44 + i * 2 ..][0..2], @intCast(@as(i32, @intCast(i % 4000)) - 2000), .little);
    const enc = std.base64.standard.Encoder;
    return enc.encode(try a.alloc(u8, enc.calcSize(wav.len)), wav);
}

/// One request to each of the three warm models; each succeeds.
fn useAll(d: *Daemon, a: std.mem.Allocator) !void {
    const chat = try d.call(a, .POST, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    if (chat.status != .ok) std.debug.print("chat: {s}\n", .{chat.body});
    try testing.expectEqual(std.http.Status.ok, chat.status);
    try testing.expect(std.mem.indexOf(u8, chat.body, "echo: hi") != null);
    const say = try d.call(a, .POST, "/v1/audio/speech", "{\"model\":\"tts\",\"input\":\"hello\",\"voice\":\"plain\"}");
    if (say.status != .ok) std.debug.print("tts: {s}\n", .{say.body});
    try testing.expectEqual(std.http.Status.ok, say.status);
    try testing.expectEqualStrings("RIFF", say.body[0..4]);
    const heard = try harness.mcpCall(d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"model\":\"stt\",\"audio\":\"{s}\"}}", .{try wavBase64(a)}));
    try testing.expect(!heard.object.get("isError").?.bool);
}

test "one box: chat, text to speech and speech to text loaded together within the budget, under one GPU lock" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const lock = "/tmp/localrouter-it-jarvis-warm-gpu.lock";
    try Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = lock, .data = "" });
    defer Io.Dir.cwd().deleteFile(testing.io, lock) catch {};
    var d = try Daemon.start(testing.allocator, testing.io, "jarvis-warm", try config(a, lock));
    defer d.stop();

    try waitWarm(&d, a);
    try testing.expect(!try lockFree(lock)); // LocalRouter holds the host's lock for all three
    for (0..2) |_| try useAll(&d, a); // alternating models: nothing reloads
    for ([_][]const u8{ "chat", "tts", "stt" }) |id| try expectState(&d, a, id, "ready");

    // Released when nothing is loaded, taken again by the next load.
    try testing.expectEqual(std.http.Status.ok, (try d.call(a, .POST, "/v1/tools/release", "")).status);
    try testing.expect(try lockFree(lock));
    try useAll(&d, a);
    try testing.expect(!try lockFree(lock));
}

test "one box: a job that needs the machine evicts stt, then tts, then chat, and they come back by themselves" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const lock = "/tmp/localrouter-it-jarvis-evict-gpu.lock";
    try Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = lock, .data = "" });
    defer Io.Dir.cwd().deleteFile(testing.io, lock) catch {};
    var d = try Daemon.start(testing.allocator, testing.io, "jarvis-evict", try config(a, lock));
    defer d.stop();
    try waitWarm(&d, a);

    // img-a (50.4 MiB) fits once the lowest priority, stt, is out.
    try testing.expectEqual(std.http.Status.ok, (try harness.image(&d, a, "img-a", "")).status);
    try expectState(&d, a, "stt", "unloaded");
    try expectState(&d, a, "tts", "ready");
    try expectState(&d, a, "chat", "ready");

    // img-b (56.4 MiB): the idle image goes first (priority 0), then tts (50); chat (100) stays.
    try testing.expectEqual(std.http.Status.ok, (try harness.image(&d, a, "img-b", "")).status);
    try expectState(&d, a, "img-a", "unloaded");
    try expectState(&d, a, "tts", "unloaded");
    try expectState(&d, a, "chat", "ready");

    // A video clip (94.4 MiB) fits only alone: img-b, then chat.
    const V = struct { id: []const u8, status: []const u8 };
    const create = "{\"model\":\"video\",\"prompt\":\"bars\",\"size\":\"256x256\",\"seconds\":\"1\",\"steps\":1,\"audio\":false}";
    const job = try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .POST, "/v1/videos", create)).body, .{ .ignore_unknown_fields = true });
    var status: []const u8 = "";
    for (0..300) |_| {
        status = (try std.json.parseFromSliceLeaky(V, a, (try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{job.id}), null)).body, .{ .ignore_unknown_fields = true })).status;
        if (std.mem.eql(u8, status, "completed") or std.mem.eql(u8, status, "failed")) break;
        sleepMs(d.io, 100);
    }
    try testing.expectEqualStrings("completed", status);
    try expectState(&d, a, "img-b", "unloaded");
    try expectState(&d, a, "chat", "unloaded");

    // Once the scheduler is idle and they fit, chat, tts and stt are loaded again, and serve.
    try waitWarm(&d, a);
    try useAll(&d, a);
    try testing.expect(!try lockFree(lock));
}
