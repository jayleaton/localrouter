//! Integration tests of the model management tools (MCP list_models, reserve_for, unload_model, load_model) on a real
//! daemon: the chat model (the fake server), two voice models (tests/fixtures/speech_worker.py), a kept image model and a
//! video model (testpattern), under a budget that makes a video need room. What reserve_for unloads, that it never
//! touches the chat model, that the unloaded models come back by themselves after the job, and that an open voice
//! session hears `localrouter.voice.paused` before a voice model goes and `localrouter.voice.resumed` when it is back.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");
const voice = @import("voice.zig");

const Daemon = harness.Daemon;
const mcpCall = harness.mcpCall;
const Value = std.json.Value;

/// chat 40 MiB (priority 100), mouth 8 (60), ears 8 (50), img 40 (10): 96 MiB warm. A testpattern video at its default
/// 1344x768 needs 32 MiB loaded and about 69.9 MiB working: 101.9 MiB.
fn config(a: std.mem.Allocator, budget_mib: u64) ![]const u8 {
    return std.fmt.allocPrint(a, "{{{s}, \"budget_bytes\": {d}, \"voice\": {{\"prime_chat\": false}}, \"tools\": [" ++
        "{{\"id\": \"chat\", \"kind\": \"chat\", \"engine\": \"openai_server\", \"resident_bytes\": 41943040, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"]}}}}," ++
        "{{\"id\": \"mouth\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true, \"priority\": 60, \"options\": {{\"voices\": [\"plain\"]}}}}," ++
        "{{\"id\": \"ears\", \"kind\": \"transcription\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608, \"keep_loaded\": true, \"priority\": 50}}," ++
        "{{\"id\": \"img\", \"kind\": \"image\", \"engine\": \"testpattern\", \"keep_loaded\": true, \"priority\": 10, \"options\": {{\"resident_mb\": 40}}}}," ++
        "{{\"id\": \"vid\", \"kind\": \"video\", \"engine\": \"testpattern\", \"idle_ttl_s\": 2, \"options\": {{\"step_ms\": 300}}}}]}}", .{ harness.reserve, budget_mib << 20, opts.chat_server, opts.speech_worker, opts.speech_worker });
}

/// The first text of an MCP result, parsed as JSON; its `isError`.
fn result(a: std.mem.Allocator, r: Value) !struct { ok: bool, v: Value, text: []const u8 } {
    const text = r.object.get("content").?.array.items[0].object.get("text").?.string;
    const v = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch Value.null;
    return .{ .ok = !r.object.get("isError").?.bool, .v = v, .text = text };
}

fn names(v: Value) []const Value {
    return v.object.get("unloads").?.array.items;
}

fn waitState(d: *Daemon, a: std.mem.Allocator, id: []const u8, want: []const u8, tries: usize) !void {
    for (0..tries) |_| {
        if (std.mem.eql(u8, try d.toolState(a, id), want)) return;
        harness.sleepMs(d.io, 250);
    }
    std.debug.print("{s}: wanted {s}, is {s}\n", .{ id, want, try d.toolState(a, id) });
    return error.TestExpectedEqual;
}

fn warm(d: *Daemon, a: std.mem.Allocator) !void {
    for ([_][]const u8{ "chat", "mouth", "ears", "img" }) |id| try waitState(d, a, id, "ready", 120);
}

/// Waits for the video job `id` (MCP get_job) to complete.
fn finishJob(d: *Daemon, a: std.mem.Allocator, out: Value) !void {
    const job = (try result(a, out)).v;
    const done = try result(a, try mcpCall(d, a, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\",\"wait_s\":60}}", .{job.object.get("id").?.string})));
    try testing.expectEqualStrings("completed", done.v.object.get("status").?.string);
}

test "model tools: reserve_for unloads the image model only, never chat, and it comes back after the video" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, testing.io, "models-room", try config(a, 190));
    defer d.stop();
    try warm(&d, a);

    // The dry run (the model named by its kind) names the image model and changes nothing.
    const dry = try result(a, try mcpCall(&d, a, "reserve_for", "{\"model\":\"video\",\"dry_run\":true}"));
    try testing.expect(dry.ok);
    try testing.expectEqual(@as(usize, 1), names(dry.v).len);
    try testing.expectEqualStrings("img", names(dry.v)[0].string);
    try testing.expect(!dry.v.object.get("pauses_voice").?.bool);
    try testing.expectEqualStrings("ready", try d.toolState(a, "img"));

    // The reservation unloads it and loads the video model; list_models says it is coming back.
    const res = try result(a, try mcpCall(&d, a, "reserve_for", "{\"model\":\"vid\"}"));
    if (!res.ok) std.debug.print("reserve_for: {s}\n", .{res.text});
    try testing.expect(res.ok);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "img"));
    try testing.expectEqualStrings("ready", try d.toolState(a, "vid"));
    try testing.expectEqualStrings("ready", try d.toolState(a, "chat"));
    const listed = (try result(a, try mcpCall(&d, a, "list_models", "{}"))).v.object.get("models").?.array.items;
    for (listed) |m| if (std.mem.eql(u8, m.object.get("id").?.string, "img")) try testing.expect(m.object.get("displaced").?.bool);

    // It stays down (held) until the video is made, then comes back by itself.
    harness.sleepMs(d.io, 3000);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "img"));
    try finishJob(&d, a, try mcpCall(&d, a, "generate_video", "{\"prompt\":\"waves\",\"seconds\":1,\"audio\":false,\"wait_s\":0}"));
    try waitState(&d, a, "img", "ready", 120);
    try waitState(&d, a, "vid", "unloaded", 40);
    try warm(&d, a);
}

test "model tools: a voice session hears voice pause before the voice models go, and resume after the video" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, testing.io, "models-voice", try config(a, 145));
    defer d.stop();
    try warm(&d, a);
    var w: voice.Ws = undefined;
    try w.open(testing.io, a, d.port, "realtime");
    defer w.close();
    var seen: std.ArrayList(Value) = .empty;
    _ = try w.until(a, "session.created", &seen);

    const dry = try result(a, try mcpCall(&d, a, "reserve_for", "{\"model\":\"vid\",\"dry_run\":true}"));
    try testing.expect(dry.v.object.get("pauses_voice").?.bool);
    const order = names(dry.v);
    try testing.expectEqual(@as(usize, 3), order.len); // lowest priority first; never the chat model
    for (order, [_][]const u8{ "img", "ears", "mouth" }) |n, want| try testing.expectEqualStrings(want, n.string);

    const res = try result(a, try mcpCall(&d, a, "reserve_for", "{\"model\":\"vid\"}"));
    if (!res.ok) std.debug.print("reserve_for: {s}\n", .{res.text});
    try testing.expect(res.ok);
    const paused = try w.until(a, "localrouter.voice.paused", &seen);
    try testing.expectEqualStrings("vid", paused.object.get("for_model").?.string);
    try testing.expectEqualStrings("ready", try d.toolState(a, "chat"));
    const models = (try result(a, try mcpCall(&d, a, "list_models", "{}"))).v;
    try testing.expect(models.object.get("memory").?.object.get("voice_paused").?.bool);

    try finishJob(&d, a, try mcpCall(&d, a, "generate_video", "{\"prompt\":\"waves\",\"seconds\":1,\"audio\":false,\"wait_s\":0}"));
    _ = try w.until(a, "localrouter.voice.resumed", &seen);
    try warm(&d, a);
}

test "model tools: the chat model is never unloaded; unload_model comes back after hold_s; load_model; refusals" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, testing.io, "models-unload", try config(a, 190));
    defer d.stop();
    try warm(&d, a);

    const self = try result(a, try mcpCall(&d, a, "unload_model", "{\"model\":\"chat\"}"));
    try testing.expect(!self.ok);
    try testing.expect(std.mem.indexOf(u8, self.text, "a chat model is not unloaded through the model tools") != null);
    try testing.expectEqualStrings("ready", try d.toolState(a, "chat"));

    const gone = try result(a, try mcpCall(&d, a, "unload_model", "{\"model\":\"img\",\"hold_s\":10}"));
    try testing.expect(gone.ok);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "img"));
    harness.sleepMs(d.io, 4000);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "img")); // held off for hold_s
    try waitState(&d, a, "img", "ready", 120); // then back by itself

    _ = try mcpCall(&d, a, "unload_model", "{\"model\":\"img\",\"hold_s\":600}");
    const back = try result(a, try mcpCall(&d, a, "load_model", "{\"model\":\"img\"}"));
    try testing.expect(back.ok);
    try testing.expectEqualStrings("ready", try d.toolState(a, "img"));

    const unknown = try result(a, try mcpCall(&d, a, "reserve_for", "{\"model\":\"nope\"}"));
    try testing.expect(!unknown.ok);
    try testing.expect(std.mem.indexOf(u8, unknown.text, "no model 'nope' on this machine") != null);
    const skill = try d.call(a, .GET, "/v1/skills/model-manager", null);
    try testing.expect(std.mem.indexOf(u8, skill.body, "\"enabled\":false") != null);
    try testing.expect(std.mem.indexOf(u8, skill.body, "reserve_for") != null);
}
