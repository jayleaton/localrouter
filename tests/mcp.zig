//! Integration tests of the MCP tool surface on a real daemon (testpattern engines): the tool list and its strict
//! schemas, list_models' state and memory, list_voices, generate_image's wait_s and job id, edit_image (and its refusal
//! without an edit model), transcribe_audio by path (only inside the config's local_files), and errors that tell the
//! caller what to do.

const std = @import("std");
const Io = std.Io;
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const mcpCall = harness.mcpCall;
const mcpText = harness.mcpText;
const Value = std.json.Value;

const tools =
    \\{"id": "img", "kind": "image", "engine": "testpattern", "options": {"step_ms": 20}},
    \\{"id": "tone", "kind": "speech", "engine": "testpattern", "options": {"voices": ["Mae", "Arlo"], "default_voice": "Mae"}},
    \\{"id": "ears", "kind": "transcription", "engine": "testpattern"}
;

/// The first text of an MCP result, as a string.
fn firstText(result: Value) []const u8 {
    for (result.object.get("content").?.array.items) |c| if (std.mem.eql(u8, c.object.get("type").?.string, "text")) return c.object.get("text").?.string;
    return "";
}

fn expectError(result: Value, needle: []const u8) !void {
    try testing.expect(result.object.get("isError").?.bool);
    const t = firstText(result);
    if (std.mem.indexOf(u8, t, needle) == null) {
        std.debug.print("missing '{s}' in '{s}'\n", .{ needle, t });
        return error.TestExpectedEqual;
    }
}

/// 300 ms of 16 kHz mono 16-bit PCM WAV.
fn wav(a: std.mem.Allocator) ![]u8 {
    const frames = 4800;
    const out = try a.alloc(u8, 44 + frames * 2);
    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(out.len - 8), .little);
    @memcpy(out[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, out[16..20], 16, .little);
    std.mem.writeInt(u16, out[20..22], 1, .little);
    std.mem.writeInt(u16, out[22..24], 1, .little);
    std.mem.writeInt(u32, out[24..28], 16000, .little);
    std.mem.writeInt(u32, out[28..32], 32000, .little);
    std.mem.writeInt(u16, out[32..34], 2, .little);
    std.mem.writeInt(u16, out[34..36], 16, .little);
    @memcpy(out[36..40], "data");
    std.mem.writeInt(u32, out[40..44], frames * 2, .little);
    for (0..frames) |i| std.mem.writeInt(i16, out[44 + i * 2 ..][0..2], @intCast(@as(i32, @intCast(i % 4000)) - 2000), .little);
    return out;
}

test "mcp tools: a strict schema each, models with state and memory, voices, images by job id, actionable errors" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "mcp-tools", "{" ++ harness.reserve ++ ", \"tools\": [" ++ tools ++ "]}");
    defer d.stop();

    // Every tool: a name, a description, an object schema that refuses unknown arguments.
    const listed = try std.json.parseFromSliceLeaky(Value, a, (try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")).body, .{});
    const list = listed.object.get("result").?.object.get("tools").?.array.items;
    try testing.expectEqual(@as(usize, 20), list.len);
    var names: std.ArrayList(u8) = .empty;
    for (list) |t| {
        const schema = t.object.get("inputSchema").?.object;
        try testing.expectEqualStrings("object", schema.get("type").?.string);
        try testing.expect(!schema.get("additionalProperties").?.bool);
        try testing.expect(t.object.get("description").?.string.len > 40);
        try names.appendSlice(a, t.object.get("name").?.string);
        try names.append(a, ' ');
    }
    for ([_][]const u8{ "list_models", "list_voices", "reserve_for", "unload_model", "load_model", "generate_image", "edit_image", "generate_video", "transcribe_audio", "synthesize_speech", "chat", "get_job", "cancel_job", "release_models", "computer_control", "computer_use", "computer_status", "computer_stop" }) |n|
        try testing.expect(std.mem.indexOf(u8, names.items, n) != null);

    // edit_image: the picture to edit, a mask and an omitted size (the input's shape) are in its schema.
    for (list) |t| if (std.mem.eql(u8, t.object.get("name").?.string, "edit_image")) {
        const schema = t.object.get("inputSchema").?.object;
        try testing.expectEqualStrings("prompt", schema.get("required").?.array.items[0].string);
        const props = schema.get("properties").?.object;
        for ([_][]const u8{ "image", "images", "mask", "size", "n", "seed", "model", "inline_images", "wait_s" }) |k| try testing.expect(props.get(k) != null);
        try testing.expect(props.get("size").?.object.get("default") == null);
    };

    // list_models: each model's state, resident bytes and lane, and the machine's memory.
    const models = (try mcpText(a, try mcpCall(&d, a, "list_models", "{}"))).object;
    try testing.expect(models.get("memory").?.object.get("mem_available_bytes").?.integer > 0);
    const first = models.get("models").?.array.items[0].object;
    try testing.expectEqualStrings("unloaded", first.get("state").?.string);
    try testing.expectEqualStrings("main", first.get("lane").?.string);
    try testing.expect(first.get("resident_bytes") != null);

    // list_voices: the speech model, its voices and default.
    const voices = (try mcpText(a, try mcpCall(&d, a, "list_voices", "{}"))).object.get("speech_models").?.array.items;
    try testing.expectEqual(@as(usize, 1), voices.len);
    try testing.expectEqualStrings("tone", voices[0].object.get("model").?.string);
    try testing.expectEqualStrings("Mae", voices[0].object.get("default_voice").?.string);
    try testing.expectEqual(@as(usize, 2), voices[0].object.get("voices").?.array.items.len);
    try testing.expect(voices[0].object.get("default").?.bool);

    // generate_image with wait_s 0: the job id, then get_job waits for it and links the PNG.
    const queued = try mcpText(a, try mcpCall(&d, a, "generate_image", "{\"prompt\":\"bars\",\"size\":\"256x256\",\"steps\":20,\"wait_s\":0}"));
    const id = queued.object.get("id").?.string;
    try testing.expect(queued.object.get("next") != null);
    const done = try mcpText(a, try mcpCall(&d, a, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\",\"wait_s\":60}}", .{id})));
    try testing.expectEqualStrings("completed", done.object.get("status").?.string);
    try testing.expectEqual(@as(usize, 1), done.object.get("files").?.array.items.len);

    // edit_image of that PNG, by its own /v1/files URL (testpattern draws edits at the size asked).
    const png_url = done.object.get("files").?.array.items[0].string;
    const edited = try mcpCall(&d, a, "edit_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"make it night\",\"image\":\"{s}\",\"size\":\"256x256\",\"inline_images\":false}}", .{png_url}));
    try testing.expect(!edited.object.get("isError").?.bool);
    try testing.expectEqualStrings("completed", (try std.json.parseFromSliceLeaky(Value, a, firstText(edited), .{})).object.get("status").?.string);

    // Errors say what to do.
    try expectError(try mcpCall(&d, a, "edit_image", "{\"prompt\":\"make it night\"}"), "edit_image needs the picture to edit");
    try expectError(try mcpCall(&d, a, "edit_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"x\",\"image\":\"{s}\",\"images\":[\"{s}\"]}}", .{ png_url, png_url })), "send image or images, not both");
    try expectError(try mcpCall(&d, a, "generate_image", "{\"prompt\":\"bars\",\"mask\":\"aGk=\"}"), "call edit_image");
    try expectError(try mcpCall(&d, a, "generate_image", "{\"size\":\"256x256\"}"), "invalid arguments for generate_image: a required argument is missing");
    try expectError(try mcpCall(&d, a, "generate_image", "{\"prompt\":\"bars\",\"n\":\"two\"}"), "an argument has the wrong type");
    try expectError(try mcpCall(&d, a, "synthesize_speech", "{\"input\":\"hi\",\"voice\":\"Zed\"}"), "model 'tone' has no voice 'Zed'; its voices: Mae, Arlo");
    try expectError(try mcpCall(&d, a, "synthesize_speech", "{\"input\":\"hi\",\"instructions\":\"calm\"}"), "model 'tone' does not take instructions; send the text to speak as input");
    try expectError(try mcpCall(&d, a, "transcribe_audio", "{}"), "audio (base64, a data URL or an http(s) URL) or path (a file on this machine) is required");
    try expectError(try mcpCall(&d, a, "transcribe_audio", "{\"path\":\"/etc/hostname\"}"), "path is off on this server");
    const unknown = try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"make_music\",\"arguments\":{}}}");
    try testing.expect(std.mem.indexOf(u8, unknown.body, "unknown tool 'make_music'; tools/list names them: list_models, list_voices") != null);
}

test "mcp transcribe_audio by path: only files inside local_files are read" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rnd: [8]u8 = undefined;
    io.random(&rnd);
    const root = try std.fmt.allocPrint(a, "/tmp/localrouter-it-mcp-files-{x:0>16}", .{std.mem.readInt(u64, &rnd, .little)});
    try Io.Dir.cwd().createDirPath(io, root);
    defer Io.Dir.cwd().deleteTree(io, root) catch {};
    const clip = try std.fmt.allocPrint(a, "{s}/clip.wav", .{root});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = clip, .data = try wav(a) });
    const outside = try std.fmt.allocPrint(a, "{s}-outside.wav", .{root});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = outside, .data = try wav(a) });
    defer Io.Dir.cwd().deleteFile(io, outside) catch {};

    const cfg = try std.fmt.allocPrint(a, "{{{s}, \"local_files\": [\"{s}\"], \"tools\": [{s}]}}", .{ harness.reserve, root, tools });
    var d = try Daemon.start(testing.allocator, io, "mcp-files", cfg);
    defer d.stop();

    const heard = try mcpCall(&d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"path\":\"{s}\"}}", .{clip}));
    try testing.expect(!heard.object.get("isError").?.bool);
    try testing.expect(std.mem.startsWith(u8, firstText(heard), "testpattern heard 300 ms of audio"));
    try expectError(try mcpCall(&d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"path\":\"{s}\"}}", .{outside})), "is outside the directories this server may read");
    try expectError(try mcpCall(&d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"path\":\"{s}/../{s}\"}}", .{ root, std.fs.path.basename(outside) })), "is outside the directories this server may read");
    try expectError(try mcpCall(&d, a, "transcribe_audio", "{\"path\":\"clip.wav\"}"), "path must be absolute");
    try expectError(try mcpCall(&d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"path\":\"{s}/missing.wav\"}}", .{root})), "no such file");
    try expectError(try mcpCall(&d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"path\":\"{s}\",\"audio\":\"aGk=\"}}", .{clip})), "send audio or path, not both");
}

test "computer control without a paired device: the tools, the helper's socket and the approval routes all refuse" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "computer-off", "{\"tools\": [" ++ tools ++ "]}");
    defer d.stop();
    try expectError(try mcpCall(&d, a, "computer_control", "{\"control\":{\"kind\":\"open_app\",\"app\":\"Safari\"}}"), "needs device pairing");
    try expectError(try mcpCall(&d, a, "computer_use", "{\"goal\":\"search for groceries\"}"), "needs device pairing");
    try expectError(try mcpCall(&d, a, "computer_stop", "{}"), "needs device pairing");
    for ([_][]const u8{ "/v1/computer/status", "/v1/computer/approvals", "/v1/computer/audit", "/v1/desktop" }) |p| {
        const r = try d.call(a, .GET, p, null);
        try testing.expectEqual(std.http.Status.unauthorized, r.status);
        try testing.expect(std.mem.indexOf(u8, r.body, "needs device pairing") != null);
    }
    const approve = try d.call(a, .POST, "/v1/computer/approvals/appr_x", "{\"decision\":\"approve\"}");
    try testing.expectEqual(std.http.Status.unauthorized, approve.status);
}

test "mcp edit_image without an edit-capable model: refused, naming the models and what they can do" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain =
        \\{"id": "plain", "kind": "image", "engine": "testpattern", "capabilities": ["text_to_image"]}
    ;
    var d = try Daemon.start(testing.allocator, io, "mcp-no-edit", "{" ++ harness.reserve ++ ", \"tools\": [" ++ plain ++ "]}");
    defer d.stop();
    try expectError(try mcpCall(&d, a, "edit_image", "{\"prompt\":\"make it night\",\"image\":\"aGk=\"}"), "no model on this machine can do image_edit: its models are plain (text_to_image)");
}
