//! Integration tests that a model is a drop-in replacement: every kind (image, video, speech, transcription, chat,
//! decision, asset3d) is served by set A, then by set B, with other ids and, where a fake exists, another adapter
//! (testpattern speech, transcription and 3D in A, the `cmd` worker tests/fixtures/speech_worker.py in B; chat and decision are
//! `openai_server` on their fixtures in both). Only the config changes: the MCP tool list and /v1/capabilities' MCP
//! tools stay the same, /v1/models lists what is configured, every endpoint and MCP tool works with and without a model
//! named, a capability a model lacks is refused naming it and what it can do, `defaults` alone moves a request, the
//! voice loop runs on either set and speaks with the configured default, and declared languages are listed and enforced.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");
const voice = @import("voice.zig");

const Daemon = harness.Daemon;
const mcpCall = harness.mcpCall;
const Value = std.json.Value;

// Not a real picture: the daemon only checks the magic bytes, and the testpattern engine hashes the contents.
const png = "\x89PNG\r\n\x1a\nimage A";

/// One model per kind, and what tells its replies apart: the transcript's first words and the speech sample rate.
const Set = struct {
    image: []const u8,
    video: []const u8,
    speech: []const u8,
    transcription: []const u8,
    chat: []const u8,
    decision: []const u8,
    asset3d: []const u8,
    gaussians: []const u8, // the 3D model's declared range, as its refusal words it
    inputs: bool, // the image model edits and the video model takes a first frame
    heard: []const u8,
    rate: u32,
    tools: []const []const u8,

    fn ids(s: Set) [7][]const u8 {
        return .{ s.image, s.video, s.speech, s.transcription, s.chat, s.decision, s.asset3d };
    }

    /// `defaults` naming this set's model for each kind's main capability.
    fn defaults(s: Set, a: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(a, "{{\"text_to_image\": \"{s}\", \"text_to_video\": \"{s}\", \"text_to_speech\": \"{s}\", \"speech_to_text\": \"{s}\", \"chat_completion\": \"{s}\", \"typed_decision\": \"{s}\", \"image_to_3d\": \"{s}\"}}", .{ s.image, s.video, s.speech, s.transcription, s.chat, s.decision, s.asset3d });
    }
};

/// A chat or decision tool (40 MiB) on its fixture server.
fn server(a: std.mem.Allocator, id: []const u8, kind: []const u8, script: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"id\": \"{s}\", \"kind\": \"{s}\", \"engine\": \"openai_server\", \"resident_bytes\": 41943040, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"]}}}}", .{ id, kind, script });
}

/// A speech or transcription tool (8 MiB) on the external worker; `extra` config keys led by a comma.
fn worker(a: std.mem.Allocator, id: []const u8, kind: []const u8, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"id\": \"{s}\", \"kind\": \"{s}\", \"cmd\": [\"python3\", \"{s}\"], \"resident_bytes\": 8388608{s}}}", .{ id, kind, opts.speech_worker, extra });
}

/// Set A: compiled-in testpattern media with every input capability.
fn setA(a: std.mem.Allocator) !Set {
    return .{ .image = "img-a", .video = "vid-a", .speech = "tts-a", .transcription = "stt-a", .chat = "chat-a", .decision = "dec-a", .asset3d = "a3d-a", .gaussians = "num_gaussians 1024 to 65536 in multiples of 32", .inputs = true, .heard = "testpattern heard", .rate = 24_000, .tools = try a.dupe([]const u8, &.{
        "{\"id\": \"img-a\", \"kind\": \"image\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 5}}",
        "{\"id\": \"vid-a\", \"kind\": \"video\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 5}}",
        "{\"id\": \"tts-a\", \"kind\": \"speech\", \"engine\": \"testpattern\", \"options\": {\"voices\": [\"plain\"]}}",
        "{\"id\": \"stt-a\", \"kind\": \"transcription\", \"engine\": \"testpattern\"}",
        try server(a, "chat-a", "chat", opts.chat_server),
        try server(a, "dec-a", "decision", opts.decide_server),
        "{\"id\": \"a3d-a\", \"kind\": \"asset3d\", \"engine\": \"testpattern\", \"options\": {\"step_ms\": 5, \"min_gaussians\": 1024, \"max_gaussians\": 65536, \"gaussian_multiple\": 32, \"default_gaussians\": 2048}}",
    }) };
}

/// Set B: other ids; image and video narrowed to text input only; speech and transcription by the `cmd` worker.
fn setB(a: std.mem.Allocator) !Set {
    return .{ .image = "img-b", .video = "vid-b", .speech = "tts-b", .transcription = "stt-b", .chat = "chat-b", .decision = "dec-b", .asset3d = "a3d-b", .gaussians = "num_gaussians 32 to 4096 in multiples of 32", .inputs = false, .heard = "external worker heard", .rate = 22_050, .tools = try a.dupe([]const u8, &.{
        "{\"id\": \"img-b\", \"kind\": \"image\", \"engine\": \"testpattern\", \"capabilities\": [\"text_to_image\"], \"options\": {\"step_ms\": 5}}",
        "{\"id\": \"vid-b\", \"kind\": \"video\", \"engine\": \"testpattern\", \"capabilities\": [\"text_to_video\"], \"options\": {\"step_ms\": 5}}",
        try worker(a, "tts-b", "speech", ", \"options\": {\"voices\": [\"plain\", \"bright\"]}"),
        try worker(a, "stt-b", "transcription", ""),
        try server(a, "chat-b", "chat", opts.chat_server),
        try server(a, "dec-b", "decision", opts.decide_server),
        try worker(a, "a3d-b", "asset3d", ", \"working_bytes\": 1048576, \"options\": {\"input_modes\": [\"synthetic\", \"real\"], \"min_gaussians\": 32, \"max_gaussians\": 4096, \"gaussian_multiple\": 32, \"default_gaussians\": 1024, \"working_bytes_by_gaussians\": {\"1024\": 1048576, \"4096\": 4194304}}"),
    }) };
}

/// A daemon config: `tools` joined, `defaults`, and `top` config keys (led by a comma).
fn config(a: std.mem.Allocator, tools: []const []const u8, defaults: []const u8, top: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{" ++ harness.reserve ++ ", \"defaults\": {s}{s}, \"tools\": [{s}]}}", .{ defaults, top, try std.mem.join(a, ", ", tools) });
}

fn start(a: std.mem.Allocator, name: []const u8, s: Set, top: []const u8) !Daemon {
    return Daemon.start(testing.allocator, testing.io, name, try config(a, s.tools, try s.defaults(a), top));
}

fn expectStatus(r: Daemon.Resp, status: std.http.Status, needles: []const []const u8) !void {
    if (r.status != status) std.debug.print("expected {d}, got {d}: {s}\n", .{ @backingInt(status), @backingInt(r.status), r.body });
    try testing.expectEqual(status, r.status);
    for (needles) |n| if (std.mem.indexOf(u8, r.body, n) == null) {
        std.debug.print("missing '{s}' in {s}\n", .{ n, r.body });
        return error.TestExpectedEqual;
    };
}

/// A 200 JSON reply's `model`.
fn modelOf(a: std.mem.Allocator, r: Daemon.Resp) ![]const u8 {
    try expectStatus(r, .ok, &.{});
    return (try std.json.parseFromSliceLeaky(struct { model: []const u8 }, a, r.body, .{ .ignore_unknown_fields = true })).model;
}

fn texts(result: Value) []const Value {
    return result.object.get("content").?.array.items;
}

/// A successful MCP result's model: the first text that is a JSON object with a `model` (the job, completion or reply).
fn mcpModel(a: std.mem.Allocator, result: Value) ![]const u8 {
    if (result.object.get("isError").?.bool) std.debug.print("tool error: {s}\n", .{texts(result)[0].object.get("text").?.string});
    try testing.expect(!result.object.get("isError").?.bool);
    for (texts(result)) |c| {
        const t = c.object.get("text") orelse continue;
        const v = std.json.parseFromSliceLeaky(Value, a, t.string, .{}) catch continue;
        if (v == .object) if (v.object.get("model")) |m| if (m == .string) return m.string;
    }
    return error.NoModel;
}

fn expectToolError(result: Value, needles: []const []const u8) !void {
    try testing.expect(result.object.get("isError").?.bool);
    const t = texts(result)[0].object.get("text").?.string;
    for (needles) |n| if (std.mem.indexOf(u8, t, n) == null) {
        std.debug.print("missing '{s}' in '{s}'\n", .{ n, t });
        return error.TestExpectedEqual;
    };
}

fn b64(a: std.mem.Allocator, data: []const u8) ![]const u8 {
    const enc = std.base64.standard.Encoder;
    return enc.encode(try a.alloc(u8, enc.calcSize(data.len)), data);
}

/// 1 s of 16 kHz mono 16-bit PCM WAV (no ffmpeg needed), a ramp.
fn wav(a: std.mem.Allocator) ![]u8 {
    const frames = 16_000;
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

/// A WAV reply's sample rate.
fn rateOf(r: Daemon.Resp) !u32 {
    try expectStatus(r, .ok, &.{});
    try testing.expectEqualStrings("RIFF", r.body[0..4]);
    return std.mem.readInt(u32, r.body[24..28], .little);
}

/// A 3D job (POST /v1/assets3d, waited for): its model, after checking it completed with an asset, a manifest and an
/// input view that the files route serves, the asset labelled as machine-generated.
fn assetModel(d: *Daemon, a: std.mem.Allocator, r: Daemon.Resp) ![]const u8 {
    try expectStatus(r, .ok, &.{ "\"object\":\"asset3d\"", "\"status\":\"completed\"", "\"representation\":\"gaussian_splat\"" });
    const J = struct { id: []const u8, model: []const u8, output: struct { asset_url: []const u8, manifest_url: []const u8, input_url: []const u8 } };
    const j = try std.json.parseFromSliceLeaky(J, a, r.body, .{ .ignore_unknown_fields = true });
    const path = struct {
        fn of(url: []const u8) []const u8 {
            return url[std.mem.indexOf(u8, url, "/v1/files/").?..];
        }
    }.of;
    const ply = try d.call(a, .GET, path(j.output.asset_url), null);
    try expectStatus(ply, .ok, &.{"comment AI-edited (machine-generated) from an input image"});
    try testing.expect(std.mem.startsWith(u8, ply.body, "ply\nformat binary_little_endian 1.0\n"));
    const content = try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/assets3d/{s}/content", .{j.id}), null);
    try testing.expectEqualSlices(u8, ply.body, content.body);
    const manifest = try d.call(a, .GET, path(j.output.manifest_url), null);
    try expectStatus(manifest, .ok, &.{});
    try testing.expectEqualStrings("asset3d", (try std.json.parseFromSliceLeaky(struct { object: []const u8 }, a, manifest.body, .{ .ignore_unknown_fields = true })).object);
    try testing.expectEqualStrings("\x89PNG", (try d.call(a, .GET, path(j.output.input_url), null)).body[0..4]);
    return j.model;
}

/// The model of a video job (POST /v1/videos) once it completes.
fn videoModel(d: *Daemon, a: std.mem.Allocator, create: Daemon.Resp) ![]const u8 {
    try expectStatus(create, .ok, &.{});
    const J = struct { id: []const u8, model: []const u8, status: []const u8 };
    const id = (try std.json.parseFromSliceLeaky(J, a, create.body, .{ .ignore_unknown_fields = true })).id;
    for (0..300) |_| {
        const st = try std.json.parseFromSliceLeaky(J, a, (try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{id}), null)).body, .{ .ignore_unknown_fields = true });
        if (std.mem.eql(u8, st.status, "completed")) return st.model;
        try testing.expect(!std.mem.eql(u8, st.status, "failed"));
        harness.sleepMs(d.io, 100);
    }
    return error.Timeout;
}

const route = "\"questions\":{\"route\":{\"type\":\"choice\",\"instructions\":\"Which action?\",\"criteria\":{\"image\":\"make a picture\",\"video\":\"make a clip\",\"reply\":null}}}";

/// `,"model":"<id>"` when `on`, else nothing.
fn modelArg(a: std.mem.Allocator, on: bool, id: []const u8) ![]const u8 {
    return if (on) std.fmt.allocPrint(a, ",\"model\":\"{s}\"", .{id}) else "";
}

/// Every endpoint and MCP tool once, the model named (`named`) or left to `defaults`; each reply is from the set's model.
fn everyEndpoint(d: *Daemon, a: std.mem.Allocator, s: Set, named: bool) !void {
    const clip = try b64(a, try wav(a));

    // The OpenAI endpoints and /v1/systemone.
    try testing.expectEqualStrings(s.image, try modelOf(a, try d.call(a, .POST, "/v1/images/generations", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":1{s}}}", .{try modelArg(a, named, s.image)}))));
    try testing.expectEqualStrings(s.video, try videoModel(d, a, try d.call(a, .POST, "/v1/videos", try std.fmt.allocPrint(a, "{{\"prompt\":\"waves\",\"size\":\"256x256\",\"seconds\":\"1\",\"audio\":false{s}}}", .{try modelArg(a, named, s.video)}))));
    try testing.expectEqual(s.rate, try rateOf(try d.call(a, .POST, "/v1/audio/speech", try std.fmt.allocPrint(a, "{{\"input\":\"hello\"{s}}}", .{try modelArg(a, named, s.speech)}))));
    try expectStatus(try d.call(a, .POST, "/v1/audio/transcriptions", try std.fmt.allocPrint(a, "{{\"file\":\"{s}\"{s}}}", .{ clip, try modelArg(a, named, s.transcription) })), .ok, &.{s.heard});
    const said = try d.call(a, .POST, "/v1/chat/completions", try std.fmt.allocPrint(a, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]{s}}}", .{try modelArg(a, named, s.chat)}));
    try testing.expectEqualStrings(s.chat, try modelOf(a, said));
    try testing.expect(std.mem.indexOf(u8, said.body, "echo: hi") != null);
    try testing.expectEqualStrings(s.asset3d, try assetModel(d, a, try d.call(a, .POST, "/v1/assets3d", try std.fmt.allocPrint(a, "{{\"image\":\"{s}\",\"input_mode\":\"real\",\"seed\":3,\"wait_s\":60{s}}}", .{ try b64(a, png), try modelArg(a, named, s.asset3d) }))));
    const decided = try d.call(a, .POST, "/v1/systemone", try std.fmt.allocPrint(a, "{{\"state\":{{\"user_said\":\"draw an image of a fox, yes\"}},{s}{s}}}", .{ route, try modelArg(a, named, s.decision) }));
    try testing.expectEqualStrings(s.decision, try modelOf(a, decided));

    // The MCP tools.
    try testing.expectEqualStrings(s.image, try mcpModel(a, try mcpCall(d, a, "generate_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":1{s}}}", .{try modelArg(a, named, s.image)}))));
    const queued = try mcpCall(d, a, "generate_video", try std.fmt.allocPrint(a, "{{\"prompt\":\"waves\",\"size\":\"256x256\",\"seconds\":1,\"audio\":false,\"wait_s\":0{s}}}", .{try modelArg(a, named, s.video)}));
    const id = (try harness.mcpText(a, queued)).object.get("id").?.string;
    const done = try mcpCall(d, a, "get_job", try std.fmt.allocPrint(a, "{{\"id\":\"{s}\",\"wait_s\":60}}", .{id}));
    try testing.expectEqualStrings(s.video, try mcpModel(a, done));
    try testing.expectEqualStrings("completed", (try harness.mcpText(a, done)).object.get("status").?.string);
    try testing.expectEqualStrings(s.speech, try mcpModel(a, try mcpCall(d, a, "synthesize_speech", try std.fmt.allocPrint(a, "{{\"input\":\"hello\"{s}}}", .{try modelArg(a, named, s.speech)}))));
    const heard = try mcpCall(d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"audio\":\"data:audio/wav;base64,{s}\"{s}}}", .{ clip, try modelArg(a, named, s.transcription) }));
    try testing.expectEqualStrings(s.transcription, try mcpModel(a, heard));
    try testing.expect(std.mem.startsWith(u8, texts(heard)[0].object.get("text").?.string, s.heard));
    const chatted = try mcpCall(d, a, "chat", try std.fmt.allocPrint(a, "{{\"prompt\":\"hi\"{s}}}", .{try modelArg(a, named, s.chat)}));
    try testing.expectEqualStrings(s.chat, try mcpModel(a, chatted));
    try testing.expectEqualStrings("echo: hi", texts(chatted)[0].object.get("text").?.string);
    const chosen = try mcpCall(d, a, "decide", try std.fmt.allocPrint(a, "{{\"state\":\"make a video of waves\",\"questions\":{{\"route\":{{\"type\":\"choice\",\"criteria\":{{\"image\":\"a picture\",\"video\":\"a clip\"}}}}}}{s}}}", .{try modelArg(a, named, s.decision)}));
    try testing.expectEqualStrings(s.decision, try mcpModel(a, chosen));
    try testing.expect(std.mem.startsWith(u8, texts(chosen)[0].object.get("text").?.string, "route: video"));
    const made = try mcpCall(d, a, "generate_3d", try std.fmt.allocPrint(a, "{{\"image\":\"data:image/png;base64,{s}\",\"wait_s\":60{s}}}", .{ try b64(a, png), try modelArg(a, named, s.asset3d) }));
    try testing.expectEqualStrings(s.asset3d, try mcpModel(a, made));
    const job = try std.json.parseFromSliceLeaky(Value, a, texts(made)[0].object.get("text").?.string, .{}); // the job first
    try testing.expectEqualStrings("completed", job.object.get("status").?.string);
    try testing.expectEqual(@as(usize, 3), job.object.get("files").?.array.items.len);
    var kinds: std.ArrayList([]const u8) = .empty; // the job, a link to each file, the manifest, the input view inline
    for (texts(made)) |c| try kinds.append(a, c.object.get("type").?.string);
    const want = [_][]const u8{ "text", "resource_link", "resource_link", "resource_link", "text", "image" };
    try testing.expectEqual(want.len, kinds.items.len);
    for (want, kinds.items) |w, k| try testing.expectEqualStrings(w, k);
    const manifest = try std.json.parseFromSliceLeaky(Value, a, texts(made)[4].object.get("text").?.string, .{});
    try testing.expectEqualStrings("asset3d", manifest.object.get("object").?.string);
}

/// MCP tools/list's tools and /v1/capabilities' `mcp.tools`, as JSON text.
fn surface(d: *Daemon, a: std.mem.Allocator) ![2][]const u8 {
    const listed = try std.json.parseFromSliceLeaky(Value, a, (try d.call(a, .POST, "/mcp", "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")).body, .{});
    const caps = try std.json.parseFromSliceLeaky(Value, a, (try d.call(a, .GET, "/v1/capabilities", null)).body, .{});
    return .{
        try std.json.Stringify.valueAlloc(a, listed.object.get("result").?.object.get("tools").?, .{}),
        try std.json.Stringify.valueAlloc(a, caps.object.get("mcp").?.object.get("tools").?, .{}),
    };
}

test "drop-in: a swapped model set has the same MCP tools and capabilities; /v1/models lists exactly the configured models" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: [2][2][]const u8 = undefined;
    for ([_]Set{ try setA(a), try setB(a) }, 0..) |s, n| {
        var d = try start(a, "dropin-surface", s, "");
        defer d.stop();
        seen[n] = try surface(&d, a);
        try testing.expect(seen[n][0].len > 1000); // every tool, schema and description
        try testing.expect(std.mem.indexOf(u8, seen[n][1], "\"edit_image\"") != null);

        // Exactly the six configured ids, in config order, each with its kind and capabilities.
        const M = struct { data: []const struct { id: []const u8, kind: []const u8, capabilities: []const []const u8 } };
        const models = try std.json.parseFromSliceLeaky(M, a, (try d.call(a, .GET, "/v1/models", null)).body, .{ .ignore_unknown_fields = true });
        try testing.expectEqual(@as(usize, 7), models.data.len);
        const kinds = [_][]const u8{ "image", "video", "speech", "transcription", "chat", "decision", "asset3d" };
        const caps = [_][]const u8{ if (s.inputs) "text_to_image,image_edit" else "text_to_image", if (s.inputs) "text_to_video,image_to_video" else "text_to_video", "text_to_speech", "speech_to_text", "chat_completion", "typed_decision", "image_to_3d" };
        for (models.data, s.ids(), kinds, caps) |got, id, kind, want| {
            try testing.expectEqualStrings(id, got.id);
            try testing.expectEqualStrings(kind, got.kind);
            try testing.expectEqualStrings(want, try std.mem.join(a, ",", got.capabilities));
        }
    }
    try testing.expectEqualStrings(seen[0][0], seen[1][0]);
    try testing.expectEqualStrings(seen[0][1], seen[1][1]);
}

test "drop-in: every endpoint and MCP tool works on either set, the model named or picked by defaults" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Set{ try setA(a), try setB(a) }) |s| {
        var d = try start(a, "dropin-endpoints", s, "");
        defer d.stop();
        try everyEndpoint(&d, a, s, false);
        try everyEndpoint(&d, a, s, true);
    }
}

test "drop-in: a capability the model lacks is refused, naming it and what it can do; the model that has it serves it" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const image = try b64(a, png);
    for ([_]Set{ try setA(a), try setB(a) }) |s| {
        var d = try start(a, "dropin-caps", s, "");
        defer d.stop();
        const mcp_edit = try mcpCall(&d, a, "edit_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"make it night\",\"image\":\"{s}\",\"size\":\"256x256\",\"inline_images\":false}}", .{image}));
        const http_edit = try d.call(a, .POST, "/v1/images/edits", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"image\":\"{s}\"}}", .{image}));
        const mcp_i2v = try mcpCall(&d, a, "generate_video", try std.fmt.allocPrint(a, "{{\"prompt\":\"waves\",\"size\":\"256x256\",\"seconds\":1,\"audio\":false,\"wait_s\":60,\"image\":\"{s}\"}}", .{image}));
        const http_i2v = try d.call(a, .POST, "/v1/videos", try std.fmt.allocPrint(a, "{{\"prompt\":\"waves\",\"size\":\"256x256\",\"seconds\":\"1\",\"audio\":false,\"input_reference\":\"{s}\"}}", .{image}));
        // Each 3D model's own Gaussian range: refused before it loads, naming the range; an unknown mode by the API.
        const too_many = try std.fmt.allocPrint(a, "{{\"image\":\"{s}\",\"num_gaussians\":100000}}", .{image});
        try expectStatus(try d.call(a, .POST, "/v1/assets3d", too_many), .bad_request, &.{ s.asset3d, s.gaussians });
        try expectToolError(try mcpCall(&d, a, "generate_3d", too_many), &.{ s.asset3d, s.gaussians });
        try expectStatus(try d.call(a, .POST, "/v1/assets3d", try std.fmt.allocPrint(a, "{{\"image\":\"{s}\",\"input_mode\":\"sketch\"}}", .{image})), .bad_request, &.{"input_mode must be synthetic"});
        try expectStatus(try d.call(a, .POST, "/v1/assets3d", "{\"input_mode\":\"real\"}"), .bad_request, &.{"image is required"});
        try testing.expectEqualStrings("unloaded", try d.toolState(a, s.asset3d));
        if (s.inputs) {
            try testing.expectEqualStrings(s.image, try mcpModel(a, mcp_edit));
            try testing.expectEqualStrings(s.image, try modelOf(a, http_edit));
            try testing.expectEqualStrings(s.video, try mcpModel(a, mcp_i2v));
            try testing.expectEqualStrings("completed", (try harness.mcpText(a, mcp_i2v)).object.get("status").?.string);
            try testing.expectEqualStrings(s.video, try videoModel(&d, a, http_i2v));
            continue;
        }
        // No model here edits: MCP names every model and its capabilities; the API refuses the image model it picked.
        try expectToolError(mcp_edit, &.{"no model on this machine can do image_edit: its models are img-b (text_to_image); vid-b (text_to_video)"});
        try expectStatus(http_edit, .bad_request, &.{"model 'img-b' cannot do image_edit (its capabilities: text_to_image)"});
        try expectToolError(mcp_i2v, &.{"model 'vid-b' cannot do image_to_video (its capabilities: text_to_video)"});
        try expectStatus(http_i2v, .bad_request, &.{"model 'vid-b' cannot do image_to_video (its capabilities: text_to_video)"});
        // Named, the same; text input on the same models still works.
        try expectToolError(try mcpCall(&d, a, "generate_video", try std.fmt.allocPrint(a, "{{\"prompt\":\"w\",\"model\":\"vid-b\",\"image\":\"{s}\"}}", .{image})), &.{ "vid-b", "image_to_video" });
        try expectStatus(try d.call(a, .POST, "/v1/images/edits", try std.fmt.allocPrint(a, "{{\"model\":\"img-b\",\"prompt\":\"bars\",\"image\":\"{s}\"}}", .{image})), .bad_request, &.{ "img-b", "image_edit" });
        try testing.expectEqualStrings("img-b", try modelOf(a, try harness.image(&d, a, "img-b", "")));
    }
}

/// list_models' default_for of model `id`, comma-joined.
fn defaultFor(d: *Daemon, a: std.mem.Allocator, id: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for ((try harness.mcpText(a, try mcpCall(d, a, "list_models", "{}"))).object.get("models").?.array.items) |m| {
        if (!std.mem.eql(u8, m.object.get("id").?.string, id)) continue;
        for (m.object.get("default_for").?.array.items) |c| try out.print(a, "{s},", .{c.string});
    }
    return out.items;
}

test "drop-in: two models of a kind; changing only defaults moves the requests that name no model" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = [_][]const u8{
        "{\"id\": \"img-1\", \"kind\": \"image\", \"engine\": \"testpattern\"}",
        "{\"id\": \"img-2\", \"kind\": \"image\", \"engine\": \"testpattern\"}",
        "{\"id\": \"tts-1\", \"kind\": \"speech\", \"engine\": \"testpattern\"}",
        try worker(a, "tts-2", "speech", ""),
        try server(a, "chat-1", "chat", opts.chat_server),
        try server(a, "chat-2", "chat", opts.chat_server),
    };
    const Pick = struct { image: []const u8, speech: []const u8, chat: []const u8, rate: u32 };
    for ([_]Pick{ .{ .image = "img-1", .speech = "tts-1", .chat = "chat-1", .rate = 24_000 }, .{ .image = "img-2", .speech = "tts-2", .chat = "chat-2", .rate = 22_050 } }) |p| {
        const defaults = try std.fmt.allocPrint(a, "{{\"text_to_image\": \"{s}\", \"image_edit\": \"{s}\", \"text_to_speech\": \"{s}\", \"chat_completion\": \"{s}\"}}", .{ p.image, p.image, p.speech, p.chat });
        var d = try Daemon.start(testing.allocator, testing.io, "dropin-defaults", try config(a, &tools, defaults, ""));
        defer d.stop();
        try testing.expectEqualStrings(p.image, try modelOf(a, try d.call(a, .POST, "/v1/images/generations", "{\"prompt\":\"bars\",\"size\":\"256x256\"}")));
        try testing.expectEqualStrings(p.image, try mcpModel(a, try mcpCall(&d, a, "generate_image", "{\"prompt\":\"bars\",\"size\":\"256x256\"}")));
        try testing.expectEqualStrings(p.image, try modelOf(a, try d.call(a, .POST, "/v1/images/edits", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"image\":\"{s}\"}}", .{try b64(a, png)}))));
        try testing.expectEqualStrings("text_to_image,image_edit,", try defaultFor(&d, a, p.image));
        try testing.expectEqual(p.rate, try rateOf(try d.call(a, .POST, "/v1/audio/speech", "{\"input\":\"hello\"}")));
        try testing.expectEqualStrings(p.speech, try mcpModel(a, try mcpCall(&d, a, "synthesize_speech", "{\"input\":\"hello\"}")));
        try testing.expectEqualStrings("text_to_speech,", try defaultFor(&d, a, p.speech));
        try testing.expectEqualStrings(p.chat, try modelOf(a, try d.call(a, .POST, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")));
        try testing.expectEqualStrings("chat_completion,", try defaultFor(&d, a, p.chat));
    }
}

/// `ms` of a sine at `hz` (0: silence), 24 kHz pcm16, as 100 ms appends (voice.zig's `Ws.audio`).
fn speak(w: *voice.Ws, a: std.mem.Allocator, ms: u32, hz: f32) !void {
    var t: usize = 0;
    var left = ms;
    while (left > 0) : (left -|= 100) {
        var bytes: [4800]u8 = undefined;
        for (0..2400) |i| {
            const x: i16 = if (hz == 0) 0 else @intFromFloat(8000 * @sin(@as(f32, @floatFromInt(t)) * 2 * std.math.pi * hz / 24000));
            std.mem.writeInt(i16, bytes[i * 2 ..][0..2], x, .little);
            t += 1;
        }
        var enc: [6400]u8 = undefined;
        try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"input_audio_buffer.append\",\"audio\":\"{s}\"}}", .{std.base64.standard.Encoder.encode(&enc, &bytes)}));
    }
}

test "drop-in: the voice loop on either set: its speech and transcription models are taken, the other set's are unknown, a turn is heard and spoken" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sets = [_]Set{ try setA(a), try setB(a) };
    for (sets, 0..) |s, n| {
        // No chat_url: the session's chat model is this daemon's chat_completion default, the set's own.
        var d = try start(a, "dropin-voice", s, ", \"voice\": {\"partial_interval_ms\": 0, \"prime_chat\": false, \"warm_speech\": false}");
        defer d.stop();
        var w: voice.Ws = undefined;
        try w.open(testing.io, a, d.port, "realtime");
        defer w.close();
        var seen: std.ArrayList(Value) = .empty;
        _ = try w.until(a, "session.created", &seen);

        try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"session.update\",\"session\":{{\"speech_model\":\"{s}\",\"input_audio_transcription\":{{\"model\":\"{s}\"}}}}}}", .{ s.speech, s.transcription }));
        const session = (try w.until(a, "session.updated", &seen)).object.get("session").?.object;
        try testing.expectEqualStrings(s.speech, session.get("speech_model").?.string);
        try testing.expectEqualStrings(s.transcription, session.get("input_audio_transcription").?.object.get("model").?.string);
        const other = sets[1 - n];
        try w.send(try std.fmt.allocPrint(a, "{{\"type\":\"session.update\",\"session\":{{\"speech_model\":\"{s}\"}}}}", .{other.speech}));
        const refused = try w.until(a, "error", &seen);
        const msg = refused.object.get("error").?.object.get("message").?.string;
        if (std.mem.indexOf(u8, msg, try std.fmt.allocPrint(a, "unknown model '{s}'", .{other.speech})) == null) {
            std.debug.print("refusal: {s}\n", .{msg});
            return error.TestExpectedEqual;
        }

        // One spoken turn: the set's transcription model hears it, its chat model answers, its speech model says it.
        try speak(&w, a, 300, 0);
        try speak(&w, a, 900, 440);
        try speak(&w, a, 800, 0);
        const heard = try w.until(a, "conversation.item.input_audio_transcription.completed", &seen);
        try testing.expect(std.mem.startsWith(u8, heard.object.get("transcript").?.string, s.heard));
        _ = try w.until(a, "response.audio.delta", &seen);
        const done = try w.until(a, "response.done", &seen);
        try testing.expectEqualStrings("completed", done.object.get("response").?.object.get("status").?.string);
    }
}

/// /v1/assistant's `voice.model`: the speech model a call speaks with when it names none.
fn callSpeech(d: *Daemon, a: std.mem.Allocator) ![]const u8 {
    const T = struct { voice: ?struct { model: []const u8 } = null };
    const v = try std.json.parseFromSliceLeaky(T, a, (try d.call(a, .GET, "/v1/assistant", null)).body, .{ .ignore_unknown_fields = true });
    return v.voice.?.model;
}

test "drop-in: the voice loop speaks with the text_to_speech default unless voice.speech_models names a chain" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // A model the old built-in chain named first, configured beside the new default: the default speaks.
    const tools = [_][]const u8{ "{\"id\": \"kitten-tts-2\", \"kind\": \"speech\", \"engine\": \"testpattern\"}", try worker(a, "tts-new", "speech", "") };
    const defaults = "{\"text_to_speech\": \"tts-new\"}";
    for ([_]struct { top: []const u8, want: []const u8 }{
        .{ .top = "", .want = "tts-new" },
        .{ .top = ", \"voice\": {\"speech_models\": [\"kitten-tts-2\", \"tts-new\"]}", .want = "kitten-tts-2" },
    }) |c| {
        var d = try Daemon.start(testing.allocator, testing.io, "dropin-speech", try config(a, &tools, defaults, c.top));
        defer d.stop();
        try testing.expectEqualStrings(c.want, try callSpeech(&d, a));
    }
}

test "drop-in: a transcription model's declared languages are listed, and a language outside them is refused before it loads" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = [_][]const u8{"{\"id\": \"ears\", \"kind\": \"transcription\", \"engine\": \"testpattern\", \"options\": {\"languages\": [\"en\", \"de\"]}}"};
    var d = try Daemon.start(testing.allocator, testing.io, "dropin-languages", try config(a, &tools, "{}", ""));
    defer d.stop();
    const M = struct { data: []const struct { id: []const u8, languages: ?[]const []const u8 = null } };
    const models = try std.json.parseFromSliceLeaky(M, a, (try d.call(a, .GET, "/v1/models", null)).body, .{ .ignore_unknown_fields = true });
    try testing.expectEqualStrings("en,de", try std.mem.join(a, ",", models.data[0].languages.?));
    const listed = (try harness.mcpText(a, try mcpCall(&d, a, "list_models", "{}"))).object.get("models").?.array.items[0];
    try testing.expectEqual(@as(usize, 2), listed.object.get("languages").?.array.items.len);

    const clip = try b64(a, try wav(a));
    const refusal = "model 'ears' does not hear language 'ja'; its languages: en, de";
    try expectStatus(try d.call(a, .POST, "/v1/audio/transcriptions", try std.fmt.allocPrint(a, "{{\"file\":\"{s}\",\"language\":\"ja\"}}", .{clip})), .bad_request, &.{refusal});
    try expectToolError(try mcpCall(&d, a, "transcribe_audio", try std.fmt.allocPrint(a, "{{\"audio\":\"data:audio/wav;base64,{s}\",\"language\":\"ja\"}}", .{clip})), &.{refusal});
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "ears")); // refused before anything loaded
    try expectStatus(try d.call(a, .POST, "/v1/audio/transcriptions", try std.fmt.allocPrint(a, "{{\"file\":\"{s}\",\"language\":\"de\"}}", .{clip})), .ok, &.{", language de"});
}
