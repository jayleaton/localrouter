//! The daemon's configuration: where it listens and stores outputs, the memory budget, and the tools it serves.

const std = @import("std");
const Kind = @import("tool/request.zig").Kind;
const Capability = @import("tool/request.zig").Capability;
const bind = @import("bind.zig");

pub const gib: u64 = 1 << 30;
pub const mib: u64 = 1 << 20;

/// One tool. `engine` names a compiled-in engine; `cmd` runs any program that speaks the worker protocol instead.
pub const ToolConfig = struct {
    id: []const u8, // the OpenAI model id
    kind: Kind,
    name: []const u8 = "",
    engine: []const u8 = "",
    cmd: []const []const u8 = &.{},
    weights: []const u8 = "",
    resident_bytes: u64 = 0, // estimate for `cmd` tools; engines compute their own
    working_bytes: u64 = 0, // per request, for `cmd` tools
    idle_ttl_s: u32 = 120, // 0: unload right after each job
    priority: i32 = 0, // higher is kept longer: memory pressure unloads lower priorities first (then least recently used)
    keep_loaded: bool = false, // load at start if it fits, skip the idle TTL, reload when memory frees; still evictable
    capabilities: []const Capability = &.{}, // empty: the engine's (a `cmd` tool: its kind's plain one, e.g. speech_to_text)
    load_timeout_s: u32 = 900,
    generate_timeout_s: u32 = 3600,
    options: ?std.json.Value = null, // engine-specific, passed through to the worker; a speech tool's `voices` are checked and listed
    // A file the model's qualify script writes on this machine. When set, the tool is listed but refused (503) until it
    // exists and holds `"qualified": true` (`engine.unqualified`); unset: no gate.
    qualification: []const u8 = "",
    lane: ?Lane = null, // the scheduler executor it runs on; null: realtime for speech, listen for transcription, chat for chat, decide for decisions, else main
};

/// The scheduler's executors: one job at a time on each. Speech runs beside a chat reply, a chat turn beside a video,
/// a transcription never waits behind a sentence of speech, and a decision never waits behind a chat reply.
pub const Lane = enum { main, realtime, chat, listen, decide };

pub fn laneOf(t: *const ToolConfig) Lane {
    return t.lane orelse switch (t.kind) {
        .speech => .realtime,
        .transcription => .listen,
        .chat => .chat,
        .decision => .decide,
        else => .main, // image, video, and every other kind
    };
}

/// The voice loop (`/voice`, `/v1/realtime`, docs/dev/VOICE.md): where the chat model is and the session defaults.
pub const Voice = struct {
    remote: bool = false, // serve /voice and /v1/realtime to non-loopback peers too
    chat_url: []const u8 = "", // an OpenAI-compatible base URL ending in /v1; "": this daemon
    chat_api_key: []const u8 = "", // sent as a bearer token to chat_url (with chat_url unset: the session's own token)
    chat_model: []const u8 = "", // "": this daemon's chat_completion default (with chat_url unset), else the server's
    chat_extra: ?std.json.Value = null, // members added to every chat request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    temperature: ?f32 = null, // a session's chat temperature until it sets its own (session.update); null: the chat server's
    // (Qwen with thinking off wants 0.7: at its server's 1.0 a spoken picture request went without the tool call 1 in 4)
    speech_model: []const u8 = "", // "": the first of speech_models that is configured and qualified
    speech_models: []const []const u8 = &.{ "fishaudio/drama-3", "kitten-tts-2" }, // in order; one that fails to serve falls through to the next; none: the text_to_speech default
    transcription_model: []const u8 = "", // "": the speech_to_text default
    voice: []const u8 = "", // "": the speech model's default voice
    instructions: []const u8 = "You are a helpful voice assistant running on the user's own machine. Answer in short, natural spoken sentences without markdown, lists or emoji.",
    partial_interval_ms: u32 = 0, // a partial transcript per this much new speech; 0 (default): none, since a partial
    // on a non-streaming STT model (Whisper) can hold the turn's final transcript back by a whole decode
    early_stt_ms: u32 = 150, // quiet after speech that starts transcribing the turn so far (reused when the wait ends with
    // nothing more said: only the transcript is early, nothing considers it before the wait ends); 0: none
    prime_chat: bool = true, // at the start and after each response the chat server reads the conversation so far
    // (max_tokens 1, discarded), so the next turn's request finds it in its prefix cache
    warm_speech: bool = true, // the first session after a start speaks one sentence (discarded) as it connects, on its own
    // thread: the speech model's first job after a load is slow (0.84 s to first audio on DGX-01, 0.1 s warm)
    greedy_transcripts: bool = true, // voice turns decode in one greedy pass (Whisper: transcripts identical to its
    // recipe on the qualify set and the e2e prompts, measured on DGX-01, about 150 ms sooner)
    max_history_chars: u32 = 24000, // older turns are dropped past this
    tools: bool = true, // offer generate_image and generate_video to the chat model
    manage_models: bool = false, // offer list_models, reserve_for, unload_model and load_model to the chat model, with the
    // model management procedure (src/voice/model_skill.md) after its instructions; the web client reads the same
    tool_wait_s: u32 = 1800, // how long a session waits for a tool's job
    shadow_decisions: bool = false, // after each reply, ask the decision model (kind `decision`) how it would have routed the
    // turn and log it beside what the chat model did (`voice shadow decision` lines); nothing acts on it
};

/// Device pairing (src/pair.zig): off by default, so every client that reaches the daemon may use it, as before.
pub const Pairing = struct {
    required: bool = false, // every route but /health, /v1/pair and /voice answers only paired devices, each within its
    // scopes (`localrouter pair`, src/pair.zig); no loopback exemption: behind `tailscale serve` every peer arrives as
    // loopback
};

/// Computer control (docs/dev/COMPUTER.md): Jarvis using a paired desktop helper's computer. Its routes need device
/// pairing; without it they refuse.
pub const Computer = struct {
    enabled: bool = true, // the /v1/desktop and /v1/computer routes and the computer tools exist
    voice: bool = true, // offer computer_control and computer_use to a voice session's chat model while a helper is connected
    decision_model: []const u8 = "", // the chat model that makes the step decisions; "": the chat_completion default
    max_steps: u32 = 15, // a computer-use task's actions (at most 15)
    timeout_s: u32 = 90, // a task's working time, not counting time waiting for a tap (at most 90)
    approval_timeout_s: u32 = 60, // how long a tap may take before the step is refused
};

pub const Config = struct {
    host: []const u8 = bind.default_host, // an IP, localhost, all (0.0.0.0) or tailscale (this machine's tailnet IPv4)
    port: u16 = 8190,
    data_dir: []const u8 = "/data",
    machine: []const u8 = "",
    budget_bytes: u64 = 0, // cap on loaded + working bytes; 0: only MemAvailable - reserve
    reserve_bytes: u64 = 8 * gib, // kept free for the OS and co-resident services
    keep_outputs_s: u32 = 24 * 3600,
    max_queue: u32 = 64,
    allow_private_urls: bool = false, // URL inputs (edits, image to video) may name loopback / private / link-local hosts
    local_files: []const []const u8 = &.{}, // directories whose audio files MCP tools may read by path (`transcribe_audio` path); none by default: the API has no auth
    gpu_lock: []const u8 = "", // the host's GPU lock: the scheduler holds it (shared) while any tool is loaded, against GPU users outside LocalRouter
    defaults: Defaults = .{}, // the model a request that names none goes to, per capability
    voice: Voice = .{},
    pairing: Pairing = .{},
    computer: Computer = .{},
    tools: []const ToolConfig = &default_tools,

    pub fn tool(c: *const Config, id: []const u8) ?*const ToolConfig {
        for (c.tools) |*t| if (std.mem.eql(u8, t.id, id)) return t;
        return null;
    }
};

/// A tool id per capability, `{"text_to_image": "qwen-image-2.1-turbo"}`: requests that name no model use it. Unset:
/// the first tool (in config order) that has the capability. `serve` checks each names a tool that can do it.
pub const Defaults = struct {
    text_to_image: ?[]const u8 = null,
    image_edit: ?[]const u8 = null,
    text_to_video: ?[]const u8 = null,
    image_to_video: ?[]const u8 = null,
    text_to_speech: ?[]const u8 = null,
    speech_to_text: ?[]const u8 = null,
    chat_completion: ?[]const u8 = null,
    typed_decision: ?[]const u8 = null,

    pub fn get(d: Defaults, cap: Capability) ?[]const u8 {
        return switch (cap) {
            inline else => |c| @field(d, @tagName(c)),
        };
    }
};

/// With no config file LocalRouter serves its self-test tool only.
pub const default_tools = [_]ToolConfig{
    .{ .id = "testpattern", .kind = .image, .name = "Test pattern (self-test)", .engine = "testpattern", .idle_ttl_s = 30 },
    .{ .id = "testpattern-video", .kind = .video, .name = "Test pattern video (self-test)", .engine = "testpattern", .idle_ttl_s = 30 },
    .{ .id = "testpattern-speech", .kind = .speech, .name = "Test tone speech (self-test)", .engine = "testpattern", .idle_ttl_s = 30 },
    .{ .id = "testpattern-transcription", .kind = .transcription, .name = "Test transcription (self-test)", .engine = "testpattern", .idle_ttl_s = 30 },
};

/// A chat tool's priority when its config names none: above every other kind's default (0).
pub const chat_priority: i32 = 100;

/// Parses a JSON config; the result lives in `arena`.
pub fn parse(arena: std.mem.Allocator, text: []const u8) !Config {
    var c = try std.json.parseFromSliceLeaky(Config, arena, text, .{ .ignore_unknown_fields = false, .allocate = .alloc_always });
    try chatDefaults(arena, text, &c);
    return c;
}

/// The chat model is the machine's always-warm default: a chat tool whose config does not say otherwise is
/// `keep_loaded` with `chat_priority`, so it is unloaded last, only for a request that needs its memory.
fn chatDefaults(arena: std.mem.Allocator, text: []const u8, c: *Config) !void {
    for (c.tools) |t| {
        if (t.kind == .chat) break;
    } else return;
    const raw = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    const list = raw.object.get("tools").?.array.items; // typed parsing succeeded: the list is there, one object a tool
    const tools = try arena.dupe(ToolConfig, c.tools);
    for (tools, list) |*t, v| {
        if (t.kind != .chat) continue;
        if (v.object.get("priority") == null) t.priority = chat_priority;
        if (v.object.get("keep_loaded") == null) t.keep_loaded = true;
    }
    c.tools = tools;
}

test "a chat tool is kept loaded at the highest default priority unless its config says otherwise" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(),
        \\{"defaults": {"chat_completion": "qwen"}, "tools": [
        \\  {"id": "img", "kind": "image", "engine": "testpattern"},
        \\  {"id": "qwen", "kind": "chat", "engine": "openai_server", "qualification": "/data/q.json"},
        \\  {"id": "small", "kind": "chat", "engine": "openai_server", "priority": 5, "keep_loaded": false}]}
    );
    try std.testing.expectEqual(chat_priority, c.tool("qwen").?.priority);
    try std.testing.expect(c.tool("qwen").?.keep_loaded);
    try std.testing.expectEqualStrings("/data/q.json", c.tool("qwen").?.qualification);
    try std.testing.expectEqual(@as(i32, 5), c.tool("small").?.priority);
    try std.testing.expect(!c.tool("small").?.keep_loaded);
    try std.testing.expectEqual(@as(i32, 0), c.tool("img").?.priority);
    try std.testing.expect(!c.tool("img").?.keep_loaded);
    try std.testing.expectEqualStrings("qwen", c.defaults.get(.chat_completion).?);
}

test "parse a config with an external tool" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(),
        \\{"port": 9000, "data_dir": "/tmp/x", "tools": [
        \\  {"id": "qwen-image-2.1", "kind": "image", "engine": "qwen_image", "weights": "/models/qwen", "options": {"precision": "nvfp4"}},
        \\  {"id": "py-tts", "kind": "video", "cmd": ["python3", "-m", "tts"], "resident_bytes": 1000}
        \\]}
    );
    try std.testing.expectEqual(@as(u16, 9000), c.port);
    try std.testing.expectEqualStrings("nvfp4", c.tool("qwen-image-2.1").?.options.?.object.get("precision").?.string);
    try std.testing.expectEqual(@as(usize, 3), c.tool("py-tts").?.cmd.len);
    try std.testing.expect(c.tool("nope") == null);
    try std.testing.expect(c.defaults.get(.text_to_image) == null);
}

test "defaults name a model per capability; unknown capabilities are rejected" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(),
        \\{"defaults": {"text_to_image": "turbo", "image_to_video": "v"}}
    );
    try std.testing.expectEqualStrings("turbo", c.defaults.get(.text_to_image).?);
    try std.testing.expectEqualStrings("v", c.defaults.get(.image_to_video).?);
    try std.testing.expect(c.defaults.get(.image_edit) == null);
    try std.testing.expectError(error.UnknownField, parse(arena.allocator(), "{\"defaults\": {\"text_to_music\": \"x\"}}"));
    const s = try parse(arena.allocator(),
        \\{"defaults": {"text_to_speech": "kitten", "speech_to_text": "whisper"}, "tools": [
        \\  {"id": "whisper", "kind": "transcription", "cmd": ["python3", "worker.py"], "resident_bytes": 4000000000}]}
    );
    try std.testing.expectEqualStrings("whisper", s.defaults.get(.speech_to_text).?);
    try std.testing.expectEqualStrings("kitten", s.defaults.get(.text_to_speech).?);
    try std.testing.expectEqual(Kind.transcription, s.tool("whisper").?.kind);
}

test "lanes: speech kinds default to realtime, a tool may name its lane; voice settings parse" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(),
        \\{"voice": {"chat_url": "http://127.0.0.1:9/v1", "partial_interval_ms": 0}, "tools": [
        \\  {"id": "tts", "kind": "speech", "cmd": ["x"]}, {"id": "img", "kind": "image", "engine": "testpattern"},
        \\  {"id": "stt", "kind": "transcription", "cmd": ["x"], "lane": "main"}, {"id": "llm", "kind": "chat", "engine": "openai_server"},
        \\  {"id": "jev", "kind": "decision", "engine": "openai_server"}]}
    );
    try std.testing.expectEqual(Lane.realtime, laneOf(c.tool("tts").?));
    try std.testing.expectEqual(Lane.main, laneOf(c.tool("img").?));
    try std.testing.expectEqual(Lane.main, laneOf(c.tool("stt").?));
    try std.testing.expectEqual(Lane.chat, laneOf(c.tool("llm").?));
    try std.testing.expectEqual(Lane.decide, laneOf(c.tool("jev").?));
    try std.testing.expectEqualStrings("http://127.0.0.1:9/v1", c.voice.chat_url);
    try std.testing.expectEqual(@as(u32, 0), c.voice.partial_interval_ms);
    try std.testing.expect(!c.voice.remote);
}

test "the default host is loopback; host accepts the keywords" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const d = try parse(arena.allocator(), "{}");
    try std.testing.expectEqualStrings("127.0.0.1", d.host);
    const t = try parse(arena.allocator(), "{\"host\": \"tailscale\"}");
    try std.testing.expectEqualStrings("tailscale", t.host);
}

test "pairing is off unless the config requires it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expect(!(try parse(arena.allocator(), "{}")).pairing.required);
    try std.testing.expect((try parse(arena.allocator(), "{\"pairing\": {\"required\": true}}")).pairing.required);
    try std.testing.expectError(error.UnknownField, parse(arena.allocator(), "{\"pairing\": {\"loopback\": true}}"));
}
