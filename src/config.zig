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
    keep_loaded: bool = false, // the legacy spelling of `residency: "always"` (used when `residency` is unset)
    residency: ?Residency = null, // when it is loaded (`residencyOf`); `<data_dir>/residency.json` overrides it (sched/residency.zig)
    // An on-demand model that may only use spare room: it never unloads a `keep_loaded` tool to load (idle tools that are
    // not kept may still go) and loads only with room left for the kept tools' working bytes; a request it cannot serve
    // so is refused (503 insufficient_memory) before anything is unloaded. For models nothing waits on, such as the
    // decision model behind the voice loop's shadow decisions.
    spare_room_only: bool = false,
    capabilities: []const Capability = &.{}, // empty: the engine's (a `cmd` tool: its kind's plain one, e.g. speech_to_text)
    load_timeout_s: u32 = 900,
    generate_timeout_s: u32 = 3600,
    // A chat request whose limit (max_tokens, max_completion_tokens, Responses' max_output_tokens) is below this, and
    // that does not choose reasoning itself, goes to the model with thinking off (`thinking_off`) and a warning, so it
    // answers rather than spending the limit on reasoning (`api/budget.zig`); 0: never.
    min_thinking_tokens: u32 = 1024,
    thinking_off: ?std.json.Value = null, // the members that switch the chat model's thinking off; null: {"chat_template_kwargs": {"enable_thinking": false}}
    options: ?std.json.Value = null, // engine-specific, passed through to the worker; a speech tool's `voices` are checked and listed
    // A file the model's qualify script writes on this machine. When set, the tool is listed but refused (503) until it
    // exists and holds `"qualified": true` (`engine.unqualified`); unset: no gate.
    qualification: []const u8 = "",
    license: ?License = null, // the weights' licence, listed by /v1/models; null: the engine's (`engine.license`), if any
    lane: ?Lane = null, // the scheduler executor it runs on; null: realtime for speech, listen for transcription, chat for chat, decide for decisions, else main
};

/// When a tool is loaded. `always`: at start if it fits, never unloaded for its idle TTL, back by itself once memory
/// frees after an eviction (still evictable, as before). `call` (chat, speech and transcription only): loaded when a
/// realtime voice session connects and kept like `always` while any call is open; unloaded `idle_ttl_s` after the last
/// call ends and its last job is done. `on_demand`: loaded by its first request, unloaded `idle_ttl_s` after its last
/// job. Outside a call, a `call` tool loads for a request as an `on_demand` one does. `disabled`: never loaded (not at
/// start, for a call, a request or a shadow decision); its requests are refused (409 `model_disabled`), and a request
/// that names no model goes to the next enabled tool of its kind.
pub const Residency = enum { always, call, on_demand, disabled };

/// A tool's residency as its config says: `residency` when set, else `always` for `keep_loaded`, else `on_demand`.
pub fn residencyOf(t: *const ToolConfig) Residency {
    return t.residency orelse if (t.keep_loaded) .always else .on_demand;
}

/// Whether a tool of `kind` may be loaded for calls: the voice loop's models only.
pub fn callAllowed(kind: Kind) bool {
    return kind == .chat or kind == .speech or kind == .transcription;
}

/// A model's licence as its publisher states it, for clients to read: LocalRouter lists it and enforces none of it.
pub const License = struct {
    name: []const u8,
    url: []const u8 = "",
    use: []const u8 = "", // what it allows, in short, e.g. "research and evaluation only"
    commercial: ?bool = null, // whether it grants commercial use at all (its conditions in `use`); null: not stated
    excluded_regions: []const []const u8 = &.{}, // where it grants nothing
    notice: []const u8 = "", // what a product or its outputs must show, if the licence asks for it
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
    remote: ?bool = null, // serve /voice and /v1/realtime to non-loopback peers too; null: `Config.voiceRemote`
    chat_url: []const u8 = "", // an OpenAI-compatible base URL ending in /v1; "": this daemon
    chat_api_key: []const u8 = "", // sent as a bearer token to chat_url (with chat_url unset: the session's own token)
    chat_model: []const u8 = "", // "": this daemon's chat_completion default (with chat_url unset), else the server's
    chat_extra: ?std.json.Value = null, // members added to every chat request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    temperature: ?f32 = null, // a session's chat temperature until it sets its own (session.update); null: the chat server's
    // (Qwen with thinking off wants 0.7: at its server's 1.0 a spoken picture request went without the tool call 1 in 4)
    speech_model: []const u8 = "", // "": the first of speech_models that is configured and qualified
    speech_models: []const []const u8 = &.{}, // in order; one that fails to serve falls through to the next; none (the
    // default): the text_to_speech default alone, so a model is swapped in `defaults` without naming it here too
    transcription_model: []const u8 = "", // "": the speech_to_text default
    voice: []const u8 = "", // "": the speech model's default voice
    // `{name}` is the assistant's name (`assistant.name`, or the one set over /v1/assistant); instructions that do not
    // mention the name get "You are <name>." before them (assistant.zig `systemPrompt`)
    instructions: []const u8 = "You are {name}, a helpful voice assistant running on the user's own machine. Answer in short, natural spoken sentences without markdown, lists or emoji.",
    partial_interval_ms: u32 = 0, // a partial transcript per this much new speech; 0 (default): none, since a partial
    // on a non-streaming STT model (Whisper) can hold the turn's final transcript back by a whole decode
    early_stt_ms: u32 = 150, // quiet after speech that starts transcribing the turn so far (reused when the wait ends with
    // nothing more said: only the transcript is early, nothing considers it before the wait ends); 0: none
    prime_chat: bool = true, // at the start and after each response the chat server reads the conversation so far
    // (max_tokens 1, discarded), so the next turn's request finds it in its prefix cache
    warm_speech: bool = true, // a kept speech model speaks one word (discarded) as it loads, and a call whose voice is not
    // the one warm in the loaded instance speaks it as it connects, before `localrouter.session.ready`: the first job
    // after a load is slow (0.84 s to first audio on DGX-01, 0.1 s warm), and on the Mac a voice switch is 4 to 6 s
    greedy_transcripts: bool = true, // voice turns decode in one greedy pass (Whisper: transcripts identical to its
    // recipe on the qualify set and the e2e prompts, measured on DGX-01, about 150 ms sooner)
    speech_first: bool = true, // while a reply is written, this daemon's own chat model is paused (SIGSTOP, lossless) whenever
    // its speech is under 2 s ahead of real time, until 3.5 s ahead (sched/yielder.zig): speech beside a decoding chat
    // model came at 0.55x real time on DGX-02, alone at 1.4x, so a long reply streams from its first sentence
    speak_written: bool = false, // true: a reply's speech starts once its spoken text is complete, not sentence by
    // sentence while the chat model still writes (with the GPU to itself the chat writes it in half the time, then speech
    // plays without a gap, but nothing is heard until the whole text is written: rejected as "not streaming at all");
    // speech first takes precedence
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

/// The assistant (src/assistant.zig): `name` is what it is called until a client sets another over /v1/assistant.
pub const Assistant = struct {
    name: []const u8 = @import("assistant.zig").default_name,
};

/// Computer control (docs/dev/COMPUTER.md): the assistant using a paired desktop helper's computer. Its routes need device
/// pairing; without it they refuse.
pub const Computer = struct {
    enabled: bool = true, // the /v1/desktop and /v1/computer routes and the computer tools exist
    voice: bool = true, // offer computer_control and computer_use to a voice session's chat model while a helper is connected
    decision_model: []const u8 = "", // the chat model that makes the step decisions; "": the chat_completion default
    max_steps: u32 = 15, // a computer-use task's actions (at most 15)
    timeout_s: u32 = 90, // a task's working time, not counting time waiting for a tap (at most 90)
    approval_timeout_s: u32 = 60, // how long a tap may take before the step is refused
};

/// How many machines serve this LocalRouter (`src/topology.zig`, docs/dev/MULTI-SPARK.md). `single` (the default): this
/// daemon is the whole box, as before. `multi`: several node daemons, each admitting its own models against its own
/// memory and GPU lock; one of them, the front, serves the API for all of them and sends each request to the node that
/// hosts its model.
pub const Topology = struct {
    mode: enum { single, multi } = .single,
    self: []const u8 = "", // this daemon's node id (`serve --node` overrides it)
    nodes: []const Node = &.{},
    token_file: []const u8 = "", // a file holding the bearer token the front sends to the other nodes (a paired device's)
    health_interval_ms: u32 = 2000, // how often the front asks each node how it is
    down_after: u32 = 3, // missed health answers in a row before a node counts as down
};

/// One machine of a multi-node topology.
pub const Node = struct {
    id: []const u8,
    addresses: []const []const u8, // IPv4 addresses of its daemon, tried in order (both ConnectX-7 links, then the LAN)
    port: u16 = 8190,
    role: enum { front, node } = .node,
    models: []const []const u8 = &.{}, // the tools it runs (ids in `tools`)
    fallback: []const []const u8 = &.{}, // front only: tools of other nodes it runs itself while their node is down
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
    assistant: Assistant = .{},
    voice: Voice = .{},
    stt: Stt = .{},
    pairing: Pairing = .{},
    computer: Computer = .{},
    topology: Topology = .{},
    tools: []const ToolConfig = &default_tools,

    pub fn tool(c: *const Config, id: []const u8) ?*const ToolConfig {
        for (c.tools) |*t| if (std.mem.eql(u8, t.id, id)) return t;
        return null;
    }

    /// The language a transcription runs in (voice turns, `/v1/audio/transcriptions`, MCP): the caller's, else
    /// `stt.language`; "auto" (from either) is "": the model detects it. Language detection on a 1 to 2 s turn is
    /// unreliable (a clean "Stop." came back as Italian and as Korean), so the server's default is a language.
    pub fn sttLanguage(c: *const Config, asked: []const u8) []const u8 {
        const l = std.mem.trim(u8, asked, " \t");
        const chosen = if (l.len > 0) l else std.mem.trim(u8, c.stt.language, " \t");
        return if (std.ascii.eqlIgnoreCase(chosen, "auto")) "" else chosen;
    }
};

/// Speech to text everywhere: `language` is what a transcription runs in when its caller names none (a Whisper code
/// such as "en"; "auto": detect it on each request).
pub const Stt = struct {
    language: []const u8 = "en",
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
    voice_clone: ?[]const u8 = null,

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
/// Whether /voice and /v1/realtime answer non-loopback peers: `voice.remote` when set; else on when the daemon listens
/// beyond loopback with pairing required (each call still needs a paired device's token with its scope), off otherwise
/// (behind `tailscale serve` every peer is loopback anyway). Without pairing a listening microphone endpoint stays local.
pub fn voiceRemote(c: Config) bool {
    return c.voice.remote orelse (c.pairing.required and !loopbackHost(c.host));
}

/// Discovery's `realtime.remote`: whether a call from another machine that reaches the daemon is served. True behind a
/// loopback-only listener too, where a proxy such as `tailscale serve` decides who reaches it (its peers are loopback).
pub fn remoteCalls(c: Config) bool {
    return voiceRemote(c) or loopbackHost(c.host);
}

/// `host` (as `bind.resolve` takes it) is a loopback address or name.
fn loopbackHost(host: []const u8) bool {
    return std.ascii.eqlIgnoreCase(host, "localhost") or std.mem.startsWith(u8, host, "127.") or std.mem.eql(u8, host, "::1");
}

pub fn parse(arena: std.mem.Allocator, text: []const u8) !Config {
    var c = try std.json.parseFromSliceLeaky(Config, arena, text, .{ .ignore_unknown_fields = false, .allocate = .alloc_always });
    try chatDefaults(arena, text, &c);
    for (c.tools) |t| if (t.residency == .call and !callAllowed(t.kind)) {
        std.log.warn("tool {s}: residency \"call\" is for chat, speech and transcription models, not {s}", .{ t.id, @tagName(t.kind) });
        return error.InvalidResidency;
    };
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
        if (v.object.get("keep_loaded") == null and v.object.get("residency") == null) t.keep_loaded = true;
    }
    c.tools = tools;
}

test "transcriptions run in the caller's language, else stt.language; auto detects" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const d = try parse(arena.allocator(), "{}");
    try std.testing.expectEqualStrings("en", d.sttLanguage(""));
    try std.testing.expectEqualStrings("de", d.sttLanguage("de"));
    try std.testing.expectEqualStrings("", d.sttLanguage("auto"));
    try std.testing.expectEqualStrings("", d.sttLanguage(" Auto "));
    const ja = try parse(arena.allocator(), "{\"stt\": {\"language\": \"ja\"}}");
    try std.testing.expectEqualStrings("ja", ja.sttLanguage(""));
    try std.testing.expectEqualStrings("en", ja.sttLanguage("en"));
    const auto = try parse(arena.allocator(), "{\"stt\": {\"language\": \"auto\"}}");
    try std.testing.expectEqualStrings("", auto.sttLanguage(""));
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

test "residency: keep_loaded is the legacy spelling of always; residency wins when set; call is for the voice loop only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(),
        \\{"tools": [
        \\  {"id": "img", "kind": "image", "engine": "testpattern"},
        \\  {"id": "kept", "kind": "image", "engine": "testpattern", "keep_loaded": true},
        \\  {"id": "qwen", "kind": "chat", "engine": "openai_server"},
        \\  {"id": "qwen-call", "kind": "chat", "engine": "openai_server", "residency": "call"},
        \\  {"id": "tts", "kind": "speech", "cmd": ["x"], "keep_loaded": true, "residency": "on_demand"}]}
    );
    try std.testing.expectEqual(Residency.on_demand, residencyOf(c.tool("img").?));
    try std.testing.expectEqual(Residency.always, residencyOf(c.tool("kept").?));
    try std.testing.expectEqual(Residency.always, residencyOf(c.tool("qwen").?)); // a chat model is kept by default
    try std.testing.expectEqual(Residency.call, residencyOf(c.tool("qwen-call").?));
    try std.testing.expect(!c.tool("qwen-call").?.keep_loaded); // residency set: the chat default does not add keep_loaded
    try std.testing.expectEqual(Residency.on_demand, residencyOf(c.tool("tts").?));
    try std.testing.expectError(error.InvalidResidency, parse(arena.allocator(),
        \\{"tools": [{"id": "img", "kind": "image", "engine": "testpattern", "residency": "call"}]}
    ));
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
    try std.testing.expect(c.voice.remote == null and !voiceRemote(c));
}

test "voice answers remote peers by default only beyond loopback with pairing required" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expect(!voiceRemote(try parse(a, "{}"))); // loopback, no pairing
    try std.testing.expect(!voiceRemote(try parse(a, "{\"host\": \"all\"}"))); // open, but nothing gates it
    try std.testing.expect(!voiceRemote(try parse(a, "{\"host\": \"localhost\", \"pairing\": {\"required\": true}}")));
    try std.testing.expect(!voiceRemote(try parse(a, "{\"host\": \"127.0.0.1\", \"pairing\": {\"required\": true}}")));
    try std.testing.expect(voiceRemote(try parse(a, "{\"host\": \"all\", \"pairing\": {\"required\": true}}")));
    try std.testing.expect(voiceRemote(try parse(a, "{\"host\": \"tailscale\", \"pairing\": {\"required\": true}}")));
    try std.testing.expect(voiceRemote(try parse(a, "{\"host\": \"192.168.1.117\", \"pairing\": {\"required\": true}}")));
    // set: as set
    try std.testing.expect(!voiceRemote(try parse(a, "{\"host\": \"all\", \"pairing\": {\"required\": true}, \"voice\": {\"remote\": false}}")));
    try std.testing.expect(voiceRemote(try parse(a, "{\"voice\": {\"remote\": true}}")));
    // discovery: false only for a daemon on the network that keeps calls to itself
    try std.testing.expect(remoteCalls(try parse(a, "{}")));
    try std.testing.expect(!remoteCalls(try parse(a, "{\"host\": \"all\"}")));
    try std.testing.expect(remoteCalls(try parse(a, "{\"host\": \"all\", \"pairing\": {\"required\": true}}")));
}

test "the default host is loopback; host accepts the keywords" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const d = try parse(arena.allocator(), "{}");
    try std.testing.expectEqualStrings("127.0.0.1", d.host);
    const t = try parse(arena.allocator(), "{\"host\": \"tailscale\"}");
    try std.testing.expectEqualStrings("tailscale", t.host);
}

test "the assistant is Jarvis unless the config names it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("Jarvis", (try parse(arena.allocator(), "{}")).assistant.name);
    try std.testing.expectEqualStrings("Ava", (try parse(arena.allocator(), "{\"assistant\": {\"name\": \"Ava\"}}")).assistant.name);
}

test "pairing is off unless the config requires it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expect(!(try parse(arena.allocator(), "{}")).pairing.required);
    try std.testing.expect((try parse(arena.allocator(), "{\"pairing\": {\"required\": true}}")).pairing.required);
    try std.testing.expectError(error.UnknownField, parse(arena.allocator(), "{\"pairing\": {\"loopback\": true}}"));
}
