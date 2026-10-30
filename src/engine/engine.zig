//! The worker-side contract every compiled-in engine implements, and the table of engines in this binary.
//! `needs` prices the current pack and request without starting a worker.

const std = @import("std");
const tool = @import("../tool/tool.zig");
const request = @import("../tool/request.zig");
const Request = request.Request;
pub const Capability = request.Capability;
const config = @import("../config.zig");
const ToolConfig = config.ToolConfig;

pub const Needs = tool.Needs;
pub const Job = tool.Job;
pub const Sink = tool.Sink;
pub const Output = tool.Output;

/// Raised by engines for a request they refuse (wrong size for the model, missing reference): the API's 400.
pub const Refused = error{Refused};

pub const Engine = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        load: *const fn (ptr: *anyopaque) anyerror!u64,
        generate: *const fn (ptr: *anyopaque, job: *const Job, sink: Sink, arena: std.mem.Allocator) anyerror!Output,
        unload: *const fn (ptr: *anyopaque) void,
        port: ?*const fn (ptr: *anyopaque) u16 = null, // an engine that runs its own HTTP server: its loopback port
    };

    /// Loads weights and kernels; returns the resident bytes it now holds.
    pub fn load(e: Engine) anyerror!u64 {
        return e.vtable.load(e.ptr);
    }
    /// Runs one request into `job.dir`; result strings live in `arena`.
    pub fn generate(e: Engine, job: *const Job, sink: Sink, arena: std.mem.Allocator) anyerror!Output {
        return e.vtable.generate(e.ptr, job, sink, arena);
    }
    /// The loopback port of the engine's own HTTP server (named in the load reply), 0 for none.
    pub fn port(e: Engine) u16 {
        return if (e.vtable.port) |f| f(e.ptr) else 0;
    }
    /// Frees everything `load` took and the engine itself.
    pub fn unload(e: Engine) void {
        e.vtable.unload(e.ptr);
    }
};

/// What an engine needs at run time besides its config.
pub const Env = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    environ: ?*const std.process.Environ.Map = null, // the worker's environment (LOCALROUTER_GPU_LOCK, ...)
};

pub const Entry = struct {
    name: []const u8,
    capabilities: []const Capability, // what the engine can do; a tool's `kind` picks its share
    /// Whether a capability the engine declares works with this tool's weights (an optional pack that may be absent).
    /// Called once per tool when the daemon starts (`resolve`), never per request.
    available: ?*const fn (io: std.Io, cfg: *const ToolConfig, cap: Capability) bool = null,
    needs: *const fn (cfg: *const ToolConfig, req: *const Request) Needs,
    create: *const fn (env: Env, cfg: *const ToolConfig) anyerror!Engine,
    /// Refuses a request this model cannot serve (a step count its schedule lacks, a size past its buffers) before it
    /// is queued or loaded, writing why to `why`. Pure and cheap, like `needs`. Null: whatever `request.validate` passes.
    check: ?*const fn (cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool = null,
    /// Fills what the request left to the model (a video's size, 0 x 0) with this model's own choice. See `fill`.
    defaults: ?*const fn (cfg: *const ToolConfig, req: *Request) void = null,
    /// The licence of the weights this engine runs, for a tool whose config names none. Null: unknown.
    license: ?config.License = null,
};

/// The tool's licence: its config's, else its engine's; null when neither states one.
pub fn license(cfg: *const ToolConfig) ?config.License {
    if (cfg.license) |l| return l;
    return if (find(cfg.engine)) |e| e.license else null;
}

/// The size a video request without one gets when its engine has no `defaults`.
pub const default_video_size = [2]u32{ 1344, 768 };

/// Fills what `req` left to the model, before validation: the engine's `defaults`, then `default_video_size`.
pub fn fill(cfg: *const ToolConfig, req: *Request) void {
    if (find(cfg.engine)) |e| if (e.defaults) |f| f(cfg, req);
    switch (req.*) {
        .video => |*v| if (v.width == 0 and v.height == 0) {
            v.width = default_video_size[0];
            v.height = default_video_size[1];
        },
        .image, .speech, .transcription, .chat, .decision => {},
    }
}

pub const entries = [_]Entry{
    @import("../engines/testpattern.zig").entry,
    @import("../engines/qwen_image_tool.zig").entry,
    @import("../engines/qwen_image_tool.zig").turbo_entry,
    @import("../engines/minimax_h3_tool.zig").entry,
    @import("../engines/kandinsky6_tool.zig").entry,
    @import("../engines/openai_server.zig").entry,
};

/// Whether the tool can serve `cap`: its configured capabilities, else its engine's, else its kind's plain one (text to
/// image, text to video, text to speech, speech to text, chat completion, typed decision); always within the tool's kind.
pub fn supports(cfg: *const ToolConfig, cap: Capability) bool {
    if (cap.kind() != cfg.kind) return false;
    const base: []const Capability = if (cfg.capabilities.len > 0) cfg.capabilities else if (find(cfg.engine)) |e| e.capabilities else &.{ .text_to_image, .text_to_video, .text_to_speech, .speech_to_text, .chat_completion, .typed_decision };
    return std.mem.indexOfScalar(Capability, base, cap) != null;
}

/// The tool's capabilities (see `supports`), allocated in `a`.
pub fn capabilityList(a: std.mem.Allocator, cfg: *const ToolConfig) ![]const Capability {
    var list: std.ArrayList(Capability) = .empty;
    for (std.enums.values(Capability)) |c| if (supports(cfg, c)) try list.append(a, c);
    return list.items;
}

/// Narrows each tool's capabilities to what its weights can do (the engine's `available`), once at start: an engine
/// capability whose optional weights are missing or stale is dropped, even when the config lists it, so `/v1/models`,
/// `/v1/capabilities` and the request checks never offer it.
pub fn resolve(io: std.Io, a: std.mem.Allocator, tools: []ToolConfig) !void {
    for (tools) |*t| {
        const e = find(t.engine) orelse continue;
        const f = e.available orelse continue;
        const declared = try capabilityList(a, t);
        var kept: std.ArrayList(Capability) = .empty;
        for (declared) |c| if (f(io, t, c)) try kept.append(a, c) else std.log.info("tool '{s}': {s} is not available with its weights (see its engine's docs)", .{ t.id, @tagName(c) });
        t.capabilities = kept.items;
    }
}

/// Whether the tool takes `req`: for speech its options (`checkSpeech`), then its engine's `Entry.check`; a `cmd` tool
/// takes anything else that is valid.
pub fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    if (req.* == .speech and !checkSpeech(cfg, req.speech, why)) return false;
    if (req.* == .transcription and !checkTranscription(cfg, req.transcription, why)) return false;
    const e = find(cfg.engine) orelse return true;
    const f = e.check orelse return true;
    return f(cfg, req, why);
}

/// A transcription tool's own limits, from its options: `languages` (the ISO-639-1 codes it hears; absent: any), so a
/// language it does not know is refused before it loads, not heard as another. An empty language (detect it) is taken.
pub fn checkTranscription(cfg: *const ToolConfig, v: request.Transcription, why: *std.Io.Writer) bool {
    if (v.language.len == 0) return true;
    const list = optionList(cfg, "languages") orelse return true;
    if (listed(list, v.language)) return true;
    why.print("model '{s}' does not hear language '{s}'; its languages: ", .{ cfg.id, v.language }) catch {};
    printList(list, why);
    return false;
}

/// A speech tool's own limits, from its options, checked before it loads (so nothing is resampled or ignored quietly):
/// `max_input_chars` (at most 4096), `min_speed` and `max_speed` (1 and 1 for a model without a speed control),
/// `voices` (the ids it has; `default_voice` fills an empty voice, see `speechDefaults`), `presets` (the model-native
/// styles a request's `preset` may name; none listed: no preset is taken), and `stream` and `references` (true: the
/// tool streams its audio as it writes it, and takes voice references; absent: it does not).
pub fn checkSpeech(cfg: *const ToolConfig, v: request.Speech, why: *std.Io.Writer) bool {
    const max_chars = optionInt(cfg, "max_input_chars", 4096);
    if ((std.unicode.utf8CountCodepoints(v.input) catch v.input.len) > max_chars) {
        why.print("model '{s}' takes at most {d} characters of input", .{ cfg.id, max_chars }) catch {};
        return false;
    }
    const lo = optionFloat(cfg, "min_speed", 0.25);
    const hi = optionFloat(cfg, "max_speed", 4);
    if (!(v.speed >= lo and v.speed <= hi)) {
        if (lo == hi) why.print("model '{s}' takes only speed {d}", .{ cfg.id, lo }) catch {} else why.print("model '{s}' takes speed {d} to {d}", .{ cfg.id, lo, hi }) catch {};
        return false;
    }
    if (v.voice_references.len > 0 and !supports(cfg, .voice_clone)) {
        why.print("model '{s}' has no cloned voices", .{cfg.id}) catch {};
        return false;
    }
    if (v.voice.len > 0 and v.voice_references.len == 0) if (voices(cfg)) |list| if (!listed(list, v.voice)) {
        why.print("model '{s}' has no voice '{s}'; its voices: ", .{ cfg.id, v.voice }) catch {};
        printList(list, why);
        return false;
    };
    if (v.preset.len > 0) {
        const list = optionList(cfg, "presets") orelse &.{};
        if (!listed(list, v.preset)) {
            why.print("model '{s}' has no preset '{s}'; its presets: ", .{ cfg.id, v.preset }) catch {};
            if (list.len == 0) why.writeAll("none") catch {} else printList(list, why);
            return false;
        }
    }
    if (v.stream and !optionBool(cfg, "stream", false)) {
        why.print("model '{s}' does not stream its audio: leave stream unset for the whole reply", .{cfg.id}) catch {};
        return false;
    }
    if (v.references.len > 0 and !optionBool(cfg, "references", false)) {
        why.print("model '{s}' takes no voice references", .{cfg.id}) catch {};
        return false;
    }
    return true;
}

/// What a speech or transcription tool takes, for the model lists (`/v1/models`, MCP list_models); null fields are
/// not shown.
pub const AudioInfo = struct {
    voices: ?[]const std.json.Value = null,
    default_voice: ?[]const u8 = null,
    presets: ?[]const std.json.Value = null,
    speed: ?[2]f32 = null,
    max_input_chars: ?u64 = null,
    stream: ?bool = null,
    references: ?bool = null,
    instructions: ?bool = null,
    max_audio_s: ?u64 = null,
    languages: ?[]const std.json.Value = null,
};

pub fn audioInfo(cfg: *const ToolConfig) AudioInfo {
    return switch (cfg.kind) {
        .speech => .{
            .voices = voices(cfg),
            .default_voice = if (optionString(cfg, "default_voice", "").len > 0) optionString(cfg, "default_voice", "") else null,
            .presets = optionList(cfg, "presets") orelse &.{},
            .speed = .{ optionFloat(cfg, "min_speed", 0.25), optionFloat(cfg, "max_speed", 4) },
            .max_input_chars = optionInt(cfg, "max_input_chars", 4096),
            .stream = optionBool(cfg, "stream", false),
            .references = optionBool(cfg, "references", false),
            .instructions = optionString(cfg, "instructions", "").len > 0,
        },
        .transcription => .{ .max_audio_s = @min(optionInt(cfg, "max_audio_s", 30 * 60), 30 * 60), .languages = optionList(cfg, "languages") },
        else => .{},
    };
}

/// `v` with the tool's `default_voice` for an empty voice: the worker always gets the voice it will use.
pub fn speechDefaults(cfg: *const ToolConfig, v: request.Speech) request.Speech {
    var out = v;
    if (out.voice.len == 0 and out.clone.len == 0) out.voice = optionString(cfg, "default_voice", "");
    return out;
}

/// The longest `instructions` a speech request may carry, in characters.
pub const max_instructions_chars = 500;

/// `input` with a speech request's `instructions` (trimmed; blank: `input` as is) written in, in `a`, as the tool's
/// `instructions` option says: a template whose one `{s}` is the instructions, such as Drama 3's `"[{s}] "` (a
/// bracketed direction before the line). It goes after the input's leading white space and `<|...|>` tags (a speaker
/// tag stays first). Null with `why` written when the tool takes none, the instructions are longer than
/// `max_instructions_chars`, hold a control character, `<|` or a character of the template (which would end the
/// direction early), or the text with them is longer than `max_input_chars`.
pub fn speechInstructions(a: std.mem.Allocator, cfg: *const ToolConfig, input: []const u8, instructions: []const u8, why: *std.Io.Writer) ?[]const u8 {
    const said = std.mem.trim(u8, instructions, " \t\r\n");
    if (said.len == 0) return input;
    const template = optionString(cfg, "instructions", "");
    const hole = std.mem.indexOf(u8, template, "{s}") orelse {
        why.print("model '{s}' does not take instructions; send the text to speak as input", .{cfg.id}) catch {};
        return null;
    };
    if ((std.unicode.utf8CountCodepoints(said) catch said.len) > max_instructions_chars) {
        why.print("instructions take at most {d} characters", .{max_instructions_chars}) catch {};
        return null;
    }
    for (said) |c| if (c < 0x20 or c == 0x7f) {
        why.writeAll("instructions may not hold a line break or control character") catch {};
        return null;
    };
    if (std.mem.indexOf(u8, said, "<|") != null) {
        why.writeAll("instructions may not hold '<|' (the model's tag syntax)") catch {};
        return null;
    }
    const outside = [_][]const u8{ template[0..hole], template[hole + 3 ..] };
    for (outside) |part| for (part) |c| if (c != ' ' and std.mem.indexOfScalar(u8, said, c) != null) {
        why.print("instructions may not hold '{c}': model '{s}' writes them into the text as {s}", .{ c, cfg.id, template }) catch {};
        return null;
    };
    var at = input.len - std.mem.trimStart(u8, input, " \t\r\n").len;
    while (std.mem.startsWith(u8, input[at..], "<|")) {
        const end = std.mem.indexOfPos(u8, input, at + 2, "|>") orelse break;
        at = end + 2;
    }
    const out = std.mem.concat(a, u8, &.{ input[0..at], outside[0], said, outside[1], input[at..] }) catch {
        why.writeAll("out of memory") catch {};
        return null;
    };
    const max_chars = optionInt(cfg, "max_input_chars", 4096);
    if ((std.unicode.utf8CountCodepoints(out) catch out.len) > max_chars) {
        why.print("model '{s}' takes at most {d} characters of input, the instructions written in included", .{ cfg.id, max_chars }) catch {};
        return null;
    }
    return out;
}

fn listed(list: []const std.json.Value, s: []const u8) bool {
    for (list) |x| if (x == .string and std.mem.eql(u8, x.string, s)) return true;
    return false;
}

fn printList(list: []const std.json.Value, why: *std.Io.Writer) void {
    var first = true;
    for (list) |x| if (x == .string) {
        why.print("{s}{s}", .{ if (first) "" else ", ", x.string }) catch {};
        first = false;
    };
}

/// Whether each configured default (`Config.defaults`) names a tool that has its capability; the first that does not
/// is written to `why`.
pub fn checkDefaults(c: *const config.Config, why: *std.Io.Writer) bool {
    for (c.tools) |*t| {
        if (t.kind != .speech) continue;
        const dv = optionString(t, "default_voice", "");
        if (dv.len > 0) if (voices(t)) |list| if (!listed(list, dv)) {
            why.print("tool '{s}': default_voice '{s}' is not one of its voices", .{ t.id, dv }) catch {};
            return false;
        };
        if (optionInt(t, "max_input_chars", 4096) > 4096 or optionFloat(t, "min_speed", 0.25) > optionFloat(t, "max_speed", 4)) {
            why.print("tool '{s}': max_input_chars must be at most 4096 and min_speed at most max_speed", .{t.id}) catch {};
            return false;
        }
        const given = if (t.options) |o| (if (o == .object) o.object.get("instructions") else null) else null;
        if (given) |g| if (g != .string or std.mem.count(u8, g.string, "{s}") != 1) {
            why.print("tool '{s}': instructions must be a template with one {{s}}, such as \"[{{s}}] \"", .{t.id}) catch {};
            return false;
        };
    }
    for (std.enums.values(Capability)) |cap| {
        const id = c.defaults.get(cap) orelse continue;
        const t = c.tool(id) orelse {
            why.print("defaults.{s} names '{s}', which is not a configured tool", .{ @tagName(cap), id }) catch {};
            return false;
        };
        if (!supports(t, cap)) {
            why.print("defaults.{s} names '{s}', which cannot do {s}", .{ @tagName(cap), id, @tagName(cap) }) catch {};
            return false;
        }
    }
    return true;
}

test "defaults must name a tool that has the capability" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const tools =
        \\"tools": [{"id": "img", "kind": "image", "engine": "testpattern", "capabilities": ["text_to_image"]},
        \\          {"id": "vid", "kind": "video", "engine": "testpattern"}]
    ;
    var buf: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    try std.testing.expect(checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"text_to_image\": \"img\", \"image_to_video\": \"vid\"}," ++ tools ++ "}"), &why));
    try std.testing.expect(!checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"image_edit\": \"img\"}," ++ tools ++ "}"), &why));
    try std.testing.expectEqualStrings("defaults.image_edit names 'img', which cannot do image_edit", why.buffered());
    why = .fixed(&buf);
    try std.testing.expect(!checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"text_to_image\": \"vid\"}," ++ tools ++ "}"), &why));
    why = .fixed(&buf);
    try std.testing.expect(!checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"text_to_image\": \"nope\"}," ++ tools ++ "}"), &why));
    try std.testing.expectEqualStrings("defaults.text_to_image names 'nope', which is not a configured tool", why.buffered());
}

pub fn find(name: []const u8) ?*const Entry {
    for (&entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// A u64 option from the tool's `options` object, else `default`.
pub fn optionInt(cfg: *const ToolConfig, key: []const u8, default: u64) u64 {
    const opts = cfg.options orelse return default;
    if (opts != .object) return default;
    const v = opts.object.get(key) orelse return default;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else default,
        else => default,
    };
}

/// A speech tool's `voices` option (voice ids), else null.
pub fn voices(cfg: *const ToolConfig) ?[]const std.json.Value {
    return optionList(cfg, "voices");
}

/// Whether the tool lists `voice` among its `voices`. A tool that lists none takes any voice, but is never chosen by one.
pub fn hasVoice(cfg: *const ToolConfig, voice: []const u8) bool {
    return listed(voices(cfg) orelse return false, voice);
}

/// An array option from the tool's `options` object, else null.
pub fn optionList(cfg: *const ToolConfig, key: []const u8) ?[]const std.json.Value {
    const opts = cfg.options orelse return null;
    if (opts != .object) return null;
    const v = opts.object.get(key) orelse return null;
    return if (v == .array) v.array.items else null;
}

/// A number option from the tool's `options` object, else `default`.
pub fn optionFloat(cfg: *const ToolConfig, key: []const u8, default: f32) f32 {
    const opts = cfg.options orelse return default;
    if (opts != .object) return default;
    const v = opts.object.get(key) orelse return default;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => default,
    };
}

test "a speech tool's voices are checked before it loads; cmd tools get their kind's plain capability" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "tts", "kind": "speech", "cmd": ["x"], "options": {"voices": ["a", "b"]}},
        \\           {"id": "stt", "kind": "transcription", "cmd": ["y"]}]}
    );
    const tts = c.tool("tts").?;
    try std.testing.expect(supports(tts, .text_to_speech));
    try std.testing.expect(!supports(tts, .speech_to_text));
    try std.testing.expect(supports(c.tool("stt").?, .speech_to_text));
    try std.testing.expect(!supports(c.tool("stt").?, .text_to_image));
    var buf: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    try std.testing.expect(check(tts, &.{ .speech = .{ .input = "hi", .voice = "b" } }, &why));
    try std.testing.expect(check(tts, &.{ .speech = .{ .input = "hi" } }, &why));
    try std.testing.expect(!check(tts, &.{ .speech = .{ .input = "hi", .voice = "c" } }, &why));
    try std.testing.expectEqualStrings("model 'tts' has no voice 'c'; its voices: a, b", why.buffered());
    try std.testing.expect(hasVoice(tts, "a") and !hasVoice(tts, "c") and !hasVoice(c.tool("stt").?, "a"));

    const k = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "kitten", "kind": "speech", "cmd": ["x"], "options": {"voices": ["Bruno", "Rosie"], "default_voice": "Bruno",
        \\  "max_input_chars": 10, "min_speed": 1, "max_speed": 1, "presets": ["stable", "expressive"]}}]}
    );
    const kt = k.tool("kitten").?;
    const refusals = [_]struct { v: request.Speech, msg: []const u8 }{
        .{ .v = .{ .input = "eleven char" }, .msg = "model 'kitten' takes at most 10 characters of input" },
        .{ .v = .{ .input = "hi", .speed = 1.5 }, .msg = "model 'kitten' takes only speed 1" },
        .{ .v = .{ .input = "hi", .preset = "whisper" }, .msg = "model 'kitten' has no preset 'whisper'; its presets: stable, expressive" },
    };
    for (refusals) |r| {
        why = .fixed(&buf);
        try std.testing.expect(!check(kt, &.{ .speech = r.v }, &why));
        try std.testing.expectEqualStrings(r.msg, why.buffered());
    }
    try std.testing.expect(check(kt, &.{ .speech = .{ .input = "ten chars!", .preset = "expressive" } }, &why));
    try std.testing.expectEqualStrings("Bruno", speechDefaults(kt, .{ .input = "hi" }).voice);
    try std.testing.expectEqualStrings("Rosie", speechDefaults(kt, .{ .input = "hi", .voice = "Rosie" }).voice);
    why = .fixed(&buf);
    try std.testing.expect(!check(tts, &.{ .speech = .{ .input = "hi", .preset = "stable" } }, &why));
    try std.testing.expectEqualStrings("model 'tts' has no preset 'stable'; its presets: none", why.buffered());
    why = .fixed(&buf);
    try std.testing.expect(checkDefaults(&k, &why));
    const bad = try config.parse(arena.allocator(), "{\"tools\": [{\"id\": \"t\", \"kind\": \"speech\", \"cmd\": [\"x\"], \"options\": {\"voices\": [\"a\"], \"default_voice\": \"b\"}}]}");
    try std.testing.expect(!checkDefaults(&bad, &why));
    try std.testing.expectEqualStrings("tool 't': default_voice 'b' is not one of its voices", why.buffered());
}

test "a transcription model's declared languages: listed, and a language outside them refused before it loads" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "ears", "kind": "transcription", "cmd": ["x"], "options": {"languages": ["en", "de"]}},
        \\           {"id": "any", "kind": "transcription", "cmd": ["x"]}]}
    );
    const ears = c.tool("ears").?;
    try std.testing.expectEqual(@as(usize, 2), audioInfo(ears).languages.?.len);
    try std.testing.expect(audioInfo(c.tool("any").?).languages == null);
    var buf: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    for ([_][]const u8{ "", "en", "de" }) |l| try std.testing.expect(check(ears, &.{ .transcription = .{ .audio = "a.wav", .language = l } }, &why));
    try std.testing.expect(check(c.tool("any").?, &.{ .transcription = .{ .audio = "a.wav", .language = "ja" } }, &why));
    try std.testing.expect(!check(ears, &.{ .transcription = .{ .audio = "a.wav", .language = "ja" } }, &why));
    try std.testing.expectEqualStrings("model 'ears' does not hear language 'ja'; its languages: en, de", why.buffered());
}

/// A boolean option from the tool's `options` object, else `default`.
pub fn optionBool(cfg: *const ToolConfig, key: []const u8, default: bool) bool {
    const opts = cfg.options orelse return default;
    if (opts != .object) return default;
    const v = opts.object.get(key) orelse return default;
    return if (v == .bool) v.bool else default;
}

/// Why a tool gated by a `qualification` file may not run on this machine, written in `buf`; null when it may (no gate,
/// or the file holds `"qualified": true`). The file is the verdict of the model's qualify script on this host's GPU.
pub fn unqualified(io: std.Io, a: std.mem.Allocator, cfg: *const ToolConfig, buf: []u8) ?[]const u8 {
    const path = cfg.qualification;
    if (path.len == 0) return null;
    const problem: []const u8 = if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20))) |text| blk: {
        const Verdict = struct { qualified: bool = false };
        const v = std.json.parseFromSliceLeaky(Verdict, a, text, .{ .ignore_unknown_fields = true }) catch break :blk "does not say \"qualified\": true";
        if (v.qualified) return null;
        break :blk "does not say \"qualified\": true";
    } else |_| "does not exist";
    return std.fmt.bufPrint(buf, "model '{s}' is not GPU-qualified on this host: {s} {s}; an operator runs the model's qualify script on this machine first", .{ cfg.id, path, problem }) catch "the model is not GPU-qualified on this host";
}

/// Why the last `load` or `generate` in this worker failed, in words (an engine sets it with `explain`; the worker
/// sends it in its failure reply and clears it). Workers are single-threaded.
var explained_buf: [512]u8 = undefined;
var explained_len: usize = 0;

pub fn explain(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(&explained_buf, fmt, args) catch {
        explained_len = explained_buf.len; // cut at the buffer's end
        return;
    };
    explained_len = s.len;
}

/// The reason `explain` recorded ("" for none), cleared for the next call.
pub fn explained() []const u8 {
    defer explained_len = 0;
    return explained_buf[0..explained_len];
}

/// A string option from the tool's `options` object, else `default`.
pub fn optionString(cfg: *const ToolConfig, key: []const u8, default: []const u8) []const u8 {
    const opts = cfg.options orelse return default;
    if (opts != .object) return default;
    const v = opts.object.get(key) orelse return default;
    return if (v == .string) v.string else default;
}

test "a qualification file gates a tool until it says qualified" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const path = "/tmp/localrouter-unit-qualification-chat.json";
    const c = try config.parse(a,
        \\{"gpu_lock": "/var/lock/gpu.lock", "tools": [
        \\  {"id": "gated", "kind": "chat", "engine": "openai_server", "qualification": "/tmp/localrouter-unit-qualification-chat.json"},
        \\  {"id": "plain", "kind": "chat", "cmd": ["y"]}]}
    );
    try std.testing.expectEqualStrings("/var/lock/gpu.lock", c.gpu_lock);
    const gated = c.tool("gated").?;
    var buf: [512]u8 = undefined;
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    try std.testing.expectEqualStrings("model 'gated' is not GPU-qualified on this host: " ++ path ++ " does not exist; an operator runs the model's qualify script on this machine first", unqualified(io, a, gated, &buf).?);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "{\"qualified\": false}" });
    try std.testing.expect(std.mem.indexOf(u8, unqualified(io, a, gated, &buf).?, path ++ " does not say \"qualified\": true") != null);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "not json" });
    try std.testing.expect(unqualified(io, a, gated, &buf) != null);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "{\"qualified\": true, \"host\": \"gb10\"}" });
    try std.testing.expect(unqualified(io, a, gated, &buf) == null);
    try std.testing.expect(unqualified(io, a, c.tool("plain").?, &buf) == null);
    try std.testing.expect(supports(gated, .chat_completion));
    try std.testing.expect(supports(c.tool("plain").?, .chat_completion)); // a `cmd` chat tool: its kind's capability
    try std.testing.expect(!supports(gated, .text_to_image));
    explain("the GPU lock {s} is held by another job", .{"/x"});
    try std.testing.expectEqualStrings("the GPU lock /x is held by another job", explained());
    try std.testing.expectEqualStrings("", explained());
}

test "stream and references need the speech tool to offer them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "gated", "kind": "speech", "cmd": ["x"], "options": {"stream": true}},
        \\           {"id": "plain", "kind": "speech", "cmd": ["y"]}]}
    );
    const gated = c.tool("gated").?;
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try std.testing.expect(check(gated, &.{ .speech = .{ .input = "hi", .stream = true } }, &w));
    try std.testing.expect(!check(c.tool("plain").?, &.{ .speech = .{ .input = "hi", .stream = true } }, &w));
    try std.testing.expectEqualStrings("model 'plain' does not stream its audio: leave stream unset for the whole reply", w.buffered());
    w = .fixed(&buf);
    try std.testing.expect(!check(gated, &.{ .speech = .{ .input = "hi", .references = "references.json" } }, &w));
    try std.testing.expectEqualStrings("model 'gated' takes no voice references", w.buffered());
    try std.testing.expectEqual(@as(?bool, true), audioInfo(gated).stream);
    try std.testing.expectEqual(@as(?bool, false), audioInfo(c.tool("plain").?).references);
}

test "instructions are written into the text by a tool's template, after a leading tag; refused without one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try config.parse(a,
        \\{"tools": [{"id": "styled", "kind": "speech", "cmd": ["x"], "options": {"instructions": "[{s}] ", "max_input_chars": 26}},
        \\           {"id": "plain", "kind": "speech", "cmd": ["y"]}]}
    );
    const styled = c.tool("styled").?;
    const plain = c.tool("plain").?;
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try std.testing.expectEqualStrings("[calmly] Hello.", speechInstructions(a, styled, "Hello.", " calmly\n", &w).?);
    try std.testing.expectEqualStrings(" <|speaker:1|>[calmly] Hi.", speechInstructions(a, styled, " <|speaker:1|>Hi.", "calmly", &w).?);
    try std.testing.expectEqualStrings("Hi.", speechInstructions(a, plain, "Hi.", "  ", &w).?);
    try std.testing.expectEqual(@as(?bool, true), audioInfo(styled).instructions);
    try std.testing.expectEqual(@as(?bool, false), audioInfo(plain).instructions);
    const long: [max_instructions_chars + 1]u8 = @splat('x');
    const refusals = [_]struct { tool: *const ToolConfig, said: []const u8, msg: []const u8 }{
        .{ .tool = plain, .said = "calmly", .msg = "model 'plain' does not take instructions; send the text to speak as input" },
        .{ .tool = styled, .said = "calm] loud", .msg = "instructions may not hold ']': model 'styled' writes them into the text as [{s}] " },
        .{ .tool = styled, .said = "calm\nloud", .msg = "instructions may not hold a line break or control character" },
        .{ .tool = styled, .said = "a <|speaker:2|>", .msg = "instructions may not hold '<|' (the model's tag syntax)" },
        .{ .tool = styled, .said = &long, .msg = "instructions take at most 500 characters" },
        .{ .tool = styled, .said = "very very very slowly", .msg = "model 'styled' takes at most 26 characters of input, the instructions written in included" },
    };
    for (refusals) |r| {
        w = .fixed(&buf);
        try std.testing.expect(speechInstructions(a, r.tool, "Hello.", r.said, &w) == null);
        try std.testing.expectEqualStrings(r.msg, w.buffered());
    }
    w = .fixed(&buf);
    try std.testing.expect(checkDefaults(&c, &w));
    const bad = try config.parse(a, "{\"tools\": [{\"id\": \"t\", \"kind\": \"speech\", \"cmd\": [\"x\"], \"options\": {\"instructions\": \"[style] \"}}]}");
    try std.testing.expect(!checkDefaults(&bad, &w));
    try std.testing.expectEqualStrings("tool 't': instructions must be a template with one {s}, such as \"[{s}] \"", w.buffered());
}

test "a tool's licence is its config's, else its engine's, else none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try config.parse(arena.allocator(),
        \\{"tools": [{"id": "own", "kind": "video", "engine": "minimax_h3", "license": {"name": "Custom", "commercial": false}},
        \\           {"id": "h3", "kind": "video", "engine": "minimax_h3"},
        \\           {"id": "turbo", "kind": "image", "engine": "qwen_image_turbo"},
        \\           {"id": "cmd", "kind": "speech", "cmd": ["x"]}]}
    );
    const own = license(c.tool("own").?).?;
    try std.testing.expectEqualStrings("Custom", own.name);
    try std.testing.expectEqual(@as(?bool, false), own.commercial);
    try std.testing.expectEqual(@as(usize, 0), own.excluded_regions.len);
    const h3 = license(c.tool("h3").?).?;
    try std.testing.expectEqualStrings("MiniMax H3 Community License Agreement", h3.name);
    try std.testing.expectEqual(@as(usize, 4), h3.excluded_regions.len);
    try std.testing.expect(h3.notice.len > 0);
    try std.testing.expectEqualStrings("Qwen Research License Agreement", license(c.tool("turbo").?).?.name);
    try std.testing.expect(license(c.tool("cmd").?) == null);
}

test "resolve drops an engine capability its weights lack, even when the config lists it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var tools = [_]ToolConfig{.{ .id = "edit-test", .kind = .image, .engine = "qwen_image", .weights = "/nonexistent/localrouter-edit-test", .capabilities = &.{ .text_to_image, .image_edit } }};
    try resolve(std.testing.io, arena.allocator(), &tools);
    try std.testing.expect(supports(&tools[0], .text_to_image));
    try std.testing.expect(!supports(&tools[0], .image_edit));
}
