//! The worker-side contract every compiled-in engine implements, and the table of engines in this binary.
//! `needs` is a pure function of the config and request, so the daemon prices a request without starting a worker.

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
    };

    /// Loads weights and kernels; returns the resident bytes it now holds.
    pub fn load(e: Engine) anyerror!u64 {
        return e.vtable.load(e.ptr);
    }
    /// Runs one request into `job.dir`; result strings live in `arena`.
    pub fn generate(e: Engine, job: *const Job, sink: Sink, arena: std.mem.Allocator) anyerror!Output {
        return e.vtable.generate(e.ptr, job, sink, arena);
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
};

pub const Entry = struct {
    name: []const u8,
    capabilities: []const Capability, // what the engine can do; a tool's `kind` picks its share
    needs: *const fn (cfg: *const ToolConfig, req: *const Request) Needs,
    create: *const fn (env: Env, cfg: *const ToolConfig) anyerror!Engine,
    /// Refuses a request this model cannot serve (a step count its schedule lacks, a size past its buffers) before it
    /// is queued or loaded, writing why to `why`. Pure and cheap, like `needs`. Null: whatever `request.validate` passes.
    check: ?*const fn (cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool = null,
};

pub const entries = [_]Entry{
    @import("../engines/testpattern.zig").entry,
    @import("../engines/qwen_image_tool.zig").entry,
    @import("../engines/qwen_image_tool.zig").turbo_entry,
    @import("../engines/minimax_h3_tool.zig").entry,
};

/// Whether the tool can serve `cap`: its configured capabilities, else its engine's, else its kind's plain one (text to
/// image, text to video, text to speech, speech to text); always within the tool's kind.
pub fn supports(cfg: *const ToolConfig, cap: Capability) bool {
    if (cap.kind() != cfg.kind) return false;
    const base: []const Capability = if (cfg.capabilities.len > 0) cfg.capabilities else if (find(cfg.engine)) |e| e.capabilities else &.{ .text_to_image, .text_to_video, .text_to_speech, .speech_to_text };
    return std.mem.indexOfScalar(Capability, base, cap) != null;
}

/// The tool's capabilities (see `supports`), allocated in `a`.
pub fn capabilityList(a: std.mem.Allocator, cfg: *const ToolConfig) ![]const Capability {
    var list: std.ArrayList(Capability) = .empty;
    for (std.enums.values(Capability)) |c| if (supports(cfg, c)) try list.append(a, c);
    return list.items;
}

/// Whether the tool takes `req`: for speech its options (`checkSpeech`), then its engine's `Entry.check`; a `cmd` tool
/// takes anything else that is valid.
pub fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    if (req.* == .speech and !checkSpeech(cfg, req.speech, why)) return false;
    const e = find(cfg.engine) orelse return true;
    const f = e.check orelse return true;
    return f(cfg, req, why);
}

/// A speech tool's own limits, from its options, checked before it loads (so nothing is resampled or ignored quietly):
/// `max_input_chars` (at most 4096), `min_speed` and `max_speed` (1 and 1 for a model without a speed control),
/// `voices` (the ids it has; `default_voice` fills an empty voice, see `speechDefaults`) and `presets` (the
/// model-native styles a request's `preset` may name; none listed: no preset is taken).
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
    if (v.voice.len > 0) if (voices(cfg)) |list| if (!listed(list, v.voice)) {
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
    max_audio_s: ?u64 = null,
};

pub fn audioInfo(cfg: *const ToolConfig) AudioInfo {
    return switch (cfg.kind) {
        .speech => .{
            .voices = voices(cfg),
            .default_voice = if (optionString(cfg, "default_voice", "").len > 0) optionString(cfg, "default_voice", "") else null,
            .presets = optionList(cfg, "presets") orelse &.{},
            .speed = .{ optionFloat(cfg, "min_speed", 0.25), optionFloat(cfg, "max_speed", 4) },
            .max_input_chars = optionInt(cfg, "max_input_chars", 4096),
        },
        .transcription => .{ .max_audio_s = @min(optionInt(cfg, "max_audio_s", 30 * 60), 30 * 60) },
        else => .{},
    };
}

/// `v` with the tool's `default_voice` for an empty voice: the worker always gets the voice it will use.
pub fn speechDefaults(cfg: *const ToolConfig, v: request.Speech) request.Speech {
    var out = v;
    if (out.voice.len == 0) out.voice = optionString(cfg, "default_voice", "");
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

/// A string option from the tool's `options` object, else `default`.
pub fn optionString(cfg: *const ToolConfig, key: []const u8, default: []const u8) []const u8 {
    const opts = cfg.options orelse return default;
    if (opts != .object) return default;
    const v = opts.object.get(key) orelse return default;
    return if (v == .string) v.string else default;
}
