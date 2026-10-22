//! What a tool is asked to make: one validated request per kind. New kinds add an arm, so every switch must handle it.

const std = @import("std");

pub const Kind = enum { image, video, speech, transcription, chat };

/// What a model can do; a request needs exactly one (`Request.capability`).
pub const Capability = enum {
    text_to_image,
    image_edit,
    text_to_video,
    image_to_video,
    text_to_speech,
    speech_to_text,
    chat_completion,

    pub fn kind(c: Capability) Kind {
        return switch (c) {
            .text_to_image, .image_edit => .image,
            .text_to_video, .image_to_video => .video,
            .text_to_speech => .speech,
            .speech_to_text => .transcription,
            .chat_completion => .chat,
        };
    }
};

/// Image generation or edit. Sizes are multiples of 16 (the DiTs' latent /16 grid).
pub const Image = struct {
    prompt: []const u8,
    negative_prompt: []const u8 = "",
    width: u32 = 1024,
    height: u32 = 1024,
    n: u32 = 1,
    seed: u64 = 0,
    steps: u32 = 0, // 0: the tool's default
    guidance: f32 = 0, // 0: the tool's default
    references: []const []const u8 = &.{}, // file names in the job directory (edits)
};

/// Video (optionally with audio). `seconds` x `fps` frames.
pub const Video = struct {
    prompt: []const u8,
    negative_prompt: []const u8 = "",
    width: u32 = 0, // 0 (with height 0): the model's own size, filled in by `engine.fill` before validation
    height: u32 = 0,
    seconds: u32 = 5,
    fps: u32 = 24,
    seed: u64 = 0,
    steps: u32 = 0,
    guidance: f32 = 0,
    audio: bool = true,
    first_frame: ?[]const u8 = null, // file name in the job directory (image to video)
};

/// Text to speech. The tool writes `speech.wav`: mono 16-bit PCM at the model's own rate; the API encodes other formats.
pub const Speech = struct {
    input: []const u8,
    voice: []const u8 = "", // "": the tool's default voice (the daemon fills in its `default_voice` when it has one)
    speed: f32 = 1,
    preset: []const u8 = "", // a model-native style the tool lists in its `presets` option; "": none
    seed: u64 = 0,
    stream: bool = false, // the tool writes `speech.wav` as it goes (header first, sizes fixed at the end), for a streamed reply
    references: []const u8 = "", // "references.json" in the job directory: the voice references the request sent; "": none
};

pub const Timestamps = enum { none, segment, word };
/// How a Whisper-style worker decodes: its model card's recipe (temperature fallback), or one greedy pass.
pub const Decode = enum { card, greedy };

/// Speech to text. `audio` names `audio.wav` in the job directory, written by the daemon as 16 kHz mono 16-bit PCM
/// whatever was uploaded (`media/audio.zig`). The tool writes `transcript.json` (`Transcript`).
pub const Transcription = struct {
    audio: []const u8,
    duration_ms: u64 = 0, // of `audio`, measured by the daemon
    language: []const u8 = "", // ISO-639-1; "": detect
    prompt: []const u8 = "", // text that conditions the decoder (spellings, previous context)
    temperature: f32 = 0,
    timestamps: Timestamps = .none,
    seed: u64 = 0,
    decode: ?Decode = null, // null: the worker's default (sent only when set)
};

/// What a speech_to_text tool writes: the text, the language it heard, and segments (words when asked for).
pub const Transcript = struct {
    text: []const u8,
    language: []const u8 = "",
    duration: f64 = 0,
    segments: []const Segment = &.{},
    words: ?[]const Word = null,

    pub const Segment = struct { id: u32 = 0, start: f64, end: f64, text: []const u8 };
    pub const Word = struct { word: []const u8, start: f64, end: f64 };
};

/// A chat completion: `body` is the OpenAI request as the model's own server receives it (`model` set to the tool's id).
/// The daemon proxies it to the server its worker runs and keeps the reply in the job directory: `completion.json`, or
/// `completion.sse` (the server's event stream as it arrives) when `stream` is set.
pub const Chat = struct {
    body: []const u8,
    stream: bool = false,
    seed: u64 = 0, // the request's own seed, if it named one (the server samples with it)
};

/// The wire form is std.json's union shape: {"image": {...}}, {"video": {...}}, {"speech": {...}}, {"transcription": {...}} or {"chat": {...}}.
pub const Request = union(Kind) {
    image: Image,
    video: Video,
    speech: Speech,
    transcription: Transcription,
    chat: Chat,

    pub fn kind(r: Request) Kind {
        return std.meta.activeTag(r);
    }

    /// What a model must be able to do to serve this request: reference images make an edit, a first frame a video from an image.
    pub fn capability(r: Request) Capability {
        return switch (r) {
            .image => |v| if (v.references.len > 0) .image_edit else .text_to_image,
            .video => |v| if (v.first_frame != null) .image_to_video else .text_to_video,
            .speech => .text_to_speech,
            .transcription => .speech_to_text,
            .chat => .chat_completion,
        };
    }

    pub fn seed(r: Request) u64 {
        return switch (r) {
            inline else => |v| v.seed,
        };
    }
};

pub const Limits = struct {
    min_side: u32 = 256,
    max_side: u32 = 2048,
    max_n: u32 = 4,
    max_seconds: u32 = 20,
    max_fps: u32 = 30,
    max_steps: u32 = 200,
    max_prompt: usize = 16 * 1024,
    max_references: usize = 5,
    max_speech_input: usize = 4096, // characters, as OpenAI's speech API
    min_speed: f32 = 0.25,
    max_speed: f32 = 4,
    max_audio_ms: u64 = 30 * 60 * 1000, // one transcription's audio
    max_transcription_prompt: usize = 4096,
    max_chat_body: usize = 16 << 20, // one chat request as sent to the model's server
};

pub const Invalid = error{Invalid};

/// Checks `r` against `l`; on failure `why` names the field, for the API's 400 message.
pub fn validate(r: Request, l: Limits, why: *[]const u8) Invalid!void {
    switch (r) {
        .image => |v| {
            try common(v.prompt, v.width, v.height, v.steps, l, why);
            if (v.n == 0 or v.n > l.max_n) return fail(why, "n must be 1 to 4");
            if (v.references.len > l.max_references) return fail(why, "at most 5 reference images");
        },
        .video => |v| {
            try common(v.prompt, v.width, v.height, v.steps, l, why);
            if (v.seconds == 0 or v.seconds > l.max_seconds) return fail(why, "seconds must be 1 to 20");
            if (v.fps == 0 or v.fps > l.max_fps) return fail(why, "fps must be 1 to 30");
        },
        .speech => |v| {
            if (std.mem.trim(u8, v.input, " \t\r\n").len == 0) return fail(why, "input is required");
            if (!std.unicode.utf8ValidateSlice(v.input)) return fail(why, "input must be UTF-8 text");
            if ((std.unicode.utf8CountCodepoints(v.input) catch v.input.len) > l.max_speech_input) return fail(why, "input must be at most 4096 characters");
            if (!(v.speed >= l.min_speed and v.speed <= l.max_speed)) return fail(why, "speed must be 0.25 to 4");
            if (v.voice.len > 64 or !token(v.voice)) return fail(why, "voice must be a voice id (letters, digits, '-', '_' and '.')");
            if (v.preset.len > 32 or !token(v.preset)) return fail(why, "preset must be a preset id (letters, digits, '-', '_' and '.')");
            if (v.references.len > 0 and !token(v.references)) return fail(why, "references must name a file in the job directory");
        },
        .transcription => |v| {
            if (v.audio.len == 0 or !token(v.audio)) return fail(why, "audio must name a file in the job directory");
            if (v.duration_ms > l.max_audio_ms) return fail(why, "the audio is longer than 30 minutes");
            if (!language(v.language)) return fail(why, "language must be an ISO-639-1 code such as en or de");
            if (v.prompt.len > l.max_transcription_prompt or !std.unicode.utf8ValidateSlice(v.prompt)) return fail(why, "prompt must be UTF-8 text of at most 4096 bytes");
            if (!(v.temperature >= 0 and v.temperature <= 1)) return fail(why, "temperature must be 0 to 1");
        },
        .chat => |v| {
            if (v.body.len == 0) return fail(why, "the chat request is empty");
            if (v.body.len > l.max_chat_body) return fail(why, "the chat request is larger than 16 MB");
        },
    }
}

/// Letters, digits, '-', '_' and '.' (no path separators): voice ids and job file names.
fn token(s: []const u8) bool {
    for (s) |c| if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.')) return false;
    return !std.mem.eql(u8, s, ".") and !std.mem.eql(u8, s, "..");
}

/// "" (detect) or a two- or three-letter lowercase code.
fn language(s: []const u8) bool {
    if (s.len == 0) return true;
    if (s.len < 2 or s.len > 3) return false;
    for (s) |c| if (!std.ascii.isLower(c)) return false;
    return true;
}

fn common(prompt: []const u8, w: u32, h: u32, steps: u32, l: Limits, why: *[]const u8) Invalid!void {
    if (prompt.len == 0) return fail(why, "prompt is required");
    if (prompt.len > l.max_prompt) return fail(why, "prompt is too long");
    if (w < l.min_side or h < l.min_side or w > l.max_side or h > l.max_side) return fail(why, "size sides must be 256 to 2048");
    if (w % 16 != 0 or h % 16 != 0) return fail(why, "size sides must be multiples of 16");
    if (steps > l.max_steps) return fail(why, "steps must be at most 200");
}

fn fail(why: *[]const u8, msg: []const u8) Invalid {
    why.* = msg;
    return error.Invalid;
}

/// "WIDTHxHEIGHT" -> (w, h); null when malformed.
pub fn parseSize(s: []const u8) ?[2]u32 {
    const x = std.mem.indexOfScalar(u8, s, 'x') orelse return null;
    const w = std.fmt.parseInt(u32, s[0..x], 10) catch return null;
    const h = std.fmt.parseInt(u32, s[x + 1 ..], 10) catch return null;
    return .{ w, h };
}

test "validate and wire shape" {
    var why: []const u8 = "";
    const ok: Request = .{ .image = .{ .prompt = "a fox", .width = 1360, .height = 768 } };
    try validate(ok, .{}, &why);
    try std.testing.expectError(error.Invalid, validate(.{ .image = .{ .prompt = "a", .width = 1000 } }, .{}, &why));
    try std.testing.expectEqualStrings("size sides must be multiples of 16", why);
    try std.testing.expectError(error.Invalid, validate(.{ .video = .{ .prompt = "a", .seconds = 0 } }, .{}, &why));
    const text = try std.json.Stringify.valueAlloc(std.testing.allocator, ok, .{});
    defer std.testing.allocator.free(text);
    const back = try std.json.parseFromSlice(Request, std.testing.allocator, text, .{});
    defer back.deinit();
    try std.testing.expectEqual(@as(u32, 1360), back.value.image.width);
    try std.testing.expectEqual([2]u32{ 1360, 768 }, parseSize("1360x768").?);
    try std.testing.expect(parseSize("1360") == null);
}

test "speech and transcription requests: capabilities, limits and wire shape" {
    var why: []const u8 = "";
    const say: Request = .{ .speech = .{ .input = "Hello there.", .voice = "expr-voice-2-m", .speed = 1.5, .seed = 3 } };
    try validate(say, .{}, &why);
    try std.testing.expectEqual(Capability.text_to_speech, say.capability());
    try std.testing.expectEqual(Kind.speech, say.capability().kind());
    try std.testing.expectEqual(@as(u64, 3), say.seed());
    const bad = [_]struct { r: Request, msg: []const u8 }{
        .{ .r = .{ .speech = .{ .input = " \n" } }, .msg = "input is required" },
        .{ .r = .{ .speech = .{ .input = "hi", .speed = 5 } }, .msg = "speed must be 0.25 to 4" },
        .{ .r = .{ .speech = .{ .input = "hi", .speed = std.math.nan(f32) } }, .msg = "speed must be 0.25 to 4" },
        .{ .r = .{ .speech = .{ .input = "hi", .voice = "../x" } }, .msg = "voice must be a voice id (letters, digits, '-', '_' and '.')" },
        .{ .r = .{ .speech = .{ .input = "\xff\xfe" } }, .msg = "input must be UTF-8 text" },
        .{ .r = .{ .transcription = .{ .audio = "../audio.wav" } }, .msg = "audio must name a file in the job directory" },
        .{ .r = .{ .transcription = .{ .audio = "audio.wav", .duration_ms = 31 * 60 * 1000 } }, .msg = "the audio is longer than 30 minutes" },
        .{ .r = .{ .transcription = .{ .audio = "audio.wav", .language = "English" } }, .msg = "language must be an ISO-639-1 code such as en or de" },
        .{ .r = .{ .transcription = .{ .audio = "audio.wav", .temperature = -1 } }, .msg = "temperature must be 0 to 1" },
    };
    for (bad) |b| {
        try std.testing.expectError(error.Invalid, validate(b.r, .{}, &why));
        try std.testing.expectEqualStrings(b.msg, why);
    }
    const long: [4097]u8 = @splat('a');
    try std.testing.expectError(error.Invalid, validate(.{ .speech = .{ .input = &long } }, .{}, &why));
    var runes: [8192]u8 = undefined; // 4096 two-byte characters: within the limit
    for (0..4096) |i| runes[i * 2 ..][0..2].* = "\u{e9}".*;
    try validate(.{ .speech = .{ .input = &runes } }, .{}, &why);

    const hear: Request = .{ .transcription = .{ .audio = "audio.wav", .duration_ms = 1500, .language = "de", .timestamps = .word } };
    try validate(hear, .{}, &why);
    try std.testing.expectEqual(Capability.speech_to_text, hear.capability());
    const text = try std.json.Stringify.valueAlloc(std.testing.allocator, hear, .{});
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "{\"transcription\":{\"audio\":\"audio.wav\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"timestamps\":\"word\"") != null);
    const back = try std.json.parseFromSlice(Request, std.testing.allocator, text, .{});
    defer back.deinit();
    try std.testing.expectEqualStrings("de", back.value.transcription.language);
}

test "chat requests: capability, limits and wire shape" {
    var why: []const u8 = "";
    const ask: Request = .{ .chat = .{ .body = "{\"model\":\"m\",\"messages\":[]}", .stream = true, .seed = 9 } };
    try validate(ask, .{}, &why);
    try std.testing.expectEqual(Capability.chat_completion, ask.capability());
    try std.testing.expectEqual(Kind.chat, ask.capability().kind());
    try std.testing.expectEqual(@as(u64, 9), ask.seed());
    try std.testing.expectError(error.Invalid, validate(.{ .chat = .{ .body = "" } }, .{}, &why));
    try std.testing.expectEqualStrings("the chat request is empty", why);
    try std.testing.expectError(error.Invalid, validate(.{ .chat = .{ .body = "{}" } }, .{ .max_chat_body = 1 }, &why));
    const text = try std.json.Stringify.valueAlloc(std.testing.allocator, ask, .{});
    defer std.testing.allocator.free(text);
    const back = try std.json.parseFromSlice(Request, std.testing.allocator, text, .{});
    defer back.deinit();
    try std.testing.expectEqualStrings(ask.chat.body, back.value.chat.body);
    try std.testing.expect(back.value.chat.stream);
}
