//! The model management procedure the chat model follows when `voice.manage_models` is on: added to the voice
//! session's instructions (`respond.zig`) and served to the web client at GET /v1/skills/model-manager. External
//! agents read the longer skills/jarvis-model-manager/SKILL.md, which teaches the same steps over MCP.
//! Its `{models}` line is this machine's models, from the config (`render`): swapping a model changes no text here.

const std = @import("std");
const Kind = @import("../tool/request.zig").Kind;

pub const model_skill = @embedFile("model_skill.md");

/// A model the procedure names: its kind, its id and the memory it takes loaded; `you`: the voice's own chat model.
pub const Member = struct { kind: Kind, id: []const u8, bytes: u64, you: bool = false };

/// The procedure with its `{models}` line filled in: the memory the models share (`budget`; 0: not said) and each
/// model's role, id and size. No models: the line is left out.
pub fn render(a: std.mem.Allocator, budget: u64, members: []const Member) error{OutOfMemory}![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    if (members.len > 0) roster(&w.writer, budget, members) catch return error.OutOfMemory;
    return std.mem.replaceOwned(u8, a, model_skill, "{models}", w.written());
}

fn roster(w: *std.Io.Writer, budget: u64, members: []const Member) std.Io.Writer.Error!void {
    if (budget > 0) try w.print("The machine has about {d} GiB for models: ", .{gib(budget)}) else try w.writeAll("Its models: ");
    for (members, 0..) |m, i| try w.print("{s}{s} {s} ({d} GiB)", .{ if (i > 0) "; " else "", role(m), m.id, gib(m.bytes) });
    try w.writeAll(". ");
}

fn role(m: Member) []const u8 {
    return switch (m.kind) {
        .chat => if (m.you) "you, the chat model" else "the chat model",
        .speech => "the voice",
        .transcription => "the ears",
        .image => "the image model",
        .video => "the video model",
        .decision => "the decision model",
    };
}

fn gib(bytes: u64) u64 {
    return (bytes + (1 << 29)) >> 30;
}

test "the models line comes from the models given; none: it is left out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try render(a, 117 << 30, &.{
        .{ .kind = .chat, .id = "chat-a", .bytes = 24 << 30, .you = true },
        .{ .kind = .speech, .id = "voice-b", .bytes = 21 << 30 },
        .{ .kind = .video, .id = "clip-c", .bytes = (40 << 30) - 1 },
    });
    try std.testing.expect(std.mem.indexOf(u8, text, "The machine has about 117 GiB for models: you, the chat model chat-a (24 GiB); the voice voice-b (21 GiB); the video model clip-c (40 GiB). The tools take") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "{models}") == null);
    const bare = try render(a, 0, &.{});
    try std.testing.expect(std.mem.indexOf(u8, bare, "for lack of room. The tools take") != null);
}
