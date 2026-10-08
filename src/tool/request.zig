//! What a tool is asked to make: one validated request per kind. New kinds add an arm, so every switch must handle it.

const std = @import("std");

pub const Kind = enum { image, video };

/// What a model can do; a request needs exactly one (`Request.capability`).
pub const Capability = enum {
    text_to_image,
    image_edit,
    text_to_video,
    image_to_video,

    pub fn kind(c: Capability) Kind {
        return switch (c) {
            .text_to_image, .image_edit => .image,
            .text_to_video, .image_to_video => .video,
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
    width: u32 = 1344,
    height: u32 = 768,
    seconds: u32 = 5,
    fps: u32 = 24,
    seed: u64 = 0,
    steps: u32 = 0,
    guidance: f32 = 0,
    audio: bool = true,
    first_frame: ?[]const u8 = null, // file name in the job directory (image to video)
};

/// The wire form is std.json's union shape: {"image": {...}} or {"video": {...}}.
pub const Request = union(Kind) {
    image: Image,
    video: Video,

    pub fn kind(r: Request) Kind {
        return std.meta.activeTag(r);
    }

    /// What a model must be able to do to serve this request: reference images make an edit, a first frame a video from an image.
    pub fn capability(r: Request) Capability {
        return switch (r) {
            .image => |v| if (v.references.len > 0) .image_edit else .text_to_image,
            .video => |v| if (v.first_frame != null) .image_to_video else .text_to_video,
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
    }
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
