//! The token ids the Kandinsky-6 text embedder feeds its two encoders, as upstream builds them
//! (kandinsky-6 eac06e39 kandinsky/core/components/text_embedder.py, transformers v5.18.0):
//!   Qwen2.5-VL: `_PROMPT_TEMPLATE.format(prompt)` (text_embedder.py:19-29, the template's spelling kept as published:
//!     "promt", "scren"), tokenized by the processor's Qwen2Tokenizer with max_length = 1024 + 129, truncation and
//!     padding="max_length" (text_embedder.py:155-163): right truncation, right padding with <|endoftext|> (151643;
//!     tokenizer_config.json pad_token, transformers' default padding_side "right"), attention mask 1 on real tokens.
//!     The encoder keeps rows 129.. (`_CROP_START`, text_embedder.py:31,172-173) and the mask's crop.
//!   CLIP: the raw prompt, max_length 77, truncation, add_special_tokens, padding="max_length" (text_embedder.py:205-
//!     212): <|startoftext|> + at most 75 tokens + <|endoftext|>, padded with <|endoftext|>; pooled row = argmax(ids)
//!     (modeling_clip.py:564-576, config eos_token_id == 2), i.e. the first <|endoftext|>.

const std = @import("std");
const tokenizer = @import("tokenizer");
pub const clip_tokenizer = @import("clip_tokenizer.zig");

/// Tokens of the padded Qwen sequence: max_length + crop (text_embedder.py:159).
pub const qwen_tokens: u32 = 1153;
/// Rows dropped from the front: the system prompt and "<|im_start|>user\n" (text_embedder.py:31).
pub const qwen_crop: u32 = 129;
/// Rows the encoder outputs: text_embedder.max_length (pro-distill.yaml `text_embedder.max_length: 1024`).
pub const qwen_rows: u32 = qwen_tokens - qwen_crop;
pub const qwen_pad: u32 = 151643; // <|endoftext|>

/// text_embedder.py:19-29 up to the `{}` placeholder, and after it.
pub const template_prefix = "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.\n" ++
    "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.\n" ++
    "Describe the location of the video, main characters or objects and their action.\n" ++
    "Describe the dynamism of the video and presented actions.\n" ++
    "Name the visual style of the video: whether it is a professional footage, user generated content, " ++
    "some kind of animation, video game or scren content.\n" ++
    "Describe the visual effects, postprocessing and transitions if they are presented in the video.\n" ++
    "Pay attention to the order of key actions shown in the scene.<|im_end|>\n" ++
    "<|im_start|>user\n";
pub const template_suffix = "<|im_end|>";

/// `ids` truncated to `out.len` and right-padded with `pad`; mask 1 on the kept ids. Returns the kept count.
pub fn padRight(ids: []const u32, pad: u32, out: []u32, mask: []u8) u32 {
    const n = @min(ids.len, out.len);
    @memcpy(out[0..n], ids[0..n]);
    @memset(out[n..], pad);
    @memset(mask[0..n], 1);
    @memset(mask[n..], 0);
    return @intCast(n);
}

/// The Qwen ids and mask for `prompt` (UTF-8); returns the count of real tokens (mask ones). A prompt holding the
/// vision placeholders <|image_pad|> / <|video_pad|> is refused: the processor's handling of them without images is
/// not part of the verified path.
pub fn qwenIds(tok: *const tokenizer.Tokenizer, gpa: std.mem.Allocator, prompt: []const u8, out_ids: *[qwen_tokens]u32, out_mask: *[qwen_tokens]u8) !u32 {
    if (!std.unicode.utf8ValidateSlice(prompt)) return error.InvalidUtf8;
    if (std.mem.indexOf(u8, prompt, "<|image_pad|>") != null or std.mem.indexOf(u8, prompt, "<|video_pad|>") != null)
        return error.UnsupportedPrompt;
    const text = try std.mem.concat(gpa, u8, &.{ template_prefix, prompt, template_suffix });
    defer gpa.free(text);
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    return padRight(ids, qwen_pad, out_ids, out_mask);
}

pub const Clip = struct {
    /// Tokens with mask 1: <|startoftext|>, the prompt's (at most 75), <|endoftext|>.
    valid: u32,
    /// The pooled row: argmax(ids), the first index of the largest id.
    eot: u32,
};

/// `encoded` (the prompt's ids without specials) as the CLIP tokenizer call frames it.
pub fn clipFrame(encoded: []const u32, out: *[clip_tokenizer.max_length]u32) Clip {
    const n = @min(encoded.len, clip_tokenizer.max_length - 2);
    out[0] = clip_tokenizer.bos;
    @memcpy(out[1 .. 1 + n], encoded[0..n]);
    out[1 + n] = clip_tokenizer.eos;
    @memset(out[2 + n ..], clip_tokenizer.pad);
    var eot: u32 = 0;
    for (out, 0..) |id, i| {
        if (id > out[eot]) eot = @intCast(i);
    }
    return .{ .valid = @intCast(n + 2), .eot = eot };
}

/// The 77 CLIP ids for `prompt` (UTF-8), with the mask count and the pooled row.
pub fn clipIds(tok: *const clip_tokenizer.ClipTokenizer, gpa: std.mem.Allocator, prompt: []const u8, out: *[clip_tokenizer.max_length]u32) !Clip {
    const ids = try tok.encode(gpa, prompt);
    defer gpa.free(ids);
    return clipFrame(ids, out);
}

const testing = std.testing;

test "padRight truncates on the right and pads with the pad id" {
    var out: [5]u32 = undefined;
    var mask: [5]u8 = undefined;
    try testing.expectEqual(@as(u32, 3), padRight(&.{ 7, 8, 9 }, 0, &out, &mask));
    try testing.expectEqualSlices(u32, &.{ 7, 8, 9, 0, 0 }, &out);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1, 0, 0 }, &mask);
    try testing.expectEqual(@as(u32, 5), padRight(&.{ 1, 2, 3, 4, 5, 6, 7 }, 0, &out, &mask));
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5 }, &out);
    try testing.expectEqualSlices(u8, &.{ 1, 1, 1, 1, 1 }, &mask);
}

test "clipFrame: specials, 75-token truncation, padding, pooled row at the first <|endoftext|>" {
    var out: [77]u32 = undefined;
    const a = clipFrame(&.{ 320, 2368 }, &out);
    try testing.expectEqual(Clip{ .valid = 4, .eot = 3 }, a);
    try testing.expectEqualSlices(u32, &.{ 49406, 320, 2368, 49407, 49407 }, out[0..5]);
    try testing.expectEqual(@as(u32, 49407), out[76]);
    var long: [100]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast(1000 + i);
    const b = clipFrame(&long, &out);
    try testing.expectEqual(Clip{ .valid = 77, .eot = 76 }, b);
    try testing.expectEqual(@as(u32, 1074), out[75]);
    const c = clipFrame(&.{ 343, 49407, 285 }, &out); // a literal <|endoftext|> in the prompt
    try testing.expectEqual(Clip{ .valid = 5, .eot = 2 }, c);
}

test "template: 1153 = 1024 + 129, the placeholder split" {
    try testing.expectEqual(@as(u32, 1024), qwen_rows);
    try testing.expect(std.mem.endsWith(u8, template_prefix, "<|im_end|>\n<|im_start|>user\n"));
    try testing.expect(std.mem.indexOf(u8, template_prefix, "promt engineer") != null);
    try testing.expect(std.mem.indexOf(u8, template_prefix, "or scren content.\nDescribe the visual") != null);
}

// ---- Exactness against the snapshot's tokenizer.json files (K6_SNAPSHOT_ROOT = the directory holding text_encoder/
// and tokenizer_2/). Expected ids come from an independent reference of the tokenizers pipelines (ASCII inputs),
// written from the two JSON configs; well-known ids agree (151644 <|im_start|>, 8948 "system", 198 "\n", 872 "user",
// 320 "a</w>", 2368 "cat</w>", 1125 "photo</w>", 539 "of</w>").

fn snapshotPath(buf: []u8, rel: []const u8) ![]const u8 {
    const root = std.testing.environ.getPosix("K6_SNAPSHOT_ROOT") orelse return error.SkipZigTest;
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ root, rel });
}

const qwen_prefix_ids = [_]u32{
    151644, 8948,  198,  2610, 525,   264,   2706,  83,   23576, 13,  60785, 279,  2766,   304,  7716,   624,
    74785,  1246,  279,  6249, 10797, 476,   62030, 11,   7512,  279, 15562, 323,  1651,   9210, 11,     3425,
    432,    11017, 279,  6171, 624,   74785, 279,   3728, 315,   279, 2766,  11,   1887,   5766, 476,    6171,
    323,    862,   1917, 624,  74785, 279,   17544, 2142, 315,   279, 2766,  323,  10449,  6168, 624,    675,
    279,    9124,  1707, 315,  279,   2766,  25,    3425, 432,   374, 264,   6584, 21852,  11,   1196,   7907,
    2213,   11,    1045, 3093, 315,   10355, 11,    2766, 1809,  476, 7862,  77,   2213,   624,  74785,  279,
    9124,   6239,  11,   1736, 20660, 323,   33592, 421,  807,   525, 10449, 304,  279,    2766, 624,    20595,
    6529,   311,   279,  1973, 315,   1376,  6168,  6839, 304,   279, 6109,  13,   151645, 198,  151644, 872,
    198,
};

test "qwen: the template prefix is 129 tokens (= _CROP_START) and the ids of the reference" {
    var pb: [4096]u8 = undefined;
    const path = try snapshotPath(&pb, "text_encoder/tokenizer.json");
    var tok = try tokenizer.loadTokenizer(std.testing.io, testing.allocator, path);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, qwen_crop), qwen_prefix_ids.len);
    const pre = try tok.encode(testing.allocator, template_prefix);
    defer testing.allocator.free(pre);
    try testing.expectEqualSlices(u32, &qwen_prefix_ids, pre);

    var ids: [qwen_tokens]u32 = undefined;
    var mask: [qwen_tokens]u8 = undefined;
    const valid = try qwenIds(&tok, testing.allocator, "A red fox runs through deep snow at dawn.", &ids, &mask);
    try testing.expectEqual(@as(u32, 140), valid);
    try testing.expectEqualSlices(u32, &qwen_prefix_ids, ids[0..qwen_crop]);
    try testing.expectEqualSlices(u32, &.{ 32, 2518, 38835, 8473, 1526, 5538, 11794, 518, 38393, 13, 151645 }, ids[qwen_crop..valid]);
    try testing.expectEqual(@as(u32, qwen_pad), ids[valid]);
    try testing.expectEqual(@as(u32, qwen_pad), ids[qwen_tokens - 1]);
    try testing.expectEqual(@as(u8, 1), mask[valid - 1]);
    try testing.expectEqual(@as(u8, 0), mask[valid]);

    try testing.expectEqual(@as(u32, 130), try qwenIds(&tok, testing.allocator, "", &ids, &mask));
    try testing.expectEqual(@as(u32, 151645), ids[129]);
    try testing.expectError(error.UnsupportedPrompt, qwenIds(&tok, testing.allocator, "x <|image_pad|>", &ids, &mask));

    // truncation: a long prompt fills all 1153 rows and loses the closing <|im_end|>
    const long = try testing.allocator.alloc(u8, 6000);
    defer testing.allocator.free(long);
    for (long, 0..) |*c, i| c.* = if (i % 2 == 0) 'a' else ' ';
    try testing.expectEqual(@as(u32, qwen_tokens), try qwenIds(&tok, testing.allocator, long, &ids, &mask));
    try testing.expect(ids[qwen_tokens - 1] != 151645);
    try testing.expectEqual(@as(u8, 1), mask[qwen_tokens - 1]);
}

test "clip: ids of the reference for several prompts" {
    var pb: [4096]u8 = undefined;
    const path = try snapshotPath(&pb, "tokenizer_2/tokenizer.json");
    var tok = try clip_tokenizer.ClipTokenizer.load(testing.allocator, std.testing.io, path);
    defer tok.deinit();
    const Case = struct { text: []const u8, ids: []const u32, eot: u32 };
    const cases = [_]Case{
        .{ .text = "a cat", .ids = &.{ 49406, 320, 2368, 49407 }, .eot = 3 },
        .{ .text = "A photo of a CAT, sitting on the mat!", .ids = &.{ 49406, 320, 1125, 539, 320, 2368, 267, 4919, 525, 518, 9063, 256, 49407 }, .eot = 12 },
        .{ .text = "Hello   World 123", .ids = &.{ 49406, 3306, 1002, 272, 273, 274, 49407 }, .eot = 6 },
        .{ .text = "don't stop", .ids = &.{ 49406, 847, 713, 1691, 49407 }, .eot = 4 },
        .{ .text = "<|endoftext|> x <|ENDOFTEXT|>", .ids = &.{ 49406, 49407, 343, 27, 347, 40786, 4160, 91, 285, 49407 }, .eot = 1 },
        .{ .text = "", .ids = &.{ 49406, 49407 }, .eot = 1 },
    };
    for (cases) |c| {
        var out: [77]u32 = undefined;
        const r = try clipIds(&tok, testing.allocator, c.text, &out);
        try testing.expectEqual(@as(u32, @intCast(c.ids.len)), r.valid);
        try testing.expectEqual(c.eot, r.eot);
        try testing.expectEqualSlices(u32, c.ids, out[0..r.valid]);
        for (out[r.valid..]) |id| try testing.expectEqual(clip_tokenizer.pad, id);
    }
}

test {
    _ = clip_tokenizer;
}
