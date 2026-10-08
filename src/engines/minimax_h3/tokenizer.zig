//! MiniMax H3's prompt tokenizer, as the twin's `te32.tokenize` (ComfyUI 0.37.0's `MiniMaxH3Tokenizer` text path): the
//! Qwen BPE of `tokenizer.json` (the H3 pack carries it; `pack.write_tokenizer` made it from ComfyUI's qwen25_tokenizer files
//! with the seven extra special tokens, and checked it against the twin's tokenizer), read by TensorFold's tokenizers
//! implementation that the image engine uses (`qwen_image.tokenizer`). No chat template, no BOS / EOS, no padding,
//! no truncation. The pre-pass is the twin's: the literal `\(` and `\)` become `(` and `)`; the text is split at
//! `(?<=\s)embedding:` (the `embedding:` keyword preceded by a whitespace character) and each piece is tokenized on its
//! own (no embeddings directory is assumed); an empty result is the pad token [151643].

const std = @import("std");
const qi = @import("qwen_image");

pub const pad_id: u32 = 151643;
/// The extra special tokens (te32.py EXTRA_TOKENS); `load` refuses a tokenizer.json that numbers them differently.
pub const extra_tokens = [_]struct { []const u8, u32 }{
    .{ "<d>", 151669 },          .{ "</d>", 151670 },            .{ "<|cutoff|>", 151671 },        .{ "<|lyrics_start|>", 151672 },
    .{ "<|lyrics_end|>", 151673 }, .{ "<|caption_start|>", 151674 }, .{ "<|caption_end|>", 151675 },
};
const keyword = "embedding:";

pub const Tokenizer = struct {
    tok: qi.tokenizer.Tokenizer,

    /// `dir/tokenizer.json` of the H3 pack.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Tokenizer {
        var a: std.heap.ArenaAllocator = .init(gpa);
        defer a.deinit();
        var t: Tokenizer = .{ .tok = try qi.tokenizer.loadTokenizer(io, gpa, try std.fs.path.join(a.allocator(), &.{ dir, "tokenizer.json" })) };
        errdefer t.deinit();
        for (extra_tokens) |e| if (t.tok.specialTokenId(e[0]) != e[1]) return error.BadTokenizer;
        return t;
    }

    /// A tokenizer from the JSON text itself (the tests'; no check of the extra tokens).
    pub fn fromJson(gpa: std.mem.Allocator, json: []const u8) !Tokenizer {
        return .{ .tok = try qi.tokenizer.parse(gpa, json) };
    }

    pub fn deinit(t: *Tokenizer) void {
        t.tok.deinit();
    }

    /// The token ids of `prompt`; caller frees with `gpa`.
    pub fn ids(t: *const Tokenizer, gpa: std.mem.Allocator, prompt: []const u8) ![]u32 {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        // escape_important / unescape_important: the same two passes through sentinels as the twin
        var text = try std.mem.replaceOwned(u8, a, prompt, "\\)", "\x00\x01");
        text = try std.mem.replaceOwned(u8, a, text, "\\(", "\x00\x02");
        text = try std.mem.replaceOwned(u8, a, text, "\x00\x01", ")");
        text = try std.mem.replaceOwned(u8, a, text, "\x00\x02", "(");
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        var at: usize = 0;
        var from: usize = 0;
        while (nextKeyword(text, from)) |m| : (from = m + keyword.len) {
            try t.piece(gpa, text[at..m], &out);
            at = m;
        }
        try t.piece(gpa, text[at..], &out);
        if (out.items.len == 0) try out.append(gpa, pad_id);
        return out.toOwnedSlice(gpa);
    }

    fn piece(t: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) !void {
        if (text.len == 0) return;
        const ids_ = try t.tok.encode(gpa, text);
        defer gpa.free(ids_);
        try out.appendSlice(gpa, ids_);
    }
};

/// Python's `\s` on a str: the characters `str.isspace` accepts.
fn isSpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// The start of the first `embedding:` at or after `from` whose preceding character is whitespace (the regex
/// `(?<=\s)embedding:`; at the start of the text there is no preceding character, so no match).
fn nextKeyword(text: []const u8, from: usize) ?usize {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, keyword)) |m| {
        if (m > 0) {
            var s = m - 1;
            while (s > 0 and text[s] & 0xC0 == 0x80) s -= 1; // back to the previous character's first byte
            const cp = std.unicode.utf8Decode(text[s..m]) catch 0xFFFD;
            if (isSpace(cp)) return m;
        }
        i = m + 1;
    }
    return null;
}

const test_json =
    \\{"added_tokens":[{"id":6,"content":"<|end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
    \\ "normalizer":{"type":"NFC"},
    \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
    \\ "decoder":{"type":"ByteLevel"},
    \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    \\  "vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"(":4,")":5},"merges":["a b"]}}
;

/// A vocabulary with the merge "Ġ e", to see whether the space before `embedding:` stays in the same piece.
const split_json =
    \\{"normalizer":{"type":"NFC"},
    \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
    \\ "decoder":{"type":"ByteLevel"},
    \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    \\  "vocab":{"x":0,"Ġ":1,"Ġe":2,"e":3,"m":4,"b":5,"d":6,"i":7,"n":8,"g":9,":":10,"y":11},"merges":["Ġ e"]}}
;

test "the keyword splits only after whitespace" {
    try std.testing.expectEqual(@as(?usize, 2), nextKeyword("x embedding:y", 0));
    try std.testing.expectEqual(@as(?usize, null), nextKeyword("embedding:y", 0));
    try std.testing.expectEqual(@as(?usize, null), nextKeyword("xembedding:y", 0));
    try std.testing.expectEqual(@as(?usize, 3), nextKeyword("x\u{a0}embedding:", 0)); // the no-break space is `\s`
    try std.testing.expectEqual(@as(?usize, 2), nextKeyword("x\tembedding:y", 0));
    try std.testing.expectEqual(@as(?usize, null), nextKeyword("x-embedding:", 0)); // a hyphen is not
    try std.testing.expectEqual(@as(?usize, 13), nextKeyword("x embedding: embedding:y", 2 + keyword.len));
}

test "escapes, added tokens and the pad token" {
    const a = std.testing.allocator;
    var t = try Tokenizer.fromJson(a, test_json);
    defer t.deinit();
    const cases = [_]struct { []const u8, []const u32 }{
        .{ "ab", &.{3} },
        .{ "a\\(b\\)", &.{ 0, 4, 1, 5 } }, // `\(` and `\)` read as parentheses
        .{ "a(b)", &.{ 0, 4, 1, 5 } },
        .{ "ab <|end|>ab", &.{ 3, 2, 6, 3 } }, // an added token is cut out; the space stays on its left
        .{ "", &.{pad_id} },
        .{ "<|end|>", &.{6} },
    };
    for (cases) |c| {
        const got = try t.ids(a, c[0]);
        defer a.free(got);
        try std.testing.expectEqualSlices(u32, c[1], got);
    }
}

test "each embedding piece is tokenized on its own" {
    const a = std.testing.allocator;
    var t = try Tokenizer.fromJson(a, split_json);
    defer t.deinit();
    const got = try t.ids(a, "x embedding:y");
    defer a.free(got);
    // "x " and "embedding:y": the space is a token of its own (no "Ġe" across the cut)
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 3, 4, 5, 3, 6, 6, 7, 8, 9, 10, 11 }, got);
    // without a whitespace before the keyword there is no cut
    const whole = try t.ids(a, "xembedding:y");
    defer a.free(whole);
    try std.testing.expectEqualSlices(u32, &.{ 0, 3, 4, 5, 3, 6, 6, 7, 8, 9, 10, 11 }, whole);
}
