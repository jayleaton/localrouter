//! CLIP ViT-L/14's tokenizer (`tokenizer_2/tokenizer.json`) exactly as transformers v5.18.0 `CLIPTokenizer` runs it
//! through Hugging Face tokenizers (models/clip/tokenization_clip.py:61-104): normalizer NFC, `\s+` -> " ", Lowercase;
//! pre-tokenizer Split(CLIP regex, Removed, invert) then ByteLevel (GPT-2 regex); BPE with the "</w>" end-of-word
//! suffix; RobertaProcessing <|startoftext|> ... <|endoftext|>.
//!
//! TensorFold's tokenizer has no Lowercase normalizer, so the model, merges, pre-tokenizer and NFC / Replace steps run
//! in TensorFold (parsed from the same file, the Lowercase step removed) and this file adds:
//!   - the raw split on `<|endoftext|>` (added token, normalized = false: tokenizers matches it in the raw text, before
//!     normalization), so it is removed from TensorFold's added tokens and a lowercased "<|ENDOFTEXT|>" stays text;
//!   - Lowercase as tokenizers' `NormalizedString::lowercase` does it: per character `char::to_lowercase` (no
//!     context-sensitive final sigma), from the table below (Unicode 16.0.0, generated with Python 3.14
//!     `chr(cp).lower()`; 1459 single-codepoint mappings in 186 runs plus U+0130 -> "i\u{307}").
//! Lowercase runs before TensorFold's NFC and whitespace Replace instead of after them. Replace commutes with it (no
//! whitespace character has a case mapping); lowercasing commutes with NFC up to canonical equivalence, and the result
//! here is NFC. tokenizers' order differs only where lowercasing an NFC string yields a non-NFC one; no such ASCII or
//! Latin input exists, but this ordering is not verified for every codepoint.

const std = @import("std");
const tokenizer = @import("tokenizer");

pub const bos: u32 = 49406; // <|startoftext|>
pub const eos: u32 = 49407; // <|endoftext|>
pub const pad: u32 = 49407; // tokenizer_config.json pad_token "<|endoftext|>"
pub const max_length: u32 = 77; // text_embedder.py:207, CLIP max_position_embeddings
const eos_text = "<|endoftext|>";

/// The normalizer this file accepts (tokenizer.json as published) and the one TensorFold runs instead.
const expected_normalizer =
    \\{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "},{"type":"Lowercase"}]}
;
const tensorfold_normalizer =
    \\{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "}]}
;
const expected_post =
    \\{"type":"RobertaProcessing","sep":["<|endoftext|>",49407],"cls":["<|startoftext|>",49406],"trim_offsets":false,"add_prefix_space":false}
;

pub const ClipTokenizer = struct {
    tok: tokenizer.Tokenizer,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !ClipTokenizer {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// Checks the file is the published CLIP tokenizer (normalizer, post-processor, the two added tokens), then hands
    /// TensorFold the same JSON without the Lowercase step and without the raw-matched <|endoftext|>.
    pub fn parse(gpa: std.mem.Allocator, json: []const u8) !ClipTokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var root = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{ .max_value_len = json.len });
        if (root != .object) return error.InvalidTokenizer;
        const obj = &root.object;
        try expectJson(a, obj.get("normalizer") orelse return error.UnsupportedClipTokenizer, expected_normalizer);
        try expectJson(a, obj.get("post_processor") orelse return error.UnsupportedClipTokenizer, expected_post);
        const added = obj.getPtr("added_tokens") orelse return error.UnsupportedClipTokenizer;
        if (added.* != .array or added.array.items.len != 2) return error.UnsupportedClipTokenizer;
        var keep: ?usize = null;
        for (added.array.items, 0..) |item, i| {
            if (item != .object) return error.UnsupportedClipTokenizer;
            const o = item.object;
            const id = o.get("id") orelse return error.UnsupportedClipTokenizer;
            const content = o.get("content") orelse return error.UnsupportedClipTokenizer;
            const normalized = o.get("normalized") orelse return error.UnsupportedClipTokenizer;
            if (id != .integer or content != .string or normalized != .bool) return error.UnsupportedClipTokenizer;
            for ([_][]const u8{ "single_word", "lstrip", "rstrip" }) |f| {
                const v = o.get(f) orelse return error.UnsupportedClipTokenizer;
                if (v != .bool or v.bool) return error.UnsupportedClipTokenizer;
            }
            if (id.integer == bos and std.mem.eql(u8, content.string, "<|startoftext|>") and normalized.bool) {
                keep = i;
            } else if (!(id.integer == eos and std.mem.eql(u8, content.string, eos_text) and !normalized.bool)) {
                return error.UnsupportedClipTokenizer;
            }
        }
        const kept = added.array.items[keep orelse return error.UnsupportedClipTokenizer];
        added.array.clearRetainingCapacity();
        try added.array.append(kept);
        try obj.put(a, "normalizer", try std.json.parseFromSliceLeaky(std.json.Value, a, tensorfold_normalizer, .{}));
        const out = try std.json.Stringify.valueAlloc(a, root, .{});
        var tok = try tokenizer.parse(gpa, out);
        errdefer tok.deinit();
        if ((tok.specialTokenId("<|startoftext|>") orelse 0) != bos) return error.UnsupportedClipTokenizer;
        return .{ .tok = tok };
    }

    pub fn deinit(c: *ClipTokenizer) void {
        c.tok.deinit();
    }

    /// tokenizers' `encode(text, add_special_tokens=False).ids`; caller frees. `text` must be UTF-8 (a Python str).
    pub fn encode(c: *const ClipTokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(gpa);
        var rest = text;
        while (true) {
            const at = std.mem.indexOf(u8, rest, eos_text);
            const seg = rest[0 .. at orelse rest.len];
            if (seg.len > 0) {
                const low = try lowercase(gpa, seg);
                defer gpa.free(low);
                const part = try c.tok.encode(gpa, low);
                defer gpa.free(part);
                try ids.appendSlice(gpa, part);
            }
            const i = at orelse break;
            try ids.append(gpa, eos);
            rest = rest[i + eos_text.len ..];
        }
        return ids.toOwnedSlice(gpa);
    }
};

fn expectJson(a: std.mem.Allocator, v: std.json.Value, want: []const u8) !void {
    const got = try std.json.Stringify.valueAlloc(a, v, .{});
    if (!std.mem.eql(u8, got, want)) return error.UnsupportedClipTokenizer;
}

const Run = struct { start: u21, count: u16, stride: u2, delta: i32 };

/// Per-codepoint lowercase (see the file comment); caller frees.
pub fn lowercase(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var it = (try std.unicode.Utf8View.init(text)).iterator();
    var buf: [4]u8 = undefined;
    while (it.nextCodepoint()) |cp| {
        if (cp == 0x130) {
            try out.appendSlice(gpa, "i\u{307}");
            continue;
        }
        const n = try std.unicode.utf8Encode(lowerCodepoint(cp), &buf);
        try out.appendSlice(gpa, buf[0..n]);
    }
    return out.toOwnedSlice(gpa);
}

pub fn lowerCodepoint(cp: u21) u21 {
    if (cp < 0x80) return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
    var lo: usize = 0;
    var hi: usize = lower_runs.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (lower_runs[mid].start <= cp) lo = mid + 1 else hi = mid;
    }
    if (lo == 0) return cp;
    const r = lower_runs[lo - 1];
    const off = cp - r.start;
    if (off % r.stride != 0 or off / r.stride >= r.count) return cp;
    return @intCast(@as(i32, cp) + r.delta);
}

/// chr(cp).lower() != chr(cp), single-codepoint results, as runs: start, count, stride, delta (Unicode 16.0.0).
const lower_runs = [_]Run{
    .{ .start = 0x41, .count = 26, .stride = 1, .delta = 32 },
    .{ .start = 0xC0, .count = 23, .stride = 1, .delta = 32 },
    .{ .start = 0xD8, .count = 7, .stride = 1, .delta = 32 },
    .{ .start = 0x100, .count = 24, .stride = 2, .delta = 1 },
    .{ .start = 0x132, .count = 3, .stride = 2, .delta = 1 },
    .{ .start = 0x139, .count = 8, .stride = 2, .delta = 1 },
    .{ .start = 0x14A, .count = 23, .stride = 2, .delta = 1 },
    .{ .start = 0x178, .count = 1, .stride = 1, .delta = -121 },
    .{ .start = 0x179, .count = 3, .stride = 2, .delta = 1 },
    .{ .start = 0x181, .count = 1, .stride = 1, .delta = 210 },
    .{ .start = 0x182, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0x186, .count = 1, .stride = 1, .delta = 206 },
    .{ .start = 0x187, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x189, .count = 2, .stride = 1, .delta = 205 },
    .{ .start = 0x18B, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x18E, .count = 1, .stride = 1, .delta = 79 },
    .{ .start = 0x18F, .count = 1, .stride = 1, .delta = 202 },
    .{ .start = 0x190, .count = 1, .stride = 1, .delta = 203 },
    .{ .start = 0x191, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x193, .count = 1, .stride = 1, .delta = 205 },
    .{ .start = 0x194, .count = 1, .stride = 1, .delta = 207 },
    .{ .start = 0x196, .count = 1, .stride = 1, .delta = 211 },
    .{ .start = 0x197, .count = 1, .stride = 1, .delta = 209 },
    .{ .start = 0x198, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x19C, .count = 1, .stride = 1, .delta = 211 },
    .{ .start = 0x19D, .count = 1, .stride = 1, .delta = 213 },
    .{ .start = 0x19F, .count = 1, .stride = 1, .delta = 214 },
    .{ .start = 0x1A0, .count = 3, .stride = 2, .delta = 1 },
    .{ .start = 0x1A6, .count = 1, .stride = 1, .delta = 218 },
    .{ .start = 0x1A7, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1A9, .count = 1, .stride = 1, .delta = 218 },
    .{ .start = 0x1AC, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1AE, .count = 1, .stride = 1, .delta = 218 },
    .{ .start = 0x1AF, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1B1, .count = 2, .stride = 1, .delta = 217 },
    .{ .start = 0x1B3, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0x1B7, .count = 1, .stride = 1, .delta = 219 },
    .{ .start = 0x1B8, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1BC, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1C4, .count = 1, .stride = 1, .delta = 2 },
    .{ .start = 0x1C5, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1C7, .count = 1, .stride = 1, .delta = 2 },
    .{ .start = 0x1C8, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1CA, .count = 1, .stride = 1, .delta = 2 },
    .{ .start = 0x1CB, .count = 9, .stride = 2, .delta = 1 },
    .{ .start = 0x1DE, .count = 9, .stride = 2, .delta = 1 },
    .{ .start = 0x1F1, .count = 1, .stride = 1, .delta = 2 },
    .{ .start = 0x1F2, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0x1F6, .count = 1, .stride = 1, .delta = -97 },
    .{ .start = 0x1F7, .count = 1, .stride = 1, .delta = -56 },
    .{ .start = 0x1F8, .count = 20, .stride = 2, .delta = 1 },
    .{ .start = 0x220, .count = 1, .stride = 1, .delta = -130 },
    .{ .start = 0x222, .count = 9, .stride = 2, .delta = 1 },
    .{ .start = 0x23A, .count = 1, .stride = 1, .delta = 10795 },
    .{ .start = 0x23B, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x23D, .count = 1, .stride = 1, .delta = -163 },
    .{ .start = 0x23E, .count = 1, .stride = 1, .delta = 10792 },
    .{ .start = 0x241, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x243, .count = 1, .stride = 1, .delta = -195 },
    .{ .start = 0x244, .count = 1, .stride = 1, .delta = 69 },
    .{ .start = 0x245, .count = 1, .stride = 1, .delta = 71 },
    .{ .start = 0x246, .count = 5, .stride = 2, .delta = 1 },
    .{ .start = 0x370, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0x376, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x37F, .count = 1, .stride = 1, .delta = 116 },
    .{ .start = 0x386, .count = 1, .stride = 1, .delta = 38 },
    .{ .start = 0x388, .count = 3, .stride = 1, .delta = 37 },
    .{ .start = 0x38C, .count = 1, .stride = 1, .delta = 64 },
    .{ .start = 0x38E, .count = 2, .stride = 1, .delta = 63 },
    .{ .start = 0x391, .count = 17, .stride = 1, .delta = 32 },
    .{ .start = 0x3A3, .count = 9, .stride = 1, .delta = 32 },
    .{ .start = 0x3CF, .count = 1, .stride = 1, .delta = 8 },
    .{ .start = 0x3D8, .count = 12, .stride = 2, .delta = 1 },
    .{ .start = 0x3F4, .count = 1, .stride = 1, .delta = -60 },
    .{ .start = 0x3F7, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x3F9, .count = 1, .stride = 1, .delta = -7 },
    .{ .start = 0x3FA, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x3FD, .count = 3, .stride = 1, .delta = -130 },
    .{ .start = 0x400, .count = 16, .stride = 1, .delta = 80 },
    .{ .start = 0x410, .count = 32, .stride = 1, .delta = 32 },
    .{ .start = 0x460, .count = 17, .stride = 2, .delta = 1 },
    .{ .start = 0x48A, .count = 27, .stride = 2, .delta = 1 },
    .{ .start = 0x4C0, .count = 1, .stride = 1, .delta = 15 },
    .{ .start = 0x4C1, .count = 7, .stride = 2, .delta = 1 },
    .{ .start = 0x4D0, .count = 48, .stride = 2, .delta = 1 },
    .{ .start = 0x531, .count = 38, .stride = 1, .delta = 48 },
    .{ .start = 0x10A0, .count = 38, .stride = 1, .delta = 7264 },
    .{ .start = 0x10C7, .count = 1, .stride = 1, .delta = 7264 },
    .{ .start = 0x10CD, .count = 1, .stride = 1, .delta = 7264 },
    .{ .start = 0x13A0, .count = 80, .stride = 1, .delta = 38864 },
    .{ .start = 0x13F0, .count = 6, .stride = 1, .delta = 8 },
    .{ .start = 0x1C89, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x1C90, .count = 43, .stride = 1, .delta = -3008 },
    .{ .start = 0x1CBD, .count = 3, .stride = 1, .delta = -3008 },
    .{ .start = 0x1E00, .count = 75, .stride = 2, .delta = 1 },
    .{ .start = 0x1E9E, .count = 1, .stride = 1, .delta = -7615 },
    .{ .start = 0x1EA0, .count = 48, .stride = 2, .delta = 1 },
    .{ .start = 0x1F08, .count = 8, .stride = 1, .delta = -8 },
    .{ .start = 0x1F18, .count = 6, .stride = 1, .delta = -8 },
    .{ .start = 0x1F28, .count = 8, .stride = 1, .delta = -8 },
    .{ .start = 0x1F38, .count = 8, .stride = 1, .delta = -8 },
    .{ .start = 0x1F48, .count = 6, .stride = 1, .delta = -8 },
    .{ .start = 0x1F59, .count = 4, .stride = 2, .delta = -8 },
    .{ .start = 0x1F68, .count = 8, .stride = 1, .delta = -8 },
    .{ .start = 0x1F88, .count = 8, .stride = 1, .delta = -8 },
    .{ .start = 0x1F98, .count = 8, .stride = 1, .delta = -8 },
    .{ .start = 0x1FA8, .count = 8, .stride = 1, .delta = -8 },
    .{ .start = 0x1FB8, .count = 2, .stride = 1, .delta = -8 },
    .{ .start = 0x1FBA, .count = 2, .stride = 1, .delta = -74 },
    .{ .start = 0x1FBC, .count = 1, .stride = 1, .delta = -9 },
    .{ .start = 0x1FC8, .count = 4, .stride = 1, .delta = -86 },
    .{ .start = 0x1FCC, .count = 1, .stride = 1, .delta = -9 },
    .{ .start = 0x1FD8, .count = 2, .stride = 1, .delta = -8 },
    .{ .start = 0x1FDA, .count = 2, .stride = 1, .delta = -100 },
    .{ .start = 0x1FE8, .count = 2, .stride = 1, .delta = -8 },
    .{ .start = 0x1FEA, .count = 2, .stride = 1, .delta = -112 },
    .{ .start = 0x1FEC, .count = 1, .stride = 1, .delta = -7 },
    .{ .start = 0x1FF8, .count = 2, .stride = 1, .delta = -128 },
    .{ .start = 0x1FFA, .count = 2, .stride = 1, .delta = -126 },
    .{ .start = 0x1FFC, .count = 1, .stride = 1, .delta = -9 },
    .{ .start = 0x2126, .count = 1, .stride = 1, .delta = -7517 },
    .{ .start = 0x212A, .count = 1, .stride = 1, .delta = -8383 },
    .{ .start = 0x212B, .count = 1, .stride = 1, .delta = -8262 },
    .{ .start = 0x2132, .count = 1, .stride = 1, .delta = 28 },
    .{ .start = 0x2160, .count = 16, .stride = 1, .delta = 16 },
    .{ .start = 0x2183, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x24B6, .count = 26, .stride = 1, .delta = 26 },
    .{ .start = 0x2C00, .count = 48, .stride = 1, .delta = 48 },
    .{ .start = 0x2C60, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x2C62, .count = 1, .stride = 1, .delta = -10743 },
    .{ .start = 0x2C63, .count = 1, .stride = 1, .delta = -3814 },
    .{ .start = 0x2C64, .count = 1, .stride = 1, .delta = -10727 },
    .{ .start = 0x2C67, .count = 3, .stride = 2, .delta = 1 },
    .{ .start = 0x2C6D, .count = 1, .stride = 1, .delta = -10780 },
    .{ .start = 0x2C6E, .count = 1, .stride = 1, .delta = -10749 },
    .{ .start = 0x2C6F, .count = 1, .stride = 1, .delta = -10783 },
    .{ .start = 0x2C70, .count = 1, .stride = 1, .delta = -10782 },
    .{ .start = 0x2C72, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x2C75, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0x2C7E, .count = 2, .stride = 1, .delta = -10815 },
    .{ .start = 0x2C80, .count = 50, .stride = 2, .delta = 1 },
    .{ .start = 0x2CEB, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0x2CF2, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0xA640, .count = 23, .stride = 2, .delta = 1 },
    .{ .start = 0xA680, .count = 14, .stride = 2, .delta = 1 },
    .{ .start = 0xA722, .count = 7, .stride = 2, .delta = 1 },
    .{ .start = 0xA732, .count = 31, .stride = 2, .delta = 1 },
    .{ .start = 0xA779, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0xA77D, .count = 1, .stride = 1, .delta = -35332 },
    .{ .start = 0xA77E, .count = 5, .stride = 2, .delta = 1 },
    .{ .start = 0xA78B, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0xA78D, .count = 1, .stride = 1, .delta = -42280 },
    .{ .start = 0xA790, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0xA796, .count = 10, .stride = 2, .delta = 1 },
    .{ .start = 0xA7AA, .count = 1, .stride = 1, .delta = -42308 },
    .{ .start = 0xA7AB, .count = 1, .stride = 1, .delta = -42319 },
    .{ .start = 0xA7AC, .count = 1, .stride = 1, .delta = -42315 },
    .{ .start = 0xA7AD, .count = 1, .stride = 1, .delta = -42305 },
    .{ .start = 0xA7AE, .count = 1, .stride = 1, .delta = -42308 },
    .{ .start = 0xA7B0, .count = 1, .stride = 1, .delta = -42258 },
    .{ .start = 0xA7B1, .count = 1, .stride = 1, .delta = -42282 },
    .{ .start = 0xA7B2, .count = 1, .stride = 1, .delta = -42261 },
    .{ .start = 0xA7B3, .count = 1, .stride = 1, .delta = 928 },
    .{ .start = 0xA7B4, .count = 8, .stride = 2, .delta = 1 },
    .{ .start = 0xA7C4, .count = 1, .stride = 1, .delta = -48 },
    .{ .start = 0xA7C5, .count = 1, .stride = 1, .delta = -42307 },
    .{ .start = 0xA7C6, .count = 1, .stride = 1, .delta = -35384 },
    .{ .start = 0xA7C7, .count = 2, .stride = 2, .delta = 1 },
    .{ .start = 0xA7CB, .count = 1, .stride = 1, .delta = -42343 },
    .{ .start = 0xA7CC, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0xA7D0, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0xA7D6, .count = 3, .stride = 2, .delta = 1 },
    .{ .start = 0xA7DC, .count = 1, .stride = 1, .delta = -42561 },
    .{ .start = 0xA7F5, .count = 1, .stride = 1, .delta = 1 },
    .{ .start = 0xFF21, .count = 26, .stride = 1, .delta = 32 },
    .{ .start = 0x10400, .count = 40, .stride = 1, .delta = 40 },
    .{ .start = 0x104B0, .count = 36, .stride = 1, .delta = 40 },
    .{ .start = 0x10570, .count = 11, .stride = 1, .delta = 39 },
    .{ .start = 0x1057C, .count = 15, .stride = 1, .delta = 39 },
    .{ .start = 0x1058C, .count = 7, .stride = 1, .delta = 39 },
    .{ .start = 0x10594, .count = 2, .stride = 1, .delta = 39 },
    .{ .start = 0x10C80, .count = 51, .stride = 1, .delta = 64 },
    .{ .start = 0x10D50, .count = 22, .stride = 1, .delta = 32 },
    .{ .start = 0x118A0, .count = 32, .stride = 1, .delta = 32 },
    .{ .start = 0x16E40, .count = 32, .stride = 1, .delta = 32 },
    .{ .start = 0x1E900, .count = 34, .stride = 1, .delta = 34 },
};

const testing = std.testing;

test "lowercase table: ASCII, Latin-1, Greek, Cyrillic, the one multi-codepoint mapping, final sigma not special" {
    try testing.expectEqual(@as(u21, 'a'), lowerCodepoint('A'));
    try testing.expectEqual(@as(u21, 0xE9), lowerCodepoint(0xC9)); // É
    try testing.expectEqual(@as(u21, 0xD7), lowerCodepoint(0xD7)); // × has no mapping
    try testing.expectEqual(@as(u21, 0x101), lowerCodepoint(0x100)); // Ā
    try testing.expectEqual(@as(u21, 0x101), lowerCodepoint(0x101));
    try testing.expectEqual(@as(u21, 0x3C3), lowerCodepoint(0x3A3)); // Σ -> σ always (per-char to_lowercase)
    try testing.expectEqual(@as(u21, 0x430), lowerCodepoint(0x410)); // А
    try testing.expectEqual(@as(u21, 0x6B), lowerCodepoint(0x212A)); // Kelvin sign
    try testing.expectEqual(@as(u21, 0x1E922), lowerCodepoint(0x1E900)); // Adlam
    const s = try lowercase(testing.allocator, "ÀB İ ΣΑΣ");
    defer testing.allocator.free(s);
    try testing.expectEqualStrings("àb i\u{307} σασ", s);
}

test "lowercase runs are sorted and disjoint" {
    var prev_end: u32 = 0;
    for (lower_runs) |r| {
        try testing.expect(r.start >= prev_end);
        try testing.expect(r.count >= 1 and r.stride >= 1);
        prev_end = @as(u32, r.start) + (@as(u32, r.count) - 1) * r.stride + 1;
    }
    try testing.expectEqual(@as(usize, 186), lower_runs.len);
}
