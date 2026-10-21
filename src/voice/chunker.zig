//! A streamed chat reply as speakable chunks for TTS: sentence (or first-clause) breaks, markdown stripped to what
//! should be spoken, and `<think>` spans dropped from the stream.

const std = @import("std");

const abbreviations = [_][]const u8{ "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e" };
const closers = [_][]const u8{ "\"", "'", ")", "]", "*", "_", "”", "’", "」", "』", "）" };
const cjk_ends = [_][]const u8{ "。", "！", "？" };
const ellipsis = "…";

pub const Chunker = struct {
    gpa: std.mem.Allocator,
    pending: std.ArrayList(u8) = .empty,
    emitted: usize = 0, // chunks returned so far
    min_chars: usize = 40, // later chunks: merge sentences until at least this long
    clause_chars: usize = 24, // the first chunk may end at a clause (, ; :) once this long
    max_chars: usize = 280, // a chunk never exceeds this: break at the last clause mark or space before it

    /// Frees the pending text.
    pub fn deinit(c: *Chunker) void {
        c.pending.deinit(c.gpa);
    }

    /// Appends streamed text.
    pub fn push(c: *Chunker, text: []const u8) !void {
        try c.pending.appendSlice(c.gpa, text);
    }

    /// The next complete chunk (trimmed), owned by `a`, or null until more text arrives.
    pub fn next(c: *Chunker, a: std.mem.Allocator) !?[]const u8 {
        c.drop(c.pending.items.len - std.mem.trimStart(u8, c.pending.items, &std.ascii.whitespace).len);
        if (c.pending.items.len == 0) return null;
        const end = c.scan(c.pending.items) orelse return null;
        const chunk = try a.dupe(u8, std.mem.trimEnd(u8, c.pending.items[0..end], &std.ascii.whitespace));
        c.drop(end);
        c.emitted += 1;
        return chunk;
    }

    /// The rest at the end of the reply, trimmed; null when only whitespace is left.
    pub fn finish(c: *Chunker, a: std.mem.Allocator) !?[]const u8 {
        defer c.pending.clearRetainingCapacity();
        const rest = std.mem.trim(u8, c.pending.items, &std.ascii.whitespace);
        if (rest.len == 0) return null;
        c.emitted += 1;
        return try a.dupe(u8, rest);
    }

    fn drop(c: *Chunker, n: usize) void {
        const items = c.pending.items;
        std.mem.copyForwards(u8, items[0 .. items.len - n], items[n..]);
        c.pending.shrinkRetainingCapacity(items.len - n);
    }

    /// The byte length of the next chunk at the start of `text` (no leading whitespace), or null to wait.
    fn scan(c: *const Chunker, text: []const u8) ?usize {
        const first = c.emitted == 0;
        var i: usize = 0;
        var n: usize = 0; // code points in text[0..i]
        var last_clause: ?usize = null;
        var last_space: ?usize = null;
        while (i < text.len) {
            if (n == c.max_chars) {
                if (last_clause) |k| return k;
                if (last_space) |k| return k;
                return i;
            }
            const b = text[i];
            if (b == '\n') {
                if (first or n >= c.min_chars) return i;
                last_space = i;
            } else if (terminator(text, i)) |run| {
                const j = i + run + closing(text[i + run ..]);
                if (j >= text.len) return null; // the next character decides ("." may become "...")
                const cjk = for (cjk_ends) |e| {
                    if (std.mem.startsWith(u8, text[i..], e)) break true;
                } else false;
                const ends = (cjk or isSpace(text[j])) and !(run == 1 and b == '.' and abbreviation(text[0..i]));
                const len = n + codePoints(text[i..j]);
                if (ends and len <= c.max_chars and (first or len >= c.min_chars)) return j;
            } else if ((b == ',' or b == ';' or b == ':') and i + 1 < text.len and isSpace(text[i + 1])) {
                last_clause = i + 1;
                if (first and n + 1 >= c.clause_chars) return i + 1;
            } else if (isSpace(b)) {
                last_space = i;
            }
            const step: usize = std.unicode.utf8ByteSequenceLength(b) catch 1;
            if (i + step > text.len) break; // a code point cut by the delta: wait for the rest
            i += step;
            n += 1;
        }
        return null;
    }
};

fn isSpace(b: u8) bool {
    return std.ascii.isWhitespace(b);
}

/// The byte length of a sentence-ending run (". ! ? …" repeated, or one CJK mark) at `text[i]`, or null.
fn terminator(text: []const u8, i: usize) ?usize {
    for (cjk_ends) |e| if (std.mem.startsWith(u8, text[i..], e)) return e.len;
    var j = i;
    while (j < text.len) {
        if (text[j] == '.' or text[j] == '!' or text[j] == '?') {
            j += 1;
        } else if (std.mem.startsWith(u8, text[j..], ellipsis)) {
            j += ellipsis.len;
        } else break;
    }
    return if (j > i) j - i else null;
}

/// The byte length of the closing quotes and brackets at the start of `text`.
fn closing(text: []const u8) usize {
    var j: usize = 0;
    outer: while (j < text.len) {
        for (closers) |q| {
            if (std.mem.startsWith(u8, text[j..], q)) {
                j += q.len;
                continue :outer;
            }
        }
        break;
    }
    return j;
}

/// Whether the word before a period (`before` ends at the dot) is an abbreviation or a one-letter initial.
fn abbreviation(before: []const u8) bool {
    var s = before.len;
    while (s > 0 and (std.ascii.isAlphabetic(before[s - 1]) or before[s - 1] == '.')) s -= 1;
    const word = before[s..];
    if (word.len == 1 and std.ascii.isAlphabetic(word[0])) return true;
    for (abbreviations) |w| if (std.ascii.eqlIgnoreCase(word, w)) return true;
    return false;
}

fn codePoints(s: []const u8) usize {
    var n: usize = 0;
    for (s) |b| n += @intFromBool((b & 0xC0) != 0x80);
    return n;
}

/// Text as it should be spoken: markdown emphasis (* _ ** __), heading #s, code backticks, list bullets at line start and `[label](url)` -> label removed; whitespace collapsed to single spaces; trimmed. Kitten markup is kept: a leading `[emotion]`, `<event>`, `(((emphasis)))`.
pub fn speakable(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    var line_start = true;
    var link_end: ?[2]usize = null; // the `](url)` of a link whose label is being copied: from, to
    while (i < text.len) {
        if (line_start) {
            while (i < text.len and (text[i] == ' ' or text[i] == '\t')) i += 1;
            i += lineMarker(text[i..]);
            line_start = false;
            continue;
        }
        if (link_end) |l| {
            if (i == l[0]) {
                i = l[1];
                link_end = null;
                continue;
            }
        }
        const b = text[i];
        if (isSpace(b)) {
            if (out.items.len > 0 and out.items[out.items.len - 1] != ' ') try out.append(a, ' ');
            line_start = b == '\n';
            i += 1;
        } else if (b == '`') {
            i += 1;
        } else if (b == '*' or b == '_') {
            var j = i;
            while (j < text.len and text[j] == b) j += 1;
            const prev: u8 = if (i > 0) text[i - 1] else ' ';
            const after: u8 = if (j < text.len) text[j] else ' ';
            const keep = if (b == '*') isSpace(prev) and isSpace(after) else std.ascii.isAlphanumeric(prev) and std.ascii.isAlphanumeric(after);
            if (keep) try out.appendSlice(a, text[i..j]);
            i = j;
        } else if ((b == '[' or (b == '!' and i + 1 < text.len and text[i + 1] == '[')) and link_end == null) {
            const open = if (b == '!') i + 1 else i;
            if (link(text, open)) |l| {
                link_end = l;
                i = open + 1;
            } else {
                try out.append(a, b);
                i += 1;
            }
        } else {
            try out.append(a, b);
            i += 1;
        }
    }
    const trimmed = std.mem.trimEnd(u8, out.items, " ").len;
    out.shrinkRetainingCapacity(trimmed);
    return out.toOwnedSlice(a);
}

/// The byte length of a heading's #s or a list bullet (with its space) at the start of a line.
fn lineMarker(line: []const u8) usize {
    var j: usize = 0;
    while (j < line.len and line[j] == '#') j += 1;
    if (j > 0) return if (j < line.len and line[j] == ' ') j + 1 else 0;
    for ([_][]const u8{ "- ", "* ", "+ ", "• " }) |m| if (std.mem.startsWith(u8, line, m)) return m.len;
    return 0;
}

/// For a `[label](url)` whose `[` is at `open`: where `](url)` starts and ends; null when it is not a link.
fn link(text: []const u8, open: usize) ?[2]usize {
    var k = open + 1;
    while (k < text.len and text[k] != ']') : (k += 1) if (text[k] == '[' or text[k] == '\n') return null;
    if (k == open + 1 or k + 1 >= text.len or text[k + 1] != '(') return null;
    var m = k + 2;
    while (m < text.len and text[m] != ')') : (m += 1) if (isSpace(text[m])) return null;
    if (m >= text.len) return null;
    return .{ k, m + 1 };
}

const stray_close = "</think>";

/// The start of a reply, held until it is clear that no reasoning leaked into it. Some models write their reasoning as
/// plain text closed by a `</think>` that was never opened: everything before such a tag is dropped with it. The hold
/// ends at that tag, once the first sentence end is followed by text that cannot begin one (the observed leak is a whole
/// answer, then "\n</think>\n\n" and the answer again), at `max` characters, or at the end of the reply; later stray
/// tags are removed. The cost is about one token past the first sentence.
pub const Lead = struct {
    max: usize = 240,
    deciding: bool = true,
    held: std.ArrayList(u8) = .empty,

    pub fn deinit(l: *Lead, gpa: std.mem.Allocator) void {
        l.held.deinit(gpa);
    }

    /// What may be shown and spoken now (in `a`); `last` at the end of the reply.
    pub fn pass(l: *Lead, gpa: std.mem.Allocator, a: std.mem.Allocator, text: []const u8, last: bool) ![]const u8 {
        try l.held.appendSlice(gpa, text);
        if (l.deciding) {
            if (std.mem.indexOf(u8, l.held.items, stray_close)) |at| {
                l.deciding = false;
                const rest = std.mem.trimStart(u8, l.held.items[at + stray_close.len ..], &std.ascii.whitespace);
                std.mem.copyForwards(u8, l.held.items[0..rest.len], rest);
                l.held.shrinkRetainingCapacity(rest.len);
            } else if (!last and l.held.items.len < l.max and !settled(l.held.items)) {
                return "";
            } else l.deciding = false;
        }
        var out: std.ArrayList(u8) = .empty;
        var from: usize = 0;
        while (std.mem.indexOfPos(u8, l.held.items, from, stray_close)) |at| {
            try out.appendSlice(a, l.held.items[from..at]);
            from = at + stray_close.len;
        }
        const keep: usize = if (last) 0 else strayPrefix(l.held.items[from..]);
        try out.appendSlice(a, l.held.items[from .. l.held.items.len - keep]);
        const tail = l.held.items.len - keep;
        std.mem.copyForwards(u8, l.held.items[0..keep], l.held.items[tail..]);
        l.held.shrinkRetainingCapacity(keep);
        return out.items;
    }
};

/// Whether a first sentence has ended (a mark then whitespace, or a line break) and what follows it is not, or cannot
/// become, a `</think>`.
fn settled(h: []const u8) bool {
    var from: usize = 0;
    while (std.mem.indexOfAnyPos(u8, h, from, ".!?\n")) |i| {
        from = i + 1;
        if (h[i] != '\n' and (from == h.len or !std.ascii.isWhitespace(h[from]))) continue; // "3.5", "e.g": not an end
        const rest = std.mem.trimStart(u8, h[from..], &std.ascii.whitespace);
        return rest.len > 0 and !std.mem.startsWith(u8, stray_close, rest[0..@min(rest.len, stray_close.len)]);
    }
    return false;
}

/// The length of the longest end of `h` that could be the start of `</think>`.
fn strayPrefix(h: []const u8) usize {
    var k: usize = @min(stray_close.len - 1, h.len);
    while (k > 0) : (k -= 1) {
        if (std.mem.eql(u8, h[h.len - k ..], stray_close[0..k])) return k;
    }
    return 0;
}

test "a reply's lead: leaked reasoning before a stray </think> is dropped; a clean start waits only a few characters" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var leak: Lead = .{};
    defer leak.deinit(gpa);
    try std.testing.expectEqualStrings("", try leak.pass(gpa, a, "The capital of France is Paris.\n", false));
    try std.testing.expectEqualStrings("", try leak.pass(gpa, a, "</thi", false));
    try std.testing.expectEqualStrings("The capital of France is Paris.", try leak.pass(gpa, a, "nk>\n\nThe capital of France is Paris.", false));
    try std.testing.expectEqualStrings(" More </th", try leak.pass(gpa, a, " More </th", true));
    var clean: Lead = .{};
    defer clean.deinit(gpa);
    try std.testing.expectEqualStrings("", try clean.pass(gpa, a, "Hello there! ", false));
    try std.testing.expectEqualStrings("Hello there! How are you doing today, my friend?", try clean.pass(gpa, a, "How are you doing today, my friend?", false));
    try std.testing.expectEqualStrings(" Fine", try clean.pass(gpa, a, " Fine</", false));
    try std.testing.expectEqualStrings(".", try clean.pass(gpa, a, "think>.", false));
    var short: Lead = .{};
    defer short.deinit(gpa);
    try std.testing.expectEqualStrings("", try short.pass(gpa, a, "Paris.", false));
    try std.testing.expectEqualStrings("Paris.", try short.pass(gpa, a, "", true));
    var quick: Lead = .{};
    defer quick.deinit(gpa);
    try std.testing.expectEqualStrings("", try quick.pass(gpa, a, "It is 3.5 km. ", false)); // "3.5" is no sentence end
    try std.testing.expectEqualStrings("", try quick.pass(gpa, a, "<", false)); // may still become </think>
    try std.testing.expectEqualStrings("It is 3.5 km. <b", try quick.pass(gpa, a, "b", false));
}

/// Drops <think>…</think> spans from a stream of deltas (tags may be split across deltas).
pub const ThinkFilter = struct {
    inside: bool = false,
    held: std.ArrayList(u8) = .empty, // a suffix that may be the start of a tag

    const open_tag = "<think>";
    const close_tag = "</think>";

    /// Frees what is held back.
    pub fn deinit(f: *ThinkFilter, gpa: std.mem.Allocator) void {
        f.held.deinit(gpa);
    }

    /// The visible part of `delta`, owned by `a` (may be empty).
    pub fn filter(f: *ThinkFilter, gpa: std.mem.Allocator, a: std.mem.Allocator, delta: []const u8) ![]const u8 {
        try f.held.appendSlice(gpa, delta);
        const buf = f.held.items;
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        var i: usize = 0;
        var keep: usize = 0; // bytes at the end of buf held for the next delta
        while (i < buf.len) {
            const tag: []const u8 = if (f.inside) close_tag else open_tag;
            if (std.mem.findPos(u8, buf, i, tag)) |p| {
                if (!f.inside) try out.appendSlice(a, buf[i..p]);
                f.inside = !f.inside;
                i = p + tag.len;
                continue;
            }
            keep = partialTag(buf[i..], tag);
            if (!f.inside) try out.appendSlice(a, buf[i .. buf.len - keep]);
            break;
        }
        std.mem.copyForwards(u8, buf[0..keep], buf[buf.len - keep ..]);
        f.held.shrinkRetainingCapacity(keep);
        return out.toOwnedSlice(a);
    }

    /// Whatever was held back at the end of the stream (visible text only), owned by `a`.
    pub fn flush(f: *ThinkFilter, gpa: std.mem.Allocator, a: std.mem.Allocator) ![]const u8 {
        defer {
            f.held.clearAndFree(gpa);
            f.inside = false;
        }
        if (f.inside) return a.dupe(u8, "");
        return a.dupe(u8, f.held.items);
    }

    /// The length of the longest suffix of `s` that is a proper prefix of `tag`.
    fn partialTag(s: []const u8, tag: []const u8) usize {
        var k: usize = @min(s.len, tag.len - 1);
        while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, s, tag[0..k])) return k;
        return 0;
    }
};

const testing = std.testing;

/// Streams `text` in `size`-byte deltas through a chunker and returns every chunk, `finish` included.
fn chunks(a: std.mem.Allocator, c: *Chunker, text: []const u8, size: usize) ![]const []const u8 {
    var got: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (at < text.len) : (at += size) {
        try c.push(text[at..@min(at + size, text.len)]);
        while (try c.next(a)) |s| try got.append(a, s);
    }
    if (try c.finish(a)) |s| try got.append(a, s);
    return got.items;
}

fn expectChunks(want: []const []const u8, got: []const []const u8) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectEqualStrings(w, g);
}

test "first sentence alone, later sentences merged up to min_chars" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Chunker = .{ .gpa = testing.allocator };
    defer c.deinit();
    // "How are you doing today?" is 24 characters, under 40: merged with the next sentence, which ends the reply.
    try expectChunks(&.{ "Hello there!", "How are you doing today? I am fine." }, try chunks(a, &c, "Hello there! How are you doing today? I am fine.", 3));
    var d: Chunker = .{ .gpa = testing.allocator };
    defer d.deinit();
    const long = "Hi. This sentence is long enough to stand alone here. Short one. And this one closes the reply now.";
    try expectChunks(&.{ "Hi.", "This sentence is long enough to stand alone here.", "Short one. And this one closes the reply now." }, try chunks(a, &d, long, 1));
}

test "a terminator waits for the next character; finish takes the rest" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Chunker = .{ .gpa = testing.allocator };
    defer c.deinit();
    try c.push("Hello");
    try c.push(".");
    try testing.expect(try c.next(a) == null);
    try c.push("..");
    try testing.expect(try c.next(a) == null);
    try c.push(" Then more");
    try testing.expectEqualStrings("Hello...", (try c.next(a)).?);
    try testing.expect(try c.next(a) == null);
    try testing.expectEqualStrings("Then more", (try c.finish(a)).?);
    try c.push("  \n ");
    try testing.expect(try c.finish(a) == null);
}

test "abbreviations, initials, decimals and quotes" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { text: []const u8, want: []const []const u8 }{
        .{ .text = "Dr. Smith is here. Yes.", .want = &.{ "Dr. Smith is here.", "Yes." } },
        .{ .text = "J. R. R. Tolkien wrote e.g. the Hobbit. Ok.", .want = &.{ "J. R. R. Tolkien wrote e.g. the Hobbit.", "Ok." } },
        .{ .text = "Mrs. Jones vs. Prof. Plum etc. Fine. Ok.", .want = &.{ "Mrs. Jones vs. Prof. Plum etc. Fine.", "Ok." } },
        .{ .text = "Pi is 3.14 or so. Ok.", .want = &.{ "Pi is 3.14 or so.", "Ok." } },
        .{ .text = "She said \"stop.\" Then left.", .want = &.{ "She said \"stop.\"", "Then left." } },
        .{ .text = "Really?! Yes.", .want = &.{ "Really?!", "Yes." } },
    };
    for (cases) |k| {
        var c: Chunker = .{ .gpa = testing.allocator };
        defer c.deinit();
        try expectChunks(k.want, try chunks(a, &c, k.text, 2));
    }
}

test "the first chunk may end at a clause once clause_chars long" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Chunker = .{ .gpa = testing.allocator };
    defer c.deinit();
    // "Well," is under 24 characters, so the break is the comma after "question," (42 characters); later clauses don't break.
    const text = "Well, that is a very interesting question, and the answer, as you know, is long.";
    try expectChunks(&.{ "Well, that is a very interesting question,", "and the answer, as you know, is long." }, try chunks(a, &c, text, 5));
}

test "a long run without punctuation breaks at a space under max_chars" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Chunker = .{ .gpa = testing.allocator };
    defer c.deinit();
    const text = try repeat(a, "word ", 120); // 600 characters
    const got = try chunks(a, &c, text, 7);
    try testing.expectEqual(@as(usize, 3), got.len);
    var joined: std.ArrayList(u8) = .empty;
    for (got, 0..) |s, i| {
        try testing.expect(s.len <= c.max_chars and std.mem.endsWith(u8, s, "word"));
        if (i > 0) try joined.append(a, ' ');
        try joined.appendSlice(a, s);
    }
    try testing.expectEqualStrings(std.mem.trimEnd(u8, text, " "), joined.items);
    // A later chunk with a clause mark before the limit breaks there instead.
    var d: Chunker = .{ .gpa = testing.allocator, .emitted = 1 };
    defer d.deinit();
    const head = try std.mem.concat(a, u8, &.{ try repeat(a, "aaaa ", 10), "x," });
    const clause = try chunks(a, &d, try std.mem.concat(a, u8, &.{ head, " ", try repeat(a, "bbbb ", 60) }), 4);
    try testing.expectEqualStrings(head, clause[0]);
}

fn repeat(a: std.mem.Allocator, s: []const u8, n: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (0..n) |_| try out.appendSlice(a, s);
    return out.items;
}

test "CJK marks end sentences without a space; newlines end sentences" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Chunker = .{ .gpa = testing.allocator };
    defer c.deinit();
    const cjk = "你好。今天天气很好！我们走吧？";
    try expectChunks(&.{ "你好。", "今天天气很好！我们走吧？" }, try chunks(a, &c, cjk, 2));
    var d: Chunker = .{ .gpa = testing.allocator, .min_chars = 1 };
    defer d.deinit();
    try expectChunks(&.{ "你好。", "今天天气很好！", "我们走吧？" }, try chunks(a, &d, cjk, 4));
    var e: Chunker = .{ .gpa = testing.allocator };
    defer e.deinit();
    const lines = "Here is a list\n- the first item of the list, quite a long one\n- second\n\nThe end";
    try expectChunks(&.{ "Here is a list", "- the first item of the list, quite a long one", "- second\n\nThe end" }, try chunks(a, &e, lines, 3));
}

test "speakable strips markdown and keeps Kitten markup" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "**bold** and `code` and [a link](http://x)", "bold and code and a link" },
        .{ "[joyful] Hi <laugh> (((wow)))", "[joyful] Hi <laugh> (((wow)))" },
        .{ "# Title\n- item one\n  * item two\n\n__Under__ and _it_ and snake_case, 2 * 3.", "Title item one item two Under and it and snake_case, 2 * 3." },
        .{ "See ![a cat](http://x/c.png) or [x] (y)\t\t ok  ", "See a cat or [x] (y) ok" },
        .{ "  \n ", "" },
    };
    for (cases) |k| try testing.expectEqualStrings(k[1], try speakable(a, k[0]));
}

test "ThinkFilter drops think spans split across deltas" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: ThinkFilter = .{};
    defer f.deinit(testing.allocator);
    var seen: std.ArrayList(u8) = .empty;
    for ([_][]const u8{ "<thi", "nk>secret</th", "ink>Visible" }) |d| try seen.appendSlice(a, try f.filter(testing.allocator, a, d));
    try seen.appendSlice(a, try f.flush(testing.allocator, a));
    try testing.expectEqualStrings("Visible", seen.items);

    var g: ThinkFilter = .{};
    defer g.deinit(testing.allocator);
    try testing.expectEqualStrings("x < y and <b>", try g.filter(testing.allocator, a, "x < y and <b>"));
    try testing.expectEqualStrings("tail ", try g.filter(testing.allocator, a, "tail <"));
    try testing.expectEqualStrings("<", try g.flush(testing.allocator, a));

    var h: ThinkFilter = .{};
    defer h.deinit(testing.allocator);
    try testing.expectEqualStrings("a b", try h.filter(testing.allocator, a, "a <think>x</think>b<think>never closed"));
    try testing.expectEqualStrings("", try h.flush(testing.allocator, a));
}
