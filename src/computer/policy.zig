//! What a computer action needs before the desktop helper may do it (docs/dev/COMPUTER.md): nothing (`auto`), the
//! user's tap in the web UI (`tap`), a yes/no judgment from the decision provider first (`judge`), or nothing at all
//! because it is not an action this feature offers (`refused`). Pure: no I/O, no clock. The rules only ever add taps:
//! a label the lexicon misses still goes to the judgment, and a judgment that fails is a tap.

const std = @import("std");

pub const Verdict = enum { auto, tap, judge, refused };

pub const Decision = struct {
    verdict: Verdict,
    /// For the approval card and the audit: why it needs a tap or was refused ("its label reads as sending").
    why: []const u8 = "",
};

/// The fixed controls the helper implements (`control` frames).
pub const ControlKind = enum { open_app, quit_app, volume, media, dark_mode, lock_screen };

/// A fixed control: the simple ones run without a tap, locking the screen asks first, anything else is refused.
pub fn control(kind: []const u8) Decision {
    const k = std.meta.stringToEnum(ControlKind, kind) orelse return .{ .verdict = .refused, .why = "not one of the fixed controls (open_app, quit_app, volume, media, dark_mode, lock_screen)" };
    return switch (k) {
        .open_app, .quit_app, .volume, .media, .dark_mode => .{ .verdict = .auto },
        .lock_screen => .{ .verdict = .tap, .why = "locking the screen" },
    };
}

/// One on-screen step the helper asks to take (`authorize` frames). Strings are the helper's, from the screen: data.
pub const Action = struct {
    kind: []const u8, // click, type, press, scroll, surface
    ref: []const u8 = "",
    role: []const u8 = "",
    label: []const u8 = "",
    text: []const u8 = "", // type: the exact text
    key: []const u8 = "", // press: enter, escape, tab, space
    field_role: []const u8 = "", // press: the box typed into before it, if any
    field_label: []const u8 = "",
    direction: []const u8 = "", // scroll: up, down, top, bottom
    what: []const u8 = "", // surface: tab or app
    name: []const u8 = "", // surface: the tab's or app's name
};

pub const max_label = 400;
pub const max_text = 4000;

/// The rule for an on-screen step. `judge` means: ask the decision provider whether it could send, buy, delete, post or
/// submit something, then `afterJudge`.
pub fn screen(a: Action) Decision {
    if (a.label.len > max_label or a.name.len > max_label or a.field_label.len > max_label) return .{ .verdict = .refused, .why = "a label longer than any control's" };
    if (eql(a.kind, "scroll")) {
        for ([_][]const u8{ "up", "down", "top", "bottom" }) |d| if (eql(a.direction, d)) return .{ .verdict = .auto };
        return .{ .verdict = .refused, .why = "an unknown scroll direction" };
    }
    if (eql(a.kind, "surface")) {
        if (!eql(a.what, "tab") and !eql(a.what, "app")) return .{ .verdict = .refused, .why = "a surface that is neither a tab nor an app" };
        if (a.name.len == 0) return .{ .verdict = .refused, .why = "a surface with no name" };
        return .{ .verdict = .auto }; // bringing a named app or tab forward changes nothing in it
    }
    // A drag is always the real pointer, and a drop can move or delete: the person's to allow.
    if (eql(a.kind, "drag")) return .{ .verdict = .tap, .why = "it drags with your mouse" };
    if (eql(a.kind, "press")) {
        if (eql(a.key, "space") or eql(a.key, "escape") or eql(a.key, "tab")) return .{ .verdict = .auto };
        if (!eql(a.key, "enter")) return .{ .verdict = .refused, .why = "a key other than Enter, Space, Escape or Tab" };
        if (a.field_role.len > 0 and searchField(a.field_role, a.field_label)) return .{ .verdict = .auto };
        return .{ .verdict = .tap, .why = "pressing Enter could send or submit what was typed" };
    }
    if (eql(a.kind, "type")) {
        if (a.text.len == 0) return .{ .verdict = .refused, .why = "typing with no text" };
        if (a.text.len > max_text) return .{ .verdict = .refused, .why = "more text than one step may type" };
        if (std.mem.indexOfAny(u8, a.text, "\r\n") != null) return .{ .verdict = .tap, .why = "the text has a line break, which can send it" };
        if (searchField(a.role, a.label)) return .{ .verdict = .auto };
        return .{ .verdict = .tap, .why = "typing into a field that is not a search or address box" };
    }
    if (eql(a.kind, "click")) {
        if (consequential(a.label)) |what| return .{ .verdict = .tap, .why = what };
        // Focusing a text box, picking a tab or selecting a row in a list commits nothing.
        if (eql(a.role, "textbox") or eql(a.role, "tab") or eql(a.role, "row")) return .{ .verdict = .auto };
        return .{ .verdict = .judge };
    }
    return .{ .verdict = .refused, .why = "not a step computer use takes (click, type, press, scroll, surface)" };
}

/// The provider's judgment for a `judge` step: the probability that it sends, buys, deletes, posts or submits
/// something, or null when the provider could not answer. Unknown is a tap.
pub fn afterJudge(probability: ?f64) Decision {
    const p = probability orelse return .{ .verdict = .tap, .why = "whether it could send, buy or delete something could not be checked" };
    if (!std.math.isFinite(p) or p < 0 or p > 1) return .{ .verdict = .tap, .why = "whether it could send, buy or delete something could not be checked" };
    if (p >= judge_threshold) return .{ .verdict = .tap, .why = "it may send, buy, delete, post or submit something" };
    return .{ .verdict = .auto };
}

/// At or above this, a judged click asks first. Low on purpose: a needless tap costs a second, a wrong click cannot be undone.
pub const judge_threshold: f64 = 0.3;

/// A search, find, filter or address box: typing there and pressing Enter runs a search or opens a page.
pub fn searchField(role: []const u8, label: []const u8) bool {
    if (!eql(role, "textbox")) return false;
    var buf: [max_label]u8 = undefined;
    const l = lower(&buf, label);
    for (search_words) |w| if (std.mem.indexOf(u8, l, w) != null) return true;
    return false;
}

const search_words = [_][]const u8{ "search", "find", "filter", "address", "url", "go to", "buscar", "búsqueda", "rechercher", "recherche", "suchen", "suche", "cerca", "pesquisar", "zoeken", "検索", "搜索" };

/// Why a control's label needs a tap, or null. Whole words, ASCII case folded; several languages, since a miss here
/// still goes to the judgment.
pub fn consequential(label: []const u8) ?[]const u8 {
    var buf: [max_label]u8 = undefined;
    const l = lower(&buf, label);
    for (phrases) |p| if (std.mem.indexOf(u8, l, p.text) != null) return p.why;
    // ASCII separators only: a multi-byte one would split UTF-8 words at its continuation bytes.
    var it = std.mem.tokenizeAny(u8, l, " \t\r\n.,;:!?()[]{}\"'`/\\|-_+*&#@<>=~^%$");
    while (it.next()) |word| {
        for (lexicon) |e| for (e.words) |w| if (eql(word, w)) return e.why;
    }
    return null;
}

const Entry = struct { why: []const u8, words: []const []const u8 };

const lexicon = [_]Entry{
    .{ .why = "its label reads as sending or posting", .words = &.{ "send", "reply", "post", "publish", "share", "tweet", "comment", "forward", "enviar", "responder", "publicar", "compartir", "envoyer", "répondre", "publier", "partager", "senden", "antworten", "veröffentlichen", "teilen", "inviare", "rispondi", "pubblica", "condividi", "verzenden", "送信", "发送", "发布" } },
    .{ .why = "its label reads as buying or paying", .words = &.{ "buy", "purchase", "pay", "checkout", "order", "subscribe", "donate", "transfer", "book", "comprar", "pagar", "suscribirse", "acheter", "payer", "commander", "kaufen", "bezahlen", "bestellen", "abonnieren", "acquista", "paga", "ordina", "kopen", "betalen", "購入", "购买", "支付" } },
    .{ .why = "its label reads as deleting", .words = &.{ "delete", "remove", "erase", "discard", "trash", "archive", "uninstall", "wipe", "eliminar", "borrar", "supprimer", "effacer", "löschen", "entfernen", "elimina", "cancella", "verwijderen", "excluir", "削除", "删除" } },
    .{ .why = "its label reads as submitting or confirming", .words = &.{ "submit", "confirm", "approve", "merge", "sign", "accept", "agree", "authorize", "authorise", "enviar", "confirmar", "aceptar", "valider", "confirmer", "accepter", "bestätigen", "absenden", "akzeptieren", "conferma", "accetta", "bevestigen", "確認", "确认", "提交" } },
};

const phrases = [_]struct { text: []const u8, why: []const u8 }{
    .{ .text = "check out", .why = "its label reads as buying or paying" },
    .{ .text = "place order", .why = "its label reads as buying or paying" },
    .{ .text = "add to cart", .why = "its label reads as buying or paying" },
    .{ .text = "empty trash", .why = "its label reads as deleting" },
    .{ .text = "move to trash", .why = "its label reads as deleting" },
};

/// ASCII-lowercases `s` into `buf` (clipped to it); other bytes are kept, so UTF-8 words still match.
fn lower(buf: []u8, s: []const u8) []const u8 {
    const n = @min(s.len, buf.len);
    for (s[0..n], buf[0..n]) |c, *o| o.* = std.ascii.toLower(c);
    return buf[0..n];
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

// ---------------------------------------------------------------- tests

const t = std.testing;

test "fixed controls: the simple ones run, locking asks, anything else is refused" {
    for ([_][]const u8{ "open_app", "quit_app", "volume", "media", "dark_mode" }) |k| try t.expectEqual(Verdict.auto, control(k).verdict);
    try t.expectEqual(Verdict.tap, control("lock_screen").verdict);
    try t.expectEqual(Verdict.refused, control("shell").verdict);
    try t.expectEqual(Verdict.refused, control("").verdict);
}

test "sending, buying, deleting and submitting labels need a tap, in several languages" {
    for ([_][]const u8{ "Send", "Reply all", "Post", "Buy now", "Place Order", "Proceed to Check out", "Delete", "Move to Trash", "Submit", "Confirm purchase", "Enviar", "Supprimer", "Löschen", "Jetzt kaufen", "送信" }) |label| {
        const d = screen(.{ .kind = "click", .role = "button", .label = label });
        try t.expectEqual(Verdict.tap, d.verdict);
        try t.expect(d.why.len > 0);
    }
}

test "words are matched whole: a label merely containing one is judged, not tapped" {
    // "Sender" and "Postcode" contain send/post but are not those words: the provider judges them.
    try t.expectEqual(Verdict.judge, screen(.{ .kind = "click", .role = "button", .label = "Sender details" }).verdict);
    try t.expectEqual(Verdict.judge, screen(.{ .kind = "click", .role = "link", .label = "Postcode lookup" }).verdict);
}

test "benign clicks: text boxes, tabs and rows run; other controls are judged" {
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "click", .role = "textbox", .label = "Message" }).verdict);
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "click", .role = "tab", .label = "Inbox" }).verdict);
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "click", .role = "row", .label = "Dana Lee" }).verdict);
    try t.expectEqual(Verdict.judge, screen(.{ .kind = "click", .role = "button", .label = "OK" }).verdict);
    try t.expectEqual(Verdict.judge, screen(.{ .kind = "click", .role = "text", .label = "Continue" }).verdict);
}

test "a label word wins over a benign role: a row labelled Delete asks" {
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "click", .role = "row", .label = "Delete conversation" }).verdict);
}

test "typing: a search or address box runs, any other field asks, a line break always asks" {
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "type", .role = "textbox", .label = "Search or type URL", .text = "weather" }).verdict);
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "type", .role = "textbox", .label = "Address and search bar", .text = "news.ycombinator.com" }).verdict);
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "type", .role = "textbox", .label = "Type a message", .text = "on my way" }).verdict);
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "type", .role = "textbox", .label = "", .text = "x" }).verdict);
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "type", .role = "textbox", .label = "Search", .text = "a\nb" }).verdict);
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "type", .role = "button", .label = "Search", .text = "a" }).verdict);
    try t.expectEqual(Verdict.refused, screen(.{ .kind = "type", .role = "textbox", .label = "Search", .text = "" }).verdict);
}

test "Enter: after a search box it runs; anywhere else it asks" {
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "press", .key = "enter", .field_role = "textbox", .field_label = "Search mail" }).verdict);
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "press", .key = "enter", .field_role = "textbox", .field_label = "Type a message" }).verdict);
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "press", .key = "enter" }).verdict);
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "press", .key = "space" }).verdict);
    try t.expectEqual(Verdict.refused, screen(.{ .kind = "press", .key = "cmd+q" }).verdict);
}

test "scrolling and bringing a named surface forward run; a drag asks; unknown steps are refused" {
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "scroll", .direction = "down" }).verdict);
    try t.expectEqual(Verdict.refused, screen(.{ .kind = "scroll", .direction = "sideways" }).verdict);
    try t.expectEqual(Verdict.auto, screen(.{ .kind = "surface", .what = "app", .name = "Safari" }).verdict);
    try t.expectEqual(Verdict.refused, screen(.{ .kind = "surface", .what = "app", .name = "" }).verdict);
    try t.expectEqual(Verdict.tap, screen(.{ .kind = "drag" }).verdict);
    try t.expectEqual(Verdict.refused, screen(.{ .kind = "shell", .text = "rm -rf ~" }).verdict);
}

test "a judgment: below the threshold runs, at or above asks, no answer or a bad number asks" {
    try t.expectEqual(Verdict.auto, afterJudge(0.1).verdict);
    try t.expectEqual(Verdict.tap, afterJudge(0.3).verdict);
    try t.expectEqual(Verdict.tap, afterJudge(0.9).verdict);
    try t.expectEqual(Verdict.tap, afterJudge(null).verdict);
    try t.expectEqual(Verdict.tap, afterJudge(std.math.nan(f64)).verdict);
    try t.expectEqual(Verdict.tap, afterJudge(-1).verdict);
}

test "an over-long label is refused rather than clipped past its consequential word" {
    var long: [604]u8 = undefined;
    @memset(long[0..600], ' ');
    @memcpy(long[600..], "Send");
    try t.expectEqual(Verdict.refused, screen(.{ .kind = "click", .role = "button", .label = &long }).verdict);
}

test "the golden cases shared with the device executor (apps/companion) all hold" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("policy_golden.json"), .{});
    for (root.object.get("control").?.array.items) |c| {
        const want = std.meta.stringToEnum(Verdict, c.object.get("verdict").?.string).?;
        try t.expectEqual(want, control(c.object.get("kind").?.string).verdict);
    }
    for (root.object.get("screen").?.array.items) |c| {
        const o = c.object.get("action").?.object;
        var act: Action = .{ .kind = o.get("kind").?.string };
        inline for (@typeInfo(Action).@"struct".field_names) |f| {
            if (o.get(f)) |v| @field(act, f) = v.string;
        }
        const want = std.meta.stringToEnum(Verdict, c.object.get("verdict").?.string).?;
        t.expectEqual(want, screen(act).verdict) catch |err| {
            std.debug.print("golden case {s} {s} {s}\n", .{ act.kind, act.role, act.label });
            return err;
        };
    }
    for (root.object.get("after_judge").?.array.items) |c| {
        const p: ?f64 = switch (c.object.get("p").?) {
            .null => null,
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => unreachable,
        };
        const want = std.meta.stringToEnum(Verdict, c.object.get("verdict").?.string).?;
        try t.expectEqual(want, afterJudge(p).verdict);
    }
}
