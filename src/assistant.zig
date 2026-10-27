//! The assistant's name (`api/assistant.zig`): what the user calls their assistant, chosen when they set up a client
//! and changed later in any client's settings. One per box: the voice loop introduces itself with it and every client
//! shows it. The config's `assistant.name` is the default ("Jarvis"); a name set over the API is saved to
//! `<data_dir>/assistant.json` (`{"name": "..."}`, mode 0600) and read at start, so an install without that file keeps
//! the default and nothing asks.

const std = @import("std");
const Io = std.Io;

pub const default_name = "Jarvis";
/// The longest name, in characters.
pub const max_chars = 32;
const max_bytes = max_chars * 4;
const file_name = "assistant.json";

pub const Error = error{ WriteFailed, InvalidName };

pub const Store = struct {
    dir: []const u8 = "", // "": nothing is saved to disk (tests, a daemon without a data directory)
    configured: []const u8 = default_name, // the config's name, used while none is set
    mutex: Io.Mutex = .init,
    buf: [max_bytes]u8 = undefined,
    len: usize = 0, // 0: none set, the configured one applies

    /// Reads the saved name in `dir`, if any; a file that does not hold a valid one is ignored with a warning.
    pub fn init(s: *Store, io: Io, gpa: std.mem.Allocator, dir: []const u8, configured: []const u8) void {
        s.dir = dir;
        s.configured = configured;
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const p = std.fmt.bufPrint(&path_buf, "{s}/" ++ file_name, .{dir}) catch return;
        const text = Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(4096)) catch |err| {
            if (err != error.FileNotFound) std.log.warn("assistant: {s}: {s}", .{ p, @errorName(err) });
            return;
        };
        defer gpa.free(text);
        const saved = std.json.parseFromSlice(struct { name: []const u8 }, gpa, text, .{ .ignore_unknown_fields = true }) catch {
            std.log.warn("assistant: {s} does not hold a name; using \"{s}\"", .{ p, configured });
            return;
        };
        defer saved.deinit();
        if (!s.keep(saved.value.name)) std.log.warn("assistant: {s}: not a valid name; using \"{s}\"", .{ p, configured });
    }

    /// The name now, copied into `a`.
    pub fn name(s: *Store, io: Io, a: std.mem.Allocator) error{OutOfMemory}![]const u8 {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        return a.dupe(u8, if (s.len > 0) s.buf[0..s.len] else s.configured);
    }

    /// Whether a name was set (over the API), rather than the configured default.
    pub fn custom(s: *Store, io: Io) bool {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        return s.len > 0;
    }

    /// Sets the name (`normalize`d) and saves it; "" (after trimming) goes back to the configured one and deletes the file.
    pub fn set(s: *Store, io: Io, raw: []const u8) Error!void {
        var tmp: [max_bytes]u8 = undefined;
        const n = normalize(&tmp, raw) orelse return error.InvalidName;
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        if (n.len == 0) {
            s.len = 0;
            if (s.dir.len > 0) {
                var path_buf: [std.fs.max_path_bytes]u8 = undefined;
                const p = std.fmt.bufPrint(&path_buf, "{s}/" ++ file_name, .{s.dir}) catch return error.WriteFailed;
                Io.Dir.cwd().deleteFile(io, p) catch |err| if (err != error.FileNotFound) return error.WriteFailed;
            }
            return;
        }
        try s.write(io, n);
        @memcpy(s.buf[0..n.len], n);
        s.len = n.len;
    }

    fn keep(s: *Store, raw: []const u8) bool {
        var tmp: [max_bytes]u8 = undefined;
        const n = normalize(&tmp, raw) orelse return false;
        if (n.len == 0) return false;
        @memcpy(s.buf[0..n.len], n);
        s.len = n.len;
        return true;
    }

    /// `{"name": n}` to `<dir>/assistant.json` through a temporary file (0600), renamed over it.
    fn write(s: *Store, io: Io, n: []const u8) Error!void {
        if (s.dir.len == 0) return;
        var text_buf: [max_bytes * 6 + 32]u8 = undefined;
        var w: Io.Writer = .fixed(&text_buf);
        std.json.Stringify.value(.{ .name = n }, .{}, &w) catch return error.WriteFailed;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dest = std.fmt.bufPrint(&buf, "{s}/" ++ file_name, .{s.dir}) catch return error.WriteFailed;
        const tmp = std.fmt.bufPrint(&tmp_buf, "{s}/." ++ file_name ++ ".tmp", .{s.dir}) catch return error.WriteFailed;
        const cwd = Io.Dir.cwd();
        cwd.createDirPath(io, s.dir) catch return error.WriteFailed;
        cwd.deleteFile(io, tmp) catch {};
        const f = cwd.createFile(io, tmp, .{ .permissions = .fromMode(0o600) }) catch return error.WriteFailed;
        {
            defer f.close(io);
            f.writeStreamingAll(io, w.buffered()) catch return error.WriteFailed;
            f.sync(io) catch return error.WriteFailed;
        }
        cwd.rename(tmp, cwd, dest, io) catch return error.WriteFailed;
    }
};

/// `raw` trimmed with its runs of spaces made one, into `out`; "" for a blank one. Null when it is not a name: over
/// `max_chars` characters, not UTF-8, without a letter or digit, or holding anything but letters, digits, spaces,
/// apostrophes, hyphens and full stops (every character past ASCII counts as a letter). It goes into a system prompt
/// and onto screens, so no line breaks, quotes, brackets or markup.
pub fn normalize(out: *[max_bytes]u8, raw: []const u8) ?[]const u8 {
    const t = std.mem.trim(u8, raw, " \t");
    if (!std.unicode.utf8ValidateSlice(t)) return null;
    var n: usize = 0;
    var chars: usize = 0;
    var any = false;
    var i: usize = 0;
    while (i < t.len) {
        const len = std.unicode.utf8ByteSequenceLength(t[i]) catch return null;
        const c = t[i];
        if (len == 1) {
            if (c == ' ' or c == '\t') {
                if (n > 0 and out[n - 1] == ' ') {
                    i += 1;
                    continue;
                }
            } else if (std.ascii.isAlphanumeric(c)) {
                any = true;
            } else if (c != '\'' and c != '-' and c != '.') return null;
        } else {
            const cp = std.unicode.utf8Decode(t[i..][0..len]) catch return null;
            if (cp < 0xa0 or (cp >= 0x2000 and cp <= 0x206f and cp != 0x2019) or cp == 0xfeff) return null; // C1 controls, spacing and format marks; ’ is an apostrophe
            if (cp != 0x2019) any = true;
        }
        chars += 1;
        if (chars > max_chars) return null;
        const piece = if (len == 1 and (c == '\t')) " " else t[i..][0..len];
        @memcpy(out[n..][0..piece.len], piece);
        n += piece.len;
        i += len;
    }
    if (n == 0) return "";
    return if (any) out[0..n] else null;
}

/// `text` with every `{name}` replaced by `n`, in `a`.
pub fn fill(a: std.mem.Allocator, text: []const u8, n: []const u8) ![]const u8 {
    return std.mem.replaceOwned(u8, a, text, "{name}", n);
}

/// The system prompt a voice reply runs with: `instructions` with `{name}` filled, led by "You are <name>." when they
/// do not mention the name themselves, so the assistant knows what it is called whatever a client sent. Blank
/// instructions stay blank: a client that sends none on purpose gets no system message.
pub fn systemPrompt(a: std.mem.Allocator, instructions: []const u8, n: []const u8) ![]const u8 {
    if (std.mem.trim(u8, instructions, " \t\r\n").len == 0) return instructions;
    const filled = try fill(a, instructions, n);
    if (n.len == 0 or std.ascii.findIgnoreCase(filled, n) != null) return filled;
    return std.fmt.allocPrint(a, "You are {s}. {s}", .{ n, filled });
}

test "names are trimmed and checked" {
    const t = std.testing;
    var b: [max_bytes]u8 = undefined;
    try t.expectEqualStrings("Friday", normalize(&b, "  Friday ").?);
    try t.expectEqualStrings("Mary Jane", normalize(&b, "Mary   Jane").?);
    try t.expectEqualStrings("J.A.R.V.I.S.", normalize(&b, "J.A.R.V.I.S.").?);
    try t.expectEqualStrings("O'Neil-2", normalize(&b, "O'Neil-2").?);
    try t.expectEqualStrings("Zoë", normalize(&b, "Zoë").?);
    try t.expectEqualStrings("小助手", normalize(&b, "小助手").?);
    try t.expectEqualStrings("", normalize(&b, "   ").?);
    for ([_][]const u8{ "a\nb", "{name}", "<b>", "x\"y", "...", "--", "a\x00", "ab\xff", "a\u{200b}b", "abcdefghijklmnopqrstuvwxyzabcdefg" }) |bad|
        try t.expect(normalize(&b, bad) == null);
    try t.expect(normalize(&b, "abcdefghijklmnopqrstuvwxyzabcdef") != null); // 32
}

test "the system prompt names the assistant" {
    const t = std.testing;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqualStrings("You are Friday, a helper.", try systemPrompt(a, "You are {name}, a helper.", "Friday"));
    try t.expectEqualStrings("You are Friday. Be brief.", try systemPrompt(a, "Be brief.", "Friday"));
    try t.expectEqualStrings("Call me friday.", try systemPrompt(a, "Call me friday.", "Friday"));
    try t.expectEqualStrings("", try systemPrompt(a, "", "Friday"));
}

test "a set name is saved and read back; reset goes back to the configured one" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var fresh: Store = .{};
    fresh.init(io, t.allocator, dir, default_name); // an existing install: no file, the default, nothing set
    try t.expectEqualStrings("Jarvis", try fresh.name(io, a));
    try t.expect(!fresh.custom(io));
    try fresh.set(io, " Friday ");
    try t.expectEqualStrings("Friday", try fresh.name(io, a));
    try t.expectError(error.InvalidName, fresh.set(io, "a\nb"));
    try t.expectEqualStrings("Friday", try fresh.name(io, a));

    var again: Store = .{};
    again.init(io, t.allocator, dir, default_name); // a restart reads it back
    try t.expectEqualStrings("Friday", try again.name(io, a));
    try t.expect(again.custom(io));
    try again.set(io, "");
    try t.expectEqualStrings("Jarvis", try again.name(io, a));

    var reset: Store = .{};
    reset.init(io, t.allocator, dir, "Ava"); // the file is gone: the configured name
    try t.expectEqualStrings("Ava", try reset.name(io, a));
    try t.expect(!reset.custom(io));
}
