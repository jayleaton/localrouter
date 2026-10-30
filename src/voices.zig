//! Cloned voices (`api/voices.zig`): a voice made from someone's own recordings, spoken by the speech model that
//! encoded it with the references it wrote (`request.Speech.voice_references`). A saved voice is one file,
//! `<data_dir>/voices/<name>.json` (mode 0600), read once at start; a draft (made for a preview, not yet named) lives
//! in memory for `draft_ttl_s` only. Nothing leaves the machine: no clip is kept, only each reference's transcript and
//! codec codes, and deleting a voice deletes its file.

const std = @import("std");
const Io = std.Io;

pub const format = "localrouter-voice/1";
pub const draft_prefix = "draft-";
pub const draft_ttl_s: i64 = 3600;
pub const max_drafts = 8;

pub const Reference = struct { text: []const u8, seconds: f64 = 0, frames: u32 = 0, vq_codes: std.json.Value };
pub const Source = struct { kind: []const u8, note: []const u8 = "" };
/// Who said it is their voice (or that its owner consented), what they agreed to, and when.
pub const Consent = struct { statement: []const u8, by: []const u8 = "", at: i64 = 0 };
/// What preparation found in one clip (`media/voiceprep.zig`): `snr_db` is its speech over the background between the
/// words (null: too little of it to measure), `background` "noise" or "music", `warnings` what may make the voice worse.
pub const Clip = struct {
    seconds_in: f64,
    seconds_kept: f64,
    speech_left_out_s: f64 = 0,
    references: usize = 0,
    snr_db: ?f64 = null,
    background: ?[]const u8 = null,
    warnings: []const []const u8 = &.{},
};
pub const ReferenceInfo = struct { text: []const u8, seconds: f64, frames: u32, clip: usize = 0, level_dbfs: f64 = 0, gain_db: f64 = 0, snr_db: ?f64 = null };

/// A voice as stored (`references` holds the codes).
pub const Voice = struct {
    format: []const u8 = format,
    id: []const u8,
    model: []const u8,
    created_at: i64,
    source: Source,
    consent: Consent,
    transcribed: bool = false, // the transcription model wrote a transcript (else the caller gave them all)
    seconds: f64 = 0,
    clips: []const Clip = &.{},
    prep: []const ReferenceInfo = &.{}, // per reference: what preparation measured
    references: []const Reference,
};

/// A voice as listed: everything but its codes.
pub const Summary = struct {
    object: []const u8 = "voice",
    id: []const u8,
    name: []const u8,
    model: []const u8,
    kind: []const u8 = "cloned",
    status: []const u8, // ready, or draft
    created_at: i64,
    expires_at: ?i64 = null,
    seconds: f64,
    transcribed: bool,
    source: Source,
    clips: []const Clip,
    references: []const ReferenceInfo,
};

const Entry = struct {
    arena: *std.heap.ArenaAllocator,
    voice: Voice,
    refs_json: []const u8, // [{text, vq_codes}]: the job file a speech request in this voice carries
    expires: i64, // 0: saved
};

pub const Error = error{ NotFound, Taken, OutOfMemory, WriteFailed };

pub const Store = struct {
    dir: []const u8 = "", // "": nothing is saved to disk (tests, a daemon without a data directory)
    gpa: std.mem.Allocator = std.heap.page_allocator,
    mutex: Io.Mutex = .init,
    entries: std.ArrayList(Entry) = .empty,

    /// Reads every saved voice in `dir`; a file that does not parse is skipped with a warning.
    pub fn init(s: *Store, io: Io, gpa: std.mem.Allocator, dir: []const u8) !void {
        s.gpa = gpa;
        s.dir = dir;
        try Io.Dir.cwd().createDirPath(io, dir);
        var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".json") or e.name[0] == '.') continue;
            const text = d.readFileAlloc(io, e.name, gpa, .limited(16 << 20)) catch |err| {
                std.log.warn("voices: {s}: {s}", .{ e.name, @errorName(err) });
                continue;
            };
            defer gpa.free(text);
            s.adopt(text, 0) catch |err| std.log.warn("voices: {s} is not a voice file: {s}", .{ e.name, @errorName(err) });
        }
        if (s.entries.items.len > 0) std.log.info("voices: {d} cloned voice(s) in {s}", .{ s.entries.items.len, dir });
    }

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| free(s.gpa, e);
        s.entries.deinit(s.gpa);
    }

    fn free(gpa: std.mem.Allocator, e: Entry) void {
        e.arena.deinit();
        gpa.destroy(e.arena);
    }

    /// A voice parsed from its JSON into an entry of its own.
    fn adopt(s: *Store, text: []const u8, expires: i64) !void {
        const arena = try s.gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(s.gpa);
        errdefer {
            arena.deinit();
            s.gpa.destroy(arena);
        }
        const a = arena.allocator();
        const v = try std.json.parseFromSliceLeaky(Voice, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        if (!std.mem.eql(u8, v.format, format) or v.references.len == 0) return error.UnknownFormat;
        const Ref = struct { text: []const u8, vq_codes: std.json.Value };
        const refs = try a.alloc(Ref, v.references.len);
        for (refs, v.references) |*r, src| r.* = .{ .text = src.text, .vq_codes = src.vq_codes };
        try s.entries.append(s.gpa, .{ .arena = arena, .voice = v, .refs_json = try std.json.Stringify.valueAlloc(a, refs, .{}), .expires = expires });
    }

    fn find(s: *Store, id: []const u8) ?usize {
        for (s.entries.items, 0..) |e, i| if (std.mem.eql(u8, e.voice.id, id)) return i;
        return null;
    }

    fn prune(s: *Store, now: i64) void {
        var i: usize = 0;
        while (i < s.entries.items.len) {
            const e = s.entries.items[i];
            if (e.expires != 0 and e.expires <= now) {
                free(s.gpa, e);
                _ = s.entries.orderedRemove(i);
            } else i += 1;
        }
    }

    /// Whether `name` is any stored voice's or draft's, ignoring case.
    pub fn taken(s: *Store, io: Io, name: []const u8) bool {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        for (s.entries.items) |e| if (std.ascii.eqlIgnoreCase(e.voice.id, name)) return true;
        return false;
    }

    /// Stores `v` (in any allocator; copied), as a draft when `v.id` starts with `draft_prefix` (the oldest draft
    /// goes when there are `max_drafts`), else saved to its file. `Taken` when the id exists, ignoring case. The summary
    /// is in `a`.
    pub fn add(s: *Store, io: Io, a: std.mem.Allocator, v: Voice, now: i64) Error!Summary {
        const text = std.json.Stringify.valueAlloc(s.gpa, v, .{}) catch return error.OutOfMemory;
        defer s.gpa.free(text);
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.prune(now);
        for (s.entries.items) |e| if (std.ascii.eqlIgnoreCase(e.voice.id, v.id)) return error.Taken;
        const draft = std.mem.startsWith(u8, v.id, draft_prefix);
        if (draft) {
            var drafts: usize = 0;
            for (s.entries.items) |e| drafts += @intFromBool(e.expires != 0);
            if (drafts >= max_drafts) for (s.entries.items, 0..) |e, i| if (e.expires != 0) {
                free(s.gpa, e);
                _ = s.entries.orderedRemove(i);
                break;
            };
        } else try s.write(io, v.id, text);
        s.adopt(text, if (draft) now + draft_ttl_s else 0) catch return error.OutOfMemory;
        return summary(s.entries.items[s.entries.items.len - 1], a) catch error.OutOfMemory;
    }

    /// Saves draft `id` as `name`, or renames saved voice `id`; its file follows. The summary is in `a`.
    pub fn rename(s: *Store, io: Io, a: std.mem.Allocator, id: []const u8, name: []const u8, now: i64) Error!Summary {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.prune(now);
        const i = s.find(id) orelse return error.NotFound;
        for (s.entries.items, 0..) |e, k| if (k != i and std.ascii.eqlIgnoreCase(e.voice.id, name)) return error.Taken;
        const e = &s.entries.items[i];
        const ea = e.arena.allocator();
        var v = e.voice;
        v.id = ea.dupe(u8, name) catch return error.OutOfMemory;
        const text = std.json.Stringify.valueAlloc(s.gpa, v, .{}) catch return error.OutOfMemory;
        defer s.gpa.free(text);
        try s.write(io, name, text);
        if (e.expires == 0 and !std.mem.eql(u8, id, name)) s.unlink(io, id);
        e.voice = v;
        e.expires = 0;
        return summary(e.*, a) catch error.OutOfMemory;
    }

    /// Deletes voice or draft `id`, and its file.
    pub fn remove(s: *Store, io: Io, id: []const u8) Error!void {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.find(id) orelse return error.NotFound;
        const e = s.entries.orderedRemove(i);
        if (e.expires == 0) s.unlink(io, id);
        free(s.gpa, e);
    }

    /// The `[{text, vq_codes}]` of `model`'s voice or draft `id`, copied into `a`; null when it has none.
    pub fn references(s: *Store, io: Io, a: std.mem.Allocator, model: []const u8, id: []const u8, now: i64) ?[]const u8 {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.prune(now);
        const i = s.find(id) orelse return null;
        const e = s.entries.items[i];
        if (!std.mem.eql(u8, e.voice.model, model)) return null;
        return a.dupe(u8, e.refs_json) catch null;
    }

    /// Whether `model` has voice or draft `id`.
    pub fn has(s: *Store, io: Io, model: []const u8, id: []const u8) bool {
        if (id.len == 0) return false;
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const i = s.find(id) orelse return false;
        return std.mem.eql(u8, s.entries.items[i].voice.model, model);
    }

    /// The saved voices' names for `model` (null: every model's), in `a`.
    pub fn names(s: *Store, io: Io, a: std.mem.Allocator, model: ?[]const u8) ![]const []const u8 {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        var out: std.ArrayList([]const u8) = .empty;
        for (s.entries.items) |e| if (e.expires == 0 and (model == null or std.mem.eql(u8, model.?, e.voice.model))) try out.append(a, try a.dupe(u8, e.voice.id));
        return out.items;
    }

    /// The saved voices of `model` (null: all), or with `id` that voice or draft alone, as listed, in `a`.
    pub fn list(s: *Store, io: Io, a: std.mem.Allocator, model: ?[]const u8, id: ?[]const u8, now: i64) ![]const Summary {
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        s.prune(now);
        var out: std.ArrayList(Summary) = .empty;
        for (s.entries.items) |e| {
            if (id) |want| {
                if (std.mem.eql(u8, want, e.voice.id)) try out.append(a, try summary(e, a));
            } else if (e.expires == 0 and (model == null or std.mem.eql(u8, model.?, e.voice.model))) try out.append(a, try summary(e, a));
        }
        return out.items;
    }

    fn path(s: *Store, buf: []u8, name: []const u8) ?[]const u8 {
        return std.fmt.bufPrint(buf, "{s}/{s}.json", .{ s.dir, name }) catch null;
    }

    /// `text` to `<dir>/<name>.json` through a temporary file (0600), renamed over it.
    fn write(s: *Store, io: Io, name: []const u8, text: []const u8) Error!void {
        if (s.dir.len == 0) return;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dest = s.path(&buf, name) orelse return error.WriteFailed;
        const tmp = std.fmt.bufPrint(&tmp_buf, "{s}/.{s}.json.tmp", .{ s.dir, name }) catch return error.WriteFailed;
        const cwd = Io.Dir.cwd();
        cwd.createDirPath(io, s.dir) catch return error.WriteFailed;
        cwd.deleteFile(io, tmp) catch {};
        const f = cwd.createFile(io, tmp, .{ .permissions = .fromMode(0o600) }) catch return error.WriteFailed;
        {
            defer f.close(io);
            f.writeStreamingAll(io, text) catch return error.WriteFailed;
            f.sync(io) catch return error.WriteFailed;
        }
        cwd.rename(tmp, cwd, dest, io) catch return error.WriteFailed;
    }

    fn unlink(s: *Store, io: Io, name: []const u8) void {
        if (s.dir.len == 0) return;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const p = s.path(&buf, name) orelse return;
        Io.Dir.cwd().deleteFile(io, p) catch |err| std.log.warn("voices: could not delete {s}: {s}", .{ p, @errorName(err) });
    }
};

/// What `e` lists as, in `a`.
fn summary(e: Entry, a: std.mem.Allocator) !Summary {
    const v = e.voice;
    return .{
        .id = try a.dupe(u8, v.id),
        .name = try a.dupe(u8, v.id),
        .model = try a.dupe(u8, v.model),
        .status = if (e.expires == 0) "ready" else "draft",
        .created_at = v.created_at,
        .expires_at = if (e.expires == 0) null else e.expires,
        .seconds = v.seconds,
        .transcribed = v.transcribed,
        .source = .{ .kind = try a.dupe(u8, v.source.kind), .note = try a.dupe(u8, v.source.note) },
        .clips = try clipsOf(a, v),
        .references = try infos(a, v),
    };
}

fn clipsOf(a: std.mem.Allocator, v: Voice) ![]const Clip {
    const out = try a.dupe(Clip, v.clips);
    for (out) |*c| {
        if (c.background) |b| c.background = try a.dupe(u8, b);
        const ws = try a.alloc([]const u8, c.warnings.len);
        for (ws, c.warnings) |*o, w| o.* = try a.dupe(u8, w);
        c.warnings = ws;
    }
    return out;
}

fn infos(a: std.mem.Allocator, v: Voice) ![]const ReferenceInfo {
    const out = try a.alloc(ReferenceInfo, v.references.len);
    for (out, v.references, 0..) |*o, r, i| {
        o.* = if (i < v.prep.len) v.prep[i] else .{ .text = "", .seconds = r.seconds, .frames = r.frames };
        o.text = try a.dupe(u8, r.text);
        o.seconds = r.seconds;
        o.frames = r.frames;
    }
    return out;
}

/// A voice name: 1 to 64 of A-Z a-z 0-9 . _ - led by a letter or digit (the speech models' own rule), not `default`
/// and not a draft's.
pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64 or !std.ascii.isAlphanumeric(name[0])) return false;
    for (name) |c| if (!(std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-')) return false;
    return !std.ascii.eqlIgnoreCase(name, "default") and !std.ascii.startsWithIgnoreCase(name, draft_prefix);
}

test "drafts expire, saving names them on disk, deleting removes the file" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}/voices", .{&tmp.sub_path});
    var s: Store = .{};
    try s.init(io, t.allocator, dir);
    defer s.deinit();
    var codes = try std.json.parseFromSlice(std.json.Value, t.allocator, "[[1,2],[3,4]]", .{});
    defer codes.deinit();
    const refs = [_]Reference{.{ .text = "Hello there.", .seconds = 0.1, .frames = 2, .vq_codes = codes.value }};
    const v: Voice = .{ .id = "draft-0001", .model = "m", .created_at = 100, .source = .{ .kind = "owner" }, .consent = .{ .statement = "mine" }, .references = &refs };
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = try s.add(io, a, v, 100);
    try t.expectEqualStrings("draft", d.status);
    try t.expect(s.has(io, "m", "draft-0001") and !s.has(io, "other", "draft-0001"));
    try t.expectEqualStrings("[{\"text\":\"Hello there.\",\"vq_codes\":[[1,2],[3,4]]}]", s.references(io, a, "m", "draft-0001", 100).?);
    try t.expectEqual(@as(usize, 0), (try s.names(io, a, "m")).len); // drafts are not listed
    try t.expect(s.references(io, a, "m", "draft-0001", 100 + draft_ttl_s) == null); // expired

    _ = try s.add(io, a, v, 200);
    const saved = try s.rename(io, a, "draft-0001", "Sam", 201);
    try t.expectEqualStrings("ready", saved.status);
    try t.expectError(error.Taken, s.add(io, a, .{ .id = "sam", .model = "m", .created_at = 1, .source = .{ .kind = "owner" }, .consent = .{ .statement = "x" }, .references = &refs }, 202));
    // a fresh store reads it back
    var again: Store = .{};
    try again.init(io, t.allocator, dir);
    defer again.deinit();
    try t.expectEqualStrings("Sam", (try again.names(io, a, "m"))[0]);
    try s.remove(io, "Sam");
    try t.expectError(error.NotFound, s.remove(io, "Sam"));
    var third: Store = .{};
    try third.init(io, t.allocator, dir);
    defer third.deinit();
    try t.expectEqual(@as(usize, 0), (try third.names(io, a, null)).len);
    try t.expect(validName("JJ.v2") and !validName("default") and !validName("draft-1") and !validName("-x") and !validName("a/b"));
}
