//! Jobs: one request's life from the API to its output files, and the table the API looks them up in.
//! Every request is a job; the synchronous image route just waits on `done`.

const std = @import("std");
const Io = std.Io;
const Request = @import("../tool/request.zig").Request;
const Kind = @import("../tool/request.zig").Kind;

pub const Status = enum { queued, in_progress, completed, failed, cancelled };

pub const Job = struct {
    arena: std.heap.ArenaAllocator,
    id: []const u8,
    tool: usize, // index into the scheduler's tools
    request: Request,
    dir: []const u8,
    created_at: i64,
    owner: []const u8 = "", // the paired device that asked for it (its id), or "" (pairing off, or the daemon itself)
    // Below: written by the scheduler under `Table.mutex`, read by the API under it.
    status: Status = .queued,
    percent: u8 = 0,
    phase: [32]u8 = @splat(0),
    files: []const []const u8 = &.{},
    seed: u64 = 0,
    ms: u64 = 0,
    error_message: []const u8 = "",
    error_type: []const u8 = "",
    http_status: u16 = 200,
    finish_reason: []const u8 = "", // why the audio ended, when the tool says (speech: stop, length, silence)
    segments: []const u8 = "", // a long-form reply's segment starts, when the tool says
    completed_at: i64 = 0,
    keep_s: ?u32 = null, // overrides keep_outputs_s (the voice loop: its audio is gone at the next expiry tick)
    cancel: std.atomic.Value(bool) = .init(false),
    done: Io.Event = .unset,

    pub fn kind(j: *const Job) Kind {
        return j.request.kind();
    }

    pub fn phaseName(j: *const Job) []const u8 {
        return std.mem.sliceTo(&j.phase, 0);
    }

    pub fn finished(j: *const Job) bool {
        return switch (j.status) {
            .completed, .failed, .cancelled => true,
            else => false,
        };
    }
};

pub const Table = struct {
    gpa: std.mem.Allocator,
    mutex: Io.Mutex = .init,
    map: std.StringHashMapUnmanaged(*Job) = .empty,

    pub fn deinit(t: *Table) void {
        var it = t.map.valueIterator();
        while (it.next()) |j| destroy(t.gpa, j.*);
        t.map.deinit(t.gpa);
    }

    /// A new job for `owner` (a device id, or "") whose request is copied into its own arena; `dir` is
    /// `<jobs_dir>/<id>`, created here.
    pub fn create(t: *Table, io: Io, jobs_dir: []const u8, tool_index: usize, req: Request, owner: []const u8, now: i64) !*Job {
        const j = try t.gpa.create(Job);
        errdefer t.gpa.destroy(j);
        j.* = .{ .arena = .init(t.gpa), .id = "", .tool = tool_index, .request = undefined, .dir = "", .created_at = now };
        errdefer j.arena.deinit();
        const a = j.arena.allocator();
        var rnd: [8]u8 = undefined;
        io.random(&rnd);
        const prefix = switch (req.kind()) {
            .image => "img",
            .video => "video",
            .speech => "speech",
            .transcription => "transcription",
            .chat => "chat",
            .decision => "decision",
        };
        j.id = try std.fmt.allocPrint(a, "{s}_{x}", .{ prefix, std.mem.readInt(u64, &rnd, .little) });
        j.request = try dupeRequest(a, req);
        j.owner = try a.dupe(u8, owner);
        j.dir = try std.fs.path.join(a, &.{ jobs_dir, j.id });
        try Io.Dir.cwd().createDirPath(io, j.dir);
        try t.mutex.lock(io);
        defer t.mutex.unlock(io);
        try t.map.put(t.gpa, j.id, j);
        return j;
    }

    /// The job, or null. The pointer stays valid until `expire` removes it (well after it finished).
    pub fn get(t: *Table, io: Io, id: []const u8) ?*Job {
        t.mutex.lockUncancelable(io);
        defer t.mutex.unlock(io);
        return t.map.get(id);
    }

    /// Removes finished jobs older than `keep_s` and their directories.
    pub fn expire(t: *Table, io: Io, now: i64, keep_s: u32) void {
        t.mutex.lockUncancelable(io);
        defer t.mutex.unlock(io);
        var it = t.map.iterator();
        while (it.next()) |e| {
            const j = e.value_ptr.*;
            if (!j.finished() or now - j.completed_at < (j.keep_s orelse keep_s)) continue;
            Io.Dir.cwd().deleteTree(io, j.dir) catch {};
            t.map.removeByPtr(e.key_ptr);
            destroy(t.gpa, j);
            it = t.map.iterator(); // removal invalidates the iterator; restart (expiry is rare and the table small)
        }
    }

    pub fn count(t: *Table, io: Io) usize {
        t.mutex.lockUncancelable(io);
        defer t.mutex.unlock(io);
        return t.map.count();
    }
};

/// Removes the job directories under `dir` last modified `keep_s` or more before `now` (unix seconds) and returns how
/// many: the outputs of jobs from before a restart, which this process's table never held and so never expires. Run
/// once at start, before any job exists; a directory's mtime is its last file's, so the age runs from the job's output.
pub fn sweep(io: Io, gpa: std.mem.Allocator, dir: []const u8, now: i64, keep_s: u32) !u32 {
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var old: std.ArrayList([]u8) = .empty; // collected first: removing entries while iterating may skip some
    defer {
        for (old.items) |name| gpa.free(name);
        old.deinit(gpa);
    }
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .directory) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        const mtime: i64 = @intCast(@divFloor(st.mtime.nanoseconds, std.time.ns_per_s));
        if (now - mtime >= keep_s) try old.append(gpa, try gpa.dupe(u8, e.name));
    }
    var n: u32 = 0;
    for (old.items) |name| {
        d.deleteTree(io, name) catch |err| {
            std.log.warn("jobs: could not remove {s}/{s}: {s}", .{ dir, name, @errorName(err) });
            continue;
        };
        n += 1;
    }
    return n;
}

test "sweep removes job directories past the retention, keeps fresh ones and plain files" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var rnd: [8]u8 = undefined;
    io.random(&rnd);
    var buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&buf, "/tmp/localrouter-sweep-{x}", .{std.mem.readInt(u64, &rnd, .little)});
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, dir);
    defer cwd.deleteTree(io, dir) catch {};
    var d = try cwd.openDir(io, dir, .{});
    defer d.close(io);
    const now: i64 = 1_800_000_000;
    const day: u32 = 24 * 3600;
    for ([_][]const u8{ "img_old", "img_fresh" }) |name| {
        try d.createDirPath(io, name);
        var sub = try d.openDir(io, name, .{});
        defer sub.close(io);
        try sub.writeFile(io, .{ .sub_path = "0.png", .data = "png" });
    }
    try d.writeFile(io, .{ .sub_path = "notes.txt", .data = "not a job" });
    // last written a day and a minute before `now`, and an hour before it
    const at = struct {
        fn ts(s: i64) Io.File.SetTimestamp {
            return .{ .new = .{ .nanoseconds = @as(i96, s) * std.time.ns_per_s } };
        }
    }.ts;
    try d.setTimestamps(io, "img_old", .{ .modify_timestamp = at(now - day - 60) });
    try d.setTimestamps(io, "img_fresh", .{ .modify_timestamp = at(now - 3600) });
    try d.setTimestamps(io, "notes.txt", .{ .modify_timestamp = at(now - 30 * day) });

    try std.testing.expectEqual(@as(u32, 1), try sweep(io, gpa, dir, now, day));
    try std.testing.expectError(error.FileNotFound, d.statFile(io, "img_old", .{}));
    _ = try d.statFile(io, "img_fresh/0.png", .{});
    _ = try d.statFile(io, "notes.txt", .{});
    try std.testing.expectEqual(@as(u32, 0), try sweep(io, gpa, dir, now, day)); // nothing more to do
}

fn destroy(gpa: std.mem.Allocator, j: *Job) void {
    j.arena.deinit();
    gpa.destroy(j);
}

/// A deep copy of `req` into `a` (strings and lists).
pub fn dupeRequest(a: std.mem.Allocator, req: Request) !Request {
    switch (req) {
        .image => |v| {
            var c = v;
            c.prompt = try a.dupe(u8, v.prompt);
            c.negative_prompt = try a.dupe(u8, v.negative_prompt);
            const refs = try a.alloc([]const u8, v.references.len);
            for (refs, v.references) |*d, s| d.* = try a.dupe(u8, s);
            c.references = refs;
            return .{ .image = c };
        },
        .video => |v| {
            var c = v;
            c.prompt = try a.dupe(u8, v.prompt);
            c.negative_prompt = try a.dupe(u8, v.negative_prompt);
            if (v.first_frame) |f| c.first_frame = try a.dupe(u8, f);
            return .{ .video = c };
        },
        .speech => |v| {
            var c = v;
            c.input = try a.dupe(u8, v.input);
            c.voice = try a.dupe(u8, v.voice);
            c.preset = try a.dupe(u8, v.preset);
            c.references = try a.dupe(u8, v.references);
            return .{ .speech = c };
        },
        .transcription => |v| {
            var c = v;
            c.audio = try a.dupe(u8, v.audio);
            c.language = try a.dupe(u8, v.language);
            c.prompt = try a.dupe(u8, v.prompt);
            return .{ .transcription = c };
        },
        .chat => |v| {
            var c = v;
            c.body = try a.dupe(u8, v.body);
            return .{ .chat = c };
        },
        .decision => |v| {
            var c = v;
            c.body = try a.dupe(u8, v.body);
            return .{ .decision = c };
        },
    }
}

test "a job owns a deep copy of its request" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var input = "hello".*;
    var lang = "en".*;
    const say = try dupeRequest(arena.allocator(), .{ .speech = .{ .input = &input, .voice = "v1" } });
    const hear = try dupeRequest(arena.allocator(), .{ .transcription = .{ .audio = "audio.wav", .language = &lang, .prompt = "Zig" } });
    input[0] = 'j';
    lang[0] = 'd';
    try std.testing.expectEqualStrings("hello", say.speech.input);
    try std.testing.expectEqualStrings("en", hear.transcription.language);
    try std.testing.expectEqualStrings("Zig", hear.transcription.prompt);
}
