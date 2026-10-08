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
    completed_at: i64 = 0,
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

    /// A new job whose request is copied into its own arena; `dir` is `<jobs_dir>/<id>`, created here.
    pub fn create(t: *Table, io: Io, jobs_dir: []const u8, tool_index: usize, req: Request, now: i64) !*Job {
        const j = try t.gpa.create(Job);
        errdefer t.gpa.destroy(j);
        j.* = .{ .arena = .init(t.gpa), .id = "", .tool = tool_index, .request = undefined, .dir = "", .created_at = now };
        errdefer j.arena.deinit();
        const a = j.arena.allocator();
        var rnd: [8]u8 = undefined;
        io.random(&rnd);
        const prefix = if (req.kind() == .video) "video" else "img";
        j.id = try std.fmt.allocPrint(a, "{s}_{x}", .{ prefix, std.mem.readInt(u64, &rnd, .little) });
        j.request = try dupeRequest(a, req);
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
            if (!j.finished() or now - j.completed_at < keep_s) continue;
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
    }
}
