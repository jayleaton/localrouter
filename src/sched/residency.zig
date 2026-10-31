//! `<data_dir>/residency.json`: the residency and idle TTL each tool was given through the API
//! (`POST /v1/tools/{id}/settings`, MCP `set_model_residency`), over its config at every start. Only overrides are kept:
//! a tool reset to its config has no entry. Written through a temporary file renamed over it.
//!
//!     {"tools": {"fishaudio/drama-3": {"residency": "call", "idle_ttl_s": 120}}}

const std = @import("std");
const Io = std.Io;
const config = @import("../config.zig");
const sched = @import("scheduler.zig");

pub const file_name = "residency.json";

pub const Override = struct { residency: ?config.Residency = null, idle_ttl_s: ?u32 = null };

const Doc = struct { tools: std.json.ArrayHashMap(Override) = .{} };

/// Sets the slots' overrides from the file in `dir` (missing: none). An entry for a tool not configured, or one this
/// tool may not have (`call` for an image model), is skipped with a warning; `skip(i)` true leaves slot `i` alone (a
/// follower rank, which loads only when its rank 0 asks).
pub fn apply(io: Io, gpa: std.mem.Allocator, dir: []const u8, slots: []sched.Slot, skip: ?[]const bool) void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const path = std.fs.path.join(a, &.{ dir, file_name }) catch return;
    const text = Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch |err| {
        if (err != error.FileNotFound) std.log.warn("{s}: {s}; the config's residency applies", .{ path, @errorName(err) });
        return;
    };
    const doc = std.json.parseFromSliceLeaky(Doc, a, text, .{ .ignore_unknown_fields = true }) catch |err| {
        std.log.warn("{s}: {s}; the config's residency applies", .{ path, @errorName(err) });
        return;
    };
    var it = doc.tools.map.iterator();
    while (it.next()) |e| {
        const i = for (slots, 0..) |s, k| {
            if (std.mem.eql(u8, s.tool.config().id, e.key_ptr.*)) break k;
        } else {
            std.log.warn("{s}: no tool {s} is configured; its entry is ignored", .{ path, e.key_ptr.* });
            continue;
        };
        if (skip) |sk| if (sk[i]) continue;
        const o = e.value_ptr.*;
        if (o.residency == .call and !config.callAllowed(slots[i].tool.config().kind)) {
            std.log.warn("{s}: {s} cannot be loaded for calls (only chat, speech and transcription models); ignored", .{ path, e.key_ptr.* });
            continue;
        }
        slots[i].residency = o.residency;
        slots[i].ttl_s = o.idle_ttl_s;
        std.log.info("residency {s}: {s}, idle_ttl_s {d} ({s})", .{ e.key_ptr.*, @tagName(sched.Scheduler.ruleOf(&slots[i])), sched.Scheduler.ttlOf(&slots[i]), file_name });
    }
}

/// Writes the tools' overrides to `<dir>/residency.json` (an empty list when none has one). The caller holds
/// `Scheduler.settings`, so the file follows the order the changes were made in.
pub fn save(io: Io, gpa: std.mem.Allocator, dir: []const u8, s: *sched.Scheduler) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var doc: Doc = .{};
    {
        s.gate.lockUncancelable(io);
        defer s.gate.unlock(io);
        for (s.slots) |slot| if (slot.residency != null or slot.ttl_s != null)
            try doc.tools.map.put(a, slot.tool.config().id, .{ .residency = slot.residency, .idle_ttl_s = slot.ttl_s });
    }
    const text = try std.json.Stringify.valueAlloc(a, doc, .{ .whitespace = .indent_2, .emit_null_optional_fields = false });
    const dest = try std.fs.path.join(a, &.{ dir, file_name });
    const tmp = try std.fs.path.join(a, &.{ dir, "." ++ file_name ++ ".tmp" });
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, dir);
    cwd.deleteFile(io, tmp) catch {};
    const f = try cwd.createFile(io, tmp, .{ .permissions = .fromMode(0o600) });
    {
        defer f.close(io);
        try f.writeStreamingAll(io, text);
        try f.sync(io);
    }
    try cwd.rename(tmp, cwd, dest, io);
}
