//! GET /v1/status and GET /metrics. A snapshot of each tool and the memory budget for an operator or a scraper.
//! `/health` stays `{"ok":true}`. Nothing here is a path, a secret, a hostname or a log line: a tool's last error is
//! its type and code (`sched.safeCode`) and the unix time, and Prometheus labels are escaped.

const std = @import("std");
const Io = std.Io;
const h = @import("http.zig");
const sched = @import("../sched/scheduler.zig");

pub const Error = struct { type: []const u8, code: []const u8, time: i64 };

pub const Tool = struct {
    id: []const u8,
    state: []const u8,
    resident_bytes: u64,
    queue_depth: u32,
    last_load_unix_s: ?i64 = null, // when its last successful load finished (unix seconds)
    last_load_ms: ?u64 = null, // how long that load took
    last_error: ?Error = null,
};

pub const Memory = struct { ceiling_bytes: u64, used_bytes: u64, free_bytes: u64 };

pub const View = struct {
    object: []const u8 = "localrouter.status",
    tools: []const Tool,
    memory: Memory,
    evictions: u64,
};

/// One consistent read of the slots (under `gate`) plus each tool's queue depth (the jobs table).
pub fn view(s: *sched.Scheduler, io: Io, a: std.mem.Allocator) !View {
    const budget = s.budget();
    const tools = try a.alloc(Tool, s.slots.len);
    // `finish` holds the jobs lock and then `gate`. Queue depth takes the jobs lock, so it runs after this block.
    const evictions = blk: {
        s.gate.lockUncancelable(io);
        defer s.gate.unlock(io);
        for (tools, s.slots) |*out, *slot| {
            const code = try a.dupe(u8, sched.safeCode(slot.last_error[0..slot.last_error_len]));
            out.* = .{
                .id = slot.tool.config().id,
                .state = sched.activity(slot.tool.state(), slot.phase),
                .resident_bytes = slot.resident,
                .queue_depth = 0,
                .last_load_unix_s = if (slot.last_load_s > 0) slot.last_load_s else null,
                .last_load_ms = if (slot.last_load_s > 0) slot.last_load_ms else null,
                .last_error = if (slot.last_error_s > 0) .{ .type = code, .code = code, .time = slot.last_error_s } else null,
            };
        }
        break :blk s.evictions;
    };
    for (tools, 0..) |*tool, i| tool.queue_depth = s.queueDepth(i);
    return .{
        .tools = tools,
        .memory = .{ .ceiling_bytes = budget.ceiling_bytes, .used_bytes = budget.booked_bytes, .free_bytes = budget.ceiling_bytes -| budget.booked_bytes },
        .evictions = evictions,
    };
}

pub fn serve(s: *sched.Scheduler, io: Io, req: *h.Request, a: std.mem.Allocator) !void {
    try h.jsonValue(req, a, .ok, try view(s, io, a));
}

pub fn metrics(s: *sched.Scheduler, io: Io, req: *h.Request, a: std.mem.Allocator) !void {
    const v = try view(s, io, a);
    var w: std.Io.Writer.Allocating = .init(a);
    try writePrometheus(&w.writer, v);
    try h.bytes(req, "text/plain; version=0.0.4; charset=utf-8", w.written());
}

/// Prometheus text exposition of `v`. Tool ids are label values, escaped; error types are passed through `safeCode` again.
pub fn writePrometheus(w: *std.Io.Writer, v: View) !void {
    try w.writeAll(
        \\# HELP localrouter_tool_resident_bytes Bytes the tool holds while loaded.
        \\# TYPE localrouter_tool_resident_bytes gauge
        \\
    );
    for (v.tools) |t| {
        try w.writeAll("localrouter_tool_resident_bytes{tool=\"");
        try label(w, t.id);
        try w.print("\"}} {d}\n", .{t.resident_bytes});
    }
    try w.writeAll(
        \\# HELP localrouter_tool_state 1 when the tool is in this state.
        \\# TYPE localrouter_tool_state gauge
        \\
    );
    for (v.tools) |t| {
        try w.writeAll("localrouter_tool_state{tool=\"");
        try label(w, t.id);
        try w.writeAll("\",state=\"");
        try label(w, t.state);
        try w.writeAll("\"} 1\n");
    }
    try w.writeAll(
        \\# HELP localrouter_memory_bytes Memory budget: ceiling, used (booked) and free.
        \\# TYPE localrouter_memory_bytes gauge
        \\
    );
    try w.print("localrouter_memory_bytes{{kind=\"ceiling\"}} {d}\n", .{v.memory.ceiling_bytes});
    try w.print("localrouter_memory_bytes{{kind=\"used\"}} {d}\n", .{v.memory.used_bytes});
    try w.print("localrouter_memory_bytes{{kind=\"free\"}} {d}\n", .{v.memory.free_bytes});
    try w.writeAll(
        \\# HELP localrouter_tool_queue_depth Jobs waiting for the tool, not counting the one running.
        \\# TYPE localrouter_tool_queue_depth gauge
        \\
    );
    for (v.tools) |t| {
        try w.writeAll("localrouter_tool_queue_depth{tool=\"");
        try label(w, t.id);
        try w.print("\"}} {d}\n", .{t.queue_depth});
    }
    try w.writeAll(
        \\# HELP localrouter_tool_last_load_timestamp_seconds Unix time of the tool's last successful load, or 0.
        \\# TYPE localrouter_tool_last_load_timestamp_seconds gauge
        \\
    );
    for (v.tools) |t| {
        try w.writeAll("localrouter_tool_last_load_timestamp_seconds{tool=\"");
        try label(w, t.id);
        try w.print("\"}} {d}\n", .{t.last_load_unix_s orelse 0});
    }
    try w.writeAll(
        \\# HELP localrouter_evictions_total Tools unloaded to make room for another.
        \\# TYPE localrouter_evictions_total counter
        \\
    );
    try w.print("localrouter_evictions_total {d}\n", .{v.evictions});
    try w.writeAll(
        \\# HELP localrouter_tool_last_error_timestamp_seconds Unix time of the tool's last failed job.
        \\# TYPE localrouter_tool_last_error_timestamp_seconds gauge
        \\
    );
    for (v.tools) |t| if (t.last_error) |e| {
        const code = sched.safeCode(e.type);
        try w.writeAll("localrouter_tool_last_error_timestamp_seconds{tool=\"");
        try label(w, t.id);
        try w.writeAll("\",type=\"");
        try label(w, code);
        try w.writeAll("\",code=\"");
        try label(w, sched.safeCode(e.code));
        try w.print("\"}} {d}\n", .{e.time});
    };
}

fn label(w: *std.Io.Writer, s: []const u8) !void {
    for (s) |c| switch (c) {
        '\\' => try w.writeAll("\\\\"),
        '"' => try w.writeAll("\\\""),
        '\n' => try w.writeAll("\\n"),
        else => try w.writeByte(c),
    };
}

// ---------------------------------------------------------------- tests

const testing = std.testing;

test "activity names the five states, and a bad error type is not published" {
    try testing.expectEqualStrings("idle", sched.activity(.ready, .free));
    try testing.expectEqualStrings("loading", sched.activity(.loading, .free));
    try testing.expectEqualStrings("loading", sched.activity(.ready, .loading));
    try testing.expectEqualStrings("committing", sched.activity(.ready, .committing));
    try testing.expectEqualStrings("busy", sched.activity(.busy, .free));
    try testing.expectEqualStrings("busy", sched.activity(.ready, .running));
    try testing.expectEqualStrings("unloaded", sched.activity(.unloaded, .free));
    try testing.expectEqualStrings("unloaded", sched.activity(.failed, .free));
    try testing.expectEqualStrings("server_error", sched.safeCode(""));
    try testing.expectEqualStrings("server_error", sched.safeCode("/data/logs/tool.log"));
    try testing.expectEqualStrings("server_error", sched.safeCode("insufficient memory: /tmp/x"));
    try testing.expectEqualStrings("insufficient_memory", sched.safeCode("insufficient_memory"));
}

test "prometheus text has the gauges and counters and no path" {
    const tools = [_]Tool{
        .{ .id = "tp", .state = "idle", .resident_bytes = 32, .queue_depth = 2, .last_load_unix_s = 10, .last_load_ms = 2500, .last_error = .{ .type = "server_error", .code = "server_error", .time = 11 } },
        .{ .id = "a\"b", .state = "unloaded", .resident_bytes = 0, .queue_depth = 0, .last_error = .{ .type = "/var/log/x", .code = "/var/log/x", .time = 3 } },
    };
    const v: View = .{ .tools = &tools, .memory = .{ .ceiling_bytes = 100, .used_bytes = 32, .free_bytes = 68 }, .evictions = 4 };
    var w: std.Io.Writer.Allocating = .init(testing.allocator);
    defer w.deinit();
    try writePrometheus(&w.writer, v);
    const text = w.written();
    try testing.expect(std.mem.indexOf(u8, text, "localrouter_tool_resident_bytes{tool=\"tp\"} 32\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "localrouter_tool_state{tool=\"tp\",state=\"idle\"} 1\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "localrouter_memory_bytes{kind=\"free\"} 68\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "localrouter_tool_queue_depth{tool=\"tp\"} 2\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "localrouter_tool_last_load_timestamp_seconds{tool=\"tp\"} 10\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "localrouter_evictions_total 4\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "type=\"server_error\",code=\"server_error\"} 11\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "tool=\"a\\\"b\"") != null);
    try testing.expect(std.mem.indexOf(u8, text, "type=\"server_error\",code=\"server_error\"} 3\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "/var/log") == null);
    try testing.expect(std.mem.indexOf(u8, text, "/data/") == null);
}
