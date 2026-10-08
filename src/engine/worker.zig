//! `localrouter worker`: serves one compiled-in engine over the worker protocol on stdin/stdout. Logs go to stderr.
//! It dies with the daemon (PR_SET_PDEATHSIG), so an orphan never holds GPU memory.

const std = @import("std");
const linux = std.os.linux;
const engine = @import("engine.zig");
const wire = @import("../tool/wire.zig");
const tool = @import("../tool/tool.zig");
const ToolConfig = @import("../config.zig").ToolConfig;

const stdin_fd = 0;
const stdout_fd = 1;

/// Runs until `exit` or EOF; `tool_json` is the tool's config as the daemon serialized it, `parent` the daemon's pid
/// (LOCALROUTER_PARENT, 0 when unset: no check; it is 1 when the daemon is a container's init, so the check cannot compare with 1).
pub fn run(gpa: std.mem.Allocator, io: std.Io, tool_json: []const u8, parent: i32) !u8 {
    _ = linux.prctl(@intFromEnum(linux.PR.SET_PDEATHSIG), @intFromEnum(linux.SIG.KILL), 0, 0, 0);
    if (parent != 0 and linux.getppid() != parent) return 0; // the daemon died before the line above
    const parsed = try std.json.parseFromSlice(ToolConfig, gpa, tool_json, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const cfg = &parsed.value;
    const entry = engine.find(cfg.engine) orelse {
        std.log.err("no engine named '{s}' in this binary", .{cfg.engine});
        return 2;
    };

    var buf: [wire.max_line]u8 = undefined;
    var lines: wire.LineReader = .{ .fd = stdin_fd, .buf = &buf };
    var loaded: ?engine.Engine = null;
    defer if (loaded) |e| e.unload();
    while (true) {
        const line = lines.next(-1) catch |err| switch (err) {
            error.EndOfStream => return 0,
            else => return err,
        };
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cmd = std.json.parseFromSliceLeaky(wire.Command, a, line, .{ .allocate = .alloc_always }) catch |err| {
            try reply(gpa, .{ .ok = false, .@"error" = @errorName(err), .type = "protocol_error" });
            continue;
        };
        switch (cmd.op) {
            .exit => return 0,
            .load => {
                if (loaded == null) {
                    const e = entry.create(.{ .gpa = gpa, .io = io }, cfg) catch |err| {
                        try fail(gpa, "load", err);
                        continue;
                    };
                    const resident = e.load() catch |err| {
                        e.unload();
                        try fail(gpa, "load", err);
                        continue;
                    };
                    loaded = e;
                    try reply(gpa, .{ .ok = true, .resident = resident });
                } else try reply(gpa, .{ .ok = true });
            },
            .generate => {
                const e = loaded orelse {
                    try reply(gpa, .{ .ok = false, .@"error" = "generate before load", .type = "protocol_error" });
                    continue;
                };
                const job: tool.Job = .{ .id = cmd.id, .dir = cmd.dir, .request = cmd.request orelse {
                    try reply(gpa, .{ .ok = false, .@"error" = "generate without a request", .type = "protocol_error" });
                    continue;
                } };
                var ctx: SinkCtx = .{ .gpa = gpa };
                const out = e.generate(&job, .{ .ctx = &ctx, .progress = SinkCtx.progress }, a) catch |err| {
                    try fail(gpa, "generate", err);
                    continue;
                };
                try reply(gpa, .{ .ok = true, .files = out.files, .seed = out.seed, .ms = out.ms });
            },
        }
    }
}

const SinkCtx = struct {
    gpa: std.mem.Allocator,

    fn progress(ptr: *anyopaque, p: tool.Progress) void {
        const s: *SinkCtx = @ptrCast(@alignCast(ptr));
        reply(s.gpa, .{ .progress = p }) catch {};
    }
};

fn reply(gpa: std.mem.Allocator, r: wire.Reply) !void {
    try wire.send(stdout_fd, gpa, r);
}

/// Reports a failed call; `error.Refused` is the caller's fault (400), anything else the tool's.
fn fail(gpa: std.mem.Allocator, what: []const u8, err: anyerror) !void {
    std.log.err("{s} failed: {s}", .{ what, @errorName(err) });
    const kind = if (err == error.Refused) "invalid_request_error" else "server_error";
    var msg_buf: [128]u8 = undefined;
    const msg = std.fmt.bufPrint(&msg_buf, "{s} failed: {s}", .{ what, @errorName(err) }) catch what;
    try reply(gpa, .{ .ok = false, .@"error" = msg, .type = kind });
}
