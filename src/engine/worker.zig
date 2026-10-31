//! `localrouter worker`: serves one compiled-in engine over the worker protocol on stdin/stdout. Logs go to stderr.
//! It dies with the daemon (PR_SET_PDEATHSIG; on macOS a kqueue watch, `orphan.zig`), so an orphan never holds GPU memory.

const std = @import("std");
const engine = @import("engine.zig");
const wire = @import("../tool/wire.zig");
const tool = @import("../tool/tool.zig");
const ToolConfig = @import("../config.zig").ToolConfig;
const speech_gate = @import("../sched/speech_gate.zig");
const orphan = @import("orphan.zig");

const stdin_fd = 0;
const stdout_fd = 1;

/// Runs until `exit` or EOF; `tool_json` is the tool's config as the daemon serialized it, `parent` the daemon's pid
/// (LOCALROUTER_PARENT, 0 when unset: no check; it is 1 when the daemon is a container's init, so the check cannot compare with 1).
pub fn run(gpa: std.mem.Allocator, io: std.Io, tool_json: []const u8, parent: i32, environ: ?*const std.process.Environ.Map) !u8 {
    const armed = orphan.arm(.KILL);
    if (parent != 0 and (!armed or std.c.getppid() != parent)) return 0; // the daemon died before the line above
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
    // Image and video jobs pause between steps while the daemon makes speech (sched/speech_gate.zig).
    var gate: speech_gate.Waiter = .init(if (environ) |m| m.get(speech_gate.env_name) orelse "" else "");
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
                    const e = entry.create(.{ .gpa = gpa, .io = io, .environ = environ }, cfg) catch |err| {
                        try fail(gpa, "load", err);
                        continue;
                    };
                    const resident = e.load() catch |err| {
                        e.unload();
                        try fail(gpa, "load", err);
                        continue;
                    };
                    loaded = e;
                    try reply(gpa, .{ .ok = true, .resident = resident, .port = e.port() });
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
                const heavy = job.request == .image or job.request == .video or job.request == .asset3d;
                var ctx: SinkCtx = .{ .gpa = gpa, .io = io, .gate = if (heavy) &gate else null };
                if (heavy) {
                    gate.reset();
                    gate.wait(io);
                }
                const out = e.generate(&job, .{ .ctx = &ctx, .progress = SinkCtx.progress }, a) catch |err| {
                    try fail(gpa, "generate", err);
                    continue;
                };
                try reply(gpa, .{ .ok = true, .files = out.files, .seed = out.seed, .ms = out.ms, .finish_reason = out.finish_reason, .segments = out.segments });
            },
        }
    }
}

const SinkCtx = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    gate: ?*speech_gate.Waiter, // an image or video job: it yields to speech at each report

    fn progress(ptr: *anyopaque, p: tool.Progress) void {
        const s: *SinkCtx = @ptrCast(@alignCast(ptr));
        reply(s.gpa, .{ .progress = p }) catch {};
        if (s.gate) |g| g.wait(s.io);
    }
};

fn reply(gpa: std.mem.Allocator, r: wire.Reply) !void {
    try wire.send(stdout_fd, gpa, r);
}

/// Reports a failed call, with the engine's explanation when it gave one; `error.Refused` is the caller's fault (400),
/// anything else the tool's.
fn fail(gpa: std.mem.Allocator, what: []const u8, err: anyerror) !void {
    std.log.err("{s} failed: {s}", .{ what, @errorName(err) });
    const kind = if (err == error.Refused) "invalid_request_error" else "server_error";
    var msg_buf: [640]u8 = undefined;
    const detail = engine.explained();
    const msg = std.fmt.bufPrint(&msg_buf, "{s} failed: {s}", .{ what, if (detail.len > 0) detail else @errorName(err) }) catch what;
    try reply(gpa, .{ .ok = false, .@"error" = msg, .type = kind });
}
