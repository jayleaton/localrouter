//! ProcessTool: a Tool served by a worker process (`localrouter worker` for compiled-in engines, or the config's `cmd`).
//! Unload is process exit, so every byte returns to the OS; abort and deadlines kill the worker's process group.
//! The tool's config reaches the worker as JSON in LOCALROUTER_TOOL. Worker stderr is appended to `<data>/logs/<id>.log`.

const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const tool = @import("tool.zig");
const wire = @import("wire.zig");
const engine = @import("../engine/engine.zig");
const Request = @import("request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const Setup = struct {
    gpa: std.mem.Allocator,
    io: Io,
    self_exe: []const u8, // this binary, for `localrouter worker`
    environ: *const std.process.Environ.Map,
    log_dir: []const u8,
};

pub const ProcessTool = struct {
    setup: Setup,
    cfg: *const ToolConfig,
    child: ?std.process.Child = null,
    lines: wire.LineReader = undefined,
    buf: []u8 = &.{},
    pid: std.atomic.Value(i32) = .init(0), // the worker's pid (= its process group), for abort from other threads
    aborted: std.atomic.Value(bool) = .init(false),
    st: std.atomic.Value(tool.State) = .init(.unloaded),
    fail_msg: [2048]u8 = undefined,
    fail: tool.Failure = .{},

    pub fn init(setup: Setup, cfg: *const ToolConfig) !ProcessTool {
        return .{ .setup = setup, .cfg = cfg, .buf = try setup.gpa.alloc(u8, wire.max_line) };
    }

    pub fn deinit(p: *ProcessTool) void {
        unload(p);
        p.setup.gpa.free(p.buf);
    }

    pub fn asTool(p: *ProcessTool) tool.Tool {
        return .{ .ptr = p, .vtable = &.{
            .config = config,
            .needs = needs,
            .load = load,
            .generate = generate,
            .unload = unload,
            .abort = abort,
            .state = state,
            .failure = failure,
        } };
    }

    fn self(ptr: *anyopaque) *ProcessTool {
        return @ptrCast(@alignCast(ptr));
    }

    fn config(ptr: *anyopaque) *const ToolConfig {
        return self(ptr).cfg;
    }

    fn needs(ptr: *anyopaque, req: *const Request) tool.Needs {
        const p = self(ptr);
        if (engine.find(p.cfg.engine)) |e| return e.needs(p.cfg, req);
        return .{ .resident = p.cfg.resident_bytes, .working = p.cfg.working_bytes };
    }

    fn state(ptr: *anyopaque) tool.State {
        return self(ptr).st.load(.acquire);
    }

    fn failure(ptr: *anyopaque) tool.Failure {
        return self(ptr).fail;
    }

    fn abort(ptr: *anyopaque) void {
        const p = self(ptr);
        p.aborted.store(true, .release);
        const pid = p.pid.load(.acquire);
        if (pid > 0) _ = posix.system.kill(-pid, posix.SIG.KILL);
    }

    fn load(ptr: *anyopaque) tool.Error!u64 {
        const p = self(ptr);
        if (p.child != null) return 0;
        p.aborted.store(false, .release);
        p.st.store(.loading, .release);
        p.spawn() catch |err| return p.failed(.unloaded, tool.Error.LoadFailed, "could not start the worker: {s}", .{@errorName(err)});
        wire.send(p.stdinFd(), p.setup.gpa, wire.Command{ .op = .load }) catch return p.died(tool.Error.LoadFailed);
        const r = p.awaitReply(null, p.cfg.load_timeout_s) catch |err| return p.died(err);
        defer r.deinit();
        if (r.value.ok != true) {
            p.kill();
            return p.failed(.unloaded, tool.Error.LoadFailed, "load failed: {s}", .{r.value.@"error"});
        }
        p.st.store(.ready, .release);
        return r.value.resident;
    }

    fn generate(ptr: *anyopaque, job: *const tool.Job, sink: tool.Sink, arena: std.mem.Allocator) tool.Error!tool.Output {
        const p = self(ptr);
        if (p.child == null) return tool.Error.WorkerDied;
        p.st.store(.busy, .release);
        const cmd: wire.Command = .{ .op = .generate, .id = job.id, .dir = job.dir, .request = job.request };
        wire.send(p.stdinFd(), p.setup.gpa, cmd) catch return p.died(tool.Error.WorkerDied);
        const r = p.awaitReply(sink, p.cfg.generate_timeout_s) catch |err| return p.died(err);
        defer r.deinit();
        if (r.value.ok != true) {
            const t = if (std.mem.eql(u8, r.value.type, "invalid_request_error")) "invalid_request_error" else "server_error";
            p.fail = .{ .message = p.copyMsg("{s}", .{r.value.@"error"}), .type = t };
            p.st.store(.ready, .release);
            return tool.Error.GenerateFailed;
        }
        p.st.store(.ready, .release);
        const files = arena.alloc([]const u8, r.value.files.len) catch return tool.Error.OutOfMemory;
        for (files, r.value.files) |*d, s| d.* = arena.dupe(u8, s) catch return tool.Error.OutOfMemory;
        return .{ .files = files, .seed = r.value.seed, .ms = r.value.ms };
    }

    /// Asks the worker to exit, then kills its group if it has not within 10 s.
    fn unload(ptr: *anyopaque) void {
        const p = self(ptr);
        if (p.child == null) return;
        wire.send(p.stdinFd(), p.setup.gpa, wire.Command{ .op = .exit }) catch {};
        while (true) _ = p.lines.next(10_000) catch break; // drain until EOF or the grace period ends
        p.kill();
        p.st.store(.unloaded, .release);
    }

    /// Reads worker lines until the final reply, forwarding progress; `timeout_s` bounds each silence.
    fn awaitReply(p: *ProcessTool, sink: ?tool.Sink, timeout_s: u32) tool.Error!std.json.Parsed(wire.Reply) {
        const ms: i32 = @intCast(@min(@as(u64, timeout_s) * 1000, std.math.maxInt(i32)));
        while (true) {
            const line = p.lines.next(ms) catch |err| return switch (err) {
                error.Timeout => tool.Error.Timeout,
                else => if (p.aborted.load(.acquire)) tool.Error.Cancelled else tool.Error.WorkerDied,
            };
            const r = std.json.parseFromSlice(wire.Reply, p.setup.gpa, line, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch continue;
            if (r.value.ok != null) return r;
            defer r.deinit();
            if (r.value.progress) |pr| if (sink) |s| s.report(pr);
        }
    }

    fn spawn(p: *ProcessTool) !void {
        const gpa = p.setup.gpa;
        const json = try std.json.Stringify.valueAlloc(gpa, p.cfg.*, .{ .emit_null_optional_fields = false });
        defer gpa.free(json);
        var env = std.process.Environ.Map.init(gpa);
        defer env.deinit();
        var it = p.setup.environ.iterator();
        while (it.next()) |kv| try env.put(kv.key_ptr.*, kv.value_ptr.*);
        try env.put("LOCALROUTER_TOOL", json);
        var pid_buf: [16]u8 = undefined; // the worker's orphan check compares its parent with this, not with 1
        try env.put("LOCALROUTER_PARENT", try std.fmt.bufPrint(&pid_buf, "{d}", .{std.os.linux.getpid()}));
        const argv: []const []const u8 = if (p.cfg.cmd.len > 0) p.cfg.cmd else &.{ p.setup.self_exe, "worker" };
        Io.Dir.cwd().createDirPath(p.setup.io, p.setup.log_dir) catch {};
        const log_path = try std.fmt.allocPrint(gpa, "{s}/{s}.log", .{ p.setup.log_dir, p.cfg.id });
        defer gpa.free(log_path);
        const log = try Io.Dir.cwd().createFile(p.setup.io, log_path, .{ .truncate = false });
        defer log.close(p.setup.io);
        _ = posix.system.lseek(log.handle, 0, posix.SEEK.END);
        const child = try std.process.spawn(p.setup.io, .{ .argv = argv, .environ_map = &env, .stdin = .pipe, .stdout = .pipe, .stderr = .{ .file = log }, .pgid = 0 });
        p.child = child;
        p.pid.store(child.id.?, .release);
        p.lines = .{ .fd = child.stdout.?.handle, .buf = p.buf };
    }

    fn stdinFd(p: *ProcessTool) posix.fd_t {
        return p.child.?.stdin.?.handle;
    }

    /// Kills the worker's group and reaps it; safe when it already exited.
    fn kill(p: *ProcessTool) void {
        var child = p.child orelse return;
        _ = posix.system.kill(-child.id.?, posix.SIG.KILL);
        _ = child.wait(p.setup.io) catch {};
        p.child = null;
        p.pid.store(0, .release);
    }

    /// The worker is gone or hung: kill it, record why with the log tail, and report `err`.
    fn died(p: *ProcessTool, err: tool.Error) tool.Error {
        p.kill();
        const why = switch (err) {
            tool.Error.Cancelled => "cancelled",
            tool.Error.Timeout => "deadline passed",
            else => "worker exited",
        };
        var tail_buf: [1024]u8 = undefined;
        p.fail = .{ .message = p.copyMsg("{s}; log tail:\n{s}", .{ why, p.logTail(&tail_buf) }), .type = "server_error" };
        p.st.store(.unloaded, .release);
        return err;
    }

    fn failed(p: *ProcessTool, to: tool.State, err: tool.Error, comptime fmt: []const u8, args: anytype) tool.Error {
        p.fail = .{ .message = p.copyMsg(fmt, args), .type = "server_error" };
        p.st.store(to, .release);
        return err;
    }

    fn copyMsg(p: *ProcessTool, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&p.fail_msg, fmt, args) catch p.fail_msg[0..];
    }

    /// The last 8 lines of the worker's log.
    fn logTail(p: *ProcessTool, out: []u8) []const u8 {
        var path_buf: [512]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "{s}/{s}.log", .{ p.setup.log_dir, p.cfg.id }) catch return "";
        const file = Io.Dir.cwd().openFile(p.setup.io, path, .{}) catch return "";
        defer file.close(p.setup.io);
        const size = posix.system.lseek(file.handle, 0, posix.SEEK.END);
        if (size <= 0) return "";
        const from: usize = @intCast(@max(0, size - @as(i64, @intCast(out.len))));
        const rc = posix.system.pread(file.handle, out.ptr, out.len, @intCast(from));
        if (posix.errno(rc) != .SUCCESS) return "";
        const text = std.mem.trimEnd(u8, out[0..@intCast(rc)], "\n");
        var cut: usize = text.len;
        var n: usize = 0;
        while (cut > 0 and n < 8) : (n += 1) cut = std.mem.lastIndexOfScalar(u8, text[0..cut], '\n') orelse 0;
        return std.mem.trimStart(u8, text[cut..], "\n");
    }
};
