//! ProcessTool: a Tool served by a worker process (`localrouter worker` for compiled-in engines, or the config's `cmd`).
//! Unload is process exit, so every byte returns to the OS; abort and deadlines kill the worker's process group.
//! The tool's config reaches the worker as JSON in LOCALROUTER_TOOL. Worker stderr is appended to `<data>/logs/<id>.log`.
//! With a host GPU lock configured, the scheduler holds it for every LocalRouter tool (`gpu_lock` in the scheduler), and
//! the worker learns so: LOCALROUTER_GPU_LOCK_HELD=1, the host lock in LOCALROUTER_HOST_GPU_LOCK, and in
//! LOCALROUTER_GPU_LOCK a lock file of its own (`<data>/locks/<id>.lock`), uncontended, for a launcher that insists on
//! taking one; so no worker can keep another LocalRouter model out. A worker whose load reply names a `port`
//! runs the model's own HTTP server there: chat and decision requests are proxied to it (`proxy`), and cancelling one ends
//! the request, not the worker.

const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const tool = @import("tool.zig");
const wire = @import("wire.zig");
const engine = @import("../engine/engine.zig");
const upstream = @import("upstream.zig");
const Request = @import("request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;
const speech_gate = @import("../sched/speech_gate.zig");
const yielder = @import("../sched/yielder.zig");
const orphan = @import("../engine/orphan.zig");

pub const Setup = struct {
    gpa: std.mem.Allocator,
    io: Io,
    self_exe: []const u8, // this binary, for `localrouter worker`
    environ: *const std.process.Environ.Map,
    log_dir: []const u8,
    gpu_lock: []const u8 = "",
    speech_gate: []const u8 = "", // the file workers wait on while speech is made (sched/speech_gate.zig); "": none
    rank: ?Rank = null, // this node's rank of a tensor-parallel group (`topology.Group`); null: not one
    guard: bool = orphan.needs_guard, // start `cmd` workers through `localrouter guard` (macOS; off in unit tests: no binary)
};

/// A tensor-parallel rank, as the worker learns it: LOCALROUTER_RANK, LOCALROUTER_WORLD and LOCALROUTER_MASTER (rank 0's
/// address on the link between the hosts), which `openai_server` puts in its `options.rank_env`.
pub const Rank = struct {
    rank: usize,
    world: usize,
    master: []const u8,
};

pub const ProcessTool = struct {
    setup: Setup,
    cfg: *const ToolConfig,
    child: ?std.process.Child = null,
    lines: wire.LineReader = undefined,
    buf: []u8 = &.{},
    pid: std.atomic.Value(i32) = .init(0), // the worker's pid (= its process group), for abort from other threads
    aborted: std.atomic.Value(bool) = .init(false),
    soft_cancel: std.atomic.Value(bool) = .init(false), // the worker said `"cancel":"signal"` when it loaded
    in_flight: std.atomic.Value(bool) = .init(false), // a generate is written and its reply not yet read: SIGUSR1 may go
    st: std.atomic.Value(tool.State) = .init(.unloaded),
    fail_msg: [2048]u8 = undefined,
    fail: tool.Failure = .{},
    port: u16 = 0, // the loopback port of the worker's own server (its load reply); 0: none
    proxying: std.atomic.Value(bool) = .init(false), // a chat request is being proxied: abort ends it, not the worker
    upstream_fd: std.atomic.Value(posix.fd_t) = .init(-1), // that request's connection
    lock_path: [1024]u8 = undefined, // `ownLock`

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
            .cancel = cancel,
            .state = state,
            .failure = failure,
            .server_pid = serverPid,
        } };
    }

    fn self(ptr: *anyopaque) *ProcessTool {
        return @ptrCast(@alignCast(ptr));
    }

    fn config(ptr: *anyopaque) *const ToolConfig {
        return self(ptr).cfg;
    }

    /// What an encode whose peak the config does not give is booked at: 8 GiB, the box's rule for an unmeasured process.
    const unmeasured_working: u64 = 8 << 30;

    fn needs(ptr: *anyopaque, req: *const Request) tool.Needs {
        const p = self(ptr);
        if (engine.find(p.cfg.engine)) |e| return e.needs(p.cfg, req);
        // encoding a voice runs the model's encoder beside the loaded model: its measured peak, else the unmeasured rule
        if (req.* == .speech and req.speech.clone.len > 0) return .{ .resident = p.cfg.resident_bytes, .working = engine.optionInt(p.cfg, "clone_working_bytes", unmeasured_working) };
        return .{ .resident = p.cfg.resident_bytes, .working = p.cfg.working_bytes };
    }

    fn state(ptr: *anyopaque) tool.State {
        return self(ptr).st.load(.acquire);
    }

    fn failure(ptr: *anyopaque) tool.Failure {
        return self(ptr).fail;
    }

    /// A loaded worker's own server (one it proxies to): the worker's child.
    fn serverPid(ptr: *anyopaque) posix.pid_t {
        const p = self(ptr);
        if (p.st.load(.acquire) == .unloaded or p.port == 0) return 0;
        return yielder.childOf(p.pid.load(.acquire));
    }

    /// Ends the running load or generate: a proxied request loses its connection (the worker and its model stay
    /// loaded); anything else kills the worker's group. An idle tool has nothing to end.
    fn abort(ptr: *anyopaque) void {
        const p = self(ptr);
        p.aborted.store(true, .seq_cst);
        if (p.proxying.load(.seq_cst)) {
            const fd = p.upstream_fd.load(.seq_cst);
            if (fd >= 0) _ = std.c.shutdown(fd, std.c.SHUT.RDWR); // else `proxy` sees `aborted` once it has connected
            return;
        }
        const st = p.st.load(.acquire);
        if (st != .loading and st != .busy) return;
        const pid = p.pid.load(.acquire);
        if (pid > 0) _ = posix.system.kill(-pid, posix.SIG.KILL);
    }

    /// SIGUSR1 to the worker alone (not its group): a worker that offers it ends the generate and stays loaded. Before
    /// the generate is written the signal would be lost (the worker ignores one between jobs), so then nothing is sent
    /// here: the caller set the job's `cancelled` first, and `generate` acts on that once it has written.
    fn cancel(ptr: *anyopaque) bool {
        const p = self(ptr);
        const pid = p.pid.load(.acquire);
        if (!p.soft_cancel.load(.acquire) or pid <= 0) return false;
        if (p.in_flight.load(.seq_cst)) _ = posix.system.kill(pid, posix.SIG.USR1);
        return true;
    }

    fn load(ptr: *anyopaque) tool.Error!u64 {
        const p = self(ptr);
        if (p.child != null) return 0;
        {
            var arena: std.heap.ArenaAllocator = .init(p.setup.gpa);
            defer arena.deinit();
            if (engine.unqualified(p.setup.io, arena.allocator(), p.cfg, &p.fail_msg)) |why| {
                p.fail = .{ .message = why, .type = "server_error" };
                return tool.Error.LoadFailed;
            }
        }
        p.aborted.store(false, .release);
        p.st.store(.loading, .release);
        p.spawn() catch |err| return p.failed(.unloaded, tool.Error.LoadFailed, "could not start the worker: {s}", .{@errorName(err)});
        wire.send(p.stdinFd(), p.setup.gpa, wire.Command{ .op = .load }) catch return p.died(tool.Error.LoadFailed);
        const r = p.awaitReply(null, p.cfg.load_timeout_s, null) catch |err| return p.died(err);
        defer r.deinit();
        if (r.value.ok != true) {
            p.kill();
            return p.failed(.unloaded, tool.Error.LoadFailed, "load failed: {s}", .{r.value.@"error"});
        }
        p.port = r.value.port;
        p.soft_cancel.store(std.mem.eql(u8, r.value.cancel, "signal"), .release);
        p.st.store(.ready, .release);
        return r.value.resident;
    }

    fn generate(ptr: *anyopaque, job: *const tool.Job, sink: tool.Sink, arena: std.mem.Allocator) tool.Error!tool.Output {
        const p = self(ptr);
        if (p.child == null) return tool.Error.WorkerDied;
        if (job.request == .chat or job.request == .decision) {
            p.aborted.store(false, .seq_cst); // before `proxying`: an abort from here on is seen (`abort`, `proxy`)
            p.proxying.store(true, .seq_cst);
            defer p.proxying.store(false, .seq_cst);
            p.st.store(.busy, .release);
            return p.proxy(job, arena);
        }
        p.st.store(.busy, .release);
        const soft = p.soft_cancel.load(.acquire);
        if (soft and isCancelled(job)) return p.cancelled(); // cancelled before it reached the worker: nothing to undo
        const cmd: wire.Command = .{ .op = .generate, .id = job.id, .dir = job.dir, .request = job.request };
        wire.send(p.stdinFd(), p.setup.gpa, cmd) catch return p.died(tool.Error.WorkerDied);
        p.in_flight.store(true, .seq_cst); // then read `cancelled`: a cancel either sees this or is seen below
        const r = p.awaitReply(sink, p.cfg.generate_timeout_s, if (soft) job.cancelled else null) catch |err| {
            p.in_flight.store(false, .seq_cst);
            return p.died(err);
        };
        p.in_flight.store(false, .seq_cst);
        defer r.deinit();
        if (r.value.ok != true and std.mem.eql(u8, r.value.type, "cancelled")) return p.cancelled(); // ended early; still loaded
        if (r.value.ok != true) {
            const t = for ([_][]const u8{ "invalid_request_error", "service_unavailable" }) |known| {
                if (std.mem.eql(u8, r.value.type, known)) break known;
            } else "server_error";
            p.fail = .{ .message = p.copyMsg("{s}", .{r.value.@"error"}), .type = t };
            p.st.store(.ready, .release);
            return tool.Error.GenerateFailed;
        }
        p.st.store(.ready, .release);
        const files = arena.alloc([]const u8, r.value.files.len) catch return tool.Error.OutOfMemory;
        for (files, r.value.files) |*d, s| d.* = arena.dupe(u8, s) catch return tool.Error.OutOfMemory;
        const finish = arena.dupe(u8, headerSafe(r.value.finish_reason)) catch return tool.Error.OutOfMemory;
        const segments = arena.dupe(u8, headerSafe(r.value.segments)) catch return tool.Error.OutOfMemory;
        return .{ .files = files, .seed = r.value.seed, .ms = r.value.ms, .finish_reason = finish, .segments = segments };
    }

    /// A worker's string that the API sends as a response header (`finish_reason`, `segments`): printable ASCII of at
    /// most 512 bytes, else dropped.
    fn headerSafe(s: []const u8) []const u8 {
        if (s.len > 512) return "";
        for (s) |c| if (c < 0x20 or c > 0x7e) return "";
        return s;
    }

    /// A chat or decision request, POSTed to the worker's own server on loopback; the reply goes into the job directory as
    /// it arrives: `completion.json`, or the event stream in `completion.sse` (created once the server has accepted the
    /// request, which is how the API knows to start streaming), or `decision.json` (to `options.decision_path`, default
    /// /v1/systemone). The server's refusals keep its status: 4xx is the caller's (400), 503 a busy server (503),
    /// anything else a failure (500). `generate_timeout_s` bounds each silence.
    fn proxy(p: *ProcessTool, job: *const tool.Job, arena: std.mem.Allocator) tool.Error!tool.Output {
        const c: struct { body: []const u8, stream: bool, seed: u64, path: []const u8, name: []const u8 } = switch (job.request) {
            .chat => |v| .{ .body = v.body, .stream = v.stream, .seed = v.seed, .path = "/v1/chat/completions", .name = if (v.stream) "completion.sse" else "completion.json" },
            .decision => |v| .{ .body = v.body, .stream = false, .seed = v.seed, .path = engine.optionString(p.cfg, "decision_path", "/v1/systemone"), .name = "decision.json" },
            else => unreachable, // `generate` proxies only these
        };
        if (p.port == 0) return p.failed(.ready, tool.Error.GenerateFailed, "model '{s}' has no server to send requests to: its worker's load reply named no port", .{p.cfg.id});
        const start = Io.Clock.awake.now(p.setup.io);
        const fd = upstream.connect(p.port) catch |err| return p.lost(err);
        defer _ = std.c.close(fd);
        p.upstream_fd.store(fd, .seq_cst);
        defer p.upstream_fd.store(-1, .seq_cst); // runs before the close above
        if (p.aborted.load(.seq_cst)) return p.lost(error.Closed);
        upstream.send(fd, "POST", c.path, c.body) catch |err| return p.lost(err);
        const buf = arena.alloc(u8, 64 << 10) catch return tool.Error.OutOfMemory;
        var r: upstream.Reader = .{ .fd = fd, .buf = buf, .timeout_ms = @intCast(@min(@as(u64, p.cfg.generate_timeout_s) * 1000, std.math.maxInt(i32))) };
        const status = r.head() catch |err| return p.lost(err);
        if (status != 200) {
            const body = r.rest(arena, 64 << 10) catch "";
            const typ = if (status >= 400 and status < 500 and status != 401 and status != 403) "invalid_request_error" else if (status == 503) "service_unavailable" else "server_error";
            p.fail = .{ .message = p.copyMsg("{s}", .{upstream.errorMessage(arena, body)}), .type = typ };
            p.st.store(.ready, .release);
            return tool.Error.GenerateFailed;
        }
        const name = c.name;
        const path = std.fs.path.join(arena, &.{ job.dir, name }) catch return tool.Error.OutOfMemory;
        const file = Io.Dir.cwd().createFile(p.setup.io, path, .{}) catch return p.failed(.ready, tool.Error.GenerateFailed, "could not write {s}", .{name});
        defer file.close(p.setup.io);
        while (r.next() catch |err| return p.lost(err)) |bytes| {
            wire.writeAll(file.handle, bytes) catch return p.failed(.ready, tool.Error.GenerateFailed, "could not write {s}", .{name});
        }
        p.st.store(.ready, .release);
        const files = arena.dupe([]const u8, &.{name}) catch return tool.Error.OutOfMemory;
        return .{ .files = files, .seed = c.seed, .ms = @intCast(start.durationTo(Io.Clock.awake.now(p.setup.io)).toMilliseconds()) };
    }

    /// A proxied request that ended early: cancelled (the model stays loaded), a deadline, or a server that is gone
    /// (both kill the worker, so the next request starts the model again).
    fn lost(p: *ProcessTool, err: upstream.Error) tool.Error {
        if (p.aborted.load(.seq_cst)) {
            p.st.store(.ready, .release);
            return tool.Error.Cancelled;
        }
        if (err == error.Timeout) return p.died(tool.Error.Timeout);
        p.kill();
        var tail_buf: [1024]u8 = undefined;
        p.fail = .{ .message = p.copyMsg("the model's server is gone ({s}); log tail:\n{s}", .{ @errorName(err), p.logTail(&tail_buf) }), .type = "server_error" };
        p.st.store(.unloaded, .release);
        return tool.Error.WorkerDied;
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

    /// A generate that ended early on `cancel`: the worker stays loaded.
    fn cancelled(p: *ProcessTool) tool.Error {
        p.fail = .{ .message = "cancelled", .type = "server_error" };
        p.st.store(.ready, .release);
        return tool.Error.Cancelled;
    }

    fn isCancelled(job: *const tool.Job) bool {
        return if (job.cancelled) |c| c.load(.seq_cst) else false;
    }

    /// How often SIGUSR1 is sent again while a cancelled generate has not replied.
    const resignal_ms = 200;

    /// Reads worker lines until the final reply, forwarding progress; `timeout_s` bounds each silence. Once `cancel_flag`
    /// is set, SIGUSR1 goes to the worker every `resignal_ms` until it replies: the first may land before the worker has
    /// read the generate, when it ignores it.
    fn awaitReply(p: *ProcessTool, sink: ?tool.Sink, timeout_s: u32, cancel_flag: ?*const std.atomic.Value(bool)) tool.Error!std.json.Parsed(wire.Reply) {
        const ms: i64 = @min(@as(i64, timeout_s) * 1000, std.math.maxInt(i32));
        var silent: i64 = 0; // since the last line
        while (true) {
            const cancelling = if (cancel_flag) |c| c.load(.seq_cst) else false;
            if (cancelling) {
                const pid = p.pid.load(.acquire);
                if (pid > 0) _ = posix.system.kill(pid, posix.SIG.USR1);
            }
            const wait: i64 = if (cancel_flag != null) @min(ms - silent, resignal_ms) else ms;
            const line = p.lines.next(@intCast(wait)) catch |err| switch (err) {
                error.Timeout => {
                    silent += wait;
                    if (silent >= ms) return tool.Error.Timeout;
                    continue;
                },
                else => return if (p.aborted.load(.acquire)) tool.Error.Cancelled else tool.Error.WorkerDied,
            };
            silent = 0;
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
        if (p.setup.gpu_lock.len > 0) {
            try env.put("LOCALROUTER_HOST_GPU_LOCK", p.setup.gpu_lock);
            try env.put("LOCALROUTER_GPU_LOCK_HELD", "1");
            try env.put("LOCALROUTER_GPU_LOCK", try p.ownLock());
        }
        var pid_buf: [16]u8 = undefined; // the worker's orphan check compares its parent with this, not with 1
        const me = try std.fmt.bufPrint(&pid_buf, "{d}", .{std.c.getpid()});
        try env.put("LOCALROUTER_PARENT", me);
        if (p.setup.speech_gate.len > 0) try env.put(speech_gate.env_name, p.setup.speech_gate);
        var rank_buf: [2][24]u8 = undefined;
        if (p.setup.rank) |r| {
            try env.put("LOCALROUTER_RANK", try std.fmt.bufPrint(&rank_buf[0], "{d}", .{r.rank}));
            try env.put("LOCALROUTER_WORLD", try std.fmt.bufPrint(&rank_buf[1], "{d}", .{r.world}));
            try env.put("LOCALROUTER_MASTER", r.master);
        }
        const argv: []const []const u8 = if (p.cfg.cmd.len == 0) &.{ p.setup.self_exe, "worker" } else if (p.setup.guard)
            try std.mem.concat(gpa, []const u8, &.{ &.{ p.setup.self_exe, "guard", me, "--" }, p.cfg.cmd }) // dies with the daemon
        else
            p.cfg.cmd;
        defer if (p.cfg.cmd.len > 0 and p.setup.guard) gpa.free(argv);
        const log_path = try std.fmt.allocPrint(gpa, "{s}/{s}.log", .{ p.setup.log_dir, p.cfg.id });
        defer gpa.free(log_path);
        Io.Dir.cwd().createDirPath(p.setup.io, std.fs.path.dirname(log_path).?) catch {}; // ids such as fishaudio/drama-3
        const log = try Io.Dir.cwd().createFile(p.setup.io, log_path, .{ .truncate = false });
        defer log.close(p.setup.io);
        _ = posix.system.lseek(log.handle, 0, posix.SEEK.END);
        const child = try std.process.spawn(p.setup.io, .{ .argv = argv, .environ_map = &env, .stdin = .pipe, .stdout = .pipe, .stderr = .{ .file = log }, .pgid = 0 });
        p.child = child;
        p.pid.store(child.id.?, .release);
        p.lines = .{ .fd = child.stdout.?.handle, .buf = p.buf };
    }

    /// This tool's own lock file, `<data>/locks/<id>.lock` ('/' in the id becomes '_'), created if missing: what the
    /// worker gets as LOCALROUTER_GPU_LOCK while the scheduler holds the host's lock.
    fn ownLock(p: *ProcessTool) ![]const u8 {
        const data = std.fs.path.dirname(p.setup.log_dir) orelse ".";
        const path = try std.fmt.bufPrint(&p.lock_path, "{s}/locks/{s}.lock", .{ data, p.cfg.id });
        for (path[path.len - p.cfg.id.len - ".lock".len ..][0..p.cfg.id.len]) |*ch| if (ch.* == '/') {
            ch.* = '_';
        };
        Io.Dir.cwd().createDirPath(p.setup.io, std.fs.path.dirname(path).?) catch {};
        const file = try Io.Dir.cwd().createFile(p.setup.io, path, .{ .truncate = false });
        file.close(p.setup.io);
        return path;
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
        p.port = 0;
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

/// A worker that, like a real one, honours SIGUSR1 only inside a generate, which lasts 3 s unless cancelled.
const slow_worker =
    \\import json, signal, sys, time
    \\busy = False
    \\class Cancelled(Exception): pass
    \\def on_usr1(*_):
    \\    if busy: raise Cancelled()
    \\signal.signal(signal.SIGUSR1, on_usr1)
    \\for line in sys.stdin:
    \\    op = json.loads(line)["op"]
    \\    if op == "load": out = {"ok": True, "resident": 1, "cancel": "signal"}
    \\    elif op == "exit": break
    \\    else:
    \\        try:
    \\            busy = True
    \\            time.sleep(3); out = {"ok": True, "files": []}
    \\        except Cancelled: out = {"ok": False, "type": "cancelled", "error": "cancelled"}
    \\        finally: busy = False
    \\    print(json.dumps(out), flush=True)
;

test "a cancel before the generate reaches the worker is not lost: before the write, and in the moment after it" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "worker.py", .data = slow_worker });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const script = try std.fmt.allocPrint(gpa, "{s}/worker.py", .{dir});
    defer gpa.free(script);
    var environ = std.process.Environ.Map.init(gpa);
    defer environ.deinit();
    if (std.c.getenv("PATH")) |v| try environ.put("PATH", std.mem.span(v));
    const cfg: ToolConfig = .{ .id = "slow", .kind = .speech, .cmd = &.{ "python3", script }, .resident_bytes = 1 };
    var p = try ProcessTool.init(.{ .gpa = gpa, .io = io, .self_exe = "/nonexistent", .environ = &environ, .log_dir = dir, .guard = false }, &cfg);
    defer p.deinit();
    const t = p.asTool();
    _ = try t.load();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const sink: tool.Sink = .{ .ctx = undefined, .progress = struct {
        fn f(_: *anyopaque, _: tool.Progress) void {}
    }.f };

    // Cancelled while still queued for the worker (loading, waiting on the speech gate): never sent, ends at once.
    var flag: std.atomic.Value(bool) = .init(true);
    try std.testing.expect(t.cancel());
    const job: tool.Job = .{ .id = "a", .dir = dir, .request = .{ .speech = .{ .input = "hi" } }, .cancelled = &flag };
    const t0 = Io.Clock.awake.now(io);
    try std.testing.expectError(tool.Error.Cancelled, t.generate(&job, sink, arena.allocator()));
    try std.testing.expect(t0.durationTo(Io.Clock.awake.now(io)).toMilliseconds() < 500);
    try std.testing.expectEqual(tool.State.ready, t.state());

    // Cancelled the instant after the write, before the worker has read it: its first SIGUSR1 is ignored, a later
    // one ends the generate well before its 3 s.
    for (0..3) |_| {
        var later: std.atomic.Value(bool) = .init(false);
        const j2: tool.Job = .{ .id = "b", .dir = dir, .request = .{ .speech = .{ .input = "hi" } }, .cancelled = &later };
        const Canceller = struct {
            fn run(tl: tool.Tool, f: *std.atomic.Value(bool), pt: *ProcessTool) void {
                while (!pt.in_flight.load(.seq_cst)) std.Thread.yield() catch {};
                f.store(true, .seq_cst);
                _ = tl.cancel();
            }
        };
        const th = try std.Thread.spawn(.{}, Canceller.run, .{ t, &later, &p });
        const t1 = Io.Clock.awake.now(io);
        const r = t.generate(&j2, sink, arena.allocator());
        th.join();
        try std.testing.expectError(tool.Error.Cancelled, r);
        try std.testing.expect(t1.durationTo(Io.Clock.awake.now(io)).toMilliseconds() < 2000);
        try std.testing.expectEqual(tool.State.ready, t.state());
    }

    // Uncancelled, it runs to its end.
    var never: std.atomic.Value(bool) = .init(false);
    const j3: tool.Job = .{ .id = "c", .dir = dir, .request = .{ .speech = .{ .input = "hi" } }, .cancelled = &never };
    _ = try t.generate(&j3, sink, arena.allocator());
}
