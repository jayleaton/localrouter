//! Remote tools on the front of a multi-node topology (`src/topology.zig`). A `NodeTool` stands for a tool another node
//! runs: it holds no memory here (`needs` is zero, `Tool.local` false) and sends each job to that node's daemon, which
//! admits it against its own memory and GPU lock. A chat request is forwarded to the node's /v1/chat/completions as it
//! is, and its reply streams into the job directory as a local chat server's does (`process.zig`'s proxy); every
//! other kind goes through the node job routes (`/v1/node/jobs`, `api/node.zig`): the request and its input files go
//! over, the job runs there, and its output files come back into the job directory.
//!
//! A tool the front lists under its `fallback` runs here instead, as an ordinary worker (`ProcessTool`), while its node
//! is down. The `Cluster` asks every node how it is every `health_interval_ms` (`/v1/node/status`, which also tells it
//! how many voice calls the front has open, so its scheduler guards the voice loop's models there too); after
//! `down_after` missed answers in a row, or a connection that fails mid-request, the node is down. When a node goes
//! down or comes back, its tools are unloaded on the scheduler (`Scheduler.refresh`) and load on their new side.

const std = @import("std");
const posix = std.posix;
const Io = std.Io;
const tool = @import("tool.zig");
const upstream = @import("upstream.zig");
const wire = @import("wire.zig");
const process = @import("process.zig");
const Request = @import("request.zig").Request;
const config = @import("../config.zig");
const ToolConfig = config.ToolConfig;
const topology = @import("../topology.zig");

/// A node's daemon as the front sees it.
const Peer = struct {
    node: *const config.Node,
    addrs: []const [4]u8,
    up: std.atomic.Value(bool) = .init(false),
    misses: std.atomic.Value(u32) = .init(0),
    changed: std.atomic.Value(bool) = .init(false), // went down or came back: its tools move (`Cluster.run`)
    preferred: std.atomic.Value(u8) = .init(0), // the address that answered last
};

pub const Cluster = struct {
    gpa: std.mem.Allocator,
    io: Io,
    peers: []Peer, // one a node, beside `topology.Plan.nodes`; this node's own is never asked
    self: usize,
    bearer: []const u8,
    interval_ms: u32,
    down_after: u32,
    tools: std.ArrayList(*NodeTool) = .empty,
    refresh: ?Refresh = null, // the scheduler, once it exists
    calls: ?Calls = null, // this front's open voice sessions
    stopping: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub const Refresh = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque, slot: usize) void };
    pub const Calls = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque) usize };

    pub fn init(gpa: std.mem.Allocator, io: Io, arena: std.mem.Allocator, p: *const topology.Plan, t: config.Topology, bearer: []const u8) !Cluster {
        const peers = try arena.alloc(Peer, p.nodes.len);
        for (peers, p.nodes) |*peer, *n| {
            const addrs = try arena.alloc([4]u8, n.addresses.len);
            for (addrs, n.addresses) |*d, s| d.* = topology.ip4(s).?; // the plan checked them
            peer.* = .{ .node = n, .addrs = addrs };
        }
        return .{ .gpa = gpa, .io = io, .peers = peers, .self = p.self, .bearer = bearer, .interval_ms = t.health_interval_ms, .down_after = t.down_after };
    }

    pub fn up(c: *const Cluster, node: usize) bool {
        return c.peers[node].up.load(.acquire);
    }

    /// Asks every other node once, before anything loads, so a tool starts on the right side.
    pub fn probeAll(c: *Cluster) void {
        for (c.peers, 0..) |*p, i| {
            if (i == c.self) continue;
            const ok = c.probe(p);
            p.up.store(ok, .release);
            std.log.info("node {s}: {s}", .{ p.node.id, if (ok) "up" else "down (no answer from its daemon)" });
        }
    }

    pub fn start(c: *Cluster) !void {
        c.thread = try std.Thread.spawn(.{}, run, .{c});
    }

    pub fn stop(c: *Cluster) void {
        c.stopping.store(true, .release);
        if (c.thread) |t| t.join();
        c.thread = null;
    }

    /// A request to `node` could not connect or broke off: it counts as down at once (the health checks bring it back).
    pub fn suspect(c: *Cluster, node: usize) void {
        const p = &c.peers[node];
        p.misses.store(c.down_after, .release);
        if (p.up.swap(false, .acq_rel)) {
            std.log.warn("node {s}: down (a request to it failed)", .{p.node.id});
            p.changed.store(true, .release);
        }
    }

    fn run(c: *Cluster) void {
        var next = nowMs(c.io);
        while (!c.stopping.load(.acquire)) {
            for (c.peers) |*p| if (p.changed.swap(false, .acq_rel)) c.moved(p);
            if (nowMs(c.io) >= next) {
                next = nowMs(c.io) + c.interval_ms;
                for (c.peers, 0..) |*p, i| if (i != c.self) c.check(p);
            }
            Io.sleep(c.io, .fromMilliseconds(50), .awake) catch {};
        }
    }

    fn check(c: *Cluster, p: *Peer) void {
        if (c.probe(p)) {
            p.misses.store(0, .release);
            if (!p.up.swap(true, .acq_rel)) {
                std.log.info("node {s}: up", .{p.node.id});
                p.changed.store(true, .release);
            }
            return;
        }
        const n = p.misses.fetchAdd(1, .acq_rel) + 1;
        if (n >= c.down_after and p.up.swap(false, .acq_rel)) {
            std.log.warn("node {s}: down ({d} health checks unanswered)", .{ p.node.id, n });
            p.changed.store(true, .release);
        }
    }

    /// `p` went down or came back: each of its tools that is loaded, or that falls back here, starts again on its new side.
    fn moved(c: *Cluster, p: *Peer) void {
        const r = c.refresh orelse return;
        for (c.tools.items) |t| {
            if (&c.peers[t.node] != p) continue;
            if (t.fallback == null and t.st.load(.acquire) == .unloaded) continue;
            r.f(r.ctx, t.slot);
        }
    }

    /// GET /v1/node/status on `p`, telling it this front's open calls; true on a 200.
    fn probe(c: *Cluster, p: *Peer) bool {
        const timeout: i32 = @intCast(@min(c.interval_ms, 2000));
        const fd = c.connect(p, timeout) catch return false;
        defer _ = std.c.close(fd);
        const calls: usize = if (c.calls) |k| k.f(k.ctx) else 0;
        upstream.sendWith(fd, "GET", "/v1/node/status", null, c.head(p, calls)) catch return false;
        var buf: [16 << 10]u8 = undefined;
        var r: upstream.Reader = .{ .fd = fd, .buf = &buf, .timeout_ms = timeout };
        const status = r.head() catch return false;
        while (r.next() catch return false) |_| {}
        return status == 200;
    }

    /// A connection to `p`'s daemon: the address that answered last first, then the others in order.
    pub fn connect(_: *Cluster, p: *Peer, timeout_ms: i32) upstream.Error!posix.fd_t {
        const first = p.preferred.load(.acquire);
        var last: upstream.Error = error.ConnectFailed;
        for (0..p.addrs.len) |k| {
            const i = (first + k) % p.addrs.len;
            const fd = upstream.connectTo(p.addrs[i], p.node.port, timeout_ms) catch |err| {
                last = err;
                continue;
            };
            p.preferred.store(@intCast(i), .release);
            return fd;
        }
        return last;
    }

    pub const Reply = struct { status: u16, body: []const u8 };

    /// One request to node `node`'s daemon, its whole reply in `a` (at most 16 MiB).
    pub fn request(c: *Cluster, a: std.mem.Allocator, node: usize, method: []const u8, path: []const u8, body: ?[]const u8, timeout_s: u32) upstream.Error!Reply {
        const p = &c.peers[node];
        const fd = try c.connect(p, 3000);
        defer _ = std.c.close(fd);
        try upstream.sendWith(fd, method, path, body, c.head(p, null));
        const buf = a.alloc(u8, 64 << 10) catch return error.TooLarge;
        var r: upstream.Reader = .{ .fd = fd, .buf = buf, .timeout_ms = timeoutMs(timeout_s) };
        const status = try r.head();
        return .{ .status = status, .body = try r.rest(a, 16 << 20) };
    }

    pub fn head(c: *const Cluster, p: *const Peer, calls: ?usize) upstream.Head {
        return .{ .host = p.node.addresses[p.preferred.load(.acquire)], .bearer = c.bearer, .calls = calls };
    }

    pub fn deinit(c: *Cluster) void {
        c.stop();
        c.tools.deinit(c.gpa);
    }
};

fn nowMs(io: Io) i64 {
    return @intCast(@divTrunc(Io.Clock.awake.now(io).toNanoseconds(), std.time.ns_per_ms));
}

pub const NodeTool = struct {
    cluster: *Cluster,
    cfg: *const ToolConfig,
    node: usize, // the node that runs it
    slot: usize = 0, // its scheduler slot, for `Scheduler.refresh`
    fallback: ?*process.ProcessTool = null, // runs it here while its node is down
    mode: std.atomic.Value(Mode) = .init(.remote), // where the loaded tool runs
    st: std.atomic.Value(tool.State) = .init(.unloaded), // remote: as the front sees it
    aborted: std.atomic.Value(bool) = .init(false),
    upstream_fd: std.atomic.Value(posix.fd_t) = .init(-1),
    fail_msg: [2048]u8 = undefined,
    fail: tool.Failure = .{},

    const Mode = enum(u8) { remote, local };

    pub fn asTool(t: *NodeTool) tool.Tool {
        return .{ .ptr = t, .vtable = &.{
            .config = config_,
            .needs = needs,
            .load = load,
            .generate = generate,
            .unload = unload,
            .abort = abort,
            .cancel = cancel,
            .state = state,
            .failure = failure,
            .local = local,
        } };
    }

    fn self(ptr: *anyopaque) *NodeTool {
        return @ptrCast(@alignCast(ptr));
    }

    fn peer(t: *NodeTool) *Peer {
        return &t.cluster.peers[t.node];
    }

    /// Where it runs: while loaded, where it was loaded; else here when its node is down and it has a fallback.
    fn local(ptr: *anyopaque) bool {
        const t = self(ptr);
        if (t.mode.load(.acquire) == .local and t.fallback.?.asTool().state() != .unloaded) return true;
        if (t.st.load(.acquire) != .unloaded) return false;
        return t.fallback != null and !t.cluster.up(t.node);
    }

    fn config_(ptr: *anyopaque) *const ToolConfig {
        return self(ptr).cfg;
    }

    fn needs(ptr: *anyopaque, req: *const Request) tool.Needs {
        const t = self(ptr);
        if (local(ptr)) return t.fallback.?.asTool().needs(req);
        return .{ .resident = 0, .working = 0 }; // its node admits it
    }

    fn state(ptr: *anyopaque) tool.State {
        const t = self(ptr);
        if (t.mode.load(.acquire) == .local) return t.fallback.?.asTool().state();
        return t.st.load(.acquire);
    }

    fn failure(ptr: *anyopaque) tool.Failure {
        const t = self(ptr);
        if (t.mode.load(.acquire) == .local) return t.fallback.?.asTool().failure();
        return t.fail;
    }

    fn load(ptr: *anyopaque) tool.Error!u64 {
        const t = self(ptr);
        if (!t.cluster.up(t.node)) {
            if (t.fallback) |fb| {
                std.log.info("{s}: node {s} is down, so it runs here (fallback)", .{ t.cfg.id, t.peer().node.id });
                t.mode.store(.local, .release);
                return fb.asTool().load();
            }
            return t.failed(.unloaded, tool.Error.LoadFailed, "model '{s}' runs on node {s}, which is down: its daemon has not answered its health checks", .{ t.cfg.id, t.peer().node.id });
        }
        t.mode.store(.remote, .release);
        t.st.store(.loading, .release);
        var arena: std.heap.ArenaAllocator = .init(t.cluster.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const path = std.fmt.allocPrint(a, "/v1/tools/{s}/load", .{t.cfg.id}) catch return tool.Error.OutOfMemory;
        const r = t.call(a, "POST", path, "{}", t.cfg.load_timeout_s) catch |err| {
            t.cluster.suspect(t.node);
            if (t.fallback) |fb| {
                std.log.info("{s}: node {s} did not answer ({s}), so it runs here (fallback)", .{ t.cfg.id, t.peer().node.id, @errorName(err) });
                t.st.store(.unloaded, .release);
                t.mode.store(.local, .release);
                return fb.asTool().load();
            }
            return t.failed(.unloaded, tool.Error.LoadFailed, "model '{s}' runs on node {s}, which did not answer: {s}", .{ t.cfg.id, t.peer().node.id, @errorName(err) });
        };
        if (r.status != 200) return t.failed(.unloaded, tool.Error.LoadFailed, "node {s} could not load '{s}': {s}", .{ t.peer().node.id, t.cfg.id, upstream.errorMessage(a, r.body) });
        t.st.store(.ready, .release);
        return 0; // its node holds the memory
    }

    fn unload(ptr: *anyopaque) void {
        const t = self(ptr);
        if (t.fallback) |fb| fb.asTool().unload();
        t.mode.store(.remote, .release); // the next load picks its side again
        t.st.store(.unloaded, .release); // its node keeps or drops it by its own rules
    }

    fn abort(ptr: *anyopaque) void {
        const t = self(ptr);
        if (t.mode.load(.acquire) == .local) return t.fallback.?.asTool().abort();
        t.aborted.store(true, .seq_cst);
        const fd = t.upstream_fd.load(.seq_cst);
        if (fd >= 0) _ = std.c.shutdown(fd, std.c.SHUT.RDWR);
    }

    fn cancel(ptr: *anyopaque) bool {
        const t = self(ptr);
        if (t.mode.load(.acquire) == .local) return t.fallback.?.asTool().cancel();
        return false; // `abort` ends a remote job and keeps the tool: its node cancels the job there
    }

    fn generate(ptr: *anyopaque, job: *const tool.Job, sink: tool.Sink, arena: std.mem.Allocator) tool.Error!tool.Output {
        const t = self(ptr);
        if (t.mode.load(.acquire) == .local) return t.fallback.?.asTool().generate(job, sink, arena);
        if (t.st.load(.acquire) == .unloaded) return tool.Error.WorkerDied;
        t.aborted.store(false, .seq_cst);
        t.st.store(.busy, .release);
        const out = if (job.request == .chat) t.chat(job, arena) else t.remoteJob(job, sink, arena);
        if (t.st.load(.acquire) == .busy) t.st.store(.ready, .release);
        return out;
    }

    /// A chat request, forwarded to the node's /v1/chat/completions as it is (its `model` is this tool's id); the reply
    /// goes into the job directory as it arrives, as a local chat server's does.
    fn chat(t: *NodeTool, job: *const tool.Job, arena: std.mem.Allocator) tool.Error!tool.Output {
        const v = job.request.chat;
        const name = if (v.stream) "completion.sse" else "completion.json";
        const start = Io.Clock.awake.now(t.cluster.io);
        const fd = t.cluster.connect(t.peer(), 3000) catch |err| return t.lost(err);
        defer _ = std.c.close(fd);
        t.upstream_fd.store(fd, .seq_cst);
        defer t.upstream_fd.store(-1, .seq_cst);
        if (t.aborted.load(.seq_cst)) return t.lost(error.Closed);
        upstream.sendWith(fd, "POST", "/v1/chat/completions", v.body, t.cluster.head(t.peer(), null)) catch |err| return t.lost(err);
        const buf = arena.alloc(u8, 64 << 10) catch return tool.Error.OutOfMemory;
        var r: upstream.Reader = .{ .fd = fd, .buf = buf, .timeout_ms = timeoutMs(t.cfg.generate_timeout_s) };
        const status = r.head() catch |err| return t.lost(err);
        if (status != 200) return t.refused(status, r.rest(arena, 64 << 10) catch "", arena);
        const path = std.fs.path.join(arena, &.{ job.dir, name }) catch return tool.Error.OutOfMemory;
        const file = Io.Dir.cwd().createFile(t.cluster.io, path, .{}) catch return t.failed(.ready, tool.Error.GenerateFailed, "could not write {s}", .{name});
        defer file.close(t.cluster.io);
        while (r.next() catch |err| return t.lost(err)) |bytes| {
            wire.writeAll(file.handle, bytes) catch return t.failed(.ready, tool.Error.GenerateFailed, "could not write {s}", .{name});
        }
        const files = arena.dupe([]const u8, &.{name}) catch return tool.Error.OutOfMemory;
        return .{ .files = files, .seed = v.seed, .ms = @intCast(start.durationTo(Io.Clock.awake.now(t.cluster.io)).toMilliseconds()) };
    }

    /// Any other kind: the request and the job directory's files (its inputs) go to the node as a job there, which is
    /// polled for progress until it ends; its output files come back into the job directory.
    fn remoteJob(t: *NodeTool, job: *const tool.Job, sink: tool.Sink, arena: std.mem.Allocator) tool.Error!tool.Output {
        const io = t.cluster.io;
        const File = struct { name: []const u8, data: []const u8 };
        var files: std.ArrayList(File) = .empty;
        {
            var dir = Io.Dir.cwd().openDir(io, job.dir, .{ .iterate = true }) catch return t.failed(.ready, tool.Error.GenerateFailed, "could not read the job directory", .{});
            defer dir.close(io);
            var it = dir.iterate();
            while (it.next(io) catch null) |e| {
                if (e.kind != .file) continue;
                const data = dir.readFileAlloc(io, e.name, arena, .limited(1 << 30)) catch return t.failed(.ready, tool.Error.GenerateFailed, "could not read input {s}", .{e.name});
                const enc = std.base64.standard.Encoder;
                const b64 = arena.alloc(u8, enc.calcSize(data.len)) catch return tool.Error.OutOfMemory;
                files.append(arena, .{ .name = arena.dupe(u8, e.name) catch return tool.Error.OutOfMemory, .data = enc.encode(b64, data) }) catch return tool.Error.OutOfMemory;
            }
        }
        const body = std.json.Stringify.valueAlloc(arena, .{ .tool = t.cfg.id, .request = job.request, .files = files.items }, .{ .emit_null_optional_fields = false }) catch return tool.Error.OutOfMemory;
        const created = t.call(arena, "POST", "/v1/node/jobs", body, 60) catch |err| return t.lost(err);
        if (created.status != 200) return t.refused(created.status, created.body, arena);
        const id = (std.json.parseFromSliceLeaky(struct { id: []const u8 }, arena, created.body, .{ .ignore_unknown_fields = true }) catch return t.failed(.ready, tool.Error.GenerateFailed, "node {s} answered the job with no id", .{t.peer().node.id})).id;
        const status_path = std.fmt.allocPrint(arena, "/v1/node/jobs/{s}", .{id}) catch return tool.Error.OutOfMemory;
        const Status = struct {
            status: []const u8,
            http_status: u16 = 0,
            error_type: []const u8 = "",
            error_message: []const u8 = "",
            percent: u32 = 0,
            phase: []const u8 = "",
            files: []const []const u8 = &.{},
            seed: u64 = 0,
            ms: u64 = 0,
            finish_reason: []const u8 = "",
            segments: []const u8 = "",
        };
        const deadline = nowMs(io) + @as(i64, t.cfg.generate_timeout_s) * 1000;
        const done: Status = while (true) {
            if (t.aborted.load(.seq_cst)) {
                const cancel_path = std.fmt.allocPrint(arena, "{s}/cancel", .{status_path}) catch return tool.Error.OutOfMemory;
                _ = t.call(arena, "POST", cancel_path, "{}", 10) catch {};
                t.st.store(.ready, .release);
                return tool.Error.Cancelled;
            }
            if (nowMs(io) > deadline) {
                const cancel_path = std.fmt.allocPrint(arena, "{s}/cancel", .{status_path}) catch return tool.Error.OutOfMemory;
                _ = t.call(arena, "POST", cancel_path, "{}", 10) catch {};
                return t.failed(.ready, tool.Error.Timeout, "deadline passed on node {s}", .{t.peer().node.id});
            }
            const r = t.call(arena, "GET", status_path, null, 30) catch |err| return t.lost(err);
            if (r.status != 200) return t.refused(r.status, r.body, arena);
            const s = std.json.parseFromSliceLeaky(Status, arena, r.body, .{ .ignore_unknown_fields = true }) catch return t.failed(.ready, tool.Error.GenerateFailed, "node {s} sent an unreadable job state", .{t.peer().node.id});
            if (!std.mem.eql(u8, s.status, "queued") and !std.mem.eql(u8, s.status, "in_progress")) break s;
            sink.report(.{ .phase = s.phase, .step = s.percent, .of = 100 });
            Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
        };
        if (!std.mem.eql(u8, done.status, "completed")) {
            if (std.mem.eql(u8, done.status, "cancelled")) {
                t.st.store(.ready, .release);
                return tool.Error.Cancelled;
            }
            return t.refusedAs(done.http_status, done.error_type, done.error_message);
        }
        for (done.files) |name| {
            if (!safeName(name)) return t.failed(.ready, tool.Error.GenerateFailed, "node {s} named an output '{s}'", .{ t.peer().node.id, name });
            const fp = std.fmt.allocPrint(arena, "/v1/files/{s}/{s}", .{ id, name }) catch return tool.Error.OutOfMemory;
            t.download(arena, fp, job.dir, name) catch |err| return t.lost(err);
        }
        return .{
            .files = arena.dupe([]const u8, done.files) catch return tool.Error.OutOfMemory,
            .seed = done.seed,
            .ms = done.ms,
            .finish_reason = done.finish_reason,
            .segments = done.segments,
        };
    }

    fn call(t: *NodeTool, a: std.mem.Allocator, method: []const u8, path: []const u8, body: ?[]const u8, timeout_s: u32) upstream.Error!Cluster.Reply {
        return t.cluster.request(a, t.node, method, path, body, timeout_s);
    }

    /// An output file of the node's job, streamed into the job directory.
    fn download(t: *NodeTool, a: std.mem.Allocator, path: []const u8, dir: []const u8, name: []const u8) upstream.Error!void {
        const io = t.cluster.io;
        const fd = try t.cluster.connect(t.peer(), 3000);
        defer _ = std.c.close(fd);
        try upstream.sendWith(fd, "GET", path, null, t.cluster.head(t.peer(), null));
        const buf = a.alloc(u8, 256 << 10) catch return error.TooLarge;
        var r: upstream.Reader = .{ .fd = fd, .buf = buf, .timeout_ms = timeoutMs(t.cfg.generate_timeout_s) };
        if (try r.head() != 200) return error.BadResponse;
        const out = std.fs.path.join(a, &.{ dir, name }) catch return error.TooLarge;
        const file = Io.Dir.cwd().createFile(io, out, .{}) catch return error.WriteFailed;
        defer file.close(io);
        while (try r.next()) |bytes| wire.writeAll(file.handle, bytes) catch return error.WriteFailed;
    }

    /// The node refused: its 4xx is the caller's (400), its 503 a busy model (503), anything else a failure.
    fn refused(t: *NodeTool, status: u16, body: []const u8, a: std.mem.Allocator) tool.Error {
        const msg = upstream.errorMessage(a, body);
        const typ = if (status >= 400 and status < 500 and status != 401 and status != 403) "invalid_request_error" else if (status == 503) "service_unavailable" else "server_error";
        t.fail = .{ .message = t.copyMsg("{s}", .{msg}), .type = typ };
        if (t.st.load(.acquire) == .busy) t.st.store(.ready, .release);
        return tool.Error.GenerateFailed;
    }

    /// A job that failed on the node: its status and error type as the node's scheduler set them.
    fn refusedAs(t: *NodeTool, status: u16, typ: []const u8, msg: []const u8) tool.Error {
        const kind = if (std.mem.eql(u8, typ, "invalid_request_error") or status == 400) "invalid_request_error" else if (status == 503) "service_unavailable" else "server_error";
        t.fail = .{ .message = t.copyMsg("{s}", .{msg}), .type = kind };
        if (t.st.load(.acquire) == .busy) t.st.store(.ready, .release);
        if (status == 504) return tool.Error.Timeout;
        return tool.Error.GenerateFailed;
    }

    /// A request that could not reach the node or broke off: cancelled (the tool stays), or the node is gone (it counts
    /// as down, and the tool is unloaded: its next load runs where it can).
    fn lost(t: *NodeTool, err: upstream.Error) tool.Error {
        if (t.aborted.load(.seq_cst)) {
            t.st.store(.ready, .release);
            return tool.Error.Cancelled;
        }
        if (err == error.Timeout) return t.failed(.ready, tool.Error.Timeout, "node {s} sent nothing for {d} s", .{ t.peer().node.id, t.cfg.generate_timeout_s });
        t.cluster.suspect(t.node);
        return t.failed(.unloaded, tool.Error.WorkerDied, "node {s}, which runs '{s}', is gone ({s})", .{ t.peer().node.id, t.cfg.id, @errorName(err) });
    }

    fn failed(t: *NodeTool, to: tool.State, err: tool.Error, comptime fmt: []const u8, args: anytype) tool.Error {
        t.fail = .{ .message = t.copyMsg(fmt, args), .type = "server_error" };
        t.st.store(to, .release);
        return err;
    }

    fn copyMsg(t: *NodeTool, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&t.fail_msg, fmt, args) catch t.fail_msg[0..];
    }
};

fn timeoutMs(s: u32) i32 {
    return @intCast(@min(@as(u64, s) * 1000, std.math.maxInt(i32)));
}

/// A file name a node may hand back: one plain component.
pub fn safeName(name: []const u8) bool {
    if (name.len == 0 or name.len > 255 or name[0] == '.') return false;
    for (name) |ch| if (ch == '/' or ch == '\\' or ch == 0) return false;
    return true;
}

test "output names from a node are single plain components" {
    for ([_][]const u8{ "0.png", "speech.wav", "completion.sse", "transcript.json" }) |n| try std.testing.expect(safeName(n));
    for ([_][]const u8{ "", ".", "..", "../x", "a/b", ".hidden", "a\\b" }) |n| try std.testing.expect(!safeName(n));
}
