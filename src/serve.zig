//! `localrouter serve`: reads the config, builds one ProcessTool per tool, starts the scheduler's executor thread, and
//! serves the API until SIGINT or SIGTERM. The first stdout line names the address (tests read it for port 0).
//! In a multi-node topology (`topology.zig`) it serves this node's tools, and on the front a NodeTool for each tool
//! another node runs (`tool/node.zig`), with the health checks that move a fallback tool here while its node is down.

const std = @import("std");
const Io = std.Io;
const posix = std.posix;
const config = @import("config.zig");
const process = @import("tool/process.zig");
const jobs = @import("sched/jobs.zig");
const sched = @import("sched/scheduler.zig");
const routes = @import("api/routes.zig");
const desktop = @import("api/desktop.zig");
const http = @import("api/http.zig");
const bind = @import("bind.zig");
const engine = @import("engine/engine.zig");
const topology = @import("topology.zig");
const node = @import("tool/node.zig");
const ranks = @import("tool/ranks.zig");

var stop_flag: std.atomic.Value(bool) = .init(false);

pub fn run(gpa: std.mem.Allocator, io: Io, arena: std.mem.Allocator, args: []const []const u8, environ: *const std.process.Environ.Map) !u8 {
    var path: ?[]const u8 = null;
    var host: ?[]const u8 = null;
    var port: ?u16 = null;
    var data: ?[]const u8 = null;
    var node_id: ?[]const u8 = null;
    var i: usize = 0;
    while (i + 1 < args.len) : (i += 2) {
        const a, const v = .{ args[i], args[i + 1] };
        if (std.mem.eql(u8, a, "--config")) path = v else if (std.mem.eql(u8, a, "--host")) host = v else if (std.mem.eql(u8, a, "--port")) {
            port = std.fmt.parseInt(u16, v, 10) catch return 2;
        } else if (std.mem.eql(u8, a, "--data")) data = v else if (std.mem.eql(u8, a, "--node")) node_id = v else {
            std.log.err("unknown option {s}", .{a});
            return 2;
        }
    }
    if (i != args.len) {
        std.log.err("option {s} needs a value", .{args[i]});
        return 2;
    }
    var cfg: config.Config = .{};
    if (path) |p| {
        const text = Io.Dir.cwd().readFileAlloc(io, p, arena, .limited(1 << 20)) catch |err| {
            std.log.err("{s}: {s}", .{ p, @errorName(err) });
            return 2;
        };
        cfg = config.parse(arena, text) catch |err| {
            std.log.err("{s}: {s}", .{ p, @errorName(err) });
            return 2;
        };
    }
    var why_buf: [512]u8 = undefined;
    var why: Io.Writer = .fixed(&why_buf);
    const plan = (try topology.plan(arena, &cfg, node_id, &why)) orelse {
        std.log.err("{s}: {s}", .{ path orelse "config", why.buffered() });
        return 2;
    };
    if (!plan.front) topology.trimDefaults(&plan, &cfg.defaults); // a node serves its own tools only
    const tools = try arena.dupe(config.ToolConfig, plan.tools);
    // what each tool's weights can do, before anything lists it; a remote tool's weights are on its node, which checks
    for (tools, plan.places, 0..) |*t, pl, k| {
        if (pl != .remote) try engine.resolve(io, arena, tools[k .. k + 1]);
        if (pl == .follower) { // another rank of a group: it loads when rank 0 asks, and stays until rank 0 stops it
            t.keep_loaded = false;
            t.idle_ttl_s = std.math.maxInt(u32);
        }
    }
    cfg.tools = tools;
    if (!engine.checkDefaults(&cfg, &why)) {
        std.log.err("{s}: {s}", .{ path orelse "config", why.buffered() });
        return 2;
    }
    if (host) |v| cfg.host = v;
    if (port) |v| cfg.port = v;
    if (data) |v| cfg.data_dir = v;
    if (cfg.machine.len == 0) cfg.machine = hostname(arena);

    const self_exe = try std.process.executablePathAlloc(io, arena);
    const jobs_dir = try std.fs.path.join(arena, &.{ cfg.data_dir, "jobs" });
    const log_dir = try std.fs.path.join(arena, &.{ cfg.data_dir, "logs" });
    try Io.Dir.cwd().createDirPath(io, jobs_dir);
    // jobs from before this start are in no table, so nothing would expire their outputs: remove the old ones now
    const swept = jobs.sweep(io, gpa, jobs_dir, sched.nowUnix(io), cfg.keep_outputs_s) catch |err| blk: {
        std.log.warn("jobs: sweeping {s} failed: {s}", .{ jobs_dir, @errorName(err) });
        break :blk 0;
    };
    if (swept > 0) std.log.info("jobs: removed {d} job directories older than {d} s from before this start", .{ swept, cfg.keep_outputs_s });
    const gate_path = try std.fs.path.join(arena, &.{ cfg.data_dir, "locks", "speech.gate" }); // speech first (sched/speech_gate.zig)
    try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(gate_path).?);

    var bearer: []const u8 = "";
    if (plan.multi and cfg.topology.token_file.len > 0) {
        const text = Io.Dir.cwd().readFileAlloc(io, cfg.topology.token_file, arena, .limited(4096)) catch |err| {
            std.log.err("topology.token_file {s}: {s}", .{ cfg.topology.token_file, @errorName(err) });
            return 2;
        };
        bearer = std.mem.trim(u8, text, " \t\r\n");
    }
    var cluster = try node.Cluster.init(gpa, io, arena, &plan, cfg.topology, bearer);
    defer cluster.deinit();

    const procs = try arena.alloc(process.ProcessTool, cfg.tools.len);
    const remotes = try arena.alloc(node.NodeTool, cfg.tools.len);
    const leads = try arena.alloc(ranks.RankTool, cfg.tools.len);
    const slots = try arena.alloc(sched.Slot, cfg.tools.len);
    for (procs, remotes, leads, slots, cfg.tools, plan.places, 0..) |*p, *r, *l, *s, *t, pl, k| {
        const rank: ?process.Rank = switch (pl) {
            .lead, .follower => |g| .{ .rank = g.rank, .world = g.world(), .master = plan.nodes[g.nodes[0]].addresses[0] },
            else => null,
        };
        p.* = try process.ProcessTool.init(.{ .gpa = gpa, .io = io, .self_exe = self_exe, .environ = environ, .log_dir = log_dir, .gpu_lock = cfg.gpu_lock, .speech_gate = gate_path, .rank = rank }, t);
        switch (pl) {
            .local, .follower => s.* = .{ .tool = p.asTool() },
            .lead => |g| {
                l.* = .{ .cluster = &cluster, .rank0 = p, .followers = g.nodes[1..] };
                s.* = .{ .tool = l.asTool() };
            },
            .remote, .fallback => |n| {
                r.* = .{ .cluster = &cluster, .cfg = t, .node = n, .slot = k, .fallback = if (pl == .fallback) p else null };
                try cluster.tools.append(gpa, r);
                s.* = .{ .tool = r.asTool() };
            },
        }
    }
    defer for (procs) |*p| p.deinit();
    if (plan.multi) {
        std.log.info("topology: node {s} ({s}), {d} tools served here", .{ plan.nodes[plan.self].id, if (plan.front) "the front" else "a node", cfg.tools.len });
        if (plan.front) cluster.probeAll(); // before anything loads: a tool starts on the side that can run it
    }

    var table: jobs.Table = .{ .gpa = gpa };
    defer table.deinit();
    var scheduler: sched.Scheduler = .{ .io = io, .gpa = gpa, .cfg = &cfg, .slots = slots, .table = &table, .speech = .init(gate_path) };
    defer scheduler.deinit();
    var app: routes.App = .{ .gpa = gpa, .io = io, .cfg = &cfg, .scheduler = &scheduler, .table = &table, .jobs_dir = jobs_dir };
    scheduler.notify = app.voice.notifier(io, gpa); // voice sessions hear when a voice model is unloaded to make room
    scheduler.loaded = .{ .ctx = &app, .f = @import("api/voice_local.zig").warmLoaded }; // a kept speech model's voice warms as it loads
    defer app.voice.deinit(gpa);
    try app.cloned.init(io, gpa, try std.fs.path.join(arena, &.{ cfg.data_dir, "voices" })); // cloned voices (voices.zig)
    defer app.cloned.deinit();
    app.assistant.init(io, gpa, cfg.data_dir, cfg.assistant.name); // the assistant's name (assistant.zig)
    app.plan = &plan;
    if (plan.multi and plan.front) {
        cluster.refresh = .{ .ctx = &scheduler, .f = refreshSlot };
        cluster.calls = .{ .ctx = &app, .f = openCalls };
        try cluster.start();
    }
    var computer: desktop.Service = undefined;
    if (cfg.computer.enabled) {
        try computer.init(&app, cfg.data_dir, arena);
        app.computer = &computer;
    }
    defer if (app.computer) |c| c.deinit();

    // `host` may be an IP, localhost, all or tailscale (see bind.zig); listen on the resolved address
    const ifaces: []const bind.Iface = if (bind.needsInterfaces(cfg.host)) bind.listInterfaces(arena) catch &.{} else &.{};
    const listen_ip = bind.resolve(arena, cfg.host, ifaces) catch |err| {
        switch (err) {
            error.NoTailnetAddress => std.log.err("no tailnet address found: is tailscale up?", .{}),
            error.InvalidHost => std.log.err("host '{s}' is not an IP address, localhost, all or tailscale", .{cfg.host}),
            error.OutOfMemory => return err,
        }
        return 2;
    };
    const address = try Io.net.IpAddress.parse(listen_ip, cfg.port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    app.self_addr = selfAddress(server.socket.address);
    var line_buf: [128]u8 = undefined;
    const line = try std.fmt.bufPrint(&line_buf, "localrouter listening on http://{s}:{d}\n", .{ listen_ip, server.socket.address.getPort() });
    try Io.File.stdout().writeStreamingAll(io, line);

    installSignals();
    const exec = try std.Thread.spawn(.{}, sched.Scheduler.run, .{ &scheduler, sched.Lane.main });
    const exec_rt = try std.Thread.spawn(.{}, sched.Scheduler.run, .{ &scheduler, sched.Lane.realtime });
    const exec_chat = try std.Thread.spawn(.{}, sched.Scheduler.run, .{ &scheduler, sched.Lane.chat });
    const exec_listen = try std.Thread.spawn(.{}, sched.Scheduler.run, .{ &scheduler, sched.Lane.listen });
    const exec_decide = try std.Thread.spawn(.{}, sched.Scheduler.run, .{ &scheduler, sched.Lane.decide });
    const watcher = try std.Thread.spawn(.{}, watchStop, .{ io, &server, &app });
    http.Server(routes.App, routes.handle).serve(io, &server, &app, &stop_flag);
    cluster.stop(); // before the scheduler goes: the health checks move tools on it
    scheduler.stop();
    exec.join();
    exec_rt.join();
    exec_chat.join();
    exec_listen.join();
    exec_decide.join();
    watcher.join();
    return 0;
}

fn refreshSlot(ctx: *anyopaque, slot: usize) void {
    const s: *sched.Scheduler = @ptrCast(@alignCast(ctx));
    s.refresh(slot);
}

fn openCalls(ctx: *anyopaque) usize {
    const app: *routes.App = @ptrCast(@alignCast(ctx));
    return app.voice.count(app.io);
}

fn onSignal(_: posix.SIG) callconv(.c) void {
    stop_flag.store(true, .release);
}

fn installSignals() void {
    const act: posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(posix.SIG.INT, &act, null);
    posix.sigaction(posix.SIG.TERM, &act, null);
}

/// Once a signal set the flag: ends the voice sessions' reads, then wakes the accept loop. On Linux shutdown closes the
/// socket under `accept`; on macOS it fails (ENOTCONN on a listening socket) and `accept` stays blocked, so there the
/// daemon connects to itself once (its own bound address: the kernel answers at once): `accept` returns, and the loop
/// sees the flag.
fn watchStop(io: Io, server: *Io.net.Server, app: *routes.App) void {
    while (!stop_flag.load(.acquire)) Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    app.voice.closeAll(io);
    if (app.computer) |c| c.closeAll();
    if (@import("builtin").os.tag != .linux) {
        if (app.self_addr.connect(io, .{ .mode = .stream })) |s| s.close(io) else |err| std.log.warn("waking the accept loop: {s}", .{@errorName(err)});
        return;
    }
    _ = posix.system.shutdown(server.socket.handle, posix.SHUT.RDWR);
}

fn hostname(a: std.mem.Allocator) []const u8 {
    var buf: [posix.HOST_NAME_MAX]u8 = undefined;
    const name = posix.gethostname(&buf) catch return "localhost";
    return a.dupe(u8, name) catch "localhost";
}

/// How the daemon reaches itself: the address it listens on, or loopback for a wildcard bind.
fn selfAddress(listen: Io.net.IpAddress) Io.net.IpAddress {
    return switch (listen) {
        .ip4 => |x| if (std.mem.allEqual(u8, &x.bytes, 0)) .{ .ip4 = .loopback(x.port) } else listen,
        .ip6 => |x| if (std.mem.allEqual(u8, &x.bytes, 0)) .{ .ip6 = .loopback(x.port) } else listen,
    };
}
