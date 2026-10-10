//! `localrouter serve`: reads the config, builds one ProcessTool per tool, starts the scheduler's executor thread, and
//! serves the API until SIGINT or SIGTERM. The first stdout line names the address (tests read it for port 0).

const std = @import("std");
const Io = std.Io;
const posix = std.posix;
const config = @import("config.zig");
const process = @import("tool/process.zig");
const jobs = @import("sched/jobs.zig");
const sched = @import("sched/scheduler.zig");
const routes = @import("api/routes.zig");
const http = @import("api/http.zig");
const bind = @import("bind.zig");
const engine = @import("engine/engine.zig");

var stop_flag: std.atomic.Value(bool) = .init(false);

pub fn run(gpa: std.mem.Allocator, io: Io, arena: std.mem.Allocator, args: []const []const u8, environ: *const std.process.Environ.Map) !u8 {
    var path: ?[]const u8 = null;
    var host: ?[]const u8 = null;
    var port: ?u16 = null;
    var data: ?[]const u8 = null;
    var i: usize = 0;
    while (i + 1 < args.len) : (i += 2) {
        const a, const v = .{ args[i], args[i + 1] };
        if (std.mem.eql(u8, a, "--config")) path = v else if (std.mem.eql(u8, a, "--host")) host = v else if (std.mem.eql(u8, a, "--port")) {
            port = std.fmt.parseInt(u16, v, 10) catch return 2;
        } else if (std.mem.eql(u8, a, "--data")) data = v else {
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
    var why_buf: [256]u8 = undefined;
    var why: Io.Writer = .fixed(&why_buf);
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

    const procs = try arena.alloc(process.ProcessTool, cfg.tools.len);
    const slots = try arena.alloc(sched.Slot, cfg.tools.len);
    for (procs, slots, cfg.tools) |*p, *s, *t| {
        p.* = try process.ProcessTool.init(.{ .gpa = gpa, .io = io, .self_exe = self_exe, .environ = environ, .log_dir = log_dir }, t);
        s.* = .{ .tool = p.asTool() };
    }
    defer for (procs) |*p| p.deinit();

    var table: jobs.Table = .{ .gpa = gpa };
    defer table.deinit();
    var scheduler: sched.Scheduler = .{ .io = io, .gpa = gpa, .cfg = &cfg, .slots = slots, .table = &table };
    defer scheduler.deinit();
    var app: routes.App = .{ .gpa = gpa, .io = io, .cfg = &cfg, .scheduler = &scheduler, .table = &table, .jobs_dir = jobs_dir };

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
    const exec = try std.Thread.spawn(.{}, sched.Scheduler.run, .{&scheduler});
    const watcher = try std.Thread.spawn(.{}, watchStop, .{ io, &server });
    http.Server(routes.App, routes.handle).serve(io, &server, &app, &stop_flag);
    scheduler.stop();
    exec.join();
    watcher.join();
    return 0;
}

fn onSignal(_: posix.SIG) callconv(.c) void {
    stop_flag.store(true, .release);
}

fn installSignals() void {
    const act: posix.Sigaction = .{ .handler = .{ .handler = onSignal }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(posix.SIG.INT, &act, null);
    posix.sigaction(posix.SIG.TERM, &act, null);
}

/// Wakes the accept loop once a signal set the flag (shutdown closes the listening socket).
fn watchStop(io: Io, server: *Io.net.Server) void {
    while (!stop_flag.load(.acquire)) Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
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
