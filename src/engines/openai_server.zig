//! `openai_server`: a worker that runs a model's own OpenAI-compatible HTTP server (TensorFold's `serve`, a llama.cpp
//! server, ...) on loopback and supervises it; the daemon proxies each request to it (`ProcessTool`). Load starts
//! `options.server` (`{port}`, `{id}` and `{weights}` substituted) in the worker's process group with its output in the
//! tool's log, and waits until `options.health` (default /health) answers 200; the load reply names the port. It takes
//! no GPU lock: the scheduler holds the host's for every LocalRouter model and admits by memory alone. Unload stops the
//! server (SIGTERM, then SIGKILL after `stop_timeout_s`). If the daemon dies, the worker takes the server with it.
//! Memory is the config's `resident_bytes` and `working_bytes`: the model's measured figures, not guessed here.

const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;
const engine = @import("../engine/engine.zig");
const upstream = @import("../tool/upstream.zig");
const ToolConfig = @import("../config.zig").ToolConfig;
const Request = @import("../tool/request.zig").Request;

pub const entry: engine.Entry = .{ .name = "openai_server", .capabilities = &.{.chat_completion}, .needs = needs, .create = create };

fn needs(cfg: *const ToolConfig, _: *const Request) engine.Needs {
    return .{ .resident = cfg.resident_bytes, .working = cfg.working_bytes };
}

/// The running server's pid, for the SIGTERM handler (the daemon died: PR_SET_PDEATHSIG).
var server_pid: std.atomic.Value(posix.pid_t) = .init(0);

const Server = struct {
    env: engine.Env,
    cfg: *const ToolConfig,
    pid: posix.pid_t = 0,
    listen_port: u16 = 0,

    fn self(ptr: *anyopaque) *Server {
        return @ptrCast(@alignCast(ptr));
    }

    fn load(ptr: *anyopaque) anyerror!u64 {
        const s = self(ptr);
        var arena: std.heap.ArenaAllocator = .init(s.env.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const template = engine.optionList(s.cfg, "server") orelse {
            engine.explain("options.server is required: the command line of the model's server", .{});
            return error.NoServerCommand;
        };
        const fixed = engine.optionInt(s.cfg, "port", 0);
        s.listen_port = if (fixed > 0 and fixed < 65536) @intCast(fixed) else try upstream.freePort();
        const port_text = try std.fmt.allocPrint(a, "{d}", .{s.listen_port});
        const argv = try a.alloc([]const u8, template.len);
        for (argv, template) |*d, v| {
            if (v != .string) {
                engine.explain("options.server must be a list of strings", .{});
                return error.BadServerCommand;
            }
            d.* = try substitute(a, v.string, port_text, s.cfg);
        }
        var env = std.process.Environ.Map.init(a);
        if (s.env.environ) |parent| {
            var it = parent.iterator();
            while (it.next()) |kv| try env.put(kv.key_ptr.*, kv.value_ptr.*);
        }
        if (s.cfg.options.?.object.get("env")) |extra| if (extra == .object) {
            var it = extra.object.iterator();
            while (it.next()) |kv| if (kv.value_ptr.* == .string) try env.put(kv.key_ptr.*, kv.value_ptr.string);
        };
        // The daemon's death reaches this worker as SIGTERM (not the default SIGKILL), so it can stop the server first.
        const act: posix.Sigaction = .{ .handler = .{ .handler = onTerm }, .mask = posix.sigemptyset(), .flags = 0 };
        posix.sigaction(posix.SIG.TERM, &act, null);
        _ = linux.prctl(@backingInt(linux.PR.SET_PDEATHSIG), @backingInt(linux.SIG.TERM), 0, 0, 0);
        std.log.info("starting {s} on 127.0.0.1:{d}", .{ argv[0], s.listen_port });
        const child = std.process.spawn(s.env.io, .{ .argv = argv, .environ_map = &env, .stdin = .ignore, .stdout = .{ .file = std.Io.File.stderr() }, .stderr = .inherit }) catch |err| {
            engine.explain("could not start {s}: {s}", .{ argv[0], @errorName(err) });
            return err;
        };
        s.pid = child.id.?;
        server_pid.store(s.pid, .release);
        try s.awaitReady();
        std.log.info("server ready on 127.0.0.1:{d}", .{s.listen_port});
        return 0; // the scheduler keeps the config's resident_bytes, the measured figure
    }

    /// Polls the health path until it answers 200, the server exits, or the load timeout passes.
    fn awaitReady(s: *Server) !void {
        const path = engine.optionString(s.cfg, "health", "/health");
        const start = std.Io.Clock.awake.now(s.env.io);
        while (true) {
            var status: c_int = 0;
            if (std.c.waitpid(s.pid, &status, std.c.W.NOHANG) == s.pid) {
                s.pid = 0;
                server_pid.store(0, .release);
                engine.explain("the model's server exited before it was ready (wait status {d}); its output is in the tool's log", .{status});
                return error.ServerExited;
            }
            if (upstream.probe(s.listen_port, path, 2000)) |code| {
                if (code == 200) return;
            } else |_| {}
            const waited = start.durationTo(std.Io.Clock.awake.now(s.env.io)).toSeconds();
            if (waited >= s.cfg.load_timeout_s) {
                engine.explain("the model's server did not answer {s} with 200 within {d} s", .{ path, s.cfg.load_timeout_s });
                return error.ServerNotReady;
            }
            std.Io.sleep(s.env.io, .fromMilliseconds(250), .awake) catch {};
        }
    }

    fn generate(_: *anyopaque, _: *const engine.Job, _: engine.Sink, _: std.mem.Allocator) anyerror!engine.Output {
        engine.explain("requests go to the model's server through the daemon, not the worker protocol", .{});
        return error.NotProxied;
    }

    fn port(ptr: *anyopaque) u16 {
        return self(ptr).listen_port;
    }

    /// Stops the server (SIGTERM, SIGKILL after `stop_timeout_s`) and frees the engine.
    fn unload(ptr: *anyopaque) void {
        const s = self(ptr);
        if (s.pid > 0) {
            _ = std.c.kill(s.pid, posix.SIG.TERM);
            const grace_ms = engine.optionInt(s.cfg, "stop_timeout_s", 8) * 1000;
            var waited: u64 = 0;
            while (waited < grace_ms) : (waited += 50) {
                if (std.c.waitpid(s.pid, null, std.c.W.NOHANG) == s.pid) break;
                std.Io.sleep(s.env.io, .fromMilliseconds(50), .awake) catch {};
            } else {
                std.log.warn("the server did not stop within {d} ms: killing it", .{grace_ms});
                _ = std.c.kill(s.pid, posix.SIG.KILL);
                _ = std.c.waitpid(s.pid, null, 0);
            }
            server_pid.store(0, .release);
        }
        s.env.gpa.destroy(s);
    }
};

fn create(env: engine.Env, cfg: *const ToolConfig) anyerror!engine.Engine {
    const s = try env.gpa.create(Server);
    s.* = .{ .env = env, .cfg = cfg };
    return .{ .ptr = s, .vtable = &.{ .load = Server.load, .generate = Server.generate, .unload = Server.unload, .port = Server.port } };
}

/// The daemon is gone: stop the server now (it holds the model's memory), then exit.
fn onTerm(_: posix.SIG) callconv(.c) void {
    const pid = server_pid.load(.acquire);
    if (pid > 0) _ = std.c.kill(pid, posix.SIG.KILL);
    std.c._exit(0);
}

/// `arg` with `{port}`, `{id}` and `{weights}` replaced.
fn substitute(a: std.mem.Allocator, arg: []const u8, port_text: []const u8, cfg: *const ToolConfig) ![]const u8 {
    const with_port = try std.mem.replaceOwned(u8, a, arg, "{port}", port_text);
    const with_id = try std.mem.replaceOwned(u8, a, with_port, "{id}", cfg.id);
    return std.mem.replaceOwned(u8, a, with_id, "{weights}", cfg.weights);
}

test "the server's command line gets its port, id and weights" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: ToolConfig = .{ .id = "qwen3.8-27b", .kind = .chat, .weights = "/data/hf" };
    try std.testing.expectEqualStrings("--port=8123", try substitute(a, "--port={port}", "8123", &cfg));
    try std.testing.expectEqualStrings("qwen3.8-27b@/data/hf", try substitute(a, "{id}@{weights}", "1", &cfg));
    try std.testing.expectEqualStrings("serve", try substitute(a, "serve", "1", &cfg));
    try std.testing.expectEqual(@as(u64, 7), needs(&.{ .id = "c", .kind = .chat, .resident_bytes = 7 }, &.{ .chat = .{ .body = "{}" } }).resident);
}
