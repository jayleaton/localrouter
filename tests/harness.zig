//! The integration tests' harness: a real `localrouter` daemon on an ephemeral port, and helpers to call it.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;

pub const mib: u64 = 1 << 20;

/// A daemon on an ephemeral port with its own data directory, killed and cleaned up at `stop`.
pub const Daemon = struct {
    io: Io,
    gpa: std.mem.Allocator,
    child: std.process.Child,
    port: u16,
    dir: []const u8,
    client: std.http.Client,

    pub fn start(gpa: std.mem.Allocator, io: Io, name: []const u8, cfg_json: []const u8) !Daemon {
        const dir = try std.fmt.allocPrint(gpa, "/tmp/localrouter-it-{s}", .{name});
        Io.Dir.cwd().deleteTree(io, dir) catch {};
        try Io.Dir.cwd().createDirPath(io, dir);
        const cfg_path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
        defer gpa.free(cfg_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = cfg_path, .data = cfg_json });
        const data = try std.fmt.allocPrint(gpa, "{s}/data", .{dir});
        defer gpa.free(data);
        var child = try std.process.spawn(io, .{ .argv = &.{ opts.localrouter_exe, "serve", "--config", cfg_path, "--host", "127.0.0.1", "--port", "0", "--data", data }, .stdout = .pipe, .stderr = .inherit });
        var buf: [256]u8 = undefined;
        var r = child.stdout.?.reader(io, &buf);
        const line = try r.interface.takeDelimiterExclusive('\n');
        const colon = std.mem.lastIndexOfScalar(u8, line, ':').?;
        const port = try std.fmt.parseInt(u16, line[colon + 1 ..], 10);
        return .{ .io = io, .gpa = gpa, .child = child, .port = port, .dir = dir, .client = .{ .allocator = gpa, .io = io } };
    }

    pub fn stop(d: *Daemon) void {
        d.client.deinit();
        _ = std.posix.system.kill(d.child.id.?, std.posix.SIG.TERM);
        _ = d.child.wait(d.io) catch {};
        Io.Dir.cwd().deleteTree(d.io, d.dir) catch {};
        d.gpa.free(d.dir);
    }

    pub const Resp = struct { status: std.http.Status, body: []u8 };

    pub fn call(d: *Daemon, a: std.mem.Allocator, method: std.http.Method, path: []const u8, body: ?[]const u8) !Resp {
        return d.callAs(a, method, path, body, null);
    }

    /// `call` with a request content type (multipart bodies).
    pub fn callAs(d: *Daemon, a: std.mem.Allocator, method: std.http.Method, path: []const u8, body: ?[]const u8, content_type: ?[]const u8) !Resp {
        var out: Io.Writer.Allocating = .init(a);
        const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}{s}", .{ d.port, path });
        const res = try d.client.fetch(.{ .location = .{ .url = url }, .method = method, .payload = body, .response_writer = &out.writer, .keep_alive = false, .headers = .{ .content_type = if (content_type) |c| .{ .override = c } else .default } });
        return .{ .status = res.status, .body = out.written() };
    }

    /// The state of tool `id` from /v1/tools.
    pub fn toolState(d: *Daemon, a: std.mem.Allocator, id: []const u8) ![]const u8 {
        const r = try d.call(a, .GET, "/v1/tools", null);
        const T = struct { data: []const struct { id: []const u8, state: []const u8 } };
        const p = try std.json.parseFromSliceLeaky(T, a, r.body, .{ .ignore_unknown_fields = true });
        for (p.data) |t| if (std.mem.eql(u8, t.id, id)) return t.state;
        return error.NoSuchTool;
    }

    /// Pids of the daemon's worker processes (children whose PPid is the daemon).
    pub fn workers(d: *Daemon, a: std.mem.Allocator) ![]i32 {
        var list: std.ArrayList(i32) = .empty;
        var proc = try Io.Dir.cwd().openDir(d.io, "/proc", .{ .iterate = true });
        defer proc.close(d.io);
        var it = proc.iterate();
        while (try it.next(d.io)) |e| {
            const pid = std.fmt.parseInt(i32, e.name, 10) catch continue;
            var buf: [4096]u8 = undefined;
            const status = readProc(try std.fmt.allocPrintSentinel(a, "/proc/{d}/status", .{pid}, 0), &buf) orelse continue;
            const at = std.mem.indexOf(u8, status, "PPid:\t") orelse continue;
            const end = std.mem.indexOfScalarPos(u8, status, at, '\n') orelse continue;
            const ppid = std.fmt.parseInt(i32, status[at + 6 .. end], 10) catch continue;
            if (ppid == d.child.id.? and std.mem.indexOf(u8, status, "Name:\tlocalrouter") != null) try list.append(a, pid);
        }
        return list.items;
    }
};

/// A /proc file (they report size 0, so read until EOF with plain reads).
pub fn readProc(path: [:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, path, .{}, 0) catch return null;
    defer _ = std.posix.system.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const got = std.posix.read(fd, buf[n..]) catch return null;
        if (got == 0) break;
        n += got;
    }
    return buf[0..n];
}

pub fn image(d: *Daemon, a: std.mem.Allocator, model: []const u8, extra: []const u8) !Daemon.Resp {
    const size = if (std.mem.indexOf(u8, extra, "\"size\"") == null) ",\"size\":\"256x256\"" else "";
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"prompt\":\"bars\"{s}{s}}}", .{ model, size, extra });
    return d.call(a, .POST, "/v1/images/generations", body);
}

pub fn firstPng(a: std.mem.Allocator, body: []const u8) ![]u8 {
    const T = struct { data: []const struct { b64_json: []const u8 } };
    const p = try std.json.parseFromSliceLeaky(T, a, body, .{ .ignore_unknown_fields = true });
    const dec = std.base64.standard.Decoder;
    const png = try a.alloc(u8, try dec.calcSizeForSlice(p.data[0].b64_json));
    try dec.decode(png, p.data[0].b64_json);
    return png;
}

pub fn sleepMs(io: Io, ms: i64) void {
    Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
}

pub const reserve = "\"reserve_bytes\": 67108864";


/// One MCP tools/call; the result object (content, isError).
pub fn mcpCall(d: *Daemon, a: std.mem.Allocator, tool: []const u8, args: []const u8) !std.json.Value {
    const body = try std.fmt.allocPrint(a, "{{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{{\"name\":\"{s}\",\"arguments\":{s}}}}}", .{ tool, args });
    const r = try d.call(a, .POST, "/mcp", body);
    try testing.expectEqual(std.http.Status.ok, r.status);
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, r.body, .{});
    return v.object.get("result") orelse error.NoResult;
}

/// The first text content of an MCP result, parsed as JSON.
pub fn mcpText(a: std.mem.Allocator, result: std.json.Value) !std.json.Value {
    for (result.object.get("content").?.array.items) |c|
        if (std.mem.eql(u8, c.object.get("type").?.string, "text")) return std.json.parseFromSliceLeaky(std.json.Value, a, c.object.get("text").?.string, .{});
    return error.NoText;
}

