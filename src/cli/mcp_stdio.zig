//! `localrouter mcp-stdio [--url http://host:8190]`: the daemon's /mcp for clients that only speak MCP over stdio. A proxy
//! with no state: each JSON-RPC line on stdin is POSTed to `<url>/mcp` and the response is written as one line on
//! stdout (on one line). Notifications (the daemon answers 202) produce no output. A daemon that cannot be reached or answers
//! something else yields a JSON-RPC error for the request's id, so the client does not hang. The URL defaults to
//! $LOCALROUTER_URL, else http://127.0.0.1:8190.

const std = @import("std");
const Io = std.Io;

pub fn run(gpa: std.mem.Allocator, io: Io, args: []const []const u8, environ: *const std.process.Environ.Map) !u8 {
    var base: []const u8 = environ.get("LOCALROUTER_URL") orelse "http://127.0.0.1:8190";
    if (args.len == 2 and std.mem.eql(u8, args[0], "--url")) base = args[1] else if (args.len != 0) {
        std.debug.print("usage: localrouter mcp-stdio [--url http://host:8190]\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const endpoint = try std.fmt.allocPrint(gpa, "{s}/mcp", .{std.mem.trimEnd(u8, base, "/")});
    defer gpa.free(endpoint);
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    var in_buf: [64 * 1024]u8 = undefined;
    var stdin = Io.File.stdin().reader(io, &in_buf);
    const r = &stdin.interface;
    while (true) {
        var line: Io.Writer.Allocating = .init(gpa);
        defer line.deinit();
        _ = r.streamDelimiterEnding(&line.writer, '\n') catch return 1;
        const eof = r.seek == r.end;
        if (std.mem.trim(u8, line.written(), " \t\r").len > 0) {
            _ = arena.reset(.retain_capacity);
            try forward(&client, io, arena.allocator(), endpoint, std.mem.trim(u8, line.written(), "\r"));
        }
        if (eof) return 0;
        r.toss(1);
    }
}

/// POSTs one message and writes the answer, if any.
fn forward(client: *std.http.Client, io: Io, a: std.mem.Allocator, endpoint: []const u8, msg: []const u8) !void {
    var out: Io.Writer.Allocating = .init(a);
    const res = client.fetch(.{
        .location = .{ .url = endpoint },
        .method = .POST,
        .payload = msg,
        .response_writer = &out.writer,
        .headers = .{ .content_type = .{ .override = "application/json" } },
        .extra_headers = &.{.{ .name = "accept", .value = "application/json, text/event-stream" }},
    }) catch |err| return reply(io, a, msg, try std.fmt.allocPrint(a, "cannot reach {s}: {s}", .{ endpoint, @errorName(err) }));
    if (res.status == .accepted or out.written().len == 0 and res.status == .ok) return;
    if (res.status != .ok) return reply(io, a, msg, try std.fmt.allocPrint(a, "{s} answered {d}", .{ endpoint, @intFromEnum(res.status) }));
    // One line per message: newlines in valid JSON are only whitespace between tokens (the schemas have some).
    const text = try a.dupe(u8, std.mem.trim(u8, out.written(), " \r\n"));
    std.mem.replaceScalar(u8, text, '\n', ' ');
    try Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(a, "{s}\n", .{text}));
}

/// A JSON-RPC error (-32000) for `msg`'s id; nothing when `msg` is a notification (no id).
fn reply(io: Io, a: std.mem.Allocator, msg: []const u8, text: []const u8) !void {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, msg, .{}) catch return;
    if (v != .object) return;
    const id = v.object.get("id") orelse return;
    const out = try std.json.Stringify.valueAlloc(a, .{ .jsonrpc = "2.0", .id = id, .@"error" = .{ .code = -32000, .message = text } }, .{});
    try Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(a, "{s}\n", .{out}));
}
