//! The HTTP side: the listening socket, one concurrent task per connection, and response helpers
//! (JSON bodies and OpenAI's error envelope). Routing lives in `routes.zig`.

const std = @import("std");
const Io = std.Io;
const http = std.http;

pub const Request = http.Server.Request;

pub const max_body: usize = 64 << 20; // edits upload up to 5 images

/// Serves connections until `stop` is set; `handle(ctx, request)` runs once per request.
pub fn Server(comptime Ctx: type, comptime handle: fn (*Ctx, *Request) anyerror!void) type {
    return struct {
        pub fn serve(io: Io, server: *Io.net.Server, ctx: *Ctx, stop: *std.atomic.Value(bool)) void {
            var group: Io.Group = .init;
            defer group.cancel(io);
            while (!stop.load(.acquire)) {
                const stream = server.accept(io) catch |err| {
                    if (stop.load(.acquire)) return;
                    std.log.warn("accept: {s}", .{@errorName(err)});
                    continue;
                };
                group.concurrent(io, connection, .{ io, stream, ctx }) catch {
                    stream.close(io); // no thread for it: refuse rather than block the accept loop
                };
            }
        }

        fn connection(io: Io, stream: Io.net.Stream, ctx: *Ctx) void {
            defer stream.close(io);
            var recv: [16 * 1024]u8 = undefined;
            var send: [16 * 1024]u8 = undefined;
            var reader = stream.reader(io, &recv);
            var writer = stream.writer(io, &send);
            var server = http.Server.init(&reader.interface, &writer.interface);
            requests(&server, ctx);
        }

        /// Every request on one connection, until it closes or a handler fails.
        fn requests(server: *http.Server, ctx: *Ctx) void {
            while (server.reader.state == .ready) {
                var req = server.receiveHead() catch return;
                // RFC 9112 6.3: a request with neither Content-Length nor Transfer-Encoding has no body. Say so, or std's
                // server reads to end of stream for the body and asserts before keeping the connection alive.
                if (req.head.method.requestHasBody() and req.head.transfer_encoding == .none and req.head.content_length == null)
                    req.head.content_length = 0;
                // Reading the body invalidates the head's strings, so keep a copy of the target to report a failure.
                var target_buf: [256]u8 = undefined;
                const target = target_buf[0..@min(req.head.target.len, target_buf.len)];
                @memcpy(target, req.head.target[0..target.len]);
                handle(ctx, &req) catch |err| {
                    std.log.warn("{s} {s}: {s}", .{ @tagName(req.head.method), target, @errorName(err) });
                    return;
                };
            }
        }
    };
}

/// Reads the whole request body (at most `max_body`).
pub fn body(req: *Request, gpa: std.mem.Allocator) ![]u8 {
    if (req.head.content_length) |n| if (n > max_body) return error.BodyTooLarge;
    var buf: [8 * 1024]u8 = undefined;
    const r = try req.readerExpectContinue(&buf);
    return r.allocRemaining(gpa, .limited(max_body)) catch |err| switch (err) {
        error.StreamTooLong => error.BodyTooLarge,
        else => err,
    };
}

pub fn json(req: *Request, status: http.Status, text: []const u8) !void {
    try req.respond(text, .{ .status = status, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
}

/// Serializes `value` and sends it.
pub fn jsonValue(req: *Request, gpa: std.mem.Allocator, status: http.Status, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(gpa, value, .{ .emit_null_optional_fields = false });
    defer gpa.free(text);
    try json(req, status, text);
}

/// OpenAI's error envelope: {"error":{"message","type","param","code"}}.
pub fn fail(req: *Request, gpa: std.mem.Allocator, status: u16, typ: []const u8, message: []const u8) !void {
    const Env = struct { @"error": struct { message: []const u8, type: []const u8, param: ?[]const u8 = null, code: ?[]const u8 = null } };
    const text = try std.json.Stringify.valueAlloc(gpa, Env{ .@"error" = .{ .message = message, .type = typ } }, .{});
    defer gpa.free(text);
    try json(req, @enumFromInt(status), text);
}

pub fn bytes(req: *Request, content_type: []const u8, data: []const u8) !void {
    try req.respond(data, .{ .extra_headers = &.{.{ .name = "content-type", .value = content_type }} });
}

/// The path without its query string.
pub fn path(req: *const Request) []const u8 {
    const t = req.head.target;
    return t[0 .. std.mem.indexOfScalar(u8, t, '?') orelse t.len];
}

/// The Host header, for absolute URLs in responses; only valid before the body is read.
pub fn host(req: *const Request) []const u8 {
    var it = req.iterateHeaders();
    while (it.next()) |h| if (std.ascii.eqlIgnoreCase(h.name, "host")) return h.value;
    return "localhost";
}

test "a bodiless POST keeps the connection serving" {
    const Ctx = struct {
        n: usize = 0,
        fn handle(c: *@This(), req: *Request) anyerror!void {
            c.n += 1;
            if (std.mem.eql(u8, req.head.target, "/read")) {
                const b = try body(req, std.testing.allocator);
                defer std.testing.allocator.free(b);
                try std.testing.expectEqual(@as(usize, 0), b.len);
            }
            try req.respond("ok", .{});
        }
    };
    // curl -X POST without -d (no Content-Length) on one kept-alive connection: answered unread, then read, then a GET
    var in: Io.Reader = .fixed("POST /v1/tools/x/release HTTP/1.1\r\nHost: a\r\n\r\nPOST /read HTTP/1.1\r\nHost: a\r\n\r\nGET /health HTTP/1.1\r\nHost: a\r\nConnection: close\r\n\r\n");
    var out_buf: [1024]u8 = undefined;
    var out: Io.Writer = .fixed(&out_buf);
    var server = http.Server.init(&in, &out);
    var ctx: Ctx = .{};
    Server(Ctx, Ctx.handle).requests(&server, &ctx);
    try std.testing.expectEqual(@as(usize, 3), ctx.n);
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, out.buffered(), "HTTP/1.1 200 OK"));
}

test "a handler failing after reading the body is logged without a crash" {
    const Ctx = struct {
        fn handle(_: *@This(), req: *Request) anyerror!void {
            const b = try body(req, std.testing.allocator); // invalidates req.head's strings
            std.testing.allocator.free(b);
            return error.ClientGone;
        }
    };
    var in: Io.Reader = .fixed("POST /v1/images/generations HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\n\r\n{}");
    var out_buf: [256]u8 = undefined;
    var out: Io.Writer = .fixed(&out_buf);
    var server = http.Server.init(&in, &out);
    var ctx: Ctx = .{};
    Server(Ctx, Ctx.handle).requests(&server, &ctx);
    try std.testing.expectEqual(@as(usize, 0), out.buffered().len);
}
