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
            while (server.reader.state == .ready) {
                var req = server.receiveHead() catch return;
                handle(ctx, &req) catch |err| {
                    std.log.warn("{s} {s}: {s}", .{ @tagName(req.head.method), req.head.target, @errorName(err) });
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
