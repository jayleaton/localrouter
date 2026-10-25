//! A small HTTP/1.1 client for a model's own server on loopback (127.0.0.1:<port>), and for another node's daemon
//! (`connectTo`, `sendWith`): one request a connection, raw sockets so another thread can end a request (`shutdown`),
//! and a deadline on every read (`poll`). A response body may be framed by Content-Length, by chunked transfer coding,
//! or by the end of the connection.

const std = @import("std");
const posix = std.posix;
const c = std.c;

pub const Error = error{ ConnectFailed, Closed, Timeout, BadResponse, ReadFailed, WriteFailed, TooLarge };

/// A TCP connection to 127.0.0.1:`port`; the caller closes it.
pub fn connect(port: u16) Error!posix.fd_t {
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM | c.SOCK.CLOEXEC, 0);
    if (fd < 0) return error.ConnectFailed;
    const addr: c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f00_0001) };
    while (true) {
        if (c.connect(fd, @ptrCast(&addr), @sizeOf(c.sockaddr.in)) == 0) return fd;
        if (posix.errno(@as(c_int, -1)) == .INTR) continue;
        _ = c.close(fd);
        return error.ConnectFailed;
    }
}

/// A TCP connection to `addr`:`port` within `timeout_ms` (a node that is down must not hold the caller for the
/// kernel's minutes of retries); the caller closes it. The socket is blocking again once connected.
pub fn connectTo(addr: [4]u8, port: u16, timeout_ms: i32) Error!posix.fd_t {
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM | c.SOCK.CLOEXEC | c.SOCK.NONBLOCK, 0);
    if (fd < 0) return error.ConnectFailed;
    errdefer _ = c.close(fd);
    const sa: c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(addr) };
    if (c.connect(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) != 0) {
        const e = posix.errno(@as(c_int, -1));
        if (e != .INPROGRESS and e != .INTR) return error.ConnectFailed;
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.OUT, .revents = 0 }};
        const ready = posix.poll(&fds, timeout_ms) catch return error.ConnectFailed;
        if (ready == 0) return error.Timeout;
        var soerr: c_int = 0;
        var len: c.socklen_t = @sizeOf(c_int);
        if (c.getsockopt(fd, c.SOL.SOCKET, c.SO.ERROR, @ptrCast(&soerr), &len) != 0 or soerr != 0) return error.ConnectFailed;
    }
    const flags = c.fcntl(fd, c.F.GETFL);
    if (flags < 0 or c.fcntl(fd, c.F.SETFL, flags & ~@as(c_int, @bitCast(c.O{ .NONBLOCK = true }))) < 0) return error.ConnectFailed;
    return fd;
}

/// What `sendWith` puts in the request head beyond the defaults.
pub const Head = struct {
    host: []const u8 = "127.0.0.1",
    bearer: []const u8 = "", // an `Authorization: Bearer` token; "": none
    calls: ?usize = null, // `x-localrouter-calls`: the front's open voice sessions (`tool/node.zig`)
};

/// A loopback port nothing listens on now (bind to port 0 and read the one the kernel chose).
pub fn freePort() Error!u16 {
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM | c.SOCK.CLOEXEC, 0);
    if (fd < 0) return error.ConnectFailed;
    defer _ = c.close(fd);
    var addr: c.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f00_0001) };
    if (c.bind(fd, @ptrCast(&addr), @sizeOf(c.sockaddr.in)) != 0) return error.ConnectFailed;
    var len: c.socklen_t = @sizeOf(c.sockaddr.in);
    if (c.getsockname(fd, @ptrCast(&addr), &len) != 0) return error.ConnectFailed;
    return std.mem.bigToNative(u16, addr.port);
}

/// Sends a request: a JSON `body` (POST) or none (GET). `Connection: close`, so the server may end a body at EOF.
pub fn send(fd: posix.fd_t, method: []const u8, path: []const u8, body: ?[]const u8) Error!void {
    return sendWith(fd, method, path, body, .{});
}

/// `send` with another Host, a bearer token or the front's call count (`Head`).
pub fn sendWith(fd: posix.fd_t, method: []const u8, path: []const u8, body: ?[]const u8, extra: Head) Error!void {
    var head_buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&head_buf);
    w.print("{s} {s} HTTP/1.1\r\nHost: {s}\r\nAccept: */*\r\n", .{ method, path, extra.host }) catch return error.WriteFailed;
    if (extra.bearer.len > 0) w.print("Authorization: Bearer {s}\r\n", .{extra.bearer}) catch return error.WriteFailed;
    if (extra.calls) |n| w.print("x-localrouter-calls: {d}\r\n", .{n}) catch return error.WriteFailed;
    if (body) |b| w.print("Content-Type: application/json\r\nContent-Length: {d}\r\n", .{b.len}) catch return error.WriteFailed;
    w.writeAll("Connection: close\r\n\r\n") catch return error.WriteFailed;
    try writeAll(fd, w.buffered());
    if (body) |b| try writeAll(fd, b);
}

fn writeAll(fd: posix.fd_t, bytes: []const u8) Error!void {
    var done: usize = 0;
    while (done < bytes.len) {
        const rc = c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (rc > 0) {
            done += @intCast(rc);
            continue;
        }
        switch (posix.errno(rc)) {
            .INTR, .AGAIN => continue,
            else => return error.WriteFailed,
        }
    }
}

/// Reads one response from `fd`: `head` first (the status), then `next` until it returns null. Body slices point into
/// `buf` and stay valid until the next call. `timeout_ms` bounds each wait for bytes (< 0: no limit).
pub const Reader = struct {
    fd: posix.fd_t,
    buf: []u8,
    timeout_ms: i32,
    start: usize = 0,
    end: usize = 0,
    framing: enum { length, chunked, close } = .close,
    left: u64 = 0, // bytes left in the body (length) or in the current chunk (chunked)
    crlf: bool = false, // chunked: the current chunk's data is followed by a CRLF still to read
    done: bool = false,

    /// The status line and the headers; returns the status code.
    pub fn head(r: *Reader) Error!u16 {
        while (true) {
            if (std.mem.indexOf(u8, r.buf[r.start..r.end], "\r\n\r\n")) |at| {
                const text = r.buf[r.start..][0..at];
                r.start += at + 4;
                const status = try r.parseHead(text);
                if (status >= 100 and status < 200) continue; // an interim response: the real one follows
                return status;
            }
            try r.fill();
        }
    }

    fn parseHead(r: *Reader, text: []const u8) Error!u16 {
        var lines = std.mem.splitSequence(u8, text, "\r\n");
        const status_line = lines.first();
        if (!std.mem.startsWith(u8, status_line, "HTTP/1.") or status_line.len < 12) return error.BadResponse;
        const status = std.fmt.parseInt(u16, status_line[9..12], 10) catch return error.BadResponse;
        r.framing = .close;
        r.done = false;
        while (lines.next()) |field| {
            const colon = std.mem.indexOfScalar(u8, field, ':') orelse continue;
            const name = std.mem.trim(u8, field[0..colon], " \t");
            const value = std.mem.trim(u8, field[colon + 1 ..], " \t");
            if (std.ascii.eqlIgnoreCase(name, "transfer-encoding") and std.ascii.findIgnoreCase(value, "chunked") != null) {
                r.framing = .chunked;
                r.left = 0;
            } else if (std.ascii.eqlIgnoreCase(name, "content-length") and r.framing != .chunked) {
                r.framing = .length;
                r.left = std.fmt.parseInt(u64, value, 10) catch return error.BadResponse;
            }
        }
        if (status == 204 or status == 304 or (status >= 100 and status < 200)) r.done = true; // no body
        return status;
    }

    /// The next bytes of the body, or null at its end.
    pub fn next(r: *Reader) Error!?[]const u8 {
        if (r.done) return null;
        switch (r.framing) {
            .close => {
                if (r.start == r.end) r.fill() catch |err| switch (err) {
                    error.Closed => {
                        r.done = true;
                        return null;
                    },
                    else => return err,
                };
                return r.take(r.end - r.start);
            },
            .length => {
                if (r.left == 0) {
                    r.done = true;
                    return null;
                }
                if (r.start == r.end) try r.fill();
                return r.take(@intCast(@min(r.left, r.end - r.start)));
            },
            .chunked => {
                if (r.left == 0) {
                    if (r.crlf) {
                        if ((try r.line()).len != 0) return error.BadResponse;
                        r.crlf = false;
                    }
                    const size_line = try r.line();
                    const hex = std.mem.trim(u8, size_line[0 .. std.mem.indexOfScalar(u8, size_line, ';') orelse size_line.len], " \t");
                    const size = std.fmt.parseInt(u64, hex, 16) catch return error.BadResponse;
                    if (size == 0) { // the last chunk, then trailers up to an empty line
                        while ((try r.line()).len != 0) {}
                        r.done = true;
                        return null;
                    }
                    r.left = size;
                    r.crlf = true;
                }
                if (r.start == r.end) try r.fill();
                return r.take(@intCast(@min(r.left, r.end - r.start)));
            },
        }
    }

    /// The whole remaining body (at most `max` bytes, the rest dropped), in `a`.
    pub fn rest(r: *Reader, a: std.mem.Allocator, max: usize) Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        while (try r.next()) |bytes| {
            const n = @min(bytes.len, max - out.items.len);
            out.appendSlice(a, bytes[0..n]) catch return error.TooLarge;
        }
        return out.items;
    }

    fn take(r: *Reader, n: usize) []const u8 {
        const s = r.buf[r.start..][0..n];
        r.start += n;
        if (r.framing != .close) r.left -= n;
        return s;
    }

    /// One CRLF-terminated line without its terminator, valid until the next call.
    fn line(r: *Reader) Error![]const u8 {
        while (true) {
            if (std.mem.indexOfScalar(u8, r.buf[r.start..r.end], '\n')) |nl| {
                const s = std.mem.trimEnd(u8, r.buf[r.start..][0..nl], "\r");
                r.start += nl + 1;
                return s;
            }
            try r.fill();
        }
    }

    /// Reads more bytes into the buffer (moving unread ones to its front first), waiting at most `timeout_ms`.
    fn fill(r: *Reader) Error!void {
        if (r.start > 0) {
            std.mem.copyForwards(u8, r.buf, r.buf[r.start..r.end]);
            r.end -= r.start;
            r.start = 0;
        }
        if (r.end == r.buf.len) return error.TooLarge;
        var fds = [_]posix.pollfd{.{ .fd = r.fd, .events = posix.POLL.IN, .revents = 0 }};
        while (true) {
            const ready = posix.poll(&fds, r.timeout_ms) catch return error.ReadFailed;
            if (ready == 0) return error.Timeout;
            const rc = c.read(r.fd, r.buf[r.end..].ptr, r.buf.len - r.end);
            if (rc > 0) {
                r.end += @intCast(rc);
                return;
            }
            if (rc == 0) return error.Closed;
            switch (posix.errno(rc)) {
                .INTR, .AGAIN => continue,
                else => return error.ReadFailed,
            }
        }
    }
};

/// The status of `GET path` on 127.0.0.1:`port` (a readiness probe), within `timeout_ms`.
pub fn probe(port: u16, path: []const u8, timeout_ms: i32) Error!u16 {
    const fd = try connect(port);
    defer _ = c.close(fd);
    try send(fd, "GET", path, null);
    var buf: [4096]u8 = undefined;
    var r: Reader = .{ .fd = fd, .buf = &buf, .timeout_ms = timeout_ms };
    return r.head();
}

/// The reason in an error body: OpenAI's `error.message`, a top-level `message` or `detail`, else the text itself
/// (at most 1 KiB).
pub fn errorMessage(a: std.mem.Allocator, body: []const u8) []const u8 {
    const raw = std.mem.trim(u8, body[0..@min(body.len, 1024)], " \t\r\n");
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return raw;
    if (v != .object) return raw;
    if (v.object.get("error")) |e| switch (e) {
        .string => |s| return s,
        .object => |o| if (o.get("message")) |m| if (m == .string) return m.string,
        else => {},
    };
    for ([_][]const u8{ "message", "detail" }) |k| if (v.object.get(k)) |m| if (m == .string) return m.string;
    return raw;
}

test "responses framed by length, chunks or the end of the connection" {
    const cases = [_]struct { wire: []const u8, status: u16, body: []const u8 }{
        .{ .wire = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"ok\":true}", .status = 200, .body = "{\"ok\":true}" },
        .{ .wire = "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\ndata: \r\n3;x=y\r\n{}\n\r\n1\r\n\n\r\n0\r\nX-Trailer: 1\r\n\r\n", .status = 200, .body = "data: {}\n\n" },
        .{ .wire = "HTTP/1.0 400 Bad Request\r\n\r\n{\"error\":{\"message\":\"n must be 1\"}}", .status = 400, .body = "{\"error\":{\"message\":\"n must be 1\"}}" },
    };
    for (cases) |case| {
        var fds: [2]posix.fd_t = undefined;
        try std.testing.expectEqual(@as(c_int, 0), c.pipe(&fds));
        defer _ = c.close(fds[0]);
        try writeAll(fds[1], case.wire);
        _ = c.close(fds[1]);
        var buf: [128]u8 = undefined;
        var r: Reader = .{ .fd = fds[0], .buf = &buf, .timeout_ms = 1000 };
        try std.testing.expectEqual(case.status, try r.head());
        var bodies: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer bodies.deinit();
        try std.testing.expectEqualStrings(case.body, try r.rest(bodies.allocator(), 1 << 20));
        try std.testing.expect((try r.next()) == null);
    }
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("n must be 1", errorMessage(arena.allocator(), "{\"error\":{\"message\":\"n must be 1\",\"type\":\"invalid_request_error\"}}"));
    try std.testing.expectEqualStrings("busy", errorMessage(arena.allocator(), "{\"message\":\"busy\"}"));
    try std.testing.expectEqualStrings("plain text", errorMessage(arena.allocator(), " plain text\n"));
}

test "a loopback connection and a readiness probe" {
    try std.testing.expectError(error.ConnectFailed, connect(try freePort())); // nothing listens there
}
