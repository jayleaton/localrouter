//! RFC 6455 framing for the WebSocket endpoint once `Request.respondWebSocket` has done the handshake: client frames
//! read off the HTTP reader (fragments reassembled, pings answered) and unmasked frames sent under a mutex.

const std = @import("std");
const Io = std.Io;

/// Largest control frame payload (RFC 6455 5.5).
const max_control = 125;

pub const Opcode = enum(u4) { continuation = 0, text = 1, binary = 2, close = 8, ping = 9, pong = 10, _ };

/// A complete data message; `data` is valid until the next `read`.
pub const Message = struct { opcode: Opcode, data: []const u8 };

pub const Error = error{ Closed, Protocol, TooLarge, ReadFailed, WriteFailed, EndOfStream, OutOfMemory };

/// One server connection after the handshake.
pub const Conn = struct {
    io: Io,
    gpa: std.mem.Allocator,
    in: *Io.Reader,
    out: *Io.Writer,
    max_message: usize = 4 << 20,
    write_mutex: Io.Mutex = .init,
    msg: std.ArrayList(u8) = .empty,
    closed: std.atomic.Value(bool) = .init(false), // a close frame was sent

    /// Frees the message buffer.
    pub fn deinit(c: *Conn) void {
        c.msg.deinit(c.gpa);
    }

    /// The next text or binary message (fragments reassembled); pings are answered, pongs skipped, protocol errors closed.
    pub fn read(c: *Conn) Error!Message {
        c.msg.clearRetainingCapacity();
        var kind: ?Opcode = null;
        while (true) {
            const h = try readHead(c.in);
            if (h.mask == null or h.rsv != 0) return c.fail(1002, error.Protocol);
            switch (h.opcode) {
                .ping, .pong, .close => {
                    if (!h.fin or h.len > max_control) return c.fail(1002, error.Protocol);
                    var ctl: [max_control]u8 = undefined;
                    const body = ctl[0..@as(usize, @intCast(h.len))];
                    try readBody(c.in, h.mask, body);
                    switch (h.opcode) {
                        .ping => c.send(.pong, body) catch |err| switch (err) {
                            error.Closed => {}, // no pong after our close
                            else => return err,
                        },
                        .close => {
                            if (body.len == 1) return c.fail(1002, error.Protocol);
                            c.sendClose(body[0..@as(usize, @min(body.len, 2))]);
                            return error.Closed;
                        },
                        else => {},
                    }
                },
                .text, .binary, .continuation => {
                    if ((h.opcode == .continuation) != (kind != null)) return c.fail(1002, error.Protocol);
                    if (h.len > c.max_message - c.msg.items.len) return c.fail(1009, error.TooLarge);
                    const at = c.msg.items.len;
                    try c.msg.resize(c.gpa, at + @as(usize, @intCast(h.len)));
                    try readBody(c.in, h.mask, c.msg.items[at..]);
                    if (kind == null) kind = h.opcode;
                    if (h.fin) return .{ .opcode = kind.?, .data = c.msg.items };
                },
                _ => return c.fail(1002, error.Protocol),
            }
        }
    }

    /// Sends one unmasked frame (fin set) and flushes; thread-safe. error.Closed after a close frame was sent.
    pub fn send(c: *Conn, opcode: Opcode, data: []const u8) Error!void {
        c.write_mutex.lockUncancelable(c.io);
        defer c.write_mutex.unlock(c.io);
        if (c.closed.load(.acquire)) return error.Closed;
        if (opcode == .close) c.closed.store(true, .release);
        try writeFrame(c.out, opcode, true, null, data);
        try c.out.flush();
    }

    /// Sends a text message.
    pub fn sendText(c: *Conn, data: []const u8) Error!void {
        return c.send(.text, data);
    }

    /// Sends a close frame with `code` and `reason` once (best effort, never fails).
    pub fn close(c: *Conn, code: u16, reason: []const u8) void {
        var p: [max_control]u8 = undefined;
        std.mem.writeInt(u16, p[0..2], code, .big);
        const n: usize = @min(reason.len, max_control - 2);
        @memcpy(p[2..][0..n], reason[0..n]);
        c.sendClose(p[0 .. 2 + n]);
    }

    fn sendClose(c: *Conn, payload: []const u8) void {
        c.send(.close, payload) catch {};
    }

    fn fail(c: *Conn, code: u16, err: Error) Error {
        c.close(code, "");
        return err;
    }
};

/// One frame as read off the wire (either direction), payload unmasked.
pub const Frame = struct { fin: bool, opcode: Opcode, data: []const u8 };

/// Writes one frame; `mask` set: a client frame (payload XORed with it).
pub fn writeFrame(w: *Io.Writer, opcode: Opcode, fin: bool, mask: ?[4]u8, data: []const u8) Io.Writer.Error!void {
    const b0: u8 = @as(u8, if (fin) 0x80 else 0) | @as(u8, @backingInt(opcode));
    const m: u8 = if (mask != null) 0x80 else 0;
    try w.writeByte(b0);
    if (data.len < 126) {
        try w.writeByte(m | @as(u8, @intCast(data.len)));
    } else if (data.len <= 0xffff) {
        try w.writeByte(m | 126);
        try w.writeInt(u16, @intCast(data.len), .big);
    } else {
        try w.writeByte(m | 127);
        try w.writeInt(u64, data.len, .big);
    }
    const key = mask orelse return w.writeAll(data);
    try w.writeAll(&key);
    var chunk: [512]u8 = undefined;
    var at: usize = 0;
    while (at < data.len) {
        const n: usize = @min(chunk.len, data.len - at);
        for (chunk[0..n], data[at..][0..n], at..) |*o, b, i| o.* = b ^ key[i & 3];
        try w.writeAll(chunk[0..n]);
        at += n;
    }
}

/// Reads one frame into `buf` (cleared first; at most `max` payload bytes, else error.TooLarge), unmasking a masked payload.
pub fn readFrame(r: *Io.Reader, gpa: std.mem.Allocator, buf: *std.ArrayList(u8), max: usize) Error!Frame {
    buf.clearRetainingCapacity();
    const h = try readHead(r);
    if (h.len > max) return error.TooLarge;
    try buf.resize(gpa, @intCast(h.len));
    try readBody(r, h.mask, buf.items);
    return .{ .fin = h.fin, .opcode = h.opcode, .data = buf.items };
}

const Head = struct { fin: bool, rsv: u3, opcode: Opcode, mask: ?[4]u8, len: u64 };

fn readHead(r: *Io.Reader) Error!Head {
    const b = try r.takeArray(2);
    const b0 = b[0];
    const b1 = b[1];
    const len7: u7 = @truncate(b1);
    const len: u64 = switch (len7) {
        126 => try r.takeInt(u16, .big),
        127 => try r.takeInt(u64, .big),
        else => len7,
    };
    const mask: ?[4]u8 = if (b1 & 0x80 != 0) (try r.takeArray(4)).* else null;
    const op: Opcode = @fromBackingInt(@as(u4, @truncate(b0)));
    return .{ .fin = b0 & 0x80 != 0, .rsv = @truncate(b0 >> 4), .opcode = op, .mask = mask, .len = len };
}

/// Fills `dst` from `r` (any length, independent of the reader's buffer) and unmasks it.
fn readBody(r: *Io.Reader, mask: ?[4]u8, dst: []u8) Error!void {
    try r.readSliceAll(dst);
    const key = mask orelse return;
    for (dst, 0..) |*b, i| b.* ^= key[i & 3];
}

const testing = std.testing;
const test_mask: [4]u8 = .{ 0x37, 0xfa, 0x21, 0x3d };

fn testConn(in: *Io.Reader, out: *Io.Writer) Conn {
    return .{ .io = testing.io, .gpa = testing.allocator, .in = in, .out = out };
}

/// The frames the server wrote, read back one at a time.
const Sent = struct {
    r: Io.Reader,
    buf: std.ArrayList(u8) = .empty,

    fn init(bytes: []const u8) Sent {
        return .{ .r = .fixed(bytes) };
    }

    fn next(s: *Sent) !Frame {
        return readFrame(&s.r, testing.allocator, &s.buf, 1 << 20);
    }

    fn deinit(s: *Sent) void {
        s.buf.deinit(testing.allocator);
    }
};

fn expectClose(bytes: []const u8, code: u16) !void {
    var s: Sent = .init(bytes);
    defer s.deinit();
    const f = try s.next();
    try testing.expect(f.fin);
    try testing.expectEqual(Opcode.close, f.opcode);
    try testing.expect(f.data.len >= 2);
    try testing.expectEqual(code, std.mem.readInt(u16, f.data[0..2], .big));
    try testing.expectError(error.EndOfStream, s.next());
}

test "a masked text frame reads back as one message" {
    var wire: Io.Writer.Allocating = .init(testing.allocator);
    defer wire.deinit();
    try writeFrame(&wire.writer, .text, true, test_mask, "hello");
    const w = wire.written();
    try testing.expectEqual(@as(u8, 0x81), w[0]);
    try testing.expectEqual(@as(u8, 0x85), w[1]);
    try testing.expectEqualSlices(u8, &test_mask, w[2..6]);
    try testing.expect(!std.mem.eql(u8, w[6..], "hello"));

    var in: Io.Reader = .fixed(w);
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&in, &out.writer);
    defer c.deinit();
    const m = try c.read();
    try testing.expectEqual(Opcode.text, m.opcode);
    try testing.expectEqualStrings("hello", m.data);
    try testing.expectError(error.EndOfStream, c.read());
    try testing.expectEqual(@as(usize, 0), out.written().len);
}

test "16-bit and 64-bit lengths, payloads larger than the reader's buffer" {
    const mid = try testing.allocator.alloc(u8, 300);
    defer testing.allocator.free(mid);
    const big = try testing.allocator.alloc(u8, 70000);
    defer testing.allocator.free(big);
    for (mid, 0..) |*b, i| b.* = @truncate(i *% 7);
    for (big, 0..) |*b, i| b.* = @truncate(i *% 13 +% 5);

    var wire: Io.Writer.Allocating = .init(testing.allocator);
    defer wire.deinit();
    try writeFrame(&wire.writer, .text, true, test_mask, mid);
    const mid_len = wire.written().len;
    try writeFrame(&wire.writer, .binary, true, test_mask, big);
    const w = wire.written();
    try testing.expectEqual(@as(u8, 0x80 | 126), w[1]);
    try testing.expectEqual(@as(u16, 300), std.mem.readInt(u16, w[2..4], .big));
    try testing.expectEqual(@as(usize, 2 + 2 + 4 + 300), mid_len);
    try testing.expectEqual(@as(u8, 0x82), w[mid_len]);
    try testing.expectEqual(@as(u8, 0x80 | 127), w[mid_len + 1]);
    try testing.expectEqual(@as(u64, 70000), std.mem.readInt(u64, w[mid_len + 2 ..][0..8], .big));

    var src: Io.Reader = .fixed(w);
    var small: [64]u8 = undefined;
    var lim = src.limited(.unlimited, &small); // a reader whose buffer is far smaller than the messages
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&lim.interface, &out.writer);
    defer c.deinit();
    const a = try c.read();
    try testing.expectEqual(Opcode.text, a.opcode);
    try testing.expectEqualSlices(u8, mid, a.data);
    const b = try c.read();
    try testing.expectEqual(Opcode.binary, b.opcode);
    try testing.expectEqualSlices(u8, big, b.data);
    try testing.expectError(error.EndOfStream, c.read());
}

test "a fragmented message with a ping between fragments" {
    var wire: Io.Writer.Allocating = .init(testing.allocator);
    defer wire.deinit();
    try writeFrame(&wire.writer, .text, false, test_mask, "hel");
    try writeFrame(&wire.writer, .ping, true, test_mask, "p1");
    try writeFrame(&wire.writer, .continuation, false, test_mask, "lo, ");
    try writeFrame(&wire.writer, .pong, true, test_mask, "ignored");
    try writeFrame(&wire.writer, .continuation, true, test_mask, "world");

    var in: Io.Reader = .fixed(wire.written());
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&in, &out.writer);
    defer c.deinit();
    const m = try c.read();
    try testing.expectEqual(Opcode.text, m.opcode);
    try testing.expectEqualStrings("hello, world", m.data);

    var s: Sent = .init(out.written());
    defer s.deinit();
    const pong = try s.next();
    try testing.expect(pong.fin);
    try testing.expectEqual(Opcode.pong, pong.opcode);
    try testing.expectEqualStrings("p1", pong.data);
    try testing.expectError(error.EndOfStream, s.next());
}

test "a close frame is answered with its code, then error.Closed" {
    var wire: Io.Writer.Allocating = .init(testing.allocator);
    defer wire.deinit();
    try writeFrame(&wire.writer, .close, true, test_mask, &.{ 0x03, 0xe9, 'b', 'y', 'e' }); // 1001 "bye"

    var in: Io.Reader = .fixed(wire.written());
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&in, &out.writer);
    defer c.deinit();
    try testing.expectError(error.Closed, c.read());
    try testing.expect(c.closed.load(.acquire));
    try expectClose(out.written(), 1001);
}

test "an unmasked client frame is a protocol error closed with 1002" {
    var wire: Io.Writer.Allocating = .init(testing.allocator);
    defer wire.deinit();
    try writeFrame(&wire.writer, .text, true, null, "hi");

    var in: Io.Reader = .fixed(wire.written());
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&in, &out.writer);
    defer c.deinit();
    try testing.expectError(error.Protocol, c.read());
    try expectClose(out.written(), 1002);
}

test "other protocol errors close with 1002" {
    const Bad = struct { op: Opcode, fin: bool, len: usize, rsv: u8 = 0 };
    const cases = [_][]const Bad{
        &.{.{ .op = .continuation, .fin = true, .len = 3 }}, // no message started
        &.{ .{ .op = .text, .fin = false, .len = 3 }, .{ .op = .binary, .fin = true, .len = 3 } }, // new data frame inside
        &.{.{ .op = .ping, .fin = false, .len = 3 }}, // fragmented control
        &.{.{ .op = .ping, .fin = true, .len = 126 }}, // control over 125 bytes
        &.{.{ .op = @fromBackingInt(@as(u4, 3)), .fin = true, .len = 3 }}, // unknown opcode
        &.{.{ .op = .text, .fin = true, .len = 3, .rsv = 0x40 }}, // reserved bit
    };
    const payload: [200]u8 = @splat('x');
    for (cases) |frames| {
        var wire: Io.Writer.Allocating = .init(testing.allocator);
        defer wire.deinit();
        for (frames) |f| {
            const at = wire.written().len;
            try writeFrame(&wire.writer, f.op, f.fin, test_mask, payload[0..f.len]);
            wire.written()[at] |= f.rsv;
        }
        var in: Io.Reader = .fixed(wire.written());
        var out: Io.Writer.Allocating = .init(testing.allocator);
        defer out.deinit();
        var c = testConn(&in, &out.writer);
        defer c.deinit();
        try testing.expectError(error.Protocol, c.read());
        try expectClose(out.written(), 1002);
    }
}

test "a message over max_message is closed with 1009" {
    var wire: Io.Writer.Allocating = .init(testing.allocator);
    defer wire.deinit();
    try writeFrame(&wire.writer, .text, false, test_mask, "0123456789");
    try writeFrame(&wire.writer, .continuation, true, test_mask, "0123456789");

    var in: Io.Reader = .fixed(wire.written());
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&in, &out.writer);
    defer c.deinit();
    c.max_message = 16;
    try testing.expectError(error.TooLarge, c.read());
    try expectClose(out.written(), 1009);
}

test "send after close is error.Closed and close is sent once" {
    var in: Io.Reader = .fixed(&.{});
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&in, &out.writer);
    defer c.deinit();
    try c.sendText("first");
    const long: [300]u8 = @splat('r');
    c.close(1000, &long);
    c.close(1011, "again");
    try testing.expectError(error.Closed, c.sendText("late"));
    try testing.expectError(error.Closed, c.send(.binary, "late"));

    var s: Sent = .init(out.written());
    defer s.deinit();
    const t = try s.next();
    try testing.expectEqual(Opcode.text, t.opcode);
    try testing.expectEqualStrings("first", t.data);
    const cl = try s.next();
    try testing.expectEqual(Opcode.close, cl.opcode);
    try testing.expectEqual(@as(usize, max_control), cl.data.len);
    try testing.expectEqual(@as(u16, 1000), std.mem.readInt(u16, cl.data[0..2], .big));
    try testing.expectError(error.EndOfStream, s.next());
}

test "readFrame reads an unmasked server frame and enforces max" {
    var in: Io.Reader = .fixed(&.{});
    var out: Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    var c = testConn(&in, &out.writer);
    defer c.deinit();
    const data: [1000]u8 = @splat(0xab);
    try c.send(.binary, &data);
    try testing.expectEqual(@as(u8, 126), out.written()[1]); // unmasked, 16-bit length

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    var r: Io.Reader = .fixed(out.written());
    const f = try readFrame(&r, testing.allocator, &buf, 4096);
    try testing.expect(f.fin);
    try testing.expectEqual(Opcode.binary, f.opcode);
    try testing.expectEqualSlices(u8, &data, f.data);

    var r2: Io.Reader = .fixed(out.written());
    try testing.expectError(error.TooLarge, readFrame(&r2, testing.allocator, &buf, 999));
}
