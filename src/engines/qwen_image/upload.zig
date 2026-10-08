//! Weights from a pack to the device at storage speed. Every loader used to read a tensor into pageable memory,
//! allocate it, and copy it synchronously: about a thousand round trips in a row, each a staged copy, with the disk
//! idle during copies and the copy engine idle during reads (19 s for 21 GB on GB10).
//!   Uploader  reads straight into a ring of pinned chunks (direct I/O: the drive's DMA, no page cache) and copies
//!             each with cuMemcpyHtoDAsync on the engine's
//!             stream, so the next read overlaps the last copy and kernels that use the weights (pack4) follow them
//!             in order; a chunk is reused once its event says the copy finished.
//!   Slab      sub-allocates device memory from a few large allocations (256-byte aligned, as cuMemAlloc).
//! Only the order and overlap of the traffic change: every byte lands where it did before.

const std = @import("std");
const cuda = @import("cuda");
const Pack = @import("pack.zig").Pack;
const Tensor = @import("pack.zig").Tensor;
const block = @import("pack.zig").block;

pub const chunk: usize = 32 << 20;
const ring = 4;

pub const Stats = struct {
    bytes: u64 = 0, // copied to the device
    read_ns: u64 = 0, // in pread
    wait_ns: u64 = 0, // waiting for a chunk to come free
    host_ns: u64 = 0, // host transforms into the chunks
};

pub const Uploader = struct {
    d: *const cuda.Driver,
    io: std.Io,
    s: cuda.Stream,
    bufs: [ring]cuda.HostBuffer,
    events: [ring]cuda.Event,
    pending: [ring]bool = @splat(false),
    next: usize = 0,
    stats: Stats = .{},

    pub fn init(d: *const cuda.Driver, io: std.Io, s: cuda.Stream) !Uploader {
        var u: Uploader = .{ .d = d, .io = io, .s = s, .bufs = undefined, .events = undefined };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            u.bufs[i].free();
            u.events[i].deinit();
        };
        for (0..ring) |i| {
            u.bufs[i] = try cuda.HostBuffer.alloc(d, chunk + 2 * block); // a direct read's block-aligned window
            errdefer u.bufs[i].free();
            u.events[i] = try cuda.Event.init(d, false);
            made += 1;
        }
        return u;
    }

    /// Waits for every copy, then frees the chunks.
    pub fn deinit(u: *Uploader) void {
        for (0..ring) |i| {
            if (u.pending[i]) u.events[i].synchronize() catch {};
            u.bufs[i].free();
            u.events[i].deinit();
        }
    }

    /// The next free chunk (waiting for its last copy if needed); page-aligned (cuMemHostAlloc).
    fn take(u: *Uploader) ![]align(block) u8 {
        const i = u.next;
        if (u.pending[i]) {
            const t0 = now(u.io);
            try u.events[i].synchronize();
            u.stats.wait_ns += since(u.io, t0);
            u.pending[i] = false;
        }
        return @alignCast(u.bufs[i].bytes);
    }

    /// Sends `n` bytes of the chunk `take` returned, from byte `from`, to `dst`, and moves to the next chunk.
    fn send(u: *Uploader, dst: u64, from: usize, n: usize) !void {
        const i = u.next;
        try u.d.check(u.d.api.cuMemcpyHtoDAsync_v2(dst, u.bufs[i].bytes.ptr + from, n, u.s.handle), "cuMemcpyHtoDAsync");
        try u.events[i].record(u.s);
        u.pending[i] = true;
        u.next = (i + 1) % ring;
        u.stats.bytes += n;
    }

    /// Tensor `t` of `p`, as stored, to `dst`.
    pub fn tensor(u: *Uploader, p: *const Pack, t: Tensor, dst: u64) !void {
        var at: usize = 0;
        while (at < t.len()) {
            const n = @min(chunk, t.len() - at);
            const buf = try u.take();
            const t0 = now(u.io);
            const from = try p.readAligned(u.io, t, at, n, buf);
            u.stats.read_ns += since(u.io, t0);
            try u.send(dst + at, from, n);
            at += n;
        }
    }

    /// Host bytes to `dst`.
    pub fn bytes(u: *Uploader, src: []const u8, dst: u64) !void {
        var at: usize = 0;
        while (at < src.len) {
            const n = @min(chunk, src.len - at);
            const buf = try u.take();
            @memcpy(buf[0..n], src[at..][0..n]);
            try u.send(dst + at, 0, n);
            at += n;
        }
    }

    /// `total` bytes made on the host straight into the chunks: `fill(ctx, offset, out)` writes `out` (the bytes at
    /// `offset`, a multiple of `step` unless it is the end; `step` divides `chunk` rounded down), then they go to `dst`.
    pub fn produce(u: *Uploader, dst: u64, total: usize, step: usize, ctx: anytype, comptime fill: fn (@TypeOf(ctx), usize, []u8) void) !void {
        const per = chunk / step * step;
        var at: usize = 0;
        while (at < total) {
            const n = @min(per, total - at);
            const buf = try u.take();
            const t0 = now(u.io);
            fill(ctx, at, buf[0..n]);
            u.stats.host_ns += since(u.io, t0);
            try u.send(dst + at, 0, n);
            at += n;
        }
    }
};

/// Device memory for a component's weights and buffers, sub-allocated from 1 GiB slabs (256-byte aligned, as
/// cuMemAlloc); a request over 256 MiB gets its own allocation, so a slab wastes at most that at its end. Freed at once.
pub const Slab = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    cur: u64 = 0, // the open slab's next free address
    left: usize = 0,
    bytes: u64 = 0, // handed out
    const slab: usize = 1 << 30;
    const own_from: usize = 256 << 20;

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator) Slab {
        return .{ .d = d, .gpa = gpa };
    }

    pub fn deinit(s: *Slab) void {
        for (s.bufs.items) |*b| b.free();
        s.bufs.deinit(s.gpa);
    }

    pub fn alloc(s: *Slab, n: usize) !u64 {
        const len = std.mem.alignForward(usize, @max(n, 256), 256);
        s.bytes += len;
        if (len > own_from) return s.own(len);
        if (len > s.left) {
            s.cur = try s.own(slab);
            s.left = slab;
        }
        const p = s.cur;
        s.cur += len;
        s.left -= len;
        return p;
    }

    fn own(s: *Slab, len: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(s.d, len);
        errdefer b.free();
        try s.bufs.append(s.gpa, b);
        return b.at(0);
    }
};

fn now(io: std.Io) std.Io.Timestamp {
    return std.Io.Clock.awake.now(io);
}

fn since(io: std.Io, t0: std.Io.Timestamp) u64 {
    return @intCast(t0.durationTo(now(io)).nanoseconds);
}
