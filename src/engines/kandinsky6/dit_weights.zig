//! The Kandinsky-6 Pro-distill DiT checkpoint (Hugging Face `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`
//! `transformer/diffusion_pytorch_model.safetensors`) on the device, as upstream kandinsky-6 eac06e39
//! `kandinsky/runtime/weights.py` `load_bf16_checkpoint` loads it: safetensors `load_file`, every F32 tensor cast to
//! BF16 (`value.to(torch.bfloat16)` on the CPU, round to nearest even), every other tensor as stored, then
//! `load_state_dict(strict=True, assign=True)`: no missing, no unexpected key, every shape the module's.
//!
//!   Header      the file's header, parsed and validated as the safetensors library validates it (header at most
//!               100,000,000 bytes, UTF-8 JSON object starting with '{', tensors contiguous from offset 0 and covering
//!               the data exactly, byte length = elements * dtype size) and stricter where the library is lax: duplicate
//!               names, unknown fields and non-integer offsets or dims are refused, never guessed.
//!   Verify      the strict key / dtype / shape check against the DiT's manifest (`layout.forEachTensor` calls
//!               `tensor(name, dtype, shape)`; the dtype is matched by tag name, so this file imports no DiT module).
//!   Device      one device allocation holding every tensor as BF16, 256-byte aligned each; `get(name) ?u64` is the
//!               `lookup` `bindWeights` takes. `load` returns `Load`: `ok`, or `failed` with the stopping error, the
//!               cleanup's quiescence result and, unless the stream was shown quiet, the `Custody` of every live
//!               allocation for the owner to release later. `Device.release` frees only on the users' streams' quiet
//!               answer. Every wait is bounded (`Quiesce`); no memory a copy may still use is freed on a guess. The DiT binds BF16 views only, so a tensor stored in any dtype but F32 or
//!               BF16 (F16 included, which upstream keeps as F16) is refused at load rather than bound with the wrong
//!               element type.
//!   Resident    the device bytes of a manifest, computed without I/O (for `needs`).
//!
//! The cast is torch's CPU conversion (c10/util/BFloat16.h round_to_nearest_even, and its vectorized forms): for every
//! non-NaN f32 the BF16 bits are (u + 0x7FFF + ((u >> 16) & 1)) >> 16, which is IEEE round to nearest even including
//! subnormals, infinities and overflow to infinity. torch's NaN result differs between its scalar (0x7FC0) and
//! vectorized (0xFFFF) paths, so a NaN in an F32 tensor is refused. Bulk bytes are read with direct I/O where the
//! filesystem allows it (no page cache, which on GB10 is GPU memory), else with pread and the cache dropped as it goes.

const std = @import("std");
const cuda = @import("cuda");
const Io = std.Io;
const posix = std.posix;
const linux = std.os.linux;

pub const Error = error{
    BadHeader,
    HeaderTooLarge,
    InvalidHeaderStart,
    DuplicateTensor,
    UnknownDtype,
    InvalidOffset,
    SizeMismatch,
    IncompleteBuffer,
    MissingWeight,
    UnexpectedWeight,
    DtypeMismatch,
    ShapeMismatch,
    UnsupportedDeviceDtype,
    NanInF32Weight,
    ShortRead,
    Cancelled,
    OutOfMemory,
};

/// safetensors' dtypes (safetensors 0.4, `Dtype`), by their header names.
pub const Dtype = enum {
    bool,
    u8,
    i8,
    f8_e5m2,
    f8_e4m3,
    i16,
    u16,
    f16,
    bf16,
    i32,
    u32,
    f32,
    f64,
    i64,
    u64,

    pub fn size(d: Dtype) u64 {
        return switch (d) {
            .bool, .u8, .i8, .f8_e5m2, .f8_e4m3 => 1,
            .i16, .u16, .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }

    pub fn parse(s: []const u8) ?Dtype {
        const names = .{
            .{ "BOOL", .bool }, .{ "U8", .u8 },   .{ "I8", .i8 },   .{ "F8_E5M2", .f8_e5m2 }, .{ "F8_E4M3", .f8_e4m3 },
            .{ "I16", .i16 },   .{ "U16", .u16 }, .{ "F16", .f16 }, .{ "BF16", .bf16 },       .{ "I32", .i32 },
            .{ "U32", .u32 },   .{ "F32", .f32 }, .{ "F64", .f64 }, .{ "I64", .i64 },         .{ "U64", .u64 },
        };
        inline for (names) |n| if (std.mem.eql(u8, s, n[0])) return n[1];
        return null;
    }
};

pub const Tensor = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    begin: u64, // within the data region
    end: u64,

    pub fn len(t: Tensor) u64 {
        return t.end - t.begin;
    }

    pub fn elements(t: Tensor) u64 {
        var n: u64 = 1;
        for (t.shape) |s| n *= s; // checked when the header was parsed
        return n;
    }
};

/// safetensors' MAX_HEADER_SIZE.
pub const max_header: u64 = 100_000_000;

pub const Header = struct {
    arena: std.heap.ArenaAllocator,
    tensors: []const Tensor = &.{}, // in data order (ascending offsets)
    index: std.StringHashMapUnmanaged(u32) = .empty,
    data_len: u64,

    /// `json`: the header bytes (8 .. 8 + n of the file); `data_len`: the file's bytes after them.
    pub fn parse(gpa: std.mem.Allocator, json: []const u8, data_len: u64) Error!Header {
        var h: Header = .{ .arena = .init(gpa), .data_len = data_len };
        errdefer h.arena.deinit();
        const a = h.arena.allocator();
        if (json.len > max_header) return error.HeaderTooLarge;
        if (!std.unicode.utf8ValidateSlice(json)) return error.BadHeader;
        if (json.len == 0 or json[0] != '{') return error.InvalidHeaderStart;
        const v = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{ .duplicate_field_behavior = .@"error" }) catch |e| {
            const any: anyerror = e;
            if (any == error.OutOfMemory) return error.OutOfMemory;
            if (any == error.DuplicateField) return error.DuplicateTensor;
            return error.BadHeader;
        };
        if (v != .object) return error.BadHeader;
        var list: std.ArrayList(Tensor) = .empty;
        var it = v.object.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            if (std.mem.eql(u8, key, "__metadata__")) {
                if (kv.value_ptr.* != .object) return error.BadHeader;
                var mi = kv.value_ptr.object.iterator();
                while (mi.next()) |m| if (m.value_ptr.* != .string) return error.BadHeader;
                continue;
            }
            if (kv.value_ptr.* != .object) return error.BadHeader;
            const o = kv.value_ptr.object;
            if (o.count() != 3) return error.BadHeader; // exactly dtype, shape, data_offsets
            const dv = o.get("dtype") orelse return error.BadHeader;
            const sv = o.get("shape") orelse return error.BadHeader;
            const ov = o.get("data_offsets") orelse return error.BadHeader;
            if (dv != .string or sv != .array or ov != .array or ov.array.items.len != 2) return error.BadHeader;
            const dt = Dtype.parse(dv.string) orelse return error.UnknownDtype;
            const shape = try a.alloc(u64, sv.array.items.len);
            var n: u64 = 1;
            for (sv.array.items, shape) |s, *d| {
                d.* = try natural(s);
                n = std.math.mul(u64, n, d.*) catch return error.SizeMismatch;
            }
            const bytes = std.math.mul(u64, n, dt.size()) catch return error.SizeMismatch;
            const t: Tensor = .{ .name = key, .dtype = dt, .shape = shape, .begin = try natural(ov.array.items[0]), .end = try natural(ov.array.items[1]) };
            if (t.end < t.begin) return error.InvalidOffset;
            if (t.len() != bytes) return error.SizeMismatch;
            try list.append(a, t);
        }
        const tensors = try list.toOwnedSlice(a);
        std.mem.sort(Tensor, tensors, {}, struct {
            fn lt(_: void, x: Tensor, y: Tensor) bool {
                return x.begin < y.begin or (x.begin == y.begin and x.end < y.end);
            }
        }.lt);
        h.tensors = tensors;
        var at: u64 = 0;
        for (h.tensors, 0..) |t, i| {
            if (t.begin != at) return error.InvalidOffset; // a gap or an overlap
            at = t.end;
            const gop = try h.index.getOrPut(a, t.name);
            if (gop.found_existing) return error.DuplicateTensor;
            gop.value_ptr.* = @intCast(i);
        }
        if (at != data_len) return error.IncompleteBuffer;
        return h;
    }

    pub fn deinit(h: *Header) void {
        h.arena.deinit();
    }

    pub fn get(h: *const Header, name: []const u8) ?Tensor {
        const i = h.index.get(name) orelse return null;
        return h.tensors[i];
    }

    /// Host bytes the parsed header holds (the arena's capacity, read at run time).
    pub fn hostBytes(h: *const Header) u64 {
        return h.arena.queryCapacity();
    }

    fn natural(v: std.json.Value) Error!u64 {
        if (v != .integer or v.integer < 0) return error.BadHeader;
        return @intCast(v.integer);
    }
};

/// The strict check (load_state_dict strict=True and its shape check): a visitor for `layout.forEachTensor`, then
/// `finish`. `dtype` is the manifest's file dtype, matched by tag name (`bf16`, `f32`, ...).
pub const Verify = struct {
    h: *const Header,
    seen: []bool,
    count: usize = 0,

    pub fn init(gpa: std.mem.Allocator, h: *const Header) Error!Verify {
        const seen = try gpa.alloc(bool, h.tensors.len);
        @memset(seen, false);
        return .{ .h = h, .seen = seen };
    }

    pub fn deinit(v: *Verify, gpa: std.mem.Allocator) void {
        gpa.free(v.seen);
    }

    pub fn tensor(v: *Verify, name: []const u8, dtype: anytype, shape: []const u64) Error!void {
        const i = v.h.index.get(name) orelse {
            std.log.warn("kandinsky6 weights: missing {s}", .{name});
            return error.MissingWeight;
        };
        const t = v.h.tensors[i];
        const want = std.meta.stringToEnum(Dtype, @tagName(dtype)) orelse return error.UnknownDtype;
        if (t.dtype != want) {
            std.log.warn("kandinsky6 weights: {s} is {s}, the manifest says {s}", .{ name, @tagName(t.dtype), @tagName(want) });
            return error.DtypeMismatch;
        }
        if (!std.mem.eql(u64, t.shape, shape)) {
            std.log.warn("kandinsky6 weights: {s} has shape {any}, the manifest says {any}", .{ name, t.shape, shape });
            return error.ShapeMismatch;
        }
        if (v.seen[i]) return error.DuplicateTensor;
        v.seen[i] = true;
        v.count += 1;
    }

    pub fn finish(v: *const Verify) Error!void {
        if (v.count == v.h.tensors.len) return;
        for (v.seen, v.h.tensors) |s, t| if (!s) std.log.warn("kandinsky6 weights: unexpected {s}", .{t.name});
        return error.UnexpectedWeight;
    }
};

/// The BF16 bits torch's CPU cast gives a non-NaN f32 (round to nearest even); null for NaN.
pub fn bf16Rne(bits: u32) ?u16 {
    if ((bits & 0x7FFF_FFFF) > 0x7F80_0000) return null;
    const lsb = (bits >> 16) & 1;
    return @intCast((bits + 0x7FFF + lsb) >> 16);
}

/// The device size of one tensor: its BF16 bytes rounded up to 256 (every tensor starts 256-byte aligned).
pub fn deviceBytes(elements: u64) u64 {
    return std.mem.alignForward(u64, elements * 2, 256);
}

/// Device bytes of a manifest, without I/O: a `layout.forEachTensor` visitor. Any dtype the device refuses counts
/// in `refused`.
pub const Resident = struct {
    bytes: u64 = 0,
    tensors: u64 = 0,
    refused: u64 = 0,

    pub fn tensor(r: *Resident, name: []const u8, dtype: anytype, shape: []const u64) error{}!void {
        _ = name;
        var n: u64 = 1;
        for (shape) |s| n *= s;
        const tag = @tagName(dtype);
        if (!std.mem.eql(u8, tag, "bf16") and !std.mem.eql(u8, tag, "f32")) r.refused += 1;
        r.bytes += deviceBytes(n);
        r.tensors += 1;
    }
};

/// Direct I/O's alignment of offsets, lengths and buffers.
pub const block: usize = 4096;
/// One staging chunk (a multiple of 4, so an F32 chunk holds whole elements).
pub const chunk: usize = 32 << 20;
const ring = 4;

/// Pinned host memory a load stages through (freed when the load returns).
pub const staging_bytes: u64 = ring * (chunk + 2 * block);

pub const Checkpoint = struct {
    header: Header,
    file: Io.File,
    direct: posix.fd_t = -1, // the file opened O_DIRECT, or -1 where the filesystem refuses it
    data_start: u64,

    pub fn open(gpa: std.mem.Allocator, io: Io, path: []const u8) !Checkpoint {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        var len_buf: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &len_buf, 0) != 8) return error.ShortRead;
        const n = std.mem.readInt(u64, &len_buf, .little);
        if (n > max_header) return error.HeaderTooLarge;
        const size: u64 = (try file.stat(io)).size;
        if (8 + n > size) return error.BadHeader;
        const json = try gpa.alloc(u8, @intCast(n));
        defer gpa.free(json);
        if (try file.readPositionalAll(io, json, 8) != json.len) return error.ShortRead;
        var header = try Header.parse(gpa, json, size - 8 - n);
        errdefer header.deinit();
        const pz = try gpa.dupeSentinel(u8, path, 0);
        defer gpa.free(pz);
        const direct = posix.openatZ(posix.AT.FDCWD, pz, .{ .ACCMODE = .RDONLY, .DIRECT = true, .CLOEXEC = true }, 0) catch -1;
        return .{ .header = header, .file = file, .direct = direct, .data_start = 8 + n };
    }

    pub fn close(ck: *Checkpoint, io: Io) void {
        if (ck.direct >= 0) _ = linux.close(ck.direct);
        ck.file.close(io);
        ck.header.deinit();
    }

    /// Bytes [off, off + n) of the data region into `buf` (block-aligned, at least n + 2 * block bytes); the result is
    /// where they start in `buf`. Direct I/O reads the covering block-aligned range; without it, pread at 0 and the page
    /// cache let go.
    fn read(ck: *const Checkpoint, io: Io, off: u64, n: usize, buf: []align(block) u8) !usize {
        std.debug.assert(buf.len >= n + 2 * block);
        const abs = ck.data_start + off;
        if (ck.direct < 0) {
            if (try ck.file.readPositionalAll(io, buf[0..n], abs) != n) return error.ShortRead;
            _ = linux.fadvise(ck.file.handle, @intCast(abs), @intCast(n), linux.POSIX_FADV.DONTNEED);
            return 0;
        }
        const start = abs / block * block;
        const lead: usize = @intCast(abs - start);
        const len = std.mem.alignForward(usize, lead + n, block);
        var got: usize = 0;
        while (got < lead + n) { // the file's last block may be short
            const rc = linux.pread(ck.direct, buf.ptr + got, len - got, @intCast(start + got));
            switch (linux.errno(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                else => |e| return posix.unexpectedErrno(e),
            }
            if (rc == 0) return error.ShortRead;
            got += rc;
        }
        return lead;
    }
};

/// Where each tensor of a header goes on the device (`offsets[i]` for `header.tensors[i]`) and the allocation's size.
pub const Plan = struct {
    offsets: []u64,
    resident: u64,
    cast: u64, // F32 tensors cast to BF16
    stored: u64, // BF16 tensors as stored

    pub fn init(gpa: std.mem.Allocator, h: *const Header) Error!Plan {
        const offsets = try gpa.alloc(u64, h.tensors.len);
        errdefer gpa.free(offsets);
        var p: Plan = .{ .offsets = offsets, .resident = 0, .cast = 0, .stored = 0 };
        for (h.tensors, offsets) |t, *o| {
            switch (t.dtype) {
                .f32 => p.cast += 1,
                .bf16 => p.stored += 1,
                else => {
                    std.log.warn("kandinsky6 weights: {s} is {s}; the DiT binds BF16 only", .{ t.name, @tagName(t.dtype) });
                    return error.UnsupportedDeviceDtype;
                },
            }
            o.* = p.resident;
            p.resident += deviceBytes(t.elements());
        }
        return p;
    }

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        gpa.free(p.offsets);
    }
};

const abi = cuda.abi;

/// A raw CUDA driver failure: the CUresult as returned and the entry point that returned it (TensorFold's wrappers
/// reduce a result to a generic error, so every call that decides custody here is made raw through `Driver.api`).
pub const RawFailure = struct {
    rc: abi.Result,
    call: []const u8,
};

/// A bound on waiting for the device to finish work that reads or writes this file's memory: the raw query is
/// polled every `poll_ms` for at most `budget_ms`; `cancel` ends the wait early.
pub const Quiesce = struct {
    budget_ms: u64 = 10_000,
    poll_ms: u64 = 1,
    cancel: ?*const std.atomic.Value(bool) = null,
};

/// What a quiescence check showed. Only `quiet` is evidence that no work uses the memory.
pub const QuiesceResult = union(enum) {
    quiet, // the query returned CUDA_SUCCESS after the last enqueue
    timeout, // the budget ran out while it returned CUDA_ERROR_NOT_READY
    cancelled, // `Quiesce.cancel` (or the Io's cancellation) ended the wait
    failed: RawFailure, // the query returned another CUresult (e.g. a sticky context error): device state unknown
};

/// cuStreamQuery on a stream: the raw probe `quiesce` polls.
pub const StreamQuery = struct {
    d: *const cuda.Driver,
    h: abi.Stream,
    pub const call = "cuStreamQuery";

    pub fn of(s: cuda.Stream) StreamQuery {
        return .{ .d = s.d, .h = s.handle };
    }

    pub fn query(p: StreamQuery) abi.Result {
        return p.d.api.cuStreamQuery(p.h);
    }
};

/// cuEventQuery on an event.
pub const EventQuery = struct {
    d: *const cuda.Driver,
    h: abi.Event,
    pub const call = "cuEventQuery";

    pub fn query(p: EventQuery) abi.Result {
        return p.d.api.cuEventQuery(p.h);
    }
};

/// Polls `probe.query()` (raw CUresult: success = done, not-ready = busy, anything else = failed, reported with
/// `@TypeOf(probe).call`) until done, the budget is spent, `q.cancel` is set or the query fails. Never answers
/// `quiet` without CUDA_SUCCESS from the query.
pub fn quiesce(io: Io, probe: anytype, q: Quiesce) QuiesceResult {
    const t0 = Io.Clock.awake.now(io);
    while (true) {
        const rc = probe.query();
        if (rc == abi.success) return .quiet;
        if (rc != abi.error_not_ready) return .{ .failed = .{ .rc = rc, .call = @TypeOf(probe).call } };
        if (q.cancel) |c| if (c.load(.acquire)) return .cancelled;
        const spent_ms: u64 = @intCast(@divFloor(t0.durationTo(Io.Clock.awake.now(io)).nanoseconds, std.time.ns_per_ms));
        if (spent_ms >= q.budget_ms) return .timeout;
        const wait_ms = @max(@min(q.poll_ms, q.budget_ms - spent_ms), 1);
        Io.sleep(io, .fromMilliseconds(@intCast(wait_ms)), .awake) catch return .cancelled;
    }
}

/// The evidence that permits freeing: nothing was ever enqueued (`idle`), or the stream was quiet after the last
/// enqueue (`quiet`).
pub const Evidence = enum { idle, quiet };

/// Evidence from a load's history and its stream check; null: none (nothing may be freed).
fn evidenceFor(enqueued: bool, r: ?QuiesceResult) ?Evidence {
    if (!enqueued) return .idle;
    const got = r orelse return null;
    return if (got == .quiet) .quiet else null;
}

/// The result of a release.
pub const ReleaseResult = union(enum) {
    released: Evidence, // every allocation this owner held is freed, each by exactly one successful call
    busy: QuiesceResult, // no evidence: nothing was freed and everything stays owned
    partial: RawFailure, // evidence held, but a free / destroy failed: the first raw failure, sticky. Every allocation
    // gets at most one free / destroy call ever: a failed one stays owned (`outstanding().failed`) and is never
    // reissued, since what the driver did with it is unknown; a later release answers the same failure and calls
    // nothing. Only the context's destruction (the worker process's exit) reclaims it.
    no_evidence, // `Device.release` with no user stream to query: nothing freed
};

/// A staging chunk's custody: `free` (no copy reads it), `recorded` (its copy's event was recorded: the event says
/// when it is free), `unrecorded` (an enqueue was attempted and its event is not recorded: only the stream can say).
const Slot = enum { free, recorded, unrecorded };

/// The chunks' bookkeeping, apart from the driver so the custody rules are testable on the host.
const Slots = struct {
    state: [ring]Slot = @splat(.free),
    enqueued: bool = false, // any copy attempted: freeing then needs the stream's evidence

    /// Before the enqueue call: from here a failure of the call or of the event record leaves the chunk in use.
    fn begin(s: *Slots, i: usize) void {
        s.state[i] = .unrecorded;
        s.enqueued = true;
    }

    fn recorded(s: *Slots, i: usize) void {
        s.state[i] = .recorded;
    }
};

const chunk_alloc: usize = chunk + 2 * block; // a direct read's block-aligned window

/// The pinned chunks and events a load copies through. Each live allocation is a non-null entry until the raw call
/// that frees it succeeds, so a failed free stays owned and listed.
const Staging = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    host: [ring]?[*]align(16) u8 = @splat(null),
    events: [ring]abi.Event = @splat(null),
    host_failed: [ring]?RawFailure = @splat(null), // its one cuMemFreeHost failed: owned, never called again
    event_failed: [ring]?RawFailure = @splat(null), // its one cuEventDestroy_v2 failed: likewise
    slots: Slots = .{},
    next: usize = 0,
    last_raw: ?RawFailure = null, // the raw failure that stopped the load, if a raw call stopped it

    /// Allocates every chunk and event. A failure leaves whatever was allocated owned here (released by the owner's
    /// `release`, which reports its raw results).
    fn init(st: *Staging) !void {
        for (0..ring) |i| {
            const hb = try cuda.HostBuffer.alloc(st.d, chunk_alloc);
            st.host[i] = hb.bytes.ptr;
            const ev = try cuda.Event.init(st.d, false);
            st.events[i] = ev.handle;
        }
    }

    /// The next chunk, waiting (bounded) for its last copy's event.
    fn take(st: *Staging, io: Io, q: Quiesce) ![]align(block) u8 {
        const i = st.next;
        switch (st.slots.state[i]) {
            .free => {},
            .recorded => switch (quiesce(io, EventQuery{ .d = st.d, .h = st.events[i] }, q)) {
                .quiet => st.slots.state[i] = .free,
                .timeout => return error.CopyTimeout,
                .cancelled => return error.Cancelled,
                .failed => |f| {
                    st.last_raw = f;
                    return error.EventQueryFailed;
                },
            },
            .unrecorded => return error.UnverifiedChunk, // send returned the failure; the load stops before this
        }
        const p = st.host[i] orelse return error.UnverifiedChunk;
        return @alignCast(p[0..chunk_alloc]);
    }

    fn send(st: *Staging, dst: u64, from: usize, n: usize) !void {
        const i = st.next;
        const p = st.host[i] orelse return error.UnverifiedChunk;
        st.slots.begin(i);
        const rc = st.d.api.cuMemcpyHtoDAsync_v2(dst, p + from, n, st.s.handle);
        if (rc != abi.success) {
            st.last_raw = .{ .rc = rc, .call = "cuMemcpyHtoDAsync" };
            return error.CopyEnqueueFailed;
        }
        const rr = st.d.api.cuEventRecord(st.events[i], st.s.handle);
        if (rr != abi.success) { // the copy stays enqueued with no event: the slot stays unrecorded
            st.last_raw = .{ .rc = rr, .call = "cuEventRecord" };
            return error.EventRecordFailed;
        }
        st.slots.recorded(i);
        st.next = (i + 1) % ring;
    }

    /// One checked raw call for every live event and chunk not already failed: success clears the entry, failure marks
    /// it failed (owned, never called again); the first failure goes to `first`. Only with evidence.
    fn releaseAll(st: *Staging, first: *?RawFailure) void {
        for (0..ring) |i| {
            if (st.events[i]) |e| {
                if (st.event_failed[i] == null) {
                    const rc = st.d.api.cuEventDestroy_v2(e);
                    if (rc == abi.success) st.events[i] = null else st.event_failed[i] = note(first, rc, "cuEventDestroy_v2");
                }
            }
            if (st.host[i]) |p| {
                if (st.host_failed[i] == null) {
                    const rc = st.d.api.cuMemFreeHost(p);
                    if (rc == abi.success) st.host[i] = null else st.host_failed[i] = note(first, rc, "cuMemFreeHost");
                }
            }
        }
    }

    fn failedCount(st: *const Staging) u32 {
        var n: u32 = 0;
        for (st.host_failed) |f| n += @intFromBool(f != null);
        for (st.event_failed) |f| n += @intFromBool(f != null);
        return n;
    }

    fn liveChunks(st: *const Staging) u32 {
        var n: u32 = 0;
        for (st.host) |h| n += @intFromBool(h != null);
        return n;
    }

    fn liveEvents(st: *const Staging) u32 {
        var n: u32 = 0;
        for (st.events) |e| n += @intFromBool(e != null);
        return n;
    }
};

/// The failure of one raw call, recorded in `first` if it is the first.
fn note(first: *?RawFailure, rc: abi.Result, call: []const u8) RawFailure {
    const f: RawFailure = .{ .rc = rc, .call = call };
    if (first.* == null) first.* = f;
    return f;
}

/// What an owner still holds.
pub const Outstanding = struct {
    device_bytes: u64, // the weights' allocation (0: freed)
    pinned_chunks: u32, // of `ring`, chunk + 2 * block bytes each
    events: u32,
    plan: bool, // host bookkeeping, kept until the device allocation it describes is freed
    failed: u32, // of the above, allocations whose one free / destroy call failed (never reissued)

    pub fn none(o: Outstanding) bool {
        return o.device_bytes == 0 and o.pinned_chunks == 0 and o.events == 0 and !o.plan;
    }
};

/// Every live allocation of a load, the stream whose quiet answer permits freeing them, and the evidence once held.
/// After a failed load the owner must keep it. A `busy` release may be repeated: polling for evidence frees nothing.
/// Once evidence is held, each allocation gets exactly one free / destroy call. If any fails, the custody is sticky:
/// `release` answers the first failure again with no call, and the failed allocations stay owned until the CUDA
/// context is destroyed (the worker process's exit), after which `forgetAfterContextDestroyed` drops the host
/// bookkeeping. This API offers no retry of a failed free and no proof that one would be safe.
pub const Custody = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    staging: Staging,
    dev_ptr: abi.DevicePtr, // 0: none or freed
    dev_len: u64,
    dev_failed: ?RawFailure = null, // its one cuMemFree_v2 failed: owned, never called again
    plan: ?Plan,
    proven: ?Evidence = null,
    first_failure: ?RawFailure = null, // the first raw free / destroy failure seen by any release

    pub fn outstanding(c: *const Custody) Outstanding {
        return .{
            .device_bytes = if (c.dev_ptr != 0) c.dev_len else 0,
            .pinned_chunks = c.staging.liveChunks(),
            .events = c.staging.liveEvents(),
            .plan = c.plan != null,
            .failed = c.staging.failedCount() + @intFromBool(c.dev_failed != null),
        };
    }

    /// After the owner destroyed the CUDA context (or as the process exits): drops the host bookkeeping (the plan) and
    /// forgets the device handles, which the context's destruction reclaimed. Makes no CUDA call. The caller asserts
    /// the context is gone; nothing here can check it.
    pub fn forgetAfterContextDestroyed(c: *Custody) void {
        if (c.plan) |*p| p.deinit(c.gpa);
        c.plan = null;
        c.dev_ptr = 0;
        c.staging = .{ .d = c.d, .s = c.s };
    }

    /// Takes the evidence (once: idle if nothing was enqueued, else the stream's quiet answer within `q`), then
    /// frees what is still owned.
    pub fn release(c: *Custody, io: Io, q: Quiesce) ReleaseResult {
        if (c.first_failure) |f| return .{ .partial = f }; // sticky: no call for a failed release
        if (c.proven == null) {
            const r: ?QuiesceResult = if (c.staging.slots.enqueued) quiesce(io, StreamQuery.of(c.s), q) else null;
            c.proven = evidenceFor(c.staging.slots.enqueued, r) orelse return .{ .busy = r.? };
        }
        return c.freeOwned();
    }

    /// Requires `proven`. One raw call for each allocation still live and not failed: success clears it, failure
    /// marks it failed. Any failure, now or before, answers the first one (sticky).
    fn freeOwned(c: *Custody) ReleaseResult {
        const ev = c.proven.?;
        var first: ?RawFailure = null;
        c.staging.releaseAll(&first);
        if (c.dev_ptr != 0 and c.dev_failed == null) {
            const rc = c.d.api.cuMemFree_v2(c.dev_ptr);
            if (rc == abi.success) c.dev_ptr = 0 else c.dev_failed = note(&first, rc, "cuMemFree_v2");
        }
        if (c.dev_ptr == 0) {
            if (c.plan) |*p| {
                p.deinit(c.gpa);
                c.plan = null;
            }
        }
        if (first) |f| {
            if (c.first_failure == null) c.first_failure = f;
        }
        if (c.first_failure) |f| return .{ .partial = f };
        return .{ .released = ev };
    }
};

/// A load that did not produce a `Device`: the error that stopped it (with the raw CUresult when a raw call stopped
/// it), the cleanup's result, and, unless that was `released`, the custody of everything still owned.
pub const Failure = struct {
    primary: anyerror,
    primary_raw: ?RawFailure,
    cleanup: ReleaseResult,
    custody: ?Custody,
};

pub const Loaded = struct {
    device: Device,
    staging_release: ?RawFailure, // the first failed free of a staging chunk / event after the quiet stream: the
    // failed allocation stays owned by `device` (`outstanding().failed`) and is never reissued; the weights' own
    // allocation still gets its one separate free in `Device.release`, which then answers `partial` with this failure
};

pub const Load = union(enum) {
    ok: Loaded,
    failed: Failure,
};

/// Every tensor as BF16 on the device. `live` until a release starts freeing; then lookups are refused.
pub const Device = struct {
    own: Custody,
    header: *const Header,
    state: State = .live,

    pub const State = enum { live, release_failed, released, forgotten };

    /// The `lookup` of `bindWeights`: the device address of tensor `name`, or null (also for a device whose release
    /// has started: its memory may be gone and its plan freed).
    pub fn get(w: *const Device, name: []const u8) ?u64 {
        if (w.state != .live or w.own.dev_ptr == 0) return null;
        const p = w.own.plan orelse return null;
        const i = w.header.index.get(name) orelse return null;
        return w.own.dev_ptr + p.offsets[i];
    }

    pub fn residentBytes(w: *const Device) u64 {
        return if (w.own.dev_ptr != 0) w.own.dev_len else 0;
    }

    pub fn outstanding(w: *const Device) Outstanding {
        return w.own.outstanding();
    }

    /// Reads and uploads every tensor of `ck` in file order on `s` (F32 cast on the host into the staging chunk).
    /// `cancel` is read between 32 MiB chunks. `q` bounds every wait: a chunk's reuse, the final quiescence and the
    /// cleanup of a failed load. `ok` only once `s` answered CUDA_SUCCESS after the last copy (the staging is then
    /// freed; a raw failure there is reported and stays owned by the device). `ck` must outlive the result.
    pub fn load(gpa: std.mem.Allocator, io: Io, d: *const cuda.Driver, s: cuda.Stream, ck: *const Checkpoint, cancel: ?*const std.atomic.Value(bool), q: Quiesce) Load {
        const nothing: ReleaseResult = .{ .released = .idle };
        var plan = Plan.init(gpa, &ck.header) catch |e| return .{ .failed = .{ .primary = e, .primary_raw = null, .cleanup = nothing, .custody = null } };
        var ptr: abi.DevicePtr = 0;
        if (plan.resident > 0) {
            const rc = d.api.cuMemAlloc_v2(&ptr, plan.resident);
            if (rc != abi.success) {
                plan.deinit(gpa);
                return .{ .failed = .{ .primary = error.OutOfDeviceMemory, .primary_raw = .{ .rc = rc, .call = "cuMemAlloc_v2" }, .cleanup = nothing, .custody = null } };
            }
        }
        var c: Custody = .{ .gpa = gpa, .d = d, .s = s, .staging = .{ .d = d, .s = s }, .dev_ptr = ptr, .dev_len = plan.resident, .plan = plan };
        c.staging.init() catch |e| return fail(io, &c, e, q); // nothing enqueued: idle evidence
        upload(io, ck, &c, cancel, q) catch |e| return fail(io, &c, e, q);
        const r = quiesce(io, StreamQuery.of(s), q);
        if (r != .quiet) return .{ .failed = .{ .primary = error.NotQuiescent, .primary_raw = null, .cleanup = .{ .busy = r }, .custody = c } };
        c.proven = .quiet;
        var first: ?RawFailure = null;
        c.staging.releaseAll(&first); // quiet: no copy reads a chunk; the weights stay
        if (first) |f| c.first_failure = f;
        return .{ .ok = .{ .device = .{ .own = c, .header = &ck.header }, .staging_release = first } };
    }

    fn fail(io: Io, c: *Custody, primary: anyerror, q: Quiesce) Load {
        const raw_primary = c.staging.last_raw;
        const r = c.release(io, q);
        return .{ .failed = .{ .primary = primary, .primary_raw = raw_primary, .cleanup = r, .custody = if (r == .released) null else c.* } };
    }

    fn upload(io: Io, ck: *const Checkpoint, c: *Custody, cancel: ?*const std.atomic.Value(bool), q: Quiesce) !void {
        const st = &c.staging;
        for (ck.header.tensors, c.plan.?.offsets) |t, dev| {
            var at: u64 = 0;
            while (at < t.len()) {
                if (cancel) |x| if (x.load(.acquire)) return error.Cancelled;
                const n: usize = @intCast(@min(chunk, t.len() - at));
                const b = try st.take(io, q);
                const from = try ck.read(io, t.begin + at, n, b);
                switch (t.dtype) {
                    .bf16 => try st.send(c.dev_ptr + dev + at, from, n),
                    .f32 => {
                        // in place, forward: element j's bf16 lands at 2j, below every f32 not yet read (at 4j' >= 4j)
                        const span = b[from..][0..n];
                        for (0..n / 4) |j| {
                            const bits = std.mem.readInt(u32, span[4 * j ..][0..4], .little);
                            const h = bf16Rne(bits) orelse {
                                std.log.warn("kandinsky6 weights: NaN in F32 tensor {s} (element {d})", .{ t.name, at / 4 + j });
                                return error.NanInF32Weight;
                            };
                            std.mem.writeInt(u16, span[2 * j ..][0..2], h, .little);
                        }
                        try st.send(c.dev_ptr + dev + at / 2, from, n / 2);
                    },
                    else => unreachable, // Plan.init refused it
                }
                at += n;
            }
        }
    }

    /// Frees the weights (and any staging a load left) only on evidence that no work still reads them: every stream
    /// in `users` (each stream the owner ran work on that read these weights, after its last such enqueue) returns
    /// CUDA_SUCCESS to cuStreamQuery within `q`. Lookups are refused from the first free on. `busy` / `no_evidence`:
    /// nothing freed, still `live`, may be repeated. Otherwise every allocation still owned gets its one free call:
    /// `released` is final (repeating it answers `released` with no call); `partial` leaves the device
    /// `release_failed` for good. Repeating it answers the same first failure with no call, and `outstanding()`
    /// stays as it was. A staging free that failed after the load (`Loaded.staging_release`) makes the eventual
    /// release `partial` the same way.
    pub fn release(w: *Device, io: Io, users: []const cuda.Stream, q: Quiesce) ReleaseResult {
        switch (w.state) {
            .released => return .{ .released = .quiet },
            .release_failed => return .{ .partial = w.own.first_failure.? }, // sticky: no call
            .forgotten => return .no_evidence, // the context is gone: nothing to free, nothing to claim
            .live => {
                if (users.len == 0) return .no_evidence;
                for (users) |u| {
                    const r = quiesce(io, StreamQuery.of(u), q);
                    if (r != .quiet) return .{ .busy = r };
                }
                w.own.proven = .quiet;
                w.state = .release_failed; // from here lookups are refused, until every free succeeded
            },
        }
        const r = w.own.freeOwned();
        if (r == .released) w.state = .released;
        return r;
    }

    /// `Custody.forgetAfterContextDestroyed` for the device (the caller asserts the context is gone). Lookups are
    /// refused from then on; `release` answers `no_evidence` with no call.
    pub fn forgetAfterContextDestroyed(w: *Device) void {
        w.own.forgetAfterContextDestroyed();
        w.state = .forgotten;
    }
};

const testing = std.testing;

fn parseText(json: []const u8, data_len: u64) Error!Header {
    return Header.parse(testing.allocator, json, data_len);
}

test "header: contiguous tensors parse, sorted by offset, metadata kept out" {
    var h = try parseText(
        \\{"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"__metadata__":{"format":"pt"}}
    , 12);
    defer h.deinit();
    try testing.expectEqual(@as(usize, 2), h.tensors.len);
    try testing.expectEqualStrings("a", h.tensors[0].name);
    try testing.expectEqual(Dtype.f32, h.get("b").?.dtype);
    try testing.expectEqual(@as(u64, 2), h.get("b").?.elements());
    try testing.expect(h.get("__metadata__") == null);
}

test "header: the library's refusals and the stricter ones" {
    const cases = .{
        .{ error.InvalidHeaderStart, " {}", 0 },
        .{ error.BadHeader, "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}} x", 4 },
        .{ error.InvalidOffset, "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}", 2 }, // gap at 0
        .{ error.InvalidOffset, "{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]},\"b\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[1,3]}}", 3 },
        .{ error.IncompleteBuffer, "{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}", 3 },
        .{ error.SizeMismatch, "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4 },
        .{ error.UnknownDtype, "{\"a\":{\"dtype\":\"F4\",\"shape\":[2],\"data_offsets\":[0,1]}}", 1 },
        .{ error.BadHeader, "{\"a\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,0]}}", 0 },
        .{ error.BadHeader, "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1],\"x\":1}}", 1 },
        .{ error.BadHeader, "{\"a\":{\"dtype\":\"U8\",\"shape\":[1.0],\"data_offsets\":[0,1]}}", 1 },
        .{ error.BadHeader, "{\"__metadata__\":{\"k\":1}}", 0 },
        .{ error.DuplicateTensor, "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}", 2 },
    };
    inline for (cases) |c| try testing.expectError(c[0], parseText(c[1], c[2]));
}

test "bf16 cast: round to nearest even, ties to even, infinities, overflow, subnormals; NaN refused" {
    try testing.expectEqual(@as(?u16, 0x3F80), bf16Rne(0x3F80_0000)); // 1.0
    try testing.expectEqual(@as(?u16, 0x3F80), bf16Rne(0x3F80_8000)); // tie, even below
    try testing.expectEqual(@as(?u16, 0x3F82), bf16Rne(0x3F81_8000)); // tie, even above
    try testing.expectEqual(@as(?u16, 0x3F81), bf16Rne(0x3F80_8001)); // above the tie
    try testing.expectEqual(@as(?u16, 0x8000), bf16Rne(0x8000_0000)); // -0
    try testing.expectEqual(@as(?u16, 0x7F80), bf16Rne(0x7F80_0000)); // +inf
    try testing.expectEqual(@as(?u16, 0xFF80), bf16Rne(0xFF80_0000)); // -inf
    try testing.expectEqual(@as(?u16, 0x7F80), bf16Rne(0x7F7F_FFFF)); // f32 max rounds to inf
    try testing.expectEqual(@as(?u16, 0x0001), bf16Rne(0x0000_8001)); // subnormal rounds up
    try testing.expectEqual(@as(?u16, 0x0000), bf16Rne(0x0000_8000)); // subnormal tie to even 0
    try testing.expectEqual(@as(?u16, null), bf16Rne(0x7FC0_0000));
    try testing.expectEqual(@as(?u16, null), bf16Rne(0xFFFF_FFFF));
    // every non-NaN agrees with an IEEE round to nearest even through f64 arithmetic on a sample
    var x: u32 = 0x0000_0001;
    while (x < 0x7F7F_0000) : (x +%= 0x0001_3579) { // below the bf16 overflow threshold (covered above)
        const f: f32 = @bitCast(x);
        const b: u32 = @as(u32, bf16Rne(x).?) << 16;
        const lo: u32 = x & 0xFFFF_0000;
        const hi: u32 = lo + 0x1_0000;
        const dlo = @as(f64, f) - @as(f64, @as(f32, @bitCast(lo)));
        const dhi = @as(f64, @as(f32, @bitCast(hi))) - @as(f64, f);
        const want = if (dlo < dhi) lo else if (dhi < dlo) hi else if ((lo >> 16) & 1 == 0) lo else hi;
        try testing.expectEqual(want, b);
    }
}

test "verify: strict keys, dtypes and shapes against a manifest" {
    var h = try parseText(
        \\{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"BF16","shape":[3],"data_offsets":[24,30]}}
    , 30);
    defer h.deinit();
    const Dt = enum { bf16, f32 };
    {
        var v = try Verify.init(testing.allocator, &h);
        defer v.deinit(testing.allocator);
        try v.tensor("w", Dt.f32, &.{ 2, 3 });
        try v.tensor("b", Dt.bf16, &.{3});
        try v.finish();
    }
    {
        var v = try Verify.init(testing.allocator, &h);
        defer v.deinit(testing.allocator);
        try v.tensor("w", Dt.f32, &.{ 2, 3 });
        try testing.expectError(error.UnexpectedWeight, v.finish());
        try testing.expectError(error.MissingWeight, v.tensor("c", Dt.bf16, &.{3}));
        try testing.expectError(error.ShapeMismatch, v.tensor("b", Dt.bf16, &.{ 1, 3 }));
        try testing.expectError(error.DtypeMismatch, v.tensor("b", Dt.f32, &.{3}));
    }
}

test "plan: BF16 and F32 placed 256-aligned; F16 refused for the BF16 binding" {
    var h = try parseText(
        \\{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},"b":{"dtype":"BF16","shape":[200],"data_offsets":[12,412]}}
    , 412);
    defer h.deinit();
    var p = try Plan.init(testing.allocator, &h);
    defer p.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 256 + 512), p.resident);
    try testing.expectEqual(@as(u64, 0), p.offsets[0]);
    try testing.expectEqual(@as(u64, 256), p.offsets[1]);
    try testing.expectEqual(@as(u64, 1), p.cast);
    var g = try parseText(
        \\{"h":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}
    , 2);
    defer g.deinit();
    try testing.expectError(error.UnsupportedDeviceDtype, Plan.init(testing.allocator, &g));
}

test "resident: a manifest's device bytes without I/O" {
    const Dt = enum { bf16, f32, i32 };
    var r: Resident = .{};
    try r.tensor("a", Dt.bf16, &.{ 4096, 4096 });
    try r.tensor("b", Dt.f32, &.{3});
    try r.tensor("c", Dt.i32, &.{1});
    try testing.expectEqual(@as(u64, 4096 * 4096 * 2 + 256 + 256), r.bytes);
    try testing.expectEqual(@as(u64, 1), r.refused);
}

test "checkpoint: a file's header and data region, read back exact" {
    const gpa = testing.allocator;
    const io = testing.io;
    const dir = "/var/tmp/k6-weights-test";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[8,10]}}  ";
    var file: [8 + header.len + 10]u8 = undefined;
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8 .. 8 + header.len], header);
    std.mem.writeInt(u32, file[8 + header.len ..][0..4], 0x3F80_8000, .little);
    std.mem.writeInt(u32, file[8 + header.len + 4 ..][0..4], 0x3F81_8000, .little);
    std.mem.writeInt(u16, file[8 + header.len + 8 ..][0..2], 0x4000, .little);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/m.safetensors", .data = &file });
    var ck = try Checkpoint.open(gpa, io, dir ++ "/m.safetensors");
    defer ck.close(io);
    try testing.expectEqual(@as(u64, 8 + header.len), ck.data_start);
    const t = ck.header.get("a").?;
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(block), 8 + 2 * block);
    defer gpa.free(buf);
    const from = try ck.read(io, t.begin, 8, buf);
    try testing.expectEqual(@as(u32, 0x3F80_8000), std.mem.readInt(u32, buf[from..][0..4], .little));
    try testing.expectEqual(@as(?u16, 0x3F82), bf16Rne(std.mem.readInt(u32, buf[from + 4 ..][0..4], .little)));
}

/// A driver whose custody calls answer scripted raw results and count successful frees (never initialized: only
/// the fields set here are called).
const fake = struct {
    var query_rc: abi.Result = abi.success;
    var queries: u32 = 0;
    var mem_free_rc: abi.Result = abi.success;
    var host_free_rc: abi.Result = abi.success;
    var destroy_rc: abi.Result = abi.success;
    var mem_calls: u32 = 0;
    var mem_frees: u32 = 0;
    var host_calls: u32 = 0;
    var host_frees: u32 = 0;
    var destroy_calls: u32 = 0;
    var destroys: u32 = 0;

    fn reset() void {
        query_rc = abi.success;
        mem_free_rc = abi.success;
        host_free_rc = abi.success;
        destroy_rc = abi.success;
        inline for (.{ "queries", "mem_calls", "mem_frees", "host_calls", "host_frees", "destroy_calls", "destroys" }) |n| @field(fake, n) = 0;
    }

    fn streamQuery(_: abi.Stream) callconv(.c) abi.Result {
        queries += 1;
        return query_rc;
    }
    fn memFree(_: abi.DevicePtr) callconv(.c) abi.Result {
        mem_calls += 1;
        if (mem_free_rc == abi.success) mem_frees += 1;
        return mem_free_rc;
    }
    fn memFreeHost(_: ?*anyopaque) callconv(.c) abi.Result {
        host_calls += 1;
        if (host_free_rc == abi.success) host_frees += 1;
        return host_free_rc;
    }
    fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
        destroy_calls += 1;
        if (destroy_rc == abi.success) destroys += 1;
        return destroy_rc;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuStreamQuery = &streamQuery;
        d.api.cuMemFree_v2 = &memFree;
        d.api.cuMemFreeHost = &memFreeHost;
        d.api.cuEventDestroy_v2 = &eventDestroy;
        return d;
    }
};

const cuda_error_deinitialized: abi.Result = 4;
const cuda_error_illegal_address: abi.Result = 700;

/// A custody over fake handles: `chunks` live pinned chunks, `events` live events, a device allocation and a real
/// plan (testing.allocator: a plan freed twice or never fails the test).
fn fakeCustody(d: *const cuda.Driver, h: *const Header, chunks: usize, events: usize, enqueued: bool) !Custody {
    const plan = try Plan.init(testing.allocator, h);
    var c: Custody = .{ .gpa = testing.allocator, .d = d, .s = .{ .d = d, .handle = @ptrFromInt(0x40) }, .staging = .{ .d = d, .s = .{ .d = d, .handle = @ptrFromInt(0x40) } }, .dev_ptr = 0x10_0000, .dev_len = plan.resident, .plan = plan };
    for (0..chunks) |i| c.staging.host[i] = @ptrFromInt(0x1000 * (i + 1));
    for (0..events) |i| c.staging.events[i] = @ptrFromInt(0x100 * (i + 1));
    c.staging.slots.enqueued = enqueued;
    return c;
}

fn oneTensor() !Header {
    return parseText(
        \\{"w":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}
    , 8);
}

/// A scripted raw probe for `quiesce`: answers `answers` in order (then the last forever), counting the questions.
const FakeProbe = struct {
    answers: []const abi.Result,
    asked: *usize,
    pub const call = "fakeQuery";

    pub fn query(p: FakeProbe) abi.Result {
        const i = @min(p.asked.*, p.answers.len - 1);
        p.asked.* += 1;
        return p.answers[i];
    }
};

test "quiesce: quiet only on CUDA_SUCCESS, after not-ready polls" {
    var asked: usize = 0;
    const r = quiesce(testing.io, FakeProbe{ .answers = &.{ abi.error_not_ready, abi.error_not_ready, abi.success }, .asked = &asked }, .{});
    try testing.expect(r == .quiet);
    try testing.expectEqual(@as(usize, 3), asked);
}

test "quiesce: never done times out" {
    var asked: usize = 0;
    try testing.expect(quiesce(testing.io, FakeProbe{ .answers = &.{abi.error_not_ready}, .asked = &asked }, .{ .budget_ms = 5 }) == .timeout);
}

test "quiesce: another CUresult is a raw failure, reported as returned" {
    var asked: usize = 0;
    const r = quiesce(testing.io, FakeProbe{ .answers = &.{ abi.error_not_ready, cuda_error_illegal_address }, .asked = &asked }, .{});
    try testing.expectEqual(cuda_error_illegal_address, r.failed.rc);
    try testing.expectEqualStrings("fakeQuery", r.failed.call);
}

test "quiesce: cancellation ends the wait without evidence" {
    var flag: std.atomic.Value(bool) = .init(true);
    var asked: usize = 0;
    try testing.expect(quiesce(testing.io, FakeProbe{ .answers = &.{abi.error_not_ready}, .asked = &asked }, .{ .cancel = &flag }) == .cancelled);
}

test "custody: an event record failure after a successful enqueue needs the stream's quiet answer" {
    var s: Slots = .{};
    try testing.expectEqual(@as(?Evidence, .idle), evidenceFor(s.enqueued, null));
    s.begin(0); // cuMemcpyHtoDAsync succeeded, cuEventRecord then failed: `recorded` is never reached
    try testing.expectEqual(Slot.unrecorded, s.state[0]);
    try testing.expectEqual(@as(?Evidence, null), evidenceFor(s.enqueued, null));
    try testing.expectEqual(@as(?Evidence, null), evidenceFor(s.enqueued, .timeout));
    try testing.expectEqual(@as(?Evidence, null), evidenceFor(s.enqueued, .cancelled));
    try testing.expectEqual(@as(?Evidence, null), evidenceFor(s.enqueued, .{ .failed = .{ .rc = cuda_error_illegal_address, .call = "cuStreamQuery" } }));
    try testing.expectEqual(@as(?Evidence, .quiet), evidenceFor(s.enqueued, .quiet));
}

fn freeCalls() u32 {
    return fake.mem_calls + fake.host_calls + fake.destroy_calls;
}

test "custody: busy polling frees nothing and may repeat; a raw destroy failure is sticky, with no further call" {
    fake.reset();
    const d = fake.driver();
    var h = try oneTensor();
    defer h.deinit();
    var c = try fakeCustody(&d, &h, 2, 2, true);
    fake.query_rc = abi.error_not_ready;
    for (0..2) |_| { // busy: only queries, repeatable
        const busy = c.release(testing.io, .{ .budget_ms = 2 });
        try testing.expect(busy == .busy and busy.busy == .timeout);
    }
    try testing.expectEqual(@as(u32, 0), freeCalls());
    fake.query_rc = abi.success;
    fake.destroy_rc = cuda_error_deinitialized;
    const part = c.release(testing.io, .{});
    try testing.expectEqual(cuda_error_deinitialized, part.partial.rc);
    try testing.expectEqualStrings("cuEventDestroy_v2", part.partial.call);
    // one call each: two failed destroys, two successful host frees, one successful device free
    try testing.expectEqual(@as(u32, 2), fake.destroy_calls);
    try testing.expectEqual(@as(u32, 2), fake.host_frees);
    try testing.expectEqual(@as(u32, 1), fake.mem_frees);
    const o = c.outstanding();
    try testing.expectEqual(@as(u32, 2), o.events);
    try testing.expectEqual(@as(u32, 2), o.failed);
    try testing.expectEqual(@as(u32, 0), o.pinned_chunks);
    try testing.expectEqual(@as(u64, 0), o.device_bytes);
    try testing.expect(!o.plan);
    // even with the driver now answering success, a repeated release calls nothing and answers the same failure
    fake.destroy_rc = abi.success;
    const calls = freeCalls();
    const queries = fake.queries;
    for (0..3) |_| {
        const again = c.release(testing.io, .{});
        try testing.expectEqual(cuda_error_deinitialized, again.partial.rc);
        try testing.expectEqualStrings("cuEventDestroy_v2", again.partial.call);
    }
    try testing.expectEqual(calls, freeCalls());
    try testing.expectEqual(queries, fake.queries);
    try testing.expectEqualDeep(o, c.outstanding());
    c.forgetAfterContextDestroyed(); // the test's stand-in for the worker's exit
}

test "custody: a raw query failure is reported with its CUresult; nothing freed" {
    fake.reset();
    const d = fake.driver();
    var h = try oneTensor();
    defer h.deinit();
    var c = try fakeCustody(&d, &h, 1, 1, true);
    fake.query_rc = cuda_error_illegal_address;
    const r = c.release(testing.io, .{});
    try testing.expectEqual(cuda_error_illegal_address, r.busy.failed.rc);
    try testing.expectEqualStrings("cuStreamQuery", r.busy.failed.call);
    try testing.expectEqual(@as(u32, 0), freeCalls());
    fake.query_rc = abi.success;
    try testing.expect(c.release(testing.io, .{}) == .released);
    try testing.expectEqual(@as(u32, 3), freeCalls()); // one each
}

test "custody: init failure (event creation failed after a chunk) is released on idle evidence; its failure sticks" {
    fake.reset();
    const d = fake.driver();
    var h = try oneTensor();
    defer h.deinit();
    var c = try fakeCustody(&d, &h, 1, 0, false);
    fake.query_rc = cuda_error_illegal_address; // must not be asked
    fake.host_free_rc = cuda_error_deinitialized;
    const r = c.release(testing.io, .{});
    try testing.expectEqualStrings("cuMemFreeHost", r.partial.call);
    try testing.expectEqual(@as(u32, 1), c.outstanding().pinned_chunks);
    try testing.expectEqual(@as(u32, 1), c.outstanding().failed);
    try testing.expectEqual(@as(u32, 0), fake.queries);
    fake.host_free_rc = abi.success;
    const calls = freeCalls();
    const again = c.release(testing.io, .{});
    try testing.expectEqual(cuda_error_deinitialized, again.partial.rc);
    try testing.expectEqual(calls, freeCalls());
    try testing.expectEqual(@as(u32, 1), c.outstanding().pinned_chunks);
    c.forgetAfterContextDestroyed();
}

test "device: lookups refused once a release starts; a failed device free is sticky; a clean release is final" {
    fake.reset();
    const d = fake.driver();
    var h = try oneTensor();
    defer h.deinit();
    const users = [_]cuda.Stream{.{ .d = &d, .handle = @ptrFromInt(0x40) }};
    {
        var w: Device = .{ .own = try fakeCustody(&d, &h, 0, 0, false), .header = &h };
        try testing.expectEqual(@as(?u64, 0x10_0000), w.get("w"));
        try testing.expect(w.release(testing.io, &.{}, .{}) == .no_evidence);
        try testing.expectEqual(Device.State.live, w.state);
        try testing.expect(w.get("w") != null);
        fake.mem_free_rc = cuda_error_deinitialized;
        const r = w.release(testing.io, &users, .{});
        try testing.expectEqualStrings("cuMemFree_v2", r.partial.call);
        try testing.expectEqual(Device.State.release_failed, w.state);
        try testing.expectEqual(@as(?u64, null), w.get("w"));
        const o = w.outstanding();
        try testing.expect(o.plan and o.device_bytes > 0 and o.failed == 1);
        fake.mem_free_rc = abi.success;
        const calls = freeCalls();
        const queries = fake.queries;
        for (0..3) |_| try testing.expectEqual(cuda_error_deinitialized, w.release(testing.io, &users, .{}).partial.rc);
        try testing.expectEqual(calls, freeCalls());
        try testing.expectEqual(queries, fake.queries);
        try testing.expectEqualDeep(o, w.outstanding());
        try testing.expectEqual(@as(?u64, null), w.get("w"));
        w.forgetAfterContextDestroyed();
        try testing.expectEqual(Device.State.forgotten, w.state);
        try testing.expect(w.release(testing.io, &users, .{}) == .no_evidence);
        try testing.expectEqual(calls, freeCalls());
    }
    {
        fake.reset();
        var w: Device = .{ .own = try fakeCustody(&d, &h, 0, 0, false), .header = &h };
        try testing.expect(w.release(testing.io, &users, .{}) == .released);
        try testing.expectEqual(Device.State.released, w.state);
        try testing.expectEqual(@as(?u64, null), w.get("w"));
        try testing.expectEqual(@as(u64, 0), w.residentBytes());
        const calls = freeCalls();
        try testing.expect(w.release(testing.io, &users, .{}) == .released);
        try testing.expectEqual(calls, freeCalls()); // nothing freed twice
        try testing.expectEqual(@as(u32, 1), fake.mem_frees);
    }
}
