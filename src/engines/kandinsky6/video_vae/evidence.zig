//! k6v-qual's native allocation evidence under the D2 memory policy `gb10-unified-d2-v1` (tools/spark/k6-video-vae/
//! D2-EVIDENCE.md): a ledger of the live device buffers, capped at 24 GiB, and the run-bound files the shared staging
//! runner reads, in its allocator-evidence format (allocator-ready.json, allocator-metrics.jsonl, allocator-terminal.json;
//! the launcher writes allocator-producer-exit.json after it reaped the process, `k6v-qual --record-exit`).
//!
//! The ledger is allocation evidence, not a process bound: the CUDA context, driver allocations, loaded modules and every
//! host buffer are outside it and fall under the container's accounting. Every device allocation goes through
//! `Ledger.reserve` first, which refuses one that would pass the cap before anything is allocated. Buffers are allocated
//! and freed on one thread; the sampler thread only reads the ledger.
//!
//! Times are CLOCK_MONOTONIC seconds, the clock the runner's own samples use (containers share the host's).

const std = @import("std");
const linux = std.os.linux;

pub const policy = "gb10-unified-d2-v1";
pub const cap_bytes: u64 = 24 << 30;
/// The runner's raw-sample budget per producer.
pub const metrics_budget: u64 = 16 << 20;
/// Exit code of a producer whose evidence failed (a sample, the sampler's stop): never a successful case.
pub const exit_evidence: u8 = 67;

pub const Ledger = struct {
    cap: u64 = cap_bytes,
    live: std.atomic.Value(u64) = .init(0),
    peak: std.atomic.Value(u64) = .init(0),

    /// Records `len` more live bytes, or refuses when they would pass the cap.
    pub fn reserve(self: *Ledger, len: u64) error{LedgerCapExceeded}!void {
        const now = self.live.load(.acquire);
        if (len > self.cap or now > self.cap - len) return error.LedgerCapExceeded;
        self.live.store(now + len, .release);
        if (now + len > self.peak.load(.monotonic)) self.peak.store(now + len, .monotonic);
    }

    pub fn release(self: *Ledger, len: u64) void {
        _ = self.live.fetchSub(len, .acq_rel);
    }
};

pub fn monotonic() f64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) / 1e9;
}

pub fn sleepMs(ms: u64) void {
    const req: linux.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * std.time.ns_per_ms) };
    _ = linux.nanosleep(&req, null);
}

pub const MemInfo = struct { free: u64, total: u64 };

/// cuMemGetInfo, or a fake under --sim and in tests. Called from the sampler thread.
pub const MemQuery = struct {
    ctx: *anyopaque,
    func: *const fn (*anyopaque) anyerror!MemInfo,
};

/// The jobs this producer may serve; the runner's admission names the same ids.
pub const Job = enum {
    @"d2-engine",
    @"d2-probe-native",
};

pub const Identity = struct {
    nonce: [32]u8,
    job: Job,
    pid: i32,

    fn write(self: Identity, w: *std.Io.Writer) !void {
        try w.print("{{\"schema\": 1, \"policy\": \"{s}\", \"nonce\": \"{s}\", \"job_id\": \"{s}\", \"pid\": {d}", .{ policy, &self.nonce, @tagName(self.job), self.pid });
    }
};

/// The runner's variables, checked: the policy, a fresh 32-hex nonce, this job, the fixed file names under K6_OUTPUT.
/// null when K6_MEMORY_POLICY is unset (a run by hand: the ledger still caps, no evidence is written).
pub const Env = struct {
    id: Identity,
    out: []const u8,

    pub fn read(env: *const std.process.Environ.Map, a: std.mem.Allocator, job: Job) !?Env {
        const p = env.get("K6_MEMORY_POLICY") orelse return null;
        if (!std.mem.eql(u8, p, policy)) return error.MemoryPolicyDiffers;
        const nonce = env.get("K6_MEMORY_NONCE") orelse return error.MemoryNonceMissing;
        if (nonce.len != 32) return error.MemoryNonceInvalid;
        for (nonce) |c| if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) return error.MemoryNonceInvalid;
        if (!std.mem.eql(u8, env.get("K6_JOB_ID") orelse "", @tagName(job))) return error.MemoryPolicyJobDiffers;
        const out = env.get("K6_OUTPUT") orelse return error.OutputMissing;
        if (out.len == 0 or out[0] != '/') return error.OutputMissing;
        for ([_][2][]const u8{ .{ "K6_ALLOCATOR_RECEIPT", "allocator-ready.json" }, .{ "K6_ALLOCATOR_LOG", "allocator-metrics.jsonl" } }) |kv| {
            const want = try std.fmt.allocPrint(a, "{s}/{s}", .{ out, kv[1] });
            if (!std.mem.eql(u8, env.get(kv[0]) orelse "", want)) return error.AllocatorPathDiffers;
        }
        var id: Identity = .{ .nonce = undefined, .job = job, .pid = linux.getpid() };
        @memcpy(&id.nonce, nonce);
        return .{ .id = id, .out = try a.dupe(u8, out) };
    }
};

fn path0(buf: []u8, dir: []const u8, name: []const u8) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{s}", .{ dir, name }, 0);
}

fn writeAll(fd: i32, bytes: []const u8) !void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = linux.write(fd, bytes[off..].ptr, bytes.len - off);
        if (linux.errno(n) != .SUCCESS) return error.EvidenceWrite;
        off += n;
    }
}

/// Creates `name` under `dir` exclusively (never over an earlier receipt), atomically: written to name.tmp, renamed.
pub fn receipt(dir: []const u8, name: []const u8, bytes: []const u8) !void {
    var b1: [4096]u8 = undefined;
    var b2: [4096]u8 = undefined;
    const final = try path0(&b1, dir, name);
    const tmp = try std.fmt.bufPrintSentinel(&b2, "{s}.tmp", .{final}, 0);
    if (linux.errno(linux.access(final, linux.F_OK)) == .SUCCESS) return error.ReceiptExists;
    const r = linux.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, 0o644);
    if (linux.errno(r) != .SUCCESS) return error.ReceiptExists;
    const fd: i32 = @intCast(r);
    defer _ = linux.close(fd);
    try writeAll(fd, bytes);
    if (linux.errno(linux.fsync(fd)) != .SUCCESS) return error.EvidenceWrite;
    if (linux.errno(linux.rename(tmp, final)) != .SUCCESS) return error.EvidenceWrite;
}

fn readSmall(a: std.mem.Allocator, path: [:0]const u8, limit: usize) ![]u8 {
    const r = linux.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, 0);
    if (linux.errno(r) != .SUCCESS) return error.FileNotFound;
    const fd: i32 = @intCast(r);
    defer _ = linux.close(fd);
    var list: std.ArrayList(u8) = .empty;
    var chunk: [65536]u8 = undefined;
    while (true) {
        const n = linux.read(fd, &chunk, chunk.len);
        if (linux.errno(n) != .SUCCESS) return error.EvidenceRead;
        if (n == 0) break;
        if (list.items.len + n > limit) return error.EvidenceOversized;
        try list.appendSlice(a, chunk[0..n]);
    }
    return list.toOwnedSlice(a);
}

fn hex(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}

/// The producer: ready receipt, then one sample every `period_ms` from its own thread, then the terminal receipt.
pub const Producer = struct {
    a: std.mem.Allocator,
    env: Env,
    ledger: *Ledger,
    query: MemQuery,
    period_ms: u64,
    fd: i32 = -1,
    total: u64 = 0,
    bytes: u64 = 0,
    last: f64 = 0,
    stop: std.atomic.Value(bool) = .init(false),
    stopped: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// `start_s`: the process's own start (monotonic), recorded so the runner can bound ready - start.
    pub fn start(self: *Producer, runtime: []const u8, engine_sha256: []const u8, start_s: f64) !void {
        var b: [4096]u8 = undefined;
        const log = try path0(&b, self.env.out, "allocator-metrics.jsonl");
        if (linux.errno(linux.access(log, linux.F_OK)) == .SUCCESS) return error.ReceiptExists;
        const r = linux.open(log, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true, .APPEND = true }, 0o644);
        if (linux.errno(r) != .SUCCESS) return error.ReceiptExists;
        self.fd = @intCast(r);
        const m = try self.query.func(self.query.ctx);
        if (m.total == 0 or m.free > m.total) return error.MemInfoInconsistent;
        self.total = m.total;
        try self.sample();
        var w: std.Io.Writer.Allocating = .init(self.a);
        try self.env.id.write(&w.writer);
        try w.writer.print(", \"status\": \"passed\", \"runtime\": \"{s}\", \"engine_sha256\": \"{s}\", \"allocator_backend\": \"explicit\", \"allocator_cap_bytes\": {d}, \"mem_get_info_total_bytes\": {d}, \"sample_period_ms\": {d}, \"process_start_monotonic_seconds\": {d}, \"monotonic_seconds\": {d}, \"limitation\": \"Live DeviceBuffer ledger only; the CUDA context, driver allocations, modules and host memory are outside it and fall under the container accounting.\"}}\n", .{ runtime, engine_sha256, self.ledger.cap, self.total, self.period_ms, start_s, monotonic() });
        try receipt(self.env.out, "allocator-ready.json", w.written());
        self.thread = try std.Thread.spawn(.{}, run, .{self});
    }

    fn sample(self: *Producer) !void {
        const m = try self.query.func(self.query.ctx);
        const live = self.ledger.live.load(.acquire);
        if (m.total != self.total or m.free > m.total or live > self.ledger.cap) return error.SampleInconsistent;
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        const t = monotonic();
        if (t <= self.last) return error.SampleTimeDidNotAdvance;
        try self.env.id.write(&w);
        try w.print(", \"free_bytes\": {d}, \"total_bytes\": {d}, \"allocated_bytes\": {d}, \"reserved_bytes\": {d}, \"monotonic_seconds\": {d}}}\n", .{ m.free, m.total, live, live, t });
        if (self.bytes + w.end > metrics_budget) return error.MetricsBudgetExceeded;
        try writeAll(self.fd, w.buffered());
        self.bytes += w.end;
        self.last = t;
    }

    fn run(self: *Producer) void {
        defer self.stopped.store(true, .release);
        while (!self.stop.load(.acquire)) {
            var waited: u64 = 0;
            while (waited < self.period_ms and !self.stop.load(.acquire)) : (waited += 50) sleepMs(@min(50, self.period_ms - waited));
            if (self.stop.load(.acquire)) return;
            self.sample() catch |err| self.fail(err);
        }
    }

    /// A sample that cannot be taken ends the process: the payload must not finish as a qualified case without it.
    fn fail(self: *Producer, err: anyerror) noreturn {
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        self.env.id.write(&w) catch {};
        w.print(", \"status\": \"failed\", \"monotonic_seconds\": {d}, \"error\": \"{s}\"}}\n", .{ monotonic(), @errorName(err) }) catch {};
        writeAll(self.fd, w.buffered()) catch {};
        linux.exit_group(exit_evidence);
    }

    /// Stops the sampler (bounded: 2 s), takes the final sample and writes the terminal receipt. Returns the exit code.
    pub fn finish(self: *Producer, payload_exit: u8, extra: []const u8) u8 {
        self.stop.store(true, .release);
        const t0 = monotonic();
        while (!self.stopped.load(.acquire)) {
            if (monotonic() - t0 > 2.0) self.fail(error.SamplerDidNotStop);
            sleepMs(10);
        }
        self.thread.?.join();
        self.sample() catch |err| self.fail(err);
        _ = linux.close(self.fd);
        var b: [4096]u8 = undefined;
        const log = readSmall(self.a, path0(&b, self.env.out, "allocator-metrics.jsonl") catch unreachable, metrics_budget) catch |err| self.fail(err);
        if (log.len != self.bytes) self.fail(error.MetricsChanged);
        var w: std.Io.Writer.Allocating = .init(self.a);
        self.env.id.write(&w.writer) catch self.fail(error.OutOfMemory);
        w.writer.print(", \"status\": \"{s}\", \"payload_exit_code\": {d}, \"sampler_stopped\": true, \"last_sample_monotonic_seconds\": {d}, \"monotonic_seconds\": {d}, \"metrics_bytes\": {d}, \"metrics_sha256\": \"{s}\", \"ledger_peak_bytes\": {d}{s}}}\n", .{ if (payload_exit == 0) "completed" else "failed", payload_exit, self.last, monotonic(), log.len, &hex(log), self.ledger.peak.load(.acquire), extra }) catch self.fail(error.OutOfMemory);
        receipt(self.env.out, "allocator-terminal.json", w.written()) catch |err| self.fail(err);
        return payload_exit;
    }
};

/// The probe's sealed phase records (probe-phases.jsonl), one line per phase as it ends.
pub const Phases = struct {
    env: Env,
    fd: i32,
    bytes: u64 = 0,
    index: u32 = 0,

    pub const names = [_][]const u8{ "pre-context", "context", "alloc-4G", "free-4G", "alloc-16G", "free-16G" };

    pub fn open(env: Env) !Phases {
        var b: [4096]u8 = undefined;
        const p = try path0(&b, env.out, "probe-phases.jsonl");
        const r = linux.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true, .APPEND = true }, 0o644);
        if (linux.errno(r) != .SUCCESS) return error.ReceiptExists;
        return .{ .env = env, .fd = @intCast(r) };
    }

    /// Records the phase that held from `start` until now; phases must come in their sealed order.
    pub fn record(self: *Phases, x_bytes: u64, start: f64) !void {
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        try self.env.id.write(&w);
        try w.print(", \"phase\": \"P{d}\", \"name\": \"{s}\", \"index\": {d}, \"x_bytes\": {d}, \"start_monotonic_seconds\": {d}, \"end_monotonic_seconds\": {d}}}\n", .{ self.index, names[self.index], self.index, x_bytes, start, monotonic() });
        try writeAll(self.fd, w.buffered());
        if (linux.errno(linux.fsync(self.fd)) != .SUCCESS) return error.EvidenceWrite;
        self.bytes += w.end;
        self.index += 1;
    }

    /// The terminal receipt's binding of the phase records: `, "phases_bytes": N, "phases_sha256": "…"`.
    pub fn seal(self: *Phases, a: std.mem.Allocator) ![]const u8 {
        _ = linux.close(self.fd);
        if (self.index != names.len) return error.PhasesIncomplete;
        var b: [4096]u8 = undefined;
        const bytes = try readSmall(a, try path0(&b, self.env.out, "probe-phases.jsonl"), 1 << 20);
        if (bytes.len != self.bytes) return error.PhasesChanged;
        return std.fmt.allocPrint(a, ", \"phases_bytes\": {d}, \"phases_sha256\": \"{s}\"", .{ bytes.len, &hex(bytes) });
    }
};

/// `k6v-qual --record-exit CODE`, run by the launcher only after it reaped the producer: the reaped record, bound to the
/// ready receipt's PID and the terminal receipt's hash (null when the producer left none). No CUDA.
pub fn recordExit(a: std.mem.Allocator, env: Env, code: u8) !void {
    var b: [4096]u8 = undefined;
    var pid: ?i64 = null;
    if (readSmall(a, try path0(&b, env.out, "allocator-ready.json"), 1 << 20)) |bytes| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
        const o = v.object;
        if (!std.mem.eql(u8, o.get("nonce").?.string, &env.id.nonce) or !std.mem.eql(u8, o.get("job_id").?.string, @tagName(env.id.job))) return error.ReapedIdentityDiffers;
        pid = o.get("pid").?.integer;
    } else |_| {}
    var term: ?[64]u8 = null;
    if (readSmall(a, try path0(&b, env.out, "allocator-terminal.json"), 1 << 20)) |bytes| term = hex(bytes) else |_| {}
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.print("{{\"schema\": 1, \"policy\": \"{s}\", \"nonce\": \"{s}\", \"job_id\": \"{s}\", \"pid\": ", .{ policy, &env.id.nonce, @tagName(env.id.job) });
    if (pid) |p| try w.writer.print("{d}", .{p}) else try w.writer.writeAll("null");
    try w.writer.print(", \"producer_exit_code\": {d}, \"monotonic_seconds\": {d}, \"terminal_sha256\": ", .{ code, monotonic() });
    if (term) |t| try w.writer.print("\"{s}\"", .{&t}) else try w.writer.writeAll("null");
    try w.writer.writeAll(", \"evidence\": \"sealed launcher waited/reaped the producer before this CPU helper\"}\n");
    try receipt(env.out, "allocator-producer-exit.json", w.written());
}

test "ledger refuses past the cap before allocating, and releases" {
    var l: Ledger = .{ .cap = 100 };
    try l.reserve(60);
    try std.testing.expectError(error.LedgerCapExceeded, l.reserve(41));
    try std.testing.expectEqual(@as(u64, 60), l.live.load(.acquire));
    try l.reserve(40);
    try std.testing.expectError(error.LedgerCapExceeded, l.reserve(1));
    l.release(100);
    try std.testing.expectError(error.LedgerCapExceeded, l.reserve(101));
    try std.testing.expectEqual(@as(u64, 0), l.live.load(.acquire));
    try std.testing.expectEqual(@as(u64, 100), l.peak.load(.acquire));
    var big: Ledger = .{};
    try big.reserve(cap_bytes);
    try std.testing.expectError(error.LedgerCapExceeded, big.reserve(1));
}

const Fake = struct {
    fn info(_: *anyopaque) anyerror!MemInfo {
        return .{ .free = 64 << 30, .total = 128 << 30 };
    }
};

test "producer writes ready, periodic samples, terminal and the reaped record" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var dirbuf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dirbuf, "/tmp/k6v-evidence-test-{d}", .{linux.getpid()}, 0);
    _ = linux.mkdir(dir, 0o700);
    var env: Env = .{ .id = .{ .nonce = "0123456789abcdef0123456789abcdef".*, .job = .@"d2-probe-native", .pid = linux.getpid() }, .out = dir };
    var ledger: Ledger = .{};
    var dummy: u8 = 0;
    var p: Producer = .{ .a = aa, .env = env, .ledger = &ledger, .query = .{ .ctx = &dummy, .func = Fake.info }, .period_ms = 100 };
    var phases = try Phases.open(env);
    try p.start("sim", "00", monotonic());
    try std.testing.expectError(error.ReceiptExists, receipt(dir, "allocator-ready.json", "{}"));
    for (Phases.names, 0..) |_, k| {
        const t = monotonic();
        if (k == 2) try ledger.reserve(4 << 30);
        if (k == 3) ledger.release(4 << 30);
        sleepMs(150);
        try phases.record(if (k >= 2) (if (k < 4) 4 << 30 else 16 << 30) else 0, t);
    }
    try std.testing.expectEqual(@as(u8, 0), p.finish(0, try phases.seal(aa)));
    try recordExit(aa, env, 0);
    try std.testing.expectError(error.ReceiptExists, recordExit(aa, env, 0));
    var b: [4096]u8 = undefined;
    const log = try readSmall(aa, try path0(&b, dir, "allocator-metrics.jsonl"), 1 << 20);
    try std.testing.expect(std.mem.count(u8, log, "\n") >= 8);
    try std.testing.expect(std.mem.indexOf(u8, log, "\"allocated_bytes\": 4294967296, \"reserved_bytes\": 4294967296") != null);
    const term = try readSmall(aa, try path0(&b, dir, "allocator-terminal.json"), 1 << 20);
    const exited = try readSmall(aa, try path0(&b, dir, "allocator-producer-exit.json"), 1 << 20);
    try std.testing.expect(std.mem.indexOf(u8, exited, &hex(term)) != null);
    for ([_][]const u8{ "allocator-ready.json", "allocator-metrics.jsonl", "allocator-terminal.json", "allocator-producer-exit.json", "probe-phases.jsonl" }) |n|
        _ = linux.unlink(try path0(&b, dir, n));
    _ = linux.rmdir(dir);
    env.out = "";
}
