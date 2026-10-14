//! Bounded quiescence at the call boundary: polls cuStreamQuery on the caller's stream until it reports success, fails,
//! or the owner's bound passes, and returns what it observed (the last raw result, how many queries, how long).
//! cuStreamSynchronize is not used: it has no bound. A `Quiesced` proof comes only from `proof` over an observation
//! that synchronized within its bound; destroying a plan or handle, or unloading a library, takes one. An observation
//! that failed or timed out yields none: the caller keeps everything allocated and mapped (custody stays held).

const std = @import("std");

/// CUDA_ERROR_NOT_READY: the stream still has work.
pub const not_ready: c_int = 600;

pub const Clock = struct {
    ctx: ?*anyopaque = null,
    nowMs: *const fn (?*anyopaque) u64,
    sleepMs: *const fn (?*anyopaque, u64) void,
};

pub const Outcome = enum { synchronized, failed, timed_out };

pub const Observation = struct {
    outcome: Outcome,
    /// The last raw cuStreamQuery result.
    last_rc: c_int,
    queries: u32,
    waited_ms: u64,
    bound_ms: u64,
};

/// Built only by `proof`.
pub const Quiesced = struct {
    obs: Observation,

    /// Re-checked by every consumer, so a hand-built value with a failed observation still refuses.
    pub fn valid(self: Quiesced) bool {
        const o = self.obs;
        return o.outcome == .synchronized and o.last_rc == 0 and o.queries >= 1 and o.waited_ms <= o.bound_ms;
    }
};

/// Polls `api.cuStreamQuery(stream)` every `interval_ms` until success, a failure, or `bound_ms` has passed.
pub fn poll(api: anytype, stream: anytype, bound_ms: u64, interval_ms: u64, clock: Clock) Observation {
    const start = clock.nowMs(clock.ctx);
    var o: Observation = .{ .outcome = .timed_out, .last_rc = not_ready, .queries = 0, .waited_ms = 0, .bound_ms = bound_ms };
    while (true) {
        o.last_rc = api.cuStreamQuery(stream);
        o.queries += 1;
        o.waited_ms = clock.nowMs(clock.ctx) -| start;
        if (o.last_rc == 0) {
            o.outcome = if (o.waited_ms <= bound_ms) .synchronized else .timed_out;
            return o;
        }
        if (o.last_rc != not_ready) {
            o.outcome = .failed;
            return o;
        }
        if (o.waited_ms >= bound_ms) {
            o.outcome = .timed_out;
            return o;
        }
        clock.sleepMs(clock.ctx, @min(interval_ms, bound_ms - o.waited_ms));
    }
}

pub fn proof(o: Observation) ?Quiesced {
    const q: Quiesced = .{ .obs = o };
    return if (q.valid()) q else null;
}

/// A proof for tests and fakes only.
pub fn testProof() Quiesced {
    return .{ .obs = .{ .outcome = .synchronized, .last_rc = 0, .queries = 1, .waited_ms = 0, .bound_ms = 1 } };
}

// ---- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;

const FakeStream = struct {
    var ready_after: u32 = 0;
    var fail_rc: c_int = 0;
    var calls: u32 = 0;
    fn cuStreamQuery(_: ?*anyopaque) callconv(.c) c_int {
        calls += 1;
        if (fail_rc != 0) return fail_rc;
        return if (calls > ready_after) 0 else not_ready;
    }
};
const fake_api = struct {
    cuStreamQuery: *const fn (?*anyopaque) callconv(.c) c_int = FakeStream.cuStreamQuery,
}{};

const FakeClock = struct {
    var t: u64 = 0;
    fn now(_: ?*anyopaque) u64 {
        return t;
    }
    fn sleep(_: ?*anyopaque, ms: u64) void {
        t += ms;
    }
    const clock: Clock = .{ .nowMs = now, .sleepMs = sleep };
};

fn reset(ready_after: u32, fail_rc: c_int) void {
    FakeStream.ready_after = ready_after;
    FakeStream.fail_rc = fail_rc;
    FakeStream.calls = 0;
    FakeClock.t = 0;
}

test "synchronizes within the bound; times out or fails without a proof" {
    reset(3, 0);
    const ok = poll(fake_api, null, 100, 10, FakeClock.clock);
    try testing.expectEqual(Outcome.synchronized, ok.outcome);
    try testing.expectEqual(@as(u32, 4), ok.queries);
    try testing.expect(proof(ok) != null);

    reset(1000, 0);
    const late = poll(fake_api, null, 50, 10, FakeClock.clock);
    try testing.expectEqual(Outcome.timed_out, late.outcome);
    try testing.expectEqual(not_ready, late.last_rc);
    try testing.expect(proof(late) == null);

    reset(0, 700); // CUDA_ERROR_ILLEGAL_ADDRESS
    const bad = poll(fake_api, null, 50, 10, FakeClock.clock);
    try testing.expectEqual(Outcome.failed, bad.outcome);
    try testing.expectEqual(@as(c_int, 700), bad.last_rc);
    try testing.expect(proof(bad) == null);

    // a hand-built proof over a failed observation is still refused by every consumer
    try testing.expect(!(Quiesced{ .obs = bad }).valid());
}
