//! The one grant-scoped diagnostic contract for pinned vendor ops, and the production rule beside it.
//!
//! A diagnostic session exists only under a grant the owner issued. The grant binds the target, the source, the library
//! lock, each plan it allows (its kind, its prerequisite facts and the digest of its exact configuration), an expiry,
//! the host and device bytes, and the quiescence bound. A session walks one fixed sequence: validate the driver in use,
//! load the pinned set, warm up the selected plans, verify the closure, measure memory, run the selected bit gates; then
//! clean up under bounded quiescence.
//!
//! Every step is `enter` then `done`, coupled: `done` completes only the step `enter` opened, rechecks the expiry, and
//! takes that step's evidence (the validated driver observation, the loaded set, the warm-up's executed plans, the
//! verified set, the measured memory, the gate results), which it checks against the grant before granting the one proof
//! the step yields. A wrong step, wrong evidence, an expired grant or a reported call failure fails the session; the
//! verdict stays failed.
//!
//! The verdict and resource custody are kept apart. `cleanup` takes the actual bounded quiescence observation: only an
//! observation that synchronized within the grant's bound yields the proof that lets plans, handles and the set be
//! destroyed, and only once. Anything else retains custody for good. `released` then reads the set's own custody.
//!
//! Inputs versus proofs: G1, G2, P1, P2, P4 and P5 (as each plan's kind requires), the G3 driver record and G4 (the
//! sealed-memfd loader being the reviewed source) must be closed when the session begins. The proofs it produces are the
//! driver validation, the sealed load (whose static SONAME / NEEDED closure `pinned.Set.open` checks before any
//! dlopen), the warm-up, the dynamic closure (G6, before any bit-gate call), the memory bound (G5) and the bit gates.
//! G7 is held: a session uses the library bytes already on the target and grants no image or publication right.
//! Production (`closedForProduction`) needs every fact and every proof; no session changes it, and a daemon request
//! cannot begin a session.

const std = @import("std");
const pinned = @import("pinned.zig");
const driver = @import("driver.zig");
const quiesce = @import("quiesce.zig");
const log = @import("report.zig"); // .err at run time, .warn in tests

pub const Fact = enum {
    g1_constants,
    g2_algo_config,
    g3_driver_record,
    /// The sealed-memfd loader that will run is the reviewed source (readiness before the load; the load's result is
    /// the `sealed_load` proof).
    g4_loader_reviewed,
    g7_redistribution,
    p1_decoder_conv,
    p2_scores_gemm,
    p4_versions,
    p5_build_sequence,
};
pub const Facts = std.EnumSet(Fact);

pub const Proof = enum { driver, sealed_load, warmup, dynamic_closure, memory, bit_gates };
pub const Proofs = std.EnumSet(Proof);

/// The steps, in order; each yields the proof of the same name (`yields`).
pub const Step = enum { driver, load, warmup, verify_closure, measure, bit_gates };

pub const Phase = enum { begun, driver_ok, loaded, warmed, closed, measured, gated };

pub const Verdict = enum { open, passed, failed, incomplete };

pub const Custody = enum { held, quiesced, released, retained, release_failed };

pub const Origin = enum { owner_check, daemon_request };

/// Where the library bytes come from: only what the target already holds. An image or a publication needs G7.
pub const LibrarySource = enum { existing_target_bytes };

pub const PlanKind = enum { conv_vocoder, conv_decoder, gemm_scores, gemm_probs_v };

/// The input facts each kind of plan rests on, fixed in source: a grant may add to them, never remove.
pub fn minimumNeeds(kind: PlanKind) Facts {
    return switch (kind) {
        .conv_vocoder => .initMany(&.{ .p4_versions, .p5_build_sequence }),
        .conv_decoder => .initMany(&.{ .p1_decoder_conv, .p4_versions, .p5_build_sequence }),
        .gemm_scores => .initMany(&.{ .g1_constants, .g2_algo_config, .p2_scores_gemm, .p4_versions }),
        .gemm_probs_v => .initMany(&.{ .g1_constants, .g2_algo_config, .p4_versions }),
    };
}

/// What every session needs, whatever it selects.
pub const session_needs: Facts = .initMany(&.{ .g3_driver_record, .g4_loader_reviewed });

/// One plan a grant allows.
pub const PlanDef = struct {
    name: []const u8,
    kind: PlanKind,
    needs: Facts,
    /// `planDigest` of the plan's exact configuration record.
    config_digest: [32]u8,
};

pub const Grant = struct {
    id: []const u8,
    /// The target's host identity as the grant names it.
    target: []const u8,
    /// The source tree the session's binary was built from.
    source_tree: []const u8,
    /// `pinned.lockDigest` of the pins the set opens.
    lock_digest: [32]u8,
    plans: []const PlanDef,
    /// Unix seconds; at or past it every step fails.
    not_after: i64,
    /// Bounds on the set's host needs and on the measured host and device growth.
    host_bytes: u64,
    device_bytes: u64,
    /// The bound every cleanup observation must have been polled with.
    quiesce_bound_ms: u64,
    library_source: LibrarySource = .existing_target_bytes,
};

/// sha256 of a plan's configuration record (its shape, descriptors, engine / algo fields, workspace), as serialized by
/// its owner.
pub fn planDigest(record: []const u8) [32]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(record, &d, .{});
    return d;
}

/// What the session observes about itself.
pub const Observed = struct { target: []const u8, source_tree: []const u8, lock_digest: [32]u8 };

/// A plan the caller selects, with the digest of the configuration it will actually build.
pub const Selection = struct { name: []const u8, config_digest: [32]u8 };

pub const Warmup = struct {
    set: *pinned.Set,
    /// The selected plans executed once each on the specified scratch.
    executed: []const []const u8,
};

/// Measured at the step's boundary: cuMemGetInfo before and after, the process's resident host bytes before and after.
pub const Memory = struct {
    device_total: u64,
    device_free_before: u64,
    device_free_after: u64,
    host_rss_before: u64,
    host_rss_after: u64,
};

pub const GateResult = struct { name: []const u8, compared: u64, unequal: u64 };

pub const Evidence = union(Step) {
    driver: *const driver.Observation,
    load: *pinned.Set,
    warmup: Warmup,
    verify_closure: *pinned.Set,
    measure: Memory,
    bit_gates: []const GateResult,
};

pub const Error = error{
    GrantInvalid,
    GrantRefused,
    GrantMismatch,
    GrantExpired,
    PlanNotGranted,
    PlanConfigMismatch,
    PlanFactOpen,
    WrongPhase,
    ProofMissing,
    EvidenceRefused,
    ResourceExceeded,
    SessionFailed,
    CleanupRepeated,
    NotQuiesced,
};

pub const max_events = 32;
pub const max_selected = 16;

pub const Mark = enum { driver, load, warmup, verify_closure, measure, bit_gates, abort, quiesce, release };

pub const Event = struct { mark: Mark, at: i64, ok: bool };

/// The first failure, kept apart from cleanup.
pub const Failure = struct { mark: Mark, err: Error };

/// Refuses a grant with no plans, a duplicate plan name, a plan whose needs omit its kind's minimum, or a zero bound.
pub fn checkGrant(g: *const Grant) Error!void {
    if (g.plans.len == 0 or g.plans.len > max_selected) return error.GrantInvalid;
    if (g.host_bytes == 0 or g.device_bytes == 0 or g.quiesce_bound_ms == 0) return error.GrantInvalid;
    for (g.plans, 0..) |p, i| {
        for (g.plans[0..i]) |q| if (std.mem.eql(u8, p.name, q.name)) return error.GrantInvalid;
        if (!p.needs.supersetOf(minimumNeeds(p.kind))) {
            log.err("grant {s}: plan {s} omits its kind's prerequisites", .{ g.id, p.name });
            return error.GrantInvalid;
        }
    }
}

pub const Session = struct {
    grant: *const Grant,
    selected: []const Selection,
    phase: Phase = .begun,
    pending: ?Step = null,
    proofs: Proofs = .empty,
    verdict: Verdict = .open,
    failure: ?Failure = null,
    custody: Custody = .held,
    /// The set the load step produced; later evidence must be this set.
    set: ?*pinned.Set = null,
    events: [max_events]Event = undefined,
    n_events: usize = 0,

    /// Refuses unless the grant is valid, the origin is the owner's check path, the grant is unexpired and names this
    /// target, source and lock, every selection is a granted plan with the same configuration digest (each once), and
    /// every fact the selected plans and the session rest on is closed.
    pub fn begin(grant: *const Grant, origin: Origin, observed: Observed, facts: Facts, selected: []const Selection, now: i64) Error!Session {
        try checkGrant(grant);
        if (origin != .owner_check) return error.GrantRefused;
        if (now >= grant.not_after) return error.GrantExpired;
        if (!std.mem.eql(u8, observed.target, grant.target) or !std.mem.eql(u8, observed.source_tree, grant.source_tree) or
            !std.mem.eql(u8, &observed.lock_digest, &grant.lock_digest)) return error.GrantMismatch;
        if (selected.len == 0 or selected.len > max_selected) return error.PlanNotGranted;
        if (!facts.supersetOf(session_needs)) return error.PlanFactOpen;
        for (selected, 0..) |sel, i| {
            for (selected[0..i]) |prev| if (std.mem.eql(u8, prev.name, sel.name)) return error.PlanNotGranted;
            const def = planDef(grant, sel.name) orelse return error.PlanNotGranted;
            if (!std.mem.eql(u8, &def.config_digest, &sel.config_digest)) return error.PlanConfigMismatch;
            if (!facts.supersetOf(def.needs)) return error.PlanFactOpen;
        }
        return .{ .grant = grant, .selected = selected };
    }

    /// Opens `step`: the session has not failed or begun cleanup, no step is open, the grant is unexpired, the phase is
    /// the step's predecessor and every proof it needs is held. Anything else fails the session.
    pub fn enter(self: *Session, step: Step, now: i64) Error!void {
        try self.live();
        if (self.pending != null) return self.fail(markOf(step), error.WrongPhase, now);
        if (now >= self.grant.not_after) return self.fail(markOf(step), error.GrantExpired, now);
        if (self.phase != from(step)) return self.fail(markOf(step), error.WrongPhase, now);
        for (requires(step)) |p| if (!self.proofs.contains(p)) return self.fail(markOf(step), error.ProofMissing, now);
        self.pending = step;
    }

    /// Completes the open step with its evidence, checked against the grant; grants that step's one proof. A step that
    /// was not opened, evidence for another step, expiry or refused evidence fails the session.
    pub fn done(self: *Session, ev: Evidence, now: i64) Error!void {
        const step = std.meta.activeTag(ev);
        try self.live();
        if (self.pending != step) return self.fail(markOf(step), error.WrongPhase, now);
        if (now >= self.grant.not_after) return self.fail(markOf(step), error.GrantExpired, now);
        self.check(ev) catch |e| return self.fail(markOf(step), e, now);
        self.proofs.insert(yields(step));
        self.phase = to(step);
        self.pending = null;
        if (step == .bit_gates) self.verdict = .passed;
        self.record(markOf(step), now, true);
    }

    /// The caller's own call for the open (or next) step failed: the session fails, terminally.
    pub fn abort(self: *Session, now: i64) void {
        const failure = self.fail(.abort, error.SessionFailed, now);
        std.debug.assert(failure == error.SessionFailed);
    }

    /// Cleanup, once, from any phase: `obs` is the actual bounded stream query, polled with the grant's bound. Only a
    /// synchronized observation yields the proof that lets plans, handles and the set be destroyed; anything else
    /// retains custody for good. A session cleaned before its bit gates passed ends `incomplete`; a failed verdict
    /// stays failed whatever the cleanup does.
    pub fn cleanup(self: *Session, obs: quiesce.Observation, now: i64) Error!quiesce.Quiesced {
        if (self.custody != .held) return error.CleanupRepeated;
        if (self.verdict == .open) self.verdict = .incomplete;
        self.pending = null;
        const q = if (obs.bound_ms == self.grant.quiesce_bound_ms) quiesce.proof(obs) else null;
        self.record(.quiesce, now, q != null);
        if (q == null) {
            log.err("session: no quiescence (outcome {t}, rc {d}, {d} ms of {d}); custody retained", .{ obs.outcome, obs.last_rc, obs.waited_ms, obs.bound_ms });
            self.custody = .retained;
            return error.NotQuiesced;
        }
        self.custody = .quiesced;
        return q.?;
    }

    /// After the caller destroyed its plans and handles and closed the set with the cleanup's proof: reads the set's
    /// own custody. Once only.
    pub fn released(self: *Session, set: *const pinned.Set, now: i64) Error!void {
        if (self.custody != .quiesced) return error.CleanupRepeated;
        if (self.set) |s| if (s != set) {
            self.custody = .release_failed;
            self.record(.release, now, false);
            return error.EvidenceRefused;
        };
        const ok = set.custody == .released;
        self.custody = if (ok) .released else .release_failed;
        self.record(.release, now, ok);
    }

    /// The transitions, for the owner's export receipt.
    pub fn receipt(self: *const Session) []const Event {
        return self.events[0..self.n_events];
    }

    fn live(self: *const Session) Error!void {
        if (self.verdict == .failed or self.verdict == .incomplete or self.custody != .held) return error.SessionFailed;
    }

    fn fail(self: *Session, mark: Mark, err: Error, now: i64) Error {
        if (self.failure == null) self.failure = .{ .mark = mark, .err = err };
        self.verdict = .failed;
        self.pending = null;
        self.record(mark, now, false);
        return err;
    }

    fn check(self: *Session, ev: Evidence) Error!void {
        const g = self.grant;
        switch (ev) {
            .driver => |o| if (!o.validated) return error.EvidenceRefused,
            .load => |set| {
                const d = pinned.lockDigest(set.pins);
                if (!std.mem.eql(u8, &d, &g.lock_digest)) return error.EvidenceRefused;
                if (set.custody != .open or set.libs.len != set.pins.len or set.loaded != set.libs.len) return error.EvidenceRefused;
                if (set.needs.total() > g.host_bytes) return error.ResourceExceeded;
                self.set = set;
            },
            .warmup => |w| {
                if (w.set != self.set.?) return error.EvidenceRefused;
                if (w.set.changed != null) return error.EvidenceRefused;
                if (!sameNames(self.selected, w.executed)) return error.EvidenceRefused;
            },
            .verify_closure => |set| {
                if (set != self.set.?) return error.EvidenceRefused;
                if (set.closure == null) return error.EvidenceRefused;
                set.checkUnchanged() catch return error.EvidenceRefused;
            },
            .measure => |m| {
                if (m.device_total == 0 or m.device_free_before > m.device_total or m.device_free_after > m.device_total or
                    m.host_rss_before == 0 or m.host_rss_after == 0) return error.EvidenceRefused;
                if (m.device_free_before -| m.device_free_after > g.device_bytes) return error.ResourceExceeded;
                if (m.host_rss_after -| m.host_rss_before > g.host_bytes) return error.ResourceExceeded;
            },
            .bit_gates => |results| {
                if (results.len != self.selected.len) return error.EvidenceRefused;
                for (self.selected) |sel| {
                    var found: usize = 0;
                    for (results) |r| if (std.mem.eql(u8, r.name, sel.name)) {
                        found += 1;
                        if (r.compared == 0 or r.unequal != 0) return error.EvidenceRefused;
                    };
                    if (found != 1) return error.EvidenceRefused;
                }
            },
        }
    }

    fn record(self: *Session, mark: Mark, now: i64, ok: bool) void {
        if (self.n_events == max_events) return;
        self.events[self.n_events] = .{ .mark = mark, .at = now, .ok = ok };
        self.n_events += 1;
    }
};

fn planDef(g: *const Grant, name: []const u8) ?*const PlanDef {
    for (g.plans) |*p| if (std.mem.eql(u8, p.name, name)) return p;
    return null;
}

/// Each selected name executed exactly once, and nothing else.
fn sameNames(selected: []const Selection, executed: []const []const u8) bool {
    if (executed.len != selected.len) return false;
    for (selected) |sel| {
        var n: usize = 0;
        for (executed) |e| {
            if (std.mem.eql(u8, e, sel.name)) n += 1;
        }
        if (n != 1) return false;
    }
    return true;
}

fn from(step: Step) Phase {
    return switch (step) {
        .driver => .begun,
        .load => .driver_ok,
        .warmup => .loaded,
        .verify_closure => .warmed,
        .measure => .closed,
        .bit_gates => .measured,
    };
}

fn to(step: Step) Phase {
    return switch (step) {
        .driver => .driver_ok,
        .load => .loaded,
        .warmup => .warmed,
        .verify_closure => .closed,
        .measure => .measured,
        .bit_gates => .gated,
    };
}

fn yields(step: Step) Proof {
    return switch (step) {
        .driver => .driver,
        .load => .sealed_load,
        .warmup => .warmup,
        .verify_closure => .dynamic_closure,
        .measure => .memory,
        .bit_gates => .bit_gates,
    };
}

/// The proofs a step needs on entry, beyond its phase.
fn requires(step: Step) []const Proof {
    return switch (step) {
        .driver => &.{},
        .load => &.{.driver},
        .warmup => &.{ .driver, .sealed_load },
        .verify_closure => &.{.warmup},
        .measure => &.{.dynamic_closure},
        .bit_gates => &.{ .dynamic_closure, .memory },
    };
}

fn markOf(step: Step) Mark {
    return switch (step) {
        .driver => .driver,
        .load => .load,
        .warmup => .warmup,
        .verify_closure => .verify_closure,
        .measure => .measure,
        .bit_gates => .bit_gates,
    };
}

/// Production: every fact (G7 included) and every proof, from the target's own qualification records. A session's
/// proofs are evidence toward it, never a substitute.
pub fn closedForProduction(facts: Facts, proofs: Proofs) bool {
    return facts.eql(.full) and proofs.eql(.full);
}

// ---- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;
const Fake = @import("fake.zig").Fake;

const conv_record = "conv_pre: Cin 128 K 7 dil 1, engine 4, workspace 0";
const scores_record = "sdpa_scores: m 218 n 218 k 64, algo 20";

const test_pin: pinned.Pin = .{ .name = "liba.so.1", .path = "/opt/vendor/liba.so.1", .sha256 = @as([32]u8, @splat(1)), .size = 10 };
const test_pins = [_]pinned.Pin{test_pin};

fn testGrant(plans: []const PlanDef) Grant {
    return .{
        .id = "test",
        .target = "node-a",
        .source_tree = "1fc10ec7",
        .lock_digest = pinned.lockDigest(&test_pins),
        .plans = plans,
        .not_after = 1000,
        .host_bytes = 1 << 34,
        .device_bytes = 1 << 30,
        .quiesce_bound_ms = 30_000,
    };
}

fn planDefs() [2]PlanDef {
    return .{
        .{ .name = "conv_pre", .kind = .conv_vocoder, .needs = minimumNeeds(.conv_vocoder), .config_digest = planDigest(conv_record) },
        .{ .name = "sdpa_scores", .kind = .gemm_scores, .needs = minimumNeeds(.gemm_scores), .config_digest = planDigest(scores_record) },
    };
}

fn convSel() [1]Selection {
    return .{.{ .name = "conv_pre", .config_digest = planDigest(conv_record) }};
}
const facts_now: Facts = .initMany(&.{ .g3_driver_record, .g4_loader_reviewed, .p4_versions, .p5_build_sequence });

fn here(g: *const Grant) Observed {
    return .{ .target = "node-a", .source_tree = "1fc10ec7", .lock_digest = g.lock_digest };
}

/// A loaded one-library set, built by hand for the session tests (the loader's own tests cover opening one).
const TestSet = struct {
    fake: Fake = .{ .files = &.{} },
    libs: [1]pinned.Library = undefined,
    set: pinned.Set = undefined,

    fn init(self: *TestSet) void {
        self.libs = .{.{ .pin = &test_pins[0], .sys = self.fake.sys(), .fd = 3, .handle = @ptrFromInt(0x100000), .base = 0x100000, .dev_major = 0, .dev_minor = 1, .ino = 1003 }};
        self.set = .{
            .gpa = testing.allocator,
            .sys = self.fake.sys(),
            .policy = .{ .preexisting = &.{}, .deny_env = &.{} },
            .pins = &test_pins,
            .libs = &self.libs,
            .loaded = 1,
            .residents = &.{},
            .needs = .{ .shared = 10 },
        };
    }
};

const validated: driver.Observation = blk: {
    var o: driver.Observation = undefined;
    o.validated = true;
    break :blk o;
};

fn synced(bound: u64) quiesce.Observation {
    return .{ .outcome = .synchronized, .last_rc = 0, .queries = 2, .waited_ms = 5, .bound_ms = bound };
}

const memory_ok: Memory = .{ .device_total = 1 << 36, .device_free_before = 1 << 35, .device_free_after = (1 << 35) - (1 << 20), .host_rss_before = 1 << 30, .host_rss_after = (1 << 30) + (1 << 20) };

fn walk(s: *Session, ts: *TestSet) !void {
    try s.enter(.driver, 11);
    try s.done(.{ .driver = &validated }, 12);
    try s.enter(.load, 13);
    try s.done(.{ .load = &ts.set }, 14);
    try s.enter(.warmup, 15);
    try s.done(.{ .warmup = .{ .set = &ts.set, .executed = &.{"conv_pre"} } }, 16);
    ts.set.closure = ts.fake.sys().counts(ts.fake.sys().ctx);
    try s.enter(.verify_closure, 17);
    try s.done(.{ .verify_closure = &ts.set }, 18);
    try s.enter(.measure, 19);
    try s.done(.{ .measure = memory_ok }, 20);
    try s.enter(.bit_gates, 21);
    try s.done(.{ .bit_gates = &.{.{ .name = "conv_pre", .compared = 4096, .unequal = 0 }} }, 22);
}

test "a granted session walks the sequence and releases once" {
    var ts: TestSet = .{};
    ts.init();
    const defs = planDefs();
    const g = testGrant(&defs);
    const conv_sel = convSel();
    var s = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try walk(&s, &ts);
    try testing.expectEqual(Verdict.passed, s.verdict);
    const q = try s.cleanup(synced(g.quiesce_bound_ms), 23);
    try testing.expect(q.valid());
    ts.set.custody = .released; // what `Set.close` sets on success
    try s.released(&ts.set, 24);
    try testing.expectEqual(Custody.released, s.custody);
    try testing.expectError(error.CleanupRepeated, s.released(&ts.set, 25));
    try testing.expectError(error.CleanupRepeated, s.cleanup(synced(g.quiesce_bound_ms), 26));
    try testing.expect(!closedForProduction(facts_now, s.proofs)); // G1, G2, G7, P1, P2 still open
}

test "done cannot bypass enter, the phase, the expiry or the step's evidence" {
    var ts: TestSet = .{};
    ts.init();
    const defs = planDefs();
    const g = testGrant(&defs);
    const conv_sel = convSel();

    // completing a step never opened, with another step's claims, from the start
    var a = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try testing.expectError(error.WrongPhase, a.done(.{ .measure = memory_ok }, 11));
    try testing.expectEqual(Verdict.failed, a.verdict);
    try testing.expectError(error.SessionFailed, a.enter(.driver, 12)); // terminal
    try testing.expectEqual(@as(usize, 0), a.proofs.count());

    // evidence for a different step than the one opened
    var b = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try b.enter(.driver, 11);
    try testing.expectError(error.WrongPhase, b.done(.{ .load = &ts.set }, 12));
    try testing.expectEqual(Verdict.failed, b.verdict);

    // completion after expiry
    var c = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try c.enter(.driver, 11);
    try testing.expectError(error.GrantExpired, c.done(.{ .driver = &validated }, 1000));
    try testing.expect(!c.proofs.contains(.driver));

    // an unvalidated driver observation, then a warm-up of another plan than the selected one
    var d = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try d.enter(.driver, 11);
    var raw = validated;
    raw.validated = false;
    try testing.expectError(error.EvidenceRefused, d.done(.{ .driver = &raw }, 12));
    var e = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try e.enter(.driver, 11);
    try e.done(.{ .driver = &validated }, 12);
    try e.enter(.load, 13);
    try e.done(.{ .load = &ts.set }, 14);
    try e.enter(.warmup, 15);
    try testing.expectError(error.EvidenceRefused, e.done(.{ .warmup = .{ .set = &ts.set, .executed = &.{"sdpa_scores"} } }, 16));

    // memory above the grant's device bound
    ts = .{};
    ts.init();
    var f = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try f.enter(.driver, 11);
    try f.done(.{ .driver = &validated }, 12);
    try f.enter(.load, 13);
    try f.done(.{ .load = &ts.set }, 14);
    try f.enter(.warmup, 15);
    try f.done(.{ .warmup = .{ .set = &ts.set, .executed = &.{"conv_pre"} } }, 16);
    ts.set.closure = ts.fake.sys().counts(ts.fake.sys().ctx);
    try f.enter(.verify_closure, 17);
    try f.done(.{ .verify_closure = &ts.set }, 18);
    try f.enter(.measure, 19);
    var big = memory_ok;
    big.device_free_after = 0;
    try testing.expectError(error.ResourceExceeded, f.done(.{ .measure = big }, 20));
    try testing.expectEqual(Verdict.failed, f.verdict);
}

test "unknown quiescence is sticky; a failed verdict survives a successful cleanup" {
    var ts: TestSet = .{};
    ts.init();
    const defs = planDefs();
    const g = testGrant(&defs);
    const conv_sel = convSel();

    var a = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    var late = synced(g.quiesce_bound_ms);
    late.outcome = .timed_out;
    late.last_rc = quiesce.not_ready;
    try testing.expectError(error.NotQuiesced, a.cleanup(late, 11));
    try testing.expectError(error.CleanupRepeated, a.cleanup(synced(g.quiesce_bound_ms), 12)); // no later permission
    try testing.expectEqual(Custody.retained, a.custody);
    try testing.expectError(error.SessionFailed, a.enter(.driver, 13));

    // polled with a different bound than the grant's
    var b = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try testing.expectError(error.NotQuiesced, b.cleanup(synced(g.quiesce_bound_ms + 1), 11));

    // the primary failure stays failed; resource cleanup is reported separately
    var c = try Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 10);
    try c.enter(.driver, 11);
    c.abort(12);
    _ = try c.cleanup(synced(g.quiesce_bound_ms), 13);
    ts.set.custody = .release_failed; // `Set.close` hit a failed dlclose
    try c.released(&ts.set, 14);
    try testing.expectEqual(Verdict.failed, c.verdict);
    try testing.expectEqual(Mark.abort, c.failure.?.mark);
    try testing.expectEqual(Custody.release_failed, c.custody);
}

test "a grant binds each plan's prerequisites and configuration" {
    const conv_sel = convSel();
    // a plan claiming fewer prerequisites than its kind needs, or a duplicate name, invalidates the grant
    var weak = planDefs();
    weak[1].needs = .initMany(&.{.p4_versions});
    const gw = testGrant(&weak);
    try testing.expectError(error.GrantInvalid, Session.begin(&gw, .owner_check, here(&gw), facts_now, &conv_sel, 10));
    var dup = planDefs();
    dup[1].name = "conv_pre";
    const gd = testGrant(&dup);
    try testing.expectError(error.GrantInvalid, Session.begin(&gd, .owner_check, here(&gd), facts_now, &conv_sel, 10));

    const defs = planDefs();
    const g = testGrant(&defs);
    // a configuration other than the granted one
    const changed = [_]Selection{.{ .name = "conv_pre", .config_digest = planDigest("conv_pre: Cin 128 K 7 dil 2") }};
    try testing.expectError(error.PlanConfigMismatch, Session.begin(&g, .owner_check, here(&g), facts_now, &changed, 10));
    // G1 / G2 / P2 open: the scores GEMM cannot be selected
    const both = [_]Selection{ conv_sel[0], .{ .name = "sdpa_scores", .config_digest = planDigest(scores_record) } };
    try testing.expectError(error.PlanFactOpen, Session.begin(&g, .owner_check, here(&g), facts_now, &both, 10));
    try testing.expectError(error.GrantRefused, Session.begin(&g, .daemon_request, here(&g), facts_now, &conv_sel, 10));
    try testing.expectError(error.GrantExpired, Session.begin(&g, .owner_check, here(&g), facts_now, &conv_sel, 1000));
    var no_g4 = facts_now;
    no_g4.remove(.g4_loader_reviewed);
    try testing.expectError(error.PlanFactOpen, Session.begin(&g, .owner_check, here(&g), no_g4, &conv_sel, 10));
    var almost: Facts = .full;
    almost.remove(.g7_redistribution);
    try testing.expect(!closedForProduction(almost, .full));
    try testing.expect(closedForProduction(.full, .full));
}
