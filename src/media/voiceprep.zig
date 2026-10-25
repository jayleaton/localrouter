//! Recorded clips made ready to clone a voice from (`api/voices.zig`): each clip trimmed to its speech, cut at pauses
//! into references of at most 30 s, and each reference brought to the same speech level, or refused with the reason.
//! The gates and their numbers are drama-tensorfold's clip preparation (`tools/zig/drama_voice_clip.py`, calibrated
//! upstream), on 10 ms frames at the codec's 44.1 kHz:
//!
//! - level: the mean power of the frames within 20 dB of the loud end (the 95th percentile); a frame 25 dB under it
//!   is quiet, and a quiet run of at least 0.15 s is a pause;
//! - refused: silence; clipping (3 or more runs of 3 samples held at a peak within 1 dB of full scale); a clip that
//!   starts inside speech (cut inside a word); a background less than 30 dB under the speech, on the whole clip and
//!   again on what is kept;
//! - kept: 0.25 s before the first loud frame to 0.25 s after the last; a clip that ends inside speech ends at its last
//!   pause instead (refused when the caller gave its transcript, which would no longer match);
//! - cut: a stretch longer than 30 s is cut inside a pause 10 to 30 s on, the last sentence-long pause (0.4 s)
//!   preferred, as upstream cuts a take;
//! - all clips together: 10 to 60 s of references, at most 8;
//! - gain: each reference to -20 dBFS speech level, less when a peak would pass -1 dBFS.
//!
//! Not checked (upstream measures these with spectra this port does not compute): a steady tone or hum in the pauses,
//! and a second speaker. The person cloning their voice is asked to record alone, somewhere quiet.

const std = @import("std");

pub const rate: u32 = 44_100;
const hop = rate / 100; // a 10 ms frame
const min_total_s = 10.0;
const max_total_s = 60.0;
const max_reference_s = 30.0;
const min_cut_s = 10.0;
pub const max_references = 8;
const level_dbfs = -20.0;
const peak_dbfs = -1.0;
const quiet_db = 25.0;
const pause_frames = 15;
const sentence_frames = 40;
const edge_frames = 25;
const boundary_frames = 2;
const clip_run = 3;
const clip_runs = 3;
const min_snr_db = 30.0;

pub const Clip = struct {
    samples: []const i16, // mono at `rate`
    transcript: bool = false, // the caller gave its words: it may not be cut or lose speech
};

pub const Reference = struct {
    clip: usize, // which clip it came from (0-based)
    samples: []const i16, // levelled, ready to encode
    seconds: f64,
    level_dbfs: f64,
    gain_db: f64,
    snr_db: f64,
};

pub const ClipReport = struct { seconds_in: f64, seconds_kept: f64, speech_left_out_s: f64, references: usize };

pub const Prepared = struct { references: []const Reference, clips: []const ClipReport, seconds: f64 };

pub const Error = error{ Invalid, OutOfMemory };

fn fail(a: std.mem.Allocator, why: *[]const u8, comptime fmt: []const u8, args: anytype) Error {
    why.* = std.fmt.allocPrint(a, fmt, args) catch "the clip cannot make a voice";
    return error.Invalid;
}

/// The clips' references and what was measured, in `a`; `error.Invalid` with `why` saying which clip and why not.
pub fn prepare(a: std.mem.Allocator, clips: []const Clip, why: *[]const u8) Error!Prepared {
    var refs: std.ArrayList(Reference) = .empty;
    const reports = try a.alloc(ClipReport, clips.len);
    var total: f64 = 0;
    for (clips, reports, 0..) |c, *rep, i| {
        const before = refs.items.len;
        rep.* = try one(a, c, i + 1, &refs, why);
        for (refs.items[before..]) |*r| r.clip = i;
        total += rep.seconds_kept;
    }
    if (total < min_total_s) return fail(a, why, "{d:.1} s of speech in all; a voice needs at least {d:.0} s: record more", .{ total, min_total_s });
    if (total > max_total_s) return fail(a, why, "{d:.1} s of speech in all; a voice takes at most {d:.0} s: send less", .{ total, max_total_s });
    if (refs.items.len > max_references) return fail(a, why, "the clips make {d} references; a voice takes at most {d}: send fewer, longer clips", .{ refs.items.len, max_references });
    return .{ .references = refs.items, .clips = reports, .seconds = total };
}

/// Clip `n` (1-based, for messages): its references appended to `refs`.
fn one(a: std.mem.Allocator, c: Clip, n: usize, refs: *std.ArrayList(Reference), why: *[]const u8) Error!ClipReport {
    const x = c.samples;
    const seconds_in = seconds(x.len);
    if (x.len < hop * 4) return fail(a, why, "clip {d} is {d:.2} s long: too short to hold speech", .{ n, seconds_in });
    var peak: u16 = 0;
    for (x) |s| peak = @max(peak, @abs(s));
    if (peak < 4) return fail(a, why, "clip {d} is silent", .{n}); // under 1e-4 of full scale
    const flat = clipped(x);
    if (flat >= clip_runs) return fail(a, why, "clip {d} is clipped ({d} flat tops): record it again with less gain", .{ n, flat });

    const db = try frameDb(a, x);
    const level = try speechLevel(a, db);
    // first on the whole clip, as noise hides the pauses the cut looks for; then on what is kept
    _ = try noise(a, db, level, n, why);
    const quiet = try a.alloc(bool, db.len);
    for (quiet, db) |*q, d| q.* = d < level - quiet_db;
    for (quiet[0..boundary_frames]) |q| if (!q) return fail(a, why, "clip {d} starts inside speech: leave a moment of silence before the first word", .{n});
    const first = std.mem.indexOfScalar(bool, quiet, false) orelse return fail(a, why, "clip {d} holds no speech", .{n});
    const last = std.mem.lastIndexOfScalar(bool, quiet, false).?;
    const start = first -| edge_frames;
    var end = @min(db.len, last + 1 + edge_frames);
    var dropped: usize = 0;
    const tail_quiet = for (quiet[quiet.len - boundary_frames ..]) |q| {
        if (!q) break false;
    } else true;
    if (!tail_quiet) {
        // it ends inside a word: end at the last pause instead
        if (c.transcript) return fail(a, why, "clip {d} ends inside speech, so it cannot match its transcript: leave a moment of silence after the last word", .{n});
        const p = lastPause(quiet, first, quiet.len) orelse return fail(a, why, "clip {d} ends inside speech and has no pause to end at", .{n});
        end = p;
        for (quiet[end..]) |q| dropped += @intFromBool(!q);
    }
    const kept_db = db[start..end];
    const kept_level = try speechLevel(a, kept_db);
    const snr = try noise(a, kept_db, kept_level, n, why);

    // references of at most 30 s, cut inside pauses
    var at = start;
    var made: usize = 0;
    while (at < end) {
        var stop = end;
        if (seconds((end - at) * hop) > max_reference_s) {
            if (c.transcript) return fail(a, why, "clip {d} is {d:.1} s of speech; with a transcript a clip may be at most {d:.0} s: split it, or leave its transcript out", .{ n, seconds((end - start) * hop), max_reference_s });
            stop = cutPoint(quiet, at, end) orelse return fail(a, why, "clip {d} runs on for more than {d:.0} s without a pause to cut it at", .{ n, max_reference_s });
        }
        try refs.append(a, try levelled(a, x[at * hop .. stop * hop], db[at..stop], snr));
        made += 1;
        // the next reference starts 0.25 s before its first loud frame
        const next = std.mem.indexOfScalarPos(bool, quiet[0..end], stop, false) orelse end;
        at = if (next >= end) end else @max(stop, next -| edge_frames);
    }
    return .{ .seconds_in = seconds_in, .seconds_kept = seconds((end - start) * hop), .speech_left_out_s = @as(f64, @floatFromInt(dropped)) / 100, .references = made };
}

fn seconds(samples: usize) f64 {
    return @as(f64, @floatFromInt(samples)) / rate;
}

/// Runs of `clip_run` or more samples held at the clip's top or bottom value, counted only when that value is within
/// 1 dB of full scale: the flat tops clipping leaves (a quiet clip's peak may repeat by chance).
fn clipped(x: []const i16) usize {
    var top: i16 = std.math.minInt(i16);
    var bottom: i16 = std.math.maxInt(i16);
    for (x) |s| {
        top = @max(top, s);
        bottom = @min(bottom, s);
    }
    const near: u32 = 29_204; // -1 dBFS
    var n: usize = 0;
    for ([_]i16{ top, bottom }) |edge| {
        if (@abs(@as(i32, edge)) < near) continue;
        var run: usize = 0;
        for (x) |s| {
            if (s == edge) {
                run += 1;
                if (run == clip_run) n += 1;
            } else run = 0;
        }
    }
    return n;
}

/// Each 10 ms frame's power in dB of full scale.
fn frameDb(a: std.mem.Allocator, x: []const i16) Error![]f64 {
    const n = x.len / hop;
    const db = try a.alloc(f64, n);
    for (db, 0..) |*d, f| {
        var sum: f64 = 0;
        for (x[f * hop ..][0..hop]) |s| {
            const v = @as(f64, @floatFromInt(s)) / 32768.0;
            sum += v * v;
        }
        d.* = 10 * std.math.log10(sum / hop + 1e-10);
    }
    return db;
}

/// numpy's default percentile (linear between the closest ranks) of `v`, which is copied.
fn percentile(a: std.mem.Allocator, v: []const f64, p: f64) Error!f64 {
    const s = try a.dupe(f64, v);
    std.mem.sort(f64, s, {}, std.sort.asc(f64));
    const pos = p / 100 * @as(f64, @floatFromInt(s.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi = @min(lo + 1, s.len - 1);
    return s[lo] + (s[hi] - s[lo]) * (pos - @floor(pos));
}

/// The mean power of the frames within 20 dB of the loud end, so pauses and soft consonants do not lower it.
fn speechLevel(a: std.mem.Allocator, db: []const f64) Error!f64 {
    const floor = try percentile(a, db, 95) - 20;
    var sum: f64 = 0;
    var n: f64 = 0;
    for (db) |d| if (d >= floor) {
        sum += std.math.pow(f64, 10, d / 10);
        n += 1;
    };
    return 10 * std.math.log10(sum / n);
}

/// Decibels from the speech level down to the quietest tenth of the frames, refused under 30 dB.
fn noise(a: std.mem.Allocator, db: []const f64, level: f64, n: usize, why: *[]const u8) Error!f64 {
    const snr = level - try percentile(a, db, 10);
    if (snr < min_snr_db) return fail(a, why, "clip {d}: the background is {d:.1} dB under the speech; a voice needs {d:.0} dB: record somewhere quieter, closer to the microphone", .{ n, snr, min_snr_db });
    return snr;
}

/// The frame to end at inside the last pause of `quiet[from..to]` after its first loud frame `from`: 0.25 s into it,
/// or its middle when it is shorter than 0.5 s. Null when there is none.
fn lastPause(quiet: []const bool, from: usize, to: usize) ?usize {
    var best: ?usize = null;
    var i = from;
    while (i < to) {
        if (!quiet[i]) {
            i += 1;
            continue;
        }
        const s = i;
        while (i < to and quiet[i]) i += 1;
        if (i - s >= pause_frames and i < to) best = s + @min(edge_frames, (i - s) / 2);
    }
    return best;
}

/// Where to cut a stretch from frame `at`: inside a pause 10 to 30 s on, the last sentence-long pause (0.4 s) first,
/// else the last pause. Null when no pause falls in that window.
fn cutPoint(quiet: []const bool, at: usize, end: usize) ?usize {
    const lo = at + @as(usize, @intFromFloat(min_cut_s * 100));
    const hi = at + @as(usize, @intFromFloat(max_reference_s * 100));
    var long: ?usize = null;
    var any: ?usize = null;
    var i = at;
    while (i < end) {
        if (!quiet[i]) {
            i += 1;
            continue;
        }
        const s = i;
        while (i < end and quiet[i]) i += 1;
        if (i - s < pause_frames or i >= end or s == at) continue;
        const cut = s + @min(edge_frames, (i - s) / 2);
        if (cut < lo or cut > hi) continue;
        any = cut;
        if (i - s >= sentence_frames) long = cut;
    }
    return long orelse any;
}

/// `x` at -20 dBFS speech level, or as near as a -1 dBFS peak allows.
fn levelled(a: std.mem.Allocator, x: []const i16, db: []const f64, snr: f64) Error!Reference {
    const level = try speechLevel(a, db);
    var peak: u16 = 1;
    for (x) |s| peak = @max(peak, @abs(s));
    const peak_db = 20 * std.math.log10(@as(f64, @floatFromInt(peak)) / 32768.0);
    const gain = @min(level_dbfs - level, peak_dbfs - peak_db);
    const g = std.math.pow(f64, 10, gain / 20);
    const out = try a.alloc(i16, x.len);
    for (out, x) |*o, s| o.* = @intFromFloat(std.math.clamp(@round(@as(f64, @floatFromInt(s)) * g), -32768, 32767));
    return .{ .clip = 0, .samples = out, .seconds = seconds(x.len), .level_dbfs = level + gain, .gain_db = gain, .snr_db = snr };
}

// ---------------------------------------------------------------- tests

const testing = std.testing;

/// (seconds, tone) of a synthetic take.
const Part = struct { f64, bool };

/// A synthetic take: `pattern` of parts, tones at -26 dBFS, the rest a -76 dBFS hiss.
fn take(a: std.mem.Allocator, pattern: []const Part) ![]i16 {
    var out: std.ArrayList(i16) = .empty;
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    var t: usize = 0;
    for (pattern) |p| {
        const n: usize = @intFromFloat(p[0] * rate);
        for (0..n) |_| {
            const hiss = (r.float(f64) * 2 - 1) * 32768.0 * 0.0005;
            const tone = if (p[1]) @sin(@as(f64, @floatFromInt(t)) * 2 * std.math.pi * 180 / rate) * 32768.0 * 0.07 else 0;
            try out.append(a, @intFromFloat(@round(hiss + tone)));
            t += 1;
        }
    }
    return out.items;
}

test "a clean take is trimmed, levelled and kept whole" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var parts: std.ArrayList(Part) = .empty;
    try parts.append(a, .{ 1.0, false });
    for (0..6) |_| try parts.appendSlice(a, &.{ .{ 2.0, true }, .{ 0.5, false } });
    try parts.append(a, .{ 1.0, false });
    var why: []const u8 = "";
    const p = try prepare(a, &.{.{ .samples = try take(a, parts.items) }}, &why);
    try testing.expectEqual(@as(usize, 1), p.references.len);
    const r = p.references[0];
    try testing.expectApproxEqAbs(@as(f64, 14.5 + 0.5), r.seconds, 0.05); // 0.25 s edges either side of 14.5 s
    try testing.expectApproxEqAbs(level_dbfs, r.level_dbfs, 0.05);
    try testing.expect(r.snr_db > 30);
}

test "a long take is cut at pauses; several clips add up; refusals say which clip and why" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var parts: std.ArrayList(Part) = .empty;
    try parts.append(a, .{ 0.5, false });
    for (0..14) |_| try parts.appendSlice(a, &.{ .{ 3.0, true }, .{ 0.5, false } }); // 49 s
    var why: []const u8 = "";
    const long = try take(a, parts.items);
    const p = try prepare(a, &.{.{ .samples = long }}, &why);
    try testing.expect(p.references.len >= 2);
    for (p.references) |r| try testing.expect(r.seconds <= max_reference_s and r.seconds >= 3);
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = long, .transcript = true }}, &why));
    try testing.expect(std.mem.indexOf(u8, why, "with a transcript a clip may be at most 30 s") != null);

    const short = try take(a, &[_]Part{ .{ 0.5, false }, .{ 4.0, true }, .{ 0.5, false } });
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = short }}, &why));
    try testing.expect(std.mem.startsWith(u8, why, "4.5 s of speech in all; a voice needs at least 10 s"));
    const three = try prepare(a, &.{ .{ .samples = short }, .{ .samples = short }, .{ .samples = short } }, &why);
    try testing.expectEqual(@as(usize, 3), three.references.len);
    try testing.expectEqual(@as(usize, 2), three.references[2].clip);

    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = try take(a, &[_]Part{ .{ 4.0, true }, .{ 0.5, false } }) }}, &why));
    try testing.expectEqualStrings("clip 1 starts inside speech: leave a moment of silence before the first word", why);
    const silent = try a.alloc(i16, rate * 12);
    @memset(silent, 0);
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = silent }}, &why));
    try testing.expectEqualStrings("clip 1 is silent", why);
    // a background only 20 dB under the speech
    const noisy = try take(a, &[_]Part{ .{ 0.5, false }, .{ 12.0, true }, .{ 0.5, false } });
    var prng: std.Random.DefaultPrng = .init(3);
    for (noisy) |*s| s.* +|= @intFromFloat((prng.random().float(f64) * 2 - 1) * 32768.0 * 0.012);
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = noisy }}, &why));
    try testing.expect(std.mem.startsWith(u8, why, "clip 1: the background is"));
    // clipped: a tone driven past full scale
    const loud = try take(a, &[_]Part{ .{ 0.5, false }, .{ 12.0, true }, .{ 0.5, false } });
    for (loud) |*s| s.* = @intCast(std.math.clamp(@as(i32, s.*) * 20, -32768, 32767));
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = loud }}, &why));
    try testing.expect(std.mem.startsWith(u8, why, "clip 1 is clipped"));
}
