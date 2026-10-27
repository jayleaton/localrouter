//! Recorded clips made ready to clone a voice from (`api/voices.zig`): each clip trimmed to its speech, cut at pauses
//! into references of at most 30 s, and each reference brought to the same speech level. Only a clear failure is
//! refused (no speech, under 10 s of it in all, or a background about as loud as the voice); anything else that may
//! make the voice worse is kept and reported as a clip warning. The trim, cut and level numbers are drama-tensorfold's
//! clip preparation (`tools/zig/drama_voice_clip.py`), on 10 ms frames at the codec's 44.1 kHz:
//!
//! - speech: where the voice activity detector heard it (`Clip.speech`: Silero, which the transcription model runs),
//!   else the energy rule alone;
//! - level: the mean power of the speech frames within 20 dB of the loud end (the 95th percentile); a frame 25 dB under
//!   it, or outside the detector's speech, is quiet, and a quiet run of at least 0.15 s is a pause;
//! - background: the mean power of the frames at least 0.1 s from any speech, so a loud, compressed or normalised
//!   recording (whose quietest frames are its softest syllables) is not read as noisy; with less than 0.3 s of them it
//!   is not measured, and passes. Without the detector: the quietest tenth of all frames, as upstream. Under 30 dB
//!   below the speech is a warning; under 5 dB is refused unless the caller forces it (`Options.force`). Its spectrum
//!   says which: flat is noise, tonal (flatness under 0.06) is music or a hum;
//! - measured on LibriSpeech speaker 1272 (`testdata/`): clean and loudness-normalised or limited takes 32 to 36 dB, a
//!   noisy take 28.6, a heavily compressed take with no pauses 6.7; speech over music 10 dB under it 3.2, 20 dB under
//!   11.0;
//! - kept: 0.25 s before the first loud frame to 0.25 s after the last; a clip that ends inside speech ends at its last
//!   pause instead, unless the caller gave its transcript, which would no longer match;
//! - cut: a stretch longer than 30 s is cut inside a pause 10 to 30 s on, the last sentence-long pause (0.4 s)
//!   preferred, as upstream cuts a take; at 30 s when it has no pause (a warning); a clip with a transcript is not
//!   cut, so a longer one has its transcript set aside (a warning) and is transcribed;
//! - all clips together: at least 10 s of references; the first 60 s and at most 8 are kept (a warning);
//! - gain: each reference to -20 dBFS speech level, less when a peak would pass -1 dBFS;
//! - warnings, not refusals: clipping (3 or more runs of 3 samples held within 1 dB of full scale), a clip that starts
//!   or ends inside speech; a clip with no speech, or silent, is left out.
//!
//! Not checked: a second speaker. The person cloning their voice is asked to record alone, somewhere quiet.

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
const good_background_db = 30.0; // under it: a warning
const min_background_db = 5.0; // under it: refused, unless forced
const background_frames = 30; // fewer frames between the words: the background is not measured
const speech_margin_frames = 10; // breath, reverb and the detector's padding next to speech are not background
const min_speech_frames = 30; // less heard speech: the clip holds none
const tonal_flatness = 0.06;
const fft_n = 2048;

pub const Clip = struct {
    samples: []const i16, // mono at `rate`
    transcript: bool = false, // the caller gave its words: it may not be cut or lose speech
    speech: ?[]const [2]f64 = null, // where the voice activity detector heard speech, in seconds; null: not run
};

pub const Options = struct { force: bool = false }; // force: keep a clip whose background is about as loud as the voice

pub const Reference = struct {
    clip: usize, // which clip it came from (0-based)
    samples: []const i16, // levelled, ready to encode
    seconds: f64,
    level_dbfs: f64,
    gain_db: f64,
    snr_db: ?f64, // its clip's speech over background, dB; null: not measured
};

pub const Background = enum { noise, music };

pub const ClipReport = struct {
    seconds_in: f64 = 0,
    seconds_kept: f64 = 0,
    speech_left_out_s: f64 = 0,
    references: usize = 0,
    snr_db: ?f64 = null, // the speech level over the background between the words; null: too little of it to measure
    background: ?Background = null, // what the background sounds like, when measured
    transcript: bool = false, // its given transcript is used (false: none was given, or it was set aside)
    whole: bool = false, // one reference holds all its speech: a transcript of the whole clip is that reference's
    warnings: []const []const u8 = &.{},
};

pub const Prepared = struct { references: []const Reference, clips: []const ClipReport, seconds: f64 };

/// Invalid: no voice can be made (`why` says why); Noisy: a clip's background is about as loud as its speech, which
/// `Options.force` accepts.
pub const Error = error{ Invalid, Noisy, OutOfMemory };

fn fail(a: std.mem.Allocator, why: *[]const u8, comptime fmt: []const u8, args: anytype) error{ Invalid, OutOfMemory } {
    why.* = std.fmt.allocPrint(a, fmt, args) catch "the clip cannot make a voice";
    return error.Invalid;
}

/// The clips' references and what was measured, in `a`; on an error `why` says which clip and why not.
pub fn prepare(a: std.mem.Allocator, clips: []const Clip, opts: Options, why: *[]const u8) Error!Prepared {
    var refs: std.ArrayList(Reference) = .empty;
    const reports = try a.alloc(ClipReport, clips.len);
    for (clips, reports, 0..) |c, *rep, i| {
        const before = refs.items.len;
        rep.* = try one(a, c, i + 1, opts, &refs, why);
        for (refs.items[before..]) |*r| r.clip = i;
    }
    // the first 60 s and 8 references are kept
    var total: f64 = 0;
    var keep: usize = 0;
    for (refs.items) |r| {
        if (keep == max_references or total + r.seconds > max_total_s) break;
        total += r.seconds;
        keep += 1;
    }
    for (refs.items[keep..]) |r| reports[r.clip].references -= 1;
    var last_clip: ?usize = null;
    for (refs.items[keep..]) |r| {
        if (last_clip == r.clip) continue;
        last_clip = r.clip;
        const rep = &reports[r.clip];
        rep.whole = false;
        rep.warnings = try with(a, rep.warnings, if (rep.references == 0) "left out: a voice takes at most 60 s of speech and 8 references, which the clips before it fill" else "partly left out: a voice takes at most 60 s of speech and 8 references");
    }
    refs.shrinkRetainingCapacity(keep);
    if (total < min_total_s) {
        for (reports, 1..) |rep, n| if (rep.references == 0 and rep.warnings.len > 0)
            return fail(a, why, "{d:.1} s of speech in all; a voice needs at least {d:.0} s: record more (clip {d}: {s})", .{ total, min_total_s, n, rep.warnings[rep.warnings.len - 1] });
        return fail(a, why, "{d:.1} s of speech in all; a voice needs at least {d:.0} s: record more", .{ total, min_total_s });
    }
    return .{ .references = refs.items, .clips = reports, .seconds = total };
}

fn with(a: std.mem.Allocator, list: []const []const u8, item: []const u8) error{OutOfMemory}![]const []const u8 {
    const out = try a.alloc([]const u8, list.len + 1);
    @memcpy(out[0..list.len], list);
    out[list.len] = item;
    return out;
}

/// Clip `n` (1-based, for messages): its references appended to `refs`.
fn one(a: std.mem.Allocator, c: Clip, n: usize, opts: Options, refs: *std.ArrayList(Reference), why: *[]const u8) Error!ClipReport {
    const x = c.samples;
    var warn: std.ArrayList([]const u8) = .empty;
    var rep: ClipReport = .{ .seconds_in = seconds(x.len), .transcript = c.transcript };
    if (x.len < hop * 4) return leftOut(a, &rep, &warn, "left out: it is too short to hold speech");
    var peak: u16 = 0;
    for (x) |s| peak = @max(peak, @abs(s));
    if (peak < 4) return leftOut(a, &rep, &warn, "left out: it is silent"); // under 1e-4 of full scale
    const flat_tops = clipped(x);
    if (flat_tops >= clip_runs) try warn.append(a, try std.fmt.allocPrint(a, "it is clipped ({d} flat tops), so the voice may sound harsh: record it with less gain", .{flat_tops}));

    const db = try frameDb(a, x);
    const heard = if (c.speech) |iv| try frames(a, db.len, iv, 0) else null;
    if (heard) |h| if (count(h) < min_speech_frames) {
        const all = try a.alloc(bool, db.len);
        @memset(all, true);
        return leftOut(a, &rep, &warn, if (try tonal(a, x, all)) "left out: no speech was heard in it, only music" else "left out: no speech was heard in it");
    };
    const level = try speechLevel(a, db, heard);

    // the background: frames well away from the speech, or without the detector the quietest tenth
    const bg = try a.alloc(bool, db.len);
    var snr: ?f64 = null;
    if (c.speech) |iv| {
        for (bg, try frames(a, db.len, iv, speech_margin_frames)) |*b, near| b.* = !near;
        if (count(bg) >= background_frames) snr = level - meanDb(db, bg);
    } else {
        const p10 = try percentile(a, db, 10);
        for (bg, db) |*b, d| b.* = d <= p10;
        snr = level - p10;
    }
    if (snr) |v| {
        const music = try tonal(a, x, bg);
        rep.snr_db = v;
        rep.background = if (music) .music else .noise;
        if (v < min_background_db and !opts.force) {
            why.* = if (music)
                try std.fmt.allocPrint(a, "clip {d}: there is music (or a hum) under the speech, only {d:.1} dB quieter than the voice; a voice needs at least {d:.0} dB: send a recording without music behind the voice, or use it anyway (force=true)", .{ n, v, min_background_db })
            else
                try std.fmt.allocPrint(a, "clip {d}: the background noise is only {d:.1} dB quieter than the voice; a voice needs at least {d:.0} dB: record somewhere quieter, closer to the microphone, or use it anyway (force=true)", .{ n, v, min_background_db });
            return error.Noisy;
        }
        if (v < good_background_db) try warn.append(a, try std.fmt.allocPrint(a, "the background ({s}) is {d:.1} dB under the speech ({d:.0} dB is best), so the voice may carry some of it", .{ if (music) "music or a hum" else "noise", v, good_background_db }));
    }

    const quiet = try a.alloc(bool, db.len);
    for (quiet, db, 0..) |*q, d, f| q.* = d < level - quiet_db or (if (heard) |h| !h[f] else false);
    for (quiet[0..boundary_frames]) |q| if (!q) {
        try warn.append(a, "it starts inside speech, so its first word may be cut");
        break;
    };
    const first = std.mem.indexOfScalar(bool, quiet, false) orelse return leftOut(a, &rep, &warn, "left out: no speech was heard in it");
    const last = std.mem.lastIndexOfScalar(bool, quiet, false).?;
    const start = first -| edge_frames;
    var end = @min(db.len, last + 1 + edge_frames);
    var dropped: usize = 0;
    const tail_quiet = for (quiet[quiet.len - boundary_frames ..]) |q| {
        if (!q) break false;
    } else true;
    if (!tail_quiet) {
        // it ends inside a word: end at the last pause instead, unless its transcript says that word
        if (!rep.transcript) if (lastPause(quiet, first, quiet.len)) |p| {
            end = p;
            for (quiet[end..]) |q| dropped += @intFromBool(!q);
        };
        if (dropped > 0)
            try warn.append(a, try std.fmt.allocPrint(a, "it ends inside speech, so its last {d:.1} s of it were left out", .{@as(f64, @floatFromInt(dropped)) / 100}))
        else
            try warn.append(a, "it ends inside speech, so its last word may be cut");
    }
    if (rep.transcript and seconds((end - start) * hop) > max_reference_s) {
        rep.transcript = false;
        try warn.append(a, "its transcript was set aside: a clip with a transcript may be at most 30 s of speech, so it was cut and transcribed instead");
    }

    // references of at most 30 s, cut inside pauses
    var at = start;
    var made: usize = 0;
    var forced_cut = false;
    while (at < end) {
        var stop = end;
        if (seconds((end - at) * hop) > max_reference_s) stop = cutPoint(quiet, at, end) orelse cut: {
            if (!forced_cut) try warn.append(a, "it runs on for more than 30 s without a pause, so it was cut mid-speech");
            forced_cut = true;
            break :cut at + @as(usize, @intFromFloat(max_reference_s * 100));
        };
        try refs.append(a, try levelled(a, x[at * hop .. stop * hop], db[at..stop], snr));
        made += 1;
        // the next reference starts 0.25 s before its first loud frame
        const next = std.mem.indexOfScalarPos(bool, quiet[0..end], stop, false) orelse end;
        at = if (next >= end) end else @max(stop, next -| edge_frames);
    }
    rep.seconds_kept = seconds((end - start) * hop);
    rep.speech_left_out_s = @as(f64, @floatFromInt(dropped)) / 100;
    rep.references = made;
    rep.whole = made == 1 and dropped == 0;
    rep.warnings = warn.items;
    return rep;
}

/// A clip that makes no reference, and why (it does not refuse the others).
fn leftOut(a: std.mem.Allocator, rep: *ClipReport, warn: *std.ArrayList([]const u8), reason: []const u8) Error!ClipReport {
    try warn.append(a, reason);
    rep.transcript = false;
    rep.warnings = warn.items;
    return rep.*;
}

/// Which of `n` frames lie within `margin` frames of an interval of `iv` (seconds).
fn frames(a: std.mem.Allocator, n: usize, iv: []const [2]f64, margin: usize) error{OutOfMemory}![]bool {
    const m = try a.alloc(bool, n);
    @memset(m, false);
    for (iv) |span| {
        const lo = @min(n, frameAt(span[0]) -| margin);
        const hi = @min(n, frameAt(span[1]) +| 1 +| margin);
        if (lo < hi) @memset(m[lo..hi], true);
    }
    return m;
}

fn frameAt(t: f64) usize {
    if (!(t > 0)) return 0;
    return @intFromFloat(@min(t, 1e7) * 100);
}

fn count(m: []const bool) usize {
    var k: usize = 0;
    for (m) |b| k += @intFromBool(b);
    return k;
}

/// The mean power of the frames `pick` keeps, in dB.
fn meanDb(db: []const f64, pick: []const bool) f64 {
    var sum: f64 = 0;
    var k: f64 = 0;
    for (db, pick) |d, p| if (p) {
        sum += std.math.pow(f64, 10, d / 10);
        k += 1;
    };
    return 10 * std.math.log10(sum / k);
}

/// Whether the frames `pick` keeps sound tonal (music, a hum) rather than like noise: their median spectral flatness
/// (geometric over arithmetic mean power, 100 Hz to 8 kHz, a 2048-sample Hann window on every third frame) under 0.06.
/// Measured: music beds 0.00 to 0.05, a quiet room 0.11 to 0.14, pink noise 0.3.
fn tonal(a: std.mem.Allocator, x: []const i16, pick: []const bool) Error!bool {
    var vals: std.ArrayList(f64) = .empty;
    const re = try a.alloc(f64, fft_n);
    const im = try a.alloc(f64, fft_n);
    var k: usize = 0;
    for (pick, 0..) |p, f| {
        if (!p) continue;
        k += 1;
        if (k % 3 != 1) continue;
        const at = f * hop;
        if (at + fft_n > x.len) break;
        for (re, im, x[at..][0..fft_n], 0..) |*r, *i, s, j| {
            const w = 0.5 - 0.5 * @cos(2 * std.math.pi * @as(f64, @floatFromInt(j)) / (fft_n - 1));
            r.* = @as(f64, @floatFromInt(s)) / 32768.0 * w;
            i.* = 0;
        }
        fft(re, im);
        var log_sum: f64 = 0;
        var sum: f64 = 0;
        for (re[5..372], im[5..372]) |r, i| {
            const pw = r * r + i * i + 1e-20;
            log_sum += @log(pw);
            sum += pw;
        }
        try vals.append(a, @exp(log_sum / 367) / (sum / 367));
    }
    if (vals.items.len == 0) return false;
    return try percentile(a, vals.items, 50) < tonal_flatness;
}

/// In-place radix-2 FFT (`re.len` a power of 2).
fn fft(re: []f64, im: []f64) void {
    const n = re.len;
    var j: usize = 0;
    for (1..n) |i| {
        var bit = n >> 1;
        while (j & bit != 0) : (bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std.mem.swap(f64, &re[i], &re[j]);
            std.mem.swap(f64, &im[i], &im[j]);
        }
    }
    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        const half = len / 2;
        const step = -2 * std.math.pi / @as(f64, @floatFromInt(len));
        var i: usize = 0;
        while (i < n) : (i += len) {
            for (0..half) |m| {
                const wr = @cos(step * @as(f64, @floatFromInt(m)));
                const wi = @sin(step * @as(f64, @floatFromInt(m)));
                const p = i + m;
                const q = p + half;
                const vr = re[q] * wr - im[q] * wi;
                const vi = re[q] * wi + im[q] * wr;
                re[q] = re[p] - vr;
                im[q] = im[p] - vi;
                re[p] += vr;
                im[p] += vi;
            }
        }
    }
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
fn frameDb(a: std.mem.Allocator, x: []const i16) error{OutOfMemory}![]f64 {
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
fn percentile(a: std.mem.Allocator, v: []const f64, p: f64) error{OutOfMemory}!f64 {
    const s = try a.dupe(f64, v);
    std.mem.sort(f64, s, {}, std.sort.asc(f64));
    const pos = p / 100 * @as(f64, @floatFromInt(s.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi = @min(lo + 1, s.len - 1);
    return s[lo] + (s[hi] - s[lo]) * (pos - @floor(pos));
}

/// The mean power of the frames within 20 dB of the loud end, so pauses and soft consonants do not lower it; of the
/// frames `heard` keeps (the detector's speech) when given.
fn speechLevel(a: std.mem.Allocator, db: []const f64, heard: ?[]const bool) error{OutOfMemory}!f64 {
    var kept: std.ArrayList(f64) = .empty;
    for (db, 0..) |d, f| if (heard == null or heard.?[f]) try kept.append(a, d);
    if (kept.items.len == 0) return -100;
    const floor = try percentile(a, kept.items, 95) - 20;
    var sum: f64 = 0;
    var n: f64 = 0;
    for (kept.items) |d| if (d >= floor) {
        sum += std.math.pow(f64, 10, d / 10);
        n += 1;
    };
    return 10 * std.math.log10(sum / n);
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
fn levelled(a: std.mem.Allocator, x: []const i16, db: []const f64, snr: ?f64) error{OutOfMemory}!Reference {
    const level = try speechLevel(a, db, null);
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

fn has(list: []const []const u8, part: []const u8) bool {
    for (list) |w| if (std.mem.indexOf(u8, w, part) != null) return true;
    return false;
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
    const p = try prepare(a, &.{.{ .samples = try take(a, parts.items) }}, .{}, &why);
    try testing.expectEqual(@as(usize, 1), p.references.len);
    const r = p.references[0];
    try testing.expectApproxEqAbs(@as(f64, 14.5 + 0.5), r.seconds, 0.05); // 0.25 s edges either side of 14.5 s
    try testing.expectApproxEqAbs(level_dbfs, r.level_dbfs, 0.05);
    try testing.expect(r.snr_db.? > 30);
    try testing.expectEqual(@as(usize, 0), p.clips[0].warnings.len);
    try testing.expect(p.clips[0].whole);
}

test "a long take is cut at pauses; several clips add up; only clear failures are refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var parts: std.ArrayList(Part) = .empty;
    try parts.append(a, .{ 0.5, false });
    for (0..14) |_| try parts.appendSlice(a, &.{ .{ 3.0, true }, .{ 0.5, false } }); // 49 s
    var why: []const u8 = "";
    const long = try take(a, parts.items);
    const p = try prepare(a, &.{.{ .samples = long }}, .{}, &why);
    try testing.expect(p.references.len >= 2);
    for (p.references) |r| try testing.expect(r.seconds <= max_reference_s and r.seconds >= 3);
    // with a transcript it cannot be cut: the transcript is set aside
    const told = try prepare(a, &.{.{ .samples = long, .transcript = true }}, .{}, &why);
    try testing.expect(!told.clips[0].transcript and has(told.clips[0].warnings, "its transcript was set aside"));
    // over 60 s in all: the first 60 s are kept
    const many = try prepare(a, &.{ .{ .samples = long }, .{ .samples = long } }, .{}, &why);
    try testing.expect(many.seconds <= max_total_s and many.seconds > 40);
    try testing.expect(has(many.clips[1].warnings, "at most 60 s"));

    const short = try take(a, &[_]Part{ .{ 0.5, false }, .{ 4.0, true }, .{ 0.5, false } });
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = short }}, .{}, &why));
    try testing.expect(std.mem.startsWith(u8, why, "4.5 s of speech in all; a voice needs at least 10 s"));
    const three = try prepare(a, &.{ .{ .samples = short }, .{ .samples = short }, .{ .samples = short } }, .{}, &why);
    try testing.expectEqual(@as(usize, 3), three.references.len);
    try testing.expectEqual(@as(usize, 2), three.references[2].clip);

    // starting inside speech is a warning (the detector says where: an unbroken tone has no quiet tenth to measure)
    const from_start = [_][2]f64{.{ 0, 12.0 }};
    const cut = try prepare(a, &.{.{ .samples = try take(a, &[_]Part{ .{ 12.0, true }, .{ 0.5, false } }), .speech = &from_start }}, .{}, &why);
    try testing.expect(has(cut.clips[0].warnings, "starts inside speech"));
    // a silent clip is left out; alone, it makes no voice
    const silent = try a.alloc(i16, rate * 12);
    @memset(silent, 0);
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = silent }}, .{}, &why));
    try testing.expectEqualStrings("0.0 s of speech in all; a voice needs at least 10 s: record more (clip 1: left out: it is silent)", why);
    // a background 12 dB under the speech: a warning
    const middle = [_][2]f64{.{ 0.5, 12.5 }};
    const noisy = try take(a, &[_]Part{ .{ 0.5, false }, .{ 12.0, true }, .{ 0.5, false } });
    var prng: std.Random.DefaultPrng = .init(3);
    for (noisy) |*s| s.* +|= @intFromFloat((prng.random().float(f64) * 2 - 1) * 32768.0 * 0.012);
    const warned = try prepare(a, &.{.{ .samples = noisy, .speech = &middle }}, .{}, &why);
    try testing.expect(warned.clips[0].snr_db.? < 30 and warned.clips[0].background == .noise);
    try testing.expect(has(warned.clips[0].warnings, "the background (noise) is"));
    // one about as loud as the speech: refused, unless forced
    const drowned = try take(a, &[_]Part{ .{ 0.5, false }, .{ 12.0, true }, .{ 0.5, false } });
    for (drowned) |*s| s.* +|= @intFromFloat((prng.random().float(f64) * 2 - 1) * 32768.0 * 0.08);
    try testing.expectError(error.Noisy, prepare(a, &.{.{ .samples = drowned, .speech = &middle }}, .{}, &why));
    try testing.expect(std.mem.startsWith(u8, why, "clip 1: the background noise is only"));
    _ = try prepare(a, &.{.{ .samples = drowned, .speech = &middle }}, .{ .force = true }, &why);
    // clipped: a tone driven past full scale, kept with a warning
    const loud = try take(a, &[_]Part{ .{ 0.5, false }, .{ 12.0, true }, .{ 0.5, false } });
    for (loud) |*s| s.* = @intCast(std.math.clamp(@as(i32, s.*) * 20, -32768, 32767));
    try testing.expect(has((try prepare(a, &.{.{ .samples = loud, .speech = &middle }}, .{}, &why)).clips[0].warnings, "it is clipped"));
}

/// A 16 kHz fixture at `rate` (linear interpolation).
fn fixture(a: std.mem.Allocator, raw: []const u8) ![]i16 {
    const in = try a.alloc(i16, raw.len / 2);
    for (in, 0..) |*s, i| s.* = std.mem.readInt(i16, raw[i * 2 ..][0..2], .little);
    const out = try a.alloc(i16, in.len * rate / 16_000);
    for (out, 0..) |*o, i| {
        const pos = @as(f64, @floatFromInt(i)) * 16_000 / rate;
        const lo: usize = @intFromFloat(@floor(pos));
        const hi = @min(lo + 1, in.len - 1);
        const t = pos - @floor(pos);
        o.* = @intFromFloat(@round(@as(f64, @floatFromInt(in[lo])) * (1 - t) + @as(f64, @floatFromInt(in[hi])) * t));
    }
    return out;
}

test "real takes: a loudness-normalised, limited MP3 passes clean; a noisy one warns; music under the speech is named" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // testdata/README.md: LibriSpeech 1272, its Silero intervals (the transcription model's detector)
    const loud = try fixture(a, @embedFile("testdata/voice-loud.s16"));
    const loud_speech = [_][2]f64{ .{ 0.578, 7.07 }, .{ 7.618, 12.062 } };
    const noisy = try fixture(a, @embedFile("testdata/voice-noisy.s16"));
    const noisy_speech = [_][2]f64{ .{ 0.578, 2.654 }, .{ 2.85, 7.55 }, .{ 7.682, 10.718 } };
    var why: []const u8 = "";
    const p = try prepare(a, &.{ .{ .samples = loud, .speech = &loud_speech }, .{ .samples = noisy, .speech = &noisy_speech } }, .{}, &why);
    try testing.expect(p.clips[0].snr_db.? >= 30);
    try testing.expect(!has(p.clips[0].warnings, "background"));
    try testing.expect(p.clips[1].snr_db.? < 30 and has(p.clips[1].warnings, "the background (noise) is"));
    try testing.expectEqual(@as(usize, 2), p.references.len);

    // the same loud take with a loud chord bed under it: refused as music, kept when forced
    const bed = try a.dupe(i16, loud);
    for (bed, 0..) |*s, i| {
        const t = @as(f64, @floatFromInt(i)) / rate;
        const note = 220 + 110 * @mod(@floor(t / 0.6), 4);
        s.* +|= @intFromFloat(@round(6000 * (@sin(2 * std.math.pi * t * note) + 0.6 * @sin(2 * std.math.pi * t * 330))));
    }
    try testing.expectError(error.Noisy, prepare(a, &.{.{ .samples = bed, .speech = &loud_speech }}, .{}, &why));
    try testing.expect(std.mem.startsWith(u8, why, "clip 1: there is music (or a hum) under the speech"));
    const kept = try prepare(a, &.{.{ .samples = bed, .speech = &loud_speech }}, .{ .force = true }, &why);
    try testing.expect(kept.clips[0].background == .music);
    // a clip in which the detector heard nothing is left out, and says so
    const none = [_][2]f64{};
    try testing.expectError(error.Invalid, prepare(a, &.{.{ .samples = bed, .speech = &none }}, .{}, &why));
    try testing.expect(std.mem.indexOf(u8, why, "no speech was heard in it, only music") != null);
}
