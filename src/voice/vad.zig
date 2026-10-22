//! Server voice activity detection: the RMS of each 20 ms frame against an adaptive noise floor, cheap enough to run
//! on the WebSocket reader thread for every audio chunk. Timings follow OpenAI's `server_vad`.

const std = @import("std");

pub const Config = struct {
    threshold: f32 = 0.5, // 0 to 1: the margin over the noise floor, 6 + 24 x threshold dB
    prefix_padding_ms: u32 = 300, // speech_started reports the onset this much earlier
    silence_duration_ms: u32 = 500, // quiet after speech that ends the turn
    min_speech_ms: u32 = 60, // loud audio in a row before speech starts
    pause_ms: u32 = 0, // quiet after speech that reports speech_paused, once a quiet run (a draft transcript); 0: never
};

pub const Kind = enum { speech_started, speech_paused, speech_stopped };

/// A turn boundary at `ms` into the audio fed so far (speech_paused: where the quiet began).
pub const Event = struct { kind: Kind, ms: u64 };

const frame_ms = 20;
const floor_start: f32 = -60;
const floor_low: f32 = -90;
const floor_high: f32 = -20;
const loud_min: f32 = -50; // a frame quieter than this is never speech
const creep: f32 = 0.05; // share of the gap the floor rises by each frame
const silent_db: f32 = -100;
// In a pause (`pause_min_frames` quiet in a row), a loud run shorter than `resume_frames` (a click, a knock, a cough) is
// not the speaker going on: it counts as quiet, and the wait is not restarted. Frames are 20 ms and a sound rarely
// starts on a frame edge, so a 40 ms click can light 3 frames; Chrome's processing can smear it further.
const pause_min_frames: u32 = 10; // 200 ms
const resume_frames: u32 = 6; // 120 ms

pub const Vad = struct {
    cfg: Config,
    rate: u32,
    frame_len: u32, // samples a frame
    margin: f32, // dB over the floor for a loud frame
    speech_frames: u32, // loud frames in a row that start speech
    silence_frames: u32, // quiet frames in a row that stop it
    pause_frames: u32, // quiet frames in a row that report a pause; 0: none
    floor: f32 = floor_start,
    fed: u64 = 0, // samples fed
    acc: u64 = 0, // sum of squares of the partial frame
    filled: u32 = 0, // samples in the partial frame
    frames: u64 = 0, // whole frames seen
    active: bool = false,
    loud_run: u32 = 0,
    quiet_run: u32 = 0,
    burst: u32 = 0, // in a turn: loud frames in a row after a pause, not yet speech
    before_burst: u32 = 0, // the quiet run the burst interrupted

    /// A detector for mono audio at `rate` Hz, starting in silence.
    pub fn init(cfg: Config, rate: u32) Vad {
        const r: u32 = @max(rate, 1);
        const t: f32 = std.math.clamp(cfg.threshold, 0, 1);
        return .{
            .cfg = cfg,
            .rate = r,
            .frame_len = @max(r / (1000 / frame_ms), 1),
            .margin = 6 + 24 * t,
            .speech_frames = @max((cfg.min_speech_ms + frame_ms - 1) / frame_ms, 1),
            .silence_frames = @max((cfg.silence_duration_ms + frame_ms - 1) / frame_ms, 1),
            .pause_frames = (cfg.pause_ms + frame_ms - 1) / frame_ms,
        };
    }

    /// Feeds mono samples (any chunk size, frames carry over between calls); appends turn events in order.
    pub fn feed(v: *Vad, gpa: std.mem.Allocator, samples: []const i16, out: *std.ArrayList(Event)) !void {
        for (samples) |s| {
            const x: i64 = s;
            v.acc += @intCast(x * x);
            v.filled += 1;
            v.fed += 1;
            if (v.filled == v.frame_len) {
                const db = dbfs(v.acc, v.filled);
                v.acc = 0;
                v.filled = 0;
                try v.frame(gpa, db, out);
            }
        }
    }

    /// Whether a turn is in progress (speech_started seen, speech_stopped not yet).
    pub fn speaking(v: *const Vad) bool {
        return v.active;
    }

    /// Where the current quiet run began (ms into the audio fed), while a turn is in progress and quiet.
    pub fn quietSince(v: *const Vad) ?u64 {
        if (!v.active or v.quiet_run == 0) return null;
        return v.msAtFrame(v.frames - v.quiet_run);
    }

    /// The quiet that ends a turn, in ms.
    pub fn silenceMs(v: *const Vad) u64 {
        return v.msAtFrame(v.silence_frames);
    }

    /// Back to silence (after input_audio_buffer.clear or a manual commit); the noise floor is kept.
    pub fn reset(v: *Vad) void {
        v.active = false;
        v.loud_run = 0;
        v.quiet_run = 0;
        v.burst = 0;
    }

    /// Milliseconds of audio fed so far.
    pub fn position(v: *const Vad) u64 {
        return v.fed * 1000 / v.rate;
    }

    fn msAtFrame(v: *const Vad, f: u64) u64 {
        return f * v.frame_len * 1000 / v.rate;
    }

    fn frame(v: *Vad, gpa: std.mem.Allocator, db: f32, out: *std.ArrayList(Event)) !void {
        const loud = db > @max(v.floor + v.margin, loud_min);
        v.frames += 1;
        if (v.active) {
            // The turn ends on one unbroken quiet run of silence_duration_ms; speech restarts it. Mid-speech any loud frame
            // does (speech has short dips); in a pause, only `resume_frames` of loud frames in a row do, and a shorter
            // burst counts as quiet.
            if (!loud) {
                v.burst = 0;
                v.quiet_run += 1;
            } else if (v.burst > 0 or v.quiet_run >= pause_min_frames) {
                if (v.burst == 0) v.before_burst = v.quiet_run;
                v.burst += 1;
                v.quiet_run = if (v.burst >= resume_frames) 0 else v.before_burst + v.burst;
                if (v.burst >= resume_frames) v.burst = 0;
            } else v.quiet_run = 0;
            if (v.pause_frames > 0 and v.quiet_run == v.pause_frames and v.pause_frames < v.silence_frames) try out.append(gpa, .{ .kind = .speech_paused, .ms = v.msAtFrame(v.frames - v.quiet_run) });
            if (v.quiet_run >= v.silence_frames) {
                v.reset();
                try out.append(gpa, .{ .kind = .speech_stopped, .ms = v.msAtFrame(v.frames) });
            }
            return; // the floor is frozen while speaking
        }
        v.loud_run = if (loud) v.loud_run + 1 else 0;
        if (v.loud_run >= v.speech_frames) {
            const onset = v.msAtFrame(v.frames - v.loud_run);
            v.active = true;
            v.loud_run = 0;
            v.quiet_run = 0;
            try out.append(gpa, .{ .kind = .speech_started, .ms = onset -| v.cfg.prefix_padding_ms });
            return;
        }
        v.floor = if (db < v.floor) db else v.floor + creep * (db - v.floor);
        v.floor = std.math.clamp(v.floor, floor_low, floor_high);
    }
};

/// The RMS level of `n` samples whose squares sum to `acc`, in dBFS; all zeros is -100.
fn dbfs(acc: u64, n: u32) f32 {
    if (acc == 0 or n == 0) return silent_db;
    const ms: f64 = @as(f64, @floatFromInt(acc)) / @as(f64, @floatFromInt(n));
    const db: f64 = 10 * @log10(ms / (32768.0 * 32768.0));
    return @floatCast(@max(db, silent_db));
}

const test_rate: u32 = 24_000;

/// Appends `ms` of a 440 Hz tone (amplitude 8000) plus optional noise to `buf`.
fn tone(buf: *std.ArrayList(i16), ms: u32, noise: ?std.Random) !void {
    const n = ms * test_rate / 1000;
    const start = buf.items.len;
    for (0..n) |i| {
        const t: f64 = @as(f64, @floatFromInt(start + i)) / @as(f64, @floatFromInt(test_rate));
        var x: i32 = @intFromFloat(8000 * @sin(2 * std.math.pi * 440 * t));
        if (noise) |r| x += r.intRangeAtMost(i32, -100, 100);
        try buf.append(std.testing.allocator, @intCast(x));
    }
}

/// Appends `ms` of silence, or of low noise (amplitude 100) when `noise` is set.
fn quiet(buf: *std.ArrayList(i16), ms: u32, noise: ?std.Random) !void {
    const n = ms * test_rate / 1000;
    for (0..n) |_| try buf.append(std.testing.allocator, if (noise) |r| r.intRangeAtMost(i16, -100, 100) else 0);
}

/// Runs `samples` through a fresh VAD in chunks of `chunk` samples.
fn run(samples: []const i16, chunk: usize, out: *std.ArrayList(Event)) !void {
    var v: Vad = .init(.{}, test_rate);
    var at: usize = 0;
    while (at < samples.len) : (at += chunk) {
        const end: usize = @min(at + chunk, samples.len);
        try v.feed(std.testing.allocator, samples[at..end], out);
    }
    try std.testing.expectEqual(@as(u64, samples.len * 1000 / test_rate), v.position());
}

fn expectEvents(want: []const Event, got: []const Event) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        try std.testing.expectEqual(w.kind, g.kind);
        try std.testing.expect(@max(w.ms, g.ms) - @min(w.ms, g.ms) <= 20);
    }
}

test "a tone then silence is one turn; odd chunk sizes give the same events" {
    const gpa = std.testing.allocator;
    var buf: std.ArrayList(i16) = .empty;
    defer buf.deinit(gpa);
    try tone(&buf, 800, null);
    try quiet(&buf, 800, null);
    var a: std.ArrayList(Event) = .empty;
    defer a.deinit(gpa);
    try run(buf.items, test_rate / 10, &a);
    try expectEvents(&.{ .{ .kind = .speech_started, .ms = 0 }, .{ .kind = .speech_stopped, .ms = 1300 } }, a.items);
    var b: std.ArrayList(Event) = .empty;
    defer b.deinit(gpa);
    try run(buf.items, 37, &b);
    try std.testing.expectEqualSlices(Event, a.items, b.items);
}

test "pause_ms reports each quiet run once, at where it began; quietSince follows it" {
    const gpa = std.testing.allocator;
    var buf: std.ArrayList(i16) = .empty;
    defer buf.deinit(gpa);
    try tone(&buf, 600, null);
    try quiet(&buf, 200, null); // a pause, then more speech
    try tone(&buf, 400, null);
    try quiet(&buf, 800, null);
    var v: Vad = .init(.{ .pause_ms = 150 }, test_rate);
    var ev: std.ArrayList(Event) = .empty;
    defer ev.deinit(gpa);
    try v.feed(gpa, buf.items[0 .. 700 * test_rate / 1000], &ev);
    try std.testing.expect(@max(v.quietSince().?, 600) - @min(v.quietSince().?, 600) <= 20);
    try v.feed(gpa, buf.items[700 * test_rate / 1000 ..], &ev);
    try std.testing.expectEqual(@as(?u64, null), v.quietSince());
    try expectEvents(&.{ .{ .kind = .speech_started, .ms = 0 }, .{ .kind = .speech_paused, .ms = 600 }, .{ .kind = .speech_paused, .ms = 1200 }, .{ .kind = .speech_stopped, .ms = 1700 } }, ev.items);
    try std.testing.expectEqual(@as(u64, 500), v.silenceMs());
}

test "a click shorter than min_speech_ms is ignored" {
    const gpa = std.testing.allocator;
    var buf: std.ArrayList(i16) = .empty;
    defer buf.deinit(gpa);
    try quiet(&buf, 200, null);
    try tone(&buf, 40, null);
    try quiet(&buf, 800, null);
    var ev: std.ArrayList(Event) = .empty;
    defer ev.deinit(gpa);
    try run(buf.items, test_rate / 10, &ev);
    try std.testing.expectEqual(@as(usize, 0), ev.items.len);
}

test "background noise does not trigger, speech over it does" {
    const gpa = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    var buf: std.ArrayList(i16) = .empty;
    defer buf.deinit(gpa);
    try quiet(&buf, 2000, r);
    var ev: std.ArrayList(Event) = .empty;
    defer ev.deinit(gpa);
    try run(buf.items, test_rate / 10, &ev);
    try std.testing.expectEqual(@as(usize, 0), ev.items.len);
    try tone(&buf, 600, r);
    try quiet(&buf, 1000, r);
    try run(buf.items, test_rate / 10, &ev);
    try expectEvents(&.{ .{ .kind = .speech_started, .ms = 1700 }, .{ .kind = .speech_stopped, .ms = 3100 } }, ev.items);
}

test "two turns apart are two pairs; a short pause does not split a turn" {
    const gpa = std.testing.allocator;
    var buf: std.ArrayList(i16) = .empty;
    defer buf.deinit(gpa);
    try tone(&buf, 500, null);
    try quiet(&buf, 1000, null);
    try tone(&buf, 400, null);
    try quiet(&buf, 300, null); // under silence_duration_ms
    try tone(&buf, 400, null);
    try quiet(&buf, 800, null);
    var ev: std.ArrayList(Event) = .empty;
    defer ev.deinit(gpa);
    try run(buf.items, test_rate / 10, &ev);
    try expectEvents(&.{
        .{ .kind = .speech_started, .ms = 0 },
        .{ .kind = .speech_stopped, .ms = 1000 },
        .{ .kind = .speech_started, .ms = 1200 },
        .{ .kind = .speech_stopped, .ms = 3100 },
    }, ev.items);
}

test "the wait restarts on speech, never adds up, and a blip in it is not speech" {
    const gpa = std.testing.allocator;
    var buf: std.ArrayList(i16) = .empty;
    defer buf.deinit(gpa);
    // Three pauses of 400 ms under the 500 ms wait: one turn, ending 500 ms after the last word.
    try tone(&buf, 400, null);
    for (0..3) |_| {
        try quiet(&buf, 400, null);
        try tone(&buf, 300, null);
    }
    try quiet(&buf, 700, null);
    var a: std.ArrayList(Event) = .empty;
    defer a.deinit(gpa);
    try run(buf.items, test_rate / 10, &a);
    try expectEvents(&.{ .{ .kind = .speech_started, .ms = 0 }, .{ .kind = .speech_stopped, .ms = 400 + 3 * 700 + 500 } }, a.items);
    // A 40 ms click 300 ms into the quiet does not restart it: the turn ends 500 ms after the speech, not the click.
    buf.clearRetainingCapacity();
    try tone(&buf, 600, null);
    try quiet(&buf, 300, null);
    try tone(&buf, 40, null);
    try quiet(&buf, 800, null);
    var b: std.ArrayList(Event) = .empty;
    defer b.deinit(gpa);
    try run(buf.items, test_rate / 10, &b);
    try expectEvents(&.{ .{ .kind = .speech_started, .ms = 0 }, .{ .kind = .speech_stopped, .ms = 600 + 500 } }, b.items);
}

test "reset returns to silence and keeps the position" {
    const gpa = std.testing.allocator;
    var buf: std.ArrayList(i16) = .empty;
    defer buf.deinit(gpa);
    try tone(&buf, 200, null);
    var v: Vad = .init(.{}, test_rate);
    var ev: std.ArrayList(Event) = .empty;
    defer ev.deinit(gpa);
    try v.feed(gpa, buf.items, &ev);
    try std.testing.expect(v.speaking());
    try std.testing.expectEqual(@as(usize, 1), ev.items.len);
    v.reset();
    try std.testing.expect(!v.speaking());
    try std.testing.expectEqual(@as(u64, 200), v.position());
    const zeros: [test_rate]i16 = @splat(0);
    try v.feed(gpa, &zeros, &ev); // a second of silence: no speech_stopped after a reset
    try std.testing.expectEqual(@as(usize, 1), ev.items.len);
    try std.testing.expectEqual(@as(u64, 1200), v.position());
}
