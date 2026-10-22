//! The stage timings of a voice response: when the transcript, the first chat text, the first text handed to speech,
//! the first speech samples and the first audible audio came, relative to the end of the spoken turn.

const std = @import("std");

/// When each stage of a response happened (awake-clock ns; 0: not reached), sent as `localrouter.response.timing`.
pub const Marks = struct {
    t0: i96 = 0, // the spoken turn's end (VAD), else the response's start
    from_turn: bool = false,
    stt_start: i96 = 0,
    stt_end: i96 = 0,
    stt_worker_ms: u64 = 0,
    stt_draft: bool = false, // the transcript was the draft made at the pause (its stt marks may precede t0)
    early_end: bool = false, // the turn ended on that draft's sentence end, before the VAD's silence_duration_ms
    speculative: bool = false, // the reply was made during the pause, held, and sent when the turn ended
    speech_end: i96 = 0, // the speaker fell quiet (before t0 by the wait, silence_duration_ms)
    chat_start: i96 = 0,
    first_token: i96 = 0, // the chat stream's first text
    first_chunk: i96 = 0, // the first text handed to speech (after the lead hold and the chunker)
    tts_start: i96 = 0,
    tts_first_pcm: i96 = 0, // the speech model's first samples
    first_audio: i96 = 0, // the first audio delta sent
    first_voiced: i96 = 0, // the first delta with a sample above the silence threshold
    voiced_audio_ms: ?u64 = null, // where that sample is in the reply's audio (leading silence before it)
};

/// The index of the first sample louder than about -42 dBFS, or null.
pub fn firstLoud(samples: []const i16) ?usize {
    for (samples, 0..) |x, i| {
        if (@abs(x) > 250) return i;
    }
    return null;
}

pub const Ms = struct { speech_end: ?i64, stt_start: ?i64, stt_end: ?i64, chat_start: ?i64, first_token: ?i64, first_chunk: ?i64, tts_start: ?i64, tts_first_pcm: ?i64, first_audio: ?i64, first_voiced: ?i64 };

/// The `localrouter.response.timing` event of response `id`: each stage in ms from the turn's end (or the response's start).
pub fn event(id: []const u8, m: Marks) struct { response_id: []const u8, from: []const u8, ms: Ms, stt_worker_ms: u64, stt_draft: bool, early_end: bool, speculative: bool, voiced_audio_ms: ?u64 } {
    var ms: Ms = undefined;
    inline for (@typeInfo(Ms).@"struct".field_names) |name| {
        const t = @field(m, name);
        @field(ms, name) = if (t == 0) null else @as(i64, @intCast(@divTrunc(t - m.t0, std.time.ns_per_ms)));
    }
    return .{ .response_id = id, .from = if (m.from_turn) "speech_stopped" else "response_start", .ms = ms, .stt_worker_ms = m.stt_worker_ms, .stt_draft = m.stt_draft, .early_end = m.early_end, .speculative = m.speculative, .voiced_audio_ms = m.voiced_audio_ms };
}

test "stage times are ms from the turn's end; stages not reached are null" {
    const ms = std.time.ns_per_ms;
    const e = event("resp_1", .{ .t0 = 1000 * ms, .from_turn = true, .stt_start = 1010 * ms, .stt_end = 1300 * ms, .chat_start = 1301 * ms, .first_token = 1450 * ms, .first_audio = 1700 * ms, .voiced_audio_ms = 40 });
    try std.testing.expectEqualStrings("speech_stopped", e.from);
    try std.testing.expectEqual(@as(?i64, 300), e.ms.stt_end);
    try std.testing.expectEqual(@as(?i64, 450), e.ms.first_token);
    try std.testing.expectEqual(@as(?i64, null), e.ms.first_voiced);
    try std.testing.expectEqual(@as(?usize, 3), firstLoud(&.{ 0, -12, 249, -700, 900 }));
}
