//! What the pipeline may run, decided before anything is allocated: every contract the coupled video / audio path rests
//! on, with its status, and the use it is admitted for. Production (the tool) admits only a path whose every contract is
//! qualified on the target; diagnostic (the check gates) runs anything that is merely unqualified and labels its outputs,
//! and still refuses what is unresolved (a value nobody can compute yet). The components keep their own lists
//! (audio/vae.zig `admitOp`, the decoder's and vocoder's `admit`); this file asks them and adds central's own.

const std = @import("std");
const rope = @import("rope.zig");
const vae_launch = @import("video_vae/launch.zig");
const audio = @import("audio/vae.zig");
const vocoder = @import("audio/vocoder.zig");
const audio_launch = @import("audio/launch.zig");
const shapes = @import("shapes.zig");

pub const Use = enum { diagnostic, production };

pub const Status = enum {
    /// Matches the native reference on the target; nothing to wait for.
    qualified,
    /// Implemented as the reference computes it; the GPU gate comparing bits has not passed yet.
    gpu_qualification_pending,
    /// Implemented, but a value or ABI is known to differ from the reference (named in `why`): diagnostic only.
    provisional,
    /// Cannot be computed yet: refused for every use.
    unresolved,
};

pub const Contract = struct { name: []const u8, status: Status, why: []const u8 };

/// The pipeline's contracts, in pipeline order. Each entry changes only with the receipt that changes it.
pub fn contracts(out: *[9]Contract) []const Contract {
    var n: usize = 0;
    out[n] = .{ .name = "text: Qwen2.5-VL + CLIP conditioning", .status = .gpu_qualification_pending, .why = "k6-components text gate" };
    n += 1;
    out[n] = .{ .name = "noise: torch CUDA randn (Philox) for video seed s, audio seed s + 1", .status = .gpu_qualification_pending, .why = "k6-components noise gate" };
    n += 1;
    out[n] = .{
        .name = "rope: get_freqs table",
        .status = if (rope.provisional) .provisional else .gpu_qualification_pending,
        .why = if (rope.provisional) "upstream evaluates SLEEF expf on the aarch64 CPU; the frozen native capture is not committed" else "k6-components rope gate",
    };
    n += 1;
    out[n] = .{ .name = "piflow: 10-step multimodal DX policy", .status = .gpu_qualification_pending, .why = "k6-components piflow gate" };
    n += 1;
    out[n] = .{ .name = "dit: PR #5 fused DiT", .status = .gpu_qualification_pending, .why = "D1's DiT gate with the chosen fp32_linear" };
    n += 1;
    out[n] = .{
        .name = "video vae: PR #4 latent input",
        .status = if (latentInF32()) .gpu_qualification_pending else .provisional,
        .why = if (latentInF32()) "D2's decode gate" else "PR #4 latent_in reads a bf16 latent; upstream divides the f32 PiFlow state and casts once to fp16",
    };
    n += 1;
    // PR #6's native load path: the decoder's stored weights are prepared on the host at load (admitted as stored, then
    // as prepared), the vocoder's stored folded weights are used as they are
    out[n] = audioContract("audio: PR #6 audio VAE decoder", audio.Decoder.init(1, shapes.audio_rows), .stored_checkpoint);
    n += 1;
    out[n] = audioContract("audio: PR #6 vocoder", vocoder.Vocoder.init(1, 2 * shapes.audio_rows), .checkpoint_effective);
    n += 1;
    out[n] = .{
        .name = "audio: PR #6 end-to-end decode and vocode entry points",
        .status = if (audioEntryPoints()) .gpu_qualification_pending else .unresolved,
        .why = if (audioEntryPoints()) "D3's end-to-end audio gate with the chosen decode / vocode policies" else "PR #6's audio/launch.zig declares no decode / vocode",
    };
    n += 1;
    return out[0..n];
}

/// PR #4's latent input ABI: the corrected head declares `latent_in_dtype = .f32` (the request relayed to D2).
fn latentInF32() bool {
    if (!@hasDecl(vae_launch, "latent_in_dtype")) return false;
    return std.mem.eql(u8, @tagName(vae_launch.latent_in_dtype), "f32");
}

/// PR #6's whole-pass API (A2): `audio/launch.zig` `decode` and `vocode`.
fn audioEntryPoints() bool {
    return @hasDecl(audio_launch, "decode") and @hasDecl(audio_launch, "vocode");
}

fn audioContract(name: []const u8, part: anytype, state: audio.WeightState) Contract {
    const p = part catch return .{ .name = name, .status = .unresolved, .why = "invalid geometry" };
    p.admit(.diagnostic, state) catch |e| return .{ .name = name, .status = .unresolved, .why = @errorName(e) };
    p.admit(.production, state) catch |e| return .{ .name = name, .status = .gpu_qualification_pending, .why = @errorName(e) };
    return .{ .name = name, .status = .qualified, .why = "" };
}

/// Refuses `use` unless every contract admits it; logs each blocker. Returns how many contracts are not qualified (the
/// diagnostic run's label).
pub fn admit(use: Use) error{ UnresolvedContract, NotQualified }!usize {
    var buf: [9]Contract = undefined;
    const cs = contracts(&buf);
    for (cs) |c| {
        if (c.status == .qualified) continue;
        if (use == .production or c.status == .unresolved)
            std.log.warn("kandinsky6: {s} is {t} ({s})", .{ c.name, c.status, c.why });
    }
    return decide(cs, use);
}

/// `admit`'s decision for a contract list, without the log.
pub fn decide(cs: []const Contract, use: Use) error{ UnresolvedContract, NotQualified }!usize {
    var open: usize = 0;
    var unresolved = false;
    for (cs) |c| {
        if (c.status == .qualified) continue;
        open += 1;
        if (c.status == .unresolved) unresolved = true;
    }
    if (unresolved) return error.UnresolvedContract;
    if (use == .production and open > 0) return error.NotQualified;
    return open;
}

test "production refuses until every contract is qualified" {
    var buf: [9]Contract = undefined;
    const cs = contracts(&buf);
    try std.testing.expectEqual(@as(usize, 9), cs.len);
    var all = true;
    for (cs) |c| all = all and c.status == .qualified;
    if (all) {
        try std.testing.expectEqual(@as(usize, 0), try decide(cs, .production));
    } else {
        if (decide(cs, .production)) |_| return error.TestUnexpectedResult else |_| {}
    }
    // rope's status follows its own flag, so committing the frozen table is the receipt that moves it
    try std.testing.expectEqual(rope.provisional, cs[2].status == .provisional);
}

test "diagnostic runs what is unqualified and refuses what is unresolved" {
    const pending: Contract = .{ .name = "a", .status = .gpu_qualification_pending, .why = "" };
    const ok: Contract = .{ .name = "b", .status = .qualified, .why = "" };
    const missing: Contract = .{ .name = "c", .status = .unresolved, .why = "" };
    try std.testing.expectEqual(@as(usize, 1), try decide(&.{ ok, pending }, .diagnostic));
    try std.testing.expectError(error.NotQualified, decide(&.{ ok, pending }, .production));
    try std.testing.expectError(error.UnresolvedContract, decide(&.{ ok, missing }, .diagnostic));
    try std.testing.expectEqual(@as(usize, 0), try decide(&.{ok}, .production));
}
