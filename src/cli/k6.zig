//! The Kandinsky-6 bit gates, native side: each writes a capture in the twin's format for
//! `python -I -m stk_twin.kandinsky6.compare ORACLE NATIVE` (tools/twin/stk_twin/kandinsky6/README.md), with the oracle's
//! op and role names, `stage` notes, and a `request` note labelled `qualification: unqualified` (a GPU gate result is the
//! compare report, never this run).
//!
//!   localrouter check k6-components --model SNAPSHOT --options OPTIONS.json --out DIR --stages S[,S...]
//!       [--seed N] [--prompt-file F] [--oracle ORACLE_DIR]
//!     noise   k6.noise.*                       (--seed)
//!     rope    k6.freqs.*, k6.rope.*
//!     clip    k6.clip.pooled per prompt        (--prompt-file; loads the text encoders)
//!     text    k6.qwen.embeds per prompt        (--prompt-file; loads the text encoders)
//!     piflow  k6.piflow.step0..9, each step from the oracle's own states and predictions (--oracle)
//!     vae     k6.vae.decode from the oracle's latent (--oracle; loads the video VAE), and k6.aux.vae.decoder_in, under
//!             OPTIONS' vae_free_bytes (the oracle's note_free_bytes replays its plan)
//!             [--vae-trace first-tile | pass-N]: diagnostic k6.aux.vae.trace.* (QUALIFICATION.md's decoder checkpoints in
//!             decoder pass N, first-tile = 0; the first blends and tile assembly), also with k6-e2e
//!     vae_gn  diagnostic: the GroupNorm fixture (k6.aux.vae.gn.fixture of the reference isolate, --oracle): every
//!             statistics candidate on this device against torch's bits (note gn_fixture); loads the VAE kernels only
//!     vae_conv diagnostic: the convolution fixture (k6.aux.vae.conv.fixture.* of the reference isolate, --oracle;
//!             [--conv-spec DIR] adds the measured orders of stk_twin.kandinsky6.vae_conv_oracle as candidates):
//!             every candidate K schedule and the production kernel on this device against torch's bits (note conv_fixture)
//!     vae_sdpa diagnostic: the attention-core fixture (k6.aux.vae.attn.sdpa[.<probe>] of the attention isolate or the
//!             SDPA probe, --oracle): k6v_sdpa_memeff under every variant against torch's bits (note sdpa_fixture)
//!     vae_blend diagnostic: the tile-blend fixture (k6.aux.vae.blend.fixture.* of stk_twin.kandinsky6.vae_blend_isolate,
//!             --oracle): the decode's k6v_blend launch against the upstream methods' bits (note blend_fixture)
//!     audio   k6.audio.* chained natively from the oracle's k6.audio.latent_in (--oracle; loads the audio VAE, vocoder)
//!             [--audio-trace STOP]: diagnostic k6.aux.audio.decoder.* hooks, stopped after STOP's op (`audio_stops`)
//!     vocoder diagnostic: the vocoder alone on the oracle's k6.audio.mel, every stage as k6.aux.audio.vocoder.*
//!   localrouter check k6-e2e --model SNAPSHOT --options OPTIONS.json --out DIR --seed N --prompt-file F [--e2e-prompt K]
//!     the whole pipeline for line K: k6.dit.step0..9, k6.vae.decode, k6.aux.vae.decoder_in, k6.audio.*, k6.e2e.frames
//!
//! OPTIONS.json is the tool's `options` object (`kandinsky6.pipeline.parseOptions`: every arithmetic and audio policy
//! field required). Both run admitted for diagnostic use: refused while any contract is unresolved. The output
//! directory must not exist. Exit 0 when every stage is ok, 1 otherwise, 2 on usage.

const std = @import("std");
const k6 = @import("kandinsky6");
const qi = @import("qwen_image");
const P = k6.pipeline;
const Rec = k6.rec.Rec;

const Stage = enum { noise, rope, clip, text, piflow, vae, vae_gn, vae_conv, vae_sdpa, vae_blend, audio, vocoder, e2e };

/// The op prefixes each stage records (compare.py STAGE_OPS, plus `audio` for the replayed chain).
fn prefixes(s: Stage) []const []const u8 {
    return switch (s) {
        .noise => &.{"k6.noise."},
        .rope => &.{ "k6.freqs.", "k6.rope." },
        .clip => &.{"k6.clip."},
        .text => &.{"k6.qwen."},
        .piflow => &.{"k6.piflow."},
        .vae => &.{"k6.vae."},
        .vae_gn => &.{"k6.aux.vae.gn."},
        .vae_conv => &.{"k6.aux.vae.conv."},
        .vae_sdpa => &.{"k6.aux.vae.attn."},
        .vae_blend => &.{"k6.aux.vae.blend."},
        .audio => &.{"k6.audio."},
        .vocoder => &.{"k6.aux.audio.vocoder."},
        .e2e => &.{ "k6.dit.", "k6.vae.", "k6.audio.", "k6.e2e." },
    };
}

const Args = struct {
    model: ?[]const u8 = null,
    options: ?[]const u8 = null,
    out: ?[]const u8 = null,
    stages: ?[]const u8 = null,
    seed: ?u64 = null,
    prompt_file: ?[]const u8 = null,
    e2e_prompt: u32 = 0,
    dit_trace: bool = false,
    dit_trace_text: bool = false,
    dit_trace_block: bool = false,
    dit_trace_audio: bool = false,
    /// --audio-trace: the decoder hook name after which the diagnostic stops (`audio_stops`).
    audio_stop: ?[]const u8 = null,
    oracle: ?[]const u8 = null,
    /// --conv-spec DIR: measured convolution orders for the vae_conv stage (stk_twin.kandinsky6.vae_conv_oracle).
    conv_spec: ?[]const u8 = null,
    /// --vae-trace first-tile | pass-N: the decode's k6.aux.vae.trace.* checkpoints (`Hooks.vae_trace`), the decoder's
    /// in pass N (first-tile: 0); null without the flag.
    vae_trace: ?u32 = null,
};

fn parse(args: []const []const u8) !Args {
    var a: Args = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--model")) a.model = v else if (std.mem.eql(u8, k, "--options")) a.options = v else if (std.mem.eql(u8, k, "--out")) a.out = v else if (std.mem.eql(u8, k, "--stages")) a.stages = v else if (std.mem.eql(u8, k, "--seed")) a.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--prompt-file")) a.prompt_file = v else if (std.mem.eql(u8, k, "--e2e-prompt")) a.e2e_prompt = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--oracle")) a.oracle = v else if (std.mem.eql(u8, k, "--dit-trace") and std.mem.eql(u8, v, "init")) a.dit_trace = true else if (std.mem.eql(u8, k, "--dit-trace") and std.mem.eql(u8, v, "text")) a.dit_trace_text = true else if (std.mem.eql(u8, k, "--dit-trace") and std.mem.eql(u8, v, "block")) a.dit_trace_block = true else if (std.mem.eql(u8, k, "--dit-trace") and std.mem.eql(u8, v, "audio")) a.dit_trace_audio = true else if (std.mem.eql(u8, k, "--audio-trace")) a.audio_stop = try onceStop(a.audio_stop, v) else if (std.mem.eql(u8, k, "--vae-trace")) a.vae_trace = try tracePass(v) else if (std.mem.eql(u8, k, "--conv-spec")) a.conv_spec = v else return error.Usage;
    }
    if (a.model == null or a.options == null or a.out == null) return error.Usage;
    return a;
}

/// `--audio-trace` values and the decoder hook (launch.zig decodeWith) each stops after. The twin's
/// stk_twin.kandinsky6.audio_trace.STOPS is the same table; both sides record every hook up to and including the stop.
pub const audio_stops = [_]struct { arg: []const u8, stop: []const u8 }{
    .{ .arg = "decoder", .stop = "first_conv" },
    .{ .arg = "residual", .stop = "first_silu" },
    .{ .arg = "residual-conv", .stop = "first_res_conv1" },
    .{ .arg = "residual-block", .stop = "first_residual" },
    .{ .arg = "attention", .stop = "attention_residual" },
    .{ .arg = "mid2", .stop = "mid2_residual" },
    .{ .arg = "up2-first", .stop = "up2_first_residual" },
    .{ .arg = "up2-second", .stop = "up2_second_residual" },
    .{ .arg = "up2-third", .stop = "up2_third_residual" },
    .{ .arg = "up1-first", .stop = "up1_first_residual" },
    .{ .arg = "up1-second", .stop = "up1_second_residual" },
    .{ .arg = "up1-third", .stop = "up1_third_residual" },
    .{ .arg = "upsample", .stop = "up1_upsample_conv" },
    .{ .arg = "up0-first", .stop = "up0_first_residual" },
    .{ .arg = "up0-second", .stop = "up0_second_residual" },
    .{ .arg = "up0-third", .stop = "up0_third_residual" },
    .{ .arg = "mel", .stop = "mel" },
};

fn audioStop(arg: []const u8) ?[]const u8 {
    for (audio_stops) |e| if (std.mem.eql(u8, e.arg, arg)) return e.stop;
    return null;
}

/// One `--audio-trace`, naming a known stop.
/// --vae-trace VALUE: "first-tile" (decoder pass 0) or "pass-N".
fn tracePass(v: []const u8) error{Usage}!u32 {
    if (std.mem.eql(u8, v, "first-tile")) return 0;
    if (!std.mem.startsWith(u8, v, "pass-")) return error.Usage;
    return std.fmt.parseInt(u32, v["pass-".len..], 10) catch error.Usage;
}

test "--vae-trace: first-tile is pass 0, pass-N any pass, anything else a usage error" {
    try std.testing.expectEqual(@as(u32, 0), try tracePass("first-tile"));
    try std.testing.expectEqual(@as(u32, 1), try tracePass("pass-1"));
    try std.testing.expectEqual(@as(u32, 111), try tracePass("pass-111"));
    try std.testing.expectError(error.Usage, tracePass("pass-"));
    try std.testing.expectError(error.Usage, tracePass("pass-x"));
    try std.testing.expectError(error.Usage, tracePass("pass--1"));
    try std.testing.expectError(error.Usage, tracePass("tile-1"));
}

fn onceStop(prev: ?[]const u8, arg: []const u8) error{Usage}![]const u8 {
    if (prev != null) return error.Usage;
    return audioStop(arg) orelse error.Usage;
}

/// One prompt a line; blank lines and `#` comments skipped (the oracle's `--prompt-file`).
fn prompts(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]const []const u8 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    var out: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (t.len == 0 or t[0] == '#') continue;
        try out.append(a, t);
    }
    if (out.items.len == 0) return error.NoPrompts;
    return out.items;
}

const usage =
    \\usage: localrouter check k6-components --model SNAPSHOT --options OPTIONS.json --out DIR --stages S[,S...]
    \\           [--seed N] [--prompt-file F] [--oracle ORACLE_DIR]   (stages: noise rope clip text piflow vae vae_gn vae_conv vae_sdpa vae_blend audio vocoder)
    \\           [--audio-trace decoder|residual|residual-conv|residual-block|attention|mid2|up2-first|up2-second|
    \\                          up2-third|up1-first|up1-second|up1-third|upsample|up0-first|up0-second|up0-third|mel]
    \\                          (audio stage: diagnostic stop after that decoder op; no audio verdict)
    \\       localrouter check k6-e2e --model SNAPSHOT --options OPTIONS.json --out DIR --seed N --prompt-file F [--e2e-prompt K]
    \\           [--dit-trace init]   (diagnostic stop after the first text block; no generation verdict)
    \\           [--dit-trace text]   (diagnostic stop after both text streams and input embeddings)
    \\           [--dit-trace block]  (diagnostic stop after the first visual self-attention)
    \\
;

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8, e2e: bool) !u8 {
    const a = parse(args) catch {
        std.debug.print(usage, .{});
        return 2;
    };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();

    var stages: std.ArrayList(Stage) = .empty;
    var only: std.ArrayList([]const u8) = .empty;
    if (e2e) {
        try stages.append(ar, .e2e);
    } else {
        var it = std.mem.splitScalar(u8, a.stages orelse {
            std.debug.print(usage, .{});
            return 2;
        }, ',');
        while (it.next()) |name| {
            const s = std.meta.stringToEnum(Stage, name) orelse {
                std.debug.print("k6-components: unknown stage {s}\n{s}", .{ name, usage });
                return 2;
            };
            if (s == .e2e) {
                std.debug.print("k6-components: e2e is `check k6-e2e`\n", .{});
                return 2;
            }
            try stages.append(ar, s);
        }
    }
    var parts: P.Parts = if (e2e) .all else .none;
    for (stages.items) |s| {
        try only.appendSlice(ar, prefixes(s));
        switch (s) {
            .clip, .text => parts.text = true,
            .vae => parts.video = true,
            .audio, .vocoder => parts.audio = true,
            .noise, .rope, .piflow, .vae_gn, .vae_conv, .vae_sdpa, .vae_blend, .e2e => {},
        }
        const missing: ?[]const u8 = switch (s) {
            .noise => if (a.seed == null) "--seed" else null,
            .clip, .text => if (a.prompt_file == null) "--prompt-file" else null,
            .piflow, .vae, .vae_gn, .vae_conv, .vae_sdpa, .vae_blend, .audio, .vocoder => if (a.oracle == null) "--oracle" else null,
            .e2e => if (a.seed == null) "--seed" else if (a.prompt_file == null) "--prompt-file" else null,
            .rope => null,
        };
        if (missing) |m| {
            std.debug.print("k6: stage {t} needs {s}\n", .{ s, m });
            return 2;
        }
    }
    if (a.conv_spec != null and std.mem.indexOfScalar(Stage, stages.items, .vae_conv) == null) {
        std.debug.print("k6: --conv-spec is for stage vae_conv\n", .{});
        return 2;
    }
    const lines: []const []const u8 = if (a.prompt_file) |f| try prompts(io, ar, f) else &.{};
    if (e2e and a.e2e_prompt >= lines.len) {
        std.debug.print("k6-e2e: {s} has {d} prompts; --e2e-prompt {d}\n", .{ a.prompt_file.?, lines.len, a.e2e_prompt });
        return 2;
    }

    const opts_text = try std.Io.Dir.cwd().readFileAlloc(io, a.options.?, ar, .limited(1 << 20));
    const opts_json = try std.json.parseFromSliceLeaky(std.json.Value, ar, opts_text, .{});
    var opts = P.parseOptions(opts_json, .diagnostic) catch |err| {
        std.debug.print("k6: {s}: {t} (see kandinsky6_tool.zig for every required option)\n", .{ a.options.?, err });
        return 2;
    };
    opts.parts = parts;

    if ((a.dit_trace or a.dit_trace_text or a.dit_trace_block or a.dit_trace_audio) and !e2e) return error.Usage;

    if (a.audio_stop != null and (e2e or !parts.audio)) return error.Usage;
    if (a.vae_trace != null and !parts.video) return error.Usage;
    opts.hooks.vae_trace = a.vae_trace != null;
    opts.hooks.vae_trace_pass = a.vae_trace orelse 0;

    var rec = try Rec.create(gpa, io, a.out.?);
    defer rec.deinit();
    if (parts.text) try only.append(ar, "k6.aux.text.");
    if (parts.video) try only.append(ar, "k6.aux.vae.");
    if (a.dit_trace) try only.append(ar, "k6.aux.dit.init.");
    if (a.dit_trace_text) try only.append(ar, "k6.aux.dit.text.");
    if (a.dit_trace_block) try only.append(ar, "k6.aux.dit.block.");
    if (a.dit_trace_audio) try only.append(ar, "k6.aux.dit.audio.");
    if (a.audio_stop != null) try only.append(ar, "k6.aux.audio.decoder.");
    rec.only = only.items;
    opts.hooks.rec = &rec;
    var audio_trace: AudioTrace = .{ .rec = &rec, .stop = "" };
    if (a.audio_stop) |stop| {
        audio_trace.stop = stop;
        opts.hooks.audio_decode = .{ .ctx = &audio_trace, .op = traceAudioDecoder };
    }
    if (std.mem.indexOfScalar(Stage, stages.items, .vocoder) != null) opts.hooks.audio_vocode = .{ .ctx = &rec, .op = traceAudioVocoder };
    if (a.dit_trace) opts.hooks.dit = .{ .ctx = &rec, .op = traceDitInit, .trace_attention = true };
    if (a.dit_trace_text) opts.hooks.dit = .{ .ctx = &rec, .op = traceDitText };
    if (a.dit_trace_block) opts.hooks.dit = .{ .ctx = &rec, .op = traceDitBlock, .trace_visual_attention = true };
    if (a.dit_trace_audio) opts.hooks.dit = .{ .ctx = &rec, .op = traceDitAudio };
    if (parts.text) opts.hooks.text = .{
        .ctx = &rec,
        .after = struct {
            fn after(ctx: *anyopaque, name: []const u8, ptr: u64, bytes: u64) anyerror!void {
                // First-layer ops plus every Qwen layer boundary locate parity failures without changing arithmetic.
                const qwen_boundary = std.mem.startsWith(u8, name, "layers.") and std.mem.endsWith(u8, name, ".mlp_residual");
                if (!std.mem.eql(u8, name, "embed_tokens") and !std.mem.eql(u8, name, "embeddings") and
                    !std.mem.startsWith(u8, name, "layers.0.") and !std.mem.startsWith(u8, name, "layers.1.") and !std.mem.startsWith(u8, name, "encoder.layers.0.") and !qwen_boundary) return;
                const r: *Rec = @ptrCast(@alignCast(ctx));
                var nb: [160]u8 = undefined;
                try r.op(try std.fmt.bufPrint(&nb, "k6.aux.text.{s}", .{name}), "diagnostic", .{}, &.{}, &.{.{
                    .role = "raw",
                    .dtype = .uint8,
                    .shape = &.{bytes},
                    .src = .{ .device = ptr },
                }});
            }
        }.after,
    };
    var buf: [k6.admission.max_contracts]k6.admission.Contract = undefined;
    try rec.note("request", .{
        .qualification = "unqualified",
        .argv = args,
        .stages = stages.items,
        .seed = a.seed,
        .prompts = lines,
        .prompt_file = a.prompt_file,
        .e2e_prompt = a.e2e_prompt,
        .model = a.model.?,
        .options = opts_json,
        .contracts = k6.admission.contracts(&buf),
    });

    const p = P.Pipeline.create(gpa, io, a.model.?, opts) catch |err| {
        for (stages.items) |s| try rec.note("stage", .{ .stage = s, .status = try std.fmt.allocPrint(ar, "failed: load: {t}", .{err}) });
        std.debug.print("k6: load failed: {t}\n", .{err});
        return 1;
    };
    defer p.destroy();

    var oracle: ?qi.capture.Capture = if (a.oracle) |dir| try .open(gpa, io, dir) else null;
    defer if (oracle) |*c| c.close();
    var conv_spec: ?qi.capture.Capture = if (a.conv_spec) |dir| try .open(gpa, io, dir) else null;
    defer if (conv_spec) |*cs| cs.close();

    var failed = false;
    for (stages.items) |s| {
        const t0 = std.Io.Clock.awake.now(io);
        const result: anyerror!void = switch (s) {
            .rope => {}, // recorded while the pipeline built its tables
            .noise => p.drawNoise(a.seed.?),
            .clip, .text => for (lines, 0..) |line, i| {
                _ = p.text(line, @intCast(i)) catch |err| break err;
            },
            .piflow => p.replayPiflow(&oracle.?),
            .vae => p.replayVideoDecode(&oracle.?),
            .vae_gn => p.replayGroupNormFixture(&oracle.?),
            .vae_conv => p.replayConvFixture(&oracle.?, if (conv_spec) |*cs| cs else null),
            .vae_sdpa => p.replaySdpaFixture(&oracle.?),
            .vae_blend => p.replayBlendFixture(&oracle.?),
            .audio => p.replayAudio(&oracle.?),
            .vocoder => p.replayVocoder(&oracle.?),
            .e2e => e2eRun(p, gpa, lines[a.e2e_prompt], a.seed.?, a.e2e_prompt),
        };
        const secs = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds())) / 1000.0;
        if (result) |_| {
            try rec.note("stage", .{ .stage = s, .status = "ok", .seconds = secs });
        } else |err| {
            failed = true;
            try rec.note("stage", .{ .stage = s, .status = try std.fmt.allocPrint(ar, "failed: {t}", .{err}), .seconds = secs });
            std.debug.print("k6: stage {t} failed: {t}\n", .{ s, err });
        }
    }
    std.debug.print("k6: capture {s} (unqualified until compare passes on the target)\n", .{a.out.?});
    return if (failed) 1 else 0;
}

/// Diagnostic capture only: stop after the first video text block, preserving the failed stage verdict.
fn traceDitInit(ctx: *anyopaque, name: []const u8, out: k6.dit.layout.View) anyerror!void {
    const r: *Rec = @ptrCast(@alignCast(ctx));
    if (!std.mem.startsWith(u8, name, "video_")) return;
    if (out.ld != out.cols) return error.NonContiguousDiagnostic;
    var nb: [256]u8 = undefined;
    try r.op(try std.fmt.bufPrint(&nb, "k6.aux.dit.init.{s}", .{name}), "diagnostic", .{}, &.{}, &.{.{
        .role = "raw",
        .dtype = .uint8,
        .shape = &.{out.bytes()},
        .src = .{ .device = out.ptr },
    }});
    if (std.mem.eql(u8, name, "video_text_transformer_blocks.0")) {
        try r.note("diagnostic_stop", .{ .after = name, .qualification = "unqualified" });
        return error.DiagnosticTraceComplete;
    }
}

/// Capture text blocks and input embeddings without tracing attention internals or changing arithmetic.
fn traceDitText(ctx: *anyopaque, name: []const u8, out: k6.dit.layout.View) anyerror!void {
    const r: *Rec = @ptrCast(@alignCast(ctx));
    if (out.ld != out.cols) return error.NonContiguousDiagnostic;
    var nb: [256]u8 = undefined;
    try r.op(try std.fmt.bufPrint(&nb, "k6.aux.dit.text.{s}", .{name}), "diagnostic", .{}, &.{}, &.{.{
        .role = "raw",
        .dtype = .uint8,
        .shape = &.{out.bytes()},
        .src = .{ .device = out.ptr },
    }});
    if (std.mem.eql(u8, name, "audio_embeddings")) {
        try r.note("diagnostic_stop", .{ .after = name, .qualification = "unqualified" });
        return error.DiagnosticTraceComplete;
    }
}

/// First visual block only; unchanged inputs and arithmetic, explicit unqualified stop.
fn traceDitBlock(ctx: *anyopaque, name: []const u8, out: k6.dit.layout.View) anyerror!void {
    const stop = "visual_transformer_blocks.0.audio_dec_block.feed_forward";
    if (!std.mem.eql(u8, name, "visual_embeddings") and !std.mem.eql(u8, name, "audio_embeddings") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.video_dec_block.visual_modulation") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.video_dec_block.self_attention") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.video_dec_block.cross_attention") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.audio_dec_block.visual_modulation") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.audio_dec_block.self_attention") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.audio_dec_block.cross_attention") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.va_modulation") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.av_modulation") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.va_cross_attention") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.av_cross_attention") and
        !std.mem.eql(u8, name, "visual_transformer_blocks.0.video_dec_block.feed_forward") and
        !std.mem.eql(u8, name, stop) and !std.mem.startsWith(u8, name, "video_attention_trace.") and
        !std.mem.startsWith(u8, name, "audio_attention_trace.") and !std.mem.startsWith(u8, name, "va_attention_trace.") and !std.mem.startsWith(u8, name, "av_attention_trace.")) return;
    const r: *Rec = @ptrCast(@alignCast(ctx));
    if (out.ld != out.cols) return error.NonContiguousDiagnostic;
    var nb: [256]u8 = undefined;
    try r.op(try std.fmt.bufPrint(&nb, "k6.aux.dit.block.{s}", .{name}), "diagnostic", .{}, &.{}, &.{.{
        .role = "raw",
        .dtype = .uint8,
        .shape = &.{out.bytes()},
        .src = .{ .device = out.ptr },
    }});
    if (std.mem.eql(u8, name, stop)) {
        try r.note("diagnostic_stop", .{ .after = name, .qualification = "unqualified" });
        return error.DiagnosticTraceComplete;
    }
}

/// First-step audio residuals across all blocks; unchanged math, explicit diagnostic stop.
fn traceDitAudio(ctx: *anyopaque, name: []const u8, out: k6.dit.layout.View) anyerror!void {
    const stop = "audio_out_layer";
    if (!std.mem.eql(u8, name, "audio_embeddings") and !std.mem.eql(u8, name, stop) and
        !std.mem.startsWith(u8, name, "audio_out_layer.") and
        !(std.mem.startsWith(u8, name, "visual_transformer_blocks.") and std.mem.endsWith(u8, name, ".audio"))) return;
    const r: *Rec = @ptrCast(@alignCast(ctx));
    if (out.ld != out.cols) return error.NonContiguousDiagnostic;
    var nb: [256]u8 = undefined;
    try r.op(try std.fmt.bufPrint(&nb, "k6.aux.dit.audio.{s}", .{name}), "diagnostic", .{}, &.{}, &.{.{
        .role = "raw",
        .dtype = .uint8,
        .shape = &.{out.bytes()},
        .src = .{ .device = out.ptr },
    }});
    if (std.mem.eql(u8, name, stop)) {
        try r.note("diagnostic_stop", .{ .after = name, .qualification = "unqualified" });
        return error.DiagnosticTraceComplete;
    }
}

fn e2eRun(p: *P.Pipeline, gpa: std.mem.Allocator, prompt: []const u8, seed: u64, index: u32) !void {
    const frames = try gpa.alloc(u8, @intCast(p.videoGeometry().rgbBytes()));
    defer gpa.free(frames);
    const wave = try gpa.alloc(f32, k6.shapes.audio_samples);
    defer gpa.free(wave);
    const quiet = struct {
        fn report(_: *anyopaque, _: []const u8, _: u32, _: u32) void {}
    };
    var ctx: u8 = 0;
    try p.generate(.{ .prompt = prompt, .seed = seed, .prompt_index = index }, frames, wave, .{ .ctx = &ctx, .report = quiet.report });
}

test "arguments: the required flags and the stage list" {
    try std.testing.expectError(error.Usage, parse(&.{ "--model", "m" }));
    const a = try parse(&.{ "--model", "m", "--options", "o.json", "--out", "d", "--seed", "42", "--e2e-prompt", "1" });
    try std.testing.expectEqual(@as(?u64, 42), a.seed);
    try std.testing.expectEqual(@as(u32, 1), a.e2e_prompt);
    try std.testing.expectError(error.Usage, parse(&.{ "--model", "m", "--options", "o.json", "--out", "d", "--bogus", "x" }));
}

/// The decoder trace's state: the capture and the hook name to stop after.
const AudioTrace = struct { rec: *Rec, stop: []const u8 };

/// Component-only trace, stopped before downstream decoder arithmetic.
fn traceAudioDecoder(ctx: *anyopaque, name: []const u8, out: k6.audio.vae.View) anyerror!void {
    const t: *const AudioTrace = @ptrCast(@alignCast(ctx));
    try traceAudio(t.rec, "k6.aux.audio.decoder.", name, out);
    if (std.mem.eql(u8, name, t.stop)) {
        try t.rec.note("diagnostic_stop", .{ .after = name, .qualification = "unqualified" });
        return error.DiagnosticTraceComplete;
    }
}

/// Every vocoder stage of the `vocoder` diagnostic; no stop (the stage ends at the wave).
fn traceAudioVocoder(ctx: *anyopaque, name: []const u8, out: k6.audio.vae.View) anyerror!void {
    try traceAudio(@ptrCast(@alignCast(ctx)), "k6.aux.audio.vocoder.", name, out);
}

fn traceAudio(r: *Rec, prefix: []const u8, name: []const u8, out: k6.audio.vae.View) !void {
    try out.validate();
    if (!out.isContiguous()) return error.NonContiguousDiagnostic;
    var nb: [128]u8 = undefined;
    try r.op(try std.fmt.bufPrint(&nb, "{s}{s}", .{ prefix, name }), "diagnostic", .{}, &.{}, &.{.{
        .role = "raw",
        .dtype = .uint8,
        .shape = &.{out.bytes()},
        .src = .{ .device = out.addr() },
    }});
}

test "arguments: --audio-trace takes one known stop, and every stop is a distinct decoder hook" {
    const base = [_][]const u8{ "--model", "m", "--options", "o.json", "--out", "d" };
    try std.testing.expectEqual(@as(?[]const u8, null), (try parse(&base)).audio_stop);
    try std.testing.expectEqualStrings("up1_first_residual", (try parse(&(base ++ [_][]const u8{ "--audio-trace", "up1-first" }))).audio_stop.?);
    try std.testing.expectEqualStrings("up1_upsample_conv", (try parse(&(base ++ [_][]const u8{ "--audio-trace", "upsample" }))).audio_stop.?);
    try std.testing.expectEqualStrings("mel", (try parse(&(base ++ [_][]const u8{ "--audio-trace", "mel" }))).audio_stop.?);
    try std.testing.expectError(error.Usage, parse(&(base ++ [_][]const u8{ "--audio-trace", "up3-first" })));
    try std.testing.expectError(error.Usage, parse(&(base ++ [_][]const u8{ "--audio-trace", "mel", "--audio-trace", "mid2" })));
    for (audio_stops, 0..) |a, i| {
        for (audio_stops[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a.arg, b.arg) and !std.mem.eql(u8, a.stop, b.stop));
        // a stop is a decoder hook: a residual-block hook (launch.zig decoder_blocks), the upsample's, or the end ops
        var known = std.mem.eql(u8, a.stop, "first_conv") or std.mem.eql(u8, a.stop, "attention_residual") or std.mem.eql(u8, a.stop, "mel");
        for (k6.audio.launch.decoder_blocks) |blk| known = known or (std.mem.startsWith(u8, a.stop, blk.name) and a.stop.len > blk.name.len and a.stop[blk.name.len] == '_');
        for (k6.audio.launch.decoder_upsamples) |up| known = known or (std.mem.startsWith(u8, a.stop, up.name) and std.mem.endsWith(u8, a.stop, "_upsample_conv"));
        try std.testing.expect(known);
    }
}

test "stages: vocoder is a diagnostic needing --oracle, recording only k6.aux.audio.vocoder." {
    try std.testing.expectEqual(Stage.vocoder, std.meta.stringToEnum(Stage, "vocoder").?);
    try std.testing.expectEqual(@as(usize, 1), prefixes(.vocoder).len);
    try std.testing.expectEqualStrings("k6.aux.audio.vocoder.", prefixes(.vocoder)[0]);
}
