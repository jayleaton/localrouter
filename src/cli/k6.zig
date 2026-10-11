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
//!     vae     k6.vae.decode from the oracle's latent (--oracle; loads the video VAE)
//!     audio   k6.audio.* chained natively from the oracle's k6.audio.latent_in (--oracle; loads the audio VAE, vocoder)
//!   localrouter check k6-e2e --model SNAPSHOT --options OPTIONS.json --out DIR --seed N --prompt-file F [--e2e-prompt K]
//!     the whole pipeline for line K: k6.dit.step0..9, k6.vae.decode, k6.audio.*, k6.e2e.frames
//!
//! OPTIONS.json is the tool's `options` object (`kandinsky6.pipeline.parseOptions`: every arithmetic and audio policy
//! field required). Both run admitted for diagnostic use: refused while any contract is unresolved. The output
//! directory must not exist. Exit 0 when every stage is ok, 1 otherwise, 2 on usage.

const std = @import("std");
const k6 = @import("kandinsky6");
const qi = @import("qwen_image");
const P = k6.pipeline;
const Rec = k6.rec.Rec;

const Stage = enum { noise, rope, clip, text, piflow, vae, audio, e2e };

/// The op prefixes each stage records (compare.py STAGE_OPS, plus `audio` for the replayed chain).
fn prefixes(s: Stage) []const []const u8 {
    return switch (s) {
        .noise => &.{"k6.noise."},
        .rope => &.{ "k6.freqs.", "k6.rope." },
        .clip => &.{"k6.clip."},
        .text => &.{"k6.qwen."},
        .piflow => &.{"k6.piflow."},
        .vae => &.{"k6.vae."},
        .audio => &.{"k6.audio."},
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
    oracle: ?[]const u8 = null,
};

fn parse(args: []const []const u8) !Args {
    var a: Args = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--model")) a.model = v else if (std.mem.eql(u8, k, "--options")) a.options = v else if (std.mem.eql(u8, k, "--out")) a.out = v else if (std.mem.eql(u8, k, "--stages")) a.stages = v else if (std.mem.eql(u8, k, "--seed")) a.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--prompt-file")) a.prompt_file = v else if (std.mem.eql(u8, k, "--e2e-prompt")) a.e2e_prompt = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--oracle")) a.oracle = v else return error.Usage;
    }
    if (a.model == null or a.options == null or a.out == null) return error.Usage;
    return a;
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
    \\           [--seed N] [--prompt-file F] [--oracle ORACLE_DIR]   (stages: noise rope clip text piflow vae audio)
    \\       localrouter check k6-e2e --model SNAPSHOT --options OPTIONS.json --out DIR --seed N --prompt-file F [--e2e-prompt K]
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
            .audio => parts.audio = true,
            .noise, .rope, .piflow, .e2e => {},
        }
        const missing: ?[]const u8 = switch (s) {
            .noise => if (a.seed == null) "--seed" else null,
            .clip, .text => if (a.prompt_file == null) "--prompt-file" else null,
            .piflow, .vae, .audio => if (a.oracle == null) "--oracle" else null,
            .e2e => if (a.seed == null) "--seed" else if (a.prompt_file == null) "--prompt-file" else null,
            .rope => null,
        };
        if (missing) |m| {
            std.debug.print("k6: stage {t} needs {s}\n", .{ s, m });
            return 2;
        }
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

    var rec = try Rec.create(gpa, io, a.out.?);
    defer rec.deinit();
    rec.only = only.items;
    opts.hooks.rec = &rec;
    var buf: [9]k6.admission.Contract = undefined;
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
            .audio => p.replayAudio(&oracle.?),
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

fn e2eRun(p: *P.Pipeline, gpa: std.mem.Allocator, prompt: []const u8, seed: u64, index: u32) !void {
    const frames = try gpa.alloc(u8, @as(usize, k6.shapes.sample_frames) * k6.shapes.width * k6.shapes.height * 3);
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
