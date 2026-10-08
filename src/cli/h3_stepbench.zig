//! `stk check h3-step-bench PACK CAPTURE [--frames N] [--reps N]`: where the time of one MiniMax H3 DiT step goes, and what
//! the fast schedule saves. From a twin capture (`stk_twin.h3.capture`) it takes the request (size, sigmas) and the text
//! states; the step's latents are the capture's when `--frames` is the capture's, else a fixed pseudo-random bf16 fill
//! (timing does not depend on values, equality is checked between the schedules, not against the twin). The step runs
//! under four configurations of the same engine: `ref` (no fusion, no step graph: the reference schedule, what
//! STK_H3_REF=1 selects), `fuse` (fused gate_add + norm_mod, attention stored as rows), `graph` (the reference launches
//! replayed as a CUDA graph) and `fuse_graph` (what the engine runs). Each is warmed, then timed `--reps` times (GPU events
//! around the step; the host's time to issue it is reported too: if it is near the GPU time the step is launch bound),
//! and its velocities are compared byte for byte with `ref`'s. Then `ref` and `fuse` run once more with an event pair
//! around every op (`profile.zig`), summed by op class over the 50 blocks, with the launch count and the time between ops.
//! One JSON line; exit 1 if any configuration's velocities differ from `ref`'s or the graphs turned themselves off.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");

pub const Opts = struct {
    pack: []const u8,
    capture: []const u8,
    frames: ?u32 = null,
    reps: u32 = 5,
};

pub fn parseArgs(args: []const []const u8) !Opts {
    var o: Opts = .{ .pack = "", .capture = "" };
    var pos: usize = 0;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--frames") or std.mem.eql(u8, a, "--reps")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            const v = try std.fmt.parseInt(u32, args[i], 10);
            if (v == 0) return error.BadValue;
            if (a[2] == 'f') o.frames = v else o.reps = v;
        } else if (std.mem.startsWith(u8, a, "--")) {
            return error.UnknownOption;
        } else {
            switch (pos) {
                0 => o.pack = a,
                1 => o.capture = a,
                else => return error.TooManyArguments,
            }
            pos += 1;
        }
    }
    if (pos != 2) return error.NeedPackAndCapture;
    return o;
}

/// The median of `v` (sorted in place); the mean of the middle two for an even count.
pub fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return if (v.len % 2 == 1) v[v.len / 2] else (v[v.len / 2 - 1] + v[v.len / 2]) / 2.0;
}

const Cfg = struct { name: []const u8, fuse: bool, graphs: bool };
const cfgs = [_]Cfg{
    .{ .name = "ref", .fuse = false, .graphs = false },
    .{ .name = "fuse", .fuse = true, .graphs = false },
    .{ .name = "graph", .fuse = false, .graphs = true },
    .{ .name = "fuse_graph", .fuse = true, .graphs = true },
};

const Timing = struct { median_ms: f64, min_ms: f64, host_ms: f64, equal: bool };
const Prof = struct { total_ms: f64, sum_ms: f64, launches: u64, sums: [h3.profile.n_classes]f64 };

fn fillBf16(buf: []u8, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const v = std.mem.bytesAsSlice(u16, buf);
    for (v) |*e| e.* = h3.layout.bf16Bits(r.float(f32) * 4.0 - 2.0);
}

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    const o = parseArgs(args) catch |err| {
        std.debug.print("usage: stk check h3-step-bench PACK CAPTURE [--frames N] [--reps N]  ({s})\n", .{@errorName(err)});
        return 2;
    };
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, o.capture);
    defer cap.close();
    const req = (cap.note("request") orelse return error.NoRequest).object;
    const width: u32 = @intCast(req.get("width").?.integer);
    const height: u32 = @intCast(req.get("height").?.integer);
    const cap_frames: u32 = @intCast(req.get("frames").?.integer);
    const step_i: usize = @intCast(req.get("step").?.integer);
    const frames = o.frames orelse cap_frames;
    const sig = (cap.note("sigmas") orelse return error.NoSigmas).object.get("values").?.array.items;
    const sigma: f32 = @floatCast(switch (sig[step_i]) {
        .float => |f| f,
        .integer => |n| @as(f64, @floatFromInt(n)),
        else => return error.BadSigma,
    });
    const t: u32 = if (frames <= 5) 2 else ((frames - 5) / 17) * 5 + 2;
    const lh = height / 16;
    const lw = width / 16;
    const audio_t: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(frames)) / 24.0 * 40.0));
    const ctx_ref = cap.ops[cap.find("text_encoder", 0) orelse return error.NoContext].out("context").?;
    const l: u32 = @intCast(ctx_ref.shape[0]);
    const ly = h3.layout.Layout.init(l, t, lh, lw, audio_t);

    var pack = try qi.pack.Pack.open(gpa, io, o.pack);
    defer pack.close(io);
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var maj: c_int = 0;
    var min: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&maj, .compute_capability_major, ctx.device), "cc");
    try d.check(d.api.cuDeviceGetAttribute(&min, .compute_capability_minor, ctx.device), "cc");
    var s = try cuda.Stream.init(&d, false);
    defer s.deinit();
    var nv = try qi.nvfp4_exec.Nvfp4.load(gpa, &d, @intCast(maj), @intCast(min));
    defer nv.unload();
    var ops = try qi.ops_launch.Ops.load(&d, qi.kernels.ops, qi.kernels.attention);
    defer ops.unload();
    var te = try qi.ops_launch.TeOps.load(&d, qi.kernels.te, qi.kernels.gemm);
    defer te.unload();
    var hops = try h3.launch.Ops.load(&d, h3.kernels.ops);
    defer hops.unload();
    var kitchen = try h3.launch.Kitchen.load(&d, h3.kernels.kitchen);
    defer kitchen.unload();
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    var w = try h3.dit.Weights.load(gpa, io, &d, &pack, &nv, &up);
    defer w.deinit();
    try s.synchronize();
    var dit = try h3.dit.Dit.init(gpa, &d, .{ .h3 = &hops, .kitchen = &kitchen, .te = &te, .ops = &ops, .nv = &nv, .major = @intCast(maj), .minor = @intCast(min) }, &w, s, ly.s, l);
    defer dit.deinit();

    // device inputs: text, video, audio latents, and the two velocities
    var bufs: [5]cuda.DeviceBuffer = undefined;
    const nvx: usize = 24 * @as(usize, t) * lh * lw * 2;
    const nax: usize = 32 * 2 * @as(usize, audio_t) * 2;
    for (&bufs, [_]usize{ ctx_ref.bytes(), nvx, nax, nvx, nax }) |*b, len| b.* = try cuda.DeviceBuffer.alloc(&d, len);
    defer for (&bufs) |*b| b.free();
    const inputs = cap.ops[cap.find("step.inputs", 0) orelse return error.NoInputs];
    const synthetic = frames != cap_frames;
    for ([_]struct { []const u8, usize }{ .{ "context", 0 }, .{ "video", 1 }, .{ "audio", 2 } }) |e| {
        const ref = if (e[1] == 0) ctx_ref else inputs.out(e[0]).?;
        const host = try a.alloc(u8, if (e[1] == 0 or !synthetic) ref.bytes() else if (e[1] == 1) nvx else nax);
        if (e[1] == 0 or !synthetic) try cap.blob(io, ref, host) else fillBf16(host, 0x48335 + e[1]);
        try bufs[e[1]].upload(0, host);
    }
    try dit.prepare(try bufs[0].at(0), l);
    const vid = try bufs[1].at(0);
    const aud = try bufs[2].at(0);
    const vel_v = try bufs[3].at(0);
    const vel_a = try bufs[4].at(0);

    var e0 = try cuda.Event.init(&d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(&d, true);
    defer e1.deinit();
    const zeros = try a.alloc(u8, @max(nvx, nax));
    @memset(zeros, 0);
    const ref_v = try a.alloc(u8, nvx);
    const ref_a = try a.alloc(u8, nax);
    const got_v = try a.alloc(u8, nvx);
    const got_a = try a.alloc(u8, nax);

    var timings: [cfgs.len]Timing = undefined;
    var all_equal = true;
    const samples = try a.alloc(f64, o.reps);
    const host_samples = try a.alloc(f64, o.reps);
    for (cfgs, 0..) |cfg, ci| {
        dit.fuse = cfg.fuse;
        dit.graphs = cfg.graphs;
        try bufs[3].upload(0, zeros[0..nvx]);
        try bufs[4].upload(0, zeros[0..nax]);
        for (0..2) |_| { // warm: a graph's first use is eager, its second captures
            try dit.step(vid, aud, sigma, t, lh, lw, audio_t, vel_v, vel_a);
            try s.synchronize();
        }
        for (samples, host_samples) |*sm, *hs| {
            try e0.record(s);
            const h0 = std.Io.Clock.awake.now(io);
            try dit.step(vid, aud, sigma, t, lh, lw, audio_t, vel_v, vel_a);
            hs.* = @as(f64, @floatFromInt(h0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
            try e1.record(s);
            try e1.synchronize();
            sm.* = try e0.elapsedMs(e1);
        }
        try d.check(d.api.cuMemcpyDtoH_v2(got_v.ptr, vel_v, got_v.len), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(got_a.ptr, vel_a, got_a.len), "cuMemcpyDtoH");
        if (ci == 0) {
            @memcpy(ref_v, got_v);
            @memcpy(ref_a, got_a);
        }
        const equal = std.mem.eql(u8, got_v, ref_v) and std.mem.eql(u8, got_a, ref_a);
        all_equal = all_equal and equal;
        var mn = samples[0];
        for (samples) |v| mn = @min(mn, v);
        timings[ci] = .{ .median_ms = median(samples), .min_ms = mn, .host_ms = median(host_samples), .equal = equal };
    }

    // the per-op profile of the reference and the fused schedule (eager, warm)
    var prof = try h3.profile.Profile.init(gpa, &d, 2048);
    defer prof.deinit();
    var profs: [2]Prof = undefined;
    for (&profs, cfgs[0..2]) |*pr, cfg| {
        dit.fuse = cfg.fuse;
        dit.graphs = false;
        dit.prof = &prof;
        prof.reset();
        const n0 = dit.launches();
        try e0.record(s);
        try dit.step(vid, aud, sigma, t, lh, lw, audio_t, vel_v, vel_a);
        try e1.record(s);
        try e1.synchronize();
        dit.prof = null;
        pr.sums = try prof.sums();
        pr.total_ms = try e0.elapsedMs(e1);
        pr.launches = dit.launches() - n0;
        pr.sum_ms = 0;
        for (pr.sums) |v| pr.sum_ms += v;
    }

    var out: std.Io.Writer.Allocating = .init(a);
    const wr = &out.writer;
    try wr.print("{{\"h3_stepbench\": {{\"gpu\": \"sm_{d}{d}\", \"text_tokens\": {d}, \"seq\": {d}, \"frames\": {d}, \"latent_t\": {d}, \"width\": {d}, \"height\": {d}, \"reps\": {d}, \"latents\": \"{s}\", \"configs\": {{", .{ maj, min, l, ly.s, frames, t, width, height, o.reps, if (synthetic) "synthetic" else "captured" });
    for (cfgs, timings, 0..) |cfg, tm, ci| {
        try wr.print("{s}\"{s}\": {{\"step_ms\": {d:.2}, \"min_ms\": {d:.2}, \"host_issue_ms\": {d:.2}, \"equal_to_ref\": {}, \"speedup\": {d:.4}}}", .{ if (ci > 0) ", " else "", cfg.name, tm.median_ms, tm.min_ms, tm.host_ms, tm.equal, timings[0].median_ms / tm.median_ms });
    }
    try wr.writeAll("}, \"profile\": {");
    for (profs, cfgs[0..2], 0..) |pr, cfg, ci| {
        try wr.print("{s}\"{s}\": {{\"total_ms\": {d:.2}, \"sum_ms\": {d:.2}, \"between_ops_ms\": {d:.2}, \"launches\": {d}, \"classes_ms\": {{", .{ if (ci > 0) ", " else "", cfg.name, pr.total_ms, pr.sum_ms, pr.total_ms - pr.sum_ms, pr.launches });
        inline for (h3.profile.names, 0..) |f, i| try wr.print("{s}\"{s}\": {d:.3}", .{ if (i > 0) ", " else "", f, pr.sums[i] });
        try wr.writeAll("}}");
    }
    try wr.writeAll("}, \"graph_error\": ");
    if (dit.graph_error) |e| try wr.print("\"{s}\"", .{@errorName(e)}) else try wr.writeAll("null");
    const pass = all_equal and dit.graph_error == null;
    try wr.print(", \"all_equal\": {}, \"pass\": {}}}}}\n", .{ all_equal, pass });
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}

test "h3 step bench: the step's code type-checks and the arguments parse" {
    _ = &run;
    const o = try parseArgs(&.{ "pack", "cap", "--frames", "124", "--reps", "3" });
    try std.testing.expectEqualStrings("pack", o.pack);
    try std.testing.expectEqual(@as(?u32, 124), o.frames);
    try std.testing.expectEqual(@as(u32, 3), o.reps);
    try std.testing.expectEqual(@as(u32, 5), (try parseArgs(&.{ "p", "c" })).reps);
    try std.testing.expectError(error.NeedPackAndCapture, parseArgs(&.{"p"}));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "p", "c", "--nope" }));
    try std.testing.expectError(error.MissingValue, parseArgs(&.{ "p", "c", "--reps" }));
    try std.testing.expectError(error.BadValue, parseArgs(&.{ "p", "c", "--reps", "0" }));
}

test "median" {
    var v = [_]f64{ 3, 1, 2 };
    try std.testing.expectEqual(@as(f64, 2), median(&v));
    var u = [_]f64{ 4, 1, 3, 2 };
    try std.testing.expectEqual(@as(f64, 2.5), median(&u));
}
