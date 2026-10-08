//! `localrouter check h3-replay PACK CAPTURE`: the MiniMax H3 DiT's bit gate on a GPU. From a twin capture
//! (`stk_twin.h3.capture`): the token refiner and one step, every op compared byte for byte, alone (each op from the
//! captured inputs) and chained (from Zig's own outputs), and the step's velocities against the twin's. One JSON line.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 2) {
        std.debug.print("usage: localrouter check h3-replay PACK CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[1]);
    defer cap.close();
    const req = (cap.note("request") orelse return error.NoRequest).object;
    const width: u32 = @intCast(req.get("width").?.integer);
    const height: u32 = @intCast(req.get("height").?.integer);
    const frames: u32 = @intCast(req.get("frames").?.integer);
    const step_i: usize = @intCast(req.get("step").?.integer);
    const sig = (cap.note("sigmas") orelse return error.NoSigmas).object.get("values").?.array.items;
    const sigma: f32 = @floatCast(switch (sig[step_i]) {
        .float => |f| f,
        .integer => |n| @as(f64, @floatFromInt(n)),
        else => return error.BadSigma,
    });
    const t: u32 = if (frames <= 5) 2 else ((frames - 5) / 17) * 5 + 2;
    const lh = height / 16;
    const lw = width / 16;
    const audio_t: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(frames)) / 24.0 * 40.0)); // no .5 ties at 17k + 5 frames
    const ctx_ref = cap.ops[cap.find("text_encoder", 0) orelse return error.NoContext].out("context").?;
    const l: u32 = @intCast(ctx_ref.shape[0]);
    const ly = h3.layout.Layout.init(l, t, lh, lw, audio_t);

    var pack = try qi.pack.Pack.open(gpa, io, args[0]);
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
    const t0 = std.Io.Clock.awake.now(io);
    var w = try h3.dit.Weights.load(gpa, io, &d, &pack, &nv, &up);
    defer w.deinit();
    try s.synchronize();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    var dit = try h3.dit.Dit.init(gpa, &d, .{ .h3 = &hops, .kitchen = &kitchen, .te = &te, .ops = &ops, .nv = &nv, .major = @intCast(maj), .minor = @intCast(min) }, &w, s, ly.s, l);
    defer dit.deinit();

    // device inputs: the text states and the step's latents, as captured
    var bufs: [5]cuda.DeviceBuffer = undefined; // text, video, audio, video velocity, audio velocity
    const nvx: usize = 24 * @as(usize, t) * lh * lw * 2;
    const nax: usize = 32 * 2 * @as(usize, audio_t) * 2;
    for (&bufs, [_]usize{ ctx_ref.bytes(), nvx, nax, nvx, nax }) |*b, len| b.* = try cuda.DeviceBuffer.alloc(&d, len);
    defer for (&bufs) |*b| b.free();
    const inputs = cap.ops[cap.find("step.inputs", 0) orelse return error.NoInputs];
    for ([_]struct { []const u8, usize }{ .{ "context", 0 }, .{ "video", 1 }, .{ "audio", 2 } }) |e| {
        const ref = if (e[1] == 0) ctx_ref else inputs.out(e[0]).?;
        const host = try a.alloc(u8, ref.bytes());
        try cap.blob(io, ref, host);
        try bufs[e[1]].upload(0, host);
    }

    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"h3_replay\": {{\"gpu\": \"sm_{d}{d}\", \"tokens\": {d}, \"seq\": {d}, \"sigma\": {d}, \"load_ms\": {d}, \"weights_bytes\": {d}", .{ maj, min, l, ly.s, sigma, load_ms, w.store.bytes });
    var pass = true;
    const outputs = cap.ops[cap.find("step.outputs", 0) orelse return error.NoOutputs];
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        dit.probe = r.probe();
        dit.layout_key = null;
        try dit.prepare(try bufs[0].at(0), l);
        try dit.step(try bufs[1].at(0), try bufs[2].at(0), sigma, t, lh, lw, audio_t, try bufs[3].at(0), try bufs[4].at(0));
        try s.synchronize();
        var vel_equal = true;
        for ([_]struct { []const u8, usize }{ .{ "video", 3 }, .{ "audio", 4 } }) |e| {
            const ref = outputs.out(e[0]).?;
            const want = try a.alloc(u8, ref.bytes());
            try cap.blob(io, ref, want);
            const got = try a.alloc(u8, ref.bytes());
            try d.check(d.api.cuMemcpyDtoH_v2(got.ptr, try bufs[e[1]].at(0), got.len), "cuMemcpyDtoH");
            vel_equal = vel_equal and std.mem.eql(u8, got, want);
        }
        pass = pass and r.stats.differ == 0 and r.stats.unmatched == 0 and vel_equal;
        try out.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"velocities_equal\": {}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.skipped, r.stats.unmatched, vel_equal });
        for (r.first_diffs.items, 0..) |m, i| try out.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
        try out.writer.writeAll("]}");
    }
    dit.probe = null;
    { // the fast schedules, probe-free: each must give the captured velocities byte for byte (the probed passes above ran the
        // reference schedule, op by op). Three steps a configuration: the first and second warm it (a graph's first use is
        // eager, its second captures), the third is a replay; the velocities are cleared before each and checked after.
        const default_fuse = dit.fuse;
        const default_graphs = dit.graphs;
        const zeros = try a.alloc(u8, @max(nvx, nax));
        @memset(zeros, 0);
        try out.writer.writeAll(", \"fast\": {");
        for ([_]struct { []const u8, bool, bool }{ .{ "ref", false, false }, .{ "fuse", true, false }, .{ "graph", false, true }, .{ "fuse_graph", true, true } }, 0..) |cfg, ci| {
            dit.fuse = cfg[1];
            dit.graphs = cfg[2];
            var equal = true;
            var ms: i64 = 0;
            for (0..3) |rep| {
                try bufs[3].upload(0, zeros[0..nvx]);
                try bufs[4].upload(0, zeros[0..nax]);
                const t1 = std.Io.Clock.awake.now(io);
                try dit.step(try bufs[1].at(0), try bufs[2].at(0), sigma, t, lh, lw, audio_t, try bufs[3].at(0), try bufs[4].at(0));
                try s.synchronize();
                if (rep == 2) ms = t1.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
                for ([_]struct { []const u8, usize }{ .{ "video", 3 }, .{ "audio", 4 } }) |e| {
                    const ref = outputs.out(e[0]).?;
                    const want = try a.alloc(u8, ref.bytes());
                    try cap.blob(io, ref, want);
                    const got = try a.alloc(u8, ref.bytes());
                    try d.check(d.api.cuMemcpyDtoH_v2(got.ptr, try bufs[e[1]].at(0), got.len), "cuMemcpyDtoH");
                    equal = equal and std.mem.eql(u8, got, want);
                }
            }
            pass = pass and equal;
            try out.writer.print("{s}\"{s}\": {{\"velocities_equal\": {}, \"step_ms\": {d}}}", .{ if (ci > 0) ", " else "", cfg[0], equal, ms });
        }
        try out.writer.writeAll("}, \"graph_error\": ");
        if (dit.graph_error) |e| try out.writer.print("\"{s}\"", .{@errorName(e)}) else try out.writer.writeAll("null");
        dit.fuse = default_fuse;
        dit.graphs = default_graphs and dit.graph_error == null;
    }
    { // a step's time, warm, as the engine runs it (the graph of the last configuration above is reused: a replay)
        const t1 = std.Io.Clock.awake.now(io);
        try dit.step(try bufs[1].at(0), try bufs[2].at(0), sigma, t, lh, lw, audio_t, try bufs[3].at(0), try bufs[4].at(0));
        try s.synchronize();
        try out.writer.print(", \"step_ms\": {d}", .{t1.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds()});
    }
    try out.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}
