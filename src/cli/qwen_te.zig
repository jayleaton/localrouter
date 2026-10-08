//! `localrouter check qwen-te TE_PACK CAPTURE`: the text encoder's bit gate on a GPU. The capture's prompt through Zig's
//! tokenizer against the twin's ids, then the 36 layers replayed alone (each op from the captured inputs) and chained
//! (from Zig's own outputs), the context rows against the twin's, and the encode time. One JSON summary line.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 2) {
        std.debug.print("usage: localrouter check qwen-te TE_PACK CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[1]);
    defer cap.close();
    var pack = try qi.pack.Pack.open(gpa, io, args[0]);
    defer pack.close(io);
    var prompt = try qi.te.Prompt.load(gpa, io, args[0]);
    defer prompt.deinit();

    // ids: the tokenizer and template against the twin's
    const req = (cap.note("request") orelse return error.NoRequest).object;
    const text = req.get("prompt").?.string;
    const ids = try prompt.ids(a, text);
    const want = (cap.note("te_ids") orelse return error.NoIds).object;
    const want_ids = want.get("ids").?.array.items;
    var ids_equal = ids.len == want_ids.len and want.get("drop").?.integer == prompt.drop;
    if (ids_equal) for (ids, want_ids) |x, y| {
        ids_equal = ids_equal and x == y.integer;
    };

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
    var ops = try qi.ops_launch.Ops.load(&d, qi.kernels.ops, qi.kernels.attention);
    defer ops.unload();
    var tk = try qi.ops_launch.TeOps.load(&d, qi.kernels.te, qi.kernels.gemm);
    defer tk.unload();
    const inv = try qi.te.invFreq(a, qi.kernels.te_inv_freq);
    const t0 = std.Io.Clock.awake.now(io);
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    var te = try qi.te.TextEncoder.init(gpa, &d, &tk, &ops, s, &pack, &up, inv, @intCast(@max(ids.len, 1)));
    try s.synchronize();
    defer te.deinit();
    try s.synchronize();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();

    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"qwen_te\": {{\"gpu\": \"sm_{d}{d}\", \"tokens\": {d}, \"ids_equal\": {}, \"load_ms\": {d}", .{ maj, min, ids.len, ids_equal, load_ms });
    var pass = ids_equal;
    const ctx_ref = cap.ops[cap.find("text_encoder", 0) orelse return error.NoContext].out("context").?;
    const rows = ids.len - prompt.drop;
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        te.probe = r.probe();
        const h = try te.forward(ids);
        try s.synchronize();
        // the context: rows drop.. of the last layer's output
        const host = try a.alloc(u8, rows * qi.te.dim * 2);
        try d.check(d.api.cuMemcpyDtoH_v2(host.ptr, h + @as(u64, prompt.drop) * qi.te.dim * 2, host.len), "cuMemcpyDtoH");
        const ref = try a.alloc(u8, ctx_ref.bytes());
        try cap.blob(io, ctx_ref, ref);
        const ctx_equal = std.mem.eql(u8, host, ref);
        pass = pass and ctx_equal and r.stats.differ == 0 and r.stats.unmatched == 0;
        try out.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"context_equal\": {}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.skipped, r.stats.unmatched, ctx_equal });
        for (r.first_diffs.items, 0..) |m, i| try out.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
        try out.writer.writeAll("]}");
    }
    te.probe = null;
    { // encode time: tokenize + forward, warm
        _ = try te.forward(ids);
        try s.synchronize();
        const t1 = std.Io.Clock.awake.now(io);
        const reps = 5;
        for (0..reps) |_| {
            const again = try prompt.ids(a, text);
            _ = try te.forward(again);
        }
        try s.synchronize();
        const us = t1.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds();
        try out.writer.print(", \"encode_ms\": {d:.2}", .{@as(f64, @floatFromInt(us)) / 1000.0 / reps});
    }
    try out.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}
