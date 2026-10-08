//! `localrouter check h3-te CHECKPOINT CAPTURE [ROPE_MODE]`: MiniMax H3's text encoder (Qwen3-VL 32B, NVFP4 AWQ checkpoint as
//! published) against a twin capture made with `stk_twin.h3.te32` (ops `te.*`, the `te32_ids` note): every op alone
//! (from the captured inputs) and chained (from Zig's own outputs), and the encode time. One JSON line.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 2 or args.len > 3) {
        std.debug.print("usage: localrouter check h3-te CHECKPOINT CAPTURE [ROPE_MODE]\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const rope_mode: i32 = if (args.len == 3) try std.fmt.parseInt(i32, args[2], 10) else 0;
    var cap = try qi.capture.Capture.open(gpa, io, args[1]);
    defer cap.close();
    const note = (cap.note("te32_ids") orelse return error.NoIds).object.get("ids").?.array.items;
    const ids = try a.alloc(u32, note.len);
    for (ids, note) |*o, v| o.* = @intCast(v.integer);

    var pack = try qi.pack.Pack.openFile(gpa, io, args[0]);
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
    var ops = try h3.te32.Ops.load(&d, h3.kernels.te32);
    defer ops.unload();
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    const t0 = std.Io.Clock.awake.now(io);
    var te = try h3.te32.TextEncoder.init(gpa, io, &d, &ops, s, &pack, &up, @intCast(ids.len));
    defer te.deinit();
    try s.synchronize();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();

    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"h3_te\": {{\"gpu\": \"sm_{d}{d}\", \"tokens\": {d}, \"rope_mode\": {d}, \"load_ms\": {d}, \"weights_bytes\": {d}", .{ maj, min, ids.len, rope_mode, load_ms, te.store.bytes });
    var pass = true;
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        te.probe = r.probe();
        _ = try te.forward(ids, rope_mode);
        try s.synchronize();
        pass = pass and r.stats.differ == 0 and r.stats.unmatched == 0 and r.stats.equal > 0;
        try out.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"skipped\": {d}, \"unmatched\": {d}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.skipped, r.stats.unmatched });
        for (r.first_diffs.items, 0..) |m, i| try out.writer.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", m });
        try out.writer.writeAll("]}");
    }
    te.probe = null;
    {
        const t1 = std.Io.Clock.awake.now(io);
        _ = try te.forward(ids, rope_mode);
        try s.synchronize();
        try out.writer.print(", \"encode_ms\": {d}", .{t1.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds()});
    }
    try out.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}
