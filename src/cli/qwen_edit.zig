//! localrouter check qwen-edit DIT_PACK TE_PACK VAE_PACK CAPTURE: full edit alone/chained replay and pixels.
const std = @import("std");
const qi = @import("qwen_image");
const png = @import("../media/png_decode.zig");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) {
        std.debug.print("usage: localrouter check qwen-edit DIT_PACK TE_PACK VAE_PACK CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[3]);
    defer cap.close();
    const request = (cap.note("request") orelse return error.NoRequest).object;
    if (!std.mem.eql(u8, request.get("kind").?.string, "image_edit")) return error.NotEditCapture;
    const width: u32 = @intCast(request.get("width").?.integer);
    const height: u32 = @intCast(request.get("height").?.integer);
    const steps: u32 = @intCast(request.get("steps").?.integer);
    const seed: u64 = @intCast(request.get("seed").?.integer);
    const refs = try a.alloc(qi.edit_prepare.Prepared, request.get("sources").?.array.items.len);
    if (refs.len == 0 or refs.len > 5) return error.BadReferences;
    var side = @max(width, height);
    for (refs, 0..) |*ref, j| {
        const path = try std.fmt.allocPrint(a, "{s}/inputs/{d}.png", .{ args[3], j });
        const decoded = try png.decode(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 << 20)));
        ref.* = try qi.edit_prepare.prepare(a, try decoded.rgba(a), decoded.width, decoded.height, 1024 * 1024);
        side = @max(side, @max(ref.width, ref.height));
    }
    const tri = try qi.triton_k.specsFromCapture(a, io, cap.dir);
    const source: qi.pipeline.TritonSource = if (tri[0] != null) .{ .specs = tri } else .pack;
    const p = try qi.pipeline.Pipeline.createEdit(gpa, io, .{ .dit = args[0], .te = args[1], .vae = args[2] }, source, side, @as(u64, side) * side, @intCast(refs.len));
    defer p.destroy();
    const sig = try p.sigmas(a, height, width, steps);
    const expected_sig = (cap.note("sigmas") orelse return error.NoSigmas).object.get("values").?.array.items;
    var sigmas_equal = sig.len == expected_sig.len;
    if (sigmas_equal) for (sig, expected_sig) |s, expected| {
        const v: f64 = switch (expected) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => return error.BadSigmas,
        };
        sigmas_equal = sigmas_equal and @as(f64, s) == v;
    };
    const pixels = try a.alloc(u8, @as(usize, width) * height * p.vae.out_channels);
    const pixel_ref = cap.ops[cap.find("vae.to_u8", 0) orelse return error.NoPixels].out("y") orelse return error.NoPixels;
    const expected = try a.alloc(u8, pixel_ref.bytes());
    try cap.blob(io, pixel_ref, expected);
    // Compare the final artifact too; an op blob alone cannot prove the saved output's channel order.
    const saved = try png.decode(a, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ args[3], "image.png" }), a, .limited(64 << 20)));
    const artifact_equal = saved.width == width and saved.height == height and std.mem.eql(u8, if (p.vae.out_channels == 4) try saved.rgba(a) else saved.rgb, expected);
    var pass = sigmas_equal and artifact_equal;
    var output: std.Io.Writer.Allocating = .init(a);
    inline for (.{ qi.replay.Mode.alone, qi.replay.Mode.chained }) |mode| {
        var replay: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &p.d, .s = p.s, .cap = &cap, .mode = mode };
        defer replay.deinit();
        _ = try p.edit(request.get("prompt").?.string, refs, height, width, steps, seed, pixels, null, replay.probe());
        const pixels_equal = std.mem.eql(u8, pixels, expected);
        // Every captured op must have been reached, and no output may have been silently skipped.
        const all_reached = replay.next == cap.ops.len and replay.missed_ops == 0;
        const ok = replay.stats.differ == 0 and replay.stats.unmatched == 0 and replay.stats.skipped == 0 and all_reached and pixels_equal;
        pass = pass and ok;
        try output.writer.print("{{\"mode\":\"{s}\",\"stats\":{s},\"all_reached\":{},\"pixels_equal\":{},\"diffs\":{s}}}\n", .{ @tagName(mode), try std.json.Stringify.valueAlloc(a, replay.stats, .{}), all_reached, pixels_equal, try std.json.Stringify.valueAlloc(a, replay.first_diffs.items, .{}) });
    }
    try output.writer.print("{{\"qwen_edit\":{{\"sigmas_equal\":{},\"artifact_equal\":{},\"pass\":{}}}}}\n", .{ sigmas_equal, artifact_equal, pass });
    try std.Io.File.stdout().writeStreamingAll(io, output.written());
    return if (pass) 0 else 1;
}
