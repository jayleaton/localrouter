//! `localrouter check h3-i2v PACK TE VAE CAPTURE`: PNG decode, encoder, vision/text, refiner and one conditioned DiT
//! step, alone and chained against capture_i2v. Source PNG hash, expanded ids/tags and velocities must also match.
const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const h3 = @import("minimax_h3");

fn pre(r: *qi.replay.Replay, name: []const u8) !void {
    const p = r.probe();
    try p.before(p.ctx, name, &.{});
}
fn post(r: *qi.replay.Replay, name: []const u8, values: []const h3.dit.Io) !void {
    const p = r.probe();
    try p.after(p.ctx, name, values);
}

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) {
        std.debug.print("usage: localrouter check h3-i2v PACK TE VAE CAPTURE\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[3]);
    defer cap.close();
    const req = (cap.note("request") orelse return error.NoRequest).object;
    const width: u32 = @intCast(req.get("width").?.integer);
    const height: u32 = @intCast(req.get("height").?.integer);
    const frames: u32 = @intCast(req.get("frames").?.integer);
    const seed: u64 = @intCast(req.get("seed").?.integer);
    const step_i: usize = @intCast(req.get("step").?.integer);
    const rgb = (try @import("h3_input.zig").canvas(gpa, io, &cap, width, height)) orelse return error.NoImage;
    defer gpa.free(rgb);
    const sh = try h3.pipeline.shapes(width, height, frames);
    var tok = try h3.tokenizer.Tokenizer.load(gpa, io, args[0]);
    defer tok.deinit();
    const tokens = try tok.imageIds(gpa, req.get("prompt").?.string, (width / 32) * (height / 32));
    defer tokens.deinit(gpa);
    const tn = (cap.note("te32_ids") orelse return error.NoIds).object;
    const want_ids = tn.get("ids").?.array.items;
    const want_tags = tn.get("tags").?.array.items;
    var ids_ok = want_ids.len == tokens.ids.len and want_tags.len == tokens.tags.len;
    if (ids_ok) for (tokens.ids, tokens.tags, want_ids, want_tags) |id, tag, wi, wt| {
        ids_ok = ids_ok and id == wi.integer and tag == wt.integer;
    };
    if (!ids_ok) return error.TokenMismatch;
    const sig = (cap.note("sigmas") orelse return error.NoSigmas).object.get("values").?.array.items;
    if (step_i >= sig.len) return error.BadSigma;
    const sigma: f32 = switch (sig[step_i]) {
        .float => |f| @floatCast(f),
        .integer => |n| @floatFromInt(n),
        else => return error.BadSigma,
    };
    var d = try qi.memcap.open();
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
    var bf = try qi.ops_launch.TeOps.load(&d, qi.kernels.te, qi.kernels.gemm);
    defer bf.unload();
    var hops = try h3.launch.Ops.load(&d, h3.kernels.ops);
    defer hops.unload();
    var kitchen = try h3.launch.Kitchen.load(&d, h3.kernels.kitchen);
    defer kitchen.unload();
    var eops = try h3.vae_video_enc.Ops.load(&d, h3.kernels.vae_enc);
    defer eops.unload();
    var vops = try h3.vae_video.Ops.load(&d, h3.kernels.gemm_f16, h3.kernels.vae_video);
    defer vops.unload();
    var visual_ops = try h3.te_vision.Ops.load(&d, h3.kernels.te_vision);
    defer visual_ops.unload();
    var teops = try h3.te32.Ops.load(&d, h3.kernels.te32);
    defer teops.unload();
    var up = try qi.upload.Uploader.init(&d, io, s);
    defer up.deinit();
    var epack = try qi.pack.Pack.openFile(gpa, io, args[1]);
    defer epack.close(io);
    var pack = try qi.pack.Pack.open(gpa, io, args[0]);
    defer pack.close(io);
    var encoder = try h3.vae_video_enc.Encoder.init(gpa, io, &d, &eops, &vops, s, args[2], &up);
    defer encoder.deinit();
    var visual = try h3.te_vision.VisionEncoder.init(gpa, io, &d, &visual_ops, &hops, &teops, s, &epack, &up, (height / 16) * (width / 16));
    defer visual.deinit();
    var te = try h3.te32.TextEncoder.init(gpa, io, &d, &teops, s, &epack, &up, @intCast(tokens.ids.len));
    defer te.deinit();
    var weights = try h3.dit.Weights.load(gpa, io, &d, &pack, &nv, &up);
    defer weights.deinit();
    const ly = h3.layout.Layout.init(@intCast(tokens.ids.len), sh.latent_t, sh.lh, sh.lw, sh.audio_t).withFirstFrame();
    var dit = try h3.dit.Dit.init(gpa, &d, .{ .h3 = &hops, .kitchen = &kitchen, .te = &bf, .ops = &ops, .nv = &nv, .major = @intCast(maj), .minor = @intCast(min) }, &weights, s, ly.s, @intCast(tokens.ids.len));
    defer dit.deinit();
    var pixels = try cuda.DeviceBuffer.fromHost(&d, std.mem.sliceAsBytes(rgb));
    defer pixels.free();
    var z = try cuda.DeviceBuffer.alloc(&d, @intCast(24 * @as(u64, sh.lh) * sh.lw * 4));
    defer z.free();
    var text = try cuda.DeviceBuffer.alloc(&d, tokens.ids.len * h3.te32.dim * 2);
    defer text.free();
    const tags = try a.alloc(i64, tokens.tags.len);
    for (tags, tokens.tags) |*dst, v| dst.* = v;
    var tb = try cuda.DeviceBuffer.fromHost(&d, std.mem.sliceAsBytes(tags));
    defer tb.free();
    var latent: [4]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (latent[0..made]) |*b| b.free();
    for (&latent, [_]usize{ @intCast(sh.n_video * 2), @intCast(sh.n_audio * 2), @intCast(sh.n_video * 2), @intCast(sh.n_audio * 2) }) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(&d, n);
        made += 1;
    }
    const input = cap.ops[cap.find("step.inputs", 0) orelse return error.NoInputs];
    const output = cap.ops[cap.find("step.outputs", 0) orelse return error.NoOutputs];
    for ([_][]const u8{ "video", "audio" }, 0..) |role, i| {
        const ref = input.out(role) orelse return error.NoInputs;
        const host = try a.alloc(u8, ref.bytes());
        try cap.blob(io, ref, host);
        try latent[i].upload(0, host);
    }
    var js: std.Io.Writer.Allocating = .init(a);
    try js.writer.print("{{\"h3_i2v\": {{\"ids_equal\": {}, \"tokens\": {d}", .{ ids_ok, tokens.ids.len });
    var pass = true;
    for ([_]qi.replay.Mode{ .alone, .chained }) |mode| {
        var r: qi.replay.Replay = .{ .gpa = gpa, .io = io, .d = &d, .s = s, .cap = &cap, .mode = mode };
        defer r.deinit();
        encoder.probe = r.probe();
        visual.probe = r.probe();
        te.probe = r.probe();
        dit.probe = r.probe();
        try pre(&r, "image.canvas");
        try post(&r, "image.canvas", &.{.{ .role = "rgb", .ptr = try pixels.at(0), .bytes = rgb.len * 4 }});
        try encoder.encode(try pixels.at(0), height, width, try z.at(0));
        try pre(&r, "first_frame");
        try post(&r, "first_frame", &.{ .{ .role = "rgb", .ptr = try pixels.at(0), .bytes = rgb.len * 4 }, .{ .role = "latent", .ptr = try z.at(0), .bytes = 24 * @as(u64, sh.lh) * sh.lw * 4 } });
        const features = try visual.forward(rgb, width, height);
        const hidden = try te.forwardImage(tokens.ids, features, tokens.start, 0);
        try hops.toBf16(s, hidden, try text.at(0), tokens.ids.len * h3.te32.dim);
        try pre(&r, "text_encoder");
        try post(&r, "text_encoder", &.{ .{ .role = "hidden", .ptr = hidden, .bytes = tokens.ids.len * h3.te32.dim * 4 }, .{ .role = "context", .ptr = try text.at(0), .bytes = tokens.ids.len * h3.te32.dim * 2 } });
        try pre(&r, "text_tags");
        try post(&r, "text_tags", &.{.{ .role = "tags", .ptr = try tb.at(0), .bytes = tags.len * 8 }});
        try dit.prepare(try text.at(0), @intCast(tokens.ids.len));
        try dit.firstFrame(try z.at(0), sh.lh, sh.lw, seed, tokens.tags);
        try pre(&r, "step.inputs");
        try post(&r, "step.inputs", &.{ .{ .role = "video", .ptr = try latent[0].at(0), .bytes = sh.n_video * 2 }, .{ .role = "audio", .ptr = try latent[1].at(0), .bytes = sh.n_audio * 2 } });
        try dit.step(try latent[0].at(0), try latent[1].at(0), sigma, sh.latent_t, sh.lh, sh.lw, sh.audio_t, try latent[2].at(0), try latent[3].at(0));
        try pre(&r, "step.outputs");
        try post(&r, "step.outputs", &.{ .{ .role = "video", .ptr = try latent[2].at(0), .bytes = sh.n_video * 2 }, .{ .role = "audio", .ptr = try latent[3].at(0), .bytes = sh.n_audio * 2 } });
        try s.synchronize();
        var velocity_ok = true;
        for ([_][]const u8{ "video", "audio" }, 0..) |role, i| {
            const ref = output.out(role) orelse return error.NoOutputs;
            const want = try a.alloc(u8, ref.bytes());
            const got = try a.alloc(u8, ref.bytes());
            try cap.blob(io, ref, want);
            try latent[i + 2].download(0, got);
            velocity_ok = velocity_ok and std.mem.eql(u8, want, got);
        }
        pass = pass and r.stats.differ == 0 and r.stats.unmatched == 0 and r.stats.equal > 0 and velocity_ok;
        try js.writer.print(", \"{s}\": {{\"equal\": {d}, \"differ\": {d}, \"unmatched\": {d}, \"velocities_equal\": {}, \"first_diffs\": [", .{ @tagName(mode), r.stats.equal, r.stats.differ, r.stats.unmatched, velocity_ok });
        for (r.first_diffs.items, 0..) |name, i| try js.writer.print("{s}\"{s}\"", .{ if (i > 0) "," else "", name });
        try js.writer.writeAll("]}");
    }
    encoder.probe = null;
    visual.probe = null;
    te.probe = null;
    dit.probe = null;
    try js.writer.print(", \"pass\": {}}}}}\n", .{pass});
    try std.Io.File.stdout().writeStreamingAll(io, js.written());
    return if (pass) 0 else 1;
}
