//! `localrouter check qwen-e2e DIT_PACK TE_PACK VAE_PACK CAPTURE [OUT.png]`: Qwen-Image end to end in Zig against a twin
//! capture made with LocalRouter's text encoder and VAE: the capture's prompt, seed, size and steps; the sigmas, the
//! text context, the final latents and the pixels compared byte for byte; then warm timings per phase.

const std = @import("std");
const qi = @import("qwen_image");
const png = @import("../media/png.zig");

pub fn run(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4 and args.len != 5) {
        std.debug.print("usage: localrouter check qwen-e2e DIT_PACK TE_PACK VAE_PACK CAPTURE [OUT.png]\n", .{});
        return 2;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var cap = try qi.capture.Capture.open(gpa, io, args[3]);
    defer cap.close();
    const req = (cap.note("request") orelse return error.NoRequest).object;
    const prompt = req.get("prompt").?.string;
    const seed: u64 = @intCast(req.get("seed").?.integer);
    const height: u32 = @intCast(req.get("height").?.integer);
    const width: u32 = @intCast(req.get("width").?.integer);
    const steps: u32 = @intCast(req.get("steps").?.integer);

    // tfimage's Triton kernels: the capture's own, or (a light capture has no launches) the DiT pack's for this GPU
    const tri = try qi.triton_k.specsFromCapture(a, io, cap.dir);
    const source: qi.pipeline.TritonSource = if (tri[0] != null) .{ .specs = tri } else .pack;
    const t0 = std.Io.Clock.awake.now(io);
    const p = try qi.pipeline.Pipeline.create(gpa, io, .{ .dit = args[0], .te = args[1], .vae = args[2] }, source, @max(height, width));
    defer p.destroy();
    const load_ms = t0.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();

    // the sigmas against the twin's
    const sig = try p.sigmas(a, height, width, steps);
    const want = (cap.note("sigmas") orelse return error.NoSigmas).object.get("values").?.array.items;
    var sigmas_equal = want.len == sig.len;
    if (sigmas_equal) for (sig, want) |x, y| {
        const f: f64 = switch (y) {
            .float => |v| v,
            .integer => |v| @floatFromInt(v),
            else => return error.BadSigmas,
        };
        sigmas_equal = sigmas_equal and @as(f64, x) == f;
    };

    const pixels = try a.alloc(u8, @as(usize, height) * width * p.vae.out_channels);
    const r = try p.generate(prompt, height, width, steps, seed, pixels, null);
    const ctx_equal = try sameAs(p, a, io, &cap, "text_encoder", "context", r.context, @as(u64, r.context_rows) * qi.te.dim * 2);
    const lat_equal = try sameAs(p, a, io, &cap, "latents", "x", r.latents, 64 * @as(u64, height / 16) * (width / 16) * 2);
    const px_ref = cap.ops[cap.find("vae.to_u8", 0) orelse return error.NoPixels].out("y").?;
    const want_px = try a.alloc(u8, px_ref.bytes());
    try cap.blob(io, px_ref, want_px);
    const px_equal = std.mem.eql(u8, pixels, want_px);
    const first = p.times;

    // warm: the same request again
    _ = try p.generate(prompt, height, width, steps, seed, pixels, null);
    const warm = p.times;
    const px_again = std.mem.eql(u8, pixels, want_px);
    // the same without step graphs: the launch time the graphs save, and the same pixels
    p.dit.graphs = false;
    _ = try p.generate(prompt, height, width, steps, seed, pixels, null);
    const nograph = p.times;
    const px_nograph = std.mem.eql(u8, pixels, want_px);
    p.dit.graphs = true;
    if (args.len == 5) {
        var file = try std.Io.Dir.cwd().createFile(io, args[4], .{});
        defer file.close(io);
        var buf: [64 * 1024]u8 = undefined;
        var fw = file.writer(io, &buf);
        try png.encode(&fw.interface, pixels, width, height, if (p.vae.out_channels == 4) .rgba8 else .rgb8, .{ .level = .fastest, .text = &.{} });
        try fw.interface.flush();
    }
    var twin_ms: [3]f64 = .{ 0, 0, 0 };
    if (cap.note("twin_warm_ms")) |tw| for (&twin_ms, [_][]const u8{ "encode", "sample", "decode" }) |*o, k| {
        o.* = switch (tw.object.get(k).?) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => 0,
        };
    };
    const pass = sigmas_equal and ctx_equal and lat_equal and px_equal and px_again and px_nograph;
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.print("{{\"qwen_e2e\": {{\"gpu\": \"sm_{d}{d}\", \"precision\": \"{s}\", \"size\": \"{d}x{d}\", \"steps\": {d}, \"load_ms\": {d}, \"load\": {s}, " ++
        "\"sigmas_equal\": {}, \"context_equal\": {}, \"latents_equal\": {}, \"pixels_equal\": {}, \"pixels_equal_warm\": {}, " ++
        "\"first_ms\": {{\"encode\": {d:.1}, \"prefix\": {d:.1}, \"sample\": {d:.1}, \"decode\": {d:.1}}}, " ++
        "\"warm_ms\": {{\"encode\": {d:.1}, \"prefix\": {d:.1}, \"sample\": {d:.1}, \"decode\": {d:.1}}}, " ++
        "\"sample_ms_without_graphs\": {d:.1}, \"pixels_equal_without_graphs\": {}, \"twin_warm_ms\": {{\"encode\": {d:.1}, \"sample\": {d:.1}, \"decode\": {d:.1}}}, \"pass\": {}}}}}\n", .{
        p.major,         p.minor,      p.dit_pack.precision, width,        height,      steps,       load_ms,
        try std.json.Stringify.valueAlloc(a, p.load, .{}),
        sigmas_equal,    ctx_equal,    lat_equal,            px_equal,     px_again,    first.encode, first.prefix,
        first.sample,    first.decode, warm.encode,          warm.prefix,  warm.sample, warm.decode, nograph.sample,
        px_nograph,      twin_ms[0],
        twin_ms[1],      twin_ms[2],   pass,
    });
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    return if (pass) 0 else 1;
}

fn sameAs(p: *qi.pipeline.Pipeline, a: std.mem.Allocator, io: std.Io, cap: *const qi.capture.Capture, op: []const u8, role: []const u8, ptr: u64, bytes: u64) !bool {
    const ref = cap.ops[cap.find(op, 0) orelse return false].out(role) orelse return false;
    if (ref.bytes() != bytes) return false;
    const want = try a.alloc(u8, bytes);
    try cap.blob(io, ref, want);
    const got = try a.alloc(u8, bytes);
    try p.d.check(p.d.api.cuMemcpyDtoH_v2(got.ptr, ptr, bytes), "cuMemcpyDtoH");
    return std.mem.eql(u8, got, want);
}
