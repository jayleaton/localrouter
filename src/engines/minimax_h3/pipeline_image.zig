//! MiniMax H3-Image text to image end to end in one process (docs/dev/H3-IMAGE.md; ComfyUI e9afbccc's `KSampler euler /
//! simple` at cfg 1 on `MiniMaxH3ImageConditioning`): the prompt's raw token ids, the 32B text encoder (layer 50, as the
//! video model), the DiT's refined context, `steps` euler steps over the PDD schedule (`pdd.zig`) from the portable noise,
//! one DiT forward a step with the step's PDD head, then the image VAE decoder's 256-px tiles. Three inputs: the bf16 pack
//! (`stk_twin.h3i.build`: the DiT and `tokenizer.json`), the text encoder checkpoint and the image VAE checkpoint as
//! published. Heap-allocated: its parts point at each other.
//!
//! Memory (*estimates*, ROCM.md section 4, until a gfx1151 run measures them): the DiT's bf16 pack 40.3 GB (the
//! checkpoint's 40.2 GB with the block norms and adaLN linears widened to fp32, as the video pack stores them), the text
//! encoder 15.7 GB, the VAE decoder 0.81 GB; scratch for `Limits` (the DiT's about 230 KB a row of [text | image], the
//! encoder's, a VAE tile's 25 MB and the sampler's buffers) allocated at `create`; a decode's canvas, strips and tail per
//! request (`requestBytes`).

const std = @import("std");
const gpu = @import("gpu");
const qi = @import("qwen_image");
const launch = @import("launch.zig");
const launch_image = @import("launch_image.zig");
const dit_mod = @import("dit.zig");
const pdd = @import("pdd.zig");
const te32 = @import("te32.zig");
const vae_video = @import("vae_video.zig");
const vae_image = @import("vae_image.zig");
const tok_mod = @import("tokenizer.zig");
const kernels = @import("minimax_kernels");

pub const Probe = dit_mod.Probe;
pub const Io = dit_mod.Io;

/// `pack`: the bf16 pack directory; `te`: qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors; `vae`:
/// minimax_h3_image_vae_fp16.safetensors.
pub const Paths = struct { pack: []const u8, te: []const u8, vae: []const u8 };

/// The model name and precision a pack must declare (`stk_twin.h3i.build`'s manifest), with its `pdd_sigmas`.
pub const pack_model = "minimax-h3-image";
pub const pack_precision = "bf16";

/// The largest request: `pixels` an image (sides multiples of 32) and `tokens` prompt tokens.
pub const Limits = struct { pixels: u32 = 1024 * 1024, tokens: u32 = 1536 };

/// Pixels a DiT row covers: a 2 x 2 patch of the /16 latent.
const pixels_a_row = 32 * 32;

/// The DiT's rows for `l`: the prompt and the image.
pub fn ditRows(l: Limits) u32 {
    return l.tokens + l.pixels / pixels_a_row;
}

/// What a request turns into: the latent [24, 1, lh, lw] (n values) and the uint8 image.
pub const Shapes = struct {
    lh: u32,
    lw: u32,
    n: u64, // 24 * lh * lw
    rows: u32, // the image's DiT rows
    pixel_bytes: u64, // height * width * 3

    pub fn of(width: u32, height: u32) !Shapes {
        if (width == 0 or height == 0 or width % 32 != 0 or height % 32 != 0) return error.SizeNotMultipleOf32;
        const lh = height / 16;
        const lw = width / 16;
        return .{ .lh = lh, .lw = lw, .n = 24 * @as(u64, lh) * lw, .rows = (lh / 2) * (lw / 2), .pixel_bytes = @as(u64, width) * height * 3 };
    }
};

/// Whether an image fits the buffers allocated for `l`.
pub fn fits(l: Limits, width: u32, height: u32) bool {
    const sh = Shapes.of(width, height) catch return false;
    return @as(u64, width) * height <= l.pixels and sh.rows <= l.pixels / pixels_a_row;
}

/// The sampler's buffers for `l`: the fp32 state and denoised, the bf16 model input and velocity, the bf16 text, the
/// fp16 latent for the VAE and the uint8 image.
fn bufferSizes(l: Limits) [7]u64 {
    const n: u64 = 24 * @as(u64, l.pixels / 256);
    return .{ 4 * n, 4 * n, 2 * n, 2 * n, @as(u64, l.tokens) * te32.dim * 2, 2 * n, 3 * @as(u64, l.pixels) };
}

/// The device bytes `create` allocates for `l` besides the weights.
pub fn scratchBytes(l: Limits) u64 {
    var total = dit_mod.Dit.imageScratchBytes(ditRows(l), l.tokens) + dit_mod.sumBuffers(te32.TextEncoder.scratch(l.tokens)) + vae_image.tileBytes();
    for (bufferSizes(l)) |n| total += @max(n, 256);
    return total;
}

/// The memory a request takes beyond `scratchBytes`, freed by the time it returns: the decode's device scratch and the
/// latent on the host in fp32 and fp16.
pub fn requestBytes(width: u32, height: u32) !u64 {
    const sh = try Shapes.of(width, height);
    return vae_image.decodeBytes(sh.lh, sh.lw) + sh.n * (4 + 2);
}

/// The pack's manifest names the H3-Image model, its bf16 precision and the checkpoint's `pdd_sigmas` (which the PDD
/// schedule and the head bank are built on), bit for bit.
pub fn checkManifest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(16 << 20));
    return manifestOk(a, text);
}

fn manifestOk(a: std.mem.Allocator, text: []const u8) !void {
    const M = struct { model: []const u8 = "", precision: []const u8 = "", pdd_sigmas: []const f64 = &.{} };
    const m = try std.json.parseFromSliceLeaky(M, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    if (!std.mem.eql(u8, m.model, pack_model)) return error.NotAnH3ImagePack;
    if (!std.mem.eql(u8, m.precision, pack_precision)) return error.NotABf16Pack;
    if (!std.mem.eql(f64, m.pdd_sigmas, &pdd.pdd_sigmas)) return error.PddSigmasMismatch;
}

pub const Progress = struct {
    ctx: *anyopaque,
    report: *const fn (ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void,
};

pub const Result = struct { width: u32, height: u32, tokens: u32 };

pub const Pipeline = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    paths: Paths,
    limits: Limits,
    d: gpu.Driver,
    ctx: gpu.Context,
    s: gpu.Stream,
    ops: qi.ops_launch.Ops,
    tk: qi.ops_launch.TeOps,
    hops: launch.Ops,
    kitchen: launch.Kitchen,
    iops: launch_image.Ops,
    te_ops: te32.Ops,
    v_ops: vae_video.Ops,
    pack: qi.pack.Pack,
    te_pack: qi.pack.Pack,
    tok: tok_mod.Tokenizer,
    w: dit_mod.Weights,
    dit: dit_mod.Dit,
    te: te32.TextEncoder,
    vae: vae_image.Decoder,
    bufs: [7]gpu.DeviceBuffer,
    /// The run's light ops under the twin's names when set: `noise`, `step.{i}`, `latents`, `pixels`.
    probe: ?Probe = null,
    ids: []u32 = &.{},

    /// Loads everything for requests up to `limits`.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, paths: Paths, limits: Limits) !*Pipeline {
        try checkManifest(gpa, io, paths.pack);
        const p = try gpa.create(Pipeline);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.io = io;
        p.paths = paths;
        p.limits = limits;
        p.probe = null;
        p.ids = &.{};
        p.d = try qi.memcap.open();
        errdefer p.d.close();
        p.ctx = try gpu.Context.init(&p.d, 0);
        errdefer p.ctx.deinit();
        p.s = try gpu.Stream.init(&p.d, false);
        errdefer p.s.deinit();
        p.ops = try qi.ops_launch.Ops.load(&p.d, qi.kernels.ops, qi.kernels.attention);
        errdefer p.ops.unload();
        p.tk = try qi.ops_launch.TeOps.load(&p.d, qi.kernels.te, qi.kernels.gemm);
        errdefer p.tk.unload();
        p.hops = try launch.Ops.load(&p.d, kernels.ops);
        errdefer p.hops.unload();
        p.kitchen = try launch.Kitchen.load(&p.d, kernels.kitchen); // its rms_rope (the attention is the bf16 one)
        errdefer p.kitchen.unload();
        p.iops = try launch_image.Ops.load(&p.d, kernels.h3_image);
        errdefer p.iops.unload();
        p.te_ops = try te32.Ops.load(&p.d, kernels.te32);
        errdefer p.te_ops.unload();
        p.v_ops = try vae_video.Ops.load(&p.d, kernels.gemm_f16, kernels.vae_video);
        errdefer p.v_ops.unload();

        p.pack = try qi.pack.Pack.open(gpa, io, paths.pack);
        errdefer p.pack.close(io);
        p.te_pack = try qi.pack.Pack.openFile(gpa, io, paths.te);
        errdefer p.te_pack.close(io);
        p.tok = try tok_mod.Tokenizer.load(gpa, io, paths.pack);
        errdefer p.tok.deinit();

        var up = try qi.upload.Uploader.init(&p.d, io, p.s);
        defer up.deinit();
        p.w = try dit_mod.Weights.loadImage(gpa, io, &p.d, &p.pack, &up);
        errdefer p.w.deinit();
        const kk: dit_mod.Kernels = .{ .h3 = &p.hops, .kitchen = &p.kitchen, .te = &p.tk, .ops = &p.ops, .nv = null, .major = 0, .minor = 0, .image = &p.iops };
        p.dit = try dit_mod.Dit.initImage(gpa, &p.d, kk, &p.w, p.s, ditRows(limits), limits.tokens);
        errdefer p.dit.deinit();
        p.te = try te32.TextEncoder.init(gpa, io, &p.d, &p.te_ops, p.s, &p.te_pack, &up, limits.tokens);
        errdefer p.te.deinit();
        p.vae = try vae_image.Decoder.init(gpa, io, &p.d, &p.v_ops, &p.iops, p.s, paths.vae, &up);
        errdefer p.vae.deinit();
        var made: usize = 0;
        errdefer for (p.bufs[0..made]) |*b| b.free();
        for (&p.bufs, bufferSizes(limits)) |*b, len| {
            b.* = try gpu.DeviceBuffer.alloc(&p.d, @intCast(@max(len, 256)));
            made += 1;
        }
        try p.s.synchronize();
        return p;
    }

    pub fn destroy(p: *Pipeline) void {
        for (&p.bufs) |*b| b.free();
        p.vae.deinit();
        p.te.deinit();
        p.dit.deinit();
        p.w.deinit();
        p.tok.deinit();
        p.te_pack.close(p.io);
        p.pack.close(p.io);
        p.v_ops.unload();
        p.te_ops.unload();
        p.iops.unload();
        p.kitchen.unload();
        p.hops.unload();
        p.tk.unload();
        p.ops.unload();
        p.s.deinit();
        p.ctx.deinit();
        p.d.close();
        if (p.ids.len > 0) p.gpa.free(p.ids);
        const gpa = p.gpa;
        gpa.destroy(p);
    }

    fn pre(p: *Pipeline, name: []const u8, ins: []const Io) !void {
        if (p.probe) |pr| try pr.before(pr.ctx, name, ins);
    }

    fn post(p: *Pipeline, name: []const u8, outs: []const Io) !void {
        if (p.probe) |pr| try pr.after(pr.ctx, name, outs);
    }

    /// One image: `rgb_out` gets the uint8 pixels [height, width, 3].
    pub fn generate(p: *Pipeline, prompt: []const u8, width: u32, height: u32, steps: u32, seed: u64, rgb_out: []u8, progress: ?Progress) !Result {
        const sh = try Shapes.of(width, height);
        if (steps == 0 or steps > 1000) return error.BadSteps;
        if (!fits(p.limits, width, height)) return error.ImageTooLarge;
        if (rgb_out.len != sh.pixel_bytes) return error.BadImageBuffer;
        const gpa = p.gpa;
        p.dit.probe = null;

        // ---- the prompt: its raw ids (no template), the encoder, the bf16 hand-off, condition_proj and the refiner
        const ids = try p.tok.ids(gpa, prompt);
        if (p.ids.len > 0) gpa.free(p.ids);
        p.ids = ids;
        if (ids.len > p.limits.tokens) return error.PromptTooLong;
        const l: u32 = @intCast(ids.len);
        try p.pre("text_encoder", &.{});
        const hidden = try p.te.forward(ids, 0);
        const text = try p.bufs[4].at(0);
        try p.hops.toBf16(p.s, hidden, text, @as(u64, l) * te32.dim);
        try p.post("text_encoder", &.{.{ .role = "context", .ptr = text, .bytes = @as(u64, l) * te32.dim * 2 }});
        try p.dit.prepare(text, l);

        // ---- the schedule and the state: portable noise in place of torch.randn, scaled by the first sigma (CONST)
        var tab: [1000]f32 = undefined;
        pdd.table(&pdd.pdd_sigmas, &tab);
        const sig = try gpa.alloc(f32, steps + 1);
        defer gpa.free(sig);
        pdd.sigmas(&tab, sig);
        const n = sh.n;
        const x = try p.bufs[0].at(0);
        {
            const z = try gpa.alloc(f32, @intCast(n));
            defer gpa.free(z);
            qi.noise.fill(seed, z);
            const s0 = sig[0];
            for (z) |*v| v.* = s0 * v.* + (1.0 - s0) * 0.0; // noise_scaling: sigma * noise + (1 - sigma) * the empty latent
            try p.dit.upload(x, std.mem.sliceAsBytes(z));
        }
        try p.pre("noise", &.{});
        try p.post("noise", &.{.{ .role = "x", .ptr = x, .bytes = n * 4 }});

        // ---- the steps: one forward each (cfg 1), the CONST denoise, euler
        const den = try p.bufs[1].at(0);
        const xb = try p.bufs[2].at(0);
        const vel = try p.bufs[3].at(0);
        var nb: [24]u8 = undefined;
        for (0..steps) |i| {
            const st = pdd.step(&pdd.pdd_sigmas, sig, i);
            const e = pdd.euler(sig, i);
            const name = try std.fmt.bufPrint(&nb, "step.{d}", .{i});
            try p.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = n * 4 }});
            try p.hops.toBf16(p.s, x, xb, n);
            try p.dit.stepImage(xb, st, sh.lh, sh.lw, vel);
            try p.hops.denoise(p.s, x, vel, e.sigma, den, n);
            try p.hops.euler32(p.s, x, den, e.sigma, e.dt, n);
            try p.post(name, &.{ .{ .role = "vel", .ptr = vel, .bytes = n * 2 }, .{ .role = "den", .ptr = den, .bytes = n * 4 }, .{ .role = "x", .ptr = x, .bytes = n * 4 } });
            if (progress) |pr| {
                try p.s.synchronize();
                pr.report(pr.ctx, "denoise", @intCast(i + 1), steps);
            }
        }
        try p.pre("latents", &.{});
        try p.post("latents", &.{.{ .role = "x", .ptr = x, .bytes = n * 4 }});

        // ---- the VAE: the fp32 state to fp16 (the VAE's own cast, round to nearest even), the tiles, uint8
        {
            try p.s.synchronize();
            const f32s = try gpa.alloc(f32, @intCast(n));
            defer gpa.free(f32s);
            try p.d.check(p.d.api.cuMemcpyDtoH_v2(f32s.ptr, x, n * 4), "cuMemcpyDtoH");
            const f16s = try gpa.alloc(f16, @intCast(n));
            defer gpa.free(f16s);
            for (f16s, f32s) |*o, v| o.* = @floatCast(v);
            const z16 = try p.bufs[5].at(0);
            try p.dit.upload(z16, std.mem.sliceAsBytes(f16s));
            const px = try p.bufs[6].at(0);
            p.vae.rec.probe = p.probe;
            try p.vae.decode(z16, sh.lh, sh.lw, px);
            try p.s.synchronize();
            try p.d.check(p.d.api.cuMemcpyDtoH_v2(rgb_out.ptr, px, rgb_out.len), "cuMemcpyDtoH");
            try p.pre("pixels", &.{});
            try p.post("pixels", &.{.{ .role = "y", .ptr = px, .bytes = rgb_out.len }});
        }
        if (progress) |pr| pr.report(pr.ctx, "decode", 1, 1);
        return .{ .width = width, .height = height, .tokens = l };
    }
};

test "the pack's manifest: the H3-Image model, bf16, the checkpoint's pdd_sigmas bit for bit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try manifestOk(a,
        \\{"format": "stk-pack/1", "model": "minimax-h3-image", "precision": "bf16", "kinds": {"L0.qkv": "bf16"},
        \\ "pdd_sigmas": [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0], "tensors": {}}
    );
    try std.testing.expectError(error.NotAnH3ImagePack, manifestOk(a,
        \\{"model": "minimax-h3-fl2va", "precision": "nvfp4", "pdd_sigmas": []}
    ));
    try std.testing.expectError(error.NotABf16Pack, manifestOk(a,
        \\{"model": "minimax-h3-image", "precision": "nvfp4"}
    ));
    try std.testing.expectError(error.PddSigmasMismatch, manifestOk(a,
        \\{"model": "minimax-h3-image", "precision": "bf16", "pdd_sigmas": [1.0, 0.967, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]}
    ));
}

test "shapes and limits: 32-pixel sides, one DiT row per 32 x 32 cell, the buffers follow the area" {
    const sh = try Shapes.of(1024, 768);
    try std.testing.expectEqual(@as(u32, 48), sh.lh);
    try std.testing.expectEqual(@as(u32, 64), sh.lw);
    try std.testing.expectEqual(@as(u32, 24 * 32), sh.rows);
    try std.testing.expectError(error.SizeNotMultipleOf32, Shapes.of(1000, 768));
    const l: Limits = .{};
    try std.testing.expect(fits(l, 1024, 1024));
    try std.testing.expect(fits(l, 2048, 512));
    try std.testing.expect(!fits(l, 1056, 1024));
    try std.testing.expectEqual(@as(u32, 1536 + 1024), ditRows(l));
    try std.testing.expect(scratchBytes(.{ .pixels = 2048 * 2048 }) > scratchBytes(l));
    try std.testing.expect(try requestBytes(1024, 1024) > vae_image.decodeBytes(64, 64));
}

test "the pipeline's code type-checks (no GPU here)" {
    _ = &Pipeline.create;
    _ = &Pipeline.generate;
    _ = &Pipeline.destroy;
}
