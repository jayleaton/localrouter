//! MiniMax H3-Image text to image end to end in one process (docs/dev/H3-IMAGE.md; ComfyUI e9afbccc's `KSampler euler /
//! simple` at cfg 1 on `MiniMaxH3ImageConditioning`): the prompt's raw token ids, the 32B text encoder (layer 50, as the
//! video model), the DiT's refined context, `steps` euler steps over the PDD schedule (`pdd.zig`) from the portable noise,
//! one DiT forward a step with the step's PDD head, then the image VAE decoder's 256-px tiles. Three inputs: the bf16 pack
//! (`stk_twin.h3i.build`: the DiT and `tokenizer.json`), the text encoder checkpoint and the image VAE checkpoint as
//! published. Heap-allocated: its parts point at each other. Edits (`edit`, H3-IMAGE.md "Decided in PR 1.7") add the
//! pictures' host path (`image.zig`), the vision tower and the VAE encoder loaded for the request, and the DiT's
//! reference rows.
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
const image = @import("image.zig");
const edit_mod = @import("edit.zig");
const vae_enc = @import("vae_video_enc.zig");
const vision = @import("te_vision.zig");
const kernels = @import("minimax_kernels");

pub const Probe = dit_mod.Probe;
pub const Io = dit_mod.Io;

/// `pack`: the bf16 pack directory; `te`: qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors; `vae`:
/// minimax_h3_image_vae_fp16.safetensors.
pub const Paths = struct { pack: []const u8, te: []const u8, vae: []const u8 };

/// The model name and precision a pack must declare (`stk_twin.h3i.build`'s manifest), with its `pdd_sigmas`.
pub const pack_model = "minimax-h3-image";
pub const pack_precision = "bf16";

/// The largest request: `pixels` an image from text (sides multiples of 32) and `tokens` prompt tokens; for edits
/// (`edit`), `pictures` input pictures at most (0: no edits), `edit_pixels` the largest target and the largest VAE copy
/// of a picture, and `vision_rows` the vision rows of all the pictures' Qwen copies ((w/32)(h/32) each).
///
/// `edit_pixels`' default, 2176 x 1024, holds every whole-scene edit's 2-megapixel target (ImageScaleToTotalPixels'
/// sides rounded to 32 reach 2 141 184 pixels) and VAE copies of short edge 1024 up to 2:1; `vision_rows`' default holds
/// five square pictures (1024 rows each), fewer of other shapes. Both are this port's defaults, not #157's.
pub const Limits = struct {
    pixels: u32 = 1024 * 1024,
    tokens: u32 = 1536,
    pictures: u32 = 0,
    edit_pixels: u32 = default_edit_pixels,
    vision_rows: u32 = edit_mod.max_pictures * 1024,
};

pub const default_edit_pixels = 2176 * 1024;

/// Pixels a DiT row covers: a 2 x 2 patch of the /16 latent.
const pixels_a_row = 32 * 32;
/// The text tokens a picture's label ("Picture k: ") and its two markers take at most.
const label_tokens = 16;

/// The text encoder's tokens for `l`: the prompt, and with edits the pictures' vision rows, labels and markers.
pub fn textTokens(l: Limits) u32 {
    return l.tokens + if (l.pictures > 0) l.vision_rows + l.pictures * label_tokens else 0;
}

/// The largest target: a text-to-image's or an edit's.
fn maxTarget(l: Limits) u32 {
    return if (l.pictures > 0) @max(l.pixels, l.edit_pixels) else l.pixels;
}

/// The DiT's rows for `l`: the text, the references (edits) and the image.
pub fn ditRows(l: Limits) u32 {
    const refs = if (l.pictures > 0) l.pictures * (l.edit_pixels / pixels_a_row) else 0;
    return textTokens(l) + refs + maxTarget(l) / pixels_a_row;
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

/// Whether an edit of `plan` fits the buffers allocated for `l`: the pictures, the target, each VAE copy and the
/// vision rows (the prompt's tokens are checked once tokenized).
pub fn editFits(l: Limits, plan: *const edit_mod.Plan) !void {
    if (l.pictures == 0) return error.NoEdits;
    if (plan.n > l.pictures) return error.TooManyPictures;
    const t = plan.target;
    _ = try Shapes.of(t.w, t.h);
    if (@as(u64, t.w) * t.h > maxTarget(l)) return error.ImageTooLarge;
    var rows: u64 = 0;
    for (plan.list()) |q| {
        if (@as(u64, q.vae.w) * q.vae.h > l.edit_pixels) return error.PictureTooLarge;
        rows += (q.vision.w / 32) * (q.vision.h / 32);
    }
    if (rows > l.vision_rows) return error.PicturesTooLarge;
}

/// The memory an edit of `plan` takes beyond `scratchBytes` at its largest phase (*estimates*), freed by the time it
/// returns: the vision tower with the pictures' features; the encoder with the features and latents; the sampling with
/// the references' rows; the decode with its fp32 copy. The host's copies of the pictures are `editHostBytes`.
pub fn editRequestBytes(plan: *const edit_mod.Plan) !u64 {
    const t = try Shapes.of(plan.target.w, plan.target.h);
    var patches: u32 = 0;
    var features: u64 = 0;
    var latents: u64 = 0;
    var rows: u64 = 0;
    var most_rows: u64 = 0;
    var most_vae: [2]u32 = .{ 32, 32 };
    for (plan.list()) |q| {
        const p = (q.vision.w / 16) * (q.vision.h / 16);
        patches = @max(patches, p);
        features += 4 * @as(u64, p / 4) * te32.dim * 4;
        const r = @as(u64, q.vae.w / 32) * (q.vae.h / 32);
        latents += 24 * 4 * r * 4;
        rows += r;
        most_rows = @max(most_rows, r);
        if (@as(u64, q.vae.w) * q.vae.h > @as(u64, most_vae[0]) * most_vae[1]) most_vae = .{ q.vae.w, q.vae.h };
    }
    const vae_px: u64 = @as(u64, most_vae[0]) * most_vae[1];
    const seeing = vision.VisionEncoder.deviceBytes(patches) + features;
    const encoding = features + latents + vae_enc.Encoder.deviceBytes(most_vae[1], most_vae[0]) + vae_px * 3 * 4 + vae_px / 256 * 48 * 2;
    const sampling = dit_mod.Dit.refBytes(rows, most_rows) + latents;
    const decoding = (try requestBytes(plan.target.w, plan.target.h)) + t.pixel_bytes * 4;
    return @max(seeing, encoding, sampling, decoding);
}

/// The host bytes an edit of `plan` holds besides `pictures` (*estimate*): each picture in fp32 with its mask, the
/// first one's source, the Qwen and VAE copies, and the composite's four fp32 images at the target.
pub fn editHostBytes(plan: *const edit_mod.Plan) u64 {
    var total: u64 = 0;
    for (plan.list()) |q| total += 16 * @as(u64, q.loaded.w) * q.loaded.h + 12 * (@as(u64, q.source.w) * q.source.h + @as(u64, q.vision.w) * q.vision.h + @as(u64, q.vae.w) * q.vae.h);
    return total + 4 * 12 * @as(u64, plan.target.w) * plan.target.h;
}

/// The sampler's buffers for `l`: the fp32 state and denoised, the bf16 model input and velocity, the bf16 text, the
/// fp16 latent for the VAE and the uint8 image.
fn bufferSizes(l: Limits) [7]u64 {
    const px = maxTarget(l);
    const n: u64 = 24 * @as(u64, px / 256);
    return .{ 4 * n, 4 * n, 2 * n, 2 * n, @as(u64, textTokens(l)) * te32.dim * 2, 2 * n, 3 * @as(u64, px) };
}

/// The device bytes `create` allocates for `l` besides the weights.
pub fn scratchBytes(l: Limits) u64 {
    var total = dit_mod.Dit.imageScratchBytes(ditRows(l), textTokens(l)) + dit_mod.sumBuffers(te32.TextEncoder.scratch(textTokens(l))) + vae_image.tileBytes();
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
        p.dit = try dit_mod.Dit.initImage(gpa, &p.d, kk, &p.w, p.s, ditRows(limits), textTokens(limits));
        errdefer p.dit.deinit();
        p.te = try te32.TextEncoder.init(gpa, io, &p.d, &p.te_ops, p.s, &p.te_pack, &up, textTokens(limits));
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

        try p.sample(sh, steps, seed, progress);
        try p.decode(sh, rgb_out, 0);
        if (progress) |pr| pr.report(pr.ctx, "decode", 1, 1);
        return .{ .width = width, .height = height, .tokens = l };
    }

    /// The schedule and the state (portable noise in place of torch.randn, scaled by the first sigma: CONST), then the
    /// steps: one forward each (cfg 1), the CONST denoise, euler. The context (and an edit's references) are prepared.
    fn sample(p: *Pipeline, sh: Shapes, steps: u32, seed: u64, progress: ?Progress) !void {
        const gpa = p.gpa;
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
    }

    /// The VAE: the fp32 state to fp16 (the VAE's own cast, round to nearest even), the tiles, uint8 into `rgb_out`;
    /// the fp32 pixels into `out_f32` too unless 0.
    fn decode(p: *Pipeline, sh: Shapes, rgb_out: []u8, out_f32: u64) !void {
        const gpa = p.gpa;
        const n = sh.n;
        const x = try p.bufs[0].at(0);
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
        try p.vae.decodeTo(z16, sh.lh, sh.lw, px, out_f32);
        try p.s.synchronize();
        try p.d.check(p.d.api.cuMemcpyDtoH_v2(rgb_out.ptr, px, rgb_out.len), "cuMemcpyDtoH");
        try p.pre("pixels", &.{});
        try p.post("pixels", &.{.{ .role = "y", .ptr = px, .bytes = rgb_out.len }});
    }

    /// The size `edit` makes for these pictures (each one's width and height), a mask or not, and the request's size
    /// (0 x 0: none).
    pub fn editSize(loaded: []const edit_mod.Size, masked: bool, width: u32, height: u32) !edit_mod.Size {
        return (try edit_mod.Plan.of(loaded, masked, width, height)).target;
    }

    /// An edit (#157's edit workflows; H3-IMAGE.md section 1.7): `pictures` 1 to `limits.pictures`, the first the one
    /// edited; `mask` an image of the first picture's size whose alpha is 0 where the edit goes (LoadImage's mask
    /// 1 - alpha / 255), null for a whole-scene edit; `width` x `height` the result (0 x 0: the first picture at
    /// ImageScaleToTotalPixels' 1 (masked) or 2 megapixels). Each picture goes through the vision tower ("Picture k: "
    /// before it) and the VAE encoder (the posterior drawn at seed 42) as a DiT reference; the decode is composited over
    /// the first picture's 8-bit copy at the target where the mask says. `rgb_out` gets the uint8 pixels at `editSize`.
    pub fn edit(p: *Pipeline, prompt: []const u8, pictures: []const Picture, mask: ?[]const u8, width: u32, height: u32, steps: u32, seed: u64, rgb_out: []u8, progress: ?Progress) !Result {
        const max = edit_mod.max_pictures;
        if (pictures.len == 0) return error.NoPicture;
        if (pictures.len > max) return error.TooManyPictures;
        var loaded: [max]edit_mod.Size = undefined;
        for (pictures, loaded[0..pictures.len]) |q, *s| {
            if (q.width == 0 or q.height == 0 or q.rgba.len != @as(u64, q.width) * q.height * 4) return error.BadPicture;
            s.* = .{ .w = q.width, .h = q.height };
        }
        const plan = try edit_mod.Plan.of(loaded[0..pictures.len], mask != null, width, height);
        try editFits(p.limits, &plan);
        const n = plan.n;
        const t = plan.target;
        const sh = try Shapes.of(t.w, t.h);
        if (steps == 0 or steps > 1000) return error.BadSteps;
        if (rgb_out.len != sh.pixel_bytes) return error.BadImageBuffer;
        const gpa = p.gpa;
        p.dit.probe = null;
        p.dit.releaseRefs();

        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        // ---- the mask: LoadImage's 1 - alpha / 255 at the first picture's size, bilinear to the target (ATen's CPU)
        var mask_t: []f32 = &.{};
        if (mask) |mk| {
            const q = pictures[0];
            if (mk.len != q.rgba.len) return error.BadMask;
            const rgb = try a.alloc(f32, @as(usize, q.width) * q.height * 3);
            const m = try a.alloc(f32, @as(usize, q.width) * q.height);
            try image.loadRgba(mk, 4, rgb, m);
            for (m) |v| {
                if (v > 0) break;
            } else return error.MaskKeepsEverything;
            mask_t = try a.alloc(f32, @as(usize, t.w) * t.h);
            try image.bilinear(m, q.width, q.height, mask_t, t.w, t.h);
        }

        // ---- the pictures on the host: LoadImage, node 11's area scale of the first, each one's Qwen and VAE copies
        var vis: [max][]f32 = undefined;
        var vae: [max][]f32 = undefined;
        var first: []f32 = &.{};
        for (pictures, plan.list(), 0..) |q, pc, i| {
            const img = try a.alloc(f32, @as(usize, q.width) * q.height * 3);
            const alpha = try a.alloc(f32, @as(usize, q.width) * q.height);
            try image.loadRgba(q.rgba, 4, img, alpha);
            var src = img;
            if (i == 0) {
                src = try a.alloc(f32, @as(usize, pc.source.w) * pc.source.h * 3);
                image.area(img, q.width, q.height, src, pc.source.w, pc.source.h);
                first = src;
            }
            vis[i] = try a.alloc(f32, @as(usize, pc.vision.w) * pc.vision.h * 3);
            try image.resizeTo(gpa, src, pc.source.w, pc.source.h, vis[i], pc.vision.w, pc.vision.h);
            vae[i] = try a.alloc(f32, @as(usize, pc.vae.w) * pc.vae.h * 3);
            try image.resizeTo(gpa, src, pc.source.w, pc.source.h, vae[i], pc.vae.w, pc.vae.h);
        }

        var up = try qi.upload.Uploader.init(&p.d, p.io, p.s);
        defer up.deinit();

        // ---- the vision tower over each Qwen copy, the features copied into buffers of their own
        var feats: [max]vision.Features = undefined;
        var fbufs: [max]?gpu.DeviceBuffer = @splat(null);
        defer for (&fbufs) |*slot| if (slot.*) |*b| {
            p.s.synchronize() catch {};
            b.free();
            slot.* = null;
        };
        {
            var vops = try vision.Ops.load(&p.d, kernels.te_vision);
            defer vops.unload();
            var most: u32 = 0;
            for (plan.list()) |pc| most = @max(most, (pc.vision.w / 16) * (pc.vision.h / 16));
            var visual = try vision.VisionEncoder.init(gpa, p.io, &p.d, &vops, &p.hops, &p.te_ops, p.s, &p.te_pack, &up, most);
            defer visual.deinit();
            defer p.s.synchronize() catch {};
            for (plan.list(), 0..) |pc, i| {
                const f = try visual.forward(vis[i], pc.vision.w, pc.vision.h);
                const bytes: u64 = @as(u64, f.rows()) * te32.dim * 4;
                fbufs[i] = try gpu.DeviceBuffer.alloc(&p.d, @intCast(@max(4 * bytes, 256)));
                const base = try fbufs[i].?.at(0);
                var g = f;
                g.merged = base;
                try p.d.check(p.d.api.cuMemcpyDtoDAsync_v2(base, f.merged, bytes, p.s.handle), "cuMemcpyDtoDAsync");
                for (0..3) |j| {
                    g.deepstack[j] = base + (j + 1) * bytes;
                    try p.d.check(p.d.api.cuMemcpyDtoDAsync_v2(g.deepstack[j], f.deepstack[j], bytes, p.s.handle), "cuMemcpyDtoDAsync");
                }
                feats[i] = g;
            }
        }

        // ---- the tokens ("Picture k: ", the markers, the vision rows; then the prompt), the encoder, the context
        var rows: [max]u32 = undefined;
        var starts: [max]u32 = undefined;
        for (feats[0..n], rows[0..n]) |f, *r| r.* = f.rows();
        const it = try p.tok.editIds(gpa, prompt, rows[0..n], starts[0..n]);
        defer gpa.free(it.tags);
        if (p.ids.len > 0) gpa.free(p.ids);
        p.ids = it.ids;
        if (it.ids.len > textTokens(p.limits)) return error.PromptTooLong;
        const l: u32 = @intCast(it.ids.len);
        var placed: [max]te32.Placed = undefined;
        for (feats[0..n], starts[0..n], placed[0..n]) |f, s, *pl| pl.* = .{ .f = f, .start = s };
        try p.pre("text_encoder", &.{});
        const hidden = try p.te.forwardImages(it.ids, placed[0..n], 0);
        const text = try p.bufs[4].at(0);
        try p.hops.toBf16(p.s, hidden, text, @as(u64, l) * te32.dim);
        try p.post("text_encoder", &.{.{ .role = "context", .ptr = text, .bytes = @as(u64, l) * te32.dim * 2 }});
        try p.dit.prepare(text, l);
        try p.s.synchronize();
        for (&fbufs) |*slot| if (slot.*) |*b| {
            b.free();
            slot.* = null;
        };

        // ---- the references: each VAE copy's moments, the posterior at seed 42 (portable noise), normalized
        var zbufs: [max]?gpu.DeviceBuffer = @splat(null);
        defer for (&zbufs) |*slot| if (slot.*) |*b| {
            p.s.synchronize() catch {};
            b.free();
        };
        var refs: [max]dit_mod.Dit.Ref = undefined;
        {
            var mean: [24]f16 = undefined;
            var stdv: [24]f16 = undefined;
            vae_image.latentStats(&mean, &stdv);
            var eops = try vae_enc.Ops.load(&p.d, kernels.vae_enc);
            defer eops.unload();
            var enc = try vae_enc.Encoder.init(gpa, p.io, &p.d, &eops, &p.v_ops, p.s, p.paths.vae, &up);
            defer enc.deinit();
            var nb: [32]u8 = undefined;
            for (plan.list(), 0..) |pc, i| {
                const hz = pc.vae.h / 16;
                const wz = pc.vae.w / 16;
                const P: usize = @as(usize, hz) * wz;
                var input = try gpu.DeviceBuffer.fromHost(&p.d, std.mem.sliceAsBytes(vae[i]));
                defer {
                    p.s.synchronize() catch {};
                    input.free();
                }
                var mom = try gpu.DeviceBuffer.alloc(&p.d, @intCast(@max(48 * P * 2, 256)));
                defer {
                    p.s.synchronize() catch {};
                    mom.free();
                }
                try enc.encodeMoments(try input.at(0), pc.vae.h, pc.vae.w, try mom.at(0));
                try p.s.synchronize();
                const moments = try gpa.alloc(f16, 48 * P);
                defer gpa.free(moments);
                try p.d.check(p.d.api.cuMemcpyDtoH_v2(moments.ptr, try mom.at(0), moments.len * 2), "cuMemcpyDtoH");
                const noise = try gpa.alloc(f32, 24 * P);
                defer gpa.free(noise);
                qi.noise.fill(edit_mod.encode_seed, noise);
                const z = try gpa.alloc(f32, 24 * P);
                defer gpa.free(z);
                edit_mod.posterior(moments, &mean, &stdv, noise, z);
                zbufs[i] = try gpu.DeviceBuffer.fromHost(&p.d, std.mem.sliceAsBytes(z));
                refs[i] = .{ .z = try zbufs[i].?.at(0), .h = hz, .w = wz };
                const name = try std.fmt.bufPrint(&nb, "ref{d}.latent", .{i});
                try p.pre(name, &.{});
                try p.post(name, &.{.{ .role = "z", .ptr = refs[i].z, .bytes = z.len * 4 }});
            }
        }
        try p.dit.imageRefs(refs[0..n], seed, it.tags);
        defer p.dit.releaseRefs();

        // ---- the steps, the decode (its fp32 pixels kept for the composite), the composite, SaveImage's uint8
        try p.sample(sh, steps, seed, progress);
        if (mask == null) {
            try p.decode(sh, rgb_out, 0);
        } else {
            var f32buf = try gpu.DeviceBuffer.alloc(&p.d, @intCast(sh.pixel_bytes * 4));
            defer {
                p.s.synchronize() catch {};
                f32buf.free();
            }
            try p.decode(sh, rgb_out, try f32buf.at(0));
            const decoded = try a.alloc(f32, @intCast(sh.pixel_bytes));
            try p.d.check(p.d.api.cuMemcpyDtoH_v2(decoded.ptr, try f32buf.at(0), sh.pixel_bytes * 4), "cuMemcpyDtoH");
            const src = plan.list()[0].source;
            const dest = try a.alloc(f32, @intCast(sh.pixel_bytes));
            try image.scaleCenter(gpa, first, src.w, src.h, dest, t.w, t.h);
            const out = try a.alloc(f32, @intCast(sh.pixel_bytes));
            image.composite(dest, decoded, mask_t, out);
            for (rgb_out, out) |*o, v| o.* = image.toU8(v);
        }
        if (progress) |pr| pr.report(pr.ctx, "decode", 1, 1);
        return .{ .width = t.w, .height = t.h, .tokens = l };
    }
};

/// An edit's input picture: 8-bit RGBA [height, width, 4] (LoadImage's RGB; the alpha is not read).
pub const Picture = struct { rgba: []const u8, width: u32, height: u32 };

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

test "edit limits: pictures, the target, each VAE copy, the vision rows; the buffers grow with edits" {
    const off: Limits = .{};
    const on: Limits = .{ .pictures = 5 };
    const one = [_]edit_mod.Size{.{ .w = 1024, .h = 1024 }};
    var plan = try edit_mod.Plan.of(&one, false, 0, 0);
    try std.testing.expectError(error.NoEdits, editFits(off, &plan));
    try editFits(on, &plan); // 1440 x 1440 whole-scene
    try std.testing.expectEqual(edit_mod.Size{ .w = 1440, .h = 1440 }, plan.target);
    try std.testing.expectError(error.TooManyPictures, editFits(.{ .pictures = 1 }, &(try edit_mod.Plan.of(&.{ one[0], one[0] }, true, 0, 0))));
    // every whole-scene target ImageScaleToTotalPixels makes at 2 megapixels fits the default edit_pixels
    var w: u32 = 64;
    while (w <= 4096) : (w += 61) {
        var h: u32 = 64;
        while (h <= 4096) : (h += 67) {
            const s = image.totalPixelsSize(w, h, edit_mod.unmasked_megapixels, 32);
            if (@min(s[0], s[1]) >= 1024) try std.testing.expect(@as(u64, s[0]) * s[1] <= default_edit_pixels);
        }
    }
    // a 3:1 reference's VAE copy (short edge 1024) is past edit_pixels; a wide target is past the target's
    try std.testing.expectError(error.PictureTooLarge, editFits(on, &(try edit_mod.Plan.of(&.{ one[0], .{ .w = 3072, .h = 1024 } }, true, 0, 0))));
    try std.testing.expectError(error.ImageTooLarge, editFits(on, &(try edit_mod.Plan.of(&one, true, 2048, 2048))));
    try std.testing.expectError(error.SizeNotMultipleOf32, editFits(on, &(try edit_mod.Plan.of(&one, true, 1000, 1024))));
    // a 4:3 reference takes 1376 vision rows (1376 x 1024): five pictures with two of them are past the default
    // 5 x 1024, four are not
    const wide: edit_mod.Size = .{ .w = 1600, .h = 1200 };
    try std.testing.expectError(error.PicturesTooLarge, editFits(on, &(try edit_mod.Plan.of(&.{ one[0], wide, wide, one[0], one[0] }, true, 0, 0))));
    try editFits(on, &(try edit_mod.Plan.of(&.{ one[0], wide, wide, one[0] }, true, 0, 0)));
    try editFits(on, &(try edit_mod.Plan.of(&.{ one[0], one[0], one[0], one[0], one[0] }, true, 0, 0)));
    try std.testing.expectError(error.TooManyPictures, edit_mod.Plan.of(&.{ one[0], one[0], one[0], one[0], one[0], one[0] }, true, 0, 0));
    // the DiT holds the text, five references and the target at their largest
    try std.testing.expectEqual(@as(u32, 1536 + 5 * 1024 + 5 * 16 + 6 * 2176), ditRows(on));
    try std.testing.expect(scratchBytes(on) > scratchBytes(off));
    // an edit's request: more than a decode at its target, the vision tower or the encoder at their largest
    plan = try edit_mod.Plan.of(&.{ one[0], one[0], one[0] }, true, 0, 0);
    const req = try editRequestBytes(&plan);
    try std.testing.expect(req > try requestBytes(1024, 1024));
    try std.testing.expect(req >= vision.VisionEncoder.deviceBytes(64 * 64));
    try std.testing.expect(req >= vae_enc.Encoder.deviceBytes(1024, 1024));
    try std.testing.expect(editHostBytes(&plan) > 4 * 12 * 1024 * 1024);
}

test "the pipeline's code type-checks (no GPU here)" {
    _ = &Pipeline.create;
    _ = &Pipeline.generate;
    _ = &Pipeline.edit;
    _ = &Pipeline.destroy;
}
