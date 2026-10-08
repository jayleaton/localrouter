//! Qwen-Image 2.1 end to end in one process, as the twin's `QwenImage.generate` with LocalRouter's kernels: the
//! template's token ids, the text encoder, the DiT's prefix and `steps` Euler steps from the portable noise, the VAE
//! decoder, uint8 pixels (HWC). Three packs: the DiT's (NVFP4 or FP8), the text encoder's (with the tokenizer and the
//! scheduler's config) and the VAE's. Heap-allocated: its parts point at each other.

const std = @import("std");
const cuda = @import("cuda");
const kernels = @import("qwen_kernels");
const Pack = @import("pack.zig").Pack;
const ops_launch = @import("ops_launch.zig");
const Nvfp4 = @import("nvfp4_exec.zig").Nvfp4;
const triton_k = @import("triton_k.zig");
const W = @import("weights.zig");
const Dit = @import("dit.zig").Dit;
const te_mod = @import("te.zig");
const Vae = @import("vae.zig").Vae;
const VaeOps = @import("vae.zig").VaeOps;
const sampler = @import("sampler.zig");
const noise = @import("noise.zig");
const upload_mod = @import("upload.zig");
const smath = @import("smath.zig");

pub const Packs = struct { dit: []const u8, te: []const u8, vae: []const u8 };

/// Where tfimage's Triton kernels come from: given specs (a capture's), or the DiT pack's `triton/` for this GPU.
pub const TritonSource = union(enum) { specs: [3]?triton_k.Spec, pack };

/// Wall times of the last `generate`, in milliseconds (the stream synchronized at each boundary).
pub const Times = struct { encode: f64 = 0, prefix: f64 = 0, sample: f64 = 0, decode: f64 = 0 };

/// Where `create`'s time went, in milliseconds (the stream synchronized at each boundary), and the uploader's
/// counters: what a slow load is made of.
pub const LoadTimes = struct {
    context: f64 = 0, // driver, context, stream
    kernels: f64 = 0, // fatbins and Triton cubins
    dit: f64 = 0,
    te: f64 = 0,
    vae: f64 = 0, // and the scratch buffers
    total: f64 = 0,
    gib: f64 = 0, // weights copied to the device
    read: f64 = 0, // in pread
    wait: f64 = 0, // waiting for a free upload chunk (copies behind reads)
    host: f64 = 0, // host-side layout work (FP8 order)
};

pub const max_tokens = 1024; // the template's tokens a prompt may take

/// Milliseconds between `create`'s phases.
const LoadClock = struct {
    io: std.Io,
    t0: std.Io.Timestamp,
    last: std.Io.Timestamp,

    fn lap(c: *LoadClock) f64 {
        const t = std.Io.Clock.awake.now(c.io);
        defer c.last = t;
        return @as(f64, @floatFromInt(c.last.durationTo(t).nanoseconds)) * 1e-6;
    }
    fn total(c: *const LoadClock) f64 {
        return @as(f64, @floatFromInt(c.t0.durationTo(std.Io.Clock.awake.now(c.io)).nanoseconds)) * 1e-6;
    }
};

pub const Pipeline = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    d: cuda.Driver,
    ctx: cuda.Context,
    s: cuda.Stream,
    major: u32,
    minor: u32,
    nv: Nvfp4,
    ops: ops_launch.Ops,
    tk: ops_launch.TeOps,
    vk: VaeOps,
    tri: triton_k.Triton,
    dit_pack: Pack,
    te_pack: Pack,
    vae_pack: Pack,
    w: W.Weights,
    prompt: te_mod.Prompt,
    te: te_mod.TextEncoder,
    dit: Dit,
    vae: Vae,
    sched: sampler.Config,
    max_h: u32,
    max_w: u32,
    bufs: [4]cuda.DeviceBuffer, // latents, the next latents, velocity, pixels
    times: Times = .{},
    load: LoadTimes = .{},
    weights_bytes: u64 = 0,

    /// Loads everything for images up to max_side x max_side pixels (multiples of 16).
    pub fn create(gpa: std.mem.Allocator, io: std.Io, packs: Packs, source: TritonSource, max_side: u32) !*Pipeline {
        const p = try gpa.create(Pipeline);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.io = io;
        p.arena = .init(gpa);
        errdefer p.arena.deinit();
        p.max_h = max_side / 16;
        p.max_w = max_side / 16;
        p.times = .{};
        p.load = .{};
        var clock = LoadClock{ .io = io, .t0 = std.Io.Clock.awake.now(io), .last = std.Io.Clock.awake.now(io) };
        p.d = try cuda.Driver.open();
        errdefer p.d.close();
        p.ctx = try cuda.Context.init(&p.d, 0);
        errdefer p.ctx.deinit();
        var maj: c_int = 0;
        var min: c_int = 0;
        try p.d.check(p.d.api.cuDeviceGetAttribute(&maj, .compute_capability_major, p.ctx.device), "cc");
        try p.d.check(p.d.api.cuDeviceGetAttribute(&min, .compute_capability_minor, p.ctx.device), "cc");
        p.major = @intCast(maj);
        p.minor = @intCast(min);
        p.s = try cuda.Stream.init(&p.d, false); // blocking: ordered with the synchronous uploads
        errdefer p.s.deinit();
        p.load.context = clock.lap();
        p.nv = try Nvfp4.load(gpa, &p.d, p.major, p.minor);
        errdefer p.nv.unload();
        p.ops = try ops_launch.Ops.load(&p.d, kernels.ops, kernels.attention);
        errdefer p.ops.unload();
        p.tk = try ops_launch.TeOps.load(&p.d, kernels.te, kernels.gemm);
        errdefer p.tk.unload();
        p.vk = try VaeOps.load(&p.d, kernels.vae);
        errdefer p.vk.unload();
        const tri = switch (source) {
            .specs => |sp| sp,
            .pack => try triton_k.specsFromPack(p.arena.allocator(), io, packs.dit, p.major, p.minor),
        };
        p.tri = try triton_k.Triton.load(&p.d, p.ctx.device, tri[0] orelse return error.NoAdalnCubin, tri[1] orelse return error.NoRopeCubin, tri[2]);
        errdefer p.tri.unload();
        p.load.kernels = clock.lap();

        p.dit_pack = try Pack.open(gpa, io, packs.dit);
        errdefer p.dit_pack.close(io);
        if (tri[2] == null and !std.mem.eql(u8, p.dit_pack.precision, "nvfp4")) return error.NoSwigluCubin; // FP8's MLP
        p.te_pack = try Pack.open(gpa, io, packs.te);
        errdefer p.te_pack.close(io);
        p.vae_pack = try Pack.open(gpa, io, packs.vae);
        errdefer p.vae_pack.close(io);
        p.sched = try schedulerConfig(p.arena.allocator(), io, packs.te);

        var up = try upload_mod.Uploader.init(&p.d, io, p.s);
        defer up.deinit();
        p.w = try W.Weights.load(gpa, io, &p.d, &p.dit_pack, &p.nv, &up);
        errdefer p.w.deinit();
        try p.s.synchronize();
        p.load.dit = clock.lap();
        p.prompt = try te_mod.Prompt.load(gpa, io, packs.te);
        errdefer p.prompt.deinit();
        const inv = try te_mod.invFreq(p.arena.allocator(), kernels.te_inv_freq);
        p.te = try te_mod.TextEncoder.init(gpa, &p.d, &p.tk, &p.ops, p.s, &p.te_pack, &up, inv, max_tokens);
        errdefer p.te.deinit();
        try p.s.synchronize();
        p.load.te = clock.lap();
        const om = try smath.omegas(kernels.rope_omega);
        const kk: @import("dit.zig").Kernels = .{ .ops = &p.ops, .nv = &p.nv, .tri = &p.tri, .major = p.major, .minor = p.minor };
        p.dit = try Dit.init(gpa, &p.d, kk, &p.w, p.s, om, p.max_h * p.max_w, max_tokens);
        errdefer p.dit.deinit();
        p.vae = try Vae.init(gpa, io, &p.d, &p.vk, &p.tk, &p.ops, p.s, &p.vae_pack, &up, packs.vae, p.max_h, p.max_w);
        errdefer p.vae.deinit();
        const n: usize = 64 * @as(usize, p.max_h) * p.max_w;
        var made: usize = 0;
        errdefer for (p.bufs[0..made]) |*b| b.free();
        for (&p.bufs, [_]usize{ 2 * n, 2 * n, 2 * n, 256 * n / 64 * 4 }) |*b, len| {
            b.* = try cuda.DeviceBuffer.alloc(&p.d, len);
            made += 1;
        }
        try p.s.synchronize();
        p.load.vae = clock.lap();
        p.load.total = clock.total();
        const ns_ms = 1e-6;
        p.load.gib = @as(f64, @floatFromInt(up.stats.bytes)) / (1 << 30);
        p.load.read = @as(f64, @floatFromInt(up.stats.read_ns)) * ns_ms;
        p.load.wait = @as(f64, @floatFromInt(up.stats.wait_ns)) * ns_ms;
        p.load.host = @as(f64, @floatFromInt(up.stats.host_ns)) * ns_ms;
        p.weights_bytes = p.w.store.bytes;
        return p;
    }

    pub fn destroy(p: *Pipeline) void {
        for (&p.bufs) |*b| b.free();
        p.vae.deinit();
        p.dit.deinit();
        p.te.deinit();
        p.prompt.deinit();
        p.w.deinit();
        p.vae_pack.close(p.io);
        p.te_pack.close(p.io);
        p.dit_pack.close(p.io);
        p.tri.unload();
        p.vk.unload();
        p.tk.unload();
        p.ops.unload();
        p.nv.unload();
        p.s.deinit();
        p.ctx.deinit();
        p.d.close();
        p.arena.deinit();
        const gpa = p.gpa;
        gpa.destroy(p);
    }

    /// The sigmas for this size and step count (`steps + 1` values, the last 0); caller frees.
    pub fn sigmas(p: *const Pipeline, a: std.mem.Allocator, height: u32, width: u32, steps: u32) ![]f32 {
        const out = try a.alloc(f32, steps + 1);
        errdefer a.free(out);
        try sampler.sigmas(p.sched, height, width, out);
        return out;
    }

    /// One image: `pixels` gets height * width * out_channels bytes (HWC). Returns the device latents after the last
    /// step and the text context (rows `drop..` of the encoder) for checks; both stay valid until the next call.
    pub fn generate(p: *Pipeline, prompt_text: []const u8, height: u32, width: u32, steps: u32, seed: u64, pixels: []u8, progress: ?Progress) !struct { latents: u64, context: u64, context_rows: u32 } {
        if (height % 16 != 0 or width % 16 != 0) return error.SizeNotMultipleOf16;
        const h = height / 16;
        const w = width / 16;
        if (h > p.max_h or w > p.max_w) return error.ImageTooLarge;
        if (pixels.len != @as(usize, height) * width * p.vae.out_channels) return error.BadPixelBuffer;
        const gpa = p.gpa;
        var clock = Clock{ .io = p.io, .s = p.s };

        try clock.start();
        const ids = try p.prompt.ids(gpa, prompt_text);
        defer gpa.free(ids);
        if (ids.len > max_tokens) return error.PromptTooLong;
        const hidden = try p.te.forward(ids);
        const rows: u32 = @intCast(ids.len - p.prompt.drop);
        const ctx = hidden + @as(u64, p.prompt.drop) * te_mod.dim * 2;
        p.times.encode = try clock.lap();

        const sig = try p.sigmas(gpa, height, width, steps);
        defer gpa.free(sig);
        const n: usize = 64 * @as(usize, h) * w;
        {
            const x0 = try gpa.alloc(f32, n);
            defer gpa.free(x0);
            noise.fill(seed, x0);
            const bf = try gpa.alloc(u16, n);
            defer gpa.free(bf);
            for (x0, bf) |v, *b| b.* = bf16(v);
            try p.dit.upload(try p.bufs[0].at(0), std.mem.sliceAsBytes(bf));
        }
        p.dit.probe = null;
        p.dit.mod_sigma = null;
        try p.dit.buildPrefix(ctx, rows, sig[0]);
        p.times.prefix = try clock.lap();
        var lat = try p.bufs[0].at(0);
        var nxt = try p.bufs[1].at(0);
        const vel = try p.bufs[2].at(0);
        for (0..steps) |i| {
            try p.dit.step(lat, sig[i], h, w, vel);
            const dt: f32 = @floatCast(@as(f64, sig[i + 1]) - @as(f64, sig[i])); // Python floats, then float32
            try p.dit.euler("euler", lat, vel, nxt, n, dt);
            std.mem.swap(u64, &lat, &nxt);
            if (progress) |pr| pr.report(pr.ctx, "denoise", @intCast(i + 1), steps);
        }
        p.times.sample = try clock.lap();
        const px = try p.bufs[3].at(0);
        try p.vae.decode(lat, h, w, px);
        try p.d.check(p.d.api.cuMemcpyDtoH_v2(pixels.ptr, px, pixels.len), "cuMemcpyDtoH");
        p.times.decode = try clock.lap();
        return .{ .latents = lat, .context = ctx, .context_rows = rows };
    }
};

pub const Progress = struct {
    ctx: *anyopaque,
    report: *const fn (ctx: *anyopaque, phase: []const u8, step: u32, of: u32) void,
};

const Clock = struct {
    io: std.Io,
    s: cuda.Stream,
    t: std.Io.Timestamp = undefined,

    fn start(c: *Clock) !void {
        try c.s.synchronize();
        c.t = std.Io.Clock.awake.now(c.io);
    }
    fn lap(c: *Clock) !f64 {
        try c.s.synchronize();
        const now = std.Io.Clock.awake.now(c.io);
        const us = c.t.durationTo(now).toMicroseconds();
        c.t = now;
        return @as(f64, @floatFromInt(us)) / 1000.0;
    }
};

fn schedulerConfig(a: std.mem.Allocator, io: std.Io, dir: []const u8) !sampler.Config {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(16 << 20));
    const M = struct { scheduler: sampler.Config };
    const m = try std.json.parseFromSliceLeaky(M, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    return m.scheduler;
}

/// f32 -> bf16, round to nearest even (the noise is finite).
fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @truncate((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}
