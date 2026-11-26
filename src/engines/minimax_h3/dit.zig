//! MiniMax H3's DiT forward, op for op as the twin (`stk_twin/h3/dit.py`) runs it, under the twin's op names and roles,
//! so `localrouter check h3-replay` matches a capture op by op. `prepare` once a run (condition_proj and the token refiner),
//! `step` once a denoising step (the audio carry, the fp32 patch projections, 50 blocks, the fp32 heads, the un-carry).
//! Batch 1; NVFP4 linears from the pack (static input scales), comfy-kitchen's INT8 attention and fused RMSNorm + RoPE,
//! LocalRouter's fp32 / bf16 GEMMs and elementwise kernels.
//!
//! Two schedules, one set of bits. The reference schedule is the twin's op sequence (it runs whenever a probe is set, and
//! with STK_H3_REF=1). The fast one (default) fuses each gate_add with the norm_mod after it into one kernel
//! (`h3_gate_add_norm_mod`: per element the same arithmetic, ops.cu says why), has the INT8 attention store its output as
//! rows (no [H, S, D] -> rows pass), and replays the step as a CUDA graph per shape. Neither changes any rounding; the
//! tests (`test_fuse.py`, `h3-replay`'s fast passes, `h3-step-bench`'s equality) compare the two byte for byte.

const std = @import("std");
const gpu = @import("gpu");
const qi = @import("qwen_image");
const launch = @import("launch.zig");
const lay = @import("layout.zig");
const pdd = @import("pdd.zig");
const launch_image = @import("launch_image.zig");

const Pack = qi.pack.Pack;
const Store = qi.weights.Store;
const Uploader = qi.upload.Uploader;
const Linear = qi.weights.Linear;
const plan = qi.nvfp4_plan;
const profile_mod = @import("profile.zig");
pub const Io = qi.dit.Io;
pub const Probe = qi.dit.Probe;
pub const Profile = profile_mod.Profile;
const Class = profile_mod.Class;

pub const layers = 50;
pub const dim = 5376;
pub const heads = 56;
pub const head_dim = 128;
pub const inner = heads * head_dim; // 7168
pub const ffn = 14336;
pub const text_dim = 5120;
pub const t_dim = 8; // the curve basis
pub const grid = 1025; // the curve table's rows
pub const ada_n = 3 * 6 * dim; // a block's adaln output (3 modalities x 6 chunks)
const eps: f32 = 1e-5;
const tile: u32 = 12;
const linear_names = [4][]const u8{ "qkv", "out", "fc1", "fc2" };

/// A step's shape and buffers: what its launches' arguments depend on (the step graph's key, less the fuse flag).
const Body = struct { video: u64, vel_v: u64, vel_a: u64, t: u32, h: u32, w: u32, a: u32, ly: lay.Layout, m: u32, row_v: u32, row_a: u32 };
const StepGraph = struct { key: [14]u64, exec: ?gpu.graph.Exec }; // exec null: seen once (run eagerly), captured on the next use
const max_graphs = 4;

pub const Kernels = struct {
    h3: *const launch.Ops,
    kitchen: *const launch.Kitchen,
    te: *const qi.ops_launch.TeOps, // the bf16 GEMM (token refiner)
    ops: *const qi.ops_launch.Ops, // bf16 attention and row copies (token refiner)
    nv: ?*qi.nvfp4_exec.Nvfp4, // null: no NVFP4 or FP8 linears (a bf16 pack: H3-Image)
    major: u32,
    minor: u32,
    image: ?*const launch_image.Ops = null, // H3-Image's PDD head mix (`stepImage`)
};

const Block = struct {
    lin: [4]Linear,
    norm1: u64, // f32 [D]
    norm2: u64,
    q_norm: u64, // bf16 [128]
    k_norm: u64,
    ada_w: u64, // f32 [ada_n, 8]
    ada_b: u64, // f32 [ada_n]
};

const Refiner = struct { norm1: u64, qkv: u64, q_norm: u64, k_norm: u64, out: u64, norm2: u64, fc1: u64, fc2: u64 };

pub const Weights = struct {
    store: Store,
    blocks: [layers]Block = undefined,
    refiner: [2]Refiner = undefined,
    cond_w: u64 = 0,
    cond_b: u64 = 0,
    final_norm: u64 = 0, // the refiner's
    vp_w: u64 = 0,
    vp_b: u64 = 0,
    ap_w: u64 = 0,
    ap_b: u64 = 0,
    fl_ada_w: u64 = 0,
    fl_ada_b: u64 = 0,
    fl_norm: u64 = 0,
    vo_w: u64 = 0,
    vo_b: u64 = 0,
    heads: u32 = 1, // `final_layer.video_out`'s blocks of 96 rows: 1, or H3-Image's PDD bank of 6
    ao_w: u64 = 0,
    ao_b: u64 = 0,
    table: []f32 = &.{}, // host: the curve [grid, 8]
    inv_freq: [16]f32 = undefined, // host: RoPE frequencies

    pub fn deinit(w: *Weights) void {
        w.store.gpa.free(w.table);
        w.store.deinit();
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const gpu.Driver, p: *const Pack, nv: *qi.nvfp4_exec.Nvfp4, up: *Uploader) !Weights {
        var w: Weights = .{ .store = .init(d, gpa) };
        errdefer w.deinit();
        const st = &w.store;
        inline for (.{
            .{ "cond_w", "condition_proj.weight" },                .{ "cond_b", "condition_proj.bias" },
            .{ "final_norm", "token_refiner.final_norm.weight" },  .{ "vp_w", "video_patch_proj.weight" },
            .{ "vp_b", "video_patch_proj.bias" },                  .{ "ap_w", "audio_patch_proj.weight" },
            .{ "ap_b", "audio_patch_proj.bias" },                  .{ "fl_ada_w", "final_layer.adaln_proj.linear.weight" },
            .{ "fl_ada_b", "final_layer.adaln_proj.linear.bias" }, .{ "fl_norm", "final_layer.norm.weight" },
            .{ "vo_w", "final_layer.video_out.weight" },           .{ "vo_b", "final_layer.video_out.bias" },
            .{ "ao_w", "final_layer.audio_out.weight" },           .{ "ao_b", "final_layer.audio_out.bias" },
        }) |e| @field(w, e[0]) = try st.tensor(up, p, e[1]);
        try w.loadRefiner(up, p);
        var nb: [96]u8 = undefined;
        for (&w.blocks, 0..) |*b, i| {
            try w.loadBlockTensors(up, p, b, i);
            for (linear_names, 0..) |nm, j| b.lin[j] = try st.linear(io, p, nv, up, try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ i, nm }));
        }
        try w.loadHost(gpa, io, p);
        try st.done(up); // every copy and pack4 has landed
        return w;
    }

    /// H3-Image's DiT from its bf16 pack (`stk_twin.h3i.build`): the video pack's tensors under the video pack's names but
    /// no audio ones, block linears of kind `bf16` (`{key}.weight` [n, k]), and `final_layer.video_out` the PDD bank
    /// [6 x 96, 5376] / [576] fp32 (`heads` = 6). No NVFP4 kernels: every linear is the bf16 GEMM.
    pub fn loadImage(gpa: std.mem.Allocator, io: std.Io, d: *const gpu.Driver, p: *const Pack, up: *Uploader) !Weights {
        var w: Weights = .{ .store = .init(d, gpa) };
        errdefer w.deinit();
        const st = &w.store;
        {
            const vw = try p.get("final_layer.video_out.weight");
            const vb = try p.get("final_layer.video_out.bias");
            if (vw.dtype != .f32 or vw.rank != 2 or vw.dim(0) != 96 * pdd.max_heads or vw.dim(1) != dim) return error.BadPddHeads;
            if (vb.dtype != .f32 or vb.rank != 1 or vb.dim(0) != 96 * pdd.max_heads) return error.BadPddHeads;
        }
        inline for (.{
            .{ "cond_w", "condition_proj.weight" },                .{ "cond_b", "condition_proj.bias" },
            .{ "final_norm", "token_refiner.final_norm.weight" },  .{ "vp_w", "video_patch_proj.weight" },
            .{ "vp_b", "video_patch_proj.bias" },                  .{ "fl_ada_w", "final_layer.adaln_proj.linear.weight" },
            .{ "fl_ada_b", "final_layer.adaln_proj.linear.bias" }, .{ "fl_norm", "final_layer.norm.weight" },
            .{ "vo_w", "final_layer.video_out.weight" },           .{ "vo_b", "final_layer.video_out.bias" },
        }) |e| @field(w, e[0]) = try st.tensor(up, p, e[1]);
        w.heads = pdd.max_heads;
        try w.loadRefiner(up, p);
        var nb: [96]u8 = undefined;
        for (&w.blocks, 0..) |*b, i| {
            try w.loadBlockTensors(up, p, b, i);
            for (linear_names, 0..) |nm, j| b.lin[j] = try bf16Linear(st, up, p, try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ i, nm }));
        }
        try w.loadHost(gpa, io, p);
        try st.done(up);
        return w;
    }

    /// A block linear of a bf16 pack: the manifest says `bf16`, `{key}.weight` is BF16 [n, k] with K % 8 == 0 (the bf16
    /// GEMM's rows); uploaded as stored.
    fn bf16Linear(st: *Store, up: *Uploader, p: *const Pack, key: []const u8) !Linear {
        if ((try p.kind(key)) != .bf16) return error.NotBf16Linear;
        var nb: [80]u8 = undefined;
        const t = try p.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{key}));
        if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) % 8 != 0) return error.BadBf16Linear;
        const n: u32 = @intCast(t.dim(0));
        return .{ .kind = .bf16, .n = n, .k = @intCast(t.dim(1)), .npad = n, .w = try st.tensorOf(up, p, t), .ws = 0, .scale = 1, .act = 1 };
    }

    fn loadRefiner(w: *Weights, up: *Uploader, p: *const Pack) !void {
        var nb: [96]u8 = undefined;
        for (&w.refiner, 0..) |*r, i| {
            inline for (.{
                .{ "norm1", "norm1.weight" },        .{ "qkv", "attn.qkv_proj.weight" }, .{ "q_norm", "attn.q_norm.weight" },
                .{ "k_norm", "attn.k_norm.weight" }, .{ "out", "attn.out_proj.weight" }, .{ "norm2", "norm2.weight" },
                .{ "fc1", "mlp.fc1.weight" },        .{ "fc2", "mlp.fc2.weight" },
            }) |e| @field(r, e[0]) = try w.store.tensor(up, p, try std.fmt.bufPrint(&nb, "token_refiner.blocks.{d}.{s}", .{ i, e[1] }));
        }
    }

    fn loadBlockTensors(w: *Weights, up: *Uploader, p: *const Pack, b: *Block, i: usize) !void {
        var nb: [96]u8 = undefined;
        inline for (.{ "norm1", "norm2", "q_norm", "k_norm", "ada_w", "ada_b" }) |nm|
            @field(b, nm) = try w.store.tensor(up, p, try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ i, nm }));
    }

    /// Host copies: the curve table and the RoPE frequencies.
    fn loadHost(w: *Weights, gpa: std.mem.Allocator, io: std.Io, p: *const Pack) !void {
        const t = try p.get("adaln_t_table");
        if (t.dim(0) != grid or t.dim(1) != t_dim) return error.BadCurveTable;
        w.table = try gpa.alloc(f32, grid * t_dim);
        try p.read(io, t, std.mem.sliceAsBytes(w.table));
        try p.read(io, try p.get("rope.inv_freq"), std.mem.sliceAsBytes(&w.inv_freq));
    }
};

/// Device scratch for sequences up to `max_s` tokens (text + audio + video) and text up to `max_l` tokens.
/// The bytes of a table of buffer sizes (`Dit.Scratch` and the like) as allocated: each at least 256.
pub fn sumBuffers(sizes: anytype) u64 {
    var total: u64 = 0;
    inline for (@typeInfo(@TypeOf(sizes)).@"struct".field_names) |name| total += @max(@field(sizes, name), 256);
    return total;
}

pub const Dit = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    k: Kernels,
    w: *const Weights,
    s: gpu.Stream,
    probe: ?Probe = null,
    bufs: std.ArrayList(gpu.DeviceBuffer) = .empty,
    max_s: u32,
    max_a: u32, // audio latent frames
    max_l: u32,
    // the run's refined text [L, D] bf16
    context: u64 = 0,
    ctx_rows: u32 = 0,
    cond_rows: u64 = 0,
    cond_emb: u64 = 0,
    cond_bufs: [2]?gpu.DeviceBuffer = .{ null, null },
    cond_hw: ?[2]u32 = null,
    text_tags: []u8 = &.{},
    // H3-Image's reference pictures (`imageRefs`): their embeddings bf16 [rows, D], held for the run (`releaseRefs`)
    ref_buf: ?gpu.DeviceBuffer = null,
    ref_shapes: [lay.max_refs][2]u32 = undefined,
    n_refs: u32 = 0,
    image_key: ?[3 + 2 * lay.max_refs]u32 = null,
    image_idx_key: ?[4 + 2 * lay.max_refs]u32 = null,
    // per step
    h: u64 = 0,
    hn: u64 = 0,
    qkv: u64 = 0,
    heads_out: u64 = 0,
    att: u64 = 0,
    o: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    q: u64 = 0,
    kk: u64 = 0,
    v: u64 = 0,
    qc: u64 = 0,
    qs: u64 = 0,
    ws: u64 = 0,
    mod: u64 = 0,
    t_emb: u64 = 0,
    idx: u64 = 0,
    rope: u64 = 0,
    a_in: u64 = 0,
    rows: u64 = 0,
    arows: u64 = 0,
    emb: u64 = 0,
    aemb: u64 = 0,
    fada: u64 = 0,
    vout: u64 = 0,
    aout: u64 = 0,
    layout_key: ?[6]u32 = null,
    /// The fast schedule (`fuse`: fused gate_add + norm_mod, attention stored as rows; `graphs`: the step's launches replayed
    /// as a CUDA graph per shape and buffer set). Both default on, both off with STK_H3_REF=1 (the reference path), and a
    /// probe or a profile always runs the eager schedule. `graph_error` is why graphs turned themselves off, if they did.
    fuse: bool = true,
    graphs: bool = true,
    graph_error: ?anyerror = null,
    prof: ?*Profile = null,
    step_graphs: std.ArrayList(StepGraph) = .empty,
    mod2: u64 = 0, // the second modulation buffer: block i + 1's, while block i's is still read by the fused kernel
    idx_host: []i32 = &.{}, // the modulation row of every token, kept per layout (uploaded when it changes)
    idx_key: ?[9]u32 = null,
    extra_launches: u64 = 0, // the NVFP4 and row-copy launches (the H3 kernels count in `launch.count`)
    /// The blocks' attention: comfy-kitchen's INT8 (false, the video model on CUDA) or the bf16 attention of the token
    /// refiner on q, k, v copied out of the rotated qkv rows (true: H3-Image on both vendors, ROCM.md section 3).
    attn_bf16: bool = false,
    head_w: u64 = 0, // H3-Image: the step's mixed PDD head, fp32 [96, D] and [96]
    head_b: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const gpu.Driver, k: Kernels, w: *const Weights, s: gpu.Stream, max_s: u32, max_l: u32, max_a: u32) !Dit {
        return initWith(gpa, d, k, w, s, max_s, max_l, max_a, scratch(max_s, max_l, max_a));
    }

    fn initWith(gpa: std.mem.Allocator, d: *const gpu.Driver, k: Kernels, w: *const Weights, s: gpu.Stream, max_s: u32, max_l: u32, max_a: u32, sizes: Scratch) !Dit {
        var x: Dit = .{ .gpa = gpa, .d = d, .k = k, .w = w, .s = s, .max_s = max_s, .max_l = max_l, .max_a = max_a };
        errdefer x.deinit();
        if (std.c.getenv("STK_H3_REF")) |v| {
            if (v[0] == '1') {
                x.fuse = false;
                x.graphs = false;
            }
        }
        x.idx_host = try gpa.alloc(i32, max_s);
        inline for (@typeInfo(Scratch).@"struct".field_names) |name| @field(x, name) = try x.alloc(@field(sizes, name));
        return x;
    }

    /// H3-Image's DiT (`Weights.loadImage`, `stepImage`): `max_s` rows of [text | target] and `max_l` text tokens, the
    /// bf16 attention, no audio, the PDD head buffers. `k.image` must be set.
    pub fn initImage(gpa: std.mem.Allocator, d: *const gpu.Driver, k: Kernels, w: *const Weights, s: gpu.Stream, max_s: u32, max_l: u32) !Dit {
        if (k.image == null or w.heads != pdd.max_heads) return error.NotAnImageDit;
        var x = try initWith(gpa, d, k, w, s, max_s, max_l, 0, imageScratch(max_s, max_l));
        errdefer x.deinit();
        x.attn_bf16 = true;
        x.graphs = false; // `stepImage` runs eagerly
        x.head_w = try x.alloc(96 * dim * 4);
        x.head_b = try x.alloc(96 * 4);
        return x;
    }

    /// `scratch` for H3-Image: no audio, no NVFP4 quantizer rows, no INT8 workspace; q, k and v (the bf16 attention's
    /// operands) for the whole sequence.
    pub fn imageScratch(max_s: u32, max_l: u32) Scratch {
        var sc = scratch(max_s, max_l, 0);
        const qkv_rows: u64 = @max(max_s, max_l);
        inline for (.{ "q", "kk", "v" }) |f| @field(sc, f) = qkv_rows * inner * 2;
        inline for (.{ "heads_out", "qc", "qs", "ws", "a_in", "arows", "aemb", "aout" }) |f| @field(sc, f) = 0;
        return sc;
    }

    /// The device bytes `initImage` allocates.
    pub fn imageScratchBytes(max_s: u32, max_l: u32) u64 {
        return sumBuffers(imageScratch(max_s, max_l)) + 96 * dim * 4 + @max(96 * 4, 256);
    }

    pub fn deinit(x: *Dit) void {
        for (x.step_graphs.items) |*g| if (g.exec) |*e| e.deinit();
        x.step_graphs.deinit(x.gpa);
        x.gpa.free(x.idx_host);
        x.gpa.free(x.text_tags);
        if (x.ref_buf) |*b| b.free();
        for (&x.cond_bufs) |*slot| if (slot.*) |*b| b.free();
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(x.gpa);
    }

    /// Kernel launches so far (the H3 and comfy-kitchen kernels, the NVFP4 quantizers and GEMMs, the row copies): a
    /// step's difference is the eager schedule's launch count (a graph replay is one launch).
    pub fn launches(x: *const Dit) u64 {
        return @atomicLoad(u64, &launch.count, .monotonic) + x.extra_launches;
    }

    /// The carried audio latents in bf16: 32 channels x 2 rows a latent frame, which a short sequence's `max_s` can be below.
    pub fn audioInBytes(max_a: u32) u64 {
        return 32 * 2 * @as(u64, max_a) * 2;
    }

    /// The bytes of each buffer `init` allocates for `max_s` rows, `max_l` text tokens and `max_a` audio latent frames.
    pub const Scratch = struct { context: u64, h: u64, hn: u64, qkv: u64, heads_out: u64, att: u64, o: u64, gu: u64, act: u64, q: u64, kk: u64, v: u64, qc: u64, qs: u64, ws: u64, mod: u64, mod2: u64, t_emb: u64, idx: u64, rope: u64, a_in: u64, rows: u64, arows: u64, emb: u64, aemb: u64, fada: u64, vout: u64, aout: u64 };

    pub fn scratch(max_s: u32, max_l: u32, max_a: u32) Scratch {
        const r: u64 = max_s;
        const bf = 2;
        const qr = plan.rows(max_s, ffn, plan.TB); // the widest quantized rows (K up to 14336)
        return .{
            .context = @as(u64, max_l) * dim * bf,
            .h = r * dim * bf,
            .hn = r * dim * bf,
            .qkv = r * 3 * inner * bf,
            .heads_out = r * inner * bf,
            .att = r * inner * bf,
            .o = r * dim * bf,
            .gu = r * 2 * ffn * bf,
            .act = r * ffn * bf,
            .q = @as(u64, max_l) * inner * bf,
            .kk = @as(u64, max_l) * inner * bf,
            .v = @as(u64, max_l) * inner * bf,
            .qc = qr.fp8_bytes,
            .qs = qr.scales_bytes,
            .ws = launch.Kitchen.Plan.of(heads, max_s).total,
            .mod = 3 * ada_n * 4,
            .mod2 = 3 * ada_n * 4,
            .t_emb = 3 * t_dim * 4,
            .idx = r * 4,
            .rope = r * 48 * 4 * bf,
            .a_in = audioInBytes(max_a),
            .rows = r * 96 * 4,
            .arows = r * 32 * 4,
            .emb = r * dim * 4,
            .aemb = r * dim * 4,
            .fada = 3 * 2 * dim * 4,
            .vout = r * 96 * 4,
            .aout = r * 32 * 4,
        };
    }

    /// The device bytes `init` allocates (each buffer at least 256).
    pub fn scratchBytes(max_s: u32, max_l: u32, max_a: u32) u64 {
        return sumBuffers(scratch(max_s, max_l, max_a));
    }

    /// Frees a first frame's condition (`firstFrame`) once the steps that read it are done, so no request's
    /// condition outlives it. The stream is synchronized first.
    pub fn releaseCondition(x: *Dit) void {
        if (x.cond_bufs[0] == null and x.cond_bufs[1] == null) return;
        x.s.synchronize() catch {};
        for (&x.cond_bufs) |*slot| if (slot.*) |*b| {
            b.free();
            slot.* = null;
        };
        x.cond_rows = 0;
        x.cond_emb = 0;
        x.cond_hw = null;
        x.layout_key = null;
    }

    /// The device bytes a first frame's condition holds now (0 between requests).
    pub fn conditionResident(x: *const Dit) u64 {
        var total: u64 = 0;
        for (x.cond_bufs) |slot| if (slot) |b| {
            total += b.len;
        };
        return total;
    }

    /// A first frame's condition rows and their embedding (`firstFrame`), for an `h` x `w` latent, with the noise drawn
    /// for them: held while the steps run (`releaseCondition`).
    pub fn conditionBytes(h: u32, w: u32) u64 {
        const rows: u64 = @as(u64, h / 2) * (w / 2);
        return 2 * rows * 96 * 4 + rows * dim * 4;
    }

    fn alloc(x: *Dit, n: u64) !u64 {
        var b = try gpu.DeviceBuffer.alloc(x.d, @intCast(@max(n, 256)));
        errdefer b.free();
        try x.bufs.append(x.gpa, b);
        return b.at(0);
    }

    pub fn upload(x: *Dit, ptr: u64, bytes: []const u8) !void {
        try x.s.synchronize();
        try x.d.check(x.d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    /// Host bytes to the device on the stream, no host sync: the driver stages pageable memory before it returns.
    fn uploadAsync(x: *Dit, ptr: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(ptr, bytes.ptr, bytes.len, x.s.handle), "cuMemcpyHtoDAsync");
    }

    fn markBegin(x: *Dit) !void {
        if (x.prof) |p| try p.begin(x.s);
    }
    fn markEnd(x: *Dit, c: Class) !void {
        if (x.prof) |p| try p.end(x.s, c);
    }

    fn pre(x: *Dit, name: []const u8, ins: []const Io) !void {
        if (x.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(x: *Dit, name: []const u8, outs: []const Io) !void {
        if (x.probe) |p| try p.after(p.ctx, name, outs);
    }

    // ------------------------------------------------------------------ recorded ops
    fn gemmBf16(x: *Dit, name: []const u8, in: u64, wt: u64, bias: u64, out: u64, m: u64, n: u64, kdim: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = m * kdim * 2 }});
        try x.k.te.gemm(x.s, in, wt, bias, out, m, n, kdim);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * n * 2 }});
    }

    fn gemmF32(x: *Dit, name: []const u8, in: u64, wt: u64, bias: u64, out: u64, m: u64, n: u64, kdim: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = m * kdim * 4 }});
        try x.k.h3.gemmF32(x.s, in, wt, bias, out, m, n, kdim);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = m * n * 4 }});
    }

    fn rms(x: *Dit, name: []const u8, in: u64, w: u64, out: u64, rows: u64, d: u64) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = rows * d * 2 }});
        try x.k.h3.rmsNorm(x.s, in, w, out, rows, d, eps);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = rows * d * 2 }});
    }

    fn add(x: *Dit, name: []const u8, a: u64, b: u64, out: u64, n: u64) !void {
        try x.pre(name, &.{ .{ .role = "x", .ptr = a, .bytes = n * 2 }, .{ .role = "z", .ptr = b, .bytes = n * 2 } });
        try x.k.h3.add(x.s, a, b, out, n);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = n * 2 }});
    }

    fn linear(x: *Dit, name: []const u8, l: Linear, in: u64, out: u64, rows: u32, gc: Class) !void {
        try x.pre(name, &.{.{ .role = "x", .ptr = in, .bytes = @as(u64, rows) * l.k * 2 }});
        try x.linearRun(l, in, out, rows, gc);
        try x.post(name, &.{.{ .role = "y", .ptr = out, .bytes = @as(u64, rows) * l.n * 2 }});
    }

    /// One block linear: the quantizer and the prompt GEMM (`Linear.forward`), or a bf16 pack's one bf16 GEMM
    /// (`Linear.forwardBf16`). Profiling runs forward's two launches itself (the same plan calls and arguments) so each
    /// gets its own span: `quant`, then `gc`.
    fn linearRun(x: *Dit, l: Linear, in: u64, out: u64, rows: u32, gc: Class) !void {
        if (l.kind == .bf16) {
            x.extra_launches += 1;
            if (x.prof != null) try x.markBegin();
            try l.forwardBf16(x.k.te, in, out, rows, x.s);
            if (x.prof != null) try x.markEnd(gc);
            return;
        }
        x.extra_launches += 2;
        const nv = x.k.nv orelse return error.NoNvfp4Kernels;
        if (x.prof == null or l.kind != .nvfp4) return l.forward(nv, x.k.major, x.k.minor, tile, in, out, rows, x.qc, x.qs, x.s);
        const tbk = plan.tb(tile, x.k.major, x.k.minor);
        const rr = plan.rows(rows, l.k, tbk);
        try x.markBegin();
        const qp = try plan.quant4(rows, l.k, l.k, tbk, l.act);
        try nv.run(&qp, .{ .x = in, .codes = x.qc, .scales = x.qs }, x.s);
        try x.markEnd(.quant);
        try x.markBegin();
        const gp = try plan.gemmPrompt(.a4, tile, x.k.major, x.k.minor, rows, l.n, l.k, l.npad, rr.mpad, false, l.alpha());
        try nv.run(&gp, .{ .x = x.qc, .xs = x.qs, .w = l.w, .ws = l.ws, .out = out }, x.s);
        try x.markEnd(gc);
    }

    // ------------------------------------------------------------------ the run's text
    /// Qwen3-VL layer-50 states `text` [L, 5120] bf16 -> the refined context [L, 5376] (condition_proj, 2 refiner blocks,
    /// final norm), kept for every step of the run.
    pub fn prepare(x: *Dit, text: u64, l: u32) !void {
        if (l > x.max_l) return error.PromptTooLong;
        x.cond_hw = null;
        x.layout_key = null;
        x.releaseRefs();
        x.gpa.free(x.text_tags);
        x.text_tags = &.{};
        const L: u64 = l;
        const w = x.w;
        try x.gemmBf16("refiner.condition_proj", text, w.cond_w, w.cond_b, x.h, L, dim, text_dim);
        var nb: [48]u8 = undefined;
        for (w.refiner, 0..) |r, i| {
            const nm = struct {
                fn f(buf: []u8, j: usize, op: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "refiner.{d}.{s}", .{ j, op }) catch unreachable;
                }
            }.f;
            try x.rms(nm(&nb, i, "norm1"), x.h, r.norm1, x.hn, L, dim);
            try x.gemmBf16(nm(&nb, i, "qkv"), x.hn, r.qkv, 0, x.qkv, L, 3 * inner, dim);
            inline for (.{ "q", "kk", "v" }, 0..) |dst, part| // q, k, v as contiguous [L, H, 128] (the twin's .contiguous())
                try x.k.ops.copyRows(x.s, @field(x, dst), x.qkv + part * inner * 2, L, inner * 2, 3 * inner * 2, inner * 2);
            try x.rms(nm(&nb, i, "q_norm"), x.q, r.q_norm, x.q, L * heads, head_dim);
            try x.rms(nm(&nb, i, "k_norm"), x.kk, r.k_norm, x.kk, L * heads, head_dim);
            const an = nm(&nb, i, "attention");
            const sz = L * inner * 2;
            try x.pre(an, &.{ .{ .role = "q", .ptr = x.q, .bytes = sz }, .{ .role = "k", .ptr = x.kk, .bytes = sz }, .{ .role = "v", .ptr = x.v, .bytes = sz } });
            try x.k.ops.attention(x.s, x.q, x.kk, x.v, x.att, l, heads, heads, l, false, 0);
            try x.post(an, &.{.{ .role = "y", .ptr = x.att, .bytes = sz }});
            try x.gemmBf16(nm(&nb, i, "out"), x.att, r.out, 0, x.o, L, dim, inner);
            try x.add(nm(&nb, i, "attn_residual"), x.o, x.h, x.h, L * dim);
            try x.rms(nm(&nb, i, "norm2"), x.h, r.norm2, x.hn, L, dim);
            try x.gemmBf16(nm(&nb, i, "fc1"), x.hn, r.fc1, 0, x.gu, L, 2 * ffn, dim);
            const sn = nm(&nb, i, "swiglu");
            try x.pre(sn, &.{.{ .role = "x", .ptr = x.gu, .bytes = L * 2 * ffn * 2 }});
            try x.k.h3.swiglu(x.s, x.gu, x.act, L, ffn, true);
            try x.post(sn, &.{.{ .role = "y", .ptr = x.act, .bytes = L * ffn * 2 }});
            try x.gemmBf16(nm(&nb, i, "fc2"), x.act, r.fc2, 0, x.o, L, dim, ffn);
            try x.add(nm(&nb, i, "mlp_residual"), x.o, x.h, x.h, L * dim);
        }
        try x.rms("refiner.final_norm", x.h, w.final_norm, x.context, L, dim);
        x.ctx_rows = l;
    }

    /// After prepare: normalized first-frame fp32 latent [24,1,h,w], modality tags of the expanded text context.
    /// Noise restarts the portable stream at the request seed. The augmented rows remain constant across steps.
    pub fn firstFrame(x: *Dit, z: u64, h: u32, w: u32, seed: u64, tags: []const u8) !void {
        if (h == 0 or w == 0 or h % 2 != 0 or w % 2 != 0 or tags.len != x.ctx_rows) return error.BadCondition;
        for (tags) |tag| if (tag > 1) return error.BadCondition;
        const rows: u64 = @as(u64, h / 2) * (w / 2);
        if (rows + x.ctx_rows > x.max_s) return error.VideoTooLong;
        const n = rows * 96;
        x.releaseCondition(); // the previous request's, before this one's is allocated
        const owned = try x.gpa.dupe(u8, tags);
        errdefer x.gpa.free(owned);
        var cb = try gpu.DeviceBuffer.alloc(x.d, @intCast(n * 4));
        errdefer {
            x.s.synchronize() catch {};
            cb.free();
        }
        var eb = try gpu.DeviceBuffer.alloc(x.d, @intCast(rows * dim * 4));
        errdefer eb.free();
        const cr = try cb.at(0);
        const ce = try eb.at(0);
        var noise = try gpu.DeviceBuffer.alloc(x.d, @intCast(n * 4));
        defer {
            x.s.synchronize() catch {};
            noise.free();
        }
        const host = try x.gpa.alloc(f32, @intCast(n));
        defer x.gpa.free(host);
        qi.noise.fill(seed, host);
        const np = try noise.at(0);
        try x.upload(np, std.mem.sliceAsBytes(host));
        try x.pre("cond_rows", &.{ .{ .role = "z", .ptr = z, .bytes = n * 4 }, .{ .role = "noise", .ptr = np, .bytes = n * 4 } });
        try x.k.h3.conditionRows(x.s, z, np, cr, h, w);
        try x.post("cond_rows", &.{.{ .role = "y", .ptr = cr, .bytes = n * 4 }});
        try x.s.synchronize();
        x.gpa.free(x.text_tags);
        x.text_tags = owned;
        x.cond_rows = cr;
        x.cond_emb = ce;
        x.cond_bufs = .{ cb, eb };
        x.cond_hw = .{ h, w };
        x.layout_key = null;
    }

    /// An H3-Image reference: its normalized fp32 latent [24, 1, h, w] on the device (even h, w).
    pub const Ref = struct { z: u64, h: u32, w: u32 };

    /// After prepare (H3-Image edits, `initImage`): each reference's rows patchified in fp32 with the condition noise
    /// 0.999 r + 0.001 noise (the portable stream restarted at `seed` for each, in place of ComfyUI's CPU torch.randn
    /// at the sampler's seed), through video_patch_proj (fp32) and rounded to bf16, once; `tags` are the expanded text's
    /// modality tags. Held for the run's steps (`releaseRefs`; `prepare` releases them). The twin's `prepare(refs=...)`.
    pub fn imageRefs(x: *Dit, refs: []const Ref, seed: u64, tags: []const u8) !void {
        if (!x.attn_bf16 or x.k.image == null) return error.NotAnImageDit;
        if (refs.len > lay.max_refs) return error.TooManyReferences;
        if (tags.len != x.ctx_rows) return error.BadCondition;
        for (tags) |tag| if (tag > 1) return error.BadCondition;
        x.releaseRefs();
        var total: u64 = 0;
        var most: u64 = 0;
        for (refs) |r| {
            if (r.h == 0 or r.w == 0 or r.h % 2 != 0 or r.w % 2 != 0) return error.BadReferenceShape;
            const rows = @as(u64, r.h / 2) * (r.w / 2);
            total += rows;
            most = @max(most, rows);
        }
        if (total + x.ctx_rows > x.max_s) return error.ImageTooLarge;
        const owned = try x.gpa.dupe(u8, tags);
        errdefer x.gpa.free(owned);
        var eb = try gpu.DeviceBuffer.alloc(x.d, @intCast(@max(total * dim * 2, 256)));
        errdefer {
            x.s.synchronize() catch {};
            eb.free();
        }
        var rb = try gpu.DeviceBuffer.alloc(x.d, @intCast(@max(most * 96 * 4, 256)));
        defer {
            x.s.synchronize() catch {};
            rb.free();
        }
        var nbuf = try gpu.DeviceBuffer.alloc(x.d, @intCast(@max(most * 96 * 4, 256)));
        defer {
            x.s.synchronize() catch {};
            nbuf.free();
        }
        const host = try x.gpa.alloc(f32, @intCast(most * 96));
        defer x.gpa.free(host);
        const e = try eb.at(0);
        const cr = try rb.at(0);
        const np = try nbuf.at(0);
        var off: u64 = 0;
        var nb: [32]u8 = undefined;
        for (refs, 0..) |r, j| {
            const rows = @as(u64, r.h / 2) * (r.w / 2);
            const n = rows * 96;
            qi.noise.fill(seed, host[0..@intCast(n)]);
            try x.upload(np, std.mem.sliceAsBytes(host[0..@intCast(n)]));
            const cn = try std.fmt.bufPrint(&nb, "ref{d}.cond_rows", .{j});
            try x.pre(cn, &.{ .{ .role = "z", .ptr = r.z, .bytes = n * 4 }, .{ .role = "noise", .ptr = np, .bytes = n * 4 } });
            try x.k.h3.conditionRows(x.s, r.z, np, cr, r.h, r.w);
            try x.post(cn, &.{.{ .role = "y", .ptr = cr, .bytes = n * 4 }});
            try x.gemmF32(try std.fmt.bufPrint(&nb, "ref{d}.patch_proj", .{j}), cr, x.w.vp_w, x.w.vp_b, x.emb, rows, dim, 96);
            try x.k.h3.toBf16(x.s, x.emb, e + off * dim * 2, rows * dim);
            x.ref_shapes[j] = .{ r.h, r.w };
            off += rows;
        }
        try x.s.synchronize();
        x.gpa.free(x.text_tags);
        x.text_tags = owned;
        x.ref_buf = eb;
        x.n_refs = @intCast(refs.len);
        x.image_key = null;
        x.image_idx_key = null;
    }

    /// Frees the references (`imageRefs`) once the steps that read them are done; the stream is synchronized first.
    pub fn releaseRefs(x: *Dit) void {
        x.n_refs = 0;
        x.image_key = null;
        x.image_idx_key = null;
        if (x.ref_buf) |*b| {
            x.s.synchronize() catch {};
            b.free();
            x.ref_buf = null;
        }
    }

    /// The device bytes `imageRefs` holds for references of `rows` DiT rows in all (and its temporaries, at most
    /// `most_rows` for one).
    pub fn refBytes(rows: u64, most_rows: u64) u64 {
        return rows * dim * 2 + 2 * most_rows * 96 * 4;
    }

    // ------------------------------------------------------------------ one step
    /// video bf16 [24, T, H, W] and the sampler's carried audio bf16 [32, 2, A] at `sigma` -> the velocities ComfyUI
    /// returns (negated; the audio back on the carry), into `vel_v` / `vel_a`. Host work a step repeats is cached per
    /// layout (the RoPE table, the token -> modulation-row map); what changes with sigma (the curve embedding) goes up
    /// on the stream, so a step needs no host sync. The carry and un-carry take sigma-derived scalars as kernel arguments
    /// and stay outside the step graph; everything between them is the graph (one per shape and buffer set: its first
    /// use runs eagerly, which loads whatever a kernel loads lazily, the second captures, later ones replay).
    pub fn step(x: *Dit, video: u64, audio: u64, sigma: f32, t: u32, h: u32, w: u32, a: u32, vel_v: u64, vel_a: u64) !void {
        const L = x.ctx_rows;
        var ly = lay.Layout.init(L, t, h, w, a);
        if (x.cond_hw) |hw| {
            if (hw[0] != h or hw[1] != w) return error.BadCondition;
            ly = ly.withFirstFrame();
        }
        if (ly.s > x.max_s or a > x.max_a) return error.VideoTooLong;
        const key: [6]u32 = .{ L, t, h, w, a, ly.cond };
        if (x.layout_key == null or !std.mem.eql(u32, &x.layout_key.?, &key)) {
            const pos = try x.gpa.alloc([3]f64, ly.s);
            defer x.gpa.free(pos);
            ly.positions(pos);
            const table = try x.gpa.alloc(u16, @as(usize, ly.s) * 48 * 4);
            defer x.gpa.free(table);
            ly.ropeTable(pos, &x.w.inv_freq, table);
            try x.upload(x.rope, std.mem.sliceAsBytes(table));
            x.layout_key = key;
            x.idx_key = null;
        }
        const sc = lay.stepScalars(sigma);
        var unique: [3]f32 = undefined;
        const mr = ly.modRows(sc, x.text_tags, &unique, x.idx_host[0..ly.s]);
        var temb: [3 * t_dim]f32 = undefined;
        lay.curveTEmb(x.w.table, grid, t_dim, unique[0..mr.m], temb[0 .. mr.m * t_dim]);
        const ik: [9]u32 = .{ L, t, h, w, a, ly.cond, mr.row_v, mr.row_a, mr.row_c };
        if (x.idx_key == null or !std.mem.eql(u32, &x.idx_key.?, &ik)) {
            try x.uploadAsync(x.idx, std.mem.sliceAsBytes(x.idx_host[0..ly.s]));
            x.idx_key = ik;
        }
        try x.uploadAsync(x.t_emb, std.mem.sliceAsBytes(temb[0 .. mr.m * t_dim]));
        const na: u64 = 32 * 2 * @as(u64, a);

        try x.pre("carry", &.{.{ .role = "x", .ptr = audio, .bytes = na * 2 }});
        try x.markBegin();
        try x.k.h3.scale(x.s, audio, sc.carry, x.a_in, na);
        try x.markEnd(.carry);
        try x.post("carry", &.{.{ .role = "y", .ptr = x.a_in, .bytes = na * 2 }});

        const bd: Body = .{ .video = video, .vel_v = vel_v, .vel_a = vel_a, .t = t, .h = h, .w = w, .a = a, .ly = ly, .m = mr.m, .row_v = mr.row_v, .row_a = mr.row_a };
        if (x.probe != null or x.prof != null or !x.graphs) try x.stepBody(bd) else try x.stepGraph(bd);

        try x.pre("uncarry", &.{ .{ .role = "a", .ptr = x.a_in, .bytes = na * 2 }, .{ .role = "v", .ptr = vel_a, .bytes = na * 2 } });
        try x.markBegin();
        try x.k.h3.uncarry(x.s, x.a_in, vel_a, sc.c1, sc.c2, na);
        try x.markEnd(.carry);
        try x.post("uncarry", &.{.{ .role = "y", .ptr = vel_a, .bytes = na * 2 }});
    }

    /// H3-Image (`initImage`): the target latent bf16 [24, 1, h, w] at PDD step `st` -> the velocity ComfyUI returns
    /// (negated), bf16 [24, 1, h, w], into `vel` (`MiniMaxH3ImageModel._forward`). [text | references | target]
    /// (`lay.ImageLayout`; the references are `imageRefs`', none for text to image), no audio and no carry; the text and
    /// the target at `st.t`, the references at max(t, 0.999); the final layer modulates the target with its own row and
    /// its head is the step's mix of the PDD bank (fp32, `sampler.mix`'s order) feeding the fp32 head GEMM. The RoPE
    /// table and the modulation-row map are cached per layout; eager (no step graph).
    pub fn stepImage(x: *Dit, latent: u64, st: pdd.Step, h: u32, w: u32, vel: u64) !void {
        const im = x.k.image orelse return error.NotAnImageDit;
        if (x.w.heads != pdd.max_heads or !x.attn_bf16) return error.NotAnImageDit;
        if (h == 0 or w == 0 or h % 2 != 0 or w % 2 != 0) return error.BadImageShape;
        const L = x.ctx_rows;
        const ly = try lay.ImageLayout.init(L, h, w, x.ref_shapes[0..x.n_refs]);
        if (ly.s > x.max_s) return error.ImageTooLarge;
        const key = ly.key();
        if (x.image_key == null or !std.mem.eql(u32, &x.image_key.?, &key)) {
            const pos = try x.gpa.alloc([3]f64, ly.s);
            defer x.gpa.free(pos);
            ly.positions(pos);
            const table = try x.gpa.alloc(u16, @as(usize, ly.s) * 48 * 4);
            defer x.gpa.free(table);
            lay.ropeRows(pos, &x.w.inv_freq, table);
            try x.upload(x.rope, std.mem.sliceAsBytes(table));
            x.image_key = key;
            x.image_idx_key = null;
            x.layout_key = null; // the video layouts' cache no longer describes the table
            x.idx_key = null;
        }
        var unique: [2]f32 = undefined;
        const mr = try ly.modRows(st.t, x.text_tags, &unique, x.idx_host[0..ly.s]);
        var temb: [2 * t_dim]f32 = undefined;
        lay.curveTEmb(x.w.table, grid, t_dim, unique[0..mr.m], temb[0 .. mr.m * t_dim]);
        var ik: [4 + 2 * lay.max_refs]u32 = undefined;
        @memcpy(ik[0 .. 3 + 2 * lay.max_refs], &key);
        ik[3 + 2 * lay.max_refs] = mr.m * 2 + mr.row_t; // the map depends on t only through these
        if (x.image_idx_key == null or !std.mem.eql(u32, &x.image_idx_key.?, &ik)) {
            try x.uploadAsync(x.idx, std.mem.sliceAsBytes(x.idx_host[0..ly.s]));
            x.image_idx_key = ik;
        }
        try x.uploadAsync(x.t_emb, std.mem.sliceAsBytes(temb[0 .. mr.m * t_dim]));

        const S: u64 = ly.s;
        const m: u64 = mr.m;
        const nv: u64 = @as(u64, h / 2) * (w / 2);
        const nx: u64 = 24 * @as(u64, h) * w;
        const vr = ly.targetRows();
        const rs = ly.refSpan();
        try x.markBegin();
        try x.pre("video_rows", &.{.{ .role = "x", .ptr = latent, .bytes = nx * 2 }});
        try x.k.h3.patchify(x.s, latent, x.rows, 24, 1, h, w);
        try x.post("video_rows", &.{.{ .role = "y", .ptr = x.rows, .bytes = nv * 96 * 4 }});
        try x.gemmF32("video_patch_proj", x.rows, x.w.vp_w, x.w.vp_b, x.emb, nv, dim, 96);
        try x.k.ops.copyRows(x.s, x.h, x.context, L, dim * 2, dim * 2, dim * 2); // [text | refs | target]
        x.extra_launches += 1;
        if (rs[1] > rs[0]) {
            const e = try x.ref_buf.?.at(0);
            try x.k.ops.copyRows(x.s, x.h + @as(u64, rs[0]) * dim * 2, e, rs[1] - rs[0], dim * 2, dim * 2, dim * 2);
            x.extra_launches += 1;
        }
        try x.k.h3.toBf16(x.s, x.emb, x.h + @as(u64, vr[0]) * dim * 2, nv * dim);
        try x.markEnd(.embed);

        if (x.fuse and x.probe == null) {
            try x.blocksFused(S, m);
        } else {
            for (&x.w.blocks, 0..) |*b, i| try x.block(i, b, S, m);
        }

        try x.markBegin();
        try x.gemmF32("final.adaln", x.t_emb, x.w.fl_ada_w, x.w.fl_ada_b, x.fada, m, 2 * dim, t_dim);
        const hv = x.h + @as(u64, vr[0]) * dim * 2;
        const shift = x.fada + @as(u64, mr.row_t) * 2 * dim * 4;
        try x.pre("final.mod_video", &.{.{ .role = "x", .ptr = hv, .bytes = nv * dim * 2 }});
        try x.k.h3.finalMod(x.s, hv, x.w.fl_norm, shift + dim * 4, shift, x.emb, nv, dim, eps);
        try x.post("final.mod_video", &.{.{ .role = "y", .ptr = x.emb, .bytes = nv * dim * 4 }});
        try x.pre("final.pdd_mix", &.{});
        try im.headMix(x.s, x.w.vo_w, x.head_w, 96 * dim, st);
        try im.headMix(x.s, x.w.vo_b, x.head_b, 96, st);
        try x.post("final.pdd_mix", &.{ .{ .role = "w", .ptr = x.head_w, .bytes = 96 * dim * 4 }, .{ .role = "b", .ptr = x.head_b, .bytes = 96 * 4 } });
        try x.gemmF32("final.video_out", x.emb, x.head_w, x.head_b, x.vout, nv, 96, dim);
        try x.pre("video_out", &.{.{ .role = "x", .ptr = x.vout, .bytes = nv * 96 * 4 }});
        try x.k.h3.unpatchifyNeg(x.s, x.vout, vel, 24, 1, h, w);
        try x.post("video_out", &.{.{ .role = "y", .ptr = vel, .bytes = nx * 2 }});
        try x.markEnd(.final);
    }

    /// softmax(q k^T / sqrt(128)) v of the rotated qkv rows [S, 3 * inner] into `att` [S, inner] with the bf16 attention
    /// (qwen_image's attention.cu; attention.hip on HIP): q, k and v copied out as contiguous [S, H, 128], as the token
    /// refiner lays them out.
    fn attentionBf16(x: *Dit, rows: u32) !void {
        std.debug.assert(rows <= x.max_s);
        const R: u64 = rows;
        try x.markBegin();
        inline for (.{ "q", "kk", "v" }, 0..) |dst, part|
            try x.k.ops.copyRows(x.s, @field(x, dst), x.qkv + part * inner * 2, R, inner * 2, 3 * inner * 2, inner * 2);
        try x.markEnd(.attn_prep);
        try x.markBegin();
        try x.k.ops.attention(x.s, x.q, x.kk, x.v, x.att, rows, heads, heads, rows, false, 0);
        try x.markEnd(.attn_kernel);
        x.extra_launches += 4;
    }

    /// The step graph of this shape and buffer set: replayed if captured; eager the first time a key is seen; captured
    /// (and launched) the second time. A capture that fails turns graphs off (`graph_error` says why) and runs eagerly.
    fn stepGraph(x: *Dit, bd: Body) !void {
        const gk: [14]u64 = .{ bd.ly.l, bd.t, bd.h, bd.w, bd.a, bd.m, bd.row_v, bd.row_a, bd.video, bd.vel_v, bd.vel_a, @intFromBool(x.fuse), bd.ly.cond, x.cond_rows };
        var at: ?usize = null;
        for (x.step_graphs.items, 0..) |sg, i| {
            if (std.mem.eql(u64, &sg.key, &gk)) {
                at = i;
                break;
            }
        }
        if (at) |i| if (x.step_graphs.items[i].exec) |e| return e.launchOn(x.s);
        if (at == null) {
            if (x.step_graphs.items.len >= max_graphs) {
                var old = x.step_graphs.orderedRemove(0);
                if (old.exec) |*e| e.deinit();
            }
            try x.step_graphs.append(x.gpa, .{ .key = gk, .exec = null });
            return x.stepBody(bd);
        }
        gpu.graph.beginCapture(x.s, .thread_local) catch |err| return x.graphFallback(err, bd);
        var failed: ?anyerror = null;
        x.stepBody(bd) catch |err| {
            failed = err;
        };
        var g = gpu.graph.endCapture(x.s) catch |err| {
            gpu.copy_source.captureEnd(x.s); // TensorFold ends its capture scope only on success; left open, it refuses every later copy
            return x.graphFallback(failed orelse err, bd);
        };
        defer g.deinit();
        if (failed) |err| return x.graphFallback(err, bd);
        const exec = g.instantiate() catch |err| return x.graphFallback(err, bd);
        x.step_graphs.items[at.?].exec = exec;
        try exec.launchOn(x.s);
    }

    fn graphFallback(x: *Dit, err: anyerror, bd: Body) !void {
        x.graphs = false;
        x.graph_error = err;
        return x.stepBody(bd);
    }

    /// The step's launches between the carry and the un-carry: the patch embeddings, the blocks, the final layer. Nothing
    /// in it depends on sigma but through `t_emb`'s contents (uploaded before), so a graph of it replays at any sigma.
    fn stepBody(x: *Dit, bd: Body) !void {
        const S: u64 = bd.ly.s;
        const m: u64 = bd.m;
        const t = bd.t;
        const h = bd.h;
        const w = bd.w;
        const a = bd.a;
        const na: u64 = 32 * 2 * @as(u64, a);
        const nv: u64 = @as(u64, t) * (h / 2) * (w / 2);
        const nvx: u64 = 24 * @as(u64, t) * h * w;
        const ar = bd.ly.audioRows();
        const vr = bd.ly.videoRows();

        try x.markBegin();
        try x.pre("video_rows", &.{.{ .role = "x", .ptr = bd.video, .bytes = nvx * 2 }});
        try x.k.h3.patchify(x.s, bd.video, x.rows, 24, t, h, w);
        try x.post("video_rows", &.{.{ .role = "y", .ptr = x.rows, .bytes = nv * 96 * 4 }});
        try x.pre("audio_rows", &.{.{ .role = "x", .ptr = x.a_in, .bytes = na * 2 }});
        try x.k.h3.packAudio(x.s, x.a_in, x.arows, 32, a);
        try x.post("audio_rows", &.{.{ .role = "y", .ptr = x.arows, .bytes = na * 4 }});
        try x.gemmF32("video_patch_proj", x.rows, x.w.vp_w, x.w.vp_b, x.emb, nv, dim, 96);
        try x.gemmF32("audio_patch_proj", x.arows, x.w.ap_w, x.w.ap_b, x.aemb, 2 * @as(u64, a), dim, 32);
        // [text | cond | audio | video] rows: the context, the image condition (if any), then the embeddings rounded to bf16
        try x.k.ops.copyRows(x.s, x.h, x.context, x.ctx_rows, dim * 2, dim * 2, dim * 2);
        x.extra_launches += 1;
        if (bd.ly.cond > 0) {
            try x.gemmF32("cond_patch_proj", x.cond_rows, x.w.vp_w, x.w.vp_b, x.cond_emb, bd.ly.cond, dim, 96);
            try x.k.h3.toBf16(x.s, x.cond_emb, x.h + @as(u64, x.ctx_rows) * dim * 2, @as(u64, bd.ly.cond) * dim);
        }
        try x.k.h3.toBf16(x.s, x.aemb, x.h + @as(u64, ar[0]) * dim * 2, 2 * @as(u64, a) * dim);
        try x.k.h3.toBf16(x.s, x.emb, x.h + @as(u64, vr[0]) * dim * 2, nv * dim);
        try x.markEnd(.embed);

        if (x.fuse and x.probe == null) {
            try x.blocksFused(S, m);
        } else {
            for (&x.w.blocks, 0..) |*b, i| try x.block(i, b, S, m);
        }

        // the final layer: fp32 curve modulation of each target segment, fp32 heads, negated, back to the latents
        try x.markBegin();
        try x.gemmF32("final.adaln", x.t_emb, x.w.fl_ada_w, x.w.fl_ada_b, x.fada, m, 2 * dim, t_dim);
        inline for (.{ .{ "final.mod_video", vr, bd.row_v, x.emb }, .{ "final.mod_audio", ar, bd.row_a, x.aemb } }) |e| {
            const rows: u64 = e[1][1] - e[1][0];
            const shift = x.fada + @as(u64, e[2]) * 2 * dim * 4;
            try x.pre(e[0], &.{.{ .role = "x", .ptr = x.h + @as(u64, e[1][0]) * dim * 2, .bytes = rows * dim * 2 }});
            try x.k.h3.finalMod(x.s, x.h + @as(u64, e[1][0]) * dim * 2, x.w.fl_norm, shift + dim * 4, shift, e[3], rows, dim, eps);
            try x.post(e[0], &.{.{ .role = "y", .ptr = e[3], .bytes = rows * dim * 4 }});
        }
        try x.gemmF32("final.video_out", x.emb, x.w.vo_w, x.w.vo_b, x.vout, nv, 96, dim);
        try x.gemmF32("final.audio_out", x.aemb, x.w.ao_w, x.w.ao_b, x.aout, 2 * @as(u64, a), 32, dim);
        try x.pre("video_out", &.{.{ .role = "x", .ptr = x.vout, .bytes = nv * 96 * 4 }});
        try x.k.h3.unpatchifyNeg(x.s, x.vout, bd.vel_v, 24, t, h, w);
        try x.post("video_out", &.{.{ .role = "y", .ptr = bd.vel_v, .bytes = nvx * 2 }});
        try x.pre("audio_out", &.{.{ .role = "x", .ptr = x.aout, .bytes = na * 4 }});
        try x.k.h3.unpackAudioNeg(x.s, x.aout, bd.vel_a, 32, a);
        try x.post("audio_out", &.{.{ .role = "y", .ptr = bd.vel_a, .bytes = na * 2 }});
        try x.markEnd(.final);
    }

    fn adalnGemm(x: *Dit, i: usize, dst: u64, m: u64) !void {
        const b = &x.w.blocks[i];
        try x.markBegin();
        try x.k.h3.gemmF32(x.s, x.t_emb, b.ada_w, b.ada_b, dst, m, ada_n, t_dim);
        try x.markEnd(.adaln);
    }

    /// The 50 blocks on the fast schedule (no probes): the reference `block`'s launches in the same order, except that (1)
    /// a gate_add and the norm_mod after it are one kernel, the gate_add1 + norm_mod2 of a block and its gate_add2 + the
    /// next block's norm_mod1 (each under its own block's modulation, so two modulation buffers alternate; the next
    /// block's adaln, which reads no activation, runs just before), and (2) the attention kernel stores rows, so the
    /// [H, S, D] -> rows pass is gone. Each kernel's arithmetic is the reference's (see ops.cu and kitchen_launch.cu).
    fn blocksFused(x: *Dit, S: u64, m: u64) !void {
        const rows: u32 = @intCast(S);
        const blks = &x.w.blocks;
        const mods = [2]u64{ x.mod, x.mod2 };
        try x.adalnGemm(0, mods[0], m);
        try x.markBegin();
        try x.k.h3.normMod(x.s, x.h, blks[0].norm1, mods[0], x.idx, x.hn, S, dim, 0, eps);
        try x.markEnd(.norm_mod);
        for (blks, 0..) |*b, i| {
            const cur = mods[i % 2];
            try x.linearRun(b.lin[0], x.hn, x.qkv, rows, .gemm_qkv);
            try x.markBegin();
            try x.k.kitchen.rmsRope(x.s, x.qkv, x.rope, b.q_norm, b.k_norm, heads, rows, 96, eps);
            try x.markEnd(.rms_rope);
            if (x.attn_bf16) {
                try x.attentionBf16(rows);
            } else {
                try x.markBegin();
                try x.k.kitchen.attentionPrep(x.s, x.qkv, x.ws, heads, rows);
                try x.markEnd(.attn_prep);
                try x.markBegin();
                try x.k.kitchen.attentionKernel(x.s, x.att, x.ws, heads, rows, true);
                try x.markEnd(.attn_kernel);
            }
            try x.linearRun(b.lin[1], x.att, x.o, rows, .gemm_out);
            try x.markBegin(); // gate_add1 + norm_mod2
            try x.k.h3.gateAddNormMod(x.s, x.h, x.o, cur, cur, x.idx, b.norm2, x.hn, S, dim, 0, 1, eps);
            try x.markEnd(.gate_norm);
            try x.linearRun(b.lin[2], x.hn, x.gu, rows, .gemm_fc1);
            try x.markBegin();
            try x.k.h3.swiglu(x.s, x.gu, x.act, S, ffn, false);
            try x.markEnd(.swiglu);
            try x.linearRun(b.lin[3], x.act, x.o, rows, .gemm_fc2);
            if (i + 1 < layers) {
                const nxt = mods[(i + 1) % 2];
                try x.adalnGemm(i + 1, nxt, m);
                try x.markBegin(); // gate_add2 + the next block's norm_mod1
                try x.k.h3.gateAddNormMod(x.s, x.h, x.o, cur, nxt, x.idx, blks[i + 1].norm1, x.hn, S, dim, 1, 0, eps);
                try x.markEnd(.gate_norm);
            } else {
                try x.markBegin();
                try x.k.h3.gateAdd(x.s, x.h, x.o, cur, x.idx, S, dim, 1);
                try x.markEnd(.gate_add);
            }
        }
    }

    /// The reference block: the twin's op sequence under its names, probed op by op.
    fn block(x: *Dit, i: usize, b: *const Block, S: u64, m: u64) !void {
        var nb: [24]u8 = undefined;
        const nm = struct {
            fn f(buf: []u8, j: usize, op: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "L{d}.{s}", .{ j, op }) catch unreachable;
            }
        }.f;
        const rows: u32 = @intCast(S);
        const xs = S * dim * 2;
        const mods = m * 3 * 6 * dim * 4;
        try x.markBegin();
        try x.gemmF32(nm(&nb, i, "adaln"), x.t_emb, b.ada_w, b.ada_b, x.mod, m, ada_n, t_dim);
        try x.markEnd(.adaln);
        inline for (.{ .{ "norm_mod1", 0, b.norm1 }, .{ "norm_mod2", 1, b.norm2 } }, 0..) |e, half| {
            const n = nm(&nb, i, e[0]);
            try x.pre(n, &.{ .{ .role = "x", .ptr = x.h, .bytes = xs }, .{ .role = "mod", .ptr = x.mod, .bytes = mods }, .{ .role = "idx", .ptr = x.idx, .bytes = S * 4 } });
            try x.markBegin();
            try x.k.h3.normMod(x.s, x.h, e[2], x.mod, x.idx, x.hn, S, dim, e[1], eps);
            try x.markEnd(.norm_mod);
            try x.post(n, &.{.{ .role = "y", .ptr = x.hn, .bytes = xs }});
            if (half == 0) { // attention half
                try x.linear(nm(&nb, i, "qkv"), b.lin[0], x.hn, x.qkv, rows, .gemm_qkv);
                const qkv_bytes = S * 3 * inner * 2;
                const rn = nm(&nb, i, "rms_rope");
                try x.pre(rn, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                try x.markBegin();
                try x.k.kitchen.rmsRope(x.s, x.qkv, x.rope, b.q_norm, b.k_norm, heads, rows, 96, eps);
                try x.markEnd(.rms_rope);
                try x.post(rn, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                const an = nm(&nb, i, "attention");
                try x.pre(an, &.{.{ .role = "qkv", .ptr = x.qkv, .bytes = qkv_bytes }});
                if (x.attn_bf16) {
                    try x.attentionBf16(rows);
                } else {
                    try x.markBegin();
                    try x.k.kitchen.attentionPrep(x.s, x.qkv, x.ws, heads, rows);
                    try x.markEnd(.attn_prep);
                    try x.markBegin();
                    try x.k.kitchen.attentionKernel(x.s, x.heads_out, x.ws, heads, rows, false);
                    try x.markEnd(.attn_kernel);
                    try x.markBegin();
                    try x.k.h3.headsToRows(x.s, x.heads_out, x.att, heads, S, head_dim);
                    try x.markEnd(.heads_rows);
                }
                try x.post(an, &.{.{ .role = "y", .ptr = x.att, .bytes = S * inner * 2 }});
                try x.linear(nm(&nb, i, "out"), b.lin[1], x.att, x.o, rows, .gemm_out);
            } else { // MLP half
                try x.linear(nm(&nb, i, "fc1"), b.lin[2], x.hn, x.gu, rows, .gemm_fc1);
                const sn = nm(&nb, i, "swiglu");
                try x.pre(sn, &.{.{ .role = "x", .ptr = x.gu, .bytes = S * 2 * ffn * 2 }});
                try x.markBegin();
                try x.k.h3.swiglu(x.s, x.gu, x.act, S, ffn, false);
                try x.markEnd(.swiglu);
                try x.post(sn, &.{.{ .role = "y", .ptr = x.act, .bytes = S * ffn * 2 }});
                try x.linear(nm(&nb, i, "fc2"), b.lin[3], x.act, x.o, rows, .gemm_fc2);
            }
            const gn = nm(&nb, i, if (half == 0) "gate_add1" else "gate_add2");
            try x.pre(gn, &.{ .{ .role = "x", .ptr = x.h, .bytes = xs }, .{ .role = "y", .ptr = x.o, .bytes = xs } });
            try x.markBegin();
            try x.k.h3.gateAdd(x.s, x.h, x.o, x.mod, x.idx, S, dim, e[1]);
            try x.markEnd(.gate_add);
            try x.post(gn, &.{.{ .role = "x", .ptr = x.h, .bytes = xs }});
        }
    }
};

test "the step's code type-checks (no GPU here: the h3-replay and h3-step-bench gates run it)" {
    _ = &Dit.step;
    _ = &Dit.prepare;
    _ = &Dit.init;
    _ = &Dit.launches;
    _ = &Dit.initImage;
    _ = &Dit.stepImage;
    _ = &Weights.loadImage;
}

test "H3-Image's scratch: no audio, NVFP4 or INT8 buffers; q, k, v for the whole sequence; the PDD head besides" {
    const rows: u32 = 1100; // a 1024 x 1024 image (1024 tokens) and a 76-token prompt
    const sc = Dit.imageScratch(rows, 512);
    try std.testing.expectEqual(@as(u64, rows) * inner * 2, sc.q);
    try std.testing.expectEqual(@as(u64, 0), sc.ws);
    try std.testing.expectEqual(@as(u64, 0), sc.qc);
    try std.testing.expect(Dit.imageScratchBytes(rows, 512) < Dit.scratchBytes(rows, 512, 0));
    try std.testing.expectEqual(sumBuffers(sc) + 96 * dim * 4 + 384, Dit.imageScratchBytes(rows, 512));
}
