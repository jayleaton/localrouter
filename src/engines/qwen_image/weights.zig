//! The DiT's weights on the device, from a pack (`pack.zig`), in the layouts the kernels read:
//!   NVFP4 linears: codes -> TensorFold's lane words (the `pack4` kernel), block scales -> [npad/64][K/64][64][4] (host)
//!   FP8 linears:   e4m3 -> the FP8 GEMM's fragment order (host); both with their static input scale `act`
//!   side weights:  bf16 as stored; the text norm's zero-centred weight + 1 in f32 (as the twin), q / k norms f32

const std = @import("std");
const cuda = @import("cuda");
const Pack = @import("pack.zig").Pack;
const Kind = @import("pack.zig").Kind;
const Tensor = @import("pack.zig").Tensor;
const block = @import("pack.zig").block;
const plan = @import("nvfp4_plan.zig");
const repack = @import("repack.zig");
const Nvfp4 = @import("nvfp4_exec.zig").Nvfp4;
const upload_mod = @import("upload.zig");
const Uploader = upload_mod.Uploader;
const Slab = upload_mod.Slab;

pub const layers = 32;
pub const dim = 4096;
pub const mlp = 12288;

/// One block linear on the device. `w` / `ws`: NVFP4 words and block scales, or FP8 bytes (ws 0).
pub const Linear = struct {
    kind: Kind,
    n: u32,
    k: u32,
    npad: u32,
    w: u64,
    ws: u64,
    scale: f64, // NVFP4 global scale, or the FP8 tensor scale
    act: f64, // the static input scale

    pub fn alpha(l: Linear) f32 {
        return plan.alpha(l.act, l.scale);
    }

    /// y[rows, n] = x[rows, k] . W^T: the rows quantized under the static input scale (NVFP4 or FP8, `qc` / `qs`
    /// scratch for the widest rows), then TensorFold's prompt GEMM at `tile` (the twin's call).
    pub fn forward(l: Linear, nv: *Nvfp4, major: u32, minor: u32, tile: u32, in: u64, out: u64, rows: u32, qc: u64, qs: u64, s: cuda.Stream) !void {
        const tb = plan.tb(tile, major, minor);
        const r = plan.rows(rows, l.k, tb);
        if (l.kind == .nvfp4) {
            const q = try plan.quant4(rows, l.k, l.k, tb, l.act);
            try nv.run(&q, .{ .x = in, .codes = qc, .scales = qs }, s);
            const g = try plan.gemmPrompt(.a4, tile, major, minor, rows, l.n, l.k, l.npad, r.mpad, false, l.alpha());
            try nv.run(&g, .{ .x = qc, .xs = qs, .w = l.w, .ws = l.ws, .out = out }, s);
        } else {
            const q = try plan.quant8(rows, l.k, l.k, tb, l.act);
            try nv.run(&q, .{ .x = in, .out = qc }, s);
            const g = try plan.gemmPrompt(.a8, tile, major, minor, rows, l.n, l.k, l.npad, r.mpad, false, l.alpha());
            try nv.run(&g, .{ .x = qc, .w = l.w, .out = out }, s);
        }
    }

    /// Output tiles [64 t0, 64 t1) of an NVFP4 linear as a view (gate and up halves of gate_up), as `Fp4Linear.tiles`.
    pub fn tiles(l: Linear, t0: u32, t1: u32) Linear {
        var v = l;
        v.w = l.w + @as(u64, t0) * l.k * 32; // a 64-row tile of words is 64 * K / 2 bytes
        v.ws = l.ws + @as(u64, t0) * (l.k / 64) * 256; // and of scales 64 * K / 16 bytes
        v.n = @min(l.n, 64 * t1) - 64 * t0;
        v.npad = 64 * (t1 - t0);
        return v;
    }
};

pub const Block = struct {
    qkv: Linear,
    out: Linear,
    gate_up: Linear,
    down: Linear,
    norm_q: u64, // f32 [128]
    norm_k: u64,
};

/// Device memory owned together, and the pack loaders every engine shares (`upload.zig` moves the bytes): plain
/// tensors, f32 scalars, and block linears in the layouts the kernels read. A load ends with `done`, which waits for
/// its copies and kernels and frees the load's scratch.
pub const Store = struct {
    mem: Slab,
    gpa: std.mem.Allocator,
    bytes: u64 = 0,
    // the load's scratch: host bytes, NVFP4 codes on the device before pack4, direct-I/O reads
    host: std.ArrayList(u8) = .empty,
    tmp: std.ArrayList(u8) = .empty,
    bulk: []align(block) u8 = &.{},
    staged: ?cuda.DeviceBuffer = null,
    stream: ?cuda.Stream = null, // where pack4 may still read `staged`

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator) Store {
        return .{ .mem = .init(d, gpa), .gpa = gpa };
    }

    pub fn deinit(w: *Store) void {
        w.freeScratch();
        w.mem.deinit();
    }

    /// Ends a load: waits for `up`'s stream (copies, pack4), then frees the scratch.
    pub fn done(w: *Store, up: *Uploader) !void {
        try up.s.synchronize();
        w.freeScratch();
    }

    fn freeScratch(w: *Store) void {
        if (w.staged) |*b| {
            if (w.stream) |s| s.synchronize() catch {};
            b.free();
        }
        w.staged = null;
        w.host.deinit(w.gpa);
        w.host = .empty;
        w.tmp.deinit(w.gpa);
        w.tmp = .empty;
        w.gpa.free(w.bulk);
        w.bulk = &.{};
    }

    pub fn alloc(w: *Store, n: usize) !u64 {
        const ptr = try w.mem.alloc(n);
        w.bytes += n;
        return ptr;
    }

    /// Host bytes to new device memory (through `up`'s chunks: `bytes` may be reused at once).
    pub fn upload(w: *Store, up: *Uploader, bytes: []const u8) !u64 {
        const ptr = try w.alloc(bytes.len);
        try up.bytes(bytes, ptr);
        return ptr;
    }

    /// Tensor `name` of `p`, as stored, to new device memory.
    pub fn tensor(w: *Store, up: *Uploader, p: *const Pack, name: []const u8) !u64 {
        return w.tensorOf(up, p, try p.get(name));
    }

    pub fn tensorOf(w: *Store, up: *Uploader, p: *const Pack, t: Tensor) !u64 {
        const ptr = try w.alloc(t.len());
        try up.tensor(p, t, ptr);
        return ptr;
    }

    /// A tensor's bytes on the host (for host-side layouts): direct I/O into scratch, valid until the next call.
    pub fn readBulk(w: *Store, io: std.Io, p: *const Pack, t: Tensor) ![]const u8 {
        const need = t.len() + 2 * block;
        if (w.bulk.len < need) {
            w.gpa.free(w.bulk);
            w.bulk = &.{};
            w.bulk = try w.gpa.alignedAlloc(u8, .fromByteUnits(block), need);
        }
        const from = try p.readAligned(io, t, 0, t.len(), w.bulk);
        return w.bulk[from..][0..t.len()];
    }

    pub fn scalar(io: std.Io, p: *const Pack, name: []const u8) !f64 {
        var b: [4]u8 = undefined;
        try p.read(io, try p.get(name), &b);
        return @as(f32, @bitCast(std.mem.readInt(u32, &b, .little)));
    }

    /// Device scratch of at least `n` bytes for codes before pack4; growing it first waits for the stream.
    fn stage(w: *Store, up: *Uploader, n: usize) !u64 {
        if (w.staged) |*b| {
            if (b.len >= n) return b.at(0);
            try up.s.synchronize();
            b.free();
            w.staged = null;
        }
        w.staged = try cuda.DeviceBuffer.alloc(w.mem.d, n);
        w.stream = up.s;
        return w.staged.?.at(0);
    }

    /// `repack.fp8Order` into an upload chunk, its 64-row blocks split over up to 4 threads.
    const Fp8Fill = struct {
        codes: []const u8,
        n: usize,
        k: usize,

        fn fill(c: Fp8Fill, at: usize, out: []u8) void {
            const blk = 64 * c.k;
            const b0 = at / blk;
            const nb = out.len / blk;
            const per = (nb + 3) / 4;
            var ts: [4]?std.Thread = @splat(null);
            var i: usize = 0;
            while (i * per < nb) : (i += 1) {
                const lo = i * per;
                const hi = @min(nb, lo + per);
                const args = .{ c.codes, c.n, c.k, b0 + lo, b0 + hi, out[lo * blk .. hi * blk] };
                ts[i] = std.Thread.spawn(.{}, repack.fp8OrderBlocks, args) catch blk2: {
                    @call(.auto, repack.fp8OrderBlocks, args);
                    break :blk2 null;
                };
            }
            for (ts) |t| if (t) |th| th.join();
        }
    };

    /// A block linear in the layout its GEMM reads: NVFP4 codes through `pack4` (queued on `up`'s stream after they
    /// arrive) and host-repacked block scales; FP8 in fragment order, made in parallel into the upload chunks.
    pub fn linear(w: *Store, io: std.Io, p: *const Pack, nv: *Nvfp4, up: *Uploader, key: []const u8) !Linear {
        var nb: [80]u8 = undefined;
        const kind = try p.kind(key);
        switch (kind) {
            .nvfp4 => {
                const codes = try p.get(try std.fmt.bufPrint(&nb, "{s}.codes", .{key}));
                const n: u32 = @intCast(codes.dim(0));
                const k: u32 = @intCast(codes.dim(1) * 2);
                const npad = (n + 63) / 64 * 64;
                const staged = try w.stage(up, codes.len());
                try up.tensor(p, codes, staged);
                const words = try w.alloc(plan.packedBytes(npad, k));
                const l = try plan.pack4(n, k, npad);
                try nv.run(&l, .{ .src = staged, .dst = words }, up.s);
                const sc = try p.get(try std.fmt.bufPrint(&nb, "{s}.scales", .{key}));
                try w.host.resize(w.gpa, sc.len());
                try p.read(io, sc, w.host.items);
                try w.tmp.resize(w.gpa, @as(usize, npad) * (k / 16));
                repack.nvfp4Scales(w.host.items, n, k, npad, w.tmp.items);
                return .{ .kind = .nvfp4, .n = n, .k = k, .npad = npad, .w = words, .ws = try w.upload(up, w.tmp.items), .scale = try Store.scalar(io, p, try std.fmt.bufPrint(&nb, "{s}.global", .{key})), .act = try Store.scalar(io, p, try std.fmt.bufPrint(&nb, "{s}.act", .{key})) };
            },
            .fp8 => {
                const w8 = try p.get(try std.fmt.bufPrint(&nb, "{s}.w8", .{key}));
                const n: u32 = @intCast(w8.dim(0));
                const k: u32 = @intCast(w8.dim(1));
                const npad = (n + 127) / 128 * 128;
                const codes = try w.readBulk(io, p, w8);
                const act = Store.scalar(io, p, try std.fmt.bufPrint(&nb, "{s}.act", .{key})) catch return error.Fp8PackWithoutStaticScales;
                const dst = try w.alloc(@as(usize, npad) * k);
                try up.produce(dst, @as(usize, npad) * k, 64 * @as(usize, k), Fp8Fill{ .codes = codes, .n = n, .k = k }, Fp8Fill.fill);
                return .{ .kind = .fp8, .n = n, .k = k, .npad = npad, .w = dst, .ws = 0, .scale = try Store.scalar(io, p, try std.fmt.bufPrint(&nb, "{s}.scale", .{key})), .act = act };
            },
            .bf16 => return error.Bf16LinearsUnsupported, // the reference precision stays in the twin
        }
    }
};

pub const Weights = struct {
    store: Store,
    precision: Kind,
    img_in: u64 = 0,
    txt_norm: u64 = 0, // f32, weight + 1
    txt_in1: u64 = 0,
    txt_in2: u64 = 0,
    t1: u64 = 0,
    t2: u64 = 0,
    mod: u64 = 0,
    norm_out: u64 = 0,
    proj_out: u64 = 0,
    blocks: [layers]Block = undefined,

    pub fn deinit(w: *Weights) void {
        w.store.deinit();
    }

    /// Loads every tensor the DiT reads through `up`; `nv` runs pack4 on its stream. Returns once all landed.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, p: *const Pack, nv: *Nvfp4, up: *Uploader) !Weights {
        var w: Weights = .{ .store = .init(d, gpa), .precision = std.meta.stringToEnum(Kind, p.precision) orelse .nvfp4 };
        errdefer w.deinit();
        const st = &w.store;
        const side = .{
            .{ "img_in.weight", &w.img_in },                                   .{ "txt_in.in_layer.weight", &w.txt_in1 },
            .{ "txt_in.out_layer.weight", &w.txt_in2 },                        .{ "time_text_embed.timestep_embedder.linear_1.weight", &w.t1 },
            .{ "time_text_embed.timestep_embedder.linear_2.weight", &w.t2 }, .{ "modulation.1.weight", &w.mod },
            .{ "norm_out.linear.weight", &w.norm_out },                        .{ "proj_out.weight", &w.proj_out },
        };
        inline for (side) |e| e[1].* = try st.tensor(up, p, e[0]);
        { // the zero-centred text norm: weight + 1 in f32
            const t = try p.get("txt_in.text_norm.weight");
            try st.host.resize(gpa, t.len());
            try p.read(io, t, st.host.items);
            const f: []align(1) f32 = std.mem.bytesAsSlice(f32, st.host.items);
            for (f) |*v| v.* += 1.0;
            w.txt_norm = try st.upload(up, st.host.items);
        }
        var name_buf: [64]u8 = undefined;
        for (0..layers) |i| {
            const b = &w.blocks[i];
            b.norm_q = try st.tensor(up, p, try std.fmt.bufPrint(&name_buf, "L{d}.norm_q", .{i}));
            b.norm_k = try st.tensor(up, p, try std.fmt.bufPrint(&name_buf, "L{d}.norm_k", .{i}));
            inline for (.{ "qkv", "out", "gate_up", "down" }) |nm| {
                const key = try std.fmt.bufPrint(&name_buf, "L{d}.{s}", .{ i, nm });
                @field(b, nm) = try st.linear(io, p, nv, up, key);
            }
        }
        try st.done(up);
        return w;
    }
};
