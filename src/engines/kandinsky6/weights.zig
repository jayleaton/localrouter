//! Kandinsky-6 weights from the published checkpoint (HF `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`, revision
//! 1018d5828716e8bea3de99609f45c924de9c5564) to the device, in the dtypes upstream computes in:
//!   transformer  BF16; its F32 tensors cast to BF16 once with round to nearest even (`runtime/weights.py:205-207`)
//!   text_encoder BF16 as stored (Qwen2.5-VL, sharded: `model.safetensors.index.json`)
//!   text_encoder_2, audio_vae, vocoder  F32 as stored
//!   vae          F32 cast to F16 (`postprocess_video.py`: the Hunyuan VAE loads as fp16), convolutions repacked by
//!                the video VAE's own `repackConv`
//! The root is the snapshot directory as published; nothing is converted on disk. Every tensor an engine binds is
//! checked against the component's own name/dtype/shape list before any byte moves, so a checkpoint that does not
//! match is refused at load, not discovered as wrong pixels.

const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const Pack = qi.pack.Pack;
const Tensor = qi.pack.Tensor;
const DType = qi.pack.DType;
const Uploader = qi.upload.Uploader;
const Slab = qi.upload.Slab;

/// Paths of the snapshot, relative to the root.
pub const paths = struct {
    pub const transformer = "transformer/diffusion_pytorch_model.safetensors";
    pub const text_encoder_index = "text_encoder/model.safetensors.index.json";
    pub const text_encoder_dir = "text_encoder";
    pub const qwen_tokenizer = "text_encoder/tokenizer.json";
    pub const clip = "text_encoder_2/model.safetensors";
    pub const clip_tokenizer = "tokenizer_2/tokenizer.json";
    pub const vae = "vae/diffusion_pytorch_model.safetensors";
    pub const audio_vae = "audio_vae/diffusion_pytorch_model.safetensors";
    pub const vocoder = "vocoder/diffusion_pytorch_model.safetensors";
};

/// Every file the engine reads, so `check` and the packaging can verify a snapshot without loading it.
pub const required = [_][]const u8{ paths.transformer, paths.text_encoder_index, paths.qwen_tokenizer, paths.clip, paths.clip_tokenizer, paths.vae, paths.audio_vae, paths.vocoder };

/// How a stored tensor becomes the device tensor.
pub const Cast = enum {
    as_stored,
    bf16, // F32 or BF16 in, BF16 out (RNE)
    f16, // F32 in, F16 out (RNE)
};

/// One component's checkpoint: a single safetensors file or the shards of an index.
pub const Source = struct {
    packs: []Pack,
    where: std.StringHashMapUnmanaged(u16) = .empty, // tensor name -> pack index
    arena: std.heap.ArenaAllocator,

    pub fn openFile(gpa: std.mem.Allocator, io: std.Io, root: []const u8, rel: []const u8) !Source {
        var s: Source = .{ .packs = &.{}, .arena = .init(gpa) };
        errdefer s.arena.deinit();
        const a = s.arena.allocator();
        const packs = try a.alloc(Pack, 1);
        packs[0] = try Pack.openFile(gpa, io, try std.fs.path.join(a, &.{ root, rel }));
        s.packs = packs;
        return s;
    }

    /// The shards an index names (`weight_map`: tensor -> file), each opened once.
    pub fn openIndex(gpa: std.mem.Allocator, io: std.Io, root: []const u8, dir: []const u8, index_rel: []const u8) !Source {
        var s: Source = .{ .packs = &.{}, .arena = .init(gpa) };
        errdefer s.arena.deinit();
        const a = s.arena.allocator();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, index_rel }), a, .limited(16 << 20));
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const map = (v.object.get("weight_map") orelse return error.BadIndex).object;
        var files: std.StringArrayHashMapUnmanaged(u16) = .empty;
        var it = map.iterator();
        while (it.next()) |kv| {
            const f = kv.value_ptr.string;
            const gop = try files.getOrPut(a, f);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(files.count() - 1);
            try s.where.put(a, kv.key_ptr.*, gop.value_ptr.*);
        }
        const packs = try a.alloc(Pack, files.count());
        var opened: usize = 0;
        errdefer for (packs[0..opened]) |*p| p.close(io);
        for (files.keys(), 0..) |f, i| {
            packs[i] = try Pack.openFile(gpa, io, try std.fs.path.join(a, &.{ root, dir, f }));
            opened += 1;
        }
        s.packs = packs;
        return s;
    }

    pub fn close(s: *Source, io: std.Io) void {
        for (s.packs) |*p| p.close(io);
        s.arena.deinit();
    }

    pub fn find(s: *const Source, name: []const u8) !struct { pack: *const Pack, t: Tensor } {
        const i: u16 = if (s.packs.len == 1) 0 else s.where.get(name) orelse return error.MissingTensor;
        const p = &s.packs[i];
        return .{ .pack = p, .t = try p.get(name) };
    }
};

/// Whether stored tensor `t` is the expected dtype and shape.
pub fn matches(t: Tensor, dtype: DType, shape: []const u64) bool {
    if (t.dtype != dtype or t.rank != shape.len) return false;
    for (shape, 0..) |d, i| if (t.shape[i] != d) return false;
    return true;
}

/// Device copies of one component's tensors, by name, in a slab; `get` answers the component binders' lookups.
pub const Store = struct {
    slab: Slab,
    addr: std.StringHashMapUnmanaged(u64) = .empty,
    arena: std.heap.ArenaAllocator,
    bytes: u64 = 0,

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator) Store {
        return .{ .slab = .init(d, gpa), .arena = .init(gpa) };
    }

    pub fn deinit(s: *Store) void {
        s.slab.deinit();
        s.arena.deinit();
    }

    pub fn get(s: *const Store, name: []const u8) ?u64 {
        return s.addr.get(name);
    }

    /// Device bytes of `t` under `cast`.
    pub fn castBytes(t: Tensor, cast: Cast) u64 {
        const n: u64 = t.len() / t.dtype.size();
        return switch (cast) {
            .as_stored => t.len(),
            .bf16, .f16 => n * 2,
        };
    }

    /// Uploads `name` from `src` under `cast`; `scratch` is a host buffer at least the stored size of the largest cast
    /// tensor (only casts use it). A name bound twice keeps its first copy.
    pub fn put(s: *Store, io: std.Io, up: *Uploader, src: *const Source, name: []const u8, cast: Cast, scratch: []u8) !u64 {
        if (s.addr.get(name)) |p| return p;
        const f = try src.find(name);
        const n = castBytes(f.t, cast);
        const dst = try s.slab.alloc(n);
        switch (cast) {
            .as_stored => try up.tensor(f.pack, f.t, dst),
            .bf16, .f16 => {
                if (f.t.dtype == .bf16 and cast == .bf16) {
                    try up.tensor(f.pack, f.t, dst);
                } else {
                    if (f.t.dtype != .f32) return error.UnsupportedCast;
                    if (scratch.len < f.t.len()) return error.ScratchTooSmall;
                    const raw = scratch[0..f.t.len()];
                    try f.pack.read(io, f.t, raw);
                    const src32 = std.mem.bytesAsSlice(f32, raw);
                    const out16 = std.mem.bytesAsSlice(u16, raw[0..n]); // narrowing in place, front to back
                    for (src32, 0..) |x, i| out16[i] = if (cast == .bf16) bf16Bits(x) else @bitCast(@as(f16, @floatCast(x)));
                    try up.bytes(raw[0..n], dst);
                }
            },
        }
        try s.addr.put(s.arena.allocator(), try s.arena.allocator().dupe(u8, name), dst);
        s.bytes += n;
        return dst;
    }

    /// Host-prepared bytes (a repacked convolution, a folded weight) as tensor `name`.
    pub fn putBytes(s: *Store, up: *Uploader, name: []const u8, data: []const u8) !u64 {
        const dst = try s.slab.alloc(data.len);
        try up.bytes(data, dst);
        try s.addr.put(s.arena.allocator(), try s.arena.allocator().dupe(u8, name), dst);
        s.bytes += data.len;
        return dst;
    }
};

/// F32 to BF16 with round to nearest even, as torch's `.to(torch.bfloat16)` (NaN kept quiet).
pub fn bf16Bits(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7fff +% lsb) >> 16);
}

test "bf16 rounding is nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Bits(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Bits(@bitCast(@as(u32, 0x3f808000)))); // tie to even (down)
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16Bits(@bitCast(@as(u32, 0x3f818000)))); // tie to even (up)
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16Bits(@bitCast(@as(u32, 0x3f808001))));
    try std.testing.expectEqual(@as(u16, 0xbf80), bf16Bits(-1.0));
}

test "a stored tensor matches its expected dtype and shape" {
    const t: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 4096, 1024, 1, 1, 1, 1 }, .begin = 0, .end = 4096 * 1024 * 2 };
    try std.testing.expect(matches(t, .bf16, &.{ 4096, 1024 }));
    try std.testing.expect(!matches(t, .f32, &.{ 4096, 1024 }));
    try std.testing.expect(!matches(t, .bf16, &.{ 4096, 1024, 1 }));
}
