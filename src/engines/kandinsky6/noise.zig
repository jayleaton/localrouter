//! The starting latents as native upstream draws them: `torch.randn(shape, generator=g, device="cuda", dtype=bf16)`
//! with a fresh `g.manual_seed(seed)` for the video [31, 60, 108, 16] and `seed + 1` for the audio [218, 40]
//! (kandinsky/core/algo/prepare_latents.py:26-29, :58-60 at eac06e39). The GPU kernel (`k6_randn_bf16`,
//! kernels/cuda/kandinsky6/central/central.cu) is the product: it runs curand's Philox4x32-10 and Box-Muller as torch
//! 2.14.1 compiles them. This file is the host side: the launch geometry and philox offsets torch derives from the
//! device (`calc_execution_policy`, aten/src/ATen/native/cuda/DistributionTemplates.h:49-61 at v2.14.1), the
//! generator's offset bookkeeping (aten/src/ATen/cuda/CUDAGeneratorImpl.cpp:218-232, :300-310, :484-500) and a
//! bit-exact Philox / curand state model, so indexing and offsets are tested without a GPU. Box-Muller is not
//! modelled: the device takes `__sincosf` (MUFU) and libdevice `logf`, which have no exact host twin.
//!
//! The values depend on the device: the grid is min(SMs * (max threads an SM / 256), ceil(numel / 256)) and every
//! thread's stride is 256 * grid. GB10 (sm_121): 48 SMs, 1536 threads an SM, so grid 288 for the video.

const std = @import("std");
const shapes = @import("shapes.zig");

/// distribution_nullary_kernel's block (`block_size_bound`, :33), curand_normal4's four values a call
/// (unroll = sizeof(float4) / sizeof(float), :116) and the offset a curand call may consume (:37).
pub const block: u32 = 256;
pub const unroll: u32 = 4;
pub const offsets_per_call: u64 = 4;

pub const video_numel: u64 = shapes.video_elems;
pub const audio_numel: u64 = shapes.audio_elems;

/// The two device properties torch reads (`getCurrentDeviceProperties()`): multiProcessorCount and
/// maxThreadsPerMultiProcessor (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16, ..._MAX_THREADS_PER_MULTIPROCESSOR 39).
pub const Device = struct {
    sm_count: u32,
    max_threads_per_sm: u32,

    pub const gb10: Device = .{ .sm_count = 48, .max_threads_per_sm = 1536 };
};

/// One draw's launch: grid (block 256) and the per-thread philox increment the generator reserves for it.
pub const Geometry = struct {
    grid: u32,
    counter_offset: u64,

    pub fn stride(g: Geometry) u64 {
        return @as(u64, block) * g.grid;
    }

    /// curand_normal4 calls each thread makes (the grid-stride loop's trip count).
    pub fn calls(g: Geometry, numel: u64) u64 {
        return (numel - 1) / (g.stride() * unroll) + 1;
    }
};

/// calc_execution_policy (DistributionTemplates.h:49-61). numel > 0.
pub fn geometry(numel: u64, dev: Device) Geometry {
    std.debug.assert(numel > 0);
    const blocks_per_sm = dev.max_threads_per_sm / block;
    const want: u64 = (numel + block - 1) / block;
    const grid: u32 = @intCast(@min(@as(u64, dev.sm_count) * blocks_per_sm, want));
    return .{ .grid = grid, .counter_offset = ((numel - 1) / (@as(u64, block) * grid * unroll) + 1) * offsets_per_call };
}

/// One kernel launch of `k6_randn_bf16`: its arguments and grid.
pub const Draw = struct {
    seed: u64,
    offset: u64,
    numel: u64,
    geo: Geometry,
};

/// A torch.Generator(device="cuda") after `manual_seed(seed)`: offset 0 (CUDAGeneratorImpl.cpp:300-310). `draw`
/// hands out the current offset and advances it by the draw's counter offset rounded up to a multiple of 4
/// (philox_cuda_state :484-500, increase :218-232).
pub const Generator = struct {
    seed: u64,
    offset: u64 = 0,

    pub fn draw(g: *Generator, numel: u64, dev: Device) Draw {
        const geo = geometry(numel, dev);
        const d: Draw = .{ .seed = g.seed, .offset = g.offset, .numel = numel, .geo = geo };
        g.offset += (geo.counter_offset + 3) / 4 * 4;
        return d;
    }
};

/// The video draw (fresh generator at `seed`) and the audio draw (fresh generator at `seed + 1`, a Python int: a
/// seed of 2^64 - 1 has no u64 successor and torch refuses it).
pub fn videoDraw(seed: u64, dev: Device) Draw {
    var g: Generator = .{ .seed = seed };
    return g.draw(video_numel, dev);
}

pub fn audioDraw(seed: u64, dev: Device) error{SeedOverflow}!Draw {
    var g: Generator = .{ .seed = std.math.add(u64, seed, 1) catch return error.SeedOverflow };
    return g.draw(audio_numel, dev);
}

// Philox4x32-10 (curand_philox4x32_x.h, CUDA 13; Random123's philox4x32_R(10)).
const m0: u32 = 0xD2511F53;
const m1: u32 = 0xCD9E8D57;
const w0: u32 = 0x9E3779B9;
const w1: u32 = 0xBB67AE85;

fn round(c: [4]u32, k: [2]u32) [4]u32 {
    const p0 = @as(u64, m0) * c[0];
    const p1 = @as(u64, m1) * c[2];
    const hi0: u32 = @truncate(p0 >> 32);
    const hi1: u32 = @truncate(p1 >> 32);
    return .{ hi1 ^ c[1] ^ k[0], @truncate(p1), hi0 ^ c[3] ^ k[1], @truncate(p0) };
}

pub fn philox(ctr: [4]u32, key: [2]u32) [4]u32 {
    var c = ctr;
    var k = key;
    for (0..9) |_| {
        c = round(c, k);
        k[0] +%= w0;
        k[1] +%= w1;
    }
    return round(c, k);
}

/// curandStatePhilox4_32_10_t: curand_init (curand_kernel.h, `skipahead_sequence` then `skipahead`) and curand4,
/// with curand's counter arithmetic as written (Philox_State_Incr / _hi).
pub const State = struct {
    ctr: [4]u32,
    key: [2]u32,
    output: [4]u32,
    pos: u32,

    pub fn init(seed: u64, subsequence: u64, offset: u64) State {
        var s: State = .{ .ctr = .{ 0, 0, 0, 0 }, .key = .{ @truncate(seed), @truncate(seed >> 32) }, .output = undefined, .pos = 0 };
        s.incrHi(subsequence);
        s.output = philox(s.ctr, s.key);
        s.pos += @intCast(offset & 3);
        var n = offset / 4;
        if (s.pos > 3) {
            n += 1;
            s.pos -= 4;
        }
        s.incrBy(n);
        s.output = philox(s.ctr, s.key);
        return s;
    }

    fn incrHi(s: *State, n: u64) void {
        const nlo: u32 = @truncate(n);
        var nhi: u32 = @truncate(n >> 32);
        s.ctr[2] +%= nlo;
        if (s.ctr[2] < nlo) nhi +%= 1;
        s.ctr[3] +%= nhi;
    }

    fn incrBy(s: *State, n: u64) void {
        const nlo: u32 = @truncate(n);
        var nhi: u32 = @truncate(n >> 32);
        s.ctr[0] +%= nlo;
        if (s.ctr[0] < nlo) nhi +%= 1;
        s.ctr[1] +%= nhi;
        if (nhi <= s.ctr[1]) return;
        s.ctr[2] +%= 1;
        if (s.ctr[2] != 0) return;
        s.ctr[3] +%= 1;
    }

    fn incr(s: *State) void {
        for (&s.ctr) |*c| {
            c.* +%= 1;
            if (c.* != 0) return;
        }
    }

    pub fn next4(s: *State) [4]u32 {
        const tmp = s.output;
        s.incr();
        s.output = philox(s.ctr, s.key);
        const o = s.output;
        return switch (s.pos) {
            0 => tmp,
            1 => .{ tmp[1], tmp[2], tmp[3], o[0] },
            2 => .{ tmp[2], tmp[3], o[0], o[1] },
            else => .{ tmp[3], o[0], o[1], o[2] },
        };
    }
};

/// Where element `li` comes from: thread, its curand_normal4 call, and the float4 lane (the grid-stride kernel,
/// DistributionTemplates.h:75-85: lane ii of call m of thread t is element t + stride * (4m + ii)).
pub const Source = struct { thread: u64, call: u64, lane: u2 };

pub fn source(geo: Geometry, li: u64) Source {
    const q = li / geo.stride();
    return .{ .thread = li % geo.stride(), .call = q / unroll, .lane = @intCast(q % unroll) };
}

/// The Philox words behind element `li`: curand_box_muller4 feeds (x, y) of its curand4 result into
/// _curand_box_muller for lanes 0 (sin) and 1 (cos), (z, w) for lanes 2 (sin) and 3 (cos) (curand_normal.h:70-87,
/// :165-178). `x` gives the radius log, `y` the angle.
pub const Words = struct { x: u32, y: u32, cos: bool };

pub fn words(d: Draw, li: u64) Words {
    const src = source(d.geo, li);
    var s = State.init(d.seed, src.thread, d.offset);
    var r = s.next4();
    var m: u64 = 0;
    while (m < src.call) : (m += 1) r = s.next4();
    const pair: usize = @as(usize, src.lane) / 2 * 2;
    return .{ .x = r[pair], .y = r[pair + 1], .cos = src.lane % 2 == 1 };
}

test "Philox4x32-10 known answers (Random123 kat_vectors)" {
    try std.testing.expectEqual([4]u32{ 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8 }, philox(.{ 0, 0, 0, 0 }, .{ 0, 0 }));
    const f = 0xffffffff;
    try std.testing.expectEqual([4]u32{ 0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd }, philox(.{ f, f, f, f }, .{ f, f }));
    try std.testing.expectEqual(
        [4]u32{ 0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1 },
        philox(.{ 0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344 }, .{ 0xa4093822, 0x299f31d0 }),
    );
}

test "curand state: subsequence, offset and counter carries" {
    const seed: u64 = 0x0123456789abcdef;
    const key: [2]u32 = .{ 0x89abcdef, 0x01234567 };
    var s = State.init(seed, 0x1_0000_0005, 0);
    try std.testing.expectEqual(philox(.{ 0, 0, 5, 1 }, key), s.next4());
    try std.testing.expectEqual(philox(.{ 1, 0, 5, 1 }, key), s.next4());
    // offset 8: two counters skipped; offset 6: half a counter, the lanes straddle two outputs
    s = State.init(seed, 7, 8);
    try std.testing.expectEqual(philox(.{ 2, 0, 7, 0 }, key), s.next4());
    s = State.init(seed, 7, 6);
    const a = philox(.{ 1, 0, 7, 0 }, key);
    const b = philox(.{ 2, 0, 7, 0 }, key);
    try std.testing.expectEqual([4]u32{ a[2], a[3], b[0], b[1] }, s.next4());
    // the low counter word carries into the next
    s = State.init(seed, 0, 4 * @as(u64, 0xffffffff));
    try std.testing.expectEqual(philox(.{ 0xffffffff, 0, 0, 0 }, key), s.next4());
    try std.testing.expectEqual(philox(.{ 0, 1, 0, 0 }, key), s.next4());
}

test "launch geometry and offsets of the two native draws on GB10" {
    const v = geometry(video_numel, Device.gb10);
    try std.testing.expectEqual(@as(u64, 3214080), video_numel);
    try std.testing.expectEqual(@as(u32, 288), v.grid);
    try std.testing.expectEqual(@as(u64, 11), v.calls(video_numel));
    try std.testing.expectEqual(@as(u64, 44), v.counter_offset);
    const a = geometry(audio_numel, Device.gb10);
    try std.testing.expectEqual(@as(u64, 8720), audio_numel);
    try std.testing.expectEqual(@as(u32, 35), a.grid);
    try std.testing.expectEqual(@as(u64, 1), a.calls(audio_numel));
    try std.testing.expectEqual(@as(u64, 4), a.counter_offset);
    // the grid, so the values, follow the device: 132 SMs x 2048 threads
    const h = geometry(video_numel, .{ .sm_count = 132, .max_threads_per_sm = 2048 });
    try std.testing.expectEqual(@as(u32, 1056), h.grid);
    try std.testing.expectEqual(@as(u64, 12), h.counter_offset);
}

test "fresh generators start at offset 0 and advance by the draw's counter offset" {
    const v = videoDraw(42, Device.gb10);
    try std.testing.expectEqual(@as(u64, 0), v.offset);
    try std.testing.expectEqual(@as(u64, 42), v.seed);
    const a = try audioDraw(42, Device.gb10);
    try std.testing.expectEqual(@as(u64, 0), a.offset);
    try std.testing.expectEqual(@as(u64, 43), a.seed);
    try std.testing.expectError(error.SeedOverflow, audioDraw(std.math.maxInt(u64), Device.gb10));
    var g: Generator = .{ .seed = 1 };
    _ = g.draw(video_numel, Device.gb10);
    try std.testing.expectEqual(@as(u64, 44), g.offset);
    try std.testing.expectEqual(@as(u64, 44), g.draw(audio_numel, Device.gb10).offset);
    try std.testing.expectEqual(@as(u64, 48), g.offset);
}

test "every element has one source, in torch's grid-stride order" {
    const v = geometry(video_numel, Device.gb10);
    try std.testing.expectEqual(Source{ .thread = 0, .call = 0, .lane = 0 }, source(v, 0));
    try std.testing.expectEqual(Source{ .thread = 0, .call = 0, .lane = 1 }, source(v, 73728));
    try std.testing.expectEqual(Source{ .thread = 5, .call = 1, .lane = 0 }, source(v, 4 * 73728 + 5));
    try std.testing.expectEqual(Source{ .thread = 43775, .call = 10, .lane = 3 }, source(v, video_numel - 1));
    // a small draw: (thread, call, lane) covers [0, numel) exactly once
    const g = geometry(5000, .{ .sm_count = 1, .max_threads_per_sm = 512 });
    try std.testing.expectEqual(@as(u32, 2), g.grid);
    var seen: [5000]bool = @splat(false);
    for (0..g.stride()) |t| for (0..g.calls(5000)) |m| for (0..4) |ii| {
        const li = t + g.stride() * (4 * m + ii);
        if (li >= 5000) continue;
        try std.testing.expect(!seen[li]);
        seen[li] = true;
        try std.testing.expectEqual(Source{ .thread = t, .call = m, .lane = @intCast(ii) }, source(g, li));
    };
    for (seen) |s| try std.testing.expect(s);
}

test "the words behind an element" {
    const d = videoDraw(7, Device.gb10);
    const key: [2]u32 = .{ 7, 0 };
    const r0 = philox(.{ 0, 0, 3, 0 }, key);
    try std.testing.expectEqual(Words{ .x = r0[0], .y = r0[1], .cos = false }, words(d, 3));
    try std.testing.expectEqual(Words{ .x = r0[2], .y = r0[3], .cos = true }, words(d, 3 + 3 * 73728));
    const r2 = philox(.{ 2, 0, 3, 0 }, key);
    try std.testing.expectEqual(Words{ .x = r2[0], .y = r2[1], .cos = true }, words(d, 3 + 9 * 73728));
}
