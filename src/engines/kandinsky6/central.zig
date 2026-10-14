//! Launches of Kandinsky-6's central kernels (kernels/cuda/kandinsky6/central/central.cu): the seeded noise, the
//! PiFlow rollout of one outer step, the DiT's visual input and the RoPE cos / sin tables, on a caller-owned stream.
//! `Ops.load` takes the fatbin image the build embeds (`kandinsky6_kernels.central`) and the pipeline's
//! borrowed `cuda.Driver`; nothing here allocates, synchronizes or reads back. The host twins are noise.zig and
//! piflow.zig.
//!
//! One outer step of the sampler (piflow_sampler.py:90-174), batch 1, state buffers owned by the pipeline:
//!   video_in = videoInput(video state, reference or null)      [200,880, 33] in the state's dtype
//!   x_audio  = the audio state itself (bf16 at step 0, fp32 after; no kernel, `x_audio=audio`, :105)
//!   time_v = time_a = steps[i].time                            (piflow.ditTime, sigma_t_src * 1000 in f32)
//!   DiT -> pred_v bf16 [10][video_elems], pred_a bf16 [10][audio_elems] (grid-major)
//!   rollout(video state, pred_v, video out f32, steps[i], subs); the same for audio
//! Step 0's state is the bf16 noise (`randn` into its own buffer); the rollout writes fp32 to a separate buffer,
//! and from step 1 on it may run in place.

const std = @import("std");
const cuda = @import("cuda");
const shapes = @import("shapes.zig");
pub const noise = @import("noise.zig");
pub const piflow = @import("piflow.zig");

const block: u32 = 256;

/// Rows of the DiT's visual input (T * H * W latent rows) and of its first frame.
pub const video_rows: u64 = @as(u64, shapes.latent_frames) * shapes.latent_h * shapes.latent_w;
pub const frame_rows: u64 = @as(u64, shapes.latent_h) * shapes.latent_w;

pub const Dtype = enum(i32) { bf16 = 0, f32 = 1 };

/// A device buffer and its element type.
pub const Tensor = struct { ptr: u64, dtype: Dtype };

/// Kernel launches made through this file: a counter for benchmarks.
pub var count: u64 = 0;

fn go(f: cuda.Function, s: cuda.Stream, grid: u32, args: *cuda.launch.Args) cuda.Error!void {
    _ = @atomicRmw(u64, &count, .Add, 1, .monotonic);
    try cuda.launch.launch(f, .{ .grid = .{ .x = grid }, .block = .{ .x = block } }, s, args);
}

fn blocks(n: u64) cuda.Error!u32 {
    const g = (n + block - 1) / block;
    if (n == 0 or g > std.math.maxInt(i32)) return error.Invalid;
    return @intCast(g);
}

fn aligned(t: Tensor) bool {
    return t.ptr != 0 and t.ptr % @as(u64, if (t.dtype == .f32) 4 else 2) == 0;
}

fn isF32(t: Tensor) i32 {
    return @intFromBool(t.dtype == .f32);
}

/// The device properties the noise geometry needs (noise.Device). TensorFold's `DeviceAttribute` names the SM count
/// but not CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR (39), so that one goes through the same
/// cuDeviceGetAttribute entry point with the raw attribute number (the enum is a c_int in the C ABI).
pub fn device(ctx: *const cuda.Context) cuda.Error!noise.Device {
    const get: *const fn (*c_int, c_int, cuda.abi.Device) callconv(.c) cuda.abi.Result = @ptrCast(ctx.d.api.cuDeviceGetAttribute);
    var threads: c_int = 0;
    try ctx.d.check(get(&threads, 39, ctx.device), "cuDeviceGetAttribute");
    const sms = try ctx.attribute(.multiprocessor_count);
    if (sms <= 0 or threads < block) return error.Invalid;
    return .{ .sm_count = @intCast(sms), .max_threads_per_sm = @intCast(threads) };
}

pub const Ops = struct {
    module: cuda.Module,
    randn_fn: cuda.Function,
    rollout_fn: cuda.Function,
    video_input_fn: cuda.Function,
    rope_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) cuda.Error!Ops {
        if (image.len == 0) return error.Invalid;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var o: Ops = undefined;
        o.module = m;
        inline for (.{
            .{ "randn_fn", "k6_randn_bf16" },        .{ "rollout_fn", "k6_piflow_rollout" },
            .{ "video_input_fn", "k6_video_input" }, .{ "rope_fn", "k6_rope_cossin" },
        }) |e| @field(o, e[0]) = try m.function(e[1]);
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.module.unload();
    }

    /// out bf16 [d.numel] = torch.randn of the draw (noise.videoDraw / noise.audioDraw with `device(ctx)`).
    pub fn randn(o: *const Ops, s: cuda.Stream, out: u64, d: noise.Draw) cuda.Error!void {
        if (out == 0 or out % 2 != 0 or d.numel == 0 or d.numel > std.math.maxInt(i32)) return error.Invalid;
        var a: cuda.launch.Args = .{};
        a.add(out);
        a.add(@as(i64, @intCast(d.numel)));
        a.add(d.seed);
        a.add(d.offset);
        a.add(@as(f32, 0.0)); // mean
        a.add(@as(f32, 1.0)); // std
        try go(o.randn_fn, s, d.geo.grid, &a);
    }

    /// One outer step's rollout of one modality: out fp32 [n] from the step's state `src` [n] (bf16 or fp32) and
    /// the DiT's grid `pred` bf16 [10][n]. `subs` = piflow.substeps(cfg, step, ...). Steps longer than
    /// piflow.Table.capacity substeps run as launches that continue from `out`; those need `out` apart from `src`
    /// (one launch at the native schedule, where `out` may equal an fp32 `src`).
    pub fn rollout(o: *const Ops, s: cuda.Stream, src: Tensor, pred: u64, out: u64, n: u64, step: piflow.Step, subs: []const piflow.Substep) cuda.Error!void {
        if (!aligned(src) or pred == 0 or pred % 2 != 0 or out == 0 or out % 4 != 0 or subs.len != step.substeps or subs.len == 0) return error.Invalid;
        if (subs.len > piflow.Table.capacity and out == src.ptr) return error.Invalid;
        const grid = try blocks(n);
        var first: usize = 0;
        while (first < subs.len) : (first += piflow.Table.capacity) {
            const cur: Tensor = if (first == 0) src else .{ .ptr = out, .dtype = .f32 };
            var a: cuda.launch.Args = .{};
            a.add(src.ptr);
            a.add(isF32(src));
            a.add(cur.ptr);
            a.add(isF32(cur));
            a.add(pred);
            a.add(out);
            a.add(@as(i64, @intCast(n)));
            a.add(piflow.Table.of(step, subs, first));
            try go(o.rollout_fn, s, grid, &a);
        }
    }

    /// out [video_rows, 33] in the state's dtype = cat(state, cond, mask) (`pretrain`): `ref` the reference latent
    /// [frame_rows, 16] (bf16 or fp32, cast to the state's dtype) for I2V, null for T2V (cond and mask zero).
    pub fn videoInput(o: *const Ops, s: cuda.Stream, state: Tensor, ref: ?Tensor, out: u64) cuda.Error!void {
        if (!aligned(state) or out == 0 or out % 4 != 0) return error.Invalid;
        if (ref) |r| if (!aligned(r)) return error.Invalid;
        var a: cuda.launch.Args = .{};
        a.add(state.ptr);
        a.add(isF32(state));
        a.add(if (ref) |r| r.ptr else @as(u64, 0));
        a.add(if (ref) |r| isF32(r) else @as(i32, 0));
        a.add(out);
        a.add(@as(i64, @intCast(video_rows)));
        a.add(@as(i64, @intCast(frame_rows)));
        try go(o.video_input_fn, s, try blocks(video_rows * shapes.video_in_channels), &a);
    }

    /// out fp32 [rows, 64, 4] = (cos a, -sin a, sin a, cos a) of args fp32 [rows, 64] (rope.py:30, :74-75); out
    /// 16-byte aligned.
    pub fn ropeCosSin(o: *const Ops, s: cuda.Stream, args: u64, out: u64, rows: u64) cuda.Error!void {
        if (args == 0 or args % 4 != 0 or out == 0 or out % 16 != 0) return error.Invalid;
        var a: cuda.launch.Args = .{};
        a.add(args);
        a.add(out);
        a.add(@as(i64, @intCast(rows)));
        try go(o.rope_fn, s, try blocks(rows * 64), &a);
    }
};

test "the central host contract" {
    try std.testing.expectEqual(@as(u64, 200880), video_rows);
    try std.testing.expectEqual(@as(u64, 6480), frame_rows);
    try std.testing.expectEqual(shapes.video_elems, video_rows * shapes.latent_channels);
    try std.testing.expectEqual(@as(u32, 785), try blocks(video_rows * 1));
    try std.testing.expectError(error.Invalid, blocks(0));
    var steps: [shapes.steps]piflow.Step = undefined;
    piflow.schedule(.{}, &steps);
    for (steps) |st| try std.testing.expect(st.substeps <= piflow.Table.capacity); // one launch a step
    try std.testing.expect(aligned(.{ .ptr = 2, .dtype = .bf16 }) and !aligned(.{ .ptr = 2, .dtype = .f32 }));
}

test {
    _ = noise;
    _ = piflow;
}
