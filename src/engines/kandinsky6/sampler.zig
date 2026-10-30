//! Kandinsky-6 Pro-distill's multimodal PiFlow sampler (upstream kandinsky-6 eac06e39
//! `kandinsky/core/algo/piflow_sampler.py` `piflow_denoise_loop`, `kandinsky/core/algo/piflow_math.py` `DXPolicy`,
//! `shift_timesteps`, `policy_rollout_fm`; settings `kandinsky/configs/checkpoints/pro-distill.yaml` `piflow`).
//!
//! Host: the schedule (`Schedule`, one `Step` a DiT call) and each step's policy rollout (`Rollout`, one `Substep` an
//! Euler substep), as the f32 tensors upstream holds them. Upstream's batch is B = 1 with the video [1, T, H, W, 16] and
//! audio [1, A, 40] states (pipeline.py:488-532 reshape them, so the loop takes the `video.ndim == 5` branch): every
//! schedule tensor has one element, and each scalar here is that element. Python floats (num_steps, the base segment,
//! shift, eps) stay f64 where Python computes with them and become f32 where torch converts a Python scalar for an
//! f32 tensor op (`f32` below, round to nearest even). Zig does not contract floats, so each expression rounds where
//! torch's separate tensor op rounds; every division is a true IEEE division (no divisor here is a CPU scalar, so
//! torch's reciprocal-multiply path is never taken); `clampMin` keeps torch's NaN propagation; `rint` is torch.round
//! (half to even).
//!
//! Device: `Kernels` launches kernels/cuda/kandinsky6/sampler/policy.cu on the pipeline's stream: one launch a
//! substep a modality (`substep`), and the DiT's visual input (`videoInput`). The policy's per-grid x0 is recomputed
//! in each substep from the step's input state and the DiT grids, with upstream's operations (bits identical to the
//! tensor upstream materializes once), so the sampler holds no [10, ...] f32 copy. Video and audio advance together
//! every step (`advance` takes both); a video-only or audio-only step is not offered, as upstream refuses
//! `sample_audio = False` for PiFlow (pipeline.py:356-358). Nothing here allocates, synchronizes or reads back.

const std = @import("std");
const cuda = @import("cuda");

pub const Error = cuda.Error || error{ InvalidConfig, NonFinite, UnsupportedDtype, ViewMismatch, Cancelled };

/// The `piflow` settings (PiFlowConfig, kandinsky/runtime/sampler.py:16-24) and the step count. `n_grid` is the DiT
/// head's grid count (dx_num_grid_points; the grids the DiT returns).
pub const Config = struct {
    num_steps: u32,
    shift: f64,
    eps: f64,
    final_step_size_scale: f64,
    num_policy_substeps: u32,
    n_grid: u32,

    /// pro-distill.yaml: generation.num_steps 10, piflow shift 5.0, eps 1e-6, final_step_size_scale 0.5,
    /// num_policy_substeps 128, dx_num_grid_points 10.
    pub const distill: Config = .{ .num_steps = 10, .shift = 5.0, .eps = 1e-6, .final_step_size_scale = 0.5, .num_policy_substeps = 128, .n_grid = 10 };

    /// PiFlowConfig's field bounds plus piflow_denoise_loop's checks (num_steps >= 1, substeps >= 1).
    pub fn validate(c: Config) Error!void {
        if (c.num_steps < 1 or c.num_policy_substeps < 1 or c.n_grid < 2) return error.InvalidConfig;
        inline for (.{ c.shift, c.eps, c.final_step_size_scale }) |v| if (!std.math.isFinite(v)) return error.InvalidConfig;
        if (!(c.shift > 0) or !(c.eps > 0) or !(c.final_step_size_scale > 0) or c.final_step_size_scale > 1) return error.InvalidConfig;
        // Every f32 the schedule makes must be finite (f32(eps) > 0, f32(shift) finite); and the substep count
        // (at most num_policy_substeps, since tau stays in (0, 1]) must convert to f32 exactly, as torch's long -> f32.
        if (!(f32of(c.eps) > 0) or !std.math.isFinite(f32of(c.shift)) or c.num_policy_substeps > 1 << 24) return error.InvalidConfig;
    }

    /// `base_segment_size` (piflow_sampler.py:84-86), in Python floats.
    fn baseSegment(c: Config) f64 {
        const fss = @max(c.final_step_size_scale, c.eps);
        const one_minus_final = 1.0 - fss;
        return 1.0 / (@as(f64, @floatFromInt(c.num_steps)) - one_minus_final);
    }
};

/// A Python float as torch converts it for an f32 tensor op (static_cast, round to nearest even).
fn f32of(v: f64) f32 {
    return @floatCast(v);
}

/// torch.clamp(min=lo) on f32: NaN stays NaN.
fn clampMin(v: f32, lo: f32) f32 {
    return if (std.math.isNan(v)) v else @max(v, lo);
}

/// torch.clamp(min=lo, max=hi) on f32: min(max(v, lo), hi), NaN stays NaN.
fn clamp(v: f32, lo: f32, hi: f32) f32 {
    return if (std.math.isNan(v)) v else @min(@max(v, lo), hi);
}

/// torch.round (nearbyint, half to even). `x` finite.
fn rint(x: f32) f32 {
    const f = @floor(x);
    const d = x - f; // exact: f <= x < f + 1 and both share x's exponent range
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1.0;
    return if (@rem(f, 2.0) == 0) f else f + 1.0;
}

/// shift_timesteps (piflow_math.py:150): shift * t / (1 + (shift - 1) * t), one tensor op each.
pub fn shiftTimesteps(t: f32, shift: f64) f32 {
    const num = t * f32of(shift);
    const den = t * f32of(shift - 1.0) + 1.0;
    return num / den;
}

/// DXPolicy._unwarp_t (piflow_math.py:50): sigma / (shift + (1 - shift) * sigma).
pub fn unwarp(sigma: f32, shift: f64) f32 {
    const den = sigma * f32of(1.0 - shift) + f32of(shift);
    return sigma / den;
}

/// One DiT call of the loop (piflow_sampler.py:90-174) and the policy built from it (piflow_math.py:29-36).
pub const Step = struct {
    index: u32,
    final: bool,
    segment: f32, // f32(segment_size): the scalar of `tau_src - segment_size` and the `torch.full` segment tensors
    tau_src: f32, // raw_t_src: the rollout's raw_t_start
    tau_dst: f32, // clamp(tau_src - segment, eps): the rollout's raw_t_end, and the next step's tau_src
    sigma: f32, // shift_timesteps(tau_src): the DiT's sigma and the policy's sigma_t_src
    time: f32, // sigma * 1000: the DiT's `time` (both modalities)
    policy_raw_src: f32, // DXPolicy.raw_t_src = _unwarp_t(sigma)
    policy_raw_dst: f32, // clamp(policy_raw_src - segment, 0)
    policy_segment: f32, // clamp(policy_raw_src - policy_raw_dst, eps)
    substeps: u32, // clamp(round((tau_src - tau_dst) * num_policy_substeps), 1)
    substep_size: f32, // (tau_src - tau_dst) / substeps
};

/// The steps in order: `next` until null.
pub const Schedule = struct {
    c: Config,
    base: f64,
    step: u32 = 0,
    tau: f32 = 1.0, // torch.ones(batch_size, dtype=float32)

    pub fn init(c: Config) Error!Schedule {
        try c.validate();
        return .{ .c = c, .base = c.baseSegment() };
    }

    pub fn next(s: *Schedule) Error!?Step {
        const c = s.c;
        if (s.step == c.num_steps) return null;
        const final = s.step == c.num_steps - 1;
        const eps = f32of(c.eps);
        const seg = f32of(s.base * (if (final) @max(c.final_step_size_scale, c.eps) else 1.0));
        const tau_src = s.tau;
        const tau_dst = clampMin(tau_src - seg, eps);
        const sigma = shiftTimesteps(tau_src, c.shift);
        const raw_src = unwarp(sigma, c.shift);
        const raw_dst = clampMin(raw_src - seg, 0);
        const delta = tau_src - tau_dst;
        const scaled = delta * @as(f32, @floatFromInt(c.num_policy_substeps));
        if (!std.math.isFinite(scaled)) return error.NonFinite;
        const n = @max(rint(scaled), 1.0);
        const st: Step = .{
            .index = s.step,
            .final = final,
            .segment = seg,
            .tau_src = tau_src,
            .tau_dst = tau_dst,
            .sigma = sigma,
            .time = sigma * 1000.0,
            .policy_raw_src = raw_src,
            .policy_raw_dst = raw_dst,
            .policy_segment = clampMin(raw_src - raw_dst, eps),
            .substeps = @intFromFloat(n),
            .substep_size = delta / n,
        };
        inline for (@typeInfo(Step).@"struct".field_names, @typeInfo(Step).@"struct".field_types) |name, T| {
            if (T == f32) {
                if (!std.math.isFinite(@field(st, name))) return error.NonFinite;
            }
        }
        s.tau = tau_dst;
        s.step += 1;
        return st;
    }
};

/// One Euler substep of policy_rollout_fm (piflow_math.py:178-187) with the policy query inside it
/// (DXPolicy.pi, `_interpolate`, piflow_math.py:39-48, 61-74).
pub const Substep = struct {
    index: u32,
    sigma: f32, // sigma_t: pi's argument
    t: f32, // (_unwarp_t(sigma) - raw_t_dst) / segment_size: _interpolate's argument
    g0: u32, // clamp(floor(clamp(t, 0, 1) * (n_grid - 1)), 0, n_grid - 2)
    w0: f32, // (g0 + 1) - clamp(t, 0, 1) * (n_grid - 1)
    w1: f32, // clamp(t, 0, 1) * (n_grid - 1) - g0
    sigma_clamped: f32, // sigma.clamp(min=eps): pi's divisor
    raw_next: f32, // clamp(raw_t - substep_size, 0)
    sigma_next: f32, // shift_timesteps(raw_next)
    dsigma: f32, // sigma_next - sigma: the Euler factor
};

/// A step's substeps in order: `next` until null. Every substep is active: with B = 1 the active mask
/// `num_substeps > substep_id` holds for each of the max_num_substeps iterations.
pub const Rollout = struct {
    c: Config,
    st: Step,
    k: u32 = 0,
    raw: f32,
    sigma: f32,

    pub fn init(c: Config, st: Step) Rollout {
        return .{ .c = c, .st = st, .raw = st.tau_src, .sigma = st.sigma };
    }

    pub fn next(r: *Rollout) Error!?Substep {
        if (r.k == r.st.substeps) return null;
        const c = r.c;
        const raw_t = unwarp(r.sigma, c.shift);
        const t = (raw_t - r.st.policy_raw_dst) / r.st.policy_segment;
        if (!std.math.isFinite(t)) return error.NonFinite;
        const tc = clamp(t, 0, 1) * @as(f32, @floatFromInt(c.n_grid - 1));
        const fl = @floor(tc);
        const g0: u32 = @intFromFloat(@min(@max(fl, 0), @as(f32, @floatFromInt(c.n_grid - 2))));
        const raw_next = clampMin(r.raw - r.st.substep_size, 0);
        const sigma_next = shiftTimesteps(raw_next, c.shift);
        const sub: Substep = .{
            .index = r.k,
            .sigma = r.sigma,
            .t = t,
            .g0 = g0,
            .w0 = @as(f32, @floatFromInt(g0 + 1)) - tc,
            .w1 = tc - @as(f32, @floatFromInt(g0)),
            .sigma_clamped = clampMin(r.sigma, f32of(c.eps)),
            .raw_next = raw_next,
            .sigma_next = sigma_next,
            .dsigma = sigma_next - r.sigma,
        };
        inline for (@typeInfo(Substep).@"struct".field_names, @typeInfo(Substep).@"struct".field_types) |name, T| {
            if (T == f32) {
                if (!std.math.isFinite(@field(sub, name))) return error.NonFinite;
            }
        }
        r.raw = raw_next;
        r.sigma = sigma_next;
        r.k += 1;
        return sub;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Device

/// Element types the policy kernels take (upstream: the noise is bf16 at step 0, the state f32 after it; the DiT grids
/// bf16). Any other dtype is refused.
pub const Dtype = enum(i32) {
    bf16 = 0,
    f32 = 1,

    pub fn size(d: Dtype) u64 {
        return switch (d) {
            .bf16 => 2,
            .f32 => 4,
        };
    }
};

/// A contiguous device tensor of `n` elements.
pub const Tensor = struct {
    ptr: u64,
    dtype: Dtype,
    n: u64,

    pub fn bytes(t: Tensor) u64 {
        return t.n * t.dtype.size();
    }
};

/// One modality of a step: the state the DiT saw (`src`, bf16 at step 0 or f32), the DiT's grids for it, grid-major
/// [n_grid, n] (the DiT ABI's `Outputs`: `view(.., n_grid, dx).movedim(-2, 1)` made contiguous, which for B = 1 is
/// upstream's normalized [1, n_grid, ...] prediction), and the f32 state the rollout writes (`out`, n elements; never
/// `src`). After the step `out` is the next step's `src`.
pub const Modality = struct {
    src: Tensor,
    pred: Tensor,
    out: u64,
};

const threads: u32 = 256;

pub const Kernels = struct {
    mod: cuda.Module,
    substep_fn: [2][2]cuda.Function, // [src dtype][pred dtype]
    video_input_fn: [2]cuda.Function, // [video dtype]

    /// `image`: the policy.cu fatbin the build embeds (empty in host-only builds: refused).
    pub fn load(d: *const cuda.Driver, image: []const u8) Error!Kernels {
        if (image.len == 0) return error.Invalid;
        var k: Kernels = undefined;
        k.mod = try cuda.Module.load(d, image);
        errdefer k.mod.unload();
        k.substep_fn[0][0] = try k.mod.function("k6_piflow_substep_bf16_bf16");
        k.substep_fn[0][1] = try k.mod.function("k6_piflow_substep_bf16_f32");
        k.substep_fn[1][0] = try k.mod.function("k6_piflow_substep_f32_bf16");
        k.substep_fn[1][1] = try k.mod.function("k6_piflow_substep_f32_f32");
        k.video_input_fn[0] = try k.mod.function("k6_piflow_video_input_bf16");
        k.video_input_fn[1] = try k.mod.function("k6_piflow_video_input_f32");
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    fn grid(n: u64) Error!u32 {
        const b = (n + threads - 1) / threads;
        if (n == 0 or b > std.math.maxInt(i32)) return error.ViewMismatch;
        return @intCast(b);
    }

    /// One substep of one modality: out = x + pi(x) * dsigma, with x = `m.src` at substep 0 and `m.out` after it.
    pub fn substep(k: *const Kernels, s: cuda.Stream, c: Config, st: Step, sub: Substep, m: Modality) Error!void {
        const n = m.src.n;
        if (m.pred.n != n * c.n_grid or m.out == 0 or m.src.ptr == 0 or m.pred.ptr == 0) return error.ViewMismatch;
        if (m.out == m.src.ptr) return error.ViewMismatch; // the step input is read by every substep
        if (sub.g0 + 1 >= c.n_grid) return error.ViewMismatch;
        var a: cuda.launch.Args = .{};
        a.add(m.src.ptr);
        a.add(m.pred.ptr);
        a.add(m.out);
        a.add(@as(i64, @intCast(n)));
        a.add(@as(i32, @intFromBool(sub.index == 0)));
        a.add(@as(i32, @intCast(sub.g0)));
        inline for (.{ st.sigma, sub.w0, sub.w1, sub.sigma_clamped, sub.dsigma }) |v| a.add(@as(f32, v));
        const f = k.substep_fn[@intCast(@backingInt(m.src.dtype))][@intCast(@backingInt(m.pred.dtype))];
        try cuda.launch.launch(f, .{ .grid = .{ .x = try grid(n) }, .block = .{ .x = threads } }, s, &a);
    }

    /// A step's policy rollout for one modality, every substep (`Rollout`). `cancel` is read between launches; a
    /// cancelled rollout leaves `m.out` partial (the job is discarded).
    pub fn rollout(k: *const Kernels, s: cuda.Stream, c: Config, st: Step, m: Modality, cancel: ?*const std.atomic.Value(bool)) Error!void {
        var r: Rollout = .init(c, st);
        while (try r.next()) |sub| {
            if (cancel) |x| if (x.load(.acquire)) return error.Cancelled;
            try k.substep(s, c, st, sub, m);
        }
    }

    /// One step's update after its DiT call: the video rollout, then the audio rollout (piflow_sampler.py:157-173).
    /// Both are required.
    pub fn advance(k: *const Kernels, s: cuda.Stream, c: Config, st: Step, video: Modality, audio: Modality, cancel: ?*const std.atomic.Value(bool)) Error!void {
        try k.rollout(s, c, st, video, cancel);
        try k.rollout(s, c, st, audio, cancel);
    }

    /// `_build_video_input` (denoise_loop.py:13-64), visual_cond_scheme "pretrain", B = 1, the DiT's `x_video`:
    /// out [rows, 2 * ch + 1] in the state's dtype = cat(state [rows, ch], cond, mask). T2V (`first == null`): cond
    /// and mask zero. I2V: the reference latent `first` (rows_first * ch elements: frame 0, rows_first = H * W) cast to
    /// the state's dtype (`first_frames.to(video.dtype)`, round to nearest even into bf16) in cond's first rows_first
    /// rows, mask 1 there. The `i2v` and `tail_cond_first_frame` schemes are not supported.
    pub fn videoInput(k: *const Kernels, s: cuda.Stream, state: Tensor, rows: u64, ch: u32, first: ?Tensor, rows_first: u64, out: u64) Error!void {
        if (state.n != rows * ch or state.ptr == 0 or out == 0 or ch == 0) return error.ViewMismatch;
        if (first) |f| if (f.n != rows_first * ch or rows_first > rows or f.ptr == 0) return error.ViewMismatch;
        var a: cuda.launch.Args = .{};
        a.add(state.ptr);
        a.add(if (first) |f| f.ptr else @as(u64, 0));
        a.add(@as(i32, if (first) |f| @backingInt(f.dtype) else -1));
        a.add(out);
        a.add(@as(i64, @intCast(rows)));
        a.add(@as(i32, @intCast(ch)));
        a.add(@as(i64, @intCast(if (first != null) rows_first else 0)));
        const n = rows * (2 * @as(u64, ch) + 1);
        const f = k.video_input_fn[@intCast(@backingInt(state.dtype))];
        try cuda.launch.launch(f, .{ .grid = .{ .x = try grid(n) }, .block = .{ .x = threads } }, s, &a);
    }
};

/// The sampler's own device memory for one job, every byte, computed from the shape (not measured): two f32 states a
/// modality (the rollout writes one while the step reads the other; the step-0 noise is the pipeline's input) and the
/// DiT's visual input [rows, 2 * ch + 1] in f32 (its bf16 step-0 form is smaller). The DiT's grid outputs, its scratch
/// and the weights are the pipeline's and the DiT's (`Plan.bytes`), listed in the README's memory table.
pub const Memory = struct {
    video_states: u64,
    audio_states: u64,
    video_input: u64,

    pub fn of(video_elems: u64, video_rows: u64, video_ch: u32, audio_elems: u64) Memory {
        return .{
            .video_states = 2 * video_elems * 4,
            .audio_states = 2 * audio_elems * 4,
            .video_input = video_rows * (2 * @as(u64, video_ch) + 1) * 4,
        };
    }

    pub fn total(m: Memory) u64 {
        return m.video_states + m.audio_states + m.video_input;
    }
};

/// Checks the schedule against a table of upstream's own f32 bits, written by `stk_twin.kandinsky6_sampler.capture
/// schedule` from upstream's tensors on the GPU (gate S0 of kernels/cuda/kandinsky6/sampler/README.md):
/// {"config": {...}, "steps": [{"tau_src": bits, ..., "substeps": [{"sigma": bits, "t", "raw_next", "sigma_next"}]}]},
/// bits as u32 integers: only tensors upstream exposes (loop and rollout arguments, policy attributes, pi and
/// _interpolate arguments); the interpolation weights, divisor and Euler factor are covered by the device gate (S1).
/// Not yet run: the table is produced under a GPU grant, then embedded by a test.
pub fn expectTable(gpa: std.mem.Allocator, json: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const cfg = root.get("config").?.object;
    const c: Config = .{
        .num_steps = @intCast(cfg.get("num_steps").?.integer),
        .shift = cfg.get("shift").?.float,
        .eps = cfg.get("eps").?.float,
        .final_step_size_scale = cfg.get("final_step_size_scale").?.float,
        .num_policy_substeps = @intCast(cfg.get("num_policy_substeps").?.integer),
        .n_grid = @intCast(cfg.get("n_grid").?.integer),
    };
    const steps = root.get("steps").?.array.items;
    var sch = try Schedule.init(c);
    const bits = struct {
        fn f(o: std.json.ObjectMap, key: []const u8) !u32 {
            const v = o.get(key) orelse return error.TableField;
            return @intCast(v.integer);
        }
    }.f;
    var i: usize = 0;
    while (try sch.next()) |st| : (i += 1) {
        if (i >= steps.len) return error.TableSteps;
        const w = steps[i].object;
        inline for (.{ "tau_src", "tau_dst", "sigma", "time", "policy_raw_src", "policy_raw_dst", "policy_segment" }) |key|
            try std.testing.expectEqual(try bits(w, key), @as(u32, @bitCast(@field(st, key))));
        const subs = w.get("substeps").?.array.items;
        try std.testing.expectEqual(subs.len, st.substeps);
        var r: Rollout = .init(c, st);
        var j: usize = 0;
        while (try r.next()) |sub| : (j += 1) {
            const ws = subs[j].object;
            inline for (.{ "sigma", "t", "raw_next", "sigma_next" }) |key|
                try std.testing.expectEqual(try bits(ws, key), @as(u32, @bitCast(@field(sub, key))));
        }
    }
    try std.testing.expectEqual(steps.len, i);
}

const testing = std.testing;

test "config bounds: upstream's refusals" {
    try Config.distill.validate();
    var c = Config.distill;
    c.num_steps = 0;
    try testing.expectError(error.InvalidConfig, c.validate());
    c = Config.distill;
    c.num_policy_substeps = 0;
    try testing.expectError(error.InvalidConfig, c.validate());
    c = Config.distill;
    c.n_grid = 1;
    try testing.expectError(error.InvalidConfig, c.validate());
    c = Config.distill;
    c.final_step_size_scale = 1.5;
    try testing.expectError(error.InvalidConfig, c.validate());
    c = Config.distill;
    c.eps = 0;
    try testing.expectError(error.InvalidConfig, c.validate());
    c = Config.distill;
    c.shift = std.math.nan(f64);
    try testing.expectError(error.InvalidConfig, c.validate());
}

test "torch.round is half to even" {
    try testing.expectEqual(@as(f32, 2), rint(2.5));
    try testing.expectEqual(@as(f32, 4), rint(3.5));
    try testing.expectEqual(@as(f32, 13), rint(12.6));
    try testing.expectEqual(@as(f32, 0), rint(0.5));
    try testing.expectEqual(@as(f32, 1), rint(1.0));
}

test "distill schedule: ten steps, half final segment, tau held at eps or above, substeps within bounds" {
    const c = Config.distill;
    var sch = try Schedule.init(c);
    var count: u32 = 0;
    var prev_tau: f32 = 1.0;
    var total: u64 = 0;
    while (try sch.next()) |st| {
        try testing.expectEqual(prev_tau, st.tau_src); // raw_t_src = tau_dst
        try testing.expect(st.tau_dst >= @as(f32, @floatCast(c.eps)));
        try testing.expect(st.tau_dst < st.tau_src);
        try testing.expect(st.substeps >= 1 and st.substeps <= c.num_policy_substeps);
        try testing.expectEqual(st.index == c.num_steps - 1, st.final);
        try testing.expectEqual(st.sigma * 1000.0, st.time);
        // 1 / (10 - 0.5) as a Python float, the final step's half of it, each then one f32 rounding
        const base: f64 = 1.0 / 9.5;
        const want: f64 = if (st.final) base * 0.5 else base;
        try testing.expectEqual(@as(f32, @floatCast(want)), st.segment);
        var r: Rollout = .init(c, st);
        var k: u32 = 0;
        var last: Substep = undefined;
        while (try r.next()) |sub| : (k += 1) {
            try testing.expectEqual(k, sub.index);
            try testing.expect(sub.g0 + 1 < c.n_grid);
            try testing.expect(sub.dsigma <= 0); // sigma never grows
            last = sub;
        }
        try testing.expectEqual(st.substeps, k);
        try testing.expect(last.raw_next >= 0);
        total += k;
        prev_tau = st.tau_dst;
        count += 1;
    }
    try testing.expectEqual(c.num_steps, count);
    try testing.expect(total >= c.num_steps);
}

test "a step's first substep queries the policy at its own source time" {
    var sch = try Schedule.init(Config.distill);
    const st = (try sch.next()).?;
    var r: Rollout = .init(Config.distill, st);
    const s0 = (try r.next()).?;
    try testing.expectEqual(st.sigma, s0.sigma);
    // unwarp(sigma) is policy_raw_src, so t = (raw_src - raw_dst) / segment
    try testing.expectEqual((st.policy_raw_src - st.policy_raw_dst) / st.policy_segment, s0.t);
}

test "memory: every sampler buffer at the 864x480 5 s preset" {
    const m = Memory.of(31 * 60 * 108 * 16, 31 * 60 * 108, 16, 218 * 40);
    try testing.expectEqual(@as(u64, 2 * 3_214_080 * 4), m.video_states);
    try testing.expectEqual(@as(u64, 2 * 8_720 * 4), m.audio_states);
    try testing.expectEqual(@as(u64, 200_880 * 33 * 4), m.video_input);
}
