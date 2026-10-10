//! The sampler's schedule as the pipeline sets it (diffusers' FlowMatchEulerDiscreteScheduler.set_timesteps with
//! sigmas = the pipeline's `sample_sigmas` when the checkpoint has them, else linspace(1, 1 / steps, steps); then the
//! resolution's mu with dynamic shifting, or the scheduler's fixed `shift` without): numpy's float32 arithmetic under
//! NEP 50 (Python floats meet float32 arrays as float32), so every value is the scheduler's f32 bit for bit. Then a
//! final 0. Qwen-Image 2.1 uses the dynamic shift and 25 linspace steps; Qwen-Image 2.1 Turbo its 8 stored sigmas with
//! shift 1 (the identity).

const std = @import("std");

/// The scheduler's config, from the text encoder's pack manifest (`scheduler`).
pub const Config = struct {
    base_image_seq_len: f64 = 256,
    max_image_seq_len: f64 = 4096,
    base_shift: f64 = 0.5,
    max_shift: f64 = 1.15,
    shift_terminal: ?f64 = null,
    shift: f64 = 1.0, // without dynamic shifting
    use_dynamic_shifting: bool = true,
    time_shift_type: []const u8 = "exponential",
    use_karras_sigmas: bool = false,
    use_exponential_sigmas: bool = false,
    use_beta_sigmas: bool = false,
    invert_sigmas: bool = false,
    sample_sigmas: ?[]const f64 = null, // the pipeline's (model_index.json), without the final 0: their count is the step count
};

/// The step count the schedule fixes (the checkpoint's `sample_sigmas`), else null: any count.
pub fn fixedSteps(c: Config) ?u32 {
    const s = c.sample_sigmas orelse return null;
    return @intCast(s.len);
}

/// The pipeline's `calculate_shift` (Python floats).
pub fn mu(c: Config, image_seq_len: f64) f64 {
    const m = (c.max_shift - c.base_shift) / (c.max_image_seq_len - c.base_image_seq_len);
    const b = c.base_shift - m * c.base_image_seq_len;
    return image_seq_len * m + b;
}

/// numpy.linspace(start, stop, n) in f64: i * step + start, the last value exactly stop.
fn linspace(start: f64, stop: f64, out: []f64) void {
    const div: f64 = @floatFromInt(out.len - 1);
    const step = (stop - start) / div;
    for (out, 0..) |*o, i| o.* = @as(f64, @floatFromInt(i)) * step + start;
    out[out.len - 1] = stop;
}

/// Sigmas for a height x width image and `steps` steps: `steps + 1` values into `out`, the last 0.
pub fn sigmas(c: Config, height: u32, width: u32, out: []f32) !void {
    if ((c.use_dynamic_shifting and !std.mem.eql(u8, c.time_shift_type, "exponential")) or c.use_karras_sigmas or
        c.use_exponential_sigmas or c.use_beta_sigmas or c.invert_sigmas) return error.UnsupportedSchedule;
    const steps = out.len - 1;
    if (steps < 1 or steps > 1000) return error.BadSteps;
    var base: [1000]f64 = undefined;
    if (c.sample_sigmas) |s| {
        if (s.len != steps) return error.BadSteps;
        @memcpy(base[0..steps], s);
    } else if (steps == 1) {
        base[0] = 1.0; // linspace(1, 1, 1)
    } else linspace(1.0, 1.0 / @as(f64, @floatFromInt(steps)), base[0..steps]);
    if (c.use_dynamic_shifting) {
        const seq: f64 = @floatFromInt((height / 16) * (width / 16));
        const em: f32 = @floatCast(@exp(mu(c, seq))); // math.exp(mu), a Python float meeting a float32 array
        for (out[0..steps], base[0..steps]) |*o, b| {
            const t: f32 = @floatCast(b); // np.array(sigmas).astype(np.float32)
            o.* = em / (em + (1.0 / t - 1.0)); // exp(mu) / (exp(mu) + (1 / t - 1) ** 1.0), all float32
        }
    } else {
        const sh: f32 = @floatCast(c.shift);
        const sm1: f32 = @floatCast(c.shift - 1.0); // (self.shift - 1): Python floats, then float32
        for (out[0..steps], base[0..steps]) |*o, b| {
            const t: f32 = @floatCast(b);
            o.* = sh * t / (1.0 + sm1 * t); // shift * sigmas / (1 + (shift - 1) * sigmas), all float32
        }
    }
    if (c.shift_terminal) |st| {
        const scale: f32 = (1.0 - out[steps - 1]) / @as(f32, @floatCast(1.0 - st)); // np.float32 / Python float
        for (out[0..steps]) |*o| o.* = 1.0 - ((1.0 - o.*) / scale);
    }
    out[steps] = 0;
}

test "the last sigma stretches to the terminal value, the first stays 1" {
    var s: [5]f32 = undefined;
    try sigmas(.{ .shift_terminal = 0.02 }, 512, 512, &s);
    try std.testing.expectEqual(@as(f32, 1.0), s[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), s[3], 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), s[4]);
    try std.testing.expect(s[1] < s[0] and s[2] < s[1]);
}

/// Qwen-Image 2.1 Turbo's `sample_sigmas` (model_index.json) and scheduler settings.
const turbo_sigmas = [_]f64{ 1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568 };

test "a checkpoint's own sigmas without dynamic shifting: numpy's float32 bits, the step count fixed" {
    const turbo: Config = .{ .use_dynamic_shifting = false, .shift = 1.0, .max_shift = 0.9, .max_image_seq_len = 8192, .sample_sigmas = &turbo_sigmas };
    try std.testing.expectEqual(@as(?u32, 8), fixedSteps(turbo));
    try std.testing.expectEqual(@as(?u32, null), fixedSteps(.{}));
    // numpy: shift * s / (1 + (shift - 1) * s) over np.array(sample_sigmas).astype(np.float32); the size does not matter
    const want1 = [_]u32{ 0x3f800000, 0x3f7a7be5, 0x3f744524, 0x3f6d375d, 0x3f6523f6, 0x3f585b9f, 0x3f345c57, 0x3ed44242, 0 };
    const want3 = [_]u32{ 0x3f800000, 0x3f7e2271, 0x3f7bf782, 0x3f796ab0, 0x3f765f9c, 0x3f7143c8, 0x3f609a2c, 0x3f2e1099, 0 };
    for ([_][2]u32{ .{ 1024, 1024 }, .{ 512, 1664 } }) |hw| {
        var s: [9]f32 = undefined;
        try sigmas(turbo, hw[0], hw[1], &s);
        try std.testing.expectEqualSlices(u32, &want1, @ptrCast(&s));
        var shifted = turbo;
        shifted.shift = 3.0;
        try sigmas(shifted, hw[0], hw[1], &s);
        try std.testing.expectEqualSlices(u32, &want3, @ptrCast(&s));
    }
    var wrong: [26]f32 = undefined; // the stored schedule has 8 steps, not 25
    try std.testing.expectError(error.BadSteps, sigmas(turbo, 1024, 1024, &wrong));
}
