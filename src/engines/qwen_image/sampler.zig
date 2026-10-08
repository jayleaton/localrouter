//! The sampler's schedule as the pipeline sets it (diffusers' FlowMatchEulerDiscreteScheduler.set_timesteps with
//! sigmas = linspace(1, 1 / steps, steps) and the resolution's mu): numpy's float32 arithmetic under NEP 50 (Python
//! floats meet float32 arrays as float32), so every value is the scheduler's f32 bit for bit. Then a final 0.

const std = @import("std");

/// The scheduler's config, from the text encoder's pack manifest (`scheduler`).
pub const Config = struct {
    base_image_seq_len: f64 = 256,
    max_image_seq_len: f64 = 4096,
    base_shift: f64 = 0.5,
    max_shift: f64 = 1.15,
    shift_terminal: ?f64 = null,
    use_dynamic_shifting: bool = true,
    time_shift_type: []const u8 = "exponential",
    use_karras_sigmas: bool = false,
    use_exponential_sigmas: bool = false,
    use_beta_sigmas: bool = false,
    invert_sigmas: bool = false,
    sample_sigmas: ?[]const f64 = null,
};

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
    if (!c.use_dynamic_shifting or !std.mem.eql(u8, c.time_shift_type, "exponential") or c.use_karras_sigmas or
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
    const seq: f64 = @floatFromInt((height / 16) * (width / 16));
    const em: f32 = @floatCast(@exp(mu(c, seq))); // math.exp(mu), a Python float meeting a float32 array
    for (out[0..steps], base[0..steps]) |*o, b| {
        const t: f32 = @floatCast(b); // np.array(sigmas).astype(np.float32)
        o.* = em / (em + (1.0 / t - 1.0)); // exp(mu) / (exp(mu) + (1 / t - 1) ** 1.0), all float32
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
