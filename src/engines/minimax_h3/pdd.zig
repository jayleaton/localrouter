//! MiniMax H3-Image's PDD schedule and head plan as the twin computes them (`stk_twin/h3i/sampler.py`, ComfyUI e9afbccc;
//! docs/dev/H3-IMAGE.md section 1.4): the sampling table over the PDD grid, ComfyUI's `simple` scheduler on it, and per
//! step the timestep the model gets, its float64 sigma and t, and the heads of `final_layer.video_out` with their fp32 dt
//! weights. Zig does not contract floats, so every f32 expression rounds where numpy's float32 does; the table's lerp is
//! the fused multiply-add torch's CPU kernel uses (`@mulAdd`). The host test holds it to the twin's JSON bit for bit.

const std = @import("std");

/// The bf16 checkpoint's `pdd_sigmas` (its safetensors config), decreasing from 1 to 0: 6 PDD heads.
pub const pdd_sigmas = [_]f64{ 1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0 };
pub const max_heads = pdd_sigmas.len - 1;

/// MiniMaxH3ImageModelSampling.sigmas: entry i - 1 is sigma(t), t = (i / 1000) * 1000, i = 1 .. 1000.
pub fn table(grid: []const f64, out: *[1000]f32) void {
    const n = grid.len - 1;
    const nf: f32 = @floatFromInt(n);
    for (out, 1..) |*o, i| {
        const fi: f32 = @floatFromInt(i);
        const ts = (fi / 1000.0) * 1000.0;
        const pos = std.math.clamp(ts / 1000.0, 0.0, 1.0) * nf;
        const k: usize = @min(@as(usize, @intFromFloat(@floor(pos))), n - 1);
        // the ascending grid: index 0 is sigma 0
        const a: f32 = @floatCast(grid[n - k]);
        const b: f32 = @floatCast(grid[n - k - 1]);
        o.* = lerp(a, b, pos - @as(f32, @floatFromInt(k)));
    }
}

/// ATen's vectorized CPU lerp (fp32): |w| < 0.5 ? fma(w, b - a, a) : fma(-(b - a), 1 - w, b).
fn lerp(a: f32, b: f32, w: f32) f32 {
    const d = b - a;
    return if (@abs(w) < 0.5) @mulAdd(f32, w, d, a) else @mulAdd(f32, -d, 1.0 - w, b);
}

/// ComfyUI's `simple` scheduler: steps + 1 sigmas into `out` (the last 0).
pub fn sigmas(tab: *const [1000]f32, out: []f32) void {
    const steps = out.len - 1;
    const ss: f64 = 1000.0 / @as(f64, @floatFromInt(steps));
    for (0..steps) |x| out[x] = tab[1000 - 1 - @as(usize, @intFromFloat(@as(f64, @floatFromInt(x)) * ss))];
    out[steps] = 0;
}

pub const Step = struct {
    /// The model's timestep: fp32 sigma * 1000.
    timestep: f32,
    /// float(timestep) / 1000 and t = 1 - sigma, in float64 as ComfyUI's model computes them.
    sigma: f64,
    t: f64,
    /// Grid interval [start, stop) the step spans; the head is row block 0 plus w[j] * block heads[j], j < nheads.
    start: u8,
    stop: u8,
    nheads: u8,
    heads: [max_heads]u8 = @splat(0),
    w: [max_heads]f32 = @splat(0),
};

/// Step i of schedule `sig` (fp32 values, the last 0) over `grid`.
pub fn step(grid: []const f64, sig: []const f32, i: usize) Step {
    const n = grid.len - 1;
    const timestep = sig[i] * 1000.0;
    const sigma = @as(f64, timestep) / 1000.0;
    // the step whose sample sigma is nearest (fp32: the float64 sigma meets the fp32 tensor as fp32), then the next one
    const s32: f32 = @floatCast(sigma);
    var j: usize = 0;
    for (sig, 0..) |s, k| {
        if (@abs(s - s32) < @abs(sig[j] - s32)) j = k;
    }
    const next: f64 = sig[@min(j + 1, sig.len - 1)];
    var start = nearest(grid, sigma);
    var stop = nearest(grid, next);
    start = @min(start, n - 1);
    stop = @max(stop, start + 1);
    var tot: f64 = 0;
    for (start..stop) |k| tot += grid[k] - grid[k + 1];
    var st: Step = .{ .timestep = timestep, .sigma = sigma, .t = 1.0 - sigma, .start = @intCast(start), .stop = @intCast(stop), .nheads = 0 };
    const first = @max(start, 1);
    for (first..stop) |k| {
        const dt: f32 = @floatCast(grid[k] - grid[k + 1]);
        st.heads[st.nheads] = @intCast(k);
        st.w[st.nheads] = dt / @as(f32, @floatCast(tot));
        st.nheads += 1;
    }
    return st;
}

/// The grid point nearest `s` in float64 (the first on a tie).
fn nearest(grid: []const f64, s: f64) usize {
    var best: usize = 0;
    for (grid, 0..) |g, k| {
        if (@abs(g - s) < @abs(grid[best] - s)) best = k;
    }
    return best;
}

test "PDD plan: the twin's table, schedules and head plans, bit for bit (pdd_plan.json from stk_twin.h3i.sampler)" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("pdd_plan.json"), .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    for (root.get("pdd_sigmas").?.array.items, pdd_sigmas) |v, g| try std.testing.expectEqual(g, num(v));
    var tab: [1000]f32 = undefined;
    table(&pdd_sigmas, &tab);
    for (root.get("table").?.array.items, tab) |v, t| try std.testing.expectEqual(@as(f32, @floatCast(num(v))), t);
    var buf: [64]f32 = undefined;
    var plans: usize = 0;
    for (root.get("plans").?.array.items) |pv| {
        const p = pv.object;
        const steps: usize = @intCast(p.get("steps").?.integer);
        const sig = buf[0 .. steps + 1];
        sigmas(&tab, sig);
        for (p.get("sigmas").?.array.items, sig) |v, s| try std.testing.expectEqual(@as(f32, @floatCast(num(v))), s);
        for (p.get("plan").?.array.items, 0..) |sv, i| {
            const want = sv.object;
            const st = step(&pdd_sigmas, sig, i);
            try std.testing.expectEqual(@as(f32, @floatCast(num(want.get("timestep").?))), st.timestep);
            try std.testing.expectEqual(num(want.get("sigma").?), st.sigma);
            try std.testing.expectEqual(num(want.get("t").?), st.t);
            try std.testing.expectEqual(want.get("start").?.integer, st.start);
            try std.testing.expectEqual(want.get("stop").?.integer, st.stop);
            const heads = want.get("heads").?.array.items;
            try std.testing.expectEqual(heads.len, st.nheads);
            for (heads, want.get("w").?.array.items, 0..) |h, w, k| {
                try std.testing.expectEqual(h.integer, st.heads[k]);
                try std.testing.expectEqual(@as(f32, @floatCast(num(w))), st.w[k]);
            }
        }
        plans += 1;
    }
    try std.testing.expect(plans >= 10);
}

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => unreachable,
    };
}
