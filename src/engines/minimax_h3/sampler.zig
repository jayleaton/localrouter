//! MiniMax H3's schedule and sampler scalars as the twin computes them (`stk_twin/h3/sampler.py`): ComfyUI's
//! `simple` scheduler on ModelSamplingAV (shift 12), and per step of `res_multistep` (eta 0) its kind and fp32
//! scalars, with -log / expm1 / exp from the portable fdlibm ports (`smath`). Zig does not contract floats, so each
//! f32 expression rounds where numpy's float32 does.

const std = @import("std");
const smath = @import("qwen_image").smath;

/// steps + 1 sigmas into `out` (the last 0).
pub fn sigmas(shift: f32, out: []f32) void {
    const steps = out.len - 1;
    const ss: f64 = 1000.0 / @as(f64, @floatFromInt(steps));
    for (0..steps) |x| {
        const i: usize = 1000 - 1 - @as(usize, @intFromFloat(@as(f64, @floatFromInt(x)) * ss)); // table[-(1 + int(x * ss))]
        const fi: f32 = @floatFromInt(i + 1);
        const t = ((fi / 1000.0) * 1000.0) / 1000.0;
        out[x] = (shift * t) / (1.0 + (shift - 1.0) * t);
    }
    out[steps] = 0;
}

pub const Step = union(enum) {
    euler: struct { sigma: f32, dt: f32 },
    res2: struct { sigma: f32, e: f32, h: f32, b1: f32, b2: f32 },
};

fn tf(s: f32) f32 {
    return -@as(f32, @floatCast(smath.log(s)));
}

/// The step plan for `sig` (steps + 1 values) into `out` (steps entries).
pub fn plan(sig: []const f32, out: []Step) void {
    for (out, 0..) |*st, i| {
        const down = sig[i + 1];
        if (down == 0 or i == 0) {
            st.* = .{ .euler = .{ .sigma = sig[i], .dt = down - sig[i] } };
            continue;
        }
        const t = tf(sig[i]);
        const h = tf(down) - t;
        const c2 = (tf(sig[i - 1]) - t) / h;
        const mt = -h;
        const phi1 = @as(f32, @floatCast(smath.expm1(mt))) / mt;
        const phi2 = (phi1 - 1.0) / mt;
        st.* = .{ .res2 = .{ .sigma = sig[i], .e = @floatCast(smath.exp(-h)), .h = h, .b1 = phi1 - phi2 / c2, .b2 = phi2 / c2 } };
    }
}

test "sigmas and step plans equal the twin's (sampler.py) bit for bit" {
    const T = struct { sigmas: []const u32, plan: []const std.json.Value };
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, @embedFile("sampler_table.json"), .{});
    defer parsed.deinit();
    const want = parsed.value.object;
    inline for (.{ 8, 20 }) |n| {
        var sig: [n + 1]f32 = undefined;
        sigmas(12.0, &sig);
        const w = want.get(std.fmt.comptimePrint("{d}", .{n})).?.object;
        for (sig, w.get("sigmas").?.array.items) |g, e| try std.testing.expectEqual(@as(u32, @intCast(e.integer)), @as(u32, @bitCast(g)));
        var st: [n]Step = undefined;
        plan(&sig, &st);
        for (st, w.get("plan").?.array.items) |g, e| {
            const o = e.object;
            const bits = struct {
                fn f(obj: std.json.ObjectMap, k: []const u8) u32 {
                    return @intCast(obj.get(k).?.integer);
                }
            }.f;
            switch (g) {
                .euler => |x| {
                    try std.testing.expectEqualStrings("euler", o.get("kind").?.string);
                    try std.testing.expectEqual(bits(o, "sigma"), @as(u32, @bitCast(x.sigma)));
                    try std.testing.expectEqual(bits(o, "dt"), @as(u32, @bitCast(x.dt)));
                },
                .res2 => |x| {
                    try std.testing.expectEqualStrings("res2", o.get("kind").?.string);
                    inline for (.{ "sigma", "e", "h", "b1", "b2" }) |k| try std.testing.expectEqual(bits(o, k), @as(u32, @bitCast(@field(x, k))));
                },
            }
        }
    }
    _ = T;
}
