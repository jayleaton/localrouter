//! Rounding a requested width and height onto a model's grid. `size_snap` asks for this; without it a size off the
//! grid is refused, as before. Each side goes to the nearest multiple of `multiple` inside the model's min and max.
//! A tie (720 and a multiple of 32: 704 and 736 are equally near) goes to the lower multiple, which is what the
//! MiniMax H3 note in docs/AGENTS.md tells a caller to send for 1280x720. If the result is over `max_pixels`, the
//! longer side shrinks by one multiple at a time until it fits.

const std = @import("std");

pub const Limits = struct {
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
    multiple: u32,
    max_pixels: ?u64 = null,
};

pub fn alignDown(n: u32, m: u32) u32 {
    if (m <= 1) return n;
    return n / m * m;
}

pub fn alignUp(n: u32, m: u32) u32 {
    if (m <= 1) return n;
    const mm: u64 = m;
    const up = (@as(u64, n) + mm - 1) / mm * mm;
    if (up > std.math.maxInt(u32)) return alignDown(std.math.maxInt(u32), m);
    return @intCast(up);
}

/// The nearest valid size. `0 x 0` is left alone: the model picks (an edit without a size, a video without one,
/// before `engine.fill`).
pub fn snap(w: u32, h: u32, lim: Limits) [2]u32 {
    if (w == 0 and h == 0) return .{ 0, 0 };
    var x = axis(w, lim.multiple, lim.min_width, lim.max_width);
    var y = axis(h, lim.multiple, lim.min_height, lim.max_height);
    if (lim.max_pixels) |px| {
        var guard: u32 = 0;
        while (@as(u64, x) * y > px and guard < 8192) : (guard += 1) {
            if (x >= y and x >= lim.min_width +| lim.multiple) {
                x -= lim.multiple;
            } else if (y >= lim.min_height +| lim.multiple) {
                y -= lim.multiple;
            } else break;
        }
    }
    return .{ x, y };
}

fn axis(n: u32, multiple: u32, min: u32, max: u32) u32 {
    const m: u32 = if (multiple == 0) 1 else multiple;
    var s: u32 = if (n % m == 0) n else blk: {
        const down = n / m * m;
        const up = down +| m;
        const down_d = n - down;
        const up_d = if (up >= n) up - n else 0;
        break :blk if (down_d < up_d) down else if (up_d < down_d) up else down;
    };
    if (s < min) s = alignUp(min, m);
    if (s > max) s = alignDown(max, m);
    if (s < min) s = min;
    if (s > max) s = max;
    return s;
}

// ---------------------------------------------------------------- tests

const t = std.testing;

test "1280x720 on a 32 grid ties downward to 1280x704, and stays inside the model's max" {
    const open: Limits = .{ .min_width = 256, .max_width = 2048, .min_height = 256, .max_height = 2048, .multiple = 32 };
    try t.expectEqual([2]u32{ 1280, 704 }, snap(1280, 720, open));
    const h3: Limits = .{ .min_width = 256, .max_width = 1280, .min_height = 256, .max_height = 704, .multiple = 32 };
    try t.expectEqual([2]u32{ 1280, 704 }, snap(1280, 720, h3));
    const small: Limits = .{ .min_width = 256, .max_width = 768, .min_height = 256, .max_height = 448, .multiple = 32 };
    try t.expectEqual([2]u32{ 768, 448 }, snap(1280, 720, small));
}

test "a 16 grid rounds halfway down, clamps to the side limits, and shrinks to max_pixels" {
    const grid: Limits = .{ .min_width = 256, .max_width = 2048, .min_height = 256, .max_height = 2048, .multiple = 16 };
    try t.expectEqual([2]u32{ 1024, 1024 }, snap(1024, 1024, grid));
    try t.expectEqual([2]u32{ 992, 992 }, snap(1000, 1000, grid)); // 8 away from 992 and from 1008
    try t.expectEqual([2]u32{ 256, 256 }, snap(100, 200, grid));
    const capped: Limits = .{ .min_width = 256, .max_width = 1664, .min_height = 256, .max_height = 1664, .multiple = 16, .max_pixels = 1048576 };
    try t.expectEqual([2]u32{ 1024, 1024 }, snap(1360, 1024, capped));
    try t.expectEqual([2]u32{ 0, 0 }, snap(0, 0, grid));
}

test "one exact size is the only place a request can land" {
    const only: Limits = .{ .min_width = 832, .max_width = 832, .min_height = 480, .max_height = 480, .multiple = 32 };
    try t.expectEqual([2]u32{ 832, 480 }, snap(1280, 720, only));
    try t.expectEqual([2]u32{ 832, 480 }, snap(832, 480, only));
}
