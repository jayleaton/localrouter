//! Qwen3-VL 32B vision tower, identical launch ABI and probe names to the twin.
const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
const smath = qi.smath;
const tw = @import("te_vision_weights.zig");
const te = @import("te32.zig");
const launch = @import("launch.zig");
const dit = @import("dit.zig");
const D: u32 = 1152;
const HD: u32 = 72;
const HEADS: u32 = 16;
fn blocks(n: u64) u32 {
    return @intCast((n + 255) / 256);
}
pub const Features = struct {
    merged: u64,
    deepstack: [3]u64,
    grid_h: u32,
    grid_w: u32,
    pub fn rows(f: Features) u32 {
        return (f.grid_h / 2) * (f.grid_w / 2);
    }
};
pub const Ops = struct {
    module: cuda.Module,
    norm_fn: cuda.Function,
    gelu_fn: cuda.Function,
    rope_fn: cuda.Function,
    position_fn: cuda.Function,
    attn_fn: cuda.Function,
    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .norm_fn = try m.function("h3v_norm"), .gelu_fn = try m.function("h3v_gelu"), .rope_fn = try m.function("h3v_rope"), .position_fn = try m.function("h3v_position"), .attn_fn = try m.function("h3v_attention") };
    }
    pub fn unload(o: *Ops) void {
        o.module.unload();
    }
};
/// Element i of torch.linspace(0, 47, n) in f32, as torch computes it (the learned position table's 48 x 48 grid): the
/// first half start + i * step, the second half end - (n - 1 - i) * step, with step = 47 / (n - 1) in f32.
fn linspace47(i: u32, n: u32) f32 {
    if (n == 1) return 0;
    const step: f32 = 47.0 / @as(f32, @floatFromInt(n - 1));
    if (i < n / 2) return @as(f32, @floatFromInt(i)) * step;
    return 47.0 - @as(f32, @floatFromInt(n - 1 - i)) * step;
}

test "linspace47 follows torch's two-sided float32 linspace" {
    try std.testing.expectEqual(@as(f32, 0), linspace47(0, 28));
    try std.testing.expectEqual(@as(f32, 47), linspace47(27, 28));
    // the second half counts back from the end
    const step: f32 = 47.0 / 51.0;
    try std.testing.expectEqual(@as(f32, 47.0) - @as(f32, 13) * step, linspace47(38, 52));
}

pub const VisionEncoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const Ops,
    h3: *const launch.Ops,
    tk: *const te.Ops,
    s: cuda.Stream,
    w: tw.Weights,
    max_patches: u32,
    probe: ?dit.Probe = null,
    pixels: u64 = 0,
    h: u64 = 0,
    x: u64 = 0,
    z: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    kk: u64 = 0,
    v: u64 = 0,
    attn: u64 = 0,
    fc: u64 = 0,
    gel: u64 = 0,
    cs: u64 = 0,
    sn: u64 = 0,
    ids: u64 = 0,
    weights: u64 = 0,
    out: [4]u64 = .{ 0, 0, 0, 0 },
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const Ops, h3: *const launch.Ops, tk: *const te.Ops, s: cuda.Stream, p: *const qi.pack.Pack, up: *qi.upload.Uploader, max_patches: u32) !VisionEncoder {
        var v: VisionEncoder = .{ .gpa = gpa, .d = d, .k = k, .h3 = h3, .tk = tk, .s = s, .w = try tw.Weights.load(gpa, io, d, p, up), .max_patches = max_patches };
        errdefer v.deinit();
        const n: u64 = max_patches;
        inline for (.{ .{ "pixels", n * 1536 * 4 }, .{ "h", n * D * 4 }, .{ "x", n * D * 4 }, .{ "z", n * D * 4 }, .{ "qkv", n * D * 3 * 4 }, .{ "q", n * D * 4 }, .{ "kk", n * D * 4 }, .{ "v", n * D * 4 }, .{ "attn", n * D * 4 }, .{ "fc", n * 4304 * 4 }, .{ "gel", n * 4304 * 4 }, .{ "cs", n * 36 * 4 }, .{ "sn", n * 36 * 4 }, .{ "ids", n * 4 * 4 }, .{ "weights", n * 4 * 4 } }) |e| @field(v, e[0]) = try v.w.store.alloc(@max(e[1], 256));
        for (&v.out) |*ptr| ptr.* = try v.w.store.alloc(@max(n / 4 * 5120 * 4, 256));
        return v;
    }
    pub fn deinit(v: *VisionEncoder) void {
        v.w.deinit();
    }
    fn pre(v: *VisionEncoder, n: []const u8, ins: []const dit.Io) !void {
        if (v.probe) |p| try p.before(p.ctx, n, ins);
    }
    fn post(v: *VisionEncoder, n: []const u8, outs: []const dit.Io) !void {
        if (v.probe) |p| try p.after(p.ctx, n, outs);
    }
    fn go(v: *VisionEncoder, f: cuda.Function, grid: cuda.launch.Dim3, b: u32, shared: u32, args: *cuda.launch.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = b }, .shared = shared }, v.s, args);
    }
    fn upload(v: *VisionEncoder, ptr: u64, bytes: []const u8) !void {
        try v.d.check(v.d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }
    fn linear(v: *VisionEncoder, name: []const u8, l: tw.Lin, x: u64, y: u64, m: u32) !void {
        try v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = @as(u64, m) * l.k * 4 }});
        var a: cuda.launch.Args = .{};
        if (l.quant) {
            inline for (.{ x, l.pqs, l.w, l.scale }) |p| a.add(p);
            a.add(l.tscale);
            a.add(l.b);
            a.add(y);
            inline for (.{ m, l.n, l.k }) |sz| a.add(@as(i64, sz));
            try v.go(v.tk.linear_fn, .{ .x = (m + 127) / 128, .y = (l.n + 63) / 64 }, 256, 0, &a);
        } else {
            inline for (.{ x, l.w, l.b, y }) |p| a.add(p);
            inline for (.{ m, l.n, l.k }) |sz| a.add(@as(i64, sz));
            try v.go(v.h3.gemm_fn, .{ .x = (l.n + 63) / 64, .y = (m + 63) / 64 }, 256, 0, &a);
        }
        try v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = @as(u64, m) * l.n * 4 }});
    }
    fn norm(v: *VisionEncoder, name: []const u8, n: tw.Norm, x: u64, y: u64, rows: u32) !void {
        const bytes = @as(u64, rows) * n.dim * 4;
        try v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = bytes }});
        var a: cuda.launch.Args = .{};
        inline for (.{ x, n.w, n.b, y }) |p| a.add(p);
        a.add(@as(i32, @intCast(n.dim)));
        try v.go(v.k.norm_fn, .{ .x = rows }, 256, 0, &a);
        try v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = bytes }});
    }
    fn gelu(v: *VisionEncoder, name: []const u8, x: u64, y: u64, n: u64, approx: i32) !void {
        try v.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = n * 4 }});
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        a.add(approx);
        try v.go(v.k.gelu_fn, .{ .x = blocks(n) }, 256, 0, &a);
        try v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = n * 4 }});
    }
    fn add(v: *VisionEncoder, name: []const u8, x: u64, b: u64, y: u64, n: u64) !void {
        try v.pre(name, &.{ .{ .role = "a", .ptr = x, .bytes = n * 4 }, .{ .role = "b", .ptr = b, .bytes = n * 4 } });
        var a: cuda.launch.Args = .{};
        inline for (.{ x, b, y }) |p| a.add(p);
        a.add(@as(i64, @intCast(n)));
        try v.go(v.tk.add_fn, .{ .x = blocks(n) }, 256, 0, &a);
        try v.post(name, &.{.{ .role = "y", .ptr = y, .bytes = n * 4 }});
    }
    fn merger(v: *VisionEncoder, j: usize, n: u32) !void {
        var nb: [64]u8 = undefined;
        const prefix = if (j == 3) "vision.merger" else try std.fmt.bufPrint(&nb, "vision.deepstack.{d}", .{j});
        var ob: [96]u8 = undefined;
        const m = v.w.mergers[j];
        try v.norm(try std.fmt.bufPrint(&ob, "{s}.norm", .{prefix}), m.norm, v.h, v.x, if (j == 3) n else n / 4);
        try v.linear(try std.fmt.bufPrint(&ob, "{s}.fc1", .{prefix}), m.fc1, v.x, v.fc, n / 4);
        try v.gelu(try std.fmt.bufPrint(&ob, "{s}.gelu", .{prefix}), v.fc, v.gel, @as(u64, n / 4) * 4608, 0);
        try v.linear(try std.fmt.bufPrint(&ob, "{s}.fc2", .{prefix}), m.fc2, v.gel, v.out[j], n / 4);
    }
    pub fn forward(v: *VisionEncoder, rgb: []const f32, width: u32, height: u32) !Features {
        if (width % 32 != 0 or height % 32 != 0 or @as(u64, width) * height < 3136 or @as(u64, width) * height > 12845056 or rgb.len != @as(u64, width) * height * 3) return error.BadVisionCanvas;
        const gh = height / 16;
        const gw = width / 16;
        const n = gh * gw;
        if (n > v.max_patches) return error.VisionCanvasTooLarge;
        const patches = try v.gpa.alloc(f32, @as(usize, n) * 1536);
        defer v.gpa.free(patches);
        const cos = try v.gpa.alloc(f32, @as(usize, n) * 36);
        defer v.gpa.free(cos);
        const sin = try v.gpa.alloc(f32, cos.len);
        defer v.gpa.free(sin);
        const ids = try v.gpa.alloc(i32, @as(usize, n) * 4);
        defer v.gpa.free(ids);
        const weights = try v.gpa.alloc(f32, ids.len);
        defer v.gpa.free(weights);
        var inv: [18]f32 = undefined;
        for (&inv, 0..) |*f, j| f.* = 1.0 / @as(f32, @floatCast(smath.exp((@as(f64, @floatFromInt(2 * j)) / 36) * smath.log(10000))));
        for (0..n) |row| {
            const bh = row / 4 / (gw / 2);
            const bw = row / 4 % (gw / 2);
            const ph = bh * 2 + row % 4 / 2;
            const pw = bw * 2 + row % 2;
            for (0..3) |c| for (0..2) |t| for (0..16) |dy| for (0..16) |dx| {
                patches[row * 1536 + c * 512 + t * 256 + dy * 16 + dx] = (rgb[((ph * 16 + dy) * width + pw * 16 + dx) * 3 + c] - 0.5) / 0.5;
            };
            for (inv, 0..) |f, j| {
                const h = smath.sincos(@as(f64, @as(f32, @floatFromInt(ph)) * f));
                const w = smath.sincos(@as(f64, @as(f32, @floatFromInt(pw)) * f));
                cos[row * 36 + j] = @floatCast(h.cos);
                sin[row * 36 + j] = @floatCast(h.sin);
                cos[row * 36 + 18 + j] = @floatCast(w.cos);
                sin[row * 36 + 18 + j] = @floatCast(w.sin);
            }
            const hp = linspace47(@intCast(ph), gh);
            const wp = linspace47(@intCast(pw), gw);
            const hi: u32 = @intFromFloat(hp);
            const wi: u32 = @intFromFloat(wp);
            const hc: u32 = @min(hi + 1, 47); // typed: @min with a constant narrows to the constant's width (u6), and hc * 48 overflows
            const wc: u32 = @min(wi + 1, 47);
            const dh = hp - @as(f32, @floatFromInt(hi));
            const dw = wp - @as(f32, @floatFromInt(wi));
            ids[row] = @intCast(hi * 48 + wi);
            ids[n + row] = @intCast(hi * 48 + wc);
            ids[n * 2 + row] = @intCast(hc * 48 + wi);
            ids[n * 3 + row] = @intCast(hc * 48 + wc);
            weights[row] = (1 - dh) * (1 - dw);
            weights[n + row] = (1 - dh) * dw;
            weights[n * 2 + row] = dh * (1 - dw);
            weights[n * 3 + row] = dh * dw;
        }
        try v.s.synchronize();
        try v.upload(v.pixels, std.mem.sliceAsBytes(patches));
        try v.upload(v.cs, std.mem.sliceAsBytes(cos));
        try v.upload(v.sn, std.mem.sliceAsBytes(sin));
        try v.upload(v.ids, std.mem.sliceAsBytes(ids));
        try v.upload(v.weights, std.mem.sliceAsBytes(weights));
        try v.pre("vision.patches", &.{});
        try v.post("vision.patches", &.{.{ .role = "y", .ptr = v.pixels, .bytes = @as(u64, n) * 1536 * 4 }});
        try v.linear("vision.patch", v.w.patch, v.pixels, v.h, n);
        try v.pre("vision.position", &.{ .{ .role = "ids", .ptr = v.ids, .bytes = @as(u64, n) * 16 }, .{ .role = "weights", .ptr = v.weights, .bytes = @as(u64, n) * 16 } });
        {
            var a: cuda.launch.Args = .{};
            inline for (.{ v.w.pos, v.ids, v.weights, v.z }) |p| a.add(p);
            a.add(@as(i32, @intCast(n)));
            a.add(@as(i32, D));
            try v.go(v.k.position_fn, .{ .x = blocks(@as(u64, n) * D) }, 256, 0, &a);
        }
        try v.post("vision.position", &.{.{ .role = "y", .ptr = v.z, .bytes = @as(u64, n) * D * 4 }});
        try v.add("vision.position_add", v.h, v.z, v.h, @as(u64, n) * D);
        var nb: [64]u8 = undefined;
        const nm = struct {
            fn f(buf: []u8, i: usize, op: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "vision.{d}.{s}", .{ i, op }) catch unreachable;
            }
        }.f;
        for (v.w.blocks, 0..) |b, i| {
            try v.norm(nm(&nb, i, "norm1"), b.n1, v.h, v.x, n);
            try v.linear(nm(&nb, i, "qkv"), b.qkv, v.x, v.qkv, n);
            // QKV projection is interleaved by token; copy each contiguous row to the shared kernel's layout.
            for (0..n) |r| for ([3]u64{ v.q, v.kk, v.v }, 0..) |dst, j| try v.d.check(v.d.api.cuMemcpyDtoDAsync_v2(dst + r * D * 4, v.qkv + (r * 3 + j) * D * 4, D * 4, v.s.handle), "cuMemcpyDtoDAsync");
            const bytes = @as(u64, n) * D * 4;
            const rn = nm(&nb, i, "rope");
            try v.pre(rn, &.{ .{ .role = "q", .ptr = v.q, .bytes = bytes }, .{ .role = "k", .ptr = v.kk, .bytes = bytes }, .{ .role = "cos", .ptr = v.cs, .bytes = @as(u64, n) * 36 * 4 }, .{ .role = "sin", .ptr = v.sn, .bytes = @as(u64, n) * 36 * 4 } });
            for ([2]u64{ v.q, v.kk }) |ptr| {
                var a: cuda.launch.Args = .{};
                a.add(ptr);
                a.add(v.cs);
                a.add(v.sn);
                a.add(@as(i32, HEADS));
                a.add(@as(i32, HD));
                try v.go(v.k.rope_fn, .{ .x = n, .y = HEADS }, 64, 0, &a);
            }
            try v.post(rn, &.{ .{ .role = "q", .ptr = v.q, .bytes = bytes }, .{ .role = "k", .ptr = v.kk, .bytes = bytes } });
            const an = nm(&nb, i, "attention");
            try v.pre(an, &.{ .{ .role = "q", .ptr = v.q, .bytes = bytes }, .{ .role = "k", .ptr = v.kk, .bytes = bytes }, .{ .role = "v", .ptr = v.v, .bytes = bytes } });
            {
                var a: cuda.launch.Args = .{};
                inline for (.{ v.q, v.kk, v.v, v.attn }) |p| a.add(p);
                a.add(@as(i32, @intCast(n)));
                a.add(@as(i32, HEADS));
                a.add(@as(i32, HD));
                a.add(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, HD)))));
                const shared = (n + 128) * 4;
                if (shared > 48 * 1024) try v.k.attn_fn.allowDynamicShared(shared);
                try v.go(v.k.attn_fn, .{ .x = n, .y = HEADS }, 128, shared, &a);
            }
            try v.post(an, &.{.{ .role = "y", .ptr = v.attn, .bytes = bytes }});
            try v.linear(nm(&nb, i, "proj"), b.proj, v.attn, v.z, n);
            try v.add(nm(&nb, i, "attention_add"), v.h, v.z, v.h, @as(u64, n) * D);
            try v.norm(nm(&nb, i, "norm2"), b.n2, v.h, v.x, n);
            try v.linear(nm(&nb, i, "fc1"), b.fc1, v.x, v.fc, n);
            try v.gelu(nm(&nb, i, "gelu"), v.fc, v.gel, @as(u64, n) * 4304, 1);
            try v.linear(nm(&nb, i, "fc2"), b.fc2, v.gel, v.z, n);
            try v.add(nm(&nb, i, "mlp_add"), v.h, v.z, v.h, @as(u64, n) * D);
            if (i == 8 or i == 16 or i == 24) try v.merger(if (i == 8) 0 else if (i == 16) 1 else 2, n);
        }
        try v.merger(3, n);
        return .{ .merged = v.out[3], .deepstack = .{ v.out[0], v.out[1], v.out[2] }, .grid_h = gh, .grid_w = gw };
    }
};
