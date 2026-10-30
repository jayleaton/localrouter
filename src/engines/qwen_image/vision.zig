//! Vision tower in twin REC order. Host plans interpolate the learned positions and freeze axial rotary phases;
//! temporal slices attend independently, patch rows are processor merge-block order.
const std = @import("std");
const cuda = @import("cuda");
const Pack = @import("pack.zig").Pack;
const DType = @import("pack.zig").DType;
const Store = @import("weights.zig").Store;
const Uploader = @import("upload.zig").Uploader;
const launch = @import("ops_launch.zig");
const vis = @import("vision_launch.zig");
const smath = @import("smath.zig");
const Io = @import("dit.zig").Io;
const Probe = @import("dit.zig").Probe;
pub const Grid = struct { t: u32 = 1, h: u32, w: u32 };
pub const Config = struct {
    depth: u32,
    hidden_size: u32,
    num_heads: u32,
    spatial_merge_size: u32 = 2,
    deepstack_visual_indexes: []const u32,
    layer_norm_eps: f32 = 1e-6,
    hidden_act: []const u8 = "gelu_pytorch_tanh",
};
const Linear = struct { w: u64, bias: u64, n: u32, k: u32 };
const Norm = struct { w: u64, bias: u64, dim: u32 };
const Block = struct { norm1: Norm, qkv: Linear, proj: Linear, norm2: Norm, fc1: Linear, fc2: Linear };
const Merger = struct { norm: Norm, fc1: Linear, fc2: Linear };
pub const Result = struct { merged: u64, deepstack: []const u64, rows: u32 };

pub fn rowCount(grid: []const Grid) !u32 {
    if (grid.len == 0) return error.BadVisionGrid;
    var rows: u64 = 0;
    for (grid) |g| {
        if (g.t == 0 or g.h == 0 or g.w == 0 or g.h % 2 != 0 or g.w % 2 != 0) return error.BadVisionGrid;
        rows = try std.math.add(u64, rows, try std.math.mul(u64, g.t, try std.math.mul(u64, g.h, g.w)));
    }
    return std.math.cast(u32, rows) orelse error.BadVisionGrid;
}
fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
}
// torch.linspace uses multiply from the nearer endpoint; preserve its fp32 arithmetic boundaries.
fn coordinate(i: u32, n: u32, side: u32) f32 {
    const end: f32 = @floatFromInt(side - 1);
    const step = end / @as(f32, @floatFromInt(n - 1));
    return if (i < n / 2) step * @as(f32, @floatFromInt(i)) else end - step * @as(f32, @floatFromInt(n - i - 1));
}
pub fn positionPlan(grid: []const Grid, side: u32, ids: []i32, weights: []u16) !void {
    const rows = try rowCount(grid);
    if (side == 0 or ids.len != 4 * @as(usize, rows) or weights.len != ids.len) return error.BadVisionGrid;
    var r: usize = 0;
    for (grid) |g| for (0..g.t) |_| {
        var bh: u32 = 0;
        while (bh < g.h) : (bh += 2) {
            var bw: u32 = 0;
            while (bw < g.w) : (bw += 2) for (0..2) |i| for (0..2) |j| {
                const h = coordinate(bh + @as(u32, @intCast(i)), g.h, side);
                const w = coordinate(bw + @as(u32, @intCast(j)), g.w, side);
                const hf: u32 = @intFromFloat(h);
                const wf: u32 = @intFromFloat(w);
                const hc = @min(hf + 1, side - 1);
                const wc = @min(wf + 1, side - 1);
                const dh = h - @as(f32, @floatFromInt(hf));
                const dw = w - @as(f32, @floatFromInt(wf));
                const ii = [_]u32{ hf * side + wf, hf * side + wc, hc * side + wf, hc * side + wc };
                const ww = [_]f32{ (1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw };
                for (ii, ww, 0..) |idx, weight, k| {
                    ids[k * rows + r] = @intCast(idx);
                    weights[k * rows + r] = bf16(weight);
                }
                r += 1;
            };
        }
    };
}
pub fn ropeTables(grid: []const Grid, inv: *const [18]f32, cos: []f32, sin: []f32) !void {
    const rows = try rowCount(grid);
    if (cos.len != @as(usize, rows) * 36 or sin.len != cos.len) return error.BadVisionGrid;
    var r: usize = 0;
    for (grid) |g| for (0..g.t) |_| {
        var bh: u32 = 0;
        while (bh < g.h) : (bh += 2) {
            var bw: u32 = 0;
            while (bw < g.w) : (bw += 2) for (0..2) |i| for (0..2) |j| {
                for ([_]u32{ bh + @as(u32, @intCast(i)), bw + @as(u32, @intCast(j)) }, 0..) |p, axis| for (inv, 0..) |f, k| {
                    const phase: f32 = @as(f32, @floatFromInt(p)) * f;
                    const sc = smath.sincos(@as(f64, phase));
                    cos[r * 36 + axis * 18 + k] = @floatCast(sc.cos);
                    sin[r * 36 + axis * 18 + k] = @floatCast(sc.sin);
                };
                r += 1;
            };
        }
    };
}

pub const VisionEncoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: *const vis.Ops,
    te: *const launch.TeOps,
    ops: *const launch.Ops,
    s: cuda.Stream,
    probe: ?Probe = null,
    store: Store,
    cfg: Config,
    max_rows: u32,
    output_dim: u32 = 0,
    side: u32 = 0,
    inv: [18]f32 = undefined,
    patch: Linear = undefined,
    position: u64 = 0,
    blocks: []Block = &.{},
    mergers: []Merger = &.{},
    final: Merger = undefined,
    outputs: []u64 = &.{},
    taps: []u32 = &.{},
    patches: u64 = 0,
    h: u64 = 0,
    x: u64 = 0,
    z: u64 = 0,
    wide: u64 = 0,
    q: u64 = 0,
    kk: u64 = 0,
    v: u64 = 0,
    at: u64 = 0,
    ids: u64 = 0,
    weights: u64 = 0,
    cos: u64 = 0,
    sin: u64 = 0,
    merged: u64 = 0,
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const vis.Ops, te: *const launch.TeOps, ops: *const launch.Ops, s: cuda.Stream, p: *const Pack, up: *Uploader, cfg: Config, max_rows: u32) !VisionEncoder {
        if (cfg.depth == 0 or cfg.num_heads == 0 or cfg.hidden_size != cfg.num_heads * 72 or cfg.spatial_merge_size != 2 or max_rows == 0 or (!std.mem.eql(u8, cfg.hidden_act, "gelu_pytorch_tanh") and !std.mem.eql(u8, cfg.hidden_act, "gelu_new"))) return error.BadVisionConfig;
        var t: VisionEncoder = .{ .gpa = gpa, .d = d, .k = k, .te = te, .ops = ops, .s = s, .store = .init(d, gpa), .cfg = cfg, .max_rows = max_rows };
        errdefer t.deinit();
        t.taps = try gpa.dupe(u32, cfg.deepstack_visual_indexes);
        t.cfg.deepstack_visual_indexes = t.taps;
        for (t.taps, 0..) |tap, i| if (tap >= cfg.depth or (i > 0 and t.taps[i - 1] >= tap)) return error.BadVisionConfig;
        t.blocks = try gpa.alloc(Block, cfg.depth);
        t.mergers = try gpa.alloc(Merger, t.taps.len);
        t.outputs = try gpa.alloc(u64, t.taps.len);
        t.patch = try t.linear(up, p, "vis.patch_embed.proj");
        if (t.patch.n != cfg.hidden_size or t.patch.k != 1536) return error.BadWeight;
        const pos = try p.get("vis.pos_embed.weight");
        if (pos.dtype != .bf16 or pos.rank != 2 or pos.shape[1] != cfg.hidden_size) return error.BadWeight;
        t.side = @intFromFloat(@sqrt(@as(f64, @floatFromInt(pos.shape[0]))));
        if (t.side == 0 or @as(usize, t.side) * t.side != pos.shape[0]) return error.BadWeight;
        t.position = try t.store.tensorOf(up, p, pos);
        const freq = try p.get("vis.inv_freq");
        if (freq.dtype != .f32 or freq.rank != 1 or freq.shape[0] != 18) return error.BadWeight;
        const raw = try t.store.readBulk(up.io, p, freq);
        for (&t.inv, 0..) |*f, i| f.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        var nb: [128]u8 = undefined;
        var wide_dim = cfg.hidden_size * 4;
        for (t.blocks, 0..) |*b, i| {
            b.norm1 = try t.normWeight(up, p, try std.fmt.bufPrint(&nb, "vis.blocks.{d}.norm1", .{i}));
            b.norm2 = try t.normWeight(up, p, try std.fmt.bufPrint(&nb, "vis.blocks.{d}.norm2", .{i}));
            b.qkv = try t.linear(up, p, try std.fmt.bufPrint(&nb, "vis.blocks.{d}.attn.qkv", .{i}));
            b.proj = try t.linear(up, p, try std.fmt.bufPrint(&nb, "vis.blocks.{d}.attn.proj", .{i}));
            b.fc1 = try t.linear(up, p, try std.fmt.bufPrint(&nb, "vis.blocks.{d}.mlp.linear_fc1", .{i}));
            b.fc2 = try t.linear(up, p, try std.fmt.bufPrint(&nb, "vis.blocks.{d}.mlp.linear_fc2", .{i}));
            if (b.norm1.dim != cfg.hidden_size or b.norm2.dim != cfg.hidden_size or b.qkv.k != cfg.hidden_size or b.qkv.n != cfg.hidden_size * 3 or b.proj.k != cfg.hidden_size or b.proj.n != cfg.hidden_size or b.fc1.k != cfg.hidden_size or b.fc2.k != b.fc1.n or b.fc2.n != cfg.hidden_size) return error.BadWeight;
            wide_dim = @max(wide_dim, b.fc1.n);
        }
        t.final = try t.merger(up, p, "vis.merger");
        t.output_dim = t.final.fc2.n;
        if (t.final.norm.dim != cfg.hidden_size) return error.BadWeight;
        wide_dim = @max(wide_dim, t.final.fc1.n);
        for (t.mergers, 0..) |*m, i| {
            m.* = try t.merger(up, p, try std.fmt.bufPrint(&nb, "vis.deepstack_merger_list.{d}", .{i}));
            if (m.norm.dim != cfg.hidden_size * 4 or m.fc2.n != t.final.fc2.n) return error.BadWeight;
            wide_dim = @max(wide_dim, m.fc1.n);
            t.outputs[i] = try t.alloc(@as(u64, max_rows) / 4 * m.fc2.n * 2);
        }
        const r: u64 = max_rows;
        t.patches = try t.alloc(r * t.patch.k * 2);
        inline for (.{ "h", "z" }) |name| @field(t, name) = try t.alloc(r * cfg.hidden_size * 2);
        t.x = try t.alloc(r * cfg.hidden_size * 4 * 2);
        t.wide = try t.alloc(r * @max(wide_dim, cfg.hidden_size * 3) * 2);
        inline for (.{ "q", "kk", "v", "at" }) |name| @field(t, name) = try t.alloc(r * cfg.num_heads * 128 * 2);
        t.ids = try t.alloc(r * 4 * 4);
        t.weights = try t.alloc(r * 4 * 2);
        t.cos = try t.alloc(r * 36 * 4);
        t.sin = try t.alloc(r * 36 * 4);
        t.merged = try t.alloc(r / 4 * t.final.fc2.n * 2);
        return t;
    }
    pub fn deinit(t: *VisionEncoder) void {
        t.store.deinit();
        t.gpa.free(t.blocks);
        t.gpa.free(t.mergers);
        t.gpa.free(t.outputs);
        t.gpa.free(t.taps);
    }
    fn alloc(t: *VisionEncoder, n: u64) !u64 {
        return t.store.alloc(@intCast(n));
    }
    fn linear(t: *VisionEncoder, up: *Uploader, p: *const Pack, name: []const u8) !Linear {
        var nb: [160]u8 = undefined;
        const w = try p.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{name}));
        if (w.dtype != .bf16 or w.rank != 2 or w.shape[0] == 0 or w.shape[1] == 0 or w.shape[1] % 8 != 0) return error.BadWeight;
        var bias: u64 = 0;
        if (p.tensors.get(try std.fmt.bufPrint(&nb, "{s}.bias", .{name}))) |b| {
            if (b.dtype != .bf16 or b.rank != 1 or b.shape[0] != w.shape[0]) return error.BadWeight;
            bias = try t.store.tensorOf(up, p, b);
        }
        return .{ .w = try t.store.tensorOf(up, p, w), .bias = bias, .n = @intCast(w.shape[0]), .k = @intCast(w.shape[1]) };
    }
    fn normWeight(t: *VisionEncoder, up: *Uploader, p: *const Pack, name: []const u8) !Norm {
        var nb: [160]u8 = undefined;
        const w = try p.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{name}));
        const b = try p.get(try std.fmt.bufPrint(&nb, "{s}.bias", .{name}));
        if (w.dtype != .bf16 or w.rank != 1 or w.shape[0] == 0 or b.dtype != .bf16 or b.rank != 1 or b.shape[0] != w.shape[0]) return error.BadWeight;
        return .{ .w = try t.store.tensorOf(up, p, w), .bias = try t.store.tensorOf(up, p, b), .dim = @intCast(w.shape[0]) };
    }
    fn merger(t: *VisionEncoder, up: *Uploader, p: *const Pack, name: []const u8) !Merger {
        var nb: [160]u8 = undefined;
        const m: Merger = .{ .norm = try t.normWeight(up, p, try std.fmt.bufPrint(&nb, "{s}.norm", .{name})), .fc1 = try t.linear(up, p, try std.fmt.bufPrint(&nb, "{s}.linear_fc1", .{name})), .fc2 = try t.linear(up, p, try std.fmt.bufPrint(&nb, "{s}.linear_fc2", .{name})) };
        if (m.fc1.k != t.cfg.hidden_size * 4 or m.fc2.k != m.fc1.n) return error.BadWeight;
        return m;
    }
    fn pre(t: *VisionEncoder, name: []const u8, ins: []const Io) !void {
        if (t.probe) |p| try p.before(p.ctx, name, ins);
    }
    fn post(t: *VisionEncoder, name: []const u8, outs: []const Io) !void {
        if (t.probe) |p| try p.after(p.ctx, name, outs);
    }
    fn gemm(t: *VisionEncoder, name: []const u8, x: u64, y: u64, rows: u32, w: Linear) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = @as(u64, rows) * w.k * 2 }});
        try t.te.gemm(t.s, x, w.w, w.bias, y, rows, w.n, w.k);
        try t.post(name, &.{.{ .role = "y", .ptr = y, .bytes = @as(u64, rows) * w.n * 2 }});
    }
    fn norm(t: *VisionEncoder, name: []const u8, x: u64, y: u64, rows: u32, w: Norm) !void {
        const n = @as(u64, rows) * w.dim * 2;
        try t.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = n }});
        try t.k.norm(t.s, x, w.w, w.bias, y, rows, w.dim, t.cfg.layer_norm_eps);
        try t.post(name, &.{.{ .role = "y", .ptr = y, .bytes = n }});
    }
    fn add(t: *VisionEncoder, name: []const u8, z: u64, n: u64) !void {
        try t.pre(name, &.{ .{ .role = "x", .ptr = t.h, .bytes = n * 2 }, .{ .role = "z", .ptr = z, .bytes = n * 2 } });
        try t.te.add(t.s, t.h, z, t.h, n);
        try t.post(name, &.{.{ .role = "y", .ptr = t.h, .bytes = n * 2 }});
    }
    fn gelu(t: *VisionEncoder, name: []const u8, x: u64, n: u64, exact: bool) !void {
        try t.pre(name, &.{.{ .role = "x", .ptr = x, .bytes = n * 2 }});
        if (exact) try t.k.unary(t.s, .gelu, x, x, n) else try t.ops.pointwise(t.s, .gelu_tanh, x, x, n);
        try t.post(name, &.{.{ .role = "y", .ptr = x, .bytes = n * 2 }});
    }
    fn merge(t: *VisionEncoder, name: []const u8, rows: u32, m: Merger, postshuffle: bool, out: u64) !void {
        var nb: [160]u8 = undefined;
        try t.norm(try std.fmt.bufPrint(&nb, "{s}.norm", .{name}), t.h, t.x, if (postshuffle) rows / 4 else rows, m.norm);
        try t.gemm(try std.fmt.bufPrint(&nb, "{s}.linear_fc1", .{name}), t.x, t.wide, rows / 4, m.fc1);
        try t.gelu(try std.fmt.bufPrint(&nb, "{s}.gelu", .{name}), t.wide, @as(u64, rows / 4) * m.fc1.n, true);
        try t.gemm(try std.fmt.bufPrint(&nb, "{s}.linear_fc2", .{name}), t.wide, out, rows / 4, m.fc2);
    }
    fn upload(t: *VisionEncoder, ptr: u64, bytes: []const u8) !void {
        try t.d.check(t.d.api.cuMemcpyHtoD_v2(ptr, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }
    pub fn forward(t: *VisionEncoder, patches: u64, dtype: DType, grid: []const Grid) !Result {
        const rows = try rowCount(grid);
        if (rows > t.max_rows or (dtype != .bf16 and dtype != .f32)) return error.BadVisionInput;
        const ids = try t.gpa.alloc(i32, @as(usize, rows) * 4);
        defer t.gpa.free(ids);
        const weights = try t.gpa.alloc(u16, ids.len);
        defer t.gpa.free(weights);
        const tables = try t.gpa.alloc(f32, @as(usize, rows) * 72);
        defer t.gpa.free(tables);
        try positionPlan(grid, t.side, ids, weights);
        try ropeTables(grid, &t.inv, tables[0 .. rows * 36], tables[rows * 36 ..]);
        try t.upload(t.ids, std.mem.sliceAsBytes(ids));
        try t.upload(t.weights, std.mem.sliceAsBytes(weights));
        try t.upload(t.cos, std.mem.sliceAsBytes(tables[0 .. rows * 36]));
        try t.upload(t.sin, std.mem.sliceAsBytes(tables[rows * 36 ..]));
        var px = patches;
        if (dtype == .f32) {
            const count = @as(u64, rows) * t.patch.k;
            try t.pre("vis.patch_cast", &.{.{ .role = "x", .ptr = patches, .bytes = count * 4 }});
            try t.k.unary(t.s, .from_f32, patches, t.patches, count);
            try t.post("vis.patch_cast", &.{.{ .role = "y", .ptr = t.patches, .bytes = count * 2 }});
            px = t.patches;
        }
        try t.gemm("vis.patch_embed.proj", px, t.h, rows, t.patch);
        try t.pre("vis.position", &.{ .{ .role = "ids", .ptr = t.ids, .bytes = @as(u64, rows) * 16 }, .{ .role = "weights", .ptr = t.weights, .bytes = @as(u64, rows) * 8 } });
        try t.k.position(t.s, t.position, t.ids, t.weights, t.z, rows, t.cfg.hidden_size);
        const n = @as(u64, rows) * t.cfg.hidden_size;
        try t.post("vis.position", &.{.{ .role = "y", .ptr = t.z, .bytes = n * 2 }});
        try t.add("vis.position_add", t.z, n);
        var nb: [160]u8 = undefined;
        const nm = struct {
            fn f(buf: []u8, i: usize, op: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "vis.blocks.{d}.{s}", .{ i, op }) catch unreachable;
            }
        }.f;
        for (t.blocks, 0..) |b, i| {
            try t.norm(nm(&nb, i, "norm1"), t.h, t.x, rows, b.norm1);
            try t.gemm(nm(&nb, i, "attn.qkv"), t.x, t.wide, rows, b.qkv);
            const rope_name = nm(&nb, i, "rope_pad");
            const tb = @as(u64, rows) * 36 * 4;
            try t.pre(rope_name, &.{ .{ .role = "x", .ptr = t.wide, .bytes = n * 6 }, .{ .role = "cos", .ptr = t.cos, .bytes = tb }, .{ .role = "sin", .ptr = t.sin, .bytes = tb } });
            try t.k.ropePad(t.s, t.wide, t.cos, t.sin, t.q, t.kk, t.v, rows, t.cfg.num_heads);
            const padded = @as(u64, rows) * t.cfg.num_heads * 128 * 2;
            try t.post(rope_name, &.{ .{ .role = "q", .ptr = t.q, .bytes = padded }, .{ .role = "k", .ptr = t.kk, .bytes = padded }, .{ .role = "v", .ptr = t.v, .bytes = padded } });
            var start: u64 = 0;
            var slice: usize = 0;
            for (grid) |g| for (0..g.t) |_| {
                const len = g.h * g.w;
                const offset = start * t.cfg.num_heads * 128 * 2;
                const size = @as(u64, len) * t.cfg.num_heads * 128 * 2;
                const name = try std.fmt.bufPrint(&nb, "vis.blocks.{d}.attention.{d}", .{ i, slice });
                try t.pre(name, &.{ .{ .role = "q", .ptr = t.q + offset, .bytes = size }, .{ .role = "k", .ptr = t.kk + offset, .bytes = size }, .{ .role = "v", .ptr = t.v + offset, .bytes = size } });
                try vis.Ops.attention(t.ops, t.s, t.q + offset, t.kk + offset, t.v + offset, t.at + offset, len, t.cfg.num_heads);
                try t.post(name, &.{.{ .role = "y", .ptr = t.at + offset, .bytes = size }});
                start += len;
                slice += 1;
            };
            const unpad_name = nm(&nb, i, "unpad");
            try t.pre(unpad_name, &.{.{ .role = "x", .ptr = t.at, .bytes = padded }});
            try t.k.unary(t.s, .unpad, t.at, t.x, n);
            try t.post(unpad_name, &.{.{ .role = "y", .ptr = t.x, .bytes = n * 2 }});
            try t.gemm(nm(&nb, i, "attn.proj"), t.x, t.z, rows, b.proj);
            try t.add(nm(&nb, i, "attn_residual"), t.z, n);
            try t.norm(nm(&nb, i, "norm2"), t.h, t.x, rows, b.norm2);
            try t.gemm(nm(&nb, i, "mlp.linear_fc1"), t.x, t.wide, rows, b.fc1);
            try t.gelu(nm(&nb, i, "mlp.gelu"), t.wide, @as(u64, rows) * b.fc1.n, false);
            try t.gemm(nm(&nb, i, "mlp.linear_fc2"), t.wide, t.z, rows, b.fc2);
            try t.add(nm(&nb, i, "mlp_residual"), t.z, n);
            for (t.taps, 0..) |tap, j| if (tap == i) try t.merge(try std.fmt.bufPrint(&nb, "vis.deepstack_merger_list.{d}", .{j}), rows, t.mergers[j], true, t.outputs[j]);
        }
        try t.merge("vis.merger", rows, t.final, false, t.merged);
        return .{ .merged = t.merged, .deepstack = t.outputs, .rows = rows / 4 };
    }
};

test "vision merge coordinates and interpolation corners" {
    const grid = [_]Grid{.{ .h = 2, .w = 2 }};
    var ids: [16]i32 = undefined;
    var weights: [16]u16 = undefined;
    try positionPlan(&grid, 48, &ids, &weights);
    try std.testing.expectEqualSlices(i32, &.{ 0, 47, 2256, 2303 }, ids[0..4]);
    try std.testing.expectEqualSlices(u16, &.{ 0x3f80, 0x3f80, 0x3f80, 0x3f80 }, weights[0..4]);
    var cos: [144]f32 = undefined;
    var sin: [144]f32 = undefined;
    const inv: [18]f32 = @splat(1);
    try ropeTables(&grid, &inv, &cos, &sin);
    try std.testing.expectEqual(@as(f32, 1), cos[0]);
    try std.testing.expectEqual(@as(f32, 0), sin[0]);
    try std.testing.expectEqual(@as(f32, 0), sin[36]);
    try std.testing.expectEqual(@as(f32, @floatCast(smath.sincos(1).sin)), sin[54]);
}

test "vision tables follow merge blocks and repeat temporal frames" {
    const grid = [_]Grid{.{ .t = 2, .h = 4, .w = 4 }};
    var ids: [128]i32 = undefined;
    var weights: [128]u16 = undefined;
    try positionPlan(&grid, 4, &ids, &weights);
    const frame = [_]i32{ 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 };
    try std.testing.expectEqualSlices(i32, &frame, ids[0..16]);
    try std.testing.expectEqualSlices(i32, &frame, ids[16..32]);
    try std.testing.expectError(error.BadVisionGrid, rowCount(&.{.{ .h = 3, .w = 2 }}));
}
