//! vision.cu launch shapes shared with the twin; GEMM, add, tanh GELU and attention reuse existing modules.
const cuda = @import("cuda");
const base = @import("ops_launch.zig");
pub const Ops = struct {
    module: cuda.Module,
    ln: cuda.Function, pos: cuda.Function, qkv: cuda.Function, unpad_fn: cuda.Function, gelu_fn: cuda.Function, cast_fn: cuda.Function,
    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .ln = try m.function("stk_vis_ln"), .pos = try m.function("stk_vis_pos"), .qkv = try m.function("stk_vis_qkv"), .unpad_fn = try m.function("stk_vis_unpad"), .gelu_fn = try m.function("stk_vis_gelu"), .cast_fn = try m.function("stk_vis_from_f32") };
    }
    pub fn unload(o: *Ops) void { o.module.unload(); }
    pub fn norm(o: *const Ops, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: u32, dim: u32, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, w, b, y }) |p| a.add(p);
        a.add(@as(i32, @intCast(dim))); a.add(eps);
        try base.Ops.go(o.ln, s, .{ .x = rows }, 0, &a);
    }
    pub fn position(o: *const Ops, s: cuda.Stream, table: u64, ids: u64, weights: u64, y: u64, rows: u32, dim: u32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ table, ids, weights, y }) |p| a.add(p);
        a.add(@as(i32, @intCast(rows))); a.add(@as(i32, @intCast(dim)));
        try base.Ops.go(o.pos, s, .{ .x = blocks(@as(u64, rows) * dim) }, 0, &a);
    }
    pub fn ropePad(o: *const Ops, s: cuda.Stream, x: u64, cos: u64, sin: u64, q: u64, k: u64, v: u64, rows: u32, heads: u32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, cos, sin, q, k, v }) |p| a.add(p);
        a.add(@as(i32, @intCast(rows))); a.add(@as(i32, @intCast(heads)));
        try base.Ops.go(o.qkv, s, .{ .x = blocks(@as(u64, rows) * heads * 128) }, 0, &a);
    }
    pub const Unary = enum { unpad, gelu, from_f32 };
    pub fn unary(o: *const Ops, s: cuda.Stream, op: Unary, x: u64, y: u64, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x); a.add(y); a.add(@as(i64, @intCast(n)));
        try base.Ops.go(switch (op) { .unpad => o.unpad_fn, .gelu => o.gelu_fn, .from_f32 => o.cast_fn }, s, .{ .x = blocks(n) }, 0, &a);
    }
    /// Existing 128-wide attention kernel, with vision's 72-wide scale.
    pub fn attention(o: *const base.Ops, s: cuda.Stream, q: u64, k: u64, v: u64, y: u64, rows: u32, heads: u32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ q, k, v, y }) |p| a.add(p);
        inline for (.{ @as(u32, 0), rows, heads, heads, @as(u32, 1), rows, @as(u32, 0) }) |n| a.add(@as(i32, @intCast(n)));
        a.add(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, 72)))));
        try cuda.launch.launch(o.attn, .{ .grid = .{ .x = (rows + base.Ops.attn_rows - 1) / base.Ops.attn_rows, .y = heads }, .block = .{ .x = base.Ops.attn_threads }, .shared = base.Ops.attn_smem }, s, &a);
    }
};
fn blocks(n: u64) u32 { return @intCast((n + 255) / 256); }
