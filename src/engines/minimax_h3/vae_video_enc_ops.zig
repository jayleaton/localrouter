//! One-frame encoder launches; matrix products and tile placement reuse the decoder's identical operations.
const cuda = @import("cuda");
fn blocks(n: u64) u32 {
    return @intCast((n + 255) / 256);
}
pub const Ops = struct {
    module: cuda.Module,
    pixels_fn: cuda.Function,
    im2col_fn: cuda.Function,
    norm_fn: cuda.Function,
    silu_fn: cuda.Function,
    add_fn: cuda.Function,
    moments_fn: cuda.Function,
    latent_fn: cuda.Function,
    pub fn load(d: *const cuda.Driver, image: []const u8) !Ops {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .pixels_fn = try m.function("ve_pixels"), .im2col_fn = try m.function("ve_im2col"), .norm_fn = try m.function("ve_group_norm"), .silu_fn = try m.function("ve_silu"), .add_fn = try m.function("ve_add"), .moments_fn = try m.function("ve_moments"), .latent_fn = try m.function("ve_latent") };
    }
    pub fn unload(o: *Ops) void {
        o.module.unload();
    }
    fn go(f: cuda.Function, s: cuda.Stream, n: u64, a: *cuda.launch.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = blocks(n) }, .block = .{ .x = 256 } }, s, a);
    }
    pub fn pixels(o: *const Ops, s: cuda.Stream, rgb: u64, out: u64, W: u32, y0: u32, x0: u32, h: u32, w: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(rgb);
        a.add(out);
        inline for (.{ W, y0, x0, h, w }) |v| a.add(@as(i32, @intCast(v)));
        try go(o.pixels_fn, s, 3 * @as(u64, h) * w, &a);
    }
    pub fn im2col(o: *const Ops, s: cuda.Stream, x: u64, out: u64, H: u32, W: u32, C: u32, k: u32, stride: u32, pad: u32) !void {
        const K = (C * k * k + 7) / 8 * 8;
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(out);
        inline for (.{ H, W, C, k, stride, pad, H / stride, W / stride, K }) |v| a.add(@as(i32, @intCast(v)));
        try go(o.im2col_fn, s, @as(u64, H / stride) * (W / stride) * K, &a);
    }
    pub fn norm(o: *const Ops, s: cuda.Stream, x: u64, weight: u64, bias: u64, y: u64, C: u32, P: u32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ x, weight, bias, y }) |v| a.add(v);
        a.add(@as(i32, @intCast(C)));
        a.add(@as(i32, @intCast(P)));
        a.add(@as(i32, 32));
        a.add(@as(f32, 1e-6));
        try cuda.launch.launch(o.norm_fn, .{ .grid = .{ .x = 32 }, .block = .{ .x = 256 } }, s, &a);
    }
    pub fn silu(o: *const Ops, s: cuda.Stream, x: u64, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        try go(o.silu_fn, s, n, &a);
    }
    pub fn add(o: *const Ops, s: cuda.Stream, x: u64, residual: u64, n: u64) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(residual);
        a.add(@as(i64, @intCast(n)));
        try go(o.add_fn, s, n, &a);
    }
    pub fn moments(o: *const Ops, s: cuda.Stream, rows: u64, chw: u64, P: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(rows);
        a.add(chw);
        a.add(@as(i32, @intCast(P)));
        try go(o.moments_fn, s, 48 * @as(u64, P), &a);
    }
    pub fn latent(o: *const Ops, s: cuda.Stream, moments_ptr: u64, mean: u64, stdv: u64, latent_ptr: u64, P: u32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ moments_ptr, mean, stdv, latent_ptr }) |v| a.add(v);
        a.add(@as(i32, @intCast(P)));
        try go(o.latent_fn, s, 24 * @as(u64, P), &a);
    }
};
