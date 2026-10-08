//! tfimage's three Triton kernels (adaln, RMSNorm + RoPE, SwiGLU) from cubins compiled by Triton 3.7, launched as its
//! own launcher does (non-constexpr arguments in signature order, then the global and profile scratch pointers).
//! The cubins come from a capture (`cubins/<sha>`) or, for the shipping engine, from the AOT set built per SM.

const std = @import("std");
const cuda = @import("cuda");

pub const Spec = struct { cubin: []const u8, name: [:0]const u8, num_warps: u32, shared: u32 };

pub const Triton = struct {
    adaln: cuda.triton.Kernel,
    rope: cuda.triton.Kernel,
    swiglu: ?cuda.triton.Kernel,

    pub fn load(d: *const cuda.Driver, device: cuda.abi.Device, adaln: Spec, rope: Spec, swiglu: ?Spec) !Triton {
        return .{
            .adaln = try kernel(d, device, adaln),
            .rope = try kernel(d, device, rope),
            .swiglu = if (swiglu) |s| try kernel(d, device, s) else null,
        };
    }

    fn kernel(d: *const cuda.Driver, device: cuda.abi.Device, s: Spec) !cuda.triton.Kernel {
        return cuda.triton.Kernel.load(d, device, s.cubin, .{ .name = s.name, .num_warps = s.num_warps, .shared = s.shared }, s.name);
    }

    pub fn unload(t: *Triton) void {
        t.adaln.unload();
        t.rope.unload();
        if (t.swiglu) |*k| k.unload();
    }

    /// out[B * rows, D] = LayerNorm(x) * (1 + s[b]) (fp32 inside, bf16 out); s is fp32 [B, D]. Grid B * rows.
    pub fn adalnRun(t: *const Triton, st: cuda.Stream, x: u64, s: u64, out: u64, b: u32, rows: u32, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        a.add(x);
        a.add(s);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(eps);
        try t.adaln.launchOn(.{ .x = b * rows }, st, &a, .{}, &.{});
    }

    /// RMSNorm + RoPE of one part (0 q, 1 k) of qkv [B, N, 3, 32, 128] into dst (row strides in elements).
    pub const Rope = struct { src: u64, w: u64, cos: u64, sin: u64, dst: u64, src_b_stride: u32, src_row_stride: u32, src_off: u32, dst_b_stride: u32, dst_row_stride: u32, n: u32, b: u32 };

    pub fn ropeRun(t: *const Triton, st: cuda.Stream, r: Rope, eps: f32) !void {
        var a: cuda.launch.Args = .{};
        inline for (.{ r.src, r.w, r.cos, r.sin, r.dst }) |p| a.add(p);
        inline for (.{ r.src_b_stride, r.src_row_stride, r.src_off, r.dst_b_stride, r.dst_row_stride, r.n }) |v| a.add(@as(i32, @intCast(v)));
        a.add(eps);
        try t.rope.launchOn(.{ .x = r.b * r.n * 32 }, st, &a, .{}, &.{});
    }

    /// [M, 2F] = [gate | up] -> silu(gate) * up [M, F] (F = 12288, BLOCK 2048). Grid (M, F / 2048).
    pub fn swigluRun(t: *const Triton, st: cuda.Stream, gu: u64, out: u64, m: u32, f: u32) !void {
        var a: cuda.launch.Args = .{};
        a.add(gu);
        a.add(out);
        try t.swiglu.?.launchOn(.{ .x = m, .y = (f + 2047) / 2048 }, st, &a, .{}, &.{});
    }
};

/// adaln, rms_rope and swiglu from a capture's ops.jsonl and cubins/ (the first launch of each); memory from `a`.
pub fn specsFromCapture(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![3]?Spec {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "ops.jsonl" }), a, .limited(256 << 20));
    var specs: [3]?Spec = .{ null, null, null };
    const names = [_][]const u8{ "_adaln_kernel", "_rms_rope_kernel", "_swiglu_kernel" };
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "\"launches\": [{") == null) continue;
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{});
        for (v.object.get("launches").?.array.items) |l| {
            const kname = l.object.get("kernel").?.string;
            for (names, 0..) |nm, i| {
                if (specs[i] != null or !std.mem.eql(u8, nm, kname)) continue;
                const sha = l.object.get("cubin").?.string;
                const raw = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "cubins", sha }), a, .limited(64 << 20));
                const cubin = try a.alignedAlloc(u8, .@"16", raw.len); // the driver loads images from aligned memory
                @memcpy(cubin, raw);
                specs[i] = .{
                    .cubin = cubin,
                    .name = try a.dupeSentinel(u8, nm, 0),
                    .num_warps = @intCast(l.object.get("num_warps").?.integer),
                    .shared = @intCast(l.object.get("shared").?.integer),
                };
            }
        }
    }
    return specs;
}

/// adaln, rms_rope and swiglu for SM major.minor from a pack's `triton/` (`stk_twin triton`); memory from `a`.
pub fn specsFromPack(a: std.mem.Allocator, io: std.Io, pack_dir: []const u8, major: u32, minor: u32) ![3]?Spec {
    const dir = try std.fs.path.join(a, &.{ pack_dir, "triton" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "triton.json" }), a, .limited(1 << 20));
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const kernels = v.object.get("kernels").?.object;
    var sm_buf: [8]u8 = undefined;
    const sm = try std.fmt.bufPrint(&sm_buf, "{d}{d}", .{ major, minor });
    var specs: [3]?Spec = .{ null, null, null };
    const names = [_][]const u8{ "_adaln_kernel", "_rms_rope_kernel", "_swiglu_kernel" };
    for (names, &specs) |nm, *out| {
        const k = kernels.get(nm) orelse continue;
        const o = (k.object.get(sm) orelse return error.NoCubinForThisGpu).object;
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, o.get("cubin").?.string }), a, .limited(64 << 20));
        const cubin = try a.alignedAlloc(u8, .@"16", raw.len);
        @memcpy(cubin, raw);
        out.* = .{ .cubin = cubin, .name = try a.dupeSentinel(u8, nm, 0), .num_warps = @intCast(o.get("num_warps").?.integer), .shared = @intCast(o.get("shared").?.integer) };
    }
    return specs;
}
