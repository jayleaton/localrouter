//! Vision tensors widened exactly to fp32; NVFP4 linears retain checkpoint storage.
const std = @import("std");
const cuda = @import("cuda");
const qi = @import("qwen_image");
pub const Lin = struct { w: u64, b: u64 = 0, n: u32, k: u32, scale: u64 = 0, tscale: f32 = 0, pqs: u64 = 0, quant: bool = false };
pub const Norm = struct { w: u64, b: u64, dim: u32 };
pub const Block = struct { n1: Norm, n2: Norm, qkv: Lin, proj: Lin, fc1: Lin, fc2: Lin };
pub const Merger = struct { norm: Norm, fc1: Lin, fc2: Lin };
pub const Weights = struct {
    store: qi.weights.Store,
    patch: Lin = undefined,
    pos: u64 = 0,
    blocks: [27]Block = undefined,
    mergers: [4]Merger = undefined,
    pub fn deinit(w: *Weights) void {
        w.store.deinit();
    }
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, p: *const qi.pack.Pack, up: *qi.upload.Uploader) !Weights {
        var w: Weights = .{ .store = .init(d, gpa) };
        errdefer w.deinit();
        if (!p.tensors.contains("visual.patch_embed.proj.weight")) {
            std.log.err("H3 vision checkpoint missing visual.patch_embed.proj.weight; expected visual.* Qwen3-VL tensors", .{});
            return error.MissingVisionCheckpoint;
        }
        const patch = try p.get("visual.patch_embed.proj.weight");
        if (patch.rank != 5 or patch.dim(0) != 1152 or patch.dim(1) != 3 or patch.dim(2) != 2 or patch.dim(3) != 16 or patch.dim(4) != 16) return error.BadVisionPatchShape;
        w.patch = try w.linear(io, p, up, "visual.patch_embed.proj", 1152, 1536);
        if (w.patch.quant) return error.QuantizedVisionPatchUnsupported;
        const pos = try p.get("visual.pos_embed.weight");
        if (pos.dim(0) != 2304 or pos.dim(1) != 1152) return error.BadVisionPositionShape;
        w.pos = try w.tensor(io, p, up, "visual.pos_embed.weight");
        var nb: [160]u8 = undefined;
        for (&w.blocks, 0..) |*b, i| {
            b.n1 = try w.norm(io, p, up, try std.fmt.bufPrint(&nb, "visual.blocks.{d}.norm1", .{i}), 1152);
            b.n2 = try w.norm(io, p, up, try std.fmt.bufPrint(&nb, "visual.blocks.{d}.norm2", .{i}), 1152);
            b.qkv = try w.linear(io, p, up, try std.fmt.bufPrint(&nb, "visual.blocks.{d}.attn.qkv", .{i}), 3456, 1152);
            b.proj = try w.linear(io, p, up, try std.fmt.bufPrint(&nb, "visual.blocks.{d}.attn.proj", .{i}), 1152, 1152);
            b.fc1 = try w.linear(io, p, up, try std.fmt.bufPrint(&nb, "visual.blocks.{d}.mlp.linear_fc1", .{i}), 4304, 1152);
            b.fc2 = try w.linear(io, p, up, try std.fmt.bufPrint(&nb, "visual.blocks.{d}.mlp.linear_fc2", .{i}), 1152, 4304);
        }
        for (&w.mergers, 0..) |*m, i| {
            const key = if (i == 3) "visual.merger" else try std.fmt.bufPrint(&nb, "visual.deepstack_merger_list.{d}", .{i});
            var kb: [180]u8 = undefined;
            m.norm = try w.norm(io, p, up, try std.fmt.bufPrint(&kb, "{s}.norm", .{key}), if (i == 3) 1152 else 4608);
            m.fc1 = try w.linear(io, p, up, try std.fmt.bufPrint(&kb, "{s}.linear_fc1", .{key}), 4608, 4608);
            m.fc2 = try w.linear(io, p, up, try std.fmt.bufPrint(&kb, "{s}.linear_fc2", .{key}), 5120, 4608);
        }
        try w.store.done(up);
        return w;
    }
    fn tensor(w: *Weights, io: std.Io, p: *const qi.pack.Pack, up: *qi.upload.Uploader, name: []const u8) !u64 {
        const t = p.get(name) catch |err| {
            std.log.err("H3 vision tensor missing: {s}", .{name});
            return err;
        };
        if (t.dtype == .f32) return w.store.tensor(up, p, name);
        if (t.dtype != .f16 and t.dtype != .bf16) {
            std.log.err("H3 vision unsupported dtype at {s}", .{name});
            return error.BadVisionWeight;
        }
        const raw = try w.store.gpa.alloc(u8, t.len());
        defer w.store.gpa.free(raw);
        try p.read(io, t, raw);
        const src: []align(1) const u16 = std.mem.bytesAsSlice(u16, raw);
        const out = try w.store.gpa.alloc(f32, src.len);
        defer w.store.gpa.free(out);
        for (out, src) |*v, b| v.* = if (t.dtype == .bf16) @bitCast(@as(u32, b) << 16) else @floatCast(@as(f16, @bitCast(b)));
        return w.store.upload(up, std.mem.sliceAsBytes(out));
    }
    fn norm(w: *Weights, io: std.Io, p: *const qi.pack.Pack, up: *qi.upload.Uploader, key: []const u8, d: u32) !Norm {
        var nb: [200]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "{s}.weight", .{key});
        if ((try p.get(name)).len() / (try p.get(name)).dtype.size() != d) return error.BadVisionNormShape;
        const wt = try w.tensor(io, p, up, name);
        const bias = try std.fmt.bufPrint(&nb, "{s}.bias", .{key});
        const bt = try p.get(bias);
        if (bt.len() / bt.dtype.size() != d) return error.BadVisionNormShape;
        return .{ .w = wt, .b = try w.tensor(io, p, up, bias), .dim = d };
    }
    fn linear(w: *Weights, io: std.Io, p: *const qi.pack.Pack, up: *qi.upload.Uploader, key: []const u8, n: u32, k: u32) !Lin {
        var nb: [200]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "{s}.weight", .{key});
        const t = try p.get(name);
        var l: Lin = .{ .w = 0, .n = n, .k = k };
        if (t.dtype == .u8) {
            if (t.dim(0) != n or t.dim(1) * 2 != k) return error.BadVisionLinearShape;
            l.quant = true;
            l.w = try w.store.tensor(up, p, name);
            const scale_name = try std.fmt.bufPrint(&nb, "{s}.weight_scale", .{key});
            const bs = try p.get(scale_name);
            if (bs.dtype.size() != 1 or bs.len() != @as(u64, (n + 127) / 128) * 128 * ((k / 16 + 3) / 4) * 4) return error.BadVisionQuantScaleShape;
            l.scale = try w.store.tensor(up, p, scale_name);
            l.tscale = @floatCast(try qi.weights.Store.scalar(io, p, try std.fmt.bufPrint(&nb, "{s}.weight_scale_2", .{key})));
            const pq = try std.fmt.bufPrint(&nb, "{s}.pre_quant_scale", .{key});
            if (p.tensors.contains(pq)) {
                const pt = try p.get(pq);
                if (pt.len() / pt.dtype.size() != k) return error.BadVisionPreQuantScaleShape;
                l.pqs = try w.tensor(io, p, up, pq);
            }
        } else {
            if (t.dim(0) != n or t.len() / t.dtype.size() != @as(u64, n) * k) return error.BadVisionLinearShape;
            l.w = try w.tensor(io, p, up, name);
        }
        const bias = try std.fmt.bufPrint(&nb, "{s}.bias", .{key});
        if (p.tensors.contains(bias)) {
            const bt = try p.get(bias);
            if (bt.len() / bt.dtype.size() != n) return error.BadVisionBiasShape;
            l.b = try w.tensor(io, p, up, bias);
        }
        return l;
    }
};
