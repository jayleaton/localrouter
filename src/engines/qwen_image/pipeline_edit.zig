//! Edit orchestration in capture order; weights and kernel modules live at stable addresses.
const std = @import("std");
const cuda = @import("cuda");
const pipeline = @import("pipeline.zig");
const vision = @import("vision.zig");
const vis_launch = @import("vision_launch.zig");
const enc = @import("vae_enc.zig");
const te = @import("te.zig");
const dit = @import("dit.zig");
const prep = @import("edit_prepare.zig");
const prompt = @import("edit_prompt.zig");
const upload = @import("upload.zig");
const kernels = @import("qwen_kernels");
pub const Result = struct { latents: u64, context: u64, context_rows: u32 };

pub const State = struct {
    p: *pipeline.Pipeline,
    vk: vis_launch.Ops,
    ek: enc.Ops,
    v: vision.VisionEncoder,
    e: enc.Encoder,
    refs_max: u32,
    pub fn create(p: *pipeline.Pipeline, packs: pipeline.Packs, up: *upload.Uploader, refs: u32) !*State {
        const t = try p.gpa.create(State);
        errdefer p.gpa.destroy(t);
        t.p = p;
        t.refs_max = refs;
        t.vk = try vis_launch.Ops.load(&p.d, kernels.vision);
        errdefer t.vk.unload();
        t.ek = try enc.Ops.load(&p.d, kernels.vae_enc);
        errdefer t.ek.unload();
        const a = p.arena.allocator();
        const text = try std.Io.Dir.cwd().readFileAlloc(p.io, try std.fs.path.join(a, &.{ packs.te, "manifest.json" }), a, .limited(64 << 20));
        const M = struct { vision: struct { config: vision.Config } };
        const m = try std.json.parseFromSliceLeaky(M, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        t.v = try vision.VisionEncoder.init(p.gpa, &p.d, &t.vk, &p.tk, &p.ops, p.s, &p.te_pack, up, m.vision.config, p.ref_area);
        errdefer t.v.deinit();
        if (t.v.output_dim != te.dim or t.v.cfg.deepstack_visual_indexes.len != 3) return error.UnsupportedVisionConfig;
        t.e = try enc.Encoder.init(p.gpa, p.io, &p.vae, &t.ek, &p.vae_pack, up, packs.vae);
        errdefer t.e.deinit();
        try t.v.store.done(up);
        try t.e.store.done(up);
        return t;
    }
    pub fn destroy(t: *State) void {
        t.e.deinit();
        t.v.deinit();
        t.ek.unload();
        t.vk.unload();
        t.p.gpa.destroy(t);
    }
    fn pre(t: *State, pr: ?dit.Probe, name: []const u8, ins: []const dit.Io) !void {
        _ = t;
        if (pr) |q| try q.before(q.ctx, name, ins);
    }
    fn post(t: *State, pr: ?dit.Probe, name: []const u8, outs: []const dit.Io) !void {
        _ = t;
        if (pr) |q| try q.after(q.ctx, name, outs);
    }
    fn host(t: *State, pr: ?dit.Probe, name: []const u8, dst: u64, bytes: []const u8) !void {
        try t.pre(pr, name, &.{});
        try t.p.dit.upload(dst, bytes);
        try t.post(pr, name, &.{.{ .role = "y", .ptr = dst, .bytes = bytes.len }});
    }
    pub fn generate(t: *State, text: []const u8, images: []const prep.Prepared, height: u32, width: u32, steps: u32, seed: u64, pixels: []u8, progress: ?pipeline.Progress, pr: ?dit.Probe) !Result {
        const p = t.p;
        if (images.len == 0 or images.len > t.refs_max) return error.BadReferences;
        if (height == 0 or width == 0 or height % 16 != 0 or width % 16 != 0) return error.SizeNotMultipleOf16;
        const h = height / 16;
        const w = width / 16;
        if (h > p.max_h or w > p.max_w or h * w > p.max_area or steps == 0) return error.ImageTooLarge;
        if (pixels.len != @as(usize, height) * width * p.vae.out_channels) return error.BadPixelBuffer;
        var ar: std.heap.ArenaAllocator = .init(p.gpa);
        defer ar.deinit();
        const a = ar.allocator();
        var mem: upload.Slab = .init(&p.d, p.gpa);
        defer {
            p.s.synchronize() catch {};
            mem.deinit();
        }
        t.v.probe = pr;
        p.te.probe = pr;
        p.vae.probe = pr;
        p.dit.probe = pr;
        defer {
            t.v.probe = null;
            p.te.probe = null;
            p.vae.probe = null;
            p.dit.probe = null;
        }
        const grids = try a.alloc(prompt.Grid, images.len);
        const visuals = try a.alloc(te.Visual, images.len);
        const refs = try a.alloc(dit.Reference, images.len);
        var nb: [64]u8 = undefined;
        for (images, grids, visuals, 0..) |image, *grid, *visual, j| {
            if (@max(image.height, image.width) > pipeline.edit_ref_side or (image.height / 16) * (image.width / 16) > p.ref_area) return error.ReferenceTooLarge;
            grid.* = .{ .h = image.grid[1], .w = image.grid[2] };
            const ptr = try mem.alloc(image.patches.len * 4);
            try t.host(pr, try std.fmt.bufPrint(&nb, "edit.ref{d}.patches", .{j}), ptr, std.mem.sliceAsBytes(image.patches));
            const out = try t.v.forward(ptr, .f32, &.{.{ .h = grid.h, .w = grid.w }});
            const bytes: usize = @as(usize, out.rows) * te.dim * 2;
            visual.* = .{ .start = 0, .count = out.rows, .embed = try mem.alloc(bytes), .deepstack = undefined };
            try p.ops.copyRows(p.s, visual.embed, out.merged, out.rows, te.dim * 2, te.dim * 2, te.dim * 2);
            for (&visual.deepstack, out.deepstack) |*dst, src| {
                dst.* = try mem.alloc(bytes);
                try p.ops.copyRows(p.s, dst.*, src, out.rows, te.dim * 2, te.dim * 2, te.dim * 2);
            }
        }
        const inputs = try prompt.prepare(a, &p.prompt, text, grids);
        for (visuals, inputs.spans) |*v, span| {
            v.start = span.start;
            if (v.count != span.count) return error.BadVisualSpan;
        }
        const hidden = try p.te.forwardMm(inputs.ids, inputs.positions, visuals);
        const rowptr = try mem.alloc(inputs.keep.len * 4);
        try p.dit.upload(rowptr, std.mem.sliceAsBytes(inputs.keep));
        const ctx = p.te.x;
        try t.pre(pr, "te.edit.keep", &.{ .{ .role = "x", .ptr = hidden, .bytes = inputs.ids.len * te.dim * 2 }, .{ .role = "rows", .ptr = rowptr, .bytes = inputs.keep.len * 4 } });
        // Alone replay may replace row indices as well as hidden states.
        const keep = try a.alloc(u32, inputs.keep.len);
        try p.s.synchronize();
        try p.d.check(p.d.api.cuMemcpyDtoH_v2(keep.ptr, rowptr, keep.len * 4), "cuMemcpyDtoH");
        for (keep, 0..) |row, j| {
            if (row >= inputs.ids.len) return error.BadGatherRow;
            try p.ops.copyRows(p.s, ctx + j * te.dim * 2, hidden + @as(u64, row) * te.dim * 2, 1, te.dim * 2, te.dim * 2, te.dim * 2);
        }
        const rows: u32 = @intCast(keep.len);
        try t.post(pr, "te.edit.keep", &.{.{ .role = "y", .ptr = ctx, .bytes = @as(u64, rows) * te.dim * 2 }});
        for (images, refs, inputs.slots, 0..) |image, *r, slot, j| {
            const rgba = try mem.alloc(image.rgba.len);
            try t.host(pr, try std.fmt.bufPrint(&nb, "edit.ref{d}.rgba", .{j}), rgba, image.rgba);
            r.* = .{ .h = image.height / 16, .w = image.width / 16, .slot = slot, .latent = try mem.alloc(@as(usize, image.height / 16) * (image.width / 16) * 64 * 2) };
            try t.e.encode(rgba, image.height, image.width, r.latent);
        }
        try t.pre(pr, "text_encoder", &.{});
        try t.post(pr, "text_encoder", &.{.{ .role = "context", .ptr = ctx, .bytes = @as(u64, rows) * te.dim * 2 }});
        for (refs, 0..) |r, j| {
            const name = try std.fmt.bufPrint(&nb, "reference_latents.{d}", .{j});
            try t.pre(pr, name, &.{});
            try t.post(pr, name, &.{.{ .role = "x", .ptr = r.latent, .bytes = @as(u64, r.h) * r.w * 64 * 2 }});
        }
        const sig = try p.sigmas(a, height, width, steps);
        const n: usize = @as(usize, h) * w * 64;
        const x0 = try a.alloc(f32, n);
        @import("noise.zig").fill(seed, x0);
        const bf = try a.alloc(u16, n);
        for (bf, x0) |*dst, v| {
            const b: u32 = @bitCast(v);
            dst.* = @truncate((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
        }
        var lat = try p.bufs[0].at(0);
        var nxt = try p.bufs[1].at(0);
        const vel = try p.bufs[2].at(0);
        try p.dit.upload(lat, std.mem.sliceAsBytes(bf));
        p.dit.mod_sigma = null;
        try p.dit.buildEditPrefix(ctx, rows, refs, h, w, sig[0]);
        for (0..steps) |j| {
            p.dit.probe = if (j == 0) pr else null;
            try p.dit.step(lat, sig[j], h, w, vel);
            try p.dit.euler(if (j == 0) "step0.euler" else "euler", lat, vel, nxt, n, @floatCast(@as(f64, sig[j + 1]) - @as(f64, sig[j])));
            std.mem.swap(u64, &lat, &nxt);
            if (progress) |v| v.report(v.ctx, "denoise", @intCast(j + 1), steps);
        }
        try t.pre(pr, "latents", &.{});
        try t.post(pr, "latents", &.{.{ .role = "x", .ptr = lat, .bytes = n * 2 }});
        const px = try p.bufs[3].at(0);
        try p.vae.decode(lat, h, w, px);
        try p.s.synchronize();
        try p.d.check(p.d.api.cuMemcpyDtoH_v2(pixels.ptr, px, pixels.len), "cuMemcpyDtoH");
        return .{ .latents = lat, .context = ctx, .context_rows = rows };
    }
};
