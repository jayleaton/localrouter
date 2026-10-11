//! k6v-qual: the engine side of the Kandinsky-6 video VAE qualification (kernels/cuda/kandinsky6/video_vae/
//! QUALIFICATION.md). Runs this component's kernels on the real checkpoint and the kit's inputs, and writes the named
//! checkpoints in the contract's record format (ops.jsonl, blobs/<sha>, env.json, summary.json) for compare.py.
//!
//!   k6v-qual --vae FILE --inputs DIR --conv FATBIN --ops FATBIN --out DIR [--cases s_dec,f_dec,s_enc,n_enc,n_dec]
//!            [--free-bytes N] [--sim]
//!
//! --vae is vae/diffusion_pytorch_model.safetensors. Decodes run tiles.decodePlan(.., .unbounded), or `.free_bytes = N`
//! with --free-bytes (the figure the oracle's mem_get_info saw), and env.json records the plan. --sim needs no GPU: every
//! launch goes through launch.Sim's bounds and race checks and ops.jsonl lists each checkpoint's name and shape with no
//! data (sha ""), so the kit can be smoke-tested on any host. Exit 0 when every case ran.

const std = @import("std");
const Io = std.Io;
const cuda = @import("cuda");
const launch = @import("launch.zig");
const tiles = @import("tiles.zig");
const encode = @import("encode.zig");
const decode = @import("decode.zig");
const weights = @import("weights.zig");

const View = launch.View;

const all_cases = [_][]const u8{ "s_dec", "f_dec", "s_enc", "n_enc", "n_dec" };

/// Per case, the checkpoints stored raw (null: all of them); the rest are hashed only. As oracle.py's RAW.
fn rawNames(case: []const u8) ?[]const []const u8 {
    if (std.mem.eql(u8, case, "n_enc")) return &.{ "encode.canvas", "encode.input", "quant_conv", "encode.mean", "encode.std", "encode.latent_kit" };
    if (std.mem.eql(u8, case, "n_dec")) return &.{ "decode.input", "decode.rgb" };
    return null;
}

fn hex(h: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(h, .lower);
}

fn sha(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return hex(h);
}

/// Device memory: the weights image, and one buffer per case for its io and scratch. In --sim both are fake addresses.
const Gpu = struct {
    d: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    mods: [2]cuda.Module,
    fns: [@typeInfo(launch.Kernel).@"enum".field_names.len]cuda.Function,

    fn run(ctx: *anyopaque, l: launch.Launch) anyerror!void {
        const self: *Gpu = @ptrCast(@alignCast(ctx));
        var a: cuda.launch.Args = .{};
        switch (l.call) {
            inline else => |p| a.add(p),
        }
        try cuda.launch.launch(self.fns[@backingInt(l.call)], .{ .grid = .{ .x = l.grid[0], .y = l.grid[1], .z = l.grid[2] }, .block = .{ .x = l.block[0], .y = l.block[1], .z = l.block[2] } }, self.stream, &a);
    }
};

const Buf = struct {
    dev: ?cuda.DeviceBuffer,
    base: u64,
    len: u64,

    fn alloc(gpu: ?*Gpu, len: u64, fake: u64) !Buf {
        if (gpu) |g| {
            const b = try cuda.DeviceBuffer.alloc(&g.d, len);
            return .{ .dev = b, .base = b.ptr, .len = len };
        }
        return .{ .dev = null, .base = fake, .len = len };
    }

    fn free(self: *Buf) void {
        if (self.dev) |*b| b.free();
    }

    fn upload(self: Buf, off: u64, bytes: []const u8) !void {
        if (self.dev) |b| try b.upload(off, bytes);
    }

    fn download(self: Buf, off: u64, out: []u8) !void {
        if (self.dev) |b| try b.download(off, out) else @memset(out, 0);
    }
};

/// The probe: the first occurrence of each name in a case, downloaded, hashed and written (QUALIFICATION.md).
const Recorder = struct {
    gpa: std.mem.Allocator,
    io: Io,
    out: Io.Dir,
    ops: std.ArrayList(u8) = .empty,
    seen: std.StringHashMapUnmanaged(void) = .empty,
    case: []const u8 = "",
    raw: ?[]const []const u8 = null,
    index: u64 = 0,
    gpu: ?*Gpu,
    bufs: []const *const Buf = &.{},
    host: std.ArrayList(u8) = .empty,

    fn probe(self: *Recorder) launch.Probe {
        return .{ .ctx = self, .func = mark };
    }

    fn begin(self: *Recorder, case: []const u8, bufs: []const *const Buf) void {
        self.case = case;
        self.raw = rawNames(case);
        self.bufs = bufs;
        var it = self.seen.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.seen.clearRetainingCapacity();
    }

    fn deinit(self: *Recorder) void {
        var it = self.seen.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.seen.deinit(self.gpa);
        self.ops.deinit(self.gpa);
        self.host.deinit(self.gpa);
    }

    fn mark(ctx: *anyopaque, name: []const u8, v: View, tokens: bool) anyerror!void {
        const self: *Recorder = @ptrCast(@alignCast(ctx));
        if (self.seen.contains(name)) return;
        if (self.gpu) |g| try g.stream.synchronize();
        const elems: u64 = @as(u64, v.t) * v.h * v.w * v.c;
        try self.host.resize(self.gpa, elems * 2);
        const buf = self.find(v.ptr) orelse return error.ProbeOutsideBuffers;
        try gather(buf, v, self.host.items);
        const shape: [4]u64 = .{ v.t, v.h, v.w, v.c };
        const tok: [2]u64 = .{ @as(u64, v.t) * v.h * v.w, v.c };
        try self.put(name, "f16", if (tokens) &tok else &shape, self.host.items);
    }

    fn find(self: *Recorder, ptr: u64) ?*const Buf {
        for (self.bufs) |b| if (ptr >= b.base and ptr < b.base + b.len) return b;
        return null;
    }

    /// Records `bytes` (contiguous little-endian) under `name` unless the name was already recorded in this case.
    fn put(self: *Recorder, name: []const u8, dtype: []const u8, shape: []const u64, bytes: []const u8) !void {
        if (self.seen.contains(name)) return;
        try self.seen.put(self.gpa, try self.gpa.dupe(u8, name), {});
        const sim = self.gpu == null;
        const digest: [64]u8 = if (sim) @splat(' ') else sha(bytes);
        const raw = !sim and (if (self.raw) |list| for (list) |n| {
            if (std.mem.eql(u8, n, name)) break true;
        } else false else true);
        if (raw) {
            const path = try std.fmt.allocPrint(self.gpa, "blobs/{s}", .{&digest});
            defer self.gpa.free(path);
            try self.out.writeFile(self.io, .{ .sub_path = path, .data = bytes });
        }
        var w: Io.Writer.Allocating = .fromArrayList(self.gpa, &self.ops);
        try w.writer.print("{{\"i\": {d}, \"case\": \"{s}\", \"name\": \"{s}\", \"dtype\": \"{s}\", \"shape\": [", .{ self.index, self.case, name, dtype });
        for (shape, 0..) |s, i| try w.writer.print("{s}{d}", .{ if (i > 0) ", " else "", s });
        try w.writer.print("], \"sha\": \"{s}\", \"raw\": {}}}\n", .{ if (sim) "" else &digest, raw });
        self.ops = w.toArrayList();
        self.index += 1;
        try self.out.writeFile(self.io, .{ .sub_path = "ops.jsonl", .data = self.ops.items });
    }
};

/// Copies the fp16 view `v` into `out` densely ([t][h][w][c]): one copy when dense, else a row or a pixel at a time.
fn gather(buf: *const Buf, v: View, out: []u8) !void {
    const off = v.ptr - buf.base;
    const row: u64 = @as(u64, v.w) * v.c;
    if (v.sw == v.c and v.sh == row and v.st == row * v.h) return buf.download(off, out);
    var o: u64 = 0;
    for (0..v.t) |t| for (0..v.h) |y| {
        const at = off + (t * v.st + y * v.sh) * 2;
        if (v.sw == v.c) {
            try buf.download(at, out[o..][0 .. row * 2]);
            o += row * 2;
        } else for (0..v.w) |x| {
            try buf.download(at + x * v.sw * 2, out[o..][0 .. @as(u64, v.c) * 2]);
            o += @as(u64, v.c) * 2;
        }
    };
}

const Input = struct { bytes: []u8, dtype: []const u8, shape: []const u64 };

fn loadInput(a: std.mem.Allocator, io: Io, dir: Io.Dir, index: std.json.Value, name: []const u8) !Input {
    const e = (index.object.get(name) orelse return error.MissingInput).object;
    const bytes = try dir.readFileAlloc(io, e.get("file").?.string, a, .limited(1 << 31));
    if (!std.mem.eql(u8, &sha(bytes), e.get("sha256").?.string)) return error.InputHashMismatch;
    const sh = e.get("shape").?.array.items;
    const shape = try a.alloc(u64, sh.len);
    for (shape, sh) |*d, s| d.* = @intCast(s.integer);
    return .{ .bytes = bytes, .dtype = e.get("dtype").?.string, .shape = shape };
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    var vae_path: ?[]const u8 = null;
    var inputs_path: ?[]const u8 = null;
    var conv_path: ?[]const u8 = null;
    var ops_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var cases_arg: []const u8 = "s_dec,f_dec,s_enc,n_enc,n_dec";
    var policy: tiles.MemPolicy = .unbounded;
    var sim = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--sim")) {
            sim = true;
            continue;
        }
        if (i + 1 >= args.len) return usage();
        i += 1;
        const v = args[i];
        if (std.mem.eql(u8, k, "--vae")) vae_path = v else if (std.mem.eql(u8, k, "--inputs")) inputs_path = v else if (std.mem.eql(u8, k, "--conv")) conv_path = v else if (std.mem.eql(u8, k, "--ops")) ops_path = v else if (std.mem.eql(u8, k, "--out")) out_path = v else if (std.mem.eql(u8, k, "--cases")) cases_arg = v else if (std.mem.eql(u8, k, "--free-bytes")) policy = .{ .free_bytes = try std.fmt.parseInt(u64, v, 10) } else return usage();
    }
    if (vae_path == null or inputs_path == null or out_path == null or (!sim and (conv_path == null or ops_path == null))) return usage();

    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, out_path.?);
    var out = try cwd.openDir(io, out_path.?, .{});
    defer out.close(io);
    try out.createDirPath(io, "blobs");
    var env: Io.Writer.Allocating = .init(a);
    var summary: Io.Writer.Allocating = .init(a);
    try env.writer.print("{{\"engine\": true, \"sim\": {}", .{sim});

    // the checkpoint: header hash, every tensor bound, one fp16 image
    const t_load = Io.Clock.awake.now(io);
    const file = try cwd.readFileAllocOptions(io, vae_path.?, gpa, .limited(2 << 30), .of(u64), null);
    defer gpa.free(file);
    var header = try weights.parseHeader(gpa, file);
    defer header.deinit();
    var image = try weights.build(gpa, .{ .header = &header, .payload = file[header.data_start..] });
    defer image.deinit(gpa);
    try env.writer.print(", \"checkpoint\": {{\"bytes\": {d}, \"header_sha256\": \"{s}\", \"tensors\": {d}, \"image_bytes\": {d}}}", .{ file.len, &sha(file[8..header.data_start]), header.entries.len, image.bytes.len });

    var gpu_store: Gpu = undefined;
    var gpu: ?*Gpu = null;
    defer if (gpu) |g| {
        for (&g.mods) |*m| m.unload();
        g.stream.deinit();
        g.ctx.deinit();
        g.d.close();
    };
    if (!sim) {
        gpu_store.d = try cuda.Driver.open();
        gpu_store.ctx = try cuda.Context.init(&gpu_store.d, 0);
        gpu_store.stream = try cuda.Stream.init(&gpu_store.d, false);
        gpu = &gpu_store;
        const g = gpu.?;
        var names: [2][]const u8 = undefined;
        for ([_][]const u8{ conv_path.?, ops_path.? }, 0..) |p, m| {
            const img = try cwd.readFileAllocOptions(io, p, a, .limited(256 << 20), .of(u64), null);
            g.mods[m] = try cuda.Module.load(&g.d, img);
            names[m] = try a.dupe(u8, &sha(img));
        }
        for (&g.fns, 0..) |*f, k| {
            const kk: launch.Kernel = @fromBackingInt(@intCast(k));
            f.* = try g.mods[@backingInt(launch.image(kk))].function(launch.symbol(kk));
        }
        var nb: [256]u8 = undefined;
        const mem = try g.ctx.memInfo();
        try env.writer.print(", \"gpu\": {{\"name\": \"{s}\", \"capability\": {d}, \"driver\": {d}, \"mem_free\": {d}, \"mem_total\": {d}}}, \"conv_fatbin_sha256\": \"{s}\", \"ops_fatbin_sha256\": \"{s}\"", .{ try g.ctx.name(&nb), try g.ctx.capability(), try g.d.version(), mem.free, mem.total, names[0], names[1] });
    }
    const self_exe = cwd.readFileAlloc(io, "/proc/self/exe", a, .limited(256 << 20)) catch null;
    if (self_exe) |b| try env.writer.print(", \"binary_sha256\": \"{s}\"", .{&sha(b)});

    var wbuf = try Buf.alloc(gpu, image.bytes.len, 0x1000_0000_0000);
    defer wbuf.free();
    try wbuf.upload(0, image.bytes);
    var enc = image.enc;
    var dec = image.dec;
    weights.relocateEncoder(&enc, wbuf.base);
    weights.relocateDecoder(&dec, wbuf.base);
    try env.writer.print(", \"load_ms\": {d}", .{t_load.durationTo(Io.Clock.awake.now(io)).toMilliseconds()});

    var inputs = try cwd.openDir(io, inputs_path.?, .{});
    defer inputs.close(io);
    const index = try std.json.parseFromSliceLeaky(std.json.Value, a, try inputs.readFileAlloc(io, "inputs.json", a, .limited(1 << 20)), .{});

    var rec: Recorder = .{ .gpa = gpa, .io = io, .out = out, .gpu = gpu };
    defer rec.deinit();
    var simr: launch.Sim = .{};
    var l: launch.Launcher = if (gpu) |g| .{ .ctx = g, .func = Gpu.run } else simr.launcher();
    l.probe = rec.probe();

    try env.writer.writeAll(", \"cases\": {");
    try summary.writer.print("{{\"engine\": true, \"sim\": {}, \"cases\": {{", .{sim});
    var all_ok = true;
    var it = std.mem.splitScalar(u8, cases_arg, ',');
    var first = true;
    while (it.next()) |case| {
        if (!for (all_cases) |c| {
            if (std.mem.eql(u8, c, case)) break true;
        } else false) return usage();
        const t0 = Io.Clock.awake.now(io);
        var info: Io.Writer.Allocating = .init(a);
        const r = runCase(a, io, gpu, &simr, l, &rec, case, inputs, index, &enc, &dec, &wbuf, policy, &info.writer);
        const ms = t0.durationTo(Io.Clock.awake.now(io)).toMilliseconds();
        const sep = if (first) "" else ", ";
        first = false;
        if (r) |peak| {
            try summary.writer.print("{s}\"{s}\": {{\"status\": \"ok\", \"wall_ms\": {d}, \"device_bytes\": {d}, \"checkpoints\": {d}}}", .{ sep, case, ms, peak + image.bytes.len, rec.seen.count() });
        } else |err| {
            all_ok = false;
            try summary.writer.print("{s}\"{s}\": {{\"status\": \"error\", \"error\": \"{s}\", \"wall_ms\": {d}}}", .{ sep, case, @errorName(err), ms });
        }
        try env.writer.print("{s}\"{s}\": {{{s}}}", .{ sep, case, info.written() });
        try out.writeFile(io, .{ .sub_path = "summary.json", .data = try std.fmt.allocPrint(a, "{s}}}}}\n", .{summary.written()}) });
    }
    try env.writer.writeAll("}}\n");
    try out.writeFile(io, .{ .sub_path = "env.json", .data = env.written() });
    try Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(a, "{s}}}}}\n", .{summary.written()}));
    return if (all_ok) 0 else 1;
}

/// One case: inputs uploaded, the graph run under the probe, the io boundaries recorded. Returns the case's device bytes.
fn runCase(a: std.mem.Allocator, io: Io, gpu: ?*Gpu, simr: *launch.Sim, l: launch.Launcher, rec: *Recorder, case: []const u8, inputs: Io.Dir, index: std.json.Value, enc: *const encode.Encoder, dec: *const decode.Decoder, wbuf: *const Buf, policy: tiles.MemPolicy, info: *Io.Writer) !u64 {
    simr.* = .{};
    simr.region(wbuf.base, wbuf.len, "weights");
    if (std.mem.endsWith(u8, case, "_dec")) {
        const z = try loadInput(a, io, inputs, index, try std.fmt.allocPrint(a, "{s}.latent", .{case}));
        if (z.shape.len != 4 or z.shape[3] != tiles.latent_channels) return error.BadLatentShape;
        const dtype: decode.LatentDtype = if (std.mem.eql(u8, z.dtype, "f32")) .f32 else if (std.mem.eql(u8, z.dtype, "bf16")) .bf16 else return error.BadLatentDtype;
        const plan = try tiles.decodePlan(@intCast(z.shape[0]), @intCast(z.shape[1]), @intCast(z.shape[2]), policy);
        const g = plan.grid;
        const t = plan.tiling;
        try info.print("\"latent_dtype\": \"{s}\", \"policy\": \"{s}\", \"plan\": {{\"tile_size\": [1, {d}, {d}, {d}], \"tile_stride\": [{d}, {d}, {d}], \"temporal_tiles\": {d}, \"spatial_tiles\": {d}, \"tiled\": {}}}", .{ @tagName(dtype), @tagName(policy), t.ft, t.ht, t.wt, t.fs, t.hs, t.ws, plan.tileCount(), g.count(), g.tiled() });
        const rgb_bytes: u64 = @as(u64, plan.sampleFramesTotal()) * g.sample_h * g.sample_w * 3;
        const lat_at: u64 = 0;
        const rgb_at = std.mem.alignForward(u64, z.bytes.len, 256);
        const scratch_at = std.mem.alignForward(u64, rgb_at + rgb_bytes, 256);
        const scratch = decode.scratchBytes(plan);
        var buf = try Buf.alloc(gpu, scratch_at + scratch, 0x2000_0000_0000);
        defer buf.free();
        simr.region(buf.base, buf.len, "case");
        rec.begin(case, &.{ &buf, wbuf });
        try buf.upload(lat_at, z.bytes);
        try decode.decode(l, dec, plan, .{ .latent = buf.base + lat_at, .latent_dtype = dtype, .rgb = buf.base + rgb_at, .scratch = buf.base + scratch_at, .scratch_bytes = scratch });
        if (gpu) |gg| try gg.stream.synchronize();
        const rgb = try a.alloc(u8, rgb_bytes);
        try buf.download(rgb_at, rgb);
        try rec.put("decode.rgb", "u8", &.{ plan.sampleFramesTotal(), g.sample_h, g.sample_w, 3 }, rgb);
        return buf.len;
    }
    const img = try loadInput(a, io, inputs, index, try std.fmt.allocPrint(a, "{s}.image", .{case}));
    const eps = try loadInput(a, io, inputs, index, try std.fmt.allocPrint(a, "{s}.eps", .{case}));
    if (img.shape.len != 3 or img.shape[2] != 3) return error.BadImageShape;
    const h: u32 = @intCast(img.shape[0]);
    const w: u32 = @intCast(img.shape[1]);
    const ls = encode.latentShape(1, h, w);
    const lat: u64 = @as(u64, ls.h) * ls.w * tiles.latent_channels * 2;
    if (eps.bytes.len != lat) return error.BadEpsShape;
    try info.print("\"canvas\": [{d}, {d}], \"latent\": [{d}, {d}, {d}]", .{ h, w, ls.t, ls.h, ls.w });
    const eps_at = std.mem.alignForward(u64, img.bytes.len, 256);
    const outs_at = std.mem.alignForward(u64, eps_at + lat, 256);
    const scratch_at = std.mem.alignForward(u64, outs_at + 3 * lat, 256);
    const scratch = encode.scratchBytes(1, h, w);
    var buf = try Buf.alloc(gpu, scratch_at + scratch, 0x2000_0000_0000);
    defer buf.free();
    simr.region(buf.base, buf.len, "case");
    rec.begin(case, &.{ &buf, wbuf });
    try buf.upload(0, img.bytes);
    try buf.upload(eps_at, eps.bytes);
    try rec.put("encode.canvas", "u8", &.{ h, w, 3 }, img.bytes);
    const mean = buf.base + outs_at;
    try encode.encode(l, enc, .{ .rgb = buf.base, .frames = 1, .h = h, .w = w, .eps = buf.base + eps_at, .mean = mean, .stdv = mean + lat, .latent = mean + 2 * lat, .scratch = buf.base + scratch_at, .scratch_bytes = scratch });
    if (gpu) |g| try g.stream.synchronize();
    const host = try a.alloc(u8, lat);
    const shape = [_]u64{ 1, ls.h, ls.w, tiles.latent_channels };
    for ([_][]const u8{ "encode.mean", "encode.std", "encode.latent_kit" }, 0..) |name, k| {
        try buf.download(outs_at + k * lat, host);
        try rec.put(name, "f16", &shape, host);
    }
    return buf.len;
}

fn usage() u8 {
    std.debug.print("usage: k6v-qual --vae FILE --inputs DIR --conv FATBIN --ops FATBIN --out DIR [--cases s_dec,f_dec,s_enc,n_enc,n_dec] [--free-bytes N] [--sim]\n", .{});
    return 2;
}
