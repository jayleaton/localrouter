//! The native memory cap's coverage, checked against the sources compiled into this binary (`qwen_image/memcap.zig`):
//! in LocalRouter only memcap.zig opens a driver or reaches an allocation entry point, and the pinned TensorFold CUDA
//! package reaches them only through the driver table memcap patches. These tests live in the root module so that any
//! change to a compiled source re-runs them (a module's cached test result would not see an edit elsewhere).

const std = @import("std");
/// What only this file may do in LocalRouter's sources: open a driver, or reach an allocation entry point directly.
const guarded = [_][]const u8{
    "Driver.open(",    "Driver.openPath(",        "api.cuMemAlloc", "api.cuMemHostAlloc", "api.cuMemHostRegister",
    "api.cuMemFree",   "api.cuMemHostUnregister", "cuMemCreate",    "cuMemAllocManaged",  "cuMemAllocAsync",
    "cuMemAllocPitch", "cudaMalloc",              "cudaHostAlloc",  "cudaHostRegister",
};

fn scanTree(io: std.Io, gpa: std.mem.Allocator, root: []const u8, skip: []const []const u8, comptime pats: []const []const u8, hits: *std.ArrayList(u8)) !usize {
    var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
    defer dir.close(io);
    var w = try dir.walk(gpa);
    defer w.deinit();
    var files: usize = 0;
    while (try w.next(io)) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.basename, ".zig") or listed(skip, e.path)) continue;
        const text = try e.dir.readFileAlloc(io, e.basename, gpa, .limited(16 << 20));
        defer gpa.free(text);
        files += 1;
        inline for (pats) |pat| if (std.mem.indexOf(u8, text, pat) != null) try hits.print(gpa, "{s}/{s}: {s}\n", .{ root, e.path, pat });
    }
    return files;
}

fn listed(list: []const []const u8, path: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, path)) return true;
    return false;
}

test "coverage: nothing in LocalRouter opens a driver or allocates GPU memory except through memcap.zig" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var hits: std.ArrayList(u8) = .empty;
    defer hits.deinit(gpa);
    // run from the build root (zig build test): a missing tree fails the test rather than passing it
    // Not compiled into this binary: the Kandinsky-6 video VAE qualification harness (built alone by its own
    // build.zig) and the DiT weights module (its own test module; the engine does not import it yet). Its raw calls
    // go through the driver table its caller passes, so wiring it in takes a driver from memcap.open.
    const outside = [_][]const u8{ "engines/kandinsky6/video_vae/qual.zig", "engines/kandinsky6/dit_weights.zig" };
    const n_src = try scanTree(io, gpa, "src", &([_][]const u8{ "engines/qwen_image/memcap.zig", "engine/memcap_coverage.zig" } ++ outside), &guarded, &hits);
    const n_k = try scanTree(io, gpa, "kernels", &.{}, &guarded, &hits);
    if (hits.items.len > 0) std.debug.print("unguarded allocation paths:\n{s}", .{hits.items});
    try std.testing.expect(n_src > 50 and n_k > 0);
    try std.testing.expectEqualStrings("", hits.items);
}

test "coverage: the pinned TensorFold CUDA package reaches allocation entry points only through the driver table" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var hits: std.ArrayList(u8) = .empty;
    defer hits.deinit(gpa);
    // an extern declaration or a symbol lookup outside abi.zig's table would bypass the patched entries. arena.zig is
    // the one exception, by name: its VMM sub-allocator (cuMemCreate) serves DeviceBuffer.alloc only once installed as
    // the `memory.setHook` allocator, so LocalRouter's sources must never reach it (`cuda.arena`, `cuda.pack`, setHook)
    const vmm = [_][]const u8{"arena.zig"};
    const reach = [_][]const u8{ "setHook(", "cuda.arena", "cuda.pack" };
    const bypass = [_][]const u8{ "extern \"c\" fn cuMem", "extern fn cuMem", "lookup(\"cuMem", "cuGetProcAddress", "cudaMalloc", "cuMemCreate", "cuMemAllocManaged", "cuMemAllocAsync" };
    // the package build.zig.zon pins, by its hash: a bump scans the new package, never a stale one left in zig-pkg
    const zon = try std.Io.Dir.cwd().readFileAlloc(io, "build.zig.zon", gpa, .limited(1 << 16));
    defer gpa.free(zon);
    const key = ".hash = \"";
    const at = (std.mem.indexOf(u8, zon, key) orelse return error.TestUnexpectedResult) + key.len;
    const hash = zon[at .. std.mem.indexOfScalarPos(u8, zon, at, '"') orelse return error.TestUnexpectedResult];
    try std.testing.expect(std.mem.startsWith(u8, hash, "tensorfold-"));
    const root = try std.fmt.allocPrint(gpa, "zig-pkg/{s}/zig/src/cuda", .{hash});
    defer gpa.free(root);
    const files = try scanTree(io, gpa, root, &vmm, &bypass, &hits);
    var own: std.ArrayList(u8) = .empty;
    defer own.deinit(gpa);
    _ = try scanTree(io, gpa, "src", &.{"engine/memcap_coverage.zig"}, &reach, &own);
    if (hits.items.len > 0) std.debug.print("allocation bypasses in TensorFold:\n{s}", .{hits.items});
    if (own.items.len > 0) std.debug.print("TensorFold's VMM sub-allocator reached from LocalRouter:\n{s}", .{own.items});
    try std.testing.expect(files > 10);
    try std.testing.expectEqualStrings("", hits.items);
    try std.testing.expectEqualStrings("", own.items);
}
