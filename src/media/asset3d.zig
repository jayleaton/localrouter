//! What an image_to_3d tool hands back, checked before its job completes: `asset.json` (`request.Manifest`) names
//! the asset and the input view among the files the tool reported, and the asset is a whole file of its format (a
//! PLY: its header parses and the body is as long as the header says). A tool that wrote anything else fails its job,
//! so a client never gets a manifest pointing at nothing or a truncated splat.

const std = @import("std");
const Io = std.Io;
const request = @import("../tool/request.zig");
const ply = @import("ply.zig");

pub const manifest_name = "asset.json";

/// The checked manifest of a finished job in `dir` (which reported `files`), parsed in `a`; null with `why` set.
pub fn check(io: Io, a: std.mem.Allocator, dir: []const u8, files: []const []const u8, req: request.Asset3d, why: *[]const u8) ?request.Manifest {
    if (!listed(files, manifest_name)) return fail(why, "the model wrote no asset.json manifest");
    const text = Io.Dir.cwd().readFileAlloc(io, std.fs.path.join(a, &.{ dir, manifest_name }) catch return fail(why, "out of memory"), a, .limited(1 << 20)) catch
        return fail(why, "the model's asset.json could not be read");
    const m = std.json.parseFromSliceLeaky(request.Manifest, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch
        return fail(why, "the model's asset.json is not a manifest: it needs object, version, representation, format, file and input");
    if (!eql(m.object, "asset3d") or m.version != 1) return fail(why, "the model's asset.json is not an asset3d manifest of version 1");
    if (!eql(m.format, @tagName(req.format))) return fail(why, "the model wrote another format than the one asked for");
    if (!listed(files, m.file) or !std.mem.endsWith(u8, m.file, req.format.extension())) return fail(why, "the model's asset.json names an asset file it did not write");
    if (!listed(files, m.input)) return fail(why, "the model's asset.json names an input view it did not write");
    switch (req.format) {
        .ply => {
            const path = std.fs.path.join(a, &.{ dir, m.file }) catch return fail(why, "out of memory");
            const f = Io.Dir.cwd().openFile(io, path, .{}) catch return fail(why, "the model's asset file could not be opened");
            defer f.close(io);
            const head = a.alloc(u8, ply.max_header) catch return fail(why, "out of memory");
            const n = f.readPositionalAll(io, head, 0) catch return fail(why, "the model's asset file could not be read");
            const h = ply.header(head[0..n]) orelse return fail(why, "the model's asset file is not a PLY");
            if (h.count == 0) return fail(why, "the model's PLY holds nothing");
            const size = (f.stat(io) catch return fail(why, "the model's asset file could not be read")).size;
            if (h.bodyBytes()) |body| if (size != h.len + body) return fail(why, "the model's PLY is not as long as its header says");
        },
    }
    return m;
}

fn listed(files: []const []const u8, name: []const u8) bool {
    for (files) |f| if (eql(f, name)) return true;
    return false;
}

fn fail(why: *[]const u8, msg: []const u8) ?request.Manifest {
    why.* = msg;
    return null;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

test "a manifest must name the files the tool wrote, and the PLY must be whole" {
    const t = std.testing;
    const io = t.io;
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    const req: request.Asset3d = .{ .image = "image_0.png" };
    const good = "{\"object\":\"asset3d\",\"version\":1,\"representation\":\"gaussian_splat\",\"format\":\"ply\",\"file\":\"asset.ply\",\"input\":\"input.png\",\"camera\":{}}";
    const splat = try ply.testSplat(a, 4);
    try tmp.dir.writeFile(io, .{ .sub_path = "asset.ply", .data = splat });
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = "png" });
    try tmp.dir.writeFile(io, .{ .sub_path = "asset.json", .data = good });
    const all = [_][]const u8{ "asset.ply", "asset.json", "input.png" };
    var why: []const u8 = "";
    const m = check(io, a, dir, &all, req, &why).?;
    try t.expectEqualStrings("gaussian_splat", m.representation);
    try t.expect(check(io, a, dir, &.{ "asset.ply", "input.png" }, req, &why) == null);
    try t.expectEqualStrings("the model wrote no asset.json manifest", why);
    try t.expect(check(io, a, dir, &.{ "asset.json", "input.png" }, req, &why) == null);
    try t.expectEqualStrings("the model's asset.json names an asset file it did not write", why);
    try tmp.dir.writeFile(io, .{ .sub_path = "asset.ply", .data = splat[0 .. splat.len - 1] });
    try t.expect(check(io, a, dir, &all, req, &why) == null);
    try t.expectEqualStrings("the model's PLY is not as long as its header says", why);
    try tmp.dir.writeFile(io, .{ .sub_path = "asset.json", .data = "{\"object\":\"asset3d\",\"version\":2}" });
    try t.expect(check(io, a, dir, &all, req, &why) == null);
}
