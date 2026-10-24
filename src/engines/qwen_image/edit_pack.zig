//! The edit marker binds capability discovery to the current manifests and required weight families.
const std = @import("std");
const Pack = @import("pack.zig").Pack;
const Marker = struct { format: []const u8, refs_max: u32, manifest_sha256: struct { te: []const u8, vae: []const u8 } };

pub fn validate(a: std.mem.Allocator, io: std.Io, root: []const u8) !u32 {
    const read = struct {
        fn file(alloc: std.mem.Allocator, i: std.Io, dir: []const u8, name: []const u8) ![]u8 {
            const path = try std.fs.path.join(alloc, &.{dir, name});
            return std.Io.Dir.cwd().readFileAlloc(i, path, alloc, .limited(64 << 20));
        }
    }.file;
    const m = try std.json.parseFromSliceLeaky(Marker, a, try read(a, io, root, "edit.json"), .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, m.format, "stk-edit/1") or m.refs_max == 0 or m.refs_max > 5) return error.BadEditMarker;
    inline for (.{ "te", "vae" }) |name| {
        const dir = try std.fs.path.join(a, &.{ root, name });
        const text = try read(a, io, dir, "manifest.json");
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (!std.mem.eql(u8, &hex, @field(m.manifest_sha256, name))) return error.StaleEditMarker;
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (parsed != .object or parsed.object.get(if (std.mem.eql(u8, name, "te")) "vision" else "encoder") == null) return error.MissingEditMetadata;
        var p = try Pack.open(a, io, dir);
        defer p.close(io);
        if (std.mem.eql(u8, name, "te")) {
            inline for (.{ "vis.patch_embed.proj.weight", "vis.pos_embed.weight", "vis.inv_freq" }) |tensor| _ = try p.get(tensor);
        } else {
            inline for (.{ "vae.encoder.conv_in.weight", "vae.quant_conv.weight", "vae.enc_latents_mean", "vae.enc_latents_std" }) |tensor| _ = try p.get(tensor);
        }
    }
    return m.refs_max;
}


test "marker binds both manifests and requires encoder and vision weights" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var random: [8]u8 = undefined; io.random(&random);
    const root = try std.fmt.allocPrint(a, ".zig-cache/edit-marker-{x}", .{std.mem.readInt(u64, &random, .little)});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    var hashes: [2][64]u8 = undefined;
    inline for (.{ "te", "vae" }, 0..) |name, index| {
        const dir = try std.fs.path.join(a, &.{root, name});
        try std.Io.Dir.cwd().createDirPath(io, dir);
        const manifest = if (index == 0) "{\"format\":\"stk-pack/1\",\"precision\":\"bf16\",\"kinds\":{},\"vision\":{}}" else "{\"format\":\"stk-pack/1\",\"precision\":\"bf16\",\"kinds\":{},\"encoder\":{}}";
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{dir, "manifest.json"}), .data = manifest });
        var digest: [32]u8 = undefined; std.crypto.hash.sha2.Sha256.hash(manifest, &digest, .{});
        hashes[index] = std.fmt.bytesToHex(digest, .lower);
        const header = if (index == 0)
            "{\"vis.patch_embed.proj.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"vis.pos_embed.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]},\"vis.inv_freq\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}"
        else
            "{\"vae.encoder.conv_in.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"vae.quant_conv.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]},\"vae.enc_latents_mean\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[4,6]},\"vae.enc_latents_std\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[6,8]}}";
        const bytes = try a.alloc(u8, 8 + header.len + 8);
        std.mem.writeInt(u64, bytes[0..8], header.len, .little);
        @memcpy(bytes[8..][0..header.len], header); @memset(bytes[8 + header.len..], 0);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{dir, "weights.safetensors"}), .data = bytes });
    }
    const marker = try std.fmt.allocPrint(a, "{{\"format\":\"stk-edit/1\",\"refs_max\":5,\"manifest_sha256\":{{\"te\":\"{s}\",\"vae\":\"{s}\"}}}}", .{hashes[0], hashes[1]});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{root, "edit.json"}), .data = marker });
    try std.testing.expectEqual(@as(u32, 5), try validate(a, io, root));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{root, "vae", "manifest.json"}), .data = "{}" });
    try std.testing.expectError(error.StaleEditMarker, validate(a, io, root));
}
