//! Host prefix plan shared by execution and the replay gate; text is causal, each reference is bidirectional.
const std = @import("std");
const rope = @import("rope.zig");
pub const Reference = struct { h: u32, w: u32, slot: u32 };
pub const Segment = struct { start: u32, end: u32, causal: bool };
pub const Piece = struct { text_start: u32, text_end: u32, ref_index: ?usize };
pub const Plan = struct { pieces: []Piece, segments: []Segment, ids: [][3]f32, next_pos: u32 };

/// The request arena owns the result; validate before any device work.
pub fn plan(a: std.mem.Allocator, text_rows: u32, refs: []const Reference, target_h: u32, target_w: u32) !Plan {
    if (refs.len == 0 or refs.len > 5 or target_h == 0 or target_w == 0) return error.BadReferences;
    var count: u64 = text_rows;
    var prev: u32 = 0;
    for (refs) |r| {
        if (r.h == 0 or r.w == 0 or r.slot < prev or r.slot > text_rows) return error.BadReferences;
        count = try std.math.add(u64, count, @as(u64, r.h) * r.w);
        prev = r.slot;
    }
    if (count > std.math.maxInt(u32)) return error.PrefixTooLong;
    const ids = try a.alloc([3]f32, @intCast(count));
    const pieces = try a.alloc(Piece, refs.len + 1);
    var segments: std.ArrayList(Segment) = .empty;
    var start: u32 = 0; var row: u32 = 0; var pos: u32 = 0;
    for (pieces, 0..) |*piece, j| {
        const end = if (j < refs.len) refs[j].slot else text_rows;
        piece.* = .{ .text_start = start, .text_end = end, .ref_index = if (j < refs.len) j else null };
        const n = end - start;
        if (n > 0) {
            rope.textIds(ids[row..][0..n], pos);
            try segments.append(a, .{ .start = row, .end = row + n, .causal = true });
            row += n; pos = try std.math.add(u32, pos, n);
        }
        if (j < refs.len) {
            const r = refs[j]; const nr = r.h * r.w;
            rope.imageIds(ids[row..][0..nr], r.h, r.w, pos, target_h, target_w);
            try segments.append(a, .{ .start = row, .end = row + nr, .causal = false });
            row += nr; pos = try std.math.add(u32, pos, @max(r.h, r.w));
        }
        start = end;
    }
    return .{ .pieces = pieces, .segments = segments.items, .ids = ids, .next_pos = pos };
}

test "duplicate insertion slots keep separate image attention boundaries and target parity" {
    var ar: std.heap.ArenaAllocator = .init(std.testing.allocator); defer ar.deinit();
    const refs = [_]Reference{ .{ .h = 2, .w = 2, .slot = 1 }, .{ .h = 1, .w = 3, .slot = 1 } };
    const p = try plan(ar.allocator(), 3, &refs, 3, 2);
    try std.testing.expectEqual(@as(u32, 8), p.next_pos);
    try std.testing.expectEqual(@as(usize, 10), p.ids.len);
    const expected = [_]Segment{ .{ .start = 0, .end = 1, .causal = true }, .{ .start = 1, .end = 5, .causal = false }, .{ .start = 5, .end = 8, .causal = false }, .{ .start = 8, .end = 10, .causal = true } };
    try std.testing.expectEqualDeep(expected[0..], p.segments);
    try std.testing.expectEqual([3]f32{ 1, -1.5, -1 }, p.ids[1]);
    try std.testing.expectEqual([3]f32{ 3, -1, -1.5 }, p.ids[5]);
    try std.testing.expectEqual([3]f32{ 6, 6, 6 }, p.ids[8]);
    try std.testing.expectError(error.BadReferences, plan(ar.allocator(), 0, &refs, 3, 2));
    const reversed = [_]Reference{ .{ .h = 2, .w = 2, .slot = 2 }, .{ .h = 2, .w = 2, .slot = 1 } };
    try std.testing.expectError(error.BadReferences, plan(ar.allocator(), 3, &reversed, 3, 2));
}
