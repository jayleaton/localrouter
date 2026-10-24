//! Expanded visual placeholders and interleaved mrope positions, matching edit.prompt_inputs.
const std = @import("std");
const Prompt = @import("te.zig").Prompt;
pub const Grid = struct { h: u32, w: u32 };
pub const Span = struct { start: u32, count: u32 };
pub const Inputs = struct { ids: []u32, positions: [][3]f32, spans: []Span, keep: []u32, slots: []u32 };
const template_start = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n";
const template_end = "<|im_end|>\n<|im_start|>assistant\n";

/// All returned storage lives in the request arena.
pub fn prepare(a: std.mem.Allocator, p: *const Prompt, prompt: []const u8, grids: []const Grid) !Inputs {
    if (grids.len == 0 or grids.len > 5) return error.BadReferences;
    var text: std.Io.Writer.Allocating = .init(a);
    try text.writer.writeAll(template_start);
    for (grids, 0..) |_, j| {
        if (j > 0) try text.writer.writeByte(' ');
        try text.writer.print("<image{d}><|vision_start|><|image_pad|><|vision_end|>", .{j + 1});
    }
    try text.writer.writeAll(prompt);
    try text.writer.writeAll(template_end);
    const raw = try p.tok.encode(a, text.written());
    const pad_ids = try p.tok.encode(a, "<|image_pad|>");
    if (pad_ids.len != 1) return error.BadImageToken;
    const pad = pad_ids[0];
    var ids: std.ArrayList(u32) = .empty;
    const spans = try a.alloc(Span, grids.len);
    var index: usize = 0;
    for (raw) |id| {
        if (id != pad) {
            try ids.append(a, id);
            continue;
        }
        if (index >= grids.len) return error.ExtraImagePlaceholder;
        const g = grids[index];
        if (g.h == 0 or g.w == 0 or g.h % 2 != 0 or g.w % 2 != 0) return error.BadVisualGrid;
        const count = g.h * g.w / 4;
        spans[index] = .{ .start = @intCast(ids.items.len), .count = count };
        try ids.appendNTimes(a, pad, count);
        index += 1;
    }
    if (index != grids.len) return error.MissingImagePlaceholder;
    const pos_tables = try positions(a, @intCast(ids.items.len), spans, grids);
    var keep: std.ArrayList(u32) = .empty;
    const slots = try a.alloc(u32, spans.len);
    if (p.drop > ids.items.len) return error.BadTemplateDrop;
    for (spans) |span| if (span.start < p.drop) return error.BadTemplateDrop;
    var cursor = p.drop;
    for (spans, slots) |span, *slot| {
        for (cursor..span.start) |r| try keep.append(a, @intCast(r));
        slot.* = @intCast(keep.items.len);
        cursor = span.start + span.count;
    }
    for (cursor..ids.items.len) |r| try keep.append(a, @intCast(r));
    return .{ .ids = ids.items, .positions = pos_tables, .spans = spans, .keep = keep.items, .slots = slots };
}

/// Text advances once a token; merged image rows use frame/row/column positions and advance by max side.
pub fn positions(a: std.mem.Allocator, length: u32, spans: []const Span, grids: []const Grid) ![][3]f32 {
    if (spans.len != grids.len) return error.BadVisualSpan;
    const out = try a.alloc([3]f32, length);
    errdefer a.free(out);
    var cursor: u32 = 0;
    var pos: u32 = 0;
    for (spans, grids) |span, grid| {
        if (grid.h == 0 or grid.w == 0 or grid.h % 2 != 0 or grid.w % 2 != 0 or span.start < cursor or span.start > length or span.count > length - span.start or span.count != (grid.h / 2) * (grid.w / 2)) return error.BadVisualSpan;
        for (out[cursor..span.start], pos..) |*row, n| row.* = @splat(@floatFromInt(n));
        pos += span.start - cursor;
        const h = grid.h / 2;
        const w = grid.w / 2;
        for (out[span.start..][0..span.count], 0..) |*row, n| row.* = .{ @floatFromInt(pos), @floatFromInt(pos + n / w), @floatFromInt(pos + n % w) };
        pos += @max(h, w);
        cursor = span.start + span.count;
    }
    for (out[cursor..], pos..) |*row, n| row.* = @splat(@floatFromInt(n));
    return out;
}

test "multimodal mrope uses merged image grid and resumes text after maximum side" {
    const a = std.testing.allocator;
    const out = try positions(a, 10, &.{.{ .start = 2, .count = 6 }}, &.{.{ .h = 4, .w = 6 }});
    defer a.free(out);
    try std.testing.expectEqual([3]f32{ 1, 1, 1 }, out[1]);
    try std.testing.expectEqual([3]f32{ 2, 2, 4 }, out[4]);
    try std.testing.expectEqual([3]f32{ 2, 3, 2 }, out[5]);
    try std.testing.expectEqual([3]f32{ 5, 5, 5 }, out[8]);
    try std.testing.expectEqual([3]f32{ 6, 6, 6 }, out[9]);
}
