//! The machine-generated label every image and video LocalRouter hands out carries in its metadata (the MiniMax H3
//! licence asks for one; all generated media gets it). Metadata only: pixels and samples are never touched.
//! A PNG gets tEXt `Software` and `Comment` chunks and an XMP packet (iTXt `XML:com.adobe.xmp`) naming the IPTC digital
//! source type; an MP4 gets `comment` (`©cmt`) and `description` (`desc`) tags in `moov/udta/meta/ilst`.
//! Compiled-in engines label at the writer (`png.Options.marker`, `mp4.Options.marker`); `outputs` labels what any
//! other tool (a `cmd` adapter, an upstream) left unlabelled, after the job. No C2PA manifest: that needs a signing
//! certificate and a manifest store, and an unsigned one proves nothing.

const std = @import("std");
const Io = std.Io;
const png = @import("png.zig");
const mp4 = @import("mp4.zig");
const Request = @import("../tool/request.zig").Request;

/// Who made a file, and whether from an input image.
pub const Marker = struct {
    model: []const u8, // the tool id
    edit: bool = false, // an image edit or a video from a first frame: IPTC's composite term
};

pub const software = "LocalRouter";
/// IPTC digital source types: wholly made by a trained model, or a model's edit of an input.
pub const trained = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia";
pub const composite = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia";

/// Model ids longer than this are cut in the label.
pub const max_id = 128;
/// Sizes that always hold `comment` and `xmp` for an id of up to `max_id` bytes.
pub const comment_max = 256;
pub const xmp_max = 2048;

/// The label of a request's outputs: an edit when it brought an input image.
pub fn forRequest(model: []const u8, req: Request) Marker {
    const edit = switch (req) {
        .image => |v| v.references.len > 0,
        .video => |v| v.first_frame != null,
        .speech, .transcription, .chat, .decision => false,
    };
    return .{ .model = model, .edit = edit };
}

pub fn sourceType(m: Marker) []const u8 {
    return if (m.edit) composite else trained;
}

/// The id as printable ASCII without XML specials (anything else becomes `_`), so it is valid Latin-1, UTF-8 and XML.
fn cleanId(buf: *[max_id]u8, model: []const u8) []const u8 {
    const n = @min(model.len, max_id);
    for (buf[0..n], model[0..n]) |*o, c| o.* = if (c > ' ' and c < 0x7f and c != '<' and c != '>' and c != '&' and c != '"' and c != '\'') c else '_';
    return buf[0..n];
}

/// The human-readable label: "AI-generated (machine-generated) by model <id> with LocalRouter".
pub fn comment(buf: *[comment_max]u8, m: Marker) []const u8 {
    var id: [max_id]u8 = undefined;
    const what = if (m.edit) "AI-edited (machine-generated) from an input image" else "AI-generated (machine-generated)";
    return std.fmt.bufPrint(buf, "{s} by model {s} with " ++ software, .{ what, cleanId(&id, m.model) }) catch unreachable;
}

/// An XMP packet carrying the IPTC digital source type, the creator tool and the comment as its description.
pub fn xmp(buf: *[xmp_max]u8, m: Marker) []const u8 {
    var c: [comment_max]u8 = undefined;
    return std.fmt.bufPrint(buf,
        \\<?xpacket begin="{s}" id="W5M0MpCehiHzreSzNTczkc9d"?>
        \\<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        \\<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/" Iptc4xmpExt:DigitalSourceType="{s}" xmp:CreatorTool="{s}">
        \\<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{s}</rdf:li></rdf:Alt></dc:description>
        \\</rdf:Description></rdf:RDF></x:xmpmeta>
        \\<?xpacket end="r"?>
    , .{ "\u{FEFF}", sourceType(m), software, comment(&c, m) }) catch unreachable;
}

/// Whether metadata bytes already carry a label: they name an IPTC trained-algorithmic-media source type.
pub fn present(bytes: []const u8) bool {
    return std.ascii.findIgnoreCase(bytes, "trainedalgorithmicmedia") != null;
}

/// Labels the PNG and MP4 files a job made that are not labelled yet, recognised by their bytes, not their names:
/// a PNG gets the chunks inserted after IHDR (rewritten through a temporary file), an MP4 is remuxed by ffmpeg with
/// `-c copy`. For a compiled-in engine's outputs this only reads. A failure is logged and leaves the file as it was.
pub fn outputs(io: Io, gpa: std.mem.Allocator, dir: []const u8, files: []const []const u8, m: Marker) void {
    for (files) |name| {
        const path = std.fs.path.join(gpa, &.{ dir, name }) catch return;
        defer gpa.free(path);
        file(io, gpa, path, m) catch |err| std.log.warn("could not label {s} as machine-generated: {t}", .{ path, err });
    }
}

fn file(io: Io, gpa: std.mem.Allocator, path: []const u8, m: Marker) !void {
    var head: [8]u8 = undefined;
    {
        const f = try Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        if (try f.readPositionalAll(io, &head, 0) < head.len) return;
    }
    if (std.mem.eql(u8, &head, png.signature)) {
        const data = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
        defer gpa.free(data);
        const marked = try png.mark(gpa, data, m) orelse return;
        defer gpa.free(marked);
        const tmp = try std.mem.concat(gpa, u8, &.{ path, ".marking" });
        defer gpa.free(tmp);
        errdefer Io.Dir.cwd().deleteFile(io, tmp) catch {};
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = marked });
        try Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, io);
    } else if (std.mem.eql(u8, head[4..8], "ftyp")) {
        if (try mp4.hasMarker(io, gpa, path)) return;
        try mp4.mark(io, gpa, path, m);
    }
}

const testing = std.testing;

test "label text: id cleaned and cut, edit term, XMP names the source type" {
    var c: [comment_max]u8 = undefined;
    try testing.expectEqualStrings("AI-generated (machine-generated) by model qwen-image with LocalRouter", comment(&c, .{ .model = "qwen-image" }));
    try testing.expectEqualStrings("AI-edited (machine-generated) from an input image by model a_b_c with LocalRouter", comment(&c, .{ .model = "a<b\"c", .edit = true }));
    const long: [300]u8 = @splat('x');
    try testing.expect(comment(&c, .{ .model = &long }).len < comment_max);
    var x: [xmp_max]u8 = undefined;
    const packet = xmp(&x, .{ .model = &long, .edit = true });
    try testing.expect(std.mem.indexOf(u8, packet, "Iptc4xmpExt:DigitalSourceType=\"" ++ composite ++ "\"") != null);
    try testing.expect(present(packet));
    try testing.expect(present(xmp(&x, .{ .model = "m" })));
    try testing.expect(!present("a cat"));
}

test "outputs labels an unlabelled PNG once and leaves other files alone" {
    const io = testing.io;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    defer a.free(dir);
    // a cmd tool's PNG, without a label
    const px: [4 * 3 * 3]u8 = @splat(77);
    var aw: Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try png.encode(&aw.writer, &px, 4, 3, .rgb8, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "0.png", .data = aw.written() });
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "plain" });
    const m: Marker = .{ .model = "tool" };
    outputs(io, a, dir, &.{ "0.png", "notes.txt" }, m);
    const got = try tmp.dir.readFileAlloc(io, "0.png", a, .unlimited);
    defer a.free(got);
    var want: Io.Writer.Allocating = .init(a);
    defer want.deinit();
    try png.encode(&want.writer, &px, 4, 3, .rgb8, .{ .marker = m });
    try testing.expectEqualSlices(u8, want.written(), got); // same bytes as the writer's own label
    outputs(io, a, dir, &.{"0.png"}, m); // already labelled: unchanged
    const again = try tmp.dir.readFileAlloc(io, "0.png", a, .unlimited);
    defer a.free(again);
    try testing.expectEqualSlices(u8, got, again);
    const txt = try tmp.dir.readFileAlloc(io, "notes.txt", a, .unlimited);
    defer a.free(txt);
    try testing.expectEqualStrings("plain", txt);
}
