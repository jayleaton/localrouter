//! Integration tests of the router features, on real daemons and workers: priorities and keep_loaded, image inputs
//! (edits and image to video over multipart, JSON and MCP), capability refusals, and the stdio MCP bridge.

const std = @import("std");
const Io = std.Io;
const opts = @import("build_options");
const testing = std.testing;
const harness = @import("harness.zig");

const Daemon = harness.Daemon;
const reserve = harness.reserve;
const mcpCall = harness.mcpCall;
const mcpText = harness.mcpText;
const firstPng = harness.firstPng;
const sleepMs = harness.sleepMs;

// Not real pictures: the daemon only checks the magic bytes, and the testpattern engine hashes the contents.
const png_a = "\x89PNG\r\n\x1a\nimage A";
const png_b = "\x89PNG\r\n\x1a\nimage B";

const boundary = "stkboundary";
const form_type = "multipart/form-data; boundary=" ++ boundary;
const Field = struct { name: []const u8, value: []const u8 };

/// A multipart body: text `fields`, then `uploads` as (part name, file bytes).
fn form(a: std.mem.Allocator, fields: []const Field, uploads: []const Field) ![]u8 {
    var w: Io.Writer.Allocating = .init(a);
    for (fields) |f| try w.writer.print("--{s}\r\nContent-Disposition: form-data; name=\"{s}\"\r\n\r\n{s}\r\n", .{ boundary, f.name, f.value });
    for (uploads) |u| {
        try w.writer.print("--{s}\r\nContent-Disposition: form-data; name=\"{s}\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n", .{ boundary, u.name });
        try w.writer.writeAll(u.value);
        try w.writer.writeAll("\r\n");
    }
    try w.writer.print("--{s}--\r\n", .{boundary});
    return w.written();
}

fn b64(a: std.mem.Allocator, data: []const u8) ![]const u8 {
    const enc = std.base64.standard.Encoder;
    return enc.encode(try a.alloc(u8, enc.calcSize(data.len)), data);
}

fn decode(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const dec = std.base64.standard.Decoder;
    const out = try a.alloc(u8, try dec.calcSizeForSlice(text));
    try dec.decode(out, text);
    return out;
}

/// POST /v1/images/edits as multipart.
fn edit(d: *Daemon, a: std.mem.Allocator, fields: []const Field, uploads: []const Field) !Daemon.Resp {
    return d.callAs(a, .POST, "/v1/images/edits", try form(a, fields, uploads), form_type);
}

/// The MP4 of a video job once it completes.
fn videoContent(d: *Daemon, a: std.mem.Allocator, create: Daemon.Resp) ![]u8 {
    try testing.expectEqual(std.http.Status.ok, create.status);
    const id = (try std.json.parseFromSliceLeaky(struct { id: []const u8 }, a, create.body, .{ .ignore_unknown_fields = true })).id;
    for (0..300) |_| {
        const st = try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{id}), null);
        const s = (try std.json.parseFromSliceLeaky(struct { status: []const u8 }, a, st.body, .{ .ignore_unknown_fields = true })).status;
        if (std.mem.eql(u8, s, "completed")) return (try d.call(a, .GET, try std.fmt.allocPrint(a, "/v1/videos/{s}/content", .{id}), null)).body;
        try testing.expect(!std.mem.eql(u8, s, "failed"));
        sleepMs(d.io, 100);
    }
    return error.Timeout;
}

fn expectRefusal(r: Daemon.Resp, needles: []const []const u8) !void {
    try testing.expectEqual(std.http.Status.bad_request, r.status);
    for (needles) |n| if (std.mem.indexOf(u8, r.body, n) == null) {
        std.debug.print("missing '{s}' in {s}\n", .{ n, r.body });
        return error.TestExpectedEqual;
    };
}

test "image inputs: edits over multipart, JSON and MCP; image to video; capability refusals" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // `plain` comes first, so a request that needs image_edit without naming a model must skip it.
    var d = try Daemon.start(testing.allocator, io, "inputs", "{" ++ reserve ++ ", \"tools\": [" ++
        "{\"id\": \"plain\", \"kind\": \"image\", \"engine\": \"testpattern\", \"capabilities\": [\"text_to_image\"]}," ++
        "{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\"}," ++
        "{\"id\": \"plainv\", \"kind\": \"video\", \"engine\": \"testpattern\", \"capabilities\": [\"text_to_video\"]}," ++
        "{\"id\": \"tpv\", \"kind\": \"video\", \"engine\": \"testpattern\"}]}");
    defer d.stop();

    // Multipart edit: the input reaches the engine (it changes the picture, deterministically).
    const base = [_]Field{ .{ .name = "model", .value = "tp" }, .{ .name = "prompt", .value = "bars" }, .{ .name = "size", .value = "256x256" }, .{ .name = "seed", .value = "7" } };
    const r_a = try edit(&d, a, &base, &.{.{ .name = "image[]", .value = png_a }});
    try testing.expectEqual(std.http.Status.ok, r_a.status);
    const out_a = try firstPng(a, r_a.body);
    try testing.expect(!std.mem.eql(u8, out_a, try firstPng(a, (try harness.image(&d, a, "tp", ",\"seed\":7")).body)));
    try testing.expectEqualSlices(u8, out_a, try firstPng(a, (try edit(&d, a, &base, &.{.{ .name = "image", .value = png_a }})).body));
    try testing.expect(!std.mem.eql(u8, out_a, try firstPng(a, (try edit(&d, a, &base, &.{.{ .name = "image[]", .value = png_b }})).body)));

    // JSON edits: base64, a data URL and a list give the same picture as the multipart file.
    const jb = try std.fmt.allocPrint(a, "{{\"model\":\"tp\",\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"image\":\"{s}\"}}", .{try b64(a, png_a)});
    try testing.expectEqualSlices(u8, out_a, try firstPng(a, (try d.call(a, .POST, "/v1/images/edits", jb)).body));
    const jd = try std.fmt.allocPrint(a, "{{\"model\":\"tp\",\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"images\":[\"data:image/png;base64,{s}\"]}}", .{try b64(a, png_a)});
    try testing.expectEqualSlices(u8, out_a, try firstPng(a, (try d.call(a, .POST, "/v1/images/edits", jd)).body));

    // Refusals: a model without image_edit (named), a wrong format, too many, none, a bad scheme, too large.
    const plain = [_]Field{ .{ .name = "model", .value = "plain" }, .{ .name = "prompt", .value = "bars" }, .{ .name = "size", .value = "256x256" } };
    try expectRefusal(try edit(&d, a, &plain, &.{.{ .name = "image[]", .value = png_a }}), &.{ "plain", "image_edit" });
    try expectRefusal(try edit(&d, a, &base, &.{.{ .name = "image[]", .value = "GIF89a..." }}), &.{"PNG, JPEG or WebP"});
    const six: [6]Field = @splat(.{ .name = "image[]", .value = png_a });
    try expectRefusal(try edit(&d, a, &base, &six), &.{"too many"});
    try expectRefusal(try edit(&d, a, &base, &.{}), &.{"image is required"});
    try expectRefusal(try d.call(a, .POST, "/v1/images/edits", "{\"model\":\"tp\",\"prompt\":\"x\",\"image\":\"file:///etc/passwd\"}"), &.{"http"});
    const big = try a.alloc(u8, 33 << 20);
    @memset(big, 0);
    @memcpy(big[0..8], "\x89PNG\r\n\x1a\n");
    try expectRefusal(try edit(&d, a, &base, &.{.{ .name = "image[]", .value = big }}), &.{"32 MB"});

    // Image to video: multipart and JSON; the first frame changes the clip; a model without the capability refuses.
    const vf = [_]Field{ .{ .name = "model", .value = "tpv" }, .{ .name = "prompt", .value = "waves" }, .{ .name = "size", .value = "256x256" }, .{ .name = "seconds", .value = "1" }, .{ .name = "audio", .value = "false" }, .{ .name = "seed", .value = "3" } };
    const vid_a = try videoContent(&d, a, try d.callAs(a, .POST, "/v1/videos", try form(a, &vf, &.{.{ .name = "input_reference", .value = png_a }}), form_type));
    try testing.expectEqualSlices(u8, "ftyp", vid_a[4..8]);
    const vid_b = try videoContent(&d, a, try d.callAs(a, .POST, "/v1/videos", try form(a, &vf, &.{.{ .name = "input_reference", .value = png_b }}), form_type));
    try testing.expect(!std.mem.eql(u8, vid_a, vid_b));
    const vj = try std.fmt.allocPrint(a, "{{\"model\":\"tpv\",\"prompt\":\"waves\",\"size\":\"256x256\",\"seconds\":\"1\",\"audio\":false,\"input_reference\":\"{s}\"}}", .{try b64(a, png_a)});
    try testing.expect((try videoContent(&d, a, try d.call(a, .POST, "/v1/videos", vj))).len > 1000);
    var pv = vf;
    pv[0].value = "plainv";
    try expectRefusal(try d.callAs(a, .POST, "/v1/videos", try form(a, &pv, &.{.{ .name = "input_reference", .value = png_a }}), form_type), &.{ "plainv", "image_to_video" });

    // MCP: an edit with base64 equals the HTTP edit and picks `tp` unasked; a URL input works; refusals are tool errors.
    const args = try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"images\":[\"{s}\"]}}", .{try b64(a, png_a)});
    const m = try mcpCall(&d, a, "generate_image", args);
    try testing.expect(!m.object.get("isError").?.bool);
    try testing.expectEqualSlices(u8, out_a, try decode(a, m.object.get("content").?.array.items[0].object.get("data").?.string));
    const info = try mcpText(a, m);
    try testing.expectEqualStrings("tp", info.object.get("model").?.string);
    const url = info.object.get("files").?.array.items[0].string;
    const fetched = (try d.call(a, .GET, url[std.mem.indexOf(u8, url, "/v1/").?..], null)).body;
    const by_url = try mcpCall(&d, a, "generate_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"images\":[\"{s}\"],\"inline_images\":false}}", .{url}));
    const by_b64 = try mcpCall(&d, a, "generate_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"images\":[\"{s}\"],\"inline_images\":false}}", .{try b64(a, fetched)}));
    try testing.expect(!by_url.object.get("isError").?.bool);
    const f_url = (try mcpText(a, by_url)).object.get("files").?.array.items[0].string;
    const f_b64 = (try mcpText(a, by_b64)).object.get("files").?.array.items[0].string;
    try testing.expectEqualSlices(u8, (try d.call(a, .GET, f_b64[std.mem.indexOf(u8, f_b64, "/v1/").?..], null)).body, (try d.call(a, .GET, f_url[std.mem.indexOf(u8, f_url, "/v1/").?..], null)).body);
    try testing.expect((try mcpCall(&d, a, "generate_image", "{\"prompt\":\"x\",\"images\":[\"file:///etc/passwd\"]}")).object.get("isError").?.bool);
    const refused = try mcpCall(&d, a, "generate_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"x\",\"model\":\"plain\",\"images\":[\"{s}\"]}}", .{try b64(a, png_a)}));
    try testing.expect(refused.object.get("isError").?.bool);
    try testing.expect(std.mem.indexOf(u8, refused.object.get("content").?.array.items[0].object.get("text").?.string, "image_edit") != null);

    // MCP image to video, and its refusal; list_models shows capabilities.
    const v = try mcpText(a, try mcpCall(&d, a, "generate_video", try std.fmt.allocPrint(a, "{{\"prompt\":\"waves\",\"size\":\"256x256\",\"seconds\":1,\"audio\":false,\"wait_s\":60,\"image\":\"{s}\"}}", .{try b64(a, png_a)})));
    try testing.expectEqualStrings("completed", v.object.get("status").?.string);
    try testing.expectEqualStrings("tpv", v.object.get("model").?.string);
    const pvr = try mcpCall(&d, a, "generate_video", try std.fmt.allocPrint(a, "{{\"prompt\":\"w\",\"model\":\"plainv\",\"image\":\"{s}\"}}", .{try b64(a, png_a)}));
    try testing.expect(pvr.object.get("isError").?.bool);
    try testing.expect(std.mem.indexOf(u8, pvr.object.get("content").?.array.items[0].object.get("text").?.string, "image_to_video") != null);
    const models = (try mcpText(a, try mcpCall(&d, a, "list_models", "{}"))).object.get("models").?.array.items;
    try testing.expectEqual(@as(usize, 1), models[0].object.get("capabilities").?.array.items.len); // plain
    try testing.expectEqualStrings("image_edit", models[1].object.get("capabilities").?.array.items[1].string); // tp
    try testing.expectEqual(@as(i64, 0), models[1].object.get("priority").?.integer);
}

const private_reason = [_][]const u8{ "loopback, private or link-local", "allow_private_urls" };

/// The URL of a fresh image from MCP, as the daemon names it (the Host header the client sent).
fn ownUrl(d: *Daemon, a: std.mem.Allocator) ![]const u8 {
    const m = try mcpCall(d, a, "generate_image", "{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"inline_images\":false}");
    try testing.expect(!m.object.get("isError").?.bool);
    return (try mcpText(a, m)).object.get("files").?.array.items[0].string;
}

test "URL inputs: private, loopback and link-local addresses are refused unless allowed; the daemon's own URLs work" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = ", \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\"}, {\"id\": \"tpv\", \"kind\": \"video\", \"engine\": \"testpattern\"}]}";
    var d = try Daemon.start(testing.allocator, io, "urls", "{" ++ reserve ++ tools);
    defer d.stop();

    // Nothing is fetched from these: a 400 with the reason, for edits, image to video and the MCP tools.
    const bad = [_][]const u8{
        "http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:1/x.png", "http://[::1]:1/x.png", "http://0.0.0.0/x.png",
        "http://10.0.0.1/x.png", "http://192.168" ++ ".1.1:8080/x.png", "http://172.16.5.5/x.png", "https://[fd00::1]/x.png", "http://[::ffff:10.0.0.1]/x.png",
    };
    for (bad) |u| {
        errdefer std.debug.print("url: {s}\n", .{u});
        try expectRefusal(try d.call(a, .POST, "/v1/images/edits", try std.fmt.allocPrint(a, "{{\"model\":\"tp\",\"prompt\":\"x\",\"image\":\"{s}\"}}", .{u})), &private_reason);
        try expectRefusal(try d.call(a, .POST, "/v1/videos", try std.fmt.allocPrint(a, "{{\"model\":\"tpv\",\"prompt\":\"x\",\"size\":\"256x256\",\"seconds\":\"1\",\"input_reference\":\"{s}\"}}", .{u})), &private_reason);
        const e = try mcpCall(&d, a, "generate_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"x\",\"images\":[\"{s}\"]}}", .{u}));
        try testing.expect(e.object.get("isError").?.bool);
        const text = e.object.get("content").?.array.items[0].object.get("text").?.string;
        for (private_reason) |n| try testing.expect(std.mem.indexOf(u8, text, n) != null);
        const v = try mcpCall(&d, a, "generate_video", try std.fmt.allocPrint(a, "{{\"prompt\":\"x\",\"image\":\"{s}\"}}", .{u}));
        try testing.expect(v.object.get("isError").?.bool);
    }

    // The daemon's own output URL (its address as the request reached it) is fetched; the same file named another way
    // (127.1 is 127.0.0.1 to the resolver, but it is not the address this request came to) is refused.
    const url = try ownUrl(&d, a);
    const ok = try mcpCall(&d, a, "generate_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"images\":[\"{s}\"],\"inline_images\":false}}", .{url}));
    try testing.expect(!ok.object.get("isError").?.bool);
    const alt = try std.mem.replaceOwned(u8, a, url, "127.0.0.1", "127.1");
    try expectRefusal(try d.call(a, .POST, "/v1/images/edits", try std.fmt.allocPrint(a, "{{\"model\":\"tp\",\"prompt\":\"x\",\"image\":\"{s}\"}}", .{alt})), &private_reason);

    // allow_private_urls: true opens it. The same alternative spelling now reaches the daemon and edits the picture.
    var d2 = try Daemon.start(testing.allocator, io, "urls-open", "{" ++ reserve ++ ", \"allow_private_urls\": true" ++ tools);
    defer d2.stop();
    const url2 = try std.mem.replaceOwned(u8, a, try ownUrl(&d2, a), "127.0.0.1", "127.1");
    const open = try mcpCall(&d2, a, "generate_image", try std.fmt.allocPrint(a, "{{\"prompt\":\"bars\",\"size\":\"256x256\",\"seed\":7,\"images\":[\"{s}\"],\"inline_images\":false}}", .{url2}));
    if (open.object.get("isError").?.bool) std.debug.print("allow_private_urls: {s}\n", .{open.object.get("content").?.array.items[0].object.get("text").?.string});
    try testing.expect(!open.object.get("isError").?.bool);
}

fn waitState(d: *Daemon, a: std.mem.Allocator, id: []const u8, want: []const u8, tries: usize) !void {
    for (0..tries) |_| {
        if (std.mem.eql(u8, try d.toolState(a, id), want)) return;
        sleepMs(d.io, 250);
    }
    std.debug.print("{s}: wanted {s}, is {s}\n", .{ id, want, try d.toolState(a, id) });
    return error.TestExpectedEqual;
}

test "priority and keep_loaded: loaded at start, kept past the TTL, evicted last, back when memory frees" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Budget 150 MiB. `keep` (40, priority 10, keep_loaded, TTL 1 s), `low` (40), `big` (80), `huge` (130).
    var d = try Daemon.start(testing.allocator, io, "keep", "{" ++ reserve ++ ", \"budget_bytes\": 157286400, \"tools\": [" ++
        "{\"id\": \"keep\", \"kind\": \"image\", \"engine\": \"testpattern\", \"priority\": 10, \"keep_loaded\": true, \"idle_ttl_s\": 1, \"options\": {\"resident_mb\": 40}}," ++
        "{\"id\": \"low\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 600, \"options\": {\"resident_mb\": 40}}," ++
        "{\"id\": \"big\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 1, \"options\": {\"resident_mb\": 80}}," ++
        "{\"id\": \"huge\", \"kind\": \"image\", \"engine\": \"testpattern\", \"idle_ttl_s\": 1, \"options\": {\"resident_mb\": 130}}]}");
    defer d.stop();

    // Up at start, without any request, and the listing says why.
    try waitState(&d, a, "keep", "ready", 40);
    const T = struct { data: []const struct { id: []const u8, priority: i32, keep_loaded: bool } };
    const listed = try std.json.parseFromSliceLeaky(T, a, (try d.call(a, .GET, "/v1/tools", null)).body, .{ .ignore_unknown_fields = true });
    try testing.expectEqual(@as(i32, 10), listed.data[0].priority);
    try testing.expect(listed.data[0].keep_loaded and !listed.data[1].keep_loaded);

    // It outlives its 1 s TTL.
    sleepMs(io, 2500);
    try testing.expectEqualStrings("ready", try d.toolState(a, "keep"));

    // `low` is the most recently used; `big` needs room for 80 MiB. Priority, not recency, picks `low` to go.
    try testing.expectEqual(std.http.Status.ok, (try harness.image(&d, a, "low", "")).status);
    try testing.expectEqual(std.http.Status.ok, (try harness.image(&d, a, "big", "")).status);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "low"));
    try testing.expectEqualStrings("ready", try d.toolState(a, "keep"));
    try testing.expectEqualStrings("ready", try d.toolState(a, "big"));

    // Priority only orders victims: `huge` (priority 0) still evicts the high-priority `keep` when nothing else makes room.
    try testing.expectEqual(std.http.Status.ok, (try harness.image(&d, a, "huge", "")).status);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "keep"));
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "big"));
    try testing.expectEqualStrings("ready", try d.toolState(a, "huge"));

    // `huge` goes idle after 1 s; `keep` comes back by itself.
    try waitState(&d, a, "huge", "unloaded", 20);
    try waitState(&d, a, "keep", "ready", 60);

    // Unloaded on request, it stays down.
    try testing.expectEqual(std.http.Status.ok, (try d.call(a, .POST, "/v1/tools/keep/unload", "")).status);
    sleepMs(io, 5000);
    try testing.expectEqualStrings("unloaded", try d.toolState(a, "keep"));
}

test "mcp-stdio: lines in, lines out, notifications silent, an unreachable daemon answers with an error" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var d = try Daemon.start(testing.allocator, io, "stdio", "{" ++ reserve ++ ", \"tools\": [{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\"}]}");
    defer d.stop();

    const input = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}}\n" ++
        "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n" ++
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n";
    const out = try pipe(io, a, try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{d.port}), input);
    var lines = std.mem.tokenizeScalar(u8, out, '\n');
    const first = try std.json.parseFromSliceLeaky(std.json.Value, a, lines.next().?, .{});
    try testing.expectEqual(@as(i64, 1), first.object.get("id").?.integer);
    try testing.expectEqualStrings("2025-03-26", first.object.get("result").?.object.get("protocolVersion").?.string);
    const second = try std.json.parseFromSliceLeaky(std.json.Value, a, lines.next().?, .{});
    try testing.expectEqual(@as(i64, 2), second.object.get("id").?.integer);
    try testing.expectEqual(@as(usize, 6), second.object.get("result").?.object.get("tools").?.array.items.len);
    try testing.expect(lines.next() == null); // the notification produced nothing

    const down = try pipe(io, a, "http://127.0.0.1:1", input);
    var dl = std.mem.tokenizeScalar(u8, down, '\n');
    const err = try std.json.parseFromSliceLeaky(std.json.Value, a, dl.next().?, .{});
    try testing.expectEqual(@as(i64, -32000), err.object.get("error").?.object.get("code").?.integer);
    try testing.expectEqual(@as(i64, 1), err.object.get("id").?.integer);
}

/// Runs `localrouter mcp-stdio --url url`, feeds it `input`, closes its stdin, and returns what it printed.
fn pipe(io: Io, a: std.mem.Allocator, url: []const u8, input: []const u8) ![]u8 {
    var child = try std.process.spawn(io, .{ .argv = &.{ opts.localrouter_exe, "mcp-stdio", "--url", url }, .stdin = .pipe, .stdout = .pipe, .stderr = .inherit });
    try child.stdin.?.writeStreamingAll(io, input);
    child.stdin.?.close(io);
    child.stdin = null;
    var buf: [4096]u8 = undefined;
    var r = child.stdout.?.reader(io, &buf);
    const out = try r.interface.allocRemaining(a, .limited(1 << 20));
    _ = try child.wait(io);
    return out;
}
