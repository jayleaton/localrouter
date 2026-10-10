//! `localrouter gen`: the command-line client over the HTTP API, for people and agents without an OpenAI SDK.
//!   localrouter gen image PROMPT [-o out.png] [--size WxH] [--n N] [--seed N] [--steps N] [--model ID] [--url URL]
//!   localrouter gen video PROMPT [-o out.mp4] [--size WxH] [--seconds N] [--seed N] [--steps N] [--model ID] [--url URL]
//!   localrouter models [--url URL]
//! The server is $LOCALROUTER_URL, else http://127.0.0.1:8190. Videos are created, polled each 2 s, then downloaded.
//! Without --model a request goes to the server's default model for what it needs (`localrouter models`, DEFAULT FOR).

const std = @import("std");
const Io = std.Io;

pub fn run(gpa: std.mem.Allocator, io: Io, a: std.mem.Allocator, args: []const []const u8, environ: *const std.process.Environ.Map) !u8 {
    if (args.len < 2) return usageError();
    const kind = args[0];
    var o: Opts = .{ .prompt = args[1], .url = environ.get("LOCALROUTER_URL") orelse "http://127.0.0.1:8190" };
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        const k, const v = .{ args[i], args[i + 1] };
        if (eql(k, "-o")) o.out = v else if (eql(k, "--size")) o.size = v else if (eql(k, "--seed")) o.seed = v else if (eql(k, "--steps")) o.steps = v else if (eql(k, "--model")) o.model = v else if (eql(k, "--url")) o.url = v else if (eql(k, "--seconds")) o.seconds = v else if (eql(k, "--n")) o.n = v else return usageError();
    }
    if (i != args.len) return usageError();
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();
    if (eql(kind, "image")) return image(&client, io, a, o);
    if (eql(kind, "video")) return video(&client, io, a, o);
    return usageError();
}

/// `localrouter models`: one line per model on the server (GET /v1/models).
pub fn models(gpa: std.mem.Allocator, io: Io, a: std.mem.Allocator, args: []const []const u8, environ: *const std.process.Environ.Map) !u8 {
    var url = environ.get("LOCALROUTER_URL") orelse "http://127.0.0.1:8190";
    if (args.len == 2 and eql(args[0], "--url")) url = args[1] else if (args.len != 0) return usageError();
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();
    const r = try call(&client, a, .GET, url, "/v1/models", null);
    if (r.status != .ok) return report(r.body);
    const M = struct { id: []const u8, kind: []const u8, running: bool = false, capabilities: []const []const u8 = &.{}, default_for: []const []const u8 = &.{} };
    const list = try std.json.parseFromSliceLeaky(struct { data: []const M }, a, r.body, .{ .ignore_unknown_fields = true });
    var buf: [4096]u8 = undefined;
    var w = Io.File.stdout().writer(io, &buf);
    const out = &w.interface;
    try out.print("{s:<28} {s:<6} {s:<7} {s:<40} {s}\n", .{ "ID", "KIND", "LOADED", "CAPABILITIES", "DEFAULT FOR" });
    for (list.data) |m| {
        try out.print("{s:<28} {s:<6} {s:<7} {s:<40} {s}\n", .{ m.id, m.kind, if (m.running) "yes" else "no", try joined(a, m.capabilities), try joined(a, m.default_for) });
    }
    try out.flush();
    return 0;
}

fn joined(a: std.mem.Allocator, items: []const []const u8) ![]const u8 {
    return if (items.len == 0) "-" else std.mem.join(a, ",", items);
}

const Opts = struct {
    prompt: []const u8,
    url: []const u8,
    out: ?[]const u8 = null,
    size: ?[]const u8 = null,
    seed: ?[]const u8 = null,
    steps: ?[]const u8 = null,
    model: ?[]const u8 = null,
    seconds: ?[]const u8 = null,
    n: ?[]const u8 = null,
};

fn image(client: *std.http.Client, io: Io, a: std.mem.Allocator, o: Opts) !u8 {
    const body = try std.json.Stringify.valueAlloc(a, .{
        .model = o.model,
        .prompt = o.prompt,
        .size = o.size orelse "1024x1024",
        .n = try int(o.n, 1),
        .seed = if (o.seed) |s| try std.fmt.parseInt(u64, s, 10) else null,
        .steps = try int(o.steps, 0),
        .response_format = "b64_json",
    }, .{ .emit_null_optional_fields = false });
    const r = try call(client, a, .POST, o.url, "/v1/images/generations", body);
    if (r.status != .ok) return report(r.body);
    const Resp = struct { data: []const struct { b64_json: []const u8 }, seed: u64 = 0 };
    const parsed = try std.json.parseFromSliceLeaky(Resp, a, r.body, .{ .ignore_unknown_fields = true });
    const stem = o.out orelse "image.png";
    for (parsed.data, 0..) |d, k| {
        const dec = std.base64.standard.Decoder;
        const png = try a.alloc(u8, try dec.calcSizeForSlice(d.b64_json));
        try dec.decode(png, d.b64_json);
        const path = if (parsed.data.len == 1) stem else try std.fmt.allocPrint(a, "{s}-{d}.png", .{ stem[0 .. std.mem.lastIndexOfScalar(u8, stem, '.') orelse stem.len], k });
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = png });
        std.debug.print("{s} (seed {d})\n", .{ path, parsed.seed + k });
    }
    return 0;
}

fn video(client: *std.http.Client, io: Io, a: std.mem.Allocator, o: Opts) !u8 {
    const body = try std.json.Stringify.valueAlloc(a, .{
        .model = o.model,
        .prompt = o.prompt,
        .size = o.size orelse "1344x768",
        .seconds = o.seconds orelse "5",
        .seed = if (o.seed) |s| try std.fmt.parseInt(u64, s, 10) else null,
        .steps = try int(o.steps, 0),
    }, .{ .emit_null_optional_fields = false });
    var r = try call(client, a, .POST, o.url, "/v1/videos", body);
    if (r.status != .ok) return report(r.body);
    const V = struct { id: []const u8, status: []const u8, progress: u8 = 0, phase: []const u8 = "", seed: u64 = 0 };
    var v = try std.json.parseFromSliceLeaky(V, a, r.body, .{ .ignore_unknown_fields = true });
    const id = v.id;
    while (eql(v.status, "queued") or eql(v.status, "in_progress")) {
        std.debug.print("\r{s} {s} {d}%   ", .{ id, v.phase, v.progress });
        try Io.sleep(io, .fromSeconds(2), .awake);
        r = try call(client, a, .GET, o.url, try std.fmt.allocPrint(a, "/v1/videos/{s}", .{id}), null);
        v = try std.json.parseFromSliceLeaky(V, a, r.body, .{ .ignore_unknown_fields = true });
    }
    std.debug.print("\n", .{});
    if (!eql(v.status, "completed")) return report(r.body);
    r = try call(client, a, .GET, o.url, try std.fmt.allocPrint(a, "/v1/videos/{s}/content", .{id}), null);
    if (r.status != .ok) return report(r.body);
    const path = o.out orelse "video.mp4";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = r.body });
    std.debug.print("{s} (seed {d})\n", .{ path, v.seed });
    return 0;
}

const Response = struct { status: std.http.Status, body: []u8 };

fn call(client: *std.http.Client, a: std.mem.Allocator, method: std.http.Method, base: []const u8, path: []const u8, body: ?[]const u8) !Response {
    var out: Io.Writer.Allocating = .init(a);
    const url = try std.fmt.allocPrint(a, "{s}{s}", .{ std.mem.trimEnd(u8, base, "/"), path });
    const res = try client.fetch(.{
        .location = .{ .url = url },
        .method = method,
        .payload = body,
        .response_writer = &out.writer,
        .headers = .{ .content_type = if (body != null) .{ .override = "application/json" } else .default },
    });
    return .{ .status = res.status, .body = out.written() };
}

fn report(body: []const u8) u8 {
    std.debug.print("error: {s}\n", .{body});
    return 1;
}

fn int(s: ?[]const u8, default: u32) !u32 {
    return if (s) |v| try std.fmt.parseInt(u32, v, 10) else default;
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn usageError() u8 {
    std.debug.print("usage: localrouter gen image|video PROMPT [-o FILE] [--size WxH] [--n N] [--seconds N] [--seed N] [--steps N] [--model ID] [--url URL]\n" ++
        "       localrouter models [--url URL]   (the model ids, and which one serves each capability by default)\n", .{});
    return 2;
}
