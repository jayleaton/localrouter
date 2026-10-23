//! `localrouter pair`: the devices and harnesses allowed to use this LocalRouter when `pairing.required` is set, and
//! the scopes each holds (`src/pair.zig`), managed on the box. The data directory is found as `serve` finds it: the
//! config's data_dir, else --data.
//!   localrouter pair add NAME [--scopes LIST] [--config FILE] [--data DIR] [--url URL]   a code for the app (10
//!                                                     minutes, once); with --url (the address the phone reaches) its link
//!   localrouter pair token NAME [--scopes LIST] [--config FILE] [--data DIR]    a device made at once: its token, printed
//!                                                     once (a harness: --scopes inference,media,read)
//!   localrouter pair scopes ID|NAME LIST [--config FILE] [--data DIR]           replaces what a device may do
//!   localrouter pair list [--config FILE] [--data DIR]                          the devices (never their tokens)
//!   localrouter pair revoke ID|NAME [--config FILE] [--data DIR]                removes them; the next request is refused
//! The daemon reads the devices file on every check, so none of these needs a restart.

const std = @import("std");
const Io = std.Io;
const config = @import("../config.zig");
const store = @import("../pair.zig");

const Opts = struct { arg: ?[]const u8 = null, arg2: ?[]const u8 = null, config: ?[]const u8 = null, data: ?[]const u8 = null, url: ?[]const u8 = null, scopes: ?[]const u8 = null };

pub fn run(io: Io, a: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 1) return usageError();
    const verb = args[0];
    var o: Opts = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.startsWith(u8, k, "--")) {
            if (i + 1 == args.len) return usageError();
            i += 1;
            if (eql(k, "--config")) o.config = args[i] else if (eql(k, "--data")) o.data = args[i] else if (eql(k, "--url")) o.url = args[i] else if (eql(k, "--scopes")) o.scopes = args[i] else return usageError();
        } else if (o.arg == null) o.arg = k else if (o.arg2 == null) o.arg2 = k else return usageError();
    }
    var cfg: config.Config = .{};
    if (o.config) |p| {
        const text = Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 20)) catch |err| return fail("{s}: {s}", .{ p, @errorName(err) });
        cfg = config.parse(a, text) catch |err| return fail("{s}: {s}", .{ p, @errorName(err) });
    }
    const s: store.Store = .{ .dir = o.data orelse cfg.data_dir };
    var buf: [4096]u8 = undefined;
    var w = Io.File.stdout().writer(io, &buf);
    const out = &w.interface;
    const at = store.now(io);
    var when: [32]u8 = undefined;

    if (eql(verb, "add") or eql(verb, "token")) {
        if (o.arg2 != null) return usageError();
        const name = std.mem.trim(u8, o.arg orelse return usageError(), " \t");
        if (name.len == 0 or name.len > store.max_name_len) return fail("NAME must be 1 to {d} bytes", .{store.max_name_len});
        const scopes: []const []const u8 = if (o.scopes) |text| (try store.parseScopes(a, text)) orelse return fail(scopes_help, .{}) else &.{};
        if (eql(verb, "token")) {
            if (o.url != null) return usageError();
            const m = s.mintScoped(io, a, name, scopes, at) catch |err| return fail("{s}/devices.json: {s}", .{ s.dir, @errorName(err) });
            try out.print("{s}\n", .{&m.token});
            try out.flush();
            std.debug.print("paired {s} \"{s}\"; the token is shown only now (revoke: localrouter pair revoke {s})\n", .{ &m.id, name, &m.id });
        } else {
            const c = s.addCodeScoped(io, a, name, scopes, at) catch |err| return fail("{s}/devices.json: {s}", .{ s.dir, @errorName(err) });
            try out.print("code     {s}\nexpires  {s} (10 minutes, single use)\n", .{ &c.text, store.utc(&when, c.expires) });
            if (o.url) |u| try out.print("link     localrouter://pair?url={f}&code={s}\n", .{ Percent{ .s = u }, &c.text });
            try out.flush();
        }
        if (!cfg.pairing.required) std.debug.print("note: pairing.required is off in {s}: every route but computer control still answers every client, whatever its scopes\n", .{o.config orelse "the default config"});
        return 0;
    }
    if (eql(verb, "list")) {
        if (o.arg != null or o.url != null or o.scopes != null) return usageError();
        const doc = s.load(io, a) catch |err| return fail("{s}/devices.json: {s}", .{ s.dir, @errorName(err) });
        try out.print("{s:<8}  {s:<32}  {s:<20}  {s}\n", .{ "ID", "NAME", "CREATED", "SCOPES" });
        for (doc.devices) |d| {
            const scopes = if (d.scopes.len == 0) "voice,mcp (legacy)" else try std.mem.join(a, ",", d.scopes);
            try out.print("{s:<8}  {s:<32}  {s:<20}  {s}\n", .{ d.id, d.name, store.utc(&when, d.created), scopes });
        }
        var live: usize = 0;
        for (doc.pending) |p| live += @intFromBool(p.expires > at);
        if (live > 0) try out.print("{d} pending code(s)\n", .{live});
        try out.flush();
        return 0;
    }
    if (eql(verb, "scopes")) {
        const key = o.arg orelse return usageError();
        if (o.url != null or o.scopes != null) return usageError();
        const scopes = (try store.parseScopes(a, o.arg2 orelse return usageError())) orelse return fail(scopes_help, .{});
        const changed = s.setScopes(io, a, key, scopes, at) catch |err| return fail("{s}/devices.json: {s}", .{ s.dir, @errorName(err) });
        if (changed.len == 0) return fail("no device has the id or name '{s}' (localrouter pair list)", .{key});
        for (changed) |d| try out.print("{s} {s}: {s}\n", .{ d.id, d.name, try std.mem.join(a, ",", scopes) });
        try out.flush();
        return 0;
    }
    if (eql(verb, "revoke")) {
        const key = o.arg orelse return usageError();
        if (o.url != null) return usageError();
        const gone = s.revoke(io, a, key, at) catch |err| return fail("{s}/devices.json: {s}", .{ s.dir, @errorName(err) });
        if (gone.len == 0) return fail("no device has the id or name '{s}' (localrouter pair list)", .{key});
        for (gone) |d| try out.print("revoked {s} {s}\n", .{ d.id, d.name });
        try out.flush();
        return 0;
    }
    return usageError();
}

/// A URL as a query value: everything but RFC 3986's unreserved characters percent-encoded.
const Percent = struct {
    s: []const u8,

    pub fn format(p: Percent, w: *Io.Writer) Io.Writer.Error!void {
        for (p.s) |c| {
            if (std.ascii.isAlphanumeric(c) or c == '-' or c == '.' or c == '_' or c == '~') try w.writeByte(c) else try w.print("%{X:0>2}", .{c});
        }
    }
};

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn fail(comptime f: []const u8, args: anytype) u8 {
    std.debug.print("localrouter pair: " ++ f ++ "\n", args);
    return 1;
}

pub const usage_text =
    \\usage: localrouter pair add NAME [--scopes LIST] [--config FILE] [--data DIR] [--url URL]   a code for the app (10 minutes, once)
    \\       localrouter pair token NAME [--scopes LIST] [--config FILE] [--data DIR]             a token now, for a harness, an MCP client, a script or the desktop helper
    \\       localrouter pair scopes ID|NAME LIST [--config FILE] [--data DIR]                    what a paired device may do
    \\       localrouter pair list [--config FILE] [--data DIR]
    \\       localrouter pair revoke ID|NAME [--config FILE] [--data DIR]
    \\LIST: comma-separated. Service scopes, least privilege for a harness:
    \\  inference   chat completions and /v1/realtime        media    image, video, speech and transcription
    \\  decisions   typed decisions (/v1/systemone)          read     models, voices and residency
    \\  manage      load, unload, reserve and release models
    \\Computer control: desktop_helper, computer, approve. Legacy: voice (/v1/realtime) and mcp (/mcp), each with every
    \\service scope; none given is voice,mcp, as every device paired before scopes. A device sees only its own jobs.
    \\  A coding harness: --scopes inference,media,read. A desktop helper: --scopes desktop_helper.
    \\  A phone that talks, drives and approves computer tasks: voice,mcp,computer,approve.
    \\
;

const scopes_help = "scopes are a comma-separated list of inference, media, decisions, read, manage, desktop_helper, computer, approve (or the legacy voice, mcp)";

fn usageError() u8 {
    std.debug.print(usage_text, .{});
    return 2;
}

test "a URL is percent-encoded for the pairing link" {
    var buf: [128]u8 = undefined;
    const got = try std.fmt.bufPrint(&buf, "{f}", .{Percent{ .s = "https://box.tail-net.ts.net:10443/x y" }});
    try std.testing.expectEqualStrings("https%3A%2F%2Fbox.tail-net.ts.net%3A10443%2Fx%20y", got);
}
