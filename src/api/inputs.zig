//! Input images for edits and image to video, whatever way they arrive: a multipart part, a base64 string or data URL,
//! or an http(s) URL the daemon fetches. Everything ends as validated bytes (PNG, JPEG or WebP, at most 32 MB each)
//! named for the job directory; the routes write them there and put the names in the request.
//! URL fetches are guarded (`Policy`): the host is resolved here, every address must be public (no loopback, private,
//! link-local, CGNAT, unique-local, multicast or unspecified), the connection goes to the address that was checked
//! (no second lookup to rebind), and redirects are not followed. Exceptions: the daemon's own address as the request
//! reached it (its output URLs), and the config's `allow_private_urls`.
//! Also the small multipart/form-data reader the routes use for forms (`Form`).

const std = @import("std");
const Io = std.Io;

pub const max_bytes: usize = 32 << 20; // one image, uploaded or fetched
pub const fetch_timeout_s = 30;

/// An input image ready for the job directory.
pub const File = struct { name: []const u8, data: []const u8 };

pub const Invalid = error{Invalid};

/// Where URL inputs may point. The default refuses every non-public address.
pub const Policy = struct {
    allow_private: bool = false, // the config's allow_private_urls
    host_header: []const u8 = "", // the Host header of the request being served: the daemon's own address as it was reached
    self_addr: Io.net.IpAddress = .{ .ip4 = .loopback(0) }, // where the daemon itself listens (a wildcard bind: loopback)
};

fn fail(why: *[]const u8, msg: []const u8) Invalid {
    why.* = msg;
    return error.Invalid;
}

/// The extension of the image `data` is, by its magic bytes; null for anything else.
pub fn sniff(data: []const u8) ?[]const u8 {
    if (std.mem.startsWith(u8, data, "\x89PNG\r\n\x1a\n")) return "png";
    if (std.mem.startsWith(u8, data, "\xff\xd8\xff")) return "jpg";
    if (data.len >= 12 and std.mem.eql(u8, data[0..4], "RIFF") and std.mem.eql(u8, data[8..12], "WEBP")) return "webp";
    return null;
}

/// Checks every image (count, size, format) and names it `<stem>_<i>.<ext>`.
pub fn validate(a: std.mem.Allocator, stem: []const u8, raws: []const []const u8, max_count: usize, why: *[]const u8) Invalid![]File {
    if (raws.len > max_count) return fail(why, "too many input images");
    const files = a.alloc(File, raws.len) catch return fail(why, "out of memory");
    for (files, raws, 0..) |*f, data, i| {
        if (data.len > max_bytes) return fail(why, "an input image is larger than 32 MB");
        const ext = sniff(data) orelse return fail(why, "input images must be PNG, JPEG or WebP");
        f.* = .{ .name = std.fmt.allocPrint(a, "{s}_{d}.{s}", .{ stem, i, ext }) catch return fail(why, "out of memory"), .data = data };
    }
    return files;
}

/// The bytes a spec names: an http(s) URL (fetched), a `data:...;base64,` URL or plain base64.
pub fn resolve(io: Io, gpa: std.mem.Allocator, a: std.mem.Allocator, spec: []const u8, policy: Policy, why: *[]const u8) Invalid![]const u8 {
    if (std.ascii.startsWithIgnoreCase(spec, "http://") or std.ascii.startsWithIgnoreCase(spec, "https://")) return fetch(io, gpa, a, spec, policy, why);
    var b64 = spec;
    if (std.mem.startsWith(u8, spec, "data:")) {
        const comma = std.mem.indexOfScalar(u8, spec, ',') orelse return fail(why, "malformed data URL");
        if (std.mem.indexOf(u8, spec[0..comma], ";base64") == null) return fail(why, "data URLs must be base64");
        b64 = spec[comma + 1 ..];
    } else if (std.mem.indexOf(u8, spec[0..@min(spec.len, 16)], "://") != null) return fail(why, "only http and https URLs are accepted");
    const dec = std.base64.standard.decoderWithIgnore(" \r\n\t");
    const out = a.alloc(u8, dec.calcSizeUpperBound(b64.len)) catch return fail(why, "out of memory");
    const n = dec.decode(out, b64) catch return fail(why, "an input image is neither valid base64 nor an http(s) URL");
    return out[0..n];
}

/// Collects image specs from a JSON value: a string, an {"image_url": string | {"url": string}} object, or an array of them.
pub fn specs(a: std.mem.Allocator, v: ?std.json.Value, out: *std.ArrayList([]const u8), why: *[]const u8) Invalid!void {
    const x = v orelse return;
    switch (x) {
        .null => {},
        .string => |s| out.append(a, s) catch return fail(why, "out of memory"),
        .array => |arr| for (arr.items) |item| try specs(a, item, out, why),
        .object => |o| {
            const u = o.get("image_url") orelse return fail(why, "an image object needs image_url");
            if (u == .object) return specs(a, u.object.get("url"), out, why);
            return specs(a, u, out, why);
        },
        else => return fail(why, "images must be strings (base64 or URLs)"),
    }
}

/// GET with a size cap and a deadline; the fetch runs beside a timer and the first to finish wins.
fn fetch(io: Io, gpa: std.mem.Allocator, a: std.mem.Allocator, url: []const u8, policy: Policy, why: *[]const u8) Invalid![]const u8 {
    const R = union(enum) { body: anyerror![]const u8, timer: Io.Cancelable!void };
    var buf: [2]R = undefined;
    var sel: Io.Select(R) = .init(io, &buf);
    defer sel.cancelDiscard();
    sel.concurrent(.body, get, .{ io, gpa, a, url, policy }) catch return fail(why, "could not start the fetch");
    sel.concurrent(.timer, nap, .{io}) catch return fail(why, "could not start the fetch");
    const first = sel.await() catch return fail(why, "cancelled");
    switch (first) {
        .timer => return fail(why, "fetching an input image timed out"),
        .body => |r| return r catch |err| switch (err) {
            error.TooLarge => fail(why, "an input image is larger than 32 MB"),
            error.BadStatus => fail(why, "fetching an input image failed: the server did not answer 200"),
            error.Redirect => fail(why, "fetching an input image failed: the server redirected, and redirects are not followed"),
            error.PrivateAddress => fail(why, "input image URLs may not point at loopback, private or link-local addresses (the operator can allow it with allow_private_urls)"),
            error.UnknownHostName => fail(why, "fetching an input image failed: the host name does not resolve"),
            else => fail(why, "fetching an input image failed"),
        },
    }
}

fn nap(io: Io) Io.Cancelable!void {
    try Io.sleep(io, .fromSeconds(fetch_timeout_s), .awake);
}

/// True for every address that is not a public unicast one: the targets an input URL must not reach. IPv4 inside IPv6
/// (mapped, compatible, NAT64, 6to4) is judged as the IPv4 address it carries.
pub fn isPrivate(addr: Io.net.IpAddress) bool {
    switch (addr) {
        .ip4 => |x| return private4(x.bytes),
        .ip6 => |x| {
            const b = x.bytes;
            const zeros10 = std.mem.allEqual(u8, b[0..10], 0);
            if (zeros10 and b[10] == 0xff and b[11] == 0xff) return private4(b[12..16].*); // ::ffff:a.b.c.d
            if (std.mem.allEqual(u8, b[0..12], 0)) return true; // ::, ::1 and the deprecated ::a.b.c.d
            if (std.mem.eql(u8, b[0..12], &.{ 0, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0 })) return private4(b[12..16].*); // 64:ff9b::/96
            if (b[0] == 0x20 and b[1] == 0x02) return private4(b[2..6].*); // 2002::/16 (6to4)
            if (b[0] & 0xfe == 0xfc) return true; // fc00::/7 unique local
            if (b[0] == 0xfe and b[1] & 0xc0 == 0x80) return true; // fe80::/10 link-local
            if (b[0] == 0xfe and b[1] & 0xc0 == 0xc0) return true; // fec0::/10 site-local (deprecated)
            if (b[0] == 0xff) return true; // multicast
            return false;
        },
    }
}

fn private4(b: [4]u8) bool {
    if (b[0] == 0 or b[0] == 10 or b[0] == 127) return true; // this network (0.0.0.0), 10/8, loopback
    if (b[0] == 169 and b[1] == 254) return true; // link-local, the cloud metadata address
    if (b[0] == 172 and b[1] & 0xf0 == 16) return true; // 172.16/12
    if (b[0] == 192 and b[1] == 168) return true;
    if (b[0] == 100 and b[1] & 0xc0 == 64) return true; // 100.64/10 carrier-grade NAT (tailnet addresses)
    return b[0] >= 224; // multicast, reserved, broadcast
}

/// The address to connect to for `host` (a name or an IP literal) at `port`, or an error. The daemon's own address
/// (an http URL whose authority equals the request's Host header and whose port is the listening one) maps to where the daemon
/// listens, so a forged Host header cannot name another machine. Otherwise every address the name resolves to must be
/// public, unless the policy allows private ones.
fn target(io: Io, host: []const u8, port: u16, tls: bool, authority: []const u8, policy: Policy) !Io.net.IpAddress {
    if (!tls and policy.host_header.len > 0 and std.ascii.eqlIgnoreCase(authority, policy.host_header) and port == policy.self_addr.getPort()) return policy.self_addr;
    var lit = host;
    if (lit.len > 1 and lit[0] == '[' and lit[lit.len - 1] == ']') lit = lit[1 .. lit.len - 1];
    if (Io.net.IpAddress.parse(lit, port)) |ip| {
        if (!policy.allow_private and isPrivate(ip)) return error.PrivateAddress;
        return ip;
    } else |_| {}
    const name = try Io.net.HostName.init(host);
    var qbuf: [32]Io.net.HostName.LookupResult = undefined;
    var q: Io.Queue(Io.net.HostName.LookupResult) = .init(&qbuf);
    var fut = io.async(Io.net.HostName.lookup, .{ name, io, &q, .{ .port = port } });
    defer fut.cancel(io) catch {};
    var first: ?Io.net.IpAddress = null;
    while (q.getOne(io)) |r| switch (r) {
        .address => |ip| {
            if (!policy.allow_private and isPrivate(ip)) return error.PrivateAddress; // one private answer refuses the lot
            if (first == null) first = ip;
        },
        .canonical_name => {},
    } else |err| switch (err) {
        error.Canceled => return error.Canceled,
        error.Closed => {},
    }
    try fut.await(io);
    return first orelse error.UnknownHostName;
}

/// `addr` as text for a host-name slot (the connection goes to this literal, not to a second lookup).
fn ipText(buf: *[48]u8, addr: Io.net.IpAddress) []const u8 {
    return switch (addr) {
        .ip4 => |x| std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{ x.bytes[0], x.bytes[1], x.bytes[2], x.bytes[3] }) catch unreachable,
        .ip6 => |x| std.fmt.bufPrint(buf, "{x}:{x}:{x}:{x}:{x}:{x}:{x}:{x}", .{
            std.mem.readInt(u16, x.bytes[0..2], .big),   std.mem.readInt(u16, x.bytes[2..4], .big),
            std.mem.readInt(u16, x.bytes[4..6], .big),   std.mem.readInt(u16, x.bytes[6..8], .big),
            std.mem.readInt(u16, x.bytes[8..10], .big),  std.mem.readInt(u16, x.bytes[10..12], .big),
            std.mem.readInt(u16, x.bytes[12..14], .big), std.mem.readInt(u16, x.bytes[14..16], .big),
        }) catch unreachable,
    };
}

fn get(io: Io, gpa: std.mem.Allocator, a: std.mem.Allocator, url: []const u8, policy: Policy) anyerror![]const u8 {
    const uri = try std.Uri.parse(url);
    const tls = std.ascii.eqlIgnoreCase(uri.scheme, "https");
    const port: u16 = uri.port orelse if (tls) 443 else 80;
    var hbuf: [Io.net.HostName.max_len]u8 = undefined;
    const host = try (uri.host orelse return error.InvalidUrl).toRaw(&hbuf);
    const authority = authorityOf(url);
    const dest = try target(io, host, port, tls, authority, policy);
    var tbuf: [48]u8 = undefined;
    const pinned: Io.net.HostName = .{ .bytes = ipText(&tbuf, dest) };
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();
    // connected to the checked address; TLS still verifies the certificate against the URL's own host name
    const proxied: ?Io.net.HostName = if (tls and !std.mem.eql(u8, host, pinned.bytes)) Io.net.HostName.init(host) catch null else null;
    const conn = try client.connectTcpOptions(.{ .host = pinned, .port = port, .protocol = if (tls) .tls else .plain, .proxied_host = proxied });
    var req = try client.request(.GET, uri, .{ .connection = conn, .keep_alive = false, .redirect_behavior = .unhandled, .headers = .{ .accept_encoding = .omit } });
    defer req.deinit();
    try req.sendBodiless();
    var redirect: [4096]u8 = undefined;
    var res = try req.receiveHead(&redirect);
    if (res.head.status.class() == .redirect) return error.Redirect;
    if (res.head.status != .ok) return error.BadStatus;
    if (res.head.content_length) |n| if (n > max_bytes) return error.TooLarge;
    var rbuf: [4096]u8 = undefined;
    return res.reader(&rbuf).allocRemaining(a, .limited(max_bytes)) catch |err| switch (err) {
        error.StreamTooLong => error.TooLarge,
        else => err,
    };
}

/// The `host[:port]` part of an absolute URL, as written (userinfo included, so it never equals a Host header).
fn authorityOf(url: []const u8) []const u8 {
    const at = (std.mem.indexOf(u8, url, "://") orelse return "") + 3;
    const rest = url[at..];
    return rest[0 .. std.mem.indexOfAny(u8, rest, "/?#") orelse rest.len];
}

// ---------------------------------------------------------------- multipart/form-data

pub const Part = struct { name: []const u8, filename: ?[]const u8, data: []const u8 };

/// A parsed multipart body; the parts point into the body.
pub const Form = struct {
    parts: []const Part,

    /// The boundary of a `multipart/form-data; boundary=...` content type; null for any other type.
    pub fn boundary(content_type: []const u8) ?[]const u8 {
        if (!std.ascii.startsWithIgnoreCase(content_type, "multipart/form-data")) return null;
        const at = std.mem.indexOf(u8, content_type, "boundary=") orelse return null;
        var b = content_type[at + "boundary=".len ..];
        b = b[0 .. std.mem.indexOfScalar(u8, b, ';') orelse b.len];
        b = std.mem.trim(u8, b, " \"");
        return if (b.len == 0) null else b;
    }

    pub fn parse(a: std.mem.Allocator, body: []const u8, bound: []const u8) !Form {
        const delim = try std.fmt.allocPrint(a, "--{s}", .{bound});
        const next_delim = try std.fmt.allocPrint(a, "\r\n--{s}", .{bound});
        var parts: std.ArrayList(Part) = .empty;
        var pos = (std.mem.indexOf(u8, body, delim) orelse return error.BadForm) + delim.len;
        while (!std.mem.startsWith(u8, body[pos..], "--")) {
            if (!std.mem.startsWith(u8, body[pos..], "\r\n")) return error.BadForm;
            pos += 2;
            const head_end = std.mem.indexOfPos(u8, body, pos, "\r\n\r\n") orelse return error.BadForm;
            const data_start = head_end + 4;
            const next = std.mem.indexOfPos(u8, body, data_start, next_delim) orelse return error.BadForm;
            if (disposition(body[pos..head_end])) |d| try parts.append(a, .{ .name = d[0], .filename = d[1], .data = body[data_start..next] });
            pos = next + next_delim.len;
            if (pos >= body.len) return error.BadForm;
        }
        return .{ .parts = parts.items };
    }

    /// The text of the first plain field `name`.
    pub fn field(f: Form, name: []const u8) ?[]const u8 {
        for (f.parts) |p| if (p.filename == null and std.mem.eql(u8, p.name, name)) return p.data;
        return null;
    }

    /// Fills the fields of `T` (strings, ints, floats, bools, optionals of them) that the form has; null on a bad value.
    pub fn fill(f: Form, comptime T: type, why: *[]const u8) Invalid!T {
        var b: T = .{};
        inline for (@typeInfo(T).@"struct".field_names, @typeInfo(T).@"struct".field_types) |name, FT| if (f.field(name)) |v| {
            @field(b, name) = parseValue(FT, v) orelse return fail(why, "a form field has a bad value: " ++ name);
        };
        return b;
    }
};

fn parseValue(comptime T: type, v: []const u8) ?T {
    return switch (@typeInfo(T)) {
        .optional => |o| parseValue(o.child, v),
        .int => std.fmt.parseInt(T, std.mem.trim(u8, v, " "), 10) catch null,
        .float => std.fmt.parseFloat(T, std.mem.trim(u8, v, " ")) catch null,
        .bool => if (std.mem.eql(u8, v, "true")) true else if (std.mem.eql(u8, v, "false")) false else null,
        .pointer => v,
        else => @compileError("unsupported form field type"),
    };
}

/// (name, filename) from a part's header block.
fn disposition(head: []const u8) ?struct { []const u8, ?[]const u8 } {
    var lines = std.mem.splitSequence(u8, head, "\r\n");
    while (lines.next()) |line| {
        if (!std.ascii.startsWithIgnoreCase(line, "content-disposition:")) continue;
        return .{ param(line, " name=\"") orelse return null, param(line, " filename=\"") };
    }
    return null;
}

fn param(line: []const u8, key: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, line, key) orelse return null;
    const rest = line[at + key.len ..];
    return rest[0 .. std.mem.indexOfScalar(u8, rest, '"') orelse return null];
}

test "multipart: fields, files and a bad boundary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "--XX\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhi\r\n--XX\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n\x1a\n..\r\n--XX\r\nContent-Disposition: form-data; name=\"n\"\r\n\r\n2\r\n--XX--\r\n";
    const f = try Form.parse(a, body, Form.boundary("multipart/form-data; boundary=XX").?);
    try std.testing.expectEqual(@as(usize, 3), f.parts.len);
    try std.testing.expectEqualStrings("hi", f.field("prompt").?);
    try std.testing.expectEqualStrings("a.png", f.parts[1].filename.?);
    try std.testing.expectEqualStrings("\x89PNG\r\n\x1a\n..", f.parts[1].data);
    var why: []const u8 = "";
    const T = struct { prompt: []const u8 = "", n: u32 = 1, seed: ?u64 = null };
    const t = try f.fill(T, &why);
    try std.testing.expectEqual(@as(u32, 2), t.n);
    try std.testing.expectError(error.BadForm, Form.parse(a, "nothing", "XX"));
    try std.testing.expect(Form.boundary("application/json") == null);
}

test "resolve base64 and data URLs; validate formats and counts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const png = try resolve(std.testing.io, std.testing.allocator, a, "data:image/png;base64,iVBORw0KGgo=", .{}, &why);
    try std.testing.expectEqualStrings("png", sniff(png).?);
    try std.testing.expectError(error.Invalid, resolve(std.testing.io, std.testing.allocator, a, "file:///etc/passwd", .{}, &why));
    try std.testing.expectEqualStrings("only http and https URLs are accepted", why);
    const files = try validate(a, "ref", &.{png}, 5, &why);
    try std.testing.expectEqualStrings("ref_0.png", files[0].name);
    try std.testing.expectError(error.Invalid, validate(a, "ref", &.{"GIF89a"}, 5, &why));
    try std.testing.expectError(error.Invalid, validate(a, "ref", &.{ png, png, png, png, png, png }, 5, &why));
}

fn ipOf(text: []const u8) Io.net.IpAddress {
    return Io.net.IpAddress.parse(text, 80) catch unreachable;
}

test "address classification: what an input URL may not reach" {
    const refused = [_][]const u8{
        "0.0.0.0",           "0.1.2.3",              "10.0.0.1",           "10.255.255.255",    "127.0.0.1",        "127.8.9.10",
        "169.254.169.254",   "169.254.0.1",          "172.16.0.1",         "172.31.255.255",    "192.168" ++ ".0.1",      "192.168" ++ ".1.100",
        "100.64.0.1",        "100.127.255.255",      "224.0.0.1",          "255.255.255.255",   "::",               "::1",
        "fc00::1",           "fd12:3456:789a::1",    "fe80::1",            "febf::1",           "fec0::1",          "ff02::1",
        "::ffff:127.0.0.1",  "::ffff:10.1.2.3",      "::ffff:169.254.169.254", "::7f00:1",      "64:ff9b::a00:1",   "2002:c0a8:101::1",
    };
    for (refused) |t| {
        errdefer std.debug.print("should be refused: {s}\n", .{t});
        try std.testing.expect(isPrivate(ipOf(t)));
    }
    const public = [_][]const u8{
        "1.1.1.1",  "8.8.8.8",  "93.184.216.34", "172.15.255.255", "172.32.0.1", "192.167.1.1", "192.169.0.1", "100.63.255.255", "100.128.0.1",
        "169.253.1.1", "223.255.255.255", "2606:4700:4700::1111", "2001:4860:4860::8888", "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::1", "fb00::1",
    };
    for (public) |t| {
        errdefer std.debug.print("should be allowed: {s}\n", .{t});
        try std.testing.expect(!isPrivate(ipOf(t)));
    }
}

test "URL inputs: private literals are refused before any connection; the daemon's own address and the opt-in are the exceptions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const urls = [_][]const u8{
        "http://127.0.0.1:1/x.png",           "http://localhost:1/x.png",     "http://169.254.169.254/latest/meta-data/", "http://[::1]/x.png",
        "http://0.0.0.0/x.png",               "http://10.1.2.3/x.png",        "http://192.168" ++ ".1.1:8080/x.png",           "https://172.16.0.9/x.png",
        "http://[::ffff:127.0.0.1]/x.png",    "http://[fd00::1]/x.png",       "http://[fe80::1]/x.png",                  "http://%31%32%37.0.0.1/x.png",
    };
    for (urls) |u| {
        errdefer std.debug.print("not refused: {s} ({s})\n", .{ u, why });
        try std.testing.expectError(error.Invalid, resolve(std.testing.io, std.testing.allocator, a, u, .{}, &why));
        try std.testing.expect(std.mem.indexOf(u8, why, "loopback, private or link-local") != null);
    }
    // the exception is exact: same authority as the Host header AND the listening port, plain http only
    const self: Policy = .{ .host_header = "192.168" ++ ".1.5:8190", .self_addr = ipOf("127.0.0.1") };
    var p = self;
    p.self_addr.setPort(8190);
    const t = try target(std.testing.io, "192.168" ++ ".1.5", 8190, false, "192.168" ++ ".1.5:8190", p);
    try std.testing.expect(t.eql(&p.self_addr)); // connects to the daemon itself, never to the name in the URL
    try std.testing.expectError(error.PrivateAddress, target(std.testing.io, "192.168" ++ ".1.5", 8190, true, "192.168" ++ ".1.5:8190", p)); // https
    try std.testing.expectError(error.PrivateAddress, target(std.testing.io, "192.168" ++ ".1.5", 9999, false, "192.168" ++ ".1.5:9999", p)); // another port
    try std.testing.expectError(error.PrivateAddress, target(std.testing.io, "192.168" ++ ".1.6", 8190, false, "192.168" ++ ".1.6:8190", p)); // another host
    var forged = p; // a Host header naming the metadata address: still only the daemon's own port, and it connects to the daemon
    forged.host_header = "169.254.169.254:8190";
    const f = try target(std.testing.io, "169.254.169.254", 8190, false, "169.254.169.254:8190", forged);
    try std.testing.expect(f.eql(&p.self_addr));
    try std.testing.expectError(error.PrivateAddress, target(std.testing.io, "169.254.169.254", 80, false, "169.254.169.254", forged));
    // allow_private_urls opens everything (the literal is returned for connecting)
    var open = p;
    open.allow_private = true;
    try std.testing.expect((try target(std.testing.io, "10.0.0.7", 80, false, "10.0.0.7", open)).eql(&ipOf("10.0.0.7")));
    try std.testing.expectEqualStrings("a.b:81", authorityOf("http://a.b:81/x?y"));
    try std.testing.expectEqualStrings("u@a.b", authorityOf("http://u@a.b#f"));
}
