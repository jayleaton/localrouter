//! Where the daemon listens: the `host` setting (config key and `--host`) turned into an address.
//! Accepted: an IP literal, `localhost` (127.0.0.1), `all` (0.0.0.0, every interface) and `tailscale` (this machine's
//! tailnet IPv4, found by listing the network interfaces; the tailscale CLI is never run).

const std = @import("std");
const builtin = @import("builtin");

pub const default_host = "127.0.0.1";

/// One IPv4 address of one network interface.
pub const Iface = struct {
    name: []const u8,
    addr: [4]u8,
};

pub const Error = error{ NoTailnetAddress, InvalidHost, OutOfMemory };

/// The tailnet IPv4 range, 100.64.0.0/10 (carrier-grade NAT space, which Tailscale hands out).
pub fn inTailnet(a: [4]u8) bool {
    return a[0] == 100 and (a[1] & 0xc0) == 64;
}

/// The tailnet address among `ifaces`: the `tailscale0` interface when it has one, else the first interface with an
/// address in 100.64/10.
pub fn pickTailnet(ifaces: []const Iface) ?[4]u8 {
    for (ifaces) |f| if (std.mem.eql(u8, f.name, "tailscale0") and inTailnet(f.addr)) return f.addr;
    for (ifaces) |f| if (inTailnet(f.addr)) return f.addr;
    return null;
}

/// Turns a `host` setting into the IP literal to listen on; the result lives in `arena`. `ifaces` is only used for
/// `tailscale`.
pub fn resolve(arena: std.mem.Allocator, host: []const u8, ifaces: []const Iface) Error![]const u8 {
    const h = std.mem.trim(u8, host, " \t");
    if (std.ascii.eqlIgnoreCase(h, "localhost")) return "127.0.0.1";
    if (std.ascii.eqlIgnoreCase(h, "all")) return "0.0.0.0";
    if (std.ascii.eqlIgnoreCase(h, "tailscale")) {
        const a = pickTailnet(ifaces) orelse return error.NoTailnetAddress;
        return std.fmt.allocPrint(arena, "{d}.{d}.{d}.{d}", .{ a[0], a[1], a[2], a[3] });
    }
    _ = std.Io.net.IpAddress.parse(h, 0) catch return error.InvalidHost;
    return arena.dupe(u8, h);
}

/// Whether `resolve` needs the interface list for this setting.
pub fn needsInterfaces(host: []const u8) bool {
    return std.ascii.eqlIgnoreCase(std.mem.trim(u8, host, " \t"), "tailscale");
}

// ---- the machine's interfaces (getifaddrs; libc is linked)

const IfAddrs = extern struct {
    next: ?*IfAddrs,
    name: [*:0]const u8,
    flags: c_uint,
    addr: ?*const extern struct { family: u16, port: u16, ip: [4]u8 },
};
extern "c" fn getifaddrs(out: *?*IfAddrs) c_int;
extern "c" fn freeifaddrs(list: ?*IfAddrs) void;
const af_inet = 2;

/// The IPv4 addresses of every interface; names and the slice live in `arena`.
pub fn listInterfaces(arena: std.mem.Allocator) ![]const Iface {
    if (builtin.os.tag != .linux) return &.{};
    var head: ?*IfAddrs = null;
    if (getifaddrs(&head) != 0) return error.InterfacesUnavailable;
    defer freeifaddrs(head);
    var list: std.ArrayList(Iface) = .empty;
    var it = head;
    while (it) |f| : (it = f.next) {
        const a = f.addr orelse continue;
        if (a.family != af_inet) continue;
        try list.append(arena, .{ .name = try arena.dupe(u8, std.mem.span(f.name)), .addr = a.ip });
    }
    return list.items;
}

test "resolve: keywords and literals" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("127.0.0.1", try resolve(a, "localhost", &.{}));
    try std.testing.expectEqualStrings("127.0.0.1", try resolve(a, "LocalHost", &.{}));
    try std.testing.expectEqualStrings("0.0.0.0", try resolve(a, "all", &.{}));
    try std.testing.expectEqualStrings("192.0.2.5", try resolve(a, "192.0.2.5", &.{}));
    try std.testing.expectEqualStrings("::1", try resolve(a, "::1", &.{}));
    try std.testing.expectEqualStrings("0.0.0.0", try resolve(a, "0.0.0.0", &.{}));
    try std.testing.expectError(error.InvalidHost, resolve(a, "example.com", &.{}));
    try std.testing.expectError(error.InvalidHost, resolve(a, "", &.{}));
    try std.testing.expectError(error.InvalidHost, resolve(a, "300.1.1.1", &.{}));
    try std.testing.expect(needsInterfaces("tailscale"));
    try std.testing.expect(!needsInterfaces("all"));
}

test "tailnet range is 100.64.0.0/10" {
    try std.testing.expect(inTailnet(.{ 100, 64, 0, 0 }));
    try std.testing.expect(inTailnet(.{ 100, 64, 0, 7 }));
    try std.testing.expect(inTailnet(.{ 100, 127, 255, 255 }));
    try std.testing.expect(!inTailnet(.{ 100, 63, 255, 255 }));
    try std.testing.expect(!inTailnet(.{ 100, 128, 0, 0 }));
    try std.testing.expect(!inTailnet(.{ 101, 64, 0, 1 }));
    try std.testing.expect(!inTailnet(.{ 10, 64, 0, 1 }));
}

test "tailscale picks the tailnet address from fake interface lists" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // tailscale0 present among others
    const typical = [_]Iface{
        .{ .name = "lo", .addr = .{ 127, 0, 0, 1 } },
        .{ .name = "eno1", .addr = .{ 192, 0, 2, 10 } },
        .{ .name = "tailscale0", .addr = .{ 100, 64, 0, 7 } },
        .{ .name = "docker0", .addr = .{ 172, 17, 0, 1 } },
    };
    try std.testing.expectEqualStrings("100.64.0.7", try resolve(a, "tailscale", &typical));
    // tailscale0 wins over another CGNAT address, wherever it sits in the list
    const two = [_]Iface{
        .{ .name = "wwan0", .addr = .{ 100, 70, 1, 2 } },
        .{ .name = "tailscale0", .addr = .{ 100, 64, 0, 9 } },
    };
    try std.testing.expectEqualStrings("100.64.0.9", try resolve(a, "tailscale", &two));
    // a userspace or renamed interface: any 100.64/10 address
    const renamed = [_]Iface{
        .{ .name = "lo", .addr = .{ 127, 0, 0, 1 } },
        .{ .name = "ts0", .addr = .{ 100, 100, 100, 100 } },
    };
    try std.testing.expectEqualStrings("100.100.100.100", try resolve(a, "tailscale", &renamed));
    // none: an error, never a fallback to another address
    const none = [_]Iface{
        .{ .name = "lo", .addr = .{ 127, 0, 0, 1 } },
        .{ .name = "eno1", .addr = .{ 192, 0, 2, 10 } },
        .{ .name = "tailscale0", .addr = .{ 169, 254, 1, 1 } }, // not a tailnet address
        .{ .name = "x", .addr = .{ 100, 128, 0, 1 } }, // just outside the range
    };
    try std.testing.expectError(error.NoTailnetAddress, resolve(a, "tailscale", &none));
    try std.testing.expectError(error.NoTailnetAddress, resolve(a, "tailscale", &.{}));
}

test "this machine's interfaces can be listed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const ifaces = try listInterfaces(arena.allocator());
    if (builtin.os.tag == .linux) {
        var lo = false;
        for (ifaces) |f| lo = lo or std.mem.eql(u8, f.name, "lo");
        try std.testing.expect(lo or ifaces.len > 0);
    }
}
