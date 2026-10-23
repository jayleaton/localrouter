//! Who may touch computer control: the paired device behind a request's token, and whether its scopes allow the route.
//! Computer control rides on LocalRouter's device pairing (one credential system for phone, web, MCP and the desktop
//! helper); until pairing is switched on, every computer route refuses, because a tailnet address alone must never reach
//! someone's desktop. The gate is an interface so the pairing module plugs in here, and tests use a fake.

const std = @import("std");
const h = @import("../api/http.zig");

pub const Scope = enum {
    desktop_helper, // connect as a desktop helper, nothing else
    computer, // call the computer_control / computer_use tools
    approve, // answer approvals and stop tasks in the web UI; never given to an agent's MCP token
};

pub const Device = struct { id: []const u8, name: []const u8 };

pub const Checked = union(enum) {
    ok: Device, // its strings are valid until this thread's next check: copy them
    off: []const u8, // pairing is not switched on (or not available): the whole feature is off
    denied: []const u8, // no token, an unknown or revoked token, or a token without the scope
};

pub const Gate = struct {
    ctx: ?*anyopaque = null,
    checkFn: *const fn (ctx: ?*anyopaque, token: []const u8, scope: Scope) Checked = unpaired,

    pub fn check(g: Gate, tok: []const u8, scope: Scope) Checked {
        return g.checkFn(g.ctx, tok, scope);
    }
};

/// The gate when no pairing store is wired in (tests): everything is off.
pub const off_message = "computer control needs device pairing, which is not available in this build";
pub const unpaired_message = "computer control needs device pairing, even when the rest of LocalRouter does not: on the box run `localrouter pair token NAME --scopes ...` and send the token as Authorization: Bearer (docs/dev/COMPUTER.md)";

fn unpaired(_: ?*anyopaque, _: []const u8, _: Scope) Checked {
    return .{ .off = off_message };
}

/// The request's token, as pairing reads it (`api/pair.zig`: Bearer, the key subprotocol, or `?api_key=`); "" when
/// none. Only valid before the body is read; copy it.
pub fn token(req: *const h.Request) []const u8 {
    return @import("../api/pair.zig").credential(req) orelse "";
}

/// A paired device's scopes decide, always: computer control needs a paired device whether or not `pairing.required`
/// gates the rest of the daemon. The token must be a paired device's and the device must have the scope. `lookup`
/// finds the device (re-read on every check, so a revoke or a scope change counts at once).
pub fn Paired(comptime Ctx: type, comptime lookup: fn (ctx: *Ctx, tok: []const u8, buf: []u8) ?Found) type {
    return struct {
        pub fn gate(ctx: *Ctx) Gate {
            return .{ .ctx = ctx, .checkFn = check };
        }

        fn check(raw: ?*anyopaque, tok: []const u8, scope: Scope) Checked {
            const ctx: *Ctx = @ptrCast(@alignCast(raw.?));
            if (tok.len == 0) return .{ .denied = unpaired_message };
            const f = lookup(ctx, tok, &found_buf) orelse return .{ .denied = "computer control needs a paired device, and this token is not one (revoked, or never paired): on the box, `localrouter pair token NAME --scopes ...` (docs/dev/COMPUTER.md)" };
            if (!f.allows(scope)) return .{ .denied = switch (scope) {
                .desktop_helper => "this device's token does not have the desktop_helper scope: `localrouter pair token NAME --scopes desktop_helper` makes one",
                .computer => "this device's token does not have the computer scope: `localrouter pair scopes NAME voice,mcp,computer` adds it",
                .approve => "this device's token does not have the approve scope: `localrouter pair scopes NAME voice,mcp,computer,approve` adds it",
            } };
            return .{ .ok = f.device };
        }
    };
}

/// Where a check's device id and name live: valid until the same thread's next check (callers copy them).
threadlocal var found_buf: [256]u8 = undefined;

/// A device as a lookup finds it: its id and name (in the lookup's buffer) and what its scopes allow.
pub const Found = struct {
    device: Device,
    desktop_helper: bool,
    computer: bool,
    approve: bool,

    fn allows(f: Found, s: Scope) bool {
        return switch (s) {
            .desktop_helper => f.desktop_helper,
            .computer => f.computer,
            .approve => f.approve,
        };
    }
};

test "without pairing every scope is off" {
    const g: Gate = .{};
    for ([_]Scope{ .desktop_helper, .computer, .approve }) |s| try std.testing.expect(g.check("any", s) == .off);
}

test "a paired device's scopes decide; no token or an unknown token are refused" {
    const Ctx = struct {
        fn lookup(_: *@This(), tok: []const u8, _: []u8) ?Found {
            if (std.mem.eql(u8, tok, "helper")) return .{ .device = .{ .id = "1", .name = "mac" }, .desktop_helper = true, .computer = false, .approve = false };
            if (std.mem.eql(u8, tok, "phone")) return .{ .device = .{ .id = "2", .name = "phone" }, .desktop_helper = false, .computer = true, .approve = true };
            return null;
        }
    };
    var ctx: Ctx = .{};
    const g = Paired(Ctx, Ctx.lookup).gate(&ctx);
    try std.testing.expect(g.check("helper", .desktop_helper) == .ok);
    try std.testing.expect(g.check("helper", .computer) == .denied);
    try std.testing.expect(g.check("helper", .approve) == .denied);
    try std.testing.expect(g.check("phone", .approve) == .ok);
    try std.testing.expect(g.check("phone", .desktop_helper) == .denied);
    try std.testing.expect(g.check("", .computer) == .denied);
    try std.testing.expect(g.check("stolen", .computer) == .denied);
}
