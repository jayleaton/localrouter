//! Device pairing (`pairing.required`): the devices that may open /v1/realtime and /mcp, and the one-time codes that
//! pair a new one. They live in `<data_dir>/devices.json` (0600), read again on every check so a revoke from the CLI
//! takes effect on the next request:
//!   {"devices":[{"id":"8 hex","name":"JJ's phone","token_sha256":"hex","created":unix_s}],
//!    "pending":[{"code_sha256":"hex","name":"JJ's phone","expires":unix_s}]}
//! Only SHA-256 hashes are kept: a token (32 random bytes, base64url without padding: 43 characters, valid as a
//! WebSocket subprotocol token) is shown once; a code (8 Crockford base32 characters, shown as XXXX-XXXX) is good for
//! 10 minutes, once. Every change is a read-modify-write under an exclusive lock on `devices.lock` beside the file,
//! written to a temporary file and renamed over it, so a reader sees the old file or the new one.

const std = @import("std");
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const code_ttl_s: i64 = 600;
pub const max_name_len = 64;
/// Crockford base32: no I, L, O or U, so a code read aloud or off a screen is not misread.
pub const alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
pub const token_len = 43;

pub const Device = struct { id: []const u8, name: []const u8, token_sha256: []const u8, created: i64 };
pub const Pending = struct { code_sha256: []const u8, name: []const u8, expires: i64 };
pub const Doc = struct { devices: []const Device = &.{}, pending: []const Pending = &.{} };

/// A code as shown, `XXXX-XXXX`, and when it stops working.
pub const Code = struct { text: [9]u8, expires: i64 };
/// A new device and its token (the only time the token exists outside the client).
pub const Minted = struct { token: [token_len]u8, id: [8]u8, name: []const u8 };

pub fn now(io: Io) i64 {
    return @intCast(@divTrunc(Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_s));
}

/// The devices file of one data directory.
pub const Store = struct {
    dir: []const u8,

    fn path(s: Store, a: std.mem.Allocator, name: []const u8) ![]const u8 {
        return std.fs.path.join(a, &.{ s.dir, name });
    }

    /// The file's contents; none yet is an empty list.
    pub fn load(s: Store, io: Io, a: std.mem.Allocator) !Doc {
        const text = Io.Dir.cwd().readFileAlloc(io, try s.path(a, "devices.json"), a, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => return .{},
            else => return err,
        };
        return std.json.parseFromSliceLeaky(Doc, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    }

    /// Takes the exclusive lock; close the file to release it.
    fn lock(s: Store, io: Io, a: std.mem.Allocator) !Io.File {
        try Io.Dir.cwd().createDirPath(io, s.dir);
        return Io.Dir.cwd().createFile(io, try s.path(a, "devices.lock"), .{ .truncate = false, .lock = .exclusive, .permissions = .fromMode(0o600) });
    }

    /// Writes `doc` (expired codes dropped) to a temporary file and renames it over the devices file.
    fn save(s: Store, io: Io, a: std.mem.Allocator, doc: Doc, at: i64) !void {
        var pending: std.ArrayList(Pending) = .empty;
        for (doc.pending) |p| if (p.expires > at) try pending.append(a, p);
        const text = try std.json.Stringify.valueAlloc(a, Doc{ .devices = doc.devices, .pending = pending.items }, .{});
        const tmp = try s.path(a, "devices.json.tmp");
        const cwd = Io.Dir.cwd();
        cwd.deleteFile(io, tmp) catch {}; // a leftover keeps its mode; a new file gets 0600
        const f = try cwd.createFile(io, tmp, .{ .permissions = .fromMode(0o600) });
        {
            defer f.close(io);
            try f.writeStreamingAll(io, text);
            try f.sync(io);
        }
        try cwd.rename(tmp, cwd, try s.path(a, "devices.json"), io);
    }

    /// A new pending code for a device to be called `name`.
    pub fn addCode(s: Store, io: Io, a: std.mem.Allocator, name: []const u8, at: i64) !Code {
        const l = try s.lock(io, a);
        defer l.close(io);
        const doc = try s.load(io, a);
        var raw: [8]u8 = undefined;
        newCode(io, &raw);
        const pending = try a.alloc(Pending, doc.pending.len + 1);
        @memcpy(pending[0..doc.pending.len], doc.pending);
        const c: Code = .{ .text = shown(raw), .expires = at + code_ttl_s };
        pending[doc.pending.len] = .{ .code_sha256 = try hexHash(a, &raw), .name = name, .expires = c.expires };
        try s.save(io, a, .{ .devices = doc.devices, .pending = pending }, at);
        return c;
    }

    /// A new device called `name`, with its token.
    pub fn mint(s: Store, io: Io, a: std.mem.Allocator, name: []const u8, at: i64) !Minted {
        const l = try s.lock(io, a);
        defer l.close(io);
        const doc = try s.load(io, a);
        return s.addDevice(io, a, doc, doc.pending, name, at);
    }

    fn addDevice(s: Store, io: Io, a: std.mem.Allocator, doc: Doc, pending: []const Pending, name: []const u8, at: i64) !Minted {
        var m: Minted = .{ .token = undefined, .id = undefined, .name = name };
        var bytes: [32]u8 = undefined;
        io.random(&bytes);
        _ = std.base64.url_safe_no_pad.Encoder.encode(&m.token, &bytes);
        while (true) { // ids are short: draw again on the rare clash
            var r: [4]u8 = undefined;
            io.random(&r);
            m.id = std.fmt.bytesToHex(r, .lower);
            for (doc.devices) |d| {
                if (std.mem.eql(u8, d.id, &m.id)) break;
            } else break;
        }
        const devices = try a.alloc(Device, doc.devices.len + 1);
        @memcpy(devices[0..doc.devices.len], doc.devices);
        devices[doc.devices.len] = .{ .id = try a.dupe(u8, &m.id), .name = name, .token_sha256 = try hexHash(a, &m.token), .created = at };
        try s.save(io, a, .{ .devices = devices, .pending = pending }, at);
        return m;
    }

    /// Trades a pending code for a device (named `name`, else the code's name); null when the code is wrong, expired
    /// or used. The code is gone either way once it has worked.
    pub fn redeem(s: Store, io: Io, a: std.mem.Allocator, text: []const u8, name: []const u8, at: i64) !?Minted {
        const raw = normalize(text) orelse return null;
        var want: [32]u8 = undefined;
        Sha256.hash(&raw, &want, .{});
        const l = try s.lock(io, a);
        defer l.close(io);
        const doc = try s.load(io, a);
        var hit: ?usize = null;
        for (doc.pending, 0..) |p, i| {
            if (sameHash(want, p.code_sha256) and p.expires > at) hit = i; // every entry is compared: no early exit
        }
        const i = hit orelse return null;
        const rest = try a.alloc(Pending, doc.pending.len - 1);
        @memcpy(rest[0..i], doc.pending[0..i]);
        @memcpy(rest[i..], doc.pending[i + 1 ..]);
        return try s.addDevice(io, a, doc, rest, if (name.len > 0) name else doc.pending[i].name, at);
    }

    /// The device whose token this is, or null.
    pub fn verify(s: Store, io: Io, a: std.mem.Allocator, token: []const u8) !?Device {
        var got: [32]u8 = undefined;
        Sha256.hash(token, &got, .{});
        var found: ?Device = null;
        for ((try s.load(io, a)).devices) |d| {
            if (sameHash(got, d.token_sha256)) found = d;
        }
        return found;
    }

    /// Removes the devices whose id or name is `key`; the ones removed.
    pub fn revoke(s: Store, io: Io, a: std.mem.Allocator, key: []const u8, at: i64) ![]const Device {
        const l = try s.lock(io, a);
        defer l.close(io);
        const doc = try s.load(io, a);
        var keep: std.ArrayList(Device) = .empty;
        var gone: std.ArrayList(Device) = .empty;
        for (doc.devices) |d| try (if (std.mem.eql(u8, d.id, key) or std.mem.eql(u8, d.name, key)) &gone else &keep).append(a, d);
        if (gone.items.len > 0) try s.save(io, a, .{ .devices = keep.items, .pending = doc.pending }, at);
        return gone.items;
    }

    /// Every pending code stops working (the brute-force guard's answer).
    pub fn discardPending(s: Store, io: Io, a: std.mem.Allocator, at: i64) !void {
        const l = try s.lock(io, a);
        defer l.close(io);
        const doc = try s.load(io, a);
        try s.save(io, a, .{ .devices = doc.devices }, at);
    }
};

/// Counts wrong codes: the fifth within 10 minutes says to discard every pending code, so guessing gets 5 tries a code.
pub const Guard = struct {
    pub const limit = 5;
    mutex: Io.Mutex = .init,
    times: [limit]i64 = @splat(std.math.minInt(i64) / 2), // the last `limit` wrong codes, a ring
    next: usize = 0,

    /// Records a wrong code at `at`; true when the pending codes must go (the count starts again).
    pub fn wrong(g: *Guard, io: Io, at: i64) bool {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        g.times[g.next] = at;
        g.next = (g.next + 1) % limit;
        if (at - g.times[g.next] >= code_ttl_s) return false; // the oldest of the last five is older than 10 minutes
        g.times = @splat(std.math.minInt(i64) / 2);
        return true;
    }
};

/// 8 random characters from `alphabet` (256 is a multiple of 32: each is uniform).
pub fn newCode(io: Io, out: *[8]u8) void {
    var r: [8]u8 = undefined;
    io.random(&r);
    for (out, r) |*c, b| c.* = alphabet[b & 31];
}

/// `XXXX-XXXX`.
pub fn shown(raw: [8]u8) [9]u8 {
    return (raw[0..4] ++ "-" ++ raw[4..8]).*;
}

/// A code as typed: any case, with or without the dash or spaces; O read as 0 and I or L as 1 (Crockford's decoding).
/// Null when it is not 8 characters of the alphabet.
pub fn normalize(text: []const u8) ?[8]u8 {
    var out: [8]u8 = undefined;
    var n: usize = 0;
    for (text) |ch| {
        if (ch == '-' or ch == ' ') continue;
        const c = switch (std.ascii.toUpper(ch)) {
            'O' => '0',
            'I', 'L' => '1',
            else => |u| u,
        };
        if (std.mem.indexOfScalar(u8, alphabet, c) == null or n == out.len) return null;
        out[n] = c;
        n += 1;
    }
    return if (n == out.len) out else null;
}

fn hexHash(a: std.mem.Allocator, data: []const u8) ![]const u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(data, &d, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(d, .lower));
}

/// Whether `hex` (a stored hash) is `digest`, compared in constant time.
pub fn sameHash(digest: [32]u8, hex: []const u8) bool {
    var stored: [32]u8 = undefined;
    if (hex.len != 64) return false;
    _ = std.fmt.hexToBytes(&stored, hex) catch return false;
    return std.crypto.timing_safe.eql([32]u8, digest, stored);
}

/// `2026-10-23 14:05 UTC` in `buf`.
pub fn utc(buf: []u8, secs: i64) []const u8 {
    const es: std.time.epoch.EpochSeconds = .{ .secs = @intCast(@max(secs, 0)) };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2} UTC", .{ yd.year, md.month.numeric(), @as(u32, md.day_index) + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour() }) catch "?";
}

const testing = std.testing;

/// A store in a fresh directory under /tmp, removed by `deinit`.
const TestStore = struct {
    store: Store,

    fn init(a: std.mem.Allocator) !TestStore {
        var r: [8]u8 = undefined;
        testing.io.random(&r);
        return .{ .store = .{ .dir = try std.fmt.allocPrint(a, "/tmp/localrouter-pair-test-{x:0>16}", .{std.mem.readInt(u64, &r, .little)}) } };
    }

    fn deinit(t: TestStore) void {
        Io.Dir.cwd().deleteTree(testing.io, t.store.dir) catch {};
    }
};

test "codes: 8 characters of Crockford base32, shown XXXX-XXXX, typed any way" {
    for (alphabet) |c| try testing.expect(std.mem.indexOfScalar(u8, "ILOU", c) == null);
    try testing.expectEqual(@as(usize, 32), alphabet.len);
    for (0..50) |_| {
        var raw: [8]u8 = undefined;
        newCode(testing.io, &raw);
        for (raw) |c| try testing.expect(std.mem.indexOfScalar(u8, alphabet, c) != null);
        const s = shown(raw);
        try testing.expectEqual(@as(u8, '-'), s[4]);
        try testing.expectEqualSlices(u8, &raw, &normalize(&s).?);
    }
    try testing.expectEqualStrings("ABCD2345", &normalize("abcd-2345").?);
    try testing.expectEqualStrings("ABCD2345", &normalize("ABCD2345").?);
    try testing.expectEqualStrings("A0C11345", &normalize(" aoci-l345 ").?);
    try testing.expect(normalize("ABCD-234") == null);
    try testing.expect(normalize("ABCD-23456") == null);
    try testing.expect(normalize("ABCD-23U5") == null);
    try testing.expect(normalize("") == null);
}

test "hashes compare in constant time and refuse what is not a hash" {
    var d: [32]u8 = undefined;
    Sha256.hash("x", &d, .{});
    const hex = std.fmt.bytesToHex(d, .lower);
    try testing.expect(sameHash(d, &hex));
    var other = hex;
    other[63] = if (other[63] == '0') '1' else '0';
    try testing.expect(!sameHash(d, &other));
    try testing.expect(!sameHash(d, hex[0..62]));
    try testing.expect(!sameHash(d, "zz" ++ hex[2..]));
}

test "a code pairs a device once; its token verifies until it is revoked; codes expire" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    const t = try TestStore.init(a);
    defer t.deinit();
    const s = t.store;
    try testing.expectEqual(@as(usize, 0), (try s.load(io, a)).devices.len);

    const c = try s.addCode(io, a, "JJ's phone", 1000);
    try testing.expectEqual(@as(i64, 1600), c.expires);
    const doc = try s.load(io, a);
    try testing.expectEqual(@as(usize, 1), doc.pending.len);
    try testing.expect(std.mem.indexOf(u8, doc.pending[0].code_sha256, &c.text) == null); // a hash, not the code

    var lower = c.text;
    for (&lower) |*ch| ch.* = std.ascii.toLower(ch.*);
    const m = (try s.redeem(io, a, &lower, "", 1100)).?;
    try testing.expectEqualStrings("JJ's phone", m.name); // the code's name
    try testing.expectEqual(@as(usize, token_len), m.token.len);
    for (m.token) |ch| try testing.expect(std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '_');
    try testing.expect(try s.redeem(io, a, &c.text, "", 1101) == null); // used
    const d = (try s.verify(io, a, &m.token)).?;
    try testing.expectEqualStrings(&m.id, d.id);
    try testing.expect(try s.verify(io, a, "not-a-token") == null);
    const file = try Io.Dir.cwd().readFileAlloc(io, try s.path(a, "devices.json"), a, .limited(1 << 20));
    try testing.expect(std.mem.indexOf(u8, file, &m.token) == null);
    const st = try Io.Dir.cwd().statFile(io, try s.path(a, "devices.json"), .{});
    try testing.expectEqual(@as(std.posix.mode_t, 0o600), st.permissions.toMode() & 0o777);

    const late = try s.addCode(io, a, "tablet", 2000);
    try testing.expect(try s.redeem(io, a, &late.text, "", 2000 + code_ttl_s) == null); // expired
    const named = try s.addCode(io, a, "tablet", 3000);
    const n = (try s.redeem(io, a, &named.text, "kitchen", 3001)).?;
    try testing.expectEqualStrings("kitchen", n.name);

    const direct = try s.mint(io, a, "claude-code", 3002);
    try testing.expectEqual(@as(usize, 3), (try s.load(io, a)).devices.len);
    try testing.expectEqual(@as(usize, 1), (try s.revoke(io, a, &m.id, 3003)).len);
    try testing.expect(try s.verify(io, a, &m.token) == null);
    try testing.expectEqual(@as(usize, 1), (try s.revoke(io, a, "claude-code", 3004)).len);
    try testing.expect(try s.verify(io, a, &direct.token) == null);
    try testing.expectEqual(@as(usize, 0), (try s.revoke(io, a, "nobody", 3005)).len);
    try testing.expect(try s.verify(io, a, &n.token) != null);
}

test "five wrong codes within 10 minutes discard the pending codes" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var g: Guard = .{};
    for (0..4) |i| try testing.expect(!g.wrong(io, 100 + @as(i64, @intCast(i))));
    try testing.expect(g.wrong(io, 104));
    try testing.expect(!g.wrong(io, 105)); // the count starts again
    // Slow guessing (one each 3 minutes) never trips it.
    var slow: Guard = .{};
    for (0..10) |i| try testing.expect(!slow.wrong(io, @as(i64, @intCast(i)) * 180));

    const t = try TestStore.init(a);
    defer t.deinit();
    const c = try t.store.addCode(io, a, "phone", 100);
    try t.store.discardPending(io, a, 101);
    try testing.expectEqual(@as(usize, 0), (try t.store.load(io, a)).pending.len);
    try testing.expect(try t.store.redeem(io, a, &c.text, "", 102) == null);
}

test "utc dates" {
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("1970-01-01 00:00 UTC", utc(&buf, 0));
    try testing.expectEqualStrings("2026-10-23 14:05 UTC", utc(&buf, 1792764300));
}
