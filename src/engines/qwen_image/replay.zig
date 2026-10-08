//! The M3 bit gate: replays a twin capture through the Zig forward. Each op is matched to the capture by name, in
//! order. Alone: every op starts from the captured inputs (contiguous ones). Chained: from Zig's own outputs. Every
//! output is compared byte for byte with the capture. `localrouter check qwen-replay PACK CAPTURE` runs both.

const std = @import("std");
const cuda = @import("cuda");
const Capture = @import("capture.zig").Capture;
const Dit = @import("dit.zig").Dit;
const Io = @import("dit.zig").Io;
const noise = @import("noise.zig");

pub const Mode = enum { alone, chained };

pub const Stats = struct { equal: u32 = 0, differ: u32 = 0, skipped: u32 = 0, unmatched: u32 = 0 };

pub const Replay = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    s: cuda.Stream,
    cap: *const Capture,
    mode: Mode,
    next: usize = 0,
    cur: ?usize = null,
    stats: Stats = .{},
    first_diffs: std.ArrayList([]const u8) = .empty,
    host: std.ArrayList(u8) = .empty,
    want: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Replay) void {
        for (r.first_diffs.items) |m| r.gpa.free(m);
        r.first_diffs.deinit(r.gpa);
        r.host.deinit(r.gpa);
        r.want.deinit(r.gpa);
    }

    pub fn probe(r: *Replay) @import("dit.zig").Probe {
        return .{ .ctx = r, .before = before, .after = after };
    }

    fn self(ctx: *anyopaque) *Replay {
        return @ptrCast(@alignCast(ctx));
    }

    fn before(ctx: *anyopaque, name: []const u8, ins: []const Io) anyerror!void {
        const r = self(ctx);
        r.cur = r.cap.find(name, r.next);
        const i = r.cur orelse {
            r.stats.unmatched += 1;
            return;
        };
        if (r.mode != .alone) return;
        const op = &r.cap.ops[i];
        for (ins) |in| {
            const ref = op.in(in.role) orelse continue;
            if (!in.contiguous or ref.bytes() != in.bytes) continue;
            try r.host.resize(r.gpa, in.bytes);
            try r.cap.blob(r.io, ref, r.host.items);
            try r.s.synchronize();
            try r.d.check(r.d.api.cuMemcpyHtoD_v2(in.ptr, r.host.items.ptr, in.bytes), "cuMemcpyHtoD");
        }
    }

    fn after(ctx: *anyopaque, name: []const u8, outs: []const Io) anyerror!void {
        const r = self(ctx);
        const i = r.cur orelse return;
        r.next = i + 1;
        const op = &r.cap.ops[i];
        try r.s.synchronize();
        for (outs) |out| {
            const ref = op.out(out.role) orelse {
                r.stats.skipped += 1;
                continue;
            };
            if (ref.bytes() != out.bytes) {
                r.stats.skipped += 1;
                try r.note("{s}.{s}: size {d} vs captured {d}", .{ name, out.role, out.bytes, ref.bytes() });
                continue;
            }
            try r.host.resize(r.gpa, out.bytes);
            try r.want.resize(r.gpa, out.bytes);
            try r.d.check(r.d.api.cuMemcpyDtoH_v2(r.host.items.ptr, out.ptr, out.bytes), "cuMemcpyDtoH");
            try r.cap.blob(r.io, ref, r.want.items);
            if (std.mem.eql(u8, r.host.items, r.want.items)) {
                r.stats.equal += 1;
            } else {
                r.stats.differ += 1;
                var n: usize = 0;
                for (r.host.items, r.want.items) |a, b| n += @intFromBool(a != b);
                try r.note("{s}.{s} ({s}): {d} of {d} bytes differ", .{ name, out.role, op.kind, n, out.bytes });
            }
        }
    }

    fn note(r: *Replay, comptime fmt: []const u8, args: anytype) !void {
        if (r.first_diffs.items.len >= 40) return;
        try r.first_diffs.append(r.gpa, try std.fmt.allocPrint(r.gpa, fmt, args));
    }
};

/// The capture's request: seed, size, sigmas and the text context (the text encoder's output).
pub const Request = struct { seed: u64, height: u32, width: u32, sigmas: []const f64, ctx_rows: u32 };

pub fn request(cap: *const Capture, a: std.mem.Allocator) !Request {
    const sig = cap.note("sigmas") orelse return error.NoSigmas;
    const o = sig.object;
    const vals = o.get("values").?.array.items;
    const sigmas = try a.alloc(f64, vals.len);
    for (vals, sigmas) |v, *s| s.* = switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => return error.BadSigmas,
    };
    const te = cap.ops[cap.find("text_encoder", 0) orelse return error.NoContext].out("context").?;
    return .{
        .seed = @intCast(o.get("seed").?.integer),
        .height = @intCast(o.get("height").?.integer),
        .width = @intCast(o.get("width").?.integer),
        .sigmas = sigmas,
        .ctx_rows = @intCast(te.shape[1]),
    };
}

/// One pass of step 0 (prefix, step, Euler) under `mode`; the starting noise is Zig's own and is compared too.
pub fn run(r: *Replay, dit: *Dit, req: Request, ctx: u64, lat: u64, vel: u64, out: u64) !void {
    const h = req.height / 16;
    const w = req.width / 16;
    const n: usize = 64 * @as(usize, h) * w;
    const x0 = try r.gpa.alloc(f32, n);
    defer r.gpa.free(x0);
    noise.fill(req.seed, x0);
    const bf = try r.gpa.alloc(u16, n);
    defer r.gpa.free(bf);
    for (x0, bf) |v, *b| b.* = bf16(v);
    if (r.cap.find("step0.euler", 0)) |ei| { // the starting noise: Zig's generator against the twin's
        const ref = r.cap.ops[ei].in("x").?;
        try r.want.resize(r.gpa, ref.bytes());
        try r.cap.blob(r.io, ref, r.want.items);
        if (std.mem.eql(u8, r.want.items, std.mem.sliceAsBytes(bf))) r.stats.equal += 1 else {
            r.stats.differ += 1;
            try r.note("noise: the starting latents differ", .{});
        }
    }
    try dit.upload(lat, std.mem.sliceAsBytes(bf));
    dit.probe = r.probe();
    dit.mod_sigma = null;
    const s0: f32 = @floatCast(req.sigmas[0]);
    try dit.buildPrefix(ctx, req.ctx_rows, s0);
    try dit.step(lat, s0, h, w, vel);
    try dit.euler("step0.euler", lat, vel, out, n, @floatCast(req.sigmas[1] - req.sigmas[0]));
}

/// f32 -> bf16, round to nearest even (as torch's `.to(torch.bfloat16)`; the noise is finite).
fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @truncate((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}
