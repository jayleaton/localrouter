//! Per-op GPU timing of an H3 step (`stk check h3-step-bench`): a pair of CUDA events around each op, summed by op class
//! over the 50 blocks. The step is run eagerly (no step graph) with `Dit.prof` set; the Dit marks a span with `begin`
//! before an op's launches and `end` after them. A span's time is GPU time between its two events, so it holds the op's
//! kernels and any idle gap inside the span (a launch-bound op shows its gaps), but not the gaps between spans: the
//! caller measures the whole step with its own event pair and reports `total - sum(classes)` as the time between ops.

const std = @import("std");
const cuda = @import("cuda");

/// The op classes. `quant` is the NVFP4 activation quantizer in front of every block linear (all four summed), `gemm_*`
/// the prompt GEMM of that linear; `attn_prep` is the INT8 attention's K anchor, Q / K / V quantizers (comfy-kitchen's
/// launches 1 to 3), `attn_kernel` its attention kernel; `heads_rows` the [H, S, D] -> rows move (reference schedule
/// only: the fused schedule stores rows from the attention kernel); `gate_norm` is gate_add + the next norm_mod in one
/// kernel (fused schedule only). `carry` is the audio carry and un-carry, `embed` the patchifies, patch projections and
/// the [text | audio | video] assembly, `final` the final layer's modulation, heads and un-patchify.
pub const Class = enum { carry, embed, adaln, norm_mod, gate_add, gate_norm, quant, gemm_qkv, gemm_out, gemm_fc1, gemm_fc2, rms_rope, attn_prep, attn_kernel, heads_rows, swiglu, final };
pub const names = std.meta.fieldNames(Class);
pub const n_classes = names.len;

pub const Profile = struct {
    gpa: std.mem.Allocator,
    ev: []cuda.Event, // two per span: start, end
    cls: []Class,
    n: usize = 0,

    /// Room for `spans` spans a step (a block makes about 20; the sides some 20 more).
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, spans: usize) !Profile {
        const ev = try gpa.alloc(cuda.Event, 2 * spans);
        errdefer gpa.free(ev);
        var made: usize = 0;
        errdefer for (ev[0..made]) |*e| e.deinit();
        for (ev) |*e| {
            e.* = try cuda.Event.init(d, true);
            made += 1;
        }
        return .{ .gpa = gpa, .ev = ev, .cls = try gpa.alloc(Class, spans) };
    }

    pub fn deinit(p: *Profile) void {
        for (p.ev) |*e| e.deinit();
        p.gpa.free(p.ev);
        p.gpa.free(p.cls);
    }

    pub fn reset(p: *Profile) void {
        p.n = 0;
    }

    pub fn begin(p: *Profile, s: cuda.Stream) !void {
        if (p.n >= p.cls.len) return error.ProfileFull;
        try p.ev[2 * p.n].record(s);
    }

    pub fn end(p: *Profile, s: cuda.Stream, c: Class) !void {
        try p.ev[2 * p.n + 1].record(s);
        p.cls[p.n] = c;
        p.n += 1;
    }

    /// ms per class, summed over the recorded spans; the stream must have finished (synchronized).
    pub fn sums(p: *const Profile) ![n_classes]f64 {
        var out: [n_classes]f64 = @splat(0);
        for (0..p.n) |i| out[@intFromEnum(p.cls[i])] += try p.ev[2 * i].elapsedMs(p.ev[2 * i + 1]);
        return out;
    }
};

test "classes name the JSON keys: distinct, lower case or digits" {
    inline for (names, 0..) |f, i| {
        inline for (names[i + 1 ..]) |g| try std.testing.expect(!std.mem.eql(u8, f, g));
        for (f) |c| try std.testing.expect(std.ascii.isLower(c) or std.ascii.isDigit(c) or c == '_');
    }
    try std.testing.expectEqual(@as(usize, 17), n_classes);
}
