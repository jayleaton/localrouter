//! TensorFold v0.6.1's NVFP4 kernels (extension tensorfold_nvfp4_ck_v6) as fatbins, with the mangled symbol of every
//! instantiation. Sources: kernels/cuda/nvfp4 (tools/kernels/sync.py); built by build/cuda.zig. Without nvcc the images
//! are empty (the host-only build, like TensorFold's `with_kernels = false`).

const std = @import("std");
const options = @import("nvfp4_options");

pub const with_kernels: bool = options.with_kernels;

/// One fatbin per .cu (SASS for sm_121a and sm_120a); empty when built without nvcc.
fn Blob(comptime name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(name).*; // the driver loads images from 8-byte aligned memory
    };
}

pub const act: []const u8 = if (with_kernels) &Blob("nvfp4_fatbin_act").bytes else &.{};
pub const gemm_ck: []const u8 = if (with_kernels) &Blob("nvfp4_fatbin_gemm_ck").bytes else &.{};
pub const gemm_ws: []const u8 = if (with_kernels) &Blob("nvfp4_fatbin_gemm_ws").bytes else &.{};
pub const lane4: []const u8 = if (with_kernels) &Blob("nvfp4_fatbin_lane4").bytes else &.{};

/// act.cu: NVFP4 / FP8 row quantizers (quant4_cuda, quant8_cuda).
pub const quant4: [:0]const u8 = "_ZN9nvfp4_act13quant4_kernelEPK13__nv_bfloat16iiifPhS3_ii";
pub const quant8: [:0]const u8 = "_ZN9nvfp4_act13quant8_kernelEPK13__nv_bfloat16iiifPhii";
/// lane4.cu: checkpoint bytes to the lane words (pack4_cuda). The lane matmul itself is not built.
pub const pack4: [:0]const u8 = "_ZN11nvfp4_lane412pack4_kernelEPKhiiPj";

/// gemm_ck.cu's gemm_kernel<MODE, BM, BN, WM, WN, STAGES, KS, F32, EPI> (module `gemm_ck`). MODE 0 = NVFP4, 1 = FP8.
pub const Ck = struct { mode: u32, bm: u32, bn: u32, wm: u32, wn: u32, stages: u32, ks: u32, f32: bool, epi: u32, symbol: [:0]const u8 };
/// gemm_ws.cu's ws_kernel<MODE, BM, BN, STAGES, KS, F32> (module `gemm_ws`).
pub const Ws = struct { mode: u32, bm: u32, bn: u32, stages: u32, ks: u32, f32: bool, symbol: [:0]const u8 };

/// The instance of gemm_kernel with these template arguments, or null (it is not built).
pub fn findCk(mode: u32, bm: u32, bn: u32, wm: u32, wn: u32, stages: u32, ks: u32, is_f32: bool, epi: u32) ?Ck {
    for (ck_instances) |c| {
        if (c.mode == mode and c.bm == bm and c.bn == bn and c.wm == wm and c.wn == wn and c.stages == stages and
            c.ks == ks and c.f32 == is_f32 and c.epi == epi) return c;
    }
    return null;
}

/// The instance of ws_kernel with these template arguments, or null.
pub fn findWs(mode: u32, bm: u32, bn: u32, stages: u32, ks: u32, is_f32: bool) ?Ws {
    for (ws_instances) |w| {
        if (w.mode == mode and w.bm == bm and w.bn == bn and w.stages == stages and w.ks == ks and w.f32 == is_f32) return w;
    }
    return null;
}

pub const ck_instances = [_]Ck{
    .{ .mode = 0, .bm = 128, .bn = 128, .wm = 2, .wn = 4, .stages = 4, .ks = 1, .f32 = false, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi128ELi2ELi4ELi4ELi1ELb0ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 0, .bm = 128, .bn = 128, .wm = 2, .wn = 4, .stages = 4, .ks = 1, .f32 = true, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi128ELi2ELi4ELi4ELi1ELb1ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 0, .bm = 128, .bn = 256, .wm = 2, .wn = 4, .stages = 2, .ks = 2, .f32 = false, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi256ELi2ELi4ELi2ELi2ELb0ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 0, .bm = 128, .bn = 256, .wm = 2, .wn = 4, .stages = 2, .ks = 2, .f32 = true, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi256ELi2ELi4ELi2ELi2ELb1ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 0, .bm = 64, .bn = 128, .wm = 2, .wn = 4, .stages = 4, .ks = 1, .f32 = false, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi64ELi128ELi2ELi4ELi4ELi1ELb0ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 0, .bm = 64, .bn = 128, .wm = 2, .wn = 4, .stages = 4, .ks = 1, .f32 = true, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi64ELi128ELi2ELi4ELi4ELi1ELb1ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 1, .bm = 128, .bn = 128, .wm = 2, .wn = 4, .stages = 3, .ks = 1, .f32 = false, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi1ELi128ELi128ELi2ELi4ELi3ELi1ELb0ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 1, .bm = 128, .bn = 128, .wm = 2, .wn = 4, .stages = 3, .ks = 1, .f32 = true, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi1ELi128ELi128ELi2ELi4ELi3ELi1ELb1ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 1, .bm = 128, .bn = 256, .wm = 2, .wn = 4, .stages = 2, .ks = 2, .f32 = false, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi1ELi128ELi256ELi2ELi4ELi2ELi2ELb0ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 1, .bm = 128, .bn = 256, .wm = 2, .wn = 4, .stages = 2, .ks = 2, .f32 = true, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi1ELi128ELi256ELi2ELi4ELi2ELi2ELb1ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 1, .bm = 64, .bn = 128, .wm = 2, .wn = 4, .stages = 4, .ks = 1, .f32 = false, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi1ELi64ELi128ELi2ELi4ELi4ELi1ELb0ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 1, .bm = 64, .bn = 128, .wm = 2, .wn = 4, .stages = 4, .ks = 1, .f32 = true, .epi = 0, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi1ELi64ELi128ELi2ELi4ELi4ELi1ELb1ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 0, .bm = 128, .bn = 256, .wm = 2, .wn = 4, .stages = 3, .ks = 2, .f32 = false, .epi = 1, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi256ELi2ELi4ELi3ELi2ELb0ELi1EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
    .{ .mode = 0, .bm = 128, .bn = 256, .wm = 2, .wn = 4, .stages = 3, .ks = 2, .f32 = false, .epi = 2, .symbol = "_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi256ELi2ELi4ELi3ELi2ELb0ELi2EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE" },
};

pub const ws_instances = [_]Ws{
    .{ .mode = 0, .bm = 128, .bn = 256, .stages = 4, .ks = 1, .f32 = false, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi0ELi128ELi256ELi4ELi1ELb0EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 0, .bm = 128, .bn = 256, .stages = 4, .ks = 1, .f32 = true, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi0ELi128ELi256ELi4ELi1ELb1EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 0, .bm = 128, .bn = 256, .stages = 3, .ks = 2, .f32 = false, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi0ELi128ELi256ELi3ELi2ELb0EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 0, .bm = 128, .bn = 256, .stages = 3, .ks = 2, .f32 = true, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi0ELi128ELi256ELi3ELi2ELb1EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 0, .bm = 128, .bn = 128, .stages = 3, .ks = 2, .f32 = false, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi0ELi128ELi128ELi3ELi2ELb0EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 0, .bm = 128, .bn = 128, .stages = 3, .ks = 2, .f32 = true, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi0ELi128ELi128ELi3ELi2ELb1EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 1, .bm = 128, .bn = 256, .stages = 3, .ks = 1, .f32 = false, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi1ELi128ELi256ELi3ELi1ELb0EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 1, .bm = 128, .bn = 256, .stages = 3, .ks = 1, .f32 = true, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi1ELi128ELi256ELi3ELi1ELb1EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 1, .bm = 128, .bn = 256, .stages = 2, .ks = 2, .f32 = false, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi1ELi128ELi256ELi2ELi2ELb0EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 1, .bm = 128, .bn = 256, .stages = 2, .ks = 2, .f32 = true, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi1ELi128ELi256ELi2ELi2ELb1EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 1, .bm = 128, .bn = 128, .stages = 3, .ks = 2, .f32 = false, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi1ELi128ELi128ELi3ELi2ELb0EEEvPKhS2_S2_S2_fPviiiiii" },
    .{ .mode = 1, .bm = 128, .bn = 128, .stages = 3, .ks = 2, .f32 = true, .symbol = "_ZN13nvfp4_gemm_ws9ws_kernelILi1ELi128ELi128ELi3ELi2ELb1EEEvPKhS2_S2_S2_fPviiiiii" },
};

test "every instantiation is listed once" {
    for (ck_instances, 0..) |a, i| for (ck_instances[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a.symbol, b.symbol));
    for (ws_instances, 0..) |a, i| for (ws_instances[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a.symbol, b.symbol));
    try std.testing.expectEqual(@as(usize, 14), ck_instances.len);
    try std.testing.expectEqual(@as(usize, 12), ws_instances.len);
    try std.testing.expect(findCk(0, 128, 256, 2, 4, 2, 2, false, 0) != null);
    try std.testing.expect(findWs(0, 128, 256, 3, 2, false) != null);
    try std.testing.expect(findWs(0, 128, 256, 3, 2, true) != null);
}
