//! Launch plans for TensorFold v0.6.1's NVFP4 prompt path (extension tensorfold_nvfp4_ck_v6): pure host functions that
//! make each C++ wrapper's launch decision (symbol, grid, block, dynamic shared memory, argument order) and the
//! Python helpers' padding and scale rules (checkpoint.py). No CUDA calls: the launcher only walks the result.
//! Sources: act.cu, lane4.cu, gemm_ck.cu, gemm_ws.cu, checkpoint.cpp (the TORCH_CHECKs become errors), checkpoint.py.

const std = @import("std");
const kernels = @import("nvfp4_kernels");

/// checkpoint.WS: prompt tiles past this run the warp-specialized (bulk-copy) GEMM.
pub const WS: u32 = 10;
/// checkpoint.TB: that GEMM's tile rows, and the quantizer's tile rows for it.
pub const TB: u32 = 128;
/// The L2 budget gemm_ck.cu / gemm_ws.cu `launch` uses to band row tiles (bytes).
pub const l2_bytes: u64 = 12 << 20;
/// Static + dynamic shared memory a block may use without opt-in (cuFuncSetAttribute above this).
pub const smem_default: u32 = 48 * 1024;

/// mma4::Mode: A4 = NVFP4 rows x NVFP4 weights (block-scaled FP4 mma), A8 = FP8 x FP8.
pub const Mode = enum(u32) { a4 = 0, a8 = 1 };

pub const Dim3 = struct { x: u32, y: u32 = 1, z: u32 = 1 };

/// Which fatbin (kernels.act, .gemm_ck, .gemm_ws, .lane4) holds the symbol.
pub const Module = enum { act, gemm_ck, gemm_ws, lane4 };

/// The scalars of gemm_ck.cu's `Up` (the pointers w, ws, codes, scales come from the launcher's buffers).
pub const UpScalars = struct { alpha: f32, qg: f32, kd: i32 };

/// gemm_ck.cu's `struct Up` by value: {w, ws, alpha, codes, scales, qg, kd}. EPI 0 kernels take it zeroed.
pub const Up = extern struct { w: u64, ws: u64, alpha: f32, codes: u64, scales: u64, qg: f32, kd: i32 };
comptime {
    std.debug.assert(@sizeOf(Up) == 48 and @offsetOf(Up, "codes") == 24 and @offsetOf(Up, "kd") == 44);
}

/// One kernel argument, in parameter order. Roles are device pointers the launcher owns; `null_ptr` is a nullptr
/// (xs / ws for FP8, `out` of the fused SwiGLU launch); `up_zero` / `up` are the 48-byte `Up` struct.
pub const Arg = union(enum) {
    x,
    xs,
    w,
    ws,
    out,
    codes,
    scales,
    src,
    dst,
    null_ptr,
    up_zero,
    up: UpScalars,
    i32: i32,
    f32: f32,
};

pub const Launch = struct {
    symbol: [:0]const u8,
    module: Module,
    grid: Dim3,
    block: Dim3,
    /// Dynamic shared memory bytes (no static shared memory in these kernels).
    smem: u32,
    args: [13]Arg = undefined,
    nargs: usize = 0,

    pub fn slice(self: *const Launch) []const Arg {
        return self.args[0..self.nargs];
    }

    /// cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, smem) before the first launch when this holds.
    pub fn needsSmemOptIn(self: *const Launch) bool {
        return self.smem > smem_default;
    }

    fn push(self: *Launch, a: Arg) void {
        self.args[self.nargs] = a;
        self.nargs += 1;
    }
};

pub const Error = error{
    KNotMultipleOf64,
    KNotMultipleOf128,
    KNotWholeStages,
    NpadNotMultipleOf64,
    NpadBelowN,
    MpadNotTiled,
    MpadBelowM,
    TbNotMultipleOf16,
    TooManyRows,
    MissingKernel,
};

fn ceilDiv(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

fn u(v: u64) u32 {
    return @intCast(v);
}

fn i(v: u64) Arg {
    return .{ .i32 = @intCast(v) };
}

// ---- checkpoint.py helpers ----

/// checkpoint.bulk_tile: whether prompt `tile` runs the bulk-copy GEMM (tile past WS, capability >= (9, 0)).
pub fn bulkTile(tile: u32, major: u32, minor: u32) bool {
    _ = minor; // (major, minor) >= (9, 0) holds for any minor
    return tile > WS and major >= 9;
}

/// checkpoint._tb: the quantizer's tile rows for a prompt `tile` (TB for the bulk-copy GEMM, else 0 = row-major).
pub fn tb(tile: u32, major: u32, minor: u32) u32 {
    return if (bulkTile(tile, major, minor)) TB else 0;
}

/// The tile argument `_gemm` passes on: `tile % WS` past WS, else `tile` (12 -> 2: ws_kernel's 128 x 256, two steps).
pub fn tileArg(tile: u32) u32 {
    return if (tile > WS) tile % WS else tile;
}

/// checkpoint._inv: 1 / input_scale in fp32 (the scale is rounded to fp32 first); quant4's `g`, gemm_gu_ck's `qg`.
pub fn inv(act: f64) f32 {
    const a: f32 = @floatCast(act);
    return @as(f32, 1.0) / a;
}

/// checkpoint.alpha: the input scale times the weight's, multiplied in fp32.
pub fn alpha(act: f64, scale: f64) f32 {
    const a: f32 = @floatCast(act);
    const s: f32 = @floatCast(scale);
    return a * s;
}

/// GB10 is sm_121: gemm_cuda's tile 0 differs there (`props->major == 12 && props->minor == 1`).
pub fn isGb10(major: u32, minor: u32) bool {
    return major == 12 and minor == 1;
}

/// What quant4 / quant8 allocate for m rows of k inputs at tile rows `tb_rows` (0 or TB).
pub const Rows = struct {
    /// Padded rows: m rounded up to tb_rows, or to 64 when row-major.
    mpad: u32,
    /// NVFP4 codes bytes: (mpad if tiled else m) * k / 2. FP8 bytes: (mpad if tiled else m) * k.
    codes_bytes: u64,
    fp8_bytes: u64,
    /// NVFP4 row scales bytes: (k / 64) * mpad * 4, laid out [k/64, mpad, 4] or tiled [mpad/tb, k/64, tb, 4].
    scales_bytes: u64,
    /// Row-major scales (tb 0) must be zeroed first (torch.zeros): the padding rows are never written.
    zero_scales: bool,
};

pub fn rows(m: u32, k: u32, tb_rows: u32) Rows {
    const step: u64 = if (tb_rows != 0) tb_rows else 64;
    const mpad = ceilDiv(m, step) * step;
    const stored: u64 = if (tb_rows != 0) mpad else m;
    return .{ .mpad = u(mpad), .codes_bytes = stored * k / 2, .fp8_bytes = stored * k, .scales_bytes = @as(u64, k) / 64 * mpad * 4, .zero_scales = tb_rows == 0 };
}

/// Bytes of pack4's output: words [npad/64, k/64, 8, 32, 2] int32 = npad * k / 2.
pub fn packedBytes(npad: u32, k: u32) u64 {
    return @as(u64, npad) * k / 2;
}

// ---- act.cu: quant4_cuda / quant8_cuda ----

fn quantGrid(k: u32, m: u32, mpad: u32, tb_rows: u32) Error!Dim3 {
    const rows_y = if (tb_rows != 0) mpad else m;
    if (rows_y > 65535) return error.TooManyRows; // grid.y is the row
    return .{ .x = u(ceilDiv(k / 16, 128)), .y = rows_y };
}

/// quant4_cuda: x bf16 [m, k] with row stride `ldx` elements (16-byte aligned, ldx % 8 == 0) -> codes, scales.
/// `act` is the input scale (g = inv(act)). Args: x, ldx, M, K, g, codes, scales, mpad, tb.
pub fn quant4(m: u32, k: u32, ldx: u32, tb_rows: u32, act: f64) Error!Launch {
    if (k % 64 != 0) return error.KNotMultipleOf64;
    if (tb_rows != 0 and tb_rows % 16 != 0) return error.TbNotMultipleOf16;
    const r = rows(m, k, tb_rows);
    var l = Launch{ .symbol = kernels.quant4, .module = .act, .grid = try quantGrid(k, m, r.mpad, tb_rows), .block = .{ .x = 128 }, .smem = 0 };
    l.push(.x);
    l.push(i(ldx));
    l.push(i(m));
    l.push(i(k));
    l.push(.{ .f32 = inv(act) });
    l.push(.codes);
    l.push(.scales);
    l.push(i(r.mpad));
    l.push(i(tb_rows));
    return l;
}

/// quant8_cuda (FP8 rows): Args: x, ldx, M, K, inv, out, mpad, tb. mpad = rows(...).mpad when tiled, else m.
pub fn quant8(m: u32, k: u32, ldx: u32, tb_rows: u32, act: f64) Error!Launch {
    if (k % 64 != 0) return error.KNotMultipleOf64;
    if (tb_rows != 0 and tb_rows % 16 != 0) return error.TbNotMultipleOf16;
    const r = rows(m, k, tb_rows);
    const mpad: u32 = if (tb_rows != 0) r.mpad else m;
    var l = Launch{ .symbol = kernels.quant8, .module = .act, .grid = try quantGrid(k, m, r.mpad, tb_rows), .block = .{ .x = 128 }, .smem = 0 };
    l.push(.x);
    l.push(i(ldx));
    l.push(i(m));
    l.push(i(k));
    l.push(.{ .f32 = inv(act) });
    l.push(.out);
    l.push(i(mpad));
    l.push(i(tb_rows));
    return l;
}

// ---- lane4.cu: pack4_cuda ----

/// pack4_cuda: checkpoint bytes [n, k/2] -> words [npad/64, k/64, 8, 32, 2]; the rows past n become zero words.
/// Args: src, N, K, dst.
pub fn pack4(n: u32, k: u32, npad: u32) Error!Launch {
    if (k % 64 != 0) return error.KNotMultipleOf64;
    if (npad % 64 != 0) return error.NpadNotMultipleOf64;
    if (k / 64 > 65535) return error.TooManyRows;
    var l = Launch{ .symbol = kernels.pack4, .module = .lane4, .grid = .{ .x = npad / 64, .y = k / 64 }, .block = .{ .x = 256 }, .smem = 0 };
    l.push(.src);
    l.push(i(n));
    l.push(i(k));
    l.push(.dst);
    return l;
}

// ---- gemm_ck.cu ----

/// gemm_ck.cu Gemm<MODE, BM, BN, WM, WN, STAGES, KS>::SMEM.
fn ckSmem(mode: Mode, bm: u32, bn: u32, stages: u32, ks: u32) u32 {
    const a4 = mode == .a4;
    const row: u32 = if (a4) 32 else 64;
    const tile: u32 = if (a4) 2048 else 4096;
    const step = bm * row + (if (a4) bm * 4 else 0) + bn / 64 * tile + (if (a4) bn * 4 else 0);
    return stages * ((ks * step + 127) / 128 * 128);
}

/// by_tile's choice: 0 picks by chip and rows (128 x 256 off GB10 from 512 rows with K/64 even, else 128 x 128).
const CkTile = struct { bm: u32, bn: u32, stages: u32, ks: u32 };

fn ckTile(mode: Mode, tile_in: u32, gb10: bool, m: u32, k: u32) CkTile {
    var t = tile_in;
    if (t == 0) t = if (!gb10 and m >= 512 and (k / 64) % 2 == 0) 2 else 1;
    return switch (t) {
        2 => .{ .bm = 128, .bn = 256, .stages = 2, .ks = 2 },
        3 => .{ .bm = 64, .bn = 128, .stages = 4, .ks = 1 },
        else => .{ .bm = 128, .bn = 128, .stages = if (mode == .a4) 4 else 3, .ks = 1 },
    };
}

/// launch(): the L2 band of row tiles, max(1, min(rows_t, 12 MiB / bytes of one row tile of x)).
fn group(rows_t: u64, bm: u32, mode: Mode, k: u32) u32 {
    const row_bytes: u64 = @as(u64, bm) * (if (mode == .a4) k / 2 else k);
    return u(@max(1, @min(rows_t, l2_bytes / row_bytes)));
}

fn checkGemm(k: u32, n: u32, npad: u32) Error!void {
    if (k % 64 != 0) return error.KNotMultipleOf64;
    if (npad % 64 != 0) return error.NpadNotMultipleOf64;
    if (n > npad) return error.NpadBelowN;
}

/// Args of gemm_kernel / ws_kernel: x, xs, w, ws, alpha, out, M, N, K, mpad, npad, group (and Up for gemm_kernel).
fn gemmArgs(l: *Launch, mode: Mode, al: f32, m: u32, n: u32, k: u32, mpad: u32, npad: u32, grp: u32, with_up: bool) void {
    l.push(.x);
    l.push(if (mode == .a4) .xs else .null_ptr);
    l.push(.w);
    l.push(if (mode == .a4) .ws else .null_ptr);
    l.push(.{ .f32 = al });
    l.push(.out);
    for ([_]u32{ m, n, k, mpad, npad, grp }) |v| l.push(i(v));
    if (with_up) l.push(.up_zero);
}

/// gemm_cuda (row-major rows, quant4 with tb 0): out (m, n) = alpha * rows @ weight. `mpad` is quant4's mpad (xs.size(1)),
/// ignored for FP8. `tile`: 0 by chip and rows, 1 128 x 128, 2 128 x 256, 3 64 x 128. `gb10`: isGb10.
pub fn gemm(mode: Mode, m: u32, n: u32, k: u32, npad: u32, mpad: u32, tile: u32, is_f32: bool, al: f32, gb10: bool) Error!Launch {
    try checkGemm(k, n, npad);
    const t = ckTile(mode, tile, gb10, m, k);
    if ((k / 64) % t.ks != 0) return error.KNotWholeStages;
    const found = kernels.findCk(@intFromEnum(mode), t.bm, t.bn, 2, 4, t.stages, t.ks, is_f32, 0) orelse return error.MissingKernel;
    const rows_t = ceilDiv(m, t.bm);
    const mp: u32 = if (mode == .a4) mpad else 0;
    var l = Launch{
        .symbol = found.symbol,
        .module = .gemm_ck,
        .grid = .{ .x = u(rows_t * ceilDiv(npad, t.bn)) },
        .block = .{ .x = 2 * 4 * 32 },
        .smem = ckSmem(mode, t.bm, t.bn, t.stages, t.ks),
    };
    gemmArgs(&l, mode, al, m, n, k, mp, npad, group(rows_t, t.bm, mode, k), true);
    return l;
}

/// gemm_gu_ck_cuda: gate|up of the same NVFP4 rows -> SiLU(gate) * up -> down's input rows (codes [m, npad/2], scales
/// [npad/64, mpad, 4], both written; the scales' padding rows too), under down's global scale (qg = inv(down.act)).
/// `wg`/`wsg` are the gate weight and scales (w/ws roles), up's go in the Up struct. `fp32`: SwiGLU without the bf16
/// rounding (checkpoint.SWIGLU_FP32, true in the reference). Each block does 128 columns of both halves.
pub fn gemmGu(m: u32, npad: u32, k: u32, mpad: u32, alpha_g: f32, alpha_u: f32, qg: f32, fp32: bool) Error!Launch {
    if (k % 128 != 0) return error.KNotMultipleOf128; // K in steps of 128 (two 64-input steps a stage)
    if (npad % 64 != 0) return error.NpadNotMultipleOf64;
    const epi: u32 = if (fp32) 2 else 1;
    const found = kernels.findCk(0, 128, 256, 2, 4, 3, 2, false, epi) orelse return error.MissingKernel;
    const rows_t = ceilDiv(m, 128);
    var l = Launch{
        .symbol = found.symbol,
        .module = .gemm_ck,
        .grid = .{ .x = u(rows_t * ceilDiv(npad, 128)) }, // cols_t over BN / 2 = 128 columns
        .block = .{ .x = 256 },
        .smem = ckSmem(.a4, 128, 256, 3, 2),
    };
    l.push(.x);
    l.push(.xs);
    l.push(.w);
    l.push(.ws);
    l.push(.{ .f32 = alpha_g });
    l.push(.null_ptr); // out: the epilogue writes up.codes / up.scales
    for ([_]u32{ m, npad, k, mpad, npad, group(rows_t, 128, .a4, k) }) |v| l.push(i(v));
    l.push(.{ .up = .{ .alpha = alpha_u, .qg = qg, .kd = @intCast(npad) } });
    return l;
}

// ---- gemm_ws.cu ----

/// gemm_ws.cu WS<MODE, BM, BN, STAGES, KS>::SMEM (stages, then the full barriers and the done counters).
fn wsSmem(mode: Mode, bm: u32, bn: u32, stages: u32, ks: u32) u32 {
    const a4 = mode == .a4;
    const tile: u32 = if (a4) 2048 else 4096;
    const ts: u32 = if (a4) 256 else 0;
    const tiles = bn / 64;
    const w_at = ks * (bm * (if (a4) @as(u32, 32) else 64) + (if (a4) bm * 4 else 0));
    const stage = w_at + tiles * ks * tile + tiles * ks * ts;
    return stages * stage + stages * 8 + stages * 4;
}

/// gemm_ws_cuda (bulk-copy GEMM, rows quantized with tb = TB): `mpad` = x.size(0) (a multiple of 128 holding every
/// row). `tile` is `tileArg(prompt tile)`: 2 128 x 256 two steps a stage, 3 128 x 128 two steps, else 128 x 256 one.
pub fn gemmWs(mode: Mode, m: u32, n: u32, k: u32, mpad: u32, npad: u32, tile: u32, is_f32: bool, al: f32) Error!Launch {
    try checkGemm(k, n, npad);
    const a4 = mode == .a4;
    const bn: u32 = if (tile == 3) 128 else 256;
    const stages: u32 = switch (tile) {
        2 => if (a4) 3 else 2,
        3 => 3,
        else => if (a4) 4 else 3,
    };
    const ks: u32 = if (tile == 2 or tile == 3) 2 else 1;
    if ((k / 64) % ks != 0) return error.KNotWholeStages;
    if (mpad % 128 != 0) return error.MpadNotTiled;
    if (mpad < m) return error.MpadBelowM;
    const found = kernels.findWs(@intFromEnum(mode), 128, bn, stages, ks, is_f32) orelse return error.MissingKernel;
    const rows_t: u64 = mpad / 128;
    var l = Launch{
        .symbol = found.symbol,
        .module = .gemm_ws,
        .grid = .{ .x = u(rows_t * ceilDiv(npad, bn)) },
        .block = .{ .x = 256 }, // eight mma warps; warp 0 also issues the bulk copies
        .smem = wsSmem(mode, 128, bn, stages, ks),
    };
    gemmArgs(&l, mode, al, m, n, k, mpad, npad, group(rows_t, 128, mode, k), false);
    return l;
}

/// checkpoint._gemm: what a prompt linear launches for prompt `tile` on capability (major, minor). `mpad` is
/// rows(m, k, tb(tile, ...)).mpad: quant4's, so tiled to TB when the bulk-copy GEMM runs and to 64 otherwise.
pub fn gemmPrompt(mode: Mode, tile: u32, major: u32, minor: u32, m: u32, n: u32, k: u32, npad: u32, mpad: u32, is_f32: bool, al: f32) Error!Launch {
    if (bulkTile(tile, major, minor)) return gemmWs(mode, m, n, k, mpad, npad, tileArg(tile), is_f32, al);
    return gemm(mode, m, n, k, npad, mpad, tileArg(tile), is_f32, al, isGb10(major, minor));
}

// ---- tests: the Qwen-Image DiT's shapes (tile 12 = the bulk-copy GEMM, tile argument 2) ----

const tt = std.testing;

test "tile 12 on sm_120 / sm_121 is the bulk-copy GEMM at 128 x 256, three stages, two steps" {
    try tt.expect(bulkTile(12, 12, 0) and bulkTile(12, 12, 1));
    try tt.expect(!bulkTile(10, 12, 0) and !bulkTile(0, 12, 0) and !bulkTile(12, 8, 9));
    try tt.expectEqual(@as(u32, 128), tb(12, 12, 0));
    try tt.expectEqual(@as(u32, 0), tb(3, 12, 0));
    try tt.expectEqual(@as(u32, 2), tileArg(12));
    try tt.expectEqual(@as(u32, 3), tileArg(3));
}

test "scales: fp32 rounding of 1 / act and act * scale" {
    try tt.expectEqual(@as(f32, 1.0) / @as(f32, 0.1), inv(0.1));
    try tt.expectEqual(@as(f32, 0.1) * @as(f32, 0.3), alpha(0.1, 0.3));
}

test "rows: padding and buffers" {
    const r = rows(4096, 4096, TB);
    try tt.expectEqual(@as(u32, 4096), r.mpad);
    try tt.expectEqual(@as(u64, 4096 * 2048), r.codes_bytes);
    try tt.expectEqual(@as(u64, 64 * 4096 * 4), r.scales_bytes);
    try tt.expect(!r.zero_scales);
    try tt.expectEqual(@as(u32, 128), rows(30, 4096, TB).mpad);
    try tt.expectEqual(@as(u32, 128), rows(80, 4096, TB).mpad);
    const p = rows(80, 4096, 0); // row-major: 64-row padding, codes only for the m rows, scales zeroed
    try tt.expectEqual(@as(u32, 128), p.mpad);
    try tt.expectEqual(@as(u64, 80 * 2048), p.codes_bytes);
    try tt.expect(p.zero_scales);
    try tt.expectEqual(@as(u32, 64), rows(30, 4096, 0).mpad);
}

test "quant4 at the DiT's activations" {
    const q = try quant4(4096, 4096, 4096, TB, 0.5);
    try tt.expectEqual(Dim3{ .x = 2, .y = 4096, .z = 1 }, q.grid); // (256 blocks of 16 + 127) / 128 columns, a row a y
    try tt.expectEqual(@as(u32, 128), q.block.x);
    try tt.expectEqual(@as(u32, 0), q.smem);
    try tt.expectEqual(@as(usize, 9), q.nargs);
    try tt.expectEqual(@as(f32, 2.0), q.slice()[4].f32);
    try tt.expectEqual(@as(i32, 4096), q.slice()[7].i32); // mpad
    try tt.expectEqual(@as(i32, 128), q.slice()[8].i32); // tb
    const down_in = try quant4(4096, 12288, 12288, TB, 1.0);
    try tt.expectEqual(@as(u32, 6), down_in.grid.x);
    const txt = try quant4(30, 4096, 4096, TB, 1.0); // tiled: the grid covers the padding rows (they become zeros)
    try tt.expectEqual(@as(u32, 128), txt.grid.y);
    const rm = try quant4(30, 4096, 4096, 0, 1.0); // row-major: only the m rows
    try tt.expectEqual(@as(u32, 30), rm.grid.y);
    try tt.expectEqual(@as(i32, 64), rm.slice()[7].i32);
    try tt.expectError(error.KNotMultipleOf64, quant4(4, 100, 100, 0, 1.0));
    try tt.expectError(error.TooManyRows, quant4(70000, 64, 64, 0, 1.0));
}

const Case = struct { m: u32, k: u32, n: u32, grid: u32, group: u32 };

test "gemm_ws at tile 12: qkv, out, down" {
    // symbol: ws_kernel<A4, 128, 256, 3, 2, false>; smem 3 * 27648 + 36
    const cases = [_]Case{
        .{ .m = 4096, .k = 4096, .n = 12288, .grid = 32 * 48, .group = 32 }, // qkv, 1024 x 1024 image
        .{ .m = 4096, .k = 4096, .n = 4096, .grid = 32 * 16, .group = 32 }, // out
        .{ .m = 4096, .k = 12288, .n = 4096, .grid = 32 * 16, .group = 16 }, // down: 12 MiB / (128 * 6144) = 16
        .{ .m = 1024, .k = 4096, .n = 12288, .grid = 8 * 48, .group = 8 }, // qkv, 512 x 512
        .{ .m = 1024, .k = 12288, .n = 4096, .grid = 8 * 16, .group = 8 },
        .{ .m = 30, .k = 4096, .n = 12288, .grid = 48, .group = 1 }, // text prefix: one row tile
        .{ .m = 80, .k = 12288, .n = 4096, .grid = 16, .group = 1 },
    };
    for (cases) |c| {
        const mpad = rows(c.m, c.k, tb(12, 12, 1)).mpad;
        const l = try gemmPrompt(.a4, 12, 12, 1, c.m, c.n, c.k, c.n, mpad, false, 0.25);
        try tt.expectEqualStrings("_ZN13nvfp4_gemm_ws9ws_kernelILi0ELi128ELi256ELi3ELi2ELb0EEEvPKhS2_S2_S2_fPviiiiii", l.symbol);
        try tt.expectEqual(Module.gemm_ws, l.module);
        try tt.expectEqual(Dim3{ .x = c.grid, .y = 1, .z = 1 }, l.grid);
        try tt.expectEqual(Dim3{ .x = 256, .y = 1, .z = 1 }, l.block);
        try tt.expectEqual(@as(u32, 3 * 27648 + 3 * 8 + 3 * 4), l.smem);
        try tt.expect(l.needsSmemOptIn());
        try tt.expectEqual(@as(usize, 12), l.nargs);
        const a = l.slice();
        try tt.expectEqual(@as(i32, @intCast(c.m)), a[6].i32);
        try tt.expectEqual(@as(i32, @intCast(c.n)), a[7].i32);
        try tt.expectEqual(@as(i32, @intCast(c.k)), a[8].i32);
        try tt.expectEqual(@as(i32, @intCast(mpad)), a[9].i32);
        try tt.expectEqual(@as(i32, @intCast(c.n)), a[10].i32); // npad = n
        try tt.expectEqual(@as(i32, @intCast(c.group)), a[11].i32);
    }
}

test "gemm_ws: other tiles, fp32 output, shape errors" {
    const l1 = try gemmWs(.a4, 4096, 4096, 4096, 4096, 4096, 1, true, 1); // 128 x 256, four stages, one step
    try tt.expectEqual(@as(u32, 4 * 13824 + 48), l1.smem);
    try tt.expect(std.mem.indexOf(u8, l1.symbol, "ILi0ELi128ELi256ELi4ELi1ELb1E") != null);
    const l3 = try gemmWs(.a4, 128, 4096, 4096, 128, 4096, 3, false, 1); // 128 x 128 two steps: 64 column tiles of 128
    try tt.expectEqual(@as(u32, 32), l3.grid.x);
    try tt.expectEqual(@as(u32, 3 * (2 * 4608 + 2 * 2 * 2048 + 2 * 2 * 256) + 36), l3.smem);
    const f8 = try gemmWs(.a8, 128, 4096, 4096, 128, 4096, 2, false, 1);
    try tt.expectEqual(Arg.null_ptr, f8.slice()[1]); // FP8 has no row scales
    try tt.expectEqual(Arg.null_ptr, f8.slice()[3]);
    try tt.expectError(error.MpadNotTiled, gemmWs(.a4, 100, 4096, 4096, 100, 4096, 2, false, 1));
    try tt.expectError(error.KNotWholeStages, gemmWs(.a4, 128, 4096, 64 * 3, 128, 4096, 2, false, 1));
    try tt.expectError(error.NpadBelowN, gemmWs(.a4, 128, 4096, 4096, 128, 64, 2, false, 1));
}

test "gemm_gu_ck: gate / up halves of 12288" {
    const l = try gemmGu(4096, 12288, 4096, 4096, 0.5, 0.25, 2.0, true);
    try tt.expectEqualStrings("_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi256ELi2ELi4ELi3ELi2ELb0ELi2EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE", l.symbol);
    try tt.expectEqual(@as(u32, 32 * 96), l.grid.x);
    try tt.expectEqual(@as(u32, 256), l.block.x);
    try tt.expectEqual(@as(u32, 3 * 27648), l.smem); // 82,944 >= the epilogue's 65,536 hand-over
    try tt.expectEqual(@as(usize, 13), l.nargs);
    const a = l.slice();
    try tt.expectEqual(Arg.null_ptr, a[5]);
    try tt.expectEqual(@as(i32, 12288), a[7].i32); // N = npad
    try tt.expectEqual(@as(i32, 32), a[11].i32);
    try tt.expectEqual(UpScalars{ .alpha = 0.25, .qg = 2.0, .kd = 12288 }, a[12].up);
    const bf = try gemmGu(80, 12288, 4096, 128, 1, 1, 1, false);
    try tt.expect(std.mem.indexOf(u8, bf.symbol, "ELb0ELi1EEE") != null);
    try tt.expectEqual(@as(u32, 96), bf.grid.x);
    try tt.expectError(error.KNotMultipleOf128, gemmGu(80, 12288, 4096 + 64, 128, 1, 1, 1, true));
}

test "gemm tile 0 (the down projection after gemm_gu_ck): by chip and rows" {
    // mlp_prompt: quant4 rows with tb 0, then _gemm(..., tile 0): not the bulk path
    const mpad = rows(4096, 12288, 0).mpad;
    const gb10 = try gemmPrompt(.a4, 0, 12, 1, 4096, 4096, 12288, 4096, mpad, false, 1);
    try tt.expectEqualStrings("_ZN13nvfp4_gemm_ck11gemm_kernelILi0ELi128ELi128ELi2ELi4ELi4ELi1ELb0ELi0EEEvPKhS2_S2_S2_fPviiiiiiNS_2UpE", gb10.symbol);
    try tt.expectEqual(@as(u32, 32 * 32), gb10.grid.x);
    try tt.expectEqual(@as(u32, 4 * 9216), gb10.smem);
    try tt.expect(!gb10.needsSmemOptIn());
    const pro = try gemmPrompt(.a4, 0, 12, 0, 4096, 4096, 12288, 4096, mpad, false, 1);
    try tt.expect(std.mem.indexOf(u8, pro.symbol, "ILi0ELi128ELi256ELi2ELi4ELi2ELi2ELb0ELi0E") != null);
    try tt.expectEqual(@as(u32, 32 * 16), pro.grid.x);
    try tt.expectEqual(@as(u32, 2 * 27648), pro.smem);
    const small = try gemmPrompt(.a4, 0, 12, 0, 80, 4096, 12288, 4096, rows(80, 12288, 0).mpad, false, 1); // < 512 rows
    try tt.expect(std.mem.indexOf(u8, small.symbol, "ILi0ELi128ELi128ELi2ELi4ELi4ELi1E") != null);
    const up = small.slice()[12];
    try tt.expectEqual(Arg.up_zero, up);
    try tt.expectEqual(@as(i32, 128), small.slice()[9].i32); // mpad of the row-major rows
}

test "every launch the plan can make has its symbol built" {
    for (kernels.ck_instances) |c| {
        const mode: Mode = @enumFromInt(c.mode);
        if (c.epi != 0) continue;
        const tile: u32 = if (c.bm == 64) 3 else if (c.bn == 256) 2 else 1;
        const l = try gemm(mode, 4096, 4096, 4096, 4096, 4096, tile, c.f32, 1, false);
        try tt.expectEqualStrings(c.symbol, l.symbol);
    }
    for (kernels.ws_instances) |c| {
        const mode: Mode = @enumFromInt(c.mode);
        const tile: u32 = if (c.bn == 128) 3 else if (c.ks == 2) 2 else 1;
        const l = try gemmWs(mode, 4096, 4096, 4096, 4096, 4096, tile, c.f32, 1);
        try tt.expectEqualStrings(c.symbol, l.symbol);
    }
}

test "pack4" {
    const l = try pack4(12288, 4096, 12288);
    try tt.expectEqual(Dim3{ .x = 192, .y = 64, .z = 1 }, l.grid);
    try tt.expectEqual(@as(u32, 256), l.block.x);
    try tt.expectEqual(@as(u64, 12288 * 2048), packedBytes(12288, 4096));
    try tt.expectEqual(@as(usize, 4), l.nargs);
}
