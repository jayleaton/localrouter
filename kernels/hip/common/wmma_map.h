// Ours: where every element of LocalRouter's HIP WMMA GEMMs lives (docs/dev/ROCM.md section 3), with no HIP in it, so
// the kernels (wmma_gemm.h) and a host test (kernels/hip/test/wmma_map_test.cpp, `zig build test`) share one copy.
//
// gfx11 wave32 WMMA, v_wmma_f32_16x16x16_{bf16,f16}: D[16 x 16] = A[16 x 16] . B[16 x 16] + C, f32 accumulate.
//   A operand: lane l holds row l % 16, its 16 k values in order (lanes 16..31 repeat lanes 0..15, as gfx11 requires);
//   B operand: lane l holds column l % 16, its 16 k values in order; for C = A . B^T with B stored [N, K], that is row
//              l % 16 of B, contiguous;
//   C / D:     lane l, register v (0..7) holds row 2 v + l / 16, column l % 16.
//
// A block tile is BM x BN, K in tiles of BK = 32 (two k16 chunks), WM x WN waves; a wave owns TM = BM / WM rows and
// TN = BN / WN columns: MI x NJ 16 x 16 accumulators. Every output element accumulates the k16 chunks 0, 1, ..
// 2 ceil(K / 32) - 1 of its own row of A and row of B, in that order, chunks past K zero filled: the same sequence
// whatever the tile shape, the grid order or M and N, which is why a row's bits depend on none of them.
#pragma once

#ifndef LR_HD
#if defined(__HIP__) || defined(__CUDACC__)
#define LR_HD __host__ __device__
#else
#define LR_HD
#endif
#endif

namespace lr_wmma {

constexpr int BK = 32;      // K a stage: two k16 chunks
constexpr int PAD = 8;      // 16-bit elements of padding a staged row (rows of 80 bytes)
constexpr int CHUNK = 8;    // 16-bit elements a 16-byte copy

template <int TBM, int TBN, int TWM, int TWN, bool MFAST>
struct Cfg {
    static constexpr int BM = TBM, BN = TBN, WM = TWM, WN = TWN;
    static constexpr bool M_FAST = MFAST;  // grid (m tiles, n tiles, batch), else (n tiles, m tiles, batch)
    static constexpr int THREADS = 32 * WM * WN;
    static constexpr int LD = BK + PAD;
    static constexpr int TM = BM / WM, TN = BN / WN;
    static constexpr int MI = TM / 16, NJ = TN / 16;
    static constexpr int CPR = BK / CHUNK;  // copies a staged row
    static constexpr int A_CH = BM * CPR / THREADS, B_CH = BN * CPR / THREADS;
    static constexpr int SMEM = 2 * (BM + BN) * LD * 2;  // two stages of A and B, bytes (static shared memory)
    static_assert(TM % 16 == 0 && TN % 16 == 0, "a wave's tile is whole 16 x 16 accumulators");
    static_assert(BM * CPR % THREADS == 0 && BN * CPR % THREADS == 0, "a stage is whole copies a thread");
    static_assert(SMEM <= 65536, "gfx11: 64 KiB of LDS a workgroup");
};

// The 16-byte copy `e` (0 .. rows * CPR - 1) of a stage: its row in the tile and its first k in the stage.
LR_HD constexpr int stage_row(int e) { return e / (BK / CHUNK); }
LR_HD constexpr int stage_col(int e) { return (e % (BK / CHUNK)) * CHUNK; }

// Block (bx, by)'s first row and column.
template <class Cf>
LR_HD constexpr int tile_m0(int bx, int by) { return (Cf::M_FAST ? bx : by) * Cf::BM; }
template <class Cf>
LR_HD constexpr int tile_n0(int bx, int by) { return (Cf::M_FAST ? by : bx) * Cf::BN; }

// Thread t's wave position in the block.
template <class Cf>
LR_HD constexpr int wave_m(int t) { return (t / 32) / Cf::WN; }
template <class Cf>
LR_HD constexpr int wave_n(int t) { return (t / 32) % Cf::WN; }

// The staged row a lane's A fragment (accumulator row block i) or B fragment (column block j) is read from.
template <class Cf>
LR_HD constexpr int frag_a_row(int wm, int i, int lane) { return wm * Cf::TM + i * 16 + lane % 16; }
template <class Cf>
LR_HD constexpr int frag_b_row(int wn, int j, int lane) { return wn * Cf::TN + j * 16 + lane % 16; }

// The output element register v of accumulator (i, j) of a lane holds, relative to the block's (m0, n0).
template <class Cf>
LR_HD constexpr int out_row(int wm, int i, int v, int lane) { return wm * Cf::TM + i * 16 + 2 * v + lane / 16; }
template <class Cf>
LR_HD constexpr int out_col(int wn, int j, int lane) { return wn * Cf::TN + j * 16 + lane % 16; }

LR_HD constexpr int k_tiles(int K) { return (K + BK - 1) / BK; }

}  // namespace lr_wmma
