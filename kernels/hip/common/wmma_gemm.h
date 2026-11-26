// Ours: the deterministic WMMA GEMM of LocalRouter's HIP kernels (docs/dev/ROCM.md section 3), C[M, N] = A[M, K] .
// B[N, K]^T with 16-bit inputs (bf16 or f16) and fp32 accumulation, for gfx11 (wave32). The element map is
// wmma_map.h's; the epilogue (bias, rounding, residual) is the caller's, one call an element.
//
// Order: every output element sums the k16 chunks of K in ascending order, each one v_wmma (whose order inside the
// instruction is the hardware's, fixed), no split-K: a row's bits depend on neither M, N, the tile shape, the grid
// order nor the other rows. Stages: tile t + 1 is read from memory into registers while tile t is multiplied out of
// LDS, then written to the other LDS stage (gfx11 has no cp.async); one barrier a K tile.
//
// Requirements: K % 8 == 0, lda / ldb % 8 == 0, A and B 16-byte aligned (16-byte loads).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "wmma_map.h"

namespace lr_wmma {

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

// 16 consecutive 16-bit values at p (16-byte aligned) as a WMMA operand.
template <class V>
__device__ __forceinline__ V frag(const uint16_t* p) {
    const uint4 lo = reinterpret_cast<const uint4*>(p)[0], hi = reinterpret_cast<const uint4*>(p)[1];
    V v;
    __builtin_memcpy(&v, &lo, 16);
    __builtin_memcpy(reinterpret_cast<char*>(&v) + 16, &hi, 16);
    return v;
}

__device__ __forceinline__ f32x8 mma(bf16x16 a, bf16x16 b, f32x8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c); }
__device__ __forceinline__ f32x8 mma(f16x16 a, f16x16 b, f32x8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

// One K tile's copies of A and B into registers: rows clamped to the matrix, zero where the row or the chunk is out of
// range (K % 8 == 0: a chunk is wholly in or out).
template <class Cf>
__device__ __forceinline__ void load_tile(uint4 (&ra)[Cf::A_CH], uint4 (&rb)[Cf::B_CH], const uint16_t* A, const uint16_t* B,
                                          long long lda, long long ldb, int M, int N, int K, int m0, int n0, int k0) {
#pragma unroll
    for (int i = 0; i < Cf::A_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS, r = stage_row(e), k = k0 + stage_col(e);
        ra[i] = (k < K && m0 + r < M) ? *reinterpret_cast<const uint4*>(A + (long long)(m0 + r) * lda + k) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < Cf::B_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS, r = stage_row(e), k = k0 + stage_col(e);
        rb[i] = (k < K && n0 + r < N) ? *reinterpret_cast<const uint4*>(B + (long long)(n0 + r) * ldb + k) : make_uint4(0, 0, 0, 0);
    }
}

template <class Cf>
__device__ __forceinline__ void store_tile(uint16_t* sa, uint16_t* sb, const uint4 (&ra)[Cf::A_CH], const uint4 (&rb)[Cf::B_CH]) {
#pragma unroll
    for (int i = 0; i < Cf::A_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS;
        *reinterpret_cast<uint4*>(sa + stage_row(e) * Cf::LD + stage_col(e)) = ra[i];
    }
#pragma unroll
    for (int i = 0; i < Cf::B_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS;
        *reinterpret_cast<uint4*>(sb + stage_row(e) * Cf::LD + stage_col(e)) = rb[i];
    }
}

// The block (blockIdx.x, blockIdx.y) of C: `epi(row, col, acc)` once for every element of it inside M x N.
// V: bf16x16 or f16x16, the operand type of A and B's 16-bit elements.
template <class Cf, class V, class Epi>
__device__ __forceinline__ void gemm(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, int M, int N, int K,
                                     long long lda, long long ldb, Epi epi) {
    __shared__ __attribute__((aligned(16))) uint16_t sa[2][Cf::BM * Cf::LD];
    __shared__ __attribute__((aligned(16))) uint16_t sb[2][Cf::BN * Cf::LD];
    const int lane = threadIdx.x % 32, wm = wave_m<Cf>(threadIdx.x), wn = wave_n<Cf>(threadIdx.x);
    const int m0 = tile_m0<Cf>(blockIdx.x, blockIdx.y), n0 = tile_n0<Cf>(blockIdx.x, blockIdx.y);
    // accumulator blocks with at least one row / column inside the matrix (uniform over the wave: WMMA runs every lane)
    bool rv[Cf::MI], cv[Cf::NJ];
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i) rv[i] = m0 + wm * Cf::TM + i * 16 < M;
#pragma unroll
    for (int j = 0; j < Cf::NJ; ++j) cv[j] = n0 + wn * Cf::TN + j * 16 < N;
    f32x8 acc[Cf::MI][Cf::NJ];
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i)
#pragma unroll
        for (int j = 0; j < Cf::NJ; ++j) acc[i][j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    const int tiles = k_tiles(K);
    uint4 ra[Cf::A_CH], rb[Cf::B_CH];
    load_tile<Cf>(ra, rb, A, B, lda, ldb, M, N, K, m0, n0, 0);
    store_tile<Cf>(sa[0], sb[0], ra, rb);
    __syncthreads();
    for (int t = 0; t < tiles; ++t) {
        const int s = t & 1;
        if (t + 1 < tiles) load_tile<Cf>(ra, rb, A, B, lda, ldb, M, N, K, m0, n0, (t + 1) * BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            V a[Cf::MI];
#pragma unroll
            for (int i = 0; i < Cf::MI; ++i)
                if (rv[i]) a[i] = frag<V>(sa[s] + frag_a_row<Cf>(wm, i, lane) * Cf::LD + kk);
#pragma unroll
            for (int j = 0; j < Cf::NJ; ++j) {
                if (!cv[j]) continue;
                const V b = frag<V>(sb[s] + frag_b_row<Cf>(wn, j, lane) * Cf::LD + kk);
#pragma unroll
                for (int i = 0; i < Cf::MI; ++i)
                    if (rv[i]) acc[i][j] = mma(a[i], b, acc[i][j]);
            }
        }
        // stage s ^ 1 was last read in tile t - 1, before the barrier that ended it
        if (t + 1 < tiles) store_tile<Cf>(sa[s ^ 1], sb[s ^ 1], ra, rb);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i)
#pragma unroll
        for (int j = 0; j < Cf::NJ; ++j) {
            if (!rv[i] || !cv[j]) continue;
            const int col = n0 + out_col<Cf>(wn, j, lane);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = m0 + out_row<Cf>(wm, i, v, lane);
                if (row < M && col < N) epi(row, col, acc[i][j][v]);
            }
        }
}

// bf16 / f16 bits <-> float, written out (no HIP conversion operators: torch's __HIP_NO_HALF_CONVERSIONS__).
__device__ __forceinline__ float bf16_to_float(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

// CUDA's __float2bfloat16_rn: round to nearest even, every NaN to 0x7fff.
__device__ __forceinline__ uint16_t float_to_bf16_rn(float f) {
    const uint32_t u = __float_as_uint(f);
    return (f != f) ? static_cast<uint16_t>(0x7fffu) : static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace lr_wmma
