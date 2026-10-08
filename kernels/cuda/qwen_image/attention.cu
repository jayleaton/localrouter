// Ours: TensorFold's prefill attention (tensorfold-zig 88c424e zig/kernels/cuda/prefill_attention.cu, blob 122f404495c73114909e923796e4eb631ade005f),
// with a key count and a causal flag, for the DiT's [prefix | target] keys; Apache-2.0, see NOTICE.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace stk_attention {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;
using qmm_frag::mma0;

constexpr int BN = 64, HALF = 32;             // keys a tile (the fold's unit), keys a staging slot
constexpr float LOG2E = 1.4426950408889634f;
constexpr int SMEM_MAX = 96 * 1024;

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(qmm_frag::smem(p)));
}

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
    return y;
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float tdiv(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ int sw(int r, int c) { return r * (D / 8) + (c ^ (r & 7)); }

__device__ __forceinline__ float rowsum(const float (&c)[8]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float x = __fadd_rn(c[j], __shfl_xor_sync(0xffffffffu, c[j], 2));
        w[j] = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 1));
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(w[0], w[4]), __fadd_rn(w[2], w[6])),
                     __fadd_rn(__fadd_rn(w[1], w[5]), __fadd_rn(w[3], w[7])));
}

template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
pattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
             const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
             int G, int nkeys, int causal, float scale) {
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, RB = WARPS / HPC, THREADS = 32 * WARPS;
    constexpr bool QREG = D <= 128;               // the queries' fragments stay in registers, else in shared memory
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * HALF * C;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = blockIdx.x * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = causal ? (p0 + last) / BN + 1 : (nkeys + BN - 1) / BN, items = 4 * tiles, limit = nkeys;
    auto stage = [&](int item) {
        const __nv_bfloat16* src = item % 4 < 2 ? kc : vc;
        const int key0 = (item / 4) * BN + (item % 2) * HALF;
        uint4* dst = slots + (item % NS) * HALF * C;
        for (int i = threadIdx.x; i < HALF * C; i += THREADS) {
            const int key = key0 + i / C;
            const bool in = key < limit;             // past the cache's filled keys: zeros, never read
            cp16z(dst + sw<D>(i / C, i % C), src + ((int64_t)(in ? key : 0) * HK + hk) * D + (i % C) * 8, in);
        }
    };
    uint32_t qa[QREG ? K16 : 1][4];
    if constexpr (QREG) {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    } else {
        for (int i = threadIdx.x; i < WARPS * 16 * C; i += THREADS) {
            const int w = i / (16 * C), r = (i / C) % 16, row = row0 + (w / HPC) * 16 + r;
            const int hd = hk * G + hg * HPC + w % HPC;
            cp16z(qs + sw<D>(i / C, i % C), q + ((int64_t)min(row, W - 1) * H + hd) * D + (i % C) * 8, row < W);
        }
        commit();
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    auto vis = [&](int c, int pos) { return c < nkeys && (!causal || c <= pos); };  // keys a query sees
    auto take = [&](int it) -> const uint4* {           // item it's slot, once every thread's copy has landed
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * HALF * C;
    };
    auto done = [&](int it) {                            // the slot is free: stage the item NS ahead into it
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    auto scores = [&](float (&s)[4][4], const uint4* sl) {
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            uint32_t a[4];
            if constexpr (QREG) {
                a[0] = qa[k][0]; a[1] = qa[k][1]; a[2] = qa[k][2]; a[3] = qa[k][3];
            } else {
                ldmatrix4(a, qs + sw<D>(warp * 16 + lane % 16, 2 * k + lane / 16));
            }
#pragma unroll
            for (int n = 0; n < 4; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], a, b[0], b[1]);
                    mma0(s[n + 1], a, b[2], b[3]);
                } else {
                    mma(s[n], a, b[0], b[1]);
                    mma(s[n + 1], a, b[2], b[3]);
                }
            }
        }
    };
    auto values = [&](const uint32_t (&pa0)[4], const uint32_t (&pa1)[4], const uint4* sl) {
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa0, b[0], b[1]);
            mma(o[i + 1], pa0, b[2], b[3]);
        }
#pragma unroll
        for (int i = 0; i < NB; i += 2) {
            uint32_t b[4];
            ldmatrix4t(b, sl + sw<D>(16 + 8 * (mt % 2) + lane % 8, i + mt / 2));
            mma(o[i], pa1, b[0], b[1]);
            mma(o[i + 1], pa1, b[2], b[3]);
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = 4 * t;
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[0]), take(it));
        done(it);
        scores(*reinterpret_cast<float(*)[4][4]>(&sc[4]), take(it + 1));
        done(it + 1);
        {                                                // _tile's fold of the tile's 64 scores
            const int kb = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = vis(c, pos0) ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                sc[n][1] = vis(c + 1, pos0) ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                sc[n][2] = vis(c, pos1) ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                sc[n][3] = vis(c + 1, pos1) ? __fmul_rn(sc[n][3], scale) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
            const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
            const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
            float c0[8], c1[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                sc[n][0] = act0 && vis(c, pos0) ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                sc[n][1] = act0 && vis(c + 1, pos0) ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                sc[n][2] = act1 && vis(c, pos1) ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                sc[n][3] = act1 && vis(c + 1, pos1) ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        values(pa[0], pa[1], take(it + 2));
        done(it + 2);
        values(pa[2], pa[3], take(it + 3));
        done(it + 3);
    }
    wait_group<0>();
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

// pattn2_kernel: pattn_kernel's arithmetic, op for op and in the same order (so the same bits), with the staging and
// the masking as parameters: SLOT keys a staging slot (32 as pattn_kernel, or 64: half the barriers), NS slots,
// FAST (a tile every row of the warp sees in full skips the per-score visibility tests; the values it computes are
// the ones the tests would let through), MINB (__launch_bounds__' blocks a multiprocessor).
template <int D, int WARPS, int HPC, int NS, int SLOT, bool FAST, int MINB>
__global__ void __launch_bounds__(32 * WARPS, MINB)
pattn2_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
              const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
              int G, int nkeys, int causal, float scale) {
    static_assert(D == 128 && (SLOT == 32 || SLOT == 64), "head dim 128; 32- or 64-key slots");
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, RB = WARPS / HPC, THREADS = 32 * WARPS;
    constexpr int KS = BN / SLOT, IT = 2 * KS, NT = SLOT / 8, PS = SLOT / 16;  // slots per K (and V) of a tile
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = blockIdx.x * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = causal ? (p0 + last) / BN + 1 : (nkeys + BN - 1) / BN, items = IT * tiles, limit = nkeys;
    auto stage = [&](int item) {
        const int j = item % IT;
        const __nv_bfloat16* src = j < KS ? kc : vc;
        const int key0 = (item / IT) * BN + (j % KS) * SLOT;
        uint4* dst = slots + (item % NS) * SLOT * C;
        for (int i = threadIdx.x; i < SLOT * C; i += THREADS) {
            const int key = key0 + i / C;
            const bool in = key < limit;
            cp16z(dst + sw<D>(i / C, i % C), src + ((int64_t)(in ? key : 0) * HK + hk) * D + (i % C) * 8, in);
        }
    };
    uint32_t qa[K16][4];
    {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    auto vis = [&](int c, int pos) { return c < nkeys && (!causal || c <= pos); };
    auto take = [&](int it) -> const uint4* {
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * SLOT * C;
    };
    auto done = [&](int it) {
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    auto scores = [&](float (*s)[4], const uint4* sl) {  // NT n8 tiles of the slot's keys
#pragma unroll
        for (int k = 0; k < K16; ++k) {
#pragma unroll
            for (int n = 0; n < NT; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], qa[k], b[0], b[1]);
                    mma0(s[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma(s[n], qa[k], b[0], b[1]);
                    mma(s[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
    };
    auto values = [&](int pa0, const uint4* sl) {  // the slot's PS 16-key blocks, pa[pa0 ..]
#pragma unroll
        for (int kk = 0; kk < PS; ++kk) {
#pragma unroll
            for (int i = 0; i < NB; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, sl + sw<D>(16 * kk + 8 * (mt % 2) + lane % 8, i + mt / 2));
                mma(o[i], pa[pa0 + kk], b[0], b[1]);
                mma(o[i + 1], pa[pa0 + kk], b[2], b[3]);
            }
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = IT * t;
#pragma unroll
        for (int j = 0; j < KS; ++j) {
            scores(&sc[j * NT], take(it + j));
            done(it + j);
        }
        // every key of the tile visible to all 16 rows of this warp
        const bool full = FAST && (t + 1) * BN <= nkeys && (!causal || (t + 1) * BN - 1 <= p0 + wrow);
        {
            const int kb = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;
            if (full) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) sc[n][e] = __fmul_rn(sc[n][e], scale);
                    tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                    tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
                }
            } else {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const int c = kb + 8 * n;
                    sc[n][0] = vis(c, pos0) ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                    sc[n][1] = vis(c + 1, pos0) ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                    sc[n][2] = vis(c, pos1) ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                    sc[n][3] = vis(c + 1, pos1) ? __fmul_rn(sc[n][3], scale) : -INFINITY;
                    tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                    tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
                }
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0, tm0) : m0, n1 = act1 ? fmaxf(m1, tm1) : m1;
            const float a0 = act0 ? (m0 == -INFINITY ? 0.0f : texp(__fsub_rn(m0, n0))) : 1.0f;
            const float a1 = act1 ? (m1 == -INFINITY ? 0.0f : texp(__fsub_rn(m1, n1))) : 1.0f;
            float c0[8], c1[8];
            if (full) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    sc[n][0] = texp(__fsub_rn(sc[n][0], n0));
                    sc[n][1] = texp(__fsub_rn(sc[n][1], n0));
                    sc[n][2] = texp(__fsub_rn(sc[n][2], n1));
                    sc[n][3] = texp(__fsub_rn(sc[n][3], n1));
                    c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                    c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
                }
            } else {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const int c = kb + 8 * n;
                    sc[n][0] = act0 && vis(c, pos0) ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                    sc[n][1] = act0 && vis(c + 1, pos0) ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                    sc[n][2] = act1 && vis(c, pos1) ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                    sc[n][3] = act1 && vis(c + 1, pos1) ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
                    c0[n] = __fadd_rn(sc[n][0], sc[n][1]);
                    c1[n] = __fadd_rn(sc[n][2], sc[n][3]);
                }
            }
            l0 = __fmaf_rn(l0, a0, rowsum(c0));
            l1 = __fmaf_rn(l1, a1, rowsum(c1));
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
#pragma unroll
        for (int j = 0; j < KS; ++j) {
            values(j * PS, take(it + KS + j));
            done(it + KS + j);
        }
    }
    wait_group<0>();
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

// pattn3_kernel: pattn2_kernel's staging with FlashAttention-2's arithmetic (new bits; its own reference): the scale
// rides in the exponent's FMA, p = 2^(s * scale * log2e - m * scale * log2e) on the raw scores and their raw row
// maximum, and each thread keeps its own partial row sums (its 16 scores a tile), reduced across the quad once at the
// end instead of every tile. Fewer multiplies and shuffles a score; the same tiles, masking and output rounding.
template <int D, int WARPS, int HPC, int NS, int SLOT, int MINB>
__global__ void __launch_bounds__(32 * WARPS, MINB)
pattn3_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
              const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
              int G, int nkeys, int causal, float scale) {
    static_assert(D == 128 && (SLOT == 32 || SLOT == 64), "head dim 128; 32- or 64-key slots");
    constexpr int C = D / 8, K16 = D / 16, NB = D / 8, RB = WARPS / HPC, THREADS = 32 * WARPS;
    constexpr int KS = BN / SLOT, IT = 2 * KS, NT = SLOT / 8, PS = SLOT / 16;  // slots per K (and V) of a tile
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int groups = G / HPC, hk = blockIdx.y / groups, hg = blockIdx.y % groups;
    const int row0 = blockIdx.x * 16 * RB, last = min(row0 + 16 * RB, W) - 1;
    const int head = hk * G + hg * HPC + warp % HPC, wrow = row0 + (warp / HPC) * 16;
    const int tiles = causal ? (p0 + last) / BN + 1 : (nkeys + BN - 1) / BN, items = IT * tiles, limit = nkeys;
    auto stage = [&](int item) {
        const int j = item % IT;
        const __nv_bfloat16* src = j < KS ? kc : vc;
        const int key0 = (item / IT) * BN + (j % KS) * SLOT;
        uint4* dst = slots + (item % NS) * SLOT * C;
        for (int i = threadIdx.x; i < SLOT * C; i += THREADS) {
            const int key = key0 + i / C;
            const bool in = key < limit;
            cp16z(dst + sw<D>(i / C, i % C), src + ((int64_t)(in ? key : 0) * HK + hk) * D + (i % C) * 8, in);
        }
    };
    uint32_t qa[K16][4];
    {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r0, W - 1) * H + head) * D);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + ((int64_t)min(r1, W - 1) * H + head) * D);
#pragma unroll
        for (int k = 0; k < K16; ++k) {
            qa[k][0] = r0 < W ? q0[8 * k + qd] : 0u;
            qa[k][1] = r1 < W ? q1[8 * k + qd] : 0u;
            qa[k][2] = r0 < W ? q0[8 * k + qd + 4] : 0u;
            qa[k][3] = r1 < W ? q1[8 * k + qd + 4] : 0u;
        }
    }
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float sc[8][4];
    uint32_t pa[4][4];
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float sl2 = __fmul_rn(scale, LOG2E);  // the scale in the base-2 exponent
    const int pos0 = p0 + wrow + g, pos1 = pos0 + 8;
    auto vis = [&](int c, int pos) { return c < nkeys && (!causal || c <= pos); };
    auto take = [&](int it) -> const uint4* {
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * SLOT * C;
    };
    auto done = [&](int it) {
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    auto scores = [&](float (*s)[4], const uint4* sl) {  // NT n8 tiles of the slot's keys
#pragma unroll
        for (int k = 0; k < K16; ++k) {
#pragma unroll
            for (int n = 0; n < NT; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
                if (k == 0) {
                    mma0(s[n], qa[k], b[0], b[1]);
                    mma0(s[n + 1], qa[k], b[2], b[3]);
                } else {
                    mma(s[n], qa[k], b[0], b[1]);
                    mma(s[n + 1], qa[k], b[2], b[3]);
                }
            }
        }
    };
    auto values = [&](int pa0, const uint4* sl) {  // the slot's PS 16-key blocks, pa[pa0 ..]
#pragma unroll
        for (int kk = 0; kk < PS; ++kk) {
#pragma unroll
            for (int i = 0; i < NB; i += 2) {
                uint32_t b[4];
                ldmatrix4t(b, sl + sw<D>(16 * kk + 8 * (mt % 2) + lane % 8, i + mt / 2));
                mma(o[i], pa[pa0 + kk], b[0], b[1]);
                mma(o[i + 1], pa[pa0 + kk], b[2], b[3]);
            }
        }
    };
    for (int t = 0; t < tiles; ++t) {
        const int it = IT * t;
#pragma unroll
        for (int j = 0; j < KS; ++j) {
            scores(&sc[j * NT], take(it + j));
            done(it + j);
        }
        // every key of the tile visible to all 16 rows of this warp
        const bool full = (t + 1) * BN <= nkeys && (!causal || (t + 1) * BN - 1 <= p0 + wrow);
        {
            const int kb = t * BN + 2 * qd;
            float tm0 = -INFINITY, tm1 = -INFINITY;  // the tile's raw row maxima (masked scores are -inf)
            if (!full) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const int c = kb + 8 * n;
                    if (!vis(c, pos0)) sc[n][0] = -INFINITY;
                    if (!vis(c + 1, pos0)) sc[n][1] = -INFINITY;
                    if (!vis(c, pos1)) sc[n][2] = -INFINITY;
                    if (!vis(c + 1, pos1)) sc[n][3] = -INFINITY;
                }
            }
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                tm0 = fmaxf(tm0, fmaxf(sc[n][0], sc[n][1]));
                tm1 = fmaxf(tm1, fmaxf(sc[n][2], sc[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
            // a row with nothing visible yet keeps max -inf: its exponent base is 0 and every p is 2^-inf = 0
            const float b0 = n0 == -INFINITY ? 0.0f : __fmul_rn(n0, sl2), b1 = n1 == -INFINITY ? 0.0f : __fmul_rn(n1, sl2);
            const float a0 = m0 == -INFINITY ? 0.0f : ex2(__fsub_rn(__fmul_rn(m0, sl2), b0));  // exactly 1 if unchanged
            const float a1 = m1 == -INFINITY ? 0.0f : ex2(__fsub_rn(__fmul_rn(m1, sl2), b1));
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                sc[n][0] = ex2(__fmaf_rn(sc[n][0], sl2, -b0));
                sc[n][1] = ex2(__fmaf_rn(sc[n][1], sl2, -b0));
                sc[n][2] = ex2(__fmaf_rn(sc[n][2], sl2, -b1));
                sc[n][3] = ex2(__fmaf_rn(sc[n][3], sl2, -b1));
                s0 = __fadd_rn(s0, __fadd_rn(sc[n][0], sc[n][1]));
                s1 = __fadd_rn(s1, __fadd_rn(sc[n][2], sc[n][3]));
            }
            l0 = __fmaf_rn(l0, a0, s0);  // this thread's share of the row sum
            l1 = __fmaf_rn(l1, a1, s1);
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[i][0] = __fmul_rn(o[i][0], a0);
                o[i][1] = __fmul_rn(o[i][1], a0);
                o[i][2] = __fmul_rn(o[i][2], a1);
                o[i][3] = __fmul_rn(o[i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[j][0] = pack(sc[2 * j][0], sc[2 * j][1]);
                pa[j][1] = pack(sc[2 * j][2], sc[2 * j][3]);
                pa[j][2] = pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                pa[j][3] = pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
#pragma unroll
        for (int j = 0; j < KS; ++j) {
            values(j * PS, take(it + KS + j));
            done(it + KS + j);
        }
    }
    wait_group<0>();
    for (int x = 1; x <= 2; x *= 2) {  // the quad's partial row sums, in a fixed order
        l0 = __fadd_rn(l0, __shfl_xor_sync(0xffffffffu, l0, x));
        l1 = __fadd_rn(l1, __shfl_xor_sync(0xffffffffu, l1, x));
    }
    const int r0 = wrow + g, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < NB; ++i) {
        const int d = 8 * i + 2 * qd;
        if (r0 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) =
                pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) =
                pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}

// pattn4_kernel: pattn_kernel's arithmetic for every row, op for op (so its bits), with MT 16-row tiles a warp: each
// K and V fragment loaded from shared memory feeds MT MMAs instead of one, halving (MT = 2) the shared-memory
// traffic a FLOP. The queries stay in shared memory (ldmatrix a k step); 64-key slots; unmasked full tiles.
template <int D, int WARPS, int NS, int MT, int MINB>
__global__ void __launch_bounds__(32 * WARPS, MINB)
pattn4_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
              const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
              int G, int nkeys, int causal, float scale) {
    static_assert(D == 128, "head dim 128");
    constexpr int SLOT = 64, C = D / 8, K16 = D / 16, NB = D / 8, THREADS = 32 * WARPS, ROWS = 16 * MT * WARPS;
    constexpr int NT = SLOT / 8, PS = SLOT / 16;
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    uint4* qs = smem + NS * SLOT * C;  // ROWS query rows
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int hk = blockIdx.y / G, head = blockIdx.y;
    const int row0 = blockIdx.x * ROWS, last = min(row0 + ROWS, W) - 1;
    const int wrow = row0 + warp * 16 * MT;
    const int tiles = causal ? (p0 + last) / BN + 1 : (nkeys + BN - 1) / BN, items = 2 * tiles, limit = nkeys;
    auto stage = [&](int item) {
        const __nv_bfloat16* src = item % 2 == 0 ? kc : vc;
        const int key0 = (item / 2) * BN;
        uint4* dst = slots + (item % NS) * SLOT * C;
        for (int i = threadIdx.x; i < SLOT * C; i += THREADS) {
            const int key = key0 + i / C;
            const bool in = key < limit;
            cp16z(dst + sw<D>(i / C, i % C), src + ((int64_t)(in ? key : 0) * HK + hk) * D + (i % C) * 8, in);
        }
    };
    for (int i = threadIdx.x; i < ROWS * C; i += THREADS) {
        const int r = i / C, row = row0 + r;
        cp16z(qs + sw<D>(r, i % C), q + ((int64_t)min(row, W - 1) * H + head) * D + (i % C) * 8, row < W);
    }
    commit();
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        if (i < items) stage(i);
        commit();
    }
    float o[MT][NB][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NB; ++i) o[m][i][0] = o[m][i][1] = o[m][i][2] = o[m][i][3] = 0.0f;
    float sc[MT][8][4];
    uint32_t pa[MT][4][4];
    float m0[MT], m1[MT], l0[MT], l1[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) m0[m] = m1[m] = -INFINITY, l0[m] = l1[m] = 0.0f;
    auto vis = [&](int c, int pos) { return c < nkeys && (!causal || c <= pos); };
    auto take = [&](int it) -> const uint4* {
        wait_group<NS - 1>();
        __syncthreads();
        return slots + (it % NS) * SLOT * C;
    };
    auto done = [&](int it) {
        __syncthreads();
        if (it + NS < items) stage(it + NS);
        commit();
    };
    for (int t = 0; t < tiles; ++t) {
        {  // scores: every K fragment feeds MT tiles of 16 rows
            const uint4* sl = take(2 * t);
#pragma unroll
            for (int k = 0; k < K16; ++k) {
                uint32_t a[MT][4];
#pragma unroll
                for (int m = 0; m < MT; ++m)
                    ldmatrix4(a[m], qs + sw<D>(warp * 16 * MT + 16 * m + lane % 16, 2 * k + lane / 16));
#pragma unroll
                for (int n = 0; n < NT; n += 2) {
                    uint32_t b[4];
                    ldmatrix4(b, sl + sw<D>(8 * (n + mt / 2) + lane % 8, 2 * k + mt % 2));
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        if (k == 0) {
                            mma0(sc[m][n], a[m], b[0], b[1]);
                            mma0(sc[m][n + 1], a[m], b[2], b[3]);
                        } else {
                            mma(sc[m][n], a[m], b[0], b[1]);
                            mma(sc[m][n + 1], a[m], b[2], b[3]);
                        }
                    }
                }
            }
            done(2 * t);
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {  // pattn_kernel's fold, per 16-row tile
            const int pos0 = p0 + wrow + 16 * m + g, pos1 = pos0 + 8;
            const bool full = (t + 1) * BN <= nkeys && (!causal || (t + 1) * BN - 1 <= p0 + wrow + 16 * m);
            const int kb = t * BN + 2 * qd;
            float (&s)[8][4] = sc[m];
            float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                s[n][0] = full || vis(c, pos0) ? __fmul_rn(s[n][0], scale) : -INFINITY;
                s[n][1] = full || vis(c + 1, pos0) ? __fmul_rn(s[n][1], scale) : -INFINITY;
                s[n][2] = full || vis(c, pos1) ? __fmul_rn(s[n][2], scale) : -INFINITY;
                s[n][3] = full || vis(c + 1, pos1) ? __fmul_rn(s[n][3], scale) : -INFINITY;
                tm0 = fmaxf(tm0, fmaxf(s[n][0], s[n][1]));
                tm1 = fmaxf(tm1, fmaxf(s[n][2], s[n][3]));
            }
            for (int x = 1; x <= 2; x *= 2) {
                tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, x));
                tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, x));
            }
            const bool act0 = tm0 != -INFINITY, act1 = tm1 != -INFINITY;
            const float n0 = act0 ? fmaxf(m0[m], tm0) : m0[m], n1 = act1 ? fmaxf(m1[m], tm1) : m1[m];
            const float a0 = act0 ? (m0[m] == -INFINITY ? 0.0f : texp(__fsub_rn(m0[m], n0))) : 1.0f;
            const float a1 = act1 ? (m1[m] == -INFINITY ? 0.0f : texp(__fsub_rn(m1[m], n1))) : 1.0f;
            float c0[8], c1[8];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int c = kb + 8 * n;
                s[n][0] = act0 && (full || vis(c, pos0)) ? texp(__fsub_rn(s[n][0], n0)) : 0.0f;
                s[n][1] = act0 && (full || vis(c + 1, pos0)) ? texp(__fsub_rn(s[n][1], n0)) : 0.0f;
                s[n][2] = act1 && (full || vis(c, pos1)) ? texp(__fsub_rn(s[n][2], n1)) : 0.0f;
                s[n][3] = act1 && (full || vis(c + 1, pos1)) ? texp(__fsub_rn(s[n][3], n1)) : 0.0f;
                c0[n] = __fadd_rn(s[n][0], s[n][1]);
                c1[n] = __fadd_rn(s[n][2], s[n][3]);
            }
            l0[m] = __fmaf_rn(l0[m], a0, rowsum(c0));
            l1[m] = __fmaf_rn(l1[m], a1, rowsum(c1));
            m0[m] = n0;
            m1[m] = n1;
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                o[m][i][0] = __fmul_rn(o[m][i][0], a0);
                o[m][i][1] = __fmul_rn(o[m][i][1], a0);
                o[m][i][2] = __fmul_rn(o[m][i][2], a1);
                o[m][i][3] = __fmul_rn(o[m][i][3], a1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pa[m][j][0] = pack(s[2 * j][0], s[2 * j][1]);
                pa[m][j][1] = pack(s[2 * j][2], s[2 * j][3]);
                pa[m][j][2] = pack(s[2 * j + 1][0], s[2 * j + 1][1]);
                pa[m][j][3] = pack(s[2 * j + 1][2], s[2 * j + 1][3]);
            }
        }
        {  // values: every V fragment feeds MT tiles; per output element the keys in order, as pattn_kernel
            const uint4* sl = take(2 * t + 1);
#pragma unroll
            for (int kk = 0; kk < PS; ++kk) {
#pragma unroll
                for (int i = 0; i < NB; i += 2) {
                    uint32_t b[4];
                    ldmatrix4t(b, sl + sw<D>(16 * kk + 8 * (mt % 2) + lane % 8, i + mt / 2));
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        mma(o[m][i], pa[m][kk], b[0], b[1]);
                        mma(o[m][i + 1], pa[m][kk], b[2], b[3]);
                    }
                }
            }
            done(2 * t + 1);
        }
    }
    wait_group<0>();
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = wrow + 16 * m + g, r1 = r0 + 8;
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int d = 8 * i + 2 * qd;
            if (r0 < W)
                *reinterpret_cast<uint32_t*>(out + ((int64_t)r0 * H + head) * D + d) =
                    pack(tdiv(o[m][i][0], l0[m]), tdiv(o[m][i][1], l0[m]));
            if (r1 < W)
                *reinterpret_cast<uint32_t*>(out + ((int64_t)r1 * H + head) * D + d) =
                    pack(tdiv(o[m][i][2], l1[m]), tdiv(o[m][i][3], l1[m]));
        }
    }
}

} // namespace stk_attention

// Head dim 128, eight warps of 16 rows, one head a block (MHA: G = 1), eight staging slots.
template __global__ void stk_attention::pattn_kernel<128, 8, 1, 8>(const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, int, int, float);

// The shipping launch (twin and engine): pattn_kernel's bits, 4 warps of 16 rows, two 64-key slots (32 KB), up to
// three blocks a multiprocessor, unmasked full tiles; 1.19x pattn_kernel at the DiT's 1 MP shape on sm_120, 1.43x on GB10.
template __global__ void stk_attention::pattn2_kernel<128, 4, 1, 2, 64, true, 3>(const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, int, int, float);
