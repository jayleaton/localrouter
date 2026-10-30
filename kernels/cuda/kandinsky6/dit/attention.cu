// The Kandinsky-6 DiT's attention: softmax(q k^T * scale) v, bidirectional, head dim 128, bf16 in and out, streamed over
// 64-key tiles (no score matrix: per block one 64-query x 64-key tile at a time), for every attention of the DiT:
//   visual self-attention (50,220 or 51,840 tokens), audio self-attention (218 rows), the text blocks' masked
//   self-attention (up to 1024 tokens), text cross-attention from either stream (masked keys) and both cross-modal
//   directions (queries and keys of different lengths and models: V->A 50,220 x 218, A->V 218 x 50,220).
// Derived from kernels/cuda/qwen_image/attention.cu's pattn2_kernel<128, 4, 1, 2, 64, true, 3> (blob
// 2888e6833c3b2384bbaa9cfc371cceb51c590eca; TensorFold's prefill attention, Apache-2.0, see NOTICE): the same fold, op
// for op, with the causal prefix replaced by
//   * separate query and key counts (W, nkeys) and explicit row strides for q, k, v and out (elements; a token's heads
//     are contiguous, head h at h * 128), so the projections can be read where the GEMMs wrote them;
//   * an optional key mask (kmask[key] != 0: the key is seen; null: every key), torch's [B, 1, 1, K] boolean key-padding
//     mask; a masked key's score is -inf and its probability exactly 0, as in the unmasked tiles' tail;
//   * a batch index in blockIdx.z with batch strides.
// Arithmetic: scores by mma.sync (fp32), scaled by one fp32 multiply, a running max and sum per row in fp32, p = 2^((s -
// m) * log2 e) (ex2.approx), p rounded to bf16 for the value mma, the output divided by the row sum (div.full) and
// rounded to bf16. Every row's bits are independent of W, of the other rows and of the grid. A row needs at least one
// visible key (the launcher checks the mask).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "frag.cuh"

namespace k6_attn {

using k6_frag::commit;
using k6_frag::cp16z;
using k6_frag::ldmatrix4;
using k6_frag::ldmatrix4t;
using k6_frag::mma;
using k6_frag::mma0;
using k6_frag::wait_group;

constexpr int D = 128, WARPS = 4, NS = 2, SLOT = 64, BN = 64, MINB = 3;
constexpr int C = D / 8, K16 = D / 16, NB = D / 8, THREADS = 32 * WARPS;
constexpr int KS = BN / SLOT, IT = 2 * KS, NT = SLOT / 8, PS = SLOT / 16;
constexpr int ROWS = 16 * WARPS;  // queries a block
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float texp(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(__fmul_rn(x, LOG2E)));
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

} // namespace k6_attn

// launch: block 128 (4 warps of 16 queries), grid (ceil(W / 64), heads, batch), dynamic shared memory
// NS * SLOT * D * 2 = 32,768 bytes. Requirements: ldq, ldk, ldv, ldo and the batch strides even (32-bit query loads and
// output stores) and ldk, ldv % 8 == 0 with k, v 16-byte aligned (cp.async), nkeys >= 1.
extern "C" __global__ void __launch_bounds__(k6_attn::WARPS * 32, k6_attn::MINB)
k6_attention(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
             __nv_bfloat16* __restrict__ out, const uint8_t* __restrict__ kmask, int W, int nkeys, long long ldq,
             long long ldk, long long ldv, long long ldo, long long bq, long long bk, long long bv, long long bo,
             long long bm, float scale) {
    using namespace k6_attn;
    extern __shared__ uint4 smem[];
    uint4* slots = smem;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, qd = lane % 4, mt = lane / 8;
    const int head = blockIdx.y, z = blockIdx.z;
    q += z * bq + head * D;
    k += z * bk + head * D;
    v += z * bv + head * D;
    out += z * bo + head * D;
    if (kmask) kmask += z * bm;
    const int row0 = blockIdx.x * ROWS, wrow = row0 + warp * 16;
    const int tiles = (nkeys + BN - 1) / BN, items = IT * tiles, limit = nkeys;
    auto stage = [&](int item) {
        const int j = item % IT;
        const __nv_bfloat16* src = j < KS ? k : v;
        const long long ld = j < KS ? ldk : ldv;
        const int key0 = (item / IT) * BN + (j % KS) * SLOT;
        uint4* dst = slots + (item % NS) * SLOT * C;
        for (int i = threadIdx.x; i < SLOT * C; i += THREADS) {
            const int key = key0 + i / C;
            const bool in = key < limit;  // past the last key: zeros, never read
            cp16z(dst + sw(i / C, i % C), src + (int64_t)(in ? key : 0) * ld + (i % C) * 8, in);
        }
    };
    uint32_t qa[K16][4];
    {
        const int r0 = wrow + g, r1 = r0 + 8;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r0, W - 1) * ldq);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + (int64_t)min(r1, W - 1) * ldq);
#pragma unroll
        for (int kk = 0; kk < K16; ++kk) {
            qa[kk][0] = r0 < W ? q0[8 * kk + qd] : 0u;
            qa[kk][1] = r1 < W ? q1[8 * kk + qd] : 0u;
            qa[kk][2] = r0 < W ? q0[8 * kk + qd + 4] : 0u;
            qa[kk][3] = r1 < W ? q1[8 * kk + qd + 4] : 0u;
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
    auto vis = [&](int c) { return c < nkeys && (kmask == nullptr || kmask[c] != 0); };  // keys every query sees
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
        for (int kk = 0; kk < K16; ++kk) {
#pragma unroll
            for (int n = 0; n < NT; n += 2) {
                uint32_t b[4];
                ldmatrix4(b, sl + sw(8 * (n + mt / 2) + lane % 8, 2 * kk + mt % 2));
                if (kk == 0) {
                    mma0(s[n], qa[kk], b[0], b[1]);
                    mma0(s[n + 1], qa[kk], b[2], b[3]);
                } else {
                    mma(s[n], qa[kk], b[0], b[1]);
                    mma(s[n + 1], qa[kk], b[2], b[3]);
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
                ldmatrix4t(b, sl + sw(16 * kk + 8 * (mt % 2) + lane % 8, i + mt / 2));
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
        // every key of the tile seen by every row: the per-score tests would let every value through unchanged
        const bool full = kmask == nullptr && (t + 1) * BN <= nkeys;
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
                    const bool v0 = vis(c), v1 = vis(c + 1);
                    sc[n][0] = v0 ? __fmul_rn(sc[n][0], scale) : -INFINITY;
                    sc[n][1] = v1 ? __fmul_rn(sc[n][1], scale) : -INFINITY;
                    sc[n][2] = v0 ? __fmul_rn(sc[n][2], scale) : -INFINITY;
                    sc[n][3] = v1 ? __fmul_rn(sc[n][3], scale) : -INFINITY;
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
                    const bool v0 = vis(c), v1 = vis(c + 1);
                    sc[n][0] = act0 && v0 ? texp(__fsub_rn(sc[n][0], n0)) : 0.0f;
                    sc[n][1] = act0 && v1 ? texp(__fsub_rn(sc[n][1], n0)) : 0.0f;
                    sc[n][2] = act1 && v0 ? texp(__fsub_rn(sc[n][2], n1)) : 0.0f;
                    sc[n][3] = act1 && v1 ? texp(__fsub_rn(sc[n][3], n1)) : 0.0f;
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
            *reinterpret_cast<uint32_t*>(out + (int64_t)r0 * ldo + d) = pack(tdiv(o[i][0], l0), tdiv(o[i][1], l0));
        if (r1 < W)
            *reinterpret_cast<uint32_t*>(out + (int64_t)r1 * ldo + d) = pack(tdiv(o[i][2], l1), tdiv(o[i][3], l1));
    }
}
