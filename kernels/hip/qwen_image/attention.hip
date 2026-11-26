// Ours: the bf16 prefill attention on gfx11 WMMA (docs/dev/ROCM.md section 3), for the H3 token refiner and, on HIP,
// the DiT. kernels/cuda/qwen_image/attention.cu's contract: q [W, H, 128], k / v [nkeys, HK, 128], out [W, H, 128] bf16,
// query head h reads kv head h / G; causal: query i (position p0 + i) sees keys 0 .. p0 + i, else every key. Its bits
// are its own (WMMA and HIP's exp2 are not mma.sync and ex2.approx), so it is gated against torch's reference op (G0),
// not against CUDA.
//
// Order, per query row: keys in tiles of 64, ascending; no split over keys. A tile:
//   s = Q . K^T            8 k16 chunks of the head dimension, ascending, v_wmma bf16 -> f32
//   x = s * (scale log2 e) one f32 multiply (scale log2 e is one f32 product, once); masked keys -inf
//   m' = max(m, max_tile x)   the tile's maximum: 4 column blocks, then the half-wave's 16 lanes (xor 8, 4, 2, 1)
//   p = exp2(x - m'), a = exp2(m - m')   (m = -inf before the first tile: a = 0)
//   l' = l a + sum_tile p  the tile's sum: 4 column blocks in order, then the half-wave (xor 8, 4, 2, 1)
//   o' = o a + bf16(p) . V 4 k16 chunks of the tile's keys, ascending, v_wmma bf16 -> f32
// then out = bf16_rn(o / l), one IEEE division.
//
// launch: block 128 (4 waves of 16 query rows, one head a block), grid (ceil(W / 64), H), dynamic shared memory 0
// (45,056 bytes static): attention.cu's pattn2 launch and arguments, without its dynamic shared memory.
#include <stdint.h>

#include "../common/wmma_gemm.h"

namespace {

constexpr int D = 128, KT = 64, ROWS = 64, WAVES = 4;
constexpr int LDK = D + 8;   // K tile [64 keys][128 dims], rows of 272 bytes
constexpr int LDV = KT + 8;  // V tile transposed [128 dims][64 keys], rows of 144 bytes
constexpr int LDP = KT + 8;  // a wave's P [16 rows][64 keys]
constexpr float LOG2E = 1.4426950408889634f;

using lr_wmma::bf16x16;
using lr_wmma::f32x8;

// The 16 lanes of a half-wave, combined in a fixed tree.
__device__ __forceinline__ float half_max(float v) {
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, 32));
    return v;
}

__device__ __forceinline__ float half_sum(float v) {
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor(v, m, 32));
    return v;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(128) stk_attention_bf16(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
                                                                      const uint16_t* __restrict__ vc, uint16_t* __restrict__ out,
                                                                      int p0, int W, int H, int HK, int G, int nkeys, int causal,
                                                                      float scale) {
    __shared__ __attribute__((aligned(16))) uint16_t ks[KT * LDK];
    __shared__ __attribute__((aligned(16))) uint16_t vt[D * LDV];
    __shared__ __attribute__((aligned(16))) uint16_t ps[WAVES][16 * LDP];
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, hw = lane / 16, col = lane % 16;
    const int head = blockIdx.y, hk = head / G;
    const int row0 = blockIdx.x * ROWS, wrow = row0 + wave * 16, last = min(row0 + ROWS, W) - 1;
    const int tiles = causal ? (p0 + last) / KT + 1 : (nkeys + KT - 1) / KT;
    const float sl = scale * LOG2E;

    bf16x16 qa[D / 16];  // the lane's query row (wrow + col), 16 dims a chunk; rows past W are zeros
    {
        const int r = wrow + col;
#pragma unroll
        for (int c = 0; c < D / 16; ++c) {
            if (r < W) qa[c] = lr_wmma::frag<bf16x16>(q + ((long long)r * H + head) * D + 16 * c);
            else qa[c] = bf16x16{};
        }
    }
    f32x8 o[D / 16];
#pragma unroll
    for (int j = 0; j < D / 16; ++j) o[j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    float m[8], l[8];  // per register v: the lane's row 2 v + hw of the wave's 16
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        m[v] = -__builtin_huge_valf();
        l[v] = 0.0f;
    }

    for (int t = 0; t < tiles; ++t) {
        const int k0 = t * KT;
        __syncthreads();  // every wave is done with the previous tile's K, V and P
        for (int e = threadIdx.x; e < KT * (D / 8); e += 128) {  // 64 keys x 16 copies of K and of V
            const int key = e / (D / 8), c = (e % (D / 8)) * 8;
            const bool in = k0 + key < nkeys;
            const long long src = ((long long)(k0 + key) * HK + hk) * D + c;
            const uint4 kv = in ? *reinterpret_cast<const uint4*>(kc + src) : make_uint4(0, 0, 0, 0);
            const uint4 vv = in ? *reinterpret_cast<const uint4*>(vc + src) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(ks + key * LDK + c) = kv;
            uint16_t vs[8];
            __builtin_memcpy(vs, &vv, 16);
#pragma unroll
            for (int i = 0; i < 8; ++i) vt[(c + i) * LDV + key] = vs[i];
        }
        __syncthreads();

        f32x8 s[KT / 16];
#pragma unroll
        for (int j = 0; j < KT / 16; ++j) {
            s[j] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int c = 0; c < D / 16; ++c) s[j] = lr_wmma::mma(qa[c], lr_wmma::frag<bf16x16>(ks + (16 * j + col) * LDK + 16 * c), s[j]);
        }
        float a[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int row = wrow + 2 * v + hw;
            float mx = -__builtin_huge_valf();
#pragma unroll
            for (int j = 0; j < KT / 16; ++j) {
                const int key = k0 + 16 * j + col;
                const bool masked = key >= nkeys || (causal && key > p0 + row);
                const float x = masked ? -__builtin_huge_valf() : __fmul_rn(s[j][v], sl);
                s[j][v] = x;
                mx = fmaxf(mx, x);
            }
            mx = half_max(mx);
            const float mn = fmaxf(m[v], mx);
            const float base = mn == -__builtin_huge_valf() ? 0.0f : mn;  // a row with no key yet: p = 0 (o, l still 0)
            a[v] = exp2f(m[v] - base);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < KT / 16; ++j) {
                const float p = exp2f(s[j][v] - base);
                const uint16_t pb = lr_wmma::float_to_bf16_rn(p);
                ps[wave][(2 * v + hw) * LDP + 16 * j + col] = pb;
                sum = __fadd_rn(sum, p);
            }
            l[v] = __fadd_rn(__fmul_rn(l[v], a[v]), half_sum(sum));
            m[v] = mn;
        }
#pragma unroll
        for (int j = 0; j < D / 16; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[j][v] = __fmul_rn(o[j][v], a[v]);
        __syncthreads();  // the wave's P is in LDS (a barrier for every wave: their tile counts are the block's)
        bf16x16 pa[KT / 16];
#pragma unroll
        for (int c = 0; c < KT / 16; ++c) pa[c] = lr_wmma::frag<bf16x16>(ps[wave] + col * LDP + 16 * c);
#pragma unroll
        for (int j = 0; j < D / 16; ++j)
#pragma unroll
            for (int c = 0; c < KT / 16; ++c) o[j] = lr_wmma::mma(pa[c], lr_wmma::frag<bf16x16>(vt + (16 * j + col) * LDV + 16 * c), o[j]);
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int row = wrow + 2 * v + hw;
        if (row >= W) continue;
#pragma unroll
        for (int j = 0; j < D / 16; ++j) out[((long long)row * H + head) * D + 16 * j + col] = lr_wmma::float_to_bf16_rn(o[j][v] / l[v]);
    }
}
