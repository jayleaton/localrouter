// src/tensorfold/cuda/nvfp4/gemm_ck.cu @ v0.6.1 (git blob 6d2d9a9bfbaa; torch includes and host wrappers cut, unnamed namespace -> nvfp4_gemm_ck); written by tools/kernels/sync.py, do not edit
// Prompt GEMM in the checkpoint's own math: NVFP4 rows times NVFP4 weights (block-scaled FP4 mma) and FP8 rows times
// FP8 weights (e4m3 mma), in lane4.cu's layouts. One fp32 chain over K a row: a row's bits never depend on its chunk.

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_frag.cuh"  // was "../kernels/qmm_frag.cuh"
#include "mma4.cuh"
#include "swiglu4.cuh"

namespace nvfp4_gemm_ck {

using namespace qmm_frag;
using namespace mma4;

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int KS>
struct Gemm {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16, NT = BN / WN / 8;          // m16 and n8 tiles a warp
    static constexpr int ROW = MODE == A4 ? 32 : 64;                     // bytes a row a step of 64 inputs
    static constexpr int TILE = MODE == A4 ? 2048 : 4096;                // a stored 64-column tile's step
    static constexpr int X = BM * ROW, SX = MODE == A4 ? BM * 4 : 0;     // a step's rows and row scales,
    static constexpr int W = BN / 64 * TILE, SW = MODE == A4 ? BN * 4 : 0;   // weights and their scales
    static constexpr int STEP = X + SX + W + SW;
    static constexpr int STAGE = (KS * STEP + 127) / 128 * 128;          // KS steps a stage
    static constexpr int SMEM = STAGES * STAGE;
};

// The fused gate|up launch's second weight (up), its factor, and down's input it writes: NVFP4 rows [M, kd/2] and
// scales [kd/64, mpad, 4] under down's global scale qg.
struct Up {
    const uint8_t* w;
    const uint8_t* ws;
    float alpha;
    uint8_t* codes;
    uint8_t* scales;
    float qg;
    int kd;
};

// x: A4 codes [M, K/2] and scales [K/64, mpad, 4]; A8 e4m3 [M, K] in fragment order. w: lane4.cu's words or bytes,
// ws its block scales [npad/64, K/64, 64, 4]. out (M, N) = alpha * the product, every row one K chain. EPI 1, 2:
// w is gate and up.w up, a block their same 128 columns, written as SiLU(gate) * up in NVFP4 (swiglu4.cuh).
template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int KS, bool F32, int EPI = 0>
__global__ void __launch_bounds__(WM * WN * 32) gemm_kernel(
        const uint8_t* __restrict__ x, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ w,
        const uint8_t* __restrict__ ws, float alpha, void* __restrict__ out, int M, int N, int K, int mpad, int npad,
        int group, Up up) {
    using G = Gemm<MODE, BM, BN, WM, WN, STAGES, KS>;
    constexpr bool GU = EPI > 0;
    static_assert(!GU || (MODE == A4 && BM == 128 && BN == 256 && WM == 2 && WN == 4), "128 x 256, 64 x 64 warps");
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, t = lane & 3, q = lane >> 3, rr = lane & 7;
    const int2 at = tile_of(blockIdx.x, M, npad, BM, GU ? BN / 2 : BN, group);
    const int m0 = at.x, n0 = at.y, KG = K / 64, tiles = npad / 64;
    constexpr int CH = G::ROW / 16;

    auto stage = [&](int s) { return buf + s * G::STAGE; };
    auto load = [&](unsigned char* p, int kg) {                     // one step of 64 inputs at p
#pragma unroll
        for (int c = tid; c < BM * CH; c += G::THREADS) {
            const int r = c / CH, ch = c % CH;
            const bool in = m0 + r < M;
            cp16z(p + r * G::ROW + chunk<MODE>(r, ch) * 16,
                  x + static_cast<size_t>(in ? m0 + r : 0) * (K / (MODE == A4 ? 2 : 1)) + kg * G::ROW + ch * 16, in);
        }
        if constexpr (MODE == A4) {
            for (int c = tid; c < BM / 4; c += G::THREADS)
                cp16z(p + G::X + c * 16, xs + (static_cast<size_t>(kg) * mpad + m0 + 4 * c) * 4, m0 + 4 * c < mpad);
        }
        unsigned char* pw = p + G::X + G::SX;
#pragma unroll
        for (int c = tid; c < G::W / 16; c += G::THREADS) {
            const int tl = c / (G::TILE / 16), off = c % (G::TILE / 16), wt = n0 / 64 + (GU ? tl % 2 : tl);
            const size_t src = (static_cast<size_t>(min(wt, tiles - 1)) * KG + kg) * G::TILE + off * 16;
            cp16z(pw + c * 16, (GU && tl >= 2 ? up.w : w) + src, wt < tiles);
        }
        if constexpr (MODE == A4) {
            for (int c = tid; c < G::SW / 16; c += G::THREADS) {
                const int tl = c / 16, off = c % 16, wt = n0 / 64 + (GU ? tl % 2 : tl);
                cp16z(pw + G::W + c * 16, (GU && tl >= 2 ? up.ws : ws) + (static_cast<size_t>(min(wt, tiles - 1)) *
                      KG + kg) * 256 + off * 16, wt < tiles);
            }
        }
    };

    float acc[G::MT][G::NT][4];
#pragma unroll
    for (int i = 0; i < G::MT; ++i)
#pragma unroll
        for (int j = 0; j < G::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    const int KT = KG / KS;                                              // stages over K (KG a multiple of KS)
    auto fill = [&](int s, int kt) {
#pragma unroll
        for (int u = 0; u < KS; ++u) load(stage(s) + u * G::STEP, kt * KS + u);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) fill(s, s);
        commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) fill((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        commit();
#pragma unroll
        for (int u = 0; u < KS; ++u) {
        const unsigned char* p = stage(kt % STAGES) + u * G::STEP;
        const unsigned char* pw = p + G::X + G::SX;
        if constexpr (MODE == A4) {
            const uint32_t* sx = reinterpret_cast<const uint32_t*>(p + G::X);
            const uint32_t* sw = reinterpret_cast<const uint32_t*>(pw + G::W);
            uint32_t a[G::MT][4], sa[G::MT];
#pragma unroll
            for (int i = 0; i < G::MT; ++i) {
                const int base = wm * (BM / WM) + i * 16, r = base + rr + (q & 1) * 8;
                ldmatrix4(a[i], p + r * G::ROW + chunk<MODE>(r, q >> 1) * 16);
                const uint32_t v = sx[base + g + 8 * (t & 1)];   // row g (t 0), g + 8 (t 1): one load, no branch
                sa[i] = v;                         // lanes 2, 3 repeat rows g, g + 8: never read
            }
#pragma unroll
            for (int j = 0; j < G::NT; ++j) {
                const int jj = wn * G::NT + j;
                const uint2 b = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                const uint32_t sb = sw[jj * 8 + g];          // every lane its column's: lane 0's is read
#pragma unroll
                for (int i = 0; i < G::MT; ++i) mma_fp4(acc[i][j], a[i], b.x, b.y, sa[i], sb);
            }
        } else {
            uint4 b[G::NT];
#pragma unroll
            for (int j = 0; j < G::NT; ++j) b[j] = reinterpret_cast<const uint4*>(pw)[(wn * G::NT + j) * 32 + lane];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t a[G::MT][4];
#pragma unroll
                for (int i = 0; i < G::MT; ++i) {
                    const int r = wm * (BM / WM) + i * 16 + rr + (q & 1) * 8;
                    ldmatrix4(a[i], p + r * G::ROW + chunk<MODE>(r, 2 * h + (q >> 1)) * 16);
                }
#pragma unroll
                for (int j = 0; j < G::NT; ++j)
#pragma unroll
                    for (int i = 0; i < G::MT; ++i)
                        mma_fp8(acc[i][j], a[i], h ? b[j].z : b[j].x, h ? b[j].w : b[j].y);
            }
        }
        }
    }
    if constexpr (GU) {                                    // row-major rows [M, kd/2], scales [kd/64, mpad, 4]
        wait<0>();
        auto put = [&](int r, int c, int b, uint32_t lo, uint32_t hi) {
            if (m0 + r < M)
                *reinterpret_cast<uint2*>(up.codes + static_cast<size_t>(m0 + r) * (up.kd / 2) + (n0 / 64 + c) * 32 +
                                          b * 8) = make_uint2(lo, hi);
        };
        auto scales = [&](int r, int c, uint32_t sw) {
            if (m0 + r < mpad)
                *reinterpret_cast<uint32_t*>(up.scales + ((static_cast<size_t>(n0 / 64 + c)) * mpad + m0 + r) * 4) =
                    sw;
        };
        swiglu4::epilogue<EPI>(acc, buf, wm, wn, lane, alpha, up.alpha, up.qg, min(2, tiles - n0 / 64), put, scales);
        return;
    }
#pragma unroll
    for (int i = 0; i < G::MT; ++i)
#pragma unroll
        for (int j = 0; j < G::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + t * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + g + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h] * alpha, v1 = acc[i][j][2 * h + 1] * alpha;
                if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0) {
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    } else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

}  // namespace nvfp4_gemm_ck

// The instantiations gemm_cuda's by_tile launches (tile 0 resolves to 1 or 2; every other value is 1), at both
// modes (A4 = 0, A8 = 1) and both outputs, and gemm_gu_ck_cuda's two epilogues (EPI 1: SwiGLU through bf16,
// EPI 2: fp32).
#define NVFP4_CK(MODE, BM, BN, WM, WN, ST, KS, F32, EPI) \
    template __global__ void nvfp4_gemm_ck::gemm_kernel<MODE, BM, BN, WM, WN, ST, KS, F32, EPI>( \
        const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, float, void*, int, int, int, int, int, int, nvfp4_gemm_ck::Up);
NVFP4_CK(0, 128, 256, 2, 4, 2, 2, true, 0)
NVFP4_CK(0, 64, 128, 2, 4, 4, 1, true, 0)
NVFP4_CK(0, 128, 128, 2, 4, 4, 1, true, 0)
NVFP4_CK(0, 128, 256, 2, 4, 2, 2, false, 0)
NVFP4_CK(0, 64, 128, 2, 4, 4, 1, false, 0)
NVFP4_CK(0, 128, 128, 2, 4, 4, 1, false, 0)
NVFP4_CK(1, 128, 256, 2, 4, 2, 2, true, 0)
NVFP4_CK(1, 64, 128, 2, 4, 4, 1, true, 0)
NVFP4_CK(1, 128, 128, 2, 4, 3, 1, true, 0)
NVFP4_CK(1, 128, 256, 2, 4, 2, 2, false, 0)
NVFP4_CK(1, 64, 128, 2, 4, 4, 1, false, 0)
NVFP4_CK(1, 128, 128, 2, 4, 3, 1, false, 0)
NVFP4_CK(0, 128, 256, 2, 4, 3, 2, false, 2)
NVFP4_CK(0, 128, 256, 2, 4, 3, 2, false, 1)
