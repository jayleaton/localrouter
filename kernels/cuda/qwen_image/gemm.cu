// Ours: a deterministic bf16 tensor-core GEMM for the text encoder's linears, the VAE's convolutions (after im2col)
// and attention, and the small dense layers. C[M, N] = A[M, K] . B[N, K]^T (+ bias[N]), bf16 in and out, fp32
// accumulation. Every output is accumulated over K tiles of 32 in ascending order, inside mma.sync m16n8k16 (whose
// order per element is fixed), with no split-K: a row's bits depend on neither M, N, the grid nor the other rows.
// Requirements: K % 8 == 0, lda / ldb % 8 == 0, A and B 16-byte aligned (cp.async), ldc even.
#include <cuda_bf16.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace stk_gemm {

using qmm_frag::commit;
using qmm_frag::cp16z;
using qmm_frag::ldmatrix4;
using qmm_frag::mma;

constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;  // rows padded to 80 bytes: ldmatrix without bank conflicts
constexpr int THREADS = 256;                            // 8 warps: 2 along M (64 rows) x 4 along N (32 columns)

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// One K tile of A (BM x BK) and B (BN x BK) into stage s: each thread copies 2 + 2 chunks of 16 bytes.
__device__ __forceinline__ void stage(__nv_bfloat16* sa, __nv_bfloat16* sb, const __nv_bfloat16* A, const __nv_bfloat16* B,
                                      long long lda, long long ldb, int M, int N, int K, int m0, int n0, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int e = threadIdx.x + i * THREADS, r = e >> 2, c = (e & 3) * 8;  // 128 rows x 4 chunks
        const int k = k0 + c;
        const bool ka = k < K;
        cp16z(sa + r * LD + c, A + (long long)min(m0 + r, M - 1) * lda + (ka ? k : 0), ka && m0 + r < M);
        cp16z(sb + r * LD + c, B + (long long)min(n0 + r, N - 1) * ldb + (ka ? k : 0), ka && n0 + r < N);
    }
}

}  // namespace stk_gemm

// launch: block 256, grid (ceil(N / 128), ceil(M / 128)), dynamic shared memory 0 (40,960 bytes static).
extern "C" __global__ void __launch_bounds__(256) stk_gemm_bf16(const __nv_bfloat16* __restrict__ A,
                                                                 const __nv_bfloat16* __restrict__ B,
                                                                 const __nv_bfloat16* __restrict__ bias,
                                                                 __nv_bfloat16* __restrict__ C, int M, int N, int K,
                                                                 long long lda, long long ldb, long long ldc) {
    using namespace stk_gemm;
    __shared__ __align__(16) __nv_bfloat16 sa[2][BM * LD];
    __shared__ __align__(16) __nv_bfloat16 sb[2][BN * LD];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / 4, wn = warp % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    const int tiles = (K + BK - 1) / BK;
    stage(sa[0], sb[0], A, B, lda, ldb, M, N, K, m0, n0, 0);
    commit();
    for (int t = 0; t < tiles; ++t) {
        const int s = t & 1;
        if (t + 1 < tiles) stage(sa[s ^ 1], sb[s ^ 1], A, B, lda, ldb, M, N, K, m0, n0, (t + 1) * BK);
        commit();
        wait_group<1>();
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)  // rows wm*64 + 16i .., the lane's row and k half as ldmatrix x4 wants
                ldmatrix4(a[i], sa[s] + (wm * 64 + i * 16 + lane % 16) * LD + kk + (lane / 16) * 8);
#pragma unroll
            for (int j = 0; j < 4; j += 2) {  // two n8 tiles a load: rows wn*32 + 8j .., k halves by lane / 8
                uint32_t b[4];
                const int mt = lane / 8;
                ldmatrix4(b, sb[s] + (wn * 32 + 8 * (j + mt / 2) + lane % 8) * LD + kk + (mt % 2) * 8);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    mma(acc[i][j], a[i], b[0], b[1]);
                    mma(acc[i][j + 1], a[i], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    const int g = lane / 4, q = lane % 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = n0 + wn * 32 + j * 8 + 2 * q;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * 64 + i * 16 + g + 8 * h;
                if (row >= M) continue;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    if (col + e >= N) continue;
                    float v = acc[i][j][2 * h + e];
                    if (bias) v = __fadd_rn(v, __bfloat162float(bias[col + e]));
                    C[(long long)row * ldc + col + e] = __float2bfloat16_rn(v);
                }
            }
        }
}
