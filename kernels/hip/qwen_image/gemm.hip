// Ours: kernels/cuda/qwen_image/gemm.cu's contract on gfx11 WMMA (docs/dev/ROCM.md section 3): the deterministic bf16
// GEMM of the text encoders' linears, the bf16 block linears (weights.zig `.bf16`), the token refiner and the small
// dense layers. C[M, N] = A[M, K] . B[N, K]^T (+ bias[N]), bf16 in and out, fp32 accumulation over the k16 chunks of K in
// ascending order (common/wmma_gemm.h), no split-K, then per element: v = acc (+ bias, one fp32 add), C = bf16_rn(v).
// Same name, arguments and launch as the CUDA kernel, so `TeOps.gemm` launches either.
// Requirements: K % 8 == 0, lda / ldb % 8 == 0, A and B 16-byte aligned.
#include <stdint.h>

#include "../common/wmma_gemm.h"

namespace {
using Tile = lr_wmma::Cfg<128, 128, 2, 4, false>;  // 8 waves, each 64 rows x 32 columns; grid (n, m) as gemm.cu
}

// launch: block 256, grid (ceil(N / 128), ceil(M / 128)), dynamic shared memory 0 (40,960 bytes static).
extern "C" __global__ void __launch_bounds__(256) stk_gemm_bf16(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
                                                                 const uint16_t* __restrict__ bias, uint16_t* __restrict__ C,
                                                                 int M, int N, int K, long long lda, long long ldb, long long ldc) {
    lr_wmma::gemm<Tile, lr_wmma::bf16x16>(A, B, M, N, K, lda, ldb, [&](int row, int col, float v) {
        if (bias) v = __fadd_rn(v, lr_wmma::bf16_to_float(bias[col]));
        C[(long long)row * ldc + col] = lr_wmma::float_to_bf16_rn(v);
    });
}
