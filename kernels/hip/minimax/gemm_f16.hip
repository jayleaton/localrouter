// Ours: kernels/cuda/minimax/gemm_f16.cu's contract on gfx11 WMMA (docs/dev/ROCM.md section 3): the deterministic fp16
// GEMM of the MiniMax VAE decoders (every linear, the post-quant 1x1x1 convolution, both GEMMs of the attention).
// C[M, N] = A[M, K] . B[N, K]^T (+ bias[N]), fp32 accumulation over the k16 chunks of K in ascending order
// (common/wmma_gemm.h), no split-K. Epilogue, per element, in this order (each torch op of the decoder is one rounding):
//   v = acc (+ bias, fp32 add);  h = half(v);                                  the Linear's output
//   with res: C = half(float(res) + float(h) * float(rscale[col]))              torch.addcmul(res, h, rscale), fp16
// (the product of two halves is exact in fp32, so a fused multiply-add and the written-out multiply and add agree; the add
// rounds in fp32, then to half). res may alias C (same element, same thread, read before the store).
//
// Batched: grid z picks a batch; A, B, C (and res) advance by sA, sB, sC (sR) elements.
//
// Four kernels, one arithmetic, under gemm_f16.cu's names, arguments, block sizes and grids (`vae_video_ops.zig`'s table
// and `Ops.pick` serve both vendors). They differ only in what the bits do not depend on: the block tile, the waves a
// block and the grid order. Every element sees the same k16 chunks in the same order (wmma_map.h), so the four are
// bit-equal, as on CUDA. Each uses static LDS (two stages, at most 61,440 bytes): no dynamic shared memory on HIP.
// Requirements: K % 8 == 0, lda / ldb % 8 == 0, A and B 16-byte aligned.
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "../common/wmma_gemm.h"

namespace {

using CfgRef = lr_wmma::Cfg<128, 128, 2, 4, false>;  // the first version's tile and grid (n, m)
using CfgWide = lr_wmma::Cfg<256, 128, 4, 2, true>;  // the linears; 8 waves of 64 x 64
using CfgStd = lr_wmma::Cfg<128, 128, 2, 4, true>;   // K <= 64: scores, the embeddings
using CfgNarrow = lr_wmma::Cfg<64, 64, 2, 2, true>;  // N <= 64: P . V; 4 waves of 32 x 32

__device__ __forceinline__ float h2f(uint16_t h) {
    __half x;
    __builtin_memcpy(&x, &h, 2);
    return __half2float(x);
}

__device__ __forceinline__ uint16_t f2h(float f) {
    const __half x = __float2half_rn(f);
    uint16_t h;
    __builtin_memcpy(&h, &x, 2);
    return h;
}

template <class Cf>
__device__ __forceinline__ void gemm16(const uint16_t* A, const uint16_t* B, const uint16_t* __restrict__ bias, uint16_t* C, int M,
                                       int N, int K, long long lda, long long ldb, long long ldc, long long sA, long long sB,
                                       long long sC, const uint16_t* res, const uint16_t* __restrict__ rscale, long long ldr,
                                       long long sR) {
    A += (long long)blockIdx.z * sA;
    B += (long long)blockIdx.z * sB;
    C += (long long)blockIdx.z * sC;
    if (res) res += (long long)blockIdx.z * sR;
    lr_wmma::gemm<Cf, lr_wmma::f16x16>(A, B, M, N, K, lda, ldb, [&](int row, int col, float v) {
        if (bias) v = __fadd_rn(v, h2f(bias[col]));
        uint16_t o = f2h(v);
        if (res) o = f2h(__fadd_rn(h2f(res[(long long)row * ldr + col]), __fmul_rn(h2f(o), h2f(rscale[col]))));
        C[(long long)row * ldc + col] = o;
    });
}

}  // namespace

#define LR_GEMM_F16_ARGS                                                                                                          \
    const uint16_t *__restrict__ A, const uint16_t *__restrict__ B, const uint16_t *__restrict__ bias, uint16_t *C, int M, int N, \
        int K, long long lda, long long ldb, long long ldc, long long sA, long long sB, long long sC, const uint16_t *res,       \
        const uint16_t *__restrict__ rscale, long long ldr, long long sR
#define LR_GEMM_F16_PASS A, B, bias, C, M, N, K, lda, ldb, ldc, sA, sB, sC, res, rscale, ldr, sR

// launch: block 256, grid (ceil(N / 128), ceil(M / 128), batch), shared memory 0 dynamic (40,960 bytes static).
extern "C" __global__ void __launch_bounds__(256) stk_gemm_f16_ref(LR_GEMM_F16_ARGS) { gemm16<CfgRef>(LR_GEMM_F16_PASS); }

// launch: block 256, grid (ceil(M / 256), ceil(N / 128), batch), shared memory 0 dynamic (61,440 bytes static).
extern "C" __global__ void __launch_bounds__(256) stk_gemm_f16(LR_GEMM_F16_ARGS) { gemm16<CfgWide>(LR_GEMM_F16_PASS); }

// launch: block 256, grid (ceil(M / 128), ceil(N / 128), batch), shared memory 0 dynamic (40,960 bytes static).
extern "C" __global__ void __launch_bounds__(256) stk_gemm_f16_s(LR_GEMM_F16_ARGS) { gemm16<CfgStd>(LR_GEMM_F16_PASS); }

// launch: block 128, grid (ceil(M / 64), ceil(N / 64), batch), shared memory 0 dynamic (20,480 bytes static).
extern "C" __global__ void __launch_bounds__(128) stk_gemm_f16_n(LR_GEMM_F16_ARGS) { gemm16<CfgNarrow>(LR_GEMM_F16_PASS); }
