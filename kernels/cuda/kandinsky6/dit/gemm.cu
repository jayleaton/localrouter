// The Kandinsky-6 DiT's linears. Two kernels, each with one fixed reduction order, so a row's bits depend on neither M,
// N, the grid nor the other rows (row-chunked calls give the same bits as one call):
//
// k6_gemm_bf16: C[M, N] = A[M, K] . B[N, K]^T, bf16 in and out, fp32 accumulation over K tiles of 32 in ascending order
//   inside mma.sync m16n8k16, no split-K. The tiling and staging are kernels/cuda/qwen_image/gemm.cu's stk_gemm_bf16
//   (blob b8a36617767be100793b7d413893bb768ca77c97) unchanged but for k_first; the epilogue adds the DiT's two output
//   forms:
//     EPI_BIAS  out = bf16(acc + float(bias[n]))           (nn.Linear in bf16: the bias joins the fp32 accumulator, one
//                                                            rounding; bias may be null)
//     EPI_GELU  out = bf16(gelu(float(bf16(acc))))         (_GELUProjection: the bias-free proj rounds to bf16, then
//                                                            F.gelu computes in fp32 and rounds again)
//   vec = 1 stages with 16-byte cp.async (K % 8 == 0, lda / ldb % 8 == 0, A and B 16-byte aligned); vec = 0 stages
//   element by element (any K and strides: the 132-feature patch embedding). Both fill the same shared tiles (zeros
//   past K and past M / N), so the arithmetic and the bits are the same. ldc >= N.
//   k_first (0 < k_first < K; anything else is one ascending pass) reduces k in [0, k_first) first, zero-padded to
//   the tile, then k_first, k_first + 32, ... to K: the order of a CUTLASS 2.x kernel (cuBLASLt's sm80 tensor-op
//   algorithms), whose predicated iterators take the partial K tile (K mod its K tile) before the full ones. Only K %
//   16 != 0 can tell the orders apart, so the caller passes K mod the native kernel's K tile there and 0 elsewhere;
//   vec needs k_first % 8 == 0 as well.
//
// k6_gemv_f32: C[M, N] = act(A[M, K]) . float(W[N, K])^T + float(bias[N]) in fp32 for the timestep MLP and the
//   modulations (M: the batch, a handful of rows). A is fp32 or bf16 (in_bf16), act is identity or SiLU (silu), C is
//   fp32 or bf16 (out_bf16). One warp an output column: lane j sums k = j, j + 32, ... in ascending order with fmaf,
//   then a fixed xor butterfly; the bias is added once to the reduced sum. Strict fp32: a diagnostic, no native claim (see
//   k6_gemv_tf32); kept as a diagnostic that isolates TF32 effects in the faithfulness capture.
//
// k6_gemv_tf32: the same contract on TF32 tensor cores. Upstream gives these linears FP32 operands under
//   torch.set_float32_matmul_precision("high") (kandinsky/pipeline/factory.py:159-162 at eac06e39), which permits
//   cuBLASLt a TF32 algorithm without fixing which one SM121 runs. Operands are rounded to TF32 (round: 0 = cvt.rna,
//   1 = low 13 mantissa bits cleared; the caller picks explicitly), multiplied in mma.sync m16n8k8 with fp32
//   accumulation, K ascending in steps of 8, no split-K; the bias joins the fp32 result once. bf16 weights are exact in
//   TF32, so only the activations round. Neither this nor k6_gemv_f32 is known to reproduce native bits: both stay
//   unqualified until SM121 byte fixtures of the native oracle settle the algorithm.
#include <cuda_bf16.h>
#include <stdint.h>

#include "frag.cuh"

namespace k6_gemm {

using k6_frag::commit;
using k6_frag::cp16z;
using k6_frag::ldmatrix4;
using k6_frag::mma;
using k6_frag::wait_group;

constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;  // rows padded to 80 bytes: ldmatrix without bank conflicts
constexpr int THREADS = 256;                            // 8 warps: 2 along M (64 rows) x 4 along N (32 columns)
constexpr int EPI_GELU = 1;  // epi: 0 EPI_BIAS, 1 EPI_GELU

// torch 2.14 GeluCUDAKernelImpl (approximate='none') in opmath fp32: x * 0.5 * (1 + erf(x * M_SQRT1_2)).
__device__ __forceinline__ float gelu(float x) {
    const float alpha = 0.70710678118654752440f;
    return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(x, alpha))));
}

// One chunk of 8 elements of row `row` from k, zeros past `rows` or K, by plain loads.
__device__ __forceinline__ void chunk8(__nv_bfloat16* dst, const __nv_bfloat16* X, long long ld, int row, int rows, int k, int K) {
#pragma unroll
    for (int e = 0; e < 8; ++e)
        dst[e] = row < rows && k + e < K ? X[(long long)row * ld + k + e] : __float2bfloat16_rn(0.0f);
}

// Tile t of the K order: k0, and the end of its segment (zeros from there on); t1 tiles cover [0, kr).
__device__ __forceinline__ void tile_at(int t, int t1, int kr, int K, int& k0, int& kend) {
    if (t < t1) {
        k0 = t * BK;
        kend = kr;
    } else {
        k0 = kr + (t - t1) * BK;
        kend = K;
    }
}

// One K tile of A (BM x BK) and B (BN x BK) into stage s: each thread copies 2 + 2 chunks of 16 bytes; K is the end
// of the tile's segment.
__device__ __forceinline__ void stage(__nv_bfloat16* sa, __nv_bfloat16* sb, const __nv_bfloat16* A, const __nv_bfloat16* B,
                                      long long lda, long long ldb, int M, int N, int K, int m0, int n0, int k0, int vec) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int e = threadIdx.x + i * THREADS, r = e >> 2, c = (e & 3) * 8;  // 128 rows x 4 chunks
        const int k = k0 + c;
        if (vec) {
            const bool ka = k < K;
            cp16z(sa + r * LD + c, A + (long long)min(m0 + r, M - 1) * lda + (ka ? k : 0), ka && m0 + r < M);
            cp16z(sb + r * LD + c, B + (long long)min(n0 + r, N - 1) * ldb + (ka ? k : 0), ka && n0 + r < N);
        } else {
            chunk8(sa + r * LD + c, A, lda, m0 + r, M, k, K);
            chunk8(sb + r * LD + c, B, ldb, n0 + r, N, k, K);
        }
    }
}

}  // namespace k6_gemm

// launch: block 256, grid (ceil(N / 128), ceil(M / 128)), dynamic shared memory 0 (40,960 bytes static).
extern "C" __global__ void __launch_bounds__(256) k6_gemm_bf16(const __nv_bfloat16* __restrict__ A,
                                                                const __nv_bfloat16* __restrict__ B,
                                                                const __nv_bfloat16* __restrict__ bias,
                                                                __nv_bfloat16* __restrict__ C, int M, int N, int K,
                                                                long long lda, long long ldb, long long ldc, int epi,
                                                                int vec, int k_first) {
    using namespace k6_gemm;
    __shared__ __align__(16) __nv_bfloat16 sa[2][BM * LD];
    __shared__ __align__(16) __nv_bfloat16 sb[2][BN * LD];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / 4, wn = warp % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    const int kr = k_first > 0 && k_first < K ? k_first : 0;
    const int t1 = (kr + BK - 1) / BK, tiles = t1 + (K - kr + BK - 1) / BK;
    int k0, kend;
    tile_at(0, t1, kr, K, k0, kend);
    stage(sa[0], sb[0], A, B, lda, ldb, M, N, kend, m0, n0, k0, vec);
    commit();
    for (int t = 0; t < tiles; ++t) {
        const int s = t & 1;
        if (t + 1 < tiles) {
            tile_at(t + 1, t1, kr, K, k0, kend);
            stage(sa[s ^ 1], sb[s ^ 1], A, B, lda, ldb, M, N, kend, m0, n0, k0, vec);
        }
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
                    __nv_bfloat16 r;
                    if (epi == EPI_GELU) {
                        r = __float2bfloat16_rn(gelu(__bfloat162float(__float2bfloat16_rn(v))));
                    } else {
                        if (bias) v = __fadd_rn(v, __bfloat162float(bias[col + e]));
                        r = __float2bfloat16_rn(v);
                    }
                    C[(long long)row * ldc + col + e] = r;
                }
            }
        }
}

namespace k6_gemm {

// torch 2.14 silu in opmath fp32: x / (1 + exp(-x)).
__device__ __forceinline__ float silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); }

}  // namespace k6_gemm

// launch: block 256 (8 warps, 8 columns a block), grid (ceil(N / 8)), dynamic shared memory 0.
extern "C" __global__ void __launch_bounds__(256) k6_gemv_f32(const void* __restrict__ A, const __nv_bfloat16* __restrict__ W,
                                                               const __nv_bfloat16* __restrict__ bias, void* __restrict__ C,
                                                               int M, int N, int K, long long lda, long long ldc, int in_bf16,
                                                               int silu, int out_bf16) {
    using namespace k6_gemm;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const __nv_bfloat16* w = W + (long long)n * K;
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (int k = lane; k < K; k += 32) {
            float a = in_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(A)[(long long)m * lda + k])
                              : static_cast<const float*>(A)[(long long)m * lda + k];
            if (silu) a = k6_gemm::silu(a);
            acc = __fmaf_rn(a, __bfloat162float(w[k]), acc);
        }
#pragma unroll
        for (int x = 16; x >= 1; x /= 2) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, x));
        if (lane == 0) {
            const float v = bias ? __fadd_rn(acc, __bfloat162float(bias[n])) : acc;
            if (out_bf16)
                static_cast<__nv_bfloat16*>(C)[(long long)m * ldc + n] = __float2bfloat16_rn(v);
            else
                static_cast<float*>(C)[(long long)m * ldc + n] = v;
        }
    }
}

namespace k6_gemm {

constexpr int TF32_TRUNCATE = 1;  // round: 0 TF32_RNA, 1 TF32_TRUNCATE

// The TF32 operand bits of an fp32 value: round to nearest with ties away from zero, or truncation.
__device__ __forceinline__ uint32_t tf32(float x, int round) {
    if (round == TF32_TRUNCATE) return __float_as_uint(x) & 0xffffe000u;
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// act(A[row, k]) in fp32, zero past M or K.
__device__ __forceinline__ float activation(const void* A, long long lda, int row, int M, int k, int K, int in_bf16,
                                            int act) {
    if (row >= M || k >= K) return 0.0f;
    const long long i = (long long)row * lda + k;
    const float a = in_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(A)[i]) : static_cast<const float*>(A)[i];
    return act ? silu(a) : a;
}

}  // namespace k6_gemm

// launch: block 256 (8 warps, 8 columns a warp, 64 a block), grid (ceil(N / 64), ceil(M / 16)), dynamic shared memory 0.
// W contiguous [N, K]. Fragments as PTX m16n8k8 tf32: a0 (g, q), a1 (g + 8, q), a2 (g, q + 4), a3 (g + 8, q + 4);
// b0 (k q, n g), b1 (k q + 4, n g); c (g, 2q), (g, 2q + 1), (g + 8, 2q), (g + 8, 2q + 1), g = lane / 4, q = lane % 4.
extern "C" __global__ void __launch_bounds__(256) k6_gemv_tf32(const void* __restrict__ A, const __nv_bfloat16* __restrict__ W,
                                                                const __nv_bfloat16* __restrict__ bias, void* __restrict__ C,
                                                                int M, int N, int K, long long lda, long long ldc, int in_bf16,
                                                                int silu, int out_bf16, int round) {
    using namespace k6_gemm;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const int n0 = blockIdx.x * 64 + warp * 8, m0 = blockIdx.y * 16;
    if (n0 >= N) return;  // warp-uniform
    const int n = n0 + g;
    const __nv_bfloat16* w = W + (long long)min(n, N - 1) * K;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < K; k0 += 8) {
        const int ka = k0 + q, kb = k0 + q + 4;
        uint32_t a[4];
        a[0] = tf32(activation(A, lda, m0 + g, M, ka, K, in_bf16, silu), round);
        a[1] = tf32(activation(A, lda, m0 + g + 8, M, ka, K, in_bf16, silu), round);
        a[2] = tf32(activation(A, lda, m0 + g, M, kb, K, in_bf16, silu), round);
        a[3] = tf32(activation(A, lda, m0 + g + 8, M, kb, K, in_bf16, silu), round);
        const uint32_t b0 = tf32(n < N && ka < K ? __bfloat162float(w[ka]) : 0.0f, round);
        const uint32_t b1 = tf32(n < N && kb < K ? __bfloat162float(w[kb]) : 0.0f, round);
        mma_tf32(acc, a, b0, b1);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int row = m0 + g + 8 * h;
        if (row >= M) continue;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = n0 + 2 * q + e;
            if (col >= N) continue;
            const float v = bias ? __fadd_rn(acc[2 * h + e], __bfloat162float(bias[col])) : acc[2 * h + e];
            if (out_bf16)
                static_cast<__nv_bfloat16*>(C)[(long long)row * ldc + col] = __float2bfloat16_rn(v);
            else
                static_cast<float*>(C)[(long long)row * ldc + col] = v;
        }
    }
}
