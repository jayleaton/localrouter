// Ours: a deterministic fp16 tensor-core GEMM for the video VAE decoder of MiniMax H3 (every linear, the post-quant 1x1x1
// convolution as a GEMM, and both GEMMs of the attention). It is kernels/cuda/qwen_image/gemm.cu's design with half
// inputs and outputs: C[M, N] = A[M, K] . B[N, K]^T (+ bias[N]), fp32 accumulation, every output accumulated over K
// tiles of 32 in ascending order inside mma.sync m16n8k16 (f16 inputs, f32 accumulate; the order per element is fixed by
// the hardware), no split-K, so a row's bits depend on neither M, N, the grid nor the other rows.
//
// Epilogue, per output element, in this order (each torch op of the decoder is one rounding):
//   v = acc (+ bias, fp32 add);  h = half(v);                                  the Linear's output
//   with res: C = half(float(res) + float(h) * float(rscale[col]))              torch.addcmul(res, h, rscale), fp16
// (the product of two halves is exact in fp32, so the fused multiply-add torch's kernel contracts to and the written-out
// multiply and add are the same; the add rounds in fp32, then to half). res may alias C (same element, same thread).
//
// Batched: grid z picks a batch; A, B, C (and res) advance by sA, sB, sC (sR) elements. The attention uses it for the
// 32 heads: A and B are windows into the one qkv buffer (lda = ldb = 3 * 2048, sA = sB = 192, the K window 64 further),
// C the score matrix [heads, S, ldc] or the attention output rows [S, 2048] (ldc = 2048, sC = 64).
//
// Requirements: K % 8 == 0, lda / ldb % 8 == 0, A and B 16-byte aligned (cp.async), ldc even.
#include <cuda_fp16.h>
#include <stdint.h>

namespace stk_gemm16 {

constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;  // rows padded to 80 bytes: ldmatrix without bank conflicts
constexpr int THREADS = 256;                            // 8 warps: 2 along M (64 rows) x 4 along N (32 columns)

__device__ __forceinline__ uint32_t smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem(dst)), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One K tile of A (BM x BK) and B (BN x BK) into stage s: each thread copies 2 + 2 chunks of 16 bytes.
__device__ __forceinline__ void stage(__half* sa, __half* sb, const __half* A, const __half* B, long long lda,
                                      long long ldb, int M, int N, int K, int m0, int n0, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int e = threadIdx.x + i * THREADS, r = e >> 2, c = (e & 3) * 8;  // 128 rows x 4 chunks
        const int k = k0 + c;
        const bool ka = k < K;
        cp16z(sa + r * LD + c, A + (long long)min(m0 + r, M - 1) * lda + (ka ? k : 0), ka && m0 + r < M);
        cp16z(sb + r * LD + c, B + (long long)min(n0 + r, N - 1) * ldb + (ka ? k : 0), ka && n0 + r < N);
    }
}

}  // namespace stk_gemm16

// launch: block 256, grid (ceil(N / 128), ceil(M / 128), batch), dynamic shared memory 0 (40,960 bytes static).
// bias: half [N] or null. res: half, row stride ldr, or null; rscale: half [N] (with res).
extern "C" __global__ void __launch_bounds__(256)
    stk_gemm_f16(const __half* __restrict__ A, const __half* __restrict__ B, const __half* __restrict__ bias,
                 __half* C, int M, int N, int K, long long lda, long long ldb, long long ldc, long long sA, long long sB,
                 long long sC, const __half* res, const __half* __restrict__ rscale, long long ldr, long long sR) {
    using namespace stk_gemm16;
    __shared__ __align__(16) __half sa[2][BM * LD];
    __shared__ __align__(16) __half sb[2][BN * LD];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / 4, wn = warp % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    A += (long long)blockIdx.z * sA;
    B += (long long)blockIdx.z * sB;
    C += (long long)blockIdx.z * sC;
    if (res) res += (long long)blockIdx.z * sR;
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
                    if (bias) v = __fadd_rn(v, __half2float(bias[col + e]));
                    __half o = __float2half_rn(v);
                    if (res) {
                        const float r = __half2float(res[(long long)row * ldr + col + e]);
                        o = __float2half_rn(__fadd_rn(r, __fmul_rn(__half2float(o), __half2float(rscale[col + e]))));
                    }
                    C[(long long)row * ldc + col + e] = o;
                }
            }
        }
}
