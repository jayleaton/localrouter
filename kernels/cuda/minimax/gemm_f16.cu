// Ours: a deterministic fp16 tensor-core GEMM for the video VAE decoder of MiniMax H3 (every linear, the post-quant 1x1x1
// convolution as a GEMM, and both GEMMs of the attention). It is kernels/cuda/qwen_image/gemm.cu's design with half
// inputs and outputs: C[M, N] = A[M, K] . B[N, K]^T (+ bias[N]), fp32 accumulation, every output accumulated over the
// k16 chunks of K in ascending order inside mma.sync m16n8k16 (f16 inputs, f32 accumulate; the order per element is fixed
// by the hardware), no split-K, so a row's bits depend on neither M, N, the grid nor the other rows.
//
// Epilogue, per output element, in this order (each torch op of the decoder is one rounding):
//   v = acc (+ bias, fp32 add);  h = half(v);                                  the Linear's output
//   with res: C = half(float(res) + float(h) * float(rscale[col]))              torch.addcmul(res, h, rscale), fp16
// (the product of two halves is exact in fp32, so the fused multiply-add torch's kernel contracts to and the written-out
// multiply and add are the same; the add rounds in fp32, then to half). res may alias C (same element, same thread).
//
// Batched: grid z picks a batch; A, B, C (and res) advance by sA, sB, sC (sR) elements. The attention uses it for the
// 32 heads (or a group of them): A and B are windows into the one qkv buffer (lda = ldb = 3 * 2048, sA = sB = 192, the K
// window 64 further), C the score matrix [heads, S, ldc] or the attention output rows [S, 2048] (ldc = 2048, sC = 64).
//
// Requirements: K % 8 == 0, lda / ldb % 8 == 0, A and B 16-byte aligned (cp.async), ldc even.
//
// Four kernels, one arithmetic. stk_gemm_f16_ref is the first version (128 x 128 x 32 tile, two stages, grid (n, m)): the
// reference the others are proven bit-equal to (test_vae_video.py "gemm_variants": torch.equal over every shape the
// decoder launches). The others are one template, gemm_tile<Cfg>, and change only what the bits do not depend on: the block
// tile, BK, the number of cp.async stages, the warps a block, the grid order, the vectorised C store, and the skipping of
// m16 / n8 tiles that lie wholly outside the matrix. What every output element sees is unchanged: acc = 0, then for each
// k16 chunk c = 0 .. ceil(K / 32) * 2 - 1 (the reference's chunks, the last of them zero filled past K) one mma.sync of the
// A fragment (rows of the element's m16 tile, k = 16 c ..) and the B fragment (the n8 tile's rows), then the epilogue above.
// A chunk's operands are the same halves in the same fragment positions wherever the block's tile sits, so the hardware
// adds the same products in the same order. No kernel splits K or reorders the chunks.
//   stk_gemm_f16         256 x 128 x 32, three stages, 8 warps of 64 x 64, one block a multiprocessor (92,160 bytes of
//                        dynamic shared memory): the linears (K, N large). 85 FLOP per byte of tile traffic.
//   stk_gemm_f16_s       128 x 128 x 32, two stages, 8 warps of 64 x 32, two blocks a multiprocessor: K <= 64 (scores, the
//                        embeddings), where the epilogue dominates and two resident blocks overlap one's stores with the other's loads.
//   stk_gemm_f16_n       64 x 64 x 32, three stages, 4 warps of 32 x 32: N <= 64 (P . V, K = 1800 long, one n tile).
// Grid of the three: (ceil(M / BM), ceil(N / BN), batch), the m tile fastest: the blocks resident together share one
// n tile's weight rows (read from memory once) and the m tiles of the activations (L2), where the reference's order made
// every row of m tiles re-read the whole weight matrix (15 x the weights of a layer, more than the LPDDR5X can feed).
// Dynamic shared memory above 48 KiB needs cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) first (92,160 bytes).
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

// The same two copies with the shared-memory address already as a 32-bit number (the template kernels keep it in registers).
__device__ __forceinline__ void cp16z_s(uint32_t dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ void ldmatrix4_s(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
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

// The epilogue of one element (the reference's, written once): bias in fp32, one rounding to half, then the addcmul
// residual with its own rounding. `ri` is the element's index in res, `col` its column (bias, rscale).
__device__ __forceinline__ __half epi(float v, const __half* __restrict__ bias, const __half* res,
                                      const __half* __restrict__ rscale, long long ri, int col) {
    if (bias) v = __fadd_rn(v, __half2float(bias[col]));
    __half o = __float2half_rn(v);
    if (res) {
        const float r = __half2float(res[ri]);
        o = __float2half_rn(__fadd_rn(r, __fmul_rn(__half2float(o), __half2float(rscale[col]))));
    }
    return o;
}

// A block tile: TBM x TBN, K tiles of TBK, ST cp.async stages, WM x WN warps (a warp owns TBM / WM rows and TBN / WN columns:
// MI m16 tiles by NJ n8 tiles of accumulators).
template <int TBM, int TBN, int TBK, int ST, int WM, int WN>
struct Cfg {
    static constexpr int BM = TBM, BN = TBN, BK = TBK, STAGES = ST, WARPS_M = WM, WARPS_N = WN;
    static constexpr int THREADS = 32 * WM * WN;
    static constexpr int LD = TBK + 8;                          // padded rows: ldmatrix without bank conflicts (80 bytes at BK 32)
    static constexpr int TM = TBM / WM, TN = TBN / WN;          // the warp's tile
    static constexpr int MI = TM / 16, NJ = TN / 8;
    static constexpr int CPR = TBK / 8;                         // 16-byte chunks a row of a stage
    static constexpr int A_CH = TBM * CPR / THREADS, B_CH = TBN * CPR / THREADS;  // chunks a thread copies a stage
    static constexpr int A_BYTES = TBM * LD * 2, B_BYTES = TBN * LD * 2;           // one stage of A, of B
    static constexpr int SMEM = ST * (A_BYTES + B_BYTES);       // dynamic shared bytes
    static_assert(TBM % (16 * WM) == 0 && TBN % (16 * WN) == 0, "warp tile: whole m16 tiles, n8 tiles in pairs");
    static_assert(TBK % 16 == 0 && (TBM * CPR) % THREADS == 0 && (TBN * CPR) % THREADS == 0, "stage copy: whole chunks a thread");
    static_assert(ST >= 2, "at least double buffered");
};
using CfgWide = Cfg<256, 128, 32, 3, 4, 2>;
using CfgStd = Cfg<128, 128, 32, 2, 2, 4>;
using CfgNarrow = Cfg<64, 64, 32, 3, 2, 2>;

// One K tile into a stage: the chunk's source row is clamped to the matrix and the copy zero-filled when the row or the
// k chunk is out of range (K % 8 == 0, so a chunk is wholly in or out), as the reference does.
template <class Cf>
__device__ __forceinline__ void load_stage(uint32_t sa, uint32_t sb, const __half* A, const __half* B, long long lda,
                                           long long ldb, int M, int N, int K, int m0, int n0, int k0) {
#pragma unroll
    for (int i = 0; i < Cf::A_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS, r = e / Cf::CPR, c = (e % Cf::CPR) * 8;
        const int k = k0 + c;
        const bool ka = k < K;
        cp16z_s(sa + (r * Cf::LD + c) * 2, A + (long long)min(m0 + r, M - 1) * lda + (ka ? k : 0), ka && m0 + r < M);
    }
#pragma unroll
    for (int i = 0; i < Cf::B_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS, r = e / Cf::CPR, c = (e % Cf::CPR) * 8;
        const int k = k0 + c;
        const bool ka = k < K;
        cp16z_s(sb + (r * Cf::LD + c) * 2, B + (long long)min(n0 + r, N - 1) * ldb + (ka ? k : 0), ka && n0 + r < N);
    }
}

template <class Cf>
__device__ __forceinline__ void gemm_tile(const __half* __restrict__ A, const __half* __restrict__ B,
                                          const __half* __restrict__ bias, __half* C, int M, int N, int K, long long lda,
                                          long long ldb, long long ldc, long long sA, long long sB, long long sC,
                                          const __half* res, const __half* __restrict__ rscale, long long ldr,
                                          long long sR) {
    extern __shared__ __align__(16) unsigned char stk_smem[];
    const uint32_t sbase = smem(stk_smem);  // stage s: A at sbase + s * A_BYTES, B at sbase + ST * A_BYTES + s * B_BYTES
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / Cf::WARPS_N, wn = warp % Cf::WARPS_N;
    const int m0 = blockIdx.x * Cf::BM, n0 = blockIdx.y * Cf::BN;
    A += (long long)blockIdx.z * sA;
    B += (long long)blockIdx.z * sB;
    C += (long long)blockIdx.z * sC;
    if (res) res += (long long)blockIdx.z * sR;
    float acc[Cf::MI][Cf::NJ][4];
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i)
#pragma unroll
        for (int j = 0; j < Cf::NJ; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    const int tiles = (K + Cf::BK - 1) / Cf::BK;
    const int kend = (K + 31) / 32 * 32;  // the reference's last k16 chunk ends here (chunks past it are never run)
    // m16 tiles / n8 tile pairs of this warp that hold at least one real row / column (a warp-wide condition)
    bool rv[Cf::MI], cv[Cf::NJ / 2];
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i) rv[i] = m0 + wm * Cf::TM + i * 16 < M;
#pragma unroll
    for (int jp = 0; jp < Cf::NJ / 2; ++jp) cv[jp] = n0 + wn * Cf::TN + jp * 16 < N;

#pragma unroll
    for (int s = 0; s < Cf::STAGES - 1; ++s) {  // prologue: tiles 0 .. STAGES - 2 (an empty group when K has fewer)
        if (s < tiles) load_stage<Cf>(sbase + s * Cf::A_BYTES, sbase + Cf::STAGES * Cf::A_BYTES + s * Cf::B_BYTES, A, B, lda, ldb, M, N, K, m0, n0, s * Cf::BK);
        commit();
    }
    // ldmatrix addresses of this lane: A rows wm * TM + 16 i + lane % 16, k half lane / 16; B rows (pairs of n8 tiles) as the
    // reference: wn * TN + 16 jp + 8 (lane / 16) + lane % 8, k half (lane / 8) % 2
    const int mt = lane / 8;
    const uint32_t a_lane = ((wm * Cf::TM + lane % 16) * Cf::LD + (lane / 16) * 8) * 2;
    const uint32_t b_lane = ((wn * Cf::TN + 8 * (mt / 2) + lane % 8) * Cf::LD + (mt % 2) * 8) * 2;
    for (int t = 0; t < tiles; ++t) {
        wait_group<Cf::STAGES - 2>();  // tile t has landed (the newest STAGES - 2 groups may still fly)
        __syncthreads();               // ... for every thread's copies, and every warp is done with tile t - 1's stage
        {
            const int tn = t + Cf::STAGES - 1, sn = tn % Cf::STAGES;  // refill the stage tile t - 1 was read from
            if (tn < tiles) load_stage<Cf>(sbase + sn * Cf::A_BYTES, sbase + Cf::STAGES * Cf::A_BYTES + sn * Cf::B_BYTES, A, B, lda, ldb, M, N, K, m0, n0, tn * Cf::BK);
            commit();
        }
        const uint32_t sa = sbase + (t % Cf::STAGES) * Cf::A_BYTES;
        const uint32_t sb = sbase + Cf::STAGES * Cf::A_BYTES + (t % Cf::STAGES) * Cf::B_BYTES;
#pragma unroll
        for (int kk = 0; kk < Cf::BK; kk += 16) {
            if (t * Cf::BK + kk >= kend) continue;
            uint32_t a[Cf::MI][4];
#pragma unroll
            for (int i = 0; i < Cf::MI; ++i)
                if (rv[i]) ldmatrix4_s(a[i], sa + a_lane + (i * 16 * Cf::LD + kk) * 2);
#pragma unroll
            for (int jp = 0; jp < Cf::NJ / 2; ++jp) {
                if (!cv[jp]) continue;
                uint32_t b[4];
                ldmatrix4_s(b, sb + b_lane + (jp * 16 * Cf::LD + kk) * 2);
#pragma unroll
                for (int i = 0; i < Cf::MI; ++i) {
                    if (!rv[i]) continue;
                    mma(acc[i][2 * jp], a[i], b[0], b[1]);
                    mma(acc[i][2 * jp + 1], a[i], b[2], b[3]);
                }
            }
        }
    }
    const int g = lane / 4, q = lane % 4;
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i)
#pragma unroll
        for (int j = 0; j < Cf::NJ; ++j) {
            const int col = n0 + wn * Cf::TN + j * 8 + 2 * q;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * Cf::TM + i * 16 + g + 8 * h;
                if (row >= M || col >= N) continue;
                __half* p = C + (long long)row * ldc + col;
                const long long ri = (long long)row * ldr + col;
                const __half o0 = epi(acc[i][j][2 * h], bias, res, rscale, ri, col);  // both read res before either is stored
                if (col + 1 < N) {
                    const __half o1 = epi(acc[i][j][2 * h + 1], bias, res, rscale, ri + 1, col + 1);
                    if ((reinterpret_cast<uintptr_t>(p) & 3) == 0)
                        *reinterpret_cast<uint32_t*>(p) = static_cast<uint32_t>(__half_as_ushort(o0)) | (static_cast<uint32_t>(__half_as_ushort(o1)) << 16);
                    else {
                        p[0] = o0;
                        p[1] = o1;
                    }
                } else {
                    p[0] = o0;
                }
            }
        }
}

}  // namespace stk_gemm16

// launch: block 256, grid (ceil(N / 128), ceil(M / 128), batch), 40,960 bytes of static shared memory, as the first version.
// bias: half [N] or null. res: half, row stride ldr, or null; rscale: half [N] (with res).
extern "C" __global__ void __launch_bounds__(256)
    stk_gemm_f16_ref(const __half* __restrict__ A, const __half* __restrict__ B, const __half* __restrict__ bias,
                     __half* C, int M, int N, int K, long long lda, long long ldb, long long ldc, long long sA,
                     long long sB, long long sC, const __half* res, const __half* __restrict__ rscale, long long ldr,
                     long long sR) {
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

// launch: block 256, grid (ceil(M / 256), ceil(N / 128), batch), dynamic shared memory 92,160 bytes (opt in above 48 KiB).
extern "C" __global__ void __launch_bounds__(256, 1)
    stk_gemm_f16(const __half* __restrict__ A, const __half* __restrict__ B, const __half* __restrict__ bias, __half* C,
                 int M, int N, int K, long long lda, long long ldb, long long ldc, long long sA, long long sB,
                 long long sC, const __half* res, const __half* __restrict__ rscale, long long ldr, long long sR) {
    stk_gemm16::gemm_tile<stk_gemm16::CfgWide>(A, B, bias, C, M, N, K, lda, ldb, ldc, sA, sB, sC, res, rscale, ldr, sR);
}

// launch: block 256, grid (ceil(M / 128), ceil(N / 128), batch), dynamic shared memory 40,960 bytes.
extern "C" __global__ void __launch_bounds__(256, 2)
    stk_gemm_f16_s(const __half* __restrict__ A, const __half* __restrict__ B, const __half* __restrict__ bias, __half* C,
                   int M, int N, int K, long long lda, long long ldb, long long ldc, long long sA, long long sB,
                   long long sC, const __half* res, const __half* __restrict__ rscale, long long ldr, long long sR) {
    stk_gemm16::gemm_tile<stk_gemm16::CfgStd>(A, B, bias, C, M, N, K, lda, ldb, ldc, sA, sB, sC, res, rscale, ldr, sR);
}

// launch: block 128, grid (ceil(M / 64), ceil(N / 64), batch), dynamic shared memory 30,720 bytes.
extern "C" __global__ void __launch_bounds__(128, 3)
    stk_gemm_f16_n(const __half* __restrict__ A, const __half* __restrict__ B, const __half* __restrict__ bias, __half* C,
                   int M, int N, int K, long long lda, long long ldb, long long ldc, long long sA, long long sB,
                   long long sC, const __half* res, const __half* __restrict__ rscale, long long ldr, long long sR) {
    stk_gemm16::gemm_tile<stk_gemm16::CfgNarrow>(A, B, bias, C, M, N, K, lda, ldb, ldc, sA, sB, sC, res, rscale, ldr, sR);
}
