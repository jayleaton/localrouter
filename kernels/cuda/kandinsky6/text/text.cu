// Kandinsky-6 text conditioning: the kernels the Qwen2.5-VL-7B text stack and the CLIP ViT-L/14 text tower need beyond
// the image engine's te.cu / gemm.cu / attention.cu (see README.md for the op table and which kernel runs each op).
// Upstream: kandinsky-6 eac06e39 kandinsky/core/components/text_embedder.py with transformers v5.18.0
// models/qwen2_5_vl/modeling_qwen2_5_vl.py and models/clip/modeling_clip.py.
//
// Every kernel names the torch expression it computes and keeps its rounding points: one fp32 rounding per torch op,
// written as explicit __fmul_rn / __fadd_rn / __fdiv_rn / __fmaf_rn, and a bf16 rounding exactly where the upstream
// tensor is bf16. Built with -O3 --fmad=false (build/kandinsky6.zig), no fast math. extern "C", raw arguments; rows are
// independent in every kernel (a row's bits never depend on the grid or the other rows).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace k6t {

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// The fp32 matmul forms (src/engines/kandinsky6/text/qwen25.zig Fp32Math, the same values as the DiT's
// launch.Fp32Linear). Native runs these matmuls under torch.set_float32_matmul_precision("high")
// (kandinsky/pipeline/factory.py:162), which permits TF32 without fixing the algorithm; none of the forms is a default.
constexpr int kTf32Rna = 0;
constexpr int kTf32Truncate = 1;
constexpr int kFp32 = 2;
static_assert(kTf32Rna == 0 && kTf32Truncate == 1 && kFp32 == 2, "the values of qwen25.zig Fp32Math");

__device__ __forceinline__ float tf32_rna(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return __uint_as_float(r);
}
__device__ __forceinline__ float tf32_truncate(float x) { return __uint_as_float(__float_as_uint(x) & 0xffffe000u); }
// An operand as the form presents it to the multiplier. A product of two TF32 values (11 + 11 significant bits) is
// exact in fp32, so in the TF32 forms an FMA with it rounds once, at the accumulation.
__device__ __forceinline__ float operand(float x, int mode) {
    if (mode == kTf32Rna) return tf32_rna(x);
    if (mode == kTf32Truncate) return tf32_truncate(x);
    return x;  // kFp32
}

// ---- torch 2.14.1 aten/src/ATen/native/cuda/layer_norm_kernel.cu (BSD-3-Clause, see README.md): the Welford
// statistics of vectorized_layer_norm_kernel (T = float), copied from kernels/cuda/kandinsky6/dit/ops.cu. Launch shape
// as torch's: block (32, 4), one block a row, vec_size 4. The expressions are torch's in plain operators; this file is
// built with --fmad=false, so none is contracted (torch's own build may contract them: see README, UNQUALIFIED).
constexpr int vec_size = 4;
constexpr int ln_threads_y = 4;

struct WelfordDataLN {
    float mean;
    float sigma2;
    float count;
};

__device__ __forceinline__ WelfordDataLN cuWelfordOnlineSum(const float val, const WelfordDataLN& curr_sum) {
    float delta = val - curr_sum.mean;
    float new_count = curr_sum.count + 1.f;
    auto fn_rcp_mul = [](auto a, auto b) { return a * (1.0f / b); };
    float new_mean = curr_sum.mean + fn_rcp_mul(delta, new_count);
    return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}

__device__ __forceinline__ WelfordDataLN cuWelfordCombine(const WelfordDataLN dataB, const WelfordDataLN dataA) {
    float delta = dataB.mean - dataA.mean;
    float count = dataA.count + dataB.count;
    float mean, sigma2;
    if (count > 0.f) {
        auto fn_rcp = [](auto a) { return 1.0f / a; };
        auto coef = fn_rcp(count);
        auto nA = dataA.count * coef;
        auto nB = dataB.count * coef;
        mean = nA * dataA.mean + nB * dataB.mean;
        sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
    } else {
        mean = 0.f;
        sigma2 = 0.f;
    }
    return {mean, sigma2, count};
}

// {mean, biased variance} of a row of N fp32 values; buf: 6 floats of shared memory.
__device__ __forceinline__ WelfordDataLN compute_stats(const float* xr, const int N, float* buf) {
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N / vec_size;
    WelfordDataLN wd{0.f, 0.f, 0.f};
    for (int i = thrx; i < n_vec_to_read; i += numx) {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) wd = cuWelfordOnlineSum(xr[i * vec_size + ii], wd);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordDataLN wdB{__shfl_down_sync(0xffffffffu, wd.mean, offset), __shfl_down_sync(0xffffffffu, wd.sigma2, offset),
                          __shfl_down_sync(0xffffffffu, wd.count, offset)};
        wd = cuWelfordCombine(wd, wdB);
    }
    float* meansigmabuf = buf;
    float* countbuf = buf + blockDim.y;
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int wrt_y = threadIdx.y - offset;
            meansigmabuf[2 * wrt_y] = wd.mean;
            meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
            countbuf[wrt_y] = wd.count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            WelfordDataLN wdB{meansigmabuf[2 * threadIdx.y], meansigmabuf[2 * threadIdx.y + 1], countbuf[threadIdx.y]};
            wd = cuWelfordCombine(wd, wdB);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        meansigmabuf[0] = wd.mean;
        meansigmabuf[1] = wd.sigma2 / float(N);
    }
    __syncthreads();
    return WelfordDataLN{meansigmabuf[0], meansigmabuf[1], 0.f};
}

constexpr int kRmsThreads = 256;
constexpr int kAttnDim = 64;    // CLIP ViT-L/14 text head dim (768 / 12)
constexpr int kAttnKeys = 128;  // >= 77, the CLIP context

}  // namespace k6t

using namespace k6t;

// ---- Qwen2.5-VL text stack ---------------------------------------------------------------------------------------

// Qwen2_5_VLRotaryEmbedding.forward (modeling_qwen2_5_vl.py:549-563) for text-only positions p = 0..rows-1 (the same
// on the three MRoPE axes, so the [16, 24, 24] recomposition selects equal values): freqs = inv_freq @ positions (a
// K = 1 fp32 matmul, `mode` the form native's "high" policy runs it in), cos / sin in fp32 (libdevice cosf / sinf, as
// torch's float kernels), times attention_scaling = 1.0 (exact, omitted). The bf16 cast (`.to(x.dtype)`) happens in
// te.cu's stk_rope_half, which reads these tables. With mode = fp32 the product is one rounding (cuBLAS's FFMA with a
// zero addend); with a TF32 form the operands are rounded first and the product is exact.
// launch: block 256, grid ceil(rows * 64 / 256). cos_t, sin_t: [rows, 64] fp32.
extern "C" __global__ void k6t_rope_table(const float* __restrict__ inv, float* __restrict__ cos_t, float* __restrict__ sin_t,
                                          int rows, int mode) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    const int p = i / 64, j = i % 64;
    const float f = __fmul_rn(operand(inv[j], mode), operand((float)p, mode));
    cos_t[i] = cosf(f);
    sin_t[i] = sinf(f);
}

// Qwen2_5_VLRMSNorm.forward (modeling_qwen2_5_vl.py:71-76) on bf16 rows of `dim`, as torch's separate ops:
//   h = x.float(); v = h.pow(2).mean(-1)  ->  sum of fp32 squares (each rounded), times inv_dim (torch's MeanOps
//   multiplies the sum by its float factor 1/N); r = rsqrt(v + eps) (fp32 add, rsqrtf); y = w * (h * r).to(bf16)
//   (fp32 product rounded to bf16, then the bf16 x bf16 weight product rounded to bf16).
// The sum runs in a fixed order: thread-strided partials (thread t adds t, t + 256, ...), then a shared-memory tree.
// Native's reduction order is torch's reduce kernel's (UNQUALIFIED, see README). launch: block 256, grid rows.
extern "C" __global__ void __launch_bounds__(256) k6t_rms_norm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                               __nv_bfloat16* __restrict__ y, int dim, long long ldx, long long ldy,
                                                               float eps, float inv_dim) {
    __shared__ float s[kRmsThreads];
    const __nv_bfloat16* xr = x + (long long)blockIdx.x * ldx;
    __nv_bfloat16* yr = y + (long long)blockIdx.x * ldy;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kRmsThreads) {
        const float v = bf(xr[i]);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int h = kRmsThreads / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s[threadIdx.x] = __fadd_rn(s[threadIdx.x], s[threadIdx.x + h]);
        __syncthreads();
    }
    const float r = rsqrtf(__fadd_rn(__fmul_rn(s[0], inv_dim), eps));
    for (int i = threadIdx.x; i < dim; i += kRmsThreads) {
        const float n = bf(tobf(__fmul_rn(bf(xr[i]), r)));
        yr[i] = tobf(__fmul_rn(bf(w[i]), n));
    }
}

// ---- CLIP text tower (fp32) --------------------------------------------------------------------------------------

// CLIPTextEmbeddings.forward (modeling_clip.py:233-256): token_embedding(ids) + position_embedding(arange), fp32 add.
// launch: block 256, grid (ceil(dim / 256), rows). out [rows, dim].
extern "C" __global__ void k6t_clip_embed(const float* __restrict__ tok, const float* __restrict__ pos, const int* __restrict__ ids,
                                          float* __restrict__ out, int dim) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    const long long r = blockIdx.y;
    out[r * dim + c] = __fadd_rn(tok[(long long)ids[r] * dim + c], pos[r * dim + c]);
}

// nn.LayerNorm(dim, eps) on fp32 rows (layer_norm1 / layer_norm2 / final_layer_norm): torch's
// vectorized_layer_norm_kernel with T = float: Welford statistics above, rstd = rsqrt(var + eps), then
// `gamma * (rstd * (x - mean)) + beta`, every step rounded on its own. dim % 4 == 0.
// launch: block (32, 4), grid rows. y must not alias x.
extern "C" __global__ void __launch_bounds__(128) k6t_layer_norm_f32(const float* __restrict__ x, const float* __restrict__ w,
                                                                     const float* __restrict__ b, float* __restrict__ y, int dim,
                                                                     long long ldx, long long ldy, float eps) {
    __shared__ float buf[ln_threads_y * 3 / 2];
    const float* xr = x + (long long)blockIdx.x * ldx;
    float* yr = y + (long long)blockIdx.x * ldy;
    const WelfordDataLN wd = compute_stats(xr, dim, buf);
    const float rstd_val = rsqrtf(wd.sigma2 + eps);
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int e = thrx * vec_size; e < dim; e += blockDim.x * blockDim.y * vec_size) {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) yr[e + ii] = w[e + ii] * (rstd_val * (xr[e + ii] - wd.mean)) + b[e + ii];
    }
}

// nn.Linear in fp32: C[M, N] = A[M, K] . W[N, K]^T (+ bias[N]); A, W, C contiguous. One thread an output; K in
// ascending order, tiles of 16, one FMA a product (`mode`: the operand form, see `operand`); the bias is added to the
// finished sum (one rounding). No split-K. launch: block (16, 16), grid (ceil(N / 16), ceil(M / 16)).
extern "C" __global__ void __launch_bounds__(256) k6t_linear_f32(const float* __restrict__ A, const float* __restrict__ W,
                                                                 const float* __restrict__ bias, float* __restrict__ C, int M, int N,
                                                                 int K, int mode) {
    __shared__ float sa[16][17];
    __shared__ float sw[16][17];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 16;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int k = k0 + tx;
        sa[ty][tx] = (m0 + ty < M && k < K) ? operand(A[(long long)(m0 + ty) * K + k], mode) : 0.0f;
        sw[ty][tx] = (n0 + ty < N && k < K) ? operand(W[(long long)(n0 + ty) * K + k], mode) : 0.0f;
        __syncthreads();
        const int kn = min(16, K - k0);
        for (int kk = 0; kk < kn; ++kk) acc = __fmaf_rn(sa[ty][kk], sw[tx][kk], acc);
        __syncthreads();
    }
    const int row = m0 + ty, col = n0 + tx;
    if (row < M && col < N) C[(long long)row * N + col] = bias != nullptr ? __fadd_rn(acc, bias[col]) : acc;
}

// QuickGELUActivation (transformers activations.py:122-123): x * sigmoid(1.702 * x) as three torch ops: the scalar
// product (1.702 as float), torch's sigmoid `1 / (1 + exp(-t))` (expf, IEEE division), the product. y may alias x.
// launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6t_quick_gelu(const float* x, float* y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float t = __fmul_rn(v, 1.702f);
    const float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-t)));
    y[i] = __fmul_rn(v, s);
}

// y = a + b (fp32 residual add). y may alias a or b. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6t_add_f32(const float* a, const float* b, float* y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __fadd_rn(a[i], b[i]);
}

// CLIP self-attention in fp32 for the padded 77-token context: q, k, v, o [rows, heads * 64] with row stride ld;
// query i sees keys j <= i with j < nkeys (transformers' causal mask AND the tokenizer's key padding mask, the bool
// mask SDPA receives). Scores: one 64-term FMA chain in dimension order, times `scale`; softmax with the row max
// (sequential), expf, the sum in key order; out = (sum_j p_j v_j, key order) / sum. Strict fp32 (no TF32).
// nkeys >= 1, rows <= 128. launch: block 64, grid (rows, heads). o must not alias q, k or v.
extern "C" __global__ void __launch_bounds__(64) k6t_attention_f32(const float* __restrict__ q, const float* __restrict__ k,
                                                                   const float* __restrict__ v, float* __restrict__ o, int ld,
                                                                   int nkeys, float scale) {
    __shared__ float qs[kAttnDim];
    __shared__ float p[kAttnKeys];
    __shared__ float red[2];
    const int i = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const long long col = (long long)h * kAttnDim;
    qs[t] = q[(long long)i * ld + col + t];
    __syncthreads();
    const int kend = min(i + 1, nkeys);
    for (int j = t; j < kend; j += kAttnDim) {
        const float* kr = k + (long long)j * ld + col;
        float s = 0.0f;
        for (int d = 0; d < kAttnDim; ++d) s = __fmaf_rn(qs[d], kr[d], s);
        p[j] = __fmul_rn(s, scale);
    }
    __syncthreads();
    if (t == 0) {
        float m = -INFINITY;
        for (int j = 0; j < kend; ++j) m = fmaxf(m, p[j]);
        red[0] = m;
    }
    __syncthreads();
    const float m = red[0];
    for (int j = t; j < kend; j += kAttnDim) p[j] = expf(__fsub_rn(p[j], m));
    __syncthreads();
    if (t == 0) {
        float l = 0.0f;
        for (int j = 0; j < kend; ++j) l = __fadd_rn(l, p[j]);
        red[1] = l;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int j = 0; j < kend; ++j) acc = __fmaf_rn(p[j], v[(long long)j * ld + col + t], acc);
    o[(long long)i * ld + col + t] = __fdiv_rn(acc, red[1]);
}

// y = bf16(x): pipeline.py:467 `v.to(dtype=torch.bfloat16)` of the pooled embedding. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6t_cast_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tobf(x[i]);
}
