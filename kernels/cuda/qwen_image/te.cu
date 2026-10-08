// Ours: the Qwen3-VL text encoder's small ops (transformers' Qwen3VLText* order of operations), shared by the twin and
// the Zig engine. extern "C", raw arguments, deterministic; fp32 inside, bf16 in and out.
#include <cuda_bf16.h>

#define TE_BLOCK 256

__device__ __forceinline__ float te_ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ __nv_bfloat16 te_st(float v) { return __float2bfloat16_rn(v); }

// Qwen3VLTextRMSNorm: y = w * bf16(x * rsqrt(mean(x^2) + eps)) with x in fp32 and the weight product in bf16 (a
// bf16 x bf16 multiply: fp32 product rounded once). Rows of D (D = 4096 hidden or 128 per head); the sum per row in
// a fixed order: thread-strided partials, then a shared-memory tree. launch: block 256, grid rows.
extern "C" __global__ void stk_rms_norm_hf(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, long long D,
                                           float eps) {
    __shared__ float s[TE_BLOCK];
    const __nv_bfloat16* xr = x + (long long)blockIdx.x * D;
    __nv_bfloat16* yr = y + (long long)blockIdx.x * D;
    float acc = 0.0f;
    for (long long i = threadIdx.x; i < D; i += TE_BLOCK) {
        const float v = te_ld(xr, i);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int h = TE_BLOCK / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) s[threadIdx.x] = __fadd_rn(s[threadIdx.x], s[threadIdx.x + h]);
        __syncthreads();
    }
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(s[0], (float)D), eps));
    for (long long i = threadIdx.x; i < D; i += TE_BLOCK) {
        const float n = __bfloat162float(te_st(__fmul_rn(te_ld(xr, i), r)));
        yr[i] = te_st(__fmul_rn(te_ld(w, i), n));
    }
}

// RoPE with rotate_half on [rows, heads, 128] in place, in transformers' bf16 order: cos / sin are cast to bf16, and
// out = bf16(bf16(x * cos) + bf16(rotate_half(x) * sin)). f32 tables cos / sin [rows, 64] (one frequency per pair
// i, i + 64; rounded to bf16 here). launch: block 256, grid ceil(rows * heads * 64 / 256).
__device__ __forceinline__ float te_bf(float v) { return __bfloat162float(te_st(v)); }

extern "C" __global__ void stk_rope_half(__nv_bfloat16* x, const float* cos_t, const float* sin_t, long long rows,
                                         long long heads) {
    const long long i = (long long)blockIdx.x * TE_BLOCK + threadIdx.x;
    if (i >= rows * heads * 64) return;
    const long long r = i / (heads * 64), j = i % 64;
    __nv_bfloat16* v = x + (i / 64) * 128;
    const float a = te_ld(v, j), b = te_ld(v, j + 64), c = te_bf(cos_t[r * 64 + j]), s = te_bf(sin_t[r * 64 + j]);
    v[j] = te_st(__fadd_rn(te_bf(__fmul_rn(a, c)), te_bf(__fmul_rn(-b, s))));
    v[j + 64] = te_st(__fadd_rn(te_bf(__fmul_rn(b, c)), te_bf(__fmul_rn(a, s))));
}

// The gated MLP's act_fn(gate) * up: silu rounded to bf16, then the bf16 product (as transformers computes it).
// launch: block 256, grid ceil(n / 256).
extern "C" __global__ void stk_silu_mul(const __nv_bfloat16* g, const __nv_bfloat16* u, __nv_bfloat16* y, long long n) {
    const long long i = (long long)blockIdx.x * TE_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = te_ld(g, i);
    const float sl = __bfloat162float(te_st(__fdiv_rn(v, __fadd_rn(1.0f, expf(-v)))));
    y[i] = te_st(__fmul_rn(sl, te_ld(u, i)));
}

// y = bf16(x + z) (a residual add); y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void stk_add(const __nv_bfloat16* x, const __nv_bfloat16* z, __nv_bfloat16* y, long long n) {
    const long long i = (long long)blockIdx.x * TE_BLOCK + threadIdx.x;
    if (i < n) y[i] = te_st(__fadd_rn(te_ld(x, i), te_ld(z, i)));
}

// out[r, :] = table[ids[r], :] (the token embedding). launch: block 256, grid (ceil(D / 256), rows).
extern "C" __global__ void stk_embed(const __nv_bfloat16* table, const int* ids, __nv_bfloat16* out, long long D) {
    const long long c = (long long)blockIdx.x * TE_BLOCK + threadIdx.x;
    if (c < D) out[(long long)blockIdx.y * D + c] = table[(long long)ids[blockIdx.y] * D + c];
}
