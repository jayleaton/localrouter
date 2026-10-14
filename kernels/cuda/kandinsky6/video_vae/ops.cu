// Kandinsky-6 video VAE: everything but the GEMMs (conv.cu). Channels-last fp16 activations; every kernel has a fixed
// reduction order that depends on neither the device nor the grid's residency, and one parameter block that
// src/engines/kandinsky6/video_vae/launch.zig mirrors field for field. Each comment gives the reference expression
// (pinned upstream kandinsky-6 eac06e39, torch fp16 module) and the exact arithmetic used. No fast math; separate torch ops
// are separate roundings, written with __f*_rn so nvcc cannot contract them.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stddef.h>
#include <stdint.h>

#define K6V_GN_CHUNK 2048  // positions a partial-sum block covers (fixed: the partial count is a function of the shape only)

// ---- GroupNorm (nn.GroupNorm(32, C, eps=1e-6, affine), fp16 in and out) ----------------------------------------------
// Statistics in two passes over x: pass 0 sums x, pass 1 sums (x - mean)^2 with fmaf. A block covers K6V_GN_CHUNK
// positions; its 256 threads are V = C / 8 channel octets by Q = 256 / V position lanes; thread (q, v) accumulates its
// eight channels over positions chunk_start + q, + Q, .. in ascending order. The block's partial of group g is then
//   sum over q = 0 .. Q-1, over the channels c of g ascending: acc[q][c / 8][c % 8]    (sequential fp32, one thread)
// The finalize kernel adds a group's partials: thread t sums chunks t, t + 256, .. ascending, then a fixed halving tree.
//   pass 0: mean = float(double(sum) / double(count));   pass 1: var = float(double(sum) / double(count)),
//   rstd = rsqrtf(var + eps)                              count = positions * C / groups
// Apply: a = rstd * gamma[c], b = fmaf(-a, mean, beta[c]), y = half(fmaf(x, a, b)); with silu (the resnet's
// nonlinearity, a separate torch op on the fp16 norm output): s = float(y), y = half(s / (1 + expf(-s))).
struct K6vGnArgs {
    const __half* x;
    float* partial;      // [chunks][groups]
    float* stats;        // [groups][2]: mean, rstd
    long long positions;  // T * H * W
    int c, groups, pass;
    float eps;
};

extern "C" __global__ void __launch_bounds__(256) k6v_gn_partial(const K6vGnArgs a) {
    __shared__ float red[256][8];
    const int V = a.c / 8, Q = 256 / V, v = threadIdx.x % V, q = threadIdx.x / V;
    const long long p0 = (long long)blockIdx.x * K6V_GN_CHUNK;
    const long long p1 = min(p0 + K6V_GN_CHUNK, a.positions);
    const int cg = a.c / a.groups;
    float acc[8], mean[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0.0f;
        mean[j] = a.pass ? a.stats[((v * 8 + j) / cg) * 2] : 0.0f;
    }
    for (long long p = p0 + q; p < p1; p += Q) {
        const uint4 raw = *reinterpret_cast<const uint4*>(a.x + p * a.c + v * 8);
        const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = __half2float(h[j]);
            if (a.pass) {
                const float d = __fsub_rn(f, mean[j]);
                acc[j] = __fmaf_rn(d, d, acc[j]);
            } else {
                acc[j] = __fadd_rn(acc[j], f);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) red[threadIdx.x][j] = acc[j];
    __syncthreads();
    if (threadIdx.x < a.groups) {
        const int g = threadIdx.x;
        float s = 0.0f;
        for (int qq = 0; qq < Q; ++qq)
            for (int c = g * cg; c < (g + 1) * cg; ++c) s = __fadd_rn(s, red[qq * V + c / 8][c % 8]);
        a.partial[(long long)blockIdx.x * a.groups + g] = s;
    }
}

// grid (groups), block 256; chunks = ceil(positions / K6V_GN_CHUNK).
extern "C" __global__ void __launch_bounds__(256) k6v_gn_finalize(const K6vGnArgs a) {
    __shared__ float red[256];
    const int g = blockIdx.x;
    const long long chunks = (a.positions + K6V_GN_CHUNK - 1) / K6V_GN_CHUNK;
    float s = 0.0f;
    for (long long i = threadIdx.x; i < chunks; i += 256) s = __fadd_rn(s, a.partial[i * a.groups + g]);
    red[threadIdx.x] = s;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (threadIdx.x < h) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + h]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double count = (double)a.positions * (double)(a.c / a.groups);
        const float r = (float)((double)red[0] / count);
        if (a.pass == 0)
            a.stats[g * 2] = r;
        else
            a.stats[g * 2 + 1] = rsqrtf(__fadd_rn(r, a.eps));
    }
}

// ---- GroupNorm statistics, torch's order (policy `torch_welford`, default off) ------------------------------------
// torch 2.14.1 aten/src/ATen/native/cuda/group_norm_kernel.cu GroupNormKernelImplInternal / RowwiseMomentsCUDAKernel
// <c10::Half, float> (lines 43-89, 602-606), block_reduce.cuh BlockReduce / WarpReduce, SharedReduceOps.h WelfordOps
// (correction 0, no sqrt), for one sample (N = 1): one block a group, blockDim 512 (C10_WARP_SIZE = 32 when the group has
// fewer than 512 elements). The group's elements are taken in torch's contiguous NCTHW order, j = c_in_group * positions
// + p, from this channels-last tensor (x[p * c + g * cg + c_in_group]). Thread t folds j = t, t + blockDim, .. :
//   n += 1; nf = float(n); delta = x - mean; mean = mean + delta / nf; m2 = m2 + delta * (x - mean)
// then WarpReduce (shfl_down 16, 8, 4, 2, 1) and, for 512 threads, lane 0 of each warp to shared, warp 0 reduces the 16
// warp results (lanes 16..31 hold the identity) with
//   combine(a, b): a.nf == 0 -> b; b.nf == 0 -> a; delta = b.mean - a.mean; n = a.nf + b.nf; r = b.nf / n;
//                  mean = a.mean + delta * r; m2 = a.m2 + b.m2 + delta * delta * a.nf * r
// project: var = m2 / nf; rstd = rsqrtf(var + eps) where eps is the module's eps cast to the input dtype first
// (static_cast<scalar_t>(eps): half(1e-6) for fp16; the host passes that float). Stats as k6v_gn_apply reads them.
// The variant bits pin nvcc's contraction of torch's source (the fixture job selects the one torch's binary computes):
//   1: reduce's m2 + delta * new_delta as one fmaf      2: combine's mean + delta * r as one fmaf
//   4: combine's (m2a + m2b) + ((delta * delta) * nf_a) * r with the last product and add as one fmaf
struct K6vGnWelfordArgs {
    const __half* x;
    float* stats;  // [groups][2]: mean, rstd
    long long positions;
    int c, groups;
    float eps;
    int variant;
};

struct K6vWelford {
    float mean, m2, nf;
};

__device__ __forceinline__ K6vWelford k6v_welford_combine(K6vWelford a, K6vWelford b, int variant) {
    if (a.nf == 0.0f) return b;
    if (b.nf == 0.0f) return a;
    const float delta = __fsub_rn(b.mean, a.mean);
    const float n = __fadd_rn(a.nf, b.nf);
    const float r = __fdiv_rn(b.nf, n);
    K6vWelford o;
    o.mean = (variant & 2) ? __fmaf_rn(delta, r, a.mean) : __fadd_rn(a.mean, __fmul_rn(delta, r));
    const float m2ab = __fadd_rn(a.m2, b.m2);
    const float t = __fmul_rn(__fmul_rn(delta, delta), a.nf);
    o.m2 = (variant & 4) ? __fmaf_rn(t, r, m2ab) : __fadd_rn(m2ab, __fmul_rn(t, r));
    o.nf = n;
    return o;
}

__device__ __forceinline__ K6vWelford k6v_welford_warp(K6vWelford v, int variant) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        K6vWelford o;
        o.mean = __shfl_down_sync(0xffffffffu, v.mean, off);
        o.m2 = __shfl_down_sync(0xffffffffu, v.m2, off);
        o.nf = __shfl_down_sync(0xffffffffu, v.nf, off);
        v = k6v_welford_combine(v, o, variant);
    }
    return v;
}

// grid (groups), block 512 (or 32 when positions * c / groups < 512).
extern "C" __global__ void __launch_bounds__(512) k6v_gn_welford(const K6vGnWelfordArgs a) {
    __shared__ K6vWelford warps[32];
    const int g = blockIdx.x, cg = a.c / a.groups;
    const long long n = a.positions * cg;
    const __half* xg = a.x + (long long)g * cg;
    K6vWelford v = {0.0f, 0.0f, 0.0f};
    long long cnt = 0, ci = 0, p = threadIdx.x;  // j = ci * positions + p, advanced without a division
    while (p >= a.positions) p -= a.positions, ++ci;
    for (long long j = threadIdx.x; j < n; j += blockDim.x) {
        const float x = __half2float(xg[p * a.c + ci]);
        p += blockDim.x;
        while (p >= a.positions) p -= a.positions, ++ci;
        cnt += 1;
        const float nf = (float)cnt;
        const float delta = __fsub_rn(x, v.mean);
        v.mean = __fadd_rn(v.mean, __fdiv_rn(delta, nf));
        const float nd = __fsub_rn(x, v.mean);
        v.m2 = (a.variant & 1) ? __fmaf_rn(delta, nd, v.m2) : __fadd_rn(v.m2, __fmul_rn(delta, nd));
        v.nf = nf;
    }
    v = k6v_welford_warp(v, a.variant);
    if (blockDim.x > 32) {
        const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
        __syncthreads();
        if (lane == 0) warps[wid] = v;
        __syncthreads();
        const K6vWelford id = {0.0f, 0.0f, 0.0f};
        v = (threadIdx.x < blockDim.x / 32) ? warps[lane] : id;
        if (wid == 0) v = k6v_welford_warp(v, a.variant);
    }
    if (threadIdx.x == 0) {
        const float var = __fdiv_rn(v.m2, v.nf);
        a.stats[g * 2] = v.mean;
        a.stats[g * 2 + 1] = rsqrtf(__fadd_rn(var, a.eps));
    }
}

struct K6vGnApplyArgs {
    const __half* x;
    __half* y;  // may alias x
    const __half* gamma;
    const __half* beta;
    const float* stats;
    long long positions;
    int c, groups, silu;
};

// grid ceil(positions * c / 8 / 256), block 256: one channel octet a thread.
extern "C" __global__ void __launch_bounds__(256) k6v_gn_apply(const K6vGnApplyArgs a) {
    const long long o = (long long)blockIdx.x * 256 + threadIdx.x;
    if (o >= a.positions * (a.c / 8)) return;
    const int c0 = (int)(o % (a.c / 8)) * 8, cg = a.c / a.groups;
    uint4 raw = *reinterpret_cast<const uint4*>(a.x + o * 8);
    __half* h = reinterpret_cast<__half*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = c0 + j, g = c / cg;
        const float mean = a.stats[g * 2], rstd = a.stats[g * 2 + 1];
        const float sc = __fmul_rn(rstd, __half2float(a.gamma[c]));
        const float sh = __fmaf_rn(-sc, mean, __half2float(a.beta[c]));
        __half y = __float2half_rn(__fmaf_rn(__half2float(h[j]), sc, sh));
        if (a.silu) {
            const float s = __half2float(y);
            y = __float2half_rn(__fdiv_rn(s, __fadd_rn(1.0f, expf(-s))));
        }
        h[j] = y;
    }
    *reinterpret_cast<uint4*>(a.y + o * 8) = raw;
}

// ---- Mid-block attention softmax ---------------------------------------------------------------------------------------
// One head (dim 512), scores S = Q K^T in fp32 (conv.cu, out_f32), the frame-causal mask of prepare_causal_attention_mask:
// token i (frame i / spf) sees token j only when j / spf <= i / spf, i.e. j < (i / spf + 1) * spf = valid.
//   x_j = S_ij * scale (scale = 512^-0.5 as float), m = max_j<valid x_j, e_j = expf(x_j - m),
//   l = sum e_j (thread t: j = t, t + 256, .. ascending; then a fixed halving tree), P_ij = half(e_j / l) for j < valid,
//   P_ij = 0 for valid <= j < ldp (masked, and the zero padding conv.cu needs past K).
// grid (rows), block 256; row r of the call is global row row0 + r.
struct K6vSoftmaxArgs {
    const float* s;
    __half* p;
    long long lds, ldp;
    int row0, spf, cols;  // cols: the number of key columns S holds for this block (>= every row's valid)
    float scale;
};

extern "C" __global__ void __launch_bounds__(256) k6v_softmax_rows(const K6vSoftmaxArgs a) {
    __shared__ float red[256];
    const long long r = blockIdx.x;
    const float* s = a.s + r * a.lds;
    __half* p = a.p + r * a.ldp;
    const int valid = ((a.row0 + (int)r) / a.spf + 1) * a.spf;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < valid; j += 256) m = fmaxf(m, __fmul_rn(s[j], a.scale));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (threadIdx.x < h) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + h]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float l = 0.0f;
    for (int j = threadIdx.x; j < valid; j += 256) l = __fadd_rn(l, expf(__fsub_rn(__fmul_rn(s[j], a.scale), m)));
    red[threadIdx.x] = l;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (threadIdx.x < h) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + h]);
        __syncthreads();
    }
    l = red[0];
    for (long long j = threadIdx.x; j < a.ldp; j += 256)
        p[j] = j < valid ? __float2half_rn(__fdiv_rn(expf(__fsub_rn(__fmul_rn(s[j], a.scale), m)), l)) : __ushort_as_half(0);
}

// V [n][c] -> V^T [c][ldn] for P . V (conv.cu wants B as [N][K] rows); columns n .. ldn - 1 are zeroed.
// grid (ceil(ldn / 32), ceil(c / 32)), block (32, 8).
struct K6vTransposeArgs {
    const __half* x;
    __half* y;
    long long n, ldn;
    int c;
};

extern "C" __global__ void __launch_bounds__(256) k6v_transpose(const K6vTransposeArgs a) {
    __shared__ __half t[32][33];
    const long long n0 = (long long)blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += 8) {
        const long long n = n0 + i;
        const int c = c0 + threadIdx.x;
        t[i][threadIdx.x] = (n < a.n && c < a.c) ? a.x[n * a.c + c] : __ushort_as_half(0);
    }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += 8) {
        const int c = c0 + i;
        const long long n = n0 + threadIdx.x;
        if (c < a.c && n < a.ldn) a.y[(long long)c * a.ldn + n] = t[threadIdx.x][i];
    }
}

// ---- Mid-block attention as the reference's SDPA runs it (the decode's, launch.zig `qualified.attn_sdpa`) ------------
// torch 2.14.1 dispatches F.scaled_dot_product_attention (fp16, one head of dim 512, a float mask) on SM121 to the
// mem-efficient kernel fmha_cutlassF_f16_aligned_32x128_gmem_sm80 (PyTorchMemEffAttention::AttentionKernel<half, Sm80,
// aligned, 32 queries, 128 keys, kMaxK 65536>: transformers/cuda/mem_eff_attention/kernel_forward.h, recorded by
// stk_twin.kandinsky6.vae_attn_isolate). One block of 4 warps a 32-query tile; per 128-key block kb:
//   S = Q K^T: fp32, over the head dim ascending (one chain from 0; m16n8k8 in the source, bit 9); warp w the keys
//   32w .. 32w + 31
//   x = S * scale + float(mask), then x * log2e (float(1.4426950408889634)); scale = float(dim^-0.5)
//   mi = max(mi, max of the block's x over its valid keys) (row max over all warps); if m' < mi: r = exp2(m' - mi),
//   s *= r, else r = 1 (mi = -inf: exp2 against 0, mi restored after the block); p = exp2(x - mi) (0 past the keys)
//   row sum: each thread its 8 values of a row (n8 tiles 0..3, then the pair) from 0, then lanes xor 1, xor 2 (one add
//   each), one partial a warp; s = s + w0 + w1 + w2 + w3; m' = mi. P = half(p)
//   O_kb = P V: per 128 output columns (warp w the 32w ..), keys of the block ascending, a fresh fp32 chain
//   output (an fp32 accumulator A in global memory, as the reference's output_accum): first block A = O, middle blocks
//   A = O + r * A, last block (a = 1 / s, s = 1 when 0; b = a * r) out = half(a * O + b * A)
// Which of these nvcc contracts into FMAs in the reference build, and whether its exp2f / reciprocal are approximate, is
// not knowable from the source: `variant` selects them (the attention fixture, check k6-components --stages vae_sdpa,
// runs every combination against torch's bits; exact-or-reject):
//   bit 0: x = fma(S, scale, mask)        (else (S * scale) + mask)
//   bit 1: p's argument fma(x0, log2e, -mi), x0 = S * scale + mask   (else (x0 * log2e) - mi; the max always on x)
//   bit 2: middle A = fma(r, A, O)        (else O + (r * A))
//   bits 3-4: last 0: (a * O) + (b * A); 1: fma(a, O, b * A); 2: fma(b, A, a * O)
//   bits 5-6: exp2 0: exp2f (no fast math); 1: ex2.approx.f32; 2: ex2.approx.ftz.f32
//   bits 7-8: a = 1 / s: 0 div.rn; 1 div.approx.ftz.f32 (fast math); 2 div.full.f32 (-prec-div=false alone)
//   bit 9: both GEMMs as m16n8k8 chains (two k8 mmas a k16: mem_eff_attention/gemm_kernel_utils.h DefaultGemmType for
//          16-bit types on Sm75+ is InstructionShape 16x8x8), else one m16n8k16 a k16
// Every other step is its own __f*_rn op. grid (ceil(n / 32)), block 128.
struct K6vSdpaArgs {
    const __half* q;     // [n][c]
    const __half* k;     // [n][c]
    const __half* vt;    // V^T [c][ldn], zero for n <= j < ldn (k6v_transpose)
    const __half* mask;  // [n][ldm] additive fp16, or null: the frame-causal mask (0 when j / spf <= i / spf, else -inf)
    float* acc;          // [n][c] fp32 output accumulator
    __half* o;           // [n][c]
    long long ldn, ldm;
    int n, c, spf, variant;
    float scale;
    int pad_;
};
static_assert(sizeof(K6vSdpaArgs) == 88, "launch.zig SdpaArgs");

namespace k6v_sdpa {

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// One k16 step: the m16n8k16 fragments (a0/a1 rows g/g+8 at k 2t, a2/a3 at k 2t+8; b0 at k 2t, b1 at k 2t+8) as one mma,
// or as two m16n8k8 (k 0..7, then 8..15).
__device__ __forceinline__ void mma_k16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, bool k8) {
    if (k8) {
        mma8(d, a0, a1, b0);
        mma8(d, a2, a3, b1);
    } else {
        mma(d, a0, a1, a2, a3, b0, b1);
    }
}

__device__ __forceinline__ uint32_t ld2(const __half* p, bool ok) { return ok ? *reinterpret_cast<const uint32_t*>(p) : 0u; }

__device__ __forceinline__ float ex2(float x, int mode) {
    float y;
    if (mode == 0) return exp2f(x);
    if (mode == 1) {
        asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    } else {
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    }
    return y;
}

__device__ __forceinline__ float recip(float d, int mode) {
    float y;
    if (mode == 1) {
        asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(1.0f), "f"(d));
        return y;
    }
    if (mode == 2) {
        asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(1.0f), "f"(d));
        return y;
    }
    return __fdiv_rn(1.0f, d);
}

}  // namespace k6v_sdpa

extern "C" __global__ void __launch_bounds__(128) k6v_sdpa_memeff(const K6vSdpaArgs a) {
    using namespace k6v_sdpa;
    constexpr float kLog2e = 1.4426950408889634074f;
    __shared__ float mi[32], m_prime[32], s_prime[32], out_rescale[32], add[4][32], rowmax[4][32];
    __shared__ bool restore[32];
    __shared__ __align__(16) __half sp[32][128 + 8];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int q0 = blockIdx.x * 32;
    const bool fma_x = a.variant & 1, fma_e = a.variant & 2, fma_mid = a.variant & 4;
    const int last_mode = (a.variant >> 3) & 3, exp_mode = (a.variant >> 5) & 3;
    const int rcp_mode = (a.variant >> 7) & 3;
    const bool k8 = (a.variant >> 9) & 1;
    if (threadIdx.x < 32) {
        s_prime[threadIdx.x] = 0.0f;
        out_rescale[threadIdx.x] = 1.0f;
        m_prime[threadIdx.x] = -INFINITY;
        mi[threadIdx.x] = -INFINITY;
    }
    const int nkb = (a.n + 127) / 128;
    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * 128, max_col = a.n - k0;
        __syncthreads();
        // S for rows q0 + 16 i + g (+ 8), keys k0 + 32 warp + 8 j + 2 t (+ 1)
        float acc[2][4][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        for (int d0 = 0; d0 < a.c; d0 += 16) {
            uint32_t af[2][4], bf[4][2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r0 = q0 + 16 * i + g, r1 = r0 + 8;
                af[i][0] = ld2(a.q + (long long)r0 * a.c + d0 + 2 * t, r0 < a.n);
                af[i][1] = ld2(a.q + (long long)r1 * a.c + d0 + 2 * t, r1 < a.n);
                af[i][2] = ld2(a.q + (long long)r0 * a.c + d0 + 2 * t + 8, r0 < a.n);
                af[i][3] = ld2(a.q + (long long)r1 * a.c + d0 + 2 * t + 8, r1 < a.n);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = k0 + 32 * warp + 8 * j + g;
                bf[j][0] = ld2(a.k + (long long)key * a.c + d0 + 2 * t, key < a.n);
                bf[j][1] = ld2(a.k + (long long)key * a.c + d0 + 2 * t + 8, key < a.n);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_k16(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bf[j][0], bf[j][1], k8);
        }
        // x (and x0, the value before the log2e scaling, for the fused exp argument); the block's row maxima
        float x0[2][4][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rl = 16 * i + 8 * h + g, row = q0 + rl;
                float m = -INFINITY;
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int col = 32 * warp + 8 * j + 2 * t + e, key = k0 + col;
                        float v = acc[i][j][2 * h + e];
                        if (col < max_col) {
                            float bias;
                            if (a.mask != nullptr) {
                                bias = row < a.n ? __half2float(a.mask[(long long)row * a.ldm + key]) : 0.0f;
                            } else {
                                bias = key / a.spf <= row / a.spf ? 0.0f : -INFINITY;
                            }
                            v = fma_x ? __fmaf_rn(v, a.scale, bias) : __fadd_rn(__fmul_rn(v, a.scale), bias);
                        } else {
                            v = __fmul_rn(v, a.scale);
                        }
                        x0[i][j][2 * h + e] = v;
                        const float x = __fmul_rn(v, kLog2e);
                        acc[i][j][2 * h + e] = x;
                        if (col < max_col) m = fmaxf(m, x);
                    }
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (t == 0) rowmax[warp][rl] = m;
            }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int id = threadIdx.x;
            float m = mi[id];
            for (int w = 0; w < 4; ++w) m = fmaxf(m, rowmax[w][id]);
            mi[id] = m;
            restore[id] = false;
            if (m_prime[id] < m) {
                const float r = ex2(__fsub_rn(m_prime[id], m), exp_mode);
                out_rescale[id] = r;
                s_prime[id] = __fmul_rn(s_prime[id], r);
            } else {
                if (m == -INFINITY) {
                    restore[id] = true;
                    mi[id] = 0.0f;
                }
                out_rescale[id] = 1.0f;
            }
        }
        __syncthreads();
        // p = exp2(x - mi), the threads' row sums, P to shared memory
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rl = 16 * i + 8 * h + g;
                const float m = mi[rl];
                float total = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int col = 32 * warp + 8 * j + 2 * t + e;
                        float pv = 0.0f;
                        if (col < max_col) {
                            const float arg = fma_e ? __fmaf_rn(x0[i][j][2 * h + e], kLog2e, -m) : __fsub_rn(acc[i][j][2 * h + e], m);
                            pv = ex2(arg, exp_mode);
                        }
                        total = __fadd_rn(total, pv);
                        sp[rl][col] = __float2half_rn(pv);
                    }
                total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, 1));
                total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, 2));
                if (t == 0) add[warp][rl] = total;
            }
        __syncthreads();
        if (threadIdx.x < 32) {
            const int id = threadIdx.x;
            float total = s_prime[id];
            if (restore[id]) {
                mi[id] = -INFINITY;
            } else {
                m_prime[id] = mi[id];
            }
            for (int w = 0; w < 4; ++w) total = __fadd_rn(total, add[w][id]);
            s_prime[id] = total;
        }
        __syncthreads();
        // O = P V per 128 output columns, then the output accumulator
        const bool first = kb == 0, last = kb == nkb - 1;
        for (int cb = 0; cb < a.c; cb += 128) {
            float o[2][4][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) o[i][j][0] = o[i][j][1] = o[i][j][2] = o[i][j][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                uint32_t af[2][4], bf[4][2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int r0 = 16 * i + g;
                    af[i][0] = *reinterpret_cast<const uint32_t*>(&sp[r0][kk + 2 * t]);
                    af[i][1] = *reinterpret_cast<const uint32_t*>(&sp[r0 + 8][kk + 2 * t]);
                    af[i][2] = *reinterpret_cast<const uint32_t*>(&sp[r0][kk + 2 * t + 8]);
                    af[i][3] = *reinterpret_cast<const uint32_t*>(&sp[r0 + 8][kk + 2 * t + 8]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = cb + 32 * warp + 8 * j + g;
                    const long long key = k0 + kk + 2 * t;
                    bf[j][0] = ld2(a.vt + (long long)col * a.ldn + key, key < a.ldn);
                    bf[j][1] = ld2(a.vt + (long long)col * a.ldn + key + 8, key + 8 < a.ldn);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) mma_k16(o[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bf[j][0], bf[j][1], k8);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int rl = 16 * i + 8 * h + g, row = q0 + rl;
                    if (row >= a.n) continue;
                    const float r = out_rescale[rl];
                    const float denom = s_prime[rl] == 0.0f ? 1.0f : s_prime[rl];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const int col = cb + 32 * warp + 8 * j + 2 * t + e;
                            const long long ix = (long long)row * a.c + col;
                            const float ov = o[i][j][2 * h + e];
                            if (!last) {
                                a.acc[ix] = first ? ov : (fma_mid ? __fmaf_rn(r, a.acc[ix], ov) : __fadd_rn(ov, __fmul_rn(r, a.acc[ix])));
                                continue;
                            }
                            const float al = recip(denom, rcp_mode);
                            float y;
                            if (first) {
                                y = __fmul_rn(al, ov);
                            } else {
                                const float be = __fmul_rn(al, r), src = a.acc[ix];
                                y = last_mode == 1   ? __fmaf_rn(al, ov, __fmul_rn(be, src))
                                    : last_mode == 2 ? __fmaf_rn(be, src, __fmul_rn(al, ov))
                                                     : __fadd_rn(__fmul_rn(al, ov), __fmul_rn(be, src));
                            }
                            a.o[ix] = __float2half_rn(y);
                        }
                }
        }
    }
}

// ---- Boundaries -----------------------------------------------------------------------------------------------------
// Decode input (postprocess_video): latent / scaling_factor on the GPU, then .to(float16). torch 2.14.1's
// div_true_kernel_cuda (BinaryDivTrueKernel.cu:35-52) takes a CPU scalar divisor as a * inv_b in opmath float, with
// inv_b = float(1.0 / double(b)) from the host; the quotient keeps the latent's dtype.
//   fp32 latent (z_f32 = 1, the PiFlow sampler's output): y = half(z * inv_sf), no intermediate rounding
//   bf16 latent (z_f32 = 0):                              y = half(float(bf16(float(z) * inv_sf)))
struct K6vLatentInArgs {
    const void* z;  // float [count] or __nv_bfloat16 [count]
    __half* y;
    long long count;
    float inv_sf;
    int z_f32;
};

extern "C" __global__ void __launch_bounds__(256) k6v_latent_in(const K6vLatentInArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.count) return;
    if (a.z_f32) {
        a.y[i] = __float2half_rn(__fmul_rn(static_cast<const float*>(a.z)[i], a.inv_sf));
    } else {
        const __nv_bfloat16 b =
            __float2bfloat16_rn(__fmul_rn(__bfloat162float(static_cast<const __nv_bfloat16*>(a.z)[i]), a.inv_sf));
        a.y[i] = __float2half_rn(__bfloat162float(b));
    }
}

// I2V encode input (encode_i2va_first_frame): the uint8 image is normalised on the CPU in fp32, tensor / 127.5 - 1.0
// (true division there), then .to(float16): half((float(u) / 127.5f) - 1.0f).
struct K6vRgb8InArgs {
    const uint8_t* x;
    __half* y;
    long long count;
};

extern "C" __global__ void __launch_bounds__(256) k6v_rgb8_in(const K6vRgb8InArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.count) return;
    a.y[i] = __float2half_rn(__fsub_rn(__fdiv_rn((float)a.x[i], 127.5f), 1.0f));
}

// Decode output (postprocess_video): ((frames.clamp(-1, 1) + 1.0) * 127.5).to(uint8), all fp16 ops then a truncating cast:
//   c = clamp(h, -1, 1);  u = half(float(c) + 1);  v = half(float(u) * 127.5);  out = uint8(trunc(float(v)))
// Same layout in and out (frames of [H][W][3]); NaN maps to 0.
struct K6vRgbOutArgs {
    const __half* x;
    uint8_t* y;
    long long count;
};

extern "C" __global__ void __launch_bounds__(256) k6v_rgb_out(const K6vRgbOutArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.count) return;
    const float c = fminf(fmaxf(__half2float(a.x[i]), -1.0f), 1.0f);
    const float u = __half2float(__float2half_rn(__fadd_rn(c, 1.0f)));
    const float v = __half2float(__float2half_rn(__fmul_rn(u, 127.5f)));
    a.y[i] = (uint8_t)__float2uint_rz(v);
}

// Encoder posterior (DiagonalGaussianDistribution over quant_conv's [P][2 * lc] output, channels mean then logvar):
//   lv = clamp(logvar, -30, 20); std = half(expf(float(half(0.5f * lv))))
//   with eps (caller's noise, fp16): x = half(mean + float(half(std * eps)))   .sample()
//   without eps:                     x = mean                                   .mode()
//   latent = half(float(x) * sf)  (latent * vae.config.scaling_factor, fp16), written [P][lc] (the packed THWC layout)
struct K6vPosteriorArgs {
    const __half* moments;
    const __half* eps;  // [P][lc] or null
    __half* mean;       // [P][lc] or null
    __half* stdv;       // [P][lc] or null
    __half* latent;     // [P][lc]
    long long positions;
    int lc;
    float sf;
};

extern "C" __global__ void __launch_bounds__(256) k6v_posterior(const K6vPosteriorArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.positions * a.lc) return;
    const long long p = i / a.lc;
    const int c = (int)(i - p * a.lc);
    const __half mu = a.moments[p * 2 * a.lc + c];
    const float lv = fminf(fmaxf(__half2float(a.moments[p * 2 * a.lc + a.lc + c]), -30.0f), 20.0f);
    const __half sd = __float2half_rn(expf(__half2float(__float2half_rn(__fmul_rn(0.5f, lv)))));
    __half x = mu;
    if (a.eps) {
        const __half se = __float2half_rn(__fmul_rn(__half2float(sd), __half2float(a.eps[i])));
        x = __float2half_rn(__fadd_rn(__half2float(mu), __half2float(se)));
    }
    if (a.mean) a.mean[i] = mu;
    if (a.stdv) a.stdv[i] = sd;
    a.latent[i] = __float2half_rn(__fmul_rn(__half2float(x), a.sf));
}

// ---- Tile blending (AutoencoderKLHunyuanVideo.blend_t / blend_v / blend_h), in place on b ------------------------------
// For y < e along the axis:  b[y] = a[dim_a - e + y] * (1 - y / e) + b[y] * (y / e), torch fp16 with Python float weights:
//   r = double(y) / double(e); w1 = float(1.0 - r); w2 = float(r)
//   b = half(float(half(float(a) * w1)) + float(half(float(b) * w2)))
// a and b are channels-last boxes [f][h][w][ch] (contiguous); the two axes not blended have equal extents.
struct K6vBlendArgs {
    const __half* a;
    __half* b;
    int fa, ha, wa;
    int fb, hb, wb;
    int ch, axis, extent;  // axis 0 time, 1 height, 2 width
};

extern "C" __global__ void __launch_bounds__(256) k6v_blend(const K6vBlendArgs a) {
    const int d0 = a.axis == 0 ? a.extent : a.fb, d1 = a.axis == 1 ? a.extent : a.hb, d2 = a.axis == 2 ? a.extent : a.wb;
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)d0 * d1 * d2 * a.ch) return;
    const int c = (int)(i % a.ch);
    long long r = i / a.ch;
    const int x = (int)(r % d2);
    r /= d2;
    const int y = (int)(r % d1);
    const int f = (int)(r / d1);
    const int pos = a.axis == 0 ? f : (a.axis == 1 ? y : x);
    const int fa = a.axis == 0 ? a.fa - a.extent + f : f;
    const int ya = a.axis == 1 ? a.ha - a.extent + y : y;
    const int xa = a.axis == 2 ? a.wa - a.extent + x : x;
    const double rr = (double)pos / (double)a.extent;
    const float w1 = (float)(1.0 - rr), w2 = (float)rr;
    const long long ib = (((long long)f * a.hb + y) * a.wb + x) * a.ch + c;
    const long long ia = (((long long)fa * a.ha + ya) * a.wa + xa) * a.ch + c;
    const float pa = __half2float(__float2half_rn(__fmul_rn(__half2float(a.a[ia]), w1)));
    const float pb = __half2float(__float2half_rn(__fmul_rn(__half2float(a.b[ib]), w2)));
    a.b[ib] = __float2half_rn(__fadd_rn(pa, pb));
}

// Box copy between channels-last tensors (tile crop and placement): dst[d0 + i] = src[s0 + i] over an f x h x w x ch box.
struct K6vCopyArgs {
    const __half* src;
    __half* dst;
    int sf, sh, sw;
    int df, dh, dw;
    int f, h, w, ch;
    int sf0, sh0, sw0;
    int df0, dh0, dw0;
};

extern "C" __global__ void __launch_bounds__(256) k6v_copy(const K6vCopyArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)a.f * a.h * a.w * a.ch) return;
    const int c = (int)(i % a.ch);
    long long r = i / a.ch;
    const int x = (int)(r % a.w);
    r /= a.w;
    const int y = (int)(r % a.h);
    const int f = (int)(r / a.h);
    const long long is = (((long long)(a.sf0 + f) * a.sh + a.sh0 + y) * a.sw + a.sw0 + x) * a.ch + c;
    const long long id = (((long long)(a.df0 + f) * a.dh + a.dh0 + y) * a.dw + a.dw0 + x) * a.ch + c;
    a.dst[id] = a.src[is];
}

// launch.zig mirrors these blocks; its comptime checks assert the same sizes.
static_assert(sizeof(K6vGnArgs) == 48 && sizeof(K6vGnApplyArgs) == 64 && sizeof(K6vSoftmaxArgs) == 48, "launch.zig");
static_assert(sizeof(K6vGnWelfordArgs) == 40, "launch.zig");
static_assert(sizeof(K6vTransposeArgs) == 40 && sizeof(K6vLatentInArgs) == 32 && sizeof(K6vRgb8InArgs) == 24, "launch.zig");
static_assert(sizeof(K6vPosteriorArgs) == 56 && sizeof(K6vBlendArgs) == 56 && sizeof(K6vCopyArgs) == 80, "launch.zig");
