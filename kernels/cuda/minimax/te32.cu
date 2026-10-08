// Ours: MiniMax H3's text encoder (ComfyUI 0.37.0's Qwen3-VL 32B text path, 50 layers, NVFP4 AWQ checkpoint) on kernels
// LocalRouter owns, shared by the twin (torch extension) and the Zig engine (fatbin). extern "C", raw arguments,
// deterministic: every reduction in a fixed order, every rounding written out. Activations are fp32 throughout; the
// weights stay in the checkpoint's own tensors (NVFP4 codes, swizzled e4m3 block scales, fp32 tensor scale) and are
// dequantized on the fly with comfy-kitchen's arithmetic. Compile WITHOUT --use_fast_math and with the default
// -fmad=true (the explicit __f*_rn / __fmaf_rn below do not depend on it; expf, rsqrtf and the divisions do).
#include <cuda_bf16.h>
#include <stdint.h>

#define TE32_BLOCK 256
#define TE32_HD 128

typedef __nv_bfloat16 te32_bf16;

// ---------------------------------------------------------------------------------------------------- embedding
// ComfyUI's quantized Embedding (comfy_quant {"format": "int8_tensorwise"}, per-row fp32 scale [V, 1]):
// dequantize_int8_embedding = float(int8) * scale (one fp32 rounding), returned in the layer's orig_dtype (the text
// encoder's compute dtype, bf16: the norm weights' dtype), then cast to fp32 by `out_dtype`. bf16_round = 1 performs
// that bf16 rounding (round to nearest even); 0 keeps the fp32 product. scale null = 1.
// launch: block 256, grid (ceil(D / 256), L).
extern "C" __global__ void te32_embed_i8(const int8_t* table, const float* scale, const int* ids, float* out,
                                         long long D, int bf16_round) {
    const long long j = (long long)blockIdx.x * TE32_BLOCK + threadIdx.x;
    if (j >= D) return;
    const long long id = ids[blockIdx.y];
    float v = (float)table[id * D + j];
    if (scale) v = __fmul_rn(v, scale[id]);
    if (bf16_round) v = __bfloat162float(__float2bfloat16_rn(v));
    out[(long long)blockIdx.y * D + j] = v;
}

// The same gather from a bf16 table (an unquantized checkpoint): out = float(table[id]). launch as te32_embed_i8.
extern "C" __global__ void te32_embed_bf16(const te32_bf16* table, const int* ids, float* out, long long D) {
    const long long j = (long long)blockIdx.x * TE32_BLOCK + threadIdx.x;
    if (j >= D) return;
    out[(long long)blockIdx.y * D + j] = __bfloat162float(table[(long long)ids[blockIdx.y] * D + j]);
}

// ---------------------------------------------------------------------------------------------- elementwise
// out = a + b, fp32, one rounding (the residual adds; out may alias a or b). launch: block 256, grid ceil(n / 256).
extern "C" __global__ void te32_add(const float* a, const float* b, float* out, long long n) {
    const long long i = (long long)blockIdx.x * TE32_BLOCK + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(a[i], b[i]);
}

// down_proj's input: torch's fp32 silu then the fp32 multiply, each rounded: out = (g / (1 + expf(-g))) * u.
// g, u, out [n] (out may alias g or u). launch: block 256, grid ceil(n / 256).
extern "C" __global__ void te32_silu_mul(const float* g, const float* u, float* out, long long n) {
    const long long i = (long long)blockIdx.x * TE32_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float x = g[i];
    const float s = __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
    out[i] = __fmul_rn(s, u[i]);
}

// ---------------------------------------------------------------------------------------------------- RMSNorm
// F.rms_norm (fp32) with the checkpoint's norm weight cast to fp32: y = (x * rsqrt(mean(x^2) + eps)) * w. The sum of
// squares: thread-strided partials (each ascending), then a shared-memory tree; mean = sum / D (fdiv); rsqrtf; two
// roundings per element. Rows of D (5120, or 128 per head for q_norm / k_norm). y may alias x.
// launch: block 256, grid rows.
extern "C" __global__ void te32_rms_norm(const float* x, const float* w, float* y, long long D, float eps) {
    __shared__ float s[TE32_BLOCK];
    const float* xr = x + (long long)blockIdx.x * D;
    float* yr = y + (long long)blockIdx.x * D;
    float acc = 0.0f;
    for (long long j = threadIdx.x; j < D; j += TE32_BLOCK) {
        const float v = xr[j];
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = TE32_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = __fadd_rn(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(s[0], (float)D), eps));
    for (long long j = threadIdx.x; j < D; j += TE32_BLOCK) yr[j] = __fmul_rn(__fmul_rn(xr[j], r), w[j]);
}

// ------------------------------------------------------------------------------------------------------- RoPE
// comfy_kitchen.apply_rope_split_half on fp32 q or k [L, heads, 128], in place, 1-D positions (row index = position).
// ComfyUI's rotation matrix per pair i (0..63) is [[cos_i, -sin_i], [sin_i, cos_i]] (cos / sin tables [L, 64] fp32:
// the first halves of cat(freqs, freqs).cos() / .sin(); the second halves are equal), and the kernels compute
//   y[i]      = f00 * x[i] + f01 * x[i + 64],   f00 = cos, f01 = -sin
//   y[i + 64] = f10 * x[i] + f11 * x[i + 64],   f10 = sin, f11 = cos
// in fp32. How the two products and the add contract to FMA is the compiler's choice in every kitchen backend, so
// `mode` selects it (the twin pins the one that matches ComfyUI's run on the pod, see TE32-PORT.md):
//   0: fmaf(f_a, x_a, f_b * x_b)  the first product fused, the second rounded (nvcc / Triton default contraction)
//   1: fmaf(f_b, x_b, f_a * x_a)  the second product fused (the eager path: out = x0 * f00; out.addcmul_(x1, f01))
//   2: (f_a * x_a) + (f_b * x_b)  no FMA
// launch: block 64 (one pair a thread), grid (L, heads).
extern "C" __global__ void te32_rope_split_half(float* x, const float* cs, const float* sn, long long heads, int mode) {
    const int i = threadIdx.x;
    const long long pos = blockIdx.x;
    float* row = x + (pos * heads + blockIdx.y) * TE32_HD;
    const float x0 = row[i], x1 = row[i + 64];
    const float c = cs[pos * 64 + i], s = sn[pos * 64 + i];
    const float f00 = c, f01 = -s, f10 = s, f11 = c;
    float y0, y1;
    if (mode == 0) {
        y0 = __fmaf_rn(f00, x0, __fmul_rn(f01, x1));
        y1 = __fmaf_rn(f10, x0, __fmul_rn(f11, x1));
    } else if (mode == 1) {
        y0 = __fmaf_rn(f01, x1, __fmul_rn(f00, x0));
        y1 = __fmaf_rn(f11, x1, __fmul_rn(f10, x0));
    } else {
        y0 = __fadd_rn(__fmul_rn(f00, x0), __fmul_rn(f01, x1));
        y1 = __fadd_rn(__fmul_rn(f10, x0), __fmul_rn(f11, x1));
    }
    row[i] = y0;
    row[i + 64] = y1;
}

// ------------------------------------------------------------------------------------------------- attention
// Causal GQA attention, fp32, head dim 128: q [L, H, 128], k and v [L, KVH, 128] (head h reads KV head h / (H / KVH),
// repeat_interleave order), out [L, H * 128] (head h at columns h * 128, the o_proj's input). One block per (query
// row, head), 128 threads:
//   logit[j] = (fmaf chain over d ascending of q[d] * k[j][d]) * scale, j = 0 .. i  (keys j > i are ComfyUI's masked
//   logits: logit + finfo(fp32).min / 4 is -8.5e37, expf(-8.5e37 - max) is exactly 0, so they are simply absent);
//   max over the row (exact, any order); e[j] = expf(logit[j] - max); sum = per-thread partials (thread t: j = t,
//   t + 128, ... ascending) then a shared-memory tree; out[d] = (fmaf chain over j ascending of e[j] * v[j][d]) / sum
//   (thread d), the division last. This replaces SDPA's EFFICIENT / MATH fp32 path with a design whose bits do not
//   depend on L's tiling.
// dynamic shared memory: (L + 128 + 128) floats (logits / weights, the q row, the reduction scratch); above 48 KB the
// launcher must raise cudaFuncAttributeMaxDynamicSharedMemorySize. launch: block 128, grid (L, H).
extern "C" __global__ void te32_attention(const float* q, const float* k, const float* v, float* out, int L, int H,
                                          int KVH, float scale) {
    extern __shared__ float sm[];
    float* s = sm;
    float* qs = sm + L;
    float* red = qs + TE32_HD;
    const int t = threadIdx.x;
    const long long i = blockIdx.x;
    const int h = blockIdx.y, kvh = h / (H / KVH);
    qs[t] = q[(i * H + h) * TE32_HD + t];
    __syncthreads();
    const long long n = i + 1;
    for (long long j = t; j < n; j += TE32_HD) {
        const float* kr = k + (j * KVH + kvh) * TE32_HD;
        float acc = 0.0f;
        for (int d = 0; d < TE32_HD; ++d) acc = __fmaf_rn(qs[d], kr[d], acc);
        s[j] = __fmul_rn(acc, scale);
    }
    __syncthreads();
    float mx = __int_as_float((int)0xFF800000u);  // -inf
    for (long long j = t; j < n; j += TE32_HD) mx = fmaxf(mx, s[j]);
    red[t] = mx;
    __syncthreads();
    for (int st = TE32_HD / 2; st > 0; st >>= 1) {
        if (t < st) red[t] = fmaxf(red[t], red[t + st]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (long long j = t; j < n; j += TE32_HD) {
        const float e = expf(__fsub_rn(s[j], mx));
        s[j] = e;
        sum = __fadd_rn(sum, e);
    }
    red[t] = sum;
    __syncthreads();
    for (int st = TE32_HD / 2; st > 0; st >>= 1) {
        if (t < st) red[t] = __fadd_rn(red[t], red[t + st]);
        __syncthreads();
    }
    const float denom = red[0];
    float acc = 0.0f;
    for (long long j = 0; j < n; ++j) acc = __fmaf_rn(s[j], v[(j * KVH + kvh) * TE32_HD + t], acc);
    out[(i * H + h) * TE32_HD + t] = __fdiv_rn(acc, denom);
}

// ----------------------------------------------------------------------------------------------- NVFP4 linear
// comfy-kitchen's NVFP4 formats, exactly as ck.dequantize_nvfp4 reads them:
//  * codes uint8 [N, K / 2]: byte b of row n holds elements 2b (HIGH nibble) and 2b + 1 (low nibble) of the row;
//    a nibble is sign (bit 3) and an e2m1 magnitude index (bits 0-2) -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
//  * block scales e4m3 (fn), one per 16 elements of a row, stored in the cuBLAS 128 x 4 swizzled layout, for the
//    (row n, block c) pair at byte offset
//      rb = n / 128, rem = n % 128, d4 = rem / 32, d3 = rem % 32, cbg = c / 4, d5 = c % 4, cbc = ceil((K/16) / 4)
//      off = ((rb * cbc + cbg) * 32 + d3) * 16 + d4 * 4 + d5
//    (comfy_kitchen scale_factor_swizzled_offset; the inverse of float_utils.from_blocked).
//  * tscale: the fp32 tensor scale (weight_scale_2).
//  w = float(e2m1) * (float(e4m3) * tscale), both products one fp32 rounding, sign applied last.
__device__ __forceinline__ float te32_e4m3(uint8_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0u) v = __fmul_rn((float)m, 0.001953125f);                  // m * 2^-9, exact
    else if (e == 15u && m == 7u) v = __int_as_float(0x7FC00000);        // NaN (e4m3fn has no inf)
    else v = __int_as_float((int)(((e + 120u) << 23) | (m << 20)));      // (1 + m / 8) * 2^(e - 7), exact
    return (b & 0x80u) ? -v : v;
}

__device__ __forceinline__ float te32_e2m1_mag(uint32_t c) {
    const uint32_t e = c >> 1, m = c & 1u;
    if (e == 0u) return m ? 0.5f : 0.0f;
    return __int_as_float((int)(((e + 126u) << 23) | (m << 22)));        // (1 + m / 2) * 2^(e - 1)
}

__device__ __forceinline__ long long te32_swz(long long n, long long c, long long cbc) {
    const long long rb = n / 128, rem = n % 128, d4 = rem / 32, d3 = rem % 32, cbg = c / 4, d5 = c % 4;
    return ((rb * cbc + cbg) * 32 + d3) * 16 + d4 * 4 + d5;
}

// y [M, N] fp32 = (x [M, K] * pqs[K]) . W^T (+ bias[N]) with W [N, K] dequantized as above, on the fly. pqs (the AWQ
// pre_quant_scale as fp32, exactly its bf16 values) null = none; the activation is x * pqs, one fp32 rounding, as
// ComfyUI's `input * pre_quant_scale`. Each output is an fmaf chain over k ascending from 0 (then + bias, one
// rounding), so its bits depend only on its x row and its weight row, not on the tiling or on M. Tiles of 128 (m) x
// 64 (n), K tiles of 16 = one block scale per weight row; a thread owns 8 x 4 outputs. The grid's x runs over M
// tiles so the blocks sharing a weight tile run together (the weight tile is read from DRAM once per wave).
// K % 16 == 0 (checked by the launcher). launch: block 256, grid (ceil(M / 128), ceil(N / 64)).
extern "C" __global__ void te32_linear_nvfp4(const float* x, const float* pqs, const uint8_t* codes,
                                             const uint8_t* bscale, float tscale, const float* bias, float* y,
                                             long long M, long long N, long long K) {
    __shared__ float sa[16][129];
    __shared__ float sb[16][65];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const long long m0 = (long long)blockIdx.x * 128, n0 = (long long)blockIdx.y * 64;
    const long long cbc = (K / 16 + 3) / 4;
    float acc[8][4] = {};
    for (long long k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 128 * 16; i += TE32_BLOCK) {
            const int r = i / 16, kk = i % 16;
            float a = 0.0f;
            if (m0 + r < M) {
                a = x[(m0 + r) * K + k0 + kk];
                if (pqs) a = __fmul_rn(a, pqs[k0 + kk]);
            }
            sa[kk][r] = a;
        }
        {
            const int r = threadIdx.x / 4, q = threadIdx.x % 4;  // row r of the tile, elements 4q .. 4q + 3
            float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (n0 + r < N) {
                const float ds = __fmul_rn(te32_e4m3(bscale[te32_swz(n0 + r, k0 / 16, cbc)]), tscale);
                const uint8_t* cp = codes + (n0 + r) * (K / 2) + k0 / 2 + q * 2;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const uint32_t byte = cp[b], hi = byte >> 4, lo = byte & 15u;
                    const float wh = __fmul_rn(te32_e2m1_mag(hi & 7u), ds), wl = __fmul_rn(te32_e2m1_mag(lo & 7u), ds);
                    w[2 * b] = (hi & 8u) ? -wh : wh;
                    w[2 * b + 1] = (lo & 8u) ? -wl : wl;
                }
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) sb[q * 4 + e][r] = w[e];
        }
        __syncthreads();
        for (int kk = 0; kk < 16; ++kk) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = __fmaf_rn(sa[kk][ty * 8 + i], sb[kk][tx * 4 + j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long m = m0 + ty * 8 + i, n = n0 + tx * 4 + j;
            if (m < M && n < N) y[m * N + n] = bias ? __fadd_rn(acc[i][j], bias[n]) : acc[i][j];
        }
}
