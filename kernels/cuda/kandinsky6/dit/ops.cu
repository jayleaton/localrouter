// The Kandinsky-6 DiT's elementwise, normalization and layout kernels (upstream kandinsky-6 eac06e39
// kandinsky/core/{tensors.py,components/dit.py}). Each kernel names the upstream expression it computes and keeps its
// rounding points: fp32 steps use __fmul_rn / __fadd_rn (never contracted into an FMA) where torch runs separate
// tensor ops, and a value is rounded to bf16 exactly where the upstream tensor is bf16. The norms are torch's own
// fused kernels' reductions and expressions (below), so a row's bits depend only on the row, as torch's do.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace k6_ops {

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// ---- torch 2.14.1 aten/src/ATen/native/cuda/layer_norm_kernel.cu (tag v2.14.1, 5c48869085; BSD-3-Clause, see
// NOTICE): the statistics of vectorized_layer_norm_kernel, which LayerNorm and _fused_rms_norm take for these rows
// (fp32 / bf16, N % 4 == 0, aligned). The arithmetic is torch's source expressions unchanged and in plain operators,
// so nvcc contracts them as it contracts torch's own build under the same flags (fmad on, IEEE division); the launch
// shape is torch's: block (32, num_threads() / 32 = 4), one block a row, vec_size 4, thread thrx reading vectors
// thrx, thrx + 128, ...
constexpr int vec_size = 4;
constexpr int ln_threads_y = 4;

struct WelfordDataLN {
    float mean;
    float sigma2;
    float count;
};

template <bool rms_norm>
__device__ __forceinline__ WelfordDataLN cuWelfordOnlineSum(const float val, const WelfordDataLN& curr_sum) {
    if constexpr (!rms_norm) {
        float delta = val - curr_sum.mean;
        float new_count = curr_sum.count + 1.f;
        auto fn_rcp_mul = [](auto a, auto b) { return a * (1.0f / b); };
        float new_mean = curr_sum.mean + fn_rcp_mul(delta, new_count);
        return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
    } else {
        return {0.f, curr_sum.sigma2 + val * val, 0};
    }
}

template <bool rms_norm>
__device__ __forceinline__ WelfordDataLN cuWelfordCombine(const WelfordDataLN dataB, const WelfordDataLN dataA) {
    if constexpr (!rms_norm) {
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
    } else {
        return {0.f, dataB.sigma2 + dataA.sigma2, 0};
    }
}

// compute_stats: {mean, biased variance} (rms_norm: {0, mean of squares}) of a row of N, element e read as load(e)
// (torch loads aligned_vector<T, 4>: the same values, per-thread order and thread mapping). buf: 6 floats of shared
// memory (torch's nshared = threads.y * 3 / 2 floats). Every thread of the block returns the same value.
template <bool rms_norm, typename Load>
__device__ __forceinline__ WelfordDataLN compute_stats(Load load, const int N, float* buf) {
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = N / vec_size;
    WelfordDataLN wd{0.f, 0.f, 0.f};
    for (int i = thrx; i < n_vec_to_read; i += numx) {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) wd = cuWelfordOnlineSum<rms_norm>(load(i * vec_size + ii), wd);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordDataLN wdB{__shfl_down_sync(0xffffffffu, wd.mean, offset), __shfl_down_sync(0xffffffffu, wd.sigma2, offset),
                          __shfl_down_sync(0xffffffffu, wd.count, offset)};
        wd = cuWelfordCombine<rms_norm>(wd, wdB);
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
            wd = cuWelfordCombine<rms_norm>(wd, wdB);
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

}  // namespace k6_ops

using namespace k6_ops;

// y[n] = bf16(x[n]): `x.to(dtype=bf16)` of the encoders' fp32 outputs and the fp32 latents before the input linears.
// launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6_cast_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tobf(x[i]);
}

// y = bf16(float(a) + float(b)) elementwise (bf16 tensor add: `time_embeddings(t) + pooled_proj`). y may alias a or b.
// launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tobf(__fadd_rn(bf(a[i]), bf(b[i])));
}

// nn.LayerNorm over the last `dim` of bf16 rows, output bf16: TextEmbeddings.norm (affine: w, b bf16) and
// OutLayerAudio's second `self.norm(x)` (w = b = null): torch's vectorized_layer_norm_kernel with T = bf16, its output
// expressions verbatim (`gamma * (rstd_val * (x - mean)) + beta` in fp32, then bf16). dim % 4 == 0.
// launch: block (32, 4), grid rows. y may alias x (a block reads its row's statistics before writing).
extern "C" __global__ void __launch_bounds__(128) k6_layer_norm(const __nv_bfloat16* x, const __nv_bfloat16* __restrict__ w,
                                                                 const __nv_bfloat16* __restrict__ b, __nv_bfloat16* y, int dim,
                                                                 long long ldx, long long ldy, float eps) {
    __shared__ float buf[ln_threads_y * 3 / 2];
    const __nv_bfloat16* xr = x + (long long)blockIdx.x * ldx;
    __nv_bfloat16* yr = y + (long long)blockIdx.x * ldy;
    const WelfordDataLN wd = compute_stats<false>([&](int e) { return bf(xr[e]); }, dim, buf);
    const float rstd_val = rsqrtf(wd.sigma2 + eps);
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int e = thrx * vec_size; e < dim; e += blockDim.x * blockDim.y * vec_size) {
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
            const float v = bf(xr[e + ii]);
            float out;
            if (w != nullptr && b != nullptr)
                out = bf(w[e + ii]) * (rstd_val * (v - wd.mean)) + bf(b[e + ii]);
            else if (w != nullptr)
                out = bf(w[e + ii]) * (rstd_val * (v - wd.mean));
            else if (b != nullptr)
                out = (rstd_val * (v - wd.mean)) + bf(b[e + ii]);
            else
                out = rstd_val * (v - wd.mean);
            yr[e + ii] = tobf(out);
        }
    }
}

// apply_scale_shift_norm(LayerNorm(dim, elementwise_affine=False), x, scale, shift):
// y = bf16((LN(float(x)) * (float(scale) + 1)) + float(shift)). The norm is torch's vectorized_layer_norm_kernel with
// T = fp32 (x.float(): the bf16 values widened exactly), its fp32 output `rstd_val * (x - mean)`; the scale / shift
// steps are separate torch ops, so each rounds on its own (explicit _rn, never contracted). scale, shift: one bf16 row
// of `dim` (the modulation's chunk). dim % 4 == 0.
// launch: block (32, 4), grid rows. y may alias x.
extern "C" __global__ void __launch_bounds__(128) k6_norm_mod(const __nv_bfloat16* x, const __nv_bfloat16* __restrict__ scale,
                                                               const __nv_bfloat16* __restrict__ shift, __nv_bfloat16* y,
                                                               int dim, long long ldx, long long ldy, float eps) {
    __shared__ float buf[ln_threads_y * 3 / 2];
    const __nv_bfloat16* xr = x + (long long)blockIdx.x * ldx;
    __nv_bfloat16* yr = y + (long long)blockIdx.x * ldy;
    const WelfordDataLN wd = compute_stats<false>([&](int e) { return bf(xr[e]); }, dim, buf);
    const float rstd_val = rsqrtf(wd.sigma2 + eps);
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int e = thrx; e < dim; e += blockDim.x * blockDim.y) {
        const float n = rstd_val * (bf(xr[e]) - wd.mean);
        yr[e] = tobf(__fadd_rn(__fmul_rn(n, __fadd_rn(bf(scale[e]), 1.0f)), bf(shift[e])));
    }
}

// apply_gate_sum(x, out, gate): x = bf16(float(x) + float(gate) * float(out)) in place; gate one bf16 row of `dim`.
// launch: block 256, grid (rows, ceil(dim / 256)) (rows on x: no 65,535 cap).
extern "C" __global__ void k6_gate_sum(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ out,
                                       const __nv_bfloat16* __restrict__ gate, int dim, long long ldx, long long ldo) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    __nv_bfloat16* xp = x + (long long)blockIdx.x * ldx + i;
    const float o = bf(out[(long long)blockIdx.x * ldo + i]);
    *xp = tobf(__fadd_rn(bf(*xp), __fmul_rn(bf(gate[i]), o)));
}

// The attention modules' `q = query_norm(q)` then `apply_rotary(q, rope).type_as(q)`, in place on a [rows, heads * 128]
// bf16 projection (row stride ld). One warp a (row, head); lane j holds elements 4j .. 4j + 3 (rotation pairs 2j, 2j + 1).
//   nn.RMSNorm(128) with eps None and a bf16 weight on bf16 input: torch 2.14's _fused_rms_norm_cuda (weight dtype ==
//   input dtype), i.e. vectorized_layer_norm_kernel<rms_norm = true> with one block (32, 4) a 128-wide row: only warp 0
//   holds data (lane j: vector j, elements 4j .. 4j + 3, `sigma2 + val * val` in order), a shuffle-down tree to lane 0,
//   the other warps add exact zeros, ms = sigma2 / 128, rstd = rsqrtf(ms + eps) with eps = FLT_EPSILON (the accumulate
//   type's epsilon, not bf16's), y = bf16(w * (rstd * x)). One warp here reproduces that order exactly, with torch's
//   plain-operator expressions (cuWelfordOnlineSum / Combine above). The launcher passes eps.
//   apply_rotary on the bf16 y: x_ = y.reshape(..., 64, 1, 2).float(); (rope * x_).sum(-1) with rope[..., j] =
//   [[cos, -sin], [sin, cos]]: r0 = cos * y0 + (-sin) * y1, r1 = sin * y0 + cos * y1, each product its own fp32
//   rounding, one add, then bf16. rope: the module's own fp32 table [rows, 64, 2, 2] (RoPE1D / RoPE3D output, the
//   singleton head axis dropped; row stride ldr floats, 256 when contiguous), read as stored; null skips the rotation
//   (the text cross-attention: rope_q = rope_kv = None).
// launch: block 256 (8 warps), grid ceil(rows * heads / 8).
extern "C" __global__ void __launch_bounds__(256) k6_qk_norm_rope(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ w,
                                                                   const float* __restrict__ rope, int rows, int heads,
                                                                   long long ld, long long ldr, float eps) {
    const int lane = threadIdx.x % 32;
    const long long item = (long long)blockIdx.x * 8 + threadIdx.x / 32;
    if (item >= (long long)rows * heads) return;
    const long long row = item / heads;
    const int head = (int)(item % heads);
    __nv_bfloat16* p = x + row * ld + head * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = bf(p[e]);
    WelfordDataLN wd{0.f, 0.f, 0.f};
#pragma unroll
    for (int e = 0; e < 4; ++e) wd = cuWelfordOnlineSum<true>(v[e], wd);
    for (int offset = 16; offset > 0; offset >>= 1) {
        const WelfordDataLN wdB{0.f, __shfl_down_sync(0xffffffffu, wd.sigma2, offset), 0.f};
        wd = cuWelfordCombine<true>(wd, wdB);
    }
    const float ms = __shfl_sync(0xffffffffu, wd.sigma2, 0) / float(128);  // lane 0's sum, as torch's buffer passes it on
    const float rstd = rsqrtf(ms + eps);
    float y[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) y[e] = bf(tobf(__fmul_rn(bf(w[4 * lane + e]), __fmul_rn(rstd, v[e]))));
    if (rope) {
        const float* r = rope + row * ldr + 8 * lane;  // pairs 2 * lane, 2 * lane + 1: a 2x2 matrix each
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float* m = r + 4 * j;
            const float y0 = y[2 * j], y1 = y[2 * j + 1];
            y[2 * j] = __fadd_rn(__fmul_rn(m[0], y0), __fmul_rn(m[1], y1));
            y[2 * j + 1] = __fadd_rn(__fmul_rn(m[2], y0), __fmul_rn(m[3], y1));
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) p[e] = tobf(y[e]);
}

// The time embedding's sinusoid: args = outer(float(time), freqs); embed = cat([cos(args), sin(args)], -1), fp32.
// time fp32 [M], freqs fp32 [half] (TimeEmbeddings.freqs as the module holds it), out fp32 [M, 2 * half].
// launch: block 256, grid (M, ceil(half / 256)).
extern "C" __global__ void k6_time_sincos(const float* __restrict__ time, const float* __restrict__ freqs,
                                          float* __restrict__ out, int half) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= half) return;
    const float a = __fmul_rn(time[blockIdx.x], freqs[i]);
    float* o = out + (long long)blockIdx.x * 2 * half;
    o[i] = cosf(a);
    o[half + i] = sinf(a);
}

// VisualEmbeddings' patchify for patch (1, 2, 2): x [T, H, W, C] (channels last, fp32 when in_f32 else bf16) ->
// rows bf16 [T * (H/2) * (W/2), 4 * C], row (t, h', w'), feature ((ph * 2 + pw) * C + c), the value rounded to bf16 as
// `x.to(dtype=bf16)` rounds it. launch: block 256, grid ceil(T * H * W * C / 256).
extern "C" __global__ void k6_patchify(const void* __restrict__ x, __nv_bfloat16* __restrict__ rows, int T, int H, int W,
                                       int C, int in_f32) {
    const long long n = (long long)T * H * W * C;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int)(i % C);
    long long r = i / C;
    const int xw = (int)(r % W);
    r /= W;
    const int yh = (int)(r % H);
    const int t = (int)(r / H);
    const __nv_bfloat16 v = in_f32 ? tobf(static_cast<const float*>(x)[i]) : static_cast<const __nv_bfloat16*>(x)[i];
    const long long row = ((long long)t * (H / 2) + yh / 2) * (W / 2) + xw / 2;
    rows[row * 4 * C + ((yh % 2) * 2 + xw % 2) * C + c] = v;
}

// OutLayer's unpatchify for patch (1, 2, 2) joined with the PiFlow head's grid split: rows [T * H' * W', 4 * C] (bf16,
// feature ((c * 2 + ph) * 2 + pw)) -> out bf16 [G, T, 2H', 2W', C / G], element (g, t, y, x, cc) from channel
// c = g * (C / G) + cc, i.e. the contiguous form of `x.view(.., n_grid, C / G).movedim(-2, 1)` (G = 1: [T, H, W, C]).
// launch: block 256, grid ceil(T * 4 * H' * W' * C / 256).
extern "C" __global__ void k6_unpatchify(const __nv_bfloat16* __restrict__ rows, __nv_bfloat16* __restrict__ out, int T, int Hp,
                                         int Wp, int C, int G) {
    const long long n = (long long)T * Hp * Wp * 4 * C;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int cg = C / G, H = 2 * Hp, W = 2 * Wp;
    const int cc = (int)(i % cg);
    long long r = i / cg;
    const int xw = (int)(r % W);
    r /= W;
    const int yh = (int)(r % H);
    r /= H;
    const int t = (int)(r % T);
    const int g = (int)(r / T);
    const int c = g * cg + cc;
    const long long row = ((long long)t * Hp + yh / 2) * Wp + xw / 2;
    out[i] = rows[row * 4 * C + (c * 2 + yh % 2) * 2 + xw % 2];
}

// The PiFlow head's grid split of a [rows, G * Cg] output: out [G, rows, Cg] (`view(.., G, Cg).movedim(-2, 1)` made
// contiguous; the audio head's [218, 10 * 40] -> [10, 218, 40]). launch: block 256, grid ceil(rows * G * Cg / 256).
extern "C" __global__ void k6_grid_major(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int rows, int G,
                                         int Cg) {
    const long long n = (long long)rows * G * Cg;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int cc = (int)(i % Cg);
    const long long r = (i / Cg) % rows;
    const int g = (int)(i / ((long long)Cg * rows));
    out[i] = x[r * G * Cg + (long long)g * Cg + cc];
}
