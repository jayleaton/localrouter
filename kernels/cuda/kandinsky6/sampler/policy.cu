// The Kandinsky-6 Pro-distill PiFlow policy, elementwise (upstream kandinsky-6 eac06e39
// kandinsky/core/algo/piflow_math.py DXPolicy, policy_rollout_fm; kandinsky/core/algo/denoise_loop.py
// _build_video_input). Each kernel names the upstream tensor expressions it computes and keeps their rounding points:
// every fp32 step is its own __fmul_rn / __fadd_rn / __fsub_rn / __fdiv_rn (torch runs each as a separate tensor op,
// so none is contracted into an FMA, and torch's tensor divisions are IEEE divisions, never a reciprocal multiply);
// bf16 inputs widen exactly to fp32 as torch's type promotion does (bf16 tensor op fp32 tensor -> fp32). The per-step
// and per-substep scalars are the host's (src/engines/kandinsky6/sampler.zig), already the f32 values upstream's
// one-element schedule tensors hold.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace k6_piflow {

__device__ __forceinline__ float ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float ld(const float* p, long long i) { return p[i]; }

// One substep of one modality (policy_rollout_fm, piflow_math.py:178-187, with DXPolicy.pi inside it), element i of
// the n-element state. Upstream, per step:
//   x0[g] = x_src - sigma_src * pred[g]                       (_u_to_x_0, :59; once a step, f32 [B, n_grid, ...])
// and per substep, with the host's g0 = t0, w0 = t1 - t, w1 = t - t0 (_interpolate, :43-48) and sc = sigma.clamp(eps):
//   x0  = w0 * x0[g0] + w1 * x0[g0 + 1]                         (:48)
//   v   = (x_t - x0) / sc                                       (pi, :74)
//   x_t = x_t + v * (sigma_next - sigma)                        (:182; torch.where(active) keeps it: every substep is
//                                                                active at B = 1)
// x0[g] is recomputed here from x_src and the grids with the same two ops, so its bits are the tensor upstream holds.
// `first`: substep 0, where x_t is the step's input x_src (bf16 at step 0); later substeps read and write `out` (f32)
// in place. `pred` is grid-major [n_grid, n] (the DiT ABI's Outputs).
template <typename S, typename P>
__device__ __forceinline__ void substep(const S* __restrict__ x_src, const P* __restrict__ pred, float* out, long long n,
                                        int first, int g0, float sigma_src, float w0, float w1, float sc, float dsigma) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xs = ld(x_src, i);
    const float a0 = __fsub_rn(xs, __fmul_rn(sigma_src, ld(pred, (long long)g0 * n + i)));
    const float a1 = __fsub_rn(xs, __fmul_rn(sigma_src, ld(pred, (long long)(g0 + 1) * n + i)));
    const float x0 = __fadd_rn(__fmul_rn(w0, a0), __fmul_rn(w1, a1));
    const float xt = first ? xs : out[i];
    const float v = __fdiv_rn(__fsub_rn(xt, x0), sc);
    out[i] = __fadd_rn(xt, __fmul_rn(v, dsigma));
}

// _build_video_input, visual_cond_scheme "pretrain", batched (video.ndim == 5, B = 1; denoise_loop.py:30-31, 49-50,
// 60, 64): cat([video, cond, mask], -1) in the video's dtype, row r of [rows, 2 * ch + 1]. cond = zeros_like(video)
// with cond[:, 0] = first_frames.to(video.dtype) (rows r < rows_first, frame 0), mask zeros with mask[:, 0] = 1 there.
// T2V: no first frames (first_dtype -1), cond and mask zero. first_dtype: 0 bf16, 1 f32.
__device__ __forceinline__ float first_at(const void* first, int dt, long long j) {
    return dt == 0 ? __bfloat162float(((const __nv_bfloat16*)first)[j]) : ((const float*)first)[j];
}
__device__ __forceinline__ void put(__nv_bfloat16* o, long long i, float v) { o[i] = __float2bfloat16_rn(v); }
__device__ __forceinline__ void put(float* o, long long i, float v) { o[i] = v; }

template <typename T>
__device__ __forceinline__ void video_input(const T* __restrict__ x, const void* __restrict__ first, int first_dtype,
                                            T* __restrict__ out, long long rows, int ch, long long rows_first) {
    const long long w = 2LL * ch + 1;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * w) return;
    const long long r = i / w;
    const int c = (int)(i % w);
    if (c < ch) {
        out[i] = x[r * ch + c];
        return;
    }
    const bool ref = first_dtype >= 0 && r < rows_first;
    if (c < 2 * ch) {
        // bf16 state: first_frames.to(bf16) rounds to nearest even (a bf16 reference passes unchanged); f32 state: exact
        if (ref) put(out, i, first_at(first, first_dtype, r * ch + (c - ch)));
        else put(out, i, 0.0f);
        return;
    }
    put(out, i, ref ? 1.0f : 0.0f);
}

}  // namespace k6_piflow

#define K6_SUBSTEP(NAME, S, P)                                                                                        \
    extern "C" __global__ void NAME(const S* __restrict__ x_src, const P* __restrict__ pred, float* out, long long n,  \
                                    int first, int g0, float sigma_src, float w0, float w1, float sc, float dsigma) { \
        k6_piflow::substep<S, P>(x_src, pred, out, n, first, g0, sigma_src, w0, w1, sc, dsigma);                       \
    }

// grid ceil(n / 256), block 256 (sampler.zig Kernels.substep)
K6_SUBSTEP(k6_piflow_substep_bf16_bf16, __nv_bfloat16, __nv_bfloat16)
K6_SUBSTEP(k6_piflow_substep_bf16_f32, __nv_bfloat16, float)
K6_SUBSTEP(k6_piflow_substep_f32_bf16, float, __nv_bfloat16)
K6_SUBSTEP(k6_piflow_substep_f32_f32, float, float)

// grid ceil(rows * (2 * ch + 1) / 256), block 256 (sampler.zig Kernels.videoInput)
extern "C" __global__ void k6_piflow_video_input_bf16(const __nv_bfloat16* __restrict__ x, const void* __restrict__ first,
                                                      int first_dtype, __nv_bfloat16* __restrict__ out, long long rows,
                                                      int ch, long long rows_first) {
    k6_piflow::video_input<__nv_bfloat16>(x, first, first_dtype, out, rows, ch, rows_first);
}
extern "C" __global__ void k6_piflow_video_input_f32(const float* __restrict__ x, const void* __restrict__ first,
                                                     int first_dtype, float* __restrict__ out, long long rows, int ch,
                                                     long long rows_first) {
    k6_piflow::video_input<float>(x, first, first_dtype, out, rows, ch, rows_first);
}
