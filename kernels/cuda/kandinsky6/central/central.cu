// Kandinsky-6 Pro-distill's central kernels around the DiT (upstream kandinskylab/kandinsky-6 eac06e39): the seeded
// starting noise, the PiFlow (DX policy) rollout of one outer step, the DiT's visual input and the RoPE cos / sin
// tables. extern "C", raw arguments, one element a thread, no reductions. Launch geometry: src/engines/kandinsky6/
// central.zig; host twins: noise.zig (Philox, geometry, offsets) and piflow.zig (schedule, scalar tables, rollout).
//
// Build with torch's flags and -O3, WITHOUT --fmad=false: k6_randn_bf16 and k6_rope_cossin compile curand's
// Box-Muller and libdevice logf / cosf / sinf as torch 2.14.1 compiles them (nvcc's default contraction; curand's
// `y * CURAND_2POW32_INV_2PI + CURAND_2POW32_INV_2PI / 2` is an FMA there). The rollout and the input builder use
// only explicit _rn intrinsics and conversions, which contraction never touches.
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

#define K6C_BLOCK 256
#define K6C_GRID 10        // PiFlow grid points (shapes.n_grid)
#define K6C_SUBSTEPS 32    // substeps a rollout launch carries (piflow.Table.capacity)

// ---------------------------------------------------------------------------------------------------------------
// torch.randn(..., device="cuda", dtype=bfloat16, generator=g) = normal_(0, 1) (TensorFactories.cpp:1346-1359,
// Distributions.cpp:274, DistributionTemplates.h:211-224), the CUDA normal_kernel (cuda/DistributionTemplates.h:460-472,
// :443-455) through distribution_nullary_kernel (:112-169) and its grid-stride kernel (:64-88), all PyTorch v2.14.1.
// unroll 4 (sizeof(float4) / sizeof(float)), curand_init(seed, thread, philox offset), one curand_normal4 a
// grid-stride iteration whose lanes go to linear_index + ii * stride, then `rand * std + mean` in float
// (TransformationHelper.h:96-98, contracted to an FMA in torch's build: exact for std 1, mean 0) and BFloat16(float)
// = __float2bfloat16 (RNE) on sm_80+ (torch/headeronly/util/BFloat16.h:125-139).
// launch: block 256, grid = noise.geometry(numel, device).grid (calc_execution_policy, :49-61); offset = the
// generator's philox offset before the draw (0 for a fresh generator).
extern "C" __global__ void __launch_bounds__(K6C_BLOCK, 4)
    k6_randn_bf16(bf16* out, long long numel, unsigned long long seed, unsigned long long offset, float mean, float stdv) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const long long stride = (long long)blockDim.x * gridDim.x;
    const long long rounded = ((numel - 1) / (stride * 4) + 1) * stride * 4;
    for (long long li0 = idx; li0 < rounded; li0 += stride * 4) {
        const float4 r = curand_normal4(&state);
        const float lanes[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            const long long li = li0 + stride * ii;
            if (li < numel) out[li] = __float2bfloat16_rn(__fmaf_rn(lanes[ii], stdv, mean));
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// One outer PiFlow step of one modality (core/algo/piflow_math.py:8-74 DXPolicy, :155-189 policy_rollout_fm,
// piflow_sampler.py:143-173), all substeps fused: every torch op there is one rounded fp32 op on the same values, so
// one thread an element applying them in torch's order gives torch's bits. The per-batch scalars (B = 1) come from
// the host (piflow.zig, the same op order in f32): `sigma` the step's sigma_t_src, and per substep j the
// interpolation index t0, its weights wa = float(t1) - t, wb = t - float(t0), the divisor max(sigma_j, eps) and
// ds = sigma_{j+1} - sigma_j.
//   x0[k] = float(x_src) - sigma * float(pred[k])                    (_u_to_x_0, :59)
//   per substep: x0 = wa * x0[t0] + wb * x0[t0 + 1]                  (_interpolate, :48)
//                v  = (x - x0) / div                                 (pi, :74)
//                x  = x + v * ds                                     (rollout, :182; torch.where keeps x_t_minus)
// x starts at `cur` (the step's state for the first chunk: x_src itself), so a step longer than 32 substeps runs
// as chunks that continue from `out`. src / cur: bf16 (is_f32 0) or fp32; pred bf16 grid-major [10][n] (grid stride
// n); out fp32, may alias cur (each thread reads its element before writing it), never src unless src == cur.
// launch: block 256, grid ceil(n / 256).
struct K6PiflowTable {
    float sigma;
    int count;
    int t0[K6C_SUBSTEPS];
    float wa[K6C_SUBSTEPS];
    float wb[K6C_SUBSTEPS];
    float div[K6C_SUBSTEPS];
    float ds[K6C_SUBSTEPS];
};

__device__ __forceinline__ float k6c_load(const void* p, int is_f32, long long i) {
    return is_f32 ? ((const float*)p)[i] : __bfloat162float(((const bf16*)p)[i]);
}

extern "C" __global__ void k6_piflow_rollout(const void* src, int src_f32, const void* cur, int cur_f32, const bf16* pred,
                                             float* out, long long n, K6PiflowTable tab) {
    const long long i = (long long)blockIdx.x * K6C_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float xs = k6c_load(src, src_f32, i);
    float x0[K6C_GRID];
#pragma unroll
    for (int k = 0; k < K6C_GRID; k++) x0[k] = __fsub_rn(xs, __fmul_rn(tab.sigma, __bfloat162float(pred[(long long)k * n + i])));
    float x = k6c_load(cur, cur_f32, i);
    for (int j = 0; j < tab.count; j++) {
        const int t0 = tab.t0[j];
        float a = x0[0];
        float b = x0[1];
#pragma unroll
        for (int k = 1; k < K6C_GRID - 1; k++) {
            if (t0 == k) {
                a = x0[k];
                b = x0[k + 1];
            }
        }
        const float xi = __fadd_rn(__fmul_rn(tab.wa[j], a), __fmul_rn(tab.wb[j], b));
        const float v = __fdiv_rn(__fsub_rn(x, xi), tab.div[j]);
        x = __fadd_rn(x, __fmul_rn(v, tab.ds[j]));
    }
    out[i] = x;
}

// ---------------------------------------------------------------------------------------------------------------
// The DiT's visual input (core/algo/denoise_loop.py:13-64, `pretrain`, batched [1, T, H, W, C] state):
// cat([video, cond, mask], -1) in the state's dtype, cond = zeros_like(video) except frame 0 = first_frames.to(dtype)
// (:30, :34, :50), mask = 1 on frame 0 (:31, :60) when a reference is given, else cond = mask = 0 (T2V).
// state [rows, 16] (bf16 or fp32, is_f32), ref [frame_rows, 16] (bf16 or fp32, ref_f32) or null, out [rows, 33] in
// the state's dtype. fp32 -> bf16 is RNE as torch's copy; bf16 -> fp32 exact. launch: block 256, grid ceil(rows*33/256).
extern "C" __global__ void k6_video_input(const void* state, int is_f32, const void* ref, int ref_f32, void* out, long long rows,
                                          long long frame_rows) {
    const long long e = (long long)blockIdx.x * K6C_BLOCK + threadIdx.x;
    if (e >= rows * 33) return;
    const long long r = e / 33;
    const int c = (int)(e - r * 33);
    const bool on_ref = ref != nullptr && r < frame_rows;
    if (is_f32) {
        float v = 0.0f;
        if (c < 16) v = ((const float*)state)[r * 16 + c];
        else if (c < 32) v = on_ref ? k6c_load(ref, ref_f32, r * 16 + (c - 16)) : 0.0f;
        else v = on_ref ? 1.0f : 0.0f;
        ((float*)out)[e] = v;
    } else {
        bf16 v = __float2bfloat16_rn(0.0f);
        if (c < 16) v = ((const bf16*)state)[r * 16 + c];
        else if (c < 32) {
            if (on_ref) v = ref_f32 ? __float2bfloat16_rn(((const float*)ref)[r * 16 + (c - 16)]) : ((const bf16*)ref)[r * 16 + (c - 16)];
        } else if (on_ref) v = __float2bfloat16_rn(1.0f);
        ((bf16*)out)[e] = v;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// RoPE tables (core/components/rope.py:30, :74-75): torch.stack([cos(args), -sin(args), sin(args), cos(args)], -1) on
// CUDA fp32 = separate cos / sin / neg kernels, `::cos(float)` / `::sin(float)` (libdevice cosf / sinf, no fast math;
// UnaryGeometricCosKernel.cu / UnaryGeometricSinKernel.cu, PyTorch v2.14.1), negation exact. The empty asm hides that
// both calls take the same value, so the compiler cannot merge them into one sincosf as torch's separate kernels
// never are. args [rows, 64] fp32, out [rows, 64, 4] fp32. launch: block 256, grid ceil(rows * 64 / 256).
extern "C" __global__ void k6_rope_cossin(const float* args, float* out, long long rows) {
    const long long p = (long long)blockIdx.x * K6C_BLOCK + threadIdx.x;
    if (p >= rows * 64) return;
    const float a = args[p];
    float a2 = a;
    asm volatile("" : "+f"(a2));
    const float c = cosf(a);
    const float s = sinf(a2);
    float4 o;
    o.x = c;
    o.y = -s;
    o.z = s;
    o.w = c;
    reinterpret_cast<float4*>(out)[p] = o;
}
