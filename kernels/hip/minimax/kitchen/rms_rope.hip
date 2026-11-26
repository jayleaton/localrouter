// Ours: a HIP port of comfy-kitchen's rms_rope (comfy_kitchen/backends/cuda/ops/rms_rope.cu @ v0.2.35, git blob
// 7f1a6091737d, and rope_device.cuh, blob a66c45bd7f95; Apache-2.0, see NOTICE), the one specialization MiniMax H3 runs:
// rope_kernel<bf16, bf16, bf16, HasRms, SplitHalf, HasK, InPlace, ContigHead> (kernels/cuda/minimax/kitchen_launch.cu).
// docs/dev/UPSTREAM.md entry 15. Same arguments and launch as that instantiation, so `Kitchen.rmsRope` launches either.
//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified for LocalRouter: ported to HIP, cut to one specialization, every rounding written out.
//
// CUDA builds it with --use_fast_math (rsqrtf and the division approximate, contractions free). This copy writes every
// rounding out instead, so its bits are its own, gated against torch's reference op (G0), not against CUDA:
//   sum   = fma(x, x, sum) over the lane's elements lane, lane + 32, .. (head_dim / 32 of them), then the wave's
//           shuffle-down tree 16, 8, 4, 2, 1, lane 0's total to every lane                         (rope_device.cuh)
//   rrms  = 1 / sqrt(sum / head_dim + eps)           IEEE division, correctly rounded sqrt, IEEE division
//   y     = bf16_rn((x * rrms) * w[d])               two f32 multiplies, left to right, one rounding to bf16
//   rot   = (f00 y0 + f01 y1, f10 y0 + f11 y1)        fma(f00, y0, f01 * y1), fma(f10, y0, f11 * y1)
//   out   = bf16_rn(rot); dims past rot_dim: y alone
// q, k [batch, dim1, dim2, head_dim] bf16 (strides in elements), freqs bf16 [.., rot_dim / 2, 2, 2], weights bf16.
//
// launch: block 128 (4 waves, one (b, i1, i2) row each), grid ceil(batch * dim1 * dim2 / 4), shared memory 0. Requires the
// contiguous-head case (q_s3 == k_s3 == 1, pairs 4-byte aligned, head_dim and rot_dim % 4 == 0), as H3's launch has.
#include <stdint.h>

namespace {

constexpr int kWaves = 4;

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

__device__ __forceinline__ uint16_t bf_rn(float f) {
    const uint32_t u = __float_as_uint(f);
    return (f != f) ? static_cast<uint16_t>(0x7fffu) : static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float rms_sum(const uint16_t* row, int head_dim, int lane) {
    float sum = 0.0f;
    for (int e = lane; e < head_dim; e += 32) {
        const float v = bf(row[e]);
        sum = __fmaf_rn(v, v, sum);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_down(sum, off, 32));
    return __shfl(sum, 0, 32);
}

__device__ __forceinline__ float rrms(float sum, int head_dim, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, static_cast<float>(head_dim)), eps)));
}

// One head row in place: norm, then rotate the split-half pairs (i, i + rot_dim / 2), two pairs a lane.
__device__ __forceinline__ void head(uint16_t* x, const uint16_t* __restrict__ w, int64_t ws, const uint16_t* __restrict__ freqs,
                                     int64_t freq_row, int64_t f_s3, int64_t f_s4, int64_t f_s5, int head_dim, int rot_dim, float r,
                                     int lane) {
    const int pairs = rot_dim / 2;
    for (int pb = lane * 2; pb < pairs; pb += 64) {
        uint16_t lo[2], hi[2], olo[2], ohi[2];
        __builtin_memcpy(lo, x + pb, 4);
        __builtin_memcpy(hi, x + pairs + pb, 4);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int pair = pb + p;
            const float y0 = bf(bf_rn(__fmul_rn(__fmul_rn(bf(lo[p]), r), bf(w[(int64_t)pair * ws]))));
            const float y1 = bf(bf_rn(__fmul_rn(__fmul_rn(bf(hi[p]), r), bf(w[(int64_t)(pair + pairs) * ws]))));
            const int64_t b = freq_row + (int64_t)pair * f_s3;
            const float f00 = bf(freqs[b]), f01 = bf(freqs[b + f_s5]), f10 = bf(freqs[b + f_s4]), f11 = bf(freqs[b + f_s4 + f_s5]);
            olo[p] = bf_rn(__fmaf_rn(f00, y0, __fmul_rn(f01, y1)));
            ohi[p] = bf_rn(__fmaf_rn(f10, y0, __fmul_rn(f11, y1)));
        }
        __builtin_memcpy(x + pb, olo, 4);
        __builtin_memcpy(x + pairs + pb, ohi, 4);
    }
    for (int d = rot_dim + lane; d < head_dim; d += 32) x[d] = bf_rn(__fmul_rn(__fmul_rn(bf(x[d]), r), bf(w[(int64_t)d * ws])));
}

}  // namespace

// rope_kernel<bf16, bf16, bf16, true, true, true, true, true>'s arguments (q_out = q, k_out = k, their strides unused).
extern "C" __global__ void __launch_bounds__(128)
    kitchen_rms_rope_bf16(const uint16_t* q, const uint16_t* k, const uint16_t* __restrict__ freqs, const uint16_t* __restrict__ q_scale,
                          const uint16_t* __restrict__ k_scale, uint16_t* q_out, uint16_t* k_out, int64_t batch, int64_t dim1,
                          int64_t dim2, int head_dim, int rot_dim, int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2,
                          int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0, int64_t k_s1, int64_t k_s2,
                          int64_t k_s3, int64_t qo_s0, int64_t qo_s1, int64_t qo_s2, int64_t qo_s3, int64_t ko_s0, int64_t ko_s1,
                          int64_t ko_s2, int64_t ko_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4,
                          int64_t f_s5, int64_t qs_stride, int64_t ks_stride, float epsilon) {
    const int lane = threadIdx.x & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kWaves + (threadIdx.x >> 5);
    if (row >= batch * dim1 * dim2) return;
    const int64_t i2 = row % dim2, i1 = (row / dim2) % dim1, i0 = row / dim2 / dim1;
    uint16_t* qr = q_out + i0 * q_s0 + i1 * q_s1 + i2 * q_s2;  // in place: q_out == q, strides q's
    uint16_t* kr = k_out + i0 * k_s0 + i1 * k_s1 + i2 * k_s2;
    const float qrr = rrms(rms_sum(qr, head_dim, lane), head_dim, epsilon);
    const float krr = rrms(rms_sum(kr, head_dim, lane), head_dim, epsilon);
    const int64_t fr = (freqs_batch == 1 ? 0 : i0) * f_s0 + (freqs_dim1 == 1 ? 0 : i1) * f_s1 + (freqs_dim2 == 1 ? 0 : i2) * f_s2;
    head(qr, q_scale, qs_stride, freqs, fr, f_s3, f_s4, f_s5, head_dim, rot_dim, qrr, lane);
    head(kr, k_scale, ks_stride, freqs, fr, f_s3, f_s4, f_s5, head_dim, rot_dim, krr, lane);
}
