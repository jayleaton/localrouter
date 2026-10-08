// comfy_kitchen/backends/cuda/ops/rms_rope.cu @ v0.2.35 (git blob 7f1a6091737d; host launchers cut, unnamed namespace -> kitchen_rms_rope); copied by tools/kernels/sync_kitchen.py, do not edit
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Unified apply-RoPE and RMSNorm+RoPE kernel family. All logical dimensions
 * are addressed through element strides; the contiguous-head specialization
 * contains no runtime layout branch.
 */
#include "dtype_dispatch.cuh"
#include "rope_device.cuh"
#include "utils.cuh"

#include <cstdint>
#include <type_traits>

namespace comfy {
namespace kitchen_rms_rope {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * kThreadsPerWarp;

template <typename InputType, typename FreqsType, typename ScaleType,
          bool HasRms, bool SplitHalf, bool HasK, bool InPlace, bool ContigHead>
__global__ __launch_bounds__(kThreads) void rope_kernel(
    const InputType *q, const InputType *k,
    const FreqsType *__restrict__ freqs,
    const ScaleType *__restrict__ q_scale,
    const ScaleType *__restrict__ k_scale, InputType *q_out,
    InputType *k_out, int64_t batch, int64_t dim1, int64_t dim2,
    int head_dim, int rot_dim, int64_t freqs_batch, int64_t freqs_dim1,
    int64_t freqs_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2,
    int64_t q_s3, int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3,
    int64_t qo_s0, int64_t qo_s1, int64_t qo_s2, int64_t qo_s3,
    int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4,
    int64_t f_s5, int64_t qs_stride, int64_t ks_stride, float epsilon) {
  using ComputeType =
      std::conditional_t<HasRms, float, FreqsType>;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t rows = batch * dim1 * dim2;
  if (row >= rows) {
    return;
  }

  const int64_t i2 = row % dim2;
  const int64_t tmp = row / dim2;
  const int64_t i1 = tmp % dim1;
  const int64_t i0 = tmp / dim1;
  const int64_t q_base = i0 * q_s0 + i1 * q_s1 + i2 * q_s2;
  const int64_t qo_base =
      InPlace ? q_base : i0 * qo_s0 + i1 * qo_s1 + i2 * qo_s2;
  int64_t k_base = 0;
  int64_t ko_base = 0;
  if constexpr (HasK) {
    k_base = i0 * k_s0 + i1 * k_s1 + i2 * k_s2;
    ko_base =
        InPlace ? k_base : i0 * ko_s0 + i1 * ko_s1 + i2 * ko_s2;
  }

  float q_rrms = 1.0f;
  float k_rrms = 1.0f;
  if constexpr (HasRms) {
    const float q_sum =
        rope::rms_sum<InputType, ContigHead>(q + q_base, head_dim, q_s3, lane);
    q_rrms = rsqrtf(q_sum / static_cast<float>(head_dim) + epsilon);
    if constexpr (HasK) {
      const float k_sum = rope::rms_sum<InputType, ContigHead>(
          k + k_base, head_dim, k_s3, lane);
      k_rrms = rsqrtf(k_sum / static_cast<float>(head_dim) + epsilon);
    }
  }

  const int64_t fi0 = freqs_batch == 1 ? 0 : i0;
  const int64_t fi1 = freqs_dim1 == 1 ? 0 : i1;
  const int64_t fi2 = freqs_dim2 == 1 ? 0 : i2;
  const int64_t freq_row = fi0 * f_s0 + fi1 * f_s1 + fi2 * f_s2;
  // Rotation covers the first rot_dim dims (split-half pairs (i, i + rot_dim/2));
  // the RMS reduction above always spans the full head_dim.
  const int pairs = rot_dim / 2;
  constexpr int kPairsPerLane = SplitHalf && ContigHead ? 2 : 1;

  for (int pair_base = lane * kPairsPerLane; pair_base < pairs;
       pair_base += kThreadsPerWarp * kPairsPerLane) {
    InputType q0_raw[kPairsPerLane], q1_raw[kPairsPerLane];
    InputType k0_raw[kPairsPerLane], k1_raw[kPairsPerLane];

    if constexpr (SplitHalf && ContigHead) {
      const auto q_lo =
          *reinterpret_cast<const rope::Pair<InputType> *>(q + q_base + pair_base);
      const auto q_hi = *reinterpret_cast<const rope::Pair<InputType> *>(
          q + q_base + pairs + pair_base);
      q0_raw[0] = q_lo.x;
      q0_raw[1] = q_lo.y;
      q1_raw[0] = q_hi.x;
      q1_raw[1] = q_hi.y;
      if constexpr (HasK) {
        const auto k_lo = *reinterpret_cast<const rope::Pair<InputType> *>(
            k + k_base + pair_base);
        const auto k_hi = *reinterpret_cast<const rope::Pair<InputType> *>(
            k + k_base + pairs + pair_base);
        k0_raw[0] = k_lo.x;
        k0_raw[1] = k_lo.y;
        k1_raw[0] = k_hi.x;
        k1_raw[1] = k_hi.y;
      }
    } else {
      rope::load_head_pair<InputType, SplitHalf, ContigHead>(
          q + q_base, pair_base, pairs, q_s3, q0_raw[0], q1_raw[0]);
      if constexpr (HasK) {
        rope::load_head_pair<InputType, SplitHalf, ContigHead>(
            k + k_base, pair_base, pairs, k_s3, k0_raw[0], k1_raw[0]);
      }
    }

    InputType qo0_raw[kPairsPerLane], qo1_raw[kPairsPerLane];
    InputType ko0_raw[kPairsPerLane], ko1_raw[kPairsPerLane];
#pragma unroll
    for (int p = 0; p < kPairsPerLane; ++p) {
      const int pair = pair_base + p;
      ComputeType q0 = static_cast<ComputeType>(q0_raw[p]);
      ComputeType q1 = static_cast<ComputeType>(q1_raw[p]);
      const int first = SplitHalf ? pair : pair * 2;
      const int second = SplitHalf ? pair + pairs : first + 1;
      if constexpr (HasRms) {
        q0 = static_cast<float>(static_cast<InputType>(
            static_cast<float>(q0) * q_rrms *
            static_cast<float>(
                q_scale[static_cast<int64_t>(first) * qs_stride])));
        q1 = static_cast<float>(static_cast<InputType>(
            static_cast<float>(q1) * q_rrms *
            static_cast<float>(
                q_scale[static_cast<int64_t>(second) * qs_stride])));
      }

      FreqsType f00_raw, f01_raw, f10_raw, f11_raw;
      rope::load_rotation(freqs, freq_row + static_cast<int64_t>(pair) * f_s3,
                          f_s4, f_s5, f00_raw, f01_raw, f10_raw, f11_raw);
      const ComputeType f00 = static_cast<ComputeType>(f00_raw);
      const ComputeType f01 = static_cast<ComputeType>(f01_raw);
      const ComputeType f10 = static_cast<ComputeType>(f10_raw);
      const ComputeType f11 = static_cast<ComputeType>(f11_raw);
      ComputeType qo0, qo1;
      rope::rotate(q0, q1, f00, f01, f10, f11, qo0, qo1);
      qo0_raw[p] = static_cast<InputType>(qo0);
      qo1_raw[p] = static_cast<InputType>(qo1);

      if constexpr (HasK) {
        ComputeType k0 = static_cast<ComputeType>(k0_raw[p]);
        ComputeType k1 = static_cast<ComputeType>(k1_raw[p]);
        if constexpr (HasRms) {
          k0 = static_cast<float>(static_cast<InputType>(
              static_cast<float>(k0) * k_rrms *
              static_cast<float>(
                  k_scale[static_cast<int64_t>(first) * ks_stride])));
          k1 = static_cast<float>(static_cast<InputType>(
              static_cast<float>(k1) * k_rrms *
              static_cast<float>(
                  k_scale[static_cast<int64_t>(second) * ks_stride])));
        }
        ComputeType ko0, ko1;
        rope::rotate(k0, k1, f00, f01, f10, f11, ko0, ko1);
        ko0_raw[p] = static_cast<InputType>(ko0);
        ko1_raw[p] = static_cast<InputType>(ko1);
      }
    }

    if constexpr (SplitHalf && ContigHead) {
      *reinterpret_cast<rope::Pair<InputType> *>(q_out + qo_base + pair_base) =
          {qo0_raw[0], qo0_raw[1]};
      *reinterpret_cast<rope::Pair<InputType> *>(
          q_out + qo_base + pairs + pair_base) = {qo1_raw[0], qo1_raw[1]};
      if constexpr (HasK) {
        *reinterpret_cast<rope::Pair<InputType> *>(
            k_out + ko_base + pair_base) = {ko0_raw[0], ko0_raw[1]};
        *reinterpret_cast<rope::Pair<InputType> *>(
            k_out + ko_base + pairs + pair_base) = {ko1_raw[0], ko1_raw[1]};
      }
    } else {
      rope::store_head_pair<InputType, SplitHalf, ContigHead>(
          q_out + qo_base, pair_base, pairs, InPlace ? q_s3 : qo_s3,
          qo0_raw[0], qo1_raw[0]);
      if constexpr (HasK) {
        rope::store_head_pair<InputType, SplitHalf, ContigHead>(
            k_out + ko_base, pair_base, pairs, InPlace ? k_s3 : ko_s3,
            ko0_raw[0], ko1_raw[0]);
      }
    }
  }

  // Norm-only tail: dims beyond rot_dim are normalized and scaled but never
  // rotated. Empty in the common rot_dim == head_dim case.
  const int64_t qo_s3_eff = InPlace ? q_s3 : qo_s3;
  const int64_t ko_s3_eff = InPlace ? k_s3 : ko_s3;
  for (int d = rot_dim + lane; d < head_dim; d += kThreadsPerWarp) {
    InputType qv = q[q_base + static_cast<int64_t>(d) * q_s3];
    if constexpr (HasRms) {
      qv = static_cast<InputType>(
          static_cast<float>(qv) * q_rrms *
          static_cast<float>(q_scale[static_cast<int64_t>(d) * qs_stride]));
    }
    q_out[qo_base + static_cast<int64_t>(d) * qo_s3_eff] = qv;
    if constexpr (HasK) {
      InputType kv = k[k_base + static_cast<int64_t>(d) * k_s3];
      if constexpr (HasRms) {
        kv = static_cast<InputType>(
            static_cast<float>(kv) * k_rrms *
            static_cast<float>(k_scale[static_cast<int64_t>(d) * ks_stride]));
      }
      k_out[ko_base + static_cast<int64_t>(d) * ko_s3_eff] = kv;
    }
  }
}
} // namespace kitchen_rms_rope
} // namespace comfy
