// Ours: the CUDA names LocalRouter's kernels use, on HIP for gfx11 (docs/dev/ROCM.md). build/hip.zig force-includes it
// (-include cuda_compat.h, with kernels/hip/compat on the include path for cuda_bf16.h and friends), so the CUDA sources
// compile for HIP unchanged and their CUDA SASS cannot move. A mapping either is the same operation (HIP's __fadd_rn and
// the rest are IEEE round-to-nearest on AMD hardware) or writes CUDA's arithmetic out (the conversions below).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;

// cuda_bf16.hpp's __float2bfloat16_rn: round to nearest even on the bits, every NaN to the canonical 0x7fff.
__host__ __device__ __forceinline__ __nv_bfloat16 lr_float2bfloat16_rn(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    const uint16_t b = (f != f) ? (uint16_t)0x7fffu : (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    __nv_bfloat16 r;
    __builtin_memcpy(&r, &b, 2);
    return r;
}
#define __float2bfloat16_rn lr_float2bfloat16_rn

// CUDA's warp shuffles take a 32-bit lane mask; HIP's take 64 bits. On gfx11 (wave32) a full 32-bit mask is every lane,
// which is what the unmasked HIP shuffles over the wave do.
#define __shfl_sync(mask, v, src, ...) __shfl((v), (src), ##__VA_ARGS__)
#define __shfl_xor_sync(mask, v, m, ...) __shfl_xor((v), (m), ##__VA_ARGS__)
#define __shfl_down_sync(mask, v, d, ...) __shfl_down((v), (d), ##__VA_ARGS__)
#define __shfl_up_sync(mask, v, d, ...) __shfl_up((v), (d), ##__VA_ARGS__)

// cuda_fp8.h's __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3): round to nearest even, finite values past 448 to
// +-448 (0x7e / 0xfe), NaN to 0x7f. E4M3: bias 7, 3 mantissa bits, subnormals k * 2^-9. The one form the kernels use.
typedef unsigned char __nv_fp8_storage_t;
enum lr_fp8_saturation { __NV_NOSAT, __NV_SATFINITE };
enum lr_fp8_interpretation { __NV_E4M3, __NV_E5M2 };
__host__ __device__ __forceinline__ __nv_fp8_storage_t lr_float_to_e4m3_satfinite(float x) {
    uint32_t u;
    __builtin_memcpy(&u, &x, 4);
    if (x != x) return 0x7f;
    const uint32_t sign = (u >> 24) & 0x80u;
    const uint32_t ua = u & 0x7fffffffu;
    float a;
    __builtin_memcpy(&a, &ua, 4);
    uint32_t code;
    if (a < 0.015625f) { // below 2^-6: a subnormal (or zero) k * 2^-9, k = rint(a * 2^9) exactly; 8 is the least normal
        code = (uint32_t)__builtin_rintf(a * 512.0f);
    } else if (a >= 464.0f) { // 464 ties between 448 (even) and 480 (odd): 448; above it, past 448: saturates either way
        code = 0x7e;
    } else {
        uint32_t e = (ua >> 23) - 127u + 7u;
        uint32_t m = (ua >> 20) & 7u;
        const uint32_t rem = ua & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) ++m;
        if (m == 8u) {
            m = 0;
            ++e;
        }
        code = (e << 3) | m;
    }
    return (__nv_fp8_storage_t)(sign | code);
}
#define __nv_cvt_float_to_fp8(x, sat, kind) lr_float_to_e4m3_satfinite(x)
