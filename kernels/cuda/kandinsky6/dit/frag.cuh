// The tensor-core fragment helpers the Kandinsky-6 DiT kernels share: the subset of kernels/cuda/qwen_image/qmm_frag.cuh
// (blob 3077d8da77b8f6f46eaedf1944b40b6aff694426, itself tensorfold-zig 88c424e zig/kernels/cuda/qmm_frag.cuh,
// Apache-2.0, see NOTICE) that gemm.cu and attention.cu use, unchanged, in their own namespace so a change to the image
// engine's copy cannot move this engine's bits.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace k6_frag {

__device__ __forceinline__ uint32_t smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem(dst)), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

__device__ __forceinline__ void ldmatrix4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}

} // namespace k6_frag
