// Qwen2.5-VL BF16 attention, matching the pinned reference's EFFICIENT_ATTENTION kernel.
// Header provenance and licenses: vendor/README.md. No Torch runtime or vendor library calls.
#include <cuda/std/bit>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

using Attention = PyTorchMemEffAttention::AttentionKernel<
    cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>;
// Keep the Zig launch contract checked against the actual compiled specialization.
static_assert(Attention::kNumWarpsPerBlock == 8);
static_assert(sizeof(Attention::SharedStorage) == 52224);
struct ZeroParams {
    alignas(Attention::Params) unsigned char bytes[sizeof(Attention::Params)];
};

// The reference expands its causal + padding mask to BF16, then passes the full key length.
// Trimming padded keys changes the online softmax's arithmetic, even for masked values.
extern "C" __global__ void k6t_attention_bias(uint16_t* bias, int length, int stride, int valid) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length * stride) return;
    const int row = index / stride, col = index % stride;
    bias[index] = col <= row && col < valid ? 0 : 0xff80; // BF16 zero / negative infinity
}

extern "C" __global__ void __launch_bounds__(Attention::kNumThreads, Attention::kMinBlocksPerSm)
k6t_qwen_attention(const cutlass::bfloat16_t* q, const cutlass::bfloat16_t* k,
                   const cutlass::bfloat16_t* v, const cutlass::bfloat16_t* bias,
                   cutlass::bfloat16_t* out, int length, int bias_stride) {
    // All unused optional pointers, mask/dropout flags and RNG payload are zero.
    // bit_cast avoids invoking Params' host-only Philox constructor on the device.
    Attention::Params p = cuda::std::bit_cast<Attention::Params>(ZeroParams{});
    p.query_ptr = q; p.key_ptr = k; p.value_ptr = v;
    p.attn_bias_ptr = bias; p.output_ptr = out;
    p.scale = 0.08838834764831845f;
    p.head_dim = p.head_dim_value = 128;
    p.num_queries = p.num_keys = p.num_keys_absolute = length;
    p.q_strideM = p.o_strideM = 28 * 128;
    p.k_strideM = p.v_strideM = 4 * 128;
    p.bias_strideM = bias_stride;
    p.q_strideH = p.k_strideH = p.v_strideH = 128;
    p.q_strideB = length * 28 * 128;
    p.k_strideB = p.v_strideB = length * 4 * 128;
    p.num_batches = 1; p.num_heads = 28; p.q_heads_per_kv = 7;
    if (!p.advance_to_block()) return;
    Attention::attention_kernel(p);
}

// CLIP's captured FP32 EFFICIENT_ATTENTION specialization, full 77-key causal/padding bias.
using ClipAttention = PyTorchMemEffAttention::AttentionKernel<
    float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>;
static_assert(ClipAttention::kNumThreads == 128);
static_assert(sizeof(ClipAttention::SharedStorage) == 36352);
struct ClipZeroParams {
    alignas(ClipAttention::Params) unsigned char bytes[sizeof(ClipAttention::Params)];
};
extern "C" __global__ void k6t_clip_attention_bias(float* bias, int length, int stride, int valid) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length * stride) return;
    const int row = index / stride, col = index % stride;
    bias[index] = col <= row && col < valid ? 0.0f : __int_as_float(0xff800000);
}
extern "C" __global__ void __launch_bounds__(ClipAttention::kNumThreads, ClipAttention::kMinBlocksPerSm)
k6t_clip_attention(const float* q, const float* k, const float* v, const float* bias,
                   float* out, int length, int bias_stride) {
    ClipAttention::Params p = cuda::std::bit_cast<ClipAttention::Params>(ClipZeroParams{});
    p.query_ptr = q; p.key_ptr = k; p.value_ptr = v;
    p.attn_bias_ptr = bias; p.output_ptr = out;
    p.scale = 0.125f;
    p.head_dim = p.head_dim_value = 64;
    p.num_queries = p.num_keys = p.num_keys_absolute = length;
    p.q_strideM = p.o_strideM = p.k_strideM = p.v_strideM = 12 * 64;
    p.bias_strideM = bias_stride;
    p.q_strideH = p.k_strideH = p.v_strideH = 64;
    p.q_strideB = p.k_strideB = p.v_strideB = length * 12 * 64;
    p.num_batches = 1; p.num_heads = 12; p.q_heads_per_kv = 1;
    if (!p.advance_to_block()) return;
    ClipAttention::attention_kernel(p);
}
