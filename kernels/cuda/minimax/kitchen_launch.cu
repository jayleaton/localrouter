// Ours: the host side of two comfy-kitchen v0.2.35 ops (Apache-2.0, the device code in kitchen/, tools/kernels/sync_kitchen.py)
// MiniMax H3 calls: torch.ops.comfy_kitchen.int8_attention and comfy_kitchen.rms_rope_split_half_. Every kernel, template
// argument, grid, block, shared-memory size, workspace layout and Hadamard / K-shift choice below is the wheel's for the
// cases H3 uses (bf16, CUDA; int8_attention: no mask, no GQA padding, D 64 or 128, scale given; rms_rope: split half, in
// place, bf16 freqs and weights). The twin compiles this file into a torch extension (stk_twin/h3/kitchen.py); the Zig
// engine compiles it to a fatbin and launches the kernels below with cuLaunchKernel, in the order and with the arguments
// listed here (the extern "C" functions are the twin's: their host code is not in the fatbin). Build flags are the wheel's
// (comfy_kitchen/backends/cuda/CMakeLists.txt): -O3 --use_fast_math (it decides `mx / 127.f`, `1.f / sc`, rsqrtf and
// expf) --expt-relaxed-constexpr --expt-extended-lambda -U__CUDA_NO_{HALF,BFLOAT16,BFLOAT162}_{OPERATORS,CONVERSIONS}__.
//
// ------------------------------------------------------------------------------------------------- int8_attention
// Python: comfy_kitchen/sage_attention.py _int8_attention_cuda -> _C.sage_sdpa (backends/cuda/dlpack_bindings.cpp
// sage_sdpa) -> launch_quant_qk_per_thread_int8, launch_quant_v_int8_kernel, launch_sage_attn_kernel.
// q [B,H,Sq,D], k, v [B,HK,Sk,D] bf16, last stride 1, any other strides (H3: views of a [S, 3*H*D] qkv buffer); out [B,H,Sq,D]
// bf16 contiguous; scale = D**-0.5 rounded to f32 when not given (D 64: 0x3e000000, D 128: 0x3db504f3).
//   cta_k    = (D >= 128 && Sk > 1024) ? 128 : 64          (_select_cta_k, no mask)
//   padded_k = ceil(Sk / cta_k) * cta_k
//   Hadamard (ROT) of Q and K: Sk <= 256: 4 (H4 blocks); else D 64: 64, D 128: 128 (the signed H128, "convrot128").
// Workspace (kitchen_int8_attention_workspace_bytes; each part starts at a multiple of 256 bytes, in this order):
//   q_int8 B*H*Sq*D i8 | k_int8 B*HK*Sk*D i8 | v_int8 B*HK*D*padded_k i8 | q_scale B*H*(ceil(Sq/128)*32) f32 |
//   k_scale B*HK*(ceil(Sk/cta_k)*4) f32 | v_scale B*HK*D f32 | anchor B*HK i32.
// Launch sequence on one stream (all blocks 1-D or 2-D as written; dynamic smem 0 unless stated):
//   1. kitchen_quant_qk::detect_k_anchor<bf16>           grid (HK, B)  block 128
//        _ZN16kitchen_quant_qk15detect_k_anchorI13__nv_bfloat16EEvPKT_Piiiilll
//        (const bf16* k, int* anchor, int Lk=Sk, int C=D, int H_kv=HK, i64 stride_b, i64 stride_h, i64 stride_n)   [k's strides]
//        Nine evenly spaced keys per (b, h) give the "representative key" (an absolute row, or -1: no shift).
//   2. kitchen_quant_qk::quant_qk_fused<bf16, NR, NL, BLKQ, WARPQ, BLKK, WARPK, CT, ROT, A4>   grid (q_oblk + k_oblk, max(H, HK), B)  block 128
//        q_oblk = ceil(Sq/128)*4, k_oblk = ceil(Sk/cta_k); q_sc_per_h = q_oblk*8, k_sc_per_h = k_oblk*4
//        (const bf16* q, i8* q_int8, f32* q_scale, const bf16* k, i8* k_int8, f32* k_scale, const int* anchor, int Lq, int Lk,
//         int C, int q_oblk_count, int H_q, int H_kv, int q_sc_per_h, int k_sc_per_h, i64 q_stride_b, q_stride_h, q_stride_n,
//         i64 k_stride_b, k_stride_h, k_stride_n)
//        D 64  (cta 64):  <bf16,4,8,128,32,64,64,1,ROT,false>   ROT 4 or 64
//          _ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi8ELi128ELi32ELi64ELi64ELi1ELi4ELb0EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll
//          _ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi8ELi128ELi32ELi64ELi64ELi1ELi64ELb0EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll
//        D 128 (cta 64):  <bf16,4,8,128,32,64,64,1,ROT,true>    ROT 4 or 128
//          _ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi8ELi128ELi32ELi64ELi64ELi1ELi4ELb1EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll
//          _ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi8ELi128ELi32ELi64ELi64ELi1ELi128ELb1EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll
//        D 128 (cta 128, Sk > 1024): <bf16,4,16,128,32,128,128,1,128,true>
//          _ZN16kitchen_quant_qk14quant_qk_fusedI13__nv_bfloat16Li4ELi16ELi128ELi32ELi128ELi128ELi1ELi128ELb1EEEvPKT_PaPfS4_S5_S6_PKiiiiiiiiillllll
//   3. kitchen_quant_v::quant_v_int8_kernel<bf16, THREADS>   grid B*HK*(D/8)  block THREADS = (Sk <= 256 ? 128 : 512)
//        _ZN15kitchen_quant_v19quant_v_int8_kernelI13__nv_bfloat16Li128EEEvPKT_PaPfiiiilll   (THREADS 128)
//        _ZN15kitchen_quant_v19quant_v_int8_kernelI13__nv_bfloat16Li512EEEvPKT_PaPfiiiilll   (THREADS 512)
//        (const bf16* v, i8* v_int8, f32* v_scale, int N=Sk, int padded_N=padded_k, int H=HK, int D, i64 sb, i64 sh, i64 sn)
//   4. qk_int_sv_i8_attn_kernel<128, CTA_K, WARP_Q, CTA_K, D, kInt8, kPerThread, kPerThread, float, false, bf16, kCudaCore,
//        kNone, false, true, false, false, FUSE>   (global namespace)   grid (ceil(Sq/128), H, B)   block (32, 128/WARP_Q)
//        dynamic smem max(128*D + 2*CTA_K*D, 256*D) (set with cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES first);
//        WARP_Q = D >= 128 ? 16 : 32; FUSE = false only when Sk <= 512 and cta_k == 64 (the kernel honours it on sm_100 and up).
//          D 128, cta 64,  FUSE false: block (32,8)  smem 32768
//   _Z24qk_int_sv_i8_attn_kernelILj128ELj64ELj16ELj64ELj128EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb0EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf
//          D 128, cta 64,  FUSE true:  block (32,8)  smem 32768
//   _Z24qk_int_sv_i8_attn_kernelILj128ELj64ELj16ELj64ELj128EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb1EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf
//          D 128, cta 128, FUSE true:  block (32,8)  smem 49152
//   _Z24qk_int_sv_i8_attn_kernelILj128ELj128ELj16ELj128ELj128EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb1EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf
//          D 64,  cta 64,  FUSE false: block (32,4)  smem 16384
//   _Z24qk_int_sv_i8_attn_kernelILj128ELj64ELj32ELj64ELj64EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb0EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf
//          D 64,  cta 64,  FUSE true:  block (32,4)  smem 16384
//   _Z24qk_int_sv_i8_attn_kernelILj128ELj64ELj32ELj64ELj64EL8DataType1EL16QuantGranularity3ELS1_3EfLb0E13__nv_bfloat16L11ComputeUnit1EL8MaskMode0ELb0ELb1ELb0ELb0ELb1EEvPaS5_S5_PT9_PfS8_S8_S8_S8_PKvllllijjjjjjjjjjjjjjjf
//        (i8* Q=q_int8, i8* K=k_int8, i8* V=v_int8, bf16* O, f32* Lse=0, f32* Q_scale, f32* K_scale, f32* V_scale, f32* V_mean=0,
//         const void* AttnMask=0, i64 mask_stride_b=0, mask_stride_h=0, mask_stride_q=0, mask_stride_k=0, int mask_dtype_code=-1,
//         u32 qo_len=Sq, kv_len=Sk, num_kv_groups=H/HK,
//         u32 stride_bz_q=H*Sq*D, stride_seq_q=D, stride_h_q=Sq*D,  stride_bz_k=HK*Sk*D, stride_seq_k=D, stride_h_k=Sk*D,
//         u32 stride_bz_v=HK*D*padded_k, stride_h_v=D*padded_k, stride_d_v=padded_k,
//         u32 stride_bz_o=H*Sq*D, stride_seq_o=D, stride_h_o=Sq*D, f32 sm_scale=scale)
//
// ---------------------------------------------------------------------------------------- rms_rope_split_half_
// Python: comfy_kitchen/backends/cuda/__init__.py _rms_rope_cuda -> _C.rms_rope -> launch_rms_rope_kernel -> rope_launcher.
// q, k [1, S, H, 128] bf16 (dims batch, dim1 = S, dim2 = H; strides in elements: (_, 3*H*128, 128, 1) for views into one
// qkv buffer), freqs bf16 [1, S, 1, rot_dim/2, 2, 2], norm weights bf16 [128] (stride 1), in place (q_out = q, k_out = k).
// One launch, no workspace: comfy::kitchen_rms_rope::rope_kernel<bf16, bf16, bf16, HasRms = true, SplitHalf = true,
//   HasK = true, InPlace = true, ContigHead>   grid ceil(batch*dim1*dim2 / 4)   block 128 (4 warps, one (b, i1, i2) row each)   smem 0
//   ContigHead = q_s3 == 1 && k_s3 == 1 && each of q, k has pointer % 4 == 0 and s0, s1, s2 even && head_dim % 4 == 0 &&
//   rot_dim % 4 == 0 (true for H3; the two specializations round alike: same per-element expressions, same reduction order).
//   _ZN5comfy16kitchen_rms_rope11rope_kernelI13__nv_bfloat16S2_S2_Lb1ELb1ELb1ELb1ELb1EEEvPKT_S5_PKT0_PKT1_SB_PS3_SC_llliilllllllllllllllllllllllllllf
//   _ZN5comfy16kitchen_rms_rope11rope_kernelI13__nv_bfloat16S2_S2_Lb1ELb1ELb1ELb1ELb0EEEvPKT_S5_PKT0_PKT1_SB_PS3_SC_llliilllllllllllllllllllllllllllf
//   (const bf16* q, const bf16* k, const bf16* freqs, const bf16* q_scale, const bf16* k_scale, bf16* q_out, bf16* k_out,
//    i64 batch, i64 dim1, i64 dim2, int head_dim, int rot_dim, i64 freqs_batch, freqs_dim1, freqs_dim2,
//    i64 q_s0, q_s1, q_s2, q_s3, k_s0, k_s1, k_s2, k_s3, qo_s0, qo_s1, qo_s2, qo_s3 (= q's), ko_s0, ko_s1, ko_s2, ko_s3 (= k's),
//    i64 f_s0, f_s1, f_s2, f_s3, f_s4, f_s5, i64 qs_stride, ks_stride, f32 epsilon)
//   H3: batch 1, dim1 S, dim2 H, head_dim 128, rot_dim 96, freqs_batch 1, freqs_dim1 S, freqs_dim2 1; f_s = (S*rot_dim*2, rot_dim*2,
//   rot_dim*2, 4, 2, 1) for a contiguous freqs (a size-1 dim's stride is never used: its index is 0).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>

#include "kitchen/quant_qk_int8.cu"
#include "kitchen/quant_v_int8.cu"
#include "kitchen/qk_int_sv_i8_cuda.cuh"
#include "kitchen/rms_rope.cu"

namespace {

typedef nv_bfloat16 bf16;
constexpr int ERR_ARGS = (int)cudaErrorInvalidValue;

inline size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

struct AttnPlan {
    int cta_k, padded_k;
    size_t off_q8, off_k8, off_v8, off_qs, off_ks, off_vs, off_anchor, total;
};

AttnPlan attn_plan(int B, int H, int HK, int Sq, int Sk, int D) {
    AttnPlan p;
    p.cta_k = (D >= 128 && Sk > 1024) ? 128 : 64;
    p.padded_k = (Sk + p.cta_k - 1) / p.cta_k * p.cta_k;
    const size_t q_scales = (size_t)((Sq + 127) / 128) * 32, k_scales = (size_t)((Sk + p.cta_k - 1) / p.cta_k) * 4;
    size_t at = 0;
    auto take = [&](size_t bytes) { size_t o = at; at = align256(at + bytes); return o; };
    p.off_q8 = take((size_t)B * H * Sq * D);
    p.off_k8 = take((size_t)B * HK * Sk * D);
    p.off_v8 = take((size_t)B * HK * D * p.padded_k);
    p.off_qs = take((size_t)B * H * q_scales * sizeof(float));
    p.off_ks = take((size_t)B * HK * k_scales * sizeof(float));
    p.off_vs = take((size_t)B * HK * D * sizeof(float));
    p.off_anchor = take((size_t)B * HK * sizeof(int));
    p.total = at;
    return p;
}

// launch_impl of sage_attn_launcher.cu at mask kNone, bf16 out. o_rows (ours): the output's strides (seq, head) are (H * D, D),
// o as [B, Sq, H, D] rows, instead of the wheel's (D, Sq * D), o as [B, H, Sq, D]. The kernel stores through
// stride_seq_o / stride_h_o only (qk_int_sv_i8_cuda.cuh, the 128-bit stores at O + b * stride_bz_o + h * stride_h_o +
// row * stride_seq_o), so every output value is the same and only its address differs; B must be 1 for the rows form.
template <int HD, int CTA_K, bool FUSE>
int attn_launch(int8_t* q8, int8_t* k8, int8_t* v8, bf16* o, float* qs, float* ks, float* vs, int Sq, int Sk, int H,
                int groups, int B, int D, int padded_k, float scale, bool o_rows, cudaStream_t stream) {
    constexpr int CTA_Q = 128;
    constexpr int WARP_Q = HD >= 128 ? 16 : 32;
    constexpr int WARP_K = CTA_K;
    const size_t smem = std::max(static_cast<size_t>(CTA_Q * HD * sizeof(int8_t) + CTA_K * HD * sizeof(int8_t) + CTA_K * HD * sizeof(int8_t)),
                                 static_cast<size_t>(CTA_Q * HD * sizeof(half)));
    auto kernel = qk_int_sv_i8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HD, DataType::kInt8, QuantGranularity::kPerThread,
                                           QuantGranularity::kPerThread, float, false, bf16, ComputeUnit::kCudaCore,
                                           MaskMode::kNone, false, true, false, false, FUSE>;
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (e != cudaSuccess) return (int)e;
    const int Hk = H / groups;
    const dim3 grid(div_ceil(Sq, CTA_Q), H, B);
    const dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
    kernel<<<grid, block, smem, stream>>>(
        q8, k8, v8, o, nullptr, qs, ks, vs, nullptr, nullptr, 0, 0, 0, 0, -1, Sq, Sk, groups,
        H * Sq * D, D, Sq * D, Hk * Sk * D, D, Sk * D, Hk * D * padded_k, D * padded_k, padded_k, H * Sq * D,
        o_rows ? H * D : D, o_rows ? D : Sq * D, scale);
    return (int)cudaGetLastError();
}

// DISPATCH_DTYPE of the launcher for kNone and bf16.
template <int HD>
int attn_dispatch(int cta_k, int Sk, int8_t* q8, int8_t* k8, int8_t* v8, bf16* o, float* qs, float* ks, float* vs, int Sq,
                  int H, int groups, int B, int D, int padded_k, float scale, bool o_rows, cudaStream_t stream) {
    if (Sk <= 512 && cta_k == 64)
        return attn_launch<HD, 64, false>(q8, k8, v8, o, qs, ks, vs, Sq, Sk, H, groups, B, D, padded_k, scale, o_rows, stream);
    if constexpr (HD == 128) {
        if (cta_k == 128)
            return attn_launch<128, 128, true>(q8, k8, v8, o, qs, ks, vs, Sq, Sk, H, groups, B, D, padded_k, scale, o_rows, stream);
    }
    return attn_launch<HD, 64, true>(q8, k8, v8, o, qs, ks, vs, Sq, Sk, H, groups, B, D, padded_k, scale, o_rows, stream);
}

// LAUNCH_FUSED of quant_qk_int8.cu: the configurations the op reaches (D 64 and D 128 at cta 64, D 128 at cta 128).
template <int NL, int BK, bool A4, int ROT>
void quant_qk(const bf16* q, int8_t* q8, float* qs, const bf16* k, int8_t* k8, float* ks, const int* anchor, int B, int H,
              int HK, int Sq, int Sk, int D, int q_oblk, int k_oblk, int64_t q_sb, int64_t q_sh, int64_t q_sn, int64_t k_sb,
              int64_t k_sh, int64_t k_sn, cudaStream_t stream) {
    const dim3 g(q_oblk + k_oblk, H > HK ? H : HK, B);
    kitchen_quant_qk::quant_qk_fused<bf16, 4, NL, 128, 32, BK, BK, 1, ROT, A4><<<g, 128, 0, stream>>>(
        q, q8, qs, k, k8, ks, anchor, Sq, Sk, D, q_oblk, H, HK, q_oblk * 8, k_oblk * 4, q_sb, q_sh, q_sn, k_sb, k_sh, k_sn);
}

// The wheel's alignment checks (an extent of one forms no offset, so its stride is exempt): Q and K for the 4-element
// vector loads of quant_qk_int8.cu, V for the 16-byte loads of quant_v_int8.cu.
bool qk_aligned(const void* p, int64_t s_b, int64_t s_h, int64_t s_n, int e_b, int e_h, int e_n) {
    auto ok = [](int64_t s, int e) { return e < 2 || (s > 0 && s % 4 == 0); };
    return reinterpret_cast<uintptr_t>(p) % (4 * sizeof(half)) == 0 && ok(s_b, e_b) && ok(s_h, e_h) && ok(s_n, e_n);
}

bool v_aligned(const void* p, int64_t s_b, int64_t s_h, int64_t s_n, int e_b, int e_h, int e_n) {
    auto ok = [](int64_t s, int e) { return e < 2 || (static_cast<size_t>(s) * sizeof(half)) % 16 == 0; };
    return reinterpret_cast<uintptr_t>(p) % 16 == 0 && ok(s_b, e_b) && ok(s_h, e_h) && ok(s_n, e_n);
}

template <typename T>
bool pair_aligned(const T* p, int64_t s0, int64_t s1, int64_t s2) {
    return reinterpret_cast<uintptr_t>(p) % (sizeof(T) * 2) == 0 && s0 % 2 == 0 && s1 % 2 == 0 && s2 % 2 == 0;
}

}  // namespace

extern "C" {

// Bytes of workspace kitchen_int8_attention needs (0 when the arguments are not a supported case).
size_t kitchen_int8_attention_workspace_bytes(int B, int H, int HK, int Sq, int Sk, int D) {
    if (B <= 0 || H <= 0 || HK <= 0 || Sq <= 0 || Sk <= 0 || (D != 64 && D != 128) || H % HK != 0) return 0;
    return attn_plan(B, H, HK, Sq, Sk, D).total;
}

// int8_attention(q, k, v, scale) for bf16 q [B,H,Sq,D], k, v [B,HK,Sk,D] (strides in elements, last stride 1) into o [B,H,Sq,D]
// bf16 contiguous. workspace: kitchen_int8_attention_workspace_bytes bytes, 256-byte aligned. Returns a cudaError_t (0 = launched;
// cudaErrorInvalidValue for an argument the wheel rejects: head dim, divisibility, alignment, int32 strides).
static int kitchen_int8_attention_impl(const void* q, const void* k, const void* v, void* o, void* workspace, int B, int H, int HK,
                                       int Sq, int Sk, int D, int64_t q_sb, int64_t q_sh, int64_t q_sn, int64_t k_sb, int64_t k_sh,
                                       int64_t k_sn, int64_t v_sb, int64_t v_sh, int64_t v_sn, float scale, bool o_rows,
                                       cudaStream_t stream) {
    if (o_rows && B != 1) return ERR_ARGS;
    const size_t need = kitchen_int8_attention_workspace_bytes(B, H, HK, Sq, Sk, D);
    if (need == 0 || !workspace || !q || !k || !v || !o || (reinterpret_cast<uintptr_t>(workspace) & 255)) return ERR_ARGS;
    if (!qk_aligned(q, q_sb, q_sh, q_sn, B, H, Sq) || !qk_aligned(k, k_sb, k_sh, k_sn, B, HK, Sk) ||
        !v_aligned(v, v_sb, v_sh, v_sn, B, HK, Sk))
        return ERR_ARGS;
    const AttnPlan P = attn_plan(B, H, HK, Sq, Sk, D);
    if ((int64_t)H * Sq * D > INT_MAX || (int64_t)HK * Sk * D > INT_MAX || (int64_t)HK * D * P.padded_k > INT_MAX) return ERR_ARGS;
    char* ws = static_cast<char*>(workspace);
    int8_t* q8 = reinterpret_cast<int8_t*>(ws + P.off_q8);
    int8_t* k8 = reinterpret_cast<int8_t*>(ws + P.off_k8);
    int8_t* v8 = reinterpret_cast<int8_t*>(ws + P.off_v8);
    float* qs = reinterpret_cast<float*>(ws + P.off_qs);
    float* ks = reinterpret_cast<float*>(ws + P.off_ks);
    float* vs = reinterpret_cast<float*>(ws + P.off_vs);
    int* anchor = reinterpret_cast<int*>(ws + P.off_anchor);

    // 1. the K anchor
    kitchen_quant_qk::detect_k_anchor<bf16><<<dim3(HK, B), 128, 0, stream>>>(static_cast<const bf16*>(k), anchor, Sk, D, HK,
                                                                              k_sb, k_sh, k_sn);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // 2. Q and K: rotation, scales, int8 (the Hadamard follows K's length)
    const int q_oblk = (Sq + 127) / 128 * 4, k_oblk = (Sk + P.cta_k - 1) / P.cta_k;
    const bf16* qb = static_cast<const bf16*>(q);
    const bf16* kb = static_cast<const bf16*>(k);
#define QK(NL, BK, A4, ROT) quant_qk<NL, BK, A4, ROT>(qb, q8, qs, kb, k8, ks, anchor, B, H, HK, Sq, Sk, D, q_oblk, k_oblk, q_sb, q_sh, q_sn, k_sb, k_sh, k_sn, stream)
    if (D == 64) {
        if (Sk <= 256) QK(8, 64, false, 4); else QK(8, 64, false, 64);
    } else if (P.cta_k == 128) {
        QK(16, 128, true, 128);  // cta 128 only past Sk = 1024, so never the H4 rotation
    } else {
        if (Sk <= 256) QK(8, 64, true, 4); else QK(8, 64, true, 128);
    }
#undef QK
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // 3. V: per-channel scale, int8, the MMA permutation, zero padding to padded_k
    const int blocks = B * HK * (D / 8);
    if (Sk <= 256)
        kitchen_quant_v::quant_v_int8_kernel<bf16, 128><<<blocks, 128, 0, stream>>>(static_cast<const bf16*>(v), v8, vs, Sk,
                                                                                   P.padded_k, HK, D, v_sb, v_sh, v_sn);
    else
        kitchen_quant_v::quant_v_int8_kernel<bf16, 512><<<blocks, 512, 0, stream>>>(static_cast<const bf16*>(v), v8, vs, Sk,
                                                                                   P.padded_k, HK, D, v_sb, v_sh, v_sn);
    e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;

    // 4. attention
    bf16* ob = static_cast<bf16*>(o);
    return D == 64 ? attn_dispatch<64>(P.cta_k, Sk, q8, k8, v8, ob, qs, ks, vs, Sq, H, H / HK, B, D, P.padded_k, scale, o_rows, stream)
                   : attn_dispatch<128>(P.cta_k, Sk, q8, k8, v8, ob, qs, ks, vs, Sq, H, H / HK, B, D, P.padded_k, scale, o_rows, stream);
}

int kitchen_int8_attention(const void* q, const void* k, const void* v, void* o, void* workspace, int B, int H, int HK, int Sq,
                           int Sk, int D, int64_t q_sb, int64_t q_sh, int64_t q_sn, int64_t k_sb, int64_t k_sh, int64_t k_sn,
                           int64_t v_sb, int64_t v_sh, int64_t v_sn, float scale, cudaStream_t stream) {
    return kitchen_int8_attention_impl(q, k, v, o, workspace, B, H, HK, Sq, Sk, D, q_sb, q_sh, q_sn, k_sb, k_sh, k_sn, v_sb, v_sh,
                                       v_sn, scale, false, stream);
}

// Ours, not the wheel's: kitchen_int8_attention with o as bf16 rows [Sq, H * D] (B 1) instead of [B, H, Sq, D]: the same
// kernels, the same values, written where the transpose to rows would have put them (see attn_launch). The engine's
// attention output feeds the out projection as rows, so this drops the separate [H, S, D] -> [S, H * D] pass.
int kitchen_int8_attention_rows(const void* q, const void* k, const void* v, void* o, void* workspace, int B, int H, int HK, int Sq,
                                int Sk, int D, int64_t q_sb, int64_t q_sh, int64_t q_sn, int64_t k_sb, int64_t k_sh, int64_t k_sn,
                                int64_t v_sb, int64_t v_sh, int64_t v_sn, float scale, cudaStream_t stream) {
    return kitchen_int8_attention_impl(q, k, v, o, workspace, B, H, HK, Sq, Sk, D, q_sb, q_sh, q_sn, k_sb, k_sh, k_sn, v_sb, v_sh,
                                       v_sn, scale, true, stream);
}

// rms_rope_split_half_(q, k, freqs, q_norm, k_norm, eps, rot_dim) in place, bf16 q, k [batch, dim1, dim2, head_dim] (strides in
// elements), bf16 freqs [freqs_batch, freqs_dim1, freqs_dim2, rot_dim/2, 2, 2], bf16 norm weights [head_dim] with strides
// qs_stride, ks_stride. rot_dim 0 rotates the whole head. Returns a cudaError_t.
int kitchen_rms_rope_split_half_(void* q, void* k, const void* freqs, const void* q_norm, const void* k_norm, int64_t batch,
                                 int64_t dim1, int64_t dim2, int head_dim, int rot_dim, int64_t f_batch, int64_t f_dim1,
                                 int64_t f_dim2, int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3, int64_t k_s0,
                                 int64_t k_s1, int64_t k_s2, int64_t k_s3, int64_t f_s0, int64_t f_s1, int64_t f_s2,
                                 int64_t f_s3, int64_t f_s4, int64_t f_s5, int64_t qs_stride, int64_t ks_stride, float eps,
                                 cudaStream_t stream) {
    if (rot_dim == 0) rot_dim = head_dim;
    if (batch <= 0 || dim1 <= 0 || dim2 <= 0 || head_dim < 32 || head_dim % 32 != 0 || rot_dim % 2 != 0 || rot_dim > head_dim ||
        !q || !k || !freqs || !q_norm || !k_norm)
        return ERR_ARGS;
    const int64_t rows = batch * dim1 * dim2;
    const int blocks = static_cast<int>((rows + comfy::kitchen_rms_rope::kWarpsPerBlock - 1) / comfy::kitchen_rms_rope::kWarpsPerBlock);
    const bool contig = q_s3 == 1 && k_s3 == 1 && pair_aligned(static_cast<const bf16*>(q), q_s0, q_s1, q_s2) &&
                        pair_aligned(static_cast<const bf16*>(k), k_s0, k_s1, k_s2) && head_dim % 4 == 0 && rot_dim % 4 == 0;
    const bf16* qc = static_cast<const bf16*>(q);
    const bf16* kc = static_cast<const bf16*>(k);
    const bf16* fc = static_cast<const bf16*>(freqs);
    const bf16* qn = static_cast<const bf16*>(q_norm);
    const bf16* kn = static_cast<const bf16*>(k_norm);
    bf16* qo = static_cast<bf16*>(q);
    bf16* ko = static_cast<bf16*>(k);
#define ROPE(CONTIG)                                                                                                          \
    comfy::kitchen_rms_rope::rope_kernel<bf16, bf16, bf16, true, true, true, true, CONTIG>                                      \
        <<<blocks, comfy::kitchen_rms_rope::kThreads, 0, stream>>>(                                                             \
            qc, kc, fc, qn, kn, qo, ko, batch, dim1, dim2, head_dim, rot_dim, f_batch, f_dim1, f_dim2, q_s0, q_s1, q_s2, q_s3, \
            k_s0, k_s1, k_s2, k_s3, q_s0, q_s1, q_s2, q_s3, k_s0, k_s1, k_s2, k_s3, f_s0, f_s1, f_s2, f_s3, f_s4, f_s5,        \
            qs_stride, ks_stride, eps)
    if (contig) ROPE(true); else ROPE(false);
#undef ROPE
    return (int)cudaGetLastError();
}

}  // extern "C"
