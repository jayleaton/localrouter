#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASH_NAMESPACE pytorch_flash
#include <cuda_runtime.h>
#include "flash.h"
#include "flash_fwd_kernel.h"
using K = Flash_fwd_kernel_traits<128,128,64,4,false,false,cutlass::bfloat16_t>;
static_assert(K::kSmemSize == 65536);
extern "C" __global__ void __launch_bounds__(128)
k6_visual_flash(const void* q,const void* k,const void* v,void* out,void* lse) {
    pytorch_flash::Flash_fwd_params p{};
    p.q_ptr=const_cast<void*>(q);p.k_ptr=const_cast<void*>(k);p.v_ptr=const_cast<void*>(v);p.o_ptr=out;
    p.softmax_lse_ptr=lse;
    p.q_batch_stride=p.k_batch_stride=p.v_batch_stride=p.o_batch_stride=50220LL*4096;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=4096;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.h=p.h_k=32;p.h_h_k_ratio=1;p.b=1;
    p.seqlen_q=p.seqlen_k=50220;p.seqlen_q_rounded=p.seqlen_k_rounded=50304;
    p.d=p.d_rounded=128;p.total_q=50220;
    p.scale_softmax=0.08838834764831845f;p.scale_softmax_log2=float(double(p.scale_softmax)*1.4426950408889634);
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=p.window_size_right=-1;p.is_bf16=true;p.is_seqlens_k_cumulative=true;p.num_splits=1;
    pytorch_flash::compute_attn<K,false,false,false,false,false,true,false,false>(p);
}

extern "C" __global__ void __launch_bounds__(128)
k6_audio_flash(const void* q,const void* k,const void* v,void* out,void* lse) {
    pytorch_flash::Flash_fwd_params p{};
    p.q_ptr=const_cast<void*>(q);p.k_ptr=const_cast<void*>(k);p.v_ptr=const_cast<void*>(v);p.o_ptr=out;
    p.softmax_lse_ptr=lse;
    p.q_batch_stride=p.k_batch_stride=p.v_batch_stride=p.o_batch_stride=218LL*2048;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=2048;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.h=p.h_k=16;p.h_h_k_ratio=1;p.b=1;
    p.seqlen_q=p.seqlen_k=218;p.seqlen_q_rounded=p.seqlen_k_rounded=256;
    p.d=p.d_rounded=128;p.total_q=218;
    p.scale_softmax=0.08838834764831845f;p.scale_softmax_log2=float(double(p.scale_softmax)*1.4426950408889634);
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=p.window_size_right=-1;p.is_bf16=true;p.is_seqlens_k_cumulative=true;p.num_splits=1;
    pytorch_flash::compute_attn<K,false,false,false,false,false,true,false,false>(p);
}

extern "C" __global__ void __launch_bounds__(128)
k6_va_flash(const void* q,const void* k,const void* v,void* out,void* lse) {
    pytorch_flash::Flash_fwd_params p{};
    p.q_ptr=const_cast<void*>(q);p.k_ptr=const_cast<void*>(k);p.v_ptr=const_cast<void*>(v);p.o_ptr=out;
    p.softmax_lse_ptr=lse;
    p.q_batch_stride=p.o_batch_stride=50220LL*4096;
    p.k_batch_stride=p.v_batch_stride=218LL*4096;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=4096;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.h=p.h_k=32;p.h_h_k_ratio=1;p.b=1;
    p.seqlen_q=50220;p.seqlen_k=218;p.seqlen_q_rounded=50304;p.seqlen_k_rounded=256;
    p.d=p.d_rounded=128;p.total_q=50220;
    p.scale_softmax=0.08838834764831845f;p.scale_softmax_log2=float(double(p.scale_softmax)*1.4426950408889634);
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=p.window_size_right=-1;p.is_bf16=true;p.is_seqlens_k_cumulative=true;p.num_splits=1;
    pytorch_flash::compute_attn<K,false,false,false,false,false,true,false,false>(p);
}

using AVK = Flash_fwd_kernel_traits<128,64,128,4,false,false,cutlass::bfloat16_t>;
static_assert(AVK::kSmemSize == 81920);
// LSE, three split LSEs and three split outputs fit the existing shared scratch.
static_assert((4*16*218+3*16*218*128)*sizeof(float) <= 32*50220*sizeof(float));
__device__ __forceinline__ pytorch_flash::Flash_fwd_params avParams(
        const void* q,const void* k,const void* v,void* out,void* scratch) {
    pytorch_flash::Flash_fwd_params p{};
    p.q_ptr=const_cast<void*>(q);p.k_ptr=const_cast<void*>(k);p.v_ptr=const_cast<void*>(v);p.o_ptr=out;
    p.softmax_lse_ptr=scratch;
    p.softmax_lseaccum_ptr=static_cast<float*>(scratch)+16*218;
    p.oaccum_ptr=static_cast<float*>(scratch)+4*16*218;
    p.q_batch_stride=p.o_batch_stride=218LL*2048;
    p.k_batch_stride=p.v_batch_stride=50220LL*2048;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=2048;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=128;
    p.h=p.h_k=16;p.h_h_k_ratio=1;p.b=1;
    p.seqlen_q=218;p.seqlen_k=50220;p.seqlen_q_rounded=256;p.seqlen_k_rounded=50304;
    p.d=p.d_rounded=128;p.total_q=218;
    p.scale_softmax=0.08838834764831845f;p.scale_softmax_log2=float(double(p.scale_softmax)*1.4426950408889634);
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=p.window_size_right=-1;p.is_bf16=true;p.is_seqlens_k_cumulative=true;p.num_splits=3;
    return p;
}
extern "C" __global__ void __launch_bounds__(128)
k6_av_flash(const void* q,const void* k,const void* v,void* out,void* scratch) {
    auto p=avParams(q,k,v,out,scratch);
    pytorch_flash::compute_attn_splitkv<AVK,false,false,false,false,true,false,true,false>(p);
}
extern "C" __global__ void __launch_bounds__(128)
k6_av_combine(const void* q,const void* k,const void* v,void* out,void* scratch) {
    auto p=avParams(q,k,v,out,scratch);
    pytorch_flash::combine_attn_seqk_parallel<AVK,4,2,true>(p);
}

// Audio head: captured 218x2048 / 400x2048 BF16 projection, split-K=8.
#include <cuda_runtime.h>
#include <cutlass/arch/wmma.h>
// Standalone candidate bridge to the installed CUDA BF16 WMMA primitive.
// This is not a recovered cuBLAS kernel definition; exact replay is the gate.
namespace cutlass { namespace arch {
template<class Shape_,class LayoutA_,class LayoutB_,class LayoutC_>
struct Wmma<Shape_,cutlass::bfloat16_t,LayoutA_,cutlass::bfloat16_t,LayoutB_,float,LayoutC_,OpMultiplyAdd> {
 using Shape=Shape_; using ElementA=cutlass::bfloat16_t; using ElementB=cutlass::bfloat16_t;
 using LayoutA=LayoutA_; using LayoutB=LayoutB_; using ElementC=float; using LayoutC=LayoutC_;
 using Operator=OpMultiplyAdd; using ArchTag=Sm80;
 using FragmentA=nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,Shape::kM,Shape::kN,Shape::kK,__nv_bfloat16,typename CutlassToWmmaLayout<LayoutA>::Layout>;
 using FragmentB=nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,Shape::kM,Shape::kN,Shape::kK,__nv_bfloat16,typename CutlassToWmmaLayout<LayoutB>::Layout>;
 using FragmentC=nvcuda::wmma::fragment<nvcuda::wmma::accumulator,Shape::kM,Shape::kN,Shape::kK,float>;
 CUTLASS_DEVICE void operator()(FragmentC &D,FragmentA const&A,FragmentB const&B,FragmentC const&C) const {nvcuda::wmma::mma_sync(D,A,B,C);}
};
}}
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
using BF=cutlass::bfloat16_t;
using G=cutlass::gemm::device::Gemm<BF,cutlass::layout::RowMajor,BF,cutlass::layout::ColumnMajor,BF,cutlass::layout::RowMajor,float,cutlass::arch::OpClassWmmaTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<32,32,128>,cutlass::gemm::GemmShape<16,16,128>,cutlass::gemm::GemmShape<16,16,16>,cutlass::epilogue::thread::LinearCombination<BF,8,float,float>,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,1,8,8,true>;
using GK=G::GemmKernel;
static_assert(GK::kThreadCount==128 && sizeof(GK::SharedStorage)==17408);
extern "C" __global__ void __launch_bounds__(128)
k6_audio_head_wmma(const BF*x,const BF*w,const BF*b,BF*y,int*workspace) {
 __shared__ GK::SharedStorage shared;
 GK::Params p({218,400,2048},{7,13,8},{const_cast<BF*>(x),2048},{const_cast<BF*>(w),2048},{const_cast<BF*>(b),0},{y,400},{1.f,1.f},workspace);
 GK{}(p,shared);
}
extern "C" __global__ void k6_audio_head_reset(int*workspace) {
 if(threadIdx.x<91) workspace[threadIdx.x]=0;
}
