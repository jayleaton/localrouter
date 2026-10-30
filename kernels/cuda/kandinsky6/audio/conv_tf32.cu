// First residual convolution fixture: exact standalone and CUDA-layout replay.
// CUTLASS e05f953, SM80 PTX preserves its half-ULP TF32 conversion.
// Caller owns the stream, scratch and dynamic shared storage. No allocations.
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
using L=cutlass::layout::TensorNHWC;
using E=cutlass::epilogue::thread::LinearCombination<float,4,float,float>;
using K=typename cutlass::conv::kernel::DefaultConv2dFprop<float,L,float,L,float,L,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<32,32,32>,cutlass::gemm::GemmShape<16,8,8>,E,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,5,cutlass::arch::OpMultiplyAdd,cutlass::conv::IteratorAlgorithm::kOptimized,cutlass::conv::StrideSupport::kUnity,4,4>::Kernel;

extern "C" __global__ void k6a_nct_to_ntc(const float* x,float* y,int C,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<C*T){int t=i/C,c=i%C;y[i]=x[c*T+t];}}
extern "C" __global__ void k6a_kcs_to_ksc(const float* x,float* y,int C,int S,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<K*S*C){int c=i%C,s=(i/C)%S,k=i/(C*S);y[i]=x[(k*C+c)*S+s];}}
extern "C" __global__ void k6a_ntc_to_nct(const float* x,float* y,int C,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<C*T){int c=i/T,t=i%T;y[i]=x[t*C+c];}}

static_assert(K::kThreadCount == 128);
static_assert(sizeof(K::SharedStorage) == 81920);
extern "C" __global__ void __launch_bounds__(128)
k6a_first_res_conv_tf32(const float* x, const float* w, float* y) {
    extern __shared__ char shared[];
    cutlass::Tensor4DCoord is(1,1,218,2048), ws(2048,1,3,2048), os(1,1,218,2048);
    cutlass::conv::Conv2dProblemSize problem(is,ws,{0,0,1,1},{1,1},{1,1},os,cutlass::conv::Mode::kCrossCorrelation,1);
    K::Arguments args{problem,{const_cast<float*>(x),L::packed(is)},
        {const_cast<float*>(w),L::packed(ws)},{y,L::packed(os)},
        {y,L::packed(os)},{1.f,0.f}};
    K::Params p(args);
    K kernel;
    kernel(p,*reinterpret_cast<K::SharedStorage*>(shared));
}

// Proven up1 block0 first-convolution fixture: 2048 input, 1024 output channels.
extern "C" __global__ void __launch_bounds__(128)
k6a_up1_first_res_conv_tf32(const float* x, const float* w, float* y) {
    extern __shared__ char shared[];
    cutlass::Tensor4DCoord is(1,1,218,2048), ws(1024,1,3,2048), os(1,1,218,1024);
    cutlass::conv::Conv2dProblemSize problem(is,ws,{0,0,1,1},{1,1},{1,1},os,cutlass::conv::Mode::kCrossCorrelation,1);
    K::Arguments args{problem,{const_cast<float*>(x),L::packed(is)},
        {const_cast<float*>(w),L::packed(ws)},{y,L::packed(os)},
        {y,L::packed(os)},{1.f,0.f}};
    K::Params p(args);
    K kernel;
    kernel(p,*reinterpret_cast<K::SharedStorage*>(shared));
}

// Proven up1 block0 second-convolution fixture: 1024 input, 1024 output channels.
extern "C" __global__ void __launch_bounds__(128)
k6a_up1_second_res_conv_tf32(const float* x, const float* w, float* y) {
    extern __shared__ char shared[];
    cutlass::Tensor4DCoord is(1,1,218,1024), ws(1024,1,3,1024), os(1,1,218,1024);
    cutlass::conv::Conv2dProblemSize problem(is,ws,{0,0,1,1},{1,1},{1,1},os,cutlass::conv::Mode::kCrossCorrelation,1);
    K::Arguments args{problem,{const_cast<float*>(x),L::packed(is)},
        {const_cast<float*>(w),L::packed(ws)},{y,L::packed(os)},
        {y,L::packed(os)},{1.f,0.f}};
    K::Params p(args);
    K kernel;
    kernel(p,*reinterpret_cast<K::SharedStorage*>(shared));
}

// QKV 1x2048x218 fixture: explicit RN ties-even operands, immutable source copies.
#include <cutlass/gemm/device/gemm.h>
using QL=cutlass::layout::ColumnMajor;
using QE=cutlass::epilogue::thread::LinearCombination<float,1,float,float>;
using QG=cutlass::gemm::device::Gemm<float,QL,float,QL,float,QL,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,QE,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,1,1,false,cutlass::arch::OpMultiplyAdd>;
using QK=QG::GemmKernel;
static_assert(QK::kThreadCount==128);
static_assert(sizeof(QK::SharedStorage)==98304);
extern "C" __global__ void k6a_tf32_rn_copy(const float* x,float* y,long long n){long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned u=__float_as_uint(x[i]);if((u&0x7f800000u)!=0x7f800000u)u=(u+0xfffu+((u>>13)&1u))&0xffffe000u;y[i]=__uint_as_float(u);}}
extern "C" __global__ void __launch_bounds__(128) k6a_attention_qkv_tf32(const float* x,const float* w,float* y){
 extern __shared__ char shared[];
 // Column-major device adapter transposes to its underlying row-major kernel.
 QK::Params p{{6144,218,2048},{48,2,1},{const_cast<float*>(w),2048},{const_cast<float*>(x),218},{y,218},{y,218},{1.f,0.f},nullptr};
 QK kernel;kernel(p,*reinterpret_cast<QK::SharedStorage*>(shared));
}

// Exact attention output projection fixture, caller-owned RN operand copies.
extern "C" __global__ void __launch_bounds__(128) k6a_attention_proj_tf32(const float* x,const float* w,float* y){
 extern __shared__ char shared[];
 QK::Params p{{2048,218,2048},{16,2,1},{const_cast<float*>(w),2048},{const_cast<float*>(x),218},{y,218},{y,218},{1.f,0.f},nullptr};
 QK kernel;kernel(p,*reinterpret_cast<QK::SharedStorage*>(shared));
}

// Fixed 1x218x2048 diagnostic SDPA, proven against pinned Torch 5c488690.
// CUTLASS e05f953 split K=172 (12 slices), RN operand copies, warp softmax.
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cmath>
#include <math_constants.h>
using SE=cutlass::epilogue::thread::LinearCombination<float,1,float,float>;
using SG=cutlass::gemm::device::GemmSplitKParallel<float,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,SE,cutlass::epilogue::thread::Convert<float,1,float>,cutlass::reduction::thread::ReduceAdd<float,float,1>,cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,3,1,1,cutlass::arch::OpMultiplyAdd>;using GV=cutlass::gemm::device::Gemm<float,cutlass::layout::RowMajor,float,cutlass::layout::ColumnMajor,float,cutlass::layout::ColumnMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,SE,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,1,1,false,cutlass::arch::OpMultiplyAdd>;

#define WARP_SHFL_XOR(x,o,w) __shfl_xor_sync(0xffffffffu,x,o,w)
namespace {
template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. Forward supports values up to 2048; backward supports values up to 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "C10_WARPS_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architectures, but there is no guarantee this will not change for future arch.
// ROCm warp size is 64 for all currently ROCm-supported GPU architectures, but this may change for future archs.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// is_masked is a flag indicating whether SoftMax or MaskedSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// The mask should have the same shape as input, with a boolean indicate if the value is masked.
// The head_chunk_size is only used for transformer mask softmax, equals to H * D * D.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked, int WARP_SIZE_PARAM>
__device__ void softmax_warp_forward(output_t *dst, const input_t *src, int batch_size, int stride, int element_count, const bool *mask = nullptr, const int head_chunk_size = -1, bool is_transformer_mask = false)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < WARP_SIZE_PARAM) ? next_power_of_two : WARP_SIZE_PARAM;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;

    if (is_transformer_mask) {
        mask += ((first_batch * stride) / head_chunk_size) * stride + local_idx;
    } else {
        mask += idx_offset;
    }
    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -CUDART_INF_F;
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        bool is_meaningful_max = false;
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_masked) {
                int idx = it*WARP_SIZE;
                if ((idx + local_idx) < batch_element_count) {
                    if (!is_transformer_mask) {
                        idx += i*element_count;
                    }
                    if (!mask[idx]) {
                        max_value[i] = (is_meaningful_max && max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                        is_meaningful_max = true;
                    }
                }
            } else {
                max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
            }
        }
        if (is_masked) {
            if (!is_meaningful_max) {
                max_value[i] = -CUDART_INF_F;
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (!is_masked) {
                if (is_log_softmax) {
                    sum[i] += std::exp(elements[i][it] - max_value[i]);
                } else {
                    elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                    sum[i] += elements[i][it];
                }
            } else {
                int idx = it*WARP_SIZE;
                bool valid = (idx + local_idx) < batch_element_count;
                if (!is_transformer_mask) {
                    idx += i*element_count;
                }
                if (valid) {
                    if (!mask[idx]) {
                        if (is_log_softmax) {
                            sum[i] += std::exp(elements[i][it] - max_value[i]);
                        } else {
                            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                            sum[i] += elements[i][it];
                        }
                    } else {
                        if (!is_log_softmax) {
                            // Masked values are treated as -infinity, and std::exp(-infinity) is 0.
                            elements[i][it] = 0;
                        }
                    }
                } else {
                    if (!is_log_softmax) {
                        elements[i][it] = 0.;
                    }
                }
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else if (sum[i] == 0) {
                    dst[i*element_count+it*WARP_SIZE] = CUDART_NAN_F;
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
}

}

__device__ float rn_tf32(float x){unsigned u=__float_as_uint(x);if((u&0x7f800000u)!=0x7f800000u)u=(u+0xfffu+((u>>13)&1u))&0xffffe000u;return __uint_as_float(u);}
extern "C" __global__ void k6a_sdpa_prepare(const float*q,const float*k,const float*v,float*qa,float*kb,float*vc,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<446464){int t=i/2048,d=i%2048,j=d*218+t;qa[j]=rn_tf32(q[i]*scale);kb[j]=rn_tf32(k[i]*scale);vc[j]=rn_tf32(v[i]);}}
extern "C" __global__ void __launch_bounds__(128) k6a_sdpa_logits(const float*qa,const float*kb,float*work){extern __shared__ char sm[];SG::GemmKernel::Params p({218,218,2048},{2,2,12},{const_cast<float*>(qa),218},{const_cast<float*>(kb),218},{work,218},{},47524);p.gemm_k_size=172;SG::GemmKernel kernel;kernel(p,*reinterpret_cast<SG::GemmKernel::SharedStorage*>(sm));}
extern "C" __global__ void k6a_sdpa_reduce(const float*work,float*logits){SG::ReductionKernel::Params p({218,218},12,47524,{const_cast<float*>(work),218},{logits,218},{logits,218},{1.f,0.f});SG::ReductionKernel kernel;SG::ReductionKernel::SharedStorage sm;kernel(p,sm);}
extern "C" __global__ void __launch_bounds__(128) k6a_sdpa_value(const float*vc,const float*p,float*out){extern __shared__ char sm[];GV::GemmKernel::Params a({2048,218,218},{16,2,1},{const_cast<float*>(vc),218},{const_cast<float*>(p),218},{out,218},{out,218},{1.f,0.f},nullptr);GV::GemmKernel kernel;kernel(a,*reinterpret_cast<GV::GemmKernel::SharedStorage*>(sm));}
extern "C" __global__ void k6a_sdpa_output(const float*x,float*y){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<446464)y[i]=x[(i%2048)*218+i/2048];}

// _safe_softmax zeros rows containing only negative infinity. Borrowed stream only.
extern "C" __global__ void k6a_sdpa_softmax(float* dst,const float* src){
 int row=blockIdx.x*4+threadIdx.y;if(row>=218)return;
 softmax_warp_forward<float,float,float,8,false,false,32>(dst,src,218,218,218);
 bool all_neg_inf=true;for(int i=threadIdx.x;i<218;i+=32)all_neg_inf=all_neg_inf && src[row*218+i]==-CUDART_INF_F;
 if(__all_sync(0xffffffffu,all_neg_inf))for(int i=threadIdx.x;i<218;i+=32)dst[row*218+i]=0.f;
}

// Proven up1 block0 shortcut fixture (MPConv1D 1x1, 2048 to 1024, T 218): v2 standalone exact in all 892,928 bytes,
// default and profiled. Split K 704/704/640 (3 slices, the captured gemm_k_size), explicit ReduceSplitK, caller-owned
// RN operand copies and workspace. Row-major [1024][2048] weight, [2048][218] input, [1024][218] output.
using UG=cutlass::gemm::device::GemmSplitKParallel<float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<16,8,8>,SE,cutlass::epilogue::thread::Convert<float,1,float>,cutlass::reduction::thread::ReduceAdd<float,float,1>,cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,3,1,1,cutlass::arch::OpMultiplyAdd>;
static_assert(UG::GemmKernel::kThreadCount==128);
static_assert(sizeof(UG::GemmKernel::SharedStorage)==98304);
extern "C" __global__ void __launch_bounds__(128) k6a_up1_shortcut_gemm(const float*w,const float*x,float*work){extern __shared__ char sm[];UG::GemmKernel::Params p({1024,218,2048},{8,2,3},{const_cast<float*>(w),2048},{const_cast<float*>(x),218},{work,218},{},223232);p.gemm_k_size=704;UG::GemmKernel kernel;kernel(p,*reinterpret_cast<UG::GemmKernel::SharedStorage*>(sm));}
extern "C" __global__ void k6a_up1_shortcut_reduce(const float*work,float*y){UG::ReductionKernel::Params p({1024,218},3,223232,{const_cast<float*>(work),218},{y,218},{y,218},{1.f,0.f});UG::ReductionKernel kernel;UG::ReductionKernel::SharedStorage sm;kernel(p,sm);}

// Proven T436 / T3488 fprop contracts (launch.zig `tf32_contracts`), each on the tile its original ran (window-2
// isolate profiles) and proven exact on every fixture byte by the window-3 tile probe (audio-w3-candidates-v1):
//   16x10   cutlass_tensorop_s1688fprop_optimized_tf32_64x64_16x10_nhwc_align4 (64x64x16, 10 stages, smem 81920)
//   128x64  cutlass_tensorop_s1688fprop_optimized_tf32_128x64_32x3_nhwc_align4 (128x64x32, 3 stages, smem 73728)
// The K tile fixes the order partial sums are added (the optimized iterator walks taps inside each channel tile), so it
// is part of the arithmetic. Problem size, padding and dilation are kernel arguments; the host launches the grid of
// GemmIdentityThreadblockSwizzle<4> (launch.zig tf32Grid), the same grid the kernel's own Params derive.
using K16=typename cutlass::conv::kernel::DefaultConv2dFprop<float,L,float,L,float,L,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<64,64,16>,cutlass::gemm::GemmShape<32,32,16>,cutlass::gemm::GemmShape<16,8,8>,E,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,10,cutlass::arch::OpMultiplyAdd,cutlass::conv::IteratorAlgorithm::kOptimized,cutlass::conv::StrideSupport::kUnity,4,4>::Kernel;
using K128=typename cutlass::conv::kernel::DefaultConv2dFprop<float,L,float,L,float,L,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,64,32>,cutlass::gemm::GemmShape<64,32,32>,cutlass::gemm::GemmShape<16,8,8>,E,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,3,cutlass::arch::OpMultiplyAdd,cutlass::conv::IteratorAlgorithm::kOptimized,cutlass::conv::StrideSupport::kUnity,4,4>::Kernel;
static_assert(K16::kThreadCount==128);
static_assert(sizeof(K16::SharedStorage)==81920);
static_assert(K128::kThreadCount==128);
static_assert(sizeof(K128::SharedStorage)==73728);
template <typename KK>
__device__ void k6a_conv_tf32_tile(const float* x,const float* w,float* y,int ci,int co,int t,int s,int p,int d){
    extern __shared__ char shared[];
    cutlass::Tensor4DCoord is(1,1,t,ci), ws(co,1,s,ci), os(1,1,t,co);
    cutlass::conv::Conv2dProblemSize problem(is,ws,{0,0,p,p},{1,1},{1,d},os,cutlass::conv::Mode::kCrossCorrelation,1);
    typename KK::Arguments args{problem,{const_cast<float*>(x),L::packed(is)},
        {const_cast<float*>(w),L::packed(ws)},{y,L::packed(os)},
        {y,L::packed(os)},{1.f,0.f}};
    typename KK::Params pp(args);
    KK kernel;
    kernel(pp,*reinterpret_cast<typename KK::SharedStorage*>(shared));
}
extern "C" __global__ void __launch_bounds__(128) k6a_conv_tf32_16x10(const float* x,const float* w,float* y,int ci,int co,int t,int s,int p,int d){k6a_conv_tf32_tile<K16>(x,w,y,ci,co,t,s,p,d);}
extern "C" __global__ void __launch_bounds__(128) k6a_conv_tf32_128x64(const float* x,const float* w,float* y,int ci,int co,int t,int s,int p,int d){k6a_conv_tf32_tile<K128>(x,w,y,ci,co,t,s,p,d);}
// The original's separate bias add (at::native CUDAFunctor_add after the convolution): one IEEE fp32 add per element.
extern "C" __global__ void k6a_add_bias_nct(float* y,const float* b,int C,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<C*T)y[i]=__fadd_rn(y[i],b[i/T]);}
// The 32x5 family (cutlass_tensorop_s1688fprop_optimized_tf32_64x64_32x5_nhwc_align4) with runtime shapes: the proven
// T218 kernel type `K`, which the window-4 vocoder sweep also proved exact on every instance of the stage-3 (96 ch) and
// stage-4 (48 ch) kernel-7 / kernel-11 AMP convolutions (launch.zig `tf32_contracts`).
extern "C" __global__ void __launch_bounds__(128) k6a_conv_tf32_32x5(const float* x,const float* w,float* y,int ci,int co,int t,int s,int p,int d){k6a_conv_tf32_tile<K>(x,w,y,ci,co,t,s,p,d);}

// Proven up0 block0 shortcut fixture (MPConv1D 1x1, 1024 to 512, T 436; decoder op up0_first_shortcut): the window-3 /
// window-5 split-K candidate exact in all 892,928 bytes, default and profiled (audio-w5-splitk-v1). The original's
// cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4 (256x64x16, 4 stages, align 4) + one cublasLt splitKreduce, as
// CUTLASS's GEMM + ReduceSplitK: split K 352/352/320 (3 slices, the captured gemm_k_size), caller-owned RN operand copies
// and workspace. Row-major [512][1024] weight, [1024][436] input, [512][436] output.
using U0G=cutlass::gemm::device::GemmSplitKParallel<float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<256,64,16>,cutlass::gemm::GemmShape<64,64,16>,cutlass::gemm::GemmShape<16,8,8>,SE,cutlass::epilogue::thread::Convert<float,1,float>,cutlass::reduction::thread::ReduceAdd<float,float,1>,cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,4,4,4,cutlass::arch::OpMultiplyAdd>;
static_assert(U0G::GemmKernel::kThreadCount==128);
static_assert(sizeof(U0G::GemmKernel::SharedStorage)==81920);
extern "C" __global__ void __launch_bounds__(128) k6a_up0_shortcut_gemm(const float*w,const float*x,float*work){extern __shared__ char sm[];U0G::GemmKernel::Params p({512,436,1024},{2,7,3},{const_cast<float*>(w),1024},{const_cast<float*>(x),436},{work,436},{},223232);p.gemm_k_size=352;U0G::GemmKernel kernel;kernel(p,*reinterpret_cast<U0G::GemmKernel::SharedStorage*>(sm));}
extern "C" __global__ void k6a_up0_shortcut_reduce(const float*work,float*y){U0G::ReductionKernel::Params p({512,436},3,223232,{const_cast<float*>(work),436},{y,436},{y,436},{1.f,0.f});U0G::ReductionKernel kernel;U0G::ReductionKernel::SharedStorage sm;kernel(p,sm);}

// Gathered-GEMM contracts (launch.zig `gather_contracts`): vocoder convolutions whose originals ran cuDNN's private
// sm80_xmma_fprop_implicit_gemm_..._nhwckrsc_nchw_128x128x16_stage4 or CUTLASS's s1688dgrad_optimized_tf32 (ConvTranspose),
// each proven exact on every instance by the window-5 sweep (audio-w5-sweep2-v1, probe k6-audio-gather-gemm-v1) as ONE
// TF32 GEMM over the reduction axis laid out channel chunk `ck` outer, taps ascending, the chunk's channels inner; no
// operand pre-rounding (the GEMM's own conversion), no split K (except the split contracts below); the bias a separate fp32 add (k6a_add_bias_nct).
// ConvTranspose1d is phase-decomposed: output phase p = t mod u reduces over its taps s, (p + pad - s) mod u == 0.
// Split contracts (window 6: the oracle-measured orders of cuDNN's xmma indexed kernels, gather-gemm v2 exact on every
// instance): the ci / ck channel chunks go round-robin to `z` slices (chunk q to slice q mod z), each slice its own GEMM
// over its chunks in ascending q, the chunk's taps ascending or descending inside it; the slices' partial sums are added
// in the contract's reduce order (k6a_gg_accumulate, one fp32 add each). z 1 is the original single-slice layout.
// Flat contracts (window 7: cuDNN's xmma indexed_wo_smem 128x32x64 alignc4, oracle2 measured, H1 predicting every
// query, gather-gemm v2 exact on all six): the reduction runs over the flat KRSC filter index f = tap * ci + c in k16
// blocks, each block two MMAs, f mod 4 in {0, 1} then f mod 4 in {2, 3}: column j holds f = 16 (j / 16) + 4 ((j % 8) / 2)
// + 2 ((j % 16) / 8) + j % 2, and a column with f >= ci * nt is zero (the last block's padding; zeros add nothing).
// The GEMM's per-output reduction does not depend on M, so the rows of a phase may be processed in any chunks.
// GG is the probe's type exactly; its SplitKSerial is inert here (grid k 1: no semaphore, beta stays 0).
using GG=cutlass::gemm::device::Gemm<float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<128,128,16>,cutlass::gemm::GemmShape<64,64,16>,cutlass::gemm::GemmShape<16,8,8>,SE,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,4,1,1,true,cutlass::arch::OpMultiplyAdd>;
static_assert(GG::GemmKernel::kThreadCount==128);
static_assert(sizeof(GG::GemmKernel::SharedStorage)==65536);
// Slice s's reduction index j -> (channel c, tap index ti): its chunks q = s, s + z, ... of ck channels x nt taps (taps
// descending when desc), then (z 1 only: launch.zig refuses a split with ci % ck != 0) the last ci % ck channels x nt taps.
__device__ __forceinline__ bool k6a_gg_split(long long j,int ci,int nt,int ck,int z,int s,int desc,int flat,int& c,int& ti){
 if(flat){const long long f=16*(j/16)+4*((j%8)/2)+2*((j%16)/8)+j%2;c=(int)(f%ci);ti=(int)(f/ci);return f<(long long)ci*nt;}
 const long long full=(long long)(ci/ck)*ck*nt;
 if(j<full){const long long span=(long long)ck*nt,q=j/span,r=j%span;ti=(int)(r/ck);c=(int)((s+q*z)*ck+r%ck);}
 else{const int last=ci-(ci/ck)*ck;const long long r=j-full;ti=(int)(r/last);c=(ci/ck)*ck+(int)(r%last);}
 if(desc)ti=nt-1-ti;
 return true;
}
// A [mc][kk] (slice sl): rows r0 .. r0 + mc of output phase ph (fprop: u 1, ph 0), tap s = s0 + ti * sstep; out-of-range taps 0.
// kk: the slice's reduction length (ci * nt for z 1).
extern "C" __global__ void k6a_gg_gather_a(const float* x,float* a,int ci,long long tin,int nt,int s0,int sstep,int pad,int dil,int u,int ph,int ck,long long r0,long long mc,int dgrad,long long kk,int z,int sl,int desc,int flat){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=mc*kk)return;
 const long long r=i/kk;int c,ti;const bool el=k6a_gg_split(i%kk,ci,nt,ck,z,sl,desc,flat,c,ti);const long long s=s0+(long long)ti*sstep;
 long long src;bool ok;
 if(dgrad){const long long n=ph+(r0+r)*u+pad-s;ok=n>=0&&n/u<tin;src=n/u;}else{src=r0+r-pad+s*dil;ok=src>=0&&src<tin;}
 a[i]=el&&ok?x[(long long)c*tin+src]:0.f;}
// B [kk][co] (slice sl) from the PyTorch weight: fprop [co][ci][taps], dgrad (ConvTranspose1d) [ci][co][taps].
extern "C" __global__ void k6a_gg_gather_b(const float* w,float* b,int ci,int co,int taps,int nt,int s0,int sstep,int ck,int dgrad,long long kk,int z,int sl,int desc,int flat){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=kk*co)return;
 const int n=(int)(i%co);int c,ti;const bool el=k6a_gg_split(i/co,ci,nt,ck,z,sl,desc,flat,c,ti);const long long s=s0+(long long)ti*sstep;
 b[i]=!el?0.f:dgrad?w[((long long)c*co+n)*taps+s]:w[((long long)n*ci+c)*taps+s];}
extern "C" __global__ void __launch_bounds__(128) k6a_gg_gemm(const float* a,const float* b,float* d,int m,int n,int k){
 extern __shared__ char sm[];
 GG::GemmKernel::Params p({m,n,k},{(m+127)/128,(n+127)/128,1},{const_cast<float*>(a),k},{const_cast<float*>(b),n},{d,n},{d,n},{1.f,0.f},nullptr);
 GG::GemmKernel kernel;kernel(p,*reinterpret_cast<GG::GemmKernel::SharedStorage*>(sm));}
// A later slice's partial sum into the running one: acc = acc + p, one IEEE fp32 add (gather-gemm v2's reduction).
extern "C" __global__ void k6a_gg_accumulate(float* acc,const float* p,long long n){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)acc[i]=__fadd_rn(acc[i],p[i]);}
// D [mc][co] -> y [co][tout] at t = ph + (r0 + r) * u.
extern "C" __global__ void k6a_gg_scatter(const float* d,float* y,int co,long long tout,int u,int ph,long long r0,long long mc){
 const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=mc*co)return;
 const long long r=i/co;const int n=(int)(i%co);y[(long long)n*tout+ph+(r0+r)*u]=d[i];}
