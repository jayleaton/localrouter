#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
using TF=cutlass::tfloat32_t;
using G=cutlass::gemm::device::Gemm<TF,cutlass::layout::RowMajor,TF,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<64,64,16>,cutlass::gemm::GemmShape<32,32,16>,cutlass::gemm::GemmShape<16,8,8>,cutlass::epilogue::thread::LinearCombination<float,4,float,float>,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,6,4,4>;
using K=G::GemmKernel;
static_assert(K::kThreadCount==128);
extern "C" __global__ void k6t_clip_tf32_rn(const float* x,TF* y,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned z=__float_as_uint(x[i]);
 if((z&0x7f800000)!=0x7f800000)z=(z+0xfff+((z>>13)&1))&~0x1fff;
 reinterpret_cast<unsigned*>(y)[i]=z;
}
extern "C" __global__ void __launch_bounds__(K::kThreadCount)
k6t_clip_linear(const TF* x,const TF* w,const float* bias,float* out,int m,int n,int k) {
 extern __shared__ char shared[];
 K::Params p({m,n,k},{(m+63)/64,(n+63)/64,1},{const_cast<TF*>(x),k},{const_cast<TF*>(w),k},{const_cast<float*>(bias),0},{out,n},{1.0f,1.0f});
 p.gather_A_indices=nullptr;p.gather_B_indices=nullptr;p.scatter_D_indices=nullptr;
 K kernel;kernel(p,*reinterpret_cast<K::SharedStorage*>(shared));
}
extern "C" __global__ void k6t_clip_abi(int* out){out[0]=K::kThreadCount;out[1]=sizeof(K::SharedStorage);}
// fc2's captured kernel uses two serial K partitions. Its bias is applied in partition zero;
// partition one reads the rounded first output. A single full-K accumulator is not bit-equivalent.
using SplitG = cutlass::gemm::device::Gemm<TF,cutlass::layout::RowMajor,TF,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<32,32,32>,cutlass::gemm::GemmShape<16,8,8>,cutlass::epilogue::thread::LinearCombination<float,4,float,float>,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,6,4,4,true>;
using SplitK = SplitG::GemmKernel;
static_assert(SplitK::kThreadCount == 128);
static_assert(sizeof(SplitK::SharedStorage) == 98304);
extern "C" __global__ void k6t_clip_clear_locks(int* locks) {
    if (threadIdx.x < 24) locks[threadIdx.x] = 0;
}
extern "C" __global__ void __launch_bounds__(SplitK::kThreadCount)
k6t_clip_linear_split(const TF* x,const TF* w,const float* bias,float* out,int m,int n,int k,int* locks) {
    extern __shared__ char shared[];
    SplitK::Params p({m,n,k},{(m+63)/64,(n+63)/64,2},{const_cast<TF*>(x),k},{const_cast<TF*>(w),k},{const_cast<float*>(bias),0},{out,n},{1.0f,1.0f},locks);
    p.gather_A_indices=nullptr;p.gather_B_indices=nullptr;p.scatter_D_indices=nullptr;
    SplitK kernel;kernel(p,*reinterpret_cast<SplitK::SharedStorage*>(shared));
}
