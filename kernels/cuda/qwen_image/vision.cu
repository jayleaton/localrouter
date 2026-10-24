// Vision arithmetic shared with the engine. Fixed reductions, bf16 boundaries, 256 threads.
#include <cuda_bf16.h>
#include <math.h>
#define VIS_BLOCK 256
__device__ __forceinline__ float vis_ld(const __nv_bfloat16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ __nv_bfloat16 vis_st(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float vis_bf(float x) { return __bfloat162float(vis_st(x)); }
__device__ float vis_sum(float v, float* s) {
    s[threadIdx.x] = v; __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (threadIdx.x < d) s[threadIdx.x] = __fadd_rn(s[threadIdx.x], s[threadIdx.x+d]);
        __syncthreads();
    }
    return s[0];
}
// LayerNorm has fp32 moments and affine, with one final bf16 cast (unlike text RMSNorm).
extern "C" __global__ void stk_vis_ln(const __nv_bfloat16* x, const __nv_bfloat16* w,
    const __nv_bfloat16* b, __nv_bfloat16* y, int D, float eps) {
    __shared__ float s[VIS_BLOCK];
    long long base = (long long)blockIdx.x * D;
    float v = 0;
    for (int j = threadIdx.x; j < D; j += VIS_BLOCK) v = __fadd_rn(v, vis_ld(x, base+j));
    float mean = __fdiv_rn(vis_sum(v, s), (float)D);
    __syncthreads();
    v = 0;
    for (int j = threadIdx.x; j < D; j += VIS_BLOCK) {
        float z = __fsub_rn(vis_ld(x, base+j), mean); v = __fadd_rn(v, __fmul_rn(z,z));
    }
    float inv = rsqrtf(__fadd_rn(__fdiv_rn(vis_sum(v,s), (float)D), eps));
    for (int j = threadIdx.x; j < D; j += VIS_BLOCK) {
        float z = __fmul_rn(__fsub_rn(vis_ld(x,base+j), mean), inv);
        y[base+j] = vis_st(__fadd_rn(__fmul_rn(z, vis_ld(w,j)), vis_ld(b,j)));
    }
}
// Reference interpolates bf16 embeddings using bf16 weights: each multiply/add rounds separately.
// ids/weights [4,R] are already in merge order; no arithmetic on device outside this kernel.
extern "C" __global__ void stk_vis_pos(const __nv_bfloat16* table, const int* ids,
    const __nv_bfloat16* weights, __nv_bfloat16* y, int R, int D) {
    long long i = (long long)blockIdx.x * VIS_BLOCK + threadIdx.x;
    if (i >= (long long)R * D) return;
    int r = i / D, d = i % D;
    float z = 0;
    for (int k = 0; k < 4; ++k) {
        float v = vis_bf(__fmul_rn(vis_ld(table,(long long)ids[k*R+r]*D+d),vis_ld(weights,k*R+r)));
        z = k == 0 ? v : vis_bf(__fadd_rn(z,v));
    }
    y[i] = vis_st(z);
}
// QKV [R,3,H,72] -> three [R,H,128]. Vision rotates in fp32 then casts once.
// cos/sin [R,36] comprise 18 row and 18 column frequencies; rotate_half pairs j,j+36.
extern "C" __global__ void stk_vis_qkv(const __nv_bfloat16* x, const float* c, const float* s,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v, int R, int H) {
    long long i = (long long)blockIdx.x * VIS_BLOCK + threadIdx.x;
    if (i >= (long long)R * H * 128) return;
    int j = i % 128, h = (i / 128) % H, r = i / (128*H);
    if (j >= 72) { q[i]=k[i]=v[i]=vis_st(0); return; }
    long long base = ((long long)r*3*H+h)*72;
    int half = j % 36, partner = j < 36 ? j+36 : j-36;
    float sign = j < 36 ? -1.0f : 1.0f;
    float co = c[r*36+half], si = s[r*36+half];
    q[i] = vis_st(__fadd_rn(__fmul_rn(vis_ld(x,base+j),co), __fmul_rn(sign*vis_ld(x,base+partner),si)));
    base += H*72;
    k[i] = vis_st(__fadd_rn(__fmul_rn(vis_ld(x,base+j),co), __fmul_rn(sign*vis_ld(x,base+partner),si)));
    v[i] = x[base+H*72+j];
}
extern "C" __global__ void stk_vis_unpad(const __nv_bfloat16* x, __nv_bfloat16* y, long long n) {
    long long i = (long long)blockIdx.x * VIS_BLOCK + threadIdx.x;
    if (i < n) y[i] = x[(i/72)*128+i%72];
}
// Merger nn.GELU uses erf, while the transformer blocks reuse stk_gelu_tanh.
extern "C" __global__ void stk_vis_gelu(const __nv_bfloat16* x, __nv_bfloat16* y, long long n) {
    long long i = (long long)blockIdx.x * VIS_BLOCK + threadIdx.x;
    if (i < n) {
        float z = vis_ld(x,i);
        y[i] = vis_st(__fmul_rn(__fmul_rn(0.5f,z), __fadd_rn(1.0f,erff(__fmul_rn(z,0.7071067811865475f)))));
    }
}

extern "C" __global__ void stk_vis_from_f32(const float* x, __nv_bfloat16* y, long long n) {
    long long i = (long long)blockIdx.x * VIS_BLOCK + threadIdx.x;
    if (i < n) y[i] = vis_st(x[i]);
}
