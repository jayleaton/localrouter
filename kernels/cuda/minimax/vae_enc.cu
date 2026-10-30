// H3 one-frame encoder. Activations are fp16 rows [H*W,C]; convolution weights are [O,C,kh,kw], with the last
// temporal tap retained by the loader. Every entry point launches 256 threads; GN launches one block per group.
#include <cuda_fp16.h>
#include <math.h>

static __device__ float ve_half(float x) { return __half2float(__float2half_rn(x)); }
static __device__ int ve_reflect(int x, int n) { return x < 0 ? -x : (x >= n ? 2*n-2-x : x); }

// Canvas fp32 RGB [H,W,3] -> one cropped tile [h*w,3]. Wrapper process_input precedes fp16 pixel normalization.
extern "C" __global__ void ve_pixels(const float* rgb, __half* out, int W, int y0, int x0, int h, int w) {
    long long e = (long long)blockIdx.x*256 + threadIdx.x;
    if (e >= 3LL*h*w) return;
    int c = e%3, x = (e/3)%w, y = e/(3LL*w);
    const float means[3] = {0.485f,0.456f,0.406f}, stds[3] = {0.229f,0.224f,0.225f};
    float v = ve_half(__fsub_rn(__fmul_rn(rgb[((long long)(y+y0)*W+x+x0)*3+c],2.0f),1.0f));
    v = ve_half(__fadd_rn(v,1.0f));
    v = ve_half(__fmul_rn(v,0.5f));
    v = ve_half(__fsub_rn(v,ve_half(means[c])));
    out[e] = __float2half_rn(__fdiv_rn(v,ve_half(stds[c])));
}

// Reflect padding (pad=1: symmetric; pad=0, k=3: bottom/right only), im2col in C,ky,kx order with K padded to 8.
extern "C" __global__ void ve_im2col(const __half* x, __half* col, int H, int W, int C, int k, int stride,
                                     int pad, int OH, int OW, int K) {
    long long e = (long long)blockIdx.x*256 + threadIdx.x;
    if (e >= (long long)OH*OW*K) return;
    int j = e%K, p = e/K;
    if (j >= C*k*k) { col[e] = __float2half_rn(0); return; }
    int c = j/(k*k), ky = (j/k)%k, kx = j%k;
    int y = ve_reflect((p/OW)*stride+ky-pad,H), xx = ve_reflect((p%OW)*stride+kx-pad,W);
    col[e] = x[((long long)y*W+xx)*C+c];
}

// GN32, eps=1e-6, two-pass fixed tree reduction. Half affine output, then a separate half SiLU, as the unfused path.
extern "C" __global__ void ve_group_norm(const __half* x, const __half* weight, const __half* bias, __half* y,
                                         int C, int P, int groups, float eps) {
    __shared__ float sums[256];
    __shared__ float mean, rstd;
    int t = threadIdx.x, cg = C/groups, g = blockIdx.x;
    long long n = (long long)P*cg;
    float sum = 0;
    for (long long j=t;j<n;j+=256) sum = __fadd_rn(sum,__half2float(x[(j/cg)*C+g*cg+j%cg]));
    sums[t]=sum; __syncthreads();
    for(int d=128;d;d>>=1) { if(t<d) sums[t]=__fadd_rn(sums[t],sums[t+d]); __syncthreads(); }
    if(t==0) mean=__fdiv_rn(sums[0],(float)n); __syncthreads();
    sum=0;
    for(long long j=t;j<n;j+=256) { float v=__fsub_rn(__half2float(x[(j/cg)*C+g*cg+j%cg]),mean);
        sum=__fadd_rn(sum,__fmul_rn(v,v)); }
    sums[t]=sum; __syncthreads();
    for(int d=128;d;d>>=1) { if(t<d) sums[t]=__fadd_rn(sums[t],sums[t+d]); __syncthreads(); }
    if(t==0) rstd=rsqrtf(__fadd_rn(__fdiv_rn(sums[0],(float)n),eps)); __syncthreads();
    for(long long j=t;j<n;j+=256) { int c=g*cg+j%cg; long long idx=(j/cg)*C+c;
        float v=__fmul_rn(__fsub_rn(__half2float(x[idx]),mean),rstd);
        y[idx]=__float2half_rn(fmaf(v,__half2float(weight[c]),__half2float(bias[c]))); }
}

extern "C" __global__ void ve_silu(__half* x, long long n) {
    long long e=(long long)blockIdx.x*256+threadIdx.x;
    if(e<n) { float v=__half2float(x[e]); x[e]=__float2half_rn(__fdiv_rn(v,__fadd_rn(1.0f,expf(-v)))); }
}

extern "C" __global__ void ve_add(__half* x, const __half* residual, long long n) {
    long long e=(long long)blockIdx.x*256+threadIdx.x;
    if(e<n) x[e]=__float2half_rn(__fadd_rn(__half2float(x[e]),__half2float(residual[e])));
}

// Rows of 48 moments -> CHW, viewed as [3,16,H,W] by the existing decoder's tile placement kernel.
extern "C" __global__ void ve_moments(const __half* rows, __half* chw, int P) {
    long long e=(long long)blockIdx.x*256+threadIdx.x;
    if(e<48LL*P) chw[e]=rows[(e%P)*48+e/P];
}

// The first 24 moments, fp32 subtraction/division against fp16-rounded checkpoint statistics.
extern "C" __global__ void ve_latent(const __half* moments, const __half* mean, const __half* stdv,
                                     float* latent, int P) {
    long long e=(long long)blockIdx.x*256+threadIdx.x;
    if(e<24LL*P) latent[e]=__fdiv_rn(__fsub_rn(__half2float(moments[e]),__half2float(mean[e/P])),
                                   __half2float(stdv[e/P]));
}
