// Qwen3-VL fp32 vision operations. Raw launch ABI shared with the engine; no fast math.
#include <stdint.h>
#include <math.h>
extern "C" __global__ void h3v_norm(const float* x,const float* w,const float* b,float* y,int D) {
    __shared__ float r[256]; int t=threadIdx.x; long long base=(long long)blockIdx.x*D;
    float s=0; for(int j=t;j<D;j+=256)s=__fadd_rn(s,x[base+j]); r[t]=s; __syncthreads();
    for(int k=128;k;k>>=1){if(t<k)r[t]=__fadd_rn(r[t],r[t+k]);__syncthreads();}
    float mean=__fdiv_rn(r[0],(float)D); __syncthreads(); s=0;
    for(int j=t;j<D;j+=256){float v=__fsub_rn(x[base+j],mean);s=__fadd_rn(s,__fmul_rn(v,v));}
    r[t]=s;__syncthreads();for(int k=128;k;k>>=1){if(t<k)r[t]=__fadd_rn(r[t],r[t+k]);__syncthreads();}
    float inv=rsqrtf(__fadd_rn(__fdiv_rn(r[0],(float)D),1.e-6f));
    for(int j=t;j<D;j+=256)y[base+j]=__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x[base+j],mean),inv),w[j]),b[j]);
}
extern "C" __global__ void h3v_gelu(const float* x,float* y,long long n,int approximate) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;if(i>=n)return;float v=x[i];
    float a=approximate?tanhf(__fmul_rn(0.7978845608028654f,__fadd_rn(v,__fmul_rn(0.044715f,__fmul_rn(__fmul_rn(v,v),v))))):erff(__fmul_rn(v,0.7071067811865475f));
    y[i]=__fmul_rn(__fmul_rn(v,0.5f),__fadd_rn(1.f,a));
}
extern "C" __global__ void h3v_rope(float* x,const float* cs,const float* sn,int H,int D) {
    int d=threadIdx.x;if(d>=D/2)return;long long row=blockIdx.x,base=(row*H+blockIdx.y)*D;
    float a=x[base+d],b=x[base+d+D/2],c=cs[row*(D/2)+d],s=sn[row*(D/2)+d];
    x[base+d]=__fsub_rn(__fmul_rn(a,c),__fmul_rn(b,s));x[base+d+D/2]=__fadd_rn(__fmul_rn(b,c),__fmul_rn(a,s));
}
// Round to the nearest bf16 (ties to even), kept in a float: torch's bf16 arithmetic is fp32 math rounded this way.
__device__ __forceinline__ float h3v_bf16(float x){unsigned u=__float_as_uint(x);if((u&0x7f800000u)==0x7f800000u)return x;
    u+=0x7fffu+((u>>16)&1u);return __uint_as_float(u&0xffff0000u);}
// ComfyUI's fast_pos_embed_interpolate runs in the table's dtype (bf16): the table and the four bilinear weights are bf16, each
// table row times its weight is a bf16 product, and the four products are summed left to right in bf16.
extern "C" __global__ void h3v_position(const float* table,const int* ids,const float* weights,float* out,int L,int D) {
    long long i=(long long)blockIdx.x*256+threadIdx.x;if(i>=(long long)L*D)return;int row=i/D,d=i%D;
    float s=h3v_bf16(__fmul_rn(h3v_bf16(table[(long long)ids[row]*D+d]),h3v_bf16(weights[row])));
    for(int k=1;k<4;k++)s=h3v_bf16(__fadd_rn(s,h3v_bf16(__fmul_rn(h3v_bf16(table[(long long)ids[k*L+row]*D+d]),h3v_bf16(weights[k*L+row])))));
    out[i]=s;
}
extern "C" __global__ void h3v_attention(const float* q,const float* k,const float* v,float* out,int L,int H,int D,float scale) {
    extern __shared__ float sm[];float* scores=sm;float* red=sm+L;int t=threadIdx.x,h=blockIdx.y;long long row=blockIdx.x;
    for(int j=t;j<L;j+=128){float a=0;for(int d=0;d<D;d++)a=__fmaf_rn(q[(row*H+h)*D+d],k[((long long)j*H+h)*D+d],a);scores[j]=__fmul_rn(a,scale);}
    __syncthreads();float mx=__int_as_float((int)0xFF800000u);for(int j=t;j<L;j+=128)mx=fmaxf(mx,scores[j]);red[t]=mx;__syncthreads();
    for(int s=64;s;s>>=1){if(t<s)red[t]=fmaxf(red[t],red[t+s]);__syncthreads();}mx=red[0];__syncthreads();
    float sum=0;for(int j=t;j<L;j+=128){float a=expf(__fsub_rn(scores[j],mx));scores[j]=a;sum=__fadd_rn(sum,a);}red[t]=sum;__syncthreads();
    for(int s=64;s;s>>=1){if(t<s)red[t]=__fadd_rn(red[t],red[t+s]);__syncthreads();}
    float den=red[0];if(t<D){float a=0;for(int j=0;j<L;j++)a=__fmaf_rn(scores[j],v[((long long)j*H+h)*D+t],a);out[(row*H+h)*D+t]=__fdiv_rn(a,den);}
}
