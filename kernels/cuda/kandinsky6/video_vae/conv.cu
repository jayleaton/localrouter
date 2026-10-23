// Kandinsky-6 video VAE (HunyuanVideo causal 3D VAE, pinned upstream kandinsky-6 eac06e39 core/components/hunyuan_vae.py):
// one deterministic fp16 tensor-core implicit GEMM for every convolution and linear of the encoder and decoder, and the
// two GEMMs of the mid-block attention. Activations are channels last ([T, H, W, C], channel stride 1).
//
//   y[m, n] = epilogue( sum_k  gather(x)[m, k] * w[n, k] )      m = (to, ho, wo), n = output channel
//
// Gather (one HunyuanVideoCausalConv3d, including the nearest upsample of HunyuanVideoUpsampleCausal3D in front of it):
//   k = ((dt * kh + dy) * kw + dx) * ci + c        (tap major, channel minor: the weight is repacked to [co][kt][kh][kw][ci])
//   tu = clamp(to * st + dt - pt, 0, tu_n - 1)      replicate padding F.pad(.., mode="replicate"): time front only (pt = kt - 1),
//   yu = clamp(ho * sh + dy - ph, 0, hu_n - 1)      height and width both sides (ph = kh / 2, pw = kw / 2); clamping in the
//   xu = clamp(wo * sw + dx - pw, 0, wu_n - 1)      upsampled grid is the reference's order (interpolate, then pad)
//   source t = up_t == 2 ? (tu == 0 ? 0 : 1 + (tu - 1) / 2) : tu      first frame only spatially upsampled, the others x2 in time
//   source y = yu / up_hw, x = xu / up_hw                            F.interpolate(mode="nearest", scale 2): floor(dst * 0.5)
// A dense GEMM (linear, 1x1x1 convolution, attention) is the same call with kt = kh = kw = 1, no padding and one "row" of
// width M: x is then A[M, ci] with row stride sxw.
//
// Accumulation: fp32, mma.sync m16n8k16 (f16 x f16 -> f32), every output element accumulated over the k16 chunks of K in
// ascending order, no split-K. The block tile (BM x BN), the warp layout and the grid change only which element a warp
// computes, never the chunks or their order, so an element's bits depend on neither M, N, the grid nor the variant.
// Positions of K past K (the last chunk) are zero on both sides: the gather writes zeros, the weight rows are zero padded
// to ldw (a multiple of 8) by the repack. A dense B (V^T) must likewise be zero past K up to the next multiple of 8.
//
// Epilogue per element (each a separate torch op of the reference, each one rounding; no contraction across them):
//   bias_mode 0  h = half(acc + float(bias))                 F.linear (cuBLASLt bias epilogue, fp32 add)
//   bias_mode 1  h = half(float(half(acc)) + float(bias))    cudnn convolution then output.add_(bias) (two roundings)
//   res          h = half(float(h) + float(res))             the residual add of a resnet / attention block
//   out_f32      y = acc (float, no bias, no residual)       attention scores
// res may alias y (the same element is read then written by one thread).
//
// Requirements (checked on the host by src/engines/kandinsky6/video_vae/launch.zig): vec kernels need ci % 8 == 0, every
// x stride % 8 == 0 and x 16-byte aligned; all kernels need ldw % 8 == 0, w 16-byte aligned, ldy even when !out_f32.
#include <cuda_fp16.h>
#include <stddef.h>
#include <stdint.h>

// One by-value parameter block; src/engines/kandinsky6/video_vae/launch.zig `ConvArgs` mirrors it field for field.
struct K6vConvArgs {
    const __half* x;      // source activation, channels last
    const __half* w;      // [co][ldw] repacked weight (or dense B rows)
    const __half* bias;   // [co] or null
    const __half* res;    // residual, same layout as y, or null
    void* y;              // __half [m][ldy] (or float when out_f32)
    long long sxt, sxh, sxw;  // source strides in elements (channel stride 1)
    long long ldw;        // weight row stride, >= k, % 8 == 0
    long long ldy;        // output (and residual) row stride
    int ti, hi, wi, ci;   // source dims
    int tu, hu, wu;       // upsampled dims (the source dims when up_t == up_hw == 1)
    int up_t, up_hw;      // nearest upsample factors, 1 or 2
    int kt, kh, kw;
    int pt, ph, pw;       // replicate pads: time front, height and width each side
    int st, sh, sw;
    int to, ho, wo, co;
    int k;                // kt * kh * kw * ci
    int bias_mode;
    int out_f32;
};
static_assert(sizeof(K6vConvArgs) == 184 && offsetof(K6vConvArgs, ti) == 80 && offsetof(K6vConvArgs, out_f32) == 176,
              "launch.zig ConvArgs");

namespace k6v {

constexpr int BK = 32, LD = BK + 8;  // 80-byte smem rows: ldmatrix without bank conflicts

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void cp16z(uint32_t dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int TBM, int TBN, int WM, int WN>
struct Cfg {
    static constexpr int BM = TBM, BN = TBN, WARPS_M = WM, WARPS_N = WN, THREADS = 32 * WM * WN;
    static constexpr int TM = TBM / WM, TN = TBN / WN, MI = TM / 16, NJ = TN / 8;
    static constexpr int A_CH = TBM * (BK / 8) / THREADS, B_CH = TBN * (BK / 8) / THREADS;
    static_assert(TM % 16 == 0 && TN % 16 == 0, "whole m16 tiles, n8 tiles in pairs");
    static_assert(A_CH * THREADS == TBM * BK / 8 && B_CH * THREADS == TBN * BK / 8, "whole 16-byte chunks a thread");
};
using CfgWide = Cfg<128, 128, 2, 4>;   // co > 32: 8 warps of 64 x 32
using CfgNarrow = Cfg<128, 32, 4, 1>;  // co <= 32 (decoder conv_out, quant convs): 4 warps of 32 x 32

// Output row m -> its (to, ho, wo) once per thread and row.
struct Row {
    int t, y, x;
    bool ok;
};
__device__ __forceinline__ Row row_of(const K6vConvArgs& a, long long m, long long M) {
    Row r;
    r.ok = m < M;
    const long long mm = r.ok ? m : M - 1;
    const long long hw = (long long)a.ho * a.wo;
    r.t = (int)(mm / hw);
    const long long rem = mm - (long long)r.t * hw;
    r.y = (int)(rem / a.wo);
    r.x = (int)(rem - (long long)r.y * a.wo);
    return r;
}

// Source element offset of (row, tap); the channel is added by the caller.
__device__ __forceinline__ long long src_off(const K6vConvArgs& a, const Row& r, int tap) {
    const int khw = a.kh * a.kw;
    const int dt = tap / khw, rr = tap - dt * khw, dy = rr / a.kw, dx = rr - dy * a.kw;
    const int tu = min(max(r.t * a.st + dt - a.pt, 0), a.tu - 1);
    const int yu = min(max(r.y * a.sh + dy - a.ph, 0), a.hu - 1);
    const int xu = min(max(r.x * a.sw + dx - a.pw, 0), a.wu - 1);
    const int t = a.up_t == 2 ? (tu == 0 ? 0 : 1 + (tu - 1) / 2) : tu;
    const int y = yu / a.up_hw, x = xu / a.up_hw;
    return (long long)t * a.sxt + (long long)y * a.sxh + (long long)x * a.sxw;
}

template <class Cf, bool VEC>
__device__ __forceinline__ void load_stage(const K6vConvArgs& a, const Row (&rows)[Cf::A_CH], __half* sa, uint32_t sa_s,
                                           uint32_t sb_s, int n0, int k0) {
#pragma unroll
    for (int i = 0; i < Cf::A_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS, r = e >> 2, c = (e & 3) * 8;
        const int k = k0 + c;
        if (VEC) {
            const bool ok = rows[i].ok && k < a.k;
            const int kk = ok ? k : 0;
            const int tap = kk / a.ci, ch = kk - tap * a.ci;
            cp16z(sa_s + (r * LD + c) * 2, a.x + src_off(a, rows[i], tap) + ch, ok);
        } else {
            __half v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int kj = k + j;
                if (rows[i].ok && kj < a.k) {
                    const int tap = kj / a.ci, ch = kj - tap * a.ci;
                    v[j] = a.x[src_off(a, rows[i], tap) + ch];
                } else {
                    v[j] = __ushort_as_half(0);
                }
            }
            *reinterpret_cast<uint4*>(sa + r * LD + c) = *reinterpret_cast<const uint4*>(v);
        }
    }
#pragma unroll
    for (int i = 0; i < Cf::B_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS, r = e >> 2, c = (e & 3) * 8;
        const int k = k0 + c, n = n0 + r;
        const bool ok = n < a.co && k < a.k;
        cp16z(sb_s + (r * LD + c) * 2, a.w + (long long)(ok ? n : 0) * a.ldw + (ok ? k : 0), ok);
    }
}

__device__ __forceinline__ __half epi(const K6vConvArgs& a, float v, int col, long long ri) {
    __half h;
    if (a.bias == nullptr) {
        h = __float2half_rn(v);
    } else if (a.bias_mode == 0) {
        h = __float2half_rn(__fadd_rn(v, __half2float(a.bias[col])));
    } else {
        h = __float2half_rn(__fadd_rn(__half2float(__float2half_rn(v)), __half2float(a.bias[col])));
    }
    if (a.res != nullptr) h = __float2half_rn(__fadd_rn(__half2float(h), __half2float(a.res[ri])));
    return h;
}

template <class Cf, bool VEC>
__device__ __forceinline__ void conv(const K6vConvArgs& a) {
    __shared__ __align__(16) __half sa[2][Cf::BM * LD];
    __shared__ __align__(16) __half sb[2][Cf::BN * LD];
    const long long M = (long long)a.to * a.ho * a.wo;
    const long long m0 = (long long)blockIdx.x * Cf::BM;
    const int n0 = blockIdx.y * Cf::BN;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / Cf::WARPS_N, wn = warp % Cf::WARPS_N;
    Row rows[Cf::A_CH];
#pragma unroll
    for (int i = 0; i < Cf::A_CH; ++i) rows[i] = row_of(a, m0 + ((threadIdx.x + i * Cf::THREADS) >> 2), M);

    float acc[Cf::MI][Cf::NJ][4];
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i)
#pragma unroll
        for (int j = 0; j < Cf::NJ; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int tiles = (a.k + BK - 1) / BK;
    load_stage<Cf, VEC>(a, rows, sa[0], smem_addr(sa[0]), smem_addr(sb[0]), n0, 0);
    commit();
    const int mt = lane / 8;
    for (int t = 0; t < tiles; ++t) {
        const int s = t & 1;
        if (t + 1 < tiles) load_stage<Cf, VEC>(a, rows, sa[s ^ 1], smem_addr(sa[s ^ 1]), smem_addr(sb[s ^ 1]), n0, (t + 1) * BK);
        commit();
        wait_group<1>();
        __syncthreads();
        const uint32_t sas = smem_addr(sa[s]), sbs = smem_addr(sb[s]);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[Cf::MI][4];
#pragma unroll
            for (int i = 0; i < Cf::MI; ++i)
                ldmatrix4(af[i], sas + ((wm * Cf::TM + i * 16 + lane % 16) * LD + kk + (lane / 16) * 8) * 2);
#pragma unroll
            for (int j = 0; j < Cf::NJ; j += 2) {
                uint32_t b[4];
                ldmatrix4(b, sbs + ((wn * Cf::TN + 8 * (j + mt / 2) + lane % 8) * LD + kk + (mt % 2) * 8) * 2);
#pragma unroll
                for (int i = 0; i < Cf::MI; ++i) {
                    mma(acc[i][j], af[i], b[0], b[1]);
                    mma(acc[i][j + 1], af[i], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }

    const int g = lane / 4, q = lane % 4;
#pragma unroll
    for (int i = 0; i < Cf::MI; ++i)
#pragma unroll
        for (int j = 0; j < Cf::NJ; ++j) {
            const int col = n0 + wn * Cf::TN + j * 8 + 2 * q;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const long long row = m0 + wm * Cf::TM + i * 16 + g + 8 * h;
                if (row >= M) continue;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    if (col + e >= a.co) continue;
                    const long long ri = row * a.ldy + col + e;
                    const float v = acc[i][j][2 * h + e];
                    if (a.out_f32)
                        static_cast<float*>(a.y)[ri] = v;
                    else
                        static_cast<__half*>(a.y)[ri] = epi(a, v, col + e, ri);
                }
            }
        }
}

}  // namespace k6v

// Grid (ceil(M / 128), ceil(co / 128)), block 256, 40,960 bytes of static shared memory.
extern "C" __global__ void __launch_bounds__(256) k6v_conv_wide(const K6vConvArgs a) { k6v::conv<k6v::CfgWide, true>(a); }
// Grid (ceil(M / 128), ceil(co / 32)), block 128, 25,600 bytes.
extern "C" __global__ void __launch_bounds__(128) k6v_conv_narrow(const K6vConvArgs a) { k6v::conv<k6v::CfgNarrow, true>(a); }
// The same two with a scalar gather (ci % 8 != 0: the encoder's conv_in, ci = 3). Same chunks, same bits as a vector gather.
extern "C" __global__ void __launch_bounds__(256) k6v_conv_wide_s(const K6vConvArgs a) { k6v::conv<k6v::CfgWide, false>(a); }
extern "C" __global__ void __launch_bounds__(128) k6v_conv_narrow_s(const K6vConvArgs a) { k6v::conv<k6v::CfgNarrow, false>(a); }
