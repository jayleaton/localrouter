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
// ascending order, no split-K (but the _k2 kernels, below). `ctile` (a conv rule, launch.zig `qualified.conv`; default 0) only renames the order of K: with ctile = 0 the
// sequence is the tap-major one above; with ctile = T (ci % T == 0, T % 16 == 0) it is channel tiles of T outer, taps
// inner, the T channels of a tap ascending (CUTLASS's optimized fprop iterator: filter s, r, t advance inside a
// channel tile of the threadblock's K):
//   k -> tile = k / (taps * T), r = k % (taps * T), tap = r / T, c = tile * T + r % T
// The weight rows keep the tap-major layout; an element's weight is w[n][tap * ci + c] in both orders.
// The _k2 kernels (a conv rule's split 2) add cudnn's split-K of two slices: channel tile j of the sequence joins
// slice j % 2 (xmma / CUTLASS conv split-K: slice s starts at tile s and strides by two tiles), each slice its own fp32
// accumulator from the first chunk, and the slices meet in fp32 before the epilogue: acc = acc0 + acc1 (one __fadd_rn;
// the split-K reduction of the reference's grid-z = 2 fprop kernel). A BK stage lies in one channel tile (ctile % 32 == 0
// is required with the split), so a stage feeds one slice. With split_by 1 the slices alternate by chunk instead: chunk
// j of the sequence (16 k) joins slice j % 2, any ctile (two accumulator chains interleaved, as a register-only kernel
// keeps its mma latency hidden); a BK = 32 stage holds chunks 2t and 2t + 1, so kk = 0 feeds slice 0 and kk = 16 slice 1. The block tile (BM x BN), the warp layout and the grid change only which element a warp
// computes, never the chunks or their order, so an element's bits depend on neither M, N, the grid nor the variant.
// `kslot` 1 (a conv rule's kslot quad32) changes which channels share a chunk, not the chunk order: the two
// chunks of a 32-channel block of one tap (a BK stage; ci or ctile a multiple of 32) take alternate 4-channel quads,
// chunk 2t: channels {0-3, 8-11, 16-19, 24-27} of the block, chunk 2t + 1: {4-7, 12-15, 20-23, 28-31} (the decoder
// conv_out's cudnn kernel, indexed_wo_smem ... alignc2, as measured on the device: stk_twin.kandinsky6.vae_conv_oracle).
// The gather writes each 8-channel vector as two 4-channel halves: half h of vector v (channels 8v + 4h ..) to smem
// columns 16h + 4v, for x and w alike, so the mma's k16 slices see those sets (the slot order inside one mma does not
// change its bits; only the set does).
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
    int ctile;            // 0: tap-major K; else channel tiles of ctile outer, taps inner (see above)
    int split_by;         // _k2 kernels: 0 slices alternate by channel tile, 1 by chunk
    int kslot;            // 0: a chunk is 16 consecutive channels of the sequence; 1: quad32 (see above)
};
static_assert(sizeof(K6vConvArgs) == 192 && offsetof(K6vConvArgs, ti) == 80 && offsetof(K6vConvArgs, out_f32) == 176 &&
                  offsetof(K6vConvArgs, ctile) == 180 && offsetof(K6vConvArgs, split_by) == 184 &&
                  offsetof(K6vConvArgs, kslot) == 188,
              "launch.zig ConvArgs");

namespace k6v {

constexpr int BK = 32, LD = BK + 8;  // 80-byte smem rows: ldmatrix without bank conflicts

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void cp16z(uint32_t dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void cp8z(uint32_t dst, const void* src, bool read) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(read ? 8 : 0));
}
// The smem column of half h (4 channels) of the 8-channel vector at stage column c (kslot 1: quad32; else c + 4h).
__device__ __forceinline__ int kslot_col(int kslot, int c, int h) { return kslot == 1 ? 16 * h + c / 2 : c + 4 * h; }
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

// Position k of the K sequence -> (tap, channel); the weight element is then w[n][tap * ci + channel].
struct Tc {
    int tap, c;
};
__device__ __forceinline__ Tc tap_ch(const K6vConvArgs& a, int k) {
    Tc r;
    if (a.ctile == 0) {
        r.tap = k / a.ci;
        r.c = k - r.tap * a.ci;
    } else {
        const int span = (a.kt * a.kh * a.kw) * a.ctile, tile = k / span, rem = k - tile * span;
        r.tap = rem / a.ctile;
        r.c = tile * a.ctile + (rem - r.tap * a.ctile);
    }
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
            const Tc tc = tap_ch(a, ok ? k : 0);
            const __half* src = a.x + src_off(a, rows[i], tc.tap) + tc.c;
            if (a.kslot == 0) {
                cp16z(sa_s + (r * LD + c) * 2, src, ok);
            } else {
                cp8z(sa_s + (r * LD + kslot_col(1, c, 0)) * 2, src, ok);
                cp8z(sa_s + (r * LD + kslot_col(1, c, 1)) * 2, src + 4, ok);
            }
        } else {
            __half v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int kj = k + j;
                if (rows[i].ok && kj < a.k) {
                    const Tc tc = tap_ch(a, kj);
                    v[j] = a.x[src_off(a, rows[i], tc.tap) + tc.c];
                } else {
                    v[j] = __ushort_as_half(0);
                }
            }
            if (a.kslot == 0) {
                *reinterpret_cast<uint4*>(sa + r * LD + c) = *reinterpret_cast<const uint4*>(v);
            } else {
                union {
                    __half h[8];
                    uint2 u[2];
                } hv;
#pragma unroll
                for (int j = 0; j < 8; ++j) hv.h[j] = v[j];
                *reinterpret_cast<uint2*>(sa + r * LD + kslot_col(1, c, 0)) = hv.u[0];
                *reinterpret_cast<uint2*>(sa + r * LD + kslot_col(1, c, 1)) = hv.u[1];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Cf::B_CH; ++i) {
        const int e = threadIdx.x + i * Cf::THREADS, r = e >> 2, c = (e & 3) * 8;
        const int k = k0 + c, n = n0 + r;
        const bool ok = n < a.co && k < a.k;
        long long off = ok ? k : 0;  // the weight row is tap-major: k itself for ctile 0
        if (a.ctile != 0) {
            const Tc tc = tap_ch(a, (int)off);
            off = (long long)tc.tap * a.ci + tc.c;
        }
        const __half* src = a.w + (long long)(ok ? n : 0) * a.ldw + off;
        if (a.kslot == 0) {
            cp16z(sb_s + (r * LD + c) * 2, src, ok);
        } else {
            cp8z(sb_s + (r * LD + kslot_col(1, c, 0)) * 2, src, ok);
            cp8z(sb_s + (r * LD + kslot_col(1, c, 1)) * 2, src + 4, ok);
        }
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

template <class Cf, bool VEC, int SPLIT>
__device__ __forceinline__ void conv(const K6vConvArgs& a) {
    static_assert(SPLIT == 1 || SPLIT == 2, "one or two split-K slices");
    __shared__ __align__(16) __half sa[2][Cf::BM * LD];
    __shared__ __align__(16) __half sb[2][Cf::BN * LD];
    const long long M = (long long)a.to * a.ho * a.wo;
    const long long m0 = (long long)blockIdx.x * Cf::BM;
    const int n0 = blockIdx.y * Cf::BN;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / Cf::WARPS_N, wn = warp % Cf::WARPS_N;
    Row rows[Cf::A_CH];
#pragma unroll
    for (int i = 0; i < Cf::A_CH; ++i) rows[i] = row_of(a, m0 + ((threadIdx.x + i * Cf::THREADS) >> 2), M);

    float acc[SPLIT][Cf::MI][Cf::NJ][4];
#pragma unroll
    for (int sp = 0; sp < SPLIT; ++sp)
#pragma unroll
        for (int i = 0; i < Cf::MI; ++i)
#pragma unroll
            for (int j = 0; j < Cf::NJ; ++j) acc[sp][i][j][0] = acc[sp][i][j][1] = acc[sp][i][j][2] = acc[sp][i][j][3] = 0.0f;
    const long long tile_span = (long long)(a.kt * a.kh * a.kw) * (a.ctile > 0 ? a.ctile : a.ci);

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
        const int tile_slice = SPLIT == 1 ? 0 : (int)(((long long)t * BK / tile_span) % SPLIT);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            const int slice = SPLIT == 1 ? 0 : (a.split_by == 1 ? (kk / 16) % SPLIT : tile_slice);
            uint32_t af[Cf::MI][4];
#pragma unroll
            for (int i = 0; i < Cf::MI; ++i)
                ldmatrix4(af[i], sas + ((wm * Cf::TM + i * 16 + lane % 16) * LD + kk + (lane / 16) * 8) * 2);
#pragma unroll
            for (int j = 0; j < Cf::NJ; j += 2) {
                uint32_t b[4];
                ldmatrix4(b, sbs + ((wn * Cf::TN + 8 * (j + mt / 2) + lane % 8) * LD + kk + (mt % 2) * 8) * 2);
#pragma unroll
                for (int sp = 0; sp < SPLIT; ++sp) {
                    if (sp != slice) continue;  // static indices: the accumulators stay in registers
#pragma unroll
                    for (int i = 0; i < Cf::MI; ++i) {
                        mma(acc[sp][i][j], af[i], b[0], b[1]);
                        mma(acc[sp][i][j + 1], af[i], b[2], b[3]);
                    }
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
                    const float v = SPLIT == 1 ? acc[0][i][j][2 * h + e] : __fadd_rn(acc[0][i][j][2 * h + e], acc[SPLIT - 1][i][j][2 * h + e]);
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
extern "C" __global__ void __launch_bounds__(256) k6v_conv_wide(const K6vConvArgs a) { k6v::conv<k6v::CfgWide, true, 1>(a); }
// Grid (ceil(M / 128), ceil(co / 32)), block 128, 25,600 bytes.
extern "C" __global__ void __launch_bounds__(128) k6v_conv_narrow(const K6vConvArgs a) { k6v::conv<k6v::CfgNarrow, true, 1>(a); }
// The same two with a scalar gather (ci % 8 != 0: the encoder's conv_in, ci = 3). Same chunks, same bits as a vector gather.
extern "C" __global__ void __launch_bounds__(256) k6v_conv_wide_s(const K6vConvArgs a) { k6v::conv<k6v::CfgWide, false, 1>(a); }
extern "C" __global__ void __launch_bounds__(128) k6v_conv_narrow_s(const K6vConvArgs a) { k6v::conv<k6v::CfgNarrow, false, 1>(a); }
// The vector-gather kernels with two split-K slices (a conv rule's split 2; ctile a multiple of 32): same grids.
extern "C" __global__ void __launch_bounds__(256) k6v_conv_wide_k2(const K6vConvArgs a) { k6v::conv<k6v::CfgWide, true, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) k6v_conv_narrow_k2(const K6vConvArgs a) { k6v::conv<k6v::CfgNarrow, true, 2>(a); }

// ---- K-order candidates (diagnostic: the convolution fixture) ---------------------------------------------------------
// One convolution of the fixture under a candidate summation schedule, for rows [m0, m0 + mcount): y = half(sum), no bias
// (the reference's cudnn output before output.add_(bias)). One warp a 16 x 8 output tile, its operands read straight from
// global memory into mma fragments (slow, exact). The schedule is a sequence of k16 chunks:
//   order 0  channel tiles of `ctile` outer, taps inner (tap = (dt * kh + dy) * kw + dx ascending), 16 channels a chunk
//            ascending; ctile = ci is the tap-major order
//   order 1  k = c * taps + tap (KCTRS, the weight's natural order), 16 consecutive k a chunk (zero past K)
//   order 2  an explicit table (a measured order, stk_twin.kandinsky6.vae_conv_oracle): spec[j * 16 + q] is the element
//            tap * ci + c of chunk j's slot q, or -1 (zero); spec_slice[j] (when given) is chunk j's accumulator chain,
//            and the chains are then joined per split_join over `split` chains; spec_chunks chunks
// mma_k 16: one m16n8k16 a chunk; 8: two m16n8k8, k 0..7 then 8..15. split > 1 (at most 16): chunk j joins slice
//   split_mode 0: (channel tile of j) % split (CUTLASS conv split-K: slice s starts at tile s, strides by split tiles)
//   split_mode 1: j / split_len (contiguous ranges: GEMM split-K, slice s = chunks [s * split_len, ..))
//   split_mode 2: (j % split_len) / (split_len / split) (K split across the warps of a threadblock, kPartitionsK:
//                 split_len chunks a threadblock k-iteration, each warp a contiguous share of every iteration)
//   split_mode 3: j % split (chunk-interleaved accumulators: consecutive mma chains alternate, as a register-only
//                 kernel does to hide the mma latency)
//   split_mode 4: (j / split_len) % split (k-tile ping-pong: split_len chunks a k-tile, consecutive tiles alternate)
//   split_mode 5: per k-tile partial: each group of split_len chunks summed from 0 in its own chain, then added to the
//                 running total, total = total + partial (fp32); y = half(total); `split` and `split_join` unused
// perm: which channel feeds mma k slot q (both operands alike, so the product set is the same; only the tensor core's
//   internal order and which chunk a channel falls in change): 0 identity; 1 "quad" within a chunk, slot 2t + i ->
//   channel 4t + i, slot 8 + 2t + i -> channel 4t + 2 + i. 2..4 across a 64-channel k-tile (four chunks s = j % 4 of one
//   tap; ctile % 64 == 0), with t = (q % 8) / 2, h = q / 8, i = q % 2, the channel within the 64:
//   2: 16t + 4s + 2h + i (a lane's 16 consecutive channels, a quad a chunk)   3: 16t + 8h + 2s + i (a pair a chunk)
//   4: 32h + 8s + 2t + i (slot halves from the two 32-channel halves)
//   5: quad32, across the two chunks s = j % 2 of a 32-channel block of one tap (ctile % 32 == 0): slot q -> channel
//   8 (q / 4) + 4 s + q % 4 of the block (alternate 4-channel quads; the production kernels' kslot 1)
// each slice its own fp32 accumulator from 0; then split_join 0: v = 0 + acc0 + acc1 + .. (fp32, ReduceSplitK),
// y = half(v); 1: h = half(acc0), h = half(acc_s + float(h)) (serial split-K through an fp16 output); 2: a pairwise fp32
// tree, v = (acc0 + acc1) + (acc2 + acc3) .. (power-of-two halving over the slices, zero-padded).
struct K6vConvOrderArgs {
    const __half* x;  // dense channels last [ti][hi][wi][ci]
    const __half* w;  // torch order [co][ci][kt][kh][kw]
    __half* y;        // [mcount][co]
    long long m0, mcount;
    int ti, hi, wi, ci;
    int tu, hu, wu, up_t, up_hw;
    int kt, kh, kw, pt, ph, pw;
    int to, ho, wo, co;
    int order, ctile, mma_k, split, split_mode, split_len, split_join;
    int perm, pad_;
    const int* spec;        // order 2: [spec_chunks][16] element ids
    const int* spec_slice;  // order 2: [spec_chunks] chain ids, or null (one chain)
    int spec_chunks, pad2_;
};
static_assert(sizeof(K6vConvOrderArgs) == 176, "launch.zig ConvOrderArgs");

namespace k6v {

__device__ __forceinline__ void mma8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ uint32_t pack2(__half lo, __half hi) {
    return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}

// Element at mma k slot q (0..15) of chunk j: its (tap, channel), or tap < 0 past K.
__device__ __forceinline__ Tc order_elem(const K6vConvOrderArgs& a, int taps, int j, int q) {
    if (a.perm == 1) {
        q = q < 8 ? 4 * (q >> 1) + (q & 1) : 4 * ((q - 8) >> 1) + 2 + (q & 1);
    } else if (a.perm == 5) {
        const int c = 8 * (q >> 2) + 4 * (j & 1) + (q & 3);
        j = (j & ~1) + c / 16;
        q = c % 16;
    } else if (a.perm >= 2) {
        const int s4 = j & 3, t = (q & 7) >> 1, h = q >> 3, i = q & 1;
        const int c = a.perm == 2 ? 16 * t + 4 * s4 + 2 * h + i : a.perm == 3 ? 16 * t + 8 * h + 2 * s4 + i : 32 * h + 8 * s4 + 2 * t + i;
        j = (j & ~3) + c / 16;
        q = c % 16;
    }
    Tc r;
    if (a.order == 2) {
        const int e = a.spec[j * 16 + q];
        r.tap = e < 0 ? -1 : e / a.ci;
        r.c = e < 0 ? 0 : e - r.tap * a.ci;
        return r;
    }
    if (a.order == 0) {
        const int per = a.ctile / 16, span = taps * per, tile = j / span, rem = j - tile * span;
        r.tap = rem / per;
        r.c = tile * a.ctile + (rem - r.tap * per) * 16 + q;
    } else {
        const int k = j * 16 + q;
        r.tap = k < a.ci * taps ? k % taps : -1;
        r.c = k / taps;
    }
    return r;
}

__device__ __forceinline__ __half order_x(const K6vConvOrderArgs& a, long long m, Tc e) {
    if (e.tap < 0 || m >= a.m0 + a.mcount) return __ushort_as_half(0);
    const long long hw = (long long)a.ho * a.wo;
    const int to = (int)(m / hw), rem = (int)(m - (long long)to * hw), ho = rem / a.wo, wo = rem - ho * a.wo;
    const int khw = a.kh * a.kw, dt = e.tap / khw, rr = e.tap - dt * khw, dy = rr / a.kw, dx = rr - dy * a.kw;
    const int tu = min(max(to + dt - a.pt, 0), a.tu - 1);
    const int yu = min(max(ho + dy - a.ph, 0), a.hu - 1);
    const int xu = min(max(wo + dx - a.pw, 0), a.wu - 1);
    const int t = a.up_t == 2 ? (tu == 0 ? 0 : 1 + (tu - 1) / 2) : tu;
    const int y = yu / a.up_hw, x = xu / a.up_hw;
    return a.x[(((long long)t * a.hi + y) * a.wi + x) * a.ci + e.c];
}

__device__ __forceinline__ __half order_w(const K6vConvOrderArgs& a, int taps, int n, Tc e) {
    if (e.tap < 0 || n >= a.co) return __ushort_as_half(0);
    return a.w[((long long)n * a.ci + e.c) * taps + e.tap];
}

}  // namespace k6v

// Grid (ceil(mcount / 64), ceil(co / 8)), block 128: warp w of block b computes rows m0 + (4 b + w) * 16 .. + 15.
extern "C" __global__ void __launch_bounds__(128) k6v_conv_order(const K6vConvOrderArgs a) {
    using namespace k6v;
    const int lane = threadIdx.x % 32, g = lane / 4, t2 = (lane % 4) * 2;
    const long long mb = a.m0 + ((long long)blockIdx.x * 4 + threadIdx.x / 32) * 16;
    if (mb >= a.m0 + a.mcount) return;
    const int nb = blockIdx.y * 8, taps = a.kt * a.kh * a.kw;
    const int chunks = a.order == 2 ? a.spec_chunks : a.order == 0 ? (a.ci / a.ctile) * taps * (a.ctile / 16) : (a.ci * taps + 15) / 16;
    float acc[16][4];
#pragma unroll
    for (int s = 0; s < 16; ++s) acc[s][0] = acc[s][1] = acc[s][2] = acc[s][3] = 0.0f;
    for (int j = 0; j < chunks; ++j) {
        int s = 0;
        if (a.split > 1 && a.order != 2) {
            if (a.split_mode == 0)
                s = (j / (taps * (a.ctile / 16))) % a.split;
            else if (a.split_mode == 1)
                s = min(j / a.split_len, a.split - 1);
            else if (a.split_mode == 2)
                s = (j % a.split_len) / (a.split_len / a.split);
            else if (a.split_mode == 3)
                s = j % a.split;
            else if (a.split_mode == 4)
                s = (j / a.split_len) % a.split;
        }
        if (a.order == 2 && a.spec_slice != nullptr) s = a.spec_slice[j];
        if (a.split_mode == 5) {
            s = 1;
            if (j % a.split_len == 0) acc[1][0] = acc[1][1] = acc[1][2] = acc[1][3] = 0.0f;
        }
        // A rows g, g + 8 and B column g at k = t2, t2 + 1 (half 0) and t2 + 8, t2 + 9 (half 1)
        uint32_t ar[2][2], br[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const Tc e0 = order_elem(a, taps, j, t2 + 8 * h), e1 = order_elem(a, taps, j, t2 + 8 * h + 1);
            ar[h][0] = pack2(order_x(a, mb + g, e0), order_x(a, mb + g, e1));
            ar[h][1] = pack2(order_x(a, mb + g + 8, e0), order_x(a, mb + g + 8, e1));
            br[h] = pack2(order_w(a, taps, nb + g, e0), order_w(a, taps, nb + g, e1));
        }
        float(&d)[4] = acc[s];
        if (a.mma_k == 16) {
            const uint32_t af[4] = {ar[0][0], ar[0][1], ar[1][0], ar[1][1]};
            mma(d, af, br[0], br[1]);
        } else {
            mma8(d, ar[0][0], ar[0][1], br[0]);
            mma8(d, ar[1][0], ar[1][1], br[1]);
        }
        if (a.split_mode == 5 && (j % a.split_len == a.split_len - 1 || j == chunks - 1)) {
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[0][r] = __fadd_rn(acc[0][r], acc[1][r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const long long m = mb + g + (r >= 2 ? 8 : 0);
        const int n = nb + t2 + (r & 1);
        if (m >= a.m0 + a.mcount || n >= a.co) continue;
        __half h;
        if (a.split <= 1 || a.split_mode == 5) {
            h = __float2half_rn(acc[0][r]);
        } else if (a.split_join == 0) {
            float v = 0.0f;
            for (int s = 0; s < a.split; ++s) v = __fadd_rn(v, acc[s][r]);
            h = __float2half_rn(v);
        } else if (a.split_join == 2) {
            float t[16];
            int n = 1;
            while (n < a.split) n *= 2;
            for (int s = 0; s < n; ++s) t[s] = s < a.split ? acc[s][r] : 0.0f;
            for (; n > 1; n /= 2)
                for (int s = 0; s < n / 2; ++s) t[s] = __fadd_rn(t[2 * s], t[2 * s + 1]);
            h = __float2half_rn(t[0]);
        } else {
            h = __float2half_rn(acc[0][r]);
            for (int s = 1; s < a.split; ++s) h = __float2half_rn(__fadd_rn(acc[s][r], __half2float(h)));
        }
        a.y[(m - a.m0) * a.co + n] = h;
    }
}
