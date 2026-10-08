// Ours: the Qwen-Image 2.1 VAE decoder's non-GEMM kernels, shared by the Python twin (torch extension) and the Zig
// engine (fatbin, looked up by name). The decoder's convolutions run as im2col + stk_gemm_bf16 (gemm.cu); everything
// else is here. Single image (batch 1, one frame), CHW bf16 in global memory, fp32 inside, bf16 at the edges.
// No fast-math, no atomics; every reduction runs in a fixed order. Build: nvcc -O3 -gencode=arch=compute_120a,code=sm_120a
// (and 121a), no -use_fast_math. Pure data moves are exact; the ops that round spell out every rounding torch performs.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef STK_BLOCK
#define STK_BLOCK 256
#endif

#ifndef STK_BF16_TYPEDEF
#define STK_BF16_TYPEDEF
typedef __nv_bfloat16 bf16;
#endif

__device__ __forceinline__ float vldf(const bf16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ bf16 vstf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float bfround(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// cols[p - p0, k] for output pixels p in [p0, p0 + np) of a conv over x [C, H, W] (optionally read through a virtual
// nearest x`up` upsample: input pixel (y, x) is x[y / up, x / up], so the conv sees an (H*up) x (W*up) image).
// Output size: Ho = (H*up + pad_t + pad_b - kh) / stride + 1, Wo likewise; p = oy * Wo + ox.
// k = c*kh*kw + ky*kw + kx (a PyTorch weight [Cout, Cin, kh, kw] flattened row-major is the matching A/B operand);
// k in [C*kh*kw, Kpad) is zero (the weight is zero-padded the same way). Out-of-image taps are zero.
// launch: block 256, grid ceil(np * Kpad / 256) (1-D, 64-bit index).
extern "C" __global__ void stk_im2col(const bf16* __restrict__ x, bf16* __restrict__ cols, int C, int H, int W, int kh,
                                      int kw, int stride, int pad_t, int pad_b, int pad_l, int pad_r, int up, int Kpad,
                                      long long p0, long long np) {
    long long e = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (e >= np * Kpad) return;
    const int Wo = (W * up + pad_l + pad_r - kw) / stride + 1;
    const long long p = p0 + e / Kpad;
    const int k = (int)(e % Kpad);
    const int K = C * kh * kw;
    bf16 v = __float2bfloat16_rn(0.0f);
    if (k < K) {
        const int c = k / (kh * kw), r = k % (kh * kw), ky = r / kw, kx = r % kw;
        const int oy = (int)(p / Wo), ox = (int)(p % Wo);
        const int iy = oy * stride - pad_t + ky, ix = ox * stride - pad_l + kx;
        if (iy >= 0 && iy < H * up && ix >= 0 && ix < W * up) v = x[((long long)c * H + iy / up) * W + ix / up];
    }
    cols[e] = v;
}

// Transpose: x [R, Cc] with row stride ldx -> y [Cc, R] with row stride ldy; y's columns in [R, ldy) are not
// touched (the caller zero-fills them when ldy > R). 32x32 tiles through shared memory.
// launch: block (32, 8), grid (ceil(R / 32), ceil(Cc / 32)) (R on x: it can exceed 65535 tiles).
extern "C" __global__ void stk_transpose_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, long long R, long long Cc,
                                              long long ldx, long long ldy) {
    __shared__ bf16 tile[32][33];
    const long long r0 = (long long)blockIdx.x * 32, c0 = (long long)blockIdx.y * 32;
    for (int i = 0; i < 32; i += 8) {
        const long long r = r0 + threadIdx.y + i, c = c0 + threadIdx.x;
        if (r < R && c < Cc) tile[threadIdx.y + i][threadIdx.x] = x[r * ldx + c];
    }
    __syncthreads();
    for (int i = 0; i < 32; i += 8) {
        const long long c = c0 + threadIdx.y + i, r = r0 + threadIdx.x;
        if (r < R && c < Cc) y[c * ldy + r] = tile[threadIdx.x][threadIdx.y + i];
    }
}

// QwenImage21RMS_norm on CHW (x [C, HW], gamma [C]); per pixel, with F.normalize in fp32 (the input is bf16):
//   ss   = sum_c x_c^2        fp32, c ascending, fmaf chain (the module's own norm order is torch's; ours is fixed)
//   den  = max(sqrt(ss), 1e-12)
//   n_c  = bf16(x_c / den)                          (F.normalize(x.float()).to(bf16); IEEE division)
//   t1_c = bf16(n_c * scale)                        (bf16 tensor * python float: fp32 multiply, one rounding)
//   t2_c = bf16(t1_c * gamma_c)                     (bf16 * bf16 parameter: one rounding)
//   y_c  = bf16(t2_c + 0.0f)                        (`+ self.bias` with bias = 0.0: a no-op except -0 -> +0)
// scale = float(sqrt(C)). y may alias x. launch: block 256, grid ceil(HW / 256) (thread per pixel; adjacent pixels coalesce).
extern "C" __global__ void stk_channel_rms_norm(const bf16* x, const bf16* __restrict__ gamma, bf16* y, int C,
                                                long long HW, float scale) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= HW) return;
    float ss = 0.0f;
    for (int c = 0; c < C; ++c) {
        const float v = vldf(x, (long long)c * HW + i);
        ss = fmaf(v, v, ss);
    }
    const float den = fmaxf(sqrtf(ss), 1e-12f);
    for (int c = 0; c < C; ++c) {
        const long long j = (long long)c * HW + i;
        const float n = bfround(__fdiv_rn(vldf(x, j), den));
        const float t1 = bfround(__fmul_rn(n, scale));
        const float t2 = bfround(__fmul_rn(t1, vldf(gamma, c)));
        y[j] = vstf(__fadd_rn(t2, 0.0f));
    }
}

// y = bf16(x + z) (torch bf16 add: fp32 add, one rounding); y may alias x or z. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void stk_add_bf16(const bf16* x, const bf16* z, bf16* y, long long n) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= n) return;
    y[i] = vstf(__fadd_rn(vldf(x, i), vldf(z, i)));
}

// F.interpolate(x.float(), scale_factor=2.0, mode="nearest-exact").type_as(x) on CHW: y[c, Y, X] = x[c, Y / 2, X / 2].
// (nearest-exact source index is floor((dst + 0.5) * 0.5) = dst >> 1; the float round trip of a bf16 is exact.)
// x [C, H, W] -> y [C, 2H, 2W]. launch: block 256, grid ceil(C * 2H * 2W / 256).
extern "C" __global__ void stk_upsample_nearest2x(const bf16* __restrict__ x, bf16* __restrict__ y, int C, int H, int W) {
    long long e = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    const long long W2 = 2LL * W, H2 = 2LL * H;
    if (e >= (long long)C * H2 * W2) return;
    const long long X = e % W2, Y = (e / W2) % H2, c = e / (W2 * H2);
    y[e] = x[(c * H + (Y >> 1)) * W + (X >> 1)];
}

// QwenImage21DupUp3D for one frame, with first_chunk (keeps the last of the factor_t time slots): x [Cin, H, W] ->
// y [Cout, H*fs, W*fs]. With factor = factor_t * fs * fs and repeats = Cout * factor / Cin:
//   y[o, Y, X] = x[ (o * factor + s) / repeats, Y / fs, X / fs ],  s = fti * fs * fs + (Y % fs) * fs + (X % fs),
// fti = factor_t - 1 (1 when the block also upsamples in time, 0 when it does not).
// launch: block 256, grid ceil(Cout * H*fs * W*fs / 256).
extern "C" __global__ void stk_dup_up(const bf16* __restrict__ x, bf16* __restrict__ y, int Cout, int H, int W, int fs,
                                      int factor, int repeats, int fti) {
    long long e = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    const long long Wo = (long long)W * fs, Ho = (long long)H * fs;
    if (e >= (long long)Cout * Ho * Wo) return;
    const long long X = e % Wo, Y = (e / Wo) % Ho, o = e / (Wo * Ho);
    const int s = fti * fs * fs + (int)(Y % fs) * fs + (int)(X % fs);
    const long long src = (o * factor + s) / repeats;
    y[e] = x[(src * H + Y / fs) * W + X / fs];
}

// Row softmax of the attention scores: S [rows, cols] bf16 (row stride cols) -> P [rows, ldp] bf16, columns in
// [cols, ldp) written as zero:  m = max_j S_j;  e_j = expf((S_j - m) * scale);  P_j = bf16(e_j / sum_j e_j).
// The sum is per-thread partials (thread t takes j = t, t + 256, ... ascending), a xor-shuffle tree inside each warp
// (offsets 16, 8, 4, 2, 1), then the 8 warp sums added in warp order. scale = float(1 / sqrt(C)) (SDPA's default).
// launch: block 256, grid rows.
extern "C" __global__ void stk_softmax_rows(const bf16* __restrict__ S, bf16* __restrict__ P, long long cols,
                                            long long ldp, float scale) {
    __shared__ float red[8];
    __shared__ float bcast;
    const bf16* s = S + (long long)blockIdx.x * cols;
    bf16* p = P + (long long)blockIdx.x * ldp;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float m = -INFINITY;
    for (long long j = t; j < cols; j += STK_BLOCK) m = fmaxf(m, vldf(s, j));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (t == 0) {
        float r = red[0];
        for (int w = 1; w < 8; ++w) r = fmaxf(r, red[w]);
        bcast = r;
    }
    __syncthreads();
    m = bcast;
    __syncthreads();
    float sum = 0.0f;
    for (long long j = t; j < cols; j += STK_BLOCK) sum = __fadd_rn(sum, expf(__fmul_rn(__fsub_rn(vldf(s, j), m), scale)));
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (t == 0) {
        float r = red[0];
        for (int w = 1; w < 8; ++w) r = __fadd_rn(r, red[w]);
        bcast = r;
    }
    __syncthreads();
    const float denom = bcast;
    for (long long j = t; j < ldp; j += STK_BLOCK) {
        float v = 0.0f;
        if (j < cols) v = __fdiv_rn(expf(__fmul_rn(__fsub_rn(vldf(s, j), m), scale)), denom);
        p[j] = vstf(v);
    }
}

// The pipeline's `latents * std + mean` on [C, HW] bf16 with per-channel bf16 mean / std:
// y = bf16( bf16(x * std_c) + mean_c ) (two torch ops, two roundings; no fused multiply-add).
// launch: block 256, grid ceil(C * HW / 256).
extern "C" __global__ void stk_chan_affine(const bf16* x, const bf16* __restrict__ mean, const bf16* __restrict__ stdv,
                                           bf16* y, long long HW, long long n) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= n) return;
    const long long c = i / HW;
    const float t = bfround(__fmul_rn(vldf(x, i), vldf(stdv, c)));
    y[i] = vstf(__fadd_rn(t, vldf(mean, c)));
}

// VaeImageProcessor.postprocess(output_type="pil") on a decoded bf16 image CHW (values already clamped to [-1, 1]):
//   d = clamp(bf16(bf16(x * 0.5) + 0.5), 0, 1)           (`images * 0.5 + 0.5` in bf16, then `.clamp(0, 1)`)
//   u = uint8(rint(float(d) * 255))                      (numpy float32 `(images * 255).round().astype("uint8")`)
// x [C, H, W] -> u [H, W, C] (HWC, the PIL layout). launch: block 256, grid ceil(C*H*W / 256).
extern "C" __global__ void stk_to_u8_hwc(const bf16* __restrict__ x, uint8_t* __restrict__ u, int C, int H, int W) {
    long long e = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (e >= (long long)C * H * W) return;
    const int c = (int)(e % C);
    const long long px = e / C, X = px % W, Y = px / W;
    const float h = bfround(__fmul_rn(vldf(x, ((long long)c * H + Y) * W + X), 0.5f));
    float d = bfround(__fadd_rn(h, 0.5f));
    d = fminf(fmaxf(d, 0.0f), 1.0f);
    u[e] = (uint8_t)rintf(__fmul_rn(d, 255.0f));
}
