// Ours: MiniMax H3's audio VAE decoder (a BigVGAN, ComfyUI 0.37.0 MiniMaxH3AudioVAE.decode + vae_decode_audio's
// normalisation), fp32 throughout, shared by the twin (torch extension) and the Zig engine (fatbin). extern "C", raw
// arguments, deterministic: every reduction in a fixed order, every rounding written out (__fmul_rn / __fadd_rn /
// __fmaf_rn / __fdiv_rn: never contracted, never fast-math). NO TF32 anywhere.
//
// libm functions called (CUDA device libm, the accurate versions, i.e. nvcc without --use_fast_math): expf, sinf
// (only in avae_snake). Build with --fmad=false and without --use_fast_math / -ftz / -prec-div=false; the twin and the
// Zig build must use the same nvcc (the libdevice expf / sinf bits are fixed by the toolkit version and these flags).
//
// Layout: every activation is channel-last, [B, T, C] fp32 contiguous (B = batch * stereo = the stereo channels are
// independent batch items; C fastest), so a pointwise / depthwise-in-time kernel reads coalesced and a convolution is
// im2col + h3_gemm_f32 (kernels/cuda/minimax/ops.cu: out = fmaf chain over k ascending from 0, then + bias; an output's
// bits depend only on its two rows). Conv weights are repacked on the host (a pure permutation) to [Cout, tap, Cin]
// so a convolution's summation order is: tap k ascending (outer), input channel ascending (inner), one fmaf chain, then
// + bias. The K = 1 and K = 7 / 3 / 11 convolutions all follow this; the depthwise filters have their own orders below.
#include <stdint.h>

#define AVAE_BLOCK 256
#define AVAE_FILT 12      // the Kaiser-sinc filters are 12 taps (UpSample1d / DownSample1d kernel_size 12)
#define AVAE_UP_PAD 5     // UpSample1d: kernel_size // ratio - 1 replicate padding each side
#define AVAE_UP_LEFT 15   // UpSample1d: pad * ratio + (kernel_size - ratio) // 2 (= pad_right: + (kernel_size - ratio + 1) // 2)
#define AVAE_DN_LEFT 5    // DownSample1d: kernel_size // 2 - 1 replicate padding on the left (6 on the right)

__device__ __forceinline__ long long avae_clampi(long long i, long long lo, long long hi) {
    return i < lo ? lo : (i > hi ? hi : i);
}

// Latent in: z fp32 [Bb, C, S, T] (the sampler's audio latent after * 0.25, [1, 32, 2, A]) ->
// rows fp32 [Bb * S, T, C] = (z.permute(0, 2, 1, 3).reshape(Bb * S, C, T) * std + mean), channel-last, as two torch
// ops (a multiply, then an add: two roundings, no fma). launch: block 256, grid ceil(Bb * S * T * C / 256).
extern "C" __global__ void avae_latent_in(const float* z, const float* mean, const float* stdv, float* rows, long long Bb,
                                          long long C, long long S, long long T) {
    const long long e = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (e >= Bb * S * T * C) return;
    const long long c = e % C, r = e / C, t = r % T, item = r / T;
    const long long bb = item / S, s = item % S;
    const float v = z[((bb * C + c) * S + s) * T + t];
    rows[e] = __fadd_rn(__fmul_rn(v, stdv[c]), mean[c]);
}

// im2col for a stride-1 "same" Conv1d with zero padding: x fp32 [B, T, C] -> col fp32 [B * T, K * C], column
// k * C + ci = x[b, t + k * dil - pad, ci] (0 outside [0, T)). The zero taps stay in the fmaf chain (they add +0).
// launch: block 256, grid ceil(B * T * K * C / 256).
extern "C" __global__ void avae_im2col(const float* x, float* col, long long B, long long T, long long C, int K, int dil,
                                       int pad) {
    const long long e = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (e >= B * T * K * C) return;
    const long long ci = e % C, k = (e / C) % K, row = e / (C * K), t = row % T, b = row / T;
    const long long ti = t + k * dil - pad;
    col[e] = (ti >= 0 && ti < T) ? x[(b * T + ti) * C + ci] : 0.0f;
}

// ConvTranspose1d (stride u, kernel K, padding pad, groups 1) as u GEMMs, one per output phase r = (t_o + pad) mod u.
// With q = (t_o + pad) div u, output t_o = u * q + r - pad takes the taps k = r + j * u (j = 0 .. J - 1, J =
// ceil((K - r) / u)) of input t_i = q - j. The phase's outputs are m = 0 .. L - 1 with q = m + qoff, qoff = 1 if r < pad
// else 0 (so t_o = u * (m + qoff) + r - pad, 0 <= t_o < u * L: the output length (L - 1) * u - 2 * pad + K = u * L
// for the (k - u) // 2 paddings of this model). im2col of phase r: x fp32 [B, L, C] -> col fp32 [B * L, J * C],
// column j * C + ci = x[b, m + qoff - j, ci] (0 outside [0, L)). launch: block 256, grid ceil(B * L * J * C / 256).
extern "C" __global__ void avae_ct_im2col(const float* x, float* col, long long B, long long L, long long C, int J,
                                          int qoff) {
    const long long e = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (e >= B * L * J * C) return;
    const long long ci = e % C, j = (e / C) % J, row = e / (C * J), m = row % L, b = row / L;
    const long long ti = m + qoff - j;
    col[e] = (ti >= 0 && ti < L) ? x[(b * L + ti) * C + ci] : 0.0f;
}

// Phase r's GEMM output ph fp32 [B * L, Cout] -> y fp32 [B, u * L, Cout] at t_o = u * (m + qoff) + r - pad (each
// output position is written by exactly one phase). launch: block 256, grid ceil(B * L * Cout / 256).
extern "C" __global__ void avae_ct_store(const float* ph, float* y, long long B, long long L, long long Cout, int u,
                                         int pad, int r, int qoff) {
    const long long e = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (e >= B * L * Cout) return;
    const long long co = e % Cout, row = e / Cout, m = row % L, b = row / L;
    const long long to = (long long)u * (m + qoff) + r - pad;
    y[(b * u * L + to) * Cout + co] = ph[e];
}

// UpSample1d (ratio 2, 12 taps), depthwise, the same filter f[12] for every channel: x fp32 [B, T, C] -> y fp32
// [B, 2T, C]. torch: replicate-pad 5 each side, conv_transpose1d(stride 2, groups C), * 2, crop [15 : -15]. Output o
// is padded-domain position p = o + 15 = 2 i + k: the six taps k = (p mod 2) + 2 j, j = 0 .. 5, of padded input
// i = p div 2 - j, i.e. x[clamp(i - 5, 0, T - 1)]; ONE fmaf chain over j ascending (k ascending) from 0, then * 2
// (exact). launch: block 256, grid ceil(B * 2T * C / 256).
extern "C" __global__ void avae_up2(const float* x, const float* f, float* y, long long T, long long C, long long n) {
    const long long e = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (e >= n) return;
    const long long c = e % C, r = e / C, o = r % (2 * T), b = r / (2 * T);
    const long long p = o + AVAE_UP_LEFT, i0 = p >> 1;
    const int kp = (int)(p & 1);
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < AVAE_FILT / 2; ++j) {
        const long long xi = avae_clampi(i0 - j - AVAE_UP_PAD, 0, T - 1);
        acc = __fmaf_rn(f[kp + 2 * j], x[(b * T + xi) * C + c], acc);
    }
    y[e] = __fmul_rn(acc, 2.0f);
}

// DownSample1d (LowPassFilter1d, stride 2, 12 taps), depthwise, one filter f[12]: x fp32 [B, T2, C] (T2 even) ->
// y fp32 [B, T2 / 2, C]. torch: replicate-pad 5 left / 6 right, conv1d(stride 2, groups C). y[o] = sum over k = 0 ..
// 11 of f[k] * x[clamp(2 o + k - 5, 0, T2 - 1)]; ONE fmaf chain over k ascending from 0.
// launch: block 256, grid ceil(B * (T2 / 2) * C / 256).
extern "C" __global__ void avae_down2(const float* x, const float* f, float* y, long long T2, long long C, long long n) {
    const long long e = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (e >= n) return;
    const long long To = T2 / 2;
    const long long c = e % C, r = e / C, o = r % To, b = r / To;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < AVAE_FILT; ++k) {
        const long long xi = avae_clampi(2 * o + k - AVAE_DN_LEFT, 0, T2 - 1);
        acc = __fmaf_rn(f[k], x[(b * T2 + xi) * C + c], acc);
    }
    y[e] = acc;
}

// SnakeBeta, pointwise, per channel (C fastest): alpha = exp(pa[c]), beta = exp(pb[c]) (the checkpoint stores logs);
// torch: t = sin(alpha * x); t *= t; t *= 1 / (beta + 1e-9); t += x, each a separate fp32 rounding, 1e-9 as the fp32
// cast of the double. y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void avae_snake(const float* x, const float* pa, const float* pb, float* y, long long C, long long n) {
    const long long e = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (e >= n) return;
    const long long c = e % C;
    const float a = expf(pa[c]);
    const float ib = __fdiv_rn(1.0f, __fadd_rn(expf(pb[c]), (float)1e-9));
    const float v = x[e];
    float t = sinf(__fmul_rn(a, v));
    t = __fmul_rn(t, t);
    t = __fmul_rn(t, ib);
    y[e] = __fadd_rn(t, v);
}

// y = a + b, fp32 (the AMP block's residual: xt.add_(x)). y may alias a or b. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void avae_add(const float* a, const float* b, float* y, long long n) {
    const long long i = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (i < n) y[i] = __fadd_rn(a[i], b[i]);
}

// The average of the three AMP blocks: s = (r0 + r1) + r2 (xs = rb0(x); xs += rb1(x); xs += rb2(x)), then xs.div_(3).
// torch's div_ by a Python scalar on CUDA multiplies by the fp32 reciprocal (recip = 1: s * (1.0f / 3.0f), the default
// to match ComfyUI); recip = 0 is the true division s / 3 (kept to pin which one the pod's torch does; the twin's test
// decides). launch: block 256, grid ceil(n / 256).
extern "C" __global__ void avae_avg3(const float* r0, const float* r1, const float* r2, float* y, long long n, int recip) {
    const long long i = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float s = __fadd_rn(__fadd_rn(r0[i], r1[i]), r2[i]);
    y[i] = recip ? __fmul_rn(s, __fdiv_rn(1.0f, 3.0f)) : __fdiv_rn(s, 3.0f);
}

// y = clamp(x, -1, 1) (NaN passes through, as torch). y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void avae_clamp(const float* x, float* y, long long n) {
    const long long i = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    y[i] = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
}

// vae_decode_audio's scale: audio fp32 [Bb, n] (n = stereo * samples), per row b:
//   std = unbiased std over the n values; s = std * 5.0f; sc[b] = (s < 1.0f) ? 1.0f : s   (NaN stays NaN).
// OUR order (torch's differs): ONE block of 256 threads per row; thread t sums the elements i = t, t + 256, ... ascending
// in fp64 (__dadd_rn), the 256 partials are combined by a shared-memory tree (strides 128, 64, ..., 1: s[t] += s[t +
// stride]); mean = sum / n (fp64 division); a SECOND pass the same way over (x - mean)^2; var = sum2 / (n - 1);
// std = float(sqrt(var)) (fp64 sqrt, one round-to-nearest to fp32). fp64 so the result is the correctly rounded
// fp32 of the true std for all practical inputs, as torch's fp64 Welford on the CPU gives. launch: block 256, grid Bb.
extern "C" __global__ void avae_std_scale(const float* audio, float* sc, long long n) {
    __shared__ double s[AVAE_BLOCK];
    const float* a = audio + (long long)blockIdx.x * n;
    double acc = 0.0;
    for (long long i = threadIdx.x; i < n; i += AVAE_BLOCK) acc = __dadd_rn(acc, (double)a[i]);
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = AVAE_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = __dadd_rn(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    const double mean = __ddiv_rn(s[0], (double)n);
    __syncthreads();
    acc = 0.0;
    for (long long i = threadIdx.x; i < n; i += AVAE_BLOCK) {
        const double d = __dsub_rn((double)a[i], mean);
        acc = __dadd_rn(acc, __dmul_rn(d, d));
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = AVAE_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = __dadd_rn(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float sd = __double2float_rn(__dsqrt_rn(__ddiv_rn(s[0], (double)(n - 1))));
        const float v = __fmul_rn(sd, 5.0f);
        sc[blockIdx.x] = v < 1.0f ? 1.0f : v;
    }
}

// audio /= std, true fp32 division (the CPU path of ComfyUI, where the intermediate device is the CPU; a CUDA tensor
// divided by a one-element CUDA tensor is a true division too). y[b, i] = audio[b, i] / sc[b]; y may alias audio.
// launch: block 256, grid ceil(Bb * n / 256).
extern "C" __global__ void avae_div_scale(const float* audio, const float* sc, float* y, long long n, long long total) {
    const long long i = (long long)blockIdx.x * AVAE_BLOCK + threadIdx.x;
    if (i < total) y[i] = __fdiv_rn(audio[i], sc[i / n]);
}
