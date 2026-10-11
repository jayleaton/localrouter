// Ours: Kandinsky-6 Pro-distill audio primitives (the MMAudio 44k mel VAE decoder's ready operators and the BigVGAN v2
// vocoder's), fp32 throughout, extern "C", raw arguments, deterministic. Independently written from the formula
// receipts in kernels/cuda/kandinsky6/audio/NOTICE.md; no restricted (CC BY-NC-SA edm2_utils) code is copied,
// translated or called. Every rounding is written out (__fmul_rn / __fadd_rn / __fdiv_rn / __fmaf_rn: never contracted,
// never fast math); NO TF32. Every reduction runs in one fixed order that the launch names explicitly (`acc` argument):
//   K6A_ACC_FMA     acc = __fmaf_rn(w, x, acc)              one fused chain
//   K6A_ACC_MUL_ADD acc = __fadd_rn(acc, __fmul_rn(w, x))   two roundings per term
// Scalar division by a host constant is explicit too (`div` argument):
//   K6A_DIV_TRUE    y = __fdiv_rn(v, c)
//   K6A_DIV_RECIP   y = __fmul_rn(v, c), c the fp32 reciprocal the host computed (what PyTorch CUDA does for a CPU
//                   scalar divisor); the caller states which one, there is no default.
//
// libm functions called (CUDA device libm, accurate versions): expf (k6a_scaled_silu, k6a_snake_beta_log), sinf
// (k6a_snake_beta_log). Build with -O3 --fmad=false and without --use_fast_math / -ftz / -prec-div=false /
// -prec-sqrt=false. None of these kernels is qualified against the native twin on SM121 yet (gpu-qualification-pending):
// the operator, chain, mel and PCM gates are open, see src/engines/kandinsky6/audio/vae.zig `contracts`.
//
// Layout: native PyTorch channel-first, contiguous [B, C, T] fp32 (the decoder's and the vocoder's own layout), so no
// operator hides a permutation. Convolution weights are the PREPARED fp32 tensors in PyTorch's own layouts:
// Conv1d [Cout, Cin, K], ConvTranspose1d [Cin, Cout, K]. Weight preparation is not here: it runs on the host at load
// (vae.zig `prepare`). Activation normalization's candidate orders are at the end of this file.
#include <stdint.h>

#define K6A_BLOCK 256
#define K6A_ACC_FMA 0
#define K6A_ACC_MUL_ADD 1
#define K6A_DIV_TRUE 0
#define K6A_DIV_RECIP 1
#define K6A_FILT 12     // BigVGAN Activation1d: UpSample1d / DownSample1d kernel_size 12 (ratio 2)
#define K6A_UP_PAD 5    // UpSample1d: kernel_size // ratio - 1 replicate samples each side
#define K6A_UP_CROP 15  // UpSample1d: pad * ratio + (kernel_size - ratio) // 2 (= the right crop, + 1 // 2)
#define K6A_DN_LEFT 5   // DownSample1d (LowPassFilter1d, even 12 taps): kernel_size // 2 - 1 replicate left, 6 right
#define K6A_CONV_OT 4   // output channels per thread in the direct convolutions

__device__ __forceinline__ float k6a_acc(float acc, float w, float x, int mode) {
    return mode == K6A_ACC_FMA ? __fmaf_rn(w, x, acc) : __fadd_rn(acc, __fmul_rn(w, x));
}

__device__ __forceinline__ float k6a_div(float v, float c, int mode) {
    return mode == K6A_DIV_TRUE ? __fdiv_rn(v, c) : __fmul_rn(v, c);
}

__device__ __forceinline__ long long k6a_clampi(long long i, long long lo, long long hi) {
    return i < lo ? lo : (i > hi ? hi : i);
}

// postprocess_audio's latent unscale and layout: rows fp32 [B, T, C] (the sampler's per-sample audio latent, C = 40)
// -> z fp32 [B, C, T] = (rows / scaling_factor) + mean_value, two roundings (a scalar divide, then an add; the add runs
// even for mean 0.0, so a -0.0 becomes +0.0 as in the reference). `scale` is the divisor (div K6A_DIV_TRUE) or its fp32
// reciprocal (K6A_DIV_RECIP). launch: block 256, grid ceil(B * C * T / 256).
extern "C" __global__ void k6a_latent_unscale(const float* rows, float* z, long long B, long long C, long long T,
                                              float scale, int div, float mean) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= B * C * T) return;
    const long long t = e % T, c = (e / T) % C, b = e / (T * C);
    z[e] = __fadd_rn(k6a_div(rows[(b * T + t) * C + c], scale, div), mean);
}

// VAE.unnormalize: y = x * data_std + data_mean, per mel channel (buffers [1, C, 1] fp32), two roundings (no fma).
// y may alias x. launch: block 256, grid ceil(B * C * T / 256).
extern "C" __global__ void k6a_mel_affine(const float* x, const float* stdv, const float* mean, float* y, long long B,
                                          long long C, long long T) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= B * C * T) return;
    const long long c = (e / T) % C;
    y[e] = __fadd_rn(__fmul_rn(x[e], stdv[c]), mean[c]);
}

// Tensor.clamp(lo, hi): NaN stays NaN (fminf / fmaxf would drop it), otherwise min(max(v, lo), hi). The decoder uses
// [-256, 256] after the middle and after every up-level residual block; the vocoder [-1, 1] at its end. y may alias x.
// launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6a_clamp(const float* x, float* y, long long n, float lo, float hi) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= n) return;
    const float v = x[e];
    y[e] = v != v ? v : (v < lo ? lo : (v > hi ? hi : v));
}

// F.interpolate(x, scale_factor=2.0, mode='nearest-exact') on [R = B * C, T] -> [R, 2T]: source index
// floor((o + 0.5) * 0.5) = o >> 1 for every o < 2^23 (exact in fp32 index arithmetic), a pure copy. launch: block 256,
// grid ceil(R * 2T / 256).
extern "C" __global__ void k6a_upsample_nearest_exact2(const float* x, float* y, long long R, long long T) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= R * 2 * T) return;
    const long long o = e % (2 * T), r = e / (2 * T);
    y[e] = x[r * T + (o >> 1)];
}

// Scaled SiLU (formula receipt: silu(x) / 0.596): s = x / (1 + expf(-x)) with each step rounded, then the explicit
// scalar division by `c` (0.596f, or its fp32 reciprocal under K6A_DIV_RECIP). y may alias x. launch: block 256,
// grid ceil(n / 256).
extern "C" __global__ void k6a_scaled_silu(const float* x, float* y, long long n, float c, int div) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= n) return;
    const float v = x[e];
    const float s = __fdiv_rn(v, __fadd_rn(1.0f, expf(-v)));
    y[e] = k6a_div(s, c, div);
}

// Magnitude-preserving residual mp_sum = lerp(a, b, t) / sqrt((1 - t)^2 + t^2) in torch.lerp's CUDA small-weight
// form: num = fmaf(t, b - a, a) (the subtraction rounded, then one fused step; t < 0.5, checked by the host, vae.zig
// `MpResidual`), then the division by `den` (or multiply by its reciprocal). t and den are fp32 constants the host
// computed. y may alias a or b. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6a_mp_residual(const float* a, const float* b, float* y, long long n, float t, float den,
                                           int div) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= n) return;
    const float av = a[e];
    const float num = __fmaf_rn(t, __fsub_rn(b[e], av), av);
    y[e] = k6a_div(num, den, div);
}

// Gain before convolution: out = w * gain, one rounding per element, into separate immutable storage (the prepared
// weight stays untouched). launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6a_scale_weights(const float* w, float* out, long long n, float gain) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= n) return;
    out[e] = __fmul_rn(w[e], gain);
}

// y = x + r (BigVGAN AMPBlock1 `x = xt + x`: y = xt + x). y may alias either. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6a_add(const float* xt, const float* x, float* y, long long n) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= n) return;
    y[e] = __fadd_rn(xt[e], x[e]);
}

// The vocoder stage merge: xs = r0; xs += r1; xs += r2; x = xs / 3 (explicit scalar division: c = 3.0f or its fp32
// reciprocal). y may alias any input. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void k6a_avg3(const float* r0, const float* r1, const float* r2, float* y, long long n, float c,
                                    int div) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= n) return;
    y[e] = k6a_div(__fadd_rn(__fadd_rn(r0[e], r1[e]), r2[e]), c, div);
}

// Direct Conv1d over PREPARED weights: x [B, Cin, Lin], w [Cout, Cin, K], bias [Cout] or null, y [B, Cout, Lout],
// Lout = (Lin + 2 * pad - dil * (K - 1) - 1) / stride + 1, zero padding, groups 1. Output (b, o, t) is one chain:
// input channel c ascending (outer), tap k ascending (inner), acc starting at +0.0f; a padded tap contributes w * 0.0f
// (it stays in the chain, as explicit zero padding does), then + bias[o] after the chain (one rounding) when bias is
// non-null. Each thread computes K6A_CONV_OT consecutive output channels of one t (shared input reads, the order of
// each output unchanged). launch: block (128), grid (ceil(Lout / 128), ceil(Cout / K6A_CONV_OT), B).
extern "C" __global__ void k6a_conv1d(const float* __restrict__ x, const float* __restrict__ w,
                                      const float* __restrict__ bias, float* __restrict__ y, long long Cin,
                                      long long Cout, long long Lin, long long Lout, int K, int stride, int pad,
                                      int dil, int acc_mode) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long o0 = (long long)blockIdx.y * K6A_CONV_OT;
    const long long b = blockIdx.z;
    if (t >= Lout) return;
    float acc[K6A_CONV_OT];
#pragma unroll
    for (int j = 0; j < K6A_CONV_OT; ++j) acc[j] = 0.0f;
    const float* xb = x + b * Cin * Lin;
    const long long base = t * stride - pad;
    for (long long c = 0; c < Cin; ++c) {
        const float* xc = xb + c * Lin;
        for (int k = 0; k < K; ++k) {
            const long long ti = base + (long long)k * dil;
            const float xv = (ti >= 0 && ti < Lin) ? xc[ti] : 0.0f;
#pragma unroll
            for (int j = 0; j < K6A_CONV_OT; ++j) {
                const long long o = o0 + j;
                if (o < Cout) acc[j] = k6a_acc(acc[j], w[(o * Cin + c) * K + k], xv, acc_mode);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < K6A_CONV_OT; ++j) {
        const long long o = o0 + j;
        if (o >= Cout) continue;
        const float v = bias ? __fadd_rn(acc[j], bias[o]) : acc[j];
        y[(b * Cout + o) * Lout + t] = v;
    }
}

// Direct ConvTranspose1d over PREPARED weights: x [B, Cin, Lin], w [Cin, Cout, K] (PyTorch's transpose layout), bias
// [Cout] or null, y [B, Cout, Lout], Lout = (Lin - 1) * stride - 2 * pad + K (output_padding 0, dilation 1, groups 1).
// Output (b, o, t) gathers the taps k with (t + pad - k) divisible by stride and i = (t + pad - k) / stride in
// [0, Lin): input channel c ascending (outer), k ascending (inner), one chain from +0.0f (taps outside the input do not
// exist in a transposed convolution and are not added), then + bias[o]. launch: block (128), grid (ceil(Lout / 128),
// ceil(Cout / K6A_CONV_OT), B).
extern "C" __global__ void k6a_conv_transpose1d(const float* __restrict__ x, const float* __restrict__ w,
                                                const float* __restrict__ bias, float* __restrict__ y, long long Cin,
                                                long long Cout, long long Lin, long long Lout, int K, int stride,
                                                int pad, int acc_mode) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long o0 = (long long)blockIdx.y * K6A_CONV_OT;
    const long long b = blockIdx.z;
    if (t >= Lout) return;
    float acc[K6A_CONV_OT];
#pragma unroll
    for (int j = 0; j < K6A_CONV_OT; ++j) acc[j] = 0.0f;
    const float* xb = x + b * Cin * Lin;
    const long long tp = t + pad;
    for (long long c = 0; c < Cin; ++c) {
        for (int k = 0; k < K; ++k) {
            const long long q = tp - k;
            if (q < 0 || q % stride != 0) continue;
            const long long i = q / stride;
            if (i >= Lin) continue;
            const float xv = xb[c * Lin + i];
#pragma unroll
            for (int j = 0; j < K6A_CONV_OT; ++j) {
                const long long o = o0 + j;
                if (o < Cout) acc[j] = k6a_acc(acc[j], w[(c * Cout + o) * K + k], xv, acc_mode);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < K6A_CONV_OT; ++j) {
        const long long o = o0 + j;
        if (o >= Cout) continue;
        const float v = bias ? __fadd_rn(acc[j], bias[o]) : acc[j];
        y[(b * Cout + o) * Lout + t] = v;
    }
}

// BigVGAN SnakeBeta with log-scale parameters (alpha_logscale true), as its forward writes it per element:
// alpha = expf(alpha_log[c]); beta = expf(beta_log[c]); y = x + (1 / (beta + eps)) * sin(x * alpha)^2, every step
// rounded: p = x * alpha; s = sinf(p); q = s * s; inv = 1 / (beta + eps); y = x + inv * q. eps is the module's
// no_div_by_zero as fp32 (1e-9f), passed explicitly. y may alias x. launch: block 256, grid ceil(B * C * T / 256).
extern "C" __global__ void k6a_snake_beta_log(const float* x, const float* alpha_log, const float* beta_log, float* y,
                                              long long B, long long C, long long T, float eps) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= B * C * T) return;
    const long long c = (e / T) % C;
    const float alpha = expf(alpha_log[c]);
    const float beta = expf(beta_log[c]);
    const float v = x[e];
    const float s = sinf(__fmul_rn(v, alpha));
    const float inv = __fdiv_rn(1.0f, __fadd_rn(beta, eps));
    y[e] = __fadd_rn(v, __fmul_rn(inv, __fmul_rn(s, s)));
}

// BigVGAN Activation1d up-sampler (UpSample1d, ratio 2, 12 taps, the module's loaded `filter` buffer f[12], the same
// for every channel): x [R = B * C, T] -> y [R, 2T]. Reference: replicate-pad 5 each side (xp[i] = x[clamp(i - 5)]),
// depthwise conv_transpose1d stride 2, times 2, crop 15 each side. Output o is full index n = o + 15 and gathers the taps
// k ascending with (n - k) even, i = (n - k) / 2 in [0, T + 10): one chain from +0.0f, then * 2.0f (exact). launch:
// block 256, grid ceil(R * 2T / 256).
extern "C" __global__ void k6a_aa_up2(const float* x, const float* f, float* y, long long R, long long T, int acc_mode) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= R * 2 * T) return;
    const long long o = e % (2 * T), r = e / (2 * T);
    const long long n = o + K6A_UP_CROP;
    const float* xr = x + r * T;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < K6A_FILT; ++k) {
        const long long q = n - k;
        if (q < 0 || (q & 1)) continue;
        const long long i = q >> 1;
        if (i >= T + 2 * K6A_UP_PAD) continue;
        acc = k6a_acc(acc, f[k], xr[k6a_clampi(i - K6A_UP_PAD, 0, T - 1)], acc_mode);
    }
    y[e] = __fmul_rn(2.0f, acc);
}

// BigVGAN Activation1d down-sampler (DownSample1d = LowPassFilter1d, ratio 2, 12 taps, loaded `filter` f[12]):
// x [R, T2] -> y [R, To], To = (T2 + 11 - 12) / 2 + 1. Reference: replicate-pad 5 left / 6 right, depthwise conv1d
// stride 2. Output m is one chain over k ascending of f[k] * xp[2m + k], xp[j] = x[clamp(j - 5)], from +0.0f. launch:
// block 256, grid ceil(R * To / 256).
extern "C" __global__ void k6a_aa_down2(const float* x, const float* f, float* y, long long R, long long T2,
                                        long long To, int acc_mode) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= R * To) return;
    const long long m = e % To, r = e / To;
    const float* xr = x + r * T2;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < K6A_FILT; ++k)
        acc = k6a_acc(acc, f[k], xr[k6a_clampi(2 * m + k - K6A_DN_LEFT, 0, T2 - 1)], acc_mode);
    y[e] = acc;
}

// Attention layout (AttnBlock1D): the K=1 QKV output viewed as [B, H, D, 3, T] (reshape of [B, H * D * 3, T]: channel
// (h * D + d) * 3 + j, j = 0 / 1 / 2 for Q / K / V) -> q, k, v each [B, H, T, D] (unbind(3) then permute(0, 1, 3, 2)).
// The activation normalization over D between the reshape and the split is a separate, unresolved operator; this kernel
// moves bits only. launch: block 256, grid ceil(B * H * T * D / 256).
extern "C" __global__ void k6a_qkv_split(const float* qkv, float* q, float* k, float* v, long long B, long long H,
                                         long long D, long long T) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= B * H * T * D) return;
    const long long d = e % D, t = (e / D) % T, bh = e / (D * T);
    const long long src = ((bh * D + d) * 3) * T + t;
    q[e] = qkv[src];
    k[e] = qkv[src + T];
    v[e] = qkv[src + 2 * T];
}

// Attention output layout: h [B, H, T, D] -> y [B, H * D, T] (permute(0, 1, 3, 2).reshape). launch: block 256, grid
// ceil(B * H * D * T / 256).
extern "C" __global__ void k6a_attn_merge(const float* h, float* y, long long B, long long H, long long D, long long T) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= B * H * D * T) return;
    const long long t = e % T, d = (e / T) % D, bh = e / (T * D);
    y[e] = h[(bh * T + t) * D + d];
}

// ---------------------------------------------------------------------------------------------------------------
// Candidate kernels for the operators whose TARGET arithmetic is still open. Their formulas come from black-box
// narrowing over permissive PyTorch primitive compositions against the CPU oracle fixtures (src/engines/kandinsky6/
// audio/vae.zig `contract`: 15/15 normalization and 5/5 weight-preparation cases with zero unequal values on CPU); the
// SM121 reduction order is not known, so the order is a launch argument the GPU harness sweeps against the native
// twin. No restricted source was read. None of these is admitted for production. (The weight preparation itself runs on
// the host at load, as the native loader does; see vae.zig `prepare`.)

#define K6A_RED_SERIAL_FMA 0      // one chain, element ascending, acc = fma(v, v, acc)
#define K6A_RED_SERIAL_MUL_ADD 1  // one chain, element ascending, acc = acc + v * v (two roundings)
#define K6A_RED_STRIDED_TREE 2    // R lanes: lane r takes elements r, r + R, ... (fma chain), then a halving tree

// L2 norm over the axis of extent N of a contiguous tensor viewed as [outer, N, inner] (the reduced axis has stride
// inner): n[o, i] = sqrtf(sum_j x[o, j, i]^2) under `order`. R is the lane count of K6A_RED_STRIDED_TREE only (a
// power of two <= blockDim.x, checked by the caller, launch.zig); the serial orders run on thread 0 and ignore R.
// launch: block (R rounded up to 32, at least 32; 32 for the serial orders), grid (outer * inner); shared R floats.
extern "C" __global__ void k6a_l2norm(const float* x, float* n, long long outer, long long N, long long inner,
                                      int order, int R) {
    extern __shared__ float lane[];
    const long long out = blockIdx.x;
    if (out >= outer * inner) return;
    const long long o = out / inner, i = out % inner;
    const float* base = x + o * N * inner + i;
    if (order != K6A_RED_STRIDED_TREE) {
        if (threadIdx.x != 0) return;
        float acc = 0.0f;
        for (long long j = 0; j < N; ++j) {
            const float v = base[j * inner];
            acc = order == K6A_RED_SERIAL_FMA ? __fmaf_rn(v, v, acc) : __fadd_rn(acc, __fmul_rn(v, v));
        }
        n[out] = __fsqrt_rn(acc);
        return;
    }
    const int r = threadIdx.x;
    if (r < R) {
        float acc = 0.0f;
        for (long long j = r; j < N; j += R) {
            const float v = base[j * inner];
            acc = __fmaf_rn(v, v, acc);
        }
        lane[r] = acc;
    }
    __syncthreads();
    for (int h = R / 2; h > 0; h /= 2) {
        if (r < h) lane[r] = __fadd_rn(lane[r], lane[r + h]);
        __syncthreads();
    }
    if (r == 0) n[out] = __fsqrt_rn(lane[0]);
}

// L2 norm in PyTorch's CUDA reduction order (ATen Reduce.cuh) for an axis that is not the fastest moving, with the
// input split over Y threads of a block and C blocks per output (the host derives Y and C from the shape and the
// device as torch's setReduceConfig does, launch.zig `torchSplit`): thread (y, c) sums the elements y + Y*c + Y*C*m in
// four interleaved fma accumulators (m % 4), combined ((a0 + a1) + a2) + a3; the Y partials of a block a halving tree
// (y += y + h); with C > 1 then per y the block partials c = y, y + Y, ... added in order onto 0, and a second halving
// tree. One thread computes one output's whole order (Y <= 32, a power of two; C >= 1). n[o, i] = sqrtf(sum).
// launch: block 128, grid ceil(outer * inner / 128).
extern "C" __global__ void k6a_l2norm_torch(const float* x, float* n, long long outer, long long N, long long inner,
                                            int Y, int C) {
    const long long out = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= outer * inner || Y < 1 || Y > 32 || C < 1) return;
    const long long o = out / inner, i = out % inner;
    const float* base = x + o * N * inner + i;
    const long long step = (long long)Y * C;
    float part[32], carry[32];
    for (int y = 0; y < Y; ++y) carry[y] = 0.0f;
    float block_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < Y; ++y) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            long long m = 0;
            for (long long j = y + (long long)Y * c; j < N; j += step, ++m) {
                const float v = base[j * inner];
                acc[m & 3] = __fmaf_rn(v, v, acc[m & 3]);
            }
            part[y] = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
        }
        for (int h = Y / 2; h > 0; h /= 2)
            for (int y = 0; y < h; ++y) part[y] = __fadd_rn(part[y], part[y + h]);
        if (C == 1) block_sum = part[0];
        else carry[c % Y] = __fadd_rn(carry[c % Y], part[0]);
    }
    if (C > 1) {
        for (int h = Y / 2; h > 0; h /= 2)
            for (int y = 0; y < h; ++y) carry[y] = __fadd_rn(carry[y], carry[y + h]);
        block_sum = carry[0];
    }
    n[out] = __fsqrt_rn(block_sum);
}

// Activation normalization apply: y[o, j, i] = x[o, j, i] / den[o, i], den = eps + n * s, s = fp32(sqrt(1 / N)).
// den_fused 1: den = fmaf(n, s, eps) (one rounding; the composition that matched the CPU oracle), 0: two roundings.
// y may alias x. launch: block 256, grid ceil(outer * N * inner / 256).
extern "C" __global__ void k6a_normalize_apply(const float* x, const float* n, float* y, long long outer, long long N,
                                               long long inner, float s, float eps, int den_fused) {
    const long long e = (long long)blockIdx.x * K6A_BLOCK + threadIdx.x;
    if (e >= outer * N * inner) return;
    const long long i = e % inner, o = e / (N * inner);
    const float nv = n[o * inner + i];
    const float den = den_fused ? __fmaf_rn(nv, s, eps) : __fadd_rn(eps, __fmul_rn(nv, s));
    y[e] = __fdiv_rn(x[e], den);
}

// Attention, math formulation (one candidate for the native fp32 SDPA; the native backend choice is open): per row
// i of q [BH, T, D]: s_j = sum_d (q[i,d] * c) * (k[j,d] * c) with c = sqrt(scale) applied to each operand (fma chain,
// d ascending); m = max_j s_j; p_j = expf(s_j - m); z = sum_j p_j (ascending); out[i,d] = sum_j (p_j / z) * v[j,d]
// (fma chain, j ascending). The row max propagates NaN like torch's softmax. c = fp32(sqrt(1 / sqrt(D))) computed in
// double by the caller. q, k, v, out contiguous [BH, T, D]; scratch holds BH * T * T floats. launch: block 256,
// grid (T, BH).
extern "C" __global__ void k6a_sdpa_math(const float* q, const float* k, const float* v, float* out, float* scratch,
                                         long long T, long long D, float c) {
    const long long i = blockIdx.x, bh = blockIdx.y;
    if (i >= T) return;  // uniform per block
    const float* qi = q + (bh * T + i) * D;
    float* p = scratch + (bh * T + i) * T;
    for (long long j = threadIdx.x; j < T; j += blockDim.x) {
        const float* kj = k + (bh * T + j) * D;
        float acc = 0.0f;
        for (long long d = 0; d < D; ++d) acc = __fmaf_rn(__fmul_rn(qi[d], c), __fmul_rn(kj[d], c), acc);
        p[j] = acc;
    }
    __syncthreads();
    __shared__ float z_s;
    if (threadIdx.x == 0) {
        float m = p[0];
        for (long long j = 1; j < T; ++j) m = (p[j] > m || p[j] != p[j]) && m == m ? p[j] : m;
        float z = 0.0f;
        for (long long j = 0; j < T; ++j) {
            p[j] = expf(__fsub_rn(p[j], m));
            z = __fadd_rn(z, p[j]);
        }
        z_s = z;
    }
    __syncthreads();
    for (long long d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (long long j = 0; j < T; ++j) acc = __fmaf_rn(__fdiv_rn(p[j], z_s), v[(bh * T + j) * D + d], acc);
        out[(bh * T + i) * D + d] = acc;
    }
}
