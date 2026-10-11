// Kandinsky-6 video VAE: everything but the GEMMs (conv.cu). Channels-last fp16 activations; every kernel has a fixed
// reduction order that depends on neither the device nor the grid's residency, and one parameter block that
// src/engines/kandinsky6/video_vae/launch.zig mirrors field for field. Each comment gives the reference expression
// (pinned upstream kandinsky-6 eac06e39, torch fp16 module) and the exact arithmetic used. No fast math; separate torch ops
// are separate roundings, written with __f*_rn so nvcc cannot contract them.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stddef.h>
#include <stdint.h>

#define K6V_GN_CHUNK 2048  // positions a partial-sum block covers (fixed: the partial count is a function of the shape only)

// ---- GroupNorm (nn.GroupNorm(32, C, eps=1e-6, affine), fp16 in and out) ----------------------------------------------
// Statistics in two passes over x: pass 0 sums x, pass 1 sums (x - mean)^2 with fmaf. A block covers K6V_GN_CHUNK
// positions; its 256 threads are V = C / 8 channel octets by Q = 256 / V position lanes; thread (q, v) accumulates its
// eight channels over positions chunk_start + q, + Q, .. in ascending order. The block's partial of group g is then
//   sum over q = 0 .. Q-1, over the channels c of g ascending: acc[q][c / 8][c % 8]    (sequential fp32, one thread)
// The finalize kernel adds a group's partials: thread t sums chunks t, t + 256, .. ascending, then a fixed halving tree.
//   pass 0: mean = float(double(sum) / double(count));   pass 1: var = float(double(sum) / double(count)),
//   rstd = rsqrtf(var + eps)                              count = positions * C / groups
// Apply: a = rstd * gamma[c], b = fmaf(-a, mean, beta[c]), y = half(fmaf(x, a, b)); with silu (the resnet's
// nonlinearity, a separate torch op on the fp16 norm output): s = float(y), y = half(s / (1 + expf(-s))).
struct K6vGnArgs {
    const __half* x;
    float* partial;      // [chunks][groups]
    float* stats;        // [groups][2]: mean, rstd
    long long positions;  // T * H * W
    int c, groups, pass;
    float eps;
};

extern "C" __global__ void __launch_bounds__(256) k6v_gn_partial(const K6vGnArgs a) {
    __shared__ float red[256][8];
    const int V = a.c / 8, Q = 256 / V, v = threadIdx.x % V, q = threadIdx.x / V;
    const long long p0 = (long long)blockIdx.x * K6V_GN_CHUNK;
    const long long p1 = min(p0 + K6V_GN_CHUNK, a.positions);
    const int cg = a.c / a.groups;
    float acc[8], mean[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0.0f;
        mean[j] = a.pass ? a.stats[((v * 8 + j) / cg) * 2] : 0.0f;
    }
    for (long long p = p0 + q; p < p1; p += Q) {
        const uint4 raw = *reinterpret_cast<const uint4*>(a.x + p * a.c + v * 8);
        const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = __half2float(h[j]);
            if (a.pass) {
                const float d = __fsub_rn(f, mean[j]);
                acc[j] = __fmaf_rn(d, d, acc[j]);
            } else {
                acc[j] = __fadd_rn(acc[j], f);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) red[threadIdx.x][j] = acc[j];
    __syncthreads();
    if (threadIdx.x < a.groups) {
        const int g = threadIdx.x;
        float s = 0.0f;
        for (int qq = 0; qq < Q; ++qq)
            for (int c = g * cg; c < (g + 1) * cg; ++c) s = __fadd_rn(s, red[qq * V + c / 8][c % 8]);
        a.partial[(long long)blockIdx.x * a.groups + g] = s;
    }
}

// grid (groups), block 256; chunks = ceil(positions / K6V_GN_CHUNK).
extern "C" __global__ void __launch_bounds__(256) k6v_gn_finalize(const K6vGnArgs a) {
    __shared__ float red[256];
    const int g = blockIdx.x;
    const long long chunks = (a.positions + K6V_GN_CHUNK - 1) / K6V_GN_CHUNK;
    float s = 0.0f;
    for (long long i = threadIdx.x; i < chunks; i += 256) s = __fadd_rn(s, a.partial[i * a.groups + g]);
    red[threadIdx.x] = s;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (threadIdx.x < h) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + h]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double count = (double)a.positions * (double)(a.c / a.groups);
        const float r = (float)((double)red[0] / count);
        if (a.pass == 0)
            a.stats[g * 2] = r;
        else
            a.stats[g * 2 + 1] = rsqrtf(__fadd_rn(r, a.eps));
    }
}

struct K6vGnApplyArgs {
    const __half* x;
    __half* y;  // may alias x
    const __half* gamma;
    const __half* beta;
    const float* stats;
    long long positions;
    int c, groups, silu;
};

// grid ceil(positions * c / 8 / 256), block 256: one channel octet a thread.
extern "C" __global__ void __launch_bounds__(256) k6v_gn_apply(const K6vGnApplyArgs a) {
    const long long o = (long long)blockIdx.x * 256 + threadIdx.x;
    if (o >= a.positions * (a.c / 8)) return;
    const int c0 = (int)(o % (a.c / 8)) * 8, cg = a.c / a.groups;
    uint4 raw = *reinterpret_cast<const uint4*>(a.x + o * 8);
    __half* h = reinterpret_cast<__half*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = c0 + j, g = c / cg;
        const float mean = a.stats[g * 2], rstd = a.stats[g * 2 + 1];
        const float sc = __fmul_rn(rstd, __half2float(a.gamma[c]));
        const float sh = __fmaf_rn(-sc, mean, __half2float(a.beta[c]));
        __half y = __float2half_rn(__fmaf_rn(__half2float(h[j]), sc, sh));
        if (a.silu) {
            const float s = __half2float(y);
            y = __float2half_rn(__fdiv_rn(s, __fadd_rn(1.0f, expf(-s))));
        }
        h[j] = y;
    }
    *reinterpret_cast<uint4*>(a.y + o * 8) = raw;
}

// ---- Mid-block attention softmax ---------------------------------------------------------------------------------------
// One head (dim 512), scores S = Q K^T in fp32 (conv.cu, out_f32), the frame-causal mask of prepare_causal_attention_mask:
// token i (frame i / spf) sees token j only when j / spf <= i / spf, i.e. j < (i / spf + 1) * spf = valid.
//   x_j = S_ij * scale (scale = 512^-0.5 as float), m = max_j<valid x_j, e_j = expf(x_j - m),
//   l = sum e_j (thread t: j = t, t + 256, .. ascending; then a fixed halving tree), P_ij = half(e_j / l) for j < valid,
//   P_ij = 0 for valid <= j < ldp (masked, and the zero padding conv.cu needs past K).
// grid (rows), block 256; row r of the call is global row row0 + r.
struct K6vSoftmaxArgs {
    const float* s;
    __half* p;
    long long lds, ldp;
    int row0, spf, cols;  // cols: the number of key columns S holds for this block (>= every row's valid)
    float scale;
};

extern "C" __global__ void __launch_bounds__(256) k6v_softmax_rows(const K6vSoftmaxArgs a) {
    __shared__ float red[256];
    const long long r = blockIdx.x;
    const float* s = a.s + r * a.lds;
    __half* p = a.p + r * a.ldp;
    const int valid = ((a.row0 + (int)r) / a.spf + 1) * a.spf;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < valid; j += 256) m = fmaxf(m, __fmul_rn(s[j], a.scale));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (threadIdx.x < h) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + h]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float l = 0.0f;
    for (int j = threadIdx.x; j < valid; j += 256) l = __fadd_rn(l, expf(__fsub_rn(__fmul_rn(s[j], a.scale), m)));
    red[threadIdx.x] = l;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (threadIdx.x < h) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + h]);
        __syncthreads();
    }
    l = red[0];
    for (long long j = threadIdx.x; j < a.ldp; j += 256)
        p[j] = j < valid ? __float2half_rn(__fdiv_rn(expf(__fsub_rn(__fmul_rn(s[j], a.scale), m)), l)) : __ushort_as_half(0);
}

// V [n][c] -> V^T [c][ldn] for P . V (conv.cu wants B as [N][K] rows); columns n .. ldn - 1 are zeroed.
// grid (ceil(ldn / 32), ceil(c / 32)), block (32, 8).
struct K6vTransposeArgs {
    const __half* x;
    __half* y;
    long long n, ldn;
    int c;
};

extern "C" __global__ void __launch_bounds__(256) k6v_transpose(const K6vTransposeArgs a) {
    __shared__ __half t[32][33];
    const long long n0 = (long long)blockIdx.x * 32;
    const int c0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += 8) {
        const long long n = n0 + i;
        const int c = c0 + threadIdx.x;
        t[i][threadIdx.x] = (n < a.n && c < a.c) ? a.x[n * a.c + c] : __ushort_as_half(0);
    }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += 8) {
        const int c = c0 + i;
        const long long n = n0 + threadIdx.x;
        if (c < a.c && n < a.ldn) a.y[(long long)c * a.ldn + n] = t[threadIdx.x][i];
    }
}

// ---- Boundaries -----------------------------------------------------------------------------------------------------
// Decode input (postprocess_video): bf16 DiT latent / scaling_factor on the GPU is a * (1 / b) in float (torch's
// div_true_kernel_cuda for a CPU scalar divisor), rounded to bf16; then .to(float16): half(float(bf16)).
// inv_sf = 1.0f / float(0.476986) computed on the host in fp32, as torch does.
struct K6vLatentInArgs {
    const __nv_bfloat16* z;
    __half* y;
    long long count;
    float inv_sf;
};

extern "C" __global__ void __launch_bounds__(256) k6v_latent_in(const K6vLatentInArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.count) return;
    const __nv_bfloat16 b = __float2bfloat16_rn(__fmul_rn(__bfloat162float(a.z[i]), a.inv_sf));
    a.y[i] = __float2half_rn(__bfloat162float(b));
}

// I2V encode input (encode_i2va_first_frame): the uint8 image is normalised on the CPU in fp32, tensor / 127.5 - 1.0
// (true division there), then .to(float16): half((float(u) / 127.5f) - 1.0f).
struct K6vRgb8InArgs {
    const uint8_t* x;
    __half* y;
    long long count;
};

extern "C" __global__ void __launch_bounds__(256) k6v_rgb8_in(const K6vRgb8InArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.count) return;
    a.y[i] = __float2half_rn(__fsub_rn(__fdiv_rn((float)a.x[i], 127.5f), 1.0f));
}

// Decode output (postprocess_video): ((frames.clamp(-1, 1) + 1.0) * 127.5).to(uint8), all fp16 ops then a truncating cast:
//   c = clamp(h, -1, 1);  u = half(float(c) + 1);  v = half(float(u) * 127.5);  out = uint8(trunc(float(v)))
// Same layout in and out (frames of [H][W][3]); NaN maps to 0.
struct K6vRgbOutArgs {
    const __half* x;
    uint8_t* y;
    long long count;
};

extern "C" __global__ void __launch_bounds__(256) k6v_rgb_out(const K6vRgbOutArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.count) return;
    const float c = fminf(fmaxf(__half2float(a.x[i]), -1.0f), 1.0f);
    const float u = __half2float(__float2half_rn(__fadd_rn(c, 1.0f)));
    const float v = __half2float(__float2half_rn(__fmul_rn(u, 127.5f)));
    a.y[i] = (uint8_t)__float2uint_rz(v);
}

// Encoder posterior (DiagonalGaussianDistribution over quant_conv's [P][2 * lc] output, channels mean then logvar):
//   lv = clamp(logvar, -30, 20); std = half(expf(float(half(0.5f * lv))))
//   with eps (caller's noise, fp16): x = half(mean + float(half(std * eps)))   .sample()
//   without eps:                     x = mean                                   .mode()
//   latent = half(float(x) * sf)  (latent * vae.config.scaling_factor, fp16), written [P][lc] (the packed THWC layout)
struct K6vPosteriorArgs {
    const __half* moments;
    const __half* eps;  // [P][lc] or null
    __half* mean;       // [P][lc] or null
    __half* stdv;       // [P][lc] or null
    __half* latent;     // [P][lc]
    long long positions;
    int lc;
    float sf;
};

extern "C" __global__ void __launch_bounds__(256) k6v_posterior(const K6vPosteriorArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= a.positions * a.lc) return;
    const long long p = i / a.lc;
    const int c = (int)(i - p * a.lc);
    const __half mu = a.moments[p * 2 * a.lc + c];
    const float lv = fminf(fmaxf(__half2float(a.moments[p * 2 * a.lc + a.lc + c]), -30.0f), 20.0f);
    const __half sd = __float2half_rn(expf(__half2float(__float2half_rn(__fmul_rn(0.5f, lv)))));
    __half x = mu;
    if (a.eps) {
        const __half se = __float2half_rn(__fmul_rn(__half2float(sd), __half2float(a.eps[i])));
        x = __float2half_rn(__fadd_rn(__half2float(mu), __half2float(se)));
    }
    if (a.mean) a.mean[i] = mu;
    if (a.stdv) a.stdv[i] = sd;
    a.latent[i] = __float2half_rn(__fmul_rn(__half2float(x), a.sf));
}

// ---- Tile blending (AutoencoderKLHunyuanVideo.blend_t / blend_v / blend_h), in place on b ------------------------------
// For y < e along the axis:  b[y] = a[dim_a - e + y] * (1 - y / e) + b[y] * (y / e), torch fp16 with Python float weights:
//   r = double(y) / double(e); w1 = float(1.0 - r); w2 = float(r)
//   b = half(float(half(float(a) * w1)) + float(half(float(b) * w2)))
// a and b are channels-last boxes [f][h][w][ch] (contiguous); the two axes not blended have equal extents.
struct K6vBlendArgs {
    const __half* a;
    __half* b;
    int fa, ha, wa;
    int fb, hb, wb;
    int ch, axis, extent;  // axis 0 time, 1 height, 2 width
};

extern "C" __global__ void __launch_bounds__(256) k6v_blend(const K6vBlendArgs a) {
    const int d0 = a.axis == 0 ? a.extent : a.fb, d1 = a.axis == 1 ? a.extent : a.hb, d2 = a.axis == 2 ? a.extent : a.wb;
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)d0 * d1 * d2 * a.ch) return;
    const int c = (int)(i % a.ch);
    long long r = i / a.ch;
    const int x = (int)(r % d2);
    r /= d2;
    const int y = (int)(r % d1);
    const int f = (int)(r / d1);
    const int pos = a.axis == 0 ? f : (a.axis == 1 ? y : x);
    const int fa = a.axis == 0 ? a.fa - a.extent + f : f;
    const int ya = a.axis == 1 ? a.ha - a.extent + y : y;
    const int xa = a.axis == 2 ? a.wa - a.extent + x : x;
    const double rr = (double)pos / (double)a.extent;
    const float w1 = (float)(1.0 - rr), w2 = (float)rr;
    const long long ib = (((long long)f * a.hb + y) * a.wb + x) * a.ch + c;
    const long long ia = (((long long)fa * a.ha + ya) * a.wa + xa) * a.ch + c;
    const float pa = __half2float(__float2half_rn(__fmul_rn(__half2float(a.a[ia]), w1)));
    const float pb = __half2float(__float2half_rn(__fmul_rn(__half2float(a.b[ib]), w2)));
    a.b[ib] = __float2half_rn(__fadd_rn(pa, pb));
}

// Box copy between channels-last tensors (tile crop and placement): dst[d0 + i] = src[s0 + i] over an f x h x w x ch box.
struct K6vCopyArgs {
    const __half* src;
    __half* dst;
    int sf, sh, sw;
    int df, dh, dw;
    int f, h, w, ch;
    int sf0, sh0, sw0;
    int df0, dh0, dw0;
};

extern "C" __global__ void __launch_bounds__(256) k6v_copy(const K6vCopyArgs a) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)a.f * a.h * a.w * a.ch) return;
    const int c = (int)(i % a.ch);
    long long r = i / a.ch;
    const int x = (int)(r % a.w);
    r /= a.w;
    const int y = (int)(r % a.h);
    const int f = (int)(r / a.h);
    const long long is = (((long long)(a.sf0 + f) * a.sh + a.sh0 + y) * a.sw + a.sw0 + x) * a.ch + c;
    const long long id = (((long long)(a.df0 + f) * a.dh + a.dh0 + y) * a.dw + a.dw0 + x) * a.ch + c;
    a.dst[id] = a.src[is];
}

// launch.zig mirrors these blocks; its comptime checks assert the same sizes.
static_assert(sizeof(K6vGnArgs) == 48 && sizeof(K6vGnApplyArgs) == 64 && sizeof(K6vSoftmaxArgs) == 48, "launch.zig");
static_assert(sizeof(K6vTransposeArgs) == 40 && sizeof(K6vLatentInArgs) == 32 && sizeof(K6vRgb8InArgs) == 24, "launch.zig");
static_assert(sizeof(K6vPosteriorArgs) == 56 && sizeof(K6vBlendArgs) == 56 && sizeof(K6vCopyArgs) == 80, "launch.zig");
