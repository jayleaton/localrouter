// Ours: the non-GEMM kernels of MiniMax H3's video VAE decoder (ComfyUI 0.37.0 comfy/ldm/minimax/vae.py, the fp16
// checkpoint), shared by the twin (torch extension, tools/twin/stk_twin/h3/vae_video.py) and the Zig engine (fatbin).
// extern "C", raw arguments, deterministic: every reduction in a fixed order, every rounding written out. fp16
// activations; inside a kernel fp32, one rounding to half per torch op (tools/twin/VVAE-PORT.md has the list).
// The GEMMs are gemm_f16.cu. Layouts: a decoder tile is rows [tokens, 2048] (token = (t * H + y) * W + x, then the 4
// register tokens and a zero token); the latents and pixels are channel-first [C, T, H, W].
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#define VV_BLOCK 256

typedef __half f16;

__device__ __forceinline__ float vv_ld(const f16* p, long long i) { return __half2float(p[i]); }
__device__ __forceinline__ f16 vv_st(float v) { return __float2half_rn(v); }
__device__ __forceinline__ float vv_h(float v) { return __half2float(__float2half_rn(v)); }  // round to half, back

// The row's sum in a fixed order: thread-strided partials in ascending index, then a shared-memory tree.
__device__ __forceinline__ float vv_tree(float acc, float* s) {
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = VV_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = __fadd_rn(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// ------------------------------------------------------------------------------------------------ latents -> tokens

// decode(): z = z * latents_std + latents_mean on [C, plane] fp16 with per-channel fp16 std / mean: two torch ops, two
// roundings, no fused multiply-add: y = half(half(z * std_c) + mean_c). launch: block 256, grid ceil(C * plane / 256).
extern "C" __global__ void vv_denorm(const f16* z, const f16* stdv, const f16* mean, f16* y, long long plane,
                                     long long n) {
    const long long i = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (i >= n) return;
    const long long c = i / plane;
    y[i] = vv_st(__fadd_rn(vv_h(__fmul_rn(vv_ld(z, i), vv_ld(stdv, c))), vv_ld(mean, c)));
}

// A tile's tokens: z [C, Tz, Hz, Wz] (the denormalised latents) -> rows [Tn * h * w, C], row (t, y, x), column c, value
// z[c][min(t0 + t, Tz - 1)][y0 + y][x0 + x]. (ComfyUI repeats the last latent frame pad_tokens times at the end; the
// clamp reads that frame again.) launch: block 256, grid ceil(Tn * h * w * C / 256).
extern "C" __global__ void vv_gather_rows(const f16* z, f16* rows, int C, int Tz, int Hz, int Wz, int t0, int Tn, int y0,
                                          int h, int x0, int w) {
    const long long e = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (e >= (long long)Tn * h * w * C) return;
    const int c = (int)(e % C);
    const long long row = e / C;
    const int x = (int)(row % w), y = (int)((row / w) % h), t = (int)(row / ((long long)w * h));
    const int tz = min(t0 + t, Tz - 1);
    rows[e] = z[(((long long)c * Tz + tz) * Hz + y0 + y) * Wz + x0 + x];
}

// h rows nP .. nP + 4: the 4 register tokens (reg [4, D]) and one zero token. launch: block 256, grid ceil(5 * D / 256).
extern "C" __global__ void vv_suffix(f16* h, const f16* reg, long long nP, long long D) {
    const long long e = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (e >= 5 * D) return;
    const long long r = e / D, j = e % D;
    h[(nP + r) * D + j] = r < 4 ? reg[r * D + j] : __float2half_rn(0.0f);
}

// The 3-D RoPE rotation table, [S, 24, 4] fp16 for a tile of T x H x W latent tokens followed by nsuf zero-id tokens.
//   create_token_ids (fp16): axis i of n: q = half((i + 0.5) * (1 / n)) (torch divides a tensor by a Python scalar as
//   a multiplication by the fp32 reciprocal), id = half(half(2 * q) - 1); suffix tokens have id 0.
//   RotaryEmbeddingND: angle = ((2 pi as fp32) * float(id)) * float(inv_freq[k]) (inv_freq is the fp16-rounded buffer,
//   8 values), pair index = axis * 8 + k; the pair's four entries (c, -s, s, c) = half(cosf), half(-sinf), half(sinf),
//   half(cosf). launch: block 256, grid ceil(S * 24 / 256).
__device__ __forceinline__ float vv_coord(int i, int n) {
    const float q = vv_h(__fmul_rn((float)i + 0.5f, __fdiv_rn(1.0f, (float)n)));
    return vv_h(__fsub_rn(vv_h(__fmul_rn(2.0f, q)), 1.0f));
}
extern "C" __global__ void vv_rope_table(const f16* inv_freq, f16* table, int T, int H, int W, int nsuf) {
    const long long e = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    const long long nP = (long long)T * H * W, S = nP + nsuf;
    if (e >= S * 24) return;
    const long long tok = e / 24;
    const int pair = (int)(e % 24), axis = pair / 8, k = pair % 8;
    float id = 0.0f;
    if (tok < nP) {
        const int x = (int)(tok % W), y = (int)((tok / W) % H), t = (int)(tok / ((long long)W * H));
        id = axis == 0 ? vv_coord(t, T) : axis == 1 ? vv_coord(y, H) : vv_coord(x, W);
    }
    const float ang = __fmul_rn(__fmul_rn(6.2831855f, id), __half2float(inv_freq[k]));
    const float c = cosf(ang), s = sinf(ang);
    f16* o = table + e * 4;
    o[0] = vv_st(c);
    o[1] = vv_st(-s);
    o[2] = vv_st(s);
    o[3] = vv_st(c);
}

// ------------------------------------------------------------------------------------------------ norms

// F.rms_norm(x, (D,), weight, eps) on fp16 rows: y = half(x * rsqrt(mean(x^2) + eps) * w), fp32 inside, one rounding
// (torch's fused kernel). Sum of squares: thread-strided partials, shared-memory tree. y may alias x.
// launch: block 256, grid rows.
extern "C" __global__ void vv_rms_norm(const f16* x, const f16* w, f16* y, long long D, float eps) {
    __shared__ float s[VV_BLOCK];
    const f16* xr = x + (long long)blockIdx.x * D;
    f16* yr = y + (long long)blockIdx.x * D;
    float acc = 0.0f;
    for (long long j = threadIdx.x; j < D; j += VV_BLOCK) {
        const float v = vv_ld(xr, j);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(vv_tree(acc, s), (float)D), eps));
    for (long long j = threadIdx.x; j < D; j += VV_BLOCK) yr[j] = vv_st(__fmul_rn(__fmul_rn(vv_ld(xr, j), r), vv_ld(w, j)));
}

// F.layer_norm(x, (D,), weight, bias, eps) on fp16 rows: mean = sum / D, var = sum((x - mean)^2) / D (two passes, each
// thread-strided partials + the shared-memory tree), y = half(fma((x - mean) * rstd, w, b)). launch: block 256, grid rows.
extern "C" __global__ void vv_layer_norm(const f16* x, const f16* w, const f16* b, f16* y, long long D, float eps) {
    __shared__ float s[VV_BLOCK];
    const f16* xr = x + (long long)blockIdx.x * D;
    f16* yr = y + (long long)blockIdx.x * D;
    float acc = 0.0f;
    for (long long j = threadIdx.x; j < D; j += VV_BLOCK) acc = __fadd_rn(acc, vv_ld(xr, j));
    const float mean = __fdiv_rn(vv_tree(acc, s), (float)D);
    acc = 0.0f;
    for (long long j = threadIdx.x; j < D; j += VV_BLOCK) {
        const float d = __fsub_rn(vv_ld(xr, j), mean);
        acc = __fadd_rn(acc, __fmul_rn(d, d));
    }
    const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(vv_tree(acc, s), (float)D), eps));
    for (long long j = threadIdx.x; j < D; j += VV_BLOCK)
        yr[j] = vv_st(__fmaf_rn(__fmul_rn(__fsub_rn(vv_ld(xr, j), mean), rstd), vv_ld(w, j), vv_ld(b, j)));
}

// ------------------------------------------------------------------------------------------------ attention

// comfy-kitchen v0.2.35 rms_rope_split_half_ (comfy/ldm/minimax/vae.py Attention.forward) for fp16, head dim 64, rot_dim
// 48, in place on the qkv buffer [S, H, 192] (q at +0, k at +64, v at +128 of a head), unit norm scale, one fp16 table
// [S, 24, 4] shared by all heads. One warp a (token, head); four warps a block. As the wheel's rope_kernel:
//   sum = fmaf chain of x^2 over the lane's dims (lane, lane + 32), then a shfl_down tree (16, 8, 4, 2, 1), lane 0's value
//   rrms = rsqrtf(sum / 64 + eps)            (/ 64 is exact whichever way the wheel's --use_fast_math emits the division)
//   x'   = half(float(x) * rrms) [* 1]       rounded to half BEFORE the rotation (the unit scale multiply is exact)
//   y0 = f00 * x0' + f01 * x1', y1 = f10 * x0' + f11 * x1'   fp32, pairs (i, i + 24), i < 24; each rounded to half
//   dims 48 .. 63: x' only.
// The wheel's y0 line is contracted by nvcc (-fmad); VARIANT picks which product is fused: 0 = fma(f00, x0, f01 * x1)
// (the first product fused, nvcc's usual form for a*b + c*d), 1 = fma(f01, x1, f00 * x0), 2 = unfused. 0 is the one the
// engine launches; 1 and 2 exist for the check against the wheel (test_vae_video.py).
// launch: block 128, grid ceil(S * H / 4).
template <int VARIANT>
__device__ __forceinline__ void vv_rope_impl(f16* qkv, const f16* table, long long S, int H, long long row_stride,
                                             int head_stride, float eps) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const long long row = (long long)blockIdx.x * 4 + warp;
    if (row >= S * H) return;
    const long long tok = row / H;
    const int head = (int)(row % H);
    f16* q = qkv + tok * row_stride + (long long)head * head_stride;
    f16* k = q + 64;
    float sums[2];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const f16* p = w == 0 ? q : k;
        float sum = 0.0f;
        for (int e = lane; e < 64; e += 32) {
            const float v = __half2float(p[e]);
            sum = __fmaf_rn(v, v, sum);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, off));
        sums[w] = __shfl_sync(0xffffffffu, sum, 0);
    }
    const float qr = rsqrtf(__fadd_rn(__fmul_rn(sums[0], 0.015625f), eps));
    const float kr = rsqrtf(__fadd_rn(__fmul_rn(sums[1], 0.015625f), eps));
    if (lane < 24) {
        const f16* f = table + (tok * 24 + lane) * 4;
        const float f00 = __half2float(f[0]), f01 = __half2float(f[1]), f10 = __half2float(f[2]), f11 = __half2float(f[3]);
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            f16* p = w == 0 ? q : k;
            const float rr = w == 0 ? qr : kr;
            const float x0 = vv_h(__fmul_rn(__half2float(p[lane]), rr));
            const float x1 = vv_h(__fmul_rn(__half2float(p[lane + 24]), rr));
            float y0, y1;
            if (VARIANT == 0) {
                y0 = __fmaf_rn(f00, x0, __fmul_rn(f01, x1));
                y1 = __fmaf_rn(f10, x0, __fmul_rn(f11, x1));
            } else if (VARIANT == 1) {
                y0 = __fmaf_rn(f01, x1, __fmul_rn(f00, x0));
                y1 = __fmaf_rn(f11, x1, __fmul_rn(f10, x0));
            } else {
                y0 = __fadd_rn(__fmul_rn(f00, x0), __fmul_rn(f01, x1));
                y1 = __fadd_rn(__fmul_rn(f10, x0), __fmul_rn(f11, x1));
            }
            p[lane] = vv_st(y0);
            p[lane + 24] = vv_st(y1);
        }
    }
    // the norm-only tail (dims 48 .. 63): 16 elements a head, one lane each
    if (lane < 16) {
        q[48 + lane] = vv_st(__fmul_rn(__half2float(q[48 + lane]), qr));
        k[48 + lane] = vv_st(__fmul_rn(__half2float(k[48 + lane]), kr));
    }
}
extern "C" __global__ void __launch_bounds__(128) vv_rms_rope(f16* qkv, const f16* table, long long S, int H,
                                                              long long row_stride, int head_stride, float eps) {
    vv_rope_impl<0>(qkv, table, S, H, row_stride, head_stride, eps);
}
extern "C" __global__ void __launch_bounds__(128) vv_rms_rope_v1(f16* qkv, const f16* table, long long S, int H,
                                                                 long long row_stride, int head_stride, float eps) {
    vv_rope_impl<1>(qkv, table, S, H, row_stride, head_stride, eps);
}
extern "C" __global__ void __launch_bounds__(128) vv_rms_rope_v2(f16* qkv, const f16* table, long long S, int H,
                                                                 long long row_stride, int head_stride, float eps) {
    vv_rope_impl<2>(qkv, table, S, H, row_stride, head_stride, eps);
}

// V transposed with zero padding, the B operand of P . V: qkv [S, H, 192] -> vt [H, 64, SP], vt[h][d][s] = v[s][h][d]
// for s < S, 0 for S <= s < SP. launch: block 256, grid ceil(H * 64 * SP / 256).
extern "C" __global__ void vv_vt(const f16* qkv, f16* vt, long long S, long long SP, int H, long long row_stride,
                                 int head_stride) {
    const long long e = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (e >= (long long)H * 64 * SP) return;
    const long long s = e % SP;
    const int d = (int)((e / SP) % 64), h = (int)(e / (SP * 64));
    vt[e] = s < S ? qkv[s * row_stride + (long long)h * head_stride + 128 + d] : __float2half_rn(0.0f);
}

// Row softmax of the attention scores in place: scores [rows, SP] fp16 (rows = heads * S, the first S columns real):
//   x_j = float(s_j) * scale;  m = max x_j;  e_j = expf(x_j - m);  P_j = half(e_j / sum_j e_j),  columns S .. SP-1 = 0.
// scale 1/8 is exact (SDPA's 1 / sqrt(64)); the sum is per-thread partials (thread t takes j = t, t + 256, ... ascending),
// an xor-shuffle tree inside each warp (offsets 16, 8, 4, 2, 1), then the 8 warp sums added in warp order.
// launch: block 256, grid rows.
extern "C" __global__ void vv_softmax(f16* s, long long S, long long SP, float scale) {
    __shared__ float red[8];
    __shared__ float bcast;
    f16* p = s + (long long)blockIdx.x * SP;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float m = -INFINITY;
    for (long long j = t; j < S; j += VV_BLOCK) m = fmaxf(m, vv_ld(p, j));
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
    for (long long j = t; j < S; j += VV_BLOCK) sum = __fadd_rn(sum, expf(__fmul_rn(__fsub_rn(vv_ld(p, j), m), scale)));
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
    for (long long j = t; j < SP; j += VV_BLOCK) {
        float v = 0.0f;
        if (j < S) v = __fdiv_rn(expf(__fmul_rn(__fsub_rn(vv_ld(p, j), m), scale)), denom);
        p[j] = vv_st(v);
    }
}

// torch.nan_to_num on the attention output, in place: NaN -> 0, +inf -> 65504, -inf -> -65504 (the half extremes).
// launch: block 256, grid ceil(n / 256).
extern "C" __global__ void vv_nan_to_num(f16* x, long long n) {
    const long long i = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = vv_ld(x, i);
    if (v != v) x[i] = __float2half_rn(0.0f);
    else if (v > 65504.0f) x[i] = __float2half_rn(65504.0f);
    else if (v < -65504.0f) x[i] = __float2half_rn(-65504.0f);
}

// ------------------------------------------------------------------------------------------------ MLP, head

// ComfyUI's swiglu (linear_input_act, F.silu(gate).mul_(up)): [M, 2F] = [gate | up] -> half(half(silu(gate)) * up), the
// SiLU (g / (1 + expf(-g)), fp32) rounded to half first as torch's fp16 ops do. launch: block 256, grid (ceil(F / 256), M).
extern "C" __global__ void vv_swiglu(const f16* gu, f16* out, long long F) {
    const long long j = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (j >= F) return;
    const long long r = blockIdx.y;
    const float g = vv_ld(gu, r * 2 * F + j);
    const float sl = vv_h(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))));
    out[r * F + j] = vv_st(__fmul_rn(sl, vv_ld(gu, r * 2 * F + F + j)));
}

// The decoder's last view / permute: proj_out rows [>= T*H*W, 3 * 4 * 16 * 16] -> tile [3, 4T, 16H, 16W],
// out[c][4t + pt][16y + py][16x + px] = rows[(t * H + y) * W + x][((c * 4 + pt) * 16 + py) * 16 + px], a pure data move.
// launch: block 256, grid ceil(3 * 4T * 16H * 16W / 256).
extern "C" __global__ void vv_unshuffle(const f16* rows, f16* out, int T, int H, int W) {
    const long long e = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    const long long PW = 16LL * W, PH = 16LL * H, F = 4LL * T;
    if (e >= 3 * F * PH * PW) return;
    const int xo = (int)(e % PW), yo = (int)((e / PW) % PH), fo = (int)((e / (PW * PH)) % F), c = (int)(e / (PW * PH * F));
    const int t = fo / 4, pt = fo % 4, y = yo / 16, py = yo % 16, x = xo / 16, px = xo % 16;
    const long long row = ((long long)t * H + y) * W + x;
    out[e] = rows[row * 3072 + ((c * 4 + pt) * 16 + py) * 16 + px];
}

// ------------------------------------------------------------------------------------------------ tiling, blending

// ComfyUI blend() on two fp16 values at position p of an overlap of `ext`: positions / ext is a multiplication by the
// fp32 reciprocal (rounded to half = w_b), w_a = half(1 - w_b); a * w_a, b * w_b each rounded to half (torch ops), the
// sum rounded to half.
__device__ __forceinline__ float vv_blend(float a, float b, int p, float inv_ext) {
    const float wb = vv_h(__fmul_rn((float)p, inv_ext));
    const float wa = vv_h(__fsub_rn(1.0f, wb));
    return __fadd_rn(vv_h(__fmul_rn(a, wa)), vv_h(__fmul_rn(b, wb)));
}

// One decoded tile into the clip canvas, as tiled_decode does it: b = the raw tile [3, F, th, tw]; if ytail: rows
// r < ey are blended (ytail = the raw tile above, its last ey rows, along y with weights at position r), then if ltail:
// columns c < ex are blended (ltail = the raw tile to the left, its last ex columns, taken UN-blended, with the tile
// as the y blend left it); the region [0, oh) x [0, ow) (the tile less its trailing overlaps) is written to the canvas
// [3, F, Hc, Wc] at (oy, ox). launch: block 256, grid ceil(3 * F * oh * ow / 256).
extern "C" __global__ void vv_place_tile(const f16* b, int F, int th, int tw, const f16* ytail, int tha, int twa, int ey,
                                         const f16* ltail, int thl, int twl, int ex, f16* canvas, int Hc, int Wc, int oy,
                                         int ox, int oh, int ow) {
    const long long e = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (e >= 3LL * F * oh * ow) return;
    const int col = (int)(e % ow), r = (int)((e / ow) % oh);
    const int f = (int)((e / ((long long)ow * oh)) % F), c = (int)(e / ((long long)ow * oh * F));
    const long long cf = (long long)c * F + f;
    float v = __half2float(b[(cf * th + r) * tw + col]);
    if (ytail && r < ey) {
        const float a = __half2float(ytail[(cf * tha + (tha - ey) + r) * twa + col]);
        v = vv_h(vv_blend(a, v, r, __fdiv_rn(1.0f, (float)ey)));
    }
    if (ltail && col < ex) {
        const float a = __half2float(ltail[(cf * thl + r) * twl + (twl - ex) + col]);
        v = vv_h(vv_blend(a, v, col, __fdiv_rn(1.0f, (float)ex)));
    }
    canvas[(cf * Hc + oy + r) * Wc + ox + col] = vv_st(v);
}

// write_part(): the temporal blend of the carried overlap, _finalize_pixels and the frame conversion in one pass.
// A part is nb frames [b0, b0 + nb) of canvas b [3, Fb, H, W]; with ext > 0 its first ext frames are blended with the
// carried frames [a0, a0 + ext) of canvas a [3, Fa, H, W] (blend(a, b, ext, dim=-3), position = frame). The first
// `copy` frames are finalised and written at output frame `pos`:
//   v = half(blend) (or the plain half);  p = clamp(float(v) * std_c + mean_c, 0, 1)   (fp32 ops: a multiply, then an
//   add, std / mean the fp16-rounded ImageNet constants as fp32; the clamp keeps NaN);  u8 = trunc(clamp(p * 255, 0, 255))
//   (VideoFromComponents.save_to: (frame * 255).clamp(0, 255).byte(); a NaN becomes 0).
// Outputs: u8 [Ftot, H, W, 3] and / or fp32 [Ftot, H, W, 3] (null = skip). launch: block 256, grid ceil(copy * H * W * 3 / 256).
extern "C" __global__ void vv_finalize(const f16* a, int Fa, int a0, const f16* b, int Fb, int b0, int ext, int copy, int H,
                                       int W, float s0, float s1, float s2, float m0, float m1, float m2, uint8_t* out_u8,
                                       float* out_f32, int pos) {
    const long long e = (long long)blockIdx.x * VV_BLOCK + threadIdx.x;
    if (e >= (long long)copy * H * W * 3) return;
    const int c = (int)(e % 3);
    const long long px = e / 3;
    const int x = (int)(px % W), y = (int)((px / W) % H), f = (int)(px / ((long long)W * H));
    float v = __half2float(b[(((long long)c * Fb + b0 + f) * H + y) * W + x]);
    if (a && f < ext) {
        const float av = __half2float(a[(((long long)c * Fa + a0 + f) * H + y) * W + x]);
        v = vv_h(vv_blend(av, v, f, __fdiv_rn(1.0f, (float)ext)));
    }
    float p = __fadd_rn(__fmul_rn(v, c == 0 ? s0 : c == 1 ? s1 : s2), c == 0 ? m0 : c == 1 ? m1 : m2);
    p = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
    const long long o = ((long long)(pos + f) * H * W + (long long)y * W + x) * 3 + c;
    if (out_f32) out_f32[o] = p;
    if (out_u8) {
        float u = __fmul_rn(p, 255.0f);
        u = u < 0.0f ? 0.0f : (u > 255.0f ? 255.0f : u);
        out_u8[o] = (u != u) ? (uint8_t)0 : (uint8_t)(int)u;
    }
}
