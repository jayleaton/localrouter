// Ours: the kernels MiniMax H3-Image adds to the H3 ones (docs/dev/H3-IMAGE.md), shared by the twin and the Zig engine:
// the DiT's PDD head mix (comfy/ldm/minimax/model.py _pdd_head, ComfyUI e9afbccc) and the image VAE decoder's 2-D RoPE,
// its split-half RMSNorm + RoPE over half the head, its 3 x 16 x 16 unpatchify and its tile blend and copies
// (comfy/ldm/minimax/vae.py ViT3DDecoder with rope_dims 2, rope_dim_ratio 0.5, patch_size_t 1; tiled_decode). The other
// decoder ops are vae_video.cu's and gemm_f16.cu's, unchanged. extern "C", raw arguments, deterministic: every reduction
// in a fixed order, every rounding written out (__fmul_rn / __fadd_rn), so contraction flags do not reach the values.
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#define HI_BLOCK 256

typedef __half f16;

__device__ __forceinline__ f16 hi_st(float v) { return __float2half_rn(v); }
__device__ __forceinline__ float hi_h(float v) { return __half2float(__float2half_rn(v)); }  // round to half, back

// ------------------------------------------------------------------------------------------------ DiT: PDD heads

// The step's head of a PDD bank (stk_twin/h3i/sampler.py `mix`): `bank` is [heads of `block` fp32 values each] (the
// weight's 96 x K rows, or the bias's 96), out = bank[0] when nheads is 0, else s = w0 * bank[h0]; s = s + wk * bank[hk]
// for k = 1 .. nheads - 1 (each product and sum rounded to fp32), out = bank[0] + s. The heads and weights are scalar
// arguments (at most 6, the checkpoint's pdd_steps). launch: block 256, grid ceil(block / 256).
extern "C" __global__ void h3i_head_mix(const float* bank, float* out, long long block, int nheads, int h0, int h1, int h2,
                                        int h3, int h4, int h5, float w0, float w1, float w2, float w3, float w4, float w5) {
    const long long e = (long long)blockIdx.x * HI_BLOCK + threadIdx.x;
    if (e >= block) return;
    const int hs[6] = {h0, h1, h2, h3, h4, h5};
    const float ws[6] = {w0, w1, w2, w3, w4, w5};
    if (nheads == 0) {
        out[e] = bank[e];
        return;
    }
    float s = __fmul_rn(ws[0], bank[(long long)hs[0] * block + e]);
    for (int k = 1; k < nheads; ++k) s = __fadd_rn(s, __fmul_rn(ws[k], bank[(long long)hs[k] * block + e]));
    out[e] = __fadd_rn(bank[e], s);
}

// ------------------------------------------------------------------------------------------------ VAE: RoPE

// create_token_ids in fp16 for axis position i of n (vae_video.cu's vv_coord): q = half((i + 0.5) * (1 / n)),
// id = half(half(2 * q) - 1).
__device__ __forceinline__ float hi_coord(int i, int n) {
    const float q = hi_h(__fmul_rn((float)i + 0.5f, __fdiv_rn(1.0f, (float)n)));
    return hi_h(__fsub_rn(hi_h(__fmul_rn(2.0f, q)), 1.0f));
}

// The 2-D RoPE rotation table, [S, 16, 4] fp16 for a tile of H x W latent tokens followed by nsuf zero-id tokens:
// RotaryEmbeddingND(32, 100, 2) on ids (y, x), pair index = axis * 8 + k, angle = ((2 pi as fp32) * float(id)) *
// float(inv_freq[k]) (the fp16 buffer, 8 values), entries (c, -s, s, c) = half(cosf), half(-sinf), half(sinf), half(cosf):
// vv_rope_table's arithmetic on two axes. launch: block 256, grid ceil(S * 16 / 256).
extern "C" __global__ void vi_rope_table(const f16* inv_freq, f16* table, int H, int W, int nsuf) {
    const long long e = (long long)blockIdx.x * HI_BLOCK + threadIdx.x;
    const long long nP = (long long)H * W, S = nP + nsuf;
    if (e >= S * 16) return;
    const long long tok = e / 16;
    const int pair = (int)(e % 16), axis = pair / 8, k = pair % 8;
    float id = 0.0f;
    if (tok < nP) {
        const int x = (int)(tok % W), y = (int)(tok / W);
        id = axis == 0 ? hi_coord(y, H) : hi_coord(x, W);
    }
    const float ang = __fmul_rn(__fmul_rn(6.2831855f, id), __half2float(inv_freq[k]));
    const float c = cosf(ang), s = sinf(ang);
    f16* o = table + e * 4;
    o[0] = hi_st(c);
    o[1] = hi_st(-s);
    o[2] = hi_st(s);
    o[3] = hi_st(c);
}

// comfy-kitchen's rms_rope_split_half_ for fp16, head dim 64, rot_dim 32, in place on the qkv buffer [S, H, 192] (q at
// +0, k at +64, v at +128 of a head), unit norm scale, one fp16 table [S, 16, 4]: vae_video.cu's vv_rope_impl<0> (its
// comment has the wheel's order) with 16 pairs (i, i + 16), i < 16, and the norm-only tail at dims 32 .. 63.
// launch: block 128, grid ceil(S * H / 4).
extern "C" __global__ void __launch_bounds__(128) vi_rms_rope(f16* qkv, const f16* table, long long S, int H,
                                                              long long row_stride, int head_stride, float eps) {
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
    if (lane < 16) {
        const f16* f = table + (tok * 16 + lane) * 4;
        const float f00 = __half2float(f[0]), f01 = __half2float(f[1]), f10 = __half2float(f[2]), f11 = __half2float(f[3]);
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            f16* p = w == 0 ? q : k;
            const float rr = w == 0 ? qr : kr;
            const float x0 = hi_h(__fmul_rn(__half2float(p[lane]), rr));
            const float x1 = hi_h(__fmul_rn(__half2float(p[lane + 16]), rr));
            const float y0 = __fmaf_rn(f00, x0, __fmul_rn(f01, x1));
            const float y1 = __fmaf_rn(f10, x0, __fmul_rn(f11, x1));
            p[lane] = hi_st(y0);
            p[lane + 16] = hi_st(y1);
        }
    }
    // the norm-only tail (dims 32 .. 63): 32 elements a head, one lane each
    q[32 + lane] = hi_st(__fmul_rn(__half2float(q[32 + lane]), qr));
    k[32 + lane] = hi_st(__fmul_rn(__half2float(k[32 + lane]), kr));
}

// ------------------------------------------------------------------------------------------------ VAE: head, tiles

// The decoder's unpatchify with patch_size_t 1: proj_out rows [>= H*W, 3 * 16 * 16] -> tile [3, 16H, 16W],
// out[c][16y + py][16x + px] = rows[y * W + x][(c * 16 + py) * 16 + px], a pure data move.
// launch: block 256, grid ceil(3 * 16H * 16W / 256).
extern "C" __global__ void vi_unshuffle(const f16* rows, f16* out, int H, int W) {
    const long long e = (long long)blockIdx.x * HI_BLOCK + threadIdx.x;
    const long long PW = 16LL * W, PH = 16LL * H;
    if (e >= 3 * PH * PW) return;
    const int xo = (int)(e % PW), yo = (int)((e / PW) % PH), c = (int)(e / (PW * PH));
    const int y = yo / 16, py = yo % 16, x = xo / 16, px = xo % 16;
    out[e] = rows[((long long)y * W + x) * 768 + (c * 16 + py) * 16 + px];
}

// tiled_decode's blend(a, b, e, dim) in place on the tile b [3, th, tw] (fp16, vv_blend's arithmetic): along y (dim 0)
// rows r < e, along x (dim 1) columns c < e, are a * (1 - p / e) + b * (p / e) at position p (r or c), with p / e the
// multiplication by the fp32 reciprocal rounded to half (w_b), w_a = half(1 - w_b), each product rounded to half and the
// sum rounded to half; a's values at (ay + r, ax + c) of a [3, ah, aw] (a's last e rows or columns, as the caller places
// them). launch: block 256, grid ceil(3 * (dim 0: e * tw, dim 1: th * e) / 256).
extern "C" __global__ void vi_blend(f16* b, int th, int tw, const f16* a, int ah, int aw, int ay, int ax, int e, int dim) {
    const long long i = (long long)blockIdx.x * HI_BLOCK + threadIdx.x;
    const int rows = dim == 0 ? e : th, cols = dim == 0 ? tw : e;
    if (i >= 3LL * rows * cols) return;
    const int col = (int)(i % cols), r = (int)((i / cols) % rows), c = (int)(i / ((long long)cols * rows));
    const long long bi = ((long long)c * th + r) * tw + col;
    const float av = __half2float(a[((long long)c * ah + ay + r) * aw + ax + col]);
    const float bv = __half2float(b[bi]);
    const int p = dim == 0 ? r : col;
    const float wb = hi_h(__fmul_rn((float)p, __fdiv_rn(1.0f, (float)e)));
    const float wa = hi_h(__fsub_rn(1.0f, wb));
    b[bi] = hi_st(__fadd_rn(hi_h(__fmul_rn(av, wa)), hi_h(__fmul_rn(bv, wb))));
}

// A window [h, w] of src [3, sh, sw] at (sy, sx) to dst [3, dh, dw] at (dy, dx), a pure data move (the tile's crops into
// the canvas, the next row's strip and the left neighbour's tail). launch: block 256, grid ceil(3 * h * w / 256).
extern "C" __global__ void vi_copy(const f16* src, int sh, int sw, int sy, int sx, f16* dst, int dh, int dw, int dy, int dx,
                                   int h, int w) {
    const long long i = (long long)blockIdx.x * HI_BLOCK + threadIdx.x;
    if (i >= 3LL * h * w) return;
    const int col = (int)(i % w), r = (int)((i / w) % h), c = (int)(i / ((long long)w * h));
    dst[((long long)c * dh + dy + r) * dw + dx + col] = src[((long long)c * sh + sy + r) * sw + sx + col];
}
