// Ours: MiniMax H3's DiT ops besides its GEMMs and attention, shared by the twin (torch extension) and the Zig engine
// (fatbin). extern "C", raw arguments, deterministic: every reduction in a fixed order, every rounding written out.
// tfvideo's block arithmetic (fp32 inside, one bf16 rounding a value) and ComfyUI's ops around the blocks.
#include <cuda_bf16.h>
#include <stdint.h>

#define H3_BLOCK 256

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float h3_ld(const bf16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ bf16 h3_st(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float h3_bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// The row's sum of squares in a fixed order: thread-strided partials, then a shared-memory tree.
__device__ __forceinline__ float h3_sumsq(const bf16* xr, long long D, float* s) {
    float acc = 0.0f;
    for (long long j = threadIdx.x; j < D; j += H3_BLOCK) {
        const float v = h3_ld(xr, j);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = H3_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = __fadd_rn(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    return s[0];
}

// tfvideo's norm_mod: y = (x * rsqrt(mean(x^2) + eps) * w) * (1 + scale) + shift, fp32, one rounding. x [S, D] bf16,
// w fp32 [D], mod fp32 [R, 6, D] (shift, scale, gate for attention, then for the MLP), idx int32 [S] (the row's
// modulation row), part 0 (attention: chunks 0, 1) or 1 (MLP: 3, 4). y may alias x. launch: block 256, grid S.
extern "C" __global__ void h3_norm_mod(const bf16* x, const float* w, const float* mod, const int* idx, bf16* y,
                                       long long D, int part, float eps) {
    __shared__ float s[H3_BLOCK];
    const bf16* xr = x + (long long)blockIdx.x * D;
    bf16* yr = y + (long long)blockIdx.x * D;
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(h3_sumsq(xr, D, s), (float)D), eps));
    const float* m = mod + (long long)idx[blockIdx.x] * 6 * D;
    const float* shift = m + (3 * part) * D;
    const float* scale = m + (3 * part + 1) * D;
    for (long long j = threadIdx.x; j < D; j += H3_BLOCK) {
        const float n = __fmul_rn(__fmul_rn(h3_ld(xr, j), r), w[j]);
        yr[j] = h3_st(__fadd_rn(__fmul_rn(n, __fadd_rn(1.0f, scale[j])), shift[j]));
    }
}

// tfvideo's gate_add: x = bf16(x + y * gate), in place, gate from chunk 2 (part 0) or 5 (part 1) of the row's
// modulation row. launch: block 256, grid (ceil(D / 256), S).
extern "C" __global__ void h3_gate_add(bf16* x, const bf16* y, const float* mod, const int* idx, long long D, int part) {
    const long long j = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (j >= D) return;
    const long long o = (long long)blockIdx.y * D + j;
    const float g = mod[(long long)idx[blockIdx.y] * 6 * D + (3 * part + 2) * D + j];
    x[o] = h3_st(__fadd_rn(h3_ld(x, o), __fmul_rn(h3_ld(y, o), g)));
}

// tfvideo's swiglu: [M, 2F] = [gate | up] -> bf16(gate / (1 + exp(-gate)) * up), fp32. launch: block 256,
// grid (ceil(F / 256), M).
extern "C" __global__ void h3_swiglu(const bf16* gu, bf16* out, long long F) {
    const long long j = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (j >= F) return;
    const long long r = blockIdx.y;
    const float g = h3_ld(gu, r * 2 * F + j), u = h3_ld(gu, r * 2 * F + F + j);
    out[r * F + j] = h3_st(__fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u));
}

// RMSNorm with a bf16 weight (ComfyUI's RMSNorm on bf16 rows): y = bf16(x * rsqrt(mean(x^2) + eps) * w), fp32, one
// rounding. Rows of D (5376, or 128 per head). y may alias x. launch: block 256, grid rows.
extern "C" __global__ void h3_rms_norm(const bf16* x, const bf16* w, bf16* y, long long D, float eps) {
    __shared__ float s[H3_BLOCK];
    const bf16* xr = x + (long long)blockIdx.x * D;
    bf16* yr = y + (long long)blockIdx.x * D;
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(h3_sumsq(xr, D, s), (float)D), eps));
    for (long long j = threadIdx.x; j < D; j += H3_BLOCK) yr[j] = h3_st(__fmul_rn(__fmul_rn(h3_ld(xr, j), r), h3_ld(w, j)));
}

// The final layer's modulation of one target segment: out = bf16(rms_norm(x) * w) * (1 + scale) + shift in fp32
// (the bf16 norm promoted by the fp32 curve modulation), rows [n, D] -> fp32 [n, D]; scale / shift fp32 [D] (the
// segment's modulation row). launch: block 256, grid n.
extern "C" __global__ void h3_final_mod(const bf16* x, const bf16* w, const float* scale, const float* shift, float* out,
                                        long long D, float eps) {
    __shared__ float s[H3_BLOCK];
    const bf16* xr = x + (long long)blockIdx.x * D;
    float* orow = out + (long long)blockIdx.x * D;
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(h3_sumsq(xr, D, s), (float)D), eps));
    for (long long j = threadIdx.x; j < D; j += H3_BLOCK) {
        const float n = h3_bf(__fmul_rn(__fmul_rn(h3_ld(xr, j), r), h3_ld(w, j)));
        orow[j] = __fadd_rn(__fmul_rn(n, __fadd_rn(1.0f, scale[j])), shift[j]);
    }
}

// y = bf16(x * c) for a bf16-valued scalar c (the audio carry). y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void h3_scale(const bf16* x, float c, bf16* y, long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i < n) y[i] = h3_st(__fmul_rn(h3_ld(x, i), c));
}

// The audio velocity back on the carried variable: v = bf16(bf16(c1 * a) + bf16(c2 * v)), a = the carried audio
// input, c1 = 1 - scale (a Python float), c2 = bf16(1 + (scale - 1) * sigma_a). launch: block 256, grid ceil(n / 256).
extern "C" __global__ void h3_uncarry(const bf16* a, bf16* v, float c1, float c2, long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i < n) v[i] = h3_st(__fadd_rn(h3_bf(__fmul_rn(c1, h3_ld(a, i))), h3_bf(__fmul_rn(c2, h3_ld(v, i)))));
}

// patchify_video: x bf16 [C, T, H, W] -> rows fp32 [T * (H/2) * (W/2), C * 4], row (t, h, w), column c * 4 + p * 2 + q,
// value x[c][t][2h + p][2w + q]. launch: block 256, grid ceil(rows * C * 4 / 256).
extern "C" __global__ void h3_patchify(const bf16* x, float* rows, int C, int T, int H, int W) {
    const long long e = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    const int h2 = H / 2, w2 = W / 2, cols = C * 4;
    const long long n = (long long)T * h2 * w2 * cols;
    if (e >= n) return;
    const long long row = e / cols;
    const int col = (int)(e % cols), c = col / 4, p = (col / 2) % 2, q = col % 2;
    const int ww = (int)(row % w2), hh = (int)((row / w2) % h2), t = (int)(row / ((long long)w2 * h2));
    rows[e] = h3_ld(x, (((long long)c * T + t) * H + 2 * hh + p) * W + 2 * ww + q);
}

// unpatchify_video of the negated head output: rows fp32 [T * (H/2) * (W/2), C * 4] -> bf16 [C, T, H, W],
// value bf16(-rows[...]). launch: block 256, grid ceil(C * T * H * W / 256).
extern "C" __global__ void h3_unpatchify_neg(const float* rows, bf16* x, int C, int T, int H, int W) {
    const long long e = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (e >= (long long)C * T * H * W) return;
    const int xw = (int)(e % W), yh = (int)((e / W) % H), t = (int)((e / ((long long)W * H)) % T);
    const int c = (int)(e / ((long long)W * H * T));
    const long long row = ((long long)t * (H / 2) + yh / 2) * (W / 2) + xw / 2;
    x[e] = h3_st(-rows[row * C * 4 + c * 4 + (yh % 2) * 2 + xw % 2]);
}

// pack_audio: a bf16 [C, 2, A] -> rows fp32 [2A, C], row ch * A + t, column c (channel-major stereo).
// launch: block 256, grid ceil(2 * A * C / 256).
extern "C" __global__ void h3_pack_audio(const bf16* a, float* rows, int C, int A) {
    const long long e = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (e >= 2LL * A * C) return;
    const int c = (int)(e % C);
    const long long r = e / C;
    rows[e] = h3_ld(a, ((long long)c * 2 + r / A) * A + r % A);
}

// unpack_audio of the negated head output: rows fp32 [2A, C] -> bf16 [C, 2, A], value bf16(-rows[...]).
// launch: block 256, grid ceil(C * 2 * A / 256).
extern "C" __global__ void h3_unpack_audio_neg(const float* rows, bf16* a, int C, int A) {
    const long long e = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (e >= 2LL * A * C) return;
    const int c = (int)(e / (2LL * A));
    const long long r = e % (2LL * A);
    a[e] = h3_st(-rows[r * C + c]);
}

// x [rows, D] bf16 -> y = bf16(float(x) + float(z)), the refiner's in-place residual (y may alias x or z).
// launch: block 256, grid ceil(n / 256).
extern "C" __global__ void h3_add(const bf16* x, const bf16* z, bf16* y, long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i < n) y[i] = h3_st(__fadd_rn(h3_ld(x, i), h3_ld(z, i)));
}

// fp32 [M, N] = A fp32 [M, K] . B fp32 [N, K]^T (+ bias[N]): each output an fmaf chain over k ascending from 0, then
// + bias; tiles of 64 x 64 (a thread 4 x 4), K tiles of 16 in shared memory. An output's bits depend only on its two
// rows. launch: block 256, grid (ceil(N / 64), ceil(M / 64)).
extern "C" __global__ void h3_gemm_f32(const float* A, const float* B, const float* bias, float* Cm, long long M,
                                       long long N, long long K) {
    __shared__ float sa[16][65];
    __shared__ float sb[16][65];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const long long m0 = (long long)blockIdx.y * 64, n0 = (long long)blockIdx.x * 64;
    float acc[4][4] = {};
    for (long long k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 64 * 16; i += H3_BLOCK) {
            const int r = i / 16, kk = i % 16;
            const long long ka = k0 + kk;
            sa[kk][r] = (m0 + r < M && ka < K) ? A[(m0 + r) * K + ka] : 0.0f;
            sb[kk][r] = (n0 + r < N && ka < K) ? B[(n0 + r) * K + ka] : 0.0f;
        }
        __syncthreads();
        const int kn = (int)min((long long)16, K - k0);
        for (int kk = 0; kk < kn; ++kk) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = __fmaf_rn(sa[kk][ty * 4 + i], sb[kk][tx * 4 + j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long long m = m0 + ty * 4 + i, n = n0 + tx * 4 + j;
            if (m < M && n < N) Cm[m * N + n] = bias ? __fadd_rn(acc[i][j], bias[n]) : acc[i][j];
        }
}

// x fp32 [n] -> bf16 (round to nearest even). launch: block 256, grid ceil(n / 256).
extern "C" __global__ void h3_f32_to_bf16(const float* x, bf16* y, long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i < n) y[i] = h3_st(x[i]);
}

// ComfyUI's swiglu (linear_input_act, the token refiner): [M, 2F] = [gate | up] -> bf16(bf16(silu(gate)) * up), the
// SiLU rounded to bf16 first as torch's bf16 ops do. launch: block 256, grid (ceil(F / 256), M).
extern "C" __global__ void h3_silu_mul_split(const bf16* gu, bf16* out, long long F) {
    const long long j = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (j >= F) return;
    const long long r = blockIdx.y;
    const float g = h3_ld(gu, r * 2 * F + j);
    const float sl = h3_bf(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))));
    out[r * F + j] = h3_st(__fmul_rn(sl, h3_ld(gu, r * 2 * F + F + j)));
}

// [H, S, D] (an attention output, one head after another) -> [S, H * D] rows, bf16, a pure data move.
// launch: block 256, grid ceil(H * S * D / 256).
extern "C" __global__ void h3_heads_to_rows(const bf16* x, bf16* y, long long H, long long S, long long D) {
    const long long e = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (e >= H * S * D) return;
    const long long d = e % D, s = (e / D) % S, h = e / (D * S);
    y[(s * H + h) * D + d] = x[e];
}

// ---------------------------------------------------------------------------------------------- the sampler
// res_multistep (eta 0) on the packed fp32 state x [N] (video then audio), every elementwise op one fp32 rounding as
// ComfyUI's separate torch kernels: denoised = x - float(out) * sigma (out: the model's bf16 velocities, packed alike).
// launch (all four): block 256, grid ceil(n / 256).
extern "C" __global__ void h3_denoise(const float* x, const bf16* out, float sigma, float* den, long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i < n) den[i] = __fsub_rn(x[i], __fmul_rn(h3_ld(out, i), sigma));
}

// Euler (the first and the last step): d = (x - den) / sigma; x = x + d * dt.
extern "C" __global__ void h3_euler32(float* x, const float* den, float sigma, float dt, long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float d = __fdiv_rn(__fsub_rn(x[i], den[i]), sigma);
    x[i] = __fadd_rn(x[i], __fmul_rn(d, dt));
}

// The second-order step: x = e * x + h * (b1 * den + b2 * old).
extern "C" __global__ void h3_res2(float* x, const float* den, const float* old, float e, float h, float b1, float b2,
                                   long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float s = __fadd_rn(__fmul_rn(b1, den[i]), __fmul_rn(b2, old[i]));
    x[i] = __fadd_rn(__fmul_rn(e, x[i]), __fmul_rn(h, s));
}

// y = x * c, fp32 (process_latent_out's audio * 0.25). y may alias x.
extern "C" __global__ void h3_scale32(const float* x, float c, float* y, long long n) {
    const long long i = (long long)blockIdx.x * H3_BLOCK + threadIdx.x;
    if (i < n) y[i] = __fmul_rn(x[i], c);
}
