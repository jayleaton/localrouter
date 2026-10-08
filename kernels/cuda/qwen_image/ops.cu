// Small Qwen-Image ops shared by the Python twin (torch extension) and the Zig engine (fatbin, looked up by name).
// fp32 inside, bf16 at the edges, no fast-math, no atomics; every reduction runs in a fixed order.
// Where torch's own kernel does something specific (mul by reciprocal for a CPU-scalar divisor, FMA contraction in
// addcmul / gelu) the same operation is written out explicitly here, so the twin keeps its bits.
// Build: nvcc -O3 -gencode=arch=compute_120a,code=sm_120a (and 121a). No -use_fast_math.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define STK_BLOCK 256

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float ldf(const bf16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ bf16 stf(float v) { return __float2bfloat16_rn(v); }

// E4M3_MAX and the clamp floor of Fp8Linear._act; torch divides by a CPU scalar as a multiply by its fp32 reciprocal.
__device__ __forceinline__ float stk_act_scale(float absmax) {
    const float inv448 = 1.0f / 448.0f;
    return fmaxf(__fmul_rn(absmax, inv448), 1e-12f);
}

// x[B,N,D] += a[B,N,D] * g[B,1,D], in place (torch addcmul_: one fused multiply-add in fp32).
// launch: block 256, grid ceil(B*N*D / 256); n_per_batch = N*D.
extern "C" __global__ void stk_gated_residual(bf16* x, const bf16* a, const bf16* g, long long total,
                                              long long n_per_batch, long long D) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= total) return;
    long long b = i / n_per_batch, d = i % D;
    x[i] = stf(fmaf(ldf(a, i), ldf(g, b * D + d), ldf(x, i)));
}

// y = silu(x) = x / (1 + expf(-x)); y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void stk_silu(const bf16* x, bf16* y, long long n) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= n) return;
    float v = ldf(x, i);
    y[i] = stf(__fdiv_rn(v, __fadd_rn(1.0f, expf(-v))));
}

// y = tanhf(x); y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void stk_tanh(const bf16* x, bf16* y, long long n) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= n) return;
    y[i] = stf(tanhf(ldf(x, i)));
}

// y = gelu(x, approximate="tanh") with torch's operation order; y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void stk_gelu_tanh(const bf16* x, bf16* y, long long n) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float kBeta = (float)(1.4142135623730951 * 1.1283791670955126 * 0.5);
    const float kKappa = 0.044715f;
    float v = ldf(x, i);
    float cube = __fmul_rn(__fmul_rn(v, v), v);
    float inner = __fmul_rn(kBeta, fmaf(kKappa, cube, v));
    y[i] = stf(__fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.0f, tanhf(inner))));
}

// y[m,:] = bf16(x * rsqrtf(mean(x^2) + eps) * w), rows [M, D]; the sum is per-thread strided partials (thread t takes
// t, t+256, ...) then a shared-memory tree 128..1. launch: block 256, grid M (one row per block).
extern "C" __global__ void stk_rms_norm_f32(const bf16* x, const float* w, bf16* y, long long D, float eps) {
    __shared__ float s[STK_BLOCK];
    const bf16* xr = x + (long long)blockIdx.x * D;
    bf16* yr = y + (long long)blockIdx.x * D;
    float acc = 0.0f;
    for (long long j = threadIdx.x; j < D; j += STK_BLOCK) {
        float v = ldf(xr, j);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = STK_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = __fadd_rn(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    float mean = __fmul_rn(s[0], 1.0f / (float)D);
    float r = rsqrtf(__fadd_rn(mean, eps));
    for (long long j = threadIdx.x; j < D; j += STK_BLOCK)
        yr[j] = stf(__fmul_rn(__fmul_rn(ldf(xr, j), r), w[j]));
}

// emb[B+1, 256] from t[B] (row B uses t = 0); the twin's _time first block. launch: block 256, grid B+1.
extern "C" __global__ void stk_time_sinusoid(const float* t, bf16* emb, long long B) {
    long long row = blockIdx.x;
    int j = threadIdx.x, i = j & 127;
    float tr = 0.0f;
    if (row < B) {
        bf16 q = stf(__fmul_rn(t[row], 1000.0f));                       // (t * 1000).to(bf16)
        tr = ldf(&q, 0);
        q = stf(__fmul_rn(tr, 1.0f / 1000.0f));                         // / 1000 (reciprocal multiply), back to bf16
        tr = ldf(&q, 0);
    }
    tr = __fmul_rn(tr, 1000.0f);
    const float c = (float)(-9.210340371976184);                       // -math.log(10000)
    float freq = expf(__fmul_rn(__fmul_rn(c, (float)i), 1.0f / 128.0f));
    float arg = __fmul_rn(tr, freq);
    emb[row * 256 + j] = stf(j < 128 ? cosf(arg) : sinf(arg));
}

// y = bf16(float(x) + dt * float(v)), mul then add each rounded; y may alias x. launch: block 256, grid ceil(n / 256).
extern "C" __global__ void stk_euler(const bf16* x, const bf16* v, bf16* y, long long n, float dt) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= n) return;
    y[i] = stf(__fadd_rn(ldf(x, i), __fmul_rn(dt, ldf(v, i))));
}

// 2-D byte copy of `rows` rows of `row_bytes` (src/dst strides in bytes); 16-byte lanes when everything is aligned,
// bytes otherwise. launch: block 256, grid (ceil(lanes_per_row / 256), rows) with lanes_per_row = row_bytes / 16 when
// aligned (all of row_bytes, strides and both pointers multiples of 16) else row_bytes.
extern "C" __global__ void stk_copy_rows(char* dst, const char* src, long long row_bytes, long long src_stride,
                                         long long dst_stride) {
    long long r = blockIdx.y;
    long long lane = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    const char* s = src + r * src_stride;
    char* d = dst + r * dst_stride;
    bool wide = (((row_bytes | src_stride | dst_stride) & 15) == 0) && ((((size_t)src | (size_t)dst) & 15) == 0);
    if (wide) {
        if (lane * 16 < row_bytes) ((uint4*)d)[lane] = ((const uint4*)s)[lane];
    } else if (lane < row_bytes) {
        d[lane] = s[lane];
    }
}

// pass 1 of absmax: partials[blockIdx.x] = max |x| over a grid-stride slice (max is exact in any order; tree fixed).
// launch: block 256, grid G = min(1024, ceil(n / 256)); partials holds G floats.
extern "C" __global__ void stk_absmax_bf16(const bf16* x, long long n, float* partials) {
    __shared__ float s[STK_BLOCK];
    float m = 0.0f;
    for (long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x; i < n; i += (long long)gridDim.x * STK_BLOCK)
        m = fmaxf(m, fabsf(ldf(x, i)));
    s[threadIdx.x] = m;
    __syncthreads();
    for (int st = STK_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = s[0];
}

// pass 2: stat[0] = max of the G partials (absmax), stat[1] = max(absmax / 448, 1e-12) (the activation scale).
// launch: block 256, grid 1.
extern "C" __global__ void stk_absmax_final(const float* partials, long long G, float* stat) {
    __shared__ float s[STK_BLOCK];
    float m = 0.0f;
    for (long long i = threadIdx.x; i < G; i += STK_BLOCK) m = fmaxf(m, partials[i]);
    s[threadIdx.x] = m;
    __syncthreads();
    for (int st = STK_BLOCK / 2; st > 0; st >>= 1) {
        if ((int)threadIdx.x < st) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + st]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        stat[0] = s[0];
        stat[1] = stk_act_scale(s[0]);
    }
}

// y = e4m3(clamp(float(x) / a, -448, 448)), RNE saturating, a = stk_act_scale(stat[0]); rows beyond valid_n (padding
// rows of the 16-row alignment) are written as zero. launch: block 256, grid ceil(total / 256), total = padded rows * K.
extern "C" __global__ void stk_quant_e4m3(const bf16* x, unsigned char* y, const float* stat, long long valid_n,
                                          long long total) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= total) return;
    if (i >= valid_n) { y[i] = 0; return; }
    float a = stk_act_scale(stat[0]);
    float v = fminf(fmaxf(__fdiv_rn(ldf(x, i), a), -448.0f), 448.0f);
    y[i] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// y[M,N] = bf16(sum_k float(x[m,k]) * float(w[n,k])), x [M,K] and w [N,K] row-major bf16 (F.linear, no bias).
// Each output is one thread's work: an fmaf chain over each 32-wide K tile in ascending order, the tile sums added in
// ascending order (blocked: error grows with 32 + K / 32 terms, not K), so a row's bits never depend on M, the grid,
// or the other rows. Tiles: 64x64 outputs per block, 4x4 per thread, K tiles of 32 through shared memory.
// launch: block 256 (thread = (ty = tid / 16, tx = tid % 16)), grid (ceil(N / 64), ceil(M / 64)).
extern "C" __global__ void stk_dense_bf16(const bf16* x, const bf16* w, bf16* y, long long M, long long N, long long K) {
    __shared__ float xs[32][65];
    __shared__ float ws[32][65];
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const long long m0 = (long long)blockIdx.y * 64, n0 = (long long)blockIdx.x * 64;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (long long k0 = 0; k0 < K; k0 += 32) {
        const int kn = (int)(K - k0 < 32 ? K - k0 : 32);
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int e = tid + r * 256, row = e >> 5, kk = e & 31;
            long long gm = m0 + row, gn = n0 + row;
            xs[kk][row] = (kk < kn && gm < M) ? ldf(x, gm * K + k0 + kk) : 0.0f;
            ws[kk][row] = (kk < kn && gn < N) ? ldf(w, gn * K + k0 + kk) : 0.0f;
        }
        __syncthreads();
        float part[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) part[i][j] = 0.0f;
        for (int kk = 0; kk < kn; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = xs[kk][ty * 4 + i]; b[i] = ws[kk][tx * 4 + i]; }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) part[i][j] = fmaf(a[i], b[j], part[i][j]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = __fadd_rn(acc[i][j], part[i][j]);
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            long long gm = m0 + ty * 4 + i, gn = n0 + tx * 4 + j;
            if (gm < M && gn < N) y[gm * N + gn] = stf(acc[i][j]);
        }
}

// y[i] = float(x[i]): the exact widening the twin does with `.float()` (modulation rows into adaln). launch: block 256,
// grid ceil(n / 256).
extern "C" __global__ void stk_bf16_to_f32(const bf16* x, float* y, long long n) {
    long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

// dst[c, r] = src[r, c] for a [rows, cols] bf16 matrix (latents between [C, H*W] and [H*W, C]); a pure data move.
// launch: block 256 as 32 x 8, grid (ceil(cols / 32), ceil(rows / 32)), 32 x 33 shared tile.
extern "C" __global__ void stk_transpose_bf16(const bf16* src, bf16* dst, long long rows, long long cols) {
    __shared__ bf16 t[32][33];
    const long long c0 = (long long)blockIdx.x * 32, r0 = (long long)blockIdx.y * 32;
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    for (int j = ty; j < 32; j += 8) {
        const long long r = r0 + j, c = c0 + tx;
        if (r < rows && c < cols) t[j][tx] = src[r * cols + c];
    }
    __syncthreads();
    for (int j = ty; j < 32; j += 8) {
        const long long c = c0 + j, r = r0 + tx;
        if (r < rows && c < cols) dst[c * rows + r] = t[tx][j];
    }
}
