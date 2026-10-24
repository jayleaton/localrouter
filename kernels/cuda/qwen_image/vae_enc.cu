// Single-frame encoder additions; include the decoder helpers so both paths share their rounding rules.
// CHW bf16 activations, fixed-order fp32 reductions, no fast-math or atomics.
#include "vae.cu"

// AvgDown3D first left-pads the time axis, then groups flattened (channel, time, dy, dx) into Cout groups.
// With one frame only temporal slot ft-1 is populated. Divide by the full group including those zeros.
// launch: block STK_BLOCK, grid ceil(Cout * Ho * Wo / STK_BLOCK).
extern "C" __global__ void stk_avg_down(const bf16* x, bf16* y, int C, int H, int W, int Cout, int ft, int fs) {
    const long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    const int Ho = H / fs, Wo = W / fs, factor = ft * fs * fs;
    const int group = C * factor / Cout;
    if (i >= (long long)Cout * Ho * Wo) return;
    const int co = i / ((long long)Ho * Wo), oy = (i / Wo) % Ho, ox = i % Wo;
    float sum = 0.0f;
    for (int g = 0; g < group; ++g) {
        const int j = co * group + g, c = j / factor, r = j % factor;
        const int t = r / (fs * fs), dy = (r / fs) % fs, dx = r % fs;
        const float v = t == ft - 1 ? vldf(x, ((long long)c * H + oy * fs + dy) * W + ox * fs + dx) : 0.0f;
        sum = __fadd_rn(sum, v);
    }
    y[i] = vstf(__fdiv_rn(sum, (float)group));
}

// Pipeline normalization: bf16 subtraction rounds before the bf16 division; no reciprocal multiply.
// x [C, HW], mean/std [C]. launch: block STK_BLOCK, grid ceil(C*HW / STK_BLOCK).
extern "C" __global__ void stk_chan_normalize(const bf16* x, const bf16* mean, const bf16* stdv, bf16* y,
                                               long long HW, long long n) {
    const long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    if (i >= n) return;
    const int c = i / HW;
    const float sub = bfround(__fsub_rn(vldf(x, i), vldf(mean, c)));
    y[i] = vstf(__fdiv_rn(sub, vldf(stdv, c)));
}

// RGBA u8 HWC -> CHW bf16: float32 /255, *2, -1, then cast. Alpha is preserved; no compositing.
// Explicit intrinsics prevent contraction across the reference's three fp32 operations.
extern "C" __global__ void stk_from_u8_rgba(const uint8_t* x, bf16* y, int H, int W) {
    const long long i = (long long)blockIdx.x * STK_BLOCK + threadIdx.x;
    const long long HW = (long long)H * W;
    if (i >= 4 * HW) return;
    const int c = i / HW;
    const float unit = __fdiv_rn((float)x[(i % HW) * 4 + c], 255.0f);
    y[i] = vstf(__fsub_rn(__fmul_rn(unit, 2.0f), 1.0f));
}
