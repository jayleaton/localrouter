// erf in fp32 from correctly rounded operations only (__fmaf_rn, __fmul_rn, __fadd_rn, __fdiv_rn and bit operations),
// so a kernel gives the same bits whichever CUDA toolkit compiles it: libdevice's erff and expf changed between CUDA
// 13.0 and 13.3, which made the engine (built with the host's nvcc) and the twin (the container's) disagree.
// erfc(x) = t exp(-x^2 + P(t)), t = 1 / (1 + x / 2): Numerical Recipes' erfcc, fractional error under 1.2e-7; exp by
// 2^k e^r with |r| <= ln2 / 2 and a degree 7 Taylor polynomial (relative error under 1e-8). |x| >= 4 gives +-1
// (erf(4) rounds to 1 in fp32).
#pragma once

__device__ __forceinline__ float stk_exp_rn(float y) { // y in [-20, 1]
    const int k = __float2int_rn(__fmul_rn(y, 1.4426950408889634f));
    const float kf = (float)k;
    float r = __fmaf_rn(-kf, 0.693145751953125f, y);   // ln 2, high part (exact product)
    r = __fmaf_rn(-kf, 1.428606765330187e-06f, r);      // ln 2, low part
    float p = 1.f / 5040.f;
    p = __fmaf_rn(p, r, 1.f / 720.f);
    p = __fmaf_rn(p, r, 1.f / 120.f);
    p = __fmaf_rn(p, r, 1.f / 24.f);
    p = __fmaf_rn(p, r, 1.f / 6.f);
    p = __fmaf_rn(p, r, 0.5f);
    p = __fmaf_rn(p, r, 1.f);
    p = __fmaf_rn(p, r, 1.f);
    return __fmul_rn(p, __int_as_float((k + 127) << 23));
}

__device__ __forceinline__ float stk_erf_rn(float x) {
    const float z = fabsf(x);
    if (z >= 4.f) return copysignf(1.f, x);
    const float t = __fdiv_rn(1.f, __fmaf_rn(0.5f, z, 1.f));
    float p = 0.17087277f;
    p = __fmaf_rn(p, t, -0.82215223f);
    p = __fmaf_rn(p, t, 1.48851587f);
    p = __fmaf_rn(p, t, -1.13520398f);
    p = __fmaf_rn(p, t, 0.27886807f);
    p = __fmaf_rn(p, t, -0.18628806f);
    p = __fmaf_rn(p, t, 0.09678418f);
    p = __fmaf_rn(p, t, 0.37409196f);
    p = __fmaf_rn(p, t, 1.00002368f);
    p = __fmaf_rn(p, t, -1.26551223f);
    const float erfc = __fmul_rn(t, stk_exp_rn(__fmaf_rn(-z, z, p)));
    return copysignf(__fadd_rn(1.f, -erfc), x);
}
