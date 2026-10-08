"""LocalRouter's MiniMax H3 kernels (kernels/cuda/minimax/ops.cu) as a torch extension, launched as the Zig engine
launches them (same grids), so the twin's bits are the engine's."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

KDIR = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[4] / "kernels")) / "cuda" / "minimax"

_DECLS = """
void k_norm_mod(at::Tensor x, at::Tensor w, at::Tensor mod, at::Tensor idx, at::Tensor y, int64_t part, double eps);
void k_gate_add_(at::Tensor x, at::Tensor y, at::Tensor mod, at::Tensor idx, int64_t part);
void k_gate_add_norm_mod_(at::Tensor x, at::Tensor y, at::Tensor gmod, at::Tensor nmod, at::Tensor idx, at::Tensor w, at::Tensor out, int64_t gpart, int64_t npart, double eps);
at::Tensor k_swiglu(at::Tensor gu);
at::Tensor k_silu_mul_split(at::Tensor gu);
at::Tensor k_rms_norm(at::Tensor x, at::Tensor w, double eps);
at::Tensor k_final_mod(at::Tensor x, at::Tensor w, at::Tensor scale, at::Tensor shift, double eps);
at::Tensor k_scale(at::Tensor x, double c);
void k_uncarry_(at::Tensor a, at::Tensor v, double c1, double c2);
at::Tensor k_patchify(at::Tensor x);
at::Tensor k_unpatchify_neg(at::Tensor rows, int64_t C, int64_t T, int64_t H, int64_t W);
at::Tensor k_pack_audio(at::Tensor a);
at::Tensor k_unpack_audio_neg(at::Tensor rows, int64_t C, int64_t A);
at::Tensor k_add(at::Tensor x, at::Tensor z);
at::Tensor k_gemm_f32(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias);
at::Tensor k_denoise(at::Tensor x, at::Tensor out, double sigma);
void k_euler32_(at::Tensor x, at::Tensor den, double sigma, double dt);
void k_res2_(at::Tensor x, at::Tensor den, at::Tensor old, double e, double h, double b1, double b2);
void k_scale32_(at::Tensor x, double c);
"""

_LAUNCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "ops.cu"
#define BF(t) reinterpret_cast<bf16*>((t).data_ptr())
#define CBF(t) reinterpret_cast<const bf16*>((t).data_ptr())
#define F32(t) (t).data_ptr<float>()
#define STREAM at::cuda::getCurrentCUDAStream()
static unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }
static void chk(const at::Tensor& t, at::ScalarType s, const char* n) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == s && t.is_contiguous(), n, ": contiguous CUDA tensor of the right dtype");
}
void k_norm_mod(at::Tensor x, at::Tensor w, at::Tensor mod, at::Tensor idx, at::Tensor y, int64_t part, double eps) {
    chk(x, at::kBFloat16, "x"); chk(w, at::kFloat, "w"); chk(mod, at::kFloat, "mod"); chk(idx, at::kInt, "idx");
    chk(y, at::kBFloat16, "y");
    long long D = w.numel();
    h3_norm_mod<<<(unsigned)(x.numel() / D), 256, 0, STREAM>>>(CBF(x), F32(w), F32(mod), idx.data_ptr<int>(), BF(y), D,
                                                               (int)part, (float)eps);
}
void k_gate_add_(at::Tensor x, at::Tensor y, at::Tensor mod, at::Tensor idx, int64_t part) {
    chk(x, at::kBFloat16, "x"); chk(y, at::kBFloat16, "y"); chk(mod, at::kFloat, "mod"); chk(idx, at::kInt, "idx");
    long long D = x.size(-1), S = x.numel() / D;
    h3_gate_add<<<dim3(cdiv(D, 256), (unsigned)S), 256, 0, STREAM>>>(BF(x), CBF(y), F32(mod), idx.data_ptr<int>(), D,
                                                                    (int)part);
}
void k_gate_add_norm_mod_(at::Tensor x, at::Tensor y, at::Tensor gmod, at::Tensor nmod, at::Tensor idx, at::Tensor w, at::Tensor out,
                          int64_t gpart, int64_t npart, double eps) {
    chk(x, at::kBFloat16, "x"); chk(y, at::kBFloat16, "y"); chk(gmod, at::kFloat, "gmod"); chk(nmod, at::kFloat, "nmod");
    chk(idx, at::kInt, "idx"); chk(w, at::kFloat, "w"); chk(out, at::kBFloat16, "out");
    long long D = w.numel();
    h3_gate_add_norm_mod<<<(unsigned)(x.numel() / D), 256, 0, STREAM>>>(BF(x), CBF(y), F32(gmod), F32(nmod), idx.data_ptr<int>(),
                                                                         F32(w), BF(out), D, (int)gpart, (int)npart, (float)eps);
}
at::Tensor k_swiglu(at::Tensor gu) {
    chk(gu, at::kBFloat16, "gu");
    long long F = gu.size(-1) / 2, M = gu.numel() / (2 * F);
    auto out = at::empty({M, F}, gu.options());
    h3_swiglu<<<dim3(cdiv(F, 256), (unsigned)M), 256, 0, STREAM>>>(CBF(gu), BF(out), F);
    return out;
}
at::Tensor k_silu_mul_split(at::Tensor gu) {
    chk(gu, at::kBFloat16, "gu");
    long long F = gu.size(-1) / 2, M = gu.numel() / (2 * F);
    auto out = at::empty({M, F}, gu.options());
    h3_silu_mul_split<<<dim3(cdiv(F, 256), (unsigned)M), 256, 0, STREAM>>>(CBF(gu), BF(out), F);
    return out;
}
at::Tensor k_rms_norm(at::Tensor x, at::Tensor w, double eps) {
    chk(x, at::kBFloat16, "x"); chk(w, at::kBFloat16, "w");
    long long D = w.numel();
    auto y = at::empty_like(x);
    h3_rms_norm<<<(unsigned)(x.numel() / D), 256, 0, STREAM>>>(CBF(x), CBF(w), BF(y), D, (float)eps);
    return y;
}
at::Tensor k_final_mod(at::Tensor x, at::Tensor w, at::Tensor scale, at::Tensor shift, double eps) {
    chk(x, at::kBFloat16, "x"); chk(w, at::kBFloat16, "w"); chk(scale, at::kFloat, "scale"); chk(shift, at::kFloat, "shift");
    long long D = w.numel(), n = x.numel() / D;
    auto out = at::empty({n, D}, x.options().dtype(at::kFloat));
    h3_final_mod<<<(unsigned)n, 256, 0, STREAM>>>(CBF(x), CBF(w), F32(scale), F32(shift), F32(out), D, (float)eps);
    return out;
}
at::Tensor k_scale(at::Tensor x, double c) {
    chk(x, at::kBFloat16, "x");
    auto y = at::empty_like(x);
    long long n = x.numel();
    h3_scale<<<cdiv(n, 256), 256, 0, STREAM>>>(CBF(x), (float)c, BF(y), n);
    return y;
}
void k_uncarry_(at::Tensor a, at::Tensor v, double c1, double c2) {
    chk(a, at::kBFloat16, "a"); chk(v, at::kBFloat16, "v");
    long long n = v.numel();
    h3_uncarry<<<cdiv(n, 256), 256, 0, STREAM>>>(CBF(a), BF(v), (float)c1, (float)c2, n);
}
at::Tensor k_patchify(at::Tensor x) {
    chk(x, at::kBFloat16, "x");
    int C = x.size(-4), T = x.size(-3), H = x.size(-2), W = x.size(-1);
    auto rows = at::empty({(long long)T * (H / 2) * (W / 2), C * 4}, x.options().dtype(at::kFloat));
    h3_patchify<<<cdiv(rows.numel(), 256), 256, 0, STREAM>>>(CBF(x), F32(rows), C, T, H, W);
    return rows;
}
at::Tensor k_unpatchify_neg(at::Tensor rows, int64_t C, int64_t T, int64_t H, int64_t W) {
    chk(rows, at::kFloat, "rows");
    auto x = at::empty({1, C, T, H, W}, rows.options().dtype(at::kBFloat16));
    h3_unpatchify_neg<<<cdiv(x.numel(), 256), 256, 0, STREAM>>>(F32(rows), BF(x), (int)C, (int)T, (int)H, (int)W);
    return x;
}
at::Tensor k_pack_audio(at::Tensor a) {
    chk(a, at::kBFloat16, "a");
    int C = a.size(-3), A = a.size(-1);
    auto rows = at::empty({2LL * A, C}, a.options().dtype(at::kFloat));
    h3_pack_audio<<<cdiv(rows.numel(), 256), 256, 0, STREAM>>>(CBF(a), F32(rows), C, A);
    return rows;
}
at::Tensor k_unpack_audio_neg(at::Tensor rows, int64_t C, int64_t A) {
    chk(rows, at::kFloat, "rows");
    auto a = at::empty({1, C, 2, A}, rows.options().dtype(at::kBFloat16));
    h3_unpack_audio_neg<<<cdiv(a.numel(), 256), 256, 0, STREAM>>>(F32(rows), BF(a), (int)C, (int)A);
    return a;
}
at::Tensor k_add(at::Tensor x, at::Tensor z) {
    chk(x, at::kBFloat16, "x"); chk(z, at::kBFloat16, "z");
    auto y = at::empty_like(x);
    long long n = x.numel();
    h3_add<<<cdiv(n, 256), 256, 0, STREAM>>>(CBF(x), CBF(z), BF(y), n);
    return y;
}
at::Tensor k_gemm_f32(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias) {
    chk(a, at::kFloat, "a"); chk(b, at::kFloat, "b");
    long long M = a.size(0), N = b.size(0), K = a.size(1);
    TORCH_CHECK(b.size(1) == K, "gemm_f32 shapes");
    auto c = at::empty({M, N}, a.options());
    const float* bp = nullptr;
    if (bias.has_value()) { chk(*bias, at::kFloat, "bias"); bp = F32(*bias); }
    h3_gemm_f32<<<dim3(cdiv(N, 64), cdiv(M, 64)), 256, 0, STREAM>>>(F32(a), F32(b), bp, F32(c), M, N, K);
    return c;
}
at::Tensor k_denoise(at::Tensor x, at::Tensor out, double sigma) {
    chk(x, at::kFloat, "x"); chk(out, at::kBFloat16, "out");
    auto den = at::empty_like(x); long long n = x.numel();
    h3_denoise<<<cdiv(n, 256), 256, 0, STREAM>>>(F32(x), CBF(out), (float)sigma, F32(den), n);
    return den;
}
void k_euler32_(at::Tensor x, at::Tensor den, double sigma, double dt) {
    chk(x, at::kFloat, "x"); chk(den, at::kFloat, "den"); long long n = x.numel();
    h3_euler32<<<cdiv(n, 256), 256, 0, STREAM>>>(F32(x), F32(den), (float)sigma, (float)dt, n);
}
void k_res2_(at::Tensor x, at::Tensor den, at::Tensor old, double e, double h, double b1, double b2) {
    chk(x, at::kFloat, "x"); chk(den, at::kFloat, "den"); chk(old, at::kFloat, "old"); long long n = x.numel();
    h3_res2<<<cdiv(n, 256), 256, 0, STREAM>>>(F32(x), F32(den), F32(old), (float)e, (float)h, (float)b1, (float)b2, n);
}
void k_scale32_(at::Tensor x, double c) {
    chk(x, at::kFloat, "x"); long long n = x.numel();
    h3_scale32<<<cdiv(n, 256), 256, 0, STREAM>>>(F32(x), (float)c, F32(x), n);
}
"""

NAMES = ["k_norm_mod", "k_gate_add_", "k_gate_add_norm_mod_", "k_swiglu", "k_silu_mul_split", "k_rms_norm", "k_final_mod", "k_scale",
         "k_uncarry_", "k_patchify", "k_unpatchify_neg", "k_pack_audio", "k_unpack_audio_neg", "k_add", "k_gemm_f32",
         "k_denoise", "k_euler32_", "k_res2_", "k_scale32_"]


@lru_cache(maxsize=1)
def mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    flags = ["-O3", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"]
    digest = hashlib.sha256((KDIR / "ops.cu").read_bytes() + _LAUNCH.encode()).hexdigest()
    return load_inline(f"stk_h3_ops_{digest[:12]}", cpp_sources=_DECLS, cuda_sources=f"// {digest}\n" + _LAUNCH,
                       functions=NAMES, extra_cuda_cflags=flags, extra_include_paths=[str(KDIR)], with_cuda=True)
