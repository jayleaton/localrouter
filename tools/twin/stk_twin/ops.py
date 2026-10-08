"""LocalRouter's small Qwen-Image kernels (kernels/cuda/qwen_image/ops.cu) as torch functions.

ops.cu is compiled verbatim (#included into one translation unit with the launchers below), so the twin and the Zig
engine run the same kernel source with the same launch shapes. Each function takes what the torch expression it
replaces took and returns the same thing.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

BLOCK = 256
ACT_PARTIALS = 1024
_CU = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[3] / "kernels")) / "cuda" / "qwen_image" / "ops.cu"

_DECLS = """
void k_gated_residual_(at::Tensor x, at::Tensor a, at::Tensor g);
at::Tensor k_silu(at::Tensor x);
at::Tensor k_tanh(at::Tensor x);
at::Tensor k_gelu_tanh(at::Tensor x);
at::Tensor k_rms_norm_f32(at::Tensor x, at::Tensor w, double eps);
at::Tensor k_time_sinusoid(at::Tensor t);
at::Tensor k_euler(at::Tensor x, at::Tensor v, double dt);
void k_copy_rows(at::Tensor dst, at::Tensor src, int64_t rows, int64_t row_bytes, int64_t src_stride, int64_t dst_stride);
at::Tensor k_absmax_stat(at::Tensor x);
at::Tensor k_quant_e4m3(at::Tensor x, at::Tensor stat, int64_t pad_rows);
at::Tensor k_dense_bf16(at::Tensor x, at::Tensor w);
at::Tensor k_gemm_bf16(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias);
at::Tensor k_rms_norm_hf(at::Tensor x, at::Tensor w, double eps);
void k_rope_half_(at::Tensor x, at::Tensor cos_t, at::Tensor sin_t);
at::Tensor k_silu_mul(at::Tensor g, at::Tensor u);
at::Tensor k_add(at::Tensor x, at::Tensor z);
at::Tensor k_embed(at::Tensor table, at::Tensor ids);
"""

_LAUNCH = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "ops.cu"
#include "gemm.cu"
#include "te.cu"

#define BF(t) reinterpret_cast<__nv_bfloat16*>((t).data_ptr())
#define STREAM at::cuda::getCurrentCUDAStream()
static inline unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }
static void chk(at::Tensor t, at::ScalarType s, const char* what) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == s && t.is_contiguous(), what, ": need contiguous CUDA ", s);
}

void k_gated_residual_(at::Tensor x, at::Tensor a, at::Tensor g) {
    chk(x, at::kBFloat16, "x"); chk(a, at::kBFloat16, "a"); chk(g, at::kBFloat16, "g");
    TORCH_CHECK(x.dim() == 3 && a.sizes() == x.sizes() && g.numel() == x.size(0) * x.size(2), "shape");
    long long total = x.numel();
    stk_gated_residual<<<cdiv(total, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(a), BF(g), total,
                                                                      x.size(1) * x.size(2), x.size(2));
}
#define POINTWISE(NAME, KERNEL) \
at::Tensor NAME(at::Tensor x) { \
    chk(x, at::kBFloat16, "x"); \
    auto y = at::empty_like(x); long long n = x.numel(); \
    KERNEL<<<cdiv(n, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(y), n); \
    return y; }
POINTWISE(k_silu, stk_silu)
POINTWISE(k_tanh, stk_tanh)
POINTWISE(k_gelu_tanh, stk_gelu_tanh)

at::Tensor k_rms_norm_f32(at::Tensor x, at::Tensor w, double eps) {
    chk(x, at::kBFloat16, "x"); chk(w, at::kFloat, "w");
    long long D = x.size(-1), M = x.numel() / D;
    TORCH_CHECK(w.numel() == D, "w");
    auto y = at::empty_like(x);
    stk_rms_norm_f32<<<(unsigned)M, STK_BLOCK, 0, STREAM>>>(BF(x), w.data_ptr<float>(), BF(y), D, (float)eps);
    return y;
}
at::Tensor k_time_sinusoid(at::Tensor t) {
    chk(t, at::kFloat, "t");
    long long B = t.numel();
    auto emb = at::empty({B + 1, 256}, t.options().dtype(at::kBFloat16));
    stk_time_sinusoid<<<(unsigned)(B + 1), STK_BLOCK, 0, STREAM>>>(t.data_ptr<float>(), BF(emb), B);
    return emb;
}
at::Tensor k_euler(at::Tensor x, at::Tensor v, double dt) {
    chk(x, at::kBFloat16, "x"); chk(v, at::kBFloat16, "v");
    TORCH_CHECK(x.numel() == v.numel(), "shape");
    auto y = at::empty_like(x); long long n = x.numel();
    stk_euler<<<cdiv(n, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(v), BF(y), n, (float)dt);
    return y;
}
void k_copy_rows(at::Tensor dst, at::Tensor src, int64_t rows, int64_t row_bytes, int64_t src_stride, int64_t dst_stride) {
    char* d = (char*)dst.data_ptr(); const char* s = (const char*)src.data_ptr();
    bool wide = (((row_bytes | src_stride | dst_stride) & 15) == 0) && ((((size_t)s | (size_t)d) & 15) == 0);
    long long lanes = wide ? row_bytes / 16 : row_bytes;
    stk_copy_rows<<<dim3(cdiv(lanes, STK_BLOCK), (unsigned)rows), STK_BLOCK, 0, STREAM>>>(d, s, row_bytes, src_stride, dst_stride);
}
at::Tensor k_absmax_stat(at::Tensor x) {
    chk(x, at::kBFloat16, "x");
    long long n = x.numel();
    unsigned G = std::min<unsigned>(1024, cdiv(n, STK_BLOCK));
    auto part = at::empty({1024}, x.options().dtype(at::kFloat));
    auto stat = at::empty({2}, x.options().dtype(at::kFloat));
    stk_absmax_bf16<<<G, STK_BLOCK, 0, STREAM>>>(BF(x), n, part.data_ptr<float>());
    stk_absmax_final<<<1, STK_BLOCK, 0, STREAM>>>(part.data_ptr<float>(), G, stat.data_ptr<float>());
    return stat;
}
at::Tensor k_quant_e4m3(at::Tensor x, at::Tensor stat, int64_t pad_rows) {
    chk(x, at::kBFloat16, "x"); chk(stat, at::kFloat, "stat");
    long long m = x.size(0), k = x.size(1), mp = (m + pad_rows - 1) / pad_rows * pad_rows;
    auto y = at::empty({mp, k}, x.options().dtype(at::kByte));
    long long total = mp * k;
    stk_quant_e4m3<<<cdiv(total, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), (unsigned char*)y.data_ptr(),
                                                                  stat.data_ptr<float>(), m * k, total);
    return y;
}
at::Tensor k_dense_bf16(at::Tensor x, at::Tensor w) {
    chk(x, at::kBFloat16, "x"); chk(w, at::kBFloat16, "w");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "shape");
    long long M = x.size(0), N = w.size(0), K = x.size(1);
    auto y = at::empty({M, N}, x.options());
    stk_dense_bf16<<<dim3(cdiv(N, 64), cdiv(M, 64)), 256, 0, STREAM>>>(BF(x), BF(w), BF(y), M, N, K);
    return y;
}
at::Tensor k_gemm_bf16(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias) {
    chk(a, at::kBFloat16, "a"); chk(b, at::kBFloat16, "b");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1) && a.size(1) % 8 == 0, "gemm shape: K % 8");
    long long M = a.size(0), N = b.size(0), K = a.size(1);
    auto c = at::empty({M, N}, a.options());
    const __nv_bfloat16* bp = nullptr;
    if (bias.has_value()) { chk(*bias, at::kBFloat16, "bias"); TORCH_CHECK(bias->numel() == N, "bias"); bp = BF(*bias); }
    stk_gemm_bf16<<<dim3(cdiv(N, 128), cdiv(M, 128)), 256, 0, STREAM>>>(BF(a), BF(b), bp, BF(c), (int)M, (int)N, (int)K,
                                                                     K, K, N);
    return c;
}
at::Tensor k_rms_norm_hf(at::Tensor x, at::Tensor w, double eps) {
    chk(x, at::kBFloat16, "x"); chk(w, at::kBFloat16, "w");
    long long D = w.numel(); auto y = at::empty_like(x);
    stk_rms_norm_hf<<<(unsigned)(x.numel() / D), 256, 0, STREAM>>>(BF(x), BF(w), BF(y), D, (float)eps);
    return y;
}
void k_rope_half_(at::Tensor x, at::Tensor cos_t, at::Tensor sin_t) {
    chk(x, at::kBFloat16, "x"); chk(cos_t, at::kFloat, "cos"); chk(sin_t, at::kFloat, "sin");
    long long rows = x.size(0), heads = x.numel() / (rows * 128);
    stk_rope_half<<<cdiv(rows * heads * 64, 256), 256, 0, STREAM>>>(BF(x), cos_t.data_ptr<float>(), sin_t.data_ptr<float>(),
                                                                  rows, heads);
}
at::Tensor k_silu_mul(at::Tensor g, at::Tensor u) {
    chk(g, at::kBFloat16, "g"); chk(u, at::kBFloat16, "u");
    auto y = at::empty_like(g); long long n = g.numel();
    stk_silu_mul<<<cdiv(n, 256), 256, 0, STREAM>>>(BF(g), BF(u), BF(y), n);
    return y;
}
at::Tensor k_add(at::Tensor x, at::Tensor z) {
    chk(x, at::kBFloat16, "x"); chk(z, at::kBFloat16, "z");
    auto y = at::empty_like(x); long long n = x.numel();
    stk_add<<<cdiv(n, 256), 256, 0, STREAM>>>(BF(x), BF(z), BF(y), n);
    return y;
}
at::Tensor k_embed(at::Tensor table, at::Tensor ids) {
    chk(table, at::kBFloat16, "table"); chk(ids, at::kInt, "ids");
    long long D = table.size(1), R = ids.numel();
    auto y = at::empty({R, D}, table.options());
    stk_embed<<<dim3(cdiv(D, 256), (unsigned)R), 256, 0, STREAM>>>(BF(table), ids.data_ptr<int>(), BF(y), D);
    return y;
}
"""


@lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    flags = ["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    digest = hashlib.sha256(b"".join((_CU.parent / f).read_bytes() for f in ("ops.cu", "gemm.cu", "te.cu", "qmm_frag.cuh"))).hexdigest()
    src = f"// ops.cu sha256 {digest}\n" + _LAUNCH
    names = ["k_gated_residual_", "k_silu", "k_tanh", "k_gelu_tanh", "k_rms_norm_f32", "k_time_sinusoid", "k_euler", "k_copy_rows",
             "k_absmax_stat", "k_quant_e4m3", "k_dense_bf16", "k_gemm_bf16", "k_rms_norm_hf", "k_rope_half_", "k_silu_mul",
             "k_add", "k_embed"]
    return load_inline(f"stk_qwen_ops_{digest[:12]}", cpp_sources=_DECLS, cuda_sources=src, functions=names,
                       extra_cuda_cflags=flags, extra_include_paths=[str(_CU.parent)], with_cuda=True)


# ---------------------------------------------------------------------------------------------- the ops
def gated_residual_(x: torch.Tensor, a: torch.Tensor, g: torch.Tensor) -> None:
    """x[B,N,D] (bf16, in place) += a[B,N,D] * g[B,1,D]; replaces `x.addcmul_(a, g)`."""
    _mod().k_gated_residual_(x, a.contiguous(), g.reshape(x.shape[0], -1).contiguous())


def silu(x: torch.Tensor) -> torch.Tensor:
    return _mod().k_silu(x.contiguous())


def tanh(x: torch.Tensor) -> torch.Tensor:
    return _mod().k_tanh(x.contiguous())


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return _mod().k_gelu_tanh(x.contiguous())


POINTWISE = {"silu": silu, "tanh": tanh, "gelu_tanh": gelu_tanh}


def rms_norm_f32(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Replaces `(x.float() * rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w).to(bf16)`; w fp32 [D]."""
    return _mod().k_rms_norm_f32(x.contiguous(), w.contiguous(), float(eps))


def time_sinusoid(t: torch.Tensor) -> torch.Tensor:
    """t [B] fp32 in [0, 1] -> emb [B + 1, 256] bf16; the last row is t = 0."""
    return _mod().k_time_sinusoid(t.reshape(-1).contiguous())


def euler(x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """Replaces `(x.float() + dt * v.float()).to(bf16)` with dt = sigma_next - sigma (a python float)."""
    return _mod().k_euler(x.contiguous(), v.contiguous(), float(dt))


def copy_rows(dst: torch.Tensor, src: torch.Tensor) -> None:
    """dst.copy_(src) for bf16 views whose last dims are contiguous: the leading dims fold to rows, taken per batch.

    Covers `kbuf[:, :P].copy_(pk)` and `vv[:, P:].copy_(qkv[:, :, 2])` ([B, N, H, D] views; B loops here, H and D fold into the row).
    """
    assert dst.shape == src.shape and dst.dtype == src.dtype == torch.bfloat16
    B, N = dst.shape[:2]
    inner = dst[0, 0].numel()
    assert dst[0, 0].is_contiguous() and src[0, 0].is_contiguous()
    es = dst.element_size()
    for b in range(B):
        _mod().k_copy_rows(dst[b], src[b], N, inner * es, src.stride(1) * es, dst.stride(1) * es)


def absmax_stat(x: torch.Tensor) -> torch.Tensor:
    """fp32 [2] = [max |x|, max(max |x| / 448, 1e-12)] (two-pass, no atomics); the first replaces x.abs().amax()."""
    return _mod().k_absmax_stat(x.contiguous())


def quant_e4m3(x: torch.Tensor, stat: torch.Tensor, pad: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] bf16 -> (e4m3 [M padded to `pad`, K], scale_a 0-dim fp32); padding rows are zero.

    Replaces `a = (x.abs().amax().float() / 448).clamp_min(1e-12)` and the `(x.float() / a).clamp(...).to(e4m3)` + pad.
    """
    xq = _mod().k_quant_e4m3(x.contiguous(), stat, pad).view(torch.float8_e4m3fn)
    return xq, stat[1:2].reshape(())


def dense_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [..., K] bf16 @ w[N, K].T -> [..., N] bf16, fp32 fmaf chain in k order; replaces `F.linear(x, w)`."""
    y = _mod().k_dense_bf16(x.reshape(-1, x.shape[-1]).contiguous(), w.contiguous())
    return y.view(*x.shape[:-1], w.shape[0])


def gemm_bf16(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """a [M, K] @ b [N, K].T (+ bias [N]) -> [M, N] bf16: LocalRouter's deterministic tensor-core GEMM (gemm.cu)."""
    return _mod().k_gemm_bf16(a.contiguous(), b.contiguous(), None if bias is None else bias.contiguous())


def rms_norm_hf(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """transformers' Qwen3VLTextRMSNorm over the last dim (= w.numel())."""
    return _mod().k_rms_norm_hf(x.contiguous(), w.contiguous(), float(eps))


def rope_half_(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """In place: x [rows, heads, 128] -> x * cos + rotate_half(x) * sin, f32 tables [rows, 64]."""
    _mod().k_rope_half_(x, cos.contiguous(), sin.contiguous())


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return _mod().k_silu_mul(g.contiguous(), u.contiguous())


def add(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return _mod().k_add(x.contiguous(), z.contiguous())


def embed(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return _mod().k_embed(table, ids.to(torch.int32).contiguous())
