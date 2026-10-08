"""comfy-kitchen v0.2.35's `int8_attention` and `rms_rope_split_half_` as torch functions on LocalRouter's copies of their
kernels (kernels/cuda/minimax/kitchen, tools/kernels/sync_kitchen.py).

kitchen_launch.cu is compiled as is (#included into one translation unit with the torch wrappers below), so the twin and
the Zig engine launch the wheel's kernels with the wheel's template arguments, grids and block sizes. Each function takes
what comfy-kitchen's took and returns the same thing; the results are bit-identical to the wheel's (stk_twin.h3.test_kitchen).
The nvcc flags are the wheel's (backends/cuda/CMakeLists.txt): --use_fast_math decides the division, rsqrt and exp codegen.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

_DIR = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[4] / "kernels")) / "cuda" / "minimax"

_DECLS = """
at::Tensor k_int8_attention(at::Tensor q, at::Tensor k, at::Tensor v, double scale);
at::Tensor k_int8_attention_rows(at::Tensor q, at::Tensor k, at::Tensor v, double scale);
void k_rms_rope_split_half_(at::Tensor q, at::Tensor k, at::Tensor freqs, at::Tensor qw, at::Tensor kw, double eps, int64_t rot_dim);
"""

_LAUNCH = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "kitchen_launch.cu"

#define STREAM at::cuda::getCurrentCUDAStream()
static void ok(int e, const char* what) { TORCH_CHECK(e == 0, what, ": CUDA error ", e, " ", cudaGetErrorString((cudaError_t)e)); }
static void bf16_cuda(const at::Tensor& t, int dim, const char* what) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == dim, what, ": need a bf16 CUDA tensor of ", dim, " dims");
}

at::Tensor k_int8_attention(at::Tensor q, at::Tensor k, at::Tensor v, double scale) {
    bf16_cuda(q, 4, "q"); bf16_cuda(k, 4, "k"); bf16_cuda(v, 4, "v");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v: one device");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1, "the last dimension of q, k and v must be contiguous");
    TORCH_CHECK(v.sizes() == k.sizes() && q.size(0) == k.size(0) && q.size(3) == k.size(3), "q, k, v: shapes");
    c10::cuda::CUDAGuard guard(q.device());
    const int B = q.size(0), H = q.size(1), Sq = q.size(2), D = q.size(3), HK = k.size(1), Sk = k.size(2);
    const size_t bytes = kitchen_int8_attention_workspace_bytes(B, H, HK, Sq, Sk, D);
    TORCH_CHECK(bytes > 0, "int8_attention: unsupported shape (head dim 64 or 128, H divisible by HK)");
    auto o = at::empty({B, H, Sq, D}, q.options());
    auto ws = at::empty({(int64_t)bytes}, q.options().dtype(at::kByte));
    ok(kitchen_int8_attention(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), ws.data_ptr(), B, H, HK, Sq, Sk, D,
                              q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                              v.stride(0), v.stride(1), v.stride(2), (float)scale, STREAM), "int8_attention");
    return o;
}

// kitchen_int8_attention_rows: the same kernels with the output written as rows [Sq, H * D] (batch 1), where the Zig engine's
// out projection reads it (no [H, S, D] -> rows pass). Bit-identical to k_int8_attention's output transposed to rows.
at::Tensor k_int8_attention_rows(at::Tensor q, at::Tensor k, at::Tensor v, double scale) {
    bf16_cuda(q, 4, "q"); bf16_cuda(k, 4, "k"); bf16_cuda(v, 4, "v");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "q, k, v: one device");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1, "the last dimension of q, k and v must be contiguous");
    TORCH_CHECK(v.sizes() == k.sizes() && q.size(0) == 1 && k.size(0) == 1 && q.size(3) == k.size(3), "q, k, v: shapes (batch 1)");
    c10::cuda::CUDAGuard guard(q.device());
    const int B = 1, H = q.size(1), Sq = q.size(2), D = q.size(3), HK = k.size(1), Sk = k.size(2);
    const size_t bytes = kitchen_int8_attention_workspace_bytes(B, H, HK, Sq, Sk, D);
    TORCH_CHECK(bytes > 0, "int8_attention_rows: unsupported shape (head dim 64 or 128, H divisible by HK)");
    auto o = at::empty({Sq, (int64_t)H * D}, q.options());
    auto ws = at::empty({(int64_t)bytes}, q.options().dtype(at::kByte));
    ok(kitchen_int8_attention_rows(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), ws.data_ptr(), B, H, HK, Sq, Sk, D,
                                   q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                                   v.stride(0), v.stride(1), v.stride(2), (float)scale, STREAM), "int8_attention_rows");
    return o;
}

void k_rms_rope_split_half_(at::Tensor q, at::Tensor k, at::Tensor freqs, at::Tensor qw, at::Tensor kw, double eps, int64_t rot_dim) {
    bf16_cuda(q, 4, "q"); bf16_cuda(k, 4, "k"); bf16_cuda(freqs, 6, "freqs"); bf16_cuda(qw, 1, "q_norm_weight"); bf16_cuda(kw, 1, "k_norm_weight");
    TORCH_CHECK(q.sizes() == k.sizes(), "q, k: shapes");
    TORCH_CHECK(q.device() == k.device() && q.device() == freqs.device() && q.device() == qw.device() && q.device() == kw.device(), "one device");
    c10::cuda::CUDAGuard guard(q.device());
    const int64_t rot = rot_dim > 0 ? rot_dim : q.size(3);
    TORCH_CHECK(freqs.size(3) == rot / 2 && freqs.size(4) == 2 && freqs.size(5) == 2, "freqs: [.., rot_dim/2, 2, 2]");
    TORCH_CHECK((freqs.size(0) == 1 || freqs.size(0) == q.size(0)) && (freqs.size(1) == 1 || freqs.size(1) == q.size(1)) &&
                (freqs.size(2) == 1 || freqs.size(2) == q.size(2)), "freqs: must broadcast to q");
    TORCH_CHECK(qw.numel() == q.size(3) && kw.numel() == q.size(3), "norm weights: [head_dim]");
    ok(kitchen_rms_rope_split_half_(q.data_ptr(), k.data_ptr(), freqs.data_ptr(), qw.data_ptr(), kw.data_ptr(),
                                    q.size(0), q.size(1), q.size(2), (int)q.size(3), (int)rot_dim,
                                    freqs.size(0), freqs.size(1), freqs.size(2),
                                    q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                                    freqs.stride(0), freqs.stride(1), freqs.stride(2), freqs.stride(3), freqs.stride(4), freqs.stride(5),
                                    qw.stride(0), kw.stride(0), (float)eps, STREAM), "rms_rope_split_half_");
}
"""

_FLAGS = ["-O3", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
          "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
          "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
          "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__"]


@lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = os.environ.get("STK_KITCHEN_ARCH", f"{major}{minor}" + ("a" if major >= 9 else ""))
    flags = _FLAGS + [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    files = sorted([_DIR / "kitchen_launch.cu", *(_DIR / "kitchen").iterdir()])
    h = hashlib.sha256(_LAUNCH.encode() + " ".join(flags).encode())
    for f in files:
        h.update(f.read_bytes())
    digest = h.hexdigest()
    src = f"// kitchen sha256 {digest}\n" + _LAUNCH
    return load_inline(f"stk_h3_kitchen_{digest[:12]}", cpp_sources=_DECLS, cuda_sources=src,
                       functions=["k_int8_attention", "k_int8_attention_rows", "k_rms_rope_split_half_"], extra_cuda_cflags=flags,
                       extra_include_paths=[str(_DIR)], with_cuda=True)


def int8_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """softmax(q k^T * scale) v with INT8 Q, K, V and U8 probabilities; q [B,H,Sq,D], k, v [B,HK,Sk,D] bf16, D 64 or 128, no mask."""
    return _mod().k_int8_attention(q, k, v, float(q.shape[-1] ** -0.5 if scale is None else scale))


def int8_attention_rows(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """int8_attention (batch 1) with the result as bf16 rows [Sq, H * D]: equal, bit for bit, to
    int8_attention(q, k, v)[0].transpose(0, 1).reshape(Sq, H * D), without that transpose (the kernel stores the rows)."""
    return _mod().k_int8_attention_rows(q, k, v, float(q.shape[-1] ** -0.5 if scale is None else scale))


def rms_rope_split_half_(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor, q_norm_weight: torch.Tensor,
                         k_norm_weight: torch.Tensor, epsilon: float, rot_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm then split-half RoPE on the first rot_dim dims, in place: q, k [B, S, H, 128] bf16 (strided views
    allowed), freqs [B|1, S|1, H|1, rot_dim/2, 2, 2] bf16, weights [128] bf16."""
    _mod().k_rms_rope_split_half_(q, k, freqs, q_norm_weight, k_norm_weight, float(epsilon), int(rot_dim))
    return q, k
