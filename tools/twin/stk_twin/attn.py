"""LocalRouter's attention (kernels/cuda/qwen_image/attention.cu) as a torch extension, so the twin runs the very
kernel the Zig engine launches: `pattn2_kernel<128, 4, 1, 2, 64, true, 3>` (pattn_kernel's bits, faster staging).
Launch shape, also the Zig side's: block 128 (4 warps x 16 rows, one head), grid (ceil(W / 64), H), dynamic shared
memory 32,768 bytes (two 64-key staging slots of 128 dims).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

KDIR = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[3] / "kernels")) / "cuda" / "qwen_image"
SMEM = 32768
ROWS = 64  # 4 warps x 16 query rows a block

CPP = "void attn(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, int64_t p0, int64_t nkeys, bool causal);"
CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention.cu"
void attn(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, int64_t p0, int64_t nkeys, bool causal) {
    const int W = q.size(0), H = q.size(1), HK = k.size(1);
    auto kernel = stk_attention::pattn2_kernel<128, 4, 1, 2, 64, true, 3>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    dim3 grid((W + 63) / 64, H);
    kernel<<<grid, 128, 32768, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)p0, W, H, HK, H / HK, (int)nkeys, causal ? 1 : 0, 1.0f / sqrtf(128.0f));
}
"""


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"
    return load_inline("stk_attention_v2", cpp_sources=CPP, cuda_sources=CU, functions=["attn"],
                       extra_cuda_cflags=["-O3", arch], extra_include_paths=[str(KDIR)], verbose=False)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, p0: int = 0) -> torch.Tensor:
    """q [W, H, 128], k / v [S, H, 128] bf16 contiguous -> [W, H, 128]. Non-causal: every query sees all S keys;
    causal: query i (position p0 + i) sees keys 0..p0 + i."""
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.shape[-1] == 128
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    _ext().attn(q, k, v, out, p0, k.shape[0], causal)
    return out
