"""Bindings for vision.cu, compiled only on first GPU use; GEMM and tanh GELU live in ops."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

KDIR = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[3] / "kernels")) / "cuda" / "qwen_image"
DECL = """
at::Tensor vis_ln(at::Tensor x, at::Tensor w, at::Tensor b, double eps);
at::Tensor vis_pos(at::Tensor table, at::Tensor ids, at::Tensor weights);
std::vector<at::Tensor> vis_qkv(at::Tensor x, at::Tensor c, at::Tensor s, int64_t heads);
at::Tensor vis_unpad(at::Tensor x);
at::Tensor vis_gelu(at::Tensor x);
at::Tensor vis_from_f32(at::Tensor x);
"""
CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "vision.cu"
#define BF(t) reinterpret_cast<__nv_bfloat16*>((t).data_ptr())
#define STREAM at::cuda::getCurrentCUDAStream()
static unsigned blocks(long long n) { return (unsigned)((n+255)/256); }
static void check(at::Tensor t, at::ScalarType dtype, at::Tensor like) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type()==dtype && t.is_contiguous(), "vision dtype/layout");
    TORCH_CHECK(t.device()==like.device(), "vision device mismatch");
}
at::Tensor vis_ln(at::Tensor x, at::Tensor w, at::Tensor b, double eps) {
    c10::cuda::CUDAGuard guard(x.device());
    check(x,at::kBFloat16,x); check(w,at::kBFloat16,x); check(b,at::kBFloat16,x);
    int D=x.size(-1); TORCH_CHECK(D>0 && w.numel()==D && b.numel()==D, "LN shape");
    auto y=at::empty_like(x);
    stk_vis_ln<<<(unsigned)(x.numel()/D),256,0,STREAM>>>(BF(x),BF(w),BF(b),BF(y),D,(float)eps);
    return y;
}
at::Tensor vis_pos(at::Tensor table, at::Tensor ids, at::Tensor weights) {
    c10::cuda::CUDAGuard guard(table.device());
    check(table,at::kBFloat16,table); check(ids,at::kInt,table); check(weights,at::kBFloat16,table);
    TORCH_CHECK(table.dim()==2 && ids.dim()==2 && ids.size(0)==4 && ids.sizes()==weights.sizes(),"position shape");
    int R=ids.size(1),D=table.size(1); auto y=at::empty({R,D},table.options());
    stk_vis_pos<<<blocks(y.numel()),256,0,STREAM>>>(BF(table),ids.data_ptr<int>(),BF(weights),BF(y),R,D);
    return y;
}
std::vector<at::Tensor> vis_qkv(at::Tensor x, at::Tensor c, at::Tensor s, int64_t heads) {
    c10::cuda::CUDAGuard guard(x.device());
    check(x,at::kBFloat16,x); check(c,at::kFloat,x); check(s,at::kFloat,x);
    TORCH_CHECK(x.dim()==2 && heads>0 && x.size(1)==3*heads*72 && c.dim()==2 && c.size(0)==x.size(0)
                && c.size(1)==36 && c.sizes()==s.sizes(),"qkv shape");
    int R=x.size(0); auto q=at::empty({R,heads,128},x.options()),k=at::empty_like(q),v=at::empty_like(q);
    stk_vis_qkv<<<blocks(q.numel()),256,0,STREAM>>>(BF(x),c.data_ptr<float>(),s.data_ptr<float>(),BF(q),BF(k),BF(v),R,heads);
    return {q,k,v};
}
at::Tensor vis_unpad(at::Tensor x) {
    c10::cuda::CUDAGuard guard(x.device()); check(x,at::kBFloat16,x);
    TORCH_CHECK(x.dim()==3 && x.size(2)==128,"unpad shape");
    auto y=at::empty({x.size(0),x.size(1)*72},x.options());
    stk_vis_unpad<<<blocks(y.numel()),256,0,STREAM>>>(BF(x),BF(y),y.numel()); return y;
}
at::Tensor vis_gelu(at::Tensor x) {
    c10::cuda::CUDAGuard guard(x.device()); check(x,at::kBFloat16,x); auto y=at::empty_like(x);
    stk_vis_gelu<<<blocks(x.numel()),256,0,STREAM>>>(BF(x),BF(y),x.numel()); return y;
}
at::Tensor vis_from_f32(at::Tensor x) {
    c10::cuda::CUDAGuard guard(x.device()); check(x,at::kFloat,x);
    auto y=at::empty_like(x,x.options().dtype(at::kBFloat16));
    stk_vis_from_f32<<<blocks(x.numel()),256,0,STREAM>>>(x.data_ptr<float>(),BF(y),x.numel()); return y;
}
"""


@lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    digest = hashlib.sha256((KDIR / "vision.cu").read_bytes() + CU.encode()).hexdigest()
    return load_inline(f"stk_vision_{digest[:12]}", cpp_sources=DECL, cuda_sources=CU,
                       functions=["vis_ln", "vis_pos", "vis_qkv", "vis_unpad", "vis_gelu", "vis_from_f32"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                       extra_include_paths=[str(KDIR)], with_cuda=True)


def layer_norm(x, w, b, eps=1e-6):
    return _mod().vis_ln(x.contiguous(), w.contiguous(), b.contiguous(), float(eps))


def position(table, ids, weights):
    return _mod().vis_pos(table.contiguous(), ids.contiguous(), weights.contiguous())


def qkv_rope_pad(x, cos, sin, heads):
    return tuple(_mod().vis_qkv(x.contiguous(), cos.contiguous(), sin.contiguous(), heads))


def unpad(x):
    return _mod().vis_unpad(x.contiguous())


def gelu(x):
    return _mod().vis_gelu(x.contiguous())


def from_f32(x):
    return _mod().vis_from_f32(x.contiguous())
