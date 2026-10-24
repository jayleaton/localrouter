"""Thin GPU bindings for the shared vision kernels; arithmetic lives in kernels/cuda/minimax/te_vision.cu."""
import hashlib
from functools import lru_cache
import torch
from .te32 import KDIR
DECLS = """
at::Tensor norm(at::Tensor x,at::Tensor w,at::Tensor b);
at::Tensor gelu(at::Tensor x,int64_t mode);
void rope_(at::Tensor x,at::Tensor cs,at::Tensor sn);
at::Tensor position(at::Tensor table,at::Tensor ids,at::Tensor weights);
at::Tensor attention(at::Tensor q,at::Tensor k,at::Tensor v,double scale);
"""
SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "te_vision.cu"
#define F(t) (t).data_ptr<float>()
#define S at::cuda::getCurrentCUDAStream()
static unsigned cd(long long n){return (n+255)/256;}
static void ck(const at::Tensor& x){TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.is_contiguous(),"vision fp32 contiguous CUDA tensor");}
at::Tensor norm(at::Tensor x,at::Tensor w,at::Tensor b){ck(x);ck(w);ck(b);TORCH_CHECK(w.numel()==x.size(-1) && b.numel()==w.numel(),"norm shape");auto y=at::empty_like(x);h3v_norm<<<x.numel()/w.numel(),256,0,S>>>(F(x),F(w),F(b),F(y),w.numel());return y;}
at::Tensor gelu(at::Tensor x,int64_t mode){ck(x);auto y=at::empty_like(x);h3v_gelu<<<cd(x.numel()),256,0,S>>>(F(x),F(y),x.numel(),mode);return y;}
void rope_(at::Tensor x,at::Tensor cs,at::Tensor sn){ck(x);ck(cs);ck(sn);TORCH_CHECK(x.dim()==3 && x.size(2)%2==0 && x.size(2)<=128 && cs.size(0)==x.size(0) && cs.size(1)==x.size(2)/2 && sn.sizes()==cs.sizes(),"rope shape");h3v_rope<<<dim3(x.size(0),x.size(1)),64,0,S>>>(F(x),F(cs),F(sn),x.size(1),x.size(2));}
at::Tensor position(at::Tensor table,at::Tensor ids,at::Tensor weights){ck(table);ck(weights);TORCH_CHECK(ids.is_cuda() && ids.scalar_type()==at::kInt && ids.is_contiguous() && ids.dim()==2 && ids.size(0)==4 && weights.sizes()==ids.sizes(),"position shape");auto y=at::empty({ids.size(1),table.size(1)},table.options());h3v_position<<<cd(y.numel()),256,0,S>>>(F(table),ids.data_ptr<int>(),F(weights),F(y),ids.size(1),table.size(1));return y;}
at::Tensor attention(at::Tensor q,at::Tensor k,at::Tensor v,double scale){ck(q);ck(k);ck(v);TORCH_CHECK(q.sizes()==k.sizes() && q.sizes()==v.sizes() && q.dim()==3 && q.size(2)<=128,"attention shape");auto y=at::empty_like(q);int bytes=(q.size(0)+128)*sizeof(float);if(bytes>48*1024)TORCH_CHECK(cudaFuncSetAttribute(h3v_attention,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)==cudaSuccess,"vision attention shared memory");h3v_attention<<<dim3(q.size(0),q.size(1)),128,bytes,S>>>(F(q),F(k),F(v),F(y),q.size(0),q.size(1),q.size(2),scale);return y;}
"""
@lru_cache(maxsize=1)
def mod():
    from torch.utils.cpp_extension import load_inline
    major, minor = torch.cuda.get_device_capability()
    digest = hashlib.sha256((KDIR / "te_vision.cu").read_bytes()+SOURCE.encode()).hexdigest()
    return load_inline("stk_h3_vision_"+digest[:12], cpp_sources=DECLS, cuda_sources=SOURCE,
                       functions=["norm", "gelu", "rope_", "position", "attention"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"],
                       extra_include_paths=[str(KDIR)], with_cuda=True)
