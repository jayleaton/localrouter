"""Bindings for vae_enc.cu; GEMM and spatial tile placement are the decoder's identical kernels."""
from functools import lru_cache
import hashlib
import torch
from .vae_video import KDIR

DECLS = """
at::Tensor k_pixels(at::Tensor rgb, int64_t y0, int64_t x0, int64_t h, int64_t w);
at::Tensor k_im2col(at::Tensor x, int64_t H, int64_t W, int64_t k, int64_t stride, int64_t pad);
at::Tensor k_group_norm(at::Tensor x, at::Tensor w, at::Tensor b);
void k_silu_(at::Tensor x);
void k_add_(at::Tensor x, at::Tensor r);
at::Tensor k_moments(at::Tensor rows, int64_t h, int64_t w);
at::Tensor k_latent(at::Tensor moments, at::Tensor mean, at::Tensor stdv);
"""

LAUNCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "vae_enc.cu"
#define HP(t) reinterpret_cast<__half*>((t).data_ptr())
#define CHP(t) reinterpret_cast<const __half*>((t).data_ptr())
#define STREAM at::cuda::getCurrentCUDAStream()
static unsigned grid(int64_t n) { return (n+255)/256; }
static void chk(const at::Tensor& t, at::ScalarType dtype) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==dtype, "contiguous CUDA tensor of expected dtype");
}
static void done() { auto e=cudaGetLastError(); TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e)); }
at::Tensor k_pixels(at::Tensor rgb, int64_t y0, int64_t x0, int64_t h, int64_t w) {
    chk(rgb,at::kFloat);
    TORCH_CHECK(rgb.dim()==3 && rgb.size(2)==3 && y0>=0 && x0>=0 && h>0 && w>0 &&
                y0+h<=rgb.size(0) && x0+w<=rgb.size(1), "RGB crop");
    c10::cuda::CUDAGuard g(rgb.device());
    auto out=at::empty({h*w,3},rgb.options().dtype(at::kHalf));
    ve_pixels<<<grid(out.numel()),256,0,STREAM>>>(rgb.data_ptr<float>(),HP(out),rgb.size(1),y0,x0,h,w);
    done(); return out;
}
at::Tensor k_im2col(at::Tensor x, int64_t H, int64_t W, int64_t k, int64_t stride, int64_t pad) {
    chk(x,at::kHalf);
    TORCH_CHECK(x.dim()==2 && x.size(0)==H*W && H>1 && W>1 && (k==1 || k==3) &&
                (stride==1 || stride==2) && (pad==0 || pad==1), "encoder im2col shape");
    c10::cuda::CUDAGuard g(x.device());
    int64_t C=x.size(1), OH=stride==2 ? H/2 : H, OW=stride==2 ? W/2 : W, K=(C*k*k+7)/8*8;
    auto out=at::empty({OH*OW,K},x.options());
    ve_im2col<<<grid(out.numel()),256,0,STREAM>>>(CHP(x),HP(out),H,W,C,k,stride,pad,OH,OW,K);
    done(); return out;
}
at::Tensor k_group_norm(at::Tensor x, at::Tensor w, at::Tensor b) {
    chk(x,at::kHalf); chk(w,at::kHalf); chk(b,at::kHalf);
    TORCH_CHECK(x.dim()==2 && x.size(1)%32==0 && w.numel()==x.size(1) && b.numel()==x.size(1), "GN32 shape");
    TORCH_CHECK(x.device()==w.device() && x.device()==b.device(), "GN32 device");
    c10::cuda::CUDAGuard g(x.device()); auto out=at::empty_like(x);
    ve_group_norm<<<32,256,0,STREAM>>>(CHP(x),CHP(w),CHP(b),HP(out),x.size(1),x.size(0),32,1e-6f);
    done(); return out;
}
void k_silu_(at::Tensor x) {
    chk(x,at::kHalf); c10::cuda::CUDAGuard g(x.device());
    ve_silu<<<grid(x.numel()),256,0,STREAM>>>(HP(x),x.numel()); done();
}
void k_add_(at::Tensor x, at::Tensor r) {
    chk(x,at::kHalf); chk(r,at::kHalf);
    TORCH_CHECK(x.sizes()==r.sizes() && x.device()==r.device(), "residual shape/device");
    c10::cuda::CUDAGuard g(x.device());
    ve_add<<<grid(x.numel()),256,0,STREAM>>>(HP(x),CHP(r),x.numel()); done();
}
at::Tensor k_moments(at::Tensor rows, int64_t h, int64_t w) {
    chk(rows,at::kHalf); TORCH_CHECK(rows.dim()==2 && rows.size(0)==h*w && rows.size(1)==48, "moments shape");
    c10::cuda::CUDAGuard g(rows.device()); auto out=at::empty({3,16,h,w},rows.options());
    ve_moments<<<grid(out.numel()),256,0,STREAM>>>(CHP(rows),HP(out),h*w); done(); return out;
}
at::Tensor k_latent(at::Tensor moments, at::Tensor mean, at::Tensor stdv) {
    chk(moments,at::kHalf); chk(mean,at::kHalf); chk(stdv,at::kHalf);
    TORCH_CHECK(moments.dim()==4 && moments.size(0)==3 && moments.size(1)==16 && mean.numel()==24 &&
                stdv.numel()==24 && moments.device()==mean.device() && moments.device()==stdv.device(), "latent shape/device");
    c10::cuda::CUDAGuard g(moments.device());
    auto out=at::empty({1,24,1,moments.size(2),moments.size(3)},moments.options().dtype(at::kFloat));
    ve_latent<<<grid(out.numel()),256,0,STREAM>>>(CHP(moments),CHP(mean),CHP(stdv),out.data_ptr<float>(),
                                               moments.size(2)*moments.size(3));
    done(); return out;
}
"""

@lru_cache(maxsize=1)
def mod():
    from torch.utils.cpp_extension import load_inline
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    digest = hashlib.sha256((KDIR / "vae_enc.cu").read_bytes() + LAUNCH.encode() + DECLS.encode()).hexdigest()
    return load_inline(f"stk_h3_venc_{digest[:12]}", cpp_sources=DECLS, cuda_sources=LAUNCH,
                       functions=["k_pixels", "k_im2col", "k_group_norm", "k_silu_", "k_add_", "k_moments", "k_latent"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                       extra_include_paths=[str(KDIR)], with_cuda=True)
