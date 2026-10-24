"""Single-frame Qwen-Image 2.1 encoding with the decoder's conv/norm/attention kernels.

patch(vae) replaces the encoder arithmetic and _encode; encode(vae, RGBA_u8) selects only the posterior mean,
then normalizes it. The public diffusers encode still builds a Gaussian posterior (unused variance arithmetic);
use this module's encode for a fully recorded edit path. Tiling, batches, temporal caches and patchified models
are deliberately rejected. Kernels live in vae_enc.cu; decoder arithmetic and stable capture names are reused.
"""
from __future__ import annotations

import hashlib
import types
from functools import lru_cache

import torch

from . import vae_ops as v
from .rec import REC

_CU = v._CU.with_name("vae_enc.cu")
_DECLS = """
at::Tensor k_avg_down(at::Tensor x, int64_t cout, int64_t ft, int64_t fs);
at::Tensor k_chan_normalize(at::Tensor x, at::Tensor mean, at::Tensor stdv);
at::Tensor k_from_u8_rgba(at::Tensor x);
"""
_LAUNCH = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "vae_enc.cu"
#define BF(t) reinterpret_cast<__nv_bfloat16*>((t).data_ptr())
#define STREAM at::cuda::getCurrentCUDAStream()
static inline unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }
static void chk(at::Tensor t, at::ScalarType s) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == s && t.is_contiguous(), "need contiguous CUDA tensor");
}
at::Tensor k_avg_down(at::Tensor x, int64_t cout, int64_t ft, int64_t fs) {
    chk(x, at::kBFloat16); c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.dim() == 3 && cout > 0 && ft > 0 && fs > 0, "avg_down shape/factors");
    TORCH_CHECK(x.size(1) % fs == 0 && x.size(2) % fs == 0 && x.size(0)*ft*fs*fs % cout == 0, "avg_down grouping");
    auto y = at::empty({cout, x.size(1)/fs, x.size(2)/fs}, x.options());
    stk_avg_down<<<cdiv(y.numel(), STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(y), x.size(0), x.size(1), x.size(2), cout, ft, fs);
    return y;
}
at::Tensor k_chan_normalize(at::Tensor x, at::Tensor mean, at::Tensor stdv) {
    chk(x, at::kBFloat16); chk(mean, at::kBFloat16); chk(stdv, at::kBFloat16);
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.dim() == 4 && x.size(0) == 1 && mean.numel() == x.size(1) && stdv.numel() == x.size(1), "normalize shape");
    TORCH_CHECK(mean.device() == x.device() && stdv.device() == x.device(), "normalize device");
    auto y = at::empty_like(x);
    stk_chan_normalize<<<cdiv(x.numel(), STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(mean), BF(stdv), BF(y), x.size(2)*x.size(3), x.numel());
    return y;
}
at::Tensor k_from_u8_rgba(at::Tensor x) {
    chk(x, at::kByte); c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.dim() == 3 && x.size(2) == 4, "RGBA HWC input");
    auto y = at::empty({1, 4, 1, x.size(0), x.size(1)}, x.options().dtype(at::kBFloat16));
    stk_from_u8_rgba<<<cdiv(y.numel(), STK_BLOCK), STK_BLOCK, 0, STREAM>>>((uint8_t*)x.data_ptr(), BF(y), x.size(0), x.size(1));
    return y;
}
"""


@lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    digest = hashlib.sha256(_CU.read_bytes() + v._CU.read_bytes() + _LAUNCH.encode() + _DECLS.encode()).hexdigest()
    return load_inline(f"stk_qwen_vae_enc_{digest[:12]}", cpp_sources=_DECLS,
                       cuda_sources=f"// source sha256 {digest}\n" + _LAUNCH,
                       functions=["k_avg_down", "k_chan_normalize", "k_from_u8_rgba"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                       extra_include_paths=[str(_CU.parent)], with_cuda=True)


def _single(x):
    if x.dim() != 5 or x.shape[0] != 1 or x.shape[2] != 1 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("encoder requires CUDA bf16 [1,C,1,H,W]")


def _avg_forward(self, x):
    _single(x)
    xc = x[0, :, 0].contiguous()
    attrs = {"cin": self.in_channels, "cout": self.out_channels, "factor_t": self.factor_t,
             "fs": self.factor_s, "group_size": self.group_size, "impl": v.IMPL}
    v.STATS["avg_down"] += 1
    with REC.op(v._name(self), "avg_down", attrs, x=xc) as o:
        y = _mod().k_avg_down(xc, self.out_channels, self.factor_t, self.factor_s)
        o.out(y=y)
    return y.view(1, self.out_channels, 1, *y.shape[-2:])


def _resdown_forward(self, x, feat_cache=None, feat_idx=None):
    if feat_cache is not None:
        raise ValueError("single-frame encoder has no feature cache")
    original = x
    for resnet in self.resnets:
        x = resnet(x)
    if self.downsampler is not None:
        x = self.downsampler(x)
    return v.add(x, self.avg_shortcut(original), f"{v._name(self)}.shortcut_add")


def _encoder_forward(self, x, feat_cache=None, feat_idx=None):
    _single(x)
    if feat_cache is not None:
        raise ValueError("single-frame encoder has no feature cache")
    x = self.conv_in(x)
    for down in self.down_blocks:
        x = down(x)
    x = self.mid_block(x)
    return self.conv_out(v.silu(self.norm_out(x), f"{v._name(self)}.act_out"))


def _encode_forward(self, x):
    _single(x)
    if self.use_tiling or self.config.patch_size is not None:
        raise ValueError("single-frame encoder requires tiling off and patch_size=None")
    self.clear_cache()
    try:
        return self.quant_conv(self.encoder(x))
    finally:
        self.clear_cache()


def patch(vae):
    """Replace only the encoder and quant_conv; decoder patching may happen before or after this call."""
    from torch import nn
    from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as m

    if next(vae.parameters()).dtype != torch.bfloat16:
        raise ValueError("load the VAE in bf16")
    if vae.training:
        raise ValueError("set the VAE to eval before patching")
    for name, mod in vae.named_modules():
        mod._stk_name = f"vae.{name}" if name else "vae"
    bindings = [(nn.Conv2d, v._conv_forward), (m.QwenImage21RMS_norm, v._norm_forward),
                (m.QwenImage21ResidualBlock, v._resblock_forward),
                (m.QwenImage21AttentionBlock, v._attn_forward),
                (m.QwenImage21AvgDown3D, _avg_forward),
                (m.QwenImage21ResidualDownBlock, _resdown_forward), (m.QwenImage21Encoder3d, _encoder_forward)]
    for _, mod in list(vae.encoder.named_modules()) + [("quant_conv", vae.quant_conv)]:
        for cls, fn in bindings:
            if isinstance(mod, cls):
                mod.forward = types.MethodType(fn, mod)
                break
    vae._encode = types.MethodType(_encode_forward, vae)
    vae._stk_encoder_patched = True
    return vae


def from_u8(rgba: torch.Tensor) -> torch.Tensor:
    """CUDA uint8 RGBA [H,W,4] -> CUDA bf16 [1,4,1,H,W], keeping alpha."""
    x = rgba.contiguous()
    v.STATS["from_u8_rgba"] += 1
    with REC.op("vae.from_u8", "from_u8_rgba", {"impl": v.IMPL}, x=x) as o:
        y = _mod().k_from_u8_rgba(x)
        o.out(y=y)
    return y


def z_to_latents(vae, z: torch.Tensor) -> torch.Tensor:
    """Posterior mean [1,C,1,h,w] -> normalized [1,C,h,w]; bf16 subtract then divide."""
    _single(z)
    x = z[:, :, 0].contiguous()
    mean = torch.tensor(vae.config.latents_mean).to(x.device, x.dtype).contiguous()
    std = torch.tensor(vae.config.latents_std).to(x.device, x.dtype).contiguous()
    v.STATS["chan_normalize"] += 1
    with REC.op("vae.normalize", "chan_normalize", {"impl": v.IMPL}, x=x, mean=mean, std=std) as o:
        y = _mod().k_chan_normalize(x, mean, std)
        o.out(y=y)
    return y


@torch.inference_mode()
def encode(vae, rgba: torch.Tensor) -> torch.Tensor:
    """Resized RGBA uint8 [H,W,4] on the model device -> normalized reference latents (mean only)."""
    if not getattr(vae, "_stk_encoder_patched", False):
        raise ValueError("call vae_enc.patch first")
    if rgba.dim() != 3 or rgba.shape[-1] != 4 or rgba.dtype != torch.uint8 or not rgba.is_cuda:
        raise ValueError("encode requires CUDA uint8 RGBA [H,W,4]")
    if rgba.shape[0] <= 0 or rgba.shape[0] % 16 or rgba.shape[1] <= 0 or rgba.shape[1] % 32:
        # Shared decoder conv2d_chw currently requires even output widths, including at latent resolution.
        raise ValueError("height must be positive/multiple of 16 and width positive/multiple of 32")
    if rgba.device != next(vae.parameters()).device:
        raise ValueError("pixels and VAE must be on the same device")
    moments = vae._encode(from_u8(rgba))
    # mode() is precisely this slice; skip logvar clamp/exp and posterior sampling entirely.
    return z_to_latents(vae, moments[:, :vae.config.z_dim])
