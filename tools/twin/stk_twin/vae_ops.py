"""The VAE decoder on LocalRouter's own kernels: `patch(vae)` swaps every arithmetic op of the decode path of a loaded
diffusers `AutoencoderKLQwenImage21` for ours (kernels/cuda/qwen_image/vae.cu + gemm.cu through `ops.gemm_bf16`), so the
Zig engine, which launches the same kernels, is bit-exact with the twin. Module structure and weights stay diffusers';
only the forwards that contain arithmetic are replaced, each op inside `REC.op` with a stable name
("vae.decoder.up_blocks.2.resnets.0.conv1"), so a capture lists the decoder op by op. See tools/twin/VAE-PORT.md.

Pure data moves that stay in torch are exact: `.view` / `.reshape` / slices of a contiguous tensor and the final
`torch.clamp(out, -1, 1)` in `AutoencoderKLQwenImage21._decode` (min / max select, they never round).
"""

from __future__ import annotations

import hashlib
import math
import os
import types
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch

from . import ops
from .rec import REC

IMPL = "stk"
BAND_BYTES = int(os.environ.get("STK_VAE_BAND_BYTES", 512 << 20))  # im2col columns per band (bits do not depend on it)
STATS: Counter = Counter()  # ops run since the last reset, by kind (the test prints it)

_CU = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[3] / "kernels")) / "cuda" / "qwen_image" / "vae.cu"

_DECLS = """
at::Tensor k_im2col(at::Tensor x, int64_t kh, int64_t kw, int64_t stride, int64_t pt, int64_t pb, int64_t pl, int64_t pr,
                    int64_t up, int64_t Kpad, int64_t p0, int64_t np);
void k_transpose(at::Tensor x, int64_t R, int64_t Cc, int64_t ldx, at::Tensor y, int64_t y_off, int64_t ldy);
at::Tensor k_rms_norm_chw(at::Tensor x, at::Tensor gamma, double scale);
at::Tensor k_add(at::Tensor x, at::Tensor z);
at::Tensor k_upsample2x(at::Tensor x);
at::Tensor k_dup_up(at::Tensor x, int64_t Cout, int64_t fs, int64_t factor, int64_t repeats, int64_t fti);
at::Tensor k_softmax_rows(at::Tensor S, int64_t ldp, double scale);
at::Tensor k_chan_affine(at::Tensor x, at::Tensor mean, at::Tensor stdv);
at::Tensor k_to_u8(at::Tensor x);
"""

_LAUNCH = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "vae.cu"

#define BF(t) reinterpret_cast<__nv_bfloat16*>((t).data_ptr())
#define STREAM at::cuda::getCurrentCUDAStream()
static inline unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }
static void chk(at::Tensor t, at::ScalarType s, const char* what) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == s && t.is_contiguous(), what, ": need contiguous CUDA ", s);
}

at::Tensor k_im2col(at::Tensor x, int64_t kh, int64_t kw, int64_t stride, int64_t pt, int64_t pb, int64_t pl, int64_t pr,
                    int64_t up, int64_t Kpad, int64_t p0, int64_t np) {
    chk(x, at::kBFloat16, "x");
    TORCH_CHECK(x.dim() == 3 && Kpad % 8 == 0 && Kpad >= x.size(0) * kh * kw, "im2col shape");
    auto cols = at::empty({np, Kpad}, x.options());
    long long total = np * Kpad;
    stk_im2col<<<cdiv(total, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(cols), (int)x.size(0), (int)x.size(1), (int)x.size(2),
        (int)kh, (int)kw, (int)stride, (int)pt, (int)pb, (int)pl, (int)pr, (int)up, (int)Kpad, p0, np);
    return cols;
}
void k_transpose(at::Tensor x, int64_t R, int64_t Cc, int64_t ldx, at::Tensor y, int64_t y_off, int64_t ldy) {
    chk(x, at::kBFloat16, "x"); chk(y, at::kBFloat16, "y");
    stk_transpose_bf16<<<dim3(cdiv(R, 32), cdiv(Cc, 32)), dim3(32, 8), 0, STREAM>>>(BF(x), BF(y) + y_off, R, Cc, ldx, ldy);
}
at::Tensor k_rms_norm_chw(at::Tensor x, at::Tensor gamma, double scale) {
    chk(x, at::kBFloat16, "x"); chk(gamma, at::kBFloat16, "gamma");
    long long C = gamma.numel(), HW = x.numel() / C;
    TORCH_CHECK(x.numel() == C * HW && x.size(0) == 1 && x.size(1) == C, "rms_norm shape");
    auto y = at::empty_like(x);
    stk_channel_rms_norm<<<cdiv(HW, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(gamma), BF(y), (int)C, HW, (float)scale);
    return y;
}
at::Tensor k_add(at::Tensor x, at::Tensor z) {
    chk(x, at::kBFloat16, "x"); chk(z, at::kBFloat16, "z");
    TORCH_CHECK(x.sizes() == z.sizes(), "add shape");
    auto y = at::empty_like(x); long long n = x.numel();
    stk_add_bf16<<<cdiv(n, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(z), BF(y), n);
    return y;
}
at::Tensor k_upsample2x(at::Tensor x) {
    chk(x, at::kBFloat16, "x");
    TORCH_CHECK(x.dim() == 3, "upsample shape");
    auto y = at::empty({x.size(0), 2 * x.size(1), 2 * x.size(2)}, x.options());
    stk_upsample_nearest2x<<<cdiv(y.numel(), STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(y), (int)x.size(0), (int)x.size(1), (int)x.size(2));
    return y;
}
at::Tensor k_dup_up(at::Tensor x, int64_t Cout, int64_t fs, int64_t factor, int64_t repeats, int64_t fti) {
    chk(x, at::kBFloat16, "x");
    TORCH_CHECK(x.dim() == 3 && x.size(0) * repeats == Cout * factor, "dup_up shape");
    auto y = at::empty({Cout, x.size(1) * fs, x.size(2) * fs}, x.options());
    stk_dup_up<<<cdiv(y.numel(), STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(y), (int)Cout, (int)x.size(1), (int)x.size(2),
        (int)fs, (int)factor, (int)repeats, (int)fti);
    return y;
}
at::Tensor k_softmax_rows(at::Tensor S, int64_t ldp, double scale) {
    chk(S, at::kBFloat16, "S");
    TORCH_CHECK(S.dim() == 2 && ldp >= S.size(1), "softmax shape");
    auto P = at::empty({S.size(0), ldp}, S.options());
    stk_softmax_rows<<<(unsigned)S.size(0), STK_BLOCK, 0, STREAM>>>(BF(S), BF(P), S.size(1), ldp, (float)scale);
    return P;
}
at::Tensor k_chan_affine(at::Tensor x, at::Tensor mean, at::Tensor stdv) {
    chk(x, at::kBFloat16, "x"); chk(mean, at::kBFloat16, "mean"); chk(stdv, at::kBFloat16, "std");
    long long C = mean.numel(), n = x.numel(), HW = n / (x.size(0) * C);
    TORCH_CHECK(x.size(0) == 1 && x.size(1) == C && stdv.numel() == C, "chan_affine shape");
    auto y = at::empty_like(x);
    stk_chan_affine<<<cdiv(n, STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), BF(mean), BF(stdv), BF(y), HW, n);
    return y;
}
at::Tensor k_to_u8(at::Tensor x) {
    chk(x, at::kBFloat16, "x");
    TORCH_CHECK(x.dim() == 3, "to_u8 shape");
    auto u = at::empty({x.size(1), x.size(2), x.size(0)}, x.options().dtype(at::kByte));
    stk_to_u8_hwc<<<cdiv(x.numel(), STK_BLOCK), STK_BLOCK, 0, STREAM>>>(BF(x), (uint8_t*)u.data_ptr(), (int)x.size(0), (int)x.size(1), (int)x.size(2));
    return u;
}
"""


@lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    flags = ["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    digest = hashlib.sha256(_CU.read_bytes() + _LAUNCH.encode()).hexdigest()
    src = f"// vae.cu sha256 {digest}\n" + _LAUNCH
    names = ["k_im2col", "k_transpose", "k_rms_norm_chw", "k_add", "k_upsample2x", "k_dup_up", "k_softmax_rows",
             "k_chan_affine", "k_to_u8"]
    return load_inline(f"stk_qwen_vae_{digest[:12]}", cpp_sources=_DECLS, cuda_sources=src, functions=names,
                       extra_cuda_cflags=flags, extra_include_paths=[str(_CU.parent)], with_cuda=True)


# ---------------------------------------------------------------------------------------------- raw functions
def _ceil8(n: int) -> int:
    return (n + 7) // 8 * 8


def transpose(x: torch.Tensor, ld_out: int | None = None) -> torch.Tensor:
    """x [R, Cc] -> [Cc, ld_out or R]; the columns in [R, ld_out) are zero."""
    R, Cc = x.shape
    ld = ld_out or R
    y = torch.zeros(Cc, ld, dtype=x.dtype, device=x.device) if ld > R else torch.empty(Cc, ld, dtype=x.dtype, device=x.device)
    x = x.contiguous()
    _mod().k_transpose(x, R, Cc, x.stride(0), y, 0, ld)
    return y


def conv2d_chw(x: torch.Tensor, wm: torch.Tensor, bias: torch.Tensor | None, cout: int, kh: int, kw: int, stride: int,
               pads: tuple[int, int, int, int], up: int = 1) -> torch.Tensor:
    """Conv over x [C, H, W] bf16 (zero pads (top, bottom, left, right), optional virtual nearest xup upsample) as
    im2col + gemm: cols [np, Kpad] . wm[Coutp, Kpad]^T (+ bias[Coutp]) -> [np, Coutp], then transposed to CHW.

    wm is the weight [Cout, Cin * kh * kw] zero-padded to Kpad columns (multiple of 8) and to an even Coutp rows (the
    GEMM's ldc must be even); bias is padded to Coutp with zeros. The GEMM puts the pixels on M so that the bias (indexed
    by N) is the output channel. Bands of output rows bound the columns' memory; a row's bits do not depend on M.
    """
    C, H, W = x.shape
    pt, pb, pl, pr = pads
    Ho, Wo = (H * up + pt + pb - kh) // stride + 1, (W * up + pl + pr - kw) // stride + 1
    assert Wo % 2 == 0, "output width must be even (GEMM ldc)"
    Coutp, Kpad = wm.shape
    out = torch.empty(cout, Ho, Wo, dtype=x.dtype, device=x.device)
    rows = max(1, BAND_BYTES // (Wo * Kpad * 2))
    x = x.contiguous()
    m = _mod()
    for y0 in range(0, Ho, rows):
        r = min(rows, Ho - y0)
        cols = m.k_im2col(x, kh, kw, stride, pt, pb, pl, pr, up, Kpad, y0 * Wo, r * Wo)
        pm = ops.gemm_bf16(cols, wm, bias)  # [r * Wo, Coutp]
        m.k_transpose(pm, r * Wo, cout, Coutp, out, y0 * Wo, Ho * Wo)
        del cols, pm
    return out


# ---------------------------------------------------------------------------------------------- recorded ops
def _name(mod, default: str = "vae") -> str:
    return getattr(mod, "_stk_name", default)


def add(x: torch.Tensor, y: torch.Tensor, name: str) -> torch.Tensor:
    STATS["add"] += 1
    with REC.op(name, "add", {"impl": IMPL}, x=x, y=y) as o:
        z = _mod().k_add(x.contiguous(), y.contiguous())
        o.out(y=z)
    return z


def silu(x: torch.Tensor, name: str) -> torch.Tensor:
    STATS["silu"] += 1
    with REC.op(name, "silu", {"impl": IMPL}, x=x) as o:
        y = ops.silu(x)
        o.out(y=y)
    return y


def _gemm(name: str, a: torch.Tensor, bm: torch.Tensor, bias, attrs: dict, rec_b: bool = False) -> torch.Tensor:
    """y [M, N] = a [M, K] . bm [N, K]^T (+ bias[N]); bm is recorded as an input only when it is an activation."""
    STATS["gemm"] += 1
    ins = {"a": a, "b": bm} if rec_b else {"a": a}
    with REC.op(name, "gemm", {"impl": "gemm_bf16", **attrs}, **ins) as o:
        y = ops.gemm_bf16(a, bm, bias)
        o.out(y=y)
    return y


def _tr(name: str, x: torch.Tensor, ld_out: int | None = None) -> torch.Tensor:
    STATS["transpose"] += 1
    with REC.op(name, "transpose", {"impl": IMPL, "ld_out": ld_out or x.shape[0]}, x=x) as o:
        y = transpose(x, ld_out)
        o.out(y=y)
    return y


def _wprep(mod):
    """The conv weight as [Coutp, Kpad] (zero padded), the padded bias, cout, K, Kpad; cached on the module."""
    p = getattr(mod, "_stk_w", None)
    if p is None:
        w = mod.weight.detach()
        assert w.dtype == torch.bfloat16, "the VAE must be loaded in bf16"
        cout = w.shape[0]
        wm = w.reshape(cout, -1)
        K = wm.shape[1]
        Kpad, Coutp = _ceil8(K), cout + (cout & 1)
        if Kpad != K or Coutp != cout:
            w2 = torch.zeros(Coutp, Kpad, dtype=w.dtype, device=w.device)
            w2[:cout, :K] = wm
            wm = w2
        else:
            wm = wm.contiguous()
        b = None
        if mod.bias is not None:
            b = torch.zeros(Coutp, dtype=w.dtype, device=w.device)
            b[:cout] = mod.bias.detach()
        p = mod._stk_w = (wm, b, cout, K, Kpad)
    return p


def _conv_forward(self, x, cache_x=None):
    """Conv2d / QwenImage21CausalConv3d on a single frame (x [1, C, (1,) H, W]) as im2col + gemm."""
    if cache_x is not None:
        raise ValueError("single-frame decode has no feature cache")
    five = x.dim() == 5
    assert x.shape[0] == 1 and (not five or x.shape[2] == 1), "batch 1, one frame"
    assert self.dilation == (1, 1) and self.groups == 1 and self.padding_mode == "zeros"
    if hasattr(self, "_padding"):  # QwenImage21CausalConv3d: F.pad order (left, right, top, bottom)
        pl, pr, pt, pb = self._padding
    else:
        (pt, pb), (pl, pr) = (self.padding[0],) * 2, (self.padding[1],) * 2
    wm, bias, cout, K, Kpad = _wprep(self)
    kh, kw = self.kernel_size
    xc = x.reshape(x.shape[1], x.shape[-2], x.shape[-1]).contiguous()
    name = _name(self)
    STATS["conv"] += 1
    attrs = {"weight": f"{name}.weight", "bias": f"{name}.bias", "cin": xc.shape[0], "cout": cout, "k": [kh, kw],
             "stride": self.stride[0], "pad": [pt, pb, pl, pr], "kpad": Kpad, "impl": "im2col+gemm_bf16"}
    with REC.op(name, "conv2d", attrs, x=xc) as o:
        y = conv2d_chw(xc, wm, bias, cout, kh, kw, self.stride[0], (pt, pb, pl, pr))
        o.out(y=y)
    return y.view(1, cout, 1, *y.shape[1:]) if five else y.view(1, *y.shape)


def _norm_forward(self, x):
    """QwenImage21RMS_norm over channels of x [1, C, ...] (see stk_channel_rms_norm for the roundings)."""
    assert self.channel_first and isinstance(self.bias, float) and self.bias == 0.0
    g = getattr(self, "_stk_g", None)
    if g is None:
        assert self.gamma.dtype == torch.bfloat16
        g = self._stk_g = self.gamma.detach().reshape(-1).contiguous()
    assert x.shape[0] == 1 and x.shape[1] == g.numel()
    STATS["rms_norm"] += 1
    name = _name(self)
    xc = x.contiguous()
    with REC.op(name, "channel_rms_norm", {"weight": f"{name}.gamma", "scale": float(self.scale), "impl": IMPL}, x=xc) as o:
        y = _mod().k_rms_norm_chw(xc, g, float(self.scale))
        o.out(y=y)
    return y


def _upsample_forward(self, x):
    assert tuple(self.scale_factor) == (2.0, 2.0) and self.mode == "nearest-exact" and x.shape[0] == 1
    STATS["upsample"] += 1
    xc = x.reshape(x.shape[1], x.shape[2], x.shape[3]).contiguous()
    with REC.op(_name(self), "upsample_nearest2x", {"impl": IMPL}, x=xc) as o:
        y = _mod().k_upsample2x(xc)
        o.out(y=y)
    return y.unsqueeze(0)


def _dupup_forward(self, x, first_chunk=False):
    """QwenImage21DupUp3D for one frame; with first_chunk only the last of the factor_t time slots survives."""
    assert x.shape[0] == 1 and x.shape[2] == 1 and (first_chunk or self.factor_t == 1)
    xc = x.reshape(x.shape[1], x.shape[3], x.shape[4]).contiguous()
    fti = self.factor_t - 1
    STATS["dup_up"] += 1
    attrs = {"cin": self.in_channels, "cout": self.out_channels, "factor_t": self.factor_t, "fs": self.factor_s,
             "repeats": self.repeats, "fti": fti, "impl": IMPL}
    with REC.op(_name(self), "dup_up", attrs, x=xc) as o:
        y = _mod().k_dup_up(xc, self.out_channels, self.factor_s, self.factor, self.repeats, fti)
        o.out(y=y)
    return y.view(1, self.out_channels, 1, *y.shape[1:])


def _resblock_forward(self, x, feat_cache=None, feat_idx=None):
    n = _name(self)
    h = self.conv_shortcut(x)
    y = silu(self.norm1(x), f"{n}.act1")
    y = self.conv1(y)
    y = silu(self.norm2(y), f"{n}.act2")
    y = self.conv2(y)  # dropout(p = 0) is the identity
    return add(y, h, f"{n}.add")


def _attn_forward(self, x):
    """Single-head attention over the HW pixels, channel dim C, no mask; SDPA's math written as our kernels:
    xn = rms_norm(x); q, k, v = xn^T W{q,k,v}^T + b (token major [L, C]); S = q k^T (bf16); P = softmax(S / sqrt(C));
    O = P v; out = proj(O) + x."""
    n = _name(self)
    assert x.shape[0] == 1 and x.shape[2] == 1
    C, H, W = x.shape[1], x.shape[3], x.shape[4]
    L = H * W
    assert C % 8 == 0 and L % 2 == 0
    Lpad = _ceil8(L)
    p = getattr(self, "_stk_attn", None)
    if p is None:
        wq = self.to_qkv.weight.detach().reshape(3 * C, C)
        bq = self.to_qkv.bias.detach()
        wp = self.proj.weight.detach().reshape(C, C).contiguous()
        p = self._stk_attn = (wq[:C], bq[:C], wq[C:2 * C], bq[C:2 * C], wq[2 * C:], bq[2 * C:], wp, self.proj.bias.detach())
    wq, bq, wk, bk, wv, bv, wp, bp = p
    xn = self.norm(x.reshape(1, C, H, W))
    xt = _tr(f"{n}.xt", xn.reshape(C, L))  # [L, C]
    q = _gemm(f"{n}.q", xt, wq, bq, {"weight": f"{n}.to_qkv.weight[0:{C}]"})
    k = _gemm(f"{n}.k", xt, wk, bk, {"weight": f"{n}.to_qkv.weight[{C}:{2 * C}]"})
    v = _gemm(f"{n}.v", xt, wv, bv, {"weight": f"{n}.to_qkv.weight[{2 * C}:{3 * C}]"})
    vt = _tr(f"{n}.vt", v, Lpad)  # [C, Lpad], zero columns past L
    s = _gemm(f"{n}.scores", q, k, None, {}, rec_b=True)  # [L, L]
    scale = 1.0 / math.sqrt(C)
    STATS["softmax"] += 1
    with REC.op(f"{n}.softmax", "softmax_rows", {"scale": scale, "ldp": Lpad, "impl": IMPL}, s=s) as o:
        pr = _mod().k_softmax_rows(s, Lpad, scale)  # [L, Lpad], zero columns past L
        o.out(y=pr)
    del s
    ov = _gemm(f"{n}.pv", pr, vt, None, {}, rec_b=True)  # [L, C]
    del pr
    po = _gemm(f"{n}.proj", ov, wp, bp, {"weight": f"{n}.proj.weight"})  # [L, C]
    pc = _tr(f"{n}.proj_t", po)  # [C, L]
    return add(pc.view(1, C, 1, H, W), x, f"{n}.add")


def _resup_forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
    n = _name(self)
    x_in = x
    for r in self.resnets:
        x = r(x)
    if self.upsampler is not None:
        x = self.upsampler(x)  # Resample: upsample3d is upsample2d for a single frame (no time_conv)
    if self.avg_shortcut is not None:
        x = add(x, self.avg_shortcut(x_in, first_chunk=first_chunk), f"{n}.shortcut_add")
    return x


def _decoder_forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
    x = self.conv_in(x)
    x = self.mid_block(x)
    for up in self.up_blocks:
        x = up(x, first_chunk=first_chunk)
    x = silu(self.norm_out(x), f"{_name(self)}.act_out")
    return self.conv_out(x)


def patch(vae):
    """Swap the decode path's arithmetic of `vae` (AutoencoderKLQwenImage21, bf16) for ours, in place; returns vae."""
    from torch import nn
    from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as m

    assert next(vae.parameters()).dtype == torch.bfloat16, "load the VAE in bf16"
    for name, mod in vae.named_modules():
        mod._stk_name = f"vae.{name}" if name else "vae"
    bind = lambda mod, fn: setattr(mod, "forward", types.MethodType(fn, mod))
    for name, mod in list(vae.decoder.named_modules()) + [("post_quant_conv", vae.post_quant_conv)]:
        if isinstance(mod, nn.Conv2d):  # includes QwenImage21CausalConv3d
            bind(mod, _conv_forward)
        elif isinstance(mod, m.QwenImage21RMS_norm):
            bind(mod, _norm_forward)
        elif isinstance(mod, m.QwenImage21Upsample):
            bind(mod, _upsample_forward)
        elif isinstance(mod, m.QwenImage21DupUp3D):
            bind(mod, _dupup_forward)
        elif isinstance(mod, m.QwenImage21ResidualBlock):
            bind(mod, _resblock_forward)
        elif isinstance(mod, m.QwenImage21AttentionBlock):
            bind(mod, _attn_forward)
        elif isinstance(mod, m.QwenImage21ResidualUpBlock):
            bind(mod, _resup_forward)
        elif isinstance(mod, m.QwenImage21Decoder3d):
            bind(mod, _decoder_forward)
        elif isinstance(mod, m.QwenImage21UpBlock):
            raise NotImplementedError("the non-residual up block is not part of Qwen-Image 2.1")
    return vae


# ---------------------------------------------------------------------------------------------- ends of the pipeline
def latents_to_z(vae, x: torch.Tensor) -> torch.Tensor:
    """Normalised latents [1, 64, h, w] bf16 -> z [1, 64, 1, h, w] = latents * std + mean (the pipeline's op, ours)."""
    mean = torch.tensor(vae.config.latents_mean).to(x.device, x.dtype)
    std = torch.tensor(vae.config.latents_std).to(x.device, x.dtype)
    xc = x.contiguous()
    STATS["chan_affine"] += 1
    with REC.op("vae.denorm", "chan_affine", {"impl": IMPL}, x=xc, mean=mean, std=std) as o:
        z = _mod().k_chan_affine(xc, mean, std)
        o.out(y=z)
    return z.unsqueeze(2)


def to_u8(img: torch.Tensor) -> torch.Tensor:
    """Decoded image [1, C, (1,) H, W] bf16 in [-1, 1] -> uint8 [H, W, C] on the GPU (the PIL pixels)."""
    xc = img.reshape(img.shape[1], img.shape[-2], img.shape[-1]).contiguous()
    with REC.op("vae.to_u8", "to_u8_hwc", {"impl": IMPL}, x=xc) as o:
        u = _mod().k_to_u8(xc)
        o.out(y=u)
    return u


@torch.inference_mode()
def decode(vae, x: torch.Tensor):
    """Latents [1, 64, h, w] (normalised, bf16) -> PIL image, every arithmetic op ours (`patch(vae)` first)."""
    from PIL import Image

    img = vae.decode(latents_to_z(vae, x), return_dict=False)[0]
    u = to_u8(img).cpu().numpy()
    return Image.fromarray(u)
