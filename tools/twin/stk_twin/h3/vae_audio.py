"""MiniMax H3's audio VAE decoder (ComfyUI 0.37.0 MiniMaxH3AudioVAE.decode, a BigVGAN, + vae_decode_audio's std
normalisation) on LocalRouter's own kernels (kernels/cuda/minimax/vae_audio.cu, plus h3_gemm_f32 from ops.cu), fp32
throughout, no TF32, launched as the Zig engine launches them. See tools/twin/AVAE-PORT.md for every op's arithmetic.

    vae = AudioVAE.from_checkpoint("$VMODELS/vae/minimax_h3_audio_vae_fp32.safetensors")
    wav = vae.decode(latent)          # latent fp32 [1, 32, 2, A] (after process_latent_out) -> fp32 [1, 2, A * 800]

Every op runs inside `REC.op` with a stable name:
    avae.latent_in                    z * std + mean, to channel-last rows
    avae.dec_in_proj                  Conv1d(32 -> 2048, k 1)
    avae.conv_pre                     Conv1d(2048 -> 1024, k 7)
    avae.ups.{i}                      ConvTranspose1d of stage i (i = 0..6)
    avae.res.{i}.{j}.{n}.a1           Activation1d (up x2, SnakeBeta, down x2) of AMP block j (0..2) of stage i, layer n (0..2)
    avae.res.{i}.{j}.{n}.c1           the layer's dilated Conv1d (convs1[n])
    avae.res.{i}.{j}.{n}.a2           Activation1d (activations[2n + 1])
    avae.res.{i}.{j}.{n}.c2           the layer's Conv1d, dilation 1 (convs2[n])
    avae.res.{i}.{j}.{n}.add          x = c2 output + x
    avae.avg.{i}                      (rb0 + rb1) + rb2, / 3
    avae.post                         activation_post (Activation1d with SnakeBeta(8))
    avae.conv_post                    Conv1d(8 -> 1, k 7, no bias)
    avae.clamp                        clamp(-1, 1)
    avae.std_scale / avae.div_scale   vae_decode_audio's std normalisation
(the AMP block's activations[2n] is a1 and activations[2n + 1] is a2; resblocks[3 i + j] is block j of stage i).
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

from ..rec import REC

KDIR = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[4] / "kernels")) / "cuda" / "minimax"

# torch's `xs.div_(3)` on CUDA multiplies by the fp32 reciprocal (the divide-by-CPU-scalar optimisation). The test
# checks this on the pod's torch and fails if the choice is wrong; flip it there if it is.
AVG3_RECIP = True

_DECLS = """
at::Tensor k_latent_in(at::Tensor z, at::Tensor mean, at::Tensor stdv);
at::Tensor k_im2col(at::Tensor x, int64_t K, int64_t dil, int64_t pad);
at::Tensor k_ct_im2col(at::Tensor x, int64_t J, int64_t qoff);
void k_ct_store_(at::Tensor ph, at::Tensor y, int64_t u, int64_t pad, int64_t r, int64_t qoff);
at::Tensor k_up2(at::Tensor x, at::Tensor f);
at::Tensor k_down2(at::Tensor x, at::Tensor f);
at::Tensor k_snake(at::Tensor x, at::Tensor pa, at::Tensor pb);
at::Tensor k_add(at::Tensor a, at::Tensor b);
at::Tensor k_avg3(at::Tensor a, at::Tensor b, at::Tensor c, int64_t recip);
at::Tensor k_clamp(at::Tensor x);
at::Tensor k_std_scale(at::Tensor audio);
at::Tensor k_div_scale(at::Tensor audio, at::Tensor sc);
at::Tensor k_gemm_f32(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias);
"""

_LAUNCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "ops.cu"
#include "vae_audio.cu"
#define F32(t) (t).data_ptr<float>()
#define STREAM at::cuda::getCurrentCUDAStream()
static unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }
static void chk(const at::Tensor& t, const char* n) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(), n, ": contiguous CUDA fp32 tensor");
}
at::Tensor k_latent_in(at::Tensor z, at::Tensor mean, at::Tensor stdv) {
    chk(z, "z"); chk(mean, "mean"); chk(stdv, "std");
    TORCH_CHECK(z.dim() == 4, "z [Bb, C, S, T]");
    long long Bb = z.size(0), C = z.size(1), S = z.size(2), T = z.size(3);
    auto rows = at::empty({Bb * S, T, C}, z.options());
    avae_latent_in<<<cdiv(rows.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(z), F32(mean), F32(stdv), F32(rows), Bb, C, S, T);
    return rows;
}
at::Tensor k_im2col(at::Tensor x, int64_t K, int64_t dil, int64_t pad) {
    chk(x, "x");
    TORCH_CHECK(x.dim() == 3 && 2 * pad == dil * (K - 1), "im2col: x [B, T, C], a same-padded conv");
    long long B = x.size(0), T = x.size(1), C = x.size(2);
    auto col = at::empty({B * T, K * C}, x.options());
    avae_im2col<<<cdiv(col.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(x), F32(col), B, T, C, (int)K, (int)dil, (int)pad);
    return col;
}
at::Tensor k_ct_im2col(at::Tensor x, int64_t J, int64_t qoff) {
    chk(x, "x");
    TORCH_CHECK(x.dim() == 3, "ct_im2col: x [B, L, C]");
    long long B = x.size(0), L = x.size(1), C = x.size(2);
    auto col = at::empty({B * L, J * C}, x.options());
    avae_ct_im2col<<<cdiv(col.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(x), F32(col), B, L, C, (int)J, (int)qoff);
    return col;
}
void k_ct_store_(at::Tensor ph, at::Tensor y, int64_t u, int64_t pad, int64_t r, int64_t qoff) {
    chk(ph, "ph"); chk(y, "y");
    long long B = y.size(0), Cout = y.size(2), L = y.size(1) / u;
    TORCH_CHECK(ph.size(0) == B * L && ph.size(1) == Cout, "ct_store shapes");
    avae_ct_store<<<cdiv(ph.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(ph), F32(y), B, L, Cout, (int)u, (int)pad, (int)r, (int)qoff);
}
at::Tensor k_up2(at::Tensor x, at::Tensor f) {
    chk(x, "x"); chk(f, "f");
    TORCH_CHECK(x.dim() == 3 && f.numel() == AVAE_FILT, "up2: x [B, T, C], 12 taps");
    long long B = x.size(0), T = x.size(1), C = x.size(2);
    auto y = at::empty({B, 2 * T, C}, x.options());
    avae_up2<<<cdiv(y.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(x), F32(f), F32(y), T, C, y.numel());
    return y;
}
at::Tensor k_down2(at::Tensor x, at::Tensor f) {
    chk(x, "x"); chk(f, "f");
    TORCH_CHECK(x.dim() == 3 && x.size(1) % 2 == 0 && f.numel() == AVAE_FILT, "down2: x [B, T2, C], T2 even, 12 taps");
    long long B = x.size(0), T2 = x.size(1), C = x.size(2);
    auto y = at::empty({B, T2 / 2, C}, x.options());
    avae_down2<<<cdiv(y.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(x), F32(f), F32(y), T2, C, y.numel());
    return y;
}
at::Tensor k_snake(at::Tensor x, at::Tensor pa, at::Tensor pb) {
    chk(x, "x"); chk(pa, "alpha"); chk(pb, "beta");
    long long C = x.size(-1);
    TORCH_CHECK(pa.numel() == C && pb.numel() == C, "snake: per-channel alpha / beta");
    auto y = at::empty_like(x);
    avae_snake<<<cdiv(x.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(x), F32(pa), F32(pb), F32(y), C, x.numel());
    return y;
}
at::Tensor k_add(at::Tensor a, at::Tensor b) {
    chk(a, "a"); chk(b, "b");
    TORCH_CHECK(a.sizes() == b.sizes(), "add shapes");
    auto y = at::empty_like(a);
    avae_add<<<cdiv(a.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(a), F32(b), F32(y), a.numel());
    return y;
}
at::Tensor k_avg3(at::Tensor a, at::Tensor b, at::Tensor c, int64_t recip) {
    chk(a, "a"); chk(b, "b"); chk(c, "c");
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), "avg3 shapes");
    auto y = at::empty_like(a);
    avae_avg3<<<cdiv(a.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(a), F32(b), F32(c), F32(y), a.numel(), (int)recip);
    return y;
}
at::Tensor k_clamp(at::Tensor x) {
    chk(x, "x");
    auto y = at::empty_like(x);
    avae_clamp<<<cdiv(x.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(x), F32(y), x.numel());
    return y;
}
at::Tensor k_std_scale(at::Tensor audio) {
    chk(audio, "audio");
    TORCH_CHECK(audio.dim() == 2, "std_scale: audio [Bb, n]");
    auto sc = at::empty({audio.size(0)}, audio.options());
    avae_std_scale<<<(unsigned)audio.size(0), AVAE_BLOCK, 0, STREAM>>>(F32(audio), F32(sc), audio.size(1));
    return sc;
}
at::Tensor k_div_scale(at::Tensor audio, at::Tensor sc) {
    chk(audio, "audio"); chk(sc, "sc");
    TORCH_CHECK(audio.dim() == 2 && sc.numel() == audio.size(0), "div_scale shapes");
    auto y = at::empty_like(audio);
    avae_div_scale<<<cdiv(audio.numel(), AVAE_BLOCK), AVAE_BLOCK, 0, STREAM>>>(F32(audio), F32(sc), F32(y), audio.size(1), audio.numel());
    return y;
}
at::Tensor k_gemm_f32(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias) {
    chk(a, "a"); chk(b, "b");
    long long M = a.size(0), N = b.size(0), K = a.size(1);
    TORCH_CHECK(b.size(1) == K, "gemm_f32 shapes");
    auto c = at::empty({M, N}, a.options());
    const float* bp = nullptr;
    if (bias.has_value()) { chk(*bias, "bias"); bp = F32(*bias); }
    h3_gemm_f32<<<dim3(cdiv(N, 64), cdiv(M, 64)), 256, 0, STREAM>>>(F32(a), F32(b), bp, F32(c), M, N, K);
    return c;
}
"""

NAMES = ["k_latent_in", "k_im2col", "k_ct_im2col", "k_ct_store_", "k_up2", "k_down2", "k_snake", "k_add", "k_avg3",
         "k_clamp", "k_std_scale", "k_div_scale", "k_gemm_f32"]


@lru_cache(maxsize=1)
def mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    # --fmad=false: nothing contracts behind our back (the kernels use explicit __f*_rn / __fmaf_rn anyway; this pins
    # libdevice's expf / sinf). No --use_fast_math, no -ftz, IEEE division and sqrt (the nvcc defaults).
    flags = ["-O3", "--fmad=false", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"]
    digest = hashlib.sha256((KDIR / "ops.cu").read_bytes() + (KDIR / "vae_audio.cu").read_bytes() + _LAUNCH.encode()
                            + " ".join(flags).encode()).hexdigest()
    return load_inline(f"stk_h3_avae_{digest[:12]}", cpp_sources=_DECLS, cuda_sources=f"// {digest}\n" + _LAUNCH,
                       functions=NAMES, extra_cuda_cflags=flags, extra_include_paths=[str(KDIR)], with_cuda=True)


# ------------------------------------------------------------------------------------------------ layers

class Conv1d:
    """Stride-1 same-padded Conv1d (zero padding, dilation d, pad (k d - d) / 2) as im2col + h3_gemm_f32 on channel-last
    activations [B, T, Cin] -> [B, T, Cout]. The weight [Cout, Cin, K] is repacked to [Cout, K * Cin] (tap-major), so an
    output is one fmaf chain over k ascending (outer), ci ascending (inner), then + bias."""

    def __init__(self, key: str, w: torch.Tensor, bias: torch.Tensor | None, dilation: int = 1):
        self.key, self.bias, self.dil = key, bias, dilation
        self.cout, self.cin, self.k = w.shape
        self.pad = (self.k * dilation - dilation) // 2
        self.w = w.permute(0, 2, 1).reshape(self.cout, self.k * self.cin).contiguous()

    def __call__(self, x: torch.Tensor, name: str) -> torch.Tensor:
        m = mod()
        B, T, _ = x.shape
        attrs = {"weight": self.key, "cin": self.cin, "cout": self.cout, "k": self.k, "dilation": self.dil,
                 "pad": self.pad, "bias": self.bias is not None}
        with REC.op(name, "conv1d", attrs, x=x) as o:
            col = x.reshape(B * T, self.cin) if self.k == 1 else m.k_im2col(x, self.k, self.dil, self.pad)
            y = m.k_gemm_f32(col, self.w, self.bias).view(B, T, self.cout)
            o.out(y=y)
        return y


class ConvTranspose1d:
    """ConvTranspose1d (stride u, kernel K, padding (K - u) / 2) on [B, L, Cin] -> [B, u L, Cout] as u phase GEMMs
    (see avae_ct_im2col); an output is one fmaf chain over the phase's taps j ascending (k = r + j u ascending, outer),
    ci ascending (inner), then + bias. The weight [Cin, Cout, K] is repacked per phase to [Cout, J * Cin]."""

    def __init__(self, key: str, w: torch.Tensor, bias: torch.Tensor, stride: int):
        self.key, self.bias, self.u = key, bias, stride
        self.cin, self.cout, self.k = w.shape
        self.pad = (self.k - stride) // 2
        assert 0 <= self.pad < stride, "phase decomposition assumes 0 <= padding < stride"
        self.phases = []
        for r in range(stride):
            J = -(-(self.k - r) // stride)
            wp = torch.stack([w[:, :, r + j * stride] for j in range(J)], 0)  # [J, Cin, Cout]
            self.phases.append((J, 1 if r < self.pad else 0, wp.permute(2, 0, 1).reshape(self.cout, J * self.cin).contiguous()))

    def __call__(self, x: torch.Tensor, name: str) -> torch.Tensor:
        m = mod()
        B, L, _ = x.shape
        attrs = {"weight": self.key, "cin": self.cin, "cout": self.cout, "k": self.k, "stride": self.u, "pad": self.pad}
        with REC.op(name, "conv_transpose1d", attrs, x=x) as o:
            y = torch.empty(B, self.u * L, self.cout, device=x.device, dtype=torch.float32)
            for r, (J, qoff, wp) in enumerate(self.phases):
                col = m.k_ct_im2col(x, J, qoff)
                m.k_ct_store_(m.k_gemm_f32(col, wp, self.bias), y, self.u, self.pad, r, qoff)
            o.out(y=y)
        return y


class Activation1d:
    """upsample x2 (replicate pad, depthwise 12-tap polyphase, * 2, crop) -> SnakeBeta -> downsample x2 (replicate pad,
    depthwise 12-tap, stride 2). [B, T, C] -> [B, T, C]; the intermediate [B, 2T, C] is not recorded."""

    def __init__(self, key: str, pa, pb, up, down):
        self.key, self.pa, self.pb, self.up, self.down = key, pa, pb, up.reshape(-1).contiguous(), down.reshape(-1).contiguous()

    def __call__(self, x: torch.Tensor, name: str) -> torch.Tensor:
        m = mod()
        with REC.op(name, "activation1d", {"act": self.key}, x=x) as o:
            y = m.k_down2(m.k_snake(m.k_up2(x, self.up), self.pa, self.pb), self.down)
            o.out(y=y)
        return y


# ------------------------------------------------------------------------------------------------ the decoder

UP_RATES = (5, 5, 2, 2, 2, 2, 2)
RES_KERNELS = (3, 7, 11)
RES_DILATIONS = (1, 3, 5)


class AudioVAE:
    def __init__(self, sd: dict[str, torch.Tensor]):
        g = lambda k: sd[k].float().contiguous()
        self.mean, self.std = g("latents_mean"), g("latents_std")
        self.dec_in = Conv1d("dec_in_proj.weight", g("dec_in_proj.weight"), g("dec_in_proj.bias"))
        self.conv_pre = Conv1d("decoder.conv_pre.weight", g("decoder.conv_pre.weight"), g("decoder.conv_pre.bias"))
        self.ups = [ConvTranspose1d(f"decoder.ups.{i}.0.weight", g(f"decoder.ups.{i}.0.weight"),
                                    g(f"decoder.ups.{i}.0.bias"), u) for i, u in enumerate(UP_RATES)]

        def act(p: str) -> Activation1d:
            return Activation1d(p, g(f"{p}.act.alpha"), g(f"{p}.act.beta"), g(f"{p}.upsample.filter"),
                                g(f"{p}.downsample.lowpass.filter"))

        self.res = []  # res[i][j] = [(a1, c1, a2, c2) for n in 0..2]
        for i in range(len(UP_RATES)):
            stage = []
            for j, k in enumerate(RES_KERNELS):
                p = f"decoder.resblocks.{3 * i + j}"
                stage.append([(act(f"{p}.activations.{2 * n}"),
                               Conv1d(f"{p}.convs1.{n}.weight", g(f"{p}.convs1.{n}.weight"), g(f"{p}.convs1.{n}.bias"),
                                      RES_DILATIONS[n]),
                               act(f"{p}.activations.{2 * n + 1}"),
                               Conv1d(f"{p}.convs2.{n}.weight", g(f"{p}.convs2.{n}.weight"), g(f"{p}.convs2.{n}.bias"), 1))
                              for n in range(3)])
            self.res.append(stage)
        self.post = act("decoder.activation_post")
        self.conv_post = Conv1d("decoder.conv_post.weight", g("decoder.conv_post.weight"), None)

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike, device: str = "cuda") -> "AudioVAE":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device=device) as f:
            keep = [k for k in f.keys() if k.startswith(("decoder.", "dec_in_proj.")) or k in ("latents_mean", "latents_std")]
            return cls({k: f.get_tensor(k) for k in keep})

    def _amp(self, x: torch.Tensor, i: int, j: int) -> torch.Tensor:
        for n, (a1, c1, a2, c2) in enumerate(self.res[i][j]):
            p = f"avae.res.{i}.{j}.{n}"
            xt = c2(a2(c1(a1(x, p + ".a1"), p + ".c1"), p + ".a2"), p + ".c2")
            with REC.op(p + ".add", "add", {}, a=xt, b=x) as o:
                x = mod().k_add(xt, x)
                o.out(y=x)
        return x

    def decode_raw(self, latent: torch.Tensor) -> torch.Tensor:
        """latent fp32 [Bb, 32, S, A] -> clamped waveform fp32 [Bb, S, A * 800] (before vae_decode_audio's normalisation)."""
        m = mod()
        assert latent.dtype == torch.float32 and latent.dim() == 4
        latent = latent.contiguous()
        Bb, _, S, _ = latent.shape
        with REC.op("avae.latent_in", "latent_in", {}, z=latent) as o:
            x = m.k_latent_in(latent, self.mean, self.std)
            o.out(y=x)
        x = self.dec_in(x, "avae.dec_in_proj")
        x = self.conv_pre(x, "avae.conv_pre")
        for i in range(len(UP_RATES)):
            x = self.ups[i](x, f"avae.ups.{i}")
            rs = [self._amp(x, i, j) for j in range(3)]
            with REC.op(f"avae.avg.{i}", "avg3", {"recip": AVG3_RECIP}, a=rs[0], b=rs[1], c=rs[2]) as o:
                x = m.k_avg3(rs[0], rs[1], rs[2], int(AVG3_RECIP))
                o.out(y=x)
        x = self.post(x, "avae.post")
        x = self.conv_post(x, "avae.conv_post")
        with REC.op("avae.clamp", "clamp", {}, x=x) as o:
            x = m.k_clamp(x)
            o.out(y=x)
        return x.reshape(Bb, S, -1)

    def normalize(self, audio: torch.Tensor) -> torch.Tensor:
        """vae_decode_audio: std = std(audio over [S, samples], unbiased) * 5, floored at 1, audio / std. [Bb, S, L]."""
        m = mod()
        flat = audio.reshape(audio.shape[0], -1).contiguous()
        with REC.op("avae.std_scale", "std_scale", {}, x=flat) as o:
            sc = m.k_std_scale(flat)
            o.out(sc=sc)
        with REC.op("avae.div_scale", "div_scale", {}, x=flat, sc=sc) as o:
            y = m.k_div_scale(flat, sc)
            o.out(y=y)
        return y.view_as(audio)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent fp32 [1, 32, 2, A] (the sampler's audio after * 0.25) -> waveform fp32 [1, 2, A * 800] at 32 kHz."""
        return self.normalize(self.decode_raw(latent))
