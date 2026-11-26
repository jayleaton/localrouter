"""CPU reference backends for the H3-Image twin: torch on the CPU, the arithmetic of exactly the kernels the image path
calls, for the structure gate (`gate_cpu.py`) on a machine without a GPU. Not a bit oracle: torch's CPU reductions and
matmuls do not keep the kernels' summation order, so these agree with the GPU kernels to rounding, not to the bit.

`act` is the activation dtype. At the GPU dtype (bf16 for the DiT, fp16 for the VAE) every value a kernel stores is rounded
where the kernel rounds it (fp32 inside); at torch.float32 nothing is rounded, which is the fp32 reference mode the gate
holds to ComfyUI's own modules run in fp32.

  CpuOps     the DiT's provider (h3/dit.py `GpuOps`'s five names): `mod()` -> the k_* of kernels/cuda/minimax/ops.cu that
             the DiT calls, `gemm_bf16` (gemm.cu), `attention` (qwen_image attention.cu), `rms_rope_split_half_` (kitchen)
  CpuVaeOps  the image decoder's provider (vae_image.py `GpuVaeOps`'s names): the video decoder's kernels
             (gemm_f16.cu, vae_video.cu) and the decoder ops that have no kernel yet
"""

from __future__ import annotations

import math

import torch

F32 = torch.float32


class CpuOps:
    """kernels/cuda/minimax/ops.cu (the k_* the DiT calls), the bf16 GEMM, the bf16 attention and kitchen's split-half
    RMS RoPE on the CPU. `mod()` returns the object itself, so `ops.mod().k_norm_mod(...)` reads as on the GPU."""

    def __init__(self, act: torch.dtype = torch.float32):
        self.act = act

    def mod(self):
        return self

    def _r(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.act)

    @staticmethod
    def _rrms(xf: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)

    # ------------------------------------------------------------------------------- ops.cu
    def k_norm_mod(self, x, w, mod, idx, y, part, eps):
        """h3_norm_mod: y = (x * rsqrt(mean(x^2) + eps) * w) * (1 + scale) + shift, fp32, one rounding."""
        xf = x.float()
        n = (xf * self._rrms(xf, eps)) * w.float()
        m = mod[idx.long()]                                       # [S, 6, D]
        y.copy_(self._r(n * (1.0 + m[:, 3 * part + 1]) + m[:, 3 * part]))

    def k_gate_add_(self, x, y, mod, idx, part):
        """h3_gate_add: x = round(x + y * gate), in place."""
        g = mod[idx.long()][:, 3 * part + 2]
        x.copy_(self._r(x.float() + y.float() * g))

    def k_gate_add_norm_mod_(self, x, y, gmod, nmod, idx, w, out, gpart, npart, eps):
        """h3_gate_add_norm_mod: the two kernels one after the other."""
        self.k_gate_add_(x, y, gmod, idx, gpart)
        self.k_norm_mod(x, w, nmod, idx, out, npart, eps)

    def k_swiglu(self, gu):
        """h3_swiglu (tfvideo): round(g / (1 + exp(-g)) * u), [gate | up]."""
        g, u = gu.float().chunk(2, dim=-1)
        return self._r(g / (1.0 + torch.exp(-g)) * u)

    def k_silu_mul_split(self, gu):
        """h3_silu_mul_split (ComfyUI's swiglu): round(round(silu(g)) * u)."""
        g, u = gu.float().chunk(2, dim=-1)
        return self._r(self._r(g / (1.0 + torch.exp(-g))).float() * u)

    def k_rms_norm(self, x, w, eps):
        """h3_rms_norm: round(x * rsqrt(mean(x^2) + eps) * w), rows of w.numel()."""
        xf = x.float()
        return self._r((xf * self._rrms(xf, eps)) * w.float())

    def k_final_mod(self, x, w, scale, shift, eps):
        """h3_final_mod: round(rms_norm(x) * w) * (1 + scale) + shift, fp32 out."""
        n = self.k_rms_norm(x, w, eps).float()
        return n * (1.0 + scale.float()) + shift.float()

    @staticmethod
    def _patchify(x):
        """[1, C, T, H, W] -> rows [T * H/2 * W/2, C * 4], column c * 4 + p * 2 + q (ComfyUI's patchify_video)."""
        _, c, t, h, w = x.shape
        r = x.reshape(c, t, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5)
        return r.reshape(t * (h // 2) * (w // 2), c * 4)

    def k_patchify(self, x):
        """h3_patchify: fp32 rows of the latent."""
        return self._patchify(x.float()).contiguous()

    def k_cond_rows(self, z, noise, aug):
        """h3_cond_rows: aug * rows + (1 - aug) * noise in fp32 (aug and 1 - aug as fp32 constants)."""
        r = self._patchify(z.float())
        a, b = torch.tensor(aug, dtype=F32), torch.tensor(1.0 - aug, dtype=F32)
        return (a * r + b * noise.float().view_as(r)).contiguous()

    def k_unpatchify_neg(self, rows, C, T, H, W):
        """h3_unpatchify_neg: round(-rows) back to [1, C, T, H, W]."""
        x = rows.float().reshape(T, H // 2, W // 2, C, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(1, C, T, H, W)
        return self._r(-x).contiguous()

    def k_add(self, a, b):
        return self._r(a.float() + b.float())

    def k_gemm_f32(self, a, b, bias):
        y = a.float() @ b.float().t()
        return y + bias.float() if bias is not None else y

    # ------------------------------------------------------------------------------- GEMM, attention, RoPE
    def gemm_bf16(self, a, b, bias=None):
        """gemm.cu: round(a . b^T (+ bias)), fp32 accumulation."""
        y = a.float() @ b.float().t()
        if bias is not None:
            y = y + bias.float()
        return self._r(y)

    def attention(self, q, k, v):
        """attention.cu, non-causal: q [W, H, 128], k / v [S, H, 128] -> round(softmax(q k^T / sqrt(128)) v), fp32."""
        qf, kf, vf = (t.float().transpose(0, 1) for t in (q, k, v))       # [H, *, 128]
        p = torch.softmax((qf @ kf.transpose(1, 2)) * (1.0 / math.sqrt(q.shape[-1])), dim=-1)
        return self._r((p @ vf).transpose(0, 1)).contiguous()

    def rms_rope_split_half_(self, q, k, freqs, qw, kw, eps, rot_dim):
        """kitchen's rms_rope_split_half_, in place on q, k [B, S, H, D] (views): x' = round(x * rrms * w), then on the
        first rot_dim dims y0 = f00 x0' + f01 x1', y1 = f10 x0' + f11 x1' over the pairs (i, i + rot_dim / 2), fp32,
        rounded; the rest x'. freqs [B|1, S, H|1, rot_dim / 2, 2, 2]."""
        f = freqs.float()
        half = rot_dim // 2
        for x, w in ((q, qw), (k, kw)):
            xf = x.float()
            xn = self._r((xf * self._rrms(xf, eps)) * w.float()).float()
            x0, x1 = xn[..., :half], xn[..., half:rot_dim]
            y0 = f[..., 0, 0] * x0 + f[..., 0, 1] * x1
            y1 = f[..., 1, 0] * x0 + f[..., 1, 1] * x1
            x.copy_(self._r(torch.cat([y0, y1, xn[..., rot_dim:]], dim=-1)))
        return q, k


class CpuVaeOps:
    """The image decoder's ops on the CPU (vae_image.py `GpuVaeOps`): gemm_f16.cu's GEMM with its bias and addcmul
    epilogues, vae_video.cu's denorm / gather / suffix / norms / SwiGLU / nan_to_num / blend / finalize, and the ops that
    have no kernel yet (the 2-D RoPE over 32 of 64 dims, the softmax attention as one op, the 768-column unshuffle). fp16
    or fp32 activations."""

    def __init__(self, act: torch.dtype = torch.float32):
        self.act = act
        self.dev = torch.device("cpu")

    def _r(self, x):
        return x.to(self.act)

    def _h(self, x):
        """round to the activation dtype and back to fp32 (vv_h)."""
        return x.to(self.act).float()

    def denorm(self, z, std, mean):
        """vv_denorm: round(round(z * std_c) + mean_c) on [C, ...]."""
        shape = (-1,) + (1,) * (z.dim() - 1)
        return self._r(self._h(z.float() * std.float().view(shape)) + mean.float().view(shape))

    def gather_rows(self, z, y0, h, x0, w):
        """vv_gather_rows (one frame): z [C, 1, Hz, Wz] -> rows [h * w, C], row y * w + x."""
        return z[:, 0, y0:y0 + h, x0:x0 + w].permute(1, 2, 0).reshape(h * w, z.shape[0]).contiguous()

    def linear(self, x, wt, bias, *, out=None, res=None, rscale=None):
        """gemm_f16.cu: round(x . wt^T + bias); with `res`: round(res + round(that) * rscale) (torch.addcmul)."""
        y = self._r(x.float() @ wt.float().t() + bias.float())
        if res is not None:
            y = self._r(res.float() + y.float() * rscale.float())
        if out is None:
            return y
        out[: y.shape[0]].copy_(y)
        return out

    def suffix_(self, hs, reg, nP):
        hs[nP:nP + 4].copy_(reg)
        hs[nP + 4].zero_()

    def rms_norm(self, x, w, eps):
        xf = x.float()
        return self._r(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float())

    def layer_norm(self, x, w, b, eps):
        xf = x.float()
        mean = xf.mean(-1, keepdim=True)
        d = xf - mean
        rstd = torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + eps)
        return self._r(d * rstd * w.float() + b.float())

    def rms_rope_(self, qkv, table, S, H, eps, rot):
        """kitchen's rms_rope_split_half_ on the qkv buffer [S, H * 192] (q +0, k +64 of a head), unit norm weight: x' =
        round(x * rsqrt(mean(x^2) + eps)), pairs (i, i + rot / 2) of the first `rot` dims rotated by the table
        [S, rot / 2, 4] (c, -s, s, c) in fp32 and rounded; dims rot .. 63 x'."""
        v = qkv[:S].view(S, H, 192)
        f = table.float().view(S, 1, rot // 2, 4)
        half = rot // 2
        for off in (0, 64):
            x = v[:, :, off:off + 64]
            xf = x.float()
            xn = self._h(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))
            x0, x1 = xn[..., :half], xn[..., half:rot]
            y0 = f[..., 0] * x0 + f[..., 1] * x1
            y1 = f[..., 2] * x0 + f[..., 3] * x1
            x.copy_(self._r(torch.cat([y0, y1, xn[..., rot:]], dim=-1)))

    def attention(self, qkv, S, H):
        """softmax(q k^T / 8) v per head from the rotated qkv [S, H * 192] -> rows [S, H * 64]: the scores rounded, the
        fp32 row softmax rounded, P v rounded (the video decoder's three steps), then nan_to_num."""
        v = qkv[:S].view(S, H, 192).float()
        q, k, vv = v[..., :64].transpose(0, 1), v[..., 64:128].transpose(0, 1), v[..., 128:].transpose(0, 1)
        s = self._h(q @ k.transpose(1, 2))
        p = self._h(torch.softmax(s * 0.125, dim=-1))
        o = self._r((p @ vv).transpose(0, 1).reshape(S, H * 64))
        return self.nan_to_num_(o)

    def nan_to_num_(self, x):
        big = torch.finfo(self.act).max
        return x.copy_(torch.nan_to_num(x, nan=0.0, posinf=big, neginf=-big))

    def swiglu(self, gu):
        """vv_swiglu: round(round(silu(g)) * u)."""
        g, u = gu.float().chunk(2, dim=-1)
        return self._r(self._h(g / (1.0 + torch.exp(-g))) * u)

    def unshuffle(self, rows, h, w):
        """proj_out rows [>= h * w, 768] -> tile [3, 1, 16 h, 16 w]: out[c][16y + py][16x + px] = rows[y w + x][(c 16 + py)
        16 + px] (ComfyUI's unpatchify with patch_size_t 1), a data move."""
        r = rows[: h * w].reshape(h, w, 3, 16, 16).permute(2, 0, 3, 1, 4)
        return r.reshape(3, 1, 16 * h, 16 * w).contiguous()

    def _blend(self, a, b, p, ext):
        """vv_blend: w_b = round(p * (1 / ext)), w_a = round(1 - w_b), round(round(a w_a) + round(b w_b))."""
        wb = self._h(p * torch.tensor(1.0 / ext, dtype=F32))
        wa = self._h(1.0 - wb)
        return self._h(self._h(a * wa) + self._h(b * wb))

    def place_tile(self, b, ytail, ey, ltail, ex, canvas, oy, ox, oh, ow):
        """vv_place_tile: rows r < ey blended along y with ytail's last ey rows, then columns c < ex along x with ltail's
        last ex columns; the region [0, oh) x [0, ow) written to canvas [3, F, Hc, Wc] at (oy, ox)."""
        v = b[:, :, :oh, :ow].float().clone()
        if ytail is not None and ey > 0:
            n = min(ey, oh)
            a = ytail[:, :, ytail.shape[2] - ey:ytail.shape[2] - ey + n, :ow].float()
            p = torch.arange(n, dtype=F32).view(1, 1, n, 1)
            v[:, :, :n] = self._blend(a, v[:, :, :n], p, ey)
        if ltail is not None and ex > 0:
            n = min(ex, ow)
            a = ltail[:, :, :oh, ltail.shape[3] - ex:ltail.shape[3] - ex + n].float()
            p = torch.arange(n, dtype=F32).view(1, 1, 1, n)
            v[:, :, :, :n] = self._blend(a, v[:, :, :, :n], p, ex)
        canvas[:, :, oy:oy + oh, ox:ox + ow] = self._r(v)

    def finalize(self, b, sm):
        """vv_finalize (one frame, no temporal blend): p = clamp(float(b) * std_c + mean_c, 0, 1) fp32 [H, W, 3] and
        u8 = trunc(clamp(p * 255, 0, 255))."""
        std = torch.tensor(sm[:3], dtype=F32).view(3, 1, 1)
        mean = torch.tensor(sm[3:], dtype=F32).view(3, 1, 1)
        p = (b[:, 0].float() * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).contiguous()
        u8 = torch.nan_to_num(p * 255.0, nan=0.0).clamp(0.0, 255.0).to(torch.uint8)
        return u8, p
