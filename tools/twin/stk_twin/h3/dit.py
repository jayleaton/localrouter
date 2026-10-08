"""MiniMax H3's DiT (ComfyUI 0.37.0 `comfy/ldm/minimax/model.py`, the curve-form checkpoint) op for op on LocalRouter's
kernels, text to video + audio, batch 1: ComfyUI's `forward` (the audio carry) and `_forward` (packing, the curve
time embedding, per-token modulation rows, RoPE table, 50 blocks, the fp32 heads), with tfvideo's block arithmetic.

Linears: "bf16" (the checkpoint's weights on our bf16 GEMM; the structure gate against ComfyUI) or "nvfp4" (TensorFold
W4A4 under static input scales; the shipping form). Attention: "bf16" (our attention kernel) or "int8" (comfy-kitchen's
INT8 attention, copied). Every GPU op is a kernel the Zig engine launches too; host-side scalars follow ComfyUI's fp32
tensor arithmetic in numpy float32. Each op is recorded (`REC`) under the names the Zig replay matches.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .. import ops as tops
from ..attn import attention as bf16_attention
from ..rec import REC
from ..smath import sincos
from . import ops as H

F32 = np.float32
LAYERS, D, HEADS, HD, FFN = 50, 5376, 56, 128, 14336
EPS = 1e-5
SHIFT_V, SHIFT_A = 12.0, 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
LINEARS = ("qkv", "out", "fc1", "fc2")
SOURCE = {"qkv": "attn.qkv_proj", "out": "attn.out_proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}


# ------------------------------------------------------------------------------------------------ host-side scalars
def f32(x) -> np.float32:
    return np.float32(x)


def time_shift_sigma(sigma: np.float32, from_shift: float, to_shift: float) -> np.float32:
    """ComfyUI's time_shift_sigma on a 0-d fp32 tensor, op by op (Python floats meet fp32 tensors as fp32)."""
    base = sigma / (f32(from_shift) + sigma * f32(1.0 - from_shift))
    return (f32(to_shift) * base) / (f32(1.0) + f32(to_shift - 1.0) * base)


def step_scalars(sigma: float) -> dict:
    """Everything a step derives from the sampler's sigma (fp32): timestep = sigma * 1000, the video and audio
    sigmas, t_v / t_a, the carry and the un-carry factor (ComfyUI `forward` / `_forward`)."""
    s = f32(sigma)
    timestep = s * f32(1000.0)
    sigma_v = max(timestep / f32(1000.0), f32(1e-6))
    sigma_a = time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A)
    scale = SHIFT_V / SHIFT_A
    carry_bf16 = float(torch.tensor(float(sigma_a / sigma_v), dtype=torch.float32).to(torch.bfloat16))
    uncarry_bf16 = float(torch.tensor(float(f32(1.0) + f32(scale - 1.0) * sigma_a)).to(torch.bfloat16))
    return {"sigma_v": float(sigma_v), "sigma_a": float(sigma_a), "t_v": float(f32(1.0) - sigma_v),
            "t_a": float(f32(1.0) - sigma_a), "carry": carry_bf16, "c1": 1.0 - scale, "c2": uncarry_bf16}


def curve_t_emb(table: np.ndarray, t_vals: list[float]) -> np.ndarray:
    """The curve form's t_emb: torch.lerp between the two grid rows around t * (grid - 1), fp32 (lerp's two-branch
    formula, unfused)."""
    g = table.shape[0]
    out = np.empty((len(t_vals), table.shape[1]), dtype=F32)
    for r, t in enumerate(t_vals):
        pos = min(max(f32(t), f32(0.0)), f32(1.0)) * f32(g - 1)
        i0 = min(int(np.floor(pos)), g - 2)
        w = pos - f32(i0)
        a, b = table[i0], table[i0 + 1]
        out[r] = (a + w * (b - a)) if w < f32(0.5) else (b - (b - a) * (f32(1.0) - w))
    return out


# ------------------------------------------------------------------------------------------------ the packed layout
def _axis(dim: int, area: float) -> np.ndarray:
    ratio, n = dim / area, dim // 2
    return (np.arange(n, dtype=np.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


@dataclass
class Layout:
    """ComfyUI's PackedLayout for text to video: [text L | audio 2A | video T*(H/2)*(W/2)], fp64 positions."""
    L: int
    T: int
    H: int
    W: int
    A: int
    pos: np.ndarray = field(init=False)
    S: int = field(init=False)

    def __post_init__(self):
        area = math.sqrt(self.H * self.W)
        hax, wax = _axis(self.H, area), _axis(self.W, area)
        hh, ww = np.meshgrid(hax, wax, indexing="ij")
        frame = np.stack([hh.reshape(-1), ww.reshape(-1)], -1)
        text = np.zeros((self.L, 3))
        text[:, 0] = np.arange(self.L)
        cursor = float(self.L)
        audio = np.zeros((2 * self.A, 3))
        audio[:, 0] = np.tile(cursor + np.arange(self.A, dtype=np.float64), 2)
        audio[: self.A, 2], audio[self.A:, 2] = wax[0], wax[-1]
        spans = np.array([FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(self.T)], dtype=np.float64)
        tgrid = cursor + np.concatenate([[0.0], np.cumsum(spans[:-1])])
        video = np.empty((self.T, frame.shape[0], 3))
        video[:, :, 0] = tgrid[:, None]
        video[:, :, 1:] = frame[None]
        self.pos = np.concatenate([text, audio, video.reshape(-1, 3)])
        self.S = self.pos.shape[0]

    @property
    def audio_rows(self) -> tuple[int, int]:
        return self.L, self.L + 2 * self.A

    @property
    def video_rows(self) -> tuple[int, int]:
        return self.L + 2 * self.A, self.S

    def rope_table(self, inv_freq: np.ndarray) -> torch.Tensor:
        """[1, S, 1, 48, 2, 2] bf16: ComfyUI's rope_rotation_table of the fp32 angles pos * inv_freq, cos / sin by the
        toolkit's portable sincos (in f64 of the fp32 angle, rounded to fp32), then bf16."""
        ang = (self.pos.astype(F32)[:, :, None] * inv_freq.astype(F32)[None, None, :]).reshape(self.S, 48)
        c = np.empty_like(ang)
        s = np.empty_like(ang)
        for idx, a in np.ndenumerate(ang):
            s[idx], c[idx] = sincos(float(a))
        table = np.stack([c, -s, s, c], -1).reshape(1, self.S, 1, 48, 2, 2)
        return torch.from_numpy(table).to(torch.bfloat16)

    def mod_rows(self, t_v: float, t_a: float) -> tuple[list[float], np.ndarray, int, int]:
        """unique timesteps, the int32 [S] modulation row of every token (t_row * 3 + tag: video 0, text 1,
        audio 2), and the final layer's rows for video and audio."""
        unique_t = sorted({t_v, t_a})
        t_row = {t: i for i, t in enumerate(unique_t)}
        idx = np.empty(self.S, dtype=np.int32)
        a0, a1 = self.audio_rows
        idx[: self.L] = t_row[t_v] * 3 + 1
        idx[a0:a1] = t_row[t_a] * 3 + 2
        idx[a1:] = t_row[t_v] * 3 + 0
        return unique_t, idx, t_row[t_v], t_row[t_a]


# ------------------------------------------------------------------------------------------------ the model
class H3Dit:
    """The DiT on device: `prepare(context)` once a run (condition_proj + token refiner), `__call__(video, audio,
    sigma)` once a step."""

    def __init__(self, w: dict[str, torch.Tensor], kind: str = "bf16", attn: str = "bf16", acts: dict | None = None,
                 device="cuda", lins: dict | None = None):
        self.dev, self.kind, self.attn = torch.device(device), kind, attn
        g = lambda k, dt=None: (w[k] if dt is None else w[k].to(dt)).to(self.dev).contiguous()
        self.table = w["adaln_t_table"].float().cpu().numpy()
        self.inv_freq = w["rope.inv_freq"].float().cpu().numpy()
        self.side = {k: g(k, torch.float32) for k in ("video_patch_proj.weight", "video_patch_proj.bias",
                                                     "audio_patch_proj.weight", "audio_patch_proj.bias",
                                                     "final_layer.video_out.weight", "final_layer.video_out.bias",
                                                     "final_layer.audio_out.weight", "final_layer.audio_out.bias",
                                                     "final_layer.adaln_proj.linear.weight",
                                                     "final_layer.adaln_proj.linear.bias")}
        self.side["final_layer.norm.weight"] = g("final_layer.norm.weight")
        self.refiner = {k: g(k) for k in w if k.startswith(("condition_proj.", "token_refiner."))}
        self.blocks = []
        for i in range(LAYERS):
            p = f"blocks.{i}."
            b = {"norm1": g(p + "norm1.weight", torch.float32), "norm2": g(p + "norm2.weight", torch.float32),
                 "q_norm": g(p + "attn.q_norm.weight"), "k_norm": g(p + "attn.k_norm.weight"),
                 "ada_w": g(p + "adaln_proj.linear.weight", torch.float32),
                 "ada_b": g(p + "adaln_proj.linear.bias", torch.float32)}
            for name in LINEARS:
                if lins is not None:
                    b[name] = lins[(i, name)]
                    continue
                wt = w[p + SOURCE[name] + ".weight"]
                b[name] = self._linear(wt, None if acts is None else acts.get(f"{i}.{name}"))
            self.blocks.append(b)
        self.context = None
        self._layout_key = None

    @classmethod
    def from_pack(cls, dir: str | Path, attn: str = "int8", device="cuda") -> "H3Dit":
        """The shipping form from its pack: NVFP4 linears from the stored codes and static scales (the bytes the Zig
        engine loads), everything else under the checkpoint's names."""
        from tensorfold.cuda.nvfp4.linear import Fp4Linear

        from .pack import read

        t, _ = read(dir)
        w = {k: v for k, v in t.items() if not k.startswith("L")}
        lins = {}
        for i in range(LAYERS):
            p, q = f"blocks.{i}.", f"L{i}."
            w[p + "norm1.weight"], w[p + "norm2.weight"] = t[q + "norm1"], t[q + "norm2"]
            w[p + "attn.q_norm.weight"], w[p + "attn.k_norm.weight"] = t[q + "q_norm"], t[q + "k_norm"]
            w[p + "adaln_proj.linear.weight"], w[p + "adaln_proj.linear.bias"] = t[q + "ada_w"], t[q + "ada_b"]
            for name in LINEARS:
                k = q + name
                lins[(i, name)] = _Fp4(Fp4Linear.from_checkpoint(
                    t[k + ".codes"].to(device), t[k + ".scales"].to(device), float(t[k + ".global"]),
                    act=float(t[k + ".act"])))
        return cls(w, kind="nvfp4", attn=attn, device=device, lins=lins)

    def _linear(self, wt: torch.Tensor, act: float | None):
        if self.kind == "bf16":
            return wt.to(self.dev, torch.bfloat16).contiguous()
        if self.kind == "nvfp4":
            from tfvideo.linear import Nvfp4Linear

            if act is None:
                raise ValueError("an NVFP4 linear needs its static input scale (calibrate first)")
            return Nvfp4Linear(wt.to(self.dev, torch.bfloat16), act=act)
        raise ValueError(f"linear kind {self.kind!r}")

    def lin(self, name: str, lin, x: torch.Tensor) -> torch.Tensor:
        with REC.op(name, "linear", {"kind": self.kind}, x=x) as o:
            y = tops.gemm_bf16(x, lin) if self.kind == "bf16" else lin(x.contiguous())
            o.out(y=y)
        return y

    # ---------------------------------------------------------------- text (once a run)
    def prepare(self, text: torch.Tensor) -> torch.Tensor:
        """Qwen3-VL layer-50 states [1, L, 5120] (bf16) -> refined text rows [L, 5376]: condition_proj, the 2-block
        token refiner (no RoPE), final norm."""
        r = self.refiner
        x = text[0].to(self.dev, torch.bfloat16).contiguous()
        with REC.op("refiner.condition_proj", "linear", {"kind": "bf16"}, x=x) as o:
            x = tops.gemm_bf16(x, r["condition_proj.weight"], r["condition_proj.bias"])
            o.out(y=x)
        L = x.shape[0]
        for i in range(2):
            p = f"token_refiner.blocks.{i}."
            n = self._rms(f"refiner.{i}.norm1", x, r[p + "norm1.weight"])
            qkv = self._gemm(f"refiner.{i}.qkv", n, r[p + "attn.qkv_proj.weight"])
            q, k, v = (t.contiguous().view(L, HEADS, HD) for t in qkv.split(HEADS * HD, dim=-1))
            q = self._rms(f"refiner.{i}.q_norm", q, r[p + "attn.q_norm.weight"])
            k = self._rms(f"refiner.{i}.k_norm", k, r[p + "attn.k_norm.weight"])
            with REC.op(f"refiner.{i}.attention", "attention", {"impl": "bf16"}, q=q, k=k, v=v) as o:
                a = bf16_attention(q, k, v)
                o.out(y=a)
            a = self._gemm(f"refiner.{i}.out", a.view(L, -1), r[p + "attn.out_proj.weight"])
            x = self._add(f"refiner.{i}.attn_residual", a, x)
            n = self._rms(f"refiner.{i}.norm2", x, r[p + "norm2.weight"])
            gu = self._gemm(f"refiner.{i}.fc1", n, r[p + "mlp.fc1.weight"])
            with REC.op(f"refiner.{i}.swiglu", "silu_mul_split", {}, x=gu) as o:
                m = H.mod().k_silu_mul_split(gu)
                o.out(y=m)
            m = self._gemm(f"refiner.{i}.fc2", m, r[p + "mlp.fc2.weight"])
            x = self._add(f"refiner.{i}.mlp_residual", m, x)
        self.context = self._rms("refiner.final_norm", x, r["token_refiner.final_norm.weight"])
        return self.context

    def _rms(self, name, x, w):
        with REC.op(name, "rms_norm", {"eps": EPS}, x=x) as o:
            y = H.mod().k_rms_norm(x.contiguous(), w, EPS)
            o.out(y=y)
        return y

    def _gemm(self, name, x, wt, bias=None):
        with REC.op(name, "linear", {"kind": "bf16"}, x=x) as o:
            y = tops.gemm_bf16(x, wt, bias)
            o.out(y=y)
        return y

    def _add(self, name, a, b):
        with REC.op(name, "add", {}, x=a, z=b) as o:
            y = H.mod().k_add(a, b)
            o.out(y=y)
        return y

    # ---------------------------------------------------------------- one step
    def layout(self, T: int, Hh: int, Ww: int, A: int) -> Layout:
        key = (self.context.shape[0], T, Hh, Ww, A)
        if key != self._layout_key:
            self._lay = Layout(*key)
            self._rope = self._lay.rope_table(self.inv_freq).to(self.dev)
            self._layout_key = key
        return self._lay

    @torch.inference_mode()
    def __call__(self, video: torch.Tensor, audio: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
        """video bf16 [1, 24, T, H, W], audio bf16 [1, 32, 2, A] (the sampler's carried audio), sigma (fp32 value)
        -> (video velocity, audio velocity) bf16, negated as ComfyUI returns them, the audio back on the carry."""
        m = H.mod()
        sc = step_scalars(sigma)
        _, _, T, Hh, Ww = video.shape
        A = audio.shape[-1]
        lay = self.layout(T, Hh, Ww, A)
        with REC.op("carry", "scale", {"c": sc["carry"]}, x=audio) as o:
            a_in = m.k_scale(audio.contiguous(), sc["carry"])
            o.out(y=a_in)
        unique_t, idx_np, row_v, row_a = lay.mod_rows(sc["t_v"], sc["t_a"])
        idx = torch.from_numpy(idx_np).to(self.dev)
        t_emb = torch.from_numpy(curve_t_emb(self.table, unique_t)).to(self.dev)
        REC.note("step", sigma=sigma, **sc, unique_t=unique_t)

        # embed: fp32 patch projections, rounded to bf16, [text | audio | video]
        with REC.op("video_rows", "patchify", {}, x=video) as o:
            vr = m.k_patchify(video.contiguous())
            o.out(y=vr)
        with REC.op("audio_rows", "pack_audio", {}, x=a_in) as o:
            ar = m.k_pack_audio(a_in)
            o.out(y=ar)
        ve = self._gemm32("video_patch_proj", vr, self.side["video_patch_proj.weight"], self.side["video_patch_proj.bias"])
        ae = self._gemm32("audio_patch_proj", ar, self.side["audio_patch_proj.weight"], self.side["audio_patch_proj.bias"])
        h = torch.cat([self.context, ae.to(torch.bfloat16), ve.to(torch.bfloat16)]).contiguous()  # exact rounds + moves

        for i, b in enumerate(self.blocks):
            h = self.block(i, b, h, t_emb, idx, lay.S)

        # final layer: fp32 curve modulation of each target segment, fp32 heads, negated, back to the latents
        ada = self._gemm32("final.adaln", t_emb, self.side["final_layer.adaln_proj.linear.weight"],
                           self.side["final_layer.adaln_proj.linear.bias"])
        shift, scale = ada[:, :D], ada[:, D:]
        (a0, a1), (v0, v1) = lay.audio_rows, lay.video_rows
        with REC.op("final.mod_video", "final_mod", {"row": row_v}, x=h[v0:v1]) as o:
            fv = m.k_final_mod(h[v0:v1].contiguous(), self.side["final_layer.norm.weight"], scale[row_v].contiguous(),
                               shift[row_v].contiguous(), EPS)
            o.out(y=fv)
        with REC.op("final.mod_audio", "final_mod", {"row": row_a}, x=h[a0:a1]) as o:
            fa = m.k_final_mod(h[a0:a1].contiguous(), self.side["final_layer.norm.weight"], scale[row_a].contiguous(),
                               shift[row_a].contiguous(), EPS)
            o.out(y=fa)
        v = self._gemm32("final.video_out", fv, self.side["final_layer.video_out.weight"], self.side["final_layer.video_out.bias"])
        a = self._gemm32("final.audio_out", fa, self.side["final_layer.audio_out.weight"], self.side["final_layer.audio_out.bias"])
        with REC.op("video_out", "unpatchify_neg", {}, x=v) as o:
            vel_v = m.k_unpatchify_neg(v, 24, T, Hh, Ww)
            o.out(y=vel_v)
        with REC.op("audio_out", "unpack_audio_neg", {}, x=a) as o:
            vel_a = m.k_unpack_audio_neg(a, 32, A)
            o.out(y=vel_a)
        with REC.op("uncarry", "uncarry", {"c1": sc["c1"], "c2": sc["c2"]}, a=a_in, v=vel_a) as o:
            m.k_uncarry_(a_in, vel_a, sc["c1"], sc["c2"])
            o.out(y=vel_a)
        return vel_v, vel_a

    def _gemm32(self, name, x, wt, bias):
        with REC.op(name, "gemm_f32", {}, x=x) as o:
            y = H.mod().k_gemm_f32(x.contiguous(), wt, bias)
            o.out(y=y)
        return y

    def block(self, i: int, b: dict, x: torch.Tensor, t_emb: torch.Tensor, idx: torch.Tensor, S: int) -> torch.Tensor:
        m, p = H.mod(), f"L{i}"
        mod = self._gemm32(f"{p}.adaln", t_emb, b["ada_w"], b["ada_b"]).view(-1, 6, D).contiguous()  # [M*3, 6, D]
        h = torch.empty_like(x)
        with REC.op(f"{p}.norm_mod1", "norm_mod", {"part": 0, "eps": EPS}, x=x, mod=mod, idx=idx) as o:
            m.k_norm_mod(x, b["norm1"], mod, idx, h, 0, EPS)
            o.out(y=h)
        qkv = self.lin(f"{p}.qkv", b["qkv"], h)                                   # [S, 3 * 7168]
        q = qkv[:, : HEADS * HD].view(1, S, HEADS, HD)
        k = qkv[:, HEADS * HD: 2 * HEADS * HD].view(1, S, HEADS, HD)
        v = qkv[:, 2 * HEADS * HD:].view(S, HEADS, HD)
        from .kitchen import int8_attention, rms_rope_split_half_

        with REC.op(f"{p}.rms_rope", "rms_rope", {"eps": EPS, "rot_dim": 96}, qkv=qkv) as o:
            rms_rope_split_half_(q, k, self._rope, b["q_norm"], b["k_norm"], EPS, 96)
            o.out(qkv=qkv)
        with REC.op(f"{p}.attention", "attention", {"impl": self.attn}, qkv=qkv) as o:
            if self.attn == "int8":
                qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q[0], k[0], v))
                att = int8_attention(qh, kh, vh)[0].transpose(0, 1).reshape(S, HEADS * HD)
            else:
                att = bf16_attention(q[0].contiguous(), k[0].contiguous(), v.contiguous()).view(S, HEADS * HD)
            att = att.contiguous()
            o.out(y=att)
        out = self.lin(f"{p}.out", b["out"], att)
        with REC.op(f"{p}.gate_add1", "gate_add", {"part": 0}, x=x, y=out) as o:
            m.k_gate_add_(x, out, mod, idx, 0)
            o.out(x=x)
        with REC.op(f"{p}.norm_mod2", "norm_mod", {"part": 1, "eps": EPS}, x=x, mod=mod, idx=idx) as o:
            m.k_norm_mod(x, b["norm2"], mod, idx, h, 1, EPS)
            o.out(y=h)
        gu = self.lin(f"{p}.fc1", b["fc1"], h)
        with REC.op(f"{p}.swiglu", "swiglu", {}, x=gu) as o:
            a = m.k_swiglu(gu)
            o.out(y=a)
        y = self.lin(f"{p}.fc2", b["fc2"], a)
        with REC.op(f"{p}.gate_add2", "gate_add", {"part": 1}, x=x, y=y) as o:
            m.k_gate_add_(x, y, mod, idx, 1)
            o.out(x=x)
        return x


class _Fp4:
    """A pack's NVFP4 linear on TensorFold's prompt GEMM (tile 12), the call tfvideo's Nvfp4Linear makes under a
    static scale."""

    def __init__(self, lin):
        self.lin = lin

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from tfvideo.linear import _ck

        return _ck().prompt(_ck().A4, x, self.lin, tile=12)


def load_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """The DiT checkpoint's tensors (CPU), the ComfyUI `diffusion_model.` prefix stripped if present."""
    from safetensors.torch import load_file

    w = load_file(str(path))
    return {k.removeprefix("model.diffusion_model.").removeprefix("diffusion_model."): v for k, v in w.items()}


def load_acts(path: str | Path) -> dict[str, float]:
    return json.loads(Path(path).read_text())
