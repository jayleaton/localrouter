"""MiniMax H3-Image's DiT (ComfyUI e9afbccc `comfy/ldm/minimax/model.py`, `MiniMaxH3ImageModel`; docs/dev/H3-IMAGE.md
section 1.3 and 1.4) op for op on LocalRouter's kernels, batch 1, one latent frame, no audio stream.

What the image model shares with the video DiT is the video twin's code (`stk_twin.h3.dit.H3Dit`, inherited): the text
path (`prepare`: condition_proj, the 2-block token refiner, its final norm), the 50 blocks (`block` / `blocks_fused`),
the split-half RoPE table (`rope_rotation`), the curve time embedding (`curve_t_emb`) and the per-token modulation rows.
What it changes:
  layout       `[text L | reference images | target]`: text at (i, 0, 0), reference j at time L + j on its own
               area-normalized grid, the target at time L + n_refs (PackedLayout with audio_t 0, refs of kind "image")
  timesteps    sigma = float(timestep) / 1000, t = 1 - sigma (float64): text and target at t, references at
               max(t, 0.999); mod rows t_row * 3 + tag (text 1 or the TE's tags, images 0); no flow shift, no audio
  references   `prepare(..., refs=[z])`: each normalized latent patchified in fp32 with the condition noise
               0.999 r + 0.001 noise (the generator restarted at `seed` for each), through video_patch_proj (fp32), once
  final layer  RMSNorm + the fp32 curve modulation of the target rows, then the PDD head of the step: row block 0 of
               `final_layer.video_out` plus the step's dt-weighted offsets, mixed on the host in fp32 by `sampler.mix`
               from `sampler.step`'s plan, on our fp32 GEMM; unpatchified and negated (`-out`)

Linears bf16 on our bf16 GEMM, attention our bf16 kernel (`attn="bf16"`; kitchen's INT8 with `"int8"`, the GPU only).
`ops` picks the kernel provider: `h3.dit.GpuOps` (default) or `cpu_ops.CpuOps` (the CPU structure gate). Every op is
recorded (`REC`) under the video twin's names where the op is the same (`L{i}.*`, `video_rows`, `video_patch_proj`,
`final.adaln`, `final.mod_video`, `final.video_out`, `video_out`) and `ref{j}.cond_rows`, `ref{j}.patch_proj`,
`final.pdd_mix` for the image model's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..rec import REC
from ..h3.dit import (EPS, GPU, HD, LINEARS, SOURCE, VISUAL_COND_TIMESTEP, H3Dit, _axis, curve_t_emb, fused,
                      load_checkpoint, rope_rotation)
from . import sampler

C_LAT = 24                     # latent channels
VDIM = C_LAT * 4               # one PDD head's rows: 24 channels x 1 x 2 x 2

__all__ = ["H3ImageDit", "ImageLayout", "frame_grid", "load_checkpoint", "VDIM"]


def frame_grid(h: int, w: int) -> np.ndarray:
    """ComfyUI's _frame_grid: the (h, w) coordinates of a latent frame's 2x2-patch rows, area-normalized, fp64 [n, 2]."""
    area = float(np.sqrt(h * w))
    hh, ww = np.meshgrid(_axis(h, area), _axis(w, area), indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], -1)


@dataclass
class ImageLayout:
    """ComfyUI's PackedLayout for the image model: [text L | refs | target (H/2)(W/2)], fp64 positions [S, 3]."""
    L: int
    H: int
    W: int
    refs: tuple = ()                       # (latent h, latent w) of each reference image
    text_tags: np.ndarray | None = None
    pos: np.ndarray = field(init=False)
    S: int = field(init=False)
    segments: list = field(init=False)     # (start, stop, kind): "text", "ref_img", "video"

    def __post_init__(self):
        text = np.zeros((self.L, 3))
        text[:, 0] = np.arange(self.L)
        parts, segs, row = [text], [(0, self.L, "text")], self.L
        for j, (rh, rw) in enumerate(self.refs):
            g = frame_grid(rh, rw)
            p = np.empty((g.shape[0], 3))
            p[:, 0] = float(self.L + j)
            p[:, 1:] = g
            parts.append(p)
            segs.append((row, row + g.shape[0], "ref_img"))
            row += g.shape[0]
        g = frame_grid(self.H, self.W)
        p = np.empty((g.shape[0], 3))
        p[:, 0] = float(self.L + len(self.refs))
        p[:, 1:] = g
        parts.append(p)
        segs.append((row, row + g.shape[0], "video"))
        self.pos = np.concatenate(parts)
        self.S = self.pos.shape[0]
        self.segments = segs

    @property
    def target_rows(self) -> tuple[int, int]:
        return self.segments[-1][0], self.segments[-1][1]

    def mod_rows(self, t: float) -> tuple[list[float], np.ndarray, int]:
        """The unique timesteps (text and target t, references max(t, 0.999)), every token's int32 modulation row
        (t_row * 3 + tag) and the target's t_row (the final layer's row)."""
        t_ref = max(t, VISUAL_COND_TIMESTEP)
        unique_t = sorted({t} | ({t_ref} if self.refs else set()))
        t_row = {v: i for i, v in enumerate(unique_t)}
        idx = np.empty(self.S, dtype=np.int32)
        for a, b, kind in self.segments:
            if kind == "text":
                idx[a:b] = t_row[t] * 3 + (self.text_tags if self.text_tags is not None else 1)
            elif kind == "ref_img":
                idx[a:b] = t_row[t_ref] * 3
            else:
                idx[a:b] = t_row[t] * 3
        return unique_t, idx, t_row[t]


class H3ImageDit(H3Dit):
    """The image DiT on device: `prepare(text, refs=...)` once a run, `__call__(x, step)` once a step with the step's
    plan from `sampler.step`."""

    def __init__(self, w: dict[str, torch.Tensor], *, pdd_sigmas=sampler.PDD_SIGMAS, attn: str = "bf16", device="cuda",
                 ops=None):
        self.ops = ops or GPU
        self.dev, self.kind, self.attn = torch.device(device), "bf16", attn
        act = self.ops.act
        if any(k.startswith(("audio_patch_proj.", "final_layer.audio_out.")) for k in w):
            raise ValueError("an audio stream in the checkpoint: this is the video DiT, not H3-Image")
        g = lambda k, dt: w[k].to(self.dev, dt).contiguous()
        self.D = w["video_patch_proj.weight"].shape[0]
        self.HEADS = w["blocks.0.attn.qkv_proj.weight"].shape[0] // (3 * HD)
        self.layers = 1 + max(int(k.split(".")[1]) for k in w if k.startswith("blocks."))
        self.table = w["adaln_t_table"].float().cpu().numpy()
        self.inv_freq = w["rope.inv_freq"].float().cpu().numpy()
        self.side = {k: g(k, torch.float32) for k in ("video_patch_proj.weight", "video_patch_proj.bias",
                                                     "final_layer.video_out.weight", "final_layer.video_out.bias",
                                                     "final_layer.adaln_proj.linear.weight",
                                                     "final_layer.adaln_proj.linear.bias")}
        self.side["final_layer.norm.weight"] = g("final_layer.norm.weight", act)
        self.refiner = {k: g(k, act) for k in w if k.startswith(("condition_proj.", "token_refiner."))}
        self.blocks = []
        for i in range(self.layers):
            p = f"blocks.{i}."
            b = {"norm1": g(p + "norm1.weight", torch.float32), "norm2": g(p + "norm2.weight", torch.float32),
                 "q_norm": g(p + "attn.q_norm.weight", act), "k_norm": g(p + "attn.k_norm.weight", act),
                 "ada_w": g(p + "adaln_proj.linear.weight", torch.float32),
                 "ada_b": g(p + "adaln_proj.linear.bias", torch.float32)}
            for name in LINEARS:
                b[name] = self._linear(w[p + SOURCE[name] + ".weight"], None)
            self.blocks.append(b)
        # the PDD bank: [n, 96, D] rows and [n, 96] biases, fp32 on the host for the mix
        self.grid = tuple(float(s) for s in pdd_sigmas)
        vw = w["final_layer.video_out.weight"].float().cpu().numpy()
        vb = w["final_layer.video_out.bias"].float().cpu().numpy()
        self.pdd_steps = vw.shape[0] // VDIM
        if vw.shape[0] != self.pdd_steps * VDIM or (self.pdd_steps > 1 and len(self.grid) != self.pdd_steps + 1):
            raise ValueError(f"video_out has {vw.shape[0]} rows: {self.pdd_steps} heads for a grid of {len(self.grid)} points")
        self.rows = vw.reshape(self.pdd_steps, VDIM, -1)
        self.brows = vb.reshape(self.pdd_steps, VDIM)
        self._heads: dict = {}
        self.context = None
        self.first_frame = None
        self.cond_rows = None
        self.text_tags = None
        self.ref_embed = None
        self.ref_shapes: tuple = ()
        self._layout_key = None

    @classmethod
    def from_pack(cls, dir: str | Path, attn: str = "bf16", device="cuda", ops=None) -> "H3ImageDit":
        """The bf16 pack of `build.py` (its names back to the checkpoint's)."""
        import json

        from safetensors.torch import load_file

        dir = Path(dir)
        t = load_file(str(dir / "weights.safetensors"))
        man = json.loads((dir / "manifest.json").read_text())
        w = {k: v for k, v in t.items() if not k.startswith("L")}
        layers = 1 + max(int(k[1:].split(".")[0]) for k in t if k.startswith("L"))
        for i in range(layers):
            p, q = f"blocks.{i}.", f"L{i}."
            w[p + "norm1.weight"], w[p + "norm2.weight"] = t[q + "norm1"], t[q + "norm2"]
            w[p + "attn.q_norm.weight"], w[p + "attn.k_norm.weight"] = t[q + "q_norm"], t[q + "k_norm"]
            w[p + "adaln_proj.linear.weight"], w[p + "adaln_proj.linear.bias"] = t[q + "ada_w"], t[q + "ada_b"]
            for name in LINEARS:
                w[p + SOURCE[name] + ".weight"] = t[q + name + ".weight"]
        return cls(w, pdd_sigmas=man["pdd_sigmas"], attn=attn, device=device, ops=ops)

    # ---------------------------------------------------------------- text and references (once a run)
    def prepare(self, text: torch.Tensor, *, refs=(), seed: int = 0, text_tags=None, ref_noise=None) -> torch.Tensor:
        """Qwen3-VL layer-50 states [1, L, text_dim] -> refined text rows (the video twin's `prepare`), then each
        reference's rows: `refs` normalized latents [1, 24, 1, h, w] fp32 (even h, w), `ref_noise` (optional) the fp32
        noise of each, else LocalRouter's portable generator at `seed` restarted for each (ComfyUI: torch.randn from a CPU
        generator at the sampler's seed; a declared substitution)."""
        ctx = super().prepare(text, text_tags=text_tags)
        m = self.ops.mod()
        embeds, shapes = [], []
        for j, z in enumerate(refs):
            z = z.to(self.dev, torch.float32).contiguous()
            if z.dim() != 5 or tuple(z.shape[:3]) != (1, C_LAT, 1) or z.shape[3] % 2 or z.shape[4] % 2:
                raise ValueError("a reference must be a normalized [1, 24, 1, h, w] latent with even sides")
            if ref_noise is not None:
                n = ref_noise[j].to(self.dev, torch.float32).reshape(-1).contiguous()
            else:
                from ..qwen_image import noise
                n = torch.from_numpy(noise(seed, z.numel())).to(self.dev).contiguous()
            with REC.op(f"ref{j}.cond_rows", "condition_rows", {"seed": seed, "aug": VISUAL_COND_TIMESTEP}, z=z, noise=n) as o:
                r = m.k_cond_rows(z, n, VISUAL_COND_TIMESTEP)
                o.out(y=r)
            e = self._gemm32(f"ref{j}.patch_proj", r, self.side["video_patch_proj.weight"], self.side["video_patch_proj.bias"])
            embeds.append(e.to(self.ops.act))
            shapes.append((z.shape[3], z.shape[4]))
        self.ref_embed = torch.cat(embeds).contiguous() if embeds else None
        self.ref_shapes = tuple(shapes)
        self._layout_key = None
        return ctx

    # ---------------------------------------------------------------- one step
    def image_layout(self, Hh: int, Ww: int) -> ImageLayout:
        key = (self.context.shape[0], Hh, Ww, self.ref_shapes)
        if key != self._layout_key:
            self._lay = ImageLayout(self.context.shape[0], Hh, Ww, self.ref_shapes, text_tags=self.text_tags)
            self._rope = rope_rotation(self._lay.pos, self.inv_freq, self.ops.act).to(self.dev)
            self._layout_key = key
        return self._lay

    def head(self, step: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's PDD head [96, D] and bias [96], fp32: `sampler.mix` of the bank over the step's heads and weights."""
        if self.pdd_steps == 1:
            return self.side["final_layer.video_out.weight"], self.side["final_layer.video_out.bias"]
        key = (tuple(step["heads"]), tuple(step["w"]))
        if key not in self._heads:
            if any(not 0 < k < self.pdd_steps for k in step["heads"]):
                raise ValueError(f"step heads {step['heads']} outside the bank of {self.pdd_steps}")
            hw, hb = sampler.mix(self.rows, self.brows, list(step["heads"]), list(step["w"]))
            self._heads[key] = (torch.from_numpy(np.ascontiguousarray(hw)).to(self.dev),
                                torch.from_numpy(np.ascontiguousarray(hb)).to(self.dev))
        return self._heads[key]

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor, step: dict) -> torch.Tensor:
        """x [1, 24, 1, H, W] in the activation dtype (bf16 on the GPU; even H, W), `step` = sampler.step(grid, sigmas, i)
        -> the velocity [1, 24, 1, H, W], negated as ComfyUI returns it."""
        m = self.ops.mod()
        _, c, T, Hh, Ww = x.shape
        if c != C_LAT or T != 1 or Hh % 2 or Ww % 2:
            raise ValueError("x must be [1, 24, 1, H, W] with even latent sides")
        lay = self.image_layout(Hh, Ww)
        t = float(step["t"])
        unique_t, idx_np, row_t = lay.mod_rows(t)
        idx = torch.from_numpy(idx_np).to(self.dev)
        t_emb = torch.from_numpy(curve_t_emb(self.table, unique_t)).to(self.dev)
        REC.note("step", timestep=step["timestep"], sigma=step["sigma"], t=t, unique_t=unique_t, heads=list(step["heads"]),
                 w=list(step["w"]), start=step["start"], stop=step["stop"])

        with REC.op("video_rows", "patchify", {}, x=x) as o:
            vr = m.k_patchify(x.contiguous())
            o.out(y=vr)
        ve = self._gemm32("video_patch_proj", vr, self.side["video_patch_proj.weight"], self.side["video_patch_proj.bias"])
        pieces = [self.context] + ([self.ref_embed] if self.ref_embed is not None else []) + [ve.to(self.ops.act)]
        h = torch.cat(pieces).contiguous()

        if fused():
            h = self.blocks_fused(h, t_emb, idx, lay.S)
        else:
            for i, b in enumerate(self.blocks):
                h = self.block(i, b, h, t_emb, idx, lay.S)

        ada = self._gemm32("final.adaln", t_emb, self.side["final_layer.adaln_proj.linear.weight"],
                           self.side["final_layer.adaln_proj.linear.bias"])
        shift, scale = ada[:, :self.D], ada[:, self.D:]
        v0, v1 = lay.target_rows
        with REC.op("final.mod_video", "final_mod", {"row": row_t}, x=h[v0:v1]) as o:
            fv = m.k_final_mod(h[v0:v1].contiguous(), self.side["final_layer.norm.weight"], scale[row_t].contiguous(),
                               shift[row_t].contiguous(), EPS)
            o.out(y=fv)
        with REC.op("final.pdd_mix", "pdd_mix", {"start": step["start"], "stop": step["stop"], "heads": list(step["heads"]),
                                                 "w": list(step["w"])}) as o:
            hw, hb = self.head(step)
            o.out(w=hw, b=hb)
        v = self._gemm32("final.video_out", fv, hw, hb)
        with REC.op("video_out", "unpatchify_neg", {}, x=v) as o:
            vel = m.k_unpatchify_neg(v, C_LAT, 1, Hh, Ww)
            o.out(y=vel)
        return vel
