"""Qwen3-VL vision tower on shared kernels; 72-wide heads padded to the existing 128-wide attention.

Inputs are processor patch rows in merge-block order, not RGB pixels. Host plans the learned
position interpolation and axial rotary tables; all GPU arithmetic is enclosed by REC.op.
The final merger normalizes patch rows before reshaping, deepstack mergers after reshaping.
"""
from __future__ import annotations

import math

import torch

from . import ops, vision_ops
from .attn import attention
from .rec import REC
from .smath import sincos


def grids(grid_thw) -> list[tuple[int, int, int]]:
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.device.type != "cpu":
            raise ValueError("vision grid must be host metadata")
        grid_thw = grid_thw.tolist()
    if len(grid_thw) == 3 and all(isinstance(v, (int, float)) for v in grid_thw):
        grid_thw = [grid_thw]
    result = []
    for row in grid_thw:
        if len(row) != 3 or any(int(v) != v for v in row):
            raise ValueError("vision grid must contain integer (t,h,w) rows")
        t, h, w = map(int, row)
        if min(t, h, w) <= 0 or h % 2 or w % 2:
            raise ValueError("vision grid requires positive t and positive even h,w")
        result.append((t, h, w))
    if not result:
        raise ValueError("vision grid is empty")
    return result


def merge_coordinates(grid_thw) -> torch.Tensor:
    """CPU int64 [R,2], contiguous 2x2 patches of each spatial merge followed by time."""
    coords = []
    for t, h, w in grids(grid_thw):
        frame = [(bh+i, bw+j) for bh in range(0, h, 2) for bw in range(0, w, 2)
                 for i in range(2) for j in range(2)]
        coords.extend(frame * t)
    return torch.tensor(coords, dtype=torch.int64)


def position_plan(grid_thw, side: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU indices int32 and interpolation coefficients bf16 [4,R], align_corners=True.

    Coefficients round to bf16 before multiplying the learned table, matching reference
    fast_pos_embed_interpolate. linspace and every coefficient arithmetic run only on CPU.
    """
    if side <= 0:
        raise ValueError("position table side must be positive")
    all_ids, all_weights = [], []
    for t, h, w in grids(grid_thw):
        hs, ws = torch.linspace(0, side-1, h, dtype=torch.float32), torch.linspace(0, side-1, w, dtype=torch.float32)
        hf, wf = hs.to(torch.int64), ws.to(torch.int64)
        hc, wc = (hf+1).clamp_max(side-1), (wf+1).clamp_max(side-1)
        dh, dw = hs-hf, ws-wf
        ids = torch.stack([(hf[:, None]*side+wf).flatten(), (hf[:, None]*side+wc).flatten(),
                           (hc[:, None]*side+wf).flatten(), (hc[:, None]*side+wc).flatten()])
        weights = torch.stack([((1-dh)[:, None]*(1-dw)).flatten(), ((1-dh)[:, None]*dw).flatten(),
                               (dh[:, None]*(1-dw)).flatten(), (dh[:, None]*dw).flatten()])
        # Permute a raster grid into exactly the processor's merge-block order, then repeat frames.
        order = torch.arange(h*w).view(h//2, 2, w//2, 2).permute(0, 2, 1, 3).reshape(-1)
        all_ids.append(ids[:, order].repeat(1, t))
        all_weights.append(weights[:, order].repeat(1, t))
    return torch.cat(all_ids, 1).to(torch.int32), torch.cat(all_weights, 1).to(torch.bfloat16)


def rope_tables(grid_thw, inv_freq: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU f32 [R,36]: 18 row then 18 column phases for rotate_half of a 72-wide head.

    Pass the reference's frozen inv_freq to avoid recomputing backend-dependent powers.
    Frequencies multiply in fp32; portable sincos computes f64 then rounds to fp32.
    """
    import numpy as np

    if inv_freq is None:
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float32) / 36))
    inv = inv_freq.detach().cpu().float().numpy()
    if inv.shape != (18,):
        raise ValueError("vision RoPE requires 18 inverse frequencies")
    coords = merge_coordinates(grid_thw).numpy()
    cos, sin = np.empty((len(coords), 36), np.float32), np.empty((len(coords), 36), np.float32)
    for r, pair in enumerate(coords):
        phase = (np.asarray(pair, dtype=np.float32)[:, None] * inv).reshape(-1)
        for j, f in enumerate(phase):
            s, c = sincos(float(f))
            cos[r, j], sin[r, j] = c, s
    return torch.from_numpy(cos), torch.from_numpy(sin)


class VisionEncoder:
    """Uses weights by reference from transformers' visual module or a containing text encoder."""

    def __init__(self, visual, device="cuda"):
        if not hasattr(visual, "patch_embed"):
            visual = visual.model.visual
        self.visual, self.dev = visual, torch.device(device)
        self.heads = visual.blocks[0].attn.num_heads
        self.hidden = visual.pos_embed.weight.shape[1]
        if self.hidden != self.heads * 72 or visual.spatial_merge_size != 2:
            raise ValueError("vision kernels require 72-wide heads and spatial_merge_size=2")
        self.side = math.isqrt(visual.pos_embed.weight.shape[0])
        if self.side**2 != visual.pos_embed.weight.shape[0]:
            raise ValueError("learned vision position table must be square")
        self.inv = visual.rotary_pos_emb.inv_freq.detach().cpu().clone()
        self.taps = list(visual.deepstack_visual_indexes)
        if self.taps != sorted(set(self.taps)) or any(i < 0 or i >= len(visual.blocks) for i in self.taps):
            raise ValueError("deepstack indexes must be unique ascending block indexes")
        cfg = getattr(visual, "config", None)
        if cfg is not None and getattr(cfg, "hidden_act", "gelu_pytorch_tanh") not in ("gelu_pytorch_tanh", "gelu_new"):
            raise ValueError("vision blocks require tanh GELU")

    def _linear(self, name, x, layer):
        weight = layer.weight.reshape(layer.weight.shape[0], -1)
        with REC.op(name, "gemm_bf16", {"weight": name, "bias": layer.bias is not None}, x=x) as o:
            y = ops.gemm_bf16(x, weight, layer.bias)
            o.out(y=y)
        return y

    def _norm(self, name, x, norm):
        with REC.op(name, "vis_layer_norm", {"weight": name, "eps": norm.eps}, x=x) as o:
            y = vision_ops.layer_norm(x, norm.weight, norm.bias, norm.eps)
            o.out(y=y)
        return y

    def _add(self, name, x, z):
        with REC.op(name, "add", {}, x=x, z=z) as o:
            y = ops.add(x, z)
            o.out(y=y)
        return y

    def _gelu(self, name, x, exact=False):
        with REC.op(name, "vis_gelu" if exact else "gelu_tanh", {}, x=x) as o:
            y = vision_ops.gelu(x) if exact else ops.gelu_tanh(x)
            o.out(y=y)
        return y

    def _merge(self, name, h, merger, postshuffle):
        dim = self.hidden*4
        x = h.view(-1, dim) if postshuffle else h
        x = self._norm(f"{name}.norm", x, merger.norm).view(-1, dim)
        x = self._linear(f"{name}.linear_fc1", x, merger.linear_fc1)
        x = self._gelu(f"{name}.gelu", x, exact=True)
        return self._linear(f"{name}.linear_fc2", x, merger.linear_fc2)

    @torch.inference_mode()
    def forward(self, patches: torch.Tensor, grid_thw) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns merged [sum(t*h*w)/4,out_dim] bf16 plus one same-shaped tensor per tap.

        Each temporal grid slice attends independently, matching reference cu_seqlens.
        Image editing supplies t=1; multiple images are also supported without cross attention.
        """
        grid = grids(grid_thw)
        lengths = [h*w for t, h, w in grid for _ in range(t)]
        R = sum(lengths)
        proj = self.visual.patch_embed.proj
        K = proj.weight[0].numel()
        if patches.ndim != 2 or tuple(patches.shape) != (R, K):
            raise ValueError(f"expected merge-ordered patches [{R},{K}]")
        patches = patches.to(self.dev).contiguous()
        if patches.dtype == torch.float32:
            with REC.op("vis.patch_cast", "vis_from_f32", {}, x=patches) as o:
                patches = vision_ops.from_f32(patches)
                o.out(y=patches)
        elif patches.dtype != torch.bfloat16:
            raise ValueError("vision patches require fp32 or bf16")
        ids, weights = (x.to(self.dev) for x in position_plan(grid, self.side))
        cos, sin = (x.to(self.dev) for x in rope_tables(grid, self.inv))
        REC.note("vis_grid", grid=[list(g) for g in grid], lengths=lengths, taps=self.taps)
        h = self._linear("vis.patch_embed.proj", patches, proj)
        with REC.op("vis.position", "vis_position", {"weight": "vis.pos_embed"}, ids=ids, weights=weights) as o:
            pos = vision_ops.position(self.visual.pos_embed.weight, ids, weights)
            o.out(y=pos)
        h = self._add("vis.position_add", h, pos)
        deepstack = []
        for i, block in enumerate(self.visual.blocks):
            p = f"vis.blocks.{i}"
            x = self._norm(f"{p}.norm1", h, block.norm1)
            x = self._linear(f"{p}.attn.qkv", x, block.attn.qkv)
            with REC.op(f"{p}.rope_pad", "vis_qkv_rope_pad", {"heads": self.heads}, x=x, cos=cos, sin=sin) as o:
                q, k, v = vision_ops.qkv_rope_pad(x, cos, sin, self.heads)
                o.out(q=q, k=k, v=v)
            pieces, start = [], 0
            for j, length in enumerate(lengths):
                qq, kk, vv = q[start:start+length], k[start:start+length], v[start:start+length]
                with REC.op(f"{p}.attention.{j}", "attention", {"causal": False, "scale": 72**-0.5}, q=qq, k=kk, v=vv) as o:
                    at = attention(qq, kk, vv, causal=False, scale=72**-0.5)
                    o.out(y=at)
                pieces.append(at)
                start += length
            # Concatenation is data movement; unpadding has its own capture for engine replay.
            at = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            with REC.op(f"{p}.unpad", "vis_unpad", {}, x=at) as o:
                x = vision_ops.unpad(at)
                o.out(y=x)
            h = self._add(f"{p}.attn_residual", h, self._linear(f"{p}.attn.proj", x, block.attn.proj))
            x = self._norm(f"{p}.norm2", h, block.norm2)
            x = self._linear(f"{p}.mlp.linear_fc1", x, block.mlp.linear_fc1)
            x = self._gelu(f"{p}.mlp.gelu", x)
            x = self._linear(f"{p}.mlp.linear_fc2", x, block.mlp.linear_fc2)
            h = self._add(f"{p}.mlp_residual", h, x)
            if i in self.taps:
                j = self.taps.index(i)
                deepstack.append(self._merge(f"vis.deepstack_merger_list.{j}", h,
                                             self.visual.deepstack_merger_list[j], True))
        return self._merge("vis.merger", h, self.visual.merger, False), deepstack
