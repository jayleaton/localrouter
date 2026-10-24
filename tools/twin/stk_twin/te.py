"""The text encoder on LocalRouter's kernels: Qwen3-VL 8B's text model for a text-only prompt, op for op as
transformers runs it (Qwen3VLTextModel, 36 layers), every GPU op a kernel the Zig engine launches too (te.cu,
gemm.cu, attention.cu). The pipeline reads the last decoder layer's output before the final norm and drops the
template's system tokens; `TextEncoder.encode` returns exactly that.

Text only, one prompt: the multimodal RoPE reduces to 1D positions 0..L-1, attention is causal with no padding.
RoPE tables: the frozen inv_freq (kernels/qwen_image/te_inv_freq.json), freq = f32(inv * pos), cos / sin from `smath`
in f64 rounded to f32 (the kernel rounds them to bf16, as transformers casts them). The Zig engine builds the same.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from . import ops
from .attn import attention
from .rec import REC
from .smath import sincos

KROOT = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[3] / "kernels"))
HEAD = 128


def inv_freq() -> list[float]:
    return json.loads((KROOT / "qwen_image" / "te_inv_freq.json").read_text())["inv_freq"]


def rope_tables(n: int, positions: torch.Tensor | None = None, sections=(24, 20, 20)) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin [n,64] fp32, Qwen3-VL interleaved mrope (T default; H/W every third frequency)."""
    import numpy as np

    inv = np.array(inv_freq(), dtype=np.float32)
    if positions is None:
        positions = torch.arange(n, dtype=torch.float32).expand(3, n)
    if tuple(positions.shape) != (3, n) or sum(sections) != 64:
        raise ValueError("expected [3,L] positions and 64 mrope frequencies")
    axis = np.zeros(64, dtype=np.int64)
    axis[1:sections[1] * 3:3] = 1
    axis[2:sections[2] * 3:3] = 2
    pos = positions.cpu().numpy().astype(np.float32)
    cos, sin = np.empty((n, 64), dtype=np.float32), np.empty((n, 64), dtype=np.float32)
    for p in range(n):
        f = inv * pos[axis, p]
        for j in range(64):
            s, c = sincos(float(f[j]))
            sin[p, j], cos[p, j] = s, c
    return torch.from_numpy(cos), torch.from_numpy(sin)


class TextEncoder:
    """Over the weights of a loaded transformers Qwen3-VL text model (`pipe.text_encoder`), by reference."""

    def __init__(self, text_encoder, device="cuda"):
        lm = getattr(text_encoder.model, "language_model", text_encoder.model)
        cfg = lm.config
        assert cfg.head_dim == HEAD and not cfg.attention_bias
        self.dev = torch.device(device)
        self.eps = cfg.rms_norm_eps
        self.heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.embed = lm.embed_tokens.weight
        self.layers = list(lm.layers)
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        self.sections = tuple(rope.get("mrope_section", (24, 20, 20)))

    def _gemm(self, name: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        with REC.op(name, "gemm_bf16", {"weight": name}, x=x) as o:
            y = ops.gemm_bf16(x, w)
            o.out(y=y)
        return y

    def _norm(self, name: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        with REC.op(name, "rms_norm_hf", {"eps": self.eps, "weight": name}, x=x) as o:
            y = ops.rms_norm_hf(x, w, self.eps)
            o.out(y=y)
        return y

    def _add(self, name: str, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with REC.op(name, "add", {}, x=x, z=z) as o:
            y = ops.add(x, z)
            o.out(y=y)
        return y

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, positions: torch.Tensor | None = None, spans=(), embeds=(), deepstack=()) -> torch.Tensor:
        """ids [L] int -> [L, 4096] bf16: the last decoder layer's output (no final norm)."""
        L = ids.numel()
        cos, sin = (t.to(self.dev) for t in rope_tables(L, positions, self.sections))
        ids = ids.to(self.dev, torch.int32)
        with REC.op("te.embed", "embed", {}, ids=ids) as o:
            h = ops.embed(self.embed, ids)
            o.out(y=h)
        if len(spans) != len(embeds) or len(spans) != len(deepstack):
            raise ValueError("one embedding and deepstack bundle per image span")
        for j, (span, visual) in enumerate(zip(spans, embeds)):
            if visual.shape != (span.count, h.shape[1]) or len(deepstack[j]) != 3:
                raise ValueError("vision output does not match reserved token rows")
            self._copy(f"te.visual{j}.embed", h[span.start:span.start + span.count], visual)
        for i, layer in enumerate(self.layers):
            p, a, m = f"te.{i}", layer.self_attn, layer.mlp
            x = self._norm(f"{p}.input_layernorm", h, layer.input_layernorm.weight)
            q = self._gemm(f"{p}.q_proj", x, a.q_proj.weight).view(L, self.heads, HEAD)
            k = self._gemm(f"{p}.k_proj", x, a.k_proj.weight).view(L, self.kv_heads, HEAD)
            v = self._gemm(f"{p}.v_proj", x, a.v_proj.weight).view(L, self.kv_heads, HEAD)
            q = self._norm(f"{p}.q_norm", q, a.q_norm.weight)
            k = self._norm(f"{p}.k_norm", k, a.k_norm.weight)
            with REC.op(f"{p}.rope", "rope_half", {}, q=q, k=k, cos=cos, sin=sin) as o:
                ops.rope_half_(q, cos, sin)
                ops.rope_half_(k, cos, sin)
                o.out(q=q, k=k)
            with REC.op(f"{p}.attention", "attention", {"causal": True}, q=q, k=k, v=v) as o:
                at = attention(q, k, v, causal=True)
                o.out(y=at)
            h = self._add(f"{p}.attn_residual", h, self._gemm(f"{p}.o_proj", at.view(L, -1), a.o_proj.weight))
            x = self._norm(f"{p}.post_attention_layernorm", h, layer.post_attention_layernorm.weight)
            g = self._gemm(f"{p}.gate_proj", x, m.gate_proj.weight)
            u = self._gemm(f"{p}.up_proj", x, m.up_proj.weight)
            with REC.op(f"{p}.silu_mul", "silu_mul", {}, g=g, u=u) as o:
                gu = ops.silu_mul(g, u)
                o.out(y=gu)
            h = self._add(f"{p}.mlp_residual", h, self._gemm(f"{p}.down_proj", gu, m.down_proj.weight))
            if i < 3:
                for j, (span, bundle) in enumerate(zip(spans, deepstack)):
                    rows = h[span.start:span.start + span.count]
                    added = self._add(f"{p}.visual{j}.deepstack", rows, bundle[i])
                    self._copy(f"{p}.visual{j}.scatter", rows, added)
        return h

    def _copy(self, name: str, dst: torch.Tensor, src: torch.Tensor) -> None:
        with REC.op(name, "copy", {}, src=src) as o:
            ops.copy_rows(dst[None], src[None])
            o.out(y=dst)

    def forward_mm(self, ids, positions, spans, embeds, deepstack) -> torch.Tensor:
        return self.forward(ids, positions, spans, embeds, deepstack)


def prompt_ids(pipe, prompt: str) -> tuple[torch.Tensor, int]:
    """The pipeline's token ids for a text-only prompt (template applied) and the number of system tokens dropped."""
    text = pipe.prompt_template_t2i.format(prompt or " ")
    ids = pipe.processor(text=[text], return_tensors="pt").input_ids[0]
    return ids, pipe._drop_idx


def encode(te: TextEncoder, pipe, prompt: str) -> torch.Tensor:
    """[1, L - drop, 4096]: what `pipe._get_qwen_prompt_embeds` returns for one prompt, on LocalRouter's kernels."""
    ids, drop = prompt_ids(pipe, prompt)
    REC.note("te_ids", ids=[int(t) for t in ids], drop=drop)
    return te.forward(ids)[drop:].unsqueeze(0).contiguous()
