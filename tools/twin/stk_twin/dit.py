"""Qwen-Image 2.1 DiT, op by op: tfimage's arithmetic (its Triton kernels, its bf16 / FP8 / NVFP4 linears on
TensorFold 0.6.1) with every GPU op inside `REC.op`, so a capture lists exactly the program the Zig engine runs.

Sequence: [text (block-causal prefix, t = 0 modulation) | target latent rows]. The prefix's keys and values are built
once per (prompt, target size) and kept; every step runs the target rows only. Weights use diffusers' names, which
are tfimage's too.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from tfimage import kernels as K
from tfimage import linear as L

from . import ops
from .rec import REC

ATTN = os.environ.get("STK_ATTN", "cudnn")  # "stk": LocalRouter's attention kernel
OPS = os.environ.get("STK_OPS", "torch")    # "stk": LocalRouter's small kernels (kernels/cuda/qwen_image/ops.cu)
AMAX: dict[str, float] | None = None        # when a dict: each linear's largest input |x| (FP8 calibration)


@dataclass(frozen=True)
class Config:
    layers: int = 32
    dim: int = 4096
    heads: int = 32
    head_dim: int = 128
    mlp: int = 12288
    in_ch: int = 64
    axes: tuple[int, int, int] = (16, 56, 56)
    theta: float = 10000.0
    eps: float = 1e-6


@dataclass
class Block:
    qkv: L.Linear
    out: L.Linear
    gate_up: L.Linear
    down: L.Linear
    norm_q: torch.Tensor
    norm_k: torch.Tensor


@dataclass(eq=False)
class Prefix:
    context: torch.Tensor
    hw: tuple[int, int]
    next_pos: int = 0
    kv: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


def rope_table(ids: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """ids [N, 3] -> cos, sin [N, D / 2] fp32: angle = id * omega in f64 (frozen frequencies), LocalRouter's portable
    sin / cos (`smath`), rounded to f32. The Zig engine computes the same bits on the host (`rope.zig`)."""
    from .smath import omegas, sincos

    om, axes = omegas(), cfg.axes
    cols = [ax for ax, d in enumerate(axes) for _ in range(d // 2)]
    cos, sin = [], []
    for row in ids.tolist():
        for j, ax in enumerate(cols):
            s_, c_ = sincos(float(row[ax]) * om[j])
            cos.append(c_)
            sin.append(s_)
    n = ids.shape[0]
    return (torch.tensor(cos, dtype=torch.float64).view(n, -1).float(), torch.tensor(sin, dtype=torch.float64).view(n, -1).float())


def image_ids(h: int, w: int, pos: int, target_hw: tuple[int, int]) -> torch.Tensor:
    hh = torch.arange(h, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - target_hw[0] % 2)
    ww = torch.arange(w, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - target_hw[1] % 2)
    return torch.stack([torch.full((h, w), float(pos)), hh[:, None].expand(h, w), ww[None, :].expand(h, w)],
                       dim=-1).reshape(-1, 3)


def policy(spec: str, layers: int):
    """`nvfp4`, `fp8`, `bf16`, or `nvfp4:edge=2,down=fp8`: (block, projection) -> kind."""
    base, _, rest = spec.partition(":")
    over = dict(kv.split("=") for kv in rest.split(",") if kv)
    edge = int(over.pop("edge", 0))
    edge_kind = over.pop("edge_kind", "fp8")
    return lambda i, name: edge_kind if (i < edge or i >= layers - edge) else over.get(name, base)


class QwenImageDiT:
    """Build with `from_bf16` (quantizing as tfimage does) or `from_pack` (the converted weights Zig loads too)."""

    def __init__(self, side: dict[str, torch.Tensor], linear, norms, cfg: Config = Config(), device="cuda"):
        self.cfg, self.dev = cfg, torch.device(device)
        dev = lambda n: side[n].to(self.dev, torch.bfloat16)  # noqa: E731
        self.img_in = dev("img_in.weight")
        self.txt_norm = side["txt_in.text_norm.weight"].to(self.dev, torch.float32) + 1.0  # zero-centred RMSNorm
        self.txt_in1, self.txt_in2 = dev("txt_in.in_layer.weight"), dev("txt_in.out_layer.weight")
        self.t1 = dev("time_text_embed.timestep_embedder.linear_1.weight")
        self.t2 = dev("time_text_embed.timestep_embedder.linear_2.weight")
        self.mod = dev("modulation.1.weight")
        self.norm_out = dev("norm_out.linear.weight")
        self.proj_out = dev("proj_out.weight")
        self.blocks = [Block(linear(i, "qkv"), linear(i, "out"), linear(i, "gate_up"), linear(i, "down"),
                             *(n.to(self.dev, torch.float32) for n in norms(i))) for i in range(cfg.layers)]
        self.prefixes: list[Prefix] = []

    @classmethod
    def from_bf16(cls, w: dict[str, torch.Tensor], precision: str = "nvfp4", acts: dict | None = None,
                  cfg: Config = Config(), device="cuda") -> "QwenImageDiT":
        from .pack import block_weights

        pick, acts = policy(precision, cfg.layers), acts or {}
        cache: dict[int, dict] = {}

        def linear(i, name):
            bw = cache.setdefault(i, block_weights(w, i))
            lin = L.make(pick(i, name), bw.pop(name).to(device, torch.bfloat16), acts.get(f"{i}.{name}"))
            if not bw:
                cache.pop(i)
            return lin

        p = "transformer_blocks.{}.attn."
        dit = cls(w, linear, lambda i: (w[p.format(i) + "norm_q.weight"], w[p.format(i) + "norm_k.weight"]), cfg, device)
        dit.precision = precision
        return dit

    @classmethod
    def from_pack(cls, pack_dir: str | Path, cfg: Config = Config(), device="cuda") -> "QwenImageDiT":
        from safetensors.torch import load_file
        from tensorfold.cuda.nvfp4.linear import Fp4Linear

        t = load_file(str(Path(pack_dir) / "weights.safetensors"))
        man = json.loads((Path(pack_dir) / "manifest.json").read_text())

        def linear(i, name):
            key, kind = f"L{i}.{name}", man["kinds"][f"L{i}.{name}"]
            if kind == "nvfp4":
                lin = object.__new__(L.Nvfp4Linear)
                lin.lin = Fp4Linear.from_checkpoint(t[key + ".codes"].to(device), t[key + ".scales"].to(device),
                                                    float(t[key + ".global"]), act=float(t[key + ".act"]))
                lin.n, lin.k, lin.dynamic, lin.tile, lin.amax = lin.lin.n, lin.lin.k, False, 12, 0.0
                return lin
            if kind == "fp8" and key + ".act" in t:
                from .linear_tf import TfFp8Linear
                return TfFp8Linear(t[key + ".w8"].to(device), float(t[key + ".scale"]), float(t[key + ".act"]))
            if kind == "fp8":
                lin = object.__new__(L.Fp8Linear)
                lin.w8, lin.ws = t[key + ".w8"].to(device), t[key + ".scale"].reshape(()).to(device)
                lin.n, lin.k = lin.w8.shape
                lin.act, lin._a = None, None
                return lin
            return L.Bf16Linear(t[key + ".weight"].to(device))

        dit = cls(t, linear, lambda i: (t[f"L{i}.norm_q"], t[f"L{i}.norm_k"]), cfg, device)
        dit.precision = man["precision"]
        return dit

    MAX_PREFIXES = 4

    def linears(self):
        for i, b in enumerate(self.blocks):
            for name in ("qkv", "out", "gate_up", "down"):
                yield f"{i}.{name}", getattr(b, name)

    def drop_prefixes(self) -> None:
        self.prefixes = []

    # ------------------------------------------------------------------ primitive ops (one recorded op each)
    def _linear(self, name: str, lin, x: torch.Tensor) -> torch.Tensor:
        attrs = {"weight": name, "backend": lin.kind, "n": lin.n, "k": lin.k}
        if isinstance(lin, L.Nvfp4Linear):
            attrs.update(act=lin.lin.act, tile=lin.tile, dynamic=lin.dynamic)
        if hasattr(lin, "act"):
            attrs.update(act=lin.act)
        if hasattr(lin, "tile"):
            attrs.update(tile=lin.tile)
        if AMAX is not None:
            key = name.split(".", 1)[0][1:] + "." + name.split(".", 1)[1]  # "L3.qkv" -> "3.qkv"
            AMAX[key] = max(AMAX.get(key, 0.0), float(x.abs().amax()))
        with REC.op(name, "linear", attrs, x=x) as o:
            y = lin(x)
            o.out(y=y)
        return y

    def _dense(self, name: str, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """A small bf16 linear (F.linear, cuBLAS) on the side path: time, modulation, text, in/out projections."""
        with REC.op(name, "dense_bf16", {"weight": name, "impl": OPS}, x=x) as o:
            y = ops.dense_bf16(x, w) if OPS == "stk" else F.linear(x, w)
            o.out(y=y)
        return y

    def _pointwise(self, name: str, fn: str, x: torch.Tensor) -> torch.Tensor:
        with REC.op(name, fn, {"impl": OPS}, x=x) as o:
            if OPS == "stk":
                y = ops.POINTWISE[fn](x)
            else:
                y = {"silu": F.silu, "tanh": torch.tanh, "gelu_tanh": lambda t: F.gelu(t, approximate="tanh")}[fn](x)
            o.out(y=y)
        return y

    def _adaln(self, name: str, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = x.contiguous()
        s = scale.reshape(B, D).float().contiguous()
        out = torch.empty_like(x)
        with REC.op(name, "adaln", {"eps": self.cfg.eps}, x=x, s=s) as o:
            grid = (B * N,)
            args = (x, s, out, N, D, self.cfg.eps)
            k = K._adaln_kernel[grid](*args, BLOCK=triton.next_power_of_2(D), num_warps=8)
            o.launch(k, grid, args)
            o.out(y=out)
        return out

    def _rms_rope(self, name: str, qkv: torch.Tensor, part: int, w, cos, sin, dst: torch.Tensor) -> None:
        B, N, _, H, D = qkv.shape
        with REC.op(name, "rms_rope", {"part": part, "eps": self.cfg.eps}, qkv=qkv, w=w, cos=cos, sin=sin) as o:
            grid = (B * N * H,)
            args = (qkv, w, cos, sin, dst, qkv.stride(0), qkv.stride(1), part * H * D, dst.stride(0), dst.stride(1),
                    N, H, D, self.cfg.eps)
            k = K._rms_rope_kernel[grid](*args, num_warps=1)
            o.launch(k, grid, args)
            o.out(y=dst)

    def _swiglu(self, name: str, gu: torch.Tensor) -> torch.Tensor:
        M, F2 = gu.shape
        out = torch.empty((M, F2 // 2), dtype=torch.bfloat16, device=gu.device)
        with REC.op(name, "swiglu", {}, gu=gu) as o:
            grid = (M, triton.cdiv(F2 // 2, 2048))
            args = (gu.contiguous(), out, F2 // 2)
            k = K._swiglu_kernel[grid](*args, BLOCK=2048, num_warps=8)
            o.launch(k, grid, args)
            o.out(y=out)
        return out

    def _attention(self, name: str, q, k, v, mask=None) -> torch.Tensor:
        """q [B, N, H, D], k / v [B, S, H, D] -> [B, N, H * D]; cuDNN first (M3 swaps in LocalRouter's kernel)."""
        if ATTN == "stk":  # LocalRouter's kernel, the one Zig launches (B = 1)
            from .attn import attention
            with REC.op(name, "attention", {"impl": "stk", "causal": mask is not None}, q=q, k=k, v=v) as o:
                y = attention(q[0], k[0], v[0], causal=mask is not None).reshape(1, q.shape[1], -1)
                o.out(y=y)
            return y
        with REC.op(name, "attention", {"impl": "sdpa", "masked": mask is not None}, q=q, k=k, v=v,
                    **({"mask": mask} if mask is not None else {})) as o:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if mask is None:
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                                  SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
                    a = F.scaled_dot_product_attention(qt, kt, vt)
            else:
                a = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
            y = a.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)
            o.out(y=y)
        return y

    def _residual(self, name: str, x: torch.Tensor, a: torch.Tensor, g: torch.Tensor) -> None:
        with REC.op(name, "gated_residual", {"impl": OPS}, x=x, a=a, g=g) as o:
            ops.gated_residual_(x, a, g) if OPS == "stk" else x.addcmul_(a, g)
            o.out(x=x)

    # ------------------------------------------------------------------ pieces
    def _time(self, t: torch.Tensor) -> torch.Tensor:
        """t [B] fp32 in [0, 1] -> temb [B + 1, dim]; the last row is t = 0 (the prefix's modulation)."""
        with REC.op("time.sinusoid", "time_sinusoid", {"half": 128, "impl": OPS}, t=t) as o:
          if OPS == "stk":
            emb = ops.time_sinusoid(t)
          else:
            tr = ((t * 1000).to(torch.bfloat16) / 1000).to(torch.bfloat16)
            tr = torch.cat([tr, tr.new_zeros(1)]).float() * 1000
            freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device=t.device) / 128)
            args = tr[:, None] * freqs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
          o.out(y=emb)
        h = self._pointwise("time.silu", "silu", self._dense("time.lin1", emb, self.t1))
        return self._dense("time.lin2", h, self.t2)

    def _modulation(self, temb: torch.Tensor, B: int):
        m = self._dense("mod", self._pointwise("mod.silu", "silu", temb), self.mod)
        s1, g1, s2, g2 = (c.unsqueeze(1) for c in m.chunk(4, dim=-1))
        g1, g2 = self._pointwise("mod.tanh_g1", "tanh", g1.contiguous()), self._pointwise("mod.tanh_g2", "tanh", g2.contiguous())
        step = tuple(t[:B] for t in (s1, g1, s2, g2))
        zero = tuple(t[B:B + 1].expand(B, -1, -1) for t in (s1, g1, s2, g2))
        return step, zero

    def _text(self, context: torch.Tensor) -> torch.Tensor:
        with REC.op("txt.norm", "rms_norm_f32", {"eps": self.cfg.eps, "impl": OPS}, x=context, w=self.txt_norm) as o:
            if OPS == "stk":
                y = ops.rms_norm_f32(context[0], self.txt_norm, self.cfg.eps)[None]
            else:
                c = context.float()
                y = (c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + self.cfg.eps) * self.txt_norm).to(torch.bfloat16)
            o.out(y=y)
        h = self._pointwise("txt.gelu", "gelu_tanh", self._dense("txt.in1", y, self.txt_in1))
        return self._dense("txt.in2", h, self.txt_in2)

    def _block(self, i: int, b: Block, x, mod, cos, sin, kv=None, mask=None, keep=None):
        s1, g1, s2, g2 = mod
        B, N, _ = x.shape
        H, D = self.cfg.heads, self.cfg.head_dim
        tag = f"L{i}" if keep is None else f"P{i}"
        h = self._adaln(f"{tag}.adaln1", x, s1)
        qkv = self._linear(f"{tag}.qkv", b.qkv, h.reshape(B * N, -1)).view(B, N, 3, H, D)
        q = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=x.device)
        self._rms_rope(f"{tag}.rope_q", qkv, 0, b.norm_q, cos, sin, q)
        if kv is None:  # the prefix pass: fresh k, v, kept afterwards
            kk = torch.empty_like(q)
            self._rms_rope(f"{tag}.rope_k", qkv, 1, b.norm_k, cos, sin, kk)
            with REC.op(f"{tag}.v", "copy", {}, src=qkv[:, :, 2]) as o:
                if OPS == "stk":
                    vv = torch.empty_like(q)
                    ops.copy_rows(vv, qkv[:, :, 2])
                else:
                    vv = qkv[:, :, 2].contiguous()
                o.out(y=vv)
            keep.append((kk, vv))
        else:  # a step: this step's k, v written behind the kept prefix rows
            kk, vv = kv
            P = kk.shape[1] - N
            self._rms_rope(f"{tag}.rope_k", qkv, 1, b.norm_k, cos, sin, kk[:, P:])
            with REC.op(f"{tag}.v", "copy", {}, src=qkv[:, :, 2]) as o:
                ops.copy_rows(vv[:, P:], qkv[:, :, 2]) if OPS == "stk" else vv[:, P:].copy_(qkv[:, :, 2])
                o.out(y=vv)
        a = self._attention(f"{tag}.attn", q, kk, vv, mask)
        self._residual(f"{tag}.res1", x, self._linear(f"{tag}.out", b.out, a.reshape(B * N, -1)).view(B, N, -1), g1)
        h2 = self._adaln(f"{tag}.adaln2", x, s2).reshape(B * N, -1)
        y = None
        if all(isinstance(l, L.Nvfp4Linear) and not l.dynamic for l in (b.gate_up, b.down)):
            with REC.op(f"{tag}.mlp", "nvfp4_mlp", {"weights": [f"{i}.gate_up", f"{i}.down"]}, x=h2) as o:
                y = L.swiglu_mlp(b.gate_up, b.down, h2)
                if y is not None:
                    o.out(y=y)
        if y is None:
            y = self._linear(f"{tag}.down", b.down, self._swiglu(f"{tag}.swiglu", self._linear(f"{tag}.gate_up", b.gate_up, h2)))
        self._residual(f"{tag}.res2", x, y.view(B, N, -1), g2)

    def _build_prefix(self, context, hw, zero_mod) -> Prefix:
        """Text tokens through every block once (t = 0 modulation, causal among themselves); keys and values kept."""
        x = self._text(context)
        B, P = x.shape[:2]
        allow = torch.ones((P, P), dtype=torch.bool, device=self.dev).tril()
        cos, sin = (t.to(self.dev) for t in rope_table(torch.arange(P, dtype=torch.float32)[:, None].expand(P, 3), self.cfg))
        pre = Prefix(context.clone(), hw, next_pos=P)
        for i, b in enumerate(self.blocks):
            self._block(i, b, x, zero_mod, cos, sin, mask=allow, keep=pre.kv)
        return pre

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x [B, 64, H, W], t [B] in [0, 1], context [B, L, 4096] -> velocity [B, 64, H, W] bf16."""
        B, C, H, W = x.shape
        context = context.to(self.dev, torch.bfloat16)
        temb = self._time(t.to(self.dev, torch.float32).reshape(-1).expand(B).contiguous())
        step_mod, zero_mod = self._modulation(temb, B)
        pre = next((p for p in self.prefixes if p.hw == (H, W) and p.context.shape == context.shape
                    and torch.equal(p.context, context)), None)
        if pre is None:
            if len(self.prefixes) >= self.MAX_PREFIXES:
                self.prefixes.pop(0)
            pre = self._build_prefix(context, (H, W), zero_mod)
            self.prefixes.append(pre)
        h = self._dense("img_in", x.to(torch.bfloat16).flatten(2).transpose(1, 2).contiguous(), self.img_in).contiguous()
        cos, sin = (v.to(self.dev) for v in rope_table(image_ids(H, W, pre.next_pos, (H, W)), self.cfg))
        P = pre.kv[0][0].shape[1]
        kbuf = torch.empty((B, P + H * W, self.cfg.heads, self.cfg.head_dim), dtype=torch.bfloat16, device=self.dev)
        vbuf = torch.empty_like(kbuf)
        for i, (b, (pk, pv)) in enumerate(zip(self.blocks, pre.kv)):
            if OPS == "stk":
                ops.copy_rows(kbuf[:, :P], pk)
                ops.copy_rows(vbuf[:, :P], pv)
            else:
                kbuf[:, :P].copy_(pk)
                vbuf[:, :P].copy_(pv)
            self._block(i, b, h, step_mod, cos, sin, kv=(kbuf, vbuf))
        scale = self._dense("norm_out", self._pointwise("out.silu", "silu", temb[:B]), self.norm_out).unsqueeze(1)
        h = self._adaln("out.adaln", h, scale)
        out = self._dense("proj_out", h, self.proj_out)
        return out.transpose(1, 2).reshape(B, C, H, W)
