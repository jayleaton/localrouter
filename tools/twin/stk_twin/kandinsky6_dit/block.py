"""The Kandinsky-6 DiT forward of src/engines/kandinsky6/dit/block.zig, op for op on LocalRouter's kernels (kernels.py):
`text_block` (TransformerEncoderBlock), `forward_block` (FusedTransformerDecoderBlock with ca_rope, cross_gates and
fix_modulation) and `forward` (PiFlowDiffusionTransformer3D on the fused path, the grid split included), with
layout.zig's config, shapes, modulation offsets and tensor manifest, and block.zig's scratch plan: one region, the same
offsets, the same buffer reuse (block.zig's views, including View.of re-reads of a buffer at another width). Same
weights, inputs and kernels as the engine: the same bits.

Weights: a dict checkpoint name -> bf16 CUDA tensor (layout.zig forEachTensor's names and shapes; the file's F32
tensors cast to bf16 round-to-nearest-even as upstream runtime/weights.py does). Captures: an optional
`hook(name, tensor)` after every named step, with block.zig's names; the tensor is a view of the step's output in the
scratch or the caller's buffers (clone it to keep it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import torch

from . import kernels as K

Hook = Callable[[str, torch.Tensor], None]


# ------------------------------------------------------------------------------------------------------- layout.zig
@dataclass(frozen=True)
class Config:
    """layout.Config: the transformer config of the Pro-distill checkpoint (defaults); fewer blocks or smaller widths
    for local tests (head_dim stays 128: the kernels are built for it)."""

    model_dim: int = 4096
    model_dim_a: int = 2048
    time_dim: int = 1024
    ff_dim: int = 16384
    ff_dim_a: int = 7168
    head_dim: int = 128
    axes_dims: tuple = (32, 48, 48)
    in_text_dim: int = 3584
    in_text_dim2: int = 768
    in_visual_dim: int = 16
    in_audio_dim: int = 40
    n_grid: int = 10
    out_visual_dim: int = 160
    out_audio_dim: int = 400
    num_text_blocks: int = 4
    num_visual_blocks: int = 60
    token_types: int = 2
    audio_freqs_scaling: float = 0.144
    scale_factor: tuple = (1.0, 2.0, 2.0)

    def visual_in(self) -> int:
        return 2 * self.in_visual_dim + 1

    def patch_in(self) -> int:
        return 4 * self.visual_in()

    def heads(self) -> int:
        return self.model_dim // self.head_dim

    def heads_a(self) -> int:
        return self.model_dim_a // self.head_dim

    def dx_visual(self) -> int:
        return self.out_visual_dim // self.n_grid

    def dx_audio(self) -> int:
        return self.out_audio_dim // self.n_grid


MAX_TEXT = 1024


@dataclass(frozen=True)
class Shape:
    """layout.Shape: latent frames, latent height / width (pixels / 8), audio rows, text tokens (padded)."""

    t: int
    h: int
    w: int
    audio: int
    text: int

    @staticmethod
    def preset(text: int) -> "Shape":
        """T2V and the default ("pretrain") I2V alike: 31 latent frames, 864x480."""
        return Shape(t=31, h=480 // 8, w=864 // 8, audio=218, text=text)

    def tokens(self) -> int:
        return self.t * (self.h // 2) * (self.w // 2)

    def per_frame(self) -> int:
        return (self.h // 2) * (self.w // 2)

    def validate(self) -> None:
        bad = (self.t == 0 or self.h == 0 or self.w == 0 or self.h % 2 != 0 or self.w % 2 != 0 or self.audio == 0
               or self.text == 0 or self.text > MAX_TEXT or self.t > 128 or self.h // 2 > 128 or self.w // 2 > 128
               or self.audio > 2048)
        if bad:
            raise ValueError(f"UnsupportedShape {self}")


class Mod:
    """layout.Mod: modulation chunk offsets, (part * 3 + which) * d with which 0 shift / 1 scale / 2 gate."""

    @staticmethod
    def dec(d: int, part: int, which: int) -> int:
        return (part * 3 + which) * d

    @staticmethod
    def enc(d: int, part: int, which: int) -> int:
        return (part * 3 + which) * d

    @staticmethod
    def out(d: int, which: int) -> int:
        return which * d

    @staticmethod
    def cross(d_self: int, d_other: int, which: int) -> tuple[int, int]:
        return {0: (0, d_self), 1: (d_self, d_self)}.get(which, (2 * d_self, d_other))


def manifest(c: Config) -> list[tuple[str, str, tuple]]:
    """layout.forEachTensor: (name, file dtype, shape) of every tensor, in its order."""
    D, Da, T, hd = c.model_dim, c.model_dim_a, c.time_dim, c.head_dim
    out: list[tuple[str, str, tuple]] = []

    def one(name, dt, shape):
        out.append((name, dt, tuple(shape)))

    def linear(p, dt, o, i, bias=True):
        one(f"{p}.weight", dt, (o, i))
        if bias:
            one(f"{p}.bias", dt, (o,))

    def attn(p, q, kv):
        linear(f"{p}.to_query", "bf16", q, q)
        linear(f"{p}.to_key", "bf16", q, kv)
        linear(f"{p}.to_value", "bf16", q, kv)
        linear(f"{p}.out_layer", "bf16", q, q)
        one(f"{p}.query_norm.weight", "f32", (hd,))
        one(f"{p}.key_norm.weight", "f32", (hd,))

    def ffn(p, d, ff):
        one(f"{p}.feed_forward.net.0.proj.weight", "bf16", (ff, d))
        one(f"{p}.feed_forward.net.2.weight", "bf16", (d, ff))

    def text_emb(p, o, i):
        linear(f"{p}.in_layer", "f32", o, i)
        one(f"{p}.norm.weight", "f32", (o,))
        one(f"{p}.norm.bias", "f32", (o,))

    linear("visual_embeddings.in_layer", "f32", D, c.patch_in())
    one("visual_token_type_embeddings.weight", "f32", (c.token_types, D))
    linear("out_layer.modulation.out_layer", "bf16", 2 * D, T)
    linear("out_layer.out_layer", "bf16", 4 * c.out_visual_dim, D)
    text_emb("audio_embeddings", Da, c.in_audio_dim)
    linear("audio_out_layer.modulation.out_layer", "bf16", 2 * Da, T)
    linear("audio_out_layer.out_layer", "bf16", c.out_audio_dim, Da)
    for p, md, ff in (("video", D, c.ff_dim), ("audio", Da, c.ff_dim_a)):
        linear(f"{p}_time_embeddings.timestep_embedder.linear_1", "f32", T, md)
        linear(f"{p}_time_embeddings.timestep_embedder.linear_2", "f32", T, T)
        text_emb(f"{p}_text_embeddings", md, c.in_text_dim)
        text_emb(f"{p}_pooled_text_embeddings", T, c.in_text_dim2)
        for i in range(c.num_text_blocks):
            b = f"{p}_text_transformer_blocks.{i}"
            attn(f"{b}.attn", md, md)
            ffn(b, md, ff)
            linear(f"{b}.text_modulation.out_layer", "bf16", 6 * md, T)
    for i in range(c.num_visual_blocks):
        for m, md, ff in (("video_dec_block", D, c.ff_dim), ("audio_dec_block", Da, c.ff_dim_a)):
            b = f"visual_transformer_blocks.{i}.{m}"
            attn(f"{b}.self_attention", md, md)
            attn(f"{b}.cross_attention", md, md)
            ffn(b, md, ff)
            linear(f"{b}.visual_modulation.out_layer", "bf16", 9 * md, T)
        attn(f"visual_transformer_blocks.{i}.va_cross_attention", D, Da)
        attn(f"visual_transformer_blocks.{i}.av_cross_attention", Da, D)
        linear(f"visual_transformer_blocks.{i}.va_modulation.out_layer", "bf16", 2 * D + Da, T)
        linear(f"visual_transformer_blocks.{i}.av_modulation.out_layer", "bf16", 2 * Da + D, T)
    return out


# --------------------------------------------------------------------------------------------- block.zig: weights
class Linear(NamedTuple):
    w: torch.Tensor  # bf16 [out, in]
    b: Optional[torch.Tensor]  # bf16 [1, out]


class Attn(NamedTuple):
    q: Linear
    k: Linear
    v: Linear
    o: Linear
    qn: torch.Tensor  # bf16 [1, 128]
    kn: torch.Tensor
    heads: int


class Ffn(NamedTuple):
    up: torch.Tensor  # net.0.proj [ff, d]
    down: torch.Tensor  # net.2 [d, ff]


class TextBlock(NamedTuple):
    mod: Linear
    attn: Attn
    ffn: Ffn


class DecBlock(NamedTuple):
    mod: Linear
    sa: Attn
    ca: Attn
    ffn: Ffn


class FusedBlock(NamedTuple):
    video: DecBlock
    audio: DecBlock
    va: Attn  # video queries, audio keys
    av: Attn  # audio queries, video keys
    va_mod: Linear  # [2 D + Da, T]
    av_mod: Linear  # [2 Da + D, T]


class TextEmbedding(NamedTuple):
    inp: Linear
    norm_w: torch.Tensor
    norm_b: torch.Tensor


class Modality(NamedTuple):
    time1: Linear
    time2: Linear
    text: TextEmbedding
    pooled: TextEmbedding
    blocks: list


class Weights(NamedTuple):
    visual_in: Linear
    type_table: torch.Tensor  # bf16 [2, D]
    out_mod: Linear
    out: Linear  # [4 * 160, D]
    audio_in: TextEmbedding
    audio_out_mod: Linear
    audio_out: Linear  # [400, Da]
    video: Modality
    audio: Modality
    blocks: list


class MissingWeight(KeyError):
    pass


class _Bind:
    """bindWeights' lookups: a bf16 view [rows, cols] of the named tensor (the manifest's shape)."""

    def __init__(self, weights: dict):
        self.w = weights

    def t(self, name: str, rows: int, cols: int) -> torch.Tensor:
        x = self.w.get(name)
        if x is None:
            raise MissingWeight(name)
        if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.numel() != rows * cols:
            raise K.ViewMismatch(f"{name}: {x.dtype} {tuple(x.shape)} is not bf16 [{rows}, {cols}]")
        return x.view(rows, cols)

    def lin(self, p: str, out: int, inp: int, bias: bool = True) -> Linear:
        return Linear(self.t(f"{p}.weight", out, inp), self.t(f"{p}.bias", 1, out) if bias else None)

    def attn(self, p: str, q: int, kv: int) -> Attn:
        return Attn(self.lin(f"{p}.to_query", q, q), self.lin(f"{p}.to_key", q, kv), self.lin(f"{p}.to_value", q, kv),
                    self.lin(f"{p}.out_layer", q, q), self.t(f"{p}.query_norm.weight", 1, 128),
                    self.t(f"{p}.key_norm.weight", 1, 128), q // 128)

    def ffn(self, p: str, d: int, ff: int) -> Ffn:
        return Ffn(self.t(f"{p}.feed_forward.net.0.proj.weight", ff, d),
                   self.t(f"{p}.feed_forward.net.2.weight", d, ff))

    def emb(self, p: str, out: int, inp: int) -> TextEmbedding:
        return TextEmbedding(self.lin(f"{p}.in_layer", out, inp), self.t(f"{p}.norm.weight", 1, out),
                             self.t(f"{p}.norm.bias", 1, out))

    def dec(self, p: str, i: int, d: int, ff: int, T: int) -> DecBlock:
        f = f"visual_transformer_blocks.{i}.{p}"
        return DecBlock(self.lin(f"{f}.visual_modulation.out_layer", 9 * d, T), self.attn(f"{f}.self_attention", d, d),
                        self.attn(f"{f}.cross_attention", d, d), self.ffn(f, d, ff))


def bind_text_block(c: Config, weights: dict, prefix: str, i: int) -> TextBlock:
    md, ff = (c.model_dim, c.ff_dim) if prefix == "video" else (c.model_dim_a, c.ff_dim_a)
    b = _Bind(weights)
    p = f"{prefix}_text_transformer_blocks.{i}"
    return TextBlock(b.lin(f"{p}.text_modulation.out_layer", 6 * md, c.time_dim), b.attn(f"{p}.attn", md, md),
                     b.ffn(p, md, ff))


def bind_block(c: Config, weights: dict, i: int) -> FusedBlock:
    D, Da, T = c.model_dim, c.model_dim_a, c.time_dim
    b = _Bind(weights)
    return FusedBlock(b.dec("video_dec_block", i, D, c.ff_dim, T), b.dec("audio_dec_block", i, Da, c.ff_dim_a, T),
                      b.attn(f"visual_transformer_blocks.{i}.va_cross_attention", D, Da),
                      b.attn(f"visual_transformer_blocks.{i}.av_cross_attention", Da, D),
                      b.lin(f"visual_transformer_blocks.{i}.va_modulation.out_layer", 2 * D + Da, T),
                      b.lin(f"visual_transformer_blocks.{i}.av_modulation.out_layer", 2 * Da + D, T))


def bind_weights(c: Config, weights: dict) -> Weights:
    """block.zig bindWeights."""
    D, Da, T = c.model_dim, c.model_dim_a, c.time_dim
    b = _Bind(weights)
    mods = {}
    for p, md in (("video", D), ("audio", Da)):
        mods[p] = Modality(b.lin(f"{p}_time_embeddings.timestep_embedder.linear_1", T, md),
                           b.lin(f"{p}_time_embeddings.timestep_embedder.linear_2", T, T),
                           b.emb(f"{p}_text_embeddings", md, c.in_text_dim),
                           b.emb(f"{p}_pooled_text_embeddings", T, c.in_text_dim2),
                           [bind_text_block(c, weights, p, i) for i in range(c.num_text_blocks)])
    return Weights(b.lin("visual_embeddings.in_layer", D, c.patch_in()),
                   b.t("visual_token_type_embeddings.weight", c.token_types, D),
                   b.lin("out_layer.modulation.out_layer", 2 * D, T),
                   b.lin("out_layer.out_layer", 4 * c.out_visual_dim, D),
                   b.emb("audio_embeddings", Da, c.in_audio_dim),
                   b.lin("audio_out_layer.modulation.out_layer", 2 * Da, T),
                   b.lin("audio_out_layer.out_layer", c.out_audio_dim, Da), mods["video"], mods["audio"],
                   [bind_block(c, weights, i) for i in range(c.num_visual_blocks)])


# ------------------------------------------------------------------------------------------------ block.zig: Plan
DEFAULT_FFN_ROWS = 4096


@dataclass
class Plan:
    """block.zig Plan: every scratch buffer as a 256-byte aligned offset into one region, in Plan.init's order."""

    c: Config
    sh: Shape
    ffn_rows: int = DEFAULT_FFN_ROWS
    rows: int = 0
    v: list = field(default_factory=list)  # [rows, D] x 8: n, q, k, v, attention out, projection out, pre-CA, text CA
    a: list = field(default_factory=list)  # [max(A, L), Da] x 7: n, q, k, v, a, o, pre-CA
    hidden: int = 0
    vis: int = 0
    aud: int = 0
    text_v: int = 0
    text_a: int = 0
    rows_in: int = 0
    audio_in: int = 0
    mod: int = 0
    mod2: int = 0
    mod3: int = 0
    mod4: int = 0
    tm_v: int = 0
    tm_a: int = 0
    t_sin: int = 0
    t_h: int = 0
    t_emb: int = 0
    pooled: int = 0
    head: int = 0
    total: int = 0

    def __post_init__(self):
        c, sh = self.c, self.sh
        sh.validate()
        if self.ffn_rows == 0:
            raise ValueError("UnsupportedShape: ffn_rows 0")
        N, A, L = sh.tokens(), sh.audio, sh.text
        D, Da, T = c.model_dim, c.model_dim_a, c.time_dim
        self.rows = max(N, L, A)  # video k / v also hold the V <- A keys [A, D] and the text keys [L, D]
        at = 0

        def put(nbytes: int) -> int:
            nonlocal at
            o = at
            at = (at + nbytes + 255) // 256 * 256
            return o

        self.v = [put(self.rows * D * 2) for _ in range(8)]
        arows = max(A, L)
        self.a = [put(arows * Da * 2) for _ in range(7)]
        self.hidden = put(self.ffn_rows * max(c.ff_dim, c.ff_dim_a) * 2)
        self.vis = put(N * D * 2)
        self.aud = put(A * Da * 2)
        self.text_v = put(L * D * 2)
        self.text_a = put(L * Da * 2)
        self.rows_in = put(N * c.patch_in() * 2)
        self.audio_in = put(A * c.in_audio_dim * 2)
        self.mod = put(9 * D * 2)
        self.mod2 = put(9 * Da * 2)
        self.mod3 = put((2 * D + Da) * 2)
        self.mod4 = put((2 * Da + D) * 2)
        self.tm_v = put(T * 2)
        self.tm_a = put(T * 2)
        self.t_sin = put(D * 4)
        self.t_h = put(T * 4)
        self.t_emb = put(T * 2)
        self.pooled = put(T * 2)
        self.head = put(max(N, A) * max(4 * c.out_visual_dim, c.out_audio_dim) * 2)
        self.total = at

    def bytes(self) -> int:
        return self.total


# ------------------------------------------------------------------------------------------- block.zig: the forward
@dataclass
class Inputs:
    """block.zig Inputs (B = 1), row views."""

    x_video: torch.Tensor  # [T * H * W, 33] fp32 (bf16 at step 0): cat(latent 16, cond 16, mask 1), channels last
    x_audio: torch.Tensor  # [A, 40] fp32 / bf16
    text: torch.Tensor  # [L, 3584] bf16
    pooled: torch.Tensor  # [1, 768] bf16
    time_v: torch.Tensor  # [1, 1] fp32: sigma * 1000
    time_a: torch.Tensor  # [1, 1] fp32
    freqs_v: torch.Tensor  # [1, D / 2] fp32: video_time_embeddings.freqs
    freqs_a: torch.Tensor  # [1, Da / 2] fp32
    visual_rope: torch.Tensor  # [N, 256] fp32: RoPE3D's table flattened (t, h', w')
    audio_rope: torch.Tensor  # [A, 256] fp32
    text_rope_v: torch.Tensor  # [L, 256] fp32
    text_rope_a: torch.Tensor  # [L, 256] fp32
    mask: Optional[K.Mask] = None  # the text key-padding mask (True = keep)


@dataclass
class Outputs:
    video: torch.Tensor  # [10 * T * H * W, 16] bf16 = [10, T, H, W, 16]
    audio: torch.Tensor  # [10 * A, 40] bf16 = [10, A, 40]


@dataclass
class BlockIo:
    vis: torch.Tensor  # [N, D] residual, in place
    aud: torch.Tensor  # [A, Da] residual, in place
    text_v: torch.Tensor  # [L, D]
    text_a: torch.Tensor  # [L, Da]
    tm_v: torch.Tensor  # [1, T] bf16
    tm_a: torch.Tensor
    vis_rope: torch.Tensor  # [N, 256] f32
    aud_rope: torch.Tensor  # [A, 256] f32
    mask: Optional[K.Mask] = None


class Bufs(NamedTuple):
    n: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    o: torch.Tensor


def cols(t: torch.Tensor, c0: int, n: int) -> torch.Tensor:
    """View.columns."""
    return t[:, c0:c0 + n]


class Dit:
    """block.zig Dit on one scratch region of `plan.bytes()` bytes (allocated here when not given; `fill` sets every
    byte first, to show that no result reads stale scratch)."""

    def __init__(self, k: K.Kernels, plan: Plan, fp32_linear: str, region: torch.Tensor | None = None,
                 hook: Hook | None = None, fill: int | None = None):
        if fp32_linear not in K.FP32_LINEAR:
            raise K.InvalidGemm(f"fp32_linear {fp32_linear!r}")
        self.k, self.p, self.hook, self.fp32_linear = k, plan, hook, fp32_linear
        if region is None:
            region = torch.empty(plan.bytes(), dtype=torch.uint8, device="cuda")
        if fill is not None:
            region.fill_(fill)
        if region.dtype != torch.uint8 or region.numel() < plan.bytes() or region.data_ptr() % 256 != 0:
            raise K.ViewMismatch("scratch region: 256-byte aligned u8 of plan.bytes()")
        self.region = region
        self.base = region.data_ptr()

    def buf(self, off: int, rows: int, ncols: int, dt: str = "bf16") -> torch.Tensor:
        n = rows * ncols * K.SIZE[dt]
        return self.region[off:off + n].view(K.TORCH[dt]).view(rows, ncols)

    def of(self, t: torch.Tensor, rows: int, ncols: int) -> torch.Tensor:
        """View.of(t.ptr, bf16, rows, cols): a scratch buffer re-read densely at another width."""
        return self.buf(t.data_ptr() - self.base, rows, ncols)

    def emit(self, name: str, out: torch.Tensor) -> None:
        if self.hook is not None:
            self.hook(name, out)

    def linear(self, l: Linear, x: torch.Tensor, out: torch.Tensor) -> None:
        self.k.gemm(x, l.w, l.b, out, K.EPI_BIAS)

    def modulation(self, l: Linear, tm: torch.Tensor, out: torch.Tensor) -> None:
        """bf16(silu(float(tm)) . float(W)^T + float(b)) into out [1, n]."""
        self.k.gemv(tm, l.w, l.b, out, True, self.fp32_linear)

    def ffn(self, f: Ffn, x: torch.Tensor, out: torch.Tensor) -> None:
        """FeedForward in row chunks of ffn_rows through the hidden buffer."""
        ff = f.up.shape[0]
        r = 0
        while r < x.shape[0]:
            n = min(self.p.ffn_rows, x.shape[0] - r)
            h = self.buf(self.p.hidden, n, ff)
            self.k.gemm(x[r:r + n], f.up, None, h, K.EPI_GELU)
            self.k.gemm(h, f.down, None, out[r:r + n], K.EPI_BIAS)
            r += self.p.ffn_rows

    def attn(self, m: Attn, xq, xkv, rope_q, rope_k, mask, b: Bufs) -> None:
        """q from xq, k / v from xkv, per-head RMSNorm, the optional ropes, attention, out_layer into b.o."""
        dq = m.q.w.shape[0]
        q = cols(b.q, 0, dq)[:xq.shape[0]]
        k = self.of(b.k, xkv.shape[0], dq)
        v = self.of(b.v, xkv.shape[0], dq)
        a = cols(b.a, 0, dq)[:xq.shape[0]]
        self.linear(m.q, xq, q)
        self.linear(m.k, xkv, k)
        self.linear(m.v, xkv, v)
        self.k.qk_norm_rope(q, m.qn, rope_q, m.heads)
        self.k.qk_norm_rope(k, m.kn, rope_k, m.heads)
        self.k.attend(q, k, v, a, m.heads, mask)
        self.linear(m.o, a, cols(b.o, 0, dq)[:xq.shape[0]])

    def video_bufs(self, rows: int, d: int) -> Bufs:
        v = self.p.v
        return Bufs(self.buf(v[0], rows, d), self.buf(v[1], rows, d), self.buf(v[2], self.p.rows, d),
                    self.buf(v[3], self.p.rows, d), self.buf(v[4], rows, d), self.buf(v[5], rows, d))

    def audio_bufs(self, rows: int, d: int) -> Bufs:
        ar = max(self.p.sh.audio, self.p.sh.text)
        a = self.p.a
        return Bufs(self.buf(a[0], rows, d), self.buf(a[1], rows, d), self.buf(a[2], ar, d), self.buf(a[3], ar, d),
                    self.buf(a[4], rows, d), self.buf(a[5], rows, d))

    def text_block(self, b: TextBlock, i: int, prefix: str, te, tm, rope, mask) -> None:
        """TransformerEncoderBlock on te [L, md] in place."""
        md, L = te.shape[1], te.shape[0]
        p = f"{prefix}_text_transformer_blocks.{i}"
        bufs = self.video_bufs(L, md)  # the text blocks run before the visual blocks: the stream scratch is free
        mod = self.buf(self.p.mod, 1, 6 * md)
        self.modulation(b.mod, tm, mod)
        self.emit(f"{p}.text_modulation", mod)
        self.k.norm_mod(te, cols(mod, Mod.enc(md, 0, 1), md), cols(mod, Mod.enc(md, 0, 0), md), bufs.n)
        self.attn(b.attn, bufs.n, bufs.n, rope, rope, mask, bufs)
        self.emit(f"{p}.attn", bufs.o)
        self.k.gate_sum(te, bufs.o, cols(mod, Mod.enc(md, 0, 2), md))
        self.k.norm_mod(te, cols(mod, Mod.enc(md, 1, 1), md), cols(mod, Mod.enc(md, 1, 0), md), bufs.n)
        self.ffn(b.ffn, bufs.n, bufs.o)
        self.emit(f"{p}.feed_forward", bufs.o)
        self.k.gate_sum(te, bufs.o, cols(mod, Mod.enc(md, 1, 2), md))
        self.emit(p, te)

    def forward_block(self, b: FusedBlock, i: int, io: BlockIo) -> None:
        """FusedTransformerDecoderBlock.forward (dit.py 436-517), ca_rope, cross_gates, fix_modulation."""
        c = self.p.c
        D, Da = c.model_dim, c.model_dim_a
        N, A = io.vis.shape[0], io.aud.shape[0]
        vb = self.video_bufs(N, D)
        ab = self.audio_bufs(A, Da)
        vpre = self.buf(self.p.v[6], N, D)
        vct = self.buf(self.p.v[7], N, D)
        apre = self.buf(self.p.a[6], A, Da)
        vm = self.buf(self.p.mod, 1, 9 * D)
        am = self.buf(self.p.mod2, 1, 9 * Da)
        vam = self.buf(self.p.mod3, 1, 2 * D + Da)
        avm = self.buf(self.p.mod4, 1, 2 * Da + D)
        p = f"visual_transformer_blocks.{i}."
        kk = self.k

        # video backbone: self-attention, then the text cross-attention computed but not yet added
        self.modulation(b.video.mod, io.tm_v, vm)
        self.emit(p + "video_dec_block.visual_modulation", vm)
        kk.norm_mod(io.vis, cols(vm, Mod.dec(D, 0, 1), D), cols(vm, Mod.dec(D, 0, 0), D), vb.n)
        self.attn(b.video.sa, vb.n, vb.n, io.vis_rope, io.vis_rope, None, vb)
        self.emit(p + "video_dec_block.self_attention", vb.o)
        kk.gate_sum(io.vis, vb.o, cols(vm, Mod.dec(D, 0, 2), D))
        self.emit(p + "video_dec_block.self_attention_residual", io.vis)
        kk.norm_mod(io.vis, cols(vm, Mod.dec(D, 1, 1), D), cols(vm, Mod.dec(D, 1, 0), D), vpre)
        self.emit(p + "video_dec_block.pre_ca", vpre)
        self.attn(b.video.ca, vpre, io.text_v, None, None, io.mask, Bufs(vb.n, vb.q, vb.k, vb.v, vb.a, vct))
        self.emit(p + "video_dec_block.cross_attention", vct)

        # audio backbone: self-attention, text cross-attention added at once
        self.modulation(b.audio.mod, io.tm_a, am)
        self.emit(p + "audio_dec_block.visual_modulation", am)
        kk.norm_mod(io.aud, cols(am, Mod.dec(Da, 0, 1), Da), cols(am, Mod.dec(Da, 0, 0), Da), ab.n)
        self.attn(b.audio.sa, ab.n, ab.n, io.aud_rope, io.aud_rope, None, ab)
        self.emit(p + "audio_dec_block.self_attention", ab.o)
        kk.gate_sum(io.aud, ab.o, cols(am, Mod.dec(Da, 0, 2), Da))
        kk.norm_mod(io.aud, cols(am, Mod.dec(Da, 1, 1), Da), cols(am, Mod.dec(Da, 1, 0), Da), apre)
        self.emit(p + "audio_dec_block.pre_ca", apre)
        self.attn(b.audio.ca, apre, io.text_a, None, None, io.mask, ab)
        self.emit(p + "audio_dec_block.cross_attention", ab.o)
        kk.gate_sum(io.aud, ab.o, cols(am, Mod.dec(Da, 1, 2), Da))

        # cross-modal: va_modulation(t_v), av_modulation(t_a) (fix_modulation)
        self.modulation(b.va_mod, io.tm_v, vam)
        self.emit(p + "va_modulation", vam)
        self.modulation(b.av_mod, io.tm_a, avm)
        self.emit(p + "av_modulation", avm)
        va_shift, va_scale, va_gate = (Mod.cross(D, Da, w) for w in range(3))  # va_gate [Da]: the audio residual's
        av_shift, av_scale, av_gate = (Mod.cross(Da, D, w) for w in range(3))  # av_gate [D]: the video residual's
        kk.gate_sum(io.vis, vct, cols(vm, Mod.dec(D, 1, 2), D))  # the deferred video text residual
        self.emit(p + "video_dec_block.cross_attention_residual", io.vis)
        kk.norm_mod(io.vis, cols(vam, *va_scale), cols(vam, *va_shift), vb.n)
        kk.norm_mod(io.aud, cols(avm, *av_scale), cols(avm, *av_shift), ab.n)
        # V <- A: queries vis_for_va (visual rope), keys / values aud_pre_ca (audio rope)
        self.attn(b.va, vb.n, apre, io.vis_rope, io.aud_rope, None, vb)
        self.emit(p + "va_cross_attention", vb.o)
        # A <- V: queries aud_for_av (audio rope), keys / values vis_pre_ca (visual rope), in the video k / v buffers
        self.attn(b.av, ab.n, vpre, io.aud_rope, io.vis_rope, None, Bufs(ab.n, ab.q, vb.k, vb.v, ab.a, ab.o))
        self.emit(p + "av_cross_attention", ab.o)
        kk.gate_sum(io.vis, vb.o, cols(avm, *av_gate))
        kk.gate_sum(io.aud, ab.o, cols(vam, *va_gate))

        # FFN
        kk.norm_mod(io.vis, cols(vm, Mod.dec(D, 2, 1), D), cols(vm, Mod.dec(D, 2, 0), D), vb.n)
        self.ffn(b.video.ffn, vb.n, vb.o)
        self.emit(p + "video_dec_block.feed_forward", vb.o)
        kk.gate_sum(io.vis, vb.o, cols(vm, Mod.dec(D, 2, 2), D))
        kk.norm_mod(io.aud, cols(am, Mod.dec(Da, 2, 1), Da), cols(am, Mod.dec(Da, 2, 0), Da), ab.n)
        self.ffn(b.audio.ffn, ab.n, ab.o)
        self.emit(p + "audio_dec_block.feed_forward", ab.o)
        kk.gate_sum(io.aud, ab.o, cols(am, Mod.dec(Da, 2, 2), Da))
        self.emit(p + "video", io.vis)
        self.emit(p + "audio", io.aud)

    def text_embedding(self, e: TextEmbedding, x, out) -> None:
        """TextEmbeddings: bf16 in_layer (+bias), then the affine LayerNorm in place."""
        self.linear(e.inp, x, out)
        self.k.layer_norm(out, e.norm_w, e.norm_b, out)

    def encode_text(self, m: Modality, prefix: str, inp: Inputs, time, freqs, rope, te, tm) -> None:
        """`_encode_text(prefix)`: token and pooled projections, time embedding + pooled, the text blocks."""
        T = self.p.c.time_dim
        self.text_embedding(m.text, inp.text, te)
        self.emit(f"{prefix}_text_embeddings", te)
        pooled = self.buf(self.p.pooled, 1, T)
        self.text_embedding(m.pooled, inp.pooled, pooled)
        self.emit(f"{prefix}_pooled_text_embeddings", pooled)
        # TimeEmbeddings: [cos, sin] of outer(t, freqs) -> fp32 linear_1 -> SiLU -> fp32 linear_2 -> bf16
        half = K.view(freqs).cols
        sincos = self.buf(self.p.t_sin, 1, 2 * half, "f32")
        h = self.buf(self.p.t_h, 1, T, "f32")
        temb = self.buf(self.p.t_emb, 1, T)
        self.k.time_sincos(time, freqs, sincos)
        self.k.gemv(sincos, m.time1.w, m.time1.b, h, False, self.fp32_linear)
        self.k.gemv(h, m.time2.w, m.time2.b, temb, True, self.fp32_linear)
        self.emit(f"{prefix}_time_embeddings", temb)
        self.k.add_bf16(temb, pooled, tm)
        self.emit(f"{prefix}_time_plus_pooled", tm)
        for i, b in enumerate(m.blocks):
            self.text_block(b, i, prefix, te, tm, rope, inp.mask)

    def forward(self, w: Weights, inp: Inputs, out: Outputs) -> None:
        """The fused forward (`DiffusionTransformer3D.forward` with both modalities), then the PiFlow grid split."""
        c, sh = self.p.c, self.p.sh
        D, Da, T = c.model_dim, c.model_dim_a, c.time_dim
        N, A, L = sh.tokens(), sh.audio, sh.text
        V = K.view
        V(inp.text).expect("bf16", L, c.in_text_dim)
        V(inp.pooled).expect("bf16", 1, c.in_text_dim2)
        V(inp.visual_rope).expect("f32", N, 4 * K.ROPE_PAIRS)
        V(inp.audio_rope).expect("f32", A, 4 * K.ROPE_PAIRS)
        V(inp.text_rope_v).expect("f32", L, 4 * K.ROPE_PAIRS)
        V(inp.text_rope_a).expect("f32", L, 4 * K.ROPE_PAIRS)
        V(inp.freqs_v).expect("f32", 1, D // 2)
        V(inp.freqs_a).expect("f32", 1, Da // 2)
        V(out.video).expect("bf16", c.n_grid * sh.t * sh.h * sh.w, c.dx_visual())
        V(out.audio).expect("bf16", c.n_grid * A, c.dx_audio())
        if inp.mask is not None and inp.mask.valid > L:
            raise K.ViewMismatch("mask valid > text")

        te_v = self.buf(self.p.text_v, L, D)
        te_a = self.buf(self.p.text_a, L, Da)
        tm_v = self.buf(self.p.tm_v, 1, T)
        tm_a = self.buf(self.p.tm_a, 1, T)
        self.encode_text(w.video, "video", inp, inp.time_v, inp.freqs_v, inp.text_rope_v, te_v, tm_v)
        self.encode_text(w.audio, "audio", inp, inp.time_a, inp.freqs_a, inp.text_rope_a, te_a, tm_a)

        # visual and audio embeddings
        vis = self.buf(self.p.vis, N, D)
        aud = self.buf(self.p.aud, A, Da)
        prow = self.buf(self.p.rows_in, N, c.patch_in())
        self.k.patchify(inp.x_video, prow, sh, c.visual_in())
        self.linear(w.visual_in, prow, vis)
        self.emit("visual_embeddings", vis)
        audio_in = inp.x_audio
        if V(inp.x_audio).dtype == "f32":
            audio_in = self.buf(self.p.audio_in, A, c.in_audio_dim)
            self.k.cast_bf16(inp.x_audio, audio_in)
        self.text_embedding(w.audio_in, audio_in, aud)
        self.emit("audio_embeddings", aud)

        io = BlockIo(vis, aud, te_v, te_a, tm_v, tm_a, inp.visual_rope, inp.audio_rope, inp.mask)
        for i, b in enumerate(w.blocks):
            self.forward_block(b, i, io)

        # OutLayer: (shift, scale) modulation, norm_mod, out_layer, unpatchify + grid split
        om = self.buf(self.p.mod, 1, 2 * D)
        self.modulation(w.out_mod, tm_v, om)
        vn = self.buf(self.p.v[0], N, D)
        self.k.norm_mod(vis, cols(om, Mod.out(D, 1), D), cols(om, Mod.out(D, 0), D), vn)
        head_v = self.buf(self.p.head, N, 4 * c.out_visual_dim)
        self.linear(w.out, vn, head_v)
        self.emit("out_layer", head_v)
        self.k.unpatchify(head_v, out.video, sh, c.out_visual_dim, c.n_grid)
        # OutLayerAudio: modulation, norm_mod, the second (non-affine) LayerNorm on bf16, out_layer, grid split
        am = self.buf(self.p.mod2, 1, 2 * Da)
        self.modulation(w.audio_out_mod, tm_a, am)
        an = self.buf(self.p.a[0], A, Da)
        self.k.norm_mod(aud, cols(am, Mod.out(Da, 1), Da), cols(am, Mod.out(Da, 0), Da), an)
        self.k.layer_norm(an, None, None, an)
        head_a = self.buf(self.p.head, A, c.out_audio_dim)
        self.linear(w.audio_out, an, head_a)
        self.emit("audio_out_layer", head_a)
        self.k.grid_major(head_a, out.audio, c.n_grid)
        self.emit("video_velocity_grid", out.video)
        self.emit("audio_velocity_grid", out.audio)


# ---------------------------------------------------------------------------------------------------- entry points
def text_block(k: K.Kernels, weights: dict, c: Config, sh: Shape, i: int, prefix: str, te, tm, rope,
               mask: K.Mask | None = None, *, fp32_linear: str, hook: Hook | None = None,
               ffn_rows: int = DEFAULT_FFN_ROWS, fill: int | None = None) -> torch.Tensor:
    """Dit.textBlock: te [L, md] bf16 updated in place (and returned); tm [1, T] bf16; rope [L, 256] fp32."""
    x = Dit(k, Plan(c, sh, ffn_rows), fp32_linear, hook=hook, fill=fill)
    x.text_block(bind_text_block(c, weights, prefix, i), i, prefix, te, tm, rope, mask)
    return te


def forward_block(k: K.Kernels, weights: dict, c: Config, sh: Shape, i: int, io: BlockIo, *, fp32_linear: str,
                  hook: Hook | None = None, ffn_rows: int = DEFAULT_FFN_ROWS, fill: int | None = None) -> BlockIo:
    """Dit.forwardBlock: io.vis [N, D] and io.aud [A, Da] (bf16) updated in place."""
    x = Dit(k, Plan(c, sh, ffn_rows), fp32_linear, hook=hook, fill=fill)
    x.forward_block(bind_block(c, weights, i), i, io)
    return io


def outputs(c: Config, sh: Shape) -> Outputs:
    return Outputs(torch.empty(c.n_grid * sh.t * sh.h * sh.w, c.dx_visual(), dtype=torch.bfloat16, device="cuda"),
                   torch.empty(c.n_grid * sh.audio, c.dx_audio(), dtype=torch.bfloat16, device="cuda"))


def forward(k: K.Kernels, weights: dict, c: Config, sh: Shape, inp: Inputs, out: Outputs | None = None,
            *, fp32_linear: str, hook: Hook | None = None, ffn_rows: int = DEFAULT_FFN_ROWS,
            fill: int | None = None) -> Outputs:
    """Dit.forward: the PiFlow heads grid-major (video [10 * T * H * W, 16], audio [10 * A, 40], bf16)."""
    out = out if out is not None else outputs(c, sh)
    x = Dit(k, Plan(c, sh, ffn_rows), fp32_linear, hook=hook, fill=fill)
    x.forward(bind_weights(c, weights), inp, out)
    return out


def capture_names(c: Config) -> list[str]:
    """Every name forward() reports, in order (block.zig's emit calls)."""
    names: list[str] = []
    for p in ("video", "audio"):
        names += [f"{p}_{s}" for s in ("text_embeddings", "pooled_text_embeddings", "time_embeddings",
                                       "time_plus_pooled")]
        for i in range(c.num_text_blocks):
            b = f"{p}_text_transformer_blocks.{i}"
            names += [f"{b}.text_modulation", f"{b}.attn", f"{b}.feed_forward", b]
    names += ["visual_embeddings", "audio_embeddings"]
    for i in range(c.num_visual_blocks):
        b = f"visual_transformer_blocks.{i}."
        names += [b + s for s in (
            "video_dec_block.visual_modulation", "video_dec_block.self_attention",
            "video_dec_block.self_attention_residual", "video_dec_block.pre_ca", "video_dec_block.cross_attention",
            "audio_dec_block.visual_modulation", "audio_dec_block.self_attention", "audio_dec_block.pre_ca",
            "audio_dec_block.cross_attention", "va_modulation", "av_modulation",
            "video_dec_block.cross_attention_residual", "va_cross_attention", "av_cross_attention",
            "video_dec_block.feed_forward", "audio_dec_block.feed_forward", "video", "audio")]
    names += ["out_layer", "audio_out_layer", "video_velocity_grid", "audio_velocity_grid"]
    return names
