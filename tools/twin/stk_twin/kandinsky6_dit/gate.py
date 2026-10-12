"""python -m stk_twin.kandinsky6_dit.gate --out DIR [--fatbins DIR] --phase {ops,blocks}
--fp32-linear {tf32_rna,tf32_truncate,fp32} [--seed N] [--sites PATTERN ...] [--text-valid 777] [--max-gib 90]
[--timeout S] [--list]: the Kandinsky-6 DiT native-receipt capture and bit gate on one CUDA GPU (SM121, approved
window only). Every UNQUALIFIED row of kernels/cuda/kandinsky6/dit/README.md "Precision receipt" is measured here at
the production shape: the upstream kandinsky-6 eac06e39 module (native.py: built as factory.py builds it, SDPA, the
pipeline's own "high" float32 matmul policy) and the twin kernel (kernels.py, block.py: the engine's fatbins) run on
the same inputs and their output bits are compared. Equality is the gate: no tolerance anywhere; an unequal site is
reported (count, max |diff|, first mismatch, both sha256), never waived. test_ops.py is the twin's self-consistency at
reduced shapes; this job adds nothing to it but the production shapes and the native receipts.

Shape: block.Shape.preset(1024), the 864x480 5 s preset: 31 x 60 x 108 latents = 50,220 visual tokens, 218 audio rows,
1024 text tokens of which --text-valid (777) are kept by the key mask; the checkpoint's widths
(native.transformer_config: D 4096 / 2048, 32 / 16 heads x 128, FFN 16384 / 7168, time 1024, text 3584, pooled 768)
with one fused and one text block a stack. Weights: test_ops.random_weights (seeded, realistic scale; values do not
steer algorithm choice).

--phase ops, one site a native op (`--list` prints them with shapes and estimates):
  linear/*     every bf16 Linear shape class (+bias, addmm): q/k/v/o at N, A and L for both widths, va / av k-v,
               patch embedding (VisualEmbeddings, K = 132), audio embedding (K = 40), text / pooled embeddings, the
               4096 -> 640 and 2048 -> 400 heads;
  ffn/*        _GELUProjection (mm, GELU) and the bias-free down projection, both modalities at N / A and in the text
               blocks at L; the twin in ffn_rows = 4096 row chunks as block.py runs it;
  fp32/*       the fp32 linears in --fp32-linear (time linear_1 / linear_2, every Modulation class: text 6-way both
               widths, video / audio dec 9-way, va, av, out, audio out), compared as the fp32 F.linear output;
  norm/*       affine LayerNorm (bf16: text, pooled, audio embeddings), norm_mod (apply_scale_shift_norm) at N, A, L,
               the OutLayerAudio second LayerNorm;
  qk/*         RMSNorm(128) q / k with and without the RoPE rotation, every rows x heads x rope combination;
  attn/*       the eight attention.Site calls (SDPA; the backend that ran is read from the profiled aten op and kernel
               names: flash / efficient / cudnn / math);
  elem/*       gated residual at N, time + pooled add, the time sinusoid (expected-equal rows, receipted once).
Both phases start with tables/rope_and_time_freqs, an input contract of the runtime rather than a twin kernel and
outside the exit status: the visual RoPE3D table (31 x 30 x 54, scale (1, 2, 2), [50,220, 256]), audio RoPE1D (218),
text RoPE1D (1024, both stacks) and both TimeEmbeddings.freqs, built as factory.py and pipeline.py _materialize_ropes
build them (args / freqs on the CPU at load, cos / sin on the GPU, the max-canvas VisualRopeCache slice), always stored
as fixtures whatever their size, checked run to run, and reported (not gated) against an all-CPU and an all-GPU
producer, so a producer knows whether it must follow torch's GPU semantics.
--phase blocks, one unit a chain, compared capture by capture (native.Capture vs block.py's hook, test_ops.recorder):
  block/time_chain (text, pooled and time embeddings, time + pooled, both modalities), block/text_video and
  block/text_audio (TransformerEncoderBlock 0), block/embeddings (visual and audio), block/fused
  (FusedTransformerDecoderBlock 0 at 50,220 tokens). The first unequal capture in execution order is the chained
  bit gate's first divergence.

Processes and logs: this process never initializes CUDA; it runs one child (`--worker NAME`) a site, unit and the
environment receipt. cuBLAS and cuBLASLt read their logging variables once, when the library initializes, so they are
set in the child's environment before its interpreter starts: CUBLASLT_LOG_LEVEL=5 and CUBLASLT_LOG_MASK=31 (error,
trace, hints, heuristics info, API trace) with CUBLASLT_LOG_FILE=DIR/logs/<site>.cublaslt.log, and
CUBLAS_LOGINFO_DBG=1 with CUBLAS_LOGDEST_DBG=DIR/logs/<site>.cublas.log (torch's bias-free GEMMs go through
cublasGemmEx). A log therefore holds that site's native calls only, in order: the profiled call, then the repeat call
(inputs are made without a GEMM and the twin makes no cuBLAS call). The logs are summarized here after the child exits,
when they are flushed. The child also writes the torch profiler trace (DIR/traces/<site>.json, CUDA kernels with grid,
block, shared memory and registers, and the aten ops), DIR/sites/<site>.json, and fixtures.

Oracle settings: native.pipeline_numerics() (set_float32_matmul_precision("high"), factory.py:162) and nothing else;
never "highest", never TF32 off. Each child asserts the policy before and after its run. Environment overrides that
change the native algorithm or precision (NVIDIA_TF32_OVERRIDE, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, DISABLE_ADDMM_CUDA_LT,
TORCH_BLAS_PREFER_CUBLASLT, CUBLAS_WORKSPACE_CONFIG, CUBLASLT_WORKSPACE_SIZE, TORCH_CUDNN_SDPA_ENABLED, CUBLAS_EMULAT*)
are setup errors; the upstream checkout must be at eac06e39 when its git directory is readable.

Outputs: DIR/env.json (torch / CUDA / driver / cuBLAS / cuBLASLt / cuDNN versions and library sha256s, device, the
precision and SDPA flags, environment variables, fatbin and upstream digests); DIR/receipt.json and
DIR/receipt-<phase>.json {env, tables, sites[], blocks{}, first_divergence, ok};
DIR/fixtures/<site>/<tensor>.{bin,json} for every input and native output of at most 4 Mi elements (raw
little-endian bytes, json header with dtype, shape and sha256) so Zig / twin runs can replay them offline; larger
tensors are recorded by sha256 only (the input tables excepted).

Resources (estimates, checked against --max-gib before a child starts and against free device memory in the child;
GB10 memory is unified, hashing copies 256 MiB at a time; torch.cuda.max_memory_allocated is recorded a site):
  ops     peak <= ~9 GiB a site (ffn/video_up: seeded weights 1.5 GiB + x 0.4 + native proj and GELU outputs, repeat
          and twin output 4 x 1.6 + compare chunks); 65 children of ~10-20 s each (interpreter, CUDA context, weights,
          hashing up to 1.6 GiB outputs) plus attn/video_self (~41 TFLOP native, the streaming twin a few times
          longer): ~15-30 min.
  blocks  peak ~19 GiB (block/fused: weights 1.5 + twin scratch plan 4.0 + native transients ~5 [fp32 LayerNorm
          temporaries, FFN hidden + GELU at 50,220 x 16384, q / k / v and SDPA transposes] + both capture sets ~7 +
          inputs); ~70 TFLOP native a fused block plus the twin: 6 children, ~5-10 min.

Exit status: 0 iff every site / capture is bit-equal, 1 otherwise (or a child failed), 2 on a setup error.
"""

from __future__ import annotations

import argparse
import ctypes
import fnmatch
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from . import block as B
from . import budget
from . import kernels as K
from . import native as N
from . import test_ops as T

BF = torch.bfloat16
F32 = torch.float32
DEV = "cuda"
GIB = 2**30
MODULE = f"{__package__}.gate"
UPSTREAM_COMMIT = "eac06e39b904940baff2c408e925cdb0c4529269"
ORACLE_TORCH = "2.14.1"
TEXT_VALID = 777
FIXTURE_MAX = 4 * 2**20  # elements: a tensor this small is written as a fixture, larger ones only hashed
CHUNK = 64 * 2**20  # elements a comparison step
HASH_BYTES = 256 * 2**20  # bytes a host copy while hashing / writing
SLACK = GIB  # compare chunks, cuBLAS workspace, allocator rounding
TIME = 700.0  # sigma 0.7 x 1000, native_inputs' default
FORBIDDEN_ENV = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "DISABLE_ADDMM_CUDA_LT",
                 "TORCH_BLAS_PREFER_CUBLASLT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE",
                 "TORCH_CUDNN_SDPA_ENABLED")
FORBIDDEN_PREFIX = ("CUBLAS_EMULAT",)
RECORDED_ENV = re.compile(r"^(CUBLAS|CUBLASLT|NVIDIA|TORCH|PYTORCH|CUDA|CUDNN|STK_K6)_|^DISABLE_ADDMM_CUDA_LT$")
UPSTREAM_FILES = ("kandinsky/core/components/dit.py", "kandinsky/core/components/piflow_dit.py",
                  "kandinsky/core/components/rope.py", "kandinsky/core/tensors.py",
                  "kandinsky/core/algo/prepare_ropes.py", "kandinsky/runtime/kernels/sdpa.py",
                  "kandinsky/runtime/kernels/attention_engine.py")
BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32, torch.uint8: torch.uint8,
        torch.int32: torch.int32, torch.bool: torch.uint8}
GEMM_OPS = ("aten::addmm", "aten::mm", "aten::bmm", "aten::_addmm_activation", "aten::native_layer_norm",
            "aten::_fused_rms_norm", "aten::gelu")


class SetupError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------- the geometry
def geometry() -> tuple[dict, B.Config, B.Shape]:
    """The checkpoint's widths with one fused and one text block a stack, at the 864x480 preset (no CUDA)."""
    tc = N.reduced(N.transformer_config(), visual_blocks=1, text_blocks=1)
    c = N.twin_config(tc)
    sh = B.Shape.preset(B.MAX_TEXT)
    sh.validate()
    return tc, c, sh


def nbytes(*shape, size: int = 2) -> int:
    return math.prod(shape) * size


def weight_bytes(c: B.Config) -> int:
    """random_weights' footprint: every manifest tensor in bf16 plus the largest fp32 draw."""
    sizes = [math.prod(s) for _n, _d, s in B.manifest(c)]
    return 2 * sum(sizes) + 4 * max(sizes)


def safe(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")


def site_seed(seed: int, name: str) -> int:
    """A site's input seed: independent of which other sites run."""
    return (seed * 1_000_003 + zlib.crc32(name.encode())) % 2**62


# --------------------------------------------------------------------------------------------------------- the cases
@dataclass
class Case:
    """One site, ready to run: the native call (upstream modules / statements), the twin call (kernels.py / block.py),
    both returning row tensors of one shape (or capture dicts), and the inputs they share."""

    inputs: dict
    native: Callable[[], object]
    twin: Callable[[], object]
    twin_math: dict
    native_call: str


@dataclass(frozen=True)
class Site:
    name: str
    kind: str
    shapes: dict
    dtypes: dict
    est: int  # peak device bytes (estimate)
    make: Callable[["Ctx"], Case]
    covers: str = ""
    weights: bool = True
    repeat: bool = True
    gated: bool = True  # False: reported in the receipt, outside the exit status


def rope_modules_built(c: B.Config, args_on: str, run_on: str) -> dict:
    """native.rope_modules' four RoPE modules with their `args` tables built on `args_on`, then moved to `run_on`
    (where compute_rope1d / VisualRopeCache index them and run cos / sin). "cpu" is factory.py:142-148: constructed
    on the meta device, runtime/weights.py materialize_meta_buffers on the CPU (reset_parameters: get_freqs' exp and
    the outer product there), then dit.to(device); "cuda" constructs them on the GPU (exp and outer on the GPU)."""
    if args_on == "cpu":
        with torch.device("meta"):
            mods = N.rope_modules(c, device="meta")
        for m in mods.values():
            N.materialize_meta_buffers(m, torch.device("cpu"))
    else:
        with torch.device(args_on):
            mods = N.rope_modules(c, device=args_on)
    return {k: m.to(run_on) for k, m in mods.items()}


def time_freqs_built(md: int, td: int, args_on: str, run_on: str) -> torch.Tensor:
    """TimeEmbeddings(md, td).freqs as native.freqs_of reads it ([1, md / 2] fp32), built on `args_on` the same two
    ways (factory: meta, then reset_parameters' get_freqs on the CPU), moved to `run_on`."""
    cls = N.upstream().dit.TimeEmbeddings
    if args_on == "cpu":
        with torch.device("meta"):
            m = cls(md, td)
        N.materialize_meta_buffers(m, torch.device("cpu"))  # touches the freqs buffer only; the Linears stay meta
    else:
        with torch.device(args_on):
            m = cls(md, td)
    return m.freqs.to(run_on).float().reshape(1, -1).contiguous()


def native_tables(c: B.Config, sh: B.Shape, args_on: str, run_on: str) -> dict:
    """The DiT's input tables as pipeline.py _materialize_ropes (146-180) makes them: the visual table from the
    max-canvas VisualRopeCache then sliced (native.ropes, prepare_ropes.py), audio / text from compute_rope1d, as the
    twin's rows (native.rope_rows: [S, 256] fp32 = [S, 64, 2, 2], (t, h', w') flattened); the time freqs."""
    r = N.ropes(rope_modules_built(c, args_on, run_on), c, sh)
    return {"visual_rope": N.rope_rows(r["visual"]), "audio_rope": N.rope_rows(r["audio"]),
            "text_rope_video": N.rope_rows(r["text"][0]), "text_rope_audio": N.rope_rows(r["text"][1]),
            "freqs_video": time_freqs_built(c.model_dim, c.time_dim, args_on, run_on),
            "freqs_audio": time_freqs_built(c.model_dim_a, c.time_dim, args_on, run_on)}


class Ctx:
    """A child's state: kernels, upstream, the geometry, the seeded generator, lazily the weights and the ropes."""

    def __init__(self, a, name: str):
        self.k = K.Kernels(fatbins=a.fatbins)
        self.u = N.upstream()
        self.tc, self.c, self.sh = geometry()
        self.valid, self.math, self.seed = a.text_valid, a.fp32_linear, a.seed
        self.site_seed = site_seed(a.seed, name)
        self.g = torch.Generator(device=DEV).manual_seed(self.site_seed)
        self._w = None
        self._ropes = None

    def weights(self) -> dict:
        if self._w is None:
            self._w = T.random_weights(self.c, self.seed)
        return self._w

    def rnd(self, *shape, scale: float = 1.0, shift: float = 0.0, dtype=BF) -> torch.Tensor:
        t = torch.randn(*shape, generator=self.g, device=DEV, dtype=dtype)
        if scale != 1.0:
            t.mul_(scale)
        if shift != 0.0:
            t.add_(shift)
        return t

    def mask(self) -> K.Mask:
        return K.Mask.of(torch.arange(self.sh.text, device=DEV) < self.valid)

    def ropes(self) -> dict:
        """The pipeline's tables (native.ropes on factory-built modules): visual [T, H', W', 1, 64, 2, 2], audio /
        text [S, 1, 64, 2, 2]."""
        if self._ropes is None:
            self._ropes = N.ropes(rope_modules_built(self.c, "cpu", DEV), self.c, self.sh)
        return self._ropes

    def rope(self, kind: str | None) -> torch.Tensor | None:
        """The table as the DiT passes it to attention (visual flattened over (t, h', w'), _embed_visual)."""
        if kind is None:
            return None
        r = self.ropes()
        if kind == "visual":
            return r["visual"].flatten(0, 2)
        return {"audio": r["audio"], "text_v": r["text"][0], "text_a": r["text"][1]}[kind]

    def linear(self, p: str, n_in: int, n_out: int, bias: bool = True) -> nn.Linear:
        W = self.weights()
        sd = {"weight": W[f"{p}.weight"]}
        if bias:
            sd["bias"] = W[f"{p}.bias"]
        return N.build(nn.Linear, n_in, n_out, bias=bias, sd=sd)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def gemm_math(x2, w, m: int, n: int, epi: str, chunk: int | None = None) -> dict:
    rows = min(m, chunk) if chunk else m
    return {"kernel": "k6_gemm_bf16", "epilogue": epi, "vec": K.Kernels.gemm_vec(x2, w), "m": m, "n": n,
            "k": int(w.shape[1]), "grid": [cdiv(n, 128), cdiv(rows, 128)], "block": 256, "row_chunk": chunk,
            "math": "mma.sync bf16, fp32 accumulation, K ascending in 32-wide tiles, no split-K, one rounding"}


def gemv_math(math_: str, m: int, n: int, k: int, silu: bool, out: str) -> dict:
    tf32 = math_ != "fp32"
    return {"kernel": "k6_gemv_tf32" if tf32 else "k6_gemv_f32", "math": math_, "m": m, "n": n, "k": k,
            "grid": [cdiv(n, 64), cdiv(m, 16)] if tf32 else [cdiv(n, 8)], "block": 256, "silu": silu, "out": out}


def ffn_twin(k: K.Kernels, x2: torch.Tensor, w: torch.Tensor, out: torch.Tensor, epi: int) -> None:
    """block.Dit.ffn's row chunking (ffn_rows = 4096): rows are independent, the chunks only bound the hidden buffer."""
    r = 0
    while r < x2.shape[0]:
        n = min(B.DEFAULT_FFN_ROWS, x2.shape[0] - r)
        k.gemm(x2[r:r + n], w, None, out[r:r + n], epi)
        r += B.DEFAULT_FFN_ROWS


# ----------------------------------------------------------------------------------------------------- op site makers
def linear_site(name: str, p: str, x_shape: tuple, n_in: int, n_out: int, covers: str, wb: int) -> Site:
    m = math.prod(x_shape[:-1])
    est = wb + nbytes(*x_shape) + 3 * nbytes(m, n_out) + SLACK

    def make(ctx: Ctx) -> Case:
        lin = ctx.linear(p, n_in, n_out)
        x = ctx.rnd(*x_shape)
        x2 = x.reshape(m, n_in)

        def twin():
            out = torch.empty(m, n_out, dtype=BF, device=DEV)
            ctx.k.gemm(x2, lin.weight, lin.bias.view(1, -1), out, K.EPI_BIAS)
            return out

        return Case({"x": x, "weight": lin.weight, "bias": lin.bias}, lambda: lin(x).reshape(m, n_out), twin,
                    gemm_math(x2, lin.weight, m, n_out, "bias"), f"nn.Linear({n_in}, {n_out}) [{p}] on {list(x_shape)}")

    return Site(name, "linear", {"x": list(x_shape), "weight": [n_out, n_in], "bias": [n_out], "out": [m, n_out]},
                {"x": "bf16", "weight": "bf16", "out": "bf16"}, est, make, covers)


def visual_embeddings_site(c: B.Config, sh: B.Shape, wb: int) -> Site:
    """VisualEmbeddings (dit.py 92-113): patchify, .to(bf16), Linear K = 132; the twin's patchify + GEMM."""
    D, C, Nt = c.model_dim, c.visual_in(), sh.tokens()
    x_shape = (1, sh.t, sh.h, sh.w, C)
    est = wb + nbytes(*x_shape, size=4) + 3 * nbytes(Nt, c.patch_in()) + 3 * nbytes(Nt, D) + SLACK

    def make(ctx: Ctx) -> Case:
        ve = N.build(ctx.u.dit.VisualEmbeddings, C, D, (1, 2, 2), sd=N.sub(ctx.weights(), "visual_embeddings."))
        x = ctx.rnd(*x_shape, dtype=F32)
        prow = torch.empty(Nt, c.patch_in(), dtype=BF, device=DEV)

        def twin():
            out = torch.empty(Nt, D, dtype=BF, device=DEV)
            ctx.k.patchify(x.reshape(-1, C), prow, sh, C)
            ctx.k.gemm(prow, ve.in_layer.weight, ve.in_layer.bias.view(1, -1), out, K.EPI_BIAS)
            return out

        return Case({"x_video": x, "weight": ve.in_layer.weight, "bias": ve.in_layer.bias},
                    lambda: ve(x).reshape(Nt, D), twin,
                    {"patchify": "k6_patchify f32", **gemm_math(prow, ve.in_layer.weight, Nt, D, "bias")},
                    f"VisualEmbeddings({C}, {D}, (1, 2, 2)) on {list(x_shape)} fp32")

    return Site("linear/visual_embeddings", "linear",
                {"x_video": list(x_shape), "weight": [D, c.patch_in()], "out": [Nt, D]},
                {"x_video": "f32", "weight": "bf16", "out": "bf16"}, est, make, "patch embedding, K = 132")


def audio_embeddings_site(c: B.Config, sh: B.Shape, wb: int) -> Site:
    """TextEmbeddings(40, 2048).in_layer on x_audio.to(bf16) (dit.py 79-81); the twin's cast + GEMM."""
    Da, Ca, A = c.model_dim_a, c.in_audio_dim, sh.audio

    def make(ctx: Ctx) -> Case:
        lin = ctx.linear("audio_embeddings.in_layer", Ca, Da)
        x = ctx.rnd(1, A, Ca, dtype=F32)

        def twin():
            xb = torch.empty(A, Ca, dtype=BF, device=DEV)
            out = torch.empty(A, Da, dtype=BF, device=DEV)
            ctx.k.cast_bf16(x.reshape(A, Ca), xb)
            ctx.k.gemm(xb, lin.weight, lin.bias.view(1, -1), out, K.EPI_BIAS)
            return out

        xb0 = torch.empty(A, Ca, dtype=BF, device=DEV)
        return Case({"x_audio": x, "weight": lin.weight, "bias": lin.bias},
                    lambda: lin(x.to(dtype=lin.weight.dtype)).reshape(A, Da), twin,
                    {"cast": "k6_cast_bf16", **gemm_math(xb0, lin.weight, A, Da, "bias")},
                    "audio_embeddings.in_layer(x_audio.to(bf16)) (TextEmbeddings.forward)")

    return Site("linear/audio_embeddings", "linear", {"x_audio": [1, A, Ca], "weight": [Da, Ca], "out": [A, Da]},
                {"x_audio": "f32", "weight": "bf16", "out": "bf16"}, wb + 4 * nbytes(A, Da) + SLACK, make, "K = 40")


def ffn_up_site(name: str, p: str, x_shape: tuple, d: int, ff: int, wb: int) -> Site:
    m = math.prod(x_shape[:-1])
    est = wb + nbytes(*x_shape) + 4 * nbytes(m, ff) + SLACK

    def make(ctx: Ctx) -> Case:
        up = ctx.weights()[f"{p}.feed_forward.net.0.proj.weight"]
        proj = N.build(ctx.u.dit._GELUProjection, d, ff, sd={"proj.weight": up})
        x = ctx.rnd(*x_shape)
        x2 = x.reshape(m, d)

        def twin():
            out = torch.empty(m, ff, dtype=BF, device=DEV)
            ffn_twin(ctx.k, x2, up, out, K.EPI_GELU)
            return out

        return Case({"x": x, "weight": up}, lambda: proj(x).reshape(m, ff), twin,
                    gemm_math(x2, up, m, ff, "gelu", B.DEFAULT_FFN_ROWS),
                    f"_GELUProjection({d}, {ff}) [{p}.feed_forward.net.0] on {list(x_shape)}: F.gelu(proj(x))")

    return Site(name, "ffn", {"x": list(x_shape), "weight": [ff, d], "out": [m, ff]},
                {"x": "bf16", "weight": "bf16", "out": "bf16"}, est, make, "bias-free GEMM (mm) + exact GELU")


def ffn_down_site(name: str, p: str, x_shape: tuple, d: int, ff: int, wb: int) -> Site:
    m = math.prod(x_shape[:-1])
    h_shape = (*x_shape[:-1], ff)
    est = wb + 2 * nbytes(m, ff) + 3 * nbytes(m, d) + SLACK

    def make(ctx: Ctx) -> Case:
        down = ctx.weights()[f"{p}.feed_forward.net.2.weight"]
        lin = N.build(nn.Linear, ff, d, bias=False, sd={"weight": down})
        h = F.gelu(ctx.rnd(*h_shape))  # a GELU output's distribution, elementwise (no GEMM before the native call)
        h2 = h.reshape(m, ff)

        def twin():
            out = torch.empty(m, d, dtype=BF, device=DEV)
            ffn_twin(ctx.k, h2, down, out, K.EPI_BIAS)
            return out

        return Case({"h": h, "weight": down}, lambda: lin(h).reshape(m, d), twin,
                    gemm_math(h2, down, m, d, "none", B.DEFAULT_FFN_ROWS),
                    f"nn.Linear({ff}, {d}, bias=False) [{p}.feed_forward.net.2] on {list(h_shape)}")

    return Site(name, "ffn", {"h": list(h_shape), "weight": [d, ff], "out": [m, d]},
                {"h": "bf16", "weight": "bf16", "out": "bf16"}, est, make, "bias-free GEMM (mm)")


def modulation_site(name: str, p: str, n_out: int, c: B.Config, wb: int) -> Site:
    """Modulation.forward (dit.py 124-131) up to its fp32 F.linear output (the `.to(x.dtype)` follows)."""
    Td = c.time_dim

    def make(ctx: Ctx) -> Case:
        W = ctx.weights()
        mod = N.build(ctx.u.dit.Modulation, Td, n_out, 1,
                      sd={"out_layer.weight": W[f"{p}.weight"], "out_layer.bias": W[f"{p}.bias"]})
        x = ctx.rnd(1, Td)  # a time + pooled embedding row, bf16
        w, b = mod.out_layer.weight, mod.out_layer.bias

        def native():
            return F.linear(mod.act(x.float()), w.float(), b.float())

        def twin():
            out = torch.empty(1, n_out, dtype=F32, device=DEV)
            ctx.k.gemv(x, w, b.view(1, -1), out, True, ctx.math)
            return out

        return Case({"x": x, "weight": w, "bias": b}, native, twin, gemv_math(ctx.math, 1, n_out, Td, True, "f32"),
                    f"F.linear(SiLU(x.float()), W.float(), b.float()) [{p}] under float32 matmul precision 'high'")

    return Site(name, "fp32", {"x": [1, Td], "weight": [n_out, Td], "out": [1, n_out]},
                {"x": "bf16", "weight": "bf16 -> f32", "out": "f32"}, wb + SLACK, make, "Modulation")


def time_linear_site(name: str, prefix: str, which: int, md: int, c: B.Config, wb: int) -> Site:
    """TimeEmbeddings.forward (dit.py 48-64): linear_1 on [cos, sin] of outer(t, freqs), or linear_2 on SiLU(h)."""
    Td = c.time_dim
    k_in = md if which == 1 else Td

    def make(ctx: Ctx) -> Case:
        te = N.build(ctx.u.dit.TimeEmbeddings, md, Td, sd=N.sub(ctx.weights(), f"{prefix}_time_embeddings."))
        e = te.timestep_embedder
        lin = e.linear_1 if which == 1 else e.linear_2
        if which == 1:
            t = torch.tensor([TIME], dtype=F32, device=DEV)
            args = torch.outer(t.float(), te.freqs.to(device=t.device, dtype=F32))
            x = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            x = ctx.rnd(1, Td, dtype=F32)  # a linear_1 output row

        def native():
            return F.linear(x if which == 1 else e.act(x), lin.weight.float(), lin.bias.float())

        def twin():
            out = torch.empty(1, Td, dtype=F32, device=DEV)
            ctx.k.gemv(x, lin.weight, lin.bias.view(1, -1), out, which == 2, ctx.math)
            return out

        stmt = "F.linear(embed, linear_1.weight.float(), ...)" if which == 1 else \
            "F.linear(act(h), linear_2.weight.float(), ...)"
        return Case({"x": x, "weight": lin.weight, "bias": lin.bias}, native, twin,
                    gemv_math(ctx.math, 1, Td, k_in, which == 2, "f32"),
                    f"{prefix}_time_embeddings: {stmt} under float32 matmul precision 'high'")

    return Site(name, "fp32", {"x": [1, k_in], "weight": [Td, k_in], "out": [1, Td]},
                {"x": "f32", "weight": "bf16 -> f32", "out": "f32"}, wb + SLACK, make, "TimeEmbeddings")


def layer_norm_site(name: str, p: str | None, x_shape: tuple, d: int, wb: int) -> Site:
    """nn.LayerNorm(d) on bf16 rows: affine (TextEmbeddings.norm, weights `p`) or plain (OutLayerAudio's second)."""
    m = math.prod(x_shape[:-1])

    def make(ctx: Ctx) -> Case:
        if p is None:
            ln = nn.LayerNorm(d, elementwise_affine=False, device=DEV)
            inputs = {}
        else:
            W = ctx.weights()
            ln = N.build(nn.LayerNorm, d, sd={"weight": W[f"{p}.weight"], "bias": W[f"{p}.bias"]})
            inputs = {"weight": ln.weight, "bias": ln.bias}
        x = ctx.rnd(*x_shape, scale=2.0, shift=0.25)
        inputs = {"x": x, **inputs}

        def twin():
            y = torch.empty(m, d, dtype=BF, device=DEV)
            wv = None if p is None else ln.weight.view(1, -1)
            bv = None if p is None else ln.bias.view(1, -1)
            ctx.k.layer_norm(x.reshape(m, d), wv, bv, y)
            return y

        return Case(inputs, lambda: ln(x).reshape(m, d), twin,
                    {"kernel": "k6_layer_norm", "grid": [m], "block": list(K.NORM_BLOCK), "affine": p is not None},
                    f"nn.LayerNorm({d}{'' if p else ', elementwise_affine=False'}) [{p or 'audio_out_layer.norm'}] "
                    f"on {list(x_shape)} bf16")

    return Site(name, "norm", {"x": list(x_shape), "out": [m, d]}, {"x": "bf16", "out": "bf16"},
                (wb if p else 0) + 6 * nbytes(m, d) + SLACK, make, weights=p is not None)


def norm_mod_site(name: str, x_shape: tuple, d: int, parts: int) -> Site:
    """apply_scale_shift_norm(LayerNorm(d, elementwise_affine=False), x, scale, shift) (tensors.py 27-32) with the
    first part's scale / shift chunks of a [1, parts * 3 * d] modulation row."""
    m = math.prod(x_shape[:-1])

    def make(ctx: Ctx) -> Case:
        ln = nn.LayerNorm(d, elementwise_affine=False, device=DEV)
        x = ctx.rnd(*x_shape, scale=2.0, shift=0.25)
        mod = ctx.rnd(1, parts * 3 * d, scale=0.5)
        shift, scale = B.cols(mod, B.Mod.dec(d, 0, 0), d), B.cols(mod, B.Mod.dec(d, 0, 1), d)

        def twin():
            y = torch.empty(m, d, dtype=BF, device=DEV)
            ctx.k.norm_mod(x.reshape(m, d), scale, shift, y)
            return y

        return Case({"x": x, "modulation": mod},
                    lambda: ctx.u.tensors.apply_scale_shift_norm(ln, x, scale, shift).type_as(x).reshape(m, d), twin,
                    {"kernel": "k6_norm_mod", "grid": [m], "block": list(K.NORM_BLOCK)},
                    f"apply_scale_shift_norm(LayerNorm({d}, elementwise_affine=False), x {list(x_shape)}, scale, "
                    "shift)")

    return Site(name, "norm", {"x": list(x_shape), "scale": [1, d], "shift": [1, d], "out": [m, d]},
                {"x": "bf16", "scale": "bf16", "out": "bf16"}, 12 * nbytes(m, d) + SLACK, make, weights=False)


def qk_site(name: str, wname: str, rows: int, heads: int, rope: str | None, batched: bool, wb: int) -> Site:
    """RMSNorm(128) on [.., S, heads, 128], then apply_rotary(.., rope).type_as (dit.py 196-200, 268-271)."""
    width = heads * 128
    x_shape = (1, rows, heads, 128) if batched else (rows, heads, 128)

    def make(ctx: Ctx) -> Case:
        w = ctx.weights()[wname]
        rms = N.build(nn.RMSNorm, 128, sd={"weight": w})
        x = ctx.rnd(*x_shape, scale=3.0)
        rn = ctx.rope(rope)
        rr = None if rn is None else N.rope_rows(rn)

        def native():
            y = rms(x)
            if rn is not None:
                y = ctx.u.tensors.apply_rotary(y, rn).type_as(y)
            return y.reshape(rows, width)

        def twin():
            y = x.reshape(rows, width).clone()
            ctx.k.qk_norm_rope(y, w.view(1, 128), rr, heads)
            return y

        inputs = {"x": x, "weight": w}
        if rr is not None:
            inputs["rope"] = rr
        return Case(inputs, native, twin,
                    {"kernel": "k6_qk_norm_rope", "grid": [cdiv(rows * heads, 8)], "block": 256, "rope": rope},
                    f"nn.RMSNorm(128) [{wname}] on {list(x_shape)}" + (f", apply_rotary({rope})" if rope else ""))

    shapes = {"x": list(x_shape), "weight": [128], "out": [rows, width]}
    if rope:
        shapes["rope"] = [rows, 256]
    return Site(name, "qk", shapes, {"x": "bf16", "weight": "bf16", "rope": "f32", "out": "bf16"},
                wb + 8 * nbytes(rows, width) + 2 * nbytes(rows, 256, size=4) + SLACK, make)


def attention_site(site: str, sq: int, sk: int, width: int, masked: bool, valid: int) -> Site:
    """attention.Site: runtime/kernels/sdpa.py (transpose, contiguous, F.scaled_dot_product_attention, back) with the
    [1, 1, 1, L] bool key mask where the DiT passes one; the twin's k6_attention over every query row."""
    heads = width // 128

    def make(ctx: Ctx) -> Case:
        q, k, v = ctx.rnd(sq, width), ctx.rnd(sk, width), ctx.rnd(sk, width)
        mask = ctx.mask() if masked else None

        def native():
            args = (q.view(1, sq, heads, 128), k.view(1, sk, heads, 128), v.view(1, sk, heads, 128))
            o = ctx.u.sdpa.sdpa(*args, attn_mask=N.mask_native(mask)) if masked else ctx.u.sdpa.sdpa(*args)
            return o.reshape(sq, width)

        def twin():
            out = torch.empty(sq, width, dtype=BF, device=DEV)
            ctx.k.attend(q, k, v, out, heads, mask)
            return out

        inputs = {"q": q, "k": k, "v": v}
        if mask is not None:
            inputs["mask"] = mask.t
        return Case(inputs, native, twin,
                    {"kernel": "k6_attention", "grid": [cdiv(sq, K.BLOCK_ROWS), heads, 1], "block": K.ATTN_THREADS,
                     "shared": K.SHARED_BYTES, "keys": sk, "mask_valid": valid if masked else None,
                     "scale": "0x3db504f3"},
                    f"sdpa(q [1, {sq}, {heads}, 128], k / v [1, {sk}, {heads}, 128]"
                    + (", attn_mask [1, 1, 1, L] bool)" if masked else ")"))

    est = 4 * nbytes(sq, width) + 4 * nbytes(sk, width) + 2 * nbytes(sq, width) + SLACK
    shapes = {"q": [sq, width], "k": [sk, width], "v": [sk, width], "heads": heads, "out": [sq, width]}
    if masked:
        shapes["mask"] = [sk]
        shapes["valid"] = valid
    return Site(f"attn/{site}", "attention", shapes, {"q": "bf16", "mask": "bool", "out": "bf16"}, est, make,
                weights=False)


def gate_sum_site(c: B.Config, sh: B.Shape) -> Site:
    D, Nt = c.model_dim, sh.tokens()

    def make(ctx: Ctx) -> Case:
        x, o = ctx.rnd(1, Nt, D), ctx.rnd(1, Nt, D)
        mod = ctx.rnd(1, 9 * D, scale=0.5)
        gate = B.cols(mod, B.Mod.dec(D, 0, 2), D)

        def twin():
            y = x.reshape(Nt, D).clone()
            ctx.k.gate_sum(y, o.reshape(Nt, D), gate)
            return y

        return Case({"x": x, "out": o, "modulation": mod},
                    lambda: ctx.u.tensors.apply_gate_sum(x, o, gate).type_as(x).reshape(Nt, D), twin,
                    {"kernel": "k6_gate_sum", "grid": [Nt, cdiv(D, 256)], "block": 256},
                    "apply_gate_sum(x, out, gate).type_as(x)")

    return Site("elem/gate_sum_video", "elementwise", {"x": [1, Nt, D], "gate": [1, D], "out": [Nt, D]},
                {"x": "bf16", "out": "bf16"}, 16 * nbytes(Nt, D) + SLACK, make, weights=False)


def add_site(c: B.Config) -> Site:
    Td = c.time_dim

    def make(ctx: Ctx) -> Case:
        a, b = ctx.rnd(1, Td), ctx.rnd(1, Td)

        def twin():
            y = torch.empty(1, Td, dtype=BF, device=DEV)
            ctx.k.add_bf16(a, b, y)
            return y

        return Case({"time": a, "pooled": b}, lambda: a + b, twin, {"kernel": "k6_add_bf16"},
                    "time_embeddings(t) + pooled_proj (dit.py _time_embed)")

    return Site("elem/time_plus_pooled", "elementwise", {"a": [1, Td], "b": [1, Td], "out": [1, Td]},
                {"a": "bf16", "out": "bf16"}, SLACK, make, weights=False)


def sincos_site(name: str, md: int) -> Site:
    half = md // 2

    def make(ctx: Ctx) -> Case:
        freqs = N.time_freqs(md)  # TimeEmbeddings.freqs: CPU exp at load, moved
        t = torch.tensor([TIME], dtype=F32, device=DEV)

        def native():
            args = torch.outer(t.float(), freqs.reshape(-1).to(device=t.device, dtype=F32))
            return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        def twin():
            out = torch.empty(1, md, dtype=F32, device=DEV)
            ctx.k.time_sincos(t.view(1, 1), freqs.view(1, half), out)
            return out

        return Case({"time": t, "freqs": freqs}, native, twin,
                    {"kernel": "k6_time_sincos", "grid": [1, cdiv(half, 256)]},
                    "cat([cos(outer(t, freqs)), sin(outer(t, freqs))], -1) (TimeEmbeddings.forward)")

    return Site(name, "elementwise", {"time": [1], "freqs": [1, half], "out": [1, md]},
                {"time": "f32", "out": "f32"}, SLACK, make, weights=False)


TABLES = "tables/rope_and_time_freqs"
TABLE_BUILD = {
    "args_built_on": "cpu", "trig_on": "cuda",
    "construction": "factory.py:142-148 (meta construction, runtime/weights.py materialize_meta_buffers on the CPU: "
                    "RoPE reset_parameters = get_freqs exp x freqs_scaling and the outer product with arange, "
                    "TimeEmbeddings.reset_parameters = get_freqs; then .to(cuda)); pipeline.py _materialize_ropes "
                    "146-180: VisualRopeCache.get at the max canvas (31, 30, 54), scale (1, 2, 2), then the slice; "
                    "compute_rope1d for audio (218, freqs x 0.144) and text (1024); index, / scale, cos, sin, stack "
                    "on the GPU (the buffers' device)",
    "layout": "native.rope_rows: [S, 256] fp32 = [S, 64, 2, 2] (cos, -sin, sin, cos), (t, h', w') flattened; freqs "
              "[1, md / 2] fp32 as native.freqs_of",
}


def tables_site(c: B.Config, sh: B.Shape) -> Site:
    """The DiT's input tables, an input contract of the runtime rather than a twin kernel: always stored as fixtures
    (the 4 Mi limit does not apply), the native pair (args on the CPU, cos / sin on the GPU) checked run to run, and
    two producers reported against it, not gated: everything on the CPU, everything on the GPU."""
    est = 6 * nbytes(sh.tokens(), 256, size=4) + 4 * nbytes(sh.tokens(), 64, size=4) + SLACK
    shapes = {"visual_rope": [sh.tokens(), 256], "audio_rope": [sh.audio, 256], "text_rope_video": [sh.text, 256],
              "text_rope_audio": [sh.text, 256], "freqs_video": [1, c.model_dim // 2],
              "freqs_audio": [1, c.model_dim_a // 2]}
    return Site(TABLES, "tables", shapes, {k: "f32" for k in shapes}, est, lambda ctx: None,
                "RoPE3D / RoPE1D tables and TimeEmbeddings.freqs (inputs the pipeline must produce)", weights=False,
                gated=False)


def op_sites(valid: int) -> list[Site]:
    _tc, c, sh = geometry()
    wb = weight_bytes(c)
    D, Da, Td, FF, FFa = c.model_dim, c.model_dim_a, c.time_dim, c.ff_dim, c.ff_dim_a
    H, Ha, Nt, A, L = c.heads(), c.heads_a(), sh.tokens(), sh.audio, sh.text
    Ct, Cp = c.in_text_dim, c.in_text_dim2
    VB = "visual_transformer_blocks.0"
    VD, AD = f"{VB}.video_dec_block", f"{VB}.audio_dec_block"
    TV, TA = "video_text_transformer_blocks.0", "audio_text_transformer_blocks.0"
    hw = (sh.t, sh.h // 2, sh.w // 2)
    s = [
        linear_site("linear/video_qkvo", f"{VD}.self_attention.to_query", (1, Nt, D), D, D,
                    "video self-attention q / k / v / o, text cross-attention q / o, va q / o", wb),
        linear_site("linear/audio_qkvo", f"{AD}.self_attention.to_query", (1, A, Da), Da, Da,
                    "audio self-attention q / k / v / o, text cross-attention q / o, av q / o", wb),
        linear_site("linear/text_video_qkvo", f"{TV}.attn.to_query", (L, D), D, D,
                    "video text blocks q / k / v / o, video text cross-attention k / v", wb),
        linear_site("linear/text_audio_qkvo", f"{TA}.attn.to_query", (L, Da), Da, Da,
                    "audio text blocks q / k / v / o, audio text cross-attention k / v", wb),
        linear_site("linear/va_kv", f"{VB}.va_cross_attention.to_key", (1, A, Da), Da, D, "va k / v", wb),
        linear_site("linear/av_kv", f"{VB}.av_cross_attention.to_key", (1, Nt, D), D, Da, "av k / v", wb),
        linear_site("linear/video_text_embedding", "video_text_embeddings.in_layer", (L, Ct), Ct, D, "", wb),
        linear_site("linear/audio_text_embedding", "audio_text_embeddings.in_layer", (L, Ct), Ct, Da, "", wb),
        linear_site("linear/pooled_embedding", "video_pooled_text_embeddings.in_layer", (1, Cp), Cp, Td,
                    "video and audio pooled projections", wb),
        linear_site("linear/out_layer", "out_layer.out_layer", (1, *hw, D), D, 4 * c.out_visual_dim, "OutLayer", wb),
        linear_site("linear/audio_out_layer", "audio_out_layer.out_layer", (1, A, Da), Da, c.out_audio_dim,
                    "OutLayerAudio", wb),
        visual_embeddings_site(c, sh, wb),
        audio_embeddings_site(c, sh, wb),
    ]
    for tag, p, x_rows, d, ff in (("video", VD, (1, Nt), D, FF), ("audio", AD, (1, A), Da, FFa),
                                  ("text_video", TV, (L,), D, FF), ("text_audio", TA, (L,), Da, FFa)):
        s.append(ffn_up_site(f"ffn/{tag}_up", p, (*x_rows, d), d, ff, wb))
        s.append(ffn_down_site(f"ffn/{tag}_down", p, (*x_rows, d), d, ff, wb))
    for tag, p, n in (("text_video", f"{TV}.text_modulation.out_layer", 6 * D),
                      ("text_audio", f"{TA}.text_modulation.out_layer", 6 * Da),
                      ("video_dec", f"{VD}.visual_modulation.out_layer", 9 * D),
                      ("audio_dec", f"{AD}.visual_modulation.out_layer", 9 * Da),
                      ("va", f"{VB}.va_modulation.out_layer", 2 * D + Da),
                      ("av", f"{VB}.av_modulation.out_layer", 2 * Da + D),
                      ("out", "out_layer.modulation.out_layer", 2 * D),
                      ("audio_out", "audio_out_layer.modulation.out_layer", 2 * Da)):
        s.append(modulation_site(f"fp32/modulation_{tag}", p, n, c, wb))
    for prefix, md in (("video", D), ("audio", Da)):
        s.append(time_linear_site(f"fp32/time_{prefix}_linear_1", prefix, 1, md, c, wb))
        s.append(time_linear_site(f"fp32/time_{prefix}_linear_2", prefix, 2, md, c, wb))
    s += [
        layer_norm_site("norm/layer_norm_video_text_embedding", "video_text_embeddings.norm", (L, D), D, wb),
        layer_norm_site("norm/layer_norm_audio_text_embedding", "audio_text_embeddings.norm", (L, Da), Da, wb),
        layer_norm_site("norm/layer_norm_pooled_embedding", "video_pooled_text_embeddings.norm", (1, Td), Td, wb),
        layer_norm_site("norm/layer_norm_audio_embedding", "audio_embeddings.norm", (1, A, Da), Da, wb),
        layer_norm_site("norm/layer_norm_audio_out_second", None, (1, A, Da), Da, wb),
        norm_mod_site("norm/norm_mod_video", (1, Nt, D), D, 3),
        norm_mod_site("norm/norm_mod_audio", (1, A, Da), Da, 3),
        norm_mod_site("norm/norm_mod_text_video", (L, D), D, 2),
        norm_mod_site("norm/norm_mod_text_audio", (L, Da), Da, 2),
    ]
    for tag, w, rows, heads, rope, batched in (
            ("video_self_q", f"{VD}.self_attention.query_norm.weight", Nt, H, "visual", True),
            ("audio_self_q", f"{AD}.self_attention.query_norm.weight", A, Ha, "audio", True),
            ("text_video_q", f"{TV}.attn.query_norm.weight", L, H, "text_v", False),
            ("text_audio_q", f"{TA}.attn.query_norm.weight", L, Ha, "text_a", False),
            ("video_text_q", f"{VD}.cross_attention.query_norm.weight", Nt, H, None, True),
            ("video_text_k", f"{VD}.cross_attention.key_norm.weight", L, H, None, False),
            ("audio_text_q", f"{AD}.cross_attention.query_norm.weight", A, Ha, None, True),
            ("audio_text_k", f"{AD}.cross_attention.key_norm.weight", L, Ha, None, False),
            ("va_k", f"{VB}.va_cross_attention.key_norm.weight", A, H, "audio", True),
            ("av_k", f"{VB}.av_cross_attention.key_norm.weight", Nt, Ha, "visual", True)):
        s.append(qk_site(f"qk/{tag}", w, rows, heads, rope, batched, wb))
    for site, sq, sk, width, masked in (("text_video_self", L, L, D, True), ("text_audio_self", L, L, Da, True),
                                        ("video_self", Nt, Nt, D, False), ("audio_self", A, A, Da, False),
                                        ("video_text", Nt, L, D, True), ("audio_text", A, L, Da, True),
                                        ("video_from_audio", Nt, A, D, False), ("audio_from_video", A, Nt, Da, False)):
        s.append(attention_site(site, sq, sk, width, masked, valid))
    s += [gate_sum_site(c, sh), add_site(c), sincos_site("elem/time_sincos_video", D),
          sincos_site("elem/time_sincos_audio", Da)]
    return s


# --------------------------------------------------------------------------------------------------------- block units
def time_chain_unit(c: B.Config, sh: B.Shape, wb: int, plan: int) -> Site:
    """dit.py _encode_text up to the text blocks, both modalities: TextEmbeddings (token, pooled), TimeEmbeddings,
    time + pooled; the twin's Dit.encode_text with no text block."""
    D, Da, Td, L = c.model_dim, c.model_dim_a, c.time_dim, sh.text

    def make(ctx: Ctx) -> Case:
        W, u = ctx.weights(), ctx.u
        nat = N.native_inputs(c, sh, rope_modules_built(c, "cpu", DEV), seed=ctx.site_seed, text_valid=ctx.valid)
        mods = {}
        for p, md in (("video", D), ("audio", Da)):
            mods[p] = (N.build(u.dit.TextEmbeddings, c.in_text_dim, md, sd=N.sub(W, f"{p}_text_embeddings.")),
                       N.build(u.dit.TextEmbeddings, c.in_text_dim2, Td, sd=N.sub(W, f"{p}_pooled_text_embeddings.")),
                       N.build(u.dit.TimeEmbeddings, md, Td, sd=N.sub(W, f"{p}_time_embeddings.")))
        freqs = tuple(mods[p][2].freqs.float().reshape(1, -1).contiguous() for p in ("video", "audio"))
        inp = N.to_twin(nat, c, freqs)

        def native():
            caps = {}
            for i, p in enumerate(("video", "audio")):
                text, pooled, temb = mods[p]
                te = text(nat["text_embed"])
                caps[f"{p}_text_embeddings"] = N.rows(te)
                pe = pooled(nat["pooled_text_embed"])
                caps[f"{p}_pooled_text_embeddings"] = N.rows(pe)
                t = temb(nat["time"][i])
                caps[f"{p}_time_embeddings"] = N.rows(t)
                caps[f"{p}_time_plus_pooled"] = N.rows(t + pe)
            return caps

        def twin():
            caps, hook = T.recorder()
            d = B.Dit(ctx.k, B.Plan(c, sh), ctx.math, hook=hook, fill=0)
            w = B.bind_weights(c, W)
            d.encode_text(w.video._replace(blocks=[]), "video", inp, inp.time_v, inp.freqs_v, inp.text_rope_v,
                          d.buf(d.p.text_v, L, D), d.buf(d.p.tm_v, 1, Td))
            d.encode_text(w.audio._replace(blocks=[]), "audio", inp, inp.time_a, inp.freqs_a, inp.text_rope_a,
                          d.buf(d.p.text_a, L, Da), d.buf(d.p.tm_a, 1, Td))
            return caps

        return Case({"text": inp.text, "pooled": inp.pooled, "time_v": inp.time_v, "time_a": inp.time_a,
                     "freqs_v": inp.freqs_v, "freqs_a": inp.freqs_a}, native, twin,
                    {"fp32_linear": ctx.math, "path": "block.Dit.encode_text (no text block)"},
                    "TextEmbeddings / TimeEmbeddings modules, time + pooled (dit.py _encode_text)")

    return Site("block/time_chain", "block", {"text": [L, c.in_text_dim], "pooled": [1, c.in_text_dim2],
                                              "time": [1], "plan_bytes": plan}, {"text": "bf16", "time": "f32"},
                wb + plan + 4 * nbytes(L, D) + SLACK, make, "time embedding chain, text / pooled embeddings",
                repeat=False)


def text_block_unit(prefix: str, c: B.Config, sh: B.Shape, wb: int, plan: int) -> Site:
    md = c.model_dim if prefix == "video" else c.model_dim_a
    L, Td = sh.text, c.time_dim

    def make(ctx: Ctx) -> Case:
        W = ctx.weights()
        te, tm = ctx.rnd(L, md), ctx.rnd(1, Td)
        rope = ctx.rope("text_v" if prefix == "video" else "text_a")
        rr, mask = N.rope_rows(rope), ctx.mask()
        enc = N.encoder_block(ctx.tc, W, prefix, 0)

        def native():
            cap = N.Capture(enc, f"{prefix}_text_transformer_blocks.0")
            enc(te, tm, rope, N.mask_native(mask))
            return cap.close()

        def twin():
            caps, hook = T.recorder()
            B.text_block(ctx.k, W, c, sh, 0, prefix, te.clone(), tm, rr, mask, fp32_linear=ctx.math, hook=hook,
                         fill=0)
            return caps

        return Case({"te": te, "tm": tm, "rope": rr, "mask": mask.t}, native, twin,
                    {"fp32_linear": ctx.math, "ffn_rows": B.DEFAULT_FFN_ROWS, "path": "block.text_block"},
                    f"TransformerEncoderBlock [{prefix}_text_transformer_blocks.0] on [{L}, {md}], mask {ctx.valid}")

    return Site(f"block/text_{prefix}", "block", {"te": [L, md], "tm": [1, Td], "rope": [L, 256], "mask": [L],
                                                  "plan_bytes": plan},
                {"te": "bf16", "rope": "f32", "mask": "u8"}, wb + plan + 40 * nbytes(L, md) + SLACK, make,
                repeat=False)


def embeddings_unit(c: B.Config, sh: B.Shape, wb: int, plan: int) -> Site:
    """dit.py _embed_visual / _embed_audio (no token types): VisualEmbeddings, flatten; TextEmbeddings(40, 2048)."""
    D, Da, Nt, A = c.model_dim, c.model_dim_a, sh.tokens(), sh.audio

    def make(ctx: Ctx) -> Case:
        W, u = ctx.weights(), ctx.u
        nat = N.native_inputs(c, sh, rope_modules_built(c, "cpu", DEV), seed=ctx.site_seed, text_valid=ctx.valid)
        inp = N.to_twin(nat, c, (N.time_freqs(D), N.time_freqs(Da)))
        ve = N.build(u.dit.VisualEmbeddings, c.visual_in(), D, (1, 2, 2), sd=N.sub(W, "visual_embeddings."))
        ae = N.build(u.dit.TextEmbeddings, c.in_audio_dim, Da, sd=N.sub(W, "audio_embeddings."))

        def native():
            return {"visual_embeddings": N.rows(ve(nat["x_video"]).flatten(1, 3)),
                    "audio_embeddings": N.rows(ae(nat["x_audio"]))}

        def twin():  # block.Dit.forward's embedding steps
            caps, hook = T.recorder()
            d = B.Dit(ctx.k, B.Plan(c, sh), ctx.math, hook=hook, fill=0)
            w = B.bind_weights(c, W)
            vis, aud = d.buf(d.p.vis, Nt, D), d.buf(d.p.aud, A, Da)
            prow = d.buf(d.p.rows_in, Nt, c.patch_in())
            ctx.k.patchify(inp.x_video, prow, sh, c.visual_in())
            d.linear(w.visual_in, prow, vis)
            d.emit("visual_embeddings", vis)
            audio_in = d.buf(d.p.audio_in, A, c.in_audio_dim)
            ctx.k.cast_bf16(inp.x_audio, audio_in)
            d.text_embedding(w.audio_in, audio_in, aud)
            d.emit("audio_embeddings", aud)
            return caps

        return Case({"x_video": inp.x_video, "x_audio": inp.x_audio}, native, twin,
                    {"path": "block.Dit.forward embeddings"}, "VisualEmbeddings + flatten, audio TextEmbeddings")

    return Site("block/embeddings", "block", {"x_video": [sh.t * sh.h * sh.w, c.visual_in()],
                                              "x_audio": [A, c.in_audio_dim], "plan_bytes": plan},
                {"x_video": "f32", "x_audio": "f32"}, wb + plan + 8 * nbytes(Nt, D) + SLACK, make, repeat=False)


def fused_unit(c: B.Config, sh: B.Shape, wb: int, plan: int) -> Site:
    D, Da, Td, FF = c.model_dim, c.model_dim_a, c.time_dim, c.ff_dim
    Nt, A, L = sh.tokens(), sh.audio, sh.text

    def make(ctx: Ctx) -> Case:
        W = ctx.weights()
        vis0, aud0 = ctx.rnd(Nt, D), ctx.rnd(A, Da)
        text_v, text_a, tm_v, tm_a = ctx.rnd(L, D), ctx.rnd(L, Da), ctx.rnd(1, Td), ctx.rnd(1, Td)
        vrope, arope = ctx.rope("visual"), ctx.rope("audio")
        vr, ar, mask = N.rope_rows(vrope), N.rope_rows(arope), ctx.mask()
        nb = N.fused_block(ctx.tc, W, 0)

        def native():
            cap = N.Capture(nb, "visual_transformer_blocks.0")
            nb(vis0.view(1, Nt, D), aud0.view(1, A, Da), text_v, text_a, (tm_v, tm_a), vrope, arope,
               N.mask_native(mask))
            return cap.close()

        def twin():
            caps, hook = T.recorder()
            io = B.BlockIo(vis0.clone(), aud0.clone(), text_v, text_a, tm_v, tm_a, vr, ar, mask)
            B.forward_block(ctx.k, W, c, sh, 0, io, fp32_linear=ctx.math, hook=hook, fill=0)
            return caps

        return Case({"vis": vis0, "aud": aud0, "text_v": text_v, "text_a": text_a, "tm_v": tm_v, "tm_a": tm_a,
                     "visual_rope": vr, "audio_rope": ar, "mask": mask.t}, native, twin,
                    {"fp32_linear": ctx.math, "ffn_rows": B.DEFAULT_FFN_ROWS, "path": "block.forward_block"},
                    f"FusedTransformerDecoderBlock [visual_transformer_blocks.0] at {Nt} / {A} tokens, "
                    f"mask {ctx.valid}")

    big = nbytes(Nt, D)
    native_tmp = max(4 * nbytes(Nt, D, size=4), 2 * nbytes(Nt, FF)) + 6 * big
    est = wb + plan + native_tmp + 2 * 9 * big + 4 * big + SLACK
    return Site("block/fused", "block", {"vis": [Nt, D], "aud": [A, Da], "text_v": [L, D], "text_a": [L, Da],
                                         "tm": [1, Td], "plan_bytes": plan},
                {"vis": "bf16", "rope": "f32", "mask": "u8"}, est, make, repeat=False)


def block_units() -> list[Site]:
    _tc, c, sh = geometry()
    wb, plan = weight_bytes(c), B.Plan(c, sh).bytes()
    return [time_chain_unit(c, sh, wb, plan), text_block_unit("video", c, sh, wb, plan),
            text_block_unit("audio", c, sh, wb, plan), embeddings_unit(c, sh, wb, plan), fused_unit(c, sh, wb, plan)]


def table(phase: str, valid: int) -> list[Site]:
    """The phase's sites; the input-table receipt leads both phases (fixtures for every run)."""
    _tc, c, sh = geometry()
    s = [tables_site(c, sh)] + (op_sites(valid) if phase == "ops" else block_units())
    names = [x.name for x in s]
    if len(set(names)) != len(names):
        raise SetupError("duplicate site names")
    return s


def select(sites: list[Site], patterns: list[str] | None) -> list[Site]:
    if not patterns:
        return sites
    unknown = [p for p in patterns if not any(fnmatch.fnmatchcase(s.name, p) for s in sites)]
    if unknown:
        raise SetupError(f"no site matches {unknown} (see --list)")
    return [s for s in sites if any(fnmatch.fnmatchcase(s.name, p) for p in patterns)]


# --------------------------------------------------------------------------------------------- comparison, hashing
def compare(nat: torch.Tensor, tw: torch.Tensor) -> dict:
    """Bit comparison in chunks of CHUNK elements (bounded memory): unequal count, max |diff| over the unequal
    elements (inf where one side is not finite), the first mismatch (flat index, row, column, both values)."""
    r = {"numel": int(tw.numel()), "shape_native": list(nat.shape), "shape_twin": list(tw.shape),
         "dtype_native": str(nat.dtype).replace("torch.", ""), "dtype_twin": str(tw.dtype).replace("torch.", "")}
    if tuple(nat.shape) != tuple(tw.shape) or nat.dtype != tw.dtype or nat.dtype not in BITS:
        r.update(equal=False, unequal=-1, max_abs=None, first_mismatch=None, nonfinite=None)
        return r
    a, b = nat.detach().contiguous().reshape(-1), tw.detach().contiguous().reshape(-1)
    ia, ib = a.view(BITS[a.dtype]), b.view(BITS[b.dtype])
    cols = int(tw.shape[-1]) if tw.dim() else 1
    unequal, first, max_abs, nonfinite = 0, None, 0.0, 0
    for s in range(0, a.numel(), CHUNK):
        ne = ia[s:s + CHUNK] != ib[s:s + CHUNK]
        x, y = a[s:s + CHUNK].float(), b[s:s + CHUNK].float()
        nonfinite += int((~torch.isfinite(x)).sum()) + int((~torch.isfinite(y)).sum())
        n = int(ne.sum())
        if n:
            unequal += n
            if first is None:
                i = int(ne.to(torch.uint8).argmax())
                first = {"index": s + i, "row": (s + i) // cols, "col": (s + i) % cols, "native": float(x[i]),
                         "twin": float(y[i])}
            d = torch.nan_to_num((x - y).abs()[ne], nan=float("inf"))
            max_abs = max(max_abs, float(d.max()))
        del ne, x, y
    r.update(equal=unequal == 0, unequal=unequal, max_abs=max_abs, first_mismatch=first, nonfinite=nonfinite)
    return r


def bits_equal(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(bits_equal(a[n], b[n]) for n in a)
    return compare(a, b)["equal"]


def tensor_record(t: torch.Tensor, out: Path, site: str, role: str, force: bool = False,
                  meta: dict | None = None) -> dict:
    """sha256 of the raw little-endian bytes (copied 256 MiB at a time); a fixture .bin + .json when it is small or
    `force`d, its header carrying `meta` (provenance)."""
    t = t.detach().contiguous()
    rec = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "numel": int(t.numel())}
    flat = t.reshape(-1).view(torch.uint8)
    h = hashlib.sha256()
    keep = force or t.numel() <= FIXTURE_MAX
    base = out / "fixtures" / safe(site) / safe(role)
    f = None
    if keep:
        base.parent.mkdir(parents=True, exist_ok=True)
        f = open(base.with_suffix(".bin"), "wb")
    try:
        for s in range(0, flat.numel(), HASH_BYTES):
            chunk = flat[s:s + HASH_BYTES].cpu().numpy()
            h.update(chunk)
            if f is not None:
                f.write(chunk.tobytes())
    finally:
        if f is not None:
            f.close()
    rec["sha256"] = h.hexdigest()
    if keep:
        header = {"site": site, "role": role, "dtype": rec["dtype"], "shape": rec["shape"], "byteorder": "little",
                  "sha256": rec["sha256"], "bin": base.with_suffix(".bin").name, **(meta or {})}
        base.with_suffix(".json").write_text(json.dumps(header, indent=1))
        rec["fixture"] = str(base.with_suffix(".bin").relative_to(out))
    return rec


def compare_captures(nat: dict, tw: dict, out: Path, site: str) -> dict:
    """Every twin capture, in its execution order, against the native capture of the same name."""
    rows, first = [], None
    for name, t in tw.items():
        if name not in nat:
            r = {"name": name, "equal": False, "missing_native": True}
        else:
            r = {"name": name, **compare(nat[name], t), "sha_native": tensor_record(nat[name], out, site,
                                                                                    f"native.{name}")["sha256"],
                 "sha_twin": tensor_record(t, out, site, f"twin.{name}")["sha256"]}
        rows.append(r)
        if first is None and not r["equal"]:
            first = {"capture": name, "index_in_order": len(rows) - 1, "first_mismatch": r.get("first_mismatch"),
                     "unequal": r.get("unequal"), "numel": r.get("numel"), "max_abs": r.get("max_abs")}
    return {"captures": rows, "native_only": [n for n in nat if n not in tw], "first_divergence": first,
            "equal": bool(rows) and first is None, "unequal": sum(max(r.get("unequal", 0) or 0, 0) for r in rows),
            "numel": sum(r.get("numel", 0) for r in rows)}


# ------------------------------------------------------------------------------------------------ profiler, logs
def parse_trace(path: Path) -> tuple[list, list]:
    data = json.loads(path.read_text())
    ev = data.get("traceEvents", []) if isinstance(data, dict) else data
    kern = sorted((e for e in ev if str(e.get("cat", "")).lower() == "kernel"), key=lambda e: e.get("ts", 0))
    ks = [{"name": e.get("name"), "grid": e.get("args", {}).get("grid"), "block": e.get("args", {}).get("block"),
           "shared_memory": e.get("args", {}).get("shared memory"),
           "registers": e.get("args", {}).get("registers per thread"), "dur_us": e.get("dur")} for e in kern]
    cpu = sorted((e for e in ev if e.get("cat") == "cpu_op" and str(e.get("name", "")).startswith("aten::")),
                 key=lambda e: e.get("ts", 0))
    return ks, [e["name"] for e in cpu]


def profiled(fn: Callable[[], object], trace: Path) -> tuple[object, dict]:
    """fn() under the torch profiler (CPU ops and CUDA kernels); its chrome trace written and parsed."""
    try:
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA])
        prof.__enter__()
    except Exception as e:  # no CUPTI: the native call still runs, the receipt says why the kernels are missing
        out = fn()
        torch.cuda.synchronize()
        return out, {"profiler_error": repr(e), "kernels": [], "ops": []}
    try:
        out = fn()
        torch.cuda.synchronize()
    finally:
        prof.__exit__(None, None, None)
    trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace))
    ks, ops = parse_trace(trace)
    return out, {"kernels": ks, "ops": ops}


def sdpa_backend(ops: list, kernels: list) -> str:
    o = " ".join(ops).lower()
    for key, name in (("_cudnn_attention", "cudnn"), ("_flash_attention", "flash"), ("_efficient_attention",
                                                                                     "efficient"),
                      ("attention_math", "math")):
        if key in o:
            return name
    k = " ".join(x["name"] or "" for x in kernels).lower()
    for key, name in (("cudnn", "cudnn"), ("flash_fwd", "flash"), ("fmha", "efficient"), ("attentionkernel",
                                                                                         "efficient"),
                      ("softmax", "math")):
        if key in k:
            return name
    return "unknown"


def backend(kind: str, ops: list, kernels: list) -> str:
    if kind == "attention":
        return sdpa_backend(ops, kernels)
    seen = [o for o in GEMM_OPS if o in ops]
    return ",".join(seen) if seen else "elementwise"


API = re.compile(r"\b(cublas(?:Lt)?[A-Z]\w*)")
LOG_KEEP = re.compile(r"Matmul|GemmEx|Gemm\(|algo|tile|splitK|split_k|stages|reduction|heuristic", re.I)


def summarize_log(path: Path, out: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(out)), "exists": False}
    data = path.read_bytes()
    text = data.decode(errors="replace")
    keep = [ln[:400] for ln in text.splitlines() if LOG_KEEP.search(ln)]
    return {"path": str(path.relative_to(out)), "exists": True, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "api_mentions": dict(Counter(API.findall(text)).most_common(40)),
            "excerpt": keep[:80], "excerpt_total": len(keep)}


# --------------------------------------------------------------------------------------------------- the environment
def _try(fn, default=None):
    try:
        return fn()
    except Exception as e:  # an API absent in this torch: recorded, not fatal
        return default if default is not None else f"unavailable: {e!r}"


def check_env() -> None:
    bad = [k for k in os.environ if k in FORBIDDEN_ENV or k.startswith(FORBIDDEN_PREFIX)]
    if bad:
        raise SetupError(f"environment overrides the native numerics: {sorted(bad)} (unset them)")
    if sys.byteorder != "little":
        raise SetupError("fixtures are little-endian")


def numerics() -> dict:
    m, cud = torch.backends.cuda.matmul, torch.backends.cudnn
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul.fp32_precision": _try(lambda: m.fp32_precision),
        "matmul.allow_tf32": _try(lambda: m.allow_tf32),
        "matmul.allow_bf16_reduced_precision_reduction": _try(lambda: m.allow_bf16_reduced_precision_reduction),
        "matmul.allow_fp16_reduced_precision_reduction": _try(lambda: m.allow_fp16_reduced_precision_reduction),
        "cudnn.allow_tf32": _try(lambda: cud.allow_tf32), "cudnn.enabled": _try(lambda: cud.enabled),
        "cudnn.benchmark": _try(lambda: cud.benchmark), "cudnn.deterministic": _try(lambda: cud.deterministic),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "preferred_blas_library": _try(lambda: str(torch.backends.cuda.preferred_blas_library())),
        "sdp": {"flash": _try(torch.backends.cuda.flash_sdp_enabled),
                "mem_efficient": _try(torch.backends.cuda.mem_efficient_sdp_enabled),
                "math": _try(torch.backends.cuda.math_sdp_enabled),
                "cudnn": _try(torch.backends.cuda.cudnn_sdp_enabled),
                "fp16_bf16_reduction_math_sdp": _try(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed)},
    }


def oracle_numerics() -> dict:
    """The pipeline's policy, and nothing else: set_float32_matmul_precision("high"), asserted (never "highest")."""
    N.pipeline_numerics()
    if torch.get_float32_matmul_precision() != "high":
        raise SetupError(f"float32 matmul precision {torch.get_float32_matmul_precision()!r}, the oracle needs 'high'")
    return numerics()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(HASH_BYTES), b""):
            h.update(b)
    return h.hexdigest()


def fatbin_digests(fatbins: str | None) -> dict:
    d = fatbins or os.environ.get("STK_K6_DIT_FATBINS")
    if not d:
        raise SetupError("pass --fatbins DIR or set STK_K6_DIT_FATBINS")
    out = {}
    for n in K.IMAGES:
        p = Path(d) / f"k6_dit_{n}.fatbin"
        if not p.is_file() or p.stat().st_size == 0:
            raise SetupError(f"missing or empty {p}")
        out[n] = sha256_file(p)
    return out


def git_head(src: Path) -> str | None:
    """The checkout's HEAD commit read from its git directory (no git process)."""
    g = src / ".git"
    try:
        if g.is_file():
            t = g.read_text().strip()
            if t.startswith("gitdir:"):
                g = (src / t[7:].strip()).resolve()
        if not g.is_dir():
            return None
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head[4:].strip()
        common = g
        if (g / "commondir").is_file():
            common = (g / (g / "commondir").read_text().strip()).resolve()
        for base in (g, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()
        packed = common / "packed-refs"
        if packed.is_file():
            for ln in packed.read_text().splitlines():
                parts = ln.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def upstream_receipt() -> dict:
    src = Path(os.environ.get("STK_K6_SRC", "/tmp/kandinsky6-upstream")).resolve()
    if not (src / UPSTREAM_FILES[0]).is_file():
        raise SetupError(f"no kandinsky-6 checkout at {src} (STK_K6_SRC)")
    head = git_head(src)
    if head is not None and head != UPSTREAM_COMMIT:
        raise SetupError(f"upstream checkout at {head}, the oracle is pinned to {UPSTREAM_COMMIT}")
    return {"src": str(src), "commit": head, "pinned": UPSTREAM_COMMIT, "pinned_ok": head == UPSTREAM_COMMIT if head
            else None, "files": {f: sha256_file(src / f) for f in UPSTREAM_FILES if (src / f).is_file()}}


def loaded_libraries() -> dict:
    """The CUDA libraries mapped into this process (after a GEMM has loaded cuBLAS / cuBLASLt): basename and path."""
    keys = ("libcublasLt", "libcublas.", "libcudnn.", "libcudart", "libnvrtc", "libnvJitLink", "libcuda.so")
    found = {}
    for ln in Path("/proc/self/maps").read_text().splitlines():
        p = ln.split()[-1] if "/" in ln else ""
        base = os.path.basename(p)
        for k in keys:
            if base.startswith(k) and k not in found:
                found[k] = p
    return found


def library_versions() -> dict:
    libs = loaded_libraries()
    out = {}
    for k, p in libs.items():
        rec = {"file": os.path.basename(p), "sha256": _try(lambda p=p: sha256_file(Path(p)))}
        lib = _try(lambda p=p: ctypes.CDLL(p), default=False)
        if lib:
            if k == "libcublasLt":
                def lt(lib=lib):
                    lib.cublasLtGetVersion.restype = ctypes.c_size_t
                    return int(lib.cublasLtGetVersion())
                rec["version"] = _try(lt)
            elif k in ("libcublas.", "libcudart"):
                def prop(lib=lib, fn="cublasGetProperty" if k == "libcublas." else None):
                    if fn is None:
                        v = ctypes.c_int()
                        lib.cudaRuntimeGetVersion(ctypes.byref(v))
                        return v.value
                    vals = []
                    for t in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                        v = ctypes.c_int()
                        if getattr(lib, fn)(ctypes.c_int(t), ctypes.byref(v)) != 0:
                            raise RuntimeError(f"{fn} failed")
                        vals.append(v.value)
                    return ".".join(map(str, vals))
                rec["version"] = _try(prop)
        out[k.rstrip(".")] = rec
    return out


def driver_version():
    def get():
        v = ctypes.c_int()
        ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(v))
        return v.value
    return _try(get)


def environment(a, num: dict) -> dict:
    x = torch.randn(64, 64, device=DEV, dtype=BF)
    F.linear(x, x, x[0])  # cuBLASLt (addmm with bias)
    torch.mm(x, x)  # cuBLAS
    torch.mm(x.float(), x.float())
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
        "torch": torch.__version__, "torch_git": _try(lambda: torch.version.git_version),
        "oracle_torch": ORACLE_TORCH, "oracle_torch_ok": torch.__version__.startswith(ORACLE_TORCH),
        "cuda": torch.version.cuda, "cuda_compiled": _try(lambda: torch._C._cuda_getCompiledVersion()),
        "driver": driver_version(), "cudnn": _try(torch.backends.cudnn.version), "libraries": library_versions(),
        "device": {"name": props.name, "capability": [props.major, props.minor],
                   "sm_count": props.multi_processor_count, "total_memory": props.total_memory},
        "numerics": num, "env": {k: v for k, v in sorted(os.environ.items()) if RECORDED_ENV.match(k)},
        "fatbins": fatbin_digests(a.fatbins), "upstream": upstream_receipt(),
        "shape": {**geometry()[2].__dict__, "tokens": geometry()[2].tokens(), "text_valid": a.text_valid},
        "config": geometry()[0], "seed": a.seed, "fp32_linear": a.fp32_linear,
    }


# ----------------------------------------------------------------------------------------------------- the child
def timed(fn: Callable[[], object]) -> tuple[object, float]:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    e.synchronize()
    return out, s.elapsed_time(e)


def run_site(ctx: Ctx, site: Site, out: Path) -> dict:
    rec = {"site": site.name, "kind": site.kind, "covers": site.covers, "shapes": site.shapes, "dtypes": site.dtypes,
           "est_gib": round(site.est / GIB, 3), "seed": ctx.site_seed, "fp32_linear": ctx.math}
    torch.cuda.reset_peak_memory_stats()
    case = site.make(ctx)
    torch.cuda.synchronize()
    nat, prof = profiled(case.native, out / "traces" / f"{safe(site.name)}.json")
    rec.update(native_call=case.native_call, native_kernels=prof["kernels"], native_ops=prof["ops"],
               backend=backend(site.kind, prof["ops"], prof["kernels"]))
    if "profiler_error" in prof:
        rec["profiler_error"] = prof["profiler_error"]
    ms = {}
    if site.repeat:  # native run-to-run: an unrepeatable oracle cannot be matched bit for bit
        again, ms["native"] = timed(case.native)
        rec["native_repeat_equal"] = bits_equal(nat, again)
        del again
    n0 = ctx.k.count
    tw, ms["twin"] = timed(case.twin)
    rec["twin_launches"] = ctx.k.count - n0
    rec["twin_math"] = case.twin_math
    rec["ms"] = ms
    if isinstance(nat, dict):
        rec.update(compare_captures(nat, tw, out, site.name))
    else:
        rec.update(compare(nat, tw))
        rec["sha_native"] = tensor_record(nat, out, site.name, "native")["sha256"]
        rec["sha_twin"] = tensor_record(tw, out, site.name, "twin")["sha256"]
    rec["inputs"] = {n: tensor_record(t, out, site.name, f"input.{n}") for n, t in case.inputs.items()}
    if site.kind == "block":
        h = hashlib.sha256()
        for n, _d, _s in B.manifest(ctx.c):
            h.update(n.encode())
            h.update(tensor_record(ctx.weights()[n], out, site.name, f"weight.{n}")["sha256"].encode())
        rec["weights_sha256"] = h.hexdigest()  # seeded test_ops.random_weights; per-tensor fixtures when small
    rec["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    rec["over_estimate"] = rec["max_memory_allocated"] > site.est
    del nat, tw, case
    torch.cuda.empty_cache()
    return rec


def run_tables(ctx: Ctx, site: Site, out: Path) -> dict:
    """The input tables: the native pair profiled, stored (always) with its provenance, rebuilt once more (run to
    run), then the all-CPU and all-GPU producers compared bit for bit with it (reported, not gated)."""
    rec = {"site": site.name, "kind": site.kind, "covers": site.covers, "shapes": site.shapes, "dtypes": site.dtypes,
           "est_gib": round(site.est / GIB, 3), "gated": False, "build": TABLE_BUILD}
    torch.cuda.reset_peak_memory_stats()
    tabs, prof = profiled(lambda: native_tables(ctx.c, ctx.sh, "cpu", DEV), out / "traces" / f"{safe(site.name)}.json")
    rec.update(native_kernels=prof["kernels"], native_ops=prof["ops"], backend="elementwise")
    rec["native_repeat_equal"] = bits_equal(tabs, native_tables(ctx.c, ctx.sh, "cpu", DEV))
    rec["tables"] = {n: tensor_record(t, out, site.name, f"native.{n}", force=True,
                                      meta={**TABLE_BUILD, "table": n}) for n, t in tabs.items()}
    rec["producers"] = {}
    for label, (args_on, run_on) in (("all_cpu", ("cpu", "cpu")), ("all_gpu", (DEV, DEV))):
        alt = native_tables(ctx.c, ctx.sh, args_on, run_on)
        rows = {}
        for n, t in tabs.items():
            r = compare(t, alt[n].to(DEV))
            rows[n] = {k: r[k] for k in ("equal", "unequal", "numel", "max_abs", "first_mismatch")}
        rec["producers"][label] = {"args_built_on": args_on, "trig_on": run_on, "vs_native": rows,
                                   "equal": all(v["equal"] for v in rows.values())}
        del alt
    rec["equal"] = rec["native_repeat_equal"]  # informational: the row is outside the exit status
    rec["max_memory_allocated"] = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    return rec


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def worker(a) -> int:
    out, name = Path(a.out), a.worker
    path = out / "env.json" if name == "env" else out / "sites" / f"{safe(name)}.json"
    try:
        check_env()
        torch.cuda.init()
        cap = budget.apply(f"gate {name}")  # before the first allocation: the step's Torch budget, when set
        num = oracle_numerics()
        if name == "env":
            write_json(path, {**environment(a, num), "torch_cap": cap})
            return 0
        site = {s.name: s for s in table(a.phase, a.text_valid)}.get(name)
        if site is None:
            raise SetupError(f"unknown site {name}")
        free, _total = torch.cuda.mem_get_info()
        if site.est > a.max_gib * GIB or site.est > free:
            raise SetupError(f"{name}: estimated {site.est / GIB:.1f} GiB, budget {a.max_gib} GiB, free "
                             f"{free / GIB:.1f} GiB")
        ctx = Ctx(a, name)
    except Exception as e:
        write_json(path, {"site": name, "equal": False, "setup_error": repr(e), "traceback": traceback.format_exc()})
        print(f"SETUP {name}: {e!r}", file=sys.stderr, flush=True)
        return 2
    try:
        with torch.no_grad():
            rec = run_tables(ctx, site, out) if site.kind == "tables" else run_site(ctx, site, out)
        torch.cuda.synchronize()
        rec["numerics"] = num
        rec["torch_cap"] = cap
        rec["gated"] = site.gated
        if torch.get_float32_matmul_precision() != "high":
            raise SetupError("the float32 matmul precision changed during the run")
        rc = 0 if rec.get("equal") or not site.gated else 1
    except Exception as e:
        rec = {"site": name, "kind": site.kind, "shapes": site.shapes, "equal": False, "error": repr(e),
               "traceback": traceback.format_exc(), "max_memory_allocated": _try(torch.cuda.max_memory_allocated)}
        print(f"ERROR {name}: {e!r}", file=sys.stderr, flush=True)
        rc = 1
    write_json(path, rec)
    return rc


# ---------------------------------------------------------------------------------------------------- the parent
def spawn(a, name: str, out: Path) -> tuple[int, Path, Path]:
    """One child for `name`, its cuBLAS / cuBLASLt logging set before its interpreter (and CUDA) starts."""
    try:
        budget.host_only("gate parent")  # a node job budgets this process as host-only: it never touches CUDA
    except budget.CapError as e:
        print(f"SETUP {e}", flush=True)
        sys.exit(2)
    lt, cb = out / "logs" / f"{safe(name)}.cublaslt.log", out / "logs" / f"{safe(name)}.cublas.log"
    for p in (lt, cb):
        p.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update({"CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_MASK": "31", "CUBLASLT_LOG_FILE": str(lt),
                "CUBLAS_LOGINFO_DBG": "1", "CUBLAS_LOGDEST_DBG": str(cb)})
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    argv = [sys.executable, "-m", MODULE, "--out", str(out), "--phase", a.phase, "--fp32-linear", a.fp32_linear,
            "--seed", str(a.seed), "--text-valid", str(a.text_valid), "--max-gib", str(a.max_gib), "--worker", name]
    if a.fatbins:
        argv += ["--fatbins", a.fatbins]
    try:
        rc = subprocess.run(argv, env=env, timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        rc = -1
    return rc, lt, cb


def line(rec: dict) -> str:
    ok = rec.get("equal") is True
    peak = (rec.get("max_memory_allocated") or 0) / GIB
    if "captures" in rec:
        fd = rec.get("first_divergence") or {}
        detail = (f"captures={len(rec['captures'])} differ={sum(not c['equal'] for c in rec['captures'])} "
                  f"first={fd.get('capture')}")
    else:
        detail = (f"unequal={rec.get('unequal')} numel={rec.get('numel')} max_abs={rec.get('max_abs')} "
                  f"backend={rec.get('backend')} native_repeat_equal={rec.get('native_repeat_equal')}")
    err = rec.get("setup_error") or rec.get("error")
    if rec.get("gated") is False:
        prod = {k: v.get("equal") for k, v in (rec.get("producers") or {}).items()}
        detail = f"native_repeat_equal={rec.get('native_repeat_equal')} producers_equal={prod} (reported, not gated)"
        tag = "INFO"
    else:
        tag = "PASS" if ok else "FAIL"
    return f"{tag} {rec.get('site')} {detail} peak={peak:.2f}GiB" + (f" error={err}" if err else "")


def first_divergence(phase: str, records: list) -> dict | None:
    for r in records:
        if r.get("equal") is True or r.get("gated") is False:
            continue
        if phase == "blocks" and r.get("first_divergence"):
            return {"unit": r["site"], **r["first_divergence"]}
        return {"site": r["site"], "first_mismatch": r.get("first_mismatch"), "unequal": r.get("unequal"),
                "error": r.get("setup_error") or r.get("error")}
    return None


def parse(argv=None):
    ap = argparse.ArgumentParser(prog=f"python -m {MODULE}", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="receipt directory")
    ap.add_argument("--fatbins", default=None, help="directory of k6_dit_*.fatbin (default STK_K6_DIT_FATBINS)")
    ap.add_argument("--phase", required=True, choices=("ops", "blocks"))
    ap.add_argument("--fp32-linear", required=True, choices=list(K.FP32_LINEAR),
                    help="launch.Fp32Linear of the twin's fp32 linears (no default: none is qualified vs native)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sites", nargs="*", default=None, help="site names or fnmatch patterns (default all)")
    ap.add_argument("--text-valid", type=int, default=TEXT_VALID, help="valid text tokens of 1024 (key mask)")
    ap.add_argument("--max-gib", type=float, default=90.0, help="refuse a site whose peak estimate exceeds this")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds a child")
    ap.add_argument("--list", action="store_true", help="print the phase's sites, shapes and estimates, then exit")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not 1 <= a.text_valid <= B.MAX_TEXT:
        ap.error(f"--text-valid must be in [1, {B.MAX_TEXT}]")
    return a


def main(argv=None) -> int:
    a = parse(argv)
    if a.worker:
        return worker(a)
    try:
        sites = select(table(a.phase, a.text_valid), a.sites)
        if a.list:
            for s in sites:
                print(f"{s.name:44s} {s.est / GIB:6.2f} GiB  {json.dumps(s.shapes)}")
            return 0
        check_env()
        fatbin_digests(a.fatbins)
        upstream_receipt()
        over = [f"{s.name} ({s.est / GIB:.1f} GiB)" for s in sites if s.est > a.max_gib * GIB]
        if over:
            raise SetupError(f"over --max-gib {a.max_gib}: {over}")
    except (SetupError, FileNotFoundError, ValueError) as e:
        print(f"SETUP {e}", flush=True)
        return 2
    out = Path(a.out).resolve()
    for d in ("logs", "sites", "traces", "fixtures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rc, lt, cb = spawn(a, "env", out)
    if rc != 0 or not (out / "env.json").is_file():
        print(f"SETUP environment receipt failed (exit {rc})", flush=True)
        return 2
    env = json.loads((out / "env.json").read_text())
    env["cublaslt_log"], env["cublas_log"] = summarize_log(lt, out), summarize_log(cb, out)
    if not env.get("oracle_torch_ok"):
        print(f"WARN torch {env.get('torch')} is not the oracle's {ORACLE_TORCH}", flush=True)
    records, setup_failed = [], False
    for s in sites:
        t0 = time.monotonic()
        rc, lt, cb = spawn(a, s.name, out)
        p = out / "sites" / f"{safe(s.name)}.json"
        rec = json.loads(p.read_text()) if p.is_file() else {"site": s.name, "kind": s.kind, "shapes": s.shapes,
                                                             "equal": False, "error": f"child exit {rc}"}
        rec.setdefault("gated", s.gated)
        if rc != 0 and rec.get("equal") is True:  # a child that died after writing an equal record is not a pass
            rec["equal"] = False
            rec["error"] = f"child exit {rc}"
        rec["exit"] = rc
        rec["wall_s"] = round(time.monotonic() - t0, 1)
        rec["cublaslt_log"], rec["cublas_log"] = summarize_log(lt, out), summarize_log(cb, out)
        write_json(p, rec)
        records.append(rec)
        setup_failed |= rc == 2
        print(line(rec), flush=True)
    tc, _c, sh = geometry()
    gated = [r for r in records if r.get("gated") is not False]
    ok = bool(records) and all(r.get("equal") is True for r in gated)
    receipt = {"gate": "kandinsky6_dit", "phase": a.phase, "fp32_linear": a.fp32_linear, "seed": a.seed,
               "shape": {**sh.__dict__, "tokens": sh.tokens(), "text_valid": a.text_valid}, "config": tc, "env": env,
               "tables": next((r for r in records if r["site"] == TABLES), None),
               "sites": [r for r in records if r["site"] != TABLES] if a.phase == "ops" else [],
               "blocks": {r["site"]: r for r in records if r["site"] != TABLES} if a.phase == "blocks" else {},
               "first_divergence": first_divergence(a.phase, records), "ok": ok}
    write_json(out / "receipt.json", receipt)
    write_json(out / f"receipt-{a.phase}.json", receipt)
    failed = [r["site"] for r in gated if r.get("equal") is not True]
    print(json.dumps({"ok": ok, "sites": len(records), "failed": failed, "receipt": str(out / "receipt.json")}))
    return 2 if setup_failed else (0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
