"""Upstream kandinsky-6 eac06e39 as the DiT twin's native reference: its own modules (core/components/{dit,piflow_dit,
rope}.py, core/tensors.py, core/algo/prepare_ropes.py, runtime/kernels/sdpa.py) imported from the pinned checkout
(STK_K6_SRC, default /tmp/kandinsky6-upstream) and built as pipeline/factory.py builds the DiT: on the meta device, the
bf16 weights dict assigned (load_state_dict strict, assign), the non-persistent RoPE / time tables materialized on the
CPU (runtime/weights.py materialize_meta_buffers), then moved to the GPU; SDPA at every attention site; the pipeline's
float32 matmul precision "high" (factory.py:162: permits TF32 for the fp32 Modulation and TimeEmbeddings linears;
the executed algorithm is uncaptured).

Helpers build native-format inputs as the pipeline does (RoPE tables from the modules' own args through
pipeline.py _materialize_ropes' visual cache at the 5 s preset canvas, the text / audio tables through compute_rope1d)
and convert inputs, outputs and per-module captures (under block.zig's capture names) to the twin's row views:
rope [S, 1, 64, 2, 2] -> [S, 256] fp32, mask bool -> u8, latents channels-last -> [T * H * W, C], grids -> rows.
The transformer config is the Pro-distill checkpoint's (HF kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers
@ 1018d582, transformer/config.json, copied below; STK_K6_CONFIG=FILE reads another), reducible in blocks and widths.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from . import block as B
from . import kernels as K

# transformer/config.json of the Pro-distill checkpoint (PiFlow heads stored expanded: 160 = 16 x 10, 400 = 40 x 10)
DISTILL = {
    "audio_freqs_scaling": 0.144, "axes_dims": [32, 48, 48], "axes_dims_a": [32, 48, 48], "ca_rope": True,
    "cross_gates": True, "ff_dim": 16384, "ff_dim_a": 7168, "fix_modulation": True, "in_audio_dim": 40,
    "in_text_dim": 3584, "in_text_dim2": 768, "in_visual_dim": 16, "is_multimodal": True, "model_dim": 4096,
    "model_dim_a": 2048, "num_text_blocks": 4, "num_visual_blocks": 60, "out_audio_dim": 400, "out_visual_dim": 160,
    "patch_size": [1, 2, 2], "scale_factor": [1.0, 2.0, 2.0], "text_token_padding": True, "time_dim": 1024,
    "time_dim_a": 1024, "visual_cond": True, "visual_token_type_num_embeddings": 2,
}
N_GRID = 10  # configs/checkpoints/pro-distill.yaml piflow.dx_num_grid_points
CANVAS = (31, 30, 54)  # the 5 s preset's patch grid (latent_frames 31, 480 x 864): the pipeline's visual rope cache
SMALL = {"model_dim": 512, "model_dim_a": 256, "ff_dim": 1024, "ff_dim_a": 512, "time_dim": 256, "time_dim_a": 256,
         "in_text_dim": 384, "in_text_dim2": 96}


# ------------------------------------------------------------------------------------------------------ the checkout
@lru_cache(maxsize=1)
def upstream() -> SimpleNamespace:
    """The pinned modules. `kandinsky/__init__.py` imports the whole pipeline (diffusers, transformers, pydantic) and
    `kandinsky.runtime{,.kernels}` pydantic configs; the three packages are stood in by empty namespace modules on the
    checkout's directories so only the DiT's own files load."""
    src = Path(os.environ.get("STK_K6_SRC", "/tmp/kandinsky6-upstream")).resolve()
    root = src / "kandinsky"
    if not (root / "core" / "components" / "dit.py").is_file():
        raise FileNotFoundError(f"no kandinsky-6 checkout at {src} (STK_K6_SRC)")
    for name in ("kandinsky", "kandinsky.runtime", "kandinsky.runtime.kernels"):
        path = root.joinpath(*name.split(".")[1:])
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod
        elif Path(list(mod.__path__)[0]).resolve() != path:
            raise ImportError(f"{name} is already imported from {list(mod.__path__)[0]}, not the pinned {path}")
    imp = importlib.import_module
    return SimpleNamespace(src=src, dit=imp("kandinsky.core.components.dit"),
                           piflow=imp("kandinsky.core.components.piflow_dit"),
                           rope=imp("kandinsky.core.components.rope"),
                           tensors=imp("kandinsky.core.tensors"), ropes=imp("kandinsky.core.algo.prepare_ropes"),
                           sdpa=imp("kandinsky.runtime.kernels.sdpa"))


def pipeline_numerics() -> None:
    """pipeline/factory.py _build_pipeline: set_float32_matmul_precision("high"), which permits TF32 for the residual
    float32 matmuls (the oracle's own policy; never forced to "highest")."""
    torch.set_float32_matmul_precision("high")


# ----------------------------------------------------------------------------------------------------------- config
def transformer_config(path: str | None = None) -> dict:
    p = path or os.environ.get("STK_K6_CONFIG")
    return json.loads(Path(p).read_text()) if p else dict(DISTILL)


def reduced(tc: dict, visual_blocks: int | None = None, text_blocks: int | None = None, small: bool = False) -> dict:
    """The config with fewer blocks and, when `small`, narrower widths (head_dim 128 and the head widths kept)."""
    tc = dict(tc)
    if visual_blocks is not None:
        tc["num_visual_blocks"] = visual_blocks
    if text_blocks is not None:
        tc["num_text_blocks"] = text_blocks
    if small:
        tc.update(SMALL)
    return tc


def twin_config(tc: dict, n_grid: int = N_GRID) -> B.Config:
    """layout.Config of a transformer config: one time width, one head width, the fused-block flags all set."""
    if (tc["time_dim_a"] != tc["time_dim"] or list(tc["axes_dims_a"]) != list(tc["axes_dims"])
            or sum(tc["axes_dims"]) != 128):
        raise ValueError("the engine has one time width and head dim 128")
    flags = ("ca_rope", "cross_gates", "fix_modulation", "text_token_padding", "is_multimodal", "visual_cond")
    if not all(tc[f] for f in flags):
        raise ValueError("block.zig implements ca_rope, cross_gates, fix_modulation, text_token_padding only")
    return B.Config(model_dim=tc["model_dim"], model_dim_a=tc["model_dim_a"], time_dim=tc["time_dim"],
                    ff_dim=tc["ff_dim"], ff_dim_a=tc["ff_dim_a"], head_dim=128, axes_dims=tuple(tc["axes_dims"]),
                    in_text_dim=tc["in_text_dim"],
                    in_text_dim2=tc["in_text_dim2"], in_visual_dim=tc["in_visual_dim"], in_audio_dim=tc["in_audio_dim"],
                    n_grid=n_grid, out_visual_dim=tc["out_visual_dim"], out_audio_dim=tc["out_audio_dim"],
                    num_text_blocks=tc["num_text_blocks"], num_visual_blocks=tc["num_visual_blocks"],
                    token_types=tc["visual_token_type_num_embeddings"], audio_freqs_scaling=tc["audio_freqs_scaling"],
                    scale_factor=tuple(tc["scale_factor"]))


def dit_kwargs(tc: dict, n_grid: int = N_GRID) -> dict:
    """factory.py's PiFlowDiffusionTransformer3D kwargs: per-grid head widths (config.py _per_grid_overrides; the
    audio head from in_audio_dim) and attention engine "sdpa"."""
    if tc["out_visual_dim"] % n_grid != 0:
        raise ValueError("out_visual_dim is not n_grid x the per-grid width")
    keys = ("in_visual_dim", "in_text_dim", "in_text_dim2", "time_dim", "model_dim", "ff_dim", "num_text_blocks",
            "num_visual_blocks", "visual_cond", "is_multimodal", "in_audio_dim", "model_dim_a", "time_dim_a",
            "ff_dim_a",
            "audio_freqs_scaling", "text_token_padding", "ca_rope", "cross_gates", "fix_modulation",
            "visual_token_type_num_embeddings")
    kw = {k: tc[k] for k in keys}
    kw.update(patch_size=tuple(tc["patch_size"]), axes_dims=tuple(tc["axes_dims"]),
              axes_dims_a=tuple(tc["axes_dims_a"]),
              attention_engine="sdpa", n_grid=n_grid, out_visual_dim=tc["out_visual_dim"] // n_grid,
              out_audio_dim=tc["in_audio_dim"])
    return kw


# --------------------------------------------------------------------------------------------------------- building
def materialize_meta_buffers(module: nn.Module, device: torch.device) -> None:
    """runtime/weights.py materialize_meta_buffers (that module imports huggingface_hub, so its body is here)."""
    for mod in module.modules():
        touched = False
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                mod._buffers[name] = torch.empty(buf.shape, dtype=buf.dtype, device=device)
                touched = True
        if touched and hasattr(mod, "reset_parameters"):
            mod.reset_parameters()


def build(cls, *args, sd: dict, device: str = "cuda", **kw) -> nn.Module:
    """factory.py's recipe for one upstream module: meta construct, assign the weights, CPU tables, to the GPU."""
    with torch.device("meta"):
        m = cls(*args, **kw)
    m.load_state_dict(sd, strict=True, assign=True)
    materialize_meta_buffers(m, torch.device("cpu"))
    return m.to(device).eval().requires_grad_(False)


def sub(weights: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}


def fused_block(tc: dict, weights: dict, i: int = 0) -> nn.Module:
    """visual_transformer_blocks.i: FusedTransformerDecoderBlock(4096, 1024, 16384, 128, 2048, 1024, 7168, 128,
    engine="sdpa", text_token_padding, ca_rope, cross_gates, fix_modulation) at the config's widths."""
    u = upstream()
    return build(u.dit.FusedTransformerDecoderBlock, tc["model_dim"], tc["time_dim"], tc["ff_dim"],
                 sum(tc["axes_dims"]),
                 tc["model_dim_a"], tc["time_dim_a"], tc["ff_dim_a"], sum(tc["axes_dims_a"]), engine="sdpa",
                 text_token_padding=True, ca_rope=True, cross_gates=True, fix_modulation=True,
                 sd=sub(weights, f"visual_transformer_blocks.{i}."))


def encoder_block(tc: dict, weights: dict, prefix: str, i: int = 0) -> nn.Module:
    """{prefix}_text_transformer_blocks.i: TransformerEncoderBlock(md, td, fd, hd, "sdpa", text_token_padding)."""
    u = upstream()
    md, td, fd, hd = ((tc["model_dim"], tc["time_dim"], tc["ff_dim"], sum(tc["axes_dims"])) if prefix == "video" else
                      (tc["model_dim_a"], tc["time_dim_a"], tc["ff_dim_a"], sum(tc["axes_dims_a"])))
    return build(u.dit.TransformerEncoderBlock, md, td, fd, hd, "sdpa", tc["text_token_padding"],
                 sd=sub(weights, f"{prefix}_text_transformer_blocks.{i}."))


def piflow_dit(tc: dict, weights: dict, n_grid: int = N_GRID) -> nn.Module:
    """The whole PiFlowDiffusionTransformer3D (factory.py build_dit with piflow enabled)."""
    u = upstream()
    return build(u.piflow.PiFlowDiffusionTransformer3D, sd=weights, **dit_kwargs(tc, n_grid))


class _Probe(nn.Module):
    """Stands in for a Linear: the identity; `weight` only carries the dtype upstream casts to."""

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False)

    def forward(self, x):
        return x


class _Zeros(nn.Module):
    """Stands in for a Modulation: zero shift and scale; apply_scale_shift_norm then passes bf16 rows unchanged."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, t):
        return torch.zeros(t.shape[0], self.n, dtype=t.dtype, device=t.device)


def visual_embeddings_layout(visual_dim: int) -> nn.Module:
    """VisualEmbeddings with the in_layer an identity: upstream's patchify and bf16 cast alone (dit.py 92-113)."""
    with torch.device("meta"):
        ve = upstream().dit.VisualEmbeddings(visual_dim, 8, (1, 2, 2))
    ve.in_layer = _Probe(torch.bfloat16)
    return ve


def out_layer_layout(width: int, visual_dim: int) -> nn.Module:
    """OutLayer with zero modulation, identity norm and out_layer: upstream's unpatchify alone (dit.py 293-321); call
    it with bf16 rows [1, T, H', W', width] and any [1, n] time embedding."""
    with torch.device("meta"):
        ol = upstream().dit.OutLayer(width, 8, visual_dim, (1, 2, 2))
    ol.modulation = _Zeros(2 * width)
    ol.norm = nn.Identity()
    ol.out_layer = nn.Identity()
    return ol


def grid_split(x: torch.Tensor, n_grid: int, dx: int) -> torch.Tensor:
    """PiFlowDiffusionTransformer3D.forward's split (piflow_dit.py 180-184): view(.., n_grid, dx).movedim(-2, 1)."""
    return x.view(*x.shape[:-1], n_grid, dx).movedim(-2, 1)


# ---------------------------------------------------------------------------------------------------------- inputs
def rope_modules(c: B.Config, device: str = "cuda") -> dict:
    """The DiT's RoPE modules as they come out of build(): args on the CPU, then moved (dit.py 595, 615, 627)."""
    u = upstream()
    hd = sum(c.axes_dims)
    mods = {"visual": u.rope.RoPE3D(tuple(c.axes_dims)),
            "audio": u.rope.RoPE1D(hd, freqs_scaling=c.audio_freqs_scaling),
            "video_text": u.rope.RoPE1D(hd), "audio_text": u.rope.RoPE1D(hd)}
    return {k: m.to(device) for k, m in mods.items()}


def rope_modules_of(dit: nn.Module) -> dict:
    return {"visual": dit.visual_rope, "audio": dit.audio_rope, "video_text": dit.video_text_rope,
            "audio_text": dit.audio_text_rope}


def ropes(mods: dict, c: B.Config, sh: B.Shape, canvas: tuple = CANVAS) -> dict:
    """pipeline.py _materialize_ropes: the visual table cut from the max-canvas cache, 1-D tables from 0."""
    u = upstream()
    scale = tuple(float(s) for s in c.scale_factor)
    tp, hp, wp = sh.t, sh.h // 2, sh.w // 2
    cache = u.ropes.VisualRopeCache()
    cache.get(mods["visual"], (max(tp, canvas[0]), max(hp, canvas[1]), max(wp, canvas[2])), scale)
    return {"visual": cache.get(mods["visual"], (tp, hp, wp), scale),
            "audio": u.ropes.compute_rope1d(mods["audio"], sh.audio),
            "text": [u.ropes.compute_rope1d(mods["video_text"], sh.text),
                     u.ropes.compute_rope1d(mods["audio_text"], sh.text)]}


def time_freqs(md: int, device: str = "cuda") -> torch.Tensor:
    """TimeEmbeddings.freqs as reset_parameters makes it on the CPU, moved: [1, md / 2] fp32."""
    return upstream().tensors.get_freqs(md // 2, 10000.0).to(device).reshape(1, -1)


def freqs_of(dit: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return (dit.video_time_embeddings.freqs.float().reshape(1, -1).contiguous(),
            dit.audio_time_embeddings.freqs.float().reshape(1, -1).contiguous())


def native_inputs(c: B.Config, sh: B.Shape, mods: dict, seed: int = 0, text_valid: int | None = None,
                  x_dtype: torch.dtype = torch.float32, sigma: float = 0.7, device: str = "cuda") -> dict:
    """One distill DiT call's kwargs (piflow_sampler.py): T2V latents (cond and mask channels zero), random text
    states, the time sigma * 1000 for both modalities, the pipeline's rope tables, the bool key-padding mask."""
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)

    x_video = torch.zeros(1, sh.t, sh.h, sh.w, c.visual_in(), device=device, dtype=torch.float32)
    x_video[..., :c.in_visual_dim] = rnd(1, sh.t, sh.h, sh.w, c.in_visual_dim)
    valid = sh.text if text_valid is None else text_valid
    r = ropes(mods, c, sh)
    t = torch.tensor([sigma * 1000.0], dtype=torch.float32, device=device)
    return {
        "x_video": x_video.to(x_dtype),
        "x_audio": rnd(1, sh.audio, c.in_audio_dim, dtype=x_dtype),
        "text_embed": rnd(sh.text, c.in_text_dim, dtype=torch.bfloat16),
        "pooled_text_embed": rnd(1, c.in_text_dim2, dtype=torch.bfloat16),
        "time": [t.clone(), t.clone()],
        "visual_rope": r["visual"],
        "audio_rope": r["audio"],
        "text_rope": r["text"],
        "attention_mask": torch.arange(sh.text, device=device) < valid,
        "visual_token_type_ids": None,
    }


def rope_rows(rope: torch.Tensor) -> torch.Tensor:
    """[..., 1, 64, 2, 2] -> [S, 256] fp32 rows (the singleton head axis dropped, (t, h', w') flattened)."""
    return rope.float().reshape(-1, 4 * K.ROPE_PAIRS).contiguous()


def rope_native(rows: torch.Tensor) -> torch.Tensor:
    return rows.view(-1, 1, K.ROPE_PAIRS, 2, 2)


def mask_native(mask: K.Mask | None) -> torch.Tensor | None:
    """The twin's u8 mask as the DiT passes it to attention ([1, 1, 1, K] bool, dit.py _normalize_attn_mask)."""
    return None if mask is None else mask.t.bool().view(1, 1, 1, -1)


def to_twin(nat: dict, c: B.Config, freqs: tuple[torch.Tensor, torch.Tensor]) -> B.Inputs:
    """native_inputs' kwargs as block.zig Inputs."""
    return B.Inputs(
        x_video=nat["x_video"].reshape(-1, c.visual_in()).contiguous(),
        x_audio=nat["x_audio"].reshape(-1, c.in_audio_dim).contiguous(),
        text=nat["text_embed"].to(torch.bfloat16).contiguous(),
        pooled=nat["pooled_text_embed"].to(torch.bfloat16).reshape(1, -1).contiguous(),
        time_v=nat["time"][0].float().reshape(1, 1).contiguous(),
        time_a=nat["time"][1].float().reshape(1, 1).contiguous(),
        freqs_v=freqs[0], freqs_a=freqs[1],
        visual_rope=rope_rows(nat["visual_rope"]), audio_rope=rope_rows(nat["audio_rope"]),
        text_rope_v=rope_rows(nat["text_rope"][0]), text_rope_a=rope_rows(nat["text_rope"][1]),
        mask=None if nat.get("attention_mask") is None else K.Mask.of(nat["attention_mask"]))


def twin_outputs(video: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The PiFlow outputs [1, G, T, H, W, dx] / [1, G, A, dx] as the twin's contiguous grid-major rows."""
    return video.contiguous().reshape(-1, video.shape[-1]), audio.contiguous().reshape(-1, audio.shape[-1])


# -------------------------------------------------------------------------------------------------------- captures
def rows(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1, t.shape[-1]).clone()


_TXT = re.compile(r"^(video|audio)_text_transformer_blocks\.(\d+)(?:\.(text_modulation|attn|feed_forward))?$")
_DEC = re.compile(r"^visual_transformer_blocks\.(\d+)\.(video|audio)_dec_block\."
                  r"(visual_modulation|self_attention|cross_attention|feed_forward|cross_attention_norm)$")
_FUS = re.compile(r"^visual_transformer_blocks\.(\d+)(?:\.(va_modulation|av_modulation|va_cross_attention|"
                  r"av_cross_attention|va_normalization))?$")
_TOP = {f"{p}_{s}" for p in ("video", "audio")
        for s in ("text_embeddings", "pooled_text_embeddings", "time_embeddings")}
_TOP |= {"visual_embeddings", "audio_embeddings"}
_HEAD = {"out_layer.out_layer": "out_layer", "audio_out_layer.out_layer": "audio_out_layer"}


def _rules(path: str) -> list[tuple[str, str]]:
    """(capture name, "out" | "in") for a module at its full upstream path: its output, or the first argument it is
    called with (the residual a norm is called on as `x.float()`, the pre-CA tensor a cross-attention gets)."""
    if path in _TOP:
        return [(path, "out")]
    if path in _HEAD:
        return [(_HEAD[path], "out")]
    m = _TXT.match(path)
    if m:
        p, i, s = m.groups()
        if s is None:
            return [(path, "out")]
        r = [(path, "out")]
        if s == "text_modulation" and i == "0":
            r.append((f"{p}_time_plus_pooled", "in"))
        return r
    m = _DEC.match(path)
    if m:
        i, md, s = m.groups()
        base = f"visual_transformer_blocks.{i}.{md}_dec_block."
        if s == "cross_attention_norm":
            return [(base + "self_attention_residual", "in")] if md == "video" else []
        r = [(base + s, "out")]
        if s == "cross_attention":
            r.append((base + "pre_ca", "in"))
        return r
    m = _FUS.match(path)
    if m:
        i, s = m.groups()
        base = f"visual_transformer_blocks.{i}"
        if s is None:
            return [(base, "out")]
        if s == "va_normalization":
            return [(base + ".video_dec_block.cross_attention_residual", "in")]
        return [(f"{base}.{s}", "out")]
    return []


class Capture:
    """Forward hooks on an upstream module tree (`base`: the root's path in the DiT, "" for the DiT itself) that
    record the activations block.zig emits, by its names, as bf16 rows. close() removes them."""

    def __init__(self, root: nn.Module, base: str = ""):
        self.out: dict[str, torch.Tensor] = {}
        self.handles = []
        for path, mod in root.named_modules():
            full = f"{base}.{path}" if base and path else (base or path)
            for name, kind in _rules(full):
                if kind == "in":
                    self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
                elif _FUS.match(full) and _FUS.match(full).group(2) is None:
                    self.handles.append(mod.register_forward_hook(self._pair(name)))
                else:
                    self.handles.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(_m, args):
            self.out[name] = rows(args[0]).to(torch.bfloat16)  # a bf16 tensor, or its exact x.float()
        return hook

    def _post(self, name):
        def hook(_m, _args, out):
            self.out[name] = rows(out)
        return hook

    def _pair(self, name):
        def hook(_m, _args, out):
            self.out[f"{name}.video"] = rows(out[0])
            self.out[f"{name}.audio"] = rows(out[1])
        return hook

    def close(self) -> dict:
        for h in self.handles:
            h.remove()
        self.handles = []
        return self.out
