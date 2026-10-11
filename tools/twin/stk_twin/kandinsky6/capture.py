"""python -I -m stk_twin.kandinsky6.capture --model DIR --src KANDINSKY_SRC --stages noise,rope,piflow,clip,text,vae,dit0,e2e
       --seed 42 --prompt-file F --out DIR [--attention sdpa] [--sdpa-backends default] [--compile none]
       [--e2e-prompt 0] [--vae-latent CAPTURE_DIR] [--oracle-prefix /opt/k6-oracle]

The native reference ("oracle") of LocalRouter's Kandinsky-6 Pro-distill runtime, recorded in the twin's capture format
(stk_twin/rec.py: ops.jsonl + content-addressed blobs/; op names in README.md). Upstream kandinsky-6 is imported at
run time from the pinned source tree --src (eac06e39, refused otherwise; never vendored) and built as
kandinsky/pipeline/factory.py _build_pipeline builds it from configs/checkpoints/pro-distill.yaml, with the snapshot
--model bound as absolute paths (no download) and these explicit, recorded choices: attention engine --attention (not
"auto"), torch SDPA priority --sdpa-backends ("default" pins torch's own order explicitly), compile --compile (none:
eager), beautifier none, no offload, no NF4, no SR, no cache. set_float32_matmul_precision("high") as factory.py:162.

Refuses to run when TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, NVIDIA_TF32_OVERRIDE or LD_PRELOAD is set, or the interpreter is
not the pinned oracle closure: CPython 3.13.16, torch 2.14.1+cu130, CUDA 13, NumPy 2.5.4. With --oracle-prefix (the
kit passes /opt/k6-oracle) sys.prefix must be that prefix and its MANIFEST.sha256 must have the pinned sha256. The
`oracle` note records the prefix pins (archive, manifest, freeze, metadata-correction receipt; a corrected install,
not official metadata) and every capture is labelled unqualified. Runs with CUBLASLT_LOG_LEVEL=5 (cublaslt.log) and cuBLAS API logging (cublas.log) in --out.

Stages (run in this order):
  noise   k6.noise.video/audio: the CUDA generator draws of prepare_latents.py (seed, seed+1), plus the device note
  rope    k6.freqs.*: TimeEmbeddings freqs and RoPE args buffers (meta build + materialize_meta_buffers as the factory);
          k6.rope.*: the tables pipeline.py _materialize_ropes builds for the preset canvas, as [rows, 256] f32
  piflow  k6.piflow.step{i}: upstream piflow_denoise_loop on the noise with synthetic bf16 grid predictions (a CUDA
          generator seeded seed+1000+i) in place of the DiT: per outer step the state in/out and sigma/tau/segment
  clip    k6.clip.pooled per prompt (the CLIP part of Kandinsky6TextEmbedder.encode)
  text    k6.qwen.embeds per prompt (Kandinsky6TextEmbedder.encode itself; ids and mask from a forward pre-hook)
  vae     k6.vae.decode: postprocess_video on the noise video (or --vae-latent's final latent), with the decode plan
  dit0    k6.dit.step0: the DiT inputs and outputs of outer step 0 of the real pipeline (a forward hook)
  e2e     the full generate: k6.dit.step{i} each step, k6.vae.decode, the audio chain
          k6.audio.latent_in/mel/wave/pcm16 (chained=true, each in = the previous out), k6.e2e.frames, the mp4
A stage whose snapshot files are missing is skipped and recorded as `skipped: missing <file>` (nothing is fabricated).
Exit 0 when every stage ran, 2 when some were skipped, 1 when one failed. One JSON line.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

from . import AUX, ORACLE, PIN_COMMIT, QUALIFICATION, read_capture, sha256_file

STAGES = ("noise", "rope", "piflow", "clip", "text", "vae", "dit0", "e2e")
REFUSED_ENV = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE", "LD_PRELOAD")
PRED_SEED = 1000  # the piflow stage's synthetic grid predictions: CUDA generator seeded seed + PRED_SEED + step
AUDIO_DOWNSAMPLE = 1024  # mmaudio FeaturesUtils.downsample_factor, a constant there; the pipeline reads it off the VAE
CONFIG = ("kandinsky", "configs", "checkpoints", "pro-distill.yaml")

CLIP_FILES = ("text_encoder_2/config.json", "text_encoder_2/model.safetensors", "tokenizer_2/tokenizer.json",
              "tokenizer_2/tokenizer_config.json")
QWEN_FILES = ("text_encoder/config.json", "text_encoder/generation_config.json",
              "text_encoder/model.safetensors.index.json",
              *(f"text_encoder/model-0000{i}-of-00005.safetensors" for i in range(1, 6)),
              "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json", "text_encoder/vocab.json",
              "text_encoder/merges.txt", "text_encoder/preprocessor_config.json", "text_encoder/chat_template.json")
VAE_FILES = ("vae/config.json", "vae/diffusion_pytorch_model.safetensors")
AUDIO_FILES = ("audio_vae/config.json", "audio_vae/diffusion_pytorch_model.safetensors", "vocoder/config.json",
               "vocoder/diffusion_pytorch_model.safetensors")
DIT_FILES = ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors")
ALL_FILES = DIT_FILES + QWEN_FILES + CLIP_FILES + VAE_FILES + AUDIO_FILES
NEEDS = {"noise": (), "rope": (), "piflow": (), "clip": CLIP_FILES, "text": QWEN_FILES + CLIP_FILES,
         "vae": VAE_FILES, "dit0": ALL_FILES, "e2e": ALL_FILES}

REC = None  # stk_twin.rec.REC, imported after the cuBLAS logging environment is set (rec.py imports torch)


class StopAfterStep0(Exception):
    """Raised from the DiT hook when only dit0 is wanted: the pipeline stops after the first outer step."""


# --------------------------------------------------------------------------------------------------- the upstream tree
def source_commit(src: Path) -> str | None:
    """The tree's commit: `.k6-source-commit` (written next to a `git archive` export) or the checkout's .git files."""
    marker = src / ".k6-source-commit"
    if marker.is_file():
        return marker.read_text().strip()
    git = src / ".git"
    if not (git / "HEAD").is_file():
        return None
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    if (git / ref).is_file():
        return (git / ref).read_text().strip()
    if (git / "packed-refs").is_file():
        for line in (git / "packed-refs").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def tree_digest(root: Path) -> str:
    """sha256 over (path, sha256) of every file under `root`: what was imported, whatever its commit claims."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(f"{p.relative_to(root).as_posix()}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


def stub_sr() -> str:
    """pipeline/config.py imports kandinsky_sr.pipeline.config.SRConfig for its `sr` section. The oracle venv leaves the
    SR package out; a frozen, field-less stand-in keeps that section at its empty default (SR is never run)."""
    try:
        importlib.import_module("kandinsky_sr.pipeline.config")
        return "kandinsky_sr installed"
    except ImportError:
        pass
    from pydantic import BaseModel, ConfigDict

    class SRConfig(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

    for name in ("kandinsky_sr", "kandinsky_sr.pipeline", "kandinsky_sr.pipeline.config"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
    sys.modules["kandinsky_sr"].pipeline = sys.modules["kandinsky_sr.pipeline"]
    sys.modules["kandinsky_sr.pipeline"].config = sys.modules["kandinsky_sr.pipeline.config"]
    sys.modules["kandinsky_sr.pipeline.config"].SRConfig = SRConfig
    return "stand-in: kandinsky_sr.pipeline.config.SRConfig (frozen, no fields; SR off)"


def upstream(src: Path) -> SimpleNamespace:
    commit = source_commit(src)
    if commit != PIN_COMMIT:
        raise SystemExit(f"capture: {src} is at {commit or 'an unknown commit'}, not the pinned {PIN_COMMIT}")
    if "kandinsky" in sys.modules:
        raise SystemExit("capture: a kandinsky package is already imported")
    sr = stub_sr()
    sys.path.insert(0, str(src))
    imp = importlib.import_module
    pkg = imp("kandinsky")
    if Path(pkg.__file__).resolve().parent != (src / "kandinsky").resolve():
        raise SystemExit(f"capture: kandinsky imported from {pkg.__file__}, not {src}")
    return SimpleNamespace(
        src=src, commit=commit, sr=sr, tree=tree_digest(src / "kandinsky"),
        factory=imp("kandinsky.pipeline.factory"), config=imp("kandinsky.pipeline.config"),
        pipeline=imp("kandinsky.pipeline.pipeline"), latents=imp("kandinsky.core.algo.prepare_latents"),
        ropes=imp("kandinsky.core.algo.prepare_ropes"), piflow=imp("kandinsky.core.algo.piflow_sampler"),
        pmath=imp("kandinsky.core.algo.piflow_math"), post_video=imp("kandinsky.core.algo.postprocess_video"),
        dit=imp("kandinsky.core.components.dit"), rope=imp("kandinsky.core.components.rope"),
        te=imp("kandinsky.core.components.text_embedder"), vae_video=imp("kandinsky.core.components.vae_video"),
        weights=imp("kandinsky.runtime.weights"), types=imp("kandinsky.core.types"))


def pipeline_config(K, model: Path, attention: str, compile_: str):
    """configs/checkpoints/pro-distill.yaml through load_config, component paths bound to the snapshot (absolute, so
    factory.bind_checkpoint_paths never downloads) and every runtime policy set explicitly."""
    C = K.config
    cfg = C.load_config(K.src.joinpath(*CONFIG))
    m = str(model)
    upd = {
        "paths": cfg.paths.model_copy(update={"dit": f"{m}/transformer/diffusion_pytorch_model.safetensors",
                                              "vae": f"{m}/vae", "qwen": f"{m}/text_encoder",
                                              "clip": f"{m}/text_encoder_2", "dit_export": None}),
        "audio_vae": cfg.audio_vae.model_copy(update={"tod_vae_ckpt": f"{m}/audio_vae"}),
        "vocoder": cfg.vocoder.model_copy(update={"ckpt": f"{m}/vocoder"}),
        "beautifier": cfg.beautifier.model_copy(update={"name": "none", "model_path": None}),
        "attention": cfg.attention.model_copy(update={"engine": attention}),
        "compile": cfg.compile.model_copy(update={"strategy": compile_}),
        "cache": cfg.cache.model_copy(update={"mode": "none"}),
        "offload": cfg.offload.model_copy(update={"strategy": "none"}),
        "profile": cfg.profile.model_copy(update={"mode": "none"}),
        "text_embedder": cfg.text_embedder.model_copy(update={"quantized_qwen": False}),
    }
    cfg = C.PipelineConfig.model_validate(cfg.model_copy(update=upd).model_dump())
    if not (cfg.piflow.enabled and cfg.dit.is_multimodal and cfg.dit.text_token_padding):
        raise SystemExit("capture: expected the Pro-distill config (piflow, multimodal, padded text)")
    return cfg


def geometry(K, cfg) -> SimpleNamespace:
    """The preset call's shapes as pipeline.py __call__ derives them (no size, time or frame arguments)."""
    g, d = cfg.generation, cfg.dit
    params = inspect.signature(K.pipeline.Kandinsky6Pipeline.__init__).parameters
    fps, afps = float(params["fps"].default), int(params["audio_fps"].default)
    T, Hl, Wl = g.latent_frames, g.height // 8, g.width // 8
    pT, pH, pW = d.patch_size
    A = K.latents.audio_latent_duration(T, fps=fps, audio_fps=afps, downsample_factor=AUDIO_DOWNSAMPLE)
    return SimpleNamespace(T=T, Hl=Hl, Wl=Wl, C=d.in_visual_dim, A=A, Ca=d.in_audio_dim, rope=(T // pT, Hl // pH, Wl // pW),
                           text_len=cfg.text_embedder.max_length, fps=fps, afps=afps)


# ---------------------------------------------------------------------------------------------------------- helpers
def rope_rows(t):
    """[..., 1, D/2, 2, 2] rotary tensor -> [rows, D/2 * 4]: per frequency (cos, -sin, sin, cos), fp32 as computed."""
    if tuple(t.shape[-2:]) != (2, 2) or t.shape[-4] != 1:
        raise ValueError(f"not a rotary tensor: {tuple(t.shape)}")
    return t.reshape(-1, t.shape[-3] * 4)


def fval(t) -> float:
    return float(t.reshape(-1)[0].item())


def noise(ctx):
    """prepare_video_latents / prepare_audio_latents exactly as the pipeline calls them (bs 1)."""
    if "noise" not in ctx.state:
        L, G = ctx.K.latents, ctx.geo
        b = L.prepare_video_latents(bs=1, duration=G.T, H_lat=G.Hl, W_lat=G.Wl, C=G.C, seed=ctx.seed, device=ctx.dev)
        ctx.state["noise"] = L.prepare_audio_latents(b, audio_duration=G.A, audio_dim=G.Ca, seed=ctx.seed,
                                                     device=ctx.dev)
    return ctx.state["noise"]


def check(ctx, name: str, ok: bool, **data) -> None:
    ctx.checks[name] = bool(ok)
    REC.note("check", check=name, ok=bool(ok), **data)


def decode_plan(vae, free: int | None) -> dict:
    """The tiling the Hunyuan VAE decode chose (hunyuan_vae.py get_dec_optimal_tiling -> apply_tiling; it depends on the
    free CUDA memory, recorded as note_free_bytes)."""
    keys = ("tile_size", "use_tiling", "use_slicing", "use_framewise_decoding", "tile_sample_min_num_frames",
            "tile_sample_stride_num_frames", "tile_sample_min_height", "tile_sample_min_width",
            "tile_sample_stride_height", "tile_sample_stride_width", "spatial_compression_ratio",
            "temporal_compression_ratio")
    plan = {}
    for k in keys:
        v = getattr(vae, k, None)
        plan[k] = list(v) if isinstance(v, tuple) else v
    plan["scaling_factor"] = float(vae.config.scaling_factor)
    plan["dtype"] = str(next(vae.parameters()).dtype).removeprefix("torch.")
    plan["note_free_bytes"] = free
    return plan


def decode_op(ctx, vae, bundle, postprocess, bs: int = 1):
    """postprocess_video (upstream's own function) recorded as k6.vae.decode; vae.decode is wrapped only to record
    its fp16 input and the free memory the tiling plan reads."""
    torch = ctx.torch
    seen = {}
    orig = vae.decode

    def decode(z, *a, **kw):
        seen["free"] = int(torch.cuda.mem_get_info(z.device)[0])
        seen["z"] = z
        return orig(z, *a, **kw)

    vae.decode = decode
    attrs = {"bs": bs}
    try:
        with REC.op("k6.vae.decode", "vae", attrs, latent=bundle.video) as o:
            frames = postprocess(bundle, vae, bs=bs)
            attrs.update(decode_plan(vae, seen.get("free")))
            o.out(frames=frames)
    finally:
        del vae.decode
    with REC.op(AUX + "vae.decoder_in", "vae", {}) as o:
        o.out(z=seen["z"])
    return frames


def free_cuda(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------- stages
def stage_noise(ctx) -> None:
    b, G = noise(ctx), ctx.geo
    src = "kandinsky/core/algo/prepare_latents.py: torch.Generator(device).manual_seed(s); torch.randn(..., bfloat16)"
    with REC.op("k6.noise.video", "noise", {"seed": ctx.seed, "note_source": src}) as o:
        o.out(x=b.video)
    with REC.op("k6.noise.audio", "noise", {"seed": ctx.seed + 1, "note_source": src}) as o:
        o.out(x=b.audio)
    check(ctx, "noise.shapes", list(b.video.shape) == [G.T, G.Hl, G.Wl, G.C] and list(b.audio.shape) == [G.A, G.Ca],
          video=list(b.video.shape), audio=list(b.audio.shape))


def stage_rope(ctx) -> None:
    torch, K, cfg, G = ctx.torch, ctx.K, ctx.cfg, ctx.geo
    d, g = cfg.dit, cfg.generation
    axes_a = tuple(d.axes_dims_a or d.axes_dims)
    with torch.device("meta"):  # factory.create_bare_dit builds the DiT on meta; these are its table holders
        h = torch.nn.Module()
        h.visual_rope = K.rope.RoPE3D(tuple(d.axes_dims))
        h.video_text_rope = K.rope.RoPE1D(sum(d.axes_dims))
        h.audio_text_rope = K.rope.RoPE1D(sum(axes_a))
        h.audio_rope = K.rope.RoPE1D(sum(axes_a), freqs_scaling=d.audio_freqs_scaling)
        h.video_time_embeddings = K.dit.TimeEmbeddings(d.model_dim, d.time_dim)
        h.audio_time_embeddings = K.dit.TimeEmbeddings(d.model_dim_a or d.model_dim, d.time_dim_a or d.time_dim)
    K.weights.materialize_meta_buffers(h, torch.device("cpu"))  # as create_bare_dit, then .to(device)
    tables = {"time_v": ("video_time_embeddings.freqs", h.video_time_embeddings.freqs),
              "time_a": ("audio_time_embeddings.freqs", h.audio_time_embeddings.freqs),
              "rope_t": ("visual_rope.args_0", h.visual_rope.args_0),
              "rope_h": ("visual_rope.args_1", h.visual_rope.args_1),
              "rope_w": ("visual_rope.args_2", h.visual_rope.args_2),
              "rope_text": ("video_text_rope.args", h.video_text_rope.args),
              "rope_audio": ("audio_rope.args", h.audio_rope.args)}
    for k, (buf, f) in tables.items():
        with REC.op(f"k6.freqs.{k}", "freqs", {"buffer": buf}) as o:
            o.out(f=f.to(ctx.dev))
    with REC.op(AUX + "freqs.rope_text_a", "freqs", {"buffer": "audio_text_rope.args"}) as o:
        o.out(f=h.audio_text_rope.args.to(ctx.dev))
    for name in ("visual_rope", "video_text_rope", "audio_text_rope", "audio_rope"):
        getattr(h, name).to(ctx.dev)
    h.patch_size, h.is_multimodal = tuple(d.patch_size), d.is_multimodal
    pipe = SimpleNamespace(raw_dit=h, scale_factor=tuple(g.scale_factor), latent_frames=g.latent_frames, height=g.height,
                           width=g.width, _visual_rope_cache=K.ropes.VisualRopeCache())
    vis, aud, txt, _ = K.pipeline.Kandinsky6Pipeline._materialize_ropes(
        pipe, T_rope=G.rope[0], H_patches=G.rope[1], W_patches=G.rope[2], text_len=G.text_len, null_text_len=0,
        audio_len=G.A)
    rows = {"visual": vis, "audio": aud, "text_v": txt[0], "text_a": txt[1]}
    ctx.state["rope"] = {}
    for k, t in rows.items():
        table = rope_rows(t)
        ctx.state["rope"][k] = table
        with REC.op(f"k6.rope.{k}", "rope", {"source_shape": list(t.shape)}) as o:
            o.out(table=table)


def schedule(ctx) -> list[dict]:
    """piflow_sampler.py piflow_denoise_loop's outer schedule (and policy_rollout_fm's substep count), on the device in
    float32 as the loop computes it; `time` is what the loop hands the DiT (sigma * 1000)."""
    torch, P, n = ctx.torch, ctx.cfg.piflow, ctx.cfg.generation.num_steps
    fss = max(float(P.final_step_size_scale), P.eps)
    base = 1.0 / (float(n) - (1.0 - fss))
    raw = torch.ones(1, device=ctx.dev, dtype=torch.float32)
    out = []
    for k in range(n):
        seg = base * (fss if k == n - 1 else 1.0)
        tau_dst = (raw - seg).clamp(min=P.eps)
        sigma = ctx.K.pmath.shift_timesteps(raw, P.shift)
        sub = ((raw - tau_dst) * P.num_policy_substeps).round().to(torch.long).clamp(min=1)
        out.append({"sigma": fval(sigma), "tau_src": fval(raw), "tau_dst": fval(tau_dst),
                    "segment": fval(torch.full((1,), seg, device=ctx.dev, dtype=torch.float32)),
                    "substeps": int(sub.item()), "time": fval(sigma * 1000)})
        raw = tau_dst
    return out


class GridOracle:
    """Stands in for the DiT inside upstream piflow_denoise_loop: returns synthetic bf16 grid predictions in the layout
    PiFlowDiffusionTransformer3D returns ([B, n_grid, *state]) and records each outer step as k6.piflow.step{i},
    opened at call i (state in, predictions) and closed with call i+1's state (or the loop's result)."""

    is_multimodal = True

    def __init__(self, ctx, sched: list[dict], visual_cond: bool):
        self.ctx, self.sched, self.visual_cond = ctx, sched, visual_cond
        self.k, self.open = 0, None

    def __call__(self, *, x_video, x_audio, time, **_):
        torch, C = self.ctx.torch, self.ctx.geo.C
        state_v = x_video[..., :C] if x_video.shape[-1] != C else x_video
        self.close(state_v, x_audio)
        k, n = self.k, self.ctx.cfg.piflow.dx_num_grid_points
        self.k += 1
        g = torch.Generator(device=self.ctx.dev)
        g.manual_seed(self.ctx.seed + PRED_SEED + k)
        pv = torch.randn(state_v.shape[0], n, *state_v.shape[1:], generator=g, device=self.ctx.dev, dtype=torch.bfloat16)
        pa = torch.randn(x_audio.shape[0], n, *x_audio.shape[1:], generator=g, device=self.ctx.dev, dtype=torch.bfloat16)
        s = self.sched[k]
        t = [fval(time[0]), fval(time[1])]
        check(self.ctx, f"piflow.step{k}.time", t == [s["time"], s["time"]], loop=t, schedule=s["time"])
        attrs = {"step": k, **{key: s[key] for key in ("sigma", "tau_src", "tau_dst", "segment", "substeps")},
                 "time": t, "note_pred_seed": self.ctx.seed + PRED_SEED + k}
        cm = REC.op(f"k6.piflow.step{k}", "piflow", attrs, state_v=state_v, state_a=x_audio, pred_v=pv, pred_a=pa)
        self.open = (cm, cm.__enter__())
        return pv, pa

    def close(self, state_v, state_a) -> None:
        if self.open is not None:
            cm, o = self.open
            o.out(state_v=state_v, state_a=state_a)
            cm.__exit__(None, None, None)
            self.open = None


def stage_piflow(ctx) -> None:
    torch, K, cfg, G = ctx.torch, ctx.K, ctx.cfg, ctx.geo
    b, P, g = noise(ctx), cfg.piflow, cfg.generation
    bundle = K.types.LatentBundle(video=b.video.reshape(1, G.T, G.Hl, G.Wl, G.C), audio=b.audio.reshape(1, G.A, G.Ca),
                                  video_cu_seqlens=b.video_cu_seqlens, audio_cu_seqlens=b.audio_cu_seqlens)
    grid = GridOracle(ctx, schedule(ctx), cfg.dit.visual_cond)
    empty = torch.empty(0, device=ctx.dev)
    res = K.piflow.piflow_denoise_loop(  # the arguments pipeline.py passes for the preset call
        bundle=bundle, dit=grid, text_embeds={"text_embeds": empty, "pooled_embed": empty}, visual_rope=None,
        audio_rope=None, text_rope=None, num_steps=g.num_steps, scheduler_scale=g.scheduler_scale, first_frames=None,
        visual_cond_scheme=g.visual_cond_scheme, sample_audio=True, attention_mask=None, visual_token_type_ids=None,
        scale_factor=tuple(g.scale_factor), eps=P.eps, final_step_size_scale=P.final_step_size_scale, shift=P.shift,
        num_policy_substeps=P.num_policy_substeps)
    grid.close(res.video, res.audio)
    check(ctx, "piflow.steps", grid.k == g.num_steps, steps=grid.k)


def stage_clip(ctx) -> None:
    TE, path = ctx.K.te, ctx.cfg.paths.clip
    model = TE._load_clip_text_model(path).to(ctx.dev).eval()  # as Kandinsky6TextEmbedder.__init__
    tok = TE.CLIPTokenizer.from_pretrained(TE.clip_tokenizer_dir(path))
    ctx.state["clip"] = {}
    for i, p in enumerate(ctx.prompts):
        # text_embedder.py Kandinsky6TextEmbedder.encode, its CLIP section, for one caption
        inputs = tok([p], max_length=77, truncation=True, add_special_tokens=True, padding="max_length",
                     return_tensors="pt").to(model.device)
        pooled = model(**inputs)["pooler_output"]
        ctx.state["clip"][i] = pooled
        with REC.op("k6.clip.pooled", "clip", {"prompt": i}, ids=inputs["input_ids"]) as o:
            o.out(pooled=pooled)
        with REC.op(AUX + "clip.mask", "clip", {"prompt": i}) as o:
            o.out(mask=inputs["attention_mask"])


def stage_text(ctx) -> None:
    torch, TE, cfg = ctx.torch, ctx.K.te, ctx.cfg
    te = TE.Kandinsky6TextEmbedder(qwen_path=cfg.paths.qwen, clip_path=cfg.paths.clip,
                                   max_length=cfg.text_embedder.max_length, device=ctx.dev,
                                   quantized_qwen=cfg.text_embedder.quantized_qwen,
                                   text_token_padding=cfg.dit.text_token_padding)  # factory.py, offload none
    seen = {}
    hooks = [te.qwen.register_forward_pre_hook(lambda m, a, kw: seen.__setitem__("qwen", kw), with_kwargs=True),
             te.clip.register_forward_pre_hook(lambda m, a, kw: seen.__setitem__("clip", kw), with_kwargs=True)]
    ctx.state["text"] = {}
    try:
        for i, p in enumerate(ctx.prompts):
            emb, cu, mask = te.encode([p])
            q, c = seen.pop("qwen"), seen.pop("clip")
            ctx.state["text"][i] = emb["text_embeds"]
            with REC.op("k6.qwen.embeds", "qwen", {"prompt": i, "crop_start": TE._CROP_START,
                                                    "max_length": te.max_length},
                        ids=q["input_ids"], mask=q["attention_mask"]) as o:
                o.out(embeds=emb["text_embeds"])
            extra = {"cu_seqlens": cu, "pooled": emb["pooled_embed"], "clip_ids": c["input_ids"]}
            if mask is not None:
                extra["mask"] = mask
            with REC.op(AUX + "qwen.extra", "qwen", {"prompt": i}) as o:
                o.out(**extra)
            if i in ctx.state.get("clip", {}):
                check(ctx, f"text.p{i}.pooled_equals_clip_stage", torch.equal(emb["pooled_embed"], ctx.state["clip"][i]))
    finally:
        for h in hooks:
            h.remove()


def stage_vae(ctx) -> None:
    torch, K, G = ctx.torch, ctx.K, ctx.geo
    vae = K.vae_video.build_vae(ctx.cfg.paths.vae, device=ctx.dev)
    if ctx.args.vae_latent:
        root = Path(ctx.args.vae_latent)
        ops, _ = read_capture(root)
        refs = [op["outs"]["video"] for op in ops if op["name"] == AUX + "e2e.latents"]
        if len(refs) != 1:
            raise ValueError(f"{root}: expected one {AUX}e2e.latents op, found {len(refs)}")
        lat = load_ref(torch, root, refs[0]).to(ctx.dev)
        REC.note("vae_latent", capture=str(root), sha=refs[0]["sha"])
    else:
        lat = noise(ctx).video.reshape(1, G.T, G.Hl, G.Wl, G.C)
    bundle = K.types.LatentBundle(video=lat, audio=None, video_cu_seqlens=None, audio_cu_seqlens=None)
    decode_op(ctx, vae, bundle, K.post_video.postprocess_video)


def load_ref(torch, root: Path, ref: dict):
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "uint8": torch.uint8,
          "int16": torch.int16, "int32": torch.int32, "int64": torch.int64}[ref["dtype"]]
    data = bytearray((root / "blobs" / ref["sha"]).read_bytes())
    return torch.frombuffer(data, dtype=torch.uint8).view(dt).reshape(ref["shape"])


def attention_fns(raw) -> list[str]:
    names = set()
    for m in raw.modules():
        slot = getattr(m, "attn", None)
        fn = getattr(slot, "_fn", None)
        if fn is not None:
            names.add(f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}")
    return sorted(names)


def stage_pipeline(ctx, dit0: bool, e2e: bool) -> None:
    """The real pipeline (factory._build_pipeline) and one generate. A forward hook on the bare DiT records the outer
    steps; postprocess_video/postprocess_audio stay upstream's, wrapped in the pipeline module only to record."""
    torch, K, cfg, G = ctx.torch, ctx.K, ctx.cfg, ctx.geo
    pipe = K.factory._build_pipeline(cfg, ctx.dev, ctx.attention, cache_mode="none", offload_strategy="none")
    raw = pipe.raw_dit
    REC.note("pipeline", dit=type(raw).__name__, cache_mode=getattr(pipe.dit, "mode", None),
             beautifier=pipe.beautifier.name, offload=type(pipe.offload).__name__, attention=attention_fns(raw),
             compiled=any("OptimizedModule" in type(m).__name__ for m in raw.modules()),
             dit_dtype=str(next(raw.parameters()).dtype).removeprefix("torch."))
    steps, audio = [], {}
    prompt = ctx.prompts[ctx.args.e2e_prompt]

    def dit_hook(mod, args, kw, output):
        k = len(steps)
        steps.append(k)
        if args:
            raise RuntimeError("the pipeline passed positional DiT arguments")
        t = torch.cat([kw["time"][0].reshape(-1), kw["time"][1].reshape(-1)])
        with REC.op(f"k6.dit.step{k}", "dit", {"step": k}, x_video=kw["x_video"], x_audio=kw["x_audio"], time=t) as o:
            o.out(video=output[0], audio=output[1])
        if k == 0:
            cond = {"text_embed": kw["text_embed"], "pooled_text_embed": kw["pooled_text_embed"],
                    "attention_mask": kw["attention_mask"], "rope_visual": rope_rows(kw["visual_rope"]),
                    "rope_audio": rope_rows(kw["audio_rope"]), "rope_text_v": rope_rows(kw["text_rope"][0]),
                    "rope_text_a": rope_rows(kw["text_rope"][1])}
            with REC.op(AUX + "dit.cond", "dit", {"none": sorted(k2 for k2, v in kw.items() if v is None)}) as o:
                o.out(**{k2: v for k2, v in cond.items() if v is not None})
            for name, key in (("visual", "rope_visual"), ("audio", "rope_audio"), ("text_v", "rope_text_v"),
                              ("text_a", "rope_text_a")):
                if name in ctx.state.get("rope", {}):
                    check(ctx, f"dit0.rope_{name}_equals_rope_stage", torch.equal(cond[key], ctx.state["rope"][name]))
            if ctx.args.e2e_prompt in ctx.state.get("text", {}):
                check(ctx, "dit0.text_embed_equals_text_stage",
                      torch.equal(kw["text_embed"], ctx.state["text"][ctx.args.e2e_prompt]))
            if not e2e:
                raise StopAfterStep0

    P = K.pipeline
    orig_pv = P.postprocess_video

    def postprocess_video(bundle, vae, bs):
        with REC.op(AUX + "e2e.latents", "latents", {}) as o:
            o.out(video=bundle.video, audio=bundle.audio)
        return decode_op(ctx, vae, bundle, orig_pv, bs)

    # the audio chain, recorded as handed over: postprocess_audio's latent, the audio VAE decode's mel, the vocoder's
    # input and output (each kept once, as a copy, and recorded after the generate in pipeline order)
    orig_pa = P.postprocess_audio
    avae, orig_wd = pipe.audio_vae, pipe.audio_vae.wrapped_decode

    def postprocess_audio(bundle, audio_vae, vocoder, *args, **kw):
        if "latent_in" in audio:
            raise RuntimeError("postprocess_audio ran twice")
        audio["latent_in"] = bundle.audio.detach().clone()
        return orig_pa(bundle, audio_vae, vocoder, *args, **kw)

    def wrapped_decode(latents, *args, **kw):
        mel = orig_wd(latents, *args, **kw)
        if "mel" in audio:
            raise RuntimeError("the audio VAE decoded twice")
        audio["mel"] = mel.detach().clone()
        return mel

    def vocoder_hook(mod, args, out):
        if "wave" in audio:
            raise RuntimeError("the vocoder ran twice")
        audio["voc_in"], audio["wave"] = args[0].detach().clone(), out.detach().clone()

    hooks = [raw.register_forward_hook(dit_hook, with_kwargs=True), pipe.vocoder.register_forward_hook(vocoder_hook)]
    P.postprocess_video, P.postprocess_audio = postprocess_video, postprocess_audio
    avae.wrapped_decode = wrapped_decode
    save = ctx.out / "e2e" / "output.mp4" if e2e else None
    try:
        out = pipe(text=prompt, seed=ctx.seed, save_path=None if save is None else str(save))
    except StopAfterStep0:
        out = None
    finally:
        P.postprocess_video, P.postprocess_audio = orig_pv, orig_pa
        del avae.wrapped_decode  # back to the class method
        for h in hooks:
            h.remove()
    check(ctx, "pipeline.steps", len(steps) == (cfg.generation.num_steps if e2e else 1), steps=len(steps))
    if out is None:
        return
    missing = [k for k in ("latent_in", "mel", "wave") if k not in audio]
    if missing:
        raise RuntimeError(f"the audio chain was not observed: {missing}")
    # the vocoder must have been fed the VAE's mel itself, so mel.out and wave.in are one blob
    check(ctx, "audio.vocoder_in_equals_vae_mel", torch.equal(audio["voc_in"], audio["mel"]))
    pcm = torch.from_numpy(out.audio[0].copy())
    with REC.op("k6.audio.latent_in", "audio", {"chained": True}) as o:
        o.out(latent=audio["latent_in"])
    with REC.op("k6.audio.mel", "audio", {"chained": True}, latent=audio["latent_in"]) as o:
        o.out(mel=audio["mel"])
    with REC.op("k6.audio.wave", "audio", {"chained": True, "sample_rate": G.afps}, mel=audio["voc_in"]) as o:
        o.out(wave=audio["wave"])
    with REC.op("k6.audio.pcm16", "audio", {"chained": True, "sample_rate": G.afps, "normalization": "clip"},
                wave=audio["wave"]) as o:
        o.out(pcm=pcm)
    with REC.op("k6.e2e.frames", "e2e", {"fps": G.fps, "prompt": ctx.args.e2e_prompt}) as o:
        o.out(frames=out.frames)
    mp4 = Path(out.path)
    REC.note("e2e_mp4", file=str(mp4.relative_to(ctx.out)), sha256=sha256_file(mp4), bytes=mp4.stat().st_size,
             prompt_used=out.prompts[0])


# ------------------------------------------------------------------------------------------------- environment notes
def redacted_env() -> dict:
    secret = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|(^|_)KEY($|_)", re.I)
    return {k: ("<redacted>" if secret.search(k) else v) for k, v in sorted(os.environ.items())}


def device_note(torch) -> dict:
    p = torch.cuda.get_device_properties(0)
    dev = {}
    for k in dir(p):
        if not k.startswith("_"):
            v = getattr(p, k, None)
            if isinstance(v, (int, float, str, bool)):
                dev[k] = v
    dev["uuid"] = str(getattr(p, "uuid", ""))
    dev["multiProcessorCount"] = p.multi_processor_count
    dev["maxThreadsPerMultiProcessor"] = getattr(p, "max_threads_per_multi_processor", None)
    dev["capability"] = list(torch.cuda.get_device_capability(0))
    return dev


def precision_note(torch) -> dict:
    b = torch.backends
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda.matmul.allow_tf32": b.cuda.matmul.allow_tf32, "cudnn.allow_tf32": b.cudnn.allow_tf32,
            "cuda.matmul.allow_bf16_reduced_precision_reduction":
                b.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cuda.matmul.allow_fp16_reduced_precision_reduction":
                b.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "cudnn.benchmark": b.cudnn.benchmark, "cudnn.deterministic": b.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "sdp.flash": b.cuda.flash_sdp_enabled(), "sdp.mem_efficient": b.cuda.mem_efficient_sdp_enabled(),
            "sdp.cudnn": b.cuda.cudnn_sdp_enabled(), "sdp.math": b.cuda.math_sdp_enabled()}


def sdpa_priority(torch, spec: str):
    """torch.nn.attention.sdpa_kernel(..., set_priority=True) over an explicit backend order. "default" reads torch's own
    built-in order and pins it as the explicit list (refused when torch does not expose it)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    names = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
             "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}
    if spec == "default":
        get = getattr(torch._C, "_get_sdp_priority_order", None)
        if get is None:
            raise SystemExit("capture: torch does not expose its SDPA priority order; pass --sdpa-backends explicitly")
        by_int = {int(v): k for k, v in names.items()}
        order = [by_int[int(i)] for i in get() if int(i) in by_int]
    else:
        order = [s.strip() for s in spec.split(",") if s.strip()]
        if not order or any(s not in names for s in order):
            raise SystemExit(f"capture: --sdpa-backends takes default or a list of {sorted(names)}")
    return order, sdpa_kernel([names[s] for s in order], set_priority=True)


def loaded_libs() -> list[str]:
    keys = ("libcublas", "libcudnn", "libcuda.so", "libcudart", "libnvrtc", "libnvJitLink", "libcufft", "libcurand",
            "libcusparse", "libnccl")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and any(k in parts[5] for k in keys):
            paths.add(parts[5].strip())
    return sorted(paths)


def versions_note(torch) -> dict:
    v = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
         "torch.version.cuda": torch.version.cuda, "torch.git": torch.version.git_version,
         "cudnn": torch.backends.cudnn.version()}
    for m in ("transformers", "diffusers", "tokenizers", "safetensors", "accelerate", "huggingface_hub", "numpy",
              "pydantic", "av", "librosa", "triton"):
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception as e:  # recorded, never fatal
            v[m] = f"unavailable: {type(e).__name__}"
    libs = {}
    for p in loaded_libs():
        name = Path(p).name
        entry = {"bytes": Path(p).stat().st_size}
        if name.startswith(("libcublas", "libcuda.so", "libcudnn.so")):
            entry["sha256"] = sha256_file(Path(p))
        try:
            if name.startswith("libcublasLt.so"):
                f = ctypes.CDLL(p).cublasLtGetVersion
                f.restype = ctypes.c_size_t
                entry["version"] = int(f())
            elif name.startswith("libcublas.so"):
                lib, parts = ctypes.CDLL(p), []
                for prop in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                    x = ctypes.c_int()
                    lib.cublasGetProperty(prop, ctypes.byref(x))
                    parts.append(x.value)
                entry["version"] = ".".join(map(str, parts))
            elif name.startswith("libcuda.so"):
                x = ctypes.c_int()
                ctypes.CDLL(p).cuDriverGetVersion(ctypes.byref(x))
                entry["driver_api_version"] = x.value
        except Exception as e:
            entry["version"] = f"unavailable: {type(e).__name__}"
        libs[p] = entry
    v["libs"] = libs
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        v["nvidia-smi"] = q.stdout.strip() or q.stderr.strip()
    return v


def oracle_interpreter(prefix: str) -> dict:
    """The interpreter is the pinned closure (asserted before torch is imported); with `prefix`, it runs from that
    staged prefix and the prefix's MANIFEST.sha256 is the pinned one (k6-job.sh has sha256sum -c'd it, read only)."""
    py = platform.python_version()
    if py != ORACLE["python"]:
        raise SystemExit(f"capture: CPython {py}, the oracle closure is {ORACLE['python']}")
    info = {"sys.prefix": sys.prefix, "sys.executable": sys.executable, "python": sys.version,
            "implementation": platform.python_implementation()}
    if prefix:
        want = Path(prefix).resolve()
        if Path(sys.prefix).resolve() != want:
            raise SystemExit(f"capture: sys.prefix {sys.prefix}, not the oracle prefix {want}")
        man = want / "MANIFEST.sha256"
        got = sha256_file(man) if man.is_file() else "missing"
        if got != ORACLE["manifest_sha256"]:
            raise SystemExit(f"capture: {man} sha256 {got}, pinned {ORACLE['manifest_sha256']}")
        info["manifest_sha256_observed"] = got
    return info


def freeze_sha256() -> tuple[str, int]:
    """sha256 of `pip freeze` as recomputed from installed metadata (name==version, case-insensitive order, pip
    left out, trailing newline). Informational next to the pinned freeze sha: the gate on the prefix is its manifest."""
    from importlib import metadata

    rows = {}
    for d in metadata.distributions():
        name = d.metadata["Name"]
        if name and name.lower() != "pip":
            rows[name.lower()] = f"{name}=={d.version}"
    text = "".join(rows[k] + "\n" for k in sorted(rows))
    return hashlib.sha256(text.encode()).hexdigest(), len(rows)


def oracle_note(torch, interp: dict) -> dict:
    import numpy

    if torch.__version__ != ORACLE["torch"]:
        raise SystemExit(f"capture: torch {torch.__version__}, the oracle needs {ORACLE['torch']}")
    cuda = torch.version.cuda or ""
    if cuda.split(".")[0] != ORACLE["cuda_major"]:
        raise SystemExit(f"capture: torch built for CUDA {cuda or 'none'}, the oracle needs CUDA {ORACLE['cuda_major']}")
    if numpy.__version__ != ORACLE["numpy"]:
        raise SystemExit(f"capture: NumPy {numpy.__version__}, the oracle needs {ORACLE['numpy']}")
    fsha, n = freeze_sha256()
    return {"pins": ORACLE, "qualification": QUALIFICATION, "interpreter": interp,
            "observed": {"python": platform.python_version(), "torch": torch.__version__, "torch.version.cuda": cuda,
                         "numpy": numpy.__version__},
            "freeze_sha256_recomputed": fsha, "freeze_packages": n,
            "freeze_matches_pin": fsha == ORACLE["freeze_sha256"]}


# ------------------------------------------------------------------------------------------------------------- main
def probe_imports(src: Path, prefix: str) -> None:
    """The capture's own upstream import (`upstream`), nothing else: what an oracle closure must provide. CPU only; it
    proves importability, never GPU behaviour or precision. `prefix` is reported (its MANIFEST.sha256, and whether that
    is the pinned one), so a candidate closure can be probed before it is pinned."""
    bad = [v for v in REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"capture: refusing to run with {', '.join(bad)} set")
    interp = oracle_interpreter("")  # a candidate closure's manifest is reported, not required to be the pinned one
    if prefix:
        man = Path(prefix).resolve() / "MANIFEST.sha256"
        interp["manifest_sha256"] = sha256_file(man) if man.is_file() else "missing"
        interp["pinned_manifest"] = interp["manifest_sha256"] == ORACLE["manifest_sha256"]
    before = set(sys.modules)
    try:
        K = upstream(src)
    except ImportError as e:
        print(f"probe: FAIL importing upstream {PIN_COMMIT[:8]}: missing module {e.name!r} ({e})")
        raise SystemExit(1) from None
    loaded = sorted({m.split(".")[0] for m in set(sys.modules) - before} - {"kandinsky", "kandinsky_sr"})
    versions = {}
    for m in loaded:
        try:
            versions[m] = importlib.metadata.version(m)
        except importlib.metadata.PackageNotFoundError:
            pass
    print(json.dumps({"probe": "pass", "qualification": QUALIFICATION, "interpreter": interp, "upstream": K.commit,
                      "tree_sha256": K.tree, "third_party": versions}, indent=1, default=str))


def main() -> None:
    global REC
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", help="the HF snapshot (Kandinsky-6.0-Pro-distill-5s-Diffusers), read only")
    ap.add_argument("--src", required=True, help=f"the upstream kandinsky-6 tree at {PIN_COMMIT[:8]}")
    ap.add_argument("--stages", help=",".join(STAGES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-file", help="one prompt per line; # comments and blank lines ignored")
    ap.add_argument("--out", help="a new capture directory (refused when it exists)")
    ap.add_argument("--attention", default="sdpa", choices=("sdpa", "flash_attention_2", "flash_attention_3", "sage"))
    ap.add_argument("--sdpa-backends", default="default", help="default, or an order such as flash,efficient,math")
    ap.add_argument("--compile", default="none", choices=("none", "torch"))
    ap.add_argument("--e2e-prompt", type=int, default=0, help="the prompt line dit0/e2e generate")
    ap.add_argument("--vae-latent", default="", help="decode this capture's final latent instead of the noise")
    ap.add_argument("--oracle-prefix", default="", help=f"require running from this staged prefix ({ORACLE['prefix']})")
    ap.add_argument("--probe-imports", action="store_true",
                    help="import upstream exactly as a capture does (CPU only, no model, no output), print the\n"
                         "third-party versions it loaded and exit; a missing module exits 1")
    a = ap.parse_args()
    if a.probe_imports:
        return probe_imports(Path(a.src).absolute(), a.oracle_prefix)
    missing = [f"--{n.replace('_', '-')}" for n in ("model", "stages", "prompt_file", "out") if getattr(a, n) is None]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")

    wanted = {s.strip() for s in a.stages.split(",") if s.strip()}
    if not wanted or wanted - set(STAGES):
        raise SystemExit(f"capture: unknown stages {sorted(wanted - set(STAGES))}; known {list(STAGES)}")
    stages = [s for s in STAGES if s in wanted]
    bad = [v for v in REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"capture: refusing to run with {', '.join(bad)} set")
    out, model, src = Path(a.out).absolute(), Path(a.model).absolute(), Path(a.src).absolute()
    if out.exists():
        raise SystemExit(f"capture: {out} exists; a capture never overwrites")
    pf = Path(a.prompt_file)
    prompts = [ln.strip() for ln in pf.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not prompts or not 0 <= a.e2e_prompt < len(prompts):
        raise SystemExit(f"capture: {pf} has {len(prompts)} prompts; --e2e-prompt {a.e2e_prompt}")
    if not model.is_dir():
        raise SystemExit(f"capture: no snapshot directory {model}")
    interp = oracle_interpreter(a.oracle_prefix)

    os.environ.update(CUBLASLT_LOG_LEVEL="5", CUBLASLT_LOG_FILE=str(out / "cublaslt.log"), CUBLAS_LOGINFO_DBG="1",
                      CUBLAS_LOGDEST_DBG=str(out / "cublas.log"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch

    oracle = oracle_note(torch, interp)
    if not torch.cuda.is_available():
        raise SystemExit("capture: no CUDA device")
    from ..rec import REC as rec

    REC = rec
    K = upstream(src)
    torch.set_float32_matmul_precision("high")  # factory.py _build_pipeline (TF32 for residual fp32 matmuls)
    cfg = pipeline_config(K, model, a.attention, a.compile)
    order, sdpa = sdpa_priority(torch, a.sdpa_backends)
    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    ctx = SimpleNamespace(torch=torch, K=K, cfg=cfg, geo=geometry(K, cfg), dev=torch.device("cuda"), seed=a.seed,
                          prompts=prompts, args=a, out=out, attention=a.attention, state={}, checks={})
    hf_dit = model / "transformer" / "config.json"
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, stages=stages, seed=a.seed, prompts=prompts, prompt_file=str(pf),
             prompt_file_sha256=sha256_file(pf), model=str(model), e2e_prompt=a.e2e_prompt, vae_latent=a.vae_latent,
             pred_seed_offset=PRED_SEED, geometry=vars(ctx.geo) | {"rope": list(ctx.geo.rope)})
    REC.note("upstream", src=str(src), commit=K.commit, tree_sha256=K.tree, kandinsky_sr=K.sr,
             config_file="/".join(CONFIG), config=cfg.model_dump(mode="json"),
             hf_transformer_config=json.loads(hf_dit.read_text()) if hf_dit.is_file() else None)
    REC.note("runtime", attention_engine=a.attention, sdpa_backends=order, sdpa_spec=a.sdpa_backends,
             compile=a.compile, beautifier="none", offload="none", cache="none", quantized_qwen=False, sr="off")
    REC.note("device", **device_note(torch))
    REC.note("precision", **precision_note(torch))
    REC.note("env", env=redacted_env())

    fns = {"noise": stage_noise, "rope": stage_rope, "piflow": stage_piflow, "clip": stage_clip, "text": stage_text,
           "vae": stage_vae, "dit0": lambda c: stage_pipeline(c, True, False),
           "e2e": lambda c: stage_pipeline(c, True, True)}
    status = {}
    with contextlib.ExitStack() as es:
        es.enter_context(torch.no_grad())
        es.enter_context(sdpa)
        for s in stages:
            if s == "dit0" and "e2e" in stages:
                continue  # step 0 is recorded by the e2e run itself
            missing = [f for f in NEEDS[s] if not (model / f).is_file()]
            t0, tb = time.time(), None
            if missing:
                status[s] = f"skipped: missing {missing[0]}"
            else:
                try:
                    fns[s](ctx)
                    status[s] = "ok"
                except Exception as e:
                    status[s] = f"failed: {type(e).__name__}: {e}"
                    tb = traceback.format_exc()
                finally:
                    free_cuda(torch)
            REC.note("stage", stage=s, status=status[s], seconds=round(time.time() - t0, 3), missing=missing,
                     traceback=tb)
            print(f"[k6 capture] {s}: {status[s]}", file=sys.stderr, flush=True)
        if "dit0" in stages and "e2e" in stages:
            status["dit0"] = status["e2e"]
            REC.note("stage", stage="dit0", status=status["dit0"], seconds=0.0, missing=[], traceback=None,
                     recorded_by="e2e")
    REC.note("precision_after", **precision_note(torch))
    REC.note("versions", **versions_note(torch))
    REC.stop()
    failed = [s for s, v in status.items() if v.startswith("failed")]
    skipped = [s for s, v in status.items() if v.startswith("skipped")]
    print(json.dumps({"k6_capture": str(out), "qualification": QUALIFICATION, "ops": REC.n, "stages": status, "checks_failed":
                      sorted(k for k, v in ctx.checks.items() if not v)}))
    sys.exit(1 if failed or not all(ctx.checks.values()) else 2 if skipped else 0)


if __name__ == "__main__":
    main()
