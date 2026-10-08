"""Qwen-Image 2.1 end to end, the reference the Zig engine is checked against.

Text encoder (Qwen3-VL 8B, last decoder layer before its final norm), prompt template, sigmas and VAE come from the
official diffusers pipeline (`Qwen/Qwen-Image-2.1`, diffusers 0.41); the DiT is `dit.QwenImageDiT` (tfimage's
arithmetic, recorded op by op). Noise is LocalRouter's own portable generator (`noise`), so Zig can reproduce it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from . import ops
from .dit import OPS, QwenImageDiT
from . import smath
from .rec import REC
from .te import TextEncoder, encode as te_encode

TE = os.environ.get("STK_TE", "hf")  # "stk": LocalRouter's text encoder (te.py)
VAE = os.environ.get("STK_VAE", "diffusers")  # "stk": the VAE decoder on LocalRouter's kernels (vae_ops.py)

M64 = (1 << 64) - 1


def uniforms(seed: int, first: int, count: int) -> np.ndarray:
    """Uniform i in [0, 1) is splitmix64's output for state seed + (i + 1) * golden: counter based, so any range."""
    i = np.arange(first + 1, first + count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & M64) + i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def noise(seed: int, n: int) -> np.ndarray:
    """n standard normals (float32). Pair j takes uniforms 2j and 2j + 1, maps them to [-1, 1); pairs with
    0 < s < 1 (s = a^2 + b^2) give a * f, b * f with f = sqrt(-2 log(s) / s) (Marsaglia's polar method, float64,
    `smath.log`),
    in pair order. The Zig engine implements the same sequence."""
    out, k, j = np.empty(n, dtype=np.float64), 0, 0
    while k < n:
        m = max(1024, n - k)                     # candidate pairs this round (about 79 % are accepted)
        u = uniforms(seed, 2 * j, 2 * m)
        a, b = 2.0 * u[0::2] - 1.0, 2.0 * u[1::2] - 1.0
        s = a * a + b * b
        ok = (s > 0) & (s < 1)
        sk = s[ok]
        f = np.sqrt(-2.0 * np.array([smath.log(float(v)) for v in sk]) / sk)  # portable log, then IEEE ops
        pairs = np.stack([a[ok] * f, b[ok] * f], axis=1).reshape(-1)
        take = min(len(pairs), n - k)
        out[k:k + take] = pairs[:take]
        k, j = k + take, j + m
    return out.astype(np.float32)


@dataclass
class Result:
    image: object  # PIL.Image
    latents: torch.Tensor
    seconds: dict


class QwenImage:
    def __init__(self, model_dir: str | Path, dit: QwenImageDiT, device="cuda"):
        from diffusers import QwenImage21Pipeline

        self.dev = torch.device(device)
        self.pipe = QwenImage21Pipeline.from_pretrained(str(model_dir), transformer=None, torch_dtype=torch.bfloat16)
        self.pipe.text_encoder.to(self.dev)
        self.pipe.vae.to(self.dev)
        self.dit = dit
        self.te = TextEncoder(self.pipe.text_encoder, self.dev) if TE == "stk" else None
        if VAE == "stk":
            from . import vae_ops
            vae_ops.patch(self.pipe.vae)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        """[1, L, 4096] bf16: the pipeline's text path (template, system tokens dropped); on LocalRouter's kernels
        with STK_TE=stk, else transformers' own forward."""
        if self.te is not None:
            return te_encode(self.te, self.pipe, prompt)
        emb, mask, _ = self.pipe._get_qwen_prompt_embeds(prompt, device=self.dev)
        return emb[:, : int(mask.sum())]

    def sigmas(self, height: int, width: int, steps: int) -> list[float]:
        """The pipeline's schedule for this size (dynamic exponential shift, terminal 0.02), with the final 0."""
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

        sc = self.pipe.scheduler
        seq = (height // 16) * (width // 16)
        mu = calculate_shift(seq, sc.config.get("base_image_seq_len", 256), sc.config.get("max_image_seq_len", 4096),
                             sc.config.get("base_shift", 0.5), sc.config.get("max_shift", 1.15))
        sigmas = self.pipe.config.get("sample_sigmas") or np.linspace(1.0, 1 / steps, steps)
        sc.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
        return [float(s) for s in sc.sigmas]

    @torch.inference_mode()
    def sample(self, context: torch.Tensor, height: int, width: int, steps: int, seed: int, record_step: int = -1):
        """Euler as the diffusers scheduler steps it: fp32 update, latents back to bf16 each step."""
        H, W = height // 16, width // 16
        sig = self.sigmas(height, width, steps)
        REC.note("sigmas", values=sig, seed=seed, height=height, width=width)
        x = torch.from_numpy(noise(seed, 64 * H * W)).view(1, 64, H, W).to(self.dev, torch.bfloat16)
        on = REC.on
        for i in range(len(sig) - 1):
            REC.on = on and (record_step < 0 or i == record_step)
            v = self.dit(x, torch.tensor([sig[i]], device=self.dev), context)
            with REC.op(f"step{i}.euler", "euler", {"sigma": sig[i], "sigma_next": sig[i + 1]}, x=x, v=v) as o:
                x = ops.euler(x, v, sig[i + 1] - sig[i]) if OPS == "stk" else (x.float() + (sig[i + 1] - sig[i]) * v.float()).to(torch.bfloat16)
                o.out(y=x)
        REC.on = on
        return x

    @torch.inference_mode()
    def decode(self, x: torch.Tensor):
        vae = self.pipe.vae
        if VAE == "stk":
            from . import vae_ops
            return vae_ops.decode(vae, x)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
        img = vae.decode(x.unsqueeze(2) * std + mean, return_dict=False)[0][:, :, 0]
        return self.pipe.image_processor.postprocess(img, output_type="pil")[0]

    def generate(self, prompt: str, height: int = 1024, width: int = 1024, steps: int = 25, seed: int = 0) -> Result:
        t = {}
        sync = torch.cuda.synchronize
        t0 = time.perf_counter()
        ctx = self.encode(prompt)
        sync()
        t["encode"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        x = self.sample(ctx, height, width, steps, seed)
        sync()
        t["sample"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        img = self.decode(x)
        sync()
        t["decode"] = time.perf_counter() - t0
        return Result(img, x, t)


def load_transformer(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """The DiT's bf16 weights from the diffusers `transformer/` shards (names as tfimage expects)."""
    w: dict[str, torch.Tensor] = {}
    for f in sorted((Path(model_dir) / "transformer").glob("*.safetensors")):
        w.update(load_file(str(f)))
    return w
