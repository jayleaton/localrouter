"""python -m stk_twin.h3.gate --models DIR [--size 768x448 --frames 56 --steps 8 --prompt ...]

The video twin's DiT against ComfyUI 0.37.0's own MiniMaxH3Model on identical inputs (the structure gate, as the
image twin's M2 gate): ComfyUI as a library (its text encoder, its DiT at bf16 with SDPA), the twin in bf16 mode
(our bf16 GEMM and attention, same checkpoint tensors). Along one trajectory (ComfyUI's velocities, Euler on the 8-step
schedule) each step's video and audio velocities: twin vs ComfyUI (backend A = flash first), against the floor
ComfyUI gives itself with another attention backend (B = cuDNN first). Also the refined text context. One JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from .shapes import PROMPT, latent_shapes


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().flatten(), b.float().flatten()
    return {"cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)), "rel_l2": float((a - b).norm() / b.norm())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("VMODELS", ""))
    ap.add_argument("--size", default="768x448")
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--attn", default="bf16", help="the twin's attention: bf16 or int8")
    a = ap.parse_args()
    sys.path.insert(0, os.environ["COMFY"])
    import comfy.model_management as mm
    import comfy.ops
    import comfy.samplers
    import comfy.sd
    from torch.nn.attention import SDPBackend

    from .dit import H3Dit

    models = Path(a.models)
    W, Hh = (int(v) for v in a.size.split("x"))
    res = {"size": a.size, "frames": a.frames, "steps": a.steps}
    t0 = time.time()

    # ComfyUI's text encoder: the Qwen3-VL 32B layer-50 states [1, L, 5120] fp32
    clip = comfy.sd.load_clip([str(models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")],
                              clip_type=comfy.sd.CLIPType.MINIMAX)
    enc = clip.encode_from_tokens(clip.tokenize(a.prompt), return_dict=True)
    cond = enc["cond"].to(torch.bfloat16)
    res["text_tokens"] = cond.shape[1]
    del clip, enc
    mm.soft_empty_cache()

    patcher = comfy.sd.load_diffusion_model(str(models / "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"))
    mm.load_models_gpu([patcher], force_full_load=True)
    base = patcher.model
    dm = base.diffusion_model
    dev = next(dm.parameters()).device
    sigmas = comfy.samplers.calculate_sigmas(patcher.get_model_object("model_sampling"), "simple", a.steps).to(torch.float32)
    res["sigmas"] = [float(s) for s in sigmas]
    # reference check: LocalRouter's own schedule (sampler.sigmas, what build / capture / generate use) is ComfyUI's, bit for bit
    from . import sampler
    res["sigmas_equal_toolkit"] = res["sigmas"] == sampler.sigmas(a.steps)
    if not res["sigmas_equal_toolkit"]:
        raise SystemExit(f"sampler.sigmas({a.steps}) differs from ComfyUI's simple schedule: {sampler.sigmas(a.steps)} vs {res['sigmas']}")
    res["load_s"] = round(time.time() - t0, 1)

    with torch.inference_mode():
        ctx = dm.preprocess_text_embeds(cond.to(dev))                       # ComfyUI: extra_conds, once a run
        twin = H3Dit({k: v for k, v in dm.state_dict().items()}, kind="bf16", attn=a.attn, device=dev)
        ours = twin.prepare(cond)
        res["context"] = cmp(ours, ctx[0])

        vs, as_ = latent_shapes(W, Hh, a.frames)
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        xv = torch.randn(vs, generator=g).to(dev)
        xa = torch.randn(as_, generator=g).to(dev)
        payload = {"audio_scale": 4.0}

        def comfy_call(v, au, sigma, prio):
            comfy.ops.SDPA_BACKEND_PRIORITY[:] = prio
            t = torch.full((1,), float(sigma), dtype=torch.float32, device=dev) * 1000.0
            return dm(x=[v, au], timestep=t, context=ctx, transformer_options={"sample_sigmas": sigmas.to(dev)},
                      minimax_payload=dict(payload))

        flash_first = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        cudnn_first = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        steps = []
        for i in range(a.steps):
            s, sn = float(sigmas[i]), float(sigmas[i + 1])
            v, au = xv.to(torch.bfloat16), xa.to(torch.bfloat16)
            ra = comfy_call(v, au, s, flash_first)
            rb = comfy_call(v, au, s, cudnn_first)
            rt = twin(v, au, s)
            row = {"sigma": s}
            for name, k in (("video", 0), ("audio", 1)):
                row[name] = {"twin": cmp(rt[k], ra[k]), "floor": cmp(rb[k], ra[k])}
                row[name]["ratio"] = row[name]["twin"]["rel_l2"] / max(row[name]["floor"]["rel_l2"], 1e-12)
            steps.append(row)
            # Euler on ComfyUI's velocity (the model returns the negated velocity: denoised = x - out * sigma)
            xv = xv + ra[0].float() * (sn - s)
            xa = xa + ra[1].float() * (sn - s)
        res["steps"] = steps
        res["worst_ratio"] = max(max(r["video"]["ratio"], r["audio"]["ratio"]) for r in steps)
        res["pass"] = res["worst_ratio"] <= 1.5 and res["context"]["cos"] > 0.999
    print(json.dumps({"h3_gate": res}))


if __name__ == "__main__":
    main()
