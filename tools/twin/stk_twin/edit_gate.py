"""Deferred full-checkpoint edit comparison with the published diffusers pipeline, using the same portable noise.
Run only in the granted GPU window; inputs/model are local, no checkpoint downloads. Component gates run separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--prompt", default="Change the lighting to a warm sunset, preserving the subject.")
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-psnr", type=float, default=20.0, help="provisional until backend-floor measurements")
    a = ap.parse_args()
    if not 1 <= len(a.image) <= 5 or a.steps <= 0:
        ap.error("need 1..5 reference images and positive steps")
    width, height = map(int, a.size.split("x"))
    if min(width, height) < 32 or width % 32 or height % 32:
        ap.error("reference pipeline comparison requires sides divisible by 32")
    for key in ("STK_OPS", "STK_ATTN", "STK_TE", "STK_VAE"):
        os.environ.setdefault(key, "stk")
        if os.environ[key] != "stk":
            ap.error(f"{key}=stk required")

    import numpy as np
    import torch
    from PIL import Image
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    from .dit import QwenImageDiT
    from .qwen_image import QwenImage, load_transformer, noise

    images = []
    for path in a.image:
        with Image.open(path) as image:
            if image.format != "PNG":
                ap.error("PNG inputs required for the pixel gate")
            images.append(image.convert("RGBA"))
    q = QwenImage(a.model, QwenImageDiT.from_bf16(load_transformer(a.model), "bf16"))
    with torch.inference_mode():
        ours = q.generate_edit(a.prompt, images, height, width, a.steps, a.seed)
    del q
    torch.cuda.empty_cache()
    # Fresh, unpatched components; sharing the twin's patched VAE would invalidate this gate.
    # Run sequentially so five-reference tests do not retain two complete model/prefix allocations.
    refdit = QwenImage21Transformer2DModel.from_pretrained(a.model, subfolder="transformer",
                                                         torch_dtype=torch.bfloat16, local_files_only=True)
    reference = QwenImage21Pipeline.from_pretrained(a.model, transformer=refdit,
                                                    torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    with torch.inference_mode():
        lat = torch.from_numpy(noise(a.seed, 64 * (height // 16) * (width // 16)))
        lat = lat.view(1, 64, height // 16, width // 16).to("cuda", torch.bfloat16)
        lat = lat.flatten(2).transpose(1, 2).contiguous()
        theirs = reference(prompt=a.prompt, image=images, height=height, width=width,
                           num_inference_steps=a.steps, latents=lat, true_cfg_scale=1.0).images[0]
    x, y = np.asarray(ours.image, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
    if x.shape != y.shape:
        raise AssertionError(f"output shape {x.shape} differs from reference {y.shape}")
    mse = float(np.mean((x-y)**2))
    score = float("inf") if mse == 0 else float(10*np.log10(255**2/mse))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ours.image.save(out / "twin.png")
    theirs.save(out / "diffusers.png")
    result = {"gate": "edit pipeline", "refs": len(images), "size": a.size, "steps": a.steps, "seed": a.seed,
              "psnr": score, "min_psnr": a.min_psnr, "twin_seconds": ours.seconds, "pass": score >= a.min_psnr}
    (out / "gate.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
