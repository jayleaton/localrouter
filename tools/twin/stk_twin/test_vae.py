"""GPU test of the patched VAE decoder: `MODEL=/path/to/Qwen-Image-2.1 python -m stk_twin.test_vae [latent_hw=64]`.

Loads the VAE ($MODEL/vae, bf16), decodes a fixed latent (stk_twin.qwen_image.noise, seed 7, 64 x h x w) with the
original diffusers code and then with `vae_ops.patch`, and reports: PSNR of the two RGB images (8-bit, as the pipeline
saves them), max abs difference of the decoded float tensors, whether our `latents * std + mean` and our uint8 conversion
equal the pipeline's, whether two patched decodes are bit-identical, and the times. Ends with one JSON line; exit status 1
if the patched decode is not deterministic or the PSNR is below 35 dB.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

from . import vae_ops
from .qwen_image import noise


def timed(fn, n: int = 2):
    fn()  # warm-up (compiles / caches weights)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / n * 1e3


def main() -> int:
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    hw = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    vae = AutoencoderKLQwenImage21.from_pretrained(os.environ["MODEL"], subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=vae.config.z_dim)
    x = torch.from_numpy(noise(7, 64 * hw * hw)).view(1, 64, hw, hw).to(torch.bfloat16).cuda()

    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
    z_ref = x.unsqueeze(2) * std + mean

    with torch.inference_mode():
        ref, t_ref = timed(lambda: vae.decode(z_ref, return_dict=False)[0])
        ref_pil = proc.postprocess(ref[:, :, 0], output_type="pil")[0]

        vae_ops.patch(vae)
        z = vae_ops.latents_to_z(vae, x)
        denorm_exact = bool(torch.equal(z, z_ref))
        vae_ops.STATS.clear()
        got, t_ours = timed(lambda: vae.decode(z, return_dict=False)[0])
        stats = {k: v // 3 for k, v in vae_ops.STATS.items()}  # timed() ran it three times
        got2 = vae.decode(z, return_dict=False)[0]
        deterministic = bool(torch.equal(got, got2))

        u = vae_ops.to_u8(got).cpu().numpy()
        got_pil = proc.postprocess(got[:, :, 0], output_type="pil")[0]
        u8_exact = bool(np.array_equal(u, np.asarray(got_pil)))

    a, b = np.asarray(ref_pil).astype(np.float64), u.astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    psnr = 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
    diff = (ref.float() - got.float()).abs()
    res = {
        "latent_hw": hw, "out_shape": list(got.shape), "psnr_db": round(psnr, 3), "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(), "denorm_bit_exact": denorm_exact, "u8_matches_pil": u8_exact,
        "deterministic": deterministic, "t_orig_ms": round(t_ref, 2), "t_patched_ms": round(t_ours, 2), "ops": stats,
    }
    print(json.dumps(res), flush=True)
    return 0 if deterministic and psnr >= 35.0 else 1


if __name__ == "__main__":
    sys.exit(main())
