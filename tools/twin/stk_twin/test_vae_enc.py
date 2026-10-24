"""Deferred GPU gate: MODEL=/path/to/Qwen-Image-2.1 python -m stk_twin.test_vae_enc [height=256] [width=height].

Compare unpatched diffusers mode/normalization against the patched encoder on deterministic RGBA pixels.
Also check input conversion and normalization in isolation byte for byte, repeatability, and finite outputs.
Only run in the granted GPU window; importing this module does not compile or initialize CUDA.
Numeric thresholds are provisional until the first window establishes the published backend floor.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

from . import vae_enc, vae_ops


def timed(fn):
    fn()  # Compile and prepare cached weights before timing.
    torch.cuda.synchronize()
    start = time.perf_counter()
    y = fn()
    torch.cuda.synchronize()
    return y, (time.perf_counter() - start) * 1000


def metrics(reference, actual):
    a, b = reference.float().flatten(), actual.float().flatten()
    delta = a - b
    return {"cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            "relative_l2": (delta.norm() / a.norm().clamp_min(1e-12)).item(),
            "max_abs": delta.abs().max().item(), "mean_abs": delta.abs().mean().item()}


@torch.inference_mode()
def main():
    from diffusers import AutoencoderKLQwenImage21

    height = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    width = int(sys.argv[2]) if len(sys.argv) > 2 else height
    if height <= 0 or height % 16 or width <= 0 or width % 32:
        raise ValueError("height must be positive/multiple of 16 and width positive/multiple of 32")
    vae = AutoencoderKLQwenImage21.from_pretrained(os.environ["MODEL"], subfolder="vae",
                                                  torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    vae.disable_tiling()
    # A CPU generator keeps the fixture independent of the active GPU architecture.
    rng = torch.Generator(device="cpu").manual_seed(7)
    rgba = torch.randint(0, 256, (height, width, 4), dtype=torch.uint8, generator=rng).cuda()
    # Transparent, translucent and opaque samples remain present in the same fixture.
    rgba[0, :, 3] = 0
    rgba[1, :, 3] = 128
    rgba[2, :, 3] = 255
    # Built on the CPU: a CUDA tensor divided by a Python scalar is multiplied by its reciprocal (an ulp off true
    # division), while image loaders divide exactly on the host, as the kernel does.
    reference_x = ((rgba.cpu().permute(2, 0, 1).float() / 255.0) * 2.0 - 1.0).to(torch.bfloat16)[None, :, None].contiguous().cuda()
    mean = torch.tensor(vae.config.latents_mean).to(reference_x.device, reference_x.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).to(reference_x.device, reference_x.dtype).view(1, -1, 1, 1, 1)
    reference_z, reference_ms = timed(lambda: vae.encode(reference_x).latent_dist.mode())
    reference = ((reference_z - mean) / std)[:, :, 0]

    # Gate every distinct published shortcut factor, including left-padded temporal slots.
    shortcut_cases = []
    seen = set()
    for mod in vae.encoder.modules():
        if type(mod).__name__ != "QwenImage21AvgDown3D":
            continue
        key = (mod.in_channels, mod.out_channels, mod.factor_t, mod.factor_s)
        if key in seen:
            continue
        seen.add(key)
        fixture = torch.randn((1, mod.in_channels, 1, 4, 4), generator=rng).to(torch.bfloat16).cuda()
        shortcut_cases.append((mod, fixture, mod(fixture)))
    vae_enc.patch(vae)
    avg_down_exact = all(torch.equal(mod(fixture), expected) for mod, fixture, expected in shortcut_cases)
    input_exact = bool(torch.equal(vae_enc.from_u8(rgba), reference_x))
    normalize_exact = bool(torch.equal(vae_enc.z_to_latents(vae, reference_z), reference))
    vae_ops.STATS.clear()
    actual, patched_ms = timed(lambda: vae_enc.encode(vae, rgba))
    stats = {k: count // 2 for k, count in vae_ops.STATS.items()}
    repeat = vae_enc.encode(vae, rgba)
    deterministic = bool(torch.equal(actual, repeat))
    finite = bool(torch.isfinite(actual).all())
    result = {"height": height, "width": width, "shape": list(actual.shape), **metrics(reference, actual),
              "input_bit_exact": input_exact, "normalize_bit_exact": normalize_exact, "avg_down_bit_exact": avg_down_exact,
              "deterministic": deterministic, "finite": finite, "reference_ms": reference_ms,
              "patched_ms": patched_ms, "ops": stats}
    cosine_min = float(os.environ.get("STK_VAE_ENC_COSINE_MIN", "0.999"))
    relative_max = float(os.environ.get("STK_VAE_ENC_RELATIVE_MAX", "0.03"))
    passed = (input_exact and normalize_exact and avg_down_exact and deterministic and finite and actual.shape == reference.shape
              and result["cosine"] >= cosine_min and result["relative_l2"] <= relative_max)
    result.update(passed=passed, cosine_min=cosine_min, relative_l2_max=relative_max)
    print(json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
