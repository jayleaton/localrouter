"""python -m stk_twin.h3.capture_avae --out DIR [--a 93 --seed 7 --models DIR]

One recorded decode of the audio VAE (`avae.*` ops, stk_twin/h3/vae_audio.py) of a fixed latent [1, 32, 2, A] from the
toolkit's portable generator, scaled by 0.25 as test_vae_audio builds its latents (the sampler's audio after
process_latent_out). For `localrouter check h3-avae`; the `avae_shapes` note carries what Zig needs. One JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..qwen_image import noise
from ..rec import REC
from .vae_audio import AudioVAE

HOP = 800  # samples per latent frame (the decoder rates multiply to 800)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("VMODELS", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", type=int, default=93)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = torch.device("cuda")
    vae = AudioVAE.from_checkpoint(Path(a.models) / "vae" / "minimax_h3_audio_vae_fp32.safetensors", device="cuda")
    shape = (1, 32, 2, a.a)
    z = torch.from_numpy(noise(a.seed, int(np.prod(shape)))).view(shape).float().mul_(0.25).to(dev).contiguous()
    REC.start(Path(a.out))
    REC.note("avae_shapes", a=a.a, seed=a.seed, latent=list(shape), wav=[1, 2, a.a * HOP], samples=a.a * HOP)
    with torch.inference_mode():
        wav = vae.decode(z)
    REC.stop()
    print(json.dumps({"avae_capture": str(a.out), "ops": REC.n, "a": a.a, "wav": list(wav.shape)}))


if __name__ == "__main__":
    main()
