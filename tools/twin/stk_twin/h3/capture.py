"""python -m stk_twin.h3.capture --models DIR --pack PACK --out DIR [--size 768x448 --frames 56 --steps 8 --step 1]

One recorded MiniMax H3 step of the shipping form (NVFP4 linears from the pack, INT8 attention) for the Zig replays:
the text through LocalRouter's 32B encoder (recorded, `te.*`, for `localrouter check h3-te`), the token refiner (recorded, `refiner.*`), then the step's every op from latents
drawn by LocalRouter's portable generator (video, then audio) at the schedule's sigma `--step`. Also the TensorFold
NVFP4 layout of L0.qkv (the golden for Zig's repack). One JSON line.
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
from . import sampler
from .build import encode
from .dit import H3Dit
from .shapes import PROMPT, latent_shapes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("VMODELS", ""))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="768x448")
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()
    W, Hh = (int(v) for v in a.size.split("x"))
    dev = torch.device("cuda")
    out = Path(a.out)
    REC.start(out)
    REC.note("request", prompt=a.prompt, seed=a.seed, width=W, height=Hh, frames=a.frames, steps=a.steps, step=a.step)
    cond = encode(Path(a.models), [a.prompt], record=True)[0]       # te.* ops and the te32_ids note
    sig = sampler.sigmas(a.steps)
    twin = H3Dit.from_pack(a.pack, attn="int8", device=dev)
    vs, as_ = latent_shapes(W, Hh, a.frames)
    nv, na = int(np.prod(vs)), int(np.prod(as_))
    z = noise(a.seed, nv + na)                                   # one stream: video first, then audio
    xv = torch.from_numpy(z[:nv]).view(vs).to(dev, torch.bfloat16)
    xa = torch.from_numpy(z[nv:]).view(as_).to(dev, torch.bfloat16)
    REC.note("sigmas", values=sig)
    with torch.inference_mode():
        with REC.op("text_encoder", "text_encoder", {}) as o:
            o.out(context=cond[0].contiguous())
        twin.prepare(cond)
        with REC.op("step.inputs", "inputs", {}) as o:
            o.out(video=xv, audio=xa)
        vel_v, vel_a = twin(xv, xa, sig[a.step])
        with REC.op("step.outputs", "outputs", {}) as o:
            o.out(video=vel_v, audio=vel_a)
        lin = twin.blocks[0]["qkv"].lin
        with REC.op("form.L0.qkv", "weight_form", {"weight": "L0.qkv"}) as o:
            o.out(words=lin.words, bs=lin.bs)
    REC.stop()
    print(json.dumps({"h3_capture": str(out), "ops": REC.n, "sigma": sig[a.step], "tokens": cond.shape[1]}))


if __name__ == "__main__":
    main()
