"""The image VAE decoder's tile plan and blend from ComfyUI e9afbccc's own code (no GPU, no checkpoint):
python -m stk_twin.h3i.fixture_tiles --comfy COMFYUI_DIR --out src/engines/minimax_h3/vae_image_tiles.json

`MiniMaxH3VideoVAE.split_tiles` and `.blend` (comfy/ldm/minimax/vae.py) are compiled from the file unchanged (as
`test_pdd.py` extracts its functions) and run on a stand-in holding what they read: tile_size 256, tile_overlap_min 64,
vae_ratio 16 (docs/dev/H3-IMAGE.md, "Decided"). The JSON has the plan of every side 16 .. 4096 in steps of 16, and for
each overlap those plans use, `blend` on fp16 CPU tensors: its weights (a = 1, b = 0 gives w_a; a = 0, b = 1 gives w_b)
and a blend of seeded random fp16 values, as raw fp16 bits. The Zig host test (`vae_image.zig`) holds `splitTiles` and its
blend formula to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .test_pdd import extract


def bits(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.contiguous().view(torch.int16).numpy().astype(np.uint16)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="ComfyUI at e9afbccc")
    ap.add_argument("--out", required=True, help="the JSON to write")
    a = ap.parse_args()
    vae = Path(a.comfy) / "comfy/ldm/minimax/vae.py"
    scope = {"math": __import__("math"), "torch": torch}
    split = extract(vae, "MiniMaxH3VideoVAE", "split_tiles", scope)
    blend = extract(vae, "MiniMaxH3VideoVAE", "blend", scope)
    vae_self = SimpleNamespace(tile_size=256, tile_overlap_min=64, vae_ratio=16)

    plans = []
    extents: set[int] = set()
    for side in range(16, 4097, 16):
        starts, lens, overlaps = split(vae_self, side)
        plans.append({"side": side, "starts": starts, "lens": lens, "overlaps": overlaps})
        extents.update(overlaps)

    gen = torch.Generator().manual_seed(0)
    blends = []
    for e in sorted(extents):
        one = torch.ones(1, e, dtype=torch.float16)
        zero = torch.zeros(1, e, dtype=torch.float16)
        wa = blend(vae_self, one, zero, e, -1)
        wb = blend(vae_self, zero, one, e, -1)
        ra = torch.randn(1, e, generator=gen).to(torch.float16)
        rb = torch.randn(1, e, generator=gen).to(torch.float16)
        out = blend(vae_self, ra, rb, e, -1)
        blends.append({"extent": e, "wa": bits(wa[0]), "wb": bits(wb[0]), "a": bits(ra[0]), "b": bits(rb[0]), "out": bits(out[0])})

    Path(a.out).write_text(json.dumps({"tile_size": 256, "tile_overlap_min": 64, "vae_ratio": 16, "plans": plans,
                                       "blends": blends, "torch": torch.__version__}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
