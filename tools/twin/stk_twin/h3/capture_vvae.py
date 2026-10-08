"""python -m stk_twin.h3.capture_vvae --out DIR [--size 768x448 --frames 22 --seed 7 --scale 1.0 --tiles 0:1:1 --layers 0,1,35 --models DIR]

One decode of the video VAE (`vvae.*` ops, stk_twin/h3/vae_video.py) of a fixed latent [1, 24, T, H/16, W/16] from the
toolkit's portable generator (scaled by --scale, fp32 as the sampler hands it over, then .to(fp16) inside decode). For
`localrouter check h3-vvae`; the `vvae_shapes` note carries what Zig needs. One JSON line.

What is recorded stays a few GB: a tile at full size (7 x 16 x 16 latents, 1797 tokens) is about 590 MB a block (the score
matrices alone are 207 MB each), so by default one tile (--tiles "chunk:row:col,..." ; row 1, col 1 of a 768x448 video has both
the y and the x blend) and three blocks (--layers; the first two and the last) are recorded. The clip-level ops (the denorm and
the part writes, with the uint8 frames) are always recorded, and the note holds the sha256 of ALL decoded frames, so the
final video is compared whole whatever was recorded. Zig decodes everything and probes only the selected tiles and blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ..qwen_image import noise
from ..rec import REC
from . import vae_video as V

LAYER = re.compile(r"\.L(\d+)\.")


def parse_size(s: str) -> tuple[int, int]:
    w, h = (int(x) for x in s.lower().split("x"))
    if w % V.RATIO or h % V.RATIO:
        raise SystemExit(f"--size {s}: width and height must be multiples of {V.RATIO}")
    return w, h


def latent_frames(frames: int) -> int:
    """T for a pixel frame count: F = 17 k + 5 for T = 5 k + 2, or one frame for T = 1."""
    if frames == 1:
        return 1
    if frames < 22 or (frames - 5) % 17:
        raise SystemExit(f"--frames {frames}: must be 1 or 17 k + 5 (22, 39, 56, ..., 124)")
    return 5 * ((frames - 5) // 17) + 2


def keep_layers(keep: set[int]) -> None:
    """Record only the blocks in `keep`: every op whose name has `.L{i}.` with i outside it runs unrecorded. (Zig probes the
    same blocks, `Decoder.layers`, so the capture stays consistent.)"""
    orig = REC.op

    @contextmanager
    def op(name, kind, attrs=None, **ins):
        m = LAYER.search(name)
        if REC.on and m and int(m.group(1)) not in keep:
            REC.on = False
            try:
                with orig(name, kind, attrs, **ins) as o:
                    yield o
            finally:
                REC.on = True
        else:
            with orig(name, kind, attrs, **ins) as o:
                yield o

    REC.op = op


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("VMODELS", ""))
    ap.add_argument("--checkpoint", default=os.environ.get("VVAE", ""), help="overrides MODELS/vae/minimax_h3_video_vae_fp16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="768x448", help="WIDTHxHEIGHT in pixels")
    ap.add_argument("--frames", type=int, default=22)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tiles", default="0:1:1", help="chunk:row:col[,...] recorded; the others run unrecorded")
    ap.add_argument("--layers", default="0,1,35", help="blocks recorded (comma separated), or 'all'")
    a = ap.parse_args()

    width, height = parse_size(a.size)
    T = latent_frames(a.frames)
    h, w = height // V.RATIO, width // V.RATIO
    tiles = [tuple(int(x) for x in t.split(":")) for t in a.tiles.split(",") if t]
    rows, cols = len(V.split_tiles(height)[0]), len(V.split_tiles(width)[0])
    nclips = 1 if T == 1 else V.temporal_chunks(T)[1]
    for c, r, k in tiles:
        if not (0 <= c < nclips and 0 <= r < rows and 0 <= k < cols):
            raise SystemExit(f"--tiles {c}:{r}:{k}: the video has {nclips} clip(s) of {rows} x {cols} tiles")
    layers = list(range(V.LAYERS)) if a.layers == "all" else sorted({int(x) for x in a.layers.split(",") if x})
    if any(not 0 <= i < V.LAYERS for i in layers):
        raise SystemExit(f"--layers: blocks are 0..{V.LAYERS - 1}")

    ck = a.checkpoint or str(Path(a.models) / "vae" / "minimax_h3_video_vae_fp16.safetensors")
    dev = torch.device("cuda")
    vae = V.VideoVAE.from_checkpoint(ck, device="cuda")
    shape = (1, V.ZC, T, h, w)
    z = torch.from_numpy(noise(a.seed, int(np.prod(shape)))).view(shape).float().mul_(a.scale).to(dev).contiguous()

    sel = set(tiles)
    keep_layers(set(layers))
    REC.start(Path(a.out))
    with torch.inference_mode():
        out = vae.decode(z, rec=lambda c, r, k: (r < 0 and k < 0) or (c, r, k) in sel)
    sha = hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
    REC.note("vvae_shapes", latent=[V.ZC, T, h, w], frames=int(out.shape[0]), width=width, height=height,
             video=list(out.shape), seed=a.seed, scale=a.scale, tiles=[list(t) for t in tiles], layers=layers,
             grid=[rows, cols], clips=nclips, sha256=sha)
    REC.stop()
    size = sum(f.stat().st_size for f in (Path(a.out) / "blobs").iterdir())
    print(json.dumps({"vvae_capture": str(a.out), "ops": REC.n, "latent": [V.ZC, T, h, w], "video": list(out.shape),
                      "tiles": [list(t) for t in tiles], "layers": layers, "blob_bytes": size, "sha256": sha}))


if __name__ == "__main__":
    main()
