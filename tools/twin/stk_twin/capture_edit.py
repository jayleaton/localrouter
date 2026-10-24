"""Record the edit twin from local PNG inputs: vision, multimodal text, encoder, prefix, first step and decode.
The directory is a fresh replay artifact, not an append destination. Run only in the granted GPU window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True, help="DiT precision pack directory (parent holds te/vae/edit.json)")
    ap.add_argument("--image", action="append", required=True, help="local PNG; repeat for 1..5 references")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", help="WIDTHxHEIGHT; defaults to last reference dimensions at about 1 MP")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not 1 <= len(a.image) <= 5 or a.steps < 1:
        ap.error("need 1..5 input images and positive steps")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        ap.error("capture output must be empty; keep prior artifacts in a different directory")
    width = height = None
    if a.size:
        try:
            width, height = map(int, a.size.split("x"))
        except ValueError:
            ap.error("size must be WIDTHxHEIGHT")
        if min(width, height) < 16 or width % 16 or height % 16:
            ap.error("size must be positive multiples of 16")
    for key in ("STK_OPS", "STK_ATTN", "STK_TE", "STK_VAE"):
        os.environ.setdefault(key, "stk")
        if os.environ[key] != "stk":
            ap.error(f"{key}=stk is required for an engine-replay capture")

    from PIL import Image
    from .dit import QwenImageDiT
    from .qwen_image import QwenImage
    from .rec import REC
    from .files import readable

    images, sources = [], []
    for path in map(Path, a.image):
        with Image.open(path) as image:
            if image.format != "PNG":
                ap.error("bit replay inputs must be PNG")
            images.append(image.convert("RGBA"))
        sources.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    root = Path(a.pack).parent
    marker = json.loads((root / "edit.json").read_text())
    if marker.get("format") != "stk-edit/1" or marker.get("refs_max", 0) < len(images):
        ap.error("pack does not support this edit request")
    for name in ("te", "vae"):
        digest = hashlib.sha256((root / name / "manifest.json").read_bytes()).hexdigest()
        if marker.get("manifest_sha256", {}).get(name) != digest:
            ap.error(f"edit marker is stale for the {name} pack")
    q = QwenImage(a.model, QwenImageDiT.from_pack(a.pack))
    REC.start(out)
    try:
        ctx, refs, slots, prepared = q.encode_edit(a.prompt, images)
        if height is None:
            width, height = prepared[-1].rgba.size
        REC.note("request", kind="image_edit", prompt=a.prompt, seed=a.seed, height=height, width=width,
                 steps=a.steps, sources=sources, slots=list(slots), precision=q.dit.precision,
                 reference_shapes=[list(r.shape) for r in refs])
        with REC.op("text_encoder", "text_encoder", {"prompt": a.prompt}) as o:
            o.out(context=ctx)
        for i, r in enumerate(refs):
            with REC.op(f"reference_latents.{i}", "reference_latents", {"slot": slots[i]}) as o:
                o.out(x=r)
        q.dit.drop_prefixes()
        x = q.sample(ctx, height, width, a.steps, a.seed, record_step=0, ref_latents=refs, image_slots=slots)
        with REC.op("latents", "latents", {}) as o:
            o.out(x=x)
        image = q.decode(x)
    finally:
        REC.stop()
    image.save(out / "image.png")
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    for i, original in enumerate(images):
        original.save(inputs / f"{i}.png")
    metadata = {"capture": str(out), "kind": "image_edit", "ops": REC.n, "width": width, "height": height,
                "references": len(refs), "slots": list(slots), "recorded_step": 0}
    (out / "edit-capture.json").write_text(json.dumps(metadata, indent=1))
    readable(out)
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
