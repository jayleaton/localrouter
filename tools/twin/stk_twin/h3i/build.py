"""python -m stk_twin.h3i.build --models DIR --out PACK [--dit FILE] [--no-tokenizer]

The H3-Image DiT's bf16 pack (stk-pack/1, what `qwen_image/pack.zig` opens and the Zig engine loads) from the published
`minimax_h3_image_pruned_bf16.safetensors`: the checkpoint's tensors under the video pack's names (`stk_twin.h3.pack`),
the block linears as stored (bf16, pack kind "bf16": `weights.zig` `Store.linear` reads `{key}.weight`), no audio
tensors, then `h3.pack.write_tokenizer` (the text encoder's tokenizer.json, as the video pack). Streamed: one tensor in
memory at a time. The image VAE is not packed: the engine reads the published fp16 VAE file. One JSON line.

The pack (L = layer 0 .. 49; shapes of the published checkpoint; "as stored" = the checkpoint's dtype, bf16):

  pack tensor                                  dtype  shape            checkpoint tensor
  adaln_t_table                                f32    [1025, 8]        adaln_t_table
  rope.inv_freq                                f32    [16]             rope.inv_freq
  video_patch_proj.weight / .bias              f32    [5376, 96] / [5376]   video_patch_proj.*
  final_layer.video_out.weight / .bias         f32    [576, 5376] / [576]   final_layer.video_out.*  (6 PDD heads x 96)
  final_layer.adaln_proj.linear.weight / .bias f32    [10752, 8] / [10752]  final_layer.adaln_proj.linear.* (F16 / BF16)
  final_layer.norm.weight                      bf16   [5376]           final_layer.norm.weight
  condition_proj.weight / .bias                bf16   [5376, 5120] / [5376]  condition_proj.*
  token_refiner.*  (2 blocks + final_norm)     bf16   as stored        token_refiner.*  (same names)
  L{i}.norm1, L{i}.norm2                       f32    [5376]           blocks.{i}.norm1.weight, .norm2.weight
  L{i}.q_norm, L{i}.k_norm                     bf16   [128]            blocks.{i}.attn.q_norm.weight, .k_norm.weight
  L{i}.ada_w, L{i}.ada_b                       f32    [96768, 8] / [96768]  blocks.{i}.adaln_proj.linear.* (F16 / BF16)
  L{i}.qkv.weight   (kind bf16)                bf16   [21504, 5376]    blocks.{i}.attn.qkv_proj.weight
  L{i}.out.weight   (kind bf16)                bf16   [5376, 7168]     blocks.{i}.attn.out_proj.weight
  L{i}.fc1.weight   (kind bf16)                bf16   [28672, 5376]    blocks.{i}.mlp.fc1.weight
  L{i}.fc2.weight   (kind bf16)                bf16   [5376, 14336]    blocks.{i}.mlp.fc2.weight

Not packed: `blocks.{i}.attn.comfy_attention.config` (U8, ComfyUI's attention selector). The manifest: format
"stk-pack/1", model "minimax-h3-image", precision "bf16", kinds {L{i}.{qkv,out,fc1,fc2}: "bf16"}, acts {} (no static
scales), `pdd_steps` and `pdd_sigmas` (the checkpoint's `__metadata__.config.transformer`, which must equal
`sampler.PDD_SIGMAS`, the grid pdd.zig carries), source, tensors {name: dtype, shape, sha256}, and `tokenizer`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import torch

from ..files import readable
from ..h3.dit import LINEARS, SOURCE
from . import sampler

DIT = "diffusion_models/minimax_h3_image_pruned_bf16.safetensors"
MODEL, PRECISION = "minimax-h3-image", "bf16"
SIDE_F32 = ["adaln_t_table", "rope.inv_freq", "video_patch_proj.weight", "video_patch_proj.bias",
            "final_layer.video_out.weight", "final_layer.video_out.bias",
            "final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias"]
_ST = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.uint8: "U8"}


def plan(keys: list[str]) -> dict[str, tuple[str, torch.dtype]]:
    """pack name -> (checkpoint key, pack dtype), in pack order. Raises on audio tensors or a key the pack drops
    unexpectedly (anything but the attention selector)."""
    keys = [k.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.") for k in keys]
    if any(k.startswith(("audio_patch_proj.", "final_layer.audio_out.")) for k in keys):
        raise ValueError("the checkpoint has an audio stream: that is the video DiT (stk_twin.h3.build)")
    layers = 1 + max(int(k.split(".")[1]) for k in keys if k.startswith("blocks."))
    out: dict[str, tuple[str, torch.dtype]] = {k: (k, torch.float32) for k in SIDE_F32}
    out["final_layer.norm.weight"] = ("final_layer.norm.weight", torch.bfloat16)
    for k in sorted(keys):
        if k.startswith(("condition_proj.", "token_refiner.")):
            out[k] = (k, torch.bfloat16)
    for i in range(layers):
        p, q = f"blocks.{i}.", f"L{i}."
        out[q + "norm1"] = (p + "norm1.weight", torch.float32)
        out[q + "norm2"] = (p + "norm2.weight", torch.float32)
        out[q + "q_norm"] = (p + "attn.q_norm.weight", torch.bfloat16)
        out[q + "k_norm"] = (p + "attn.k_norm.weight", torch.bfloat16)
        out[q + "ada_w"] = (p + "adaln_proj.linear.weight", torch.float32)
        out[q + "ada_b"] = (p + "adaln_proj.linear.bias", torch.float32)
        for name in LINEARS:
            out[q + name + ".weight"] = (p + SOURCE[name] + ".weight", torch.bfloat16)
    used = {src for src, _ in out.values()}
    missing = sorted(used - set(keys))
    if missing:
        raise ValueError(f"the checkpoint lacks {missing[:8]}")
    dropped = sorted(k for k in set(keys) - used if not k.endswith(".attn.comfy_attention.config"))
    if dropped:
        raise ValueError(f"tensors the pack would drop: {dropped[:8]}")
    return out


def pdd_config(meta: dict | None) -> tuple[int, list[float]]:
    """(pdd_steps, pdd_sigmas) from the safetensors `__metadata__.config` (`{"transformer": {...}}`)."""
    try:
        tr = json.loads((meta or {})["config"])["transformer"]
        steps, grid = int(tr["pdd_steps"]), [float(s) for s in tr["pdd_sigmas"]]
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError("the checkpoint's __metadata__ has no transformer pdd_steps / pdd_sigmas config") from e
    if len(grid) != steps + 1:
        raise ValueError(f"pdd_sigmas has {len(grid)} points for {steps} heads")
    if tuple(grid) != sampler.PDD_SIGMAS:
        raise ValueError(f"pdd_sigmas {grid} is not the grid the twin and pdd.zig carry ({list(sampler.PDD_SIGMAS)})")
    return steps, grid


def _bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes() if t.numel() else b""


def write(src: str | Path, out: str | Path, source: dict) -> dict:
    """The pack from the checkpoint at `src` into `out` (weights.safetensors, manifest.json), one tensor at a time."""
    from safetensors import safe_open

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with safe_open(str(src), framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata()
        steps, grid = pdd_config(meta)
        strip = {k.removeprefix("model.diffusion_model.").removeprefix("diffusion_model."): k for k in keys}
        names = plan(keys)
        # the header: every tensor's shape from the checkpoint's, its size from the pack dtype
        header, off = {}, 0
        for name, (key, dt) in names.items():
            shape = list(f.get_slice(strip[key]).get_shape())
            n = dt.itemsize
            for s in shape:
                n *= s
            header[name] = {"dtype": _ST[dt], "shape": shape, "data_offsets": [off, off + n]}
            off += n
        hj = json.dumps(header, separators=(",", ":")).encode()
        hj += b" " * (-len(hj) % 8)
        digest = {}
        with open(out / "weights.safetensors", "wb") as fo:
            fo.write(struct.pack("<Q", len(hj)))
            fo.write(hj)
            for name, (key, dt) in names.items():
                t = f.get_tensor(strip[key]).to(dt).contiguous()
                b = _bytes(t)
                assert len(b) == header[name]["data_offsets"][1] - header[name]["data_offsets"][0], name
                fo.write(b)
                digest[name] = hashlib.sha256(b).hexdigest()
    kinds = {n.removesuffix(".weight"): "bf16" for n in names if n.startswith("L") and n.endswith(tuple(f".{x}.weight" for x in LINEARS))}
    manifest = {"format": "stk-pack/1", "model": MODEL, "precision": PRECISION, "source": source, "kinds": kinds,
                "acts": {}, "pdd_steps": steps, "pdd_sigmas": grid,
                "tensors": {n: {"dtype": str(dt).removeprefix("torch."), "shape": header[n]["shape"], "sha256": digest[n]}
                            for n, (_, dt) in names.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    readable(out)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("VMODELS", ""))
    ap.add_argument("--dit", help=f"the checkpoint (default MODELS/{DIT})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tokenizer", action="store_true", help="skip tokenizer.json (needs transformers and te32's files)")
    a = ap.parse_args()
    t0 = time.time()
    src = Path(a.dit) if a.dit else Path(a.models) / DIT
    man = write(src, a.out, {"repo": "Comfy-Org/MiniMax-H3-Image", "dit": DIT if not a.dit else src.name})
    tok = None
    if not a.no_tokenizer:
        from ..h3.pack import write_tokenizer

        tok = write_tokenizer(a.out)["sha256"]
    print(json.dumps({"h3i_build": {"pack": a.out, "tensors": len(man["tensors"]), "linears": len(man["kinds"]),
                                    "pdd_sigmas": man["pdd_sigmas"], "tokenizer": tok, "seconds": round(time.time() - t0, 1)}}))


if __name__ == "__main__":
    main()
