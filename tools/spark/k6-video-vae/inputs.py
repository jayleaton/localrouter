"""Deterministic inputs of the D2 (Kandinsky-6 video VAE) qualification kit, written as raw little-endian files that the
oracle and the engine both read (kernels/cuda/kandinsky6/video_vae/QUALIFICATION.md, Cases).

  <case>.latent.bf16  bf16 [T][h][w][16]: float32 standard normal from NumPy PCG64, rounded to bf16 to nearest even
  <case>.latent.f32   fp32 [T][h][w][16]: float32 standard normal from PCG64 (the Pro-distill sampler's output dtype)
  <case>.image.u8     uint8 [H][W][3]: a smooth RGB pattern (sines of position) plus seeded integer noise, clipped
  <case>.eps.f16      fp16 [1][h][w][16]: float32 standard normal from PCG64, cast with NumPy's round to nearest even
  inputs.json         {name: {file, case, kind, dtype, shape, seed, sha256}}

Each array has its own PCG64 seed (SEEDS), so a file never depends on which other files are generated.

  python inputs.py --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# name: (case, kind, dtype, shape, seed)
SPECS = {
    "s_dec.latent": ("s_dec", "latent", "bf16", (3, 8, 12, 16), 0x6B36_0001),
    "f_dec.latent": ("f_dec", "latent", "f32", (3, 8, 12, 16), 0x6B36_0003),
    "s_enc.image": ("s_enc", "image", "u8", (64, 96, 3), 0x6B36_0101),
    "s_enc.eps": ("s_enc", "eps", "f16", (1, 8, 12, 16), 0x6B36_0201),
    "n_enc.image": ("n_enc", "image", "u8", (480, 864, 3), 0x6B36_0102),
    "n_enc.eps": ("n_enc", "eps", "f16", (1, 60, 108, 16), 0x6B36_0202),
    "n_dec.latent": ("n_dec", "latent", "f32", (31, 60, 108, 16), 0x6B36_0002),
}
EXT = {"bf16": "bf16", "u8": "u8", "f16": "f16", "f32": "f32"}


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round to nearest even. Inputs here are finite."""
    u = np.ascontiguousarray(x, dtype="<f4").view("<u4").astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def latent(shape, seed, dtype) -> np.ndarray:
    x = np.random.Generator(np.random.PCG64(seed)).standard_normal(shape, dtype=np.float32)
    return bf16_rne(x) if dtype == "bf16" else x.astype("<f4")


def eps(shape, seed) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(shape, dtype=np.float32).astype("<f2")


def image(shape, seed) -> np.ndarray:
    h, w, _ = shape
    y, x = np.meshgrid(np.arange(h, dtype=np.float64) / h, np.arange(w, dtype=np.float64) / w, indexing="ij")
    base = np.stack([
        128 + 90 * np.sin(2 * np.pi * (1.5 * x + 0.25 * y)),
        128 + 90 * np.sin(2 * np.pi * (1.0 * y - 0.75 * x) + 1.0),
        128 + 90 * np.cos(2 * np.pi * np.hypot(x - 0.4, y - 0.6) * 2.0),
    ], axis=-1)
    noise = np.random.Generator(np.random.PCG64(seed)).integers(-12, 13, size=shape)
    return np.clip(np.rint(base) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = {"latent": latent, "eps": eps, "image": image}
    index = {}
    for name, (case, kind, dtype, shape, seed) in SPECS.items():
        arr = gen[kind](shape, seed, dtype) if kind == "latent" else gen[kind](shape, seed)
        data = arr.tobytes()
        f = f"{name}.{EXT[dtype]}"
        (out / f).write_bytes(data)
        index[name] = {"file": f, "case": case, "kind": kind, "dtype": dtype, "shape": list(shape), "seed": seed,
                       "sha256": hashlib.sha256(data).hexdigest()}
    (out / "inputs.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({"inputs": str(out), "files": {k: v["sha256"] for k, v in index.items()}}))


if __name__ == "__main__":
    main()
