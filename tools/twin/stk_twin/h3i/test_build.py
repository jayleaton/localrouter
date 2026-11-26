"""Host check: python -m stk_twin.h3i.test_build --scratch DIR (no GPU, no checkpoint).

`build.write` on a tiny random safetensors with the published checkpoint's key names, dtypes and metadata (2 layers,
small widths, the real head dim 128, the 6-head bank), read back: the pack's names are exactly the contract in build.py's
docstring (the video pack's names for every shared tensor, `L{i}.{qkv,out,fc1,fc2}.weight` bf16 [n, k] of kind "bf16",
no audio, no attention selector), every dtype and shape, every value equal to the source's after the stated conversion,
every sha256, the manifest fields (format, model, precision, kinds, acts, pdd_steps, pdd_sigmas). Then the pack drives
the twin: `H3ImageDit.from_pack` and `H3ImageDit` on the source tensors give the same bits on one CPU step. And the
refusals: an audio stream, a missing pdd config, a foreign grid. One JSON line; exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import build, sampler
from ..h3 import pack as video_pack

LAYERS, D, HEADS, FFN, TEXT, GRID = 2, 256, 2, 384, 192, 33
BF, F16, F32 = torch.bfloat16, torch.float16, torch.float32


def synthetic(g: torch.Generator) -> dict[str, torch.Tensor]:
    """The published key set at small shapes, each in the checkpoint's dtype."""
    r = lambda *s, dt=BF, sc=0.05: (torch.randn(*s, generator=g) * sc).to(dt)
    inner = HEADS * 128
    t = {"adaln_t_table": r(GRID, 8, dt=F32, sc=1.0), "rope.inv_freq": r(16, dt=F32, sc=1.0),
         "video_patch_proj.weight": r(D, 96, dt=F32), "video_patch_proj.bias": r(D, dt=F32),
         "condition_proj.weight": r(D, TEXT), "condition_proj.bias": r(D),
         "final_layer.adaln_proj.linear.weight": r(2 * D, 8, dt=F16), "final_layer.adaln_proj.linear.bias": r(2 * D),
         "final_layer.norm.weight": r(D, sc=1.0), "final_layer.video_out.weight": r(576, D, dt=F32),
         "final_layer.video_out.bias": r(576, dt=F32), "token_refiner.final_norm.weight": r(D, sc=1.0)}
    for pre, n in (("token_refiner.blocks.", 2), ("blocks.", LAYERS)):
        for i in range(n):
            p = f"{pre}{i}."
            t |= {p + "norm1.weight": r(D, sc=1.0), p + "norm2.weight": r(D, sc=1.0),
                  p + "attn.q_norm.weight": r(128, sc=1.0), p + "attn.k_norm.weight": r(128, sc=1.0),
                  p + "attn.qkv_proj.weight": r(3 * inner, D), p + "attn.out_proj.weight": r(D, inner),
                  p + "mlp.fc1.weight": r(2 * FFN, D), p + "mlp.fc2.weight": r(D, FFN)}
            if pre == "blocks.":
                t |= {p + "adaln_proj.linear.weight": r(18 * D, 8, dt=F16), p + "adaln_proj.linear.bias": r(18 * D),
                      p + "attn.comfy_attention.config": torch.frombuffer(bytearray(b'[{"attention": "comfy_kitchen_int8"}]'),
                                                                          dtype=torch.uint8).clone()}
    return t


def contract(src: dict[str, torch.Tensor]) -> dict[str, tuple[str, torch.dtype]]:
    """build.py's table, written out independently: pack name -> (source key, pack dtype)."""
    side32 = ["adaln_t_table", "rope.inv_freq", "video_patch_proj.weight", "video_patch_proj.bias",
              "final_layer.video_out.weight", "final_layer.video_out.bias", "final_layer.adaln_proj.linear.weight",
              "final_layer.adaln_proj.linear.bias"]
    c = {k: (k, F32) for k in side32}
    c["final_layer.norm.weight"] = ("final_layer.norm.weight", BF)
    c |= {k: (k, BF) for k in src if k.startswith(("condition_proj.", "token_refiner."))}
    for i in range(LAYERS):
        p, q = f"blocks.{i}.", f"L{i}."
        c |= {q + "norm1": (p + "norm1.weight", F32), q + "norm2": (p + "norm2.weight", F32),
              q + "q_norm": (p + "attn.q_norm.weight", BF), q + "k_norm": (p + "attn.k_norm.weight", BF),
              q + "ada_w": (p + "adaln_proj.linear.weight", F32), q + "ada_b": (p + "adaln_proj.linear.bias", F32),
              q + "qkv.weight": (p + "attn.qkv_proj.weight", BF), q + "out.weight": (p + "attn.out_proj.weight", BF),
              q + "fc1.weight": (p + "mlp.fc1.weight", BF), q + "fc2.weight": (p + "mlp.fc2.weight", BF)}
    return c


def config(grid=sampler.PDD_SIGMAS, steps=6) -> dict:
    return {"config": json.dumps({"transformer": {"pdd_steps": steps, "pdd_sigmas": list(grid)}})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, help="a scratch directory (created; its h3i-build-test/ is replaced)")
    a = ap.parse_args()
    root = Path(a.scratch) / "h3i-build-test"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    bad: list[str] = []
    g = torch.Generator().manual_seed(3)
    src = synthetic(g)
    ck = root / "minimax_h3_image_pruned_bf16.safetensors"
    save_file(src, str(ck), metadata=config())
    man = build.write(ck, root / "pack", {"test": True})

    want = contract(src)
    with safe_open(str(root / "pack" / "weights.safetensors"), framework="pt") as f:
        names = set(f.keys())
        if names != set(want):
            bad.append(f"names: extra {sorted(names - set(want))[:5]}, missing {sorted(set(want) - names)[:5]}")
        for n in sorted(names & set(want)):
            key, dt = want[n]
            t = f.get_tensor(n)
            ref = src[key].to(dt)
            if t.dtype != dt or tuple(t.shape) != tuple(ref.shape) or not torch.equal(t, ref):
                bad.append(f"{n}: {t.dtype} {tuple(t.shape)} vs {dt} {tuple(ref.shape)}, equal {torch.equal(t.to(dt), ref)}")
            sha = hashlib.sha256(t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()).hexdigest()
            m = man["tensors"].get(n, {})
            if m.get("sha256") != sha or m.get("dtype") != str(dt).removeprefix("torch.") or m.get("shape") != list(t.shape):
                bad.append(f"{n}: manifest entry {m}")
    on_disk = json.loads((root / "pack" / "manifest.json").read_text())
    kinds = {f"L{i}.{x}": "bf16" for i in range(LAYERS) for x in ("qkv", "out", "fc1", "fc2")}
    for k, v in {"format": "stk-pack/1", "model": "minimax-h3-image", "precision": "bf16", "kinds": kinds, "acts": {},
                 "pdd_steps": 6, "pdd_sigmas": list(sampler.PDD_SIGMAS)}.items():
        if on_disk.get(k) != v:
            bad.append(f"manifest {k}: {on_disk.get(k)!r}")
    # the video pack's names for every tensor the two DiTs share
    shared = [k for k in video_pack.SIDE_F32 if "audio" not in k]
    if any(k not in names or want[k][1] != F32 for k in shared):
        bad.append("a shared side tensor is not under the video pack's name / dtype")
    if any("audio" in n or "comfy_attention" in n for n in names):
        bad.append("audio or attention-selector tensors in the pack")

    # the pack drives the twin: same bits as the source tensors, one CPU step
    from .cpu_ops import CpuOps
    from .dit import H3ImageDit

    ops = CpuOps(torch.float32)
    a_ = H3ImageDit.from_pack(root / "pack", ops=ops, device="cpu")
    b_ = H3ImageDit({k: v for k, v in src.items() if "comfy_attention" not in k}, ops=ops, device="cpu")
    text = torch.randn(1, 5, TEXT, generator=g)
    x = torch.randn(1, 24, 1, 8, 8, generator=g)
    sig = sampler.sigmas(sampler.PDD_SIGMAS, 3)
    st = sampler.step(sampler.PDD_SIGMAS, sig, 1)
    a_.prepare(text)
    b_.prepare(text)
    if not torch.equal(a_(x, st), b_(x, st)):
        bad.append("from_pack's step differs from the checkpoint's")

    # refusals
    refusals = 0
    for name, tensors, meta in (("audio", src | {"audio_patch_proj.weight": torch.zeros(4, 4)}, config()),
                                ("no config", src, None), ("foreign grid", src, config(grid=(1.0, 0.5, 0.0), steps=2))):
        p = root / f"bad-{name.replace(' ', '-')}.safetensors"
        save_file(tensors, str(p), metadata=meta)
        try:
            build.write(p, root / "bad-pack", {})
            bad.append(f"accepted a checkpoint with {name}")
        except ValueError:
            refusals += 1
    shutil.rmtree(root)
    print(json.dumps({"ok": not bad, "tensors": len(names), "linears": len(on_disk["kinds"]), "refusals": refusals,
                      "errors": bad[:20]}))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
