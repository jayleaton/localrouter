"""CPU check: python -m stk_twin.h3.test_vision_position --comfy COMFYUI_DIR (no GPU, no checkpoint).
The vision tower's absolute positions, ours against ComfyUI's own `fast_pos_embed_interpolate` (its source is executed
as written, on a random bf16 48x48 table): our host tables (`vision_tables`: rows, ids, weights) and the arithmetic of
`h3v_position` (te_vision.cu: bf16 weights, bf16 products, a left-to-right bf16 sum), emulated here, must give the same
bits at every canvas the engine accepts. One JSON line; exit 1 on any difference.
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .te_vision import vision_tables

GRIDS = ((6, 8), (8, 6), (28, 48), (48, 28), (14, 14), (32, 52), (96, 64), (2, 2))


def comfy_interpolate(comfy):
    """ComfyUI 0.37.0's Qwen35VisionModel.fast_pos_embed_interpolate, compiled from its source file unchanged."""
    path = Path(comfy) / "comfy/text_encoders/qwen35.py"
    tree = ast.parse(path.read_text())
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "Qwen35VisionModel":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "fast_pos_embed_interpolate":
                    scope = {"torch": torch}
                    exec(compile(ast.Module([fn], []), str(path), "exec"), scope)
                    return scope[fn.name]
    raise SystemExit(f"fast_pos_embed_interpolate not found in {path}")


def bf16(x):
    """Round float32 to the nearest bf16, ties to even, as h3v_bf16 does (finite values)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def ours(table, ids, weights):
    t = bf16(table)
    w = bf16(weights)
    s = bf16(t[ids[0]] * w[0][:, None])
    for k in range(1, 4):
        s = bf16(s + bf16(t[ids[k]] * w[k][:, None]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=os.environ.get("COMFY", ""))
    ap.add_argument("--dim", type=int, default=64)
    a = ap.parse_args()
    if not a.comfy:
        ap.error("--comfy or COMFY is required")
    interpolate = comfy_interpolate(a.comfy)
    g = torch.Generator().manual_seed(7)
    emb = torch.nn.Embedding(48 * 48, a.dim, dtype=torch.bfloat16)
    with torch.no_grad():
        emb.weight.copy_(torch.randn(48 * 48, a.dim, generator=g).to(torch.bfloat16))
    model = SimpleNamespace(pos_embed=emb, num_grid_per_side=48, spatial_merge_size=2)
    table = emb.weight.detach().float().numpy()
    results, ok = {}, True
    for gh, gw in GRIDS:
        with torch.no_grad():
            ref = interpolate(model, torch.tensor([[1, gh, gw]])).float().numpy()
        _, _, ids, weights = vision_tables(gh, gw)
        got = ours(table, ids.numpy(), weights.numpy())
        differ = int((got != ref).sum()) if got.shape == ref.shape else -1
        results[f"{gh}x{gw}"] = {"differ": differ, "max_abs": float(np.abs(got - ref).max()) if differ >= 0 else None}
        ok &= differ == 0
    print(json.dumps({"vision_position": results, "pass": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
