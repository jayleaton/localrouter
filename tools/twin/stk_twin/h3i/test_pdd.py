"""CPU check: python -m stk_twin.h3i.test_pdd --comfy COMFYUI_DIR [--fixture PATH] (no GPU, no checkpoint).

The PDD schedule and head plan (`sampler.py`) against ComfyUI e9afbccc's own code, its source executed as written
(extracted from the files, with stubs only for what the extracted lines call on `self`):
  - MiniMaxH3ImageModelSampling.sigma (comfy/ldm/minimax/sampling.py) through ModelSamplingDiscreteFlow.set_parameters
    (comfy/model_sampling.py): the 1000-entry table, every bit;
  - simple_scheduler (comfy/samplers.py): the sigmas at every step count in sampler.FIXTURE_STEPS;
  - FinalLayer.forward's head pick and _pdd_head's mix (comfy/ldm/minimax/model.py) on a random fp32 head bank: start,
    stop, the dt weights and the mixed weight and bias, every bit, for every step of every schedule.
With --fixture, also that the Zig host test's JSON (src/engines/minimax_h3/pdd_plan.json) is what sampler.py writes.
One JSON line; exit 1 on any difference.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import sampler

F32 = np.float32


def extract(path: Path, cls: str | None, fn: str, scope: dict):
    """Function `fn` (of class `cls`, or module level) compiled from ComfyUI's source file unchanged, in `scope`."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == fn)
    exec(compile(ast.Module([node], []), str(path), "exec"), scope)
    return scope[fn]


def bits(x) -> bytes:
    return np.asarray(x, dtype=F32).tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="ComfyUI at e9afbccc")
    ap.add_argument("--fixture", help="src/engines/minimax_h3/pdd_plan.json to check")
    a = ap.parse_args()
    root = Path(a.comfy)
    torch.manual_seed(0)
    bad: list[str] = []
    grid = sampler.PDD_SIGMAS

    # the table: sigma() through set_parameters, on a stand-in holding what they read and write
    ms = SimpleNamespace()
    ms.grid = torch.tensor(grid, dtype=torch.float32).flip(0)
    ms.register_buffer = lambda name, t: setattr(ms, name, t)
    sig_fn = extract(root / "comfy/ldm/minimax/sampling.py", "MiniMaxH3ImageModelSampling", "sigma", {"torch": torch})
    ms.sigma = lambda timestep: sig_fn(ms, timestep)
    set_params = extract(root / "comfy/model_sampling.py", "ModelSamplingDiscreteFlow", "set_parameters", {"torch": torch})
    set_params(ms)
    if bits(ms.sigmas.numpy()) != bits(sampler.table(grid)):
        d = int(np.count_nonzero(ms.sigmas.numpy() != sampler.table(grid)))
        bad.append(f"table: {d} of 1000 entries differ")

    simple = extract(root / "comfy/samplers.py", None, "simple_scheduler", {"torch": torch})

    # FinalLayer.forward's pick and _pdd_head's mix, with stubs for the parts of the layer that are not the head
    n = len(grid) - 1
    hidden, vdim = 8, 4
    head = SimpleNamespace(weight=torch.randn(vdim * n, hidden, dtype=torch.float32), bias=torch.randn(vdim * n, dtype=torch.float32))
    seen: dict = {}

    def linear(x, w, b):
        seen["w"], seen["b"] = w.clone(), b.clone()
        return x @ w.T + b

    @contextlib.contextmanager
    def cast(h, _x, offloadable=True):
        yield h.weight, h.bias

    comfy = SimpleNamespace(ops=SimpleNamespace(CastBiasWeightContext=cast))
    nn = SimpleNamespace(functional=SimpleNamespace(linear=linear))
    pdd_head = extract(root / "comfy/ldm/minimax/model.py", None, "_pdd_head", {"torch": torch, "comfy": comfy, "nn": nn})

    def pdd_spy(hd, h, g, start, stop):
        seen["start"], seen["stop"] = start, stop
        return pdd_head(hd, h, g, start, stop)

    fwd = extract(root / "comfy/ldm/minimax/model.py", "FinalLayer", "forward", {"torch": torch, "_pdd_head": pdd_spy})
    layer = SimpleNamespace(adaln_proj=lambda t: (None, None), pdd_steps=n, _modulate=lambda x, sh, sc, seg: x, video_out=head, audio_out=None)
    x = torch.randn(3, hidden, dtype=torch.float32)
    rows = head.weight.numpy().reshape(n, vdim, hidden)
    brows = head.bias.numpy().reshape(n, vdim)

    checked = 0
    multi_err, multi_same, multi_n = [0.0], [0], [0]
    for steps in sampler.FIXTURE_STEPS:
        ref = simple(SimpleNamespace(sigmas=ms.sigmas), steps)
        mine = sampler.plan(grid, steps)
        if bits(ref.numpy()) != bits(mine["sigmas"]):
            bad.append(f"sigmas at {steps} steps differ")
            continue
        for i, st in enumerate(mine["plan"]):
            timestep = torch.tensor([ref[i].item()], dtype=torch.float32) * 1000  # model_sampling.timestep: sigma * multiplier
            sigma = float(timestep.flatten()[0]) / 1000.0  # model.py:869, as written
            if st["timestep"] != float(timestep[0]) or st["sigma"] != sigma or st["t"] != 1.0 - sigma:
                bad.append(f"{steps} steps, step {i}: timestep / sigma / t")
            seen.clear()
            fwd(layer, x, None, None, None, sigma, ref, (list(grid),))
            if (seen["start"], seen["stop"]) != (st["start"], st["stop"]):
                bad.append(f"{steps} steps, step {i}: heads ({seen['start']}, {seen['stop']}) vs ({st['start']}, {st['stop']})")
                continue
            w, b = sampler.mix(rows, brows, st["heads"], st["w"])
            same = bits(seen["w"].numpy()) == bits(w) and bits(seen["b"].numpy()) == bits(b)
            if len(st["heads"]) <= 1 and not same:
                bad.append(f"{steps} steps, step {i}: mixed head differs (heads {st['heads']}, w {st['w']})")
            elif len(st["heads"]) > 1:  # einsum's order is its backend's: ours is fixed, so compare within rounding
                err = float(np.max(np.abs(seen["w"].numpy() - w)))
                multi_err[0] = max(multi_err[0], err)
                multi_same[0] += same
                multi_n[0] += 1
                if err > 1e-6:
                    bad.append(f"{steps} steps, step {i}: multi-head mix off by {err}")
            checked += 1

    if a.fixture:
        want = {"pdd_sigmas": list(grid), "table": [float(v) for v in sampler.table(grid)],
                "plans": [sampler.plan(grid, s) for s in sampler.FIXTURE_STEPS]}
        if json.loads(Path(a.fixture).read_text()) != want:
            bad.append(f"{a.fixture} is not what sampler.py writes (python -m stk_twin.h3i.sampler --fixture ...)")

    multi = sorted({len(st["heads"]) for s in sampler.FIXTURE_STEPS for st in sampler.plan(grid, s)["plan"]})
    print(json.dumps({"ok": not bad, "steps_checked": checked, "heads_a_step": multi, "multi_head_steps": multi_n[0], "multi_head_bit_equal": multi_same[0], "multi_head_max_err": multi_err[0], "torch": torch.__version__, "errors": bad[:20]}))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
