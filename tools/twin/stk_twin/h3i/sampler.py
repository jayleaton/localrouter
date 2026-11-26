"""MiniMax H3-Image's PDD schedule and head plan as ComfyUI e9afbccc computes them (docs/dev/H3-IMAGE.md section 1.4), in
numpy float32 / Python float64 exactly where ComfyUI's torch and Python arithmetic round. Pure host arithmetic: the twin's
sampler and the Zig engine's `pdd.zig` run the same plan, and `test_pdd.py` checks it bit for bit against ComfyUI's own
code (on the CPU, no GPU, no checkpoint).

  table(grid)          MiniMaxH3ImageModelSampling.sigmas: sigma(t) at t = (i / 1000) * 1000, i = 1 .. 1000, with
                       sigma(t) = lerp(g[k], g[k + 1], pos - k), pos = clamp(t / 1000, 0, 1) * n, k = min(floor(pos), n - 1)
                       over the ascending grid g (comfy/ldm/minimax/sampling.py:20-25, model_sampling.py:308-312)
  sigmas(grid, steps)  the `simple` scheduler on that table (comfy/samplers.py:645-652)
  step(grid, sig, i)   what step i's model call derives: the timestep the model gets (fp32 sigma * 1000), its float64
                       sigma and t = 1 - sigma, and the PDD heads: (start, stop) by the nearest grid point and their fp32
                       dt weights (comfy/ldm/minimax/model.py:349-380, 869-871)
  plan(grid, steps)    sigmas + every step

`python -m stk_twin.h3i.sampler --fixture PATH` writes the JSON the Zig host test compares against.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction

import numpy as np

F32 = np.float32
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)  # the bf16 file's
FIXTURE_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 30, 40, 50)


def fma(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """a * b + c with one rounding to fp32 (round to nearest even), exactly: Zig's @mulAdd(f32, ...)."""
    r = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    f = F32(float(r))                                     # within an ulp of the answer; pick the nearest fp32 to r
    best = None
    for cand in (np.nextafter(f, F32(-np.inf)), f, np.nextafter(f, F32(np.inf))):
        d = abs(Fraction(float(cand)) - r)
        even = (int(np.asarray(cand, F32).view(np.uint32)) & 1) == 0
        if best is None or d < best[0] or (d == best[0] and even):
            best = (d, cand)
    return F32(best[1])


def lerp(a: np.float32, b: np.float32, w: np.float32) -> np.float32:
    """ATen's vectorized CPU lerp in fp32, the kernel that fills the table: |w| < 0.5 ? fma(w, b - a, a) :
    fma(-(b - a), 1 - w, b) (test_pdd.py: every entry of the table equal to torch's; the unfused form differs in 43)."""
    d = F32(b - a)
    return fma(w, d, a) if abs(w) < F32(0.5) else fma(F32(-d), F32(F32(1.0) - w), b)


def mix(rows: np.ndarray, brows: np.ndarray, heads: list[int], w: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """The step's head: row block 0 plus the dt-weighted offsets, fp32. ComfyUI's einsum leaves the order of the sum
    over heads to its backend; ours is fixed: s = w_0 r_0, then s = s + w_k r_k (each product and sum rounded), then
    rows[0] + s. A step that spans one grid interval (every step of a schedule of at least pdd_steps steps) has one
    head and w = [1.0], so the result is exactly ComfyUI's."""
    if not heads:
        return rows[0].astype(F32), brows[0].astype(F32)
    s = (F32(w[0]) * rows[heads[0]]).astype(F32)
    sb = (F32(w[0]) * brows[heads[0]]).astype(F32)
    for k, wk in zip(heads[1:], w[1:]):
        s = (s + (F32(wk) * rows[k]).astype(F32)).astype(F32)
        sb = (sb + (F32(wk) * brows[k]).astype(F32)).astype(F32)
    return (rows[0] + s).astype(F32), (brows[0] + sb).astype(F32)


def table(grid: tuple[float, ...] = PDD_SIGMAS) -> np.ndarray:
    g = np.array(grid, dtype=F32)[::-1]                   # ascending: index 0 is sigma 0
    n = len(g) - 1
    out = np.empty(1000, dtype=F32)
    for i in range(1, 1001):
        ts = F32(F32(i) / F32(1000)) * F32(1000)          # (arange(1, 1001) / 1000) * 1000: int / int -> fp32, then * 1000
        pos = F32(min(max(F32(ts / F32(1000)), F32(0.0)), F32(1.0)) * F32(n))
        k = min(int(math.floor(pos)), n - 1)
        out[i - 1] = lerp(g[k], g[k + 1], F32(pos - F32(k)))
    return out


def sigmas(grid: tuple[float, ...], steps: int) -> list[float]:
    t = table(grid)
    ss = len(t) / steps
    return [float(t[-(1 + int(x * ss))]) for x in range(steps)] + [0.0]


def step(grid: tuple[float, ...], sig: list[float], i: int) -> dict:
    """Step i of schedule `sig` (fp32 values): the model's timestep and float64 sigma / t, and the PDD heads."""
    n = len(grid) - 1
    s32 = F32(sig[i])
    timestep = F32(s32 * F32(1000.0))                     # model_base: sigma * multiplier, fp32
    sigma = float(timestep) / 1000.0                      # model.py:869: float(timestep) / 1000.0
    t = 1.0 - sigma
    # FinalLayer.forward: the step whose sample sigma is nearest (fp32 tensor - float -> fp32), the next one, then the
    # nearest grid points in float64 (min over k: the first on a tie)
    diffs = [abs(F32(F32(s) - F32(sigma))) for s in sig]
    j = min(range(len(sig)), key=lambda k: (diffs[k], k))
    nxt = float(F32(sig[min(j + 1, len(sig) - 1)]))
    start, stop = (min(range(n + 1), key=lambda k: abs(grid[k] - float(s))) for s in (sigma, nxt))
    start = min(start, n - 1)
    stop = max(stop, start + 1)
    dt = [grid[k] - grid[k + 1] for k in range(start, stop)]   # Python floats
    tot = sum(dt)
    w = [float(F32(F32(d) / F32(tot))) for d in dt]       # torch.tensor(dt, fp32) / sum(dt): the scalar meets fp32 as fp32
    first = max(start, 1)
    return {"timestep": float(timestep), "sigma": sigma, "t": t, "start": start, "stop": stop,
            "heads": list(range(first, stop)), "w": w[first - start:]}


def plan(grid: tuple[float, ...], steps: int) -> dict:
    sig = sigmas(grid, steps)
    return {"steps": steps, "sigmas": sig, "plan": [step(grid, sig, i) for i in range(steps)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="where to write the Zig host test's JSON")
    a = ap.parse_args()
    out = {"pdd_sigmas": list(PDD_SIGMAS), "table": [float(x) for x in table()], "plans": [plan(PDD_SIGMAS, s) for s in FIXTURE_STEPS]}
    with open(a.fixture, "w") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
