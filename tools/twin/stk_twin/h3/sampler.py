"""MiniMax H3's sampler as ComfyUI runs it (`res_multistep`, eta 0, cfg 1: one model call a step) on the packed fp32
state [video | audio] (the sampler's carried audio). Per step the scalars come from fp32 tensor ops, with ComfyUI's
CUDA log / expm1 / exp replaced by LocalRouter's portable fdlibm ports (`smath`, in f64, rounded to f32; the Zig
engine's `sampler.zig` is the same); the elementwise updates are kernels with ComfyUI's per-op roundings
(`ops.cu` h3_denoise / h3_euler32 / h3_res2). The schedule is ComfyUI's `simple` on ModelSamplingAV (`sigmas`).
"""

from __future__ import annotations

import numpy as np

from .. import smath

F32 = np.float32


def _t(s: np.float32) -> np.float32:
    """t_fn(sigma) = -log(sigma), fp32."""
    return -F32(smath.log(float(s)))


def plan(sigmas: list[float]) -> list[dict]:
    """Per step i: the kind ("euler" or "res2") and its fp32 scalars, from the schedule (fp32 values)."""
    sg = [F32(s) for s in sigmas]
    steps = []
    for i in range(len(sg) - 1):
        down = sg[i + 1]                                     # eta 0: sigma_down = sigma_next, sigma_up = 0
        if down == F32(0.0) or i == 0:
            steps.append({"kind": "euler", "sigma": float(sg[i]), "dt": float(down - sg[i])})
            continue
        t, t_old, t_next, t_prev = _t(sg[i]), _t(sg[i]), _t(down), _t(sg[i - 1])   # old_sigma_down = sigma_i
        h = t_next - t
        c2 = (t_prev - t_old) / h
        mt = -h                                              # phi*_fn(-h)
        phi1 = F32(smath.expm1(float(mt))) / mt
        phi2 = (phi1 - F32(1.0)) / mt
        b1 = phi1 - phi2 / c2
        b2 = phi2 / c2
        e = F32(smath.exp(float(-h)))                        # sigma_fn(h) = exp(-h)
        steps.append({"kind": "res2", "sigma": float(sg[i]), "e": float(e), "h": float(h), "b1": float(b1), "b2": float(b2)})
    return steps


def sigmas(steps: int, shift: float = 12.0) -> list[float]:
    """ComfyUI's `simple` scheduler on ModelSamplingAV(shift): the 1000-entry fp32 table sigma(t) = shift t /
    (1 + (shift - 1) t) at t = ((i / 1000) * 1000) / 1000, i = 1..1000 (fp32 ops), then entries -(1 + int(x * ss)),
    and 0."""
    i = np.arange(1, 1001, dtype=np.float32)
    t = ((i / F32(1000.0)) * F32(1000.0)) / F32(1000.0)
    table = (F32(shift) * t) / (F32(1.0) + F32(shift - 1.0) * t)
    ss = len(table) / steps
    return [float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0]
