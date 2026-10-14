"""The native Kandinsky-6 PiFlow sampler in Python: src/engines/kandinsky6/sampler.zig's host scalars line by line in
numpy float32 (one rounding an op, as Zig rounds them: no contraction, true division, NaN-propagating clamps,
half-to-even rounding), and kernels/cuda/kandinsky6/sampler/policy.cu from the fatbin the Zig build embeds, launched
through the CUDA driver as sampler.zig `Kernels` launches it: same image, argument types and order, grid and block.
Same image, arguments and grid: the twin's state bits are the engine's.

Image: `Policy(fatbin=PATH)` or STK_K6_SAMPLER_FATBIN=PATH (the build's policy.cu fatbin). Launches go on torch's
current stream in torch's primary context. No kandinsky6 DiT module is imported.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

F = np.float32
THREADS = 256  # sampler.zig `threads`
I32_MAX = 2**31 - 1


@dataclass(frozen=True)
class Config:
    """sampler.zig Config; `distill` is pro-distill.yaml's piflow block."""

    num_steps: int = 10
    shift: float = 5.0
    eps: float = 1e-6
    final_step_size_scale: float = 0.5
    num_policy_substeps: int = 128
    n_grid: int = 10

    def validate(self) -> None:
        if self.num_steps < 1 or self.num_policy_substeps < 1 or self.n_grid < 2:
            raise ValueError("InvalidConfig")
        for v in (self.shift, self.eps, self.final_step_size_scale):
            if not math.isfinite(v):
                raise ValueError("InvalidConfig")
        if not (self.shift > 0 and self.eps > 0 and 0 < self.final_step_size_scale <= 1):
            raise ValueError("InvalidConfig")
        if not F(self.eps) > 0 or not np.isfinite(F(self.shift)) or self.num_policy_substeps > 1 << 24:
            raise ValueError("InvalidConfig")

    def base_segment(self) -> float:
        fss = max(self.final_step_size_scale, self.eps)
        return 1.0 / (float(self.num_steps) - (1.0 - fss))


def f32(v: float) -> np.float32:
    return F(v)


def clamp_min(v: np.float32, lo) -> np.float32:
    return v if np.isnan(v) else np.maximum(v, F(lo))


def clamp(v: np.float32, lo, hi) -> np.float32:
    return v if np.isnan(v) else np.minimum(np.maximum(v, F(lo)), F(hi))


def shift_timesteps(t: np.float32, shift: float) -> np.float32:
    return (t * f32(shift)) / (t * f32(shift - 1.0) + F(1.0))


def unwarp(sigma: np.float32, shift: float) -> np.float32:
    return sigma / (sigma * f32(1.0 - shift) + f32(shift))


def finite(d: dict) -> dict:
    for k, v in d.items():
        if isinstance(v, np.float32) and not np.isfinite(v):
            raise ValueError(f"NonFinite {k}")
    return d


def schedule(c: Config):
    """sampler.zig Schedule: one dict a step (its fields)."""
    c.validate()
    base = c.base_segment()
    tau = F(1.0)
    eps = f32(c.eps)
    for i in range(c.num_steps):
        final = i == c.num_steps - 1
        seg = f32(base * (max(c.final_step_size_scale, c.eps) if final else 1.0))
        tau_src = tau
        tau_dst = clamp_min(tau_src - seg, eps)
        sigma = shift_timesteps(tau_src, c.shift)
        raw_src = unwarp(sigma, c.shift)
        raw_dst = clamp_min(raw_src - seg, 0)
        delta = tau_src - tau_dst
        scaled = delta * F(c.num_policy_substeps)
        if not np.isfinite(scaled):
            raise ValueError("NonFinite")
        n = np.maximum(np.rint(scaled), F(1.0))
        yield finite({"index": i, "final": final, "segment": seg, "tau_src": tau_src, "tau_dst": tau_dst,
                      "sigma": sigma, "time": sigma * F(1000.0), "policy_raw_src": raw_src,
                      "policy_raw_dst": raw_dst, "policy_segment": clamp_min(raw_src - raw_dst, eps),
                      "substeps": int(n), "substep_size": delta / n})
        tau = tau_dst


def rollout(c: Config, st: dict):
    """sampler.zig Rollout: one dict a substep."""
    raw, sigma = st["tau_src"], st["sigma"]
    for k in range(st["substeps"]):
        t = (unwarp(sigma, c.shift) - st["policy_raw_dst"]) / st["policy_segment"]
        if not np.isfinite(t):
            raise ValueError("NonFinite")
        tc = clamp(t, 0, 1) * F(c.n_grid - 1)
        g0 = int(min(max(np.floor(tc), F(0)), F(c.n_grid - 2)))
        raw_next = clamp_min(raw - st["substep_size"], 0)
        sigma_next = shift_timesteps(raw_next, c.shift)
        yield finite({"index": k, "sigma": sigma, "t": t, "g0": g0, "w0": F(g0 + 1) - tc, "w1": tc - F(g0),
                      "sigma_clamped": clamp_min(sigma, f32(c.eps)), "raw_next": raw_next, "sigma_next": sigma_next,
                      "dsigma": sigma_next - sigma})
        raw, sigma = raw_next, sigma_next


def u32(v: np.float32) -> int:
    return int(np.asarray(v, dtype=F).view(np.uint32))


# ----------------------------------------------------------------------------------------------------- device
DTYPE = {"bfloat16": 0, "float32": 1}  # sampler.zig Dtype


def _dtype(t) -> int:
    name = str(t.dtype).replace("torch.", "")
    if name not in DTYPE:
        raise TypeError(f"UnsupportedDtype {t.dtype}")
    return DTYPE[name]


def _driver() -> ctypes.CDLL:
    cu = ctypes.CDLL("libcuda.so.1")
    P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
    for name, args in {
        "cuInit": [U], "cuDeviceGet": [ctypes.POINTER(I), I], "cuDevicePrimaryCtxRetain": [ctypes.POINTER(P), I],
        "cuCtxGetCurrent": [ctypes.POINTER(P)], "cuCtxSetCurrent": [P],
        "cuModuleLoadData": [ctypes.POINTER(P), P], "cuModuleUnload": [P],
        "cuModuleGetFunction": [ctypes.POINTER(P), P, ctypes.c_char_p],
        "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, ctypes.POINTER(P), ctypes.POINTER(P)],
        "cuGetErrorString": [I, ctypes.POINTER(ctypes.c_char_p)],
    }.items():
        fn = getattr(cu, name)
        fn.argtypes = args
        fn.restype = I
    return cu


class Policy:
    """sampler.zig `Kernels`: the policy.cu module and its six functions, in torch's primary context."""

    SUBSTEP = {(0, 0): "k6_piflow_substep_bf16_bf16", (0, 1): "k6_piflow_substep_bf16_f32",
               (1, 0): "k6_piflow_substep_f32_bf16", (1, 1): "k6_piflow_substep_f32_f32"}
    VIDEO_INPUT = {0: "k6_piflow_video_input_bf16", 1: "k6_piflow_video_input_f32"}

    def __init__(self, fatbin: str | None = None):
        import torch

        self.torch = torch
        path = fatbin or os.environ.get("STK_K6_SAMPLER_FATBIN")
        if not path:
            raise FileNotFoundError("set STK_K6_SAMPLER_FATBIN=PATH (the build's policy.cu fatbin) or pass fatbin=")
        image = Path(path).read_bytes()
        if not image:
            raise RuntimeError("empty kernel image (a host-only build)")  # Kernels.load: error.Invalid
        self.digest = hashlib.sha256(image).hexdigest()
        torch.cuda.init()
        torch.cuda.current_stream()
        self.cu = cu = _driver()
        self._check(cu.cuInit(0), "cuInit")
        dev, prim, cur = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()
        self._check(cu.cuDeviceGet(ctypes.byref(dev), torch.cuda.current_device()), "cuDeviceGet")
        self._check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(prim), dev), "cuDevicePrimaryCtxRetain")
        self._check(cu.cuCtxGetCurrent(ctypes.byref(cur)), "cuCtxGetCurrent")
        if cur.value != prim.value:
            self._check(cu.cuCtxSetCurrent(prim), "cuCtxSetCurrent")
        buf = (ctypes.c_uint64 * (len(image) // 8 + 2))()  # 8-byte aligned, zero padded
        ctypes.memmove(buf, image, len(image))
        self.mod = ctypes.c_void_p()
        self._check(cu.cuModuleLoadData(ctypes.byref(self.mod), ctypes.cast(buf, ctypes.c_void_p)), "cuModuleLoadData")
        self.fns = {n: self._fn(n) for n in (*self.SUBSTEP.values(), *self.VIDEO_INPUT.values())}
        self.count = 0

    def _check(self, r: int, what: str) -> None:
        if r != 0:
            s = ctypes.c_char_p()
            msg = s.value.decode() if self.cu.cuGetErrorString(r, ctypes.byref(s)) == 0 and s.value else ""
            raise RuntimeError(f"{what}: CUresult {r} {msg}")

    def _fn(self, name: str) -> ctypes.c_void_p:
        f = ctypes.c_void_p()
        self._check(self.cu.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode()), f"cuModuleGetFunction {name}")
        return f

    def _go(self, name: str, n: int, args: list) -> None:
        blocks = (n + THREADS - 1) // THREADS
        if n == 0 or blocks > I32_MAX:  # Kernels.grid
            raise ValueError("ViewMismatch")
        params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        stream = ctypes.c_void_p(self.torch.cuda.current_stream().cuda_stream)
        self._check(self.cu.cuLaunchKernel(self.fns[name], blocks, 1, 1, THREADS, 1, 1, 0, stream, params, None),
                    f"cuLaunchKernel {name}")
        self.count += 1

    def close(self) -> None:
        self.cu.cuModuleUnload(self.mod)

    @staticmethod
    def _flat(t):
        if not t.is_cuda or not t.is_contiguous():
            raise ValueError("ViewMismatch: a contiguous CUDA tensor is required")
        return t

    def substep(self, c: Config, st: dict, sub: dict, src, pred, out) -> None:
        """Kernels.substep: `pred` grid-major [n_grid, n] (any shape with n_grid * n elements), `out` f32 n."""
        src, pred, out = self._flat(src), self._flat(pred), self._flat(out)
        n = src.numel()
        if pred.numel() != n * c.n_grid or out.numel() != n or str(out.dtype) != "torch.float32":
            raise ValueError("ViewMismatch")
        if out.data_ptr() == src.data_ptr() or sub["g0"] + 1 >= c.n_grid:
            raise ValueError("ViewMismatch")
        args = [ctypes.c_uint64(src.data_ptr()), ctypes.c_uint64(pred.data_ptr()), ctypes.c_uint64(out.data_ptr()),
                ctypes.c_int64(n), ctypes.c_int32(1 if sub["index"] == 0 else 0), ctypes.c_int32(sub["g0"])]
        args += [ctypes.c_float(float(v)) for v in (st["sigma"], sub["w0"], sub["w1"], sub["sigma_clamped"], sub["dsigma"])]
        self._go(self.SUBSTEP[(_dtype(src), _dtype(pred))], n, args)

    def rollout(self, c: Config, st: dict, src, pred, out) -> None:
        for sub in rollout(c, st):
            self.substep(c, st, sub, src, pred, out)

    def advance(self, c: Config, st: dict, video: tuple, audio: tuple) -> None:
        """Kernels.advance: video then audio, both required. Each a (src, pred, out)."""
        self.rollout(c, st, *video)
        self.rollout(c, st, *audio)

    def video_input(self, state, rows: int, ch: int, first, rows_first: int, out) -> None:
        """Kernels.videoInput: out [rows, 2 * ch + 1] in the state's dtype."""
        state, out = self._flat(state), self._flat(out)
        if state.numel() != rows * ch or out.dtype != state.dtype or out.numel() != rows * (2 * ch + 1) or ch == 0:
            raise ValueError("ViewMismatch")
        if first is not None:
            first = self._flat(first)
            if first.numel() != rows_first * ch or rows_first > rows:
                raise ValueError("ViewMismatch")
        args = [ctypes.c_uint64(state.data_ptr()), ctypes.c_uint64(first.data_ptr() if first is not None else 0),
                ctypes.c_int32(_dtype(first) if first is not None else -1), ctypes.c_uint64(out.data_ptr()),
                ctypes.c_int64(rows), ctypes.c_int32(ch), ctypes.c_int64(rows_first if first is not None else 0)]
        self._go(self.VIDEO_INPUT[_dtype(state)], rows * (2 * ch + 1), args)
