"""Typed launchers for the Kandinsky-6 audio kernels (kernels/cuda/kandinsky6/audio/audio.cu) on torch's CUDA context.

The fatbin is loaded with cuda-python (`cuda.bindings.driver`, or the older `cuda.cuda`) or, failing that, ctypes over
libcuda.so.1, into the context torch made current (cuCtxGetCurrent). Every launch runs on
torch.cuda.current_stream() with the argument types of the audio.cu signature (long long -> int64, int -> int32,
float -> float32, pointers -> uint64 device addresses) and the grid / block / shared sizes of
src/engines/kandinsky6/audio/launch.zig (or the audio.cu header comment for kernels launch.zig does not launch).
Reduction order (`acc`, `order`) and scalar division (`div`) are required keyword arguments: there are no defaults.
"""

import ctypes
import math
import struct

import torch

ACC_FMA, ACC_MUL_ADD = 0, 1
DIV_TRUE, DIV_RECIP = 0, 1
RED_SERIAL_FMA, RED_SERIAL_MUL_ADD, RED_STRIDED_TREE = 0, 1, 2
RED_TORCH = 3  # k6a_l2norm_torch (PyTorch Reduce.cuh order), not an audio.cu K6A_RED_* mode of k6a_l2norm
BLOCK, CONV_BLOCK, CONV_OT, FILTER_TAPS = 256, 128, 4, 12

# audio.cu extern "C" signatures: p pointer (u64), q long long (i64), i int (i32), f float (f32).
SIGNATURES = {
    "k6a_latent_unscale": "ppqqqfif",
    "k6a_mel_affine": "ppppqqq",
    "k6a_clamp": "ppqff",
    "k6a_upsample_nearest_exact2": "ppqq",
    "k6a_scaled_silu": "ppqfi",
    "k6a_mp_residual": "pppqffi",
    "k6a_scale_weights": "ppqf",
    "k6a_add": "pppq",
    "k6a_avg3": "ppppqfi",
    "k6a_conv1d": "ppppqqqqiiiii",
    "k6a_conv_transpose1d": "ppppqqqqiiii",
    "k6a_snake_beta_log": "ppppqqqf",
    "k6a_aa_up2": "pppqqi",
    "k6a_aa_down2": "pppqqqi",
    "k6a_qkv_split": "ppppqqqq",
    "k6a_attn_merge": "ppqqqq",
    "k6a_l2norm": "ppqqqii",
    "k6a_l2norm_torch": "ppqqqii",
    "k6a_normalize_apply": "pppqqqffi",
    "k6a_sdpa_math": "pppppqqf",
}
_CTYPE = {"p": ctypes.c_uint64, "q": ctypes.c_int64, "i": ctypes.c_int32, "f": ctypes.c_float}


class CudaError(RuntimeError):
    pass


def f32(x):
    """Python float -> the nearest fp32 value (round to nearest even), as a Python float."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def recip32(d):
    """1.0f / f32(d) in fp32 (division in double then one rounding is exact for fp32 operands)."""
    return f32(1.0 / f32(d))


def scalar_div(divisor, div):
    """vae.zig ScalarDiv.of: the value the kernel receives for a Python-float divisor under `div`."""
    if div == DIV_TRUE:
        return f32(divisor)
    if div == DIV_RECIP:
        return recip32(divisor)
    raise ValueError(f"div must be DIV_TRUE or DIV_RECIP, got {div!r}")


def mp_residual_consts(t, div):
    """vae.zig MpResidual.of: t = f32(t) (lerp small-weight branch only, t in [0, 0.5)), den = f32(sqrt((1 - t)^2 +
    t^2)) (or its reciprocal)."""
    if not 0.0 <= t < 0.5:
        raise ValueError("mp_residual: only torch.lerp's small-weight branch (t < 0.5) is specified")
    u = 1.0 - t
    return f32(t), scalar_div(math.sqrt(u * u + t * t), div)


def torch_split(outer, N, inner, sm_count, max_threads_per_sm):
    """vae.zig torchSplit: (Y, C) of PyTorch's CUDA reduction (setReduceConfig) for a vector norm over a non-fastest
    axis, from the device's multiprocessor count and max threads per multiprocessor."""
    if sm_count <= 0 or max_threads_per_sm <= 0 or N <= 0 or outer * inner <= 0:
        raise ValueError("torch_split: invalid geometry or device limits")
    outputs = outer * inner
    vec = 4
    while vec > 1 and inner % vec:
        vec //= 2
    dim0, mx = outputs // vec, 512 // vec
    p2 = lambda n: 1 << (n.bit_length() - 1)  # noqa: E731
    d0 = p2(dim0) if dim0 < mx else mx
    d1 = p2(N) if N < mx else mx
    width = min(d0, 32)
    height = min(d1, mx // width)
    width = min(d0, mx // height)
    if not (N >= height * 16 or N >= 256):
        return 1, 1
    per_thread = -(-N // height)
    grid = -(-dim0 // width)
    target = sm_count * (max_threads_per_sm // (width * height))
    ctas = 1
    if per_thread >= 256 and grid <= target:
        ctas = max(min(-(-target // grid), -(-per_thread // 16)), -(-per_thread // 256))
    if height > 32:
        raise ValueError("torch_split: block height above 32 is not supported by k6a_l2norm_torch")
    return height, ctas


def _blocks(n, b=BLOCK):
    return (n + b - 1) // b


def _check_mode(name, v, allowed):
    if v not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {v!r}")


# ------------------------------------------------------------------------------------------------- driver backends

class _CudaPython:
    name = "cuda-python"

    def __init__(self):
        try:
            from cuda.bindings import driver as d
        except ImportError:
            from cuda import cuda as d
        self.d = d
        self._keep = []
        self._ok(d.cuInit(0)[0], "cuInit")

    def _ok(self, err, what):
        if int(err) != 0:
            raise CudaError(f"{what} failed: CUresult {int(err)} ({err})")

    def ctx_current(self):
        err, ctx = self.d.cuCtxGetCurrent()
        self._ok(err, "cuCtxGetCurrent")
        return int(ctx)

    def driver_version(self):
        err, v = self.d.cuDriverGetVersion()
        self._ok(err, "cuDriverGetVersion")
        return int(v)

    def load(self, image):
        buf = ctypes.create_string_buffer(image, len(image))
        self._keep.append(buf)
        err, mod = self.d.cuModuleLoadData(ctypes.addressof(buf))
        self._ok(err, "cuModuleLoadData")
        return mod

    def function(self, mod, name):
        err, fn = self.d.cuModuleGetFunction(mod, name.encode())
        self._ok(err, f"cuModuleGetFunction({name})")
        return fn

    def launch(self, fn, grid, block, shared, stream, params_addr):
        (err,) = self.d.cuLaunchKernel(fn, *grid, *block, shared, self.d.CUstream(stream), params_addr, 0)
        self._ok(err, "cuLaunchKernel")


class _Ctypes:
    name = "ctypes-libcuda"

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        vp, u = ctypes.c_void_p, ctypes.c_uint
        self.lib.cuLaunchKernel.argtypes = [vp, u, u, u, u, u, u, u, vp, vp, vp]
        self.lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        self.lib.cuModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
        self._ok(self.lib.cuInit(0), "cuInit")

    def _ok(self, err, what):
        if err != 0:
            s = ctypes.c_char_p()
            self.lib.cuGetErrorName(err, ctypes.byref(s))
            raise CudaError(f"{what} failed: CUresult {err} ({(s.value or b'?').decode()})")

    def ctx_current(self):
        c = ctypes.c_void_p()
        self._ok(self.lib.cuCtxGetCurrent(ctypes.byref(c)), "cuCtxGetCurrent")
        return c.value or 0

    def driver_version(self):
        v = ctypes.c_int()
        self._ok(self.lib.cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
        return v.value

    def load(self, image):
        m = ctypes.c_void_p()
        self._ok(self.lib.cuModuleLoadData(ctypes.byref(m), image), "cuModuleLoadData")
        return m

    def function(self, mod, name):
        f = ctypes.c_void_p()
        self._ok(self.lib.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), f"cuModuleGetFunction({name})")
        return f

    def launch(self, fn, grid, block, shared, stream, params_addr):
        self._ok(self.lib.cuLaunchKernel(fn, *grid, *block, shared, stream, params_addr, None), "cuLaunchKernel")


def _driver(backend):
    if backend in ("auto", "cuda-python"):
        try:
            return _CudaPython()
        except ImportError:
            if backend == "cuda-python":
                raise
    return _Ctypes()


# ---------------------------------------------------------------------------------------------------- launchers

def _t(x, shape=None, name="tensor"):
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()):
        raise ValueError(f"{name}: need a contiguous fp32 CUDA tensor")
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name}: shape {tuple(x.shape)} != {tuple(shape)}")
    return x


def _ptr(x):
    return 0 if x is None else x.data_ptr()


class K6A:
    def __init__(self, fatbin_path, backend="auto"):
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # make torch's primary context current on this thread
        self.drv = _driver(backend)
        self.ctx = self.drv.ctx_current()
        if not self.ctx:
            raise CudaError("no current CUDA context after torch.cuda.init()")
        with open(fatbin_path, "rb") as f:
            self.image = f.read()
        self.module = self.drv.load(self.image)
        self.fn = {n: self.drv.function(self.module, n) for n in SIGNATURES}

    @property
    def backend(self):
        return self.drv.name

    def _go(self, name, grid, block, shared, *args):
        sig = SIGNATURES[name]
        if len(sig) != len(args):
            raise ValueError(f"{name}: {len(args)} args for signature {sig}")
        vals = [_CTYPE[c](_ptr(a) if c == "p" else a) for c, a in zip(sig, args)]
        params = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
        grid = tuple(grid) + (1,) * (3 - len(grid))
        stream = torch.cuda.current_stream().cuda_stream
        self.drv.launch(self.fn[name], grid, (block, 1, 1), shared, stream, ctypes.addressof(params))

    # --- decoder elementwise

    def latent_unscale(self, rows, *, scale, div, mean):
        B, T, C = _t(rows).shape
        z = torch.empty(B, C, T, device=rows.device)
        self._go("k6a_latent_unscale", (_blocks(B * C * T),), BLOCK, 0, rows, z, B, C, T,
                 scalar_div(scale, div), div, f32(mean))
        return z

    def mel_affine(self, x, stdv, mean):
        B, C, T = _t(x).shape
        _t(stdv, (C,), "stdv"), _t(mean, (C,), "mean")
        y = torch.empty_like(x)
        self._go("k6a_mel_affine", (_blocks(B * C * T),), BLOCK, 0, x, stdv, mean, y, B, C, T)
        return y

    def clamp(self, x, *, lo, hi):
        y = torch.empty_like(_t(x))
        self._go("k6a_clamp", (_blocks(x.numel()),), BLOCK, 0, x, y, x.numel(), f32(lo), f32(hi))
        return y

    def upsample_nearest_exact2(self, x):
        B, C, T = _t(x).shape
        y = torch.empty(B, C, 2 * T, device=x.device)
        self._go("k6a_upsample_nearest_exact2", (_blocks(B * C * 2 * T),), BLOCK, 0, x, y, B * C, T)
        return y

    def scaled_silu(self, x, *, divisor, div):
        _check_mode("div", div, (DIV_TRUE, DIV_RECIP))
        y = torch.empty_like(_t(x))
        self._go("k6a_scaled_silu", (_blocks(x.numel()),), BLOCK, 0, x, y, x.numel(), scalar_div(divisor, div), div)
        return y

    def mp_residual(self, a, b, *, t, div):
        _check_mode("div", div, (DIV_TRUE, DIV_RECIP))
        _t(b, _t(a).shape, "b")
        y = torch.empty_like(a)
        tv, den = mp_residual_consts(t, div)
        self._go("k6a_mp_residual", (_blocks(a.numel()),), BLOCK, 0, a, b, y, a.numel(), tv, den, div)
        return y

    def scale_weights(self, w, *, gain):
        out = torch.empty_like(_t(w))
        self._go("k6a_scale_weights", (_blocks(w.numel()),), BLOCK, 0, w, out, w.numel(), f32(gain))
        return out

    # --- convolutions

    def conv1d(self, x, w, bias, *, stride, pad, dil, acc):
        _check_mode("acc", acc, (ACC_FMA, ACC_MUL_ADD))
        B, Cin, Lin = _t(x).shape
        Cout, _, K = _t(w, (w.shape[0], Cin, w.shape[2]), "w").shape
        if bias is not None:
            _t(bias, (Cout,), "bias")
        if stride <= 0 or Lin + 2 * pad < dil * (K - 1) + 1:
            raise ValueError("conv1d: invalid geometry")
        Lout = (Lin + 2 * pad - dil * (K - 1) - 1) // stride + 1
        y = torch.empty(B, Cout, Lout, device=x.device)
        grid = (_blocks(Lout, CONV_BLOCK), _blocks(Cout, CONV_OT), B)
        self._go("k6a_conv1d", grid, CONV_BLOCK, 0, x, w, bias, y, Cin, Cout, Lin, Lout, K, stride, pad, dil, acc)
        return y

    def conv_transpose1d(self, x, w, bias, *, stride, pad, acc):
        _check_mode("acc", acc, (ACC_FMA, ACC_MUL_ADD))
        B, Cin, Lin = _t(x).shape
        _, Cout, K = _t(w, (Cin, w.shape[1], w.shape[2]), "w").shape
        if bias is not None:
            _t(bias, (Cout,), "bias")
        if stride <= 0 or (Lin - 1) * stride + K <= 2 * pad:
            raise ValueError("conv_transpose1d: invalid geometry")
        Lout = (Lin - 1) * stride - 2 * pad + K
        y = torch.empty(B, Cout, Lout, device=x.device)
        grid = (_blocks(Lout, CONV_BLOCK), _blocks(Cout, CONV_OT), B)
        self._go("k6a_conv_transpose1d", grid, CONV_BLOCK, 0, x, w, bias, y, Cin, Cout, Lin, Lout, K, stride, pad, acc)
        return y

    # --- vocoder elementwise / resamplers

    def add(self, xt, x):
        _t(x, _t(xt).shape, "x")
        y = torch.empty_like(x)
        self._go("k6a_add", (_blocks(x.numel()),), BLOCK, 0, xt, x, y, x.numel())
        return y

    def avg3(self, r0, r1, r2, *, divisor, div):
        _check_mode("div", div, (DIV_TRUE, DIV_RECIP))
        for r in (r1, r2):
            _t(r, _t(r0).shape)
        y = torch.empty_like(r0)
        self._go("k6a_avg3", (_blocks(r0.numel()),), BLOCK, 0, r0, r1, r2, y, r0.numel(), scalar_div(divisor, div), div)
        return y

    def snake_beta_log(self, x, alpha_log, beta_log, *, eps):
        B, C, T = _t(x).shape
        _t(alpha_log, (C,), "alpha_log"), _t(beta_log, (C,), "beta_log")
        y = torch.empty_like(x)
        self._go("k6a_snake_beta_log", (_blocks(B * C * T),), BLOCK, 0, x, alpha_log, beta_log, y, B, C, T, f32(eps))
        return y

    def aa_up2(self, x, filt, *, acc):
        _check_mode("acc", acc, (ACC_FMA, ACC_MUL_ADD))
        B, C, T = _t(x).shape
        _t(filt, (FILTER_TAPS,), "filter")
        y = torch.empty(B, C, 2 * T, device=x.device)
        self._go("k6a_aa_up2", (_blocks(B * C * 2 * T),), BLOCK, 0, x, filt, y, B * C, T, acc)
        return y

    def aa_down2(self, x, filt, *, acc):
        _check_mode("acc", acc, (ACC_FMA, ACC_MUL_ADD))
        B, C, T2 = _t(x).shape
        _t(filt, (FILTER_TAPS,), "filter")
        To = (T2 + 11 - FILTER_TAPS) // 2 + 1
        y = torch.empty(B, C, To, device=x.device)
        self._go("k6a_aa_down2", (_blocks(B * C * To),), BLOCK, 0, x, filt, y, B * C, T2, To, acc)
        return y

    # --- attention layout + candidates

    def qkv_split(self, qkv):
        B, H, D, three, T = _t(qkv).shape
        if three != 3:
            raise ValueError("qkv_split: need [B, H, D, 3, T]")
        q, k, v = (torch.empty(B, H, T, D, device=qkv.device) for _ in range(3))
        self._go("k6a_qkv_split", (_blocks(B * H * T * D),), BLOCK, 0, qkv, q, k, v, B, H, D, T)
        return q, k, v

    def attn_merge(self, h):
        B, H, T, D = _t(h).shape
        y = torch.empty(B, H * D, T, device=h.device)
        self._go("k6a_attn_merge", (_blocks(B * H * D * T),), BLOCK, 0, h, y, B, H, D, T)
        return y

    @staticmethod
    def _axis_view(x, axis):
        outer = math.prod(x.shape[:axis])
        return outer, x.shape[axis], math.prod(x.shape[axis + 1:])

    def l2norm(self, x, *, axis, order, R):
        """Candidate: n[outer * inner] = sqrt(sum x^2 over `axis`). Block per the audio.cu header: R rounded up to 32
        (at least 32) with R shared floats; for the serial orders R is unused by the kernel (block 32, shared 32 floats)."""
        _check_mode("order", order, (RED_SERIAL_FMA, RED_SERIAL_MUL_ADD, RED_STRIDED_TREE))
        if order == RED_STRIDED_TREE and not (1 <= R <= 1024 and R & (R - 1) == 0):
            raise ValueError("l2norm: R must be a power of two in [1, 1024]")
        if order != RED_STRIDED_TREE:
            R = 32
        outer, N, inner = self._axis_view(_t(x), axis)
        n = torch.empty(outer * inner, device=x.device)
        block = max(32, (R + 31) // 32 * 32)
        self._go("k6a_l2norm", (outer * inner,), block, 4 * R, x, n, outer, N, inner, order, R)
        return n

    def l2norm_torch(self, x, *, axis, sm_count, max_threads_per_sm):
        """Candidate: n[outer * inner] in PyTorch's CUDA reduction order (k6a_l2norm_torch, split by torch_split from
        the device limits). Block 128, grid ceil(outer * inner / 128)."""
        outer, N, inner = self._axis_view(_t(x), axis)
        Y, C = torch_split(outer, N, inner, sm_count, max_threads_per_sm)
        n = torch.empty(outer * inner, device=x.device)
        self._go("k6a_l2norm_torch", (_blocks(outer * inner, 128),), 128, 0, x, n, outer, N, inner, Y, C)
        return n

    def normalize_apply(self, x, n, *, axis, s, eps, den_fused):
        _check_mode("den_fused", den_fused, (0, 1))
        outer, N, inner = self._axis_view(_t(x), axis)
        _t(n, (outer * inner,), "n")
        y = torch.empty_like(x)
        self._go("k6a_normalize_apply", (_blocks(x.numel()),), BLOCK, 0, x, n, y, outer, N, inner, f32(s), f32(eps),
                 den_fused)
        return y

    def sdpa_math(self, q, k, v, *, c):
        """Candidate math SDPA over contiguous [..., T, D] (leading dims flattened to BH)."""
        T, D = _t(q).shape[-2:]
        _t(k, q.shape, "k"), _t(v, q.shape, "v")
        BH = q.numel() // (T * D)
        out = torch.empty_like(q)
        scratch = torch.empty(BH * T * T, device=q.device)
        self._go("k6a_sdpa_math", (T, BH), BLOCK, 0, q, k, v, out, scratch, T, D, f32(c))
        return out
