"""LocalRouter's Kandinsky-6 DiT kernels (kernels/cuda/kandinsky6/dit/{gemm,attention,ops}.cu) from the fatbin images
the Zig build embeds, launched through the CUDA driver API as src/engines/kandinsky6/dit/launch.zig launches them: one
wrapper a launcher, the same checks of dtype, shape, row stride and alignment, the arguments packed in kernel order
with the kernel's types, the same grid, block and dynamic shared memory. Same image, arguments and grid: the twin's
bits are the engine's.

Images: `Kernels(fatbins=DIR)` or STK_K6_DIT_FATBINS=DIR, holding k6_dit_gemm.fatbin, k6_dit_attention.fatbin and
k6_dit_ops.fatbin (or `Kernels(images={"gemm": bytes, ...})`). Tensors are launch.zig's `layout.View`s: 2-D CUDA
tensors with unit column stride, `ld` the row stride in elements; a 1-D tensor is one row. Launches go on torch's
current stream in torch's primary context, so torch must have initialised CUDA first. A launch never synchronizes.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

IMAGES = ("gemm", "attention", "ops")
I32_MAX = 2**31 - 1
U32_MAX = 2**32 - 1

# attention.zig
HEAD_DIM = 128
BLOCK_ROWS = 64
SHARED_BYTES = 2 * 64 * 128 * 2
ATTN_THREADS = 128
# layout.zig
ROPE_PAIRS = 64
LN_EPS = 1e-5  # f32(1e-5) once packed
ATTN_SCALE = struct.unpack("<f", struct.pack("<I", 0x3DB504F3))[0]  # 1 / sqrt(128) as SDPA rounds it
# launch.zig rms_eps: float32's epsilon (the fused RMSNorm's accumulate type), not bf16's
RMS_EPS = 2.0**-23
# launch.zig Epilogue
EPI_BIAS, EPI_GELU = 0, 1
NORM_BLOCK = (32, 4)  # launch.norm_block: torch's layer-norm block
FP32_LINEAR = {"tf32_rna": 0, "tf32_truncate": 1, "fp32": 2}  # launch.Fp32Linear

CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32", torch.uint8: "u8", torch.int32: "i32"}
TORCH = {v: k for k, v in DTYPES.items()}
SIZE = {"bf16": 2, "f32": 4, "u8": 1, "i32": 4}


class ViewMismatch(ValueError):
    pass


class InvalidGemm(ValueError):
    pass


class InvalidAttention(ValueError):
    pass


class CudaError(RuntimeError):
    pass


@dataclass(frozen=True)
class View:
    """layout.View: rows of cols elements, row r at ptr + r * ld * size."""

    ptr: int
    dtype: str
    rows: int
    cols: int
    ld: int

    def expect(self, dtype: str, rows: int, cols: int) -> None:
        if self.dtype != dtype or self.rows != rows or self.cols != cols or self.ld < cols or self.ptr == 0:
            raise ViewMismatch(f"view {self} != {dtype} [{rows}, {cols}]")


def view(t) -> View:
    """A tensor as launch.zig sees it. The stride of a size-1 dimension carries nothing: one row's ld is at least its
    width, one column needs no column stride."""
    if isinstance(t, View):
        return t
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise ViewMismatch("not a CUDA tensor")
    if t.dim() == 1:
        if t.numel() > 1 and t.stride(0) != 1:
            raise ViewMismatch("1-D view with a stride")
        t = t.view(1, -1)
    if t.dim() != 2 or (t.shape[1] > 1 and t.stride(1) != 1):
        raise ViewMismatch(f"not a row view: shape {tuple(t.shape)} stride {t.stride()}")
    dt = DTYPES.get(t.dtype)
    if dt is None:
        raise ViewMismatch(f"dtype {t.dtype}")
    rows, cols = int(t.shape[0]), int(t.shape[1])
    ld = int(t.stride(0)) if rows > 1 else max(int(t.stride(0)), cols)
    return View(int(t.data_ptr()), dt, rows, cols, ld)


@dataclass(frozen=True)
class Mask:
    """attention.Mask.keys: device u8 [nkeys] (nonzero: the key is seen) and how many are nonzero (host-known)."""

    t: torch.Tensor
    valid: int

    @property
    def ptr(self) -> int:
        return int(self.t.data_ptr())

    @staticmethod
    def of(mask: torch.Tensor) -> "Mask":
        """From torch's bool key-padding mask (True = keep), any of [K], [1, K], [1, 1, 1, K]; synchronizes once."""
        m = mask.reshape(-1).to(torch.uint8).contiguous()
        return Mask(m, int(m.sum().item()))


def grid_rows(rows: int) -> int:
    """launch.gridRows: a row count on grid.x (grid.y / .z stop at 65,535)."""
    if rows == 0 or rows > I32_MAX:
        raise ViewMismatch(f"rows {rows}")
    return rows


def blocks(n: int, per: int) -> int:
    return (n + per - 1) // per


def i32(v: int) -> ctypes.c_int32:
    if not 0 <= v <= I32_MAX:  # launch.zig i32of: @intCast
        raise OverflowError(f"{v} does not fit i32")
    return ctypes.c_int32(v)


def i64(v: int) -> ctypes.c_int64:
    return ctypes.c_int64(v)


def u64(v: int) -> ctypes.c_uint64:
    return ctypes.c_uint64(v)


def f32(v: float) -> ctypes.c_float:
    return ctypes.c_float(v)


def load_images(fatbins=None) -> dict[str, bytes]:
    """The three images: from a directory (STK_K6_DIT_FATBINS when None) or a dict name -> path."""
    if isinstance(fatbins, dict):
        return {n: Path(fatbins[n]).read_bytes() for n in IMAGES}
    d = fatbins if fatbins is not None else os.environ.get("STK_K6_DIT_FATBINS")
    if not d:
        raise FileNotFoundError("set STK_K6_DIT_FATBINS=DIR (k6_dit_{gemm,attention,ops}.fatbin) or pass fatbins=")
    return {n: (Path(d) / f"k6_dit_{n}.fatbin").read_bytes() for n in IMAGES}


def _driver() -> ctypes.CDLL:
    cu = ctypes.CDLL("libcuda.so.1")
    P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
    sig = {
        "cuInit": [U],
        "cuDeviceGet": [ctypes.POINTER(I), I],
        "cuDevicePrimaryCtxRetain": [ctypes.POINTER(P), I],
        "cuCtxGetCurrent": [ctypes.POINTER(P)],
        "cuCtxSetCurrent": [P],
        "cuModuleLoadData": [ctypes.POINTER(P), P],
        "cuModuleUnload": [P],
        "cuModuleGetFunction": [ctypes.POINTER(P), P, ctypes.c_char_p],
        "cuFuncSetAttribute": [P, I, I],
        "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, ctypes.POINTER(P), ctypes.POINTER(P)],
        "cuGetErrorString": [I, ctypes.POINTER(ctypes.c_char_p)],
    }
    for name, args in sig.items():
        fn = getattr(cu, name)
        fn.argtypes = args
        fn.restype = I
    return cu


class Kernels:
    """launch.zig `Kernels`: the three modules and every function, loaded into torch's primary context."""

    def __init__(self, images: dict[str, bytes] | None = None, fatbins=None):
        if images is None:
            images = load_images(fatbins)
        if any(len(images.get(n, b"")) == 0 for n in IMAGES):
            raise CudaError("empty kernel image (a host-only build)")  # Kernels.load: error.Invalid
        self.digests = {n: hashlib.sha256(images[n]).hexdigest() for n in IMAGES}
        torch.cuda.init()
        torch.cuda.current_stream()
        self.cu = cu = _driver()
        self._check(cu.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(cu.cuDeviceGet(ctypes.byref(dev), torch.cuda.current_device()), "cuDeviceGet")
        prim, cur = ctypes.c_void_p(), ctypes.c_void_p()
        self._check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(prim), dev), "cuDevicePrimaryCtxRetain")
        self._check(cu.cuCtxGetCurrent(ctypes.byref(cur)), "cuCtxGetCurrent")
        if cur.value != prim.value:
            self._check(cu.cuCtxSetCurrent(prim), "cuCtxSetCurrent")
        self.mods = {n: self._load(images[n]) for n in IMAGES}
        g, a, o = self.mods["gemm"], self.mods["attention"], self.mods["ops"]
        self.gemm_fn = self._fn(g, "k6_gemm_bf16")
        self.gemv_fn = self._fn(g, "k6_gemv_f32")
        self.gemv_tf32_fn = self._fn(g, "k6_gemv_tf32")
        self.attn_fn = self._fn(a, "k6_attention")
        self._check(cu.cuFuncSetAttribute(self.attn_fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SHARED_BYTES),
                    "cuFuncSetAttribute")
        self.cast_fn = self._fn(o, "k6_cast_bf16")
        self.add_fn = self._fn(o, "k6_add_bf16")
        self.ln_fn = self._fn(o, "k6_layer_norm")
        self.norm_mod_fn = self._fn(o, "k6_norm_mod")
        self.gate_fn = self._fn(o, "k6_gate_sum")
        self.qk_fn = self._fn(o, "k6_qk_norm_rope")
        self.sincos_fn = self._fn(o, "k6_time_sincos")
        self.patchify_fn = self._fn(o, "k6_patchify")
        self.unpatchify_fn = self._fn(o, "k6_unpatchify")
        self.grid_major_fn = self._fn(o, "k6_grid_major")
        self.count = 0  # launches, as launch.zig counts them

    # ------------------------------------------------------------------------------------------------------- driver
    def _check(self, r: int, what: str) -> None:
        if r != 0:
            s = ctypes.c_char_p()
            ok = hasattr(self, "cu") and self.cu.cuGetErrorString(r, ctypes.byref(s)) == 0 and s.value
            msg = s.value.decode() if ok else ""
            raise CudaError(f"{what}: CUresult {r} {msg}")

    def _load(self, data: bytes) -> ctypes.c_void_p:
        buf = (ctypes.c_uint64 * (len(data) // 8 + 2))()  # 8-byte aligned, zero padded
        ctypes.memmove(buf, data, len(data))
        m = ctypes.c_void_p()
        self._check(self.cu.cuModuleLoadData(ctypes.byref(m), ctypes.cast(buf, ctypes.c_void_p)), "cuModuleLoadData")
        return m

    def _fn(self, mod: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
        f = ctypes.c_void_p()
        self._check(self.cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), f"cuModuleGetFunction {name}")
        return f

    def _go(self, f, grid: tuple, threads: int | tuple, shared: int, args: list) -> None:
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = (threads,) if isinstance(threads, int) else tuple(threads)
        b = b + (1,) * (3 - len(b))
        if any(not 0 <= x <= U32_MAX for x in g):
            raise OverflowError(f"grid {g}")
        self.count += 1
        params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._check(self.cu.cuLaunchKernel(f, g[0], g[1], g[2], b[0], b[1], b[2], shared, stream, params, None),
                    "cuLaunchKernel")

    def close(self) -> None:
        for m in self.mods.values():
            self.cu.cuModuleUnload(m)
        self.mods = {}

    # ---------------------------------------------------------------------------------------------------- launchers
    @staticmethod
    def gemm_vec(x, w) -> bool:
        """The staging gemm() picks: 16-byte cp.async when K, both row strides and both pointers allow it."""
        x, w = view(x), view(w)
        return x.cols % 8 == 0 and x.ld % 8 == 0 and w.ld % 8 == 0 and x.ptr % 16 == 0 and w.ptr % 16 == 0

    def gemm(self, x, w, bias, out, epi=EPI_BIAS) -> None:
        """out[M, N] = x[M, K] . w[N, K]^T, epilogue bias (+ bias[N] in fp32; None for none) or gelu (of the
        bf16-rounded product, no bias)."""
        epi = {"bias": EPI_BIAS, "gelu": EPI_GELU}.get(epi, epi)
        x, w, out = view(x), view(w), view(out)
        b = view(bias) if bias is not None else None
        m, n, kd = x.rows, w.rows, x.cols
        w.expect("bf16", n, kd)
        x.expect("bf16", m, kd)
        out.expect("bf16", m, n)
        if b is not None:
            b.expect("bf16", 1, n)
        if epi not in (EPI_BIAS, EPI_GELU) or (epi == EPI_GELU and b is not None):
            raise InvalidGemm("gelu takes no bias")
        vec = kd % 8 == 0 and x.ld % 8 == 0 and w.ld % 8 == 0 and x.ptr % 16 == 0 and w.ptr % 16 == 0
        if m == 0 or m > I32_MAX or n > I32_MAX:
            raise InvalidGemm(f"M {m} N {n}")
        args = [u64(x.ptr), u64(w.ptr), u64(b.ptr if b is not None else 0), u64(out.ptr), i32(m), i32(n), i32(kd),
                i64(x.ld), i64(w.ld), i64(out.ld), ctypes.c_int32(epi), ctypes.c_int32(int(vec))]
        self._go(self.gemm_fn, (blocks(n, 128), blocks(m, 128)), 256, 0, args)

    def gemv(self, x, w, bias, out, silu: bool, math: str) -> None:
        """out[M, N] = act(x[M, K]) . float(w[N, K])^T + float(bias[N]): x fp32 or bf16, act SiLU when `silu`, out
        fp32 or bf16 (one rounding), in `math` (launch.Fp32Linear: "tf32_rna", "tf32_truncate" or "fp32"; none is a
        default and none is qualified against native)."""
        if math not in FP32_LINEAR:
            raise InvalidGemm(f"gemv math {math!r}")
        x, w, out = view(x), view(w), view(out)
        b = view(bias) if bias is not None else None
        m, n = x.rows, w.rows
        w.expect("bf16", n, x.cols)
        if w.ld != x.cols:
            raise InvalidGemm("gemv weight must be dense")
        if x.dtype not in ("f32", "bf16"):
            raise ViewMismatch(f"gemv x {x.dtype}")
        x.expect(x.dtype, m, x.cols)
        if out.dtype not in ("f32", "bf16"):
            raise ViewMismatch(f"gemv out {out.dtype}")
        out.expect(out.dtype, m, n)
        if b is not None:
            b.expect("bf16", 1, n)
        args = [u64(x.ptr), u64(w.ptr), u64(b.ptr if b is not None else 0), u64(out.ptr), i32(m), i32(n), i32(x.cols),
                i64(x.ld), i64(out.ld), ctypes.c_int32(int(x.dtype == "bf16")), ctypes.c_int32(int(bool(silu))),
                ctypes.c_int32(int(out.dtype == "bf16"))]
        if math == "fp32":
            self._go(self.gemv_fn, (blocks(n, 8),), 256, 0, args)
        else:
            self._go(self.gemv_tf32_fn, (blocks(n, 64), blocks(m, 16)), 256, 0, args + [ctypes.c_int32(FP32_LINEAR[math])])

    def attend(self, q, k, v, out, heads: int, mask: Mask | None = None, scale: float = ATTN_SCALE) -> None:
        """attention.Call: softmax(q k^T * scale) v a head; q [W, heads * 128], k / v [nkeys, heads * 128]."""
        q, k, v, out = view(q), view(k), view(v), view(out)
        # attention.Call.validate
        width = heads * HEAD_DIM
        q.expect("bf16", q.rows, width)
        k.expect("bf16", k.rows, width)
        v.expect("bf16", k.rows, width)
        out.expect("bf16", q.rows, width)
        if heads == 0 or q.rows == 0 or k.rows == 0:
            raise InvalidAttention("empty call")
        if q.rows > I32_MAX or k.rows > I32_MAX:
            raise InvalidAttention("too many rows")
        if q.ld % 2 != 0 or out.ld % 2 != 0 or q.ptr % 4 != 0 or out.ptr % 4 != 0:
            raise InvalidAttention("query / output alignment")
        if k.ld % 8 != 0 or v.ld % 8 != 0 or k.ptr % 16 != 0 or v.ptr % 16 != 0:
            raise InvalidAttention("key / value alignment")
        if mask is not None and (mask.ptr == 0 or mask.valid == 0 or mask.valid > k.rows):
            raise InvalidAttention("mask")
        args = [u64(q.ptr), u64(k.ptr), u64(v.ptr), u64(out.ptr), u64(mask.ptr if mask is not None else 0),
                i32(q.rows), i32(k.rows), i64(q.ld), i64(k.ld), i64(v.ld), i64(out.ld)]
        args += [i64(0) for _ in range(5)]  # batch strides: one batch a call
        args.append(f32(scale))
        self._go(self.attn_fn, (blocks(q.rows, BLOCK_ROWS), heads, 1), ATTN_THREADS, SHARED_BYTES, args)

    def cast_bf16(self, x, y) -> None:
        x, y = view(x), view(y)
        x.expect("f32", x.rows, x.cols)
        y.expect("bf16", x.rows, x.cols)
        if x.ld != x.cols or y.ld != y.cols:
            raise ViewMismatch("cast_bf16 wants contiguous views")
        n = x.rows * x.cols
        self._go(self.cast_fn, (blocks(n, 256),), 256, 0, [u64(x.ptr), u64(y.ptr), i64(n)])

    def add_bf16(self, x, z, y) -> None:
        """y = bf16(x + z), contiguous views of one shape (y may alias x or z)."""
        x, z, y = view(x), view(z), view(y)
        for t in (x, z, y):
            t.expect("bf16", x.rows, x.cols)
            if t.ld != t.cols:
                raise ViewMismatch("add_bf16 wants contiguous views")
        n = x.rows * x.cols
        self._go(self.add_fn, (blocks(n, 256),), 256, 0, [u64(x.ptr), u64(z.ptr), u64(y.ptr), i64(n)])

    def layer_norm(self, x, w, b, y) -> None:
        """LayerNorm (eps 1e-5) of bf16 rows into bf16 rows, affine when w / b are given (y may alias x)."""
        x, y = view(x), view(y)
        w = view(w) if w is not None else None
        b = view(b) if b is not None else None
        x.expect("bf16", x.rows, x.cols)
        y.expect("bf16", x.rows, x.cols)
        if x.cols % 4:
            raise ViewMismatch("norm cols % 4")
        if w is not None:
            w.expect("bf16", 1, x.cols)
        if b is not None:
            b.expect("bf16", 1, x.cols)
        args = [u64(x.ptr), u64(w.ptr if w is not None else 0), u64(b.ptr if b is not None else 0), u64(y.ptr),
                i32(x.cols), i64(x.ld), i64(y.ld), f32(LN_EPS)]
        self._go(self.ln_fn, (grid_rows(x.rows),), NORM_BLOCK, 0, args)

    def norm_mod(self, x, scale, shift, y) -> None:
        """apply_scale_shift_norm with a non-affine LayerNorm (eps 1e-5); scale / shift one bf16 row each."""
        x, scale, shift, y = view(x), view(scale), view(shift), view(y)
        x.expect("bf16", x.rows, x.cols)
        y.expect("bf16", x.rows, x.cols)
        if x.cols % 4:
            raise ViewMismatch("norm cols % 4")
        scale.expect("bf16", 1, x.cols)
        shift.expect("bf16", 1, x.cols)
        args = [u64(x.ptr), u64(scale.ptr), u64(shift.ptr), u64(y.ptr), i32(x.cols), i64(x.ld), i64(y.ld), f32(LN_EPS)]
        self._go(self.norm_mod_fn, (grid_rows(x.rows),), NORM_BLOCK, 0, args)

    def gate_sum(self, x, out, gate) -> None:
        """apply_gate_sum in place: x = bf16(x + gate * out), gate one bf16 row."""
        x, out, gate = view(x), view(out), view(gate)
        x.expect("bf16", x.rows, x.cols)
        out.expect("bf16", x.rows, x.cols)
        gate.expect("bf16", 1, x.cols)
        args = [u64(x.ptr), u64(out.ptr), u64(gate.ptr), i32(x.cols), i64(x.ld), i64(out.ld)]
        self._go(self.gate_fn, (grid_rows(x.rows), blocks(x.cols, 256)), 256, 0, args)

    def qk_norm_rope(self, x, w, rope, heads: int) -> None:
        """In place: per-head RMSNorm (bf16 weight [128], eps FLT_EPSILON) then the RoPE table (fp32 [rows, 256]), or
        no rotation when rope is None."""
        x, w = view(x), view(w)
        r = view(rope) if rope is not None else None
        x.expect("bf16", x.rows, heads * HEAD_DIM)
        w.expect("bf16", 1, HEAD_DIM)
        if r is not None:
            r.expect("f32", x.rows, ROPE_PAIRS * 4)
        args = [u64(x.ptr), u64(w.ptr), u64(r.ptr if r is not None else 0), i32(x.rows), i32(heads), i64(x.ld),
                i64(r.ld if r is not None else 0), f32(RMS_EPS)]
        self._go(self.qk_fn, (blocks(x.rows * heads, 8),), 256, 0, args)

    def time_sincos(self, time, freqs, out) -> None:
        """time fp32 [M, 1], freqs fp32 [1, half], out fp32 [M, 2 * half]."""
        time, freqs, out = view(time), view(freqs), view(out)
        time.expect("f32", time.rows, 1)
        freqs.expect("f32", 1, freqs.cols)
        out.expect("f32", time.rows, 2 * freqs.cols)
        if out.ld != out.cols or time.ld != 1:
            raise ViewMismatch("time_sincos wants dense time and out")
        args = [u64(time.ptr), u64(freqs.ptr), u64(out.ptr), i32(freqs.cols)]
        self._go(self.sincos_fn, (grid_rows(time.rows), blocks(freqs.cols, 256)), 256, 0, args)

    def patchify(self, x, rows, sh, c: int) -> None:
        """[T, H, W, C] channels last (fp32 or bf16, x as [T * H * W, C]) -> bf16 rows [T * H/2 * W/2, 4C]."""
        x, rows = view(x), view(rows)
        n_in = sh.t * sh.h * sh.w
        if x.dtype not in ("f32", "bf16"):
            raise ViewMismatch(f"patchify x {x.dtype}")
        x.expect(x.dtype, n_in, c)
        rows.expect("bf16", sh.tokens(), 4 * c)
        if x.ld != c or rows.ld != rows.cols:
            raise ViewMismatch("patchify wants contiguous views")
        args = [u64(x.ptr), u64(rows.ptr), i32(sh.t), i32(sh.h), i32(sh.w), i32(c),
                ctypes.c_int32(int(x.dtype == "f32"))]
        self._go(self.patchify_fn, (blocks(n_in * c, 256),), 256, 0, args)

    def unpatchify(self, rows, out, sh, c: int, g: int) -> None:
        """The video head's unpatchify into grid-major [G, T, H, W, C / G] bf16 (out as [G * T * H * W, C / G])."""
        rows, out = view(rows), view(out)
        rows.expect("bf16", sh.tokens(), 4 * c)
        if c % g != 0 or rows.ld != rows.cols:
            raise ViewMismatch("unpatchify rows")
        out.expect("bf16", g * sh.t * sh.h * sh.w, c // g)
        if out.ld != out.cols:
            raise ViewMismatch("unpatchify out")
        args = [u64(rows.ptr), u64(out.ptr), i32(sh.t), i32(sh.h // 2), i32(sh.w // 2), i32(c), i32(g)]
        self._go(self.unpatchify_fn, (blocks(sh.tokens() * 4 * c, 256),), 256, 0, args)

    def grid_major(self, x, out, g: int) -> None:
        """[rows, G * Cg] -> grid-major [G * rows, Cg] (contiguous)."""
        x, out = view(x), view(out)
        x.expect("bf16", x.rows, x.cols)
        if x.cols % g != 0 or x.ld != x.cols:
            raise ViewMismatch("grid_major x")
        out.expect("bf16", x.rows * g, x.cols // g)
        if out.ld != out.cols:
            raise ViewMismatch("grid_major out")
        args = [u64(x.ptr), u64(out.ptr), i32(x.rows), i32(g), i32(x.cols // g)]
        self._go(self.grid_major_fn, (blocks(x.rows * x.cols, 256),), 256, 0, args)
