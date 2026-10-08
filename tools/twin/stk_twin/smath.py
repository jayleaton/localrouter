"""Portable sin / cos for the RoPE tables: Cody-Waite reduction by pi/2 (exact for |x| < 2^20 * pi / 2) and fdlibm's
kernel polynomials, plain IEEE double operations in a fixed order. `src/engines/qwen_image/smath.zig` performs the
same operations, so both sides get the same bits on any machine (libm sin / cos differ between libraries).
The RoPE frequencies are frozen in `kernels/qwen_image/rope_omega.json` for the same reason.
"""

from __future__ import annotations

import json
import math
import os
import struct
from functools import lru_cache
from pathlib import Path

INVPIO2 = 6.36619772367581382433e-01
PIO2_1 = 1.57079632673412561417e+00   # first 33 bits of pi/2
PIO2_2 = 6.07710050630396597660e-11   # next 33 bits
PIO2_3 = 2.02226624871116645580e-21   # next 33 bits
S1, S2, S3 = -1.66666666666666324348e-01, 8.33333333332248946124e-03, -1.98412698298579493134e-04
S4, S5, S6 = 2.75573137070700676789e-06, -2.50507602534068634195e-08, 1.58969099521155010221e-10
C1, C2, C3 = 4.16666666666666019037e-02, -1.38888888888741095749e-03, 2.48015872894767294178e-05
C4, C5, C6 = -2.75573143513906633035e-07, 2.08757232129817482790e-09, -1.13596475577881948265e-11


def _ksin(x: float) -> float:
    z = x * x
    v = z * x
    r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))
    return x + v * (S1 + z * r)


def _kcos(x: float) -> float:
    z = x * x
    r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))))
    hz = 0.5 * z
    w = 1.0 - hz
    return w + (((1.0 - w) - hz) + z * r)


def sincos(x: float) -> tuple[float, float]:
    n = math.floor(x * INVPIO2 + 0.5)
    fn = float(n)
    r = ((x - fn * PIO2_1) - fn * PIO2_2) - fn * PIO2_3
    s, c = _ksin(r), _kcos(r)
    q = n & 3
    return ((s, c), (c, -s), (-s, -c), (-c, s))[q]


@lru_cache(maxsize=1)
def omegas() -> list[float]:
    root = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[3] / "kernels"))
    path = root / "qwen_image" / "rope_omega.json"
    return [struct.unpack(">d", bytes.fromhex(h))[0] for h in json.loads(path.read_text())["omega_f64_be_hex"]]


LN2_HI, LN2_LO, TWO54 = 6.93147180369123816490e-01, 1.90821492927058770002e-10, 1.80143985094819840000e+16
LG1, LG2, LG3 = 6.666666666666735130e-01, 3.999999999940941908e-01, 2.857142874366239149e-01
LG4, LG5, LG6, LG7 = 2.222219843214978396e-01, 1.818357216161805012e-01, 1.531383769920937332e-01, 1.479819860511658591e-01


def _words(x: float) -> tuple[int, int]:
    b = struct.unpack("<Q", struct.pack("<d", x))[0]
    return b >> 32, b & 0xFFFFFFFF


def _with_high(x: float, hi: int) -> float:
    lo = struct.unpack("<Q", struct.pack("<d", x))[0] & 0xFFFFFFFF
    return struct.unpack("<d", struct.pack("<Q", ((hi & 0xFFFFFFFF) << 32) | lo))[0]


def log(x: float) -> float:
    """fdlibm's __ieee754_log for finite x > 0, operation for operation (smath.zig `log` is the same)."""
    hx, _ = _words(x)
    k = 0
    if hx < 0x00100000:  # subnormal
        k -= 54
        x *= TWO54
        hx, _ = _words(x)
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _with_high(x, hx | (i ^ 0x3FF00000))
    k += i >> 20
    f = x - 1.0
    if (0x000FFFFF & (2 + hx)) < 3:
        if f == 0.0:
            return 0.0 if k == 0 else k * LN2_HI + k * LN2_LO
        r = f * f * (0.5 - 0.33333333333333333 * f)
        return f - r if k == 0 else k * LN2_HI - ((r - k * LN2_LO) - f)
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (LG2 + w * (LG4 + w * LG6))
    t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)))
    i |= j
    r = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        return f - (hfsq - s * (hfsq + r)) if k == 0 else dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
    return f - s * (f - r) if k == 0 else dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)


# fdlibm e_exp.c / s_expm1.c constants
_HALF = (0.5, -0.5)
_LN2HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)
_LN2LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)
_INVLN2 = 1.44269504088896338700e+00
_P1, _P2, _P3 = 1.66666666666666019037e-01, -2.77777777770155933842e-03, 6.61375632143793436117e-05
_P4, _P5 = -1.65339022054652515390e-06, 4.13813679705723846039e-08
_Q1, _Q2, _Q3 = -3.33333333333331316428e-02, 1.58730158725481460165e-03, -7.93650757867487942473e-05
_Q4, _Q5 = 4.00821782732936239552e-06, -2.01099218183624371326e-07


def _add_exp(y: float, k: int) -> float:
    hi, _ = _words(y)
    return _with_high(y, hi + (k << 20))


def exp(x: float) -> float:
    """fdlibm's __ieee754_exp for |x| < 700, operation for operation (smath.zig `exp` is the same)."""
    hx, _ = _words(x)
    xsb = (hx >> 31) & 1
    hx &= 0x7FFFFFFF
    assert hx < 0x4086232B, "exp: |x| < 700 only"
    k, hi, lo = 0, 0.0, 0.0
    if hx > 0x3FD62E42:
        if hx < 0x3FF0A2B2:
            hi, lo, k = x - _LN2HI[xsb], _LN2LO[xsb], 1 - xsb - xsb
        else:
            k = int(_INVLN2 * x + _HALF[xsb])
            t = float(k)
            hi, lo = x - t * _LN2HI[0], t * _LN2LO[0]
        x = hi - lo
    elif hx < 0x3E300000:
        return 1.0 + x
    t = x * x
    c = x - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    assert k >= -1021
    return _add_exp(y, k)


def expm1(x: float) -> float:
    """fdlibm's expm1 for |x| < 56 ln 2, operation for operation (smath.zig `expm1` is the same)."""
    hx, _ = _words(x)
    xsb = hx & 0x80000000
    hx &= 0x7FFFFFFF
    assert hx < 0x4043687A, "expm1: |x| < 56 ln 2 only"
    k, c = 0, 0.0
    if hx > 0x3FD62E42:
        if hx < 0x3FF0A2B2:
            if xsb == 0:
                hi, lo, k = x - _LN2HI[0], _LN2LO[0], 1
            else:
                hi, lo, k = x + _LN2HI[0], -_LN2LO[0], -1
        else:
            k = int(_INVLN2 * x + (0.5 if xsb == 0 else -0.5))
            t = float(k)
            hi, lo = x - t * _LN2HI[0], t * _LN2LO[0]
        x = hi - lo
        c = (hi - x) - lo
    elif hx < 0x3C900000:
        return x
    hfx = 0.5 * x
    hxs = x * hfx
    r1 = 1.0 + hxs * (_Q1 + hxs * (_Q2 + hxs * (_Q3 + hxs * (_Q4 + hxs * _Q5))))
    t = 3.0 - r1 * hfx
    e = hxs * ((r1 - t) / (6.0 - x * t))
    if k == 0:
        return x - (x * e - hxs)
    e = x * (e - c) - c
    e -= hxs
    if k == -1:
        return 0.5 * (x - e) - 0.5
    if k == 1:
        return -2.0 * (e - (x + 0.5)) if x < -0.25 else 1.0 + 2.0 * (x - e)
    if k <= -2 or k > 56:
        return _add_exp(1.0 - (e - x), k) - 1.0
    if k < 20:
        t = _with_high(1.0, 0x3FF00000 - (0x200000 >> k))
        return _add_exp(t - (e - x), k)
    t = _with_high(1.0, (0x3FF - k) << 20)
    y = x - (e + t)
    y += 1.0
    return _add_exp(y, k)


def table(path) -> None:
    """The cross-check table for smath.zig's tests: inputs and f64 bit patterns of log, exp and expm1."""
    import json
    import random

    rnd = random.Random(1234)
    xs = [0.0, 1e-30, -1e-30, 1e-17, 0.25, -0.25, 0.34657359, -0.34657359, 0.5, -0.5, 1.0397, -1.0397, 2.0, -2.0,
          0.6931471805599453, -0.6931471805599453] + [rnd.uniform(-38.0, 38.0) for _ in range(300)] \
         + [rnd.uniform(-1.2, 1.2) for _ in range(300)]
    bits = lambda v: struct.unpack("<Q", struct.pack("<d", v))[0]
    rows = []
    for x in xs:
        row = {"x": bits(x), "exp": bits(exp(x)), "expm1": bits(expm1(x))}
        if x > 0:
            row["log"] = bits(log(x))
        rows.append(row)
    json.dump(rows, open(path, "w"))
