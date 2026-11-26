"""MiniMax H3-Image edits on the host, as ComfyUI e9afbccc and #157's workflows run them (docs/dev/H3-IMAGE.md section 1.6),
in numpy float32 exactly where ComfyUI's torch, numpy and PIL round. Pure host arithmetic: the Zig engine's `image.zig`
and `edit.zig` run the same steps, held to the JSON `--fixture` writes, and `test_edit.py` checks each step against
ComfyUI's own code (on the CPU, no GPU, no checkpoint).

The graph (minimax-h3-image-edit.json; -edit-unmasked.json has no composite):
  load_rgba          LoadImage: uint8 / 255 in float32; the MASK output 1 - alpha / 255 (nodes.py:1771-1772)
  total_pixels_size  ImageScaleToTotalPixels' size: round(side * sqrt(MP * 2^20 / area) / 32) * 32, Python's round
                     (nodes_post_processing.py:249-256); node 11, MP 1.0 masked, 2.0 whole scene
  area               its `area` resize: F.interpolate(mode="area") on the channels-last view, i.e. adaptive_avg_pool2d:
                     window [floor(o * in / out), ceil((o + 1) * in / out)), the fp32 sum row by row, then / kh / kw
  short_edge_size    _size_short_edge (nodes_minimax_h3.py:677-680): Qwen's copy at [1024, 1024], the VAE's [1024, 2048]
  resize_to          _resize_to: the image itself when the size matches, else lanczos (`comfy.utils.lanczos`):
                     trunc(clip(255 x)) to uint8, PIL's LANCZOS (`h3.image.resize_lanczos`, Pillow's 8-bit Resample.c),
                     / 255 in float32
  crop_center        common_upscale's crop="center" offsets (comfy/utils.py:1145-1156); node 15's ImageScale(lanczos)
  bilinear           the composite's mask resize: F.interpolate(mode="bilinear", align_corners=False) on [1, 1, h, w], the
                     generic CPU kernel (output H + W > 128, which every canvas of 256 px or more is): fp32 source index
                     scale * (i + 0.5) - 0.5 (scale = in / out), clamped at 0, lambda = src - floor(src), each row lerp
                     fma(t0, l0, t1 * l1) along W, then the same along H (test_edit.py: bit-equal to torch)
  composite          ImageCompositeMasked: m * decoded + (1 - m) * dest, each product and the sum rounded to fp32
                     (nodes_mask.py:12-46)
  to_u8              SaveImage: trunc(clip(255 x, 0, 255)) (nodes.py:1707-1708)
  posterior          H3ImageVAE.encode's tail (vae.py:1022-1028): fma(exp(0.5 clamp(logvar, -30, 20)), noise, mean) (CPU addcmul is fused: test_edit.py), then
                     (z - latents_mean) / latents_std, in fp32; noise is LocalRouter's portable generator at seed 42 and exp
                     the toolkit's portable fdlibm exp (`smath.exp`) rounded to fp32, in place of CPU torch.randn and
                     torch.exp (declared substitutions; test_edit.py reports the exp's distance in ulps)
  mrope_positions    qwen2vl_mrope_position_ids (comfy/text_encoders/qwen_vl.py:121-145) for several pictures

`python -m stk_twin.h3i.edit --fixture PATH` writes the JSON the Zig host tests compare against.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from .. import smath
from ..h3.image import resize_lanczos

F32 = np.float32
CANVAS_MULTIPLE = 32
IMAGE_REF_SHORT_EDGE = 1024          # Qwen's copy, and the VAE's lower bound
IMAGE_REF_VAE_MAX_SHORT_EDGE = 2048
ENCODE_SEED = 42                     # IMAGE_ENCODE_SEED
MASKED_MP, UNMASKED_MP = 1.0, 2.0    # #157's node 11


def load_rgba(px: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """uint8 [h, w, 3 or 4] -> the float32 image [h, w, 3] and, with alpha, the mask [h, w] (1 where transparent)."""
    if px.dtype != np.uint8 or px.ndim != 3 or px.shape[2] not in (3, 4):
        raise ValueError("an RGB or RGBA uint8 image")
    img = np.divide(px[..., :3], 255.0, dtype=F32)
    if px.shape[2] == 3:
        return img, None
    return img, (F32(1.0) - np.divide(px[..., 3], 255.0, dtype=F32)).astype(F32)


def py_round(x: float) -> int:
    return round(x)


def total_pixels_size(w: int, h: int, megapixels: float, steps: int = CANVAS_MULTIPLE) -> tuple[int, int]:
    scale = math.sqrt(megapixels * 1024 * 1024 / (w * h))
    return round(w * scale / steps) * steps, round(h * scale / steps) * steps


def area(img: np.ndarray, W: int, H: int) -> np.ndarray:
    h, w, c = img.shape
    o_h, o_w = np.arange(H), np.arange(W)
    a, b = (o_h * h) // H, ((o_h + 1) * h + H - 1) // H
    lo, hi = (o_w * w) // W, ((o_w + 1) * w + W - 1) // W
    kh, kw = b - a, hi - lo
    s = np.zeros((H, W, c), F32)
    for i in range(int(kh.max())):        # the sum in the kernel's order: window rows, then columns, one rounding each
        for j in range(int(kw.max())):
            take = (i < kh)[:, None] & (j < kw)[None, :]
            v = img[np.minimum(a + i, h - 1)][:, np.minimum(lo + j, w - 1)]
            s = np.where(take[..., None], (s + v).astype(F32), s)
    return ((s / kh.astype(F32)[:, None, None]).astype(F32) / kw.astype(F32)[None, :, None]).astype(F32)


def short_edge_size(w: int, h: int, low: int, high: int) -> tuple[int, int]:
    scale = min(max(min(w, h), low), high) / min(w, h)
    return tuple(max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE) for side in (w, h))


def quantize8(img: np.ndarray) -> np.ndarray:
    return np.clip((F32(255.0) * img).astype(F32), 0, 255).astype(np.uint8)


def lanczos(img: np.ndarray, W: int, H: int) -> np.ndarray:
    return (resize_lanczos(quantize8(img), W, H).astype(F32) / F32(255.0)).astype(F32)


def resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == tuple(size):
        return img
    return lanczos(img, size[0], size[1])


def crop_center(w: int, h: int, W: int, H: int) -> tuple[int, int]:
    old, new = w / h, W / H
    x = y = 0
    if old > new:
        x = round((w - w * (new / old)) / 2)
    elif old < new:
        y = round((h - h * (old / new)) / 2)
    return x, y


def image_scale_center(img: np.ndarray, W: int, H: int) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = crop_center(w, h, W, H)
    return lanczos(img[y:h - y, x:w - x], W, H)


def fma(a, b, c) -> np.float32:
    """a * b + c with one rounding to fp32: the product of two fp32 is exact in fp64 and the fp64 sum is within half an
    fp64 ulp, which can only double-round on a tie of the fp32 rounding; `fma_exact` settles those."""
    a, b, c = F32(a), F32(b), F32(c)
    return fma_exact(a, b, c)


def fma_exact(a, b, c) -> np.float32:
    from fractions import Fraction
    s = np.float64(a) * np.float64(b) + np.float64(c)
    f = F32(s)
    r = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    best = None
    for cand in (np.nextafter(f, F32(-np.inf)), f, np.nextafter(f, F32(np.inf))):
        if not np.isfinite(cand):
            continue
        d = abs(Fraction(float(cand)) - r)
        even = (int(np.asarray(cand, F32).view(np.uint32)) & 1) == 0
        if best is None or d < best[0] or (d == best[0] and even):
            best = (d, cand)
    return F32(best[1])


def _lin_index(n_in: int, n_out: int) -> list[tuple[int, int, np.float32, np.float32]]:
    out = []
    scale = F32(F32(n_in) / F32(n_out))
    for i in range(n_out):
        if n_in == n_out:
            out.append((i, i, F32(1.0), F32(0.0)))
            continue
        src = F32(F32(scale * F32(F32(i) + F32(0.5))) - F32(0.5))
        if src < 0:
            src = F32(0.0)
        i0 = min(int(math.floor(src)), n_in - 1)
        l1 = F32(min(max(F32(src - F32(i0)), F32(0.0)), F32(1.0)))
        out.append((i0, i0 + (1 if i0 < n_in - 1 else 0), F32(F32(1.0) - l1), l1))
    return out


def bilinear(m: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = m.shape
    if (h, w) == (H, W):  # lambda 0 on both axes: t * 1 + t * 0 on either kernel
        return m.astype(F32, copy=True)
    if H + W <= 128:
        raise ValueError("torch takes its vectorized kernel for an output of H + W <= 128; not modelled")
    ry, rx = _lin_index(h, H), _lin_index(w, W)
    xi0 = np.array([r[0] for r in rx]); xi1 = np.array([r[1] for r in rx])
    xl0 = np.array([r[2] for r in rx], F32); xl1 = np.array([r[3] for r in rx], F32)
    out = np.empty((H, W), F32)

    def row(y):  # fma(t0, l0, t1 * l1) element-wise, exactly
        t0, t1 = m[y, xi0].astype(F32), m[y, xi1].astype(F32)
        return _fma_vec(t0, xl0, (t1 * xl1).astype(F32))

    for y, (y0, y1, a0, a1) in enumerate(ry):
        r0, r1 = row(y0), row(y1)
        out[y] = _fma_vec(r0, np.full(W, a0, F32), (r1 * a1).astype(F32))
    return out


def _fma_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    s = a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)
    f = s.astype(F32)
    # fp64 holds a * b exactly; the fp64 sum rounds at most once, so the fp32 result can differ from the fused one only
    # where s sits exactly halfway between two fp32 values after that rounding: redo those exactly
    lo, hi = np.nextafter(f, F32(-np.inf)), np.nextafter(f, F32(np.inf))
    mid_lo = (f.astype(np.float64) + lo.astype(np.float64)) / 2
    mid_hi = (f.astype(np.float64) + hi.astype(np.float64)) / 2
    tie = (s == mid_lo) | (s == mid_hi)
    for k in np.flatnonzero(tie):
        f[k] = fma_exact(a[k], b[k], c[k])
    return f


def composite(dest: np.ndarray, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """dest, src [H, W, 3] fp32, mask [H, W] fp32 (already at the size) -> m * src + (1 - m) * dest."""
    m = mask[..., None]
    inv = (F32(1.0) - m).astype(F32)
    return ((m * src).astype(F32) + (inv * dest).astype(F32)).astype(F32)


def to_u8(img: np.ndarray) -> np.ndarray:
    return quantize8(img)


def portable_noise(seed: int, n: int) -> np.ndarray:
    """`stk_twin.qwen_image.noise` (the portable generator the engine's `qi.noise.fill` implements), compiled from that
    file's `M64`, `uniforms` and `noise` alone: importing the module pulls in the GPU stack, which the CPU path must not."""
    global _NOISE
    if _NOISE is None:
        import ast
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "qwen_image.py"
        tree = ast.parse(path.read_text())
        nodes = [x for x in tree.body if (isinstance(x, ast.FunctionDef) and x.name in ("uniforms", "noise")) or
                 (isinstance(x, ast.Assign) and [getattr(t, "id", None) for t in x.targets] == ["M64"])]
        scope = {"np": np, "smath": smath}
        exec(compile(ast.Module(nodes, []), str(path), "exec"), scope)
        _NOISE = scope["noise"]
    return _NOISE(seed, n)


_NOISE = None


def exp32(x: np.ndarray) -> np.ndarray:
    return np.array([F32(smath.exp(float(v))) for v in x.reshape(-1)], F32).reshape(x.shape)


def posterior(moments: np.ndarray, mean: np.ndarray, std: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """moments fp16 [48, P] (the encoder's canvas), latents_mean / latents_std fp16 [24], noise fp32 [24 * P] -> the
    normalized latent fp32 [24, P]."""
    mo = moments.astype(F32)
    mu, lv = mo[:24], mo[24:]
    e = exp32((F32(0.5) * np.clip(lv, F32(-30.0), F32(20.0))).astype(F32))
    z = _fma_vec(e.reshape(-1), noise.reshape(-1), mu.reshape(-1)).reshape(mu.shape)  # torch.addcmul: one fused rounding
    return ((z - mean.astype(F32)[:, None]).astype(F32) / std.astype(F32)[:, None]).astype(F32)


def mrope_positions(seq_len: int, images: list[tuple[int, int, int]]) -> np.ndarray:
    """(start, gh, gw) of each picture's vision span (gh, gw: the 16-px patch grid; (gh/2)(gw/2) rows) in order -> the
    [3, seq_len] T/H/W positions as ComfyUI builds them, fp32."""
    pos = np.tile(np.arange(seq_len, dtype=F32), (3, 1))
    offset = 0
    for start, gh, gw in images:
        n = (gh // 2) * (gw // 2)
        end = start + n
        len_max = max(1, gh, gw) // 2
        pos[:, end:] = np.arange(start + len_max + offset, start + len_max + (seq_len - end) + offset, dtype=F32)
        pos[0, start:end] = start + offset
        mh, mw = gh // 2, gw // 2
        pos[1, start:end] = np.repeat(np.arange(start + offset, start + mh + offset), math.ceil(n / mh))[:n]
        pos[2, start:end] = np.tile(np.arange(start + offset, start + mw + offset), math.ceil(n / mw))[:n]
        offset += len_max - n
    return pos


# ---------------------------------------------------------------- the fixture

def _hex(a: np.ndarray) -> str:
    return np.ascontiguousarray(a).tobytes().hex()


def _sha(a: np.ndarray) -> str:
    """The SHA-256 of the array's bytes (little-endian, C order): the fixture holds inputs as bytes, outputs as digests."""
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


SIZES = [(1024, 1024), (1216, 832), (832, 1216), (4000, 3000), (3000, 4000), (640, 480), (1920, 1080), (300, 2000),
         (2048, 2048), (1000, 1000), (1023, 769), (512, 512), (4096, 4096), (257, 1999), (1440, 1440), (777, 333)]
MPS = (1.0, 2.0, 0.5)


def fixture() -> dict:
    """Inputs as hex bytes, every output as the SHA-256 of its bytes (`_sha`); the decoded image of the composite is a
    second uint8 image / 255, the posterior's noise the portable generator at seed 42 (the engine draws it itself)."""
    rng = np.random.default_rng(1157)
    out: dict = {}
    out["total_pixels"] = [[w, h, mp, *total_pixels_size(w, h, mp)] for w, h in SIZES for mp in MPS]
    out["short_edge"] = [[w, h, lo, hi, *short_edge_size(w, h, lo, hi)] for w, h in SIZES
                         for lo, hi in ((1024, 1024), (1024, 2048))]
    out["crop"] = [[w, h, W, H, *crop_center(w, h, W, H)] for w, h in SIZES for W, H in ((1024, 1024), (1216, 832), (64, 96))]
    h, w = 37, 61
    rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    img, mask = load_rgba(rgba)
    cases = []
    for W, H in ((40, 24), (90, 56), (61, 37), (128, 64)):
        a = area(img, W, H)
        e = {"w": W, "h": H, "area": _sha(a)}
        e["lanczos"] = [[W2, H2, _sha(resize_to(a, (W2, H2)))] for W2, H2 in ((W, H), (W + 13, H + 7), (max(W // 2, 1), max(H // 3, 1)))]
        x, y = crop_center(W, H, 48, 40)
        e["center"] = [48, 40, x, y, _sha(image_scale_center(a, 48, 40))]
        cases.append(e)
    m = bilinear(mask, 96, 64)
    dec_u8 = rng.integers(0, 256, (96, 64, 3), dtype=np.uint8)
    dec = load_rgba(dec_u8)[0]
    dest = image_scale_center(img, 64, 96)
    comp = composite(dest, dec, m)
    out["image"] = {"w": w, "h": h, "rgba": _hex(rgba), "rgb": _sha(img), "mask": _sha(mask), "cases": cases,
                    "bilinear": [[96, 64, _sha(m)], [37, 61, _sha(bilinear(mask, 37, 61))], [300, 20, _sha(bilinear(mask, 300, 20))]],
                    "decoded": _hex(dec_u8), "dest": _sha(dest), "composite": _sha(comp), "u8": _sha(to_u8(comp))}
    P = 6 * 4
    moments = (rng.standard_normal((48, P)) * np.r_[np.full(24, 1.5), np.full(24, 8.0)][:, None]).astype(np.float16)
    moments[24, :3] = (-40.0, 25.0, 0.0)  # both clamps
    mean = rng.standard_normal(24).astype(np.float16)
    std = (rng.random(24) + 0.5).astype(np.float16)
    n = portable_noise(ENCODE_SEED, 24 * P)
    out["posterior"] = {"p": P, "moments": _hex(moments), "mean": _hex(mean), "std": _hex(std), "noise": _sha(n),
                        "z": _sha(posterior(moments, mean, std, n))}
    seqs = []
    for imgs, L in (([(3, 64, 64)], 1100), ([(3, 64, 90), (1450, 90, 64), (2900, 32, 32)], 3200), ([(5, 2, 8)], 40)):
        seqs.append({"len": L, "images": [list(t) for t in imgs], "pos": _sha(mrope_positions(L, imgs))})
    out["mrope"] = seqs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="where to write the Zig host tests' JSON")
    a = ap.parse_args()
    with open(a.fixture, "w") as f:
        json.dump(fixture(), f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
