"""CPU check: python -m stk_twin.h3i.test_edit --comfy COMFYUI_DIR [--fixture PATH] (no GPU, no checkpoint).

`edit.py`'s host steps against ComfyUI e9afbccc's own code, its source executed as written (extracted from the files as
test_pdd.py does, with stubs only for what the extracted lines call that is not arithmetic), with PIL 13 and CPU torch:
  - ImageScaleToTotalPixels.execute (area) and its size, every bit;
  - _size_short_edge, _resize_to (through _resize, comfy.utils.common_upscale and lanczos: PIL), every bit;
  - ImageScale.upscale (lanczos, crop center), every bit;
  - ImageCompositeMasked.execute (composite, its bilinear mask resize), every bit;
  - LoadImage's float image and mask from 8-bit RGBA, and SaveImage's uint8, every bit;
  - H3ImageVAE.encode's posterior and normalization on given moments and noise: every bit with torch.exp in place of
    the portable exp, and the portable exp's distance from torch.exp in ulps (at most 1);
  - qwen2vl_mrope_position_ids for one to five pictures, every bit.
With --fixture, also that the Zig host test's JSON (src/engines/minimax_h3/edit_image.json) is what edit.py writes.
One JSON line; exit 1 on any difference.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from . import edit

F32 = np.float32


def extract(path: Path, cls: str | None, fn: str, scope: dict):
    """Function `fn` (of class `cls`, or module level) compiled from ComfyUI's source file, its decorators dropped."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == fn)
    node.decorator_list = []
    exec(compile(ast.Module([node], []), str(path), "exec"), scope)
    return scope[fn]


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="ComfyUI at e9afbccc")
    ap.add_argument("--fixture", help="src/engines/minimax_h3/edit_image.json to check")
    a = ap.parse_args()
    root = Path(a.comfy)
    rng = np.random.default_rng(7)
    bad: list[str] = []
    counts: dict[str, int] = {}

    def check(name, ok, detail=""):
        counts[name] = counts.get(name, 0) + 1
        if not ok:
            bad.append(f"{name}: {detail}")

    utils = root / "comfy/utils.py"
    us: dict = {"torch": torch, "np": np, "Image": Image, "math": math}
    lanczos = extract(utils, None, "lanczos", us)
    us["bislerp"] = None
    common_upscale = extract(utils, None, "common_upscale", us)
    repeat = extract(utils, None, "repeat_to_batch_size", us)
    cutils = SimpleNamespace(common_upscale=common_upscale, lanczos=lanczos, repeat_to_batch_size=repeat)
    comfy = SimpleNamespace(utils=cutils)
    node = root / "comfy_extras/nodes_minimax_h3.py"
    ns = {"comfy": comfy, "math": math, "CANVAS_MULTIPLE": 32}
    extract(node, None, "_resize", ns)
    resize_to = extract(node, None, "_resize_to", ns)
    short_edge = extract(node, None, "_size_short_edge", ns)
    io = SimpleNamespace(NodeOutput=lambda *x: x)
    total = extract(root / "comfy_extras/nodes_post_processing.py", "ImageScaleToTotalPixels", "execute",
                    {"math": math, "comfy": comfy, "io": io})
    upscale = extract(root / "nodes.py", "ImageScale", "upscale", {"comfy": comfy})
    helpers = SimpleNamespace(image_alpha_fix=extract(root / "node_helpers.py", None, "image_alpha_fix", {"torch": torch}))
    composite_fn = extract(root / "comfy_extras/nodes_mask.py", None, "composite", {"torch": torch, "comfy": comfy})
    composite_node = extract(root / "comfy_extras/nodes_mask.py", "ImageCompositeMasked", "execute",
                             {"node_helpers": helpers, "composite": composite_fn, "IO": SimpleNamespace(NodeOutput=lambda *x: x)})
    mrope = extract(root / "comfy/text_encoders/qwen_vl.py", None, "qwen2vl_mrope_position_ids", {"torch": torch, "math": math})

    # sizes
    for w, h in edit.SIZES + [(int(x), int(y)) for x, y in rng.integers(64, 5000, (200, 2))]:
        for mp in edit.MPS + (1.37,):
            t = torch.zeros(1, h, w, 3)
            scale = math.sqrt(mp * 1024 * 1024 / (w * h))  # the node's arithmetic, read without running its resize
            ref = (round(w * scale / 32) * 32, round(h * scale / 32) * 32)
            check("total_pixels_size", edit.total_pixels_size(w, h, mp) == ref, (w, h, mp))
            del t
        for lo, hi in ((1024, 1024), (1024, 2048)):
            check("short_edge_size", tuple(edit.short_edge_size(w, h, lo, hi)) == tuple(short_edge(w, h, lo, hi)), (w, h))

    # LoadImage / SaveImage
    alpha = np.arange(256, dtype=np.uint8)
    check("load_divide", same(np.divide(alpha, 255.0, dtype=F32), torch.div(torch.from_numpy(alpha), 255.0).numpy()))
    src_lines = (root / "nodes.py").read_text()
    check("save_source", "i = 255. * image.cpu().numpy()" in src_lines and "np.clip(i, 0, 255).astype(np.uint8)" in src_lines)

    # the image path at small megapixels (the arithmetic is size-independent), then a real 1 MP canvas
    for (h, w, mp) in ((37, 61, 0.004), (120, 90, 0.02), (300, 400, 0.03), (50, 41, 0.01), (700, 500, 1.0)):
        rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        img, mask = edit.load_rgba(rgba)
        timg = torch.from_numpy(rgba[..., :3].copy()).div(255.0)[None]
        tmask = (1.0 - torch.from_numpy(rgba[..., 3:].copy()).div(255.0)[..., -1])[None]
        check("load_rgba", same(img, timg[0].numpy()) and same(mask, tmask[0].numpy()))
        (scaled,) = total(None, timg, "area", mp, 32)
        W, H = edit.total_pixels_size(w, h, mp)
        ours = edit.area(img, W, H)
        check("area", same(ours, scaled[0].numpy()), f"{w}x{h} -> {W}x{H}: {int((ours != scaled[0].numpy()).sum())}")
        for lo, hi in ((1024, 1024), (1024, 2048), (W + 32, W + 64), (32, 64)):
            size = short_edge(W, H, lo, hi)
            ref = resize_to(scaled, size)
            mine = edit.resize_to(ours, size)
            check("resize_to", same(mine, ref[0].numpy()), f"{W}x{H} -> {size}")
        for W2, H2 in ((W, H), (W + 32, H), (max(32, W - 64), H + 32)):
            (dst,) = upscale(None, scaled, "lanczos", W2, H2, "center")
            check("image_scale_center", same(edit.image_scale_center(ours, W2, H2), dst[0].numpy()), f"{W}x{H} -> {W2}x{H2}")
        # the composite at the size of a decode: a decoded image of the node-15 size and the request's mask
        for (Wd, Hd) in ((W, H), (256, 320), (512, 256)):
            if Wd + Hd <= 128:
                continue
            (dst,) = upscale(None, scaled, "lanczos", Wd, Hd, "center")
            dec = torch.from_numpy(rng.random((1, Hd, Wd, 3), dtype=F32))
            (out,) = composite_node(None, dst, dec, 0, 0, False, tmask)
            m = edit.bilinear(mask, Hd, Wd)
            mine = edit.composite(dst[0].numpy(), dec[0].numpy(), m)
            check("composite", same(mine, out[0].numpy()), f"mask {w}x{h} -> {Wd}x{Hd}: {int((mine != out[0].numpy()).sum())}")
            check("to_u8", same(edit.to_u8(mine), np.clip(255. * out[0].numpy(), 0, 255).astype(np.uint8)))

    # the posterior
    vae_src = root / "comfy/ldm/minimax/vae.py"
    check("encode_seed", f"IMAGE_ENCODE_SEED = {edit.ENCODE_SEED}" in vae_src.read_text())
    enc = extract(vae_src, "H3ImageVAE", "encode", {"IMAGE_ENCODE_SEED": edit.ENCODE_SEED})
    worst = 0
    for P_h, P_w in ((4, 6), (16, 16), (32, 20)):
        P = P_h * P_w
        mo = (rng.standard_normal((48, P)) * np.r_[np.full(24, 1.5), np.full(24, 12.0)][:, None]).astype(np.float16)
        mo[24, :2] = (-60.0, 40.0)
        mean = rng.standard_normal(24).astype(np.float16)
        std = (rng.random(24) + 0.5).astype(np.float16)
        n = edit.portable_noise(edit.ENCODE_SEED, 24 * P)
        stub = SimpleNamespace(_adaptive_encode=lambda x: torch.from_numpy(mo).reshape(1, 48, 1, P_h, P_w),
                               _normalize_pixels=lambda x: x,
                               latents_mean=torch.from_numpy(mean.astype(F32)), latents_std=torch.from_numpy(std.astype(F32)))
        tshim = SimpleNamespace(randn=lambda shape, generator=None: torch.from_numpy(n).reshape(shape),
                                Generator=lambda: SimpleNamespace(manual_seed=lambda s: None),
                                exp=torch.exp, addcmul=torch.addcmul)
        enc.__globals__["torch"] = tshim
        ref = enc(stub, torch.zeros(1)).reshape(24, P).numpy()
        saved = edit.exp32
        edit.exp32 = lambda x: torch.exp(torch.from_numpy(np.ascontiguousarray(x))).numpy()
        try:
            with_torch_exp = edit.posterior(mo, mean, std, n)
        finally:
            edit.exp32 = saved
        check("posterior_torch_exp", same(with_torch_exp, ref), int((with_torch_exp != ref).sum()))
        x = (F32(0.5) * np.clip(mo[24:].astype(F32), F32(-30), F32(20))).astype(F32)
        d = np.abs(edit.exp32(x).view(np.int32).astype(np.int64) - torch.exp(torch.from_numpy(x)).numpy().view(np.int32).astype(np.int64))
        worst = max(worst, int(d.max()))
    check("posterior_exp_ulps", worst <= 1, worst)

    # MRoPE positions
    for L, imgs in ((1100, [(3, 64, 64)]), (3200, [(3, 64, 90), (1450, 90, 64), (2900, 32, 32)]),
                    (6000, [(4, 64, 64), (1030, 64, 64), (2056, 64, 96), (3600, 96, 64), (5150, 32, 48)]), (40, [(5, 2, 8)])):
        info = [{"type": "image", "index": s, "size": (gh // 2) * (gw // 2), "extra": {"grid": torch.tensor([[1, gh, gw]])}}
                for s, gh, gw in imgs]
        ref = mrope(info, L, "cpu").numpy()
        check("mrope", same(edit.mrope_positions(L, imgs), ref.astype(F32)) and ref.dtype == np.float32, (L, imgs))

    if a.fixture:
        if json.loads(Path(a.fixture).read_text()) != json.loads(json.dumps(edit.fixture())):
            bad.append(f"{a.fixture} is not what edit.py writes (python -m stk_twin.h3i.edit --fixture ...)")

    print(json.dumps({"ok": not bad, "checks": counts, "exp_max_ulps": worst, "torch": torch.__version__,
                      "pil": Image.__version__, "errors": bad[:20]}))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
