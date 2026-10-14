"""python -I -m stk_twin.kandinsky6.vae_blend_isolate --src KANDINSKY_SRC --out DIR [--oracle-prefix /opt/k6-oracle]

The video VAE's tile blends, isolated (diagnostic, unqualified): the pinned upstream AutoencoderKLHunyuanVideo.blend_v /
blend_h / blend_t (hunyuan_vae.py; `self` unused) on fp16 decoder-output-like tiles (B 1, C 3), at the extents a decode
plan uses and beyond. Each case runs the method itself on CUDA tensors: `b[..., x] = a[..., -e + x] * (1 - x / e) +
b[..., x] * (x / e)`, fp16 tensors and Python float weights, exactly as the decode does.

Recorded in the twin format (--out, a new directory), per case:
  k6.aux.vae.blend.fixture.<case>  attrs axis (0 time, 1 height, 2 width), extent (as passed; the method clamps it to
                          both tiles), fa ha wa / fb hb wb; outs a, b (before) and y (b after), fp16 [t*h*w][3] channels
                          last (native's tile layout)
The native stage `check k6-components --stages vae_blend --oracle DIR` runs k6v_blend on each and compares with y.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from . import AUX, QUALIFICATION
from . import capture as C

# (case, axis, a dims (t, h, w), b dims, extent)
CASES = (
    ("h_e8", 2, (5, 40, 72), (5, 40, 72), 8),
    ("h_e24", 2, (5, 40, 72), (5, 40, 72), 24),
    ("h_e64", 2, (3, 24, 96), (3, 24, 80), 64),
    ("v_e8", 1, (5, 72, 40), (5, 72, 40), 8),
    ("v_e32", 1, (5, 72, 40), (5, 72, 40), 32),
    ("v_e64", 1, (3, 96, 24), (3, 80, 24), 64),
    ("t_e1", 0, (9, 32, 40), (9, 32, 40), 1),
    ("t_e4", 0, (9, 32, 40), (9, 32, 40), 4),
    ("t_e8", 0, (17, 24, 32), (13, 24, 32), 8),
    ("h_clamped", 2, (5, 40, 16), (5, 40, 72), 24),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-prefix", default="")
    a = ap.parse_args()
    bad = [v for v in C.REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"blend_isolate: refusing to run with {', '.join(bad)} set")
    out = Path(a.out).absolute()
    if out.exists():
        raise SystemExit(f"blend_isolate: {out} exists")
    interp = C.oracle_interpreter(a.oracle_prefix)
    import torch
    from ..rec import REC

    oracle = C.oracle_note(torch, interp)
    C.upstream(Path(a.src).absolute())
    hv = importlib.import_module("kandinsky.core.components.hunyuan_vae")
    cls = hv.AutoencoderKLHunyuanVideo
    method = {0: cls.blend_t, 1: cls.blend_v, 2: cls.blend_h}
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(a.seed)

    def tile(t, h, w):
        # decoder-output-like: mostly in [-1, 1], some beyond (conv_out is unclamped before postprocess)
        x = torch.randn(1, 3, t, h, w, generator=g) * 0.6
        return x.clamp(-1.6, 1.6).to(torch.float16).to(dev)

    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, seed=a.seed, cases=[c[0] for c in CASES])
    REC.note("device", **C.device_note(torch))
    with torch.no_grad():
        for name, axis, da, db, ext in CASES:
            ta, tb = tile(*da), tile(*db)
            a0, b0 = ta.clone(), tb.clone()
            y = method[axis](None, ta, tb, ext)
            torch.cuda.synchronize()
            if not torch.equal(ta.view(torch.int16), a0.view(torch.int16)):
                raise SystemExit(f"blend_isolate: {name}: the method changed a")
            rows = {k: C.trace_rows(v, "ncthw").cpu() for k, v in (("a", a0), ("b", b0), ("y", y))}
            with REC.op(f"{AUX}vae.blend.fixture.{name}", "vae", {"axis": axis, "extent": ext, "fa": da[0], "ha": da[1],
                                                                  "wa": da[2], "fb": db[0], "hb": db[1], "wb": db[2]}) as op:
                op.out(**rows)
    REC.note("versions", **C.versions_note(torch))
    REC.stop()
    print(json.dumps({"k6_vae_blend_isolate": str(out), "qualification": QUALIFICATION, "cases": len(CASES)}))


if __name__ == "__main__":
    main()
