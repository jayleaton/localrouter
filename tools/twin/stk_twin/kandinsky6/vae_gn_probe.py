"""python -I -m stk_twin.kandinsky6.vae_gn_probe --out DIR [--seed 0] [--oracle-prefix /opt/k6-oracle]

The video VAE's GroupNorm probed on other data (diagnostic, unqualified). The GroupNorm fixture (vae_gn_isolate) records
one input, and several statistics candidates can tie on it: k6v_gn_welford's variants V1 / V3 / V5 / V7 all matched
the first tile. A tie on one input proves nothing about the others, so here torch's own fp16 nn.GroupNorm(32, C,
eps=1e-6, affine) and the resnet's nn.SiLU run on NCTHW inputs of the decoder's shapes with varied statistics (offsets,
scales, outliers; one shape small enough for torch's one-warp path). Each case is recorded like the fixture.

Recorded in the twin format (--out, a new directory), per case:
  k6.aux.vae.gn.fixture.<case>  attrs groups, note_eps, note_thw; outs x [P][C] channels last, gamma, beta [C], norm
                                and silu [P][C] channels last (vae_gn_isolate's format)
The native stage `check k6-components --stages vae_gn --oracle DIR` runs every statistics candidate on each; a policy
is the candidate exact on every case (and on the isolate's).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import AUX, QUALIFICATION
from . import capture as C

# (case, C, T, H, W, mean, std, outlier scale)
CASES = (
    ("c512_tile_unit", 512, 5, 32, 29, 0.0, 1.0, 0.0),
    ("c512_tile_offset", 512, 5, 32, 29, 3.0, 0.5, 0.0),
    ("c512_tile_wide", 512, 5, 32, 29, -2.0, 6.0, 40.0),
    ("c512_up0", 512, 5, 64, 58, 0.5, 2.0, 0.0),
    ("c256_up2", 256, 9, 128, 116, -0.3, 1.5, 20.0),
    ("c128_up3", 128, 17, 256, 232, 0.1, 0.8, 8.0),
    ("c512_small_warp", 512, 1, 4, 4, 1.0, 3.0, 0.0),
    ("c128_tiny_std", 128, 5, 32, 29, 4.0, 0.01, 0.0),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-prefix", default="")
    a = ap.parse_args()
    bad = [v for v in C.REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"gn_probe: refusing to run with {', '.join(bad)} set")
    out = Path(a.out).absolute()
    if out.exists():
        raise SystemExit(f"gn_probe: {out} exists")
    interp = C.oracle_interpreter(a.oracle_prefix)
    import torch
    from ..rec import REC

    oracle = C.oracle_note(torch, interp)
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, seed=a.seed, cases=[c_[0] for c_ in CASES])
    REC.note("device", **C.device_note(torch))
    act = torch.nn.SiLU()
    with torch.no_grad():
        for name, ch, t, h, w, mean, std, outlier in CASES:
            norm = torch.nn.GroupNorm(32, ch, eps=1e-6, affine=True)
            norm.weight.copy_(1.0 + 0.3 * torch.randn(ch, generator=g))
            norm.bias.copy_(0.2 * torch.randn(ch, generator=g))
            norm = norm.half().to(dev)
            x = torch.randn(1, ch, t, h, w, generator=g) * std + mean
            if outlier:
                hit = torch.rand(1, ch, t, h, w, generator=g) < 1e-3
                x = torch.where(hit, x + outlier * torch.randn(1, ch, t, h, w, generator=g), x)
            x = x.clamp(-60000, 60000).half().to(dev)
            y = norm(x)
            s = act(y)
            torch.cuda.synchronize()
            rows = {k: C.trace_rows(v, "ncthw").cpu() for k, v in (("x", x), ("norm", y), ("silu", s))}
            with REC.op(f"{AUX}vae.gn.fixture.{name}", "vae", {"module": f"nn.GroupNorm(32, {ch})", "groups": 32,
                                                               "note_eps": 1e-6, "note_thw": [t, h, w]}) as op:
                op.out(x=rows["x"], gamma=norm.weight.detach(), beta=norm.bias.detach(), norm=rows["norm"], silu=rows["silu"])
            del x, y, s, rows
            torch.cuda.empty_cache()
    REC.note("versions", **C.versions_note(torch))
    REC.stop()
    print(json.dumps({"k6_vae_gn_probe": str(out), "qualification": QUALIFICATION, "cases": len(CASES)}))


if __name__ == "__main__":
    main()
