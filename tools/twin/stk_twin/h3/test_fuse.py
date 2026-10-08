"""GPU test of the fused H3 DiT kernels against the unfused ones they replace: `python -m stk_twin.h3.test_fuse`.

The fast schedule of the DiT step (src/engines/minimax_h3/dit.zig, `H3Dit.blocks_fused`) differs from the reference one in
two kernels, and each must give the reference's bits (torch.equal, no tolerance), at H3's shapes:

  * `k_gate_add_norm_mod_` (ops.cu h3_gate_add_norm_mod) == `k_gate_add_` then `k_norm_mod`: both the updated residual x and
    the normalized rows. Both parts of the pair the DiT uses (gate_add1 + norm_mod2: parts 0, 1 under one modulation;
    gate_add2 + the next block's norm_mod1: parts 1, 0 under two block's modulations), with extreme and tiny values, and a
    width that is not a multiple of the block (4100).
  * `int8_attention_rows` (kitchen_launch.cu kitchen_int8_attention_rows) == `int8_attention` transposed to rows
    [S, H * D], at S 200 (the H4 rotation), 400 (no fuse variant), 700, 1500 (cta 128), 5967 (768x448, 56 frames), on
    views of one packed qkv buffer as H3 passes them, plain and with a K offset (the wheel's K shift taken).

Each also runs twice for determinism. Ends with one JSON line; exit status 1 on any failure.
"""

from __future__ import annotations

import json
import sys

import torch

from . import kitchen
from . import ops as H

BF = torch.bfloat16
EPS = 1e-5
CASES: list[dict] = []


def randn(shape, seed, scale=1.0, dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g) * scale).to(dtype)


def report(op: str, shape: str, ok: bool, **extra) -> bool:
    CASES.append(dict(op=op, shape=shape, ok=ok, **extra))
    print(f"{'PASS' if ok else 'FAIL'} {op:18s} {shape:54s} {extra}", flush=True)
    return ok


def gate_norm() -> bool:
    m = H.mod()
    ok = True
    for S, D in ((5967, 5376), (777, 5376), (129, 4100)):
        for variant in ("plain", "outliers", "tiny"):
            x = randn((S, D), 1, 2.0, BF)
            if variant == "outliers":
                x[:, ::97] *= 60
                x[::13] = 0
            elif variant == "tiny":
                x = (x.float() * 1e-4).to(BF)
            y = randn((S, D), 2, 1.0, BF)
            w = torch.rand(D, device="cuda") + 0.5
            gmod, nmod = randn((6, 6, D), 3, 0.3), randn((6, 6, D), 4, 0.3)
            idx = torch.randint(0, 6, (S,), device="cuda", dtype=torch.int32)
            for gpart, npart, same in ((0, 1, True), (1, 0, False), (0, 1, False)):
                nm = gmod if same else nmod  # the same block's two halves share one modulation; across blocks they do not
                xr, hr = x.clone(), torch.empty_like(x)
                m.k_gate_add_(xr, y, gmod, idx, gpart)
                m.k_norm_mod(xr, w, nm, idx, hr, npart, EPS)
                xf, hf = x.clone(), torch.empty_like(x)
                m.k_gate_add_norm_mod_(xf, y, gmod, nm, idx, w, hf, gpart, npart, EPS)
                xg, hg = x.clone(), torch.empty_like(x)
                m.k_gate_add_norm_mod_(xg, y, gmod, nm, idx, w, hg, gpart, npart, EPS)
                eq_x, eq_h = torch.equal(xf, xr), torch.equal(hf, hr)
                det = torch.equal(xf, xg) and torch.equal(hf, hg)
                ok &= report("gate_add_norm_mod", f"S{S} D{D} {variant} gate{gpart} norm{npart} {'1 mod' if same else '2 mods'}",
                             eq_x and eq_h and det, x_equal=eq_x, out_equal=eq_h, deterministic=det)
            del x, y, xr, hr, xf, hf, xg, hg
    return ok


def packed(S, H_, D, seed, biased):
    """q, k, v [1, H, S, D] as strided views into one packed [S, 3 * H * D] buffer, as H3 passes them."""
    qkv = randn((S, 3 * H_ * D), seed, 1.0, BF)
    if biased:
        qkv[:, H_ * D:2 * H_ * D] += randn((1, H_ * D), seed + 1, 4.0, BF)
        qkv[:, :H_ * D] *= 2
    return [qkv[:, i * H_ * D:(i + 1) * H_ * D].view(1, S, H_, D).transpose(1, 2) for i in range(3)]


def attention_rows() -> bool:
    ok = True
    for S in (200, 400, 700, 1500, 5967):
        for biased in (False, True):
            q, k, v = packed(S, 56, 128, 10 + S % 7, biased)
            ref = kitchen.int8_attention(q, k, v)[0].transpose(0, 1).reshape(S, 56 * 128)
            got = kitchen.int8_attention_rows(q, k, v)
            again = kitchen.int8_attention_rows(q, k, v)
            eq, det = torch.equal(got, ref), torch.equal(got, again)
            extra = {} if eq else {"mismatched": int((got != ref).sum())}
            ok &= report("attention_rows", f"S{S} H56 D128 {'biased' if biased else 'plain'}", eq and det, equal=eq,
                         deterministic=det, **extra)
            del q, k, v, ref, got, again
    return ok


def main() -> int:
    ok = gate_norm()
    ok &= attention_rows()
    print(json.dumps({"ok": bool(ok), "checks": len(CASES), "failed": [f"{c['op']} {c['shape']}" for c in CASES if not c["ok"]]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
