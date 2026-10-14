"""python -I -m stk_twin.kandinsky6.vae_sdpa_probe --isolate ATTN_ISOLATE --out DIR [--oracle-prefix /opt/k6-oracle]

The video VAE attention's SDPA kernel probed on the device (diagnostic, unqualified). torch's mem-efficient kernel (the
one the decode's call runs: fmha_cutlassF_f16_aligned_32x128_gmem_sm80, recorded by vae_attn_isolate) is called with
SDPBackend.EFFICIENT_ATTENTION forced on crafted inputs; every call's kernel is checked by name against that one. Inputs
come from the isolate's k6.aux.vae.attn.sdpa op (q, k, v, mask, out [n][512]; one head).

Recorded in the twin format (--out, a new directory), each op with outs q, k, v, out [n][c] and mask [n][n] (fp16) as
k6.aux.vae.attn.sdpa (the native stage `--stages vae_sdpa` replays them all):
  k6.aux.vae.attn.sdpa.real           the recorded call again (must reproduce its out: note sdpa_probe.reproduces)
  k6.aux.vae.attn.sdpa.q0             q = 0: S = 0, every visible p = exp2(0) = 1, mi = 0, s = the visible count; isolates
                                       P V, the per-block output combination and 1 / s
  k6.aux.vae.attn.sdpa.nomask         the recorded q, k, v with a zero mask (spf = n): no -inf, no fully masked blocks
  k6.aux.vae.attn.sdpa.one_block      the first 128 tokens, zero mask: one key block (first and last at once)
  k6.aux.vae.attn.sdpa.q0_one_block   the same with q = 0: one P V chain over 128 keys, then half(O / 128)
notes  sdpa_probe        the kernels of each token count (kernel choice depends on shapes, not values), reproduces
       sdpa_pv_steps     P V's rounding steps measured directly: q = 0 (every P = 1 exactly), n = 256 tokens (two key
                         blocks), no mask; output column c answers one query (a, b, k): V[a][c] = 2^15, V[b][c] = -2^15,
                         V[k][c] = 2^-16 (exact fp16 subnormal), zero elsewhere; out[0][c] = 2^-24 != 0 iff k's product
                         was added after +-2^15 met (it is absorbed otherwise: 2^15 + 2^-16 rounds to 2^15 in fp32, and a
                         31-bit gap is beyond a tensor core's alignment guard bits). Pairs (2m, 2m + 1) against every
                         other key: keys answering alike on every pair but their own two are one step (classes). Every
                         query is also predicted under two models (one chain of k8 steps a key block, or of k16 steps,
                         the blocks' chains joined in order in fp32) and the agreement counted.
Exit 0 when the calls run the recorded kernel and `real` reproduces; otherwise 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import AUX, QUALIFICATION, read_capture
from . import capture as C

KERNEL = "fmha_cutlassF_f16_aligned_32x128_gmem_sm80"
BIG, SMALL = 2.0 ** 15, 2.0 ** -16  # a 31-bit gap: fp32 (24 bits) and an mma's alignment guard bits both drop SMALL


def predict(a: int, b: int, k: int, step) -> int:
    """1 iff k survives, under `step(key) -> (block, step)`: each block one chain of steps from 0 (a step whose addends
    include a +-2^15 drops every 2^-16 addend: truncating mma), the blocks' results joined in order (an fp32 add of a
    2^-16 to a +-2^15 loses it)."""
    val = {a: (1, 0), b: (-1, 0), k: (0, 1)}
    blocks = {}
    for key in (a, b, k):
        bl, st = step(key)
        blocks.setdefault(bl, {}).setdefault(st, []).append(val[key])
    total = (0, 0)
    for bl in sorted(blocks):
        acc = (0, 0)
        for st in sorted(blocks[bl]):
            adds = [acc] + blocks[bl][st]
            big = sum(x[0] for x in adds)
            small = 0 if any(x[0] != 0 for x in adds) else sum(x[1] for x in adds)
            acc = (big, small)
        adds = [total, acc]
        big = sum(x[0] for x in adds)
        small = 0 if any(x[0] != 0 for x in adds) else sum(x[1] for x in adds)
        total = (big, small)
    return 1 if total[1] else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--isolate", required=True, help="the attention isolate capture (vae_attn_isolate) with its blobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-prefix", default="")
    a = ap.parse_args()
    bad = [v for v in C.REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"sdpa_probe: refusing to run with {', '.join(bad)} set")
    out, iso = Path(a.out).absolute(), Path(a.isolate)
    if out.exists():
        raise SystemExit(f"sdpa_probe: {out} exists")
    interp = C.oracle_interpreter(a.oracle_prefix)
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .vae_gn_isolate import kernels
    from ..rec import REC

    oracle = C.oracle_note(torch, interp)
    ops, _ = read_capture(iso)
    rop = next((o for o in ops if o["name"] == f"{AUX}vae.attn.sdpa"), None)
    if rop is None:
        raise SystemExit(f"sdpa_probe: {iso} has no {AUX}vae.attn.sdpa")
    dev = torch.device("cuda")
    q, k, v, mask, want = (C.load_ref(torch, iso, rop["outs"][r]).to(dev) for r in ("q", "k", "v", "mask", "out"))
    n, c = q.shape
    if q.dtype != torch.float16 or mask.shape != (n, n):
        raise SystemExit(f"sdpa_probe: unexpected recorded shapes {tuple(q.shape)} {q.dtype} {tuple(mask.shape)}")
    attrs = rop.get("attrs", {})

    def call(qq, kk, vv, mm):
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            o = F.scaled_dot_product_attention(qq[None, None], kk[None, None], vv[None, None], attn_mask=mm[None, None],
                                               dropout_p=0.0, is_causal=False)
        torch.cuda.synchronize()
        return o[0, 0]

    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, isolate=str(iso), tokens=n, channels=c)
    REC.note("device", **C.device_note(torch))
    REC.note("precision", **C.precision_note(torch))
    shapes, ok = {}, True
    with torch.no_grad():
        def kernel_ok(nn, qq, kk, vv, mm):
            if nn not in shapes:
                ks = kernels(torch, lambda: call(qq, kk, vv, mm), out)
                (out / "profile.json").rename(out / f"profile-n{nn}.json")
                names = [x["name"] for x in ks if x.get("name") and "fmha" in x["name"]]
                shapes[nn] = {"kernels": names, "grids": [x.get("grid") for x in ks if x.get("name") and "fmha" in x["name"]],
                              "same": len(names) == 1 and names[0].startswith(KERNEL)}
            return shapes[nn]["same"]

        def record(name, qq, kk, vv, mm, desc):
            nonlocal ok
            nn = qq.shape[0]
            ok = kernel_ok(nn, qq, kk, vv, mm) and ok
            o = call(qq, kk, vv, mm)
            with REC.op(f"{AUX}vae.attn.sdpa.{name}", "vae", {**attrs, "probe": desc, "tokens": nn}) as op:
                op.out(q=qq.contiguous(), k=kk.contiguous(), v=vv.contiguous(), mask=mm.contiguous(), out=o.contiguous())
            return o

        real = record("real", q, k, v, mask, "the recorded call again")
        reproduces = bool(torch.equal(real.view(torch.int16), want.view(torch.int16)))
        ok = ok and reproduces
        zq = torch.zeros_like(q)
        record("q0", zq, k, v, mask, "q = 0: P = 1 on the visible keys")
        record("nomask", q, k, v, torch.zeros_like(mask), "zero mask")
        m128 = torch.zeros(128, 128, dtype=torch.float16, device=dev)
        record("one_block", q[:128], k[:128], v[:128], m128, "the first 128 tokens, zero mask: one key block")
        record("q0_one_block", zq[:128], k[:128], v[:128], m128, "q = 0, 128 tokens: one P V chain")
        REC.note("sdpa_probe", reproduces=reproduces, kernels=shapes)

        # P V's rounding steps, measured: 256 keys (two blocks), q = 0, no mask; one query a column
        nk = 256
        qz = torch.zeros(nk, c, dtype=torch.float16, device=dev)
        kz = torch.zeros(nk, c, dtype=torch.float16, device=dev)
        mz = torch.zeros(nk, nk, dtype=torch.float16, device=dev)
        ok = kernel_ok(nk, qz, kz, kz, mz) and ok

        def queries(trip):
            res = []
            for s in range(0, len(trip), c):
                batch = trip[s:s + c]
                vv = torch.zeros(nk, c, dtype=torch.float32)
                for col, (x, y, z) in enumerate(batch):
                    vv[x, col], vv[y, col], vv[z, col] = BIG, -BIG, SMALL
                o = call(qz, kz, vv.to(torch.float16).to(dev), mz)[0, :len(batch)].float().cpu()
                res.extend(int(val != 0) for val in o.tolist())
            return res

        pairs = [(2 * m, 2 * m + 1) for m in range(nk // 2)]
        trip = [(x, y, z) for x, y in pairs for z in range(nk) if z not in (x, y)]
        got = queries(trip)
        sig = torch.full((nk, len(pairs)), -1, dtype=torch.int8)
        for (x, y, z), r in zip(trip, got):
            sig[z, x // 2] = r
        # the answers' classes: z ~ z2 when they answer alike on every pair but their own two (union-find)
        parent = list(range(nk))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for z in range(nk):
            for z2 in range(z + 1, nk):
                keep = torch.ones(len(pairs), dtype=torch.bool)
                keep[z // 2] = False
                keep[z2 // 2] = False
                if torch.equal(sig[z][keep], sig[z2][keep]):
                    parent[find(z)] = find(z2)
        cls = {}
        for z in range(nk):
            cls.setdefault(find(z), []).append(z)
        classes = sorted(cls.values(), key=lambda v_: v_[0])
        models = {"k8": lambda key: (key // 128, (key % 128) // 8), "k16": lambda key: (key // 128, (key % 128) // 16)}
        agree = {m: sum(predict(x, y, z, f) == r for (x, y, z), r in zip(trip, got)) for m, f in models.items()}
        mism = {m: [[x, y, z, r] for (x, y, z), r in zip(trip, got) if predict(x, y, z, f) != r][:20] for m, f in models.items()}
        REC.note("sdpa_pv_steps", queries=len(trip), class_sizes=sorted({len(v_) for v_ in classes}), classes_head=classes[:6],
                 agree=agree, mismatches_head=mism)
    REC.note("versions", **C.versions_note(torch))
    REC.stop()
    print(json.dumps({"k6_vae_sdpa_probe": str(out), "qualification": QUALIFICATION, "reproduces": reproduces,
                      "kernels_same": all(s_["same"] for s_ in shapes.values()), "pv_queries": len(trip), "pv_agree": agree,
                      "pv_class_sizes": sorted({len(v_) for v_ in classes})}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
