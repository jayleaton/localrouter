"""python -I -m stk_twin.kandinsky6.vae_conv_oracle --model DIR --src KANDINSKY_SRC --isolate CONV_ISOLATE --out DIR
       [--target conv_out] [--oracle-prefix /opt/k6-oracle]

A decoder convolution's summation order, measured on the device (diagnostic, unqualified). The pinned module's inner
nn.Conv3d runs through its own _conv_forward (the cudnn call the module makes) on an input of the shape the isolate
recorded for the target (`conv_isolate.conv_input_shape`: the padded, upsampled tensor), all weights 2^14, a zero bias.
Its kernels must be the ones the isolate recorded for the target, before and after the queries.

A query (a, b, k) names three elements of the reduction, e = tap * ci + c (tap = (dt * kh + dh) * kw + dw, as the
native fixture's spec): one output position's input is +2^14 at a's voxel, -2^14 at b's and 2^-14 at k's, zero
elsewhere, so the products are +2^28, -2^28 and 1 in fp32. The output is 1 iff k's contribution is added only after +2^28
and -2^28 have met (it is absorbed otherwise: 2^28 + 1 rounds to 2^28 in fp32), and 0 if not. Outputs 3 apart in t, h
and w have disjoint receptive fields: every lattice point answers its own query, ~40,000 a forward for conv_out; every
output channel answers the same query (they must agree).

Measured (a step's elements go into its spec chunk in ascending slots: the order inside one mma is not measured):
  1. deterministic (one batch twice) and uniform (one triple at every lattice point);
  2. signatures: every pair (tap, 2m), (tap, 2m+1) against every other element k. Elements with equal signatures are one
     rounding step (an mma's chunk: its products meet before the accumulator);
  3. per step g, Z(g) = the steps whose k gives 0 for a pair inside g (and its own other elements must all give 0,
     or all 1: measured). In one sequential chain Z(g) is the chain's prefix
     up to g; chains are the union-find components of Z, ordered by |Z| (checks: nested prefixes, no ties);
  4. with three or more chains, the order they are joined in (a, b, k in three different chains; checked sequential).
The model holds when all of these check and no query gave anything but 0 or 1. The k6.aux.vae.conv.spec op needs
all but the join order's check (the fixture also runs the other joins).

Recorded in the twin format (--out, a new directory):
  k6.aux.vae.conv.oracle.<target>  outs signatures int8 [pairs][elements] (0, 1; 2 another value; 3 the output channels
                          disagree; a pair's own elements 0), pairs int32 [pairs][2]; attrs ci, taps, the input shape
  k6.aux.vae.conv.spec.<target>  when the steps and chains hold (the join order may not): outs elements int32
                          [steps][16] (the step's elements, -1 padded; the steps of each chain in order), chains int32
                          [steps] (the chain's join position);
                          attrs atom (8 when no step has more than 8 elements, else 16), chains, join (measured)
  notes  conv_oracle      the checks, step sizes, chains and their lengths, the kernels, the number of forwards
The native fixture (`check k6-components --stages vae_conv --conv-spec DIR`) runs the spec as its candidates.
Exit 0 when the model holds; otherwise 1 (the signatures are still written).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import AUX, QUALIFICATION
from . import capture as C

BIG, SMALL = 2.0 ** 14, 2.0 ** -14  # with weights 2^14: products 2^28 and 1


def isolate_notes(iso: Path, target: str) -> tuple[dict, list]:
    """The isolate's conv_isolate and conv_kernels notes for the target (ops.jsonl only: its blobs may be gone)."""
    notes = [json.loads(ln) for ln in (iso / "ops.jsonl").read_text().splitlines() if ln.strip()]
    shape = [n for n in notes if n.get("note") == "conv_isolate" and n.get("target") == target]
    kern = [n for n in notes if n.get("note") == "conv_kernels" and n.get("target") == target]
    if not shape or not kern:
        raise SystemExit(f"conv_oracle: {iso} has no conv_isolate / conv_kernels note for {target}")
    return shape[-1], kern[-1]["kernels"]


def derive(n: int, Z: list) -> tuple[list, dict]:
    """Z[g] = the steps h whose k gives 0 for a pair inside g -> chains (union-find over Z, each ordered by |Z|) and the
    model's checks. As the audio window-6 oracle's derive."""
    checks = {}  # g in Z(g) holds by construction here; the oracle measures it as own_step_inconsistent
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for g in range(n):
        for h in Z[g]:
            parent[find(g)] = find(h)
    comp = {}
    for g in range(n):
        comp.setdefault(find(g), []).append(g)
    chains, nested, ties = [], True, 0
    for members in comp.values():
        seq = sorted(members, key=lambda g: (len(Z[g]), g))
        for i, g in enumerate(seq):
            if len(Z[g]) != i + 1:
                ties += 1
            if i and not Z[seq[i - 1]] <= Z[g]:
                nested = False
        chains.append(seq)
    chains.sort(key=lambda s: min(s))
    checks.update({"nested_prefixes": nested, "rank_ties": ties})
    return chains, checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--isolate", required=True, help="the conv isolate capture (vae_conv_isolate) with the target's notes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="conv_out")
    ap.add_argument("--oracle-prefix", default="")
    a = ap.parse_args()
    bad = [v for v in C.REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"conv_oracle: refusing to run with {', '.join(bad)} set")
    out, iso = Path(a.out).absolute(), Path(a.isolate)
    if out.exists():
        raise SystemExit(f"conv_oracle: {out} exists")
    from .vae_conv_isolate import CONV_KERNEL, TARGETS
    tgt = {t[0]: t for t in TARGETS}.get(a.target)
    if tgt is None:
        raise SystemExit(f"conv_oracle: unknown target {a.target}; known {sorted(t[0] for t in TARGETS)}")
    shape_note, iso_kernels = isolate_notes(iso, a.target)
    def ident(ks):  # name, grid and block: a split-K variant of one kernel name differs in grid z
        return sorted([k["name"], k.get("grid"), k.get("block")] for k in ks if k.get("name") and CONV_KERNEL.search(k["name"]))
    want_kernels = ident(iso_kernels)
    interp = C.oracle_interpreter(a.oracle_prefix)
    import torch
    from .vae_gn_isolate import kernels
    from ..rec import REC

    oracle = C.oracle_note(torch, interp)
    K = C.upstream(Path(a.src).absolute())
    torch.set_float32_matmul_precision("high")  # factory.py, as capture.py
    cfg = C.pipeline_config(K, Path(a.model).absolute(), "sdpa", "none")
    dev = torch.device("cuda")
    vae = K.vae_video.build_vae(cfg.paths.vae, device=dev)
    conv = dict(vae.named_modules())[tgt[1]].conv
    if conv.weight.dtype != torch.float16 or list(conv.stride) != [1, 1, 1] or list(conv.padding) != [0, 0, 0]:
        raise SystemExit(f"conv_oracle: {tgt[1]} is not an fp16 stride-1 unpadded conv")
    ci, co = conv.in_channels, conv.out_channels
    kt, kh, kw = conv.kernel_size
    taps = kt * kh * kw
    E = taps * ci
    xshape = shape_note["conv_input_shape"]  # [1, ci, T, H, W] as cudnn saw it
    if len(xshape) != 5 or xshape[0] != 1 or xshape[1] != ci:
        raise SystemExit(f"conv_oracle: conv input shape {xshape}")
    _, _, T, H, W = xshape
    to, ho, wo = T - kt + 1, H - kh + 1, W - kw + 1
    # the lattice: outputs (3 apart: kernel <= 3) with disjoint receptive fields
    st, sh, sw = max(kt, 1), max(kh, 1), max(kw, 1)
    lt, lh, lw = torch.arange(0, to, st), torch.arange(0, ho, sh), torch.arange(0, wo, sw)
    pts = torch.cartesian_prod(lt, lh, lw).to(dev)  # [P][3]
    P = pts.shape[0]
    tap_off = torch.tensor([[t // (kh * kw), (t // kw) % kh, t % kw] for t in range(taps)], device=dev)

    x = torch.zeros(xshape, dtype=torch.float16, device=dev)
    xf = x.view(-1)
    w = torch.full(conv.weight.shape, BIG, dtype=torch.float16, device=dev)
    b0 = torch.zeros(co, dtype=torch.float16, device=dev) if conv.bias is not None else None
    out_lin = (pts[:, 0] * ho + pts[:, 1]) * wo + pts[:, 2]  # [P] into one output channel
    stats = {"forwards": 0}

    def voxel(e, p):
        tap, c = e // ci, e % ci
        off = tap_off[tap]
        q = pts[p] + off
        return ((c * T + q[:, 0]) * H + q[:, 1]) * W + q[:, 2]

    def forward():
        stats["forwards"] += 1
        return conv._conv_forward(x, w, b0)

    def query(tri):
        """tri [n][3] int64 (device) elements (a, b, k) -> int8 [n]: 0, 1, 2 (another value, NaN), 3 (channels disagree)"""
        res = torch.empty(tri.shape[0], dtype=torch.int8, device=dev)
        for s in range(0, tri.shape[0], P):
            t3 = tri[s:s + P]
            n = t3.shape[0]
            p = torch.arange(n, device=dev)
            x.zero_()
            for col, val in ((0, BIG), (1, -BIG), (2, SMALL)):
                xf.index_fill_(0, voxel(t3[:, col], p), val)
            y = forward()[0].reshape(co, -1).index_select(1, out_lin[:n]).float()  # [co][n]
            lo, hi = y.amin(0), y.amax(0)
            r = torch.full((n,), 2, dtype=torch.int8, device=dev)
            r[lo == 0] = 0
            r[lo == 1] = 1
            r[(lo != hi) & ~torch.isnan(y).any(0)] = 3  # NaN / inf (an fp16 intermediate overflowed) stays 2
            res[s:s + n] = r
        return res

    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, isolate=str(iso), target=a.target, module=tgt[1])
    REC.note("device", **C.device_note(torch))
    REC.note("precision", **C.precision_note(torch))
    rep = {"target": a.target, "module": tgt[1], "ci": ci, "co": co, "taps": taps, "elements": E, "input_shape": xshape,
           "lattice": P, "isolate_kernels": want_kernels}
    with torch.no_grad():
        def kernel_check(tag):
            got = kernels(torch, forward, out)
            (out / "profile.json").rename(out / f"profile-{tag}.json")
            return {"kernels": got, "same": ident(got) == want_kernels}

        rep["kernel_check_before"] = kernel_check("before")
        if not rep["kernel_check_before"]["same"]:
            REC.note("conv_oracle", **rep, model_holds=False)
            REC.stop()
            raise SystemExit(f"conv_oracle: the oracle's call runs other kernels than the isolate's: {rep['kernel_check_before']}")
        g = torch.Generator(device="cpu").manual_seed(0)
        probe = torch.stack([torch.randperm(E, generator=g)[:3] for _ in range(P)]).to(dev)
        rep["deterministic"] = bool(torch.equal(query(probe), query(probe)))
        one = torch.tensor([[0, 1, E - 1], [E - 1, E - 2, 0], [ci, ci + 1, 2]], device=dev)
        uni = [query(one[i:i + 1].expand(P, 3).contiguous()) for i in range(one.shape[0])]
        rep["uniform"] = all(bool((u == u[0]).all()) for u in uni)
        rep["uniform_values"] = [int(u[0]) for u in uni]
        # 2. signatures: pair (tap, 2m), (tap, 2m+1) against every element
        pa = torch.arange(0, E, 2, device=dev)
        pairs = torch.stack([pa, pa + 1], 1)  # c even: c + 1 is the same tap's next channel (ci is even)
        npairs = pairs.shape[0]
        ks = torch.arange(E, device=dev)
        sig = torch.zeros(npairs, E, dtype=torch.int8, device=dev)
        rows = max(1, (4 * P) // E)  # a few forwards' worth of pairs at once
        for s in range(0, npairs, rows):
            pr = pairs[s:s + rows]
            n = pr.shape[0]
            tri = torch.cat([pr.repeat_interleave(E, 0), ks.repeat(n).unsqueeze(1)], 1)
            keep = (tri[:, 2] != tri[:, 0]) & (tri[:, 2] != tri[:, 1])
            r = torch.zeros(tri.shape[0], dtype=torch.int8, device=dev)
            r[keep] = query(tri[keep])
            sig[s:s + n] = r.view(n, E)
        odd = int((sig >= 2).sum())
        rep["odd_results"] = odd
        rep["odd_by_kind"] = {"other_value": int((sig == 2).sum()), "channels_disagree": int((sig == 3).sum())}
        # steps. Pairs in one step see the same later elements: after(p) = {k: 1}. Whether a step's own other elements
        # give 0 (the mma rounds +2^28 + 1 inside: "truncating") or 1 ("exact") is measured, not assumed: pairs group by
        # after(p) in the first case, by after(p) + their own elements in the second; the coarser grouping is the steps
        import numpy as np
        sg = sig.cpu().numpy()
        prs = pairs.cpu().numpy()
        aft = sg == 1
        own = aft.copy()
        own[np.arange(npairs), prs[:, 0]] = True
        own[np.arange(npairs), prs[:, 1]] = True

        def group(m):
            cls = {}
            for pi in range(npairs):
                cls.setdefault(m[pi].tobytes(), []).append(pi)
            return sorted(cls.values(), key=lambda v: v[0])
        g1, g2 = group(aft), group(own)
        semantics = "truncating" if len(g1) < len(g2) else "exact" if len(g2) < len(g1) else "undetermined"
        pair_steps = g2 if semantics == "exact" else g1
        step_list = [sorted(int(e) for pi in ps for e in prs[pi]) for ps in pair_steps]
        step_list.sort(key=lambda v: v[0])
        of = np.zeros(E, dtype=np.int64)
        for gi, members in enumerate(step_list):
            of[members] = gi
        n = len(step_list)
        sizes = sorted({len(st_) for st_ in step_list})
        rep["intra_step"] = semantics
        rep["steps"] = n
        rep["step_sizes"] = sizes
        # 3. Z per step: the steps whose elements give 0 for the step's pairs, and the step itself. Every pair of a step
        # must give one value per other step (steps_disagree counts the exceptions, a straddling pair among them)
        lo = np.stack([sg[:, st_].min(1) for st_ in step_list], 1)  # [pairs][steps]
        hi = np.stack([sg[:, st_].max(1) for st_ in step_list], 1)
        pstep = of[prs[:, 0]]
        other = np.ones((npairs, n), dtype=bool)
        other[np.arange(npairs), pstep] = False
        disagree = int(((lo != hi) & other).sum())
        # a pair's own step, measured: its other elements all give 0 (truncating) or all 1 (exact)
        own_v = 0 if semantics != "exact" else 1
        own_bad = 0
        for pi in range(npairs):
            rest = [e for e in step_list[int(pstep[pi])] if e not in (prs[pi, 0], prs[pi, 1])]
            own_bad += int(bool(rest) and (sg[pi, rest] != own_v).any())
        rep["own_step_inconsistent"] = own_bad
        Z = [{gi} for gi in range(n)]  # a step is in its own prefix (own_step_inconsistent checks the measurement)
        for pi in range(npairs):
            Z[int(pstep[pi])].update(int(h) for h in np.nonzero((lo[pi] == 0) & (hi[pi] == 0) & other[pi])[0])
        rep["steps_disagree"] = disagree
        chains, checks = derive(n, Z)
        rep["checks"] = checks
        rep["chains"] = len(chains)
        rep["chain_lengths"] = [len(c) for c in chains]
        rep["kernel_check_after"] = kernel_check("after")
        # the steps and chains (written as the spec); the join order is checked on top (the fixture also tries the others)
        steps_ok = (rep["deterministic"] and rep["uniform"] and odd == 0 and disagree == 0 and own_bad == 0
                    and checks["nested_prefixes"] and checks["rank_ties"] == 0 and sizes[-1] <= 16
                    and len(chains) <= 16 and rep["kernel_check_after"]["same"])
        # 4. the join order over three or more chains (measured only when the steps hold); two chains commute
        # in fp32, so their order [0, 1] is not measured (only the fp16 join candidate would depend on it)
        order = list(range(len(chains)))
        rep["join_sequential_consistent"] = True
        rep["join_measured"] = False
        if steps_ok and len(chains) > 2:
            rep["join_measured"] = True
            rp = [step_list[c[0]][0] for c in chains]
            idx = [(i, j, l) for i in range(len(chains)) for j in range(len(chains)) for l in range(len(chains))
                   if len({i, j, l}) == 3]
            res = query(torch.tensor([[rp[i], rp[j], rp[l]] for i, j, l in idx], device=dev)).cpu().tolist()
            table = dict(zip(idx, res))
            score = [0] * len(chains)
            for (i, j, l), r in table.items():
                score[l] += r == 1
            order = sorted(range(len(chains)), key=lambda l: (score[l], l))
            pos = {c: m for m, c in enumerate(order)}
            rep["join_sequential_consistent"] = all((r == 1) == (pos[l] > max(pos[i], pos[j])) for (i, j, l), r in table.items())
            rep["join_table"] = {f"{i},{j},{l}": r for (i, j, l), r in table.items()}
        if rep["join_measured"]:  # the join's forwards ran after the first "after" check
            rep["kernel_check_after_join"] = kernel_check("after-join")
        rep["join_order"] = order
        rep["forwards"] = stats["forwards"]
        holds = (steps_ok and rep["join_sequential_consistent"]
                 and rep.get("kernel_check_after_join", {"same": True})["same"])
        rep["steps_hold"] = bool(steps_ok)
        rep["model_holds"] = bool(holds)
        # the steps as the spec lists them: chains in join order, each chain's steps in order
        listing = [(pos_, gi) for pos_, ch in enumerate(order) for gi in chains[ch]]
        rep["first_steps"] = [{"chain": c, "elements": [[e // ci, e % ci] for e in step_list[gi]]} for c, gi in listing[:8]]
        REC.note("conv_oracle", **rep)
        with REC.op(f"{AUX}vae.conv.oracle.{a.target}", "vae", {"ci": ci, "taps": taps, "input_shape": xshape}) as op:
            op.out(signatures=sig.cpu(), pairs=pairs.to(torch.int32).cpu())
        if steps_ok:
            el = torch.full((len(listing), 16), -1, dtype=torch.int32)
            for j, (_, gi) in enumerate(listing):
                el[j, :len(step_list[gi])] = torch.tensor(step_list[gi], dtype=torch.int32)
            ch = torch.tensor([c for c, _ in listing], dtype=torch.int32)
            with REC.op(f"{AUX}vae.conv.spec.{a.target}", "vae", {"atom": 8 if sizes[-1] <= 8 else 16, "chains": len(chains),
                                                                   "join": order, "join_sequential": rep["join_sequential_consistent"],
                                                                   "module": tgt[1]}) as op:
                op.out(elements=el, chains=ch)
        (out / "oracle.json").write_text(json.dumps(rep, indent=1, default=str))
    REC.note("versions", **C.versions_note(torch))
    REC.stop()
    print(json.dumps({"k6_vae_conv_oracle": str(out), "qualification": QUALIFICATION, "target": a.target,
                      "model_holds": rep["model_holds"], "steps": rep["steps"], "step_sizes": rep["step_sizes"],
                      "chains": rep["chains"], "chain_lengths": rep["chain_lengths"], "join_order": rep["join_order"],
                      "odd_results": rep["odd_results"], "forwards": rep["forwards"]}))
    sys.exit(0 if holds else 1)


if __name__ == "__main__":
    main()
