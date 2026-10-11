"""Host-only comparison of two D2 record sets (kernels/cuda/kandinsky6/video_vae/QUALIFICATION.md, Record format):
the oracle's and the engine's. Checkpoints are matched by (case, name); a checkpoint is equal when dtype, shape and
sha256 are equal. There is no threshold. For raw blobs that differ it reports, as information only, the first
differing element (in the record's own coordinates: T,H,W,C / H,W,C / N,C), the number of differing elements and the
largest ULP distance (u8: absolute difference).

Oracle-only names: `encode.eps_native` and `encode.latent_native` are pipeline fixtures the engine does not produce;
they are listed, never required. `encode.canvas` is the native resize and crop: the engine's input must equal it, so
when the engine set has no `encode.canvas` the oracle's is compared with the kit input `<case>.image` (--inputs).
Every other oracle checkpoint is expected on the engine side, and engine names the oracle lacks are failures too.

  python compare.py --oracle DIR --engine DIR [--inputs DIR] [--cases s_dec,s_enc,...]
Exit 0 only if every expected checkpoint is equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ORACLE_ONLY = {"encode.eps_native", "encode.latent_native"}
NP = {"f16": "<f2", "bf16": "<u2", "f32": "<f4", "u8": "u1"}
BITS = {"f16": ("<u2", 0x8000), "bf16": ("<u2", 0x8000), "f32": ("<u4", 0x8000_0000)}
AXES = {4: "THWC", 3: "HWC", 2: "NC", 1: "N"}


def load(d: Path) -> dict:
    recs = {}
    for line in (d / "ops.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = (r["case"], r["name"])
        if k in recs:
            raise SystemExit(f"compare: {d}: {k} recorded twice")
        recs[k] = r
    return recs


def blob(d: Path, r: dict) -> np.ndarray | None:
    p = d / "blobs" / r["sha"]
    if not r.get("raw") or not p.exists():
        return None
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != r["sha"]:
        raise SystemExit(f"compare: {p} does not hash to its name")
    return np.frombuffer(data, dtype=NP[r["dtype"]]).reshape(r["shape"])


def ordered(a: np.ndarray, dtype: str) -> np.ndarray:
    """Float bit patterns mapped to integers whose difference is the ULP distance (+0 and -0 coincide)."""
    u, sign = BITS[dtype]
    b = a.view(u).astype(np.int64)
    return np.where(b & sign, -(b & (sign - 1)), b)


def diff(a: np.ndarray, b: np.ndarray, dtype: str) -> dict:
    if dtype == "u8":
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    else:
        d = np.abs(ordered(a, dtype) - ordered(b, dtype))
    ne = d != 0
    first = np.unravel_index(int(np.argmax(ne)), a.shape)
    axes = AXES.get(a.ndim, "?" * a.ndim)
    return {"first": dict(zip(axes, map(int, first))), "count": int(ne.sum()), "of": int(a.size),
            "max_ulp" if dtype != "u8" else "max_abs": int(d.max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--inputs", help="kit inputs (inputs.json): the engine's encode input when it records no canvas")
    ap.add_argument("--cases", help="comma list; default every case in the oracle set")
    a = ap.parse_args()
    od, ed = Path(a.oracle), Path(a.engine)
    O, E = load(od), load(ed)
    cases = a.cases.split(",") if a.cases else sorted({c for c, _ in O})
    index = json.loads((Path(a.inputs) / "inputs.json").read_text()) if a.inputs else None
    rows = []
    for case in cases:
        names = [n for c, n in sorted(O, key=lambda k: O[k]["i"]) if c == case]
        names += [n for c, n in sorted(E, key=lambda k: E[k]["i"]) if c == case and (c, n) not in O]
        if not names:
            rows.append({"case": case, "name": "*", "status": "missing-oracle", "detail": "case absent from the oracle"})
        for n in names:
            o, e = O.get((case, n)), E.get((case, n))
            row = {"case": case, "name": n, "dtype": (o or e)["dtype"], "shape": (o or e)["shape"]}
            if o is None:
                row["status"] = "missing-oracle"
            elif e is None and n in ORACLE_ONLY:
                row["status"] = "oracle-only"
            elif e is None and n == "encode.canvas":
                inp = index and index.get(f"{case}.image")
                if inp is None:
                    row["status"], row["detail"] = "missing-engine", "no engine canvas and no --inputs image"
                else:
                    same = inp["dtype"] == o["dtype"] and inp["shape"] == o["shape"] and inp["sha256"] == o["sha"]
                    row["status"] = "equal" if same else "DIFF"
                    row["detail"] = f"vs kit input {inp['file']}"
            elif e is None:
                row["status"] = "missing-engine"
            elif (o["dtype"], o["shape"]) != (e["dtype"], e["shape"]):
                row["status"] = "DIFF"
                row["detail"] = f"dtype/shape {o['dtype']}{o['shape']} vs {e['dtype']}{e['shape']}"
            elif o["sha"] == e["sha"]:
                row["status"] = "equal"
            else:
                row["status"] = "DIFF"
                x, y = blob(od, o), blob(ed, e)
                row["detail"] = diff(x, y, o["dtype"]) if x is not None and y is not None else "sha differs (hashed only)"
            rows.append(row)
    ok = all(r["status"] in ("equal", "oracle-only") for r in rows)
    w = max([len(r["name"]) for r in rows] + [4])
    print(f"{'case':6} {'name':{w}} {'dtype':5} {'shape':22} status")
    for r in rows:
        det = r.get("detail", "")
        det = json.dumps(det) if isinstance(det, dict) else det
        print(f"{r['case']:6} {r['name']:{w}} {r.get('dtype', ''):5} {str(r.get('shape', '')):22} {r['status']} {det}".rstrip())
    count = {}
    for r in rows:
        count[r["status"]] = count.get(r["status"], 0) + 1
    print(json.dumps({"compare": "pass" if ok else "fail", "oracle": str(od), "engine": str(ed), "cases": cases,
                      "counts": count, "not_equal": [f"{r['case']}/{r['name']}" for r in rows
                                                     if r["status"] not in ("equal", "oracle-only")]}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
