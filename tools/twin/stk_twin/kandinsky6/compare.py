"""python -I -m stk_twin.kandinsky6.compare ORACLE NATIVE [--ops PREFIX[,PREFIX...]] [--json FILE] [--allow-alone]

The bit gate between two capture directories in the twin format (stk_twin/rec.py: ops.jsonl + blobs/): the oracle
(stk_twin.kandinsky6.capture) and the native runtime (`localrouter check k6-*`). Ops pair by name and occurrence (the
k-th op of a name on one side with the k-th on the other). Within a pair every role of ins and outs must exist on both
sides with the same dtype, shape and bits (blobs are content-addressed, so equal sha256 is equal bits; otherwise every
element is compared bitwise and the first mismatch and max |difference| are reported). Attrs present on both sides
must agree (floats as float32); `note_*` attrs are informational. A `stage` note whose status is not ok on either side
fails the gate for that stage's ops.

Pass = zero unequal values and zero unmatched ops and roles. k6.aux.* ops are diagnostics: compared when both sides
have them, never unmatched. --ops restricts both sides to ops whose names start with the given prefixes. One line per op,
then one JSON line; exit 1 unless everything passes.

The audio chain k6.audio.latent_in -> mel -> wave -> pcm16 has its own rule (attr `chained`):
  - Full-chain qualification needs every chain op on both sides with chained=true, each op's `in` ref equal by sha256
    to the previous op's `out` on the same side (latent_in.latent -> mel.latent, mel.mel -> wave.mel,
    wave.wave -> pcm16.wave), and every role of every chain op bit-equal.
  - An op with chained=false on either side (a per-job exporter that seeds the operator with its own input) is an
    alone check of that operator. It is still compared role by role, and the report labels it per op, but it never
    counts toward chain qualification.
  - A chain op missing on either side, while any chain op is present, is a failure.
  - A chain that is present but not qualified (alone checks) fails the exit code unless --allow-alone is given; the
    summary's audio_chain says why either way. `chained` itself is not compared as an attr. A pass is labelled unqualified: it is a same-node bit
match, not a qualification of the oracle prefix on the GPU/NGC ABI.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from array import array
from pathlib import Path

from . import AUX, ELEM, QUALIFICATION, numel, read_capture, sha256_file

try:
    import numpy as np
except ImportError:  # the stdlib path is slower but gives the same report
    np = None

CHUNK = 1 << 18  # elements per read
RAW = {"bfloat16": "H", "float16": "H", "float32": "I", "float64": "Q", "uint8": "B", "int8": "b", "bool": "B",
       "int16": "h", "int32": "i", "int64": "q"}
NP_RAW = {"bfloat16": "<u2", "float16": "<u2", "float32": "<u4", "float64": "<u8", "uint8": "u1", "int8": "i1",
          "bool": "u1", "int16": "<i2", "int32": "<i4", "int64": "<i8"}
FLOATS = {"bfloat16", "float16", "float32", "float64"}
# the audio chain in pipeline order; (op, its `in` role, the previous op's `out` role it must equal by sha256)
CHAIN = (("k6.audio.latent_in", None, None), ("k6.audio.mel", "latent", "latent"), ("k6.audio.wave", "mel", "mel"),
         ("k6.audio.pcm16", "wave", "wave"))
CHAIN_NAMES = tuple(c[0] for c in CHAIN)
STAGE_OPS = {"noise": ("k6.noise.",), "rope": ("k6.freqs.", "k6.rope."), "piflow": ("k6.piflow.",),
             "clip": ("k6.clip.",), "text": ("k6.qwen.",), "vae": ("k6.vae.",), "dit0": ("k6.dit.step0",),
             "e2e": ("k6.dit.", "k6.vae.", "k6.audio.", "k6.e2e.")}


def value(raw: int, dtype: str):
    if dtype == "bfloat16":
        return struct.unpack("<f", struct.pack("<I", raw << 16))[0]
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", raw))[0]
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", raw))[0]
    if dtype == "float64":
        return struct.unpack("<d", struct.pack("<Q", raw))[0]
    return raw


def np_values(raw, dtype: str):
    if dtype == "bfloat16":
        return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dtype == "float16":
        return raw.view(np.float16).astype(np.float64)
    if dtype == "float32":
        return raw.view(np.float32).astype(np.float64)
    if dtype == "float64":
        return raw.view(np.float64)
    return raw.astype(np.int64)


def unravel(i: int, shape: list[int]) -> list[int]:
    at = []
    for d in reversed(shape):
        at.append(i % d if d else 0)
        i //= d if d else 1
    return at[::-1]


def diff_blobs(pa: Path, pb: Path, dtype: str, shape: list[int]) -> dict:
    """Bitwise element comparison of two blobs of the same dtype and shape."""
    size, n = ELEM[dtype], numel(shape)
    unequal, nan, maxabs, first = 0, 0, 0.0, None
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        base = 0
        while base < n:
            k = min(CHUNK, n - base)
            ba, bb = fa.read(k * size), fb.read(k * size)
            if ba != bb:
                if np is not None:
                    xa, xb = np.frombuffer(ba, NP_RAW[dtype]), np.frombuffer(bb, NP_RAW[dtype])
                    idx = np.flatnonzero(xa != xb)
                    unequal += int(idx.size)
                    if first is None:
                        j = int(idx[0])
                        first = (base + j, value(int(xa[j]), dtype), value(int(xb[j]), dtype))
                    va, vb = np_values(xa[idx], dtype), np_values(xb[idx], dtype)
                    if dtype in FLOATS:
                        bad = np.isnan(va) | np.isnan(vb)
                        nan += int(bad.sum())
                        va, vb = va[~bad], vb[~bad]
                    if va.size:
                        maxabs = max(maxabs, float(np.max(np.abs(va - vb))))
                else:
                    xa, xb = array(RAW[dtype], ba), array(RAW[dtype], bb)
                    for j, (u, v) in enumerate(zip(xa, xb)):
                        if u == v:
                            continue
                        unequal += 1
                        fu, fv = value(u, dtype), value(v, dtype)
                        if first is None:
                            first = (base + j, fu, fv)
                        if dtype in FLOATS and (math.isnan(fu) or math.isnan(fv)):
                            nan += 1
                        else:
                            maxabs = max(maxabs, abs(fu - fv))
            base += k
    rep = {"unequal": unequal, "of": n, "max_abs_diff": maxabs, "nan_mismatches": nan}
    if first is not None:
        rep["first"] = {"index": first[0], "at": unravel(first[0], shape), "oracle": first[1], "native": first[2]}
    return rep


def f32(x: float) -> bytes:
    return struct.pack("<f", x) if not math.isnan(x) else b"nan"


def same_attr(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        try:
            return f32(float(a)) == f32(float(b))
        except OverflowError:
            return float(a) == float(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_attr(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_attr(x, y) for x, y in zip(a, b))
    return a == b


def compare_pair(ro: Path, rn: Path, oo: dict, on: dict) -> tuple[list[str], dict]:
    problems, roles = [], {}
    for side in ("ins", "outs"):
        a, b = oo.get(side, {}), on.get(side, {})
        for key in sorted(set(a) | set(b)):
            tag = f"{side}.{key}"
            if key not in a or key not in b:
                problems.append(f"{tag}: unmatched (only in {'oracle' if key in a else 'native'})")
                roles[tag] = {"status": "unmatched"}
                continue
            x, y = a[key], b[key]
            if x["dtype"] != y["dtype"] or list(x["shape"]) != list(y["shape"]):
                problems.append(f"{tag}: {x['dtype']}{x['shape']} vs {y['dtype']}{y['shape']}")
                roles[tag] = {"status": "dtype/shape", "oracle": [x["dtype"], x["shape"]], "native": [y["dtype"], y["shape"]]}
                continue
            if x["sha"] == y["sha"]:
                roles[tag] = {"status": "equal", "dtype": x["dtype"], "shape": x["shape"]}
                continue
            d = diff_blobs(ro / "blobs" / x["sha"], rn / "blobs" / y["sha"], x["dtype"], x["shape"])
            roles[tag] = {"status": "unequal" if d["unequal"] else "equal", "dtype": x["dtype"], "shape": x["shape"], **d}
            if d["unequal"]:
                fi = d["first"]
                problems.append(f"{tag}: {d['unequal']}/{d['of']} unequal, first {fi['at']} {fi['oracle']!r} vs "
                                f"{fi['native']!r}, max|d| {d['max_abs_diff']:.6g}")
    ao, an = oo.get("attrs", {}), on.get("attrs", {})
    for key in sorted(set(ao) & set(an)):
        if not key.startswith("note_") and key != "chained" and not same_attr(ao[key], an[key]):
            problems.append(f"attrs.{key}: {ao[key]!r} vs {an[key]!r}")
    return problems, roles


def audio_chain(groups: dict, report: list[dict]) -> dict:
    """The chain rule (module docstring). Labels chain ops in `report` and returns the audio_chain verdict."""
    present = [n for n in CHAIN_NAMES if n in groups]
    if not present:
        return {"status": "absent", "qualified": False, "failure": False, "reasons": []}
    reasons, failure, alone = [], False, []
    by_tag = {r["op"]: r for r in report}
    counts = []
    for name in CHAIN_NAMES:
        lo, ln = groups.get(name, ([], []))
        if not lo or not ln:
            failure = True
            reasons.append(f"{name}: missing on {'both sides' if not lo and not ln else 'oracle' if not lo else 'native'}")
        counts.append(min(len(lo), len(ln)))
    for k in range(min(counts) if not failure else 0):
        prev = None
        for name, in_role, prev_role in CHAIN:
            lo, ln = groups[name]
            pair = (lo[k], ln[k])
            r = by_tag.get(f"{name}#{k}", {})
            flags = [op.get("attrs", {}).get("chained") for op in pair]
            sides_alone = [side for side, f in zip(("oracle", "native"), flags) if f is not True]
            if sides_alone:
                alone.append(f"{name}#{k}")
                r["chain"] = (f"alone check (chained={flags[0]!r} oracle, {flags[1]!r} native): an operator check "
                              "that never counts toward chain qualification")
                reasons.append(f"{name}#{k}: not chained on {', '.join(sides_alone)}")
            else:
                r["chain"] = "chained on both sides"
            if r.get("status") != "pass":
                reasons.append(f"{name}#{k}: roles not all bit-equal")
            if prev is not None:
                for side, op, pop in (("oracle", pair[0], prev[0]), ("native", pair[1], prev[1])):
                    a, b = op.get("ins", {}).get(in_role), pop.get("outs", {}).get(prev_role)
                    if not a or not b or a["sha"] != b["sha"]:
                        reasons.append(f"{name}#{k}: {side} ins.{in_role} is not the previous op's outs.{prev_role}")
                        r.setdefault("chain_links", []).append(f"{side} link broken")
            prev = pair
    qualified = not failure and not reasons
    status = "qualified" if qualified else "failure: chain op missing" if failure else (
        "not qualified: alone checks" if alone else "not qualified")
    return {"status": status, "qualified": qualified, "failure": failure, "alone_ops": alone, "reasons": reasons}


def verify_blobs(root: Path, ops: list[dict]) -> list[str]:
    """Content addressing holds: a blob's name is its sha256."""
    bad, seen = [], set()
    for op in ops:
        for side in ("ins", "outs"):
            for ref in op.get(side, {}).values():
                if ref["sha"] not in seen:
                    seen.add(ref["sha"])
                    if sha256_file(root / "blobs" / ref["sha"]) != ref["sha"]:
                        bad.append(f"{root}: blob {ref['sha']} does not hash to its name")
    return bad


def stage_failures(label: str, notes: list[dict], prefixes: tuple[str, ...]) -> list[str]:
    out = []
    for n in notes:
        if n.get("note") != "stage" or str(n.get("status", "")).startswith("ok"):
            continue
        ops = STAGE_OPS.get(n.get("stage"), ("",))
        if not prefixes or any(o.startswith(p) or p.startswith(o) for o in ops for p in prefixes):
            out.append(f"{label} stage {n.get('stage')}: {n.get('status')}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("oracle")
    ap.add_argument("native")
    ap.add_argument("--ops", default="", help="comma-separated op name prefixes to compare (default: all)")
    ap.add_argument("--json", default="", help="write the full report here")
    ap.add_argument("--allow-alone", action="store_true",
                    help="exit 0 when the audio chain holds only alone checks (chained=false); never qualifies it")
    a = ap.parse_args()
    ro, rn = Path(a.oracle), Path(a.native)
    prefixes = tuple(p.strip() for p in a.ops.split(",") if p.strip())
    errors = []
    try:
        (oops, onotes), (nops, nnotes) = read_capture(ro), read_capture(rn)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(json.dumps({"k6_compare": {"pass": False, "error": f"{type(e).__name__}: {e}"}}))
        sys.exit(1)
    keep = (lambda op: op["name"].startswith(prefixes)) if prefixes else (lambda op: True)
    oops, nops = [op for op in oops if keep(op)], [op for op in nops if keep(op)]
    errors += verify_blobs(ro, oops) + verify_blobs(rn, nops)
    errors += stage_failures("oracle", onotes, prefixes) + stage_failures("native", nnotes, prefixes)
    groups: dict[str, tuple[list, list]] = {}
    for side, ops in ((0, oops), (1, nops)):
        for op in ops:
            groups.setdefault(op["name"], ([], []))[side].append(op)
    report, unmatched, unequal_total, failed = [], [], 0, 0
    for name in sorted(groups):
        lo, ln = groups[name]
        for k in range(max(len(lo), len(ln))):
            tag = f"{name}#{k}"
            if k >= len(lo) or k >= len(ln):
                if name.startswith(AUX):
                    report.append({"op": tag, "status": "aux, one side only"})
                    continue
                unmatched.append(tag)
                failed += 1
                where = "oracle" if k < len(lo) else "native"
                report.append({"op": tag, "status": f"unmatched (only in {where})"})
                print(f"FAIL {tag}: unmatched (only in {where})")
                continue
            problems, roles = compare_pair(ro, rn, lo[k], ln[k])
            unequal_total += sum(r.get("unequal", 0) for r in roles.values())
            report.append({"op": tag, "status": "fail" if problems else "pass", "problems": problems, "roles": roles})
            if problems:
                failed += 1
                print(f"FAIL {tag}: " + "; ".join(problems))
            else:
                print(f"PASS {tag}: {len(roles)} roles bit-equal")
    if not groups:
        errors.append("nothing to compare")
    chain = audio_chain(groups, report)
    for r in report:
        if "chain" in r and r["chain"].startswith("alone"):
            print(f"NOTE {r['op']}: {r['chain']}")
    if chain["status"] != "absent":
        tag = "PASS" if chain["qualified"] else "NOTE" if a.allow_alone and not chain["failure"] else "FAIL"
        print(f"{tag} audio chain: {chain['status']}"
              + (f" ({'; '.join(chain['reasons'])})" if chain["reasons"] else ""))
    if chain["failure"]:
        errors.append("audio chain: " + "; ".join(chain["reasons"]))
    for e in errors:
        print(f"FAIL {e}")
    ok = not errors and failed == 0
    chain_ok = chain["status"] == "absent" or chain["qualified"] or (a.allow_alone and not chain["failure"])
    oracle_pins = next((n.get("pins") for n in onotes if n.get("note") == "oracle"), None)
    summary = {"pass": ok, "qualification": QUALIFICATION, "oracle_prefix": oracle_pins, "oracle": str(ro), "native": str(rn), "ops_prefixes": list(prefixes),
               "pairs": sum(1 for r in report if r["status"] in ("pass", "fail")), "failed": failed,
               "unequal_values": unequal_total, "unmatched": unmatched, "errors": errors, "audio_chain": chain}
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summary, "ops": report}, indent=1))
    print(json.dumps({"k6_compare": summary}))
    sys.exit(0 if ok and chain_ok else 1)


if __name__ == "__main__":
    main()
