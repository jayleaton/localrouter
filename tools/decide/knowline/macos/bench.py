"""Runs corpus.json through a KnowLine /v1/systemone server and records every answer's probabilities with its latency;
compares two such records. The macOS baseline (../README.md#macos) is two of these runs, bf16 and the 5-bit MLX quant.

    bench.py run OUT.json --front-end DIR [--weights DIR] [--passes 2]   starts serve_mlx.py (wrap it in gpulock), times
                                                                         the load, measures the peak footprint, stops it
    bench.py run OUT.json --url http://127.0.0.1:PORT --model ID         an already running server (the Spark's, say)
    bench.py compare REF.json OTHER.json                                 choice agreement, probability deltas, accuracy

A record's `answers` are the last pass's; every pass's answers must be identical (the scorer is deterministic), which
the run checks and reports. Latency is per request (all its questions), wall clock at the client.
"""

import argparse
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def post(url, body, timeout=300):
    req = urllib.request.Request(f"{url}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def start(front_end, weights, port):
    cmd = [sys.executable, "-I", os.path.join(HERE, "serve_mlx.py"), front_end, "--port", str(port),
           "--served-model-name", "knowline-4b"] + (["--weights", weights] if weights else [])
    t = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr)
    while True:
        if p.poll() is not None:
            sys.exit(f"the server exited ({p.returncode}) before it was healthy")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return p, time.monotonic() - t
        except OSError:
            time.sleep(0.1)


def run(a):
    cases = json.load(open(os.path.join(HERE, "corpus.json")))["cases"]
    proc, load_s, url, model = None, None, a.url, a.model
    if not url:
        import memory
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        proc, load_s = start(a.front_end, a.weights, port)
        url, model = f"http://127.0.0.1:{port}", "knowline-4b"
        loaded = memory.footprint(proc.pid)
    try:
        passes = []
        for _ in range(a.passes):
            out = {}
            for c in cases:
                t = time.monotonic()
                code, r = post(url, dict(c["request"], model=model))
                ms = (time.monotonic() - t) * 1000
                if code != 200:
                    sys.exit(f"{c['id']}: {code} {r}")
                out[c["id"]] = {"ms": ms, "answers": r["answers"], "input_tokens": r["usage"]["input_tokens"]}
            passes.append(out)
        mem = {}
        if proc:
            mem = {"loaded_footprint": loaded, "serving_footprint": memory.footprint(proc.pid),
                   "peak_footprint": memory.peak(proc.pid)}
    finally:
        if proc:
            proc.send_signal(signal.SIGTERM)
            proc.wait(10)
    last = passes[-1]
    stable = all(p[k]["answers"] == last[k]["answers"] for p in passes for k in last)
    rec = {"weights": a.weights or a.front_end or url, "host": platform.machine() + " " + platform.mac_ver()[0],
           "load_s": load_s, **mem, "deterministic": stable,
           "ms_by_pass": [{k: v["ms"] for k, v in p.items()} for p in passes],
           "cases": {k: {"answers": v["answers"], "input_tokens": v["input_tokens"]} for k, v in last.items()}}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    warm = sorted(rec["ms_by_pass"][-1].values())
    first = rec["ms_by_pass"][0][cases[0]["id"]]
    print(f"{a.out}: {len(cases)} cases, load {load_s and round(load_s, 1)} s, first request {first:.0f} ms, deterministic {stable}, warm request "
          f"median {statistics.median(warm):.0f} ms p90 {warm[int(0.9 * (len(warm) - 1))]:.0f} ms max {warm[-1]:.0f} ms"
          + (f", peak {mem['peak_footprint'] / 2**30:.2f} GiB" if mem else ""))


def dist(ans):
    """A question's answer as {key: probability} (a noul as true/false)."""
    if ans["type"] == "noul":
        return {"true": ans["noul"], "false": 1 - ans["noul"]}
    return ans["probabilities"]


def top(ans):
    d = dist(ans)
    return max(d, key=d.__getitem__)


def compare(a):
    ref, oth = json.load(open(a.ref)), json.load(open(a.other))
    cases = {c["id"]: c for c in json.load(open(os.path.join(HERE, "corpus.json")))["cases"]}
    n = agree = 0
    right = {"ref": 0, "other": 0}
    expected = 0
    deltas, kls, rows = [], [], []
    for cid, rc in ref["cases"].items():
        for qid, ra in rc["answers"].items():
            oa = oth["cases"][cid]["answers"][qid]
            p, q = dist(ra), dist(oa)
            d = max(abs(p[k] - q[k]) for k in p)
            kl = math.fsum(p[k] * math.log(p[k] / max(q[k], 1e-300)) for k in p if p[k] > 0)
            n += 1
            agree += top(ra) == top(oa)
            deltas.append(d)
            kls.append(kl)
            want = cases[cid]["expect"].get(qid)
            if want is not None:
                expected += 1
                right["ref"] += top(ra) == want
                right["other"] += top(oa) == want
            rows.append((d, cid, qid, top(ra), p[top(ra)], top(oa), q[top(oa)], want))
    deltas.sort()
    print(f"{n} questions: same top answer {agree}/{n}; max |dp| per question: median {statistics.median(deltas):.4f}, "
          f"p90 {deltas[int(0.9 * (n - 1))]:.4f}, max {deltas[-1]:.4f}; KL(ref||other) mean {statistics.mean(kls):.5f} "
          f"max {max(kls):.5f}")
    print(f"expected answers right: ref {right['ref']}/{expected}, other {right['other']}/{expected}")
    print("largest deltas (case question: ref top p -> other top p, expected):")
    for d, cid, qid, rt, rp, ot, op, want in sorted(rows, reverse=True)[:a.show]:
        print(f"  {d:.4f}  {cid} {qid}: {rt} {rp:.3f} -> {ot} {op:.3f}  ({want})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--front-end")
    r.add_argument("--weights")
    r.add_argument("--url")
    r.add_argument("--model", default="knowline-4b")
    r.add_argument("--passes", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("other")
    c.add_argument("--show", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "run" and not (a.url or a.front_end):
        p.error("run needs --front-end (start serve_mlx.py) or --url (a running server)")
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
