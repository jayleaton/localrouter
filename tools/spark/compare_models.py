"""Image models side by side through a running LocalRouter daemon (the real engines, as an agent calls them).

  python3 tools/spark/compare_models.py --url http://127.0.0.1:8190 --models qwen-image-2.1,qwen-image-2.1-turbo \
      --out DIR [--prompts tools/twin/prompts/compare.json] [--size 1024x1024] [--warm 3]

Each model in turn: every model is released first, so its first request loads it from scratch (cold: worker start,
weights, kernels; the page cache is left alone, so weights read before may come from memory). Then each prompt (with its fixed
seed): one first request, then --warm repeats of the same request, whose median is the warm time. Each model runs with
its own defaults: steps and guidance are not sent. The same seed gives each model the same starting noise, but the
models then follow their own schedules (Qwen-Image 2.1: 25 dynamically shifted steps; Turbo: its 8 stored sigmas), so
their pictures are not comparable step by step. Writes DIR/<model>-<i>.png (the first request's image), DIR/results.json
(per request: wall seconds, the PNG's sha256, whether the repeats were byte-identical; each model's load report from
the worker log) and DIR/summary.md. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path


def call(url: str, path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method="POST" if data is not None or path.endswith("release") else "GET",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_reports(log_dir: Path, model: str) -> list[dict]:
    """The worker's `load ms: {...}` lines for this tool (the engine's own breakdown of a load)."""
    out = []
    f = log_dir / f"{model}.log"
    if f.exists():
        for line in f.read_text(errors="replace").splitlines():
            m = re.search(r"load ms: (\{.*\})", line)
            if m:
                out.append(json.loads(m.group(1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8190")
    ap.add_argument("--models", required=True, help="comma-separated model ids, in run order")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "twin/prompts/compare.json"))
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--logs", help="the daemon's data/logs directory, for the workers' load reports")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(a.prompts))
    listed = {m["id"]: m for m in call(a.url, "/v1/models")["data"]}
    res = {"url": a.url, "size": a.size, "warm_repeats": a.warm, "prompts": prompts,
           "models": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for model in a.models.split(","):
        if model not in listed:
            raise SystemExit(f"{model} is not served at {a.url}: {sorted(listed)}")
        call(a.url, "/v1/tools/release", {})
        runs = []
        for i, (prompt, seed) in enumerate(prompts):
            body = {"model": model, "prompt": prompt, "size": a.size, "seed": seed, "n": 1}
            times, shas = [], []
            for rep in range(1 + a.warm):
                t0 = time.perf_counter()
                r = call(a.url, "/v1/images/generations", body)
                times.append(time.perf_counter() - t0)
                png = base64.b64decode(r["data"][0]["b64_json"])
                shas.append(hashlib.sha256(png).hexdigest())
                if rep == 0:
                    (out / f"{model}-{i}.png").write_bytes(png)
            runs.append({"prompt": i, "seed": seed, "first_s": times[0], "cold": i == 0, "warm_s": times[1:],
                         "warm_median_s": statistics.median(times[1:]) if a.warm else None,
                         "png_sha256": shas[0], "repeats_identical": len(set(shas)) == 1})
            print(json.dumps({"model": model, **runs[-1]}), flush=True)
        res["models"][model] = {"capabilities": listed[model].get("capabilities"), "runs": runs,
                                "load": load_reports(Path(a.logs), model) if a.logs else None}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    lines = [f"# Image models compared ({a.size}, each with its own defaults)", "",
             "| model | prompt | seed | first request s | warm median s (n) | repeats identical |", "| --- | --- | --- | --- | --- | --- |"]
    for model, m in res["models"].items():
        for r in m["runs"]:
            first = f"{r['first_s']:.1f}" + (" (cold: includes the load)" if r["cold"] else "")
            lines.append(f"| {model} | {r['prompt']} | {r['seed']} | {first} | {r['warm_median_s']:.2f} ({len(r['warm_s'])}) | {r['repeats_identical']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"compare": str(out), "models": list(res["models"])}))


if __name__ == "__main__":
    main()
