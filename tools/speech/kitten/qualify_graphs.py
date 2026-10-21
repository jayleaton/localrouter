"""Kitten TTS 2 qualification, part two: the graphs decode's gate, and the marker the daemon's gate reads.

    VENV/bin/python -I tools/speech/kitten/qualify.py gate   ROOT OUT --lm-decode graphs [--accept-baseline]
    VENV/bin/python -I tools/speech/kitten/qualify.py marker ROOT OUT --lm-decode eager|graphs --marker FILE [--revoke]

`lm_decode: "graphs"` runs the language model's decode step as captured CUDA graphs over a static KV cache: other
kernels than kittenml's eager reference, so its codec tokens (and so its audio) differ from the reference once a
step's logits differ. That is a deliberate, named exception to the bit-exact rule (EXCEPTIONS below), and it relaxes
nothing else. The gate checks, and report.json records:
  op        bitwise, as for eager: text, prompts, prefill logits, flow mel, vocoder (the shared load path)
  self      bitwise: the graphs path against itself, cold and warm in one process and across two processes
  baseline  bitwise: against the accepted baseline for this GPU and runtime (baselines/graphs-<gpu>.json); with
            --accept-baseline a missing or different baseline is replaced by this run's, which the report and the
            marker name (the operator's recorded decision; commit the written file to keep it)
  quality   within the exception's named tolerances, per fixture against the eager reference: the audio's length
            and its longest silence
OUT/gate/report.json and OUT/bench/report.json are what `marker` reads.
"""

import hashlib
import json
import os
import re
import time

import qualify as q

EXCEPTION = "kitten-tts-2-graphs-decode"
EXCEPTIONS = {
    EXCEPTION: {
        "scope": "lm_decode graphs only: the decode step's logits (CUDA graphs over a static KV cache) and so the "
                 "sampled codec tokens and audio. Still exact: text, prompts, prefill logits, flow mel and vocoder "
                 "(op); the graphs path against itself (self) and against its accepted baseline (baseline)",
        "tolerances": {"duration_ratio": [0.75, 1.33], "silence_over_reference_s": 0.5, "silence_floor_s": 1.0},
        "evidence": "docs/dev/KITTEN-TTS2.md, Numerics: the accepted GB10 baseline and its listening check",
    },
}
SILENCE_DBFS = -45.0
FRAME_S = 0.02


def sha(b):
    return hashlib.sha256(b).hexdigest()


def longest_silence_s(f32, rate):
    """The longest run of 20 ms frames below -45 dBFS, in seconds."""
    import numpy as np
    x = np.frombuffer(f32, "<f4")
    n = int(rate * FRAME_S)
    frames = x[: len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    silent = rms < 10 ** (SILENCE_DBFS / 20)
    best = run = 0
    for s in silent:
        run = run + 1 if s else 0
        best = max(best, run)
    return best * FRAME_S


def run_worker(args, label, names):
    """The worker with lm_decode graphs over the protocol: load, every fixture, then the first again (warm)."""
    import wave
    d = os.path.join(args.out, label)
    os.makedirs(d, exist_ok=True)
    w = q.Worker(q.tool_config(args.root, args.device, args.threads, lm_decode="graphs"), os.path.join(d, "worker.log"))
    w.send({"op": "load"})
    loaded = w.reply()
    if not loaded.get("ok"):
        raise RuntimeError(f"graphs worker {label}: load failed: {loaded}")
    loaded["load_s"] = time.monotonic() - w.t0
    out = {}
    for name in names + [names[0] + "-warm"]:
        jd = os.path.join(d, name)
        r = w.generate(jd, dict(q.FIXTURES[name.removesuffix("-warm")]))
        if not r.get("ok"):
            raise RuntimeError(f"graphs worker {label} {name}: {r}")
        with wave.open(os.path.join(jd, "speech.wav")) as wf:
            pcm, rate = wf.readframes(wf.getnframes()), wf.getframerate()
        chunks = json.load(open(os.path.join(jd, "speech.chunks.json")))
        out[name] = {"tokens": json.dumps([c["tokens"] for c in chunks]).encode(), "pcm": pcm, "rate": rate,
                     "f32": q.read_blob(jd, "speech.f32"), "samples": [c["samples"] for c in chunks],
                     "wall_s": r["wall_s"]}
    if w.exit() != 0:
        raise RuntimeError(f"graphs worker {label}: exit code")
    return loaded, out


def runtime(torch):
    import transformers
    env = q.environment(torch)
    return {k: env.get(k) for k in ("torch", "kittenml", "cuda", "gpu", "machine")} | {
        "transformers": transformers.__version__}


def baseline_path(args, env):
    slug = re.sub(r"[^a-z0-9]+", "-", (env.get("gpu") or "cpu").lower()).strip("-")
    return os.path.join(args.baseline, f"graphs-{slug}.json")


def gate(args):
    import torch
    os.makedirs(args.out, exist_ok=True)
    names = args.fixtures.split(",")
    checks = q.op_checks(args)
    ref = os.path.join(args.out, "reference")
    q.run_role("reference", args, ref, args.fixtures)
    loaded_a, a = run_worker(args, "graphs-a", names)
    loaded_b, b = run_worker(args, "graphs-b", names)
    # self: cold against warm, and one process against another
    for k in ("tokens", "pcm"):
        cold, warm = a[names[0]][k], a[names[0] + "-warm"][k]
        checks.append({"level": "self", **q.compare(f"{names[0]}.cold-warm.{k}", cold, warm)})
        for n in names:
            checks.append({"level": "self", **q.compare(f"{n}.a-b.{k}", a[n][k], b[n][k])})
    # baseline: this GPU and runtime's accepted bits
    env = runtime(torch)
    record = {"exception": EXCEPTION, "env": env,
              "fixtures": {n: {"tokens_sha256": sha(a[n]["tokens"]), "pcm_sha256": sha(a[n]["pcm"]),
                               "samples": a[n]["samples"]} for n in names}}
    path = baseline_path(args, env)
    try:
        accepted = json.load(open(path))
    except (OSError, ValueError):
        accepted = None
    same = accepted is not None and accepted.get("env") == env and accepted.get("fixtures") == record["fixtures"]
    baseline = {"path": path, "found": accepted is not None, "equal": same,
                "env_equal": accepted is not None and accepted.get("env") == env}
    if not same and args.accept_baseline:
        written = os.path.join(args.out, "baseline", os.path.basename(path))
        os.makedirs(os.path.dirname(written), exist_ok=True)
        with open(written, "w") as f:
            json.dump(record, f, indent=1)
        baseline.update(accepted_now=True, written=written, sha256=sha(open(written, "rb").read()))
    checks.append({"level": "baseline", "check": "graphs baseline", "equal": same or bool(baseline.get("accepted_now")),
                   **baseline})
    # quality: within the exception's tolerances, against the eager reference
    tol = EXCEPTIONS[EXCEPTION]["tolerances"]
    for n in names:
        rate = a[n]["rate"]
        ref_f32 = q.read_blob(ref, f"{n}.f32")
        got_s, ref_s = len(a[n]["f32"]) / 4 / rate, len(ref_f32) / 4 / rate
        ratio = got_s / ref_s if ref_s else None
        ok = ratio is not None and tol["duration_ratio"][0] <= ratio <= tol["duration_ratio"][1]
        checks.append({"level": "quality", "check": f"{n}.duration_ratio", "equal": ok, "graphs_s": got_s,
                       "reference_s": ref_s, "ratio": ratio, "exception": EXCEPTION})
        gap, ref_gap = longest_silence_s(a[n]["f32"], rate), longest_silence_s(ref_f32, rate)
        limit = max(ref_gap + tol["silence_over_reference_s"], tol["silence_floor_s"])
        checks.append({"level": "quality", "check": f"{n}.longest_silence", "equal": gap <= limit, "graphs_s": gap,
                       "reference_s": ref_gap, "limit_s": limit, "exception": EXCEPTION})
        ref_tokens = json.loads(q.read_blob(ref, f"{n}.tokens"))
        got_tokens = json.loads(a[n]["tokens"])
        same_prefix = 0
        for x, y in zip([t for c in ref_tokens for t in c], [t for c in got_tokens for t in c]):
            if x != y:
                break
            same_prefix += 1
        checks.append({"level": "info", "check": f"{n}.tokens_same_as_reference_for", "equal": True,
                       "tokens": same_prefix, "reference_tokens": sum(map(len, ref_tokens))})
    report = {
        "gate": "kitten-tts-2 lm_decode graphs: op exact, self exact, baseline exact, quality within the exception",
        "pass": all(c["equal"] for c in checks), "lm_decode": "graphs", "exceptions": [EXCEPTION],
        "exception_terms": EXCEPTIONS, "baseline": baseline, "env": env, "device": args.device,
        "worker_resident": loaded_a.get("resident"), "worker_load_s": [loaded_a["load_s"], loaded_b["load_s"]],
        "warm_wall_s": a[names[0] + "-warm"]["wall_s"], "checks": checks,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"pass": report["pass"], "checks": len(checks), "baseline": baseline,
                      "failed": [c["check"] for c in checks if not c["equal"]]}))
    return 0 if report["pass"] else 1


def marker(args):
    """Writes the qualification file (`--marker`) from OUT/gate and OUT/bench, or with --revoke removes the one this
    run wrote. The daemon reads only "qualified"; the rest records what was qualified and with which exceptions."""
    import kitten_worker as kw
    path = args.marker
    if not path:
        raise SystemExit("marker: --marker FILE is required")
    if args.revoke:
        try:
            if json.load(open(path)).get("report") == args.out:
                os.remove(path)
                print(f"marker: removed {path}, written by this run")
        except (OSError, ValueError):
            pass
        return 0
    gate_r = json.load(open(os.path.join(args.out, "gate", "report.json")))
    bench_r = json.load(open(os.path.join(args.out, "bench", "report.json")))
    if gate_r.get("pass") is not True or bench_r.get("complete") is not True:
        raise SystemExit(f"marker: no marker: gate pass {gate_r.get('pass')}, bench complete {bench_r.get('complete')}")
    if gate_r.get("lm_decode", "eager") != args.lm_decode or bench_r.get("lm_decode", "eager") != args.lm_decode:
        raise SystemExit(f"marker: the gate and bench did not both run lm_decode {args.lm_decode}")
    if bench_r.get("device") != "cuda":
        raise SystemExit("marker: a qualification is a CUDA run")
    runs = {r["run"]: r for r in bench_r["runs"]}
    exceptions = gate_r.get("exceptions", [])
    record = {"qualified": True, "model": "KittenML/kitten-tts-2", "lm_decode": args.lm_decode,
              "host": os.uname().nodename, "gpu": bench_r["env"].get("gpu"),
              "pins_sha256": sha(open(kw.PINS, "rb").read()),
              "exceptions": exceptions, "exception_evidence": {n: EXCEPTIONS[n]["evidence"] for n in exceptions},
              "baseline": gate_r.get("baseline"),
              "bench": {"cold_load_s": bench_r.get("cold_load_s"), "warm_rtf": runs["warm2"].get("rtf"),
                        "warm_first_chunk_s": runs["warm2"].get("first_chunk_s"), "long_rtf": runs["long"].get("rtf"),
                        "cuda_peak_reserved_mib": bench_r.get("cuda_peak_reserved_mib"),
                        "mem_available_drop_after_load": bench_r.get("mem_available_drop_after_load"),
                        "resident_reported": bench_r.get("resident_reported")},
              "report": args.out, "qualified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    text = json.dumps(record, indent=1) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
    with open(os.path.join(args.out, "marker.json"), "w") as f:
        f.write(text)
    print(f"marker: wrote {path}\n{text}", end="")
    return 0
