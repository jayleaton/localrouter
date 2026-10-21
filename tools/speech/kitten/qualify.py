#!/usr/bin/env python3
"""Kitten TTS 2 qualification: the bits gate against the Python reference, and the worker's measurements.

    VENV/bin/python -I tools/speech/kitten/qualify.py gate  ROOT OUT [--fixtures F1,F2] [--device cuda] [--threads N]
    VENV/bin/python -I tools/speech/kitten/qualify.py bench ROOT OUT [--device cuda] [--threads N]

ROOT is the asset root fetch.sh filled. OUT receives report.json and the audio. Every role runs in its own process.

gate compares the worker with the reference, which is kittenml's own public `KittenTTS2.generate` loaded with the
package defaults. It checks three things, all for bitwise equality:
  ops      text normalization, chunks and prompt ids; the language model's prefill logits; the S3 flow's mel; the
           HiFT vocoder's waveform. Each is computed once in a reference process and once in a process that loads the
           model the way the worker does.
  chained  per chunk: the codec tokens the language model sampled, and their decoded samples.
  full     the joined float32 waveform, and the 16-bit PCM in the worker's speech.wav against the reference
           waveform quantized the same way. The worker repeats the first fixture last, so the cold and the warm
           request must match too.
bench drives the worker over the protocol and measures cold load, time to the first chunk, total time, the real-time
factor, peak memory, cancellation (SIGKILL of the worker's group, as the daemon does) and whether memory comes back.
It exits 0 only when every measurement was taken; report.json says what is missing otherwise.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "kitten_worker.py")

FIXTURES = {
    "F1": {"input": "One day, a little girl named Lily found a needle in her room.", "voice": "Bruno", "seed": 7},
    "F2": {"input": "On March 3rd, 2026, the ticket cost $12.50, about 40% less than before.", "voice": "Luna",
           "seed": 11},
    "F3": {"input": "[joyful] We won the grant <laugh> I can (((hardly))) believe it!", "voice": "Kiki",
           "preset": "expressive", "seed": 13},
    "F4": {"input": "Guten Morgen. Ich wünsche dir einen wunderschönen Tag.", "voice": "German", "seed": 17},
    "F5": {"input": ("The lighthouse keeper climbed the spiral stairs every evening, one hundred and twelve steps, "
                     "counting each one aloud as his father had. At the top he trimmed the wick, polished the great "
                     "lens until it shone, and watched the last fishing boats turn for home. Storms came often in "
                     "the autumn. On those nights he did not sleep at all, but sat by the lamp with a mug of tea "
                     "going cold in his hands, listening to the wind test every window in the tower. In the morning "
                     "the sea was always calm again, as if it had forgotten everything, and he would walk down to "
                     "the harbour to hear which boats had made it back."), "voice": "Bruno", "seed": 19},
}
# The bench sentence: about four seconds of speech, one chunk.
BENCH = {"input": FIXTURES["F1"]["input"], "voice": "Bruno", "seed": 7}
BENCH_LONG = dict(FIXTURES["F5"])
LANGUAGE_VOICES = {"Arabic", "Chinese", "French", "German", "Hindi", "Italian", "Portuguese", "Russian", "Spanish"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def tool_config(root, device, threads, keep_float=True, lm_decode="eager"):
    opts = {"device": device, "keep_float": keep_float, "min_speed": 1, "max_speed": 1, "lm_decode": lm_decode}
    if threads:
        opts["threads"] = threads
    return {"id": "kitten-tts-2", "kind": "speech", "weights": root, "options": opts}


# ── roles that load the model (run in their own processes) ──────────────────────────────────────────────────────────

def load_reference(root, device, threads):
    """kittenml's model as a user loads it; only the decoder weights' download is answered from the pinned copy."""
    sys.path.insert(0, HERE)
    import kitten_worker as kw
    pins = kw.load_pins()
    kitten_dir, s3gen_path = kw.asset_paths(root, pins)
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import kittenml.kittentts2.vocoder as vocoder
    from kittenml.kittentts2.model import KittenTTS2
    vocoder.hf_hub_download = lambda **_: s3gen_path
    if threads:
        torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(kitten_dir, "config.json")) as f:
        config = json.load(f)
    return KittenTTS2(kitten_dir, config, device=device)


def load_worker_engine(root, device, threads, lm_decode="eager"):
    """The model loaded by the worker's own code path (pins, offline patch, determinism flags, threads)."""
    sys.path.insert(0, HERE)
    import kitten_worker as kw
    engine = kw.KittenEngine(kw.Options(tool_config(root, device, threads, lm_decode=lm_decode)))
    engine.load()
    return engine.model


def role_ops(args):
    """The individual operations on fixed inputs, dumped as raw bytes."""
    import numpy as np
    import torch
    from kittenml.kittentts2 import text as text_utils
    from kittenml.kittentts2.prompt import build_generation_prompt
    m = (load_reference if args.engine == "reference" else load_worker_engine)(args.root, args.device, args.threads)
    out = {}
    for name in args.fixtures.split(","):
        f = FIXTURES[name]
        spoken = text_utils.normalize_text(f["input"], normalize=f["voice"] not in LANGUAGE_VOICES)
        chunks = text_utils.split_for_synthesis(spoken, max_chars=m.chunk_chars, min_chars=m.chunk_min_chars)
        ref = m._voice_reference(f["voice"])
        prompts = [build_generation_prompt(m.token_map, m.tokenizer.encode(c, add_special_tokens=False),
                                           reference_prefix=ref["prefix"] if m.use_reference_prompt else None)
                   for c in chunks]
        out[f"{name}.text"] = json.dumps({"spoken": spoken, "chunks": chunks}).encode()
        out[f"{name}.prompts"] = np.asarray([t for p in prompts for t in p + [-1]], np.int64).tobytes()
    # The language model's prefill on F1's first prompt, with F1's speaker: the logits at the last position.
    f = FIXTURES["F1"]
    ref = m._voice_reference(f["voice"])
    chunk = text_utils.split_for_synthesis(text_utils.normalize_text(f["input"]), max_chars=m.chunk_chars,
                                           min_chars=m.chunk_min_chars)[0]
    prompt = build_generation_prompt(m.token_map, m.tokenizer.encode(chunk, add_special_tokens=False),
                                     reference_prefix=ref["prefix"])
    with torch.no_grad():
        m.model._pending_speaker_embedding = ref["embedding"]
        try:
            logits = m.model(input_ids=torch.tensor([prompt], device=m.device)).logits[0, -1]
        finally:
            m.model._pending_speaker_embedding = None
    out["lm_prefill_logits"] = logits.float().cpu().numpy().tobytes()
    # The decoder on fixed tokens: the reference voice's own first 100 codec tokens, seeded flow noise.
    tokens = torch.tensor(np.load(m.voices[f["voice"]]["artifacts"])["reference_tokens"][:100].astype(np.int64),
                          device=m.device)
    torch.manual_seed(1234)
    mel = m.codec.s3gen.flow_inference(tokens.unsqueeze(0), ref_dict=ref["conditioning"], n_cfm_timesteps=2,
                                       finalize=True)
    out["s3_flow_mel"] = mel.float().cpu().numpy().tobytes()
    torch.manual_seed(1234)
    wav, _ = m.codec.s3gen.hift_inference(mel.to(dtype=m.codec.s3gen.dtype), None)
    out["hift_wav"] = wav.float().cpu().numpy().tobytes()
    write_blobs(args.out, out)


def role_reference(args):
    """Full generation through the package's public API, with each chunk's tokens and samples recorded."""
    import numpy as np
    m = load_reference(args.root, args.device, args.threads)
    record = {}
    orig = m._generate_tokens
    decode = m.codec.decode

    def tokens_hook(*a, **k):
        ids = orig(*a, **k)
        record.setdefault("tokens", []).append(list(ids))
        return ids

    def decode_hook(cond, ids):
        wav = decode(cond, ids)
        record.setdefault("wavs", []).append(None if wav is None else np.asarray(wav, "<f4").tobytes())
        return wav

    m._generate_tokens, m.codec.decode = tokens_hook, decode_hook
    out = {}
    for name in args.fixtures.split(","):
        f = FIXTURES[name]
        record.clear()
        t = time.monotonic()
        wav = m.generate(f["input"], voice=f["voice"], preset=f.get("preset"),
                         normalize=f["voice"] not in LANGUAGE_VOICES, advanced={"seed": f["seed"]})
        out[f"{name}.ms"] = str(int((time.monotonic() - t) * 1000)).encode()
        out[f"{name}.f32"] = np.asarray(wav, "<f4").tobytes()
        out[f"{name}.tokens"] = json.dumps(record["tokens"]).encode()
        for i, w in enumerate(record["wavs"]):
            out[f"{name}.chunk{i}"] = w or b""
    write_blobs(args.out, out)


def write_blobs(path, blobs):
    os.makedirs(path, exist_ok=True)
    for k, v in blobs.items():
        with open(os.path.join(path, k), "wb") as f:
            f.write(v)


def read_blob(path, name):
    with open(os.path.join(path, name), "rb") as f:
        return f.read()


# ── the worker over the protocol ───────────────────────────────────────────────────────────────────────────────────

class Worker:
    """kitten_worker.py as the daemon runs it: its own process group, config in LOCALROUTER_TOOL, logs to a file."""

    def __init__(self, tool, log):
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(tool), LOCALROUTER_PARENT=str(os.getpid()))
        self.t0 = time.monotonic()
        self.p = subprocess.Popen([sys.executable, "-I", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(log, "w"), text=True, env=env, process_group=0)

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def reply(self, on_progress=None):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker exited")
            o = json.loads(line)
            if "progress" in o:
                if on_progress:
                    on_progress(o["progress"], time.monotonic())
                continue
            return o

    def peak_rss(self):
        with open(f"/proc/{self.p.pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        return 0

    def generate(self, d, speech):
        os.makedirs(d, exist_ok=True)
        events = []
        start = time.monotonic()
        self.send({"op": "generate", "id": os.path.basename(d), "dir": d, "request": {"speech": speech}})
        r = self.reply(lambda p, t: events.append((p["step"], p["of"], t - start)))
        r["wall_s"] = time.monotonic() - start
        r["first_chunk_s"] = events[0][2] if events else None
        r["chunks"] = events[-1][1] if events else 0
        return r

    def exit(self):
        self.send({"op": "exit"})
        return self.p.wait(timeout=60)

    def kill(self):
        os.killpg(self.p.pid, signal.SIGKILL)
        self.p.wait()


def run_role(role, args, out, fixtures):
    cmd = [sys.executable, "-I", os.path.abspath(__file__), role, args.root, out, "--fixtures", fixtures,
           "--device", args.device, "--threads", str(args.threads), "--lm-decode", args.lm_decode]
    if role == "ops":
        cmd += ["--engine", args.engine]
    subprocess.run(cmd, check=True)


def compare(name, a, b, dtype=None):
    """Bitwise equality of two byte strings; on a mismatch the largest difference when they are numbers."""
    r = {"check": name, "equal": a == b, "bytes": len(a), "sha256": sha(a)}
    if a != b:
        r["other_sha256"] = sha(b)
        r["other_bytes"] = len(b)
        if dtype and len(a) == len(b):
            import numpy as np
            x, y = np.frombuffer(a, dtype), np.frombuffer(b, dtype)
            r["max_abs_diff"] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            r["differing"] = int(np.count_nonzero(x != y))
    return r


def op_checks(args):
    """ops: a reference process versus the worker's load path, same fixed inputs, bitwise."""
    checks = []
    for engine in ("reference", "worker"):
        args.engine = engine
        run_role("ops", args, os.path.join(args.out, f"ops-{engine}"), args.fixtures)
    a, b = os.path.join(args.out, "ops-reference"), os.path.join(args.out, "ops-worker")
    for blob in sorted(os.listdir(a)):
        dtype = None if blob.endswith((".text", ".prompts")) else "<f4"
        checks.append({"level": "op", **compare(blob, read_blob(a, blob), read_blob(b, blob), dtype)})
    return checks


def gate(args):
    if args.lm_decode == "graphs":
        sys.path.insert(0, HERE)
        import qualify_graphs
        return qualify_graphs.gate(args)
    import numpy as np
    sys.path.insert(0, HERE)
    import kitten_worker as kw
    os.makedirs(args.out, exist_ok=True)
    names = args.fixtures.split(",")
    checks = op_checks(args)
    # chained and full: the reference's public generate, then the worker over the protocol (cold, then warm).
    ref = os.path.join(args.out, "reference")
    run_role("reference", args, ref, args.fixtures)
    w = Worker(tool_config(args.root, args.device, args.threads), os.path.join(args.out, "worker.log"))
    w.send({"op": "load"})
    loaded = w.reply()
    if not loaded.get("ok"):
        raise RuntimeError(f"worker load failed: {loaded}")
    runs = [(n, f"{n}") for n in names] + [(names[0], f"{names[0]}-warm")]
    timings = {}
    for name, label in runs:
        d = os.path.join(args.out, "worker", label)
        r = w.generate(d, dict(FIXTURES[name]))
        if not r.get("ok"):
            raise RuntimeError(f"worker {label}: {r}")
        timings[label] = {k: r[k] for k in ("ms", "wall_s", "first_chunk_s", "chunks", "seed")}
        chunks = json.load(open(os.path.join(d, "speech.chunks.json")))
        ref_tokens = json.loads(read_blob(ref, f"{name}.tokens"))
        checks.append({"level": "chained", **compare(f"{label}.tokens", json.dumps(ref_tokens).encode(),
                                                    json.dumps([c["tokens"] for c in chunks]).encode())})
        wav = read_blob(d, "speech.f32")
        checks.append({"level": "full", **compare(f"{label}.f32", read_blob(ref, f"{name}.f32"), wav, "<f4")})
        pcm = kw.pcm16(np.frombuffer(read_blob(ref, f"{name}.f32"), "<f4")).tobytes()
        import wave
        with wave.open(os.path.join(d, "speech.wav")) as wf:
            fmt = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
            data = wf.readframes(wf.getnframes())
        checks.append({"level": "full", **compare(f"{label}.wav_pcm16", pcm, data, "<i2"), "wav_format": fmt})
        # chunk samples: the worker's join input is not written, so compare the chunk count and lengths instead.
        lens = [len(read_blob(ref, f"{name}.chunk{i}")) // 4 for i in range(len(ref_tokens))]
        checks.append({"level": "chained", "check": f"{label}.chunk_samples",
                       "equal": lens == [c["samples"] for c in chunks], "reference": lens,
                       "worker": [c["samples"] for c in chunks]})
    w.exit()
    import torch
    report = {
        "gate": "kitten-tts-2 worker vs kittenml reference, bitwise",
        "pass": all(c["equal"] for c in checks),
        "fixtures": {n: FIXTURES[n] for n in names},
        "device": args.device, "threads": args.threads, "env": environment(torch),
        "worker_resident": loaded.get("resident"), "timings": timings,
        "reference_ms": {n: int(read_blob(ref, f"{n}.ms")) for n in names},
        "checks": checks,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"pass": report["pass"], "checks": len(checks),
                      "failed": [c["check"] for c in checks if not c["equal"]]}))
    return 0 if report["pass"] else 1


def environment(torch):
    import platform
    import kittenml
    env = {"python": platform.python_version(), "torch": torch.__version__, "kittenml": kittenml.__version__,
           "machine": platform.machine(), "cpu_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    with open("/proc/loadavg") as f:
        env["loadavg"] = f.read().split()[:3]
    return env


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def bench(args):
    """The measurements; exit 0 only when every one was taken (a failed or partial bench is not a measured result)."""
    import torch
    import wave
    os.makedirs(args.out, exist_ok=True)
    tool = tool_config(args.root, args.device, args.threads, keep_float=False, lm_decode=args.lm_decode)
    result = {"device": args.device, "threads": args.threads, "lm_decode": args.lm_decode, "env": environment(torch),
              "missing": []}
    missing = result["missing"]

    def save():
        result["complete"] = not missing
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(result, f, indent=1)

    try:
        before = mem_available()
        w = Worker(tool, os.path.join(args.out, "worker.log"))
        t = time.monotonic()
        w.send({"op": "load"})
        loaded = w.reply()
        if not loaded.get("ok"):
            missing.append(f"load: {loaded}")
            return 1
        result["cold_load_s"] = time.monotonic() - t
        result["process_start_to_ready_s"] = time.monotonic() - w.t0
        result["resident_reported"] = loaded.get("resident")
        result["rss_after_load"] = w.peak_rss()
        result["mem_available_drop_after_load"] = before - mem_available()
        runs = result["runs"] = []
        for label, speech in (("first", BENCH), ("warm1", BENCH), ("warm2", BENCH), ("long", BENCH_LONG)):
            r = w.generate(os.path.join(args.out, label), dict(speech))
            run = {"run": label, "ok": r.get("ok"), "wall_s": r["wall_s"], "first_chunk_s": r["first_chunk_s"],
                   "chunks": r["chunks"], "peak_rss": w.peak_rss()}
            if r.get("ok"):
                with wave.open(os.path.join(args.out, label, "speech.wav")) as wf:
                    run["audio_s"] = wf.getnframes() / wf.getframerate()
                run["rtf"] = r["wall_s"] / run["audio_s"] if run["audio_s"] else None
            if not r.get("ok") or not run.get("audio_s") or run["first_chunk_s"] is None:
                missing.append(f"run {label}: {r}")
            runs.append(run)
        result["exit_code"] = w.exit()
        import re
        with open(os.path.join(args.out, "worker.log")) as f:
            peaks = [int(m) for m in re.findall(r"cuda peak reserved (\d+) MiB", f.read())]
        result["cuda_peak_reserved_mib"] = max(peaks) if peaks else None
        if args.device == "cuda" and not peaks:
            missing.append("cuda peak memory not reported")
        if result["exit_code"] != 0:
            missing.append(f"worker exit {result['exit_code']}")
        # Cancellation: a long request, the worker's group killed once its first chunk is out (as the daemon aborts).
        settled = mem_available()
        w = Worker(tool, os.path.join(args.out, "worker-cancel.log"))
        w.send({"op": "load"})
        if not w.reply().get("ok"):
            missing.append("cancel: load failed")
            return 1
        loaded_avail = mem_available()
        d = os.path.join(args.out, "cancel")
        os.makedirs(d, exist_ok=True)
        w.send({"op": "generate", "id": "cancel", "dir": d, "request": {"speech": dict(BENCH_LONG)}})
        while True:
            line = w.p.stdout.readline()
            if not line:
                missing.append("cancel: worker exited before its first chunk")
                return 1
            o = json.loads(line)
            if "progress" in o:
                break
            if "ok" in o:
                missing.append(f"cancel: finished before it could be cancelled: {o}")
                return 1
        t = time.monotonic()
        w.kill()
        result["cancel"] = {"kill_to_reaped_s": time.monotonic() - t, "mem_available_loaded": loaded_avail,
                            "mem_available_before_load": settled}
        time.sleep(3)
        result["cancel"]["mem_available_after_kill"] = mem_available()
        result["cancel"]["reclaimed_bytes"] = result["cancel"]["mem_available_after_kill"] - loaded_avail
        result["cancel"]["files_left"] = sorted(os.listdir(d))
        return 1 if missing else 0
    except Exception as e:  # a crash is recorded, never silently a partial "result"
        missing.append(f"{type(e).__name__}: {e}")
        return 1
    finally:
        save()
        print(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["gate", "bench", "marker", "ops", "reference"])
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--fixtures", default="F1,F2,F3,F4,F5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--engine", default="reference", choices=["reference", "worker"])
    ap.add_argument("--lm-decode", default="eager", choices=["eager", "graphs"])
    ap.add_argument("--baseline", default=os.path.join(HERE, "baselines"), help="the accepted graphs baselines")
    ap.add_argument("--accept-baseline", action="store_true",
                    help="graphs: record this run's bits as the accepted baseline (the operator's decision)")
    ap.add_argument("--marker", help="marker: the qualification file the tool's config names")
    ap.add_argument("--revoke", action="store_true", help="marker: remove the marker if this run (OUT) wrote it")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    if args.role == "marker":
        sys.path.insert(0, HERE)
        import qualify_graphs
        return qualify_graphs.marker(args)
    roles = {"gate": gate, "bench": bench, "ops": role_ops, "reference": role_reference}
    return roles[args.role](args) or 0


if __name__ == "__main__":
    sys.exit(main())
