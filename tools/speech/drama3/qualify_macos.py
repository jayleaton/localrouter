#!/usr/bin/env python3
"""Qualifies Fish Audio Drama 3 on this Mac (Apple silicon, Metal) and measures it: the macOS counterpart of qualify.sh.

    python3 -I tools/speech/drama3/qualify_macos.py CONFIG OUT [--tool fishaudio/drama-3] [--texts N] \
        [--accept-native-numerics]

CONFIG is a LocalRouter config holding the tool (tools/speech/drama3/config.example-macos.json's shape), whose
`weights` is the ROOT setup-macos.sh filled. Everything runs through drama3_worker_macos.py, the worker the daemon
runs, against the server it starts; run it alone on the machine (no other model loaded), since the memory figures
read the whole machine too. Steps, each recorded in OUT/report.json:

- assets: pins-macos.json's launcher and every checkpoint file by sha256, and the engine binaries against
  ROOT/engine-macos.json;
- load: a cold start, timed from the worker's `load` to the server's readiness;
- e2e: one sentence whole twice (byte-identical) and streamed (the same samples), an unknown voice and inline references
  refused with a 400;
- voices: each configured voice speaks the first N lines of compare-texts.txt, streamed straight from the server the
  worker started: first audio (the first PCM bytes after the request), RTF (wall time over audio seconds), the level,
  each reply 44.1 kHz mono and ending by itself;
- clone: a 13 to 20 s passage in the model's built-in voice (a voice the model invented: no person's) is cloned with
  the worker's `clone` op (drama3-generate --encode beside the running server; its peak footprint and resident size
  measured on a second, separate run under /usr/bin/time -l), then spoken with `voice_references` (the first request
  restarts the server with the voice in its catalog: timed), then measured as a named voice;
- unload: the launcher and the server's group gone, the lock free.

Memory: the server's own `phys_footprint_peak` (footprint(1): every Metal allocation is the process's), sampled
resident size, and the drop in the machine's free+inactive pages (vm_stat) from before the load. Only when every step
passes, warm first audio p50 is at most 0.5 s, warm RTF p50 under 1.0, the peak footprint within the tool's
`resident_bytes`, and --accept-native-numerics is given (the Metal build is a new numerical baseline, not the GB10's),
does it write the tool's `qualification` file, bound to this host, pins-macos.json and the server binary.
"""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import drama3_worker_macos as mac  # noqa: E402

base = mac.base
CLONE_TEXT = ("I keep a small notebook by the window for the birds that visit in the morning. Most days it is "
              "sparrows and a pair of blackbirds, but last week a woodpecker came and stayed for nearly an hour, "
              "tapping at the old apple tree.")
FIRST_AUDIO_GATE_S = 0.5
RTF_GATE = 1.0


def sha256(path):
    return base.sha256_file(path)


def vm_available():
    out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = sum(int(re.search(rf"^{k}:\s+(\d+)\.", out, re.M).group(1)) for k in ("Pages free", "Pages inactive"))
    return page * pages


def footprint(pid):
    """phys_footprint and phys_footprint_peak of a process, bytes."""
    out = subprocess.run(["/usr/bin/footprint", "-p", str(pid), "--noCategories", "-f", "bytes"], capture_output=True,
                         text=True, timeout=60).stdout
    got = dict(re.findall(r"(phys_footprint(?:_peak)?): (\d+) B", out))
    return int(got.get("phys_footprint", 0)), int(got.get("phys_footprint_peak", 0))


def rss(pid):
    out = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else 0


class Sampler:
    """Every 0.5 s: the server's resident size and the machine's free+inactive bytes."""

    def __init__(self):
        self.pid, self.max_rss, self.min_available, self.stop = None, 0, None, threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop.wait(0.5):
            try:
                a = vm_available()
                self.min_available = a if self.min_available is None else min(self.min_available, a)
                if self.pid:
                    self.max_rss = max(self.max_rss, rss(self.pid))
            except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
                pass


def wav_stats(data):
    """(seconds, level dBFS) of a canonical mono 16-bit WAV, else None."""
    rate = base.canonical_wav(data)
    if rate != 44100:
        return None
    pcm = data[base.HEADER:]
    return len(pcm) / 2 / rate, level(pcm)


def level(pcm):
    n = len(pcm) // 2
    if not n:
        return -200.0
    s = struct.unpack(f"<{n}h", pcm[:n * 2])
    mean = sum(x * x for x in s) / n
    return 10 * __import__("math").log10(mean / 32768 ** 2) if mean else -200.0


def stream(port, voice, text):
    """One streamed request straight to the server: first-audio and total seconds, the PCM, the finish reason."""
    body = {"model": base.MODEL, "input": text, "response_format": "pcm", "stream": True}
    if voice:
        body["voice"] = voice
    began = time.monotonic()
    res = base.Http(port, "POST", "/v1/audio/speech", json.dumps(body).encode())
    try:
        if res.status != 200:
            raise RuntimeError(f"HTTP {res.status}: {res.error_message()}")
        first, parts = None, []
        for data in res.chunks():
            if data and first is None:
                first = time.monotonic() - began
            parts.append(data)
        total = time.monotonic() - began
        return first, total, b"".join(parts), res.trailers.get("x-finish-reason", "stop"), \
            int(res.headers.get("x-sample-rate", "0"))
    finally:
        res.close()


def p50(xs):
    return round(statistics.median(xs), 3) if xs else None


class Report:
    def __init__(self, out):
        self.out, self.steps = out, {}

    def add(self, name, ok, **measured):
        self.steps[name] = {"ok": bool(ok), **measured}
        print(f"qualify_macos: {name}: {'pass' if ok else 'FAIL'} {json.dumps(measured)[:400]}", flush=True)

    def passed(self):
        return bool(self.steps) and all(s["ok"] for s in self.steps.values())


def load_tool(path, tool_id):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for t in cfg.get("tools", []):
        if t.get("id") == tool_id:
            return t
    sys.exit(f"qualify_macos: {path} has no tool {tool_id!r}")


def speak(w, job, **speech):
    reply = w.generate({"id": "q", "dir": job, "request": {"speech": speech}})
    data = b""
    if reply.get("ok") and os.path.isfile(os.path.join(job, base.OUTPUT)):
        with open(os.path.join(job, base.OUTPUT), "rb") as f:
            data = f.read()
    return reply, data


def bench_voice(port, voice, texts, audio_dir):
    rows = []
    for i, text in enumerate(texts):
        first, total, pcm, finish, rate = stream(port, voice, text)
        seconds = len(pcm) / 2 / 44100
        rows.append({"first_audio_s": round(first or total, 3), "total_s": round(total, 3),
                     "audio_s": round(seconds, 3), "rtf": round(total / seconds, 3) if seconds else None,
                     "level_dbfs": round(level(pcm), 1), "finish": finish, "rate": rate})
        if i == 0:
            with open(os.path.join(audio_dir, f"{voice or 'default'}.wav"), "wb") as f:
                f.write(base.wav_header(44100, len(pcm)) + pcm)
    warm = rows[1:]
    ok = all(r["rate"] == 44100 and r["finish"] == "stop" and r["audio_s"] > 0.5 and r["level_dbfs"] > -45
             for r in rows)
    return ok, {"first_sentence_first_audio_s": rows[0]["first_audio_s"],
                "warm_first_audio_p50_s": p50([r["first_audio_s"] for r in warm]),
                "warm_rtf_p50": p50([r["rtf"] for r in warm]), "requests": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--tool", default="fishaudio/drama-3")
    ap.add_argument("--texts", type=int, default=5, help="compare-texts.txt lines per voice (the first fills the voice)")
    ap.add_argument("--accept-native-numerics", action="store_true")
    args = ap.parse_args()
    tool = load_tool(args.config, args.tool)
    os.makedirs(args.out, exist_ok=False)
    audio_dir = os.path.join(args.out, "audio")
    os.makedirs(audio_dir)
    report = Report(args.out)
    with open(os.path.join(HERE, "compare-texts.txt"), encoding="utf-8") as f:
        texts = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")][:args.texts]

    # assets
    w = mac.MacWorker({**tool, "options": {**tool.get("options", {}), "verify": "sha256"}}, require_qualification=False)
    o = w.opts
    began = time.monotonic()
    try:
        w._check_files()
        record = json.load(open(os.path.join(o.root, mac.load_pins()["build"]["record"]), encoding="utf-8"))
        engine_ok = record["native_sha256"] == sha256(o.native_bin) and record["generate_sha256"] == sha256(o.generate_bin)
        report.add("assets", engine_ok, seconds=round(time.monotonic() - began, 1), patched_head=record["patched_head"],
                   native_sha256=record["native_sha256"])
    except (OSError, ValueError, KeyError) as e:
        report.add("assets", False, error=str(e))
        return finish(args, tool, report, w)

    # load
    sampler = Sampler()
    w.opts.verify = "size"
    before = vm_available()
    began = time.monotonic()
    loaded = w.load()
    load_s = time.monotonic() - began
    if not loaded.get("ok"):
        report.add("load", False, error=loaded.get("error"))
        return finish(args, tool, report, w)
    pid = sampler.pid = w.unit
    report.add("load", True, seconds=round(load_s, 2), server_pid=pid, available_before_gib=round(before / 2**30, 2))

    # e2e through the worker
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [tempfile.mkdtemp(dir=tmp) for _ in range(5)]
        text = "The train leaves at seven fifteen from platform two."
        a, wav_a = speak(w, jobs[0], input=text, voice="")
        b, wav_b = speak(w, jobs[1], input=text, voice="")
        c, wav_c = speak(w, jobs[2], input=text, voice="", stream=True)
        unknown, _ = speak(w, jobs[3], input=text, voice="Nobody")
        with open(os.path.join(jobs[4], "r.json"), "w") as f:
            json.dump([{"text": "Hi.", "vq_codes": [[1]] * 10}], f)
        refs, _ = speak(w, jobs[4], input=text, references="r.json")
        stats = wav_stats(wav_a)
        report.add("e2e", a.get("ok") and b.get("ok") and c.get("ok") and stats is not None and stats[1] > -45
                   and wav_a == wav_b and wav_a[base.HEADER:] == wav_c[base.HEADER:]
                   and unknown.get("type") == "invalid_request_error" and refs.get("type") == "invalid_request_error",
                   whole_repeat_identical=wav_a == wav_b, streamed_equals_whole=wav_a[base.HEADER:] == wav_c[base.HEADER:],
                   audio_s=stats and round(stats[0], 2), level_dbfs=stats and round(stats[1], 1),
                   unknown_voice=unknown.get("type"), references=refs.get("type"))

    # the named voices, streamed straight from the server
    voices = [v for v in dict.fromkeys(o.voices) if v != "default"]
    results = {}
    for v in voices:
        ok, m = bench_voice(o.port, v, texts, audio_dir)
        results[v] = m
        report.add(f"voice {v}", ok, **{k: m[k] for k in m if k != "requests"})

    # a clone of a model-invented voice (its first request restarts the server: the first server's peak is kept)
    _, peak_named = footprint(w.unit)
    clone = {}
    with tempfile.TemporaryDirectory() as tmp:
        job = tempfile.mkdtemp(dir=tmp)
        _, _, pcm, finish_reason, _ = stream(o.port, None, CLONE_TEXT)
        seconds = len(pcm) / 2 / 44100
        clip = os.path.join(job, "ref0.wav")
        with open(clip, "wb") as f:
            f.write(base.wav_header(44100, len(pcm)) + pcm)
        with open(os.path.join(audio_dir, "clone-source.wav"), "wb") as f:
            f.write(base.wav_header(44100, len(pcm)) + pcm)
        with open(os.path.join(job, "clone.json"), "w") as f:
            json.dump({"name": "Invented", "references": [{"audio": "ref0.wav", "text": CLONE_TEXT}]}, f)
        enc = subprocess.run(["/usr/bin/time", "-l", o.generate_bin, "--codec", os.path.join(o.checkpoint, "codec.pth"),
                              "--ref-audio", clip, "--encode", os.path.join(tmp, "probe.codes")],
                             capture_output=True, text=True, timeout=300)
        enc_rss = re.search(r"(\d+)\s+maximum resident set size", enc.stderr)
        enc_peak = re.search(r"(\d+)\s+peak memory footprint", enc.stderr)
        began = time.monotonic()
        made = w.generate({"id": "c", "dir": job, "request": {"speech": {"input": "", "clone": "clone.json"}}})
        clone_s = time.monotonic() - began
        if not made.get("ok") or not 5 < seconds <= 30:
            report.add("clone", False, source_audio_s=round(seconds, 2), reply=made)
        else:
            job2 = tempfile.mkdtemp(dir=tmp)
            os.replace(os.path.join(job, base.VOICE_OUTPUT), os.path.join(job2, "voice.json"))
            began = time.monotonic()
            first, data = speak(w, job2, input=texts[0], voice="Invented", voice_references="voice.json", stream=True)
            first_s = time.monotonic() - began
            name = sorted(w.loaded - set(voices) - {"default"})
            ok = first.get("ok") and wav_stats(data) is not None and w.reloads == 1 and len(name) == 1
            if ok:
                sampler.pid = w.unit
                bok, m = bench_voice(o.port, name[0], texts, audio_dir)
                results["clone"] = m
                ok = ok and bok
            clone = {"source_audio_s": round(seconds, 2), "source_finish": finish_reason,
                     "clone_op_s": round(clone_s, 2), "encoder_max_rss_bytes": enc_rss and int(enc_rss.group(1)),
                     "encoder_peak_footprint_bytes": enc_peak and int(enc_peak.group(1)),
                     "first_request_with_restart_s": round(first_s, 2), "catalog_name": name[0] if name else None,
                     **({k: v for k, v in results.get("clone", {}).items() if k != "requests"})}
            report.add("clone", ok, **clone)

    # memory, then unload
    now, peak = footprint(w.unit)
    peak = max(peak, peak_named)
    sampler.stop.set()
    lowest = sampler.min_available
    lock = w.environ.get("LOCALROUTER_GPU_LOCK", "") or os.path.join(o.runtime_dir, "drama3.lock")
    server = w.unit
    began = time.monotonic()
    w.exit()
    unload_s = time.monotonic() - began
    gone = subprocess.run(["/bin/ps", "-g", str(server), "-o", "pid="], capture_output=True, text=True).stdout.strip() == ""
    with open(lock, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free = True
        except BlockingIOError:
            free = False
    report.add("unload", gone and free, seconds=round(unload_s, 2), server_group_gone=gone, lock_free=free)
    memory = {"server_phys_footprint_bytes": now, "server_phys_footprint_peak_bytes": peak,
              "server_phys_footprint_peak_named_voices_bytes": peak_named,
              "server_max_rss_sampled_bytes": sampler.max_rss,
              "available_drop_bytes": before - lowest if lowest is not None else None,
              "resident_bytes_configured": tool.get("resident_bytes")}
    report.add("memory", bool(peak) and (not tool.get("resident_bytes") or peak <= tool["resident_bytes"]), **memory)

    named = [results[v] for v in voices if v in results]
    warm_first = [r["first_audio_s"] for m in named for r in m["requests"][1:]]
    warm_rtf = [r["rtf"] for m in named for r in m["requests"][1:]]
    bench = {"load_s": round(load_s, 2), "voices": voices, "texts_per_voice": len(texts),
             "warm_first_audio_p50_s": p50(warm_first), "warm_rtf_p50": p50(warm_rtf),
             "first_sentence_first_audio_s": {v: results[v]["first_sentence_first_audio_s"] for v in results},
             "per_voice": {v: {k: m[k] for k in m if k != "requests"} for v, m in results.items()},
             "clone": clone, "memory": memory}
    report.add("gates", p50(warm_first) is not None and p50(warm_first) <= FIRST_AUDIO_GATE_S
               and p50(warm_rtf) is not None and p50(warm_rtf) < RTF_GATE,
               warm_first_audio_p50_s=bench["warm_first_audio_p50_s"], warm_rtf_p50=bench["warm_rtf_p50"])
    with open(os.path.join(args.out, "requests.json"), "w") as f:
        json.dump(results, f, indent=1)
    return finish(args, tool, report, w, bench)


def finish(args, tool, report, w, bench=None):
    w._stop()
    pins = mac.load_pins()
    chip = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    memsize = int(subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0)
    doc = {"passed": report.passed(), "steps": report.steps, "bench": bench, "chip": chip, "memory_bytes": memsize,
           "macos": platform.mac_ver()[0], "pins_sha256": sha256(mac.PINS),
           "native_sha256": sha256(w.opts.native_bin) if os.path.isfile(w.opts.native_bin) else None,
           "upstream_commit": pins["upstream"]["commit"], "patched_head": pins["tensorfold"]["patched_head"],
           "checkpoint_revision": pins["checkpoint"]["revision"],
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    marker = tool.get("qualification") or ""
    if doc["passed"] and args.accept_native_numerics and marker:
        q = {"qualified": True, "host": os.uname().nodename, "platform": "macos-arm64", "chip": chip,
             "pins_sha256": doc["pins_sha256"], "native_sha256": doc["native_sha256"],
             "upstream_commit": doc["upstream_commit"], "patched_head": doc["patched_head"],
             "checkpoint_revision": doc["checkpoint_revision"],
             "numerics": "accepted: the native Metal build (NVFP4 experts unpacked in its kernels, FP32 accumulation) "
                         "is this host's baseline; not bit-identical with the GB10's CUDA build or the reference",
             "bench": bench, "report": os.path.abspath(os.path.join(args.out, "report.json")),
             "qualified_at": doc["measured_at"]}
        with open(marker + ".tmp", "w") as f:
            json.dump(q, f, indent=1)
            f.write("\n")
        os.replace(marker + ".tmp", marker)
        print(f"qualify_macos: qualified; wrote {marker}")
    elif doc["passed"]:
        print("qualify_macos: passed; no marker written (needs --accept-native-numerics and the tool's qualification)")
    else:
        print(f"qualify_macos: not qualified; see {args.out}/report.json")
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
