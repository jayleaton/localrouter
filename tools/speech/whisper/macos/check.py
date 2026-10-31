"""python tools/speech/whisper/macos/check.py --weights DIR --vad-model silero_vad.jit --fixtures DIR [--report out.json]

The macOS worker over its protocol, each case in its own worker process as the daemon runs it (its own process
group). Real weights: run it under the machine's model lock. The fixtures directory is tools/speech/fixtures' output.

  failure    no VAD model, a VAD model with one byte changed, an unknown vad mode, an unknown dtype, a weights directory
             with no weights (fails after the VAD loaded), a config.json that does not fit the weights: load, load
             again and generate are refused, no transcript.json, the process holds no model (footprint back near its
             start), and it exits 0
  lifecycle  load reports resident (the footprint) and `cancel: "signal"`, a second load is a no-op, exit releases
  refusals   requests the worker must refuse as invalid_request_error and keep serving after: not a transcription,
             no audio, a path outside the job, unreadable or wrong-format WAVs, an unknown language, timestamps,
             decode or temperature out of range, greedy with a temperature; protocol errors for a bad line or op
  shape      text, language, duration, segments (id, start, end, text), words only for timestamps "word" (in order,
             inside the audio and its segments), vad (model, speech, dropped_segments); silence gated to no text;
             a fixed language, a prompt, greedy decode, a seeded temperature reproduces; progress lines in order
  memory     after each generate the footprint is back near the loaded one (the decode's buffers are returned)
  cancel     SIGUSR1 mid-decode replies "cancelled" within a second and the model stays loaded (the next generate
             runs without a reload); SIGUSR1 sent as the generate is written cancels that job; SIGUSR1 when idle is
             ignored (the next job runs); SIGKILL of the process group mid-decode leaves no
             process; a worker started without its daemon (LOCALROUTER_PARENT) exits at once, and one whose daemon
             goes away mid-decode ends within two seconds (macOS has no parent-death signal)
  vad off    with debug: whisper_debug.json is valid JSON and its model_transcript equals transcript.json
Exits 1 on any failure."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1 << 30


class Worker:
    def __init__(self, options, weights, parent=None, via_daemon=False):
        """via_daemon: a stand-in daemon process starts the worker (sharing these pipes) with LOCALROUTER_PARENT set
        to itself, so killing self.p (the stand-in) orphans the worker."""
        cfg = {"id": "whisper-mlx-check", "kind": "transcription", "weights": weights, "options": options}
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(cfg), PYTHONDONTWRITEBYTECODE="1")
        env.pop("LOCALROUTER_PARENT", None)
        if parent:
            env["LOCALROUTER_PARENT"] = str(parent)
        argv = [sys.executable, os.path.join(HERE, "worker.py")]
        if via_daemon:
            argv = [sys.executable, "-c", "import os, subprocess, sys, time; subprocess.Popen(sys.argv[1:], "
                    "env=dict(os.environ, LOCALROUTER_PARENT=str(os.getpid()))); time.sleep(3600)"] + argv
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=env, text=True, bufsize=1, start_new_session=True)
        self.progress = []

    def send(self, cmd):
        self.p.stdin.write((cmd if isinstance(cmd, str) else json.dumps(cmd)) + "\n")
        self.p.stdin.flush()

    def reply(self, on_progress=None):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited ({self.p.poll()})")
            msg = json.loads(line)
            if "progress" in msg:
                self.progress.append(msg["progress"])
                if on_progress:
                    on_progress(msg["progress"])
                continue
            return msg

    def call(self, cmd):
        self.send(cmd)
        return self.reply()

    def stats(self):
        return self.call({"op": "stats"})["stats"]

    def close(self):
        self.send({"op": "exit"})
        return self.p.wait(timeout=120)


def gone(pgid):
    try:
        os.killpg(pgid, 0)
        return False
    except ProcessLookupError:
        return True


def write_wav(path, seconds, rate=16000, channels=1):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * channels * int(seconds * rate))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--vad-model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report")
    a = ap.parse_args()
    fx = lambda name: os.path.join(a.fixtures, name)  # noqa: E731
    speech, german, long_wav = fx("en_librispeech_1272-128104-0000.wav"), fx("de_fleurs_01.wav"), fx("long_en_librispeech_270s.wav")
    root = tempfile.mkdtemp(prefix="whisper-mlx-check-")
    base = {"threads": a.threads}
    good = dict(base, vad="silero", vad_model=a.vad_model)
    checks, failed = [], 0
    n = [0]

    def check(name, ok, **detail):
        nonlocal failed
        checks.append({"check": name, "ok": bool(ok), **detail})
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, ensure_ascii=False)[:300]}", flush=True)

    def job(wav=None, name="audio.wav"):
        n[0] += 1
        d = os.path.join(root, f"job{n[0]}")
        os.makedirs(d)
        if wav:
            shutil.copyfile(wav, os.path.join(d, name))
        return d

    def generate(w, wav, debug=False, **req):
        d = job(wav)
        req.setdefault("audio", "audio.wav")
        r = w.call({"op": "generate", "id": f"tr_{n[0]}", "dir": d, "debug": debug, "request": {"transcription": req}})
        tr = json.load(open(os.path.join(d, "transcript.json"))) if r.get("ok") else None
        return r, tr, d

    # failure: every bad configuration refuses everything and holds nothing
    bad_vad = os.path.join(root, "silero_vad_changed.jit")
    data = bytearray(open(a.vad_model, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(bad_vad, "wb").write(data)
    empty_weights = os.path.join(root, "no-weights")
    os.makedirs(empty_weights)
    misfit = os.path.join(root, "misfit-config")
    os.makedirs(misfit)
    os.symlink(os.path.join(a.weights, "weights.safetensors"), os.path.join(misfit, "weights.safetensors"))
    cfg = json.load(open(os.path.join(a.weights, "config.json")))
    json.dump(dict(cfg, n_text_layer=cfg["n_text_layer"] + 1), open(os.path.join(misfit, "config.json"), "w"))
    cases = {
        "vad model missing": (dict(base, vad="silero", vad_model=os.path.join(root, "absent.jit")), a.weights),
        "vad model sha256 wrong": (dict(base, vad="silero", vad_model=bad_vad), a.weights),
        "vad mode unknown": (dict(base, vad="webrtc", vad_model=a.vad_model), a.weights),
        "dtype unknown": (dict(good, dtype="int3"), a.weights),
        "no weights, after the vad loaded": (good, empty_weights),
        "config does not fit the weights": (good, misfit),
    }
    for name, (opts, weights) in cases.items():
        w = Worker(opts, weights)
        before = w.stats()
        first, second = w.call({"op": "load"}), w.call({"op": "load"})
        gen, tr, d = generate(w, speech)
        after = w.stats()
        code = w.close()
        held = after["footprint"] - before["footprint"]
        check(f"failure: {name}: refused, nothing written, nothing held", first.get("ok") is False
              and second.get("ok") is False and gen.get("ok") is False and first.get("type") == "server_error"
              and not os.path.exists(os.path.join(d, "transcript.json")) and held < GIB // 4 and code == 0,
              load=first.get("error", "")[:160], footprint_before=before["footprint"],
              footprint_after=after["footprint"], exit=code)

    # lifecycle, refusals, shape, memory, soft cancel: one valid worker
    w = Worker(good, a.weights)
    idle = w.stats()
    t = time.monotonic()
    load = w.call({"op": "load"})
    load_s = time.monotonic() - t
    loaded = w.stats()
    again = w.call({"op": "load"})
    check("lifecycle: load reports its footprint and soft cancel; a second load is a no-op",
          load.get("ok") and load.get("cancel") == "signal" and abs(load["resident"] - loaded["footprint"]) < GIB // 4
          and again.get("ok") and abs(again["resident"] - load["resident"]) < GIB // 4,
          load_s=round(load_s, 2), resident=load.get("resident"), idle_footprint=idle["footprint"],
          mlx_active=loaded["mlx_active"])

    d = job(speech)
    tdir = job()
    write_wav(os.path.join(tdir, "stereo.wav"), 1, channels=2)
    write_wav(os.path.join(tdir, "8k.wav"), 1, rate=8000)
    refusals = {
        "not a transcription": {"op": "generate", "id": "x", "dir": d, "request": {"speech": {"input": "hi"}}},
        "no audio": {"transcription": {}},
        "audio outside the job": {"transcription": {"audio": "../job1/audio.wav"}},
        "audio missing": {"transcription": {"audio": "absent.wav"}},
        "stereo": ("stereo.wav", {}),
        "8 kHz": ("8k.wav", {}),
        "unknown language": {"transcription": {"audio": "audio.wav", "language": "klingon"}},
        "timestamps char": {"transcription": {"audio": "audio.wav", "timestamps": "char"}},
        "temperature 1.5": {"transcription": {"audio": "audio.wav", "temperature": 1.5}},
        "decode beam": {"transcription": {"audio": "audio.wav", "decode": "beam"}},
        "greedy with temperature": {"transcription": {"audio": "audio.wav", "decode": "greedy", "temperature": 0.4}},
    }
    for name in ("malformed_truncated.wav", "malformed_random.wav", "malformed_empty.wav", "malformed_notaudio.png"):
        shutil.copyfile(fx(name), os.path.join(tdir, name))
        refusals[name] = (name, {})
    for name, req in refusals.items():
        if isinstance(req, tuple):
            cmd = {"op": "generate", "id": "x", "dir": tdir, "request": {"transcription": {"audio": req[0], **req[1]}}}
        elif "op" in req:
            cmd = req
        else:
            cmd = {"op": "generate", "id": "x", "dir": d, "request": req}
        r = w.call(cmd)
        check(f"refusal: {name}", r.get("ok") is False and r.get("type") == "invalid_request_error",
              error=r.get("error", "")[:160])
    r1, r2 = w.call("{not json"), w.call({"op": "fly"})
    check("refusal: a bad line and an unknown op are protocol errors",
          r1.get("type") == "protocol_error" and r2.get("type") == "protocol_error")

    shapes = {}
    for ts in ("none", "segment", "word"):
        gen, tr, _ = generate(w, speech, timestamps=ts, seed=0)
        shapes[ts] = tr
        after = w.stats()
        ok = gen.get("ok") and isinstance(tr.get("text"), str) and tr["text"] and tr.get("language") == "en"
        ok = ok and abs(tr.get("duration", 0) - 5.855) < 0.01 and isinstance(tr.get("segments"), list) and tr["segments"]
        ok = ok and all(set(s) == {"id", "start", "end", "text"} and s["id"] == i and 0 <= s["start"] <= s["end"]
                        for i, s in enumerate(tr["segments"]))
        ok = ok and set(tr.get("vad", {})) == {"model", "speech", "dropped_segments"} and tr["vad"]["dropped_segments"] == []
        ok = ok and set(tr) == {"text", "language", "duration", "segments", "vad"} | ({"words"} if ts == "word" else set())
        if ts == "word":
            words = tr["words"]
            ok = ok and words and all(set(x) == {"word", "start", "end"} and x["start"] <= x["end"] <= tr["duration"] + 0.05
                                      for x in words)
            ok = ok and all(p["start"] <= q["start"] for p, q in zip(words, words[1:]))
            ok = ok and all(any(s["start"] - 0.05 <= x["start"] <= s["end"] + 0.05 for s in tr["segments"]) for x in words)
            ok = ok and len(words) == len(tr["text"].split())
        check(f"shape: timestamps {ts}", ok, text=tr and tr["text"][:100], words=tr and len(tr.get("words", [])),
              ms=gen.get("ms"), footprint=after["footprint"])
        check(f"memory: after timestamps {ts} the footprint is back near the loaded one",
              after["footprint"] - loaded["footprint"] < GIB // 2 and after["mlx_cache"] == 0,
              loaded=loaded["footprint"], after=after["footprint"], peak=after["footprint_peak"])
    check("shape: the text does not depend on the timestamps asked for",
          shapes["none"]["text"] == shapes["segment"]["text"], none=shapes["none"]["text"][:80])

    gen, tr, _ = generate(w, fx("silence_10s.wav"), timestamps="segment")
    check("shape: silence gated to no text", gen.get("ok") and tr["text"] == "" and tr["segments"] == []
          and tr["vad"]["speech"] == [], vad=tr and tr["vad"])
    gen, tr, _ = generate(w, german, language="de", timestamps="segment")
    check("shape: a fixed language", gen.get("ok") and tr["language"] == "de" and tr["text"], text=tr and tr["text"][:100])
    gen, tr, _ = generate(w, german, timestamps="segment")
    check("shape: a detected language", gen.get("ok") and tr["language"] == "de", language=tr and tr["language"])
    gen, tr, _ = generate(w, speech, prompt="Mister Quilter, the apostle of the middle classes.")
    check("shape: a prompt is accepted and not echoed", gen.get("ok") and tr["text"] and "apostle" in tr["text"].lower()
          and tr["text"].count("apostle") == 1, text=tr and tr["text"][:100])
    gen, tr, _ = generate(w, speech, decode="greedy", language="en")
    check("shape: greedy decode", gen.get("ok") and tr["text"] == shapes["none"]["text"], text=tr and tr["text"][:80])
    texts = [generate(w, speech, temperature=0.8, seed=7)[1]["text"] for _ in range(2)]
    check("shape: a seeded temperature reproduces", texts[0] == texts[1], text=texts[0][:80])
    w.progress = []
    gen, tr, _ = generate(w, fx("long_en_librispeech_60s.wav"), timestamps="word")
    steps = [(p["step"], p["of"]) for p in w.progress]
    check("shape: long-form, progress in order to the end", gen.get("ok") and len(tr["segments"]) > 3 and steps
          and steps[-1][0] == steps[-1][1] == 3 and all(p <= q for p, q in zip(steps, steps[1:])),
          progress=steps, segments=tr and len(tr["segments"]), ms=gen.get("ms"))

    # soft cancel: SIGUSR1 mid-decode, then again while idle
    d = job(long_wav)
    w.send({"op": "generate", "id": "tr_cancel", "dir": d, "request": {"transcription": {"audio": "audio.wav"}}})
    first = threading.Event()
    sent = {}

    def cancel_on_progress(p):
        if p["step"] >= 1 and not first.is_set():
            first.set()
            sent["t"] = time.monotonic()
            os.kill(w.p.pid, signal.SIGUSR1)
    r = w.reply(cancel_on_progress)
    took = time.monotonic() - sent.get("t", time.monotonic())
    st = w.stats()
    check("cancel: SIGUSR1 mid-decode replies cancelled within a second, nothing written",
          r.get("type") == "cancelled" and took < 1.0 and not os.path.exists(os.path.join(d, "transcript.json")),
          reply=r, seconds=round(took, 3), footprint=st["footprint"])
    # the daemon's cancel can race the start of the job: SIGUSR1 right after the generate command is written, before
    # the worker has read it or begun to decode, must still cancel that job (not be lost, nor cancel a later one)
    d = job(long_wav)
    w.send({"op": "generate", "id": "tr_early", "dir": d, "request": {"transcription": {"audio": "audio.wav"}}})
    t = time.monotonic()
    os.kill(w.p.pid, signal.SIGUSR1)
    r = w.reply()
    took = time.monotonic() - t
    check("cancel: SIGUSR1 as the generate arrives cancels that job, before its decode",
          r.get("type") == "cancelled" and took < 1.0 and not os.path.exists(os.path.join(d, "transcript.json")),
          reply=r, seconds=round(took, 3))
    os.kill(w.p.pid, signal.SIGUSR1)
    time.sleep(0.2)
    t = time.monotonic()
    gen, tr, _ = generate(w, speech)
    check("cancel: the model stayed loaded (no reload) and SIGUSR1 while idle is ignored",
          gen.get("ok") and tr["text"] == shapes["none"]["text"] and w.p.poll() is None
          and time.monotonic() - t < load_s, ms=gen.get("ms"))
    code = w.close()
    check("lifecycle: exit releases and ends 0", code == 0, exit=code)

    # hard cancel: SIGKILL the group mid-decode
    w = Worker(good, a.weights)
    w.call({"op": "load"})
    pgid = os.getpgid(w.p.pid)
    d = job(long_wav)
    w.send({"op": "generate", "id": "tr_kill", "dir": d, "request": {"transcription": {"audio": "audio.wav"}}})
    killed = {}

    def kill_on_progress(p):
        if p["step"] >= 1 and not killed:
            killed["t"] = time.monotonic()
            os.killpg(pgid, signal.SIGKILL)
    try:
        w.reply(kill_on_progress)
    except RuntimeError:
        pass
    w.p.wait(timeout=10)
    time.sleep(0.2)
    check("cancel: SIGKILL of the group mid-decode leaves no process", bool(killed) and gone(pgid), pgid=pgid)

    # a daemon that goes away: the worker's LOCALROUTER_PARENT is not its parent at start, or dies mid-decode
    w = Worker(good, a.weights, parent=1 << 22)
    code = w.p.wait(timeout=60)
    check("lifecycle: a worker started without its daemon exits 0 at once", code == 0, exit=code)
    w = Worker(good, a.weights, via_daemon=True)
    w.call({"op": "load"})
    pgid = os.getpgid(w.p.pid)
    d = job(long_wav)
    w.send({"op": "generate", "id": "tr_orphan", "dir": d, "request": {"transcription": {"audio": "audio.wav"}}})
    died = {}

    def daemon_dies(p):
        if p["step"] >= 1 and not died:
            died["t"] = time.monotonic()
            w.p.kill()  # the stand-in daemon only; the worker is left running in the group
    try:
        w.reply(daemon_dies)
    except RuntimeError:
        pass
    while not gone(pgid) and time.monotonic() - died.get("t", 0) < 5:
        time.sleep(0.05)
    took = time.monotonic() - died.get("t", time.monotonic())
    check("lifecycle: a daemon that goes away mid-decode ends the worker within two seconds",
          bool(died) and gone(pgid) and took < 2.0, seconds=round(took, 2))
    if not gone(pgid):
        os.killpg(pgid, signal.SIGKILL)

    # vad off with debug: the model's own transcript is the transcript
    w = Worker(dict(base, vad="off"), a.weights)
    w.call({"op": "load"})
    ok_debug, detail = True, {}
    for name, wav in (("off-silence", fx("silence_10s.wav")), ("off-speech", speech)):
        gen, tr, d = generate(w, wav, debug=True, timestamps="word")
        try:
            dbg = json.load(open(os.path.join(d, "whisper_debug.json")))
            same = dbg["model_transcript"] == tr and "vad" not in tr
        except (OSError, ValueError, KeyError, TypeError) as e:
            same, tr = False, {"error": repr(e)}
        ok_debug &= bool(gen.get("ok") and same)
        detail[name] = (tr or {}).get("text", (tr or {}).get("error"))
    check("vad off with debug: valid JSON, model transcript unchanged", ok_debug, **detail)
    w.close()

    shutil.rmtree(root, ignore_errors=True)
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        json.dump({"failures": failed, "checks": checks}, open(a.report, "w"), indent=1, ensure_ascii=False)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
