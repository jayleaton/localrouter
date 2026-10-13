"""python tools/speech/whisper/loadcheck.py --weights DIR --vad-model silero_vad.jit --audio speech.wav --silence silence.wav
    [--device cuda] [--dtype float16] [--report out.json]

The worker's load is all or nothing and its voice activity gate cannot be bypassed. Each bad configuration runs in its
own worker process over the worker protocol: load, load again, then generate on a speech file must all be refused
(ok false), no transcript.json may appear, and the process must hold no model afterwards (RSS back near its start,
no CUDA memory allocated). The bad configurations: no VAD model file, a VAD model with one byte changed (wrong
sha256), an unknown vad mode, and a model load that fails part way (a CUDA device that does not exist), after the VAD
loaded. Then a valid worker: load, generate on silence (empty text, the gate's record present) and on speech. Then the
gate off with debug: whisper_debug.json is valid JSON and its model_transcript equals transcript.json (nothing gated).
Exits 1 on any failure."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1 << 30


class Worker:
    def __init__(self, options, weights):
        cfg = {"id": "whisper-loadcheck", "kind": "transcription", "weights": weights, "options": options}
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(cfg))
        env.pop("LOCALROUTER_PARENT", None)
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=env, text=True, bufsize=1)

    def call(self, cmd):
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker exited")
            msg = json.loads(line)
            if "progress" not in msg:
                return msg

    def stats(self):
        return self.call({"op": "stats"})["stats"]

    def close(self):
        self.p.stdin.write(json.dumps({"op": "exit"}) + "\n")
        self.p.stdin.flush()
        return self.p.wait(timeout=120)


def job(root, name, wav):
    d = os.path.join(root, name)
    os.makedirs(d)
    shutil.copyfile(wav, os.path.join(d, "audio.wav"))
    return d


def generate(w, d, debug=False):
    return w.call({"op": "generate", "id": f"tr_{os.path.basename(d)}", "dir": d, "debug": debug,
                   "request": {"transcription": {"audio": "audio.wav", "timestamps": "segment", "seed": 0}}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--vad-model", required=True)
    ap.add_argument("--audio", required=True, help="a speech WAV (16 kHz mono 16-bit)")
    ap.add_argument("--silence", required=True, help="a WAV with no speech")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    root = tempfile.mkdtemp(prefix="whisper-loadcheck-")
    base = {"device": a.device, "dtype": a.dtype, "attn": "sdpa", "threads": a.threads}
    bad_vad = os.path.join(root, "silero_vad_changed.jit")
    data = bytearray(open(a.vad_model, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(bad_vad, "wb").write(data)
    cases = {
        "vad model missing": dict(base, vad="silero", vad_model=os.path.join(root, "absent.jit")),
        "vad model sha256 wrong": dict(base, vad="silero", vad_model=bad_vad),
        "vad mode unknown": dict(base, vad="webrtc", vad_model=a.vad_model),
        "model load fails after the vad": dict(base, device="cuda:63", vad="silero", vad_model=a.vad_model),
    }
    checks, failed = [], 0

    def check(name, ok, **detail):
        nonlocal failed
        checks.append({"check": name, "ok": bool(ok), **detail})
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail)[:300]}", flush=True)

    for name, opts in cases.items():
        w = Worker(opts, a.weights)
        before = w.stats()
        first, second = w.call({"op": "load"}), w.call({"op": "load"})
        d = job(root, name.replace(" ", "-"), a.audio)
        gen = generate(w, d)
        after = w.stats()
        code = w.close()
        held = after["rss"] - before["rss"]
        check(f"{name}: load, load again and generate refused, nothing written, nothing held",
              first.get("ok") is False and second.get("ok") is False and gen.get("ok") is False
              and not os.path.exists(os.path.join(d, "transcript.json")) and held < GIB
              and not after.get("cuda_allocated") and code == 0,
              load=first.get("error", "")[:160], again=second.get("error", "")[:80], generate=gen.get("error", "")[:80],
              rss_before=before["rss"], rss_after=after["rss"], cuda_allocated=after.get("cuda_allocated"), exit=code)

    w = Worker(dict(base, vad="silero", vad_model=a.vad_model), a.weights)
    load = w.call({"op": "load"})
    d = job(root, "valid-silence", a.silence)
    gen = generate(w, d)
    tr = json.load(open(os.path.join(d, "transcript.json"))) if gen.get("ok") else {}
    check("valid load: silence gated to empty text", load.get("ok") and gen.get("ok") and tr.get("text") == ""
          and not tr.get("segments") and isinstance(tr.get("vad"), dict), text=tr.get("text"), vad=tr.get("vad"))
    d = job(root, "valid-speech", a.audio)
    gen = generate(w, d)
    tr = json.load(open(os.path.join(d, "transcript.json"))) if gen.get("ok") else {}
    check("valid load: speech kept", gen.get("ok") and tr.get("text") and tr.get("vad", {}).get("dropped_segments") == [],
          text=(tr.get("text") or "")[:120], vad=tr.get("vad"))
    w.close()

    w = Worker(dict(base, vad="off"), a.weights)
    load = w.call({"op": "load"})
    ok_debug, detail = True, {}
    for name, wav in (("off-silence", a.silence), ("off-speech", a.audio)):
        d = job(root, name, wav)
        gen = generate(w, d, debug=True)
        try:
            tr = json.load(open(os.path.join(d, "transcript.json")))
            dbg = json.load(open(os.path.join(d, "whisper_debug.json")))
            same = dbg["model_transcript"] == tr and "vad" not in tr and "_debug" not in dbg["model_transcript"]
        except (OSError, ValueError, KeyError) as e:
            same, tr = False, {"error": repr(e)}
        ok_debug &= bool(load.get("ok") and gen.get("ok") and same)
        detail[name] = tr.get("text", tr.get("error"))
    check("vad off with debug: valid JSON, model transcript unchanged", ok_debug, **detail)
    w.close()

    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump({"failures": failed, "checks": checks}, open(a.report, "w"), indent=1, ensure_ascii=False)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
