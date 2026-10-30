"""python tools/speech/whisper/macos/bench.py --weights DIR --vad-model silero_vad.jit --fixtures DIR
    [--only a,b] [--warm 2] [--compare other.json] [--report out.json]

The macOS worker over its protocol on the speech fixtures (tools/speech/fixtures), as the daemon runs it. Real weights:
run it under the machine's model lock.
  cold      process start -> load reply (load + warm-up decode), and the load reply's resident bytes
  per file  the HTTP API's request (model card recipe, detected language, segment timestamps): text, language,
            WER (CER for ja/zh) against the human reference, ms and RTF (processing time / audio duration); then the
            same file with word timestamps (words in order, inside the audio, one per text word) and the voice loop's
            request (greedy, the reference language): its ms, and whether its text equals the recipe's
  warm      the first English file --warm more times: ms and RTF
  memory    the worker's physical footprint (proc_pid_rusage, what `footprint` reports) after load, after the run and
            its lifetime peak; MLX's own active/peak bytes
  compare   --compare FILE ({"fixture": "text"} or a report of this script or ../twin.py-style rows with "fixture" and
            "text"): WER of this run's text against that run's (agreement with another engine's transcript)
Prints and writes a JSON report (default ~/.cache/localrouter-speech/out/)."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qualify import scores  # noqa: E402  (stdlib only: the daemon qualification's own WER/CER)

OUT = os.path.expanduser("~/.cache/localrouter-speech/out")


class Worker:
    def __init__(self, cfg):
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(cfg), PYTHONDONTWRITEBYTECODE="1")
        env["LOCALROUTER_PARENT"] = str(os.getpid())
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=env, text=True, bufsize=1, start_new_session=True)

    def call(self, cmd):
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited ({self.p.poll()})")
            msg = json.loads(line)
            if "progress" not in msg:
                return msg

    def stats(self):
        return self.call({"op": "stats"})["stats"]

    def exit(self):
        self.p.stdin.write(json.dumps({"op": "exit"}) + "\n")
        self.p.stdin.flush()
        return self.p.wait(timeout=120)


def load_compare(path):
    if not path:
        return {}
    obj = json.load(open(path))
    if isinstance(obj, dict) and "fixtures" in obj:
        obj = obj["fixtures"]
    if isinstance(obj, list):
        return {r["fixture"]: r["text"] for r in obj}
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--vad-model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--only", help="comma list of fixture names (default: every WAV fixture)")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--compare")
    ap.add_argument("--report", default=os.path.join(OUT, "whisper-mlx-bench.json"))
    a = ap.parse_args()
    refs = json.load(open(os.path.join(a.fixtures, "refs.json")))
    only = set(a.only.split(",")) if a.only else None
    names = [n for n, r in refs.items() if not r.get("malformed") and (not only or n in only)
             and os.path.exists(os.path.join(a.fixtures, n + ".wav"))]
    other = load_compare(a.compare)
    root = tempfile.mkdtemp(prefix="whisper-mlx-bench-")
    n = [0]

    def run(w, name, **req):
        n[0] += 1
        d = os.path.join(root, f"job{n[0]}")
        os.makedirs(d)
        shutil.copyfile(os.path.join(a.fixtures, name + ".wav"), os.path.join(d, "audio.wav"))
        t = time.monotonic()
        r = w.call({"op": "generate", "id": f"tr_{n[0]}", "dir": d,
                    "request": {"transcription": {"audio": "audio.wav", "seed": 0, **req}}})
        wall = time.monotonic() - t
        if not r.get("ok"):
            raise RuntimeError(f"{name}: {r}")
        tr = json.load(open(os.path.join(d, "transcript.json")))
        shutil.rmtree(d)
        return tr, wall

    cfg = {"id": "whisper-mlx-bench", "kind": "transcription", "weights": a.weights,
           "options": {"vad": "silero", "vad_model": a.vad_model, "threads": a.threads}}
    t = time.monotonic()
    w = Worker(cfg)
    load = w.call({"op": "load"})
    cold = time.monotonic() - t
    if not load.get("ok"):
        raise SystemExit(f"load failed: {load}")
    loaded = w.stats()
    report = {"weights": a.weights, "cold_s": round(cold, 2), "resident": load["resident"], "loaded": loaded,
              "fixtures": []}
    print(f"cold {cold:.2f}s resident {load['resident'] / 2**30:.2f} GiB", flush=True)

    for name in names:
        ref = refs[name]
        lang = ref.get("language") or ""
        tr, wall = run(w, name, timestamps="segment")
        dur = tr["duration"]
        row = {"fixture": name, "language_ref": lang, "language": tr["language"], "duration_s": dur,
               "wall_s": round(wall, 3), "rtf": round(wall / dur, 4), "text": tr["text"], "segments": len(tr["segments"]),
               "vad_dropped": len(tr["vad"]["dropped_segments"])}
        if ref.get("text"):
            row["wer"], row["cer"] = scores(tr["text"], ref["text"], lang)
        if name in other:
            row["agreement_wer"], row["agreement_cer"] = scores(tr["text"], other[name], lang)
            row["other_text"] = other[name]
        tw, wall_w = run(w, name, timestamps="word")
        words = tw.get("words", [])
        row["words"] = len(words)
        row["words_wall_s"] = round(wall_w, 3)
        row["words_ok"] = (all(x["start"] <= x["end"] <= dur + 0.05 for x in words)
                           and all(p["start"] <= q["start"] for p, q in zip(words, words[1:])))
        row["words_text_same"] = tw["text"] == tr["text"]
        if lang:
            tg, wall_g = run(w, name, decode="greedy", language=lang)
            row["greedy_wall_s"] = round(wall_g, 3)
            row["greedy_same"] = tg["text"] == tr["text"]
        report["fixtures"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("text", "other_text")}, ensure_ascii=False), flush=True)

    en = next((x for x in names if x.startswith("en_librispeech")), None)
    if en and a.warm:
        runs = [run(w, en, timestamps="segment")[1] for _ in range(a.warm)]
        runs_g = [run(w, en, decode="greedy", language="en")[1] for _ in range(a.warm)]
        dur = refs[en]["duration_s"]
        report["warm"] = {"fixture": en, "wall_s": [round(x, 3) for x in runs], "rtf": [round(x / dur, 4) for x in runs],
                          "greedy_wall_s": [round(x, 3) for x in runs_g]}
    report["after"] = w.stats()
    report["exit"] = w.exit()
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(report, open(a.report, "w"), indent=1, ensure_ascii=False)
    gib = 2 ** 30
    print(f"loaded footprint {loaded['footprint'] / gib:.2f} GiB, after {report['after']['footprint'] / gib:.2f} GiB, "
          f"peak {report['after']['footprint_peak'] / gib:.2f} GiB, mlx peak {report['after']['mlx_peak'] / gib:.2f} GiB")
    print(f"report: {a.report}")


if __name__ == "__main__":
    main()
