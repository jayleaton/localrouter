"""python tools/speech/whisper/bench.py WAV... [--warm 2] [--cancel LONG.wav] [--report out.json]

Times the worker over its protocol, as the daemon runs it (its own process group):
  cold      process start -> load reply
  per file  the first generate after load and --warm more; RTF = processing time / audio duration
  memory    VmRSS / VmHWM (peak RSS) and, on CUDA, allocated / reserved and their maxima, after load and each generate
  cancel    a fresh worker, generate on a long file, SIGKILL its process group mid-run (after the first progress line
            or --cancel-after seconds): the group must be gone; MemAvailable before load, loaded and after the kill.
Prints and writes a JSON report."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_model import DEFAULT_ROOT, REVISION  # noqa: E402

OUT = os.path.expanduser("~/.cache/localrouter-speech/out")


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


class Worker:
    def __init__(self, cfg, python):
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(cfg), PYTHONDONTWRITEBYTECODE="1")
        env["LOCALROUTER_PARENT"] = str(os.getpid())
        self.t0 = time.monotonic()
        self.p = subprocess.Popen([python, os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=env, text=True, bufsize=1, start_new_session=True)
        self.pgid = os.getpgid(self.p.pid)

    def send(self, cmd):
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()

    def reply(self, on_progress=None):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited ({self.p.poll()})")
            msg = json.loads(line)
            if "progress" in msg:
                if on_progress:
                    on_progress(msg["progress"])
                continue
            return msg

    def call(self, cmd):
        self.send(cmd)
        return self.reply()

    def stats(self):
        return self.call({"op": "stats"}).get("stats")

    def exit(self):
        self.send({"op": "exit"})
        return self.p.wait(timeout=120)


def job(run_dir, wav, i):
    d = os.path.join(run_dir, f"job{i}")
    os.makedirs(d)
    shutil.copyfile(wav, os.path.join(d, "audio.wav"))
    return d


def duration(wav):
    with wave.open(wav, "rb") as w:
        return w.getnframes() / w.getframerate()


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def cgroup_cpu_max():
    """This process's cgroup v2 cpu.max ("quota period", e.g. "800000 100000" = 8 CPUs), or None."""
    try:
        rel = open("/proc/self/cgroup").read().strip().split("::", 1)[1]
        return open(f"/sys/fs/cgroup{rel}/cpu.max").read().strip()
    except (OSError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--weights", default=os.path.join(DEFAULT_ROOT, "whisper-large-v3-turbo", REVISION))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--threads", type=int, default=0, help="0: torch's default")
    ap.add_argument("--vad-model", required=True, help="the pinned silero_vad.jit (the worker's voice activity gate)")
    ap.add_argument("--timestamps", default="segment")
    ap.add_argument("--language", default="")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--cancel", help="long WAV for the cancellation check")
    ap.add_argument("--cancel-after", type=float, default=0, help="seconds after generate; 0: at the first progress")
    ap.add_argument("--report", default=os.path.join(OUT, "bench-report.json"))
    args = ap.parse_args()
    opts = {"device": args.device, "dtype": args.dtype, "attn": args.attn}
    if args.threads:
        opts["threads"] = args.threads
    opts.update(vad="silero", vad_model=args.vad_model)
    cfg = {"id": "whisper-large-v3-turbo", "kind": "transcription", "weights": args.weights, "options": opts}
    run_dir = os.path.join(OUT, "bench", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir)
    report = {"model": "openai/whisper-large-v3-turbo", "revision": REVISION, "config": opts,
              "host": {"online_cpus": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0)),
                       "cgroup_cpu_max": cgroup_cpu_max(), "torch_threads_requested": args.threads or None,
                       "loadavg_start": os.getloadavg()}, "files": []}

    n = 0
    if args.wavs:
        mem0 = mem_available()
        w = Worker(cfg, args.python)
        load = w.call({"op": "load"})
        if not load.get("ok"):
            raise SystemExit(f"load failed: {load}")
        report["cold_load_s"] = round(time.monotonic() - w.t0, 3)
        report["resident"] = load["resident"]
        report["after_load"] = w.stats()
        report["mem_available"] = {"before_start": mem0, "after_load": mem_available()}
        for wav in args.wavs:
            dur = duration(wav)
            runs = []
            for k in range(1 + args.warm):
                d = job(run_dir, wav, n)
                n += 1
                t = time.monotonic()
                r = w.call({"op": "generate", "id": f"tr_bench{n}", "dir": d, "request": {"transcription": {
                    "audio": "audio.wav", "language": args.language, "timestamps": args.timestamps}}})
                wall = time.monotonic() - t
                if not r.get("ok"):
                    raise SystemExit(f"generate failed on {wav}: {r}")
                runs.append({"kind": "first" if k == 0 else "warm", "wall_s": round(wall, 3), "ms": r["ms"],
                             "rtf": round(r["ms"] / 1000 / dur, 4), "stats": w.stats(),
                             "text": json.load(open(os.path.join(d, "transcript.json")))["text"][:120]})
            warm = [x["ms"] for x in runs[1:]] or [runs[0]["ms"]]
            report["files"].append({"file": os.path.basename(wav), "duration_s": dur, "runs": runs,
                                    "warm_ms_min": min(warm), "warm_rtf_min": round(min(warm) / 1000 / dur, 4)})
        report["exit_code"] = w.exit()
        report["peak_hwm"] = max([report["after_load"]["hwm"]] +
                                 [x["stats"]["hwm"] for f in report["files"] for x in f["runs"]])

    if args.cancel:
        mem0 = mem_available()
        w = Worker(cfg, args.python)
        load = w.call({"op": "load"})
        loaded = mem_available()
        started = threading.Event()
        d = job(run_dir, args.cancel, n)
        w.send({"op": "generate", "id": "tr_cancel", "dir": d, "request": {"transcription": {
            "audio": "audio.wav", "timestamps": args.timestamps, "language": args.language}}})
        t = time.monotonic()
        if args.cancel_after:
            time.sleep(args.cancel_after)
        else:
            def wait_progress():
                try:
                    w.reply(on_progress=lambda p: started.set())
                except RuntimeError:
                    pass
            threading.Thread(target=wait_progress, daemon=True).start()
            started.wait(timeout=600)
            time.sleep(2)  # into the decode, past the first progress line
        during = mem_available()
        stats_rss = int(open(f"/proc/{w.p.pid}/status").read().split("VmRSS:")[1].split()[0]) * 1024
        os.killpg(w.pgid, signal.SIGKILL)
        rc = w.p.wait(timeout=30)
        killed_after = time.monotonic() - t
        time.sleep(2)
        report["cancel"] = {
            "file": os.path.basename(args.cancel), "duration_s": duration(args.cancel), "load_ok": load.get("ok"),
            "killed_after_s": round(killed_after, 2), "returncode": rc,
            "pid_gone": not os.path.exists(f"/proc/{w.p.pid}"), "group_gone": not group_alive(w.pgid),
            "worker_rss_at_kill": stats_rss,
            "mem_available": {"before_start": mem0, "loaded": loaded, "running": during, "after_kill": mem_available()},
            "transcript_written": os.path.exists(os.path.join(d, "transcript.json")),
        }
        c = report["cancel"]["mem_available"]
        report["cancel"]["reclaimed"] = c["after_kill"] - c["running"]
        report["cancel"]["after_kill_minus_before_start"] = c["after_kill"] - c["before_start"]
    report["host"]["loadavg_end"] = os.getloadavg()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
