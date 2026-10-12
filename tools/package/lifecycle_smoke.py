#!/usr/bin/env python3
"""The on-demand lifecycle of one LocalRouter video model, checked on the running package, with a JSON receipt.

    python3 -I tools/package/lifecycle_smoke.py --profile tools/package/profiles/<model>.json --out receipt.json

It drives the daemon through tools/package/localrouterctl (the same LR_* environment selects binary or docker mode),
so what it checks is what an operator runs:

  start      daemon up; the model unloaded; no worker process; the memory baseline
  refusals   each unsupported request is a 400 and nothing loads
  cold       one generation from unloaded: status, phases, timings; the MP4's video and audio streams (ffprobe)
  release    with idle_ttl_s 0 the model unloads after the job: the worker and its children are reaped, none of them
             holds GPU memory (nvidia-smi, when present), MemAvailable comes back within the tolerance
  cycles     N more load/generate/unload cycles: MemAvailable must not drift
  cancel     a job cancelled in each named phase ends `cancelled`, without output, and its worker tree is reaped
  crash      the worker killed mid-job fails that job only; the next job succeeds
  recover    a generation after the cancels and the crash
  unload     (--explicit-unload) POST /v1/tools/{id}/load, then /unload: the operator's unload releases it too
  robust     (--probe-bodiless-post) a POST without a body or Content-Length leaves the daemon serving
  after      another model (optional) loads and generates on the freed device
  stop       the daemon and every process it started exit; the port closes

Every check is pass, fail or skipped (with the reason). The exit status is 1 if any check failed. GPU checks are skipped,
not passed, on a machine without nvidia-smi; `gpu_verified` in the receipt says which. A profile that lists
`provisional` fields (a model contract its engine's owner has not confirmed yet) gives the result `provisional`, not
`pass`, and exit status 2.
"""

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


class Smoke:
    def __init__(self, args):
        self.args = args
        self.profile = json.load(open(args.profile))
        self.model = self.profile.get("model") or (self.profile.get("tool") or {}).get("id")
        if not self.model:
            sys.exit("the profile names no model: set \"model\" (or use the profile install.sh resolved, $LR_HOME/profile.json)")
        self.ctl = args.ctl
        self.url = self.run_ctl("url").strip()
        self.checks = []
        self.receipt = {
            "profile": os.path.basename(args.profile),
            "model": self.model,
            "machine": {"arch": platform.machine(), "kernel": platform.release()},
            "gpu": gpu_info(),
            "checks": self.checks,
            "timings_s": {},
            "memory": [],
        }
        self.receipt["gpu_verified"] = self.receipt["gpu"] is not None
        # beside the receipt, not in /tmp: that is often a small RAM disk, and on a Spark RAM is the GPU's memory
        self.tmp = tempfile.mkdtemp(prefix="lr-smoke-", dir=os.path.dirname(os.path.abspath(args.out)))

    # ---------------------------------------------------------------- plumbing

    def run_ctl(self, *argv, check=True):
        r = subprocess.run([self.ctl, *argv], capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"localrouterctl {' '.join(argv)}: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout

    def http(self, method, path, body=None, timeout=60):
        # a POST always carries a Content-Length (an empty body for the control routes)
        data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if data is not None:
            req.add_header("content-type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return r.status, raw if r.headers.get("content-type", "").startswith("video/") else json.loads(raw or b"null")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, raw.decode(errors="replace")

    def check(self, name, ok, **detail):
        state = "skipped" if ok is None else ("pass" if ok else "fail")
        self.checks.append({"check": name, "result": state, **detail})
        print(f"[{state:7}] {name}" + (f"  {json.dumps(detail)}" if detail else ""), flush=True)
        return ok

    def tool(self):
        _, d = self.http("GET", "/v1/tools")
        for t in d["data"]:
            if t["id"] == self.model:
                return t, d["mem_available_bytes"]
        raise RuntimeError(f"model {self.model} is not configured: {[t['id'] for t in d['data']]}")

    def tree(self):
        return [int(p) for p in self.run_ctl("pids").split()]

    def workers(self):
        return self.tree()[1:]

    def sample(self, label):
        t, avail = self.tool()
        s = {"at": label, "state": t["state"], "resident_bytes": t["resident_bytes"], "mem_available_bytes": avail,
             "host_mem_available_bytes": meminfo_available(), "processes": len(self.tree())}
        self.receipt["memory"].append(s)
        return s

    def wait(self, what, pred, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            time.sleep(0.1)
        return pred()

    def submit(self, body=None):
        req = {"model": self.model, **self.profile["request"], **(body or {})}
        return self.http("POST", "/v1/videos", {k: v for k, v in req.items() if v is not None})  # null: leave it out

    def follow(self, vid, on_poll=None, timeout=None):
        """Polls a job to its end; returns (final object, {phase: first seen s}, worker pids seen)."""
        t0 = time.monotonic()
        phases, seen = {}, set()
        timeout = timeout or self.profile.get("job_timeout_s", 3600)
        while True:
            _, v = self.http("GET", f"/v1/videos/{vid}")
            phase = v.get("phase") or ""
            if phase and phase not in phases:
                phases[phase] = round(time.monotonic() - t0, 2)
            seen.update(self.workers())
            if v["status"] not in ("queued", "in_progress"):
                return v, phases, sorted(seen)
            if on_poll and on_poll(v):
                on_poll = None
            if time.monotonic() - t0 > timeout:
                self.http("DELETE", f"/v1/videos/{vid}")
                raise RuntimeError(f"{vid} did not finish in {timeout} s")
            time.sleep(0.05)

    def released(self, label, pids, baseline):
        """The model is unloaded, its worker tree is gone, no pid of it holds GPU memory, memory is back."""
        ttl = self.tool()[0]["idle_ttl_s"]
        timeout = self.profile.get("unload_timeout_s", 60) + ttl
        t0 = time.monotonic()
        ok = self.wait("unload", lambda: self.tool()[0]["state"] == "unloaded" and not self.workers(), timeout)
        took = round(time.monotonic() - t0, 2)
        alive = [{"pid": p, "state": proc_state(p)} for p in pids if os.path.exists(f"/proc/{p}")]
        self.check(f"{label}: unloaded and worker tree reaped", ok and not alive, seconds=took, worker_pids=pids, alive=alive)
        apps = gpu_apps()
        if apps is None:
            self.check(f"{label}: no worker pid holds GPU memory", None, reason="no nvidia-smi on this machine")
        else:
            held = [a for a in apps if a["pid"] in pids]
            self.check(f"{label}: no worker pid holds GPU memory", not held, held=held)
        s = self.sample(f"{label} released")
        drift = baseline["mem_available_bytes"] - s["mem_available_bytes"]
        tol = int(self.args.mem_tolerance_gib * 2**30)
        self.check(f"{label}: MemAvailable back to baseline", drift <= tol, drift_gib=round(drift / 2**30, 2),
                   tolerance_gib=self.args.mem_tolerance_gib)
        return took

    def generate(self, label, baseline):
        t0 = time.monotonic()
        st, v = self.submit({"seed": self.profile["request"].get("seed", 1)})
        if st != 200:
            self.check(f"{label}: accepted", False, status=st, reply=v)
            return None
        v, phases, pids = self.follow(v["id"])
        wall = round(time.monotonic() - t0, 2)
        if not self.check(f"{label}: completed", v["status"] == "completed", id=v["id"], status=v["status"],
                          error=v.get("error"), wall_s=wall, phases=phases):
            return None
        st, mp4 = self.http("GET", f"/v1/videos/{v['id']}/content", timeout=600)
        path = os.path.join(self.tmp, f"{label}.mp4")
        with open(path, "wb") as f:
            f.write(mp4 if isinstance(mp4, bytes) else b"")
        self.check_media(label, path)
        self.receipt["timings_s"][label] = {"wall": wall, "phases_first_seen": phases}
        self.released(label, pids, baseline)
        if self.args.keep_outputs:
            shutil.copy(path, os.path.join(self.args.keep_outputs, f"{label}.mp4"))
        return v

    def check_media(self, label, path):
        exp = self.profile["expect"]
        info = ffprobe(path)
        if info is None:
            self.check(f"{label}: MP4 has the expected streams", False, reason="ffprobe could not read the output")
            return
        video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
        got = {}
        if video:
            num, den = (int(x) for x in video["r_frame_rate"].split("/"))
            got.update(width=video["width"], height=video["height"], frames=int(video.get("nb_read_frames", 0)),
                       fps=round(num / den, 3))
        want = {k: exp[k] for k in ("width", "height", "frames", "fps")}
        self.check(f"{label}: video stream", video is not None and all(got.get(k) == v for k, v in want.items()),
                   want=want, got=got)
        a = exp.get("audio")
        if a is None:
            self.check(f"{label}: no audio stream", audio is None)
            return
        agot = {}
        if audio:
            agot = {"sample_rate": int(audio["sample_rate"]), "channels": audio["channels"],
                    "duration_s": round(float(audio.get("duration", 0)), 3)}
        vdur = exp["frames"] / exp["fps"]
        ok = (audio is not None and agot["sample_rate"] == a["sample_rate"] and agot["channels"] == a["channels"]
              and abs(agot["duration_s"] - vdur) <= a.get("duration_tolerance_s", 0.1))
        self.check(f"{label}: audio stream coupled to the video", ok, want={**a, "video_duration_s": round(vdur, 3)}, got=agot)

    # ---------------------------------------------------------------- the run

    def run(self):
        a = self.args
        if not a.no_start:
            t0 = time.monotonic()
            self.run_ctl("start")
            self.receipt["timings_s"]["daemon_start"] = round(time.monotonic() - t0, 2)
        t, _ = self.tool()
        self.receipt["tool"] = {k: t[k] for k in ("id", "kind", "idle_ttl_s", "keep_loaded", "priority", "capabilities")}
        ttl = self.profile.get("idle_ttl_s", 0)
        self.check("start: model configured on demand", t["state"] == "unloaded" and not t["keep_loaded"] and t["idle_ttl_s"] == ttl,
                   state=t["state"], keep_loaded=t["keep_loaded"], idle_ttl_s=t["idle_ttl_s"], want_idle_ttl_s=ttl)
        self.check("start: daemon holds no worker", not self.workers(), workers=self.workers())
        baseline = self.sample("baseline")

        for r in self.profile.get("refusals", []):
            st, v = self.submit(r["body"])
            msg = v.get("error", {}).get("message") if isinstance(v, dict) else v
            if st == 200:  # accepted: cancel it, it must not run
                self.http("DELETE", f"/v1/videos/{v['id']}")
            loaded = self.tool()[0]["state"] != "unloaded" or bool(self.workers())
            self.check(f"refusal: {r['why']}", st == r.get("status", 400) and not loaded, status=st, message=msg)

        if not self.generate("cold", baseline):
            return self.stop()
        for i in range(a.cycles):
            self.generate(f"cycle{i + 1}", baseline)

        if not a.skip_cancel:
            for phase in self.profile.get("cancel_phases", []):
                self.cancel(phase, baseline)
            self.crash(baseline)
            self.generate("recover", baseline)

        if a.explicit_unload:
            self.explicit_unload(baseline)
        if a.probe_bodiless_post:
            self.bodiless_post()
        if a.after_model:
            st, v = self.http("POST", "/v1/videos", {"model": a.after_model, "prompt": "after", **self.profile.get("after_request", {})})
            if st == 200:
                v, _, pids = self.follow(v["id"])
            self.check(f"after: {a.after_model} generates on the freed device", st == 200 and v["status"] == "completed",
                       status=st, reply=v if st != 200 else v["status"])
        self.stop()

    def cancel(self, phase, baseline):
        st, v = self.submit({"seed": 7})
        if st != 200:
            self.check(f"cancel in {phase}: accepted", False, status=st, reply=v)
            return
        vid, hit = v["id"], {}

        def at_phase(cur):
            if cur.get("phase") == phase:
                hit["pids"] = self.workers()
                self.http("DELETE", f"/v1/videos/{vid}")
                hit["t"] = time.monotonic()
                return True
            return False

        v, phases, pids = self.follow(vid, at_phase)
        if "t" not in hit:
            self.check(f"cancel in {phase}: phase reached", False, status=v["status"], phases=phases)
            return
        self.check(f"cancel in {phase}: job cancelled", v["status"] == "cancelled", status=v["status"], error=v.get("error"))
        st, _ = self.http("GET", f"/v1/videos/{vid}/content")
        self.check(f"cancel in {phase}: no output served", st == 409, status=st)
        self.released(f"cancel in {phase}", sorted(set(pids) | set(hit["pids"])), baseline)

    def crash(self, baseline):
        """Kills the worker mid-job: that job fails with the reason, the daemon stays up, the tool can load again."""
        st, v = self.submit({"seed": 9})
        if st != 200:
            return self.check("crash: accepted", False, status=st, reply=v)
        vid, hit = v["id"], {}
        phase = self.profile.get("crash_phase", "load")

        def kill(cur):
            w = self.workers()
            if cur.get("phase") == phase and w:
                hit["pid"] = w[0]
                self.run_ctl("kill", str(w[0]), "KILL")  # in docker mode the worker is the image's user's, not ours
                return True
            return False

        v, _, pids = self.follow(vid, kill)
        if "pid" not in hit:
            return self.check("crash: worker killed mid-job", False, status=v["status"])
        self.check("crash: that job failed, with a reason", v["status"] == "failed" and bool((v.get("error") or {}).get("message")),
                   status=v["status"], error=v.get("error"))
        self.check("crash: daemon still serving", self.http("GET", "/health")[0] == 200)
        self.released("crash", pids, baseline)

    def explicit_unload(self, baseline):
        t0 = time.monotonic()
        st, _ = self.http("POST", f"/v1/tools/{self.model}/load", timeout=self.profile.get("job_timeout_s", 3600))
        loaded = self.tool()[0]["state"] in ("ready", "busy")
        self.check("unload: explicit load", st == 200 and loaded, status=st, seconds=round(time.monotonic() - t0, 2))
        pids = self.workers()
        self.sample("explicitly loaded")
        st, _ = self.http("POST", f"/v1/tools/{self.model}/unload")
        self.check("unload: explicit unload accepted", st == 200, status=st)
        self.released("unload", pids, baseline)

    def bodiless_post(self):
        """A client's POST with no body and no Content-Length (curl -X POST): the daemon must answer and keep serving."""
        host, port = self.url.split("//")[1].split(":")
        try:
            with socket.create_connection((host, int(port)), timeout=10) as c:
                c.sendall(f"POST /v1/tools/{self.model}/unload HTTP/1.1\r\nHost: {host}\r\nAccept: */*\r\n\r\n".encode())  # as `curl -X POST` sends it: keep-alive
                reply = c.recv(64).decode(errors="replace").split("\r\n")[0]
        except OSError as e:
            reply = f"{type(e).__name__}: {e}"
        time.sleep(0.5)
        try:
            up = self.http("GET", "/health", timeout=5)[0] == 200
        except OSError:
            up = False
        self.check("robust: a bodiless POST leaves the daemon serving", up, reply=reply)
        if not up:  # the daemon died: restart it so the remaining checks (stop) still run against a package
            self.run_ctl("stop", check=False)
            self.run_ctl("start")

    def stop(self):
        pids = self.tree()
        t0 = time.monotonic()
        r = subprocess.run([self.ctl, "stop"], capture_output=True, text=True)
        took = round(time.monotonic() - t0, 2)
        alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
        try:
            up = self.http("GET", "/health", timeout=2)[0] == 200
        except OSError:
            up = False
        self.check("stop: daemon and all its processes exited", r.returncode == 0 and not alive and not up,
                   seconds=took, alive=alive, port_open=up, output=(r.stdout + r.stderr).strip())
        apps = gpu_apps()
        self.check("stop: no process of the package holds GPU memory",
                   None if apps is None else not [x for x in apps if x["pid"] in pids],
                   **({"reason": "no nvidia-smi on this machine"} if apps is None else {"apps": apps}))
        self.receipt["host_mem_available_after_stop_bytes"] = meminfo_available()


def proc_state(pid):
    """The process's state letter (R, S, D, Z for a zombie its parent has not reaped...) and its name."""
    try:
        stat = open(f"/proc/{pid}/stat").read()
        return stat[stat.rindex(")") + 2] + " " + stat[stat.index("(") + 1:stat.rindex(")")]
    except (OSError, ValueError):
        return "gone"


def meminfo_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def gpu_info():
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_apps():
    """The processes holding a GPU context, or None without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    apps = []
    for line in r.stdout.strip().splitlines():
        pid, used = (x.strip() for x in line.split(",", 1))
        apps.append({"pid": int(pid), "used_mib": used})
    return apps


def ffprobe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", path],
                       capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", required=True, help="the model's smoke profile (tools/package/profiles/*.json)")
    p.add_argument("--out", required=True, help="where to write the JSON receipt")
    p.add_argument("--ctl", default=os.path.join(HERE, "localrouterctl"))
    p.add_argument("--cycles", type=int, default=1, help="load/generate/unload cycles after the cold one")
    p.add_argument("--skip-cancel", action="store_true", help="skip the cancel, crash and recover checks")
    p.add_argument("--after-model", help="another configured video model to run once the device is free")
    p.add_argument("--mem-tolerance-gib", type=float, default=2.0, help="allowed MemAvailable drift after a release")
    p.add_argument("--explicit-unload", action="store_true", help="also load and unload the model through /v1/tools")
    p.add_argument("--probe-bodiless-post", action="store_true", help="also check that a POST without a body cannot stop the daemon")
    p.add_argument("--no-start", action="store_true", help="use the daemon that is already running")
    p.add_argument("--keep-outputs", help="a directory to copy the generated MP4s to")
    a = p.parse_args()
    s = Smoke(a)
    t0 = time.monotonic()
    try:
        s.run()
    except Exception as e:  # a crash of the harness is a failed run, with what it had so far
        s.check("harness", False, error=f"{type(e).__name__}: {e}")
        try:
            subprocess.run([a.ctl, "stop"], capture_output=True, timeout=120)
        except Exception:
            pass
    finally:
        shutil.rmtree(s.tmp, ignore_errors=True)
    s.receipt["total_s"] = round(time.monotonic() - t0, 2)
    failed = [c["check"] for c in s.checks if c["result"] == "fail"]
    # a profile whose model contract is not yet confirmed by the engine's owner can fail a run, never pass one
    pending = s.profile.get("provisional", [])
    s.receipt["provisional"] = pending
    s.receipt["result"] = "fail" if failed else ("provisional" if pending else "pass")
    with open(a.out, "w") as f:
        json.dump(s.receipt, f, indent=1)
    print(f"{s.receipt['result']}: {len(s.checks)} checks, {len(failed)} failed, gpu_verified={s.receipt['gpu_verified']} -> {a.out}")
    return 1 if failed else (2 if pending else 0)


if __name__ == "__main__":
    sys.exit(main())
