#!/usr/bin/env python3
"""Fish Audio Drama 3 qualification: the pinned assets, the worker end to end on the GPU, and the marker.

    python3 -I tools/speech/drama3/qualify.py assets ROOT CONFIG OUT [--tool ID]
    python3 -I tools/speech/drama3/qualify.py e2e    ROOT CONFIG OUT [--tool ID]
    python3 -I tools/speech/drama3/qualify.py marker ROOT CONFIG OUT [--tool ID] [--write --accept-native-numerics]

ROOT is the asset root fetch.sh and build.sh filled. CONFIG is the LocalRouter config: its drama3 tool (the speech
tool whose cmd ends in drama3_worker.py; --tool picks one by id) and its top-level "gpu_lock". qualify.sh runs the
three in order, each under a time limit, and is what an operator runs; OUT receives report.json and the audio.

assets checks that the tool serves ROOT's own layout, that the upstream checkout and every checkpoint file match
pins.json by sha256 (fetch.sh --check, which also runs upstream's checkpoint.sha256), and that ROOT/engine.json
names the pinned patched TensorFold HEAD and the binary the worker runs, by sha256.
e2e drives drama3_worker.Worker in this process, as the daemon's worker process does, with the tool's own config
and require_qualification=False, on the real GPU:
  load     cold load and time to ready; the shared GPU lock is held and the pilot's unit is active while loaded.
  F1       a short sentence whole (RIFF/WAVE PCM, mono, 16-bit, 44100 Hz, finish "stop"); again warm, byte for byte
           the same file (Drama 3 seeds each synthesis from its prompt rows); streamed, the same PCM data.
  F2       a 316-character paragraph whole and streamed: the same PCM data; the warm real-time factor.
  voices   one request per configured voice other than "default"; a voice file's text and codes as references,
           when the tool offers references and its voices_dir has one.
  refusal  an unknown voice is refused as invalid_request_error, and the server stays up.
  long     with long_form, an input over long_form_chars is spoken in segments.
  unload   exit: the unit is inactive, the lock is free and MemAvailable is back within 2 GiB of its level before load.
marker writes the tool's "qualification" file (--write, only with --accept-native-numerics and passing assets and
e2e reports); without --write it writes nothing and warns if a marker that says qualified is already there.
"""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "fishaudio/drama-3"
GIB = 1 << 30
NUMERICS = ("native TensorFold build of drama-tensorfold; not bit-exact with SGLang-Omni ab095ef9; accepted by the "
            "operator with --accept-native-numerics")

FIXTURES = {
    "F1": "The train leaves at seven fifteen.",
    "F2": ("The ferry to the island runs twice a day in winter, once at dawn and once in the late afternoon. "
           "Passengers wait in a wooden shelter by the pier, where a kettle is warm and the timetable is chalked "
           "on a board. When the weather turns, the captain rings the harbour office, and the evening crossing "
           "waits for the tide."),
}
WAV_FORMAT = {"format_tag": 1, "channels": 1, "rate": 44100, "bits": 16, "block_align": 2, "byte_rate": 88200}


def worker_module():
    sys.path.insert(0, HERE)
    import drama3_worker
    return drama3_worker


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_tool(config_path, tool_id):
    """The config and its drama3 tool: the one with this id, or the only one."""
    with open(config_path) as f:
        cfg = json.load(f)
    drama = [t for t in cfg.get("tools") or [] if t.get("kind") == "speech" and t.get("cmd")
             and str(t["cmd"][-1]).endswith("drama3_worker.py")]
    chosen = [t for t in drama if t.get("id") == tool_id] or (drama if len(drama) == 1 else [])
    if len(chosen) != 1:
        raise SystemExit(f"qualify.py: {config_path} has no single drama3 speech tool (--tool {tool_id})")
    return cfg, chosen[0]


def layout(root):
    """The paths the worker derives from its weights (ROOT) when no option overrides them."""
    return {"pilot": os.path.join(root, "drama-tensorfold", "recipe", "pilot.py"),
            "native_bin": os.path.join(root, "build", "tensorfold", "zig-out", "native", "bin", "tensorfold-native"),
            "checkpoint": os.path.join(root, "drama-3"),
            "runtime_dir": os.path.join(root, "runtime")}


def tool_paths(tool):
    opts = tool.get("options") or {}
    return {k: opts.get(k) or v for k, v in layout(tool.get("weights", "")).items()}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def git_head(path):
    r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = r.stdout.strip().splitlines()
    return names[0].strip() if r.returncode == 0 and names else None


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def lock_free(path):
    """Whether a non-blocking exclusive flock on the GPU lock succeeds; a lock it takes is released at once."""
    with open(path) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def unit_state(unit):
    r = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "LoadState"],
                       capture_output=True, text=True, timeout=10)
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)


def read_wav(path):
    """The fmt fields and the PCM data of a RIFF/WAVE file; ValueError for anything malformed."""
    with open(path, "rb") as f:
        b = f.read()
    if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("not RIFF/WAVE")
    if struct.unpack_from("<I", b, 4)[0] != len(b) - 8:
        raise ValueError(f"RIFF size {struct.unpack_from('<I', b, 4)[0]}, file {len(b)} bytes")
    fmt = data = None
    i = 12
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack_from("<I", b, i + 4)[0]
        body = b[i + 8:i + 8 + size]
        if len(body) != size:
            raise ValueError(f"chunk {cid!r} truncated")
        if cid == b"fmt " and size >= 16:
            tag, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", body)
            fmt = {"format_tag": tag, "channels": channels, "rate": rate, "bits": bits, "block_align": align,
                   "byte_rate": byte_rate}
        elif cid == b"data":
            data = body
        i += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("no fmt or data chunk")
    return fmt, data, sha(b)


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, **measured):
        self.checks.append({"check": name, "pass": bool(ok), **measured})
        print(json.dumps({"check": name, "pass": bool(ok)}), flush=True)
        return bool(ok)

    def passed(self):
        return bool(self.checks) and all(c["pass"] for c in self.checks)

    def write(self, out, **extra):
        report = {"pass": self.passed(), **extra, "checks": self.checks}
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps({"pass": report["pass"], "checks": len(self.checks),
                          "failed": [c["check"] for c in self.checks if not c["pass"]]}))
        return 0 if report["pass"] else 1


# ── assets ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def assets(args):
    kw = worker_module()
    pins = kw.load_pins()
    rep = Report()
    cfg, tool = load_tool(args.config, args.tool)
    root = os.path.realpath(args.root)
    rep.add("weights_is_root", os.path.realpath(tool.get("weights") or "/nonexistent") == root,
            weights=tool.get("weights"), root=root)
    # The worker must run what is verified here: overrides of the pilot, the binary or the checkpoint are refused.
    paths, expected = tool_paths(tool), layout(root)
    for key in ("pilot", "native_bin", "checkpoint"):
        rep.add(f"{key}_in_root", os.path.realpath(paths[key]) == os.path.realpath(expected[key]),
                path=paths[key], expected=expected[key])
    lock = cfg.get("gpu_lock") or ""
    rep.add("gpu_lock", bool(lock) and os.path.isfile(lock), gpu_lock=lock)
    rep.add("qualification_path", bool(tool.get("qualification")), qualification=tool.get("qualification"))
    with open(os.path.join(args.out, "fetch-check.log"), "w") as log:
        t = time.monotonic()
        r = subprocess.run([os.path.join(HERE, "fetch.sh"), root, "--check"], stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL)
    rep.add("pinned_files_sha256", r.returncode == 0, exit=r.returncode, seconds=time.monotonic() - t,
            log="fetch-check.log", files=len(pins["checkpoint"]["files"]))
    engine_path = os.path.join(root, "engine.json")
    try:
        engine = read_json(engine_path)
    except (OSError, ValueError) as e:
        rep.add("engine_json", False, error=f"{engine_path}: {e}")
        return rep.write(args.out, pins_sha256=kw.sha256_file(kw.PINS))
    want = pins["tensorfold"]["patched_head"]
    rep.add("engine_patched_head", engine.get("patched_head") == want, engine=engine.get("patched_head"), pinned=want)
    rep.add("engine_upstream_commit", engine.get("upstream_commit") == pins["upstream"]["commit"],
            engine=engine.get("upstream_commit"), pinned=pins["upstream"]["commit"])
    head = git_head(os.path.join(root, "build", "tensorfold"))
    rep.add("build_checkout_head", head == want, head=head, pinned=want)
    native = paths["native_bin"]
    rep.add("engine_native_bin", os.path.realpath(engine.get("native_bin") or "/nonexistent") ==
            os.path.realpath(native), engine=engine.get("native_bin"), tool=native)
    digest = kw.sha256_file(native) if os.path.isfile(native) else None
    rep.add("native_sha256", digest is not None and digest == engine.get("native_sha256"), sha256=digest,
            engine=engine.get("native_sha256"))
    return rep.write(args.out, pins_sha256=kw.sha256_file(kw.PINS), engine=engine)


# ── e2e ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def speech(text, voice, stream=False):
    return {"input": text, "voice": voice, "speed": 1, "preset": "", "seed": 0, "stream": stream, "references": ""}


def synth(w, out, label, s, references=None):
    """One request through the worker into OUT/<label>/; the reply, the timing and the parsed speech.wav."""
    d = os.path.join(out, label)
    os.makedirs(d, exist_ok=True)
    wav = os.path.join(d, "speech.wav")
    for name in ("speech.wav", "references.json"):
        if os.path.exists(os.path.join(d, name)):
            os.remove(os.path.join(d, name))
    s = dict(s)
    if references is not None:
        with open(os.path.join(d, "references.json"), "w") as f:
            json.dump(references, f)
        s["references"] = "references.json"
    t = time.monotonic()
    r = w.generate({"op": "generate", "id": label.replace("/", "-"), "dir": d, "request": {"speech": s}})
    res = {"label": label, "reply": r, "wall_s": time.monotonic() - t, "input_chars": len(s["input"]),
           "voice": s["voice"], "stream": s["stream"]}
    if references is not None:
        # A voice file holds a person's voice: its codes do not stay in the report.
        os.remove(os.path.join(d, "references.json"))
    if r.get("ok"):
        try:
            fmt, data, digest = read_wav(wav)
            res.update(fmt=fmt, data=data, sha256=digest, audio_s=len(data) / WAV_FORMAT["byte_rate"])
            res["rtf"] = res["wall_s"] / res["audio_s"] if res["audio_s"] else None
        except (OSError, ValueError) as e:
            res["wav_error"] = f"{type(e).__name__}: {e}"
    return res


def measured(res):
    """A synth result for the report, without the PCM."""
    return {k: v for k, v in res.items() if k != "data"}


def wav_ok(res, finish="stop"):
    r = res["reply"]
    return (r.get("ok") is True and "speech.wav" in (r.get("files") or []) and res.get("fmt") == WAV_FORMAT
            and len(res.get("data") or b"") > 0 and len(res["data"]) % 2 == 0
            and (finish is None or r.get("finish_reason") == finish))


def voice_file(voices_dir):
    """The first voice file in voices_dir as one reference {text, vq_codes}, and its name; (None, None) if none."""
    if not voices_dir or not os.path.isdir(voices_dir):
        return None, None
    for name in sorted(os.listdir(voices_dir)):
        path = os.path.join(voices_dir, name)
        if name.startswith(".") or not name.endswith(".json") or os.path.islink(path) or not os.path.isfile(path):
            continue
        try:
            v = read_json(path)
        except (OSError, ValueError):
            continue
        entry = v if v.get("format") == "drama3-voice/1" else (v.get("references") or [None])[0]
        if isinstance(entry, dict) and entry.get("text") and entry.get("vq_codes"):
            return {"text": entry["text"], "vq_codes": entry["vq_codes"]}, name
    return None, None


def e2e(args):
    kw = worker_module()
    pins = kw.load_pins()
    cfg, tool = load_tool(args.config, args.tool)
    opts = tool.get("options") or {}
    lock = cfg.get("gpu_lock") or ""
    rep = Report()
    env = {"host": os.uname().nodename, "gpu": gpu_name(), "python": platform.python_version(),
           "machine": platform.machine(), "pins_sha256": kw.sha256_file(kw.PINS), "tool": tool.get("id"),
           "options": opts, "gpu_lock": lock}
    with open("/proc/loadavg") as f:
        env["loadavg"] = f.read().split()[:3]
    # A timeout's SIGTERM unwinds through the finally below, so the server is stopped and the lock released.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    w = None
    exited = False
    try:
        if not lock or not os.path.isfile(lock):
            rep.add("gpu_lock", False, error=f"the config's gpu_lock {lock!r} is not an existing file")
            return 1
        if not lock_free(lock):
            rep.add("gpu_lock_free_before_load", False, error="another GPU job holds the lock; nothing was loaded")
            return 1
        before = mem_available()
        environ = dict(os.environ, LOCALROUTER_GPU_LOCK=lock)
        w = kw.Worker(dict(tool), require_qualification=False, environ=environ)
        t = time.monotonic()
        loaded = w.load()
        ready_s = time.monotonic() - t
        if not rep.add("load", loaded.get("ok") is True, reply=loaded, ready_s=ready_s,
                       mem_available_before=before, mem_available_loaded=mem_available()):
            return 1
        rep.add("gpu_lock_held_while_loaded", not lock_free(lock))
        unit = w.unit
        state = unit_state(unit) if unit else {}
        rep.add("unit_active", state.get("ActiveState") == "active", unit=unit, state=state)

        voice = opts.get("default_voice") or "default"
        f1 = synth(w, args.out, "F1", speech(FIXTURES["F1"], voice))
        rep.add("F1_whole", wav_ok(f1), **measured(f1))
        f1w = synth(w, args.out, "F1-warm", speech(FIXTURES["F1"], voice))
        rep.add("F1_warm_identical", wav_ok(f1w) and f1w.get("sha256") == f1.get("sha256"), **measured(f1w),
                cold_sha256=f1.get("sha256"))
        if opts.get("stream", True):
            f1s = synth(w, args.out, "F1-stream", speech(FIXTURES["F1"], voice, stream=True))
            rep.add("F1_stream_equal", wav_ok(f1s) and f1s.get("data") == f1.get("data"), **measured(f1s),
                    whole_sha256=f1.get("sha256"))
        f2 = synth(w, args.out, "F2", speech(FIXTURES["F2"], voice))
        rep.add("F2_whole", wav_ok(f2), **measured(f2), warm_rtf=f2.get("rtf"))
        if opts.get("stream", True):
            f2s = synth(w, args.out, "F2-stream", speech(FIXTURES["F2"], voice, stream=True))
            rep.add("F2_stream_equal", wav_ok(f2s) and f2s.get("data") == f2.get("data"), **measured(f2s),
                    whole_sha256=f2.get("sha256"))

        for v in [v for v in opts.get("voices") or ["default"] if v != "default"]:
            r = synth(w, args.out, f"voice-{v}", speech(FIXTURES["F1"], v))
            rep.add(f"voice_{v}", wav_ok(r, finish=None), **measured(r))
        if opts.get("references"):
            ref, name = voice_file(opts.get("voices_dir"))
            if ref is None:
                rep.add("references", True, skipped=f"no voice file in voices_dir {opts.get('voices_dir')!r}")
            else:
                r = synth(w, args.out, "references", speech(FIXTURES["F1"], "default"), references=[ref])
                rep.add("references", wav_ok(r, finish=None), **measured(r), voice_file=name)

        d = os.path.join(args.out, "refusal-voice")
        os.makedirs(d, exist_ok=True)
        r = w.generate({"op": "generate", "id": "refusal-voice", "dir": d,
                        "request": {"speech": speech(FIXTURES["F1"], "no-such-voice")}})
        rep.add("refusal_unknown_voice", r.get("ok") is False and r.get("type") == "invalid_request_error", reply=r)
        state = unit_state(unit) if unit else {}
        rep.add("unit_active_after_refusal", state.get("ActiveState") == "active", state=state)

        if opts.get("long_form"):
            chars = int(opts.get("long_form_chars", 500))
            limit = int(opts.get("max_input_chars", 4096))
            text = FIXTURES["F2"]
            while len(text) <= chars:
                text += " " + FIXTURES["F2"]
            if len(text) > limit:
                rep.add("long_form", False, error=f"no input over long_form_chars {chars} fits max_input_chars {limit}")
            else:
                r = synth(w, args.out, "long-form", speech(text, voice))
                rep.add("long_form", wav_ok(r, finish=None) and bool(r["reply"].get("segments")), **measured(r),
                        long_form_chars=chars, segments=r["reply"].get("segments"))

        loaded_avail = mem_available()
        t = time.monotonic()
        w.exit()
        exited = True
        exit_s = time.monotonic() - t
        deadline = time.monotonic() + 60
        state = unit_state(unit) if unit else {}
        while unit and state.get("ActiveState") != "inactive" and time.monotonic() < deadline:
            time.sleep(0.5)
            state = unit_state(unit)
        rep.add("unload_unit_inactive", bool(unit) and state.get("ActiveState") == "inactive", state=state,
                exit_s=exit_s)
        rep.add("unload_lock_free", lock_free(lock))
        deadline = time.monotonic() + 120
        after = mem_available()
        while after < before - 2 * GIB and time.monotonic() < deadline:
            time.sleep(1)
            after = mem_available()
        rep.add("unload_memory_recovered", after >= before - 2 * GIB, mem_available_before=before,
                mem_available_loaded=loaded_avail, mem_available_after=after, recovered_bytes=after - loaded_avail)
        return 0
    except Exception as e:  # a crash is recorded, never silently a partial result
        rep.add("crash", False, error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        if w is not None and not exited:
            try:
                w.exit()
            except Exception as e:
                rep.add("exit_after_failure", False, error=f"{type(e).__name__}: {e}")
        rc = rep.write(args.out, model=MODEL, fixtures=FIXTURES, env=env,
                       upstream_commit=pins["upstream"]["commit"], checkpoint_revision=pins["checkpoint"]["revision"])
        if rc:
            sys.exit(rc)


# ── marker ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def marker(args):
    kw = worker_module()
    pins = kw.load_pins()
    _, tool = load_tool(args.config, args.tool)
    path = tool.get("qualification") or ""
    if not path:
        print(f"qualify.py: tool {tool.get('id')} has no \"qualification\" path; no marker", file=sys.stderr)
        return 1
    try:
        existing = read_json(path)
    except (OSError, ValueError):
        existing = None
    if args.revoke:
        # Only the marker this run wrote (its report is OUT) is removed.
        if isinstance(existing, dict) and existing.get("report") == args.out:
            os.remove(path)
            print(f"qualify.py: removed {path}, written by this run")
        return 0
    if not args.write:
        if isinstance(existing, dict) and existing.get("qualified") is True:
            print(f"qualify.py: WARNING: {path} exists and says qualified (report {existing.get('report')}); this run "
                  "did not qualify, and it was left as it is")
        else:
            print(f"qualify.py: not qualified; no marker written at {path}")
        return 0
    if not args.accept_native_numerics:
        print("qualify.py: --write needs --accept-native-numerics", file=sys.stderr)
        return 1
    for step in ("assets", "e2e"):
        try:
            ok = read_json(os.path.join(args.out, step, "report.json")).get("pass") is True
        except (OSError, ValueError):
            ok = False
        if not ok:
            print(f"qualify.py: {args.out}/{step}/report.json does not pass; no marker", file=sys.stderr)
            return 1
    engine = read_json(os.path.join(os.path.realpath(args.root), "engine.json"))
    native = tool_paths(tool)["native_bin"]
    digest = kw.sha256_file(native)
    if digest != engine.get("native_sha256") or engine.get("patched_head") != pins["tensorfold"]["patched_head"]:
        print(f"qualify.py: {native} sha256 {digest} is not engine.json's {engine.get('native_sha256')}; no marker",
              file=sys.stderr)
        return 1
    gpu = gpu_name()
    if not gpu:
        print("qualify.py: nvidia-smi names no GPU; no marker", file=sys.stderr)
        return 1
    record = {"qualified": True, "model": MODEL, "host": os.uname().nodename, "gpu": gpu,
              "pins_sha256": kw.sha256_file(kw.PINS), "native_sha256": digest,
              "upstream_commit": pins["upstream"]["commit"], "checkpoint_revision": pins["checkpoint"]["revision"],
              "numerics": NUMERICS, "report": args.out,
              "qualified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    text = json.dumps(record, indent=1) + "\n"
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
    with open(os.path.join(args.out, "marker.json"), "w") as f:
        f.write(text)
    print(f"qualify.py: wrote {path}")
    print(text, end="")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["assets", "e2e", "marker"])
    ap.add_argument("root")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--tool", default=MODEL)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--revoke", action="store_true")
    ap.add_argument("--accept-native-numerics", action="store_true")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    return {"assets": assets, "e2e": e2e, "marker": marker}[args.role](args) or 0


if __name__ == "__main__":
    sys.exit(main())
