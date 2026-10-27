#!/usr/bin/env python3
"""Fish Audio Drama 3 qualification: the pinned assets, the worker end to end on the GPU, its speed, and the marker.

    python3 -I tools/speech/drama3/qualify.py assets ROOT CONFIG OUT [--tool ID]
    python3 -I tools/speech/drama3/qualify.py e2e    ROOT CONFIG OUT [--tool ID]
    python3 -I tools/speech/drama3/qualify.py bench  ROOT CONFIG OUT [--tool ID] [--coresident]
    python3 -I tools/speech/drama3/qualify.py accuracy ROOT CONFIG OUT --reference RUNTIME [--tool ID]
    python3 -I tools/speech/drama3/qualify.py marker ROOT CONFIG OUT [--tool ID] [--write --accept-native-numerics
                                                                     [--accept-lossy]]

ROOT is the asset root fetch.sh and build.sh filled. CONFIG is the LocalRouter config: its drama3 tool (the speech
tool whose cmd ends in drama3_worker.py; --tool picks one by id) and its top-level "gpu_lock". qualify.sh runs the
five in order, each under a time limit, and is what an operator runs; OUT receives report.json and the audio.

e2e and bench hold the GPU lock as the daemon does (LocalRouter PR #14): the host's "gpu_lock" shared (flock
LOCK_SH, non-blocking: a job holding it exclusively fails the "host_gpu_lock" check and nothing is loaded) for as
long as the model is loaded, and the worker gets a private lock file in OUT (LOCALROUTER_GPU_LOCK, with
LOCALROUTER_GPU_LOCK_HELD=1 and LOCALROUTER_HOST_GPU_LOCK=<host lock>), which the launcher takes exclusively. The
worker runs ROOT/launcher/pilot.py with admission "external" unless the tool says otherwise. accuracy's guard takes
the host lock exclusively itself, so it runs after bench has released it (each role is its own process).

assets checks that the tool serves ROOT's own layout, that the upstream checkout and every checkpoint file match
pins.json by sha256 (fetch.sh --check, which also runs upstream's checkpoint.sha256), and that ROOT/engine.json
names the pinned patched TensorFold HEAD and the binary the worker runs, by sha256.
e2e drives drama3_worker.Worker in this process, as the daemon's worker process does, with the tool's own config
and require_qualification=False, on the real GPU:
  load     cold load and time to ready; the private lock is held and the pilot's unit is active while loaded.
  F1       a short sentence whole (RIFF/WAVE PCM, mono, 16-bit, 44100 Hz, finish "stop"); again warm, byte for byte
           the same file (Drama 3 seeds each synthesis from its prompt rows); streamed, the same PCM data.
  F2       a 316-character paragraph whole and streamed: the same PCM data; the warm real-time factor.
  voices   one request per configured voice other than "default"; a voice file's text and codes as references,
           when the tool offers references and its voices_dir has one.
  refusal  an unknown voice is refused as invalid_request_error, and the server stays up.
  long     with long_form, an input over long_form_chars is spoken in segments.
  unload   exit: the unit is inactive, the private lock is free and MemAvailable is back within 2 GiB of its level
           before load.
bench loads the worker the same way and runs upstream's own benchmark client, ROOT/drama-tensorfold/bench/
drama_bench.py, against the server it started, as upstream's bench/before-after.sh does, so the numbers stand next
to upstream's: whole WAV replies on the fresh process (1 cold, 2 warmup, 3 warm passes of the 24-item corpus), then
streamed PCM on the same process (2 warmup, 3 warm passes). Every 0.5 s while they run it samples the unit's cgroup
memory.current and memory.peak, MemAvailable and nvidia-smi memory.used. It passes when both runs exit 0 (every
request valid audio, the server unchanged), the warm streamed first_frame_s p50 (first playable audio) and the warm
real-time factor p50 of both runs are within THRESHOLDS, and the peak (the larger of the cgroup's peak and the drop in
MemAvailable from before the load: GB10's CUDA memory does not show in the cgroup) is within the tool's resident_bytes.
The ratios to upstream's claims (UPSTREAM_CLAIMS) are recorded, not gated. The report records MemAvailable before
load and, when a LocalRouter daemon answers on 127.0.0.1:8190, its /v1/tools: with --coresident the operator says
chat and speech-to-text are loaded by a running daemon (the Jarvis box: one GB10, all of them warm together). Its
models hold the host lock shared, so e2e and bench run beside them, and accuracy's exclusive guard cannot.
accuracy runs upstream's own qualification/ checks against the SGLang-Omni reference (see accuracy()); their
verdicts are recorded, and do not decide acceptance.
marker writes the tool's "qualification" file (--write, only with --accept-native-numerics, passing assets, e2e and
bench reports and an accuracy report with valid evidence, whose results it records with bench's measurements);
without --write it writes nothing and warns if a marker that says qualified is already there.

A tool with the option "variant" (nvfp4: plotarmordev/drama-3-NVFP4-Pruned, pins-nvfp4.json) is checked against its
own pins, checkpoint, engine build and ROOT/engine-nvfp4.json. Upstream's accuracy checks do not apply to it (the
SGLang-Omni reference does not load the quant, and its engine builds no oracle tools): accuracy records that, and
the marker needs --accept-lossy, the operator's recorded acceptance of a lossy checkpoint (pins-nvfp4.json's "lossy"
names who approved it), and binds both pins files. Its quality is measured against the full model by compare.py.

bench's peak is also measured per process, the GB10 way: the server's nvidia-smi used_memory (its device
allocations, which show in no RSS) plus its RssAnon and RssShmem, sampled with the rest; when that was measured it is
the peak the gate and the marker use (MemAvailable's drop moves with every co-resident model), else the drop.
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
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "fishaudio/drama-3"
GIB = 1 << 30
NUMERICS = ("native TensorFold build of drama-tensorfold; not bit-exact with SGLang-Omni ab095ef9; accepted by the "
            "operator with --accept-native-numerics")

# bench's gate. Real time is the reason Drama 3 is the Jarvis box's text to speech: a reply must be spoken faster
# than it plays, and its first audio must come within half a second. The peak (see with_drop) must fit the memory the
# scheduler admitted the load with (the tool's resident_bytes, else the launcher's 54 GiB cap).
THRESHOLDS = {"stream_first_frame_s_p50_max": 0.5, "warm_rtf_p50_below": 1.0, "default_resident_bytes": 54 * GIB}
# Upstream's claims for the same client, corpus and schedule (README.md, results/qualification.json and
# results/before-after-02/: warm, whole RTF and streamed first audio); patch 0040 later cost about 5% (README.md,
# results/backbone-attention-tiles.json: RTF 0.682 on its 48-request capture). Recorded beside ours, not gated.
UPSTREAM_CLAIMS = {"first_audio_p50_s": 0.141, "rtf_p50": 0.634, "rtf_p50_after_patch_0040": 0.682,
                   "source": "drama-tensorfold README.md and results/before-after-02 (GB10, native TensorFold)"}
CORPUS = os.path.join("bench", "corpus", "drama-bench-en-v1.jsonl")
CORPUS_SHA256 = "b8995194915f75ffa8860eaca5fc75821d30d209db907d80a07af946ebf59d3b"  # upstream's bench/README.md
# drama_bench.py's own defaults (2 warmup, 3 repeats, 120 s a request) give 75 and 74 requests, about 500 s each at
# upstream's speed. Each run is bounded, and the role leaves itself 300 s of the step's 3000 s to unload.
BENCH_RUN_LIMIT_S = 1200
BENCH_BUDGET_S = 2700
DAEMON_TOOLS = "http://127.0.0.1:8190/v1/tools"  # the LocalRouter daemon's default port

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


def tool_variant(tool):
    """The tool's weights variant and its pins (None for the full model, pins.json's)."""
    variant = (tool.get("options") or {}).get("variant", "full")
    return variant, worker_module().load_variant_pins(variant)


def build_dir(root, vpins):
    return os.path.join(root, vpins["build"]["dir"] if vpins else os.path.join("build", "tensorfold"))


def git_checkouts(root, pins, vpins):
    """The git checkouts under ROOT that fetch.sh --check and this script read: upstream's, and the engine build's."""
    return [os.path.join(root, pins["upstream"]["dir"]), build_dir(root, vpins)]


def layout(root, vpins=None):
    """The paths the worker derives from its weights (ROOT) when no option overrides them."""
    return {"pilot": os.path.join(root, "launcher", "pilot.py"),
            "native_bin": os.path.join(build_dir(root, vpins), "zig-out", "native", "bin", "tensorfold-native"),
            "checkpoint": os.path.join(root, vpins["checkpoint"]["dir"] if vpins else "drama-3"),
            "runtime_dir": os.path.join(root, "runtime")}


def tool_paths(tool):
    opts = tool.get("options") or {}
    return {k: opts.get(k) or v for k, v in layout(tool.get("weights", ""), tool_variant(tool)[1]).items()}


def engine_record(root, vpins):
    """build.sh's record of the engine: ROOT/engine.json, or the variant's."""
    return os.path.join(root, vpins["build"]["record"] if vpins else "engine.json")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def git_env(*paths, base=None):
    """The environment for this script's read-only git calls on PATHS: each a safe.directory (git refuses a checkout
    another user owns, as when the qualify runs as root on assets fetched by the owner), and no optional locks (git
    status does not rewrite the index of a checkout that is not ours)."""
    env = dict(os.environ if base is None else base)
    n = int(env.get("GIT_CONFIG_COUNT", "0") or 0)
    for p in paths:
        env[f"GIT_CONFIG_KEY_{n}"], env[f"GIT_CONFIG_VALUE_{n}"] = "safe.directory", os.path.realpath(p)
        n += 1
    env["GIT_CONFIG_COUNT"] = str(n)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def ownership_note(path, uid=None):
    """Why git may refuse PATH: it is owned by another user than this run's (None: same owner, or no such path)."""
    uid = os.getuid() if uid is None else uid
    try:
        owner = os.stat(path).st_uid
    except OSError:
        return None
    if owner == uid:
        return None
    return f"{path} is owned by uid {owner} and this run is uid {uid}: git refuses such a checkout (dubious ownership)"


def git_head(path):
    """The checkout's HEAD and None, or None and why not (PATH itself must be the checkout: a plain directory inside
    another repository is not)."""
    r = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel", "HEAD"], capture_output=True, text=True,
                       timeout=60, env=git_env(path))
    if r.returncode == 0:
        top, head = (r.stdout.splitlines() + ["", ""])[:2]
        if os.path.realpath(top) != os.path.realpath(path):
            return None, f"{path} is not a git checkout (it is inside {top})"
        return head, None
    why = (r.stderr.strip().splitlines() or [f"git exit {r.returncode}"])[0]
    note = ownership_note(path) if "dubious ownership" in r.stderr else None
    return None, f"{why}{'; ' + note if note else ''}"


def gpu_lock_problem(lock):
    """Why the config's gpu_lock cannot be held (None: it can)."""
    if not lock:
        return ("the config has no gpu_lock: set its top-level \"gpu_lock\" to the host's GPU lock file (the one the "
                "daemon holds), or to an empty file made for this host")
    if not os.path.isfile(lock):
        return f"the config's gpu_lock {lock!r} is not an existing file"
    return None


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
    """Whether a non-blocking exclusive flock on a lock file succeeds; a lock it takes is released at once."""
    with open(path) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def hold_shared(path):
    """The host GPU lock held shared, as the daemon holds it while a model is loaded; None while a job holds it
    exclusively. Closing the file releases it."""
    f = open(path)
    try:
        fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def worker_environ(out, host_lock):
    """The worker's environment as the daemon gives it: a private lock (created in OUT) the launcher takes, the
    host lock already held."""
    private = os.path.join(out, "drama3.lock")
    open(private, "a").close()
    return private, dict(os.environ, LOCALROUTER_GPU_LOCK=private, LOCALROUTER_GPU_LOCK_HELD="1",
                         LOCALROUTER_HOST_GPU_LOCK=host_lock)


def unit_state(unit):
    r = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "LoadState"],
                       capture_output=True, text=True, timeout=10)
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)


def unit_property(unit, name):
    r = subprocess.run(["systemctl", "--user", "show", unit, "-p", name, "--value"], capture_output=True, text=True,
                       timeout=10)
    return r.stdout.strip() if r.returncode == 0 else ""


def wait_inactive(unit, seconds):
    deadline = time.monotonic() + seconds
    state = unit_state(unit) if unit else {}
    while unit and state.get("ActiveState") != "inactive" and time.monotonic() < deadline:
        time.sleep(0.5)
        state = unit_state(unit)
    return state


def daemon_tools():
    """What a running LocalRouter daemon has loaded (its /v1/tools), or why there is no answer."""
    try:
        with urllib.request.urlopen(DAEMON_TOOLS, timeout=5) as r:
            return {"url": DAEMON_TOOLS, "tools": json.loads(r.read(1 << 20))}
    except (OSError, ValueError) as e:
        return {"url": DAEMON_TOOLS, "error": f"{type(e).__name__}: {e}"}


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
    variant, vpins = tool_variant(tool)
    root = os.path.realpath(args.root)
    rep.add("weights_is_root", os.path.realpath(tool.get("weights") or "/nonexistent") == root,
            weights=tool.get("weights"), root=root)
    # The worker must run what is verified here: overrides of the pilot, the binary or the checkpoint are refused.
    paths, expected = tool_paths(tool), layout(root, vpins)
    for key in ("pilot", "native_bin", "checkpoint"):
        rep.add(f"{key}_in_root", os.path.realpath(paths[key]) == os.path.realpath(expected[key]),
                path=paths[key], expected=expected[key])
    lock = cfg.get("gpu_lock") or ""
    problem = gpu_lock_problem(lock)
    rep.add("gpu_lock", problem is None, gpu_lock=lock, error=problem)
    rep.add("qualification_path", bool(tool.get("qualification")), qualification=tool.get("qualification"))
    with open(os.path.join(args.out, "fetch-check.log"), "w") as log:
        t = time.monotonic()
        r = subprocess.run([os.path.join(HERE, "fetch.sh"), root, "--variant", variant, "--check"], stdout=log,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           env=git_env(*git_checkouts(root, pins, vpins)))
    rep.add("pinned_files_sha256", r.returncode == 0, exit=r.returncode, seconds=time.monotonic() - t,
            log="fetch-check.log", variant=variant, files=len((vpins or pins)["checkpoint"]["files"]))
    engine_path = engine_record(root, vpins)
    try:
        engine = read_json(engine_path)
    except (OSError, ValueError) as e:
        rep.add("engine_json", False, error=f"{engine_path}: {e}")
        return rep.write(args.out, pins_sha256=kw.sha256_file(kw.PINS))
    want = (vpins or pins)["engine_patches"]["head"]  # the engine's commit, then LocalRouter's engine patches (build.sh)
    rep.add("engine_patched_head", engine.get("patched_head") == want, engine=engine.get("patched_head"), pinned=want)
    if vpins:
        rep.add("engine_tensorfold_commit", engine.get("tensorfold_commit") == vpins["tensorfold"]["commit"],
                engine=engine.get("tensorfold_commit"), pinned=vpins["tensorfold"]["commit"])
    else:
        rep.add("engine_upstream_commit", engine.get("upstream_commit") == pins["upstream"]["commit"],
                engine=engine.get("upstream_commit"), pinned=pins["upstream"]["commit"])
    head, why = git_head(build_dir(root, vpins))
    rep.add("build_checkout_head", head == want, head=head, pinned=want, error=why)
    native = paths["native_bin"]
    rep.add("engine_native_bin", os.path.realpath(engine.get("native_bin") or "/nonexistent") ==
            os.path.realpath(native), engine=engine.get("native_bin"), tool=native)
    digest = kw.sha256_file(native) if os.path.isfile(native) else None
    rep.add("native_sha256", digest is not None and digest == engine.get("native_sha256"), sha256=digest,
            engine=engine.get("native_sha256"))
    return rep.write(args.out, pins_sha256=kw.sha256_file(kw.PINS), variant=variant, engine=engine)


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
    # A timeout's SIGTERM unwinds through the finally below, so the server is stopped and the locks released.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    w = None
    held = None
    exited = False
    try:
        problem = gpu_lock_problem(lock)
        if problem:
            rep.add("gpu_lock", False, error=problem)
            return 1
        held = hold_shared(lock)
        if not rep.add("host_gpu_lock", held is not None, gpu_lock=lock, mode="shared",
                       error=None if held else "a GPU job holds the host lock exclusively; nothing was loaded"):
            return 1
        private, environ = worker_environ(args.out, lock)
        env["private_lock"] = private
        before = mem_available()
        w = kw.Worker(dict(tool), require_qualification=False, environ=environ)
        t = time.monotonic()
        loaded = w.load()
        ready_s = time.monotonic() - t
        if not rep.add("load", loaded.get("ok") is True, reply=loaded, ready_s=ready_s,
                       mem_available_before=before, mem_available_loaded=mem_available()):
            return 1
        rep.add("private_lock_held_while_loaded", not lock_free(private), private_lock=private)
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
        state = wait_inactive(unit, 60)
        rep.add("unload_unit_inactive", bool(unit) and state.get("ActiveState") == "inactive", state=state,
                exit_s=exit_s)
        rep.add("unload_private_lock_free", lock_free(private))
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
        if held is not None:
            held.close()  # the host lock, held shared since before the load
        rc = rep.write(args.out, model=MODEL, fixtures=FIXTURES, env=env,
                       upstream_commit=pins["upstream"]["commit"],
                       checkpoint_revision=(tool_variant(tool)[1] or pins)["checkpoint"]["revision"])
        if rc:
            sys.exit(rc)


# ── bench ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def read_int(path):
    text = read_text(path)
    return int(text) if text and text.isdigit() else None


def gpu_used_mib():
    """nvidia-smi's memory.used; None where it reports none (a GB10's unified memory may read [N/A])."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = (r.stdout.strip().splitlines() or [""])[0].strip()
    return int(first) if r.returncode == 0 and first.isdigit() else None


def process_used_mib(pid):
    """nvidia-smi's used_memory of one process (its device allocations), MiB; None when it lists none for it."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
            return int(fields[1])
    return None


def process_rss(pid):
    """RssAnon and RssShmem of one process, bytes ({} once it is gone)."""
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key in ("RssAnon", "RssShmem"):
                    out[key] = int(value.split()[0]) * 1024
    except OSError:
        return {}
    return out


class MemorySampler:
    """Every 0.5 s, into a JSON-lines log: the unit's cgroup memory.current and memory.peak, MemAvailable,
    nvidia-smi's memory.used, and the server process's own footprint (`pid`: its nvidia-smi used_memory plus RssAnon
    and RssShmem, the GB10 measure). The highest of each is kept, and MemAvailable's lowest and highest."""

    def __init__(self, cgroup, path, pid=0):
        self.cgroup, self.path, self.pid = cgroup, path, pid
        self.stop = threading.Event()
        self.high = {"cgroup_current_bytes": None, "cgroup_peak_bytes": None, "gpu_used_mib": None,
                     "process_bytes": None, "process_gpu_bytes": None, "process_rss_anon_bytes": None,
                     "process_rss_shmem_bytes": None}
        self.available = []
        self.samples = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        s = {"t": time.time(), "cgroup_current_bytes": read_int(os.path.join(self.cgroup, "memory.current")),
             "cgroup_peak_bytes": read_int(os.path.join(self.cgroup, "memory.peak")),
             "mem_available_bytes": mem_available(), "gpu_used_mib": gpu_used_mib()}
        if self.pid:
            used, rss = process_used_mib(self.pid), process_rss(self.pid)
            s["process_gpu_bytes"] = used << 20 if used is not None else None
            s["process_rss_anon_bytes"], s["process_rss_shmem_bytes"] = rss.get("RssAnon"), rss.get("RssShmem")
            parts = (s["process_gpu_bytes"], s["process_rss_anon_bytes"], s["process_rss_shmem_bytes"])
            s["process_bytes"] = sum(parts) if all(v is not None for v in parts) else None
        for k, v in self.high.items():
            if k not in s:
                continue
            if s[k] is not None:
                self.high[k] = s[k] if v is None else max(v, s[k])
        a = s["mem_available_bytes"]
        self.available = [min(self.available[0], a), max(self.available[1], a)] if self.available else [a, a]
        self.samples += 1
        return s

    def run(self):
        with open(self.path, "w") as f:
            while True:
                f.write(json.dumps(self.sample()) + "\n")
                f.flush()
                if self.stop.wait(0.5):
                    return

    def start(self):
        self.thread.start()

    def finish(self):
        """Stops the sampling; one last read, since memory.peak covers the cgroup's whole life."""
        if not self.stop.is_set():
            self.stop.set()
            self.thread.join(timeout=15)
            self.sample()
        peaks = [v for v in (self.high["cgroup_current_bytes"], self.high["cgroup_peak_bytes"]) if v is not None]
        return {"samples": self.samples, "interval_s": 0.5, "log": os.path.basename(self.path), "cgroup": self.cgroup,
                "cgroup_current_max_bytes": self.high["cgroup_current_bytes"],
                "cgroup_peak_max_bytes": self.high["cgroup_peak_bytes"], "peak_bytes": max(peaks) if peaks else None,
                "mem_available_min_bytes": self.available[0] if self.available else None,
                "mem_available_max_bytes": self.available[1] if self.available else None,
                "gpu_used_max_mib": self.high["gpu_used_mib"], "pid": self.pid,
                "process_peak_bytes": self.high["process_bytes"],
                "process_peaks": {k: self.high[k] for k in ("process_gpu_bytes", "process_rss_anon_bytes",
                                                             "process_rss_shmem_bytes")}}


def pilot_evidence(w):
    """The launcher's evidence directory: the worker's, if it names one, else the one under ROOT/evidence whose
    session.json names the worker's unit (the newest, should there be two)."""
    named = getattr(w, "evidence", None)
    if named:
        return named if os.path.isdir(named) else None
    base = os.path.join(w.opts.root, "evidence")
    try:
        names = os.listdir(base)
    except OSError:
        return None
    found = []
    for name in names:
        d = os.path.join(base, name)
        try:
            session = read_json(os.path.join(d, "session.json"))
        except (OSError, ValueError):
            continue
        if isinstance(session, dict) and w.unit and session.get("unit") == w.unit:
            found.append((os.path.getmtime(d), d))
    return max(found)[1] if found else None


def bench_summary(path):
    """What the gate and the report need from one drama_bench.py summary.json."""
    try:
        s = read_json(path)
    except (OSError, ValueError) as e:
        return {"error": f"no summary.json: {e}"}
    phases = s.get("phases") or {}

    def stat(phase, key, form, p):
        return (((phases.get(phase) or {}).get(key) or {}).get(form) or {}).get(p)

    warm = phases.get("warm") or {}
    out = {"complete": s.get("complete"), "scheduled": s.get("scheduled"), "completed": s.get("completed"),
           "server_process_unchanged": s.get("server_process_unchanged"),
           "failures": {phase: v.get("failures") for phase, v in phases.items()},
           "warm": {"attempted": warm.get("attempted"), "succeeded": warm.get("succeeded"),
                    "rtf_p50": stat("warm", "realtime_factor", "ok_only", "p50"),
                    "rtf_p95": stat("warm", "realtime_factor", "ok_only", "p95"),
                    "rtf_p50_all_attempts": stat("warm", "realtime_factor", "all_attempts", "p50"),
                    "end_to_end_s_p50": stat("warm", "end_to_end_s", "ok_only", "p50"),
                    "end_to_end_s_p95": stat("warm", "end_to_end_s", "ok_only", "p95"),
                    "first_frame_s_p50": stat("warm", "first_frame_s", "ok_only", "p50"),
                    "first_frame_s_p95": stat("warm", "first_frame_s", "ok_only", "p95"),
                    "audio_s_total": (warm.get("audio_s") or {}).get("total"),
                    "clipped_samples": warm.get("clipped_samples"),
                    "items_with_varying_audio": warm.get("items_with_varying_audio")}}
    if "cold" in phases:
        out["cold"] = {"end_to_end_s": stat("cold", "end_to_end_s", "ok_only", "p50"),
                       "realtime_factor": stat("cold", "realtime_factor", "ok_only", "p50"),
                       "first_frame_s": stat("cold", "first_frame_s", "ok_only", "p50")}
    return out


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def with_drop(memory, before):
    """`memory` with the drop in MemAvailable from `before` (the level before the load) to its lowest, and `peak_bytes`:
    the server process's own peak (nvidia-smi used_memory + RssAnon + RssShmem) when it was measured, else the larger
    of that drop and the cgroup's peak. On GB10 the GPU's memory is system memory and CUDA allocations do not show in
    the cgroup or RSS, which is why the process's nvidia-smi figure is added; co-resident models that change meanwhile
    move the drop, so it is recorded as an upper bound."""
    low = memory.get("mem_available_min_bytes")
    drop = before - low if number(before) and number(low) else None
    process = memory.get("process_peak_bytes")
    peaks = [v for v in (memory.get("cgroup_peak_max_bytes"), memory.get("cgroup_current_max_bytes"), drop)
             if number(v)]
    peak = process if number(process) else max(peaks) if peaks else None
    return {**memory, "mem_available_before_bytes": before, "mem_available_drop_bytes": drop,
            "peak_method": "process" if number(process) else "drop", "peak_bytes": peak}


def bench_result(runs, memory, resident, cold_ready_s):
    """The measurements the marker keeps, and their ratios to upstream's claims."""
    whole = (runs.get("whole") or {}).get("warm") or {}
    stream = (runs.get("stream") or {}).get("warm") or {}
    first, rtf = stream.get("first_frame_s_p50"), whole.get("rtf_p50")

    def ratio(v, claim):
        return v / claim if number(v) else None

    return {"first_audio_p50_s": first, "first_audio_p95_s": stream.get("first_frame_s_p95"), "rtf_p50": rtf,
            "rtf_p95": whole.get("rtf_p95"), "rtf_p50_stream": stream.get("rtf_p50"),
            "end_to_end_s_p50": whole.get("end_to_end_s_p50"),
            "cold_end_to_end_s": ((runs.get("whole") or {}).get("cold") or {}).get("end_to_end_s"),
            "cold_ready_s": cold_ready_s, "peak_memory_bytes": memory.get("peak_bytes"), "resident_bytes": resident,
            "ratios_to_upstream": {"first_audio": ratio(first, UPSTREAM_CLAIMS["first_audio_p50_s"]),
                                   "rtf": ratio(rtf, UPSTREAM_CLAIMS["rtf_p50"]),
                                   "rtf_after_patch_0040": ratio(rtf, UPSTREAM_CLAIMS["rtf_p50_after_patch_0040"])}}


def bench(args):
    """Upstream's own benchmark client against the server the worker started (see the module docstring)."""
    kw = worker_module()
    pins = kw.load_pins()
    cfg, tool = load_tool(args.config, args.tool)
    opts = tool.get("options") or {}
    lock = cfg.get("gpu_lock") or ""
    root = os.path.realpath(args.root)
    up = os.path.join(root, "drama-tensorfold")
    paths = tool_paths(tool)
    resident = int(tool.get("resident_bytes") or THRESHOLDS["default_resident_bytes"])
    rep = Report()
    env = {"host": os.uname().nodename, "gpu": gpu_name(), "python": platform.python_version(),
           "machine": platform.machine(), "pins_sha256": kw.sha256_file(kw.PINS), "tool": tool.get("id"),
           "options": opts, "gpu_lock": lock, "coresident": args.coresident,
           "mem_available_before_load": mem_available(), "localrouter_daemon": daemon_tools()}
    with open("/proc/loadavg") as f:
        env["loadavg"] = f.read().split()[:3]
    began = time.monotonic()
    # A timeout's SIGTERM unwinds through the finally below: the client is killed, the server stopped, the locks
    # released.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    w = held = sampler = before = None
    exited = False
    runs, memory, cold_ready_s = {}, {}, None
    try:
        corpus = os.path.join(up, CORPUS)
        digest = kw.sha256_file(corpus) if os.path.isfile(corpus) else None
        if not rep.add("corpus_sha256", digest == CORPUS_SHA256, corpus=corpus, sha256=digest, expected=CORPUS_SHA256):
            return 1
        problem = gpu_lock_problem(lock)
        if problem:
            rep.add("gpu_lock", False, error=problem)
            return 1
        held = hold_shared(lock)
        if not rep.add("host_gpu_lock", held is not None, gpu_lock=lock, mode="shared",
                       error=None if held else "a GPU job holds the host lock exclusively; nothing was loaded"):
            return 1
        private, environ = worker_environ(args.out, lock)
        env["private_lock"] = private
        before = mem_available()
        # The benchmark client asks the server for "default": --default-voice keeps that the tool's default voice.
        w = kw.Worker(dict(tool), require_qualification=False, environ=environ, server_default_voice=True)
        t = time.monotonic()
        loaded = w.load()
        if not rep.add("load", loaded.get("ok") is True, reply=loaded, ready_s=time.monotonic() - t,
                       admission=w.opts.admission, mem_available_before=before, mem_available_loaded=mem_available()):
            return 1
        unit = w.unit
        main_pid = unit_property(unit, "MainPID") if unit else ""
        pid = int(main_pid) if main_pid.isdigit() else 0
        # The unit runs env, which execs the server: its main process is the binary assets verified.
        exe = os.path.realpath(f"/proc/{pid}/exe") if pid else None
        if not rep.add("server_pid", pid > 0 and exe == os.path.realpath(paths["native_bin"]), unit=unit, pid=pid,
                       exe=exe, native_bin=paths["native_bin"]):
            return 1
        evidence = pilot_evidence(w)
        try:
            cold_ready_s = read_json(os.path.join(evidence, "ready.json")).get("cold_ready_s") if evidence else None
        except (OSError, ValueError):
            evidence = None
        if not rep.add("pilot_evidence", evidence is not None, path=evidence, cold_ready_s=cold_ready_s):
            return 1
        group = unit_property(unit, "ControlGroup")
        cgroup = "/sys/fs/cgroup" + group if group else ""
        memory_max = {"unit_memory_max": unit_property(unit, "MemoryMax"),
                      "cgroup_memory_max": read_text(os.path.join(cgroup, "memory.max")) if cgroup else None}
        if not rep.add("cgroup", bool(group) and read_int(os.path.join(cgroup, "memory.current")) is not None,
                       cgroup=cgroup, **memory_max):
            return 1
        sampler = MemorySampler(cgroup, os.path.join(args.out, "memory.jsonl"), pid)
        sampler.start()

        _, vpins = tool_variant(tool)
        if vpins:
            # The pinned upstream commit has no checksums for a variant: its pins go to the client in their format.
            checkpoint_pins = os.path.join(args.out, "checkpoint.sha256")
            with open(checkpoint_pins, "w") as f:
                f.writelines(f"{digest}  {name}\n" for name, (_, digest) in sorted(vpins["checkpoint"]["files"].items()))
            precision = f"native TensorFold {vpins['tensorfold']['commit'][:8]}, {vpins['variant']} weights"
        else:
            checkpoint_pins = os.path.join(up, "recipe", "checksums", "checkpoint.sha256")
            precision = f"native TensorFold (drama-tensorfold {pins['upstream']['commit'][:7]})"
        common = ["--endpoint", f"http://127.0.0.1:{w.opts.port}", "--model", MODEL, "--corpus", corpus,
                  "--precision", precision, "--server-pid", str(pid), "--server-source", build_dir(root, vpins),
                  "--checkpoint", paths["checkpoint"], "--checkpoint-pins", checkpoint_pins,
                  "--pilot-evidence", evidence]
        # The client's frame limit is the server's: a tool that raises max_frames raises both.
        if opts.get("max_frames") is not None and int(opts["max_frames"]) != 1024:
            common += ["--max-frames", str(int(opts["max_frames"]))]
        deadline = began + BENCH_BUDGET_S
        # As upstream's before-after.sh: whole replies on the fresh process (its first request is the cold one),
        # then streamed on the same process.
        for name, start, flags in (("whole", "fresh", ["--response-format", "wav"]),
                                   ("stream", "reused", ["--response-format", "pcm", "--stream"])):
            limit = min(BENCH_RUN_LIMIT_S, deadline - time.monotonic())
            if limit < 60:
                runs[name] = {"error": "no time left in the step"}
                rep.add(f"{name}_run", False, error=runs[name]["error"])
                continue
            argv = [sys.executable, "-I", os.path.join(up, "bench", "drama_bench.py"), "run", *common,
                    "--evidence-dir", os.path.join(args.out, name), "--label", f"localrouter-qualify-{name}",
                    "--server-start", start, *flags]
            t = time.monotonic()
            rc = run_logged(argv, os.path.join(args.out, f"{name}.log"), limit)
            runs[name] = {"exit": rc, "seconds": time.monotonic() - t, "limit_s": limit, "log": f"{name}.log",
                          **bench_summary(os.path.join(args.out, name, "summary.json"))}
            # Exit 0: every request was valid audio and the server process did not change (1: some failed; 2:
            # refused, nothing measured; None: past its limit).
            rep.add(f"{name}_run", rc == 0, **{k: runs[name].get(k) for k in (
                "exit", "seconds", "complete", "server_process_unchanged", "failures", "error")})
        memory = with_drop({**sampler.finish(), **memory_max}, before)

        r = bench_result(runs, memory, resident, cold_ready_s)
        first, cap = r["first_audio_p50_s"], THRESHOLDS["stream_first_frame_s_p50_max"]
        rep.add("first_audio_p50", number(first) and first <= cap, first_frame_s_p50=first,
                first_frame_s_p95=r["first_audio_p95_s"], max_s=cap)
        below = THRESHOLDS["warm_rtf_p50_below"]
        for name in ("whole", "stream"):
            rtf = ((runs.get(name) or {}).get("warm") or {}).get("rtf_p50")
            rep.add(f"rtf_p50_{name}", number(rtf) and rtf < below, rtf_p50=rtf, below=below)
        peak = memory.get("peak_bytes")
        rep.add("peak_memory", number(peak) and peak <= resident, peak_memory_bytes=peak, resident_bytes=resident,
                **memory_max)

        t = time.monotonic()
        w.exit()
        exited = True
        exit_s = time.monotonic() - t
        state = wait_inactive(unit, 60)
        rep.add("unload_unit_inactive", state.get("ActiveState") == "inactive", state=state, exit_s=exit_s)
        rep.add("unload_private_lock_free", lock_free(private))
        return 0
    except Exception as e:  # a crash is recorded, never silently a partial result
        rep.add("crash", False, error=f"{type(e).__name__}: {e}")
        return 1
    finally:
        if sampler is not None:
            memory = with_drop({**memory, **sampler.finish()}, before)
        if w is not None and not exited:
            try:
                w.exit()
            except Exception as e:
                rep.add("exit_after_failure", False, error=f"{type(e).__name__}: {e}")
        if held is not None:
            held.close()  # the host lock, held shared since before the load
        rc = rep.write(args.out, model=MODEL, thresholds=THRESHOLDS, upstream_claims=UPSTREAM_CLAIMS,
                       result=bench_result(runs, memory, resident, cold_ready_s), runs=runs, memory=memory, env=env,
                       upstream_commit=pins["upstream"]["commit"],
                       checkpoint_revision=(tool_variant(tool)[1] or pins)["checkpoint"]["revision"])
        if rc:
            sys.exit(rc)


# ── accuracy ───────────────────────────────────────────────────────────────────────────────────────────────────────

GUARD_LIMIT_S = 900  # each guarded run: the guard's own 600 s unit deadline, plus admission and teardown


def run_logged(argv, log_path, timeout):
    """One command with its output in log_path; its exit code, or None when it outlived timeout."""
    with open(log_path, "w") as log:
        log.write(" ".join(argv) + "\n")
        log.flush()
        try:
            return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return None


def accuracy(args):
    """Upstream's own accuracy checks against the SGLang-Omni reference, run as qualification/README.md runs them.

    backbone  qualification/backbone-reference-check.py: the 40-layer backbone prefill of upstream's 182-token
              fixture, native chunks 1/7/128 against one reference extend (hidden states and semantic logits), in
              run-bounded-oracle.py's guard (the launcher's admission, cap, abort floor and the shared GPU lock).
    rescore   qualification/rescore-native.py: the same native outputs under upstream's registered worst-row bar
              (0.8712518909203973), every other threshold unchanged; CPU.
    tower     qualification/tower-reference-check.py: the 13-layer tower, teacher-forced, at 1, 7 and 128 rows
              against the reference's unfused AudioDecoder, in the guard.
    They need RUNTIME (--reference): venv/ with the reference's dependencies and sglang-omni/ checked out at
    recipe/sources.json's reference revision, as upstream's recipe describes. Their verdicts are recorded, pass or
    fail; they do not decide the qualification (--accept-native-numerics does). This role fails only when the checks
    could not produce valid evidence.
    """
    _, tool = load_tool(args.config, args.tool)
    variant, vpins = tool_variant(tool)
    if vpins:
        report = {"run": False, "variant": variant, "applicable": False, "evidence_valid": False, "passed": False,
                  "reason": "upstream's accuracy checks compare the full checkpoint with the SGLang-Omni reference, "
                            "which does not load this lossy variant, and its engine builds no oracle tools; its "
                            "acceptance is --accept-lossy (pins: lossy), its quality compare.py's against the full model"}
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps({"run": False, "applicable": False, "variant": variant}))
        return 0
    root = os.path.realpath(args.root)
    up = os.path.join(root, "drama-tensorfold")
    q = os.path.join(up, "qualification")
    sources = read_json(os.path.join(up, "recipe", "sources.json"))
    revision = sources["reference_runtime"]["revision"]
    report = {"run": False, "reference_revision": revision, "upstream_reports_qualified": sources.get("native_drama_qualified"),
              "checks": {}, "passed": False}

    def finish(code):
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps({k: report[k] for k in ("run", "passed")} | {"error": report.get("error")}))
        return code

    if not args.reference:
        report["error"] = ("no --reference RUNTIME: upstream's accuracy checks need the SGLang-Omni reference "
                           f"(venv/ and sglang-omni/ at {revision})")
        return finish(1)
    runtime = os.path.realpath(args.reference)
    py = os.path.join(runtime, "venv", "bin", "python")
    head, why = git_head(os.path.join(runtime, "sglang-omni"))
    engine = os.path.join(root, "build", "tensorfold")
    missing = [p for p in (py, os.path.join(engine, "zig-out/oracle/bin/tf-drama-backbone-oracle"),
                           os.path.join(engine, "zig-out/oracle/bin/tf-drama-tower-oracle")) if not os.path.isfile(p)]
    if missing or head != revision:
        report["error"] = (f"the reference runtime or the oracle tools are not ready: missing {missing}, sglang-omni at "
                           f"{head or why}")
        return finish(1)
    lock = (read_json(args.config).get("gpu_lock") or "")
    checkpoint = tool_paths(tool)["checkpoint"]
    report["run"] = True
    guard = [sys.executable, "-B", os.path.join(q, "run-bounded-oracle.py"), "--gpu-lock", lock, "--runtime-dir", runtime]
    common = ["--checkpoint", checkpoint, "--reference", os.path.join(runtime, "sglang-omni"), "--engine", engine]
    backbone, tower = os.path.join(args.out, "backbone"), os.path.join(args.out, "tower")
    rescore = os.path.join(args.out, "rescore.json")

    rc = run_logged([*guard, os.path.join(args.out, "backbone-guard"), py, "-B",
                     os.path.join(q, "backbone-reference-check.py"), backbone, *common],
                    os.path.join(args.out, "backbone.log"), GUARD_LIMIT_S)
    report["checks"]["backbone"] = summarize_backbone(backbone, rc)
    if report["checks"]["backbone"].get("evidence_valid"):
        rc = run_logged([py, "-B", os.path.join(q, "rescore-native.py"), os.path.join(backbone, "native"), backbone, rescore],
                        os.path.join(args.out, "rescore.log"), 300)
        report["checks"]["rescore"] = summarize_rescore(rescore, rc)
    else:
        report["checks"]["rescore"] = {"evidence_valid": False, "passed": False, "error": "no valid backbone outputs"}
    rc = run_logged([*guard, os.path.join(args.out, "tower-guard"), py, "-B",
                     os.path.join(q, "tower-reference-check.py"), tower, *common],
                    os.path.join(args.out, "tower.log"), GUARD_LIMIT_S)
    report["checks"]["tower"] = summarize_tower(tower, rc)
    checks = report["checks"].values()
    report["evidence_valid"] = all(c.get("evidence_valid") for c in checks)
    report["passed"] = report["evidence_valid"] and all(c.get("passed") for c in checks)
    return finish(0 if report["evidence_valid"] else 1)


def summarize_backbone(out, rc):
    try:
        r = read_json(os.path.join(out, "results.json"))
    except (OSError, ValueError) as e:
        return {"evidence_valid": False, "passed": False, "exit": rc, "error": f"no results.json: {e}"}
    chunks = {}
    for name, c in (r.get("comparisons") or {}).items():
        h, s = c["hidden"], c["semantic"]
        chunks[name] = {"pass": c["pass"], "hidden_cosine": h["all_rows"]["cosine"],
                        "hidden_relative_l2": h["all_rows"]["relative_l2"], "hidden_min_row_cosine": h["rows"]["min_cosine"],
                        "hidden_last_row_cosine": h["last_row"]["cosine"], "semantic_top1_equal": s["top1_equal"],
                        "semantic_top30_overlap": s["top30_overlap"], "semantic_kl": s["kl_reference_to_native"]}
    return {"evidence_valid": r.get("evidence_valid") is True, "passed": r.get("passed") is True, "exit": rc,
            "thresholds": r.get("thresholds"), "chunks": chunks, "errors": r.get("errors", [])}


def summarize_rescore(path, rc):
    try:
        r = read_json(path)
    except (OSError, ValueError) as e:
        return {"evidence_valid": False, "passed": False, "exit": rc, "error": f"no rescore: {e}"}
    return {"evidence_valid": True, "passed": r.get("passed") is True, "exit": rc,
            "worst_row_bar": r["thresholds_revised"]["hidden_min_row_cosine"],
            "chunks": {k: {"hidden_min_row_cosine": c["hidden_min_row_cosine"], "failing_gates": c["failing_gates"]}
                       for k, c in r.get("chunks", {}).items()}}


def summarize_tower(out, rc):
    try:
        r = read_json(os.path.join(out, "results.json"))
    except (OSError, ValueError) as e:
        return {"evidence_valid": False, "passed": False, "exit": rc, "error": f"no results.json: {e}"}
    cases = []
    for c in r.get("cases", []):
        h, head = c.get("hidden") or {}, c.get("head9") or {}
        cases.append({"rows": c.get("rows"), "pass": c.get("pass"), "hidden_cosine": h.get("cosine"),
                      "hidden_relative_l2": h.get("relative_l2"), "head9_top1_equal": head.get("top1_equal"),
                      "head9_top30_overlap": head.get("top30_overlap"), "head9_kl": head.get("kl_reference_to_native")})
    return {"evidence_valid": bool(cases), "passed": r.get("passed") is True, "exit": rc,
            "thresholds": r.get("thresholds"), "cases": cases,
            "torch_version": r.get("torch_version"), "sglang_version": r.get("sglang_version")}


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
    variant, vpins = tool_variant(tool)
    if vpins and not args.accept_lossy:
        print(f"qualify.py: the {variant} weights are lossy ({vpins['lossy']['what']}): --write needs --accept-lossy",
              file=sys.stderr)
        return 1
    reports = {}
    for step in ("assets", "e2e", "bench"):
        try:
            reports[step] = read_json(os.path.join(args.out, step, "report.json"))
        except (OSError, ValueError):
            reports[step] = {}
        if reports[step].get("pass") is not True:
            print(f"qualify.py: {args.out}/{step}/report.json does not pass; no marker", file=sys.stderr)
            return 1
    measured_bench = reports["bench"].get("result") or {}
    try:
        acc = read_json(os.path.join(args.out, "accuracy", "report.json"))
    except (OSError, ValueError):
        acc = {}
    if not vpins and not (acc.get("run") and acc.get("evidence_valid")):
        # Their verdict does not decide acceptance, but a qualification records it: no valid run, no marker.
        print(f"qualify.py: upstream's accuracy checks did not run with valid evidence ({acc.get('error')}); no marker",
              file=sys.stderr)
        return 1
    engine = read_json(engine_record(os.path.realpath(args.root), vpins))
    native = tool_paths(tool)["native_bin"]
    digest = kw.sha256_file(native)
    if digest != engine.get("native_sha256") or engine.get("patched_head") != (vpins or pins)["engine_patches"]["head"]:
        print(f"qualify.py: {native} sha256 {digest} is not engine.json's {engine.get('native_sha256')}; no marker",
              file=sys.stderr)
        return 1
    gpu = gpu_name()
    if not gpu:
        print("qualify.py: nvidia-smi names no GPU; no marker", file=sys.stderr)
        return 1
    record = {"qualified": True, "model": MODEL, "host": os.uname().nodename, "gpu": gpu,
              "pins_sha256": kw.sha256_file(kw.PINS), "native_sha256": digest,
              "upstream_commit": pins["upstream"]["commit"],
              "checkpoint_revision": (vpins or pins)["checkpoint"]["revision"], "numerics": NUMERICS,
              "upstream_accuracy": {k: acc.get(k) for k in ("reference_revision", "passed", "upstream_reports_qualified",
                                                             "checks")},
              "bench": {**{k: measured_bench.get(k) for k in ("first_audio_p50_s", "rtf_p50", "peak_memory_bytes",
                                                              "resident_bytes", "ratios_to_upstream")},
                        "coresident": (reports["bench"].get("env") or {}).get("coresident"),
                        "upstream_claims": reports["bench"].get("upstream_claims")},
              "report": args.out,
              **({"variant": variant, "variant_pins_sha256": kw.sha256_file(kw.variant_pins_path(variant)),
                  "checkpoint": vpins["checkpoint"]["repo"], "tensorfold_commit": vpins["tensorfold"]["commit"],
                  "lossy": {**vpins["lossy"], "accepted": "--accept-lossy"}} if vpins else {}),
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
    ap.add_argument("role", choices=["assets", "e2e", "bench", "accuracy", "marker"])
    ap.add_argument("root")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--tool", default=MODEL)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--revoke", action="store_true")
    ap.add_argument("--accept-native-numerics", action="store_true")
    ap.add_argument("--accept-lossy", action="store_true",
                    help="marker: the operator accepts a lossy variant's weights (pins-VARIANT.json's lossy)")
    ap.add_argument("--reference", help="accuracy: the SGLang-Omni reference runtime (venv/ and sglang-omni/)")
    ap.add_argument("--coresident", action="store_true",
                    help="bench: chat and speech-to-text are loaded by a running LocalRouter daemon (recorded)")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    return {"assets": assets, "e2e": e2e, "bench": bench, "accuracy": accuracy, "marker": marker}[args.role](args) or 0


if __name__ == "__main__":
    sys.exit(main())
