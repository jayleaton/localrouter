"""LocalRouter `cmd` worker for Whisper large-v3-turbo speech to text on Apple Silicon (MLX), the macOS counterpart of
../worker.py with the same protocol and transcript (docs/ARCHITECTURE.md, "the cmd adapter").

One JSON object a line on stdin and stdout; stdout carries protocol lines only (anything else that writes to fd 1 is
redirected to stderr), logs go to stderr.
  {"op":"load"}                                   -> {"ok":true,"resident":<bytes>,"cancel":"signal"}
  {"op":"generate","id","dir","request":{"transcription":{...}}}
                     -> {"progress":{...}}* then {"ok":true,"files":["transcript.json"],"seed","ms"} or {"ok":false,...}
  {"op":"exit"} or EOF                            -> release and exit 0
SIGUSR1 during a generate (from the command's arrival to its reply) ends it with {"ok":false,"type":"cancelled"} and
the model stays loaded (the daemon's soft cancel, src/tool/process.zig); with no generate in hand it is ignored. SIGKILL (the daemon's abort) is always safe.
The tool's config arrives as JSON in LOCALROUTER_TOOL: `weights` is the MLX model directory (config.json and
weights.safetensors, fetch_model.py), `options` holds dtype (auto|float16|float32; auto is float16), threads (torch CPU
threads for the gate), max_new_tokens, and the voice activity gate: vad ("silero", the default, or "off") and
vad_model (the pinned silero_vad.jit; required unless vad is "off", so the gate is never skipped silently).
`resident` is the process's physical footprint (what Activity Monitor and `footprint` call memory: on unified memory
it includes the GPU buffers MLX holds), measured after the warm-up decode.
A generate command with "debug": true also writes whisper_debug.json; {"op":"stats"} (check.py and bench.py only)
replies with the process's memory now and at peak."""
import ctypes
import gc
import json
import os
import select
import signal
import sys
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROTO = os.fdopen(os.dup(1), "w", buffering=1)  # the protocol stream; fd 1 itself now points at stderr
os.dup2(2, 1)
sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mlx_core  # noqa: E402

PARENT = int(os.environ.get("LOCALROUTER_PARENT") or 0)


class Cancelled(Exception):
    """SIGUSR1 arrived while a generate ran."""


def log(msg):
    print(f"whisper-mlx-worker: {msg}", file=sys.stderr, flush=True)


def send(obj):
    PROTO.write(json.dumps(obj, separators=(",", ":")) + "\n")
    PROTO.flush()


def orphaned():
    # 0: no check; 1: the daemon is a container's init, so the parent pid cannot tell (src/engine/worker.zig).
    return PARENT not in (0, 1) and os.getppid() != PARENT


def watch_parent():
    """macOS has no PR_SET_PDEATHSIG: a thread ends the process within a second of the daemon going away, also in the
    middle of a decode (read_line covers only the idle wait)."""
    def run():
        while not orphaned():
            time.sleep(1.0)
        log("daemon gone, exiting")
        os._exit(0)
    if PARENT not in (0, 1):
        threading.Thread(target=run, name="parent", daemon=True).start()


_pending = bytearray()


def generate_waiting():
    """A generate command the daemon has sent and this worker has not started: buffered, or unread on stdin."""
    if b'"generate"' in _pending:
        return True
    try:
        return bool(select.select([0], [], [], 0)[0])
    except (OSError, ValueError):
        return False


def read_line():
    """The next stdin line, or None on EOF or when the daemon is gone (checked at least once a second)."""
    while True:
        nl = _pending.find(b"\n")
        if nl >= 0:
            line = bytes(_pending[:nl])
            del _pending[:nl + 1]
            return line
        if orphaned():
            log("daemon gone, exiting")
            return None
        try:
            ready, _, _ = select.select([0], [], [], 1.0)
        except InterruptedError:
            continue
        if ready:
            chunk = os.read(0, 1 << 16)
            if not chunk:
                return bytes(_pending) if _pending.strip() else None
            _pending.extend(chunk)


class _RusageInfoV4(ctypes.Structure):
    # <sys/resource.h> struct rusage_info_v4: a 16-byte uuid, then 35 uint64 fields (padded generously).
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("v", ctypes.c_uint64 * 64)]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True) if sys.platform == "darwin" else None
_PHYS_FOOTPRINT, _LIFETIME_MAX_PHYS_FOOTPRINT, _RESIDENT = 7, 28, 6  # indices into v (after the uuid)


def footprint():
    """(physical footprint now, its lifetime peak, resident size) of this process in bytes, from proc_pid_rusage."""
    if _libc is None:
        return 0, 0, 0
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:  # RUSAGE_INFO_V4
        return 0, 0, 0
    return int(info.v[_PHYS_FOOTPRINT]), int(info.v[_LIFETIME_MAX_PHYS_FOOTPRINT]), int(info.v[_RESIDENT])


class Worker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = cfg.get("options") or {}
        self.model = self.vad = self.dtype = None
        self.weights = ""
        self.mode = None  # the gate of the loaded state: "silero" or "off"; None while not loaded
        self.generating = False  # from a generate command until its reply is out: a SIGUSR1 then is for this job
        self.decoding = False  # a decode runs: SIGUSR1 cancels it at once
        self.sending = False  # ...but not halfway through a protocol line: then it is held until the line is out
        self.cancel_held = False  # a SIGUSR1 for this job that came before its decode (or mid-line): honoured next

    def ready(self):
        """Loaded, with the gate the options select: the only state generate runs in."""
        return self.model is not None and self.mode is not None and (self.mode == "off" or self.vad is not None)

    def resident(self):
        return footprint()[0]

    def load(self):
        """All or nothing: the gate is checked first (mode, then the pinned VAD), the model loaded into locals, and the
        state published only when every part loaded. Any failure leaves the worker unloaded with its memory released,
        so a retry loads again from scratch and generate never runs without the selected gate."""
        if self.ready():
            return {"ok": True, "resident": self.resident(), "cancel": "signal"}
        self.release()
        model = vad = None
        try:
            mode = self.opts.get("vad", "silero")
            if mode not in ("silero", "off"):
                raise RuntimeError(f"options.vad must be silero or off, got {mode!r}")
            if mode == "silero":
                vad = mlx_core.load_vad(self.opts.get("vad_model") or "")
            weights = self.cfg.get("weights") or ""
            dtype = mlx_core.resolve_dtype(self.opts.get("dtype", "auto"))
            t = time.monotonic()
            model = mlx_core.load(weights, dtype)
            loaded = time.monotonic() - t
            # One throwaway decode of a second of silence, with word timestamps and the gate: the first request would
            # otherwise pay the Metal kernels' compilation, numba's compilation of the DTW and TorchScript's first
            # optimizing runs of Silero (2.4 s against 0.5 s for a 6 s file with word timestamps on an M5 Ultra).
            t = time.monotonic()
            mlx_core.transcribe(model, weights, dtype, np.zeros(mlx_core.SAMPLE_RATE, dtype=np.float32),
                                {"language": "en", "timestamps": "word"}, vad=vad)
            mx.clear_cache()
        except BaseException:
            del model, vad
            self.release()
            raise
        self.model, self.vad, self.mode, self.weights, self.dtype = model, vad, mode, weights, dtype
        log(f"loaded {weights} {dtype} on {mx.default_device()} threads={torch.get_num_threads()} vad={mode} "
            f"in {loaded:.1f}s, warmed in {time.monotonic() - t:.1f}s")
        return {"ok": True, "resident": self.resident(), "cancel": "signal"}

    def generate(self, cmd):
        t = time.monotonic()
        self.generating = True
        req = (cmd.get("request") or {}).get("transcription")
        if not isinstance(req, dict):
            raise mlx_core.InvalidRequest("this tool serves transcription requests only")
        job_dir = cmd.get("dir") or ""
        if not os.path.isdir(job_dir):
            raise mlx_core.InvalidRequest("job directory does not exist")
        audio = req.get("audio") or ""
        if not audio or os.path.basename(audio) != audio or audio.startswith("."):
            raise mlx_core.InvalidRequest("audio must name a file in the job directory")
        samples = mlx_core.read_wav(os.path.join(job_dir, audio))  # a bad file is the caller's, loaded or not
        if not self.ready():
            self.load()  # raises (and stays unloaded) unless the model and its selected gate both load
        if not self.ready() or (self.mode == "silero") != (self.vad is not None):
            raise RuntimeError("the worker is not loaded with its voice activity gate")
        opts = {k: req.get(k) for k in ("language", "prompt", "temperature", "timestamps", "seed", "decode")}
        opts["max_new_tokens"] = self.opts.get("max_new_tokens")

        def progress(step, of):
            self.sending = True
            send({"progress": {"phase": "transcribe", "step": step, "of": of}})
            self.sending = False
            if self.cancel_held:
                raise Cancelled()

        self.decoding = True
        try:
            if self.cancel_held:  # cancelled while the audio was read or the model loaded
                raise Cancelled()
            result = mlx_core.transcribe(self.model, self.weights, self.dtype, samples, opts, progress=progress,
                                         debug=bool(cmd.get("debug")), vad=self.vad if self.mode == "silero" else None)
        finally:
            self.decoding = self.sending = self.cancel_held = False
            mx.clear_cache()  # the decode's buffers go back to the system: what stays is the booked resident model
        files = ["transcript.json"]
        dbg = result.pop("_debug", None)
        if dbg is not None:
            write_json(os.path.join(job_dir, "whisper_debug.json"), dbg)
            files.append("whisper_debug.json")
        write_json(os.path.join(job_dir, "transcript.json"), result)
        return {"ok": True, "files": files, "seed": int(opts.get("seed") or 0),
                "ms": int((time.monotonic() - t) * 1000)}

    def stats(self):
        """Memory now and at peak (check.py and bench.py only; the daemon never sends "stats")."""
        now, peak, rss = footprint()
        return {"ok": True, "stats": {"footprint": now, "footprint_peak": peak, "rss": rss,
                                      "mlx_active": mx.get_active_memory(), "mlx_cache": mx.get_cache_memory(),
                                      "mlx_peak": mx.get_peak_memory()}}

    def release(self):
        self.model = self.vad = self.mode = self.dtype = None
        self.weights = ""
        mlx_core.release()
        gc.collect()
        mx.clear_cache()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    if orphaned():
        return 0
    watch_parent()
    try:
        cfg = json.loads(os.environ.get("LOCALROUTER_TOOL") or "{}")
    except json.JSONDecodeError as e:
        log(f"LOCALROUTER_TOOL is not JSON: {e}")
        return 2
    threads = (cfg.get("options") or {}).get("threads")
    if threads:
        torch.set_num_threads(int(threads))
    w = Worker(cfg)

    def on_cancel(signum, frame):
        # The daemon signals only a job it has sent: one being decoded is cancelled at once; one whose generate is
        # still being read, checked or loaded for, or is waiting on stdin, at the start of its decode. A signal with
        # no generate in hand (it raced a reply already sent) is ignored, so it can never cancel the next job.
        if w.decoding and not w.sending:
            raise Cancelled()
        if w.decoding or w.generating or generate_waiting():
            w.cancel_held = True
    signal.signal(signal.SIGUSR1, on_cancel)

    while True:
        line = read_line()
        if line is None:
            break
        try:
            cmd = json.loads(line)
            op = cmd.get("op")
        except (json.JSONDecodeError, AttributeError) as e:
            send({"ok": False, "error": f"bad command line: {e}", "type": "protocol_error"})
            continue
        if op == "exit":
            break
        try:
            if op == "load":
                send(w.load())
            elif op == "generate":
                send(w.generate(cmd))
            elif op == "stats":
                send(w.stats())
            else:
                send({"ok": False, "error": f"unknown op {op!r}", "type": "protocol_error"})
        except Cancelled:
            log(f"{op} cancelled")
            send({"ok": False, "error": "cancelled", "type": "cancelled"})
        except mlx_core.InvalidRequest as e:
            send({"ok": False, "error": str(e), "type": "invalid_request_error"})
        except Exception as e:  # keep serving: one failed job is not a dead worker
            log(f"{op} failed: {type(e).__name__}: {e}")
            send({"ok": False, "error": f"{op} failed: {type(e).__name__}: {e}", "type": "server_error"})
        finally:
            w.generating = w.cancel_held = False  # the reply is out: a later signal is not for this job
    w.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
