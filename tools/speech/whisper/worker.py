"""LocalRouter `cmd` worker for Whisper large-v3-turbo speech to text (docs/ARCHITECTURE.md, "the cmd adapter").

One JSON object a line on stdin and stdout; stdout carries protocol lines only (anything else that writes to fd 1 is
redirected to stderr), logs go to stderr.
  {"op":"load"}                                   -> {"ok":true,"resident":<bytes>}
  {"op":"generate","id","dir","request":{"transcription":{...}}}
                     -> {"progress":{...}}* then {"ok":true,"files":["transcript.json"],"seed","ms"} or {"ok":false,...}
  {"op":"exit"} or EOF                            -> release and exit 0
The tool's config arrives as JSON in LOCALROUTER_TOOL: `weights` is the model directory, `options` holds device
(auto|cpu|cuda), dtype (auto|float16|bfloat16|float32), attn (sdpa|eager), threads, max_new_tokens, and the voice
activity gate (whisper_core): vad ("silero", the default, or "off") and vad_model (the pinned silero_vad.jit;
required unless vad is "off", so the gate is never skipped silently).
A generate command with "debug": true (twin.py only) also writes whisper_debug.json; {"op":"stats"} (bench.py only)
replies with the process's memory now and at peak."""
import ctypes
import gc
import json
import os
import select
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch: deterministic cuBLAS
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROTO = os.fdopen(os.dup(1), "w", buffering=1)  # the protocol stream; fd 1 itself now points at stderr
os.dup2(2, 1)
sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402  (after CUBLAS_WORKSPACE_CONFIG and the stdout redirect)

import whisper_core  # noqa: E402

PARENT = int(os.environ.get("LOCALROUTER_PARENT") or 0)


def log(msg):
    print(f"whisper-worker: {msg}", file=sys.stderr, flush=True)


def send(obj):
    PROTO.write(json.dumps(obj, separators=(",", ":")) + "\n")
    PROTO.flush()


def orphaned():
    # 0: no check; 1: the daemon is a container's init, so the parent pid cannot tell (src/engine/worker.zig).
    return PARENT not in (0, 1) and os.getppid() != PARENT


def die_with_parent():
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


_pending = bytearray()


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
        ready, _, _ = select.select([0], [], [], 1.0)
        if ready:
            chunk = os.read(0, 1 << 16)
            if not chunk:
                return bytes(_pending) if _pending.strip() else None
            _pending.extend(chunk)


def proc_status(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    return 0


class Worker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = cfg.get("options") or {}
        self.model = self.processor = self.vad = None
        self.mode = None  # the gate of the loaded state: "silero" or "off"; None while not loaded

    def ready(self):
        """Loaded, with the gate the options select: the only state generate runs in."""
        return self.model is not None and self.mode is not None and (self.mode == "off" or self.vad is not None)

    def resident(self):
        n = proc_status("VmRSS")
        if self.model is not None and self.model.device.type == "cuda":
            n += torch.cuda.memory_reserved()
        return n

    def load(self):
        """All or nothing: the gate is checked first (mode, then the pinned VAD), the model loaded into locals, and the
        state published only when every part loaded. Any failure leaves the worker unloaded with its memory released,
        so a retry loads again from scratch and generate never runs without the selected gate."""
        if self.ready():
            return {"ok": True, "resident": self.resident()}
        self.release()
        model = processor = vad = None
        try:
            mode = self.opts.get("vad", "silero")
            if mode not in ("silero", "off"):
                raise RuntimeError(f"options.vad must be silero or off, got {mode!r}")
            if mode == "silero":
                vad = whisper_core.load_vad(self.opts.get("vad_model") or "")
            weights = self.cfg.get("weights") or ""
            if not os.path.isfile(os.path.join(weights, "model.safetensors")):
                raise RuntimeError(f"no model.safetensors in weights dir {weights!r}")
            device, dtype = whisper_core.resolve_device_dtype(self.opts.get("device", "auto"),
                                                              self.opts.get("dtype", "auto"))
            t = time.monotonic()
            model, processor = whisper_core.load(weights, device, dtype, self.opts.get("attn", "sdpa"))
            loaded = time.monotonic() - t
            # One throwaway decode of a second of silence, so the first request does not pay the first-call CUDA
            # setup (on GB10 a voice turn's transcript took 1.4 to 2.0 s right after a reload, 0.4 s warm).
            t = time.monotonic()
            whisper_core.transcribe(model, processor, np.zeros(whisper_core.SAMPLE_RATE, dtype=np.float32), {"language": "en"})
        except BaseException:
            del model, processor, vad
            self.release()
            raise
        self.model, self.processor, self.vad, self.mode = model, processor, vad, mode
        log(f"loaded {weights} on {device} {dtype} attn={model.config._attn_implementation} "
            f"threads={torch.get_num_threads()} vad={mode} in {loaded:.1f}s, warmed in {time.monotonic() - t:.1f}s")
        return {"ok": True, "resident": self.resident()}

    def generate(self, cmd):
        t = time.monotonic()
        req = (cmd.get("request") or {}).get("transcription")
        if not isinstance(req, dict):
            raise whisper_core.InvalidRequest("this tool serves transcription requests only")
        job_dir = cmd.get("dir") or ""
        if not os.path.isdir(job_dir):
            raise whisper_core.InvalidRequest("job directory does not exist")
        audio = req.get("audio") or ""
        if not audio or os.path.basename(audio) != audio or audio.startswith("."):
            raise whisper_core.InvalidRequest("audio must name a file in the job directory")
        samples = whisper_core.read_wav(os.path.join(job_dir, audio))  # a bad file is the caller's, loaded or not
        if not self.ready():
            self.load()  # raises (and stays unloaded) unless the model and its selected gate both load
        if not self.ready() or (self.mode == "silero") != (self.vad is not None):
            raise RuntimeError("the worker is not loaded with its voice activity gate")
        opts = {k: req.get(k) for k in ("language", "prompt", "temperature", "timestamps", "seed", "decode")}
        opts["max_new_tokens"] = self.opts.get("max_new_tokens")

        def progress(step, of):
            send({"progress": {"phase": "transcribe", "step": step, "of": of}})

        result = whisper_core.transcribe(self.model, self.processor, samples, opts, progress=progress,
                                         debug=bool(cmd.get("debug")), vad=self.vad if self.mode == "silero" else None)
        files = ["transcript.json"]
        dbg = result.pop("_debug", None)
        if dbg is not None:
            write_json(os.path.join(job_dir, "whisper_debug.json"), dbg)
            files.append("whisper_debug.json")
        write_json(os.path.join(job_dir, "transcript.json"), result)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        return {"ok": True, "files": files, "seed": int(opts.get("seed") or 0),
                "ms": int((time.monotonic() - t) * 1000)}

    def stats(self):
        """Memory now and at peak (bench.py only; the daemon never sends "stats")."""
        st = {"rss": proc_status("VmRSS"), "hwm": proc_status("VmHWM")}
        if torch.cuda.is_available():  # loaded or not: a failed load must leave nothing on the GPU either
            st.update(cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                      cuda_max_allocated=torch.cuda.max_memory_allocated(),
                      cuda_max_reserved=torch.cuda.max_memory_reserved())
        return {"ok": True, "stats": st}

    def release(self):
        self.model = self.processor = self.vad = self.mode = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    die_with_parent()
    if orphaned():
        return 0
    try:
        cfg = json.loads(os.environ.get("LOCALROUTER_TOOL") or "{}")
    except json.JSONDecodeError as e:
        log(f"LOCALROUTER_TOOL is not JSON: {e}")
        return 2
    threads = (cfg.get("options") or {}).get("threads")
    if threads:
        torch.set_num_threads(int(threads))
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    w = Worker(cfg)
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
        except whisper_core.InvalidRequest as e:
            send({"ok": False, "error": str(e), "type": "invalid_request_error"})
        except Exception as e:  # keep serving: one failed job is not a dead worker
            log(f"{op} failed: {type(e).__name__}: {e}")
            send({"ok": False, "error": f"{op} failed: {type(e).__name__}: {e}", "type": "server_error"})
    w.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
