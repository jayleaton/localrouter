#!/usr/bin/env python3
"""LocalRouter `cmd` worker for any OpenAI-compatible audio server: text to speech through `POST /v1/audio/speech`,
speech to text through `POST /v1/audio/transcriptions` (docs/ARCHITECTURE.md, "the cmd adapter" and "Speech tools").

Standard library only, so it runs under the system python3 with no venv. One JSON object a line on stdin and stdout;
stdout carries protocol lines only, logs go to stderr.
  {"op":"load"}      -> starts options.server (if any) and waits for options.health -> {"ok":true,"resident":0}
  {"op":"generate","id","dir","request":{"speech":{...}}}        -> speech.wav (RIFF PCM16 mono; streamed if asked)
  {"op":"generate","id","dir","request":{"transcription":{...}}} -> transcript.json
  {"op":"exit"} or EOF -> stops the server it started, then exits 0

The tool's config arrives as JSON in LOCALROUTER_TOOL. Options:
  server        argv of the model's own server ({port}, {id}, {weights} substituted); absent: use `url` as is
  env           extra environment for the server
  port          fixed loopback port for the server (absent: a free one)
  url           base URL of an already running server (only when `server` is absent), e.g. http://127.0.0.1:8091
  health        path that answers 200 when the server is ready (default /v1/models)
  model         the model id to send (default: the tool's id)
  sample_rate   PCM rate of the server's "pcm" output (default 24000; an x-sample-rate header wins)
  stream        true: speech is written to speech.wav as it arrives (the daemon forwards it as HTTP chunks)
  send_seed     true (default): pass the request's seed through (folded to 32 bits)
  extra         object merged into every speech request body (e.g. {"language": "English"})
  transcription_format  json (default) or verbose_json: what a request without timestamps asks the server for
  timestamps    the timestamp_granularities the server returns: [] (default), ["segment"], ["segment", "word"]. A
                request for a granularity listed here asks for verbose_json with it; "word" not listed is refused
  stop_timeout_s  seconds between SIGTERM and SIGKILL when stopping the server (default 15)
Memory is the config's resident_bytes and working_bytes: measured figures, not guessed here.
"""
import ctypes
import json
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

PROTO = os.fdopen(os.dup(1), "w", buffering=1)  # the protocol stream; fd 1 itself now points at stderr
os.dup2(2, 1)
sys.stdout = sys.stderr
PARENT = int(os.environ.get("LOCALROUTER_PARENT") or 0)


class InvalidRequest(Exception):
    pass


class Unavailable(Exception):
    pass


def log(msg):
    print(f"openai-audio-worker: {msg}", file=sys.stderr, flush=True)


def send(obj):
    PROTO.write(json.dumps(obj, separators=(",", ":")) + "\n")
    PROTO.flush()


def orphaned():
    return PARENT not in (0, 1) and os.getppid() != PARENT


def die_with_parent():
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG: stop the server first
    except (OSError, AttributeError):
        pass


_pending = bytearray()


def read_line():
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


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wav_header(rate, data_bytes):
    return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", data_bytes))


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


class Worker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = cfg.get("options") or {}
        self.proc = None
        self.base = None
        self.model = self.opts.get("model") or cfg.get("id") or ""
        self.timestamps = [g for g in (self.opts.get("timestamps") or []) if g in ("segment", "word")]

    # ---- the server
    def healthy(self, timeout=2.0):
        try:
            with urllib.request.urlopen(self.base + self.opts.get("health", "/v1/models"), timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def load(self):
        if self.base and (self.proc is None or self.proc.poll() is None) and self.healthy():
            return {"ok": True, "resident": 0}
        self.stop()
        server = self.opts.get("server")
        if server:
            port = int(self.opts.get("port") or 0) or free_port()
            sub = lambda v: (v.replace("{port}", str(port)).replace("{id}", self.cfg.get("id") or "")
                             .replace("{weights}", self.cfg.get("weights") or ""))
            argv = [sub(str(a)) for a in server]
            env = dict(os.environ)
            env.update({k: sub(str(v)) for k, v in (self.opts.get("env") or {}).items()})
            log(f"starting {argv[0]} on 127.0.0.1:{port}")
            self.proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
                                         start_new_session=True)
            self.base = f"http://127.0.0.1:{port}"
        else:
            self.base = (self.opts.get("url") or "").rstrip("/")
            if not self.base:
                raise RuntimeError("options.server or options.url is required")
        deadline = time.monotonic() + float(self.cfg.get("load_timeout_s") or 300)
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                code = self.proc.returncode
                self.proc = None
                raise RuntimeError(f"the model's server exited before it was ready (status {code}); see the tool's log")
            if self.healthy():
                log(f"server ready at {self.base}")
                return {"ok": True, "resident": 0}
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"the model's server did not answer {self.opts.get('health', '/v1/models')} in time")

    def stop(self):
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + float(self.opts.get("stop_timeout_s") or 15)
        while p.poll() is None and time.monotonic() < deadline:
            send({"progress": {"phase": "unload", "step": 0, "of": 1}})  # the daemon waits while memory is still held
            time.sleep(0.5)
        if p.poll() is None:
            log("server ignored SIGTERM; killing it")
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()

    def ensure(self):
        if self.base is None or (self.proc is not None and self.proc.poll() is not None):
            self.load()

    def post(self, path, body, ctype, timeout):
        req = urllib.request.Request(self.base + path, body, {"Content-Type": ctype})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            msg = e.read()[:500].decode("utf-8", "replace")
            if 400 <= e.code < 500:
                raise InvalidRequest(f"the model's server refused the request ({e.code}): {msg}")
            if e.code == 503:
                raise Unavailable(f"the model's server is busy ({e.code}): {msg}")
            raise RuntimeError(f"the model's server failed ({e.code}): {msg}")
        except (urllib.error.URLError, ConnectionError) as e:
            raise Unavailable(f"the model's server is not reachable: {e}")

    # ---- jobs
    def generate(self, cmd):
        t = time.monotonic()
        req = cmd.get("request") or {}
        job_dir = cmd.get("dir") or ""
        if not os.path.isdir(job_dir):
            raise InvalidRequest("job directory does not exist")
        self.ensure()
        timeout = float(self.cfg.get("generate_timeout_s") or 600)
        if isinstance(req.get("speech"), dict):
            out = self.speech(req["speech"], job_dir, timeout)
        elif isinstance(req.get("transcription"), dict):
            out = self.transcription(req["transcription"], job_dir, timeout)
        else:
            raise InvalidRequest("this tool serves speech and transcription requests only")
        out["ms"] = int((time.monotonic() - t) * 1000)
        return out

    def speech(self, s, job_dir, timeout):
        if s.get("voice_references") or s.get("references") or s.get("clone"):
            raise InvalidRequest("this tool has no voice cloning or references")
        text = s.get("input") or ""
        if not text.strip():
            raise InvalidRequest("the input has nothing to speak")
        body = dict(self.opts.get("extra") or {})
        body.update({"model": self.model, "input": text, "response_format": "pcm"})
        if s.get("voice"):
            body["voice"] = s["voice"]
        if s.get("speed") not in (None, 0, 1, 1.0):
            body["speed"] = s["speed"]
        seed = int(s.get("seed") or 0)
        if self.opts.get("send_seed", True) and seed:
            body["seed"] = seed & 0xFFFFFFFF  # the daemon's seeds are 64-bit; servers commonly take a uint32
        r = self.post("/v1/audio/speech", json.dumps(body).encode(), "application/json", timeout)
        rate = int(r.headers.get("x-sample-rate") or self.opts.get("sample_rate") or 24000)
        path = os.path.join(job_dir, "speech.wav")
        n, odd = 0, b""
        with open(path, "wb") as f:
            f.write(wav_header(rate, 0))  # placeholder sizes; the real ones are written at the end
            f.flush()
            while True:
                chunk = r.read1(1 << 15) if hasattr(r, "read1") else r.read(1 << 15)
                if not chunk:
                    break
                chunk = odd + chunk
                cut = len(chunk) & ~1  # whole 16-bit samples only
                chunk, odd = chunk[:cut], chunk[cut:]
                f.write(chunk)
                n += len(chunk)
                if s.get("stream"):
                    f.flush()
            if n == 0:
                raise RuntimeError("the model's server returned no audio")
            f.seek(0)
            f.write(wav_header(rate, n))
        return {"ok": True, "files": ["speech.wav"], "seed": seed}

    def transcription_fields(self, q):
        """The form fields of one transcription request: the request's options forwarded as asked, never replaced.
        temperature goes through as given (a server that cannot honour it refuses, and that refusal is the caller's
        400). timestamps: "none" asks for options.transcription_format; "segment" or "word" asks for verbose_json with
        that timestamp_granularities[] when options.timestamps lists it. "word" without it is refused here, before the
        server is asked. "segment" without it gets the whole clip as one segment (verbose_json, srt and vtt carry
        segments, so the daemon asks every such request for them)."""
        want = q.get("timestamps") or "none"
        if want not in ("none", "segment", "word"):
            raise InvalidRequest(f"timestamps must be none, segment or word, not {want!r}")
        if want == "word" and "word" not in self.timestamps:
            raise InvalidRequest("this tool has no word timestamps (its options.timestamps does not list word)")
        try:
            temperature = float(q.get("temperature") or 0)
        except (TypeError, ValueError):
            raise InvalidRequest("temperature must be a number")
        fields = [("model", self.model), ("temperature", f"{temperature:g}")]
        if want in self.timestamps:
            fields.append(("response_format", "verbose_json"))
            grans = ["segment", "word"] if want == "word" and "segment" in self.timestamps else [want]
            fields += [("timestamp_granularities[]", g) for g in grans]
        else:
            fields.append(("response_format", self.opts.get("transcription_format", "json")))
        for k in ("language", "prompt"):
            if q.get(k):
                fields.append((k, str(q[k])))
        return fields

    def transcription(self, q, job_dir, timeout):
        audio = q.get("audio") or ""
        if not audio or os.path.basename(audio) != audio or audio.startswith("."):
            raise InvalidRequest("audio must name a file in the job directory")
        fields = self.transcription_fields(q)
        b = uuid.uuid4().hex
        body = bytearray()
        for k, v in fields:
            body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        with open(os.path.join(job_dir, audio), "rb") as f:
            data = f.read()
        body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{audio}"\r\n'
                 f'Content-Type: audio/wav\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
        with self.post("/v1/audio/transcriptions", bytes(body), f"multipart/form-data; boundary={b}", timeout) as r:
            res = json.loads(r.read())
        want = q.get("timestamps") or "none"
        if want in self.timestamps:
            key = "words" if want == "word" else "segments"
            if not res.get(key):
                raise RuntimeError(f"the model's server returned no {key}, though the tool's options.timestamps lists {want}")
        duration = float(q.get("duration_ms") or 0) / 1000 or float(res.get("duration") or 0)
        text = (res.get("text") or "").strip()
        segments = res.get("segments") or [{"id": 0, "start": 0.0, "end": round(duration, 3), "text": text}]
        out = {"text": text, "language": res.get("language") or q.get("language") or "", "duration": duration,
               "segments": segments}
        if res.get("words"):
            out["words"] = res["words"]
        write_json(os.path.join(job_dir, "transcript.json"), out)
        return {"ok": True, "files": ["transcript.json"], "seed": int(q.get("seed") or 0)}


def main():
    die_with_parent()
    if orphaned():
        return 0
    try:
        cfg = json.loads(os.environ.get("LOCALROUTER_TOOL") or "{}")
    except json.JSONDecodeError as e:
        log(f"LOCALROUTER_TOOL is not JSON: {e}")
        return 2
    w = Worker(cfg)
    signal.signal(signal.SIGTERM, lambda *_: (w.stop(), sys.exit(0)))
    try:
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
                else:
                    send({"ok": False, "error": f"unknown op {op!r}", "type": "protocol_error"})
            except InvalidRequest as e:
                send({"ok": False, "error": str(e), "type": "invalid_request_error"})
            except Unavailable as e:
                send({"ok": False, "error": str(e), "type": "service_unavailable"})
            except Exception as e:  # keep serving: one failed job is not a dead worker
                log(f"{op} failed: {type(e).__name__}: {e}")
                send({"ok": False, "error": f"{op} failed: {type(e).__name__}: {e}", "type": "server_error"})
    finally:
        w.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
