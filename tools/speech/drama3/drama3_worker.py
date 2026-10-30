#!/usr/bin/env python3
"""LocalRouter worker for Fish Audio Drama 3: text to 44.1 kHz speech, whole or streamed. Built with Fish Audio.

Speaks the worker protocol (docs/ARCHITECTURE.md): one JSON object a line on stdin and stdout, logs on stderr.

    {"op":"load"}                                   -> {"ok":true,"resident":0,"cancel":"signal"}
    SIGUSR1 during a generate                       -> {"ok":false,"type":"cancelled"} (the server connection closed, still loaded)
    {"op":"generate","id","dir","request":{"speech":{"input":"Hello.","voice":"default","stream":false,
        "references":"","voice_references":""}}}    -> {"ok":true,"files":["speech.wav"],"seed":0,"ms":…,
                                                        "finish_reason":"stop","segments":""}
    {"op":"generate","id","dir","request":{"speech":{"input":"","clone":"clone.json"}}}
                                                    -> {"ok":true,"files":["voice.json"],"seed":0,"ms":…}
    {"op":"exit"}

`references` and `voice_references` name JSON files in the job directory, each a list of {text, vq_codes} (the latter
may be a clone's whole voice.json; only text and vq_codes are sent). With
`voice_references` (a cloned voice's, voice.json's `references`) the server gets the request's references and then
those, and no `voice`: the built-in voice with exactly these references; `voice` then names the clone for the logs only.
Otherwise a `voice` of "" or "default" is sent as the tool's `default_voice` (when that is not "default"): the worker,
not the server's --default-voice, names it, so a clone's request is not given the default voice's references too, and
every other request builds the same prompt rows, so the same audio, as before.

`clone` (clone.json: {"name", "references": [{"audio": "ref0.wav", "text": the clip's exact words}]}, 1 to 8 clips
the daemon has trimmed and levelled, 16-bit mono 44.1 kHz, at most 30 s each) encodes a voice: each clip through the
engine binary's `voice encode` (its codec encoder on the GPU, BF16, the running server untouched; ENCODE_TIMEOUT_S a
clip), into voice.json: {"format": "localrouter-voice-codes/1", "references": [{"text", "vq_codes" (10 x frames),
"frames", "audio_sha256"}], "encoder": the first clip's, "native_sha256": the engine binary's}. A clip or transcript the
encoder refuses is invalid_request_error ("reference N: …", N from 0); any other failure server_error; SIGUSR1 kills
the encode and the reply is cancelled. The tool's `clone_working_bytes` is the daemon's booking for it.

The model is plotarmordev/drama-tensorfold's native server (TensorFold's Zig/CUDA engine with the Drama 3 patch
series), built from the pins in pins.json and started by that repository's own launcher, recipe/pilot.py, with
LocalRouter's pinned admission patch (patches/pilot-external-admission.patch; fetch.sh writes ROOT/launcher/pilot.py).
LocalRouter's scheduler is the only admission authority: it admits the load from the tool's resident_bytes (the
measured peak and a small margin, docs/dev/DRAMA3.md) against its budget, so the worker passes --external-admission and the launcher skips its
own 88 GiB MemAvailable admission and 34 GiB abort. The launcher still runs the server as a transient systemd user
unit capped at 54 GiB on 127.0.0.1, takes the lock it is given (LOCALROUTER_GPU_LOCK: the private, uncontended
per-tool lock file the daemon names; a lock under the runtime directory when run outside the daemon), and stops the
unit and releases the lock when it is told to stop. This worker is a client of that server:

- load: the qualification marker (written by qualify.sh on this machine), the pinned files, then the launcher; ready
  when the launcher prints DRAMA_PILOT_READY. The configured voices must be ones the server loaded.
- generate: POST /v1/audio/speech. A whole reply is asked for as WAV and written as speech.wav. A streamed one
  (`stream`) is asked for as raw PCM and written to speech.wav as it arrives: a header with placeholder sizes first,
  then each chunk, then the real sizes once the server has sent its last chunk, so the daemon can send the samples
  on while they come. `X-Finish-Reason` (stop, length, silence) and `X-Long-Form-Segments`, from the head of a whole
  reply or the trailers of a stream, come back as `finish_reason` and `segments`.
- Errors: the server's 400 is invalid_request_error (the daemon's 400), its 503 service_unavailable (503), anything
  else server_error (500). A server that is gone ends this worker without a reply, so the daemon fails the job with
  503 and the next request starts the model again.
- exit: SIGTERM to the launcher, which stops the unit and then releases the lock. The launcher runs in its own
  session with a parent-death signal, so a worker killed by the daemon (a cancel, a deadline) still stops it.

The option `variant` picks the weights: `full` (fishaudio/drama-3, pins.json; the default) or `nvfp4`
(plotarmordev/drama-3-NVFP4-Pruned, pins-nvfp4.json: its own checkpoint directory, engine build and serve flags,
and a marker of its own, which must name the variant and its pins).

Drama 3 seeds each synthesis from its prompt, so `seed` is not sent and the reply's seed is 0. `speed`, `preset`
and `instructions` are refused by the daemon before a request gets here (the tool's options declare none).
"""

import collections
import ctypes
import hashlib
import json
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

OUTPUT = "speech.wav"
HEADER = 44
PINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pins.json")
MODEL = "fishaudio/drama-3"  # the name the launcher serves the model under
UNIT_FILE = "localrouter-drama3.unit"  # in the runtime directory: the launcher's unit while this worker runs it
READY = "DRAMA_PILOT_READY"
BUSY_WAIT_S = 10  # how long a request waits while the server drops a cancelled synthesis (`Worker._post`)
VOICE_OUTPUT = "voice.json"  # a clone's output (`Worker._clone`)
VOICE_FORMAT = "localrouter-voice-codes/1"
MAX_CLONE_REFERENCES = 8
ENCODE_TIMEOUT_S = 120  # one reference's `tensorfold-native voice encode`
CODEBOOKS = 10
# `voice encode`'s own words (exit 1, "tensorfold: …") when the clip or its transcript is what it refuses: the clip's
# WAV (families/drama/clip.zig) or a transcript its voice file loader would not take (voice_cli.voiceFile).
ENCODE_REFUSALS = ("the voice file would not load", "reference audio lasts", "the clip ", "the WAV ",
                   "unsupported WAV", "malformed WAVE", "resampling ")
PR_SET_PDEATHSIG = 1
# resolved before any fork, so the child only calls it; Linux only (drama3_worker_macos.py imports this module)
_prctl = ctypes.CDLL(None, use_errno=True).prctl if sys.platform.startswith("linux") else None


class Cancelled(BaseException):
    """SIGUSR1 during a generate: the daemon cancels it (barge-in). Not an Exception, so no handler turns it into a
    server error; the reply is {"ok": false, "type": "cancelled"} and the model stays loaded."""


GENERATING = False


def on_cancel(signum, frame):
    if GENERATING:
        raise Cancelled()  # interrupts the blocking read of the stream (PEP 475); _speak closes the connection


class Refused(Exception):
    """The server refused the request: the API's 400."""


class Unavailable(Exception):
    """The server is busy or not ready: the API's 503."""


class Gone(Exception):
    """The server or its launcher is gone: the worker exits."""


def log(msg):
    print(f"drama3_worker: {msg}", file=sys.stderr, flush=True)


def load_pins():
    with open(PINS, encoding="utf-8") as f:
        return json.load(f)


VARIANTS = ("full", "nvfp4")


def variant_pins_path(variant):
    """pins-VARIANT.json beside pins.json: another checkpoint's pins (pins.json holds the full model's)."""
    return os.path.join(os.path.dirname(PINS), f"pins-{variant}.json")


def load_variant_pins(variant):
    """The variant's own pins, None for the full model."""
    if variant == "full":
        return None
    with open(variant_pins_path(variant), encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wav_header(rate, data_bytes):
    """A canonical 44-byte RIFF header for mono 16-bit PCM."""
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", (36 + data_bytes) & 0xFFFFFFFF, b"WAVE", b"fmt ", 16, 1, 1,
                       rate, rate * 2, 2, 16, b"data", data_bytes & 0xFFFFFFFF)


def canonical_wav(data):
    """The sample rate of a canonical mono 16-bit PCM WAV whose sizes match its length, else None."""
    if len(data) < HEADER:
        return None
    riff, size, wave, fmt, fmt_len, tag, channels, rate, _, _, bits, chunk, data_len = struct.unpack(
        "<4sI4s4sIHHIIHH4sI", data[:HEADER])
    if (riff, wave, fmt, chunk) != (b"RIFF", b"WAVE", b"fmt ", b"data") or (fmt_len, tag, channels, bits) != (16, 1, 1, 16):
        return None
    if data_len != len(data) - HEADER or size != 36 + data_len or rate == 0:
        return None
    return rate


class Options:
    """The tool's config (LOCALROUTER_TOOL): `weights` is the root fetch.sh and build.sh filled; `options` the rest."""

    def __init__(self, tool):
        o = tool.get("options") or {}
        known = {"voices", "default_voice", "max_input_chars", "min_speed", "max_speed", "presets", "stream",
                 "references", "port", "max_frames", "long_form", "long_form_chars", "long_form_history",
                 "voices_dir", "verify", "pilot", "native_bin", "checkpoint", "runtime_dir", "ready_timeout_s",
                 "admission", "variant", "guidance", "clone_working_bytes"}
        unknown = sorted(set(o) - known)
        if unknown:
            log(f"options not used by the worker: {unknown}")
        root = tool.get("weights") or ""
        if not root:
            raise ValueError("the tool's weights must be the Drama 3 root that fetch.sh and build.sh filled")
        self.root = root
        self.pilot = o.get("pilot") or os.path.join(root, "launcher", "pilot.py")
        # external: LocalRouter admits the load (the default); launcher: upstream's own admission and abort, for a
        # launch outside any scheduler.
        self.admission = o.get("admission", "external")
        if self.admission not in ("external", "launcher"):
            raise ValueError("admission must be external or launcher")
        # full: fishaudio/drama-3 (pins.json); nvfp4: plotarmordev/drama-3-NVFP4-Pruned with its own engine build and
        # serve flags (pins-nvfp4.json), which fetch.sh and build.sh --variant nvfp4 put beside the full model's.
        self.variant = o.get("variant", "full")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {', '.join(VARIANTS)}")
        self.variant_pins = load_variant_pins(self.variant)
        v = self.variant_pins
        build = v["build"]["dir"] if v else os.path.join("build", "tensorfold")
        self.native_bin = o.get("native_bin") or os.path.join(root, build, "zig-out", "native", "bin", "tensorfold-native")
        self.checkpoint = o.get("checkpoint") or os.path.join(root, v["checkpoint"]["dir"] if v else "drama-3")
        # The variant's serve flags (nvfp4: classifier-free guidance at γ 1.5, batched, as it was measured upstream);
        # false serves it without them: a second backbone sequence less a frame, unguided.
        self.guidance = o.get("guidance", True)
        if not isinstance(self.guidance, bool):
            raise ValueError("guidance must be true or false")
        self.runtime_dir = o.get("runtime_dir") or os.path.join(root, "runtime")
        self.port = int(o.get("port", 18793))
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be 1 to 65535")
        self.max_frames = o.get("max_frames")
        self.long_form = bool(o.get("long_form", False))
        self.long_form_chars = o.get("long_form_chars")
        self.long_form_history = o.get("long_form_history")
        if not self.long_form and (self.long_form_chars is not None or self.long_form_history is not None):
            raise ValueError("long_form_chars and long_form_history need long_form true")
        self.voices_dir = o.get("voices_dir") or ""
        if self.voices_dir and not os.path.isabs(self.voices_dir):
            raise ValueError("voices_dir must be an absolute path (the server resolves a relative one elsewhere)")
        self.voices = o.get("voices") or ["default"]
        self.default_voice = o.get("default_voice") or "default"
        self.verify = o.get("verify", "size")
        if self.verify not in ("size", "sha256", "none"):
            raise ValueError("verify must be size, sha256 or none")
        self.ready_timeout_s = float(o.get("ready_timeout_s", 900))
        for flag in ("min_speed", "max_speed"):
            if o.get(flag, 1) != 1:
                raise ValueError("Drama 3 has no speed control: min_speed and max_speed must be 1")
        if o.get("presets"):
            raise ValueError("Drama 3 has no presets: its style is written in the input ([sighs], [laughs])")

    def serve_args(self, default_voice=False):
        """The server flags after the launcher's `--`: the variant's own first (nvfp4: the guidance it was measured
        with). The worker names the default voice itself (`Worker._speak`), so the server keeps its built-in one for a
        cloned voice's references; `default_voice` adds --default-voice for a client that talks to the server
        directly (qualify.py's benchmark, whose requests say "default")."""
        args = list(self.variant_pins["serve"]) if self.variant_pins and self.guidance else []
        if self.max_frames is not None:
            args += ["--max-frames", str(int(self.max_frames))]
        if self.long_form:
            args.append("--long-form")
            if self.long_form_chars is not None:
                args += ["--long-form-chars", str(int(self.long_form_chars))]
            if self.long_form_history is not None:
                args += ["--long-form-history", str(int(self.long_form_history))]
        if self.voices_dir:
            args += ["--voices", self.voices_dir]
        if default_voice and self.default_voice != "default":
            args += ["--default-voice", self.default_voice]
        return args


class Http:
    """One HTTP/1.1 exchange with the server on loopback, by hand: http.client drops a stream's trailers."""

    def __init__(self, port, method, path, body=None, timeout=3600):
        try:
            self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
            head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n"
            if body is not None:
                head += f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            self.sock.sendall(head.encode() + b"\r\n" + (body or b""))
            self.r = self.sock.makefile("rb")
            line = self.r.readline()
        except OSError as e:
            raise Gone(f"the Drama 3 server cannot be reached: {e}") from e
        parts = line.split(None, 2)
        if len(parts) < 2 or not parts[0].startswith(b"HTTP/1."):
            raise Gone("the Drama 3 server closed the connection without a reply")
        self.status = int(parts[1])
        self.headers = self._fields()
        self.trailers = {}

    def _fields(self):
        fields = {}
        while True:
            line = self.r.readline()
            if line in (b"\r\n", b"\n", b""):
                return fields
            name, _, value = line.decode("latin-1").partition(":")
            fields[name.strip().lower()] = value.strip()

    def chunks(self):
        """The body's pieces as they arrive; a chunked body that ends before its last chunk raises EOFError."""
        if self.headers.get("transfer-encoding", "").lower() == "chunked":
            while True:
                line = self.r.readline()
                if not line.endswith(b"\n"):
                    raise EOFError("the stream ended before its last chunk")
                size = int(line.split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.trailers = self._fields()
                    return
                data = self.r.read(size)
                if len(data) < size or self.r.read(2) != b"\r\n":
                    raise EOFError("the stream ended inside a chunk")
                yield data
        elif "content-length" in self.headers:
            want = int(self.headers["content-length"])
            data = self.r.read(want)
            if len(data) < want:
                raise EOFError(f"the reply ended after {len(data)} of {want} bytes")
            yield data
        else:
            yield self.r.read()

    def body(self):
        return b"".join(self.chunks())

    def close(self):
        try:
            self.r.close()
            self.sock.close()
        except OSError:
            pass

    def error_message(self):
        text = self.body().decode("utf-8", "replace")
        try:
            return json.loads(text)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            return text.strip()[:500] or f"HTTP {self.status}"


class Worker:
    """The Drama 3 server behind one LocalRouter tool. Every method returns a protocol reply."""

    def __init__(self, tool, require_qualification=True, environ=None, server_default_voice=False):
        self.tool = tool
        self.require_qualification = require_qualification
        self.server_default_voice = server_default_voice  # --default-voice too (`Options.serve_args`)
        self.environ = os.environ if environ is None else environ
        self.opts = Options(tool)
        self.pilot = None
        self.unit = None
        self.evidence = None  # the launcher's evidence directory (session.json, ready.json, server.log)
        self.lines = queue.Queue()
        self.tail = collections.deque(maxlen=8)

    # ------------------------------------------------------------ load

    def load(self):
        try:
            self._load()
            # The launcher's caps bound it: the config's estimate stands. SIGUSR1 ends a generate early (`generate`).
            return {"ok": True, "resident": 0, "cancel": "signal"}
        except Exception as e:  # noqa: BLE001 - every failure is the reply's reason
            self._stop()
            why = str(e) or type(e).__name__
            if self.tail:
                why += "; launcher: " + " | ".join(self.tail)
            log(f"load failed: {why}")
            return {"ok": False, "error": why, "type": "server_error"}

    def _load(self):
        if self.require_qualification:
            self._check_qualification()
        self._check_files()
        os.makedirs(self.opts.runtime_dir, exist_ok=True)
        # The launcher insists on a lock. The daemon names this tool's own (uncontended) one; without it, one of ours.
        lock = self.environ.get("LOCALROUTER_GPU_LOCK", "") or os.path.join(self.opts.runtime_dir, "drama3.lock")
        if not os.path.isfile(lock):
            open(lock, "a").close()
        self._stop_stale_unit()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        evidence = self.evidence = os.path.join(self.opts.root, "evidence", f"{stamp}-{os.getpid()}")
        os.makedirs(os.path.dirname(evidence), exist_ok=True)
        argv = [sys.executable, "-I", self.opts.pilot, "--native-bin", self.opts.native_bin,
                "--runtime-dir", self.opts.runtime_dir, "--checkpoint", self.opts.checkpoint, "--gpu-lock", lock,
                "--evidence-dir", evidence, "--port", str(self.opts.port),
                "--ready-timeout", str(self.opts.ready_timeout_s)]
        if self.opts.admission == "external":
            argv.append("--external-admission")
        serve = self.opts.serve_args(self.server_default_voice)
        if serve:
            argv += ["--", *serve]
        log(f"starting the launcher: {' '.join(argv)}")
        parent = os.getpid()

        def detach():
            _prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
            if os.getppid() != parent:  # the worker died before the signal was armed
                os.kill(os.getpid(), signal.SIGTERM)

        self.pilot = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      start_new_session=True, preexec_fn=detach)
        readers = [threading.Thread(target=self._read, args=(pipe, protocol), daemon=True)
                   for pipe, protocol in ((self.pilot.stdout, True), (self.pilot.stderr, False))]
        for t in readers:
            t.start()
        deadline = time.monotonic() + self.opts.ready_timeout_s + 30
        while True:
            try:
                line = self.lines.get(timeout=1)
            except queue.Empty:
                line = None
            if line == READY:
                break
            if line and line.startswith("{"):
                try:
                    self.unit = json.loads(line).get("unit") or self.unit
                except ValueError:
                    pass
                if self.unit:
                    with open(os.path.join(self.opts.runtime_dir, UNIT_FILE), "w", encoding="utf-8") as f:
                        f.write(self.unit + "\n")
            if line is None and self.pilot.poll() is not None:
                for t in readers:  # its last words (a refusal) are in the pipes until they are read
                    t.join(timeout=5)
                raise RuntimeError(f"the launcher exited ({self.pilot.returncode}) before the server was ready")
            if time.monotonic() > deadline:
                raise RuntimeError("the server was not ready in time")
        self._check_voices()
        log(f"ready on 127.0.0.1:{self.opts.port} (unit {self.unit})")

    def _read(self, pipe, protocol):
        """Copies one of the launcher's pipes to the log; stdout lines also go to the load's queue."""
        for raw in iter(pipe.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\n")
            log(f"launcher: {line}")
            if protocol:
                self.lines.put(line)
            else:
                self.tail.append(line)

    def _check_qualification(self):
        path = self.tool.get("qualification") or ""
        if not path:
            raise ValueError("Drama 3 runs only where it is GPU-qualified: set the tool's qualification to the file "
                             "tools/speech/drama3/qualify.sh writes")
        try:
            with open(path, encoding="utf-8") as f:
                marker = json.load(f)
        except (OSError, ValueError) as e:
            raise ValueError(f"not GPU-qualified on this host: {path} cannot be read ({e})") from e
        host = os.uname().nodename
        if marker.get("qualified") is not True:
            raise ValueError(f"not GPU-qualified on this host: {path} does not say \"qualified\": true")
        if marker.get("host") != host:
            raise ValueError(f"not GPU-qualified on this host: {path} qualified {marker.get('host')!r}, this is {host!r}")
        if marker.get("pins_sha256") != sha256_file(PINS):
            raise ValueError(f"not GPU-qualified on this host: {path} qualified other pins; run qualify.sh again")
        variant = self.opts.variant
        if marker.get("variant", "full") != variant:
            raise ValueError(f"not GPU-qualified on this host: {path} qualified the {marker.get('variant', 'full')} "
                             f"weights, the tool serves the {variant} weights")
        if variant != "full" and marker.get("variant_pins_sha256") != sha256_file(variant_pins_path(variant)):
            raise ValueError(f"not GPU-qualified on this host: {path} qualified other {variant} pins; run qualify.sh again")
        if not os.path.isfile(self.opts.native_bin) or marker.get("native_sha256") != sha256_file(self.opts.native_bin):
            raise ValueError(f"not GPU-qualified on this host: {path} qualified another engine build; run qualify.sh again")

    def _check_files(self):
        o = self.opts
        for path, what in ((o.pilot, "the launcher (fetch.sh)"), (o.native_bin, "the engine (build.sh)")):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} is missing: {what}")
        if not os.access(o.native_bin, os.X_OK):
            raise PermissionError(f"{o.native_bin} is not executable")
        if o.verify == "none":
            return
        pins = load_pins()
        launcher = pins["launcher"]["sha256"]
        if sha256_file(o.pilot) != launcher:
            raise ValueError(f"{o.pilot} is not the pinned launcher (sha256 {launcher}): run fetch.sh")
        checkpoint = (o.variant_pins or pins)["checkpoint"]
        for name, (size, digest) in sorted(checkpoint["files"].items()):
            path = os.path.join(o.checkpoint, name)
            if not os.path.isfile(path) or os.path.getsize(path) != size:
                raise ValueError(f"{path} is missing or not {size} bytes: run fetch.sh")
            if o.verify == "sha256" and sha256_file(path) != digest:
                raise ValueError(f"{path} sha256 differs from the pin {digest}")

    def _check_voices(self):
        # the default voice too: the worker names it in every request that names none, so it must be loaded
        wanted = [v for v in dict.fromkeys(self.opts.voices + [self.opts.default_voice]) if v != "default"]
        if not wanted:
            return
        r = Http(self.opts.port, "GET", "/v1/voices", timeout=30)
        try:
            names = {v.get("name") for v in json.loads(r.body()).get("data", [])}
        finally:
            r.close()
        missing = [v for v in wanted if v not in names]
        if missing:
            raise ValueError(f"the config lists voices the server did not load from voices_dir: {missing}")

    # ------------------------------------------------------------ generate

    def generate(self, cmd):
        began = time.monotonic()
        try:
            r = cmd["request"]["speech"]
            if r.get("clone"):
                self._clone(cmd["dir"], r["clone"])
                return {"ok": True, "files": [VOICE_OUTPUT], "seed": 0, "ms": int((time.monotonic() - began) * 1000)}
            finish, segments = self._speak(cmd["dir"], r)
        except Refused as e:
            return {"ok": False, "error": str(e), "type": "invalid_request_error"}
        except Unavailable as e:
            return {"ok": False, "error": str(e), "type": "service_unavailable"}
        except Gone as e:
            return {"ok": False, "error": str(e), "type": "server_error", "gone": True}
        except Exception as e:  # noqa: BLE001
            if self.pilot is not None and self.pilot.poll() is not None:
                return {"ok": False, "error": f"{e}; the launcher exited", "type": "server_error", "gone": True}
            return {"ok": False, "error": str(e) or type(e).__name__, "type": "server_error"}
        return {"ok": True, "files": [OUTPUT], "seed": 0, "ms": int((time.monotonic() - began) * 1000),
                "finish_reason": finish, "segments": segments}

    def _speak(self, job_dir, r):
        if self.pilot is None or self.pilot.poll() is not None:
            raise Gone("the Drama 3 server is not running")
        stream = bool(r.get("stream"))
        body = {"model": MODEL, "input": r["input"], "response_format": "pcm" if stream else "wav"}
        references = None
        if r.get("references"):
            with open(os.path.join(job_dir, r["references"]), encoding="utf-8") as f:
                references = json.load(f)
        if r.get("voice_references"):
            # A cloned voice: its references after the request's own, on the built-in voice (no `voice`, so the
            # server adds none); the request's voice names it only in the daemon's logs.
            # The list, or the voice.json a clone wrote; only text and vq_codes go on (the server refuses other fields).
            with open(os.path.join(job_dir, r["voice_references"]), encoding="utf-8") as f:
                cloned = json.load(f)
            if isinstance(cloned, dict) and cloned.get("format") == VOICE_FORMAT:
                cloned = cloned.get("references")
            if not isinstance([] if references is None else references, list) or not isinstance(cloned, list) \
                    or not cloned or not all(isinstance(c, dict) for c in cloned):
                raise Refused("references and voice_references must each be a list of {text, vq_codes}")
            cloned = [{"text": c.get("text"), "vq_codes": c.get("vq_codes")} for c in cloned]
            references = ([] if references is None else references) + cloned
        else:
            # "" and "default" are the operator's default voice, as the server's --default-voice made them: the same
            # voice file, so the same prompt rows and the same audio.
            voice = r.get("voice") or ""
            if voice in ("", "default") and self.opts.default_voice != "default":
                voice = self.opts.default_voice
            if voice:
                body["voice"] = voice
        if references is not None:
            body["references"] = references
        if stream:
            body["stream"] = True
        res = self._post(json.dumps(body).encode())
        try:
            if res.status != 200:
                why = res.error_message()
                if res.status in (400, 413, 422):
                    raise Refused(why)
                if res.status == 503:
                    raise Unavailable(f"the Drama 3 server is busy: {why}")
                raise RuntimeError(f"the Drama 3 server failed ({res.status}): {why}")
            out = os.path.join(job_dir, OUTPUT)
            if stream:
                self._stream_to(res, out)
                fields = res.trailers
            else:
                data = res.body()
                if canonical_wav(data) is None:
                    raise RuntimeError("the Drama 3 server sent audio that is not mono 16-bit PCM WAV")
                with open(out, "wb") as f:
                    f.write(data)
                fields = res.headers
            return fields.get("x-finish-reason", "stop"), fields.get("x-long-form-segments", "")
        finally:
            res.close()

    def _post(self, body):
        """POST /v1/audio/speech. The server takes one synthesis at a time and the daemon sends one request at a time,
        so a 503 "busy" means a synthesis the worker cancelled is still being dropped: retried every 50 ms for up to
        BUSY_WAIT_S, then returned as it is."""
        deadline = time.monotonic() + BUSY_WAIT_S
        while True:
            res = Http(self.opts.port, "POST", "/v1/audio/speech", body)
            if res.status != 503 or time.monotonic() >= deadline:
                return res
            res.body()
            res.close()
            time.sleep(0.05)

    @staticmethod
    def _stream_to(res, out):
        """Raw PCM chunks into speech.wav as they arrive; the header's sizes are fixed after the last chunk."""
        rate = int(res.headers.get("x-sample-rate", "0"))
        if rate <= 0 or res.headers.get("x-channels", "1") != "1":
            raise RuntimeError("the Drama 3 stream does not say it is mono PCM at a sample rate")
        n = 0
        with open(out, "wb") as f:
            f.write(wav_header(rate, 0xFFFFFFFF - 36))  # placeholder sizes until the audio is complete
            f.flush()
            try:
                for data in res.chunks():
                    f.write(data)
                    f.flush()
                    n += len(data)
            except (EOFError, OSError) as e:
                raise RuntimeError(f"the Drama 3 stream is incomplete after {n} bytes: {e}") from e
            if n % 2:
                raise RuntimeError("the Drama 3 stream ended inside a sample")
            os.pwrite(f.fileno(), wav_header(rate, n), 0)

    # ------------------------------------------------------------ clone

    def _clone(self, job_dir, name):
        """A voice from clips: each of clone.json's references through `voice encode` (the engine binary's codec
        encoder, on the GPU beside the running server, which it does not touch), then voice.json, the references a
        speech request sends as `voice_references`. The per-reference files are removed whatever the outcome."""
        try:
            with open(os.path.join(job_dir, name), encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, ValueError) as e:
            raise Refused(f"clone: {name} cannot be read ({e})") from e
        refs = spec.get("references") if isinstance(spec, dict) else None
        if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_CLONE_REFERENCES:
            raise Refused(f"clone: references must be a list of 1 to {MAX_CLONE_REFERENCES} {{audio, text}}")
        for i, ref in enumerate(refs):
            audio = ref.get("audio") if isinstance(ref, dict) else None
            text = ref.get("text") if isinstance(ref, dict) else None
            if not isinstance(audio, str) or os.path.basename(audio) != audio or audio.startswith("."):
                raise Refused(f"reference {i}: audio must name a file in the job directory")
            if not os.path.isfile(os.path.join(job_dir, audio)):
                raise Refused(f"reference {i}: {audio} is missing")
            if not isinstance(text, str) or not text.strip():
                raise Refused(f"reference {i}: text must be the clip's exact words")
        log(f"cloning voice {spec.get('name')!r} from {len(refs)} reference(s)")
        native_sha256 = sha256_file(self.opts.native_bin)
        outs = [os.path.join(job_dir, f"ref{i}.codes.json") for i in range(len(refs))]
        try:
            encoded = [self._encode(job_dir, i, ref, out) for i, (ref, out) in enumerate(zip(refs, outs))]
        finally:
            for out in outs:
                try:
                    os.unlink(out)
                except OSError:
                    pass
        voice = {"format": VOICE_FORMAT,
                 "references": [{k: e[k] for k in ("text", "vq_codes", "frames", "audio_sha256")} for e in encoded],
                 "encoder": encoded[0]["encoder"], "native_sha256": native_sha256}
        out = os.path.join(job_dir, VOICE_OUTPUT)
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            json.dump(voice, f, ensure_ascii=False)
            f.write("\n")
        os.replace(out + ".tmp", out)

    def _encode(self, job_dir, i, ref, out):
        """One reference through `voice encode`, its output checked: a [10, frames] grid of codes for this clip. Its
        output is logged, never passed on (stdout is the protocol). A cancel kills it and goes on up as Cancelled."""
        clip = os.path.join(job_dir, ref["audio"])
        stem = os.path.basename(out)[:-len(".json")]  # --out must be NAME.json
        argv = [self.opts.native_bin, "voice", "encode", clip, "--model", self.opts.checkpoint, "--text", ref["text"],
                "--name", stem, "--out", out]
        try:
            os.unlink(out)  # nothing stale is read back as this clip's codes
        except OSError:
            pass
        began = time.monotonic()
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             preexec_fn=lambda: _prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0))
        try:
            stdout, stderr = p.communicate(timeout=ENCODE_TIMEOUT_S)
        except BaseException as e:  # a timeout or a cancel (SIGUSR1): the encode ends with it
            p.kill()
            p.wait()
            if isinstance(e, subprocess.TimeoutExpired):
                raise RuntimeError(f"reference {i}: the encoder did not finish within {ENCODE_TIMEOUT_S} s") from e
            log(f"reference {i}: encode killed ({type(e).__name__})")
            raise
        lines = [ln for ln in (stdout + stderr).decode("utf-8", "replace").splitlines() if ln.strip()]
        for ln in lines:
            log(f"encode ref{i}: {ln}")
        if p.returncode != 0:
            said = [ln for ln in stderr.decode("utf-8", "replace").splitlines() if ln.strip()]
            why = said[-1] if said else f"exit {p.returncode}"
            for prefix in ("tensorfold: ", "tensorfold-native voice: error: "):
                if why.startswith(prefix):
                    why = why[len(prefix):]
            refused = (p.returncode == 1 and any(why.startswith(m) for m in ENCODE_REFUSALS)) or \
                      (p.returncode == 2 and why == "--text is empty")
            if refused:
                raise Refused(f"reference {i}: {why}")
            raise RuntimeError(f"reference {i}: the encoder failed (exit {p.returncode}): {why}")
        try:
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"reference {i}: the encoder wrote no readable {os.path.basename(out)} ({e})") from e
        frames, codes = doc.get("frames"), doc.get("vq_codes")
        if not (isinstance(frames, int) and frames > 0 and isinstance(codes, list) and len(codes) == CODEBOOKS
                and all(isinstance(row, list) and len(row) == frames
                        and all(type(c) is int and 0 <= c < 4096 for c in row) for row in codes)):
            raise RuntimeError(f"reference {i}: the encoder's vq_codes are not a [{CODEBOOKS}, frames] grid of codes")
        if not isinstance(doc.get("text"), str) or not isinstance(doc.get("encoder"), dict):
            raise RuntimeError(f"reference {i}: the encoder's file has no text or encoder")
        if doc.get("audio_sha256") != sha256_file(clip):
            raise RuntimeError(f"reference {i}: the encoder's audio_sha256 is not this clip's")
        log(f"reference {i}: {frames} frames in {time.monotonic() - began:.2f} s")
        return doc

    # ------------------------------------------------------------ exit

    def exit(self):
        self._stop()
        return {"ok": True}

    def _stop(self):
        """Asks the launcher to stop the unit; it releases the GPU lock once the unit is down."""
        p, self.pilot = self.pilot, None
        if p is None:
            return
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                log("the launcher did not stop within 120 s; killing it and stopping its unit")
                p.kill()
                p.wait()
                self._stop_unit(self.unit)
        log(f"launcher exited ({p.returncode})")
        try:
            os.unlink(os.path.join(self.opts.runtime_dir, UNIT_FILE))
        except OSError:
            pass

    def _stop_stale_unit(self):
        """A unit left by a launcher that was killed outright (its name is in the runtime directory) is stopped."""
        path = os.path.join(self.opts.runtime_dir, UNIT_FILE)
        try:
            with open(path, encoding="utf-8") as f:
                unit = f.read().strip()
        except OSError:
            return
        if unit.startswith("drama3-pilot-"):
            log(f"stopping {unit}, left by an earlier run")
            self._stop_unit(unit)
        os.unlink(path)

    @staticmethod
    def _stop_unit(unit):
        if not unit:
            return
        try:
            subprocess.run(["systemctl", "--user", "stop", unit], check=False, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as e:
            log(f"could not stop {unit}: {e}")


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def watch_parent(worker):
    """Exit when the daemon that started us is gone (LOCALROUTER_PARENT; 0 or unset: no check)."""
    parent = int(os.environ.get("LOCALROUTER_PARENT", "0") or 0)
    if parent <= 0:
        return

    def run():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                log("the daemon is gone; stopping")
                worker._stop()
                os._exit(0)

    threading.Thread(target=run, daemon=True).start()


def generate(worker, cmd):
    """A generate that SIGUSR1 can end early: the connection to the server is closed, so the server cancels the
    synthesis, and the model stays loaded. Without a signal it is exactly Worker.generate."""
    global GENERATING
    try:
        try:
            GENERATING = True  # inside the try: a signal right after it is still caught
            return worker.generate(cmd)
        finally:
            GENERATING = False
    except Cancelled:
        log("generate cancelled; the server drops the synthesis (the next request waits while it is busy)")
        return {"ok": False, "type": "cancelled", "error": "cancelled"}


def main():
    signal.signal(signal.SIGUSR1, on_cancel)
    try:
        worker = Worker(json.loads(os.environ.get("LOCALROUTER_TOOL", "{}")))
    except (ValueError, TypeError) as e:
        worker, problem = None, str(e)
    if worker is not None:
        watch_parent(worker)
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd.get("op")
        if worker is None:
            if op == "exit":
                return
            send({"ok": False, "error": f"bad tool config: {problem}", "type": "server_error"})
            continue
        if op == "load":
            send(worker.load())
        elif op == "generate":
            reply = generate(worker, cmd)
            if reply.pop("gone", False):
                log(f"{reply['error']}: exiting so the daemon restarts the model")
                worker._stop()
                os._exit(1)
            send(reply)
        elif op == "exit":
            # The daemon waits while lines arrive: it loads nothing else until the unit is down and the lock free.
            stopping = threading.Thread(target=worker.exit)
            stopping.start()
            while stopping.is_alive():
                stopping.join(timeout=5)
                if stopping.is_alive():
                    send({"progress": {"phase": "unload"}})
            return
    if worker is not None:
        worker._stop()


if __name__ == "__main__":
    main()
