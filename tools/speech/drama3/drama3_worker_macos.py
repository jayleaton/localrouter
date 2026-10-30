#!/usr/bin/env python3
"""LocalRouter worker for Fish Audio Drama 3 on macOS (Apple silicon, Metal). Built with Fish Audio.

The same worker protocol, replies and streaming as drama3_worker.py (whose Http, Options and Worker this extends; the
GB10 path is unchanged), against plotarmordev/drama-tensorfold's Mac path: TensorFold's native Zig/Metal engine at its
Mac NVFP4 pin and patches, serving plotarmordev/drama-3-NVFP4-Pruned, all from pins-macos.json (setup-macos.sh). What
differs from the GB10:

- Lifecycle. No systemd, cgroup or prctl: the launcher is upstream's recipe/pilot_macos.py with LocalRouter's pinned
  patch (patches/pilot-macos-external-admission.patch). It runs the server in a process group of its own holding the
  inherited lock, and settles that group (TERM, then KILL) when it is told to stop. `--external-admission` leaves
  admission to LocalRouter's scheduler (no free+inactive floor, no 34 GiB abort; Metal has no memory cap to keep).
  `--parent-pid` stops it when this worker is gone, which a parent-death signal does on Linux. A server group left by
  a launcher that was itself killed outright is recorded in `runtime/localrouter-drama3.server` and stopped by the next
  load, after checking that the group's leader is still that engine binary.
- Voices. The Metal server speaks only the voices of its `--voices` catalog (drama3-http-voice/1, frame-major codes,
  attested), loaded at start; it takes no references in a request. The worker assembles that catalog in
  `runtime/voices` at each load: the converted named voices (`voices_dir`, default ROOT/voices-macos, written by
  setup-macos.sh) and the cloned voices already spoken (`runtime/clones`, the CLONE_CACHE most recent).
- Cloned voices (`voice_references`, the daemon's voice.json): converted to a catalog voice named for its content
  (`clone-` and 16 hex digits of its sha256), its references packed into the catalog's 4 (adjacent ones joined), with
  the source `consented` (the daemon recorded the caller's consent statement when the voice was cloned). A clone the
  running server has not loaded is written to `runtime/clones` and the server restarted, so the first request in a new
  clone waits for one reload; later ones, and later loads, have it in the catalog.
- Request `references` (codes sent inline) are refused: the tool must set the option `references` false, so the daemon
  does not offer them.
- Clone encodes run upstream's `drama3-generate --encode` (the codec alone on Metal, beside the running server; the
  clip a canonical mono 16-bit 44.1 kHz WAV of at most 30 s) and write the same voice.json as the GB10's
  (`localrouter-voice-codes/1`, codebook-major vq_codes), so a saved voice is the same file on either host.
- No long form (the Metal server has no --long-form); `max_frames` up to 8190.
- The qualification marker (qualify_macos.py) is bound to pins-macos.json, this host and the server binary.
"""

import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python3 -I leaves the script's own directory off sys.path

import drama3_worker as base  # noqa: E402
import voices_macos  # noqa: E402

PINS = os.path.join(HERE, "pins-macos.json")
SERVER_FILE = "localrouter-drama3.server"  # in the runtime directory: the server's process group while it runs
CLONE_PREFIX = "clone-"
CLONE_CACHE = 16  # cloned voices kept in the catalog, most recently spoken first
CLONE_NOTE = ("Cloned through LocalRouter's POST /v1/voices: the caller stated consent=true, that the voice is their "
              "own or that its owner consented on record (source owner or consented); the daemon keeps that statement "
              "with the voice. Codes from the Drama 3 codec, transposed to frame-major.")
STOP_WAIT_S = 120
WAV_HEADER = 44


def load_pins():
    with open(PINS, encoding="utf-8") as f:
        return json.load(f)


class MacOptions(base.Options):
    """The GB10 options, less long form and the full weights, with the Mac pins' paths as the defaults."""

    def __init__(self, tool):
        o = dict(tool.get("options") or {})
        if o.get("variant", "nvfp4") != "nvfp4":
            raise ValueError("on macOS Drama 3 serves the nvfp4 weights only (pins-macos.json)")
        if o.get("long_form"):
            raise ValueError("the Metal engine has no long form: long_form must be false")
        if o.get("references"):
            raise ValueError("the Metal engine takes no references in a request: references must be false")
        given = {k for k in ("pilot", "native_bin", "checkpoint", "voices_dir") if o.get(k)}
        o["variant"] = "nvfp4"
        super().__init__({**tool, "options": o})
        pins = load_pins()
        self.variant_pins = pins  # its serve flags (guidance) and checkpoint
        root = self.root
        if "pilot" not in given:
            self.pilot = os.path.join(root, pins["launcher"]["path"])
        binaries = pins["build"]["binaries"]
        build = os.path.join(root, pins["build"]["dir"])
        if "native_bin" not in given:
            self.native_bin = os.path.join(build, binaries["native"])
        elif self.native_bin.endswith(os.sep + binaries["native"]):
            build = self.native_bin[:-len(binaries["native"])]
        self.generate_bin = os.path.join(build, binaries["generate"])  # the clone encoder, from the same build
        if "checkpoint" not in given:
            self.checkpoint = os.path.join(root, pins["checkpoint"]["dir"])
        # the named voices (setup-macos.sh's conversion); the server gets runtime/voices, assembled at each load
        self.catalog_dir = self.voices_dir if "voices_dir" in given else os.path.join(root, pins["voices"]["dir"])
        self.voices_dir = os.path.join(self.runtime_dir, "voices")
        self.clones_dir = os.path.join(self.runtime_dir, "clones")
        if self.max_frames is not None and not 1 <= int(self.max_frames) <= 8190:
            raise ValueError("max_frames must be 1 to 8190 on the Metal engine")


class MacWorker(base.Worker):
    """The Drama 3 Metal server behind one LocalRouter tool."""

    def __init__(self, tool, require_qualification=True, environ=None, server_default_voice=False):
        self.tool = tool
        self.require_qualification = require_qualification
        self.server_default_voice = server_default_voice
        self.environ = os.environ if environ is None else environ
        self.opts = MacOptions(tool)
        self.pilot = None
        self.unit = None
        self.evidence = None
        self.lines = base.queue.Queue()
        self.tail = base.collections.deque(maxlen=8)
        self.loaded = set()  # the voice names the running server listed
        self.reloads = 0  # restarts for a new cloned voice, for the logs and qualify_macos.py
        self.loads = 0  # this worker's launches: each gets an evidence directory of its own

    # ------------------------------------------------------------ load

    def _load(self):
        if self.require_qualification:
            self._check_qualification()
        self._check_files()
        o = self.opts
        os.makedirs(o.runtime_dir, exist_ok=True)
        lock = self.environ.get("LOCALROUTER_GPU_LOCK", "") or os.path.join(o.runtime_dir, "drama3.lock")
        if not os.path.isfile(lock):
            open(lock, "a").close()
        self._stop_stale_server()
        self._assemble_catalog()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.loads += 1
        evidence = self.evidence = os.path.join(o.root, "evidence", f"{stamp}-{os.getpid()}-{self.loads}")
        os.makedirs(os.path.dirname(evidence), exist_ok=True)
        argv = [sys.executable, "-I", o.pilot, "--native-bin", o.native_bin, "--runtime-dir", o.runtime_dir,
                "--checkpoint", o.checkpoint, "--gpu-lock", lock, "--evidence-dir", evidence, "--port", str(o.port),
                "--ready-timeout", str(o.ready_timeout_s), "--parent-pid", str(os.getpid())]
        if o.admission == "external":
            argv.append("--external-admission")
        serve = o.serve_args(self.server_default_voice)
        if serve:
            argv += ["--", *serve]
        base.log(f"starting the launcher: {' '.join(argv)}")
        self.lines = base.queue.Queue()
        self.pilot = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      start_new_session=True)
        readers = [base.threading.Thread(target=self._read, args=(pipe, protocol), daemon=True)
                   for pipe, protocol in ((self.pilot.stdout, True), (self.pilot.stderr, False))]
        for t in readers:
            t.start()
        deadline = time.monotonic() + o.ready_timeout_s + 30
        while True:
            try:
                line = self.lines.get(timeout=1)
            except base.queue.Empty:
                line = None
            if line == base.READY:
                break
            if line is None and self.pilot.poll() is not None:
                for t in readers:
                    t.join(timeout=5)
                raise RuntimeError(f"the launcher exited ({self.pilot.returncode}) before the server was ready")
            if time.monotonic() > deadline:
                raise RuntimeError("the server was not ready in time")
        self._record_server()
        self._check_voices()
        base.log(f"ready on 127.0.0.1:{o.port} (server group {self.unit})")

    def _check_qualification(self):
        path = self.tool.get("qualification") or ""
        if not path:
            raise ValueError("Drama 3 runs only where it is GPU-qualified: set the tool's qualification to the file "
                             "tools/speech/drama3/qualify_macos.py writes")
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
        if marker.get("pins_sha256") != base.sha256_file(PINS):
            raise ValueError(f"not GPU-qualified on this host: {path} qualified other pins; run qualify_macos.py again")
        if not os.path.isfile(self.opts.native_bin) or marker.get("native_sha256") != base.sha256_file(self.opts.native_bin):
            raise ValueError(f"not GPU-qualified on this host: {path} qualified another engine build; run qualify_macos.py again")

    def _check_files(self):
        o = self.opts
        for path, what in ((o.pilot, "the launcher (setup-macos.sh)"), (o.native_bin, "the engine (setup-macos.sh)")):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} is missing: {what}")
        if not os.access(o.native_bin, os.X_OK):
            raise PermissionError(f"{o.native_bin} is not executable")
        if o.verify == "none":
            return
        pins = load_pins()
        launcher = pins["launcher"]["sha256"]
        if base.sha256_file(o.pilot) != launcher:
            raise ValueError(f"{o.pilot} is not the pinned launcher (sha256 {launcher}): run setup-macos.sh")
        for name, (size, digest) in sorted(pins["checkpoint"]["files"].items()):
            path = os.path.join(o.checkpoint, name)
            if not os.path.isfile(path) or os.path.getsize(path) != size:
                raise ValueError(f"{path} is missing or not {size} bytes: run setup-macos.sh")
            if o.verify == "sha256" and base.sha256_file(path) != digest:
                raise ValueError(f"{path} sha256 differs from the pin {digest}")

    def _assemble_catalog(self):
        """runtime/voices: every named voice, and the cached clones (the CLONE_CACHE most recently spoken), as regular
        files (the server refuses symlinks); anything else there is removed, since the server loads every *.json."""
        o = self.opts
        live = o.voices_dir
        os.makedirs(live, exist_ok=True)
        os.makedirs(o.clones_dir, exist_ok=True)
        for name in os.listdir(live):
            os.unlink(os.path.join(live, name))
        if not os.path.isdir(o.catalog_dir):
            raise FileNotFoundError(f"{o.catalog_dir} is missing: the named voices (setup-macos.sh)")
        for name in sorted(os.listdir(o.catalog_dir)):
            if name.endswith(".json"):
                if name.startswith(CLONE_PREFIX):
                    raise ValueError(f"{o.catalog_dir}/{name}: the {CLONE_PREFIX} names are the worker's")
                shutil.copyfile(os.path.join(o.catalog_dir, name), os.path.join(live, name))
        clones = sorted((n for n in os.listdir(o.clones_dir) if n.startswith(CLONE_PREFIX) and n.endswith(".json")),
                        key=lambda n: os.path.getmtime(os.path.join(o.clones_dir, n)), reverse=True)
        for i, name in enumerate(clones):
            path = os.path.join(o.clones_dir, name)
            if i < CLONE_CACHE:
                shutil.copyfile(path, os.path.join(live, name))
            else:
                os.unlink(path)

    def _record_server(self):
        """The server's process group, from the launcher's session.json, in runtime/SERVER_FILE."""
        try:
            with open(os.path.join(self.evidence, "session.json"), encoding="utf-8") as f:
                pgid = int(json.load(f)["server_pgid"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"the launcher's session.json names no server group ({e})") from e
        self.unit = pgid
        path = os.path.join(self.opts.runtime_dir, SERVER_FILE)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"pgid": pgid, "native_bin": os.path.realpath(self.opts.native_bin)}, f)
        os.replace(path + ".tmp", path)

    def _check_voices(self):
        r = base.Http(self.opts.port, "GET", "/v1/voices", timeout=30)
        try:
            listed = json.loads(r.body()).get("voices", [])
        finally:
            r.close()
        self.loaded = {v for v in listed if isinstance(v, str)}
        wanted = [v for v in dict.fromkeys(self.opts.voices + [self.opts.default_voice]) if v != "default"]
        missing = [v for v in wanted if v not in self.loaded]
        if missing:
            raise ValueError(f"the config lists voices the server did not load from {self.opts.catalog_dir}: {missing}")

    # ------------------------------------------------------------ generate

    def _speak(self, job_dir, r):
        if r.get("references"):
            raise base.Refused("this Drama 3 engine (Metal) speaks catalog voices only: references in a request are "
                               "not taken; clone a voice instead")
        if not r.get("voice_references"):
            return super()._speak(job_dir, r)
        name = self._clone_voice(job_dir, r["voice_references"])
        return super()._speak(job_dir, {**r, "voice_references": "", "voice": name})

    def _clone_voice(self, job_dir, file):
        """The catalog name of a cloned voice's references; a voice the server has not loaded is added to the clone
        cache and the server restarted with it."""
        with open(os.path.join(job_dir, file), encoding="utf-8") as f:
            cloned = json.load(f)
        if isinstance(cloned, dict) and cloned.get("format") == base.VOICE_FORMAT:
            cloned = cloned.get("references")
        if not isinstance(cloned, list) or not cloned or not all(isinstance(c, dict) for c in cloned):
            raise base.Refused("voice_references must be a list of {text, vq_codes}")
        refs = voices_macos.pack([{"text": c.get("text"), "vq_codes": c.get("vq_codes")} for c in cloned])
        if refs is None:
            raise base.Refused(f"the cloned voice's references do not fit the Metal catalog's "
                               f"{voices_macos.MAX_REFERENCES} references of 30 s")
        try:
            data = voices_macos.dumps(voices_macos.catalog(refs, "consented", CLONE_NOTE))
        except voices_macos.Invalid as e:
            raise base.Refused(f"the cloned voice: {e}") from e
        name = CLONE_PREFIX + hashlib.sha256(data).hexdigest()[:16]
        path = os.path.join(self.opts.clones_dir, name + ".json")
        os.makedirs(self.opts.clones_dir, exist_ok=True)
        if not os.path.isfile(path):
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
        os.utime(path)  # most recently spoken: kept in the cache
        if name not in self.loaded:
            began = time.monotonic()
            base.log(f"voice {name} is new to the server: restarting it with the voice in its catalog")
            self.reloads += 1
            self._stop()
            try:
                self._load()
            except BaseException:
                self._stop()
                raise
            base.log(f"restarted with {name} in {time.monotonic() - began:.1f} s")
            if name not in self.loaded:
                raise RuntimeError(f"the server did not load the cloned voice {name}")
        return name

    # ------------------------------------------------------------ clone

    def _encode(self, job_dir, i, ref, out):
        """One reference through `drama3-generate --encode` (the codec alone, on Metal): int32 [frames, 10] codes,
        returned as the GB10 encoder's document (codebook-major vq_codes). A cancel kills it and goes on up."""
        clip = os.path.join(job_dir, ref["audio"])
        o = self.opts
        with open(clip, "rb") as f:
            wav = f.read()
        rate = base.canonical_wav(wav)
        if rate != 44100:
            raise base.Refused(f"reference {i}: the clip must be a mono 16-bit PCM WAV at 44100 Hz")
        seconds = (len(wav) - WAV_HEADER) / 2 / rate
        if seconds > 30:
            raise base.Refused(f"reference {i}: the clip lasts {seconds:.1f} s; a reference is at most 30 s")
        try:
            voices_macos.check_text(ref["text"], f"reference {i}")
        except voices_macos.Invalid as e:
            raise base.Refused(str(e)) from e
        if not os.access(o.generate_bin, os.X_OK):
            raise RuntimeError(f"{o.generate_bin} is missing: the encoder (setup-macos.sh)")
        codes_path = out[:-len(".json")] + ".codes"
        try:
            os.unlink(codes_path)
        except OSError:
            pass
        argv = [o.generate_bin, "--codec", os.path.join(o.checkpoint, "codec.pth"), "--ref-audio", clip,
                "--encode", codes_path]
        began = time.monotonic()
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            try:
                stdout, stderr = p.communicate(timeout=base.ENCODE_TIMEOUT_S)
            except BaseException as e:
                p.kill()
                p.wait()
                if isinstance(e, subprocess.TimeoutExpired):
                    raise RuntimeError(f"reference {i}: the encoder did not finish within {base.ENCODE_TIMEOUT_S} s") from e
                base.log(f"reference {i}: encode killed ({type(e).__name__})")
                raise
            for ln in (stdout + stderr).decode("utf-8", "replace").splitlines():
                if ln.strip():
                    base.log(f"encode ref{i}: {ln}")
            if p.returncode != 0:
                said = [ln for ln in stderr.decode("utf-8", "replace").splitlines() if ln.strip()]
                raise RuntimeError(f"reference {i}: the encoder failed (exit {p.returncode}): "
                                   f"{said[-1] if said else 'no message'}")
            try:
                with open(codes_path, "rb") as f:
                    raw = f.read()
            except OSError as e:
                raise RuntimeError(f"reference {i}: the encoder wrote no codes ({e})") from e
        finally:
            try:
                os.unlink(codes_path)
            except OSError:
                pass
        row = 4 * base.CODEBOOKS
        if not raw or len(raw) % row:
            raise RuntimeError(f"reference {i}: the encoder's codes are not whole [frames, {base.CODEBOOKS}] int32 rows")
        flat = struct.unpack(f"<{len(raw) // 4}i", raw)
        frames = len(flat) // base.CODEBOOKS
        vq_codes = [list(flat[k::base.CODEBOOKS]) for k in range(base.CODEBOOKS)]
        try:
            voices_macos.check_vq_codes(vq_codes, f"reference {i}")
        except voices_macos.Invalid as e:
            raise RuntimeError(f"the encoder's codes: {e}") from e
        base.log(f"reference {i}: {frames} frames in {time.monotonic() - began:.2f} s")
        return {"text": ref["text"], "vq_codes": vq_codes, "frames": frames,
                "audio_sha256": hashlib.sha256(wav).hexdigest(),
                "encoder": {"engine": "drama3-generate --encode (TensorFold Metal)",
                            "generate_sha256": base.sha256_file(o.generate_bin),
                            "codec_sha256": load_pins()["checkpoint"]["files"]["codec.pth"][1],
                            "sample_rate": 44100, "frame_length": 2048, "device": "Apple Metal"}}

    def _clone(self, job_dir, name):
        base.Worker._clone(self, job_dir, name)
        out = os.path.join(job_dir, base.VOICE_OUTPUT)
        with open(out, encoding="utf-8") as f:
            voice = json.load(f)
        if voices_macos.pack(voice["references"]) is None:
            os.unlink(out)
            raise base.Refused(f"these references do not fit the Metal catalog's {voices_macos.MAX_REFERENCES} "
                               f"references of 30 s, even with adjacent ones joined")

    # ------------------------------------------------------------ exit

    def _stop(self):
        """SIGTERM to the launcher, which settles the server's group and then releases the lock."""
        p, self.pilot = self.pilot, None
        if p is None:
            return
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
                p.wait(timeout=STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                base.log(f"the launcher did not stop within {STOP_WAIT_S} s; killing it and its server group")
                p.kill()
                p.wait()
                self._kill_group(self.unit, self.opts.native_bin)
        base.log(f"launcher exited ({p.returncode})")
        self.loaded = set()
        try:
            os.unlink(os.path.join(self.opts.runtime_dir, SERVER_FILE))
        except OSError:
            pass

    def _stop_stale_server(self):
        """A server group left by a launcher that was killed outright is stopped, if its leader is still our engine."""
        path = os.path.join(self.opts.runtime_dir, SERVER_FILE)
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            return
        self._kill_group(rec.get("pgid"), rec.get("native_bin"))
        os.unlink(path)

    @staticmethod
    def _kill_group(pgid, native_bin):
        if not isinstance(pgid, int) or pgid <= 1 or not native_bin:
            return
        try:
            leader = subprocess.run(["/bin/ps", "-o", "comm=", "-p", str(pgid)], capture_output=True, text=True,
                                    timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return
        if os.path.realpath(leader) != os.path.realpath(native_bin) and os.path.basename(leader) != os.path.basename(native_bin):
            return  # gone, or the number is someone else's now
        base.log(f"stopping server group {pgid}, left by an earlier run")
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.1)


def main():
    signal.signal(signal.SIGUSR1, base.on_cancel)
    try:
        worker = MacWorker(json.loads(os.environ.get("LOCALROUTER_TOOL", "{}")))
    except (ValueError, TypeError, OSError) as e:
        worker, problem = None, str(e)
    if worker is not None:
        base.watch_parent(worker)
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd.get("op")
        if worker is None:
            if op == "exit":
                return
            base.send({"ok": False, "error": f"bad tool config: {problem}", "type": "server_error"})
            continue
        if op == "load":
            base.send(worker.load())
        elif op == "generate":
            reply = base.generate(worker, cmd)
            if reply.pop("gone", False):
                base.log(f"{reply['error']}: exiting so the daemon restarts the model")
                worker._stop()
                os._exit(1)
            base.send(reply)
        elif op == "exit":
            stopping = base.threading.Thread(target=worker.exit)
            stopping.start()
            while stopping.is_alive():
                stopping.join(timeout=5)
                if stopping.is_alive():
                    base.send({"progress": {"phase": "unload"}})
            return
    if worker is not None:
        worker._stop()


if __name__ == "__main__":
    main()
