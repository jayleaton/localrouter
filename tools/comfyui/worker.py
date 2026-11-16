#!/usr/bin/env python3
"""LocalRouter `cmd` worker that serves image and video tools through ComfyUI (docs/ARCHITECTURE.md, "the cmd adapter").

The worker starts its own ComfyUI (or uses one already running), fills a ComfyUI API-format workflow from the request
through the tool's `bindings`, queues it, follows its progress, and copies the outputs into the job directory:
`0.png`, `1.png`, ... for an image request (one workflow run per image), `video.mp4` for a video request.

Standard library only. One JSON object a line on stdin and stdout; stdout carries protocol lines only, logs go to stderr.
  {"op":"load"}      -> starts ComfyUI and waits for /system_stats -> {"ok":true,"resident":N,"cancel":"signal"}
  {"op":"generate","id","dir","request":{"image":{...}}|{"video":{...}}} -> {"ok":true,"files":[...],"seed":S,"ms":T}
  SIGUSR1 during a generate -> ComfyUI /interrupt, the queued prompt dropped -> {"ok":false,"type":"cancelled"}
  {"op":"exit"} or EOF -> stops the ComfyUI it started, then exits 0

The tool's config arrives as JSON in LOCALROUTER_TOOL. Options:
  comfy_dir     ComfyUI checkout (main.py) to start; absent: use `url`
  python        the interpreter of ComfyUI's environment (default: python3 on PATH)
  args          extra ComfyUI arguments (e.g. ["--disable-mmap", "--cache-none"])
  env           extra environment for ComfyUI ({port} substituted)
  port          fixed loopback port (absent: a free one)
  url           base URL of a ComfyUI already running (only when comfy_dir is absent; its outputs are left in place)
  workflow      path of the API-format workflow ("Export (API)" in ComfyUI); absolute, or relative to the daemon's
                working directory
  bindings      request field -> "node.input" (or a list of them) in the workflow:
                  prompt (required), negative_prompt, width, height, seed, steps, guidance,
                  frames (video: the frame count), fps, audio ("node.input" removed when a request asks for no audio)
                A request that sets a field the bindings do not name (a negative prompt, steps, guidance) is refused,
                never ignored.
  output_nodes  ids of the nodes whose outputs are the result (default: every output node)
  size_multiple width and height must be multiples of this (default 8); min_side, max_side, max_pixels: limits
  frame_multiple, frame_offset  the model's frame grid: frames = the smallest offset + k*multiple >= seconds*fps
  fps           the frame rates the model takes (a list; default: any)
  stop_timeout_s  seconds between SIGTERM and SIGKILL when stopping ComfyUI (default 30)
Memory is the config's resident_bytes and working_bytes (measured, not guessed here); the load reports resident_bytes.
"""
import base64
import ctypes
import glob
import json
import os
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PROTO = os.fdopen(os.dup(1), "w", buffering=1)  # the protocol stream; fd 1 itself now points at stderr
os.dup2(2, 1)
sys.stdout = sys.stderr
PARENT = int(os.environ.get("LOCALROUTER_PARENT") or 0)
CANCEL = [False]
IMAGE_EXT = (".png",)
VIDEO_EXT = (".mp4",)


class InvalidRequest(Exception):
    pass


class Unavailable(Exception):
    pass


class Cancelled(Exception):
    pass


def log(msg):
    print(f"comfyui-worker: {msg}", file=sys.stderr, flush=True)


def send(obj):
    PROTO.write(json.dumps(obj, separators=(",", ":")) + "\n")
    PROTO.flush()


def orphaned():
    return PARENT not in (0, 1) and os.getppid() != PARENT


def die_with_parent():
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG: stop ComfyUI first
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
        try:
            ready, _, _ = select.select([0], [], [], 1.0)
        except InterruptedError:
            continue
        if ready:
            chunk = os.read(0, 1 << 16)
            if not chunk:
                return bytes(_pending) if _pending.strip() else None
            _pending.extend(chunk)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Events:
    """ComfyUI's /ws progress events (a minimal RFC 6455 client: text frames in, nothing out but the close)."""

    def __init__(self, base, client_id):
        u = urllib.parse.urlparse(base)
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /ws?clientId={client_id} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
                           f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket closed during the handshake")
            head += chunk
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError("websocket upgrade refused")
        self.buf = bytearray(head.split(b"\r\n\r\n", 1)[1])
        self.sock.setblocking(False)

    def poll(self, timeout):
        """Text messages that arrived within `timeout` seconds, parsed."""
        out = []
        try:
            ready, _, _ = select.select([self.sock], [], [], timeout)
        except InterruptedError:
            ready = []
        if ready:
            try:
                chunk = self.sock.recv(1 << 20)
            except BlockingIOError:
                chunk = None
            if chunk == b"":
                raise ConnectionError("websocket closed")
            if chunk:
                self.buf.extend(chunk)
        while len(self.buf) >= 2:
            op, n, i = self.buf[0] & 0x0F, self.buf[1] & 0x7F, 2
            if n == 126:
                if len(self.buf) < 4:
                    break
                n, i = struct.unpack(">H", self.buf[2:4])[0], 4
            elif n == 127:
                if len(self.buf) < 10:
                    break
                n, i = struct.unpack(">Q", self.buf[2:10])[0], 10
            if len(self.buf) < i + n:
                break
            payload = bytes(self.buf[i:i + n])
            del self.buf[:i + n]
            if op == 1:
                try:
                    out.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
            elif op == 8:
                raise ConnectionError("websocket closed")
        return out

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Worker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = cfg.get("options") or {}
        self.proc = None
        self.base = None
        self.out_dir = None  # the private --output-directory of the ComfyUI this worker started
        self.workflow = self.load_workflow()
        self.bindings = {k: ([v] if isinstance(v, str) else list(v)) for k, v in (self.opts.get("bindings") or {}).items()}
        if "prompt" not in self.bindings:
            raise RuntimeError("options.bindings.prompt is required")
        for field, targets in self.bindings.items():
            for t in targets:
                node, _, inp = t.partition(".")
                if node not in self.workflow or not inp or (field != "audio" and inp not in self.workflow[node].get("inputs", {})):
                    raise RuntimeError(f"binding {field} -> {t}: no such node input in the workflow")

    def load_workflow(self):
        path = self.opts.get("workflow") or ""
        if not path:
            raise RuntimeError("options.workflow is required")
        for p in [path]:
            if os.path.isfile(p):
                with open(p) as f:
                    wf = json.load(f)
                if not isinstance(wf, dict) or not all(isinstance(v, dict) and "class_type" in v for v in wf.values()):
                    raise RuntimeError(f"{p} is not an API-format workflow (use Export (API) in ComfyUI)")
                return wf
        raise RuntimeError(f"workflow {path} not found")

    # ---- ComfyUI itself
    def healthy(self, timeout=2.0):
        try:
            with urllib.request.urlopen(self.base + "/system_stats", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def loaded(self):
        return {"ok": True, "resident": int(self.cfg.get("resident_bytes") or 0), "cancel": "signal"}

    def load(self):
        if self.base and (self.proc is None or self.proc.poll() is None) and self.healthy():
            return self.loaded()
        self.stop()
        comfy = self.opts.get("comfy_dir")
        if comfy:
            port = int(self.opts.get("port") or 0) or free_port()
            self.out_dir = tempfile.mkdtemp(prefix="lr-comfyui-")
            argv = [self.opts.get("python") or "python3", "main.py", "--listen", "127.0.0.1", "--port", str(port),
                    "--output-directory", self.out_dir] + [str(a) for a in self.opts.get("args") or []]
            env = dict(os.environ)
            env.update({k: str(v).replace("{port}", str(port)) for k, v in (self.opts.get("env") or {}).items()})
            log(f"starting ComfyUI ({comfy}) on 127.0.0.1:{port}")
            self.proc = subprocess.Popen(argv, cwd=comfy, env=env, stdin=subprocess.DEVNULL, stdout=sys.stderr,
                                         stderr=sys.stderr, start_new_session=True)
            self.base = f"http://127.0.0.1:{port}"
        else:
            self.base = (self.opts.get("url") or "").rstrip("/")
            if not self.base:
                raise RuntimeError("options.comfy_dir or options.url is required")
        deadline = time.monotonic() + float(self.cfg.get("load_timeout_s") or 300)
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                code = self.proc.returncode
                self.proc = None
                raise RuntimeError(f"ComfyUI exited before it was ready (status {code}); see the tool's log")
            if self.healthy():
                log(f"ComfyUI ready at {self.base}")
                return self.loaded()
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("ComfyUI did not answer /system_stats in time")

    def stop(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + float(self.opts.get("stop_timeout_s") or 30)
            while p.poll() is None and time.monotonic() < deadline:
                send({"progress": {"phase": "unload", "step": 0, "of": 1}})  # the daemon waits while memory is still held
                time.sleep(0.5)
            if p.poll() is None:
                log("ComfyUI ignored SIGTERM; killing it")
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        if self.out_dir:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.out_dir = None

    def ensure(self):
        if self.base is None or (self.proc is not None and self.proc.poll() is not None):
            self.load()

    def call(self, path, body=None, timeout=30):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data, {"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw[:1] in (b"{", b"[") else raw
        except urllib.error.HTTPError as e:
            msg = e.read()[:2000].decode("utf-8", "replace")
            raise RuntimeError(f"ComfyUI {path} failed ({e.code}): {msg}")
        except (urllib.error.URLError, ConnectionError) as e:
            raise Unavailable(f"ComfyUI is not reachable: {e}")

    # ---- requests
    def set(self, wf, field, value):
        for t in self.bindings.get(field, []):
            node, _, inp = t.partition(".")
            wf[node]["inputs"][inp] = value

    def check_size(self, w, h):
        m = int(self.opts.get("size_multiple") or 8)
        if w % m or h % m:
            raise InvalidRequest(f"model '{self.cfg.get('id')}' takes sizes in multiples of {m}, not {w}x{h}")
        lo, hi = int(self.opts.get("min_side") or 0), int(self.opts.get("max_side") or 0)
        if (lo and min(w, h) < lo) or (hi and max(w, h) > hi):
            raise InvalidRequest(f"model '{self.cfg.get('id')}' takes sides of {lo or 1} to {hi or 'any'} pixels, not {w}x{h}")
        px = int(self.opts.get("max_pixels") or 0)
        if px and w * h > px:
            raise InvalidRequest(f"model '{self.cfg.get('id')}' takes at most {px} pixels, not {w}x{h}")

    def fill(self, r, seed):
        """The workflow for one run: a deep copy with the request's fields written into the bound inputs."""
        wf = json.loads(json.dumps(self.workflow))
        self.set(wf, "prompt", r.get("prompt") or "")
        for field in ("negative_prompt", "steps", "guidance"):
            v = r.get(field)
            if v and field not in self.bindings:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' takes no {field.replace('_', ' ')}")
            if v:
                self.set(wf, field, v)
        if r.get("width") and r.get("height"):
            if "width" not in self.bindings or "height" not in self.bindings:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' has a fixed size")
            self.check_size(int(r["width"]), int(r["height"]))
            self.set(wf, "width", int(r["width"]))
            self.set(wf, "height", int(r["height"]))
        self.set(wf, "seed", seed)
        return wf

    def generate(self, cmd):
        t0 = time.monotonic()
        CANCEL[0] = False
        req = cmd.get("request") or {}
        job_dir = cmd.get("dir") or ""
        if not os.path.isdir(job_dir):
            raise InvalidRequest("job directory does not exist")
        if isinstance(req.get("image"), dict):
            r = req["image"]
            if r.get("references") or r.get("mask"):
                raise InvalidRequest(f"model '{self.cfg.get('id')}' makes images from text only (no edits)")
            seed = int(r.get("seed") or 0)
            n = max(1, int(r.get("n") or 1))
            runs = [(self.fill(r, seed + i), f"{i}.png", IMAGE_EXT) for i in range(n)]
        elif isinstance(req.get("video"), dict):
            r = req["video"]
            if r.get("first_frame"):
                raise InvalidRequest(f"model '{self.cfg.get('id')}' makes videos from text only (no first frame)")
            seed = int(r.get("seed") or 0)
            wf = self.fill(r, seed)
            fps = int(r.get("fps") or 24)
            allowed = self.opts.get("fps")
            if allowed and fps not in allowed:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' takes fps {', '.join(map(str, allowed))}, not {fps}")
            self.set(wf, "fps", fps)
            frames = int(r.get("seconds") or 5) * fps
            mul, off = int(self.opts.get("frame_multiple") or 1), int(self.opts.get("frame_offset") or 0)
            if mul > 1:
                frames = off + max(0, -(-(frames - off) // mul)) * mul
            if "frames" not in self.bindings:
                raise RuntimeError("options.bindings.frames is required for a video tool")
            self.set(wf, "frames", frames)
            if r.get("audio") is False:
                if "audio" not in self.bindings:
                    raise InvalidRequest(f"model '{self.cfg.get('id')}' always makes sound: leave audio on")
                for t in self.bindings["audio"]:
                    node, _, inp = t.partition(".")
                    wf[node]["inputs"].pop(inp, None)
            runs = [(wf, "video.mp4", VIDEO_EXT)]
        else:
            raise InvalidRequest("this tool serves image and video requests only")
        self.ensure()
        files = []
        for i, (wf, name, exts) in enumerate(runs):
            self.run(wf, os.path.join(job_dir, name), exts, i, len(runs))
            files.append(name)
        return {"ok": True, "files": files, "seed": seed, "ms": int((time.monotonic() - t0) * 1000)}

    def run(self, wf, dest, exts, index, count):
        client = uuid.uuid4().hex
        events = None
        try:
            events = Events(self.base, client)
        except (OSError, ConnectionError) as e:
            log(f"no progress events ({e}); polling only")
        try:
            res = self.call("/prompt", {"prompt": wf, "client_id": client})
            if res.get("node_errors"):
                raise RuntimeError(f"ComfyUI refused the workflow: {json.dumps(res['node_errors'])[:1500]}")
            pid = res["prompt_id"]
            deadline = time.monotonic() + float(self.cfg.get("generate_timeout_s") or 3600)
            last_poll = 0.0
            step, of = index, count  # the last fraction reported: node phases repeat it, so progress never goes back
            send({"progress": {"phase": "queued", "step": step, "of": of}})
            while True:
                if CANCEL[0]:
                    self.cancel(pid)
                    raise Cancelled()
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(f"ComfyUI exited during the job (status {self.proc.returncode}); see the tool's log")
                if time.monotonic() > deadline:
                    self.cancel(pid)
                    raise RuntimeError("the job ran past generate_timeout_s")
                if events is not None:
                    try:
                        for m in events.poll(0.5):
                            d = m.get("data") or {}
                            if m.get("type") == "progress" and d.get("prompt_id") in (None, pid) and d.get("max"):
                                # one run of several: its steps are a slice of the whole
                                of = int(d["max"]) * count
                                step = int(d["max"]) * index + min(int(d.get("value") or 0), int(d["max"]))
                                send({"progress": {"phase": "denoise", "step": step, "of": of}})
                            elif m.get("type") == "executing" and d.get("prompt_id") == pid and d.get("node"):
                                node = wf.get(str(d["node"]), {}).get("class_type", str(d["node"]))
                                send({"progress": {"phase": node, "step": step, "of": of}})
                    except (OSError, ConnectionError):
                        events.close()
                        events = None
                else:
                    time.sleep(0.5)
                if time.monotonic() - last_poll < 1.0:
                    continue
                last_poll = time.monotonic()
                h = self.call(f"/history/{pid}")
                entry = h.get(pid) if isinstance(h, dict) else None
                status = (entry or {}).get("status") or {}
                if entry and (status.get("completed") or status.get("status_str") in ("success", "error")):
                    break
            if status.get("status_str") == "error" or not status.get("completed", True):
                why = "the workflow failed"
                for kind, data in status.get("messages") or []:
                    if kind == "execution_error":
                        why = f"{data.get('node_type')}: {data.get('exception_type')}: {(data.get('exception_message') or '').strip()}"
                    elif kind == "execution_interrupted":
                        raise Cancelled()
                raise RuntimeError(f"ComfyUI: {why}"[:2000])
            self.fetch(entry.get("outputs") or {}, dest, exts)
        finally:
            if events is not None:
                events.close()

    def cancel(self, pid):
        for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
            try:
                self.call(path, body, timeout=10)
            except Exception as e:
                log(f"cancel: {path}: {e}")

    def fetch(self, outputs, dest, exts):
        want = [str(n) for n in self.opts.get("output_nodes") or []]
        found = []
        for node, out in outputs.items():
            if want and node not in want:
                continue
            for key in ("images", "videos", "gifs"):
                for f in out.get(key) or []:
                    if isinstance(f, dict) and f.get("filename", "").lower().endswith(exts) and f.get("type", "output") == "output":
                        found.append(f)
        if not found:
            raise RuntimeError(f"the workflow wrote no {'/'.join(exts)} output")
        if len(found) > 1:
            log(f"{len(found)} outputs; keeping the first ({found[0]['filename']})")
        f = found[0]
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"})
        data = self.call(f"/view?{q}", timeout=120)
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise RuntimeError("ComfyUI returned an empty output file")
        tmp = dest + ".tmp"
        with open(tmp, "wb") as o:
            o.write(data)
        os.replace(tmp, dest)
        if self.out_dir:  # our own output directory: leave nothing behind
            for p in glob.glob(os.path.join(self.out_dir, "*")):
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.unlink(p)


def main():
    die_with_parent()
    if orphaned():
        return 0
    try:
        cfg = json.loads(os.environ.get("LOCALROUTER_TOOL") or "{}")
        w = Worker(cfg)
    except Exception as e:
        log(f"bad tool config: {e}")
        line = read_line()
        if line is not None:
            send({"ok": False, "error": f"bad tool config: {e}", "type": "server_error"})
        return 2
    signal.signal(signal.SIGTERM, lambda *_: (w.stop(), sys.exit(0)))
    signal.signal(signal.SIGUSR1, lambda *_: CANCEL.__setitem__(0, True))
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
            except Cancelled:
                send({"ok": False, "error": "cancelled", "type": "cancelled"})
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
