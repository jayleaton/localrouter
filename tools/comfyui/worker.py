#!/usr/bin/env python3
"""LocalRouter `cmd` worker that serves image and video tools through ComfyUI (docs/ARCHITECTURE.md, "the cmd adapter").

The worker starts its own ComfyUI (or uses one already running), fills a ComfyUI API-format workflow from the request
through the tool's `bindings`, queues it, follows its progress, and copies the outputs into the job directory:
`0.png`, `1.png`, ... for an image request (one workflow run per image), `video.mp4` for a video request. Input images
(an edit's references and mask, a video's first frame) are uploaded to ComfyUI and wired in through LoadImage nodes.
An edit without a size keeps whatever size the workflow derives from its input image.

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
                File inputs: references (a list: the n-th target takes the n-th input image; more images than
                targets are refused), first_frame, mask. For each, the worker uploads the file to ComfyUI and links
                a LoadImage node it adds: its IMAGE output, or for `mask` its MASK output (1 where the mask is
                transparent, which is where an edit goes)
  modes         per capability, what changes for it: {"image_edit": {...}, "image_to_video": {...}}, each with its
                own `workflow` (absent: the tool's), `bindings` (merged over the tool's; a JSON null drops a binding)
                and `output_nodes`. `defaults` fills steps, guidance or negative_prompt when the request leaves them
                unset (0 or absent). image_edit may also set `unmasked` (the same fields), used when a request has
                input images but no mask, so a whole-scene restyle can use heavier defaults than a masked local edit.
                A request may still set size, steps or guidance. A request with input images uses `image_edit` (or
                its unmasked overlay), a video request with a first frame `image_to_video`; a tool without the mode
                refuses them
  output_nodes  ids of the nodes whose outputs are the result (default: every output node)
  size_multiple width and height must be multiples of this (default 8); min_side, max_side, max_pixels: limits
  frame_multiple, frame_offset  the model's frame grid: frames = the smallest offset + k*multiple >= seconds*fps
  fps           the frame rates the model takes (a list; default: any)
  stop_timeout_s  seconds between SIGTERM and SIGKILL when stopping ComfyUI (default 30)
Memory is the config's resident_bytes and working_bytes (measured, not guessed here); the load reports resident_bytes.
"""
import ctypes
import glob
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script's directory out
from comfy_events import Events  # noqa: E402

PROTO = os.fdopen(os.dup(1), "w", buffering=1)  # the protocol stream; fd 1 itself now points at stderr
os.dup2(2, 1)
sys.stdout = sys.stderr
PARENT = int(os.environ.get("LOCALROUTER_PARENT") or 0)
CANCEL = [False]
IMAGE_EXT = (".png",)
FILE_FIELDS = ("references", "first_frame", "mask")
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


def comfy_preexec():
    """In ComfyUI's process, before exec: SIGTERM when this worker dies by any means (PR_SET_PDEATHSIG). ComfyUI also
    stays in the worker's process group (no new session), so the daemon's SIGKILL of that group on abort or timeout
    takes it down with the worker; this covers the worker dying alone."""
    die_with_parent()


def tree(pid):
    """`pid` and every process below it (Linux: /proc/PID/task/TID/children), parents first."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop(0)
        out.append(p)
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/children") as f:
                    todo += [int(c) for c in f.read().split()]
        except (OSError, ValueError):
            pass
    return out


def signal_tree(pids, sig):
    for p in pids:
        try:
            os.kill(p, sig)
        except ProcessLookupError:
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


class Worker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.opts = cfg.get("options") or {}
        self.proc = None
        self.base = None
        self.out_dir = None  # the private --output-directory of the ComfyUI this worker started
        self.in_dir = None   # and its --input-directory, where uploaded input images land
        self.modes = {"": self.mode(self.opts, None)}
        for name, m in (self.opts.get("modes") or {}).items():
            if name not in ("image_edit", "image_to_video"):
                raise RuntimeError(f"options.modes.{name}: only image_edit and image_to_video are known")
            m = m or {}
            self.modes[name] = self.mode({k: v for k, v in m.items() if k != "unmasked"}, self.modes[""])
            if name == "image_edit" and m.get("unmasked") is not None:
                self.modes["image_edit_unmasked"] = self.mode(m["unmasked"], self.modes["image_edit"])

    def mode(self, o, base):
        """(workflow, bindings, output_nodes, defaults) for the tool, or for one of its modes over the tool's."""
        wf = self.load_workflow(o["workflow"]) if o.get("workflow") else (base[0] if base else self.load_workflow(""))
        bindings = dict(base[1]) if base else {}
        for k, v in (o.get("bindings") or {}).items():
            if v is None:
                bindings.pop(k, None)
            else:
                bindings[k] = [v] if isinstance(v, str) else list(v)
        if "prompt" not in bindings:
            raise RuntimeError("options.bindings.prompt is required")
        for field, targets in bindings.items():
            for t in targets:
                node, _, inp = t.partition(".")
                loose = field in FILE_FIELDS or field == "audio"  # optional inputs that may be unconnected
                if node not in wf or not inp or (not loose and inp not in wf[node].get("inputs", {})):
                    raise RuntimeError(f"binding {field} -> {t}: no such node input in the workflow")
        outs = [str(n) for n in (o.get("output_nodes") or (base[2] if base else self.opts.get("output_nodes")) or [])]
        defaults = dict(base[3]) if base else {}
        defaults.update(o.get("defaults") or {})
        for field in defaults:
            if field not in bindings:
                raise RuntimeError(f"defaults.{field}: no binding")
        return wf, bindings, outs, defaults

    def load_workflow(self, path):
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
            self.in_dir = os.path.join(self.out_dir, "in")  # outside ComfyUI's own output folders
            os.makedirs(self.in_dir)
            os.makedirs(os.path.join(self.out_dir, "out"))
            argv = [self.opts.get("python") or "python3", "main.py", "--listen", "127.0.0.1", "--port", str(port),
                    "--output-directory", os.path.join(self.out_dir, "out"), "--input-directory", self.in_dir] + \
                   [str(a) for a in self.opts.get("args") or []]
            env = dict(os.environ)
            env.update({k: str(v).replace("{port}", str(port)) for k, v in (self.opts.get("env") or {}).items()})
            log(f"starting ComfyUI ({comfy}) on 127.0.0.1:{port}")
            # in this worker's process group (the daemon kills the group on abort or timeout), dying with the worker
            self.proc = subprocess.Popen(argv, cwd=comfy, env=env, stdin=subprocess.DEVNULL, stdout=sys.stderr,
                                         stderr=sys.stderr, preexec_fn=comfy_preexec)
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
            signal_tree(tree(p.pid), signal.SIGTERM)  # ComfyUI and what it started (the group is this worker's too)
            deadline = time.monotonic() + float(self.opts.get("stop_timeout_s") or 30)
            while p.poll() is None and time.monotonic() < deadline:
                send({"progress": {"phase": "unload", "step": 0, "of": 1}})  # the daemon waits while memory is still held
                time.sleep(0.5)
            if p.poll() is None:
                log("ComfyUI ignored SIGTERM; killing it")
                signal_tree(tree(p.pid), signal.SIGKILL)
                p.wait()
        if self.out_dir:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            self.out_dir = self.in_dir = None

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
    def _defaults_note(self, mode):
        d = mode[3] if len(mode) > 3 and mode[3] else {}
        if not d:
            return ""
        return "; defaults " + ", ".join(f"{k}={v}" for k, v in d.items())

    def set(self, wf, bindings, field, value):
        for t in bindings.get(field, []):
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

    def fill(self, mode, r, seed, inputs):
        """The workflow for one run: a deep copy with the request's fields written into the bound inputs, and a
        LoadImage node for each input image (`inputs`: field -> list of names ComfyUI gave the uploaded files)."""
        src, b, _, defaults = mode
        wf = json.loads(json.dumps(src))
        self.set(wf, b, "prompt", r.get("prompt") or "")
        for field in ("negative_prompt", "steps", "guidance"):
            given = r.get(field)
            v = given if given else defaults.get(field)
            if given and field not in b:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' takes no {field.replace('_', ' ')}")
            if v:
                self.set(wf, b, field, v)
        if r.get("width") and r.get("height"):
            if "width" not in b or "height" not in b:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' has a fixed size")
            self.check_size(int(r["width"]), int(r["height"]))
            self.set(wf, b, "width", int(r["width"]))
            self.set(wf, b, "height", int(r["height"]))
        self.set(wf, b, "seed", seed)
        nid = 900000
        for field, names in inputs.items():
            targets = b[field] if field == "references" else b[field] * len(names)
            for name, t in zip(names, targets):
                node, _, inp = t.partition(".")
                while str(nid) in wf:
                    nid += 1
                wf[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": name}}
                wf[node]["inputs"][inp] = [str(nid), 1 if field == "mask" else 0]
        return wf

    def inputs(self, mode, r, job_dir):
        """Checks and uploads a request's input images -> field -> the names ComfyUI stored them under."""
        b = mode[1]
        files = {"references": list(r.get("references") or []), "first_frame": [r["first_frame"]] if r.get("first_frame") else [],
                 "mask": [r["mask"]] if r.get("mask") else []}
        out = {}
        for field, names in files.items():
            if not names:
                continue
            if field not in b:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' takes no {field.replace('_', ' ')}")
            if field == "references" and len(names) > len(b[field]):
                raise InvalidRequest(f"model '{self.cfg.get('id')}' takes at most {len(b[field])} input images")
            for n in names:
                if os.path.basename(n) != n or n.startswith(".") or not os.path.isfile(os.path.join(job_dir, n)):
                    raise InvalidRequest(f"{field} must name a file in the job directory")
            out[field] = names
        return out

    def upload(self, job_dir, names, tag):
        stored = []
        for n in names:
            with open(os.path.join(job_dir, n), "rb") as f:
                data = f.read()
            b = uuid.uuid4().hex
            body = bytearray()
            for k, v in (("type", "input"), ("subfolder", f"localrouter-{tag}"), ("overwrite", "true")):
                body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            body += (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{n}"\r\n'
                     f'Content-Type: application/octet-stream\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
            req = urllib.request.Request(self.base + "/upload/image", bytes(body), {"Content-Type": f"multipart/form-data; boundary={b}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    res = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"ComfyUI /upload/image failed ({e.code}): {e.read()[:500].decode('utf-8', 'replace')}")
            except (urllib.error.URLError, ConnectionError) as e:
                raise Unavailable(f"ComfyUI is not reachable: {e}")
            stored.append(f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"])
        return stored

    def generate(self, cmd):
        t0 = time.monotonic()
        CANCEL[0] = False
        req = cmd.get("request") or {}
        job_dir = cmd.get("dir") or ""
        if not os.path.isdir(job_dir):
            raise InvalidRequest("job directory does not exist")
        if isinstance(req.get("image"), dict):
            r = req["image"]
            if r.get("references") and "image_edit" not in self.modes:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' makes images from text only (no edits)")
            if r.get("references"):
                if r.get("mask") or "image_edit_unmasked" not in self.modes:
                    mode = self.modes["image_edit"]
                    log("image_edit" + (" with a mask" if r.get("mask") else "") + self._defaults_note(mode))
                else:
                    mode = self.modes["image_edit_unmasked"]
                    log("image_edit without a mask: unmasked workflow" + self._defaults_note(mode))
            else:
                mode = self.modes[""]
            inputs = self.inputs(mode, r, job_dir)
            seed = int(r.get("seed") or 0)
            n = max(1, int(r.get("n") or 1))
            for i in range(n):  # check every run before anything loads or uploads
                self.fill(mode, r, seed + i, {k: v for k, v in inputs.items()})
            kind, ext, names = "image", IMAGE_EXT, [f"{i}.png" for i in range(n)]
        elif isinstance(req.get("video"), dict):
            r = req["video"]
            if r.get("first_frame") and "image_to_video" not in self.modes:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' makes videos from text only (no first frame)")
            mode = self.modes["image_to_video" if r.get("first_frame") else ""]
            inputs = self.inputs(mode, r, job_dir)
            seed = int(r.get("seed") or 0)
            wf = self.fill(mode, r, seed, inputs)
            fps = int(r.get("fps") or 24)
            allowed = self.opts.get("fps")
            if allowed and fps not in allowed:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' takes fps {', '.join(map(str, allowed))}, not {fps}")
            b = mode[1]
            frames = int(r.get("seconds") or 5) * fps
            mul, off = int(self.opts.get("frame_multiple") or 1), int(self.opts.get("frame_offset") or 0)
            if mul > 1:
                frames = off + max(0, -(-(frames - off) // mul)) * mul
            if "frames" not in b:
                raise RuntimeError("options.bindings.frames is required for a video tool")
            if r.get("audio") is False and "audio" not in b:
                raise InvalidRequest(f"model '{self.cfg.get('id')}' always makes sound: leave audio on")
            kind, ext, names = "video", VIDEO_EXT, ["video.mp4"]

            def video_wf(stored):
                w = self.fill(mode, r, seed, stored)
                self.set(w, b, "fps", fps)
                self.set(w, b, "frames", frames)
                if r.get("audio") is False:
                    for t in b["audio"]:
                        node, _, inp = t.partition(".")
                        w[node]["inputs"].pop(inp, None)
                return w
        else:
            raise InvalidRequest("this tool serves image and video requests only")
        self.ensure()
        tag = (cmd.get("id") or uuid.uuid4().hex)[-24:]
        stored = {field: self.upload(job_dir, names_, tag) for field, names_ in inputs.items()}
        files = []
        for i, name in enumerate(names):
            wf = self.fill(mode, r, seed + i, stored) if kind == "image" else video_wf(stored)
            self.run(wf, os.path.join(job_dir, name), ext, i, len(names), mode[2])
            files.append(name)
        if self.in_dir:  # our own input directory: leave nothing behind
            shutil.rmtree(os.path.join(self.in_dir, f"localrouter-{tag}"), ignore_errors=True)
        return {"ok": True, "files": files, "seed": seed, "ms": int((time.monotonic() - t0) * 1000)}

    def run(self, wf, dest, exts, index, count, output_nodes):
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
            self.fetch(entry.get("outputs") or {}, dest, exts, output_nodes)
        finally:
            if events is not None:
                events.close()

    def cancel(self, pid):
        for path, body in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
            try:
                self.call(path, body, timeout=10)
            except Exception as e:
                log(f"cancel: {path}: {e}")

    def fetch(self, outputs, dest, exts, want):
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
            for p in glob.glob(os.path.join(self.out_dir, "out", "*")):
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
