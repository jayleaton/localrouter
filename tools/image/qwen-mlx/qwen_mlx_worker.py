#!/usr/bin/env python3
"""LocalRouter worker for Qwen-Image 2.1 Turbo on macOS (Apple Silicon): text to image and image edits on MLX.

Speaks the worker protocol (docs/ARCHITECTURE.md): one JSON object a line on stdin and stdout, logs on stderr.

    {"op":"load"}                                   -> {"ok":true,"resident":<bytes>,"cancel":"signal"}
    {"op":"generate","id","dir","request":{"image":{"prompt","width","height","n","seed","steps",
        "references":["ref_0.png",…],"mask":"mask_0.png"}}}
                                                    -> {"ok":true,"files":["0.png",…],"seed":7,"ms":…}
    SIGUSR1 during a generate                       -> {"ok":false,"type":"cancelled"} (still loaded)
    {"op":"exit"}

The model runs in MLX-Serve (github.com/ddalcu/mlx-serve, a native Zig/MLX server; release and pack pinned in
pins-macos.json, fetched by fetch.sh), which this worker starts on load on 127.0.0.1 and is the client of: it is the
"models that wrap their own server" shape of docs/ARCHITECTURE.md. The pack is ddalcu/Qwen-Image-2.1-Turbo-MLX-Serve-8bit
(MLX affine 8-bit DiT and text-encoder linears, dense f32 VAE, the Qwen3-VL vision tower kept for edits). It runs the
checkpoint's own eight-step `sample_sigmas` schedule (unshifted), which MLX-Serve reads from model_index.json since
26.10.2; an older server ran the base model's schedule on it, so load refuses one.

What the worker owns, so the daemon's request means the same here as on the Spark engine (src/engines/qwen_image_tool.zig):

- Limits: `max_side` (default 1664) and `max_pixels` (default max_side squared) on the output, up to `edit_refs` input
  images (default 5, at most 5), steps 0 or the schedule's 8. Anything else is refused (invalid_request_error), never
  changed quietly. `guidance` and `negative_prompt` are not used, as on the Spark engine (Turbo runs without CFG).
- n images: one request a seed, `seed`, `seed+1`, … (`0.png`, `1.png`, …).
- Edits: input images in order (the first is the one edited, the others guide it), sent as MLX-Serve's `image` and
  `ref_images`. Without a size, the last image's shape at about one megapixel within the limits (the Spark tool's
  `editSize`). MLX-Serve floors an explicit edit size to multiples of 32, so other sizes are refused. Each input is
  resized to about one megapixel for conditioning; one past 4:1 or under 32 pixels a side is refused. WebP inputs
  are converted to PNG losslessly (MLX-Serve decodes PNG and JPEG).
- A mask (alpha 0 where the edit goes, as OpenAI's) keeps the first image wherever it is opaque: both are resized to
  the output (Lanczos, premultiplied for RGBA) and the mask's alpha blends the two (the Spark tool's `Keep`).
- Output: MLX-Serve's PNG with `prompt` and `seed` text chunks added (pixels untouched); the daemon adds the
  machine-generated label. RGBA when the option `alpha` is on (default: the VAE's own alpha, as the Spark engine).
- Memory: `resident` is what MLX-Serve holds once loaded and warmed (the larger of MLX's active bytes and the server's
  resident set). LocalRouter's scheduler is the only admission authority, so the server's own load pre-flight is off
  (--skip-mem-preflight); its per-request working-set check still refuses an edit that cannot fit (a 503 here).
- Cancel: SIGUSR1 closes the connection; MLX-Serve stops at the next denoise step and stays loaded.
- A server that dies takes the worker with it (no reply): the job fails with a 503 and the next request starts it again.
  The server is in the worker's process group, so the daemon's kill of the group stops it too.

Options (the tool's `options`): `server` (the mlx-serve binary; default <weights>/../mlx-serve-macos-arm64/mlx-serve),
`max_side`, `max_pixels`, `edit_refs`, `steps` (8), `alpha` (true), `w8a8` (false: MLX-Serve's lossy int8 x int8 DiT
linears stay off, the documented Mac baseline), `warmup` (true: a 256x256 image at load, so the first request is
not slower), `port` (0: a free one), `load_timeout_s` (600), `verify` (size: every pinned file present with its size;
none). `weights` is the pack's directory (fetch.sh).
"""

import base64
import http.client
import io
import json
import math
import os
import platform
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "pins-macos.json")

MAX_REFS = 5  # LocalRouter's API limit (request.Limits.max_references)
REF_PIXELS = 1024 * 1024  # each input is conditioned at about this many pixels (the pipeline's calculate_dimensions)
REF_SIDE = 2048  # the longest conditioned side: inputs past 4:1 are refused
MIN_VERSION = (26, 10, 2)  # the first MLX-Serve that runs the pack's own sample_sigmas schedule
OPTIONS = {"server", "max_side", "max_pixels", "edit_refs", "steps", "alpha", "w8a8", "warmup", "port",
           "load_timeout_s", "verify"}


def log(msg):
    print(f"qwen-mlx: {msg}", file=sys.stderr, flush=True)


class Refused(Exception):
    """A request this model cannot serve: invalid_request_error (the API's 400)."""


class Unavailable(Exception):
    """The server is busy or short of memory for this request: service_unavailable (503), retry later."""


class Cancelled(BaseException):
    """SIGUSR1 during a generate. Not an Exception, so no handler turns it into a server error."""


GENERATING = False


def on_cancel(signum, frame):
    if GENERATING:
        raise Cancelled()


# -- sizes, as the Spark tool computes them (src/engines/qwen_image_tool.zig, qwen_image/edit_prepare.zig) --------------

def dimensions(width, height, area):
    """calculate_dimensions: the shape of width x height at about `area` pixels, sides multiples of 32 (round half even)."""
    if width <= 0 or height <= 0 or area <= 0:
        raise ValueError("invalid dimensions")
    ratio = width / height
    w = round(math.sqrt(area * ratio) / 32) * 32
    h = round(w / ratio / 32) * 32
    if w < 32 or h < 32:
        raise ValueError("empty grid")
    return w, h


def edit_size(width, height, max_side, max_pixels):
    """An edit without a size: the last input's shape at about one megapixel, within max_side and max_pixels."""
    area = min(max_pixels, REF_PIXELS)
    while area >= 32 * 32:
        w, h = dimensions(width, height, area)
        if max(w, h) <= max_side and w * h <= max_pixels:
            return w, h
        area -= 32 * 32
    raise Refused("no size this model takes keeps the input's shape: send size")


# -- PNG text chunks --------------------------------------------------------------------------------------------------

def add_text(png, items):
    """The PNG with tEXt chunks (key, value) inserted right after IHDR; every other byte is kept."""
    if png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    end = 8 + 8 + struct.unpack(">I", png[8:12])[0] + 4
    chunks = b""
    for key, value in items:
        body = key.encode("latin-1") + b"\0" + value.encode("utf-8")
        chunks += struct.pack(">I", len(body)) + b"tEXt" + body
        chunks += struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF)
    return png[:end] + chunks + png[end:]


# -- the worker ----------------------------------------------------------------------------------------------------

class Options:
    def __init__(self, tool):
        o = tool.get("options") or {}
        unknown = sorted(set(o) - OPTIONS)
        if unknown:
            log(f"options not used by the worker: {unknown}")
        self.id = tool.get("id") or "qwen-image-2.1-turbo"
        self.weights = tool.get("weights") or ""
        if not self.weights:
            raise ValueError("weights (the MLX-Serve pack's directory) is required")
        self.server = o.get("server") or os.path.join(os.path.dirname(os.path.abspath(self.weights)),
                                                      "mlx-serve-macos-arm64", "mlx-serve")
        self.max_side = int(o.get("max_side", 1664)) // 16 * 16
        self.max_pixels = min(int(o.get("max_pixels", self.max_side * self.max_side)), self.max_side * self.max_side) // 256 * 256
        self.edit_refs = min(int(o.get("edit_refs", MAX_REFS)), MAX_REFS)
        self.steps = int(o.get("steps", 8))
        self.alpha = bool(o.get("alpha", True))
        self.w8a8 = bool(o.get("w8a8", False))
        self.warmup = bool(o.get("warmup", True))
        self.port = int(o.get("port", 0))
        self.load_timeout_s = float(o.get("load_timeout_s", 600))
        self.verify = o.get("verify", "size")
        if self.verify not in ("size", "none"):
            raise ValueError("verify must be size or none")
        if self.max_side < 256 or self.edit_refs < 0 or self.steps <= 0:
            raise ValueError("max_side must be at least 256, edit_refs and steps positive")


class Worker:
    def __init__(self, tool):
        self.opts = Options(tool)
        self.proc = None
        self.port = 0
        self.conn = None

    # -- load ----------------------------------------------------------------------------------------------------

    def load(self):
        try:
            self._check_host()
            self._check_weights()
            self._check_server()
            self._start()
            if self.opts.warmup:
                t = time.monotonic()
                self._image({"size": "256x256", "seed": 0, "steps": self.opts.steps, "prompt": "warm up"}, None)
                log(f"warm-up image in {time.monotonic() - t:.1f} s")
            resident = self._resident()
            log(f"loaded: resident {resident >> 20} MiB")
            return {"ok": True, "resident": resident, "cancel": "signal"}
        except (Refused, ValueError, OSError, RuntimeError) as e:
            log(f"load failed: {e}")
            self._stop()
            return {"ok": False, "error": str(e), "type": "server_error"}

    def _check_host(self):
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("this worker runs on macOS on Apple Silicon only (MLX)")

    def _check_weights(self):
        """The pinned files are there with their sizes (fetch.sh checked their digests), and the schedule has `steps` nodes."""
        w = self.opts.weights
        if self.opts.verify == "size":
            pins = read_json(PINS)
            for f in pins["selected_files"]:
                p = os.path.join(w, f["path"])
                if not os.path.isfile(p) or os.path.getsize(p) != f["size_bytes"]:
                    raise RuntimeError(f"{p}: missing or not the pinned size ({f['size_bytes']} bytes): run fetch.sh")
        try:
            sigmas = read_json(os.path.join(w, "model_index.json")).get("sample_sigmas")
        except (OSError, ValueError) as e:
            raise RuntimeError(f"{w}/model_index.json: {e}")
        if not isinstance(sigmas, list) or len(sigmas) != self.opts.steps:
            raise RuntimeError(f"{w}: the pack's sample_sigmas has {len(sigmas) if isinstance(sigmas, list) else 0} "
                               f"steps, the tool's steps option {self.opts.steps}")

    def _check_server(self):
        try:
            out = subprocess.run([self.opts.server, "--version"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"{self.opts.server}: {e}: run fetch.sh, or set the option server")
        m = re.search(r"mlx-serve (\d+)\.(\d+)\.(\d+)", out)
        if not m:
            raise RuntimeError(f"{self.opts.server} --version said {out.strip()!r}, not an MLX-Serve version")
        version = tuple(int(x) for x in m.groups())
        if version < MIN_VERSION:
            raise RuntimeError(f"MLX-Serve {'.'.join(map(str, version))} runs the base model's schedule on Turbo: "
                               f"{'.'.join(map(str, MIN_VERSION))} or newer is needed (fetch.sh)")

    def _start(self):
        self.port = self.opts.port or free_port()
        argv = [self.opts.server, "--serve", "--model", self.opts.weights, "--host", "127.0.0.1",
                "--port", str(self.port), "--skip-mem-preflight", "--w8a8" if self.opts.w8a8 else "--no-w8a8"]
        log(f"starting {' '.join(argv)}")
        t = time.monotonic()
        # stdout is the protocol: the server's output goes to the log. Same process group as the worker.
        self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
        deadline = t + self.opts.load_timeout_s
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"mlx-serve exited with {self.proc.returncode} while loading (see the log)")
            if time.monotonic() > deadline:
                raise RuntimeError(f"mlx-serve was not ready within {self.opts.load_timeout_s:.0f} s")
            try:
                status, body = self._get("/v1/models")
                if status == 200 and any(m.get("loaded") and m.get("state") == "ready" for m in json.loads(body).get("data", [])):
                    break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            time.sleep(0.25)
        log(f"mlx-serve ready on port {self.port} in {time.monotonic() - t:.1f} s")

    def _resident(self):
        active = 0
        try:
            status, body = self._get("/props")
            if status == 200:
                active = int(json.loads(body).get("memory", {}).get("active_bytes", 0))
        except (OSError, ValueError, http.client.HTTPException):
            pass
        return max(active, rss(self.proc.pid))

    # -- generate ------------------------------------------------------------------------------------------------

    def generate(self, cmd, progress):
        start = time.monotonic()
        req = (cmd.get("request") or {}).get("image")
        if req is None:
            raise Refused("this model makes images: send an image request")
        job = cmd.get("dir") or ""
        o = self.opts
        refs = req.get("references") or []
        mask = req.get("mask")
        width, height = int(req.get("width", 1024)), int(req.get("height", 1024))
        n, seed, steps = int(req.get("n", 1)), int(req.get("seed", 0)), int(req.get("steps", 0))
        prompt = req.get("prompt") or ""

        if len(refs) > o.edit_refs:
            raise Refused(f"model '{o.id}' edits with up to {o.edit_refs} input image{'' if o.edit_refs == 1 else 's'}")
        if mask is not None and not refs:
            raise Refused("a mask needs an input image to edit")
        if steps not in (0, o.steps):
            raise Refused(f"model '{o.id}' runs its checkpoint's {o.steps}-step schedule: omit steps or send {o.steps}")
        if not prompt:
            raise Refused("prompt is required")
        if n < 1 or n > 4:
            raise Refused("n must be 1 to 4")

        inputs = [self._input(job, name) for name in refs]
        if width == 0 and height == 0:
            if not inputs:
                raise Refused(f"model '{o.id}' needs a size")
            width, height = edit_size(*inputs[-1][1].size, o.max_side, o.max_pixels)
        if max(width, height) > o.max_side:
            raise Refused(f"model '{o.id}' takes sizes up to {o.max_side} pixels a side")
        if width * height > o.max_pixels:
            raise Refused(f"model '{o.id}' takes at most {o.max_pixels} pixels (width x height), such as 1024x1024 or 1360x768")
        if width % 16 or height % 16 or width < 256 or height < 256:
            raise Refused(f"size {width}x{height}: each side must be a multiple of 16, at least 256")
        if inputs and (width % 32 or height % 32):
            raise Refused(f"size {width}x{height}: an edit's sides must be multiples of 32 on this model (MLX-Serve)")
        keep = Keep(job, mask, inputs[0][1], width, height) if mask is not None else None

        body = {"size": f"{width}x{height}", "steps": o.steps, "stream": True}
        if o.alpha:
            body["transparent"] = True
        if inputs:
            body["mode"] = "edit"
            body["image"] = inputs[0][0]
            if len(inputs) > 1:
                body["ref_images"] = [b for b, _ in inputs[1:]]
        files = []
        for i in range(n):
            s = (seed + i) % (1 << 64)
            # the prompt last: MLX-Serve finds the other fields by their first key match in the body
            png = self._image({**body, "seed": s, "prompt": prompt}, progress)
            if keep is not None:
                png = keep.apply(png)
            name = f"{i}.png"
            with open(os.path.join(job, name), "wb") as f:
                f.write(add_text(png, [("prompt", prompt), ("seed", str(s))]))
            files.append(name)
        return {"ok": True, "files": files, "seed": seed, "ms": int((time.monotonic() - start) * 1000)}

    def _input(self, job, name):
        """An input image as MLX-Serve takes it (base64 PNG or JPEG) and as Pillow reads it, after the size checks."""
        path = os.path.join(job, name)
        if os.path.dirname(os.path.normpath(name)) or name.startswith("."):
            raise Refused("references must name files in the job directory")
        try:
            with open(path, "rb") as f:
                data = f.read()
            im = Image.open(io.BytesIO(data))
            im.load()
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            raise Refused(f"an input image could not be read ({e}): send a PNG, JPEG or WebP")
        try:
            if min(im.size) < 32:
                raise ValueError("too small")
            w, h = dimensions(im.width, im.height, REF_PIXELS)
        except ValueError:
            raise Refused(f"an input image of {im.width}x{im.height} cannot be edited: send one with sides of at least 32 pixels")
        if max(w, h) > REF_SIDE:
            raise Refused(f"an input image of {im.width}x{im.height} is too long and thin to edit: at most 4:1")
        if im.format not in ("PNG", "JPEG"):
            buf = io.BytesIO()
            im.save(buf, "PNG")  # lossless, alpha kept
            data = buf.getvalue()
        return base64.b64encode(data).decode("ascii"), im

    def _image(self, body, progress):
        """One image from MLX-Serve: its PNG bytes. Streams progress; a cancel closes the connection."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=None)
        self.conn = conn
        try:
            conn.request("POST", "/v1/images/generations", body=json.dumps(body), headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200:
                msg = server_message(r.read())
                if r.status == 400 and "GPU memory" in msg:
                    raise Unavailable(msg)
                if r.status == 400:
                    raise Refused(msg)
                if r.status == 503:
                    raise Unavailable(msg)
                raise RuntimeError(f"mlx-serve answered {r.status}: {msg}")
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                ev = json.loads(line[5:])
                kind = ev.get("type")
                if kind == "progress" and progress is not None:
                    progress({"phase": "denoise", "step": ev.get("step", 0), "of": ev.get("total", 0)})
                elif kind == "complete":
                    return base64.b64decode(ev["data"][0]["b64_json"])
                elif kind == "error":
                    raise RuntimeError(f"mlx-serve: {ev.get('message', 'generation failed')}")
            raise RuntimeError("mlx-serve ended the stream without an image")
        finally:
            self.conn = None
            conn.close()

    def _get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def server_alive(self):
        return self.proc is not None and self.proc.poll() is None

    # -- exit ------------------------------------------------------------------------------------------------------

    def _stop(self):
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log("mlx-serve did not stop in 30 s; killing it")
            p.kill()
            p.wait()


class Keep:
    """A mask's composite: the first input at the output's size, kept wherever the mask is opaque (OpenAI's edits:
    alpha 0 marks where the edit goes). The mask must be the first image's size."""

    def __init__(self, job, name, first, width, height):
        path = os.path.join(job, name)
        if os.path.dirname(os.path.normpath(name)) or name.startswith("."):
            raise Refused("mask must name a file in the job directory")
        try:
            m = Image.open(path)
            m.load()
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            raise Refused(f"the mask could not be used ({e}): send a PNG of the first image's size, transparent where the edit goes")
        if m.size != first.size:
            raise Refused(f"the mask is {m.width}x{m.height}, the first image {first.width}x{first.height}: "
                          "send a PNG of the first image's size, transparent where the edit goes")
        self.original = first.convert("RGBA").resize((width, height), Image.LANCZOS)  # Pillow premultiplies RGBA
        self.alpha = m.convert("RGBA").getchannel("A").resize((width, height), Image.LANCZOS)

    def apply(self, png):
        edited = Image.open(io.BytesIO(png))
        edited.load()
        original = self.original if edited.mode == "RGBA" else self.original.convert(edited.mode)
        out = Image.composite(original, edited, self.alpha)
        buf = io.BytesIO()
        out.save(buf, "PNG", compress_level=1)
        return buf.getvalue()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def server_message(raw):
    try:
        e = json.loads(raw).get("error")
        return (e.get("message") if isinstance(e, dict) else str(e)) or "error"
    except (ValueError, AttributeError):
        return raw.decode("utf-8", "replace").strip() or "error"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rss(pid):
    """A process's resident set in bytes (unified memory: MLX's Metal buffers are in it), 0 when unreadable."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip() or 0) * 1024
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def watch_parent(worker):
    """Stop the server and exit when the daemon that started us is gone (LOCALROUTER_PARENT; 0 or unset: no check)."""
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
    global GENERATING
    try:
        try:
            GENERATING = True  # inside the try: a signal right after it is still caught
            return worker.generate(cmd, lambda p: send({"progress": p}))
        finally:
            GENERATING = False
    except Cancelled:
        log("generate cancelled; mlx-serve stops at its next step")
        return {"ok": False, "type": "cancelled", "error": "cancelled"}
    except Refused as e:
        return {"ok": False, "type": "invalid_request_error", "error": str(e)}
    except Unavailable as e:
        return {"ok": False, "type": "service_unavailable", "error": str(e)}
    except (OSError, RuntimeError, ValueError, KeyError, http.client.HTTPException) as e:
        if not worker.server_alive():
            return {"gone": True, "error": f"mlx-serve is gone ({e})"}
        log(f"generate failed: {e}")
        return {"ok": False, "type": "server_error", "error": str(e)}


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
            stopping = threading.Thread(target=worker._stop)
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
