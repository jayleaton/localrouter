#!/usr/bin/env python3
"""LocalRouter `cmd` worker for Wan 2.2 TI2V-5B (Wan-AI/Wan2.2-TI2V-5B-Diffusers): text or image to video, no audio.

Speaks the worker protocol (docs/ARCHITECTURE.md, "Any program"): one JSON object a line on stdin and stdout; stdout
carries protocol lines only (anything else that writes to fd 1 goes to stderr), logs go to stderr.

    {"op":"load"}      -> {"ok":true,"resident":<bytes>,"cancel":"signal"}
    {"op":"generate","id":"vid_…","dir":"<job dir>","request":{"video":{"prompt":"…","width":704,"height":1280,
        "seconds":5,"fps":24,"seed":7,"steps":0,"guidance":0,"audio":false,"first_frame":"first_frame_0.png"}}}
                       -> {"progress":{"phase":"encode|denoise|decode|write","step":k,"of":n}}*
                       -> {"ok":true,"files":["video.mp4"],"seed":7,"ms":…}
                          or {"ok":false,"type":"invalid_request_error"|"cancelled"|"service_unavailable"|"server_error","error":…}
    {"op":"exit"} / EOF -> exit

SIGUSR1 during a generate ends it at the next denoising step with {"ok":false,"type":"cancelled"}; the weights stay
loaded. After every job and at exit the worker logs the GPU memory it still holds (allocated and reserved).

The model runs through Diffusers' WanPipeline (text to video) and WanImageToVideoPipeline (a first frame), which share
one set of modules. The model makes 4k+1 frames: a clip of `seconds` x 24 frames is made one frame longer and the last
frame is dropped. A first frame of another shape is cut to the clip's shape (centred) and resized, never stretched.
The MP4 is H.264 (yuv420p) without an audio track; the daemon adds the machine-generated label afterwards.
Nothing contacts the network: HF_HUB_OFFLINE is forced and every path is local.

Config (LOCALROUTER_TOOL): `weights` is the Diffusers directory fetch.sh fills. Options: `steps` (default 50), `guidance`
(default 5.0), `negative_prompt` (default the model card's), `transformer_storage` (`fp8`: the transformer's weights
are stored in float8 e4m3 and computed in bf16, which keeps a 16 GB GPU from spilling into shared memory; `bf16`:
as loaded), `offload` (`model`: each module moves to the GPU only while it runs; `none`: everything stays on the GPU),
`vae_tiling` (default true), `crf` (H.264 quality, default 16). The daemon's own limits (`max_pixels`, `max_side`,
`size_multiple`, `default_size`, `max_seconds`, `fps`, `audio`) are checked again here.
"""

import gc
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import threading
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROTOCOL = sys.stdout  # main() moves it off fd 1, so stray prints from libraries go to stderr

FPS = 24
OPTIONS = {"steps", "guidance", "negative_prompt", "transformer_storage", "offload", "vae_tiling", "crf"}
DAEMON_OPTIONS = {"max_pixels", "max_side", "size_multiple", "default_size", "max_seconds", "fps", "audio"}
# The negative prompt of the official Wan 2.2 configs (wan/configs/shared_config.py, sample_neg_prompt).
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
            "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def log(msg):
    print(f"wan: {msg}", file=sys.stderr, flush=True)


def send(obj):
    PROTOCOL.write(json.dumps(obj) + "\n")
    PROTOCOL.flush()


class Refused(Exception):
    """A request this model cannot serve: invalid_request_error (the API's 400)."""


class Cancelled(BaseException):
    """SIGUSR1 during a generate. Not an Exception, so no handler turns it into a server error."""


GENERATING = False
CANCEL = threading.Event()


def on_cancel(signum, frame):
    if GENERATING:
        CANCEL.set()


def frames_for(seconds, fps=FPS):
    """(frames the model makes, frames kept): the model makes 4k+1 frames, the clip keeps seconds x fps."""
    keep = seconds * fps
    return 4 * math.ceil((keep - 1) / 4) + 1, keep


def fit(image, width, height):
    """`image` cut to width:height (centred) and resized to width x height with Lanczos."""
    from PIL import Image

    image = image.convert("RGB")
    w, h = image.size
    if w * height > h * width:
        cw, ch = round(h * width / height), h
    else:
        cw, ch = w, round(w * height / width)
    left, top = (w - cw) // 2, (h - ch) // 2
    image = image.crop((left, top, left + cw, top + ch))
    return image if image.size == (width, height) else image.resize((width, height), Image.Resampling.LANCZOS)


def write_mp4(frames, path, fps, crf):
    """`frames` (uint8 [n, h, w, 3]) as H.264 yuv420p at `fps`, through the system ffmpeg."""
    n, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", path]
    p = subprocess.run(cmd, input=frames.tobytes(), stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.decode(errors='replace').strip()}")


def host_rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class Engine:
    """The Diffusers pipelines. Separate from Worker so the protocol tests run without torch or weights."""

    def __init__(self, weights, opts):
        self.weights, self.opts = weights, opts
        self.t2v = self.i2v = None

    def load(self):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel

        weights = self.weights
        # The transformer is published in fp32 (20 GB). Loaded on the host, its conversion peaks at 28 GiB of RAM; loaded
        # shard by shard onto the GPU (10 GB in bf16) and cast to its storage type there, the host peaks at 5 GiB.
        transformer = WanTransformer3DModel.from_pretrained(weights, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                            device_map="cuda")
        if self.opts.get("transformer_storage", "fp8") == "fp8":
            transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        if self.opts.get("offload", "model") == "model":
            transformer.to("cpu")
            torch.cuda.empty_cache()
        vae = AutoencoderKLWan.from_pretrained(weights, subfolder="vae", torch_dtype=torch.float32)
        self.t2v = WanPipeline.from_pretrained(weights, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
        c = self.t2v.components  # the same modules (from_pipe copies the text encoder: 30 GiB of host memory)
        self.i2v = WanImageToVideoPipeline(tokenizer=c["tokenizer"], text_encoder=c["text_encoder"], vae=vae, scheduler=c["scheduler"],
                                           transformer=transformer, expand_timesteps=self.t2v.config.expand_timesteps)
        self.big = [self.t2v.text_encoder, self.t2v.transformer, vae]
        if self.opts.get("offload", "model") == "model":
            # Diffusers' chain onloads each module and offloads only the one before it in its sequence, so the VAE (used
            # first, for the first frame) would sit on the GPU through denoising, and the text encoder through that encode.
            # On a 16 GB card that pages and runs 3x slower. Here each of the three takes the GPU alone.
            self.i2v.enable_model_cpu_offload()
            for m in self.big[:2]:
                m.register_forward_pre_hook(lambda mod, args: self.alone(mod))
            for name in ("encode", "decode"):
                method = getattr(vae, name)
                setattr(vae, name, lambda *a, _m=method, **k: (self.alone(vae), _m(*a, **k))[1])
        else:
            self.t2v.to("cuda")
        if self.opts.get("vae_tiling", True):
            vae.enable_tiling()
        torch.cuda.synchronize()

    def alone(self, module):
        """Every big module but `module` off the GPU; the offload hook then brings `module` on."""
        import torch

        moved = False
        for m in self.big:
            if m is not module and m.device.type == "cuda":
                m.to("cpu")
                moved = True
        if moved:
            torch.cuda.empty_cache()

    def release_gpu(self):
        """After every job, done, failed or cancelled: a fresh scheduler (UniPC keeps the job's tensors), the big
        modules off the GPU with offload, and the cache emptied even when Diffusers has already moved them to the CPU."""
        old = self.t2v.scheduler
        self.t2v.scheduler = self.i2v.scheduler = type(old).from_config(old.config)
        del old
        if self.opts.get("offload", "model") == "model":
            self.alone(None)
        self.flush("idle")

    def unload(self):
        self.t2v = self.i2v = None
        self.big = []
        self.flush("unloaded")

    def flush(self, when):
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        log(f"{when}: GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
            f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved")

    def resident(self):
        import torch

        return host_rss() + torch.cuda.memory_allocated()

    def generate(self, v, image, made, steps, guidance, negative, on_step):
        import torch

        torch.cuda.reset_peak_memory_stats()

        def callback(pipe, i, t, kwargs):
            if CANCEL.is_set():
                raise Cancelled()
            on_step(i + 1)
            return kwargs

        args = dict(prompt=v["prompt"], negative_prompt=negative, height=v["height"], width=v["width"], num_frames=made,
                    num_inference_steps=steps, guidance_scale=guidance, output_type="np",
                    generator=torch.Generator("cpu").manual_seed(v["seed"]), callback_on_step_end=callback)
        failed = None
        try:
            out = self.t2v(**args) if image is None else self.i2v(image=image, **args)
        except BaseException as e:  # noqa: BLE001  re-raised below, without the frames that hold the job's tensors
            failed = e.with_traceback(None)
        self.release_gpu()  # nothing stays on the GPU between jobs: the desktop shares it
        if failed is not None:
            raise failed
        frames = out.frames[0]
        return (frames * 255).round().clip(0, 255).astype("uint8"), torch.cuda.max_memory_allocated()


def _int(v, name):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise Refused(f"{name} must be a whole number")
    return v


class Worker:
    def __init__(self, tool, engine=None):
        opts = tool.get("options") or {}
        unknown = set(opts) - OPTIONS - DAEMON_OPTIONS
        if unknown:
            raise ValueError(f"unknown options: {', '.join(sorted(unknown))}")
        if not tool.get("weights"):
            raise ValueError("weights (the Diffusers directory) is required")
        if opts.get("transformer_storage", "fp8") not in ("fp8", "bf16"):
            raise ValueError("transformer_storage must be fp8 or bf16")
        if opts.get("offload", "model") not in ("model", "none"):
            raise ValueError("offload must be model or none")
        self.opts, self.weights = opts, tool["weights"]
        self.engine = engine or Engine(tool["weights"], opts)
        self.loaded = False

    def load(self):
        if not self.loaded:
            index = os.path.join(self.weights, "model_index.json")
            if not os.path.isfile(index):
                return {"ok": False, "type": "server_error", "error": f"no Diffusers weights at {self.weights} (run fetch.sh)"}
            t0 = time.monotonic()
            try:
                self.engine.load()
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "type": "server_error", "error": f"load failed: {type(e).__name__}: {e}"}
            self.loaded = True
            log(f"loaded in {time.monotonic() - t0:.1f}s")
        return {"ok": True, "resident": self.engine.resident(), "cancel": "signal"}

    def unload(self):
        if self.loaded:
            self.engine.unload()
            self.loaded = False

    def parse(self, cmd):
        req = (cmd.get("request") or {}).get("video")
        if not isinstance(req, dict):
            raise Refused("this worker makes videos: the request must be {\"video\": {...}}")
        v = {"prompt": req.get("prompt") or "", "negative_prompt": req.get("negative_prompt") or "",
             "width": _int(req.get("width", 0), "width"), "height": _int(req.get("height", 0), "height"),
             "seconds": _int(req.get("seconds", 5), "seconds"), "fps": _int(req.get("fps", FPS), "fps"),
             "seed": _int(req.get("seed", 0), "seed"), "steps": _int(req.get("steps", 0), "steps"),
             "guidance": req.get("guidance") or 0, "first_frame": req.get("first_frame")}
        if not v["prompt"].strip():
            raise Refused("prompt is required")
        if req.get("audio"):
            raise Refused("this model makes no audio: omit audio or send false")
        if v["fps"] != FPS:
            raise Refused(f"this model makes {FPS} fps only")
        if v["width"] % 32 or v["height"] % 32 or v["width"] == 0 or v["height"] == 0:
            raise Refused("width and height must be multiples of 32")
        o = self.opts
        if o.get("max_side") and max(v["width"], v["height"]) > o["max_side"]:
            raise Refused(f"at most {o['max_side']} pixels a side")
        if o.get("max_pixels") and v["width"] * v["height"] > o["max_pixels"]:
            raise Refused(f"at most {o['max_pixels']} pixels a frame")
        if v["seconds"] == 0 or (o.get("max_seconds") and v["seconds"] > o["max_seconds"]):
            raise Refused(f"seconds must be 1 to {o.get('max_seconds', 20)}")
        if not isinstance(v["guidance"], (int, float)) or v["guidance"] < 0:
            raise Refused("guidance must be a number of at least 0")
        return v

    def generate(self, cmd, progress):
        global GENERATING
        v = self.parse(cmd)
        job = cmd.get("dir") or ""
        image = None
        if v["first_frame"]:
            from PIL import Image

            path = os.path.join(job, os.path.basename(v["first_frame"]))
            try:
                with Image.open(path) as im:
                    image = fit(im, v["width"], v["height"])
            except OSError as e:
                raise Refused(f"the first frame is not an image this worker can read: {e}") from e
        if not self.loaded:
            r = self.load()
            if not r["ok"]:
                return r
        steps = v["steps"] or int(self.opts.get("steps", 50))
        guidance = float(v["guidance"] or self.opts.get("guidance", 5.0))
        negative = v["negative_prompt"] or self.opts.get("negative_prompt", NEGATIVE)
        made, keep = frames_for(v["seconds"])
        t0 = time.monotonic()
        CANCEL.clear()
        GENERATING = True
        try:
            progress({"phase": "encode", "step": 0, "of": 1})
            frames, peak = self.engine.generate(v, image, made, steps, guidance, negative,
                                                lambda k: progress({"phase": "denoise", "step": k, "of": steps}))
        finally:
            GENERATING = False
        progress({"phase": "write", "step": 0, "of": 1})
        out = os.path.join(job, "video.mp4")
        write_mp4(frames[:keep], out, FPS, int(self.opts.get("crf", 16)))
        ms = int((time.monotonic() - t0) * 1000)
        log(f"{v['width']}x{v['height']} {keep} frames, {steps} steps, {'i2v' if image else 't2v'}: {ms / 1000:.1f}s, "
            f"peak GPU allocation {peak / 2**30:.2f} GiB")
        return {"ok": True, "files": ["video.mp4"], "seed": v["seed"], "ms": ms}


def run_generate(worker, cmd):
    return run_generate_with(worker, cmd, lambda p: send({"progress": p}))


def run_generate_with(worker, cmd, progress):
    try:
        return worker.generate(cmd, progress)
    except Cancelled:
        return {"ok": False, "type": "cancelled", "error": "cancelled"}
    except Refused as e:
        return {"ok": False, "type": "invalid_request_error", "error": str(e)}
    except Exception as e:  # noqa: BLE001  the model failed: the reply says why, the worker stays up
        msg = f"{type(e).__name__}: {e}"
        log(f"generate failed: {msg}")
        if "out of memory" in str(e).lower():
            return {"ok": False, "type": "service_unavailable", "error": msg}
        return {"ok": False, "type": "server_error", "error": msg}


def watch_parent():
    """Exit when the daemon that started us is gone (LOCALROUTER_PARENT; 0 or unset: no check)."""
    parent = int(os.environ.get("LOCALROUTER_PARENT", "0") or 0)
    if parent <= 0:
        return

    def run():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                log("the daemon is gone; exiting")
                os._exit(0)

    threading.Thread(target=run, daemon=True).start()


def main():
    global PROTOCOL
    PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    signal.signal(signal.SIGUSR1, on_cancel)
    log(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        worker, problem = Worker(json.loads(os.environ.get("LOCALROUTER_TOOL", "{}"))), None
    except (ValueError, TypeError) as e:
        worker, problem = None, str(e)
    watch_parent()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            cmd = json.loads(line)
        except ValueError:
            send({"ok": False, "type": "invalid_request_error", "error": "not a JSON line"})
            continue
        op = cmd.get("op")
        if op == "exit":
            break
        if worker is None:
            send({"ok": False, "type": "server_error", "error": f"bad tool config: {problem}"})
        elif op == "load":
            send(worker.load())
        elif op == "generate":
            send(run_generate(worker, cmd))
        else:
            send({"ok": False, "type": "invalid_request_error", "error": f"unknown op {op!r}"})
    if worker is not None:
        worker.unload()


if __name__ == "__main__":
    main()
    sys.stderr.flush()
    os._exit(0)  # no interpreter teardown of the loaded models
