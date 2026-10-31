#!/usr/bin/env python3
"""LocalRouter `cmd` worker for Idol (speridlabs/idol): one image to a 3D Gaussian splat (PLY).

Speaks the worker protocol (docs/ARCHITECTURE.md, "Any program"): one JSON object a line on stdin and stdout; stdout
carries protocol lines only (anything else that writes to fd 1 goes to stderr), logs go to stderr.

    {"op":"load"}      -> {"ok":true,"resident":<bytes>,"cancel":"signal"}
    {"op":"generate","id":"a3d_…","dir":"<job dir>","request":{"asset3d":{"image":"image_0.png",
        "input_mode":"synthetic"|"real","format":"ply","seed":42,"steps":50,"guidance":3.0,"num_gaussians":262144}}}
                       -> {"progress":{"phase":"preprocess|encode|denoise|decode|write","step":k,"of":n}}*
                       -> {"ok":true,"files":["asset.ply","asset.json","input.png"],"seed":42,"ms":…}
                          or {"ok":false,"type":"invalid_request_error"|"cancelled"|"service_unavailable"|"server_error","error":…}
    {"op":"stats"}     -> {"ok":true,"footprint":…,"footprint_peak":…,…} (measurement scripts only)
    {"op":"exit"} / EOF -> exit (the process ends, so every weight is released)

SIGUSR1 during a generate ends it at the next Euler step (or phase boundary) with {"ok":false,"type":"cancelled"} and
the weights stay loaded; with no generate in hand it is ignored.

The model is the unmodified idol-sdk (pinned commit, pyproject.toml) driven stage by stage, exactly as
IdolPipeline.__call__ runs them (preprocess, seed, encoders, noise, Euler sampler, octree + Gaussian decoders), so a
request gives the SDK's own PLY bit for bit. Two loader differences, neither in the SDK's code:
- DINOv3 ViT-H+/16 comes from the ungated timm checkpoint (timm/vit_huge_plus_patch16_dinov3.lvd1689m), remapped at
  load into Transformers' DINOv3 graph with the official model's config (`dinov3_from_timm`; no derived file on disk).
  Every parameter is byte-identical to the gated official checkpoint and the features are bit-identical (research).
- The FLUX.2 autoencoder comes from black-forest-labs/FLUX.2-klein-4B's vae/ (Apache-2.0, ungated, BF16): the loaded
  BF16 state is bit-identical to FLUX.2-dev's vae/, the precision Idol runs it at.
Nothing contacts the network: HF_HUB_OFFLINE is forced and every path is local.

Backends (option `backend`):
- pytorch (default): all of the above in this process.
- tensorfold: the SDK's preprocess (BiRefNet background removal and the crop) runs here in PyTorch; everything after it
  (both encoders, the denoiser, the decoders, the PLY) runs in TensorFold's native Metal `tf-idol-generate`, one process
  a job (option `tf_idol_generate`: its path), from the same weights. Only BiRefNet stays loaded here. SIGUSR1 ends the
  child at once; progress comes by phase (the child reports its stages, not its steps). Its starting noise is torch's
  CPU-seeded randn and its other draws are native, so its PLY is not the SDK's bit for bit: the same model, sampled
  differently (TensorFold docs/recipes/idol.md compares the two). Option `preprocess` names where preprocessing runs:
  pytorch, the only one until TensorFold's BiRefNet lands; `Preprocess` is the one place that changes then.

Config (LOCALROUTER_TOOL): `weights` is the directory fetch.sh fills (idol/, dinov3/, flux2-vae/; pins.json). Options:
backend, tf_idol_generate, preprocess (above),
device (auto: cuda, else mps), optimize (false; CUDA only: the SDK's compiled denoiser + batched CFG), min_gaussians,
max_gaussians, gaussian_multiple, default_gaussians, default_steps, default_guidance, max_steps, max_guidance,
input_modes, formats, verify (size: every pinned file present with its size; none), memory_stats (false: sample the
torch allocator during a generate, for {"op":"stats"}).
"""

import ctypes
import gc
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

os.environ["HF_HUB_OFFLINE"] = "1"  # never the network: every file is local
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")  # an op missing on MPS is an error, never a silent CPU run

if __name__ == "__main__":
    PROTO = os.fdopen(os.dup(1), "w", buffering=1)  # the protocol stream; fd 1 itself now points at stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
else:  # imported by test_worker.py: it reads the replies through send()
    PROTO = None

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "pins.json")

SDK = {"repo": "speridlabs/idol-sdk", "commit": "4702a7f00850e34b6e83a2170cc123c3e9570744"}
COMPONENTS = [
    {"role": "denoiser, decoders, background removal", "repo": "speridlabs/idol",
     "revision": "2731c2291fc65a797c5cb627a5e5526fc1490902", "license": "Apache-2.0"},
    {"role": "image encoder (DINOv3 ViT-H+/16)", "repo": "timm/vit_huge_plus_patch16_dinov3.lvd1689m",
     "revision": "dd0addc09788111fa893d24799a44744ba022eee", "license": "DINOv3 License"},
    {"role": "image encoder (FLUX.2 autoencoder)", "repo": "black-forest-labs/FLUX.2-klein-4B", "path": "vae/",
     "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294", "license": "Apache-2.0"},
]
# facebook/dinov3-vith16plus-pretrain-lvd1689m@c807c9ee config.json: the graph Idol runs DINOv3 in (its weights are
# the timm file, remapped)
DINOV3_CONFIG = {
    "hidden_size": 1280, "intermediate_size": 5120, "num_hidden_layers": 32, "num_attention_heads": 20,
    "patch_size": 16, "num_register_tokens": 4, "num_channels": 3, "image_size": 224, "hidden_act": "silu",
    "use_gated_mlp": True, "layer_norm_eps": 1e-05, "layerscale_value": 1.0, "rope_theta": 100.0,
    "query_bias": True, "key_bias": False, "value_bias": True, "proj_bias": True, "mlp_bias": True,
    "pos_embed_rescale": 2.0, "pos_embed_shift": None, "pos_embed_jitter": None, "attention_dropout": 0.0,
    "drop_path_rate": 0.0, "initializer_range": 0.02,
}

CROP_MARGIN = {"synthetic": 1.2, "real": 1.7}  # idol.preprocess.CROP_MARGIN, the SDK's `mode`
RESCALE_T = 3.0
FILES = ["asset.ply", "asset.json", "input.png"]
OPTIONS = {"weights", "backend", "tf_idol_generate", "preprocess", "device", "optimize", "min_gaussians", "max_gaussians", "gaussian_multiple",
           "default_gaussians", "default_steps", "default_guidance", "max_steps", "max_guidance", "input_modes",
           "formats", "verify", "memory_stats"}
DAEMON_OPTIONS = {"working_bytes_by_gaussians"}  # the scheduler reads it; the worker leaves it alone


def log(msg):
    print(f"idol: {msg}", file=sys.stderr, flush=True)


class Refused(Exception):
    """A request this model cannot serve: invalid_request_error (the API's 400)."""


class Cancelled(BaseException):
    """SIGUSR1 during a generate. Not an Exception, so no handler turns it into a server error."""


GENERATING = False
CANCEL = threading.Event()


def on_cancel(signum, frame):
    """SIGUSR1: honoured at the generate's next check (Euler step or phase boundary); ignored while idle."""
    if GENERATING:
        CANCEL.set()


def check_cancel():
    if CANCEL.is_set():
        raise Cancelled()


# -- memory ------------------------------------------------------------------------------------------------------

class _RusageInfoV4(ctypes.Structure):
    # <sys/resource.h> struct rusage_info_v4: a 16-byte uuid, then uint64 fields (padded generously)
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("v", ctypes.c_uint64 * 64)]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True) if sys.platform == "darwin" else None
_RESIDENT, _PHYS_FOOTPRINT, _LIFETIME_MAX_PHYS_FOOTPRINT = 6, 7, 28  # indices into v


def footprint(pid=None):
    """(physical footprint, its lifetime peak) in bytes. macOS: proc_pid_rusage, which counts Metal (MPS) buffers on
    unified memory. Linux: the resident set (CUDA memory is added by the engine). (0, 0) when unreadable."""
    if _libc is not None:
        info = _RusageInfoV4()
        if _libc.proc_pid_rusage(pid or os.getpid(), 4, ctypes.byref(info)) != 0:  # RUSAGE_INFO_V4
            return 0, 0
        return int(info.v[_PHYS_FOOTPRINT]), int(info.v[_LIFETIME_MAX_PHYS_FOOTPRINT])
    try:
        with open(f"/proc/{pid or 'self'}/status") as f:
            fields = dict(line.split(":", 1) for line in f if ":" in line)
        kib = lambda k: int(fields.get(k, "0 kB").split()[0]) * 1024  # noqa: E731
        return kib("VmRSS"), kib("VmHWM")
    except (OSError, ValueError):
        return 0, 0


# -- the frame and the camera ---------------------------------------------------------------------------------------

# idol.gaussian.Gaussian.save_ply turns the model's z-up object "to y-up" with a +90 degree rotation about x:
# PLY (x, y, z) = model (x, -z, y), and each rotation is left-multiplied by that rotation's quaternion. It takes the
# model's +z (the object's up) to PLY -y. The camera below is put through the same map, so it is in the PLY's frame.
FROM_MODEL = ("PLY (x, y, z) = model (x, -z, y): a +90 degree rotation about x (idol-sdk gaussian.py save_ply); "
              "rotations q_ply = q_x(+90 deg) * q_model; the model is z-up, so the object's up is PLY -y")


def model_to_ply(v):
    x, y, z = v
    return [float(x), float(-z), float(y)]


def camera_in_frame(yaw, pitch, radius):
    """The predicted input camera, in the PLY's frame: (position, up). Isolated so a rendering check can correct it.

    idol-sdk camera.py decodes the camera latent's direction d (normalised first three values, model frame) as
    yaw = atan2(dx, dy), pitch = asin(dz); radius = 1 / inverse distance. So d = (sin(yaw) cos(pitch),
    cos(yaw) cos(pitch), sin(pitch)), and the camera sits at radius * d looking at the origin, with the model's +z (the
    object's up) as its up. Both then go through model_to_ply."""
    d = (math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch))
    position = model_to_ply([radius * c for c in d])
    up = model_to_ply((0.0, 0.0, 1.0))
    return position, up


def camera_block(cam):
    """The manifest's camera from the SDK's decode_camera dict and the raw inverse distance."""
    position, up = camera_in_frame(cam["yaw"], cam["pitch"], cam["radius"])
    return {
        "projection": "orthographic" if cam["inverse_distance"] < 1e-3 else "perspective",
        "fov_y_deg": cam["fov"],
        "position": position,
        "target": [0.0, 0.0, 0.0],
        "up": up,
        "model": {"yaw_rad": cam["yaw"], "pitch_rad": cam["pitch"], "radius": cam["radius"], "fov_deg": cam["fov"]},
    }


def frame_camera(block, xyz, opacity, margin):
    """`block` (camera_block's) framed the way preprocessing framed the input: the decoded fov frames 1.4 to 1.6 times
    too much, by a factor that is not constant. The splats that are at least half opaque are projected into the
    predicted camera (three.js lookAt basis; tangent space for a perspective one); their box, times the input mode's
    crop margin, fills the view, and the target moves to its centre. Checked against input.png by rendering both with
    Spark: mask IoU 0.92 (avocado, orthographic) and 0.86 (a photo, perspective), against 0.38 for the decoded fov."""
    import numpy as np
    pos, tgt, up = (np.asarray(block[k], dtype=np.float64) for k in ("position", "target", "up"))
    z = pos - tgt
    dist = float(np.linalg.norm(z))
    z /= dist
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pts = np.asarray(xyz, dtype=np.float64)[np.asarray(opacity) >= 0] - pos
    if len(pts) == 0:
        return dict(block, framing="decoded: no splat is at least half opaque")
    vx, vy, vz = pts @ x, pts @ y, pts @ z
    ortho = block["projection"] == "orthographic"
    if not ortho:
        vx, vy = vx / -vz, vy / -vz
    cx, cy = (vx.min() + vx.max()) / 2, (vy.min() + vy.max()) / 2
    half = max(vx.max() - vx.min(), vy.max() - vy.min()) / 2 * margin
    shift = (cx * x + cy * y) * (1.0 if ortho else dist)
    out = dict(block)
    out["position"] = [float(v) for v in pos + shift]
    out["target"] = [float(v) for v in tgt + shift]
    if ortho:
        out["half_height"] = float(half)
        out["fov_y_deg"] = math.degrees(2 * math.atan(half / dist))
    else:
        out["fov_y_deg"] = math.degrees(2 * math.atan(half))
    out["framing"] = f"fitted: the opaque splats' projected box times crop margin {margin:g} fills the view"
    return out


GAUSSIAN_FIELDS = {
    "position": "x y z float32 in frame",
    "color": "f_dc_0..2 SH degree-0; rgb = 0.5 + 0.28209479177387814*f_dc",
    "opacity": "logit; alpha = sigmoid(opacity)",
    "scale": "scale_0..2 natural log of per-axis std dev",
    "rotation": "rot_0..3 unit quaternion w x y z",
    "normal": "nx ny nz zero, unused",
}


def ply_summary(path):
    """(vertex count, bounds min, bounds max, positions, opacity logits) read back from a written binary little-endian
    3DGS PLY."""
    import numpy as np
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise RuntimeError(f"{path}: no PLY header")
            head += line
        text = head.decode("ascii")
        if "format binary_little_endian 1.0" not in text:
            raise RuntimeError(f"{path}: not a binary little-endian PLY")
        count = int(re.search(r"element vertex (\d+)", text).group(1))
        names = re.findall(r"property float (\w+)", text)
        data = np.fromfile(f, dtype=[(n, "<f4") for n in names], count=count)
    if len(data) != count:
        raise RuntimeError(f"{path}: {len(data)} of {count} vertices")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    if count == 0 or not all(np.isfinite(data[n]).all() for n in names):
        raise RuntimeError(f"{path}: empty or non-finite values")
    return count, [float(v) for v in xyz.min(axis=0)], [float(v) for v in xyz.max(axis=0)], xyz, data["opacity"]


def manifest(*, count, ply_bytes, lo, hi, cam, settings, runtime, timing, xyz=None, opacity=None):
    return {
        "object": "asset3d", "version": 1, "representation": "gaussian_splat", "format": "ply",
        "file": "asset.ply", "input": "input.png", "bytes": ply_bytes,
        "gaussians": {"count": count, "sh_degree": 0, "fields": GAUSSIAN_FIELDS},
        "frame": {"up": model_to_ply((0.0, 0.0, 1.0)), "handedness": "right", "units": "normalized object box",
                  "bounds": {"min": lo, "max": hi}, "from_model": FROM_MODEL},
        "camera": camera_block(cam) if xyz is None else frame_camera(camera_block(cam), xyz, opacity,
                                                                         settings["crop_margin"]),
        "model": {"name": "Idol", "code": dict(SDK), "components": [dict(c) for c in COMPONENTS]},
        "settings": settings, "runtime": runtime, "timing_ms": timing,
    }


# -- the DINOv3 loader ------------------------------------------------------------------------------------------------

_DIRECT = {"cls_token": "embeddings.cls_token", "reg_token": "embeddings.register_tokens",
           "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
           "patch_embed.proj.bias": "embeddings.patch_embeddings.bias",
           "norm.weight": "norm.weight", "norm.bias": "norm.bias"}
_BLOCK = {"attn.proj.": "attention.o_proj.", "gamma_1": "layer_scale1.lambda1", "gamma_2": "layer_scale2.lambda1",
          "mlp.fc1_g.": "mlp.gate_proj.", "mlp.fc1_x.": "mlp.up_proj.", "mlp.fc2.": "mlp.down_proj.",
          "norm1.": "norm1.", "norm2.": "norm2."}


def timm_to_transformers(key):
    """A timm DINOv3 key -> its Transformers DINOv3ViTModel key(s) (research dino-mapping.md; the reverse of timm's
    checkpoint filter). The fused qkv weight is split into contiguous Q, K, V thirds: three keys. None: unknown."""
    if key in _DIRECT:
        return [_DIRECT[key]]
    m = re.fullmatch(r"blocks\.(\d+)\.(.+)", key)
    if not m:
        return None
    i, rest = m.groups()
    if rest == "attn.qkv.weight":
        return [f"layer.{i}.attention.{p}_proj.weight" for p in "qkv"]
    for a, b in _BLOCK.items():
        if rest.startswith(a):
            return [f"layer.{i}.{b}{rest[len(a):]}"]
    return None


def dinov3_from_timm(path, torch):
    """Transformers' DINOv3 ViT-H+/16 (the official config, BF16) with the timm checkpoint's values. The official
    model's Q and V biases (zero there) and its mask token (unused unmasked) are zero; anything else unmapped fails."""
    from safetensors import safe_open
    from transformers import AutoModel, DINOv3ViTConfig
    from transformers.initialization import no_init_weights

    with no_init_weights():  # every parameter is overwritten below; the RoPE buffer is computed in __init__
        model = AutoModel.from_config(DINOv3ViTConfig(**DINOV3_CONFIG), dtype=torch.bfloat16)
    model = model.eval().requires_grad_(False)
    sd = model.state_dict(keep_vars=True)
    live = lambda k: k if k in sd else "model." + k  # noqa: E731  Transformers 5 serialises layer.* as model.layer.*
    used = set()
    with torch.no_grad(), safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            targets = timm_to_transformers(key)
            if targets is None:
                raise RuntimeError(f"{path}: unexpected tensor {key}")
            x = f.get_tensor(key)
            parts = x.chunk(3, dim=0) if len(targets) == 3 else (x,)
            for target, part in zip(targets, parts):
                target = live(target)
                if target not in sd or sd[target].shape != part.shape:
                    raise RuntimeError(f"{path}: {key} does not fit {target}")
                sd[target].copy_(part)
                used.add(target)
        for k, v in sd.items():
            if k in used:
                continue
            if k != "embeddings.mask_token" and not k.endswith(("attention.q_proj.bias", "attention.v_proj.bias")):
                raise RuntimeError(f"{path}: nothing for {k}")
            v.zero_()
    return model


# -- the engine: idol-sdk, stage by stage -------------------------------------------------------------------------------

class _Stepper:
    """Stands in for the denoiser in idol.sampler.sample_flow: counts its calls, so each Euler step's start is known
    (passes calls a step), and runs on_step(k) there (k steps done). Values pass through untouched."""

    def __init__(self, model, passes, on_step):
        self.model, self.passes, self.on_step, self.calls = model, passes, on_step, 0

    def __call__(self, *args):
        if self.calls % self.passes == 0:
            self.on_step(self.calls // self.passes)
        self.calls += 1
        return self.model(*args)


class IdolEngine:
    """The real model. test_worker.py swaps in a fake with the same methods."""

    def __init__(self, weights, device, optimize):
        self.weights, self.device_name, self.want_optimize = weights, device, optimize
        self.p = None
        self.mem_peak = 0
        self._sampler = None

    def load(self):
        import torch
        from diffusers.models import AutoencoderKLFlux2
        from idol.conditioning import ImageEncoder
        from idol.models import BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
        from idol.pipeline import IdolPipeline, compile_for_sampling
        from safetensors.torch import load_file

        self.torch = torch
        name = self.device_name
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else ""
            if not name:
                raise RuntimeError("no CUDA or MPS device (set the option device to cpu to run on the CPU, unqualified)")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device cuda: torch sees no CUDA device")
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device mps: torch sees no MPS device")
        device = torch.device(name)
        idol_dir = os.path.join(self.weights, "idol")
        with open(os.path.join(idol_dir, "config.json")) as f:
            config = json.load(f)

        # IdolPipeline.__init__, with the encoders from the ungated local copies (and no access check)
        p = object.__new__(IdolPipeline)
        p.device, p.config = device, config

        def load(model, file):
            model.load_state_dict(load_file(os.path.join(idol_dir, f"{file}.safetensors")), strict=True)
            return model.eval().requires_grad_(False).to(device)

        t = time.monotonic()
        p.denoiser = load(LatentSeqMMFlowModel(**config["denoiser"]["args"]), "denoiser")
        p.decoder = load(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), "decoder")
        p.decoder_gs = load(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), "decoder_gs")
        p.background_remover = load(BiRefNet(), "birefnet")
        if device.type == "cuda":
            p.background_remover.half()
        log(f"idol models in {time.monotonic() - t:.1f} s")
        t = time.monotonic()
        enc = object.__new__(ImageEncoder)
        enc.device = device
        enc.dinov3 = dinov3_from_timm(os.path.join(self.weights, "dinov3", "model.safetensors"), torch).to(device)
        enc.flux2_vae = AutoencoderKLFlux2.from_pretrained(os.path.join(self.weights, "flux2-vae"), subfolder="vae",
                                                           torch_dtype=torch.bfloat16, local_files_only=True)
        enc.flux2_vae = enc.flux2_vae.eval().requires_grad_(False).to(device)
        p.image_encoder = enc
        log(f"encoders in {time.monotonic() - t:.1f} s")
        p.optimize = bool(self.want_optimize) and device.type == "cuda"
        if p.optimize:
            compile_for_sampling(p.denoiser)
        self.p = p
        self.release()

    @property
    def device(self):
        return self.p.device.type

    @property
    def optimize(self):
        return self.p.optimize

    def sync(self):
        if self.p.device.type == "mps":
            self.torch.mps.synchronize()
        elif self.p.device.type == "cuda":
            self.torch.cuda.synchronize()

    def release(self):
        gc.collect()
        if self.p is None:
            return
        self.sync()
        if self.p.device.type == "mps":
            self.torch.mps.empty_cache()
        elif self.p.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def resident(self):
        now, _ = footprint()
        if self.p is not None and self.p.device.type == "cuda":
            now += self.torch.cuda.memory_reserved()  # device memory is not in the resident set
        return now

    def torch_version(self):
        return self.torch.__version__

    # memory sampling (memory_stats): the allocator's driver bytes, every 20 ms during a generate
    def start_sampling(self):
        self.mem_peak = 0
        stop = threading.Event()

        def run():
            while not stop.wait(0.02):
                self.mem_peak = max(self.mem_peak, self.allocator_bytes())

        th = threading.Thread(target=run, daemon=True)
        th.start()
        self._sampler = (stop, th)

    def stop_sampling(self):
        if self._sampler:
            stop, th = self._sampler
            stop.set()
            th.join()
            self._sampler = None
            self.mem_peak = max(self.mem_peak, self.allocator_bytes())

    def allocator_bytes(self):
        if self.p is None:
            return 0
        if self.p.device.type == "mps":
            return int(self.torch.mps.driver_allocated_memory())
        if self.p.device.type == "cuda":
            return int(self.torch.cuda.memory_reserved())
        return 0

    # the stages of IdolPipeline.__call__, in its order (the RNG stream depends on it)
    def prepare(self, image, mode):
        with self.torch.no_grad():  # as inside IdolPipeline.__call__
            prepared = self.p.preprocess(image, mode=mode)  # ValueError on an empty alpha
        self.sync()
        return prepared

    def encode(self, prepared, seed):
        import numpy as np
        from tensordict import TensorDict
        torch, p = self.torch, self.p
        with torch.no_grad():
            pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(p.device)
            torch.manual_seed(seed)
            cond = p.image_encoder(pixels)
            noise = TensorDict({
                "latent": torch.randn(1, p.denoiser.q_token_length, p.denoiser.in_channels, device=p.device),
                "camera": torch.randn(1, 1, p.denoiser.cam_channels, device=p.device),
            }, batch_size=1, device=p.device)
        self.sync()
        return cond, noise

    def denoise(self, state, steps, guidance, on_step):
        """sample_flow over the denoiser; on_step(k) after each Euler step k (1..steps), at a synchronised point."""
        from idol.sampler import sample_flow
        cond, noise = state
        p = self.p
        passes = 2 if guidance > 0 and not p.optimize else 1  # eager CFG: a conditional and a negative pass a step

        def boundary(k):
            if k > 0:
                self.sync()
                on_step(k)
            check_cancel()

        with self.torch.no_grad(), p._sampling_settings():
            sample = sample_flow(_Stepper(p.denoiser, passes, boundary), noise, cond=cond,
                                 neg_cond=cond.apply(self.torch.zeros_like), steps=steps, rescale_t=RESCALE_T,
                                 cfg_strength=guidance, verbose=False, batch_cfg=p.optimize)
        self.sync()
        on_step(steps)
        return sample

    def decode(self, sample, count):
        g = self.p.decode(sample["latent"], count)
        self.sync()
        return g

    def camera(self, sample):
        from idol.camera import decode_camera
        latent = sample["camera"][0, 0]
        cam = decode_camera(latent)
        cam["inverse_distance"] = float(latent[3])
        return cam

    def save_ply(self, gaussian, path):
        gaussian.save_ply(path)


# -- the native engine: PyTorch preprocess, then TensorFold's tf-idol-generate ---------------------------------------------

class Preprocess:
    """Background removal and the crop to the 1,024 x 1,024 picture Idol sees: the SDK's own preprocess with its
    BiRefNet, in PyTorch. The one piece of the native backend still in PyTorch; when TensorFold's BiRefNet lands, a
    `native` kind replaces this class and the torch import with it."""

    def __init__(self, weights, device_name):
        self.weights, self.device_name = weights, device_name
        self.p = None

    def load(self):
        import torch
        from idol.models import BiRefNet
        from idol.pipeline import IdolPipeline
        from safetensors.torch import load_file
        self.torch = torch
        name = self.device_name
        if name == "auto":
            name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(name)
        idol_dir = os.path.join(self.weights, "idol")
        with open(os.path.join(idol_dir, "config.json")) as f:
            config = json.load(f)
        p = object.__new__(IdolPipeline)  # IdolPipeline.preprocess needs only these two and the background remover
        p.device, p.config = device, config
        m = BiRefNet()
        m.load_state_dict(load_file(os.path.join(idol_dir, "birefnet.safetensors")), strict=True)
        p.background_remover = m.eval().requires_grad_(False).to(device)
        if device.type == "cuda":
            p.background_remover.half()
        self.p = p

    def __call__(self, image, mode):
        with self.torch.no_grad():
            prepared = self.p.preprocess(image, mode=mode)  # ValueError on an empty alpha
        self.sync()
        return prepared

    def sync(self):
        if self.p.device.type == "mps":
            self.torch.mps.synchronize()

    def release(self):
        gc.collect()
        if self.p is not None and self.p.device.type == "mps":
            self.torch.mps.empty_cache()


# tf-idol-generate prints a line as each stage ends (zig/tests/idol_generate.zig): the line that ends one stage starts
# the next phase. VAE: the encoders are done, so denoising starts; denoiser: decoding starts; decoders: writing.
_TF_NEXT = (("VAE:", "denoise"), ("features: given", "denoise"), ("denoiser:", "decode"), ("decoders:", "write"))


class NativeEngine:
    """backend tensorfold. Same interface as IdolEngine where the worker shares code; `make` runs the child."""

    def __init__(self, weights, binary, preprocess):
        self.weights, self.binary = weights, binary
        self.pre = preprocess
        self.p = None  # the Worker clears it after a failed load
        self.mem_peak = 0
        self.child_peak = 0

    def load(self):
        if not self.binary or not os.path.isfile(self.binary) or not os.access(self.binary, os.X_OK):
            raise RuntimeError(f"tf_idol_generate {self.binary!r} is not an executable file (TensorFold's tf-idol-generate)")
        self.pre.load()
        self.p = self.pre.p
        self.release()

    @property
    def device(self):
        return "metal"

    @property
    def optimize(self):
        return False

    def release(self):
        self.pre.release()

    def resident(self):
        return footprint()[0]

    def torch_version(self):
        return self.pre.torch.__version__

    def start_sampling(self):
        self.mem_peak = 0

    def stop_sampling(self):
        pass

    def allocator_bytes(self):
        return 0

    def prepare(self, image, mode):
        return self.pre(image, mode)

    def make(self, prepared, s, job, progress):
        """The picture through tf-idol-generate: (the PLY's path in `job`, the camera dict, the child's own seconds).
        Cancellable: SIGUSR1 ends the child within 50 ms."""
        import numpy as np
        from safetensors.numpy import save_file
        work = os.path.join(job, ".tf-idol")
        os.makedirs(work, exist_ok=True)
        picture = os.path.join(work, "picture.safetensors")
        save_file({"image": np.asarray(prepared, dtype=np.float32) / 255.0}, picture)  # [1024, 1024, 3] in [0, 1]
        w = self.weights
        argv = [self.binary, os.path.join(w, "idol"), os.path.join(w, "flux2-vae", "vae"), os.path.join(w, "dinov3"),
                picture, work, "--seed", str(s["seed"]), "--steps", str(s["steps"]), "--cfg", repr(s["guidance"]),
                "--gaussians", str(s["num_gaussians"])]
        self.release()  # BiRefNet's activations go back before the child allocates, so the two peaks do not stack
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        tail = []
        self.child_peak = 0

        def read():
            for line in child.stdout:
                line = line.rstrip()
                tail.append(line)
                del tail[:-20]
                for key, phase in _TF_NEXT:
                    if line.startswith(key):
                        if phase == "decode":  # the denoiser reports its steps only when it is done
                            progress({"phase": "denoise", "step": s["steps"], "of": s["steps"]})
                        progress({"phase": phase, "step": 0, "of": s["steps"] if phase == "denoise" else 1})

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        progress({"phase": "encode", "step": 0, "of": 1})
        try:
            while child.poll() is None:
                if CANCEL.is_set():
                    child.terminate()
                    child.wait()
                    raise Cancelled()
                self.child_peak = max(self.child_peak, footprint(child.pid)[0])
                time.sleep(0.05)
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.wait()
            shutil.rmtree(work, ignore_errors=True)
            raise
        finally:
            reader.join(timeout=5)
            child.stdout.close()
        if child.returncode != 0:
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"tf-idol-generate exited {child.returncode}: {' | '.join(tail[-4:])}")
        with open(os.path.join(work, "camera.json")) as f:
            cam = json.load(f)
        # decode_camera's convention, as gaussian.camera writes it: radius 1000 when the inverse distance is under 1e-3
        cam["inverse_distance"] = 0.0 if cam["radius"] >= 1000 else 1.0 / cam["radius"]
        stages = {}
        try:
            with open(os.path.join(work, "tf-idol-generate.json")) as f:
                stages = {k: v for k, v in json.load(f).items() if k.endswith("_s") or k == "metal_peak_allocated_gb"}
        except (OSError, ValueError):
            pass
        ply = os.path.join(job, "asset.ply")
        os.replace(os.path.join(work, "generate.ply"), ply)
        shutil.rmtree(work, ignore_errors=True)
        return ply, cam, stages


# -- the worker -------------------------------------------------------------------------------------------------------

def _int(v, name):
    if isinstance(v, bool) or not isinstance(v, int):
        raise Refused(f"{name} must be an integer")
    return v


def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Refused(f"{name} must be a number")
    return float(v)


class Options:
    def __init__(self, tool):
        o = tool.get("options") or {}
        unknown = sorted(set(o) - OPTIONS - DAEMON_OPTIONS)
        if unknown:
            log(f"options not used by the worker: {unknown}")
        self.id = tool.get("id") or "idol"
        self.weights = tool.get("weights") or o.get("weights") or ""
        if not self.weights:
            raise ValueError("weights (the directory fetch.sh fills: idol/, dinov3/, flux2-vae/) is required")
        self.backend = o.get("backend", "pytorch")
        if self.backend not in ("pytorch", "tensorfold"):
            raise ValueError("backend must be pytorch or tensorfold")
        self.tf_idol_generate = o.get("tf_idol_generate", "")
        if self.backend == "tensorfold" and not self.tf_idol_generate:
            raise ValueError("backend tensorfold needs tf_idol_generate: the path of TensorFold's tf-idol-generate")
        self.preprocess = o.get("preprocess", "pytorch")
        if self.preprocess != "pytorch":
            raise ValueError("preprocess must be pytorch (TensorFold's BiRefNet is not ported yet)")
        self.device = o.get("device", "auto")
        if self.device not in ("auto", "mps", "cuda", "cpu"):
            raise ValueError("device must be auto, mps, cuda or cpu")
        self.optimize = bool(o.get("optimize", False))
        self.multiple = int(o.get("gaussian_multiple", 32))
        self.min_gaussians = int(o.get("min_gaussians", 32768))
        self.max_gaussians = int(o.get("max_gaussians", 1048576))
        self.default_gaussians = int(o.get("default_gaussians", 262144))
        self.default_steps = int(o.get("default_steps", 50))
        self.default_guidance = float(o.get("default_guidance", 3.0))
        self.max_steps = int(o.get("max_steps", 100))
        self.max_guidance = float(o.get("max_guidance", 10.0))
        self.input_modes = list(o.get("input_modes", ["synthetic", "real"]))
        self.formats = list(o.get("formats", ["ply"]))
        self.verify = o.get("verify", "size")
        self.memory_stats = bool(o.get("memory_stats", False))
        if self.verify not in ("size", "none"):
            raise ValueError("verify must be size or none")
        if self.multiple % 32 or self.multiple <= 0:
            raise ValueError("gaussian_multiple must be a positive multiple of 32 (Gaussians come 32 an anchor)")
        if not (32 <= self.min_gaussians <= self.default_gaussians <= self.max_gaussians <= 1048576):
            raise ValueError("need 32 <= min_gaussians <= default_gaussians <= max_gaussians <= 1048576")
        if any(n % self.multiple for n in (self.min_gaussians, self.default_gaussians, self.max_gaussians)):
            raise ValueError("min/default/max_gaussians must be multiples of gaussian_multiple")
        if not (1 <= self.default_steps <= self.max_steps) or not (0 <= self.default_guidance <= self.max_guidance):
            raise ValueError("default_steps must be 1..max_steps and default_guidance 0..max_guidance")
        if not set(self.input_modes) <= set(CROP_MARGIN) or not self.input_modes:
            raise ValueError(f"input_modes must be some of {sorted(CROP_MARGIN)}")
        if self.formats != ["ply"]:
            raise ValueError("formats: this worker writes ply only")


class Worker:
    def __init__(self, tool, engine=None):
        self.opts = Options(tool)
        o = self.opts
        if engine is None:
            engine = (NativeEngine(o.weights, o.tf_idol_generate, Preprocess(o.weights, o.device)) if o.backend == "tensorfold"
                      else IdolEngine(o.weights, o.device, o.optimize))
        self.engine = engine
        self.loaded = False

    # -- load ----------------------------------------------------------------------------------------------------

    def check_weights(self):
        """Every pinned file there with its size (fetch.sh checked the digests): a clear error names the first missing."""
        with open(PINS) as f:
            pins = json.load(f)
        missing = []
        for p in pins["selected_files"]:
            path = os.path.join(self.opts.weights, p["path"])
            if not os.path.isfile(path):
                missing.append(f"{p['path']} (missing)")
            elif self.opts.verify == "size" and os.path.getsize(path) != p["size_bytes"]:
                missing.append(f"{p['path']} ({os.path.getsize(path)} bytes, pinned {p['size_bytes']})")
        if missing:
            raise RuntimeError(f"weights {self.opts.weights}: {', '.join(missing)}: run tools/asset3d/idol/fetch.sh")

    def load(self):
        if self.loaded:
            return {"ok": True, "resident": self.engine.resident(), "cancel": "signal"}
        t = time.monotonic()
        try:
            self.check_weights()
            self.engine.load()
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
            log(f"load failed: {e}")
            self.engine.p = None
            gc.collect()
            return {"ok": False, "type": "server_error", "error": f"load failed: {e}"}
        self.loaded = True
        resident = self.engine.resident()
        log(f"loaded on {self.engine.device} in {time.monotonic() - t:.1f} s: resident {resident / 2**30:.2f} GiB")
        return {"ok": True, "resident": resident, "cancel": "signal"}

    # -- generate --------------------------------------------------------------------------------------------------

    def parse(self, cmd):
        """The request's settings, or Refused. Absent fields take the tool's defaults; nothing is changed quietly."""
        req = (cmd.get("request") or {}).get("asset3d")
        if not isinstance(req, dict):
            raise Refused("this model makes 3D assets: send an asset3d request")
        o = self.opts
        job = cmd.get("dir") or ""
        if not job or not os.path.isdir(job):
            raise Refused("the job directory is missing")
        get = lambda k, d=None: d if req.get(k) is None else req[k]  # noqa: E731  null is absent
        name = get("image")
        if not isinstance(name, str) or not name:
            raise Refused("image (an input file in the job directory) is required")
        if os.path.dirname(os.path.normpath(name)) or name.startswith(".") or os.sep in name:
            raise Refused("image must name a file in the job directory")
        mode = get("input_mode", "synthetic")
        if mode not in o.input_modes:
            raise Refused(f"input_mode must be one of {o.input_modes}")
        fmt = get("format", "ply")
        if fmt not in o.formats:
            raise Refused(f"format must be one of {o.formats}")
        seed = get("seed")
        seed = int.from_bytes(os.urandom(4), "little") if seed is None else _int(seed, "seed")
        if not 0 <= seed < 1 << 64:
            raise Refused("seed must be 0 to 2^64-1")
        steps = _int(get("steps", o.default_steps), "steps")
        if not 1 <= steps <= o.max_steps:
            raise Refused(f"steps must be 1 to {o.max_steps}")
        guidance = _num(get("guidance", o.default_guidance), "guidance")
        if not 0 <= guidance <= o.max_guidance:
            raise Refused(f"guidance must be 0 to {o.max_guidance:g} (0 turns the negative pass off)")
        count = _int(get("num_gaussians", o.default_gaussians), "num_gaussians")
        if not o.min_gaussians <= count <= o.max_gaussians or count % o.multiple:
            raise Refused(f"num_gaussians must be {o.min_gaussians} to {o.max_gaussians}, a multiple of {o.multiple}")
        image = self.read_image(os.path.join(job, name))
        return job, image, {"input_mode": mode, "seed": seed, "steps": steps, "guidance": guidance,
                            "rescale_t": RESCALE_T, "num_gaussians": count, "crop_margin": CROP_MARGIN[mode]}

    @staticmethod
    def read_image(path):
        from PIL import Image
        try:
            im = Image.open(path)
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError(f"{im.format} is not taken")
            im.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
            raise Refused(f"the input image could not be read ({e}): send a PNG, JPEG or WebP")
        if min(im.size) < 16:
            raise Refused(f"the input image is {im.width}x{im.height}: send one at least 16 pixels a side")
        if im.mode == "RGBA" and im.getchannel("A").getextrema()[1] == 0:
            raise Refused("the input image is fully transparent: its alpha marks no object")
        return im

    def generate(self, cmd, progress):
        if not self.loaded:
            raise RuntimeError("not loaded")
        start = time.monotonic()
        job, image, s = self.parse(cmd)
        e = self.engine
        timing = {}
        clock = [time.monotonic()]

        def phase_done(name):
            now = time.monotonic()
            timing[name] = round((now - clock[0]) * 1000)
            clock[0] = now

        progress({"phase": "preprocess", "step": 0, "of": 1})
        try:
            prepared = e.prepare(image, s["input_mode"])
        except ValueError as err:  # the SDK's "no object found in the image (the alpha channel is empty)"
            raise Refused(f"no object found in the image: {err}")
        phase_done("preprocess")
        check_cancel()
        if self.opts.backend == "tensorfold":
            return self.finish(cmd, job, prepared, s, start, timing, phase_done, progress)

        progress({"phase": "encode", "step": 0, "of": 1})
        state = e.encode(prepared, s["seed"])
        phase_done("encode")
        check_cancel()

        steps = s["steps"]
        progress({"phase": "denoise", "step": 0, "of": steps})
        sample = e.denoise(state, steps, s["guidance"], lambda k: progress({"phase": "denoise", "step": k, "of": steps}))
        del state
        phase_done("denoise")
        check_cancel()

        progress({"phase": "decode", "step": 0, "of": 1})
        gaussian = e.decode(sample, s["num_gaussians"])
        cam = e.camera(sample)
        del sample
        phase_done("decode")
        check_cancel()

        progress({"phase": "write", "step": 0, "of": 1})
        ply = os.path.join(job, "asset.ply")
        e.save_ply(gaussian, ply)
        del gaussian
        prepared.save(os.path.join(job, "input.png"))
        count, lo, hi, xyz, opacity = ply_summary(ply)
        phase_done("write")
        timing["total"] = round((time.monotonic() - start) * 1000)
        doc = manifest(count=count, ply_bytes=os.path.getsize(ply), lo=lo, hi=hi, cam=cam, settings=s,
                       runtime={"backend": "pytorch", "device": e.device, "torch": e.torch_version(),
                                "optimize": e.optimize}, timing=timing, xyz=xyz, opacity=opacity)
        with open(os.path.join(job, "asset.json"), "w") as f:
            json.dump(doc, f, indent=1)
        log(f"{cmd.get('id', '')}: {count} Gaussians in {timing['total'] / 1000:.1f} s {timing}")
        return {"ok": True, "files": list(FILES), "seed": s["seed"], "ms": timing["total"]}

    def finish(self, cmd, job, prepared, s, start, timing, phase_done, progress):
        """backend tensorfold: the rest in tf-idol-generate, then the same files and manifest as the PyTorch path."""
        e = self.engine
        ply, cam, stages = e.make(prepared, s, job, progress)
        phase_done("generate")
        progress({"phase": "write", "step": 0, "of": 1})
        prepared.save(os.path.join(job, "input.png"))
        count, lo, hi, xyz, opacity = ply_summary(ply)
        phase_done("write")
        timing["total"] = round((time.monotonic() - start) * 1000)
        doc = manifest(count=count, ply_bytes=os.path.getsize(ply), lo=lo, hi=hi, cam=cam, settings=s,
                       runtime={"backend": "tensorfold", "device": "metal", "preprocess": f"pytorch {e.torch_version()}",
                                "binary": os.path.basename(e.binary), "stages_s": stages,
                                "child_peak_footprint": e.child_peak}, timing=timing, xyz=xyz, opacity=opacity)
        with open(os.path.join(job, "asset.json"), "w") as f:
            json.dump(doc, f, indent=1)
        log(f"{cmd.get('id', '')}: {count} Gaussians in {timing['total'] / 1000:.1f} s {timing}; child peak "
            f"{e.child_peak / 2**30:.2f} GiB")
        return {"ok": True, "files": list(FILES), "seed": s["seed"], "ms": timing["total"]}

    def stats(self):
        now, peak = footprint()
        return {"ok": True, "footprint": now, "footprint_peak": peak, "resident": self.engine.resident(),
                "allocator": self.engine.allocator_bytes(), "allocator_peak_last_generate": self.engine.mem_peak,
                "child_peak_last_generate": getattr(self.engine, "child_peak", 0)}


def send(obj):
    PROTO.write(json.dumps(obj) + "\n")
    PROTO.flush()


def run_generate(worker, cmd, out=None):
    """One generate: the reply object. Progress lines go out through `out` (send by default). The device cache is
    emptied after the exception (if any) is handled: a live traceback would still hold the job's tensors."""
    reply = _generate(worker, cmd, out or send)
    worker.engine.release()
    return reply


def _generate(worker, cmd, out):
    global GENERATING
    sampling = worker.loaded and worker.opts.memory_stats
    try:
        try:
            CANCEL.clear()
            GENERATING = True  # inside the try: a signal right after it is still honoured
            if sampling:
                worker.engine.start_sampling()
            return worker.generate(cmd, lambda p: out({"progress": p}))
        finally:
            GENERATING = False
            CANCEL.clear()
            if sampling:
                worker.engine.stop_sampling()
    except Cancelled:
        log(f"{cmd.get('id', '')}: cancelled; still loaded")
        return {"ok": False, "type": "cancelled", "error": "cancelled"}
    except Refused as e:
        return {"ok": False, "type": "invalid_request_error", "error": str(e)}
    except Exception as e:  # noqa: BLE001  the model failed: the reply says why, the worker stays up
        msg = f"{type(e).__name__}: {e}"
        log(f"generate failed: {msg}")
        if any(k in str(e).lower() for k in ("out of memory", "outofmemory")):  # torch's words, or Zig's error name
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
            return
        if worker is None:
            send({"ok": False, "type": "server_error", "error": f"bad tool config: {problem}"})
        elif op == "load":
            send(worker.load())
        elif op == "generate":
            send(run_generate(worker, cmd))
        elif op == "stats":
            send(worker.stats())
        else:
            send({"ok": False, "type": "invalid_request_error", "error": f"unknown op {op!r}"})


if __name__ == "__main__":
    main()
    sys.stderr.flush()
    os._exit(0)  # no interpreter teardown of the loaded models: the process's memory goes back at once
