"""The D2 native oracle: the pinned kandinsky-6 code (eac06e39) on the real checkpoint, with forward hooks that record the
named checkpoints of kernels/cuda/kandinsky6/video_vae/QUALIFICATION.md into a record set (ops.jsonl, blobs/<sha>,
env.json, summary.json). Run in the offline aarch64 bundle (run-oracle.sh); nothing here changes the native arithmetic.

Native call path, unchanged: `vae_video.build_vae(vae_path)`, `postprocess_video(LatentBundle(...), vae, bs=1)`
(s_dec, f_dec, n_dec) and `prepare_latents.encode_i2va_first_frame(PIL image, vae, device, height, width)` (s_enc, n_enc).
The only global setting applied is the pinned pipeline's own `torch.set_float32_matmul_precision("high")` (factory.py:162,
before construction); GLOBAL_AUDIT lists what else the pinned package sets and why it is not on this path.

What is wrapped, and how (every wrapper records and then returns the native value untouched):
  * module checkpoints: forward hooks on the named modules, first call only (later temporal tiles are not recorded);
    a resnet's `nonlinearity` is one shared nn.SiLU called twice: calls 1 and 2 are `R.nonlinearity.0` / `.1`. In the
    mid-block attention diffusers 0.40.0's AttnProcessor2_0 calls `group_norm(h.transpose(1, 2)).transpose(1, 2)`, so
    the group_norm output is (B, C, N) and is recorded transposed, as the canonical [N][C].
  * decode.input / decode.rgb: an instance attribute `vae.decode` wrapping the bound method (its z argument), and the
    return value of postprocess_video, both as [T][H][W][C].
  * encode.canvas: `torchvision.transforms.functional.crop` replaced for the duration of the call (prepare_latents
    imports TF inside the function and looks up `TF.crop` at call time); its uint8 output is the tensor the next
    statement divides by 127.5. encode.input: an instance attribute `vae.encode` (its x argument); encode.mean /
    encode.std from the posterior it returns.
  * encode.eps_native: `diffusers.models.autoencoders.vae.randn_tensor` (the name DiagonalGaussianDistribution.sample
    calls) replaced for the duration of the call; the implicit default generator is untouched.
  * encode.latent_kit: the native ops on the kit eps: `(mean + std * eps)`, then `.squeeze(0).permute(1, 2, 3, 0) *
    vae.config.scaling_factor` as prepare_latents.py:236-237. encode.latent_native: the function's return value.
  * the tiling: an instance attribute `vae.get_dec_optimal_tiling` records mem_get_info and the plan it returns; a
    counter on vae.decoder gives the number of decoder passes (temporal x spatial tiles).
Each case also runs once more without hooks under torch.profiler (s_dec, f_dec, s_enc only) for its kernel-name list.

`--device cpu --dtype float32` is a plumbing test of the hooks and the record format, NOT an oracle: the fp16
checkpoint is loaded by build_vae on the CPU and cast to float32, and env.json says "oracle": false.

  python oracle.py --upstream DIR --vae DIR --inputs DIR --out DIR --cases s_dec,f_dec,s_enc,n_enc,n_dec [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import struct
import subprocess
import sys
import time
import traceback
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

UPSTREAM_SHA = "eac06e39b904940baff2c408e925cdb0c4529269"
PINS = {  # upstream file: sha256 (RECEIPT.md, Source pins)
    "kandinsky/core/components/hunyuan_vae.py": "f5e8dd402aba625f23aef86ab7d5324dc6a4e8585b72917a3bf27bd46745bbaf",
    "kandinsky/core/components/vae_video.py": "7ad34af7bbf99319e223ebaf5a2048e15d324692740d0fbcbc0718030df1ec28",
    "kandinsky/core/algo/prepare_latents.py": "edd9cdef58ce31a20442eea568621578fd129722325992850b31cdaa95751ee2",
    "kandinsky/core/algo/postprocess_video.py": "476474b6827476d06c62b0e70957bd2f807e04261676292c3b116b9087c92847",
}
VAE_CONFIG_SHA = "8dd1ef5ce49d2a56cdb5ea4c6553244320cd7185918463578c4c08a31f55e50a"
VAE_HEADER_SHA = "ec504903cc22243034f7fec06892e5d82970b0c527bdc22e5ce85a2a924471f0"  # the JSON header, without its u64 length

# Global torch state the pinned package touches (audit of kandinsky/ at eac06e39, all files).
GLOBAL_APPLIED = ['torch.set_float32_matmul_precision("high")  # pipeline/factory.py:162, before model construction']
GLOBAL_AUDIT = {
    "pipeline/factory.py:162": "torch.set_float32_matmul_precision('high'): applied",
    "pipeline/pipeline.py:78": "torch.no_grad() around the pipeline call: postprocess_video and encode_i2va_first_frame "
                               "are themselves @torch.no_grad(); no other global state",
    "runtime/aoti.py:337": "set_float32_matmul_precision('high'): the AOTI export path only; same value, not separate",
    "runtime/aoti.py:297-341": "OMP/MKL/OPENBLAS_NUM_THREADS, TORCHINDUCTOR_*: AOTI export only, not applied",
    "cli.py:308": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (setdefault): the `export` command only, not the "
                  "generate path; not applied",
    "cli.py:169-170": "KANDINSKY_LOG_DIR / KANDINSKY_PROFILE_DIR: logging only, not applied",
    "none": "no torch.backends.* (TF32, cuDNN benchmark/deterministic, SDPA backend) setting anywhere in the package",
}
RAW = {  # per case: None = every checkpoint raw, else the names stored raw (the rest hashed)
    "s_dec": None, "f_dec": None, "s_enc": None,
    "n_enc": {"encode.canvas", "encode.input", "quant_conv", "encode.mean", "encode.std", "encode.latent_kit",
              "encode.eps_native", "encode.latent_native"},
    "n_dec": {"decode.input", "decode.rgb"},
}
PROFILE = {"s_dec", "f_dec", "s_enc"}
DT = {torch.float16: "f16", torch.bfloat16: "bf16", torch.uint8: "u8", torch.float32: "f32"}
MODULE = re.compile(r"^(post_quant_conv|quant_conv|(encoder|decoder)\.(conv_in|conv_act|conv_out)"
                    r"|encoder\.down_blocks\.\d+\.downsamplers\.0|decoder\.up_blocks\.\d+\.upsamplers\.0)$")
RESNET = re.compile(r"^(encoder\.down_blocks\.\d+|decoder\.up_blocks\.\d+|(encoder|decoder)\.mid_block)\.resnets\.\d+$")
ATTN = re.compile(r"^(encoder|decoder)\.mid_block\.attentions\.0$")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def channels_last(t: torch.Tensor, layout: str) -> torch.Tensor:
    """Host copy in the contract's layout: ncthw -> [T][H][W][C], nnc -> [N][C], ncn -> [N][C], nchw -> [H][W][C],
    keep -> as is. The batch must be 1."""
    t = t.detach().to("cpu")  # dense views keep their strides: no device-side copy
    if layout == "keep":
        return t.contiguous()
    assert t.shape[0] == 1, (layout, tuple(t.shape))
    perm = {"ncthw": (0, 2, 3, 4, 1), "nnc": (0, 1, 2), "ncn": (0, 2, 1), "nchw": (0, 2, 3, 1)}[layout]
    return t.permute(*perm)[0].contiguous()


class Records:
    """A record set directory, appended to case by case (one process per case is the normal use)."""

    def __init__(self, out: Path):
        self.out = out
        (out / "blobs").mkdir(parents=True, exist_ok=True)
        self.ops = out / "ops.jsonl"
        self.i = 0
        self.cases = set()
        if self.ops.exists():
            for line in self.ops.read_text().splitlines():
                r = json.loads(line)
                self.i = max(self.i, r["i"] + 1)
                self.cases.add(r["case"])
        self.case = None
        self.raw = None
        self.names = set()
        self.count = 0

    def begin(self, case: str):
        if case in self.cases:
            raise SystemExit(f"oracle: {self.out} already holds case {case}; use a fresh --out")
        self.case, self.raw, self.names, self.count = case, RAW[case], set(), 0

    def put(self, name: str, t: torch.Tensor, layout: str):
        assert name not in self.names, f"checkpoint {name} recorded twice"
        self.names.add(name)
        x = channels_last(t, layout)
        a = (x.view(torch.int16) if x.dtype == torch.bfloat16 else x).numpy()
        data = memoryview(np.ascontiguousarray(a)).cast("B")
        sha = hashlib.sha256(data).hexdigest()
        raw = self.raw is None or name in self.raw
        if raw and not (self.out / "blobs" / sha).exists():
            (self.out / "blobs" / sha).write_bytes(data)
        rec = {"i": self.i, "case": self.case, "name": name, "dtype": DT[x.dtype], "shape": list(x.shape), "sha": sha,
               "raw": raw}
        with open(self.ops, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.i += 1
        self.count += 1


def merge_json(p: Path, update: dict):
    d = json.loads(p.read_text()) if p.exists() else {}
    cases = {**d.get("cases", {}), **update.pop("cases", {})}
    d.update(update)
    d["cases"] = cases
    p.write_text(json.dumps(d, indent=1))


def import_upstream(up: Path):
    """The upstream modules without running kandinsky/__init__.py (it imports the whole pipeline: transformers, the
    text encoders, ...): a namespace stub for the top package, the real subpackages below it."""
    bad = {f: s for f, s in PINS.items() if sha_file(up / f) != s}
    if bad:
        raise SystemExit(f"oracle: upstream files differ from the pins: {sorted(bad)}")
    pkg = types.ModuleType("kandinsky")
    pkg.__path__ = [str(up / "kandinsky")]
    sys.modules["kandinsky"] = pkg
    from kandinsky.core.algo import postprocess_video, prepare_latents
    from kandinsky.core.components import vae_video
    from kandinsky.core.types import LatentBundle
    return vae_video, postprocess_video, prepare_latents, LatentBundle


def check_vae(vae_dir: Path) -> dict:
    cfg = sha_file(vae_dir / "config.json")
    with open(vae_dir / "diffusion_pytorch_model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = hashlib.sha256(f.read(n)).hexdigest()
    if cfg != VAE_CONFIG_SHA or hdr != VAE_HEADER_SHA:
        raise SystemExit(f"oracle: checkpoint differs from the pins (config {cfg}, header {hdr})")
    return {"path": str(vae_dir), "config_sha256": cfg, "safetensors_header_sha256": hdr, "header_bytes": n}


def _get(f):
    try:
        return f()
    except Exception as e:  # attribute absent in this torch
        return f"unavailable: {type(e).__name__}"


def precision() -> dict:
    b = torch.backends
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp32_precision": _get(lambda: b.fp32_precision),
        "cuda.matmul.fp32_precision": _get(lambda: b.cuda.matmul.fp32_precision),
        "cuda.matmul.allow_tf32": _get(lambda: b.cuda.matmul.allow_tf32),
        "cuda.matmul.allow_fp16_reduced_precision_reduction":
            _get(lambda: b.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "cuda.matmul.allow_bf16_reduced_precision_reduction":
            _get(lambda: b.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "cuda.preferred_blas_library": _get(lambda: str(b.cuda.preferred_blas_library())),
        "cudnn.enabled": b.cudnn.enabled,
        "cudnn.allow_tf32": _get(lambda: b.cudnn.allow_tf32),
        "cudnn.fp32_precision": _get(lambda: b.cudnn.fp32_precision),
        "cudnn.conv.fp32_precision": _get(lambda: b.cudnn.conv.fp32_precision),
        "cudnn.benchmark": b.cudnn.benchmark,
        "cudnn.deterministic": b.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sdpa": {"flash": _get(b.cuda.flash_sdp_enabled), "mem_efficient": _get(b.cuda.mem_efficient_sdp_enabled),
                 "math": _get(b.cuda.math_sdp_enabled), "cudnn": _get(b.cuda.cudnn_sdp_enabled),
                 "fp16_bf16_reduction_math_allowed": _get(b.cuda.fp16_bf16_reduction_math_sdp_allowed)},
    }


def environment(dev: torch.device, oracle: bool, dtype: torch.dtype, up: Path, vae: dict) -> dict:
    gpu = None
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        drv = _get(lambda: subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                          capture_output=True, text=True, timeout=30).stdout.strip())
        gpu = {"name": p.name, "capability": list(torch.cuda.get_device_capability(dev)), "driver": drv,
               "total_memory": p.total_memory, "multi_processor_count": p.multi_processor_count}
    keep = re.compile(r"^(TORCH|PYTORCH|CUDA|CUBLAS|CUBLASLT|CUDNN|NVIDIA|NCCL|OMP|MKL|PYTHON|LD_LIBRARY)")
    return {
        "oracle": oracle,
        "note": None if oracle else "plumbing test of hooks and record format on CPU float32: NOT an oracle",
        "device": str(dev), "dtype": str(dtype).removeprefix("torch."), "gpu": gpu,
        "cuda": torch.version.cuda, "torch": torch.__version__, "cudnn": _get(torch.backends.cudnn.version),
        "diffusers": md.version("diffusers"), "torchvision": md.version("torchvision"), "numpy": np.__version__,
        "python": sys.version, "executable": sys.executable, "machine": platform.machine(),
        "upstream": {"path": str(up), "sha": UPSTREAM_SHA, "files": PINS}, "vae": vae,
        "global_settings_applied": GLOBAL_APPLIED, "global_settings_audit": GLOBAL_AUDIT,
        "precision": precision(),
        "env": {k: v for k, v in sorted(os.environ.items()) if keep.match(k)},
        "packages": {d.metadata["Name"]: d.version for d in sorted(md.distributions(), key=lambda d: d.metadata["Name"])},
    }


class Hooks:
    """Forward hooks on the contract's modules, first call only (nonlinearity: first two calls)."""

    def __init__(self, vae, rec: Records):
        self.rec, self.handles, self.calls = rec, [], {}
        mods = dict(vae.named_modules())

        def hook(name, layout, ncalls=1):
            def f(_m, _inp, out):
                k = self.calls.get(name, 0)
                self.calls[name] = k + 1
                if k < ncalls:
                    rec.put(f"{name}.{k}" if ncalls > 1 else name, out, layout)
            self.handles.append(mods[name].register_forward_hook(f))

        for name, m in mods.items():
            if MODULE.match(name):
                hook(name, "ncthw")
            elif RESNET.match(name):
                hook(f"{name}.nonlinearity", "ncthw", 2)
                hook(f"{name}.conv1", "ncthw")
                if m.conv_shortcut is not None:
                    hook(f"{name}.conv_shortcut", "ncthw")
                hook(name, "ncthw")
            elif ATTN.match(name):
                hook(f"{name}.group_norm", "ncn")  # applied to h.transpose(1, 2): (B, C, N)
                for p in ("to_q", "to_k", "to_v"):
                    hook(f"{name}.{p}", "nnc")
                hook(name, "nnc")
        self.decoder_calls = 0

        def count(*_):
            self.decoder_calls += 1
        self.handles.append(vae.decoder.register_forward_pre_hook(count))

    def remove(self):
        for h in self.handles:
            h.remove()


@contextmanager
def patched(obj, attr, value):
    """Replace obj.attr for the block; an instance attribute shadowing a method is deleted again afterwards."""
    own = attr in vars(obj)
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if own:
            setattr(obj, attr, old)
        else:
            delattr(obj, attr)


def loaded_libs() -> list:
    """Shared objects of the CUDA stack mapped into this process (which cuBLAS / cuDNN / runtime actually ran)."""
    pat = re.compile(r"lib(cuda|cudart|cublas|cublasLt|cudnn[a-z_]*|nvrtc|nvJitLink|cusparse|cufft|curand|nccl"
                     r"|torch_cuda|c10_cuda)\.so")
    try:
        paths = {l.split()[-1] for l in open("/proc/self/maps") if "/" in l and pat.search(l.rsplit("/", 1)[-1])}
    except OSError:
        return []
    return sorted(paths)


def mem_info(dev):
    return list(torch.cuda.mem_get_info(dev)) if dev.type == "cuda" else None


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def load(inputs: Path, index: dict, name: str) -> torch.Tensor:
    e = index[name]
    assert sha_file(inputs / e["file"]) == e["sha256"], f"input {name} differs from inputs.json"
    np_dt = {"bf16": "<i2", "f16": "<f2", "u8": "u1", "f32": "<f4"}[e["dtype"]]
    t = torch.from_numpy(np.fromfile(inputs / e["file"], dtype=np_dt).reshape(e["shape"]).copy())
    return t.view(torch.bfloat16) if e["dtype"] == "bf16" else t


def run_decode(case, vae, mods, inp, index, dev, rec=None, tiling=None):
    _, postprocess_video, _, LatentBundle = mods
    # packed (T, h, w, 16): bf16 (s_dec) or fp32 (f_dec, n_dec: the PiFlow sampler's output)
    z = load(inp, index, f"{case}.latent").to(dev)
    if tiling is not None:
        probe = vae.get_dec_optimal_tiling([1, 16, *z.shape[:3]], device=dev)
        tiling["probe"] = {"mem_get_info": mem_info(dev), "tile_size": list(probe[0]), "tile_stride": list(probe[1])}
    bundle = LatentBundle(video=z, audio=None, video_cu_seqlens=torch.tensor([0, z.shape[0]], dtype=torch.int32,
                                                                              device=dev), audio_cu_seqlens=None)
    if rec is None:
        return postprocess_video.postprocess_video(bundle, vae, bs=1)
    decode, choose = vae.decode, vae.get_dec_optimal_tiling

    def decode_w(zz, *a, **k):
        rec.put("decode.input", zz, "ncthw")
        return decode(zz, *a, **k)

    def choose_w(shape, device=None):
        free = mem_info(dev)
        r = choose(shape, device=device)
        tiling["chosen"] = {"mem_get_info": free, "shape": list(shape), "tile_size": list(r[0]),
                            "tile_stride": list(r[1])}
        return r
    with patched(vae, "decode", decode_w), patched(vae, "get_dec_optimal_tiling", choose_w):
        rgb = postprocess_video.postprocess_video(bundle, vae, bs=1)
    rec.put("decode.rgb", rgb, "ncthw")
    return rgb


def run_encode(case, vae, mods, inp, index, dev, rec=None, extra=None):
    from diffusers.models.autoencoders import vae as dvae
    from PIL import Image
    import torchvision.transforms.functional as TF
    _, _, prepare_latents, _ = mods
    img = load(inp, index, f"{case}.image").numpy()
    h, w, _ = img.shape
    pil = Image.fromarray(img)  # uint8 [H][W][3]: mode RGB
    if rec is None:
        return prepare_latents.encode_i2va_first_frame(pil, vae, dev, height=h, width=w)
    if extra is not None:  # the implicit generator that draws eps_native (torch 2.14 seeds the CPU one per process)
        st = (torch.cuda.get_rng_state(dev) if dev.type == "cuda" else torch.get_rng_state()).numpy().tobytes()
        extra["default_generator"] = {
            "device": dev.type, "initial_seed": torch.cuda.initial_seed() if dev.type == "cuda" else torch.initial_seed(),
            "state_sha256_before_encode": hashlib.sha256(st).hexdigest()}
    encode, crop, randn = vae.encode, TF.crop, dvae.randn_tensor
    seen = {}

    def crop_w(*a, **k):
        r = crop(*a, **k)
        if r.dtype == torch.uint8 and "canvas" not in seen:
            seen["canvas"] = True
            rec.put("encode.canvas", r, "nchw")  # (1, 3, H, W) uint8, the operand of `/ 127.5`
        return r

    def encode_w(x, *a, **k):
        rec.put("encode.input", x, "ncthw")
        out = encode(x, *a, **k)
        seen["dist"] = out.latent_dist
        rec.put("encode.mean", out.latent_dist.mean, "ncthw")
        rec.put("encode.std", out.latent_dist.std, "ncthw")
        return out

    def randn_w(*a, **k):
        r = randn(*a, **k)
        rec.put("encode.eps_native", r, "ncthw")
        return r
    with patched(TF, "crop", crop_w), patched(vae, "encode", encode_w), patched(dvae, "randn_tensor", randn_w):
        latent, ph, pw = prepare_latents.encode_i2va_first_frame(pil, vae, dev, height=h, width=w)
    assert "canvas" in seen, "encode.canvas was not captured (crop not called on the uint8 canvas)"
    rec.put("encode.latent_native", latent, "keep")
    dist = seen["dist"]
    eps = load(inp, index, f"{case}.eps")  # [1][h][w][16] = (T, h, w, C)
    eps = eps.permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(device=dev, dtype=dist.mean.dtype)
    x = dist.mean + dist.std * eps
    rec.put("encode.latent_kit", x.squeeze(0).permute(1, 2, 3, 0) * vae.config.scaling_factor, "keep")
    return latent


def kernels(fn, dev) -> dict:
    """Kernel names (CUDA) or operator names (CPU) the native call launches, in first-launch order, with counts."""
    from torch.profiler import ProfilerActivity, profile
    from torch.autograd import DeviceType
    want = DeviceType.CUDA if dev.type == "cuda" else DeviceType.CPU
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with profile(activities=acts) as prof:
        fn()
        sync(dev)
    names = {}
    for e in prof.events():
        if e.device_type == want:
            names[e.name] = names.get(e.name, 0) + 1
    return {"activity": want.name, "launches": sum(names.values()), "names": [[k, v] for k, v in names.items()]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--upstream", required=True, help="kandinsky-6 checkout at eac06e39")
    ap.add_argument("--vae", required=True, help="the checkpoint's vae/ directory")
    ap.add_argument("--inputs", required=True, help="inputs.py output")
    ap.add_argument("--out", required=True, help="record set directory (appended per case)")
    ap.add_argument("--cases", default="s_dec,f_dec,s_enc,n_enc,n_dec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    a = ap.parse_args()
    dev = torch.device(a.device)
    oracle = dev.type == "cuda" and a.dtype == "float16"
    if dev.type != "cuda" and a.dtype != "float32":
        raise SystemExit("oracle: off-GPU runs are the float32 plumbing test (--device cpu --dtype float32)")
    if dev.type == "cuda" and a.dtype != "float16":
        raise SystemExit("oracle: the oracle is the native fp16 VAE; float32 is for the CPU plumbing test only")
    cases = [c for c in a.cases.split(",") if c]
    for c in cases:
        if c not in RAW:
            raise SystemExit(f"oracle: unknown case {c}")
        if not oracle and c.startswith("n_"):
            raise SystemExit("oracle: the plumbing test runs the s_* cases only")
    up, inp, out = Path(a.upstream), Path(a.inputs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((inp / "inputs.json").read_text())
    vae_meta = check_vae(Path(a.vae))
    mods = import_upstream(up)

    torch.set_float32_matmul_precision("high")  # factory.py:162, before construction; nothing else is set
    t0 = time.perf_counter()
    vae = mods[0].build_vae(str(a.vae), device=dev)
    if a.dtype == "float32":
        vae = vae.float()
    sync(dev)
    load_s = time.perf_counter() - t0
    env = environment(dev, oracle, next(vae.parameters()).dtype, up, vae_meta)
    env["vae_load_s"] = load_s
    merge_json(out / "env.json", env)

    rec = Records(out)
    summary = {}
    for case in cases:
        rec.begin(case)
        info = {"precision_at_start": precision(), "mem_get_info_at_start": mem_info(dev)}
        res = {"status": "started"}
        merge_json(out / "summary.json", {"cases": {case: res}})
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        hooks = Hooks(vae, rec)
        t0 = time.perf_counter()
        try:
            if case.endswith("_dec"):
                info["tiling"] = {}
                run_decode(case, vae, mods, inp, index, dev, rec, info["tiling"])
            else:
                run_encode(case, vae, mods, inp, index, dev, rec, info)
            sync(dev)
            res["status"] = "ok"
        except Exception as e:
            res["status"] = "error"
            res["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        res["wall_s"] = time.perf_counter() - t0
        hooks.remove()
        info["decoder_calls"] = hooks.decoder_calls
        if "tiling" in info:
            info["tiling"]["decoder_passes"] = hooks.decoder_calls
        res["checkpoints"] = rec.count
        if dev.type == "cuda":
            res["max_memory_allocated"] = torch.cuda.max_memory_allocated(dev)
            res["max_memory_reserved"] = torch.cuda.max_memory_reserved(dev)
        if case in PROFILE and res["status"] == "ok":
            fn = run_decode if case.endswith("_dec") else run_encode
            try:
                t1 = time.perf_counter()
                info["profiler"] = kernels(lambda: fn(case, vae, mods, inp, index, dev), dev)
                info["profiler"]["wall_s_profiled_unhooked"] = time.perf_counter() - t1
            except Exception as e:
                info["profiler"] = {"error": f"{type(e).__name__}: {e}"}
        merge_json(out / "env.json", {"cases": {case: info}})
        merge_json(out / "summary.json", {"oracle": oracle, "cases": {case: res}})
        summary[case] = res
    merge_json(out / "env.json", {"loaded_libs": loaded_libs()})
    print(json.dumps({"oracle": oracle, "out": str(out), "cases": summary}))
    sys.exit(0 if all(r["status"] == "ok" for r in summary.values()) else 1)


if __name__ == "__main__":
    main()
