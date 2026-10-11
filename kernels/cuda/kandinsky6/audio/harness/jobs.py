#!/usr/bin/env python3
"""Kandinsky-6 audio GPU qualification jobs: A (mel VAE decoder) and B (BigVGAN v2 vocoder), one GPU node each.

Every comparison is by float32 bit pattern (int32 view): it passes only with 0 unequal values; max abs diff and the
NaN-mask mismatch are diagnostics. Mismatches are data: the process exits 0 whenever the job ran and nonzero only on
infrastructure failure (refused native source or checkpoint, missing fatbin, CUDA fault, resource self-limit).
Weights are the real HF 1018d582 checkpoints (sha256-verified) loaded through the native MIT loaders, or seeded
synthetic weights with --weights synthetic (fallback smoke test). Inputs are seeded synthetic latents / mel.

  jobs.py A|B --fatbin audio.fatbin --native-src UPSTREAM --models MODELS --out OUT [--site-packages DIR]
  jobs.py A|B --dry-run-cpu --weights synthetic --native-src UPSTREAM --out OUT   (tiny, CPU, native paths only)
"""

import os
import site
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I does not add the script directory
if "--site-packages" in sys.argv[:-1]:  # the sealed prefix's site dir, explicit (-I drops PYTHONPATH)
    site.addsitedir(sys.argv[sys.argv.index("--site-packages") + 1])

import argparse  # noqa: E402
import copy  # noqa: E402
import ctypes  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import native  # noqa: E402
from k6a_cuda import ACC_FMA, ACC_MUL_ADD, DIV_RECIP, DIV_TRUE, CudaError, K6A, f32  # noqa: E402

HARNESS_FILES = ("jobs.py", "k6a_cuda.py", "native.py", "run_job.sh", "make_kit.sh", "MANIFEST.md")
GPU_LIMIT, RSS_LIMIT = 40 << 30, 100 << 30
SCALING_FACTOR, MEAN_VALUE, SILU_DIV, RESIDUAL_T, CLIP_ACT = 0.417, 0.0, 0.596, 0.3, 256.0
ACCS, DIVS = (ACC_FMA, ACC_MUL_ADD), (DIV_TRUE, DIV_RECIP)
SPECIALS = [0.0, -0.0, math.inf, -math.inf, math.nan, 1e-40, -1e-40, 1e-30, -1e-30, 20.0, -20.0, 88.0, -88.0, 89.0,
            -104.0, 1e30, -1e30, 3.4e38, -3.4e38]
CHECKPOINTS = {  # MODELS-relative path, size, sha256 (HF 1018d582)
    "audio_vae": ("audio_vae/diffusion_pytorch_model.safetensors", 1711158920,
                  "dfccdf9d0d8d3de270a23b097400f21adb91f4ec683bdc26369cb65b08a5af87"),
    "vocoder": ("vocoder/diffusion_pytorch_model.safetensors", 488692656,
                "abf0536824bdce8dcf8ee10b815770111f4b5d05d0122e1bab4b5719324edca7"),
}
DECODER_CONVS, VOCODER_TENSORS = 29, 667
# (MPConv1D path in the decoder, T it runs at as a multiple of the latent T)
CONV_NODES = [("conv_in", 1), ("mid.block_1.conv1", 1), ("up.1.block.0.nin_shortcut", 1), ("mid.attn_1.qkv", 1),
              ("conv_out", 2)]

FULL = {
    "A": dict(norm_C=(512, 1024, 2048), norm_T=(218, 436), packed=(1, 1, 2048, 3, 218), special=(1, 2048, 218),
              prep=[(2048, 40, 3), (2048, 2048, 3), (1024, 2048, 1), (6144, 2048, 1), (128, 512, 3)],
              elem=(2048, 436), sdpa=[(218, 2048), (436, 2048)], dec_dim=512, dec_T=218),
    "B": dict(stage=3, snake_T=4096, aa_T=3488, post_T=4097, conv_T=2048, convt_T=[436, 3488, 13952, 27904, 55808, 111616],
              pre_T=436, conv_post_T=223232, mel=(1, 128, 436), voc_ch=1536,
              wn=[("conv1d", 96, 96, 7, 1), ("conv_transpose1d", 96, 48, 4, 2)]),
}
DRY = {
    "A": dict(norm_C=(8, 16), norm_T=(8,), packed=(1, 1, 16, 3, 8), special=(1, 16, 8),
              prep=[(16, 40, 3), (16, 16, 3), (8, 16, 1), (48, 16, 1), (128, 8, 3)],
              elem=(16, 8), sdpa=[(8, 16)], dec_dim=8, dec_T=8),
    "B": dict(stage=1, snake_T=16, aa_T=16, post_T=9, conv_T=32, convt_T=[4, 32, 128, 256, 512, 1024], pre_T=6,
              conv_post_T=64, mel=(1, 128, 4), voc_ch=64, wn=[("conv1d", 4, 4, 7, 1), ("conv_transpose1d", 4, 2, 4, 2)]),
}
EXPECTED_VOCODER = dict(upsample_rates=[8, 4, 2, 2, 2, 2], upsample_kernel_sizes=[16, 8, 4, 4, 4, 4],
                        upsample_initial_channel=1536, resblock_kernel_sizes=[3, 7, 11],
                        resblock_dilation_sizes=[[1, 3, 5]] * 3, resblock="1", activation="snakebeta",
                        snake_logscale=True, use_tanh_at_final=False, use_bias_at_final=False, num_mels=128)


class Fatal(Exception):
    pass


def is_fatal(e):
    return isinstance(e, (CudaError, Fatal)) or type(e).__name__ == "AcceleratorError" or "CUDA error" in str(e)


def compare(got, want):
    """Bit comparison of two fp32 tensors (values, not strides). pass <=> 0 unequal int32 patterns."""
    got, want = got.detach().contiguous(), want.detach().contiguous()
    r = {"shape_got": list(got.shape), "shape_want": list(want.shape)}
    if got.shape != want.shape or got.dtype != torch.float32 or want.dtype != torch.float32:
        return dict(r, unequal=None, pass_=False, error="shape or dtype mismatch")
    want = want.to(got.device)
    fin = torch.isfinite(got) & torch.isfinite(want)
    d = (got.double() - want.double()).abs()[fin]
    r.update(unequal=int((got.view(torch.int32) != want.view(torch.int32)).sum()), numel=got.numel(),
             nan_mask_mismatch=int((torch.isnan(got) != torch.isnan(want)).sum()),
             max_abs_diff=float(d.max()) if d.numel() else 0.0)
    r["pass_"] = r["unequal"] == 0
    return r


def brief(c):
    return {k: c.get(k) for k in ("unequal", "max_abs_diff", "nan_mask_mismatch", "pass_")}


def best_of(results):
    """Key with the fewest unequal values (first on ties)."""
    ok = [(c["unequal"], i, k) for i, (k, c) in enumerate(results.items()) if c.get("unequal") is not None]
    return min(ok)[2] if ok else None


def plain(t):
    """A normal (non-inference) copy: the native decode runs under torch.inference_mode."""
    with torch.inference_mode(False):
        return t.detach().clone()


def tensor_bytes(t):
    c = t.detach().to("cpu").contiguous()
    return ctypes.string_at(c.data_ptr(), c.numel() * c.element_size()), c


def module_at(root, path):
    m = root
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


class Run:
    def __init__(self, a):
        self.a, self.dry, self.t0 = a, a.dry_run_cpu, time.monotonic()
        self.limit = a.time_limit_min * 60
        self.dev = torch.device("cpu" if self.dry else "cuda")
        self.S = (DRY if self.dry else FULL)[a.job]
        self.real = a.weights == "real"
        self.k = self.ns = self.vae = self.decode = self.voc = None
        self.cur, self.hashes, self.norm_totals = "setup", [], {}
        self.best = dict(norm=(1, 32, "s64", 1), silu_div=DIV_RECIP, res_div=DIV_RECIP, lat_div=DIV_RECIP,
                         avg_div=DIV_RECIP, chosen_from="defaults (no op-level result yet)")
        self.tensors = os.path.join(a.out, "tensors")
        self.doc = {"job": a.job, "route": native.ROUTE, "dry_run_cpu": self.dry, "weights": a.weights,
                    "pass_criterion": "0 unequal float32 bit patterns per comparison; max_abs_diff and "
                                      "nan_mask_mismatch are diagnostics",
                    "evidence_status": "qualification evidence only after review",
                    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "args": {k: str(v) for k, v in vars(a).items()}, "cases": [], "dumps": [], "finished": False}
        os.makedirs(self.tensors, exist_ok=True)

    def elapsed(self):
        return time.monotonic() - self.t0

    def gen(self, tag):
        return torch.Generator().manual_seed((self.a.seed << 32) ^ zlib.crc32(tag.encode()))

    def randn(self, shape, tag, scale=1.0, specials=False):
        x = torch.randn(shape, generator=self.gen(tag)) * scale
        if specials:
            x.view(-1)[:len(SPECIALS)] = torch.tensor(SPECIALS)
        return x.to(self.dev)

    def H(self, **ts):
        out = {}
        for n, t in ts.items():
            b, c = tensor_bytes(t)
            h = hashlib.sha256(b).hexdigest()
            out[n] = {"shape": list(c.shape), "dtype": str(c.dtype), "sha256": h}
            self.hashes.append(f"{self.cur}\t{n}\t{'x'.join(map(str, c.shape))}\t{h}")
        return out

    def dump(self, name, t):
        """Write t as raw little-endian fp32 to OUT/tensors/<name>.f32le; returns its record."""
        b, c = tensor_bytes(t)
        if c.dtype != torch.float32:
            raise TypeError(f"dump {name}: {c.dtype}")
        path = os.path.join(self.tensors, name + ".f32le")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b)
        rec = {"name": name, "file": os.path.relpath(path, self.a.out), "shape": list(c.shape),
               "sha256": hashlib.sha256(b).hexdigest(), "case": self.cur}
        self.doc["dumps"].append(rec)
        self.hashes.append(f"{self.cur}\tdump:{name}\t{'x'.join(map(str, c.shape))}\t{rec['sha256']}")
        return rec

    def sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()

    def check_limits(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        gpu = torch.cuda.max_memory_allocated() if self.dev.type == "cuda" else 0
        if rss > RSS_LIMIT or gpu > GPU_LIMIT:
            raise Fatal(f"resource self-limit: host peak RSS {rss} B, GPU peak {gpu} B")

    def case(self, name, fn, *args):
        if self.elapsed() > self.limit:
            self.doc["cases"].append({"name": name, "status": "skipped_time_limit"})
            self.flush()
            return None
        self.cur, t, rec = name, time.monotonic(), {"name": name}
        print(f"[{self.elapsed():7.1f}s] {name}", file=sys.stderr, flush=True)
        try:
            rec["result"] = fn(*args)
            self.sync()
            rec["status"] = "ok"
        except Exception as e:  # mismatch is data; only infrastructure faults are fatal
            rec.update(status="fatal" if is_fatal(e) else "error", error=f"{type(e).__name__}: {e}"[:2000],
                       traceback=traceback.format_exc()[-4000:])
        rec["seconds"] = round(time.monotonic() - t, 3)
        self.doc["cases"].append(rec)
        self.flush()
        if rec["status"] == "fatal":
            raise Fatal(f"{name}: {rec['error']}")
        self.check_limits()
        return rec.get("result")

    def flush(self):
        base = os.path.join(self.a.out, f"results-{self.a.job}")
        self.doc["elapsed_s"] = round(self.elapsed(), 3)
        self.doc["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if self.dev.type == "cuda":
            self.doc["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        with open(base + ".json.tmp", "w") as f:
            json.dump(self.doc, f, indent=1, default=str)
        os.replace(base + ".json.tmp", base + ".json")
        with open(base + "-sha256.tsv", "w") as f:
            f.write("case\ttensor\tshape\tsha256\n" + "".join(h + "\n" for h in self.hashes))


def set_numerics():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)


def env_record(R):
    e = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
         "machine": platform.machine(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
         "torch_git": getattr(torch.version, "git_version", None),
         "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
         "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
         "float32_matmul_precision": torch.get_float32_matmul_precision(),
         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
         "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
         "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "grad_enabled": torch.is_grad_enabled(),
         "harness_sha256": {f: native.sha256_file(os.path.join(HERE, f)) for f in HARNESS_FILES
                            if os.path.isfile(os.path.join(HERE, f))}}
    if R.dev.type == "cuda":
        p = torch.cuda.get_device_properties(R.dev)
        e.update(device_name=p.name, capability=list(torch.cuda.get_device_capability(R.dev)),
                 total_memory=p.total_memory, arch_list=torch.cuda.get_arch_list(),
                 sdp_flags={n: getattr(torch.backends.cuda, n)() for n in
                            ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled",
                             "cudnn_sdp_enabled") if hasattr(torch.backends.cuda, n)})
    if R.k:
        e.update(cuda_driver_version=R.k.drv.driver_version(), driver_backend=R.k.backend,
                 fatbin=R.a.fatbin, fatbin_sha256=hashlib.sha256(R.k.image).hexdigest())
    return e


def checkpoint_path(R, name):
    return Path(R.a.models) / CHECKPOINTS[name][0]


def case_verify_checkpoints(R):
    out = {}
    for name, (rel, size, sha) in CHECKPOINTS.items():
        p = checkpoint_path(R, name)
        got_size = p.stat().st_size if p.is_file() else None
        got = native.sha256_file(p) if got_size == size else None
        out[name] = {"path": str(p), "size": got_size, "sha256": got}
        if got != sha:
            raise Fatal(f"checkpoint {p}: size {got_size} sha256 {got}, expected {size} {sha}")
    return out


def safetensor_keys(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        return list(f.keys())


def safetensor_get(path, keys):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in keys}


# ================================================================================================ job A: decoder

NORM_REDUCTIONS = [(0, 32), (1, 32)] + [(2, 1 << p) for p in range(1, 11)]
NORM_VARIANTS = [(o, r, sm, fu) for o, r in NORM_REDUCTIONS for sm in ("s64", "s32") for fu in (0, 1)]


def norm_s(N, mode):
    """s = fp32(sqrt(1 / N)): sqrt in double then rounded (s64), or the fp32 sqrt of fp32(1 / N) (s32)."""
    return f32(math.sqrt(1.0 / N)) if mode == "s64" else f32(math.sqrt(f32(1.0 / N)))


def vname(v):
    o, r, sm, fu = v
    return f"order{o}" + (f"-R{r}" if o == 2 else "") + f"-{sm}-den_{'fused' if fu else 'split'}"


def norm_eps(R):
    eps = R.a.norm_eps if R.a.norm_eps is not None else R.ns.normalize_eps
    if eps is None:
        raise RuntimeError("no eps: native normalize has no signature default and --norm-eps was not given")
    return eps


def k_normalize(R, x, axis, v):
    o, r, sm, fu = v
    n = R.k.l2norm(x, axis=axis, order=o, R=r)
    return R.k.normalize_apply(x, n, axis=axis, s=norm_s(x.shape[axis], sm), eps=norm_eps(R), den_fused=fu)


def case_norm(R, x, axis):
    want = R.ns.normalize(x, dim=axis)
    res = {"axis": axis, "inputs": R.H(x=x), "native": R.H(y=want)}
    if R.k:
        res["eps_used"] = norm_eps(R)
        res["variants"] = {}
        for v in NORM_VARIANTS:
            c = brief(compare(k_normalize(R, x, axis, v), want))
            res["variants"][vname(v)] = c
            tot = R.norm_totals.get(v, 0)
            R.norm_totals[v] = None if tot is None or c["unequal"] is None else tot + c["unequal"]
        res["zero_variants"] = [n for n, c in res["variants"].items() if c["pass_"]]
    return res


def summarize_norm(R):
    if not R.norm_totals:
        return {"note": "no kernel results"}
    ok = [(t, i, v) for i, (v, t) in enumerate(R.norm_totals.items()) if t is not None]
    if ok:
        R.best["norm"] = min(ok)[2]
    tot = {vname(v): t for v, t in R.norm_totals.items()}
    return {"zero_in_all_cases": [n for n, t in tot.items() if t == 0], "total_unequal": tot,
            "best": vname(R.best["norm"])}


def special_inputs(R, shape):
    g = R.gen("special-ldr")
    ldr = torch.randn(shape, generator=g) * torch.pow(10.0, torch.empty(shape).uniform_(-18.0, 18.0, generator=g))
    imp = torch.zeros(shape)
    for t in range(shape[2]):
        imp[0, (7 * t) % shape[1], t] = 1.0 if t % 2 else 1e4
    return {"zero": torch.zeros(shape), "tiny": torch.randn(shape, generator=R.gen("special-tiny")) * 1e-25,
            "large_dynamic_range": ldr, "impulse": imp}


def decoder_convs(R):
    return [(n, m) for n, m in R.vae.decoder.named_modules() if isinstance(m, R.ns.MPConv1D)]


def case_decoder_load(R):
    """Native load order: build on CPU, load (real: AutoEncoderModule from MODELS/audio_vae), remove_weight_norm on
    CPU, dump the CPU-prepared decoder weights, then .to(device)."""
    if R.real:
        ae = R.ns.AutoEncoderModule(vae_ckpt_path=str(checkpoint_path(R, "audio_vae").parent), mode="44k",
                                    need_vae_encoder=False, need_vae_decoder=True)
        R.vae, R.decode, how = ae.vae, ae.decode, "native AutoEncoderModule (load_state_dict, remove_weight_norm, CPU)"
    else:
        vae = R.ns.VAE(data_dim=128, embed_dim=40, hidden_dim=R.S["dec_dim"]).eval()
        for name, p in vae.decoder.named_parameters():
            p.copy_(torch.randn(p.shape, generator=R.gen("dec:" + name)) * (0.1 if "learnable_gain" in name else 1.0))
        vae.remove_weight_norm()
        del vae.encoder
        R.vae, R.decode, how = vae, (lambda z: vae.decode(z)), "synthetic seeded weights, native remove_weight_norm on CPU"
    convs = decoder_convs(R)
    if not R.dry and len(convs) != DECODER_CONVS:
        raise RuntimeError(f"{len(convs)} decoder convs, expected {DECODER_CONVS}")
    dumps = [R.dump(f"prepared_cpu/vae.decoder.{n}.weight", m.weight) for n, m in convs]
    gain = R.vae.decoder.learnable_gain.detach()
    res = {"how": how, "prepared_on": str(convs[0][1].weight.device), "convs": len(convs), "dumps": dumps,
           "learnable_gain": {"value": float(gain), "f32_bits": f"{int(gain.view(torch.int32)):#010x}"},
           "data_std": R.H(std=R.vae.data_std), "data_mean": R.H(mean=R.vae.data_mean)}
    R.vae.to(R.dev).eval()
    return res


def mpconv_prepared(R, state, cout, cin, kk, where):
    m = R.ns.MPConv1D(cin, cout, kernel_size=kk).eval()
    if set(state) == {"weight"}:  # synthetic: only the stored weight is given
        m.weight.copy_(state["weight"])
    else:
        m.load_state_dict(state, strict=True)
    m = m.to(where)
    m.remove_weight_norm()
    return m.weight.detach()


def case_weight_prep_real(R):
    """Each of the 29 stored decoder conv weights prepared again on CPU and on the device by fresh native MPConv1D
    modules, against the loaded CPU-prepared weight."""
    path = checkpoint_path(R, "audio_vae")
    keys = safetensor_keys(path)
    out = {}
    for n, m in decoder_convs(R):
        prefix = f"vae.decoder.{n}."
        state = {k[len(prefix):]: v for k, v in safetensor_get(path, [k for k in keys if k.startswith(prefix)]).items()}
        cout, cin, kk = m.weight.shape
        loaded = m.weight.detach()
        rec = {"stored": R.H(**state), "cpu_reprepare": brief(compare(mpconv_prepared(R, state, cout, cin, kk, "cpu"), loaded))}
        if R.dev.type == "cuda":
            rec["device_prepare_vs_cpu"] = brief(compare(mpconv_prepared(R, state, cout, cin, kk, R.dev), loaded))
        out[n] = rec
    return out


def case_weight_prep_synthetic(R, shape):
    cout, cin, kk = shape
    stored = torch.randn(shape, generator=R.gen(f"prep-{shape}"))
    res, prepared = {"inputs": R.H(stored=stored)}, {}
    for where in dict.fromkeys(["cpu", str(R.dev)]):
        w = mpconv_prepared(R, {"weight": stored}, cout, cin, kk, where)
        prepared[where] = w
        res[where] = {"ran_on": str(w.device), "prepared": R.H(w=w)}
    if len(prepared) == 2:
        res["device_vs_cpu"] = compare(prepared[str(R.dev)], prepared["cpu"])
    return res


def case_conv(R, path, tmul):
    m = module_at(R.vae.decoder, path)
    w = m.weight.detach()
    cout, cin, kk = w.shape
    T = tmul * R.S["dec_T"]
    x = R.randn((1, cin, T), f"conv-x-{path}")
    res = {"T": T, "weight_shape": list(w.shape), "pad_assumed": kk // 2, "inputs": R.H(x=x, w=w)}
    for gname, gain in (("default", None), ("1.3", 1.3), ("loaded_plus_one", R.vae.decoder.learnable_gain + 1)):
        want = m(x) if gain is None else m(x, gain=gain)
        rec = {"native": R.H(y=want)}
        if R.k:
            wk = w
            if gain is not None:
                rec["gain_f32"] = gv = f32(float(gain))
                wk = R.k.scale_weights(w, gain=gv)
                rec["scaled_weight_vs_torch_w_times_gain"] = brief(compare(wk, w * gain))
            for acc in ACCS:
                rec[f"acc{acc}"] = compare(R.k.conv1d(x, wk, None, stride=1, pad=kk // 2, dil=1, acc=acc), want)
        res[gname] = rec
    return res


def div_cases(R, want, run):
    rec = {"native": R.H(y=want)}
    if R.k:
        for div in DIVS:
            rec[f"div{div}"] = compare(run(div), want)
        rec["best"] = best_of({d: rec[f"div{d}"] for d in DIVS})
    return rec


def case_silu(R):
    x = R.randn((1,) + R.S["elem"], "silu-x", 4.0, specials=True)
    rec = dict(div_cases(R, R.ns.mp_silu(x), lambda d: R.k.scaled_silu(x, divisor=SILU_DIV, div=d)), inputs=R.H(x=x))
    if R.k:
        R.best["silu_div"] = rec["best"]
    return rec


def case_mp_sum(R):
    a = R.randn((1,) + R.S["elem"], "mpsum-a", 3.0, specials=True)
    b = R.randn((1,) + R.S["elem"], "mpsum-b", 3.0)
    rec = div_cases(R, R.ns.mp_sum(a, b, t=RESIDUAL_T), lambda d: R.k.mp_residual(a, b, t=RESIDUAL_T, div=d))
    rec["inputs"] = R.H(a=a, b=b)
    if R.k:
        R.best["res_div"] = rec["best"]
    return rec


def affine_buffers(R):
    return R.vae.data_std.reshape(-1).contiguous(), R.vae.data_mean.reshape(-1).contiguous()


def case_elementwise(R):
    C, T = R.S["elem"]
    k, res = R.k, {}
    rows = R.randn((1, T, 40), "latent-rows", 2.0, specials=True)
    res["latent_unscale"] = dict(div_cases(R, native.latent_unscale(rows, SCALING_FACTOR, MEAN_VALUE),
                                           lambda d: k.latent_unscale(rows, scale=SCALING_FACTOR, div=d, mean=MEAN_VALUE)),
                                 inputs=R.H(rows=rows))
    if k:
        R.best["lat_div"] = res["latent_unscale"]["best"]
    std, mean = affine_buffers(R)
    x = R.randn((1, 128, 2 * T), "mel-x", 3.0, specials=True)
    want = native.mel_affine(x, std, mean)
    res["mel_affine"] = {"inputs": R.H(x=x), "native": R.H(y=want)}
    if k:
        res["mel_affine"]["kernel"] = compare(k.mel_affine(x, std, mean), want)
    x = R.randn((1, C, T), "clamp-x", 300.0, specials=True)
    for lo, hi in ((-CLIP_ACT, CLIP_ACT), (-1.0, 1.0)):
        want = native.clamp(x, lo, hi)
        r = res[f"clamp_{hi:g}"] = {"inputs": R.H(x=x), "native": R.H(y=want)}
        if k:
            r["kernel"] = compare(k.clamp(x, lo=lo, hi=hi), want)
    x = R.randn((1, C // 2, T), "upsample-x", 1.0, specials=True)
    want = native.upsample_nearest_exact2(x)
    res["upsample_nearest_exact2"] = {"inputs": R.H(x=x), "native": R.H(y=want)}
    if k:
        res["upsample_nearest_exact2"]["kernel"] = compare(k.upsample_nearest_exact2(x), want)
    y, h = R.randn((1, 3 * C, T), "qkv-y"), R.randn((1, 1, T, C), "merge-h")
    res["layout"] = {"inputs": R.H(y=y, h=h)}
    if k:
        res["layout"].update({n: brief(compare(a, b)) for n, a, b in
                              zip("qkv", k.qkv_split(y.view(1, 1, C, 3, T)), native.qkv_split(y, 1))})
        res["layout"]["attn_merge"] = brief(compare(k.attn_merge(h), native.attn_merge(h)))
    return res


def sdpa_c(D):
    """Candidate operand scale: fp32(sqrt(1 / sqrt(D))) computed in double (the math backend's sqrt(scale))."""
    return f32(math.sqrt(1.0 / math.sqrt(D)))


def sdpa_choice(q, k, v):
    try:
        i = int(torch._fused_sdp_choice(q, k, v))
        names = {int(b.value): n for n, b in torch.nn.attention.SDPBackend.__members__.items()}
        return {"index": i, "backend": names.get(i)}
    except Exception as e:
        return {"error": repr(e)[:300]}


def case_sdpa(R, T, D):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    y = R.randn((1, 3 * D, T), f"sdpa-{T}-{D}")
    q, k, v = R.ns.normalize(y.reshape(1, 1, D, 3, T), dim=2).unbind(3)  # exactly as AttnBlock1D builds them
    qkv = tuple(t.permute(0, 1, 3, 2) for t in (q, k, v))
    res = {"inputs": R.H(qkv_conv_out=y), "candidate_c": sdpa_c(D), "note": "k6a_sdpa_math is a candidate"}
    for lname, (q, k, v) in (("native_strided", qkv), ("contiguous", tuple(t.contiguous() for t in qkv))):
        default = native.sdpa(q, k, v)
        lr = {"q_stride": list(q.stride()), "fused_sdp_choice": sdpa_choice(q, k, v), "default": R.H(o=default)["o"]}
        outs = {"default": default}
        for b in ("MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"):
            try:
                with sdpa_kernel([getattr(SDPBackend, b)]):
                    outs[b] = native.sdpa(q, k, v)
                lr[b] = {"ok": True, "out": R.H(o=outs[b])["o"], "vs_default": brief(compare(outs[b], default))}
            except Exception as e:
                lr[b] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
        lr["backends_bit_equal_to_default"] = [b for b in outs if b != "default" and lr[b]["vs_default"]["pass_"]]
        if R.k:
            km = R.k.sdpa_math(*(t.contiguous() for t in (q, k, v)), c=sdpa_c(D))
            lr["k6a_sdpa_math_vs"] = {n: compare(km, o) for n, o in outs.items()}
        res[lname] = lr
    return res


# --- decoder chain

def dec_nodes(dec):
    """(name, module, clamp after) in Decoder1D.forward order, conv_out excluded."""
    n = [("conv_in", dec.conv_in, False), ("mid.block_1", dec.mid.block_1, False),
         ("mid.attn_1", dec.mid.attn_1, False), ("mid.block_2", dec.mid.block_2, True)]
    for lvl in reversed(range(dec.num_layers)):
        if len(dec.up[lvl].attn):
            raise RuntimeError("unexpected up-level attention")
        n += [(f"up.{lvl}.block.{b}", dec.up[lvl].block[b], True) for b in range(dec.num_res_blocks + 1)]
        if lvl in dec.down_layers:
            n.append((f"up.{lvl}.upsample", dec.up[lvl].upsample, False))
    return n


def capture(mods):
    cap, hooks = {}, []
    for name, m in mods:
        hooks.append(m.register_forward_hook(
            lambda mod, inp, out, n=name: cap.__setitem__(n, (plain(inp[0]), plain(out)))))
    return cap, hooks


class DecoderKernels:
    def __init__(self, R, cfg):
        self.R, self.k, self.cfg = R, R.k, cfg

    def conv(self, x, m, gain=None):
        w = m.weight.detach()
        if gain is not None:
            w = self.k.scale_weights(w, gain=gain)
        return self.k.conv1d(x.contiguous(), w, None, stride=1, pad=w.shape[2] // 2, dil=1, acc=self.cfg["acc"])

    def silu(self, x):
        return self.k.scaled_silu(x.contiguous(), divisor=SILU_DIV, div=self.cfg["silu_div"])

    def res(self, a, b):
        return self.k.mp_residual(a, b, t=RESIDUAL_T, div=self.cfg["res_div"])

    def node(self, m, x):
        x, ns = x.contiguous(), self.R.ns
        if isinstance(m, ns.MPConv1D):
            return self.conv(x, m)
        if isinstance(m, ns.ResnetBlock1D):
            xn = k_normalize(self.R, x, 1, self.cfg["norm"])
            h = self.conv(self.silu(self.conv(self.silu(xn), m.conv1)), m.conv2)
            return self.res(self.conv(xn, m.nin_shortcut) if m.in_dim != m.out_dim else xn, h)
        if isinstance(m, ns.AttnBlock1D):
            y = self.conv(x, m.qkv)
            B, C3, T = y.shape
            D = C3 // 3 // m.num_heads
            q, k, v = self.k.qkv_split(k_normalize(self.R, y.view(B, m.num_heads, D, 3, T), 2, self.cfg["norm"]))
            return self.res(x, self.conv(self.k.attn_merge(self.k.sdpa_math(q, k, v, c=sdpa_c(D))), m.proj_out))
        if isinstance(m, ns.Upsample1D):
            return self.conv(self.k.upsample_nearest_exact2(x), m.conv)
        raise TypeError(type(m).__name__)


def case_decoder_chain(R):
    dec, T = R.vae.decoder, R.S["dec_T"]
    nodes = dec_nodes(dec)
    cap, hooks = capture([(n, m) for n, m, _ in nodes] + [("conv_out", dec.conv_out)])
    rows = R.randn((1, T, 40), "dec-rows")  # the sampler's per-sample latent rows [B, T, 40]
    z = native.latent_unscale(rows, SCALING_FACTOR, MEAN_VALUE)  # the transposed view the native decoder receives
    try:
        mel = plain(R.decode(z))  # decoder + VAE.unnormalize
    finally:
        for h in hooks:
            h.remove()
    std, mean = affine_buffers(R)
    gain = f32(float(dec.learnable_gain + 1))
    res = {"T": T, "weights": R.a.weights,
           "candidates": "normalization (k6a_l2norm + k6a_normalize_apply) and attention (k6a_sdpa_math) are "
                         "candidate kernels; the variants are the best op-level ones of this run",
           "inputs": R.H(rows=rows), "native": R.H(z=z, mel=mel), "conv_out_gain_f32": gain,
           "dumps": [R.dump("decoder/latent_rows", rows), R.dump("decoder/native_z", z), R.dump("decoder/native_mel", mel)]}
    res["native_nodes"] = {n: R.H(out=o)["out"] for n, (_, o) in cap.items()}
    if not R.k:
        return res
    last = nodes[-1][0]
    for acc in ACCS:
        cfg = dict(R.best, acc=acc)
        K = DecoderKernels(R, cfg)
        fr, tf = {}, {}
        h = R.k.latent_unscale(rows, scale=SCALING_FACTOR, div=cfg["lat_div"], mean=MEAN_VALUE)
        fr["latent_unscale"] = brief(compare(h, z))
        for name, m, clamped in nodes:
            h = K.node(m, h)
            fr[name], tf[name] = brief(compare(h, cap[name][1])), brief(compare(K.node(m, cap[name][0]), cap[name][1]))
            if clamped:
                want = cap[name][1].clamp(-CLIP_ACT, CLIP_ACT)
                h = R.k.clamp(h, lo=-CLIP_ACT, hi=CLIP_ACT)
                fr[name + ".clamp"] = brief(compare(h, want))
                tf[name + ".clamp"] = brief(compare(R.k.clamp(cap[name][1], lo=-CLIP_ACT, hi=CLIP_ACT), want))
        h = K.silu(h)
        fr["silu"] = brief(compare(h, cap["conv_out"][0]))
        tf["silu"] = brief(compare(K.silu(cap[last][1].clamp(-CLIP_ACT, CLIP_ACT)), cap["conv_out"][0]))
        h = K.conv(h, dec.conv_out, gain=gain)
        fr["conv_out"] = brief(compare(h, cap["conv_out"][1]))
        tf["conv_out"] = brief(compare(K.conv(cap["conv_out"][0], dec.conv_out, gain=gain), cap["conv_out"][1]))
        h = R.k.mel_affine(h, std, mean)
        tf["mel_affine"] = brief(compare(R.k.mel_affine(cap["conv_out"][1].contiguous(), std, mean), mel))
        res[f"acc{acc}"] = {"config": dict(cfg, norm=vname(cfg["norm"])), "free_running": fr, "teacher_forced": tf,
                            "mel": compare(h, mel), "kernel_mel": R.dump(f"decoder/kernel_mel_acc{acc}", h)}
    return res


def job_a(R):
    S = R.S
    if R.real:
        R.case("checkpoints/verify", case_verify_checkpoints, R)
    if R.case("decoder/load_prepare_dump", case_decoder_load, R) is None:
        return
    for C in S["norm_C"]:
        for T in S["norm_T"]:
            R.case(f"normalize/residual[1,{C},{T}]/axis1", case_norm, R, R.randn((1, C, T), f"norm-{C}-{T}"), 1)
    R.case(f"normalize/packed{list(S['packed'])}/axis2", case_norm, R, R.randn(S["packed"], "norm-packed"), 2)
    for name, x in special_inputs(R, S["special"]).items():
        R.case(f"normalize/{name}{list(S['special'])}/axis1", case_norm, R, x.to(R.dev), 1)
    R.case("normalize/summary", summarize_norm, R)
    if R.real:
        R.case("mpconv_prepare/checkpoint_29_convs", case_weight_prep_real, R)
    for shape in S["prep"]:
        R.case(f"mpconv_prepare/synthetic{list(shape)}", case_weight_prep_synthetic, R, shape)
    for path, tmul in CONV_NODES:
        R.case(f"conv1d_prepared/{path}", case_conv, R, path, tmul)
    R.case("scaled_silu", case_silu, R)
    R.case("mp_residual", case_mp_sum, R)
    R.case("elementwise", case_elementwise, R)
    for T, D in S["sdpa"]:
        R.case(f"sdpa_preflight/T{T}/D{D}", case_sdpa, R, T, D)
    if R.k:
        R.best["chosen_from"] = "op-level results of this run"
    R.case(f"decoder_chain/T{S['dec_T']}", case_decoder_chain, R)


# ================================================================================================ job B: vocoder

def synth_params(R, model, prefix):
    for name, p in model.named_parameters():
        g = R.gen(prefix + name)
        scale = 0.3 if name.endswith((".alpha", ".beta")) else 0.05 if name.endswith("bias") else 0.01
        p.copy_(torch.randn(p.shape, generator=g) * scale)


def act_k(R, x, a, acc):
    if not (isinstance(a.act, R.ns.SnakeBeta) and a.act.alpha_logscale):
        raise TypeError("kernel chain supports SnakeBeta with log-scale parameters only")
    up = R.k.aa_up2(x.contiguous(), a.upsample.filter.reshape(-1).contiguous(), acc=acc)
    s = R.k.snake_beta_log(up, a.act.alpha.detach(), a.act.beta.detach(), eps=f32(a.act.no_div_by_zero))
    return R.k.aa_down2(s, a.downsample.lowpass.filter.reshape(-1).contiguous(), acc=acc)


def conv_k(R, x, m, acc):
    x, w, b = x.contiguous(), m.weight.detach(), None if m.bias is None else m.bias.detach()
    if m.groups != 1 or m.padding_mode != "zeros":
        raise TypeError("groups 1, zero padding only")
    if isinstance(m, nn.ConvTranspose1d):
        if m.output_padding[0] != 0 or m.dilation[0] != 1:
            raise TypeError("conv_transpose1d: output_padding 0, dilation 1 only")
        return R.k.conv_transpose1d(x, w, b, stride=m.stride[0], pad=m.padding[0], acc=acc)
    return R.k.conv1d(x, w, b, stride=m.stride[0], pad=m.padding[0], dil=m.dilation[0], acc=acc)


def amp_k(R, x, blk, acc, trace=None):
    for i, (c1, c2, a1, a2) in enumerate(zip(blk.convs1, blk.convs2, blk.activations[::2], blk.activations[1::2])):
        xt = x
        for name, f in ((f"activations.{2 * i}", lambda t: act_k(R, t, a1, acc)), (f"convs1.{i}", lambda t: conv_k(R, t, c1, acc)),
                        (f"activations.{2 * i + 1}", lambda t: act_k(R, t, a2, acc)), (f"convs2.{i}", lambda t: conv_k(R, t, c2, acc))):
            xt = f(xt)
            if trace:
                trace(name, xt)
        x = R.k.add(xt, x)
    return x


def case_vocoder_load(R):
    h_bundled = R.ns.load_hparams(R.ns.vocoder_config)
    if R.real:
        ck = checkpoint_path(R, "vocoder")
        keys = safetensor_keys(ck)
        if (ck.parent / "config.json").is_file():
            model, how = R.ns.load_bigvgan_v2(str(ck.parent)), "native load_bigvgan_v2(MODELS/vocoder)"
        else:  # load_bigvgan_v2's own steps (MIT autoencoder.py) with the bundled config: config.json is absent
            from safetensors.torch import load_file
            model, state = R.ns.BigVGAN(h_bundled), load_file(str(ck))
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError:
                model.remove_weight_norm()
                model.load_state_dict(state, strict=True)
            model.remove_weight_norm()
            how = "load_bigvgan_v2 steps with the bundled config.json (MODELS/vocoder/config.json absent)"
        bad = [k for k in keys if k.endswith(("weight_g", "weight_v")) or "parametrizations" in k]
        model_keys = set(model.state_dict())
        keyset = {"tensors": len(keys), "expected_tensors": VOCODER_TENSORS, "weight_norm_keys": bad,
                  "missing_in_file": sorted(model_keys - set(keys))[:20], "extra_in_file": sorted(set(keys) - model_keys)[:20]}
        if bad or keyset["missing_in_file"] or keyset["extra_in_file"] or len(keys) != VOCODER_TENSORS:
            raise RuntimeError(f"vocoder checkpoint key set assertion failed: {keyset}")
    else:
        h = h_bundled
        if R.dry:
            h["upsample_initial_channel"] = R.S["voc_ch"]
        model, keyset, how = R.ns.BigVGAN(h), None, "synthetic seeded weights after native remove_weight_norm"
        model.remove_weight_norm()
        synth_params(R, model, "voc:")
    hp = dict(model.h)
    R.voc = model.to(R.dev).eval()
    return {"how": how, "key_set": keyset, "no_fold_arithmetic": "plain folded weight tensors loaded as stored",
            "hparams_differs_from_expected": {k: (hp.get(k), v) for k, v in EXPECTED_VOCODER.items() if hp.get(k) != v},
            "hparams": hp}


def stage_block(R, j):
    st = R.S["stage"]
    return R.voc.resblocks[3 * st + j]


def case_snake(R):
    m = stage_block(R, 0).activations[0].act
    x = R.randn((1, m.in_features, R.S["snake_T"]), "snake-x", 3.0, specials=True)
    want = m(x)
    res = {"module": f"resblocks.{3 * R.S['stage']}.activations.0.act", "eps_f32": f32(m.no_div_by_zero),
           "inputs": R.H(x=x, alpha=m.alpha, beta=m.beta), "native": R.H(y=want)}
    if R.k:
        res["kernel"] = compare(R.k.snake_beta_log(x, m.alpha.detach(), m.beta.detach(), eps=f32(m.no_div_by_zero)), want)
    return res


def case_aa(R, which, filt):
    a = stage_block(R, 0).activations[0] if which == "stage" else R.voc.activation_post
    if filt == "synthetic":
        a = copy.deepcopy(a)
        a.upsample.filter.copy_(torch.randn(a.upsample.filter.shape, generator=R.gen("aa-fu")).to(R.dev) * 0.3)
        a.downsample.lowpass.filter.copy_(torch.randn(a.downsample.lowpass.filter.shape, generator=R.gen("aa-fd")).to(R.dev) * 0.3)
    C, T = a.act.in_features, R.S["aa_T"] if which == "stage" else R.S["post_T"]
    x = R.randn((1, C, T), f"aa-x-{which}", 2.0)
    x2 = R.randn((1, C, 2 * T + 1), f"aa-x2-{which}", 2.0)  # 2T and 2T + 1 (odd) for the down-sampler
    even = x2[..., :-1].contiguous()
    up, down, down_odd, full = a.upsample(x), a.downsample(even), a.downsample(x2), a(x)
    res = {"filters": filt, "inputs": R.H(x=x, x2=x2, filter_up=a.upsample.filter, filter_down=a.downsample.lowpass.filter),
           "native": R.H(up=up, down=down, down_odd=down_odd, activation1d=full)}
    if R.k:
        fu, fd = a.upsample.filter.reshape(-1).contiguous(), a.downsample.lowpass.filter.reshape(-1).contiguous()
        for acc in ACCS:
            res[f"acc{acc}"] = {"aa_up2": compare(R.k.aa_up2(x, fu, acc=acc), up),
                                "aa_down2": compare(R.k.aa_down2(even, fd, acc=acc), down),
                                "aa_down2_odd": compare(R.k.aa_down2(x2, fd, acc=acc), down_odd),
                                "activation1d": compare(act_k(R, x, a, acc), full)}
    return res


def case_conv_module(R, name, m, T):
    x = R.randn((1, m.in_channels, T), f"conv-x-{name}")
    want = m(x)
    res = {"module": name, "geometry": dict(cin=m.in_channels, cout=m.out_channels, K=m.kernel_size[0],
                                            stride=m.stride[0], pad=m.padding[0], dil=m.dilation[0], T=T,
                                            bias=m.bias is not None),
           "inputs": R.H(x=x, w=m.weight, **({} if m.bias is None else {"b": m.bias})), "native": R.H(y=want)}
    if R.k:
        for acc in ACCS:
            res[f"acc{acc}"] = compare(conv_k(R, x, m, acc), want)
    return res


def case_vocoder_elementwise(R):
    shape = (1, stage_block(R, 0).activations[0].act.in_features, R.S["snake_T"])
    r0, r1, r2 = (R.randn(shape, f"avg-r{i}", 2.0, specials=(i == 0)) for i in range(3))
    res = {"avg3": dict(div_cases(R, native.avg3(r0, r1, r2, R.voc.num_kernels),
                                  lambda d: R.k.avg3(r0, r1, r2, divisor=float(R.voc.num_kernels), div=d)),
                        inputs=R.H(r0=r0, r1=r1, r2=r2))}
    if R.k:
        R.best["avg_div"] = res["avg3"]["best"]
    want = native.add(r0, r1)
    res["add"] = {"native": R.H(y=want), **({"kernel": compare(R.k.add(r0, r1), want)} if R.k else {})}
    x = R.randn(shape, "vclamp-x", 2.0, specials=True)
    want = native.clamp(x, -1.0, 1.0)
    res["clamp_1"] = {"inputs": R.H(x=x), "native": R.H(y=want),
                      **({"kernel": compare(R.k.clamp(x, lo=-1.0, hi=1.0), want)} if R.k else {})}
    return res


def case_amp(R):
    blk = stage_block(R, 1)
    mods = [(f"activations.{i}", a) for i, a in enumerate(blk.activations)]
    mods += [(f"convs1.{i}", c) for i, c in enumerate(blk.convs1)] + [(f"convs2.{i}", c) for i, c in enumerate(blk.convs2)]
    cap, hooks = capture(mods)
    x = R.randn((1, blk.convs1[0].in_channels, R.S["conv_T"]), "amp-x")
    try:
        want = blk(x)
    finally:
        for hk in hooks:
            hk.remove()
    res = {"module": f"resblocks.{3 * R.S['stage'] + 1}", "K": blk.convs1[0].kernel_size[0],
           "inputs": R.H(x=x), "native": R.H(y=want)}
    if R.k:
        for acc in ACCS:
            steps = {}
            out = amp_k(R, x, blk, acc, trace=lambda n, t: steps.__setitem__(n, brief(compare(t, cap[n][1]))))
            res[f"acc{acc}"] = {"steps_free_running": steps, "out": compare(out, want)}
    return res


def case_vocoder_chain(R):
    model = R.voc
    nk, nu = model.num_kernels, model.num_upsamples
    mods = [("conv_pre", model.conv_pre)] + [(f"ups.{i}", model.ups[i][0]) for i in range(nu)]
    mods += [(f"resblocks.{j}", b) for j, b in enumerate(model.resblocks)]
    mods += [("activation_post", model.activation_post), ("conv_post", model.conv_post)]
    cap, hooks = capture(mods)
    mel = R.randn(R.S["mel"], "voc-mel", 2.0) - 5.0  # deterministic log-mel-like input [1, 128, frames]
    try:
        pcm = plain(model(mel))
    finally:
        for hk in hooks:
            hk.remove()
    avg_native = [cap[f"ups.{i + 1}"][0] if i + 1 < nu else cap["activation_post"][0] for i in range(nu)]
    res = {"weights": R.a.weights, "use_tanh_at_final": model.use_tanh_at_final,
           "use_bias_at_final": model.use_bias_at_final, "inputs": R.H(mel=mel), "native": R.H(pcm=pcm),
           "native_stages": {n: R.H(out=o)["out"] for n, (_, o) in cap.items()},
           "dumps": [R.dump("vocoder/mel", mel), R.dump("vocoder/native_pcm", pcm)]}
    if not R.k:
        return res
    if nk != 3 or model.use_tanh_at_final:
        raise RuntimeError("kernel chain expects 3 AMP kernels per stage and the clamp(-1, 1) ending")
    div = R.best["avg_div"]
    for acc in ACCS:
        fr, tf = {}, {}
        x = conv_k(R, mel, model.conv_pre, acc)
        fr["conv_pre"] = tf["conv_pre"] = brief(compare(x, cap["conv_pre"][1]))
        for i in range(nu):
            n, up = f"ups.{i}", model.ups[i][0]
            x = conv_k(R, x, up, acc)
            fr[n], tf[n] = brief(compare(x, cap[n][1])), brief(compare(conv_k(R, cap[n][0], up, acc), cap[n][1]))
            rs = []
            for j in range(i * nk, (i + 1) * nk):
                n, blk = f"resblocks.{j}", model.resblocks[j]
                rs.append(amp_k(R, x, blk, acc))
                fr[n], tf[n] = brief(compare(rs[-1], cap[n][1])), brief(compare(amp_k(R, cap[n][0], blk, acc), cap[n][1]))
            x = R.k.avg3(*rs, divisor=float(nk), div=div)
            nat = [cap[f"resblocks.{j}"][1].contiguous() for j in range(i * nk, (i + 1) * nk)]
            fr[f"stage.{i}.avg3"] = brief(compare(x, avg_native[i]))
            tf[f"stage.{i}.avg3"] = brief(compare(R.k.avg3(*nat, divisor=float(nk), div=div), avg_native[i]))
        for n, f in (("activation_post", lambda t: act_k(R, t, model.activation_post, acc)),
                     ("conv_post", lambda t: conv_k(R, t, model.conv_post, acc))):
            x = f(x)
            fr[n], tf[n] = brief(compare(x, cap[n][1])), brief(compare(f(cap[n][0]), cap[n][1]))
        x = R.k.clamp(x, lo=-1.0, hi=1.0)
        tf["clamp"] = brief(compare(R.k.clamp(cap["conv_post"][1].contiguous(), lo=-1.0, hi=1.0), pcm))
        res[f"acc{acc}"] = {"config": {"acc": acc, "avg_div": div}, "free_running": fr, "teacher_forced": tf,
                            "pcm": compare(x, pcm), "kernel_pcm": R.dump(f"vocoder/kernel_pcm_acc{acc}", x)}
    return res


def case_weight_norm(R, kind, cin, cout, kk, stride):
    from torch.nn.utils.parametrizations import weight_norm
    from torch.nn.utils.parametrize import remove_parametrizations
    res = {"diagnostic": "not on the native load path (the checkpoint stores folded weights); candidate host fold "
                         "w = v * (g / ||v||) in torch fp32 against torch remove_parametrizations"}
    for where in dict.fromkeys(["cpu", str(R.dev)]):
        m = weight_norm(nn.Conv1d(cin, cout, kk) if kind == "conv1d" else nn.ConvTranspose1d(cin, cout, kk, stride))
        pw = m.parametrizations.weight
        g = torch.rand(pw.original0.shape, generator=R.gen(f"wn-g-{kind}")) + 0.5
        v = torch.randn(pw.original1.shape, generator=R.gen(f"wn-v-{kind}")) * 0.05
        m = m.to(where)
        m.parametrizations.weight.original0.copy_(g)
        m.parametrizations.weight.original1.copy_(v)
        remove_parametrizations(m, "weight")
        w, cand = m.weight.detach(), native.weight_norm_candidate(g.to(where), v.to(where))
        res[where] = {"ran_on": str(w.device), "inputs": R.H(g=g, v=v), "folded": R.H(w=w), "candidate": R.H(w=cand),
                      "candidate_vs_folded": compare(cand, w)}
    return res


def job_b(R):
    S = R.S
    if R.real:
        R.case("checkpoints/verify", case_verify_checkpoints, R)
    if R.case("vocoder/load", case_vocoder_load, R) is None:
        return
    st = S["stage"]
    R.case("snake_beta_log", case_snake, R)
    for which in ("stage", "activation_post"):
        for filt in ("module", "synthetic"):
            R.case(f"activation1d/{which}/{filt}_filters", case_aa, R, which, filt)
    for j in range(3):
        blk = stage_block(R, j)
        for i, c in enumerate(blk.convs1):
            R.case(f"conv1d/resblocks.{3 * st + j}.convs1.{i}", case_conv_module, R,
                   f"resblocks.{3 * st + j}.convs1.{i}", c, S["conv_T"])
        R.case(f"conv1d/resblocks.{3 * st + j}.convs2.0", case_conv_module, R, f"resblocks.{3 * st + j}.convs2.0",
               blk.convs2[0], S["conv_T"])
    R.case("conv1d/conv_pre", case_conv_module, R, "conv_pre", R.voc.conv_pre, S["pre_T"])
    R.case("conv1d/conv_post", case_conv_module, R, "conv_post", R.voc.conv_post, S["conv_post_T"])
    for i in range(R.voc.num_upsamples):
        R.case(f"conv_transpose1d/ups.{i}", case_conv_module, R, f"ups.{i}.0", R.voc.ups[i][0], S["convt_T"][i])
    R.case("vocoder_elementwise", case_vocoder_elementwise, R)
    R.case("ampblock1_chain", case_amp, R)
    R.case(f"vocoder_chain/mel{list(S['mel'])}", case_vocoder_chain, R)
    for kind, cin, cout, kk, stride in S["wn"]:
        R.case(f"weight_norm_fold/{kind}", case_weight_norm, R, kind, cin, cout, kk, stride)


# ======================================================================================================= driver

def summarize(doc):
    n = {"comparisons": 0, "zero_unequal": 0}

    def walk(o):
        if isinstance(o, dict):
            if "pass_" in o and "unequal" in o:
                n["comparisons"] += 1
                n["zero_unequal"] += bool(o["pass_"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk([c.get("result") for c in doc["cases"]])
    status = {}
    for c in doc["cases"]:
        status[c["status"]] = status.get(c["status"], 0) + 1
    return dict(n, case_status=status, note="sweep variants are expected to mismatch; read per-case results")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", choices=("A", "B"))
    p.add_argument("--fatbin")
    p.add_argument("--native-src", required=True, help="Kandinsky-6 checkout at eac06e39 (UPSTREAM)")
    p.add_argument("--models", help="MODELS dir with audio_vae/ and vocoder/ (required for --weights real)")
    p.add_argument("--weights", choices=("real", "synthetic"), default="real")
    p.add_argument("--out", required=True)
    p.add_argument("--site-packages", help="the sealed prefix's site-packages, added before torch")
    p.add_argument("--time-limit-min", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--norm-eps", type=float, default=None, help="override the eps given to the candidate normalize")
    p.add_argument("--backend", choices=("auto", "cuda-python", "ctypes"), default="auto")
    p.add_argument("--dry-run-cpu", action="store_true", help="tiny shapes on CPU, native paths only, no launches")
    a = p.parse_args(argv)
    R = Run(a)
    code = 0
    try:
        set_numerics()
        if a.dry_run_cpu and a.weights == "real":
            raise Fatal("--dry-run-cpu uses tiny shapes: pass --weights synthetic")
        if a.weights == "real" and not a.models:
            raise Fatal("--models is required for --weights real")
        try:
            R.ns = native.load(a.native_src)
        except native.NativeRefused as e:
            raise Fatal(f"native source refused: {e}")
        R.doc["native"] = native.provenance(R.ns)
        if not R.dry:
            if not a.fatbin or not os.path.isfile(a.fatbin):
                raise Fatal("--fatbin is required (and must exist) unless --dry-run-cpu")
            if not torch.cuda.is_available():
                raise Fatal("torch reports no CUDA device")
            R.k = K6A(a.fatbin, backend=a.backend)
        R.doc["env"] = env_record(R)
        R.flush()
        (job_a if a.job == "A" else job_b)(R)
        R.doc["finished"] = True
        R.doc["exit_reason"] = "completed" if R.elapsed() <= R.limit else "time limit reached (later cases skipped)"
    except Fatal as e:
        R.doc["exit_reason"] = f"infrastructure failure: {e}"
        code = 2
    except Exception as e:
        R.doc["exit_reason"] = f"infrastructure failure: {type(e).__name__}: {e}"
        R.doc["traceback"] = traceback.format_exc()[-4000:]
        code = 3
    R.doc["best_variants"] = {k: (vname(v) if k == "norm" else v) for k, v in R.best.items()}
    R.doc["summary"] = summarize(R.doc)
    R.flush()
    print(json.dumps({"job": a.job, "exit_reason": R.doc["exit_reason"], "summary": R.doc["summary"],
                      "elapsed_s": R.doc["elapsed_s"]}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
