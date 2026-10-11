"""Native references for the Kandinsky-6 audio qualification jobs.

(a) Permissive operators are written directly as the native graph code writes them (PyTorch functional calls,
    postprocess_audio / VAE.unnormalize expressions).
(b) Restricted-native operators (normalize, MPConv1D, mp_silu, mp_sum and the vae_modules / vae modules that call them)
    and the BigVGAN v2 modules are imported from the pinned Kandinsky-6 checkout given at runtime (--native-src) and
    only called, as a black box. Nothing from that checkout is copied, read as text or vendored here; the checkout is
    refused unless edm2_utils.py has the pinned sha256 (and, when the checkout reports a commit, the pinned commit).

This is the reference-only route and requires Captain approval before its results count as evidence.
"""

import hashlib
import importlib
import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROUTE = "reference-only route, requires Captain approval"
PINNED_COMMIT = "eac06e39b904940baff2c408e925cdb0c4529269"
EXT_REL = "kandinsky/core/components/mmaudio/ext"
EDM2_REL = EXT_REL + "/autoencoder/edm2_utils.py"
EDM2_SHA256 = "9a61ca2344f3b6ddd1a745e89669725eb182575453a395c224e7df8152e956a7"
PKG = "k6native"  # private package name: the checkout's own __init__ chain (pipeline, safetensors) is never run


class NativeRefused(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commit(root):
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=20, env={"PATH": os.environ.get("PATH", ""), "GIT_TERMINAL_PROMPT": "0"})
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod


def load(root):
    """Verify the pinned checkout and import its modules. Raises NativeRefused on any provenance mismatch."""
    root = Path(root).resolve()
    edm2 = root / EDM2_REL
    if not edm2.is_file():
        raise NativeRefused(f"{EDM2_REL} not found under {root}")
    digest = sha256_file(edm2)
    if digest != EDM2_SHA256:
        raise NativeRefused(f"edm2_utils.py sha256 {digest} != pinned {EDM2_SHA256}")
    commit = source_commit(root)
    if commit is not None and commit != PINNED_COMMIT:
        raise NativeRefused(f"checkout commit {commit} != pinned {PINNED_COMMIT}")
    ext = root / EXT_REL
    for name, rel in ((PKG, ""), (PKG + ".autoencoder", "autoencoder"), (PKG + ".bigvgan_v2", "bigvgan_v2"),
                      (PKG + ".bigvgan_v2.alias_free_activation", "bigvgan_v2/alias_free_activation"),
                      (PKG + ".bigvgan_v2.alias_free_activation.torch", "bigvgan_v2/alias_free_activation/torch")):
        _package(name, ext / rel)
    imp = lambda m: importlib.import_module(f"{PKG}.{m}")  # noqa: E731
    e2, vm, vae = imp("autoencoder.edm2_utils"), imp("autoencoder.vae_modules"), imp("autoencoder.vae")
    bv, acts = imp("bigvgan_v2.bigvgan"), imp("bigvgan_v2.activations")
    act, res = imp("bigvgan_v2.alias_free_activation.torch.act"), imp("bigvgan_v2.alias_free_activation.torch.resample")
    ae = imp("autoencoder.autoencoder")  # MIT loader: AutoEncoderModule, load_bigvgan_v2 (needs safetensors)
    eps = inspect.signature(e2.normalize).parameters.get("eps")
    return types.SimpleNamespace(
        root=str(root), commit=commit, edm2_sha256=digest, route=ROUTE,
        normalize_eps=None if eps is None or eps.default is inspect.Parameter.empty else float(eps.default),
        # restricted-native, black box
        normalize=e2.normalize, MPConv1D=e2.MPConv1D, mp_silu=e2.mp_silu, mp_sum=e2.mp_sum,
        ResnetBlock1D=vm.ResnetBlock1D, AttnBlock1D=vm.AttnBlock1D, Upsample1D=vm.Upsample1D,
        Decoder1D=vae.Decoder1D, VAE=vae.VAE, DATA_MEAN=list(vae.DATA_MEAN_128D), DATA_STD=list(vae.DATA_STD_128D),
        AutoEncoderModule=ae.AutoEncoderModule, load_bigvgan_v2=ae.load_bigvgan_v2,
        # BigVGAN v2 (MIT / Apache-2.0)
        BigVGAN=bv.BigVGAN, AMPBlock1=bv.AMPBlock1, load_hparams=bv.load_hparams_from_json,
        vocoder_config=str(ext / "bigvgan_v2" / "config.json"), SnakeBeta=acts.SnakeBeta,
        Activation1d=act.Activation1d, UpSample1d=res.UpSample1d, DownSample1d=res.DownSample1d,
    )


def provenance(ns):
    return {"route": ROUTE, "root": ns.root, "commit": ns.commit, "commit_pinned": PINNED_COMMIT,
            "edm2_utils_sha256": ns.edm2_sha256, "normalize_eps_signature_default": ns.normalize_eps}


# -------------------------------------------------------------------------- (a) permissive operators, as written

def latent_unscale(rows, scaling_factor, mean_value):
    """postprocess_audio: audio / scaling_factor + mean_value, then stack(...).transpose(1, 2) -> [B, C, T] view."""
    return (rows / scaling_factor + mean_value).transpose(1, 2)


def mel_affine(x, data_std, data_mean):
    """VAE.unnormalize with buffers viewed [1, C, 1]."""
    return x * data_std.view(1, -1, 1) + data_mean.view(1, -1, 1)


def clamp(x, lo, hi):
    return x.clamp(lo, hi)


def upsample_nearest_exact2(x):
    return F.interpolate(x, scale_factor=2.0, mode="nearest-exact")


def conv1d(x, w, b, stride, pad, dil):
    return F.conv1d(x, w, b, stride=stride, padding=pad, dilation=dil)


def conv_transpose1d(x, w, b, stride, pad):
    return F.conv_transpose1d(x, w, b, stride=stride, padding=pad)


def add(xt, x):
    return xt + x


def avg3(r0, r1, r2, num_kernels=3):
    """BigVGAN stage merge: xs = r0; xs += r1; xs += r2; xs / num_kernels (on a copy so r0 is untouched)."""
    xs = r0.clone()
    xs += r1
    xs += r2
    return xs / num_kernels


def qkv_split(y, heads):
    """AttnBlock1D layout without the normalization: reshape [B, H, D, 3, T], unbind(3), permute(0, 1, 3, 2)."""
    y = y.reshape(y.shape[0], heads, -1, 3, y.shape[-1])
    return tuple(t.permute(0, 1, 3, 2) for t in y.unbind(3))


def attn_merge(h):
    return h.permute(0, 1, 3, 2).reshape(h.shape[0], -1, h.shape[2])


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def weight_norm_candidate(g, v):
    """Diagnostic candidate host fold (torch fp32): w = v * (g / ||v||) with the norm over every dim except 0."""
    return v * (g / torch.linalg.vector_norm(v, dim=tuple(range(1, v.dim())), keepdim=True))
