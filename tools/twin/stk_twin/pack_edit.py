"""Add edit weights to existing text / decoder packs, leaving DiT packs untouched.
Marker appears only after both packs were written; it identifies their exact manifests for later capability gating.
Full quant_conv (128 channels) matches the recorded encoder; only its first 64 output channels condition the DiT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .edit import TEMPLATE
from .pack import _save
from .files import readable

# The vision tower's architecture as the checkpoint's config.json gives it, in this order: the fields the engine reads
# (src/engines/qwen_image/vision.zig) and the shapes they set. Not the whole `to_dict()`: that also carries the
# installed transformers' version and its defaults, so te/manifest.json (and edit.json, which hashes it) differed
# between boxes with byte-identical weights (#84: transformers 5.20.2 against 5.21.0).
VISION_CONFIG = ("depth", "hidden_size", "hidden_act", "intermediate_size", "num_heads", "in_channels", "patch_size",
                 "spatial_merge_size", "temporal_patch_size", "out_hidden_size", "num_position_embeddings",
                 "deepstack_visual_indexes")


def base(path: Path, model: str):
    man = json.loads((path / "manifest.json").read_text())
    if man.get("format") != "stk-pack/1" or man.get("model") != model:
        raise ValueError(f"{path} is not a {model} pack")
    tensors = load_file(str(path / "weights.safetensors"), device="cpu")
    if set(tensors) != set(man["tensors"]):
        raise ValueError(f"{path} tensor names differ from its manifest")
    for name, tensor in tensors.items():
        spec = man["tensors"][name]
        digest = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if list(tensor.shape) != spec["shape"] or str(tensor.dtype).removeprefix("torch.") != spec["dtype"] or digest != spec["sha256"]:
            raise ValueError(f"{path}: corrupt tensor {name}")
    return tensors, {k: v for k, v in man.items() if k not in ("format", "tensors")}


def write_vision(pipe, root: Path):
    path = root / "te"
    tensors, man = base(path, "qwen-image-2.1-te")
    tensors = {k: v for k, v in tensors.items() if not k.startswith("vis.")}
    visual = pipe.text_encoder.model.visual
    for name, tensor in visual.state_dict().items():
        key = f"vis.{name}"
        value = tensor.detach().to("cpu", torch.bfloat16)
        if name == "patch_embed.proj.weight":
            # Conv3d [D,3,2,16,16] on one patch is the twin's GEMM [D,1536]. No rank-5 pack tensor.
            value = value.reshape(value.shape[0], -1)
        tensors[key] = value.contiguous()
    cfg = {k: visual.config.to_dict()[k] for k in VISION_CONFIG}
    if cfg["hidden_size"] // cfg["num_heads"] != 72 or cfg["spatial_merge_size"] != 2:
        raise ValueError("vision pack requires 72-wide heads and merge size 2")
    man["vision"] = {"config": cfg, "template_edit": TEMPLATE, "heads_padded": 128,
                     "position_rounding": "bf16-products-sequential-add", "source": man["source"]}
    # Freeze inverse frequencies consumed by the twin, rather than re-evaluating powers on a new backend.
    tensors["vis.inv_freq"] = visual.rotary_pos_emb.inv_freq.detach().cpu().float().contiguous()
    _save(tensors, path, man)
    return len(tensors)


def write_encoder(pipe, root: Path):
    from torch import nn
    from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as m

    path = root / "vae"
    tensors, man = base(path, "qwen-image-2.1-vae")
    vae = pipe.vae
    if vae.config.patch_size is not None:
        raise ValueError("edit pack supports the unpatchified single-frame VAE only")
    prefixes = ("vae.encoder.", "vae.quant_conv.", "vae.enc_latents_")
    tensors = {k: v for k, v in tensors.items() if not k.startswith(prefixes)}
    convs = {k: v for k, v in man["convs"].items() if not k.startswith(prefixes)}
    norms = {k: v for k, v in man["norms"].items() if not k.startswith(prefixes)}
    enc_convs, enc_norms, downs = {}, {}, {}
    modules = [("quant_conv", vae.quant_conv)] + [(f"encoder.{name}" if name else "encoder", mod)
                                                 for name, mod in vae.encoder.named_modules()]
    for name, mod in modules:
        if ".time_conv" in name:
            continue
        key = f"vae.{name}"
        if isinstance(mod, nn.Conv2d):
            if mod.weight.dim() != 4 or mod.dilation != (1, 1) or mod.groups != 1:
                raise ValueError(f"unsupported encoder convolution {name}")
            if hasattr(mod, "_padding"):
                pl, pr, pt, pb = mod._padding
            else:
                pt = pb = mod.padding[0]
                pl = pr = mod.padding[1]
            tensors[key + ".weight"] = mod.weight.detach().to("cpu", torch.bfloat16).contiguous()
            if mod.bias is not None:
                tensors[key + ".bias"] = mod.bias.detach().to("cpu", torch.bfloat16).contiguous()
            enc_convs[key] = {"stride": mod.stride[0], "pad": [pt, pb, pl, pr]}
        elif isinstance(mod, m.QwenImage21RMS_norm):
            tensors[key + ".gamma"] = mod.gamma.detach().to("cpu", torch.bfloat16).reshape(-1).contiguous()
            enc_norms[key] = float(mod.scale)
        elif isinstance(mod, m.QwenImage21AvgDown3D):
            downs[key] = {"cin": mod.in_channels, "cout": mod.out_channels, "factor_t": mod.factor_t,
                          "fs": mod.factor_s, "group_size": mod.group_size}
    for key, values in (("vae.enc_latents_mean", vae.config.latents_mean), ("vae.enc_latents_std", vae.config.latents_std)):
        tensors[key] = torch.tensor(values, dtype=torch.bfloat16)
    # Decoder loading iterates the top-level maps. Keep encoder metadata separate so text-only loads neither
    # allocate encoder weights nor reject its 4-channel (K=36, padded by the twin) input convolution.
    man.pop("downs", None)
    man.update(convs=convs, norms=norms,
               encoder={"latent_channels": int(vae.config.z_dim), "quant_channels": int(vae.quant_conv.weight.shape[0]),
                        "config": dict(vae.config), "convs": enc_convs, "norms": enc_norms, "downs": downs})
    if man["encoder"]["latent_channels"] != 64 or man["encoder"]["quant_channels"] != 128:
        raise ValueError("encoder pack must emit 128 moments for 64-channel latents")
    _save(tensors, path, man)
    return len(tensors)


def write(pipe, out: str | Path):
    """OUT contains existing te/ and vae/ packs. Failures remove capability marker, keeping text generation usable."""
    root = Path(out)
    marker = root / "edit.json"
    marker.unlink(missing_ok=True)
    counts = {"te": write_vision(pipe, root), "vae": write_encoder(pipe, root)}
    return {"pack": str(root), "precision": "edit", "tensors": counts, **mark(root)}


def mark(root: Path) -> dict:
    """edit.json for the te/ and vae/ manifests now in ROOT."""
    hashes = {name: hashlib.sha256((root / name / "manifest.json").read_bytes()).hexdigest() for name in ("te", "vae")}
    result = {"format": "stk-edit/1", "refs_max": 5, "manifest_sha256": hashes}
    temp = root / "edit.json.tmp"
    temp.write_text(json.dumps(result, indent=1))
    temp.replace(root / "edit.json")
    readable(root)
    return result


def normalise(out: str | Path) -> bool:
    """An edit pack written before VISION_CONFIG (#84): te/manifest.json rewritten with only those fields, as `write`
    now writes it (the rest of the file is kept byte for byte: json round-trips it), then edit.json for it. Metadata
    only; the weights are not touched. True when it rewrote anything."""
    root = Path(out)
    te, marker = root / "te" / "manifest.json", root / "edit.json"
    if not marker.exists():
        return False
    man = json.loads(te.read_text())
    cfg = man["vision"]["config"]
    if list(cfg) == list(VISION_CONFIG):
        return False
    man["vision"]["config"] = {k: cfg[k] for k in VISION_CONFIG}
    marker.unlink()   # no marker naming a manifest that is being replaced: edits stay off until both are written
    temp = te.with_name("manifest.json.tmp")
    temp.write_text(json.dumps(man, indent=1))
    temp.replace(te)
    mark(root)
    return True
