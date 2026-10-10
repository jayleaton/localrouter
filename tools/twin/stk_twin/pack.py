"""LocalRouter's converted weights ("pack"): what the Zig engine loads instead of the bf16 checkpoint.

<dir>/weights.safetensors and <dir>/manifest.json. Per block linear `L{i}.{qkv,out,gate_up,down}`:
  nvfp4: `.codes` uint8 [N, K/2] (e2m1, low nibble first), `.scales` e4m3 [N, K/16], `.global` f32 [1], `.act` f32 [1]
  fp8:   `.w8` e4m3 [N, K], `.scale` f32 [1]  (inputs are scaled per call from their own absmax)
Side weights stay bf16 under their diffusers names (img_in, txt_in.*, time_text_embed.*, modulation.1, norm_out.linear,
proj_out) and the q/k norms fp32. The manifest has each tensor's sha256, the source revision, the calibration and the
checkpoint's schedule.

The text encoder's pack (`write_te`, precision "bf16"): every weight bf16 under the twin's op names (te.embed,
te.{i}.{input_layernorm,q_proj,k_proj,v_proj,q_norm,k_norm,o_proj,post_attention_layernorm,gate_proj,up_proj,
down_proj}), the pipeline's tokenizer.json beside it and the template / system-token count in the manifest.

The VAE decoder's pack (`write_vae`, precision "bf16"): the decode path's parameters under the twin's op names
(`vae.decoder.up_blocks.2.resnets.0.conv1.weight` [Cout, Cin, kh, kw], `.bias`, norms' `.gamma` flattened to [C]),
`vae.latents_mean` / `vae.latents_std` as the pipeline's bf16 tensors, and every conv's stride and pads in the manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tfimage import linear as L

from .dit import Config, policy
from .files import readable
from .qwen_image import scheduler_settings

SIDE = ["img_in.weight", "txt_in.text_norm.weight", "txt_in.in_layer.weight", "txt_in.out_layer.weight",
        "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_2.weight",
        "modulation.1.weight", "norm_out.linear.weight", "proj_out.weight"]


def block_weights(w: dict[str, torch.Tensor], i: int) -> dict[str, torch.Tensor]:
    p = f"transformer_blocks.{i}."
    return {
        "qkv": torch.cat([w[p + "attn.to_q.weight"], w[p + "attn.to_k.weight"], w[p + "attn.to_v.weight"]]),
        "out": w[p + "attn.to_out.0.weight"],
        "gate_up": torch.cat([w[p + "img_mlp.gate_layer.weight"], w[p + "img_mlp.proj.weight"]]),
        "down": w[p + "img_mlp.out.weight"],
    }


def _save(t: dict[str, torch.Tensor], out: Path, extra: dict) -> dict:
    save_file(t, str(out / "weights.safetensors"))
    digest = {n: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n, v in t.items()}
    manifest = {"format": "stk-pack/1", **extra, "tensors": {
        n: {"dtype": str(v.dtype).removeprefix("torch."), "shape": list(v.shape), "sha256": digest[n]} for n, v in t.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    readable(out)
    return manifest


TE_NAMES = {"input_layernorm": "input_layernorm", "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
            "v_proj": "self_attn.v_proj", "q_norm": "self_attn.q_norm", "k_norm": "self_attn.k_norm",
            "o_proj": "self_attn.o_proj", "post_attention_layernorm": "post_attention_layernorm",
            "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj", "down_proj": "mlp.down_proj"}


def write_te(pipe, out: str | Path, source: dict) -> dict:
    """The text encoder's pack from the loaded pipeline (its Qwen3-VL text model and tokenizer)."""
    import shutil

    from .te import TextEncoder

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    te = TextEncoder(pipe.text_encoder, "cpu")
    t = {"te.embed": te.embed.detach().to("cpu", torch.bfloat16).contiguous()}
    for i, layer in enumerate(te.layers):
        for name, path in TE_NAMES.items():
            mod = layer.get_submodule(path)
            t[f"te.{i}.{name}"] = mod.weight.detach().to("cpu", torch.bfloat16).contiguous()
    tok = pipe.processor.tokenizer
    tok_dir = out / "tokenizer"
    tok.save_pretrained(str(tok_dir))
    shutil.copy(tok_dir / "tokenizer.json", out / "tokenizer.json")
    cfg = getattr(pipe.text_encoder.model, "language_model", pipe.text_encoder.model).config
    sched = scheduler_settings(source["dir"])
    return _save(t, out, {"model": "qwen-image-2.1-te", "precision": "bf16", "source": source, "kinds": {},
                          "template": pipe.prompt_template_t2i, "drop": pipe._drop_idx, "scheduler": sched,
                          "config": {"layers": len(te.layers), "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads,
                                     "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
                                     "mlp": cfg.intermediate_size, "eps": cfg.rms_norm_eps,
                                     "theta": (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta",
                                                                                                getattr(cfg, "rope_theta", None))}})


def write_vae(vae, out: str | Path, source: dict) -> dict:
    """The VAE decoder's pack from the loaded diffusers `AutoencoderKLQwenImage21` (bf16)."""
    from torch import nn
    from diffusers.models.autoencoders import autoencoder_kl_qwenimage21 as m

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t, convs, norms, dups = {}, {}, {}, {}
    mods = [("post_quant_conv", vae.post_quant_conv)] + [(f"decoder.{n}" if n else "decoder", md) for n, md in vae.decoder.named_modules()]
    for name, mod in mods:
        key = f"vae.{name}"
        if isinstance(mod, nn.Conv2d):
            if hasattr(mod, "_padding"):
                pl, pr, pt, pb = mod._padding
            else:
                (pt, pb), (pl, pr) = (mod.padding[0],) * 2, (mod.padding[1],) * 2
            assert mod.weight.dim() == 4 and mod.dilation == (1, 1) and mod.groups == 1
            t[key + ".weight"] = mod.weight.detach().to("cpu", torch.bfloat16).contiguous()
            t[key + ".bias"] = mod.bias.detach().to("cpu", torch.bfloat16).contiguous()
            convs[key] = {"stride": mod.stride[0], "pad": [pt, pb, pl, pr]}
        elif isinstance(mod, m.QwenImage21RMS_norm):
            t[key + ".gamma"] = mod.gamma.detach().to("cpu", torch.bfloat16).reshape(-1).contiguous()
            norms[key] = float(mod.scale)
        elif isinstance(mod, m.QwenImage21DupUp3D):
            dups[key] = {"cin": mod.in_channels, "cout": mod.out_channels, "factor_t": mod.factor_t, "fs": mod.factor_s,
                         "factor": mod.factor, "repeats": mod.repeats, "fti": mod.factor_t - 1}
    t = {k: v for k, v in t.items() if ".time_conv." not in k}  # unused for one frame
    convs = {k: v for k, v in convs.items() if ".time_conv" not in k}
    t["vae.latents_mean"] = torch.tensor(vae.config.latents_mean).to(torch.bfloat16)
    t["vae.latents_std"] = torch.tensor(vae.config.latents_std).to(torch.bfloat16)
    return _save(t, out, {"model": "qwen-image-2.1-vae", "precision": "bf16", "source": source, "kinds": {},
                          "convs": convs, "norms": norms, "dups": dups, "out_channels": int(vae.decoder.conv_out.weight.shape[0])})


def write(w: dict[str, torch.Tensor], precision: str, acts: dict, out: str | Path, source: dict, cfg=Config(),
          model: str = "qwen-image-2.1"):
    """The DiT's pack. Its manifest carries the checkpoint's schedule (`scheduler`, as the text encoder's pack does),
    which the engine takes over the text encoder pack's: a DiT with its own schedule (Turbo) shares the te and vae packs."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pick = policy(precision, cfg.layers)
    t: dict[str, torch.Tensor] = {n: w[n].to(torch.bfloat16).contiguous() for n in SIDE}
    t["txt_in.text_norm.weight"] = w["txt_in.text_norm.weight"].float().contiguous()
    kinds = {}
    for i in range(cfg.layers):
        p = f"transformer_blocks.{i}."
        t[f"L{i}.norm_q"] = w[p + "attn.norm_q.weight"].float().contiguous()
        t[f"L{i}.norm_k"] = w[p + "attn.norm_k.weight"].float().contiguous()
        for name, wt in block_weights(w, i).items():
            kind, key = pick(i, name), f"L{i}.{name}"
            kinds[key] = kind
            wt = wt.cuda().to(torch.bfloat16)
            if kind == "nvfp4":
                codes, scales, g = L.nvfp4_weight(wt)
                t[key + ".codes"] = codes.cpu().contiguous()
                t[key + ".scales"] = scales.cpu().contiguous()
                t[key + ".global"] = torch.tensor([g], dtype=torch.float32)
                t[key + ".act"] = torch.tensor([acts[f"{i}.{name}"]], dtype=torch.float32)
            elif kind == "fp8":
                lin = L.Fp8Linear(wt)
                t[key + ".w8"] = lin.w8.cpu().contiguous()
                t[key + ".scale"] = lin.ws.reshape(1).cpu().contiguous()
                if f"{i}.{name}" in acts:  # static input scale: TensorFold's FP8 prompt GEMM
                    t[key + ".act"] = torch.tensor([acts[f"{i}.{name}"]], dtype=torch.float32)
            else:
                t[key + ".weight"] = wt.cpu().contiguous()
    return _save(t, out, {"model": model, "precision": precision, "source": source, "kinds": kinds,
                           "acts": acts, "scheduler": scheduler_settings(source["dir"])})
