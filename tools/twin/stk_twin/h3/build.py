"""python -m stk_twin.h3.build --models DIR --out PACK [--lora] [--prompts N]

The H3 DiT's shipping pack in one run: the checkpoint (bf16), optionally the Turbo 8-step LoRA merged in fp32 (W +
(alpha / rank) * B A on our deterministic fp32 GEMM, rounded to bf16; ComfyUI's `calculate_weight` arithmetic), then
the static NVFP4 input scales: every block linear's input absmax over bf16 trajectories of the calibration prompts
(768x448, 56 frames, the schedule's steps, Euler on the twin's own velocities), times a margin of 2, over 6 x 448.
Then `pack.write`. Text through LocalRouter's 32B encoder (`te32`). One JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from . import ops as H
from . import sampler
from .dit import H3Dit, LAYERS, LINEARS, load_checkpoint
from .shapes import latent_shapes

LORA = "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
DIT = "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
TE = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
MARGIN, FP4_MAX, E4M3_MAX = 2.0, 6.0, 448.0
CALIB = Path(__file__).resolve().parents[2] / "prompts" / "video_calib.json"


def merge_lora(w: dict[str, torch.Tensor], lora_path: Path, dev) -> int:
    """In place: every weight the LoRA names, W = bf16(f32(W) + f32(alpha / rank) * (B A)), fp32 throughout."""
    from safetensors.torch import load_file

    lo = load_file(str(lora_path))
    n = 0
    for k in [k for k in lo if k.endswith(".lora_A.weight")]:
        base = k[: -len(".lora_A.weight")]
        name = base.removeprefix("diffusion_model.") + ".weight"
        A = lo[k].to(dev, torch.float32)                         # [r, K]
        B = lo[base + ".lora_B.weight"].to(dev, torch.float32)   # [N, r]
        alpha = float(lo[base + ".alpha"].item()) / A.shape[0]
        diff = H.mod().k_gemm_f32(B.contiguous(), A.t().contiguous(), None)       # [N, K] = B A, k-ordered fp32
        wf = w[name].to(dev, torch.float32)
        w[name] = (wf + diff * torch.tensor(alpha, dtype=torch.float32, device=dev)).to(torch.bfloat16).cpu()
        n += 1
    return n


def encode(models: Path, prompts: list[str], record: bool = False) -> list[torch.Tensor]:
    """LocalRouter's text encoder (`te32`, the twin's own; ComfyUI's is only the gate's reference): [1, L, 5120] as
    bf16, as ComfyUI's extra_conds hands the states to the DiT. `record`: the encoder's ops go to REC."""
    from ..rec import REC
    from .te32 import TextEncoder32

    te = TextEncoder32(models / TE)
    out = []
    for p in prompts:
        on = REC.on
        REC.on = on and record
        h = te.encode(p)
        REC.on = on
        out.append(h.to(torch.bfloat16).unsqueeze(0))
    del te
    torch.cuda.empty_cache()
    return out


def main() -> None:
    import tensorfold.cuda.nvfp4.checkpoint  # noqa: F401  the pack writer needs it: fail now, not after the merge and calibration
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("VMODELS", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=4)
    a = ap.parse_args()
    from . import pack

    models, dev = Path(a.models), torch.device("cuda")
    t0 = time.time()
    prompts = json.loads(CALIB.read_text())[: a.prompts]
    conds = encode(models, prompts)
    w = load_checkpoint(models / DIT)
    merged = merge_lora(w, models / LORA, dev) if a.lora else 0
    sig = sampler.sigmas(a.steps)                # LocalRouter's own `simple` schedule (gate.py asserts it equals ComfyUI's)
    twin = H3Dit(w, kind="bf16", attn="bf16", device=dev)
    amax: dict[str, float] = {}
    orig = twin.lin

    def lin(name, l, x):                       # name "L{i}.{qkv,out,fc1,fc2}"
        key = name[1:]
        amax[key] = max(amax.get(key, 0.0), float(x.abs().amax()))
        return orig(name, l, x)

    twin.lin = lin
    vs, as_ = latent_shapes(768, 448, 56)
    with torch.inference_mode():
        for n, c in enumerate(conds):
            twin.prepare(c)
            g = torch.Generator(device="cpu").manual_seed(1000 + n)
            xv, xa = torch.randn(vs, generator=g).to(dev), torch.randn(as_, generator=g).to(dev)
            for i in range(len(sig) - 1):
                vv, va = twin(xv.to(torch.bfloat16), xa.to(torch.bfloat16), sig[i])
                xv = xv + vv.float() * (sig[i + 1] - sig[i])
                xa = xa + va.float() * (sig[i + 1] - sig[i])
    acts = {f"{i}.{nm}": max(amax[f"{i}.{nm}"], 1e-6) / (FP4_MAX * E4M3_MAX) * MARGIN
            for i in range(LAYERS) for nm in LINEARS}
    del twin
    torch.cuda.empty_cache()
    man = pack.write(w, acts, a.out, {"repo": "Comfy-Org/MiniMax-H3", "dit": DIT, "lora": LORA if a.lora else None,
                                      "calibration": {"prompts": len(conds), "steps": a.steps, "margin": MARGIN}})
    tok = pack.write_tokenizer(a.out)                          # tokenizer.json beside the weights (the Zig pipeline's only text file)
    print(json.dumps({"h3_build": {"pack": a.out, "lora_weights_merged": merged, "tensors": len(man["tensors"]),
                                   "tokenizer": tok["sha256"],
                                   "sigmas": sig, "act_range": [min(acts.values()), max(acts.values())],
                                   "seconds": round(time.time() - t0, 1)}}))


if __name__ == "__main__":
    main()
