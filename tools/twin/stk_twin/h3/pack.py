"""The MiniMax H3 DiT's pack (stk-pack/1, what the Zig engine and the twin's NVFP4 mode load): per block linear
`L{i}.{qkv,out,fc1,fc2}` NVFP4 (`.codes` uint8 [N, K/2] e2m1 low nibble first, `.scales` e4m3 [N, K/16], `.global` f32
[1], `.act` f32 [1], the static input scale), the block's norms fp32, q/k norms bf16, the curve adaln fp32
(`L{i}.ada_w` [96768, 8], `.ada_b`), and the side tensors under their checkpoint names: patch projections and heads
fp32, the final norm bf16, the curve table and RoPE frequencies fp32, condition_proj and the token refiner bf16.
`write_tokenizer` adds the text encoder's `tokenizer.json` beside them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..files import readable
from .dit import LAYERS, LINEARS, SOURCE

SIDE_F32 = ["adaln_t_table", "rope.inv_freq", "video_patch_proj.weight", "video_patch_proj.bias",
            "audio_patch_proj.weight", "audio_patch_proj.bias", "final_layer.video_out.weight",
            "final_layer.video_out.bias", "final_layer.audio_out.weight", "final_layer.audio_out.bias",
            "final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias"]


def write(w: dict[str, torch.Tensor], acts: dict[str, float], out: str | Path, source: dict) -> dict:
    from tfvideo.linear import nvfp4_weight

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t: dict[str, torch.Tensor] = {k: w[k].float().contiguous() for k in SIDE_F32}
    t["final_layer.norm.weight"] = w["final_layer.norm.weight"].to(torch.bfloat16).contiguous()
    for k, v in w.items():
        if k.startswith(("condition_proj.", "token_refiner.")):
            t[k] = v.to(torch.bfloat16).contiguous()
    kinds = {}
    for i in range(LAYERS):
        p = f"blocks.{i}."
        t[f"L{i}.norm1"] = w[p + "norm1.weight"].float().contiguous()
        t[f"L{i}.norm2"] = w[p + "norm2.weight"].float().contiguous()
        t[f"L{i}.q_norm"] = w[p + "attn.q_norm.weight"].to(torch.bfloat16).contiguous()
        t[f"L{i}.k_norm"] = w[p + "attn.k_norm.weight"].to(torch.bfloat16).contiguous()
        t[f"L{i}.ada_w"] = w[p + "adaln_proj.linear.weight"].float().contiguous()
        t[f"L{i}.ada_b"] = w[p + "adaln_proj.linear.bias"].float().contiguous()
        for name in LINEARS:
            key = f"L{i}.{name}"
            codes, scales, g = nvfp4_weight(w[p + SOURCE[name] + ".weight"].cuda().to(torch.bfloat16))
            t[key + ".codes"] = codes.cpu().contiguous()
            t[key + ".scales"] = scales.cpu().contiguous()
            t[key + ".global"] = torch.tensor([g], dtype=torch.float32)
            t[key + ".act"] = torch.tensor([acts[f"{i}.{name}"]], dtype=torch.float32)
            kinds[key] = "nvfp4"
    save_file(t, str(out / "weights.safetensors"))
    digest = {n: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n, v in t.items()}
    manifest = {"format": "stk-pack/1", "model": "minimax-h3-fl2va", "precision": "nvfp4", "source": source,
                "kinds": kinds, "acts": acts, "tensors": {
                    n: {"dtype": str(v.dtype).removeprefix("torch."), "shape": list(v.shape), "sha256": digest[n]}
                    for n, v in t.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    readable(out)
    return manifest


TOKENIZER_SAMPLES = [
    "", "a", "a cat", "A street cook in a white apron, flips noodles in a flaming wok. Sound: sizzling wok, rain.",
    "  leading and  double  spaces\n\nnewlines\tand tabs  ", "<|im_start|>user\nhi<|im_end|>", "<d>x</d> <|cutoff|>",
    "<|lyrics_start|>la<|lyrics_end|><|caption_start|>c<|caption_end|>", "<think>t</think><|endoftext|>",
    "naïve café 日本語 😀 ǅ", "it's we're I'll 123 4567 x_y-z", "\\(escaped\\) (plain) embedding:name  embedding: x",
]


def write_tokenizer(out: str | Path) -> dict:
    """The H3 text encoder's tokenizer next to the weights, so the Zig engine needs no other file: `tokenizer.json` in
    the Hugging Face tokenizers format (what TensorFold's tokenizer reads), made from the same vendored Qwen2.5 tokenizer files as
    te32.tokenize (`te32.tokenizer_dir()`) with the seven extra special tokens added, then checked: the saved file's ids
    must equal te32's tokenizer's on `TOKENIZER_SAMPLES` (the engine adds only the `\\(` / `embedding:` pre-pass). The
    manifest gets a `tokenizer` entry (file, sha256, extra tokens, pad id)."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from . import te32

    out = Path(out)
    fast = AutoTokenizer.from_pretrained(str(te32.tokenizer_dir()), use_fast=True)
    fast.add_special_tokens({"additional_special_tokens": list(te32.EXTRA_TOKENS)})
    for s, i in te32.EXTRA_TOKENS.items():
        assert fast.convert_tokens_to_ids(s) == i, f"special token {s} is not id {i}"
    path = out / "tokenizer.json"
    fast.backend_tokenizer.save(str(path))
    ref, slow = Tokenizer.from_file(str(path)), te32._tokenizer()
    bad = [s for s in TOKENIZER_SAMPLES
           if ref.encode(s, add_special_tokens=False).ids != [int(t) for t in slow(s)["input_ids"]]]
    if bad:
        raise RuntimeError(f"tokenizer.json differs from the twin's tokenizer on {bad!r}")
    info = {"file": "tokenizer.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "extra_tokens": te32.EXTRA_TOKENS, "pad_id": te32.PAD_ID}
    mf = out / "manifest.json"
    man = json.loads(mf.read_text())
    man["tokenizer"] = info
    mf.write_text(json.dumps(man, indent=1))
    readable(out)
    return info


def read(dir: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """The pack's tensors (CPU) and its static input scales."""
    from safetensors.torch import load_file

    dir = Path(dir)
    return load_file(str(dir / "weights.safetensors")), json.loads((dir / "manifest.json").read_text())["acts"]
