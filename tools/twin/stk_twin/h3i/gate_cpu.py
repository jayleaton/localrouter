"""CPU structure gate: python -m stk_twin.h3i.gate_cpu --comfy COMFYUI_SRC [--mode f32|lowp|both] [--controls]
(no GPU, no checkpoint).

ComfyUI e9afbccc's own modules, imported from its source as a library on the CPU (`--cpu`; no server, no nodes), random-
initialized at a small config, against the H3-Image twin on the CPU reference backends (`cpu_ops`), same weights, same
inputs:
  DiT    MiniMaxH3ImageModel with 2 blocks (and the 2 refiner blocks), hidden 640, 4 heads of 128 (the real head dim and
         RoPE: 3 axes x 16 frequencies, 96 of 128 dims rotated), FFN 1024, text 384, the curve table [1025, 8], the 6-head
         PDD bank on the real pdd_sigmas. The refined text, then one step (ComfyUI `forward` with sample_sigmas, against
         `H3ImageDit.__call__` with `sampler.step`) at: 6 steps, steps 0 and 3 (one head each: row block 0, then
         rows 0 + 3); 3 steps, step 1 (a two-head mix); 30 steps, step 3 with one 128-px reference image and TE tags
         (references at max(t, 0.999), two timestep rows). 256 x 256 px (16 x 16 latents).
  VAE    MiniMaxH3ImageVAE with 2 of its 24 blocks at the real width (1024, 16 heads of 64, 2-D RoPE on 32 dims): one
         decode at 256 x 256 (one tile) and at 320 x 320 (2 x 2 tiles, overlap 192: the tile plan crossing), ComfyUI's
         `decode` (its tiled_decode, its blend) against `ImageVAE.decode`.
The ComfyUI-side randomness the twin replaces (the reference's condition noise) is drawn here as ComfyUI draws it and
handed to the twin. Modes: f32 (both sides fp32; the gate) and lowp (ComfyUI bf16 DiT / fp16 VAE, the twin's CPU kernels
at bf16 / fp16; informative). Pass: every f32 output at cosine >= 0.9999 AND relative L2 <= 1e-5. The two sides differ
only in summation order and in the twin's reciprocal-multiplies where torch divides (measured: rel L2 ~4e-7); cosine
alone is too blunt at these random weights: a reference at t instead of max(t, 0.999), the target's RoPE time one off
or a blend against the raw tiles move the cosine only to 0.99995 .. 0.999996, but the rel L2 to 3e-3 .. 7e-2.
`--controls` runs those faults (and a wrong PDD head, swapped mix weights, no y blend, half the RoPE pairs) one at a time
and fails unless the f32 thresholds catch each. lowp is held to cosine >= 0.999 (bf16 / fp16 rounding at different
points: the kernels' fp32-inside arithmetic vs torch's per-op rounding). One JSON line; exit 1 if a gated output fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager

import torch

from . import sampler

F32_COS, F32_REL, LOWP_COS = 0.9999, 1e-5, 0.999
CFG = dict(hidden_size=640, num_layers=2, token_refiner_num_layers=2, num_attention_heads=4, attention_head_dim=128,
           ffn_hidden_size=1024, text_dim=384, adaln_curve_grid=1025, time_embed_dim=8, rope_inv_freq_len=16, pdd_steps=6)
TEXT_LEN = 12
DIT_CASES = (  # (name, steps, step index, refs)
    ("s6_i0", 6, 0, False), ("s6_i3", 6, 3, False), ("s3_i1", 3, 1, False), ("s30_i3_ref", 30, 3, True))
VAE_SIDES = (256, 320)
VAE_LAYERS = 2


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    return {"cos": round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 7),
            "rel_l2": float((a - b).norm() / b.norm()), "max_abs": float((a - b).abs().max()), "ref_max": float(b.abs().max())}


def _fill(sd: dict, g: torch.Generator, skip=()) -> dict:
    """Random values for every float tensor of a state dict, by role: Linear weights N(0, 1/fan_in), biases 0.02 N,
    norm weights 1 + 0.1 N, the curve adaLN 0.2 N (shift, scale and gates of a few tenths), the residual scales
    0.5 + 0.1 N."""
    out = {}
    for k, v in sd.items():
        if any(s in k for s in skip) or not v.is_floating_point():
            continue
        shape = tuple(v.shape)
        r = torch.randn(shape, generator=g, dtype=torch.float32)
        if k == "rope.inv_freq":
            t = 10000.0 ** (-torch.arange(shape[0], dtype=torch.float32) / shape[0])
        elif k == "adaln_t_table" or k.endswith("register_tokens"):
            t = r
        elif "adaln_proj" in k:
            t = 0.2 * r
        elif k.endswith(("scale1", "scale2")):
            t = 0.5 + 0.1 * r
        elif ("norm" in k and k.endswith(".weight") and len(shape) == 1):
            t = 1.0 + 0.1 * r
        elif k.endswith(".bias"):
            t = 0.02 * r
        elif len(shape) >= 2:
            fan_in = math.prod(shape[1:])
            t = r / math.sqrt(fan_in)
        else:
            t = r
        out[k] = t
    return out


def _comfy(root: str):
    sys.argv = [sys.argv[0], "--cpu"]
    sys.path.insert(0, root)
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.ldm.minimax.model as M
    import comfy.ldm.minimax.vae as V
    import comfy.ops

    return M, V, comfy.ops


def dit_gate(M, ops, mode: str, seed: int) -> dict:
    from .cpu_ops import CpuOps
    from .dit import H3ImageDit

    cdt = torch.float32 if mode == "f32" else torch.bfloat16
    model = M.MiniMaxH3ImageModel(pdd_sigmas=list(sampler.PDD_SIGMAS), dtype=cdt, device="cpu",
                                  operations=ops.disable_weight_init, **CFG)
    g = torch.Generator().manual_seed(seed)
    w = _fill(model.state_dict(), g, skip=("comfy_attention",))
    # the curve adaLN and the heads are fp32 in ComfyUI (the curve form's adaln_dtype, video_out's fp32 island); the rest
    # at the model dtype, as a checkpoint load casts them
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in w.items()}, strict=False)
    model.eval().requires_grad_(False)          # as ComfyUI's model patcher holds it (kitchen's in-place RoPE needs it)
    sd = model.state_dict()
    twin = H3ImageDit({k: sd[k].clone() for k in w}, ops=CpuOps(cdt), device="cpu")
    res: dict = {}
    text = torch.randn(1, TEXT_LEN, CFG["text_dim"], generator=g).to(cdt)
    lat = torch.randn(1, 24, 1, 16, 16, generator=g)
    ref = torch.randn(1, 24, 1, 8, 8, generator=g)
    tags = torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    with torch.inference_mode():
        ctx = model.preprocess_text_embeds(text)
    for name, steps, i, refs in DIT_CASES:
        sig = sampler.sigmas(sampler.PDD_SIGMAS, steps)
        st = sampler.step(sampler.PDD_SIGMAS, sig, i)
        x = (torch.tensor(sig[i], dtype=torch.float32) * lat).to(cdt)
        payload: dict = {"seed": seed}
        ref_noise = None
        if refs:
            payload.update({"refs": [{"kind": "image", "latent": ref, "latent_h": 8, "latent_w": 8}],
                            "cond_video_latents": [ref], "text_token_tags": tags})
            ref_noise = [torch.randn((16, 96), generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)]
        with torch.inference_mode():
            timestep = torch.tensor([sig[i]], dtype=torch.float32) * 1000
            want = model(x, timestep, ctx, transformer_options={"sample_sigmas": torch.tensor(sig, dtype=torch.float32)},
                         minimax_payload=payload)
        tctx = twin.prepare(text, refs=[ref] if refs else (), seed=seed, text_tags=tags.numpy() if refs else None,
                            ref_noise=ref_noise)
        got = twin(x, st)
        res[name] = {"heads": st["heads"], "w": st["w"], "t": st["t"], **cmp(got, want)}
    res["context"] = cmp(tctx, ctx[0])
    return res


def vae_gate(V, mode: str, seed: int) -> dict:
    from .cpu_ops import CpuVaeOps
    from .vae_image import ImageVAE

    dt = torch.float32 if mode == "f32" else torch.float16
    vae = V.MiniMaxH3ImageVAE(num_layers=VAE_LAYERS)
    g = torch.Generator().manual_seed(seed + 1)
    sd = vae.state_dict()
    dec = {k: v for k, v in sd.items() if k.startswith(("decoder.", "post_quant_conv."))}
    w = _fill(dec, g, skip=("mask_token",))
    vae.load_state_dict(w, strict=False)
    vae = vae.to(dt).eval().requires_grad_(False)
    sd = vae.state_dict()
    twin = ImageVAE(ImageVAE.weights({k: sd[k] for k in w}, "cpu", dt), ops=CpuVaeOps(dt))
    res = {}
    for side in VAE_SIDES:
        n = side // 16
        z = torch.randn(1, 24, 1, n, n, generator=g)
        with torch.inference_mode():
            want = vae.decode(z.to(dt))[0, :, 0].permute(1, 2, 0)       # [H, W, 3] fp32 in [0, 1]
        _, got = twin.decode(z)
        res[f"{side}px"] = {"tiles": len(sampler_tiles(side)) ** 2, **cmp(got, want),
                            "clamped": round(float(((want <= 0) | (want >= 1)).float().mean()), 4)}
    return res


def sampler_tiles(side: int) -> list[int]:
    from ..h3.vae_video import split_tiles

    return split_tiles(side)[0]


def _ok(c: dict, mode: str) -> bool:
    return c["cos"] >= F32_COS and c["rel_l2"] <= F32_REL if mode == "f32" else c["cos"] >= LOWP_COS


@contextmanager
def _patched(obj, name, fn):
    orig = getattr(obj, name)
    setattr(obj, name, fn(orig))
    try:
        yield
    finally:
        setattr(obj, name, orig)


def controls(M, V, ops, seed: int) -> dict:
    """Negative controls (fp32): one deliberate fault at a time in the twin, the gated outputs it moves and their cosine.
    Each must fail the f32 thresholds, or the gate could not see that fault."""
    from .cpu_ops import CpuVaeOps
    from .dit import H3ImageDit, ImageLayout
    from .vae_image import ImageVAE

    def ref_t_unpinned(orig):                       # references at t, not max(t, 0.999)
        def f(self, t):
            u, idx, r = orig(self, t)
            for a, b, k in self.segments:
                if k == "ref_img":
                    idx[a:b] = r * 3
            return u, idx, r
        return f

    def target_time_early(orig):                    # the target's RoPE time at L + n_refs - 1
        def f(self):
            orig(self)
            a, b, _ = self.segments[-1]
            self.pos[a:b, 0] -= 1.0
        return f

    def raw_tile_blend(orig):                       # ComfyUI 0.37.0's rule: blend against the raw neighbour tiles
        def f(self, latents, tile_fn=None):
            raw, cur = {}, [None]

            def tf(z, y0, h, x0, w, name):
                i, j = (int(v) for v in name.split(".t")[1].split("_"))
                cur[0] = (i, j)
                raw[(i, j)] = self.decode_tile(z, y0, h, x0, w, name)
                return raw[(i, j)]

            place = self.ops.place_tile

            def pt(b, ytail, ey, ltail, ex, canvas, oy, ox, oh, ow):
                i, j = cur[0]
                return place(b, raw.get((i - 1, j)) if ytail is not None else None, ey,
                             raw.get((i, j - 1)) if ltail is not None else None, ex, canvas, oy, ox, oh, ow)

            self.ops.place_tile = pt
            try:
                return orig(self, latents, tile_fn=tf)
            finally:
                del self.ops.place_tile
        return f

    faults = {
        "ref_t_unpinned": (ImageLayout, "mod_rows", ref_t_unpinned, "dit", ["s30_i3_ref"]),
        "pdd_head0_only": (H3ImageDit, "head", lambda o: lambda self, st: o(self, {**st, "heads": [], "w": []}), "dit",
                           ["s6_i3", "s3_i1"]),
        "pdd_weights_swapped": (H3ImageDit, "head", lambda o: lambda self, st: o(self, {**st, "w": st["w"][::-1]}), "dit",
                                ["s3_i1"]),
        "target_time_early": (ImageLayout, "__post_init__", target_time_early, "dit", ["s6_i3", "s30_i3_ref"]),
        "vae_raw_tile_blend": (ImageVAE, "decode", raw_tile_blend, "vae", ["320px"]),
        "vae_no_y_blend": (CpuVaeOps, "place_tile", lambda o: lambda self, b, yt, ey, lt, ex, *r: o(self, b, None, 0, lt, ex, *r),
                           "vae", ["320px"]),
        "vae_rope_8_pairs": (CpuVaeOps, "rms_rope_", lambda o: lambda self, qkv, tb, S, H, eps, rot: o(
            self, qkv, tb[:, :8].contiguous(), S, H, eps, 16), "vae", ["256px", "320px"]),
    }
    out = {}
    for name, (obj, attr, fn, part, keys) in faults.items():
        with _patched(obj, attr, fn):
            r = dit_gate(M, ops, "f32", seed) if part == "dit" else vae_gate(V, "f32", seed)
        got = {k: {"cos": r[k]["cos"], "rel_l2": r[k]["rel_l2"]} for k in keys}
        out[name] = {**got, "seen": all(not _ok(v, "f32") for v in got.values())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="ComfyUI e9afbccc's source tree (imported as a library, CPU)")
    ap.add_argument("--mode", default="both", choices=("f32", "lowp", "both"))
    ap.add_argument("--controls", action="store_true", help="also the negative controls (deliberate faults, fp32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    M, V, ops = _comfy(a.comfy)
    out: dict = {"torch": torch.__version__, "config": {"dit": CFG, "vae_layers": VAE_LAYERS}, "thresholds":
                 {"f32": {"cos": F32_COS, "rel_l2": F32_REL}, "lowp": {"cos": LOWP_COS}}}
    ok = True
    for mode in (("f32", "lowp") if a.mode == "both" else (a.mode,)):
        r = {"dit": dit_gate(M, ops, mode, a.seed), "vae": vae_gate(V, mode, a.seed)}
        every = [v for part in r.values() for v in part.values()]
        r["min_cos"] = min(v["cos"] for v in every)
        r["max_rel_l2"] = max(v["rel_l2"] for v in every)
        r["pass"] = all(_ok(v, mode) for v in every)
        out[mode] = r
        ok &= r["pass"] or mode == "lowp"
    if a.controls:
        out["controls"] = controls(M, V, ops, a.seed)
        ok &= all(c["seen"] for c in out["controls"].values())
    out["pass"] = ok
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
