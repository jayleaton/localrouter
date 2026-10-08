"""python -m stk_twin <command> --model DIR ...   (one GPU; prints one JSON summary line per command)

  gate       bf16 twin vs the diffusers transformer (per-step cosine) and the whole image vs the diffusers pipeline
  calibrate  NVFP4 static input scales over the calibration prompts -> --out acts.json
  pack       converted weights for --precision nvfp4|fp8 (+ --acts), te (the text encoder) or vae -> --out DIR
  render     the gate prompts with a pack (or --precision bf16) -> --out DIR (PNGs + timings.json)
  bench      warm end-to-end and per-step times with a pack
  tegate     LocalRouter's text encoder vs transformers' (cosine per prompt), its determinism and time
  triton     tfimage's Triton cubins from a capture (--pack CAPTURE) into --out/triton (per SM)
  capture    one recorded denoising step (and the prefix build) with a pack -> --out DIR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .dit import QwenImageDiT
from .qwen_image import QwenImage, load_transformer, noise
from .rec import REC

HERE = Path(__file__).resolve().parent.parent


def size(s: str) -> tuple[int, int]:
    w, h = s.split("x")
    return int(h), int(w)  # (height, width)


def cos_rel(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().flatten(), b.float().flatten()
    return {"cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)), "rel_l2": float((a - b).norm() / b.norm())}


def psnr(a, b) -> float:
    import numpy as np
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mse = ((x - y) ** 2).mean()
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def twin(a, precision: str | None = None) -> QwenImage:
    p = precision or a.precision
    if p == "none":  # the pipeline's text encoder and VAE only
        return QwenImage(a.model, None)
    dit = QwenImageDiT.from_bf16(load_transformer(a.model), "bf16") if p == "bf16" else QwenImageDiT.from_pack(a.pack)
    return QwenImage(a.model, dit)


def cmd_gate(a):
    from diffusers import QwenImage21Transformer2DModel

    q = twin(a, "bf16")
    ref = QwenImage21Transformer2DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=torch.bfloat16).cuda()
    prompt, seed = json.load(open(HERE / "prompts/gate.json"))[0]
    H, W = size(a.size)
    emb, mask, pad = q.pipe._get_qwen_prompt_embeds(prompt, device=q.dev)
    ctx = emb[:, : int(mask.sum())]
    h, w = H // 16, W // 16
    x = torch.from_numpy(noise(seed, 64 * h * w)).view(1, 64, h, w).cuda().to(torch.bfloat16)
    img_mask = torch.cat([pad, pad.new_ones(1, h * w // 4)], dim=1)
    steps = {}
    with torch.inference_mode():
        for s in (1.0, 0.6, 0.2):
            ours = q.dit(x, torch.tensor([s], device="cuda"), ctx)
            t = (torch.tensor([s * 1000], dtype=torch.float32).to(torch.bfloat16) / 1000).cuda()
            theirs = ref(hidden_states=x.flatten(2).transpose(1, 2), timestep=t, encoder_hidden_states=emb,
                         encoder_hidden_states_mask=mask, img_shapes=[[(1, h, w)]], img_mask=img_mask, return_dict=False)[0][:, -h * w:]  # joint sequence: keep the image rows
            steps[str(s)] = cos_rel(ours.flatten(2).transpose(1, 2), theirs)
    ours_img = q.generate(prompt, H, W, a.steps, seed).image
    q.pipe.transformer = ref
    lat = x.flatten(2).transpose(1, 2).contiguous()
    with torch.inference_mode():
        ref_img = q.pipe(prompt=prompt, height=H, width=W, num_inference_steps=a.steps, latents=lat).images[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ours_img.save(out / "twin-bf16.png")
    ref_img.save(out / "diffusers.png")
    res = {"gate": "bf16 twin vs diffusers", "size": a.size, "steps_cos": steps, "image_psnr": psnr(ours_img, ref_img),
           "pass": all(v["cos"] >= 0.9999 for v in steps.values())}
    (out / "gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def cmd_floor(a):
    """The bf16 noise floor: the diffusers transformer against itself with another attention backend, same inputs.
    The twin's gate passes when its distance to diffusers is of this order."""
    from diffusers import QwenImage21Transformer2DModel
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = QwenImage(a.model, None)
    ref = QwenImage21Transformer2DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=torch.bfloat16).cuda()
    prompt, seed = json.load(open(HERE / "prompts/gate.json"))[0]
    H, W = size(a.size)
    emb, mask, pad = q.pipe._get_qwen_prompt_embeds(prompt, device=q.dev)
    h, w = H // 16, W // 16
    x = torch.from_numpy(noise(seed, 64 * h * w)).view(1, 64, h, w).cuda().to(torch.bfloat16).flatten(2).transpose(1, 2)
    img_mask = torch.cat([pad, pad.new_ones(1, h * w // 4)], dim=1)
    res = {}
    with torch.inference_mode():
        for s in (1.0, 0.6, 0.2):
            t = (torch.tensor([s * 1000], dtype=torch.float32).to(torch.bfloat16) / 1000).cuda()
            outs = []
            for be in (SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
                with sdpa_kernel([be]):
                    outs.append(ref(hidden_states=x, timestep=t, encoder_hidden_states=emb, encoder_hidden_states_mask=mask,
                                    img_shapes=[[(1, h, w)]], img_mask=img_mask, return_dict=False)[0][:, -h * w:])
            res[str(s)] = {"efficient": cos_rel(outs[1], outs[0]), "math": cos_rel(outs[2], outs[0])}
    print(json.dumps({"floor": "diffusers: cuDNN attention vs memory-efficient and math backends", "size": a.size, "steps_cos": res}))


def cmd_attncheck(a):
    """LocalRouter's attention against cuDNN at the DiT's shapes: accuracy (vs an fp32 reference too) and speed."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from .attn import attention

    torch.manual_seed(0)
    res = {}
    for name, (w, p, causal) in {"step-1024": (4096, 64, False), "step-512": (1024, 64, False),
                                 "prefix": (77, 0, True)}.items():
        s_ = w + p if not causal else w
        q = torch.randn(w, 32, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(s_, 32, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(s_, 32, 128, device="cuda", dtype=torch.bfloat16)
        qt, kt, vt = (t.transpose(0, 1)[None] for t in (q, k, v))
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
            ref = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)[0].transpose(0, 1)
        exact = torch.nn.functional.scaled_dot_product_attention(qt.float(), kt.float(), vt.float(), is_causal=causal)[0].transpose(0, 1)
        ours = attention(q, k, v, causal=causal)
        again = attention(q, k, v, causal=causal)

        def timed(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 20 * 1000

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
            t_ref = timed(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal))
        t_ours = timed(lambda: attention(q, k, v, causal=causal))
        res[name] = {"vs_cudnn": cos_rel(ours, ref), "ours_vs_fp32": cos_rel(ours, exact), "cudnn_vs_fp32": cos_rel(ref, exact),
                     "deterministic": bool(torch.equal(ours, again)), "ms_ours": t_ours, "ms_cudnn": t_ref}
    print(json.dumps({"attncheck": res}))


def cmd_calibrate(a):
    """Static input scales: NVFP4 as tfimage calibrates them; FP8 as each linear's largest input |x| over the same
    runs x FP8_MARGIN / 448 (TensorFold's A8 rows are e4m3(x / act))."""
    from tfimage.calibrate import calibrate
    from tfimage.sampling import euler

    from . import dit as D

    dit = QwenImageDiT.from_bf16(load_transformer(a.model), a.precision)
    q = QwenImage(a.model, dit)
    ctxs = [q.encode(p) for p in json.load(open(HERE / "prompts/calib.json"))]
    del q.pipe.text_encoder
    torch.cuda.empty_cache()
    if a.precision == "nvfp4":
        acts = calibrate(dit, ctxs)
    else:
        D.AMAX = {}
        g = torch.Generator(device="cuda").manual_seed(1234)
        sizes = ((64, 64), (76, 62), (48, 80))
        for i, ctx in enumerate(ctxs):
            dit.drop_prefixes()
            euler(dit, torch.randn((1, 64, *sizes[i % 3]), generator=g, device="cuda"), ctx.cuda(), 12)
        acts = {k: v * FP8_MARGIN / 448.0 for k, v in D.AMAX.items()}
        D.AMAX = None
    Path(a.out).write_text(json.dumps(acts, indent=1))
    print(json.dumps({"calibrate": len(acts), "precision": a.precision, "out": a.out}))


FP8_MARGIN = 2.0


def cmd_pack(a):
    from . import pack

    if a.precision == "te":  # the text encoder's bf16 pack and tokenizer
        man = pack.write_te(twin(a, "none").pipe, a.out, {"repo": "Qwen/Qwen-Image-2.1", "dir": a.model})
        print(json.dumps({"pack": a.out, "precision": "te", "tensors": len(man["tensors"])}))
        return
    if a.precision == "vae":  # the VAE decoder's bf16 pack
        man = pack.write_vae(twin(a, "none").pipe.vae, a.out, {"repo": "Qwen/Qwen-Image-2.1", "dir": a.model})
        print(json.dumps({"pack": a.out, "precision": "vae", "tensors": len(man["tensors"])}))
        return
    acts = json.load(open(a.acts)) if a.acts else {}
    man = pack.write(load_transformer(a.model), a.precision, acts, a.out, {"repo": "Qwen/Qwen-Image-2.1", "dir": a.model})
    print(json.dumps({"pack": a.out, "precision": a.precision, "tensors": len(man["tensors"])}))


def cmd_render(a):
    q = twin(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H, W = size(a.size)
    times = []
    for i, (prompt, seed) in enumerate(json.load(open(HERE / "prompts/gate.json"))):
        r = q.generate(prompt, H, W, a.steps, seed)
        r.image.save(out / f"{i}-{seed}.png")
        times.append(r.seconds)
    (out / "timings.json").write_text(json.dumps(times, indent=1))
    print(json.dumps({"render": str(out), "precision": getattr(q.dit, "precision", a.precision), "size": a.size, "seconds": times[-1]}))


def cmd_bench(a):
    q = twin(a)
    H, W = size(a.size)
    prompt, seed = json.load(open(HERE / "prompts/gate.json"))[0]
    q.generate(prompt, H, W, a.steps, seed)                       # warm: kernels built, prefix kept
    r = q.generate(prompt, H, W, a.steps, seed + 1)                # same prompt: prefix reused, as a server would
    ctx = q.encode(prompt)
    x = torch.from_numpy(noise(seed, 64 * (H // 16) * (W // 16))).view(1, 64, H // 16, W // 16).cuda().to(torch.bfloat16)
    t = torch.tensor([0.5], device="cuda")
    for _ in range(2):
        q.dit(x, t, ctx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        q.dit(x, t, ctx)
    torch.cuda.synchronize()
    step = (time.perf_counter() - t0) / 10
    res = {"bench": getattr(q.dit, "precision", a.precision), "size": a.size, "steps": a.steps, "warm_end_to_end": r.seconds, "step_s": step,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    print(json.dumps(res))


def cmd_tegate(a):
    """Gate prompts + calibration prompts: our text encoder against `pipe._get_qwen_prompt_embeds`."""
    from .te import TextEncoder, encode

    q = twin(a, "none")
    te = TextEncoder(q.pipe.text_encoder, q.dev)
    prompts = [p for p, _ in json.load(open(HERE / "prompts/gate.json"))] + json.load(open(HERE / "prompts/calib.json"))[:4]
    rows, worst = [], 1.0
    with torch.inference_mode():
        for p in prompts:
            emb, mask, _ = q.pipe._get_qwen_prompt_embeds(p, device=q.dev)
            ref = emb[:, : int(mask.sum())]
            ours = encode(te, q.pipe, p)
            again = encode(te, q.pipe, p)
            r = {"tokens": ref.shape[1], "same_shape": list(ours.shape) == list(ref.shape),
                 "deterministic": bool(torch.equal(ours, again))}
            if r["same_shape"]:
                r.update(cos_rel(ours, ref))
                worst = min(worst, r["cos"])
            rows.append(r)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            encode(te, q.pipe, prompts[0])
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 3 * 1e3
    print(json.dumps({"tegate": rows, "worst_cos": worst, "encode_ms": ms,
                      "pass": all(r["same_shape"] and r["deterministic"] for r in rows) and worst > 0.999}))


def cmd_triton(a):
    """tfimage's three Triton kernels from a capture (--pack is the capture dir) into --out/triton: triton.json maps
    kernel -> SM -> {cubin, num_warps, shared}, for the capture's own GPU and every SM in its cubins_by_arch."""
    import shutil

    cap, out = Path(a.pack), Path(a.out) / "triton"
    out.mkdir(parents=True, exist_ok=True)
    want = {"_adaln_kernel", "_rms_rope_kernel", "_swiglu_kernel"}
    kernels: dict = {}
    major, minor = torch.cuda.get_device_capability()
    for line in open(cap / "ops.jsonl"):
        rec = json.loads(line)
        for l in rec.get("launches", []):
            k = l["kernel"]
            if k not in want or k in kernels:
                continue
            arches = {f"{major}{minor}": {"cubin": l["cubin"], "num_warps": l["num_warps"], "shared": l["shared"]}}
            for sm, o in l.get("cubins_by_arch", {}).items():
                arches[sm] = {"cubin": o["cubin"], "num_warps": l["num_warps"], "shared": o["shared"]}
            for o in arches.values():
                shutil.copy(cap / "cubins" / o["cubin"], out / o["cubin"])
            kernels[k] = arches
    (out / "triton.json").write_text(json.dumps({"format": "stk-triton/1", "kernels": kernels}, indent=1))
    from .files import readable

    readable(out)
    print(json.dumps({"triton": str(out), "kernels": {k: sorted(v) for k, v in kernels.items()}}))


def capture_light(a):
    """The end-to-end gate's reference without the op stream: request, sigmas, text context, final latents and pixels
    (LocalRouter's text encoder and VAE: STK_TE=stk STK_VAE=stk), plus the twin's wall times."""
    from . import vae_ops

    q = twin(a)
    H, W = size(a.size)
    prompt, seed = json.load(open(HERE / "prompts/gate.json"))[a.prompt]
    out = Path(a.out)
    t = {}
    sync = torch.cuda.synchronize
    for rep in range(2):  # the second run is warm
        sync(); t0 = time.perf_counter()
        ctx = q.encode(prompt); sync(); t["encode"] = time.perf_counter() - t0; t0 = time.perf_counter()
        x = q.sample(ctx, H, W, a.steps, seed); sync(); t["sample"] = time.perf_counter() - t0; t0 = time.perf_counter()
        u = vae_ops.to_u8(q.pipe.vae.decode(vae_ops.latents_to_z(q.pipe.vae, x), return_dict=False)[0]); sync()
        t["decode"] = time.perf_counter() - t0
    REC.start(out)
    REC.note("request", prompt=prompt, seed=seed, height=H, width=W, steps=a.steps, precision=a.precision)
    REC.note("sigmas", values=q.sigmas(H, W, a.steps), seed=seed, height=H, width=W)
    with REC.op("text_encoder", "text_encoder", {"prompt": prompt}) as o:
        o.out(context=ctx)
    with REC.op("latents", "latents", {}) as o:
        o.out(x=x)
    with REC.op("vae.to_u8", "to_u8_hwc", {}) as o:
        o.out(y=u)
    REC.note("twin_warm_ms", **{k: v * 1e3 for k, v in t.items()})
    REC.stop()
    from PIL import Image
    Image.fromarray(u.cpu().numpy()).save(out / "image.png")
    print(json.dumps({"capture": str(out), "light": True, "warm_ms": {k: round(v * 1e3, 1) for k, v in t.items()}}))


def cmd_capture(a):
    if a.light:
        return capture_light(a)
    q = twin(a)
    H, W = size(a.size)
    prompt, seed = json.load(open(HERE / "prompts/gate.json"))[a.prompt]
    out = Path(a.out)
    REC.start(out)
    REC.note("request", prompt=prompt, seed=seed, height=H, width=W, steps=a.steps, precision=a.precision)
    ctx = q.encode(prompt)
    with REC.op("text_encoder", "text_encoder", {"prompt": prompt}) as o:
        o.out(context=ctx)
    x = q.sample(ctx, H, W, a.steps, seed, record_step=a.step)
    with REC.op("latents", "latents", {}) as o:
        o.out(x=x)
    from .qwen_image import VAE
    img = q.decode(x) if VAE == "stk" else None  # LocalRouter's VAE records its ops (vae.*) too
    REC.stop()
    lin = q.dit.blocks[0].qkv
    if hasattr(lin, "lin") and hasattr(lin.lin, "words"):  # TensorFold's NVFP4 tile layout, the golden for Zig's repack
        REC.start(out)
        with REC.op("form.L0.qkv", "weight_form", {"weight": "L0.qkv"}) as o:
            o.out(words=lin.lin.words, bs=lin.lin.bs)
        REC.stop()
    if hasattr(lin, "lin") and hasattr(lin.lin, "w8"):  # TensorFold's FP8 fragment order, the golden for Zig's repack
        REC.start(out)
        with REC.op("form.L0.qkv", "weight_form", {"weight": "L0.qkv"}) as o:
            o.out(w8=lin.lin.w8)
        REC.stop()
    (img or q.decode(x)).save(out / "image.png")
    print(json.dumps({"capture": str(out), "ops": REC.n}))


def main():
    ap = argparse.ArgumentParser(prog="stk_twin")
    ap.add_argument("command", choices=["gate", "floor", "attncheck", "calibrate", "pack", "render", "bench", "tegate", "triton", "capture"])
    ap.add_argument("--model", required=True, help="the Qwen/Qwen-Image-2.1 snapshot directory")
    ap.add_argument("--pack", help="a pack directory (render, bench, capture)")
    ap.add_argument("--precision", default="nvfp4")
    ap.add_argument("--acts")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--step", type=int, default=0, help="capture: the denoising step to record")
    ap.add_argument("--prompt", type=int, default=0, help="capture: the gate prompt (index into prompts/gate.json)")
    ap.add_argument("--light", action="store_true", help="capture: request, sigmas, context, latents, pixels only")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    globals()["cmd_" + a.command](a)


if __name__ == "__main__":
    main()
