"""python -m stk_twin.h3.generate --models DIR --pack PACK --out DIR [--prompt P --seed N --size 768x448 --frames 56 --steps 8]

MiniMax H3 text to video + audio end to end, the shipping form (the 32B text encoder on LocalRouter's kernels, the pack's
NVFP4 DiT with INT8 attention, the two VAEs), chained as ComfyUI 0.37.0 runs the workflow and as the Zig pipeline
(src/engines/minimax_h3/pipeline.zig) runs it, so a Zig run from the same prompt and seed is bit-exact with this one.

The loop, on the packed fp32 sampler state x = [video | audio] (the audio is the carried variable, see gate.py), drawn by
LocalRouter's portable generator (one stream, video first; no process_latent_in for text to video). Per step i:
  1. xb = bf16(x), round to nearest even (ComfyUI casts the packed state to bf16 before the model);
  2. the DiT on (xb video, xb audio, sigma_i): the audio carry, the blocks, the un-carry are inside `H3Dit.__call__`;
     its bf16 velocities (negated, the audio back on the carried variable) are packed [video | audio];
  3. den = x - float(vel) * sigma_i                          (ops.cu h3_denoise: ComfyUI's CONST.calculate_denoised);
  4. step 0 and the last step Euler: x = x + ((x - den) / sigma_i) * dt, dt = sigma_next - sigma_i   (h3_euler32);
     the others res_multistep (eta 0): x = e * x + h * (b1 * den + b2 * old), old = the previous step's den (h3_res2);
     the scalars are `sampler.plan` (portable fdlibm log / expm1 / exp, fp32 ops);
  5. old = den.
After the last step: process_latent_out, audio *= 0.25 in fp32 (h3_scale32; the video is unchanged); the audio VAE decodes the
audio latents (fp32 [1, 32, 2, A], the VAE normalises the waveform), the video VAE the video latents (fp32, which it casts to
fp16 itself). The frame count snaps up to 17 k + 5 (ComfyUI's align_frame_count, at least 5).

Written to --out: the capture (REC) with only light ops, in order: `text_encoder` (fp32 states and the bf16 context), `noise`
(x), `step.{i}` (in: x; out: vel_v, vel_a, den, x), `latents` (video, audio after the 0.25), `waveform` (y); the notes
`request`, `sigmas`, `te32_ids`, `twin_ms`, `result` (sha256 of the uint8 frames and of the fp32 waveform); plus
`frames.u8` ([F, H, W, 3]), `audio.f32` ([2, A * 800] at 32 kHz) and `result.json`. `localrouter check h3-e2e` replays it. One JSON line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ..qwen_image import noise
from ..rec import REC
from . import ops as H
from . import sampler
from .build import TE
from .dit import H3Dit
from .shapes import PROMPT, latent_shapes
from .te32 import TextEncoder32, tokenize
from .vae_audio import AudioVAE
from .vae_video import VideoVAE

VAE_AUDIO = "vae/minimax_h3_audio_vae_fp32.safetensors"
VAE_VIDEO = "vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_SCALE_OUT = 0.25   # process_latent_out: audio * 1 / audio_scale (4)


def align_frames(n: int) -> int:
    """ComfyUI's align_frame_count(max(5, n)): up to the next 17 k + 5."""
    n = max(5, n)
    while n % 17 != 5:
        n += 1
    return n


@contextmanager
def quiet():
    """The heavy ops (encoder, DiT, VAEs) run unrecorded: only the light ops of this module go to REC."""
    on, REC.on = REC.on, False
    try:
        yield
    finally:
        REC.on = on


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def generate(models: Path, pack: str, prompt: str, seed: int, width: int, height: int, frames: int, steps: int,
             out: Path, dev=torch.device("cuda")) -> dict:
    if width % 32 or height % 32:
        raise SystemExit(f"--size {width}x{height}: width and height must be multiples of 32")
    requested, frames = frames, align_frames(frames)
    ms: dict[str, float] = {}
    REC.start(out)
    REC.note("request", prompt=prompt, seed=seed, width=width, height=height, frames=frames, frames_requested=requested,
             steps=steps)
    sig = sampler.sigmas(steps)
    plan = sampler.plan(sig)
    REC.note("sigmas", values=sig)

    # ---- encode: tokens, the 32B encoder (freed after), the bf16 hand-off, condition_proj + token refiner
    t0 = sync()
    ids = tokenize(prompt)
    REC.note("te32_ids", ids=ids)
    with quiet():
        te = TextEncoder32(Path(models) / TE)
        t1 = sync()
        hidden = te.forward(torch.tensor(ids))                      # fp32 [L, 5120]
        context = hidden.to(torch.bfloat16).contiguous()            # [L, 5120]
        del te
        torch.cuda.empty_cache()
        t2 = sync()
        dit = H3Dit.from_pack(pack, attn="int8", device=dev)
        t3 = sync()
        dit.prepare(context.unsqueeze(0))
    with REC.op("text_encoder", "text_encoder", {}) as o:
        o.out(hidden=hidden, context=context)
    t4 = sync()
    ms["encode"] = ((t2 - t1) + (t4 - t3)) * 1e3

    # ---- the sampler state
    vs, as_ = latent_shapes(width, height, frames)
    nv, na = int(np.prod(vs)), int(np.prod(as_))
    x = torch.from_numpy(noise(seed, nv + na)).to(dev).contiguous()   # fp32 [video | audio]
    with REC.op("noise", "noise", {"seed": seed}) as o:
        o.out(x=x)
    mod = H.mod()
    old = None
    t5 = sync()
    for i, st in enumerate(plan):
        sigma = sig[i]
        with REC.op(f"step.{i}", "sampler_step", {"i": i, **st}, x=x) as o:
            xb = x.to(torch.bfloat16)
            with quiet():
                vv, va = dit(xb[:nv].view(vs), xb[nv:].view(as_), sigma)
            vel = torch.cat([vv.reshape(-1), va.reshape(-1)])
            den = mod.k_denoise(x, vel, sigma)
            if st["kind"] == "euler":
                mod.k_euler32_(x, den, st["sigma"], st["dt"])
            else:
                mod.k_res2_(x, den, old, st["e"], st["h"], st["b1"], st["b2"])
            o.out(vel_v=vv, vel_a=va, den=den, x=x)
        old = den
    mod.k_scale32_(x[nv:], AUDIO_SCALE_OUT)
    lat_v, lat_a = x[:nv].view(vs), x[nv:].view(as_)
    with REC.op("latents", "latents", {}) as o:
        o.out(video=lat_v, audio=lat_a)
    t6 = sync()
    ms["sample"] = (t6 - t5) * 1e3
    del dit
    torch.cuda.empty_cache()

    # ---- decode
    with quiet():
        avae = AudioVAE.from_checkpoint(Path(models) / VAE_AUDIO, device="cuda")
        t7 = sync()
        wav = avae.decode(lat_a)                                    # fp32 [1, 2, A * 800]
    with REC.op("waveform", "waveform", {}) as o:
        o.out(y=wav)
    t8 = sync()
    ms["audio"] = (t8 - t7) * 1e3
    del avae
    with quiet():
        vvae = VideoVAE.from_checkpoint(Path(models) / VAE_VIDEO, device="cuda")
        t9 = sync()
        px = vvae.decode(lat_v)                                     # uint8 [F, H, W, 3]
    t10 = sync()
    ms["video"] = (t10 - t9) * 1e3

    out = Path(out)
    frames_u8 = px.cpu().numpy()
    wav_f32 = wav.cpu().numpy()
    frames_u8.tofile(out / "frames.u8")
    wav_f32.tofile(out / "audio.f32")
    res = {"frames_sha256": hashlib.sha256(frames_u8.tobytes()).hexdigest(), "frames_shape": list(frames_u8.shape),
           "waveform_sha256": sha(wav), "waveform_shape": list(wav.shape), "tokens": len(ids),
           "latent": {"video": list(vs), "audio": list(as_)}}
    REC.note("twin_ms", **ms)
    REC.note("result", **res)
    REC.stop()
    (out / "result.json").write_text(json.dumps({"request": {"prompt": prompt, "seed": seed, "width": width,
                                                              "height": height, "frames": frames, "steps": steps},
                                                  **res, "twin_ms": ms}, indent=1))
    return {"out": str(out), "ops": REC.n, "ms": {k: round(v, 1) for k, v in ms.items()}, **res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("VMODELS", ""))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", default="768x448")
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"h3_generate": generate(Path(a.models), a.pack, a.prompt, a.seed, w, h, a.frames, a.steps,
                                              Path(a.out))}))


if __name__ == "__main__":
    main()
