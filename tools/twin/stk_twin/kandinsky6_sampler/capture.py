"""python -m stk_twin.kandinsky6_sampler.capture {schedule,run} ...: upstream kandinsky-6 eac06e39's own PiFlow
sampler, recorded. Nothing here computes a sampler value: every number written is a tensor upstream made, read from
the module, the loop's arguments or its return values through wrappers that call the original functions unchanged.

  schedule --upstream SRC --out FILE [--steps 10 --shift 5.0 --eps 1e-6 --final-step-size-scale 0.5
           --substeps 128 --grids 10] --grant ID
      The schedule table sampler.zig's `expectTable` reads (gate S0). Runs `piflow_denoise_loop` on the GPU with a stub
      DiT (multimodal, visual_cond, zero bf16 grids of the right shape) on a tiny state: the schedule depends on the
      settings only, not on the latents, so no checkpoint is read. Recorded per step: tau_src and sigma (the loop's
      `shift_timesteps` call), time (the DiT's `time` argument), tau_dst (`policy_rollout_fm`'s raw_t_end), the policy's
      raw_t_src / raw_t_dst / segment_size, the substep count (`pi` calls); per substep: pi's sigma_t, _interpolate's t,
      and the rollout's `shift_timesteps` input and output (raw_next, sigma_next). Video and audio are recorded apart and
      must agree.

  run --upstream SRC --config pro-distill.yaml --manifest CASES.json --out DIR --grant ID [--case NAME ...]
      The full reference capture (gates S1 to S3): the published Pro-distill pipeline (`get_pipeline`, the 80.6 GB
      checkpoint; needs its own owner admission and memory plan) for each case of the manifest
      {"cases": [{"name": ..., "prompt": ..., "seed": ..., "image": optional path}]}. Recorded per case: the loop's
      inputs (the noise as drawn, in its dtype; first_frames; text_embeds, pooled_embed, the key mask; every RoPE table;
      the DiT's `freqs` buffers), per step the DiT's inputs and raw and normalized outputs and the state after each
      rollout, and the final latents. Tensors: DIR/<case>/<name>.bin (raw little-endian bytes) + .json (dtype, shape,
      sha256, byteorder); DIR/manifest.json lists every record, the environment and the upstream file digests.

Both refuse to start without --grant (the owner's admission id, recorded in the receipt): they initialize CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path

UPSTREAM_COMMIT = "eac06e39b904940baff2c408e925cdb0c4529269"
UPSTREAM_FILES = (
    "kandinsky/core/algo/piflow_sampler.py",
    "kandinsky/core/algo/piflow_math.py",
    "kandinsky/core/algo/denoise_loop.py",
    "kandinsky/core/algo/prepare_latents.py",
    "kandinsky/runtime/weights.py",
    "kandinsky/runtime/sampler.py",
    "kandinsky/pipeline/pipeline.py",
    "kandinsky/configs/checkpoints/pro-distill.yaml",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def upstream_receipt(src: Path) -> dict:
    """The checkout's commit (must be eac06e39 when readable) and the digests of the files the sampler runs."""
    rec = {"path": str(src), "files": {f: sha256_file(src / f) for f in UPSTREAM_FILES if (src / f).is_file()}}
    try:
        rec["commit"] = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rec["commit"] = None
    if rec["commit"] is not None and rec["commit"] != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream checkout is at {rec['commit']}, not {UPSTREAM_COMMIT}")
    missing = [f for f in UPSTREAM_FILES if f not in rec["files"]]
    if missing:
        raise SystemExit(f"upstream checkout lacks {missing}")
    return rec


def env_receipt(torch) -> dict:
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def bits(t) -> int:
    """The u32 bits of a one-element f32 tensor, read back as stored."""
    if t.dtype.is_floating_point and t.element_size() == 4 and t.numel() == 1:
        return struct.unpack("<I", t.detach().reshape(-1).cpu().numpy().tobytes())[0]
    raise ValueError(f"expected a one-element f32 tensor, got {t.dtype} {tuple(t.shape)}")


# ---------------------------------------------------------------------------------------------------------- hooks
class Recorder:
    """Wraps upstream's sampler functions in place (restored on exit): each wrapper calls the original and hands its
    tensors to `self.on`."""

    def __init__(self, on):
        self.on = on
        self._saved = []

    def _patch(self, obj, name, new):
        self._saved.append((obj, name, obj.__dict__[name] if isinstance(obj, type) else getattr(obj, name)))
        setattr(obj, name, new)

    def __enter__(self):
        from kandinsky.core.algo import piflow_math as pm
        from kandinsky.core.algo import piflow_sampler as ps

        on = self.on
        loop_shift, roll_shift = ps.shift_timesteps, pm.shift_timesteps
        rollout, normalize = ps.policy_rollout_fm, ps._normalize_grid_prediction
        pi, interp = pm.DXPolicy.pi, pm.DXPolicy._interpolate

        def loop_shift_w(t, shift):
            out = loop_shift(t, shift)
            on("step_shift", tau_src=t, sigma=out)
            return out

        def roll_shift_w(t, shift):
            out = roll_shift(t, shift)
            on("substep_shift", raw_next=t, sigma_next=out)
            return out

        def rollout_w(x_t_start, sigma_t_start, raw_t_start, raw_t_end, total_substeps, policy):
            on("rollout_begin", x=x_t_start, sigma=sigma_t_start, raw_start=raw_t_start, raw_end=raw_t_end,
               policy_raw_src=policy.raw_t_src, policy_raw_dst=policy.raw_t_dst, policy_segment=policy.segment_size)
            out = rollout(x_t_start, sigma_t_start, raw_t_start, raw_t_end, total_substeps, policy)
            on("rollout_end", x=out[0])
            return out

        def normalize_w(prediction, state, name):
            out = normalize(prediction, state, name)
            on("normalized", name=name, raw=prediction, grids=out)
            return out

        def pi_w(self_, x_t, sigma_t):
            on("pi", sigma=sigma_t)
            return pi(self_, x_t, sigma_t)

        def interp_w(x, t):
            on("interpolate", t=t)
            return interp(x, t)

        self._patch(ps, "shift_timesteps", loop_shift_w)
        self._patch(pm, "shift_timesteps", roll_shift_w)
        self._patch(ps, "policy_rollout_fm", rollout_w)
        self._patch(ps, "_normalize_grid_prediction", normalize_w)
        self._patch(pm.DXPolicy, "pi", pi_w)
        self._patch(pm.DXPolicy, "_interpolate", staticmethod(interp_w))
        return self

    def __exit__(self, *exc):
        for obj, name, old in reversed(self._saved):
            setattr(obj, name, old)
        self._saved.clear()
        return False


class DitRecorder:
    """The DiT as the loop sees it: attributes forwarded, every call's inputs and outputs handed to `on`."""

    def __init__(self, dit, on):
        object.__setattr__(self, "_dit", dit)
        object.__setattr__(self, "_on", on)

    def __getattr__(self, name):
        return getattr(self._dit, name)

    def __call__(self, **kw):
        self._on("dit_in", **kw)
        out = self._dit(**kw)
        self._on("dit_out", video=out[0], audio=out[1])
        return out


class ScheduleTable:
    """Collects the schedule records into sampler.zig's table: one list a modality, which must agree."""

    def __init__(self):
        self.steps = []
        self.mod = None  # the rollout in progress: its step dict

    def on(self, kind, **kw):
        if kind == "step_shift":
            self.steps.append({"tau_src": bits(kw["tau_src"]), "sigma": bits(kw["sigma"]), "rollouts": []})
        elif kind == "dit_in":
            t = kw["time"]
            if not (isinstance(t, (list, tuple)) and len(t) == 2 and bits(t[0]) == bits(t[1])):
                raise SystemExit("the DiT's two times differ")
            self.steps[-1]["time"] = bits(t[0])
        elif kind == "rollout_begin":
            r = {"tau_src": bits(kw["raw_start"]), "tau_dst": bits(kw["raw_end"]), "sigma": bits(kw["sigma"]),
                 "policy_raw_src": bits(kw["policy_raw_src"]), "policy_raw_dst": bits(kw["policy_raw_dst"]),
                 "policy_segment": bits(kw["policy_segment"]), "substeps": []}
            self.steps[-1]["rollouts"].append(r)
            self.mod = r
        elif kind == "pi":
            self.mod["substeps"].append({"sigma": bits(kw["sigma"])})
        elif kind == "interpolate":
            self.mod["substeps"][-1]["t"] = bits(kw["t"])
        elif kind == "substep_shift":
            self.mod["substeps"][-1].update(raw_next=bits(kw["raw_next"]), sigma_next=bits(kw["sigma_next"]))
        elif kind == "rollout_end":
            self.mod = None

    def table(self) -> list:
        out = []
        for i, s in enumerate(self.steps):
            v, a = s["rollouts"]  # video, then audio (piflow_sampler.py:157, 166)
            if v != a:
                raise SystemExit(f"step {i}: the video and audio schedules differ")
            if v["tau_src"] != s["tau_src"] or v["sigma"] != s["sigma"]:
                raise SystemExit(f"step {i}: the rollout's start is not the loop's tau / sigma")
            if i > 0 and out[-1]["tau_dst"] != s["tau_src"]:
                raise SystemExit(f"step {i}: tau_src is not the previous tau_dst")
            st = {k: v[k] for k in ("tau_src", "tau_dst", "sigma", "policy_raw_src", "policy_raw_dst", "policy_segment")}
            st["time"] = s["time"]
            st["substeps"] = v["substeps"]
            out.append(st)
        return out


def require_cuda(a):
    if not a.grant:
        raise SystemExit("--grant ID is required: this initializes CUDA (owner-admitted window only)")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    return torch


def cmd_schedule(a) -> None:
    torch = require_cuda(a)
    sys.path.insert(0, str(a.upstream))
    rec_up = upstream_receipt(Path(a.upstream))
    from kandinsky.core.algo.piflow_sampler import piflow_denoise_loop
    from kandinsky.core.types import LatentBundle

    dev = torch.device("cuda")
    T, H, W, C, A, D = 2, 2, 2, 16, 3, 40

    class StubDit:
        is_multimodal = True
        visual_cond = True

        def __call__(self, **kw):
            return (torch.zeros(1, a.grids, T, H, W, C, device=dev, dtype=torch.bfloat16),
                    torch.zeros(1, a.grids, A, D, device=dev, dtype=torch.bfloat16))

    bundle = LatentBundle(video=torch.zeros(1, T, H, W, C, device=dev, dtype=torch.bfloat16),
                          audio=torch.zeros(1, A, D, device=dev, dtype=torch.bfloat16),
                          video_cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=dev),
                          audio_cu_seqlens=torch.tensor([0, A], dtype=torch.int32, device=dev))
    tab = ScheduleTable()
    with Recorder(tab.on):
        piflow_denoise_loop(bundle=bundle, dit=DitRecorder(StubDit(), tab.on),
                            text_embeds={"text_embeds": torch.zeros(1, 1, 8, device=dev),
                                         "pooled_embed": torch.zeros(1, 8, device=dev)},
                            visual_rope=None, audio_rope=None, text_rope=None, num_steps=a.steps,
                            scheduler_scale=a.shift, eps=a.eps, final_step_size_scale=a.final_step_size_scale,
                            shift=a.shift, num_policy_substeps=a.substeps)
    doc = {"config": {"num_steps": a.steps, "shift": float(a.shift), "eps": float(a.eps),
                      "final_step_size_scale": float(a.final_step_size_scale), "num_policy_substeps": a.substeps,
                      "n_grid": a.grids},
           "provenance": {"upstream": rec_up, "env": env_receipt(torch), "grant": a.grant},
           "steps": tab.table()}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"out": a.out, "steps": len(doc["steps"]),
                      "substeps": sum(len(s["substeps"]) for s in doc["steps"])}))


# ------------------------------------------------------------------------------------------------------------ run
HASH_BYTES = 256 << 20


def save(torch, out: Path, name: str, t, meta=None) -> dict:
    """t's raw little-endian bytes to out/name.bin, its header to out/name.json; the record."""
    t = t.detach().contiguous()
    flat = t.reshape(-1).view(torch.uint8)
    h = hashlib.sha256()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.bin", "wb") as f:
        for s in range(0, flat.numel(), HASH_BYTES):
            chunk = flat[s:s + HASH_BYTES].cpu().numpy()
            h.update(chunk)
            f.write(chunk.tobytes())
    rec = {"name": name, "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "byteorder": "little",
           "sha256": h.hexdigest(), "bin": f"{name}.bin", **(meta or {})}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


class RunRecorder:
    def __init__(self, torch, out: Path):
        self.torch, self.out, self.step, self.records, self.rollouts = torch, out, -1, [], 0

    def put(self, name, t, **meta):
        if t is None:
            return
        self.records.append(save(self.torch, self.out, name, t, meta))

    def on(self, kind, **kw):
        torch = self.torch
        if kind == "step_shift":
            self.step += 1
            self.rollouts = 0
            self.put(f"step{self.step:02d}.tau_src", kw["tau_src"])
            self.put(f"step{self.step:02d}.sigma", kw["sigma"])
        elif kind == "dit_in":
            p = f"step{self.step:02d}.dit_in"
            for k in ("x_video", "x_audio", "text_embed", "pooled_text_embed", "visual_rope", "audio_rope",
                      "attention_mask", "visual_token_type_ids"):
                if isinstance(kw.get(k), torch.Tensor):
                    self.put(f"{p}.{k}", kw[k])
            for i, t in enumerate(kw["time"]):
                self.put(f"{p}.time{i}", t)
            tr = kw.get("text_rope")
            for i, t in enumerate(tr if isinstance(tr, (list, tuple)) else [tr]):
                if isinstance(t, torch.Tensor):
                    self.put(f"{p}.text_rope{i}", t)
        elif kind == "dit_out":
            self.put(f"step{self.step:02d}.dit_out.video", kw["video"])
            self.put(f"step{self.step:02d}.dit_out.audio", kw["audio"])
        elif kind == "normalized":
            self.put(f"step{self.step:02d}.grids.{kw['name']}", kw["grids"])
        elif kind == "rollout_end":
            m = ("video", "audio")[self.rollouts]
            self.put(f"step{self.step:02d}.x_after.{m}", kw["x"])
            self.rollouts += 1


def cmd_run(a) -> None:
    torch = require_cuda(a)
    sys.path.insert(0, str(a.upstream))
    rec_up = upstream_receipt(Path(a.upstream))
    import kandinsky.pipeline.pipeline as pp
    from kandinsky.pipeline.factory import get_pipeline

    manifest_path = Path(a.manifest)
    cases = json.loads(manifest_path.read_text())["cases"]
    if a.case:
        cases = [c for c in cases if c["name"] in a.case]
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    pipe = get_pipeline(a.config)
    loop = pp.piflow_denoise_loop
    receipt = {"upstream": rec_up, "env": env_receipt(torch), "grant": a.grant, "config": a.config,
               "config_sha256": sha256_file(Path(a.config)), "manifest_sha256": sha256_file(manifest_path),
               "cases": {}}
    for case in cases:
        out = out_root / case["name"]
        rr = RunRecorder(torch, out)

        def loop_w(*args, **kw):
            if args:
                raise SystemExit("piflow_denoise_loop called positionally; expected the pipeline's keyword call")
            b = kw["bundle"]
            rr.put("in.video_noise", b.video)
            rr.put("in.audio_noise", b.audio)
            rr.put("in.video_cu_seqlens", b.video_cu_seqlens)
            rr.put("in.audio_cu_seqlens", b.audio_cu_seqlens)
            rr.put("in.first_frames", kw.get("first_frames"))
            rr.put("in.text_embeds", kw["text_embeds"]["text_embeds"])
            rr.put("in.pooled_embed", kw["text_embeds"]["pooled_embed"])
            rr.put("in.attention_mask", kw.get("attention_mask"))
            rr.put("in.visual_rope", kw.get("visual_rope"))
            rr.put("in.audio_rope", kw.get("audio_rope"))
            named = getattr(kw["dit"], "named_buffers", None)
            for name, buf in (named() if named is not None else ()):
                if name.endswith("freqs"):
                    rr.put(f"in.buffer.{name}", buf)
            rr.records.append({"name": "in.args", "visual_cond_scheme": kw.get("visual_cond_scheme"),
                               "sample_video": kw.get("sample_video", True), "sample_audio": kw.get("sample_audio"),
                               "num_steps": kw.get("num_steps"), "shift": kw.get("shift"), "eps": kw.get("eps"),
                               "final_step_size_scale": kw.get("final_step_size_scale"),
                               "num_policy_substeps": kw.get("num_policy_substeps"),
                               "visual_token_type_ids": kw.get("visual_token_type_ids") is not None})
            kw["dit"] = DitRecorder(kw["dit"], rr.on)
            res = loop(**kw)
            rr.put("out.video", res.video)
            rr.put("out.audio", res.audio)
            return res

        pp.piflow_denoise_loop = loop_w
        try:
            with Recorder(rr.on):  # the pipeline's own grad / precision settings, untouched
                pipe(case["prompt"], seed=int(case["seed"]), image=case.get("image"),
                     save_path=str(out / "reference.mp4"))
        finally:
            pp.piflow_denoise_loop = loop
        receipt["cases"][case["name"]] = {"case": case, "steps": rr.step + 1, "records": rr.records}
        with contextlib.suppress(OSError):
            os.sync()
    (out_root / "manifest.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"out": str(out_root), "cases": list(receipt["cases"])}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("schedule")
    s.add_argument("--upstream", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--grant", default="")
    s.add_argument("--steps", type=int, default=10)
    s.add_argument("--shift", type=float, default=5.0)
    s.add_argument("--eps", type=float, default=1e-6)
    s.add_argument("--final-step-size-scale", type=float, default=0.5)
    s.add_argument("--substeps", type=int, default=128)
    s.add_argument("--grids", type=int, default=10)
    r = sub.add_parser("run")
    r.add_argument("--upstream", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--grant", default="")
    r.add_argument("--case", action="append")
    a = ap.parse_args()
    {"schedule": cmd_schedule, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
