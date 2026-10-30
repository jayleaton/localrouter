"""python -m stk_twin.kandinsky6_sampler.gate {schedule,sampler} ...: the Kandinsky-6 PiFlow sampler's bit gates
(kernels/cuda/kandinsky6/sampler/README.md "Gates"). Equality is the gate: no tolerance anywhere. An unequal tensor
is reported (unequal count, first mismatch, both sha256; max |diff| as information only), never waived. One JSON
document on stdout; exit status 1 on any inequality.

  schedule --table FILE [--config-from-table]
      S0, host: twin.schedule / twin.rollout (sampler.zig's arithmetic, line by line) against the table capture.py
      `schedule` recorded from upstream's own tensors. sampler.zig's `expectTable` checks the Zig side against the
      same file.

  sampler --capture DIR --case NAME --fatbin PATH [--chained]
      S1, sampler math alone (independent of the native DiT): for every step, upstream's captured DiT outputs (the
      normalized grids) drive the native policy through the fatbin. Checked bit for bit per step: the DiT's visual
      input built from the step's captured state (and captured first frames), the time scalar, and the video and audio
      states after the rollout. Default: every step from its own captured input state (isolates a step). --chained:
      the native state carries from the captured noise (only the DiT outputs are captured), ending with the final
      latents. Inputs are read from the capture only; nothing is re-derived.

  selftest
      compare's contract on the CPU: exact shape (equal bytes and numel under another shape are unequal), dtype,
      single-bit difference.

S2 (native DiT + native policy per step) and S3 (final latents from the native DiT) need the native DiT (PR #5 merged)
and its attention / GEMV qualification (M1b); they are not run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import twin


def load(torch, d: Path, name: str):
    """A capture record: the raw bytes checked against the header's sha256, as a CUDA tensor of its dtype / shape."""
    h = json.loads((d / f"{name}.json").read_text())
    data = (d / h["bin"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != h["sha256"] or h.get("byteorder") != "little":
        raise SystemExit(f"{name}: fixture bytes do not match their header")
    dt = getattr(torch, h["dtype"])
    t = torch.frombuffer(bytearray(data), dtype=dt) if data else torch.empty(0, dtype=dt)
    return t.reshape(h["shape"]).cuda()


def has(d: Path, name: str) -> bool:
    return (d / f"{name}.json").is_file()


def compare(torch, native, ref) -> dict:
    """Bit equality of two tensors: dtype and exact shape must agree first (equal bytes under another shape are
    unequal); statistics are informational. The native tensor carries its own shape, never the reference's."""
    r = {"dtype": [str(native.dtype), str(ref.dtype)], "shape": [list(native.shape), list(ref.shape)]}
    if native.dtype != ref.dtype:
        r["equal"] = False
        r["why"] = "dtype differs"
        return r
    if list(native.shape) != list(ref.shape):
        r["equal"] = False
        r["why"] = "shape differs"
        return r
    a = native.contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    b = ref.contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    r["sha_native"] = hashlib.sha256(a.tobytes()).hexdigest()
    r["sha_reference"] = hashlib.sha256(b.tobytes()).hexdigest()
    r["equal"] = r["sha_native"] == r["sha_reference"]
    if not r["equal"]:
        size = native.element_size()
        ea, eb = a.reshape(-1, size), b.reshape(-1, size)
        bad = np.nonzero((ea != eb).any(axis=1))[0]
        r["unequal"] = int(bad.size)
        r["first_mismatch"] = int(bad[0])
        diff = (native.float().reshape(-1) - ref.float().reshape(-1)).abs()
        r["max_abs_diff_informational"] = float(diff.max())
    return r


def cmd_schedule(a) -> int:
    doc = json.loads(Path(a.table).read_text())
    c = twin.Config(**doc["config"])
    res = {"gate": "S0 schedule", "config": doc["config"], "steps": [], "equal": True}
    got = list(twin.schedule(c))
    if len(got) != len(doc["steps"]):
        res["equal"] = False
        res["why"] = f"{len(got)} steps, the table has {len(doc['steps'])}"
    for st, want in zip(got, doc["steps"]):
        bad = [k for k in ("tau_src", "tau_dst", "sigma", "time", "policy_raw_src", "policy_raw_dst", "policy_segment")
               if twin.u32(st[k]) != want[k]]
        subs = list(twin.rollout(c, st))
        if len(subs) != len(want["substeps"]):
            bad.append(f"substeps {len(subs)} != {len(want['substeps'])}")
        for sub, w in zip(subs, want["substeps"]):
            bad += [f"substep {sub['index']} {k}" for k in ("sigma", "t", "raw_next", "sigma_next")
                    if twin.u32(sub[k]) != w[k]]
        res["steps"].append({"index": st["index"], "equal": not bad, "unequal": bad})
        res["equal"] &= not bad
    print(json.dumps(res, indent=1))
    return 0 if res["equal"] else 1


def cmd_sampler(a) -> int:
    import torch

    d = Path(a.capture) / a.case
    pol = twin.Policy(a.fatbin)
    args = json.loads((Path(a.capture) / "manifest.json").read_text())["cases"][a.case]["records"]
    run_args = next(r for r in args if r.get("name") == "in.args")
    if run_args["visual_cond_scheme"] != "pretrain" or run_args["visual_token_type_ids"]:
        raise SystemExit(f"unsupported capture: scheme {run_args['visual_cond_scheme']}, token type ids "
                         f"{run_args['visual_token_type_ids']} (only pretrain without type ids is native)")
    if not (run_args["sample_video"] and run_args["sample_audio"]):
        raise SystemExit("PiFlow advances video and audio together: a capture without both is refused")
    c = twin.Config(num_steps=run_args["num_steps"], shift=run_args["shift"], eps=run_args["eps"],
                    final_step_size_scale=run_args["final_step_size_scale"],
                    num_policy_substeps=run_args["num_policy_substeps"], n_grid=a.grids)
    first = load(torch, d, "in.first_frames") if has(d, "in.first_frames") else None
    noise_v, noise_a = load(torch, d, "in.video_noise"), load(torch, d, "in.audio_noise")
    if noise_v.dim() != 5 or noise_v.shape[0] != 1 or noise_a.dim() != 3 or noise_a.shape[0] != 1:
        raise SystemExit(f"B = 1 batched latents expected, got {tuple(noise_v.shape)} / {tuple(noise_a.shape)}")
    _, T, H, W, C = noise_v.shape
    rows, rows_first = T * H * W, H * W
    res = {"gate": "S1 sampler" + (" chained" if a.chained else " per step"), "case": a.case,
           "fatbin_sha256": pol.digest, "config": c.__dict__, "steps": [], "equal": True}
    state_v, state_a = noise_v, noise_a
    bufs = [torch.empty(noise_v.numel(), device="cuda", dtype=torch.float32) for _ in range(2)]
    bufa = [torch.empty(noise_a.numel(), device="cuda", dtype=torch.float32) for _ in range(2)]
    for st in twin.schedule(c):
        k = st["index"]
        p = f"step{k:02d}"
        if not a.chained and k > 0:
            state_v, state_a = load(torch, d, f"step{k - 1:02d}.x_after.video"), load(torch, d, f"step{k - 1:02d}.x_after.audio")
        row = {"index": k}
        xin = torch.empty(rows * (2 * C + 1), device="cuda", dtype=state_v.dtype)
        pol.video_input(state_v.reshape(-1), rows, C, None if first is None else first.reshape(-1), rows_first, xin)
        # native shapes from the contract: x_video [1, T, H, W, 2C + 1] (cat on the last axis), time [B] = [1]
        row["dit_in.x_video"] = compare(torch, xin.reshape(1, T, H, W, 2 * C + 1), load(torch, d, f"{p}.dit_in.x_video"))
        row["dit_in.time"] = compare(torch, torch.tensor([float(st["time"])], dtype=torch.float32, device="cuda"),
                                     load(torch, d, f"{p}.dit_in.time0"))
        gv, ga = load(torch, d, f"{p}.grids.video"), load(torch, d, f"{p}.grids.audio")
        if gv.shape[1] != c.n_grid or ga.shape[1] != c.n_grid:
            raise SystemExit(f"{p}: grids {tuple(gv.shape)} / {tuple(ga.shape)}, expected n_grid {c.n_grid} on dim 1")
        ov, oa = bufs[k % 2], bufa[k % 2]
        pol.advance(c, st, (state_v.reshape(-1), gv.reshape(-1), ov), (state_a.reshape(-1), ga.reshape(-1), oa))
        torch.cuda.synchronize()
        rv, ra = load(torch, d, f"{p}.x_after.video"), load(torch, d, f"{p}.x_after.audio")
        row["x_after.video"] = compare(torch, ov.reshape(noise_v.shape), rv)  # the state keeps [1, T, H, W, C]
        row["x_after.audio"] = compare(torch, oa.reshape(noise_a.shape), ra)  # and [1, A, 40]
        row["equal"] = all(v["equal"] for v in row.values() if isinstance(v, dict))
        res["steps"].append(row)
        res["equal"] &= row["equal"]
        state_v, state_a = ov.reshape(noise_v.shape), oa.reshape(noise_a.shape)
    if a.chained:
        fv, fa = load(torch, d, "out.video"), load(torch, d, "out.audio")
        res["final.video"] = compare(torch, state_v.reshape(noise_v.shape), fv)
        res["final.audio"] = compare(torch, state_a.reshape(noise_a.shape), fa)
        res["equal"] &= res["final.video"]["equal"] and res["final.audio"]["equal"]
    res["launches"] = pol.count
    pol.close()
    print(json.dumps(res, indent=1))
    return 0 if res["equal"] else 1


def cmd_selftest(a) -> int:
    """compare's own contract on the CPU (no CUDA, no capture). Written for the future check run; not yet run."""
    import torch

    x = torch.arange(12, dtype=torch.float32)
    cases = {
        "equal": (compare(torch, x.reshape(3, 4), x.reshape(3, 4).clone()), True),
        "same bytes and numel, different shape": (compare(torch, x.reshape(3, 4), x.reshape(4, 3)), False),
        "same bytes and numel, extra unit axis": (compare(torch, x.reshape(1, 3, 4), x.reshape(3, 4)), False),
        "different dtype": (compare(torch, x, x.to(torch.bfloat16)), False),
        "one bit differs": (compare(torch, x, x.clone().index_fill_(0, torch.tensor([5]), 5.000001)), False),
    }
    res = {name: {"equal": got["equal"], "why": got.get("why"), "ok": got["equal"] == want}
           for name, (got, want) in cases.items()}
    res["ok"] = all(v["ok"] for v in res.values())
    res["shape_reason"] = cases["same bytes and numel, different shape"][0].get("why") == "shape differs"
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] and res["shape_reason"] else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("schedule")
    s.add_argument("--table", required=True)
    m = sub.add_parser("sampler")
    m.add_argument("--capture", required=True)
    m.add_argument("--case", required=True)
    m.add_argument("--fatbin", default=None)
    m.add_argument("--grids", type=int, default=10)
    m.add_argument("--chained", action="store_true")
    a = ap.parse_args()
    sys.exit({"schedule": cmd_schedule, "sampler": cmd_sampler, "selftest": cmd_selftest}[a.cmd](a))


if __name__ == "__main__":
    main()
