"""python -m stk_twin.h3.test_ops: the H3 kernels (kernels/cuda/minimax/ops.cu) against ComfyUI's and tfvideo's
formulas in torch, at H3 shapes. Data moves must be exact; arithmetic within one bf16 ulp of the torch formula
(the fp32 sum order differs), the fp32 GEMM within the fp32 accumulation bound of fp64; all deterministic.
One JSON line."""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

from . import ops as H

RES: dict[str, dict] = {}


def ulps(a, b):
    def line(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (line(a) - line(b)).abs()


def check(name, got, ref, max_ulp=1):
    if got.dtype == torch.bfloat16:
        u = int(ulps(got, ref).max())
        ok = u <= max_ulp
        RES[name] = {"ok": ok, "max_ulp": u, "identical": float((got == ref).float().mean())}
    else:
        ok = bool(torch.equal(got, ref))
        RES[name] = {"ok": ok, "identical": float((got == ref).float().mean())}
    print(f"{'PASS' if ok else 'FAIL'} {name} {RES[name]}", flush=True)


def det(name, fn):
    a, b = fn(), fn()
    RES.setdefault(name, {})["deterministic"] = bool(torch.equal(a, b))


def main() -> int:
    torch.manual_seed(0)
    m = H.mod()
    dev = "cuda"
    S, D = 1000, 5376
    x = (torch.randn(S, D, device=dev) * 2).bfloat16()
    w = (torch.rand(D, device=dev) + 0.5)
    mod = torch.randn(6, 6, D, device=dev) * 0.3
    idx = torch.randint(0, 6, (S,), device=dev, dtype=torch.int32)
    # norm_mod (tfvideo): (x * rsqrt(mean(x^2) + eps) * w) * (1 + scale) + shift, fp32, one rounding
    for part in (0, 1):
        y = torch.empty_like(x)
        m.k_norm_mod(x, w, mod, idx, y, part, 1e-5)
        xf = x.float()
        n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        prod, shift = n * (1 + mod[idx.long(), 3 * part + 1]), mod[idx.long(), 3 * part]
        # within one bf16 ulp of the terms' magnitude (the sum of squares' order moves n by an fp32 ulp; near-zero
        # results of prod + shift cancellation make ulp counts of the result meaningless)
        d = (y.float() - (prod + shift)).abs()
        ok = bool((d <= (prod.abs() + shift.abs()) * 2.0 ** -8).all())
        RES[f"norm_mod/{part}"] = {"ok": ok, "max_abs": float(d.max()), "identical": float((y == (prod + shift).bfloat16()).float().mean())}
    yv = (torch.randn(S, D, device=dev)).bfloat16()
    xg = x.clone()
    m.k_gate_add_(xg, yv, mod, idx, 1)
    check("gate_add", xg, (x.float() + yv.float() * mod[idx.long(), 5]).bfloat16())
    gu = (torch.randn(S, 2 * 14336, device=dev) * 3).bfloat16()
    g, u = gu.float()[:, :14336], gu.float()[:, 14336:]
    check("swiglu", m.k_swiglu(gu), (g / (1 + torch.exp(-g)) * u).bfloat16())
    check("silu_mul_split", m.k_silu_mul_split(gu), (F.silu(gu[:, :14336]) * gu[:, 14336:]))
    wb = (torch.rand(D, device=dev) + 0.5).bfloat16()
    check("rms_norm", m.k_rms_norm(x, wb, 1e-5), F.rms_norm(x, (D,), wb, 1e-5))
    xh = x.view(-1, 128)
    check("rms_norm/head", m.k_rms_norm(xh, wb[:128].contiguous(), 1e-5), F.rms_norm(xh, (128,), wb[:128], 1e-5))
    sc, sh = torch.randn(D, device=dev) * 0.2, torch.randn(D, device=dev) * 0.2
    fm = m.k_final_mod(x, wb, sc, sh, 1e-5)
    ref = F.rms_norm(x, (D,), wb, 1e-5) * (1.0 + sc) + sh
    RES["final_mod"] = {"ok": float((fm - ref).abs().max()) < 1e-2 * float(ref.abs().max()),
                        "max_abs": float((fm - ref).abs().max())}
    # data moves vs ComfyUI's own functions
    sys.path.insert(0, os.environ["COMFY"])
    from comfy.ldm.minimax.model import pack_audio, patchify_video, unpack_audio, unpatchify_video

    vid = torch.randn(1, 24, 17, 28, 48, device=dev).bfloat16()
    rows = m.k_patchify(vid)
    check("patchify", rows, patchify_video(vid.float()))
    vr = torch.randn(17 * 14 * 24, 96, device=dev)
    check("unpatchify_neg", m.k_unpatchify_neg(vr, 24, 17, 28, 48), -unpatchify_video(vr, 17, 14, 24, 24).bfloat16(), 0)
    au = torch.randn(1, 32, 2, 93, device=dev).bfloat16()
    check("pack_audio", m.k_pack_audio(au), pack_audio(au.float()))
    ar = torch.randn(186, 32, device=dev)
    check("unpack_audio_neg", m.k_unpack_audio_neg(ar, 32, 93), -unpack_audio(ar).bfloat16(), 0)
    check("scale", m.k_scale(au, 0.75), au * torch.tensor(0.75, dtype=torch.bfloat16), 0)
    vv = torch.randn_like(au.float()).bfloat16()
    got = vv.clone()
    m.k_uncarry_(au, got, -3.0, 1.5)
    check("uncarry", got, (-3.0) * au + torch.tensor(1.5, dtype=torch.bfloat16) * vv, 0)
    # fp32 GEMM: within the fp32 accumulation bound of the fp64 product; the patch / head / adaln shapes
    for name, (M, N, K) in {"video_patch": (8000, 5376, 96), "audio_patch": (186, 5376, 32), "video_out": (8000, 96, 5376),
                            "adaln": (2, 96768, 8), "ragged": (77, 130, 41)}.items():
        a, b, bias = torch.randn(M, K, device=dev), torch.randn(N, K, device=dev) / K ** 0.5, torch.randn(N, device=dev)
        c = m.k_gemm_f32(a, b, bias)
        ref = a.double() @ b.double().T + bias.double()
        bound = (a.double().abs() @ b.double().abs().T + bias.double().abs()) * (K + 1) * 2.0 ** -24
        ok = bool(((c.double() - ref).abs() <= bound).all())
        RES[f"gemm_f32/{name}"] = {"ok": ok, "max_abs": float((c.double() - ref).abs().max())}
        det(f"gemm_f32/{name}", lambda a=a, b=b, bias=bias: m.k_gemm_f32(a, b, bias))
    det("norm_mod/det", lambda: (lambda y: (m.k_norm_mod(x, w, mod, idx, y, 0, 1e-5), y)[1])(torch.empty_like(x)))
    ok = all(r.get("ok", True) and r.get("deterministic", True) for r in RES.values())
    print(json.dumps({"ok": ok, "checks": len(RES), "failed": [k for k, r in RES.items()
                                                             if not (r.get("ok", True) and r.get("deterministic", True))]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
