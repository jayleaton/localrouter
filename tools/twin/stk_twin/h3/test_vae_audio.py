"""python -m stk_twin.h3.test_vae_audio: the audio VAE decoder (kernels/cuda/minimax/vae_audio.cu, stk_twin/h3/vae_audio.py).

1. Kernel checks against torch fp64 references at the model's shapes: convolutions within the fp32 accumulation bound
   of the fp64 result (|err| <= (taps + 2) * 2^-24 * sum|w x|), the filters / snake within a few ulps, data moves and
   elementwise adds exact, the std scale against fp64 torch, the torch behaviours our choices rest on (div_ by a
   scalar, true division by a one-element tensor) bit for bit on the pod's torch.
2. End to end against ComfyUI's own decode (MiniMaxH3AudioVAE through comfy.sd.VAE, then vae_decode_audio's std
   normalisation) on the same latents, TF32 DISABLED on the ComfyUI side (torch.backends.cudnn.allow_tf32 = False): the
   like-for-like comparison. SNR and max |diff| of the raw (clamped) decode and of the normalised waveform; both sides
   against a ComfyUI-module run in fp64; one run of ComfyUI with TF32 at torch's default, for the record.
3. Determinism (bitwise equal across repeated decodes).

Needs a GPU, $COMFY (ComfyUI 0.37.0 checkout) and $VMODELS/vae/minimax_h3_audio_vae_fp32.safetensors. Env: STK_AVAE_A
(comma list of latent lengths, default "93,207"), STK_AVAE_LATENT (a .pt [1, 32, 2, A] fp32 real latent, run as well),
STK_AVAE_F64=0 to skip the fp64 reference. Progress on stderr, one JSON line {"pass": ...} on stdout."""

from __future__ import annotations

import json
import math
import os
import sys

import torch
import torch.nn.functional as F

from . import vae_audio as V

RES: dict[str, dict] = {}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def rec(name: str, ok: bool, **kw) -> bool:
    RES[name] = {"ok": bool(ok), **kw}
    log(("PASS " if ok else "FAIL ") + name, kw)
    return bool(ok)


def snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double(), got.double()
    num, den = float((ref * ref).sum()), float(((ref - got) ** 2).sum())
    return 400.0 if den == 0 else (-400.0 if num == 0 else min(400.0, 10 * math.log10(num / den)))


def ulp_err(got: torch.Tensor, ref64: torch.Tensor) -> float:
    """max |got - ref| in ulps of the fp32 value of ref (ulp at max(|ref|, 2^-100))."""
    r32 = ref64.float()
    ulp = torch.nextafter(r32.abs().clamp_min(2.0**-100), torch.tensor(float("inf"), device=r32.device)) - r32.abs().clamp_min(2.0**-100)
    return float(((got.double() - ref64).abs() / ulp.double()).max())


# ------------------------------------------------------------------------------------------------ kernel checks

def conv_bound_ok(name, got, ref, w64, x64, taps, conv):
    """|got - ref| <= (taps + 2) 2^-24 * conv(|w|, |x|) elementwise (the fmaf-chain bound), plus the bias."""
    bound = (taps + 2) * 2.0**-24 * conv(w64.abs(), x64.abs()) + 1e-30
    err = (got.double() - ref).abs()
    return rec(name, bool((err <= bound).all()), max_err=float(err.max()), max_err_over_bound=float((err / bound).max()))


def kernel_checks(dev) -> None:
    torch.manual_seed(0)
    m = V.mod()
    f64 = torch.float64

    # latent_in: z * std + mean, exact (two torch ops, two roundings)
    Bb, C, S, A = 1, 32, 2, 93
    z = torch.randn(Bb, C, S, A, device=dev) * 0.25
    mean, std = torch.randn(C, device=dev) * 0.3, torch.rand(C, device=dev) + 1.0
    ref = (z.permute(0, 2, 1, 3).reshape(Bb * S, C, A) * std.view(1, -1, 1) + mean.view(1, -1, 1)).permute(0, 2, 1).contiguous()
    rec("latent_in", torch.equal(m.k_latent_in(z, mean, std), ref))

    # Conv1d (im2col + gemm), every (K, dilation) of the model plus the 1x1 and the 2048-channel conv_pre shape class
    for (K, d, ci, co, T) in [(1, 1, 32, 70, 93), (7, 1, 300, 64, 93), (3, 1, 40, 36, 211), (3, 5, 40, 36, 211),
                              (7, 3, 33, 21, 100), (11, 5, 50, 70, 130), (7, 1, 8, 1, 1000)]:
        w = torch.randn(co, ci, K, device=dev) / math.sqrt(ci * K)
        b = torch.randn(co, device=dev) if co != 1 else None
        x = torch.randn(2, T, ci, device=dev)
        layer = V.Conv1d("t", w, b, d)
        got = layer(x, "t").permute(0, 2, 1)
        pad = (K * d - d) // 2
        conv = lambda ww, xx: F.conv1d(xx, ww, None, padding=pad, dilation=d)
        ref = conv(w.double(), x.permute(0, 2, 1).double()) + (b.double().view(1, -1, 1) if b is not None else 0)
        conv_bound_ok(f"conv1d_k{K}_d{d}_{ci}x{co}", got, ref, w.double(), x.permute(0, 2, 1).double(), K * ci, conv)
        again = layer(x, "t").permute(0, 2, 1)
        RES[f"conv1d_k{K}_d{d}_{ci}x{co}"]["deterministic"] = bool(torch.equal(got, again))

    # ConvTranspose1d, the model's two shapes (k 9 / stride 5 and k 4 / stride 2)
    for (K, u, ci, co, L) in [(9, 5, 40, 24, 93), (4, 2, 30, 17, 211), (4, 2, 16, 8, 65)]:
        w = torch.randn(ci, co, K, device=dev) / math.sqrt(ci * 2)
        b = torch.randn(co, device=dev)
        x = torch.randn(2, L, ci, device=dev)
        got = V.ConvTranspose1d("t", w, b, u)(x, "t").permute(0, 2, 1)
        pad = (K - u) // 2
        conv = lambda ww, xx: F.conv_transpose1d(xx, ww, None, stride=u, padding=pad)
        ref = conv(w.double(), x.permute(0, 2, 1).double()) + b.double().view(1, -1, 1)
        ok_shape = got.shape == ref.shape
        if ok_shape:
            conv_bound_ok(f"conv_transpose1d_k{K}_s{u}", got, ref, w.double(), x.permute(0, 2, 1).double(), -(-K // u) * ci, conv)
        else:
            rec(f"conv_transpose1d_k{K}_s{u}", False, got=list(got.shape), ref=list(ref.shape))

    # Activation1d parts against the torch formulas (UpSample1d / DownSample1d / SnakeBeta of audio_vae.py) in fp64
    f = torch.randn(12, device=dev) * 0.2
    Cc, T = 24, 150
    x = torch.randn(2, T, Cc, device=dev)
    xc = x.permute(0, 2, 1).double()
    p = F.pad(xc, (5, 5), mode="replicate")
    up_ref = (F.conv_transpose1d(p, f.double().view(1, 1, 12).expand(Cc, -1, -1), stride=2, groups=Cc) * 2)[..., 15:-15]
    up = m.k_up2(x, f)
    err = (up.permute(0, 2, 1).double() - up_ref).abs()
    bound = 7 * 2.0**-24 * 2 * F.conv_transpose1d(F.pad(xc.abs(), (5, 5), mode="replicate"), f.double().abs().view(1, 1, 12).expand(Cc, -1, -1), stride=2, groups=Cc)[..., 15:-15] + 1e-30
    rec("up2", up.shape == (2, 2 * T, Cc) and bool((err <= bound).all()), max_err=float(err.max()))
    x2 = torch.randn(2, 2 * T, Cc, device=dev)
    xc2 = x2.permute(0, 2, 1).double()
    dn_ref = F.conv1d(F.pad(xc2, (5, 6), mode="replicate"), f.double().view(1, 1, 12).expand(Cc, -1, -1), stride=2, groups=Cc)
    dn = m.k_down2(x2, f)
    err = (dn.permute(0, 2, 1).double() - dn_ref).abs()
    bound = 14 * 2.0**-24 * F.conv1d(F.pad(xc2.abs(), (5, 6), mode="replicate"), f.double().abs().view(1, 1, 12).expand(Cc, -1, -1), stride=2, groups=Cc) + 1e-30
    rec("down2", dn.shape == (2, T, Cc) and bool((err <= bound).all()), max_err=float(err.max()))

    # SnakeBeta
    pa, pb = torch.randn(Cc, device=dev) * 0.5, torch.randn(Cc, device=dev) * 0.5
    xs = torch.randn(2, T, Cc, device=dev) * 2
    got = m.k_snake(xs, pa, pb)
    a64, b64 = pa.double().exp(), pb.double().exp()
    ref = xs.double() + torch.sin(a64 * xs.double()) ** 2 / (b64 + 1e-9)
    err = (got.double() - ref).abs()
    tol = 2e-5 * (1 + ref.abs()) / b64.clamp(max=1)
    rec("snake", bool((err <= tol).all()), max_err=float(err.max()), max_ulp=ulp_err(got, ref))
    # ... and bit for bit vs torch's own fp32 ops in the same order (the ComfyUI formula), informational: torch sin / exp
    # are CUDA's sinf / expf as ours
    al, be = torch.exp(pa).view(1, 1, -1), torch.exp(pb).view(1, 1, -1)
    t = torch.sin(al * xs)
    t = t.mul_(t).mul_((be + 1e-9).reciprocal()).add_(xs)
    RES["snake"]["bitwise_vs_torch_fp32"] = bool(torch.equal(got, t))
    log("snake bitwise vs torch fp32:", RES["snake"]["bitwise_vs_torch_fp32"])

    # add (exact) and clamp (exact, NaN passes)
    a, b = torch.randn(1000003, device=dev), torch.randn(1000003, device=dev)
    rec("add", torch.equal(m.k_add(a, b), a + b))
    c = torch.randn(10007, device=dev) * 3
    c[5] = float("nan")
    rec("clamp", torch.equal(m.k_clamp(c).nan_to_num(7.0), c.clamp(-1.0, 1.0).nan_to_num(7.0)) and bool(torch.isnan(m.k_clamp(c)[5])))

    # avg3: (r0 + r1) + r2, then torch's xs.div_(3): which of {multiply by fp32 reciprocal, true division} torch CUDA does
    r = [torch.randn(2, 4097, 16, device=dev) * 2 for _ in range(3)]
    xs = r[0].clone()
    xs += r[1]
    xs += r[2]
    xs = xs.div_(3)
    got_recip, got_div = m.k_avg3(*r, 1), m.k_avg3(*r, 0)
    want = V.AVG3_RECIP
    rec("avg3", torch.equal(got_recip if want else got_div, xs), AVG3_RECIP=want, torch_matches_recip=bool(torch.equal(got_recip, xs)),
        torch_matches_div=bool(torch.equal(got_div, xs)))

    # std scale: sc = max-floor(std(unbiased) * 5, 1) per row, against fp64 torch (correctly rounded std) and against
    # torch's own fp32 on the device and on the CPU (information)
    for name, sd in [("quiet", 0.05), ("loud", 0.4), ("near1", 0.2)]:
        aud = (torch.randn(2, 2 * 40000, device=dev) * sd).clamp(-1, 1)
        sc = m.k_std_scale(aud)
        s64 = aud.double().std(dim=1, unbiased=True)
        want = (s64.float() * 5.0)
        want = torch.where(want < 1.0, torch.ones_like(want), want)
        torch_gpu = (aud.std(dim=1) * 5.0)
        torch_gpu = torch.where(torch_gpu < 1.0, torch.ones_like(torch_gpu), torch_gpu)
        torch_cpu = (aud.cpu().std(dim=1) * 5.0)
        torch_cpu = torch.where(torch_cpu < 1.0, torch.ones_like(torch_cpu), torch_cpu)
        d = float((sc.double() - want.double()).abs().max())
        rec(f"std_scale_{name}", d <= float(want.max()) * 2.0**-23 and bool(torch.equal(sc, m.k_std_scale(aud))), sc=sc.tolist(),
            bitwise_vs_fp64_rounded=bool(torch.equal(sc, want)), bitwise_vs_torch_gpu=bool(torch.equal(sc, torch_gpu)),
            bitwise_vs_torch_cpu=bool(torch.equal(sc.cpu(), torch_cpu)))
    # audio /= std: true division (torch GPU tensor / one-element GPU tensor, and the CPU one)
    aud = torch.randn(1, 2 * 12345, device=dev)
    scv = torch.tensor([1.7320508], device=dev)
    got = m.k_div_scale(aud, scv)
    rec("div_scale", torch.equal(got, aud / scv.view(1, 1)) and torch.equal(got.cpu(), aud.cpu() / scv.cpu().view(1, 1)),
        bitwise_vs_torch_gpu=bool(torch.equal(got, aud / scv.view(1, 1))), bitwise_vs_torch_cpu=bool(torch.equal(got.cpu(), aud.cpu() / scv.cpu().view(1, 1))))


# ------------------------------------------------------------------------------------------------ end to end

def comfy_setup(ckpt: str):
    sys.path.insert(0, os.environ["COMFY"])
    argv, sys.argv = sys.argv, sys.argv[:1]
    import comfy.sd
    import comfy.utils

    sys.argv = argv
    sd = comfy.utils.load_torch_file(ckpt)
    return comfy.sd.VAE(sd=sd), sd


def comfy_decode(vae, lat):
    """ComfyUI's own: vae.decode (the clamped raw waveform) and vae_decode_audio (the normalised one)."""
    from comfy_extras.nodes_audio import vae_decode_audio

    with torch.no_grad():
        raw = vae.decode(lat.clone()).movedim(-1, 1).float().cpu()
        post = vae_decode_audio(vae, {"samples": lat.clone()})["waveform"].float().cpu()
    return raw, post


def e2e(ckpt: str, dev, latents: dict[str, torch.Tensor], do_f64: bool) -> bool:
    ok_all = True
    ours = V.AudioVAE.from_checkpoint(ckpt)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    vae, sd = comfy_setup(ckpt)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # the stored filters vs the filters ComfyUI would compute (information: ComfyUI loads the stored ones)
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE, kaiser_sinc_filter1d

    up_f, dn_f = kaiser_sinc_filter1d(0.25, 0.3, 12), kaiser_sinc_filter1d(0.25, 0.3, 12)
    st = [sd["decoder.resblocks.0.activations.0.upsample.filter"].float().cpu(), sd["decoder.resblocks.0.activations.0.downsample.lowpass.filter"].float().cpu()]
    rec("stored_filters", True, max_diff_up=float((st[0] - up_f).abs().max()), max_diff_down=float((st[1] - dn_f).abs().max()))

    m64 = None
    if do_f64:
        try:
            m64 = MiniMaxH3AudioVAE()
            m64.load_state_dict(sd, strict=True)
            m64 = m64.double().to(dev).eval()
        except Exception as e:  # noqa: BLE001
            log("fp64 reference unavailable:", repr(e)[:200])
            m64 = None

    for name, lat in latents.items():
        lat = lat.to(dev)
        A = lat.shape[-1]
        got_raw = ours.decode_raw(lat).float().cpu()
        got_post = ours.decode(lat).float().cpu()
        cr, cp = comfy_decode(vae, lat)
        info = {"A": A, "shape_ok": tuple(got_post.shape) == (1, 2, A * 800) == tuple(cp.shape),
                "snr_raw_db": snr_db(cr, got_raw), "max_abs_raw": float((cr - got_raw).abs().max()),
                "snr_post_db": snr_db(cp, got_post), "max_abs_post": float((cp - got_post).abs().max()),
                "clamped_frac": float((got_raw.abs() >= 1.0).float().mean()), "post_scale": float(got_post.abs().max() / max(float(got_raw.abs().max()), 1e-30))}
        ok = info["shape_ok"] and info["snr_raw_db"] >= 60 and info["snr_post_db"] >= 60
        if m64 is not None:
            with torch.no_grad():
                r64 = m64.decode(lat.double()).float().cpu()
            info["snr_comfy_vs_f64_db"], info["snr_ours_vs_f64_db"] = snr_db(r64, cr), snr_db(r64, got_raw)
            info["max_abs_comfy_vs_f64"], info["max_abs_ours_vs_f64"] = float((r64 - cr).abs().max()), float((r64 - got_raw).abs().max())
            # ours must be about as close to the truth as ComfyUI's fp32 (3x its error, i.e. -9.5 dB)
            ok = ok and info["snr_ours_vs_f64_db"] >= info["snr_comfy_vs_f64_db"] - 9.6
        # determinism
        again = ours.decode(lat).float().cpu()
        info["deterministic"] = bool(torch.equal(again, got_post))
        ok = ok and info["deterministic"]
        rec(f"e2e_{name}", ok, **info)
        ok_all &= ok

    # ComfyUI at torch's default (TF32 allowed in cuDNN convs), for the record: how far TF32 moves its own output
    name, lat = next(iter(latents.items()))
    lat = lat.to(dev)
    torch.backends.cudnn.allow_tf32 = True
    cr_tf32, cp_tf32 = comfy_decode(vae, lat)
    torch.backends.cudnn.allow_tf32 = False
    cr, cp = comfy_decode(vae, lat)
    got_raw = ours.decode_raw(lat).float().cpu()
    RES["comfy_tf32_default"] = {"ok": True, "snr_comfy_tf32_vs_comfy_fp32_db": snr_db(cr, cr_tf32),
                                 "snr_ours_vs_comfy_tf32_db": snr_db(cr_tf32, got_raw), "snr_ours_vs_comfy_fp32_db": snr_db(cr, got_raw)}
    log("comfy tf32 default:", RES["comfy_tf32_default"])
    return ok_all


def main() -> int:
    dev = "cuda"
    ckpt = os.path.join(os.environ["VMODELS"], "vae", "minimax_h3_audio_vae_fp32.safetensors")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel_checks(dev)

    torch.manual_seed(1)
    latents = {f"A{a}": torch.randn(1, 32, 2, a) * 0.25 for a in (int(v) for v in os.environ.get("STK_AVAE_A", "93,207").split(","))}
    if os.environ.get("STK_AVAE_LATENT"):
        latents["real"] = torch.load(os.environ["STK_AVAE_LATENT"], weights_only=True).float().reshape(1, 32, 2, -1)
    ok_e2e = e2e(ckpt, dev, latents, os.environ.get("STK_AVAE_F64", "1") != "0")

    ok = ok_e2e and all(v["ok"] for v in RES.values()) and all(v.get("deterministic", True) for v in RES.values())
    print(json.dumps({"pass": bool(ok), "checks": RES}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
