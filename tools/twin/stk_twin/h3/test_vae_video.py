"""python -m stk_twin.h3.test_vae_video: the video VAE twin (stk_twin.h3.vae_video; kernels/cuda/minimax/gemm_f16.cu and
vae_video.cu) on a GPU. One JSON line {"pass": ...} at the end.

  1. kernel checks: the fp16 GEMM (fp64 bound, row independence, batching, epilogues, determinism), denorm / swiglu /
     blend arithmetic against torch (bit-exact where torch's op order is known), RMSNorm / LayerNorm / softmax against torch
     within 1 half ulp, the RoPE rotation against the comfy-kitchen wheel (bit-exact; which contraction variant matches is
     reported) and the rotation table against ComfyUI's RotaryEmbeddingND (bit-exact).
  2. structure: ComfyUI's own tiled_decode / decode_temporal / _finalize_pixels around a fake tile decoder (a deterministic
     function of the tile's latents) against ours around the same fake: bit-exact fp32 pixels and uint8 frames, over several
     T and spatial sizes (the clip plan, the tile table, the blends, the temporal overlap, the padding at the end).
  3. end to end: ComfyUI's VAE.decode of the real checkpoint on the same latents against ours (PSNR and max abs difference of
     the uint8 frames), and determinism (two decodes equal).

ComfyUI: sys.path gets $COMFY (the 0.37.0 tree); the checkpoint is $VMODELS/vae/minimax_h3_video_vae_fp16.safetensors
(or $VVAE). Checks that need ComfyUI or the checkpoint are skipped (and listed) when they are not there.
Environment: VVAE_T, VVAE_H, VVAE_W (the end-to-end latent, default 7, 28, 48), VVAE_MIN_PSNR (default 40), VVAE_MAX_ABS
(default 32): the thresholds of check 3 are guesses until measured on the pod.
"""

from __future__ import annotations

import json
import os
import sys
import types

import torch
import torch.nn.functional as F

from . import vae_video as V

RES: dict[str, dict] = {}
SKIPPED: list[str] = []
DEV = "cuda"


def ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def line(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (line(a) - line(b)).abs()


def record(name: str, ok: bool, **info) -> None:
    RES[name] = {"ok": bool(ok), **info}
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)


def check_half(name: str, got: torch.Tensor, ref: torch.Tensor, max_ulp: int = 0) -> None:
    u = ulps(got, ref)
    record(name, int(u.max()) <= max_ulp, max_ulp=int(u.max()), identical=float((u == 0).float().mean()))


def lin(a, w, bias=None, res=None, scale=None):
    M, K = a.shape
    N = w.shape[0]
    out = torch.empty(M, N, dtype=torch.float16, device=a.device)
    V.mod().k_gemm(a, w, bias, out, res, scale, M, N, K, K, K, N, 0, 0, 0, 0, 0, 0, 1, N if res is not None else 0, 0)
    return out


def rnd(*shape, scale=1.0, seed=None):
    return (torch.randn(*shape, device=DEV) * scale).half()


# ------------------------------------------------------------------------------------------------ 1. kernels
def test_gemm() -> None:
    torch.manual_seed(1)
    for M, N, K in ((300, 6144, 2048), (1797, 2048, 24), (1797, 24, 24), (300, 2048, 8192), (130, 3072, 2048), (17, 100, 40)):
        a, w, b = rnd(M, K, scale=1.0), rnd(N, K, scale=0.05), rnd(N, scale=0.5)
        got = lin(a, w, b)
        ref = a.double() @ w.double().T + b.double()
        absprod = a.abs().double() @ w.abs().double().T
        bound = ref.abs() * 2.0 ** -10 + K * 2.0 ** -24 * absprod + 2.0 ** -24
        err = (got.double() - ref).abs()
        record(f"gemm_{M}x{N}x{K}", bool((err <= bound).all()), max_err=float(err.max()), max_ulp_of_bound=float((err / bound).max()))
    M, N, K = 300, 512, 1024
    a, w, b = rnd(M, K), rnd(N, K, scale=0.05), rnd(N, scale=0.5)
    full = lin(a, w, b)
    part = lin(a[100:164].contiguous(), w, b)
    odd = lin(a[100:107].contiguous(), w[:200].contiguous(), b[:200].contiguous())
    record("gemm_row_independence", torch.equal(full[100:164], part) and torch.equal(full[100:107, :200], odd))
    record("gemm_deterministic", torch.equal(full, lin(a, w, b)))
    # epilogue: res = addcmul(res, y, rscale), y = the Linear's rounded output; in place (out aliases res) as the decoder runs it
    res, sc = rnd(M, N, scale=2.0), rnd(N, scale=0.3)
    y = lin(a, w, b)
    ref = torch.addcmul(res, y, sc)
    out = torch.empty_like(res)
    V.mod().k_gemm(a, w, b, out, res, sc, M, N, K, K, K, N, 0, 0, 0, 0, 0, 0, 1, N, 0)
    inplace = res.clone()
    V.mod().k_gemm(a, w, b, inplace, inplace, sc, M, N, K, K, K, N, 0, 0, 0, 0, 0, 0, 1, N, 0)
    record("gemm_addcmul_epilogue", torch.equal(out, ref) and torch.equal(inplace, ref))
    # the attention's batched windows (A, B windows into one qkv buffer; C a head's columns) = per-head calls
    S, H = 204, 32
    SP = (S + 7) // 8 * 8
    qkv = rnd(S, H * 192)
    sc_all = torch.zeros(H, S, SP, dtype=torch.float16, device=DEV)
    V.mod().k_gemm(qkv, qkv, None, sc_all, None, None, S, S, 64, qkv.stride(0), qkv.stride(0), SP, 0, 64, 0, 192, 192, S * SP, H, 0, 0)
    ok = True
    for h in (0, 5, 31):
        q = qkv[:, h * 192:h * 192 + 64].contiguous()
        k = qkv[:, h * 192 + 64:h * 192 + 128].contiguous()
        ok &= torch.equal(sc_all[h, :, :S], lin(q, k))
    pad_untouched = bool((sc_all[:, :, S:] == 0).all())
    p = torch.softmax(sc_all.float(), -1).half()
    p[:, :, S:] = 0
    vt = V.mod().k_vt(qkv, S, SP, H)
    att = torch.empty(S, 2048, dtype=torch.float16, device=DEV)
    V.mod().k_gemm(p, vt, None, att, None, None, S, 64, SP, SP, SP, 2048, 0, 0, 0, S * SP, 64 * SP, 64, H, 0, 0)
    for h in (0, 5, 31):
        ok &= torch.equal(att[:, h * 64:(h + 1) * 64], lin(p[h].contiguous(), vt[h].contiguous()))
    ref_vt = torch.zeros(H, 64, SP, dtype=torch.float16, device=DEV)
    ref_vt[:, :, :S] = qkv.view(S, H, 192)[:, :, 128:].permute(1, 2, 0)
    record("gemm_batched_windows_and_vt", bool(ok) and pad_untouched and torch.equal(vt, ref_vt))


def test_elementwise() -> None:
    torch.manual_seed(2)
    m = V.mod()
    z = rnd(24, 5, 7, 9, scale=1.5)
    std, mean = (torch.rand(24, device=DEV) * 2 + 0.4).half(), rnd(24)
    ref = z * std.view(24, 1, 1, 1) + mean.view(24, 1, 1, 1)
    check_half("denorm", m.k_denorm(z, std, mean), ref)
    gu = rnd(50, 16384, scale=2.5)
    g, u = gu.chunk(2, -1)
    check_half("swiglu", m.k_swiglu(gu), F.silu(g).mul_(u))
    x = rnd(1797, 2048, scale=3.0)
    w = (torch.rand(2048, device=DEV) + 0.5).half()
    b = rnd(2048, scale=0.2)
    check_half("rms_norm", m.k_rms_norm(x, w, 1e-5), F.rms_norm(x, (2048,), w, 1e-5), max_ulp=1)
    check_half("layer_norm", m.k_layer_norm(x, w, b, 1e-5), F.layer_norm(x, (2048,), w, b, 1e-5), max_ulp=1)
    check_half("layer_norm_vs_fp64", m.k_layer_norm(x, w, b, 1e-5),
               F.layer_norm(x.double(), (2048,), w.double(), b.double(), 1e-5).half(), max_ulp=1)
    nan = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.5, -2.0, 65504.0], device=DEV).half()
    ref = torch.nan_to_num(nan)
    got = nan.clone()
    m.k_nan_to_num_(got)
    record("nan_to_num", torch.equal(got, ref))
    S, SP, rows = 1797, 1800, 96
    s = rnd(rows, SP, scale=20.0)
    s[:, S:] = 7  # junk in the pad: the kernel must write zeros there
    ref = torch.softmax(s[:, :S].float() * 0.125, -1).half()
    got = s.clone()
    m.k_softmax_(got, S, SP, 0.125)
    check_half("softmax", got[:, :S], ref, max_ulp=1)
    record("softmax_pad_zero", bool((got[:, S:] == 0).all()))
    reg = rnd(4, 2048)
    hs = torch.full((9, 2048), 3, dtype=torch.float16, device=DEV)
    m.k_suffix_(hs, reg, 4)
    record("suffix", torch.equal(hs[4:8], reg) and bool((hs[8] == 0).all()) and bool((hs[:4] == 3).all()))
    # the unshuffle is ComfyUI's view / permute / reshape
    T, H, W = 3, 2, 3
    rows = rnd(T * H * W + 5, 3072)
    ref = rows[:T * H * W].view(1, T, H, W, 3, 4, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(3, 4 * T, 16 * H, 16 * W)
    record("unshuffle", torch.equal(m.k_unshuffle(rows, T, H, W), ref))
    # gather: clamp past the last frame
    zz = rnd(24, 4, 6, 7)
    g = m.k_gather_rows(zz, 2, 5, 1, 3, 2, 4)
    idx = torch.arange(2, 7, device=DEV).clamp(max=3)
    ref = zz[:, idx, 1:4, 2:6].permute(1, 2, 3, 0).reshape(-1, 24)
    record("gather_rows", torch.equal(g, ref))


def import_comfy():
    path = os.environ.get("COMFY")
    if not path:
        return None
    sys.path.insert(0, path)
    try:
        import comfy.ldm.minimax.vae as cvae

        return cvae
    except Exception as e:  # noqa: BLE001
        print(f"ComfyUI import failed: {e!r}", flush=True)
        return None


def test_rope(cvae) -> None:
    torch.manual_seed(3)
    m = V.mod()
    T, H, W = 7, 16, 16
    S = T * H * W + 5
    table = m.k_rope_table(V.rope_inv_freq().to(DEV).contiguous(), T, H, W, 5)
    if cvae is not None:
        ids = cvae.create_token_ids((T, H, W), DEV, torch.float16).expand(1, -1, -1)
        ids = torch.cat([ids, torch.zeros(1, 5, 3, device=DEV, dtype=torch.float16)], 1)
        emb = cvae.RotaryEmbeddingND(48, 100.0, 3).to(DEV).half()
        ref = emb(ids).view(S, 24, 4)
        record("rope_table_vs_comfy", torch.equal(table, ref), identical=float((table == ref).float().mean()))
    else:
        SKIPPED.append("rope_table_vs_comfy")
    qkv = rnd(S, 32 * 192, scale=2.0)
    base = qkv.clone()
    m.k_rms_rope_(qkv, table, S, 32, 1e-5, 0)
    # structure against an fp32 emulation in torch (the per-head RMSNorm, rounded to half, then the rotation)
    q = base.view(S, 32, 192)[:, :, :64].float()
    rr = torch.rsqrt(q.pow(2).sum(-1, keepdim=True) / 64 + 1e-5)
    xn = (q * rr).half().float()
    f = table.view(S, 1, 24, 2, 2).float()
    x0, x1 = xn[..., :24], xn[..., 24:48]
    y0 = (f[..., 0, 0] * x0 + f[..., 0, 1] * x1).half()
    y1 = (f[..., 1, 0] * x0 + f[..., 1, 1] * x1).half()
    got = qkv.view(S, 32, 192)
    d = max(int(ulps(got[:, :, :24], y0).max()), int(ulps(got[:, :, 24:48], y1).max()), int(ulps(got[:, :, 48:64], xn[..., 48:].half()).max()))
    record("rope_vs_torch_emulation", d <= 1, max_ulp=d, v_untouched=bool(torch.equal(got[:, :, 128:], base.view(S, 32, 192)[:, :, 128:])))
    if cvae is None:
        SKIPPED.append("rope_vs_kitchen")
        return
    try:
        import comfy.quant_ops as qo

        rope = qo.ck.rms_rope_split_half_
    except Exception as e:  # noqa: BLE001
        print(f"comfy_kitchen not available: {e!r}", flush=True)
        SKIPPED.append("rope_vs_kitchen")
        return
    ones = torch.ones(64, device=DEV, dtype=torch.float16)
    ref = base.clone()
    v = ref.view(1, S, 32, 192)
    qq, kk = v[..., :64], v[..., 64:128]
    rope(qq, kk, table.view(1, S, 1, 24, 2, 2), ones, epsilon=1e-5, rot_dim=48)
    out = {}
    for var in (0, 1, 2):
        t = base.clone()
        m.k_rms_rope_(t, table, S, 32, 1e-5, var)
        out[var] = bool(torch.equal(t, ref))
    record("rope_vs_kitchen_wheel", out[0], identical_by_variant=out,
           note="variant 0 = fma(f00, x0, f01 * x1) is the engine's; if another variant matches, switch vv_rms_rope to it")


class _Fake:
    """A deterministic stand-in for the ViT3D tile decoder: a function of the tile's latents and of the position inside the
    tile (so overlapping tiles differ and the blends do something)."""

    @staticmethod
    def pixels(zt: torch.Tensor) -> torch.Tensor:  # zt [..., 24, T, h, w] half -> [..., 3, 4T, 16h, 16w] half
        base = zt[..., :3, :, :, :].float().tanh() * 1.4
        base = base.repeat_interleave(4, -3).repeat_interleave(16, -2).repeat_interleave(16, -1)
        Tn, Hh, Ww = base.shape[-3:]
        ramp = (torch.arange(Hh, device=zt.device)[:, None] + 2 * torch.arange(Ww, device=zt.device)[None, :]).float() / 700.0
        tr = torch.arange(Tn, device=zt.device).float()[:, None, None] / 90.0
        return (base + ramp - 0.4 + tr).half()


def make_shell(cvae, mean, std):
    M = cvae.MiniMaxH3VideoVAE
    o = M.__new__(M)
    torch.nn.Module.__init__(o)
    o.vae_ratio, o.vae_ratio_t, o.clip_length, o.token_drop = 16, 4, 17, 3
    o.frame_pre_padding = (-17) % 4
    o.tokens_chunk_size = 5
    o.token_overlap = (-3) % 5
    o.frame_overlap = max(o.token_overlap * 4 - o.frame_pre_padding, 0)
    o.tiling, o.tile_size, o.tile_overlap_min = True, 256, 64
    o.register_buffer("latents_mean", mean.clone())
    o.register_buffer("latents_std", std.clone())
    o.register_buffer("pixel_mean", torch.tensor(cvae.IMAGENET_MEAN).view(1, 3, 1, 1, 1).half().to(DEV), persistent=False)
    o.register_buffer("pixel_std", torch.tensor(cvae.IMAGENET_STD).view(1, 3, 1, 1, 1).half().to(DEV), persistent=False)
    o.decoder = types.SimpleNamespace(out_channels=3)
    o._decode_pixels = _Fake.pixels
    return o


def test_structure(cvae) -> None:
    if cvae is None:
        SKIPPED.append("structure")
        return
    torch.manual_seed(4)
    mean = rnd(24, scale=0.5)
    std = (torch.rand(24, device=DEV) * 1.5 + 0.5).half()
    shell = make_shell(cvae, mean, std)
    vv = V.VideoVAE.bare(mean, std, DEV)

    def fake_tile(z, t0, Tn, y0, h, w, x0, name):
        idx = torch.arange(t0, t0 + Tn, device=z.device).clamp(max=z.shape[1] - 1)
        return _Fake.pixels(z[:, idx, y0:y0 + h, x0:x0 + w].contiguous())

    for T, h, w in ((1, 10, 12), (2, 10, 12), (7, 17, 33), (9, 28, 48), (12, 17, 33), (13, 28, 20)):
        z = (torch.randn(1, 24, T, h, w, device=DEV) * 1.2).float()
        with torch.no_grad():
            ref = shell.decode(z.half())  # [1, 3, F, H, W] fp32 in [0, 1]
        reff = ref[0].permute(1, 2, 3, 0).contiguous().to(DEV)
        refu = (reff * 255).clamp(0, 255).byte()
        u8, f32 = vv.decode(z, tile_fn=fake_tile, return_float=True)
        record(f"structure_T{T}_{h}x{w}", u8.shape == refu.shape and torch.equal(f32, reff) and torch.equal(u8, refu),
               frames=int(u8.shape[0]), f32_equal=bool(torch.equal(f32, reff)), u8_equal=bool(torch.equal(u8, refu)),
               tiles=[len(V.split_tiles(h * 16)[0]), len(V.split_tiles(w * 16)[0])])


# ------------------------------------------------------------------------------------------------ 3. end to end
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.double() - b.double()) ** 2).mean())
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item()


def test_end_to_end(cvae) -> None:
    ck = os.environ.get("VVAE") or os.path.join(os.environ.get("VMODELS", ""), "vae", "minimax_h3_video_vae_fp16.safetensors")
    if cvae is None or not os.path.exists(ck):
        SKIPPED.append("end_to_end")
        return
    T, h, w = (int(os.environ.get(k, d)) for k, d in (("VVAE_T", 7), ("VVAE_H", 28), ("VVAE_W", 48)))
    g = torch.Generator().manual_seed(7)
    z = torch.randn(1, 24, T, h, w, generator=g)  # fp32 on the CPU, as VAEDecode hands it to VAE.decode
    import comfy.sd
    import comfy.utils

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(ck))
    with torch.inference_mode():
        ref = vae.decode(z)  # [1, F, H, W, 3] fp32
    refu = (ref[0].to(DEV) * 255).clamp(0, 255).byte()
    del vae
    torch.cuda.empty_cache()
    ours = V.VideoVAE.from_checkpoint(ck, DEV)
    out = ours.decode(z)
    again = ours.decode(z)
    diff = (out.int() - refu.int()).abs()
    p = psnr(out, refu)
    min_psnr, max_abs = float(os.environ.get("VVAE_MIN_PSNR", 40)), int(os.environ.get("VVAE_MAX_ABS", 32))
    record("end_to_end_vs_comfy", out.shape == refu.shape and p >= min_psnr and int(diff.max()) <= max_abs,
           psnr=p, max_abs=int(diff.max()), mean_abs=float(diff.float().mean()), identical=float((diff == 0).float().mean()),
           frames=int(out.shape[0]), latent=[T, h, w], min_psnr=min_psnr, max_abs_allowed=max_abs)
    record("determinism", torch.equal(out, again))


def main() -> int:
    cvae = import_comfy()
    test_gemm()
    test_elementwise()
    test_rope(cvae)
    test_structure(cvae)
    test_end_to_end(cvae)
    ok = all(r["ok"] for r in RES.values())
    print(json.dumps({"pass": ok, "checks": RES, "skipped": SKIPPED}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
