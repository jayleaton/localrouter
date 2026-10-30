"""python -m stk_twin.kandinsky6_dit.test_ops [--out k6_dit_receipt.json]: the Kandinsky-6 DiT twin (kernels.py on the
engine's fatbins, block.py) on one CUDA GPU, random bf16 weights and inputs at the real widths (--small: narrower) and
reduced token counts with ragged tails: 3 x 14 x 22 latents = 231 visual tokens (not a multiple of 64 or 128), 218
audio rows, 96 text tokens of which 77 are valid; fewer blocks (--blocks, --text-blocks).

Asserted, bit for bit (the exit status):
  * the twin run twice, scratch filled 0x00 then 0xFF, gives the same captures and outputs (text block, fused block,
    reduced forward), and the FFN in 50-row chunks the same as in one;
  * a GEMM in row chunks equals the whole call; the element-wise staging (vec = 0: an odd row stride, a misaligned
    weight) equals the cp.async staging (vec = 1) on the same data;
  * attention rows do not depend on the other queries (a query subset equals those rows of the whole call), and masked
    keys are exactly invisible (the masked call equals the call on the valid keys alone);
  * patchify, unpatchify + grid split and the audio grid split equal upstream's own view / permute / movedim
    (VisualEmbeddings, OutLayer, PiFlow); cast, the bf16 add, gate_sum and the RoPE rotation equal the upstream torch
    expressions (tensors.py), which round at the same points;
  * the norms against torch on this GPU: norm_mod vs apply_scale_shift_norm, LayerNorm plain and affine, RMSNorm
    (ops.cu ports torch 2.14's vectorized_layer_norm_kernel / _fused_rms_norm order and expressions).
Reported (the receipt; each stays UNQUALIFIED until it is bit-equal): the GEMV in every launch.Fp32Linear form and the
sinusoid against their torch statements, and every op, one text block, one fused block and the reduced forward
against upstream kandinsky-6 eac06e39 (native.py, SDPA everywhere, the oracle's own "high" matmul policy): max |diff|,
max relative diff, unequal count, cosine, per capture.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import torch
import torch.nn.functional as F
from torch import nn

from . import block as B
from . import budget
from . import kernels as K
from . import native as N

BF = torch.bfloat16
DEV = "cuda"
R: dict = {"exact": {}, "reported": {}, "faithfulness": {}}


# ---------------------------------------------------------------------------------------------------------- records
def exact(name: str, got: torch.Tensor, ref: torch.Tensor, **extra) -> bool:
    same = tuple(got.shape) == tuple(ref.shape) and got.dtype == ref.dtype
    ok = bool(same and torch.equal(got, ref))
    unequal = int((got != ref).sum()) if same else -1
    R["exact"][name] = {"ok": ok, "unequal": unequal, "numel": got.numel(), **extra}
    print(f"{'PASS' if ok else 'FAIL'} {name} unequal={unequal}", flush=True)
    return ok


def exact_caps(name: str, a: dict, b: dict) -> bool:
    differ = [n for n in a if n not in b or not torch.equal(a[n], b[n])] + [n for n in b if n not in a]
    ok = list(a) == list(b) and not differ
    R["exact"][name] = {"ok": ok, "captures": len(a), "differ": differ[:20]}
    print(f"{'PASS' if ok else 'FAIL'} {name} captures={len(a)} differ={len(differ)}", flush=True)
    return ok


def metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.detach().double().reshape(-1), ref.detach().double().reshape(-1)
    if g.numel() != r.numel():
        return {"shape_mismatch": [list(got.shape), list(ref.shape)]}
    d = (g - r).abs()
    nz = r != 0
    den = float(g.norm() * r.norm())
    same_dt = got.dtype == ref.dtype
    return {
        "max_abs": float(d.max()) if d.numel() else 0.0,
        "max_rel": float((d[nz] / r[nz].abs()).max()) if bool(nz.any()) else 0.0,
        "unequal": int((got.reshape(-1) != ref.reshape(-1)).sum()) if same_dt else int((g != r).sum()),
        "numel": int(g.numel()),
        "cosine": float((g * r).sum()) / den if den > 0 else 1.0,
        "nonfinite": int((~torch.isfinite(g)).sum()) + int((~torch.isfinite(r)).sum()),
    }


def report(section: str, name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    R[section][name] = m = metrics(got, ref)
    print(f"INFO {section} {name} {json.dumps(m)}", flush=True)


def report_caps(section: str, twin: dict, native: dict) -> None:
    sec = R["faithfulness"].setdefault(section, {})
    for n, t in twin.items():
        if n in native:
            sec[n] = metrics(t, native[n])
    sec["_missing_native"] = [n for n in twin if n not in native]
    worst = min((v["cosine"] for k, v in sec.items() if isinstance(v, dict) and "cosine" in v), default=1.0)
    print(f"INFO faithfulness {section} captures={len(twin)} matched={len(twin) - len(sec['_missing_native'])} "
          f"min_cosine={worst:.6f}", flush=True)


def recorder() -> tuple[dict, B.Hook]:
    caps: dict = {}

    def hook(name: str, t: torch.Tensor) -> None:
        caps[name] = t.clone()

    return caps, hook


# ---------------------------------------------------------------------------------------------------------- weights
def random_weights(c: B.Config, seed: int) -> dict:
    """Every tensor of the manifest, random at unit-ish activation scale, then bf16 (the file's F32 ones as upstream
    casts them)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = {}
    for name, _dt, shape in B.manifest(c):
        t = torch.randn(shape, generator=g, device=DEV, dtype=torch.float32)
        if name.endswith("norm.weight"):
            t = 1.0 + 0.1 * t
        elif name.endswith("norm.bias") or name == "visual_token_type_embeddings.weight":
            t = 0.1 * t
        elif name.endswith(".bias"):
            t = 0.05 * t
        elif "modulation" in name:
            t = t * (0.5 * shape[1] ** -0.5)
        else:
            t = t * shape[-1] ** -0.5
        w[name] = t.to(BF)
    return w


# -------------------------------------------------------------------------------------------------------------- ops
def ops(k: K.Kernels, u, tc: dict, c: B.Config, sh: B.Shape, W: dict, valid: int, nb: nn.Module, mods: dict) -> None:
    g = torch.Generator(device=DEV).manual_seed(1)

    def rnd(*shape, scale=1.0, dtype=BF):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)

    D, Da, T = c.model_dim, c.model_dim_a, c.time_dim
    H, Ha = c.heads(), c.heads_a()
    N_, A, L = sh.tokens(), sh.audio, sh.text
    sa = "visual_transformer_blocks.0.video_dec_block.self_attention"
    ffp = "visual_transformer_blocks.0.video_dec_block.feed_forward.net"
    rp = N.ropes(mods, c, sh)
    vr, ar, tr = N.rope_rows(rp["visual"]), N.rope_rows(rp["audio"]), N.rope_rows(rp["text"][0])
    cuts = [r for r in (77, 200) if 0 < r < N_]

    def chunks(n):
        e = [0] + [r for r in cuts if r < n] + [n]
        return list(zip(e[:-1], e[1:]))

    # ---- GEMM: rows independent, staging independent
    x = rnd(N_, D)
    wq, bq = W[f"{sa}.to_query.weight"], W[f"{sa}.to_query.bias"]
    up, down = W[f"{ffp}.0.proj.weight"], W[f"{ffp}.2.weight"]
    whole = torch.empty(N_, D, dtype=BF, device=DEV)
    k.gemm(x, wq, bq, whole, K.EPI_BIAS)
    part = torch.empty_like(whole)
    for r0, r1 in chunks(N_):
        k.gemm(x[r0:r1], wq, bq, part[r0:r1], K.EPI_BIAS)
    exact("gemm/rows_chunked/bias", part, whole)
    gw = torch.empty(N_, up.shape[0], dtype=BF, device=DEV)
    k.gemm(x, up, None, gw, K.EPI_GELU)
    gp = torch.empty_like(gw)
    for r0, r1 in chunks(N_):
        k.gemm(x[r0:r1], up, None, gp[r0:r1], K.EPI_GELU)
    exact("gemm/rows_chunked/gelu", gp, gw)
    xp = rnd(N_, c.patch_in())  # K = 132: the element-wise staging only
    pw, pb = W["visual_embeddings.in_layer.weight"], W["visual_embeddings.in_layer.bias"]
    pe = torch.empty(N_, D, dtype=BF, device=DEV)
    k.gemm(xp, pw, pb, pe, K.EPI_BIAS)
    pp = torch.empty_like(pe)
    for r0, r1 in chunks(N_):
        k.gemm(xp[r0:r1], pw, pb, pp[r0:r1], K.EPI_BIAS)
    exact("gemm/rows_chunked/patch_k132", pp, pe, vec=K.Kernels.gemm_vec(xp, pw))
    # k_first: the native patch kernel's order (k < 4 first, then 4 + 32 j) is a plain pass over K padded as
    # [x[:, :4], 28 zeros, x[:, 4:]] (the same mma chunks, plus one all-zero chunk that adds exact zeros)
    kf = c.patch_k_first()
    pk = torch.empty_like(pe)
    k.gemm(xp, pw, pb, pk, K.EPI_BIAS, kf)
    pkc = torch.empty_like(pe)
    for r0, r1 in chunks(N_):
        k.gemm(xp[r0:r1], pw, pb, pkc[r0:r1], K.EPI_BIAS, kf)
    exact("gemm/rows_chunked/patch_k132_k_first", pkc, pk)

    def padded(t):
        z = torch.zeros(t.shape[0], 32 - kf, dtype=t.dtype, device=DEV)
        return torch.cat([t[:, :kf], z, t[:, kf:]], 1).contiguous()

    pr = torch.empty_like(pe)
    k.gemm(padded(xp), padded(pw), pb, pr, K.EPI_BIAS)
    exact("gemm/k_first_equals_padded_reorder/patch_k132", pk, pr)
    pad = torch.zeros(N_, D + 1, dtype=BF, device=DEV)
    pad[:, :D] = x
    xu = pad[:, :D]  # row stride D + 1: no cp.async
    flat = torch.empty(wq.numel() + 8, dtype=BF, device=DEV)
    wu = flat[1:1 + wq.numel()].view(wq.shape)  # 2 bytes off 16: no cp.async
    wu.copy_(wq)
    vecs = {"aligned": K.Kernels.gemm_vec(x, wq), "odd_ld": K.Kernels.gemm_vec(xu, wq),
            "misaligned_w": K.Kernels.gemm_vec(x, wu)}
    o = torch.empty_like(whole)
    k.gemm(xu, wq, bq, o, K.EPI_BIAS)
    exact("gemm/vec0_equals_vec1/odd_ld/bias", o, whole, vec=vecs)
    o = torch.empty_like(whole)
    k.gemm(x, wu, bq, o, K.EPI_BIAS)
    exact("gemm/vec0_equals_vec1/misaligned_w/bias", o, whole, vec=vecs)
    o = torch.empty_like(gw)
    k.gemm(xu, up, None, o, K.EPI_GELU)
    exact("gemm/vec0_equals_vec1/odd_ld/gelu", o, gw, vec=vecs)
    o1, o0 = torch.empty_like(whole), torch.empty_like(whole)
    k.gemm(x, wq, bq, o1, K.EPI_BIAS, 8)    # cp.async staging from k 8
    k.gemm(xu, wq, bq, o0, K.EPI_BIAS, 8)   # element-wise staging
    exact("gemm/vec0_equals_vec1/k_first8/bias", o0, o1,
          vec={"aligned": K.Kernels.gemm_vec(x, wq, 8), "odd_ld": K.Kernels.gemm_vec(xu, wq, 8)})
    report("faithfulness", "op/linear_bias (nn.Linear)", whole, nb.video_dec_block.self_attention.to_query(x))
    report("faithfulness", "op/linear_gelu (_GELUProjection)", gw, nb.video_dec_block.feed_forward.net[0](x))
    hh = torch.empty_like(gw)
    ffo = torch.empty(N_, D, dtype=BF, device=DEV)
    k.gemm(x, up, None, hh, K.EPI_GELU)
    k.gemm(hh, down, None, ffo, K.EPI_BIAS)
    report("faithfulness", "op/feed_forward (FeedForward)", ffo, nb.video_dec_block.feed_forward(x))
    report("faithfulness", "op/patch_linear_k132 (nn.Linear)", pk, F.linear(xp, pw, pb))

    # ---- RMSNorm + RoPE: the rotation exact, the norm reported
    q = rnd(N_, D, scale=3.0)
    qn = W[f"{sa}.query_norm.weight"].view(1, 128)
    norm_only = q.clone()
    k.qk_norm_rope(norm_only, qn, None, H)
    roped = q.clone()
    k.qk_norm_rope(roped, qn, vr, H)
    ref = u.tensors.apply_rotary(norm_only.view(N_, H, 128), N.rope_native(vr)).type_as(norm_only).reshape(N_, D)
    exact("qk_norm_rope/rotation_visual (apply_rotary)", roped, ref)
    qa = rnd(A, Da, scale=3.0)
    qa_n = qa.clone()
    k.qk_norm_rope(qa_n, qn, None, Ha)
    qa_r = qa.clone()
    k.qk_norm_rope(qa_r, qn, ar, Ha)
    ref = u.tensors.apply_rotary(qa_n.view(A, Ha, 128), N.rope_native(ar)).type_as(qa_n).reshape(A, Da)
    exact("qk_norm_rope/rotation_audio (apply_rotary)", qa_r, ref)
    rms = nb.video_dec_block.self_attention.query_norm
    report("faithfulness", "op/rms_norm (nn.RMSNorm(128), eps None)", norm_only, rms(q.view(N_, H, 128)).reshape(N_, D))
    native_qr = u.tensors.apply_rotary(rms(q.view(N_, H, 128)), N.rope_native(vr)).type_as(q).reshape(N_, D)
    report("faithfulness", "op/qk_norm_rope (RMSNorm + apply_rotary)", roped, native_qr)

    # ---- attention
    kx, vx = rnd(N_, D, scale=3.0), rnd(N_, D)
    kn = W[f"{sa}.key_norm.weight"].view(1, 128)
    k.qk_norm_rope(kx, kn, vr, H)
    full = torch.empty(N_, D, dtype=BF, device=DEV)
    k.attend(roped, kx, vx, full, H)
    r0, r1 = min(37, N_ - 1), min(150, N_)
    s = torch.empty(r1 - r0, D, dtype=BF, device=DEV)
    k.attend(roped[r0:r1], kx, vx, s, H)
    exact("attention/rows_independent/self", s, full[r0:r1])
    mask = K.Mask.of(torch.arange(L, device=DEV) < valid)
    tk, tv = rnd(L, D, scale=3.0), rnd(L, D)
    tk[valid:] = 1.0e4  # masked keys carry extreme values: they must not matter at all
    tv[valid:] = -1.0e4
    mfull = torch.empty(N_, D, dtype=BF, device=DEV)
    k.attend(roped, tk, tv, mfull, H, mask)
    ms = torch.empty(r1 - r0, D, dtype=BF, device=DEV)
    k.attend(roped[r0:r1], tk, tv, ms, H, mask)
    exact("attention/rows_independent/masked", ms, mfull[r0:r1])
    trunc = torch.empty_like(mfull)
    k.attend(roped, tk[:valid], tv[:valid], trunc, H)
    exact("attention/masked_equals_truncated", mfull, trunc)
    ak, av_ = rnd(A, D, scale=3.0), rnd(A, D)
    cx = torch.empty(N_, D, dtype=BF, device=DEV)
    k.attend(roped, ak, av_, cx, H)
    cs = torch.empty(r1 - r0, D, dtype=BF, device=DEV)
    k.attend(roped[r0:r1], ak, av_, cs, H)
    exact("attention/rows_independent/cross_v_from_a", cs, cx[r0:r1])
    vk, vv = rnd(N_, Da, scale=3.0), rnd(N_, Da)
    ax = torch.empty(A, Da, dtype=BF, device=DEV)
    k.attend(qa_r, vk, vv, ax, Ha)
    a0, a1 = 5, min(150, A)
    as_ = torch.empty(a1 - a0, Da, dtype=BF, device=DEV)
    k.attend(qa_r[a0:a1], vk, vv, as_, Ha)
    exact("attention/rows_independent/cross_a_from_v", as_, ax[a0:a1])

    def sdpa(qq, kk, vv, h, m=None):
        args = (qq.view(1, -1, h, 128), kk.view(1, -1, h, 128), vv.view(1, -1, h, 128))
        o = u.sdpa.sdpa(*args) if m is None else u.sdpa.sdpa(*args, attn_mask=N.mask_native(m))
        return o.reshape(-1, h * 128)

    report("faithfulness", "op/attention_self (sdpa)", full, sdpa(roped, kx, vx, H))
    report("faithfulness", "op/attention_masked (sdpa, bool mask)", mfull, sdpa(roped, tk, tv, H, mask))
    report("faithfulness", "op/attention_v_from_a (sdpa)", cx, sdpa(roped, ak, av_, H))
    report("faithfulness", "op/attention_a_from_v (sdpa)", ax, sdpa(qa_r, vk, vv, Ha))

    # ---- element-wise: exact against the upstream expressions
    vm = rnd(1, 9 * D, scale=0.5)
    gate = B.cols(vm, B.Mod.dec(D, 0, 2), D)
    res = rnd(N_, D)
    wide = rnd(N_, D + 64)
    out = wide[:, :D]  # out with a row stride
    xg = res.clone()
    k.gate_sum(xg, out, gate)
    exact("gate_sum (apply_gate_sum)", xg, u.tensors.apply_gate_sum(res, out, gate))
    xf = torch.randn(A, c.in_audio_dim, generator=g, device=DEV)
    xf[0, :4] = torch.tensor([1.0 + 2.0**-8, 1.0 + 3 * 2.0**-8, -(1.0 + 2.0**-8), 65504.0], device=DEV)  # ties
    yb = torch.empty(A, c.in_audio_dim, dtype=BF, device=DEV)
    k.cast_bf16(xf, yb)
    exact("cast_bf16 (x.to(bf16))", yb, xf.to(BF))
    t1, t2 = rnd(1, T), rnd(1, T)
    ys = torch.empty(1, T, dtype=BF, device=DEV)
    k.add_bf16(t1, t2, ys)
    exact("add_bf16 (time_embeddings(t) + pooled_proj)", ys, t1 + t2)

    # ---- layouts: exact against upstream's own modules / expressions
    C = c.visual_in()
    ve = N.visual_embeddings_layout(C)
    for nm, dt in (("f32", torch.float32), ("bf16", BF)):
        x5 = torch.randn(1, sh.t, sh.h, sh.w, C, generator=g, device=DEV).to(dt)
        prow = torch.empty(N_, 4 * C, dtype=BF, device=DEV)
        k.patchify(x5.reshape(-1, C), prow, sh, C)
        exact(f"patchify/{nm} (VisualEmbeddings)", prow, ve(x5).reshape(N_, 4 * C))
    hv = rnd(N_, 4 * c.out_visual_dim)
    vo = torch.empty(c.n_grid * sh.t * sh.h * sh.w, c.dx_visual(), dtype=BF, device=DEV)
    k.unpatchify(hv, vo, sh, c.out_visual_dim, c.n_grid)
    ol = N.out_layer_layout(4 * c.out_visual_dim, c.out_visual_dim)
    nv = ol(hv.view(1, sh.t, sh.h // 2, sh.w // 2, -1), torch.zeros(1, 8, dtype=BF, device=DEV))
    nv = N.grid_split(nv, c.n_grid, c.dx_visual()).contiguous().reshape(-1, c.dx_visual())
    exact("unpatchify + grid split (OutLayer, PiFlow movedim)", vo, nv)
    ha = rnd(A, c.out_audio_dim)
    ao = torch.empty(c.n_grid * A, c.dx_audio(), dtype=BF, device=DEV)
    k.grid_major(ha, ao, c.n_grid)
    na = N.grid_split(ha.view(1, A, -1), c.n_grid, c.dx_audio()).contiguous().reshape(-1, c.dx_audio())
    exact("grid_major (PiFlow movedim, audio)", ao, na)

    # ---- the norms: exact against torch (the ported kernels); the GEMV: reported
    xr = rnd(N_, D, scale=2.0)
    sc, shf = B.cols(vm, B.Mod.dec(D, 0, 1), D), B.cols(vm, B.Mod.dec(D, 0, 0), D)
    y = torch.empty(N_, D, dtype=BF, device=DEV)
    k.norm_mod(xr, sc, shf, y)
    ln = nn.LayerNorm(D, elementwise_affine=False).to(DEV)
    exact("norm_mod vs apply_scale_shift_norm", y, u.tensors.apply_scale_shift_norm(ln, xr, sc, shf))
    xa = rnd(A, Da, scale=2.0)
    ya = torch.empty_like(xa)
    k.layer_norm(xa, None, None, ya)
    exact("layer_norm plain vs nn.LayerNorm (bf16)", ya,
           nn.LayerNorm(Da, elementwise_affine=False).to(DEV)(xa))
    wl, bl = W["video_text_embeddings.norm.weight"].view(1, D), W["video_text_embeddings.norm.bias"].view(1, D)
    yl = torch.empty_like(xr)
    k.layer_norm(xr, wl, bl, yl)
    exact("layer_norm affine vs F.layer_norm (bf16)", yl,
           F.layer_norm(xr, (D,), wl.view(-1), bl.view(-1), 1e-5))
    exact("rms_norm vs F.rms_norm(eps None)", norm_only,
           F.rms_norm(q.view(N_, H, 128), (128,), qn.view(-1)).reshape(N_, D))
    tm = rnd(1, T)
    mw, mb = W["visual_transformer_blocks.0.video_dec_block.visual_modulation.out_layer.weight"], \
        W["visual_transformer_blocks.0.video_dec_block.visual_modulation.out_layer.bias"]
    ref64 = F.silu(tm.double()) @ mw.double().T + mb.double()
    nat_mod = nb.video_dec_block.visual_modulation(tm)  # the oracle at its own "high" policy, never forced
    for math in K.FP32_LINEAR:  # every form reported; none qualified until SM121 native byte fixtures pass
        mo = torch.empty(1, 9 * D, dtype=BF, device=DEV)
        k.gemv(tm, mw, mb, mo, True, math)
        report("reported", f"gemv modulation/{math} vs fp64", mo, ref64.to(BF))
        report("faithfulness", f"op/modulation/{math} (Modulation, native 'high' policy)", mo, nat_mod)

    # ---- the time embedding and the text / audio embeddings against upstream modules
    tv_ = torch.tensor([[700.0]], device=DEV)
    te_mod = N.build(u.dit.TimeEmbeddings, D, T, sd=N.sub(W, "video_time_embeddings."))
    freqs = te_mod.freqs.float().reshape(1, -1).contiguous()
    sincos = torch.empty(1, D, device=DEV)
    k.time_sincos(tv_, freqs, sincos)
    args = torch.outer(tv_.view(-1), freqs.view(-1))
    report("reported", "time_sincos vs torch cos / sin", sincos, torch.cat([torch.cos(args), torch.sin(args)], dim=-1))
    nat_temb = te_mod(tv_.view(-1))
    for math in K.FP32_LINEAR:
        h1 = torch.empty(1, T, device=DEV)
        temb = torch.empty(1, T, dtype=BF, device=DEV)
        k.gemv(sincos, W["video_time_embeddings.timestep_embedder.linear_1.weight"],
               W["video_time_embeddings.timestep_embedder.linear_1.bias"], h1, False, math)
        k.gemv(h1, W["video_time_embeddings.timestep_embedder.linear_2.weight"],
               W["video_time_embeddings.timestep_embedder.linear_2.bias"], temb, True, math)
        report("faithfulness", f"op/time_embeddings/{math} (TimeEmbeddings, native 'high' policy)", temb, nat_temb)
    txt = rnd(L, c.in_text_dim)
    temod = N.build(u.dit.TextEmbeddings, c.in_text_dim, D, sd=N.sub(W, "video_text_embeddings."))
    tw = torch.empty(L, D, dtype=BF, device=DEV)
    k.gemm(txt, W["video_text_embeddings.in_layer.weight"], W["video_text_embeddings.in_layer.bias"], tw, K.EPI_BIAS)
    k.layer_norm(tw, wl, bl, tw)
    report("faithfulness", "op/text_embeddings (TextEmbeddings)", tw, temod(txt))
    ve_mod = N.build(u.dit.VisualEmbeddings, C, D, (1, 2, 2), sd=N.sub(W, "visual_embeddings."))
    x5 = torch.randn(1, sh.t, sh.h, sh.w, C, generator=g, device=DEV)
    prow = torch.empty(N_, 4 * C, dtype=BF, device=DEV)
    k.patchify(x5.reshape(-1, C), prow, sh, C)
    pv = torch.empty(N_, D, dtype=BF, device=DEV)
    k.gemm(prow, pw, pb, pv, K.EPI_BIAS)
    report("faithfulness", "op/visual_embeddings (VisualEmbeddings)", pv, ve_mod(x5).reshape(N_, D))


# ----------------------------------------------------------------------------------------------------------- blocks
def blocks(k: K.Kernels, u, tc: dict, c: B.Config, sh: B.Shape, W: dict, valid: int, nb: nn.Module, mods: dict,
           math: str) -> None:
    g = torch.Generator(device=DEV).manual_seed(2)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=DEV).to(BF)

    D, Da, T = c.model_dim, c.model_dim_a, c.time_dim
    N_, A, L = sh.tokens(), sh.audio, sh.text
    rp = N.ropes(mods, c, sh)
    vr, ar, tr = N.rope_rows(rp["visual"]), N.rope_rows(rp["audio"]), N.rope_rows(rp["text"][0])
    mask = K.Mask.of(torch.arange(L, device=DEV) < valid)
    text_v, text_a, tm_v, tm_a = rnd(L, D), rnd(L, Da), rnd(1, T), rnd(1, T)
    vis0, aud0 = rnd(N_, D), rnd(A, Da)

    # ---- text block (video stack, block 0)
    def text(fill, ffn_rows=B.DEFAULT_FFN_ROWS):
        caps, hook = recorder()
        B.text_block(k, W, c, sh, 0, "video", text_v.clone(), tm_v, tr, mask, fp32_linear=math, hook=hook,
                     ffn_rows=ffn_rows, fill=fill)
        return caps

    t1 = text(0x00)
    exact_caps("text_block/repeat_scratch_0x00_0xff", t1, text(0xFF))
    exact_caps("text_block/ffn_rows_50", t1, text(0x00, 50))
    enc = N.encoder_block(tc, W, "video", 0)
    cap = N.Capture(enc, "video_text_transformer_blocks.0")
    enc(text_v, tm_v, N.rope_native(tr), N.mask_native(mask))
    report_caps("text_block", t1, cap.close())

    # ---- fused block 0
    def fused(fill, ffn_rows=B.DEFAULT_FFN_ROWS):
        caps, hook = recorder()
        io = B.BlockIo(vis0.clone(), aud0.clone(), text_v, text_a, tm_v, tm_a, vr, ar, mask)
        B.forward_block(k, W, c, sh, 0, io, fp32_linear=math, hook=hook, ffn_rows=ffn_rows, fill=fill)
        return caps

    f1 = fused(0x00)
    exact_caps("fused_block/repeat_scratch_0x00_0xff", f1, fused(0xFF))
    exact_caps("fused_block/ffn_rows_50", f1, fused(0x00, 50))
    cap = N.Capture(nb, "visual_transformer_blocks.0")
    nb(vis0.view(1, N_, D), aud0.view(1, A, Da), text_v, text_a, (tm_v, tm_a), N.rope_native(vr), N.rope_native(ar),
       N.mask_native(mask))
    report_caps("fused_block", f1, cap.close())


# ---------------------------------------------------------------------------------------------------------- forward
def whole(k: K.Kernels, u, tc: dict, c: B.Config, sh: B.Shape, W: dict, valid: int, seed: int, math: str) -> None:
    dit = N.piflow_dit(tc, W)
    nat = N.native_inputs(c, sh, N.rope_modules_of(dit), seed=seed, text_valid=valid)
    inp = N.to_twin(nat, c, N.freqs_of(dit))

    def run(fill, ffn_rows=B.DEFAULT_FFN_ROWS):
        caps, hook = recorder()
        B.forward(k, W, c, sh, inp, fp32_linear=math, hook=hook, ffn_rows=ffn_rows, fill=fill)
        return caps

    w1 = run(0x00)
    exact_caps("forward/repeat_scratch_0x00_0xff", w1, run(0xFF))
    names_ok = list(w1) == B.capture_names(c)
    R["exact"]["forward/capture_names"] = {"ok": names_ok, "captures": len(w1)}
    print(f"{'PASS' if names_ok else 'FAIL'} forward/capture_names", flush=True)
    dit.clear_text_proj_cache()
    cap = N.Capture(dit, "")
    video, audio = dit(**nat)
    nc = cap.close()
    nc["video_velocity_grid"], nc["audio_velocity_grid"] = N.twin_outputs(video, audio)
    report_caps("forward", w1, nc)


# ------------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stk_twin.kandinsky6_dit.test_ops")
    ap.add_argument("--out", default="k6_dit_receipt.json")
    ap.add_argument("--fatbins", default=None, help="directory of k6_dit_*.fatbin (default STK_K6_DIT_FATBINS)")
    ap.add_argument("--blocks", type=int, default=2, help="fused blocks of the reduced forward")
    ap.add_argument("--text-blocks", type=int, default=1)
    ap.add_argument("--small", action="store_true", help="narrower widths (fast); default the checkpoint's")
    ap.add_argument("--latent", default="3x14x22", help="T x H x W latent grid (H, W even)")
    ap.add_argument("--audio", type=int, default=218)
    ap.add_argument("--text", type=int, default=96)
    ap.add_argument("--valid", type=int, default=77, help="valid text tokens (right padding)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32-linear", required=True, choices=list(K.FP32_LINEAR),
                    help="launch.Fp32Linear for the block / forward runs (no default: none is qualified vs native)")
    a = ap.parse_args(argv)
    if a.blocks < 1:
        ap.error("--blocks must be >= 1")

    torch.cuda.init()
    cap = budget.apply("test_ops")  # before the first allocation: the step's Torch budget, when a node job set one
    k = K.Kernels(fatbins=a.fatbins)
    u = N.upstream()
    N.pipeline_numerics()
    tc = N.reduced(N.transformer_config(), visual_blocks=a.blocks, text_blocks=a.text_blocks, small=a.small)
    c = N.twin_config(tc)
    t, h, w = (int(v) for v in a.latent.split("x"))
    sh = B.Shape(t=t, h=h, w=w, audio=a.audio, text=a.text)
    sh.validate()
    plan = B.Plan(c, sh)
    R["setup"] = {
        "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "fatbins": k.digests, "upstream": str(u.src), "config": tc, "twin_config": c.__dict__,
        "shape": sh.__dict__, "tokens": sh.tokens(), "valid_text": a.valid, "seed": a.seed,
        "plan_bytes": plan.bytes(), "fp32_linear": a.fp32_linear, "torch_cap": cap,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdp": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled()},
    }
    with torch.no_grad():
        W = random_weights(c, a.seed)
        nb = N.fused_block(tc, W, 0)
        mods = N.rope_modules(c)
        ops(k, u, tc, c, sh, W, a.valid, nb, mods)
        blocks(k, u, tc, c, sh, W, a.valid, nb, mods, a.fp32_linear)
        whole(k, u, tc, c, sh, W, a.valid, a.seed, a.fp32_linear)
        torch.cuda.synchronize()
    failed = [n for n, r in R["exact"].items() if not r["ok"]]
    R["ok"] = not failed
    R["launches"] = k.count
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({"ok": not failed, "exact": len(R["exact"]), "failed": failed, "receipt": a.out}))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
