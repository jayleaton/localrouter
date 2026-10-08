"""python -m stk_twin.h3.test_te32 --ckpt qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors [--write-inv-freq]

The H3 text encoder's kernels (kernels/cuda/minimax/te32.cu) and `TextEncoder32` against torch formulas, comfy-kitchen and
ComfyUI 0.37.0 ($COMFY = its checkout; optional: those checks are skipped without it, except the end to end ones, which
need it). GPU only. One JSON line {"pass": ..., "checks": {...}}.

  (i)  kernels: the NVFP4 dequantization (through the linear, with one-hot rows) bitwise against an independent torch
       unswizzle + LUT and against comfy_kitchen.dequantize_nvfp4 on real layers; the linear against fp64 (error within the
       fp32 chain bound), independent of M and deterministic; rms_norm, rope (all three FMA modes against comfy-kitchen's own
       output), attention (vs fp64 SDPA), silu_mul, embed.
  (ii) end to end: our token ids against ComfyUI's tokenizer (equal) and our [L, 5120] output against ComfyUI's own
       `CLIP.encode_from_tokens` for 3 prompts (cosine and relative L2 are printed; the thresholds below are heuristics).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import te32 as T

RES: dict[str, dict] = {}
COS_MIN, REL_L2_MAX = 0.999999, 1e-3

PROMPTS = [
    "A red fox runs through fresh snow at dawn.",
    "Cinematic tracking shot of a woman in a yellow raincoat walking through a crowded night market in Hanoi, neon signs "
    "reflecting on wet pavement, steam rising from street food stalls, shallow depth of field, handheld camera, warm "
    "tungsten light mixed with cold blue shadows. She stops, turns toward the camera and smiles; distant traffic, "
    "vendors calling, rain drumming on tarpaulin roofs. " * 2,
    "<d> hello (world:1.2) \\(escaped\\) embedding:x test <|caption_start|>caption<|caption_end|> 日本語 and emoji 🎬 \n\ndone",
]


def check(name: str, ok: bool, **info) -> bool:
    RES[name] = {"ok": bool(ok), **info}
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    return bool(ok)


def ulps32(a: torch.Tensor, b: torch.Tensor) -> int:
    def line(t):
        i = t.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)
    return int((line(a) - line(b)).abs().max())


# ------------------------------------------------------------------------------------------ reference formulas
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def ref_dequant(codes: torch.Tensor, bscale: torch.Tensor, tscale: float, N: int, K: int) -> torch.Tensor:
    """kitchen's eager dequantize_nvfp4 in fp32, the unswizzle written as float_utils.from_blocked does (reshape /
    transpose / permute, no index formula): independent of the kernel's offset arithmetic."""
    lut = torch.tensor(E2M1, dtype=torch.float32, device=codes.device)
    v = torch.stack([lut[(codes >> 4).long()], lut[(codes & 15).long()]], -1).reshape(N, K)
    ncb, nrb = K // 16, (N + 127) // 128
    ncg = (ncb + 3) // 4
    b = bscale.view(torch.float8_e4m3fn).reshape(-1)
    s = b.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2).reshape(nrb, ncg, 4, 32, 4)
    s = s.reshape(nrb, ncg, 128, 4).permute(0, 2, 1, 3).reshape(nrb * 128, ncg * 4)[:N, :ncb]
    total = torch.tensor(tscale, dtype=torch.float32, device=codes.device) * s.float()
    return (v.reshape(N, ncb, 16) * total.unsqueeze(-1)).reshape(N, K)


def kernel_dequant_cols(m, lin: T.Lin, j0: int, w: int = 256) -> torch.Tensor:
    """W[:, j0:j0 + w] through the linear kernel: one-hot activation rows give y[r, n] = W[n, j0 + r] exactly (fmaf(1, w, 0))."""
    dev = lin.codes.device
    x = torch.zeros(w, lin.K, device=dev)
    x[torch.arange(w, device=dev), j0 + torch.arange(w, device=dev)] = 1.0
    return m.k_linear_nvfp4(x, None, lin.codes, lin.bscale, lin.tscale, None).T


def ref_rope(x: torch.Tensor, cs: torch.Tensor, sn: torch.Tensor, mode: int) -> torch.Tensor:
    """The three FMA contractions with exact fp64 products (a product of two fp32 values is exact in fp64; the sum's
    double rounding is negligible)."""
    L = x.shape[0]
    x0, x1 = x[..., :64], x[..., 64:]
    c, s = cs[:L, None, :], sn[:L, None, :]

    def fma(a, b, c_):
        return (a.double() * b.double() + c_.double()).float()
    f00, f01, f10, f11 = c, -s, s, c
    if mode == 0:
        y0, y1 = fma(f00, x0, f01 * x1), fma(f10, x0, f11 * x1)
    elif mode == 1:
        y0, y1 = fma(f01, x1, f00 * x0), fma(f11, x1, f10 * x0)
    else:
        y0, y1 = f00 * x0 + f01 * x1, f10 * x0 + f11 * x1
    return torch.cat([y0, y1], -1)


# ------------------------------------------------------------------------------------------ (i) kernel checks
def kernel_checks(ckpt: str, ck) -> int:
    from safetensors import safe_open

    m = T.mod()
    dev = "cuda"
    torch.manual_seed(0)
    best_mode = 0

    # rms_norm (D 5120 and per-head 128)
    for D in (5120, 128):
        x = torch.randn(257, D, device=dev) * 3
        w = torch.rand(D, device=dev) + 0.5
        y = m.k_rms_norm(x, w, T.EPS)
        ref = (x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + T.EPS) * w.double()).float()
        check(f"rms_norm.{D}", ulps32(y, ref) <= 4 and torch.equal(y, m.k_rms_norm(x, w, T.EPS)), max_ulp_vs_fp64=ulps32(y, ref))

    # silu_mul and add
    g, u = torch.randn(1000, 777, device=dev) * 4, torch.randn(1000, 777, device=dev)
    y = m.k_silu_mul(g, u)
    ref = F.silu(g) * u
    check("silu_mul", ulps32(y, ref) <= 1, max_ulp_vs_torch=ulps32(y, ref), identical=float((y == ref).float().mean()))
    check("add", torch.equal(m.k_add(g, u), g + u))

    # rope: all three modes against the exact-product emulation, and against comfy-kitchen when importable
    L = 300
    cs, sn = (t.to(dev) for t in T.rope_tables(L))
    for H in (64, 8):
        x = torch.randn(L, H, 128, device=dev)
        for mode in (0, 1, 2):
            y = x.clone()
            m.k_rope_(y, cs, sn, mode)
            r = ref_rope(x, cs, sn, mode)
            check(f"rope.H{H}.mode{mode}", ulps32(y, r) <= 1, max_ulp=ulps32(y, r), identical=float((y == r).float().mean()))
    if ck is not None:
        x = torch.randn(L, 64, 128, device=dev)
        k = torch.randn(L, 8, 128, device=dev)
        mat = torch.stack([cs, -sn, sn, cs], -1).reshape(1, 1, L, 64, 2, 2)
        kq, kk = ck.apply_rope_split_half(x.permute(1, 0, 2)[None].contiguous(), k.permute(1, 0, 2)[None].contiguous(), mat)
        kq = kq[0].permute(1, 0, 2).contiguous()
        match = {}
        for mode in (0, 1, 2):
            y = x.clone()
            m.k_rope_(y, cs, sn, mode)
            match[mode] = float((y == kq).float().mean())
        best_mode = max(match, key=match.get)
        check("rope.vs_kitchen", match[best_mode] == 1.0, identical_by_mode=match, best_mode=best_mode)

    # attention vs fp64 SDPA (causal, GQA 64 over 8) and determinism
    for L in (1, 7, 129, 700):
        q, k, v = (torch.randn(L, h, 128, device=dev) for h in (64, 8, 8))
        out = m.k_attention(q, k, v, T.HD ** -0.5)
        kr, vr = k.repeat_interleave(8, dim=1), v.repeat_interleave(8, dim=1)
        ref = F.scaled_dot_product_attention(q.double().transpose(0, 1)[None], kr.double().transpose(0, 1)[None],
                                             vr.double().transpose(0, 1)[None], is_causal=True)[0]
        ref = ref.transpose(0, 1).reshape(L, -1)
        err = float((out.double() - ref).abs().max())
        bound = 2e-6 * max(1.0, (L / 64) ** 0.5)  # fp32 sums over L keys: the error grows like a random walk, sqrt(L)
        check(f"attention.L{L}", err < bound and torch.equal(out, m.k_attention(q, k, v, T.HD ** -0.5)), max_abs_err_vs_fp64=err, bound=bound)

    # embed: the int8 table with per-row scales, bf16 rounding (ComfyUI's orig_dtype) then fp32
    V, D = 5000, 5120
    tab = torch.randint(-127, 128, (V, D), device=dev, dtype=torch.int8)
    sc = torch.rand(V, 1, device=dev) * 0.02
    ids = torch.randint(0, V, (333,), device=dev, dtype=torch.int32)
    ref = (tab[ids.long()].float() * sc[ids.long()]).bfloat16().float()
    check("embed_i8", torch.equal(m.k_embed_i8(tab, sc, ids, 1), ref))
    check("embed_i8.fp32", torch.equal(m.k_embed_i8(tab, sc, ids, 0), tab[ids.long()].float() * sc[ids.long()]))
    tb = torch.randn(V, D, device=dev).bfloat16()
    check("embed_bf16", torch.equal(m.k_embed_bf16(tb, ids), tb[ids.long()].float()))

    # real layers: dequantization bitwise, linear vs fp64
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        for layer in (0, 24, 49):
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
                key = f"model.layers.{layer}.{name}"
                lin = T.load_lin(f, key, dev)
                ref = ref_dequant(lin.codes, lin.bscale, lin.tscale, lin.N, lin.K)
                starts = list(range(0, lin.K - 255, 256)) if lin.K <= 8192 else [0, 4096 + 16, lin.K // 2, lin.K - 256]
                ok = all(torch.equal(kernel_dequant_cols(m, lin, j0), ref[:, j0:j0 + 256]) for j0 in starts)
                info = {"windows": len(starts), "N": lin.N, "K": lin.K, "pqs": lin.pqs is not None}
                if ck is not None:
                    kit = ck.dequantize_nvfp4(lin.codes, torch.tensor(lin.tscale, dtype=torch.float32, device=dev),
                                              lin.bscale.view(torch.float8_e4m3fn).reshape(lin.N, lin.K // 16), torch.float32)
                    info["ref_equals_kitchen"] = bool(torch.equal(kit, ref))
                    ok = ok and info["ref_equals_kitchen"]
                check(f"dequant.{key}", ok, **info)

                M = 300
                x = torch.randn(M, lin.K, device=dev)
                y = m.k_linear_nvfp4(x, lin.pqs, lin.codes, lin.bscale, lin.tscale, None)
                a = x if lin.pqs is None else x * lin.pqs
                bound = (a.abs().double() @ ref.abs().double().T) * (lin.K * 2.0 ** -24)
                err = (y.double() - a.double() @ ref.double().T).abs()
                y5 = m.k_linear_nvfp4(x[:5].contiguous(), lin.pqs, lin.codes, lin.bscale, lin.tscale, None)
                same = torch.equal(y[:5], y5) and torch.equal(y, m.k_linear_nvfp4(x, lin.pqs, lin.codes, lin.bscale, lin.tscale, None))
                check(f"linear.{key}", bool((err <= bound).all()) and same, max_err_over_bound=float((err / bound.clamp_min(1e-30)).max()),
                      independent_of_M_and_deterministic=same)
                del lin, ref
                torch.cuda.empty_cache()
    return best_mode


def inv_freq_check(write: bool) -> None:
    """Our portable inv_freq / cos / sin against ComfyUI's GPU computation (`precompute_freqs_cis`)."""
    dev = "cuda"
    num = torch.arange(0, 128, 2, device=dev).float()
    comfy_inv = (1.0 / (T.THETA ** (num / 128))).cpu().numpy()
    ours = T.inv_freq()
    bad = int((comfy_inv != ours).sum())
    check("inv_freq.vs_comfy_gpu", bad == 0, mismatching=bad, of=64)
    if bad and write:
        p = T.KROOT / "minimax" / "te32_inv_freq.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"inv_freq": [float(v) for v in comfy_inv]}))
        print(f"wrote {p}; rerun", flush=True)
    L = 1024
    pos = torch.arange(L, device=dev)[None]
    f = (torch.from_numpy(comfy_inv).to(dev)[None, :, None].float() @ pos[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((f, f), -1)
    cc, ss = emb.cos()[0, :, :64].cpu(), emb.sin()[0, :, :64].cpu()
    cs, sn = T.rope_tables(L)
    check("rope_tables.vs_cuda_cosf_sinf", True, cos_mismatching=int((cc != cs).sum()), sin_mismatching=int((ss != sn).sum()),
          of=L * 64, note="informational: CUDA cosf / sinf are not correctly rounded; ours are")


# ------------------------------------------------------------------------------------------ (ii) end to end
def comfy_refs(ckpt: str, root: str) -> list[dict]:
    sys.path.insert(0, root)
    import comfy.model_management
    import comfy.sd

    clip = comfy.sd.load_clip([ckpt], embedding_directory=None, clip_type=comfy.sd.CLIPType.MINIMAX)
    out = []
    for p in PROMPTS:
        tokens = clip.tokenize(p)
        ids = [int(t[0]) for t in tokens["qwen3vl_32b"][0]]
        cond = clip.encode_from_tokens(tokens)
        out.append({"ids": ids, "cond": cond[0].float().cpu()})
    del clip
    comfy.model_management.unload_all_models()
    torch.cuda.empty_cache()
    return out


def end_to_end(ckpt: str, refs: list[dict], rope_mode: int) -> None:
    enc = T.TextEncoder32(ckpt, rope_mode=rope_mode)
    for n, (p, ref) in enumerate(zip(PROMPTS, refs)):
        ids = T.tokenize(p)
        check(f"tokenize.prompt{n}", ids == ref["ids"], ours=len(ids), comfy=len(ref["ids"]))
        y = enc.forward(torch.tensor(ref["ids"])).cpu()
        r = ref["cond"]
        cos = float(F.cosine_similarity(y.double().flatten(), r.double().flatten(), dim=0))
        rel = float((y.double() - r.double()).norm() / r.double().norm())
        row_rel = ((y.double() - r.double()).norm(dim=-1) / r.double().norm(dim=-1)).max().item()
        check(f"encode.prompt{n}", cos >= COS_MIN and rel <= REL_L2_MAX, L=len(ref["ids"]), cosine=cos, rel_l2=rel,
              max_row_rel_l2=row_rel, shape=list(y.shape), finite=bool(torch.isfinite(y).all()))
        y2 = enc.forward(torch.tensor(ref["ids"])).cpu()
        check(f"encode.prompt{n}.deterministic", torch.equal(y, y2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("STK_TE32_CKPT"))
    ap.add_argument("--write-inv-freq", action="store_true", help="freeze ComfyUI's GPU inv_freq into kernels/minimax/te32_inv_freq.json when ours differs")
    ap.add_argument("--rope-mode", type=int, default=None, help="FMA mode for the end to end run (default: the one matching comfy-kitchen)")
    args = ap.parse_args()
    if not args.ckpt:
        print("need --ckpt or $STK_TE32_CKPT", file=sys.stderr)
        return 2
    root = os.environ.get("COMFY")
    ck = None
    if root:
        sys.path.insert(0, root)
        try:
            import comfy_kitchen as ck  # noqa: F811
        except Exception as e:  # the kitchen checks are skipped
            print(f"comfy_kitchen not importable: {e}", flush=True)
    best = kernel_checks(args.ckpt, ck)
    if root:
        inv_freq_check(args.write_inv_freq)
        end_to_end(args.ckpt, comfy_refs(args.ckpt, root), best if args.rope_mode is None else args.rope_mode)
    else:
        check("end_to_end.skipped", False, reason="set $COMFY to run the tokenizer and encode comparisons")
    ok = all(v["ok"] for v in RES.values())
    print(json.dumps({"pass": ok, "checks": RES}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
