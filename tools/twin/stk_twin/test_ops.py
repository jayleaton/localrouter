"""GPU test of stk_twin.ops against the torch expressions it replaces: `python -m stk_twin.test_ops`.

Per op: error vs torch (max abs / max rel / max bf16 ulp / fraction bit-identical), then determinism (two runs, same bits).
Gate: elementwise ops and quantisation bit-identical or within 1 bf16 ulp; reductions within 1 ulp; absmax and the
quantised codes exact. Ends with one JSON line; exit status 1 on any failure.
"""

from __future__ import annotations

import json
import math
import sys

import torch
import torch.nn.functional as F

from . import ops

EPS = 1e-6
RESULTS: dict[str, dict] = {}


def ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 representable steps (sign-magnitude mapped to a monotone integer line)."""
    def line(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (line(a) - line(b)).abs()


def compare(name: str, got: torch.Tensor, ref: torch.Tensor, max_ulp: int, note: str = "") -> bool:
    if got.dtype == torch.bfloat16:
        u = ulps(got, ref)
        g, r = got.float(), ref.float()
        d = (g - r).abs()
        rel = d / r.abs().clamp_min(1e-30)
        r_ = dict(max_abs=d.max().item(), max_rel=rel.max().item(), max_ulp=int(u.max().item()),
                  identical=(u == 0).float().mean().item())
    else:  # integer codes / fp32: exact compare
        eq = (got.view(torch.uint8) == ref.view(torch.uint8)) if got.element_size() == 1 else (got == ref)
        r_ = dict(max_abs=float((got.float() - ref.float()).abs().max()), max_rel=0.0, max_ulp=0 if bool(eq.all()) else -1,
                  identical=eq.float().mean().item())
    ok = 0 <= r_["max_ulp"] <= max_ulp
    r_.update(ok=ok, gate_ulp=max_ulp, note=note)
    RESULTS[name] = r_
    print(f"{'PASS' if ok else 'FAIL'} {name:34s} ulp<={r_['max_ulp']:<3d} abs={r_['max_abs']:.3e} rel={r_['max_rel']:.3e} "
          f"identical={r_['identical']:.6f} {note}", flush=True)
    return ok


def determinism(name: str, fn) -> bool:
    a, b = fn(), fn()
    same = torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a.view(torch.int32),
                       b.view(torch.uint8) if b.dtype != torch.float32 else b.view(torch.int32))
    RESULTS.setdefault(name, {})["deterministic"] = same
    print(f"{'PASS' if same else 'FAIL'} {name:34s} deterministic", flush=True)
    return same


def bf(shape, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g) * scale).to(torch.bfloat16)


def edge(n: int) -> torch.Tensor:
    """Edge values: zeros (both signs), tiny, subnormal-ish, large finite, exact ties."""
    v = [0.0, -0.0, 1e-30, -1e-30, 1e-38, 3e38, -3e38, 1e30, -1e30, 88.0, -88.0, 89.0, -89.0, 1.0, -1.0, 0.5, 20.0, -20.0, 1e-3]
    t = torch.tensor(v, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    return t.repeat(n // len(v) + 1)[:n]


def main() -> int:
    ok = True
    dev_name = torch.cuda.get_device_name()
    print(f"device {dev_name} cc {torch.cuda.get_device_capability()} torch {torch.__version__}", flush=True)
    D = 4096
    shapes = [(1, 1024, D), (1, 4096, D)]

    # --- pointwise
    refs = {"silu": F.silu, "tanh": torch.tanh, "gelu_tanh": lambda t: F.gelu(t, approximate="tanh")}
    for fn, ref in refs.items():
        for shp in [(60, D), (1, 1, D), *[s[1:] for s in shapes]]:
            for label, x in (("rand", bf(shp, 3.0)), ("wide", bf(shp, 40.0, 1)), ("edge", edge(math.prod(shp)).view(shp))):
                ok &= compare(f"{fn}/{shp[-2]}x{shp[-1]}/{label}", ops.POINTWISE[fn](x), ref(x), 1,
                              "tanh/exp libm path vs torch's own expf/tanhf: same source, same ops" if fn != "silu" else "")
        x = bf((1, 4096, D), 3.0)
        ok &= determinism(f"{fn}/det", lambda x=x, f=ops.POINTWISE[fn]: f(x))

    # --- gated residual (torch addcmul_ is one fp32 fma; ours is fmaf)
    for B, N, _ in [(1, 1024, D), (1, 4096, D), (2, 64, D)]:
        for label, mk in (("rand", lambda s: bf(s, 1.0, 5)), ("edge", lambda s: edge(math.prod(s)).view(s))):
            x0, a, g = mk((B, N, D)), mk((B, N, D)), bf((B, 1, D), 1.0, 7) if label == "rand" else edge(B * D).view(B, 1, D)
            ref = x0.clone().addcmul_(a, g)
            got = x0.clone()
            ops.gated_residual_(got, a, g)
            ok &= compare(f"gated_residual/{B}x{N}/{label}", got, ref, 1, "fma vs torch's contraction of a + alpha*b*c")
    x0, a, g = bf((1, 4096, D), 1.0, 5), bf((1, 4096, D), 1.0, 6), bf((1, 1, D), 1.0, 7)
    def run():
        x = x0.clone(); ops.gated_residual_(x, a, g); return x
    ok &= determinism("gated_residual/det", run)

    # --- rms norm
    w = torch.rand(D, device="cuda") * 2 + 0.01
    def rms_ref(c):
        c = c.float()
        return (c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + EPS) * w).to(torch.bfloat16)
    for rows in (60, 1, 77):
        for label, x in (("rand", bf((rows, D), 2.0, 3)), ("big", bf((rows, D), 300.0, 4)), ("tiny", bf((rows, D), 1e-4, 5)),
                         ("zeros", torch.zeros(rows, D, device="cuda", dtype=torch.bfloat16))):
            ok &= compare(f"rms_norm/{rows}/{label}", ops.rms_norm_f32(x, w, EPS), rms_ref(x), 1,
                          "sum order differs from torch's reduction (fp32 sum, then one bf16 rounding)")
    x = bf((60, D), 2.0, 3)
    ok &= determinism("rms_norm/det", lambda: ops.rms_norm_f32(x, w, EPS))

    # --- time sinusoid
    def time_ref(t):
        tr = ((t * 1000).to(torch.bfloat16) / 1000).to(torch.bfloat16)
        tr = torch.cat([tr, tr.new_zeros(1)]).float() * 1000
        freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32, device=t.device) / 128)
        args = tr[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(11)
    ts = torch.cat([torch.rand(4096, device="cuda", generator=g), torch.tensor([0.0, 1.0, 1e-7, 0.5, 0.999], device="cuda")])
    ok &= compare("time_sinusoid/B=1", ops.time_sinusoid(ts[:1]), time_ref(ts[:1]), 1)
    ok &= compare("time_sinusoid/B=2", ops.time_sinusoid(ts[2:4]), time_ref(ts[2:4]), 1)
    ok &= compare("time_sinusoid/4096+edges", ops.time_sinusoid(ts), time_ref(ts), 1,
                  "cos/sin of large args (up to 1000): ours is full-precision cosf/sinf like torch's")
    ok &= determinism("time_sinusoid/det", lambda: ops.time_sinusoid(ts))

    # --- euler
    for n in (64 * 64 * 64, 64 * 128 * 128, D * 64):
        x, v = bf((n,), 1.0, 8), bf((n,), 2.0, 9)
        for dt in (-0.04, -0.0123456789, 0.5, -1.0, 1e-9):
            ok &= compare(f"euler/{n}/dt={dt}", ops.euler(x, v, dt), (x.float() + dt * v.float()).to(torch.bfloat16), 1)
    x, v = edge(1 << 16), edge(1 << 16).roll(3)
    ok &= compare("euler/edge", ops.euler(x, v, -0.0371), (x.float() + (-0.0371) * v.float()).to(torch.bfloat16), 1)
    ok &= determinism("euler/det", lambda: ops.euler(x, v, -0.0371))

    # --- copy rows
    B, P, N, H, Dh = 1, 60, 1024, 32, 128
    qkv = bf((B, N, 3, H, Dh), 1.0, 12)
    vv = torch.zeros((B, P + N, H, Dh), dtype=torch.bfloat16, device="cuda")
    ref = vv.clone(); ref[:, P:].copy_(qkv[:, :, 2])
    ops.copy_rows(vv[:, P:], qkv[:, :, 2])
    ok &= compare("copy_rows/v", vv, ref, 0)
    pk = bf((B, P, H, Dh), 1.0, 13)
    kb = torch.zeros((B, P + N, H, Dh), dtype=torch.bfloat16, device="cuda")
    refk = kb.clone(); refk[:, :P].copy_(pk)
    ops.copy_rows(kb[:, :P], pk)
    ok &= compare("copy_rows/prefix", kb, refk, 0)
    odd = bf((1, 7, 3, 5), 1.0, 14)  # unaligned rows exercise the byte path
    dd = torch.zeros((1, 9, 3, 5), dtype=torch.bfloat16, device="cuda")
    rr = dd.clone(); rr[:, 1:8].copy_(odd)
    ops.copy_rows(dd[:, 1:8], odd)
    ok &= compare("copy_rows/unaligned", dd, rr, 0)

    # --- fp8: absmax exact, scale exact, codes exact
    for M, K in ((60, D), (1024, D), (4096, D), (1000, 12288), (4096, 12288)):
        for label, x in (("rand", bf((M, K), 1.0, 20)), ("outlier", torch.cat([bf((M - 1, K), 1.0, 21), bf((1, K), 400.0, 22)])),
                         ("zeros", torch.zeros(M, K, device="cuda", dtype=torch.bfloat16))):
            stat = ops.absmax_stat(x)
            amax_ref = x.abs().amax().float()
            a_ref = (amax_ref / 448).clamp_min(1e-12)
            ok &= compare(f"absmax/{M}x{K}/{label}", stat[0:1], amax_ref.view(1), 0, "exact")
            ok &= compare(f"act_scale/{M}x{K}/{label}", stat[1:2], a_ref.view(1), 0,
                          "torch divides by the CPU scalar 448 as a multiply by fp32 1/448; ours does the same")
            xq, a = ops.quant_e4m3(x, stat)
            pad = -M % 16
            q_ref = (x.float() / a_ref).clamp(-448, 448).to(torch.float8_e4m3fn)
            if pad:
                q_ref = F.pad(q_ref.view(torch.uint8), (0, 0, 0, pad)).view(torch.float8_e4m3fn)
            ok &= compare(f"quant_e4m3/{M}x{K}/{label}", xq.view(torch.uint8), q_ref.view(torch.uint8), 0,
                          "codes exact (fp32 IEEE divide, RNE saturating cvt)")
    x = bf((1000, D), 1.0, 30)
    ok &= determinism("absmax/det", lambda: ops.absmax_stat(x))
    ok &= determinism("quant_e4m3/det", lambda: ops.quant_e4m3(x, ops.absmax_stat(x))[0].view(torch.uint8))

    # --- dense bf16 (F.linear): not bit-equal to cuBLAS by design; relative error, determinism, row invariance
    dense_shapes = [("time.lin1", 2, 256, 4096), ("time.lin2", 2, 4096, 4096), ("mod", 2, 4096, 16384),
                    ("txt.in", 60, 4096, 4096), ("img_in", 4096, 64, 4096), ("img_in1k", 1024, 64, 4096),
                    ("norm_out", 1, 4096, 4096), ("proj_out", 4096, 4096, 64), ("proj_out1k", 1024, 4096, 64),
                    ("ragged", 77, 4093 // 1, 100)]
    for name, M, K, N in dense_shapes:
        x, wt = bf((M, K), 1.0, 40), bf((N, K), 1.0 / math.sqrt(K), 41)
        got, ref = ops.dense_bf16(x, wt), F.linear(x, wt)
        d = (got.float() - ref.float()).abs()
        rel = (d.max() / ref.float().abs().max()).item()
        u = ulps(got, ref)
        r_ = dict(max_abs=d.max().item(), max_rel=rel, max_ulp=int(u.max().item()), identical=(u == 0).float().mean().item(),
                  ok=rel < 2e-2, gate_ulp=-1, note="max |d| / max |ref|; fp32 accumulation order differs from cuBLAS")
        RESULTS[f"dense/{name}/{M}x{K}->{N}"] = r_
        print(f"{'PASS' if r_['ok'] else 'FAIL'} dense/{name} {M}x{K}->{N} rel_to_max={rel:.3e} max_ulp={r_['max_ulp']} "
              f"identical={r_['identical']:.4f}", flush=True)
        ok &= r_["ok"]
        ok &= determinism(f"dense/{name}/det", lambda x=x, wt=wt: ops.dense_bf16(x, wt))
    x, wt = bf((4096, 4096), 1.0, 42), bf((4096, 4096), 1.0 / 64, 43)
    full = ops.dense_bf16(x, wt)
    inv = all(torch.equal(ops.dense_bf16(x[i:i + m], wt), full[i:i + m]) for i, m in ((0, 1), (0, 2), (5, 1), (100, 60), (4095, 1), (63, 65)))
    RESULTS["dense/row_invariance"] = dict(ok=inv, deterministic=True)
    print(f"{'PASS' if inv else 'FAIL'} dense/row_invariance (rows of an M=4096 call == smaller calls, bitwise)", flush=True)
    ok &= inv

    # --- gemm bf16 (gemm.cu, tensor cores): vs the fp32 product rounded once; row / column invariance; bias
    gemm_shapes = [("te.q", 100, 4096, 4096), ("te.kv", 100, 4096, 1024), ("te.gate", 100, 4096, 12288),
                   ("te.down", 100, 12288, 4096), ("one", 1, 4096, 4096), ("ragged", 77, 40, 130),
                   ("vae.conv", 16384, 1296, 144), ("vae.wide", 4096, 2592, 288)]
    def gemm_check(name, got, x, wt, b=None):
        """vs the fp64 product rounded once: |d| <= one bf16 ulp of the exact value + an fp32 accumulation bound
        (K * 2^-24 * sum |x||w|); also how often each side lands on the correctly rounded value."""
        ref = x.double() @ wt.double().T + (0 if b is None else b.double())
        bound = (x.double().abs() @ wt.double().abs().T) * (x.shape[1] * 2.0 ** -24)
        ulp = torch.ldexp(torch.ones_like(ref), torch.frexp(ref.abs().clamp_min(1e-38))[1] - 8)
        d = (got.double() - ref).abs()
        ok_ = bool((d <= ulp + bound).all())
        exact = (got == ref.to(torch.bfloat16)).float().mean().item()
        torch_ = F.linear(x, wt, b)
        texact = (torch_ == ref.to(torch.bfloat16)).float().mean().item()
        RESULTS[name] = dict(ok=ok_, max_abs=d.max().item(), correctly_rounded=exact, torch_correctly_rounded=texact,
                             note="|d| <= ulp + fp32 bound vs fp64")
        print(f"{'PASS' if ok_ else 'FAIL'} {name:34s} max_abs={d.max().item():.3e} correctly_rounded={exact:.6f} "
              f"(torch {texact:.6f})", flush=True)
        return ok_

    for name, M, K, N in gemm_shapes:
        x, wt = bf((M, K), 1.0, 50), bf((N, K), 1.0 / math.sqrt(K), 51)
        ok &= gemm_check(f"gemm/{name}/{M}x{K}->{N}", ops.gemm_bf16(x, wt), x, wt)
        ok &= determinism(f"gemm/{name}/det", lambda x=x, wt=wt: ops.gemm_bf16(x, wt))
    x, wt, b = bf((300, 1296), 1.0, 52), bf((144, 1296), 0.03, 53), bf((144,), 1.0, 54)
    ok &= gemm_check("gemm/bias/300x1296->144", ops.gemm_bf16(x, wt, b), x, wt, b)
    x, wt = bf((1000, 4096), 1.0, 55), bf((4096, 4096), 1.0 / 64, 56)
    full = ops.gemm_bf16(x, wt)
    inv = all(torch.equal(ops.gemm_bf16(x[i:i + m], wt), full[i:i + m]) for i, m in ((0, 1), (5, 1), (100, 60), (999, 1), (63, 129)))
    inv &= all(torch.equal(ops.gemm_bf16(x, wt[j:j + n]), full[:, j:j + n]) for j, n in ((0, 8), (128, 1024), (4000, 96)))
    RESULTS["gemm/invariance"] = dict(ok=inv, deterministic=True)
    print(f"{'PASS' if inv else 'FAIL'} gemm/invariance (row and column blocks == the full call, bitwise)", flush=True)
    ok &= inv

    # --- text encoder ops (te.cu) vs transformers' Qwen3-VL text formulas in torch
    for L, Dn, label in ((100, 4096, "hidden"), (100 * 32, 128, "q_norm"), (1, 4096, "one")):
        x, w = bf((L, Dn), 2.0, 60), bf((Dn,), 0.5, 61) + 1
        xf = x.float()
        ref = w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        ok &= compare(f"rms_norm_hf/{label}", ops.rms_norm_hf(x, w, 1e-6), ref, 1, "fp32 sum order")
    from .te import rope_tables
    L, H = 97, 32
    cos, sin = rope_tables(L)
    x = bf((L, H, 128), 1.0, 62)
    cb, sb = cos.cuda().to(torch.bfloat16), sin.cuda().to(torch.bfloat16)
    cb, sb = torch.cat([cb, cb], -1)[:, None], torch.cat([sb, sb], -1)[:, None]
    rot = torch.cat([-x[..., 64:], x[..., :64]], -1)
    ref = x * cb + rot * sb
    got = x.clone()
    ops.rope_half_(got, cos.cuda(), sin.cuda())
    ok &= compare("rope_half/97x32", got, ref, 0, "transformers' bf16 order")
    g, u = bf((100, 12288), 3.0, 63), bf((100, 12288), 1.0, 64)
    ok &= compare("silu_mul/100x12288", ops.silu_mul(g, u), F.silu(g) * u, 1, "expf path")
    a_, z_ = bf((100, 4096), 1.0, 65), bf((100, 4096), 1.0, 66)
    ok &= compare("add/100x4096", ops.add(a_, z_), a_ + z_, 0)
    tab, ids = bf((151936, 4096), 1.0, 67), torch.randint(0, 151936, (100,), device="cuda")
    ok &= compare("embed/100", ops.embed(tab, ids), tab[ids], 0)

    ok &= all(r.get("deterministic", True) for r in RESULTS.values())
    worst = {k: v.get("max_ulp") for k, v in RESULTS.items() if v.get("max_ulp", 0) > 0}
    print(json.dumps(dict(ok=bool(ok), device=dev_name, checks=len(RESULTS), nonidentical_ops=worst,
                          failed=[k for k, v in RESULTS.items() if not v.get("ok", True) or not v.get("deterministic", True)])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
