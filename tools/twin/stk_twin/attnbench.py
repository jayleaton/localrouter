"""python -m stk_twin.attnbench: the attention kernel's variants (attention.cu `pattn2_kernel`: staging slot size,
slot count, the unmasked fast path for full tiles, blocks per multiprocessor) against the reference `pattn_kernel` at
the engine's shapes ("ship" in the output: it shipped until w4s64n2b3, which ships now, so "ship_ms" is the reference's
time, not the shipping one's). A variant is a candidate only if its output is bitwise the reference's at every shape, so
adopting it changes no reference. Times are CUDA-event medians; cuDNN (SDPA) is the target. One JSON line.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch

KDIR = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[3] / "kernels")) / "cuda" / "qwen_image"

# name: (kernel, warps, slots, slot keys, fast, min blocks); "ship" is the reference pattn_kernel<128, 8, 1, 8>; the
# engine ships w4s64n2b3 (attention.cu). pattn2 keeps the
# shipping arithmetic (must be bit-equal); pattn3 is FlashAttention-2's (judged against an fp32 reference instead).
VARIANTS = {
    "same": (2, 8, 8, 32, False, 1),
    "s64n3": (2, 8, 3, 64, True, 1),
    "w4s64n2b3": (2, 4, 2, 64, True, 3),
    "w4s32n4b3": (2, 4, 4, 32, True, 3),
    "fa_w4s64n2b3": (3, 4, 2, 64, True, 3),
    "fa_w4s64n2b2": (3, 4, 2, 64, True, 2),
    "fa_w4s32n4b3": (3, 4, 4, 32, True, 3),
    "fa_w4s64n3b2": (3, 4, 3, 64, True, 2),
    "fa_w8s64n3": (3, 8, 3, 64, True, 1),
    "fa_w8s64n2b2": (3, 8, 2, 64, True, 2),
    "fa_w2s64n2b6": (3, 2, 2, 64, True, 6),
    # pattn4: (4, warps, slots, 16-row tiles a warp, min blocks); the shipping arithmetic, so bit-equal
    "r_w4m1n2b2": (4, 4, 2, 1, 2),
    "r_w4m2n2": (4, 4, 2, 2, 1),
    "r_w4m2n3": (4, 4, 3, 2, 1),
    "r_w8m2n2": (4, 8, 2, 2, 1),
    "r_w2m2n2b2": (4, 2, 2, 2, 2),
    "r_w4m2n2b2": (4, 4, 2, 2, 2),
}


def _src() -> tuple[str, str, list[str]]:
    cpp, cu, names = [], ['#include <torch/extension.h>', '#include <ATen/cuda/CUDAContext.h>', '#include "attention.cu"'], []
    sig = "(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor out, int64_t p0, int64_t nkeys, bool causal)"
    body = """{{
    const int W = q.size(0), H = q.size(1), HK = k.size(1);
    auto kernel = {kernel};
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});
    dim3 grid((W + {rows} - 1) / {rows}, H);
    kernel<<<grid, {threads}, {smem}, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        (int)p0, W, H, HK, H / HK, (int)nkeys, causal ? 1 : 0, 1.0f / sqrtf(128.0f));
}}"""
    for name, cfg in [("ship", None), *VARIANTS.items()]:
        if cfg is None:
            kernel, smem, rows, threads = "stk_attention::pattn_kernel<128, 8, 1, 8>", 65536, 128, 256
        else:
            kind = cfg[0]
            if kind == 4:
                _, warps, ns, mt, minb = cfg
                kernel = f"stk_attention::pattn4_kernel<128, {warps}, {ns}, {mt}, {minb}>"
                smem, rows, threads = ns * 64 * 256 + 16 * mt * warps * 256, 16 * mt * warps, 32 * warps
                fn = f"attn_{name}"
                cpp.append(f"void {fn}{sig};")
                cu.append(f"void {fn}{sig} " + body.format(kernel=kernel, smem=smem, rows=rows, threads=threads))
                names.append(fn)
                continue
            kind, warps, ns, slot, fast, minb = cfg
            if kind == 2:
                kernel = f"stk_attention::pattn2_kernel<128, {warps}, 1, {ns}, {slot}, {'true' if fast else 'false'}, {minb}>"
            else:
                kernel = f"stk_attention::pattn3_kernel<128, {warps}, 1, {ns}, {slot}, {minb}>"
            smem, rows, threads = ns * slot * 256, 16 * warps, 32 * warps
        fn = f"attn_{name}"
        cpp.append(f"void {fn}{sig};")
        cu.append(f"void {fn}{sig} " + body.format(kernel=kernel, smem=smem, rows=rows, threads=threads))
        names.append(fn)
    return "\n".join(cpp), "\n".join(cu), names


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load_inline
    import hashlib

    cpp, cu, names = _src()
    major, minor = torch.cuda.get_device_capability()
    arch = f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"
    digest = hashlib.sha256((KDIR / "attention.cu").read_bytes() + cu.encode()).hexdigest()[:12]
    return load_inline(f"stk_attnbench_{digest}", cpp_sources=cpp, cuda_sources=cu, functions=names,
                       extra_cuda_cflags=["-O3", arch], extra_include_paths=[str(KDIR)], verbose=False)


def run(name: str, q, k, v, p0: int, causal: bool) -> torch.Tensor:
    out = torch.empty_like(q)
    getattr(_ext(), f"attn_{name}")(q, k, v, out, p0, k.shape[0], causal)
    return out


def timed(fn, reps: int = 30) -> float:
    for _ in range(3):
        fn()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) for a, b in ev)[reps // 2]


def err(out: torch.Tensor, exact: torch.Tensor) -> dict:
    d = (out.float() - exact).flatten()
    return {"rel_l2": float(d.norm() / exact.flatten().norm()), "max_abs": float(d.abs().max())}


def main() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(0)
    # the DiT step: target rows over [prefix | target] keys (non-causal); the DiT prefix and the text encoder: causal
    shapes = {"step-1024": (4096, 32, 32, 80, False), "step-512": (1024, 32, 32, 80, False),
              "prefix": (80, 32, 32, 0, True), "te": (80, 32, 8, 0, True)}
    res = {"device": torch.cuda.get_device_name(), "shapes": {}}
    for sname, (w, h, hk, p, causal) in shapes.items():
        s_ = w + p if not causal else w
        q = torch.randn(w, h, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(s_, hk, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(s_, hk, 128, device="cuda", dtype=torch.bfloat16)
        kx, vx = (t.repeat_interleave(h // hk, dim=1) for t in (k, v))
        qt, kt, vt = (t.transpose(0, 1)[None] for t in (q, kx, vx))
        # causal here: query i sees keys 0..p0 + i with p0 = 0, SDPA's top-left mask
        exact = torch.nn.functional.scaled_dot_product_attention(qt.float(), kt.float(), vt.float(), is_causal=causal)[0].transpose(0, 1)
        ship = run("ship", q, k, v, p, causal)
        row = {"ship_ms": timed(lambda: run("ship", q, k, v, p, causal)), "ship_err": err(ship, exact)}
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
            ref = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)[0].transpose(0, 1)
            row["cudnn_ms"] = timed(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal))
        row["cudnn_err"] = err(ref, exact)
        for name in VARIANTS:
            out = run(name, q, k, v, p, causal)
            again = run(name, q, k, v, p, causal)
            row[name] = {"bits_equal": bool(torch.equal(out, ship)), "deterministic": bool(torch.equal(out, again)),
                         "err": err(out, exact), "ms": timed(lambda: run(name, q, k, v, p, causal))}
        res["shapes"][sname] = row
    sh = res["shapes"]
    bit_equal = [n for n in VARIANTS if VARIANTS[n][0] in (2, 4) and all(r[n]["bits_equal"] for r in sh.values())]
    # a new-arithmetic candidate: deterministic, and no less accurate than the shipping kernel (5 % slack) everywhere
    accurate = [n for n in VARIANTS if VARIANTS[n][0] == 3 and all(
        r[n]["deterministic"] and r[n]["err"]["rel_l2"] <= 1.05 * r["ship_err"]["rel_l2"] for r in sh.values())]
    t = lambda n: sh["step-1024"][n]["ms"]
    res.update(bit_equal=bit_equal, accurate=accurate,
               best_bit_equal=min(bit_equal, key=t) if bit_equal else None,
               best_new=min(accurate, key=t) if accurate else None)
    for key in ("best_bit_equal", "best_new"):
        if res[key]:
            res[key + "_speedup_vs_ship"] = sh["step-1024"]["ship_ms"] / t(res[key])
            res[key + "_vs_cudnn"] = sh["step-1024"]["cudnn_ms"] / t(res[key])
    print(json.dumps(res))


if __name__ == "__main__":
    main()
