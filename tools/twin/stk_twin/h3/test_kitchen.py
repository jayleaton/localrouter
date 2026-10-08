"""GPU test of stk_twin.h3.kitchen against the comfy-kitchen 0.2.35 wheel: `python -m stk_twin.h3.test_kitchen`.

Both ops are compared bitwise (torch.equal) with `torch.ops.comfy_kitchen.int8_attention` and
`comfy_kitchen.rms_rope_split_half_` on random inputs at H3's shapes, then run twice for determinism. Needs the wheel
(`pip install comfy-kitchen==0.2.35`, the CUDA extension on this GPU). Ends with one JSON line; exit status 1 on any failure.
Inputs are bf16 views into one packed qkv buffer [S, 3*H*D] (H3's layout) and, for attention, also contiguous [1,H,S,D];
K gets a per-channel offset in the "biased" variant so the wheel's K shift (the anchor key) is taken.
"""

from __future__ import annotations

import json
import sys

import torch

from . import kitchen

BF = torch.bfloat16
EPS, ROT = 1e-5, 96
# (S, H, D): H3's attention shapes, then its audio-size shape at D 64
ATTN = [(80, 56, 128), (4096, 56, 128), (9000, 56, 128), (1797, 32, 64)]
ROPE = [(80, 56), (4096, 56), (9000, 56), (1797, 32)]
CASES: list[dict] = []


def randn(shape, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g) * scale).to(BF)


def report(op: str, shape: str, got, ref, again) -> bool:
    equal = torch.equal(got, ref)
    det = torch.equal(got, again)
    r = dict(op=op, shape=shape, equal=equal, deterministic=det)
    if not equal:
        r["mismatched"] = int((got != ref).sum())
        r["max_abs"] = float((got.float() - ref.float()).abs().max())
    CASES.append(r)
    print(f"{'PASS' if equal and det else 'FAIL'} {op:10s} {shape:42s} equal={equal} deterministic={det}", flush=True)
    return equal and det


def qkv_views(S, H, D, seed, biased):
    """q, k, v [1,H,S,D] as strided views into one packed [S, 3*H*D] buffer."""
    qkv = randn((S, 3 * H * D), seed)
    if biased:  # a channel offset on K (and a larger Q), as real keys carry
        off = randn((1, H * D), seed + 1, 4.0)
        qkv[:, H * D:2 * H * D] += off
        qkv[:, :H * D] *= 2
    return [qkv[:, i * H * D:(i + 1) * H * D].view(1, S, H, D).transpose(1, 2) for i in range(3)]


def attention() -> bool:
    ok = True
    for S, H, D in ATTN:
        for layout in ("packed", "contiguous"):
            for biased in (False, True):
                if layout == "packed":
                    q, k, v = qkv_views(S, H, D, 10, biased)
                else:
                    q, k, v = (randn((1, H, S, D), 20 + i, 2.0 if i == 0 and biased else 1.0) for i in range(3))
                    if biased:
                        k = k + randn((1, H, 1, D), 30, 4.0)
                ref = torch.ops.comfy_kitchen.int8_attention(q, k, v, None)
                got = kitchen.int8_attention(q, k, v)
                ok &= report("attention", f"S{S} H{H} D{D} {layout} {'biased' if biased else 'plain'}", got, ref,
                             kitchen.int8_attention(q, k, v))
                del ref, got
    return ok


def rope_freqs(S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ang = torch.rand(1, S, 1, ROT // 2, device="cuda", generator=g) * 6.2831853
    c, s = ang.cos(), ang.sin()
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(BF)  # [1, S, 1, ROT/2, 2, 2]


def rope() -> bool:
    import comfy_kitchen

    ok = True
    D = 128
    for S, H in ROPE:
        qkv = randn((S, 3 * H * D), 40)
        freqs = rope_freqs(S, 41)
        qw, kw = [(1 + 0.1 * randn((D,), 42 + i)).to(BF) for i in range(2)]

        def run(fn, buf):
            q, k = (buf[:, i * H * D:(i + 1) * H * D].view(1, S, H, D) for i in range(2))
            fn(q, k, freqs, qw, kw, EPS, ROT)
            return buf

        ref = run(comfy_kitchen.rms_rope_split_half_, qkv.clone())
        got = run(kitchen.rms_rope_split_half_, qkv.clone())
        again = run(kitchen.rms_rope_split_half_, qkv.clone())
        # the whole buffer: q and k rewritten, v untouched
        ok &= report("rms_rope", f"S{S} H{H} D{D} rot{ROT}", got, ref, again)
    return ok


def main() -> int:
    import comfy_kitchen  # noqa: F401  (registers torch.ops.comfy_kitchen)

    print(f"device {torch.cuda.get_device_name()} cc {torch.cuda.get_device_capability()} torch {torch.__version__} "
          f"comfy_kitchen {getattr(comfy_kitchen, '__version__', '?')}", flush=True)
    ok = bool(attention())
    ok &= bool(rope())
    print(json.dumps({"pass": bool(ok), "cases": CASES}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
