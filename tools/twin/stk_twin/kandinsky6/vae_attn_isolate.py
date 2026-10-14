"""python -I -m stk_twin.kandinsky6.vae_attn_isolate --model DIR --src KANDINSKY_SRC --trace REF_TRACE --out DIR
       [--tile-thw 5,32,29] [--oracle-prefix /opt/k6-oracle]

The Kandinsky-6 video VAE decoder's mid-block attention, isolated (diagnostic, unqualified). The pinned upstream
module `decoder.mid_block.attentions.0` (diffusers Attention, AttnProcessor2_0) runs as HunyuanVideoMidBlock3D calls
it: the trace's `decoder.mid_block.resnets.0` output (k6.aux.vae.trace.*, [positions][channels]) rebuilt into
(1, C, T, H, W), then permute(0, 2, 3, 4, 1).flatten(1, 3), with prepare_causal_attention_mask. Forward hooks on
group_norm, to_q, to_k, to_v and to_out.0, and a wrapper around F.scaled_dot_product_attention for the call, read
every value. Nothing changes torch's arithmetic.

`reproduces`: group_norm, to_q, to_k, to_v and the module output each equal the trace's checkpoint byte for byte; the
fixtures are written only then.

Recorded in the twin format (--out, a new directory):
  k6.aux.vae.conv.fixture.attn_to_{q,k,v,out}
                     the projections, in the convolution fixture's format: a 1x1x1 "convolution" over one row of N
                     positions (attrs ti = hi = 1, wi = N, ci = co = 512, k 1, no pad, linear: true, input_contiguous).
                     Outs: x [N][512] (the projection's input values), w [co][ci][1][1][1], bias, acc (F.linear without
                     the bias, on the same tensor layout), y (the module's output). For to_out (contiguous: addmm with
                     the cuBLASLt bias epilogue) acc is a different call (mm), so only its y is evidence. The native fixture compares the
                     production kernel with y under both bias modes: separate (at::linear on a non-contiguous input,
                     matmul then add_) and fused (addmm / the cuBLASLt bias epilogue).
  k6.aux.vae.attn.sdpa  outs q, k, v, out [N][512] and mask [N][N] (fp16), attrs scale, the SDPA backend flags; the
                     fixture of the next round's SDPA candidates
  notes  attn_isolate  reproduces (per checkpoint), deterministic, the layouts (contiguity, strides) of each input
         attn_kernels  torch.profiler's CUDA kernels of each projection, of the SDPA call and of the whole module
         attn_sdpa_backend  torch._fused_sdp_choice for the call, and each SDPBackend forced on it: runs, and whether its
                       output equals the call's bit for bit
Exit 0 when the module reproduces deterministically; otherwise 1 (and no fixture is written).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import AUX, QUALIFICATION, read_capture
from . import capture as C

MODULE = "decoder.mid_block.attentions.0"
RES_IN = "decoder.mid_block.resnets.0"


def rows(t):
    """(1, N, C) -> [N][C], a contiguous host copy."""
    return t.detach()[0].contiguous().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--trace", required=True, help="a capture with k6.aux.vae.trace.* (capture --vae-trace first-tile)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-thw", default="5,32,29")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--oracle-prefix", default="")
    a = ap.parse_args()
    bad = [v for v in C.REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"attn_isolate: refusing to run with {', '.join(bad)} set")
    out, trace = Path(a.out).absolute(), Path(a.trace)
    if out.exists():
        raise SystemExit(f"attn_isolate: {out} exists")
    interp = C.oracle_interpreter(a.oracle_prefix)
    import torch
    import torch.nn.functional as F
    from .vae_gn_isolate import kernels
    from ..rec import REC

    oracle = C.oracle_note(torch, interp)
    t, h, w = (int(v) for v in a.tile_thw.split(","))
    n = t * h * w
    ops, _ = read_capture(trace)
    by = {op["name"]: op for op in ops}

    def trace_x(name):
        op = by.get(f"{AUX}vae.trace.{name}")
        if op is None:
            raise SystemExit(f"attn_isolate: {trace} lacks k6.aux.vae.trace.{name}")
        return C.load_ref(torch, trace, op["outs"]["x"])

    K = C.upstream(Path(a.src).absolute())
    torch.set_float32_matmul_precision("high")  # factory.py, as capture.py
    cfg = C.pipeline_config(K, Path(a.model).absolute(), "sdpa", "none")
    dev = torch.device("cuda")
    vae = K.vae_video.build_vae(cfg.paths.vae, device=dev)
    mods = dict(vae.named_modules())
    attn = mods[MODULE]
    if attn.heads != 1:
        raise SystemExit(f"attn_isolate: {MODULE} has {attn.heads} heads; the SDPA fixture is written for one")
    hv = sys.modules[type(mods["decoder.mid_block"]).__module__]  # hunyuan_vae: prepare_causal_attention_mask

    x2 = trace_x(RES_IN)
    if x2.shape[0] != n:
        raise SystemExit(f"attn_isolate: {RES_IN} has {x2.shape[0]} positions, the tile {n}")
    x5 = x2.reshape(t, h, w, -1).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dev)
    hidden = x5.permute(0, 2, 3, 4, 1).flatten(1, 3)  # as HunyuanVideoMidBlock3D.forward
    mask = hv.prepare_causal_attention_mask(t, h * w, hidden.dtype, hidden.device, 1)

    seen = {}
    sdpa_orig = F.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kw):
        o = sdpa_orig(q, k, v, *args, **kw)
        seen["sdpa"] = (q, k, v, kw.get("attn_mask", args[0] if args else None), o, dict(kw))
        return o

    hooks = []
    for name in ("group_norm", "to_q", "to_k", "to_v", "to_out.0"):
        sub = dict(attn.named_modules())[name]

        def f(_m, inp, o, name=name):
            seen[name] = (inp[0], o)
        hooks.append(sub.register_forward_hook(f))
    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, trace=str(trace), module=MODULE, tile_thw=[t, h, w])
    REC.note("device", **C.device_note(torch))
    REC.note("precision", **C.precision_note(torch))
    with torch.no_grad():
        F.scaled_dot_product_attention = sdpa
        try:
            runs = []
            for _ in range(max(1, a.repeat)):
                y = attn(hidden, attention_mask=mask)
                torch.cuda.synchronize()
                runs.append(rows(y))
            got = dict(seen)
        finally:
            F.scaled_dot_product_attention = sdpa_orig
            for hk in hooks:
                hk.remove()
        deterministic = all(torch.equal(r, runs[0]) for r in runs)
        # the trace's checkpoints: group_norm (its (B, C, N) output as [N][C]), the projections, the module output
        checks = {
            "group_norm": got["group_norm"][1].detach()[0].transpose(0, 1).contiguous().cpu(),
            "to_q": rows(got["to_q"][1]), "to_k": rows(got["to_k"][1]), "to_v": rows(got["to_v"][1]),
            "": runs[0],
        }
        reproduces = {}
        for key, val in checks.items():
            ref = trace_x(f"{MODULE}.{key}" if key else MODULE)
            reproduces[key or "output"] = bool(val.shape == ref.shape and torch.equal(val, ref))
        layouts = {k: {"shape": list(v[0].shape), "stride": list(v[0].stride()), "contiguous": v[0].is_contiguous()}
                   for k, v in got.items() if k != "sdpa"}
        q, k_, v_, m_, o_, kw = got["sdpa"]
        plain = {kk: (vv if isinstance(vv, (int, float, bool, str)) or vv is None else str(type(vv)))
                 for kk, vv in kw.items() if kk != "attn_mask"}
        layouts["sdpa"] = {"q": [list(q.shape), list(q.stride())], "kwargs": plain,
                           "mask": None if m_ is None else [list(m_.shape), str(m_.dtype)]}
        ok = deterministic and all(reproduces.values())
        be = torch.backends.cuda
        REC.note("attn_isolate", processor=type(attn.processor).__name__, reproduces=reproduces,
                 deterministic=bool(deterministic), layouts=layouts,
                 sdpa_flags={"flash": be.flash_sdp_enabled(), "mem_efficient": be.mem_efficient_sdp_enabled(),
                             "math": be.math_sdp_enabled(), "cudnn": be.cudnn_sdp_enabled()})
        # the SDPA backend: torch's own choice for these inputs, and each backend forced on the recorded call (one that
        # runs and returns the call's bits is a candidate identity; the profiled kernel below names the one used)
        backends = {}
        try:
            backends["fused_sdp_choice"] = int(torch._fused_sdp_choice(q, k_, v_, m_, kw.get("dropout_p", 0.0), kw.get("is_causal", False)))
        except Exception as e:  # private API: record why, never fail the isolate on it
            backends["fused_sdp_choice"] = f"unavailable: {type(e).__name__}: {e}"
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            for be_name in ("EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION", "MATH"):
                be_ = getattr(SDPBackend, be_name, None)
                if be_ is None:
                    continue
                try:
                    with sdpa_kernel([be_]):
                        forced = sdpa_orig(q, k_, v_, **{**kw, "attn_mask": m_})
                    torch.cuda.synchronize()
                    backends[be_name] = {"runs": True, "equal_to_call": bool(torch.equal(forced.view(torch.int16), o_.view(torch.int16))), "enum": int(be_)}
                except Exception as e:
                    backends[be_name] = {"runs": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        except Exception as e:
            backends["error"] = f"{type(e).__name__}: {e}"
        REC.note("attn_sdpa_backend", **backends)
        # kernels: each projection on its own input, the SDPA call, the whole module
        kk = {}
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            sub = dict(attn.named_modules())[name]
            kk[name] = kernels(torch, lambda sub=sub, xin=got[name][0]: sub(xin), out)
            (out / "profile.json").rename(out / f"profile-{name}.json")
        kk["sdpa"] = kernels(torch, lambda: sdpa_orig(q, k_, v_, **{**kw, "attn_mask": m_}), out)
        (out / "profile.json").rename(out / "profile-sdpa.json")
        kk["module"] = kernels(torch, lambda: attn(hidden, attention_mask=mask), out)
        (out / "profile.json").rename(out / "profile-module.json")
        REC.note("attn_kernels", kernels=kk)
        if ok:
            for name, tag in (("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out.0", "out")):
                sub = dict(attn.named_modules())[name]
                xin, yo = got[name]
                acc = F.linear(xin, sub.weight, None)  # the same layout, no bias
                ci, co = sub.in_features, sub.out_features
                geo = {"ti": 1, "hi": 1, "wi": n, "ci": ci, "co": co, "kt": 1, "kh": 1, "kw": 1, "pt": 0, "ph": 0, "pw": 0,
                       "up_t": 1, "up_hw": 1, "to": 1, "ho": 1, "wo": n}
                with REC.op(f"{AUX}vae.conv.fixture.attn_to_{tag}", "vae", {**geo, "module": f"{MODULE}.{name}", "linear": True,
                                                                             "input_contiguous": xin.is_contiguous()}) as op:
                    op.out(x=rows(xin), w=sub.weight.detach().reshape(co, ci, 1, 1, 1).contiguous(), bias=sub.bias.detach(),
                           acc=rows(acc), y=rows(yo))
            with REC.op(f"{AUX}vae.attn.sdpa", "vae", {"scale": kw.get("scale"), "dropout_p": kw.get("dropout_p", 0.0),
                                                        "is_causal": kw.get("is_causal", False), "heads": attn.heads}) as op:
                op.out(q=q[0, 0].contiguous(), k=k_[0, 0].contiguous(), v=v_[0, 0].contiguous(),
                       mask=m_.reshape(m_.shape[-2], m_.shape[-1]).contiguous(), out=o_[0, 0].contiguous())
    REC.note("versions", **C.versions_note(torch))
    REC.stop()
    print(json.dumps({"k6_vae_attn_isolate": str(out), "qualification": QUALIFICATION, "reproduces": reproduces,
                      "deterministic": bool(deterministic)}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
