"""python -I -m stk_twin.kandinsky6.vae_gn_isolate --model DIR --src KANDINSKY_SRC --trace REF_TRACE --out DIR
       [--module decoder.mid_block.resnets.0] [--norm norm1] [--input decoder.conv_in] [--want ...nonlinearity.0]
       [--thw 5,32,29] [--repeat 3] [--oracle-prefix /opt/k6-oracle] [--sass]

One torch GroupNorm of the Kandinsky-6 video VAE, isolated (diagnostic, unqualified). The pinned upstream VAE is built as
capture.py's vae stage builds it (fp16, the snapshot's weights). Its `--module`'s `--norm` (an nn.GroupNorm) and the
resnet's own SiLU module run on a recorded input. That input is a first-tile trace capture's (`capture --vae-trace
first-tile`) `k6.aux.vae.trace.<--input>`, [positions][channels] channels last, rebuilt into the module's NCTHW layout
with --thw. Nothing here changes torch's arithmetic: the modules are called as the decoder calls them.

Recorded in the twin format (--out, a new directory):
  k6.aux.vae.gn.fixture   outs x (the input, [P][C] channels last), gamma, beta (the module's fp16 parameters, [C]),
                          norm (the GroupNorm output) and silu (the SiLU of it), both [P][C] channels last
  notes  gn_isolate       the module and its settings (groups, eps, affine, dtypes); `reproduces`: silu equals the
                          trace's `k6.aux.vae.trace.<--want>` byte for byte, i.e. the isolate is the in-situ computation
                          (a fixture that does not reproduce is refused: exit 1); `deterministic` over --repeat runs;
                          torch.native_group_norm's fp16 mean / rstd
         gn_kernels       torch.profiler's CUDA kernels of one norm + SiLU call: name, grid, block (chrome trace in
                          profile.json)
         gn_sass          with --sass: cuobjdump -sass of libtorch_cuda.so's RowwiseMomentsCUDAKernel,
                          ComputeFusedParamsCUDAKernel, GroupNormKernelImplInternal (the apply's `a * x + b`
                          lambda) and silu kernels (the arch the device loads), their opcode
                          counts (FFMA vs FMUL+FADD) and the text in sass/ (bounded: one pass, 1,800 s)
  oracle, device, versions  as capture.py records them
Exit 0 when the fixture reproduces deterministically; otherwise 1, and no fixture op is written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

from . import AUX, QUALIFICATION, read_capture
from . import capture as C

SASS_FUNCS = re.compile(r"RowwiseMomentsCUDAKernel|ComputeFusedParamsCUDAKernel|GroupNormKernelImplInternal|silu")


def channels_last(t):
    """(1, C, T, H, W) -> [T*H*W][C] contiguous."""
    return t.detach().permute(0, 2, 3, 4, 1).reshape(-1, t.shape[1]).contiguous()


def sass(torch, out: Path) -> dict:
    """The device's SASS for torch's GroupNorm and SiLU kernels, from libtorch_cuda.so (best effort, bounded)."""
    tool = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    lib = Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"
    if not Path(tool).exists() or not lib.is_file():
        return {"status": f"skipped: cuobjdump {tool} or {lib} missing"}
    major, minor = torch.cuda.get_device_capability()
    try:
        elfs = subprocess.run([tool, "-lelf", str(lib)], capture_output=True, text=True, timeout=600).stdout
    except subprocess.TimeoutExpired:
        return {"status": "failed: cuobjdump -lelf timed out"}
    archs = sorted(set(re.findall(r"sm_(\d+)", elfs)), key=int)
    dev = major * 10 + minor
    usable = [int(a) for a in archs if int(a) // 10 == major and int(a) <= dev]
    if not usable:
        return {"status": "no SASS for this major: PTX JIT", "archs": archs, "device": f"sm_{dev}"}
    arch = f"sm_{max(usable)}"
    d = out / "sass"
    d.mkdir()
    funcs, cur, buf = {}, None, []
    p = subprocess.Popen([tool, "-sass", "-arch", arch, str(lib)], stdout=subprocess.PIPE, text=True, errors="replace")
    timer = threading.Timer(1800, p.kill)  # the bound: the read loop below blocks until cuobjdump exits
    timer.start()
    try:
        for line in p.stdout:
            m = re.match(r"\s*Function : (\S+)", line)
            if m:
                if cur:
                    funcs.setdefault(cur, []).extend(buf)
                cur, buf = (m.group(1) if SASS_FUNCS.search(m.group(1)) else None), []
            elif cur:
                buf.append(line)
        if cur:
            funcs.setdefault(cur, []).extend(buf)
        p.wait()
    finally:
        timer.cancel()
    if p.returncode != 0:
        return {"status": f"failed: cuobjdump -sass exit {p.returncode} (killed after 1,800 s if negative)", "arch": arch}
    summary = {}
    for i, (name, lines) in enumerate(sorted(funcs.items())):
        (d / f"{i:03d}.sass").write_text(name + "\n" + "".join(lines))
        ops = Counter(m.group(1).split(".")[0] for ln in lines if (m := re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", ln)))
        summary[name] = {"file": f"sass/{i:03d}.sass", "ops": {k: ops[k] for k in ("FFMA", "FMUL", "FADD", "MUFU", "SHFL") if ops[k]}}
    return {"status": "ok", "arch": arch, "archs_in_lib": archs, "device": f"sm_{dev}", "functions": summary}


def kernels(torch, fn, out: Path) -> list:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = out / "profile.json"
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text()).get("traceEvents", [])
    return [{"name": e.get("name"), "grid": e.get("args", {}).get("grid"), "block": e.get("args", {}).get("block"),
             "registers": e.get("args", {}).get("registers per thread"),
             "shared_memory": e.get("args", {}).get("shared memory"), "dur_us": e.get("dur")}
            for e in events if e.get("cat") == "kernel"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--trace", required=True, help="a capture with k6.aux.vae.trace.* (capture --vae-trace first-tile)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--module", default="decoder.mid_block.resnets.0")
    ap.add_argument("--norm", default="norm1", choices=("norm1", "norm2"))
    ap.add_argument("--input", default="decoder.conv_in", help="the trace checkpoint the norm reads")
    ap.add_argument("--want", default="decoder.mid_block.resnets.0.nonlinearity.0", help="the trace's SiLU output")
    ap.add_argument("--thw", default="5,32,29", help="the input's T,H,W (positions = T*H*W)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--oracle-prefix", default="")
    ap.add_argument("--sass", action="store_true")
    a = ap.parse_args()
    bad = [v for v in C.REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"gn_isolate: refusing to run with {', '.join(bad)} set")
    out, trace = Path(a.out).absolute(), Path(a.trace)
    if out.exists():
        raise SystemExit(f"gn_isolate: {out} exists")
    interp = C.oracle_interpreter(a.oracle_prefix)
    import torch
    from ..rec import REC

    oracle = C.oracle_note(torch, interp)
    t, h, w = (int(v) for v in a.thw.split(","))
    ops, _ = read_capture(trace)
    by = {op["name"]: op for op in ops}
    xin, want = by.get(f"{AUX}vae.trace.{a.input}"), by.get(f"{AUX}vae.trace.{a.want}")
    if xin is None or want is None:
        raise SystemExit(f"gn_isolate: {trace} lacks the trace ops {a.input} / {a.want}")
    K = C.upstream(Path(a.src).absolute())
    torch.set_float32_matmul_precision("high")  # factory.py, as capture.py
    cfg = C.pipeline_config(K, Path(a.model).absolute(), "sdpa", "none")
    dev = torch.device("cuda")
    vae = K.vae_video.build_vae(cfg.paths.vae, device=dev)
    res = dict(vae.named_modules())[a.module]
    norm, act = getattr(res, a.norm), res.nonlinearity
    if not isinstance(norm, torch.nn.GroupNorm) or norm.weight.dtype != torch.float16:
        raise SystemExit(f"gn_isolate: {a.module}.{a.norm} is {type(norm).__name__} {norm.weight.dtype}")
    x2 = C.load_ref(torch, trace, xin["outs"]["x"])  # [P][C] fp16
    if x2.dtype != torch.float16 or x2.shape[0] != t * h * w:
        raise SystemExit(f"gn_isolate: input {tuple(x2.shape)} {x2.dtype} vs thw {t, h, w}")
    x = x2.reshape(t, h, w, -1).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dev)
    want_cl = C.load_ref(torch, trace, want["outs"]["x"])

    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, trace=str(trace), module=a.module, norm=a.norm,
             input=a.input, want=a.want, thw=[t, h, w])
    REC.note("device", **C.device_note(torch))
    with torch.no_grad():
        runs = []
        for _ in range(max(1, a.repeat)):
            y = norm(x)
            s = act(y)
            torch.cuda.synchronize()
            runs.append((channels_last(y).cpu(), channels_last(s).cpu()))
        y_cl, s_cl = runs[0]
        deterministic = all(torch.equal(r[0], y_cl) and torch.equal(r[1], s_cl) for r in runs)
        reproduces = s_cl.dtype == want_cl.dtype and s_cl.shape == want_cl.shape and torch.equal(s_cl, want_cl)
        diff = int((s_cl.view(torch.int16) != want_cl.view(torch.int16)).sum()) if s_cl.shape == want_cl.shape else None
        _, mean, rstd = torch.native_group_norm(x, norm.weight, norm.bias, 1, x.shape[1], t * h * w, norm.num_groups,
                                                norm.eps)
        REC.note("gn_isolate", reproduces=bool(reproduces), unequal_vs_trace=diff, deterministic=bool(deterministic),
                 repeat=len(runs), groups=norm.num_groups, eps=norm.eps, affine=norm.affine,
                 weight_dtype=str(norm.weight.dtype), input_dtype=str(x.dtype), input_shape=list(x.shape),
                 input_contiguous=x.is_contiguous(), silu=type(act).__name__,
                 mean_fp16=mean.float().cpu().view(-1).tolist(), rstd_fp16=rstd.float().cpu().view(-1).tolist())
        REC.note("gn_kernels", kernels=kernels(torch, lambda: act(norm(x)), out))
        if a.sass:
            REC.note("gn_sass", **sass(torch, out))
        if not (reproduces and deterministic):  # a fixture that is not the in-situ computation is never written
            REC.stop()
            print(json.dumps({"k6_vae_gn_isolate": str(out), "reproduces": bool(reproduces),
                              "deterministic": bool(deterministic), "unequal_vs_trace": diff}))
            sys.exit(1)
        with REC.op(AUX + "vae.gn.fixture", "vae", {"module": f"{a.module}.{a.norm}", "groups": norm.num_groups,
                                                    "note_eps": norm.eps, "note_thw": [t, h, w]}) as o:
            o.out(x=x2.contiguous(), gamma=norm.weight.detach(), beta=norm.bias.detach(), norm=y_cl, silu=s_cl)
    REC.note("versions", **C.versions_note(torch))
    REC.stop()
    print(json.dumps({"k6_vae_gn_isolate": str(out), "qualification": QUALIFICATION, "reproduces": bool(reproduces),
                      "deterministic": bool(deterministic), "unequal_vs_trace": diff}))
    sys.exit(0 if reproduces and deterministic else 1)


if __name__ == "__main__":
    main()
