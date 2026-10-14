"""python -I -m stk_twin.kandinsky6.vae_conv_isolate --model DIR --src KANDINSKY_SRC --trace REF_TRACE --out DIR
       [--tile-thw 5,32,29] [--targets conv_in,mid_conv,...] [--oracle-prefix /opt/k6-oracle] [--sass]

The Kandinsky-6 video VAE decoder's convolutions, isolated (diagnostic, unqualified): one per distinct shape group.
Every decoder conv of the first tile has its exact input and output in a first-tile trace capture (`capture --vae-trace
first-tile`, [positions][channels] channels last), so each target runs the pinned upstream module that holds the conv
(the HunyuanVideoCausalConv3d itself, or the HunyuanVideoUpsampleCausal3D around it) on the recorded input. Its output
must equal the trace's byte for byte: the isolate is then the in-situ computation. A forward hook on the inner nn.Conv3d
reads the tensor cudnn convolved (padded, upsampled) and its output; `acc` is the same F.conv3d without the bias (torch's
_convolution runs cudnn_convolution then output.add_(bias)), and `bias_separate` checks y == acc.add_(bias).

Targets (the decoder's shapes; a 5 x 32 x 29 latent tile, all 112 passes of the reference plan are this tile):
@TARGETS@

Recorded in the twin format (--out, a new directory), per reproducing target:
  k6.aux.vae.conv.fixture.<target>  attrs ti, hi, wi, ci, co, kt, kh, kw, pt, ph, pw, up_t, up_hw, to, ho, wo (the
                          production gather's geometry), module, input, output; outs x [ti*hi*wi][ci], w [co][ci][kt][kh][kw]
                          (the module's fp16 weight), bias [co], acc and y [to*ho*wo][co], all fp16, channels last
  notes  conv_isolate     per target: reproduces, unequal_vs_trace, deterministic, bias_separate, shapes
         conv_kernels     per target: torch.profiler's CUDA kernels of one module call (name, grid, block, registers,
                          shared memory) and cudnn's version; the chrome trace in profile-<target>.json
         conv_sass        with --sass: cuobjdump -sass of the cudnn libraries' functions with those kernel names (the
                          device's arch), their HMMA / FFMA counts, text in sass/ (bounded: 1,800 s a library)
Exit 0 when every target reproduces deterministically; otherwise 1 (only reproducing targets are written).
"""

from __future__ import annotations

import argparse
import glob
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

# (target, the HunyuanVideoCausalConv3d, the module called, its input checkpoint, its output checkpoint, resolution level)
TARGETS = (
    ("conv_in", "decoder.conv_in", "decoder.conv_in", "post_quant_conv", "decoder.conv_in", 0),
    ("mid_conv", "decoder.mid_block.resnets.0.conv1", "decoder.mid_block.resnets.0.conv1",
     "decoder.mid_block.resnets.0.nonlinearity.0", "decoder.mid_block.resnets.0.conv1", 0),
    ("up0_upsample", "decoder.up_blocks.0.upsamplers.0.conv", "decoder.up_blocks.0.upsamplers.0",
     "decoder.up_blocks.0.resnets.2", "decoder.up_blocks.0.upsamplers.0", 0),
    ("up1_conv", "decoder.up_blocks.1.resnets.0.conv1", "decoder.up_blocks.1.resnets.0.conv1",
     "decoder.up_blocks.1.resnets.0.nonlinearity.0", "decoder.up_blocks.1.resnets.0.conv1", 1),
    ("up1_upsample", "decoder.up_blocks.1.upsamplers.0.conv", "decoder.up_blocks.1.upsamplers.0",
     "decoder.up_blocks.1.resnets.2", "decoder.up_blocks.1.upsamplers.0", 1),
    ("up2_conv_in", "decoder.up_blocks.2.resnets.0.conv1", "decoder.up_blocks.2.resnets.0.conv1",
     "decoder.up_blocks.2.resnets.0.nonlinearity.0", "decoder.up_blocks.2.resnets.0.conv1", 2),
    ("up2_shortcut", "decoder.up_blocks.2.resnets.0.conv_shortcut", "decoder.up_blocks.2.resnets.0.conv_shortcut",
     "decoder.up_blocks.1.upsamplers.0", "decoder.up_blocks.2.resnets.0.conv_shortcut", 2),
    ("up2_conv", "decoder.up_blocks.2.resnets.1.conv1", "decoder.up_blocks.2.resnets.1.conv1",
     "decoder.up_blocks.2.resnets.1.nonlinearity.0", "decoder.up_blocks.2.resnets.1.conv1", 2),
    ("up2_upsample", "decoder.up_blocks.2.upsamplers.0.conv", "decoder.up_blocks.2.upsamplers.0",
     "decoder.up_blocks.2.resnets.2", "decoder.up_blocks.2.upsamplers.0", 2),
    ("up3_conv_in", "decoder.up_blocks.3.resnets.0.conv1", "decoder.up_blocks.3.resnets.0.conv1",
     "decoder.up_blocks.3.resnets.0.nonlinearity.0", "decoder.up_blocks.3.resnets.0.conv1", 3),
    ("up3_shortcut", "decoder.up_blocks.3.resnets.0.conv_shortcut", "decoder.up_blocks.3.resnets.0.conv_shortcut",
     "decoder.up_blocks.2.upsamplers.0", "decoder.up_blocks.3.resnets.0.conv_shortcut", 3),
    ("up3_conv", "decoder.up_blocks.3.resnets.1.conv1", "decoder.up_blocks.3.resnets.1.conv1",
     "decoder.up_blocks.3.resnets.1.nonlinearity.0", "decoder.up_blocks.3.resnets.1.conv1", 3),
    ("conv_out", "decoder.conv_out", "decoder.conv_out", "decoder.conv_act", "decoder.conv_out", 3),
)
__doc__ = __doc__.replace("@TARGETS@", "\n".join(f"  {t[0]:14} {t[1]} (input {t[3]})" for t in TARGETS), 1)
CONV_KERNEL = re.compile(r"fprop|conv|xmma|cutlass|implicit|gemm|winograd|fft", re.I)


def level_thw(t0: int, h0: int, w0: int, level: int) -> tuple[int, int, int]:
    """The decoder's activation dims at an up-block level: up0 doubles space, up1 and up2 also time (first frame
    once: 2T - 1)."""
    t = t0 if level < 2 else (2 * t0 - 1 if level == 2 else 2 * (2 * t0 - 1) - 1)
    return t, h0 << level, w0 << level


def channels_last(t):
    return t.detach().permute(0, 2, 3, 4, 1).reshape(-1, t.shape[1]).contiguous()


def cudnn_libs(torch) -> list[str]:
    base = Path(torch.__file__).parent
    pats = [base / "lib" / "libcudnn*.so*", base.parent / "nvidia" / "cudnn" / "lib" / "libcudnn*.so*"]
    return sorted({os.path.realpath(p) for pat in pats for p in glob.glob(str(pat))})


def sass(torch, out: Path, names: set[str]) -> dict:
    """cuobjdump -sass of the cudnn libraries' functions named like the kernels the profiler saw (best effort)."""
    tool = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    if not Path(tool).exists():
        return {"status": f"skipped: no {tool}"}
    major, minor = torch.cuda.get_device_capability()
    dev = major * 10 + minor
    d = out / "sass"
    d.mkdir(exist_ok=True)
    found, libs = {}, []
    # profiler names are demangled (`void cutlass::...<...>(...)`), cuobjdump's mangled: match on the kernels' own
    # identifiers (cutlass_tensorop_..., sm80_xmma_..., and the like), which appear verbatim in both
    want = {w for n in names for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{11,}", n)
            if re.search(r"fprop|xmma|cutlass_tensorop|implicit|conv|winograd|fft|gemm", w, re.I)}
    for lib in cudnn_libs(torch):
        try:
            elfs = subprocess.run([tool, "-lelf", lib], capture_output=True, text=True, timeout=600).stdout
        except subprocess.TimeoutExpired:
            libs.append({"lib": lib, "status": "lelf timed out"})
            continue
        archs = sorted({int(a) for a in re.findall(r"sm_(\d+)", elfs)})
        usable = [a for a in archs if a // 10 == major and a <= dev]
        if not usable:
            libs.append({"lib": lib, "status": "no SASS for this major", "archs": archs})
            continue
        arch = f"sm_{max(usable)}"
        p = subprocess.Popen([tool, "-sass", "-arch", arch, lib], stdout=subprocess.PIPE, text=True, errors="replace")
        timer = threading.Timer(1800, p.kill)
        timer.start()
        cur, buf = None, []
        try:
            for line in p.stdout:
                m = re.match(r"\s*Function : (\S+)", line)
                if m:
                    if cur:
                        found.setdefault(cur, []).extend(buf)
                    cur, buf = (m.group(1) if any(w and w in m.group(1) for w in want) else None), []
                elif cur:
                    buf.append(line)
            if cur:
                found.setdefault(cur, []).extend(buf)
            p.wait()
        finally:
            timer.cancel()
        libs.append({"lib": lib, "arch": arch, "exit": p.returncode})
    funcs = {}
    for i, (name, lines) in enumerate(sorted(found.items())):
        (d / f"conv-{i:03d}.sass").write_text(name + "\n" + "".join(lines))
        ops = Counter(m.group(1) for ln in lines
                      if (m := re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9]+)*)", ln)))
        funcs[name] = {"file": f"sass/conv-{i:03d}.sass",
                       "mma": {k: v for k, v in ops.items() if k.startswith(("HMMA", "IMMA", "QMMA", "UTC"))},
                       "ffma": ops.get("FFMA", 0), "fadd": ops.get("FADD", 0)}
    return {"status": "ok", "device": f"sm_{dev}", "libs": libs, "functions": funcs, "wanted": sorted(want)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--trace", required=True, help="a capture with k6.aux.vae.trace.* (capture --vae-trace first-tile)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-thw", default="5,32,29", help="the latent tile's T,H,W")
    ap.add_argument("--targets", default=",".join(t[0] for t in TARGETS))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--oracle-prefix", default="")
    ap.add_argument("--sass", action="store_true")
    a = ap.parse_args()
    bad = [v for v in C.REFUSED_ENV if v in os.environ]
    if bad:
        raise SystemExit(f"conv_isolate: refusing to run with {', '.join(bad)} set")
    out, trace = Path(a.out).absolute(), Path(a.trace)
    if out.exists():
        raise SystemExit(f"conv_isolate: {out} exists")
    want_targets = [t for t in TARGETS if t[0] in set(a.targets.split(","))]
    if len(want_targets) != len(set(a.targets.split(","))):
        raise SystemExit(f"conv_isolate: unknown targets in {a.targets}; known {[t[0] for t in TARGETS]}")
    interp = C.oracle_interpreter(a.oracle_prefix)
    import torch
    import torch.nn.functional as F
    from .vae_gn_isolate import kernels
    from ..rec import REC

    oracle = C.oracle_note(torch, interp)
    t0, h0, w0 = (int(v) for v in a.tile_thw.split(","))
    ops, _ = read_capture(trace)
    by = {op["name"]: op for op in ops}
    K = C.upstream(Path(a.src).absolute())
    torch.set_float32_matmul_precision("high")  # factory.py, as capture.py
    cfg = C.pipeline_config(K, Path(a.model).absolute(), "sdpa", "none")
    dev = torch.device("cuda")
    vae = K.vae_video.build_vae(cfg.paths.vae, device=dev)
    mods = dict(vae.named_modules())

    out.parent.mkdir(parents=True, exist_ok=True)
    REC.start(out)
    REC.note("oracle", **oracle)
    REC.note("request", qualification=QUALIFICATION, argv=sys.argv, trace=str(trace), tile_thw=[t0, h0, w0],
             targets=[t[0] for t in want_targets])
    REC.note("device", **C.device_note(torch))
    REC.note("precision", **C.precision_note(torch))
    failed, kernel_names = [], set()
    with torch.no_grad():
        for name, conv_path, outer_path, in_name, out_name, level in want_targets:
            xin, yout = by.get(f"{AUX}vae.trace.{in_name}"), by.get(f"{AUX}vae.trace.{out_name}")
            if xin is None or yout is None:
                raise SystemExit(f"conv_isolate: {trace} lacks k6.aux.vae.trace.{in_name} / {out_name}")
            causal, outer = mods[conv_path], mods[outer_path]
            conv = causal.conv
            ti, hi, wi = level_thw(t0, h0, w0, level)
            x2 = C.load_ref(torch, trace, xin["outs"]["x"])  # [P][C] fp16
            if x2.dtype != torch.float16 or x2.shape[0] != ti * hi * wi or x2.shape[1] != conv.in_channels:
                raise SystemExit(f"conv_isolate: {name}: input {tuple(x2.shape)} {x2.dtype} vs {ti, hi, wi} x {conv.in_channels}")
            x = x2.reshape(ti, hi, wi, -1).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dev)
            want = C.load_ref(torch, trace, yout["outs"]["x"])
            seen = {}

            def hook(_m, inp, o, seen=seen):
                seen["in"], seen["out"] = inp[0], o
            h = conv.register_forward_hook(hook)
            try:
                runs = []
                for _ in range(max(1, a.repeat)):
                    y = outer(x)
                    torch.cuda.synchronize()
                    runs.append(channels_last(y).cpu())
                    del y
                cin, cout = seen["in"], seen["out"]  # the last checked call's
            finally:
                h.remove()
            kern = kernels(torch, lambda: outer(x), out)
            (out / "profile.json").rename(out / f"profile-{name}.json")
            y_cl = runs[0]
            deterministic = all(torch.equal(r, y_cl) for r in runs)
            reproduces = y_cl.shape == want.shape and torch.equal(y_cl, want)
            diff = int((y_cl.view(torch.int16) != want.view(torch.int16)).sum()) if y_cl.shape == want.shape else None
            acc = F.conv3d(cin, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
            bias_separate = torch.equal(acc.clone().add_(conv.bias.view(1, -1, 1, 1, 1)), cout)
            pad = causal.time_causal_padding  # (w, w, h, h, t front, t back)
            tu, hu, wu = cin.shape[2] - pad[4] - pad[5], cin.shape[3] - pad[2] - pad[3], cin.shape[4] - pad[0] - pad[1]
            up_t = 2 if (ti > 1 and tu == 2 * ti - 1) else 1
            up_hw = hu // hi
            if (up_t == 1 and tu != ti) or hu != hi * up_hw or wu != wi * up_hw or up_hw not in (1, 2):
                raise SystemExit(f"conv_isolate: {name}: conv input {tuple(cin.shape)} is no upsample of {ti, hi, wi}")
            kt, kh, kw = conv.kernel_size
            to, ho, wo = acc.shape[2:]
            geo = {"ti": ti, "hi": hi, "wi": wi, "ci": conv.in_channels, "co": conv.out_channels, "kt": kt, "kh": kh,
                   "kw": kw, "pt": pad[4], "ph": pad[2], "pw": pad[0], "up_t": up_t, "up_hw": up_hw, "to": to, "ho": ho,
                   "wo": wo}
            REC.note("conv_isolate", target=name, reproduces=bool(reproduces), unequal_vs_trace=diff,
                     deterministic=bool(deterministic), bias_separate=bool(bias_separate), repeat=len(runs),
                     module=conv_path, called=outer_path, conv_input_shape=list(cin.shape),
                     conv_input_contiguous=cin.is_contiguous(), weight_dtype=str(conv.weight.dtype),
                     stride=list(conv.stride), padding=list(conv.padding), dilation=list(conv.dilation),
                     groups=conv.groups, geometry=geo)
            REC.note("conv_kernels", target=name, cudnn=torch.backends.cudnn.version(), kernels=kern)
            kernel_names.update(k["name"] for k in kern if k.get("name") and CONV_KERNEL.search(k["name"]))
            if not (reproduces and deterministic):
                failed.append(name)
                del x, x2, cin, cout, acc, runs
                seen.clear()
                torch.cuda.empty_cache()
                continue
            with REC.op(f"{AUX}vae.conv.fixture.{name}", "vae", {**geo, "module": conv_path, "input": in_name,
                                                                  "output": out_name}) as o:
                o.out(x=x2.contiguous(), w=conv.weight.detach().contiguous(), bias=conv.bias.detach(),
                      acc=channels_last(acc), y=y_cl)
            del x, x2, cin, cout, acc, runs, y_cl
            seen.clear()
            torch.cuda.empty_cache()
        if a.sass:
            REC.note("conv_sass", **sass(torch, out, kernel_names))
    REC.note("versions", **C.versions_note(torch))
    REC.stop()
    print(json.dumps({"k6_vae_conv_isolate": str(out), "qualification": QUALIFICATION, "failed": failed,
                      "targets": [t[0] for t in want_targets]}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
