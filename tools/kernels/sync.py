"""Writes (--write) or checks (default) kernels/cuda/nvfp4: copies of TensorFold v0.6.1's NVFP4 device code
(extension tensorfold_nvfp4_ck_v6), read at the git tag and never from the working tree.

Same lines and namespaces (so the same mangled names and SASS as the Python extension); only the torch/ATen includes
and the host wrappers are cut, and the explicit instantiations of every kernel configuration the wrappers can launch
are appended. Each copy starts with one line naming its source, the tag and the source's git blob hash.

usage: python3 tools/kernels/sync.py [--write] [--repo <tensorfold-py>] [--ref v0.6.1]
       python3 tools/kernels/sync.py --sources      (prints "copy source-path", for sass_check.sh)
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kernels" / "cuda" / "nvfp4"
REPO = os.environ.get("TF_PY", str(ROOT.parent / "ref" / "tensorfold-py"))  # a read-only clone of the upstream
REF = "v0.6.1"
NV = "src/tensorfold/cuda/nvfp4"
KN = "src/tensorfold/cuda/kernels"

CAST = ("const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, float, void*, int, int, int, int, int, int")


def footer_gemm_ck() -> str:
    out = ["// The instantiations gemm_cuda's by_tile launches (tile 0 resolves to 1 or 2; every other value is 1), at both\n"
           "// modes (A4 = 0, A8 = 1) and both outputs, and gemm_gu_ck_cuda's two epilogues (EPI 1: SwiGLU through bf16,\n"
           "// EPI 2: fp32).\n",
           "#define NVFP4_CK(MODE, BM, BN, WM, WN, ST, KS, F32, EPI) \\\n"
           "    template __global__ void nvfp4_gemm_ck::gemm_kernel<MODE, BM, BN, WM, WN, ST, KS, F32, EPI>( \\\n"
           f"        {CAST}, nvfp4_gemm_ck::Up);\n"]
    for mode in (0, 1):
        for f32 in ("true", "false"):
            out.append(f"NVFP4_CK({mode}, 128, 256, 2, 4, 2, 2, {f32}, 0)\n")                    # tile 2
            out.append(f"NVFP4_CK({mode}, 64, 128, 2, 4, 4, 1, {f32}, 0)\n")                     # tile 3
            out.append(f"NVFP4_CK({mode}, 128, 128, 2, 4, {4 if mode == 0 else 3}, 1, {f32}, 0)\n")  # tile 1 (default)
    out.append("NVFP4_CK(0, 128, 256, 2, 4, 3, 2, false, 2)\n")                                  # gemm_gu_ck, fp32
    out.append("NVFP4_CK(0, 128, 256, 2, 4, 3, 2, false, 1)\n")                                  # gemm_gu_ck, bf16
    return "".join(out)


def footer_gemm_ws() -> str:
    out = ["// The instantiations gemm_ws_cuda's by_tile launches (tile 2, 3, else 1) at both modes and both outputs.\n",
           "#define NVFP4_WS(MODE, BM, BN, ST, KS, F32) \\\n"
           "    template __global__ void nvfp4_gemm_ws::ws_kernel<MODE, BM, BN, ST, KS, F32>( \\\n"
           f"        {CAST});\n"]
    for mode in (0, 1):
        for f32 in ("true", "false"):
            out.append(f"NVFP4_WS({mode}, 128, 256, {3 if mode == 0 else 2}, 2, {f32})\n")     # tile 2
            out.append(f"NVFP4_WS({mode}, 128, 128, 3, 2, {f32})\n")                           # tile 3
            out.append(f"NVFP4_WS({mode}, 128, 256, {4 if mode == 0 else 3}, 1, {f32})\n")     # tile 1 (default)
    return "".join(out)


# The unnamed namespaces of the four .cu files become named ones: an unnamed namespace's mangled names carry a hash of
# the build (file path, command line), so no fixed symbol could name a kernel. Nothing else changes in the device code.
NAMESPACE = {"act.cu": "nvfp4_act", "gemm_ck.cu": "nvfp4_gemm_ck", "gemm_ws.cu": "nvfp4_gemm_ws", "lane4.cu": "nvfp4_lane4"}

# name -> (source path, kept 1-based inclusive line ranges, {line: replacement}, footer)
INC = '#include "qmm_frag.cuh"  // was "../kernels/qmm_frag.cuh"\n'
COPIES = {
    "act.cu": (f"{NV}/act.cu", ((1, 4), (6, 8), (10, 93)), {}, None),
    "gemm_ck.cu": (f"{NV}/gemm_ck.cu", ((1, 3), (6, 199), (237, 237)), {11: INC}, footer_gemm_ck),
    "gemm_ws.cu": (f"{NV}/gemm_ws.cu", ((1, 4), (7, 213), (249, 249)), {12: INC}, footer_gemm_ws),
    "lane4.cu": (f"{NV}/lane4.cu", ((1, 5), (8, 263), (330, 330), (355, 373)), {15: INC}, None),
    "mma4.cuh": (f"{NV}/mma4.cuh", ((1, 41),), {}, None),
    "nvfp4q.cuh": (f"{NV}/nvfp4q.cuh", ((1, 38),), {}, None),
    "swiglu4.cuh": (f"{NV}/swiglu4.cuh", ((1, 100),), {}, None),
    "qmm_frag.cuh": (f"{KN}/qmm_frag.cuh", ((1, 118),), {}, None),
}


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read(repo: str, ref: str, rel: str) -> bytes:
    """A file's bytes at `ref`, never the working tree."""
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{rel}"], check=True, capture_output=True).stdout


def render(repo: str, ref: str, name: str) -> str:
    rel, ranges, replace, footer = COPIES[name]
    data = read(repo, ref, rel)
    src = data.decode().splitlines(keepends=True)
    whole = ranges == ((1, len(src)),)
    what = "whole file" if whole else "torch includes and host wrappers cut"
    if name in NAMESPACE:
        what += f", unnamed namespace -> {NAMESPACE[name]}"
    out = [f"// {rel} @ {ref} (git blob {blob_hash(data)[:12]}; {what}); written by tools/kernels/sync.py, do not edit\n"]
    for a, b in ranges:
        for n in range(a, b + 1):
            line = replace.get(n, src[n - 1])
            if name in NAMESPACE and line == "namespace {\n":
                line = f"namespace {NAMESPACE[name]} {{\n"
            elif name in NAMESPACE and line == "}  // namespace\n":
                line = f"}}  // namespace {NAMESPACE[name]}\n"
            out.append(line)
    if footer:
        out.append("\n" + footer())
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--sources", action="store_true")
    args = ap.parse_args()
    if args.sources:
        for name, (rel, *_rest) in COPIES.items():
            print(name, rel)
        return 0
    bad = 0
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
    for name in COPIES:
        text = render(args.repo, args.ref, name)
        path = OUT / name
        if args.write:
            path.write_text(text)
            print(f"wrote {name}")
        elif not path.exists() or path.read_text() != text:
            print(f"DIFFERS {name}")
            bad += 1
        else:
            print(f"ok {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
