"""Writes (--write) or checks (default) kernels/cuda/minimax/kitchen: copies of comfy-kitchen v0.2.35's device code for
the two ops MiniMax H3 uses, `int8_attention` and `rms_rope_split_half_`, read at the git tag and never from the
working tree.

Same lines and namespaces (so the same SASS as the wheel); only the host launchers are cut. The unnamed namespaces of
the three .cu files become named ones: an unnamed namespace's mangled names carry a hash of the build, so no fixed
symbol could name a kernel (the same rule as tools/kernels/sync.py). Nothing else changes in the device code. Each
copy starts with one line naming its source, the tag and the source's git blob hash. kernels/cuda/minimax/kitchen_launch.cu
holds our own host entry points.

usage: python3 tools/kernels/sync_kitchen.py [--write] [--repo <comfy-kitchen>] [--ref v0.2.35]
       python3 tools/kernels/sync_kitchen.py --sources      (prints "copy source-path")
Prints one JSON summary; exit status 1 when a copy differs (check mode).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kernels" / "cuda" / "minimax" / "kitchen"
REPO = os.environ.get("CK_PY", str(ROOT.parent / "ref" / "comfy-kitchen"))  # a read-only clone of the upstream
REF = "v0.2.35"
CU = "comfy_kitchen/backends/cuda"
SAGE = f"{CU}/sage_attention"

# name -> (source path, kept 1-based inclusive line ranges, {line: replacement}, what was cut)
# `None` ranges: the whole file.
UNNAMED = "namespace {\n"
UNNAMED_END = "} // namespace\n"
COPIES = {
    # the quantizers and the attention kernel (int8_attention)
    "quant_qk_int8.cu": (f"{SAGE}/quant_qk_int8.cu", ((1, 689),),
                         {31: "namespace kitchen_quant_qk {\n", 689: "} // namespace kitchen_quant_qk\n"},
                         "host launcher cut, unnamed namespace -> kitchen_quant_qk"),
    "quant_v_int8.cu": (f"{SAGE}/quant_v_int8.cu", ((1, 173),),
                        {27: "namespace kitchen_quant_v {\n", 173: "} // namespace kitchen_quant_v\n"},
                        "host launcher cut, unnamed namespace -> kitchen_quant_v"),
    "qk_int_sv_i8_cuda.cuh": (f"{SAGE}/qk_int_sv_i8_cuda.cuh", None, {}, "whole file"),
    "attn_utils.cuh": (f"{SAGE}/attn_utils.cuh", None, {}, "whole file"),
    "cp_async.cuh": (f"{SAGE}/cp_async.cuh", None, {}, "whole file"),
    "math.cuh": (f"{SAGE}/math.cuh", None, {}, "whole file"),
    "mma.cuh": (f"{SAGE}/mma.cuh", None, {}, "whole file"),
    "numeric_conversion.cuh": (f"{SAGE}/numeric_conversion.cuh", None, {}, "whole file"),
    "permuted_smem.cuh": (f"{SAGE}/permuted_smem.cuh", None, {}, "whole file"),
    "float_utils.cuh": (f"{CU}/float_utils.cuh", None, {}, "whole file"),
    "dtype_dispatch.cuh": (f"{CU}/dtype_dispatch.cuh", None, {}, "whole file"),
    # the RoPE kernel (rms_rope_split_half_)
    "rms_rope.cu": (f"{CU}/ops/rms_rope.cu", ((1, 229), (326, 327)),
                    {31: "namespace kitchen_rms_rope {\n", 326: "} // namespace kitchen_rms_rope\n"},
                    "host launchers cut, unnamed namespace -> kitchen_rms_rope"),
    "rope_device.cuh": (f"{CU}/ops/rope_device.cuh", None, {}, "whole file"),
    "utils.cuh": (f"{CU}/utils.cuh", None, {}, "whole file"),
}


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read(repo: str, ref: str, rel: str) -> bytes:
    """A file's bytes at `ref`, never the working tree."""
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{rel}"], check=True, capture_output=True).stdout


def render(repo: str, ref: str, name: str) -> tuple[str, str]:
    """(the copy's text, the source's blob hash)."""
    rel, ranges, replace, what = COPIES[name]
    data = read(repo, ref, rel)
    src = data.decode().splitlines(keepends=True)
    out = [f"// {rel} @ {ref} (git blob {blob_hash(data)[:12]}; {what}); copied by tools/kernels/sync_kitchen.py, do not edit\n"]
    for a, b in ranges or ((1, len(src)),):
        for n in range(a, b + 1):
            if n in replace:
                # the line being replaced is the unnamed namespace (or its end): anything else means the tag moved
                assert src[n - 1] in (UNNAMED, UNNAMED_END), f"{rel}:{n}: {src[n - 1]!r}"
                out.append(replace[n])
            else:
                out.append(src[n - 1])
    return "".join(out), blob_hash(data)


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
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
    files, bad = [], 0
    for name, (rel, *_rest) in COPIES.items():
        text, blob = render(args.repo, args.ref, name)
        path = OUT / name
        if args.write:
            path.write_text(text)
            state = "wrote"
        elif path.exists() and path.read_text() == text:
            state = "ok"
        else:
            state = "differs"
            bad += 1
        files.append({"copy": name, "source": rel, "blob": blob[:12], "state": state})
    print(json.dumps({"ref": args.ref, "ok": bad == 0, "files": files}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
