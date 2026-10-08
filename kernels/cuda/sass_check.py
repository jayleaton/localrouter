"""Compares cuobjdump -sass of TensorFold's NVFP4 extension object against our fatbin, kernel by kernel: every kernel
of ours must exist in the Python build with the same SASS (instructions and encodings). Kernels are keyed by their
mangled name with the first namespace normalised (ours is named nvfp4_<file>; the Python files' is an unnamed
namespace, whose mangled name carries a hash of the build).

usage: python3 sass_check.py <python.sass> <ours.sass> [--sm 121]
"""

from __future__ import annotations

import re
import sys


def key(name: str) -> str:
    """The name with its first (length-prefixed) namespace component replaced by @."""
    m = re.match(r"_ZN(\d+)", name)
    if not m:
        return name
    return "_ZN@" + name[m.end() + int(m.group(1)):]


def kernels(path: str, sm: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    arch_ok = True
    for line in open(path):
        a = re.match(r"\s*arch = sm_(\d+)(a?)", line)
        if a:
            arch_ok = a.group(1) == sm
            continue
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            current = key(m.group(1)) if arch_ok else None
            if current is not None:
                out[current] = []
            continue
        if current is not None and re.match(r"\s*(/\*[0-9a-f]{4}\*/|/\* 0x)", line):
            out[current].append(re.sub(r"\s+", " ", line.strip()))
    return out


def main() -> int:
    sm = "121"
    args = sys.argv[1:]
    if "--sm" in args:
        i = args.index("--sm")
        sm = args[i + 1]
        del args[i:i + 2]
    python, ours = kernels(args[0], sm), kernels(args[1], sm)
    bad = missing = 0
    for k in sorted(ours):
        if k not in python:
            missing += 1
            print(f"ONLY-OURS {k}")
        elif python[k] != ours[k]:
            bad += 1
            print(f"SASS-DIFFER {k}: {len(ours[k])} / {len(python[k])} lines")
    extra = sorted(set(python) - set(ours))
    print(f"sm_{sm}: {len(ours) - bad - missing}/{len(ours)} kernels of ours SASS-equal to the Python build, {bad} differ, "
          f"{missing} not in it; {len(extra)} only in the Python build (not ours: lane, reduce)")
    return 1 if bad or missing or not ours else 0


if __name__ == "__main__":
    raise SystemExit(main())
