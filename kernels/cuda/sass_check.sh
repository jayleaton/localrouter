#!/bin/bash
# Offline bit-identity evidence (no GPU): builds TensorFold's NVFP4 extension from the sources at the git tag the way
# torch.utils.cpp_extension does (torch headers and flags, -O3, -gencode for the `a` target of one SM, in the NGC
# image) and our kernels the way build/cuda.zig does, then compares the SASS of every kernel of ours with
# sass_check.py. usage: [OUT=<dir>] [SMS="121 120"] [FATBINS=<zig-out/fatbin>] bash kernels/cuda/sass_check.sh
# FATBINS: also compares the built fatbins (what a kit ships) against the Python build.
# Native nvcc (a GPU pod in the NGC image, /usr/local/cuda/bin/nvcc on the PATH or NATIVE=1): no docker, the same
# commands run in place. Needs torch (cpp_extension) and cuobjdump there.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$REPO/.zig-cache/sass-nvfp4}"
SMS="${SMS:-121 120}"
IMAGE="${STK_NVCC_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF="${TF_PY:-$REPO/../ref/tensorfold-py}"
REF="${TF_REF:-v0.6.1}"
if [ -z "${NATIVE:-}" ]; then command -v nvcc > /dev/null 2>&1 && NATIVE=1 || NATIVE=0; fi
# inimg <mem> <script> [env...]: run a bash script in the NGC image, or in place; /out, /k and /f name $OUT, the
# kernels dir and $FATBINS in both (in place they are substituted)
inimg() {
  local script="$2"; shift 2
  if [ "$NATIVE" = 1 ]; then
    script="${script//\/out/$OUT}"; script="${script//\/k\//$HERE/}"; script="${script//\/f\//${FATBINS:-/nonexistent}/}"
    OUT_DIR="$OUT" env "$@" bash -c "$script" 2>&1 | grep -v "No CUDA runtime" || true
  else
    docker run --rm --network none --memory 24g -u "$(id -u):$(id -g)" -e HOME=/tmp -e OUT_DIR=/out "${DENV[@]}" \
      -v "$HERE:/k:ro" -v "$OUT:/out" ${FATBINS:+-v "$FATBINS:/f:ro"} --entrypoint bash "$IMAGE" -c "$script" 2>&1 | grep -v "No CUDA runtime" || true
  fi
}
DENV=()
mkdir -p "$OUT"
rm -rf "$OUT/src" && mkdir -p "$OUT/src"
git -C "$TF" archive "$REF" src/tensorfold/cuda/nvfp4 src/tensorfold/cuda/kernels/qmm_frag.cuh | tar -x -C "$OUT/src"
cp "$HERE/sass_check.py" "$OUT/sass_check.py"
# the extension exactly as checkpoint.py _ext() loads it; torch adds its own flags, the arch list is the one target
cat > "$OUT/build_ext.py" <<'PY'
import os, sys
sm = sys.argv[1]
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{sm[:-1]}.{sm[-1]}a"
from torch.utils import cpp_extension
out = os.environ["OUT_DIR"]
here = f"{out}/src/src/tensorfold/cuda/nvfp4"
bd = f"{out}/ext{sm}"
os.makedirs(bd, exist_ok=True)
try:
    cpp_extension.load(name="tensorfold_nvfp4_ck_v6",
                       sources=[f"{here}/{f}" for f in ("checkpoint.cpp", "act.cu", "lane4.cu", "gemm_ck.cu", "gemm_ws.cu")],
                       extra_include_paths=[here], extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{sm}a,code=sm_{sm}a"],
                       build_directory=bd, verbose=True)
except ImportError as e:  # no GPU here: the library built, only its import needs a driver
    print("import:", e)
PY
rc=0
for sm in $SMS; do
  echo "== sm_$sm"
  # ours: nvcc -fatbin per file with the flags of build/cuda.zig (torch's six, -O3, the arch-specific target)
  ours='set -e; mkdir -p /out/ours'
  for f in act gemm_ck gemm_ws lane4; do
    ours="$ours; nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_${sm}a,code=sm_${sm}a -o /out/ours/$f.$sm.fatbin /k/nvfp4/$f.cu 2>/dev/null; cuobjdump -sass /out/ours/$f.$sm.fatbin >> /out/ours/ours.$sm.sass"
  done
  rm -f "$OUT/ours/ours.$sm.sass"; mkdir -p "$OUT/ours"
  inimg 16g "$ours"
  # the Python build (the long step), then its SASS
  DENV=(-e MAX_JOBS=4)
  inimg 24g "python /out/build_ext.py $sm > /out/ext$sm.log 2>&1; cuobjdump -sass /out/ext$sm/tensorfold_nvfp4_ck_v6.so > /out/py.$sm.sass" MAX_JOBS=4
  grep -h "gencode" "$OUT/ext$sm.log" | head -1 | grep -o "\-gencode[^ ]*" | sort -u | tr '\n' ' '; echo
  python3 "$OUT/sass_check.py" "$OUT/py.$sm.sass" "$OUT/ours/ours.$sm.sass" --sm "$sm" || rc=1
  if [ -n "${FATBINS:-}" ]; then
    : > "$OUT/ours/built.$sm.sass"
    inimg 4g 'for f in /f/nvfp4_*.fatbin; do cuobjdump -sass $f >> /out/ours/built.'"$sm"'.sass; done'
    echo -n "built fatbins: "
    python3 "$OUT/sass_check.py" "$OUT/py.$sm.sass" "$OUT/ours/built.$sm.sass" --sm "$sm" || rc=1
  fi
  # the symbols kernels/cuda/nvfp4.zig names are exactly the kernels of the fatbin
  grep -h "Function :" "$OUT/ours/ours.$sm.sass" | sed 's/.*Function : //' | sort > "$OUT/ours/syms.$sm.txt"
  grep -o '"_Z[^"]*"' "$REPO/kernels/cuda/nvfp4.zig" | tr -d '"' | sort > "$OUT/ours/zig.txt"
  if diff -q "$OUT/ours/syms.$sm.txt" "$OUT/ours/zig.txt" > /dev/null; then echo "symbols in nvfp4.zig: all $(wc -l < "$OUT/ours/zig.txt") match the fatbin"; else
    echo "SYMBOL MISMATCH between nvfp4.zig and the fatbin:"; diff "$OUT/ours/syms.$sm.txt" "$OUT/ours/zig.txt" | head; rc=1; fi
done
exit $rc
