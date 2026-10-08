#!/bin/bash
# nvcc from the NGC PyTorch image for hosts without CUDA: `zig build -Dnvcc=<this script> ...`. The repo and the Zig
# caches are mounted at the same absolute paths, so arguments pass through as is; the dependency file (-MF) keeps only
# the files under the repo (the image's CUDA headers do not exist on the host; the compiler's version text already
# keys every fatbin).
set -euo pipefail
IMAGE="${STK_NVCC_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$HOME/.cache/zig"
mounts=(-v "$REPO:$REPO" -v "$HOME/.cache/zig:$HOME/.cache/zig")
docker run --rm -i --network none --memory 8g -u "$(id -u):$(id -g)" "${mounts[@]}" -w "$PWD" \
  --entrypoint /usr/local/cuda/bin/nvcc "$IMAGE" "$@"
dep=""
prev=""
for a in "$@"; do
  [ "$prev" = "-MF" ] && dep="$a"
  prev="$a"
done
if [ -n "$dep" ] && [ -f "$dep" ]; then
  python3 - "$dep" "$REPO" "$PWD" <<'PY'
import os, sys
path, keep, cwd = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path).read().replace("\\\n", " ")
target, _, deps = text.partition(":")
out = []
for d in deps.split():
    full = d if os.path.isabs(d) else os.path.join(cwd, d)
    if os.path.realpath(full).startswith(keep):
        out.append(d)
open(path, "w").write(target + ": " + " \\\n  ".join(out) + "\n")
PY
fi
