#!/usr/bin/env bash
# Sets up the decision model on a Mac (Apple silicon): PelaAI/KnowLine-4B-Gen5.5 at pins-macos.json's revision into the
# model store (every file checked against the repository's SHA256SUMS, then `.revision` written; the same directory as
# on the Spark, ../fetch.sh), and the MLX venv from this directory's uv.lock.
# Idempotent: weights already present at the pinned revision are not fetched again.
#   tools/decide/knowline/macos/fetch.sh            (JARVIS: the root, default ~/jarvis)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
J=${JARVIS:-$HOME/jarvis}
die() { echo "$(basename "$0"): $*" >&2; exit 1; }
[ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || die "macOS on Apple silicon only (the Spark's is ../fetch.sh)"
pins=$here/pins-macos.json
repo=$(jq -r .model.repo "$pins") rev=$(jq -r .model.revision "$pins")
dir=$J/models/decide/$(jq -r .model.dir "$pins")
if [ "$(cat "$dir/.revision" 2> /dev/null)" != "$rev" ]; then
  if command -v hf > /dev/null; then hf=(hf); else hf=(uvx --from huggingface_hub hf); fi
  HF_HUB_DISABLE_TELEMETRY=1 "${hf[@]}" download "$repo" --revision "$rev" --local-dir "$dir"
  (cd "$dir" && shasum -a 256 --quiet -c SHA256SUMS) || die "the weights do not match SHA256SUMS"
  echo "$rev" > "$dir/.revision"
fi
echo "weights: $dir ($(du -sh "$dir" | cut -f1), revision $rev)"
export UV_CACHE_DIR=${UV_CACHE_DIR:-$J/uv-cache}
(cd "$here" && UV_PROJECT_ENVIRONMENT=$J/venvs/decide-mlx uv sync --frozen)
"$J/venvs/decide-mlx/bin/python" -I -c 'import mlx.core as mx, mlx_lm, transformers; assert mx.metal.is_available(); print("venv: mlx", mx.__version__, "mlx-lm", mlx_lm.__version__, "transformers", transformers.__version__)'
