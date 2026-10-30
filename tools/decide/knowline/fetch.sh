#!/usr/bin/env bash
# Sets up the decision model on a box: PelaAI/KnowLine-4B-Gen5.5 at the pinned revision into the model store (every file
# checked against the repository's SHA256SUMS, then `.revision` written), and its venv from this directory's uv.lock.
# Idempotent: weights already present at the pinned revision are not fetched again.
#   tools/decide/knowline/fetch.sh            (JARVIS: the root, default ~/jarvis)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../../../deploy/jarvis/env.sh"
dir=$J/models/decide/knowline-4b-gen5.5
if [ "$(cat "$dir/.revision" 2> /dev/null)" != "$KNOWLINE_REV" ]; then
  hf=$(command -v hf || echo "$J/venvs/whisper/bin/hf")
  [ -x "$hf" ] || die "needs the hf CLI (huggingface_hub)"
  HF_HUB_DISABLE_TELEMETRY=1 "$hf" download PelaAI/KnowLine-4B-Gen5.5 --revision "$KNOWLINE_REV" --local-dir "$dir"
  (cd "$dir" && sha256sum --quiet -c SHA256SUMS) || die "the weights do not match SHA256SUMS"
  echo "$KNOWLINE_REV" > "$dir/.revision"
fi
echo "weights: $dir ($(du -sh "$dir" | cut -f1), revision $KNOWLINE_REV)"
export UV_CACHE_DIR=${UV_CACHE_DIR:-$J/uv-cache}
(cd "$here" && UV_PROJECT_ENVIRONMENT=$J/venvs/decide uv sync --frozen)
"$J/venvs/decide/bin/python" -I -c 'import torch, transformers; assert torch.cuda.is_available(); print("venv: torch", torch.__version__, "transformers", transformers.__version__)'
