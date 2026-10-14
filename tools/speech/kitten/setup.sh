#!/usr/bin/env bash
# Creates the Kitten TTS 2 worker's Python 3.12 environment from the hash-pinned lock for this machine.
#   tools/speech/kitten/setup.sh VENV [WHEELHOUSE]     x86_64: CPU torch; aarch64 (GB10): CUDA 13.0 torch
# With WHEELHOUSE (a directory of the lock's files, e.g. from `pip download`) nothing is fetched from an index.
# Needs uv (https://docs.astral.sh/uv/). The worker then runs as VENV/bin/python -I tools/speech/kitten/kitten_worker.py.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
venv=${1:?usage: setup.sh VENV [WHEELHOUSE]}
case $(uname -m) in
  x86_64) lock=requirements-x86_64-cpu.txt index=https://download.pytorch.org/whl/cpu ;;
  aarch64) lock=requirements-aarch64-cu130.txt index=https://download.pytorch.org/whl/cu130 ;;
  *) echo "setup.sh: no lock for $(uname -m)" >&2; exit 1 ;;
esac
if [ $# -ge 2 ]; then
  source=(--offline --no-index --find-links "$2")
else
  source=(--index-url "$index" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match)
fi
uv venv --python 3.12 "$venv"
VIRTUAL_ENV=$venv uv pip install --require-hashes --no-deps "${source[@]}" -r "$here/$lock"
"$venv/bin/python" -I -c 'import kittenml, torch; print("kittenml", kittenml.__version__, "torch", torch.__version__)'
