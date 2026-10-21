#!/usr/bin/env bash
# The SGLang-Omni reference runtime that qualify.sh's accuracy step needs (--reference RUNTIME), on an aarch64 GB10.
#   tools/speech/drama3/reference.sh RUNTIME [UV]
# RUNTIME/sglang-omni is sgl-project/sglang-omni at drama-tensorfold's recipe/sources.json revision (ab095ef9), and
# RUNTIME/venv a Python 3.12 venv with its dependencies and its fishaudio extra (torch cu130 from PyTorch's index),
# then sglang-omni itself, editable and without dependencies. Two dependencies are left out: nemo_text_processing and
# dots.tts both need pynini, which has no aarch64 wheel and does not build without OpenFst; only the zonos2 and
# dots_tts models import them, not fishaudio_drama3. Upstream pins the checkout, not the venv's packages: the qualify
# report records the torch and SGLang versions the accuracy checks ran with. UV: the uv binary (default: uv on PATH).
set -euo pipefail
runtime=${1:?usage: reference.sh RUNTIME [UV]}
uv=${2:-uv}
revision=ab095ef9e779e56af33b6ed2c863645c9903372d
mkdir -p "$runtime"
cd "$runtime"
[ -d sglang-omni ] || git clone --quiet https://github.com/sgl-project/sglang-omni.git sglang-omni
git -C sglang-omni checkout --quiet --detach "$revision"
[ "$(git -C sglang-omni rev-parse HEAD)" = "$revision" ] || { echo "reference.sh: sglang-omni is not at $revision" >&2; exit 1; }
[ -x venv/bin/python ] || "$uv" venv --quiet --python 3.12 venv
python3 -I - sglang-omni/pyproject.toml > requirements.txt <<'PY'
import sys, tomllib
p = tomllib.load(open(sys.argv[1], "rb"))["project"]
deps = p["dependencies"] + p["optional-dependencies"]["fishaudio"]
print("\n".join(d for d in deps if not d.startswith(("nemo_text_processing", "dots.tts"))))
PY
printf 'protobuf>=6.31.1,<7.0.0\n' > overrides.txt # sglang-omni's own uv override-dependencies
export VIRTUAL_ENV=$runtime/venv UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-5}
"$uv" pip install --prerelease=allow -r requirements.txt --override overrides.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
"$uv" pip install --no-deps -e ./sglang-omni
venv/bin/python -c 'import torch, sglang, sglang_omni; print("reference.sh: torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), "sglang", sglang.__version__)'
