#!/usr/bin/env bash
# Offline fallback for the oracle image: builds the native oracle venv on a Spark node from prestaged files only.
#   WORK=<dir> PYTAR=<cpython-3.13.16+20261009-aarch64-unknown-linux-gnu-install_only.tar.gz> WHEELS=<dir> \
#     bash setup-venv.sh            -> $WORK/venv  (prints python/torch versions; exit 0 iff the pins hold)
# PYTAR is python-build-standalone 20261009 (sha256 below); WHEELS holds the `pip download "torch==2.14.1+cu130"`
# aarch64 set and lock.txt (`name==version --hash=sha256:...`). No network: pip runs --no-index --require-hashes,
# so the venv is exactly upstream's torch==2.14.1 on Python 3.13. Runs after T0 only; touches nothing outside WORK.
set -uo pipefail
PY_SHA=397ca401f416c3c99330a6be16d735d0cbeaf53e82070b80395b2fb4a8891f34
: "${WORK:?}" "${PYTAR:?}" "${WHEELS:?}"
die() { echo "setup-venv: FAILED: $*" >&2; exit 1; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1

[[ "$(uname -m)" == aarch64 ]] || die "not aarch64"
echo "$PY_SHA  $PYTAR" | sha256sum -c --quiet - || die "python tarball sha256"
[[ -f "$WHEELS/lock.txt" ]] || die "no $WHEELS/lock.txt"
if [[ -f "$WHEELS/SHA256SUMS" ]]; then (cd "$WHEELS" && sha256sum -c --quiet SHA256SUMS) || die "wheel sha256"; fi
[[ -e "$WORK/venv" ]] && die "$WORK/venv exists; refusing to modify it"

mkdir -p "$WORK/py" || die "mkdir"
tar -xzf "$PYTAR" -C "$WORK/py" || die "extract python"
"$WORK/py/python/bin/python3" -m venv "$WORK/venv" || die "venv"
"$WORK/venv/bin/python" -m pip install --no-cache-dir --no-index --find-links "$WHEELS" --require-hashes \
    -r "$WHEELS/lock.txt" || die "pip install"
"$WORK/venv/bin/python" - <<'EOF' || die "pin check"
import sys, torch
assert sys.version_info[:2] == (3, 13), sys.version
assert torch.__version__ == "2.14.1+cu130", torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
EOF
"$WORK/venv/bin/python" -m pip freeze > "$WORK/venv/freeze.txt"
echo "setup-venv: ok $WORK/venv"
