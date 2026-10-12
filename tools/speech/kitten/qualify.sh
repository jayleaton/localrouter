#!/usr/bin/env bash
# The Kitten TTS 2 qualification run on a target machine (GB10: CUDA). Bounded: each step has a time limit, and
# everything it writes goes to OUT.
#   tools/speech/kitten/qualify.sh ROOT VENV OUT [--fixtures F1,F2,F3,F4,F5] [--device cuda]
# ROOT: the asset root (filled by fetch.sh, verified here by sha256). VENV: made by setup.sh. Exit 0 only when the
# unit tests pass, every pinned file matches, and the bits gate passes; bench numbers are recorded either way.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${1:?usage: qualify.sh ROOT VENV OUT [--fixtures LIST] [--device DEV]}
venv=${2:?}
out=${3:?}
shift 3
fixtures=F1,F2,F3,F4,F5 device=cuda
while [ $# -gt 0 ]; do
  case $1 in
    --fixtures) fixtures=$2; shift 2 ;;
    --device) device=$2; shift 2 ;;
    *) echo "qualify.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$out"
py="$venv/bin/python"
status=0
step() { # name seconds command...
  local name=$1 secs=$2; shift 2
  echo "== $name (limit ${secs}s) $(date -u +%FT%TZ)" | tee -a "$out/steps.log"
  timeout --kill-after=30 "$secs" "$@" > "$out/$name.log" 2>&1
  local rc=$?
  echo "== $name exit $rc $(date -u +%FT%TZ)" | tee -a "$out/steps.log"
  return $rc
}

{
  git -C "$here" rev-parse HEAD
  uname -a
  "$py" -I -c 'import torch, kittenml; print("torch", torch.__version__, "cuda", torch.version.cuda, "kittenml", kittenml.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || true
} > "$out/environment.txt" 2>&1

# Memory every 0.5 s while the model runs: MemAvailable (GB10 memory is unified) and what nvidia-smi reports.
( while true; do
    printf '%s %s %s\n' "$(date +%s.%N)" "$(awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo)" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    sleep 0.5
  done ) > "$out/memory.log" 2>/dev/null &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

step unit 120 "$py" -I -m unittest discover -s "$here" -p 'test_*.py' || status=1
step assets 900 "$py" -I -c '
import sys; sys.path.insert(0, sys.argv[1]); import kitten_worker as kw
kw.verify_assets(sys.argv[2], kw.load_pins(), "sha256", "packed", "default"); print("pinned files verified (sha256)")
' "$here" "$root" || status=1
step gate 2700 "$py" -I "$here/qualify.py" gate "$root" "$out/gate" --fixtures "$fixtures" --device "$device" || status=1
step bench 1200 "$py" -I "$here/qualify.py" bench "$root" "$out/bench" --device "$device" || true

# The reports and the audio, not the float dumps (large, reproducible from the commands above).
find "$out" -name '*.f32' -delete
find "$out" -path '*/ops-*' -type f ! -name '*.text' -delete
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "qualify.sh: exit $status, results in $out"
exit $status
