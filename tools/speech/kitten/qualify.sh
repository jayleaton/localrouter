#!/usr/bin/env bash
# The Kitten TTS 2 qualification run on a target machine (GB10: CUDA). Bounded: every step has a time limit (then
# SIGTERM, and SIGKILL 30 s later), and everything it writes goes to OUT.
#   tools/speech/kitten/qualify.sh ROOT VENV OUT [--fixtures F1,F2,F3,F4,F5] [--device cuda]
# ROOT: the asset root (filled by fetch.sh, verified here by sha256). VENV: made by setup.sh.
# Exit 0 (and "qualified": true in OUT/status.json) only when the unit tests pass, every pinned file matches, the bits
# gate passes, and the bench completes with every measurement. Each step's exit code is in status.json either way.
# Worst case: environment 150 + unit 150 + assets 930 + gate 2730 + bench 1230 + cleanup and manifest 130 = 5320 s.
# A run cut short (any limit, or an outer timeout) leaves no "qualified": true: status.json is written only at the end.
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
declare -A rc
step() { # name seconds command...
  local name=$1 secs=$2; shift 2
  echo "== $name (limit ${secs}s) $(date -u +%FT%TZ)" | tee -a "$out/steps.log"
  timeout --kill-after=30 "$secs" "$@" > "$out/$name.log" 2>&1
  rc[$name]=$?
  echo "== $name exit ${rc[$name]} $(date -u +%FT%TZ)" | tee -a "$out/steps.log"
}

{
  git -C "$here" rev-parse HEAD 2>/dev/null || cat "$here/../../../COMMIT" # a kit carries the commit in COMMIT
  uname -a
  timeout --kill-after=10 120 "$py" -I -c 'import torch, kittenml; print("torch", torch.__version__, "cuda", torch.version.cuda, "kittenml", kittenml.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'
  timeout 20 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || true
} > "$out/environment.txt" 2>&1

# Memory every 0.5 s while the model runs: MemAvailable (GB10 memory is unified) and what nvidia-smi reports.
( while true; do
    printf '%s %s %s\n' "$(date +%s.%N)" "$(awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo)" \
      "$(timeout 5 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    sleep 0.5
  done ) > "$out/memory.log" 2>/dev/null &
sampler=$!
stop_sampler() { kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null; }
trap stop_sampler EXIT

step unit 120 "$py" -I -m unittest discover -s "$here" -p 'test_*.py'
step assets 900 "$py" -I -c '
import sys; sys.path.insert(0, sys.argv[1]); import kitten_worker as kw
kw.verify_assets(sys.argv[2], kw.load_pins(), "sha256", "packed", "default"); print("pinned files verified (sha256)")
' "$here" "$root"
step gate 2700 "$py" -I "$here/qualify.py" gate "$root" "$out/gate" --fixtures "$fixtures" --device "$device"
step bench 1200 "$py" -I "$here/qualify.py" bench "$root" "$out/bench" --device "$device"

# Nothing may write into OUT once the manifest is taken: stop the sampler (and wait for it) first.
stop_sampler
trap - EXIT
qualified=false
# A qualification needs every fixture: a subset (--fixtures) is a diagnostic run.
[ "${rc[unit]}" = 0 ] && [ "${rc[assets]}" = 0 ] && [ "${rc[gate]}" = 0 ] && [ "${rc[bench]}" = 0 ] &&
  [ "$fixtures" = F1,F2,F3,F4,F5 ] && qualified=true
printf '{"unit": %s, "assets": %s, "gate": %s, "bench": %s, "fixtures": "%s", "device": "%s", "qualified": %s}\n' \
  "${rc[unit]}" "${rc[assets]}" "${rc[gate]}" "${rc[bench]}" "$fixtures" "$device" "$qualified" > "$out/status.json"
# The reports and the audio, not the float dumps (large, reproducible from the commands above).
timeout --kill-after=10 120 bash -c '
  find "$1" -name "*.f32" -delete
  find "$1" -path "*/ops-*" -type f ! -name "*.text" -delete
  cd "$1" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS' _ "$out" || {
  # No complete manifest: the run cannot be a qualification, whatever the steps said.
  qualified=false
  rm -f "$out/SHA256SUMS"
  sed -i 's/"qualified": true}/"qualified": false, "manifest": "failed"}/' "$out/status.json"
}
echo "qualify.sh: qualified $qualified ($(cat "$out/status.json")), results in $out"
[ "$qualified" = true ]
