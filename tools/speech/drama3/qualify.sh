#!/usr/bin/env bash
# The Fish Audio Drama 3 qualification run on a target machine (GB10: sm_121, CUDA 13). Bounded: every step has a
# time limit (then SIGTERM, and SIGKILL 30 s later), and everything it writes goes to OUT, except the marker.
#   tools/speech/drama3/qualify.sh ROOT CONFIG OUT [--tool ID] [--accept-native-numerics]
# ROOT: the asset root (fetch.sh, build.sh), verified here by sha256. CONFIG: the LocalRouter config holding the
# drama3 tool (--tool, default fishaudio/drama-3) and the host's "gpu_lock"; its weights must be ROOT.
# Steps: assets (the upstream checkout, ROOT/engine.json and its binary, every checkpoint file), e2e (the real worker
# on the GPU through its Python API: load, whole, warm and streamed replies, voices, refusals, unload), marker.
# The native engine is not bit-exact with the SGLang-Omni reference, and upstream reports it as testing, not
# qualified. --accept-native-numerics is the operator's explicit acceptance of it as LocalRouter's numerical
# baseline, recorded in the marker. Exit 0 (and "qualified": true in OUT/status.json) only when assets and e2e pass,
# that acceptance is given and the marker named by the tool's "qualification" is written. Each step's exit code is
# in status.json either way. Without a qualification, no marker is written and an existing one is left as it is.
# Worst case: environment 150 + assets 2430 + e2e 3630 + marker 150 + manifest 130 = 6490 s.
# A run cut short (any limit, or an outer timeout) leaves no "qualified": true: status.json is written only at the end.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${1:?usage: qualify.sh ROOT CONFIG OUT [--tool ID] [--accept-native-numerics]}
config=${2:?}
out=${3:?}
shift 3
tool=fishaudio/drama-3 accept=false
while [ $# -gt 0 ]; do
  case $1 in
    --tool) tool=$2; shift 2 ;;
    --accept-native-numerics) accept=true; shift ;;
    *) echo "qualify.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$out"
out=$(cd "$out" && pwd)
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
  sha256sum "$here/pins.json"
  timeout 20 python3 -I -c 'import platform; print("python", platform.python_version())'
  timeout 20 nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1 || true
  timeout 20 systemd-run --version 2>&1 | head -1
  cat "$root/engine.json" 2>&1
  grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
  echo "tool $tool, accept-native-numerics $accept"
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

step assets 2400 python3 -I "$here/qualify.py" assets "$root" "$config" "$out/assets" --tool "$tool"
# The model is loaded only from verified assets.
if [ "${rc[assets]}" = 0 ]; then
  step e2e 3600 python3 -I "$here/qualify.py" e2e "$root" "$config" "$out/e2e" --tool "$tool"
else
  rc[e2e]=null
  echo "== e2e not run: assets failed" | tee -a "$out/steps.log"
fi

# Nothing may write into OUT once the manifest is taken: stop the sampler (and wait for it) first.
stop_sampler
trap - EXIT
qualified=false write=()
[ "${rc[assets]}" = 0 ] && [ "${rc[e2e]}" = 0 ] && [ "$accept" = true ] && write=(--write --accept-native-numerics)
# With --write the marker is written; without, an existing marker is only reported.
step marker 120 python3 -I "$here/qualify.py" marker "$root" "$config" "$out" --tool "$tool" "${write[@]}"
cat "$out/marker.log"
[ ${#write[@]} = 2 ] && [ "${rc[marker]}" = 0 ] && qualified=true
printf '{"assets": %s, "e2e": %s, "marker": %s, "tool": "%s", "accept_native_numerics": %s, "qualified": %s}\n' \
  "${rc[assets]}" "${rc[e2e]}" "${rc[marker]}" "$tool" "$accept" "$qualified" > "$out/status.json"
timeout --kill-after=10 120 bash -c 'cd "$1" && find . -type f ! -name SHA256SUMS -print0 | sort -z |
  xargs -0 sha256sum > SHA256SUMS' _ "$out" || {
  # No complete manifest: the run cannot be a qualification, whatever the steps said. The marker this run wrote goes.
  [ "$qualified" = true ] && python3 -I "$here/qualify.py" marker "$root" "$config" "$out" --tool "$tool" --revoke
  qualified=false
  rm -f "$out/SHA256SUMS"
  sed -i 's/"qualified": true}/"qualified": false, "manifest": "failed"}/' "$out/status.json"
}
echo "qualify.sh: qualified $qualified ($(cat "$out/status.json")), results in $out"
[ "$qualified" = true ]
