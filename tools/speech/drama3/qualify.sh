#!/usr/bin/env bash
# The Fish Audio Drama 3 qualification run on a target machine (GB10: sm_121, CUDA 13). Bounded: every step has a
# time limit (then SIGTERM, and SIGKILL 30 s later), and everything it writes goes to OUT, except the marker.
#   tools/speech/drama3/qualify.sh ROOT CONFIG OUT --reference RUNTIME [--tool ID] [--accept-native-numerics]
#     [--accept-lossy] [--coresident]
# ROOT: the asset root (fetch.sh, build.sh), verified here by sha256. CONFIG: the LocalRouter config holding the
# drama3 tool (--tool, default fishaudio/drama-3) and the host's "gpu_lock"; its weights must be ROOT.
# Steps: assets (the upstream checkout, ROOT/engine.json and its binary, every checkpoint file), e2e (the real worker
# on the GPU through its Python API: load, whole, warm and streamed replies, voices, refusals, unload), bench
# (upstream's own bench/drama_bench.py against the worker's server, whole then streamed, as upstream measured its
# claims; first audio, real-time factor and the unit's peak memory, gated by qualify.py's THRESHOLDS), accuracy
# (upstream's own qualification/ checks against the SGLang-Omni reference: the backbone and tower reference checks
# in upstream's bounded GPU guard, and the backbone rescored under its registered worst-row bar), marker.
# e2e and bench hold the host's gpu_lock shared while the model is loaded, as the daemon does, and give the worker a
# private lock in OUT; accuracy's guard takes the host lock exclusively, after bench has released it.
# --coresident: chat and speech-to-text are already loaded by a running LocalRouter daemon (one GB10, all warm). It is
# recorded with MemAvailable and the daemon's /v1/tools in environment.txt and the bench report, and is not required.
# The daemon's models hold the host lock shared, so accuracy cannot run beside them: such a run does not qualify.
# RUNTIME holds venv/ (the reference's dependencies) and sglang-omni/ at recipe/sources.json's reference revision,
# as upstream's recipe describes. The accuracy verdicts are recorded in the report and the marker, pass or fail;
# they do not decide acceptance, but a run whose checks produced no valid evidence writes no marker.
# The native engine is not bit-exact with the SGLang-Omni reference, and upstream reports it as testing, not
# qualified. --accept-native-numerics is the operator's explicit acceptance of it as LocalRouter's numerical
# baseline, recorded in the marker. Exit 0 (and "qualified": true in OUT/status.json) only when assets, e2e and
# bench pass, the accuracy checks ran with valid evidence,
# that acceptance is given and the marker named by the tool's "qualification" is written. Each step's exit code is
# in status.json either way. Without a qualification, no marker is written and an existing one is left as it is.
# Worst case: environment 160 + assets 2430 + e2e 3630 + bench 3030 + accuracy 2430 + marker 150 + manifest 130
# = 11960 s.
# A tool with "variant" (nvfp4: pins-nvfp4.json) is checked against its own pins, checkpoint and engine record;
# upstream's accuracy checks do not apply to its lossy weights (accuracy records why and exits 0, no --reference
# needed), and its marker needs --accept-lossy too, the operator's recorded acceptance of them. Such a run also
# qualifies with --coresident, since no exclusive guard runs: that is how it is measured beside the warm models.
# A run cut short (any limit, or an outer timeout) leaves no "qualified": true: status.json is written only at the end.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${1:?usage: qualify.sh ROOT CONFIG OUT --reference RUNTIME [--tool ID] [--accept-native-numerics] [--accept-lossy] [--coresident]}
config=${2:?}
out=${3:?}
shift 3
tool=fishaudio/drama-3 accept=false lossy=false reference= coresident=false
while [ $# -gt 0 ]; do
  case $1 in
    --tool) tool=$2; shift 2 ;;
    --reference) reference=$2; shift 2 ;;
    --accept-native-numerics) accept=true; shift ;;
    --accept-lossy) lossy=true; shift ;;
    --coresident) coresident=true; shift ;;
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
  sha256sum "$here"/pins*.json
  timeout 20 python3 -I -c 'import platform; print("python", platform.python_version())'
  timeout 20 nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1 || true
  timeout 20 systemd-run --version 2>&1 | head -1
  cat "$root/engine.json" 2>&1
  grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
  echo "tool $tool, accept-native-numerics $accept, accept-lossy $lossy, reference ${reference:-none}, coresident $coresident"
  [ -n "$reference" ] && git -C "$reference/sglang-omni" rev-parse HEAD
  # What a running LocalRouter daemon has loaded (chat, speech-to-text), if one answers on its default port.
  echo "localrouter /v1/tools: $(timeout 10 curl -s --max-time 5 http://127.0.0.1:8190/v1/tools || echo none)"
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
# Speed is measured only on a worker that passed end to end.
if [ "${rc[assets]}" = 0 ] && [ "${rc[e2e]}" = 0 ]; then
  cores=()
  [ "$coresident" = true ] && cores=(--coresident)
  step bench 3000 python3 -I "$here/qualify.py" bench "$root" "$config" "$out/bench" --tool "$tool" "${cores[@]}"
else
  rc[bench]=null
  echo "== bench not run: assets or e2e failed" | tee -a "$out/steps.log"
fi
# Upstream's accuracy checks take the host GPU lock exclusively themselves: they run after bench has unloaded the
# model and released its shared hold.
if [ "${rc[assets]}" = 0 ]; then
  step accuracy 2400 python3 -I "$here/qualify.py" accuracy "$root" "$config" "$out/accuracy" --tool "$tool" \
    ${reference:+--reference "$reference"}
else
  rc[accuracy]=null
  echo "== accuracy not run: assets failed" | tee -a "$out/steps.log"
fi

# Nothing may write into OUT once the manifest is taken: stop the sampler (and wait for it) first.
stop_sampler
trap - EXIT
qualified=false write=()
[ "${rc[assets]}" = 0 ] && [ "${rc[e2e]}" = 0 ] && [ "${rc[bench]}" = 0 ] && [ "${rc[accuracy]}" = 0 ] &&
  [ "$accept" = true ] && write=(--write --accept-native-numerics)
[ ${#write[@]} = 2 ] && [ "$lossy" = true ] && write+=(--accept-lossy)
# With --write the marker is written; without, an existing marker is only reported.
step marker 120 python3 -I "$here/qualify.py" marker "$root" "$config" "$out" --tool "$tool" "${write[@]}"
cat "$out/marker.log"
[ ${#write[@]} -ge 2 ] && [ "${rc[marker]}" = 0 ] && qualified=true
printf '{"assets": %s, "e2e": %s, "bench": %s, "accuracy": %s, "marker": %s, "tool": "%s", "accept_native_numerics": %s, "accept_lossy": %s, "coresident": %s, "qualified": %s}\n' \
  "${rc[assets]}" "${rc[e2e]}" "${rc[bench]}" "${rc[accuracy]}" "${rc[marker]}" "$tool" "$accept" "$lossy" "$coresident" \
  "$qualified" > "$out/status.json"
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
