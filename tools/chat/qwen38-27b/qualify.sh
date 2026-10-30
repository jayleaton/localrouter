#!/usr/bin/env bash
# The Qwen3.8-27B qualification run on a GB10 (aarch64, CUDA). Bounded: every step has a time limit (then SIGTERM to
# its process group, which holds the server or daemon it started, and SIGKILL 30 s later). Results go to OUT; the
# marker goes to MARKER only when everything passed.
#   tools/chat/qwen38-27b/qualify.sh ROOT LOCALROUTER_BIN OUT MARKER [--config FILE] [--tool ID] [--any-gpu]
# ROOT: fetch.sh's root (setup.sh wrote ROOT/config.json, the default FILE). LOCALROUTER_BIN: the localrouter binary
# under test. OUT: a new or empty directory. MARKER: the tool's "qualification" path (absolute; must equal it).
# Steps: pins (archive, extracted tree, every pinned file), direct (the tool's server argv run by qualify.py),
# daemon (the same prompts through LOCALROUTER_BIN serve), compare (determinism, daemon = direct, stream = whole),
# marker (only when the four exit 0). Each step's exit code is in OUT/status.json; OUT/report.json has the memory
# peak (MemAvailable every 0.5 s during direct and daemon) and the timings.
# Refuses unless this is aarch64 with a GB10 and MemAvailable is at least 64 GiB. --any-gpu skips the host check for
# a dry run elsewhere; a dry run never writes the marker. Stop the LocalRouter daemon (and anything else on the GPU)
# first. Worst case: preflight 120 + pins 1830 + direct 3630 + daemon 3630 + compare 330 + marker 150 + report 150
# + manifest 130 = 9970 s.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: qualify.sh ROOT LOCALROUTER_BIN OUT MARKER [--config FILE] [--tool ID] [--any-gpu]"
[ $# -ge 4 ] || { echo "$usage" >&2; exit 2; }
root=$1 lr=$2 out=$3 marker=$4
shift 4
config= tool=qwen3.8-27b any_gpu=0
while [ $# -gt 0 ]; do
  case $1 in
    --config) config=${2:?$usage}; shift 2 ;;
    --tool) tool=${2:?$usage}; shift 2 ;;
    --any-gpu) any_gpu=1; shift ;;
    *) echo "qualify.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "qualify.sh: $*" >&2; exit 1; }
command -v python3 >/dev/null || die "needs python3"
root=$(cd "$root" && pwd)
[ -n "$config" ] || config=$root/config.json
[ -f "$config" ] || die "$config is missing: run setup.sh $root first, or pass --config"
config=$(cd "$(dirname "$config")" && pwd)/$(basename "$config")
lr=$(cd "$(dirname "$lr")" && pwd)/$(basename "$lr")
case $marker in /*) ;; *) die "MARKER must be an absolute path" ;; esac

mem_avail() { # MemAvailable in bytes
  local key value rest
  while read -r key value rest; do
    if [ "$key" = MemAvailable: ]; then
      echo $((value * 1024))
      return 0
    fi
  done < /proc/meminfo
  echo 0
}

# The host: a GB10 with room for the model.
arch=$(uname -m)
gpus=$(timeout 20 nvidia-smi -L 2>&1) || gpus="nvidia-smi -L failed: $gpus"
if [ "$any_gpu" = 0 ]; then
  [ "$arch" = aarch64 ] || die "this is $arch, not a GB10 (aarch64); --any-gpu for a dry run"
  grep -q GB10 <<< "$gpus" || die "nvidia-smi -L lists no GB10 ($gpus); --any-gpu for a dry run"
fi
avail=$(mem_avail)
[ "$avail" -ge $((64 << 30)) ] || die "MemAvailable is $((avail >> 20)) MiB, under 64 GiB: free memory first"

mkdir -p "$out"
out=$(cd "$out" && pwd)
listing=$(ls -A "$out")
[ -z "$listing" ] || die "$out is not empty: give each run a new OUT"
py=(python3 -I "$here/qualify.py")
common=(--root "$root" --config "$config" --tool "$tool" --out "$out")
timeout --kill-after=10 120 "${py[@]}" preflight "${common[@]}" --bin "$lr" --marker "$marker" || die "preflight failed"

{
  git -C "$here" rev-parse HEAD 2>/dev/null || cat "$here/../../../COMMIT" 2>/dev/null || echo "commit unknown"
  uname -a
  sha256sum "$here/pins.json" "$here/prompts.json" "$here/qualify.py" "$here/qualify_lib.py" "$here/qualify.sh" \
    "$lr" "$config"
  timeout 20 python3 -I -c 'import platform; print("python", platform.python_version())'
  echo "$gpus"
  timeout 20 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
  grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
  cat "$root/tensorfold-version.txt"
  echo "tool $tool, config $config, marker $marker, any-gpu $any_gpu"
} > "$out/environment.txt" 2>&1 || true

declare -A rc=([pins]=null [direct]=null [daemon]=null [compare]=null [marker]=null)
step() { # name seconds command...
  local name=$1 secs=$2 r=0
  shift 2
  echo "== $name (limit ${secs}s) $(date -u +%FT%TZ)" | tee -a "$out/steps.log"
  timeout --kill-after=30 "$secs" "$@" > "$out/$name.log" 2>&1 || r=$?
  rc[$name]=$r
  echo "== $name exit $r $(date -u +%FT%TZ)" | tee -a "$out/steps.log"
}
phase() { # name start|end: the time and MemAvailable, for report.json
  printf '%s %s %s %s\n' "$1" "$2" "$(date +%s.%N)" "$(mem_avail)" >> "$out/phases.log"
}

step pins 1800 "${py[@]}" pins "${common[@]}"
# The model is loaded only from verified files.
if [ "${rc[pins]}" = 0 ]; then
  ( while true; do
      printf '%s %s\n' "$(date +%s.%N)" "$(mem_avail)"
      sleep 0.5
    done ) > "$out/memory.log" 2>/dev/null &
  sampler=$!
  trap 'kill "$sampler" 2>/dev/null || true' EXIT
  phase direct start
  step direct 3600 "${py[@]}" direct "${common[@]}"
  phase direct end
  phase daemon start
  step daemon 3600 "${py[@]}" daemon "${common[@]}" --bin "$lr"
  phase daemon end
  # Nothing may write into OUT once the manifest is taken: stop the sampler (and wait for it) first.
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  trap - EXIT
  step compare 300 "${py[@]}" compare "${common[@]}"
else
  echo "== direct, daemon, compare not run: pins failed" | tee -a "$out/steps.log"
fi

qualified=false
if [ "${rc[pins]}" = 0 ] && [ "${rc[direct]}" = 0 ] && [ "${rc[daemon]}" = 0 ] && [ "${rc[compare]}" = 0 ]; then
  if [ "$any_gpu" = 0 ]; then
    step marker 120 "${py[@]}" marker "${common[@]}" --marker "$marker"
    cat "$out/marker.log"
    [ "${rc[marker]}" != 0 ] || qualified=true
  else
    echo "== marker not written: --any-gpu dry run" | tee -a "$out/steps.log"
  fi
else
  echo "== marker not written: a step failed (an existing $marker is left as it is)" | tee -a "$out/steps.log"
fi
printf '{"pins": %s, "direct": %s, "daemon": %s, "compare": %s, "marker": %s, "tool": "%s", "any_gpu": %s, "qualified": %s}\n' \
  "${rc[pins]}" "${rc[direct]}" "${rc[daemon]}" "${rc[compare]}" "${rc[marker]}" "$tool" \
  "$([ "$any_gpu" = 1 ] && echo true || echo false)" "$qualified" > "$out/status.json"
timeout --kill-after=10 120 "${py[@]}" report "${common[@]}" > "$out/report.log" 2>&1 ||
  echo "qualify.sh: WARNING: report.json not written (see $out/report.log)" >&2
timeout --kill-after=10 120 bash -c 'cd "$1" && find . -type f ! -name SHA256SUMS -print0 | sort -z |
  xargs -0 sha256sum > SHA256SUMS' _ "$out" || {
  # No complete manifest: the run cannot be a qualification, whatever the steps said. The marker this run wrote goes.
  if [ "$qualified" = true ]; then
    "${py[@]}" marker "${common[@]}" --marker "$marker" --revoke || true
  fi
  qualified=false
  rm -f "$out/SHA256SUMS"
  sed -i 's/"qualified": true}/"qualified": false, "manifest": "failed"}/' "$out/status.json"
}
echo "qualify.sh: qualified $qualified ($(cat "$out/status.json")), results in $out"
[ "$qualified" = true ]
