#!/usr/bin/env bash
# One command from a checkout to the running Jarvis stack (INSTALL.md): the host tools and LocalRouter's build, each
# model's fetch, setup and GPU qualification, then deploy/jarvis/install.sh and start.sh. All of it comes from the
# profile's manifest (deploy/profiles/PROFILE.json); every download is checked against its pin by the step that makes it.
#   tools/install/bootstrap.sh --profile jarvis-gb10 [--models chat,tts,stt,image,video] [--prefix DIR] [--port N]
#     [--https-port N] [--unit NAME] [--accept] [--no-gpu] [--dry-run] [--force]
# --accept: the user has read and accepted what the plan lists under "needs acceptance" (model licences, Drama 3's
#   numerics); without it the steps that need it stop the run. --no-gpu: every step that needs no GPU (downloads,
#   builds), then stop: for a box whose GPU is busy now; run again without it later. --dry-run: the plan (each step,
#   download, disk and memory) and the checks; changes nothing. --force: go on although the box does not match the
#   profile's hardware. Idempotent and resumable: a step whose result is in place is skipped, so after a failure do what
#   it says and run the same command again. Logs: PREFIX/install/logs/MODEL-STEP.log.
set -uo pipefail
source "$(dirname "$0")/lib.sh"

profile="" list=all PREFIX=$HOME/jarvis PORT=8190 HTTPS_PORT=10443 UNIT=localrouter-jarvis dry=0 force=0 accept=0 nogpu=0
args=("$@")
while [ $# -gt 0 ]; do
  case $1 in
    --profile) profile=${2:?}; shift 2 ;;
    --models) list=${2:?}; shift 2 ;;
    --prefix) PREFIX=$(realpath -m "${2:?}"); shift 2 ;;
    --port) PORT=${2:?}; shift 2 ;;
    --https-port) HTTPS_PORT=${2:?}; shift 2 ;;
    --unit) UNIT=${2:?}; shift 2 ;;
    --accept) accept=1; shift ;;
    --no-gpu) nogpu=1; shift ;;
    --dry-run) dry=1; shift ;;
    --force) force=1; shift ;;
    -h | --help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done
[ -n "$profile" ] || die "--profile is required (have: $(profiles))"
for t in jq curl sha256sum awk; do command -v $t > /dev/null || die "needs $t first: sudo apt-get install -y jq curl coreutils gawk"; done
p=$(profile_file "$profile")
mapfile -t keys < <(selected "$p" "$list")
[ ${#keys[@]} -gt 0 ] || die "no models selected"
step_env "$p"
LOGS=$PREFIX/install/logs
again="$0 ${args[*]}"
again=${again// --dry-run/}

echo "== $profile ($(jq -r .status "$p")): $(jq -r .description "$p")"
echo "box: $(hw_summary)"
echo "prefix $PREFIX, port $PORT, HTTPS port $HTTPS_PORT (0: none), unit $UNIT; models: ${keys[*]}"

# ---- 1. the box: the profile's hardware, then the prerequisites of these models
blocked=0
while read -r line; do
  [ -n "$line" ] || continue
  case $line in warning:*) echo "$line" ;; *) echo "mismatch: $line"; blocked=1 ;; esac
done < <(hw_match "$p")
[ $blocked = 0 ] || [ $force = 1 ] || [ $dry = 1 ] || die "this box does not match $profile (INSTALL.md, Profiles); --force to go on anyway"
missing=0
while read -r q; do
  if ok_test "$q"; then echo "ok    prereq $(jq -r .id <<< "$q")"; else
    echo "MISSING prereq $(jq -r '.id + ": " + .title' <<< "$q")"; echo "      fix: $(jq -r .fix <<< "$q")"; missing=1
  fi
done < <(prereqs "$p" "${keys[@]}")

# ---- 2. the plan: each step, done or to do; what the rest needs
mapfile -t todo < <(steps "$p" "${keys[@]}")
need_dl=0 need_disk=0 warm=0 peak=0 accepts=()
for k in host "${keys[@]}"; do
  left=0
  for s in "${todo[@]}"; do [ "$(jq -r .model <<< "$s")" = "$k" ] && ! is_done "$s" && left=1; done
  if [ $k = host ]; then m=$(jq -c .host "$p"); else m=$(model "$p" "$k"); fi
  [ $left = 1 ] && { need_dl=$((need_dl + $(jq .download_bytes <<< "$m"))); need_disk=$((need_disk + $(jq .disk_bytes <<< "$m"))); }
  [ $k = host ] && continue
  b=$(jq '.memory_bytes + (.working_bytes // 0)' <<< "$m")
  if [ "$(jq .warm <<< "$m")" = true ]; then warm=$((warm + b)); elif [ "$b" -gt "$peak" ]; then peak=$b; fi
  while read -r a; do [ -n "$a" ] && accepts+=("$k: $a"); done < <(jq -r '.accept[]? // empty' <<< "$m")
done
free=$(hw_disk_free "$PREFIX") margin=$(($(jq .disk_margin_gib "$p") * 1073741824))
echo "plan:"
for s in "${todo[@]}"; do
  printf '  %-5s %-16s %s%s\n' "$(is_done "$s" && echo done || echo todo)" "$(step_id "$s")" "$(jq -r .title <<< "$s")" \
    "$(jq -r 'if .gpu then " [GPU, \(.min_free_gib) GiB free]" else "" end' <<< "$s")"
done
printf '  %-5s %-16s %s\n' todo deploy "deploy/jarvis/install.sh (config, unit $UNIT, service on 127.0.0.1:$PORT), then start.sh until the warm models are ready"
echo "needs: download $(gib $need_dl) GiB, disk $(gib $need_disk) GiB + $(gib $margin) margin (free: $(gib "$free") GiB)"
echo "memory: $(gib $warm) GiB for the warm models$([ $peak -gt 0 ] && echo ", plus up to $(gib $peak) GiB for an on-demand one (it unloads the image model to fit)") of $(gib "$(hw_mem)") GiB"
if [ ${#accepts[@]} -gt 0 ]; then
  echo "needs acceptance (ask the user; with their yes, run with --accept):"
  for a in "${accepts[@]}"; do echo "  - $a"; done
  jq -r --arg keys " ${keys[*]} " '.models[] | select(.accept and (.key as $k | $keys | contains(" " + $k + " "))) | "    \(.key) licence: \(.licence.name): \(.licence.url)"' "$p"
fi
[ "$free" -ge $((need_disk + margin)) ] || { echo "not enough free disk under $PREFIX: $(gib "$free") GiB, needs $(gib $((need_disk + margin))) GiB"; [ $dry = 1 ] || exit 1; }
if [ $dry = 1 ]; then echo "dry run: nothing changed. To install: ${again}"; exit 0; fi
[ $missing = 0 ] || die "install the missing prerequisites first (each 'fix' above), then run again: $again"

# ---- 3. the steps
mkdir -p "$LOGS" "$PREFIX/bin"
t_all=$(date +%s)
for s in "${todo[@]}"; do
  id=$(step_id "$s")
  if is_done "$s"; then echo "ok    $id (done)"; continue; fi
  if [ "$(jq -r '.gpu // false' <<< "$s")" = true ] && [ $nogpu = 1 ]; then echo "skip  $id (needs the GPU; --no-gpu)"; continue; fi
  if [ "$(jq -r '.needs_accept // false' <<< "$s")" = true ] && [ $accept = 0 ]; then
    echo "STOP  $id needs the user's acceptance (the plan's 'needs acceptance' list; --dry-run prints it). With their yes: $again --accept" >&2
    exit 1
  fi
  log=$LOGS/${id//\//-}.log
  echo "run   $id: $(jq -r .title <<< "$s")"
  echo "      log: $log"
  t=$(date +%s)
  echo "== $(date -u +%FT%TZ) $id" >> "$log"
  bash -c "set -euo pipefail; $(jq -r .run <<< "$s")" >> "$log" 2>&1
  rc=$?
  if [ $rc != 0 ] || ! is_done "$s"; then
    echo "FAILED $id (exit $rc). The end of $log:" >&2
    tail -n 15 "$log" | sed 's/^/  | /' >&2
    echo "next: $(jq -r .fix <<< "$s")" >&2
    echo "then run the same command again (it resumes here): $again" >&2
    exit 1
  fi
  echo "ok    $id in $(($(date +%s) - t)) s"
done
if [ $nogpu = 1 ]; then echo "== --no-gpu: the GPU steps are left; when the GPU is free run: ${again// --no-gpu/}"; exit 0; fi

# ---- 4. deploy: the service on the selected models (install.sh waits for open voice sessions before a restart)
echo "run   deploy (log: $LOGS/deploy.log)"
tools=$(for k in "${keys[@]}"; do jq -r --arg k "$k" '.models[] | select(.key == $k) | .id' "$p"; done | tr '\n' ' ')
{ TOOLS=$tools SKIP_BUILD=1 bash "$repo/deploy/jarvis/install.sh" && bash "$repo/deploy/jarvis/start.sh"; } > >(tee -a "$LOGS/deploy.log") 2>&1 || {
  echo "FAILED deploy: the end of $LOGS/deploy.log is above. next: tools/install/doctor.sh --profile $profile --prefix $PREFIX --port $PORT --unit $UNIT; fix what it lists, then run again: $again" >&2
  exit 1
}
echo "== done in $(($(date +%s) - t_all)) s: $UNIT serves http://127.0.0.1:$PORT (voice: /voice, API: /v1, MCP: /mcp)"
echo "check it: LOCK=/dev/null bash $repo/tools/jarvis/smoke.sh --quick --url http://127.0.0.1:$PORT"
