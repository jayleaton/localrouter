#!/usr/bin/env bash
# One command from a checkout to the running Jarvis stack (INSTALL.md): the host tools and LocalRouter's build, each
# model's fetch, setup and GPU qualification, then deploy/jarvis/install.sh and start.sh. All of it comes from the
# profile's manifest (deploy/profiles/PROFILE.json); every download is checked against its pin by the step that makes it.
#   tools/install/bootstrap.sh --profile jarvis-gb10 [--models chat,tts,stt,image,video] [--prefix DIR] [--port N]
#     [--https-port N] [--unit NAME] [--models-dir DIR] [--accept] [--no-gpu] [--dry-run] [--force]
#     [--topology FILE --node ID] [--peer-store NODE=USER@HOST:DIR]... [--allow-download]
# --models-dir DIR: an existing install (default: ~/jarvis, when --prefix is elsewhere; "none": no reuse) whose model
#   files are hard-linked in instead of downloaded again (no second copy on disk; each step still checks them against
#   its pins and fetches only what is missing or differs); its uv and zig caches are shared too.
# --accept: the user has read and accepted what the plan lists under "needs acceptance" (model licences, Drama 3's
#   numerics); without it the steps that need it stop the run. --no-gpu: every step that needs no GPU (downloads,
#   builds), then stop: for a box whose GPU is busy now; run again without it later. --dry-run: the plan (each step,
#   download, disk and memory) and the checks; changes nothing. --force: go on although the box does not match the
#   profile's hardware. Idempotent and resumable: a step whose result is in place is skipped, so after a failure do what
#   it says and run the same command again. Logs: PREFIX/install/logs/MODEL-STEP.log.
# --topology FILE --node ID: this box is node ID of a multi-node topology (deploy/jarvis/topology.sh): it installs the
#   models the node runs (and on the front its fallback ones; --models only narrows that), its port is the node's, and
#   the deploy writes the topology into its config. --peer-store NODE=USER@HOST:DIR (repeatable): another node's
#   install DIR; a model whose files are missing here and not in --models-dir is copied from it with rsync over ssh
#   (to the node's first topology address, else HOST) before its steps, which check every file against the pins.
#   With either, a model whose weights no peer has stops the run before anything downloads (--dry-run lists them)
#   unless --allow-download (only with the user's yes).
set -uo pipefail
source "$(dirname "$0")/lib.sh"

profile="" list=all PREFIX=$HOME/jarvis PORT=8190 HTTPS_PORT=10443 UNIT=localrouter-jarvis dry=0 force=0 accept=0 nogpu=0 STORE=
topology="" node="" port_set=0 peers=() allow_dl=0
args=("$@")
while [ $# -gt 0 ]; do
  case $1 in
    --profile) profile=${2:?}; shift 2 ;;
    --models) list=${2:?}; shift 2 ;;
    --prefix) PREFIX=$(abspath "${2:?}"); shift 2 ;;
    --port) PORT=${2:?}; port_set=1; shift 2 ;;
    --https-port) HTTPS_PORT=${2:?}; shift 2 ;;
    --unit) UNIT=${2:?}; shift 2 ;;
    --models-dir) STORE=${2:?}; shift 2 ;;
    --accept) accept=1; shift ;;
    --no-gpu) nogpu=1; shift ;;
    --dry-run) dry=1; shift ;;
    --force) force=1; shift ;;
    --topology) topology=${2:?}; shift 2 ;;
    --node) node=${2:?}; shift 2 ;;
    --peer-store) peers+=("${2:?}"); shift 2 ;;
    --allow-download) allow_dl=1; shift ;;
    -h | --help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done
[ -n "$profile" ] || die "--profile is required (have: $(profiles))"
if [ $darwin = 1 ]; then
  for t in jq curl shasum awk; do command -v $t > /dev/null || die "needs $t first: brew install jq (curl, shasum and awk come with macOS)"; done
else
  for t in jq curl sha256sum awk; do command -v $t > /dev/null || die "needs $t first: sudo apt-get install -y jq curl coreutils gawk"; done
fi
p=$(profile_file "$profile") || exit 1
topo_note="" listen=127.0.0.1 fbk=" "
if [ -n "$topology$node" ]; then # one node of several: the profile's models it runs
  use_topology
  fb=" $(jq -r --arg n "$node" '.nodes[] | select(.id == $n and .role == "front") | .fallback // [] | join(" ")' "$topology") "
  fbk+="$(jq -r --arg m "$fb" '.models[] | select(.id as $i | $m | contains(" " + $i + " ")) | .key' "$p" | paste -sd ' ') "   # loaded only while their node is down
  extra=$(for t in $(topo_local "$topology" "$node"); do jq -e --arg t "$t" 'any(.models[]; .id == $t)' "$p" > /dev/null || echo "$t"; done | paste -sd ' ')
  topo_note="node $node of $topology ($(topo_role "$topology" "$node"))${extra:+; it also runs $extra, which $profile does not install: set those up by hand first (deploy/jarvis/README.md)}"
fi
readlines keys < <(selected "$p" "$list")
[ ${#keys[@]} -gt 0 ] || [ "$(jq -r .status "$p")" = skeleton ] || die "no models selected"   # a skeleton: the host's steps alone
step_env "$p"
# the install to reuse model files from: --models-dir, else the default prefix when this one is elsewhere
[ -z "$STORE" ] && [ "$PREFIX" != "$HOME/jarvis" ] && [ -d "$HOME/jarvis" ] && STORE=$HOME/jarvis
[ "$STORE" = none ] && STORE=
if [ -n "$STORE" ]; then
  STORE=$(realpath "$STORE") || die "--models-dir $STORE does not exist"
  [ "$STORE" != "$PREFIX" ] || STORE=
fi
if [ -n "$STORE" ]; then
  [ -d "$STORE/uv-cache" ] && export UV_CACHE_DIR=$STORE/uv-cache
  [ -d "$STORE/zig-cache" ] && export ZIG_GLOBAL_CACHE_DIR=$STORE/zig-cache
fi
# reusable KEY: the model's reuse paths present in the store and not yet in this prefix, one a line
reusable() {
  [ -n "$STORE" ] || return 0
  local r
  jq -r --arg k "$1" '.models[] | select(.key == $k) | .reuse.paths[]? // empty' "$p" | while read -r r; do
    [ -e "$STORE/$r" ] && [ ! -e "$PREFIX/$r" ] && echo "$r"
  done
}
# the other nodes' stores (--peer-store): each as NODE, its ssh login and its install dir; one reachable login each
pnode=() plogin=() pdir=()
pssh() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$@"; }
for spec in "${peers[@]}"; do
  [[ $spec =~ ^([^=]+)=([^@]+)@([^:]+):(/.*)$ ]] || die "--peer-store $spec: give NODE=USER@HOST:/DIR"
  for t in ssh rsync; do command -v $t > /dev/null || die "--peer-store needs $t: sudo apt-get install -y openssh-client rsync"; done
  n=${BASH_REMATCH[1]} u=${BASH_REMATCH[2]} h=${BASH_REMATCH[3]} d=${BASH_REMATCH[4]} login=""
  for a in $([ -n "$topology" ] && topo_addrs "$topology" "$n") "$h"; do   # its ConnectX-7 links first
    pssh "$u@$a" true < /dev/null 2> /dev/null && { login=$u@$a; break; }
  done
  if [ -z "$login" ]; then
    [ $dry = 1 ] || die "--peer-store $spec: no ssh login (ssh -o BatchMode=yes $u@$h true fails): ssh-copy-id $u@$h first"
    echo "warning: --peer-store $spec: no ssh login (ssh -o BatchMode=yes $u@$h true fails); left out"; continue
  fi
  pnode+=("$n") plogin+=("$login") pdir+=("$d")
done
# what to copy from a peer: a model with a weights step (reused) still to do, the paths not in the store, from the
# first peer that has them (rsync then sends only what differs, so a copy that broke off resumes)
if [ "${BASH_VERSINFO[0]}" -ge 4 ]; then declare -A copy_list=() copy_from=(); else
  [ ${#pnode[@]} = 0 ] || die "--peer-store needs bash 4 or newer (macOS: brew install bash, then run this with it)"
fi
fetching() {
  local s
  for s in "${todo[@]}"; do [ "$(jq -r '.model + " " + (.reused // false | tostring)' <<< "$s")" = "$1 true" ] && ! is_done "$s" && return 0; done
  return 1
}
LOGS=$PREFIX/install/logs
again="$0 ${args[*]}"
again=${again// --dry-run/}

echo "== $profile ($(jq -r .status "$p")): $(jq -r .description "$p")"
echo "box: $(hw_summary)"
echo "prefix $PREFIX, port $PORT, HTTPS port $HTTPS_PORT (0: none), unit $UNIT; models: ${keys[*]}"
[ -z "$topo_note" ] || echo "$topo_note"

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
readlines todo < <(steps "$p" "${keys[@]}")
need_dl=0 need_disk=0 warm=0 peak=0 accepts=() saved=0 reuse_lines=() reusing=" "
for k in "${keys[@]}"; do   # what the store already has: linked, not downloaded
  while read -r r; do
    [ -n "$r" ] || continue
    b=$(du -s --apparent-size -B1 "$STORE/$r" 2> /dev/null | awk '{print $1}')
    reusing+="$k "
    reuse_lines+=("$(printf '  %-5s %-16s %s: %s GiB already present, reusing (hard links; checked against the pins)' reuse "$k" "$STORE/$r" "$(gib "${b:-0}")")")
  done < <(reusable "$k")
done
copying=" " copied=0 weights=() weights_dl=0
for k in "${keys[@]}"; do   # what a peer has: copied, not downloaded
  [ ${#pnode[@]} -gt 0 ] && fetching "$k" || continue
  while read -r r; do
    [ -n "$r" ] && { [ -z "$STORE" ] || [ ! -e "$STORE/$r" ]; } || continue
    for i in "${!pnode[@]}"; do
      q=$(printf %q "${pdir[$i]}/$r")
      b=$(pssh "${plogin[$i]}" "test -e $q && du -s --apparent-size -B1 $q" < /dev/null 2> /dev/null | awk '{print $1}')
      [ -n "$b" ] || continue
      copy_list[$k]+="$r " copy_from[$k/$r]=$i copying+="$k "
      reuse_lines+=("$(printf '  %-5s %-16s %s: %s GiB, copy from %s (rsync from %s; checked against the pins)' copy "$k" "$r" "$(gib "$b")" "${pnode[$i]}" "${plogin[$i]}:${pdir[$i]}/$r")")
      break
    done
  done < <(jq -r --arg k "$k" '.models[] | select(.key == $k) | .reuse.paths[]? // empty' "$p")
done
for s in "${todo[@]}"; do   # what the steps still to do (in this run: not the GPU ones with --no-gpu) download and leave on disk
  is_done "$s" && continue
  [ $nogpu = 1 ] && [ "$(jq -r '.gpu // false' <<< "$s")" = true ] && continue
  if [ "$(jq -r '.reused // false' <<< "$s")" = true ] && [[ $reusing == *" $(jq -r .model <<< "$s") "* ]]; then
    saved=$((saved + $(jq '.download_bytes // 0' <<< "$s"))); continue   # the store has its result
  fi
  if [ "$(jq -r '.reused // false' <<< "$s")" = true ] && [[ $copying == *" $(jq -r .model <<< "$s") "* ]]; then
    copied=$((copied + $(jq '.download_bytes // 0' <<< "$s"))); need_disk=$((need_disk + $(jq '.disk_bytes // 0' <<< "$s"))); continue   # a peer has it
  fi
  if [ "$(jq -r '.reused // false' <<< "$s")" = true ] && [ "$(jq '.download_bytes // 0' <<< "$s")" -gt 0 ]; then   # model weights
    weights+=("$(step_id "$s")"); weights_dl=$((weights_dl + $(jq '.download_bytes' <<< "$s")))
  fi
  need_dl=$((need_dl + $(jq '.download_bytes // 0' <<< "$s"))); need_disk=$((need_disk + $(jq '.disk_bytes // 0' <<< "$s")))
done
for k in "${keys[@]}"; do
  m=$(model "$p" "$k")
  b=$(jq '.memory_bytes + (.working_bytes // 0)' <<< "$m")
  if [ "$(jq .warm <<< "$m")" = true ] && [[ $fbk != *" $k "* ]]; then warm=$((warm + b)); elif [ "$b" -gt "$peak" ]; then peak=$b; fi
  while read -r a; do [ -n "$a" ] && accepts+=("$k: $a"); done < <(jq -r '.accept[]? // empty' <<< "$m")
done
free=$(hw_disk_free "$PREFIX") margin=$(($(jq .disk_margin_gib "$p") * 1073741824))
echo "plan:"
for l in "${reuse_lines[@]}"; do echo "$l"; done
for s in "${todo[@]}"; do
  st=todo; [ "$(jq -r '.reused // false' <<< "$s")" = true ] && [[ $copying == *" $(jq -r .model <<< "$s") "* ]] && st=copy
  [ "$(jq -r '.reused // false' <<< "$s")" = true ] && [[ $reusing == *" $(jq -r .model <<< "$s") "* ]] && st=reuse; is_done "$s" && st=done
  printf '  %-5s %-16s %s%s\n' "$st" "$(step_id "$s")" "$(jq -r .title <<< "$s")" \
    "$(jq -r 'if .gpu then " [GPU, \(.min_free_gib) GiB free]" else "" end' <<< "$s")"
done
if jq -e .deploy "$p" > /dev/null; then
  svc="unit $UNIT"; [ "$(jq -r '.deploy.service // "systemd"' "$p")" = launchd ] && svc="launchd agent $(launchd_label "$UNIT")"
  printf '  %-5s %-16s %s\n' todo deploy "$(jq -r .deploy.install "$p") (config, $svc, service on $listen:$PORT), then $(basename "$(jq -r .deploy.start "$p")") until the warm models are ready"
else
  printf '  %-5s %-16s %s\n' none deploy "this profile has no deploy yet: the run ends after its steps"
fi
echo "needs: download $(gib $need_dl) GiB, disk $(gib $need_disk) GiB + $(gib $margin) margin (free: $(gib "$free") GiB)$([ $saved -gt 0 ] && echo "; $(gib $saved) GiB reused from $STORE instead of downloaded")$([ $copied -gt 0 ] && echo "; $(gib $copied) GiB copied from peers instead of downloaded")"
echo "memory: $(gib $warm) GiB for the warm models$([ $peak -gt 0 ] && echo ", plus up to $(gib $peak) GiB for an on-demand one (it unloads the image model to fit)") of $(gib "$(hw_mem)") GiB"
gpu_need=0   # the most free memory a GPU step still to do needs
for s in "${todo[@]}"; do
  [ "$(jq -r '.gpu // false' <<< "$s")" = true ] && ! is_done "$s" && [ "$(jq '.min_free_gib // 0' <<< "$s")" -gt $gpu_need ] && gpu_need=$(jq .min_free_gib <<< "$s")
done
avail_gib=$(($(hw_avail) / 1073741824))
[ $nogpu = 0 ] && [ $gpu_need -gt 0 ] && [ $avail_gib -lt $gpu_need ] &&
  echo "warning: only $avail_gib GiB of memory is available; the GPU steps need $gpu_need GiB free (INSTALL.md step 6: free the GPU, or run with --no-gpu now)"
if [ ${#accepts[@]} -gt 0 ]; then
  echo "needs acceptance (ask the user; with their yes, run with --accept):"
  for a in "${accepts[@]}"; do echo "  - $a"; done
  jq -r --arg keys " ${keys[*]} " '.models[] | select(.accept and (.key as $k | $keys | contains(" " + $k + " "))) | "    \(.key) licence: \(.licence.name): \(.licence.url)"' "$p"
fi
[ "$free" -ge $((need_disk + margin)) ] || { echo "not enough free disk under $PREFIX: $(gib "$free") GiB, needs $(gib $((need_disk + margin))) GiB"; [ $dry = 1 ] || exit 1; }
# with a topology or peers, weights come from a peer or the store: downloading them needs the user's yes
guard=0; { [ -n "$topology" ] || [ ${#peers[@]} -gt 0 ]; } && guard=1
if [ $guard = 1 ] && [ ${#weights[@]} -gt 0 ]; then
  echo "needs approval: no peer or store has these weights, so they would be downloaded ($(gib $weights_dl) GiB): ${weights[*]}"
  echo "  ask the user; with their yes, run with --allow-download (or name a peer that has them: --peer-store NODE=USER@HOST:DIR)"
fi
[ ${#accepts[@]} -gt 0 ] && [[ " $again " != *" --accept "* ]] && again="$again --accept   (only with the user's yes to the list above)"
if [ $dry = 1 ]; then
  hint=""
  [ ${#accepts[@]} -gt 0 ] && [[ " $again " != *" --accept"* ]] && hint=" --accept   (--accept only with the user's yes to the list above)"
  echo "dry run: nothing changed. To install: ${again}${hint}"
  exit 0
fi
[ $missing = 0 ] || die "install the missing prerequisites first (each 'fix' above), then run again: $again"
[ $guard = 0 ] || [ ${#weights[@]} = 0 ] || [ $allow_dl = 1 ] ||
  die "STOP: downloading ${weights[*]} needs the user's yes (above). With it: $again --allow-download"

# ---- 3. the steps
mkdir -p "$LOGS" "$PREFIX/bin"
t_all=$(date +%s)
for s in "${todo[@]}"; do
  id=$(step_id "$s")
  k=$(jq -r .model <<< "$s")
  if [ "$k" != "${reused:-}" ]; then   # a model's first step: link what the store has, copy what a peer has (its licence accepted)
    reused=$k
    readlines rs < <(reusable "$k")
    cs=(); [ ${#pnode[@]} = 0 ] || read -ra cs <<< "${copy_list[$k]:-}"
    if [ $((${#rs[@]} + ${#cs[@]})) -gt 0 ] && { [ $accept = 1 ] || [ "$(model "$p" "$k" | jq '(.accept // []) | length')" = 0 ]; }; then
      for r in "${rs[@]}"; do echo "reuse $k: $STORE/$r"; $STEPS reuse "$STORE/$r" "$PREFIX/$r" >> "$LOGS/$k-reuse.log" 2>&1; done
      for r in "${cs[@]}"; do
        i=${copy_from[$k/$r]}; src=${plogin[$i]}:${pdir[$i]}/$r
        echo "copy  $k: $r from ${pnode[$i]} ($src; log: $LOGS/$k-copy.log)"
        mkdir -p "$(dirname "$PREFIX/$r")"
        rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=5" "$src" "$(dirname "$PREFIX/$r")/" >> "$LOGS/$k-copy.log" 2>&1 && continue
        [ $allow_dl = 1 ] && { echo "      the copy failed (log above): the steps download instead (--allow-download)"; continue; }
        die "the copy of $r from ${pnode[$i]} failed: the end of $LOGS/$k-copy.log says why; fix it and run again (it resumes): $again"
      done
      v=$(model "$p" "$k" | jq -r '.reuse.verify // empty')
      if [ -n "$v" ]; then
        echo "      checking the reused files against the pins (log: $LOGS/$k-reuse.log)"
        bash -c "set -euo pipefail; $v" >> "$LOGS/$k-reuse.log" 2>&1 || echo "      they differ from the pins: the steps make them as usual"
      fi
    fi
  fi
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
    [ "$(grep -vc '^== ' "$log")" = 0 ] && echo "  (the step printed nothing before it stopped: a bug in the step, not on this box; report it with the log)" >&2
    echo "next: $(jq -r .fix <<< "$s")" >&2
    echo "then run the same command again (it resumes here): $again" >&2
    exit 1
  fi
  echo "ok    $id in $(($(date +%s) - t)) s"
done
if [ $nogpu = 1 ]; then echo "== --no-gpu: the GPU steps are left; when the GPU is free run: ${again// --no-gpu/}"; exit 0; fi

jq -e .deploy "$p" > /dev/null || { echo "== the steps are done; $profile has no deploy yet (its manifest names none)"; exit 0; }

# ---- 4. deploy: the service on the selected models, by the manifest's deploy.install and deploy.start (deploy/jarvis:
# a systemd user unit; deploy/mac: a launchd user agent). install.sh waits for open voice sessions before a restart.
echo "run   deploy (log: $LOGS/deploy.log)"
tools=$(for k in "${keys[@]}"; do jq -r --arg k "$k" '.models[] | select(.key == $k) | .id' "$p"; done | tr '\n' ' ')
{ TOPOLOGY=$topology NODE=$node TOOLS=$tools SKIP_BUILD=1 bash "$repo/$(jq -r .deploy.install "$p")" &&
    bash "$repo/$(jq -r .deploy.start "$p")"; } > >(tee -a "$LOGS/deploy.log") 2>&1 || {
  echo "FAILED deploy: the end of $LOGS/deploy.log is above. next: tools/install/doctor.sh --profile $profile --prefix $PREFIX --port $PORT --unit $UNIT${topology:+ --topology $topology --node $node}; fix what it lists, then run again: $again" >&2
  exit 1
}
if [ "$listen" != 127.0.0.1 ]; then   # a node that is not the front: the API and its checks are the front's
  echo "== done in $(($(date +%s) - t_all)) s: $UNIT serves node $node's models on http://$listen:$PORT for the front"
  echo "check it from the front: tools/install/doctor.sh --profile $profile --topology $topology --node <the front>"
  exit 0
fi
svc=$UNIT; [ "$(jq -r '.deploy.service // "systemd"' "$p")" = launchd ] && svc=$(launchd_label "$UNIT")
echo "== done in $(($(date +%s) - t_all)) s: $svc serves http://127.0.0.1:$PORT (voice: /voice, API: /v1, MCP: /mcp)"
if [ "$(jq -r .deploy.install "$p")" = deploy/jarvis/install.sh ]; then
  echo "check it: LOCK=/dev/null bash $repo/tools/jarvis/smoke.sh --quick --url http://127.0.0.1:$PORT"
else
  echo "check it: bash $repo/$(jq -r .deploy.status "$p")"
fi
