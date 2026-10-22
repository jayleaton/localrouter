#!/usr/bin/env bash
# What a box lacks for a profile (INSTALL.md), each with the command that fixes it: the hardware, the prerequisites,
# every install step of the manifest (deploy/profiles/PROFILE.json, by its own `done` test), then the service: unit,
# health, the models' states and memory. Changes nothing.
#   tools/install/doctor.sh --profile jarvis-gb10 [--models chat,tts,stt,image,video] [--prefix DIR] [--port N] [--unit NAME]
# Exit 0 when nothing is missing (warnings do not count), 1 otherwise.
set -uo pipefail
source "$(dirname "$0")/lib.sh"

profile="" list=all PREFIX=$HOME/jarvis PORT=8190 HTTPS_PORT=10443 UNIT=localrouter-jarvis
while [ $# -gt 0 ]; do
  case $1 in
    --profile) profile=${2:?}; shift 2 ;;
    --models) list=${2:?}; shift 2 ;;
    --prefix) PREFIX=$(realpath -m "${2:?}"); shift 2 ;;
    --port) PORT=${2:?}; shift 2 ;;
    --https-port) HTTPS_PORT=${2:?}; shift 2 ;;
    --unit) UNIT=${2:?}; shift 2 ;;
    -h | --help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument $1 (see --help)" ;;
  esac
done
[ -n "$profile" ] || die "--profile is required (have: $(profiles))"
command -v jq > /dev/null || { echo "MISSING jq"; echo "  fix: sudo apt-get install -y jq"; exit 1; }
p=$(profile_file "$profile") || exit 1
mapfile -t keys < <(selected "$p" "$list")
step_env "$p"
boot="tools/install/bootstrap.sh --profile $profile --models $(IFS=,; echo "${keys[*]}") --prefix $PREFIX --port $PORT --https-port $HTTPS_PORT --unit $UNIT"
bad=0
ok() { echo "ok      $*"; }
miss() { echo "MISSING $1"; echo "        fix: $2"; bad=1; }
warn() { echo "warning $1"; [ -n "${2:-}" ] && echo "        fix: $2"; }

echo "== box: $(hw_summary)"
while read -r line; do
  [ -n "$line" ] || continue
  case $line in warning:*) warn "${line#warning: }" ;; *) miss "$line" "this box does not fit $profile (INSTALL.md, Profiles)" ;; esac
done < <(hw_match "$p")
avail=$(hw_disk_free "$PREFIX")
echo "        disk free under $PREFIX: $(gib "$avail") GiB"
[ $(($(hw_avail) / 1073741824)) -ge 88 ] || systemctl --user is-active -q "$UNIT" || warn "only $(gib "$(hw_avail)") GiB of memory is available: the GPU steps need up to 88 GiB free and the warm models about 107 GiB" "see what holds it: nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv (INSTALL.md step 6)"

echo "== prerequisites"
while read -r q; do
  if ok_test "$q"; then ok "$(jq -r .title <<< "$q")"; else miss "$(jq -r .title <<< "$q")" "$(jq -r .fix <<< "$q")"; fi
done < <(prereqs "$p" "${keys[@]}")

echo "== install steps (under $PREFIX)"
while read -r s; do
  if is_done "$s"; then ok "$(step_id "$s"): $(jq -r .title <<< "$s")"
  else miss "$(step_id "$s"): $(jq -r .title <<< "$s")" "$boot   (it resumes at the first step not done)"; fi
done < <(steps "$p" "${keys[@]}")

echo "== service $UNIT on 127.0.0.1:$PORT"
url=http://127.0.0.1:$PORT
if [ ! -f "$HOME/.config/systemd/user/$UNIT.service" ]; then
  miss "the unit $UNIT is not installed" "$boot"
elif ! systemctl --user is-active -q "$UNIT"; then
  miss "the unit $UNIT is installed but not running ($(systemctl --user is-active "$UNIT"))" "bash $repo/deploy/jarvis/start.sh   (why it stopped: journalctl --user -u $UNIT -n 50)"
else
  ok "unit $UNIT active since $(systemctl --user show "$UNIT" -p ActiveEnterTimestamp --value)"
  [ "$(systemctl --user is-enabled "$UNIT" 2> /dev/null)" = enabled ] || warn "the unit does not start at boot" "systemctl --user enable $UNIT"
  if ! curl -fs -m 5 "$url/health" > /dev/null; then
    miss "the daemon does not answer on $url/health" "journalctl --user -u $UNIT -n 50; then systemctl --user restart $UNIT"
  else
    ok "health on $url"
    t=$(curl -fs -m 10 "$url/v1/tools")
    for k in "${keys[@]}"; do
      m=$(model "$p" "$k")
      id=$(jq -r .id <<< "$m")
      st=$(jq -r --arg id "$id" '.data[] | select(.id == $id) | .state' <<< "$t")
      if [ -z "$st" ]; then miss "$k ($id) is not in the service's config" "$boot"
      elif [ "$(jq -r .warm <<< "$m")" = true ] && [ "$st" != ready ]; then
        err=$(jq -r --arg id "$id" '.data[] | select(.id == $id) | .last_error' <<< "$t")
        miss "$k ($id) should be warm, is $st${err:+ (last error: $err)}" "wait for it (start.sh waits until warm), else journalctl --user -u $UNIT and $PREFIX/deploy/data/logs/"
      else ok "$k ($id): $st"; fi
    done
    q=$(curl -fs -m 10 "$url/v1/models" | jq -r '[.data[] | select(.qualified == false) | .id] | join(" ")')
    [ -z "$q" ] || miss "not GPU-qualified on this box: $q" "$boot   (it runs each model's qualify step)"
    echo "        MemAvailable $(jq -r '.mem_available_bytes / 1073741824 * 10 | round / 10' <<< "$t") GiB, reserve $(jq -r '.reserve_bytes / 1073741824' <<< "$t") GiB, voice sessions $(jq -r '.voice_sessions // "?"' <<< "$t")"
  fi
fi
if [ "$HTTPS_PORT" != 0 ]; then
  if ! command -v tailscale > /dev/null; then warn "no tailscale: reach the service with ssh -L $PORT:127.0.0.1:$PORT (deploy/jarvis/README.md)"
  elif tailscale serve status 2> /dev/null | grep -q "127.0.0.1:$PORT"; then ok "tailscale serve: HTTPS $HTTPS_PORT to 127.0.0.1:$PORT"
  else warn "tailscale serve has no HTTPS entry for 127.0.0.1:$PORT" "sudo tailscale serve --bg --https=$HTTPS_PORT http://127.0.0.1:$PORT"; fi
fi

[ $bad = 0 ] && { echo "== nothing missing"; exit 0; }
echo "== something is missing: do each fix above (bootstrap.sh does the install steps and resumes where it stopped)"
exit 1
