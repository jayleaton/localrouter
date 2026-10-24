#!/usr/bin/env bash
# A GPU window on a live LocalRouter box (the Jarvis box): a bench, a test daemon, a qualification, anything that
# takes GPU memory outside the live scheduler. It never runs beside a voice call:
#   tools/spark/window.sh NAME [--unload TOOL]... -- CMD [ARG]...
#
# - Takes the box's lock ($BOX_LOCK, default $JARVIS/.dgx01.lock), waiting at most $WINDOW_LOCK_WAIT_S (5400 s).
# - Waits at most $WINDOW_CALL_WAIT_S (1800 s) until no realtime voice session is open on the live service ($URL).
# - Unloads each --unload TOOL from the live service (POST /v1/tools/TOOL/unload), and loads it again at the end.
#   Unload nothing else by hand: a tool the window unloads itself stays down after it (a kept model unloaded through
#   the API is not brought back), as the image model was on DGX-01 on 2026-10-24.
# - Runs CMD in its own process group, at most $WINDOW_MAX_S (2700 s), with WINDOW_LABEL=jarvis.window=NAME-PID in
#   its environment: a `docker run` in it must pass `--label "$WINDOW_LABEL"`, so the window can stop the container.
# - Every 2 s while CMD runs: if a voice session opens, the window is stopped at once (SIGTERM to its process group
#   and `docker stop` of its labelled containers, SIGKILL and `docker kill` 15 s later), so the call keeps the GPU.
# Exit: CMD's status; 75 when a voice session stopped it, or no quiet came in time; 74 without the lock.
# Logs to stderr with UTC times. Nothing is killed by name: only this window's own process group and containers.
set -uo pipefail
J=${JARVIS:-$HOME/jarvis}
URL=${URL:-http://127.0.0.1:8190}
LOCK=${BOX_LOCK:-$J/.dgx01.lock}
TOKEN_FILE=${TOKEN_FILE:-$J/deploy/operator.token}
name=${1:?usage: window.sh NAME [--unload TOOL]... -- CMD [ARG]...}
shift
unload=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case $1 in
    --unload) unload+=("${2:?--unload needs a tool id}"); shift 2 ;;
    *) echo "window.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ "${1:-}" = -- ] && [ $# -ge 2 ] || { echo "window.sh: usage: window.sh NAME [--unload TOOL]... -- CMD [ARG]..." >&2; exit 2; }
shift
log() { echo "window $name $(date -u +%T) $*" >&2; }
api() { # api METHOD PATH: the live service, with its operator token when it has one
  local auth=()
  [ -r "$TOKEN_FILE" ] && auth=(-H "authorization: Bearer $(cat "$TOKEN_FILE")")
  curl -fs -m 5 -X "$1" "${auth[@]}" "$URL$2"
}
# Open realtime voice sessions on the live service: a number; 0 when the service is down; "?" when it is up but its
# sessions cannot be read (then nothing may start, and a running window stops: it cannot see a call coming).
calls() {
  local body n
  if body=$(api GET /v1/tools); then
    n=$(jq -r '.voice_sessions // empty' <<< "$body" 2> /dev/null)
    [ -n "$n" ] && echo "$n" || echo "?"
  elif curl -fs -m 5 -o /dev/null "$URL/health"; then
    echo "?"
  else
    echo 0
  fi
}

exec 9> "$LOCK"
flock -w "${WINDOW_LOCK_WAIT_S:-5400}" 9 || { log "no lock within ${WINDOW_LOCK_WAIT_S:-5400} s"; exit 74; }
log "lock taken ($LOCK)"
waited=0
while [ "$(calls)" != 0 ]; do
  [ "$waited" -ge "${WINDOW_CALL_WAIT_S:-1800}" ] && { log "a voice session stayed open ($(calls)); not run"; exit 75; }
  [ "$waited" = 0 ] && log "a voice session is open ($(calls)): waiting for it to end"
  sleep 10
  waited=$((waited + 10))
done

done_unload=()
restore() {
  for t in "${done_unload[@]}"; do
    if api POST "/v1/tools/$t/load" > /dev/null; then log "reloaded $t"; continue; fi
    # refused (a call is open: loads wait for it) or not up yet: retried in the background for 15 min, since a kept
    # model the window unloaded does not come back by itself
    log "could not reload $t now; retrying every 10 s in the background"
    ( for _ in $(seq 90); do sleep 10; api POST "/v1/tools/$t/load" > /dev/null && exit 0; done ) > /dev/null 2>&1 &
    disown
  done
}
for t in "${unload[@]}"; do
  if api POST "/v1/tools/$t/unload" > /dev/null; then done_unload+=("$t"); log "unloaded $t from the live service"; else log "could not unload $t"; fi
done

export WINDOW_LABEL="jarvis.window=$name-$$"
containers() { command -v docker > /dev/null && docker ps -q --filter "label=$WINDOW_LABEL" 2> /dev/null; }
setsid timeout --kill-after=30 "${WINDOW_MAX_S:-2700}" "$@" &
child=$!
preempted=0
while kill -0 "$child" 2> /dev/null; do
  n=$(calls)
  if [ "$n" != 0 ]; then
    log "voice session open ($n): stopping the window so the call keeps the GPU"
    preempted=1
    kill -TERM -- "-$child" 2> /dev/null
    ids=$(containers); [ -n "$ids" ] && docker stop -t 10 $ids > /dev/null 2>&1 &
    for _ in $(seq 15); do kill -0 "$child" 2> /dev/null || break; sleep 1; done
    kill -KILL -- "-$child" 2> /dev/null
    ids=$(containers); [ -n "$ids" ] && docker kill $ids > /dev/null 2>&1
    break
  fi
  sleep 2
done
wait "$child"
rc=$?
restore
[ "$preempted" = 1 ] && { log "stopped for a voice session (command status $rc)"; exit 75; }
log "done, status $rc"
exit "$rc"
