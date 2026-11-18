#!/usr/bin/env bash
# Runs one Gufo server (https://github.com/gufo-org/gufo, the AMD Strix Halo engine) in rootless Podman, in the
# foreground, for LocalRouter's `openai_server` engine and the openai_audio `cmd` worker:
#   tools/gufo/serve.sh NAME PORT [podman options...] -- gufo serve ... [--port 8080 is added]
# The container listens on 127.0.0.1:PORT only. SIGTERM/SIGINT (the daemon unloading the tool) stops the container and
# waits until it is gone, so its GPU memory is free before this script exits. A SIGKILL of this script's process group
# (the daemon's, on abort or timeout: the trap never runs, and conmon keeps the container going outside the group) is
# covered by a watcher in its own session, which removes this run's container (by id) once the script is gone.
# `--replace` clears a container left over from anything else.
# Environment: GUFO_IMAGE (default: tools/gufo/IMAGE, the runtime pinned by digest), GUFO_STOP_TIMEOUT (default 10 s),
# TZ (the host zone passed into the container; unset: read from the host). The image has no zoneinfo of its own, so
# when the host has one it is mounted read-only at /usr/share/zoneinfo (GUFO_ZONEINFO overrides that host path).
# An image named without @sha256: is refused unless GUFO_ALLOW_UNPINNED=1: a tag can move between two installs.
# Devices: /dev/kfd and /dev/dri; the video and render groups are passed by name, since keep-groups needs crun and
# many hosts run runc.
# Logs: the container's output reaches the daemon through this script's stdout and stderr, which it copies into the
# tool's log. podman's default log driver on a systemd host (journald) also wrote every line to the journal, stderr at
# priority 3, so each worker message appeared twice and as an error. --log-driver=none keeps the attached output and
# drops the journal copy (`podman logs` then has nothing; the tool's log has it all). A --log-driver among the podman
# options given to this script comes later on the command line and wins.
set -euo pipefail
name=${1:?usage: serve.sh NAME PORT [podman options...] -- gufo serve ...}; port=${2:?port}; shift 2
popts=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do popts+=("$1"); shift; done
[ "${1:-}" = -- ] || { echo "serve.sh: missing -- before the gufo command" >&2; exit 2; }
shift
image=${GUFO_IMAGE:-$(cat "$(dirname "$0")/IMAGE")}
case $image in
  *@sha256:????????????????????????????????????????????????????????????????) ;;
  *) [ "${GUFO_ALLOW_UNPINNED:-0}" = 1 ] || { echo "serve.sh: GUFO_IMAGE '$image' is not pinned by digest (name@sha256:...); GUFO_ALLOW_UNPINNED=1 runs it anyway" >&2; exit 2; } ;;
esac
# A zone name only (Area/City, UTC, ...): the container's clock matches the host's, and a value with a newline or a
# shell metacharacter cannot become another podman argument.
host_tz() {
  local tz="${TZ:-}"
  if [ -z "$tz" ] && [ -r /etc/timezone ]; then tz=$(tr -d ' \t\r\n' < /etc/timezone); fi
  if [ -z "$tz" ]; then
    local link
    link=$(readlink /etc/localtime 2> /dev/null || true)
    case $link in
      */zoneinfo/*) tz=${link#*/zoneinfo/} ;;
    esac
  fi
  case ${tz:-UTC} in
    *[!A-Za-z0-9_+/-]*) printf '%s' UTC ;;
    *) printf '%s' "${tz:-UTC}" ;;
  esac
}
tz=$(host_tz)
# TZ alone leaves the container on UTC: the image ships neither /usr/share/zoneinfo nor /etc/localtime.
zoneinfo=${GUFO_ZONEINFO:-/usr/share/zoneinfo}
zi=()
if [ -d "$zoneinfo" ]; then zi=(-v "${zoneinfo}:/usr/share/zoneinfo:ro"); fi
cname="lr-gufo-$name"
stop() { podman stop -t "${GUFO_STOP_TIMEOUT:-10}" "$cname" > /dev/null 2>&1 || true; wait "$pid" 2> /dev/null || true; exit 0; }
trap stop TERM INT
# "gufo serve [global opts] KIND ...": insert host/port right after "serve"
args=(); placed=0
for a in "$@"; do args+=("$a"); if [ $placed = 0 ] && [ "$a" = serve ]; then args+=(--host 0.0.0.0 --port 8080); placed=1; fi; done
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/lr-gufo-$name.XXXXXX")
# the watcher: polls this script; when it is gone, removes the container this run made (not a later run's of the name)
setsid -f sh -c 'while kill -0 "$1" 2> /dev/null; do sleep 0.5; done
  [ -s "$2/cid" ] && podman rm -f -t "$3" "$(cat "$2/cid")" > /dev/null 2>&1; rm -rf "$2"' \
  gufo-watch "$$" "$run_dir" "${GUFO_STOP_TIMEOUT:-10}" < /dev/null > /dev/null 2>&1
podman run --rm --replace --cidfile "$run_dir/cid" --name "$cname" --userns=keep-id:uid="$(id -u)",gid="$(id -g)" \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render --ulimit memlock=-1 \
  --log-driver=none -e "TZ=$tz" ${zi[@]+"${zi[@]}"} -p "127.0.0.1:$port:8080" ${popts[@]+"${popts[@]}"} "$image" "${args[@]}" &
pid=$!
wait "$pid"
