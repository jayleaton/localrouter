#!/usr/bin/env bash
# Runs a model's OpenAI-compatible server in a Docker container with the NVIDIA GPUs, in the foreground, for
# LocalRouter's `openai_server` engine:
#   tools/docker/serve.sh NAME [docker run options...] -- IMAGE [command...]
# The container is lr-NAME (characters Docker refuses in a name become _). The options choose its network and mounts:
# with --network host the server's own --host 127.0.0.1 --port {port} keeps it on loopback. SIGTERM/SIGINT (the daemon
# unloading the tool) stops the container (DOCKER_STOP_TIMEOUT, default 10 s) and waits until it is gone, so its GPU
# memory is free before this script exits. A SIGKILL of this script's process group (the daemon's, on abort or
# timeout: the trap never runs, and dockerd keeps the container going outside the group) is covered by a watcher in
# its own session, which removes this run's container (by id) once the script is gone: exited, or a zombie its parent
# has not reaped yet. A container of the same name left over from anything else is removed first.
# IMAGE must be pinned: a local image id (sha256:<64 hex>) or name@sha256:<64 hex>. A tag can move between two
# installs, so one is refused unless DOCKER_ALLOW_UNPINNED=1.
set -euo pipefail
name=${1:?usage: serve.sh NAME [docker run options...] -- IMAGE [command...]}; shift
dopts=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do dopts+=("$1"); shift; done
[ "${1:-}" = -- ] || { echo "serve.sh: missing -- before the image" >&2; exit 2; }
shift
image=${1:?serve.sh: no image after --}; shift
case $image in
  sha256:????????????????????????????????????????????????????????????????) ;;
  *@sha256:????????????????????????????????????????????????????????????????) ;;
  *) [ "${DOCKER_ALLOW_UNPINNED:-0}" = 1 ] || { echo "serve.sh: image '$image' is not pinned (sha256:<id> or name@sha256:<digest>); DOCKER_ALLOW_UNPINNED=1 runs it anyway" >&2; exit 2; } ;;
esac
cname="lr-$(printf '%s' "$name" | tr -c 'A-Za-z0-9_.-' _)"
stop_s=${DOCKER_STOP_TIMEOUT:-10}
pid=
stop() {
  docker stop -t "$stop_s" "$cname" > /dev/null 2>&1 || true
  [ -z "$pid" ] || wait "$pid" 2> /dev/null || true
  exit 0
}
trap stop TERM INT
docker rm -f "$cname" > /dev/null 2>&1 || true
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/$cname.XXXXXX")
# the watcher: polls this script; when it is gone, removes the container this run made (not a later run's of the name).
# A zombie counts as gone: `kill -0` succeeds on one, so a parent that is slow to reap this script (a subreaper) would
# keep the container running after a group SIGKILL. Where /proc is mounted the watcher reads the State: line instead.
setsid -f sh -c 'running() {
    if [ -d /proc/self ]; then
      st=$(sed -n "s/^State:[[:space:]]*\([A-Za-z]\).*/\1/p" "/proc/$1/status" 2> /dev/null)
      [ -n "$st" ] && [ "$st" != Z ] && [ "$st" != X ]
    else kill -0 "$1" 2> /dev/null; fi
  }
  while running "$1"; do sleep 0.5; done
  [ -s "$2/cid" ] && docker rm -f "$(cat "$2/cid")" > /dev/null 2>&1; rm -rf "$2"' \
  docker-watch "$$" "$run_dir" < /dev/null > /dev/null 2>&1
docker run --rm --cidfile "$run_dir/cid" --name "$cname" --label localrouter.tool="$name" --gpus all \
  ${dopts[@]+"${dopts[@]}"} "$image" "$@" &
pid=$!
wait "$pid"
