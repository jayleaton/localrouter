#!/usr/bin/env bash
# Runs one Gufo server (https://github.com/gufo-org/gufo, the AMD Strix Halo engine) in rootless Podman, in the
# foreground, for LocalRouter's `openai_server` engine and the openai_audio `cmd` worker:
#   tools/gufo/serve.sh NAME PORT [podman options...] -- gufo serve ... [--port 8080 is added]
# The container listens on 127.0.0.1:PORT only. SIGTERM/SIGINT (the daemon unloading the tool) stops the container and
# waits until it is gone, so its GPU memory is free before this script exits. `--replace` clears a container left over
# from a worker that was killed outright.
# Environment: GUFO_IMAGE (default ghcr.io/gufo-org/toolboxes/gufo-runtime:latest), GUFO_STOP_TIMEOUT (default 10 s).
# Devices: /dev/kfd and /dev/dri; the video and render groups are passed by name, since keep-groups needs crun and
# many hosts run runc.
set -euo pipefail
name=${1:?usage: serve.sh NAME PORT [podman options...] -- gufo serve ...}; port=${2:?port}; shift 2
popts=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do popts+=("$1"); shift; done
[ "${1:-}" = -- ] || { echo "serve.sh: missing -- before the gufo command" >&2; exit 2; }
shift
image=${GUFO_IMAGE:-ghcr.io/gufo-org/toolboxes/gufo-runtime:latest}
cname="lr-gufo-$name"
stop() { podman stop -t "${GUFO_STOP_TIMEOUT:-10}" "$cname" > /dev/null 2>&1 || true; wait "$pid" 2> /dev/null || true; exit 0; }
trap stop TERM INT
# "gufo serve [global opts] KIND ...": insert host/port right after "serve"
args=(); placed=0
for a in "$@"; do args+=("$a"); if [ $placed = 0 ] && [ "$a" = serve ]; then args+=(--host 0.0.0.0 --port 8080); placed=1; fi; done
podman run --rm --replace --name "$cname" --userns=keep-id:uid="$(id -u)",gid="$(id -g)" \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render --ulimit memlock=-1 \
  -p "127.0.0.1:$port:8080" ${popts[@]+"${popts[@]}"} "$image" "${args[@]}" &
pid=$!
wait "$pid"
