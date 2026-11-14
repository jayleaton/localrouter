#!/usr/bin/env bash
# After the service stops (ExecStopPost), what a killed worker can leave running outside the service's cgroup:
# - the Drama 3 server runs in a transient user unit of its own, so a worker killed before it could stop it (the stop
#   timeout) leaves it holding about 42 GiB. The worker names that unit in $DRAMA_ROOT/runtime/localrouter-drama3.unit
#   while it runs it; only that unit is stopped.
# - a model served from a Docker container (tools/chat/qwen38-flash-next/serve.sh) runs under dockerd. Its server stops
#   the container itself, and a watchdog does when the server is killed; any container still running with this
#   install's label (org.localrouter.owner=$DEPLOY) is stopped here, by its id. Other containers are not touched.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
if command -v docker > /dev/null; then
  for c in $(docker ps -q --filter "label=org.localrouter.owner=$DEPLOY" 2> /dev/null); do
    echo "stopping container $c ($(docker inspect -f '{{.Name}}' "$c" 2> /dev/null)), left running by a model's server"
    docker stop -t 20 "$c" > /dev/null || true
  done
fi
f=$DRAMA_ROOT/runtime/localrouter-drama3.unit
unit=$(cat "$f" 2> /dev/null) || exit 0
case $unit in
  drama3-pilot-*.service)
    if systemctl --user is-active -q "$unit"; then
      echo "stopping $unit, left running by the service's Drama 3 worker"
      systemctl --user stop "$unit"
    fi
    rm -f "$f" ;;
esac
exit 0
