#!/usr/bin/env bash
# After the service stops (ExecStopPost), what runs outside the service's cgroup and would keep holding memory:
# - the Drama 3 server, in a transient user unit of its own: a worker killed before it could stop it (the stop timeout)
#   leaves it holding about 42 GiB. The worker names that unit in $DRAMA_ROOT/runtime/localrouter-drama3.unit while it
#   runs it; only that unit is stopped.
# - a container of tools/docker/serve.sh (dockerd runs it; serve.sh's watcher is in the cgroup and goes with it): the
#   deployed config's tools that run serve.sh, each container lr-<id>, are removed.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
if command -v docker > /dev/null; then
  jq -r '.tools[] | select((.options.server[0] // "") | endswith("/tools/docker/serve.sh"))
    | if .options.server[1] == "{id}" then .id else .options.server[1] end' "$DEPLOY/config.json" 2> /dev/null |
    while read -r name; do
      c=lr-$(printf '%s' "$name" | tr -c 'A-Za-z0-9_.-' _)
      if [ -n "$(docker ps -aq --filter "name=^$c\$" 2> /dev/null)" ]; then
        echo "removing container $c, left running by the service's $name worker"
        docker rm -f "$c" > /dev/null
      fi
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
