#!/usr/bin/env bash
# After the service stops (ExecStopPost): the Drama 3 server runs in a transient user unit of its own, outside the
# service's cgroup, so a worker killed before it could stop it (the stop timeout) leaves it holding about 42 GiB. The
# worker names that unit in $DRAMA_ROOT/runtime/localrouter-drama3.unit while it runs it; only that unit is stopped.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
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
