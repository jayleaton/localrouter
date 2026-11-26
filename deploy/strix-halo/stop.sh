#!/usr/bin/env bash
# Stops the daemon (its workers stop their model servers), then any Gufo container a killed worker left behind.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$here/../.." && pwd)
source "$here/env.sh"; [ -z "${SITE:-}" ] || source "$SITE"
systemctl --user stop "$UNIT" 2> /dev/null
left=$(podman ps -q --filter name=lr-gufo- 2> /dev/null)
[ -z "$left" ] || { echo "stopping leftover containers: $(podman ps --format '{{.Names}}' --filter name=lr-gufo- | tr '\n' ' ')"; podman stop -t 10 $left > /dev/null; }
source "$REPO/tools/install/lib.sh"
hw_amd && echo "GTT in use: $(gib "$(hw_gtt_used)") GiB"
exit 0
