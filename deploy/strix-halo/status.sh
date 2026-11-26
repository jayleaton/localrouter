#!/usr/bin/env bash
# The unit, /health, each tool's state, the model containers and GTT.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$here/../.." && pwd)
source "$here/env.sh"; [ -z "${SITE:-}" ] || source "$SITE"
source "$REPO/tools/install/lib.sh"
echo "unit $UNIT: $(systemctl --user is-active "$UNIT" 2> /dev/null)"
curl -fs -m 3 "http://127.0.0.1:$PORT/health" > /dev/null && echo "health: ok" || echo "health: no answer on :$PORT"
curl -fs -m 5 "http://127.0.0.1:$PORT/v1/tools" 2> /dev/null | jq -r '.data[] | "  \(.id): \(.state)"' 2> /dev/null
echo "containers: $(podman ps --format '{{.Names}}' --filter name=lr-gufo- | tr '\n' ' ')"
hw_amd && echo "GTT in use: $(gib "$(hw_gtt_used)") of $(gib "$(hw_gtt)") GiB"
