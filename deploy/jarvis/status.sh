#!/usr/bin/env bash
# The Jarvis service at a glance: the unit, the build, each model's state and memory, the machine's memory, the URLs.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
echo "unit:    $(systemctl --user is-active "$UNIT") ($(systemctl --user is-enabled "$UNIT" 2> /dev/null)), since $(systemctl --user show "$UNIT" -p ActiveEnterTimestamp --value)"
echo "build:   $(readlink "$DEPLOY/current" 2> /dev/null)"
echo "health:  $(curl -fs -m 3 "$URL/health" || echo down)"
curl -fs -m 5 "$URL/v1/tools" | jq -r '"memory:  MemAvailable \(.mem_available_bytes / 1073741824 * 10 | round / 10) GiB, reserve \(.reserve_bytes / 1073741824) GiB",
  (.data[] | "  \(.id | . + " " * (24 - length)) \(.state | . + " " * (9 - length)) \(.resident_bytes / 1073741824 * 10 | round / 10) GiB  priority \(.priority)\(if .keep_loaded then ", kept warm" else "" end)\(if .last_error != "" then "  last error: " + .last_error else "" end)")' 2> /dev/null
echo "urls:    $(public_url || true) (voice: /voice, API: /v1, MCP: /mcp); loopback $URL"
echo "logs:    journalctl --user -u $UNIT -f; models: $DEPLOY/data/logs/"
