#!/usr/bin/env bash
# LocalRouter's launchd agent at a glance: its state, the build, each model's state and memory, the URLs, the logs.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
[ -f "$PLIST" ] || { echo "agent:   $LABEL is not installed ($PLIST)"; exit 1; }
last=$(launchctl print "$DOMAIN/$LABEL" 2> /dev/null | sed -n 's/^[[:space:]]*last exit code = //p' | head -1)
echo "agent:   $LABEL $(agent_state)$([ -n "$(agent_pid)" ] && echo ", pid $(agent_pid)")$([ -n "$last" ] && echo ", last exit $last")"
echo "build:   $(readlink "$DEPLOY/current" 2> /dev/null)"
echo "health:  $(curl -fs -m 3 "$URL/health" || echo down)"
lr_curl -fs -m 5 "$URL/v1/tools" | jq -r '"memory:  available \(.mem_available_bytes / 1073741824 * 10 | round / 10) GiB, reserve \(.reserve_bytes / 1073741824 * 10 | round / 10) GiB",
  (.data[] | "  \(.id | . + " " * (24 - length)) \(.state | . + " " * (9 - length)) \(.resident_bytes / 1073741824 * 10 | round / 10) GiB  priority \(.priority)\(if .keep_loaded then ", kept warm" else "" end)\(if .last_error != "" then "  last error: " + .last_error else "" end)")' 2> /dev/null
pub=$(public_url || true)
echo "urls:    ${pub:+$pub, }loopback $URL (voice: /voice, API: /v1, MCP: /mcp)"
echo "logs:    $LOG; models: $DEPLOY/data/logs/"
