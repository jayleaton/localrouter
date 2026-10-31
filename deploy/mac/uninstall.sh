#!/usr/bin/env bash
# Removes LocalRouter's launchd agent: stops it (and its models), deletes its plist and its tailscale serve entry.
# The install ($DEPLOY: builds, config, paired devices, logs) stays unless --purge; the models (under $J) always stay.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
agent_unload
rm -f "$PLIST"
ts=$(tailscale_cli)
if [ "$HTTPS_PORT" != 0 ] && [ -n "$ts" ] && "$ts" serve status 2> /dev/null | grep -q "127.0.0.1:$PORT"; then
  "$ts" serve --https="$HTTPS_PORT" off > /dev/null 2>&1 || echo "warning: could not remove tailscale serve --https=$HTTPS_PORT" >&2
fi
if [ "${1:-}" = --purge ]; then
  rm -rf "$DEPLOY"
  echo "removed $LABEL and $DEPLOY"
else
  echo "removed $LABEL; the install stays in $DEPLOY (--purge removes it)"
fi
