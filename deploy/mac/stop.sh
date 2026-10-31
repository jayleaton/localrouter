#!/usr/bin/env bash
# Stops LocalRouter's launchd agent and every model with it (they are its children). It stays installed and starts
# again at the next login; start.sh starts it now, uninstall.sh removes it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
[ "$(agent_state)" != "not loaded" ] || { echo "$LABEL is not running"; exit 0; }
agent_unload
echo "stopped $LABEL"
