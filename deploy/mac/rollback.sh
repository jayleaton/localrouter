#!/usr/bin/env bash
# Goes back to what the last install.sh, redeploy.sh or rollback.sh replaced: the build current.prev names and
# config.json.prev (and the agent's $LABEL.plist.prev, if install.sh changed it), then a restart and the warm wait.
# What it replaces becomes the new *.prev, so a second rollback goes forward again. It waits for no open voice session
# first, as install.sh does (IDLE_WAIT_S, FORCE_RESTART).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
[ -f "$DEPLOY/current.prev" ] && [ -f "$DEPLOY/config.json.prev" ] ||
  die "nothing to roll back to: no $DEPLOY/current.prev and config.json.prev (install.sh or redeploy.sh keeps them)"
build=$(cat "$DEPLOY/current.prev")
[ -x "$DEPLOY/$build/zig-out/bin/localrouter" ] || die "the previous build $DEPLOY/$build is gone: nothing changed"
tmp=$(mktemp -d "$DEPLOY/rollback.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
cp -p "$DEPLOY/config.json.prev" "$tmp/config.json"
plist=
if [ -f "$DEPLOY/$LABEL.plist.prev" ] && ! cmp -s "$DEPLOY/$LABEL.plist.prev" "$PLIST"; then
  cp -p "$DEPLOY/$LABEL.plist.prev" "$tmp/agent.plist"; plist=$tmp/agent.plist
fi
loaded=0; [ "$(agent_state)" = "not loaded" ] || loaded=1
if [ $loaded = 1 ]; then
  wait_idle
  echo "== stopping $LABEL to roll back to ${build#src/}"
  agent_unload
fi
swap_in "$build" "$tmp/config.json" "$plist"
agent_up "${build#src/} (rolled back)"
wait_warm "${WARM_TIMEOUT_S:-1200}"
