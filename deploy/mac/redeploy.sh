#!/usr/bin/env bash
# Restarts LocalRouter onto a build already in $DEPLOY/src and/or a hand-edited config, without rebuilding or
# regenerating anything (install.sh does both, from this checkout):
#   deploy/mac/redeploy.sh [COMMIT]    COMMIT: a built $DEPLOY/src/COMMIT (default: the current one)
# A config staged as $DEPLOY/config.json.next replaces config.json; one of the two must change. It waits until no
# voice session is open (IDLE_WAIT_S, FORCE_RESTART as install.sh), keeps what it replaces as current.prev and
# config.json.prev (deploy/mac/rollback.sh goes back), restarts the agent and waits until the warm models are ready.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
[ -f "$PLIST" ] || die "not installed: deploy/mac/install.sh first"
cur=$(readlink "$DEPLOY/current" 2> /dev/null || true)
commit=${1:-${cur#src/}}
[ -n "$commit" ] || die "usage: deploy/mac/redeploy.sh COMMIT (no current build in $DEPLOY)"
[ -x "$DEPLOY/src/$commit/zig-out/bin/localrouter" ] ||
  die "no build at $DEPLOY/src/$commit (install.sh builds the checkout's HEAD there; SKIP_BUILD=1 reuses it)"
next=$DEPLOY/config.json.next
[ -f "$next" ] || next=
[ -n "$next" ] || [ "$cur" != "src/$commit" ] || die "nothing to change: $LABEL already runs $commit and no config.json.next is staged"
if [ -n "$next" ]; then
  jq -e '(.tools | length > 0) and (tostring | test("@[A-Z_]+@") | not)' "$next" > /dev/null ||
    die "$next is not a valid config (not JSON, no tools, or a @…@ left): nothing changed"
fi
loaded=0; [ "$(agent_state)" = "not loaded" ] || loaded=1
if [ $loaded = 1 ]; then
  wait_idle
  echo "== stopping $LABEL to restart it onto $commit${next:+ with config.json.next}"
  agent_unload
fi
swap_in "src/$commit" "$next" ""
agent_up "$commit"
wait_warm "${WARM_TIMEOUT_S:-1200}" || die "not warm: check $LOG; deploy/mac/rollback.sh goes back to $(cat "$DEPLOY/current.prev" 2> /dev/null || echo "the previous build")"
