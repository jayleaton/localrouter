#!/usr/bin/env bash
# Paired devices of LocalRouter on this Mac: the deployed binary's `localrouter pair` on the deployed config and data.
#   deploy/mac/pair.sh add "JJ's phone"     a code for the app (10 minutes, once) and, with tailscale serve, its link
#   deploy/mac/pair.sh token claude-code    a token now, for MCP clients and scripts (shown once)
#   deploy/mac/pair.sh list | revoke ID|NAME
# The daemon reads the devices file on every request: nothing here restarts it. Tokens are needed only once
# pairing.required is set (PAIRING_REQUIRED=1 install.sh).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
bin=$DEPLOY/current/zig-out/bin/localrouter
[ -x "$bin" ] || die "not installed: deploy/mac/install.sh first"
[ $# -ge 1 ] || exec "$bin" pair
extra=()
if [ "$1" = add ]; then
  url=$(public_url || true)
  [ -n "$url" ] && extra=(--url "$url")
fi
# the deployed paths first, so options given here (a different --url) win
exec "$bin" pair "$1" --config "$DEPLOY/config.json" --data "$DEPLOY/data" ${extra[@]+"${extra[@]}"} "${@:2}"
