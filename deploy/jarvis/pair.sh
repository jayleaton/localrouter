#!/usr/bin/env bash
# Paired devices of the Jarvis service: the deployed binary's `localrouter pair` on the deployed config and data.
#   deploy/jarvis/pair.sh add "JJ's phone"     a code for the app (10 minutes, once) and, with tailscale serve, its link
#   deploy/jarvis/pair.sh token claude-code    a token now, for MCP clients and scripts (shown once)
#   deploy/jarvis/pair.sh list | revoke ID|NAME
# The daemon reads the devices file on every request: nothing here restarts it. Tokens are needed only once
# pairing.required is set (README.md, "Pairing").
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
bin=$DEPLOY/current/zig-out/bin/localrouter
[ -x "$bin" ] || die "not installed: deploy/jarvis/install.sh first"
[ $# -ge 1 ] || exec "$bin" pair
extra=()
if [ "$1" = add ]; then
  url=$(public_url || true)
  [ -n "$url" ] && extra=(--url "$url")
fi
# the deployed paths first, so options given here (a different --url) win
exec "$bin" pair "$1" --config "$DEPLOY/config.json" --data "$DEPLOY/data" ${extra[@]+"${extra[@]}"} "${@:2}"
