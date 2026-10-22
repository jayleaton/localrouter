# Shared settings of the Jarvis deployment scripts (sourced). Every value can be overridden from the environment.
J=${JARVIS:-$HOME/jarvis}                     # the root everything lives under
DEPLOY=${DEPLOY:-$J/deploy}                   # the install: src/<commit>, current -> it, config.json, data/
PORT=${PORT:-8190}                            # the daemon, on 127.0.0.1 only
HTTPS_PORT=${HTTPS_PORT:-10443}               # tailscale serve's HTTPS port for the tailnet (0: none)
UNIT=${UNIT:-localrouter-jarvis}
# The qualified installs (their qualify scripts made them; their markers must match this tree's pins).
CHAT_ROOT=${CHAT_ROOT:-$J/qualify-chat/root}
CHAT_MARKER=${CHAT_MARKER:-$J/qualify-chat/qwen38-27b-qualification.json}
DRAMA_ROOT=${DRAMA_ROOT:-$J/qualify-drama3/root}
DRAMA_MARKER=${DRAMA_MARKER:-$J/qualify-drama3/drama3-qualification.json}
KITTEN_MARKER=${KITTEN_MARKER:-$J/models/kitten/qualification-graphs.json}
URL=http://127.0.0.1:$PORT
export PATH=$J/bin:$PATH XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
die() { echo "$(basename "$0"): $*" >&2; exit 1; }
mem_gib() { awk '/^MemAvailable:/{printf "%.1f", $2 / 1048576}' /proc/meminfo; }
# The tailnet URL (https://<name>:<port>), or nothing without tailscale serve.
public_url() {
  [ "$HTTPS_PORT" != 0 ] && command -v tailscale > /dev/null || return 0
  local name
  name=$(tailscale status --json 2> /dev/null | jq -r '.Self.DNSName // ""' | sed 's/\.$//')
  [ -n "$name" ] && echo "https://$name:$HTTPS_PORT"
}
# Waits until every keep_loaded tool is ready (or one failed); $1 seconds at most.
wait_warm() {
  local limit=${1:-1200} t0 ids state
  t0=$(date +%s)
  ids=$(jq -r '[.tools[] | select(.keep_loaded == true) | .id] | join(" ")' "$DEPLOY/config.json")
  while true; do
    state=$(curl -fs -m 5 "$URL/v1/tools" | jq -r --arg ids "$ids" \
      '[.data[] | select(.id as $i | $ids | split(" ") | index($i)) | "\(.id)=\(.state)"] | join(" ")' 2> /dev/null || true)
    [ -n "$state" ] && [ "$(grep -o '=ready' <<< "$state" | wc -l)" -eq "$(wc -w <<< "$ids")" ] && {
      echo "warm in $(($(date +%s) - t0)) s: $state"; return 0; }
    grep -q '=failed' <<< "$state" && { echo "a kept tool failed: $state" >&2; return 1; }
    [ $(($(date +%s) - t0)) -lt "$limit" ] || { echo "not warm within $limit s: $state" >&2; return 1; }
    sleep 5
  done
}
