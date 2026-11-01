# Shared settings of the macOS deployment scripts (sourced). Every value can be overridden from the environment;
# bootstrap.sh sets them from its options (--prefix, --port, --https-port, --unit) and the profile's env block.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/install/lib.sh"   # die, sha256, launchd_label
J=${JARVIS:-$HOME/jarvis}                     # the root everything lives under (bootstrap's --prefix)
DEPLOY=${DEPLOY:-$J/deploy}                   # the install: src/<commit>, current -> it, config.json, data/, logs/
PORT=${PORT:-8190}                            # the daemon, on 127.0.0.1 only
HTTPS_PORT=${HTTPS_PORT:-10443}               # tailscale serve's HTTPS port for the tailnet (0: none)
UNIT=${UNIT:-localrouter-jarvis}
LABEL=${LABEL:-$(launchd_label "$UNIT")}      # the launchd user agent (default com.localrouter.jarvis)
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
DOMAIN=gui/$(id -u)
LOG=$DEPLOY/logs/daemon.log                   # the daemon's stdout and stderr; each model's own log: $DEPLOY/data/logs/
# The models' installs, where the profile's steps put them (deploy/profiles/mac-apple-silicon.json, env).
CHAT_ROOT=${CHAT_ROOT:-$J/chat}
DRAMA_ROOT=${DRAMA_ROOT:-$J/qualify-drama3/root}
DRAMA_MARKER=${DRAMA_MARKER:-$J/qualify-drama3/drama3-macos-qualification.json}
DRAMA_PORT=${DRAMA_PORT:-$((PORT + 10603))}   # the Drama 3 server, on 127.0.0.1
IMAGE_ROOT=${IMAGE_ROOT:-$J/image-mac}
HF_HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}   # chat's checkpoint and drafter (chat/fetch)
URL=http://127.0.0.1:$PORT
# The deployment's own token (scope read), for these scripts once pairing.required is on; install.sh makes it.
# $LOCALROUTER_TOKEN wins when set.
TOKEN_FILE=${TOKEN_FILE:-$DEPLOY/operator.token}
lr_token() { if [ -n "${LOCALROUTER_TOKEN:-}" ]; then echo "$LOCALROUTER_TOKEN"; elif [ -r "$TOKEN_FILE" ]; then cat "$TOKEN_FILE"; fi; }
lr_curl() {
  local t; t=$(lr_token)
  if [ -n "$t" ]; then curl -H "authorization: Bearer $t" "$@"; else curl "$@"; fi
}
# tailscale's CLI: on PATH, else inside the app
tailscale_cli() {
  if command -v tailscale > /dev/null; then command -v tailscale
  elif [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]; then echo /Applications/Tailscale.app/Contents/MacOS/Tailscale; fi
}
# The tailnet URL (https://<name>:<port>), or nothing without tailscale serve.
public_url() {
  local ts name
  ts=$(tailscale_cli)
  [ "$HTTPS_PORT" != 0 ] && [ -n "$ts" ] || return 0
  name=$("$ts" status --json 2> /dev/null | jq -r '.Self.DNSName // ""' | sed 's/\.$//')
  [ -n "$name" ] && echo "https://$name:$HTTPS_PORT"
}
# The agent's launchd state (running, waiting, ...), or "not loaded".
agent_state() { launchctl print "$DOMAIN/$LABEL" 2> /dev/null | awk '$1 == "state" {print $3; exit}' | grep . || echo "not loaded"; }
agent_pid() { launchctl print "$DOMAIN/$LABEL" 2> /dev/null | awk '$1 == "pid" {print $3; exit}'; }
# Loads the agent. `launchctl bootout` returns before launchd has removed the service, and a bootstrap meanwhile fails
# (error 5, "Input/output error"): so after a bootout, bootstrap is retried for up to 30 s.
agent_load() {
  local i
  for i in $(seq 30); do
    if launchctl bootstrap "$DOMAIN" "$PLIST" 2> /dev/null; then
      [ "$i" = 1 ] || echo "  launchd took $((i - 1)) s to let $LABEL load again (bootout had not finished)"
      return 0
    fi
    [ "$(agent_state)" != running ] || return 0   # loaded meanwhile (RunAtLoad at login): nothing to do
    sleep 1   # still there from the bootout (its state "SIGTERMed" until the daemon has exited)
  done
  launchctl bootstrap "$DOMAIN" "$PLIST"   # once more, with its error shown
}
# Unloads the agent and waits until launchd has removed it. The daemon stops its models first (a SIGTERM, then
# SIGKILL after the plist's ExitTimeOut).
agent_unload() {
  [ "$(agent_state)" = "not loaded" ] && return 0
  launchctl bootout "$DOMAIN/$LABEL" 2> /dev/null || true
  local i
  for i in $(seq 200); do [ "$(agent_state)" = "not loaded" ] && return 0; sleep 1; done
  echo "$LABEL is still loaded after 200 s" >&2; return 1
}
# Waits until every tool whose residency is "always" now (the config's, or what Settings changed it to) is ready, or
# one failed; $1 seconds at most.
wait_warm() {
  local limit=${1:-1200} t0 ids state
  t0=$(date +%s)
  ids=$(lr_curl -fs -m 5 "$URL/v1/tools" | jq -r '[.data[] | select(.keep_loaded == true) | .id] | join(" ")' 2> /dev/null || true)
  [ -n "$ids" ] || { echo "no model is kept warm"; return 0; }
  while true; do
    state=$(lr_curl -fs -m 5 "$URL/v1/tools" | jq -r --arg ids "$ids" \
      '[.data[] | select(.id as $i | $ids | split(" ") | index($i)) | "\(.id)=\(.state)"] | join(" ")' 2> /dev/null || true)
    [ -n "$state" ] && [ "$(grep -o '=ready' <<< "$state" | wc -l)" -eq "$(wc -w <<< "$ids")" ] && {
      echo "warm in $(($(date +%s) - t0)) s: $state"; return 0; }
    grep -q '=failed' <<< "$state" && { echo "a kept model failed: $state (its log: $DEPLOY/data/logs/)" >&2; return 1; }
    [ $(($(date +%s) - t0)) -lt "$limit" ] || { echo "not warm within $limit s: $state" >&2; return 1; }
    sleep 5
  done
}
# Waits until no voice session is open, so a restart never cuts a live call: IDLE_WAIT_S at most (default 1800), then
# fails with nothing changed. FORCE_RESTART=1 skips the wait.
wait_idle() {
  local t0 n
  t0=$(date +%s)
  while [ "${FORCE_RESTART:-0}" != 1 ]; do
    n=$(lr_curl -fs -m 5 "$URL/v1/tools" | jq -r '.voice_sessions // 0' 2> /dev/null || echo 0)
    [ "$n" = 0 ] && return 0
    [ $(($(date +%s) - t0)) -lt "${IDLE_WAIT_S:-1800}" ] ||
      die "$n voice session(s) still open after ${IDLE_WAIT_S:-1800} s: nothing changed (FORCE_RESTART=1 to restart anyway)"
    echo "  waiting: $n voice session(s) open"
    sleep 10
  done
}
# swap_in BUILD CONFIG PLIST: makes src/<commit> (BUILD, relative to $DEPLOY), the config file CONFIG and the agent
# file PLIST the deployed ones, keeping what they replace as current.prev, config.json.prev and $LABEL.plist.prev for
# rollback.sh. An empty argument leaves that one as it is (and still saves it, so a rollback restores a whole state).
swap_in() {
  rm -f "$DEPLOY/current.prev" "$DEPLOY/config.json.prev" "$DEPLOY/$LABEL.plist.prev"
  if [ -L "$DEPLOY/current" ]; then readlink "$DEPLOY/current" > "$DEPLOY/current.prev"; fi
  if [ -f "$DEPLOY/config.json" ]; then cp -p "$DEPLOY/config.json" "$DEPLOY/config.json.prev"; fi
  if [ -f "$PLIST" ]; then cp -p "$PLIST" "$DEPLOY/$LABEL.plist.prev"; fi
  if [ -n "$1" ]; then ln -sfh "$1" "$DEPLOY/current"; fi   # -h: replace the link itself (BSD ln follows a link to a directory)
  if [ -n "$2" ]; then mv "$2" "$DEPLOY/config.json"; fi
  if [ -n "$3" ]; then mkdir -p "$(dirname "$PLIST")" && mv "$3" "$PLIST"; fi
}
# Loads the agent and waits until the daemon answers on $URL (30 s at most); $1 names what it runs, for the messages.
agent_up() {
  agent_load || die "launchctl bootstrap $DOMAIN $PLIST failed"
  for _ in $(seq 150); do curl -fs -m 2 "$URL/health" > /dev/null && break; sleep 0.2; done
  curl -fs -m 2 "$URL/health" > /dev/null || die "restarted $LABEL onto $1, but it does not answer on $URL: tail $LOG"
  echo "== restarted $LABEL onto $1; it answers on $URL"
}
