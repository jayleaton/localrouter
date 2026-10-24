#!/usr/bin/env bash
# Qualifies an outside harness against a LocalRouter box, from the machine the harness runs on (docs/harnesses/).
# Real runs in fresh directories, each checked on its outcome, not on what the model says:
#   text      the harness writes answer.txt (the sum of 1..100) with its own file or shell tool and reads it back:
#             answer.txt is 5050
#   continue  (codex) the same session resumed for a second turn: next.txt is answer.txt's number plus one, written
#             from the conversation so far (the earlier calls and their outputs replayed to the box)
#   image     the harness calls the box's MCP generate_image: the tool call is in its events and the URL it returns
#             opens as a PNG from this machine
#
#   tools/harness/qualify.sh opencode|dsh|codex --bin PATH --config FILE --url https://box.tailnet.ts.net:10443 \
#       [--token TOKEN] [--out DIR] [--version X] [--sandbox MODE]
#
# opencode: --bin the opencode binary, --config an opencode.json (docs/harnesses/opencode.json).
# dsh:      --bin the dsh launcher, --config a cordis.patch.yml (docs/harnesses/dsh.cordis.patch.yml).
# codex:    --bin the codex CLI, --config a config.toml (docs/harnesses/codex.config.toml; its placeholder box address
#           becomes --url). --sandbox is Codex's sandbox_mode (default workspace-write; danger-full-access on a host
#           whose kernel refuses the user namespaces Codex's bubblewrap sandbox needs).
# The harness's own state (OpenCode's XDG dirs, DSH_HOME, CODEX_HOME) lives under --out, so the user's setup is not
# touched.
# Needs bash, curl and jq. Exit 0 only when every check passed.
set -uo pipefail
harness=${1:-}; shift || true
bin= config= url= token=${LOCALROUTER_TOKEN:-} out= version= sandbox=workspace-write
while [ $# -gt 0 ]; do
  case $1 in
    --bin) bin=$2; shift 2 ;; --config) config=$2; shift 2 ;; --url) url=${2%/}; shift 2 ;;
    --token) token=$2; shift 2 ;; --out) out=$2; shift 2 ;; --version) version=$2; shift 2 ;; --sandbox) sandbox=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
case $harness in opencode) pin=1.19.4 ;; dsh) pin=0.2.1-rc.2 ;; codex) pin=0.164.0 ;; *) echo "usage: qualify.sh opencode|dsh|codex --bin PATH --config FILE --url URL [--token T] [--out DIR] [--sandbox MODE]" >&2; exit 2 ;; esac
[ -x "$bin" ] && [ -f "$config" ] && [ -n "$url" ] || { echo "--bin, --config and --url are required" >&2; exit 2; }
out=${out:-$(mktemp -d /tmp/lr-qualify-$harness.XXXXXX)}
mkdir -p "$out"
export LOCALROUTER_URL=$url LOCALROUTER_TOKEN=${token:-none}
fails=0
check() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; fails=$((fails + 1)); fi; }

TEXT='Create a file named answer.txt in the current directory whose only content is the sum of the integers from 1 to 100. Then read the file back and tell me exactly what it contains.'
CONTINUE='Now write a file named next.txt whose only content is the number you wrote to answer.txt plus one.'
IMAGE='Use the spark generate_image tool to make a 512x512 picture of a red fox sitting in fresh snow, with wait_s 300 and inline_images false. Then reply with only the image URL the tool returned.'

got=$("$bin" --version 2>/dev/null | tail -1 | sed 's/^codex-cli //' | tr -d 'v \r')
if [ $harness = codex ]; then # Codex reads only $CODEX_HOME/config.toml: the recipe's, pointed at --url
  mkdir -p "$out/.codex" && sed "s|https://BOX.TAILNET.ts.net:10443|$url|g" "$config" > "$out/.codex/config.toml"
fi
[ "$got" = "${version:-$pin}" ]; check $? "$harness version $got (pinned ${version:-$pin})"
caps=$(curl -sS -m 30 "$url/v1/capabilities")
[ "$(jq -r .contract <<<"$caps" 2>/dev/null)" != null ] && [ -n "$caps" ]; check $? "box contract $(jq -r .contract <<<"$caps" 2>/dev/null)"

run() { # name, task[, resume]: the harness's events in $out/name/events.jsonl (resume: continue in $out/text)
  local d=$out/$1
  [ -n "${3:-}" ] && d=$out/text
  [ -n "${3:-}" ] || { rm -rf "$d"; mkdir -p "$d"; }
  local events=$out/$1.events.jsonl
  local t0=$(date +%s)
  if [ $harness = codex ]; then
    local sub=(exec); [ -n "${3:-}" ] && sub=(exec resume --last)
    (cd "$d" && CODEX_HOME=$out/.codex timeout 900 "$bin" "${sub[@]}" --json --skip-git-repo-check -c "sandbox_mode=\"$sandbox\"" "$2" > "$events" 2> "$out/$1.stderr.log" < /dev/null)
  elif [ $harness = opencode ]; then
    (cd "$d" && OPENCODE_CONFIG=$config XDG_CONFIG_HOME=$out/.oc/config XDG_DATA_HOME=$out/.oc/data XDG_CACHE_HOME=$out/.oc/cache \
      XDG_STATE_HOME=$out/.oc/state OPENCODE_DISABLE_AUTOUPDATE=1 \
      timeout 900 "$bin" run --format json --auto -m spark/qwen3.8-27b "$2" > events.jsonl 2> stderr.log)
  else
    mkdir -p "$out/.dsh/profiles/headless" && cp "$config" "$out/.dsh/profiles/headless/cordis.patch.yml"
    (cd "$d" && DSH_HOME=$out/.dsh DSH_TELEMETRY_DISABLED=1 timeout 900 "$bin" --profile headless --json "$2" > events.jsonl 2> stderr.log)
  fi
  local rc=$?
  [ -f "$d/events.jsonl" ] && [ $harness != codex ] && cp "$d/events.jsonl" "$events"
  echo "$1: exit $rc in $(( $(date +%s) - t0 )) s"
  return $rc
}
tools() { # the tool names the harness called (Codex: command_execution for its shell, server.tool for MCP)
  if [ $harness = codex ]; then jq -r 'select(.type=="item.completed") | .item | if .type=="mcp_tool_call" then "\(.server).\(.tool)" else .type end' "$1"
  elif [ $harness = opencode ]; then jq -r 'select(.type=="tool_use") | .part.tool' "$1"; else jq -r 'select(.type=="tool_call") | .tool' "$1"; fi
}
reply() { # the harness's final text
  if [ $harness = codex ]; then jq -r 'select(.type=="item.completed" and .item.type=="agent_message") | .item.text' "$1"
  elif [ $harness = opencode ]; then jq -r 'select(.type=="text") | .part.text' "$1"; else jq -r 'select(.type=="final") | .text' "$1"; fi
}

run text "$TEXT"; check $? "text run exits 0"
[ "$(tr -d ' \r\n' < "$out/text/answer.txt" 2>/dev/null)" = 5050 ]; check $? "answer.txt is 5050"
called=$(tools "$out/text.events.jsonl" | tr '\n' ' ')
own=write; [ $harness = codex ] && own=command_execution
grep -qw $own <<<"$called"; check $? "text: the harness's own $own tool ran ($called)"

if [ $harness = codex ]; then
  run continue "$CONTINUE" resume; check $? "continue: the resumed session exits 0"
  [ "$(tr -d ' \r\n' < "$out/text/next.txt" 2>/dev/null)" = 5051 ]; check $? "continue: next.txt is 5051"
  called=$(tools "$out/continue.events.jsonl" | tr '\n' ' ')
  grep -qw command_execution <<<"$called"; check $? "continue: a tool ran in the second turn ($called)"
fi

run image "$IMAGE"; check $? "image run exits 0"
called=$(tools "$out/image.events.jsonl" | tr '\n' ' ')
grep -q 'generate_image' <<<"$called"; check $? "image: the box's MCP generate_image ran ($called)"
link=$(reply "$out/image.events.jsonl" | grep -oE 'https?://[^ )"`]+\.png' | tail -1)
curl -sSf -m 60 "$link" -o "$out/image/image.png" 2>/dev/null && [ "$(head -c 4 "$out/image/image.png" | od -An -c | tr -d ' ')" = '211PNG' ]
check $? "image: the returned URL opens as a PNG here ($link)"

echo "$([ $fails = 0 ] && echo QUALIFIED || echo "NOT QUALIFIED: $fails check(s) failed") ($harness, events in $out)"
[ $fails = 0 ]
