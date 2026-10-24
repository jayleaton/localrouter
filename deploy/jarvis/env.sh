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
# Drama 3's weights: nvfp4 (plotarmordev/drama-3-NVFP4-Pruned, lossy, approved by JJ 2026-10-24) or full
# (fishaudio/drama-3). Each has its own marker and booking (its measured peak plus 256 MiB); the full model's install
# stays beside the quant's, so DRAMA_VARIANT=full and install.sh is the whole rollback.
DRAMA_VARIANT=${DRAMA_VARIANT:-nvfp4}
case $DRAMA_VARIANT in
  full) DRAMA_MARKER=${DRAMA_MARKER:-$J/qualify-drama3/drama3-qualification.json} DRAMA_RESIDENT=${DRAMA_RESIDENT:-42144366592} ;;
  nvfp4) DRAMA_MARKER=${DRAMA_MARKER:-$J/qualify-drama3/drama3-nvfp4-qualification.json} DRAMA_RESIDENT=${DRAMA_RESIDENT:-22075092992} ;;
  *) echo "env.sh: DRAMA_VARIANT must be nvfp4 or full, not '$DRAMA_VARIANT'" >&2; exit 2 ;;
esac
DRAMA_PORT=${DRAMA_PORT:-$((PORT + 10603))}   # the Drama 3 server, on 127.0.0.1 (18793 beside the default PORT)
KNOWLINE_REV=779944d2cc8ffa19cef386aee83c83bcb967ade2   # PelaAI/KnowLine-4B-Gen5.5, the decision model (tools/decide/knowline)
URL=http://127.0.0.1:$PORT
export PATH=$J/bin:$PATH XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}
# The deployment's own token (scope read: models and residency), for these scripts once pairing.required is on;
# install.sh makes it (delete the file to make a new one). $LOCALROUTER_TOKEN wins when set.
TOKEN_FILE=${TOKEN_FILE:-$DEPLOY/operator.token}
lr_token() { if [ -n "${LOCALROUTER_TOKEN:-}" ]; then echo "$LOCALROUTER_TOKEN"; elif [ -r "$TOKEN_FILE" ]; then cat "$TOKEN_FILE"; fi; }
# curl, with the token when there is one (every route but /health and pairing needs it once pairing.required is on)
lr_curl() {
  local t; t=$(lr_token)
  if [ -n "$t" ]; then curl -H "authorization: Bearer $t" "$@"; else curl "$@"; fi
}
die() { echo "$(basename "$0"): $*" >&2; exit 1; }
mem_gib() { awk '/^MemAvailable:/{printf "%.1f", $2 / 1048576}' /proc/meminfo; }
# The tailnet URL (https://<name>:<port>), or nothing without tailscale serve.
public_url() {
  [ "$HTTPS_PORT" != 0 ] && command -v tailscale > /dev/null || return 0
  local name
  name=$(tailscale status --json 2> /dev/null | jq -r '.Self.DNSName // ""' | sed 's/\.$//')
  [ -n "$name" ] && echo "https://$name:$HTTPS_PORT"
}
# Waits until every keep_loaded tool is ready (or one failed), then until the box answers at warm speed: a ready
# state is not a ready box (DGX-01, 2026-10-24: right after a restart every kept tool was ready while a transcription
# took 1.7 s against 0.4 s warm, and the first voice turns missed their budget). $1 seconds at most, both together.
wait_warm() {
  local limit=${1:-1200} t0 ids state
  t0=$(date +%s)
  ids=$(jq -r '[.tools[] | select(.keep_loaded == true) | .id] | join(" ")' "$DEPLOY/config.json")
  while true; do
    state=$(lr_curl -fs -m 5 "$URL/v1/tools" | jq -r --arg ids "$ids" \
      '[.data[] | select(.id as $i | $ids | split(" ") | index($i)) | "\(.id)=\(.state)"] | join(" ")' 2> /dev/null || true)
    [ -n "$state" ] && [ "$(grep -o '=ready' <<< "$state" | wc -l)" -eq "$(wc -w <<< "$ids")" ] && {
      echo "warm in $(($(date +%s) - t0)) s: $state"; break; }
    grep -q '=failed' <<< "$state" && { echo "a kept tool failed: $state" >&2; return 1; }
    [ $(($(date +%s) - t0)) -lt "$limit" ] || { echo "not warm within $limit s: $state" >&2; return 1; }
    sleep 5
  done
  wait_ready $((limit - ($(date +%s) - t0)))
}

# Primed rounds until the box answers at warm speed: a short chat turn (thinking off), a sentence of speech and the
# transcription of that same audio, READY_ROUNDS (3) times in a row within READY_CHAT_S (1.0), READY_SPEECH_S (3.0)
# and READY_STT_S (0.8) seconds each (warm on DGX-01: 0.3, 1.9 and 0.35 s). $1 seconds at most. The budgets of the
# smoke test are not touched: this only waits until the box can meet them.
# timed OUT ARGS...: curl's seconds for one request whose body goes to OUT, or 99 unless it answered 200
timed() {
  local out=$1 r; shift
  r=$(lr_curl -s -o "$out" -w '%{http_code} %{time_total}' "$@" 2> /dev/null || true)
  if [ "${r%% *}" = 200 ]; then echo "${r#* }"; else echo 99; fi
}
wait_ready() {
  local limit=${1:-600} t0 ok=0 round=0 dir chat=99 speech=99 stt=99 heard
  t0=$(date +%s); dir=$(mktemp -d "${TMPDIR:-/tmp}/lr-ready.XXXXXX")
  local rounds=${READY_ROUNDS:-3} max_chat=${READY_CHAT_S:-1.0} max_speech=${READY_SPEECH_S:-3.0} max_stt=${READY_STT_S:-0.8}
  while [ "$ok" -lt "$rounds" ]; do
    [ $(($(date +%s) - t0)) -lt "$limit" ] || { echo "not answering at warm speed within $limit s (last: chat $chat s, speech $speech s, stt $stt s)" >&2; rm -rf "$dir"; return 1; }
    round=$((round + 1))
    chat=$(timed /dev/null -m 30 "$URL/v1/chat/completions" -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say ready."}],"max_tokens":8,"reasoning_effort":"none"}')
    rm -f "$dir/ready.wav"
    speech=$(timed "$dir/ready.wav" -m 60 "$URL/v1/audio/speech" -H 'content-type: application/json' \
      -d '{"input":"Ready when you are.","response_format":"wav"}')
    stt=99 heard=""
    if [ "$speech" != 99 ] && [ -s "$dir/ready.wav" ]; then
      stt=$(timed "$dir/heard.json" -m 60 "$URL/v1/audio/transcriptions" -F file=@"$dir/ready.wav" -F language=en)
      heard=$(jq -r '.text // ""' "$dir/heard.json" 2> /dev/null | tr 'A-Z' 'a-z' || true)
    fi
    if awk -v c="$chat" -v s="$speech" -v t="$stt" -v mc="$max_chat" -v ms="$max_speech" -v mt="$max_stt" \
         'BEGIN { exit !(c <= mc && s <= ms && t <= mt) }' && grep -q ready <<< "$heard"; then ok=$((ok + 1)); else ok=0; fi
    [ "$ok" -ge "$rounds" ] || sleep 2
  done
  echo "answering at warm speed after $(($(date +%s) - t0)) s ($round rounds): chat $chat s, speech $speech s, stt $stt s"
  rm -rf "$dir"
}
