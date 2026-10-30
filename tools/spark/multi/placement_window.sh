#!/usr/bin/env bash
# Single box against placement (docs/dev/MULTI-SPARK.md) on the front host, interleaved A B A B, inside a GPU window
# (tools/spark/window.sh, which also unloads what the live service must give up for the test daemons):
#   placement_window.sh BASE_CONFIG LOCALROUTER TOPOLOGY NODE_URL OUT
# BASE_CONFIG: the deployed config (its tools, voice settings and reserve). TOPOLOGY: the multi topology whose front is
# this host (`self` is set here). NODE_URL: the chat node's daemon (already serving the chat tool).
#   A, single:    a test daemon here with chat, speech and transcription (chat on this GPU).
#   B, placement: a test daemon here as the front with speech and transcription; chat on the node.
# Each run: the contention bench and the voice e2e smoke (placement_bench.py, tools/spark/jarvis/voice_e2e.py), and in
# B the link bench (front against the node itself). Env: CHAT (qwen3.8-27b), TTS (fishaudio/drama-3), STT
# (whisper-large-v3-turbo), ROUNDS (2), E2E (2: smoke runs a mode), PORT (28391), DRAMA_PORT (18794: not the live one).
# Results: OUT/<n>-<mode>/{bench.jsonl,link.jsonl,e2e*/,daemon.log,gpu.log}. Exit non-zero on the first failure.
set -euo pipefail
base=${1:?usage: placement_window.sh BASE_CONFIG LOCALROUTER TOPOLOGY NODE_URL OUT}
lr=${2:?} topo=${3:?} node_url=${4:?} out=${5:?}
here=$(cd "$(dirname "$0")" && pwd)
CHAT=${CHAT:-qwen3.8-27b} TTS=${TTS:-fishaudio/drama-3} STT=${STT:-whisper-large-v3-turbo}
ROUNDS=${ROUNDS:-2} E2E=${E2E:-2} PORT=${PORT:-28391} DRAMA_PORT=${DRAMA_PORT:-18794}
mkdir -p "$out"
log() { echo "placement $(date -u +%T) $*" >&2; }

# the test configs: the deployed tools for chat, speech and transcription; A without a topology, B as the front
jq --arg c "$CHAT" --arg t "$TTS" --arg s "$STT" --argjson dp "$DRAMA_PORT" '
  .tools |= [.[] | select(.id == $c or .id == $t or .id == $s) | if .id == $t then .options.port = $dp else . end]
  | .gpu_lock = "" | del(.topology) | .pairing = {required: false} | .computer = {enabled: false}
  | .voice.shadow_decisions = false | .voice.manage_models = false
  | .defaults = {chat_completion: $c, text_to_speech: $t, speech_to_text: $s}' "$base" > "$out/single.json"
front=$(jq -r '.nodes[] | select(.role == "front") | .id' "$topo")
jq --slurpfile t "$topo" --arg f "$front" '.topology = ($t[0] + {self: $f})' "$out/single.json" > "$out/placement.json"

pid= sampler=
stop() {
  [ -n "$pid" ] && kill -TERM "$pid" 2> /dev/null && wait "$pid" 2> /dev/null
  [ -n "$sampler" ] && kill "$sampler" 2> /dev/null # it holds the window's lock descriptor too
  pid= sampler=
}
trap stop EXIT
ready() { # every tool of the test daemon ready, within 600 s
  local t
  for _ in $(seq 1 300); do
    kill -0 "$pid" 2> /dev/null || { log "the test daemon exited: $(tail -3 "$d/daemon.log")"; return 1; }
    t=$(curl -fs -m 5 "http://127.0.0.1:$PORT/v1/tools" || true)
    [ -n "$t" ] && [ "$(jq '[.data[] | select(.state != "ready")] | length' <<< "$t")" = 0 ] && return 0
    sleep 2
  done
  log "tools not ready: $t"
  return 1
}

n=0
for mode in single placement single placement; do
  n=$((n + 1))
  d=$out/$n-$mode
  mkdir -p "$d"
  ( while sleep 1; do echo "$(date +%s.%N | cut -c1-14) $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"; done ) > "$d/gpu.log" 2>&1 &
  sampler=$!
  log "$n $mode: daemon"
  "$lr" serve --config "$out/$mode.json" --host 127.0.0.1 --port "$PORT" --data "$d/data" > "$d/daemon.out" 2> "$d/daemon.log" &
  pid=$!
  ready
  log "$n $mode: contention"
  python3 -I "$here/placement_bench.py" contention "http://127.0.0.1:$PORT" --chat "$CHAT" --tts "$TTS" --rounds "$ROUNDS" --label "$mode" > "$d/bench.jsonl"
  for k in $(seq 1 "$E2E"); do
    log "$n $mode: voice e2e $k"
    python3 -I "$here/../jarvis/voice_e2e.py" --url "http://127.0.0.1:$PORT" --out "$d/e2e$k" --tts "$TTS" --stt "$STT" --chat "$CHAT" --smoke --skip-barge > "$d/e2e$k.log" 2>&1 || log "$n $mode: e2e $k checks failed (see $d/e2e$k.log)"
  done
  if [ "$mode" = placement ]; then
    log "$n $mode: link"
    python3 -I "$here/placement_bench.py" link "http://127.0.0.1:$PORT" "$node_url" --chat "$CHAT" --label "$mode" > "$d/link.jsonl"
  fi
  stop
  rm -rf "$d/data/jobs" # outputs: audio the checks already read
done
log "done: $out"
