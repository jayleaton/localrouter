#!/usr/bin/env bash
# Kitten's qualification gate through the daemon (#14's `qualification`, #18's markers) and the ASR check of the graphs
# baseline, on DGX-01. One daemon (the #18 tree's own build, loopback) with:
#   whisper-large-v3-turbo   speech to text, for the ASR check
#   kitten-tts-2             eager, qualification-eager.json      -> serves
#   kitten-tts-2-graphs      graphs, qualification-graphs.json    -> serves
#   kitten-unqualified       graphs, a marker that does not exist -> 503 from the daemon
#   kitten-graphs-4096       graphs, graph_tokens 4096, under the 2048 marker -> the worker refuses to load
#   kitten_gate_check.sh SRC GATE_EAGER GATE_GRAPHS OUT
set -euo pipefail
src=${1:?SRC}; eager=${2:?GATE_EAGER}; graphs=${3:?GATE_GRAPHS}; O=${4:?OUT}
J=$HOME/jarvis; PORT=28190; URL=http://127.0.0.1:$PORT
mkdir -p "$O"
export PATH=$J/bin:$PATH
rev=$(ls "$J/models/whisper/whisper-large-v3-turbo")
ex=$src/tools/speech/kitten/config.example.json
k() { # id lm_decode marker [extra options json]
    jq --arg id "$1" --arg m "$3" --arg py "$J/venvs/kitten/bin/python" --arg w "$src/tools/speech/kitten/kitten_worker.py" \
       --arg weights "$J/models/kitten" --argjson extra "${4:-{\}}" --arg mode "$2" \
       '.tools[] | select(.options.lm_decode == $mode) | .id = $id | .cmd = [$py, "-I", $w] | .weights = $weights
        | .qualification = $m | .options += $extra' "$ex"
}
jq -n --argjson e "$(k kitten-tts-2 eager "$J/models/kitten/qualification-eager.json")" \
      --argjson g "$(k kitten-tts-2-graphs graphs "$J/models/kitten/qualification-graphs.json")" \
      --argjson u "$(k kitten-unqualified graphs "$J/models/kitten/no-such-qualification.json")" \
      --argjson t "$(k kitten-graphs-4096 graphs "$J/models/kitten/qualification-graphs.json" '{"graph_tokens": 4096}')" \
      --arg py "$J/venvs/whisper/bin/python" --arg ww "$src/tools/speech/whisper/worker.py" \
      --arg wm "$J/models/whisper/whisper-large-v3-turbo/$rev" --arg vad "$J/models/vad/silero_vad.jit" \
  '{"reserve_bytes": 8589934592, "tools": [
     {"id": "whisper-large-v3-turbo", "kind": "transcription", "cmd": [$py, $ww], "weights": $wm,
      "resident_bytes": 10737418240, "working_bytes": 2147483648, "idle_ttl_s": 600,
      "options": {"device": "cuda", "dtype": "float16", "attn": "sdpa", "vad": "silero", "vad_model": $vad}},
     $e, $g, $u, $t]}' > "$O/tools.json"
trap 'systemctl --user stop jarvis-kgate 2> /dev/null || true' EXIT
systemd-run --user --unit=jarvis-kgate -q -p MemoryMax=48G -p RuntimeMaxSec=3600 -E PATH \
    "$src/zig-out/bin/localrouter" serve --config "$O/tools.json" --host 127.0.0.1 --port $PORT --data "$O/data"
for _ in $(seq 100); do curl -fs $URL/health > /dev/null && break; sleep 0.2; done
curl -s $URL/v1/models | jq -c '[.data[] | {id, qualified}]' | tee "$O/models.json"
say() { # id -> HTTP status, the reply saved
    curl -s -o "$O/$1.out" -w '%{http_code}' -m 900 $URL/v1/audio/speech -H 'content-type: application/json' \
         -d "{\"model\": \"$1\", \"input\": \"The gate is checked through the daemon.\", \"voice\": \"Bruno\", \"seed\": 7}"
}
for id in kitten-unqualified kitten-graphs-4096 kitten-tts-2 kitten-tts-2-graphs; do
    code=$(say $id)
    msg=$(head -c 300 "$O/$id.out" | tr -d '\0' | grep -a -o '"message":"[^"]*"' || true)
    echo "{\"model\": \"$id\", \"http\": $code, \"error\": \"${msg#\"message\":}\"}" | tee -a "$O/gate.jsonl"
    systemctl --user is-active -q jarvis-kgate || { echo "daemon died" >&2; exit 1; }
done
# the ASR check: every fixture, eager against graphs, through this daemon's Whisper
"$J/venvs/kitten/bin/python" -I "$src/tools/speech/kitten/asr_check.py" $URL whisper-large-v3-turbo "$eager" "$graphs" \
    "$O/asr.json" | tee "$O/asr.log"
cp "$O"/data/logs/*.log "$O/" 2> /dev/null || true
