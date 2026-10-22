#!/usr/bin/env bash
# Smoke test of a live Jarvis deployment (deploy/jarvis): every capability once, pass/fail and seconds each, memory
# before and after. Runs on the host, against the daemon's loopback URL, under the host's exclusive lock.
#   tools/jarvis/smoke.sh [--quick] [--url URL] [--out DIR]
# The warm set is chat, Drama 3, Whisper and the image model; Kitten loads on demand. Voice: warm turns reach first audio
# in 1.2 s (median); an image job on the warm image model unloads nothing; a video unloads only the image model (and
# Kitten), the first voice turns after an image or a video reach first audio in 1.5 s, and the image model comes back.
# --quick (about 1 min) skips the video, the barge-in, the picture turn, the after-image turns and Kitten; the full run
# (under 10 min) has them, then waits for the warm set to be whole again. Exit 0 only when every check passed. Writes
# OUT/results.jsonl, OUT/summary.json, OUT/watch-{image,video}.txt and each check's files.
# Env: CHAT, DRAMA, KITTEN, STT, IMAGE, VIDEO (model ids; the deploy's defaults), STT_FIXTURE (the pinned LibriSpeech
# WAV, tools/speech/fixtures/manifest.json; found under ~/jarvis by its sha256 when unset), LOCK (~/jarvis/.dgx01.lock).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
quick=0 url=http://127.0.0.1:8190 out=""
while [ $# -gt 0 ]; do
  case $1 in
    --quick) quick=1; shift ;;
    --url) url=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    *) echo "smoke.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
CHAT=${CHAT:-qwen3.8-27b} DRAMA=${DRAMA:-fishaudio/drama-3} KITTEN=${KITTEN:-kitten-tts-2-graphs}
STT=${STT:-whisper-large-v3-turbo} IMAGE=${IMAGE:-qwen-image-2.1-turbo} VIDEO=${VIDEO:-minimax-h3}
FIXTURE_SHA=799f78ed4beb4de7ceae3a809262d4ce242394342ccd1d58cef7d49dbc2def46
FIXTURE_TEXT="mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
out=${out:-$HOME/jarvis/deploy/smoke/$(date -u +%Y%m%dT%H%M%SZ)$([ $quick = 1 ] && echo -quick)}
mkdir -p "$out"
exec 9> "${LOCK:-$HOME/jarvis/.dgx01.lock}"
flock -w 900 9 || { echo "smoke.sh: the host lock is held by someone else" >&2; exit 1; }
exec > >(tee -a "$out/smoke.log") 2>&1
: > "$out/results.jsonl"
t_all=$(date +%s.%N)
now() { date +%s.%N; }
secs() { awk -v a="$1" -v b="$(now)" 'BEGIN{printf "%.2f", b - a}'; }
# record NAME OK SECONDS DETAIL
record() {
  jq -nc --arg c "$1" --argjson ok "$2" --argjson s "$3" --arg d "$4" '{check: $c, pass: $ok, seconds: $s, detail: $d}' >> "$out/results.jsonl"
  printf '%-4s %-26s %7s s  %s\n' "$([ "$2" = true ] && echo PASS || echo FAIL)" "$1" "$3" "${4:0:110}"
}
post() { curl -s -m "${3:-300}" -H 'content-type: application/json' "$url$1" -d "$2"; }
memory() { # NAME: MemAvailable and every model's state, into OUT/memory-NAME.json
  curl -fs -m 10 "$url/v1/tools" | jq '{mem_available_gib: (.mem_available_bytes / 1073741824 * 10 | round / 10),
    tools: [.data[] | {id, state, resident_gib: (.resident_bytes / 1073741824 * 10 | round / 10)}]}' > "$out/memory-$1.json"
  echo "memory $1: $(jq -c '{mem_available_gib, warm: [.tools[] | select(.state == "ready") | .id]}' "$out/memory-$1.json")"
}
words() { tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' ' ' | tr -s ' ' '\n' | grep -v '^$' | sort -u; }

echo "== smoke $([ $quick = 1 ] && echo quick || echo full) against $url, results in $out"
memory before

# /v1/models: every capability's model is listed and none is refused as unqualified
t=$(now); m=$(curl -fs -m 10 "$url/v1/models")
missing=$(jq -r --arg ids "$CHAT $DRAMA $KITTEN $STT $IMAGE $VIDEO" '[($ids | split(" "))[] as $i | select([.data[].id] | index($i) | not) | $i] | join(" ")' <<< "$m")
unq=$(jq -r '[.data[] | select(.qualified == false) | .id] | join(" ")' <<< "$m")
record models "$([ -n "$m" ] && [ -z "$missing$unq" ] && echo true || echo false)" "$(secs $t)" "$(jq -r '[.data[].id] | join(", ")' <<< "$m") ${missing:+missing: $missing} ${unq:+unqualified: $unq}"

# chat, whole
t=$(now)
r=$(post /v1/chat/completions "{\"model\":\"$CHAT\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One word.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
c=$(jq -r '.choices[0].message.content // ""' <<< "$r" 2> /dev/null); echo "$r" > "$out/chat.json"
record chat_whole "$(grep -qi paris <<< "$c" && echo true || echo false)" "$(secs $t)" "$c"

# chat, streamed: chunks with content, then [DONE]; the first content chunk's time
t=$(now)
curl -sN -m 120 -H 'content-type: application/json' "$url/v1/chat/completions" \
  -d "{\"model\":\"$CHAT\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to five in words.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" |
  while IFS= read -r line; do printf '%s %s\n' "$(now)" "$line"; done > "$out/chat-stream.txt"
first=$(grep -m1 -E '"content": ?"[^"]' "$out/chat-stream.txt" | cut -d' ' -f1)
text=$(sed -n 's/^[0-9.]* data: //p' "$out/chat-stream.txt" | grep -v '^\[DONE\]' | jq -rj '.choices[0].delta.content // ""' 2> /dev/null)
done_=$(grep -c 'data: \[DONE\]' "$out/chat-stream.txt")
record chat_stream "$([ -n "$first" ] && [ "$done_" = 1 ] && grep -qi five <<< "$text" && echo true || echo false)" "$(secs $t)" \
  "first content $( [ -n "$first" ] && awk -v a="$t" -v b="$first" 'BEGIN{printf "%.2f", b - a}') s: $text"

# a tool call
t=$(now)
r=$(post /v1/chat/completions "{\"model\":\"$CHAT\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Tokyo right now? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":128,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
echo "$r" > "$out/tool.json"
call=$(jq -c '.choices[0].message.tool_calls[0].function // empty' <<< "$r" 2> /dev/null)
record tool_call "$(jq -e '.name == "get_weather" and (.arguments | test("tokyo"; "i"))' <<< "${call:-null}" > /dev/null 2>&1 && echo true || echo false)" "$(secs $t)" "${call:-$(head -c 200 <<< "$r")}"

# speech to text on the pinned fixture
fixture=${STT_FIXTURE:-}
if [ -z "$fixture" ]; then
  for f in $(find "$HOME/jarvis" -name en_librispeech_1272-128104-0000.wav 2> /dev/null); do
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = $FIXTURE_SHA ] && { fixture=$f; break; }
  done
fi
t=$(now)
if [ -n "$fixture" ]; then
  r=$(curl -s -m 300 "$url/v1/audio/transcriptions" -F "model=$STT" -F "file=@$fixture")
  heard=$(jq -r '.text // ""' <<< "$r")
  common=$(comm -12 <(words <<< "$FIXTURE_TEXT") <(words <<< "$heard") | wc -l)
  total=$(words <<< "$FIXTURE_TEXT" | wc -l)
  record stt "$([ $((common * 10)) -ge $((total * 9)) ] && echo true || echo false)" "$(secs $t)" "$common/$total words: $heard"
else
  record stt false 0 "no fixture with sha256 $FIXTURE_SHA (set STT_FIXTURE)"
fi

# text to speech: Drama 3 streamed (time to first audio) and whole (Kitten, on demand, at the end of the full run)
tts() { # NAME MODEL
  local t r bytes dur
  t=$(now)
  r=$(curl -s -m 300 -o "$out/$1.wav" -w '%{http_code}' -H 'content-type: application/json' "$url/v1/audio/speech" \
      -d "{\"model\":\"$2\",\"input\":\"Good evening. The smoke test is speaking now.\",\"response_format\":\"wav\"}")
  bytes=$(stat -c %s "$out/$1.wav" 2> /dev/null || echo 0)
  dur=$(python3 -I -c 'import sys, wave; w = wave.open(sys.argv[1]); print(round(w.getnframes() / w.getframerate(), 2))' "$out/$1.wav" 2> /dev/null || echo 0)
  record "$1" "$([ "$r" = 200 ] && [ "$(head -c 4 "$out/$1.wav")" = RIFF ] && awk -v d="$dur" 'BEGIN{exit !(d > 1)}' && echo true || echo false)" "$(secs $t)" "HTTP $r, $dur s of audio, $bytes B"
}
t=$(now)
fa=$(curl -s -m 120 -o "$out/tts-drama-stream.pcm" -w '%{http_code} %{time_starttransfer}' -H 'content-type: application/json' "$url/v1/audio/speech" \
     -d "{\"model\":\"$DRAMA\",\"input\":\"Good evening. The smoke test is speaking now.\",\"response_format\":\"pcm\",\"stream\":true}")
record tts_drama_stream "$([ "${fa%% *}" = 200 ] && [ "$(stat -c %s "$out/tts-drama-stream.pcm")" -gt 44100 ] && echo true || echo false)" "$(secs $t)" \
  "HTTP ${fa%% *}, first audio byte after ${fa##* } s, $(stat -c %s "$out/tts-drama-stream.pcm") B"
tts tts_drama "$DRAMA"

# voice NAME OUT_SUBDIR DRIVER_ARGS...: the e2e driver over /v1/realtime; its checks recorded as NAME: ...
voice() {
  local name=$1 dir=$2 t rc c med
  shift 2
  t=$(now)
  python3 -I "$here/../spark/jarvis/voice_e2e.py" --smoke "$@" --url "$url" --out "$out/$dir" \
    --tts "$DRAMA" --stt "$STT" --chat "$CHAT" > "$out/$dir.log" 2>&1
  rc=$?
  for c in $(jq -r '.checks[] | @base64' "$out/$dir/results.json" 2> /dev/null); do
    c=$(base64 -d <<< "$c")
    record "$name: $(jq -r .check <<< "$c" | cut -c1-36)" "$(jq .ok <<< "$c")" 0 "$(jq -c .detail <<< "$c")"
  done
  med=$(jq -r '.first_audio_ms | "first audio median \(.median) ms, turns \(.turns | tojson)"' "$out/$dir/results.json" 2> /dev/null)
  record "$name" "$([ $rc = 0 ] && echo true || echo false)" "$(secs $t)" "${med:-driver exit $rc, see $dir.log}"
}
# first_turn NAME DIR: the first turn after another model's job is what a user waits for: it alone must reach 1.5 s
first_turn() {
  local ms
  ms=$(jq -r '.first_audio_ms.turns.warmup // "null"' "$out/$2/results.json" 2> /dev/null)
  record "$1" "$([ "$ms" != null ] && [ -n "$ms" ] && [ "$ms" -le 1500 ] && echo true || echo false)" 0 "first turn's first audio ${ms} ms"
}
# watch_start NAME / watch_stop NAME: MemAvailable and every model's state each 0.5 s while a job runs; watch_stop
# prints the lowest MemAvailable (GiB) and the models that were ready at the start and unloaded during it
watch_start() {
  (while true; do
     printf '%s %s\n' "$(awk '/^MemAvailable:/{printf "%.1f", $2 / 1048576}' /proc/meminfo)" \
       "$(curl -fs -m 2 "$url/v1/tools" | jq -c '[.data[] | select(.state == "ready" or .state == "busy") | .id]')"
     sleep 0.5
   done) > "$out/watch-$1.txt" &
  watcher=$!
}
watch_stop() {
  kill "$watcher" 2> /dev/null; wait "$watcher" 2> /dev/null
  local first gone low
  low=$(awk '{print $1}' "$out/watch-$1.txt" | sort -n | head -1)
  first=$(head -1 "$out/watch-$1.txt" | cut -d' ' -f2-)
  gone=$(cut -d' ' -f2- "$out/watch-$1.txt" | jq -sr --argjson f "${first:-[]}" '[.[] | ($f - .)] | add // [] | unique | join(" ")')
  echo "$low|$gone"
}
# NAME may lose only these models: assert the unloaded set against them
only_lost() { # NAME LOST ALLOWED...
  local name=$1 lost=$2 bad=""; shift 2
  for m in $lost; do [[ " $* " == *" $m "* ]] || bad="$bad $m"; done
  record "$name" "$([ -z "$bad" ] && echo true || echo false)" 0 "unloaded: ${lost:-none}${bad:+; not allowed:$bad}"
}

# the warm set: chat, Drama 3, Whisper and the image model, kept warm (Kitten loads on demand)
kept=$(curl -fs -m 5 "$url/v1/tools" | jq -r '[.data[] | select(.keep_loaded) | "\(.id)=\(.state)"] | join(" ")')
want="$CHAT $DRAMA $STT $IMAGE"
miss=""; for m in $want; do grep -q "$m=ready" <<< "$kept" || miss="$miss $m"; done
record warm_set "$([ -z "$miss" ] && ! grep -q "$KITTEN" <<< "$kept" && echo true || echo false)" 0 "$kept${miss:+; not warm:$miss}"

# warm voice turns (median first audio <= 1.2 s); in the full run, barge-in and a spoken picture request
extra=(--skip-barge); [ $quick = 1 ] || extra=(--picture --picture-prompt "Generate a picture of a dog.")
voice voice_warm voice --max-median-ms 1200 "${extra[@]}"
if [ $quick = 0 ]; then
  record barge_in "$(jq -e '.barge_in.cut_status == "cancelled" and .barge_in.audio_deltas_after_speech_started == 0' "$out/voice/results.json" > /dev/null 2>&1 && echo true || echo false)" 0 \
    "$(jq -c '.barge_in // "missing"' "$out/voice/results.json" 2> /dev/null)"
  record picture_turn "$(jq -e '.tool.job.status == "completed" and .tool.speech_end_to_first_audio_ms != null' "$out/voice/results.json" > /dev/null 2>&1 && echo true || echo false)" 0 \
    "$(jq -c '.tool | {first_audio_ms: .speech_end_to_first_audio_ms, job_done_s: .speech_end_to_job_done_s, confirmation_audio_s: .speech_end_to_confirmation_audio_s}' "$out/voice/results.json" 2> /dev/null)"
fi

# image (1024x1024, as the client asks), on the warm image model: generate, fetch; nothing may be unloaded
watch_start image; t=$(now)
r=$(post /v1/images/generations "{\"model\":\"$IMAGE\",\"prompt\":\"a red fox sitting in fresh snow\",\"size\":\"1024x1024\",\"seed\":1,\"response_format\":\"url\"}" 600)
iu=$(jq -r '.data[0].url // ""' <<< "$r")
code=$( [ -n "$iu" ] && curl -s -m 60 -o "$out/image.png" -w '%{http_code}' "$iu")
record image "$([ "$code" = 200 ] && [ "$(head -c 8 "$out/image.png" | od -An -tx1 | tr -d ' ')" = 89504e470d0a1a0a ] && echo true || echo false)" "$(secs $t)" \
  "${iu:-$(head -c 200 <<< "$r")}"
w=$(watch_stop image); echo "image: lowest MemAvailable ${w%%|*} GiB"
only_lost image_evicted_nothing "${w#*|}"
if [ $quick = 0 ]; then
  voice voice_after_image voice-after-image --skip-barge --count-warmup --max-median-ms 1500
  first_turn first_turn_after_image voice-after-image
fi

# video (full run): a 1 s clip, then its MP4; it may evict only the image model (and Kitten); the voice turns right
# after it within 1.5 s; then the image model is back
if [ $quick = 0 ]; then
  watch_start video; t=$(now)
  id=$(post /v1/videos "{\"model\":\"$VIDEO\",\"prompt\":\"waves at dusk\",\"size\":\"768x448\",\"seconds\":\"1\",\"steps\":8,\"seed\":1}" | jq -r '.id // ""')
  st=none
  for _ in $(seq 300); do
    [ -n "$id" ] || break
    st=$(curl -fs -m 10 "$url/v1/videos/$id" | jq -r .status); [ "$st" = completed ] || [ "$st" = failed ] && break; sleep 1
  done
  [ "$st" = completed ] && curl -s -m 60 -o "$out/video.mp4" "$url/v1/videos/$id/content"
  record video "$([ "$st" = completed ] && [ "$(head -c 8 "$out/video.mp4" 2> /dev/null | tail -c 4)" = ftyp ] && echo true || echo false)" "$(secs $t)" "job ${id:-none}: $st, $(stat -c %s "$out/video.mp4" 2> /dev/null || echo 0) B"
  w=$(watch_stop video); echo "video: lowest MemAvailable ${w%%|*} GiB"
  only_lost video_evicted_only_image "${w#*|}" "$IMAGE" "$KITTEN"
  voice voice_after_video voice-after-video --skip-barge --count-warmup --max-median-ms 1500
  first_turn first_turn_after_video voice-after-video
  # Kitten on demand (the fallback voice), then the warm set must be whole again (the image model back, Kitten gone first)
  tts tts_kitten "$KITTEN"
  t=$(now); state=""
  for _ in $(seq 120); do
    state=$(curl -fs -m 5 "$url/v1/tools" | jq -r '[.data[] | select(.keep_loaded) | "\(.id)=\(.state)"] | join(" ")')
    [ -n "$state" ] && ! grep -qv '=ready' <<< "$(tr ' ' '\n' <<< "$state")" && break; sleep 2
  done
  record warm_again "$(! grep -qv '=ready' <<< "$(tr ' ' '\n' <<< "$state")" && echo true || echo false)" "$(secs $t)" "$state"
fi
memory after

jq -s --arg mode "$([ $quick = 1 ] && echo quick || echo full)" --argjson s "$(secs "$t_all")" --slurpfile b "$out/memory-before.json" --slurpfile a "$out/memory-after.json" \
  '{mode: $mode, seconds: $s, passed: ([.[] | select(.pass)] | length), failed: [.[] | select(.pass | not) | .check],
    memory_before: $b[0], memory_after: $a[0], checks: .}' "$out/results.jsonl" > "$out/summary.json"
jq -r '"== \(.mode): \(.passed) passed, \(.failed | length) failed \(.failed) in \(.seconds) s; MemAvailable \(.memory_before.mem_available_gib) -> \(.memory_after.mem_available_gib) GiB"' "$out/summary.json"
[ "$(jq '.failed | length' "$out/summary.json")" = 0 ]
