#!/usr/bin/env bash
# Re-runs the public review "Nemo" did of a LocalRouter box, as a plain HTTP client from any machine: no ssh, no host
# lock, nothing on the box. Each check is timed and judged on what came back, in this order:
#   models      GET /v1/models lists a chat, a speech, a transcription and an image model (video optional)
#   chat        "What is 17 x 19?" with thinking off, temperature 0: the reply holds 323 and finish_reason is stop
#   speech      POST /v1/audio/speech, a known sentence as WAV: HTTP 200, RIFF/WAVE, more than 1 s of audio
#   transcribe  POST /v1/audio/transcriptions of that WAV: at least 70% of the sentence's words come back
#   image       POST /v1/images/generations, one 512x512 still: a PNG (its AI-generated marker reported as info)
#   edit        POST /v1/images/edits of that still: an edited PNG, or a clear 4xx refusal with error.message
#   video       POST /v1/videos, polled to completion (bounded), then its content: an MP4 (marker as info); skipped
#               when no video model is listed
#   chat_again  the chat question again: 323 (chat came back after a job that may have needed the whole machine)
#
#   tools/harness/nemo_check.sh [--url http://<box>:8190] [--out DIR]
#
# --url  the daemon (default $LOCALROUTER_URL, else http://127.0.0.1:8190).
# --out  where results and outputs go (default ./nemo-check-<UTC timestamp>): results.jsonl, summary.json, speech.wav,
#        image.png, edit.png, video.mp4 and every response body.
# Env: LOCALROUTER_TOKEN (sent as a bearer token, for a box with pairing.required: scopes inference,media,read);
# CHAT, SPEECH, STT, IMAGE, EDIT, VIDEO (model ids; by default each is the model /v1/models lists as the default for
# the capability, else the first that has it); VOICE (default: the speech model's own default voice); IMAGE_SIZE
# (512x512); VIDEO_SIZE (default: the model's own size); VIDEO_SECONDS (1); VIDEO_TIMEOUT (900 s).
# Needs bash, curl and jq; ffprobe, when installed, reads the MP4's tags. Exit 0 only when no check failed.
set -uo pipefail
url=${LOCALROUTER_URL:-http://127.0.0.1:8190} out=""
while [ $# -gt 0 ]; do
  case $1 in
    --url) url=${2:-}; shift 2 || shift ;;
    --out) out=${2:-}; shift 2 || shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "nemo_check.sh: unknown argument $1 (usage: nemo_check.sh [--url URL] [--out DIR])" >&2; exit 2 ;;
  esac
done
url=${url%/}
[ -n "$url" ] || { echo "nemo_check.sh: --url is empty" >&2; exit 2; }
for dep in curl jq od; do command -v "$dep" > /dev/null || { echo "nemo_check.sh: needs $dep" >&2; exit 2; }; done
out=${out:-./nemo-check-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$out" || { echo "nemo_check.sh: cannot create $out" >&2; exit 2; }
out=$(cd "$out" && pwd)
: > "$out/results.jsonl"

SENTENCE="The quick brown fox jumps over the lazy dog."
QUESTION="What is 17 × 19? Answer with just the number."
IMAGE_SIZE=${IMAGE_SIZE:-512x512} VIDEO_SIZE=${VIDEO_SIZE:-} VIDEO_SECONDS=${VIDEO_SECONDS:-1}
VIDEO_TIMEOUT=${VIDEO_TIMEOUT:-900} VOICE=${VOICE:-}
MARK='trainedAlgorithmicMedia|AI[- ]generated|machine[- ]generated'

t_all=$(date +%s.%N)
now() { date +%s.%N; }
secs() { awk -v a="$1" -v b="$(now)" 'BEGIN{printf "%.2f", b - a}'; }
# record NAME true|false|skip SECONDS DETAIL
record() {
  local pass=$2 word=FAIL
  case $2 in true) word=PASS ;; skip) word=SKIP; pass=null ;; *) pass=false ;; esac
  jq -nc --arg c "$1" --argjson ok "$pass" --argjson s "$3" --arg d "$4" '{check: $c, pass: $ok, seconds: $s, detail: $d}' >> "$out/results.jsonl"
  printf '%-4s %-12s %8s s  %s\n' "$word" "$1" "$3" "${4:0:140}"
}
yes_no() { if "$@"; then echo true; else echo false; fi; }
# Every request carries $LOCALROUTER_TOKEN when it is set.
curl() { if [ -n "${LOCALROUTER_TOKEN:-}" ]; then command curl -H "authorization: Bearer $LOCALROUTER_TOKEN" "$@"; else command curl "$@"; fi; }
# call TIMEOUT BODYFILE CURL_ARGS...: the HTTP status (000 when there was no answer); the body in BODYFILE
call() {
  local m=$1 f=$2 code; shift 2
  code=$(curl -sS -m "$m" -o "$f" -w '%{http_code}' "$@" 2> "$f.curl-err") || true
  [ -f "$f" ] || : > "$f"
  echo "${code:-000}"
}
# the error message of an OpenAI error envelope, or the start of a body that is not one
errmsg() { jq -r '.error.message // empty' "$1" 2> /dev/null || true; }
brief() { local e; e=$(errmsg "$1"); [ -n "$e" ] && { echo "$e"; return; }; LC_ALL=C tr -c '[:print:]' ' ' < "$1" | cut -c1-160; cat "$1.curl-err" 2> /dev/null | tr '\n' ' ' | cut -c1-120; }
magic() { od -An -tx1 -N"$2" -j"${3:-0}" "$1" 2> /dev/null | tr -d ' \n'; } # FILE COUNT [OFFSET]: hex bytes
u32() { od -An -tu4 -N4 -j"$2" "$1" 2> /dev/null | tr -d ' \n'; } # FILE OFFSET: little-endian 32-bit (LE hosts)
u16() { od -An -tu2 -N2 -j"$2" "$1" 2> /dev/null | tr -d ' \n'; }
words() { tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' ' ' | tr -s ' ' '\n' | grep -v '^$' | sort -u; }
is_png() { [ "$(magic "$1" 8)" = 89504e470d0a1a0a ]; }
png_marker() { # FILE: whether a text chunk names it machine-generated (info)
  if LC_ALL=C grep -qaE 'tEXt|iTXt|zTXt' "$1" 2> /dev/null && LC_ALL=C grep -qaiE "$MARK" "$1" 2> /dev/null; then echo "marker: yes"
  else echo "marker: none found"; fi
}
wav_seconds() { # FILE: duration from the fmt chunk's byte rate and the data chunk (to the end of file if it overruns)
  local f=$1 size off=12 id len rate=0 data=0
  size=$(stat -c %s "$f" 2> /dev/null || echo 0)
  while [ $((off + 8)) -le "$size" ]; do
    id=$(dd if="$f" bs=1 skip="$off" count=4 2> /dev/null)
    len=$(u32 "$f" $((off + 4))); [ -n "$len" ] || break
    case $id in
      'fmt ') rate=$(u32 "$f" $((off + 16))) ;;
      data) data=$len; [ $((off + 8 + len)) -gt "$size" ] && data=$((size - off - 8)); break ;;
    esac
    off=$((off + 8 + len + (len % 2)))
  done
  awk -v d="$data" -v r="${rate:-0}" 'BEGIN{ if (r > 0) printf "%.2f", d / r; else print 0 }'
}

echo "== nemo check against $url, results in $out"

# 1. list models ------------------------------------------------------------------------------------------------------
t=$(now)
code=$(call 30 "$out/models.json" "$url/v1/models")
if [ "$code" = 200 ] && jq -e '.data | type == "array"' "$out/models.json" > /dev/null 2>&1; then
  pick() { # CAPABILITY: the default model for it, else the first that has it
    jq -r --arg c "$1" '[.data[] | select((.default_for // []) | index($c))][0].id // [.data[] | select((.capabilities // []) | index($c))][0].id // empty' "$out/models.json"
  }
  CHAT=${CHAT:-$(pick chat_completion)} SPEECH=${SPEECH:-$(pick text_to_speech)} STT=${STT:-$(pick speech_to_text)}
  IMAGE=${IMAGE:-$(pick text_to_image)} EDIT=${EDIT:-$(pick image_edit)} VIDEO=${VIDEO:-$(pick text_to_video)}
  jq -r '.data[] | "  \(.id)  kind=\(.kind // "?")\(if .license.name then "  license=\(.license.name)" else "" end)\(if .qualified == false then "  (not qualified)" else "" end)"' "$out/models.json"
  missing=""
  for k in chat speech transcription image; do
    jq -e --arg k "$k" '[.data[] | select(.kind == $k)] | length > 0' "$out/models.json" > /dev/null || missing="$missing $k"
  done
  record models "$(yes_no [ -z "$missing" ])" "$(secs "$t")" "$(jq -r '[.data[] | "\(.id) (\(.kind))"] | join(", ")' "$out/models.json")${missing:+; missing kinds:$missing}"
  if jq -e '[.data[] | select(.kind == "video")] | length > 0' "$out/models.json" > /dev/null; then :; else
    record models_video skip 0 "no video model listed"
  fi
else
  record models false "$(secs "$t")" "HTTP $code: $(brief "$out/models.json")"
fi
CHAT=${CHAT:-} SPEECH=${SPEECH:-} STT=${STT:-} IMAGE=${IMAGE:-} EDIT=${EDIT:-} VIDEO=${VIDEO:-}
echo "models: chat=${CHAT:-none} speech=${SPEECH:-none} stt=${STT:-none} image=${IMAGE:-none} edit=${EDIT:-none} video=${VIDEO:-none}"

# 2 and 8. chat with thinking off --------------------------------------------------------------------------------------
chat() { # NAME TIMEOUT
  local t code body c fin
  t=$(now)
  if [ -z "$CHAT" ]; then record "$1" false 0 "no chat model listed (set CHAT)"; return; fi
  body=$(jq -nc --arg m "$CHAT" --arg q "$QUESTION" \
    '{model: $m, messages: [{role: "user", content: $q}], reasoning_effort: "none", temperature: 0, max_tokens: 64}')
  code=$(call "$2" "$out/$1.json" -H 'content-type: application/json' "$url/v1/chat/completions" -d "$body")
  c=$(jq -r '.choices[0].message.content // ""' "$out/$1.json" 2> /dev/null || true)
  fin=$(jq -r '.choices[0].finish_reason // ""' "$out/$1.json" 2> /dev/null || true)
  if [ "$code" = 200 ]; then
    local ok=false
    grep -q 323 <<< "$c" && [ "$fin" = stop ] && ok=true
    record "$1" "$ok" "$(secs "$t")" "finish_reason=${fin:-none} ($CHAT), content: $(tr '\n' ' ' <<< "$c")"
  else
    record "$1" false "$(secs "$t")" "HTTP $code: $(brief "$out/$1.json")"
  fi
}
chat chat 600

# 3. speech -------------------------------------------------------------------------------------------------------------
t=$(now); wav=$out/speech.wav; have_wav=0
if [ -z "$SPEECH" ]; then record speech false 0 "no speech model listed (set SPEECH)"; else
  body=$(jq -nc --arg m "$SPEECH" --arg s "$SENTENCE" --arg v "$VOICE" \
    '{model: $m, input: $s, response_format: "wav"} + (if $v == "" then {} else {voice: $v} end)')
  code=$(call 900 "$wav" -H 'content-type: application/json' "$url/v1/audio/speech" -d "$body")
  bytes=$(stat -c %s "$wav" 2> /dev/null || echo 0)
  if [ "$code" = 200 ] && [ "$(magic "$wav" 4)" = 52494646 ] && [ "$(magic "$wav" 4 8)" = 57415645 ]; then
    dur=$(wav_seconds "$wav"); have_wav=1
    record speech "$(yes_no awk -v d="$dur" 'BEGIN{exit !(d > 1)}')" "$(secs "$t")" "HTTP 200, RIFF/WAVE, $dur s of audio, $bytes B ($SPEECH${VOICE:+, voice $VOICE})"
  else
    record speech false "$(secs "$t")" "HTTP $code, $bytes B, not a RIFF/WAVE file: $(brief "$wav")"
  fi
fi

# 4. transcription of the same file ------------------------------------------------------------------------------------
t=$(now)
if [ -z "$STT" ]; then record transcribe false 0 "no transcription model listed (set STT)"
elif [ $have_wav = 0 ]; then record transcribe false 0 "no WAV from the speech check to transcribe"
else
  code=$(call 900 "$out/transcribe.json" "$url/v1/audio/transcriptions" -F "model=$STT" -F "file=@$wav;type=audio/wav")
  heard=$(jq -r '.text // ""' "$out/transcribe.json" 2> /dev/null || true)
  if [ "$code" = 200 ]; then
    common=$(comm -12 <(words <<< "$SENTENCE") <(words <<< "$heard") | wc -l)
    total=$(words <<< "$SENTENCE" | wc -l)
    record transcribe "$(yes_no [ $((common * 10)) -ge $((total * 7)) ])" "$(secs "$t")" "$common/$total words: $heard"
  else
    record transcribe false "$(secs "$t")" "HTTP $code: $(brief "$out/transcribe.json")"
  fi
fi

# 5. one still ----------------------------------------------------------------------------------------------------------
t=$(now); png=$out/image.png; have_png=0
if [ -z "$IMAGE" ]; then record image false 0 "no image model listed (set IMAGE)"; else
  body=$(jq -nc --arg m "$IMAGE" --arg s "$IMAGE_SIZE" \
    '{model: $m, prompt: "a red fox sitting in fresh snow", n: 1, size: $s, response_format: "b64_json"}')
  code=$(call 900 "$out/image.json" -H 'content-type: application/json' "$url/v1/images/generations" -d "$body")
  : > "$png"
  jq -r '.data[0].b64_json // empty' "$out/image.json" 2> /dev/null | base64 -d > "$png" 2> /dev/null
  if [ "$code" = 200 ] && is_png "$png"; then
    have_png=1
    record image true "$(secs "$t")" "PNG, $(stat -c %s "$png") B, ${IMAGE_SIZE} ($IMAGE), seed $(jq -r '.seed // "?"' "$out/image.json"); $(png_marker "$png")"
  else
    record image false "$(secs "$t")" "HTTP $code, no PNG: $(brief "$out/image.json")"
  fi
fi

# 6. edit of that still: an edited PNG, or a clear refusal ---------------------------------------------------------------
t=$(now); target=${EDIT:-$IMAGE}
if [ $have_png = 0 ]; then record edit false 0 "no PNG from the image check to edit"
elif [ -z "$target" ]; then record edit false 0 "no image model listed (set EDIT or IMAGE)"
else
  code=$(call 900 "$out/edit.json" "$url/v1/images/edits" -F "model=$target" -F 'prompt=make it night, with stars' \
    -F "size=$IMAGE_SIZE" -F 'n=1' -F 'response_format=b64_json' -F "image[]=@$png;type=image/png")
  : > "$out/edit.png"
  jq -r '.data[0].b64_json // empty' "$out/edit.json" 2> /dev/null | base64 -d > "$out/edit.png" 2> /dev/null
  msg=$(errmsg "$out/edit.json")
  if [ "$code" = 200 ] && is_png "$out/edit.png"; then
    record edit true "$(secs "$t")" "edited PNG, $(stat -c %s "$out/edit.png") B ($target); $(png_marker "$out/edit.png")"
  elif [ "${code:0:1}" = 4 ] && [ -n "$msg" ]; then
    record edit true "$(secs "$t")" "pass (refused clearly): HTTP $code $(jq -r '.error.type // ""' "$out/edit.json" 2> /dev/null): $msg"
  elif [ "$code" = 000 ]; then
    record edit false "$(secs "$t")" "no answer within 900 s: $(brief "$out/edit.json")"
  elif [ "$code" = 200 ]; then
    record edit false "$(secs "$t")" "HTTP 200 without an image: $(brief "$out/edit.json")"
  else
    record edit false "$(secs "$t")" "HTTP $code$([ -s "$out/edit.json" ] || echo ', empty body'): $(brief "$out/edit.json")"
  fi
fi

# 7. video last ---------------------------------------------------------------------------------------------------------
t=$(now); mp4=$out/video.mp4
if [ -z "$VIDEO" ]; then record video skip 0 "no video model listed (set VIDEO to name one)"; else
  body=$(jq -nc --arg m "$VIDEO" --arg s "$VIDEO_SIZE" --arg n "$VIDEO_SECONDS" \
    '{model: $m, prompt: "waves rolling onto a beach at dusk", seconds: $n} + (if $s == "" then {} else {size: $s} end)')
  code=$(call 120 "$out/video-create.json" -H 'content-type: application/json' "$url/v1/videos" -d "$body")
  id=$(jq -r '.id // empty' "$out/video-create.json" 2> /dev/null || true)
  st=$(jq -r '.status // empty' "$out/video-create.json" 2> /dev/null || true)
  if [ -z "$id" ]; then
    record video false "$(secs "$t")" "HTTP $code, no job: $(brief "$out/video-create.json")"
  else
    deadline=$(( $(date +%s) + VIDEO_TIMEOUT )); progress=""
    while [ "$st" != completed ] && [ "$st" != failed ] && [ "$st" != cancelled ] && [ "$(date +%s)" -lt "$deadline" ]; do
      sleep 3
      call 30 "$out/video-status.json" "$url/v1/videos/$id" > /dev/null
      s=$(jq -r '.status // empty' "$out/video-status.json" 2> /dev/null || true)
      [ -n "$s" ] && st=$s
      progress=$(jq -r '.progress // empty' "$out/video-status.json" 2> /dev/null || true)
    done
    if [ "$st" = completed ]; then
      code=$(call 300 "$mp4" "$url/v1/videos/$id/content")
      if [ "$code" = 200 ] && [ "$(magic "$mp4" 4 4)" = 66747970 ]; then
        tags=""
        if command -v ffprobe > /dev/null; then
          tags=$(ffprobe -v error -show_entries format_tags -of default=nw=1 "$mp4" 2> /dev/null | tr '\n' ' ' | cut -c1-200)
        fi
        if LC_ALL=C grep -qaiE "$MARK" "$mp4" 2> /dev/null || grep -qiE "$MARK" <<< "$tags"; then mark="marker: yes"; else mark="marker: none found"; fi
        record video true "$(secs "$t")" "job $id: MP4, $(stat -c %s "$mp4") B ($VIDEO); $mark${tags:+; tags: $tags}"
      else
        record video false "$(secs "$t")" "job $id completed, content HTTP $code not an MP4: $(brief "$mp4")"
      fi
    elif [ "$st" = failed ] || [ "$st" = cancelled ]; then
      record video false "$(secs "$t")" "job $id $st: $(jq -r '.error.message // .error // ""' "$out/video-status.json" 2> /dev/null)"
    else
      record video false "$(secs "$t")" "job $id still ${st:-unknown} (progress ${progress:-?}) after $VIDEO_TIMEOUT s"
    fi
  fi
fi

# 8. chat again ---------------------------------------------------------------------------------------------------------
chat chat_again 600

jq -s --arg url "$url" --argjson s "$(secs "$t_all")" '{url: $url, total_seconds: $s,
  passed: ([.[] | select(.pass == true)] | length), failed: ([.[] | select(.pass == false)] | length),
  skipped: ([.[] | select(.pass == null)] | length), failed_checks: [.[] | select(.pass == false) | .check]}' \
  "$out/results.jsonl" > "$out/summary.json"
jq -r '"== \(.passed) passed, \(.failed) failed \(.failed_checks), \(.skipped) skipped in \(.total_seconds) s"' "$out/summary.json"
[ "$(jq '.failed' "$out/summary.json")" = 0 ]
