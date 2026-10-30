#!/usr/bin/env bash
# Writes the native daemon's tools.json for the Jarvis measurements: Whisper (STT) and Kitten (TTS) as cmd workers,
# plus the image and video engines over the packs tools/spark/install.sh made. Speech tools stay warm (keep_loaded)
# and outrank the image and video models, so a generation job evicts those last.
#   [KEEP=true|false] daemon_config.sh OUT.json [speech|all]   (KEEP=false: speech loads on first use, to time it)
set -euo pipefail
out=${1:?OUT.json}; set_=${2:-all}
J=$HOME/jarvis; R=$J/src/localrouter; W=$J/lr/weights
whisper_rev=$(ls "$J/models/whisper/whisper-large-v3-turbo")
kitten=$J/models/kitten
voices='"Bella","Jasper","Luna","Bruno","Rosie","Hugo","Kiki","Leo","Matthew","Elliot","Willow","Dolores","Victor","Dante","Alfred","Saoirse","Claire","Raven","Marcus","Herbert","Diana","Laurence","Maeve","Walter","Edith","Miles","Grace","Reginald","Iris","Frank","Serena","Julian","Eleanor","Otis","Vincent","Martha","Sable","Victoria","Arabic","German","Spanish","French","Italian","Portuguese","Russian","Chinese","Hindi"'
speech=$(cat << JSON
  {"id": "whisper-large-v3-turbo", "kind": "transcription", "name": "Whisper large-v3-turbo",
   "cmd": ["$J/venvs/whisper/bin/python", "$R/tools/speech/whisper/worker.py"],
   "weights": "$J/models/whisper/whisper-large-v3-turbo/$whisper_rev",
   "resident_bytes": 10737418240, "working_bytes": 2147483648, "idle_ttl_s": 3600, "priority": 20, "keep_loaded": ${KEEP:-true},
   "options": {"device": "cuda", "dtype": "float16", "attn": "sdpa", "vad": "silero", "vad_model": "$J/models/vad/silero_vad.jit"}},
  {"id": "kitten-tts-2", "kind": "speech", "name": "Kitten TTS 2",
   "cmd": ["$J/venvs/kitten/bin/python", "-I", "$R/tools/speech/kitten/kitten_worker.py"],
   "weights": "$kitten", "resident_bytes": 12884901888, "working_bytes": 2147483648, "idle_ttl_s": 3600,
   "priority": 20, "keep_loaded": ${KEEP:-true}, "load_timeout_s": 600, "generate_timeout_s": 900,
   "options": {"device": "cuda", "max_input_chars": 4096, "min_speed": 1, "max_speed": 1, "default_voice": "Bruno",
               "presets": ["stable", "expressive"], "voices": [$voices]}}
JSON
)
media=$(cat << JSON
  ,{"id": "qwen-image-2.1-turbo", "kind": "image", "name": "Qwen-Image 2.1 Turbo (FP8, 8 steps)", "engine": "qwen_image_turbo",
   "weights": "$W/qwen-image-2.1", "idle_ttl_s": 600, "priority": 10, "options": {"precision": "fp8s", "max_side": 1664}},
  {"id": "minimax-h3", "kind": "video", "name": "MiniMax H3 (NVFP4, Turbo 8 steps, with audio)", "engine": "minimax_h3",
   "weights": "$W/minimax-h3", "idle_ttl_s": 600, "options": {"max_width": 768, "max_height": 448, "max_seconds": 5}}
JSON
)
[[ $set_ == all ]] || media=""
printf '{"machine": "dgx-01", "reserve_bytes": 8589934592,\n "defaults": {"text_to_speech": "kitten-tts-2"},\n "tools": [\n%s%s\n ]}\n' \
    "$speech" "$media" > "$out"
