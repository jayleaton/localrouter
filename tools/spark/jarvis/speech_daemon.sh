#!/usr/bin/env bash
# LocalRouter's own speech tools through the native daemon (loopback): Kitten TTS 2 then Whisper large-v3-turbo, each
# loaded by its first request (cold), then warm requests; memory per worker (MemAvailable drop, nvidia-smi, the
# worker's reported resident bytes). Whisper transcribes Kitten's WAV and the saved Drama 3 WAV.
#   speech_daemon.sh [DRAMA_WAV]
set -euo pipefail
J=$HOME/jarvis; S=$J/scripts; O=$J/out/speech-daemon; mkdir -p "$O"
PORT=${PORT:-28190}; URL=http://127.0.0.1:$PORT
LR=$J/src/localrouter-cuda/zig-out/bin/localrouter
TEXT="Good evening. Your calendar is clear tomorrow morning, but you have a dentist appointment at two and dinner with Sam at seven."
export PATH=$J/bin:$PATH   # the daemon decodes uploads with ffmpeg
phase() { echo "$(date +%s%3N) $1 $(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" | tee -a "$O/phases.txt"; }
tools() { curl -s $URL/v1/tools | jq -c '[.data[] | {id, state, resident_gib: (.resident_bytes / 1073741824 * 100 | round / 100)}]'; }
procs() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r pid mib; do
    echo "{\"pid\": $pid, \"gpu_mib\": ${mib% MiB}, \"rss_kib\": $(awk '/^VmRSS/{print $2}' /proc/$pid/status 2> /dev/null || echo 0), \"cmd\": \"$(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-90)\"}"; done; }
: > "$O/phases.txt"
KEEP=false bash "$S/daemon_config.sh" "$O/tools.json" speech
bash "$S/memwatch.sh" "$O/mem.csv" 0.5 & mw=$!
trap 'kill $mw 2> /dev/null; systemctl --user stop jarvis-lr 2> /dev/null || true' EXIT
sleep 2; phase baseline
systemd-run --user --unit=jarvis-lr -q -p MemoryMax=48G -p RuntimeMaxSec=3600 -E PATH \
    "$LR" serve --config "$O/tools.json" --host 127.0.0.1 --port $PORT --data "$O/data"
for _ in $(seq 100); do curl -fs $URL/health > /dev/null && break; sleep 0.2; done
phase daemon_up; tools | tee "$O/tools-0.json"
python3 -I "$S/speech_bench.py" $URL kitten-tts-2 whole 1 "$O/kitten-cold" "$TEXT" | tee "$O/kitten-cold.jsonl"
phase kitten_loaded; tools | tee "$O/tools-kitten.json"; procs | tee "$O/procs-kitten.jsonl"
python3 -I "$S/speech_bench.py" $URL kitten-tts-2 whole 3 "$O/kitten" "$TEXT" | tee "$O/kitten-warm.jsonl"
phase kitten_warm
python3 -I "$S/speech_bench.py" $URL whisper-large-v3-turbo stt 1 "$O/kitten/whole0.wav" | tee "$O/whisper-cold.jsonl"
phase whisper_loaded; tools | tee "$O/tools-both.json"; procs | tee "$O/procs-both.jsonl"
python3 -I "$S/speech_bench.py" $URL whisper-large-v3-turbo stt 3 "$O/kitten/whole0.wav" | tee "$O/whisper-warm.jsonl"
[[ -n ${1:-} ]] && python3 -I "$S/speech_bench.py" $URL whisper-large-v3-turbo stt 1 "$1" | tee "$O/whisper-drama.jsonl"
phase done
systemctl --user stop jarvis-lr; sleep 3; phase stopped
cp "$O"/data/logs/*.log "$O/" 2> /dev/null || true
kill $mw; trap - EXIT
bash "$S/memwatch.sh" --summary "$O/mem.csv" "$(awk '$2=="baseline"{print $3}' "$O/phases.txt")" | tee "$O/mem-summary.txt"
