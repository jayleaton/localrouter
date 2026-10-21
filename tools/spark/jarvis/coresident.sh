#!/usr/bin/env bash
# The whole Jarvis box on DGX-01 at once: the 27B chat server (TensorFold, single user), LocalRouter's native daemon
# with Whisper and Kitten kept warm, optionally TensorFold's Drama 3 server; total memory; then one image job and one
# video job through LocalRouter's scheduler, recording what it unloads to make room and whether the job is admitted.
#   [VIDEO_SECONDS=1..5] coresident.sh NAME with-drama|no-drama
set -euo pipefail
name=${1:?NAME}; drama=${2:?with-drama|no-drama}
J=$HOME/jarvis; S=$J/scripts; O=$J/out/co-$name; mkdir -p "$O"
LR=$J/src/localrouter-cuda/zig-out/bin/localrouter; URL=http://127.0.0.1:28190
export PATH=$J/bin:$PATH HF_HOME=$J/hf HF_HUB_CACHE=$J/hf/hub
phase() { echo "$(date +%s%3N) $1 $(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" | tee -a "$O/phases.txt"; }
tools() { curl -s $URL/v1/tools | jq -c '{avail_gib: (.mem_available_bytes / 1073741824 * 10 | round / 10), tools: [.data[] | {id, state, gib: (.resident_bytes / 1073741824 * 10 | round / 10)}]}' | tee -a "$O/tools.jsonl"; }
procs() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tee -a "$O/procs.txt"; }
stop_all() { systemctl --user stop jarvis-lr jarvis-drama jarvis-chat 2> /dev/null || true; }
: > "$O/phases.txt"
bash "$S/memwatch.sh" "$O/mem.csv" 0.5 & mw=$!
trap 'kill $mw ${tp:-} 2> /dev/null; stop_all' EXIT
sleep 2; phase baseline; s0=$(date +%s)
systemd-run --user --unit=jarvis-chat -q -p MemoryMax=96G -p RuntimeMaxSec=5400 -E HF_HOME -E HF_HUB_CACHE \
    "$J/bin/tensorfold" serve TensorFold/Qwen3.8-27B-MLX-4bit --name qwen27b --host 127.0.0.1 --port 28080 \
    --context 32768 --temperature 0 --no-thinking --no-update-check --drafter z-lab/Qwen3.8-27B-DFlash2 \
    --parallel 1 --prompt-cache-gib 2
python3 -I "$S/chat_bench.py" http://127.0.0.1:28080 qwen27b ready 900; phase chat
KEEP=true bash "$S/daemon_config.sh" "$O/tools.json" all
systemd-run --user --unit=jarvis-lr -q -p RuntimeMaxSec=5400 -E PATH \
    "$LR" serve --config "$O/tools.json" --host 127.0.0.1 --port 28190 --data "$O/data"
for _ in $(seq 600); do   # keep_loaded tools load at start; wait for both
    [[ $(curl -s $URL/v1/tools | jq '[.data[] | select(.keep_loaded and .state == "ready")] | length') == 2 ]] && break; sleep 0.5
done
phase speech; tools; procs
python3 -I "$S/speech_bench.py" $URL kitten-tts-2 whole 1 "$O/tts" "Your timer is done." | tee -a "$O/requests.jsonl"
python3 -I "$S/speech_bench.py" $URL whisper-large-v3-turbo stt 1 "$O/tts/whole0.wav" | tee -a "$O/requests.jsonl"
python3 -I "$S/chat_bench.py" http://127.0.0.1:28080 qwen27b smoke | cut -c1-300 | tee -a "$O/requests.jsonl"
if [[ $drama == with-drama ]]; then
    systemd-run --user --unit=jarvis-drama -q -p MemoryMax=64G -p RuntimeMaxSec=5400 -E HF_HOME -E HF_HUB_CACHE \
        "$J/bin/tensorfold" serve fishaudio/drama-3 --name fishaudio/drama-3 --host 127.0.0.1 --port 28793 \
        --no-update-check --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0 --keep-warm 0
    python3 -I "$S/chat_bench.py" http://127.0.0.1:28793 fishaudio/drama-3 ready 900 && phase drama
    python3 -I "$S/speech_bench.py" http://127.0.0.1:28793 fishaudio/drama-3 whole 1 "$O/drama" "Your timer is done." | tee -a "$O/requests.jsonl" || true
fi
phase warm_set; tools; procs
( while :; do tools > /dev/null; sleep 2; done ) & tp=$!   # tool states every 2 s through both jobs: who was unloaded
# an image job: the scheduler may unload idle tools (lowest priority first); chat and Drama are outside it
t0=$(date +%s.%N)
code=$(curl -s -m 1200 -o "$O/image.json" -w '%{http_code}' $URL/v1/images/generations -H 'content-type: application/json' \
    -d '{"model": "qwen-image-2.1-turbo", "prompt": "a red fox in fresh snow, morning light", "size": "1024x1024", "seed": 1}')
echo "{\"job\": \"image\", \"http\": $code, \"wall_s\": $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}'), \"error\": $(jq -c '.error // null' "$O/image.json")}" | tee -a "$O/requests.jsonl"
phase image_done; tools; procs
for _ in $(seq 60); do sleep 1; done; phase image_settled; tools
# a video job (VIDEO_SECONDS, default 1 as install.sh checks, at 768x448, 8 steps)
t0=$(date +%s.%N)
id=$(curl -s $URL/v1/videos -H 'content-type: application/json' \
    -d '{"model": "minimax-h3", "prompt": "waves at dusk", "size": "768x448", "seconds": "'"${VIDEO_SECONDS:-1}"'", "steps": 8, "seed": 1}' | tee "$O/video-create.json" | jq -r '.id // empty')
st=none
if [[ -n $id ]]; then
    for _ in $(seq 1200); do
        st=$(curl -s $URL/v1/videos/$id | tee "$O/video.json" | jq -r .status)
        [[ $st == completed || $st == failed || $st == cancelled ]] && break
        sleep 1
    done
fi
echo "{\"job\": \"video\", \"status\": \"$st\", \"wall_s\": $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}'), \"error\": $(jq -c '.error // null' "$O/video.json" 2> /dev/null || echo null)}" | tee -a "$O/requests.jsonl"
phase video_done; tools; procs
for _ in $(seq 60); do sleep 1; done; phase video_settled; tools
python3 -I "$S/chat_bench.py" http://127.0.0.1:28080 qwen27b smoke | cut -c1-200 | tee -a "$O/requests.jsonl"
journalctl --user -u jarvis-chat -u jarvis-drama -u jarvis-lr --since "@$s0" --no-pager -o cat > "$O/journal.log" 2>&1 || true
cp "$O"/data/logs/*.log "$O/" 2> /dev/null || true
kill $tp 2> /dev/null || true; stop_all; sleep 3; phase stopped
kill $mw; trap - EXIT
bash "$S/memwatch.sh" --summary "$O/mem.csv" "$(awk '$2=="baseline"{print $3}' "$O/phases.txt")" | tee "$O/mem-summary.txt"
