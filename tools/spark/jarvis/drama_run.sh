#!/usr/bin/env bash
# TensorFold v1.1.1's native Drama 3 speech server on DGX-01: load time, memory, streamed first audio, RTF, one WAV.
#   drama_run.sh NAME cold|warm [extra serve flags...]
set -euo pipefail
name=${1:?NAME}; cache=${2:?cold|warm}; shift 2
J=$HOME/jarvis; S=$J/scripts; O=$J/out/drama-$name; mkdir -p "$O"
PORT=${PORT:-28793}; URL=http://127.0.0.1:$PORT; MODEL=fishaudio/drama-3
TEXT=${TEXT:-"The morning train leaves the station at seven fifteen, crosses the river by the old stone bridge, and reaches the coast just before nine."}
export HF_HOME=$J/hf HF_HUB_CACHE=$J/hf/hub
phase() { echo "$(date +%s%3N) $1 $(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" | tee -a "$O/phases.txt"; }
: > "$O/phases.txt"
[[ $cache == cold ]] && { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; }
bash "$S/memwatch.sh" "$O/mem.csv" 0.5 & mw=$!
trap 'kill $mw 2> /dev/null; systemctl --user stop jarvis-drama 2> /dev/null || true' EXIT
sleep 2; phase baseline
systemd-run --user --unit=jarvis-drama -q -p MemoryMax=64G -p RuntimeMaxSec=3600 -E HF_HOME -E HF_HUB_CACHE \
    "$J/bin/tensorfold" serve $MODEL --name $MODEL --host 127.0.0.1 --port "$PORT" --no-update-check "$@"
phase start; start_s=$(date +%s)
python3 -I "$S/chat_bench.py" $URL $MODEL ready 900 | tee "$O/ready.json"
phase ready; sleep 5; phase loaded_idle
curl -s -H "Host: 127.0.0.1:$PORT" $URL/memory > "$O/memory-loaded.json" || true
python3 -I "$S/speech_bench.py" $URL $MODEL whole 1 "$O" "Warm up." > /dev/null
python3 -I "$S/speech_bench.py" $URL $MODEL stream 3 44100 "$O" "$TEXT" | tee "$O/stream.jsonl"
python3 -I "$S/speech_bench.py" $URL $MODEL whole 3 "$O" "$TEXT" | tee "$O/whole.jsonl"
phase done
curl -s -H "Host: 127.0.0.1:$PORT" $URL/memory > "$O/memory-done.json" || true
journalctl --user -u jarvis-drama --since "@$start_s" --no-pager -o cat > "$O/server.log" 2>&1 || true
systemctl --user stop jarvis-drama; sleep 3; phase stopped
kill $mw; trap - EXIT
bash "$S/memwatch.sh" --summary "$O/mem.csv" "$(awk '$2=="baseline"{print $3}' "$O/phases.txt")" | tee "$O/mem-summary.txt"
