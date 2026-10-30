#!/usr/bin/env bash
# One measured TensorFold chat run on DGX-01: cold (or warm-cache) start, smoke, TTFT, decode, a long-context fill,
# memory at each phase. Everything lives under ~/jarvis; the server runs as a bounded transient user unit.
#   chat_run.sh NAME cold|warm [extra serve flags...]
#   e.g. chat_run.sh drafter cold --drafter z-lab/Qwen3.8-27B-DFlash2      chat_run.sh plain warm --no-drafts
set -euo pipefail
name=${1:?NAME}; cache=${2:?cold|warm}; shift 2
J=$HOME/jarvis; S=$J/scripts; O=$J/out/chat-$name; mkdir -p "$O"
PORT=${PORT:-18080}; URL=http://127.0.0.1:$PORT; MODEL=qwen27b
export HF_HOME=$J/hf HF_HUB_CACHE=$J/hf/hub
phase() { echo "$(date +%s%3N) $1 $(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" | tee -a "$O/phases.txt"; }
: > "$O/phases.txt"
[[ $cache == cold ]] && { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'; }
bash "$S/memwatch.sh" "$O/mem.csv" 0.5 & mw=$!
trap 'kill $mw 2> /dev/null; systemctl --user stop jarvis-chat 2> /dev/null || true' EXIT
sleep 2; phase baseline
systemd-run --user --unit=jarvis-chat -q -p MemoryMax=96G -p RuntimeMaxSec=5400 -E HF_HOME -E HF_HUB_CACHE \
    "$J/bin/tensorfold" serve TensorFold/Qwen3.8-27B-MLX-4bit --name $MODEL --host 127.0.0.1 --port "$PORT" \
    --context 32768 --temperature 0 --no-thinking --no-update-check "$@"
phase start
python3 -I "$S/chat_bench.py" $URL $MODEL ready 900 | tee "$O/ready.json"
phase ready; sleep 5; phase loaded_idle
curl -s -H "Host: 127.0.0.1:$PORT" $URL/memory > "$O/memory-loaded.json" || true
python3 -I "$S/chat_bench.py" $URL $MODEL smoke | tee "$O/smoke.json"
python3 -I "$S/chat_bench.py" $URL $MODEL ttft 3 1000 | tee "$O/ttft.jsonl"
python3 -I "$S/chat_bench.py" $URL $MODEL decode 3 512 | tee "$O/decode.jsonl"
phase fill_start
python3 -I "$S/chat_bench.py" $URL $MODEL fill 30000 | tee "$O/fill.json"
phase fill_done
curl -s -H "Host: 127.0.0.1:$PORT" $URL/memory > "$O/memory-filled.json" || true
journalctl --user -u jarvis-chat --since "@$(( $(awk '$2=="start"{print $1}' "$O/phases.txt") / 1000 ))" --no-pager -o cat > "$O/server.log" 2>&1 || true
systemctl --user stop jarvis-chat; sleep 3; phase stopped
kill $mw; trap - EXIT
bash "$S/memwatch.sh" --summary "$O/mem.csv" "$(awk '$2=="baseline"{print $3}' "$O/phases.txt")" | tee "$O/mem-summary.txt"
