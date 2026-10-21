#!/usr/bin/env bash
# Next-turn TTFT on a long conversation for each prompt-cache size: whether the retained prefix spares a re-prefill.
#   chat_turns.sh PROMPT_TOKENS GIB...      e.g. chat_turns.sh 12000 2 8
set -euo pipefail
tokens=${1:?PROMPT_TOKENS}; shift
J=$HOME/jarvis; S=$J/scripts; O=$J/out/chat-turns; mkdir -p "$O"
export HF_HOME=$J/hf HF_HUB_CACHE=$J/hf/hub
trap 'systemctl --user stop jarvis-chat 2> /dev/null || true' EXIT
for gib in "$@"; do
    s=$(date +%s); a0=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    systemd-run --user --unit=jarvis-chat -q -p MemoryMax=96G -p RuntimeMaxSec=1800 -E HF_HOME -E HF_HUB_CACHE \
        "$J/bin/tensorfold" serve TensorFold/Qwen3.8-27B-MLX-4bit --name qwen27b --host 127.0.0.1 --port 28080 \
        --context 32768 --temperature 0 --no-thinking --no-update-check --drafter z-lab/Qwen3.8-27B-DFlash2 \
        --parallel 1 --prompt-cache-gib "$gib"
    python3 -I "$S/chat_bench.py" http://127.0.0.1:28080 qwen27b ready 900 > /dev/null
    r=$(python3 -I "$S/chat_bench.py" http://127.0.0.1:28080 qwen27b turn "$tokens")
    a1=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    echo "{\"prompt_cache_gib\": $gib, \"drop_gib\": $(awk -v a=$a0 -v b=$a1 'BEGIN{printf "%.2f", (a-b)/1048576}'), \"turn\": $r}" | tee -a "$O/turns.jsonl"
    journalctl --user -u jarvis-chat --since "@$s" --no-pager -o cat | grep -E "prompt cache|done chatcmpl" > "$O/server-$gib.log" || true
    systemctl --user stop jarvis-chat; sleep 3
done
