#!/usr/bin/env bash
# V1 (one PRO 6000): the video reference's environment and weights, then stock ComfyUI 0.37.0 headless on MiniMax H3
# text to video + audio at 768x448, 2 s: bf16 DiT with ComfyUI's default attention, with comfy-kitchen's INT8
# attention (--use-ck-attention), and with the Turbo LoRA at 8 steps. Two runs each (cold, then warm with a new seed).
set -uo pipefail
source "$(dirname "$0")/video-env.sh"
R=$STK/results/v1-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/v1-latest
run() { local name=$1; shift; echo "[v1 $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
run setup bash $TWIN/pod/video-setup.sh
source $TWIN/pod/video-env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $R/gpu.txt; free -g >> $R/gpu.txt
C="python -m stk_twin.comfy_run --width 768 --height 448 --seconds 2"
run stock-bf16 $C --steps 20 --out $R/stock-bf16
run stock-ck $C --steps 20 --out $R/stock-ck --comfy-args=--use-ck-attention
run stock-turbo $C --steps 8 --lora --out $R/stock-turbo --comfy-args=--use-ck-attention
find $R -name "*.mp4" -exec ls -la {} \; > $R/videos.txt
echo "[v1 $(date +%T)] done: $R"
