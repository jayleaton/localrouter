#!/usr/bin/env bash
# V2 (one PRO 6000): the video twin's first checks. The H3 kernels against ComfyUI's and tfvideo's formulas, the
# copied comfy-kitchen kernels against the installed wheel (bitwise), then the structure gate: the twin's DiT in bf16
# mode against ComfyUI 0.37.0's own MiniMaxH3Model, step by step, against the attention-backend floor.
set -uo pipefail
source "$(dirname "$0")/video-env.sh"
R=$STK/results/v2-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/v2-latest
run() { local name=$1; shift; echo "[v2 $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
run setup bash $TWIN/pod/video-setup.sh
source $TWIN/pod/video-env.sh
run test-h3-ops python -m stk_twin.h3.test_ops
run test-kitchen python -m stk_twin.h3.test_kitchen
run gate-bf16 python -m stk_twin.h3.gate --models $VMODELS --size 768x448 --frames 56 --steps 8
run gate-int8 python -m stk_twin.h3.gate --models $VMODELS --size 768x448 --frames 56 --steps 8 --attn int8
echo "[v2 $(date +%T)] done: $R"
