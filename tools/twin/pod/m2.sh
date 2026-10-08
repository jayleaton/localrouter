#!/usr/bin/env bash
# M2 on one PRO 6000: gate, calibrate, packs, renders, benches, captures. Each step logs to $R/<step>.log and the
# summary lines to $R/summary.jsonl; a failing step does not stop the others.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m2-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m2-latest
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $R/gpu.txt
run() { local name=$1; shift; echo "[m2 $(date +%T)] $name"; local t0=$(date +%s)
    python -m stk_twin "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; }
P=$STK/packs
t0=$(date +%s); bash $TWIN/pod/local.sh > $R/local.log 2>&1; echo "{\"step\": \"local-copy\", \"rc\": $?, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl
steps=${M2_STEPS:-gate calibrate pack render bench capture}
[[ $steps == *gate* ]] && run gate gate --model $MODEL --size 1024x1024 --out $R/gate
[[ $steps == *calibrate* ]] && run calibrate calibrate --model $MODEL --out $P/acts-nvfp4.json
if [[ $steps == *pack* ]]; then
    run pack-nvfp4 pack --model $MODEL --precision nvfp4 --acts $P/acts-nvfp4.json --out $P/qwen-image-2.1-nvfp4
    run pack-fp8 pack --model $MODEL --precision fp8 --out $P/qwen-image-2.1-fp8
fi
for p in nvfp4 fp8; do
    [[ $steps == *render* ]] && run render-$p render --model $MODEL --pack $P/qwen-image-2.1-$p --out $R/render-$p
    [[ $steps == *bench* ]] && for s in 1024x1024 608x480; do run bench-$p-$s bench --model $MODEL --pack $P/qwen-image-2.1-$p --size $s; done
    [[ $steps == *capture* ]] && run capture-$p capture --model $MODEL --pack $P/qwen-image-2.1-$p --size 512x512 --steps 4 --out $STK/captures/m2-$p
done
[[ $steps == *render* ]] && run render-bf16 render --model $MODEL --precision bf16 --out $R/render-bf16
echo "[m2 $(date +%T)] done: $R"
