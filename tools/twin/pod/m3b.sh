#!/usr/bin/env bash
# M3b: after the blocked dense kernel: the bf16 gate with every toolkit kernel, clean benches, captures.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m3b-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m3b-latest
bash $TWIN/pod/local.sh > $R/local.log 2>&1
run() { local name=$1; shift; echo "[m3b $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; }
T="python -m stk_twin"; P=$STK/packs
export STK_ATTN=stk STK_OPS=stk
run test-ops python -m stk_twin.test_ops
run gate $T gate --model $MODEL --size 1024x1024 --out $R/gate
for p in nvfp4 fp8s; do
    run bench-$p $T bench --model $MODEL --pack $P/qwen-image-2.1-$p --size 1024x1024
    rm -rf $STK/captures/m3-$p
    run capture-$p $T capture --model $MODEL --pack $P/qwen-image-2.1-$p --size 512x512 --steps 4 --out $STK/captures/m3-$p
done
STK_ATTN=cudnn STK_OPS=torch run bench-nvfp4-torch $T bench --model $MODEL --pack $P/qwen-image-2.1-nvfp4 --size 1024x1024
echo "[m3b $(date +%T)] done: $R"
