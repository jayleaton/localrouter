#!/usr/bin/env bash
# M3a on one PRO 6000: LocalRouter's small ops and attention against torch, the TensorFold FP8 path (calibrate, pack,
# bench), the bf16 gate with every toolkit kernel adopted, renders and fresh captures (with Triton metadata).
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m3a-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m3a-latest
bash $TWIN/pod/local.sh > $R/local.log 2>&1
run() { local name=$1; shift; echo "[m3a $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; }
T="python -m stk_twin"
P=$STK/packs
run test-ops python -m stk_twin.test_ops
run attncheck $T attncheck --model $MODEL
export STK_ATTN=stk STK_OPS=stk
run gate $T gate --model $MODEL --size 1024x1024 --out $R/gate
run calibrate-fp8 $T calibrate --model $MODEL --precision fp8 --out $P/acts-fp8.json
run pack-fp8s $T pack --model $MODEL --precision fp8 --acts $P/acts-fp8.json --out $P/qwen-image-2.1-fp8s
for p in nvfp4 fp8s; do
    for s in 1024x1024 608x480; do run bench-$p-$s $T bench --model $MODEL --pack $P/qwen-image-2.1-$p --size $s; done
    run render-$p $T render --model $MODEL --pack $P/qwen-image-2.1-$p --out $R/render-$p
    rm -rf $STK/captures/m3-$p
    run capture-$p $T capture --model $MODEL --pack $P/qwen-image-2.1-$p --size 512x512 --steps 4 --out $STK/captures/m3-$p
done
echo "[m3a $(date +%T)] done: $R"
