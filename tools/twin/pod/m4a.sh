#!/usr/bin/env bash
# M4a on one small sm_120 pod: the new kernels' tests (gemm.cu, te.cu in test_ops; vae.cu in test_vae), the twin's
# text encoder against transformers' (tegate), the text encoder's pack, an NVFP4 capture with LocalRouter's text
# encoder, then the stk binary (every fatbin) and the Zig text encoder's replay gate (qwen-te) plus the M3 replay as
# a regression check. Steps run on even if one fails; $R/steps.jsonl has each rc. STEPS="a b" runs a subset.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m4a-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m4a-latest
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
run() { local name=$1; shift; want $name || return 0; echo "[m4a $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
T="python -m stk_twin"; P=$STK/packs; TEP=/tmp/localrouter/packs/te
export STK_ATTN=stk STK_OPS=stk STK_TE=stk
run local bash $TWIN/pod/local.sh
pcp() { mkdir -p "$2"; (cd "$1" && find . -type f -print0 | xargs -0 -P 16 -I{} sh -c 'mkdir -p "$(dirname "$0/{}")" && cp -u "{}" "$0/{}"' "$2"); }
run copy-pack bash -c "$(declare -f pcp); pcp $P/qwen-image-2.1-nvfp4 /tmp/localrouter/packs/nvfp4"
run test-ops python -m stk_twin.test_ops
run test-vae env MODEL=$MODEL python -m stk_twin.test_vae
run tegate $T tegate --model $MODEL
run pack-te $T pack --model $MODEL --precision te --out $TEP
want capture && rm -rf /tmp/localrouter/captures/m4-nvfp4
run capture $T capture --model $MODEL --pack /tmp/localrouter/packs/nvfp4 --size 512x512 --steps 4 --out /tmp/localrouter/captures/m4-nvfp4

ZIG=$STK/zig/zig; SRC=/tmp/localrouter/stk-src
if want build; then rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/; fi
cd $SRC 2>/dev/null || true
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
run build $ZIG build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $R/out fatbins install
run test-unit $ZIG build test-unit
run te-replay $R/out/bin/localrouter check qwen-te $TEP /tmp/localrouter/captures/m4-nvfp4
run replay-m3 $R/out/bin/localrouter check qwen-replay /tmp/localrouter/packs/nvfp4 /tmp/localrouter/captures/m4-nvfp4
echo "[m4a $(date +%T)] done: $R"
