#!/usr/bin/env bash
# M4b on one small sm_120 pod: the text encoder's and the VAE's packs, captures with LocalRouter's text encoder and
# VAE (NVFP4 and FP8, 512x512 and 1024x1024, 4 steps), the kernel tests, then the stk binary and the Zig gates:
# qwen-te and qwen-vae (alone + chained) and qwen-e2e (prompt to pixels, byte for byte, with timings).
# Steps run on even if one fails; $R/steps.jsonl has each rc. STEPS="a b" runs a subset.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m4b-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m4b-latest
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
run() { local name=$1; shift; want $name || return 0; echo "[m4b $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
T="python -m stk_twin"; P=$STK/packs; LP=/tmp/localrouter/packs; C=/tmp/localrouter/captures
export STK_ATTN=stk STK_OPS=stk STK_TE=stk STK_VAE=stk
run local bash $TWIN/pod/local.sh
pcp() { mkdir -p "$2"; (cd "$1" && find . -type f -print0 | xargs -0 -P 16 -I{} sh -c 'mkdir -p "$(dirname "$0/{}")" && cp -u "{}" "$0/{}"' "$2"); }
run copy-packs bash -c "$(declare -f pcp); pcp $P/qwen-image-2.1-nvfp4 $LP/nvfp4 && pcp $P/qwen-image-2.1-fp8s $LP/fp8s"
run pack-te $T pack --model $MODEL --precision te --out $LP/te
run pack-vae $T pack --model $MODEL --precision vae --out $LP/vae
for p in nvfp4 fp8s; do
    for sz in 512 1024; do
        want capture && rm -rf $C/m4b-$p-$sz
        run capture-$p-$sz $T capture --model $MODEL --pack $LP/$p --size ${sz}x${sz} --steps 4 --out $C/m4b-$p-$sz
    done
done
run test-ops python -m stk_twin.test_ops

ZIG=$STK/zig/zig; SRC=/tmp/localrouter/stk-src
if want build; then rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/; fi
cd $SRC 2>/dev/null || true
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
run build $ZIG build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $R/out fatbins install
run test-unit $ZIG build test-unit
S=$R/out/bin/stk
run te-replay $S check qwen-te $LP/te $C/m4b-nvfp4-512
run vae-replay-512 $S check qwen-vae $LP/vae $C/m4b-nvfp4-512
run vae-replay-1024 $S check qwen-vae $LP/vae $C/m4b-nvfp4-1024
for p in nvfp4 fp8s; do
    for sz in 512 1024; do
        run e2e-$p-$sz $S check qwen-e2e $LP/$p $LP/te $LP/vae $C/m4b-$p-$sz $R/e2e-$p-$sz.png
        cp $C/m4b-$p-$sz/image.png $R/twin-$p-$sz.png 2>/dev/null
    done
done
echo "[m4b $(date +%T)] done: $R"
