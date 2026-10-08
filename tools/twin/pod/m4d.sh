#!/usr/bin/env bash
# M4d, after m4b on the same pod: the end-to-end gate over the 6 gate prompts. tfimage's Triton cubins go into each
# DiT pack (from m4b's captures); light captures (request, sigmas, context, final latents, pixels and the twin's warm
# times; no op stream) for every prompt at 512x512 and 1024x1024 with NVFP4 and 4 steps, and prompt 0 at 1024x1024
# with 25 steps for both precisions (the timing comparison); then `localrouter check qwen-e2e` on each, from m4b's binary
# rebuilt from the staged repo.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m4d-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m4d-latest
run() { local name=$1; shift; echo "[m4d $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
T="python -m stk_twin"; LP=/tmp/localrouter/packs; C=/tmp/localrouter/captures
export STK_ATTN=stk STK_OPS=stk STK_TE=stk STK_VAE=stk
rm -rf $C/m4-nvfp4  # M4a's, superseded
for p in nvfp4 fp8s; do run triton-$p $T triton --model $MODEL --pack $C/m4b-$p-512 --out $LP/$p; done
SRC=/tmp/localrouter/stk-src; rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/; cd $SRC
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
run build $STK/zig/zig build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $R/out fatbins install
S=$R/out/bin/stk
e2e() { # PRECISION SIZE STEPS PROMPT
    local c=$C/light-$1-$2-$3-p$4
    rm -rf $c
    run cap-$1-$2-$3-p$4 $T capture --light --model $MODEL --pack $LP/$1 --size ${2}x${2} --steps $3 --prompt $4 --out $c
    run e2e-$1-$2-$3-p$4 $S check qwen-e2e $LP/$1 $LP/te $LP/vae $c $R/zig-$1-$2-$3-p$4.png
    cp $c/image.png $R/twin-$1-$2-$3-p$4.png
}
for i in 0 1 2 3 4 5; do for sz in 512 1024; do e2e nvfp4 $sz 4 $i; done; done
for p in nvfp4 fp8s; do e2e $p 1024 25 0; done
echo "[m4d $(date +%T)] done: $R"
