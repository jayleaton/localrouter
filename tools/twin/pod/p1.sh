#!/usr/bin/env bash
# P1 (speed work, one small sm_120 pod): the packs every later session reuses (text encoder, VAE, tfimage's Triton
# cubins for sm_120 and sm_121 inside the DiT packs; kept on the volume), attnbench (bit-equal staging variants and
# FlashAttention-2 arithmetic variants vs the shipping kernel, an fp32 reference and cuDNN), and the Zig step graphs:
# the replays (probe path, no graphs) and qwen-e2e with graphs on and off (same pixels; the time saved).
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/p1-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/p1-latest
run() { local name=$1; shift; echo "[p1 $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
T="python -m stk_twin"; P=$STK/packs; LP=/tmp/localrouter/packs; C=/tmp/localrouter/captures
export STK_ATTN=stk STK_OPS=stk STK_TE=stk STK_VAE=stk
pcp() { mkdir -p "$2"; (cd "$1" && find . -type f -print0 | xargs -0 -P 16 -I{} sh -c 'mkdir -p "$(dirname "$0/{}")" && cp -u "{}" "$0/{}"' "$2"); }
run local bash $TWIN/pod/local.sh
run copy-packs bash -c "$(declare -f pcp); pcp $P/qwen-image-2.1-nvfp4 $LP/nvfp4 && pcp $P/qwen-image-2.1-fp8s $LP/fp8s"
run attnbench python -m stk_twin.attnbench
[[ -f $P/qwen-image-2.1-te/manifest.json ]] && run copy-te bash -c "$(declare -f pcp); pcp $P/qwen-image-2.1-te $LP/te" \
    || { run pack-te $T pack --model $MODEL --precision te --out $LP/te; cp -r $LP/te $P/qwen-image-2.1-te; }
[[ -f $P/qwen-image-2.1-vae/manifest.json ]] && run copy-vae bash -c "$(declare -f pcp); pcp $P/qwen-image-2.1-vae $LP/vae" \
    || { run pack-vae $T pack --model $MODEL --precision vae --out $LP/vae; cp -r $LP/vae $P/qwen-image-2.1-vae; }
rm -rf $C/p1-nvfp4-512
export STK_AOT_ARCHS=121; run capture $T capture --model $MODEL --pack $LP/nvfp4 --size 512x512 --steps 2 --out $C/p1-nvfp4-512; unset STK_AOT_ARCHS
for p in nvfp4 fp8s; do  # FP8's set needs its own capture (swiglu); made once and kept on the volume
    [[ $p == fp8s ]] && { export STK_AOT_ARCHS=121; run capture-fp8s $T capture --model $MODEL --pack $LP/fp8s --size 512x512 --steps 2 --out $C/p1-fp8s-512; unset STK_AOT_ARCHS; }
    run triton-$p $T triton --model $MODEL --pack $C/p1-$p-512 --out $LP/$p
    mkdir -p $P/qwen-image-2.1-$p/triton && cp $LP/$p/triton/* $P/qwen-image-2.1-$p/triton/
done
SRC=/tmp/localrouter/stk-src; rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/; cd $SRC
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
run build $STK/zig/zig build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $R/out fatbins install
S=$R/out/bin/stk
run replay $S check qwen-replay $LP/nvfp4 $C/p1-nvfp4-512
run te-replay $S check qwen-te $LP/te $C/p1-nvfp4-512
run vae-replay $S check qwen-vae $LP/vae $C/p1-nvfp4-512
run e2e-full $S check qwen-e2e $LP/nvfp4 $LP/te $LP/vae $C/p1-nvfp4-512
for p in nvfp4 fp8s; do
    c=$C/p1-light-$p-1024-25; rm -rf $c
    run cap-$p $T capture --light --model $MODEL --pack $LP/$p --size 1024x1024 --steps 25 --prompt 0 --out $c
    run e2e-$p $S check qwen-e2e $LP/$p $LP/te $LP/vae $c
done
rm -rf $C/p1-*   # captures are cheap to remake; keep the disk free
echo "[p1 $(date +%T)] done: $R"
