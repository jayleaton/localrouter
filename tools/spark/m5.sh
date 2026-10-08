#!/usr/bin/env bash
# M5 on a DGX Spark (GB10, sm_121), inside the NGC PyTorch container (docs/dev/RUNBOOK-M5.md starts it). Everything the
# pods did on sm_120, on sm_121: packs from the official checkpoint (checked byte for byte against the references),
# the twin's captures, the localrouter binary built natively, then the bit gates (DiT, text encoder, VAE replays; prompt to
# pixels for the 6 gate prompts at two sizes), and GB10's times (warm and cold, 1 MP and 0.3 MP, both precisions),
# peak memory and the attention variants. Steps go on if one fails; $R/steps.jsonl has each rc.
# Outputs: /workspace/localrouter/results/m5-<time>/ (summary.jsonl first). Captures are deleted at the end.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export STK=/workspace/localrouter
source $STK/twin/pod/env.sh
R=$STK/results/m5-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m5-latest
run() { local name=$1; shift; echo "[m5 $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
# GB10's memory is the system's: sample MemAvailable every second for the peak
( while :; do echo "$(date +%s) $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; sleep 1; done ) > $R/memavail.log &
MEM=$!
trap 'kill $MEM 2>/dev/null' EXIT

run preflight bash -c 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; awk "/MemAvailable/ {print \$2 / 1048576 \" GiB available\"}" /proc/meminfo'
run setup bash $STK/twin/pod/setup.sh
source $STK/twin/pod/env.sh   # again: on a fresh /workspace/localrouter the venv exists only now (env.sh activates it if present)
export MODEL=$MODEL_VOL   # the Spark's own disk is local: no copy
T="python -m stk_twin"; W=$STK/weights; C=$STK/captures; PK=$STK/repo/tools/twin/packs
export STK_ATTN=stk STK_OPS=stk STK_TE=stk STK_VAE=stk
run pack-nvfp4 $T pack --model $MODEL --precision nvfp4 --acts $PK/acts-nvfp4.json --out $W/nvfp4
run digests-nvfp4 python $PK/check.py $W/nvfp4 $PK/digests-nvfp4.json
run pack-fp8 $T pack --model $MODEL --precision fp8 --acts $PK/acts-fp8.json --out $W/fp8s
run digests-fp8 python $PK/check.py $W/fp8s $PK/digests-fp8s.json
run pack-te $T pack --model $MODEL --precision te --out $W/te
run pack-vae $T pack --model $MODEL --precision vae --out $W/vae
for p in nvfp4 fp8s; do  # each precision's own capture: its Triton set differs (FP8's MLP uses swiglu)
    rm -rf $C/m5-$p-512
    run capture-$p $T capture --model $MODEL --pack $W/$p --size 512x512 --steps 2 --out $C/m5-$p-512
    run triton-$p $T triton --model $MODEL --pack $C/m5-$p-512 --out $W/$p
done
run attnbench python -m stk_twin.attnbench

ZIG=$STK/zig/zig
[[ -x $ZIG ]] || run zig-install bash -c "mkdir -p $STK/zig && curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz | tar -xJ -C $STK/zig --strip-components=1"
SRC=/tmp/localrouter/stk-src; rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/; cd $SRC
export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
run build $ZIG build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $R/out fatbins install
run sass-nvfp4 env NATIVE=1 SMS=121 TF_PY=$STK/src/TensorFold TF_REF=$TENSORFOLD_TAG OUT=$R/sass FATBINS=$R/out/fatbin bash kernels/cuda/sass_check.sh
S=$R/out/bin/localrouter
run replay $S check qwen-replay $W/nvfp4 $C/m5-nvfp4-512
run te-replay $S check qwen-te $W/te $C/m5-nvfp4-512
run vae-replay $S check qwen-vae $W/vae $C/m5-nvfp4-512
e2e() { # PRECISION SIZE STEPS PROMPT
    local c=$C/m5-light-$1-$2-$3-p$4; rm -rf $c
    run cap-$1-$2-$3-p$4 $T capture --light --model $MODEL --pack $W/$1 --size $2 --steps $3 --prompt $4 --out $c
    run e2e-$1-$2-$3-p$4 $S check qwen-e2e $W/$1 $W/te $W/vae $c $R/zig-$1-$2-$3-p$4.png
    rm -rf $c
}
for i in 0 1 2 3 4 5; do for sz in 512x512 1024x1024; do e2e nvfp4 $sz 4 $i; done; done
for p in nvfp4 fp8s; do for sz in 1024x1024 576x576; do e2e $p $sz 25 0; done; done
rm -rf $C/m5-*
echo "[m5 $(date +%T)] done: $R"
