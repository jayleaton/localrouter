#!/usr/bin/env bash
# Qwen-Image 2.1 Turbo on a DGX Spark (GB10, sm_121), inside the NGC PyTorch container with $LOCALROUTER_HOME at
# /workspace/localrouter (tools/spark/turbo-window.sh starts it). Needs the Qwen-Image-2.1 snapshot in models/ (or it is
# downloaded) and the Turbo snapshot's own files (transformer/, scheduler/, model_index.json) in models/Qwen-Image-2.1-Turbo/;
# Turbo's text encoder, processor and VAE are Qwen-Image 2.1's (the same tensors; Turbo stores the VAE as base's fp32
# rounded to bf16, which is what the pipeline and the packs load), linked in. Steps go on if one fails; $R/steps.jsonl
# has each rc.
#   1. the localrouter binary: $STK/bin/localrouter if present (cross-built), else zig build here
#   2. Qwen-Image 2.1's packs (fp8s checked against the committed digests; te, vae made if missing)
#   3. Turbo's FP8 calibration (tools/twin/packs/acts-fp8-turbo.json unless committed), its pack (turbo-fp8s/, checked
#      against digests-turbo-fp8s.json when committed, else the digests are written for committing), its Triton cubins
#   4. the bit gates: the twin's light captures against `localrouter check qwen-e2e` (sigmas, text context, final
#      latents, decoded pixels byte for byte): Qwen-Image 2.1 at 25 steps (1 prompt, 1024) as the regression, run
#      before step 3 while Turbo's files may still be arriving (it waits for models/Qwen-Image-2.1-Turbo/.complete),
#      then Turbo at its 8 steps (2 prompts x 512 and 1024); then the bf16 twin against diffusers for Turbo
# Output: $STK/results/turbo-<time>/ (summary.jsonl first). Captures are deleted after their check.
set -uo pipefail
export STK=/workspace/localrouter
source $STK/twin/pod/env.sh
R=${R:-$STK/results/turbo-$(date -u +%Y%m%d-%H%M%S)}; mkdir -p $R; ln -sfn $R $STK/results/turbo-latest
run() { local name=$1; shift; echo "[turbo $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
run preflight bash -c 'nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader; nvcc --version | tail -1; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; awk "/MemAvailable/ {print \$2 / 1048576 \" GiB available\"}" /proc/meminfo'
[[ -f $MODEL_VOL/text_encoder/model.safetensors.index.json ]] && touch $MODEL_VOL/.complete   # staged: setup.sh skips the download
run setup bash $STK/twin/pod/setup.sh
source $STK/twin/pod/env.sh
export MODEL=$MODEL_VOL STK_ATTN=stk STK_OPS=stk STK_TE=stk STK_VAE=stk
TW=$STK/models/Qwen-Image-2.1-Turbo
T="python -m stk_twin"; W=$STK/weights/qwen-image-2.1; C=$STK/captures; PK=$STK/repo/tools/twin/packs
mkdir -p $W $C

# ---- 1. the binary
S=$STK/bin/localrouter
if [[ ! -x $S ]]; then
    ZIG=$STK/zig/zig
    [[ -x $ZIG ]] || run zig-install bash -c "mkdir -p $STK/zig && curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz | tar -xJ -C $STK/zig --strip-components=1"
    SRC=/tmp/localrouter/stk-src; rm -rf $SRC; mkdir -p $SRC; cp -r $STK/repo/. $SRC/
    export ZIG_GLOBAL_CACHE_DIR=$STK/zig-cache
    (cd $SRC && run build $ZIG build -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix $STK)
fi
run binary bash -c "file $S; sha256sum $S"

# ---- 2. Qwen-Image 2.1's packs
[[ -f $W/fp8s/manifest.json ]] || run pack-fp8s $T pack --model $MODEL --precision fp8 --acts $PK/acts-fp8.json --out $W/fp8s
run digests-fp8s python $PK/check.py $W/fp8s $PK/digests-fp8s.json
[[ -f $W/te/manifest.json ]] || run pack-te $T pack --model $MODEL --precision te --out $W/te
[[ -f $W/vae/manifest.json ]] || run pack-vae $T pack --model $MODEL --precision vae --out $W/vae
if [[ ! -f $W/fp8s/triton/triton.json ]]; then
    rm -rf $C/turbo-base-512
    run capture-fp8s $T capture --model $MODEL --pack $W/fp8s --size 512x512 --steps 2 --out $C/turbo-base-512
    run triton-fp8s $T triton --model $MODEL --pack $C/turbo-base-512 --out $W/fp8s
    rm -rf $C/turbo-base-512
fi

# ---- the Qwen-Image 2.1 regression (25 steps, 1024) first: it needs nothing of Turbo's, which may still be arriving
e2e() { # NAME MODEL_DIR PACK SIZE STEPS PROMPT
    local c=$C/turbo-light-$1-$4-p$6; rm -rf $c
    run cap-$1-$4-p$6 $T capture --light --model $2 --pack $W/$3 --size $4 --steps $5 --prompt $6 --out $c
    run e2e-$1-$4-p$6 $S check qwen-e2e $W/$3 $W/te $W/vae $c $R/zig-$1-$4-p$6.png
    cp $c/image.png $R/twin-$1-$4-p$6.png 2> /dev/null
    rm -rf $c
}
e2e base $MODEL fp8s 1024x1024 25 0
until [[ -f $TW/.complete ]]; do sleep 10; done   # written once Turbo's files are in place and sha256-checked
for d in text_encoder processor vae; do [[ -e $TW/$d ]] || ln -s ../Qwen-Image-2.1/$d $TW/$d; done

# ---- 3. Turbo's calibration, pack and cubins
ACTS=$PK/acts-fp8-turbo.json
[[ -f $ACTS ]] || { run calibrate-turbo $T calibrate --model $TW --precision fp8 --out $ACTS; cp $ACTS $R/; }
if [[ ! -f $W/turbo-fp8s/manifest.json ]]; then
    run pack-turbo $T pack --model $TW --repo Qwen/Qwen-Image-2.1-Turbo --precision fp8 --acts $ACTS --out $W/turbo-fp8s
fi
if [[ -f $PK/digests-turbo-fp8s.json ]]; then
    run digests-turbo python $PK/check.py $W/turbo-fp8s $PK/digests-turbo-fp8s.json
else   # the reference digests, for committing beside acts-fp8-turbo.json
    python -c "import json,sys; m=json.load(open(sys.argv[1])); json.dump({'pack': 'qwen-image-2.1-turbo-fp8s', 'precision': m['precision'], 'tensors': {k: v['sha256'] for k, v in m['tensors'].items()}}, open(sys.argv[2], 'w'), indent=0)" \
        $W/turbo-fp8s/manifest.json $R/digests-turbo-fp8s.json
fi
python -c "import json,sys; print(json.dumps({'turbo_pack_scheduler': json.load(open(sys.argv[1]))['scheduler']}))" $W/turbo-fp8s/manifest.json >> $R/summary.jsonl
if [[ ! -f $W/turbo-fp8s/triton/triton.json ]]; then
    rm -rf $C/turbo-512
    run capture-turbo $T capture --model $TW --pack $W/turbo-fp8s --size 512x512 --out $C/turbo-512
    run triton-turbo $T triton --model $TW --pack $C/turbo-512 --out $W/turbo-fp8s
    rm -rf $C/turbo-512
fi

# ---- 4. Turbo's bit gates (Zig against the twin), then the twin against diffusers
for p in 0 1; do for sz in 512x512 1024x1024; do e2e turbo $TW turbo-fp8s $sz 8 $p; done; done
run gate-turbo $T gate --model $TW --size 1024x1024 --out $R/gate-turbo
python -c "import sys; sys.path.insert(0, '$TWIN'); from stk_twin.files import readable; readable('$W')"
echo "[turbo $(date +%T)] done: $R"
