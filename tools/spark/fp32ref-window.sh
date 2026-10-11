#!/usr/bin/env bash
# `stk_twin fp32ref` for Qwen-Image 2.1 Turbo and Qwen-Image 2.1 on a DGX Spark, in a window the owner granted, after
# tools/spark/turbo-window.sh staged $LOCALROUTER_HOME (models/, venv, twin, kernels; nothing is downloaded or built).
# Per model two container runs: the bf16 phase (twin, then diffusers with cuDNN and memory-efficient attention), then
# diffusers alone in fp32 on the same inputs (a comparison reference, not ground truth; the gate's verdict stands).
#   DEADLINE=<unix time the window ends> bash tools/spark/fp32ref-window.sh [--preflight]
# --preflight only checks the staging (models, venv, image, memory, GPU) and exits. The run stops at DEADLINE minus
# CLEANUP_S (default 600). A model is not run, and no result is claimed for it, when MemAvailable is under FP32_GIB
# (default 45; the 7B transformer is about 28 GiB in fp32) before its fp32 load, or falls under MIN_GIB (default 7) at
# any time (the container is killed). The lr-fp32ref container is removed on exit. Env: LOCALROUTER_HOME
# (/data/stk-qwen21turbo), MODELS (Qwen-Image-2.1-Turbo Qwen-Image-2.1). Output: $LOCALROUTER_HOME/results/fp32ref-<time>/:
# env.json, preflight.log, window.log, memavail.log, guard.log,
# <model>/{bf16,fp32}.log and <model>/fp32ref.json.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H=${LOCALROUTER_HOME:-/data/stk-qwen21turbo}
NGC=nvcr.io/nvidia/pytorch:26.07-py3
MODELS=${MODELS:-Qwen-Image-2.1-Turbo Qwen-Image-2.1}
avail_kib() { awk '/MemAvailable/ {print $2}' /proc/meminfo; }

preflight() {
    local bad=0 m
    for m in $MODELS; do
        for f in transformer/config.json transformer/diffusion_pytorch_model.safetensors.index.json \
                 text_encoder/config.json vae/config.json model_index.json; do
            [[ -e $H/models/$m/$f ]] || { echo "missing: models/$m/$f"; bad=1; }
        done
    done
    for f in venv/bin/activate twin/pod/env.sh twin/stk_twin/__main__.py kernels/cuda/qwen_image/ops.cu; do
        [[ -e $H/$f ]] || { echo "missing: $f"; bad=1; }
    done
    docker image inspect $NGC --format 'image {{.Id}}' || bad=1
    echo "MemAvailable $(( $(avail_kib) / 1048576 )) GiB; GPU apps: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ')"
    docker ps --format '{{.Names}}' | grep -qx lr-fp32ref && { echo "lr-fp32ref already running"; bad=1; }
    echo "preflight: $([[ $bad == 0 ]] && echo ok || echo FAILED)"
    return $bad
}
if [[ ${1:-} == --preflight ]]; then preflight; exit; fi

: "${DEADLINE:?DEADLINE (unix seconds) is required}"
CLEANUP_S=${CLEANUP_S:-600}
MIN_KIB=$(( ${MIN_GIB:-7} * 1048576 )); FP32_KIB=$(( ${FP32_GIB:-45} * 1048576 ))
RUN=fp32ref-$(date -u +%Y%m%d-%H%M%S); R=$H/results/$RUN; mkdir -p $R
step() { echo "[fp32ref $(date -u +%T)] $*" | tee -a $R/window.log; }
left() { echo $(( DEADLINE - CLEANUP_S - $(date +%s) )); }
cleanup() {
    docker rm -f lr-fp32ref > /dev/null 2>&1
    kill $GUARD 2> /dev/null
    step "cleanup: container removed; GPU: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ')"
}
( while :; do
    m=$(avail_kib); echo "$(date +%s) $m"
    if (( m < MIN_KIB )) && docker kill lr-fp32ref > /dev/null 2>&1; then
        echo "$(date -u +%T) guard: MemAvailable $m KiB under ${MIN_GIB:-7} GiB, lr-fp32ref killed" >&2
    fi
    sleep 1
  done ) > $R/memavail.log 2> $R/guard.log &
GUARD=$!
trap cleanup EXIT
trap 'exit 130' INT TERM
preflight 2>&1 | tee $R/preflight.log; (( PIPESTATUS[0] == 0 )) || { step "preflight failed: not starting"; exit 1; }
(( $(left) > 600 )) || { step "less than 10 minutes left before the deadline: not starting"; exit 1; }
cp "$REPO/tools/twin/stk_twin/__main__.py" $H/twin/stk_twin/__main__.py   # the only file this adds to the staged twin
python3 - "$R/env.json" "$REPO" "$NGC" "$H/twin/stk_twin/__main__.py" <<'PY'
import hashlib, json, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump({"gpu": sh("nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader"),
           "kernel": sh("uname -srm"), "image": sh(f"docker image inspect {sys.argv[3]} --format '{{{{.Id}}}}'"),
           "tree": sh(f"git -C {sys.argv[2]} rev-parse HEAD 2>/dev/null") or "copied tree",
           "twin_main_sha256": hashlib.sha256(open(sys.argv[4], "rb").read()).hexdigest()}, open(sys.argv[1], "w"), indent=1)
PY

for m in $MODELS; do
    mkdir -p $R/$m
    for phase in bf16 fp32; do
        (( $(left) > 120 )) || { step "$m $phase: no time left, not run"; exit 1; }
        if [[ $phase == fp32 ]] && (( $(avail_kib) < FP32_KIB )); then
            step "$m fp32: MemAvailable $(( $(avail_kib) / 1048576 )) GiB under ${FP32_GIB:-45}: not run, no result"; continue 2
        fi
        step "$m $phase ($(left) s before cleanup, MemAvailable $(( $(avail_kib) / 1048576 )) GiB)"
        timeout --signal=TERM --kill-after=30 $(left) docker run --name lr-fp32ref --rm --gpus all --ipc host \
            --ulimit memlock=-1 --ulimit stack=67108864 -v $H:/workspace/localrouter -v $H:/workspace/stk $NGC bash -c \
            "source /workspace/localrouter/twin/pod/env.sh && python -m stk_twin fp32ref --phase $phase \
             --model /workspace/localrouter/models/$m --size 1024x1024 --out /workspace/localrouter/results/$RUN/$m" \
            > $R/$m/$phase.log 2>&1
        rc=$?; step "$m $phase rc $rc"; docker rm -f lr-fp32ref > /dev/null 2>&1
        (( rc == 0 )) || { step "$m: stopped after $phase, no result"; continue 2; }
    done
    rm -f $R/$m/bf16.pt
done
step "done: $R"
