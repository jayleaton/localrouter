#!/usr/bin/env bash
# Qwen-Image 2.1 Turbo on a DGX Spark in one granted window: the packs and bit gates (turbo-inner.sh in the NGC PyTorch
# container), then Qwen-Image 2.1 and Turbo compared through the real daemon (`localrouter serve` in the same image,
# compare_models.py from the host). Run on the Spark as root, only in a window the owner granted:
#   DEADLINE=<unix time the window ends> bash tools/spark/turbo-window.sh
# Everything stops at DEADLINE minus CLEANUP_S (default 600): containers are named lr-turbo-*, and whatever happens
# they are removed on exit, so the GPU is free when the script returns. Nothing else on the machine is touched.
# The tree ($REPO) is copied into $LOCALROUTER_HOME (repo, twin, kernels) first. Env: LOCALROUTER_HOME
# (/data/stk-qwen21turbo), DEADLINE (required), CLEANUP_S, SKIP_INNER=1 (the comparison only), WARM (3 repeats).
# Output: $LOCALROUTER_HOME/results/turbo-<time>/ (inner steps, compare/ with the six PNGs, results.json, summary.md,
# env.json: the hardware, driver, image, checkpoint revisions, tree and pack digests).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H=${LOCALROUTER_HOME:-/data/stk-qwen21turbo}
NGC=nvcr.io/nvidia/pytorch:26.07-py3
: "${DEADLINE:?DEADLINE (unix seconds) is required}"
CLEANUP_S=${CLEANUP_S:-600}
RUN=turbo-$(date -u +%Y%m%d-%H%M%S); R=$H/results/$RUN; mkdir -p $R/compare
step() { echo "[turbo-window $(date -u +%T)] $*" | tee -a $R/window.log; }
left() { echo $(( DEADLINE - CLEANUP_S - $(date +%s) )); }
cleanup() {
    docker rm -f lr-turbo-inner lr-turbo-serve > /dev/null 2>&1
    kill $MEM 2> /dev/null
    step "cleanup: containers removed; GPU: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ')"
}
( while :; do echo "$(date +%s) $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; sleep 1; done ) > $R/memavail.log &
MEM=$!
trap cleanup EXIT
trap 'exit 130' INT TERM
(( $(left) > 600 )) || { step "less than 10 minutes left before the deadline: not starting"; exit 1; }

for d in tools/twin:twin kernels:kernels .:repo; do
    src=${d%%:*} dst=${d##*:}; mkdir -p $H/$dst
    tar -C "$REPO/$src" --exclude=.zig-cache --exclude=zig-out --exclude=zig-pkg --exclude=results --exclude=__pycache__ \
        --exclude=.git --exclude=.agents -cf - . | tar -C $H/$dst -xf -
done
python3 - "$R/env.json" "$REPO" <<'PY'
import json, subprocess, sys
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
json.dump({"gpu": sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader"),
           "kernel": sh("uname -srm"), "image": sh("docker image inspect nvcr.io/nvidia/pytorch:26.07-py3 --format '{{.Id}}'"),
           "tree": sh(f"git -C {sys.argv[2]} rev-parse HEAD 2>/dev/null") or "copied tree (see repo/)",
           "checkpoints": {"Qwen/Qwen-Image-2.1": "d26bb61231c349cf6b7896fa83353113880e1ba3",
                           "Qwen/Qwen-Image-2.1-Turbo": "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"}},
          open(sys.argv[1], "w"), indent=1)
PY

DOCKER=(docker run --name lr-turbo-inner --rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 -v $H:/workspace/localrouter \
    -v $H:/workspace/stk)   # the staged venv and editable installs were made under /workspace/stk
if [[ ${SKIP_INNER:-0} != 1 ]]; then
    step "inner: packs and bit gates ($(left) s before cleanup)"
    timeout --signal=TERM --kill-after=30 $(left) "${DOCKER[@]}" -e R=/workspace/localrouter/results/$RUN/inner $NGC \
        bash /workspace/localrouter/repo/tools/spark/turbo-inner.sh > $R/inner.log 2>&1
    step "inner rc $? (steps: $R/inner/steps.jsonl)"
    docker rm -f lr-turbo-inner > /dev/null 2>&1
fi

# ---- the comparison: both models from the same packs, each with its own defaults
(( $(left) > 900 )) || { step "under 15 minutes left: skipping the comparison"; exit 1; }
cat > $R/compare/tools.json <<'EOF'
{"reserve_bytes": 8589934592, "defaults": {"text_to_image": "qwen-image-2.1-turbo"}, "tools": [
 {"id": "qwen-image-2.1", "kind": "image", "name": "Qwen-Image 2.1 (FP8)", "engine": "qwen_image", "weights": "/models/qwen-image-2.1",
  "idle_ttl_s": 600, "options": {"precision": "fp8s", "max_side": 1664}},
 {"id": "qwen-image-2.1-turbo", "kind": "image", "name": "Qwen-Image 2.1 Turbo (FP8, 8 steps)", "engine": "qwen_image_turbo",
  "weights": "/models/qwen-image-2.1", "idle_ttl_s": 600, "options": {"precision": "fp8s", "max_side": 1664}}]}
EOF
mkdir -p $R/compare/data && chmod 777 $R/compare/data
docker run -d --name lr-turbo-serve --gpus all -p 127.0.0.1:8195:8190 -v $H/weights:/models:ro -v $R/compare:/cfg \
    -v $H/bin:/lr:ro --entrypoint /lr/localrouter $NGC serve --config /cfg/tools.json --host all --data /cfg/data > $R/compare/container.id
for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8195/health > /dev/null && break; sleep 1; done
step "compare ($(left) s before cleanup)"
timeout $(left) nice python3 $REPO/tools/spark/compare_models.py --url http://127.0.0.1:8195 \
    --models qwen-image-2.1-turbo,qwen-image-2.1 --warm ${WARM:-3} --logs $R/compare/data/logs \
    --out $R/compare > $R/compare/compare.log 2>&1
step "compare rc $?"
docker logs lr-turbo-serve > $R/compare/serve.log 2>&1
step "done: $R"
