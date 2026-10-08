#!/usr/bin/env bash
# MiniMax H3 (video + audio) on a DGX Spark (GB10, sm_121): the Spark's V3 (tools/twin/pod/v3.sh), in one short window.
# Run from a checkout of the branch under test, on the Spark, only in a window the owner granted:
#   bash tools/spark/h3-window.sh
# Copies the tree into $LOCALROUTER_HOME (repo, twin, kernels; like install.sh), opens the window (WINDOW_OPEN, the command that
# takes the GPU from another service; WINDOW_CLOSE restores it and runs last whatever happens), then runs
# tools/spark/h3-inner.sh in the NGC PyTorch container with $LOCALROUTER_HOME at /workspace/localrouter: the video venv and weights, the
# twin's GPU tests, the Turbo-LoRA pack, the captures, the zig build, `localrouter check h3-te|h3-replay|h3-avae|h3-vvae|h3-e2e`.
# Steps go on if one fails. Weights (about 61 GB), the pack, the venv and the zig build persist in $LOCALROUTER_HOME: a re-run
# skips what is done. Captures are deleted after their check (disk), logs and summaries are kept.
# Env (all optional): LOCALROUTER_HOME ($HOME/localrouter), WINDOW_OPEN, WINDOW_CLOSE, HF_TOKEN (passed in), and for re-runs after a fix,
# passed to h3-inner.sh: SKIP_SETUP=1, SKIP_TWIN_TESTS=1, SKIP_PACK=1 (reuse the pack), SKIP_CAPTURE=1 (reuse the captures,
# kept), ONLY="h3-te h3-replay" (only these steps; captures kept), KEEP_CAPTURES=1, KITCHEN_SRC_BUILD=1 (see h3-inner.sh).
# Output: $LOCALROUTER_HOME/results/h3-<time>/ (steps.jsonl, summary.jsonl, one .log per step, memavail.log, window.log);
# $LOCALROUTER_HOME/results/h3-latest points at it.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOCALROUTER_HOME=${LOCALROUTER_HOME:-$HOME/localrouter}
NGC=nvcr.io/nvidia/pytorch:26.07-py3
RUN=h3-$(date -u +%Y%m%d-%H%M%S); R=$LOCALROUTER_HOME/results/$RUN
step() { echo "[h3 $(date +%T)] $*" | tee -a "$R/window.log"; }
die() { echo "h3-window: $*" >&2; exit 1; }
opened=0
close() { [[ $opened == 0 || -z ${WINDOW_CLOSE:-} ]] || { step "close: $WINDOW_CLOSE"; bash -c "$WINDOW_CLOSE" >> "$R/window.log" 2>&1; }; }
trap close EXIT
trap 'exit 130' INT TERM

# ---- the machine and the tree, before the window
command -v nvidia-smi > /dev/null || die "no nvidia-smi"
command -v docker > /dev/null || die "no docker"
mkdir -p "$LOCALROUTER_HOME" 2> /dev/null || sudo install -d -o "$(id -u)" -g "$(id -g)" "$LOCALROUTER_HOME"
mkdir -p "$R" "$LOCALROUTER_HOME/repo" "$LOCALROUTER_HOME/twin" "$LOCALROUTER_HOME/kernels"
ln -sfn "$RUN" "$LOCALROUTER_HOME/results/h3-latest"
free_gb=$(df -BG --output=avail "$LOCALROUTER_HOME" | tail -1 | tr -dc 0-9)
[[ -f $LOCALROUTER_HOME/models/minimax-h3/.complete ]] || (( free_gb >= 150 )) ||
    die "$LOCALROUTER_HOME has ${free_gb} GB free; the first run needs about 150 GB (weights 61 GB, pack, captures)"
# a tree mid-merge does not build: stop before the window
if grep -rlE '^(<<<<<<<|>>>>>>>) ' --include='*.zig' --include='*.py' --include='*.sh' "$REPO/src" "$REPO/tools" "$REPO/kernels" 2> /dev/null | grep .; then
    die "merge conflict markers in the files above"
fi
step "tree -> $LOCALROUTER_HOME"
for d in tools/twin:twin kernels:kernels .:repo; do
    src=${d%%:*} dst=${d##*:}
    tar -C "$REPO/$src" --exclude=.git --exclude=.zig-cache --exclude=zig-out --exclude=zig-pkg --exclude=results --exclude=__pycache__ \
        -cf - . | tar -C "$LOCALROUTER_HOME/$dst" -xf -
done

# ---- the window
others=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | grep -v '^$' || true)
[[ -z $others ]] || echo "h3-window: warning: GPU processes before the window:"$'\n'"$others" | tee -a "$R/window.log" >&2
[[ -z ${WINDOW_OPEN:-} ]] || { step "open: $WINDOW_OPEN"; opened=1; bash -c "$WINDOW_OPEN" >> "$R/window.log" 2>&1; }

envs=(-e "H3_RUN=$RUN")
for v in SKIP_SETUP SKIP_TWIN_TESTS SKIP_PACK SKIP_CAPTURE ONLY KEEP_CAPTURES KITCHEN_SRC_BUILD HF_TOKEN \
         H3_SPEED_VAE H3_SPEED_DIT H3_SPEED_MEM STOP_ON_BUILD_FAIL WITH_COMFY; do
    [[ -z ${!v:-} ]] || envs+=(-e "$v")
done
step "container ($NGC)"
docker run --rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 "${envs[@]}" \
    -v "$LOCALROUTER_HOME":/workspace/localrouter "$NGC" bash /workspace/localrouter/repo/tools/spark/h3-inner.sh 2>&1 | tee -a "$R/window.log"
rc=${PIPESTATUS[0]}

# ---- what happened
step "container exit $rc; results: $R"
if [[ -f $R/steps.jsonl ]]; then
    python3 - "$R/steps.jsonl" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    s = json.loads(l)
    print(f"  {'ok  ' if s['rc'] == 0 else 'FAIL'} {s['step']:<16} rc={s['rc']:<3} {s['s']} s")
PY
fi
tail -n 1 "$R/summary.jsonl" 2> /dev/null | grep -F peak_used_gib
fails=$(grep -vc '"rc": 0,' "$R/steps.jsonl" 2> /dev/null)
[[ $rc == 0 && ${fails:-1} == 0 ]]
