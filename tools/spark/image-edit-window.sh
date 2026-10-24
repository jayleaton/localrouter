#!/usr/bin/env bash
# Run in the granted GPU window on a machine with the model, packs, references and tools already local.
# Usage: image-edit-window.sh REF.png [REF2.png ... REF5.png]
# Required environment: MODEL, PACKS, COMFY. Optional: ZIG, VENV, ONLY, SKIP_PACK, SKIP_CAPTURE, RUN, SIZE, STEPS.
set -uo pipefail
# The checkout is this script's own (not $STK: the pod env exports STK as the shared data root, which lacks this twin).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO" || exit 1
: "${MODEL:?local model snapshot required}" "${PACKS:?existing DiT/TE/VAE packs required}" "${COMFY:?local ComfyUI reference required}"
if (($#)); then IMAGES=("$@"); elif [[ -n ${IMAGE:-} ]]; then IMAGES=("$IMAGE"); else
    echo "provide one to five positional PNG references (or set IMAGE)" >&2; exit 2
fi
if ((${#IMAGES[@]} < 1 || ${#IMAGES[@]} > 5)); then echo "provide one to five PNG references" >&2; exit 2; fi
export MODEL PACKS COMFY
[[ -z ${VENV:-} ]] || source "$VENV/bin/activate"
export PYTHONPATH="$REPO/tools/twin${PYTHONPATH:+:$PYTHONPATH}"
export STK_KROOT="$REPO/kernels"   # this checkout's kernels (the pod env points STK_KROOT at the shared copy)
export STK_OPS=stk STK_ATTN=stk STK_TE=stk STK_VAE=stk
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda
ZIG=${ZIG:-$HOME/.local/opt/zig/zig}
R=${RUN:-$REPO/results/image-edit-$(date -u +%Y%m%d-%H%M%S)}
C=${CAPTURE:-$R/capture}
PRECISION=${PRECISION:-nvfp4}
SIZE=${SIZE:-512x512}
STEPS=${STEPS:-4}
ONLY=${ONLY:-}
mkdir -p "$R"
FAILED=0
IMAGE_ARGS=()
for image in "${IMAGES[@]}"; do IMAGE_ARGS+=(--image "$image"); done
sel() { [[ -z $ONLY || " $ONLY " == *" $1 "* ]]; }
run() {
    local name=$1; shift
    sel "$name" || return 0
    echo "[image-edit $(date +%T)] $name"
    local start=$(date +%s) rc
    "$@" > "$R/$name.log" 2>&1; rc=$?
    tail -n 1 "$R/$name.log" | grep '^{' >> "$R/summary.jsonl" || true
    echo "{\"step\":\"$name\",\"rc\":$rc,\"s\":$(( $(date +%s) - start ))}" >> "$R/steps.jsonl"
    [[ $rc == 0 ]] || FAILED=1
    return "$rc"
}
( while :; do awk -v t="$(date +%s)" '/MemAvailable/ {print t, $2}' /proc/meminfo; sleep 1; done ) > "$R/memavail.log" &
MEM=$!
finish() {
    kill "$MEM" 2>/dev/null || true
    wait "$MEM" 2>/dev/null || true
    awk 'NR==1 {a=$2} m=="" || $2<m {m=$2} END {printf "{\"memavail_start_gib\":%.1f,\"peak_used_gib\":%.1f}\n",a/1048576,(a-m)/1048576}' "$R/memavail.log" >> "$R/summary.jsonl"
}
trap finish EXIT

run setup python - "$MODEL" "$PACKS" "$COMFY" "$ZIG" "$PRECISION" "${IMAGES[@]}" <<'PY'
import json, os, pathlib, sys
import torch, transformers, diffusers, PIL, numpy, safetensors
model, packs, comfy, zig = map(pathlib.Path, sys.argv[1:5])
precision, *images = sys.argv[5:]
assert torch.cuda.is_available(), "GPU window required"
for path, label in ((comfy, "COMFY"), (zig, "ZIG")):
    assert path.exists() and (path.is_file() if label == "ZIG" else path.is_dir()), f"missing local {label}: {path}"
assert os.access(zig, os.X_OK), f"ZIG is not executable: {zig}"
assert pathlib.Path("/usr/local/cuda/bin/nvcc").is_file(), "missing local CUDA compiler"
assert (model / "model_index.json").is_file(), f"missing local model snapshot: {model}"
for rel in ("text_encoder/config.json", "vae/config.json", "transformer/config.json"):
    assert (model / rel).is_file(), f"missing model file: {rel}"
for rel in ("te/manifest.json", "vae/manifest.json", f"{precision}/manifest.json"):
    assert (packs / rel).is_file(), f"missing local pack manifest: {rel}"
assert any((model / name / "tokenizer.json").is_file() for name in ("processor", "tokenizer")), "missing local tokenizer"
for path in map(pathlib.Path, images):
    assert path.is_file(), f"missing local PNG: {path}"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
print(json.dumps({"torch":torch.__version__, "cuda":torch.version.cuda,
                  "sm":torch.cuda.get_device_capability(), "references":len(images)}))
PY
[[ $? == 0 ]] || exit 1

run test-edit python -m stk_twin.test_edit --comfy "$COMFY" --gpu
run test-vision python -m stk_twin.test_vision --gpu --model "$MODEL/text_encoder"
run test-vae-encoder env MODEL="$MODEL" python -m stk_twin.test_vae_enc 256 256
if [[ ${SKIP_PACK:-0} != 1 ]]; then
    run pack-edit python -m stk_twin pack --model "$MODEL" --precision edit --out "$PACKS"
fi
run verify-edit-pack python - "$PACKS" "${#IMAGES[@]}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
marker = json.loads((root / "edit.json").read_text())
assert marker.get("format") == "stk-edit/1" and marker.get("refs_max", 0) >= int(sys.argv[2]), "edit pack marker missing or invalid"
for name in ("te", "vae"):
    digest = hashlib.sha256((root / name / "manifest.json").read_bytes()).hexdigest()
    assert marker.get("manifest_sha256", {}).get(name) == digest, f"edit pack marker is stale for {name}"
print(json.dumps({"edit_pack": "valid", "refs_max": marker["refs_max"]}))
PY
[[ $? == 0 ]] || exit 1
if [[ ${SKIP_CAPTURE:-0} != 1 ]]; then
    run capture python -m stk_twin.capture_edit --model "$MODEL" --pack "$PACKS/$PRECISION" "${IMAGE_ARGS[@]}" \
        --prompt "${PROMPT:-make the scene a snowy winter evening}" --size "$SIZE" --steps "$STEPS" --seed "${SEED:-7}" --out "$C"
fi
run reference-gate python -m stk_twin.edit_gate --model "$MODEL" "${IMAGE_ARGS[@]}" --prompt "${PROMPT:-make the scene a snowy winter evening}" --size "$SIZE" --steps "$STEPS" --seed "${SEED:-7}" --out "$R/reference-gate"
run zig-build "$ZIG" build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 || exit 1
run qwen-edit "$REPO/zig-out/bin/localrouter" check qwen-edit "$PACKS/$PRECISION" "$PACKS/te" "$PACKS/vae" "$C"
exit "$FAILED"
