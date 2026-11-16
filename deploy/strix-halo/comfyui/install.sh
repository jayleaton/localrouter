#!/usr/bin/env bash
# Installs ComfyUI for the MiniMax tools of deploy/strix-halo, without sudo, idempotently (a step whose stamp matches
# its pins is skipped; downloads resume). Usage:
#   install.sh comfyui      ComfyUI and ComfyUI-GGUF at pinned commits, a Python venv with PyTorch ROCm, the GGUF loader node
#   install.sh models       the MiniMax H3 / H3-Image weights (pinned revisions) and their links into ComfyUI/models
#   install.sh ok-comfyui | ok-models    exit 0 when that step is done (the profile's "done" checks)
# Environment: COMFY_ROOT ($J/comfy), COMFY_DIR ($COMFY_ROOT/ComfyUI), COMFY_VENV ($COMFY_ROOT/venv),
# COMFY_HF ($COMFY_ROOT/hf: the downloaded weights; point it at an existing download to reuse it), UV (uv to use).
# The MiniMax weights' licences grant nothing in the US, the EU, the UK or South Korea: see deploy/strix-halo/README.md.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
J=${J:-$HOME/localrouter-amd}
COMFY_ROOT=${COMFY_ROOT:-$J/comfy}
COMFY_DIR=${COMFY_DIR:-$COMFY_ROOT/ComfyUI}
COMFY_VENV=${COMFY_VENV:-$COMFY_ROOT/venv}
COMFY_HF=${COMFY_HF:-$COMFY_ROOT/hf}
UV=${UV:-$( [ -x "$J/bin/uv" ] && echo "$J/bin/uv" || command -v uv || true)}

COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
COMFYUI_REV=e9afbccc2669677076feb08e52658f129cd8c12b        # 2026-11-15: has the MiniMax H3 / H3-Image nodes
GGUF_REPO=https://github.com/city96/ComfyUI-GGUF.git
GGUF_REV=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
PYTHON_VERSION=3.13
TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
TORCH_PINS="torch==2.14.1+rocm7.2 torchvision==0.29.1+rocm7.2 torchaudio==2.11.0+rocm7.2"
HF_HUB=huggingface_hub==2.4.0
# repo revision local-subdir files...
WEIGHTS=(
  "Comfy-Org/MiniMax-H3 4b53a35191437de05370da548e59aeabe2a5a09a Comfy-Org-MiniMax-H3
     diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
     vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors
     loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
  "unsloth/MiniMax-H3-Image-GGUF 2eda8f761a1625818e2933e7e726e337d197bc27 unsloth-MiniMax-H3-Image-GGUF
     minimax_h3_image_pruned-Q4_K_M.gguf"
  "Comfy-Org/MiniMax-H3-Image 0bb149f51e3156ed4b90dd45d5620346bfe1df8f Comfy-Org-MiniMax-H3-Image
     vae/minimax_h3_image_vae_fp16.safetensors"
)

say() { echo "comfyui-install: $*" >&2; }
die() { say "$*"; exit 1; }
need_uv() { [ -n "$UV" ] && [ -x "$UV" ] || die "uv not found (the profile's uv step installs it into \$J/bin; or set UV)"; }

checkout() {  # DIR REPO REV: a clone of REPO at REV (an existing clone is moved to REV; local changes stop it)
  local d=$1 repo=$2 rev=$3
  if [ ! -d "$d/.git" ]; then
    [ ! -e "$d" ] || die "$d exists and is not a git checkout"
    git clone -q --filter=blob:none "$repo" "$d"
  fi
  local dirty; dirty=$(git -C "$d" status --porcelain --untracked-files=no)
  [ -z "$dirty" ] || die "$d has local changes; not moving it to $rev"
  [ "$(git -C "$d" rev-parse HEAD)" = "$rev" ] && return 0
  git -C "$d" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$d" fetch -q origin "$rev"
  git -C "$d" -c advice.detachedHead=false checkout -q "$rev"
  say "$(basename "$d") at ${rev:0:12}"
}

venv_stamp() {  # what the venv was built from: the pins and both requirements files
  { echo "$PYTHON_VERSION $TORCH_INDEX $TORCH_PINS"; cat "$COMFY_DIR/requirements.txt" "$COMFY_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt"; } | sha256sum | cut -c1-64
}

ok_comfyui() {
  [ "$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null)" = "$COMFYUI_REV" ] &&
  [ "$(git -C "$COMFY_DIR/custom_nodes/ComfyUI-GGUF" rev-parse HEAD 2>/dev/null)" = "$GGUF_REV" ] &&
  cmp -s "$here/nodes/minimax_h3_image_gguf.py" "$COMFY_DIR/custom_nodes/minimax_h3_image_gguf.py" &&
  [ -x "$COMFY_VENV/bin/python" ] && [ "$(cat "$COMFY_VENV/.localrouter-stamp" 2>/dev/null)" = "$(venv_stamp)" ]
}

install_comfyui() {
  command -v git > /dev/null || die "git is required"
  mkdir -p "$COMFY_ROOT"
  checkout "$COMFY_DIR" "$COMFYUI_REPO" "$COMFYUI_REV"
  checkout "$COMFY_DIR/custom_nodes/ComfyUI-GGUF" "$GGUF_REPO" "$GGUF_REV"
  cmp -s "$here/nodes/minimax_h3_image_gguf.py" "$COMFY_DIR/custom_nodes/minimax_h3_image_gguf.py" ||
    install -m 644 "$here/nodes/minimax_h3_image_gguf.py" "$COMFY_DIR/custom_nodes/minimax_h3_image_gguf.py"
  local stamp; stamp=$(venv_stamp)
  if [ -x "$COMFY_VENV/bin/python" ] && [ "$(cat "$COMFY_VENV/.localrouter-stamp" 2>/dev/null)" = "$stamp" ]; then
    say "venv up to date ($COMFY_VENV)"
  else
    need_uv
    [ -x "$COMFY_VENV/bin/python" ] || "$UV" venv -q --python "$PYTHON_VERSION" "$COMFY_VENV"
    # PyTorch from its ROCm index first, then ComfyUI's requirements without torch (so pip keeps the ROCm build)
    # shellcheck disable=SC2086
    "$UV" pip install -q --python "$COMFY_VENV/bin/python" --index-url "$TORCH_INDEX" $TORCH_PINS
    local req; req=$(mktemp)
    grep -hvE '^(torch|torchvision|torchaudio)([<>=~ ;]|$)' "$COMFY_DIR/requirements.txt" "$COMFY_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt" > "$req"
    # shellcheck disable=SC2086
    "$UV" pip install -q --python "$COMFY_VENV/bin/python" -r "$req" $TORCH_PINS --index-url https://pypi.org/simple --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
    rm -f "$req"
    "$COMFY_VENV/bin/python" -c "import torch; assert torch.version.hip, 'not a ROCm build'; print('torch', torch.__version__, 'HIP', torch.version.hip, 'GPU', torch.cuda.is_available() and torch.cuda.get_device_name(0))" >&2
    echo "$stamp" > "$COMFY_VENV/.localrouter-stamp"
  fi
  ok_comfyui || die "ComfyUI is not complete; see above"
  say "ComfyUI ready: $COMFY_DIR (python $COMFY_VENV/bin/python)"
}

dest_dir() {  # the ComfyUI models folder for a weights file
  case "$1" in
    *.gguf) echo unet ;; diffusion_models/*) echo diffusion_models ;; text_encoders/*) echo text_encoders ;;
    vae/*) echo vae ;; loras/*) echo loras ;; *) die "no ComfyUI folder for $1" ;;
  esac
}

ok_models() {
  local e repo rev sub f
  for e in "${WEIGHTS[@]}"; do
    read -r repo rev sub files <<< "$(echo $e)"
    for f in $files; do
      [ -s "$COMFY_HF/$sub/$f" ] && [ ! -e "$COMFY_HF/$sub/$f.incomplete" ] || return 1
      [ "$(readlink -f "$COMFY_DIR/models/$(dest_dir "$f")/$(basename "$f")")" = "$(readlink -f "$COMFY_HF/$sub/$f")" ] || return 1
    done
  done
}

install_models() {
  [ -d "$COMFY_DIR/models" ] || die "run 'install.sh comfyui' first"
  need_uv
  local uvx; uvx="$(dirname "$UV")/uvx"; [ -x "$uvx" ] || uvx="$UV tool run"
  local e repo rev sub f
  for e in "${WEIGHTS[@]}"; do
    read -r repo rev sub files <<< "$(echo $e)"
    # shellcheck disable=SC2086
    $uvx --from "$HF_HUB" hf download "$repo" $files --revision "$rev" --local-dir "$COMFY_HF/$sub" > /dev/null
    for f in $files; do
      mkdir -p "$COMFY_DIR/models/$(dest_dir "$f")"
      ln -sfn "$COMFY_HF/$sub/$f" "$COMFY_DIR/models/$(dest_dir "$f")/$(basename "$f")"
    done
    say "$repo@${rev:0:12}: $(echo $files | wc -w) file(s) in $COMFY_HF/$sub"
  done
  ok_models || die "weights incomplete; run again (downloads resume)"
}

case "${1:-}" in
  comfyui) install_comfyui ;;
  models) install_models ;;
  ok-comfyui) ok_comfyui ;;
  ok-models) ok_models ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
