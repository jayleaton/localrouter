#!/usr/bin/env bash
# Installs ComfyUI for the MiniMax tools of deploy/strix-halo, without sudo, idempotently (a step whose stamp matches
# its pins is skipped; downloads resume). Usage:
#   install.sh comfyui      ComfyUI and ComfyUI-GGUF at pinned commits, a Python venv with PyTorch ROCm, the GGUF loader node
#   install.sh models       the MiniMax H3 / H3-Image weights (pinned revisions, every file's size and sha256 checked)
#                           and their links into ComfyUI/models
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
REQUIREMENTS=$here/requirements-rocm7.2.txt   # the whole venv (PyTorch ROCm included), every package pinned by sha256
HF_HUB=huggingface_hub==2.4.0
HF_PINNED=(python3 -I "$here/../../../tools/install/hf_pinned.py")
PINS=${COMFY_PINS:-$here/../pins}                # (COMFY_PINS: for the tests)
# manifest (repo, revision, every file's size and sha256) and local subdirectory of $COMFY_HF
WEIGHTS=(
  "minimax-h3.json Comfy-Org-MiniMax-H3"
  "minimax-h3-image-gguf.json unsloth-MiniMax-H3-Image-GGUF"
  "minimax-h3-image-vae.json Comfy-Org-MiniMax-H3-Image"
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

venv_stamp() {  # what the venv was built from: the Python version, the index and the hashed requirements
  { echo "$PYTHON_VERSION $TORCH_INDEX"; cat "$REQUIREMENTS"; } | sha256sum | cut -c1-64
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
    # every package, PyTorch ROCm included, from the hashed lock: a download whose sha256 differs is refused
    "$UV" pip install -q --python "$COMFY_VENV/bin/python" --require-hashes -r "$REQUIREMENTS" \
      --index-url https://pypi.org/simple --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
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

manifest_stamp() { sha256sum < "$PINS/$1" | cut -c1-64; }  # written once a full sha256 check of that manifest passed

ok_models() {  # each manifest fully verified once (its stamp), sizes still right, links in place
  local e m sub f
  for e in "${WEIGHTS[@]}"; do
    read -r m sub <<< "$e"
    [ "$(cat "$COMFY_HF/$sub/.localrouter-verified" 2>/dev/null)" = "$(manifest_stamp "$m")" ] || return 1
    "${HF_PINNED[@]}" verify "$PINS/$m" "$COMFY_HF/$sub" --sizes > /dev/null || return 1
    while read -r f; do
      [ "$(readlink -f "$COMFY_DIR/models/$(dest_dir "$f")/$(basename "$f")")" = "$(readlink -f "$COMFY_HF/$sub/$f")" ] || return 1
    done < <("${HF_PINNED[@]}" files "$PINS/$m")
  done
}

install_models() {
  [ -d "$COMFY_DIR/models" ] || die "run 'install.sh comfyui' first"
  need_uv
  local uvx; uvx="$(dirname "$UV")/uvx"; [ -x "$uvx" ] || uvx="$UV tool run"
  local e m sub f n
  for e in "${WEIGHTS[@]}"; do
    read -r m sub <<< "$e"
    if [ "$(cat "$COMFY_HF/$sub/.localrouter-verified" 2>/dev/null)" = "$(manifest_stamp "$m")" ] &&
       "${HF_PINNED[@]}" verify "$PINS/$m" "$COMFY_HF/$sub" --sizes > /dev/null; then
      say "$m: verified before"
    else
      # downloads (hf resumes) at the manifest's revision, then checks every file's size and sha256; refuses on a mismatch
      rm -f "$COMFY_HF/$sub/.localrouter-verified"
      "${HF_PINNED[@]}" download "$PINS/$m" "$COMFY_HF/$sub" --hf "$uvx --from $HF_HUB hf" > /dev/null ||
        die "$m: download or sha256 check failed (above); run again"
      manifest_stamp "$m" > "$COMFY_HF/$sub/.localrouter-verified"
    fi
    n=0
    while read -r f; do
      mkdir -p "$COMFY_DIR/models/$(dest_dir "$f")"
      ln -sfn "$COMFY_HF/$sub/$f" "$COMFY_DIR/models/$(dest_dir "$f")/$(basename "$f")"
      n=$((n + 1))
    done < <("${HF_PINNED[@]}" files "$PINS/$m")
    say "$m: $n file(s) in $COMFY_HF/$sub, sha256 checked"
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
