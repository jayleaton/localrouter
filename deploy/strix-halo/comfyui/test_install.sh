#!/usr/bin/env bash
# install.sh models: downloads through hf_pinned.py, refuses bytes that differ from the pins, links into ComfyUI/models,
# and ok-models holds only after a full check. A stub hf writes files from a source tree; no network, no GPU.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
pass=0 fail=0
check() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL: $1" >&2; fi; }

mkdir -p "$t/bin" "$t/pins" "$t/src" "$t/ComfyUI/models"
# the stub: uvx --from PKG hf download REPO FILES... --revision REV --local-dir DIR, copying from $t/src/REPO
cat > "$t/bin/uvx" <<'STUB'
#!/usr/bin/env bash
echo call >> "$STUB_LOG"
shift 2; [ "$1 $2" = "hf download" ] || exit 9; shift 2
repo=$1; shift; files=()
while [ "$1" != --revision ]; do files+=("$1"); shift; done
dir=$4
for f in "${files[@]}"; do mkdir -p "$dir/$(dirname "$f")"; cp "$STUB_SRC/$repo/$f" "$dir/$f"; done
STUB
printf '#!/bin/sh\nexit 0\n' > "$t/bin/uv"; chmod +x "$t/bin/uv" "$t/bin/uvx"

manifest() {  # NAME REPO FILE...: a manifest of the source files as they are now
  local name=$1 repo=$2; shift 2
  python3 - "$t/src/$repo" "$repo" "$@" > "$t/pins/$name" <<'PY'
import hashlib, json, os, sys
root, repo, files = sys.argv[1], sys.argv[2], sys.argv[3:]
print(json.dumps({"repo": repo, "revision": "0" * 40, "files": {f: [os.path.getsize(os.path.join(root, f)),
      hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()] for f in files}}))
PY
}
put() { mkdir -p "$t/src/$1/$(dirname "$2")"; printf '%s' "$3" > "$t/src/$1/$2"; }
put a/h3 diffusion_models/dit.safetensors dit; put a/h3 vae/video.safetensors video; put a/h3 loras/turbo.safetensors lora
put b/gguf q4.gguf q4q4; put c/vae vae/image.safetensors image
manifest minimax-h3.json a/h3 diffusion_models/dit.safetensors vae/video.safetensors loras/turbo.safetensors
manifest minimax-h3-image-gguf.json b/gguf q4.gguf
manifest minimax-h3-image-vae.json c/vae vae/image.safetensors

export UV="$t/bin/uv" COMFY_PINS="$t/pins" COMFY_DIR="$t/ComfyUI" COMFY_HF="$t/hf" STUB_SRC="$t/src" STUB_LOG="$t/calls"
run() { bash "$here/install.sh" "$@" > "$t/out" 2>&1; }

run ok-models; check "ok-models fails before any download" '[ $? -ne 0 ]'
run models; check "models succeeds on matching bytes" '[ $? -eq 0 ]'
check "three downloads" '[ "$(wc -l < "$t/calls")" = 3 ]'
check "each manifest stamped" '[ "$(ls "$t"/hf/*/.localrouter-verified | wc -l)" = 3 ]'
check "dit linked into diffusion_models" '[ "$(readlink -f "$t/ComfyUI/models/diffusion_models/dit.safetensors")" = "$(readlink -f "$t/hf/Comfy-Org-MiniMax-H3/diffusion_models/dit.safetensors")" ]'
check "gguf linked into unet" '[ -L "$t/ComfyUI/models/unet/q4.gguf" ]'
check "vae and lora linked" '[ -L "$t/ComfyUI/models/vae/video.safetensors" ] && [ -L "$t/ComfyUI/models/vae/image.safetensors" ] && [ -L "$t/ComfyUI/models/loras/turbo.safetensors" ]'
run ok-models; check "ok-models after" '[ $? -eq 0 ]'
run models; check "a second run downloads nothing" '[ $? -eq 0 ] && [ "$(wc -l < "$t/calls")" = 3 ] && grep -q "verified before" "$t/out"'

# a truncated file: ok-models fails, models downloads again and passes
printf 'q4' > "$t/hf/unsloth-MiniMax-H3-Image-GGUF/q4.gguf"
run ok-models; check "ok-models fails on a wrong size" '[ $? -ne 0 ]'
run models; check "models repairs it" '[ $? -eq 0 ] && [ "$(cat "$t/hf/unsloth-MiniMax-H3-Image-GGUF/q4.gguf")" = q4q4 ]'

# a moved pin: the stamp no longer matches
manifest minimax-h3-image-vae.json c/vae vae/image.safetensors; python3 -c "import json,sys;p=sys.argv[1];m=json.load(open(p));m['revision']='1'*40;json.dump(m,open(p,'w'))" "$t/pins/minimax-h3-image-vae.json"
run ok-models; check "ok-models fails when a manifest changes" '[ $? -ne 0 ]'
run models; check "models checks it again" '[ $? -eq 0 ]'

# bytes that differ from the pin (same size): refused, no stamp, not done
rm -rf "$t/hf/Comfy-Org-MiniMax-H3-Image"; put c/vae vae/image.safetensors IMAGE
run models; check "models refuses a sha256 mismatch" '[ $? -ne 0 ] && grep -q "sha256 differs" "$t/out"'
check "no stamp for the refused manifest" '[ ! -e "$t/hf/Comfy-Org-MiniMax-H3-Image/.localrouter-verified" ]'
run ok-models; check "ok-models fails after a refusal" '[ $? -ne 0 ]'

echo "comfyui install.sh: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
