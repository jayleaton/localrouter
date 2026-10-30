#!/usr/bin/env bash
# Downloads what the macOS Qwen-Image worker runs (tools/image/qwen-mlx/pins-macos.json) into ROOT and checks it.
#   tools/image/qwen-mlx/fetch.sh ROOT
# ROOT/mlx-serve-macos-arm64/  the MLX-Serve release binary (its tarball's sha256 checked; quarantine cleared)
# ROOT/qwen-image-2.1-turbo-mlx-q8/  the 8-bit MLX-Serve pack (17.7 GB; every file's size and digest checked by
#                                    tools/package/verify_assets.py): the tool's `weights`
# ROOT/venv/  Python for the worker, with Pillow from requirements-macos.txt (hashes required)
# Files already present with the pinned size are kept (and checked again). Apple Silicon macOS only. The model is under
# the Qwen Research License: non-commercial research or evaluation only.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
root=${1:?usage: fetch.sh ROOT}
[ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || { echo "fetch.sh: macOS on Apple Silicon only" >&2; exit 1; }
mkdir -p "$root"
root=$(cd "$root" && pwd)
pins=$here/pins-macos.json
pin() { python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2:]]; print(v)' "$pins" "$@"; }

# MLX-Serve
url=$(pin mlx_serve url) digest=$(pin mlx_serve sha256) dir=$(pin mlx_serve dir) version=$(pin mlx_serve version)
if ! "$root/$dir/mlx-serve" --version 2>/dev/null | grep -q "^mlx-serve $version\$"; then
  tmp="$root/.mlx-serve.tar.gz.part"
  curl -fsSL --retry 3 -o "$tmp" "$url"
  got=$(shasum -a 256 "$tmp" | cut -d' ' -f1)
  if [ "$got" != "$digest" ]; then
    rm -f "$tmp"
    echo "fetch.sh: $url sha256 $got, pinned $digest" >&2
    exit 1
  fi
  rm -rf "$root/$dir"
  tar -xzf "$tmp" -C "$root"
  rm -f "$tmp"
  xattr -dr com.apple.quarantine "$root/$dir" 2>/dev/null || true
fi
echo "fetch.sh: $("$root/$dir/mlx-serve" --version 2>/dev/null | grep '^mlx-serve ') in $root/$dir"

# The pack: each pinned file from its revision, then every digest
repo_id=$(pin repo_id) rev=$(pin revision)
weights=$root/qwen-image-2.1-turbo-mlx-q8
python3 -I -c 'import json,sys; [print(f["path"], f["size_bytes"]) for f in json.load(open(sys.argv[1]))["selected_files"]]' "$pins" |
while read -r path size; do
  dest=$weights/$path
  if [ -f "$dest" ] && [ "$(stat -f %z "$dest")" = "$size" ]; then continue; fi
  mkdir -p "$(dirname "$dest")"
  echo "fetch.sh: $repo_id/$path ($size bytes)"
  curl -fSL --retry 3 -C - -o "$dest.part" "https://huggingface.co/$repo_id/resolve/$rev/$path"
  mv "$dest.part" "$dest"
done
python3 -I "$repo/tools/package/verify_assets.py" "$pins" "$weights"

# The worker's Python
if [ ! -x "$root/venv/bin/python" ]; then
  if command -v uv >/dev/null; then uv venv -q -p 3.12 "$root/venv"; else python3 -m venv "$root/venv"; fi
fi
if command -v uv >/dev/null; then
  VIRTUAL_ENV=$root/venv uv pip install -q --require-hashes -r "$here/requirements-macos.txt"
else
  "$root/venv/bin/python" -m pip install -q --require-hashes -r "$here/requirements-macos.txt"
fi
echo "fetch.sh: done. The tool: \"cmd\": [\"$root/venv/bin/python\", \"-I\", \"$here/qwen_mlx_worker.py\"], \"weights\": \"$weights\""
