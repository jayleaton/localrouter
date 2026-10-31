#!/usr/bin/env bash
# Downloads what the macOS Qwen-Image worker runs (tools/image/qwen-mlx/pins-macos.json) into ROOT and checks it.
#   tools/image/qwen-mlx/fetch.sh ROOT [--pack q8|q4]
# ROOT/mlx-serve-macos-arm64/  the MLX-Serve release binary (its tarball's sha256 checked; quarantine cleared)
# ROOT/qwen-image-2.1-turbo-mlx-PACK/  the MLX-Serve pack, q8 (8-bit, 17.7 GB, pins-macos.json; the default) or q4
#                                      (4-bit, 10.5 GB, pins-macos-q4.json); every file's size and digest checked by
#                                      tools/package/verify_assets.py: the tool's `weights` (and its option `pack`)
# ROOT/venv/  Python for the worker, with Pillow from requirements-macos.txt (hashes required)
# Files already present with the pinned size are kept (and checked again). Apple Silicon macOS only. The model is under
# the Qwen Research License: non-commercial research or evaluation only.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
root=${1:?usage: fetch.sh ROOT [--pack q8|q4]}
pack=q8
[ "${2:-}" = --pack ] && pack=${3:?--pack q8 or q4}
case $pack in
  q8) packpins=$here/pins-macos.json ;;
  q4) packpins=$here/pins-macos-q4.json ;;
  *) echo "fetch.sh: --pack q8 or q4, not $pack" >&2; exit 2 ;;
esac
[ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || { echo "fetch.sh: macOS on Apple Silicon only" >&2; exit 1; }
mkdir -p "$root"
root=$(cd "$root" && pwd)
pins=$here/pins-macos.json
pin() { python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); [v := v[k] for k in sys.argv[2:]]; print(v)' "$@"; }   # FILE KEY...

# MLX-Serve
url=$(pin "$pins" mlx_serve url) digest=$(pin "$pins" mlx_serve sha256) dir=$(pin "$pins" mlx_serve dir) version=$(pin "$pins" mlx_serve version)
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
repo_id=$(pin "$packpins" repo_id) rev=$(pin "$packpins" revision)
weights=$root/qwen-image-2.1-turbo-mlx-$pack
python3 -I -c 'import json,sys; [print(f["path"], f["size_bytes"]) for f in json.load(open(sys.argv[1]))["selected_files"]]' "$packpins" |
while read -r path size; do
  dest=$weights/$path
  if [ -f "$dest" ] && [ "$(stat -f %z "$dest")" = "$size" ]; then continue; fi
  mkdir -p "$(dirname "$dest")"
  echo "fetch.sh: $repo_id/$path ($size bytes)"
  curl -fSL --retry 3 -C - -o "$dest.part" "https://huggingface.co/$repo_id/resolve/$rev/$path"
  mv "$dest.part" "$dest"
done
python3 -I "$repo/tools/package/verify_assets.py" "$packpins" "$weights"

# The worker's Python
if [ ! -x "$root/venv/bin/python" ]; then
  if command -v uv >/dev/null; then uv venv -q -p 3.12 "$root/venv"; else python3 -m venv "$root/venv"; fi
fi
if command -v uv >/dev/null; then
  VIRTUAL_ENV=$root/venv uv pip install -q --require-hashes -r "$here/requirements-macos.txt"
else
  "$root/venv/bin/python" -m pip install -q --require-hashes -r "$here/requirements-macos.txt"
fi
echo "fetch.sh: done. The tool: \"cmd\": [\"$root/venv/bin/python\", \"-I\", \"$here/qwen_mlx_worker.py\"], \"weights\": \"$weights\", \"options\": {\"pack\": \"$pack\"}"
