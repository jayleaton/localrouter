#!/usr/bin/env bash
# Downloads the Wan 2.2 TI2V-5B worker's weights (tools/video/wan/pins.json) into WEIGHTS and checks every file's digest.
#   tools/video/wan/fetch.sh WEIGHTS [--venv VENV]
# WEIGHTS/            Wan-AI/Wan2.2-TI2V-5B-Diffusers (Apache-2.0) in Diffusers' layout: model_index.json, scheduler/,
#                     text_encoder/ (UMT5-XXL, bf16), tokenizer/, transformer/ (5B, fp32 as published), vae/; 34.2 GB
# --venv VENV         also builds the worker's Python there from the lock (uv sync --locked; uv must be on PATH)
# The repo is ungated and fetched at its pinned revision without a token. Files are fetched into WEIGHTS.staging, and
# WEIGHTS appears, or replaces the previous snapshot, only once every digest there matches; a failed check leaves
# WEIGHTS as it was and the file in WEIGHTS.staging to inspect (the next run fetches it anew). A file is kept (from
# WEIGHTS.staging, or linked from WEIGHTS) only when its size and digest are the pinned ones; a partial download
# resumes. WEIGHTS holds only the pinned files. Each run publishes through its own WEIGHTS.publish.XXXXXX and removes
# only that.
# x86_64 Linux and WSL2.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dest=${1:?usage: fetch.sh WEIGHTS [--venv VENV]}
venv=
if [ "${2:-}" = --venv ]; then venv=${3:?--venv needs a directory}; fi
pins=$here/pins.json
mkdir -p "$(dirname "$dest")"
dest=$(cd "$(dirname "$dest")" && pwd)/$(basename "$dest")
stage=$dest.staging
if [ ! -e "$dest" ]; then  # a publish cut short between its two moves: the previous snapshot back
  for old in "$dest".publish.*/old; do if [ -d "$old" ]; then mv "$old" "$dest"; break; fi; done
fi
if [ -d "$dest" ] && [ -n "$(ls -A "$dest")" ] && [ ! -f "$dest/model_index.json" ]; then
  echo "fetch.sh: $dest is not empty and holds no model_index.json: give a new directory" >&2
  exit 1
fi
mkdir -p "$stage"

size_of() { wc -c <"$1" | tr -d ' '; }
sha_of() { sha256sum "$1" | cut -d ' ' -f 1; }
pinned() { [ -f "$1" ] && [ "$(size_of "$1")" = "$2" ] && [ "$(sha_of "$1")" = "$3" ]; }

manifest=$(python3 -I -c '
import json, sys
m = json.load(open(sys.argv[1]))
for f in m["selected_files"]:
    s = m["sources"][f["source"]]
    print(f["path"], f["size_bytes"], f["sha256"], s["repo"], s["revision"], f["src"])
' "$pins")
while read -r path size sha hf_repo rev src; do
  out=$stage/$path
  if pinned "$out" "$size" "$sha"; then continue; fi
  rm -f "$out"
  mkdir -p "$(dirname "$out")"
  prev=$dest/$path
  if pinned "$prev" "$size" "$sha"; then
    ln -f "$prev" "$out" 2>/dev/null || cp "$prev" "$out"
    continue
  fi
  echo "fetch.sh: $hf_repo@${rev:0:7} $src ($size bytes)"
  curl -fSL --retry 3 -C - -o "$out.part" "https://huggingface.co/$hf_repo/resolve/$rev/$src"
  got=$(size_of "$out.part")
  if [ "$got" != "$size" ]; then
    echo "fetch.sh: $out.part is $got bytes, pinned $size: delete it and run again" >&2
    exit 1
  fi
  mv "$out.part" "$out"
  if [ "$(sha_of "$out")" != "$sha" ]; then
    echo "fetch.sh: $out does not match its sha256 in pins.json; kept to inspect, fetched anew next run ($dest is unchanged)" >&2
    exit 1
  fi
done <<<"$manifest"

# the published snapshot holds the manifest's files only: anything else in staging (an older pin's) is dropped
pub=$(mktemp -d "$dest.publish.XXXXXX")
trap 'if [ ! -e "$dest" ] && [ -d "$pub/old" ]; then mv "$pub/old" "$dest"; fi; rm -rf "$pub"' EXIT
while read -r path _; do
  mkdir -p "$(dirname "$pub/new/$path")"
  ln "$stage/$path" "$pub/new/$path" 2>/dev/null || mv "$stage/$path" "$pub/new/$path"
done <<<"$manifest"
if [ -d "$dest" ]; then mv "$dest" "$pub/old"; fi
mv "$pub/new" "$dest"
rm -rf "$stage"

if [ -n "$venv" ]; then
  command -v uv >/dev/null || { echo "fetch.sh: --venv needs uv on PATH" >&2; exit 1; }
  UV_PROJECT_ENVIRONMENT=$venv uv sync --locked --python 3.12 --project "$here"
  echo "fetch.sh: the tool: \"cmd\": [\"$venv/bin/python\", \"-I\", \"$here/worker.py\"], \"weights\": \"$dest\""
else
  echo "fetch.sh: done; \"weights\": \"$dest\""
fi
