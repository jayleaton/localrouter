#!/usr/bin/env bash
# setup.sh for macOS on Apple Silicon: checks that the TensorFold binary fetch-macos.sh installed runs here, reports
# Metal, and sees the checkpoint, then writes and prints the daemon config for the qwen3.8-27b tool.
#   tools/chat/qwen38-27b/setup-macos.sh ROOT [--hf DIR] [--prompt-cache-gib 4|1]
# ROOT and DIR as for fetch-macos.sh (DIR default the shared Hugging Face cache). --prompt-cache-gib (default 4, the
# measured configuration) sets the server's --prompt-cache-gib and the matching resident_bytes: 4 GiB books 26.75 GiB
# (measured, docs/dev/CHAT.md); 1 GiB books 23.75 GiB (an estimate until bench_macos.py measures it) for a Mac that
# keeps chat, speech and transcription warm beside each other. Writes ROOT/tensorfold-version.txt,
# ROOT/tensorfold-capabilities.json, ROOT/tensorfold-models.txt, ROOT/tensorfold-info.txt and ROOT/config.json
# (config.macos.example.json with these paths). No root; touches nothing outside ROOT. Needs jq. The model only
# loads (and uses the GPU) in bench_macos.py, not here.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pins=$here/pins-macos.json
usage="usage: setup-macos.sh ROOT [--hf DIR] [--prompt-cache-gib 4|1]"
root=${1:?$usage}
shift
hf=""
cache_gib=4
while [ $# -gt 0 ]; do
  case $1 in
    --hf) hf=${2:?$usage}; shift 2 ;;
    --prompt-cache-gib) cache_gib=${2:?$usage}; shift 2 ;;
    *) echo "setup-macos.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "setup-macos.sh: $*" >&2; exit 1; }
[ "$(uname -s) $(uname -m)" = "Darwin arm64" ] || die "this is $(uname -s) $(uname -m); on Linux use setup.sh"
command -v jq >/dev/null || die "needs jq"
# The booking for each prompt cache size: the loaded footprint, the long-prompt working peak and the cache itself.
case $cache_gib in
  4) resident=28723101696 memory_note="" ;;  # 26.75 GiB, measured: 26.06 GiB peak on the live stack plus margin
  1) resident=25501876224  # 23.75 GiB, estimated: the 4 GiB booking less the 3 GiB of cache this size cannot hold
     memory_note="--prompt-cache-gib 1: 23.75 GiB is an estimate (the measured 4 GiB booking, 26.75 GiB, less 3 GiB of prompt cache); measure it with bench_macos.py serve before relying on it" ;;
  *) die "--prompt-cache-gib takes 4 or 1 (the sizes with a booking), not $cache_gib" ;;
esac
root=$(cd "$root" && pwd)
[ -n "$hf" ] || hf=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
hf=$(cd "$hf" && pwd)

tf=$root/$(jq -er '.tensorfold.dir' "$pins")
bin=$tf/$(jq -er '.tensorfold.binary' "$pins")
grammar=$tf/$(jq -er '.tensorfold.grammar_lib' "$pins")
version=$(jq -er '.tensorfold.version' "$pins")
repo=$(jq -er '.checkpoint.repo' "$pins")
[ -x "$bin" ] || die "$bin is missing or not executable: run fetch-macos.sh $root first"
[ -f "$tf.archive-sha256" ] || die "$tf was not installed by fetch-macos.sh (no $tf.archive-sha256)"
[ -f "$grammar" ] || die "$grammar is missing: tool_choice required/named and response_format need libtfgrammar"
echo "setup-macos.sh: $grammar present"

timeout 60 "$bin" --version > "$root/tensorfold-version.txt" 2>&1 ||
  { cat "$root/tensorfold-version.txt" >&2; die "$bin --version failed on this host"; }
got=$(cat "$root/tensorfold-version.txt")
case $got in
  *"$version"*) echo "setup-macos.sh: $bin --version: $got" ;;
  *) die "$bin --version says '$got', pinned $version" ;;
esac
timeout 60 "$bin" capabilities --json > "$root/tensorfold-capabilities.json.tmp" \
  2> "$root/tensorfold-capabilities.err" ||
  { cat "$root/tensorfold-capabilities.err" >&2; die "$bin capabilities --json failed"; }
jq -e . "$root/tensorfold-capabilities.json.tmp" > /dev/null || die "capabilities --json did not print JSON"
mv "$root/tensorfold-capabilities.json.tmp" "$root/tensorfold-capabilities.json"
rm -f "$root/tensorfold-capabilities.err"
# The Mac build has one GPU backend; anything else means the wrong archive or a host without Metal.
jq -e '.backends == ["metal"]' "$root/tensorfold-capabilities.json" > /dev/null ||
  die "capabilities lists backends $(jq -c .backends "$root/tensorfold-capabilities.json"), expected [\"metal\"]"
echo "setup-macos.sh: capabilities saved to $root/tensorfold-capabilities.json" \
  "(chip $(jq -r .chip "$root/tensorfold-capabilities.json"), backends metal)"

caps=$(cat "$root/tensorfold-capabilities.json")
flags=$(jq -r '.tools[0].options.server[] | select(startswith("--"))' "$here/config.macos.example.json")
missing=
while read -r flag; do
  grep -qF -- "\"$flag" <<< "$caps" || missing="$missing $flag"
done <<< "$flags"
[ -z "$missing" ] || die "capabilities --json does not list:$missing"

env HF_HUB_CACHE="$hf" HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1 timeout 300 "$bin" models \
  > "$root/tensorfold-models.txt" 2>&1 || { cat "$root/tensorfold-models.txt" >&2; die "tensorfold models failed"; }
models=$(cat "$root/tensorfold-models.txt")
grep -qF -- "$repo" <<< "$models" ||
  echo "setup-macos.sh: WARNING: 'models' does not list $repo (see $root/tensorfold-models.txt)"
env HF_HUB_CACHE="$hf" HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1 timeout 300 "$bin" info "$repo" \
  > "$root/tensorfold-info.txt" 2>&1 || { cat "$root/tensorfold-info.txt" >&2; die "tensorfold info $repo failed"; }
echo "setup-macos.sh: tensorfold info $repo:"
cat "$root/tensorfold-info.txt"

jq --arg hf "$hf" --arg bin "$bin" --arg gib "$cache_gib" --argjson resident "$resident" --arg note "$memory_note" \
  '.tools[0].weights = $hf | .tools[0].options.env.HF_HUB_CACHE = $hf | .tools[0].options.server[0] = $bin
   | .tools[0].resident_bytes = $resident
   | .tools[0].options.server |= (index("--prompt-cache-gib") as $i | .[$i + 1] = $gib)
   | if $note != "" then .tools[0].options.memory = $note else . end' \
  "$here/config.macos.example.json" > "$root/config.json.tmp"
mv "$root/config.json.tmp" "$root/config.json"
cat <<EOF
setup-macos.sh: wrote $root/config.json. Merge its "defaults" and "tools" entries into the daemon's config:
$(cat "$root/config.json")
setup-macos.sh: measure this host before serving it: tools/chat/qwen38-27b/bench_macos.py (see README.md)
EOF
