#!/usr/bin/env bash
# Checks that the TensorFold binary fetch.sh installed runs on this host and sees the checkpoint, then writes and
# prints the daemon config for the qwen3.8-27b tool.
#   tools/chat/qwen38-27b/setup.sh ROOT [--hf DIR] [--marker PATH] [--gpu-lock PATH]
# ROOT and DIR as for fetch.sh (DIR default ROOT/hf). PATH (the tool's "qualification" marker) defaults to
# DIR/../qwen38-27b-qualification.json. --gpu-lock "" leaves the host lock out (default: the example's).
# Writes ROOT/tensorfold-version.txt, ROOT/tensorfold-capabilities.json, ROOT/tensorfold-models.txt,
# ROOT/tensorfold-info.txt and ROOT/config.json (config.example.json with these paths). No root; touches nothing
# outside ROOT. Needs jq. The model only loads (and uses the GPU) in qualify.sh, not here.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pins=$here/pins.json
usage="usage: setup.sh ROOT [--hf DIR] [--marker PATH] [--gpu-lock PATH]"
root=${1:?$usage}
shift
hf= marker= lock_set=0 lock=
while [ $# -gt 0 ]; do
  case $1 in
    --hf) hf=${2:?$usage}; shift 2 ;;
    --marker) marker=${2:?$usage}; shift 2 ;;
    --gpu-lock) [ $# -ge 2 ] || { echo "$usage" >&2; exit 2; }; lock=$2 lock_set=1; shift 2 ;;
    *) echo "setup.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "setup.sh: $*" >&2; exit 1; }
command -v jq >/dev/null || die "needs jq"
root=$(cd "$root" && pwd)
[ -n "$hf" ] || hf=$root/hf
hf=$(cd "$hf" && pwd)
[ -n "$marker" ] || marker=$(dirname "$hf")/qwen38-27b-qualification.json

tf=$root/$(jq -er '.tensorfold.dir' "$pins")
bin=$tf/$(jq -er '.tensorfold.binary' "$pins")
grammar=$tf/$(jq -er '.tensorfold.grammar_lib' "$pins")
version=$(jq -er '.tensorfold.version' "$pins")
repo=$(jq -er '.checkpoint.repo' "$pins")
[ -x "$bin" ] || die "$bin is missing or not executable: run fetch.sh $root first"
[ -f "$tf.archive-sha256" ] || die "$tf was not installed by fetch.sh (no $tf.archive-sha256)"
[ -f "$grammar" ] || die "$grammar is missing: tool_choice required/named and response_format need libtfgrammar"
echo "setup.sh: $grammar present"

# The binary runs here (glibc, libcuda.so.1 from the driver) and is the pinned version.
timeout 60 "$bin" --version > "$root/tensorfold-version.txt" 2>&1 ||
  { cat "$root/tensorfold-version.txt" >&2; die "$bin --version failed on this host"; }
got=$(cat "$root/tensorfold-version.txt")
case $got in
  *"$version"*) echo "setup.sh: $bin --version: $got" ;;
  *) die "$bin --version says '$got', pinned $version" ;;
esac
timeout 60 "$bin" capabilities --json > "$root/tensorfold-capabilities.json.tmp" \
  2> "$root/tensorfold-capabilities.err" ||
  { cat "$root/tensorfold-capabilities.err" >&2; die "$bin capabilities --json failed"; }
jq -e . "$root/tensorfold-capabilities.json.tmp" > /dev/null || die "capabilities --json did not print JSON"
mv "$root/tensorfold-capabilities.json.tmp" "$root/tensorfold-capabilities.json"
rm -f "$root/tensorfold-capabilities.err"
echo "setup.sh: capabilities saved to $root/tensorfold-capabilities.json"

# The config's serve flags should all be ones this binary lists (a text search: the JSON layout is TensorFold's).
caps=$(cat "$root/tensorfold-capabilities.json")
flags=$(jq -r '.tools[0].options.server[] | select(startswith("--"))' "$here/config.example.json")
missing=
while read -r flag; do
  grep -qF -- "\"$flag" <<< "$caps" || missing="$missing $flag"
done <<< "$flags"
[ -z "$missing" ] ||
  echo "setup.sh: WARNING: capabilities --json does not mention:$missing (qualify.sh shows whether serve takes them)"

# The cache as TensorFold sees it, offline: the checkpoint is listed and its info reads.
env HF_HUB_CACHE="$hf" HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1 timeout 300 "$bin" models \
  > "$root/tensorfold-models.txt" 2>&1 || { cat "$root/tensorfold-models.txt" >&2; die "tensorfold models failed"; }
models=$(cat "$root/tensorfold-models.txt")
grep -qF -- "$repo" <<< "$models" ||
  echo "setup.sh: WARNING: 'models' does not list $repo (see $root/tensorfold-models.txt)"
env HF_HUB_CACHE="$hf" HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1 timeout 300 "$bin" info "$repo" \
  > "$root/tensorfold-info.txt" 2>&1 || { cat "$root/tensorfold-info.txt" >&2; die "tensorfold info $repo failed"; }
echo "setup.sh: tensorfold info $repo:"
cat "$root/tensorfold-info.txt"

# The tool config with this host's paths.
filter='.tools[0].weights = $hf | .tools[0].options.env.HF_HUB_CACHE = $hf | .tools[0].qualification = $marker
  | .tools[0].options.server[0] = $bin'
[ "$lock_set" = 0 ] || filter="$filter | if \$lock == \"\" then del(.gpu_lock) else .gpu_lock = \$lock end"
jq --arg hf "$hf" --arg marker "$marker" --arg bin "$bin" --arg lock "$lock" "$filter" "$here/config.example.json" \
  > "$root/config.json.tmp"
mv "$root/config.json.tmp" "$root/config.json"
cat <<EOF
setup.sh: wrote $root/config.json. Merge its "gpu_lock", "defaults" and "tools" entries into the daemon's config:
$(cat "$root/config.json")
setup.sh: the tool stays refused (503, not GPU-qualified) until qualify.sh writes $marker:
  tools/chat/qwen38-27b/qualify.sh $root /path/to/localrouter OUT $marker
EOF
