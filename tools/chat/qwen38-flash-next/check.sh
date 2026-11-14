#!/usr/bin/env bash
# Whether this machine can serve Qwen3.8-Flash-Next TP1 (pins.json), without starting it (no GPU, no download):
#   check.sh IMAGE MODEL_DIR
# docker runs for this user with the NVIDIA runtime; IMAGE is the recipe's TP1 image (its engine tree and runner assets
# are the pinned ones); MODEL_DIR holds the pinned checkpoint: its config, its index and every shard the index names, of
# the pinned count (and, when the download left a `.p0-source` naming its repo@revision, that one). One line a check;
# exit 1 at the first that fails, with what to do. tools/chat/qwen38-flash-next/README.md says how to get each.
set -euo pipefail
[ $# = 2 ] || { echo "usage: check.sh IMAGE MODEL_DIR" >&2; exit 2; }
image=$1 model=$2
here=$(cd "$(dirname "$0")" && pwd)
pins=$here/pins.json
fail() { echo "qwen3.8-flash-next: $*" >&2; exit 1; }
command -v docker > /dev/null || fail "no docker (install Docker with the NVIDIA container runtime)"
runtimes=$(docker info --format '{{json .Runtimes}}' 2> /dev/null || true)
grep -q nvidia <<< "$runtimes" ||
  fail "docker does not answer for $(id -un), or has no nvidia runtime (add the user to the docker group; install nvidia-container-toolkit)"
labels=$(docker image inspect -f '{{json .Config.Labels}}' "$image" 2> /dev/null) ||
  fail "no image $image (build it with the recipe, or load it from another node: README.md, The image)"
want=$(jq -c '.image.labels' "$pins")
jq -e --argjson w "$want" '. as $l | all($w | to_entries[]; $l[.key] == .value)' <<< "$labels" > /dev/null ||
  fail "image $image has labels $(jq -c --argjson w "$want" 'with_entries(select(.key as $k | $w | has($k)))' <<< "$labels"), pins.json wants $want"
echo "  qwen3.8-flash-next: image $image (engine tree $(jq -r '.image.labels["org.tensorfold.engine-tree"]' "$pins" | cut -c1-12))"
[ -f "$model/config.json" ] && [ -f "$model/model.safetensors.index.json" ] || fail "no checkpoint at $model (README.md, The checkpoint)"
rev=$(jq -r '.checkpoint.repo + "@" + .checkpoint.revision' "$pins")
if [ -f "$model/.p0-source" ] && [ "$(cat "$model/.p0-source")" != "$rev" ]; then
  fail "$model is $(cat "$model/.p0-source"), pins.json wants $rev"
fi
shards=$(jq -r '[.weight_map[]] | unique[]' "$model/model.safetensors.index.json")
n=0
for s in $shards; do
  [ -s "$model/$s" ] || fail "$model lacks $s, which its index names (an incomplete download)"
  n=$((n + 1))
done
[ "$n" = "$(jq -r .checkpoint.files "$pins")" ] || fail "$model has $n shards, pins.json's checkpoint has $(jq -r .checkpoint.files "$pins")"
echo "  qwen3.8-flash-next: checkpoint $model ($n shards, $rev)"
