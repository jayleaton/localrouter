#!/usr/bin/env bash
# Qwen3.8-Flash-Next TP1 on one GB10, as LocalRouter's openai_server engine runs a model's server (options.server):
#   serve.sh PORT NAME MODEL_DIR
# It runs the recipe's TP1 image (pins.json; FLASHNEXT_IMAGE) in the foreground with the recipe's boot line (README,
# "Serve on one Spark (TP1)") and its engine settings (tf.env: the compute lane and the runner on), listening on
# 127.0.0.1:PORT and serving the model as NAME. Two settings are added, neither a change to what the model computes:
#   TENSORFOLD_CUDA_MEMORY_LIMIT_GB (FLASHNEXT_CUDA_LIMIT_GIB, default 103.5): the engine sizes its streams and prompt
#     cache to the free memory; this caps it at the recipe's measured boot (model 77.15 GiB, then 8 streams of 2.28 GiB
#     and an 8 GiB prompt cache in 26.3 GiB), so the scheduler's booking holds on any node;
#   --no-update-check (and TENSORFOLD_NO_UPDATE_CHECK=1): the server asks nobody about updates.
# The container is named localrouter-NAME (each character outside [A-Za-z0-9_.-] becomes -) and labelled with the tool
# (org.localrouter.tool) and the install (org.localrouter.owner, FLASHNEXT_OWNER): only a container with this exact name
# and this tool's label is ever stopped or removed.
# Stopping: SIGTERM or SIGINT stops the container (docker stop, FLASHNEXT_STOP_S seconds, default 20, then SIGKILL),
# then exits. When this script is killed outright (the worker does that when its daemon dies), a watchdog it started
# stops the container. A container left by an earlier run (a host crash) is removed before the new one starts. The
# install's leftovers.sh stops any of its labelled containers after the service stops.
set -euo pipefail
[ $# = 3 ] || { echo "usage: serve.sh PORT NAME MODEL_DIR" >&2; exit 2; }
port=$1 name=$2 model=$3
here=$(cd "$(dirname "$0")" && pwd)
image=${FLASHNEXT_IMAGE:?FLASHNEXT_IMAGE: the TP1 image of the recipe, pins.json}
limit=${FLASHNEXT_CUDA_LIMIT_GIB:-103.5}
stop_s=${FLASHNEXT_STOP_S:-20}
owner=${FLASHNEXT_OWNER:-}
box=localrouter-${name//[^A-Za-z0-9_.-]/-}
tree=$(jq -r '.image.labels["org.tensorfold.engine-tree"]' "$here/pins.json")

command -v docker > /dev/null || { echo "serve.sh: no docker" >&2; exit 1; }
got=$(docker image inspect -f '{{index .Config.Labels "org.tensorfold.engine-tree"}}' "$image" 2> /dev/null) ||
  { echo "serve.sh: no image $image here (tools/chat/qwen38-flash-next/setup.sh)" >&2; exit 1; }
[ "$got" = "$tree" ] || { echo "serve.sh: image $image is engine tree $got, pins.json wants $tree" >&2; exit 1; }
[ -f "$model/config.json" ] && [ -f "$model/model.safetensors.index.json" ] ||
  { echo "serve.sh: no checkpoint at $model" >&2; exit 1; }

# a container of this name: ours (this tool's label) is a leftover, removed; anyone else's is left alone
if left=$(docker inspect -f '{{index .Config.Labels "org.localrouter.tool"}}' "$box" 2> /dev/null); then
  [ "$left" = "$name" ] || { echo "serve.sh: a container named $box exists and is not this tool's (label '$left'): not touching it" >&2; exit 1; }
  echo "serve.sh: removing $box, left by an earlier server" >&2
  docker rm -f "$box" > /dev/null
fi

cid_dir=$(mktemp -d "${TMPDIR:-/tmp}/lr-flashnext.XXXXXX")
cid_file=$cid_dir/cid # this run's container id: what is stopped is the container this run started, never a later one
docker run --rm --name "$box" --cidfile "$cid_file" --label "org.localrouter.tool=$name" --label "org.localrouter.owner=$owner" \
  --init --gpus all --ipc=host --network host --ulimit memlock=-1 \
  --env-file "$here/tf.env" -e "TENSORFOLD_CUDA_MEMORY_LIMIT_GB=$limit" -e TENSORFOLD_NO_UPDATE_CHECK=1 \
  -v "$model":/model:ro "$image" \
  serve /model --host 127.0.0.1 --port "$port" --name "$name" --context 65536 --thinking-budget 8192 --no-update-check &
run=$!
# stop_box: stops this run's container (waiting for docker run to name it while that is still starting)
stop_box() {
  local cid="" _
  for _ in $(seq 60); do
    cid=$(cat "$cid_file" 2> /dev/null || true)
    [ -n "$cid" ] || ! kill -0 "$run" 2> /dev/null || { sleep 0.5; continue; }
    break
  done
  [ -z "$cid" ] || docker stop -t "$stop_s" "$cid" > /dev/null 2>&1 || true
  rm -f "$cid_file"
  rmdir "$cid_dir" 2> /dev/null || true
}
# the watchdog: once this script is gone (killed outright), it stops the container; it ignores TERM and INT
me=$$
(
  trap '' TERM INT
  while kill -0 "$me" 2> /dev/null; do sleep 1; done
  stop_box
) &
dog=$!
trap 'stop_box; wait "$run" || true; kill -KILL "$dog" 2> /dev/null || true; exit 0' TERM INT
rc=0
wait "$run" || rc=$?
kill -KILL "$dog" 2> /dev/null || true
rm -f "$cid_file"
rmdir "$cid_dir" 2> /dev/null || true
exit "$rc"
