#!/usr/bin/env bash
# Copies the weights from the network volume to the pod's local disk, 16 files at a time (the volume gives about
# 20-100 MB/s a stream; mmap page faults on it are far slower). Idempotent.
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ -f $MODEL/.complete ]] && exit 0
mkdir -p $MODEL
(cd $MODEL_VOL && find . -path ./.cache -prune -o -type f -print0 | xargs -0 -P 16 -I{} sh -c 'mkdir -p "$(dirname "$0/{}")" && cp "{}" "$0/{}"' $MODEL)
touch $MODEL/.complete
du -sh $MODEL
