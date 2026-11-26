#!/usr/bin/env bash
# The qwen3.8-27b qualification on a GB10: tools/chat/qualify.sh with this model's pins, prompts and qualify.py.
#   tools/chat/qwen38-27b/qualify.sh ROOT LOCALROUTER_BIN OUT MARKER [--config FILE] [--tool ID] [--any-gpu]
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 4 ] || { echo "usage: qualify.sh ROOT LOCALROUTER_BIN OUT MARKER [--config FILE] [--tool ID] [--any-gpu]" >&2; exit 2; }
exec bash "$here/../qualify.sh" "$here" "${@:1:4}" --tool qwen3.8-27b "${@:5}"
