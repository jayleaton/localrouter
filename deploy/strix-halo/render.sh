#!/usr/bin/env bash
# Renders deploy/strix-halo/config.json.in into $DEPLOY/config.json from env.sh (+ $SITE). budget_bytes defaults to
# the GPU's GTT pool less GTT_MARGIN_BYTES (tools/install/lib.sh hw_gtt). Prints the path.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$here/../.." && pwd)
source "$REPO/tools/install/lib.sh"
source "$here/env.sh"; [ -z "${SITE:-}" ] || source "$SITE"
gtt=$(hw_gtt); BUDGET_BYTES=${BUDGET_BYTES:-$(( gtt > GTT_MARGIN_BYTES ? gtt - GTT_MARGIN_BYTES : 0 ))}
SRC=$REPO
[ "$H3_RESIDENT" != budget ] || H3_RESIDENT=$BUDGET_BYTES
[ "$H3IMG_RESIDENT" != budget ] || H3IMG_RESIDENT=$BUDGET_BYTES
mkdir -p "$DEPLOY/inbox" "$DEPLOY/data" "$DEPLOY/logs"
export MACHINE RESERVE_BYTES BUDGET_BYTES DEPLOY CHAT_DEFAULT STT_DEFAULT VOICE_CHAT VOICE FLASHNEXT_DIR FLASHNEXT_RESIDENT \
  GUFO_LLM_PORT Q3_DIR Q3_RESIDENT LLAMA_BIN LLAMA_PORT PYTHON SRC TTS_DIR TTS_RESIDENT GUFO_TTS_PORT ASR_DIR ASR_RESIDENT \
  GUFO_ASR_PORT J WHISPER_DIR COMFY_DIR COMFY_PYTHON COMFY_H3_PORT COMFY_H3IMG_PORT H3_RESIDENT H3IMG_RESIDENT
python3 -I - "$here/config.json.in" "$DEPLOY/config.json" <<'PY'
import json, os, re, sys
src, out = sys.argv[1], sys.argv[2]
text = re.sub(r"@([A-Z0-9_]+)@", lambda m: os.environ[m.group(1)], open(src).read())
json.loads(text)  # a template or site value that breaks JSON fails here, not in the daemon
open(out + ".tmp", "w").write(text); os.replace(out + ".tmp", out)
PY
echo "$DEPLOY/config.json"
