# Defaults for deploy/strix-halo (sourced by render.sh, start.sh, stop.sh, status.sh). Override any of them in a site
# file passed as SITE=path/to/site.env; a site file is sourced after this one.
J=${J:-$HOME/localrouter-amd}                  # the deployment prefix: venvs, models, deploy/{config.json,data,logs}
DEPLOY=${DEPLOY:-$J/deploy}
PORT=${PORT:-8190}                             # the daemon, on loopback
UNIT=${UNIT:-localrouter-amd}                  # the transient systemd user unit start.sh runs the daemon in
MACHINE=${MACHINE:-strix-halo}
LR_BIN=${LR_BIN:-$REPO/zig-out/bin/localrouter}
PYTHON=${PYTHON:-/usr/bin/python3}
# models (Hugging Face snapshots / GGUF directories)
FLASHNEXT_DIR=${FLASHNEXT_DIR:-$J/models/Qwen3.8-Flash-Next-GGUF}   # UD-Q4_K_XL/, MTP/, mmproj-BF16.gguf (Gufo)
Q3_DIR=${Q3_DIR:-$FLASHNEXT_DIR}                                    # UD-Q3_K_XL/, MTP/ (llama.cpp)
TTS_DIR=${TTS_DIR:-$J/models/Qwen3-TTS-12Hz-1.7B-CustomVoice}
ASR_DIR=${ASR_DIR:-$J/models/Qwen3-ASR-1.7B}
WHISPER_DIR=${WHISPER_DIR:-$J/models/whisper/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9}
LLAMA_BIN=${LLAMA_BIN:-$J/llama.cpp/bin}                            # a llama.cpp build with llama-server
GUFO_IMAGE=${GUFO_IMAGE:-$(cat "$REPO/tools/gufo/IMAGE")}           # the Gufo runtime, pinned by digest; start.sh passes it to the unit
# which model serves what
CHAT_DEFAULT=${CHAT_DEFAULT:-qwen3.8-flash-next}
VOICE_CHAT=${VOICE_CHAT:-qwen3.8-flash-next-q3}  # Q3 + TTS + ASR fit the 96 GiB GTT together; Q4 + both do not
STT_DEFAULT=${STT_DEFAULT:-qwen3-asr}
VOICE=${VOICE:-serena}
# fixed loopback ports of the model servers
GUFO_LLM_PORT=${GUFO_LLM_PORT:-8081}; LLAMA_PORT=${LLAMA_PORT:-8080}
GUFO_TTS_PORT=${GUFO_TTS_PORT:-8091}; GUFO_ASR_PORT=${GUFO_ASR_PORT:-8092}
# memory: measured GTT peaks on a Radeon 8060S (gfx1151) with 96 GiB GTT; the budget is the GTT pool less a margin,
# because GPU allocations beyond GTT fail even when MemAvailable has room
FLASHNEXT_RESIDENT=${FLASHNEXT_RESIDENT:-94489280512}   # 88 GiB
Q3_RESIDENT=${Q3_RESIDENT:-66571993088}                 # 62 GiB
TTS_RESIDENT=${TTS_RESIDENT:-6979321856}                # 6.5 GiB
ASR_RESIDENT=${ASR_RESIDENT:-5905580032}                # 5.5 GiB
RESERVE_BYTES=${RESERVE_BYTES:-4294967296}
GTT_MARGIN_BYTES=${GTT_MARGIN_BYTES:-2147483648}
