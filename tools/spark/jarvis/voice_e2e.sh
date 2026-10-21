#!/usr/bin/env bash
# The voice loop end to end on DGX-01: the integration build, the qualified chat (Qwen3.8-27B) and Drama 3 setups,
# Whisper and Kitten from the bring-up (docs/dev/JARVIS-BRINGUP.md on #16), the image and video packs from install.sh,
# driven over /v1/realtime by voice_e2e.py with recorded speech (prompts spoken by Drama 3 itself).
#
#   voice_e2e.sh stage [REF] [HOST]   on the dev box: git archive of REF (default integration/jarvis) plus this kit
#                                     (voice_e2e.sh, voice_e2e.py, SHA256SUMS) to HOST:~/jarvis/e2e/kit/<sha>/; no build,
#                                     no interpreter; prints the run command. HOST defaults to dgx01@192.168.1.116.
#   voice_e2e.sh run SHA              on the GPU host, in a granted window: preflight, build, config, daemon, driver.
#
# Everything goes under ~/jarvis/e2e: src/<sha> (the build), out/<UTC time>/ (config, logs, results, audio,
# SHA256SUMS), data/ (the daemon's). Inputs it reads and never writes: ~/jarvis/qualify-chat (its config.json with
# the chat tool and its qualification marker), ~/jarvis/qualify-drama3/config.json (Drama 3 and its marker),
# ~/jarvis/venvs/{whisper,kitten}, ~/jarvis/models/{whisper,vad,kitten}, ~/jarvis/lr/weights (image and video),
# ~/jarvis/bin (zig, ffmpeg). Overrides: CHAT_CONFIG, DRAMA_CONFIG, PORT (28390), NVCC, SM (121), SKIP_BUILD=1.
# Fail closed: a missing input, an unqualified marker, another GPU user or too little memory stops it before the
# daemon starts; the daemon is a transient user unit stopped on exit; exit 0 only when every driver check passed.
set -euo pipefail

J=$HOME/jarvis
E=$J/e2e
die() { echo "voice_e2e.sh: $*" >&2; exit 1; }

stage() {
  local ref=${1:-integration/jarvis} host=${2:-dgx01@192.168.1.116}
  local here top sha tmp
  here=$(cd "$(dirname "$0")" && pwd)
  top=$(git -C "$here" rev-parse --show-toplevel) # archive the whole tree (from a subdirectory git archives only it)
  sha=$(git -C "$top" rev-parse --verify "$ref^{commit}") || die "no commit $ref"
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN
  git -C "$top" archive --format=tar.gz --prefix=localrouter/ -o "$tmp/localrouter-$sha.tar.gz" "$sha"
  tar -tzf "$tmp/localrouter-$sha.tar.gz" localrouter/build.zig > /dev/null || die "the archive has no build.zig"
  git -C "$top" show "$sha:tools/spark/jarvis/voice_e2e.sh" > "$tmp/voice_e2e.sh"
  git -C "$top" show "$sha:tools/spark/jarvis/voice_e2e.py" > "$tmp/voice_e2e.py"
  (cd "$tmp" && sha256sum localrouter-"$sha".tar.gz voice_e2e.sh voice_e2e.py > SHA256SUMS)
  ssh -o BatchMode=yes "$host" "mkdir -p jarvis/e2e/kit/$sha"
  scp -q -o BatchMode=yes "$tmp"/* "$host:jarvis/e2e/kit/$sha/"
  echo "staged $sha on $host:~/jarvis/e2e/kit/$sha ($(cut -c1-12 "$tmp/SHA256SUMS" | tr '\n' ' '))"
  echo "run it (in a granted window):"
  echo "  ssh -o BatchMode=yes $host 'cd ~/jarvis/e2e && XDG_RUNTIME_DIR=/run/user/\$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/\$(id -u)/bus setsid nohup bash kit/$sha/voice_e2e.sh run $sha > run-$sha.log 2>&1 < /dev/null &'"
}

mem_gib() { awk '/^MemAvailable:/{printf "%d", $2 / 1048576}' /proc/meminfo; }

# The config.json under $1 whose tools include a `kind` $2 (the qualified setups keep the tool there).
find_config() {
  local dir=$1 kind=$2 f
  for f in "$dir/config.json" "$dir/root/config.json" $(find "$dir" -maxdepth 3 -name config.json 2>/dev/null | sort); do
    [ -f "$f" ] && jq -e --arg k "$kind" '[.tools[]? | select(.kind == $k)] | length > 0' "$f" > /dev/null 2>&1 && { echo "$f"; return 0; }
  done
  return 1
}

# Fails unless the tool (JSON on stdin) names a qualification marker that says "qualified": true.
check_marker() {
  local tool=$1 marker
  marker=$(jq -r '.qualification // ""' <<< "$tool")
  [ -n "$marker" ] || die "$(jq -r .id <<< "$tool") has no qualification marker in its config"
  [ -f "$marker" ] || die "$(jq -r .id <<< "$tool"): $marker does not exist"
  jq -e '.qualified == true' "$marker" > /dev/null || die "$(jq -r .id <<< "$tool"): $marker does not say qualified"
  echo "$(jq -r .id <<< "$tool"): qualified ($marker, sha256 $(sha256sum "$marker" | cut -c1-16))"
}

run() {
  local sha=${1:?usage: voice_e2e.sh run SHA}
  local kit=$E/kit/$sha port=${PORT:-28390} stamp src bin
  out="" sampler="" # globals: the EXIT trap uses them after run returns
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  out=$E/out/$stamp
  src=$E/src/$sha
  mkdir -p "$out" "$E/data"
  exec > >(tee -a "$out/run.log") 2>&1
  echo "== voice e2e $sha at $stamp on $(hostname)"
  export PATH=$J/bin:$PATH # zig, ffmpeg (MP4 encoding), tensorfold
  for t in jq curl python3 sha256sum systemd-run nvidia-smi; do command -v "$t" > /dev/null || die "needs $t"; done
  [ -n "${XDG_RUNTIME_DIR:-}" ] || export XDG_RUNTIME_DIR=/run/user/$(id -u)
  [ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ] || export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus

  # Preflight: the kit's bytes, the host, nothing else on the GPU, room for the whole loop plus one generation job.
  (cd "$kit" && sha256sum -c --quiet SHA256SUMS) || die "the kit in $kit does not match its SHA256SUMS"
  [ "$(uname -m)" = aarch64 ] || die "not aarch64"
  nvidia-smi -L | grep -q GB10 || die "no GB10"
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || die "another process uses the GPU: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"
  ! systemctl --user is-active --quiet jarvis-e2e || die "a jarvis-e2e unit is already running"
  ! curl -fs -m 2 "http://127.0.0.1:$port/health" > /dev/null || die "something answers on port $port"
  [ "$(mem_gib)" -ge 100 ] || die "MemAvailable is $(mem_gib) GiB, under 100"

  # Inputs: the qualified tools, as qualified.
  local chat_cfg drama_cfg chat drama gpu_lock
  chat_cfg=${CHAT_CONFIG:-$(find_config "$J/qualify-chat" chat)} || die "no chat config under $J/qualify-chat"
  drama_cfg=${DRAMA_CONFIG:-$(find_config "$J/qualify-drama3" speech)} || die "no Drama 3 config under $J/qualify-drama3"
  chat=$(jq -c '[.tools[] | select(.kind == "chat")][0]' "$chat_cfg")
  drama=$(jq -c '[.tools[] | select(.id == "fishaudio/drama-3")][0] // [.tools[] | select(.kind == "speech")][0]' "$drama_cfg")
  check_marker "$chat"
  check_marker "$drama"
  gpu_lock=$(jq -r '.gpu_lock // ""' "$chat_cfg")
  [ -z "$gpu_lock" ] || [ -e "$gpu_lock" ] || gpu_lock=""
  local whisper_rev kitten=0
  whisper_rev=$(ls "$J/models/whisper/whisper-large-v3-turbo" 2> /dev/null | head -1 || true)
  [ -n "$whisper_rev" ] && [ -x "$J/venvs/whisper/bin/python" ] && [ -f "$J/models/vad/silero_vad.jit" ] || die "Whisper (venv, model, VAD) is not set up under $J"
  [ -x "$J/venvs/kitten/bin/python" ] && [ -d "$J/models/kitten" ] && kitten=1
  [ -d "$J/lr/weights/qwen-image-2.1" ] && [ -d "$J/lr/weights/minimax-h3" ] || die "the image and video packs are not under $J/lr/weights"

  # Build: the staged tree, the native daemon with its CUDA kernels (as on #16).
  if [ "${SKIP_BUILD:-0}" != 1 ] || [ ! -x "$src/zig-out/bin/localrouter" ]; then
    rm -rf "$src"
    mkdir -p "$src"
    tar -xzf "$kit/localrouter-$sha.tar.gz" -C "$src" --strip-components=1
    echo "== build"
    (cd "$src" && ZIG_GLOBAL_CACHE_DIR=$J/zig-cache systemd-run --user --scope -q -p MemoryMax=24G -p CPUQuota=800% \
      timeout 2400 zig build -Doptimize=ReleaseSafe -Dnvcc="${NVCC:-/usr/local/cuda/bin/nvcc}" -Dsm="${SM:-121}") > "$out/build.log" 2>&1 \
      || { tail -30 "$out/build.log"; die "the build failed (log: $out/build.log)"; }
  fi
  bin=$src/zig-out/bin/localrouter
  sha256sum "$bin" | tee "$out/binary.sha256"

  # Config: chat 100 and the voice models 90, kept warm; Kitten the fallback voice (loaded on demand); the image and
  # video models below them with short TTLs (#16), so a generation job never keeps the voice down.
  local R=$src W=$J/lr/weights
  jq -n --argjson chat "$chat" --argjson drama "$drama" --arg J "$J" --arg R "$R" --arg W "$W" --arg rev "$whisper_rev" \
    --arg lock "$gpu_lock" --argjson kitten "$kitten" --argjson drama_resident "$(jq '.tools[0].resident_bytes // null' "$R/tools/speech/drama3/config.example.json")" '
    {machine: "dgx-01", reserve_bytes: 8589934592}
    + (if $lock == "" then {} else {gpu_lock: $lock} end)
    + {defaults: {chat_completion: $chat.id, text_to_speech: $drama.id, speech_to_text: "whisper-large-v3-turbo",
                  text_to_image: "qwen-image-2.1-turbo", text_to_video: "minimax-h3"},
       voice: {chat_extra: {chat_template_kwargs: {enable_thinking: false}},
               speech_models: ([$drama.id] + (if $kitten == 1 then ["kitten-tts-2"] else [] end)), tool_wait_s: 900},
       tools: ([
         ($chat + {keep_loaded: true, priority: 100}),
         ($drama + {keep_loaded: true, priority: 90, lane: "realtime", resident_bytes: ($drama_resident // $drama.resident_bytes)}
           + (if ($drama.cmd // [] | length) == 3 then {cmd: [$drama.cmd[0], $drama.cmd[1], ($R + "/tools/speech/drama3/drama3_worker.py")]} else {} end)),
         {id: "whisper-large-v3-turbo", kind: "transcription", name: "Whisper large-v3-turbo",
          cmd: [($J + "/venvs/whisper/bin/python"), ($R + "/tools/speech/whisper/worker.py")],
          weights: ($J + "/models/whisper/whisper-large-v3-turbo/" + $rev), resident_bytes: 4294967296, working_bytes: 2147483648,
          idle_ttl_s: 3600, priority: 90, keep_loaded: true,
          options: {device: "cuda", dtype: "float16", attn: "sdpa", vad: "silero", vad_model: ($J + "/models/vad/silero_vad.jit")}}]
         + (if $kitten == 1 then [{id: "kitten-tts-2", kind: "speech", name: "Kitten TTS 2",
          cmd: [($J + "/venvs/kitten/bin/python"), "-I", ($R + "/tools/speech/kitten/kitten_worker.py")],
          weights: ($J + "/models/kitten"), resident_bytes: 12884901888, working_bytes: 2147483648, idle_ttl_s: 600,
          priority: 50, keep_loaded: false, load_timeout_s: 600, generate_timeout_s: 900,
          options: {device: "cuda", max_input_chars: 4096, min_speed: 1, max_speed: 1, default_voice: "Bruno",
                    presets: ["stable", "expressive"], voices: ["Bruno", "Rosie", "Luna", "Jasper"]}}] else [] end)
         + [{id: "qwen-image-2.1-turbo", kind: "image", name: "Qwen-Image 2.1 Turbo (FP8, 8 steps)", engine: "qwen_image_turbo",
          weights: ($W + "/qwen-image-2.1"), idle_ttl_s: 60, priority: 10, options: {precision: "fp8s", max_side: 1664}},
         {id: "minimax-h3", kind: "video", name: "MiniMax H3 (NVFP4, Turbo 8 steps, with audio)", engine: "minimax_h3",
          weights: ($W + "/minimax-h3"), idle_ttl_s: 30, priority: 0,
          options: {max_width: 768, max_height: 448, max_seconds: 5}}])}' > "$out/config.json"
  jq -c '.tools[] | {id, kind, priority, keep_loaded, lane}' "$out/config.json"

  # Memory, every 0.5 s, for the whole run.
  (while true; do echo "$(date +%s.%N) $(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"; sleep 0.5; done) > "$out/memory.log" &
  sampler=$!
  local url=http://127.0.0.1:$port
  stop_all() {
    systemctl --user stop jarvis-e2e 2> /dev/null || true
    kill "$sampler" 2> /dev/null || true
    journalctl --user -u jarvis-e2e --no-pager -o short-precise > "$out/daemon.log" 2> /dev/null || true
    cp -r "$E/data/logs" "$out/worker-logs" 2> /dev/null || true
    (cd "$out" && find . -type f ! -name SHA256SUMS ! -name run.log -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS) || true
  }
  trap stop_all EXIT

  echo "== daemon on $url, MemAvailable $(mem_gib) GiB"
  systemd-run --user --unit=jarvis-e2e -q -p MemoryMax=112G -p RuntimeMaxSec=7200 --setenv=PATH="$PATH" \
    "$bin" serve --config "$out/config.json" --host 127.0.0.1 --port "$port" --data "$E/data"
  for _ in $(seq 150); do curl -fs "$url/health" > /dev/null && break; sleep 0.2; done
  curl -fs "$url/health" > /dev/null || die "the daemon did not come up"

  # Warm: chat, Drama 3 and Whisper are keep_loaded; wait until all three are ready (or one failed).
  local t0 ids state
  t0=$(date +%s)
  ids=$(jq -r '[.tools[] | select(.keep_loaded == true) | .id] | join(" ")' "$out/config.json")
  while true; do
    state=$(curl -fs "$url/v1/tools" | jq -r --arg ids "$ids" '[.data[] | select(.id as $i | $ids | split(" ") | index($i)) | "\(.id)=\(.state)"] | join(" ")')
    echo "$(($(date +%s) - t0)) s: $state"
    grep -q failed <<< "$state" && die "a kept tool failed to load: $(curl -fs "$url/v1/tools" | jq -c '[.data[] | {id, state, last_error}]')"
    [ "$(grep -o '=ready' <<< "$state" | wc -l)" -eq "$(wc -w <<< "$ids")" ] && break
    [ $(($(date +%s) - t0)) -lt 900 ] || die "the kept tools were not ready within 900 s: $state"
    sleep 5
  done
  echo "warm after $(($(date +%s) - t0)) s, MemAvailable $(mem_gib) GiB"
  curl -fs "$url/v1/tools" > "$out/tools-warm.json"

  echo "== driver"
  local rc=0
  python3 -I "$kit/voice_e2e.py" --url "$url" --out "$out" --tts "$(jq -r .id <<< "$drama")" \
    --stt whisper-large-v3-turbo --chat "$(jq -r .id <<< "$chat")" --video minimax-h3 || rc=$?
  curl -fs "$url/v1/tools" > "$out/tools-end.json" || true
  echo "== done: driver exit $rc; results in $out/results.json"
  return $rc
}

case ${1:-} in
  stage) shift; stage "$@" ;;
  run) shift; run "$@" ;;
  *) die "usage: voice_e2e.sh stage [REF] [HOST] | run SHA" ;;
esac
