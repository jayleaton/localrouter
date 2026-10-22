#!/usr/bin/env bash
# Installs LocalRouter as the Jarvis box on this GB10 host (DGX-01), from the checkout this script is in:
#   deploy/jarvis/install.sh            then deploy/jarvis/start.sh
# 1. Preflight: aarch64 and a GB10; zig 0.17, nvcc, jq, curl, python3; the qualified installs (chat and Drama 3)
#    and their markers, each "qualified": true and made from the same pins as this tree (else: re-qualify with
#    that model's qualify.sh first); Whisper, the VAD, the image and video packs.
# 2. The tree goes to $DEPLOY/src/<commit> (git archive of HEAD from a clean checkout, or a copy of an archive that
#    carries COMMIT) and is built there with its CUDA kernels; $DEPLOY/current points at it.
# 3. $DEPLOY/config.json from config.json.in, and the systemd user unit from localrouter-jarvis.service.in, enabled
#    (lingering keeps it running without a login). A running service is restarted onto the new build.
# 4. tailscale serve: HTTPS on the tailnet at HTTPS_PORT to the daemon's loopback port (sudo; other serve entries are
#    left as they are). The daemon itself listens on 127.0.0.1 only: LocalRouter has no API keys, so the tailnet (its
#    ACLs) is who may connect; nothing binds 0.0.0.0.
# Settings: deploy/jarvis/env.sh (JARVIS, DEPLOY, PORT, HTTPS_PORT, CHAT_ROOT, CHAT_MARKER, DRAMA_ROOT, DRAMA_MARKER).
# Also SKIP_BUILD=1 (reuse a built $DEPLOY/src/<commit>), NVCC, SM (121).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
source "$here/env.sh"

# ---- 1. preflight
[ "$(uname -m)" = aarch64 ] || die "not aarch64"
nvidia-smi -L | grep -q GB10 || die "no GB10"
for t in jq curl python3 sha256sum systemctl systemd-run zig; do command -v $t > /dev/null || die "needs $t (zig: under $J/bin)"; done
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
[ -x "$nvcc" ] || die "no nvcc at $nvcc"
# marker FILE PINS NAME: qualified, and qualified with this tree's pins
marker() {
  [ -f "$1" ] || die "$3: no qualification marker $1 (run its qualify.sh on this machine)"
  jq -e '.qualified == true' "$1" > /dev/null || die "$3: $1 does not say qualified"
  local want got
  want=$(sha256sum "$2" | cut -d' ' -f1)
  got=$(jq -r '.pins_sha256 // ""' "$1")
  [ "$got" = "$want" ] || die "$3: $1 was qualified with pins $got, this tree has $want: re-qualify first"
  echo "  $3: qualified ($(basename "$1") sha256 $(sha256sum "$1" | cut -c1-12), pins match)"
}
echo "== preflight"
marker "$CHAT_MARKER" "$repo/tools/chat/qwen38-27b/pins.json" "qwen3.8-27b"
marker "$DRAMA_MARKER" "$repo/tools/speech/drama3/pins.json" "fishaudio/drama-3"
[ -x "$CHAT_ROOT/tensorfold-1.1.1/bin/tensorfold-native" ] && [ -d "$CHAT_ROOT/hf" ] || die "no chat install at $CHAT_ROOT"
[ -d "$DRAMA_ROOT" ] || die "no Drama 3 install at $DRAMA_ROOT"
whisper=$(ls -d "$J"/models/whisper/whisper-large-v3-turbo/*/ 2> /dev/null | head -1)
[ -n "$whisper" ] && [ -x "$J/venvs/whisper/bin/python" ] && [ -f "$J/models/vad/silero_vad.jit" ] || die "Whisper is not set up under $J"
[ -d "$J/lr/weights/qwen-image-2.1" ] && [ -d "$J/lr/weights/minimax-h3" ] || die "no image and video packs under $J/lr/weights"

# ---- 2. the tree and the build
if git -C "$repo" rev-parse --git-dir > /dev/null 2>&1; then
  git -C "$repo" diff --quiet HEAD -- || die "the checkout has uncommitted changes: commit them, or install from a clean tree"
  commit=$(git -C "$repo" rev-parse HEAD)
else
  commit=$(cat "$repo/COMMIT" 2> /dev/null) || die "neither a git checkout nor a COMMIT file in $repo"
fi
src=$DEPLOY/src/$commit
mkdir -p "$DEPLOY/src" "$DEPLOY/data" "$DEPLOY/logs"
if [ "${SKIP_BUILD:-0}" != 1 ] || [ ! -x "$src/zig-out/bin/localrouter" ]; then
  echo "== build $commit"
  rm -rf "$src.tmp" && mkdir -p "$src.tmp"
  if git -C "$repo" rev-parse --git-dir > /dev/null 2>&1; then
    git -C "$repo" archive HEAD | tar -x -C "$src.tmp"
  else
    tar -C "$repo" --exclude=.zig-cache --exclude=zig-out -cf - . | tar -x -C "$src.tmp"
  fi
  echo "$commit" > "$src.tmp/COMMIT"
  (cd "$src.tmp" && ZIG_GLOBAL_CACHE_DIR=$J/zig-cache systemd-run --user --scope -q -p MemoryMax=24G -p CPUQuota=800% \
    timeout 2400 zig build -Doptimize=ReleaseSafe -Dnvcc="$nvcc" -Dsm="${SM:-121}") > "$DEPLOY/logs/build-$commit.log" 2>&1 ||
    { tail -20 "$DEPLOY/logs/build-$commit.log"; die "the build failed: $DEPLOY/logs/build-$commit.log"; }
  rm -rf "$src" && mv "$src.tmp" "$src"
fi
ln -sfn "src/$commit" "$DEPLOY/current"
sha256sum "$src/zig-out/bin/localrouter"

# ---- 3. config and unit
render() {
  sed -e "s|@J@|$J|g" -e "s|@SRC@|$DEPLOY/current|g" -e "s|@DEPLOY@|$DEPLOY|g" -e "s|@PORT@|$PORT|g" \
      -e "s|@CHAT_ROOT@|$CHAT_ROOT|g" -e "s|@CHAT_MARKER@|$CHAT_MARKER|g" -e "s|@DRAMA_ROOT@|$DRAMA_ROOT|g" \
      -e "s|@DRAMA_MARKER@|$DRAMA_MARKER|g" -e "s|@WHISPER@|${whisper%/}|g" "$1"
}
render "$here/config.json.in" > "$DEPLOY/config.json.new"
jq -e '.tools | length == 6' "$DEPLOY/config.json.new" > /dev/null || die "the rendered config is not valid"
! grep -q '@[A-Z_]*@' "$DEPLOY/config.json.new" || die "unrendered placeholders in the config"
mv "$DEPLOY/config.json.new" "$DEPLOY/config.json"
mkdir -p "$HOME/.config/systemd/user"
render "$here/localrouter-jarvis.service.in" > "$HOME/.config/systemd/user/$UNIT.service"
systemctl --user daemon-reload
systemctl --user enable "$UNIT" > /dev/null 2>&1
[ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] ||
  echo "warning: lingering is off: the service stops at logout (sudo loginctl enable-linger $(id -un))" >&2
if systemctl --user is-active -q "$UNIT"; then
  # never under a live call: wait until no voice session is open (/v1/tools, else, on an older build, no connection
  # to the daemon's port), IDLE_WAIT_S at most (default 1800; FORCE_RESTART=1 skips the wait)
  t0=$(date +%s)
  while [ "${FORCE_RESTART:-0}" != 1 ]; do
    n=$(curl -fs -m 5 "$URL/v1/tools" | jq -r '.voice_sessions // empty' 2> /dev/null || true)
    [ -n "$n" ] || n=$(ss -Htn state established "( sport = :$PORT )" 2> /dev/null | wc -l)
    [ "$n" = 0 ] && break
    [ $(($(date +%s) - t0)) -lt "${IDLE_WAIT_S:-1800}" ] ||
      die "$n voice session(s) still open after ${IDLE_WAIT_S:-1800} s: not restarting (FORCE_RESTART=1 to restart anyway)"
    echo "  waiting: $n voice session(s) open"
    sleep 10
  done
  echo "== restarting the running service onto $commit"
  systemctl --user restart "$UNIT"
fi

# ---- 4. HTTPS for the tailnet
if [ "$HTTPS_PORT" != 0 ] && command -v tailscale > /dev/null; then
  sudo -n tailscale serve --bg --https="$HTTPS_PORT" "http://127.0.0.1:$PORT" > /dev/null ||
    echo "warning: tailscale serve --https=$HTTPS_PORT failed (needs sudo); use an ssh tunnel (deploy/jarvis/README.md)" >&2
fi
echo "installed $commit: $DEPLOY (config.json, current -> src/$commit), unit $UNIT"
echo "start it: deploy/jarvis/start.sh    URL: $(public_url || true) (loopback: $URL)"
