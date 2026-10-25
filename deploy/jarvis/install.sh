#!/usr/bin/env bash
# Installs LocalRouter as the Jarvis box on this GB10 host (DGX-01), from the checkout this script is in:
#   deploy/jarvis/install.sh            then deploy/jarvis/start.sh
# 1. Preflight: aarch64 and a GB10; zig 0.17, nvcc, jq, curl, python3; the qualified installs (chat and Drama 3)
#    and their markers, each "qualified": true and made from the same pins as this tree (else: re-qualify with
#    that model's qualify.sh first); Whisper, the VAD, the image and video packs.
# 2. The tree goes to $DEPLOY/src/<commit> (git archive of HEAD from a clean checkout, or a copy of an archive that
#    carries COMMIT) and is built there with its CUDA kernels; $DEPLOY/current points at it.
# 3. $DEPLOY/config.json from config.json.in, and the systemd user unit from localrouter-jarvis.service.in, enabled
#    (lingering keeps it running without a login). A running service is restarted onto the new build once no voice
#    session is open; until then (and when a check fails) current, the config and the unit stay as they were.
# 4. tailscale serve: HTTPS on the tailnet at HTTPS_PORT to the daemon's loopback port (sudo; other serve entries are
#    left as they are). The daemon itself listens on 127.0.0.1 only and nothing binds 0.0.0.0: the tailnet (its ACLs)
#    is who may connect, and with pairing.required only paired devices, each within its scopes, use it (pair.sh).
# Settings: deploy/jarvis/env.sh (JARVIS, DEPLOY, PORT, HTTPS_PORT, UNIT, CHAT_ROOT, CHAT_MARKER, DRAMA_ROOT, DRAMA_MARKER,
# DRAMA_PORT).
# Also SKIP_BUILD=1 (reuse a built $DEPLOY/src/<commit>), BUILD_ONLY=1 (preflight of the tools and step 2 only: the
# binary that the models' qualify.sh runs), TOOLS (the tool ids to serve, space separated; default: every tool in
# config.json.in), NVCC, SM (121), PAIRING_REQUIRED (1: every route but /health and pairing needs a paired device with
# the scope it needs, and the scripts get $DEPLOY/operator.token (scope read); 0: not; unset:
# as the deployed config has it).
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
all=$(sed -e 's/@DRAMA_PORT@/0/' -e 's/@DRAMA_RESIDENT@/0/' "$here/config.json.in" | jq -r '[.tools[].id] | join(" ")')
TOOLS=${TOOLS:-$all}
for t in $TOOLS; do [[ " $all " == *" $t "* ]] || die "TOOLS has '$t'; config.json.in has: $all"; done
wanted() { [[ " $TOOLS " == *" $1 "* ]]; }
# marker FILE PINS NAME: qualified, and qualified with this tree's pins
marker() {
  [ -f "$1" ] || die "$3: no qualification marker $1 (run its qualify.sh on this machine)"
  jq -e '.qualified == true' "$1" > /dev/null || die "$3: $1 does not say qualified"
  local want got
  want=$(sha256sum "$2" | cut -d' ' -f1)
  got=$(jq -r '.pins_sha256 // ""' "$1")
  [ "$got" = "$want" ] || die "$3: $1 was qualified with pins $got, this tree has $want: re-qualify first"
  if [ -n "${4:-}" ]; then # a variant's marker: its weights and their own pins (pins-VARIANT.json) as well
    got=$(jq -r '.variant // "full"' "$1")
    [ "$got" = "$4" ] || die "$3: $1 qualified the $got weights, DRAMA_VARIANT is $4"
    if [ "$4" != full ]; then
      want=$(sha256sum "$(dirname "$2")/pins-$4.json" | cut -d' ' -f1)
      got=$(jq -r '.variant_pins_sha256 // ""' "$1")
      [ "$got" = "$want" ] || die "$3: $1 was qualified with $4 pins $got, this tree has $want: re-qualify first"
    fi
  fi
  echo "  $3: qualified ($(basename "$1") sha256 $(sha256sum "$1" | cut -c1-12), pins match)"
}
whisper=$(ls -d "$J"/models/whisper/whisper-large-v3-turbo/*/ 2> /dev/null | awk 'NR == 1' || true)   # none yet: fine for BUILD_ONLY
whisper=${whisper%/}
if [ "${BUILD_ONLY:-0}" != 1 ]; then
  echo "== preflight: $TOOLS"
  if wanted qwen3.8-27b; then
    marker "$CHAT_MARKER" "$repo/tools/chat/qwen38-27b/pins.json" "qwen3.8-27b"
    [ -x "$CHAT_ROOT/tensorfold-1.1.4/bin/tensorfold-native" ] && [ -d "$CHAT_ROOT/hf" ] || die "no chat install at $CHAT_ROOT"
  fi
  if wanted fishaudio/drama-3; then
    marker "$DRAMA_MARKER" "$repo/tools/speech/drama3/pins.json" "fishaudio/drama-3" "$DRAMA_VARIANT"
    [ -d "$DRAMA_ROOT" ] || die "no Drama 3 install at $DRAMA_ROOT"
  fi
  if wanted whisper-large-v3-turbo; then
    [ -n "$whisper" ] && [ -x "$J/venvs/whisper/bin/python" ] && [ -f "$J/models/vad/silero_vad.jit" ] || die "Whisper is not set up under $J"
  fi
  if wanted qwen-image-2.1-turbo; then
    [ -f "$J/lr/weights/qwen-image-2.1/turbo-fp8s/manifest.json" ] || die "no image pack under $J/lr/weights/qwen-image-2.1"
    # edits need the edit pack (the vision tower and the VAE encoder; tools/spark/prepare.sh adds it): without it the
    # image model serves text to image only and does not list image_edit
    [ -f "$J/lr/weights/qwen-image-2.1/edit.json" ] || echo "install: no edit pack under $J/lr/weights/qwen-image-2.1: image edits are off (re-run tools/spark/prepare.sh to add it)" >&2
  fi
  if wanted knowline-4b; then # tools/decide/knowline/README.md: the pinned weights (checked against SHA256SUMS) and its venv
    [ "$(cat "$J/models/decide/knowline-4b-gen5.5/.revision" 2> /dev/null)" = "$KNOWLINE_REV" ] ||
      die "knowline-4b: no weights at $J/models/decide/knowline-4b-gen5.5 of revision $KNOWLINE_REV (tools/decide/knowline/fetch.sh)"
    [ -x "$J/venvs/decide/bin/python" ] || die "knowline-4b: no venv at $J/venvs/decide (tools/decide/knowline/fetch.sh)"
  fi
  if wanted minimax-h3; then
    [ -f "$J/lr/weights/minimax-h3/h3-turbo-nvfp4/manifest.json" ] || die "no video pack under $J/lr/weights/minimax-h3"
  fi
fi

# ---- 2. the tree and the build
if git -C "$repo" rev-parse --git-dir > /dev/null 2>&1; then
  git -C "$repo" diff --quiet HEAD -- || die "the checkout has uncommitted changes: commit them, or install from a clean tree"
  commit=$(git -C "$repo" rev-parse HEAD)
else
  commit=$(cat "$repo/COMMIT" 2> /dev/null) || die "neither a git checkout nor a COMMIT file in $repo"
fi
src=$DEPLOY/src/$commit
mkdir -p "$DEPLOY/src" "$DEPLOY/data" "$DEPLOY/logs" "$DEPLOY/inbox"
if [ "${SKIP_BUILD:-0}" != 1 ] || [ ! -x "$src/zig-out/bin/localrouter" ]; then
  echo "== build $commit"
  rm -rf "$src.tmp" && mkdir -p "$src.tmp"
  if git -C "$repo" rev-parse --git-dir > /dev/null 2>&1; then
    git -C "$repo" archive HEAD | tar -x -C "$src.tmp"
  else
    tar -C "$repo" --exclude=.zig-cache --exclude=zig-out -cf - . | tar -x -C "$src.tmp"
  fi
  echo "$commit" > "$src.tmp/COMMIT"
  (cd "$src.tmp" && ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-$J/zig-cache} systemd-run --user --scope -q -p MemoryMax=24G -p CPUQuota=800% \
    timeout 2400 zig build -Doptimize=ReleaseSafe -Dnvcc="$nvcc" -Dsm="${SM:-121}") > "$DEPLOY/logs/build-$commit.log" 2>&1 ||
    { tail -20 "$DEPLOY/logs/build-$commit.log"; die "the build failed: $DEPLOY/logs/build-$commit.log"; }
  rm -rf "$src" && mv "$src.tmp" "$src"
fi
sha256sum "$src/zig-out/bin/localrouter"
[ "${BUILD_ONLY:-0}" = 1 ] && { echo "built $commit: $src/zig-out/bin/localrouter (BUILD_ONLY=1: nothing else changed)"; exit 0; }

# ---- 3. config and unit, rendered and checked first; nothing the running service uses changes until it is idle
render() {
  sed -e "s|@J@|$J|g" -e "s|@SRC@|$DEPLOY/current|g" -e "s|@DEPLOY@|$DEPLOY|g" -e "s|@PORT@|$PORT|g" \
      -e "s|@CHAT_ROOT@|$CHAT_ROOT|g" -e "s|@CHAT_MARKER@|$CHAT_MARKER|g" -e "s|@DRAMA_ROOT@|$DRAMA_ROOT|g" \
      -e "s|@DRAMA_MARKER@|$DRAMA_MARKER|g" -e "s|@DRAMA_PORT@|$DRAMA_PORT|g" -e "s|@DRAMA_VARIANT@|$DRAMA_VARIANT|g" \
      -e "s|@DRAMA_RESIDENT@|$DRAMA_RESIDENT|g" -e "s|@WHISPER@|${whisper:-unused}|g" "$1"
}
# pairing (pair.sh) carries over from the deployed config; PAIRING_REQUIRED=1 turns it on, 0 off
pairing=$(jq -c '.pairing // {}' "$DEPLOY/config.json" 2> /dev/null || echo '{}')
case ${PAIRING_REQUIRED:-} in 1) pairing='{"required":true}' ;; 0) pairing='{}' ;; esac
# the selected tools only (TOOLS), and the voice and defaults that name them
render "$here/config.json.in" | jq --arg t " $TOOLS " --argjson pairing "$pairing" '
  def kept: . as $id | $t | contains(" " + $id + " ");
  .tools |= map(select(.id | kept)) | .voice.speech_models |= map(select(kept))
  | .defaults |= with_entries(select(.value | kept))
  | if $pairing == {} then . else .pairing = $pairing end' > "$DEPLOY/config.json.new"
jq -e '(.tools | length > 0) and (tostring | test("@[A-Z_]+@") | not)' "$DEPLOY/config.json.new" > /dev/null ||
  die "the rendered config is not valid (no tools, or a @…@ left)"
# with pairing required, these scripts' own token (read only: models and residency, env.sh), made once
if jq -e '.pairing.required == true' "$DEPLOY/config.json.new" > /dev/null && [ ! -s "$TOKEN_FILE" ]; then
  (umask 077 && "$src/zig-out/bin/localrouter" pair token jarvis-operator --scopes read --data "$DEPLOY/data" > "$TOKEN_FILE.tmp") &&
    mv "$TOKEN_FILE.tmp" "$TOKEN_FILE" || die "could not make the operator token ($TOKEN_FILE)"
fi
mkdir -p "$HOME/.config/systemd/user"
render "$here/localrouter-jarvis.service.in" > "$DEPLOY/$UNIT.service.new"
! grep -q '@[A-Z_]*@' "$DEPLOY/$UNIT.service.new" || die "unrendered placeholders in the unit"
running=0; systemctl --user is-active -q "$UNIT" && running=1
if [ $running = 1 ] && [ "$(readlink "$DEPLOY/current")" = "src/$commit" ] && cmp -s "$DEPLOY/config.json.new" "$DEPLOY/config.json" &&
   cmp -s "$DEPLOY/$UNIT.service.new" "$HOME/.config/systemd/user/$UNIT.service"; then
  # nothing to change: a re-run leaves the running service alone
  rm -f "$DEPLOY/config.json.new" "$DEPLOY/$UNIT.service.new"
  echo "== unchanged: $UNIT already runs $commit with this config; not restarting"
  running=unchanged
fi
if [ $running = 1 ]; then
  # never under a live call: wait until no voice session is open (/v1/tools, else, on an older build, no connection
  # to the daemon's port), IDLE_WAIT_S at most (default 1800; FORCE_RESTART=1 skips the wait)
  t0=$(date +%s)
  while [ "${FORCE_RESTART:-0}" != 1 ]; do
    n=$(lr_curl -fs -m 5 "$URL/v1/tools" | jq -r '.voice_sessions // empty' 2> /dev/null || true)
    [ -n "$n" ] || n=$(ss -Htn state established "( sport = :$PORT )" 2> /dev/null | wc -l)
    [ "$n" = 0 ] && break
    [ $(($(date +%s) - t0)) -lt "${IDLE_WAIT_S:-1800}" ] ||
      die "$n voice session(s) still open after ${IDLE_WAIT_S:-1800} s: nothing changed (FORCE_RESTART=1 to restart anyway)"
    echo "  waiting: $n voice session(s) open"
    sleep 10
  done
fi
if [ $running != unchanged ]; then
  ln -sfn "src/$commit" "$DEPLOY/current.new" && mv -T "$DEPLOY/current.new" "$DEPLOY/current"
  mv "$DEPLOY/config.json.new" "$DEPLOY/config.json"
  mv "$DEPLOY/$UNIT.service.new" "$HOME/.config/systemd/user/$UNIT.service"
  systemctl --user daemon-reload
  systemctl --user enable "$UNIT" > /dev/null 2>&1
fi
[ "$(loginctl show-user "$(id -un)" -p Linger --value 2> /dev/null)" = yes ] ||
  echo "warning: lingering is off: the service stops at logout (sudo loginctl enable-linger $(id -un))" >&2
if [ $running = 1 ]; then
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
