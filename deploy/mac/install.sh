#!/usr/bin/env bash
# Installs LocalRouter on this Apple Silicon Mac as a launchd user agent, from the checkout this script is in.
# bootstrap.sh --profile mac-apple-silicon runs it after the models' steps; by hand:
#   deploy/mac/install.sh            then deploy/mac/start.sh
# 1. Preflight: macOS on arm64; jq, curl, python3, plutil, launchctl, zig; each selected model installed where the
#    profile's steps put it (Drama 3's Metal qualification marker made from this tree's pins).
# 2. The tree goes to $DEPLOY/src/<commit> (git archive of HEAD from a clean checkout, or a copy of an archive that
#    carries COMMIT) and is built there; $DEPLOY/current points at it.
# 3. $DEPLOY/config.json, generated: the selected models' tool entries from config.json.in, each with its booking
#    (memory_bytes, working_bytes, reference_working_bytes) and residency (warm: always, else on_demand) from the
#    profile, and the
#    profile's deploy.daemon settings (reserve_bytes). The agent ~/Library/LaunchAgents/$LABEL.plist from
#    localrouter.plist.in. A loaded agent is restarted onto the new build once no voice session is open; until then
#    (and when a check fails) current, the config and the agent stay as they were. What a change replaces is kept as
#    $DEPLOY/current.prev, config.json.prev and $LABEL.plist.prev: deploy/mac/rollback.sh goes back to it.
# 4. tailscale serve: HTTPS on the tailnet at HTTPS_PORT to the daemon's loopback port (other serve entries are left
#    as they are). The daemon itself listens on 127.0.0.1 only.
# Settings: deploy/mac/env.sh (JARVIS, DEPLOY, PORT, HTTPS_PORT, UNIT or LABEL, the models' roots). Also TOOLS (the tool
# ids to serve, space separated; default: every model of the profile), PROFILE_FILE (the manifest; bootstrap sets it),
# SKIP_BUILD=1 (reuse a built $DEPLOY/src/<commit>), PAIRING_REQUIRED (1: every route but /health and pairing needs a
# paired device, and the scripts get $DEPLOY/operator.token (scope read); 0: not; unset: as the deployed config has
# it), CHECK_MODELS=0 (skip step 1's model checks: a test install, where each model fails to load until installed),
# RENDER_ONLY=1 (print the generated config and stop: nothing is checked, built or changed; runs on any OS).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
source "$here/env.sh"
PROFILE_FILE=${PROFILE_FILE:-$repo/deploy/profiles/mac-apple-silicon.json}
all=$(jq -r '[.models[].id] | join(" ")' "$PROFILE_FILE")
TOOLS=${TOOLS:-$all}
for t in $TOOLS; do [[ " $all " == *" $t "* ]] || die "TOOLS has '$t'; the profile has: $all"; done
wanted() { [[ " $TOOLS " == *" $1 "* ]]; }
chat_pins=$repo/tools/chat/qwen38-27b/pins-macos.json
CHAT_BIN=$CHAT_ROOT/$(jq -r .tensorfold.dir "$chat_pins")/$(jq -r .tensorfold.binary "$chat_pins")
WHISPER=$J/models/whisper-mlx/whisper-large-v3-turbo-mlx/$(sed -n 's/^REVISION = "\(.*\)"$/\1/p' "$repo/tools/speech/whisper/macos/fetch_model.py")
PYTHON=$(command -v python3 || echo python3)   # absolute: launchd's PATH is not the shell's

# render FILE: the @…@ values filled in
render() {
  sed -e "s|@J@|$J|g" -e "s|@SRC@|$DEPLOY/current|g" -e "s|@DEPLOY@|$DEPLOY|g" -e "s|@PORT@|$PORT|g" \
      -e "s|@LABEL@|$LABEL|g" -e "s|@LOG@|$LOG|g" -e "s|@PYTHON@|$PYTHON|g" -e "s|@HF_HUB@|$HF_HUB|g" \
      -e "s|@CHAT_BIN@|$CHAT_BIN|g" -e "s|@DRAMA_ROOT@|$DRAMA_ROOT|g" -e "s|@DRAMA_MARKER@|$DRAMA_MARKER|g" \
      -e "s|@DRAMA_PORT@|$DRAMA_PORT|g" -e "s|@WHISPER@|$WHISPER|g" -e "s|@IMAGE_ROOT@|$IMAGE_ROOT|g" "$1"
}
# config PAIRING_JSON: the daemon's config for TOOLS. Every model of the profile has its entry in config.json.in and
# every entry a model; its booking and residency are the profile's, so the two never disagree. The residency is the
# default: Settings (POST /v1/tools/{id}/settings, kept in data/residency.json) changes it without a reinstall.
config() {
  render "$here/config.json.in" | jq --slurpfile prof "$PROFILE_FILE" --arg t " $TOOLS " --argjson pairing "$1" '
    def kept: . as $id | $t | contains(" " + $id + " ");
    ($prof[0].models | map({key: .id, value: .}) | from_entries) as $m
    | if ([.tools[].id] | sort) != ($m | keys) then
        error("config.json.in has the tools \([.tools[].id] | sort), the profile the models \($m | keys)") else . end
    | .tools |= map(select(.id | kept) | $m[.id] as $p
        | .resident_bytes = $p.memory_bytes | .working_bytes = ($p.working_bytes // 0)
        | .residency = (if $p.warm == true then "always" else "on_demand" end) | del(.keep_loaded)
        | if $p.reference_working_bytes then .options.reference_working_bytes = $p.reference_working_bytes else . end)
    | .voice.speech_models |= map(select(kept)) | .defaults |= with_entries(select(.value | kept))
    | . + ($prof[0].deploy.daemon // {})
    | if $pairing == {} then . else .pairing = $pairing end'
}
# pairing (pair.sh) carries over from the deployed config; PAIRING_REQUIRED=1 turns it on, 0 off
pairing=$(jq -c '.pairing // {}' "$DEPLOY/config.json" 2> /dev/null || echo '{}')
case ${PAIRING_REQUIRED:-} in 1) pairing='{"required":true}' ;; 0) pairing='{}' ;; esac
if [ "${RENDER_ONLY:-0}" = 1 ]; then config "$pairing"; exit 0; fi

# ---- 1. preflight
[ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || die "not macOS on Apple Silicon (deploy/jarvis is the GB10's)"
for t in jq curl python3 shasum plutil launchctl git zig; do command -v $t > /dev/null || die "needs $t"; done
if [ "${CHECK_MODELS:-1}" = 1 ]; then
  echo "== preflight: $TOOLS"
  # have WHAT PATH...: every PATH is there, else the step that makes it
  have() { local what=$1 f; shift; for f in "$@"; do [ -e "$f" ] || die "$what: no $f (run bootstrap.sh --profile mac-apple-silicon for it)"; done; echo "  $what: installed"; }
  if wanted qwen3.8-27b; then have qwen3.8-27b "$CHAT_BIN" "$HF_HUB/models--TensorFold--Qwen3.8-27B-MLX-4bit"; fi
  if wanted fishaudio/drama-3; then
    have fishaudio/drama-3 "$DRAMA_ROOT" "$DRAMA_MARKER"
    want=$(sha256 "$repo/tools/speech/drama3/pins-macos.json" | cut -c1-64)
    jq -e --arg p "$want" '.qualified == true and .pins_sha256 == $p' "$DRAMA_MARKER" > /dev/null ||
      die "fishaudio/drama-3: $DRAMA_MARKER is not qualified with this tree's pins ($want): re-run the tts/qualify step"
  fi
  if wanted whisper-large-v3-turbo; then have whisper-large-v3-turbo "$J/venvs/whisper-mlx/bin/python" "$WHISPER" "$J/models/vad/silero_vad.jit"; fi
  if wanted knowline-4b; then have knowline-4b "$J/venvs/decide-mlx/bin/python" "$J/models/decide/knowline-4b-gen5.5"; fi
  if wanted qwen-image-2.1-turbo; then
    have qwen-image-2.1-turbo "$IMAGE_ROOT/venv/bin/python" "$IMAGE_ROOT/qwen-image-2.1-turbo-mlx-q4" "$IMAGE_ROOT/mlx-serve-macos-arm64/mlx-serve"
  fi
else
  echo "== preflight: CHECK_MODELS=0, the models are not checked (each fails to load until it is installed)"
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
  echo "== build $commit (log: $DEPLOY/logs/build-$commit.log)"
  rm -rf "$src.tmp" && mkdir -p "$src.tmp"
  if git -C "$repo" rev-parse --git-dir > /dev/null 2>&1; then
    git -C "$repo" archive HEAD | tar -x -C "$src.tmp"
  else
    tar -C "$repo" --exclude=.zig-cache --exclude=zig-out -cf - . | tar -x -C "$src.tmp"
  fi
  echo "$commit" > "$src.tmp/COMMIT"
  (cd "$src.tmp" && DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer} \
    ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-$J/zig-cache} zig build -Doptimize=ReleaseSafe) > "$DEPLOY/logs/build-$commit.log" 2>&1 ||
    { tail -20 "$DEPLOY/logs/build-$commit.log"; die "the build failed: $DEPLOY/logs/build-$commit.log"; }
  rm -rf "$src" && mv "$src.tmp" "$src"
fi
sha256 "$src/zig-out/bin/localrouter"

# ---- 3. config and agent, made and checked first; nothing the running agent uses changes until it is idle
config "$pairing" > "$DEPLOY/config.json.new" || die "could not generate the config (above)"
jq -e '(.tools | length > 0) and (tostring | test("@[A-Z_]+@") | not)' "$DEPLOY/config.json.new" > /dev/null ||
  die "the generated config is not valid (no tools, or a @…@ left)"
# with pairing required, these scripts' own token (read only: models and residency), made once
if jq -e '.pairing.required == true' "$DEPLOY/config.json.new" > /dev/null && [ ! -s "$TOKEN_FILE" ]; then
  (umask 077 && "$src/zig-out/bin/localrouter" pair token "$LABEL-operator" --scopes read --config "$DEPLOY/config.json.new" --data "$DEPLOY/data" > "$TOKEN_FILE.tmp") &&
    mv "$TOKEN_FILE.tmp" "$TOKEN_FILE" || die "could not make the operator token ($TOKEN_FILE)"
fi
mkdir -p "$(dirname "$PLIST")"
render "$here/localrouter.plist.in" > "$DEPLOY/$LABEL.plist.new"
! grep -q '@[A-Z_]*@' "$DEPLOY/$LABEL.plist.new" || die "unrendered placeholders in the agent's plist"
plutil -lint -s "$DEPLOY/$LABEL.plist.new" || die "the agent's plist is not valid: $DEPLOY/$LABEL.plist.new"
loaded=0; [ "$(agent_state)" = "not loaded" ] || loaded=1
if [ $loaded = 1 ] && [ "$(readlink "$DEPLOY/current")" = "src/$commit" ] && cmp -s "$DEPLOY/config.json.new" "$DEPLOY/config.json" &&
   cmp -s "$DEPLOY/$LABEL.plist.new" "$PLIST"; then
  rm -f "$DEPLOY/config.json.new" "$DEPLOY/$LABEL.plist.new"   # nothing to change: a re-run leaves the agent alone
  echo "== unchanged: $LABEL already runs $commit with this config; not restarting"
  loaded=unchanged
fi
if [ $loaded = 1 ]; then
  wait_idle   # never under a live call
  echo "== stopping $LABEL to restart it onto $commit"
  agent_unload
fi
if [ $loaded != unchanged ]; then
  swap_in "src/$commit" "$DEPLOY/config.json.new" "$DEPLOY/$LABEL.plist.new"   # the replaced ones: *.prev (rollback.sh)
fi
if [ $loaded = 1 ]; then agent_up "$commit"; fi

# ---- 4. HTTPS for the tailnet
ts=$(tailscale_cli)
if [ "$HTTPS_PORT" != 0 ] && [ -n "$ts" ]; then
  "$ts" serve --bg --https="$HTTPS_PORT" "http://127.0.0.1:$PORT" > /dev/null 2>&1 ||
    echo "warning: tailscale serve --https=$HTTPS_PORT failed; reach the service with ssh -L $PORT:127.0.0.1:$PORT" >&2
fi
echo "installed $commit: $DEPLOY (config.json, current -> src/$commit), agent $LABEL ($PLIST), log $LOG"
pub=$(public_url || true)
echo "start it: deploy/mac/start.sh    URL: ${pub:+$pub, }loopback $URL"
