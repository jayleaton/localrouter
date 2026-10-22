#!/usr/bin/env bash
# The small steps the profile's manifest names beside each model's own scripts (deploy/profiles/*.json, `run` and
# `done`). Run by bootstrap.sh with its variables (PREFIX, J, REPO, ...); usable by hand with the same ones set.
#   steps.sh tool NAME            the host tool NAME from the manifest's host.tools: download (resumable) to $J/dl,
#                                 check its sha256, extract to $J/opt, link its binaries into $J/bin
#   steps.sh tool-ok NAME         true when $J/bin has NAME at the manifest's version
#   steps.sh file URL SHA256 DEST a single pinned file: download, check, then move into place
#   steps.sh stamp NAME FILE...   records that NAME finished for these inputs ($J/install/state/NAME: their sha256)
#   steps.sh stamped NAME FILE... true when NAME finished for exactly these inputs
#   steps.sh free GIB             true when MemAvailable is at least GIB (a GPU step's precondition); says what to stop
#   steps.sh qualified MARKER PINS a qualification marker that says "qualified": true and was made from PINS (as install.sh checks)
#   steps.sh drama3-config        $J/qualify-drama3/config.json: the deployed Drama 3 tool (deploy/jarvis/config.json.in)
#                                 with this box's paths, for its qualify.sh
set -euo pipefail
: "${J:?J (the prefix) is not set}" "${REPO:?REPO is not set}" "${PROFILE_FILE:?PROFILE_FILE is not set}"
die() { echo "steps.sh: $*" >&2; exit 1; }
sum_of() { cat "$@" | sha256sum | cut -c1-64; }

tool() {
  local t url sha top name f
  t=$(jq -c --arg n "$1" '.host.tools[$n] // empty' "$PROFILE_FILE")
  [ -n "$t" ] || die "no host tool '$1' in $PROFILE_FILE"
  url=$(jq -r .url <<< "$t") sha=$(jq -r .sha256 <<< "$t")
  name=$(basename "$url")
  mkdir -p "$J/dl" "$J/opt" "$J/bin"
  f=$J/dl/$name
  if [ ! -f "$f" ] || [ "$(sha256sum "$f" | cut -c1-64)" != "$sha" ]; then
    curl -fL --retry 5 -C - -o "$f.part" "$url" || curl -fL --retry 5 -o "$f.part" "$url"
    [ "$(sha256sum "$f.part" | cut -c1-64)" = "$sha" ] || { rm -f "$f.part"; die "$name: sha256 differs from the pin $sha (removed; run again)"; }
    mv "$f.part" "$f"
  fi
  top=$(tar -tf "$f" | awk -F/ 'NR == 1 {print $1}')   # the archive's one top directory (awk reads it all: no SIGPIPE)
  rm -rf "$J/opt/$top.tmp" && mkdir -p "$J/opt/$top.tmp"
  tar -xf "$f" -C "$J/opt/$top.tmp"
  rm -rf "$J/opt/$top" && mv "$J/opt/$top.tmp/$top" "$J/opt/$top" && rm -rf "$J/opt/$top.tmp"
  jq -r '.bin[]' <<< "$t" | while read -r b; do ln -sfn "$J/opt/$top/$b" "$J/bin/$(basename "$b")"; done
}

tool_ok() {
  local t
  t=$(jq -c --arg n "$1" '.host.tools[$n]' "$PROFILE_FILE")
  bash -c "$(jq -r .version_cmd <<< "$t")" 2> /dev/null | grep -qF "$(jq -r .version <<< "$t")"
}

file() {
  local url=$1 sha=$2 dest=$3
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 5 -o "$dest.part" "$url"
  [ "$(sha256sum "$dest.part" | cut -c1-64)" = "$sha" ] || { rm -f "$dest.part"; die "$(basename "$dest"): sha256 differs from the pin $sha"; }
  mv "$dest.part" "$dest"
}

stamp() { local n=$1; shift; mkdir -p "$J/install/state"; sum_of "$@" > "$J/install/state/$n"; }
stamped() { local n=$1; shift; [ -f "$J/install/state/$n" ] && [ "$(cat "$J/install/state/$n")" = "$(sum_of "$@")" ]; }

qualified() { [ -f "$1" ] && jq -e --arg p "$(sha256sum "$2" | cut -c1-64)" '.qualified == true and .pins_sha256 == $p' "$1" > /dev/null; }

free() {
  local have
  have=$(awk '/^MemAvailable:/{printf "%d", $2 / 1048576}' /proc/meminfo)
  [ "$have" -ge "$1" ] && return 0
  echo "steps.sh: this step needs $1 GiB free, MemAvailable is $have GiB. Stop what uses the GPU first (a running" >&2
  echo "  LocalRouter: systemctl --user stop ${UNIT:-localrouter-jarvis}; others: nvidia-smi --query-compute-apps=pid,name --format=csv)" >&2
  return 1
}

drama3_config() {
  local out=$J/qualify-drama3/config.json
  mkdir -p "$J/qualify-drama3"
  sed -e "s|@SRC@|$REPO|g" -e "s|@DRAMA_ROOT@|$DRAMA_ROOT|g" -e "s|@DRAMA_MARKER@|$DRAMA_MARKER|g" -e "s|@DRAMA_PORT@|$DRAMA_PORT|g" -e "s|@[A-Z_]*@|unused|g" \
      "$REPO/deploy/jarvis/config.json.in" |
    jq --arg lock "$J/qualify-drama3/gpu.lock" '{host: "127.0.0.1", gpu_lock: $lock, defaults: {text_to_speech: "fishaudio/drama-3"},
        tools: [.tools[] | select(.id == "fishaudio/drama-3")]}' > "$out.tmp"
  jq -e '.tools | length == 1' "$out.tmp" > /dev/null || die "no fishaudio/drama-3 tool in deploy/jarvis/config.json.in"
  mv "$out.tmp" "$out"
  echo "wrote $out"
}

cmd=${1:?usage: steps.sh tool|tool-ok|file|stamp|stamped|qualified|free|drama3-config ...}
shift
case $cmd in
  tool) tool "$@" ;;
  tool-ok) tool_ok "$@" ;;
  file) file "$@" ;;
  stamp) stamp "$@" ;;
  stamped) stamped "$@" ;;
  qualified) qualified "$@" ;;
  free) free "$@" ;;
  drama3-config) drama3_config ;;
  *) die "unknown step '$cmd'" ;;
esac
