#!/usr/bin/env bash
# Downloads the pinned TensorFold release and the Qwen3.8-27B checkpoint and DFlash2 drafter (pins.json), and checks
# every byte against the pins.
#   tools/chat/qwen38-27b/fetch.sh ROOT [--hf DIR]
# ROOT/downloads: the release archive for this host (uname -m). ROOT/tensorfold-1.1.4: the extracted archive, every
# file checked against the archive's own SHA256SUMS; ROOT/tensorfold-1.1.4.archive-sha256 records which archive.
# DIR (default ROOT/hf): a Hugging Face cache, models--Org--Name/{refs/main,snapshots/REV/...}, plain files.
# Idempotent and resumable: files with the pinned size and hash are kept, partial downloads (.part) resume. HF_TOKEN,
# if set, is sent to huggingface.co only (curl reads it from stdin, never argv) and is never printed.
# Needs curl, jq, tar, sha256sum, sha1sum. Nothing from the downloads is executed.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pins=$here/pins.json
usage="usage: fetch.sh ROOT [--hf DIR]"
root=${1:?$usage}
shift
hf=
while [ $# -gt 0 ]; do
  case $1 in
    --hf) hf=${2:?$usage}; shift 2 ;;
    *) echo "fetch.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "fetch.sh: $*" >&2; exit 1; }
for tool in curl jq tar sha256sum sha1sum; do
  command -v "$tool" >/dev/null || die "needs $tool"
done
mkdir -p "$root"
root=$(cd "$root" && pwd)
[ -n "$hf" ] || hf=$root/hf
mkdir -p "$hf"
hf=$(cd "$hf" && pwd)

sha256() { sha256sum "$1" | cut -d' ' -f1; }
git_sha1() { # size file: git's blob id
  { printf 'blob %d\0' "$1"; cat "$2"; } | sha1sum | cut -d' ' -f1
}
size_of() { stat -c %s "$1"; }
# kind digest size file: 0 when the file has the pinned size and hash.
matches() {
  [ -f "$4" ] && [ "$(size_of "$4")" = "$3" ] || return 1
  case $1 in
    sha256) [ "$(sha256 "$4")" = "$2" ] ;;
    git_sha1) [ "$(git_sha1 "$3" "$4")" = "$2" ] ;;
    *) return 1 ;;
  esac
}
# url out auth: download (resuming out) with curl; auth=hf sends HF_TOKEN.
download() {
  local args=(-fsSL --retry 5 --retry-delay 5 --connect-timeout 30 -C - -o "$2" "$1")
  if [ "$3" = hf ] && [ -n "${HF_TOKEN:-}" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" | curl -K - "${args[@]}"
  else
    curl "${args[@]}"
  fi
}
# url dest size kind digest auth: fetch to dest.part, check, move into place.
fetch_one() {
  local url=$1 dest=$2 size=$3 kind=$4 digest=$5 auth=$6 part
  if matches "$kind" "$digest" "$size" "$dest"; then
    return 0
  fi
  [ ! -e "$dest" ] || die "$dest exists with another size or hash than pinned ($kind $digest); move it away first"
  part=$dest.part
  mkdir -p "$(dirname "$dest")"
  # A .part longer than the file cannot be resumed; a complete one goes straight to the check.
  if [ -f "$part" ] && [ "$(size_of "$part")" -gt "$size" ]; then
    rm -f "$part"
  fi
  if [ ! -f "$part" ] || [ "$(size_of "$part")" -lt "$size" ]; then
    echo "fetch.sh: downloading $url ($size bytes)"
    download "$url" "$part" "$auth" || die "download of $url failed (rerun to resume)"
  fi
  if ! matches "$kind" "$digest" "$size" "$part"; then
    local got
    got=$(size_of "$part" 2>/dev/null || echo 0)
    rm -f "$part"
    die "$url: $got bytes, or its $kind, does not match the pin ($size bytes, $digest)"
  fi
  mv "$part" "$dest"
}

# The release archive for this host.
case $(uname -m) in
  aarch64) platform=linux-aarch64 ;;
  x86_64) platform=linux-x86_64 ;;
  *) die "no TensorFold archive pinned for $(uname -m)" ;;
esac
version=$(jq -er '.tensorfold.version' "$pins")
tfdir=$(jq -er '.tensorfold.dir' "$pins")
binary=$(jq -er '.tensorfold.binary' "$pins")
line=$(jq -er --arg p "$platform" \
  '.tensorfold.archives[$p] | "\(.url) \(.size) \(.sha256) \(.top_dir)"' "$pins")
read -r url asize asha top <<< "$line"
[ -n "$top" ] || die "pins.json has no complete archive entry for $platform"
archive=$root/downloads/$(basename "$url")
fetch_one "$url" "$archive" "$asize" sha256 "$asha" github
echo "fetch.sh: $(basename "$archive") verified (sha256 $asha)"

# Extracted once into ROOT/tensorfold-VERSION; an existing tree must come from the same archive and still match it.
dest=$root/$tfdir
stamp=$dest.archive-sha256
if [ -e "$dest" ]; then
  [ -f "$stamp" ] || die "$dest exists but $stamp does not: it was not installed by fetch.sh; move it away first"
  have=$(cat "$stamp")
  [ "$have" = "$asha" ] || die "$dest was extracted from archive sha256 $have, not the pinned $asha; move it away first"
else
  tmp=$(mktemp -d "$root/.tensorfold-extract.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT
  tar --no-same-owner -xzf "$archive" -C "$tmp"
  entries=$(cd "$tmp" && ls -A)
  [ "$entries" = "$top" ] || die "the archive's top level is '$entries', expected only '$top'"
  [ -d "$tmp/$top" ] || die "the archive's $top is not a directory"
  mv "$tmp/$top" "$dest"
  printf '%s\n' "$asha" > "$stamp"
  rmdir "$tmp"
  trap - EXIT
fi
[ -f "$dest/SHA256SUMS" ] || die "$dest/SHA256SUMS is missing"
(cd "$dest" && sha256sum --check --quiet --strict SHA256SUMS) || die "$dest does not match its SHA256SUMS"
[ -x "$dest/$binary" ] || die "$dest/$binary is not executable"
got=$(cat "$dest/VERSION")
[ "$got" = "$version" ] || die "$dest/VERSION says $got, pinned $version"
echo "fetch.sh: $dest verified against its SHA256SUMS (version $version, $platform)"

# The checkpoint and the drafter, into the Hugging Face cache layout.
for which in checkpoint drafter; do
  repo=$(jq -er --arg k "$which" '.[$k].repo' "$pins")
  rev=$(jq -er --arg k "$which" '.[$k].revision' "$pins")
  cache=$hf/models--${repo//\//--}
  snap=$cache/snapshots/$rev
  plan=$(jq -er --arg k "$which" '.[$k].files[]
    | "\(.path) \(.size) \(if .sha256 then "sha256 \(.sha256)" else "git_sha1 \(.git_sha1)" end)"' "$pins")
  n=0 bytes=0
  while read -r path size kind digest; do
    fetch_one "https://huggingface.co/$repo/resolve/$rev/$path" "$snap/$path" "$size" "$kind" "$digest" hf
    n=$((n + 1)) bytes=$((bytes + size))
  done <<< "$plan"
  mkdir -p "$cache/refs"
  printf '%s' "$rev" > "$cache/refs/main.tmp"
  mv "$cache/refs/main.tmp" "$cache/refs/main"
  echo "fetch.sh: $repo @ $rev: $n files, $bytes bytes verified in $snap"
done
echo "fetch.sh: done. Next: tools/chat/qwen38-27b/setup.sh $root --hf $hf"
