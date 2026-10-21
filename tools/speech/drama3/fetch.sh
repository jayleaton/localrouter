#!/usr/bin/env bash
# Fetches the pinned Fish Audio Drama 3 sources (tools/speech/drama3/pins.json) into ROOT and checks every sha256.
#   tools/speech/drama3/fetch.sh ROOT [--no-weights] [--check]
# ROOT/drama-tensorfold is plotarmordev/drama-tensorfold at the pinned commit: the launcher (recipe/pilot.py), the
# engine build script and the patch series. ROOT/drama-3 is the fishaudio/drama-3 checkpoint at the pinned revision
# (41.5 GB). Files already present with the pinned size and hash are kept. --no-weights stops after the upstream
# checkout. --check fetches and changes nothing: it verifies what is there and fails on any difference (build.sh and
# qualify.sh use it). Nothing is executed from the downloads, and nothing here accepts the weights' licence.
set -euo pipefail
export LC_ALL=C # glob and sort order is byte order, the order build-engine.sh applies the patches in
here=$(cd "$(dirname "$0")" && pwd)
root=${1:?usage: fetch.sh ROOT [--no-weights] [--check]}
shift
weights=1 check=0
while [ $# -gt 0 ]; do
  case $1 in
    --no-weights) weights=0; shift ;;
    --check) check=1; shift ;;
    *) echo "fetch.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "fetch.sh: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
[ "$check" = 1 ] || mkdir -p "$root"
[ -d "$root" ] || die "$root is not a directory"
root=$(cd "$root" && pwd)

# One line a pin: repo URL COMMIT DIR | file PATH SHA256 | patch NAME SHA256 | series COUNT SHA256 |
# ckpt REPO REVISION DIR PATH SIZE SHA256.
plan=$(python3 -I - "$here/pins.json" <<'EOF'
import json, sys
pins = json.load(open(sys.argv[1]))
u, p, c = pins["upstream"], pins["patches"], pins["checkpoint"]
print("repo", u["repository"], u["commit"], u["dir"])
for path, digest in sorted(u["files"].items()):
    print("file", path, digest)
for name, digest in sorted(p["files"].items()):
    print("patch", p["dir"] + "/" + name, digest)
print("series", p["count"], p["series_sha256"], p["dir"])
for path, (size, digest) in sorted(c["files"].items()):
    print("ckpt", c["repo"], c["revision"], c["dir"], path, size, digest)
EOF
)
pinned() { # kind: the plan's lines of that kind, without the kind
  local kind line
  kind=$1
  while read -r line; do
    case $line in "$kind "*) echo "${line#"$kind "}" ;; esac
  done <<< "$plan"
}

# The upstream checkout: cloned, or an existing clean clone moved to the pinned commit (--check only looks).
read -r url commit updir <<< "$(pinned repo)"
up=$root/$updir
if [ -e "$up" ]; then
  git -C "$up" rev-parse --git-dir >/dev/null 2>&1 || die "$up exists and is not a git checkout"
  [ -z "$(git -C "$up" status --porcelain --untracked-files=no)" ] || die "$up has local changes; refusing to use it"
elif [ "$check" = 1 ]; then
  die "$up is missing; run fetch.sh ROOT first"
else
  git clone --quiet "$url" "$up"
fi
head=$(git -C "$up" rev-parse HEAD)
if [ "$head" != "$commit" ]; then
  [ "$check" = 0 ] || die "$up is at $head, pinned $commit"
  git -C "$up" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$up" fetch --quiet origin "$commit"
  git -C "$up" -c advice.detachedHead=false checkout --quiet --detach "$commit"
  [ "$(git -C "$up" rev-parse HEAD)" = "$commit" ] || die "$up did not check out $commit"
fi

# Every pinned upstream file and patch by sha256, the exact set of patches, and the series as one hash.
bad=0
while read -r path digest; do
  got=$(sha "$up/$path" 2>/dev/null || echo missing)
  [ "$got" = "$digest" ] || { echo "fetch.sh: $updir/$path sha256 $got, pinned $digest" >&2; bad=1; }
done <<< "$(pinned file; pinned patch)"
read -r count series pdir <<< "$(pinned series)"
want=$(pinned patch | while read -r path _; do basename "$path"; done)
have=$(cd "$up/$pdir" && for f in *.patch; do [ -e "$f" ] && echo "$f"; done)
[ "$have" = "$want" ] || { echo "fetch.sh: $updir/$pdir holds other *.patch files than the $count pinned" >&2; bad=1; }
[ "$(grep -c . <<< "$want")" = "$count" ] || { echo "fetch.sh: pins.json lists other than $count patches" >&2; bad=1; }
got=$(cat "$up/$pdir"/*.patch 2>/dev/null | sha256sum | cut -d' ' -f1 || echo missing)
[ "$got" = "$series" ] || { echo "fetch.sh: patch series sha256 $got, pinned $series" >&2; bad=1; }
[ "$bad" = 0 ] || die "$up does not match pins.json; refusing it"
echo "fetch.sh: $updir at $commit verified ($count patches, series $series)"
[ "$weights" = 1 ] || exit 0

if [ "$check" = 0 ]; then
  cat <<'EOF'
fetch.sh: the fishaudio/drama-3 weights and codec are licensed under the Fish Audio Community License, Version 1.0
  (FACL v1.0): https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE
  Built with Fish Audio.
  Keep this attribution with any copy: "These Fish Audio Materials are licensed under the Fish Audio Community
  License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved."
  Section III(c) grants no licence for a Hosted Service (third parties generating speech from text or audio they
  supply, for example through an API) without a separate written licence from Fish Audio: keep LocalRouter on
  loopback. Read the licence; downloading the files does not accept it on anyone's behalf.
EOF
fi

# The checkpoint, file by file: downloaded to .part, checked, then moved into place.
n=0
while read -r repo rev dir path size digest; do
  dest=$root/$dir/$path
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ] && [ "$(sha "$dest")" = "$digest" ]; then
    n=$((n + 1)); continue
  fi
  if [ "$check" = 1 ]; then
    echo "fetch.sh: $dir/$path is missing or differs from its pinned size $size and sha256 $digest" >&2
    bad=1; continue
  fi
  mkdir -p "$(dirname "$dest")"
  echo "fetch.sh: downloading $repo/$path ($size bytes)"
  # A partial .part resumes; a complete one goes straight to the hash check.
  if [ ! -f "$dest.part" ] || [ "$(stat -c %s "$dest.part")" -lt "$size" ]; then
    curl -fsSL --retry 3 -C - -o "$dest.part" "https://huggingface.co/$repo/resolve/$rev/$path"
  fi
  got=$(sha "$dest.part")
  if [ "$got" != "$digest" ]; then
    rm -f "$dest.part"
    die "$repo/$path sha256 $got, pinned $digest"
  fi
  mv "$dest.part" "$dest"
  n=$((n + 1))
done <<< "$(pinned ckpt)"
[ "$bad" = 0 ] || die "$root does not match pins.json"
read -r _ _ ckdir _ <<< "$(pinned ckpt)" # the first line's dir
# Upstream's own check of the same files.
(cd "$root/$ckdir" && sha256sum --check --quiet --strict "$up/recipe/checksums/checkpoint.sha256") ||
  die "$ckdir fails $updir/recipe/checksums/checkpoint.sha256"
echo "fetch.sh: $n checkpoint files verified under $root/$ckdir"
