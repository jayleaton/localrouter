#!/usr/bin/env bash
# Builds the arm64 LocalRouter image from a clean, committed checkout on a build machine, and exports it as one
# compressed file a GPU node loads without building anything:
#   bash tools/package/build-image.sh [OUT_DIR]          (default ~/localrouter-dist)
# Writes OUT_DIR/localrouter-<commit>-arm64.tar.zst and OUT_DIR/localrouter-<commit>-arm64.json: the source commit,
# the image id, the file's SHA-256 and size, the build arguments and the Dockerfile's SHA-256. install scripts check
# the file against that receipt before `docker load`.
# It refuses to run on an arm64 host (a GPU node such as a DGX Spark builds nothing: its memory belongs to the models)
# unless LR_BUILD_ON_ARM=1, and refuses a checkout with uncommitted changes (the receipt names a commit).
# Env: BUILD_ARGS ("KEY=VALUE ..." passed as --build-arg, e.g. an engine selection the Dockerfile takes),
#      DOCKERFILE (default docker/Dockerfile), LR_ALLOW_DIRTY=1 (a dev build; the receipt records "dirty": true),
#      SRC (the checkout to build; default this one).
set -euo pipefail
REPO=$(cd "${SRC:-$(dirname "$0")/../..}" && pwd)
OUT=${1:-$HOME/localrouter-dist}
DOCKERFILE=${DOCKERFILE:-$REPO/docker/Dockerfile}
die() { echo "build-image: $*" >&2; exit 1; }

[[ $(uname -m) != aarch64 || ${LR_BUILD_ON_ARM:-0} == 1 ]] || die "this is an arm64 host; build on the build machine and copy the file (LR_BUILD_ON_ARM=1 overrides)"
command -v docker > /dev/null || die "no docker"
docker buildx version > /dev/null 2>&1 || die "no docker buildx"
command -v zstd > /dev/null || die "no zstd"
cd "$REPO"
commit=$(git rev-parse HEAD)
dirty=false
if [[ -n $(git status --porcelain) ]]; then
    [[ ${LR_ALLOW_DIRTY:-0} == 1 ]] || die "uncommitted changes; commit them (or LR_ALLOW_DIRTY=1 for a dev build)"
    dirty=true
fi
tag=localrouter:${commit:0:12}
args=()
for kv in ${BUILD_ARGS:-}; do args+=(--build-arg "$kv"); done
mkdir -p "$OUT"
base=$OUT/localrouter-${commit:0:12}-arm64
echo "[build-image] $tag from $commit${BUILD_ARGS:+ ($BUILD_ARGS)}"
docker buildx build -f "$DOCKERFILE" --platform linux/arm64 "${args[@]}" -t "$tag" --load "$REPO"
id=$(docker image inspect -f '{{.Id}}' "$tag")
echo "[build-image] export $id"
docker save "$tag" | zstd -T0 -10 -q -o "$base.tar.zst.part"
mv "$base.tar.zst.part" "$base.tar.zst"
sha=$(sha256sum "$base.tar.zst" | cut -d' ' -f1)
python3 -I - "$base.json" "$commit" "$dirty" "$tag" "$id" "$base.tar.zst" "$sha" "$(sha256sum "$DOCKERFILE" | cut -d' ' -f1)" "${BUILD_ARGS:-}" <<'PY'
import json, os, sys
out, commit, dirty, tag, image_id, path, sha, dockerfile, build_args = sys.argv[1:]
json.dump({"commit": commit, "dirty": dirty == "true", "tag": tag, "image_id": image_id, "platform": "linux/arm64",
           "file": os.path.basename(path), "file_bytes": os.path.getsize(path), "file_sha256": sha,
           "dockerfile_sha256": dockerfile, "build_args": build_args.split()}, open(out, "w"), indent=1)
PY
echo "[build-image] $base.tar.zst ($(du -h "$base.tar.zst" | cut -f1)), receipt $base.json"
