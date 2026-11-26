#!/usr/bin/env bash
# Fetches everything the Inkling-2-Mini tool runs on a GB10, each part against pins.json:
#   tools/chat/inkling2-mini/fetch.sh ROOT [--models DIR]
# 1. the recipe (jayleaton/inkling-2-mini-tensorfold) at its pinned commit, in ROOT/recipe;
# 2. the checkpoint and the DFlash2 drafter into DIR/Inkling-2-Mini and DIR/Inkling-2-Mini-DFlash2 (DIR default
#    ROOT/models; the recipe's scripts/fetch.sh, resumable), then every file by size and sha256 (hf_pinned.py verify).
#    A DIR that already holds them (the recipe's ~/models/hf) is only verified;
# 3. the sm_121 image, built by the recipe's scripts/prepare.sh as pins.json's image tag (about 40 GB under Docker's
#    root, the first build takes a while), and checked by its tf.patches label.
# Idempotent: a part that is already there and checks out is kept. HF_TOKEN is sent when set (both repos are public).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pins=$here/pins.json
usage="usage: fetch.sh ROOT [--models DIR]"
root=${1:?$usage}
shift
models=
while [ $# -gt 0 ]; do
  case $1 in
    --models) models=${2:?$usage}; shift 2 ;;
    *) echo "fetch.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "fetch.sh: $*" >&2; exit 1; }
for t in git jq docker python3; do command -v $t > /dev/null || die "needs $t"; done
mkdir -p "$root"
root=$(cd "$root" && pwd)
[ -n "$models" ] || models=$root/models
mkdir -p "$models"
models=$(cd "$models" && pwd)
pin() { jq -er "$1" "$pins"; }
manifest() { # PART: its hf_pinned manifest, as a file
  jq ".$1" "$pins" > "$root/$1.manifest.json"
  echo "$root/$1.manifest.json"
}

# 1. the recipe
repo=$(pin .recipe.repo) commit=$(pin .recipe.commit)
recipe=$root/recipe
[ -d "$recipe/.git" ] || git clone -q "$repo" "$recipe"
git -C "$recipe" cat-file -e "$commit^{commit}" 2> /dev/null || git -C "$recipe" fetch -q origin "$commit"
git -C "$recipe" -c advice.detachedHead=false checkout -q --detach "$commit"
[ "$(git -C "$recipe" rev-parse HEAD)" = "$commit" ] || die "$recipe is not at $commit"
[ -z "$(git -C "$recipe" status --porcelain --untracked-files=no)" ] || die "$recipe has local changes: move it away and run again"
echo "fetch.sh: recipe $repo at ${commit:0:12}"

# 2. the checkpoint and the drafter
ck=$models/Inkling-2-Mini dr=$models/Inkling-2-Mini-DFlash2
hfp=(python3 -I "$here/../../install/hf_pinned.py")
ckm=$(manifest checkpoint) drm=$(manifest drafter)
if ! "${hfp[@]}" verify "$ckm" "$ck" --sizes 2> /dev/null; then
  HF_REVISION=$(pin .checkpoint.revision) bash "$recipe/scripts/fetch.sh" "$ck"
fi
if ! "${hfp[@]}" verify "$drm" "$dr" --sizes 2> /dev/null; then
  DRAFTER_REV=$(pin .drafter.revision) bash "$recipe/scripts/fetch.sh" drafter "$dr"
fi
"${hfp[@]}" verify "$ckm" "$ck" || die "$ck differs from the pins (the lines above): delete the named files and run again"
"${hfp[@]}" verify "$drm" "$dr" || die "$dr differs from the pins (the lines above): delete the named files and run again"
echo "fetch.sh: $(pin .checkpoint.repo)@$(pin .checkpoint.revision | cut -c1-12) in $ck, $(pin .drafter.repo)@$(pin .drafter.revision | cut -c1-12) in $dr: verified"

# 3. the image
tag=$(pin .image.tag) label=$(pin '.image["tf.patches"]')
mkdir -p "$root/kernel-cache" "$root/state" "$root/logs"
IMAGE=$tag MODEL_DIR=$ck DRAFTER_DIR=$dr MODEL_FETCH=0 DRAFTER_FETCH=0 KERNEL_CACHE=$root/kernel-cache \
  STATE_DIR=$root/state LOG_DIR=$root/logs bash "$recipe/scripts/prepare.sh"
got=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}} {{index .Config.Labels "tf.kernels"}}' "$tag")
[ "$got" = "$label $(pin '.image["tf.kernels"]')" ] || die "$tag has labels '$got', pinned '$label $(pin '.image["tf.kernels"]')'"
echo "fetch.sh: image $tag ($(docker image inspect -f '{{.Id}}' "$tag")), tf.patches $label"
echo "fetch.sh: done. Next: tools/chat/inkling2-mini/setup.sh $root --models $models"
