#!/usr/bin/env bash
# Fetches and builds everything the Inkling-2-Mini tool runs, each part against pins.json:
#   tools/chat/inkling2-mini/fetch.sh ROOT --engine metal|cuda [--models DIR] [--hf DIR]
# 1. the recipe's public tree in ROOT/recipe at its pinned commit and clean (a public repository: no credentials).
#    It carries the recipe kit's release in kit/: its sha256 and the recipe's [kit] rev must be the pinned ones, and
#    kit/recipe checks the release against kit/SHA256SUMS before every run. A host without GitHub points git's
#    url.<mirror>.insteadOf at mirrors of the recipe and TensorFold; the commits stay pinned;
# 2. the checkpoint into DIR/Inkling-2-Mini (DIR default ROOT/models), then every file by size and sha256
#    (tools/install/hf_pinned.py verify). A DIR that already holds it is only verified;
# 3. metal only: the DFlash2 drafter into the Hugging Face cache layout under DIR (--hf; default the shared cache,
#    ~/.cache/huggingface/hub): models--incoai--Inkling-2-Mini-DFlash2/{refs/main,snapshots/REV/...}, where the
#    server resolves the recipe's --drafter incoai/Inkling-2-Mini-DFlash2 (HF_HUB_OFFLINE=1), and verified the same
#    way. cuda drafts from the checkpoint's own mtp.safetensors (DSpark) and fetches no drafter;
# 4. `recipe prepare --engine ENGINE ROOT/engine` from ROOT/recipe: TensorFold at the recipe's pin with its patch
#    series, built (metal: zig and xcrun, DEVELOPER_DIR; cuda: zig and nvcc, NVCC default /usr/local/cuda/bin/nvcc),
#    plus libtfgrammar from xgrammar, which the kit clones from GitHub at its pinned tag unless XGRAMMAR=DIR;
# 5. `recipe serve PROFILE ROOT/engine CHECKPOINT --dry-run` (mac for metal, spark for cuda): the server argv the
#    recipe renders, saved as ROOT/serve-PROFILE.json. check_args.py holds a tool config's argv to it.
# Idempotent and resumable: a checkout at its commit, a verified checkpoint (ROOT/PART.verified records the manifest
# it passed; a rerun checks sizes only) and a fetched engine tree are kept; partial downloads (.part) resume; the kit
# rebuilds incrementally. HF_TOKEN, if set, is sent to huggingface.co only (curl reads it from stdin, never argv).
# Needs git, jq, curl, python3 3.11 or newer, zig 0.17.0. Nothing downloaded runs but the recipe, the kit release
# it carries and the engine they build.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pins=$here/pins.json
usage="usage: fetch.sh ROOT --engine metal|cuda [--models DIR] [--hf DIR]"
root=${1:?$usage}
shift
engine='' models='' hf=''
while [ $# -gt 0 ]; do
  case $1 in
    --engine) engine=${2:?$usage}; shift 2 ;;
    --models) models=${2:?$usage}; shift 2 ;;
    --hf) hf=${2:?$usage}; shift 2 ;;
    *) echo "fetch.sh: unknown argument $1" >&2; echo "$usage" >&2; exit 2 ;;
  esac
done
die() { echo "fetch.sh: $*" >&2; exit 1; }
case $engine in
  metal) [ "$(uname -s) $(uname -m)" = "Darwin arm64" ] || die "--engine metal needs an Apple Silicon Mac, this is $(uname -s) $(uname -m)" ;;
  cuda) [ "$(uname -s)" = Linux ] || die "--engine cuda needs Linux, this is $(uname -s)" ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
for t in git jq curl python3 zig; do command -v $t > /dev/null || die "needs $t"; done
python3 -I -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "needs python3 3.11 or newer (the kit reads TOML)"
pin() { jq -er "$1" "$pins"; }
profile=$(pin ".engines.$engine.profile")
mkdir -p "$root"
root=$(cd "$root" && pwd -P)   # physical, as the kit resolves its paths
[ -n "$models" ] || models=$root/models
mkdir -p "$models"
models=$(cd "$models" && pwd -P)
if [ "$engine" = metal ]; then
  [ -n "$hf" ] || hf=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
  mkdir -p "$hf"
  hf=$(cd "$hf" && pwd -P)
fi

# the size of the file a link names (a Hugging Face cache links snapshots into blobs/); GNU stat, else BSD (macOS)
size_of() { stat -L -c %s "$1" 2> /dev/null || stat -L -f %z "$1"; }
sha256() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# 1. the recipe, with the kit release it carries
recipe=$root/recipe kit=$root/recipe/kit/recipe
url=$(pin .recipe.repo) recipe_commit=$(pin .recipe.commit)
if [ ! -d "$recipe/.git" ]; then
  [ ! -e "$recipe" ] || die "$recipe exists but is not a git checkout: move it away first"
  git clone -q "$url" "$recipe" || die "git clone $url failed (the network)"
fi
if ! git -C "$recipe" cat-file -e "$recipe_commit^{commit}" 2> /dev/null; then
  git -C "$recipe" fetch -q "$url" "$recipe_commit" || die "git fetch $url $recipe_commit failed"
fi
[ -z "$(git -C "$recipe" status --porcelain --untracked-files=no 2> /dev/null)" ] ||
  die "$recipe has local changes: move it away and run again"
git -C "$recipe" -c advice.detachedHead=false checkout -q --detach "$recipe_commit"
[ "$(git -C "$recipe" rev-parse HEAD)" = "$recipe_commit" ] || die "$recipe is not at $recipe_commit"
echo "fetch.sh: recipe $url at ${recipe_commit:0:12}"
kit_commit=$(pin .kit.commit) kit_sha256=$(pin .kit.sha256)
want=$(python3 -I -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["kit"]["rev"])' "$recipe/recipe.toml") ||
  die "$recipe/recipe.toml has no [kit] rev"
[ "$want" = "$kit_commit" ] || die "the recipe at ${recipe_commit:0:12} names kit $want, pins.json pins $kit_commit: update pins.json"
[ -x "$kit" ] && [ "$(cut -d' ' -f1 "$recipe/kit/SHA256SUMS")" = "$kit_sha256" ] ||
  die "$recipe/kit holds no kit release with sha256 $kit_sha256: is $url the published recipe?"
echo "fetch.sh: kit release ${kit_commit:0:12} (sha256 ${kit_sha256:0:12}) in $recipe/kit"

# 2 and 3. the weights
hfp=(python3 -I "$here/../../install/hf_pinned.py")
# download URL OUT: resumes OUT with curl; HF_TOKEN goes in a config on stdin
download() {
  local args=(-fsSL --retry 5 --retry-delay 5 --connect-timeout 30 -C - -o "$2" "$1")
  if [ -n "${HF_TOKEN:-}" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" | curl -K - "${args[@]}"
  else
    curl "${args[@]}"
  fi
}
# part PART DIR: the manifest pins.json has for PART, fetched into DIR (each file by size, then all by sha256)
part() {
  local name=$1 dir=$2 repo rev manifest stamp path size dest have
  repo=$(pin ".$name.repo") rev=$(pin ".$name.revision")
  manifest=$root/$name.manifest.json stamp=$root/$name.verified
  jq ".$name" "$pins" > "$manifest.tmp" && mv "$manifest.tmp" "$manifest"
  have="$(sha256 "$manifest") $dir"
  if [ "$(cat "$stamp" 2> /dev/null)" = "$have" ] && "${hfp[@]}" verify "$manifest" "$dir" --sizes 2> /dev/null; then
    echo "fetch.sh: $repo @ ${rev:0:12} in $dir: verified before, sizes unchanged"
    return 0
  fi
  rm -f "$stamp"
  while read -r path size; do
    dest=$dir/$path
    if [ -f "$dest" ]; then
      [ "$(size_of "$dest")" = "$size" ] ||
        die "$dest exists with $(size_of "$dest") bytes, pinned $size: move it away first"
      continue
    fi
    mkdir -p "$(dirname "$dest")"
    # A .part longer than the file cannot be resumed.
    if [ -f "$dest.part" ] && [ "$(size_of "$dest.part")" -gt "$size" ]; then rm -f "$dest.part"; fi
    if [ ! -f "$dest.part" ] || [ "$(size_of "$dest.part")" -lt "$size" ]; then
      echo "fetch.sh: downloading $repo/$path ($size bytes)"
      download "https://huggingface.co/$repo/resolve/$rev/$path" "$dest.part" ||
        die "download of $repo/$path failed (rerun to resume; HF_TOKEN if huggingface.co asks for one)"
    fi
    [ "$(size_of "$dest.part")" = "$size" ] || die "$dest.part has $(size_of "$dest.part") bytes, pinned $size: delete it and run again"
    mv "$dest.part" "$dest"
  done < <(jq -r --arg k "$name" '.[$k].files | to_entries[] | "\(.key) \(.value[0])"' "$pins")
  "${hfp[@]}" verify "$manifest" "$dir" || die "$dir differs from the pins (the lines above): delete the named files and run again"
  printf '%s\n' "$have" > "$stamp"
  echo "fetch.sh: $repo @ ${rev:0:12}: $(jq '.files | length' "$manifest") files verified in $dir"
}
ck=$models/Inkling-2-Mini
part checkpoint "$ck"
if [ "$engine" = metal ]; then
  repo=$(pin .drafter.repo) rev=$(pin .drafter.revision)
  [ "$repo" = "$(pin .engines.metal.drafter)" ] || die "pins.json: the drafter is $repo, the mac profile's $(pin .engines.metal.drafter)"
  cache=$hf/models--${repo//\//--}
  part drafter "$cache/snapshots/$rev"
  mkdir -p "$cache/refs"
  printf '%s' "$rev" > "$cache/refs/main.tmp"
  mv "$cache/refs/main.tmp" "$cache/refs/main"
fi

# 4. the engine
(cd "$recipe" && "$kit" prepare --engine "$engine" "$root/engine") ||
  die "recipe prepare --engine $engine failed (above): a missing toolchain, the network, or a build error"
bin=$root/engine/$(pin ".engines.$engine.binary")
[ -x "$bin" ] || die "recipe prepare left no $bin"
[ -f "$root/engine/$(pin ".engines.$engine.grammar_lib")" ] ||
  die "recipe prepare left no $(pin ".engines.$engine.grammar_lib"): tool_choice required/named and response_format need it"

# 5. the argv the recipe serves
(cd "$recipe" && "$kit" serve "$profile" "$root/engine" "$ck" --dry-run) > "$root/serve-$profile.json.tmp" ||
  die "recipe serve $profile --dry-run failed (above)"
jq -e --arg bin "$bin" --arg ck "$ck" 'length == 1 and .[0].argv[0] == $bin and .[0].argv[1:3] == ["serve", $ck]' \
  "$root/serve-$profile.json.tmp" > /dev/null || die "recipe serve $profile --dry-run printed another launch: $root/serve-$profile.json.tmp"
mv "$root/serve-$profile.json.tmp" "$root/serve-$profile.json"
echo "fetch.sh: $root/serve-$profile.json: $(jq -r '.[0].argv | join(" ")' "$root/serve-$profile.json")"
echo "fetch.sh: done ($engine, the recipe's $profile profile)"
