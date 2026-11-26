#!/usr/bin/env bash
# Checks what fetch.sh made and writes the daemon config for the Inkling-2-Mini tool:
#   tools/chat/inkling2-mini/setup.sh ROOT [--models DIR] [--marker PATH] [--gpu-lock PATH]
# ROOT and DIR as for fetch.sh (DIR default ROOT/models). PATH (the tool's "qualification" marker) defaults to
# ROOT/inkling2-mini-qualification.json. --gpu-lock "" leaves the host lock out (default: the example's).
# The image tagged in pins.json must carry the pinned tf.patches label; the config runs it by its local id, so a later
# build that moves the tag does not change what the tool runs. tensorfold-native then parses the config's serve
# arguments (no GPU, no network: a server that never starts). Writes ROOT/config.json (config.example.json with this
# host's paths and image id) and prints it. Needs jq and docker.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../../.." && pwd)
pins=$here/pins.json
usage="usage: setup.sh ROOT [--models DIR] [--marker PATH] [--gpu-lock PATH]"
root=${1:?$usage}
shift
models= marker= lock_set=0 lock=
while [ $# -gt 0 ]; do
  case $1 in
    --models) models=${2:?$usage}; shift 2 ;;
    --marker) marker=${2:?$usage}; shift 2 ;;
    --gpu-lock) [ $# -ge 2 ] || { echo "$usage" >&2; exit 2; }; lock=$2 lock_set=1; shift 2 ;;
    *) echo "setup.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "setup.sh: $*" >&2; exit 1; }
for t in jq docker python3; do command -v $t > /dev/null || die "needs $t"; done
root=$(cd "$root" && pwd)
[ -n "$models" ] || models=$root/models
models=$(cd "$models" && pwd)
[ -n "$marker" ] || marker=$root/inkling2-mini-qualification.json
case $marker in /*) ;; *) die "--marker must be an absolute path" ;; esac

hfp=(python3 -I "$here/../../install/hf_pinned.py")
for part in checkpoint drafter; do
  [ -f "$root/$part.manifest.json" ] || die "no $root/$part.manifest.json: run fetch.sh $root first"
done
"${hfp[@]}" verify "$root/checkpoint.manifest.json" "$models/Inkling-2-Mini" --sizes || die "the checkpoint is not in $models: run fetch.sh"
"${hfp[@]}" verify "$root/drafter.manifest.json" "$models/Inkling-2-Mini-DFlash2" --sizes || die "the drafter is not in $models: run fetch.sh"

tag=$(jq -er .image.tag "$pins") label=$(jq -er '.image["tf.patches"]' "$pins")
id=$(docker image inspect -f '{{.Id}}' "$tag" 2> /dev/null) || die "no image $tag: run fetch.sh $root"
got=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$id")
[ "$got" = "$label" ] || die "$tag ($id) has tf.patches '$got', pinned '$label': run fetch.sh $root"
echo "setup.sh: image $tag is $id (tf.patches $label)"
echo "image $tag $id tf.patches $label recipe $(git -C "$root/recipe" rev-parse HEAD 2> /dev/null)" > "$root/image.txt"
mkdir -p "$root/kernel-cache/$label" "$root/sessions"

filter='walk(if type == "string" then gsub("/opt/localrouter/inkling/models"; $models) | gsub("/opt/localrouter/inkling"; $root)
    | gsub("/opt/localrouter/src"; $src) | gsub("sha256:0{64}"; $id) else . end)
  | .tools[0].qualification = $marker'
[ "$lock_set" = 0 ] || filter="$filter | if \$lock == \"\" then del(.gpu_lock) else .gpu_lock = \$lock end"
jq --arg models "$models" --arg root "$root" --arg src "$src" --arg id "$id" --arg marker "$marker" --arg lock "$lock" \
  "$filter" "$here/config.example.json" > "$root/config.json.tmp"

# The serve arguments, as the config passes them, parsed by this image's binary: a model path that does not exist
# makes it stop right after parsing, so an unknown or malformed flag shows as a usage error (exit 2).
mapfile -t serve < <(jq -r '.tools[0].options.server | .[index("tensorfold-native") + 2:][]
  | gsub("\\{id\\}"; "Inkling-2-Mini") | gsub("\\{port\\}"; "8888")' "$root/config.json.tmp")
code=0
docker run --rm --network none --entrypoint "$(jq -er .image.binary "$pins")" "$id" serve /nonexistent/argument-check \
  "${serve[@]}" > /dev/null 2> "$root/args.err" || code=$?
if [ $code = 2 ] || ! grep -q 'neither a directory' "$root/args.err"; then
  cat "$root/args.err" >&2
  die "tensorfold-native serve refuses the config's arguments (exit $code)"
fi
echo "setup.sh: tensorfold-native accepts the serve arguments"
mv "$root/config.json.tmp" "$root/config.json"
cat <<EOF
setup.sh: wrote $root/config.json. Merge its "gpu_lock" and "tools" entries into the daemon's config:
$(cat "$root/config.json")
setup.sh: the tool stays refused (503, not GPU-qualified) until qualify.sh writes $marker:
  tools/chat/inkling2-mini/qualify.sh $root /path/to/localrouter OUT $marker
EOF
