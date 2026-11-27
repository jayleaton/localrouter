#!/usr/bin/env bash
# Checks what fetch.sh --engine cuda made and writes the daemon config for the Inkling-2-Mini tool on a GB10:
#   tools/chat/inkling2-mini/setup.sh ROOT [--models DIR] [--marker PATH] [--gpu-lock PATH]
# ROOT and DIR as for fetch.sh (DIR default ROOT/models). PATH (the tool's "qualification" marker) defaults to
# ROOT/inkling2-mini-qualification.json. --gpu-lock "" leaves the host lock out (default: the example's).
# The checkpoint against the pins (sizes, after fetch.sh's full check; else every sha256), the engine recipe prepare
# built (it runs here: --version, ROOT/tensorfold-version.txt) and its grammar library; then ROOT/config.json
# (config.example.json with this host's paths; conversation states in ROOT/sessions), whose server argv must be the
# one the recipe's spark profile renders (ROOT/serve-spark.json, check_args.py). Prints the config. No GPU: the
# model only loads in qualify.sh. Needs jq and python3.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
pins=$here/pins.json
usage="usage: setup.sh ROOT [--models DIR] [--marker PATH] [--gpu-lock PATH]"
root=${1:?$usage}
shift
models='' marker='' lock_set=0 lock=''
while [ $# -gt 0 ]; do
  case $1 in
    --models) models=${2:?$usage}; shift 2 ;;
    --marker) marker=${2:?$usage}; shift 2 ;;
    --gpu-lock) [ $# -ge 2 ] || { echo "$usage" >&2; exit 2; }; lock=$2 lock_set=1; shift 2 ;;
    *) echo "setup.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "setup.sh: $*" >&2; exit 1; }
for t in jq python3 timeout; do command -v $t > /dev/null || die "needs $t"; done
[ -d "$root" ] || die "no $root: run fetch.sh $root --engine cuda first"
root=$(cd "$root" && pwd -P)
[ -n "$models" ] || models=$root/models
models=$(cd "$models" && pwd -P)
[ -n "$marker" ] || marker=$root/inkling2-mini-qualification.json
case $marker in /*) ;; *) die "--marker must be an absolute path" ;; esac
sha256() { sha256sum "$1" | cut -d' ' -f1; }

# The checkpoint: fetch.sh's full check stands while the manifest and the sizes do.
ck=$models/Inkling-2-Mini manifest=$root/checkpoint.manifest.json
[ -f "$manifest" ] || die "no $manifest: run fetch.sh $root --engine cuda first"
[ "$(jq -S .checkpoint "$pins")" = "$(jq -S . "$manifest")" ] || die "$manifest is not this tree's pins: run fetch.sh again"
hfp=(python3 -I "$here/../../install/hf_pinned.py")
if [ "$(cat "$root/checkpoint.verified" 2> /dev/null)" = "$(sha256 "$manifest") $ck" ]; then
  "${hfp[@]}" verify "$manifest" "$ck" --sizes || die "the checkpoint in $ck changed since fetch.sh verified it: run fetch.sh"
else
  "${hfp[@]}" verify "$manifest" "$ck" || die "the checkpoint in $ck differs from the pins: run fetch.sh"
fi
echo "setup.sh: $(jq -r .checkpoint.repo "$pins")@$(jq -r .checkpoint.revision "$pins" | cut -c1-12) in $ck: verified"

# The engine.
bin=$root/engine/$(jq -er .engines.cuda.binary "$pins")
grammar=$root/engine/$(jq -er .engines.cuda.grammar_lib "$pins")
serve=$root/serve-$(jq -er .engines.cuda.profile "$pins").json
[ -x "$bin" ] || die "$bin is missing: run fetch.sh $root --engine cuda first"
[ -f "$grammar" ] || die "$grammar is missing: tool_choice required/named and response_format need libtfgrammar"
[ -f "$serve" ] || die "no $serve: run fetch.sh $root --engine cuda first"
timeout 60 "$bin" --version > "$root/tensorfold-version.txt" 2>&1 ||
  { cat "$root/tensorfold-version.txt" >&2; die "$bin --version failed on this host"; }
echo "setup.sh: $bin --version: $(cat "$root/tensorfold-version.txt")"
mkdir -p "$root/sessions"

filter='walk(if type == "string" then gsub("/opt/localrouter/inkling/models"; $models) | gsub("/opt/localrouter/inkling"; $root) else . end)
  | .tools[0].qualification = $marker'
[ "$lock_set" = 0 ] || filter="$filter | if \$lock == \"\" then del(.gpu_lock) else .gpu_lock = \$lock end"
jq --arg models "$models" --arg root "$root" --arg marker "$marker" --arg lock "$lock" \
  "$filter" "$here/config.example.json" > "$root/config.json.tmp"
python3 -I "$here/check_args.py" "$root/config.json.tmp" "$serve" ||
  die "config.example.json does not serve the recipe's argv ($serve, above)"
mv "$root/config.json.tmp" "$root/config.json"
cat <<EOF
setup.sh: wrote $root/config.json. Merge its "gpu_lock" and "tools" entries into the daemon's config:
$(cat "$root/config.json")
setup.sh: the tool stays refused (503, not GPU-qualified) until qualify.sh writes $marker:
  tools/chat/inkling2-mini/qualify.sh $root /path/to/localrouter OUT $marker
EOF
