#!/usr/bin/env bash
# Assembles the install kit for one model: everything a GPU node needs besides its weights, in one directory to copy
# there, checked as a whole by its SHA256SUMS:
#   bash tools/package/kit.sh --image FILE.tar.zst --profile PROFILE.json [--manifest FILE] [--options FILE.json] OUT_DIR
# OUT_DIR gets tools/package (these scripts and the one profile), the profile's asset manifest at the path the profile
# names (--manifest supplies it when it is not in this checkout), the image file and its build receipt, the options
# file when given (a JSON object for TOOL_OPTIONS, e.g. a GPU qualification's chosen values), fingerprint.json
# (build_fingerprint.py on the image's source commit and the image: its build inputs, packages and licence
# conditions), kit.json (the packaging commit, the image's source commit and digests, every input's SHA-256, the
# fingerprint's conditions, the install command) and SHA256SUMS.
# It checks the image file against its receipt and reads its manifest and config digests from the archive first.
# It refuses a packaging checkout with uncommitted changes (kit.json names a commit) unless LR_ALLOW_DIRTY=1.
# On the node:  cd OUT_DIR && sha256sum -c SHA256SUMS && (follow kit.json's "install")
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
die() { echo "kit: $*" >&2; exit 1; }

image="" profile="" manifest="" options="" out=""
while (($#)); do
    case $1 in
        --image) image=$2; shift ;; --profile) profile=$2; shift ;; --manifest) manifest=$2; shift ;;
        --options) options=$2; shift ;; -h | --help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown argument $1" ;; *) out=$1 ;;
    esac
    shift
done
[[ -n $out ]] || die "OUT_DIR is required"
[[ -f $image && -f ${image%.tar.zst}.json ]] || die "--image FILE.tar.zst is required, with its receipt ${image%.tar.zst}.json beside it"
[[ -f $profile ]] || die "--profile FILE is required"
[[ -z $options ]] || python3 -I -c 'import json,sys; assert isinstance(json.load(open(sys.argv[1])), dict)' "$options" 2> /dev/null ||
    die "--options $options is not a JSON object"
dirty=false
if [[ -n $(git -C "$REPO" status --porcelain -- tools/package) ]]; then
    [[ ${LR_ALLOW_DIRTY:-0} == 1 ]] || die "tools/package has uncommitted changes; commit them (or LR_ALLOW_DIRTY=1)"
    dirty=true
fi
[[ ! -e $out || -z $(ls -A "$out" 2> /dev/null) ]] || die "$out exists and is not empty"
rel=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("manifest") or "")' "$profile")
[[ -n $rel && $rel != /* ]] || die "the profile names no repository-relative manifest"
manifest=${manifest:-$REPO/$rel}
[[ -f $manifest ]] || die "no asset manifest at $manifest: give it with --manifest"
python3 -I -c 'import json,sys; m=json.load(open(sys.argv[1])); assert isinstance(m, dict) and m.get("selected_files")' \
    "$manifest" 2> /dev/null || die "$manifest is not an asset manifest (a JSON object with selected_files)"

receipt=${image%.tar.zst}.json
want=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["file_sha256"])' "$receipt")
got=$(sha256sum "$image" | cut -d' ' -f1)
[[ $got == "$want" ]] || die "the image file's SHA-256 is $got, its receipt says $want"
digests=$(python3 -I "$HERE/image_digests.py" "$image") || die "the image file is not a readable image archive"
source=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$receipt")
git -C "$REPO" cat-file -e "$source^{commit}" 2> /dev/null || die "the image's source commit $source is not in this repository: fetch it"
fingerprint=$(cd "$REPO" && python3 -I "$HERE/build_fingerprint.py" "$source" --image "$image" --manifest "$manifest") ||
    die "could not fingerprint the image's source commit $source"

mkdir -p "$out/tools/package/profiles" "$out/$(dirname "$rel")"
out=$(cd "$out" && pwd)
cp "$HERE"/*.sh "$HERE"/*.py "$HERE/localrouterctl" "$HERE/README.md" "$out/tools/package/"
cp "$profile" "$out/tools/package/profiles/"
cp "$manifest" "$out/$rel"
cp "$image" "$receipt" "$out/"
[[ -z $options ]] || cp "$options" "$out/options.json"
printf '%s\n' "$fingerprint" > "$out/fingerprint.json"
python3 -I - "$out" "$(git -C "$REPO" rev-parse HEAD)" "$dirty" "$receipt" "$digests" "$(basename "$profile")" "$rel" \
    "$(basename "$image")" "${options:+options.json}" <<'PY'
import hashlib, json, os, sys
out, commit, dirty, receipt, digests, profile, manifest, image, options = sys.argv[1:]
r, d = json.load(open(receipt)), json.loads(digests)
sha = lambda p: hashlib.sha256(open(os.path.join(out, p), "rb").read()).hexdigest()
opts = '"$(cat options.json)"' if options else "'{...the qualified options...}'"
json.dump({"packaging_commit": commit, "packaging_dirty": dirty == "true",
           "image": {"file": image, "file_sha256": r["file_sha256"], "source_commit": r["commit"], "source_dirty": r["dirty"],
                     "manifest_digest": d["manifest_digest"], "config_digest": d["config_digest"], "platform": d["platform"]},
           "profile": {"file": "tools/package/profiles/" + profile, "sha256": sha("tools/package/profiles/" + profile)},
           "asset_manifest": {"file": manifest, "sha256": sha(manifest)},
           "options": {"file": options, "sha256": sha(options)} if options else None,
           "fingerprint": {"file": "fingerprint.json", "sha256": sha("fingerprint.json"),
                           "conditions": json.load(open(os.path.join(out, "fingerprint.json")))["conditions"]},
           "install": "TOOL_OPTIONS=%s bash tools/package/install.sh --profile tools/package/profiles/%s --weights <weights dir> --image %s --smoke"
                      % (opts, profile, image)},
          open(os.path.join(out, "kit.json"), "w"), indent=1)
PY
sums=$(cd "$out" && find . -type f -printf '%P\n' | sort | xargs -d '\n' sha256sum)
printf '%s\n' "$sums" > "$out/SHA256SUMS"
echo "kit: $out ($(du -sh "$out" | cut -f1)); on the node: cd into it, sha256sum -c SHA256SUMS, then kit.json's install"
