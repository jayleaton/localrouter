#!/usr/bin/env bash
# Sets up Fish Audio Drama 3 on macOS (Apple silicon) from tools/speech/drama3/pins-macos.json, and checks every hash.
#   ZIG=/path/to/zig-0.17.0/zig tools/speech/drama3/setup-macos.sh ROOT [--no-weights] [--no-build] [--verify size|sha256]
# ROOT/drama-tensorfold is plotarmordev/drama-tensorfold at the pinned commit (the Mac launcher, the engine build script
# and the Mac NVFP4 patch series), each pinned file by sha256. ROOT/launcher/pilot_macos.py is upstream's Mac launcher
# with LocalRouter's pinned patch (patches/pilot-macos-external-admission.patch), beside the module it imports.
# ROOT/voices-macos holds the six named voices converted to the Metal server's catalog (voices_macos.py), each by
# sha256. The weights, plotarmordev/drama-3-NVFP4-Pruned at the pinned revision (22.3 GB), go into the Hugging Face
# CLI's shared cache (`hf` on PATH, or HF=/path/to/hf), and ROOT/drama-3-NVFP4-Pruned links to that snapshot; each
# file is checked by size and sha256 (--verify size: by size only, for a re-run). The engine is built by upstream's
# scripts/build-engine.sh --weights nvfp4 in ROOT/build/tensorfold-nvfp4, its HEAD checked against the pinned
# patched_head, and ROOT/engine-macos.json records the three binaries' sha256 (the qualification marker carries the
# server's). The build needs Zig 0.17.0 and Xcode's Metal compiler (`xcrun -f metal`; DEVELOPER_DIR picks Xcode when
# xcode-select names the Command Line Tools): a missing compiler, or an Xcode licence not yet accepted, stops here with
# exit 3 after everything else is in place. Nothing is executed from the
# downloads, and nothing here accepts the weights' licence. Spark/Linux hosts use fetch.sh and build.sh instead.
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
root=${1:?usage: setup-macos.sh ROOT [--no-weights] [--no-build] [--verify size|sha256]}
shift
weights=1 build=1 verify=sha256
while [ $# -gt 0 ]; do
  case $1 in
    --no-weights) weights=0; shift ;;
    --no-build) build=0; shift ;;
    --verify) verify=${2:-}; shift 2 ;;
    *) echo "setup-macos.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
die() { echo "setup-macos.sh: $*" >&2; exit 1; }
case $verify in size | sha256) ;; *) die "--verify must be size or sha256" ;; esac
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "this is the Apple silicon setup; a GB10 uses fetch.sh and build.sh"
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
mkdir -p "$root"
root=$(cd "$root" && pwd)
pins=$here/pins-macos.json

# One line a pin: repo URL COMMIT DIR | file PATH SHA256 | patch DIR NAME SHA256 | tf REPO COMMIT HEAD BUILD RECORD |
# launcher PATCH PATCH_SHA BASE PATH SHA | support FROM TO | voice NAME SHA | voicedir SRC DIR | zig VERSION |
# ckpt REPO REVISION DIR | cfile PATH SIZE SHA256.
plan=$(python3 -I - "$pins" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
u, t, l, v, c = p["upstream"], p["tensorfold"], p["launcher"], p["voices"], p["checkpoint"]
print("repo", u["repository"], u["commit"], u["dir"])
for path, digest in sorted(u["files"].items()):
    print("file", path, digest)
for name, digest in sorted(t["patches"]["files"].items()):
    print("patch", t["patches"]["dir"], name, digest)
print("tf", t["repository"], t["commit"], t["patched_head"], p["build"]["dir"], p["build"]["record"])
print("launcher", l["patch"], l["patch_sha256"], l["base"], l["path"], l["sha256"])
for src, dst in sorted(l["support"].items()):
    print("support", src, dst)
print("voicedir", v["source"], v["dir"])
for name, digest in sorted(v["files"].items()):
    print("voice", name, digest)
print("zig", p["toolchain"]["zig"])
print("ckpt", c["repo"], c["revision"], c["dir"])
for path, (size, digest) in sorted(c["files"].items()):
    print("cfile", path, size, digest)
EOF
)
pinned() { # kind: the plan's lines of that kind, without the kind
  local line
  while read -r line; do
    case $line in "$1 "*) echo "${line#"$1 "}" ;; esac
  done <<< "$plan"
}

# The upstream checkout: cloned, or an existing clean clone moved to the pinned commit; every pinned file and the
# exact set of Mac NVFP4 patches by sha256.
read -r url commit updir <<< "$(pinned repo)"
up=$root/$updir
if [ -e "$up" ]; then
  git -C "$up" rev-parse --git-dir >/dev/null 2>&1 || die "$up exists and is not a git checkout"
  [ -z "$(git -C "$up" status --porcelain --untracked-files=no)" ] || die "$up has local changes; refusing to use it"
else
  git clone --quiet "$url" "$up"
fi
if [ "$(git -C "$up" rev-parse HEAD)" != "$commit" ]; then
  git -C "$up" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$up" fetch --quiet origin "$commit"
  git -C "$up" -c advice.detachedHead=false checkout --quiet --detach "$commit"
fi
bad=0
while read -r path digest; do
  got=$(sha "$up/$path" 2>/dev/null || echo missing)
  [ "$got" = "$digest" ] || { echo "setup-macos.sh: $updir/$path sha256 $got, pinned $digest" >&2; bad=1; }
done <<< "$(pinned file)"
want='' pdir=''
while read -r dir name digest; do
  pdir=$dir want="$want$name "
  got=$(sha "$up/$dir/$name" 2>/dev/null || echo missing)
  [ "$got" = "$digest" ] || { echo "setup-macos.sh: $updir/$dir/$name sha256 $got, pinned $digest" >&2; bad=1; }
done <<< "$(pinned patch)"
have=$(cd "$up/$pdir" && for f in *.patch; do [ -e "$f" ] && printf '%s ' "$f"; done)
[ "$have" = "$want" ] || { echo "setup-macos.sh: $updir/$pdir holds other *.patch files than the pinned" >&2; bad=1; }
[ "$bad" = 0 ] || die "$up does not match pins-macos.json; refusing it"
echo "setup-macos.sh: $updir at $commit verified"

# LocalRouter's launcher: upstream's Mac launcher plus the pinned patch, beside the module it imports, under
# ROOT/launcher (outside the upstream checkout, which stays clean).
read -r lpatch lpatch_sha lbase lout lout_sha <<< "$(pinned launcher)"
[ "$(sha "$here/$lpatch")" = "$lpatch_sha" ] || die "$lpatch sha256 differs from the pin $lpatch_sha"
tmp=$(mktemp -d)
mkdir -p "$tmp/$(dirname "$lbase")" "$root/$(dirname "$lout")"
cp "$up/$lbase" "$tmp/$lbase"
(cd "$tmp" && git apply "$here/$lpatch") || { rm -rf "$tmp"; die "$lpatch does not apply to $updir/$lbase"; }
mv "$tmp/$lbase" "$root/$lout.tmp" && rm -rf "$tmp" && mv "$root/$lout.tmp" "$root/$lout"
[ "$(sha "$root/$lout")" = "$lout_sha" ] || die "$lout sha256 differs from the pin $lout_sha"
while read -r from to; do
  cp "$up/$from" "$root/$to.tmp" && mv "$root/$to.tmp" "$root/$to"
done <<< "$(pinned support)"
echo "setup-macos.sh: $lout is $updir/$lbase with $lpatch"

# The named voices, converted; the directory must then hold exactly the pinned files (the server loads every *.json).
read -r vsrc vdir <<< "$(pinned voicedir)"
names=$(pinned voice | while read -r name _; do echo "$here/$vsrc/$name"; done)
rm -rf "$root/$vdir.tmp"
# shellcheck disable=SC2086 # one path a line, none with spaces (pinned names)
python3 -I "$here/voices_macos.py" convert "$root/$vdir.tmp" $names >/dev/null
while read -r name digest; do
  got=$(sha "$root/$vdir.tmp/$name")
  [ "$got" = "$digest" ] || { rm -rf "$root/$vdir.tmp"; die "$vdir/$name converts to sha256 $got, pinned $digest"; }
done <<< "$(pinned voice)"
rm -rf "${root:?}/${vdir:?}"
mv "$root/$vdir.tmp" "$root/$vdir"
echo "setup-macos.sh: $(pinned voice | grep -c .) voices converted and verified in $root/$vdir"

# The weights: the pinned revision through the Hugging Face CLI into its shared cache, ROOT/DIR a link to the snapshot.
if [ "$weights" = 1 ]; then
  read -r repo rev ckdir <<< "$(pinned ckpt)"
  cat <<EOF
setup-macos.sh: the Drama 3 weights and codec are licensed under the Fish Audio Community License, Version 1.0
  (FACL v1.0): https://huggingface.co/$repo/blob/$rev/LICENSE
  $repo is an unofficial community quant of fishaudio/drama-3 (not made, reviewed or endorsed by Fish Audio).
  Built with Fish Audio. Keep this attribution with any copy: "These Fish Audio Materials are licensed under the Fish
  Audio Community License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved."
  Section III(c) grants no licence for a Hosted Service (third parties generating speech from text or audio they
  supply, for example through an API) without a separate written licence from Fish Audio: keep LocalRouter on
  loopback. Read the licence; downloading the files does not accept it on anyone's behalf.
EOF
  hf=$(command -v "${HF:-hf}") || die "the Hugging Face CLI (hf) is missing: install huggingface_hub in a venv and set HF"
  snapshot=$("$hf" download "$repo" --revision "$rev" --quiet | tail -1)
  [ -d "$snapshot" ] || die "hf download did not print a snapshot directory (got '$snapshot')"
  if [ -L "$root/$ckdir" ] || [ ! -e "$root/$ckdir" ]; then
    ln -sfn "$snapshot" "$root/$ckdir"
  fi
  n=0
  while read -r path size digest; do
    f=$root/$ckdir/$path
    [ -f "$f" ] && [ "$(stat -L -f %z "$f")" = "$size" ] || die "$ckdir/$path is missing or not $size bytes"
    if [ "$verify" = sha256 ] && [ "$(sha "$f")" != "$digest" ]; then
      die "$ckdir/$path sha256 differs from the pin $digest"
    fi
    n=$((n + 1))
  done <<< "$(pinned cfile)"
  echo "setup-macos.sh: $n checkpoint files verified by $verify in $root/$ckdir -> $snapshot"
fi
[ "$build" = 1 ] || exit 0

# The engine, with upstream's own build script; Xcode's Metal compiler first, since that is what a fresh Mac lacks.
read -r zig_version <<< "$(pinned zig)"
read -r _ tcommit head bdir record <<< "$(pinned tf)"
zig=$(command -v "${ZIG:-zig}") || die "Zig not found; set ZIG"
[ "$("$zig" version)" = "$zig_version" ] || die "Zig $("$zig" version) found; the engine pins $zig_version"
if ! metal=$(xcrun -f metal 2>&1); then
  echo "setup-macos.sh: Xcode's Metal compiler is not usable: $(tail -1 <<< "$metal")" >&2
  echo "  run: sudo xcodebuild -license accept (and xcodebuild -downloadComponent MetalToolchain if xcrun asks);" >&2
  echo "  with only the Command Line Tools selected, set DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer" >&2
  exit 3
fi
xcrun metal --version >/dev/null 2>&1 || {
  echo "setup-macos.sh: $metal does not run: install the Metal toolchain (xcodebuild -downloadComponent MetalToolchain)" >&2
  exit 3
}
build_dir=$root/$bdir
rm -f "$root/$record"
mkdir -p "$(dirname "$build_dir")"
ZIG=$zig "$up/scripts/build-engine.sh" --weights nvfp4 "$build_dir"
got=$(git -C "$build_dir" rev-parse HEAD)
[ "$got" = "$head" ] || die "$build_dir is at $got, pinned patched_head $head"
[ -z "$(git -C "$build_dir" status --porcelain --untracked-files=no)" ] || die "$build_dir has local changes after the build"
python3 -I - "$pins" "$root/$record" "$build_dir" "$head" "$tcommit" "$("$zig" version)" "$(date -u +%FT%TZ)" <<'EOF'
import hashlib, json, os, sys
pins, path, build, head, commit, zig, built_at = sys.argv[1:]
record = {"platform": "macos-arm64", "variant": "nvfp4", "patched_head": head, "tensorfold_commit": commit, "zig": zig,
          "built_at": built_at}
for key, rel in json.load(open(pins))["build"]["binaries"].items():
    binary = os.path.join(build, rel)
    if not os.access(binary, os.X_OK):
        sys.exit(f"setup-macos.sh: {binary} was not built")
    with open(binary, "rb") as f:
        record[f"{key}_bin"], record[f"{key}_sha256"] = binary, hashlib.sha256(f.read()).hexdigest()
with open(path + ".tmp", "w") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
os.replace(path + ".tmp", path)
print(f"setup-macos.sh: engine {head} built; tensorfold-native sha256 {record['native_sha256']}, recorded in {path}")
EOF
