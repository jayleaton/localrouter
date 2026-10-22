#!/usr/bin/env bash
# Builds the patched TensorFold native engine for Drama 3 with upstream's own scripts/build-engine.sh, and records it.
#   ZIG=/path/to/zig-0.17.0/zig NVCC=/usr/local/cuda/bin/nvcc tools/speech/drama3/build.sh ROOT
# ROOT/drama-tensorfold is the checkout fetch.sh made; it is verified against pins.json again first, and a checkout
# at another commit, or any file or patch that differs, is refused. The engine is built in ROOT/build/tensorfold: the
# pinned TensorFold commit plus the patch series, applied with a fixed committer so that its HEAD is pins.json's
# patched_head, then LocalRouter's engine patches (patches/engine/, pins.json engine_patches: the voice prefix
# cache) the same way, to engine_patches.head; the server is built again. ROOT/engine.json then records the binary and its sha256, which the qualification marker carries and
# the worker checks at load. Needs git, Zig 0.17.0 and CUDA 13's nvcc, as upstream does. JOBS and DRAMA_TOOLS pass
# through to build-engine.sh.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${1:?usage: build.sh ROOT}
[ $# = 1 ] || { echo "build.sh: usage: build.sh ROOT" >&2; exit 2; }
die() { echo "build.sh: $*" >&2; exit 1; }
[ -d "$root" ] || die "$root is not a directory; run fetch.sh ROOT first"
root=$(cd "$root" && pwd)

# One line: zig cuda upstream-dir upstream-commit patched-head.
pins=$(python3 -I - "$here/pins.json" <<'EOF'
import json, sys
pins = json.load(open(sys.argv[1]))
print(pins["toolchain"]["zig"], pins["toolchain"]["cuda"], pins["upstream"]["dir"], pins["upstream"]["commit"],
      pins["tensorfold"]["patched_head"], pins["engine_patches"]["head"], pins["engine_patches"]["dir"],
      *sorted(pins["engine_patches"]["files"]))
EOF
)
read -r want_zig want_cuda updir upstream patched_head final_head epdir epnames <<< "$pins"
engine_patches=()
for p in $epnames; do engine_patches+=("$here/$epdir/$p"); done
zig=$(command -v "${ZIG:-zig}") || die "Zig not found; set ZIG"
nvcc=$(command -v "${NVCC:-nvcc}") || die "nvcc not found; set NVCC"
have=$("$zig" version)
[ "$have" = "$want_zig" ] || die "Zig $have found; the engine pins $want_zig"
version=$("$nvcc" --version)
grep -q "release $want_cuda\." <<< "$version" || die "$nvcc is not CUDA $want_cuda: $(tail -1 <<< "$version")"

"$here/fetch.sh" "$root" --check --no-weights
build=$root/build/tensorfold
bin=$build/zig-out/native/bin/tensorfold-native
# A failed build leaves no record of an older binary.
rm -f "$root/engine.json"
mkdir -p "$root/build"
ZIG=$zig NVCC=$nvcc "$root/$updir/scripts/build-engine.sh" "$build"

head=$(git -C "$build" rev-parse HEAD)
[ "$head" = "$patched_head" ] || die "$build is at $head, pinned patched_head $patched_head"
[ -z "$(git -C "$build" status --porcelain --untracked-files=no)" ] || die "$build has local changes after the build"

# LocalRouter's own engine patches (pins.json engine_patches, verified by fetch.sh above) on top of upstream's
# series, with the same fixed committer, then the server is built again.
GIT_COMMITTER_NAME=build-engine GIT_COMMITTER_EMAIL=build-engine@example.invalid \
  git -C "$build" am --quiet --committer-date-is-author-date "${engine_patches[@]}" \
  || { git -C "$build" am --abort || true; die "LocalRouter's engine patches do not apply to $patched_head"; }
head=$(git -C "$build" rev-parse HEAD)
[ "$head" = "$final_head" ] || die "$build is at $head after LocalRouter's engine patches, pinned $final_head"
(cd "$build" && "$zig" build native -Dnvcc="$nvcc" --summary none ${JOBS:+-j"$JOBS"})
patched_head=$final_head
[ -x "$bin" ] || die "$bin was not built"
digest=$(sha256sum "$bin" | cut -d' ' -f1)
python3 -I - "$root/engine.json" "$patched_head" "$bin" "$digest" "$upstream" "$(date -u +%FT%TZ)" <<'EOF'
import json, os, sys
path, head, binary, digest, upstream, built_at = sys.argv[1:]
record = {"patched_head": head, "native_bin": binary, "native_sha256": digest, "upstream_commit": upstream,
          "built_at": built_at}
with open(path + ".tmp", "w") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
os.replace(path + ".tmp", path)
EOF
echo "build.sh: $bin sha256 $digest (TensorFold $patched_head), recorded in $root/engine.json"
