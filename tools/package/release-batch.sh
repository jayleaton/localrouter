#!/usr/bin/env bash
# One release batch for an exact commit, on the build machine: the project's checks on that commit, then its image.
#   bash tools/package/release-batch.sh COMMIT [BASE] [OUT_DIR]
# COMMIT is checked out alone in a temporary worktree (nothing uncommitted can leak into the image). On it:
#   zig fmt --check build.zig build src tests   compared with BASE's findings by identity: any new finding fails
#   zig build, zig build test                   must pass
# and only then build-image.sh exports the image. BASE defaults to the merge base of COMMIT and origin/main.
# OUT_DIR (default ~/localrouter-dist/<commit>) gets the image file and its receipt, every check's log, and batch.json:
# the commit, base, each step's status and seconds, and the change in free disk over the batch (other writers to the
# same file system count too; build-image.sh's receipt has the image build's own figures). The temporary worktrees
# are removed.
# LR_SKIP_IMAGE=1 runs the checks only (result "checks-pass").
# Run it in the machine's build slot when one is shared, e.g. heavy.sh bash tools/package/release-batch.sh <commit>.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
die() { echo "release-batch: $*" >&2; exit 1; }
command -v zig > /dev/null || die "no zig on PATH (Zig 0.17.0)"
[[ $(zig version) == 0.17.0 ]] || die "zig $(zig version); the project uses 0.17.0"
commit=$(git -C "$REPO" rev-parse --verify "${1:?usage: release-batch.sh COMMIT [BASE] [OUT_DIR]}^{commit}")
base=$(git -C "$REPO" rev-parse --verify "${2:-$(git -C "$REPO" merge-base "$commit" origin/main)}^{commit}")
OUT=${3:-$HOME/localrouter-dist/${commit:0:12}}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
work=$(mktemp -d "${TMPDIR_BUILD:-$HOME/.cache}/lr-release.XXXXXX")   # not /tmp: builds need real disk
cleanup() {
    git -C "$REPO" worktree remove --force "$work/commit" 2> /dev/null || true
    git -C "$REPO" worktree remove --force "$work/base" 2> /dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT
steps=()
step() {   # step NAME DIR COMMAND...: runs it in DIR, logs to OUT/NAME.log, records status and seconds
    local name=$1 dir=$2 t0 rc=0
    shift 2
    t0=$(date +%s)
    echo "[release-batch $(date +%T)] $name"
    (cd "$dir" && "$@") > "$OUT/$name.log" 2>&1 || rc=$?
    steps+=("$name $rc $(($(date +%s) - t0))")
    return $rc
}
finish() {
    python3 -I - "$OUT/batch.json" "$commit" "$base" "$1" "$disk0" "$(df -B1 --output=avail "$work" | tail -1)" "${steps[@]}" <<'PY'
import json, sys
out, commit, base, result, before, after, *steps = sys.argv[1:]
json.dump({"commit": commit, "base": base, "result": result, "free_space_drop_bytes": int(before) - int(after),
           "free_space_drop_includes_other_writers": True,
           "steps": [{"step": s.split()[0], "exit": int(s.split()[1]), "seconds": int(s.split()[2])} for s in steps]},
          open(out, "w"), indent=1)
PY
    echo "[release-batch] $1: $OUT/batch.json"
}
disk0=$(df -B1 --output=avail "$work" | tail -1)
git -C "$REPO" worktree add -q --detach "$work/commit" "$commit"
git -C "$REPO" worktree add -q --detach "$work/base" "$base"

fmt() { { zig fmt --check build.zig build src tests || true; } | sort; }
step fmt-base "$work/base" fmt
step fmt-commit "$work/commit" fmt
comm -13 "$OUT/fmt-base.log" "$OUT/fmt-commit.log" > "$OUT/fmt-new.log"
if [[ -s $OUT/fmt-new.log ]]; then
    steps+=("fmt-identity 1 0")
    finish fail
    die "new zig fmt findings against $base: $OUT/fmt-new.log"
fi
steps+=("fmt-identity 0 0")
step build "$work/commit" zig build || { finish fail; die "zig build failed: $OUT/build.log"; }
step test "$work/commit" zig build test || { finish fail; die "zig build test failed: $OUT/test.log"; }
[[ ${LR_SKIP_IMAGE:-0} != 1 ]] || { finish checks-pass; exit 0; }
step image "$work/commit" env SRC="$work/commit" bash "$HERE/build-image.sh" "$OUT" || { finish fail; die "the image build failed: $OUT/image.log"; }
finish pass
