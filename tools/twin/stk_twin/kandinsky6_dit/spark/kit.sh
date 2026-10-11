#!/usr/bin/env bash
# The Kandinsky-6 DiT Spark job kit: one immutable, reproducible tarball a Spark node runs node-job.sh against. Runs on the
# dev host, offline (no network, no docker, no build):
#   FATBINS=<dir with k6_dit_{gemm,attention,ops}.fatbin> K6_SRC=<kandinsky-6 git checkout> \
#       [CREATED_UTC=YYYY-MM-DDTHH:MM:SSZ] bash tools/twin/stk_twin/kandinsky6_dit/spark/kit.sh <OUT dir>
# Contents (paths inside the tarball, no leading directory):
#   repo/{tools/twin/stk_twin,kernels/cuda/kandinsky6,src/engines/kandinsky6}  `git archive HEAD` of this repository
#   fatbins/k6_dit_{gemm,attention,ops}.fatbin                                the prebuilt images, byte copies
#   kandinsky6-upstream/                                                       `git archive` of upstream commit $UPSTREAM
#   MANIFEST.txt                     '#' header (repo commit, upstream commit, fatbin sha256, created UTC) and a
#                                    sha256sum line for every other file
# Only committed content goes in: uncommitted or untracked files under the archived paths refuse the kit (KIT_ALLOW_DIRTY=1
# archives HEAD anyway and records that in the header). The tar is sorted, every mtime is the repo commit time, owner and
# group 0, modes normalized, gzip -n: the same HEAD, fatbins, upstream commit and CREATED_UTC give the same sha256 (with
# the same GNU tar and gzip). CREATED_UTC defaults to now and is printed: pass it back to reproduce the kit.
# Writes OUT/k6-dit-kit-<commit12>.tar.gz, .sha256 and .MANIFEST.txt; refuses to replace a different existing kit.
set -uo pipefail
UPSTREAM=eac06e39b904940baff2c408e925cdb0c4529269
PATHS=(tools/twin/stk_twin kernels/cuda/kandinsky6 src/engines/kandinsky6)
IMAGES=(gemm attention ops)
die() { echo "kit: $*" >&2; exit 1; }

OUT=${1:-}
[[ -n $OUT ]] || die "usage: FATBINS=DIR K6_SRC=DIR [CREATED_UTC=...] bash kit.sh OUT_DIR"
: "${FATBINS:?FATBINS (directory of k6_dit_{gemm,attention,ops}.fatbin) is required}"
: "${K6_SRC:?K6_SRC (a kandinsky-6 git checkout containing $UPSTREAM) is required}"
for t in git tar gzip sha256sum find sort xargs install mktemp; do
    command -v "$t" > /dev/null || die "missing tool: $t"
done
tar --sort=name --version > /dev/null 2>&1 || die "GNU tar with --sort (1.28 or later) is required"

# ---- the repository at HEAD
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2> /dev/null) || die "not inside a git checkout"
COMMIT=$(git -C "$REPO" rev-parse --verify 'HEAD^{commit}') || die "no HEAD commit"
MTIME=$(git -C "$REPO" show -s --format=%ct "$COMMIT") || die "no commit time"
for p in "${PATHS[@]}"; do
    git -C "$REPO" cat-file -e "$COMMIT:$p" 2> /dev/null || die "$p is not in HEAD ($COMMIT)"
done
dirty=$(git -C "$REPO" status --porcelain --untracked-files=all -- "${PATHS[@]}")
if [[ -n $dirty ]]; then
    echo "$dirty" >&2
    [[ ${KIT_ALLOW_DIRTY:-0} == 1 ]] || die "uncommitted changes under the kit paths (above); commit them, or KIT_ALLOW_DIRTY=1 to archive HEAD without them"
    echo "kit: KIT_ALLOW_DIRTY=1: archiving HEAD $COMMIT; the changes above are NOT in the kit" >&2
fi
GATE=absent
git -C "$REPO" cat-file -e "$COMMIT:tools/twin/stk_twin/kandinsky6_dit/gate.py" 2> /dev/null && GATE=present
[[ $GATE == present ]] || echo "kit: WARNING: kandinsky6_dit/gate.py is not in HEAD: the ops and blocks jobs will refuse this kit" >&2

# ---- the fatbins and the upstream commit
for n in "${IMAGES[@]}"; do
    [[ -f $FATBINS/k6_dit_$n.fatbin && -s $FATBINS/k6_dit_$n.fatbin ]] || die "missing or empty: $FATBINS/k6_dit_$n.fatbin"
done
got=$(git -C "$K6_SRC" rev-parse --verify --quiet "$UPSTREAM^{commit}") || die "$K6_SRC does not contain commit $UPSTREAM"
[[ $got == "$UPSTREAM" ]] || die "$K6_SRC resolves $UPSTREAM to $got"
k6_head=$(git -C "$K6_SRC" rev-parse HEAD 2> /dev/null || echo unknown)
[[ $k6_head == "$UPSTREAM" ]] || echo "kit: note: $K6_SRC HEAD is $k6_head; archiving the pinned $UPSTREAM itself" >&2

CREATED_UTC=${CREATED_UTC:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}
[[ $CREATED_UTC =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || die "CREATED_UTC must be YYYY-MM-DDTHH:MM:SSZ"

# ---- stage, manifest, pack
umask 022
mkdir -p "$OUT" || die "cannot create $OUT"
OUT=$(cd "$OUT" && pwd)
NAME=k6-dit-kit-${COMMIT:0:12}.tar.gz
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/k6-dit-kit.XXXXXX") || die "mktemp failed"
trap 'rm -rf "$STAGE"' EXIT
K=$STAGE/kit
mkdir -p "$K/repo" "$K/fatbins" "$K/kandinsky6-upstream" "$STAGE/check"
git -C "$REPO" archive --format=tar "$COMMIT" -- "${PATHS[@]}" | tar -C "$K/repo" -xf - ||
    die "git archive of the repository failed"
git -C "$K6_SRC" archive --format=tar "$UPSTREAM" | tar -C "$K/kandinsky6-upstream" -xf - ||
    die "git archive of kandinsky-6 $UPSTREAM failed"
for n in "${IMAGES[@]}"; do
    install -m 0644 "$FATBINS/k6_dit_$n.fatbin" "$K/fatbins/k6_dit_$n.fatbin" || die "copy of k6_dit_$n.fatbin failed"
done
[[ -z $(find "$K" ! -type f ! -type d -print -quit) ]] || die "the kit would contain a symlink or special file"

{
    echo "# k6-dit-kit 1"
    echo "# repo_commit: $COMMIT"
    echo "# repo_paths: ${PATHS[*]}"
    echo "# repo_uncommitted_excluded: $([[ -n $dirty ]] && echo "yes ($(wc -l <<< "$dirty") paths)" || echo no)"
    echo "# gate_py: $GATE"
    echo "# upstream: kandinsky-6 $UPSTREAM"
    for n in "${IMAGES[@]}"; do
        echo "# fatbin: k6_dit_$n.fatbin $(sha256sum "$K/fatbins/k6_dit_$n.fatbin" | cut -d' ' -f1)"
    done
    echo "# created_utc: $CREATED_UTC"
    (cd "$K" && find . -type f ! -path ./MANIFEST.txt -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
} > "$STAGE/MANIFEST.txt" || die "manifest failed"
mv "$STAGE/MANIFEST.txt" "$K/MANIFEST.txt"

tar -C "$K" --sort=name --mtime="@$MTIME" --owner=0 --group=0 --numeric-owner --mode='u+rwX,go+rX,go-w' \
    --format=gnu -cf - . | gzip -n -9 > "$STAGE/$NAME" || die "tar failed"
SHA=$(sha256sum "$STAGE/$NAME" | cut -d' ' -f1)

# the tarball as a node will see it: extract and check every file against the manifest
tar -C "$STAGE/check" -xzf "$STAGE/$NAME" || die "self-check extract failed"
(cd "$STAGE/check" && grep -v '^#' MANIFEST.txt | sha256sum -c --quiet --strict -) || die "self-check against MANIFEST failed"
files=$(cd "$STAGE/check" && find . -type f ! -path ./MANIFEST.txt | wc -l)
listed=$(grep -vc '^#' "$K/MANIFEST.txt")
(( files == listed )) || die "self-check: $files files, $listed in MANIFEST"

if [[ -e $OUT/$NAME ]]; then
    old=$(sha256sum "$OUT/$NAME" | cut -d' ' -f1)
    [[ $old == "$SHA" ]] || die "$OUT/$NAME exists with sha256 $old (this build: $SHA); not replacing it (kits are immutable)"
    echo "kit: reproduced the existing $OUT/$NAME bit for bit"
else
    mv "$STAGE/$NAME" "$OUT/$NAME" || die "cannot write $OUT/$NAME"
fi
cp "$K/MANIFEST.txt" "$OUT/$NAME.MANIFEST.txt"
(cd "$OUT" && echo "$SHA  $NAME" > "$NAME.sha256")
echo "kit: $OUT/$NAME ($listed files; repo $COMMIT; upstream $UPSTREAM; gate.py $GATE)"
echo "kit: CREATED_UTC=$CREATED_UTC (pass it back with the same inputs to reproduce)"
echo "KIT_SHA=$SHA"
