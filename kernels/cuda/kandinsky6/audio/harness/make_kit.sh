#!/usr/bin/env bash
# Build an immutable job kit: the harness files plus one given fatbin, a SHA256SUMS over them, and a reproducible
# tarball (sorted names, mtime 0, numeric root owner, gzip -n) with its own .sha256.
#
#   make_kit.sh FATBIN DESTDIR      -> DESTDIR/k6a-kit-<fatbin sha256[:12]>-<harness sha256[:12]>.tar.gz(.sha256)
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 FATBIN DESTDIR" >&2; exit 64; }
FATBIN=$1; DEST=$2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FILES=(jobs.py k6a_cuda.py native.py run_job.sh make_kit.sh MANIFEST.md)
[ -f "$FATBIN" ] || { echo "no fatbin at $FATBIN" >&2; exit 65; }
mkdir -p "$DEST"
FSHA=$(sha256sum "$FATBIN" | cut -c1-12)
HSHA=$(cd "$HERE" && cat "${FILES[@]}" | sha256sum | cut -c1-12)
NAME="k6a-kit-$FSHA-$HSHA"
STAGE=$(mktemp -d)
trap 'chmod -R u+w "$STAGE"; rm -rf "$STAGE"' EXIT
mkdir "$STAGE/$NAME"
(cd "$HERE" && cp "${FILES[@]}" "$STAGE/$NAME/")
cp "$FATBIN" "$STAGE/$NAME/audio.fatbin"
chmod 0555 "$STAGE/$NAME"/*.sh
chmod 0444 "$STAGE/$NAME"/*.py "$STAGE/$NAME"/MANIFEST.md "$STAGE/$NAME"/audio.fatbin
(cd "$STAGE/$NAME" && sha256sum audio.fatbin "${FILES[@]}" > SHA256SUMS && chmod 0444 SHA256SUMS)
chmod 0555 "$STAGE/$NAME"
[ ! -e "$DEST/$NAME.tar.gz" ] || { echo "$DEST/$NAME.tar.gz exists (kits are immutable)" >&2; exit 65; }
tar -C "$STAGE" --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - "$NAME" | gzip -n -9 > "$DEST/$NAME.tar.gz"
chmod 0444 "$DEST/$NAME.tar.gz"
(cd "$DEST" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" && chmod 0444 "$NAME.tar.gz.sha256")
cat "$DEST/$NAME.tar.gz.sha256"
