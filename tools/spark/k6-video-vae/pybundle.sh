#!/usr/bin/env bash
# Builds the D2 oracle's offline aarch64 Python into $KIT/pybundle (run on an x86 or arm64 host with internet; nothing
# is compiled): python-build-standalone CPython 3.13 aarch64 (install_only), and site-packages cross-installed for
# aarch64-manylinux_2_28 / Python 3.13 with the exact pins and hashes of the upstream uv.lock (kandinsky-6@eac06e39,
# extra "consumer"), restricted to the closure the VAE path imports: torch 2.14.1+cu130 (with its nvidia-* / cuda-*
# runtime and triton), torchvision 0.29.1+cu130, diffusers 0.40.0, numpy, pillow, safetensors, and accelerate 1.15.0
# (in the native environment diffusers' from_pretrained takes its low_cpu_mem_usage load path when accelerate imports).
# The CPython patch release and build tag are the ones `uv python list cpython-3.13-linux-aarch64-gnu` reports for the
# pinned uv (uv itself would fetch the install_only_stripped variant of the same build).
# Outputs: $KIT/pybundle/{python/,site-packages/,dist/<cpython tarball>,requirements.txt,MANIFEST.sha256} and
# $KIT/pybundle.json (versions, total bytes, sha256 of the manifest). Env: KIT, UPSTREAM (default $KIT/upstream).
set -euo pipefail
KIT=${KIT:-/home/sireggy/Documents/.t3/k6-staging/d2-kit}
UP=${UPSTREAM:-$KIT/upstream}
PB=$KIT/pybundle
UV_VERSION=0.12.24
UV_SHA256=b4dfaef47d491a7296981f8374a4595f55dbf84e8937c8ecd2983574d8bb3da6   # uv-x86_64-unknown-linux-gnu.tar.gz
PBS_TAG=20261003
PY_VERSION=3.13.16
PY_TARBALL=cpython-$PY_VERSION+$PBS_TAG-aarch64-unknown-linux-gnu-install_only.tar.gz
PY_SHA256=6477121f22904963a066ac8ff89983baae781510815590e894daa3c11ce86652
UPSTREAM_SHA=eac06e39b904940baff2c408e925cdb0c4529269
LOCK_SHA256=$(sha256sum "$UP/uv.lock" | cut -d' ' -f1)
ROOTS="torch torchvision diffusers numpy pillow safetensors accelerate"

# uv: standalone x86_64 binary, checksum pinned
UVD=$HOME/.local/opt/uv-$UV_VERSION; UV=$UVD/uv
if [[ ! -x $UV ]]; then
    T=$(mktemp -d); curl -fsSL -o $T/uv.tgz https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz
    echo "$UV_SHA256  $T/uv.tgz" | sha256sum -c --quiet
    mkdir -p $UVD; tar -xzf $T/uv.tgz -C $UVD --strip-components=1; rm -rf $T
fi
[[ $($UV --version) == "uv $UV_VERSION"* ]] || { echo "pybundle: unexpected $($UV --version)" >&2; exit 1; }
[[ $(git -C "$UP" rev-parse HEAD) == "$UPSTREAM_SHA" ]] || { echo "pybundle: $UP is not $UPSTREAM_SHA" >&2; exit 1; }

rm -rf $PB; mkdir -p $PB/dist
# CPython
curl -fsSL -o $PB/dist/$PY_TARBALL "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/${PY_TARBALL//+/%2B}"
echo "$PY_SHA256  $PB/dist/$PY_TARBALL" | sha256sum -c --quiet
tar -xzf $PB/dist/$PY_TARBALL -C $PB   # -> $PB/python

# the locked requirements (hashes included), filtered to the closure of $ROOTS in the lock
ALL=$(mktemp); (cd "$UP" && $UV export --frozen --extra consumer --no-emit-project --no-header --no-annotate \
    --format requirements.txt) > $ALL
python3 -I - "$UP/uv.lock" $ALL $ROOTS > $PB/requirements.txt <<'PY'
import re, sys, tomllib
lock, req, roots = sys.argv[1], sys.argv[2], sys.argv[3:]
pkgs = {}
for p in tomllib.load(open(lock, "rb"))["package"]:
    pkgs.setdefault(p["name"], []).append(p)
def variant(name, version=None):
    vs = pkgs[name]
    if version: return [p for p in vs if p["version"] == version]
    cu = [p for p in vs if p["version"].endswith("+cu130")]   # the consumer extra's torch / torchvision
    return cu or vs
seen, todo = set(), [(r, None, ()) for r in roots]
while todo:
    name, version, extras = todo.pop()
    key = (name, version, tuple(extras))
    if key in seen: continue
    seen.add(key)
    for p in variant(name, version):
        deps = list(p.get("dependencies", []))
        for e in extras: deps += p.get("optional-dependencies", {}).get(e, [])
        for d in deps: todo.append((d["name"], d.get("version"), tuple(d.get("extra", ()))))
names = {k[0] for k in seen}
blocks, cur = [], []
for line in open(req):
    if not line.startswith(" ") and cur: blocks.append(cur); cur = []
    cur.append(line)
if cur: blocks.append(cur)
kept = set()
for b in blocks:
    m = re.match(r"([A-Za-z0-9_.-]+)==", b[0])
    if m and m.group(1) in names: sys.stdout.write("".join(b)); kept.add(m.group(1))
missing = names - kept
if missing: sys.exit(f"closure packages missing from the export: {sorted(missing)}")
PY
rm -f $ALL

# site-packages: exact pins, hash-checked, binary wheels only, no dependency resolution beyond the list. Each package
# comes from its lock source only: torch / torchvision from the cu130 index (explicit in pyproject.toml), the rest from
# PyPI (the cu130 index mirrors some of them, e.g. triton, as different files whose hashes the lock does not have).
CU=$(mktemp); PYPI=$(mktemp)
awk -v cu=$CU -v py=$PYPI '/^[A-Za-z]/ {f = ($0 ~ /^(torch|torchvision)==/) ? cu : py} {print > f}' $PB/requirements.txt
inst() { $UV pip install --target $PB/site-packages --no-deps --require-hashes --only-binary :all: --no-compile \
    --python-platform aarch64-manylinux_2_28 --python-version 3.13 --index-url "$1" -r "$2"; }
inst https://pypi.org/simple $PYPI
inst https://download.pytorch.org/whl/cu130 $CU
rm -f $CU $PYPI

# manifest and summary
(cd $PB && find . -type f ! -name MANIFEST.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $PB/MANIFEST.sha256
MAN_SHA=$(sha256sum $PB/MANIFEST.sha256 | cut -d' ' -f1)
BYTES=$(du -sb $PB | cut -f1)
python3 -I - $PB "$MAN_SHA" "$BYTES" "$UV_VERSION" "$PY_VERSION" "$PBS_TAG" "$PY_TARBALL" "$PY_SHA256" "$UPSTREAM_SHA" \
    "$LOCK_SHA256" > $KIT/pybundle.json <<'PY'
import json, pathlib, re, sys
pb, man, nbytes, uv, py, tag, tarball, pysha, up, lock = sys.argv[1:]
pins = {}  # what was installed (requirements.txt also lists pins whose markers exclude this platform)
for m in sorted(pathlib.Path(pb, "site-packages").glob("*.dist-info/METADATA")):
    meta = dict(l.split(": ", 1) for l in m.read_text().splitlines() if l.startswith(("Name: ", "Version: ")))
    pins[meta["Name"]] = meta["Version"]
files = sum(1 for _ in open(f"{pb}/MANIFEST.sha256"))
print(json.dumps({"path": pb, "platform": "aarch64-manylinux_2_28", "python": py, "python_build_standalone": tag,
                  "python_tarball": tarball, "python_tarball_sha256": pysha, "uv": uv, "upstream": up,
                  "uv_lock_sha256": lock, "packages": pins, "files": files, "total_bytes": int(nbytes),
                  "manifest_sha256": man}, indent=1))
PY
echo "{\"pybundle\": \"$PB\", \"total_bytes\": $BYTES, \"manifest_sha256\": \"$MAN_SHA\", \"packages\": $(ls -d $PB/site-packages/*.dist-info | wc -l)}"
