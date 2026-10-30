#!/usr/bin/env bash
# Packs the inputs of the two Spark lane jobs of whisper-job.sh, on the development machine at idle priority. Nothing
# is compiled, run, verified or hashed here: digests are pinned in the scripts and checked on the Spark, and the
# owner's tooling computes the transfer digests.
#   KIT=<output dir> bash tools/spark/whisper-kit.sh
# The working tree becomes a candidate commit (git commit-tree on HEAD; no branch or ref moves) in repo.bundle, so the
# Spark checks exactly this tree and, once it passes, the same tree is committed. Outputs:
#   $KIT/checks/in/, $KIT/checks.tar (+ .sha256), $KIT/checks.json  the CPU job: COMMIT, TREE, BASE, repo.bundle,
#                   zig-pkg/, Zig 0.17.0 aarch64, static arm64 ffmpeg, basefmt/, run.sh (= whisper-job.sh)
#   $KIT/gpu/in/, $KIT/gpu.json   the GPU job: the same plus the python-build-standalone CPython 3.12 aarch64 tarball,
#                   wheels/ + requirements.txt (uv.lock's aarch64 pins with their hashes), model/<revision>/,
#                   vad/silero_vad.jit (Silero VAD, pinned in whisper_core.py) and fixtures/. Left as a directory of
#                   hard links (no 4 GB tar written or hashed here); every file in it is pinned and checked on the node.
# Needs git, uv, python3, curl, and the caches in $CACHE (model, fixtures, wheels, Python; missing wheels, Zig and
# ffmpeg are downloaded into $CACHE/cache once). BASEFMT: the base's `zig fmt --check` findings and per-file diffs.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
KIT=${KIT:?KIT (output directory) is required}
CACHE=${CACHE:-$HOME/.cache/localrouter-speech}
BASEFMT=${BASEFMT:-$CACHE/checks}
BASE=${BASE:-a71e0a63eab57fea3a87f17338e44e7bb690556a}
ZIG=https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
FFMPEG=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-12-13-12/ffmpeg-n9.0.2-27-g7e93014c5b-linuxarm64-gpl-9.0.tar.xz
PYTHON=https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.12.15%2B20261003-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz
VAD=https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.jit
REV=$(python3 -I -c "import ast,sys; t=open(sys.argv[1]).read(); print([n.value.value for n in ast.parse(t).body if isinstance(n, ast.Assign) and n.targets[0].id == 'REVISION'][0])" "$REPO/tools/speech/whisper/fetch_model.py")
MODEL=${MODEL:-$CACHE/models/whisper-large-v3-turbo/$REV}
FIXTURES=${FIXTURES:-$CACHE/fixtures}
[[ -d $MODEL && -f $FIXTURES/manifest.json ]] || { echo "fetch the model and fixtures first" >&2; exit 1; }
mkdir -p "$CACHE/cache/wheels-aarch64"
rm -rf "$KIT" && mkdir -p "$KIT/checks/in" "$KIT/gpu/in"
C=$KIT/checks/in G=$KIT/gpu/in

echo "== candidate commit"
TREE=$(cd "$REPO" && export GIT_INDEX_FILE=$(mktemp -u) && git add -A && git write-tree && rm -f "$GIT_INDEX_FILE")
COMMIT=$(git -C "$REPO" commit-tree "$TREE" -p HEAD -m "lane candidate: the working tree on $(git -C "$REPO" rev-parse --short HEAD)")
PARENT=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" update-ref refs/lane/whisper-candidate "$COMMIT"
git -C "$REPO" update-ref refs/lane/whisper-parent "$PARENT"
git -C "$REPO" update-ref refs/lane/whisper-base "$BASE"
git -C "$REPO" bundle create "$C/repo.bundle" refs/lane/whisper-candidate refs/lane/whisper-parent refs/lane/whisper-base
for r in candidate parent base; do git -C "$REPO" update-ref -d refs/lane/whisper-$r; done
# Verified on the Spark (git bundle verify, explicit fetch of the three refs, every pin), not here.
echo "$COMMIT" > "$C/COMMIT"; echo "$TREE" > "$C/TREE"; echo "$BASE" > "$C/BASE"; echo "$PARENT" > "$C/PARENT"
cp "$REPO/tools/spark/whisper-job.sh" "$C/run.sh"
cp -al "$REPO/zig-pkg" "$C/zig-pkg"
mkdir -p "$C/basefmt" && cp "$BASEFMT/base-fmt.txt" "$BASEFMT"/basefmt/*.diff "$C/basefmt/"

echo "== zig and ffmpeg (aarch64; checked on the node)"
for u in $ZIG $FFMPEG; do
    f=$CACHE/cache/$(basename "$u")
    if [[ ! -s $f ]]; then curl -fsSL --limit-rate 20M "$u" -o "$f.part" && mv "$f.part" "$f"; fi
    [[ -s $f ]] || { echo "could not fetch $u" >&2; exit 1; }
    ln "$f" "$C/$(basename "$u")"
done

echo "== the GPU job's input: the checks input plus Python, wheels, model, fixtures"
cp -al "$C/." "$G/"
mkdir -p "$G/wheels" "$G/model" "$G/fixtures" "$G/vad"
for u in $PYTHON $VAD; do # pinned in whisper-job.sh / whisper_core.py and checked on the node
    f=$CACHE/cache/$(python3 -I -c 'import sys, urllib.parse; print(urllib.parse.unquote(sys.argv[1].rsplit("/", 1)[1]))' "$u")
    if [[ ! -s $f ]]; then curl -fsSL --limit-rate 20M "$u" -o "$f.part" && mv "$f.part" "$f"; fi
done
ln "$CACHE/cache/cpython-3.12.15+20261003-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz" "$G/"
ln "$CACHE/cache/silero_vad.jit" "$G/vad/silero_vad.jit"
uv export --project "$REPO/tools/speech/whisper" --frozen --no-header --no-emit-project > "$KIT/requirements-all.txt"
python3 -I - "$REPO/tools/speech/whisper/uv.lock" "$KIT/requirements-all.txt" "$G" "$CACHE/cache/wheels-aarch64" <<'PY'
import os, sys, tomllib, urllib.parse, urllib.request
lock, reqs, kit, cache = sys.argv[1:]
pins = {}
for line in open(reqs).read().replace("\\\n", "").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    req, _, marker = line.partition(";")
    if "aarch64" in marker.split("--hash")[0]:
        name, ver = req.strip().split("==")
        pins[name.lower()] = ver
def fits(fn):
    py, abi, plat = fn[:-4].split("-")[-3:]
    if "musllinux" in plat or not ("aarch64" in plat or plat == "any"):
        return False
    return (py == "cp312" or py.startswith("py3") or py == "py2.py3" or (abi == "abi3" and py.startswith("cp3") and int(py[3:]) <= 12)) \
        and abi in ("cp312", "abi3", "none")
lines, fetched = [], 0
for p in tomllib.load(open(lock, "rb"))["package"]:
    n = p["name"].lower()
    if pins.get(n) != p["version"]:
        continue
    w = next(w for w in p.get("wheels", []) if fits(urllib.parse.unquote(w["url"].rsplit("/", 1)[1])))
    fn = urllib.parse.unquote(w["url"].rsplit("/", 1)[1])
    cached = os.path.join(cache, fn)
    if not os.path.exists(cached) or (w.get("size") and os.path.getsize(cached) != w["size"]):
        req = urllib.request.Request(w["url"], headers={"User-Agent": "pip/25 (localrouter whisper-kit)"})  # the PyTorch CDN refuses urllib's default
        with urllib.request.urlopen(req, timeout=600) as r, open(cached + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(cached + ".part", cached)
        fetched += 1
    os.link(cached, os.path.join(kit, "wheels", fn))
    lines.append(f"{p['name']}=={p['version']} --hash={w['hash']}")
missing = set(pins) - {l.split("==")[0].lower() for l in lines}
if missing:
    sys.exit(f"no aarch64 wheel for: {sorted(missing)}")
open(os.path.join(kit, "requirements.txt"), "w").write("\n".join(sorted(lines)) + "\n")
print(f"   {len(lines)} wheels ({fetched} downloaded)")
PY
rm "$KIT/requirements-all.txt"
cp -al "$MODEL" "$G/model/$REV"
cp -l "$FIXTURES"/*.wav "$FIXTURES"/*.mp3 "$FIXTURES"/*.ogg "$FIXTURES"/*.flac "$FIXTURES"/*.png "$FIXTURES"/refs.json "$FIXTURES"/manifest.json "$G/fixtures/"

echo "== the CPU job's tar (no digest here: the owner's tooling computes it on the Spark side)"
tar -C "$C" -cf "$KIT/checks.tar" .
python3 -I - "$KIT" "$COMMIT" "$TREE" <<'PY'
import json, os, sys
kit, commit, tree = sys.argv[1:]
gib = lambda d: round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 2**30, 2)
json.dump({"id": f"doug-whisper-checks-{tree[:8]}", "node": "dgx02", "input_tar": os.path.join(kit, "checks.tar"),
           "input_sha256": "computed by the owner's tooling at staging", "cmd": "LANE_CPUS=8 STEPS='setup checks' bash run.sh", "minutes": 90, "mem_gib": 32,
           "disk_gib": 20, "cpus": 8, "gpu": False}, open(os.path.join(kit, "checks.json"), "w"), indent=1)
json.dump({"id": f"doug-whisper-gpu-{tree[:8]}", "node": "dgx02", "input_dir": os.path.join(kit, "gpu", "in"),
           "input_gib": gib(os.path.join(kit, "gpu", "in")), "input_sha256": "computed by the owner at staging (per-file pins are checked on the node)",
           "cmd": "LANE_CPUS=8 STEPS='setup verify build venv loadcheck twin bench serve' bash run.sh", "minutes": 150, "mem_gib": 48,
           "disk_gib": 20, "cpus": 8, "gpu": True}, open(os.path.join(kit, "gpu.json"), "w"), indent=1)
PY
echo "kit: $KIT; candidate commit $COMMIT (tree $TREE) on HEAD $(git -C "$REPO" rev-parse --short HEAD)"
ls -la "$KIT"; cat "$KIT/checks.json"
