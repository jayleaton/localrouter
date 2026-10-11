#!/usr/bin/env bash
# Builds the arm64 LocalRouter image from a clean, committed checkout on a build machine, and exports it as one
# compressed file a GPU node loads without building anything:
#   bash tools/package/build-image.sh [OUT_DIR]          (default ~/localrouter-dist)
# Writes OUT_DIR/localrouter-<commit>-arm64.tar.zst and OUT_DIR/localrouter-<commit>-arm64.json: the source commit,
# the image id, the file's SHA-256 and size, the build arguments and the Dockerfile's SHA-256. install scripts check
# the file against that receipt before `docker load`.
# It refuses to run on an arm64 host (a GPU node such as a DGX Spark builds nothing: its memory belongs to the models)
# unless LR_BUILD_ON_ARM=1, and refuses a checkout with uncommitted changes (the receipt names a commit).
# Env: BUILD_ARGS ("KEY=VALUE ..." passed as --build-arg, e.g. an engine selection the Dockerfile takes),
#      DOCKERFILE (default docker/Dockerfile), LR_ALLOW_DIRTY=1 (a dev build; the receipt records "dirty": true),
#      SRC (the checkout to build; default this one).
# The build runs in a builder of its own (buildx docker-container driver) whose container is capped: LR_BUILDER_MEMORY
# (default 7g, no swap beyond it) and LR_BUILDER_CPUS (a cpuset, default 0-3). A build that needs more is OOM-killed
# and fails; it cannot push the machine into swap. The receipt records the cap, the builder's actual memory.peak and
# its OOM kills, and the builder is removed afterwards. Such a builder cannot see the Docker daemon's images, so each
# entry IMAGE[=PLATFORM] of LR_BASE_IMAGES (already pulled; PLATFORM defaults to this machine's, the one a build stage
# on $BUILDPLATFORM uses, e.g. "nvcr.io/nvidia/pytorch:26.07-py3") is exported once as an OCI layout of that platform
# under LR_OCI_CACHE (default ~/.cache/localrouter-oci, about the image's size on disk) and given to the build in place
# of a pull. Those layouts are removed once the image file is written (LR_OCI_KEEP=1 keeps them), and kept after a
# failure for the retry.
# Disk: before exporting or building, it checks the free space of the OCI cache's and Docker's file systems against what
# the run can take (each base image not yet exported twice over, its layout and the builder's own copy, plus
# LR_DISK_MARGIN_GB, default 30) and stops if that does not fit. The receipt records each layout's measured size and the
# largest drop in Docker's file system's free space during the run, sampled every 2 s: an upper bound on the run's own
# use, since it also counts anything else writing to that file system meanwhile.
set -euo pipefail
REPO=$(cd "${SRC:-$(dirname "$0")/../..}" && pwd)
OUT=${1:-$HOME/localrouter-dist}
DOCKERFILE=${DOCKERFILE:-$REPO/docker/Dockerfile}
die() { echo "build-image: $*" >&2; exit 1; }

[[ $(uname -m) != aarch64 || ${LR_BUILD_ON_ARM:-0} == 1 ]] || die "this is an arm64 host; build on the build machine and copy the file (LR_BUILD_ON_ARM=1 overrides)"
command -v docker > /dev/null || die "no docker"
docker buildx version > /dev/null 2>&1 || die "no docker buildx"
command -v zstd > /dev/null || die "no zstd"
cd "$REPO"
commit=$(git rev-parse HEAD)
dirty=false
if [[ -n $(git status --porcelain) ]]; then
    [[ ${LR_ALLOW_DIRTY:-0} == 1 ]] || die "uncommitted changes; commit them (or LR_ALLOW_DIRTY=1 for a dev build)"
    dirty=true
fi
tag=localrouter:${commit:0:12}
args=()
for kv in ${BUILD_ARGS:-}; do args+=(--build-arg "$kv"); done
mkdir -p "$OUT"
base=$OUT/localrouter-${commit:0:12}-arm64
MEM=${LR_BUILDER_MEMORY:-7g}
CPUS=${LR_BUILDER_CPUS:-0-3}
OCI=${LR_OCI_CACHE:-$HOME/.cache/localrouter-oci}
host_platform=linux/$(docker version -f '{{.Server.Arch}}')
avail() { df -B1 --output=avail "$1" | tail -1 | tr -d ' '; }
mkdir -p "$OCI"
droot=$(docker info -f '{{.DockerRootDir}}')
need=$(( ${LR_DISK_MARGIN_GB:-30} * 1000000000 ))
for entry in ${LR_BASE_IMAGES:-}; do
    image=${entry%%=*}
    size=$(docker image inspect -f '{{.Size}}' "$image" 2> /dev/null) || die "base image $image is not in the local store (docker pull it first)"
    need=$(( need + 2 * size ))
done
for fs in "$OCI" "$droot"; do
    free=$(avail "$fs" 2> /dev/null) || continue
    (( free >= need )) || die "$fs has $((free / 1000000000)) GB free; this build can take about $((need / 1000000000)) GB (base image layouts and the builder's copy, plus LR_DISK_MARGIN_GB): not starting"
done
echo "[build-image] disk: $(( $(avail "$OCI") / 1000000000 )) GB free, the run can take about $((need / 1000000000)) GB"
start_free=$(avail "$droot")
builder="" sampler=""
stop_sampler() { [[ -z $sampler ]] || { kill "$sampler" 2> /dev/null; wait "$sampler" 2> /dev/null; sampler=""; } || true; }
cleanup() {   # every exit, from here on: the sampler is stopped before its file goes, and the builder is removed
    stop_sampler
    rm -f "$OUT/.disk-min.$$"
    [[ -z $builder ]] || docker buildx rm "$builder" > /dev/null 2>&1 || true
    echo "[build-image] cleanup: disk sampler stopped${builder:+, builder $builder $(docker buildx inspect "$builder" > /dev/null 2>&1 && echo "NOT removed" || echo removed)}" >&2
}
trap cleanup EXIT
# it also ends by itself within 2 s if this script is killed outright
(min=$start_free; while kill -0 $$ 2> /dev/null; do f=$(avail "$droot"); ((f < min)) && min=$f; echo "$min" > "$OUT/.disk-min.$$"; sleep 2; done) &
sampler=$!
dirs=()
for entry in ${LR_BASE_IMAGES:-}; do
    image=${entry%%=*} platform=$host_platform
    [[ $entry != *=* ]] || platform=${entry#*=}
    iid=$(docker image inspect -f '{{.Id}}' "$image" 2> /dev/null) || die "base image $image is not in the local store (docker pull it first)"
    dir=$OCI/${iid#sha256:}-${platform//\//-}
    if [[ ! -f $dir/.complete ]]; then
        echo "[build-image] exporting $image ($platform) as an OCI layout (once): $dir"
        rm -rf "$dir" && mkdir -p "$dir"
        docker save --platform "$platform" "$image" | tar -x -C "$dir" || die "could not export $image ($platform): is that platform pulled?"
        touch "$dir/.complete"
    fi
    dirs+=("$dir")
    # pinned to the exported manifest: exact, and needed where the layout carries no tag annotation
    # (attestation manifests, which name no platform, are skipped)
    digest=$(python3 -I -c 'import json,sys
os_, arch = sys.argv[2].split("/")[:2]
m = [x for x in json.load(open(sys.argv[1]))["manifests"] if x.get("platform", {}).get("os") == os_ and x["platform"].get("architecture") == arch]
assert len(m) == 1, m
print(m[0]["digest"])' "$dir/index.json" "$platform") || die "the OCI layout of $image has no single $platform manifest: $dir/index.json"
    args+=(--build-context "$image=oci-layout://$dir@$digest")
done
builder=lr-build-$$
docker buildx create --name "$builder" --driver docker-container \
    --driver-opt "memory=$MEM,memory-swap=$MEM,cpuset-cpus=$CPUS" > /dev/null
docker buildx inspect --bootstrap "$builder" > /dev/null
cid=$(docker ps -q --no-trunc -f "name=buildx_buildkit_${builder}0")
cg=$(find /sys/fs/cgroup -maxdepth 4 -type d -name "*$cid*" 2> /dev/null | head -1)
echo "[build-image] $tag from $commit${BUILD_ARGS:+ ($BUILD_ARGS)}; builder capped at $MEM, cpus $CPUS"
rc=0
docker buildx build --builder "$builder" -f "$DOCKERFILE" --platform linux/arm64 "${args[@]}" -t "$tag" --load "$REPO" || rc=$?
peak=$(cat "$cg/memory.peak" 2> /dev/null || echo null)
ooms=$(sed -n 's/^oom_kill //p' "$cg/memory.events" 2> /dev/null || echo null)
echo "[build-image] builder memory.peak $peak bytes, oom kills $ooms"
((rc == 0)) || die "the build failed (exit $rc)$([[ $ooms =~ ^[1-9] ]] && echo ": the builder ran out of its $MEM")"
id=$(docker image inspect -f '{{.Id}}' "$tag")
echo "[build-image] export $id"
docker save "$tag" | zstd -T0 -10 -q -o "$base.tar.zst.part"
mv "$base.tar.zst.part" "$base.tar.zst"
sha=$(sha256sum "$base.tar.zst" | cut -d' ' -f1)
stop_sampler
disk_peak=$(( start_free - $(cat "$OUT/.disk-min.$$" 2> /dev/null || echo "$start_free") ))
oci_sizes=""
for d in "${dirs[@]}"; do oci_sizes+="$(basename "$d")=$(du -sb "$d" | cut -f1) "; done
python3 -I - "$base.json" "$commit" "$dirty" "$tag" "$id" "$base.tar.zst" "$sha" "$(sha256sum "$DOCKERFILE" | cut -d' ' -f1)" \
    "${BUILD_ARGS:-}" "$MEM" "$CPUS" "$peak" "$ooms" "${LR_BASE_IMAGES:-}" "$disk_peak" "$oci_sizes" <<'PY'
import json, os, sys
out, commit, dirty, tag, image_id, path, sha, dockerfile, build_args, mem, cpus, peak, ooms, bases, disk_peak, oci = sys.argv[1:]
num = lambda v: None if v == "null" else int(v)
json.dump({"commit": commit, "dirty": dirty == "true", "tag": tag, "image_id": image_id, "platform": "linux/arm64",
           "file": os.path.basename(path), "file_bytes": os.path.getsize(path), "file_sha256": sha,
           "dockerfile_sha256": dockerfile, "build_args": build_args.split(), "base_images_from_local_store": bases.split(),
           "builder": {"memory_max": mem, "cpuset": cpus, "memory_peak_bytes": num(peak), "oom_kills": num(ooms)},
           "disk": {"peak_free_space_drop_bytes": int(disk_peak), "free_space_drop_includes_other_writers": True, "oci_layout_bytes": {k: int(v) for k, v in (x.split("=") for x in oci.split())}}},
          open(out, "w"), indent=1)
PY
echo "[build-image] $base.tar.zst ($(du -h "$base.tar.zst" | cut -f1)), receipt $base.json; free space dropped by at most $((disk_peak / 1000000)) MB during the run (all writers)"
if [[ ${LR_OCI_KEEP:-0} != 1 ]]; then
    for d in "${dirs[@]}"; do rm -rf "$d"; done   # only the layouts this run used; nothing else in the cache or the daemon
fi
