#!/usr/bin/env bash
# Installs one model as an on-demand LocalRouter service on a GPU node, from files already on the node (no build, no
# download):
#   bash tools/package/install.sh --profile tools/package/profiles/<model>.json --weights DIR --image FILE.tar.zst \
#        [--manifest FILE] [--quick-assets] [--start] [--smoke]
#   bash tools/package/install.sh --uninstall
# 1. checks the machine: GPU and driver, Docker with the NVIDIA runtime, disk for the image;
# 2. checks the weights against the model's pinned manifest (every file's size and SHA-256; --quick-assets: sizes only),
#    so a partial or wrong download is refused before anything loads;
# 3. checks the image file against its build receipt (FILE.json from build-image.sh), then `docker load`s it;
# 4. writes $LR_HOME/tools.json with this one model: keep_loaded false, idle_ttl_s from the profile (0: unloaded right
#    after each job), and $LR_HOME/env for localrouterctl (docker mode, the image by id, the weights, the port);
# 5. with --start, starts the daemon (it holds no GPU memory until a request); with --smoke, also runs
#    lifecycle_smoke.py and leaves its receipt in $LR_HOME/smoke-receipt.json, then stops the daemon.
# Nothing starts at boot and nothing restarts on its own: localrouterctl start / stop is the service.
# A profile field that is null (not decided yet, e.g. a precision mode a GPU qualification picks) must be given in the
# environment: TOOL_ID, TOOL_ENGINE, TOOL_OPTIONS (JSON, merged over the profile's options), e.g.
#   TOOL_OPTIONS='{"fp32_linear": "...", "clip_linear": "..."}'
# An option the profile lists in "file_options" names a file on this machine: it is checked (size, when the profile
# gives one), copied to $LR_HOME/files, given to the daemon at /etc/localrouter/files/<option> (read-only) and its
# SHA-256 recorded in $LR_HOME/profile.json ("installed_files").
# Env: LR_HOME (default ~/.localrouter), LR_HOST (127.0.0.1), LR_PORT (8190).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export LR_HOME=${LR_HOME:-$HOME/.localrouter}
LR_HOST=${LR_HOST:-127.0.0.1}
LR_PORT=${LR_PORT:-8190}
CTL=$HERE/localrouterctl
step() { echo "[install $(date +%T)] $*"; }
die() { echo "install: $*" >&2; exit 1; }

profile="" weights="" image="" manifest="" quick=0 start=0 smoke=0 uninstall=0
while (($#)); do
    case $1 in
        --profile) profile=$2; shift ;; --weights) weights=$2; shift ;; --image) image=$2; shift ;;
        --manifest) manifest=$2; shift ;; --quick-assets) quick=1 ;; --start) start=1 ;; --smoke) start=1 smoke=1 ;;
        --uninstall) uninstall=1 ;; -h | --help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
    shift
done

if ((uninstall)); then
    # the image and this state directory go; the weights stay where they are
    "$CTL" stop || true
    img=$(sed -n 's/^LR_IMAGE=//p' "$LR_HOME/env" 2> /dev/null || true)
    [[ -z $img ]] || docker rmi "$img" > /dev/null 2>&1 || true
    rm -rf "$LR_HOME/tools.json" "$LR_HOME/env" "$LR_HOME/profile.json" "$LR_HOME/data" "$LR_HOME/files" "$LR_HOME/daemon.log" \
        "$LR_HOME/assets-receipt.json" "$LR_HOME/smoke-receipt.json"
    step "uninstalled (weights untouched)"
    exit 0
fi
[[ -f $profile ]] || die "--profile FILE is required"
[[ -d $weights ]] || die "--weights DIR is required (the model's weight directory)"
[[ -f $image && -f ${image%.tar.zst}.json ]] || die "--image FILE.tar.zst is required, with its receipt ${image%.tar.zst}.json beside it"
weights=$(cd "$weights" && pwd -P)

# ---- the profile, with the fields its engine owner has not confirmed supplied from the environment
mkdir -p "$LR_HOME"
python3 -I - "$profile" "$LR_HOME/profile.json" <<'PY' || exit 1
import json, os, sys
p = json.load(open(sys.argv[1]))
t = p["tool"]
for key, env in (("id", "TOOL_ID"), ("engine", "TOOL_ENGINE")):
    if os.environ.get(env):
        t[key] = os.environ[env]
if os.environ.get("TOOL_OPTIONS"):  # merged over the profile's options
    t["options"] = {**(t.get("options") or {}), **json.loads(os.environ["TOOL_OPTIONS"])}
missing = [k for k in ("id", "engine") if t.get(k) is None] + ["options." + k for k, v in (t.get("options") or {}).items() if v is None]
if missing:
    sys.exit("install: the profile leaves %s unset (the engine's owner or a qualification decides them): set them with "
             "TOOL_ID, TOOL_ENGINE or TOOL_OPTIONS (JSON, merged over the profile's options)" % ", ".join(missing))
import hashlib, shutil
files = os.path.join(os.environ["LR_HOME"], "files")
shutil.rmtree(files, ignore_errors=True)
p["installed_files"] = {}
for key, want in (p.get("file_options") or {}).items():  # options naming a file: copied in, mounted read-only
    src = t["options"][key]
    if not os.path.isfile(src):
        sys.exit("install: options.%s names %s, which is not a file" % (key, src))
    if want.get("bytes") is not None and os.path.getsize(src) != want["bytes"]:
        sys.exit("install: options.%s (%s) is %d bytes, not %d" % (key, src, os.path.getsize(src), want["bytes"]))
    os.makedirs(files, exist_ok=True)
    shutil.copyfile(src, os.path.join(files, key))
    p["installed_files"][key] = {"from": os.path.realpath(src), "sha256": hashlib.sha256(open(src, "rb").read()).hexdigest()}
    t["options"][key] = "/etc/localrouter/files/" + key
p["model"] = t["id"]
if t.get("keep_loaded") or "idle_ttl_s" not in t:
    sys.exit("install: an on-demand profile sets keep_loaded false and an explicit idle_ttl_s")
p["idle_ttl_s"] = t["idle_ttl_s"]
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
manifest=${manifest:-$(python3 -I -c 'import json,sys; m=json.load(open(sys.argv[1])).get("manifest") or ""; print(m)' "$LR_HOME/profile.json")}
[[ -z $manifest || $manifest == /* ]] || manifest=$REPO/$manifest
[[ -f $manifest ]] || die "no asset manifest (--manifest FILE, or the profile's \"manifest\"): $manifest"

# ---- 1. the machine
step "checks"
command -v nvidia-smi > /dev/null || die "no nvidia-smi: this needs an NVIDIA GPU and its driver"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  gpu: /'
command -v docker > /dev/null || die "no docker"
docker info --format '{{json .Runtimes}}' | grep -q nvidia || die "Docker has no nvidia runtime (install the NVIDIA Container Toolkit)"
command -v zstd > /dev/null || die "no zstd"
root=$(docker info --format '{{.DockerRootDir}}')
need_gb=$(($(stat -c %s "$image") * 3 / 1000000000 + 1))   # the unpacked layers, with room to spare
free_gb=$(df -BG --output=avail "$root" 2> /dev/null | tail -1 | tr -dc 0-9 || echo 0)
[[ -z $free_gb ]] || ((free_gb >= need_gb)) || die "$root has $free_gb GB free; loading the image needs about $need_gb GB"
others=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | grep -v '^$' || true)
[[ -z $others ]] || echo "  note: other GPU processes are running (they share the memory):"$'\n'"$others" | sed 's/^/  /'
echo "  MemAvailable: $(awk '/MemAvailable/ { printf "%.1f GiB", $2 / 1048576 }' /proc/meminfo)"

# ---- 2. the weights
step "weights: $weights against $manifest"
python3 -I "$HERE/verify_assets.py" "$manifest" "$weights" --receipt "$LR_HOME/assets-receipt.json" $( ((quick)) && echo --quick) |
    grep -v '^ok ' || true
[[ $(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"])' "$LR_HOME/assets-receipt.json") == pass ]] ||
    die "the weights do not match the manifest ($LR_HOME/assets-receipt.json)"

# ---- 3. the image
step "image: $image"
receipt=${image%.tar.zst}.json
read -r want_sha want_id dirty < <(python3 -I -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["file_sha256"], r["image_id"], r["dirty"])' "$receipt")
[[ $dirty == False ]] || echo "  note: the image was built from a checkout with uncommitted changes (a dev build)"
got_sha=$(sha256sum "$image" | cut -d' ' -f1)
[[ $got_sha == "$want_sha" ]] || die "the image file's SHA-256 is $got_sha, its receipt says $want_sha"
if [[ $(docker image inspect -f '{{.Id}}' "$want_id" 2> /dev/null) != "$want_id" ]]; then
    zstd -dc "$image" | docker load > /dev/null
fi
[[ $(docker image inspect -f '{{.Id}}' "$want_id" 2> /dev/null) == "$want_id" ]] || die "after docker load, image $want_id is not present"
echo "  $want_id"

# ---- 4. the configuration
step "config: $LR_HOME"
python3 -I - "$LR_HOME/profile.json" "$LR_HOME/tools.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
tool = {"kind": "video", "weights": "/models", "priority": 0, **p["tool"]}
cfg = {"reserve_bytes": p.get("reserve_bytes", 8 << 30), "keep_outputs_s": p.get("keep_outputs_s", 86400), "tools": [tool],
       "defaults": {c: tool["id"] for c in p.get("default_for", [])}}
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY
mkdir -p "$LR_HOME/data"
printf '%s\n' "LR_MODE=docker" "LR_IMAGE=$want_id" "LR_NAME=localrouter" "LR_MODELS=$weights" "LR_CONFIG=$LR_HOME/tools.json" \
    "LR_HOST=$LR_HOST" "LR_PORT=$LR_PORT" > "$LR_HOME/env.new"
mv "$LR_HOME/env.new" "$LR_HOME/env"
model=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$LR_HOME/profile.json")

# ---- 5. start, check
if ((smoke)); then
    step "lifecycle smoke (starts and stops the daemon)"
    rc=0
    python3 -I "$HERE/lifecycle_smoke.py" --profile "$LR_HOME/profile.json" --out "$LR_HOME/smoke-receipt.json" || rc=$?
    ((rc == 0)) || die "the lifecycle smoke did not pass (exit $rc): $LR_HOME/smoke-receipt.json"
elif ((start)); then
    "$CTL" start
fi
cat << EOF

Installed $model (on demand: it loads on a request and unloads after it).
  start / stop:  LR_HOME=$LR_HOME $CTL start | stop
  generate:      LR_HOME=$LR_HOME $CTL generate $model "a prompt" -o out.mp4
  status, logs:  LR_HOME=$LR_HOME $CTL status | logs
  uninstall:     LR_HOME=$LR_HOME bash $0 --uninstall
EOF
