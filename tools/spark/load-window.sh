#!/usr/bin/env bash
# The streaming loader on GB10: cold load before and after, in one short window on a Spark that already ran
# install.sh (its packs and tools.json in $LOCALROUTER_HOME, its image as localrouter:dev). Run from a checkout of the
# branch under test, on the Spark, only in a window the owner granted:
#   bash tools/spark/load-window.sh
# If another service owns the machine, WINDOW_OPEN / WINDOW_CLOSE are its commands to stop and restore it (they run
# first and, whatever happens, last). The image already there is measured as "before"; this checkout is built and
# measured as "after" (it is built and checked first, outside the window). Each: three cold loads (models released first), a warm image, an FP8 cold load; the PNGs of
# both must be byte-equal (the loader moves bytes, it does not change them). About 15 minutes.
# PREFLIGHT_ONLY=1 stops before the window; BUILD_IN_WINDOW=1 builds inside it. Output: $LOCALROUTER_HOME/results/load-<time>/ (summary.json first; the workers' load reports in load-*.log).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOCALROUTER_HOME=${LOCALROUTER_HOME:-$HOME/localrouter}
R=$LOCALROUTER_HOME/results/load-$(date -u +%Y%m%d-%H%M%S); mkdir -p "$R/data-before" "$R/data-after"; chmod 777 "$R"/data-*
PORT=8191
step() { echo "[load $(date +%T)] $*" | tee -a "$R/window.log"; }
opened=0
close() { docker rm -f localrouter-load localrouter-pre > /dev/null 2>&1 || true; [[ $opened == 0 || -z ${WINDOW_CLOSE:-} ]] || { step "close: $WINDOW_CLOSE"; bash -c "$WINDOW_CLOSE" >> "$R/window.log" 2>&1; }; }
trap close EXIT

# Build, then serve the CPU test tool from the new image as the container's PID 1 and make one image, so a broken
# image is never measured. Before the window by default; BUILD_IN_WINDOW=1 builds inside it (a co-tenant with
# little free memory, as a serving LLM, should not share the machine with a build).
prepare() {
    docker image inspect localrouter:before > /dev/null 2>&1 || docker tag localrouter:dev localrouter:before
    step "build after (this checkout)"
    docker buildx build -f "$REPO/docker/Dockerfile" -t localrouter:after --load "$REPO" > "$R/build.log" 2>&1
    echo '{"tools": [{"id": "tp", "kind": "image", "engine": "testpattern"}]}' > "$R/pre.json"
    docker run -d --name localrouter-pre -p 127.0.0.1:$PORT:8190 -v "$R/pre.json:/etc/localrouter/tools.json:ro" localrouter:after \
        serve --config /etc/localrouter/tools.json --data /tmp/d > /dev/null
    for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 1; done
    curl -sf -m 60 "http://127.0.0.1:$PORT/v1/images/generations" -H 'content-type: application/json' \
        -d '{"model": "tp", "prompt": "x", "size": "256x256"}' > /dev/null || { step "preflight failed: the new image does not serve"; docker logs localrouter-pre >> "$R/window.log" 2>&1; exit 1; }
    docker rm -f localrouter-pre > /dev/null
    step "preflight ok"
}
[[ ${BUILD_IN_WINDOW:-0} == 1 ]] || prepare
[[ ${PREFLIGHT_ONLY:-0} == 1 ]] && exit 0   # build and check now; the window later
[[ -z ${WINDOW_OPEN:-} ]] || { step "open: $WINDOW_OPEN"; opened=1; bash -c "$WINDOW_OPEN" >> "$R/window.log" 2>&1; }
docker compose -p localrouter down > /dev/null 2>&1 || true   # the installed service, if it runs
[[ ${BUILD_IN_WINDOW:-0} != 1 ]] || prepare

bench() { # IMAGE: serve it on $PORT and measure through the HTTP API
    local tag=$1
    docker rm -f localrouter-load > /dev/null 2>&1 || true
    sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    docker run -d --name localrouter-load --gpus all -p 127.0.0.1:$PORT:8190 -v "$LOCALROUTER_HOME/weights:/models:ro" \
        -v "$LOCALROUTER_HOME/tools.json:/etc/localrouter/tools.json:ro" -v "$R/data-$tag:/data" "localrouter:$tag" \
        serve --config /etc/localrouter/tools.json --data /data > /dev/null
    for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 1; done
    step "measure $tag"
    python3 - "$PORT" "$tag" "$R" <<'PY' | tee -a "$R/window.log"
import base64, hashlib, json, sys, time, urllib.error, urllib.request
port, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
def post(path, body=None):
    req = urllib.request.Request(base + path, json.dumps(body or {}).encode(), {"content-type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        sys.exit(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")
    return time.time() - t0, r
def image(model):
    dt, r = post("/v1/images/generations", {"model": model, "prompt": "a red fox in fresh snow at dawn, telephoto", "size": "1024x1024", "seed": 7})
    return dt, hashlib.sha256(base64.b64decode(r["data"][0]["b64_json"])).hexdigest()
res = {"image": tag, "cold": [], "warm": None, "fp8_cold": None, "png": {}}
for _ in range(3):
    post("/v1/tools/release")
    dt, res["png"]["nvfp4"] = image("qwen-image-2.1")
    res["cold"].append(round(dt, 2))
res["warm"] = round(image("qwen-image-2.1")[0], 2)
post("/v1/tools/release")
dt, res["png"]["fp8"] = image("qwen-image-2.1-fp8")
res["fp8_cold"] = round(dt, 2)
post("/v1/tools/release")
json.dump(res, open(f"{out}/{tag}.json", "w"))
print(json.dumps(res))
PY
    cat "$R/data-$tag"/logs/*.log > "$R/load-$tag.log" 2>/dev/null || true
    docker rm -f localrouter-load > /dev/null
}
bench before
bench after
python3 - "$R" <<'PY' | tee "$R/summary.json"
import json, sys
b, a = (json.load(open(f"{sys.argv[1]}/{t}.json")) for t in ("before", "after"))
print(json.dumps({"before": b, "after": a, "png_equal": b["png"] == a["png"],
                  "cold_speedup": round(min(b["cold"]) / min(a["cold"]), 2)}))
PY
step "done: $R"
