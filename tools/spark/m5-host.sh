#!/usr/bin/env bash
# M5's host half on the DGX Spark, after tools/spark/m5.sh made the packs: LocalRouter's Docker image built natively
# (arm64), its self-test, then the image tool through the API from a cold page cache (cold load + first image) and
# warm (second image), with the container's peak memory. Run as the user who owns docker; `sudo` is used once for
# dropping the page cache. Output: $STK_HOST/results/m5-host-<time>/.
set -uo pipefail
STK_HOST=${STK_HOST:-/data/localrouter}
REPO=$STK_HOST/repo
R=$STK_HOST/results/m5-host-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R
run() { local name=$1; shift; echo "[m5-host $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
run image docker buildx build -f $REPO/docker/Dockerfile --platform linux/arm64 -t localrouter:m5 --load $REPO
run selftest docker run --rm --gpus all --entrypoint localrouter localrouter:m5 check selftest
mkdir -p $R/data && chmod 777 $R/data   # the image runs as uid 10001
# GB10's memory is the system's (docker stats misses the GPU's): sample MemAvailable every second for the peak
( while :; do echo "$(date +%s) $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; sleep 1; done ) > $R/memavail.log &
MEM=$!
trap 'kill $MEM 2>/dev/null' EXIT
cat > $R/tools.json <<EOF
{"reserve_bytes": 8589934592, "tools": [
 {"id": "qwen-image-2.1", "kind": "image", "engine": "qwen_image", "weights": "/models", "idle_ttl_s": 600,
  "options": {"precision": "nvfp4", "max_side": 1664}},
 {"id": "qwen-image-2.1-fp8", "kind": "image", "engine": "qwen_image", "weights": "/models", "idle_ttl_s": 0,
  "options": {"precision": "fp8s", "max_side": 1664}}]}
EOF
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
docker run -d --name localrouter-m5 --gpus all -p 127.0.0.1:8190:8190 -v $STK_HOST/weights:/models:ro -v $R:/cfg:ro \
    -v $R/data:/data localrouter:m5 serve --config /cfg/tools.json --data /data > $R/container.id
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8190/health > /dev/null && break; sleep 1; done
( while docker inspect localrouter-m5 > /dev/null 2>&1; do docker stats --no-stream --format '{{.MemUsage}}' localrouter-m5; sleep 2; done ) > $R/docker-mem.log 2>&1 &
req() { # MODEL SIZE NAME
    python3 -c "import json,sys; print(json.dumps({'model': sys.argv[1], 'prompt': json.load(open(sys.argv[3]))[0][0], 'size': sys.argv[2], 'seed': 7, 'steps': 25}))" \
        "$1" "$2" $REPO/tools/twin/prompts/gate.json > $R/$3.req.json
    local t0=$(date +%s.%N)
    curl -sS -m 1800 http://127.0.0.1:8190/v1/images/generations -H 'content-type: application/json' -d @$R/$3.req.json > $R/$3.json
    echo "{\"request\": \"$3\", \"wall_s\": $(echo "$(date +%s.%N) - $t0" | bc)}" | tee -a $R/summary.jsonl
}
run api-cold-1024 req qwen-image-2.1 1024x1024 cold-1024
run api-warm-1024 req qwen-image-2.1 1024x1024 warm-1024
run api-warm-576 req qwen-image-2.1 576x576 warm-576
run api-fp8-1024 req qwen-image-2.1-fp8 1024x1024 fp8-1024
docker logs localrouter-m5 > $R/serve.log 2>&1
docker rm -f localrouter-m5 > /dev/null
echo "[m5-host $(date +%T)] done: $R"
