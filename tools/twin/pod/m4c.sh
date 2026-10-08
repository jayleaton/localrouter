#!/usr/bin/env bash
# M4c, after m4d on the same pod: the image tool through the daemon. `localrouter serve` runs with a qwen-image tool per precision, and one image per precision is asked for over the
# OpenAI-style API (the tool loads, generates and unloads in its own worker process). Uses m4b's packs and m4d's binary.
set -uo pipefail
source "$(dirname "$0")/env.sh"
R=$STK/results/m4c-$(date -u +%Y%m%d-%H%M%S); mkdir -p $R; ln -sfn $R $STK/results/m4c-latest
run() { local name=$1; shift; echo "[m4c $(date +%T)] $name"; local t0=$(date +%s)
    "$@" > $R/$name.log 2>&1; local rc=$?
    tail -n 1 $R/$name.log | grep '^{' >> $R/summary.jsonl
    echo "{\"step\": \"$name\", \"rc\": $rc, \"s\": $(( $(date +%s) - t0 ))}" >> $R/steps.jsonl; return $rc; }
LP=/tmp/localrouter/packs; S=$(readlink -f $STK/results/m4d-latest)/out/bin/stk   # m4d's binary; m4d exported the Triton cubins
# the packs as the engine expects them: <weights>/<precision>, te, vae
mkdir -p /tmp/localrouter/weights && for d in nvfp4 fp8s te vae; do ln -sfn $LP/$d /tmp/localrouter/weights/$d; done
cat > $R/tools.json <<EOF
{"reserve_bytes": 1073741824, "tools": [
 {"id": "qwen-image-2.1", "kind": "image", "engine": "qwen_image", "weights": "/tmp/localrouter/weights", "idle_ttl_s": 0,
  "options": {"precision": "nvfp4", "max_side": 1024, "resident_mb": 24000}},
 {"id": "qwen-image-2.1-fp8", "kind": "image", "engine": "qwen_image", "weights": "/tmp/localrouter/weights", "idle_ttl_s": 0,
  "options": {"precision": "fp8s", "max_side": 1024, "resident_mb": 26000}}]}
EOF
$S serve --config $R/tools.json --port 8190 --data $R/data > $R/serve.log 2>&1 &
SERVE=$!
for i in $(seq 1 30); do $S check health http://127.0.0.1:8190 > /dev/null 2>&1 && break; sleep 1; done
for m in qwen-image-2.1 qwen-image-2.1-fp8; do
    python - "$m" "$R" "$TWIN" <<'PY'
import json, sys
m, r, twin = sys.argv[1:]
prompt = json.load(open(f"{twin}/prompts/gate.json"))[1][0]
json.dump({"model": m, "prompt": prompt, "size": "1024x1024", "seed": 7, "steps": 25, "response_format": "b64_json"},
          open(f"{r}/{m}.req.json", "w"))
PY
    run api-$m bash -c "curl -sS -m 900 http://127.0.0.1:8190/v1/images/generations -H 'content-type: application/json' -d @$R/$m.req.json > $R/$m.json; \
        python -c \"import json,base64; d=json.load(open('$R/$m.json')); b=base64.b64decode(d['data'][0]['b64_json']); open('$R/$m.png','wb').write(b); print(json.dumps({'model': '$m', 'png_bytes': len(b)}))\""
done
curl -sS http://127.0.0.1:8190/v1/models > $R/models.json
kill $SERVE; wait $SERVE 2>/dev/null
run attnbench env STK_ATTN=stk python -m stk_twin.attnbench   # the attention speed work's first measurement
echo "[m4c $(date +%T)] done: $R"
